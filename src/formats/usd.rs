//! USD (`.usd`, `.usda`, `.usdc`, `.usdz`), read with the pure-Rust `openusd` crate. The composed stage's visible
//! surfaces become one `meshes::Mesh`: references, payloads, variants, instances and point instancers resolved, each
//! surface placed by its world transform, the stage's up axis and units turned into glTF's (Y up, meters).
//!
//! Meshes keep their polygons' outlines (concave faces are ear-clipped, `holeIndices` faces left out) and their
//! normals, texture coordinates and display colors in any interpolation; with no normals a subdivision surface is
//! smoothed and a polygonal one faceted. The implicit surfaces (Cube, Sphere, Cylinder, Cone, Capsule, Plane) are
//! tessellated. Values are read at the stage's start time when it gives a time range, else at the earliest sample,
//! so an animated file shows its first frame.
//!
//! Materials come from `UsdPreviewSurface`: the diffuse color or texture, the normal map, opacity, emissive color,
//! metallic and roughness (with the shader's fallbacks), textures read from the `.usdz` itself or found in the pack.
//! A surface with no material shows its display color: one color as the material, more as vertex colors.
//!
//! A file is read with `openusd`'s own resolver. A model in memory is read with [`MemResolver`], which serves the
//! layers it names from [`Assets::read`].

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::io::{self, Read as _};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Context as _, Result};
use openusd::usd::SchemaBase as _;
use openusd::{ar, gf, sdf, tf, usd};
use openusd_schemas::geom::{
    self, CapsuleSchema, ConeSchema, CubeSchema, CylinderSchema, GprimSchema, ImageableExt, MeshSchema, PlaneSchema,
    PointBasedSchema, PointInstancerSchema, PrimvarsAPI, SphereSchema, SubsetSchema,
};
use openusd_schemas::shade::{self, MaterialBindingAPI};

use super::Input;
use crate::assets::{Assets, memory_key};
use crate::mesh::{GRAY, Image, MAX_TRIANGLES, Material, Mesh, Prim, triangulate};
use crate::num::f32_of;

/// The file's meshes, or `None` when it has none to show (a material library, a layer of overrides).
pub(crate) fn load(input: Input<'_>, assets: &dyn Assets) -> Result<Option<Mesh>> {
    let path = input.path;
    let (stage, paths) = match input.bytes() {
        Some(bytes) => open_in_memory(bytes, input, assets)?,
        #[cfg(feature = "fs")]
        None => {
            let stage = usd::Stage::builder()
                .schema_registry(openusd_schemas::schema_registry())
                .open(path.to_str().context("the path isn't UTF-8")?)
                .map_err(|e| invalid!("{e}"))?;
            let paths = prim_paths(&stage)?;
            (stage, paths)
        }
        #[cfg(not(feature = "fs"))]
        None => unreachable!("a model without its bytes needs the `fs` feature"),
    };
    let z_up = matches!(stage.stage_metadata("upAxis"), Ok(Some(v)) if v.as_str() == Some("Z"));
    let meters = match stage.stage_metadata("metersPerUnit") {
        Ok(Some(sdf::Value::Double(m))) if m > 0.0 => m,
        Ok(Some(sdf::Value::Float(m))) if m > 0.0 => f64::from(m),
        _ => 0.01,
    };
    // A value that isn't animated reads the same at any time; an animated one shows the first frame.
    let time = if stage.has_authored_time_code_range() {
        usd::TimeCode::new(stage.start_time_code())
    } else {
        usd::TimeCode::EARLIEST
    };
    let mut out = Builder {
        stage: &stage,
        file: path,
        assets,
        time,
        usdz: Usdz::open(input),
        mesh: Mesh::default(),
        materials: HashMap::new(),
        images: HashMap::new(),
        prims: HashMap::new(),
    };
    let mut xforms = geom::XformCache::new(time);
    let prims: Vec<usd::Prim> = paths.iter().filter_map(|p| stage.prim(p.clone()).ok()).collect();
    // A point instancer's prototypes show only where it places them.
    let instancers: Vec<&usd::Prim> = prims
        .iter()
        .filter(|p| kind(p).as_deref() == Some("PointInstancer"))
        .collect();
    for prim in &prims {
        if instancers
            .iter()
            .any(|i| within(prim.path(), i.path()) && prim.path() != i.path())
        {
            continue;
        }
        if !out.shown(prim) {
            continue;
        }
        if kind(prim).as_deref() == Some("PointInstancer") {
            out.instancer(prim, &prims, &mut xforms)?;
        } else if let Some(shape) = out.shape(prim) {
            let world = xforms.local_to_world_transform(prim).unwrap_or_default();
            out.emit(prim, &shape, world)?;
        }
    }
    let Builder { mut mesh, .. } = out;
    let scale = f32_of(meters);
    for prim in &mut mesh.prims {
        for v in &mut prim.positions {
            *v = up(*v, z_up).map(|c| c * scale);
        }
        for n in &mut prim.normals {
            *n = up(*n, z_up);
        }
    }
    Ok((!mesh.prims.is_empty()).then_some(mesh))
}

/// The paths of the stage's prims, composed (which reads the layers they reference).
fn prim_paths(stage: &usd::Stage) -> Result<Vec<sdf::Path>> {
    let mut paths = Vec::new();
    stage
        .traverse(usd::PrimPredicate::DEFAULT.with_instance_proxies(true), |p| {
            paths.push(p.clone());
        })
        .map_err(|e| invalid!("{e}"))?;
    Ok(paths)
}

/// Z-up to glTF's Y-up: (x, y, z) becomes (x, z, -y).
fn up([x, y, z]: [f32; 3], z_up: bool) -> [f32; 3] {
    if z_up { [x, z, -y] } else { [x, y, z] }
}

/// The prim's schema type name.
fn kind(prim: &usd::Prim) -> Option<String> {
    prim.type_name().ok().flatten().map(|t| t.as_str().to_owned())
}

/// Whether `path` is `root` or beneath it.
fn within(path: &sdf::Path, root: &sdf::Path) -> bool {
    let (p, r) = (path.as_str(), root.as_str());
    p == r || (p.starts_with(r) && p.as_bytes().get(r.len()) == Some(&b'/'))
}

struct Builder<'a> {
    stage: &'a usd::Stage,
    file: &'a Path,
    assets: &'a dyn Assets,
    time: usd::TimeCode,
    usdz: Option<Usdz<'a>>,
    mesh: Mesh,
    /// Material prim path (or `display:<rgba>` for a display color) to its index in `mesh.materials`.
    materials: HashMap<String, usize>,
    /// Texture asset path to its index in `mesh.images`, or `None` when it couldn't be found.
    images: HashMap<String, Option<usize>>,
    /// Material index to the prim in `mesh.prims` collecting its triangles.
    prims: HashMap<usize, usize>,
}

/// How a primvar's values spread over the mesh.
#[derive(Clone, Copy, PartialEq)]
enum Spread {
    Constant,
    Uniform,
    Vertex,
    FaceVarying,
}

impl Spread {
    fn of(token: Option<&str>) -> Self {
        match token {
            Some("constant") => Self::Constant,
            Some("uniform") => Self::Uniform,
            Some("faceVarying") => Self::FaceVarying,
            _ => Self::Vertex,
        }
    }

    /// The value index for one face corner: `face` is the face, `point` the point it uses, `corner` its place
    /// among all the face corners.
    fn index(self, face: usize, point: usize, corner: usize) -> usize {
        match self {
            Self::Constant => 0,
            Self::Uniform => face,
            Self::Vertex => point,
            Self::FaceVarying => corner,
        }
    }
}

/// A primvar's values with its indices and interpolation.
#[derive(Clone)]
struct Primvar<T> {
    values: Vec<T>,
    indices: Option<Vec<i32>>,
    spread: Spread,
}

impl<T: Copy> Primvar<T> {
    fn at(&self, face: usize, point: usize, corner: usize) -> Option<T> {
        let i = self.spread.index(face, point, corner);
        let i = match &self.indices {
            Some(ix) => usize::try_from(*ix.get(i)?).ok()?,
            None => i,
        };
        self.values.get(i).copied()
    }

    /// One value for the whole surface.
    fn constant(&self) -> bool {
        self.spread == Spread::Constant || (self.values.len() == 1 && self.indices.is_none())
    }
}

impl Builder<'_> {
    fn primvar<T>(
        &self,
        api: &PrimvarsAPI,
        name: &str,
        decode: impl Fn(sdf::Value) -> Option<Vec<T>>,
    ) -> Option<Primvar<T>> {
        let pv = api.primvar(name).ok()?;
        let values = decode(pv.attribute().get_at::<sdf::Value>(self.time).ok()??)?;
        let indices = pv.indices(self.time).ok().flatten().filter(|i| !i.is_empty());
        let spread = pv
            .interpolation()
            .ok()
            .map_or(Spread::Vertex, |i| Spread::of(Some(i.as_token())));
        Some(Primvar {
            values,
            indices,
            spread,
        })
    }

    fn value(&self, attr: &usd::Attribute) -> Option<sdf::Value> {
        attr.get_at::<sdf::Value>(self.time).ok().flatten()
    }

    /// A number attribute, double or float, or its fallback.
    fn number(&self, attr: &usd::Attribute, fallback: f64) -> f32 {
        f32_of(match self.value(attr) {
            Some(sdf::Value::Double(v)) => v,
            Some(sdf::Value::Float(v)) => f64::from(v),
            Some(sdf::Value::Half(v)) => f64::from(v.to_f32()),
            Some(sdf::Value::Int(v)) => f64::from(v),
            _ => fallback,
        })
    }

    fn token(&self, attr: &usd::Attribute) -> Option<String> {
        attr.get_at::<tf::Token>(self.time)
            .ok()
            .flatten()
            .map(|t| t.as_str().to_owned())
    }

    /// Visible, and drawn for rendering: guides and proxies stand in for the real thing.
    fn shown(&self, prim: &usd::Prim) -> bool {
        let img = geom::Imageable::from_prim_unchecked(prim.clone());
        let visible = !matches!(img.compute_visibility(self.time), Ok(geom::Visibility::Invisible));
        let purpose = img.compute_purpose().unwrap_or_default();
        visible && !matches!(purpose, geom::Purpose::Guide | geom::Purpose::Proxy)
    }

    /// The prim's surface as polygons in its own space: a mesh's, or an implicit surface's tessellation.
    fn shape(&self, prim: &usd::Prim) -> Option<Shape> {
        let t = |attr: &usd::Attribute| self.token(attr);
        match kind(prim)?.as_str() {
            "Mesh" => self.mesh_shape(&geom::Mesh::from_prim_unchecked(prim.clone())),
            "Cube" => {
                let s = self.number(&geom::Cube::from_prim_unchecked(prim.clone()).size_attr(), 2.0);
                Some(Tess::cube(s))
            }
            "Sphere" => {
                let r = self.number(&geom::Sphere::from_prim_unchecked(prim.clone()).radius_attr(), 1.0);
                Some(Tess::lathe(&Tess::sphere_rows(r, r, 0.0), None))
            }
            "Cylinder" | "Cylinder_1" => {
                let c = geom::Cylinder::from_prim_unchecked(prim.clone());
                let r = self.number(&c.radius_attr(), 1.0);
                let h = self.number(&c.height_attr(), 2.0);
                let (bottom, top) = if kind(prim)?.as_str() == "Cylinder_1" {
                    (
                        self.number(&prim.attribute("radiusBottom"), f64::from(r)),
                        self.number(&prim.attribute("radiusTop"), f64::from(r)),
                    )
                } else {
                    (r, r)
                };
                Some(Tess::lathe(
                    &Tess::frustum_rows(bottom, top, h),
                    t(&c.axis_attr()).as_deref(),
                ))
            }
            "Cone" => {
                let c = geom::Cone::from_prim_unchecked(prim.clone());
                let r = self.number(&c.radius_attr(), 1.0);
                let h = self.number(&c.height_attr(), 2.0);
                Some(Tess::lathe(
                    &Tess::frustum_rows(r, 0.0, h),
                    t(&c.axis_attr()).as_deref(),
                ))
            }
            "Capsule" | "Capsule_1" => {
                let c = geom::Capsule::from_prim_unchecked(prim.clone());
                let r = self.number(&c.radius_attr(), 0.5);
                let h = self.number(&c.height_attr(), 1.0);
                let (bottom, top) = if kind(prim)?.as_str() == "Capsule_1" {
                    (
                        self.number(&prim.attribute("radiusBottom"), f64::from(r)),
                        self.number(&prim.attribute("radiusTop"), f64::from(r)),
                    )
                } else {
                    (r, r)
                };
                let mut rows = Tess::sphere_rows(bottom, top, h);
                #[allow(clippy::float_cmp, reason = "identical rows, not near ones, make an empty band")]
                rows.dedup_by(|a, b| a == b);
                Some(Tess::lathe(&rows, t(&c.axis_attr()).as_deref()))
            }
            "Plane" => {
                let p = geom::Plane::from_prim_unchecked(prim.clone());
                let w = self.number(&p.width_attr(), 2.0);
                let l = self.number(&p.length_attr(), 2.0);
                Some(Tess::plane(w, l, t(&p.axis_attr()).as_deref()))
            }
            _ => None,
        }
    }

    fn mesh_shape(&self, mesh: &geom::Mesh) -> Option<Shape> {
        let points = vec3s(self.value(&mesh.points_attr())?)?;
        let counts = ints(self.value(&mesh.face_vertex_counts_attr()));
        let corners = ints(self.value(&mesh.face_vertex_indices_attr()));
        if points.is_empty() || counts.is_empty() {
            return None;
        }
        let holes = ints(self.value(&mesh.hole_indices_attr()));
        let left_handed = self.token(&mesh.orientation_attr()).as_deref() == Some("leftHanded");
        let api = PrimvarsAPI::from_prim_unchecked(mesh.prim().clone());
        let uvs = UV_NAMES.iter().find_map(|n| self.primvar(&api, n, vec2s));
        let normals = self.primvar(&api, "normals", vec3s).or_else(|| {
            let attr = mesh.normals_attr();
            let values = vec3s(self.value(&attr)?)?;
            let spread = attr.get_metadata::<sdf::Value>("interpolation").ok().flatten();
            Some(Primvar {
                values,
                indices: None,
                spread: Spread::of(spread.as_ref().and_then(sdf::Value::as_str)),
            })
        });
        // No normals: a subdivision surface is smooth, a polygon mesh faceted.
        let smooth = normals.is_none() && self.token(&mesh.subdivision_scheme_attr()).as_deref() != Some("none");
        let mut hole = vec![false; counts.len()];
        for h in holes {
            if let Some(slot) = usize::try_from(h).ok().and_then(|h| hole.get_mut(h)) {
                *slot = true;
            }
        }
        Some(Shape {
            points,
            counts,
            corners,
            hole,
            normals,
            uvs,
            left_handed,
            smooth,
        })
    }

    /// A point instancer's prototypes, each placed at every instance that names it.
    fn instancer(&mut self, prim: &usd::Prim, all: &[usd::Prim], xforms: &mut geom::XformCache) -> Result<()> {
        let pi = geom::PointInstancer::from_prim_unchecked(prim.clone());
        let protos = pi.prototypes_rel().targets().unwrap_or_default();
        let which = ints(self.value(&pi.proto_indices_attr()));
        let positions = self.value(&pi.positions_attr()).and_then(vec3s).unwrap_or_default();
        let orients = self
            .value(&prim.attribute("orientationsf"))
            .or_else(|| self.value(&pi.orientations_attr()))
            .and_then(quats)
            .unwrap_or_default();
        let scales = self
            .value(&prim.attribute("scales"))
            .and_then(vec3s)
            .unwrap_or_default();
        let ids = int64s(self.value(&prim.attribute("ids")));
        let hidden = int64s(self.value(&prim.attribute("invisibleIds")));
        let world = xforms.local_to_world_transform(prim).unwrap_or_default();

        // Each prototype's shown surfaces, placed relative to the prototype root's parent: the root's own
        // transform applies, its parents' don't.
        let mut parts: Vec<Vec<(usd::Prim, Shape, gf::Matrix4d)>> = Vec::new();
        for root in &protos {
            let base = match all.iter().find(|p| p.path() == root).and_then(usd::Prim::parent) {
                Some(parent) => xforms.local_to_world_transform(&parent).unwrap_or_default(),
                None => gf::Matrix4d::IDENTITY,
            };
            let inv = base.inverse().unwrap_or(gf::Matrix4d::IDENTITY);
            let mut part = Vec::new();
            for p in all.iter().filter(|p| within(p.path(), root)) {
                if !self.shown(p) {
                    continue;
                }
                if let Some(shape) = self.shape(p) {
                    let m = xforms.local_to_world_transform(p).unwrap_or_default() * inv;
                    part.push((p.clone(), shape, m));
                }
            }
            parts.push(part);
        }
        for (i, &w) in which.iter().enumerate() {
            let id = ids
                .get(i)
                .copied()
                .unwrap_or_else(|| i64::try_from(i).unwrap_or(i64::MAX));
            if hidden.contains(&id) {
                continue;
            }
            let Some(part) = usize::try_from(w).ok().and_then(|w| parts.get(w)) else {
                continue;
            };
            let s = scales.get(i).copied().unwrap_or([1.0; 3]);
            let q = orients.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]);
            let t = positions.get(i).copied().unwrap_or_default();
            let place = instance_matrix(s, q, t) * world;
            for (p, shape, m) in part.clone() {
                self.emit(&p, &shape, m * place)?;
            }
        }
        Ok(())
    }

    /// Adds a surface's faces, placed by `world`.
    fn emit(&mut self, prim: &usd::Prim, shape: &Shape, world: gf::Matrix4d) -> Result<()> {
        let points: Vec<[f32; 3]> = shape
            .points
            .iter()
            .map(|p| {
                let v = world.transform_point(gf::vec3f(p[0], p[1], p[2]));
                [v.x, v.y, v.z]
            })
            .collect();
        let normal_matrix = normal_matrix(&world);
        let flip = shape.left_handed != (det3(&world) < 0.0);
        let (counts, corners) = (&shape.counts, &shape.corners);

        // The faces to draw, by their first corner.
        let mut faces = Vec::with_capacity(counts.len());
        let mut start = 0usize;
        for (face, &n) in counts.iter().enumerate() {
            let n = usize::try_from(n).unwrap_or(0);
            let end = start.saturating_add(n);
            let ok = n >= 3
                && end <= corners.len()
                && !shape.hole.get(face).copied().unwrap_or(false)
                && corners[start..end]
                    .iter()
                    .all(|&c| usize::try_from(c).is_ok_and(|c| c < points.len()));
            if ok {
                faces.push((face, start, n));
            }
            start = end;
        }
        let point_normals = if shape.smooth {
            smooth_normals(&points, corners, &faces, flip)
        } else {
            Vec::new()
        };

        let api = PrimvarsAPI::from_prim_unchecked(prim.clone());
        let display = self.primvar(&api, "displayColor", vec3s);
        let opacity = self.primvar(&api, "displayOpacity", floats);
        let binding = bound(prim);
        // Display colors show only where no material is bound.
        let vertex_colors = binding.is_none()
            && (display.as_ref().is_some_and(|d| !d.constant()) || opacity.as_ref().is_some_and(|o| !o.constant()));
        let own = match binding {
            Some(path) => self.material(Some(path), GRAY),
            None if vertex_colors => self.material(None, [1.0; 4]),
            None => {
                let [r, g, b] = display
                    .as_ref()
                    .and_then(|d| d.values.first().copied())
                    .unwrap_or([GRAY[0], GRAY[1], GRAY[2]]);
                let a = opacity.as_ref().and_then(|o| o.values.first().copied()).unwrap_or(1.0);
                self.material(None, [r, g, b, a.clamp(0.0, 1.0)])
            }
        };

        // Each face's material: a material-binding subset's, else the surface's own.
        let mut face_material = vec![own; counts.len()];
        if let Ok(children) = prim.children() {
            for child in children {
                let Ok(Some(subset)) = geom::Subset::get(self.stage, child.path().clone()) else {
                    continue;
                };
                let family = self.token(&subset.family_name_attr());
                if family.as_deref().is_some_and(|f| f != "materialBind") {
                    continue;
                }
                let material = match bound(&child) {
                    Some(path) => self.material(Some(path), GRAY),
                    None => own,
                };
                for i in ints(self.value(&subset.indices_attr())) {
                    if let Some(slot) = usize::try_from(i).ok().and_then(|i| face_material.get_mut(i)) {
                        *slot = material;
                    }
                }
            }
        }

        let mut tris = Vec::new();
        let mut poly = Vec::new();
        for &(face, start, n) in &faces {
            ensure!(
                self.mesh.triangles() + n - 2 <= MAX_TRIANGLES,
                "too many triangles to preview"
            );
            poly.clear();
            #[allow(clippy::cast_sign_loss, reason = "faces keep only corners in 0..points.len()")]
            poly.extend(corners[start..start + n].iter().map(|&c| points[c as usize]));
            let flat = (shape.normals.is_none() && !shape.smooth).then(|| polygon_normal(&poly, flip));
            let material = face_material[face];
            let prim = self.prim_for(material);
            let first = u32::try_from(prim.positions.len()).context("too many vertices")?;
            let base = prim.positions.len();
            if shape.uvs.is_some() || !prim.uvs.is_empty() {
                prim.uvs.resize(base, [0.0, 0.0]);
            }
            if vertex_colors || !prim.colors.is_empty() {
                prim.colors.resize(base, [1.0; 4]);
            }
            for (c, &corner) in corners.iter().enumerate().skip(start).take(n) {
                #[allow(clippy::cast_sign_loss, reason = "faces keep only corners in 0..points.len()")]
                let point = corner as usize;
                prim.positions.push(points[point]);
                let normal = match (&shape.normals, flat) {
                    (Some(nv), _) => nv.at(face, point, c).map(|v| transform_normal(&normal_matrix, v)),
                    (None, Some(f)) => Some(f),
                    (None, None) => point_normals.get(point).copied(),
                };
                prim.normals.push(normal.unwrap_or([0.0, 1.0, 0.0]));
                if let Some(uv) = &shape.uvs {
                    let [u, v] = uv.at(face, point, c).unwrap_or_default();
                    prim.uvs.push([u, 1.0 - v]);
                } else if !prim.uvs.is_empty() {
                    prim.uvs.push([0.0, 0.0]);
                }
                if vertex_colors {
                    let [r, g, b] = display.as_ref().and_then(|d| d.at(face, point, c)).unwrap_or([1.0; 3]);
                    let a = opacity.as_ref().and_then(|o| o.at(face, point, c)).unwrap_or(1.0);
                    prim.colors.push([r, g, b, a.clamp(0.0, 1.0)]);
                } else if !prim.colors.is_empty() {
                    prim.colors.push([1.0; 4]);
                }
            }
            tris.clear();
            triangulate(&poly, &mut tris);
            for &[a, b, c] in &tris {
                let tri = if flip { [a, c, b] } else { [a, b, c] };
                prim.indices.extend(tri.map(|k| first + k));
            }
        }
        Ok(())
    }

    /// The prim collecting a material's triangles. Its UVs and colors stay empty until a surface with them adds
    /// to it, then are padded so every vertex has one.
    fn prim_for(&mut self, material: usize) -> &mut Prim {
        let next = self.mesh.prims.len();
        let i = *self.prims.entry(material).or_insert(next);
        if i == next {
            self.mesh.prims.push(Prim {
                material,
                ..Prim::default()
            });
        }
        &mut self.mesh.prims[i]
    }

    /// The material at `bound`, made once per material prim; with none, a plain one in `display`.
    fn material(&mut self, bound: Option<sdf::Path>, display: [f32; 4]) -> usize {
        let key = match &bound {
            Some(p) => p.as_str().to_owned(),
            None => format!("display:{display:?}"),
        };
        if let Some(&i) = self.materials.get(&key) {
            return i;
        }
        let material = match bound {
            Some(path) => self.preview_surface(&path),
            None => Material::plain("default", display),
        };
        let i = self.mesh.materials.len();
        self.mesh.materials.push(material);
        self.materials.insert(key, i);
        i
    }

    fn preview_surface(&mut self, path: &sdf::Path) -> Material {
        let name = path.as_str().rsplit('/').next().unwrap_or("material").to_owned();
        let Ok(Some(ps)) = shade::read_preview_surface(self.stage, path) else {
            return Material::plain(&name, GRAY);
        };
        // UsdPreviewSurface colors are linear, as glTF's are. Unauthored inputs take the shader's fallbacks.
        let mut color = match ps.diffuse_color.value() {
            Some(c) => [c.x, c.y, c.z, 1.0],
            None if ps.diffuse_color.texture().is_some() => [1.0; 4],
            None => [0.18, 0.18, 0.18, 1.0],
        };
        if let Some(&o) = ps.opacity.value() {
            color[3] = o.clamp(0.0, 1.0);
        }
        let specular_workflow = self
            .stage
            .prim(ps.shader.as_str())
            .ok()
            .and_then(|s| match self.value(&s.attribute("inputs:useSpecularWorkflow")) {
                Some(sdf::Value::Int(v)) => Some(v != 0),
                _ => None,
            })
            .unwrap_or(false);
        let metallic = if specular_workflow {
            0.0
        } else {
            ps.metallic.value().copied().unwrap_or(0.0)
        };
        let emissive = ps.emissive_color.value().map_or([0.0; 3], |c| [c.x, c.y, c.z]);
        let color_image = ps.diffuse_color.texture().and_then(|t| self.image(t));
        let normal_image = ps.normal.texture().and_then(|t| self.image(t));
        Material {
            color_image,
            normal_image,
            emissive,
            metallic: metallic.clamp(0.0, 1.0),
            roughness: ps.roughness.value().copied().unwrap_or(0.5).clamp(0.0, 1.0),
            ..Material::plain(&name, color)
        }
    }
}

/// A surface as polygons in its prim's space, with what it carries per point or corner.
#[derive(Clone)]
struct Shape {
    points: Vec<[f32; 3]>,
    counts: Vec<i32>,
    corners: Vec<i32>,
    /// Faces `holeIndices` leaves out.
    hole: Vec<bool>,
    normals: Option<Primvar<[f32; 3]>>,
    uvs: Option<Primvar<[f32; 2]>>,
    /// Faces wound clockwise.
    left_handed: bool,
    /// Smoothed when it has no normals: a subdivision surface.
    smooth: bool,
}

/// How finely the implicit surfaces are tessellated: around their axis, and from pole to pole. Multiples of four
/// and two, so the tessellation reaches the surface's extent on every axis.
const AROUND: usize = 32;
const STACKS: usize = 16;

/// A tessellation of an implicit surface, built around Z and then turned to the surface's axis.
#[derive(Default)]
struct Tess {
    points: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    counts: Vec<i32>,
    corners: Vec<i32>,
}

impl Tess {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "a tessellated shape has AROUND * (STACKS + 1) points at most"
    )]
    fn face(&mut self, corners: &[([f32; 3], [f32; 3])]) {
        self.counts.push(corners.len() as i32);
        for &(p, n) in corners {
            self.corners.push(self.points.len() as i32);
            self.points.push(p);
            self.normals.push(n);
        }
    }

    /// The shape, its Z axis turned to `axis` (a cyclic swap, which keeps the winding).
    fn finish(self, axis: Option<&str>) -> Shape {
        let turn = |v: [f32; 3]| match axis {
            Some("X") => [v[2], v[0], v[1]],
            Some("Y") => [v[1], v[2], v[0]],
            _ => v,
        };
        Shape {
            points: self.points.into_iter().map(turn).collect(),
            counts: self.counts,
            corners: self.corners,
            hole: Vec::new(),
            normals: Some(Primvar {
                values: self.normals.into_iter().map(turn).collect(),
                indices: None,
                spread: Spread::FaceVarying,
            }),
            uvs: None,
            left_handed: false,
            smooth: false,
        }
    }

    /// A cube of edge `size` about the origin.
    fn cube(size: f32) -> Shape {
        let h = size / 2.0;
        let mut t = Self::default();
        for k in 0..3 {
            let (u, v) = ((k + 1) % 3, (k + 2) % 3);
            for s in [1.0f32, -1.0] {
                let mut quad: Vec<([f32; 3], [f32; 3])> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                    .iter()
                    .map(|&(a, b)| {
                        let mut p = [0.0; 3];
                        let mut n = [0.0; 3];
                        p[k] = s * h;
                        p[u] = a * h;
                        p[v] = b * h;
                        n[k] = s;
                        (p, n)
                    })
                    .collect();
                if s < 0.0 {
                    quad.reverse();
                }
                t.face(&quad);
            }
        }
        t.finish(None)
    }

    /// A plane facing `axis`: `width` along X (Z when facing X), `length` along Y (Z when facing Y).
    fn plane(width: f32, length: f32, axis: Option<&str>) -> Shape {
        let (w, l) = (width / 2.0, length / 2.0);
        let (corners, n): ([[f32; 3]; 4], [f32; 3]) = match axis {
            Some("X") => (
                [[0.0, -l, -w], [0.0, l, -w], [0.0, l, w], [0.0, -l, w]],
                [1.0, 0.0, 0.0],
            ),
            Some("Y") => (
                [[-w, 0.0, -l], [-w, 0.0, l], [w, 0.0, l], [w, 0.0, -l]],
                [0.0, 1.0, 0.0],
            ),
            _ => (
                [[-w, -l, 0.0], [w, -l, 0.0], [w, l, 0.0], [-w, l, 0.0]],
                [0.0, 0.0, 1.0],
            ),
        };
        let mut t = Self::default();
        t.face(&corners.map(|p| (p, n)));
        t.finish(None)
    }

    /// Rows of a sphere (`h` 0) or capsule from bottom to top: radius, height, and the normal's radial and axial
    /// parts. The bottom half has radius `bottom` about -h/2, the top half `top` about h/2.
    fn sphere_rows(bottom: f32, top: f32, h: f32) -> Vec<[f32; 4]> {
        let mut rows = Vec::new();
        let half = STACKS / 2;
        for k in 0..=STACKS {
            let (s, c) = match k {
                0 => (-1.0, 0.0),
                k if k == STACKS => (1.0, 0.0),
                #[allow(clippy::cast_precision_loss, reason = "a small loop counter")]
                k => (std::f32::consts::PI * (k as f32 - half as f32) / STACKS as f32).sin_cos(),
            };
            if k <= half {
                rows.push([bottom * c, -h / 2.0 + bottom * s, c, s]);
            }
            if k >= half {
                rows.push([top * c, h / 2.0 + top * s, c, s]);
            }
        }
        rows
    }

    /// Rows of a capped cylinder, cone (`top` 0) or frustum of height `h`.
    fn frustum_rows(bottom: f32, top: f32, h: f32) -> Vec<[f32; 4]> {
        let len = h.hypot(bottom - top).max(f32::MIN_POSITIVE);
        let (nr, nz) = (h / len, (bottom - top) / len);
        vec![
            [0.0, -h / 2.0, 0.0, -1.0],
            [bottom, -h / 2.0, 0.0, -1.0],
            [bottom, -h / 2.0, nr, nz],
            [top, h / 2.0, nr, nz],
            [top, h / 2.0, 0.0, 1.0],
            [0.0, h / 2.0, 0.0, 1.0],
        ]
    }

    /// A surface of revolution about Z through `rows`, turned to `axis`.
    #[allow(
        clippy::float_cmp,
        reason = "a radius of exactly 0 is a pole; any other closes a band"
    )]
    fn lathe(rows: &[[f32; 4]], axis: Option<&str>) -> Shape {
        let ring = |row: [f32; 4], j: usize| {
            #[allow(clippy::cast_precision_loss, reason = "a small loop counter")]
            let (s, c) = (std::f32::consts::TAU * (j % AROUND) as f32 / AROUND as f32).sin_cos();
            ([row[0] * c, row[0] * s, row[1]], [row[2] * c, row[2] * s, row[3]])
        };
        let mut t = Self::default();
        for pair in rows.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if (a[0] == b[0] && a[1] == b[1]) || (a[0] == 0.0 && b[0] == 0.0) {
                continue;
            }
            for i in 0..AROUND {
                if a[0] == 0.0 {
                    t.face(&[ring(a, i), ring(b, i + 1), ring(b, i)]);
                } else if b[0] == 0.0 {
                    t.face(&[ring(a, i), ring(a, i + 1), ring(b, i)]);
                } else {
                    t.face(&[ring(a, i), ring(a, i + 1), ring(b, i + 1), ring(b, i)]);
                }
            }
        }
        t.finish(axis)
    }
}

/// The material bound to `prim` (directly or through a parent), if any.
fn bound(prim: &usd::Prim) -> Option<sdf::Path> {
    MaterialBindingAPI::from_prim_unchecked(prim.clone())
        .compute_bound_material("")
        .ok()
        .flatten()
}

/// A point instancer's placement of one instance: scale, then orientation (w, x, y, z), then translation, as
/// row vectors compose.
fn instance_matrix(s: [f32; 3], q: [f32; 4], t: [f32; 3]) -> gf::Matrix4d {
    let len = q.iter().map(|c| f64::from(*c).powi(2)).sum::<f64>().sqrt();
    let [w, x, y, z] = if len > 0.0 {
        q.map(|c| f64::from(c) / len)
    } else {
        [1.0, 0.0, 0.0, 0.0]
    };
    // The rotation for column vectors; row vectors take its transpose.
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut m = gf::Matrix4d::IDENTITY;
    for row in 0..3 {
        for col in 0..3 {
            m.0[row * 4 + col] = f64::from(s[row]) * r[col][row];
        }
        m.0[12 + row] = f64::from(t[row]);
    }
    m
}

/// Smooth normals as Hydra makes them: each face corner adds the cross product of its two edges to its point.
#[allow(clippy::cast_sign_loss, reason = "faces keep only corners in 0..points.len()")]
fn smooth_normals(points: &[[f32; 3]], corners: &[i32], faces: &[(usize, usize, usize)], flip: bool) -> Vec<[f32; 3]> {
    let mut acc = vec![[0f64; 3]; points.len()];
    for &(_, start, n) in faces {
        let ix = &corners[start..start + n];
        for k in 0..n {
            let (p, c, nx) = (ix[(k + n - 1) % n] as usize, ix[k] as usize, ix[(k + 1) % n] as usize);
            let at = |i: usize| points[i].map(f64::from);
            let (a, b, o) = (at(nx), at(p), at(c));
            let (u, v) = (
                [a[0] - o[0], a[1] - o[1], a[2] - o[2]],
                [b[0] - o[0], b[1] - o[1], b[2] - o[2]],
            );
            let x = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            for (s, d) in acc[c].iter_mut().zip(x) {
                *s += d;
            }
        }
    }
    acc.into_iter().map(|n| unit(n, flip)).collect()
}

/// A polygon's normal (Newell's), unit length.
fn polygon_normal(poly: &[[f32; 3]], flip: bool) -> [f32; 3] {
    let mut n = [0f64; 3];
    for (i, a) in poly.iter().enumerate() {
        let (a, b) = (a.map(f64::from), poly[(i + 1) % poly.len()].map(f64::from));
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    unit(n, flip)
}

fn unit(n: [f64; 3], flip: bool) -> [f32; 3] {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let s = if flip { -1.0 } else { 1.0 };
    if len > 0.0 {
        n.map(|x| f32_of(s * x / len))
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn vec3s(v: sdf::Value) -> Option<Vec<[f32; 3]>> {
    Some(match v {
        sdf::Value::Vec3fVec(v) => v.into_iter().map(|p| [p.x, p.y, p.z]).collect(),
        sdf::Value::Vec3dVec(v) => v.into_iter().map(|p| [p.x, p.y, p.z].map(f32_of)).collect(),
        sdf::Value::Vec3hVec(v) => v
            .into_iter()
            .map(|p| [p.x.to_f32(), p.y.to_f32(), p.z.to_f32()])
            .collect(),
        sdf::Value::Vec3f(p) => vec![[p.x, p.y, p.z]],
        sdf::Value::Vec3d(p) => vec![[p.x, p.y, p.z].map(f32_of)],
        _ => return None,
    })
}

fn vec2s(v: sdf::Value) -> Option<Vec<[f32; 2]>> {
    Some(match v {
        sdf::Value::Vec2fVec(v) => v.into_iter().map(|p| [p.x, p.y]).collect(),
        sdf::Value::Vec2dVec(v) => v.into_iter().map(|p| [p.x, p.y].map(f32_of)).collect(),
        sdf::Value::Vec2hVec(v) => v.into_iter().map(|p| [p.x.to_f32(), p.y.to_f32()]).collect(),
        _ => return None,
    })
}

fn floats(v: sdf::Value) -> Option<Vec<f32>> {
    Some(match v {
        sdf::Value::FloatVec(v) => v,
        sdf::Value::DoubleVec(v) => v.into_iter().map(f32_of).collect(),
        sdf::Value::HalfVec(v) => v.into_iter().map(openusd::gf::f16::to_f32).collect(),
        sdf::Value::Float(x) => vec![x],
        sdf::Value::Double(x) => vec![f32_of(x)],
        _ => return None,
    })
}

/// Quaternions as (w, x, y, z).
fn quats(v: sdf::Value) -> Option<Vec<[f32; 4]>> {
    Some(match v {
        sdf::Value::QuathVec(v) => v
            .into_iter()
            .map(|q| [q.w.to_f32(), q.x.to_f32(), q.y.to_f32(), q.z.to_f32()])
            .collect(),
        sdf::Value::QuatfVec(v) => v.into_iter().map(|q| [q.w, q.x, q.y, q.z]).collect(),
        _ => return None,
    })
}

fn ints(v: Option<sdf::Value>) -> Vec<i32> {
    match v {
        Some(sdf::Value::IntVec(v)) => v,
        _ => Vec::new(),
    }
}

fn int64s(v: Option<sdf::Value>) -> Vec<i64> {
    match v {
        Some(sdf::Value::Int64Vec(v)) => v,
        Some(sdf::Value::IntVec(v)) => v.into_iter().map(i64::from).collect(),
        _ => Vec::new(),
    }
}

/// Names texture coordinates go by: USD's own `st`, then what exporters write.
const UV_NAMES: &[&str] = &["st", "st0", "st_0", "UVMap", "uv", "uv0", "map1", "UVChannel_1"];

impl Builder<'_> {
    /// A texture by its asset path: inside the `.usdz`, beside the file, or anywhere in the pack by name.
    fn image(&mut self, asset: &str) -> Option<usize> {
        if let Some(&i) = self.images.get(asset) {
            return i;
        }
        let reference = asset.replace("<UDIM>", "1001").replace("<udim>", "1001");
        let reference = reference.trim_start_matches("./");
        let image = self
            .usdz
            .as_mut()
            .and_then(|z| z.read(reference))
            .and_then(|bytes| self.assets.image(&bytes, Path::new(reference)))
            .or_else(|| {
                let dir = self.file.parent().unwrap_or(Path::new(""));
                self.assets
                    .find(reference, dir)
                    .and_then(|p| self.assets.load_image(&p))
            });
        let i = image.map(|img: Image| {
            self.mesh.images.push(img);
            self.mesh.images.len() - 1
        });
        self.images.insert(asset.to_owned(), i);
        i
    }
}

/// A `.usdz` opened for its textures.
struct Usdz<'a>(zip::ZipArchive<Box<dyn super::ReadSeek + 'a>>);

impl<'a> Usdz<'a> {
    fn open(input: Input<'a>) -> Option<Self> {
        let is_usdz = input.path.extension().is_some_and(|e| e.eq_ignore_ascii_case("usdz"));
        let stream = input.seekable().ok().filter(|_| is_usdz)?;
        zip::ZipArchive::new(stream).ok().map(Self)
    }

    /// An entry by its path in the archive, or failing that by its file name.
    fn read(&mut self, reference: &str) -> Option<Vec<u8>> {
        let name = reference.rsplit('/').next()?.to_lowercase();
        let names: Vec<String> = self
            .0
            .file_names()
            .filter_map(Result::ok)
            .map(std::borrow::Cow::into_owned)
            .collect();
        let entry = names.iter().find(|n| n.eq_ignore_ascii_case(reference)).or_else(|| {
            names
                .iter()
                .find(|n| n.rsplit('/').next().is_some_and(|f| f.to_lowercase() == name))
        })?;
        let mut f = self.0.by_name(entry).ok()?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).ok()?;
        Some(bytes)
    }
}

/// How many times a model in memory is reopened for the layers it names: each round fetches the layers the last
/// one found missing, so this is the depth of references a model can have.
const FETCH_ROUNDS: usize = 32;

/// The stage of a model in memory, and its prims' paths: opened and composed, then again with whatever layers it
/// named that `assets` has, until it names no more.
fn open_in_memory(bytes: &[u8], input: Input<'_>, assets: &dyn Assets) -> Result<(usd::Stage, Vec<sdf::Path>)> {
    let mut root = memory_key(&input.path.to_string_lossy());
    let ext = input.path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if !["usd", "usda", "usdc", "usdz"]
        .iter()
        .any(|e| ext.eq_ignore_ascii_case(e))
    {
        // The format is read from the name: a model named without its extension is given one.
        root.push_str(if bytes.starts_with(b"PK") { ".usdz" } else { ".usd" });
    }
    let mut files = HashMap::new();
    files.insert(root.clone(), Arc::<[u8]>::from(bytes));
    let mut tried = BTreeSet::new();
    let mut round = 0;
    loop {
        let resolver = MemResolver::new(files.clone());
        let missing = Rc::clone(&resolver.missing);
        let stage = usd::Stage::builder()
            .schema_registry(openusd_schemas::schema_registry())
            .resolver(resolver)
            .open(&root)
            .map_err(|e| invalid!("{e}"))?;
        let paths = prim_paths(&stage)?;
        round += 1;
        let mut found = false;
        for id in missing.take() {
            if tried.insert(id.clone())
                && let Some(b) = assets.read(Path::new(&id))
            {
                files.insert(id, Arc::from(b.as_ref()));
                found = true;
            }
        }
        if !found || round == FETCH_ROUNDS {
            return Ok((stage, paths));
        }
    }
}

/// Resolves a model in memory: its layers are `files`, keyed by their paths among the assets (`/`-separated,
/// relative), and a package's layers are its entries (`pkg.usdz[inner.usda]`). A layer it doesn't have is noted in
/// `missing`, for the next round to fetch.
struct MemResolver {
    files: HashMap<String, Arc<[u8]>>,
    missing: Rc<RefCell<BTreeSet<String>>>,
    id: u64,
}

impl MemResolver {
    fn new(files: HashMap<String, Arc<[u8]>>) -> Self {
        // Ids apart from the ones `openusd` gives its own resolvers, which count up from zero.
        static NEXT: AtomicU64 = AtomicU64::new(1 << 62);
        Self {
            files,
            missing: Rc::default(),
            id: NEXT.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Whether the plain path (no package) is here, noting it missing when it isn't.
    fn has(&self, path: &str) -> bool {
        let here = self.files.contains_key(path);
        if !here {
            self.missing.borrow_mut().insert(path.to_owned());
        }
        here
    }
}

/// `leaf` placed inside the deepest package of `base`: `pkg.usdz[inner.usdz]` and `a.usda` give
/// `pkg.usdz[inner.usdz[a.usda]]`.
fn nest(base: &str, leaf: &str) -> String {
    match ar::split_package_relative_path_outer(base) {
        Some((package, inner)) => ar::join_package_relative_path(&package, &nest(&inner, leaf)),
        None => ar::join_package_relative_path(base, leaf),
    }
}

impl ar::Resolver for MemResolver {
    fn create_identifier(&self, asset_path: &str, anchor: Option<&ar::ResolvedPath>) -> String {
        if asset_path.is_empty() {
            return String::new();
        }
        if let Some((package, inner)) = ar::split_package_relative_path_outer(asset_path) {
            return nest(&self.create_identifier(&package, anchor), &memory_key(&inner));
        }
        let Some(anchor) = anchor.map(|a| a.to_string_lossy().replace('\\', "/")) else {
            return memory_key(asset_path);
        };
        if let Some((package, inner)) = ar::split_package_relative_path_inner(&anchor) {
            // A layer inside a package refers to its neighbours in the package.
            let dir = inner.rsplit_once('/').map_or("", |(dir, _)| dir);
            return nest(&package, &memory_key(&format!("{dir}/{asset_path}")));
        }
        if anchor.to_ascii_lowercase().ends_with(".usdz") {
            // A package's root layer refers to the package's entries.
            return ar::join_package_relative_path(&anchor, &memory_key(asset_path));
        }
        if asset_path.starts_with(['/', '\\']) {
            return memory_key(asset_path);
        }
        let dir = anchor.rsplit_once('/').map_or("", |(dir, _)| dir);
        memory_key(&format!("{dir}/{asset_path}"))
    }

    fn resolve(&self, asset_path: &str) -> Option<ar::ResolvedPath> {
        if asset_path.is_empty() {
            return None;
        }
        let found = match ar::split_package_relative_path_outer(asset_path) {
            Some((package, inner)) => self.has(&package) && openusd::usdz::package_contains(self, &package, &inner),
            None => self.has(asset_path),
        };
        found.then(|| ar::ResolvedPath::new(asset_path))
    }

    fn resolve_for_new_asset(&self, _asset_path: &str) -> Option<ar::ResolvedPath> {
        None
    }

    fn open_asset(&self, resolved_path: &ar::ResolvedPath) -> io::Result<Box<dyn ar::Asset>> {
        let path = resolved_path.to_string_lossy();
        let bytes = match ar::split_package_relative_path_outer(&path) {
            Some((package, inner)) => openusd::usdz::package_entry(self, &package, &inner)?,
            None => match self.files.get(path.as_ref()) {
                Some(b) => ar::AssetBuffer::from(Arc::clone(b)),
                None => {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("{path} isn't among the assets"),
                    ));
                }
            },
        };
        Ok(Box::new(io::Cursor::new(bytes)))
    }

    fn get_modification_timestamp(&self, _: &str, _: &ar::ResolvedPath) -> Option<std::time::SystemTime> {
        None
    }

    fn identity(&self) -> String {
        format!("mithril-memory-{}", self.id)
    }

    fn cache_id(&self) -> Option<u64> {
        Some(self.id)
    }
}

/// The determinant of the matrix's rotation and scale: negative when it mirrors.
fn det3(m: &gf::Matrix4d) -> f64 {
    let a = &m.0;
    a[0] * (a[5] * a[10] - a[6] * a[9]) - a[1] * (a[4] * a[10] - a[6] * a[8]) + a[2] * (a[4] * a[9] - a[5] * a[8])
}

/// The cofactors of the matrix's rotation and scale, which carry normals as the inverse transpose does (up to a
/// scale, removed by normalizing).
fn normal_matrix(m: &gf::Matrix4d) -> [[f64; 3]; 3] {
    let a = |r: usize, c: usize| m.0[r * 4 + c];
    let sign = if det3(m) < 0.0 { -1.0 } else { 1.0 };
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
            let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
            *v = sign * (a(r1, c1) * a(r2, c2) - a(r1, c2) * a(r2, c1));
        }
    }
    out
}

/// A normal through the cofactor matrix (row vector times matrix, as USD composes), unit length. Authored normals
/// are taken as they are, whatever the winding.
fn transform_normal(m: &[[f64; 3]; 3], n: [f32; 3]) -> [f32; 3] {
    let n = n.map(f64::from);
    let v: [f64; 3] = std::array::from_fn(|c| n[0] * m[0][c] + n[1] * m[1][c] + n[2] * m[2][c]);
    unit(v, false)
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;

    const CUBE: &str = r#"#usda 1.0
(
    upAxis = "Z"
    metersPerUnit = 1
)

def Xform "World"
{
    double3 xformOp:translate = (10, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Quad" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        uniform token subdivisionScheme = "none"
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (interpolation = "faceVarying")
        rel material:binding = </World/Looks/Red>
    }

    def Mesh "Hidden"
    {
        token visibility = "invisible"
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
    }

    def Scope "Looks"
    {
        def Material "Red"
        {
            token outputs:surface.connect = </World/Looks/Red/Surface.outputs:surface>
            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.8, 0.1, 0.1)
                token outputs:surface
            }
        }
    }
}
"#;

    #[test]
    fn reads_meshes_transforms_and_materials() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("quad.usda");
        std::fs::write(&path, CUBE).unwrap();
        let mesh = load(Input::file(&path), &crate::assets::Folder::new(dir.path()))
            .unwrap()
            .unwrap();
        assert_eq!(mesh.prims.len(), 1, "the invisible mesh is left out");
        assert_eq!(mesh.triangles(), 2);
        let prim = &mesh.prims[0];
        // Translated 10 along X; Z up becomes Y up, so the quad in the XY plane lies in XZ.
        assert_eq!(prim.positions[1], [11.0, 0.0, 0.0]);
        assert_eq!(prim.positions[2], [11.0, 0.0, -1.0]);
        let n = prim.normals[0];
        assert!((n[1] - 1.0).abs() < 1e-5, "the +Z face normal points up: {n:?}");
        assert_eq!(prim.uvs[2], [1.0, 0.0], "V flipped to glTF's top-left origin");
        let m = &mesh.materials[prim.material];
        assert_eq!(m.name, "Red");
        assert!((m.color[0] - 0.8).abs() < 1e-6 && (m.color[1] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn a_material_library_has_no_mesh() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("looks.usda");
        std::fs::write(
            &path,
            "#usda 1.0
def Material \"Planks\"
{
}
",
        )
        .unwrap();
        assert!(
            load(Input::file(&path), &crate::assets::Folder::new(dir.path()))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_mesh_with_no_material_shows_its_display_color() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tri.usda");
        std::fs::write(
            &path,
            "#usda 1.0\ndef Mesh \"Tri\"\n{\n    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n    color3f[] primvars:displayColor = [(0, 0.5, 1)] (interpolation = \"constant\")\n}\n",
        )
        .unwrap();
        let mesh = load(Input::file(&path), &crate::assets::Folder::new(dir.path()))
            .unwrap()
            .unwrap();
        assert_eq!(mesh.materials[mesh.prims[0].material].color, [0.0, 0.5, 1.0, 1.0]);
        // Default units are centimeters.
        assert_eq!(mesh.prims[0].positions[1], [0.01, 0.0, 0.0]);
    }
}
