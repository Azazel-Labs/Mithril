//! USD (`.usd`, `.usda`, `.usdc`, `.usdz`) as a glTF document, read with the pure-Rust `openusd` crate. The composed
//! stage (references, payloads, variants and instances resolved) becomes the node hierarchy, each prim a node with its
//! local transform. Meshes are in their prim's own space and shared between instances; a point instancer places its
//! prototypes under a node per instance. The stage's up axis and units become a root node that carries them to glTF's
//! Y-up meters, and are recorded in `asset.extras.source`.
//!
//! Meshes keep their polygons' outlines (concave faces are ear-clipped, `holeIndices` faces left out) and their
//! normals, texture coordinates and display colors in any interpolation; with no normals a subdivision surface is
//! smoothed and a polygonal one faceted. The implicit surfaces (Cube, Sphere, Cylinder, Cone, Capsule, Plane) are
//! tessellated. Values are read at the stage's start time when it gives a time range, else at the earliest sample,
//! so the rest pose is the first frame.
//!
//! `UsdSkel` skeletons become joint nodes under their skeleton's node, skinned meshes glTF skins (joint influences
//! and the geometry's bind transform), blend shapes morph targets. Animated transforms, skeletal animations and
//! blend shape weights become one animation, sampled where the file has samples (long turns cut into short steps).
//! Cameras and lights (`KHR_lights_punctual`) come across with their prims; `userProperties:` attributes and
//! `customData` become node extras.
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
    PointBasedSchema, PointInstancerSchema, PrimvarsAPI, SphereSchema, SubsetSchema, XformQuery,
};
use openusd_schemas::shade::{self, MaterialBindingAPI};
use serde_json::{Map, Value, json};

use super::Input;
use super::xform::{IDENTITY, M4, Trs, decompose, inverse, mul, scale, set_transform, translate};
use crate::assets::{Assets, memory_key};
use crate::document::{Builder, Document, Keep, Part};
use crate::images::{Images, Location};
use crate::material::{self, Alpha, Clearcoat, Factor, Full, Specular, Tex, UvTransform};
use crate::mesh::{GRAY, MAX_TRIANGLES, Material, Prim, triangulate};
use crate::num::f32_of;

/// Prim types that are no node: shading networks, and what the prims using them read (subsets, blend shapes,
/// skeletal animations).
const NOT_NODES: &[&str] = &[
    "Material",
    "Shader",
    "NodeGraph",
    "GeomSubset",
    "BlendShape",
    "SkelAnimation",
];

/// The longest turn between two animation samples, in degrees: longer ones are cut into steps.
const MAX_STEP_DEGREES: f64 = 10.0;

/// The most nodes a stage makes: point instancers can place millions of instances.
const MAX_NODES: usize = 1 << 20;

/// How deeply point instancers nest in each other's prototypes.
const MAX_NESTING: usize = 8;

/// The stage as a document; one with nothing to show (a material library, a layer of overrides) has no nodes.
pub(crate) fn load(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
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
    let tcps = Some(stage.time_codes_per_second())
        .filter(|t| t.is_finite() && *t > 0.0)
        .unwrap_or(24.0);
    let prims: Vec<usd::Prim> = paths.iter().filter_map(|p| stage.prim(p.clone()).ok()).collect();
    let mut r = Reader {
        stage: &stage,
        file: path,
        assets,
        keep,
        time,
        meters,
        usdz: Usdz::open(input),
        b: Builder::default(),
        nodes: Vec::new(),
        meshes: Vec::new(),
        mesh_triangles: Vec::new(),
        mesh_of: HashMap::new(),
        materials: Vec::new(),
        material_of: HashMap::new(),
        images: Images::new(!keep.image_bytes, input.dir()),
        image_of: HashMap::new(),
        skins: Vec::new(),
        cameras: Vec::new(),
        lights: Vec::new(),
        animated: Vec::new(),
        skeletons: HashMap::new(),
        pending: Vec::new(),
        morphed: Vec::new(),
        skinned_roots: Vec::new(),
        drawn: 0,
    };
    let roots = r.walk(&prims, &prims, None, 0)?;
    r.skin_meshes()?;
    r.pose_weights();
    if r.nodes.is_empty() {
        return Ok(Document::empty(Map::new()));
    }
    let animations = if keep.animations {
        r.animations(tcps)
    } else {
        Vec::new()
    };

    let mut basis = if z_up {
        [[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]]
    } else {
        IDENTITY
    };
    basis = mul(&basis, &scale([meters; 3]));
    let mut scene_nodes = roots;
    #[allow(clippy::float_cmp, reason = "only the exact identity needs no root")]
    if basis != IDENTITY {
        let mut root = json!({ "name": "RootNode", "children": scene_nodes });
        set_transform(&mut root, &basis);
        r.nodes.push(root);
        scene_nodes = vec![r.nodes.len() - 1];
    }
    scene_nodes.append(&mut r.skinned_roots);

    let mut source = json!({
        "format": "USD",
        "upAxis": if z_up { "Z" } else { "Y" },
        "metersPerUnit": meters,
        "timeCodesPerSecond": tcps,
    });
    if stage.has_authored_time_code_range() {
        source["startTimeCode"] = json!(stage.start_time_code());
        source["endTimeCode"] = json!(stage.end_time_code());
    }
    if let Some(p) = stage.default_prim() {
        source["defaultPrim"] = json!(p.as_str());
    }
    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": scene_nodes }]));
    json.insert("nodes".into(), Value::Array(std::mem::take(&mut r.nodes)));
    if !r.meshes.is_empty() {
        json.insert("meshes".into(), Value::Array(std::mem::take(&mut r.meshes)));
    }
    if !r.skins.is_empty() {
        json.insert("skins".into(), Value::Array(std::mem::take(&mut r.skins)));
    }
    if !animations.is_empty() {
        json.insert("animations".into(), Value::Array(animations));
    }
    if !r.cameras.is_empty() {
        json.insert("cameras".into(), Value::Array(std::mem::take(&mut r.cameras)));
    }
    if !r.lights.is_empty() {
        json.insert("extensionsUsed".into(), json!(["KHR_lights_punctual"]));
        json.insert(
            "extensions".into(),
            json!({ "KHR_lights_punctual": { "lights": std::mem::take(&mut r.lights) } }),
        );
    }
    let (materials, images) = (std::mem::take(&mut r.materials), std::mem::take(&mut r.images));
    let mut b = r.b;
    b.materials(&materials, images.list, &mut json);
    Ok(b.finish(json, Map::new()))
}

/// The paths of the stage's prims, composed (which reads the layers they reference), parents before children.
fn prim_paths(stage: &usd::Stage) -> Result<Vec<sdf::Path>> {
    let mut paths = Vec::new();
    stage
        .traverse(usd::PrimPredicate::DEFAULT.with_instance_proxies(true), |p| {
            paths.push(p.clone());
        })
        .map_err(|e| invalid!("{e}"))?;
    Ok(paths)
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

/// The last part of a path: a prim's name, or a joint's.
fn leaf(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// A USD matrix (row vectors) as an [`M4`] (column vectors): its transpose.
fn m4(m: &gf::Matrix4d) -> M4 {
    std::array::from_fn(|i| std::array::from_fn(|j| m.0[j * 4 + i]))
}

/// The rotation for a unit quaternion (x, y, z, w).
fn rotation([x, y, z, w]: [f64; 4]) -> M4 {
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            0.0,
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            0.0,
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// A quaternion (w, x, y, z), as USD lists one, as glTF's unit (x, y, z, w); `None` when it has no length.
fn unit_quaternion([w, x, y, z]: [f32; 4]) -> Option<[f64; 4]> {
    let q = [x, y, z, w].map(f64::from);
    let len = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    (len > 0.0 && len.is_finite()).then(|| q.map(|c| c / len))
}

/// The angle between two unit quaternions' rotations, in degrees.
fn turn(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let dot: f64 = (0..4).map(|k| a[k] * b[k]).sum();
    2.0 * dot.abs().min(1.0).acos().to_degrees()
}

/// Times sorted, without repeats or ones that aren't finite.
fn timeline(mut times: Vec<f64>) -> Vec<f64> {
    times.retain(|t| t.is_finite());
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    times
}

/// A `UsdSkel` skeleton drawn as joint nodes.
struct Skel {
    /// Joint paths (`Hips/Spine`), in the skeleton's order.
    joints: Vec<String>,
    /// Each joint's node.
    nodes: Vec<usize>,
    /// Each joint's bind transform in the skeleton's space (row vectors), identity where the file has none.
    binds: Vec<gf::Matrix4d>,
    /// Each joint's local transform with no animation: its rest transform, or what its bind transform makes it.
    rest: Vec<M4>,
    /// The `SkelAnimation` driving it.
    animation: Option<sdf::Path>,
}

/// A skinned mesh waiting for its skeleton's joint nodes.
struct Pending {
    /// The mesh prim's node, which keeps its place in the hierarchy.
    node: usize,
    mesh: usize,
    prim: usd::Prim,
    skeleton: String,
}

/// A skeletal animation's values at one time, per joint of `joints`.
struct SkelSample {
    joints: Vec<String>,
    translations: Vec<[f32; 3]>,
    /// (w, x, y, z).
    rotations: Vec<[f32; 4]>,
    scales: Vec<[f32; 3]>,
}

impl SkelSample {
    /// Joint `k`'s local transform as translation, rotation and scale, `rest`'s where the animation has none.
    fn trs(&self, k: usize, rest: &Trs) -> ([f64; 3], [f64; 4], [f64; 3]) {
        let t = self.translations.get(k).map_or(rest.t, |v| v.map(f64::from));
        let r = self
            .rotations
            .get(k)
            .and_then(|q| unit_quaternion(*q))
            .unwrap_or(rest.r);
        let s = self.scales.get(k).map_or(rest.s, |v| v.map(f64::from));
        (t, r, s)
    }
}

/// One animation channel's samples, before they go into the buffer.
struct Channel {
    node: usize,
    path: &'static str,
    times: Vec<f64>,
    values: Vec<f64>,
}

struct Reader<'a> {
    stage: &'a usd::Stage,
    file: &'a Path,
    assets: &'a dyn Assets,
    keep: Keep,
    time: usd::TimeCode,
    meters: f64,
    usdz: Option<Usdz<'a>>,
    b: Builder,
    nodes: Vec<Value>,
    meshes: Vec<Value>,
    /// Each mesh's triangles.
    mesh_triangles: Vec<usize>,
    /// A surface (its prototype's path when it's an instance, its material binding, whether it's skinned) to its
    /// mesh, `None` when it has no faces.
    mesh_of: HashMap<String, Option<usize>>,
    materials: Vec<Full>,
    /// Material prim path (or `display:<rgba>` for a display color) to its index in `materials`.
    material_of: HashMap<String, usize>,
    images: Images,
    /// Texture asset path to its index in `images`, or `None` when it couldn't be found.
    image_of: HashMap<String, Option<usize>>,
    skins: Vec<Value>,
    cameras: Vec<Value>,
    lights: Vec<Value>,
    /// Nodes whose transforms are animated, and their prims.
    animated: Vec<(usize, usd::Prim)>,
    /// Skeleton prim path to its joints.
    skeletons: HashMap<String, Skel>,
    pending: Vec<Pending>,
    /// Nodes drawing a mesh with blend shapes, and the mesh prim.
    morphed: Vec<(usize, usd::Prim)>,
    /// Nodes for skinned meshes, at the scene's root: glTF places a skinned mesh by its joints alone.
    skinned_roots: Vec<usize>,
    /// Triangles drawn, every instance counted.
    drawn: usize,
}

impl Reader<'_> {
    fn push_node(&mut self, node: Value) -> Result<usize> {
        ensure!(self.nodes.len() < MAX_NODES, "too many nodes (over {MAX_NODES})");
        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    fn add_child(&mut self, parent: usize, child: usize) {
        let p = &mut self.nodes[parent];
        match p.get_mut("children").and_then(Value::as_array_mut) {
            Some(c) => c.push(json!(child)),
            None => p["children"] = json!([child]),
        }
    }

    /// Nodes for `prims` (parents before children): all of them when `scope` is `None`, else the prototype rooted
    /// at `scope.0`, hung under node `scope.1`. `all` is every prim, where point instancers find their prototypes.
    /// Returns the nodes with no parent node.
    fn walk(
        &mut self,
        prims: &[usd::Prim],
        all: &[usd::Prim],
        scope: Option<(&sdf::Path, usize)>,
        depth: usize,
    ) -> Result<Vec<usize>> {
        let mut node_of: HashMap<&str, usize> = HashMap::new();
        let mut roots = Vec::new();
        // A point instancer's prototypes show only where it places them.
        let mut instancers: Vec<&sdf::Path> = Vec::new();
        for prim in prims {
            let path = prim.path();
            if scope.is_some_and(|(root, _)| !within(path, root)) || instancers.iter().any(|i| within(path, i)) {
                continue;
            }
            let Some(kind) = kind(prim) else { continue };
            if NOT_NODES.contains(&kind.as_str()) {
                continue;
            }
            let parent = match scope {
                Some((root, top)) if path == root => Some(top),
                _ => match prim.parent() {
                    Some(p) if p.path().as_str() != "/" => match node_of.get(p.path().as_str()) {
                        Some(&n) => Some(n),
                        // Its parent isn't drawn.
                        None => continue,
                    },
                    _ if scope.is_some() => continue,
                    _ => None,
                },
            };
            if !self.shown(prim) {
                continue;
            }
            let query = XformQuery::for_prim(prim).ok();
            let local = query
                .as_ref()
                .and_then(|q| q.local_transformation(self.time).ok())
                .map_or(IDENTITY, |m| m4(&m));
            let mut node = json!({ "name": leaf(path.as_str()) });
            set_transform(&mut node, &local);
            if self.keep.extras {
                let extras = self.extras(prim);
                if !extras.is_empty() {
                    node["extras"] = Value::Object(extras);
                }
            }
            let n = self.push_node(node)?;
            // A prim that resets the transform stack ignores its parents' transforms.
            let resets = query.as_ref().is_some_and(XformQuery::resets_xform_stack);
            match (parent, scope) {
                (_, Some((_, top))) if resets => self.add_child(top, n),
                (Some(p), _) if !resets => self.add_child(p, n),
                _ => roots.push(n),
            }
            node_of.insert(path.as_str(), n);
            if self.keep.animations
                && query
                    .as_ref()
                    .is_some_and(|q| q.transform_might_be_time_varying().unwrap_or(false))
            {
                self.animated.push((n, prim.clone()));
            }
            match kind.as_str() {
                "PointInstancer" => {
                    instancers.push(path);
                    if depth < MAX_NESTING {
                        self.instancer(prim, all, n, depth)?;
                    }
                }
                "Skeleton" => self.skeleton(prim, n)?,
                "Camera" if self.keep.cameras => {
                    let camera = self.camera(prim);
                    self.cameras.push(camera);
                    self.nodes[n]["camera"] = json!(self.cameras.len() - 1);
                }
                "DistantLight" | "SphereLight" | "DiskLight" | "RectLight" | "CylinderLight" if self.keep.lights => {
                    let light = self.light(prim, &kind);
                    self.lights.push(light);
                    self.nodes[n]["extensions"] = json!({ "KHR_lights_punctual": { "light": self.lights.len() - 1 } });
                }
                _ => {
                    if let Some(shape) = self.shape(prim) {
                        self.surface(prim, &shape, n)?;
                    }
                }
            }
        }
        Ok(roots)
    }

    /// A point instancer's instances: a node each under `node`, holding its prototype's nodes.
    fn instancer(&mut self, prim: &usd::Prim, all: &[usd::Prim], node: usize, depth: usize) -> Result<()> {
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
        // Each prototype's prims, gathered once.
        let parts: Vec<Vec<usd::Prim>> = protos
            .iter()
            .map(|root| all.iter().filter(|p| within(p.path(), root)).cloned().collect())
            .collect();
        for (i, &w) in which.iter().enumerate() {
            let id = ids
                .get(i)
                .copied()
                .unwrap_or_else(|| i64::try_from(i).unwrap_or(i64::MAX));
            if hidden.contains(&id) {
                continue;
            }
            let Some(w) = usize::try_from(w).ok().filter(|&w| w < protos.len()) else {
                continue;
            };
            let s = scales.get(i).copied().unwrap_or([1.0; 3]);
            let q = orients.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]);
            let t = positions.get(i).copied().unwrap_or_default();
            let mut inst = json!({ "name": format!("instance_{i}") });
            set_transform(&mut inst, &m4(&instance_matrix(s, q, t)));
            let n = self.push_node(inst)?;
            self.add_child(node, n);
            self.walk(&parts[w], all, Some((&protos[w], n)), depth + 1)?;
        }
        Ok(())
    }

    /// A skeleton's joints, as nodes under its node `node`.
    fn skeleton(&mut self, prim: &usd::Prim, node: usize) -> Result<()> {
        let joints = tokens(self.value(&prim.attribute("joints")));
        let binds = matrices(self.value(&prim.attribute("bindTransforms")));
        let rests = matrices(self.value(&prim.attribute("restTransforms")));
        let binds: Vec<gf::Matrix4d> = if binds.len() == joints.len() {
            binds
        } else {
            vec![gf::Matrix4d::IDENTITY; joints.len()]
        };
        let index: HashMap<&str, usize> = joints.iter().enumerate().map(|(i, j)| (j.as_str(), i)).collect();
        let parents: Vec<Option<usize>> = joints
            .iter()
            .enumerate()
            .map(|(i, j)| {
                j.rsplit_once('/')
                    .and_then(|(p, _)| index.get(p).copied())
                    .filter(|&p| p < i)
            })
            .collect();
        let rest: Vec<M4> = (0..joints.len())
            .map(|i| {
                if rests.len() == joints.len() {
                    return m4(&rests[i]);
                }
                let bind = m4(&binds[i]);
                parents[i].map_or(bind, |p| mul(&inverse(&m4(&binds[p])), &bind))
            })
            .collect();
        let animation = inherited_target(prim, "skel:animationSource");
        let posed = animation
            .as_ref()
            .and_then(|a| self.skel_sample(a, self.time))
            .map(|s| {
                let at: HashMap<&str, usize> = s.joints.iter().enumerate().map(|(k, j)| (j.as_str(), k)).collect();
                joints
                    .iter()
                    .enumerate()
                    .map(|(i, j)| {
                        let k = *at.get(j.as_str())?;
                        let (t, r, sc) = s.trs(k, &decompose(&rest[i]));
                        Some(mul(&mul(&translate(t), &rotation(r)), &scale(sc)))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut nodes = Vec::with_capacity(joints.len());
        for (i, j) in joints.iter().enumerate() {
            let local = posed.get(i).copied().flatten().unwrap_or(rest[i]);
            let mut n = json!({ "name": leaf(j) });
            set_transform(&mut n, &local);
            let n = self.push_node(n)?;
            self.add_child(parents[i].map_or(node, |p| nodes[p]), n);
            nodes.push(n);
        }
        self.skeletons.insert(
            prim.path().as_str().to_owned(),
            Skel {
                joints,
                nodes,
                binds,
                rest,
                animation,
            },
        );
        Ok(())
    }

    /// A skeletal animation's values at `time`.
    fn skel_sample(&self, animation: &sdf::Path, time: usd::TimeCode) -> Option<SkelSample> {
        let a = self.stage.prim(animation.clone()).ok()?;
        let at = |name: &str| a.attribute(name).get_at::<sdf::Value>(time).ok().flatten();
        Some(SkelSample {
            joints: tokens(at("joints")),
            translations: at("translations").and_then(vec3s).unwrap_or_default(),
            rotations: at("rotations").and_then(quats).unwrap_or_default(),
            scales: at("scales").and_then(vec3s).unwrap_or_default(),
        })
    }

    /// A camera's glTF form. Apertures and focal length are in tenths of a scene unit; the clipping planes are in
    /// scene units, which a node's scale doesn't reach, so they're converted here.
    fn camera(&self, prim: &usd::Prim) -> Value {
        let num = |name: &str, fallback: f64| f64::from(self.number(&prim.attribute(name), fallback));
        let focal = num("focalLength", 50.0).max(1e-6);
        let h = num("horizontalAperture", 20.955).max(1e-6);
        let v = num("verticalAperture", 15.2908).max(1e-6);
        let [near, far] = self
            .value(&prim.attribute("clippingRange"))
            .as_ref()
            .and_then(pair)
            .unwrap_or([1.0, 1_000_000.0]);
        let near = (near * self.meters).max(1e-6);
        let far = (far * self.meters).max(near * 1.000_001);
        if self.token(&prim.attribute("projection")).as_deref() == Some("orthographic") {
            return json!({ "type": "orthographic", "orthographic": {
                "xmag": h / 20.0 * self.meters, "ymag": v / 20.0 * self.meters, "znear": near, "zfar": far,
            } });
        }
        json!({ "type": "perspective", "perspective": {
            "yfov": 2.0 * (v / 2.0 / focal).atan(), "aspectRatio": h / v, "znear": near, "zfar": far,
        } })
    }

    /// A light's `KHR_lights_punctual` form. A disk or rectangle lights one side, as a spot with a hemisphere's cone.
    fn light(&self, prim: &usd::Prim, kind: &str) -> Value {
        // UsdLux's inputs, or the names before they had the `inputs:` prefix.
        let input = |name: &str| {
            self.value(&prim.attribute(format!("inputs:{name}").as_str()))
                .or_else(|| self.value(&prim.attribute(name)))
        };
        let number = |name: &str, fallback: f64| match input(name) {
            Some(sdf::Value::Float(v)) => f64::from(v),
            Some(sdf::Value::Double(v)) => v,
            Some(sdf::Value::Half(v)) => f64::from(v.to_f32()),
            _ => fallback,
        };
        let color = input("color")
            .and_then(vec3s)
            .and_then(|c| c.first().copied())
            .unwrap_or([1.0; 3]);
        let intensity = number("intensity", 1.0).max(0.0) * number("exposure", 0.0).exp2();
        let mut out = json!({ "color": color, "intensity": intensity });
        let half = std::f64::consts::FRAC_PI_2;
        match kind {
            "DistantLight" => out["type"] = json!("directional"),
            "DiskLight" | "RectLight" => {
                out["type"] = json!("spot");
                out["spot"] = json!({ "innerConeAngle": 0.0, "outerConeAngle": half });
            }
            _ => {
                let cone = number("shaping:cone:angle", 90.0);
                if kind == "SphereLight" && cone < 90.0 {
                    let outer = cone.to_radians().clamp(1e-4, half);
                    let soft = number("shaping:cone:softness", 0.0).clamp(0.0, 1.0);
                    out["type"] = json!("spot");
                    out["spot"] = json!({ "innerConeAngle": outer * (1.0 - soft) * 0.999, "outerConeAngle": outer });
                } else {
                    out["type"] = json!("point");
                }
            }
        }
        out
    }

    /// A prim's `userProperties:` attributes (by their names after the prefix) and its `customData`.
    fn extras(&self, prim: &usd::Prim) -> Map<String, Value> {
        let mut out = Map::new();
        for attr in prim
            .authored_attributes_in_namespace("userProperties")
            .unwrap_or_default()
        {
            let name = attr.name();
            let name = name.strip_prefix("userProperties:").unwrap_or(name);
            if let Some(v) = self.value(&attr).as_ref().and_then(json_of) {
                out.insert(name.to_owned(), v);
            }
        }
        if let Ok(Some(data)) = prim.custom_data()
            && let Some(v) = json_of(&data).filter(|v| v.as_object().is_some_and(|o| !o.is_empty()))
        {
            out.insert("customData".into(), v);
        }
        out
    }

    /// A surface on node `node`: its mesh there, or for a skinned surface, on a node of its own once its skeleton
    /// is drawn.
    fn surface(&mut self, prim: &usd::Prim, shape: &Shape, node: usize) -> Result<()> {
        let skin = if self.keep.skins { self.skin_binding(prim) } else { None };
        let Some(mesh) = self.mesh(prim, shape, skin.as_ref().map(|s| s.1))? else {
            return Ok(());
        };
        self.drawn += self.mesh_triangles[mesh];
        ensure!(self.drawn <= MAX_TRIANGLES, "too many triangles (over {MAX_TRIANGLES})");
        if let Some((skeleton, _)) = skin {
            self.pending.push(Pending {
                node,
                mesh,
                prim: prim.clone(),
                skeleton,
            });
        } else {
            self.hold(node, mesh, prim);
        }
        Ok(())
    }

    /// Puts `mesh` on node `node`.
    fn hold(&mut self, node: usize, mesh: usize, prim: &usd::Prim) {
        self.nodes[node]["mesh"] = json!(mesh);
        if self.meshes[mesh].get("weights").is_some() {
            self.morphed.push((node, prim.clone()));
        }
    }

    /// The skeleton a skinned surface is bound to and how many joints its influences name, when it has joint
    /// influences and a skeleton.
    fn skin_binding(&self, prim: &usd::Prim) -> Option<(String, usize)> {
        self.value(&prim.attribute("primvars:skel:jointIndices"))?;
        let path = inherited_target(prim, "skel:skeleton")?;
        let skel = self.stage.prim(path.clone()).ok()?;
        if kind(&skel).as_deref() != Some("Skeleton") {
            return None;
        }
        let own = tokens(self.value(&prim.attribute("skel:joints")));
        let count = if own.is_empty() {
            tokens(self.value(&skel.attribute("joints"))).len()
        } else {
            own.len()
        };
        (count > 0).then(|| (path.as_str().to_owned(), count))
    }

    /// Skins for the skinned surfaces, now that every skeleton has its joint nodes.
    fn skin_meshes(&mut self) -> Result<()> {
        for p in std::mem::take(&mut self.pending) {
            let Some(skel) = self.skeletons.get(&p.skeleton).filter(|s| !s.nodes.is_empty()) else {
                // The skeleton isn't drawn: the surface goes unskinned where it is.
                if let Some(shape) = self.shape(&p.prim)
                    && let Some(mesh) = self.mesh(&p.prim, &shape, None)?
                {
                    self.hold(p.node, mesh, &p.prim);
                }
                continue;
            };
            let own = tokens(self.value(&p.prim.attribute("skel:joints")));
            let names = if own.is_empty() { &skel.joints } else { &own };
            let geom_bind = match self.value(&p.prim.attribute("primvars:skel:geomBindTransform")) {
                Some(sdf::Value::Matrix4d(m)) => m,
                _ => gf::Matrix4d::IDENTITY,
            };
            let mut joints = Vec::with_capacity(names.len());
            let mut ibms = Vec::with_capacity(names.len() * 16);
            for name in names {
                // A joint the skeleton doesn't have takes its first.
                let j = skel.joints.iter().position(|s| s == name).unwrap_or(0);
                joints.push(skel.nodes[j]);
                // Mesh space to skeleton space at bind time, then to the joint's: row vectors, whose layout is
                // glTF's column-major one for column vectors.
                let ibm = geom_bind * skel.binds[j].inverse().unwrap_or(gf::Matrix4d::IDENTITY);
                ibms.extend(ibm.0.map(f32_of));
            }
            let acc = self.b.floats(ibms, "MAT4", false);
            let name = self.nodes[p.node]["name"].clone();
            self.skins
                .push(json!({ "joints": joints, "inverseBindMatrices": acc, "skeleton": skel.nodes[0] }));
            let n = self.push_node(json!({ "name": name, "skin": self.skins.len() - 1 }))?;
            self.hold(n, p.mesh, &p.prim);
            self.skinned_roots.push(n);
        }
        Ok(())
    }

    /// Blend shape weights at the rest time, on the nodes drawing meshes with blend shapes.
    fn pose_weights(&mut self) {
        for (node, prim) in std::mem::take(&mut self.morphed) {
            if let Some(w) = self.weights_at(node, &prim, self.time) {
                self.nodes[node]["weights"] = json!(w);
            }
            self.morphed.push((node, prim));
        }
    }

    /// The `SkelAnimation` that sets a surface's blend shape weights: its skeleton's.
    fn blend_animation(&self, prim: &usd::Prim) -> Option<usd::Prim> {
        let skel = self.stage.prim(inherited_target(prim, "skel:skeleton")?).ok()?;
        self.stage.prim(inherited_target(&skel, "skel:animationSource")?).ok()
    }

    /// The weights of the mesh on `node` at `time`, in its targets' order, from the surface's animation.
    fn weights_at(&self, node: usize, prim: &usd::Prim, time: usd::TimeCode) -> Option<Vec<f32>> {
        let anim = self.blend_animation(prim)?;
        let mesh = usize::try_from(self.nodes[node]["mesh"].as_u64()?).ok()?;
        let names = self.meshes[mesh].pointer("/extras/targetNames")?.as_array()?;
        let animated = tokens(anim.attribute("blendShapes").get_at::<sdf::Value>(time).ok().flatten());
        let w = floats(anim.attribute("blendShapeWeights").get_at::<sdf::Value>(time).ok()??)?;
        Some(
            names
                .iter()
                .map(|n| {
                    animated
                        .iter()
                        .position(|a| Some(a.as_str()) == n.as_str())
                        .and_then(|k| w.get(k).copied())
                        .unwrap_or(0.0)
                })
                .collect(),
        )
    }

    /// One animation: animated transforms, skeletal animations and blend shape weights.
    fn animations(&mut self, tcps: f64) -> Vec<Value> {
        let mut channels = Vec::new();
        for (node, prim) in std::mem::take(&mut self.animated) {
            self.transform_channels(node, &prim, &mut channels);
        }
        let mut skels: Vec<&Skel> = self.skeletons.values().filter(|s| s.animation.is_some()).collect();
        skels.sort_by_key(|s| s.nodes.first().copied());
        for skel in skels {
            let Some(a) = skel.animation.as_ref().and_then(|a| self.stage.prim(a.clone()).ok()) else {
                continue;
            };
            let attrs = ["translations", "rotations", "scales"].map(|n| a.attribute(n));
            let times = timeline(usd::Attribute::unioned_time_samples(&attrs).unwrap_or_default());
            if times.len() < 2 {
                continue;
            }
            let path = skel.animation.clone().unwrap_or_else(|| a.path().clone());
            let samples: Vec<SkelSample> = times
                .iter()
                .filter_map(|&t| self.skel_sample(&path, usd::TimeCode::new(t)))
                .collect();
            if samples.len() != times.len() {
                continue;
            }
            for (j, name) in skel.joints.iter().enumerate() {
                let Some(k) = samples[0].joints.iter().position(|n| n == name) else {
                    continue;
                };
                let rest = decompose(&skel.rest[j]);
                let mut t = Vec::with_capacity(times.len() * 3);
                let mut r: Vec<[f64; 4]> = Vec::with_capacity(times.len());
                let mut s = Vec::with_capacity(times.len() * 3);
                for sample in &samples {
                    let (tv, mut rv, sv) = sample.trs(k, &rest);
                    if let Some(prev) = r.last()
                        && (0..4).map(|c| prev[c] * rv[c]).sum::<f64>() < 0.0
                    {
                        rv = rv.map(|c| -c);
                    }
                    t.extend(tv);
                    r.push(rv);
                    s.extend(sv);
                }
                let node = skel.nodes[j];
                for (path, values) in [("translation", t), ("rotation", r.into_flattened()), ("scale", s)] {
                    let width = if path == "rotation" { 4 } else { 3 };
                    let first = &values[..width];
                    if values
                        .chunks_exact(width)
                        .all(|v| v.iter().zip(first).all(|(a, b)| (a - b).abs() < 1e-7))
                    {
                        continue;
                    }
                    channels.push(Channel {
                        node,
                        path,
                        times: times.clone(),
                        values,
                    });
                }
            }
        }
        for (node, prim) in self.morphed.clone() {
            self.weight_channel(node, &prim, &mut channels);
        }
        if channels.is_empty() {
            return Vec::new();
        }
        let shift = channels.iter().map(|c| c.times[0]).fold(0.0, f64::min);
        let mut inputs: HashMap<Vec<u64>, usize> = HashMap::new();
        let (mut samplers, mut out) = (Vec::new(), Vec::new());
        for c in channels {
            let key: Vec<u64> = c.times.iter().map(|t| t.to_bits()).collect();
            let input = if let Some(&i) = inputs.get(&key) {
                i
            } else {
                let secs = c.times.iter().map(|&t| f32_of((t - shift) / tcps)).collect();
                let i = self.b.floats(secs, "SCALAR", true);
                inputs.insert(key, i);
                i
            };
            let kind = match c.path {
                "translation" | "scale" => "VEC3",
                "rotation" => "VEC4",
                _ => "SCALAR",
            };
            let output = self.b.floats(c.values.into_iter().map(f32_of).collect(), kind, false);
            samplers.push(json!({ "input": input, "output": output, "interpolation": "LINEAR" }));
            out.push(json!({ "sampler": samplers.len() - 1, "target": { "node": c.node, "path": c.path } }));
        }
        vec![json!({ "channels": out, "samplers": samplers })]
    }

    /// Translation, rotation and scale channels for a node whose prim's transform is animated: its local transform
    /// at each sample, taken apart.
    fn transform_channels(&mut self, node: usize, prim: &usd::Prim, channels: &mut Vec<Channel>) {
        const PROBES: usize = 8;
        let Ok(q) = XformQuery::for_prim(prim) else { return };
        let mut times = timeline(q.time_samples().unwrap_or_default());
        if times.len() < 2 {
            return;
        }
        let at = |t: f64| {
            q.local_transformation(usd::TimeCode::new(t))
                .map_or(IDENTITY, |m| m4(&m))
        };
        // Cut long turns into short steps; probing within each span catches a full turn.
        let mut fine = Vec::with_capacity(times.len());
        for w in times.windows(2) {
            fine.push(w[0]);
            #[allow(clippy::cast_precision_loss, reason = "a small step count")]
            let probes: Vec<[f64; 4]> = (0..=PROBES)
                .map(|k| decompose(&at(w[0] + (w[1] - w[0]) * k as f64 / PROBES as f64)).r)
                .collect();
            let total: f64 = probes.windows(2).map(|p| turn(&p[0], &p[1])).sum();
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a small step count"
            )]
            let n = ((total / MAX_STEP_DEGREES).ceil() as usize).min(360);
            for k in 1..n {
                #[allow(clippy::cast_precision_loss, reason = "a small step count")]
                fine.push(w[0] + (w[1] - w[0]) * k as f64 / n as f64);
            }
        }
        fine.extend(times.last());
        times = fine;
        let mut trs: Vec<Trs> = times.iter().map(|&t| decompose(&at(t))).collect();
        // Keep each quaternion on the same side as the last, so slerp takes the short way.
        for i in 1..trs.len() {
            let prev = trs[i - 1].r;
            let r = &mut trs[i].r;
            if (0..4).map(|k| prev[k] * r[k]).sum::<f64>() < 0.0 {
                *r = r.map(|c| -c);
            }
        }
        let rest = decompose(&q.local_transformation(self.time).map_or(IDENTITY, |m| m4(&m)));
        let paths: [(&'static str, Vec<f64>, Vec<f64>); 3] = [
            ("translation", trs.iter().flat_map(|x| x.t).collect(), rest.t.to_vec()),
            ("rotation", trs.iter().flat_map(|x| x.r).collect(), rest.r.to_vec()),
            ("scale", trs.iter().flat_map(|x| x.s).collect(), rest.s.to_vec()),
        ];
        for (path, values, rest) in paths {
            let still = values
                .chunks_exact(rest.len())
                .all(|v| v.iter().zip(&rest).all(|(a, b)| (a - b).abs() < 1e-7));
            if !still {
                channels.push(Channel {
                    node,
                    path,
                    times: times.clone(),
                    values,
                });
            }
        }
        // An animated node holds TRS, never a matrix.
        if self.nodes[node]
            .as_object_mut()
            .and_then(|n| n.remove("matrix"))
            .is_some()
        {
            let n = &mut self.nodes[node];
            n["translation"] = json!(rest.t);
            n["rotation"] = json!(rest.r);
            n["scale"] = json!(rest.s);
        }
    }

    /// A weights channel for a node whose mesh's blend shape weights are animated.
    fn weight_channel(&self, node: usize, prim: &usd::Prim, channels: &mut Vec<Channel>) {
        let Some(anim) = self.blend_animation(prim) else { return };
        let times = timeline(
            anim.attribute("blendShapeWeights")
                .time_sample_times()
                .unwrap_or_default(),
        );
        if times.len() < 2 {
            return;
        }
        let mut values = Vec::new();
        for &t in &times {
            let Some(w) = self.weights_at(node, prim, usd::TimeCode::new(t)) else {
                return;
            };
            values.extend(w.into_iter().map(f64::from));
        }
        channels.push(Channel {
            node,
            path: "weights",
            times,
            values,
        });
    }
}

/// The first target of relationship `name` on `prim` or the nearest ancestor that has one: `UsdSkel`'s bindings are
/// inherited.
fn inherited_target(prim: &usd::Prim, name: &str) -> Option<sdf::Path> {
    let mut at = Some(prim.clone());
    while let Some(p) = at {
        if let Some(t) = p.relationship(name).targets().ok().and_then(|t| t.into_iter().next()) {
            return Some(t);
        }
        at = p.parent();
    }
    None
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

/// One material's part of a mesh, with the point each of its vertices came from.
struct Part_ {
    prim: Prim,
    points: Vec<usize>,
}

impl Reader<'_> {
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

    /// The mesh for a surface, made once per prototype, material binding and skinning; `None` when it has no
    /// faces. `joints` is how many joints a skinned surface's influences name.
    fn mesh(&mut self, prim: &usd::Prim, shape: &Shape, joints: Option<usize>) -> Result<Option<usize>> {
        let source = match prim.prim_in_prototype() {
            Ok(Some(p)) if prim.is_instance_proxy().unwrap_or(false) => p.path().as_str().to_owned(),
            _ => prim.path().as_str().to_owned(),
        };
        let binding = bound(prim).map(|b| b.as_str().to_owned()).unwrap_or_default();
        let key = format!("{source}|{binding}|{}", joints.is_some());
        if let Some(&m) = self.mesh_of.get(&key) {
            return Ok(m);
        }
        let m = self.build_mesh(prim, shape, joints)?;
        self.mesh_of.insert(key, m);
        Ok(m)
    }

    #[allow(clippy::too_many_lines, reason = "one pass over the faces, filling every attribute")]
    fn build_mesh(&mut self, prim: &usd::Prim, shape: &Shape, joints: Option<usize>) -> Result<Option<usize>> {
        let points = &shape.points;
        let flip = shape.left_handed;
        let (counts, corners) = (&shape.counts, &shape.corners);

        // The faces to draw, by their first corner.
        let mut faces = Vec::with_capacity(counts.len());
        let mut start = 0usize;
        let mut triangles = 0usize;
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
                triangles += n - 2;
                ensure!(triangles <= MAX_TRIANGLES, "too many triangles (over {MAX_TRIANGLES})");
            }
            start = end;
        }
        if faces.is_empty() {
            return Ok(None);
        }
        let point_normals = if shape.smooth {
            smooth_normals(points, corners, &faces, flip)
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

        let mut parts: Vec<Part_> = Vec::new();
        let mut part_of: HashMap<usize, usize> = HashMap::new();
        let mut tris = Vec::new();
        let mut poly = Vec::new();
        for &(face, start, n) in &faces {
            poly.clear();
            #[allow(clippy::cast_sign_loss, reason = "faces keep only corners in 0..points.len()")]
            poly.extend(corners[start..start + n].iter().map(|&c| points[c as usize]));
            let flat = (shape.normals.is_none() && !shape.smooth).then(|| polygon_normal(&poly, flip));
            let material = face_material[face];
            let next = parts.len();
            let i = *part_of.entry(material).or_insert(next);
            if i == next {
                parts.push(Part_ {
                    prim: Prim {
                        material,
                        ..Prim::default()
                    },
                    points: Vec::new(),
                });
            }
            let part = &mut parts[i];
            let prim = &mut part.prim;
            let first = u32::try_from(prim.positions.len()).context("too many vertices")?;
            for (c, &corner) in corners.iter().enumerate().skip(start).take(n) {
                #[allow(clippy::cast_sign_loss, reason = "faces keep only corners in 0..points.len()")]
                let point = corner as usize;
                prim.positions.push(points[point]);
                part.points.push(point);
                let normal = match (&shape.normals, flat) {
                    (Some(nv), _) => nv.at(face, point, c).map(|v| unit(v.map(f64::from), false)),
                    (None, Some(f)) => Some(f),
                    (None, None) => point_normals.get(point).copied(),
                };
                prim.normals.push(normal.unwrap_or([0.0, 1.0, 0.0]));
                if let Some(uv) = &shape.uvs {
                    let [u, v] = uv.at(face, point, c).unwrap_or_default();
                    prim.uvs.push([u, 1.0 - v]);
                }
                if vertex_colors {
                    let [r, g, b] = display.as_ref().and_then(|d| d.at(face, point, c)).unwrap_or([1.0; 3]);
                    let a = opacity.as_ref().and_then(|o| o.at(face, point, c)).unwrap_or(1.0);
                    prim.colors.push([r, g, b, a.clamp(0.0, 1.0)]);
                }
            }
            tris.clear();
            triangulate(&poly, &mut tris);
            for &[a, b, c] in &tris {
                let tri = if flip { [a, c, b] } else { [a, b, c] };
                prim.indices.extend(tri.map(|k| first + k));
            }
        }

        let influences = joints.and_then(|j| self.influences(prim, points.len(), j));
        let joint_sets = influences.as_ref().map_or(0, |inf| {
            inf.iter().map(|v| v.len().div_ceil(4)).max().unwrap_or(0).min(4)
        });
        let targets = if self.keep.morph_targets {
            self.blend_shapes(prim, points.len())
        } else {
            Vec::new()
        };
        let mut primitives = Vec::new();
        let mut drawn = 0;
        for Part_ { prim: p, points } in parts {
            let material = p.material;
            drawn += p.indices.len() / 3;
            let Some(mut json) = self.b.primitive(p) else { continue };
            json["material"] = json!(material);
            if let Some(inf) = &influences {
                for set in 0..joint_sets {
                    let mut js = Vec::with_capacity(points.len() * 4);
                    let mut ws = Vec::with_capacity(points.len() * 4);
                    for &pt in &points {
                        for k in set * 4..set * 4 + 4 {
                            let (j, w) = inf[pt].get(k).copied().unwrap_or((0, 0.0));
                            js.push(j);
                            ws.push(w);
                        }
                    }
                    let n = points.len();
                    let j = self.b.accessor(Part::u16(js), 5123, n, "VEC4", Some(34962));
                    let w = self.b.accessor(Part::f32(ws), 5126, n, "VEC4", Some(34962));
                    json["attributes"][format!("JOINTS_{set}")] = json!(j);
                    json["attributes"][format!("WEIGHTS_{set}")] = json!(w);
                }
            }
            if !targets.is_empty() {
                let t: Vec<Value> = targets
                    .iter()
                    .map(|(_, offsets)| {
                        let d = points.iter().map(|&pt| offsets[pt]).collect();
                        json!({ "POSITION": self.b.positions(d) })
                    })
                    .collect();
                json["targets"] = json!(t);
            }
            primitives.push(json);
        }
        if primitives.is_empty() {
            return Ok(None);
        }
        let mut mesh = json!({ "name": leaf(prim.path().as_str()), "primitives": primitives });
        if !targets.is_empty() {
            mesh["weights"] = json!(vec![0.0; targets.len()]);
            mesh["extras"] = json!({ "targetNames": targets.iter().map(|t| t.0.as_str()).collect::<Vec<_>>() });
        }
        self.meshes.push(mesh);
        self.mesh_triangles.push(drawn);
        Ok(Some(self.meshes.len() - 1))
    }

    /// A skinned surface's joint influences per point: (joint, weight), heaviest first, up to 16, summing to one.
    /// A point no joint weighs follows the first joint.
    fn influences(&self, prim: &usd::Prim, points: usize, joints: usize) -> Option<Vec<Vec<(u16, f32)>>> {
        let ji = prim.attribute("primvars:skel:jointIndices");
        let index = ints(self.value(&ji));
        let weight = floats(self.value(&prim.attribute("primvars:skel:jointWeights"))?)?;
        if index.is_empty() || index.len() != weight.len() {
            return None;
        }
        let size = match ji.get_metadata::<sdf::Value>("elementSize") {
            Ok(Some(sdf::Value::Int(n))) if n > 0 => usize::try_from(n).unwrap_or(1),
            _ => 1,
        };
        let constant = matches!(
            ji.get_metadata::<sdf::Value>("interpolation"),
            Ok(Some(v)) if v.as_str() == Some("constant")
        );
        let mut out = Vec::with_capacity(points);
        for p in 0..points {
            let base = if constant { 0 } else { p * size };
            let mut inf: Vec<(u16, f32)> = (base..base + size)
                .filter_map(|k| {
                    let j = usize::try_from(*index.get(k)?).ok().filter(|&j| j < joints)?;
                    let w = *weight.get(k)?;
                    (w > 0.0 && w.is_finite()).then(|| Some((u16::try_from(j).ok()?, w)))?
                })
                .collect();
            inf.sort_by(|a, b| b.1.total_cmp(&a.1));
            inf.truncate(16);
            let sum: f32 = inf.iter().map(|x| x.1).sum();
            if sum > 0.0 {
                for x in &mut inf {
                    x.1 /= sum;
                }
            } else {
                inf = vec![(0, 1.0)];
            }
            out.push(inf);
        }
        Some(out)
    }

    /// A surface's blend shapes: each one's name and its offset per point.
    fn blend_shapes(&self, prim: &usd::Prim, points: usize) -> Vec<(String, Vec<[f32; 3]>)> {
        let names = tokens(self.value(&prim.attribute("skel:blendShapes")));
        let targets = prim
            .relationship("skel:blendShapeTargets")
            .targets()
            .unwrap_or_default();
        names
            .into_iter()
            .zip(targets)
            .filter_map(|(name, path)| {
                let shape = self.stage.prim(path).ok()?;
                let offsets = vec3s(self.value(&shape.attribute("offsets"))?)?;
                let at = ints(self.value(&shape.attribute("pointIndices")));
                let mut d = vec![[0.0; 3]; points];
                if at.is_empty() {
                    for (slot, o) in d.iter_mut().zip(&offsets) {
                        *slot = *o;
                    }
                } else {
                    for (&i, o) in at.iter().zip(&offsets) {
                        if let Some(slot) = usize::try_from(i).ok().and_then(|i| d.get_mut(i)) {
                            *slot = *o;
                        }
                    }
                }
                Some((name, d))
            })
            .collect()
    }

    /// The material at `bound`, made once per material prim; with none, a plain one in `display`.
    fn material(&mut self, bound: Option<sdf::Path>, display: [f32; 4]) -> usize {
        let key = match &bound {
            Some(p) => p.as_str().to_owned(),
            None => format!("display:{display:?}"),
        };
        if let Some(&i) = self.material_of.get(&key) {
            return i;
        }
        let material = match bound {
            // Without materials kept, the shader isn't read; the plain one goes in the prune.
            Some(path) if !self.keep.materials => Material::plain(leaf(path.as_str()), display).into(),
            Some(path) => self.preview_surface(&path),
            None => Material::plain("default", display).into(),
        };
        let i = self.materials.len();
        self.materials.push(material);
        self.material_of.insert(key, i);
        i
    }

    #[allow(clippy::too_many_lines, reason = "one input after another")]
    fn preview_surface(&mut self, path: &sdf::Path) -> Full {
        let name = leaf(path.as_str()).to_owned();
        let Ok(Some(ps)) = shade::read_preview_surface(self.stage, path) else {
            return Material::plain(&name, GRAY).into();
        };
        let shader = self.stage.prim(ps.shader.as_str()).ok();
        let source = |input: &str| shader.as_ref().and_then(|s| self.texture_source(s, input));
        let (diffuse, normal, emissive, occlusion) = (
            source("diffuseColor"),
            source("normal"),
            source("emissiveColor"),
            source("occlusion"),
        );
        let (metallic_src, roughness_src, opacity_src) = (source("metallic"), source("roughness"), source("opacity"));
        let specular_src = source("specularColor");
        let specular_workflow = shader
            .as_ref()
            .and_then(|s| match self.value(&s.attribute("inputs:useSpecularWorkflow")) {
                Some(sdf::Value::Int(v)) => Some(v != 0),
                _ => None,
            })
            .unwrap_or(false);
        let mut tex = |file: Option<&str>, src: Option<&Source>| {
            let image = self.image(file?)?;
            Some(src.map_or_else(|| Tex::new(image), |s| s.tex(image)))
        };

        // UsdPreviewSurface colors are linear, as glTF's are. Unauthored inputs take the shader's fallbacks.
        let base_color_tex = tex(ps.diffuse_color.texture(), diffuse.as_ref());
        let mut color = match ps.diffuse_color.value() {
            Some(c) => [c.x, c.y, c.z, 1.0],
            // A texture's scale tints it.
            None if ps.diffuse_color.texture().is_some() => diffuse.as_ref().map_or([1.0; 4], |d| d.scale),
            None => [0.18, 0.18, 0.18, 1.0],
        };
        let mut alpha = Alpha::Opaque;
        if let Some(&o) = ps.opacity.value() {
            color[3] = o.clamp(0.0, 1.0);
            if o < 1.0 {
                alpha = Alpha::Blend;
            }
        } else if let (Some(o), Some(d)) = (&opacity_src, &diffuse) {
            // Opacity from the color texture's alpha is glTF's own arrangement.
            if o.file == d.file && o.channel == "a" {
                alpha = Alpha::Blend;
            }
        }
        if let Some(&t) = ps.opacity_threshold.value().filter(|t| **t > 0.0) {
            alpha = Alpha::Mask(t);
        }
        let mut metallic = if specular_workflow {
            0.0
        } else {
            ps.metallic.value().copied().unwrap_or(0.0)
        };
        let mut roughness = ps.roughness.value().copied().unwrap_or(0.5);
        // glTF reads roughness from green and metalness from blue of one image.
        let mut metallic_roughness_tex = None;
        if let Some(r) = roughness_src.as_ref().filter(|r| r.channel == "g")
            && let Some(file) = ps.roughness.texture()
        {
            let metal_too = metallic_src
                .as_ref()
                .is_some_and(|m| m.channel == "b" && m.file == r.file);
            if metal_too || ps.metallic.texture().is_none() {
                metallic_roughness_tex = tex(Some(file), Some(r));
                roughness = 1.0;
                if metal_too && !specular_workflow {
                    metallic = 1.0;
                }
            }
        }
        let emissive_tex = tex(ps.emissive_color.texture(), emissive.as_ref());
        let emissive_color = match ps.emissive_color.value() {
            Some(c) => [c.x, c.y, c.z],
            None if emissive_tex.is_some() => emissive
                .as_ref()
                .map_or([1.0; 3], |e| [e.scale[0], e.scale[1], e.scale[2]]),
            None => [0.0; 3],
        };
        let occlusion_tex = occlusion
            .as_ref()
            .filter(|o| o.channel == "r")
            .and_then(|o| tex(ps.occlusion.texture(), Some(o)));
        let specular = specular_workflow.then(|| {
            // UsdPreviewSurface's specular color is what reflects head on; glTF's scales a dielectric's 4%.
            let c = ps.specular_color.value().map_or([0.0; 3], |c| [c.x, c.y, c.z]);
            Specular {
                factor: Factor::of(1.0),
                color: if ps.specular_color.texture().is_some() {
                    [1.0 / 0.04; 3]
                } else {
                    c.map(|c| c / 0.04)
                },
                color_tex: tex(ps.specular_color.texture(), specular_src.as_ref()),
            }
        });
        let clearcoat = ps.clearcoat.value().filter(|c| **c > 0.0).map(|&c| Clearcoat {
            factor: Factor::of(c),
            roughness: Factor::of(ps.clearcoat_roughness.value().copied().unwrap_or(0.01)),
            normal: None,
        });
        Full {
            name,
            base_color: color,
            base_color_tex,
            metallic,
            roughness,
            metallic_roughness_tex,
            normal_tex: tex(ps.normal.texture(), normal.as_ref()),
            occlusion_tex,
            emissive: emissive_color,
            emissive_tex,
            alpha,
            specular,
            clearcoat,
            ior: ps.ior.value().copied(),
            ..Full::default()
        }
    }

    /// The `UsdUVTexture` an input of `shader` reads, through node graphs: the output it uses (`rgb`, `r`, ...),
    /// its wrap, its scale, and the `UsdTransform2d` on its coordinates.
    fn texture_source(&self, shader: &usd::Prim, input: &str) -> Option<Source> {
        let (tex, channel) = self.connected(shader, &format!("inputs:{input}"))?;
        let file = match self.value(&tex.attribute("inputs:file")) {
            Some(sdf::Value::AssetPath(a)) => Some(a.authored_path),
            _ => None,
        };
        let wrap = |k: &str| match self.token(&tex.attribute(k)).as_deref() {
            Some("mirror") => material::MIRROR,
            Some("clamp" | "black") => material::CLAMP,
            _ => material::REPEAT,
        };
        let scale = match self.value(&tex.attribute("inputs:scale")) {
            Some(sdf::Value::Vec4f(v)) => [v.x, v.y, v.z, v.w],
            #[allow(clippy::cast_possible_truncation, reason = "colors")]
            Some(sdf::Value::Vec4d(v)) => [v.x as f32, v.y as f32, v.z as f32, v.w as f32],
            _ => [1.0; 4],
        };
        let mut transform = None;
        if let Some((t, _)) = self.connected(&tex, "inputs:st")
            && self.token(&t.attribute("info:id")).as_deref() == Some("UsdTransform2d")
        {
            let [su, sv] = self.vec2(&t.attribute("inputs:scale")).unwrap_or([1.0; 2]);
            let [tu, tv] = self.vec2(&t.attribute("inputs:translation")).unwrap_or([0.0; 2]);
            let angle = self.number(&t.attribute("inputs:rotation"), 0.0).to_radians();
            if (su, sv, tu, tv, angle) != (1.0, 1.0, 0.0, 0.0, 0.0) {
                // USD's `st` has V up; the document's UVs are flipped.
                let (sin, cos) = angle.sin_cos();
                transform = Some(UvTransform {
                    offset: [tu - sv * sin, 1.0 - tv - sv * cos],
                    rotation: angle,
                    scale: [su, sv],
                });
            }
        }
        Some(Source {
            file,
            channel,
            wrap: [wrap("inputs:wrapS"), wrap("inputs:wrapT")],
            scale,
            transform,
        })
    }

    /// The shader and output an attribute connects to, followed through node graphs' outputs.
    fn connected(&self, prim: &usd::Prim, attr: &str) -> Option<(usd::Prim, String)> {
        let mut path = prim.attribute(attr).connections().ok()?.into_iter().next()?;
        for _ in 0..8 {
            let (at, prop) = path.as_str().rsplit_once('.')?;
            let p = self.stage.prim(at).ok()?;
            let next = p.attribute(prop).connections().ok().and_then(|c| c.into_iter().next());
            match next {
                Some(n) if prop.starts_with("outputs:") => path = n,
                _ => return Some((p, prop.strip_prefix("outputs:").unwrap_or(prop).to_owned())),
            }
        }
        None
    }

    fn vec2(&self, attr: &usd::Attribute) -> Option<[f32; 2]> {
        match self.value(attr)? {
            sdf::Value::Vec2f(v) => Some([v.x, v.y]),
            #[allow(clippy::cast_possible_truncation, reason = "texture coordinates")]
            sdf::Value::Vec2d(v) => Some([v.x as f32, v.y as f32]),
            _ => None,
        }
    }
}

/// A `UsdUVTexture` as an input reads it.
struct Source {
    file: Option<String>,
    /// The output read: `rgb`, `r`, `g`, `b` or `a`.
    channel: String,
    wrap: [u32; 2],
    /// `inputs:scale`, which tints what it reads.
    scale: [f32; 4],
    transform: Option<UvTransform>,
}

impl Source {
    fn tex(&self, image: usize) -> Tex {
        Tex {
            transform: self.transform,
            wrap: self.wrap,
            ..Tex::new(image)
        }
    }
}

/// A value as JSON, for extras: numbers, strings, tokens, asset paths, vectors, matrices, arrays of them and
/// dictionaries.
fn json_of(v: &sdf::Value) -> Option<Value> {
    use sdf::Value as V;
    let h = |x: gf::f16| x.to_f32();
    Some(match v {
        V::Bool(b) => json!(b),
        V::Uchar(x) => json!(x),
        V::Int(x) => json!(x),
        V::Uint(x) => json!(x),
        V::Int64(x) => json!(x),
        V::Uint64(x) => json!(x),
        V::Half(x) => json!(h(*x)),
        V::Float(x) => json!(x),
        V::Double(x) => json!(x),
        V::String(s) => json!(s),
        V::Token(t) => json!(t.as_str()),
        V::AssetPath(a) => json!(a.authored_path),
        V::BoolVec(v) => json!(v),
        V::UcharVec(v) => json!(v),
        V::IntVec(v) => json!(v),
        V::UintVec(v) => json!(v),
        V::Int64Vec(v) => json!(v),
        V::Uint64Vec(v) => json!(v),
        V::HalfVec(v) => json!(v.iter().map(|x| h(*x)).collect::<Vec<_>>()),
        V::FloatVec(v) => json!(v),
        V::DoubleVec(v) => json!(v),
        V::StringVec(v) => json!(v),
        V::TokenVec(v) => json!(v.iter().map(tf::Token::as_str).collect::<Vec<_>>()),
        V::AssetPathVec(v) => json!(v.iter().map(|a| a.authored_path.as_str()).collect::<Vec<_>>()),
        V::Vec2f(p) => json!([p.x, p.y]),
        V::Vec2d(p) => json!([p.x, p.y]),
        V::Vec2i(p) => json!([p.x, p.y]),
        V::Vec2h(p) => json!([h(p.x), h(p.y)]),
        V::Vec3f(p) => json!([p.x, p.y, p.z]),
        V::Vec3d(p) => json!([p.x, p.y, p.z]),
        V::Vec3i(p) => json!([p.x, p.y, p.z]),
        V::Vec3h(p) => json!([h(p.x), h(p.y), h(p.z)]),
        V::Vec4f(p) => json!([p.x, p.y, p.z, p.w]),
        V::Vec4d(p) => json!([p.x, p.y, p.z, p.w]),
        V::Vec4i(p) => json!([p.x, p.y, p.z, p.w]),
        V::Vec4h(p) => json!([h(p.x), h(p.y), h(p.z), h(p.w)]),
        V::Vec2fVec(v) => json!(v.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>()),
        V::Vec2dVec(v) => json!(v.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>()),
        V::Vec3fVec(v) => json!(v.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>()),
        V::Vec3dVec(v) => json!(v.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>()),
        V::Vec4fVec(v) => json!(v.iter().map(|p| [p.x, p.y, p.z, p.w]).collect::<Vec<_>>()),
        V::Vec4dVec(v) => json!(v.iter().map(|p| [p.x, p.y, p.z, p.w]).collect::<Vec<_>>()),
        V::Matrix4d(m) => json!(m.0),
        V::Dictionary(d) => {
            let mut keys: Vec<&String> = d.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .filter_map(|k| Some((k.clone(), json_of(&d[k])?)))
                    .collect(),
            )
        }
        _ => return None,
    })
}

/// Token values (a `token[]`, or `string[]`).
fn tokens(v: Option<sdf::Value>) -> Vec<String> {
    match v {
        Some(sdf::Value::TokenVec(v)) => v.iter().map(|t| t.as_str().to_owned()).collect(),
        Some(sdf::Value::StringVec(v)) => v,
        _ => Vec::new(),
    }
}

fn matrices(v: Option<sdf::Value>) -> Vec<gf::Matrix4d> {
    match v {
        Some(sdf::Value::Matrix4dVec(v)) => v,
        _ => Vec::new(),
    }
}

/// Two numbers, as a clipping range is.
fn pair(v: &sdf::Value) -> Option<[f64; 2]> {
    match v {
        sdf::Value::Vec2f(p) => Some([f64::from(p.x), f64::from(p.y)]),
        sdf::Value::Vec2d(p) => Some([p.x, p.y]),
        _ => None,
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
        // USD matrices are row vectors, so row `row` of the transform is column `row` of `r`.
        for (col, r_col) in r.iter().enumerate() {
            m.0[row * 4 + col] = f64::from(s[row]) * r_col[row];
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

impl Reader<'_> {
    /// A texture by its asset path: inside the `.usdz`, beside the file, or anywhere in the pack by name.
    fn image(&mut self, asset: &str) -> Option<usize> {
        if !(self.keep.materials && self.keep.textures) {
            return None;
        }
        if let Some(&i) = self.image_of.get(asset) {
            return i;
        }
        let reference = asset.replace("<UDIM>", "1001").replace("<udim>", "1001");
        let reference = reference.trim_start_matches("./");
        let assets = self.assets;
        let i = self
            .usdz
            .as_mut()
            .and_then(|z| z.read(reference))
            .and_then(|(entry, bytes)| {
                let convert = |b: &[u8]| assets.image(b, Path::new(reference));
                self.images
                    .bytes(bytes.into(), reference, Location::Archive(entry), convert)
            })
            .or_else(|| {
                let dir = self.file.parent().unwrap_or(Path::new(""));
                self.images.file(assets, assets.find(reference, dir)?, reference)
            });
        self.image_of.insert(asset.to_owned(), i);
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
    fn read(&mut self, reference: &str) -> Option<(String, Vec<u8>)> {
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
        Some((entry.clone(), bytes))
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
        format!("polysmelt-memory-{}", self.id)
    }

    fn cache_id(&self) -> Option<u64> {
        Some(self.id)
    }
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;
    use crate::document::Pose;

    fn read(path: &Path, dir: &Path) -> Option<crate::mesh::Mesh> {
        load(Input::file(path), &crate::assets::Folder::new(dir), Keep::ALL)
            .unwrap()
            .flatten(Pose::Rest)
            .unwrap()
    }

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
        let mesh = read(&path, dir.path()).unwrap();
        assert_eq!(mesh.prims.len(), 1, "the invisible mesh is left out");
        assert_eq!(mesh.triangles(), 2);
        let prim = &mesh.prims[0];
        // Translated 10 along X; Z up becomes Y up, so the quad in the XY plane lies in XZ.
        for (got, want) in [
            (prim.positions[1], [11.0, 0.0, 0.0]),
            (prim.positions[2], [11.0, 0.0, -1.0]),
        ] {
            assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-5), "{got:?}");
        }
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
        assert!(read(&path, dir.path()).is_none());
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
        let mesh = read(&path, dir.path()).unwrap();
        assert_eq!(mesh.materials[mesh.prims[0].material].color, [0.0, 0.5, 1.0, 1.0]);
        // Default units are centimeters.
        assert_eq!(mesh.prims[0].positions[1], [0.01, 0.0, 0.0]);
    }
}
