//! PLY (the Stanford polygon file format), ASCII and binary in either byte order.
//!
//! Read: positions, normals, vertex colors (sRGB, as integers or floats), texture coordinates
//! under any of their usual names, polygons (triangulated in their plane) and triangle strips;
//! MeshLab's per-face texture coordinates and `comment TextureFile`; per-face colors. Elements
//! the reader doesn't use are skipped. A file without faces is a point cloud: no mesh, but its
//! point count.

use std::path::Path;

use serde_json::{Map, Value, json};

use crate::assets::Assets;
use crate::mesh::{
    FastMap, MAX_TRIANGLES, Material, Mesh, Prim, smooth_normals, srgb_to_linear, triangulate,
};
use crate::{Context as _, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Ascii,
    Le,
    Be,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &[u8]) -> Result<Ty> {
        Ok(match s {
            b"char" | b"int8" => Ty::I8,
            b"uchar" | b"uint8" => Ty::U8,
            b"short" | b"int16" => Ty::I16,
            b"ushort" | b"uint16" => Ty::U16,
            b"int" | b"int32" => Ty::I32,
            b"uint" | b"uint32" => Ty::U32,
            b"float" | b"float32" => Ty::F32,
            b"double" | b"float64" => Ty::F64,
            t => bail!("unknown PLY type {}", String::from_utf8_lossy(t)),
        })
    }

    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }

    /// The value that means full intensity for a color channel of this type.
    fn color_scale(self) -> f64 {
        match self {
            Ty::U8 | Ty::I8 => 255.0,
            Ty::U16 | Ty::I16 => 65535.0,
            _ => 1.0,
        }
    }
}

#[derive(Debug, Clone)]
struct Prop {
    name: String,
    ty: Ty,
    /// For a list: the type of its count.
    count: Option<Ty>,
}

#[derive(Debug, Clone)]
struct Element {
    name: String,
    count: usize,
    props: Vec<Prop>,
}

impl Element {
    fn slot(&self, names: &[&str]) -> Option<usize> {
        names.iter().find_map(|n| self.props.iter().position(|p| p.name == *n && p.count.is_none()))
    }

    fn list(&self, names: &[&str]) -> Option<usize> {
        names.iter().find_map(|n| self.props.iter().position(|p| p.name == *n && p.count.is_some()))
    }
}

/// Reads values in whichever encoding the file uses.
struct Reader<'a> {
    format: Format,
    data: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn value(&mut self, ty: Ty) -> Result<f64> {
        if self.format == Format::Ascii {
            let d = self.data;
            let mut i = self.pos;
            while i < d.len() && d[i].is_ascii_whitespace() {
                i += 1;
            }
            let start = i;
            while i < d.len() && !d[i].is_ascii_whitespace() {
                i += 1;
            }
            self.pos = i;
            ensure!(i > start, "PLY data ends early");
            return std::str::from_utf8(&d[start..i])
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .context("bad PLY number");
        }
        let n = ty.size();
        let b = self.data.get(self.pos..self.pos + n).context("PLY data ends early")?;
        self.pos += n;
        let mut a = [0u8; 8];
        a[..n].copy_from_slice(b);
        if self.format == Format::Be {
            a[..n].reverse();
        }
        Ok(match ty {
            Ty::I8 => f64::from(a[0] as i8),
            Ty::U8 => f64::from(a[0]),
            Ty::I16 => f64::from(i16::from_le_bytes([a[0], a[1]])),
            Ty::U16 => f64::from(u16::from_le_bytes([a[0], a[1]])),
            Ty::I32 => f64::from(i32::from_le_bytes([a[0], a[1], a[2], a[3]])),
            Ty::U32 => f64::from(u32::from_le_bytes([a[0], a[1], a[2], a[3]])),
            Ty::F32 => f64::from(f32::from_le_bytes([a[0], a[1], a[2], a[3]])),
            Ty::F64 => f64::from_le_bytes(a),
        })
    }
}

/// The header: format, elements, the texture file MeshLab names, and where the data starts.
fn header(data: &[u8]) -> Result<(Format, Vec<Element>, Option<String>, usize)> {
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    let mut texture = None;
    let mut at = 0;
    let mut first = true;
    loop {
        let rest = data.get(at..).filter(|r| !r.is_empty()).context("PLY header has no end_header")?;
        let len = rest.iter().position(|&c| c == b'\n').map_or(rest.len(), |n| n + 1);
        let line = rest[..len].trim_ascii();
        at += len;
        if first {
            ensure!(line == b"ply", "not a PLY file");
            first = false;
            continue;
        }
        let w: Vec<&[u8]> = line.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty()).collect();
        match w.as_slice() {
            [b"end_header", ..] => break,
            [b"format", f, ..] => {
                format = Some(match *f {
                    b"ascii" => Format::Ascii,
                    b"binary_little_endian" => Format::Le,
                    b"binary_big_endian" => Format::Be,
                    f => bail!("unknown PLY format {}", String::from_utf8_lossy(f)),
                });
            }
            [b"comment", kind, ..] if kind.eq_ignore_ascii_case(b"texturefile") => {
                let name = line[b"comment".len()..].trim_ascii_start()[kind.len()..].trim_ascii();
                texture = Some(String::from_utf8_lossy(name).into_owned());
            }
            [b"element", name, count] => elements.push(Element {
                name: String::from_utf8_lossy(name).into_owned(),
                count: std::str::from_utf8(count)
                    .ok()
                    .and_then(|c| c.parse().ok())
                    .context("bad PLY element count")?,
                props: Vec::new(),
            }),
            [b"property", b"list", c, i, name] => elements
                .last_mut()
                .context("PLY property before element")?
                .props
                .push(Prop {
                    name: String::from_utf8_lossy(name).into_owned(),
                    ty: Ty::parse(i)?,
                    count: Some(Ty::parse(c)?),
                }),
            [b"property", ty, name] => elements
                .last_mut()
                .context("PLY property before element")?
                .props
                .push(Prop {
                    name: String::from_utf8_lossy(name).into_owned(),
                    ty: Ty::parse(ty)?,
                    count: None,
                }),
            _ => {}
        }
    }
    Ok((format.context("PLY header names no format")?, elements, texture, at))
}

/// Vertex data, by vertex.
#[derive(Default)]
struct Vertices {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    col: Vec<[f32; 4]>,
    uv: Vec<[f32; 2]>,
}

/// Face data: each face's corners (vertex indices), and per corner the texture coordinate and
/// color when the file gives them per face.
#[derive(Default)]
struct Faces {
    starts: Vec<usize>,
    corners: Vec<u32>,
    corner_uv: Vec<[f32; 2]>,
    corner_col: Vec<[f32; 4]>,
    /// Triangle strips: their triangles, already split.
    strip_tris: Vec<[u32; 3]>,
}

const X: &[&str] = &["x"];
const Y: &[&str] = &["y"];
const Z: &[&str] = &["z"];
const U: &[&str] = &["u", "s", "texture_u", "texture_s", "tx"];
const V: &[&str] = &["v", "t", "texture_v", "texture_t", "ty"];
const R: &[&str] = &["red", "diffuse_red", "r"];
const G: &[&str] = &["green", "diffuse_green", "g"];
const B: &[&str] = &["blue", "diffuse_blue", "b"];
const A: &[&str] = &["alpha", "diffuse_alpha", "a"];

fn color(el: &Element, vals: &[f64], slots: &[usize; 3], alpha: Option<usize>) -> [f32; 4] {
    let ch = |s: usize| {
        let c = (vals[s] / el.props[s].ty.color_scale()).clamp(0.0, 1.0) as f32;
        srgb_to_linear(c)
    };
    let a = alpha.map_or(1.0, |s| (vals[s] / el.props[s].ty.color_scale()).clamp(0.0, 1.0) as f32);
    [ch(slots[0]), ch(slots[1]), ch(slots[2]), a]
}

pub(crate) fn ply(path: &Path, assets: &dyn Assets) -> Result<(Option<Mesh>, Map<String, Value>)> {
    let data = std::fs::read(path)?;
    let (format, elements, texture, start) = header(&data)?;
    let mut r = Reader { format, data: &data[start..], pos: 0 };
    let mut verts = Vertices::default();
    let mut faces = Faces::default();
    let mut vals: Vec<f64> = Vec::new();
    let mut list: Vec<f64> = Vec::new();
    let mut lists: Vec<Vec<f64>> = Vec::new();
    let mut has_faces = false;
    for el in &elements {
        // Every item takes at least a byte (binary) or two (ASCII, a value and a space).
        ensure!(el.count <= r.data.len(), "PLY element {} counts more items than the file holds", el.name);
        let is_vertex = el.name == "vertex";
        let is_face = el.name == "face";
        let is_strip = el.name == "tristrips";
        has_faces |= (is_face || is_strip) && el.count > 0;
        let (x, y, z) = (el.slot(X), el.slot(Y), el.slot(Z));
        let n = match (el.slot(&["nx"]), el.slot(&["ny"]), el.slot(&["nz"])) {
            (Some(a), Some(b), Some(c)) => Some([a, b, c]),
            _ => None,
        };
        let rgb = match (el.slot(R), el.slot(G), el.slot(B)) {
            (Some(a), Some(b), Some(c)) => Some([a, b, c]),
            _ => None,
        };
        let alpha = el.slot(A);
        let uv = el.slot(U).zip(el.slot(V));
        let index_list = el.list(&["vertex_indices", "vertex_index"]);
        let uv_list = el.list(&["texcoord"]);
        if is_vertex {
            ensure!(x.is_some() && y.is_some() && z.is_some(), "PLY vertices have no x, y and z");
            verts.pos.reserve(el.count);
        }
        lists.resize(el.props.len(), Vec::new());
        for _ in 0..el.count {
            vals.clear();
            for (k, p) in el.props.iter().enumerate() {
                match p.count {
                    None => vals.push(r.value(p.ty)?),
                    Some(cty) => {
                        let len = r.value(cty)?;
                        ensure!((0.0..=1e6).contains(&len), "bad PLY list length");
                        list.clear();
                        for _ in 0..len as usize {
                            list.push(r.value(p.ty)?);
                        }
                        vals.push(0.0);
                        std::mem::swap(&mut lists[k], &mut list);
                    }
                }
            }
            if is_vertex {
                let g = |s: Option<usize>| s.map_or(0.0, |s| vals[s] as f32);
                verts.pos.push([g(x), g(y), g(z)]);
                if let Some([a, b, c]) = n {
                    verts.nrm.push([vals[a], vals[b], vals[c]].map(|v| v as f32));
                }
                if let Some(s) = rgb {
                    verts.col.push(color(el, &vals, &s, alpha));
                }
                if let Some((u, v)) = uv {
                    verts.uv.push([vals[u] as f32, vals[v] as f32]);
                }
            } else if is_face {
                let Some(li) = index_list else { continue };
                let idx = &lists[li];
                faces.starts.push(faces.corners.len());
                faces.corners.extend(idx.iter().map(|&i| if i < 0.0 { u32::MAX } else { i as u32 }));
                if let Some(ui) = uv_list {
                    let t = &lists[ui];
                    for c in 0..idx.len() {
                        let get = |k: usize| t.get(k).copied().unwrap_or(0.0) as f32;
                        faces.corner_uv.push([get(2 * c), get(2 * c + 1)]);
                    }
                }
                if let Some(s) = rgb {
                    let c = color(el, &vals, &s, alpha);
                    faces.corner_col.extend(std::iter::repeat_n(c, idx.len()));
                }
            } else if is_strip {
                // Strips separated by -1; every other triangle's order flips to keep the winding.
                let Some(li) = index_list else { continue };
                let mut strip: Vec<u32> = Vec::new();
                for &i in lists[li].iter().chain([-1.0].iter()) {
                    if i < 0.0 {
                        for k in 2..strip.len() {
                            let (a, b, c) = (strip[k - 2], strip[k - 1], strip[k]);
                            faces.strip_tris.push(if k % 2 == 0 { [a, b, c] } else { [b, a, c] });
                        }
                        strip.clear();
                    } else {
                        strip.push(i as u32);
                    }
                }
            }
        }
    }
    let mut facts = Map::new();
    if !has_faces {
        facts.insert("points".into(), json!(verts.pos.len()));
        return Ok((None, facts));
    }
    let mesh = build(verts, faces, texture, path, assets)?;
    facts.insert("triangles".into(), json!(mesh.triangles()));
    facts.insert("vertices".into(), json!(mesh.prims.iter().map(|p| p.positions.len()).sum::<usize>()));
    Ok((Some(mesh), facts))
}

/// The primitive: shared vertices as the file has them, or split where faces give their corners
/// their own texture coordinates or colors.
fn build(
    verts: Vertices,
    faces: Faces,
    texture: Option<String>,
    path: &Path,
    assets: &dyn Assets,
) -> Result<Mesh> {
    let nv = verts.pos.len();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut local: Vec<[u32; 3]> = Vec::new();
    let mut pts: Vec<[f32; 3]> = Vec::new();
    // Triangles as corner numbers (into faces.corners), so per-corner data follows.
    let mut corner_tris: Vec<[usize; 3]> = Vec::new();
    for (f, &s) in faces.starts.iter().enumerate() {
        let end = faces.starts.get(f + 1).copied().unwrap_or(faces.corners.len());
        let face = &faces.corners[s..end];
        if face.len() < 3 || face.iter().any(|&i| i as usize >= nv) {
            continue;
        }
        if face.len() == 3 {
            corner_tris.push([s, s + 1, s + 2]);
        } else {
            pts.clear();
            pts.extend(face.iter().map(|&i| verts.pos[i as usize]));
            local.clear();
            triangulate(&pts, &mut local);
            corner_tris.extend(local.iter().map(|t| t.map(|k| s + k as usize)));
        }
        ensure!(corner_tris.len() <= MAX_TRIANGLES, "too many triangles to convert");
    }
    tris.extend(faces.strip_tris.iter().filter(|t| t.iter().all(|&i| (i as usize) < nv)));
    ensure!(corner_tris.len() + tris.len() <= MAX_TRIANGLES, "too many triangles to convert");
    ensure!(!corner_tris.is_empty() || !tris.is_empty(), "PLY has no faces");

    let per_corner = !faces.corner_uv.is_empty() || !faces.corner_col.is_empty();
    let has_uv = !faces.corner_uv.is_empty() || verts.uv.len() == nv && nv > 0;
    let has_col = !faces.corner_col.is_empty() || verts.col.len() == nv && nv > 0;
    let has_nrm = verts.nrm.len() == nv;
    let flip = |t: [f32; 2]| [t[0], 1.0 - t[1]];
    let mut p = Prim::default();
    if per_corner {
        // One output vertex per distinct (vertex, corner data).
        let mut seen: FastMap<(u32, [u32; 2], [u32; 4]), u32> = FastMap::default();
        let mut emit = |p: &mut Prim, v: u32, uv: Option<[f32; 2]>, col: Option<[f32; 4]>| -> u32 {
            let key = (v, uv.unwrap_or_default().map(f32::to_bits), col.unwrap_or_default().map(f32::to_bits));
            *seen.entry(key).or_insert_with(|| {
                p.positions.push(verts.pos[v as usize]);
                if has_nrm {
                    p.normals.push(verts.nrm[v as usize]);
                }
                if has_uv {
                    p.uvs.push(flip(uv.or_else(|| verts.uv.get(v as usize).copied()).unwrap_or_default()));
                }
                if has_col {
                    p.colors.push(col.or_else(|| verts.col.get(v as usize).copied()).unwrap_or([1.0; 4]));
                }
                (p.positions.len() - 1) as u32
            })
        };
        for t in &corner_tris {
            for &c in t {
                let id = emit(&mut p, faces.corners[c], faces.corner_uv.get(c).copied(), faces.corner_col.get(c).copied());
                p.indices.push(id);
            }
        }
        for t in &tris {
            for &v in t {
                let id = emit(&mut p, v, None, None);
                p.indices.push(id);
            }
        }
    } else {
        p.indices.reserve((corner_tris.len() + tris.len()) * 3);
        p.indices.extend(corner_tris.iter().flatten().map(|&c| faces.corners[c]));
        p.indices.extend(tris.iter().flatten());
        p.positions = verts.pos;
        if has_nrm {
            p.normals = verts.nrm;
        }
        if has_uv {
            p.uvs = verts.uv.into_iter().map(flip).collect();
        }
        if has_col {
            p.colors = verts.col;
        }
    }
    if has_nrm {
        for n in &mut p.normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            *n = if len > 0.0 && len.is_finite() { n.map(|c| c / len) } else { [0.0, 1.0, 0.0] };
        }
    } else {
        p.normals = smooth_normals(&p.positions, &p.indices);
    }

    let mut mesh = Mesh::default();
    // Vertex colors show as they are; otherwise the usual gray.
    let base = if has_col { [1.0; 4] } else { crate::mesh::GRAY };
    let mut mat = Material::plain("default", base);
    if has_uv {
        let dir = path.parent().unwrap_or(Path::new(""));
        if let Some(img) = texture.and_then(|t| assets.find(&t, dir)).and_then(|f| assets.load_image(&f)) {
            mesh.images.push(img);
            mat.color_image = Some(0);
            mat.color = [1.0; 4];
        }
    }
    mesh.materials.push(mat);
    mesh.prims.push(p);
    Ok(mesh)
}
