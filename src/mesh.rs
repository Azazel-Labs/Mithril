//! A model as plain triangle lists: what the format readers produce, written out as a GLB.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::Result;

/// The most triangles a reader accepts, against hostile or runaway files.
pub const MAX_TRIANGLES: usize = 4_000_000;

/// One material's worth of triangles.
#[derive(Debug, Default, Clone)]
pub struct Prim {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates, glTF convention (origin top-left); empty when the model has none.
    pub uvs: Vec<[f32; 2]>,
    /// Vertex colors, linear RGBA; empty when the model has none.
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub material: usize,
}

/// A material: a base color, optionally textured, an optional normal map, and the metallic-roughness scalars.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    /// Linear RGBA.
    pub color: [f32; 4],
    /// Index into `Mesh::images`.
    pub color_image: Option<usize>,
    pub normal_image: Option<usize>,
    /// Linear RGB light given off.
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    /// Seen from both sides rather than culled from behind.
    pub double_sided: bool,
}

impl Material {
    /// A material of one color, without textures.
    pub fn plain(name: &str, color: [f32; 4]) -> Self {
        Self {
            name: name.to_owned(),
            color,
            color_image: None,
            normal_image: None,
            emissive: [0.0; 3],
            metallic: 0.0,
            roughness: 0.85,
            double_sided: false,
        }
    }
}

/// An image ready to embed: PNG or JPEG bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    /// The encoded image.
    pub bytes: Vec<u8>,
    /// `image/png` or `image/jpeg`.
    pub mime: &'static str,
}

/// A model as triangles grouped by material, ready to write as glTF.
#[derive(Debug, Default, Clone)]
pub struct Mesh {
    pub prims: Vec<Prim>,
    pub materials: Vec<Material>,
    pub images: Vec<Image>,
}

impl Mesh {
    /// The number of triangles across all primitives.
    pub fn triangles(&self) -> usize {
        self.prims.iter().map(|p| p.indices.len() / 3).sum()
    }
}

/// The color of a material a format leaves unspecified.
pub const GRAY: [f32; 4] = [0.72, 0.72, 0.74, 1.0];

impl Image {
    /// `bytes` as an image when they are PNG or JPEG, the formats glTF embeds without an
    /// extension.
    pub fn png_or_jpeg(bytes: &[u8]) -> Option<Self> {
        let mime = if bytes.starts_with(b"\x89PNG") {
            "image/png"
        } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
            "image/jpeg"
        } else {
            return None;
        };
        Some(Self {
            bytes: bytes.to_vec(),
            mime,
        })
    }
}

/// An sRGB-encoded color channel in linear light.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Smooth normals from the triangles, for models that don't carry any.
pub fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut nrm = vec![[0.0f32; 3]; positions.len()];
    for t in indices.as_chunks::<3>().0 {
        let n = face_normal(
            positions[t[0] as usize],
            positions[t[1] as usize],
            positions[t[2] as usize],
        );
        for &i in t {
            for (a, b) in nrm[i as usize].iter_mut().zip(n) {
                *a += b;
            }
        }
    }
    for n in &mut nrm {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        *n = if len > 0.0 {
            [n[0] / len, n[1] / len, n[2] / len]
        } else {
            [0.0, 1.0, 0.0]
        };
    }
    nrm
}

/// One gray primitive.
// Shared by the format readers; which ones use it depends on the enabled features.
#[allow(dead_code)]
pub(crate) fn single(p: Prim) -> Mesh {
    Mesh {
        prims: vec![p],
        materials: vec![Material::plain("default", GRAY)],
        images: Vec::new(),
    }
}

/// A gray triangle soup.
// Shared by the format readers; which ones use it depends on the enabled features.
#[allow(dead_code)]
pub(crate) fn gray_soup(tris: &[[[f32; 3]; 3]]) -> Mesh {
    single(soup(tris))
}

pub(crate) fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// A triangle soup with flat normals.
// Shared by the format readers; which ones use it depends on the enabled features.
#[allow(dead_code)]
pub(crate) fn soup(tris: &[[[f32; 3]; 3]]) -> Prim {
    let mut p = Prim::default();
    for t in tris {
        let n = face_normal(t[0], t[1], t[2]);
        for v in t {
            p.indices
                .push(u32::try_from(p.positions.len()).unwrap_or(u32::MAX));
            p.positions.push(*v);
            p.normals.push(n);
        }
    }
    p
}

/// Facets grouped by color (gray for those without one): a primitive and plain material for each.
// Shared by the format readers; which ones use it depends on the enabled features.
#[allow(dead_code)]
pub(crate) fn colored_soup(tris: &[[[f32; 3]; 3]], colors: &[Option<[u8; 3]>]) -> Mesh {
    type Tri = [[f32; 3]; 3];
    let mut groups: Vec<(Option<[u8; 3]>, Vec<Tri>)> = Vec::new();
    let mut index: HashMap<Option<[u8; 3]>, usize> = HashMap::new();
    for (t, c) in tris.iter().zip(colors) {
        let i = *index.entry(*c).or_insert_with(|| {
            groups.push((*c, Vec::new()));
            groups.len() - 1
        });
        groups[i].1.push(*t);
    }
    let mut mesh = Mesh::default();
    for (i, (c, tris)) in groups.into_iter().enumerate() {
        let mut p = soup(&tris);
        p.material = i;
        mesh.prims.push(p);
        let color = c.map_or(GRAY, |c| {
            let [r, g, b] = c.map(|v| srgb_to_linear(f32::from(v) / 255.0));
            [r, g, b, 1.0]
        });
        mesh.materials
            .push(Material::plain(&format!("color_{i}"), color));
    }
    mesh
}

/// A GLB with one mesh: a primitive per `Prim`, its materials, and the images embedded.
pub fn to_glb(mesh: &Mesh) -> Result<Vec<u8>> {
    let mut bin: Vec<u8> = Vec::new();
    let mut views: Vec<Value> = Vec::new();
    let mut push_view = |bin: &mut Vec<u8>, bytes: &[u8], target: Option<u32>| -> usize {
        let offset = bin.len();
        bin.extend_from_slice(bytes);
        bin.resize(bin.len().div_ceil(4) * 4, 0);
        let mut v = json!({ "buffer": 0, "byteOffset": offset, "byteLength": bytes.len() });
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        views.push(v);
        views.len() - 1
    };
    let mut accessors = Vec::new();
    let mut primitives = Vec::new();
    for p in &mesh.prims {
        if p.indices.is_empty() {
            continue;
        }
        let f32s = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        let pv = push_view(&mut bin, &f32s(p.positions.as_flattened()), Some(34962));
        let nv = push_view(&mut bin, &f32s(p.normals.as_flattened()), Some(34962));
        let iv = push_view(
            &mut bin,
            &p.indices
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<u8>>(),
            Some(34963),
        );
        let (mut min, mut max) = ([f32::MAX; 3], [f32::MIN; 3]);
        for v in &p.positions {
            for k in 0..3 {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
            }
        }
        let a = accessors.len();
        accessors.push(json!({ "bufferView": pv, "componentType": 5126, "count": p.positions.len(), "type": "VEC3", "min": min, "max": max }));
        accessors.push(json!({ "bufferView": nv, "componentType": 5126, "count": p.normals.len(), "type": "VEC3" }));
        accessors.push(json!({ "bufferView": iv, "componentType": 5125, "count": p.indices.len(), "type": "SCALAR" }));
        let mut attributes = json!({ "POSITION": a, "NORMAL": a + 1 });
        if p.uvs.len() == p.positions.len() {
            let uv = push_view(&mut bin, &f32s(p.uvs.as_flattened()), Some(34962));
            accessors.push(json!({ "bufferView": uv, "componentType": 5126, "count": p.uvs.len(), "type": "VEC2" }));
            attributes["TEXCOORD_0"] = json!(accessors.len() - 1);
        }
        if p.colors.len() == p.positions.len() {
            let clamped: Vec<f32> = p.colors.as_flattened().iter().map(|c| c.clamp(0.0, 1.0)).collect();
            let cv = push_view(&mut bin, &f32s(&clamped), Some(34962));
            accessors.push(json!({ "bufferView": cv, "componentType": 5126, "count": p.colors.len(), "type": "VEC4" }));
            attributes["COLOR_0"] = json!(accessors.len() - 1);
        }
        primitives.push(json!({ "attributes": attributes, "indices": a + 2, "material": p.material.min(mesh.materials.len().saturating_sub(1)) }));
    }
    ensure!(!primitives.is_empty(), "nothing to draw");
    let mut images = Vec::new();
    for img in &mesh.images {
        let v = push_view(&mut bin, &img.bytes, None);
        images.push(json!({ "bufferView": v, "mimeType": img.mime }));
    }
    let materials: Vec<Value> = mesh
        .materials
        .iter()
        .map(|m| {
            let color = m.color.map(|c| c.clamp(0.0, 1.0));
            let mut pbr = json!({
                "baseColorFactor": color,
                "metallicFactor": m.metallic.clamp(0.0, 1.0),
                "roughnessFactor": m.roughness.clamp(0.0, 1.0),
            });
            if let Some(i) = m.color_image.filter(|&i| i < images.len()) {
                pbr["baseColorTexture"] = json!({ "index": i });
            }
            let mut mat = json!({ "name": m.name, "pbrMetallicRoughness": pbr });
            if let Some(i) = m.normal_image.filter(|&i| i < images.len()) {
                mat["normalTexture"] = json!({ "index": i });
            }
            if m.color[3] < 1.0 {
                mat["alphaMode"] = json!("BLEND");
            }
            if m.emissive.iter().any(|&e| e > 0.0) {
                mat["emissiveFactor"] = json!(m.emissive.map(|e| e.clamp(0.0, 1.0)));
            }
            if m.double_sided {
                mat["doubleSided"] = json!(true);
            }
            mat
        })
        .collect();
    let mut doc = json!({
        "asset": { "version": "2.0", "generator": "mithril" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{ "primitives": primitives }],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{ "byteLength": bin.len() }],
    });
    if !images.is_empty() {
        // One texture per image, all sharing a repeating, mipmapped sampler.
        doc["samplers"] =
            json!([{ "magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497 }]);
        doc["textures"] = json!(
            (0..images.len())
                .map(|i| json!({ "sampler": 0, "source": i }))
                .collect::<Vec<_>>()
        );
        doc["images"] = json!(images);
    }
    let mut json_bytes = serde_json::to_vec(&doc)?;
    json_bytes.resize(json_bytes.len().div_ceil(4) * 4, b' ');
    let total = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&u32::try_from(total)?.to_le_bytes());
    out.extend_from_slice(&u32::try_from(json_bytes.len())?.to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend(json_bytes);
    out.extend_from_slice(&u32::try_from(bin.len())?.to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend(bin);
    Ok(out)
}

/// Triangulates one polygon face, its corners in order, appending triangles as indices into
/// `corners` that keep the polygon's winding. Convex polygons (nearly every face) are fanned;
/// concave ones are ear-clipped in the polygon's plane, so no triangle falls outside the
/// outline.
// Shared by the format readers; which ones use it depends on the enabled features.
#[allow(dead_code)]
pub(crate) fn triangulate(corners: &[[f32; 3]], out: &mut Vec<[u32; 3]>) {
    let n = corners.len();
    let fan = |out: &mut Vec<[u32; 3]>, idx: &[u32]| {
        for k in 1..idx.len().saturating_sub(1) {
            out.push([idx[0], idx[k], idx[k + 1]]);
        }
    };
    if n < 3 {
        return;
    }
    if n == 3 {
        out.push([0, 1, 2]);
        return;
    }
    // The plane's normal (Newell), then 2D coordinates along its two other axes.
    let mut nrm = [0f64; 3];
    for (i, a) in corners.iter().enumerate() {
        let b = corners[(i + 1) % n];
        let (a, b) = (a.map(f64::from), b.map(f64::from));
        nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
        nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
        nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let axis = (0..3)
        .max_by(|&i, &j| nrm[i].abs().total_cmp(&nrm[j].abs()))
        .unwrap_or(2);
    let all: Vec<u32> = (0..n as u32).collect();
    if nrm[axis] == 0.0 || n > 4096 {
        fan(out, &all);
        return;
    }
    // Project so the polygon winds counter-clockwise in 2D.
    let (u, v) = match axis {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let flip = nrm[axis] < 0.0;
    let p: Vec<[f64; 2]> = corners
        .iter()
        .map(|c| {
            let (x, y) = (f64::from(c[u]), f64::from(c[v]));
            if flip { [y, x] } else { [x, y] }
        })
        .collect();
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    if (0..n).all(|i| cross(p[i], p[(i + 1) % n], p[(i + 2) % n]) >= 0.0) {
        fan(out, &all);
        return;
    }
    // Ear clipping: cut off a convex corner whose triangle holds no other corner.
    let mut idx = all;
    while idx.len() > 3 {
        let m = idx.len();
        let ear = (0..m).find(|&i| {
            let (a, b, c) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            let (pa, pb, pc) = (p[a as usize], p[b as usize], p[c as usize]);
            if cross(pa, pb, pc) <= 0.0 {
                return false;
            }
            !idx.iter().any(|&q| {
                if q == a || q == b || q == c {
                    return false;
                }
                let pq = p[q as usize];
                cross(pa, pb, pq) >= 0.0 && cross(pb, pc, pq) >= 0.0 && cross(pc, pa, pq) >= 0.0
            })
        });
        let Some(i) = ear else {
            // Self-intersecting or degenerate: nothing better to do than fan what's left.
            break;
        };
        out.push([idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]]);
        idx.remove(i);
    }
    fan(out, &idx);
}

#[cfg(test)]
mod triangulate_tests {
    use super::triangulate;

    fn area(c: &[[f32; 3]], t: &[[u32; 3]]) -> f32 {
        t.iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| c[i as usize]);
                ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
            })
            .sum()
    }

    #[test]
    fn concave_and_convex() {
        let ell = [[2., 1., 0.], [1., 1., 0.], [1., 2., 0.], [0., 2., 0.], [0., 0., 0.], [2., 0., 0.]];
        let mut t = Vec::new();
        triangulate(&ell, &mut t);
        assert_eq!(t.len(), 4);
        // Signed: every triangle counter-clockwise like the polygon, and they tile it exactly.
        assert!(t.iter().all(|t| area(&ell, &[*t]) > 0.0));
        assert_eq!(area(&ell, &t), 3.0);
        // Clockwise (seen from +z) stays clockwise.
        let rev: Vec<[f32; 3]> = ell.iter().rev().copied().collect();
        t.clear();
        triangulate(&rev, &mut t);
        assert_eq!(area(&rev, &t), -3.0);
        let quad = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
        t.clear();
        triangulate(&quad, &mut t);
        assert_eq!(t, [[0, 1, 2], [0, 2, 3]]);
    }
}

/// A multiply-rotate hasher (the Fx hash) for small integer keys: much faster than the
/// default SipHash, and keys here come from the file being read, not from an attacker who can
/// see the hash.
#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher(u64);

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(b));
        }
    }
    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.write_u64(n as u64);
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A `HashMap` with [`FxHasher`].
pub(crate) type FastMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<FxHasher>>;
