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
    pub indices: Vec<u32>,
    pub material: usize,
}

/// A material: a base color, optionally textured, and an optional normal map.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub color: [f32; 4],
    /// Index into `Mesh::images`.
    pub color_image: Option<usize>,
    pub normal_image: Option<usize>,
}

impl Material {
    /// A material of one color, without textures.
    pub fn plain(name: &str, color: [f32; 4]) -> Self {
        Self {
            name: name.to_owned(),
            color,
            color_image: None,
            normal_image: None,
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

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
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
            let mut pbr = json!({ "baseColorFactor": m.color, "metallicFactor": 0.0, "roughnessFactor": 0.85 });
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
