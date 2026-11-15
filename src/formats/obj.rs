//! OBJ, with its MTL materials and textures.

use std::collections::HashMap;
use std::path::Path;

use crate::assets::Assets;
use crate::mesh::{MAX_TRIANGLES, Material, Mesh, Prim, smooth_normals};
use crate::{Context as _, Result};

/// One face corner of an OBJ file: position, texture coordinate and normal indices (the last two optional).
type Corner = (usize, Option<usize>, Option<usize>);

/// One `newmtl` block of an MTL file.
#[derive(Debug, Default)]
struct MtlEntry {
    color: Option<[f32; 3]>,
    alpha: Option<f32>,
    color_map: Option<String>,
    normal_map: Option<String>,
}

/// An MTL file: material name → what it says. Texture options (`-bm 1.0`, `-s 1 1 1`) are skipped; the file name is
/// the last word.
fn read_mtl(path: &Path) -> HashMap<String, MtlEntry> {
    let mut out = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        return out;
    };
    let mut current: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        let (key, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let rest = rest.trim();
        if key.eq_ignore_ascii_case("newmtl") {
            current = Some(rest.to_owned());
            out.entry(rest.to_owned()).or_insert_with(MtlEntry::default);
            continue;
        }
        let Some(entry) = current.as_ref().and_then(|c| out.get_mut(c)) else {
            continue;
        };
        let floats: Vec<f32> = rest
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .collect();
        let file = || rest.split_whitespace().last().map(str::to_owned);
        match key.to_ascii_lowercase().as_str() {
            "kd" if floats.len() >= 3 => entry.color = Some([floats[0], floats[1], floats[2]]),
            "d" if !floats.is_empty() => entry.alpha = Some(floats[0]),
            "tr" if !floats.is_empty() => entry.alpha = Some(1.0 - floats[0]),
            "map_kd" => entry.color_map = file(),
            "map_bump" | "bump" | "norm" | "map_kn" => entry.normal_map = file(),
            _ => {}
        }
    }
    out
}

/// OBJ, with MTL colors and textures (base color and normal map). Faces may mix corners with and without texture
/// coordinates or normals; polygons are fanned into triangles.
pub(crate) fn obj(path: &Path, assets: &dyn Assets) -> Result<Mesh> {
    let text = std::fs::read_to_string(path).context("not a text OBJ file")?;
    let dir = path.parent().unwrap_or(Path::new(""));
    let mut v: Vec<[f32; 3]> = Vec::new();
    let mut vt: Vec<[f32; 2]> = Vec::new();
    let mut vn: Vec<[f32; 3]> = Vec::new();
    let mut mtl: HashMap<String, MtlEntry> = HashMap::new();
    // Faces grouped by material name, as corner triples (position, uv, normal), uv/normal may be missing.
    let mut groups: Vec<(String, Vec<[Corner; 3]>)> = Vec::new();
    let mut current = String::new();
    let resolve = |i: &str, len: usize| -> Option<usize> {
        let n: i64 = i.parse().ok()?;
        let idx = if n < 0 {
            i64::try_from(len).ok()? + n
        } else {
            n - 1
        };
        usize::try_from(idx).ok().filter(|&i| i < len)
    };
    let mut triangles = 0usize;
    for line in text.lines() {
        let mut words = line.split_whitespace();
        let Some(key) = words.next() else { continue };
        let nums = |w: std::str::SplitWhitespace<'_>| -> Vec<f32> {
            w.filter_map(|x| x.parse().ok()).collect()
        };
        match key {
            "v" => {
                let f = nums(words);
                if f.len() >= 3 {
                    v.push([f[0], f[1], f[2]]);
                }
            }
            "vt" => {
                let f = nums(words);
                if !f.is_empty() {
                    vt.push([f[0], f.get(1).copied().unwrap_or(0.0)]);
                }
            }
            "vn" => {
                let f = nums(words);
                if f.len() >= 3 {
                    vn.push([f[0], f[1], f[2]]);
                }
            }
            "mtllib" => {
                let name = line.trim_start()[6..].trim();
                if let Some(p) = assets.find(name, dir) {
                    mtl.extend(read_mtl(&p));
                }
            }
            "usemtl" => current = words.collect::<Vec<_>>().join(" "),
            "f" => {
                let corners: Vec<Corner> = words
                    .filter_map(|c| {
                        let mut parts = c.split('/');
                        let p = resolve(parts.next()?, v.len())?;
                        let t = parts
                            .next()
                            .filter(|s| !s.is_empty())
                            .and_then(|s| resolve(s, vt.len()));
                        let n = parts
                            .next()
                            .filter(|s| !s.is_empty())
                            .and_then(|s| resolve(s, vn.len()));
                        Some((p, t, n))
                    })
                    .collect();
                if corners.len() < 3 {
                    continue;
                }
                if groups.last().is_none_or(|(m, _)| *m != current) {
                    groups.push((current.clone(), Vec::new()));
                }
                let faces = &mut groups.last_mut().context("face group")?.1;
                for k in 1..corners.len() - 1 {
                    faces.push([corners[0], corners[k], corners[k + 1]]);
                }
                triangles += corners.len() - 2;
                ensure!(triangles <= MAX_TRIANGLES, "too many triangles to convert");
            }
            _ => {}
        }
    }
    ensure!(triangles > 0, "OBJ has no faces");

    let mut mesh = Mesh::default();
    let mut material_of: HashMap<String, usize> = HashMap::new();
    let mut image_of: HashMap<std::path::PathBuf, Option<usize>> = HashMap::new();
    for (name, faces) in groups {
        let material = if let Some(&m) = material_of.get(&name) {
            m
        } else {
            let entry = mtl.get(&name);
            let d = entry.and_then(|e| e.color).unwrap_or([0.8, 0.8, 0.8]);
            let mut mat = Material::plain(
                if name.is_empty() { "default" } else { &name },
                [d[0], d[1], d[2], entry.and_then(|e| e.alpha).unwrap_or(1.0)],
            );
            let mut texture = |reference: Option<&String>| -> Option<usize> {
                let found = assets.find(reference?, dir)?;
                *image_of.entry(found.clone()).or_insert_with(|| {
                    let img = assets.load_image(&found)?;
                    mesh.images.push(img);
                    Some(mesh.images.len() - 1)
                })
            };
            mat.color_image = texture(entry.and_then(|e| e.color_map.as_ref()));
            mat.normal_image = texture(entry.and_then(|e| e.normal_map.as_ref()));
            mesh.materials.push(mat);
            material_of.insert(name.clone(), mesh.materials.len() - 1);
            mesh.materials.len() - 1
        };
        let has_uvs = faces.iter().flatten().any(|c| c.1.is_some());
        let has_normals = faces.iter().flatten().all(|c| c.2.is_some());
        let mut prim = Prim {
            material,
            ..Prim::default()
        };
        let mut seen: HashMap<Corner, u32> = HashMap::new();
        for tri in &faces {
            for &corner in tri {
                let id = *seen.entry(corner).or_insert_with(|| {
                    prim.positions.push(v[corner.0]);
                    if has_normals {
                        prim.normals
                            .push(corner.2.map_or([0.0, 1.0, 0.0], |n| vn[n]));
                    }
                    if has_uvs {
                        // OBJ puts the texture origin bottom-left; glTF top-left.
                        let t = corner.1.map_or([0.0, 0.0], |t| vt[t]);
                        prim.uvs.push([t[0], 1.0 - t[1]]);
                    }
                    u32::try_from(prim.positions.len() - 1).unwrap_or(u32::MAX)
                });
                prim.indices.push(id);
            }
        }
        if !has_normals {
            prim.normals = smooth_normals(&prim.positions, &prim.indices);
        }
        mesh.prims.push(prim);
    }
    // A texture needs coordinates: drop textures from materials whose triangles have none.
    for (i, mat) in mesh.materials.iter_mut().enumerate() {
        if !mesh
            .prims
            .iter()
            .any(|p| p.material == i && !p.uvs.is_empty())
        {
            mat.color_image = None;
            mat.normal_image = None;
        }
    }
    Ok(mesh)
}
