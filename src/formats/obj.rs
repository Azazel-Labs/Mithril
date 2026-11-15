//! OBJ, with its MTL materials and textures.
//!
//! Read as bytes: statements may continue over lines ending in a backslash, and text that isn't
//! UTF-8 (names from older tools) is read as Latin-1. Polygons are triangulated in their plane,
//! so concave faces stay inside their outline. Beyond the original format, vertex colors
//! (`v x y z r g b`) and the MTL PBR keys (`Pr`, `Pm`, `Ke`, `norm`) are read.

use std::collections::HashMap;
use std::path::Path;

use crate::Result;
use crate::assets::Assets;
use crate::mesh::{
    FastMap, MAX_TRIANGLES, Material, Mesh, Prim, smooth_normals, srgb_to_linear, triangulate,
};

/// Marks a corner without a texture coordinate or normal.
const NONE: u32 = u32::MAX;

/// One face corner: position, texture coordinate and normal indices (the last two may be
/// [`NONE`]).
type Corner = [u32; 3];

/// One `newmtl` block of an MTL file.
#[derive(Debug, Default)]
struct MtlEntry {
    color: Option<[f32; 3]>,
    alpha: Option<f32>,
    emissive: Option<[f32; 3]>,
    roughness: Option<f32>,
    metallic: Option<f32>,
    shininess: Option<f32>,
    color_map: Option<String>,
    normal_map: Option<String>,
}

/// Text as UTF-8 when it is, otherwise as Latin-1 (each byte one character).
fn text(b: &[u8]) -> String {
    match std::str::from_utf8(b) {
        Ok(s) => s.to_owned(),
        Err(_) => b.iter().map(|&c| char::from(c)).collect(),
    }
}

fn trim(b: &[u8]) -> &[u8] {
    b.trim_ascii()
}

/// Calls `f` with each logical line: `\r` trimmed, and lines ending in a backslash joined with
/// the next.
fn for_each_line(data: &[u8], mut f: impl FnMut(&[u8])) {
    let mut joined: Vec<u8> = Vec::new();
    for line in data.split(|&c| c == b'\n') {
        if let Some(head) = line.trim_ascii_end().strip_suffix(b"\\") {
            joined.extend_from_slice(head);
            joined.push(b' ');
            continue;
        }
        if joined.is_empty() {
            f(line);
        } else {
            joined.extend_from_slice(line);
            f(&joined);
            joined.clear();
        }
    }
    if !joined.is_empty() {
        f(&joined);
    }
}

/// A line's first word and the rest, trimmed.
fn split_key(line: &[u8]) -> (&[u8], &[u8]) {
    let line = line.trim_ascii_start();
    let end = line
        .iter()
        .position(u8::is_ascii_whitespace)
        .unwrap_or(line.len());
    (&line[..end], trim(&line[end..]))
}

fn words(b: &[u8]) -> impl Iterator<Item = &[u8]> {
    b.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty())
}

fn num(w: &[u8]) -> Option<f32> {
    std::str::from_utf8(w).ok()?.parse().ok()
}

/// Up to `N` leading numbers of `rest`, stopping at a comment; and how many there were.
fn nums<const N: usize>(rest: &[u8]) -> ([f32; N], usize) {
    let mut out = [0.0; N];
    let mut n = 0;
    for w in words(rest) {
        if n == N || w[0] == b'#' {
            break;
        }
        match num(w) {
            Some(x) => out[n] = x,
            None => break,
        }
        n += 1;
    }
    (out, n)
}

/// A texture statement's file name, after any options (`-s 1 1 1`, `-bm 0.5`, ...). The name
/// is the rest of the line, so it may contain spaces.
fn map_file(rest: &[u8]) -> Option<String> {
    let mut rest = trim(rest);
    while rest.first() == Some(&b'-') {
        let (opt, after) = split_key(rest);
        let args = match opt.to_ascii_lowercase().as_slice() {
            b"-blendu" | b"-blendv" | b"-bm" | b"-boost" | b"-cc" | b"-clamp" | b"-imfchan"
            | b"-texres" | b"-type" => 1,
            b"-mm" => 2,
            // One to three numbers.
            b"-o" | b"-s" | b"-t" => 3,
            _ => break,
        };
        rest = after;
        for k in 0..args {
            let (w, after) = split_key(rest);
            // Optional arguments end at the first word that isn't a number.
            if k > 0 && args == 3 && num(w).is_none() {
                break;
            }
            rest = after;
        }
    }
    (!rest.is_empty()).then(|| text(rest))
}

/// Adds an MTL file's materials to `out`.
fn read_mtl(data: &[u8], out: &mut HashMap<String, MtlEntry>) {
    let mut current: Option<String> = None;
    for_each_line(data, |line| {
        let (key, rest) = split_key(line);
        let key = key.to_ascii_lowercase();
        if key == b"newmtl" {
            let name = text(rest);
            out.entry(name.clone()).or_default();
            current = Some(name);
            return;
        }
        let Some(e) = current.as_ref().and_then(|c| out.get_mut(c)) else {
            return;
        };
        let (f, n) = nums::<3>(rest);
        let rgb = || (n >= 3).then_some(f).or((n == 1).then_some([f[0]; 3]));
        match key.as_slice() {
            b"kd" => e.color = rgb().or(e.color),
            b"ke" => e.emissive = rgb().or(e.emissive),
            b"d" if n > 0 => e.alpha = Some(f[0]),
            b"tr" if n > 0 => e.alpha = Some(1.0 - f[0]),
            b"pr" if n > 0 => e.roughness = Some(f[0]),
            b"pm" if n > 0 => e.metallic = Some(f[0]),
            b"ns" if n > 0 => e.shininess = Some(f[0]),
            b"map_kd" => e.color_map = map_file(rest),
            b"map_bump" | b"bump" | b"norm" | b"map_kn" => e.normal_map = map_file(rest),
            _ => {}
        }
    });
}

/// The material libraries `mtllib` names: the whole rest of the line as one file (names may
/// contain spaces), or failing that each word as a file.
fn read_libraries(
    libs: &[String],
    dir: &Path,
    assets: &dyn Assets,
) -> HashMap<String, MtlEntry> {
    let mut out = HashMap::new();
    for lib in libs {
        let found = match assets.find(lib, dir) {
            Some(p) => vec![p],
            None => lib
                .split_whitespace()
                .filter_map(|w| assets.find(w, dir))
                .collect(),
        };
        for p in found {
            if let Ok(data) = std::fs::read(&p) {
                read_mtl(&data, &mut out);
            }
        }
    }
    out
}

/// OBJ, with MTL colors and textures (base color and normal map). Faces may mix corners with
/// and without texture coordinates or normals.
pub(crate) fn obj(path: &Path, assets: &dyn Assets) -> Result<Mesh> {
    let data = std::fs::read(path)?;
    let dir = path.parent().unwrap_or(Path::new(""));
    let mut v: Vec<[f32; 3]> = Vec::new();
    // Vertex colors, one per position once any position has one.
    let mut colors: Vec<[f32; 3]> = Vec::new();
    let mut vt: Vec<[f32; 2]> = Vec::new();
    let mut vn: Vec<[f32; 3]> = Vec::new();
    let mut libs: Vec<String> = Vec::new();
    // Triangle corners by material name, in first-use order.
    let mut groups: Vec<(String, Vec<Corner>)> = Vec::new();
    let mut group_of: HashMap<String, usize> = HashMap::new();
    let mut current: Option<usize> = None;
    let mut face: Vec<Corner> = Vec::new();
    let mut pts: Vec<[f32; 3]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut triangles = 0usize;
    let resolve = |w: &[u8], len: usize| -> Option<u32> {
        let n: i64 = std::str::from_utf8(w).ok()?.parse().ok()?;
        let i = if n < 0 { len as i64 + n } else { n - 1 };
        (0..len as i64).contains(&i).then_some(i as u32)
    };
    for_each_line(&data, |line| {
        let (key, rest) = split_key(line);
        match key {
            b"v" => {
                let (f, n) = nums::<7>(rest);
                if n < 3 {
                    return;
                }
                v.push([f[0], f[1], f[2]]);
                if n >= 6 {
                    colors.resize(v.len() - 1, [1.0; 3]);
                    colors.push([f[3], f[4], f[5]]);
                } else if !colors.is_empty() {
                    colors.push([1.0; 3]);
                }
            }
            b"vt" => {
                let (f, n) = nums::<2>(rest);
                if n > 0 {
                    vt.push([f[0], f[1]]);
                }
            }
            b"vn" => {
                let (f, n) = nums::<3>(rest);
                if n == 3 {
                    vn.push(f);
                }
            }
            b"mtllib" => libs.push(text(rest)),
            b"usemtl" => {
                let name = text(rest);
                let next = groups.len();
                let g = *group_of.entry(name.clone()).or_insert(next);
                if g == next {
                    groups.push((name, Vec::new()));
                }
                current = Some(g);
            }
            b"f" | b"fo" => {
                face.clear();
                for w in words(rest) {
                    if w[0] == b'#' {
                        break;
                    }
                    let mut parts = w.split(|&c| c == b'/');
                    let p = parts.next().and_then(|p| resolve(p, v.len()));
                    let mut opt = |len: usize| match parts.next() {
                        None | Some(b"") => Some(NONE),
                        Some(s) => resolve(s, len),
                    };
                    let (t, n) = (opt(vt.len()), opt(vn.len()));
                    match (p, t, n) {
                        (Some(p), Some(t), Some(n)) => face.push([p, t, n]),
                        // A corner naming something that isn't there: drop the face.
                        _ => return,
                    }
                }
                if face.len() < 3 {
                    return;
                }
                let g = *current.get_or_insert_with(|| {
                    let next = groups.len();
                    *group_of.entry(String::new()).or_insert_with(|| {
                        groups.push((String::new(), Vec::new()));
                        next
                    })
                });
                let corners = &mut groups[g].1;
                if face.len() == 3 {
                    corners.extend_from_slice(&face);
                    triangles += 1;
                } else {
                    pts.clear();
                    pts.extend(face.iter().map(|c| v[c[0] as usize]));
                    tris.clear();
                    triangulate(&pts, &mut tris);
                    corners.extend(tris.iter().flatten().map(|&k| face[k as usize]));
                    triangles += tris.len();
                }
            }
            _ => {}
        }
    });
    ensure!(triangles > 0, "OBJ has no faces");
    ensure!(triangles <= MAX_TRIANGLES, "too many triangles to convert");

    for n in &mut vn {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        *n = if len > 0.0 {
            n.map(|c| c / len)
        } else {
            [0.0, 1.0, 0.0]
        };
    }
    // Vertex colors: 0-1 floats, or 0-255 from some tools; sRGB either way.
    let scale = if colors.iter().flatten().any(|&c| c > 1.0) {
        255.0
    } else {
        1.0
    };
    let colors: Vec<[f32; 4]> = colors
        .iter()
        .map(|c| {
            let [r, g, b] = c.map(|x| srgb_to_linear((x / scale).clamp(0.0, 1.0)));
            [r, g, b, 1.0]
        })
        .collect();

    let mtl = read_libraries(&libs, dir, assets);
    let mut mesh = Mesh::default();
    let mut image_of: HashMap<std::path::PathBuf, Option<usize>> = HashMap::new();
    for (name, corners) in groups {
        if corners.is_empty() {
            continue;
        }
        let entry = mtl.get(&name);
        // Without a material, vertex colors show as they are; otherwise the usual 0.8 gray.
        let fallback = if colors.is_empty() { [0.8; 3] } else { [1.0; 3] };
        let d = entry.and_then(|e| e.color).unwrap_or(fallback);
        let mut mat = Material::plain(
            if name.is_empty() { "default" } else { &name },
            [d[0], d[1], d[2], entry.and_then(|e| e.alpha).unwrap_or(1.0)],
        );
        if let Some(e) = entry {
            mat.emissive = e.emissive.unwrap_or([0.0; 3]);
            mat.metallic = e.metallic.unwrap_or(0.0);
            // Blender's reading of the specular exponent when there's no roughness.
            if let Some(r) = e
                .roughness
                .or(e.shininess.map(|ns| 1.0 - (ns.clamp(0.0, 1000.0) / 1000.0).sqrt()))
            {
                mat.roughness = r;
            }
        }
        let has_uvs = corners.iter().any(|c| c[1] != NONE);
        let mut texture = |reference: Option<&String>| -> Option<usize> {
            if !has_uvs {
                return None;
            }
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

        let has_normals = corners.iter().all(|c| c[2] != NONE);
        let has_colors = !colors.is_empty();
        let mut prim = Prim {
            material: mesh.materials.len() - 1,
            ..Prim::default()
        };
        let mut seen: FastMap<Corner, u32> = FastMap::default();
        prim.indices.reserve(corners.len());
        for c in corners {
            let id = *seen.entry(c).or_insert_with(|| {
                prim.positions.push(v[c[0] as usize]);
                if has_normals {
                    prim.normals.push(vn[c[2] as usize]);
                }
                if has_uvs {
                    // OBJ puts the texture origin bottom-left; glTF top-left.
                    let t = vt.get(c[1] as usize).copied().unwrap_or([0.0; 2]);
                    prim.uvs.push([t[0], 1.0 - t[1]]);
                }
                if has_colors {
                    prim.colors.push(colors[c[0] as usize]);
                }
                (prim.positions.len() - 1) as u32
            });
            prim.indices.push(id);
        }
        if !has_normals {
            prim.normals = smooth_normals(&prim.positions, &prim.indices);
        }
        mesh.prims.push(prim);
    }
    Ok(mesh)
}
