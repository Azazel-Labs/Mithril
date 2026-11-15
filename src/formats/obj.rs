//! OBJ, with its MTL materials and textures.
//!
//! Read as bytes, a block at a time: statements may continue over lines ending in a backslash, and text that isn't
//! UTF-8 (names from older tools) is read as Latin-1. Polygons are triangulated in their plane,
//! so concave faces stay inside their outline. Beyond the original format, vertex colors
//! (`v x y z r g b`) and the MTL PBR keys (`Pr`, `Pm`, `Ke`, `norm`) are read.

use std::collections::HashMap;
use std::path::Path;

use super::Input;
use super::stream;

use crate::Result;
use crate::assets::Assets;
use crate::mesh::{MAX_TRIANGLES, Material, Mesh, Prim, smooth_normals, srgb_to_linear, triangulate};
use crate::num::u32_of;

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
    let end = line.iter().position(u8::is_ascii_whitespace).unwrap_or(line.len());
    (&line[..end], trim(&line[end..]))
}

fn num(w: &[u8]) -> Option<f32> {
    stream::f32_from(w)
}

/// Up to `N` leading numbers of `rest`, stopping at a comment; and how many there were.
#[inline]
fn nums<const N: usize>(rest: &[u8]) -> ([f32; N], usize) {
    let mut out = [0.0; N];
    let mut n = 0;
    let mut i = 0;
    let len = rest.len();
    while n < N {
        while i < len && stream::space(rest[i]) {
            i += 1;
        }
        if i == len {
            break;
        }
        let start = i;
        while i < len && !stream::space(rest[i]) {
            i += 1;
        }
        let w = &rest[start..i];
        if w[0] == b'#' {
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
            b"-blendu" | b"-blendv" | b"-bm" | b"-boost" | b"-cc" | b"-clamp" | b"-imfchan" | b"-texres" | b"-type" => {
                1
            }
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
fn read_libraries(libs: &[String], dir: &Path, assets: &dyn Assets) -> HashMap<String, MtlEntry> {
    let mut out = HashMap::new();
    for lib in libs {
        let found = match assets.find(lib, dir) {
            Some(p) => vec![p],
            None => lib.split_whitespace().filter_map(|w| assets.find(w, dir)).collect(),
        };
        for p in found {
            if let Some(data) = assets.read(&p) {
                read_mtl(&data, &mut out);
            }
        }
    }
    out
}

/// Text up to and including the last line break that ends a statement (not one continued by a
/// backslash).
fn statements(d: &[u8]) -> Option<usize> {
    let mut end = d.len();
    loop {
        let i = d[..end].iter().rposition(|&b| b == b'\n')?;
        let line = &d[..i];
        let kept = line.iter().rposition(|&b| !matches!(b, b' ' | b'\t' | b'\r' | b'\x0c'));
        if kept.is_none_or(|k| line[k] != b'\\') {
            return Some(i + 1);
        }
        end = i;
    }
}

/// An index as written: an optional sign and up to 18 digits.
#[inline]
fn int(w: &[u8]) -> Option<i64> {
    let (neg, d) = match w.first()? {
        b'-' => (true, &w[1..]),
        b'+' => (false, &w[1..]),
        _ => (false, w),
    };
    if d.is_empty() || d.len() > 18 {
        return None;
    }
    let mut n: i64 = 0;
    for &c in d {
        let k = c.wrapping_sub(b'0');
        if k > 9 {
            return None;
        }
        n = n * 10 + i64::from(k);
    }
    Some(if neg { -n } else { n })
}

/// An OBJ index (1-based, or negative counting back from the last) into `len` items.
#[inline]
fn resolve(w: &[u8], len: usize) -> Option<u32> {
    let n = int(w)?;
    let len = i64::try_from(len).ok()?;
    let i = if n < 0 { len + n } else { n - 1 };
    if (0..len).contains(&i) {
        u32::try_from(i).ok().filter(|&i| i != NONE)
    } else {
        None
    }
}

/// One material's triangles, built as the faces are read.
struct Group {
    name: String,
    prim: Prim,
    /// Vertex colors as the file gives them, converted once the scale is known.
    raw_colors: Vec<[f32; 3]>,
    /// Whether every corner so far has a normal.
    all_normals: bool,
}

/// The statements read so far.
struct Reader {
    v: Vec<[f32; 3]>,
    /// Vertex colors, one per position once any position has one.
    colors: Vec<[f32; 3]>,
    vt: Vec<[f32; 2]>,
    vn: Vec<[f32; 3]>,
    /// Output vertices, found by position: the newest output vertex at each position, and for
    /// each output vertex the next at the same position, its group, texture coordinate and
    /// normal, and its index in its group.
    first: Vec<u32>,
    next: Vec<u32>,
    key: Vec<[u32; 3]>,
    local: Vec<u32>,
    /// Positions some output vertex uses.
    used: usize,
    libs: Vec<String>,
    /// By material name, in first-use order.
    groups: Vec<Group>,
    group_of: HashMap<String, usize>,
    current: Option<usize>,
    face: Vec<Corner>,
    pts: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    triangles: usize,
    /// Faces read, and the file offset of the first, for sizing the arrays.
    faces: usize,
    face_from: usize,
}

impl Reader {
    fn new() -> Self {
        Reader {
            v: Vec::new(),
            colors: Vec::new(),
            vt: Vec::new(),
            vn: Vec::new(),
            first: Vec::new(),
            next: Vec::new(),
            key: Vec::new(),
            local: Vec::new(),
            used: 0,
            libs: Vec::new(),
            groups: Vec::new(),
            group_of: HashMap::new(),
            current: None,
            face: Vec::new(),
            pts: Vec::new(),
            tris: Vec::new(),
            triangles: 0,
            faces: 0,
            face_from: 0,
        }
    }

    /// One statement, which starts `at` bytes into a file of `size`.
    #[inline]
    fn line(&mut self, line: &[u8], at: usize, size: usize) {
        let line = line.trim_ascii_start();
        let end = line.iter().position(|&b| stream::space(b)).unwrap_or(line.len());
        let (key, rest) = (&line[..end], &line[end..]);
        match key {
            b"v" => {
                let (f, n) = nums::<7>(rest);
                if n < 3 {
                    return;
                }
                self.v.push([f[0], f[1], f[2]]);
                self.first.push(NONE);
                if n >= 6 {
                    self.colors.resize(self.v.len() - 1, [1.0; 3]);
                    self.colors.push([f[3], f[4], f[5]]);
                } else if !self.colors.is_empty() {
                    self.colors.push([1.0; 3]);
                }
            }
            b"vt" => {
                let (f, n) = nums::<2>(rest);
                if n > 0 {
                    self.vt.push([f[0], f[1]]);
                }
            }
            b"vn" => {
                let (f, n) = nums::<3>(rest);
                if n == 3 {
                    let len = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
                    self.vn
                        .push(if len > 0.0 { f.map(|c| c / len) } else { [0.0, 1.0, 0.0] });
                }
            }
            b"mtllib" => self.libs.push(text(trim(rest))),
            b"usemtl" => {
                let name = text(trim(rest));
                let g = self.group(name);
                self.current = Some(g);
            }
            b"f" | b"fo" => self.face(rest, at, size),
            _ => {}
        }
    }

    fn group(&mut self, name: String) -> usize {
        let next = self.groups.len();
        *self.group_of.entry(name).or_insert_with_key(|name| {
            self.groups.push(Group {
                name: name.clone(),
                prim: Prim::default(),
                raw_colors: Vec::new(),
                all_normals: true,
            });
            next
        })
    }

    fn face(&mut self, rest: &[u8], at: usize, size: usize) {
        let mut face = std::mem::take(&mut self.face);
        face.clear();
        let mut ok = true;
        let mut i = 0;
        let n = rest.len();
        loop {
            while i < n && stream::space(rest[i]) {
                i += 1;
            }
            if i == n {
                break;
            }
            let start = i;
            while i < n && !stream::space(rest[i]) {
                i += 1;
            }
            let w = &rest[start..i];
            if w[0] == b'#' {
                break;
            }
            let mut parts = w.split(|&c| c == b'/');
            let p = parts.next().and_then(|p| resolve(p, self.v.len()));
            let mut opt = |len: usize| match parts.next() {
                None | Some(b"") => Some(NONE),
                Some(s) => resolve(s, len),
            };
            let (t, nm) = (opt(self.vt.len()), opt(self.vn.len()));
            let (Some(p), Some(t), Some(nm)) = (p, t, nm) else {
                // A corner naming something that isn't there: drop the face.
                ok = false;
                break;
            };
            face.push([p, t, nm]);
        }
        if ok && face.len() >= 3 {
            let g = if let Some(g) = self.current {
                g
            } else {
                let g = self.group(String::new());
                self.current = Some(g);
                g
            };
            if face.len() == 3 {
                for &c in &face {
                    let id = self.vertex(g, c);
                    self.groups[g].prim.indices.push(id);
                }
                self.triangles += 1;
            } else {
                let mut pts = std::mem::take(&mut self.pts);
                let mut tris = std::mem::take(&mut self.tris);
                pts.clear();
                pts.extend(face.iter().map(|c| self.v[c[0] as usize]));
                tris.clear();
                triangulate(&pts, &mut tris);
                for &k in tris.iter().flatten() {
                    let id = self.vertex(g, face[k as usize]);
                    self.groups[g].prim.indices.push(id);
                }
                self.triangles += tris.len();
                self.pts = pts;
                self.tris = tris;
            }
            self.faces += 1;
            if self.faces == 1 {
                self.face_from = at;
            } else if self.faces == 4096 {
                self.size(g, at, size);
            }
        }
        self.face = face;
    }

    /// Reserves the arrays once the faces so far say how big they'll get: bytes per face, and
    /// output vertices per position used.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a capacity guess from rates; MAX_TRIANGLES caps it"
    )]
    fn size(&mut self, g: usize, at: usize, size: usize) {
        let per_face = (at - self.face_from) as f64 / (self.faces - 1) as f64;
        if per_face <= 0.0 {
            return;
        }
        let more_faces = size.saturating_sub(at) as f64 / per_face;
        let tris = (more_faces * self.triangles as f64 / self.faces as f64 * 1.02) as usize + 64;
        let tris = tris.min(MAX_TRIANGLES);
        let prim = &mut self.groups[g].prim;
        prim.indices.reserve_exact(tris * 3);
        if self.groups.len() > 1 {
            return;
        }
        // One material so far: its vertices are about every position, times how many output
        // vertices each position has needed (texture seams, hard edges).
        let per_pos = self.key.len() as f64 / self.used.max(1) as f64;
        let total = (self.v.len() as f64 * per_pos * 1.02) as usize + 64;
        let more = total.saturating_sub(self.key.len());
        let grp = &mut self.groups[0];
        let prim = &mut grp.prim;
        prim.positions.reserve_exact(more);
        if !prim.uvs.is_empty() {
            prim.uvs.reserve_exact(more);
        }
        if grp.all_normals && !prim.normals.is_empty() {
            prim.normals.reserve_exact(more);
        }
        if !grp.raw_colors.is_empty() {
            grp.raw_colors.reserve_exact(more);
        }
        self.next.reserve_exact(more);
        self.key.reserve_exact(more);
        self.local.reserve_exact(more);
    }

    /// The group's vertex for a corner, added if it's new.
    #[inline]
    fn vertex(&mut self, g: usize, c: Corner) -> u32 {
        let [p, t, n] = c;
        let want = [u32_of(g), t, n];
        let mut id = self.first[p as usize];
        while id != NONE {
            if self.key[id as usize] == want {
                return self.local[id as usize];
            }
            id = self.next[id as usize];
        }
        let head = self.first[p as usize];
        if head == NONE {
            self.used += 1;
        }
        self.first[p as usize] = u32_of(self.key.len());
        self.key.push(want);
        self.next.push(head);
        let grp = &mut self.groups[g];
        let prim = &mut grp.prim;
        let k = prim.positions.len();
        self.local.push(u32_of(k));
        prim.positions.push(self.v[p as usize]);
        if t != NONE {
            if prim.uvs.len() < k {
                prim.uvs.resize(k, [0.0, 1.0]);
            }
            // OBJ puts the texture origin bottom-left; glTF top-left.
            let q = self.vt[t as usize];
            prim.uvs.push([q[0], 1.0 - q[1]]);
        } else if !prim.uvs.is_empty() {
            prim.uvs.push([0.0, 1.0]);
        }
        if grp.all_normals {
            if n == NONE {
                grp.all_normals = false;
                prim.normals = Vec::new();
            } else {
                prim.normals.push(self.vn[n as usize]);
            }
        }
        if !self.colors.is_empty() {
            // Positions before the first colored one are white.
            grp.raw_colors.resize(k, [1.0; 3]);
            grp.raw_colors.push(self.colors[p as usize]);
        }
        u32_of(k)
    }
}

fn trim_to_len<T>(v: &mut Vec<T>) {
    if v.capacity() > v.len() + v.len() / 16 + 16 {
        v.shrink_to_fit();
    }
}

/// OBJ, with MTL colors and textures (base color and normal map). Faces may mix corners with
/// and without texture coordinates or normals. The file is read a block at a time, and each
/// corner becomes a vertex of its material's primitive as it's read.
pub(crate) fn obj(input: Input<'_>, assets: &dyn Assets) -> Result<Mesh> {
    let mut blocks = input.blocks()?;
    let size = blocks.size_hint();
    let dir = input.dir();
    let mut r = Reader::new();
    // A statement continued over lines ending in a backslash.
    let mut joined: Vec<u8> = Vec::new();
    let mut read = 0;
    while let Some(chunk) = blocks.text(statements)? {
        let mut at = 0;
        while at < chunk.len() {
            let end = chunk[at..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(chunk.len(), |i| at + i);
            let line = &chunk[at..end];
            let start = read + at;
            at = end + 1;
            if let Some(head) = line.trim_ascii_end().strip_suffix(b"\\") {
                joined.extend_from_slice(head);
                joined.push(b' ');
                continue;
            }
            if joined.is_empty() {
                r.line(line, start, size);
            } else {
                joined.extend_from_slice(line);
                r.line(&joined, start, size);
                joined.clear();
            }
        }
        read += chunk.len();
        ensure!(r.triangles <= MAX_TRIANGLES, "too many triangles to convert");
    }
    if !joined.is_empty() {
        r.line(&joined, read, size);
    }
    ensure!(r.triangles > 0, "OBJ has no faces");
    ensure!(r.triangles <= MAX_TRIANGLES, "too many triangles to convert");

    let (colors, libs, groups) = (
        std::mem::take(&mut r.colors),
        std::mem::take(&mut r.libs),
        std::mem::take(&mut r.groups),
    );
    drop(r);
    // Vertex colors: 0-1 floats, or 0-255 from some tools; sRGB either way.
    let has_colors = !colors.is_empty();
    let scale = if colors.iter().flatten().any(|&c| c > 1.0) {
        255.0
    } else {
        1.0
    };
    drop(colors);

    let mtl = read_libraries(&libs, dir, assets);
    let mut mesh = Mesh::default();
    let mut image_of: HashMap<std::path::PathBuf, Option<usize>> = HashMap::new();
    for Group {
        name,
        mut prim,
        raw_colors,
        all_normals,
    } in groups
    {
        if prim.indices.is_empty() {
            continue;
        }
        let entry = mtl.get(&name);
        // Without a material, vertex colors show as they are; otherwise the usual 0.8 gray.
        let fallback = if has_colors { [1.0; 3] } else { [0.8; 3] };
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
        let nv = prim.positions.len();
        let has_uvs = !prim.uvs.is_empty();
        if has_uvs {
            prim.uvs.resize(nv, [0.0, 1.0]);
        }
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
        prim.material = mesh.materials.len() - 1;

        if has_colors {
            let mut raw = raw_colors;
            raw.resize(nv, [1.0; 3]);
            prim.colors = raw
                .iter()
                .map(|c| {
                    let [r, g, b] = c.map(|x| srgb_to_linear((x / scale).clamp(0.0, 1.0)));
                    [r, g, b, 1.0]
                })
                .collect();
        }
        trim_to_len(&mut prim.positions);
        trim_to_len(&mut prim.uvs);
        trim_to_len(&mut prim.indices);
        if all_normals && prim.normals.len() == nv {
            trim_to_len(&mut prim.normals);
        } else {
            prim.normals = smooth_normals(&prim.positions, &prim.indices);
        }
        mesh.prims.push(prim);
    }
    Ok(mesh)
}
