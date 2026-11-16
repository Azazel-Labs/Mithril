//! OBJ, with its MTL materials and textures.
//!
//! Read as bytes, a block at a time: statements may continue over lines ending in a backslash, and text that isn't
//! UTF-8 (names from older tools) is read as Latin-1. Polygons are triangulated in their plane,
//! so concave faces stay inside their outline. Beyond the original format, vertex colors
//! (`v x y z r g b`) and the MTL PBR keys (`Pr`, `Pm`, `Ke`, `norm`) are read.
//!
//! Each object (`o`) or group (`g`) is a node drawing its own mesh, with a primitive per material
//! and its points (`p`) and lines (`l`) as primitives of their own. Every MTL statement goes in
//! its material's extras; the file's opening comments, its libraries and the statements not
//! converted (free-form curves and surfaces) go in the asset's.

use std::collections::HashMap;
use std::path::Path;

use serde_json::{Map, Value, json};

use super::Input;
use super::stream;

use crate::Result;
use crate::assets::Assets;
use crate::document::{Builder, Document, Keep, Part};
use crate::images::Images;
use crate::material::{self, Clearcoat, Factor, Full, Sheen, Specular, Tex, UvTransform};
use crate::mesh::{MAX_TRIANGLES, Material, Prim, smooth_normals, srgb_to_linear, triangulate};
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
    color_map: Option<MapRef>,
    normal_map: Option<MapRef>,
    /// `map_Ke`, `map_Ks`, `map_Ps`, and `map_Pr` and `map_Pm` (glTF's metal-roughness texture when one file).
    emissive_map: Option<MapRef>,
    specular_map: Option<MapRef>,
    sheen_map: Option<MapRef>,
    roughness_map: Option<MapRef>,
    metallic_map: Option<MapRef>,
    /// `Ks`, Phong's highlight color.
    specular: Option<[f32; 3]>,
    /// `Ni`.
    ior: Option<f32>,
    /// `Ps`, `Pc`, `Pcr` and `aniso`: the PBR extension's sheen, clearcoat and anisotropy.
    sheen: Option<[f32; 3]>,
    clearcoat: Option<f32>,
    clearcoat_roughness: Option<f32>,
    anisotropy: Option<f32>,
    /// `illum`.
    illum: Option<f32>,
    /// Every statement as written: numbers as numbers, the rest as text.
    raw: Map<String, Value>,
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

/// A texture statement: its file, and the options glTF has a place for.
#[derive(Debug, Clone, PartialEq)]
struct MapRef {
    file: String,
    /// `-o` and `-s`: `uv' = uv * scale + offset`, V up.
    offset: [f32; 2],
    scale: [f32; 2],
    /// `-clamp on`.
    clamp: bool,
    /// `-bm`: a bump or normal map's strength.
    bump: f32,
}

impl MapRef {
    /// The texture of `image` as this statement lays it out, in glTF's V-down UVs.
    fn tex(&self, image: usize) -> Tex {
        let mut t = Tex::new(image);
        let [su, sv] = self.scale;
        let [ou, ov] = self.offset;
        #[allow(clippy::float_cmp, reason = "the identity is written exactly")]
        let moved = self.scale != [1.0; 2] || self.offset != [0.0; 2];
        if moved {
            t.transform = Some(UvTransform {
                offset: [ou, 1.0 - sv - ov],
                rotation: 0.0,
                scale: [su, sv],
            });
        }
        if self.clamp {
            t.wrap = [material::CLAMP; 2];
        }
        t.scale = self.bump;
        t
    }
}

/// A texture statement: the file name after any options (`-s 1 1 1`, `-bm 0.5`, ...), and the options' values.
fn map_ref(rest: &[u8]) -> Option<MapRef> {
    let mut m = MapRef {
        file: String::new(),
        offset: [0.0; 2],
        scale: [1.0; 2],
        clamp: false,
        bump: 1.0,
    };
    let mut rest = trim(rest);
    while rest.first() == Some(&b'-') {
        let (opt, after) = split_key(rest);
        let opt = opt.to_ascii_lowercase();
        let args = match opt.as_slice() {
            b"-blendu" | b"-blendv" | b"-bm" | b"-boost" | b"-cc" | b"-clamp" | b"-imfchan" | b"-texres" | b"-type" => {
                1
            }
            b"-mm" => 2,
            // One to three numbers.
            b"-o" | b"-s" | b"-t" => 3,
            _ => break,
        };
        rest = after;
        let mut values = Vec::new();
        for k in 0..args {
            let (w, after) = split_key(rest);
            // Optional arguments end at the first word that isn't a number.
            if k > 0 && args == 3 && num(w).is_none() {
                break;
            }
            values.push(w);
            rest = after;
        }
        let n = |i: usize| values.get(i).and_then(|w| num(w)).filter(|x| x.is_finite());
        match opt.as_slice() {
            b"-o" => m.offset = [n(0).unwrap_or(0.0), n(1).unwrap_or(0.0)],
            b"-s" => m.scale = [n(0).unwrap_or(1.0), n(1).unwrap_or(1.0)],
            b"-clamp" => m.clamp = values.first().is_some_and(|w| w.eq_ignore_ascii_case(b"on")),
            b"-bm" => m.bump = n(0).unwrap_or(1.0),
            _ => {}
        }
    }
    m.file = text(rest);
    (!rest.is_empty()).then_some(m)
}

/// What a material shows beyond its flat view, from the MTL statements glTF has a place for.
#[derive(Default)]
struct Look {
    color: Option<Tex>,
    normal: Option<Tex>,
    metal_rough: Option<Tex>,
    emissive: Option<Tex>,
    specular: Option<Specular>,
    ior: Option<f32>,
    sheen: Option<Sheen>,
    clearcoat: Option<Clearcoat>,
    anisotropy: Option<f32>,
    unlit: bool,
}

impl Look {
    fn full(&self, m: &Material) -> Full {
        let mut f = Full::from(m);
        // `guess_texture` may have found a color texture the MTL file doesn't name.
        if let (Some(t), Some(c)) = (&mut f.base_color_tex, &self.color) {
            *t = Tex {
                image: t.image,
                ..c.clone()
            };
        }
        if let (Some(t), Some(n)) = (&mut f.normal_tex, &self.normal) {
            *t = Tex {
                image: t.image,
                ..n.clone()
            };
        }
        if let Some(t) = &self.metal_rough {
            f.metallic_roughness_tex = Some(t.clone());
            // The texture is the value, unscaled, unless the file says otherwise.
            if m.metallic <= 0.0 {
                f.metallic = 1.0;
            }
        }
        f.emissive_tex.clone_from(&self.emissive);
        f.specular.clone_from(&self.specular);
        f.ior = self.ior;
        f.sheen.clone_from(&self.sheen);
        f.clearcoat.clone_from(&self.clearcoat);
        f.anisotropy = self.anisotropy.map(|a| material::Anisotropy {
            strength: Factor::of(a),
            rotation: 0.0,
        });
        f.unlit = self.unlit;
        f
    }
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
        let (written, _) = split_key(line);
        if !written.is_empty() && written[0] != b'#' {
            e.raw.insert(text(written), raw_value(rest));
        }
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
            b"ks" => e.specular = rgb().or(e.specular),
            b"ni" if n > 0 => e.ior = Some(f[0]),
            b"ps" => e.sheen = rgb().or(e.sheen),
            b"pc" if n > 0 => e.clearcoat = Some(f[0]),
            b"pcr" if n > 0 => e.clearcoat_roughness = Some(f[0]),
            b"aniso" if n > 0 => e.anisotropy = Some(f[0]),
            b"illum" if n > 0 => e.illum = Some(f[0]),
            b"map_kd" => e.color_map = map_ref(rest),
            b"map_bump" | b"bump" | b"norm" | b"map_kn" => e.normal_map = map_ref(rest),
            b"map_ke" => e.emissive_map = map_ref(rest),
            b"map_ks" => e.specular_map = map_ref(rest),
            b"map_ps" => e.sheen_map = map_ref(rest),
            b"map_pr" => e.roughness_map = map_ref(rest),
            b"map_pm" => e.metallic_map = map_ref(rest),
            _ => {}
        }
    });
}

/// A statement's arguments: a number, a list of numbers, or failing that the text.
fn raw_value(rest: &[u8]) -> Value {
    let words: Vec<&[u8]> = rest.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty()).collect();
    let numbers: Option<Vec<f64>> = words
        .iter()
        .map(|w| {
            std::str::from_utf8(w)
                .ok()?
                .parse()
                .ok()
                .filter(|x: &f64| x.is_finite())
        })
        .collect();
    match numbers {
        Some(n) if n.len() == 1 => json!(n[0]),
        Some(n) if !n.is_empty() => json!(n),
        _ => json!(text(rest)),
    }
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

/// One part's triangles in one material, built as the faces are read, and its points and lines.
struct Group {
    part: usize,
    material: usize,
    prim: Prim,
    /// Vertex colors as the file gives them, converted once the scale is known.
    raw_colors: Vec<[f32; 3]>,
    /// Whether every corner so far has a normal.
    all_normals: bool,
    /// Positions (file indices) drawn as points, and in pairs as lines.
    points: Vec<u32>,
    lines: Vec<u32>,
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
    /// Material names in first-use order (`""` for none), and the current one.
    materials: Vec<String>,
    material_of: HashMap<String, usize>,
    material: usize,
    /// Parts (an object's and a group's name) in first-use order, the current names, and the
    /// current part once a face needs it.
    parts: Vec<(Option<String>, Option<String>)>,
    part_of: HashMap<(Option<String>, Option<String>), usize>,
    object: Option<String>,
    group: Option<String>,
    part: Option<usize>,
    /// By part and material, in first-use order.
    groups: Vec<Group>,
    group_of: HashMap<(usize, usize), usize>,
    current: Option<usize>,
    face: Vec<Corner>,
    pts: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    triangles: usize,
    points: usize,
    lines: usize,
    /// Faces read, and the file offset of the first, for sizing the arrays.
    faces: usize,
    face_from: usize,
    /// The comments before the first statement.
    comments: Vec<String>,
    /// Statements read but not converted, by keyword.
    skipped: Vec<(Vec<u8>, usize)>,
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
            materials: vec![String::new()],
            material_of: HashMap::from([(String::new(), 0)]),
            material: 0,
            parts: Vec::new(),
            part_of: HashMap::new(),
            object: None,
            group: None,
            part: None,
            groups: Vec::new(),
            group_of: HashMap::new(),
            current: None,
            face: Vec::new(),
            pts: Vec::new(),
            tris: Vec::new(),
            triangles: 0,
            points: 0,
            lines: 0,
            faces: 0,
            face_from: 0,
            comments: Vec::new(),
            skipped: Vec::new(),
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
            b"f" | b"fo" => self.face(rest, at, size),
            b"s" => {}
            b"mtllib" => self.libs.push(text(trim(rest))),
            b"usemtl" => {
                let next = self.materials.len();
                self.material = *self.material_of.entry(text(trim(rest))).or_insert_with_key(|name| {
                    self.materials.push(name.clone());
                    next
                });
                self.current = None;
            }
            b"o" | b"g" => {
                let name = text(trim(rest));
                let name = (!name.is_empty()).then_some(name);
                if key == b"o" {
                    self.object = name;
                } else {
                    self.group = name;
                }
                (self.part, self.current) = (None, None);
            }
            b"l" | b"p" => self.loose(key == b"l", rest),
            _ => self.other(key, line),
        }
    }

    /// A statement this reader doesn't convert: a comment, kept while the file is starting, or
    /// one counted as skipped.
    #[cold]
    fn other(&mut self, key: &[u8], line: &[u8]) {
        if key.is_empty() {
            return;
        }
        if key[0] == b'#' {
            let starting = self.v.is_empty() && self.libs.is_empty() && self.parts.is_empty();
            if starting && self.object.is_none() && self.group.is_none() && self.comments.len() < 64 {
                self.comments.push(text(trim(&line[1..])));
            }
            return;
        }
        if let Some(k) = self.skipped.iter_mut().find(|k| k.0 == key) {
            k.1 += 1;
        } else if self.skipped.len() < 64 {
            self.skipped.push((key.to_vec(), 1));
        }
    }

    /// The group for the current part and material, added if it's new.
    fn group(&mut self) -> usize {
        if let Some(g) = self.current {
            return g;
        }
        let part = if let Some(p) = self.part {
            p
        } else {
            let next = self.parts.len();
            let key = (self.object.clone(), self.group.clone());
            let p = *self.part_of.entry(key).or_insert_with_key(|k| {
                self.parts.push(k.clone());
                next
            });
            self.part = Some(p);
            p
        };
        let next = self.groups.len();
        let material = self.material;
        let g = *self.group_of.entry((part, material)).or_insert_with(|| {
            self.groups.push(Group {
                part,
                material,
                prim: Prim::default(),
                raw_colors: Vec::new(),
                all_normals: true,
                points: Vec::new(),
                lines: Vec::new(),
            });
            next
        });
        self.current = Some(g);
        g
    }

    /// `p` (points) or, when `lines`, `l` (a line through each position in turn); the texture
    /// coordinates a line may name aren't kept. Dropped if it names a position that isn't there.
    fn loose(&mut self, lines: bool, rest: &[u8]) {
        let mut at = std::mem::take(&mut self.face);
        at.clear();
        for w in rest.split(|&b| stream::space(b)).filter(|w| !w.is_empty()) {
            if w[0] == b'#' {
                break;
            }
            let Some(p) = w.split(|&c| c == b'/').next().and_then(|p| resolve(p, self.v.len())) else {
                at.clear();
                break;
            };
            at.push([p, NONE, NONE]);
        }
        if lines && at.len() >= 2 {
            let g = self.group();
            let out = &mut self.groups[g].lines;
            for pair in at.windows(2) {
                out.extend([pair[0][0], pair[1][0]]);
            }
            self.lines += at.len() - 1;
        } else if !lines && !at.is_empty() {
            let g = self.group();
            self.groups[g].points.extend(at.iter().map(|c| c[0]));
            self.points += at.len();
        }
        self.face = at;
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
            let g = self.group();
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

/// Points or lines on vertices of their own: the positions they use (with their colors), and
/// for lines the pairs as indices into those.
fn loose_prim(b: &mut Builder, at: &[u32], lines: bool, v: &[[f32; 3]], colors: &[[f32; 3]], scale: f32) -> Value {
    let mut local: HashMap<u32, u32> = HashMap::new();
    let mut positions = Vec::new();
    let mut indices = Vec::with_capacity(at.len());
    for &p in at {
        let k = *local.entry(p).or_insert_with(|| {
            positions.push(p);
            u32_of(positions.len() - 1)
        });
        indices.push(k);
    }
    let pos: Vec<[f32; 3]> = positions.iter().map(|&p| v[p as usize]).collect();
    let n = pos.len();
    let mut attributes = json!({ "POSITION": b.positions(pos) });
    if !colors.is_empty() {
        let c: Vec<f32> = positions
            .iter()
            .flat_map(|&p| {
                let c = colors.get(p as usize).copied().unwrap_or([1.0; 3]);
                let [r, g, b] = c.map(|x| srgb_to_linear((x / scale).clamp(0.0, 1.0)));
                [r, g, b, 1.0]
            })
            .collect();
        attributes["COLOR_0"] = json!(b.accessor(Part::f32(c), 5126, n, "VEC4", Some(34962)));
    }
    if lines {
        let count = indices.len();
        let i = b.accessor(Part::u32(indices), 5125, count, "SCALAR", Some(34963));
        json!({ "attributes": attributes, "indices": i, "mode": 1 })
    } else {
        json!({ "attributes": attributes, "mode": 0 })
    }
}

/// OBJ, with MTL colors and textures (base color and normal map). Faces may mix corners with
/// and without texture coordinates or normals. The file is read a block at a time, and each
/// corner becomes a vertex of its part's primitive in its material as it's read.
pub(crate) fn obj(input: Input<'_>, assets: &dyn Assets, keep: Keep) -> Result<Document<'static>> {
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
    ensure!(r.triangles + r.points + r.lines > 0, "OBJ has no faces");
    ensure!(r.triangles <= MAX_TRIANGLES, "too many triangles to convert");

    // Vertex colors: 0-1 floats, or 0-255 from some tools; sRGB either way.
    let has_colors = !r.colors.is_empty();
    let scale = if r.colors.iter().flatten().any(|&c| c > 1.0) {
        255.0
    } else {
        1.0
    };
    let mut b = Builder::default();
    // Points and lines first, while the file's positions are here.
    let loose: Vec<Vec<Value>> = r
        .groups
        .iter()
        .map(|g| {
            let mut out = Vec::new();
            if !g.points.is_empty() {
                out.push(loose_prim(&mut b, &g.points, false, &r.v, &r.colors, scale));
            }
            if !g.lines.is_empty() {
                out.push(loose_prim(&mut b, &g.lines, true, &r.v, &r.colors, scale));
            }
            out
        })
        .collect();
    let Reader {
        libs,
        groups,
        materials: names,
        parts,
        triangles,
        points,
        lines,
        comments,
        skipped,
        ..
    } = r;

    // Material libraries and textures that would only be dropped aren't read.
    let textures = keep.materials && keep.textures;
    let mtl = if keep.materials {
        read_libraries(&libs, dir, assets)
    } else {
        HashMap::new()
    };
    // The materials used, in first-use order, and whether any triangles using each have texture
    // coordinates.
    let mut index_of: HashMap<usize, usize> = HashMap::new();
    let mut used: Vec<usize> = Vec::new();
    let mut textured: Vec<bool> = Vec::new();
    for (g, extra) in groups.iter().zip(&loose) {
        if g.prim.indices.is_empty() && extra.is_empty() {
            continue;
        }
        let m = *index_of.entry(g.material).or_insert_with(|| {
            used.push(g.material);
            textured.push(false);
            used.len() - 1
        });
        textured[m] |= !g.prim.uvs.is_empty();
    }
    let mut images = Images::new(!keep.image_bytes, dir);
    let mut image_of: HashMap<std::path::PathBuf, Option<usize>> = HashMap::new();
    let mut out_materials = Vec::new();
    let mut looks: Vec<Look> = Vec::new();
    for (m, &name) in used.iter().enumerate() {
        let name = &names[name];
        let entry = mtl.get(name);
        // Without a material, vertex colors show as they are; otherwise the usual 0.8 gray.
        let fallback = if has_colors { [1.0; 3] } else { [0.8; 3] };
        let d = entry.and_then(|e| e.color).unwrap_or(fallback);
        let mut mat = Material::plain(
            if name.is_empty() { "default" } else { name },
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
        let mut texture = |reference: Option<&MapRef>| -> Option<Tex> {
            if !textures || !textured[m] {
                return None;
            }
            let reference = reference?;
            let found = assets.find(&reference.file, dir)?;
            let image = (*image_of
                .entry(found.clone())
                .or_insert_with(|| images.file(assets, found, &reference.file)))?;
            Some(reference.tex(image))
        };
        let color = texture(entry.and_then(|e| e.color_map.as_ref()));
        let normal = texture(entry.and_then(|e| e.normal_map.as_ref()));
        mat.color_image = color.as_ref().map(|t| t.image);
        mat.normal_image = normal.as_ref().map(|t| t.image);
        let mut look = Look {
            color,
            normal,
            ..Look::default()
        };
        if let Some(e) = entry {
            look.emissive = texture(e.emissive_map.as_ref());
            // glTF reads roughness and metalness from one image's green and blue.
            if e.roughness_map.is_some() && e.roughness_map == e.metallic_map {
                look.metal_rough = texture(e.roughness_map.as_ref());
            }
            let pbr = e.roughness.is_some() || e.metallic.is_some();
            let spec_tex = texture(e.specular_map.as_ref());
            if !pbr && (e.specular.is_some() || spec_tex.is_some()) {
                // Phong's highlight, where half is glTF's usual.
                look.specular = Some(Specular {
                    factor: Factor::of(1.0),
                    color: e.specular.unwrap_or([0.5; 3]).map(|c| c * 2.0),
                    color_tex: spec_tex,
                });
            }
            // Every exporter writes Ni, often a placeholder of 1 (which would take the highlight away).
            look.ior = e.ior.filter(|i| *i > 1.0 && *i <= 5.0);
            let sheen_tex = texture(e.sheen_map.as_ref());
            if e.sheen.is_some() || sheen_tex.is_some() {
                look.sheen = Some(Sheen {
                    color: e.sheen.unwrap_or([1.0; 3]),
                    color_tex: sheen_tex,
                    roughness: Factor::of(0.0),
                });
            }
            if e.clearcoat.is_some() || e.clearcoat_roughness.is_some() {
                look.clearcoat = Some(Clearcoat {
                    factor: Factor::of(e.clearcoat.unwrap_or(1.0)),
                    roughness: Factor::of(e.clearcoat_roughness.unwrap_or(0.0)),
                    normal: None,
                });
            }
            look.anisotropy = e.anisotropy;
            // Color on, ambient off: the color as it is, unlit.
            #[allow(clippy::float_cmp, reason = "illum is a whole number")]
            {
                look.unlit = e.illum == Some(0.0);
            }
        }
        looks.push(look);
        out_materials.push(mat);
    }
    let linear = super::linear_colors(input, "obj");
    if !linear {
        for m in &mut out_materials {
            for c in m.color[..3].iter_mut().chain(&mut m.emissive) {
                *c = srgb_to_linear(*c);
            }
        }
        for s in looks.iter_mut().filter_map(|l| l.sheen.as_mut()) {
            s.color = s.color.map(srgb_to_linear);
        }
    }
    if textures {
        super::guess_texture(&mut out_materials, &mut images, &textured, input.path, assets);
    }

    // A mesh per part, in the order the parts turn up.
    let mut prims_of: Vec<Vec<Value>> = vec![Vec::new(); parts.len()];
    for (g, extra) in groups.into_iter().zip(loose) {
        let Group {
            part,
            material,
            mut prim,
            raw_colors,
            all_normals,
            ..
        } = g;
        let Some(&m) = index_of.get(&material) else {
            continue;
        };
        if !prim.indices.is_empty() {
            let nv = prim.positions.len();
            if !prim.uvs.is_empty() {
                prim.uvs.resize(nv, [0.0, 1.0]);
            }
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
            if let Some(mut p) = b.primitive(prim) {
                p["material"] = json!(m);
                prims_of[part].push(p);
            }
        }
        for mut p in extra {
            p["material"] = json!(m);
            prims_of[part].push(p);
        }
    }
    let (mut nodes, mut meshes) = (Vec::new(), Vec::new());
    for ((object, group), primitives) in parts.into_iter().zip(prims_of) {
        if primitives.is_empty() {
            continue;
        }
        let mut mesh = json!({ "primitives": primitives });
        let mut node = json!({ "mesh": meshes.len() });
        // Named for its group, or failing that its object; a group inside an object keeps that
        // object's name in extras.
        if let Some(name) = group.as_ref().or(object.as_ref()) {
            mesh["name"] = json!(name);
            node["name"] = json!(name);
        }
        if let (Some(o), Some(_)) = (&object, &group) {
            node["extras"] = json!({ "object": o });
        }
        meshes.push(mesh);
        nodes.push(node);
    }

    let mut source = json!({ "format": "OBJ" });
    if !libs.is_empty() {
        source["mtllib"] = json!(libs);
    }
    if !comments.is_empty() {
        source["comments"] = json!(comments);
    }
    if linear {
        source["linearColors"] = json!(true);
    }
    if has_colors {
        source["vertexColorScale"] = json!(scale);
    }
    if !skipped.is_empty() {
        let m: Map<String, Value> = skipped.iter().map(|(k, n)| (text(k), json!(n))).collect();
        source["skipped"] = Value::Object(m);
    }
    let mut facts = Map::new();
    if triangles > 0 {
        facts.insert("triangles".into(), json!(triangles));
    }
    if points > 0 {
        facts.insert("points".into(), json!(points));
    }
    if lines > 0 {
        facts.insert("lines".into(), json!(lines));
    }
    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert(
        "scenes".into(),
        json!([{ "nodes": (0..nodes.len()).collect::<Vec<_>>() }]),
    );
    json.insert("nodes".into(), Value::Array(nodes));
    json.insert("meshes".into(), Value::Array(meshes));
    let fulls = out_materials
        .iter()
        .enumerate()
        .map(|(i, m)| looks.get(i).map_or_else(|| Full::from(m), |l| l.full(m)));
    b.materials(fulls, images.list, &mut json);
    // Every MTL statement, in its material's extras.
    if let Some(Value::Array(list)) = json.get_mut("materials") {
        for (mat, &name) in list.iter_mut().zip(&used) {
            if let Some(e) = mtl.get(&names[name]).filter(|e| !e.raw.is_empty()) {
                mat["extras"] = json!({ "mtl": e.raw });
            }
        }
    }
    Ok(b.finish(json, facts))
}
