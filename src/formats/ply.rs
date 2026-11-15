//! PLY (the Stanford polygon file format), ASCII and binary in either byte order.
//!
//! Read: positions, normals, vertex colors (sRGB, as integers or floats), texture coordinates
//! under any of their usual names, polygons (triangulated in their plane) and triangle strips;
//! `MeshLab`'s per-face texture coordinates and `comment TextureFile`; per-face colors. Elements
//! the reader doesn't use are skipped. A file without faces is a point cloud: no mesh, but its
//! point count.

use std::path::Path;

use serde_json::{Map, Value, json};

use super::Input;
use super::stream::{self, Blocks};
use crate::assets::Assets;
use crate::mesh::{FastMap, MAX_TRIANGLES, Material, Mesh, Prim, smooth_normals, srgb_to_linear, triangulate};
use crate::num::{f32_of, u32_of};
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
        names
            .iter()
            .find_map(|n| self.props.iter().position(|p| p.name == *n && p.count.is_none()))
    }

    fn list(&self, names: &[&str]) -> Option<usize> {
        names
            .iter()
            .find_map(|n| self.props.iter().position(|p| p.name == *n && p.count.is_some()))
    }
}

/// Why a record couldn't be read from a stretch of data.
enum Stop {
    /// The stretch ends first: read more of the file and try the record again.
    Short,
    Bad(&'static str),
}

type Got<T> = std::result::Result<T, Stop>;

/// Reads values in whichever encoding the file uses, from a stretch of its data that holds
/// whole tokens (ASCII) but maybe not whole records.
struct Cursor<'a> {
    format: Format,
    d: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    #[inline]
    fn token(&mut self) -> Got<&[u8]> {
        let d = self.d;
        let mut i = self.pos;
        while i < d.len() && stream::space(d[i]) {
            i += 1;
        }
        let start = i;
        while i < d.len() && !stream::space(d[i]) {
            i += 1;
        }
        if i == start {
            return Err(Stop::Short);
        }
        self.pos = i;
        Ok(&d[start..i])
    }

    #[inline]
    fn bytes<const N: usize>(&mut self) -> Got<[u8; N]> {
        let b = self.d.get(self.pos..self.pos + N).ok_or(Stop::Short)?;
        self.pos += N;
        let mut a = [0u8; N];
        a.copy_from_slice(b);
        if self.format == Format::Be {
            a.reverse();
        }
        Ok(a)
    }

    #[inline]
    fn value(&mut self, ty: Ty) -> Got<f64> {
        if self.format == Format::Ascii {
            let t = self.token()?;
            let v = match ty {
                // Rounded once, straight to the type the file declares.
                Ty::F32 => stream::f32_from(t).map(f64::from),
                _ => stream::f64_from(t),
            };
            return v.ok_or(Stop::Bad("bad PLY number"));
        }
        Ok(match ty {
            Ty::I8 => f64::from(self.bytes::<1>()?[0].cast_signed()),
            Ty::U8 => f64::from(self.bytes::<1>()?[0]),
            Ty::I16 => f64::from(i16::from_le_bytes(self.bytes()?)),
            Ty::U16 => f64::from(u16::from_le_bytes(self.bytes()?)),
            Ty::I32 => f64::from(i32::from_le_bytes(self.bytes()?)),
            Ty::U32 => f64::from(u32::from_le_bytes(self.bytes()?)),
            Ty::F32 => f64::from(f32::from_le_bytes(self.bytes()?)),
            Ty::F64 => f64::from_le_bytes(self.bytes()?),
        })
    }

    /// A vertex index; `u32::MAX` (never valid) for a negative or NaN one.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "`as` saturates, and the vertex check refuses what's past the end"
    )]
    #[inline]
    fn index(&mut self, ty: Ty) -> Got<u32> {
        let neg = |v: i64| u32::try_from(v).unwrap_or(u32::MAX);
        match (self.format, ty) {
            (Format::Ascii, _) => {
                let t = self.token()?;
                if t.len() < 10 && t.iter().all(u8::is_ascii_digit) {
                    return Ok(t.iter().fold(0, |n, &c| n * 10 + u32::from(c - b'0')));
                }
                let v = stream::f64_from(t).ok_or(Stop::Bad("bad PLY number"))?;
                Ok(if v >= 0.0 { v as u32 } else { u32::MAX })
            }
            (_, Ty::I32) => Ok(neg(i64::from(i32::from_le_bytes(self.bytes()?)))),
            (_, Ty::U32) => Ok(u32::from_le_bytes(self.bytes()?)),
            _ => {
                let v = self.value(ty)?;
                Ok(if v >= 0.0 { v as u32 } else { u32::MAX })
            }
        }
    }

    /// Steps over a value.
    #[inline]
    fn skip(&mut self, ty: Ty, n: usize) -> Got<()> {
        if self.format == Format::Ascii {
            for _ in 0..n {
                self.token()?;
            }
            return Ok(());
        }
        let end = self.pos + ty.size() * n;
        if end > self.d.len() {
            return Err(Stop::Short);
        }
        self.pos = end;
        Ok(())
    }

    /// A list's length.
    #[inline]
    fn len(&mut self, ty: Ty) -> Got<usize> {
        let v = match (self.format, ty) {
            (Format::Le | Format::Be, Ty::U8) => return Ok(usize::from(self.bytes::<1>()?[0])),
            _ => self.value(ty)?,
        };
        if !(0.0..=1e6).contains(&v) {
            return Err(Stop::Bad("bad PLY list length"));
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "checked to be in 0..=1e6"
        )]
        Ok(v as usize)
    }
}

/// A binary value at the start of `b`, which holds at least its size.
#[inline]
fn at(b: &[u8], ty: Ty, be: bool) -> f64 {
    let mut a = [0u8; 8];
    let n = ty.size();
    a[..n].copy_from_slice(&b[..n]);
    if be {
        a[..n].reverse();
    }
    match ty {
        Ty::I8 => f64::from(a[0].cast_signed()),
        Ty::U8 => f64::from(a[0]),
        Ty::I16 => f64::from(i16::from_le_bytes([a[0], a[1]])),
        Ty::U16 => f64::from(u16::from_le_bytes([a[0], a[1]])),
        Ty::I32 => f64::from(i32::from_le_bytes([a[0], a[1], a[2], a[3]])),
        Ty::U32 => f64::from(u32::from_le_bytes([a[0], a[1], a[2], a[3]])),
        Ty::F32 => f64::from(f32::from_le_bytes([a[0], a[1], a[2], a[3]])),
        Ty::F64 => f64::from_le_bytes(a),
    }
}

/// The header: format, elements, the texture file `MeshLab` names, and where the data starts.
fn header(data: &[u8]) -> Result<(Format, Vec<Element>, Option<String>, usize)> {
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    let mut texture = None;
    let mut at = 0;
    let mut first = true;
    loop {
        let rest = data
            .get(at..)
            .filter(|r| !r.is_empty())
            .context("PLY header has no end_header")?;
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

/// Vertex data, by vertex. Normals are unit length and texture coordinates flipped to glTF's
/// convention as they're read.
#[derive(Default)]
struct Vertices {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    col: Vec<[f32; 4]>,
    uv: Vec<[f32; 2]>,
}

/// Faces kept as the file gives them, for when they can't become triangles as they're read:
/// they come before the vertices, or give their corners their own texture coordinates or
/// colors. Each face's corners (vertex indices), and per corner the texture coordinate
/// (flipped) and color.
#[derive(Default)]
struct Faces {
    starts: Vec<usize>,
    corners: Vec<u32>,
    corner_uv: Vec<[f32; 2]>,
    corner_col: Vec<[f32; 4]>,
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
        let c = f32_of((vals[s] / el.props[s].ty.color_scale()).clamp(0.0, 1.0));
        srgb_to_linear(c)
    };
    let a = alpha.map_or(1.0, |s| {
        f32_of((vals[s] / el.props[s].ty.color_scale()).clamp(0.0, 1.0))
    });
    [ch(slots[0]), ch(slots[1]), ch(slots[2]), a]
}

fn unit(n: [f32; 3]) -> [f32; 3] {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 && len.is_finite() {
        n.map(|c| c / len)
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Where an element's properties are, and which of them the reader uses.
struct Plan {
    xyz: [usize; 3],
    n: Option<[usize; 3]>,
    rgb: Option<[usize; 3]>,
    alpha: Option<usize>,
    uv: Option<(usize, usize)>,
    index_list: Option<usize>,
    uv_list: Option<usize>,
    /// The scalar properties read; the rest are stepped over.
    wanted: Vec<bool>,
    /// Binary records of fixed size (no lists): the size and each property's offset.
    fixed: Option<(usize, Vec<usize>)>,
}

impl Plan {
    fn new(el: &Element) -> Plan {
        let tri = |a: Option<usize>, b: Option<usize>, c: Option<usize>| Some([a?, b?, c?]);
        let xyz = [el.slot(X), el.slot(Y), el.slot(Z)];
        let mut plan = Plan {
            xyz: xyz.map(|s| s.unwrap_or(usize::MAX)),
            n: tri(el.slot(&["nx"]), el.slot(&["ny"]), el.slot(&["nz"])),
            rgb: tri(el.slot(R), el.slot(G), el.slot(B)),
            alpha: el.slot(A),
            uv: el.slot(U).zip(el.slot(V)),
            index_list: el.list(&["vertex_indices", "vertex_index"]),
            uv_list: el.list(&["texcoord"]),
            wanted: vec![false; el.props.len()],
            fixed: None,
        };
        let used: Vec<usize> = xyz
            .into_iter()
            .flatten()
            .chain(plan.n.into_iter().flatten())
            .chain(plan.rgb.into_iter().flatten())
            .chain(plan.alpha)
            .chain(plan.uv.into_iter().flat_map(|(u, v)| [u, v]))
            .collect();
        for s in used {
            plan.wanted[s] = true;
        }
        if el.props.iter().all(|p| p.count.is_none()) {
            let mut offsets = Vec::with_capacity(el.props.len());
            let mut size = 0;
            for p in &el.props {
                offsets.push(size);
                size += p.ty.size();
            }
            plan.fixed = Some((size, offsets));
        }
        plan
    }
}

/// Everything read from the elements.
#[derive(Default)]
struct Body {
    verts: Vertices,
    /// Whether the vertices have been read (faces after them become triangles at once).
    have_verts: bool,
    /// Triangles, as vertex indices.
    tris: Vec<u32>,
    /// Faces kept for later.
    faces: Faces,
    /// Triangle strips' triangles, checked against the vertex count at the end.
    strip_tris: Vec<[u32; 3]>,
    points: usize,
    // Scratch, reused from record to record.
    vals: Vec<f64>,
    idx: Vec<u32>,
    tc: Vec<f32>,
    pts: Vec<[f32; 3]>,
    local: Vec<[u32; 3]>,
}

pub(crate) fn ply(input: Input<'_>, assets: &dyn Assets) -> Result<(Option<Mesh>, Map<String, Value>)> {
    let path = input.path;
    let mut blocks = input.blocks()?;
    // The header, whole.
    let mut want = 4096;
    let (format, elements, texture, start) = loop {
        blocks.need(want)?;
        let d = blocks.data();
        let ended = d
            .windows(11)
            .position(|w| w == b"\nend_header")
            .is_some_and(|i| d[i + 11..].contains(&b'\n'));
        if ended || blocks.at_end() {
            break header(d)?;
        }
        want *= 2;
    };
    blocks.consume(start);
    let room = blocks.size_hint().saturating_sub(start);
    let has_faces = elements
        .iter()
        .any(|e| (e.name == "face" || e.name == "tristrips") && e.count > 0);
    let mut body = Body::default();
    for el in &elements {
        // Every item takes at least a byte (binary) or two (ASCII, a value and a space).
        ensure!(
            el.count <= room,
            "PLY element {} counts more items than the file holds",
            el.name
        );
        read_element(&mut blocks, format, el, has_faces, &mut body)?;
    }
    let mut facts = Map::new();
    if !has_faces {
        facts.insert("points".into(), json!(body.points));
        return Ok((None, facts));
    }
    let mesh = build(body, texture, path, assets)?;
    facts.insert("triangles".into(), json!(mesh.triangles()));
    facts.insert(
        "vertices".into(),
        json!(mesh.prims.iter().map(|p| p.positions.len()).sum::<usize>()),
    );
    Ok((Some(mesh), facts))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Vertex,
    Face,
    Strip,
    Other,
}

/// Reads one element's items, a block of the file at a time.
fn read_element(blocks: &mut Blocks<'_>, format: Format, el: &Element, has_faces: bool, body: &mut Body) -> Result<()> {
    let kind = match el.name.as_str() {
        "vertex" => Kind::Vertex,
        "face" => Kind::Face,
        "tristrips" => Kind::Strip,
        _ => Kind::Other,
    };
    let plan = Plan::new(el);
    if kind == Kind::Vertex {
        ensure!(
            plan.xyz.iter().all(|&s| s != usize::MAX),
            "PLY vertices have no x, y and z"
        );
        if has_faces {
            let v = &mut body.verts;
            v.pos.reserve_exact(el.count);
            if plan.n.is_some() {
                v.nrm.reserve_exact(el.count);
            }
            if plan.rgb.is_some() {
                v.col.reserve_exact(el.count);
            }
            if plan.uv.is_some() {
                v.uv.reserve_exact(el.count);
            }
        }
    }
    // Faces become triangles as they're read when their vertices are known and their corners
    // share the vertices' data.
    let direct = kind == Kind::Face && body.have_verts && plan.rgb.is_none() && plan.uv_list.is_none();
    // Sized from the first block's triangles per face once it's read.
    let mut sized = !direct;
    body.vals.resize(el.props.len(), 0.0);
    let mut done = 0;
    while done < el.count {
        blocks.fill()?;
        let all = blocks.data();
        let end = if blocks.at_end() || format != Format::Ascii {
            all.len()
        } else {
            stream::words(all).unwrap_or(0)
        };
        let mut cur = Cursor {
            format,
            d: &all[..end],
            pos: 0,
        };
        let mut kept = 0;
        let before = done;
        while done < el.count {
            let r = match kind {
                Kind::Vertex => vertex(&mut cur, el, &plan, has_faces, body),
                Kind::Face => face(&mut cur, el, &plan, direct, body),
                Kind::Strip => strip(&mut cur, el, &plan, body),
                Kind::Other => other(&mut cur, el, &plan),
            };
            match r {
                Ok(()) => {
                    done += 1;
                    kept = cur.pos;
                }
                Err(Stop::Short) => break,
                Err(Stop::Bad(m)) => bail!("{m}"),
            }
        }
        let buffered = all.len();
        blocks.consume(kept);
        if done == before {
            ensure!(!blocks.at_end(), "PLY data ends early");
            // One record is longer than what's buffered.
            blocks.need(buffered * 2 + 64)?;
        }
        ensure!(body.tris.len() / 3 <= MAX_TRIANGLES, "too many triangles to convert");
        if !sized && done > 0 {
            sized = true;
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a capacity guess from the rate so far; MAX_TRIANGLES caps it"
            )]
            let est = (body.tris.len() as f64 / done as f64 * el.count as f64 * 1.01) as usize + 48;
            let est = est.min(MAX_TRIANGLES * 3);
            body.tris.reserve_exact(est.saturating_sub(body.tris.len()));
        }
    }
    if kind == Kind::Vertex {
        body.have_verts = true;
    }
    Ok(())
}

/// Reads an item's scalar properties into `vals` (those the plan wants) and its lists into
/// `idx` (vertex indices) and `tc` (texture coordinates).
#[inline]
fn record(
    cur: &mut Cursor,
    el: &Element,
    plan: &Plan,
    vals: &mut [f64],
    idx: &mut Vec<u32>,
    tc: &mut Vec<f32>,
) -> Got<()> {
    if let (Some((size, offsets)), false) = (&plan.fixed, cur.format == Format::Ascii) {
        let rec = cur.d.get(cur.pos..cur.pos + size).ok_or(Stop::Short)?;
        let be = cur.format == Format::Be;
        for (k, p) in el.props.iter().enumerate() {
            if plan.wanted[k] {
                vals[k] = at(&rec[offsets[k]..], p.ty, be);
            }
        }
        cur.pos += size;
        return Ok(());
    }
    idx.clear();
    tc.clear();
    for (k, p) in el.props.iter().enumerate() {
        match p.count {
            None if plan.wanted[k] => vals[k] = cur.value(p.ty)?,
            None => cur.skip(p.ty, 1)?,
            Some(cty) => {
                let n = cur.len(cty)?;
                if Some(k) == plan.index_list {
                    for _ in 0..n {
                        idx.push(cur.index(p.ty)?);
                    }
                } else if Some(k) == plan.uv_list {
                    for _ in 0..n {
                        tc.push(f32_of(cur.value(p.ty)?));
                    }
                } else {
                    cur.skip(p.ty, n)?;
                }
            }
        }
    }
    Ok(())
}

fn vertex(cur: &mut Cursor, el: &Element, plan: &Plan, keep: bool, body: &mut Body) -> Got<()> {
    record(cur, el, plan, &mut body.vals, &mut body.idx, &mut body.tc)?;
    body.points += 1;
    if !keep {
        return Ok(());
    }
    let vals = &body.vals;
    let v = &mut body.verts;
    let g = |s: usize| f32_of(vals[s]);
    v.pos.push(plan.xyz.map(g));
    if let Some(n) = plan.n {
        v.nrm.push(unit(n.map(g)));
    }
    if let Some(s) = plan.rgb {
        v.col.push(color(el, vals, &s, plan.alpha));
    }
    if let Some((a, b)) = plan.uv {
        v.uv.push([g(a), 1.0 - g(b)]);
    }
    Ok(())
}

fn face(cur: &mut Cursor, el: &Element, plan: &Plan, direct: bool, body: &mut Body) -> Got<()> {
    record(cur, el, plan, &mut body.vals, &mut body.idx, &mut body.tc)?;
    if plan.index_list.is_none() {
        return Ok(());
    }
    let idx = &body.idx;
    if direct {
        let nv = body.verts.pos.len();
        if idx.len() < 3 || idx.iter().any(|&i| i as usize >= nv) {
            return Ok(());
        }
        if idx.len() == 3 {
            body.tris.extend_from_slice(idx);
            return Ok(());
        }
        body.pts.clear();
        body.pts.extend(idx.iter().map(|&i| body.verts.pos[i as usize]));
        body.local.clear();
        triangulate(&body.pts, &mut body.local);
        body.tris
            .extend(body.local.iter().flat_map(|t| t.map(|k| idx[k as usize])));
        return Ok(());
    }
    let f = &mut body.faces;
    f.starts.push(f.corners.len());
    f.corners.extend_from_slice(idx);
    if plan.uv_list.is_some() {
        let t = &body.tc;
        for c in 0..idx.len() {
            let get = |k: usize| t.get(k).copied().unwrap_or(0.0);
            f.corner_uv.push([get(2 * c), 1.0 - get(2 * c + 1)]);
        }
    }
    if let Some(s) = plan.rgb {
        let c = color(el, &body.vals, &s, plan.alpha);
        f.corner_col.extend(std::iter::repeat_n(c, idx.len()));
    }
    Ok(())
}

/// Strips separated by -1; every other triangle's order flips to keep the winding.
fn strip(cur: &mut Cursor, el: &Element, plan: &Plan, body: &mut Body) -> Got<()> {
    record(cur, el, plan, &mut body.vals, &mut body.idx, &mut body.tc)?;
    if plan.index_list.is_none() {
        return Ok(());
    }
    for run in body.idx.split(|&i| i == u32::MAX) {
        for k in 2..run.len() {
            let (a, b, c) = (run[k - 2], run[k - 1], run[k]);
            body.strip_tris.push(if k % 2 == 0 { [a, b, c] } else { [b, a, c] });
        }
    }
    Ok(())
}

/// An element the reader doesn't use: stepped over.
fn other(cur: &mut Cursor, el: &Element, plan: &Plan) -> Got<()> {
    if let (Some((size, _)), false) = (&plan.fixed, cur.format == Format::Ascii) {
        if cur.pos + size > cur.d.len() {
            return Err(Stop::Short);
        }
        cur.pos += size;
        return Ok(());
    }
    for p in &el.props {
        let n = match p.count {
            None => 1,
            Some(cty) => cur.len(cty)?,
        };
        cur.skip(p.ty, n)?;
    }
    Ok(())
}

/// The primitive: shared vertices as the file has them, or split where faces give their corners
/// their own texture coordinates or colors.
fn build(body: Body, texture: Option<String>, path: &Path, assets: &dyn Assets) -> Result<Mesh> {
    let Body {
        verts,
        mut tris,
        faces,
        strip_tris,
        ..
    } = body;
    let nv = verts.pos.len();
    // Faces kept for later become triangles, as corner numbers (into faces.corners) so
    // per-corner data follows.
    let mut corner_tris: Vec<[usize; 3]> = Vec::new();
    let mut local: Vec<[u32; 3]> = Vec::new();
    let mut pts: Vec<[f32; 3]> = Vec::new();
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
    let strips: Vec<[u32; 3]> = strip_tris
        .into_iter()
        .filter(|t| t.iter().all(|&i| (i as usize) < nv))
        .collect();
    let total = tris.len() / 3 + corner_tris.len() + strips.len();
    ensure!(total <= MAX_TRIANGLES, "too many triangles to convert");
    ensure!(total > 0, "PLY has no faces");

    let per_corner = !faces.corner_uv.is_empty() || !faces.corner_col.is_empty();
    let has_uv = !faces.corner_uv.is_empty() || verts.uv.len() == nv && nv > 0;
    let has_col = !faces.corner_col.is_empty() || verts.col.len() == nv && nv > 0;
    let has_nrm = verts.nrm.len() == nv;
    let mut p = Prim::default();
    if per_corner {
        // One output vertex per distinct (vertex, corner data).
        let mut seen: FastMap<(u32, [u32; 2], [u32; 4]), u32> = FastMap::default();
        let mut emit = |p: &mut Prim, v: u32, uv: Option<[f32; 2]>, col: Option<[f32; 4]>| -> u32 {
            let key = (
                v,
                uv.unwrap_or_default().map(f32::to_bits),
                col.unwrap_or_default().map(f32::to_bits),
            );
            *seen.entry(key).or_insert_with(|| {
                p.positions.push(verts.pos[v as usize]);
                if has_nrm {
                    p.normals.push(verts.nrm[v as usize]);
                }
                if has_uv {
                    p.uvs
                        .push(uv.or_else(|| verts.uv.get(v as usize).copied()).unwrap_or([0.0, 1.0]));
                }
                if has_col {
                    p.colors
                        .push(col.or_else(|| verts.col.get(v as usize).copied()).unwrap_or([1.0; 4]));
                }
                u32_of(p.positions.len() - 1)
            })
        };
        for &v in &tris {
            let id = emit(&mut p, v, None, None);
            p.indices.push(id);
        }
        for t in &corner_tris {
            for &c in t {
                let id = emit(
                    &mut p,
                    faces.corners[c],
                    faces.corner_uv.get(c).copied(),
                    faces.corner_col.get(c).copied(),
                );
                p.indices.push(id);
            }
        }
        for t in &strips {
            for &v in t {
                let id = emit(&mut p, v, None, None);
                p.indices.push(id);
            }
        }
    } else {
        tris.reserve_exact(total * 3 - tris.len());
        tris.extend(corner_tris.iter().flatten().map(|&c| faces.corners[c]));
        tris.extend(strips.iter().flatten());
        if tris.capacity() > tris.len() + tris.len() / 16 + 16 {
            tris.shrink_to_fit();
        }
        p.indices = tris;
        p.positions = verts.pos;
        if has_nrm {
            p.normals = verts.nrm;
        }
        if has_uv {
            p.uvs = verts.uv;
        }
        if has_col {
            p.colors = verts.col;
        }
    }
    if !has_nrm {
        p.normals = smooth_normals(&p.positions, &p.indices);
    }

    let mut mesh = Mesh::default();
    // Vertex colors show as they are; otherwise the usual gray.
    let base = if has_col { [1.0; 4] } else { crate::mesh::GRAY };
    let mut mat = Material::plain("default", base);
    if has_uv {
        let dir = path.parent().unwrap_or(Path::new(""));
        if let Some(img) = texture
            .and_then(|t| assets.find(&t, dir))
            .and_then(|f| assets.load_image(&f))
        {
            mesh.images.push(img);
            mat.color_image = Some(0);
            mat.color = [1.0; 4];
        }
    }
    mesh.materials.push(mat);
    mesh.prims.push(p);
    Ok(mesh)
}
