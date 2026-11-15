//! glTF 2.0, `.gltf` and `.glb`, read into a [`Mesh`]. glTF's space is the mesh's (Y up,
//! metres), so geometry comes over as it is, placed by the scene's node transforms.
//!
//! - The default scene is drawn (else scene 0; with no scenes, every root node; with no nodes,
//!   every mesh). Node transforms are baked into the vertices; a mirroring transform flips the
//!   winding back. `EXT_mesh_gpu_instancing` instances are baked the same way.
//! - Skinned meshes take the pose the nodes are in (linear blend skinning); morph targets take
//!   the node's or mesh's default weights.
//! - Triangles, strips and fans; points and lines are counted, not drawn.
//! - `POSITION`, `NORMAL`, the color texture's `TEXCOORD_n` (its `KHR_texture_transform`
//!   baked in) and `COLOR_0`, in any component type `KHR_mesh_quantization` allows, sparse
//!   accessors included. Without `NORMAL` a primitive keeps no normals: glTF shades it flat.
//! - Materials: base color, metallic and roughness factors, emissive, double-sidedness, and
//!   the color and normal textures (`KHR_materials_pbrSpecularGlossiness` as its diffuse).
//! - Meshopt compression is decoded first. Draco (`KHR_draco_mesh_compression`) is refused.
//!
//! A GLB file isn't read whole: accessors are read from the file a block at a time, so the peak
//! is little more than the mesh. A GLB in memory is read in place. External buffers must sit
//! inside the model's folder; for a model in memory they come from [`Assets::read`].

use std::borrow::Cow;
#[cfg(feature = "fs")]
use std::fs::File;
#[cfg(feature = "fs")]
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use serde_json::{Map, Value, json};

use super::Input;
use crate::assets::Assets;
use crate::gltf as container;
use crate::mesh::{Image, MAX_TRIANGLES, Material, Mesh, Prim};
use crate::{Context as _, Result};

/// How much of a buffer is read at once.
const BLOCK: usize = 256 * 1024;

/// How much of the file the first read takes.
#[cfg(feature = "fs")]
const HEAD: usize = 64 * 1024;

/// The most vertices an accessor without data (all zeros) may claim.
const MAX_VERTICES: usize = 3 * MAX_TRIANGLES;

/// Reads the glTF or GLB at `path`, as its mesh (`None` when it draws no triangles) and facts.
pub(crate) fn load(input: Input<'_>, assets: &dyn Assets) -> Result<(Option<Mesh>, Map<String, Value>)> {
    let mut r = Reader::open(input, assets)?;
    let mesh = r.mesh()?;
    let mut facts = Map::new();
    if r.points > 0 {
        facts.insert("points".into(), json!(r.points));
    }
    if r.lines > 0 {
        facts.insert("lines".into(), json!(r.lines));
    }
    Ok((mesh, facts))
}

/// Where a buffer's bytes are.
enum Buf<'a> {
    /// A stretch of a file: the GLB's binary chunk, or an external `.bin`.
    #[cfg(feature = "fs")]
    File {
        file: File,
        base: u64,
        len: u64,
        /// Where the file's cursor is, relative to `base`, to skip needless seeks.
        at: u64,
    },
    Mem(Cow<'a, [u8]>),
}

impl Buf<'_> {
    fn len(&self) -> u64 {
        match self {
            #[cfg(feature = "fs")]
            Buf::File { len, .. } => *len,
            Buf::Mem(v) => v.len() as u64,
        }
    }

    fn read_at(&mut self, off: u64, out: &mut [u8]) -> Result<()> {
        let end = off
            .checked_add(out.len() as u64)
            .filter(|&e| e <= self.len())
            .context("glTF data runs past the end of its buffer")?;
        match self {
            #[cfg(feature = "fs")]
            Buf::File { file, base, at, .. } => {
                if *at != off {
                    file.seek(SeekFrom::Start(*base + off))?;
                }
                file.read_exact(out)?;
                *at = end;
            }
            Buf::Mem(v) => out.copy_from_slice(&v[off as usize..end as usize]),
        }
        Ok(())
    }
}

/// An accessor, resolved.
#[derive(Clone)]
struct Acc {
    view: Option<usize>,
    offset: u64,
    count: usize,
    ctype: u64,
    /// Components per element.
    n: usize,
    normalized: bool,
    sparse: Option<Value>,
}

impl Acc {
    fn csize(&self) -> usize {
        match self.ctype {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            _ => 4,
        }
    }

    fn esize(&self) -> usize {
        self.csize() * self.n
    }
}

/// A 4x4 matrix, column-major as glTF writes it.
type M4 = [f64; 16];

const IDENTITY: M4 = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

/// Where one mesh is drawn: its transforms (several for instances), morph weights and skin.
struct Use {
    mesh: usize,
    weights: Vec<f32>,
    skin: Option<usize>,
    transforms: Vec<M4>,
}

struct Reader<'a> {
    json: Map<String, Value>,
    /// The GLB's binary chunk, until buffer 0 claims it.
    bin: Option<Buf<'a>>,
    bufs: Vec<Option<Buf<'a>>>,
    input: Input<'a>,
    assets: &'a dyn Assets,
    scratch: Vec<u8>,
    /// glTF material → mesh material.
    materials: Vec<Option<usize>>,
    /// The material for primitives that name none.
    default_material: Option<usize>,
    /// glTF image → mesh image, once tried.
    images: Vec<Option<Option<usize>>>,
    /// Every node's world transform, for joints.
    world: Vec<M4>,
    /// Triangles still allowed.
    budget: usize,
    points: usize,
    lines: usize,
}

fn list<'v>(obj: &'v Map<String, Value>, key: &str) -> &'v [Value] {
    obj.get(key)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn index(v: &Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().map(|i| i as usize)
}

fn floats<const N: usize>(v: Option<&Value>, default: [f64; N]) -> [f64; N] {
    let mut out = default;
    if let Some(a) = v.and_then(Value::as_array).filter(|a| a.len() == N) {
        for (o, x) in out.iter_mut().zip(a) {
            *o = x.as_f64().unwrap_or(*o);
        }
    }
    out
}

fn f32s<const N: usize>(v: Option<&Value>, default: [f32; N]) -> [f32; N] {
    floats(v, default.map(f64::from)).map(|x| x as f32)
}

/// `rel` when it's a plain relative path that stays inside the model's folder.
fn checked(rel: &str) -> Result<&str> {
    let safe = !rel.contains(':')
        && Path::new(rel)
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
    ensure!(safe, "refusing to read {rel:?}: not a relative path inside the model's folder");
    Ok(rel)
}

/// An external file the model names by URI, whole: from the model's folder, or for a model in
/// memory from its assets.
fn external<'a>(input: Input<'a>, assets: &'a dyn Assets, uri: &str) -> Result<Cow<'a, [u8]>> {
    let rel = container::percent_decode(uri);
    let p = input.dir().join(checked(&rel)?);
    #[cfg(feature = "fs")]
    if input.bytes().is_none() {
        return Ok(Cow::Owned(std::fs::read(&p).map_err(|e| invalid!("glTF buffer {rel:?}: {e}"))?));
    }
    assets
        .read(&p)
        .with_context(|| format!("glTF buffer {rel:?} isn't among the assets"))
}

/// Where a GLB's binary chunk is: offset and length.
type Span = (u64, u64);

/// A GLB's JSON and where its binary chunk is, from its first bytes. `read(off, len)` reads
/// the container.
fn glb_layout(
    total: u64,
    mut read: impl FnMut(u64, usize) -> Result<Vec<u8>>,
) -> Result<(Map<String, Value>, Option<Span>)> {
    let head = read(0, 20).context("GLB is truncated")?;
    let u32_at = |b: &[u8], at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    ensure!(u32_at(&head, 4) == 2, "GLB version {} is not 2", u32_at(&head, 4));
    ensure!(&head[16..20] == b"JSON", "GLB doesn't start with a JSON chunk");
    let json_len = u64::from(u32_at(&head, 12));
    ensure!(20 + json_len <= total, "GLB is truncated");
    let text = read(20, json_len as usize)?;
    let json = match serde_json::from_slice(&text).context("GLB JSON doesn't parse")? {
        Value::Object(m) => m,
        _ => bail!("glTF JSON is not an object"),
    };
    let at = (20 + json_len).next_multiple_of(4);
    let mut bin = None;
    if at + 8 <= total {
        let h = read(at, 8)?;
        if &h[4..8] == b"BIN\0" {
            let base = at + 8;
            bin = Some((base, u64::from(u32_at(&h, 0)).min(total - base)));
        }
    }
    Ok((json, bin))
}

/// A GLB in memory, as its JSON and binary chunk: borrowed bytes sliced, owned ones moved out
/// without a copy.
fn glb_in_memory(bytes: Cow<'_, [u8]>) -> Result<(Map<String, Value>, Option<Buf<'_>>)> {
    let total = bytes.len() as u64;
    let (json, bin) = glb_layout(total, |off, len| {
        bytes
            .get(off as usize..off as usize + len)
            .map(<[u8]>::to_vec)
            .context("GLB is truncated")
    })?;
    let bin = bin.map(|(base, len)| {
        let (base, end) = (base as usize, (base + len) as usize);
        Buf::Mem(match bytes {
            Cow::Borrowed(b) => Cow::Borrowed(&b[base..end]),
            Cow::Owned(mut v) => {
                v.truncate(end);
                v.drain(..base);
                Cow::Owned(v)
            }
        })
    });
    Ok((json, bin))
}

/// glTF JSON (a `.gltf`), after any byte-order mark.
fn gltf_json(bytes: &[u8]) -> Result<Map<String, Value>> {
    let text = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    match serde_json::from_slice(text) {
        Ok(Value::Object(m)) => Ok(m),
        _ => bail!("not a glTF file: neither GLB nor glTF JSON"),
    }
}

/// The JSON and binary chunk of a glTF or GLB file, read in pieces.
#[cfg(feature = "fs")]
fn open_file(path: &Path) -> Result<(Map<String, Value>, Option<Buf<'static>>)> {
    let mut file = File::open(path)?;
    let total = file.metadata()?.len();
    // One read for the header, the JSON and the binary chunk's header in the usual case,
    // and the whole file when it's small.
    let mut head = vec![0; usize::try_from(total).unwrap_or(HEAD).min(HEAD)];
    file.read_exact(&mut head)?;
    let glb = head.starts_with(b"glTF");
    if glb && head.len() as u64 == total {
        return glb_in_memory(Cow::Owned(head));
    }
    if !glb {
        file.read_to_end(&mut head)?;
        return Ok((gltf_json(&head)?, None));
    }
    let mut pos = head.len() as u64;
    let (json, bin) = glb_layout(total, |off, len| {
        if let Some(b) = head.get(off as usize..off as usize + len) {
            return Ok(b.to_vec());
        }
        let mut b = vec![0; len];
        file.seek(SeekFrom::Start(off))?;
        file.read_exact(&mut b).context("GLB is truncated")?;
        pos = off + len as u64;
        Ok(b)
    })?;
    drop(head);
    let bin = bin.map(|(base, len)| Buf::File {
        file,
        base,
        len,
        at: pos.wrapping_sub(base),
    });
    Ok((json, bin))
}

impl<'a> Reader<'a> {
    fn open(input: Input<'a>, assets: &'a dyn Assets) -> Result<Self> {
        let (mut json, mut bin) = match input.bytes() {
            Some(b) if b.starts_with(b"glTF") => glb_in_memory(Cow::Borrowed(b))?,
            Some(b) => (gltf_json(b)?, None),
            #[cfg(feature = "fs")]
            None => open_file(input.path)?,
            #[cfg(not(feature = "fs"))]
            None => unreachable!("a model without its bytes needs the `fs` feature"),
        };
        let version = json
            .get("asset")
            .and_then(|a| a.get("version"))
            .and_then(Value::as_str)
            .unwrap_or("");
        ensure!(
            version.starts_with("2."),
            "glTF version {version:?} isn't 2.x (glTF 1.0 isn't read)"
        );
        let required = |name: &str| {
            list(&json, "extensionsRequired")
                .iter()
                .chain(list(&json, "extensionsUsed"))
                .any(|e| e.as_str() == Some(name))
        };
        ensure!(
            !required("KHR_draco_mesh_compression"),
            "glTF uses Draco compression (KHR_draco_mesh_compression), which isn't read"
        );
        let meshopt = container::MESHOPT_EXTENSIONS.iter().any(|e| required(e))
            || list(&json, "bufferViews").iter().any(|v| {
                v.get("extensions")
                    .and_then(Value::as_object)
                    .is_some_and(|x| container::MESHOPT_EXTENSIONS.iter().any(|e| x.contains_key(*e)))
            });
        if meshopt {
            drop(bin);
            let bytes = input.read()?;
            let plain = if bytes.starts_with(b"glTF") {
                container::decompress(&bytes)?
            } else {
                container::pack_glb_decompressed(&bytes, |uri| {
                    external(input, assets, uri).map(Cow::into_owned)
                })?
            };
            drop(bytes);
            (json, bin) = glb_in_memory(Cow::Owned(plain))?;
        }
        let n = |key| list(&json, key).len();
        Ok(Self {
            bufs: (0..n("buffers")).map(|_| None).collect(),
            materials: vec![None; n("materials")],
            images: vec![None; n("images")],
            json,
            bin,
            input,
            assets,
            scratch: Vec::new(),
            default_material: None,
            world: Vec::new(),
            budget: MAX_TRIANGLES,
            points: 0,
            lines: 0,
        })
    }

    fn get(&self, key: &str, i: usize) -> Result<&Value> {
        list(&self.json, key)
            .get(i)
            .with_context(|| format!("glTF {key}[{i}] doesn't exist"))
    }

    fn buf(&mut self, i: usize) -> Result<&mut Buf<'a>> {
        ensure!(i < self.bufs.len(), "glTF buffers[{i}] doesn't exist");
        if self.bufs[i].is_none() {
            let b = self.get("buffers", i)?.clone();
            let declared = b.get("byteLength").and_then(Value::as_u64).unwrap_or(0);
            let buf = match b.get("uri").and_then(Value::as_str) {
                Some(uri) if uri.starts_with("data:") => {
                    Buf::Mem(Cow::Owned(container::data_uri(uri)?.1))
                }
                #[cfg(feature = "fs")]
                Some(uri) if self.input.bytes().is_none() => {
                    let rel = container::percent_decode(uri);
                    let p = self.input.dir().join(checked(&rel)?);
                    let file = File::open(&p).map_err(|e| invalid!("glTF buffer {rel:?}: {e}"))?;
                    let len = file.metadata()?.len();
                    Buf::File {
                        file,
                        base: 0,
                        len,
                        at: u64::MAX,
                    }
                }
                Some(uri) => Buf::Mem(external(self.input, self.assets, uri)?),
                None if i == 0 && self.bin.is_some() => self.bin.take().unwrap(),
                None => bail!("glTF buffer {i} has no data"),
            };
            ensure!(
                buf.len() >= declared,
                "glTF buffer {i} holds {} bytes but declares {declared}",
                buf.len()
            );
            self.bufs[i] = Some(buf);
        }
        Ok(self.bufs[i].as_mut().unwrap())
    }

    /// A buffer view as (buffer, offset, length, stride).
    fn view(&self, i: usize) -> Result<(usize, u64, u64, Option<usize>)> {
        let v = self.get("bufferViews", i)?;
        let buffer = index(v, "buffer").context("glTF buffer view has no buffer")?;
        let offset = v.get("byteOffset").and_then(Value::as_u64).unwrap_or(0);
        let len = v
            .get("byteLength")
            .and_then(Value::as_u64)
            .context("glTF buffer view has no byteLength")?;
        Ok((buffer, offset, len, index(v, "byteStride")))
    }

    /// A buffer view's bytes.
    fn view_bytes(&mut self, i: usize) -> Result<Vec<u8>> {
        let (buffer, offset, len, _) = self.view(i)?;
        let b = self.buf(buffer)?;
        ensure!(offset.saturating_add(len) <= b.len(), "glTF buffer view runs past its buffer");
        let mut out = vec![0; len as usize];
        b.read_at(offset, &mut out)?;
        Ok(out)
    }

    fn acc(&self, i: usize) -> Result<Acc> {
        let a = self.get("accessors", i)?;
        let ctype = a.get("componentType").and_then(Value::as_u64).unwrap_or(0);
        ensure!(
            matches!(ctype, 5120..=5123 | 5125 | 5126),
            "glTF accessor {i} has component type {ctype}"
        );
        let n = match a.get("type").and_then(Value::as_str) {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            Some("MAT4") => 16,
            t => bail!("glTF accessor {i} has type {t:?}, not one read here"),
        };
        Ok(Acc {
            view: index(a, "bufferView"),
            offset: a.get("byteOffset").and_then(Value::as_u64).unwrap_or(0),
            count: index(a, "count").context("glTF accessor has no count")?,
            ctype,
            n,
            normalized: a.get("normalized").and_then(Value::as_bool).unwrap_or(false),
            sparse: a.get("sparse").cloned(),
        })
    }

    /// Calls `f(first, bytes, stride)` over an accessor's elements a block at a time: element
    /// `first + j` is at `bytes[j * stride..]`. Nothing for an accessor without a buffer view.
    fn blocks(&mut self, a: &Acc, mut f: impl FnMut(usize, &[u8], usize)) -> Result<()> {
        let Some(view) = a.view else {
            ensure!(a.count <= MAX_VERTICES, "glTF accessor counts {} elements", a.count);
            return Ok(());
        };
        if a.count == 0 {
            return Ok(());
        }
        let (buffer, voff, vlen, stride) = self.view(view)?;
        let esize = a.esize();
        let stride = stride.filter(|&s| s > 0).unwrap_or(esize);
        ensure!(stride >= esize, "glTF buffer view stride {stride} is under an element's {esize} bytes");
        let span = (a.count as u64 - 1)
            .checked_mul(stride as u64)
            .and_then(|s| s.checked_add(a.offset + esize as u64))
            .filter(|&s| s <= vlen)
            .with_context(|| format!("glTF accessor's {} elements run past their buffer view", a.count))?;
        let start = voff + a.offset;
        let mut scratch = std::mem::take(&mut self.scratch);
        let b = self.buf(buffer)?;
        ensure!(start + span <= b.len(), "glTF buffer view runs past its buffer");
        #[cfg_attr(not(feature = "fs"), allow(irrefutable_let_patterns))]
        if let Buf::Mem(v) = b {
            f(0, &v[start as usize..(start + span) as usize], stride);
            self.scratch = scratch;
            return Ok(());
        }
        let per = (BLOCK / stride).max(1);
        let mut i = 0;
        while i < a.count {
            let k = per.min(a.count - i);
            let len = (k - 1) * stride + esize;
            if scratch.len() < len {
                scratch.resize(len, 0);
            }
            b.read_at(start + (i * stride) as u64, &mut scratch[..len])?;
            f(i, &scratch[..len], stride);
            i += k;
        }
        self.scratch = scratch;
        Ok(())
    }

    /// An accessor's elements as floats, components past the accessor's from `fill`, integer
    /// components normalized when the accessor says so.
    fn read<const N: usize>(&mut self, i: usize, fill: [f32; N]) -> Result<Vec<[f32; N]>> {
        let a = self.acc(i)?;
        self.read_acc(&a, fill)
    }

    fn read_acc<const N: usize>(&mut self, a: &Acc, fill: [f32; N]) -> Result<Vec<[f32; N]>> {
        let mut zero = fill;
        for z in zero.iter_mut().take(a.n) {
            *z = 0.0;
        }
        if a.view.is_none() {
            ensure!(a.count <= MAX_VERTICES, "glTF accessor counts {} elements", a.count);
        }
        let mut out = Vec::new();
        out.try_reserve_exact(a.count)
            .map_err(|_| invalid!("glTF accessor counts {} elements", a.count))?;
        out.resize(a.count, zero);
        let m = a.n.min(N);
        let cs = a.csize();
        let (ctype, normalized) = (a.ctype, a.normalized);
        self.blocks(a, |first, bytes, stride| {
            let out = &mut out[first..];
            if ctype == 5126 && m == N && stride == 4 * N {
                // Tightly packed floats (what nearly every exporter writes): one flat loop.
                for (o, b) in out.as_flattened_mut().iter_mut().zip(bytes.chunks_exact(4)) {
                    *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
                return;
            }
            match (ctype, normalized) {
                (5126, _) => put(out, bytes, stride, cs, m, |b| {
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                }),
                (5121, true) => put(out, bytes, stride, cs, m, |b| f32::from(b[0]) / 255.0),
                (5121, false) => put(out, bytes, stride, cs, m, |b| f32::from(b[0])),
                (5120, true) => put(out, bytes, stride, cs, m, |b| {
                    (f32::from(b[0] as i8) / 127.0).max(-1.0)
                }),
                (5120, false) => put(out, bytes, stride, cs, m, |b| f32::from(b[0] as i8)),
                (5123, true) => put(out, bytes, stride, cs, m, |b| {
                    f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0
                }),
                (5123, false) => put(out, bytes, stride, cs, m, |b| {
                    f32::from(u16::from_le_bytes([b[0], b[1]]))
                }),
                (5122, true) => put(out, bytes, stride, cs, m, |b| {
                    (f32::from(i16::from_le_bytes([b[0], b[1]])) / 32767.0).max(-1.0)
                }),
                (5122, false) => put(out, bytes, stride, cs, m, |b| {
                    f32::from(i16::from_le_bytes([b[0], b[1]]))
                }),
                _ => put(out, bytes, stride, cs, m, |b| {
                    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32
                }),
            }
        })?;
        if let Some(s) = a.sparse.clone() {
            let (at, values) = self.sparse(a, &s)?;
            let values = self.read_acc(&values, fill)?;
            for (k, v) in at.into_iter().zip(values) {
                if let Some(o) = out.get_mut(k as usize) {
                    *o = v;
                }
            }
        }
        Ok(out)
    }

    /// An index accessor's values.
    fn read_indices(&mut self, i: usize) -> Result<Vec<u32>> {
        let a = self.acc(i)?;
        ensure!(
            a.n == 1 && matches!(a.ctype, 5121 | 5123 | 5125),
            "glTF index accessor {i} isn't unsigned integer scalars"
        );
        self.read_u32s(&a)
    }

    fn read_u32s(&mut self, a: &Acc) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        out.try_reserve_exact(a.count)
            .map_err(|_| invalid!("glTF accessor counts {} elements", a.count))?;
        out.resize(a.count, 0u32);
        let ctype = a.ctype;
        self.blocks(a, |first, bytes, stride| {
            let out = &mut out[first..];
            match ctype {
                5121 => {
                    for (o, e) in out.iter_mut().zip(bytes.chunks(stride)) {
                        *o = u32::from(e[0]);
                    }
                }
                5123 if stride == 2 => {
                    for (o, e) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                        *o = u32::from(u16::from_le_bytes([e[0], e[1]]));
                    }
                }
                5125 if stride == 4 => {
                    for (o, e) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                        *o = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                    }
                }
                5123 => {
                    for (o, e) in out.iter_mut().zip(bytes.chunks(stride)) {
                        *o = u32::from(u16::from_le_bytes([e[0], e[1]]));
                    }
                }
                _ => {
                    for (o, e) in out.iter_mut().zip(bytes.chunks(stride)) {
                        *o = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                    }
                }
            }
        })?;
        if let Some(s) = a.sparse.clone() {
            let (at, values) = self.sparse(a, &s)?;
            let values = self.read_u32s(&values)?;
            for (k, v) in at.into_iter().zip(values) {
                if let Some(o) = out.get_mut(k as usize) {
                    *o = v;
                }
            }
        }
        Ok(out)
    }

    /// A sparse accessor's indices, and an accessor for its values.
    fn sparse(&mut self, a: &Acc, s: &Value) -> Result<(Vec<u32>, Acc)> {
        let count = index(s, "count").context("glTF sparse accessor has no count")?;
        ensure!(count <= a.count, "glTF sparse accessor replaces more elements than it has");
        let part = |key: &str| -> Result<(usize, u64)> {
            let p = s.get(key).with_context(|| format!("glTF sparse accessor has no {key}"))?;
            Ok((
                index(p, "bufferView").with_context(|| format!("glTF sparse {key} has no bufferView"))?,
                p.get("byteOffset").and_then(Value::as_u64).unwrap_or(0),
            ))
        };
        let (iv, ioff) = part("indices")?;
        let ictype = s["indices"].get("componentType").and_then(Value::as_u64).unwrap_or(0);
        ensure!(matches!(ictype, 5121 | 5123 | 5125), "glTF sparse indices' component type {ictype}");
        let idx = Acc {
            view: Some(iv),
            offset: ioff,
            count,
            ctype: ictype,
            n: 1,
            normalized: false,
            sparse: None,
        };
        let at = self.read_u32s(&idx)?;
        let (vv, voff) = part("values")?;
        let values = Acc {
            view: Some(vv),
            offset: voff,
            sparse: None,
            count,
            ..a.clone()
        };
        Ok((at, values))
    }

    /// The scene's meshes, where they are drawn.
    fn uses(&mut self) -> Result<Vec<Use>> {
        let nodes = list(&self.json, "nodes").to_vec();
        let mut world = vec![None::<M4>; nodes.len()];
        // Parents first: every node reachable from a root takes its parent's transform.
        let mut child = vec![false; nodes.len()];
        for n in &nodes {
            for c in list_of(n, "children") {
                if let Some(f) = child.get_mut(c) {
                    *f = true;
                }
            }
        }
        let mut stack: Vec<(usize, M4)> = (0..nodes.len())
            .filter(|&i| !child[i])
            .map(|i| (i, IDENTITY))
            .collect();
        while let Some((i, parent)) = stack.pop() {
            if world[i].is_some() {
                continue;
            }
            let m = mul(&parent, &local(&nodes[i]));
            world[i] = Some(m);
            for c in list_of(&nodes[i], "children") {
                if c < nodes.len() && world[c].is_none() {
                    stack.push((c, m));
                }
            }
        }

        let scenes = list(&self.json, "scenes");
        let roots: Vec<usize> = if scenes.is_empty() {
            if nodes.is_empty() {
                // No nodes at all: every mesh, as it is.
                return Ok((0..list(&self.json, "meshes").len())
                    .map(|mesh| Use {
                        mesh,
                        weights: Vec::new(),
                        skin: None,
                        transforms: vec![IDENTITY],
                    })
                    .collect());
            }
            (0..nodes.len()).filter(|&i| !child[i]).collect()
        } else {
            let s = self.json.get("scene").and_then(Value::as_u64).unwrap_or(0) as usize;
            list_of(scenes.get(s).context("glTF default scene doesn't exist")?, "nodes")
        };

        let mut seen = vec![false; nodes.len()];
        let mut order = Vec::new();
        let mut stack: Vec<usize> = roots.into_iter().rev().collect();
        while let Some(i) = stack.pop() {
            if i >= nodes.len() || seen[i] {
                continue;
            }
            seen[i] = true;
            order.push(i);
            stack.extend(list_of(&nodes[i], "children").into_iter().rev());
        }

        let mut uses: Vec<Use> = Vec::new();
        for i in order {
            let n = &nodes[i];
            let Some(mesh) = index(n, "mesh") else { continue };
            let skin = index(n, "skin");
            let weights: Vec<f32> = n
                .get("weights")
                .or_else(|| list(&self.json, "meshes").get(mesh).and_then(|m| m.get("weights")))
                .and_then(Value::as_array)
                .map(|w| w.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
                .unwrap_or_default();
            let w = world[i].unwrap_or(IDENTITY);
            let transforms = if skin.is_some() {
                // A skinned mesh is placed by its joints, not its node.
                vec![IDENTITY]
            } else {
                self.instances(n, &w)?
            };
            let same = uses.iter_mut().find(|u| {
                u.mesh == mesh
                    && u.skin == skin
                    && u.weights.iter().map(|x| x.to_bits()).eq(weights.iter().map(|x| x.to_bits()))
            });
            match same {
                Some(_) if skin.is_some() => {}
                Some(u) => u.transforms.extend(transforms),
                None => uses.push(Use {
                    mesh,
                    weights,
                    skin,
                    transforms,
                }),
            }
        }
        // Joints need every node's transform, in the scene or not.
        self.world = world.into_iter().map(|w| w.unwrap_or(IDENTITY)).collect();
        Ok(uses)
    }

    /// A node's transforms: its own, or one per `EXT_mesh_gpu_instancing` instance.
    fn instances(&mut self, node: &Value, world: &M4) -> Result<Vec<M4>> {
        let Some(attrs) = node
            .get("extensions")
            .and_then(|e| e.get("EXT_mesh_gpu_instancing"))
            .and_then(|e| e.get("attributes"))
        else {
            return Ok(vec![*world]);
        };
        let get = |k: &str| index(attrs, k);
        let t = get("TRANSLATION").map(|a| self.read(a, [0.0; 3])).transpose()?;
        let r = get("ROTATION").map(|a| self.read(a, [0.0, 0.0, 0.0, 1.0])).transpose()?;
        let s = get("SCALE").map(|a| self.read(a, [1.0; 3])).transpose()?;
        let count = [t.as_ref().map(Vec::len), r.as_ref().map(Vec::len), s.as_ref().map(Vec::len)]
            .into_iter()
            .flatten()
            .min()
            .unwrap_or(0);
        ensure!(count <= MAX_TRIANGLES, "glTF node has {count} instances");
        Ok((0..count)
            .map(|k| {
                let t = t.as_ref().map_or([0.0; 3], |v| v[k].map(f64::from));
                let r = r.as_ref().map_or([0.0, 0.0, 0.0, 1.0], |v| v[k].map(f64::from));
                let s = s.as_ref().map_or([1.0; 3], |v| v[k].map(f64::from));
                mul(world, &trs(t, r, s))
            })
            .collect())
    }
}

/// `node[key]` as indices.
fn list_of(v: &Value, key: &str) -> Vec<usize> {
    v.get(key)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).map(|i| i as usize).collect())
        .unwrap_or_default()
}

/// Writes `m` components of each element in `bytes` into `out`.
#[inline]
fn put<const N: usize>(
    out: &mut [[f32; N]],
    bytes: &[u8],
    stride: usize,
    csize: usize,
    m: usize,
    conv: impl Fn(&[u8]) -> f32,
) {
    if m == N {
        // Every component given (the usual case): a loop the compiler unrolls.
        for (o, e) in out.iter_mut().zip(bytes.chunks(stride)) {
            if e.len() >= N * csize {
                *o = std::array::from_fn(|c| conv(&e[c * csize..]));
            }
        }
        return;
    }
    for (o, e) in out.iter_mut().zip(bytes.chunks(stride)) {
        for (c, x) in o.iter_mut().enumerate().take(m) {
            *x = conv(&e[c * csize..]);
        }
    }
}

fn mul(a: &M4, b: &M4) -> M4 {
    let mut r = [0.0; 16];
    for c in 0..4 {
        for row in 0..4 {
            r[c * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[c * 4 + k]).sum();
        }
    }
    r
}

/// Translation, rotation (a quaternion x y z w) and scale as one matrix.
fn trs(t: [f64; 3], q: [f64; 4], s: [f64; 3]) -> M4 {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let [x, y, z, w] = if len > 0.0 { q.map(|c| c / len) } else { [0.0, 0.0, 0.0, 1.0] };
    let r = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ];
    let mut m = IDENTITY;
    for col in 0..3 {
        for row in 0..3 {
            m[col * 4 + row] = r[row][col] * s[col];
        }
    }
    m[12..15].copy_from_slice(&t);
    m
}

/// A node's own transform.
fn local(node: &Value) -> M4 {
    if let Some(m) = node.get("matrix").and_then(Value::as_array).filter(|m| m.len() == 16) {
        let mut out = IDENTITY;
        for (o, x) in out.iter_mut().zip(m) {
            *o = x.as_f64().unwrap_or(*o);
        }
        return out;
    }
    trs(
        floats(node.get("translation"), [0.0; 3]),
        floats(node.get("rotation"), [0.0, 0.0, 0.0, 1.0]),
        floats(node.get("scale"), [1.0; 3]),
    )
}

/// The upper 3x3's determinant.
fn det3(m: &M4) -> f64 {
    m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2])
        + m[8] * (m[1] * m[6] - m[5] * m[2])
}

/// The matrix normals go through: the upper 3x3's cofactors (its inverse transpose, scaled),
/// signed so a mirror doesn't turn normals inward. Row-major.
fn normal_matrix(m: &M4) -> [[f64; 3]; 3] {
    let a = |r: usize, c: usize| m[c * 4 + r];
    let sign = if det3(m) < 0.0 { -1.0 } else { 1.0 };
    let mut n = [[0.0; 3]; 3];
    for (r, row) in n.iter_mut().enumerate() {
        for (c, x) in row.iter_mut().enumerate() {
            let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
            let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
            *x = sign * (a(r1, c1) * a(r2, c2) - a(r1, c2) * a(r2, c1));
        }
    }
    n
}

fn apply_point(m: &M4, p: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = p.map(f64::from);
    [0, 1, 2].map(|r| (m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r]) as f32)
}

fn apply_normal(n: &[[f64; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = v.map(f64::from);
    let o = [0, 1, 2].map(|r| n[r][0] * x + n[r][1] * y + n[r][2] * z);
    let len = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
    if len > 0.0 {
        o.map(|c| (c / len) as f32)
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn normalize(v: &mut [f32; 3]) {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        for c in v.iter_mut() {
            *c /= len;
        }
    } else {
        *v = [0.0, 0.0, 1.0];
    }
}

/// One primitive's geometry before it is placed.
struct Geometry {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

/// Triangle lists from a primitive's indices, by mode (4 triangles, 5 strip, 6 fan), per the
/// glTF 2.0 specification's section 3.7.2.1.
fn triangles(mode: u64, idx: Vec<u32>) -> Vec<u32> {
    match mode {
        5 => {
            let n = idx.len().saturating_sub(2);
            let mut out = Vec::with_capacity(n * 3);
            for i in 0..n {
                if i % 2 == 0 {
                    out.extend_from_slice(&[idx[i], idx[i + 1], idx[i + 2]]);
                } else {
                    out.extend_from_slice(&[idx[i], idx[i + 2], idx[i + 1]]);
                }
            }
            out
        }
        6 => {
            let n = idx.len().saturating_sub(2);
            let mut out = Vec::with_capacity(n * 3);
            for i in 0..n {
                out.extend_from_slice(&[idx[i + 1], idx[i + 2], idx[0]]);
            }
            out
        }
        _ => {
            let mut idx = idx;
            idx.truncate(idx.len() / 3 * 3);
            idx
        }
    }
}

impl Reader<'_> {
    fn mesh(&mut self) -> Result<Option<Mesh>> {
        let uses = self.uses()?;
        let mut mesh = Mesh::default();
        for u in &uses {
            let m = self.get("meshes", u.mesh)?.clone();
            for p in list_of_values(&m, "primitives") {
                self.primitive(&p, u, &mut mesh)?;
            }
        }
        Ok((mesh.triangles() > 0).then_some(mesh))
    }

    fn primitive(&mut self, p: &Value, u: &Use, mesh: &mut Mesh) -> Result<()> {
        ensure!(
            p.get("extensions")
                .and_then(|e| e.get("KHR_draco_mesh_compression"))
                .is_none(),
            "glTF uses Draco compression (KHR_draco_mesh_compression), which isn't read"
        );
        let mode = p.get("mode").and_then(Value::as_u64).unwrap_or(4);
        let attrs = p.get("attributes").cloned().unwrap_or(Value::Null);
        let Some(pos) = index(&attrs, "POSITION") else {
            return Ok(());
        };
        if !matches!(mode, 4..=6) {
            let n = match index(p, "indices") {
                Some(i) => self.acc(i)?.count,
                None => self.acc(pos)?.count,
            };
            match mode {
                0 => self.points += n * u.transforms.len(),
                1 => self.lines += n / 2 * u.transforms.len(),
                2 => self.lines += n * u.transforms.len(),
                3 => self.lines += n.saturating_sub(1) * u.transforms.len(),
                _ => {}
            }
            return Ok(());
        }

        // Check the triangle count before reading any vertices.
        let corners = match index(p, "indices") {
            Some(i) => self.acc(i)?.count,
            None => self.acc(pos)?.count,
        };
        let tris = if mode == 4 { corners / 3 } else { corners.saturating_sub(2) };
        let total = tris.saturating_mul(u.transforms.len());
        ensure!(total <= self.budget, "too many triangles to convert");

        let mut positions = self.read(pos, [0.0; 3])?;
        let n = positions.len();
        let idx = match index(p, "indices") {
            Some(i) => self.read_indices(i)?,
            None => (0..n as u32).collect(),
        };
        let mut indices = triangles(mode, idx);
        if indices.iter().copied().max().is_some_and(|m| m as usize >= n) {
            // Drop triangles that name vertices that don't exist.
            let mut w = 0;
            for t in 0..indices.len() / 3 {
                let tri = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
                if tri.iter().all(|&i| (i as usize) < n) {
                    indices[w..w + 3].copy_from_slice(&tri);
                    w += 3;
                }
            }
            indices.truncate(w);
        }
        if indices.is_empty() {
            return Ok(());
        }

        let material = index(p, "material");
        let (set, uv_matrix) = self.uv_set(material);
        let attr = |name: &str| index(&attrs, name);
        let mut normals = match attr("NORMAL") {
            Some(a) => self.read(a, [0.0; 3])?,
            None => Vec::new(),
        };
        let mut uvs = match attr(&format!("TEXCOORD_{set}")) {
            Some(a) => self.read(a, [0.0; 2])?,
            None => Vec::new(),
        };
        let colors = match attr("COLOR_0") {
            Some(a) => self.read(a, [0.0, 0.0, 0.0, 1.0])?,
            None => Vec::new(),
        };
        // Attributes that don't match the positions are dropped rather than trusted.
        if normals.len() != n {
            normals = Vec::new();
        }
        if uvs.len() != n {
            uvs = Vec::new();
        }
        let colors = if colors.len() == n { colors } else { Vec::new() };
        if let Some(t) = uv_matrix {
            for uv in &mut uvs {
                let [u0, v0] = *uv;
                *uv = [t[0] * u0 + t[1] * v0 + t[2], t[3] * u0 + t[4] * v0 + t[5]];
            }
        }

        // Morph targets at their default weights.
        for (k, target) in list_of_values(p, "targets").iter().enumerate() {
            let w = u.weights.get(k).copied().unwrap_or(0.0);
            if w == 0.0 {
                continue;
            }
            if let Some(a) = index(target, "POSITION") {
                for (v, d) in positions.iter_mut().zip(self.read(a, [0.0; 3])?) {
                    for c in 0..3 {
                        v[c] += w * d[c];
                    }
                }
            }
            if !normals.is_empty()
                && let Some(a) = index(target, "NORMAL")
            {
                for (v, d) in normals.iter_mut().zip(self.read(a, [0.0; 3])?) {
                    for c in 0..3 {
                        v[c] += w * d[c];
                    }
                }
            }
        }
        if let Some(skin) = u.skin {
            self.skin(skin, &attrs, &mut positions, &mut normals)?;
        } else if !u.weights.is_empty() {
            normals.iter_mut().for_each(normalize);
        }

        let g = Geometry {
            positions,
            normals,
            uvs,
            colors,
            indices,
        };
        let material = self.material(material, mesh);
        self.budget -= g.indices.len() / 3 * u.transforms.len();
        mesh.prims.push(place(g, &u.transforms, material));
        Ok(())
    }

    /// Poses a skinned primitive: each vertex goes through its joints' matrices, blended by
    /// weight.
    fn skin(
        &mut self,
        skin: usize,
        attrs: &Value,
        positions: &mut [[f32; 3]],
        normals: &mut [[f32; 3]],
    ) -> Result<()> {
        let s = self.get("skins", skin)?.clone();
        let joints = list_of(&s, "joints");
        let ibm = match index(&s, "inverseBindMatrices") {
            Some(a) => self.read(a, IDENTITY.map(|x| x as f32))?,
            None => Vec::new(),
        };
        let mats: Vec<M4> = joints
            .iter()
            .enumerate()
            .map(|(k, &j)| {
                let w = self.world.get(j).copied().unwrap_or(IDENTITY);
                let b = ibm.get(k).map_or(IDENTITY, |m| m.map(f64::from));
                mul(&w, &b)
            })
            .collect();
        let (Some(ja), Some(wa)) = (index(attrs, "JOINTS_0"), index(attrs, "WEIGHTS_0")) else {
            return Ok(());
        };
        // Joint indices are integers, read as floats exactly.
        let ji = self.read(ja, [0.0; 4])?;
        let wi = self.read(wa, [0.0; 4])?;
        if ji.len() != positions.len() || wi.len() != positions.len() {
            return Ok(());
        }
        for (v, (j, w)) in ji.iter().zip(&wi).enumerate() {
            let sum: f32 = w.iter().sum();
            if sum <= 0.0 {
                continue;
            }
            let mut m = [0.0; 16];
            for k in 0..4 {
                if w[k] == 0.0 {
                    continue;
                }
                let jm = mats.get(j[k] as usize).unwrap_or(&IDENTITY);
                let f = f64::from(w[k] / sum);
                for (o, x) in m.iter_mut().zip(jm) {
                    *o += f * x;
                }
            }
            positions[v] = apply_point(&m, positions[v]);
            if let Some(nv) = normals.get_mut(v) {
                *nv = apply_normal(&normal_matrix(&m), *nv);
            }
        }
        Ok(())
    }

    /// The texture coordinate set a material's color texture reads, and its
    /// `KHR_texture_transform` as a 2x3 matrix (row-major).
    fn uv_set(&self, material: Option<usize>) -> (u64, Option<[f32; 6]>) {
        let Some(info) = material
            .and_then(|m| list(&self.json, "materials").get(m))
            .and_then(color_texture)
        else {
            return (0, None);
        };
        let mut set = info.get("texCoord").and_then(Value::as_u64).unwrap_or(0);
        let Some(t) = info
            .get("extensions")
            .and_then(|e| e.get("KHR_texture_transform"))
        else {
            return (set, None);
        };
        if let Some(s) = t.get("texCoord").and_then(Value::as_u64) {
            set = s;
        }
        let [ox, oy] = f32s(t.get("offset"), [0.0; 2]);
        let [sx, sy] = f32s(t.get("scale"), [1.0; 2]);
        let r = t.get("rotation").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let (sin, cos) = r.sin_cos();
        // translation * rotation * scale, as the extension defines them.
        (set, Some([cos * sx, sin * sy, ox, -sin * sx, cos * sy, oy]))
    }

    /// The mesh material for glTF material `i` (or glTF's default material).
    fn material(&mut self, i: Option<usize>, mesh: &mut Mesh) -> usize {
        let slot = match i {
            Some(i) if i < self.materials.len() => self.materials[i],
            _ => self.default_material,
        };
        if let Some(s) = slot {
            return s;
        }
        let mut m = Material {
            name: String::new(),
            color: [1.0; 4],
            color_image: None,
            normal_image: None,
            emissive: [0.0; 3],
            metallic: 1.0,
            roughness: 1.0,
            double_sided: false,
        };
        let def = i.and_then(|i| list(&self.json, "materials").get(i)).cloned();
        if let Some(d) = &def {
            m.name = d.get("name").and_then(Value::as_str).unwrap_or("").to_owned();
            let sg = d
                .get("extensions")
                .and_then(|e| e.get("KHR_materials_pbrSpecularGlossiness"));
            match (d.get("pbrMetallicRoughness"), sg) {
                (None, Some(sg)) => {
                    m.color = f32s(sg.get("diffuseFactor"), [1.0; 4]);
                    m.metallic = 0.0;
                    m.roughness =
                        1.0 - sg.get("glossinessFactor").and_then(Value::as_f64).unwrap_or(1.0) as f32;
                }
                (pbr, _) => {
                    let pbr = pbr.cloned().unwrap_or(Value::Null);
                    m.color = f32s(pbr.get("baseColorFactor"), [1.0; 4]);
                    m.metallic = pbr.get("metallicFactor").and_then(Value::as_f64).unwrap_or(1.0) as f32;
                    m.roughness = pbr.get("roughnessFactor").and_then(Value::as_f64).unwrap_or(1.0) as f32;
                }
            }
            m.emissive = f32s(d.get("emissiveFactor"), [0.0; 3]);
            m.double_sided = d.get("doubleSided").and_then(Value::as_bool).unwrap_or(false);
            if d.get("alphaMode").and_then(Value::as_str).unwrap_or("OPAQUE") == "OPAQUE" {
                m.color[3] = 1.0;
            }
            if let Some(t) = color_texture(d).and_then(|t| index(t, "index")) {
                m.color_image = self.texture(t, mesh);
            }
            if let Some(t) = d.get("normalTexture").and_then(|t| index(t, "index")) {
                m.normal_image = self.texture(t, mesh);
            }
        }
        mesh.materials.push(m);
        let s = mesh.materials.len() - 1;
        match i {
            Some(i) if i < self.materials.len() => self.materials[i] = Some(s),
            _ => self.default_material = Some(s),
        }
        s
    }

    /// The mesh image for glTF texture `t`: its source, or failing that one an extension
    /// names (`EXT_texture_webp`, `KHR_texture_basisu`, ...), if an image can be made of it.
    fn texture(&mut self, t: usize, mesh: &mut Mesh) -> Option<usize> {
        let tex = list(&self.json, "textures").get(t)?.clone();
        let mut sources: Vec<usize> = index(&tex, "source").into_iter().collect();
        if let Some(ext) = tex.get("extensions").and_then(Value::as_object) {
            sources.extend(ext.values().filter_map(|e| index(e, "source")));
        }
        sources.into_iter().find_map(|s| self.image(s, mesh))
    }

    fn image(&mut self, s: usize, mesh: &mut Mesh) -> Option<usize> {
        if let Some(done) = *self.images.get(s)? {
            return done;
        }
        let img = list(&self.json, "images").get(s)?.clone();
        let mime = img.get("mimeType").and_then(Value::as_str);
        let named = |mime: Option<&str>| {
            PathBuf::from(match mime {
                Some("image/jpeg") => "image.jpg",
                Some("image/webp") => "image.webp",
                Some("image/ktx2") => "image.ktx2",
                _ => "image.png",
            })
        };
        let image: Option<Image> = if let Some(v) = index(&img, "bufferView") {
            self.view_bytes(v)
                .ok()
                .and_then(|b| self.assets.image(&b, &named(mime)))
        } else {
            match img.get("uri").and_then(Value::as_str) {
                Some(uri) if uri.starts_with("data:") => container::data_uri(uri)
                    .ok()
                    .and_then(|(m, b)| self.assets.image(&b, &named(m.as_deref().or(mime)))),
                Some(uri) => self
                    .assets
                    .find(&container::percent_decode(uri), self.input.dir())
                    .and_then(|p| self.assets.load_image(&p)),
                None => None,
            }
        };
        let out = image.map(|i| {
            mesh.images.push(i);
            mesh.images.len() - 1
        });
        self.images[s] = Some(out);
        out
    }
}

/// A material's color texture info: `baseColorTexture`, or the specular-glossiness diffuse.
fn color_texture(m: &Value) -> Option<&Value> {
    m.get("pbrMetallicRoughness")
        .and_then(|p| p.get("baseColorTexture"))
        .or_else(|| {
            m.get("extensions")?
                .get("KHR_materials_pbrSpecularGlossiness")?
                .get("diffuseTexture")
        })
}

fn list_of_values(v: &Value, key: &str) -> Vec<Value> {
    v.get(key).and_then(Value::as_array).cloned().unwrap_or_default()
}

/// A primitive's geometry at each of `transforms`, as one primitive.
fn place(g: Geometry, transforms: &[M4], material: usize) -> Prim {
    if let [t] = transforms
        && *t == IDENTITY
    {
        return Prim {
            positions: g.positions,
            normals: g.normals,
            uvs: g.uvs,
            colors: g.colors,
            indices: g.indices,
            material,
        };
    }
    let n = g.positions.len();
    let k = transforms.len();
    let mut p = Prim {
        positions: Vec::with_capacity(n * k),
        normals: Vec::with_capacity(g.normals.len() * k),
        uvs: Vec::with_capacity(g.uvs.len() * k),
        colors: Vec::with_capacity(g.colors.len() * k),
        indices: Vec::with_capacity(g.indices.len() * k),
        material,
    };
    for t in transforms {
        let base = p.positions.len() as u32;
        p.positions.extend(g.positions.iter().map(|&v| apply_point(t, v)));
        let nm = normal_matrix(t);
        p.normals.extend(g.normals.iter().map(|&v| apply_normal(&nm, v)));
        p.uvs.extend_from_slice(&g.uvs);
        p.colors.extend_from_slice(&g.colors);
        let flip = det3(t) < 0.0;
        for tri in g.indices.chunks_exact(3) {
            let [a, b, c] = [tri[0] + base, tri[1] + base, tri[2] + base];
            p.indices.extend_from_slice(&if flip { [a, c, b] } else { [a, b, c] });
        }
    }
    p
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use crate::assets::Folder;
    use crate::mesh::{Material, Mesh, Prim, to_glb};

    /// A bumpy grid with normals, UVs and colors, in two materials.
    fn grid(n: u32) -> Mesh {
        let mut prims = Vec::new();
        for m in 0..2u32 {
            let mut p = Prim { material: m as usize, ..Prim::default() };
            for y in 0..=n {
                for x in 0..=n {
                    let (u, v) = (x as f32 / n as f32, y as f32 / n as f32);
                    p.positions.push([u + m as f32, (u * 7.0).sin() * 0.1, v]);
                    p.normals.push([0.0, 1.0, 0.0]);
                    p.uvs.push([u, v]);
                    p.colors.push([u, v, 0.5, 1.0]);
                }
            }
            for y in 0..n {
                for x in 0..n {
                    let i = y * (n + 1) + x;
                    p.indices.extend_from_slice(&[i, i + n + 1, i + 1, i + 1, i + n + 1, i + n + 2]);
                }
            }
            prims.push(p);
        }
        let mut red = Material::plain("Red", [1.0, 0.0, 0.0, 1.0]);
        red.metallic = 0.5;
        red.roughness = 0.25;
        red.emissive = [0.1, 0.2, 0.3];
        red.double_sided = true;
        Mesh { prims, materials: vec![red, Material::plain("Blue", [0.0, 0.0, 1.0, 0.5])], images: Vec::new() }
    }

    /// Each triangle's corners, rotated to start at its smallest, so triangles that begin at
    /// another corner (as meshopt's triangle codec may write them) compare equal.
    fn triangles(p: &Prim) -> Vec<[[u32; 12]; 3]> {
        let key = |i: u32| {
            let i = i as usize;
            let f = |v: &[f32]| v.iter().map(|c| c.to_bits()).collect::<Vec<_>>();
            let mut k = [0u32; 12];
            let all = [f(&p.positions[i]), f(&p.normals[i]), f(&p.uvs[i]), f(&p.colors[i])].concat();
            k.copy_from_slice(&all);
            k
        };
        let mut out: Vec<_> = p
            .indices
            .chunks(3)
            .map(|t| {
                let c = [key(t[0]), key(t[1]), key(t[2])];
                let r = (0..3).min_by_key(|&s| c[s]).unwrap();
                [c[r], c[(r + 1) % 3], c[(r + 2) % 3]]
            })
            .collect();
        out.sort();
        out
    }

    fn load(dir: &std::path::Path, name: &str, bytes: &[u8]) -> Mesh {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        let ext = name.rsplit('.').next().unwrap();
        let mesh = crate::formats::load(&path, ext, &Folder::new(dir)).unwrap().mesh.unwrap();
        // From memory, with the folder's files as the assets, it reads the same.
        let mut memory = crate::assets::Memory::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let p = entry.unwrap().path();
            memory.insert(&p.file_name().unwrap().to_string_lossy(), std::fs::read(&p).unwrap());
        }
        let in_memory = crate::formats::load_bytes(bytes, name, ext, &memory).unwrap().mesh.unwrap();
        assert!(to_glb(&mesh).unwrap() == to_glb(&in_memory).unwrap(), "{name} reads differently from memory");
        mesh
    }

    fn same(a: &Mesh, b: &Mesh) {
        assert_eq!(a.prims.len(), b.prims.len());
        for (pa, pb) in a.prims.iter().zip(&b.prims) {
            assert_eq!(triangles(pa), triangles(pb));
            assert_eq!(pa.material, pb.material);
        }
        for (ma, mb) in a.materials.iter().zip(&b.materials) {
            assert_eq!(
                (&ma.name, ma.color, ma.emissive, ma.metallic, ma.roughness, ma.double_sided),
                (&mb.name, mb.color, mb.emissive, mb.metallic, mb.roughness, mb.double_sided)
            );
        }
    }

    #[test]
    fn reads_back_what_to_glb_writes() {
        let dir = tempfile::tempdir().unwrap();
        let mesh = grid(30);
        let glb = to_glb(&mesh).unwrap();
        same(&mesh, &load(dir.path(), "plain.glb", &glb));
    }

    #[test]
    fn streamed_glb_cut_short_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let glb = to_glb(&grid(60)).unwrap();
        assert!(glb.len() > 2 * super::HEAD);
        let path = dir.path().join("cut.glb");
        std::fs::write(&path, &glb[..glb.len() - 1000]).unwrap();
        let err = crate::formats::load(&path, "glb", &Folder::new(dir.path())).unwrap_err();
        assert!(err.to_string().contains("declares"), "{err}");
    }

    #[test]
    fn reads_meshopt_compressed_glb() {
        let dir = tempfile::tempdir().unwrap();
        let mesh = grid(30);
        let packed = crate::gltf::compress(&to_glb(&mesh).unwrap()).unwrap();
        same(&mesh, &load(dir.path(), "packed.glb", &packed));
    }

    #[test]
    fn reads_meshopt_compressed_gltf_with_external_buffers() {
        // A compressed GLB split into a .gltf and its buffer files.
        let dir = tempfile::tempdir().unwrap();
        let mesh = grid(10);
        let packed = crate::gltf::compress(&to_glb(&mesh).unwrap()).unwrap();
        let json_len = u32::from_le_bytes(packed[12..16].try_into().unwrap()) as usize;
        let mut json: serde_json::Value = serde_json::from_slice(&packed[20..20 + json_len]).unwrap();
        let bin = &packed[20 + json_len + 8..];
        json["buffers"][0]["uri"] = "packed.bin".into();
        std::fs::write(dir.path().join("packed.bin"), bin).unwrap();
        same(&mesh, &load(dir.path(), "packed.gltf", &serde_json::to_vec(&json).unwrap()));
    }
}

