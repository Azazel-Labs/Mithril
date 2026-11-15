//! glTF 2.0, `.gltf` and `.glb`, read as the [`Document`] it already is. Nothing is decoded:
//! extensions (known or not), extras, and meshopt or Draco compression pass through as they are.
//!
//! - A GLB file isn't read whole: its binary chunk stays in the file, read as needed. A GLB in
//!   memory is used in place.
//! - External buffers must sit inside the model's folder; for a model in memory they come from
//!   [`Assets::read`]. A meshopt fallback buffer that isn't there holds nothing: views on it
//!   decode from their compressed data.
//! - Images named by URI (a data URI, or a file found as buffers are) move into a buffer, so the
//!   document stands on its own. One that isn't found keeps its URI.

use std::borrow::Cow;
#[cfg(feature = "fs")]
use std::fs::File;
#[cfg(feature = "fs")]
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path};

use serde_json::{Map, Value, json};

use super::Input;
use crate::assets::Assets;
#[cfg(feature = "fs")]
use crate::document::FileSpan;
use crate::document::{Blob, Document};
use crate::gltf as container;
use crate::num::usize_of;
use crate::{Context as _, Result};

/// How much of the file the first read takes.
#[cfg(feature = "fs")]
const HEAD: usize = 64 * 1024;

/// Reads the glTF or GLB `input`.
pub(crate) fn load<'a>(input: Input<'a>, assets: &'a dyn Assets) -> Result<Document<'a>> {
    let (json, mut bin) = match input.bytes() {
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
    let mut doc = Document {
        json,
        buffers: Vec::new(),
        facts: Map::new(),
    };
    let n = doc.json.get("buffers").and_then(Value::as_array).map_or(0, Vec::len);
    for i in 0..n {
        let b = &mut doc.json["buffers"][i];
        let declared = b.get("byteLength").and_then(Value::as_u64).unwrap_or(0);
        let fallback = container::is_fallback(b);
        let blob = match b.get("uri").and_then(Value::as_str) {
            Some(uri) if uri.starts_with("data:") => Blob::Bytes(Cow::Owned(container::data_uri(uri)?.1)),
            Some(uri) => match external(input, assets, uri) {
                Ok(blob) => blob,
                Err(_) if fallback => Blob::Missing(declared),
                Err(e) => return Err(e),
            },
            None if i == 0 && bin.is_some() => bin.take().expect("checked"),
            None if fallback => Blob::Missing(declared),
            None => bail!("glTF buffer {i} has no data"),
        };
        ensure!(
            blob.len() >= declared,
            "glTF buffer {i} holds {} bytes but declares {declared}",
            blob.len()
        );
        if let Some(b) = b.as_object_mut() {
            b.remove("uri");
        }
        doc.buffers.push(blob);
    }
    embed_images(&mut doc, input, assets);
    Ok(doc)
}

/// Moves each image named by URI into a buffer of its own, when it's found and its type known.
fn embed_images(doc: &mut Document<'_>, input: Input<'_>, assets: &dyn Assets) {
    let n = doc.json.get("images").and_then(Value::as_array).map_or(0, Vec::len);
    for i in 0..n {
        let img = &doc.json["images"][i];
        let Some(uri) = img.get("uri").and_then(Value::as_str) else {
            continue;
        };
        let found = if uri.starts_with("data:") {
            container::data_uri(uri).ok()
        } else {
            let name = container::percent_decode(uri);
            read_external(input, assets, uri)
                .ok()
                .map(|b| (container::mime_from_name(&name).map(str::to_owned), b.into_owned()))
        };
        let Some((named, bytes)) = found else { continue };
        let Some(mime) = container::sniff_mime(&bytes)
            .map(str::to_owned)
            .or_else(|| img.get("mimeType").and_then(Value::as_str).map(str::to_owned))
            .or(named)
        else {
            continue;
        };
        let len = bytes.len();
        let buffer = doc.push_buffer(bytes);
        let view = doc.push_view(json!({ "buffer": buffer, "byteLength": len }));
        let img = doc.json["images"][i]
            .as_object_mut()
            .expect("an image with a uri is an object");
        img.remove("uri");
        img.insert("bufferView".into(), json!(view));
        img.insert("mimeType".into(), json!(mime));
    }
}

/// `rel` when it's a plain relative path that stays inside the model's folder.
fn checked(rel: &str) -> Result<&str> {
    let safe = !rel.contains(':')
        && Path::new(rel)
            .components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir));
    ensure!(
        safe,
        "refusing to read {rel:?}: not a relative path inside the model's folder"
    );
    Ok(rel)
}

/// An external buffer: from the model's folder, left in its file, or for a model in memory from
/// its assets.
fn external<'a>(input: Input<'a>, assets: &'a dyn Assets, uri: &str) -> Result<Blob<'a>> {
    #[cfg(feature = "fs")]
    if input.bytes().is_none() {
        let rel = container::percent_decode(uri);
        let p = input.dir().join(checked(&rel)?);
        let file = File::open(&p).map_err(|e| invalid!("glTF buffer {rel:?}: {e}"))?;
        let len = file.metadata()?.len();
        return Ok(Blob::File(FileSpan::new(file, 0, len, 0)));
    }
    read_external(input, assets, uri).map(Blob::Bytes)
}

/// An external file the model names by URI, whole: from the model's folder, or for a model in
/// memory from its assets.
fn read_external<'a>(input: Input<'a>, assets: &'a dyn Assets, uri: &str) -> Result<Cow<'a, [u8]>> {
    let rel = container::percent_decode(uri);
    let p = input.dir().join(checked(&rel)?);
    #[cfg(feature = "fs")]
    if input.bytes().is_none() {
        return Ok(Cow::Owned(
            std::fs::read(&p).map_err(|e| invalid!("glTF file {rel:?}: {e}"))?,
        ));
    }
    assets
        .read(&p)
        .with_context(|| format!("glTF file {rel:?} isn't among the assets"))
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
    let text = read(20, usize_of(json_len))?;
    let Value::Object(json) = serde_json::from_slice(&text).context("GLB JSON doesn't parse")? else {
        bail!("glTF JSON is not an object")
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
fn glb_in_memory(bytes: Cow<'_, [u8]>) -> Result<(Map<String, Value>, Option<Blob<'_>>)> {
    let total = bytes.len() as u64;
    let (json, bin) = glb_layout(total, |off, len| {
        bytes
            .get(usize_of(off)..usize_of(off).saturating_add(len))
            .map(<[u8]>::to_vec)
            .context("GLB is truncated")
    })?;
    let bin = bin.map(|(base, len)| {
        let (base, end) = (usize_of(base), usize_of(base + len));
        Blob::Bytes(match bytes {
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

/// The JSON and binary chunk of a glTF or GLB file, read in pieces: the binary chunk stays in
/// the file.
#[cfg(feature = "fs")]
fn open_file(path: &Path) -> Result<(Map<String, Value>, Option<Blob<'static>>)> {
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
        if let Some(b) = head.get(usize_of(off)..usize_of(off).saturating_add(len)) {
            return Ok(b.to_vec());
        }
        let mut b = vec![0; len];
        file.seek(SeekFrom::Start(off))?;
        file.read_exact(&mut b).context("GLB is truncated")?;
        pos = off + len as u64;
        Ok(b)
    })?;
    drop(head);
    let bin = bin.map(|(base, len)| Blob::File(FileSpan::new(file, base, len, pos.wrapping_sub(base))));
    Ok((json, bin))
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use std::path::{Path, PathBuf};

    use serde_json::Value;

    use crate::assets::Folder;
    use crate::document::{Document, Keep};

    /// A GLB's JSON and binary chunk.
    fn split(glb: &[u8]) -> (Value, &[u8]) {
        let n = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let json = serde_json::from_slice(&glb[20..20 + n]).unwrap();
        let bin = glb.get(20 + n + 8..).unwrap_or(&[]);
        (json, bin)
    }

    /// The bytes of each buffer view.
    fn views(json: &Value, bin: &[u8]) -> Vec<Vec<u8>> {
        let empty = Vec::new();
        let list = json["bufferViews"].as_array().unwrap_or(&empty);
        list.iter()
            .map(|v| {
                if v["buffer"].as_u64() != Some(0) {
                    return Vec::new();
                }
                let off = usize::try_from(v["byteOffset"].as_u64().unwrap_or(0)).unwrap();
                let len = usize::try_from(v["byteLength"].as_u64().unwrap()).unwrap();
                bin[off..off + len].to_vec()
            })
            .collect()
    }

    /// The JSON without what writing may restate: views' default `byteOffset` and where in the
    /// binary chunk each view sits.
    fn normalized(mut json: Value) -> Value {
        for v in json["bufferViews"].as_array_mut().into_iter().flatten() {
            let o = v.as_object_mut().unwrap();
            if o.get("buffer").and_then(Value::as_u64) == Some(0) {
                o.remove("byteOffset");
            }
        }
        if let Some(b) = json.get_mut("buffers").and_then(|b| b.get_mut(0)) {
            b.as_object_mut().unwrap().remove("byteLength");
        }
        json
    }

    fn glbs() -> Vec<PathBuf> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus/gltf");
        let mut out = Vec::new();
        for case in std::fs::read_dir(root).unwrap().flatten() {
            for f in std::fs::read_dir(case.path()).unwrap().flatten() {
                if f.path().extension().is_some_and(|e| e == "glb") {
                    out.push(f.path());
                }
            }
        }
        out.sort();
        out
    }

    #[test]
    fn glb_round_trips() {
        let mut n = 0;
        for path in glbs() {
            let dir = path.parent().unwrap();
            let Ok(doc) = crate::formats::load(&path, "glb", &Folder::new(dir), Keep::ALL) else {
                // Cases that are meant to fail.
                continue;
            };
            let original = std::fs::read(&path).unwrap();
            let written = doc.to_glb().unwrap();
            let (a, abin) = split(&original);
            let (b, bbin) = split(&written);
            let name = path.display();
            assert_eq!(normalized(a.clone()), normalized(b.clone()), "{name}: the JSON changed");
            assert!(
                views(&a, abin) == views(&b, bbin),
                "{name}: a buffer view's bytes changed"
            );
            // And again, now exactly.
            let again = crate::formats::load_bytes(&written, "x.glb", "glb", &crate::assets::Memory::new(), Keep::ALL)
                .and_then(|d: Document<'_>| d.to_glb())
                .unwrap();
            assert!(again == written, "{name}: a second round trip changed it");
            n += 1;
        }
        assert!(n > 20, "only {n} GLBs round-tripped");
    }
}
