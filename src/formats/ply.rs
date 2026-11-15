//! PLY, ASCII and binary.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::mesh::{MAX_TRIANGLES, Mesh, Prim, single, smooth_normals};
use crate::{Context as _, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    Le,
    Be,
}

#[derive(Debug, Clone)]
enum PlyProp {
    Scalar {
        name: String,
        ty: String,
    },
    List {
        name: String,
        count_ty: String,
        item_ty: String,
    },
}

#[derive(Debug, Clone)]
struct PlyElement {
    name: String,
    count: usize,
    props: Vec<PlyProp>,
}

fn ply_size(ty: &str) -> Result<usize> {
    Ok(match ty {
        "char" | "uchar" | "int8" | "uint8" => 1,
        "short" | "ushort" | "int16" | "uint16" => 2,
        "int" | "uint" | "float" | "int32" | "uint32" | "float32" => 4,
        "double" | "float64" => 8,
        t => bail!("unknown PLY type {t}"),
    })
}

/// Reads PLY values one at a time, in whichever encoding the file uses.
struct PlyReader<'a> {
    format: PlyFormat,
    bin: &'a [u8],
    pos: usize,
    words: std::str::SplitAsciiWhitespace<'a>,
}

impl PlyReader<'_> {
    fn value(&mut self, ty: &str) -> Result<f64> {
        if self.format == PlyFormat::Ascii {
            return self
                .words
                .next()
                .context("PLY data ends early")?
                .parse::<f64>()
                .context("bad PLY number");
        }
        let n = ply_size(ty)?;
        let b = self
            .bin
            .get(self.pos..self.pos + n)
            .context("PLY data ends early")?;
        self.pos += n;
        let mut a = [0u8; 8];
        a[..n].copy_from_slice(b);
        if self.format == PlyFormat::Be {
            a[..n].reverse();
        }
        Ok(match ty {
            "char" | "int8" => f64::from(a[0] as i8),
            "uchar" | "uint8" => f64::from(a[0]),
            "short" | "int16" => f64::from(i16::from_le_bytes([a[0], a[1]])),
            "ushort" | "uint16" => f64::from(u16::from_le_bytes([a[0], a[1]])),
            "int" | "int32" => f64::from(i32::from_le_bytes([a[0], a[1], a[2], a[3]])),
            "uint" | "uint32" => f64::from(u32::from_le_bytes([a[0], a[1], a[2], a[3]])),
            "float" | "float32" => f64::from(f32::from_le_bytes([a[0], a[1], a[2], a[3]])),
            _ => f64::from_le_bytes(a),
        })
    }
}
pub(crate) fn ply(data: &[u8]) -> Result<(Option<Mesh>, Map<String, Value>)> {
    let end = data
        .windows(11)
        .position(|w| w == b"end_header\n")
        .context("not a PLY file")?
        + 11;
    let header = std::str::from_utf8(&data[..end]).context("bad PLY header")?;
    let mut lines = header.lines();
    ensure!(lines.next().map(str::trim) == Some("ply"), "not a PLY file");
    let mut format = PlyFormat::Ascii;
    let mut elements: Vec<PlyElement> = Vec::new();
    for line in lines {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.as_slice() {
            ["format", f, ..] => {
                format = match *f {
                    "ascii" => PlyFormat::Ascii,
                    "binary_little_endian" => PlyFormat::Le,
                    "binary_big_endian" => PlyFormat::Be,
                    f => bail!("unknown PLY format {f}"),
                }
            }
            ["element", name, count] => {
                elements.push(PlyElement {
                    name: (*name).to_owned(),
                    count: count.parse()?,
                    props: Vec::new(),
                });
            }
            ["property", "list", c, i, name] => elements
                .last_mut()
                .context("PLY property before element")?
                .props
                .push(PlyProp::List {
                    name: (*name).to_owned(),
                    count_ty: (*c).to_owned(),
                    item_ty: (*i).to_owned(),
                }),
            ["property", ty, name] => elements
                .last_mut()
                .context("PLY property before element")?
                .props
                .push(PlyProp::Scalar {
                    name: (*name).to_owned(),
                    ty: (*ty).to_owned(),
                }),
            _ => {}
        }
    }
    let body = &data[end..];
    let text = if format == PlyFormat::Ascii {
        std::str::from_utf8(body).context("bad ASCII PLY")?
    } else {
        ""
    };
    let mut r = PlyReader {
        format,
        bin: body,
        pos: 0,
        words: text.split_ascii_whitespace(),
    };
    let (mut pos, mut nrm) = (Vec::new(), Vec::new());
    let mut faces: Vec<Vec<u32>> = Vec::new();
    for el in &elements {
        ensure!(
            el.count <= MAX_TRIANGLES * 3,
            "PLY element {} is too large to preview",
            el.name
        );
        for _ in 0..el.count {
            let mut vals: HashMap<&str, f64> = HashMap::new();
            let mut list: Option<Vec<u32>> = None;
            for p in &el.props {
                match p {
                    PlyProp::Scalar { name, ty } => {
                        vals.insert(name.as_str(), r.value(ty)?);
                    }
                    PlyProp::List {
                        name,
                        count_ty,
                        item_ty,
                    } => {
                        let n = r.value(count_ty)? as usize;
                        ensure!(n <= 64, "PLY list is too long");
                        let mut items = Vec::with_capacity(n);
                        for _ in 0..n {
                            items.push(r.value(item_ty)? as u32);
                        }
                        if name == "vertex_indices" || name == "vertex_index" {
                            list = Some(items);
                        }
                    }
                }
            }
            match el.name.as_str() {
                "vertex" => {
                    let g = |k: &str| vals.get(k).copied().unwrap_or(0.0) as f32;
                    pos.push([g("x"), g("y"), g("z")]);
                    if vals.contains_key("nx") {
                        nrm.push([g("nx"), g("ny"), g("nz")]);
                    }
                }
                "face" => faces.extend(list),
                _ => {}
            }
        }
    }
    let mut facts = Map::new();
    if faces.is_empty() {
        facts.insert("points".into(), json!(pos.len()));
        return Ok((None, facts));
    }
    let mut indices = Vec::new();
    for f in &faces {
        for k in 1..f.len().saturating_sub(1) {
            let tri = [f[0], f[k], f[k + 1]];
            if tri.iter().all(|&i| (i as usize) < pos.len()) {
                indices.extend(tri);
            }
        }
    }
    ensure!(
        indices.len() / 3 <= MAX_TRIANGLES,
        "too many triangles to preview"
    );
    // Smooth normals from the faces when the file has none.
    if nrm.len() != pos.len() {
        nrm = smooth_normals(&pos, &indices);
    }
    facts.insert("triangles".into(), json!(indices.len() / 3));
    facts.insert("vertices".into(), json!(pos.len()));
    Ok((
        Some(single(Prim {
            positions: pos,
            normals: nrm,
            uvs: Vec::new(),
            indices,
            material: 0,
        })),
        facts,
    ))
}
