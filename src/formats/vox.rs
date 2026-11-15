//! `MagicaVoxel` `.vox` (the format as documented in ephtracy/voxel-model, versions 150 and 200).
//!
//! Every model is placed where the scene graph puts it (`nTRN` translations and rotations through
//! `nGRP` groups down to `nSHP` shapes, first animation frame), skipping hidden nodes and layers.
//! A placed model's voxels are centred on its pivot, `floor(size / 2)`, as `MagicaVoxel` shows them;
//! a file without a scene graph keeps the raw voxel coordinates. Only faces between a voxel and
//! empty space are drawn, grouped by palette color. Colors are sRGB (the default palette when the
//! file has none), and `MATL` metal, roughness and emission carry over.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::mesh::{Material, Mesh, Prim, srgb_to_linear};
use crate::{Context as _, Result};

const MAX_VOXELS: usize = 4_000_000;

/// The palette `MagicaVoxel` uses when a file has no `RGBA` chunk, indexed by color index (0 is
/// empty): a 6x6x6 cube of sRGB levels, then ramps of red, green, blue and gray.
fn default_palette() -> [[u8; 4]; 256] {
    let mut p = [[0u8; 4]; 256];
    let levels = [0xff, 0xcc, 0x99, 0x66, 0x33, 0x00];
    let mut i = 1;
    for &r in &levels {
        for &g in &levels {
            for &b in &levels {
                if i < 216 {
                    p[i] = [r, g, b, 0xff];
                    i += 1;
                }
            }
        }
    }
    let ramp = [0xee, 0xdd, 0xbb, 0xaa, 0x88, 0x77, 0x55, 0x44, 0x22, 0x11];
    for channel in 0..4 {
        for &v in &ramp {
            p[i] = match channel {
                0 => [v, 0, 0, 0xff],
                1 => [0, v, 0, 0xff],
                2 => [0, 0, v, 0xff],
                _ => [v, v, v, 0xff],
            };
            i += 1;
        }
    }
    p
}

struct Model {
    size: [i32; 3],
    voxels: Vec<[u8; 4]>,
}

/// A rotation (rows of -1/0/1) and translation, all integers.
#[derive(Clone, Copy)]
struct Xform {
    r: [[i32; 3]; 3],
    t: [i32; 3],
}

impl Xform {
    const IDENTITY: Xform = Xform {
        r: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        t: [0; 3],
    };

    fn rotate(&self, v: [i32; 3]) -> [i32; 3] {
        std::array::from_fn(|i| self.r[i][0] * v[0] + self.r[i][1] * v[1] + self.r[i][2] * v[2])
    }

    /// `self` applied after `local`.
    fn then(&self, local: &Xform) -> Xform {
        let r = std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| self.r[i][k] * local.r[k][j]).sum()));
        let lt = self.rotate(local.t);
        Xform {
            r,
            t: std::array::from_fn(|i| lt[i] + self.t[i]),
        }
    }

    fn mirrors(&self) -> bool {
        let r = &self.r;
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        det < 0
    }

    /// The rotation byte of an `nTRN` frame's `_r`.
    fn from_rotation_byte(b: u8) -> Xform {
        let first = usize::from(b & 3);
        let second = usize::from((b >> 2) & 3);
        let mut r = [[0; 3]; 3];
        if first > 2 || second > 2 || first == second {
            return Xform::IDENTITY;
        }
        let third = 3 - first - second;
        let sign = |bit: u8| if b >> bit & 1 == 1 { -1 } else { 1 };
        r[0][first] = sign(4);
        r[1][second] = sign(5);
        r[2][third] = sign(6);
        Xform { r, t: [0; 3] }
    }
}

enum Node {
    Transform {
        child: u32,
        layer: i32,
        hidden: bool,
        xform: Xform,
    },
    Group {
        children: Vec<u32>,
    },
    Shape {
        models: Vec<u32>,
    },
}

#[derive(Default, Clone, Copy)]
struct Matl {
    metal: f32,
    rough: Option<f32>,
    emit: f32,
}

/// Reads little-endian values and `DICT`s from a chunk body.
struct Body<'a> {
    b: &'a [u8],
    at: usize,
}

impl Body<'_> {
    fn u32(&mut self) -> Result<u32> {
        let v = self.b.get(self.at..self.at + 4).context("VOX chunk ends early")?;
        self.at += 4;
        Ok(u32::from_le_bytes(v.try_into()?))
    }

    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()?.cast_signed())
    }

    fn string(&mut self) -> Result<&str> {
        let n = self.u32()? as usize;
        let s = self
            .b
            .get(self.at..self.at.saturating_add(n))
            .context("VOX chunk ends early")?;
        self.at += n;
        Ok(std::str::from_utf8(s).unwrap_or(""))
    }

    fn dict(&mut self) -> Result<HashMap<String, String>> {
        let n = self.u32()?;
        ensure!(n <= 4096, "VOX dictionary too large");
        let mut d = HashMap::new();
        for _ in 0..n {
            let k = self.string()?.to_owned();
            let v = self.string()?.to_owned();
            d.insert(k, v);
        }
        Ok(d)
    }
}

/// `MagicaVoxel`: every visible model where the scene puts it, exposed faces grouped by color.
pub(crate) fn vox(data: &[u8]) -> Result<(Option<Mesh>, Map<String, Value>)> {
    ensure!(data.len() >= 20 && &data[..4] == b"VOX ", "not a MagicaVoxel file");
    let le = |at: usize| -> Result<u32> {
        Ok(u32::from_le_bytes(
            data.get(at..at + 4).context("VOX ends early")?.try_into()?,
        ))
    };
    ensure!(&data[8..12] == b"MAIN", "VOX file has no MAIN chunk");
    let mut models: Vec<Model> = Vec::new();
    let mut size: Option<[i32; 3]> = None;
    let mut palette: Option<[[u8; 4]; 256]> = None;
    let mut nodes: HashMap<u32, Node> = HashMap::new();
    let mut hidden_layers: Vec<i32> = Vec::new();
    let mut matl: HashMap<u8, Matl> = HashMap::new();
    let mut total = 0usize;
    // MAIN's children follow its 12-byte chunk header, after the 8-byte file header.
    let mut at = 8 + 12;
    while at + 12 <= data.len() {
        let id = &data[at..at + 4];
        let content = le(at + 4)? as usize;
        let start = at + 12;
        let end = start.checked_add(content).context("VOX chunk overflow")?;
        let next = end.checked_add(le(at + 8)? as usize).context("VOX chunk overflow")?;
        let body = data.get(start..end).context("VOX chunk ends early")?;
        let mut b = Body { b: body, at: 0 };
        match id {
            b"SIZE" => size = Some([b.i32()?, b.i32()?, b.i32()?]),
            b"XYZI" => {
                let n = b.u32()? as usize;
                total = total.saturating_add(n);
                ensure!(total <= MAX_VOXELS, "too many voxels to convert ({total})");
                let raw = body.get(4..4 + n * 4).context("VOX voxels end early")?;
                let size = size.take().context("VOX voxels without a SIZE")?;
                models.push(Model {
                    size,
                    voxels: raw.as_chunks::<4>().0.to_vec(),
                });
            }
            b"RGBA" => {
                // Entry i is color index i + 1.
                let raw = body.get(..1024).context("VOX palette ends early")?;
                let mut p = [[0u8; 4]; 256];
                for (i, c) in raw.as_chunks::<4>().0.iter().take(255).enumerate() {
                    p[i + 1] = *c;
                }
                palette = Some(p);
            }
            b"nTRN" => {
                let id = b.u32()?;
                let attrs = b.dict()?;
                let child = b.u32()?;
                let _reserved = b.i32()?;
                let layer = b.i32()?;
                let frames = b.u32()?;
                let mut xform = Xform::IDENTITY;
                if frames > 0 {
                    let f = b.dict()?;
                    if let Some(r) = f.get("_r").and_then(|r| r.trim().parse::<u8>().ok()) {
                        xform = Xform::from_rotation_byte(r);
                    }
                    if let Some(t) = f.get("_t") {
                        let v: Vec<i32> = t.split_whitespace().filter_map(|w| w.parse().ok()).collect();
                        if let [x, y, z] = v[..] {
                            xform.t = [x, y, z];
                        }
                    }
                }
                let hidden = attrs.get("_hidden").is_some_and(|h| h == "1");
                nodes.insert(
                    id,
                    Node::Transform {
                        child,
                        layer,
                        hidden,
                        xform,
                    },
                );
            }
            b"nGRP" => {
                let id = b.u32()?;
                b.dict()?;
                let n = b.u32()?;
                ensure!(
                    n as usize <= body.len() / 4,
                    "VOX group lists more children than it holds"
                );
                let children = (0..n).map(|_| b.u32()).collect::<Result<_>>()?;
                nodes.insert(id, Node::Group { children });
            }
            b"nSHP" => {
                let id = b.u32()?;
                b.dict()?;
                let n = b.u32()?;
                ensure!(
                    n as usize <= body.len() / 4,
                    "VOX shape lists more models than it holds"
                );
                let mut ms = Vec::new();
                for _ in 0..n {
                    ms.push(b.u32()?);
                    b.dict()?;
                }
                nodes.insert(id, Node::Shape { models: ms });
            }
            b"LAYR" => {
                let id = b.i32()?;
                if b.dict()?.get("_hidden").is_some_and(|h| h == "1") {
                    hidden_layers.push(id);
                }
            }
            b"MATL" => {
                let id = b.u32()?;
                let d = b.dict()?;
                let f = |k: &str| d.get(k).and_then(|v| v.parse::<f32>().ok());
                let ty = d.get("_type").map_or("_diffuse", String::as_str);
                if let Ok(index) = u8::try_from(id) {
                    matl.insert(
                        index,
                        Matl {
                            metal: if ty == "_metal" {
                                f("_metal").unwrap_or(0.0)
                            } else {
                                0.0
                            },
                            rough: (ty == "_metal").then(|| f("_rough")).flatten(),
                            emit: if ty == "_emit" {
                                f("_emit").unwrap_or(0.0) * (1.0 + f("_flux").unwrap_or(0.0))
                            } else {
                                0.0
                            },
                        },
                    );
                }
            }
            _ => {}
        }
        at = next;
    }
    ensure!(!models.is_empty(), "VOX file has no model");

    // Where each model goes: walked from the root transform, or each model as it is.
    let mut placed: Vec<(usize, Xform, bool)> = Vec::new();
    if nodes.contains_key(&0) {
        let mut stack = vec![(0u32, Xform::IDENTITY, 0usize)];
        while let Some((id, parent, depth)) = stack.pop() {
            ensure!(depth < 64, "VOX scene graph too deep");
            match nodes.get(&id) {
                Some(Node::Transform {
                    child,
                    layer,
                    hidden,
                    xform,
                }) => {
                    if !*hidden && !hidden_layers.contains(layer) {
                        stack.push((*child, parent.then(xform), depth + 1));
                    }
                }
                Some(Node::Group { children }) => {
                    stack.extend(children.iter().rev().map(|&c| (c, parent, depth + 1)));
                }
                Some(Node::Shape { models: ms }) => {
                    for &m in ms {
                        if (m as usize) < models.len() {
                            placed.push((m as usize, parent, true));
                        }
                    }
                }
                None => {}
            }
        }
    } else {
        placed.extend((0..models.len()).map(|m| (m, Xform::IDENTITY, false)));
    }

    let palette = palette.unwrap_or_else(default_palette);
    let mut prims: HashMap<u8, Prim> = HashMap::new();
    for &(m, xform, centred) in &placed {
        mesh_model(&models[m], &xform, centred, &mut prims)?;
    }
    let mut prims: Vec<(u8, Prim)> = prims.into_iter().collect();
    prims.sort_by_key(|(c, _)| *c);

    let mut facts = Map::new();
    facts.insert(
        "voxels".into(),
        json!(placed.iter().map(|&(m, ..)| models[m].voxels.len()).sum::<usize>()),
    );
    let s = models[0].size;
    facts.insert("voxel_size".into(), json!(format!("{}x{}x{}", s[0], s[1], s[2])));
    if models.len() > 1 {
        facts.insert("models".into(), json!(models.len()));
    }
    if prims.is_empty() {
        return Ok((None, facts));
    }
    let mut mesh = Mesh::default();
    for (i, (c, mut p)) in prims.into_iter().enumerate() {
        p.material = i;
        mesh.prims.push(p);
        let rgba = palette[usize::from(c)];
        let [r, g, b] = [rgba[0], rgba[1], rgba[2]].map(|v| srgb_to_linear(f32::from(v) / 255.0));
        let mut mat = Material::plain(&format!("color_{c}"), [r, g, b, 1.0]);
        if let Some(x) = matl.get(&c) {
            mat.metallic = x.metal.clamp(0.0, 1.0);
            if let Some(rough) = x.rough {
                mat.roughness = rough.clamp(0.0, 1.0);
            }
            let e = x.emit.clamp(0.0, 1.0);
            mat.emissive = [r * e, g * e, b * e];
        }
        mesh.materials.push(mat);
    }
    Ok((Some(mesh), facts))
}

/// Faces as (neighbor direction, four corners counter-clockwise from outside), Z-up.
const FACES: [([i32; 3], [[i32; 3]; 4]); 6] = [
    ([1, 0, 0], [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
    ([-1, 0, 0], [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]),
    ([0, 1, 0], [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
    ([0, -1, 0], [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
    ([0, 0, 1], [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
    ([0, 0, -1], [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
];

/// Adds a model's exposed faces, placed by `xform`, to the primitive of each voxel's color.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "model sizes are checked to 1..=1024, so every coordinate is small and signs are checked"
)]
fn mesh_model(model: &Model, xform: &Xform, centred: bool, prims: &mut HashMap<u8, Prim>) -> Result<()> {
    let [sx, sy, sz] = model.size;
    ensure!(
        (1..=1024).contains(&sx) && (1..=1024).contains(&sy) && (1..=1024).contains(&sz),
        "VOX model size {sx}x{sy}x{sz} is out of range"
    );
    let (sx, sy, sz) = (sx as usize, sy as usize, sz as usize);
    let cell = |x: usize, y: usize, z: usize| (z * sy + y) * sx + x;
    let mut occupied = vec![0u64; (sx * sy * sz).div_ceil(64)];
    let inside = |v: &[u8; 4]| (v[0] as usize) < sx && (v[1] as usize) < sy && (v[2] as usize) < sz && v[3] != 0;
    for v in model.voxels.iter().filter(|v| inside(v)) {
        let c = cell(v[0].into(), v[1].into(), v[2].into());
        occupied[c / 64] |= 1 << (c % 64);
    }
    let filled = |x: i32, y: i32, z: i32| {
        if x < 0 || y < 0 || z < 0 || x as usize >= sx || y as usize >= sy || z as usize >= sz {
            return false;
        }
        let c = cell(x as usize, y as usize, z as usize);
        occupied[c / 64] >> (c % 64) & 1 == 1
    };
    let pivot = if centred {
        [sx / 2, sy / 2, sz / 2].map(|p| p as i32)
    } else {
        [0; 3]
    };
    let mirrored = xform.mirrors();
    // MagicaVoxel is Z-up, glTF Y-up: (x, y, z) -> (x, z, -y).
    let to_gltf = |p: [i32; 3]| [p[0] as f32, p[2] as f32, -p[1] as f32];
    let mut seen = vec![0u64; occupied.len()];
    for v in model.voxels.iter().filter(|v| inside(v)) {
        let (x, y, z) = (i32::from(v[0]), i32::from(v[1]), i32::from(v[2]));
        // A voxel listed twice draws once.
        let c = cell(v[0].into(), v[1].into(), v[2].into());
        if seen[c / 64] >> (c % 64) & 1 == 1 {
            continue;
        }
        seen[c / 64] |= 1 << (c % 64);
        for (d, corners) in &FACES {
            if filled(x + d[0], y + d[1], z + d[2]) {
                continue;
            }
            let p = prims.entry(v[3]).or_default();
            let base = u32::try_from(p.positions.len())?;
            let normal = to_gltf(xform.rotate(*d));
            for k in 0..4 {
                // Mirroring reverses the winding, so walk the corners the other way.
                let c = corners[if mirrored { 3 - k } else { k }];
                let local = [x + c[0] - pivot[0], y + c[1] - pivot[1], z + c[2] - pivot[2]];
                let w = xform.rotate(local);
                p.positions
                    .push(to_gltf([w[0] + xform.t[0], w[1] + xform.t[1], w[2] + xform.t[2]]));
                p.normals.push(normal);
            }
            p.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    Ok(())
}
