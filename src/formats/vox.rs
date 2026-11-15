//! `MagicaVoxel` `.vox` (the format as documented in ephtracy/voxel-model, versions 150 and 200).
//!
//! Each model becomes a mesh of the faces between a voxel and empty space, a primitive per
//! palette color, in its own space: centred on its pivot, `floor(size / 2)`, when a scene graph
//! places it, as `MagicaVoxel` shows it; a file without a scene graph keeps the raw voxel
//! coordinates, a node per model. The scene graph becomes nodes: each `nTRN` transform (first
//! frame, its name, layer and attributes in extras) with the `nGRP` group or `nSHP` shape under
//! it; a placed model's nodes share its mesh. Hidden nodes and nodes on hidden layers are kept,
//! hidden with `KHR_node_visibility`. More than one frame is a step animation at
//! [`FRAMES_PER_SECOND`]. Colors are sRGB (the default palette when the file has none), and
//! `MATL` metal, roughness and emission carry over. The asset's `extras.source` keeps the
//! palette, every `MATL`, the layers, render settings, cameras, notes and index map. Coordinates
//! turn from Z-up to glTF's Y-up.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::document::{Builder, Document, Keep};
use crate::mesh::{Material, Prim, srgb_to_linear};
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

/// A chunk's `DICT`.
type Dict = HashMap<String, String>;

/// One `nTRN` frame: its frame number and where it puts the child.
struct Frame {
    number: u32,
    xform: Xform,
}

enum Node {
    Transform {
        attrs: Dict,
        child: u32,
        layer: i32,
        frames: Vec<Frame>,
    },
    Group {
        attrs: Dict,
        children: Vec<u32>,
    },
    Shape {
        attrs: Dict,
        models: Vec<(u32, Dict)>,
    },
}

#[derive(Default, Clone, Copy)]
struct Matl {
    metal: f32,
    rough: Option<f32>,
    emit: f32,
}

impl Matl {
    fn new(d: &Dict) -> Matl {
        let f = |k: &str| d.get(k).and_then(|v| v.parse::<f32>().ok());
        let ty = d.get("_type").map_or("_diffuse", String::as_str);
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
        }
    }
}

/// `.vox` has no frame rate; animation frames are this many to a second.
const FRAMES_PER_SECOND: f32 = 10.0;

/// The most glTF nodes a scene graph may unfold into (a group listed twice is walked twice).
const MAX_NODES: usize = 1 << 16;

fn dict_json(d: &Dict) -> Value {
    Value::Object(d.iter().map(|(k, v)| (k.clone(), json!(v))).collect())
}

/// Everything a `.vox` file holds, read.
#[derive(Default)]
struct Vox {
    version: u32,
    models: Vec<Model>,
    palette: Option<[[u8; 4]; 256]>,
    nodes: HashMap<u32, Node>,
    layers: Vec<(i32, Dict)>,
    matl: Vec<(u32, Dict)>,
    render: Vec<Value>,
    cameras: Vec<Value>,
    notes: Vec<String>,
    index_map: Vec<u8>,
    skipped: Vec<String>,
}

/// `MagicaVoxel`: each model a mesh of the faces between its voxels and empty space, one
/// primitive per color; the scene graph as nodes, hidden ones kept but hidden.
pub(crate) fn vox(data: &[u8], keep: Keep) -> Result<Document<'static>> {
    let v = read(data)?;
    let scene = v.nodes.contains_key(&0);
    let models = &v.models;

    // Each model in its own space: centred on its pivot when a scene graph places it.
    let mut meshed: Vec<Vec<(u8, Prim)>> = Vec::with_capacity(models.len());
    for m in models {
        let mut prims: HashMap<u8, Prim> = HashMap::new();
        mesh_model(m, scene, &mut prims)?;
        let mut prims: Vec<(u8, Prim)> = prims.into_iter().collect();
        prims.sort_by_key(|(c, _)| *c);
        meshed.push(prims);
    }
    let mut colors: Vec<u8> = meshed.iter().flatten().map(|(c, _)| *c).collect();
    colors.sort_unstable();
    colors.dedup();

    let mut facts = Map::new();
    let s = models[0].size;
    facts.insert("voxel_size".into(), json!(format!("{}x{}x{}", s[0], s[1], s[2])));
    if models.len() > 1 {
        facts.insert("models".into(), json!(models.len()));
    }
    if colors.is_empty() {
        facts.insert("voxels".into(), json!(0));
        return Ok(Document::empty(facts));
    }

    let mut b = Builder::default();
    let mut meshes = Vec::new();
    let mut mesh_of = Vec::with_capacity(models.len());
    for (m, prims) in meshed.into_iter().enumerate() {
        let prims = prims
            .into_iter()
            .map(|(c, mut p)| {
                p.material = colors.binary_search(&c).unwrap_or(0);
                p
            })
            .collect();
        mesh_of.push(b.mesh(prims, colors.len()).map(|mut mesh| {
            mesh["name"] = json!(format!("model_{m}"));
            meshes.push(mesh);
            meshes.len() - 1
        }));
    }

    let mut walk = Walk {
        v: &v,
        mesh_of: &mesh_of,
        out: Vec::new(),
        animated: Vec::new(),
        voxels: 0,
        hidden: false,
    };
    let roots = if scene {
        walk.visit(0, 0, true)?.into_iter().collect()
    } else {
        // No scene graph: every model where its voxels are.
        for (m, model) in models.iter().enumerate() {
            walk.voxels += model.voxels.len();
            if let Some(mesh) = mesh_of[m] {
                walk.out.push(json!({ "mesh": mesh }));
            }
        }
        (0..walk.out.len()).collect::<Vec<_>>()
    };
    facts.insert("voxels".into(), json!(walk.voxels));
    let Walk {
        out: nodes,
        animated,
        hidden,
        ..
    } = walk;

    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source(&v) } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": roots }]));
    json.insert("nodes".into(), Value::Array(nodes));
    json.insert("meshes".into(), Value::Array(meshes));
    if hidden {
        json.insert("extensionsUsed".into(), json!(["KHR_node_visibility"]));
    }
    if keep.animations && !animated.is_empty() {
        json.insert("animations".into(), json!([animation(&mut b, &animated)]));
    }

    let palette = v.palette.unwrap_or_else(default_palette);
    let matl: HashMap<u32, Matl> = v.matl.iter().map(|(i, d)| (*i, Matl::new(d))).collect();
    let materials: Vec<Material> = colors
        .iter()
        .map(|&c| {
            let rgba = palette[usize::from(c)];
            let [r, g, b] = [rgba[0], rgba[1], rgba[2]].map(|v| srgb_to_linear(f32::from(v) / 255.0));
            let mut mat = Material::plain(&format!("color_{c}"), [r, g, b, 1.0]);
            if let Some(x) = matl.get(&u32::from(c)) {
                mat.metallic = x.metal.clamp(0.0, 1.0);
                if let Some(rough) = x.rough {
                    mat.roughness = rough.clamp(0.0, 1.0);
                }
                let e = x.emit.clamp(0.0, 1.0);
                mat.emissive = [r * e, g * e, b * e];
            }
            mat
        })
        .collect();
    b.materials(&materials, Vec::new(), &mut json);
    if let Some(list) = json.get_mut("materials").and_then(Value::as_array_mut) {
        for (mat, c) in list.iter_mut().zip(&colors) {
            mat["extras"] = json!({ "paletteIndex": c });
        }
    }
    Ok(b.finish(json, facts))
}

/// The asset's `extras.source`: what the file says that the nodes and materials don't.
fn source(v: &Vox) -> Value {
    let mut s = Map::new();
    s.insert("format".into(), json!("VOX"));
    s.insert("version".into(), json!(v.version));
    s.insert("upAxis".into(), json!("Z"));
    if let Some(p) = &v.palette {
        let hex: Vec<String> = p[1..]
            .iter()
            .map(|c| format!("#{:02x}{:02x}{:02x}{:02x}", c[0], c[1], c[2], c[3]))
            .collect();
        s.insert("palette".into(), json!(hex));
    }
    if !v.matl.is_empty() {
        let m: Map<String, Value> = v.matl.iter().map(|(i, d)| (i.to_string(), dict_json(d))).collect();
        s.insert("materials".into(), Value::Object(m));
    }
    if !v.layers.is_empty() {
        let layers: Vec<Value> = v
            .layers
            .iter()
            .map(|(id, d)| {
                let mut l = dict_json(d);
                l["id"] = json!(id);
                l
            })
            .collect();
        s.insert("layers".into(), json!(layers));
    }
    if v.nodes.contains_key(&0) {
        s.insert("framesPerSecond".into(), json!(FRAMES_PER_SECOND));
    }
    for (key, list) in [("render", &v.render), ("cameras", &v.cameras)] {
        if !list.is_empty() {
            s.insert(key.into(), json!(list));
        }
    }
    if !v.notes.is_empty() {
        s.insert("notes".into(), json!(v.notes));
    }
    if !v.index_map.is_empty() {
        s.insert("indexMap".into(), json!(v.index_map));
    }
    if !v.skipped.is_empty() {
        s.insert("skipped".into(), json!(v.skipped));
    }
    Value::Object(s)
}

/// Reads every chunk.
fn read(data: &[u8]) -> Result<Vox> {
    ensure!(data.len() >= 20 && &data[..4] == b"VOX ", "not a MagicaVoxel file");
    let le = |at: usize| -> Result<u32> {
        Ok(u32::from_le_bytes(
            data.get(at..at + 4).context("VOX ends early")?.try_into()?,
        ))
    };
    ensure!(&data[8..12] == b"MAIN", "VOX file has no MAIN chunk");
    let mut v = Vox {
        version: le(4)?,
        ..Vox::default()
    };
    let mut size: Option<[i32; 3]> = None;
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
                v.models.push(Model {
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
                v.palette = Some(p);
            }
            b"nTRN" => {
                let id = b.u32()?;
                let attrs = b.dict()?;
                let child = b.u32()?;
                let _reserved = b.i32()?;
                let layer = b.i32()?;
                let n = b.u32()?;
                ensure!(
                    n as usize <= body.len() / 4,
                    "VOX transform lists more frames than it holds"
                );
                let mut frames = Vec::new();
                for k in 0..n {
                    let f = b.dict()?;
                    let mut xform = Xform::IDENTITY;
                    if let Some(r) = f.get("_r").and_then(|r| r.trim().parse::<u8>().ok()) {
                        xform = Xform::from_rotation_byte(r);
                    }
                    if let Some(t) = f.get("_t") {
                        let v: Vec<i32> = t.split_whitespace().filter_map(|w| w.parse().ok()).collect();
                        if let [x, y, z] = v[..] {
                            xform.t = [x, y, z];
                        }
                    }
                    let number = f.get("_f").and_then(|f| f.trim().parse().ok()).unwrap_or(k);
                    frames.push(Frame { number, xform });
                }
                v.nodes.insert(
                    id,
                    Node::Transform {
                        attrs,
                        child,
                        layer,
                        frames,
                    },
                );
            }
            b"nGRP" => {
                let id = b.u32()?;
                let attrs = b.dict()?;
                let n = b.u32()?;
                ensure!(
                    n as usize <= body.len() / 4,
                    "VOX group lists more children than it holds"
                );
                let children = (0..n).map(|_| b.u32()).collect::<Result<_>>()?;
                v.nodes.insert(id, Node::Group { attrs, children });
            }
            b"nSHP" => {
                let id = b.u32()?;
                let attrs = b.dict()?;
                let n = b.u32()?;
                ensure!(
                    n as usize <= body.len() / 4,
                    "VOX shape lists more models than it holds"
                );
                let mut models = Vec::new();
                for _ in 0..n {
                    models.push((b.u32()?, b.dict()?));
                }
                v.nodes.insert(id, Node::Shape { attrs, models });
            }
            b"LAYR" => {
                let id = b.i32()?;
                v.layers.push((id, b.dict()?));
            }
            b"MATL" => {
                let id = b.u32()?;
                v.matl.push((id, b.dict()?));
            }
            b"rOBJ" => v.render.push(dict_json(&b.dict()?)),
            b"rCAM" => {
                let id = b.u32()?;
                let mut cam = dict_json(&b.dict()?);
                cam["id"] = json!(id);
                v.cameras.push(cam);
            }
            b"NOTE" => {
                let n = b.u32()?;
                ensure!(n as usize <= body.len() / 4, "VOX note lists more names than it holds");
                for _ in 0..n {
                    let note = b.string()?.to_owned();
                    v.notes.push(note);
                }
            }
            b"IMAP" => v.index_map = body.get(..256).context("VOX index map ends early")?.to_vec(),
            b"PACK" => {}
            _ => {
                let name = String::from_utf8_lossy(id).into_owned();
                if !v.skipped.contains(&name) {
                    v.skipped.push(name);
                }
            }
        }
        at = next;
    }
    ensure!(!v.models.is_empty(), "VOX file has no model");
    Ok(v)
}

/// Unfolds the scene graph into glTF nodes.
struct Walk<'a> {
    v: &'a Vox,
    mesh_of: &'a [Option<usize>],
    out: Vec<Value>,
    /// Each node with more than one frame, and those frames.
    animated: Vec<(usize, &'a [Frame])>,
    /// Voxels the visible shapes draw.
    voxels: usize,
    /// Whether any node is hidden.
    hidden: bool,
}

impl Walk<'_> {
    /// The glTF node for scene node `id` (a transform with the group or shape under it), its
    /// index; `None` for an id that isn't there.
    fn visit(&mut self, id: u32, depth: usize, visible: bool) -> Result<Option<usize>> {
        ensure!(depth < 64, "VOX scene graph too deep");
        ensure!(
            self.out.len() < MAX_NODES,
            "VOX scene graph unfolds into too many nodes"
        );
        let Some(node) = self.v.nodes.get(&id) else {
            return Ok(None);
        };
        let mut out = Map::new();
        let mut extras = Map::new();
        let mut visible = visible;
        let content = match node {
            Node::Transform {
                attrs,
                child,
                layer,
                frames,
            } => {
                if let Some(name) = attrs.get("_name") {
                    out.insert("name".into(), json!(name));
                }
                let layer_hidden = self
                    .v
                    .layers
                    .iter()
                    .any(|(l, d)| l == layer && d.get("_hidden").is_some_and(|h| h == "1"));
                if attrs.get("_hidden").is_some_and(|h| h == "1") || layer_hidden {
                    out.insert(
                        "extensions".into(),
                        json!({ "KHR_node_visibility": { "visible": false } }),
                    );
                    self.hidden = true;
                    visible = false;
                }
                if let Some(f) = frames.first() {
                    trs(&f.xform, &mut out);
                }
                let rest: Dict = attrs
                    .iter()
                    .filter(|(k, _)| *k != "_name" && *k != "_hidden")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                if !rest.is_empty() {
                    extras.insert("transform".into(), dict_json(&rest));
                }
                if *layer >= 0 {
                    extras.insert("layer".into(), json!(layer));
                }
                self.v.nodes.get(child).map(|n| (n, *child))
            }
            _ => Some((node, id)),
        };
        let mut children = Vec::new();
        match content {
            Some((Node::Group { attrs, children: ids }, _)) => {
                if !attrs.is_empty() {
                    extras.insert("group".into(), dict_json(attrs));
                }
                for &c in ids {
                    children.extend(self.visit(c, depth + 1, visible)?);
                }
            }
            Some((Node::Shape { attrs, models }, _)) => {
                if !attrs.is_empty() {
                    extras.insert("shape".into(), dict_json(attrs));
                }
                if let Some(&(m, _)) = models.first() {
                    let m = m as usize;
                    if let Some(model) = self.v.models.get(m) {
                        if visible {
                            self.voxels += model.voxels.len();
                        }
                        if let Some(mesh) = self.mesh_of[m] {
                            out.insert("mesh".into(), json!(mesh));
                        }
                    }
                }
                // A model per frame: the first is drawn, the rest kept by number.
                if models.len() > 1 || models.first().is_some_and(|(_, d)| !d.is_empty()) {
                    let list: Vec<Value> = models
                        .iter()
                        .map(|(m, d)| {
                            let mut e = dict_json(d);
                            e["model"] = json!(m);
                            e
                        })
                        .collect();
                    extras.insert("models".into(), json!(list));
                }
            }
            Some((Node::Transform { .. }, child)) => {
                children.extend(self.visit(child, depth + 1, visible)?);
            }
            None => {}
        }
        if !children.is_empty() {
            out.insert("children".into(), json!(children));
        }
        if !extras.is_empty() {
            out.insert("extras".into(), Value::Object(extras));
        }
        self.out.push(Value::Object(out));
        let index = self.out.len() - 1;
        if let Node::Transform { frames, .. } = node
            && frames.len() > 1
        {
            self.animated.push((index, frames));
        }
        Ok(Some(index))
    }
}

/// The glTF translation, rotation and scale of `x`, a `MagicaVoxel` (Z-up) transform: a
/// mirror is a scale of -1 on X after a proper rotation.
fn gltf_trs(x: &Xform) -> ([f32; 3], [f32; 4], [f32; 3]) {
    // (x, y, z) -> (x, z, -y), as the vertices: C R C^T, C t.
    const C: [[i32; 3]; 3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]];
    let mul = |a: &[[i32; 3]; 3], b: &[[i32; 3]; 3]| -> [[i32; 3]; 3] {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    };
    let ct: [[i32; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| C[j][i]));
    let mut r = mul(&mul(&C, &x.r), &ct);
    let mut scale = [1.0; 3];
    if x.mirrors() {
        for row in &mut r {
            row[0] = -row[0];
        }
        scale[0] = -1.0;
    }
    #[allow(clippy::cast_precision_loss, reason = "translations are small integers")]
    let t = [x.t[0] as f32, x.t[2] as f32, -x.t[1] as f32];
    (t, quaternion(&r), scale)
}

/// Sets a node's `translation`, `rotation` and `scale` from `x`, leaving out the defaults.
#[allow(clippy::float_cmp, reason = "exact: every value comes from small integers")]
fn trs(x: &Xform, out: &mut Map<String, Value>) {
    let (t, q, s) = gltf_trs(x);
    if t != [0.0; 3] {
        out.insert("translation".into(), json!(t));
    }
    if q != [0.0, 0.0, 0.0, 1.0] {
        out.insert("rotation".into(), json!(q));
    }
    if s != [1.0; 3] {
        out.insert("scale".into(), json!(s));
    }
}

/// The unit quaternion (x, y, z, w) of a rotation matrix of -1, 0 and 1.
#[allow(clippy::cast_precision_loss, reason = "entries are -1, 0 or 1")]
fn quaternion(r: &[[i32; 3]; 3]) -> [f32; 4] {
    let m = |i: usize, j: usize| r[i][j] as f32;
    let trace = m(0, 0) + m(1, 1) + m(2, 2);
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (m(2, 1) - m(1, 2)) / s,
            (m(0, 2) - m(2, 0)) / s,
            (m(1, 0) - m(0, 1)) / s,
            s / 4.0,
        ]
    } else if m(0, 0) >= m(1, 1) && m(0, 0) >= m(2, 2) {
        let s = (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt() * 2.0;
        [
            s / 4.0,
            (m(0, 1) + m(1, 0)) / s,
            (m(0, 2) + m(2, 0)) / s,
            (m(2, 1) - m(1, 2)) / s,
        ]
    } else if m(1, 1) >= m(2, 2) {
        let s = (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt() * 2.0;
        [
            (m(0, 1) + m(1, 0)) / s,
            s / 4.0,
            (m(1, 2) + m(2, 1)) / s,
            (m(0, 2) - m(2, 0)) / s,
        ]
    } else {
        let s = (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt() * 2.0;
        [
            (m(0, 2) + m(2, 0)) / s,
            (m(1, 2) + m(2, 1)) / s,
            s / 4.0,
            (m(1, 0) - m(0, 1)) / s,
        ]
    };
    // Exact for quarter turns, so the identity comes out as (0, 0, 0, 1).
    q.map(|c| if c.abs() < 1e-6 { 0.0 } else { c })
}

/// One animation stepping each animated node through its frames.
#[allow(clippy::float_cmp, reason = "exact: a scale is -1 or 1")]
fn animation(b: &mut Builder, animated: &[(usize, &[Frame])]) -> Value {
    let (mut channels, mut samplers) = (Vec::new(), Vec::new());
    for &(node, frames) in animated {
        let mut frames: Vec<&Frame> = frames.iter().collect();
        frames.sort_by_key(|f| f.number);
        frames.dedup_by_key(|f| f.number);
        #[allow(clippy::cast_precision_loss, reason = "frame numbers are small")]
        let times: Vec<f32> = frames.iter().map(|f| f.number as f32 / FRAMES_PER_SECOND).collect();
        let input = b.floats(times, "SCALAR", true);
        let trs: Vec<_> = frames.iter().map(|f| gltf_trs(&f.xform)).collect();
        let mirrors = trs.iter().any(|(.., s)| *s != [1.0; 3]);
        let mut paths = vec![
            (
                "translation",
                trs.iter().flat_map(|(t, ..)| *t).collect::<Vec<_>>(),
                "VEC3",
            ),
            ("rotation", trs.iter().flat_map(|(_, q, _)| *q).collect(), "VEC4"),
        ];
        if mirrors {
            paths.push(("scale", trs.iter().flat_map(|(.., s)| *s).collect(), "VEC3"));
        }
        for (path, values, kind) in paths {
            let output = b.floats(values, kind, false);
            channels.push(json!({ "sampler": samplers.len(), "target": { "node": node, "path": path } }));
            samplers.push(json!({ "input": input, "output": output, "interpolation": "STEP" }));
        }
    }
    json!({ "name": "frames", "channels": channels, "samplers": samplers })
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

/// Adds a model's exposed faces to the primitive of each voxel's color, centred on its pivot
/// when `centred`.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "model sizes are checked to 1..=1024, so every coordinate is small and signs are checked"
)]
fn mesh_model(model: &Model, centred: bool, prims: &mut HashMap<u8, Prim>) -> Result<()> {
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
            let normal = to_gltf(*d);
            for c in corners {
                p.positions
                    .push(to_gltf([x + c[0] - pivot[0], y + c[1] - pivot[1], z + c[2] - pivot[2]]));
                p.normals.push(normal);
            }
            p.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every rotation byte's translation, rotation and scale rebuild its matrix, turned Y-up.
    #[test]
    fn every_rotation_byte_round_trips_through_its_quaternion() {
        let mut seen = 0;
        for byte in 0..=127u8 {
            let (first, second) = (byte & 3, (byte >> 2) & 3);
            if first > 2 || second > 2 || first == second {
                continue;
            }
            seen += 1;
            let x = Xform::from_rotation_byte(byte);
            let (_, [qx, qy, qz, qw], s) = gltf_trs(&x);
            let len = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "byte {byte}: |q| = {len}");
            // The quaternion's matrix, times the scale on X.
            let q = [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qz * qw),
                    2.0 * (qx * qz + qy * qw),
                ],
                [
                    2.0 * (qx * qy + qz * qw),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qx * qw),
                ],
                [
                    2.0 * (qx * qz - qy * qw),
                    2.0 * (qy * qz + qx * qw),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ];
            // Y-up: rows and columns (x, y, z) -> (x, z, -y).
            let c = |i: usize| match i {
                0 => (0, 1),
                1 => (2, 1),
                _ => (1, -1),
            };
            for (i, row) in q.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    let ((ri, si), (rj, sj)) = (c(i), c(j));
                    #[allow(clippy::cast_precision_loss, reason = "-1, 0 or 1")]
                    let want = (x.r[ri][rj] * si * sj) as f32;
                    let got = v * s[j];
                    assert!(
                        (got - want).abs() < 1e-5,
                        "byte {byte}: [{i}][{j}] is {got}, expected {want}"
                    );
                }
            }
        }
        assert_eq!(seen, 48);
    }
}
