//! 3MF, the 3D-printing format: a zip of XML models. Read here: meshes, components (in the same model or another
//! part of the zip) and the build's placed items, in the model's unit (scaled to glTF's metres); the 3MF materials
//! extension's base materials, color groups (per triangle or blended across one) and 2D textures; and what slicers
//! keep beside them: the filaments' colors, which filament each object and part prints in (Bambu Studio,
//! `OrcaSlicer`, `PrusaSlicer`), and the filaments painted onto triangles.
//!
//! The build is a node per item and component, each with its 3MF transform, under a root node that
//! turns the model's Z-up millimetres (or its unit) into glTF's Y-up metres; an object's triangles
//! are a mesh in its own space, shared wherever it's placed in the same filament. Object names,
//! types and part numbers, the model's metadata, its unit and the slicer's filament colors go in
//! extras.

use std::collections::{HashMap, HashSet};
use std::io::{BufReader, Read};

use crate::document::{Builder, Document};
use crate::{Context as _, Result};
use quick_xml::events::{BytesStart, Event};
use serde_json::{Map, Value, json};
use zip::ZipArchive;

/// The 3MF package.
type Zip<'a> = ZipArchive<Box<dyn super::ReadSeek + 'a>>;

use crate::mesh::{GRAY, Image, MAX_TRIANGLES, Material, Prim, face_normal, srgb_to_linear};

/// A 3MF transform: the first three columns of a 4x4 matrix, row by row (points are row vectors).
type Mat = [f32; 12];
const IDENTITY: Mat = [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.];
/// The most read from one part of the zip, unpacked: more is a zip bomb, not a model.
const MAX_PART: u64 = 2 << 30;
/// Slicer settings files are small; anything bigger isn't one.
const MAX_CONFIG: u64 = 64 << 20;
/// Filament colors for a slicer project that assigns filaments without saying what color they are.
const FALLBACK: [[u8; 3]; 6] = [
    [0xB8, 0xBC, 0xC4],
    [0xD9, 0x48, 0x3B],
    [0x3B, 0x7D, 0xD8],
    [0x4C, 0xAF, 0x50],
    [0xF2, 0xB7, 0x05],
    [0x8E, 0x5B, 0xC4],
];

#[derive(Default)]
struct Tri {
    v: [u32; 3],
    pid: Option<u32>,
    /// The property of each corner in the group (p2 and p3 default to p1).
    p1: Option<u32>,
    p2: Option<u32>,
    p3: Option<u32>,
    /// The filament painted on it (0: none, the part's own).
    paint: u8,
}

struct Component {
    path: Option<String>,
    id: u32,
    transform: Mat,
}

#[derive(Default)]
struct Object {
    /// The `name`, `type` and `partnumber` attributes.
    name: Option<String>,
    kind: Option<String>,
    partnumber: Option<String>,
    pid: Option<u32>,
    pindex: Option<u32>,
    verts: Vec<[f32; 3]>,
    tris: Vec<Tri>,
    components: Vec<Component>,
}

/// A property group of the materials extension.
enum Group {
    /// Base materials or a color group: each entry's color (sRGB, with alpha) and, for base
    /// materials, name.
    Colors(Vec<(Option<[u8; 4]>, Option<String>)>),
    /// A 2D texture's coordinates (origin bottom-left).
    Coords { texture: u32, uvs: Vec<[f32; 2]> },
}

/// A build item: an object (in another model file, by the production extension) placed by a transform.
struct Item {
    path: Option<String>,
    id: u32,
    transform: Mat,
    partnumber: Option<String>,
}

/// One model file of the zip.
#[derive(Default)]
struct Part {
    objects: HashMap<u32, Object>,
    groups: HashMap<u32, Group>,
    /// 2D textures by resource id: the zip part holding the image.
    textures: HashMap<u32, String>,
    build: Vec<Item>,
    application: Option<String>,
    /// Every `<metadata name>` and its text.
    metadata: Vec<(String, String)>,
    unit: Option<String>,
}

/// Metres per unit, for the units the core specification allows (millimetres when unsaid).
fn metres(unit: Option<&str>) -> Option<f64> {
    Some(match unit.unwrap_or("millimeter") {
        "micron" => 1e-6,
        "millimeter" => 1e-3,
        "centimeter" => 1e-2,
        "inch" => 0.0254,
        "foot" => 0.3048,
        "meter" => 1.0,
        _ => return None,
    })
}

/// What a slicer project says about filaments: their colors, and which one each object, part (Bambu Studio,
/// `OrcaSlicer`) or range of triangles (`PrusaSlicer`'s volumes) prints in. Filaments count from 1.
#[derive(Default)]
struct Slicer {
    filaments: Vec<[u8; 3]>,
    objects: HashMap<u32, usize>,
    parts: HashMap<(u32, u32), usize>,
    volumes: HashMap<u32, Vec<(u32, u32, usize)>>,
}

impl Slicer {
    fn filament(&self, n: usize) -> [u8; 3] {
        let i = n.saturating_sub(1);
        self.filaments.get(i).copied().unwrap_or(FALLBACK[i % FALLBACK.len()])
    }
}

/// Load a 3MF: the build as nodes, an object's triangles as a mesh with a primitive per look, and
/// the facts worth keeping.
pub(crate) fn load(input: super::Input<'_>) -> Result<Document<'static>> {
    let mut zip: Zip<'_> = ZipArchive::new(input.seekable()?).context("not a 3MF file (it isn't a zip)")?;
    let names: HashMap<String, String> = zip
        .file_names()
        .filter_map(Result::ok)
        .map(|n| (norm(&n), n.into_owned()))
        .collect();
    let root = text(&mut zip, &names, "_rels/.rels", MAX_CONFIG)
        .and_then(|rels| root_target(&rels))
        .map_or_else(|| "3d/3dmodel.model".to_owned(), |t| norm(&t));

    // The root model, then every model file its components point into.
    let mut parts: HashMap<String, Part> = HashMap::new();
    let mut todo = vec![root.clone()];
    let mut triangles = 0;
    while let Some(key) = todo.pop() {
        if parts.contains_key(&key) {
            continue;
        }
        ensure!(parts.len() < 1024, "too many model files in this 3MF");
        let Some(real) = names.get(&key) else {
            ensure!(key != root, "no 3D model in this 3MF");
            continue;
        };
        let reader = zip.by_name(real)?.take(MAX_PART);
        let part = parse_part(reader, &mut triangles).with_context(|| format!("reading {real}"))?;
        for o in part.objects.values() {
            todo.extend(o.components.iter().filter_map(|c| c.path.as_deref()).map(norm));
        }
        todo.extend(part.build.iter().filter_map(|i| i.path.as_deref()).map(norm));
        parts.insert(key, part);
    }
    let slicer = read_slicer(&mut zip, &names);
    let main = &parts[&root];
    ensure!(!main.build.is_empty(), "this 3MF builds nothing");
    let unit = main.unit.clone().unwrap_or_else(|| "millimeter".to_owned());
    let Some(scale) = metres(Some(&unit)) else {
        bail!("unknown 3MF unit {unit:?}")
    };

    let mut out = Emit {
        parts: &parts,
        slicer: &slicer,
        looks: HashMap::new(),
        order: Vec::new(),
        names: HashMap::new(),
        textures: Vec::new(),
        distinct: HashSet::new(),
        b: Builder::default(),
        meshes: Vec::new(),
        mesh_triangles: Vec::new(),
        mesh_of: HashMap::new(),
        nodes: Vec::new(),
        triangles: 0,
        built: 0,
    };
    let mut items = Vec::new();
    for item in &main.build {
        let key = item.path.as_deref().map_or_else(|| root.clone(), norm);
        let filament = slicer.objects.get(&item.id).copied();
        if let Some(n) = out.node(&key, item.id, &item.transform, item.id, filament, 0)? {
            if let Some(pn) = &item.partnumber {
                out.nodes[n]["extras"]["itemPartnumber"] = json!(pn);
            }
            items.push(n);
        }
    }
    ensure!(out.triangles > 0, "this 3MF has no triangles");

    let mut facts = Map::new();
    facts.insert("triangles".into(), json!(out.triangles));
    facts.insert("objects".into(), json!(main.build.len()));
    facts.insert("unit".into(), json!(unit));
    if out.distinct.len() > 1 {
        facts.insert("colors".into(), json!(out.distinct.len()));
    }
    if let Some(app) = main.application.as_ref().filter(|a| !a.is_empty()) {
        facts.insert("application".into(), json!(app));
    }

    let mut images = Vec::new();
    let mut image_of: HashMap<usize, Option<usize>> = HashMap::new();
    let lin = |c: [u8; 4]| {
        let [r, g, b] = [c[0], c[1], c[2]].map(|v| srgb_to_linear(f32::from(v) / 255.0));
        [r, g, b, f32::from(c[3]) / 255.0]
    };
    let mut materials = Vec::new();
    for (i, look) in out.order.iter().enumerate() {
        let named = |fallback: String| out.names.get(look).cloned().unwrap_or(fallback);
        let material = match *look {
            Look::Plain => Material::plain("default", GRAY),
            Look::Color(c) => Material::plain(&named(format!("color_{i}")), lin(c)),
            Look::Vertex => Material::plain("vertex_colors", [1.0; 4]),
            Look::Texture(t) => {
                let image = *image_of.entry(t).or_insert_with(|| {
                    let real = names.get(&out.textures[t])?;
                    let mut bytes = Vec::new();
                    zip.by_name(real).ok()?.take(MAX_PART).read_to_end(&mut bytes).ok()?;
                    images.push(Image::png_or_jpeg(&bytes)?);
                    Some(images.len() - 1)
                });
                match image {
                    Some(i) => Material {
                        color_image: Some(i),
                        ..Material::plain(&format!("texture_{t}"), [1.0; 4])
                    },
                    None => Material::plain(&format!("texture_{t}"), GRAY),
                }
            }
        };
        materials.push(material);
    }

    let mut source = json!({ "format": "3MF", "unit": unit, "metersPerUnit": scale, "upAxis": "Z" });
    if !main.metadata.is_empty() {
        let m: Map<String, Value> = main.metadata.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
        source["metadata"] = Value::Object(m);
    }
    if !slicer.filaments.is_empty() {
        let hex: Vec<String> = slicer
            .filaments
            .iter()
            .map(|c| format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2]))
            .collect();
        source["filaments"] = json!(hex);
    }
    let Emit {
        mut b,
        meshes,
        mut nodes,
        ..
    } = out;
    // Z-up to Y-up (a quarter turn about X), and the unit to metres.
    let h = std::f32::consts::FRAC_1_SQRT_2;
    nodes.push(
        json!({ "name": "RootNode", "children": items, "rotation": [-h, 0.0, 0.0, h], "scale": [scale, scale, scale] }),
    );
    let mut json = Map::new();
    json.insert("asset".into(), json!({ "extras": { "source": source } }));
    json.insert("scene".into(), json!(0));
    json.insert("scenes".into(), json!([{ "nodes": [nodes.len() - 1] }]));
    json.insert("nodes".into(), Value::Array(nodes));
    json.insert("meshes".into(), Value::Array(meshes));
    b.materials(&materials, images, &mut json);
    Ok(b.finish(json, facts))
}

/// How a triangle looks, which decides its primitive.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Look {
    Plain,
    Color([u8; 4]),
    /// Colors blended across it from its corners.
    Vertex,
    /// Textured: an index into `Emit::textures`.
    Texture(usize),
}

/// The build's objects as nodes, their triangles as meshes with a primitive per look.
struct Emit<'a> {
    parts: &'a HashMap<String, Part>,
    slicer: &'a Slicer,
    /// Each look's material, and the looks in that order.
    looks: HashMap<Look, usize>,
    order: Vec<Look>,
    /// Base material names, for the looks that come from one.
    names: HashMap<Look, String>,
    /// The zip parts of the textures used.
    textures: Vec<String>,
    /// The colors shown, for the facts.
    distinct: HashSet<[u8; 3]>,
    b: Builder,
    meshes: Vec<Value>,
    mesh_triangles: Vec<usize>,
    /// The mesh of an object (model file, id) in a filament, at the top of an item or not.
    mesh_of: HashMap<(String, u32, Option<usize>, bool), Option<usize>>,
    nodes: Vec<Value>,
    /// Triangles drawn, every placement counted.
    triangles: usize,
    /// Triangles in meshes.
    built: usize,
}

impl Emit<'_> {
    /// The node for object `id` of model `key` placed by `m`, inside the build item for object `item` (the one
    /// slicers' settings are kept against), printing in `filament` unless its parts or triangles say otherwise;
    /// `None` when the object isn't there.
    fn node(
        &mut self,
        key: &str,
        id: u32,
        m: &Mat,
        item: u32,
        filament: Option<usize>,
        depth: u32,
    ) -> Result<Option<usize>> {
        ensure!(depth < 16, "this 3MF's components nest too deep");
        ensure!(self.nodes.len() < 1 << 20, "this 3MF places too many components");
        let parts = self.parts;
        let Some(o) = parts.get(key).and_then(|p| p.objects.get(&id)) else {
            return Ok(None);
        };
        let mut node = Map::new();
        if let Some(name) = &o.name {
            node.insert("name".into(), json!(name));
        }
        #[allow(clippy::float_cmp_const, reason = "only the exact identity is left out")]
        if *m != IDENTITY {
            // Row vectors, so the rows are glTF's columns.
            let g = [
                m[0], m[1], m[2], 0.0, m[3], m[4], m[5], 0.0, m[6], m[7], m[8], 0.0, m[9], m[10], m[11], 1.0,
            ];
            node.insert("matrix".into(), json!(g));
        }
        let mut extras = json!({ "objectId": id });
        if let Some(k) = &o.kind {
            extras["type"] = json!(k);
        }
        if let Some(pn) = &o.partnumber {
            extras["partnumber"] = json!(pn);
        }
        node.insert("extras".into(), extras);
        let mut children = Vec::new();
        for c in &o.components {
            let sub = c.path.as_deref().map_or_else(|| key.to_owned(), norm);
            let f = self.slicer.parts.get(&(item, c.id)).copied().or(filament);
            children.extend(self.node(&sub, c.id, &c.transform, item, f, depth + 1)?);
        }
        if !children.is_empty() {
            node.insert("children".into(), json!(children));
        }
        if !o.tris.is_empty() {
            let at = (key.to_owned(), id, filament, depth == 0);
            let mesh = if let Some(&mesh) = self.mesh_of.get(&at) {
                mesh
            } else {
                let mesh = self.mesh(parts.get(key).context("model file")?, o, id, filament, depth == 0)?;
                self.mesh_of.insert(at, mesh);
                mesh
            };
            if let Some(mesh) = mesh {
                node.insert("mesh".into(), json!(mesh));
                self.triangles += self.mesh_triangles[mesh];
            }
        }
        self.nodes.push(Value::Object(node));
        Ok(Some(self.nodes.len() - 1))
    }

    /// Object `id`'s triangles, in its own space, as a mesh with a primitive per look; `top` when it is a build
    /// item's own object, which slicers' volumes divide.
    fn mesh(&mut self, part: &Part, o: &Object, id: u32, filament: Option<usize>, top: bool) -> Result<Option<usize>> {
        let volumes = if top { self.slicer.volumes.get(&id) } else { None };
        let mut prims: Vec<Prim> = Vec::new();
        let mut prim_of: HashMap<usize, usize> = HashMap::new();
        let mut count = 0;
        for (i, t) in o.tris.iter().enumerate() {
            ensure!(self.built < MAX_TRIANGLES, "too many triangles to convert");
            let [Some(a), Some(b), Some(c)] = t.v.map(|k| o.verts.get(k as usize)) else {
                continue;
            };
            let i = u32::try_from(i).unwrap_or(u32::MAX);
            let volume = volumes.and_then(|vs| vs.iter().find(|v| (v.0..=v.1).contains(&i)).map(|v| v.2));
            let filament = if t.paint > 0 {
                Some(usize::from(t.paint))
            } else {
                volume.or(filament)
            };
            let (look, uvs, colors) = match filament {
                Some(n) => {
                    let [r, g, b] = self.slicer.filament(n);
                    (Look::Color([r, g, b, 255]), None, None)
                }
                None => self.properties(part, o, t),
            };
            match look {
                Look::Color(c) => {
                    self.distinct.insert([c[0], c[1], c[2]]);
                }
                Look::Vertex => {
                    for c in colors.iter().flatten() {
                        self.distinct.insert([c[0], c[1], c[2]]);
                    }
                }
                _ => {}
            }
            let material = *self.looks.entry(look).or_insert_with(|| {
                self.order.push(look);
                self.order.len() - 1
            });
            let pi = *prim_of.entry(material).or_insert_with(|| {
                prims.push(Prim {
                    material,
                    ..Prim::default()
                });
                prims.len() - 1
            });
            let p = &mut prims[pi];
            let pos = [*a, *b, *c];
            let n = face_normal(pos[0], pos[1], pos[2]);
            for k in 0..3 {
                p.indices.push(u32::try_from(p.positions.len()).unwrap_or(u32::MAX));
                p.positions.push(pos[k]);
                p.normals.push(n);
                if let Some(uv) = &uvs {
                    p.uvs.push([uv[k][0], 1.0 - uv[k][1]]);
                }
                if let Some(cs) = &colors {
                    let c = cs[k];
                    let [r, g, b] = [c[0], c[1], c[2]].map(|v| srgb_to_linear(f32::from(v) / 255.0));
                    p.colors.push([r, g, b, f32::from(c[3]) / 255.0]);
                }
            }
            count += 1;
            self.built += 1;
        }
        // Materials are numbered as looks turn up, so every one used is already there.
        let Some(mut mesh) = self.b.mesh(prims, usize::MAX) else {
            return Ok(None);
        };
        if let Some(name) = &o.name {
            mesh["name"] = json!(name);
        }
        self.meshes.push(mesh);
        self.mesh_triangles.push(count);
        Ok(Some(self.meshes.len() - 1))
    }

    /// A triangle's look from the materials extension: its property group (its own, else its object's) and the
    /// entry for each corner (p1, p2 and p3; else the object's pindex).
    #[allow(clippy::type_complexity)]
    fn properties(&mut self, part: &Part, o: &Object, t: &Tri) -> (Look, Option<[[f32; 2]; 3]>, Option<[[u8; 4]; 3]>) {
        let plain = (Look::Plain, None, None);
        let Some(group) = t.pid.or(o.pid).and_then(|g| part.groups.get(&g)) else {
            return plain;
        };
        let p1 = t.p1.or(o.pindex).unwrap_or(0) as usize;
        let at = [p1, t.p2.map_or(p1, |v| v as usize), t.p3.map_or(p1, |v| v as usize)];
        match group {
            Group::Colors(cs) => {
                let c = at.map(|i| cs.get(i).and_then(|e| e.0));
                if c[0] == c[1] && c[1] == c[2] {
                    let look = c[0].map_or(Look::Plain, Look::Color);
                    if let Some(name) = cs.get(p1).and_then(|e| e.1.as_ref())
                        && look != Look::Plain
                    {
                        self.names.entry(look).or_insert_with(|| name.clone());
                    }
                    (look, None, None)
                } else {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "a constant in 0..=1"
                    )]
                    let gray = GRAY.map(|v| (v * 255.0) as u8);
                    (Look::Vertex, None, Some(c.map(|c| c.unwrap_or(gray))))
                }
            }
            Group::Coords { texture, uvs } => {
                let (Some(path), [Some(a), Some(b), Some(c)]) =
                    (part.textures.get(texture), at.map(|i| uvs.get(i).copied()))
                else {
                    return plain;
                };
                let key = norm(path);
                let t = if let Some(t) = self.textures.iter().position(|k| *k == key) {
                    t
                } else {
                    self.textures.push(key);
                    self.textures.len() - 1
                };
                (Look::Texture(t), Some([a, b, c]), None)
            }
        }
    }
}

/// A zip entry name as looked up: no leading slash, lower case (3MF part names aren't case-sensitive).
fn norm(name: &str) -> String {
    name.trim_start_matches('/').to_ascii_lowercase()
}

fn text(zip: &mut Zip<'_>, names: &HashMap<String, String>, name: &str, limit: u64) -> Option<String> {
    let mut s = String::new();
    zip.by_name(names.get(&norm(name))?)
        .ok()?
        .take(limit)
        .read_to_string(&mut s)
        .ok()?;
    Some(s)
}

fn each_attr(e: &BytesStart<'_>, mut f: impl FnMut(&[u8], &[u8])) {
    for a in e.attributes().with_checks(false).flatten() {
        f(a.key.local_name().as_ref(), &a.value);
    }
}

fn attr(e: &BytesStart<'_>, name: &[u8]) -> Option<String> {
    let mut out = None;
    each_attr(e, |k, v| {
        if k == name {
            out = Some(String::from_utf8_lossy(v).into_owned());
        }
    });
    out
}

fn num<T: std::str::FromStr>(v: &[u8]) -> Option<T> {
    std::str::from_utf8(v).ok()?.trim().parse().ok()
}

fn transform(v: &[u8]) -> Mat {
    let n: Vec<f32> = std::str::from_utf8(v)
        .unwrap_or("")
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    n.try_into().unwrap_or(IDENTITY)
}

/// `#RRGGBB` or `#RRGGBBAA`.
fn hex_rgba(s: &str) -> Option<[u8; 4]> {
    let h = s.trim().strip_prefix('#')?;
    if h.len() != 6 && h.len() != 8 {
        return None;
    }
    let byte = |i: usize| u8::from_str_radix(h.get(i..i + 2)?, 16).ok();
    let a = if h.len() == 8 { byte(6)? } else { 255 };
    Some([byte(0)?, byte(2)?, byte(4)?, a])
}

fn hex_color(s: &str) -> Option<[u8; 3]> {
    hex_rgba(s).map(|[r, g, b, _]| [r, g, b])
}

/// The root model's name, from the package's relationships.
fn root_target(rels: &str) -> Option<String> {
    let mut reader = quick_xml::Reader::from_str(rels);
    loop {
        match reader.read_event() {
            Ok(Event::Start(e) | Event::Empty(e)) if e.local_name().as_ref() == b"Relationship" => {
                if attr(&e, b"Type").is_some_and(|t| t.ends_with("/3dmodel")) {
                    return attr(&e, b"Target");
                }
            }
            Ok(Event::Eof) | Err(_) => return None,
            _ => {}
        }
    }
}

fn parse_part(r: impl Read, triangles: &mut usize) -> Result<Part> {
    let mut reader = quick_xml::Reader::from_reader(BufReader::new(r));
    let mut buf = Vec::new();
    let mut part = Part::default();
    let mut object: Option<(u32, Object)> = None;
    let mut group: Option<u32> = None;
    let mut in_build = false;
    // The `<metadata name>` whose text is being read.
    let mut meta: Option<String> = None;
    loop {
        let (e, empty) = match reader.read_event_into(&mut buf)? {
            Event::Start(e) => (e, false),
            Event::Empty(e) => (e, true),
            Event::Text(t) => {
                if let Some(name) = meta.take() {
                    let text = String::from_utf8_lossy(&t).trim().to_owned();
                    if name.eq_ignore_ascii_case("Application") {
                        part.application = Some(text.clone());
                    }
                    part.metadata.push((name, text));
                }
                buf.clear();
                continue;
            }
            Event::End(e) => {
                match e.local_name().as_ref() {
                    b"object" => {
                        if let Some((id, o)) = object.take() {
                            part.objects.insert(id, o);
                        }
                    }
                    b"basematerials" | b"colorgroup" | b"texture2dgroup" => group = None,
                    b"build" => in_build = false,
                    b"metadata" => meta = None,
                    _ => {}
                }
                buf.clear();
                continue;
            }
            Event::Eof => break,
            _ => {
                buf.clear();
                continue;
            }
        };
        match e.local_name().as_ref() {
            b"model" => part.unit = attr(&e, b"unit"),
            b"object" => {
                let mut o = Object::default();
                let mut id = None;
                each_attr(&e, |k, v| match k {
                    b"id" => id = num(v),
                    b"pid" => o.pid = num(v),
                    b"pindex" => o.pindex = num(v),
                    b"name" => o.name = Some(String::from_utf8_lossy(v).into_owned()),
                    b"type" => o.kind = Some(String::from_utf8_lossy(v).into_owned()),
                    b"partnumber" => o.partnumber = Some(String::from_utf8_lossy(v).into_owned()),
                    _ => {}
                });
                if let Some(id) = id {
                    if empty {
                        part.objects.insert(id, o);
                    } else {
                        object = Some((id, o));
                    }
                }
            }
            b"vertex" => {
                if let Some((_, o)) = &mut object {
                    let mut p = [0.0f32; 3];
                    each_attr(&e, |k, v| match k {
                        b"x" => p[0] = num(v).unwrap_or(0.0),
                        b"y" => p[1] = num(v).unwrap_or(0.0),
                        b"z" => p[2] = num(v).unwrap_or(0.0),
                        _ => {}
                    });
                    o.verts.push(p);
                }
            }
            b"triangle" => {
                if let Some((_, o)) = &mut object {
                    let mut t = Tri {
                        v: [u32::MAX; 3],
                        ..Tri::default()
                    };
                    each_attr(&e, |k, v| match k {
                        b"v1" => t.v[0] = num(v).unwrap_or(u32::MAX),
                        b"v2" => t.v[1] = num(v).unwrap_or(u32::MAX),
                        b"v3" => t.v[2] = num(v).unwrap_or(u32::MAX),
                        b"pid" => t.pid = num(v),
                        b"p1" => t.p1 = num(v),
                        b"p2" => t.p2 = num(v),
                        b"p3" => t.p3 = num(v),
                        // Bambu Studio and OrcaSlicer; PrusaSlicer (`slic3rpe:`).
                        b"paint_color" | b"mmu_segmentation" => t.paint = paint_state(v),
                        _ => {}
                    });
                    *triangles += 1;
                    ensure!(*triangles <= MAX_TRIANGLES, "too many triangles to preview");
                    o.tris.push(t);
                }
            }
            b"component" => {
                if let Some((_, o)) = &mut object {
                    let mut c = Component {
                        path: None,
                        id: u32::MAX,
                        transform: IDENTITY,
                    };
                    each_attr(&e, |k, v| match k {
                        b"objectid" => c.id = num(v).unwrap_or(u32::MAX),
                        b"transform" => c.transform = transform(v),
                        b"path" => c.path = Some(String::from_utf8_lossy(v).into_owned()),
                        _ => {}
                    });
                    o.components.push(c);
                }
            }
            kind @ (b"basematerials" | b"colorgroup" | b"texture2dgroup") => {
                group = attr(&e, b"id").and_then(|v| v.parse().ok());
                if let Some(g) = group {
                    let new = if kind == b"texture2dgroup" {
                        let texture = attr(&e, b"texid").and_then(|v| v.parse().ok());
                        Group::Coords {
                            texture: texture.unwrap_or(u32::MAX),
                            uvs: Vec::new(),
                        }
                    } else {
                        Group::Colors(Vec::new())
                    };
                    part.groups.insert(g, new);
                }
                if empty {
                    group = None;
                }
            }
            name @ (b"base" | b"color") => {
                if let Some(Group::Colors(cs)) = group.and_then(|g| part.groups.get_mut(&g)) {
                    let key: &[u8] = if name == b"base" { b"displaycolor" } else { b"color" };
                    let label = if name == b"base" { attr(&e, b"name") } else { None };
                    cs.push((attr(&e, key).and_then(|v| hex_rgba(&v)), label));
                }
            }
            b"tex2coord" => {
                if let Some(Group::Coords { uvs, .. }) = group.and_then(|g| part.groups.get_mut(&g)) {
                    let mut uv = [0.0f32; 2];
                    each_attr(&e, |k, v| match k {
                        b"u" => uv[0] = num(v).unwrap_or(0.0),
                        b"v" => uv[1] = num(v).unwrap_or(0.0),
                        _ => {}
                    });
                    uvs.push(uv);
                }
            }
            b"texture2d" => {
                if let (Some(id), Some(path)) = (attr(&e, b"id").and_then(|v| v.parse().ok()), attr(&e, b"path")) {
                    part.textures.insert(id, path);
                }
            }
            b"build" => in_build = !empty,
            b"item" if in_build => {
                let (mut id, mut transform_, mut path, mut partnumber) = (None, IDENTITY, None, None);
                each_attr(&e, |k, v| match k {
                    b"objectid" => id = num(v),
                    b"transform" => transform_ = transform(v),
                    b"path" => path = Some(String::from_utf8_lossy(v).into_owned()),
                    b"partnumber" => partnumber = Some(String::from_utf8_lossy(v).into_owned()),
                    _ => {}
                });
                if let Some(id) = id {
                    part.build.push(Item {
                        path,
                        id,
                        transform: transform_,
                        partnumber,
                    });
                }
            }
            // The model's own metadata, not an object's or item's (`metadatagroup`).
            b"metadata" if object.is_none() && !in_build => meta = if empty { None } else { attr(&e, b"name") },
            _ => {}
        }
        buf.clear();
    }
    if let Some((id, o)) = object {
        part.objects.insert(id, o);
    }
    Ok(part)
}

/// The filament most of a painted triangle prints in (0: the part's own). Slicers store the painting as a tree of
/// subdivisions in hex, its nibbles read from the end: the low two bits of a node say how many sides it splits (none:
/// a leaf; else into that many plus one), the high two a leaf's filament (3: the next nibble plus 3).
fn paint_state(code: &[u8]) -> u8 {
    fn walk(n: &[u8], pos: &mut usize, w: f32, area: &mut [f32; 19], depth: u32) -> Option<()> {
        let code = *n.get(*pos)?;
        *pos += 1;
        let sides = code & 0b11;
        if sides == 0 {
            let state = if code >> 2 == 3 {
                *pos += 1;
                n.get(*pos - 1)? + 3
            } else {
                code >> 2
            };
            area[usize::from(state)] += w;
        } else {
            if depth > 24 {
                return None;
            }
            let children = sides + 1;
            for _ in 0..children {
                walk(n, pos, w / f32::from(children), area, depth + 1)?;
            }
        }
        Some(())
    }
    let nibbles: Vec<u8> = code
        .iter()
        .rev()
        .filter_map(|&c| char::from(c).to_digit(16))
        .filter_map(|d| u8::try_from(d).ok())
        .collect();
    let mut area = [0.0f32; 19];
    walk(&nibbles, &mut 0, 1.0, &mut area, 0);
    let mut best = 0;
    for (i, a) in area.iter().enumerate() {
        if *a > area[best] {
            best = i;
        }
    }
    u8::try_from(best).unwrap_or(0)
}

/// Filament colors and assignments from a slicer project's settings files, when it has them.
fn read_slicer(zip: &mut Zip<'_>, names: &HashMap<String, String>) -> Slicer {
    let mut s = Slicer::default();
    // Bambu Studio and OrcaSlicer: JSON.
    if let Some(cfg) = text(zip, names, "Metadata/project_settings.config", MAX_CONFIG)
        && let Ok(v) = serde_json::from_str::<Value>(&cfg)
    {
        s.filaments = v
            .get("filament_colour")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|c| c.as_str().and_then(hex_color).unwrap_or(FALLBACK[0]))
            .collect();
    }
    // PrusaSlicer: `; key = value` lines; an extruder's own color wins over its filament's.
    if s.filaments.is_empty()
        && let Some(cfg) = text(zip, names, "Metadata/Slic3r_PE.config", MAX_CONFIG)
    {
        let list = |key: &str| -> Vec<Option<[u8; 3]>> {
            cfg.lines()
                .find_map(|l| l.strip_prefix("; ")?.strip_prefix(key)?.trim_start().strip_prefix('='))
                .map(|v| v.trim().trim_matches('"').split(';').map(hex_color).collect())
                .unwrap_or_default()
        };
        let (extruder, filament) = (list("extruder_colour"), list("filament_colour"));
        s.filaments = (0..extruder.len().max(filament.len()))
            .map(|i| {
                extruder
                    .get(i)
                    .copied()
                    .flatten()
                    .or(filament.get(i).copied().flatten())
                    .unwrap_or(FALLBACK[i % 6])
            })
            .collect();
    }
    for name in ["Metadata/model_settings.config", "Metadata/Slic3r_PE_model.config"] {
        if let Some(xml) = text(zip, names, name, MAX_CONFIG) {
            read_assignments(&xml, &mut s);
        }
    }
    s
}

/// `<object id>` with `<metadata key="extruder" value>`, inside it Bambu's `<part id>` and Prusa's
/// `<volume firstid lastid>`, each of which may name its own.
fn read_assignments(xml: &str, s: &mut Slicer) {
    let mut reader = quick_xml::Reader::from_str(xml);
    let (mut object, mut part, mut volume) = (None::<u32>, None::<u32>, None::<(u32, u32)>);
    loop {
        let (e, empty) = match reader.read_event() {
            Ok(Event::Start(e)) => (e, false),
            Ok(Event::Empty(e)) => (e, true),
            Ok(Event::End(e)) => {
                match e.local_name().as_ref() {
                    b"object" => object = None,
                    b"part" => part = None,
                    b"volume" => volume = None,
                    _ => {}
                }
                continue;
            }
            Ok(Event::Eof) | Err(_) => return,
            _ => continue,
        };
        let id = |k: &[u8]| attr(&e, k).and_then(|v| v.parse::<u32>().ok());
        match e.local_name().as_ref() {
            b"object" if !empty => object = id(b"id"),
            b"part" if !empty => part = id(b"id"),
            b"volume" if !empty => volume = id(b"firstid").zip(id(b"lastid")),
            b"metadata" if attr(&e, b"key").as_deref() == Some("extruder") => {
                let (Some(o), Some(n)) = (object, attr(&e, b"value").and_then(|v| v.parse::<usize>().ok())) else {
                    continue;
                };
                if n == 0 {
                    continue;
                }
                if let Some(p) = part {
                    s.parts.insert((o, p), n);
                } else if let Some((a, b)) = volume {
                    s.volumes.entry(o).or_default().push((a, b, n));
                } else {
                    s.objects.insert(o, n);
                }
            }
            _ => {}
        }
    }
}

#[cfg(all(test, feature = "fs"))]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn zip_of(files: &[(&str, &str)]) -> tempfile::NamedTempFile {
        let f = tempfile::Builder::new().suffix(".3mf").tempfile().unwrap();
        let mut zip = zip::ZipWriter::new(f.reopen().unwrap());
        for (name, body) in files {
            zip.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
        f
    }

    const RELS: &str = r#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>"#;

    const QUAD: &str = r#"<vertices><vertex x="0" y="0" z="0"/><vertex x="10" y="0" z="0"/><vertex x="10" y="10" z="0"/><vertex x="0" y="10" z="0"/></vertices>"#;

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "`as` saturates to 0..=255"
    )]
    fn colors(mesh: &crate::mesh::Mesh) -> Vec<[u8; 3]> {
        let to = |c: f32| (c.powf(1.0 / 2.2) * 255.0).round() as u8;
        mesh.materials
            .iter()
            .map(|m| [to(m.color[0]), to(m.color[1]), to(m.color[2])])
            .collect()
    }

    fn flat(doc: &Document<'_>) -> crate::mesh::Mesh {
        doc.flatten(crate::document::Pose::Rest).unwrap().unwrap()
    }

    #[test]
    fn paint_states() {
        assert_eq!(paint_state(b"4"), 1);
        assert_eq!(paint_state(b"8"), 2);
        assert_eq!(paint_state(b"0C"), 3);
        assert_eq!(paint_state(b"1C"), 4);
        // Split on three sides into four: filament 1 twice, 2 once, unpainted once.
        assert_eq!(paint_state(b"08443"), 1);
        assert_eq!(paint_state(b""), 0);
    }

    #[test]
    fn base_material_colors_and_build_transforms() {
        let model = format!(
            r##"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
            <metadata name="Application">Test Exporter</metadata>
            <resources><basematerials id="1"><base name="Red" displaycolor="#FF0000"/><base name="Blue" displaycolor="#0000FFFF"/></basematerials>
            <object id="2" type="model" pid="1" pindex="0"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/><triangle v1="0" v2="2" v3="3" pid="1" p1="1" p2="1" p3="1"/></triangles></mesh></object></resources>
            <build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 100 0 0"/><item objectid="2"/></build></model>"##
        );
        let f = zip_of(&[("_rels/.rels", RELS), ("3D/3dmodel.model", &model)]);
        let doc = load(crate::formats::Input::file(f.path())).unwrap();
        let (mesh, facts) = (flat(&doc), &doc.facts);
        assert_eq!(facts["triangles"], 4);
        assert_eq!(facts["objects"], 2);
        assert_eq!(facts["colors"], 2);
        assert_eq!(facts["application"], "Test Exporter");
        assert_eq!(colors(&mesh), [[255, 0, 0], [0, 0, 255]]);
        // The first item moved 100 mm along x.
        assert!(mesh.prims[0].positions.iter().any(|p| p[0] >= 0.1));
    }

    #[test]
    fn slicer_filaments_parts_and_paint() {
        // Bambu Studio's layout: the root object's parts live in another model file.
        let root = r#"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">
            <resources><object id="5" type="model"><components>
              <component p:path="/3D/Objects/object_1.model" objectid="1"/>
              <component p:path="/3D/Objects/object_1.model" objectid="2" transform="1 0 0 0 1 0 0 0 1 0 0 20"/>
            </components></object></resources><build><item objectid="5"/></build></model>"#;
        let parts = format!(
            r#"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>
            <object id="1" type="model"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/><triangle v1="0" v2="2" v3="3" paint_color="0C"/></triangles></mesh></object>
            <object id="2" type="model"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/></triangles></mesh></object>
            </resources></model>"#
        );
        let settings = r#"<?xml version="1.0"?><config><object id="5"><metadata key="extruder" value="1"/>
            <part id="1" subtype="normal_part"><metadata key="extruder" value="1"/></part>
            <part id="2" subtype="normal_part"><metadata key="extruder" value="2"/></part></object></config>"#;
        let project = r##"{"filament_colour": ["#FFFFFF", "#FF0000", "#00FF00"]}"##;
        let f = zip_of(&[
            ("_rels/.rels", RELS),
            ("3D/3dmodel.model", root),
            ("3D/Objects/object_1.model", &parts),
            ("Metadata/model_settings.config", settings),
            ("Metadata/project_settings.config", project),
        ]);
        let doc = load(crate::formats::Input::file(f.path())).unwrap();
        assert_eq!(doc.facts["triangles"], 3);
        // Part 1 in white with one triangle painted green; part 2 in red.
        assert_eq!(colors(&flat(&doc)), [[255, 255, 255], [0, 255, 0], [255, 0, 0]]);
    }

    #[test]
    fn prusa_volumes() {
        let model = format!(
            r#"<?xml version="1.0"?><model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>
            <object id="1" type="model"><mesh>{QUAD}<triangles><triangle v1="0" v2="1" v3="2"/><triangle v1="0" v2="2" v3="3"/></triangles></mesh></object>
            </resources><build><item objectid="1"/></build></model>"#
        );
        let cfg = r#"<?xml version="1.0"?><config><object id="1" instances_count="1"><metadata type="object" key="extruder" value="1"/>
            <volume firstid="0" lastid="0"><metadata type="volume" key="extruder" value="2"/></volume>
            <volume firstid="1" lastid="1"></volume></object></config>"#;
        let settings = "; extruder_colour = \"#0000FF;\"\n; filament_colour = \"#FFFFFF;#FFFF00\"\n";
        let f = zip_of(&[
            ("3D/3dmodel.model", &model),
            ("Metadata/Slic3r_PE_model.config", cfg),
            ("Metadata/Slic3r_PE.config", settings),
        ]);
        let doc = load(crate::formats::Input::file(f.path())).unwrap();
        // Volume 0 in extruder 2 (no color of its own: its filament's yellow); volume 1 in the object's extruder 1.
        assert_eq!(colors(&flat(&doc)), [[255, 255, 0], [0, 0, 255]]);
    }
}
