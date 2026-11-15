//! Standards conformance: every case under `tests/corpus/<format>/<case>/` is a model file (with whatever it
//! refers to) and an `expect.json` saying which part of which standard it exercises, where it came from, and what
//! the GLB made from it must hold. Each case is loaded, written as GLB, validated as glTF 2.0, decoded, and checked
//! against its expectations. See `tests/corpus/README.md` for the layout and the expectation keys.
//!
//! Each case is also read from memory, with its folder's files as [`Memory`] assets, and must read the same.

#![cfg(feature = "fs")]

mod glb;

use std::path::{Path, PathBuf};

use mithril::assets::{Folder, Memory};
use mithril::formats::{self, EXTENSIONS};
use serde_json::{Map, Value};

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

/// The cargo feature that reads `ext`, and whether this build has it.
#[allow(clippy::match_like_matches_macro)] // the arms are cfg! values that differ per build
fn enabled(ext: &str) -> bool {
    match ext {
        "stl" => cfg!(feature = "stl"),
        "ply" => cfg!(feature = "ply"),
        "vox" => cfg!(feature = "vox"),
        "obj" => cfg!(feature = "obj"),
        "3ds" => cfg!(feature = "3ds"),
        "fbx" => cfg!(feature = "fbx"),
        "3mf" => cfg!(feature = "3mf"),
        "usd" | "usda" | "usdc" | "usdz" => cfg!(feature = "usd"),
        "gltf" | "glb" => cfg!(feature = "gltf"),
        _ => false,
    }
}

/// Every model extension a full build reads, so cases are found in builds without some readers.
const ALL_EXTENSIONS: &[&str] = &[
    "stl", "ply", "vox", "obj", "3ds", "fbx", "3mf", "usd", "usda", "usdc", "usdz", "gltf", "glb",
];

fn ext_of(p: &Path) -> String {
    p.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// A case: its folder, model file and expectations.
struct Case {
    name: String,
    dir: PathBuf,
    file: PathBuf,
    expect: Map<String, Value>,
}

fn cases(format: &str) -> Vec<Case> {
    let root = corpus().join(format);
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("no corpus for {format} at {}: {e}", root.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter()
        .map(|dir| {
            let name = format!("{format}/{}", dir.file_name().unwrap().to_string_lossy());
            let text = std::fs::read_to_string(dir.join("expect.json"))
                .unwrap_or_else(|e| panic!("{name}: no expect.json: {e}"));
            let expect: Map<String, Value> =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: expect.json doesn't parse: {e}"));
            for key in ["spec", "source"] {
                assert!(
                    expect.get(key).is_some_and(Value::is_string),
                    "{name}: expect.json needs \"{key}\""
                );
            }
            let file = match expect.get("file").and_then(Value::as_str) {
                Some(f) => dir.join(f),
                None => {
                    let models: Vec<PathBuf> = std::fs::read_dir(&dir)
                        .unwrap()
                        .flatten()
                        .map(|e| e.path())
                        .filter(|p| p.is_file() && ALL_EXTENSIONS.contains(&ext_of(p).as_str()))
                        .collect();
                    assert!(
                        models.len() == 1,
                        "{name}: expected one model file, found {models:?}; name it with \"file\""
                    );
                    models.into_iter().next().unwrap()
                }
            };
            assert!(file.is_file(), "{name}: {} is missing", file.display());
            Case {
                name,
                dir,
                file,
                expect,
            }
        })
        .collect()
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).map(|f| f as f32).collect())
        .unwrap_or_default()
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
}

/// A path under `root`, `/`-separated and relative to it.
fn relative(p: &Path, root: &Path) -> String {
    p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/")
}

/// Reads the case from memory, with every file in its folder as an asset, and checks it reads as it did from its
/// file: the same GLB and facts, or an error where that failed.
fn same_from_memory(case: &Case, ext: &str, from_file: &mithril::Result<formats::Loaded>) -> Result<(), String> {
    let mut memory = Memory::new();
    let mut dirs = vec![case.dir.clone()];
    while let Some(dir) = dirs.pop() {
        for p in std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()) {
            if p.is_dir() {
                dirs.push(p);
            } else {
                memory.insert(&relative(&p, &case.dir), std::fs::read(&p).unwrap());
            }
        }
    }
    let bytes = std::fs::read(&case.file).unwrap();
    let from_memory = formats::load_bytes(&bytes, &relative(&case.file, &case.dir), ext, &memory);
    let glb = |m: &Option<mithril::mesh::Mesh>| m.as_ref().map(|m| mithril::mesh::to_glb(m).unwrap());
    match (from_file, &from_memory) {
        (Ok(a), Ok(b)) if glb(&a.mesh) == glb(&b.mesh) && a.facts == b.facts => Ok(()),
        (Ok(_), Ok(_)) => Err("reads differently from memory".into()),
        (Err(_), Err(_)) => Ok(()),
        (Ok(_), Err(err)) => Err(format!("fails to load from memory: {err}")),
        (Err(_), Ok(_)) => Err("loads from memory, though not from its file".into()),
    }
}

/// Runs one case, returning what's wrong with it.
fn run(case: &Case) -> Result<(), String> {
    let e = &case.expect;
    let ext = ext_of(&case.file);
    let tol = e.get("tolerance").and_then(Value::as_f64).unwrap_or(1e-4) as f32;
    let loaded = formats::load(&case.file, &ext, &Folder::new(&case.dir));
    same_from_memory(case, &ext, &loaded)?;
    if let Some(want) = e.get("error").and_then(Value::as_str) {
        return match loaded {
            Err(err) if err.to_string().to_lowercase().contains(&want.to_lowercase()) => Ok(()),
            Err(err) => Err(format!("failed with {err:?}, expected an error containing {want:?}")),
            Ok(_) => Err(format!("loaded, expected an error containing {want:?}")),
        };
    }
    let loaded = loaded.map_err(|err| format!("failed to load: {err}"))?;
    if let Some(Value::Object(want)) = e.get("facts") {
        for (k, v) in want {
            if loaded.facts.get(k) != Some(v) {
                return Err(format!("fact {k} is {:?}, expected {v}", loaded.facts.get(k)));
            }
        }
    }
    if e.get("no_mesh").and_then(Value::as_bool) == Some(true) {
        return match loaded.mesh {
            None => Ok(()),
            Some(_) => Err("produced a mesh, expected none".into()),
        };
    }
    let mesh = loaded.mesh.ok_or("produced no mesh")?;
    let bytes = mithril::mesh::to_glb(&mesh).map_err(|err| format!("to_glb failed: {err}"))?;
    let g = glb::validate(&bytes).map_err(|err| format!("invalid GLB: {err}"))?;
    if g.triangles() != mesh.triangles() {
        return Err(format!(
            "GLB has {} triangles, the mesh {}",
            g.triangles(),
            mesh.triangles()
        ));
    }
    if let Some(t) = loaded.facts.get("triangles")
        && t.as_u64() != Some(g.triangles() as u64)
    {
        return Err(format!("fact triangles {t} != GLB triangles {}", g.triangles()));
    }
    // Every material a primitive uses exists, and every texture's image decodes.
    for (i, (_, img)) in g.images.iter().enumerate() {
        image::load_from_memory(img).map_err(|err| format!("image {i} doesn't decode: {err}"))?;
    }

    let count = |key: &str, got: usize| -> Result<(), String> {
        match e.get(key).and_then(Value::as_u64) {
            Some(want) if want as usize != got => Err(format!("{key} is {got}, expected {want}")),
            _ => Ok(()),
        }
    };
    count("triangles", g.triangles())?;
    count("vertices", g.vertices())?;
    count("primitives", g.prims.len())?;
    count("material_count", g.materials.len())?;
    count("image_count", g.images.len())?;

    // Total surface area: catches polygons triangulated outside their outline.
    if let Some(want) = e.get("area").and_then(Value::as_f64) {
        let area: f64 = g
            .prims
            .iter()
            .flat_map(|p| {
                p.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|t| t.map(|i| p.positions[i as usize]))
            })
            .map(|[a, b, c]| {
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]].map(f64::from);
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]].map(f64::from);
                let x = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                0.5 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt()
            })
            .sum();
        if (area - want).abs() > f64::from(tol) * (1.0 + want) {
            return Err(format!("surface area is {area}, expected {want}"));
        }
    }
    if let Some(b) = e.get("bounds") {
        let (want_min, want_max) = (floats(&b["min"]), floats(&b["max"]));
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        for p in g.prims.iter().flat_map(|p| &p.positions) {
            for k in 0..3 {
                min[k] = min[k].min(p[k]);
                max[k] = max[k].max(p[k]);
            }
        }
        if !close(&min, &want_min, tol) || !close(&max, &want_max, tol) {
            return Err(format!(
                "bounds are {min:?}..{max:?}, expected {want_min:?}..{want_max:?}"
            ));
        }
    }
    let flag = |key: &str, has: bool| -> Result<(), String> {
        match e.get(key).and_then(Value::as_bool) {
            Some(want) if want != has => Err(format!("{key} is {has}, expected {want}")),
            _ => Ok(()),
        }
    };
    flag("uvs", g.prims.iter().any(|p| !p.uvs.is_empty()))?;
    flag("colors", g.prims.iter().any(|p| !p.colors.is_empty()))?;

    // Points, UVs and colors that must appear somewhere.
    let contains = |key: &str, have: &[Vec<f32>]| -> Result<(), String> {
        for want in e.get(key).and_then(Value::as_array).into_iter().flatten() {
            let want = floats(want);
            if !have.iter().any(|h| close(h, &want, tol)) {
                return Err(format!("no {key} entry matches {want:?}"));
            }
        }
        Ok(())
    };
    let all = |f: &dyn Fn(&glb::Prim) -> Vec<Vec<f32>>| -> Vec<Vec<f32>> { g.prims.iter().flat_map(f).collect() };
    contains(
        "positions_include",
        &all(&|p| p.positions.iter().map(|v| v.to_vec()).collect()),
    )?;
    contains("uvs_include", &all(&|p| p.uvs.iter().map(|v| v.to_vec()).collect()))?;
    contains(
        "colors_include",
        &all(&|p| p.colors.iter().map(|v| v.to_vec()).collect()),
    )?;
    // Every normal is one of these (a faceted model's face directions).
    if let Some(dirs) = e.get("normals_in").and_then(Value::as_array) {
        let dirs: Vec<Vec<f32>> = dirs.iter().map(floats).collect();
        // A primitive without normals is shaded flat, as the glTF spec tells clients to.
        let flat = |p: &glb::Prim| -> Vec<[f32; 3]> {
            p.indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| {
                    let [a, b, c] = t.map(|i| p.positions[i as usize]);
                    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let f = [
                        u[1] * v[2] - u[2] * v[1],
                        u[2] * v[0] - u[0] * v[2],
                        u[0] * v[1] - u[1] * v[0],
                    ];
                    let len = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
                    f.map(|x| x / len)
                })
                .filter(|n| n.iter().all(|x| x.is_finite()))
                .collect()
        };
        let normals: Vec<[f32; 3]> = g
            .prims
            .iter()
            .flat_map(|p| {
                if p.normals.is_empty() {
                    flat(p)
                } else {
                    p.normals.clone()
                }
            })
            .collect();
        for n in &normals {
            if !dirs.iter().any(|d| close(n, d, 1e-3)) {
                return Err(format!("normal {n:?} is none of {dirs:?}"));
            }
        }
    }
    // Every triangle's winding agrees with its vertex normals (counter-clockwise front faces).
    if e.get("winding_matches_normals").and_then(Value::as_bool) == Some(true) {
        // Flat-shaded primitives (no normals) take their normals from the winding: nothing to
        // compare.
        for p in g.prims.iter().filter(|p| !p.normals.is_empty()) {
            for t in p.indices.as_chunks::<3>().0 {
                let [a, b, c] = t.map(|i| p.positions[i as usize]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let face = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let n = p.normals[t[0] as usize];
                let n2 = p.normals[t[1] as usize];
                let n3 = p.normals[t[2] as usize];
                let dot = |m: [f32; 3]| face[0] * m[0] + face[1] * m[1] + face[2] * m[2];
                if dot(n) + dot(n2) + dot(n3) < 0.0 {
                    return Err(format!("triangle {t:?} winds against its normals"));
                }
            }
        }
    }

    // For a convex model: every triangle winds to face away from the middle of the bounds (catches
    // mirrored transforms and flipped files, where faces turn inside out).
    if e.get("faces_outward").and_then(Value::as_bool) == Some(true) {
        let pts = || g.prims.iter().flat_map(|p| p.positions.iter());
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for p in pts() {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let mid = [0, 1, 2].map(|k| (lo[k] + hi[k]) / 2.0);
        for p in &g.prims {
            for t in p.indices.as_chunks::<3>().0 {
                let [a, b, c] = t.map(|i| p.positions[i as usize]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let face = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let out = [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0 - mid[k]);
                if face[0] * out[0] + face[1] * out[1] + face[2] * out[2] <= 0.0 {
                    return Err(format!("triangle {t:?} faces inward"));
                }
            }
        }
    }

    for (i, want) in e
        .get("materials")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let m = match want.get("name").and_then(Value::as_str) {
            Some(name) => g.materials.iter().find(|m| m.name == name).ok_or_else(|| {
                format!(
                    "no material named {name:?} in {:?}",
                    g.materials.iter().map(|m| &m.name).collect::<Vec<_>>()
                )
            })?,
            None => g.materials.get(i).ok_or(format!("no material {i}"))?,
        };
        let label = if m.name.is_empty() {
            format!("material {i}")
        } else {
            format!("material {:?}", m.name)
        };
        if let Some(c) = want.get("color") {
            let c = floats(c);
            let tol = want.get("tolerance").and_then(Value::as_f64).map_or(2e-3, |t| t as f32);
            if !close(&m.color[..c.len()], &c, tol) {
                return Err(format!("{label} color is {:?}, expected {c:?}", m.color));
            }
        }
        if let Some(c) = want.get("emissive") {
            let c = floats(c);
            if !close(&m.emissive, &c, 2e-3) {
                return Err(format!("{label} emissive is {:?}, expected {c:?}", m.emissive));
            }
        }
        for (key, got) in [("metallic", m.metallic), ("roughness", m.roughness)] {
            if let Some(v) = want.get(key).and_then(Value::as_f64)
                && (got - v as f32).abs() > 2e-3
            {
                return Err(format!("{label} {key} is {got}, expected {v}"));
            }
        }
        for (key, got) in [
            ("textured", m.color_image.is_some()),
            ("normal_map", m.normal_image.is_some()),
            ("double_sided", m.double_sided),
        ] {
            if let Some(v) = want.get(key).and_then(Value::as_bool)
                && v != got
            {
                return Err(format!("{label} {key} is {got}, expected {v}"));
            }
        }
        if let Some(mode) = want.get("alpha_mode").and_then(Value::as_str)
            && mode != m.alpha_mode
        {
            return Err(format!("{label} alphaMode is {}, expected {mode}", m.alpha_mode));
        }
        if let Some(px) = want.get("pixel") {
            let px = floats(px);
            let img = m.color_image.ok_or(format!("{label} has no color texture to sample"))?;
            let rgb = image::load_from_memory(&g.images[img].1).unwrap().to_rgb8();
            let got = rgb.get_pixel(0, 0).0.map(f32::from);
            if !close(&got, &px, 0.02) {
                return Err(format!("{label} texture pixel is {got:?}, expected {px:?}"));
            }
        }
    }
    Ok(())
}

/// Runs a format's cases, reporting every failure at once.
fn check(format: &str) {
    let cases = cases(format);
    assert!(!cases.is_empty(), "no cases for {format}");
    let mut failures = Vec::new();
    let mut ran = 0;
    for case in &cases {
        if !enabled(&ext_of(&case.file)) {
            continue;
        }
        ran += 1;
        if let Err(why) = run(case) {
            failures.push(format!("{}: {why}", case.name));
        }
    }
    println!("{format}: {ran} of {} cases ran", cases.len());
    assert!(
        failures.is_empty(),
        "{} of {ran} {format} cases failed:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn obj() {
    check("obj");
}
#[test]
fn stl() {
    check("stl");
}
#[test]
fn ply() {
    check("ply");
}
#[test]
fn vox() {
    check("vox");
}
#[test]
fn tds() {
    check("3ds");
}
#[test]
fn threemf() {
    check("3mf");
}
#[test]
fn fbx() {
    check("fbx");
}
#[test]
fn usd() {
    check("usd");
}

#[test]
fn gltf() {
    check("gltf");
}

/// Every extension `formats::load` takes has cases, and every format a corpus folder.
#[test]
fn corpus_covers_every_extension() {
    let mut seen = std::collections::HashSet::new();
    for dir in std::fs::read_dir(corpus()).unwrap().flatten() {
        if !dir.path().is_dir() || dir.file_name() == "tools" {
            continue;
        }
        let format = dir.file_name().to_string_lossy().into_owned();
        for case in cases(&format) {
            seen.insert(ext_of(&case.file));
        }
    }
    let missing: Vec<_> = EXTENSIONS.iter().filter(|e| !seen.contains(**e)).collect();
    assert!(missing.is_empty(), "no conformance cases for {missing:?}");
}
