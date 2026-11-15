# polysmelt

Pure-Rust 3D asset interchange, with glTF as the hub. Read what other tools write, keep what
you don't understand, and write compact or plain files back out.

## Status

Early. What exists today:

- **`meshopt`**: the `EXT_meshopt_compression` and `KHR_meshopt_compression` bitstreams. It
  has meshoptimizer's attribute (v0), triangle and index-sequence encoders, the octahedral
  filter's encoder, and decoders for every mode (attributes v0 and v1, triangles, indices)
  and filter (octahedral, quaternion, exponential, color).
- **`gltf`**: `decompress` turns a meshopt-compressed GLB into a plain one (quantization is
  kept); `compress` does the reverse, losslessly; `pack_glb` and `pack_glb_decompressed` turn
  a `.gltf` and its files into one GLB, with `files_in` as a fetch that stays inside the
  model's folder. `cargo run --example meshopt -- sample out.glb` writes a small test file.
- **`formats`**: readers for STL (binary and ASCII), PLY, MagicaVoxel `.vox`, OBJ with MTL
  materials and textures, FBX (binary and ASCII), 3DS, 3MF, USD (`.usda`, `.usdc`,
  `.usdz`) and glTF 2.0 (`.gltf`, `.glb`, meshopt- or Draco-compressed or not).
  `formats::load` reads a file and `formats::load_bytes` a model already in memory; both
  dispatch on the extension and return a `document::Document`.
- **`document`**: the common form, a glTF 2.0 document (its JSON and its buffers' bytes).
  glTF passes through untouched: nothing is decoded, and `Document::to_glb` writes the
  buffers back as they are, compression included, with images embedded. The other formats
  are converted into it. `Document::flatten(Pose)` derives a flat, posed triangle
  `mesh::Mesh` (transforms, instances, skins and morphs baked in), and `Document::counts`
  counts triangles, points and lines from the JSON alone. `Document::decode` rewrites it
  as plain glTF any reader takes: meshopt views and Draco primitives decoded, quantized
  attributes turned into floats, the scene graph, animation and materials untouched. A
  `Keep` passed to the load says what to keep: everything by default, or for instance
  `Keep { animations: false, ..Keep::ALL }` for static meshes that still carry their skins.
- **Conformance corpus**: 489 cases under `tests/corpus`, each a file and the GLB it must
  become. They cover every format above and the variants exporters really write: OBJ's face
  forms and MTL dialects, both PLY byte orders, every FBX version and axis system, and USD's
  three encodings and its composition, and glTF's accessor types, node hierarchy, skins and
  extensions. Most are generated against the specifications; the
  rest are Blender 5.2 exports. See [tests/corpus/README.md](tests/corpus/README.md). Every
  GLB the conformance tests write passes the Khronos glTF validator (`tools/validate`), which
  CI runs.
- **Benchmarks**: `bench/` races Polysmelt's readers against tobj, stl_io, ply-rs and gltf on load
  time and peak heap. Run it on a clean GitHub runner with `gh workflow run bench.yml` (or
  from the Actions tab); the tables land in the run summary.
- **`mesh`**: the flat triangle-list model `flatten` produces (primitives, materials,
  images), written out as a plain GLB by `mesh::to_glb`.
- **`assets`**: the `Assets` trait readers use to find and read referenced files (MTL
  libraries, textures, glTF buffers, USD layers) and turn texture bytes into embeddable
  images. `Folder` searches a pack's folder tree; `Memory` holds files by path, for models
  loaded from memory. By default only PNG and JPEG embed; override `Assets::image` to convert
  TGA, DDS, PSD and others.

### Cargo features

| Feature | Format | Extra dependencies |
| --- | --- | --- |
| `stl` | STL | none |
| `ply` | PLY | none |
| `vox` | MagicaVoxel | none |
| `obj` | OBJ + MTL | none |
| `3ds` | 3DS | none |
| `gltf` | glTF 2.0 + GLB | none |
| `fbx` | FBX | `fbxcel`, `base64` |
| `3mf` | 3MF | `zip`, `quick-xml` |
| `usd` | USD | `openusd`, `openusd-schemas`, `zip` |
| `draco` | Draco-compressed glTF (KHR_draco_mesh_compression) | `draco-core` |
| `fs` | Reading files: `formats::load`, `assets::Folder`, `gltf::files_in` | none |

Every reader is on by default, and so is `fs`. `minimal` is the six readers with no
dependencies, plus `fs`, for `default-features = false, features = ["minimal"]`. Asking
`formats::load` for a format whose feature is off returns an error that names the feature.

Without `fs` the file APIs don't exist, so a build that has no file system (such as
`wasm32-unknown-unknown`) fails to compile rather than failing when it runs. Models load from
memory instead:

```toml
polysmelt = { version = "0.1", default-features = false, features = ["stl", "ply", "obj", "gltf", "usd"] }
```

```rust
use polysmelt::{assets::Memory, document::{Keep, Pose}, formats};

let assets = Memory::new().with("scene.bin", bin).with("albedo.png", png);
let mut doc = formats::load_bytes(&gltf, "scene.gltf", "gltf", &assets, Keep::ALL)?;
let glb = doc.to_glb()?; // the scene, losslessly
let mesh = doc.flatten(Pose::Rest)?; // or flat triangles
doc.decode()?; // or plain glTF, for a reader without Draco, meshopt or quantization
```

### Minimum Rust

Rust 1.88, with any features but `usd`. The `usd` feature, and so the default build, needs
1.96, because `openusd` does. CI builds `minimal` on 1.88 and everything on 1.96.

## Planned scope

- **Lossless conversion**: OBJ and 3DS converted into the document with
  their hierarchy, skins, animations and full materials (glTF, FBX, USD, STL, PLY, VOX and
  3MF already are),
  units and axes recorded in extras, and readers that skip what `Keep` drops. glTF 1.0.
- **Typed views** over the document's meshes, accessors, nodes, skins and animations.
- **Accessors**: typed access through the document (the mesh reader already reads every
  component type, normalized and sparse).
- **Compression**: meshopt v1 encoding and filter encoders beyond octahedral, and
  writing `KHR_mesh_quantization`.
- **Scene**: animations read and written, and instancing kept rather than baked.
- **Writer**: a builder with plain output (any viewer, Blender) or compact output
  (quantized and meshopt-compressed).
- **Validation** of glTF documents.
- **Texture slots** (`KHR_texture_basisu`/KTX2), so a cook step can swap textures without
  touching geometry.

Out of scope: rendering, material semantics, and conversion through external tools such as
Blender.

## License

MIT-0. See [LICENSE](LICENSE).
