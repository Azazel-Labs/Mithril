# mithril

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
  `.usdz`) and glTF 2.0 (`.gltf`, `.glb`, meshopt-compressed or not). Each produces a
  `mesh::Mesh`, with textures embedded. `formats::load` reads a file and `formats::load_bytes`
  a model already in memory; both dispatch on the extension.
- **Conformance corpus**: 489 cases under `tests/corpus`, each a file and the GLB it must
  become. They cover every format above and the variants exporters really write: OBJ's face
  forms and MTL dialects, both PLY byte orders, every FBX version and axis system, and USD's
  three encodings and its composition, and glTF's accessor types, node hierarchy, skins and
  extensions. Most are generated against the specifications; the
  rest are Blender 5.2 exports. See [tests/corpus/README.md](tests/corpus/README.md).
- **Benchmarks**: `bench/` races Mithril's readers against tobj, stl_io, ply-rs and gltf on load
  time and peak heap. Run it on a clean GitHub runner with `gh workflow run bench.yml` (or
  from the Actions tab); the tables land in the run summary.
- **`mesh`**: that triangle-list model (primitives, materials, images), written out as a
  plain GLB by `mesh::to_glb`.
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
| `fs` | Reading files: `formats::load`, `assets::Folder`, `gltf::files_in` | none |

Every reader is on by default, and so is `fs`. `minimal` is the six readers with no
dependencies, plus `fs`, for `default-features = false, features = ["minimal"]`. Asking
`formats::load` for a format whose feature is off returns an error that names the feature.

Without `fs` the file APIs don't exist, so a build that has no file system (such as
`wasm32-unknown-unknown`) fails to compile rather than failing when it runs. Models load from
memory instead:

```toml
mithril = { version = "0.1", default-features = false, features = ["stl", "ply", "obj", "gltf", "usd"] }
```

```rust
use mithril::{assets::Memory, formats};

let assets = Memory::new().with("scene.bin", bin).with("albedo.png", png);
let loaded = formats::load_bytes(&gltf, "scene.gltf", "gltf", &assets)?;
```

## Planned scope

- **glTF document**: a lossless JSON document, so unknown extensions and extras survive a
  round trip. Typed views over meshes, accessors, nodes, skins and animations.
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
