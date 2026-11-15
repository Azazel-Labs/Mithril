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
  materials and textures, FBX (binary and ASCII), 3DS, 3MF and USD (`.usda`, `.usdc`,
  `.usdz`). Each produces a `mesh::Mesh`, with textures embedded. `formats::load` dispatches
  on the extension.
- **Conformance corpus**: 430 cases under `tests/corpus`, each a file and the GLB it must
  become. They cover every format above and the variants exporters really write: OBJ's face
  forms and MTL dialects, both PLY byte orders, every FBX version and axis system, and USD's
  three encodings and its composition. Most are generated against the specifications; the
  rest are Blender 5.2 exports. See [tests/corpus/README.md](tests/corpus/README.md).
- **Benchmarks**: `bench/` races Mithril's readers against tobj, stl_io and ply-rs on load
  time and peak heap. Run it on a clean GitHub runner with `gh workflow run bench.yml` (or
  from the Actions tab); the tables land in the run summary.
- **`mesh`**: that triangle-list model (primitives, materials, images), written out as a
  plain GLB by `mesh::to_glb`.
- **`assets`**: the `Assets` trait readers use to find referenced files and turn texture
  bytes into embeddable images. `Folder` searches a pack's folder tree. By default only PNG
  and JPEG embed; override `Assets::image` to convert TGA, DDS, PSD and others.

### Cargo features

| Feature | Format | Extra dependencies |
| --- | --- | --- |
| `stl` | STL | none |
| `ply` | PLY | none |
| `vox` | MagicaVoxel | none |
| `obj` | OBJ + MTL | none |
| `3ds` | 3DS | none |
| `fbx` | FBX | `fbxcel`, `base64` |
| `3mf` | 3MF | `zip`, `quick-xml` |
| `usd` | USD | `openusd`, `openusd-schemas`, `zip` |

Every reader is on by default. `minimal` is the five with no dependencies, for
`default-features = false, features = ["minimal"]`. Asking `formats::load` for a format whose feature is off
returns an error that names the feature.

## Planned scope

- **glTF document**: a lossless JSON document, so unknown extensions and extras survive a
  round trip. Typed views over meshes, accessors, nodes, skins and animations.
- **Accessors**: reading every component type, normalized and sparse.
- **Compression**: meshopt v1 encoding and filter encoders beyond octahedral. Also
  `KHR_mesh_quantization` and `KHR_texture_transform` baking.
- **Scene**: `EXT_mesh_gpu_instancing`, and animations read and written.
- **Writer**: a builder with plain output (any viewer, Blender) or compact output
  (quantized and meshopt-compressed).
- **Validation** of glTF documents.
- **Texture slots** (`KHR_texture_basisu`/KTX2), so a cook step can swap textures without
  touching geometry.

Out of scope: rendering, material semantics, and conversion through external tools such as
Blender.

## License

MIT-0. See [LICENSE](LICENSE).
