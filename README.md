# mithril

Pure-Rust 3D asset interchange, with glTF as the hub. Read what other tools write, keep what
you don't understand, and write compact or plain files back out.

## Status

Early. What exists today:

- **`meshopt`**: the `EXT_meshopt_compression` bitstreams. It has meshoptimizer's attribute
  (v0) and triangle encoders, the octahedral filter's encoder, and decoders for every mode
  (attributes, triangles, indices) and filter (octahedral, quaternion, exponential).

## Planned scope

- **glTF document**: a lossless JSON document, so unknown extensions and extras survive a
  round trip. Typed views over meshes, accessors, nodes, skins and animations.
- **Containers**: GLB and `.gltf`, with a fetch callback for external buffers and images.
- **Accessors**: reading every component type, normalized and sparse.
- **Compression**: meshopt v0 and v1 codecs with every filter (including color), for both
  `EXT_meshopt_compression` and `KHR_meshopt_compression`. Also `KHR_mesh_quantization`
  and `KHR_texture_transform` baking.
- **Scene**: `EXT_mesh_gpu_instancing`, and animations read and written.
- **Writer**: a builder with plain output (any viewer, Blender) or compact output
  (quantized and meshopt-compressed).
- **Format readers**, each behind a cargo feature: STL, PLY, VOX, OBJ, FBX (binary and
  ASCII), 3DS, 3MF and USD.
- **Validation** of glTF documents.
- **Texture slots** (`KHR_texture_basisu`/KTX2), so a cook step can swap textures without
  touching geometry.

Out of scope: rendering, material semantics, and conversion through external tools such as
Blender.

## License

MIT-0. See [LICENSE-MIT0](LICENSE-MIT0).
