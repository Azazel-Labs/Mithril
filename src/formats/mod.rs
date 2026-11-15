//! Model formats read natively: STL (binary and ASCII), PLY (ASCII and binary), MagicaVoxel
//! `.vox`, OBJ (with its MTL materials and textures), FBX (binary and ASCII), 3DS, 3MF and USD.
//! Each becomes a [`Mesh`]; textures are embedded so the mesh stands on its own. Each format is
//! behind a cargo feature of its name (`3ds` and `3mf` for those two).

use std::path::Path;

use serde_json::{Map, Value, json};

use crate::Result;
use crate::assets::Assets;
use crate::mesh::Mesh;

#[cfg(feature = "fbx")]
mod fbx;
#[cfg(feature = "fbx")]
pub mod fbx_ascii;
#[cfg(feature = "obj")]
mod obj;
#[cfg(feature = "ply")]
mod ply;
#[cfg(feature = "stl")]
mod stl;
#[cfg(feature = "3ds")]
mod tds;
#[cfg(feature = "3mf")]
mod threemf;
#[cfg(feature = "usd")]
mod usd;
#[cfg(feature = "vox")]
mod vox;

/// The extensions [`load`] handles, whether or not this build has their feature.
pub const EXTENSIONS: &[&str] = &[
    "stl", "ply", "vox", "obj", "fbx", "3ds", "3mf", "usd", "usda", "usdc", "usdz",
];

/// A model read from a file.
#[derive(Debug, Default)]
pub struct Loaded {
    /// The geometry; `None` for files with nothing to draw (a PLY point cloud, a USD layer of
    /// overrides).
    pub mesh: Option<Mesh>,
    /// Facts worth recording: `triangles`, and per format `points`, `vertices`, `voxels`,
    /// `voxel_size`, `models`, `objects`, `colors`, `application`.
    pub facts: Map<String, Value>,
}

/// Reads the model at `path`, whose format is named by `ext` (lowercase, without the dot).
pub fn load(path: &Path, ext: &str, assets: &dyn Assets) -> Result<Loaded> {
    let counted = |m: Mesh| {
        let mut facts = Map::new();
        facts.insert("triangles".into(), json!(m.triangles()));
        Loaded {
            mesh: Some(m),
            facts,
        }
    };
    let _ = (&counted, path, assets);
    match ext {
        #[cfg(feature = "stl")]
        "stl" => stl::stl(&std::fs::read(path)?).map(counted),
        #[cfg(feature = "ply")]
        "ply" => ply::ply(path, assets).map(|(mesh, facts)| Loaded { mesh, facts }),
        #[cfg(feature = "vox")]
        "vox" => vox::vox(&std::fs::read(path)?).map(|(mesh, facts)| Loaded { mesh, facts }),
        #[cfg(feature = "3mf")]
        "3mf" => threemf::load(path).map(|(mesh, facts)| Loaded { mesh, facts }),
        #[cfg(feature = "3ds")]
        "3ds" => {
            let mut mesh = tds::load(path, assets)?;
            guess_color_texture(&mut mesh, path, assets);
            Ok(counted(mesh))
        }
        #[cfg(feature = "usd")]
        "usd" | "usda" | "usdc" | "usdz" => {
            Ok(usd::load(path, assets)?.map_or_else(Loaded::default, counted))
        }
        #[cfg(feature = "obj")]
        "obj" => Ok(counted(colors_and_textures(
            obj::obj(path, assets)?,
            path,
            ext,
            assets,
        ))),
        #[cfg(feature = "fbx")]
        "fbx" => {
            let mut mesh = fbx::load(path, assets)?;
            guess_color_texture(&mut mesh, path, assets);
            Ok(counted(mesh))
        }
        e if EXTENSIONS.contains(&e) => bail!("built without the `{e}` feature"),
        _ => bail!("not a model format handled here: .{ext}"),
    }
}

/// OBJ and FBX: colors to linear unless the file wrote them so, and a guessed color texture.
#[cfg(feature = "obj")]
fn colors_and_textures(mut mesh: Mesh, path: &Path, ext: &str, assets: &dyn Assets) -> Mesh {
    if !linear_colors(path, ext) {
        for m in &mut mesh.materials {
            for c in m.color[..3].iter_mut().chain(&mut m.emissive) {
                *c = crate::mesh::srgb_to_linear(*c);
            }
        }
    }
    guess_color_texture(&mut mesh, path, assets);
    mesh
}

#[cfg(feature = "obj")]
/// Whether the file's material colors are linear values (Blender's OBJ and FBX exporters write them so) rather
/// than the sRGB other tools write.
fn linear_colors(path: &Path, ext: &str) -> bool {
    use std::io::Read as _;
    let mut head = Vec::new();
    let Ok(f) = std::fs::File::open(path) else {
        return false;
    };
    if f.take(65_536).read_to_end(&mut head).is_err() {
        return false;
    }
    if ext == "obj" {
        let text: String = String::from_utf8_lossy(&head).chars().take(500).collect();
        text.lines().any(|l| l.starts_with("# Blender"))
    } else {
        head.windows(7).any(|w| w == b"Blender")
    }
}

#[cfg(any(feature = "obj", feature = "fbx", feature = "3ds"))]
/// A model that names no color texture takes one by name: `Barrel.png` or
/// `Barrel_Albedo.png` for `Barrel.obj`, from anywhere in the pack. Its color then shows as painted.
fn guess_color_texture(mesh: &mut Mesh, path: &Path, assets: &dyn Assets) {
    let has_uvs = mesh.prims.iter().any(|p| !p.uvs.is_empty());
    if !has_uvs || mesh.materials.iter().any(|m| m.color_image.is_some()) {
        return;
    }
    let Some(img) = assets
        .color_texture_for(path)
        .and_then(|p| assets.load_image(&p))
    else {
        return;
    };
    mesh.images.push(img);
    let i = mesh.images.len() - 1;
    for (n, m) in mesh.materials.iter_mut().enumerate() {
        if mesh
            .prims
            .iter()
            .any(|p| p.material == n && !p.uvs.is_empty())
        {
            m.color_image = Some(i);
            m.color[..3].copy_from_slice(&[1.0, 1.0, 1.0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Folder;
    #[allow(unused_imports)]
    use crate::mesh::{GRAY, to_glb};

    /// Reads `path` with assets found under `root`, as (mesh, facts).
    #[allow(dead_code)]
    fn load_in(path: &Path, ext: &str, root: &Path) -> (Option<Mesh>, Map<String, Value>) {
        let loaded = load(path, ext, &Folder::new(root)).unwrap();
        (loaded.mesh, loaded.facts)
    }

    #[allow(dead_code)]
    fn load_bytes(name: &str, bytes: &[u8]) -> (Option<Mesh>, Map<String, Value>) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join(name);
        std::fs::write(&p, bytes).unwrap();
        load_in(&p, name.rsplit('.').next().unwrap(), dir.path())
    }

    #[allow(dead_code)]
    fn png(rgb: [u8; 3]) -> Vec<u8> {
        let mut png = Vec::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb(rgb))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    #[allow(dead_code)]
    const QUAD: &str = "v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
";

    #[test]
    fn missing_feature_and_unknown_extension() {
        let err = |ext| {
            load(Path::new("x"), ext, &Folder::new("."))
                .unwrap_err()
                .to_string()
        };
        assert!(err("dae").contains("not a model format"));
        #[cfg(not(feature = "usd"))]
        assert!(err("usdz").contains("without the `usdz` feature"));
    }

    #[test]
    #[cfg(feature = "stl")]
    fn stl_binary_and_ascii() {
        let mut bin = vec![0u8; 80];
        bin.extend_from_slice(&1u32.to_le_bytes());
        for v in [0f32, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0.] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        bin.extend_from_slice(&[0, 0]);
        let (m, f) = load_bytes("a.stl", &bin);
        assert_eq!(f["triangles"], 1);
        let glb = to_glb(&m.unwrap()).unwrap();
        assert!(glb.starts_with(b"glTF"));

        let ascii = "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        let (m, _) = load_bytes("b.stl", ascii.as_bytes());
        assert_eq!(m.unwrap().triangles(), 1);
    }

    #[test]
    #[cfg(feature = "stl")]
    fn stl_facet_colors() {
        let stl = |header: &[u8], attrs: &[u16]| {
            let mut bin = header.to_vec();
            bin.resize(80, 0);
            bin.extend_from_slice(&u32::try_from(attrs.len()).unwrap().to_le_bytes());
            for &a in attrs {
                for v in [0f32, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0.] {
                    bin.extend_from_slice(&v.to_le_bytes());
                }
                bin.extend_from_slice(&a.to_le_bytes());
            }
            load_bytes("c.stl", &bin).0.unwrap()
        };
        // Magics: red in the low bits; a set top bit takes the header's color.
        let m = stl(b"COLOR=\x00\xff\x00\xff", &[0x001F, 0x8000, 0x001F]);
        assert_eq!(m.prims.len(), 2);
        assert_eq!(m.prims[0].indices.len(), 6);
        assert_eq!(m.materials[0].color, [1.0, 0.0, 0.0, 1.0]);
        let m = stl(b"COLOR=\x00\x00\xff\xff", &[0x03E0, 0x8000]);
        assert_eq!(m.materials[0].color, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(m.materials[1].color, [0.0, 0.0, 1.0, 1.0]);
        // VisCAM: the top bit marks a colored facet, blue in the low bits.
        let m = stl(b"binary", &[0x801F, 0xFC00, 0]);
        assert_eq!(
            m.materials.iter().map(|m| m.color).collect::<Vec<_>>(),
            [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0], GRAY]
        );
        // One color without a header is as likely junk as a color: plain gray.
        let m = stl(b"binary", &[0xFFFF, 0xFFFF]);
        assert_eq!((m.prims.len(), m.materials[0].color), (1, GRAY));
    }

    #[test]
    #[cfg(feature = "ply")]
    fn ply_ascii_quad_and_point_cloud() {
        let ply = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\n\
                   element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n";
        let (m, f) = load_bytes("q.ply", ply.as_bytes());
        assert_eq!(f["triangles"], 2, "a quad fans into two triangles");
        assert_eq!(m.unwrap().prims[0].normals[0], [0.0, 0.0, 1.0]);

        let mut bin = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n".to_vec();
        for v in [0f32, 0., 0., 1., 2., 3.] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        let (m, f) = load_bytes("p.ply", &bin);
        assert!(m.is_none());
        assert_eq!(f["points"], 2);
    }

    #[test]
    #[cfg(feature = "vox")]
    fn vox_exposed_faces_by_color() {
        // Two adjacent voxels with different colors: 12 faces minus the 2 touching ones.
        let chunk = |id: &[u8], body: &[u8]| {
            let mut c = id.to_vec();
            c.extend_from_slice(&(body.len() as u32).to_le_bytes());
            c.extend_from_slice(&0u32.to_le_bytes());
            c.extend_from_slice(body);
            c
        };
        let mut size = Vec::new();
        for v in [2u32, 1, 1] {
            size.extend_from_slice(&v.to_le_bytes());
        }
        let mut xyzi = 2u32.to_le_bytes().to_vec();
        xyzi.extend_from_slice(&[0, 0, 0, 1, 1, 0, 0, 2]);
        let mut rgba = vec![0u8; 1024];
        rgba[..8].copy_from_slice(&[255, 0, 0, 255, 0, 0, 255, 255]);
        let children = [
            chunk(b"SIZE", &size),
            chunk(b"XYZI", &xyzi),
            chunk(b"RGBA", &rgba),
        ]
        .concat();
        let mut file = b"VOX ".to_vec();
        file.extend_from_slice(&150u32.to_le_bytes());
        file.extend_from_slice(b"MAIN");
        file.extend_from_slice(&0u32.to_le_bytes());
        file.extend_from_slice(&(children.len() as u32).to_le_bytes());
        file.extend(children);
        let (m, f) = load_bytes("v.vox", &file);
        let m = m.unwrap();
        assert_eq!(f["voxels"], 2);
        assert_eq!(m.prims.len(), 2);
        assert_eq!(m.triangles(), 10 * 2);
        assert_eq!(m.materials[m.prims[0].material].color, [1.0, 0.0, 0.0, 1.0]);
        assert!(to_glb(&m).unwrap().starts_with(b"glTF"));
    }

    #[test]
    #[cfg(feature = "obj")]
    fn obj_mixed_faces_negative_indices_and_quads() {
        let dir = tempfile::tempdir().unwrap();
        // A quad with UVs and normals by negative index, then a triangle without UVs (a cap).
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n\
                   f -4/-4/-1 -3/-3/-1 -2/-2/-1 -1/-1/-1\nf 1 2 3\n";
        std::fs::write(dir.path().join("m.obj"), obj).unwrap();
        let (m, f) = load_in(&dir.path().join("m.obj"), "obj", dir.path());
        let m = m.unwrap();
        assert_eq!(f["triangles"], 3);
        let p = &m.prims[0];
        assert_eq!(
            p.uvs.len(),
            p.positions.len(),
            "every vertex gets a UV, missing ones zero"
        );
        assert_eq!(p.normals.len(), p.positions.len());
    }
    #[test]
    #[cfg(feature = "obj")]
    fn obj_texture_in_another_format_and_color_space() {
        let dir = tempfile::tempdir().unwrap();
        // The MTL names a TGA; the pack ships it as PNG.
        std::fs::write(dir.path().join("rock.png"), png([10, 20, 30])).unwrap();
        std::fs::write(
            dir.path().join("a.mtl"),
            "newmtl Rock\nKd 0.5 0.5 0.5\nmap_Kd rock.tga\nnewmtl Plain\nKd 0.5 0.5 0.5\n",
        )
        .unwrap();
        let obj =
            format!("mtllib a.mtl\n{QUAD}usemtl Rock\nf 1/1 2/2 3/3\nusemtl Plain\nf 1 3 4\n");
        std::fs::write(dir.path().join("a.obj"), &obj).unwrap();
        let (m, _) = load_in(&dir.path().join("a.obj"), "obj", dir.path());
        let m = m.unwrap();
        let rock = m.materials.iter().find(|x| x.name == "Rock").unwrap();
        assert_eq!(rock.color_image, Some(0));
        // sRGB 0.5 → linear ~0.214 for the glTF factor.
        let plain = m.materials.iter().find(|x| x.name == "Plain").unwrap();
        assert!((plain.color[0] - 0.214).abs() < 0.01, "{:?}", plain.color);

        // Blender writes linear values: kept as they are.
        std::fs::write(dir.path().join("a.obj"), format!("# Blender 4.2\n{obj}")).unwrap();
        let (m, _) = load_in(&dir.path().join("a.obj"), "obj", dir.path());
        let plain = m
            .unwrap()
            .materials
            .into_iter()
            .find(|x| x.name == "Plain")
            .unwrap();
        assert!((plain.color[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    #[cfg(feature = "obj")]
    fn obj_without_textures_takes_one_by_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("Textures")).unwrap();
        std::fs::write(
            dir.path().join("Textures/Barrel_Normal.png"),
            png([128, 128, 255]),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("Textures/Barrel_Albedo.png"),
            png([200, 100, 50]),
        )
        .unwrap();
        std::fs::write(dir.path().join("Textures/Barrels.png"), png([0, 0, 0])).unwrap();
        std::fs::write(
            dir.path().join("Barrel.obj"),
            format!("{QUAD}f 1/1 2/2 3/3 4/4\n"),
        )
        .unwrap();
        let (m, _) = load_in(&dir.path().join("Barrel.obj"), "obj", dir.path());
        let m = m.unwrap();
        assert_eq!(m.images.len(), 1);
        assert_eq!(m.materials[0].color_image, Some(0));
        assert_eq!(m.materials[0].color, [1.0, 1.0, 1.0, 1.0]);
        let px = image::load_from_memory(&m.images[0].bytes)
            .unwrap()
            .to_rgb8();
        assert_eq!(px.get_pixel(0, 0).0, [200, 100, 50]);
    }

    #[test]
    #[cfg(feature = "obj")]
    fn obj_with_mtl_and_texture() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("Models/textures")).unwrap();
        let mut png = Vec::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb([200, 100, 50]))
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        // Referenced by a foreign absolute path; found by name.
        std::fs::write(dir.path().join("Models/textures/crate.png"), &png).unwrap();
        std::fs::write(
            dir.path().join("Models/crate.mtl"),
            "newmtl Wood\nKd 0.9 0.8 0.7\nmap_Kd D:\\Work\\crate.png\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("Models/crate.obj"),
            "mtllib crate.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl Wood\nf 1/1 2/2 3/3 4/4\n",
        )
        .unwrap();
        let (m, f) = load_in(&dir.path().join("Models/crate.obj"), "obj", dir.path());
        let m = m.unwrap();
        assert_eq!(f["triangles"], 2);
        assert_eq!(m.materials[m.prims[0].material].name, "Wood");
        assert_eq!(m.materials[m.prims[0].material].color_image, Some(0));
        assert_eq!(m.prims[0].uvs[0], [0.0, 1.0], "V flipped");
        assert!(to_glb(&m).unwrap().starts_with(b"glTF"));
    }
}
