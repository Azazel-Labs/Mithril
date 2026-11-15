//! MagicaVoxel `.vox`.

use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::mesh::{Material, Mesh, Prim};
use crate::{Context as _, Result};

const MAX_VOXELS: usize = 4_000_000;

/// `MagicaVoxel`: the first model, its exposed faces grouped by palette color.
pub(crate) fn vox(data: &[u8]) -> Result<(Option<Mesh>, Map<String, Value>)> {
    ensure!(
        data.len() >= 20 && &data[..4] == b"VOX ",
        "not a MagicaVoxel file"
    );
    let le = |at: usize| -> Result<u32> {
        Ok(u32::from_le_bytes(
            data.get(at..at + 4).context("VOX ends early")?.try_into()?,
        ))
    };
    let (mut size, mut voxels, mut palette, mut models) = (None, None, None, 0usize);
    // MAIN's children follow its 12-byte chunk header, after the 8-byte file header.
    let mut at = 8 + 12;
    while at + 12 <= data.len() {
        let id = &data[at..at + 4];
        let content = le(at + 4)? as usize;
        let body = at + 12;
        let next = body
            .checked_add(content)
            .context("VOX chunk overflow")?
            .checked_add(le(at + 8)? as usize)
            .context("overflow")?;
        match id {
            b"SIZE" => {
                models += 1;
                if size.is_none() {
                    size = Some((le(body)?, le(body + 4)?, le(body + 8)?));
                }
            }
            b"XYZI" if voxels.is_none() => {
                let n = le(body)? as usize;
                ensure!(n <= MAX_VOXELS, "too many voxels to preview ({n})");
                let raw = data
                    .get(body + 4..body + 4 + n * 4)
                    .context("VOX voxels end early")?;
                voxels = Some(raw.as_chunks::<4>().0.to_vec());
            }
            b"RGBA" => {
                let raw = data
                    .get(body..body + 1024)
                    .context("VOX palette ends early")?;
                palette = Some(raw.as_chunks::<4>().0.to_vec());
            }
            _ => {}
        }
        at = next;
    }
    let (sx, sy, sz) = size.context("VOX file has no model")?;
    let voxels = voxels.context("VOX file has no voxels")?;
    let occupied: std::collections::HashSet<(i32, i32, i32)> = voxels
        .iter()
        .map(|v| (i32::from(v[0]), i32::from(v[1]), i32::from(v[2])))
        .collect();
    // Color index i uses palette[i - 1]. Without a palette chunk, spread hues so shapes still read.
    let color = |i: u8| -> [f32; 4] {
        if let Some(p) = &palette {
            let c = p[usize::from(i.wrapping_sub(1))];
            return [
                f32::from(c[0]) / 255.0,
                f32::from(c[1]) / 255.0,
                f32::from(c[2]) / 255.0,
                1.0,
            ];
        }
        let h = f32::from(i) / 255.0 * 6.0;
        let x = 1.0 - (h % 2.0 - 1.0).abs();
        let (r, g, b) = match h as u32 {
            0 => (1.0, x, 0.0),
            1 => (x, 1.0, 0.0),
            2 => (0.0, 1.0, x),
            3 => (0.0, x, 1.0),
            4 => (x, 0.0, 1.0),
            _ => (1.0, 0.0, x),
        };
        [0.25 + r * 0.6, 0.25 + g * 0.6, 0.25 + b * 0.6, 1.0]
    };
    // Faces as (neighbor offset, normal, four corners); MagicaVoxel is Z-up, glTF Y-up: (x, y, z) -> (x, z, -y).
    let faces: [([i32; 3], [[f32; 3]; 4]); 6] = [
        (
            [1, 0, 0],
            [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.], [1., 0., 1.]],
        ),
        (
            [-1, 0, 0],
            [[0., 0., 0.], [0., 0., 1.], [0., 1., 1.], [0., 1., 0.]],
        ),
        (
            [0, 1, 0],
            [[0., 1., 0.], [0., 1., 1.], [1., 1., 1.], [1., 1., 0.]],
        ),
        (
            [0, -1, 0],
            [[0., 0., 0.], [1., 0., 0.], [1., 0., 1.], [0., 0., 1.]],
        ),
        (
            [0, 0, 1],
            [[0., 0., 1.], [1., 0., 1.], [1., 1., 1.], [0., 1., 1.]],
        ),
        (
            [0, 0, -1],
            [[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1., 0., 0.]],
        ),
    ];
    let to_gltf = |p: [f32; 3]| [p[0], p[2], -p[1]];
    let mut by_color: HashMap<u8, Prim> = HashMap::new();
    for v in &voxels {
        let (x, y, z) = (i32::from(v[0]), i32::from(v[1]), i32::from(v[2]));
        for (d, corners) in &faces {
            if occupied.contains(&(x + d[0], y + d[1], z + d[2])) {
                continue;
            }
            let p = by_color.entry(v[3]).or_default();
            let base = u32::try_from(p.positions.len())?;
            let normal = to_gltf([d[0] as f32, d[1] as f32, d[2] as f32]);
            for c in corners {
                p.positions
                    .push(to_gltf([x as f32 + c[0], y as f32 + c[1], z as f32 + c[2]]));
                p.normals.push(normal);
            }
            p.indices
                .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    let mut prims: Vec<(u8, Prim)> = by_color.into_iter().collect();
    prims.sort_by_key(|(c, _)| *c);
    let mut facts = Map::new();
    facts.insert("voxels".into(), json!(voxels.len()));
    facts.insert("voxel_size".into(), json!(format!("{sx}x{sy}x{sz}")));
    if models > 1 {
        facts.insert("models".into(), json!(models));
    }
    let mut mesh = Mesh::default();
    for (i, (c, mut p)) in prims.into_iter().enumerate() {
        p.material = i;
        mesh.prims.push(p);
        mesh.materials
            .push(Material::plain(&format!("color_{c}"), color(c)));
    }
    Ok((Some(mesh), facts))
}
