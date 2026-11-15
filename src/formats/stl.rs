//! STL, binary (with facet colors) and ASCII.

use crate::mesh::{MAX_TRIANGLES, Mesh, colored_soup, single, soup};
use crate::{Context as _, Result};

pub(crate) fn stl(data: &[u8]) -> Result<Mesh> {
    let le_f32 = |b: &[u8], at: usize| f32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
    // Binary files may also start with "solid", so trust the size arithmetic first.
    if data.len() >= 84 {
        let n = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
        if 84 + n * 50 == data.len() {
            ensure!(n <= MAX_TRIANGLES, "too many triangles to preview ({n})");
            let tris: Vec<[[f32; 3]; 3]> = (0..n)
                .map(|i| {
                    let at = 84 + i * 50 + 12;
                    let v = |k: usize| {
                        [
                            le_f32(data, at + k * 12),
                            le_f32(data, at + k * 12 + 4),
                            le_f32(data, at + k * 12 + 8),
                        ]
                    };
                    [v(0), v(1), v(2)]
                })
                .collect();
            let attrs: Vec<u16> = (0..n)
                .map(|i| u16::from_le_bytes([data[84 + i * 50 + 48], data[84 + i * 50 + 49]]))
                .collect();
            return Ok(match stl_colors(&data[..80], &attrs) {
                Some(colors) => colored_soup(&tris, &colors),
                None => single(soup(&tris)),
            });
        }
    }
    let text = std::str::from_utf8(data).context("neither binary nor ASCII STL")?;
    ensure!(text.trim_start().starts_with("solid"), "not an STL file");
    let mut verts = Vec::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if it.next() == Some("vertex") {
            let v: Vec<f32> = it.take(3).filter_map(|s| s.parse().ok()).collect();
            ensure!(v.len() == 3, "bad vertex line in STL");
            verts.push([v[0], v[1], v[2]]);
        }
    }
    ensure!(
        verts.len() / 3 <= MAX_TRIANGLES,
        "too many triangles to preview"
    );
    let tris: Vec<[[f32; 3]; 3]> = verts.as_chunks::<3>().0.to_vec();
    ensure!(!tris.is_empty(), "STL has no triangles");
    Ok(single(soup(&tris)))
}

/// Each facet's color (sRGB, 0-255) from a binary STL's attribute bytes, when the file has colors. Materialize
/// Magics writes `COLOR=rgba` in the header, the default for facets whose top bit is set, and red in the low bits of
/// the rest (as three.js reads it). `VisCAM` and `SolidView` set the top bit on colored facets, blue in the low bits;
/// there's no header saying so, and other tools leave junk in those bytes, so only two colors or more count.
fn stl_colors(header: &[u8], attrs: &[u16]) -> Option<Vec<Option<[u8; 3]>>> {
    let five = |v: u16| u8::try_from(u32::from(v & 0x1F) * 255 / 31).unwrap_or(u8::MAX);
    if let Some(at) = header
        .windows(6)
        .position(|w| w == b"COLOR=")
        .filter(|at| at + 10 <= header.len())
    {
        let default = [header[at + 6], header[at + 7], header[at + 8]];
        return Some(
            attrs
                .iter()
                .map(|&a| {
                    Some(if a & 0x8000 == 0 {
                        [five(a), five(a >> 5), five(a >> 10)]
                    } else {
                        default
                    })
                })
                .collect(),
        );
    }
    let mut seen = std::collections::HashSet::new();
    for &a in attrs.iter().filter(|&&a| a & 0x8000 != 0) {
        seen.insert(a);
        if seen.len() > 1 {
            return Some(
                attrs
                    .iter()
                    .map(|&a| (a & 0x8000 != 0).then(|| [five(a >> 10), five(a >> 5), five(a)]))
                    .collect(),
            );
        }
    }
    None
}
