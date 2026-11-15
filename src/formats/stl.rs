//! STL, binary (with facet colors) and ASCII.
//!
//! The stored facet normals are ignored: many exporters write zeros, and the vertex order is the
//! one thing the format requires to be right (counter-clockwise seen from outside), so normals
//! come from the winding. Facets with a coordinate that isn't a finite number are dropped.

use crate::mesh::{MAX_TRIANGLES, Mesh, Prim, colored_soup, single};
use crate::Result;

type Tri = [[f32; 3]; 3];

pub(crate) fn stl(data: &[u8]) -> Result<Mesh> {
    if let Some(n) = binary_count(data) {
        ensure!(n <= MAX_TRIANGLES, "too many triangles to convert ({n})");
        let facets = data[84..84 + n * 50].as_chunks::<50>().0;
        let attrs: Vec<u16> = facets.iter().map(|f| u16::from_le_bytes([f[48], f[49]])).collect();
        if let Some(colors) = stl_colors(&data[..80], &attrs) {
            let (tris, colors): (Vec<Tri>, Vec<_>) = facets
                .iter()
                .map(facet)
                .zip(colors)
                .filter(|(t, _)| finite(t))
                .unzip();
            ensure!(!tris.is_empty(), "STL has no triangles");
            return Ok(colored_soup(&tris, &colors));
        }
        let mut p = Prim::default();
        p.reserve(n * 3);
        for f in facets {
            p.push_flat(facet(f));
        }
        ensure!(!p.indices.is_empty(), "STL has no triangles");
        return Ok(single(p));
    }
    ascii(data)
}

/// The facet count when `data` is binary STL. The 80-byte header may itself start with "solid",
/// so the size is what decides: exactly 50 bytes per facet, or more (some tools pad the end)
/// when the text doesn't read as ASCII STL.
fn binary_count(data: &[u8]) -> Option<usize> {
    let n = u32::from_le_bytes(data.get(80..84)?.try_into().ok()?) as usize;
    let need = n.checked_mul(50)?.checked_add(84)?;
    if need == data.len() {
        return Some(n);
    }
    (need < data.len() && !looks_ascii(data)).then_some(n)
}

/// Whether the file starts like ASCII STL: `solid`, then a `facet` or `endsolid` soon after.
fn looks_ascii(data: &[u8]) -> bool {
    let head = &data[..data.len().min(1024)];
    let starts = head.trim_ascii_start().get(..5).is_some_and(|w| w.eq_ignore_ascii_case(b"solid"));
    starts
        && head
            .windows(5)
            .skip(5)
            .any(|w| w.eq_ignore_ascii_case(b"facet") || w.eq_ignore_ascii_case(b"endso"))
}

fn facet(f: &[u8; 50]) -> Tri {
    let at = |k: usize| f32::from_le_bytes([f[k], f[k + 1], f[k + 2], f[k + 3]]);
    let v = |k: usize| [at(12 + k * 12), at(16 + k * 12), at(20 + k * 12)];
    [v(0), v(1), v(2)]
}

fn finite(t: &Tri) -> bool {
    t.as_flattened().iter().all(|c| c.is_finite())
}

impl Prim {
    /// Room for `n` more unshared vertices.
    fn reserve(&mut self, n: usize) {
        self.positions.reserve(n);
        self.normals.reserve(n);
        self.indices.reserve(n);
    }

    /// One flat-shaded triangle, unless a coordinate isn't finite.
    fn push_flat(&mut self, t: Tri) {
        if !finite(&t) {
            return;
        }
        let n = crate::mesh::face_normal(t[0], t[1], t[2]);
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&t);
        self.normals.extend_from_slice(&[n; 3]);
        self.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

/// ASCII STL: `solid`, then `facet normal ... outer loop`, three `vertex x y z`, `endloop
/// endfacet`, repeated; `endsolid`. Keywords in any case, any whitespace between words, and
/// any number of solids one after another. Only the `vertex` lines matter.
fn ascii(data: &[u8]) -> Result<Mesh> {
    ensure!(
        data.trim_ascii_start().get(..5).is_some_and(|w| w.eq_ignore_ascii_case(b"solid")),
        "not an STL file"
    );
    let mut p = Prim::default();
    p.reserve(data.len() / 80);
    let mut words = data.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty());
    let mut tri: Tri = [[0.0; 3]; 3];
    let mut k = 0;
    while let Some(w) = words.next() {
        if !w.eq_ignore_ascii_case(b"vertex") {
            continue;
        }
        for c in &mut tri[k] {
            let num = words
                .next()
                .and_then(|w| std::str::from_utf8(w).ok())
                .and_then(|s| s.parse().ok());
            *c = num.ok_or_else(|| invalid!("bad vertex line in STL"))?;
        }
        k += 1;
        if k == 3 {
            p.push_flat(tri);
            k = 0;
            ensure!(p.indices.len() / 3 <= MAX_TRIANGLES, "too many triangles to convert");
        }
    }
    ensure!(!p.indices.is_empty(), "STL has no triangles");
    Ok(single(p))
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
