//! STL, binary (with facet colors) and ASCII.
//!
//! The stored facet normals are ignored: many exporters write zeros, and the vertex order is the
//! one thing the format requires to be right (counter-clockwise seen from outside). Corners at
//! the same point are shared, and the mesh carries no normals, so glTF clients shade it flat from
//! the winding, as STL means. Facets with a coordinate that isn't a finite number are dropped.
//!
//! The file is read a block at a time, never whole.

use std::collections::HashMap;

use super::Input;
use super::stream::{self, Blocks};
use crate::Result;
use crate::mesh::{GRAY, MAX_TRIANGLES, Material, Mesh, Welder, single, srgb_to_linear};

type Tri = [[f32; 3]; 3];

pub(crate) fn stl(input: Input<'_>) -> Result<Mesh> {
    let mut blocks = input.blocks()?;
    let size = blocks.size_hint() as u64;
    let head = blocks.need(1024)?;
    if let Some(n) = binary_count(head, size) {
        ensure!(n <= MAX_TRIANGLES, "too many triangles to convert ({n})");
        let header: [u8; 80] = head[..80].try_into().unwrap_or([0; 80]);
        blocks.consume(84);
        let colors = magics(&header);
        let plain = colors.is_none();
        return match binary(blocks, n, colors.unwrap_or(Colors::None))? {
            Some(mesh) => Ok(mesh),
            // Two colors flagged the VisCAM way: read it again with colors.
            None if plain => {
                let mut again = input.blocks()?;
                again.need(84)?;
                again.consume(84);
                binary(again, n, Colors::VisCam)?.ok_or_else(|| invalid!("STL changed while being read"))
            }
            None => bail!("STL changed while being read"),
        };
    }
    ascii(blocks)
}

/// The facet count when the file is binary STL, from its first bytes and its size. The 80-byte
/// header may itself start with "solid", so the size is what decides: exactly 50 bytes per
/// facet, or more (some tools pad the end) when the text doesn't read as ASCII STL.
fn binary_count(head: &[u8], size: u64) -> Option<usize> {
    let n = u32::from_le_bytes(head.get(80..84)?.try_into().ok()?);
    let need = u64::from(n) * 50 + 84;
    if need == size {
        return usize::try_from(n).ok();
    }
    (need < size && !looks_ascii(head))
        .then(|| usize::try_from(n).ok())
        .flatten()
}

/// Whether the file starts like ASCII STL: `solid`, then a `facet` or `endsolid` soon after.
fn looks_ascii(head: &[u8]) -> bool {
    let head = &head[..head.len().min(1024)];
    let starts = head
        .trim_ascii_start()
        .get(..5)
        .is_some_and(|w| w.eq_ignore_ascii_case(b"solid"));
    starts
        && head
            .windows(5)
            .skip(5)
            .any(|w| w.eq_ignore_ascii_case(b"facet") || w.eq_ignore_ascii_case(b"endso"))
}

/// How a binary STL's attribute bytes color its facets.
#[derive(Clone, Copy)]
enum Colors {
    /// Not at all, as far as is known yet.
    None,
    /// Materialise Magics: a `COLOR=rgba` header is the default for facets whose top bit is set,
    /// and the rest have red in the low bits (as three.js reads it).
    Magics([u8; 3]),
    /// `VisCAM` and `SolidView` set the top bit on colored facets, blue in the low bits. There's no
    /// header saying so, and other tools leave junk in those bytes, so only two colors or more
    /// count.
    VisCam,
}

fn magics(header: &[u8; 80]) -> Option<Colors> {
    let at = header
        .windows(6)
        .position(|w| w == b"COLOR=")
        .filter(|at| at + 10 <= 80)?;
    Some(Colors::Magics([header[at + 6], header[at + 7], header[at + 8]]))
}

impl Colors {
    /// A facet's sRGB color, 0-255.
    fn of(self, a: u16) -> Option<[u8; 3]> {
        let five = |v: u16| u8::try_from(u32::from(v & 0x1F) * 255 / 31).unwrap_or(u8::MAX);
        match self {
            Colors::None => None,
            Colors::Magics(default) => Some(if a & 0x8000 == 0 {
                [five(a), five(a >> 5), five(a >> 10)]
            } else {
                default
            }),
            Colors::VisCam => (a & 0x8000 != 0).then(|| [five(a >> 10), five(a >> 5), five(a)]),
        }
    }
}

/// `n` binary facets. `None` when reading without colors finds two `VisCAM` colors.
fn binary(mut blocks: Blocks<'_>, n: usize, colors: Colors) -> Result<Option<Mesh>> {
    let mut plain = Welder::with_capacity(if let Colors::None = colors { n } else { 0 });
    // Colored: a primitive per color, in order of first appearance.
    let mut groups: Vec<(Option<[u8; 3]>, Welder)> = Vec::new();
    let mut group: HashMap<Option<[u8; 3]>, usize> = HashMap::new();
    let mut flagged: Option<u16> = None;
    let mut left = n;
    while left > 0 {
        let data = blocks.need(50)?;
        let k = (data.len() / 50).min(left);
        ensure!(k > 0, "STL changed while being read");
        for f in data[..k * 50].as_chunks::<50>().0 {
            let a = u16::from_le_bytes([f[48], f[49]]);
            let t = facet(f);
            if !finite(&t) {
                continue;
            }
            if let Colors::None = colors {
                if a & 0x8000 != 0 && *flagged.get_or_insert(a) != a {
                    return Ok(None);
                }
                plain.push(t);
            } else {
                let c = colors.of(a);
                let g = *group.entry(c).or_insert_with(|| {
                    groups.push((c, Welder::with_capacity(0)));
                    groups.len() - 1
                });
                groups[g].1.push(t);
            }
        }
        blocks.consume(k * 50);
        left -= k;
    }
    if let Colors::None = colors {
        ensure!(plain.triangles() > 0, "STL has no triangles");
        return Ok(Some(single(plain.finish())));
    }
    ensure!(!groups.is_empty(), "STL has no triangles");
    let mut mesh = Mesh::default();
    for (i, (c, w)) in groups.into_iter().enumerate() {
        let mut p = w.finish();
        p.material = i;
        mesh.prims.push(p);
        let color = c.map_or(GRAY, |c| {
            let [r, g, b] = c.map(|v| srgb_to_linear(f32::from(v) / 255.0));
            [r, g, b, 1.0]
        });
        mesh.materials.push(Material::plain(&format!("color_{i}"), color));
    }
    Ok(Some(mesh))
}

#[inline]
fn facet(f: &[u8; 50]) -> Tri {
    let at = |k: usize| f32::from_le_bytes([f[k], f[k + 1], f[k + 2], f[k + 3]]);
    let v = |k: usize| [at(12 + k * 12), at(16 + k * 12), at(20 + k * 12)];
    [v(0), v(1), v(2)]
}

#[inline]
fn finite(t: &Tri) -> bool {
    t.as_flattened().iter().all(|c| c.is_finite())
}

/// ASCII STL: `solid`, then `facet normal ... outer loop`, three `vertex x y z`, `endloop
/// endfacet`, repeated; `endsolid`. Keywords in any case, any whitespace between words, and
/// any number of solids one after another. Only the `vertex` lines matter.
fn ascii(mut blocks: Blocks<'_>) -> Result<Mesh> {
    let head = blocks.need(5)?;
    ensure!(
        head.trim_ascii_start()
            .get(..5)
            .is_some_and(|w| w.eq_ignore_ascii_case(b"solid")),
        "not an STL file"
    );
    // Sized from the first block's bytes per facet once it's read.
    let size = blocks.size_hint();
    let mut w = Welder::with_capacity((size / 250).min(4096));
    let mut read = 0;
    let mut sized = false;
    let mut tri: Tri = [[0.0; 3]; 3];
    // The corner being read (0-2), and how many of its coordinates are still to come (0 when
    // not in a `vertex` line).
    let (mut corner, mut want) = (0, 0);
    while let Some(text) = blocks.text(stream::words)? {
        if !sized && w.triangles() > 0 {
            sized = true;
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a capacity guess from the rate so far; MAX_TRIANGLES caps it"
            )]
            let est = (size as f64 * w.triangles() as f64 / read as f64 * 1.02) as usize + 16;
            w.expect(est.min(MAX_TRIANGLES));
        }
        read += text.len();
        let mut i = 0;
        let n = text.len();
        loop {
            while i < n && stream::space(text[i]) {
                i += 1;
            }
            if i == n {
                break;
            }
            let start = i;
            while i < n && !stream::space(text[i]) {
                i += 1;
            }
            let word = &text[start..i];
            if want > 0 {
                let c = &mut tri[corner][3 - want];
                *c = stream::f32_from(word).ok_or_else(|| invalid!("bad vertex line in STL"))?;
                want -= 1;
                if want == 0 {
                    corner += 1;
                    if corner == 3 {
                        corner = 0;
                        if finite(&tri) {
                            w.push(tri);
                            ensure!(w.triangles() <= MAX_TRIANGLES, "too many triangles to convert");
                        }
                    }
                }
            } else if word.len() == 6 && word.eq_ignore_ascii_case(b"vertex") {
                want = 3;
            }
        }
    }
    ensure!(want == 0, "bad vertex line in STL");
    ensure!(w.triangles() > 0, "STL has no triangles");
    Ok(single(w.finish()))
}
