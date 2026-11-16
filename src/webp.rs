//! WebP images re-encoded as PNG, for the mesh writer, which embeds only PNG and JPEG.

use std::io::{Cursor, Write as _};

use flate2::{Compression, Crc, write::ZlibEncoder};

use crate::num::usize_of;

/// `bytes`, a WebP image (an animation's first frame), as PNG bytes.
pub(crate) fn to_png(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut d = image_webp::WebPDecoder::new(Cursor::new(bytes)).ok()?;
    let (w, h) = d.dimensions();
    let channels = if d.has_alpha() { 4 } else { 3 };
    let mut pixels = vec![0; d.output_buffer_size()?];
    d.read_image(&mut pixels).ok()?;
    png(w, h, channels, &pixels)
}

/// 8-bit RGB (`channels` 3) or RGBA (4) pixels as a PNG, each row filtered the way that
/// leaves the smallest residuals.
fn png(w: u32, h: u32, channels: usize, pixels: &[u8]) -> Option<Vec<u8>> {
    let row = usize_of(u64::from(w)).checked_mul(channels)?;
    if row == 0 || pixels.len() != row.checked_mul(usize_of(u64::from(h)))? {
        return None;
    }
    let mut z = ZlibEncoder::new(Vec::new(), Compression::fast());
    let (mut prev, mut best, mut cand) = (vec![0; row], vec![0; row], vec![0; row]);
    for line in pixels.chunks_exact(row) {
        let mut best_kind = 0;
        let mut best_score = u64::MAX;
        for kind in 0..5 {
            filter(kind, channels, line, &prev, &mut cand);
            let score = cand.iter().map(|&b| u64::from(b.cast_signed().unsigned_abs())).sum();
            if score < best_score {
                (best_kind, best_score) = (kind, score);
                std::mem::swap(&mut best, &mut cand);
            }
        }
        z.write_all(&[best_kind]).ok()?;
        z.write_all(&best).ok()?;
        prev.copy_from_slice(line);
    }
    let data = z.finish().ok()?;

    let mut out = Vec::with_capacity(data.len() + 64);
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut ihdr = [0; 13];
    ihdr[..4].copy_from_slice(&w.to_be_bytes());
    ihdr[4..8].copy_from_slice(&h.to_be_bytes());
    // 8 bits a channel, truecolor with or without alpha; deflate, adaptive filtering, no
    // interlacing.
    ihdr[8] = 8;
    ihdr[9] = if channels == 4 { 6 } else { 2 };
    chunk(&mut out, *b"IHDR", &ihdr)?;
    chunk(&mut out, *b"IDAT", &data)?;
    chunk(&mut out, *b"IEND", &[])?;
    Some(out)
}

/// One row through PNG filter `kind` (none, sub, up, average, Paeth) into `out`.
fn filter(kind: u8, bpp: usize, line: &[u8], prev: &[u8], out: &mut [u8]) {
    for (x, o) in out.iter_mut().enumerate() {
        let a = if x >= bpp { line[x - bpp] } else { 0 };
        let b = prev[x];
        let c = if x >= bpp { prev[x - bpp] } else { 0 };
        let predict = match kind {
            0 => 0,
            1 => a,
            2 => b,
            3 => a.midpoint(b),
            _ => paeth(a, b, c),
        };
        *o = line[x].wrapping_sub(predict);
    }
}

/// Whichever of left, up and up-left is nearest `left + up - up_left`.
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (i16::from(a), i16::from(b), i16::from(c));
    let p = ia + ib - ic;
    let (pa, pb, pc) = ((p - ia).abs(), (p - ib).abs(), (p - ic).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

fn chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) -> Option<()> {
    out.extend_from_slice(&u32::try_from(data.len()).ok()?.to_be_bytes());
    out.extend_from_slice(&kind);
    out.extend_from_slice(data);
    let mut crc = Crc::new();
    crc.update(&kind);
    crc.update(data);
    out.extend_from_slice(&crc.sum().to_be_bytes());
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pixels a filter must predict: a gradient, noise and flat runs.
    fn pixels(w: u32, h: u32, channels: usize) -> Vec<u8> {
        let mut rng = crate::test_rng::Pcg32::new(7);
        let mut p = Vec::new();
        for y in 0..h {
            for x in 0..w {
                for c in 0..channels {
                    p.push(match (x / 4 + y) % 3 {
                        0 => u8::try_from((x * 7 + y * 3 + u32::try_from(c).unwrap() * 50) % 256).unwrap(),
                        1 => rng.next_u32().to_le_bytes()[0],
                        _ => 200,
                    });
                }
            }
        }
        p
    }

    #[test]
    fn lossless_webp_comes_back_as_the_same_pixels() {
        for (channels, color) in [(4, image_webp::ColorType::Rgba8), (3, image_webp::ColorType::Rgb8)] {
            let (w, h) = (37, 23);
            let p = pixels(w, h, channels);
            let mut webp = Vec::new();
            image_webp::WebPEncoder::new(&mut webp).encode(&p, w, h, color).unwrap();
            let png = to_png(&webp).unwrap();
            let back = image::load_from_memory_with_format(&png, image::ImageFormat::Png).unwrap();
            assert_eq!((back.width(), back.height()), (w, h));
            let got = if channels == 4 {
                back.to_rgba8().into_raw()
            } else {
                back.to_rgb8().into_raw()
            };
            assert_eq!(got, p, "{channels} channels");
        }
    }

    #[test]
    fn not_webp_is_none() {
        assert!(to_png(b"\x89PNG\r\n\x1a\n").is_none());
        assert!(to_png(b"RIFF\0\0\0\0WEBP").is_none());
        assert!(png(0, 1, 3, &[]).is_none());
        assert!(png(2, 2, 3, &[0; 3]).is_none());
    }
}
