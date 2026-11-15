//! The `EXT_meshopt_compression` bitstreams: meshoptimizer's attribute (v0) and triangle (v1)
//! encoders, the octahedral filter's encoder, and decoders for every mode and filter.

use crate::{Error, Result};

/// Elements per delta group.
const GROUP: usize = 16;

/// The triangle codec's `codeaux` table, meshoptimizer's (built from symbol frequencies).
const CODEAUX: [u8; 16] = [
    0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0, 0,
];

/// Rotations that put a matched edge first.
const ORDER: [[usize; 3]; 3] = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];

fn bad(what: &str) -> Error {
    Error::InvalidInput(format!("EXT_meshopt_compression: {what}"))
}

fn zigzag8(v: u8) -> u8 {
    if v & 0x80 != 0 { !(v << 1) } else { v << 1 }
}

fn unzigzag8(v: u8) -> u8 {
    if v & 1 != 0 { !(v >> 1) } else { v >> 1 }
}

/// Elements per attribute block.
fn block_elements(stride: usize) -> usize {
    ((8192 / stride) & !15).min(256)
}

/// Mode 0 (`ATTRIBUTES`): `data` holds elements of `stride` bytes (a multiple of 4, at most
/// 256). Each byte position is delta coded against the previous element in groups of 16.
pub fn encode_vertex_buffer(data: &[u8], stride: usize) -> Vec<u8> {
    assert!(stride > 0 && stride <= 256 && stride.is_multiple_of(4));
    assert!(data.len().is_multiple_of(stride));
    let count = data.len() / stride;
    let first = if count > 0 {
        data[..stride].to_vec()
    } else {
        vec![0; stride]
    };
    let mut out = vec![0xa0];
    let mut last = first.clone();
    let mut deltas = [0u8; 256];
    let mut start = 0;
    while start < count {
        let n = (count - start).min(block_elements(stride));
        let padded = n.div_ceil(GROUP) * GROUP;
        for (k, prev) in last.iter_mut().enumerate() {
            for (i, d) in deltas[..n].iter_mut().enumerate() {
                let v = data[(start + i) * stride + k];
                *d = zigzag8(v.wrapping_sub(*prev));
                *prev = v;
            }
            deltas[n..padded].fill(0);
            encode_bytes(&mut out, &deltas[..padded]);
        }
        start += n;
    }
    // Tail: the baseline element, padded in front to 32 bytes.
    out.resize(out.len() + 32usize.saturating_sub(stride), 0);
    out.extend_from_slice(&first);
    out
}

/// One data block: 2 header bits per group, then each group in the smallest of its modes.
fn encode_bytes(out: &mut Vec<u8>, deltas: &[u8]) {
    let header = out.len();
    out.resize(header + (deltas.len() / GROUP).div_ceil(4), 0);
    for (g, d) in deltas.chunks_exact(GROUP).enumerate() {
        let size = |bits: usize| match bits {
            0 if d.iter().all(|&x| x == 0) => 0,
            0 => usize::MAX,
            8 => GROUP,
            b => {
                let sentinel = (1u8 << b) - 1;
                GROUP * b / 8 + d.iter().filter(|&&x| x >= sentinel).count()
            }
        };
        let (mode, bits) = [(0u8, 0usize), (1, 2), (2, 4), (3, 8)]
            .into_iter()
            .min_by_key(|&(_, b)| size(b))
            .unwrap();
        out[header + g / 4] |= mode << ((g % 4) * 2);
        match bits {
            0 => {}
            8 => out.extend_from_slice(d),
            b => {
                let sentinel = (1u8 << b) - 1;
                let per = 8 / b;
                for chunk in d.chunks_exact(per) {
                    let mut byte = 0u8;
                    for &x in chunk {
                        byte = (byte << b) | x.min(sentinel);
                    }
                    out.push(byte);
                }
                out.extend(d.iter().filter(|&&x| x >= sentinel));
            }
        }
    }
}

/// Group bit widths per header value: version 0, then version 1 under control modes 0 and 1.
const BITS_V0: [u8; 4] = [0, 2, 4, 8];
const BITS_V1: [[u8; 4]; 2] = [[0, 1, 2, 4], [1, 2, 4, 8]];

/// Decodes mode 0 into `count` elements of `stride` bytes. Reads both attribute codec versions:
/// 0 (`EXT_meshopt_compression`, header `0xa0`) and 1 (`KHR_meshopt_compression`, `0xa1`).
pub fn decode_vertex_buffer(src: &[u8], count: usize, stride: usize) -> Result<Vec<u8>> {
    let size = count.checked_mul(stride).ok_or_else(|| bad("attribute count"))?;
    // A block of at most 256 elements costs at least a quarter byte per element byte, so a
    // stream can't hold more than 1024 times its size: refuse before allocating.
    if size / 1024 > src.len() {
        return Err(bad("attribute data truncated"));
    }
    let mut out = vec![0u8; size];
    decode_vertex_buffer_into(src, &mut out, stride)?;
    Ok(out)
}

/// Decodes mode 0 into `out`, whole elements of `stride` bytes, without allocating.
pub fn decode_vertex_buffer_into(src: &[u8], out: &mut [u8], stride: usize) -> Result<()> {
    if stride == 0 || stride > 256 || !stride.is_multiple_of(4) || !out.len().is_multiple_of(stride) {
        return Err(bad("attribute stride"));
    }
    let count = out.len() / stride;
    let header = *src.first().ok_or_else(|| bad("attribute header"))?;
    if header & 0xf0 != 0xa0 {
        return Err(bad("attribute header"));
    }
    let v1 = match header & 15 {
        0 => false,
        1 => true,
        v => return Err(bad(&format!("unsupported attribute codec version {v}"))),
    };
    let channels_len = if v1 { stride / 4 } else { 0 };
    let tail = stride + channels_len;
    let tail_pad = tail.max(if v1 { 24 } else { 32 });
    if src.len() < 1 + tail_pad {
        return Err(bad("attribute data truncated"));
    }
    // Block data must end exactly where the (padded) tail starts.
    let end = src.len() - tail_pad;
    let tail_at = src.len() - tail;
    let mut last = [0u8; 256];
    last[..stride].copy_from_slice(&src[tail_at..tail_at + stride]);
    let channels = &src[tail_at + stride..];
    if channels.iter().any(|&c| c & 3 == 3) {
        return Err(bad("attribute channel mode"));
    }
    let mut pos = 1;
    // One byte plane per byte of a 4-byte channel, padded to whole groups.
    let mut planes = [[0u8; 256]; 4];
    let mut start = 0;
    while start < count {
        let n = (count - start).min(block_elements(stride));
        let groups = n.div_ceil(GROUP);
        let control = pos;
        pos += channels_len;
        if pos > end {
            return Err(bad("attribute data truncated"));
        }
        for k in (0..stride).step_by(4) {
            let ctrl_byte = if v1 { src[control + k / 4] } else { 0 };
            for (j, plane) in planes.iter_mut().enumerate() {
                let ctrl = (ctrl_byte >> (j * 2)) & 3;
                match ctrl {
                    // Literal bytes, unpadded.
                    3 => {
                        if pos + n > end {
                            return Err(bad("attribute data truncated"));
                        }
                        plane[..n].copy_from_slice(&src[pos..pos + n]);
                        pos += n;
                    }
                    2 => plane[..n].fill(0),
                    _ => {
                        let bits = if v1 { &BITS_V1[ctrl as usize] } else { &BITS_V0 };
                        let header = pos;
                        pos += groups.div_ceil(4);
                        if pos > end {
                            return Err(bad("attribute data truncated"));
                        }
                        for g in 0..groups {
                            let h = (src[header + g / 4] >> ((g % 4) * 2)) & 3;
                            let d = &mut plane[g * GROUP..(g + 1) * GROUP];
                            pos = decode_group(src, pos, end, bits[h as usize], d)?;
                        }
                    }
                }
            }
            let channel = if v1 { channels[k / 4] } else { 0 };
            let prev = &mut last[k..k + 4];
            let rows = out[start * stride..].chunks_exact_mut(stride).take(n);
            match channel & 3 {
                // Byte deltas.
                0 => {
                    for (i, row) in rows.enumerate() {
                        for j in 0..4 {
                            prev[j] = prev[j].wrapping_add(unzigzag8(planes[j][i]));
                        }
                        row[k..k + 4].copy_from_slice(prev);
                    }
                }
                // Two 16-bit deltas.
                1 => {
                    let mut p = [
                        u16::from_le_bytes([prev[0], prev[1]]),
                        u16::from_le_bytes([prev[2], prev[3]]),
                    ];
                    for (i, row) in rows.enumerate() {
                        for h in 0..2 {
                            let v = u16::from_le_bytes([planes[2 * h][i], planes[2 * h + 1][i]]);
                            let d = if v & 1 != 0 { !(v >> 1) } else { v >> 1 };
                            p[h] = p[h].wrapping_add(d);
                            row[k + 2 * h..k + 2 * h + 2].copy_from_slice(&p[h].to_le_bytes());
                        }
                    }
                    prev[..2].copy_from_slice(&p[0].to_le_bytes());
                    prev[2..].copy_from_slice(&p[1].to_le_bytes());
                }
                // A 32-bit XOR delta, rotated.
                _ => {
                    let rot = u32::from(channel >> 4);
                    let mut p = u32::from_le_bytes([prev[0], prev[1], prev[2], prev[3]]);
                    for (i, row) in rows.enumerate() {
                        let v = u32::from_le_bytes([planes[0][i], planes[1][i], planes[2][i], planes[3][i]]);
                        p ^= v.rotate_right(rot);
                        row[k..k + 4].copy_from_slice(&p.to_le_bytes());
                    }
                    prev.copy_from_slice(&p.to_le_bytes());
                }
            }
        }
        start += n;
    }
    if pos != end {
        return Err(bad("attribute data has trailing bytes"));
    }
    Ok(())
}

/// Decodes one group of 16 deltas packed `bits` wide (0, 1, 2, 4 or 8) into `d`.
fn decode_group(src: &[u8], mut pos: usize, end: usize, bits: u8, d: &mut [u8]) -> Result<usize> {
    let truncated = || bad("attribute data truncated");
    match bits {
        0 => d.fill(0),
        8 => {
            if pos + GROUP > end {
                return Err(truncated());
            }
            d.copy_from_slice(&src[pos..pos + GROUP]);
            pos += GROUP;
        }
        _ => {
            let bits = usize::from(bits);
            let sentinel = (1u8 << bits) - 1;
            let packed = GROUP * bits / 8;
            if pos + packed > end {
                return Err(truncated());
            }
            let per = 8 / bits;
            for (i, x) in d.iter_mut().enumerate() {
                let byte = src[pos + i / per];
                // 1-bit groups run least significant bit first, wider ones most significant.
                let shift = if bits == 1 { i % 8 } else { 8 - bits * (i % per + 1) };
                *x = (byte >> shift) & sentinel;
            }
            pos += packed;
            for x in d.iter_mut().filter(|x| **x == sentinel) {
                if pos >= end {
                    return Err(truncated());
                }
                *x = src[pos];
                pos += 1;
            }
        }
    }
    Ok(pos)
}

/// The triangle codec's recent edges and vertices.
struct Fifos {
    edges: [[u32; 2]; 16],
    edge_at: usize,
    verts: [u32; 16],
    vert_at: usize,
}

impl Fifos {
    fn new() -> Self {
        Fifos {
            edges: [[u32::MAX; 2]; 16],
            edge_at: 0,
            verts: [u32::MAX; 16],
            vert_at: 0,
        }
    }
    /// The `i`th most recent edge.
    fn edge_at(&self, i: usize) -> [u32; 2] {
        self.edges[self.edge_at.wrapping_sub(1 + i) & 15]
    }
    /// The `i`th most recent vertex.
    fn vert(&self, i: usize) -> u32 {
        self.verts[self.vert_at.wrapping_sub(1 + i) & 15]
    }
    /// The most recent edge of triangle (a, b, c), and which rotation of it starts there.
    fn find_edge(&self, a: u32, b: u32, c: u32) -> Option<(usize, usize)> {
        (0..16).find_map(|i| match self.edge_at(i) {
            e if e == [a, b] => Some((i, 0)),
            e if e == [b, c] => Some((i, 1)),
            e if e == [c, a] => Some((i, 2)),
            _ => None,
        })
    }
    fn find_vert(&self, v: u32) -> Option<usize> {
        (0..16).find(|&i| self.vert(i) == v)
    }
    fn push_edge(&mut self, a: u32, b: u32) {
        self.edges[self.edge_at] = [a, b];
        self.edge_at = (self.edge_at + 1) & 15;
    }
    fn push_vert(&mut self, v: u32) {
        self.verts[self.vert_at] = v;
        self.vert_at = (self.vert_at + 1) & 15;
    }
}

fn write_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn read_varint(src: &[u8], pos: &mut usize, end: usize) -> Result<u32> {
    let mut v = 0u32;
    for shift in (0..35).step_by(7) {
        let byte = *src
            .get(*pos)
            .filter(|_| *pos < end)
            .ok_or_else(|| bad("index data truncated"))?;
        *pos += 1;
        v |= ((byte & 0x7f) as u32) << shift;
        if byte < 0x80 {
            return Ok(v);
        }
    }
    Err(bad("index varint too long"))
}

/// `index` as a zigzag delta from `last`.
fn write_index(out: &mut Vec<u8>, index: u32, last: u32) {
    let d = index.wrapping_sub(last);
    write_varint(out, (d << 1) ^ ((d as i32 >> 31) as u32));
}

fn read_index(src: &[u8], pos: &mut usize, end: usize, last: &mut u32) -> Result<u32> {
    let v = read_varint(src, pos, end)?;
    let delta = if v & 1 != 0 { !(v >> 1) } else { v >> 1 };
    *last = last.wrapping_add(delta);
    Ok(*last)
}

/// Mode 1 (`TRIANGLES`): a triangle list, most triangles in one byte when the list is ordered
/// for the vertex cache and its vertices by first use.
pub fn encode_index_buffer(indices: &[u32]) -> Vec<u8> {
    assert!(indices.len().is_multiple_of(3));
    let mut codes = Vec::with_capacity(indices.len() / 3);
    let mut data = Vec::new();
    let mut f = Fifos::new();
    let (mut next, mut last) = (0u32, 0u32);
    for t in indices.chunks_exact(3) {
        if let Some((fe, rot)) = f.find_edge(t[0], t[1], t[2]).filter(|&(fe, _)| fe < 15) {
            // A recent edge and a third vertex: recent, next, last ± 1 or explicit.
            let [a, b, c] = ORDER[rot].map(|k| t[k]);
            let mut fec = match f.find_vert(c) {
                Some(i) if (1..13).contains(&i) => i,
                _ if c == next => {
                    next += 1;
                    0
                }
                _ => 15,
            };
            if fec == 15 {
                if c.wrapping_add(1) == last {
                    fec = 13;
                    last = c;
                }
                if c == last.wrapping_add(1) {
                    fec = 14;
                    last = c;
                }
            }
            codes.push((fe << 4 | fec) as u8);
            if fec == 15 {
                write_index(&mut data, c, last);
                last = c;
            }
            if fec == 0 || fec >= 13 {
                f.push_vert(c);
            }
            f.push_edge(c, b);
            f.push_edge(a, c);
        } else {
            // No recent edge: three vertices, each recent, next or explicit.
            let rot = if t[1] == next {
                1
            } else if t[2] == next {
                2
            } else {
                0
            };
            let [a, b, c] = ORDER[rot].map(|k| t[k]);
            // 0, 1, 2 after other vertices restarts `next` (concatenated streams).
            let reset = [a, b, c] == [0, 1, 2] && next > 0;
            if reset {
                next = 0;
                f.verts = [u32::MAX; 16];
            }
            let (fb, fc) = (f.find_vert(b), f.find_vert(c));
            let mut take = |v: u32, fifo: Option<usize>| match fifo {
                Some(i) if i < 14 => i + 1,
                _ if v == next => {
                    next += 1;
                    0
                }
                _ => 15,
            };
            let fea = take(a, None);
            let feb = take(b, fb);
            let fec = take(c, fc);
            let aux = (feb << 4 | fec) as u8;
            match CODEAUX.iter().position(|&x| x == aux) {
                Some(i) if fea == 0 && i < 14 && !reset => codes.push(0xf0 | i as u8),
                _ => {
                    codes.push(if fea == 0 { 0xfe } else { 0xff });
                    data.push(aux);
                }
            }
            for (v, fe) in [(a, fea), (b, feb), (c, fec)] {
                if fe == 15 {
                    write_index(&mut data, v, last);
                    last = v;
                }
            }
            for (v, fe) in [(a, fea), (b, feb), (c, fec)] {
                if fe == 0 || fe == 15 {
                    f.push_vert(v);
                }
            }
            f.push_edge(b, a);
            f.push_edge(c, b);
            f.push_edge(a, c);
        }
    }
    let mut out = Vec::with_capacity(1 + codes.len() + data.len() + 16);
    out.push(0xe1);
    out.extend_from_slice(&codes);
    out.extend_from_slice(&data);
    out.extend_from_slice(&CODEAUX);
    out
}

/// Decodes mode 1 into `count` indices.
pub fn decode_index_buffer(src: &[u8], count: usize) -> Result<Vec<u32>> {
    let tris = count / 3;
    if !count.is_multiple_of(3) || src.len() < 1 + tris + 16 || src[0] & 0xf0 != 0xe0 {
        return Err(bad("triangle header"));
    }
    if src[0] & 15 != 1 {
        return Err(bad("unsupported triangle codec version"));
    }
    let end = src.len() - 16;
    let table = &src[end..];
    let mut pos = 1 + tris;
    let mut f = Fifos::new();
    let (mut next, mut last) = (0u32, 0u32);
    let mut out = Vec::with_capacity(count);
    for &code in &src[1..1 + tris] {
        let (hi, lo) = ((code >> 4) as usize, (code & 15) as usize);
        let tri = if hi < 15 {
            let [a, b] = f.edge_at(hi);
            let c = match lo {
                0 => {
                    next += 1;
                    next - 1
                }
                1..=12 => f.vert(lo),
                13 => {
                    last = last.wrapping_sub(1);
                    last
                }
                14 => {
                    last = last.wrapping_add(1);
                    last
                }
                _ => read_index(src, &mut pos, end, &mut last)?,
            };
            if lo == 0 || lo >= 13 {
                f.push_vert(c);
            }
            f.push_edge(c, b);
            f.push_edge(a, c);
            [a, b, c]
        } else {
            let aux = if lo < 14 {
                table[lo]
            } else {
                let x = *src
                    .get(pos)
                    .filter(|_| pos < end)
                    .ok_or_else(|| bad("index data truncated"))?;
                pos += 1;
                if x == 0 {
                    next = 0;
                }
                x
            };
            let (feb, fec) = ((aux >> 4) as usize, (aux & 15) as usize);
            let fea = if lo == 15 { 15 } else { 0 };
            // FIFO reads and `next` increments happen in a, b, c order, before any pushes.
            let mut pick = |fe: usize| -> Option<u32> {
                match fe {
                    0 => {
                        next += 1;
                        Some(next - 1)
                    }
                    15 => None,
                    i => Some(f.vert(i - 1)),
                }
            };
            let (a, b, c) = (pick(fea), pick(feb), pick(fec));
            let mut explicit = |v: Option<u32>| match v {
                Some(v) => Ok(v),
                None => read_index(src, &mut pos, end, &mut last),
            };
            let (a, b, c) = (explicit(a)?, explicit(b)?, explicit(c)?);
            f.push_vert(a);
            if feb == 0 || feb == 15 {
                f.push_vert(b);
            }
            if fec == 0 || fec == 15 {
                f.push_vert(c);
            }
            f.push_edge(b, a);
            f.push_edge(c, b);
            f.push_edge(a, c);
            [a, b, c]
        };
        out.extend_from_slice(&tri);
    }
    Ok(out)
}

/// Decodes mode 2 (`INDICES`, any index sequence) into `count` indices.
pub fn decode_index_sequence(src: &[u8], count: usize) -> Result<Vec<u32>> {
    if src.len() < 1 + count + 4 || src[0] != 0xd1 {
        return Err(bad("index sequence header"));
    }
    let end = src.len() - 4;
    let mut pos = 1;
    let mut last = [0u32; 2];
    (0..count)
        .map(|_| {
            let v = read_varint(src, &mut pos, end)?;
            let base = (v & 1) as usize;
            let delta = if v & 2 != 0 { !(v >> 2) } else { v >> 2 };
            last[base] = last[base].wrapping_add(delta);
            Ok(last[base])
        })
        .collect()
}

/// Mode 2 (`INDICES`): any index sequence (strips, lines, points, sparse accessor indices).
/// Fails only when an index lies more than 2^30 from both of the two running baselines.
pub fn encode_index_sequence(indices: &[u32]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(indices.len() + 5);
    out.push(0xd1);
    let mut last = [0u32; 2];
    for &i in indices {
        let d = [0, 1].map(|b| i.wrapping_sub(last[b]) as i32);
        let base = usize::from(d[1].unsigned_abs() < d[0].unsigned_abs());
        let d = d[base];
        if !(-(1 << 30)..1 << 30).contains(&d) {
            return Err(bad("index sequence delta out of range"));
        }
        let zz = ((d << 1) ^ (d >> 31)) as u32;
        write_varint(&mut out, (zz << 1) | base as u32);
        last[base] = i;
    }
    out.extend_from_slice(&[0; 4]);
    Ok(out)
}

fn snorm(v: f32, bits: u32) -> i32 {
    let scale = ((1 << (bits - 1)) - 1) as f32;
    let v = v.clamp(-1.0, 1.0);
    (v * scale + if v >= 0.0 { 0.5 } else { -0.5 }) as i32
}

/// The octahedral filter's 8-bit input for unit vector `n` and fourth component `w`
/// (passed through; ±1 for a tangent's handedness).
pub fn encode_oct8(n: [f32; 3], w: f32) -> [u8; 4] {
    let l = n[0].abs() + n[1].abs() + n[2].abs();
    let s = if l == 0.0 { 0.0 } else { 1.0 / l };
    let (x, y, z) = (n[0] * s, n[1] * s, n[2] * s);
    let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let u = if z >= 0.0 { x } else { (1.0 - y.abs()) * sign(x) };
    let v = if z >= 0.0 { y } else { (1.0 - x.abs()) * sign(y) };
    [snorm(u, 8), snorm(v, 8), snorm(1.0, 8), snorm(w, 8)].map(|c| c as i8 as u8)
}

/// Applies the named filter to decoded attribute data in place.
pub fn apply_filter(filter: &str, data: &mut [u8], stride: usize) -> Result<()> {
    match filter {
        "NONE" => {}
        "OCTAHEDRAL" if stride == 4 => {
            for e in data.chunks_exact_mut(4) {
                let [x, y, z] = octahedral([e[0] as i8 as f32, e[1] as i8 as f32, e[2] as i8 as f32], 127.0);
                e[..3].copy_from_slice(&[x as i8 as u8, y as i8 as u8, z as i8 as u8]);
            }
        }
        "OCTAHEDRAL" if stride == 8 => {
            for e in data.chunks_exact_mut(8) {
                let c = |i: usize| i16::from_le_bytes([e[2 * i], e[2 * i + 1]]) as f32;
                let out = octahedral([c(0), c(1), c(2)], 32767.0);
                for (i, v) in out.into_iter().enumerate() {
                    e[2 * i..2 * i + 2].copy_from_slice(&(v as i16).to_le_bytes());
                }
            }
        }
        "QUATERNION" if stride == 8 => {
            for e in data.chunks_exact_mut(8) {
                let c = |i: usize| i16::from_le_bytes([e[2 * i], e[2 * i + 1]]);
                let range = std::f32::consts::FRAC_1_SQRT_2;
                let one = (c(3) | 3) as f32;
                let x = c(0) as f32 / one * range;
                let y = c(1) as f32 / one * range;
                let z = c(2) as f32 / one * range;
                let w = (1.0 - x * x - y * y - z * z).max(0.0).sqrt();
                let max = (c(3) & 3) as usize;
                let mut out = [0i16; 4];
                for (k, v) in [(1, x), (2, y), (3, z), (0, w)] {
                    out[(max + k) % 4] = (v * 32767.0).round() as i16;
                }
                for (i, v) in out.into_iter().enumerate() {
                    e[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        "EXPONENTIAL" if stride.is_multiple_of(4) => {
            for e in data.chunks_exact_mut(4) {
                let v = i32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                let m = (v << 8) >> 8;
                let f = (m as f32) * 2f32.powi(v >> 24);
                e.copy_from_slice(&f.to_le_bytes());
            }
        }
        "COLOR" if stride == 4 => {
            for e in data.chunks_exact_mut(4) {
                let rgba = ycocg([e[0].into(), e[1] as i8 as i32, e[2] as i8 as i32, e[3].into()], 255.0);
                for (o, v) in e.iter_mut().zip(rgba) {
                    *o = v as u8;
                }
            }
        }
        "COLOR" if stride == 8 => {
            for e in data.chunks_exact_mut(8) {
                let c = |i: usize| u16::from_le_bytes([e[2 * i], e[2 * i + 1]]);
                let rgba = ycocg(
                    [c(0).into(), c(1) as i16 as i32, c(2) as i16 as i32, c(3).into()],
                    65535.0,
                );
                for (i, v) in rgba.into_iter().enumerate() {
                    e[2 * i..2 * i + 2].copy_from_slice(&(v as u16).to_le_bytes());
                }
            }
        }
        _ => return Err(bad(&format!("filter {filter} with stride {stride}"))),
    }
    Ok(())
}

/// The color filter (`KHR_meshopt_compression`): Y, Co, Cg and a marked alpha, `bits` wide, to
/// RGBA in units of the component's maximum (255 or 65535, by the element size).
fn ycocg([y, co, cg, alpha]: [i32; 4], max: f32) -> [i32; 4] {
    // The alpha's top set bit marks its width; `as` is all ones up to it.
    let mut scale = alpha;
    scale |= scale >> 1;
    scale |= scale >> 2;
    scale |= scale >> 4;
    scale |= scale >> 8;
    let a = ((alpha << 1) & scale) | (alpha & 1);
    let ss = if scale == 0 { 0.0 } else { max / scale as f32 };
    [y + co - cg, y + cg, y - co - cg, a].map(|v| (v as f32 * ss + 0.5) as i32)
}

/// The octahedral filter: components in units of `one` (the third input) to a unit vector in
/// units of `max`.
fn octahedral(c: [f32; 3], max: f32) -> [f32; 3] {
    let (mut x, mut y) = (c[0] / c[2], c[1] / c[2]);
    let z = 1.0 - x.abs() - y.abs();
    let t = z.min(0.0);
    x += if x >= 0.0 { t } else { -t };
    y += if y >= 0.0 { t } else { -t };
    let len = (x * x + y * y + z * z).sqrt();
    [x, y, z].map(|v| (v / len * max).round())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        (len > 0.0).then(|| v.map(|c| c / len))
    }
    use crate::test_rng::Pcg32;

    #[test]
    fn attributes_roundtrip() {
        let mut r = Pcg32::new(3);
        for (count, stride) in [(0, 4), (1, 4), (17, 8), (300, 12), (1000, 16), (40, 256), (700, 64)] {
            // Smooth data with occasional jumps exercises every group mode.
            let mut data = vec![0u8; count * stride];
            for i in 1..count {
                for k in 0..stride {
                    let prev = data[(i - 1) * stride + k];
                    let step = match r.below(10) {
                        0 => r.below(256) as u8,
                        1..=3 => r.below(9) as u8,
                        _ => 0,
                    };
                    data[i * stride + k] = prev.wrapping_add(step);
                }
            }
            let enc = encode_vertex_buffer(&data, stride);
            assert_eq!(
                decode_vertex_buffer(&enc, count, stride).unwrap(),
                data,
                "{count}x{stride}"
            );
        }
    }

    #[test]
    fn triangles_roundtrip() {
        // A grid in cache order, a shuffled one, two streams concatenated (reset) and
        // strip-like and far-apart explicit indices.
        let n = 30u32;
        let mut grid = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                grid.extend_from_slice(&[a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        let mut r = Pcg32::new(9);
        let mut shuffled: Vec<[u32; 3]> = grid.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        for i in (1..shuffled.len()).rev() {
            shuffled.swap(i, r.below(i as u32 + 1) as usize);
        }
        let shuffled = shuffled.concat();
        let concat = [&grid[..], &grid[..]].concat();
        let wild: Vec<u32> = (0..300).map(|_| r.next_u32() >> r.below(32)).collect();
        let strip: Vec<u32> = (0..100u32).flat_map(|i| [i + 2, i + 1, i]).collect();
        for idx in [&grid, &shuffled, &concat, &wild, &strip, &vec![]] {
            // Triangles keep their order and winding, but may start at another vertex.
            let canon = |v: &[u32]| -> Vec<[u32; 3]> {
                v.chunks_exact(3)
                    .map(|t| {
                        let r = (0..3).min_by_key(|&k| t[k]).unwrap();
                        [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
                    })
                    .collect()
            };
            let enc = encode_index_buffer(idx);
            assert_eq!(canon(&decode_index_buffer(&enc, idx.len()).unwrap()), canon(idx));
        }
        // Ordered input costs about a byte per triangle.
        assert!(encode_index_buffer(&grid).len() < grid.len() / 3 * 3 / 2);
    }

    /// The worked example in the extension's specification (4-bit sentinel group).
    #[test]
    fn spec_group_example() {
        let src = [0x17, 0x5f, 0xf0, 0xbc, 0x77, 0xa9, 0x21, 0x00, 0x34, 0xb5];
        let mut d = [0u8; 16];
        assert_eq!(decode_group(&src, 0, src.len(), 4, &mut d).unwrap(), src.len());
        let got: Vec<i8> = d.iter().map(|&x| unzigzag8(x) as i8).collect();
        assert_eq!(got, [-1, -4, -3, 26, -91, 0, -6, 6, -4, -4, 5, -5, 1, -1, 0, 0]);
    }

    #[test]
    fn index_sequence_decodes() {
        // 5, 3 (baseline 0), 100 (baseline 1), 4 (baseline 0).
        let mut src = vec![0xd1];
        for (delta, base) in [(5i32, 0u32), (-2, 0), (100, 1), (1, 0)] {
            let z = ((delta << 1) ^ (delta >> 31)) as u32;
            write_varint(&mut src, z << 1 | base);
        }
        src.extend_from_slice(&[0; 4]);
        assert_eq!(decode_index_sequence(&src, 4).unwrap(), vec![5, 3, 100, 4]);
    }

    /// meshoptimizer's `PV` test vertices (u16 x3, u8 x2, u16 x2) as bytes.
    fn pv_bytes() -> Vec<u8> {
        let pv: [([u16; 3], [u8; 2], [u16; 2]); 4] = [
            ([0, 0, 0], [0, 0], [0, 0]),
            ([300, 0, 0], [0, 0], [500, 0]),
            ([0, 300, 0], [0, 0], [0, 500]),
            ([300, 300, 0], [0, 0], [500, 500]),
        ];
        let mut out = Vec::new();
        for (p, n, t) in pv {
            p.iter().for_each(|v| out.extend(v.to_le_bytes()));
            out.extend(n);
            t.iter().for_each(|v| out.extend(v.to_le_bytes()));
        }
        out
    }

    /// meshoptimizer's own encodings (`demo/tests.cpp`): version 0, version 1, and a version 1
    /// stream from a custom encoder that uses every control and channel mode.
    #[test]
    fn reference_vertex_streams() {
        let v0: &[u8] = &[
            0xa0, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x58, 0x57, 0x58, 0x01, 0x26, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x00, 0x00,
            0x00, 0x58, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x17, 0x18,
            0x17, 0x01, 0x26, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x00, 0x00, 0x00, 0x17, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let v1: &[u8] = &[
            0xa1, 0xee, 0xaa, 0xee, 0x00, 0x4b, 0x4b, 0x4b, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x7d, 0x7d, 0x7d, 0x00, 0x00,
            0x7d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x62, 0x00, 0x62,
        ];
        let v1_custom: &[u8] = &[
            0xa1, 0xd4, 0x94, 0xd4, 0x01, 0x0e, 0x00, 0x58, 0x57, 0x58, 0x02, 0x02, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x03, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x02, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x7d, 0x7d, 0x7d, 0x01, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x7d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x62,
        ];
        let want = pv_bytes();
        for (name, src) in [("v0", v0), ("v1", v1), ("v1 custom", v1_custom)] {
            assert_eq!(decode_vertex_buffer(src, 4, 12).unwrap(), want, "{name}");
            // One byte more or less is a malformed stream.
            let mut long = src.to_vec();
            long.push(0);
            assert!(decode_vertex_buffer(&long, 4, 12).is_err(), "{name} long");
            assert!(
                decode_vertex_buffer(&src[..src.len() - 1], 4, 12).is_err(),
                "{name} short"
            );
        }
    }

    /// meshoptimizer's 16-bit delta stream (`decodeVertexV1Deltas`).
    #[test]
    fn reference_vertex_deltas() {
        let want: [u16; 64] = [
            248, 248, 240, 240, 249, 250, 243, 244, 250, 252, 246, 248, 251, 254, 249, 252, 252, 256, 252, 256, 253,
            258, 255, 260, 254, 260, 258, 264, 255, 262, 261, 268, 256, 264, 264, 272, 257, 262, 267, 268, 258, 260,
            270, 264, 259, 258, 273, 260, 260, 256, 276, 256, 261, 254, 279, 252, 262, 252, 282, 248, 263, 250, 285,
            244,
        ];
        let src: &[u8] = &[
            0xa1, 0x99, 0x99, 0x01, 0x2a, 0xaa, 0xaa, 0xaa, 0x02, 0x04, 0x44, 0x44, 0x44, 0x43, 0x33, 0x33, 0x33, 0x02,
            0x06, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x02, 0x08, 0x88, 0x88, 0x88, 0x87, 0x77, 0x77, 0x77, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x00, 0xf8, 0x00, 0xf0,
            0x00, 0xf0, 0x00, 0x01, 0x01,
        ];
        let got = decode_vertex_buffer(src, 16, 8).unwrap();
        let got: Vec<u16> = got.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(got, want);
    }

    /// meshoptimizer's color filter vectors (`encodeFilterColor8`/`12`), decoded.
    #[test]
    fn color_filter() {
        let want = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.25],
            [0.4, 0.4, 0.4, 0.75],
        ];
        let mut c8 = vec![
            0x40, 0x7f, 0xc1, 0xff, 0x7f, 0x00, 0x7f, 0xc0, 0x40, 0x81, 0xc0, 0xa0, 0x66, 0x00, 0x00, 0xdf,
        ];
        apply_filter("COLOR", &mut c8, 4).unwrap();
        for (i, &v) in c8.iter().enumerate() {
            assert!((f32::from(v) / 255.0 - want[i / 4][i % 4]).abs() < 1e-2, "{i}: {v}");
        }
        assert!(c8[12] == c8[13] && c8[13] == c8[14]);
        let c12: [u16; 16] = [
            0x0400, 0x07ff, 0xfc01, 0x0fff, 0x07ff, 0x0000, 0x07ff, 0x0c00, 0x0400, 0xf801, 0xfc00, 0x0a00, 0x0666,
            0x0000, 0x0000, 0x0dff,
        ];
        let mut b: Vec<u8> = c12.iter().flat_map(|v| v.to_le_bytes()).collect();
        apply_filter("COLOR", &mut b, 8).unwrap();
        for (i, c) in b.chunks_exact(2).enumerate() {
            let v = f32::from(u16::from_le_bytes([c[0], c[1]])) / 65535.0;
            assert!((v - want[i / 4][i % 4]).abs() < 1e-3, "{i}: {v}");
        }
    }

    #[test]
    fn index_sequence_roundtrip() {
        // meshoptimizer's reference encoding decodes, and ours round-trips.
        let reference = [
            0xd1, 0x00, 0x04, 0xcd, 0x01, 0x04, 0x07, 0x98, 0x1f, 0x00, 0x00, 0x00, 0x00,
        ];
        let seq = [0, 1, 51, 2, 49, 1000];
        assert_eq!(decode_index_sequence(&reference, 6).unwrap(), seq);
        let mut r = Pcg32::new(5);
        let wild: Vec<u32> = (0..500).map(|_| r.next_u32() >> 2).collect();
        let strip: Vec<u32> = (0..300).collect();
        for idx in [&seq[..], &wild, &strip, &[]] {
            let enc = encode_index_sequence(idx).unwrap();
            assert_eq!(decode_index_sequence(&enc, idx.len()).unwrap(), idx);
        }
        assert!(encode_index_sequence(&[0, 1 << 31]).is_err());
    }

    #[test]
    fn octahedral_roundtrip() {
        let mut r = Pcg32::new(1);
        for _ in 0..2000 {
            let v = [r.f32() - 0.5, r.f32() - 0.5, r.f32() - 0.5];
            let n = normalize(v).unwrap_or([0.0, 0.0, 1.0]);
            let mut e = encode_oct8(n, -1.0).to_vec();
            apply_filter("OCTAHEDRAL", &mut e, 4).unwrap();
            let d = normalize([e[0] as i8 as f32, e[1] as i8 as f32, e[2] as i8 as f32]).unwrap();
            let dot: f32 = (0..3).map(|i| d[i] * n[i]).sum();
            assert!(dot > 0.999, "{n:?} -> {d:?}");
            assert_eq!(e[3] as i8, -127);
        }
    }
}
