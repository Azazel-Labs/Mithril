//! Reading a file a block at a time, so a reader never holds the whole file, and parsing numbers
//! out of its text.

use std::fs::File;
use std::io::{self, Read};

/// How much of the file is in memory at once, unless a single line or token is longer.
const BLOCK: usize = 256 * 1024;

/// A file read through a buffer of one block. `fill` tops the buffer up and returns the bytes
/// not yet consumed; `consume` marks some of them done.
pub(crate) struct Blocks {
    file: File,
    buf: Vec<u8>,
    /// Consumed bytes at the front of `buf`.
    pos: usize,
    /// Filled bytes in `buf`.
    len: usize,
    eof: bool,
    /// The file's size when it was opened.
    size: u64,
}

impl Blocks {
    pub fn new(file: File) -> Self {
        let size = file.metadata().map_or(BLOCK as u64, |m| m.len());
        let cap = usize::try_from(size).unwrap_or(BLOCK).clamp(64, BLOCK);
        Self {
            file,
            buf: vec![0; cap],
            pos: 0,
            len: 0,
            eof: false,
            size,
        }
    }

    /// The unconsumed bytes, after reading as much as fits. Empty only at the end of the file.
    pub fn fill(&mut self) -> io::Result<&[u8]> {
        if self.pos > 0 {
            self.buf.copy_within(self.pos..self.len, 0);
            self.len -= self.pos;
            self.pos = 0;
        }
        while self.len < self.buf.len() && !self.eof {
            match self.file.read(&mut self.buf[self.len..]) {
                Ok(0) => self.eof = true,
                Ok(n) => self.len += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(&self.buf[..self.len])
    }

    /// The file's size, for estimating what it holds.
    pub fn size_hint(&self) -> usize {
        usize::try_from(self.size).unwrap_or(usize::MAX)
    }

    /// The unconsumed bytes, as they are.
    pub fn data(&self) -> &[u8] {
        &self.buf[self.pos..self.len]
    }

    pub fn consume(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.len);
    }

    /// Whether the whole file has been read into the buffer (the unconsumed bytes are all that
    /// is left).
    pub fn at_end(&self) -> bool {
        self.eof
    }

    /// Room for more than a block: for a line or record longer than the buffer.
    fn grow(&mut self) {
        let n = self.buf.len() * 2;
        self.buf.resize(n, 0);
    }

    /// At least `n` unconsumed bytes, or fewer at the end of the file.
    pub fn need(&mut self, n: usize) -> io::Result<&[u8]> {
        if self.len - self.pos < n && !self.eof {
            while self.buf.len() < n {
                self.grow();
            }
            self.fill()?;
        }
        Ok(&self.buf[self.pos..self.len])
    }

    /// The next stretch of text, cut where `cut` says (the length of a prefix that ends on a
    /// line or token boundary), and consumed. The rest of the file once it has all been read;
    /// `None` after that.
    pub fn text(&mut self, cut: impl Fn(&[u8]) -> Option<usize>) -> io::Result<Option<&[u8]>> {
        loop {
            self.fill()?;
            let len = self.len;
            if len == 0 {
                return Ok(None);
            }
            let n = if self.eof {
                len
            } else if let Some(n) = cut(&self.buf[..len]).filter(|&n| n > 0) {
                n
            } else {
                // One line fills the buffer: make room for more of it.
                self.grow();
                continue;
            };
            self.pos = n;
            return Ok(Some(&self.buf[..n]));
        }
    }
}

/// Text up to and including the last line break.
#[cfg(test)]
pub(crate) fn lines(data: &[u8]) -> Option<usize> {
    data.iter().rposition(|&b| b == b'\n').map(|i| i + 1)
}

/// Text up to and including the last byte that separates tokens.
pub(crate) fn words(data: &[u8]) -> Option<usize> {
    data.iter().rposition(|&b| space(b)).map(|i| i + 1)
}

/// Whether `b` separates tokens: ASCII whitespace, or any other control character.
#[inline]
pub(crate) fn space(b: u8) -> bool {
    b <= b' '
}

/// A plain decimal number, `[+-]digits[.digits][(e|E)[+-]digits]`, as its sign, its digits as
/// an integer and the power of ten that scales them. `None` for anything else, or more digits
/// than a `u64` holds.
#[inline]
fn decimal(s: &[u8]) -> Option<(bool, u64, i32)> {
    let mut i = 0;
    let neg = match s.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    let mut m: u64 = 0;
    let mut digits = 0;
    let mut exp: i32 = 0;
    while let Some(&c) = s.get(i) {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            break;
        }
        m = m.checked_mul(10)?.checked_add(u64::from(d))?;
        digits += 1;
        i += 1;
    }
    if s.get(i) == Some(&b'.') {
        i += 1;
        while let Some(&c) = s.get(i) {
            let d = c.wrapping_sub(b'0');
            if d > 9 {
                break;
            }
            m = m.checked_mul(10)?.checked_add(u64::from(d))?;
            digits += 1;
            exp -= 1;
            i += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if let Some(b'e' | b'E') = s.get(i) {
        i += 1;
        let eneg = match s.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        let start = i;
        let mut e: i32 = 0;
        while let Some(&c) = s.get(i) {
            let d = c.wrapping_sub(b'0');
            if d > 9 || i - start >= 6 {
                break;
            }
            e = e * 10 + i32::from(d);
            i += 1;
        }
        if i == start {
            return None;
        }
        exp += if eneg { -e } else { e };
    }
    (i == s.len()).then_some((neg, m, exp))
}

const POW10_F32: [f32; 11] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10];
const POW10_F64: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// `s` as an `f32`, rounded exactly as `str::parse` rounds it. Short decimals (up to seven
/// digits or so, what exporters write) take a fast path; everything else goes to the standard
/// parser.
#[inline]
pub(crate) fn f32_from(s: &[u8]) -> Option<f32> {
    if let Some((neg, m, e)) = decimal(s) {
        // Both the digits and the power of ten are exact in an f32, so one multiply or divide
        // rounds correctly (Clinger's fast path).
        if m <= 1 << 24 && (-10..=10).contains(&e) {
            let v = m as f32;
            let p = POW10_F32[e.unsigned_abs() as usize];
            let v = if e < 0 { v / p } else { v * p };
            return Some(if neg { -v } else { v });
        }
    }
    std::str::from_utf8(s).ok()?.parse().ok()
}

/// `s` as an `f64`, like [`f32_from`].
#[inline]
pub(crate) fn f64_from(s: &[u8]) -> Option<f64> {
    if let Some((neg, m, e)) = decimal(s)
        && m <= 1 << 53
        && (-22..=22).contains(&e)
    {
        let v = m as f64;
        let p = POW10_F64[e.unsigned_abs() as usize];
        let v = if e < 0 { v / p } else { v * p };
        return Some(if neg { -v } else { v });
    }
    std::str::from_utf8(s).ok()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_round_like_the_standard_parser() {
        let cases = [
            "0",
            "-0",
            "+1",
            "1.",
            ".5",
            "-.5",
            "1e5",
            "1E-5",
            "1.5e+3",
            "-1.234567e-01",
            "0.1",
            "0.2",
            "0.3",
            "3.4028235e38",
            "1e-45",
            "1e39",
            "16777217",
            "123.456789",
            "0.000001",
            "9.999999",
            "1.17549435e-38",
            "inf",
            "-infinity",
            "NaN",
            "1e",
            "e5",
            "--1",
            "1.2.3",
            "",
            ".",
            "+",
            "0x10",
            "1_0",
            "99999999999999999999999",
            "7e-11",
            "1234567.0",
            "0.12345678",
            "-12.345678",
        ];
        for s in cases {
            let want32 = s.parse::<f32>().ok();
            let got32 = f32_from(s.as_bytes());
            assert_eq!(
                got32.map(f32::to_bits),
                want32.map(f32::to_bits),
                "f32 {s:?}: {got32:?} vs {want32:?}"
            );
            let want64 = s.parse::<f64>().ok();
            let got64 = f64_from(s.as_bytes());
            assert_eq!(
                got64.map(f64::to_bits),
                want64.map(f64::to_bits),
                "f64 {s:?}"
            );
        }
        // Every short decimal the fast path takes, against the standard parser.
        let mut x: u32 = 1;
        for _ in 0..200_000 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let m = x % 16_777_216;
            let e = (x >> 24) % 21;
            let s = format!("{m}e-{e}");
            assert_eq!(
                f32_from(s.as_bytes()).map(f32::to_bits),
                s.parse::<f32>().ok().map(f32::to_bits),
                "{s}"
            );
            let s = format!("-{m}.{e}e{e}");
            assert_eq!(
                f32_from(s.as_bytes()).map(f32::to_bits),
                s.parse::<f32>().ok().map(f32::to_bits),
                "{s}"
            );
        }
    }

    #[test]
    fn blocks_cut_on_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.txt");
        let line = "abc def\n".repeat(100_000);
        std::fs::write(&p, &line).unwrap();
        let mut b = Blocks::new(File::open(&p).unwrap());
        let mut all = Vec::new();
        while let Some(t) = b.text(lines).unwrap() {
            assert!(t.ends_with(b"\n"));
            all.extend_from_slice(t);
        }
        assert_eq!(all, line.as_bytes());
        // One line longer than a block.
        let long = format!("{}\nx", "y".repeat(BLOCK * 3));
        std::fs::write(&p, &long).unwrap();
        let mut b = Blocks::new(File::open(&p).unwrap());
        let mut n = 0;
        while let Some(t) = b.text(lines).unwrap() {
            assert!(t.ends_with(b"\n") || t.ends_with(b"x"));
            n += t.len();
        }
        assert_eq!(n, long.len());
    }
}
