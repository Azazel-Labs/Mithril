//! Pure-Rust 3D asset interchange, with glTF as the hub.
//!
//! - [`formats`]: readers for STL, PLY, `MagicaVoxel`, OBJ, FBX, 3DS, 3MF, USD and glTF, each behind a
//!   cargo feature, producing a [`Document`](document::Document).
//! - [`document`]: that common form, a glTF 2.0 document (JSON and buffers) written out as a GLB
//!   as it is, flattened into a posed [`Mesh`](mesh::Mesh) on demand, and pruned to what a
//!   [`Keep`](document::Keep) asks for.
//! - [`mesh`]: the flat triangle-list model, written out as a GLB.
//! - [`assets`]: how readers find the textures and material libraries a model refers to.
//! - [`gltf`]: GLB packing, and meshopt compression added to or decoded away from a GLB.
//! - [`meshopt`]: the `EXT_meshopt_compression` and `KHR_meshopt_compression` bitstreams.

use std::fmt::Display;

/// Returns an [`Error::InvalidInput`] built from a format string.
macro_rules! bail {
    ($($t:tt)*) => {
        return Err($crate::Error::InvalidInput(format!($($t)*)))
    };
}

/// Bails unless the condition holds.
macro_rules! ensure {
    ($cond:expr, $($t:tt)*) => {
        if !$cond {
            bail!($($t)*)
        }
    };
}

/// An [`Error::InvalidInput`] built from a format string.
#[allow(unused_macros)]
macro_rules! invalid {
    ($($t:tt)*) => {
        $crate::Error::InvalidInput(format!($($t)*))
    };
}

pub mod assets;
pub mod document;
mod flatten;
pub mod formats;
pub mod gltf;
mod keep;
pub mod mesh;
pub mod meshopt;
mod num;

#[cfg(test)]
mod test_rng;

/// What can go wrong reading or writing an asset.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The input is malformed or uses something unsupported.
    #[error("{0}")]
    InvalidInput(String),
    /// Reading a file failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// A result with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Turns a missing value or a foreign error into an [`Error::InvalidInput`] that says what was
/// being done.
#[allow(dead_code)]
pub(crate) trait Context<T> {
    fn context(self, what: impl Display) -> Result<T>;
    fn with_context<D: Display>(self, what: impl FnOnce() -> D) -> Result<T>;
}

impl<T> Context<T> for Option<T> {
    fn context(self, what: impl Display) -> Result<T> {
        self.ok_or_else(|| Error::InvalidInput(what.to_string()))
    }
    fn with_context<D: Display>(self, what: impl FnOnce() -> D) -> Result<T> {
        self.ok_or_else(|| Error::InvalidInput(what().to_string()))
    }
}

impl<T, E: Display> Context<T> for std::result::Result<T, E> {
    fn context(self, what: impl Display) -> Result<T> {
        self.map_err(|e| Error::InvalidInput(format!("{what}: {e}")))
    }
    fn with_context<D: Display>(self, what: impl FnOnce() -> D) -> Result<T> {
        self.map_err(|e| Error::InvalidInput(format!("{}: {e}", what())))
    }
}

/// Foreign errors met while parsing, as invalid input.
macro_rules! invalid_from {
    ($($t:ty),* $(,)?) => {
        $(impl From<$t> for Error {
            fn from(e: $t) -> Self {
                Error::InvalidInput(e.to_string())
            }
        })*
    };
}

invalid_from!(
    std::num::ParseIntError,
    std::num::ParseFloatError,
    std::num::TryFromIntError,
    std::array::TryFromSliceError,
    std::str::Utf8Error,
    serde_json::Error,
);
#[cfg(any(feature = "3mf", feature = "usd"))]
invalid_from!(zip::result::ZipError);
#[cfg(feature = "3mf")]
invalid_from!(quick_xml::Error);
