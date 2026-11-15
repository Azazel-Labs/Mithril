//! Pure-Rust 3D asset interchange, with glTF as the hub.
//!
//! So far: the `EXT_meshopt_compression` bitstreams ([`meshopt`]).

pub mod meshopt;

#[cfg(test)]
mod test_rng;

/// What can go wrong reading or writing an asset.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The input is malformed or uses something unsupported.
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
