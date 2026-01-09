//! # midnight_proofs

#![cfg_attr(docsrs, feature(doc_cfg))]
// The actual lints we want to disable.
#![allow(clippy::op_ref, clippy::many_single_char_names)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![cfg_attr(not(feature = "gpu"), deny(unsafe_code))]
#![cfg_attr(feature = "gpu", allow(unsafe_code))]

pub mod circuit;
pub mod plonk;
pub mod poly;
pub mod transcript;

pub mod dev;
pub mod utils;

/// GPU acceleration bridge module.
/// 
/// Provides a thin integration layer between midnight-proofs and the
/// GPU acceleration backend. Use this module for all GPU-related operations.
#[cfg(feature = "gpu")]
pub mod gpu_accel;

// Re-export GPU initialization function for easy access
#[cfg(feature = "gpu")]
#[doc(inline)]
pub use gpu_accel::init_gpu_backend;

// Also re-export from the old location for backwards compatibility
#[cfg(feature = "gpu")]
#[doc(inline)]
pub use poly::kzg::msm::init_gpu_backend as init_gpu_backend_legacy;

#[cfg(not(feature = "gpu"))]
/// Stub for non-GPU builds
pub fn init_gpu_backend() -> Option<std::time::Duration> {
    None
}
