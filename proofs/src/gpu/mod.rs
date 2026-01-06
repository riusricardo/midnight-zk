//! GPU acceleration for midnight-zk proof generation
//!
//! This module provides GPU-accelerated implementations of computationally intensive
//! operations in the PLONK prover, specifically Multi-Scalar Multiplication (MSM).
//!
//! # Features
//!
//! Enable GPU support by compiling with the `gpu` feature:
//! ```toml
//! midnight-proofs = { version = "*", features = ["gpu"] }
//! ```
//!
//! # Backend Support
//!
//! Currently supports:
//! - **CUDA**: NVIDIA GPUs (requires ICICLE CUDA backend)
//! - **CPU**: Automatic fallback when GPU unavailable
//!
//! # Setup
//!
//! 1. Build and install the CUDA backend:
//!    ```bash
//!    cd cuda-backend
//!    mkdir build && cd build
//!    cmake .. -DCMAKE_BUILD_TYPE=Release
//!    make icicle -j$(nproc)
//!    sudo make icicle-install
//!    ```
//!
//! 2. The backend will be automatically loaded when the prover is initialized.
//!
//! The CUDA backend is fully open source and can be found in the `cuda-backend` directory.
//!
//! # Example
//!
//! ```rust,no_run
//! use midnight_proofs::gpu::{GpuBackend, GpuConfig};
//!
//! // Initialize GPU backend with default config
//! let backend = GpuBackend::new(GpuConfig::default())?;
//!
//! // Backend will automatically select GPU or CPU based on availability
//! ```

pub mod backend;
pub mod config;
pub mod msm;
pub mod ntt;
pub mod stream;
pub mod types;

pub use backend::{GpuBackend, GpuError, is_gpu_available};
#[cfg(feature = "gpu")]
pub use backend::ensure_backend_loaded;
pub use config::{DeviceType, GpuConfig};
#[cfg(feature = "gpu")]
pub use msm::{GpuMsmContext, MsmError, MsmHandle, G2MsmHandle};
#[cfg(feature = "gpu")]
pub use ntt::{GpuNttContext, NttError, NttHandle};
#[cfg(feature = "gpu")]
pub use stream::ManagedStream;
pub use types::TypeConverter;

/// Check if GPU support is compiled in
pub const GPU_SUPPORT: bool = cfg!(feature = "gpu");

/// Initialize and warmup GPU backend for proof generation
/// 
/// Call this at application startup to avoid first-request latency.
/// This function:
/// 1. Initializes the ICICLE CUDA backend
/// 2. Creates the global MSM executor
/// 3. Runs a small MSM to trigger GPU memory allocation
/// 
/// Returns the warmup duration, or None if GPU is not available.
/// 
/// # Example
/// ```rust,no_run
/// use midnight_proofs::gpu::warmup_gpu;
/// 
/// #[tokio::main]
/// async fn main() {
///     // Warmup GPU before starting server
///     if let Some(duration) = warmup_gpu() {
///         println!("GPU ready in {:?}", duration);
///     }
///     
///     // Start server...
/// }
/// ```
pub fn warmup_gpu() -> Option<std::time::Duration> {
    use tracing::info;
    
    #[cfg(feature = "gpu")]
    {
        info!("Warming up GPU backend...");
        
        match GpuMsmContext::new() {
            Ok(ctx) => match ctx.warmup() {
                Ok(duration) => {
                    info!("GPU warmup successful: backend=CUDA, duration={:?}", duration);
                    Some(duration)
                }
                Err(e) => {
                    tracing::warn!("GPU warmup failed: {:?}", e);
                    None
                }
            },
            Err(e) => {
                tracing::warn!("GPU context creation failed: {:?}", e);
                None
            }
        }
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_support() {
        // This test just verifies the module compiles
        assert_eq!(GPU_SUPPORT, cfg!(feature = "gpu"));
    }
}
