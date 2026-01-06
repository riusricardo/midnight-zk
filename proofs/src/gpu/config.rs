//! GPU device configuration (icicle-halo2 pattern)
//!
//! Simple device selection following icicle-halo2's approach.
//! When GPU is not used, operations fall back to BLST (not ICICLE CPU).
//!
//! # Environment Variables
//!
//! - `ICICLE_BACKEND_INSTALL_DIR`: Path to ICICLE backend (default: `/opt/icicle/lib/backend`)
//! - `MIDNIGHT_GPU_MIN_K`: Minimum K for GPU usage (default: 16, meaning 2^16 = 65536 points)
//! - `MIDNIGHT_DEVICE`: Device selection ("auto", "gpu", or "cpu")
//!   - `auto` (default): Use GPU for large operations, BLST for small ones
//!   - `gpu`: Force GPU for all operations regardless of size
//!   - `cpu`: Force BLST for all operations (disable GPU)

use std::sync::OnceLock;
use tracing::{debug, info, warn};

/// Device selection mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// Automatically select: GPU for large operations, BLST for small ones
    Auto,
    /// Force GPU for all operations (ignore size threshold)
    Gpu,
    /// Force BLST for all operations (disable GPU)
    Cpu,
}

impl DeviceType {
    /// Parse device type from environment variable MIDNIGHT_DEVICE
    pub fn from_env() -> Self {
        std::env::var("MIDNIGHT_DEVICE")
            .ok()
            .and_then(|s| match s.to_lowercase().as_str() {
                "gpu" | "cuda" => Some(DeviceType::Gpu),
                "cpu" | "blst" => Some(DeviceType::Cpu),
                "auto" => Some(DeviceType::Auto),
                other => {
                    warn!("Unknown MIDNIGHT_DEVICE value '{}', using Auto", other);
                    None
                }
            })
            .unwrap_or(DeviceType::Auto)
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Auto => write!(f, "Auto"),
            DeviceType::Gpu => write!(f, "GPU"),
            DeviceType::Cpu => write!(f, "CPU (BLST)"),
        }
    }
}

// ============================================================================
// Simple configuration functions (icicle-halo2 pattern)
// ============================================================================

/// Get the configured device type.
///
/// Reads from `MIDNIGHT_DEVICE` environment variable.
/// Returns cached value after first call.
pub fn device_type() -> DeviceType {
    static DEVICE_TYPE: OnceLock<DeviceType> = OnceLock::new();
    *DEVICE_TYPE.get_or_init(|| {
        let dt = DeviceType::from_env();
        if dt != DeviceType::Auto {
            info!("Device type from MIDNIGHT_DEVICE: {}", dt);
        }
        dt
    })
}

/// Minimum problem size threshold for GPU usage.
///
/// Problems smaller than this will use BLST due to GPU transfer overhead.
/// Parsed from `MIDNIGHT_GPU_MIN_K` environment variable (as log2 value).
/// Default: 2^15 = 32768 points
pub fn min_gpu_size() -> usize {
    static MIN_SIZE: OnceLock<usize> = OnceLock::new();

    *MIN_SIZE.get_or_init(|| {
        std::env::var("MIDNIGHT_GPU_MIN_K")
            .ok()
            .and_then(|s| s.parse::<u8>().ok())
            .map(|k| {
                let size = 1usize << k;
                info!("MIDNIGHT_GPU_MIN_K={} -> min_gpu_size={}", k, size);
                size
            })
            .unwrap_or(32768) // Default: K >= 15
    })
}

/// Check if a problem size should use GPU.
///
/// Returns true if GPU should be used, false means use BLST.
///
/// - `DeviceType::Auto`: Use GPU if size >= threshold
/// - `DeviceType::Gpu`: Always use GPU
/// - `DeviceType::Cpu`: Always use BLST (never GPU)
///
/// # Arguments
/// * `size` - Number of elements (scalars, points, etc.)
#[inline]
pub fn should_use_gpu(size: usize) -> bool {
    match device_type() {
        DeviceType::Gpu => true,  // Force GPU regardless of size
        DeviceType::Cpu => false, // Force BLST (disable GPU)
        DeviceType::Auto => size >= min_gpu_size(),
    }
}

/// Check if GPU should be used for a batch of operations.
///
/// This considers the **total work** across all operations, not just individual size.
/// GPU excels at throughput, so batching many smaller MSMs can still be beneficial
/// even if each individual MSM is below the single-operation threshold.
///
/// # Decision Logic
///
/// For batch operations, GPU is beneficial when:
/// 1. Total work (batch_size × individual_size) exceeds threshold, OR
/// 2. Individual size is large enough (traditional threshold)
///
/// The batch threshold is lower because:
/// - GPU kernel launch overhead is amortized across batch
/// - Memory transfers can be pipelined
/// - GPU memory is already warm after first operation
///
/// # Arguments
/// * `individual_size` - Size of each individual operation (e.g., points per MSM)
/// * `batch_count` - Number of operations in the batch
///
/// # Returns
/// `true` if GPU should be used for this batch
#[inline]
pub fn should_use_gpu_batch(individual_size: usize, batch_count: usize) -> bool {
    // For batch operations, we use the same threshold as single operations.
    // GPU overhead for small MSMs is significant, so batching small MSMs
    // on GPU is actually slower than BLST on CPU.
    //
    // The key insight from benchmarking:
    // - 4096 points: CPU is faster (even batched)
    // - 8192 points: CPU is still faster 
    // - 16384 points: CPU is still faster
    // - 32768+ points: GPU wins (threshold K=15)
    //
    // So we just use the standard single-operation threshold (K>=15).
    // The batch_count parameter is kept for future optimization when
    // GPU memory pooling and stream pipelining reduce overhead.
    let _ = batch_count; // Unused for now
    should_use_gpu(individual_size)
}

/// Get the ICICLE backend installation path.
///
/// Reads from `ICICLE_BACKEND_INSTALL_DIR` environment variable.
/// Falls back to `/opt/icicle/lib/backend` if not set.
pub fn backend_path() -> String {
    std::env::var("ICICLE_BACKEND_INSTALL_DIR")
        .unwrap_or_else(|_| "/opt/icicle/lib/backend".to_string())
}

/// Get the device ID to use (for multi-GPU systems).
///
/// Currently always returns 0 (first GPU).
/// Future: Could be configurable via environment variable.
#[inline]
pub const fn device_id() -> i32 {
    0
}

/// Log current GPU configuration.
///
/// Useful for debugging and verifying configuration at startup.
pub fn log_config() {
    debug!("GPU Configuration:");
    debug!("  Backend path: {}", backend_path());
    debug!("  Device type: {}", device_type());
    debug!("  Device ID: {}", device_id());
    debug!(
        "  Min GPU size: {} (K >= {})",
        min_gpu_size(),
        min_gpu_size().trailing_zeros()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::Auto.to_string(), "Auto");
        assert_eq!(DeviceType::Gpu.to_string(), "GPU");
        assert_eq!(DeviceType::Cpu.to_string(), "CPU (BLST)");
    }

    #[test]
    fn test_min_gpu_size_default() {
        let size = min_gpu_size();
        assert!(size > 0);
        assert!(size.is_power_of_two());
        assert_eq!(size, 32768); // Default K=15
    }

    #[test]
    fn test_should_use_gpu_threshold() {
        let threshold = min_gpu_size();

        // In Auto mode (default), should respect threshold
        if device_type() == DeviceType::Auto {
            assert!(!should_use_gpu(threshold - 1));
            assert!(should_use_gpu(threshold));
            assert!(should_use_gpu(threshold * 2));
        }
    }

    #[test]
    fn test_should_use_gpu_batch() {
        let threshold = min_gpu_size(); // 32768
        
        // In Auto mode, batch uses same threshold as single operation
        if device_type() == DeviceType::Auto {
            // Below threshold - should not use GPU regardless of batch count
            assert!(!should_use_gpu_batch(threshold - 1, 1));
            assert!(!should_use_gpu_batch(threshold - 1, 10));
            assert!(!should_use_gpu_batch(4096, 100)); // 4096 < 32768
            
            // At or above threshold - should use GPU
            assert!(should_use_gpu_batch(threshold, 1));
            assert!(should_use_gpu_batch(threshold, 10));
            assert!(should_use_gpu_batch(threshold * 2, 5));
        }
    }

    #[test]
    fn test_backend_path() {
        let path = backend_path();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_device_id() {
        assert_eq!(device_id(), 0);
    }
}
