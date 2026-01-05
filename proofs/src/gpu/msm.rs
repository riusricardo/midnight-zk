//! Multi-Scalar Multiplication (MSM) implementations
//!
//! This module provides GPU-accelerated MSM with automatic CPU fallback.

use crate::gpu::{DeviceType, GpuBackend, GpuConfig, GpuError, TypeConverter};
use midnight_curves::{Fq as Scalar, G1Affine, G1Projective, G2Affine, G2Projective};
use ff::Field;
use group::Group; // For identity() method
use group::prime::PrimeCurveAffine; // For generator() method
use tracing::{debug, info, warn};

/// Trait for MSM backend implementations
pub trait MsmBackend {
    /// Compute MSM: sum(scalars[i] * points[i])
    fn compute_msm(
        &self,
        scalars: &[Scalar],
        points: &[G1Affine],
    ) -> Result<G1Projective, MsmError>;
}

/// Errors that can occur during MSM operations
#[derive(Debug)]
pub enum MsmError {
    /// GPU operation failed
    GpuError(GpuError),
    /// CPU fallback also failed
    CpuError(String),
    /// Invalid input
    InvalidInput(String),
}

impl std::fmt::Display for MsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MsmError::GpuError(e) => write!(f, "GPU MSM error: {}", e),
            MsmError::CpuError(e) => write!(f, "CPU MSM error: {}", e),
            MsmError::InvalidInput(e) => write!(f, "Invalid MSM input: {}", e),
        }
    }
}

impl std::error::Error for MsmError {}

impl From<GpuError> for MsmError {
    fn from(e: GpuError) -> Self {
        MsmError::GpuError(e)
    }
}

/// GPU MSM backend using ICICLE
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuMsmBackend {
    pub(crate) backend: GpuBackend,
}

#[cfg(feature = "gpu")]
impl GpuMsmBackend {
    /// Create a new GPU MSM backend with the given configuration
    pub fn new(config: GpuConfig) -> Result<Self, GpuError> {
        let backend = GpuBackend::new(config)?;
        Ok(Self { backend })
    }
    
    fn compute_msm_internal(
        &self,
        scalars: &[Scalar],
        points: &[G1Affine],
    ) -> Result<G1Projective, MsmError> {
        use icicle_core::msm::{msm, MSMConfig};
        use icicle_core::ecntt::Projective; // For zero() method
        use icicle_runtime::memory::{DeviceVec, HostSlice};
        use icicle_bls12_381::curve::G1Projective as IcicleG1Projective;
        
        if scalars.len() != points.len() {
            return Err(MsmError::InvalidInput(format!(
                "Scalar and point count mismatch: {} vs {}",
                scalars.len(),
                points.len()
            )));
        }
        
        if scalars.is_empty() {
            return Ok(G1Projective::identity());
        }
        
        // OPTIMIZATION: Zero-copy scalar conversion (Phase 1)
        // Uses transmute - O(1) pointer cast instead of O(n) conversion
        // The raw bytes are in Montgomery form, so we set are_scalars_montgomery_form=true
        // The CUDA backend will convert from Montgomery to standard form on GPU
        let icicle_scalars = TypeConverter::scalar_slice_as_icicle(scalars);
        
        // TODO(Phase 1): Use zero-copy for points when layout is verified
        // For now, use parallel conversion for points
        let icicle_points = TypeConverter::g1_affine_slice_to_icicle_vec(points);
        
        // Allocate device result buffer
        let mut device_result = DeviceVec::<IcicleG1Projective>::device_malloc(1)
            .map_err(|e| MsmError::GpuError(GpuError::OperationFailed(format!("{:?}", e))))?;
        
        // Configure MSM - tell the backend that scalars are in Montgomery form
        // This triggers GPU-side conversion from Montgomery to standard form
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        
        msm(
            HostSlice::from_slice(icicle_scalars),
            HostSlice::from_slice(&icicle_points),
            &cfg,
            &mut device_result[..]
        ).map_err(|e| MsmError::GpuError(GpuError::OperationFailed(format!("{:?}", e))))?;
        
        // Copy result back to host
        let mut host_result = vec![IcicleG1Projective::zero(); 1];
        device_result.copy_to_host(HostSlice::from_mut_slice(&mut host_result))
            .map_err(|e| MsmError::GpuError(GpuError::OperationFailed(format!("{:?}", e))))?;
        
        // Convert back to midnight types
        Ok(TypeConverter::icicle_to_g1_projective(&host_result[0]))
    }
    
    /// Compute G2 MSM on GPU using ICICLE
    fn compute_g2_msm_internal(
        &self,
        scalars: &[Scalar],
        points: &[G2Affine],
    ) -> Result<G2Projective, MsmError> {
        use icicle_core::msm::{msm, MSMConfig};
        use icicle_core::ecntt::Projective; // For zero() method
        use icicle_runtime::memory::{DeviceVec, HostSlice};
        use icicle_bls12_381::curve::G2Projective as IcicleG2Projective;
        
        if scalars.len() != points.len() {
            return Err(MsmError::InvalidInput(format!(
                "Scalar and point count mismatch: {} vs {}",
                scalars.len(),
                points.len()
            )));
        }
        
        if scalars.is_empty() {
            return Ok(G2Projective::identity());
        }
        
        // OPTIMIZATION: Zero-copy scalar conversion (Phase 1)
        let icicle_scalars = TypeConverter::scalar_slice_as_icicle(scalars);
        
        // TODO(Phase 1): Use zero-copy for G2 points when layout is verified
        let icicle_points = TypeConverter::g2_affine_slice_to_icicle_vec(points);
        
        // Allocate device result buffer
        let mut device_result = DeviceVec::<IcicleG2Projective>::device_malloc(1)
            .map_err(|e| MsmError::GpuError(GpuError::OperationFailed(format!("{:?}", e))))?;
        
        // Configure and execute G2 MSM
        // Note: We set are_scalars_montgomery_form = true because zero-copy transmute
        // passes raw Montgomery bytes. The CUDA backend handles conversion on GPU.
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        msm(
            HostSlice::from_slice(icicle_scalars),
            HostSlice::from_slice(&icicle_points),
            &cfg,
            &mut device_result[..]
        ).map_err(|e| MsmError::GpuError(GpuError::OperationFailed(format!("G2 MSM failed: {:?}", e))))?;
        
        // Copy result back to host
        let mut host_result = vec![IcicleG2Projective::zero(); 1];
        device_result.copy_to_host(HostSlice::from_mut_slice(&mut host_result))
            .map_err(|e| MsmError::GpuError(GpuError::OperationFailed(format!("{:?}", e))))?;
        
        // Convert back to midnight types
        Ok(TypeConverter::icicle_to_g2_projective(&host_result[0]))
    }
    
    /// Public method to compute G2 MSM
    pub fn compute_g2_msm(
        &self,
        scalars: &[Scalar],
        points: &[G2Affine],
    ) -> Result<G2Projective, MsmError> {
        self.compute_g2_msm_internal(scalars, points)
    }
}

#[cfg(feature = "gpu")]
impl MsmBackend for GpuMsmBackend {
    fn compute_msm(
        &self,
        scalars: &[Scalar],
        points: &[G1Affine],
    ) -> Result<G1Projective, MsmError> {
        self.compute_msm_internal(scalars, points)
    }
}

/// CPU MSM backend (fallback)
#[derive(Debug)]
pub struct CpuMsmBackend;

impl MsmBackend for CpuMsmBackend {
    fn compute_msm(
        &self,
        scalars: &[Scalar],
        points: &[G1Affine],
    ) -> Result<G1Projective, MsmError> {
        if scalars.len() != points.len() {
            return Err(MsmError::InvalidInput(format!(
                "Scalar and point count mismatch: {} vs {}",
                scalars.len(),
                points.len()
            )));
        }
        
        if scalars.is_empty() {
            return Ok(G1Projective::identity());
        }
        
        // Convert affine to projective for multi_exp
        let points_proj: Vec<G1Projective> = points.iter().map(|p| G1Projective::from(*p)).collect();
        Ok(G1Projective::multi_exp(&points_proj, scalars))
    }
}

impl CpuMsmBackend {
    /// Compute G2 MSM on CPU
    pub fn compute_g2_msm(
        &self,
        scalars: &[Scalar],
        points: &[G2Affine],
    ) -> Result<G2Projective, MsmError> {
        if scalars.len() != points.len() {
            return Err(MsmError::InvalidInput(format!(
                "Scalar and point count mismatch: {} vs {}",
                scalars.len(),
                points.len()
            )));
        }
        
        if scalars.is_empty() {
            return Ok(G2Projective::identity());
        }
        
        // Convert affine to projective for multi_exp
        let points_proj: Vec<G2Projective> = points.iter().map(|p| G2Projective::from(*p)).collect();
        Ok(G2Projective::multi_exp(&points_proj, scalars))
    }
}

/// MSM executor that automatically selects GPU or CPU backend
#[derive(Debug)]
pub struct MsmExecutor {
    #[cfg(feature = "gpu")]
    gpu_backend: Option<GpuMsmBackend>,
    cpu_backend: CpuMsmBackend,
    config: GpuConfig,
}

impl MsmExecutor {
    /// Create a new MSM executor with the given configuration
    pub fn new(config: GpuConfig) -> Self {
        #[cfg(feature = "gpu")]
        let gpu_backend = match GpuMsmBackend::new(config.clone()) {
            Ok(backend) => {
                debug!("GPU MSM backend initialized");
                Some(backend)
            }
            Err(e) => {
                warn!("GPU MSM backend initialization failed: {}, using CPU fallback", e);
                None
            }
        };
        
        Self {
            #[cfg(feature = "gpu")]
            gpu_backend,
            cpu_backend: CpuMsmBackend,
            config,
        }
    }
    
    /// Execute MSM with automatic backend selection based on size
    /// 
    /// Decision is based on config.device_type and config.min_gpu_size.
    /// Override via MIDNIGHT_DEVICE and MIDNIGHT_GPU_MIN_K environment variables.
    pub fn execute(
        &self,
        scalars: &[Scalar],
        points: &[G1Affine],
    ) -> Result<G1Projective, MsmError> {
        let size = scalars.len();
        let k = (size as f64).log2().ceil() as u8;
        
        if self.should_use_gpu(size) {
            #[cfg(feature = "gpu")]
            if let Some(gpu) = &self.gpu_backend {
                debug!(
                    "Using GPU for MSM: {} points (K={}), device_type={:?}",
                    size, k, self.config.device_type
                );
                return gpu.compute_msm(scalars, points);
            }
            
            #[cfg(not(feature = "gpu"))]
            {
                return Err(MsmError::InvalidInput(
                    "GPU requested but feature not enabled".to_string()
                ));
            }
        }
        
        debug!(
            "Using CPU for MSM: {} points (K={}), device_type={:?}, min_size={}",
            size, k, self.config.device_type, self.config.min_gpu_size
        );
        self.cpu_backend.compute_msm(scalars, points)
    }
    
    /// Execute G2 MSM with automatic backend selection based on size
    /// 
    /// Same decision logic as execute() but for G2 points.
    /// G2 points are in the quadratic extension field Fq2.
    pub fn execute_g2(
        &self,
        scalars: &[Scalar],
        points: &[G2Affine],
    ) -> Result<G2Projective, MsmError> {
        let size = scalars.len();
        let k = (size as f64).log2().ceil() as u8;
        
        if self.should_use_gpu(size) {
            #[cfg(feature = "gpu")]
            if let Some(gpu) = &self.gpu_backend {
                debug!(
                    "Using GPU for G2 MSM: {} points (K={}), device_type={:?}",
                    size, k, self.config.device_type
                );
                return gpu.compute_g2_msm(scalars, points);
            }
            
            #[cfg(not(feature = "gpu"))]
            {
                return Err(MsmError::InvalidInput(
                    "GPU requested but feature not enabled".to_string()
                ));
            }
        }
        
        debug!(
            "Using CPU for G2 MSM: {} points (K={}), device_type={:?}, min_size={}",
            size, k, self.config.device_type, self.config.min_gpu_size
        );
        self.cpu_backend.compute_g2_msm(scalars, points)
    }
    
    /// Execute MSM using pre-uploaded GPU bases (zero-copy optimization)
    /// 
    /// Eliminates per-call conversion and upload overhead by using bases
    /// cached in GPU memory. Primary optimization for GPU-accelerated commitments.
    #[cfg(feature = "gpu")]
    pub fn execute_with_device_bases(
        &self,
        scalars: &[Scalar],
        device_bases: &icicle_runtime::memory::DeviceVec<icicle_bls12_381::curve::G1Affine>,
    ) -> Result<G1Projective, MsmError> {
        use icicle_core::msm::{msm, MSMConfig};
        use icicle_core::ecntt::Projective; // For zero() method
        use icicle_runtime::memory::{DeviceVec, HostSlice, HostOrDeviceSlice};
        use icicle_bls12_381::curve::G1Projective as IcicleG1Projective;
        
        if scalars.is_empty() {
            return Ok(G1Projective::identity());
        }
        
        // CRITICAL: Must set device context in multi-threaded environment
        // Cached GPU bases require active device context to access memory
        // Use backend's cached device to avoid creating new device (which triggers backend reload)
        use icicle_runtime::set_device;
        if let Some(ref gpu_backend) = self.gpu_backend {
            set_device(gpu_backend.backend.device())
                .map_err(|e| MsmError::GpuError(GpuError::DeviceSetFailed(format!("{:?}", e))))?;
        } else {
            return Err(MsmError::GpuError(GpuError::NotAvailable));
        }
        
        if scalars.len() > device_bases.len() {
            return Err(MsmError::InvalidInput(format!(
                "More scalars ({}) than available bases ({})",
                scalars.len(),
                device_bases.len()
            )));
        }
        
        #[cfg(feature = "trace-msm")]
        eprintln!("   [GPU] Using pre-uploaded bases (zero-copy MSM) - {} points", scalars.len());
        
        // OPTIMIZATION: Zero-copy scalar conversion (Phase 1)
        // Uses transmute - O(1) pointer cast instead of O(n) allocation + conversion
        let icicle_scalars = TypeConverter::scalar_slice_as_icicle(scalars);
        
        // Allocate device result buffer
        let mut device_result = DeviceVec::<IcicleG1Projective>::device_malloc(1)
            .map_err(|e| MsmError::GpuError(GpuError::OperationFailed(format!("Device malloc failed: {:?}", e))))?;
        
        // Execute MSM directly on device bases (no upload!)
        // Note: We set are_scalars_montgomery_form = true because zero-copy transmute
        // passes raw Montgomery bytes. The CUDA backend handles conversion on GPU.
        let mut cfg = MSMConfig::default();
        cfg.are_scalars_montgomery_form = true;
        
        #[cfg(feature = "trace-msm")]
        eprintln!("   [GPU] Calling ICICLE msm() with {} scalars, slice range 0..{}", icicle_scalars.len(), scalars.len());
        
        msm(
            HostSlice::from_slice(icicle_scalars),
            &device_bases[..scalars.len()],  // Use slice of pre-uploaded bases
            &cfg,
            &mut device_result[..]
        ).map_err(|e| MsmError::GpuError(GpuError::OperationFailed(format!("MSM operation failed: {:?}", e))))?;
        
        // Copy result back to host
        let mut host_result = vec![IcicleG1Projective::zero(); 1];
        device_result.copy_to_host(HostSlice::from_mut_slice(&mut host_result))
            .map_err(|e| MsmError::GpuError(GpuError::OperationFailed(format!("Copy to host failed: {:?}", e))))?;
        
        // Convert back to midnight types
        Ok(TypeConverter::icicle_to_g1_projective(&host_result[0]))
    }
    
    /// Determine if GPU should be used for the given problem size
    /// 
    /// Decision logic:
    /// 1. If MIDNIGHT_DEVICE=cpu, always use CPU
    /// 2. If MIDNIGHT_DEVICE=gpu, always use GPU (if available)
    /// 3. If MIDNIGHT_DEVICE=auto (default), use GPU for size >= min_gpu_size
    pub fn should_use_gpu(&self, size: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            // Check device type override first
            if self.config.device_type.is_cpu_forced() {
                return false;
            }
            
            if self.config.device_type.is_gpu_forced() {
                return self.gpu_backend.is_some();
            }
            
            // Auto mode: use GPU for large problems
            self.gpu_backend.is_some() && size >= self.config.min_gpu_size
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = size; // Suppress unused warning
            false
        }
    }
}

impl Default for MsmExecutor {
    fn default() -> Self {
        Self::new(GpuConfig::default())
    }
}

impl MsmExecutor {
    /// Warmup the MSM executor by running a small MSM operation
    /// 
    /// This should be called at application startup to:
    /// 1. Initialize the ICICLE CUDA backend
    /// 2. Allocate GPU memory
    /// 3. Run initial CUDA context setup
    /// 
    /// Without warmup, the first proof request pays ~500-1000ms initialization cost.
    pub fn warmup(&self) -> Result<std::time::Duration, MsmError> {
        use std::time::Instant;
        
        let start = Instant::now();
        
        #[cfg(feature = "gpu")]
        if let Some(gpu) = &self.gpu_backend {
            info!("GPU warmup: initializing ICICLE backend...");
            
            // Run a small MSM to trigger full GPU initialization
            // Size 1024 is enough to initialize without being slow
            let warmup_size = 1024;
            let scalars: Vec<Scalar> = (0..warmup_size)
                .map(|i| {
                    use ff::Field;
                    let mut s = Scalar::ONE;
                    for _ in 0..i % 10 {
                        s = s.double();
                    }
                    s
                })
                .collect();
            let points: Vec<G1Affine> = (0..warmup_size)
                .map(|_| G1Affine::generator())
                .collect();
            
            // Force GPU execution regardless of size threshold
            let _ = gpu.compute_msm(&scalars, &points)?;
            
            let elapsed = start.elapsed();
            info!("GPU warmup complete in {:?}", elapsed);
            return Ok(elapsed);
        }
        
        debug!("GPU warmup skipped (no GPU backend available)");
        Ok(start.elapsed())
    }
    
    /// Check if GPU backend is available
    pub fn has_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.gpu_backend.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
    
    /// Get the current device type configuration
    pub fn device_type(&self) -> DeviceType {
        self.config.device_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_msm_empty() {
        let backend = CpuMsmBackend;
        let result = backend.compute_msm(&[], &[]).unwrap();
        assert_eq!(result, G1Projective::identity());
    }

    #[test]
    fn test_cpu_msm_size_mismatch() {
        use ff::Field;
        let backend = CpuMsmBackend;
        let scalars = vec![Scalar::ONE];
        let points = vec![];
        let result = backend.compute_msm(&scalars, &points);
        assert!(matches!(result, Err(MsmError::InvalidInput(_))));
    }

    #[test]
    fn test_msm_executor_default() {
        let executor = MsmExecutor::default();
        // Just verify it constructs without panicking
        let _ = executor.should_use_gpu(1000);
    }
}
