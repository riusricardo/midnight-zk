use std::{any::TypeId, fmt::Debug};

#[cfg(feature = "gpu")]
use std::sync::OnceLock;

use ff::Field;
use group::Group;
use halo2curves::{
    pairing::{Engine, MillerLoopResult, MultiMillerLoop},
    CurveAffine,
};
use midnight_curves::{Fq, G1Projective};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

#[cfg(feature = "gpu")]
use midnight_curves::G1Affine;
#[cfg(feature = "gpu")]
use midnight_bls12_381_cuda::GpuMsmContext;

#[cfg(feature = "gpu")]
/// Global GPU MSM context instance
/// 
/// This is initialized on first use (lazy) or can be initialized eagerly
/// via init_gpu_backend() to avoid first-call latency.
static GLOBAL_MSM_CONTEXT: OnceLock<GpuMsmContext> = OnceLock::new();

#[cfg(feature = "gpu")]
/// Get or initialize the global GPU MSM context
fn get_msm_context() -> &'static GpuMsmContext {
    GLOBAL_MSM_CONTEXT.get_or_init(|| {
        #[cfg(feature = "trace-msm")]
        eprintln!("[MSM] Initializing global GPU MSM context (lazy)");
        GpuMsmContext::new().expect("Failed to create GPU MSM context")
    })
}

/// Initialize GPU backend eagerly
/// 
/// Call this at application startup to avoid ~500-1000ms initialization
/// latency on the first proof request. If the backend is already initialized,
/// this is a no-op.
/// 
/// Returns the warmup duration if GPU is available and was initialized.
/// 
/// # Example
/// ```rust,no_run
/// use midnight_proofs::poly::kzg::msm::init_gpu_backend;
/// 
/// fn main() {
///     // Initialize GPU at startup
///     if let Some(duration) = init_gpu_backend() {
///         println!("GPU backend ready in {:?}", duration);
///     }
///     
///     // Start proof server...
/// }
/// ```
#[cfg(feature = "gpu")]
pub fn init_gpu_backend() -> Option<std::time::Duration> {
    use tracing::info;
    
    // Check if already initialized
    if GLOBAL_MSM_CONTEXT.get().is_some() {
        info!("GPU backend already initialized");
        return None;
    }
    
    info!("Initializing GPU backend eagerly...");
    let ctx = GpuMsmContext::new().expect("Failed to create GPU MSM context");
    
    // Warmup to trigger full CUDA initialization
    let warmup_result = ctx.warmup().ok();
    
    // Store in global
    let _ = GLOBAL_MSM_CONTEXT.set(ctx);
    
    if let Some(duration) = warmup_result {
        info!("GPU backend initialized and warmed up in {:?}", duration);
    }
    
    warmup_result
}

#[cfg(not(feature = "gpu"))]
/// Stub for initializing GPU backend when GPU feature is disabled
///
/// Returns `None` since GPU is not available without the `gpu` feature flag.
pub fn init_gpu_backend() -> Option<std::time::Duration> {
    None
}

use super::params::ParamsVerifierKZG;
use crate::{
    poly::{
        commitment::{Guard, PolynomialCommitmentScheme},
        kzg::KZGCommitmentScheme,
        Error,
    },
    utils::{
        arithmetic::{CurveExt, MSM},
        helpers::ProcessedSerdeObject,
    },
};

/// A multiscalar multiplication in the polynomial commitment scheme
#[derive(Clone, Default, Debug)]
pub struct MSMKZG<E: Engine> {
    pub(crate) scalars: Vec<E::Fr>,
    pub(crate) bases: Vec<E::G1>,
}

impl<E: Engine> MSMKZG<E> {
    /// Create an empty MSM instance
    pub fn init() -> Self {
        MSMKZG {
            scalars: vec![],
            bases: vec![],
        }
    }

    /// Create an MSM from various MSMs
    pub fn from_many(msms: Vec<Self>) -> Self {
        let len = msms.iter().map(|m| m.scalars.len()).sum();

        let mut scalars = Vec::with_capacity(len);
        let mut bases = Vec::with_capacity(len);

        for mut msm in msms {
            scalars.append(&mut msm.scalars);
            bases.append(&mut msm.bases);
        }

        Self { scalars, bases }
    }

    /// Create a new MSM from a given base (with scalar of 1).
    pub fn from_base(base: &E::G1) -> Self {
        MSMKZG {
            scalars: vec![E::Fr::ONE],
            bases: vec![*base],
        }
    }
}

impl<E: Engine + Debug> MSM<E::G1Affine> for MSMKZG<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn append_term(&mut self, scalar: E::Fr, point: E::G1) {
        self.scalars.push(scalar);
        self.bases.push(point);
    }

    fn add_msm(&mut self, other: &Self) {
        self.scalars.reserve(other.scalars().len());
        self.scalars.extend_from_slice(&other.scalars());

        self.bases.reserve(other.bases().len());
        self.bases.extend_from_slice(&other.bases());
    }

    fn scale(&mut self, factor: E::Fr) {
        self.scalars.par_iter_mut().for_each(|s| {
            *s *= &factor;
        })
    }

    fn check(&self) -> bool {
        bool::from(self.eval().is_identity())
    }

    fn eval(&self) -> E::G1 {
        if self.scalars == vec![E::Fr::ONE] {
            self.bases[0]
        } else {
            msm_specific::<E::G1Affine>(&self.scalars, &self.bases)
        }
    }

    fn bases(&self) -> Vec<E::G1> {
        self.bases.clone()
    }

    fn scalars(&self) -> Vec<E::Fr> {
        self.scalars.clone()
    }
}

/// Check if BLST optimization is available for the given curve type
/// 
/// This should always return true for midnight_curves::G1Affine (BLS12-381).
/// If this returns false, the code will panic in msm_specific().
#[inline]
pub fn is_blst_available<C: CurveAffine>() -> bool {
    TypeId::of::<C>() == TypeId::of::<midnight_curves::G1Affine>()
}

#[cfg(feature = "gpu")]
/// MSM using pre-uploaded GPU bases (zero-copy optimization)
/// 
/// Uses bases cached in GPU memory, eliminating per-call conversion and upload overhead.
/// This is the primary optimization that provides GPU acceleration for K≥14.
/// 
/// # Arguments
/// * `coeffs` - Scalars for MSM computation
/// * `device_bases` - Pre-uploaded GPU bases (from ParamsKZG::get_or_upload_gpu_bases)
/// 
/// # Panics
/// Panics if type is not midnight_curves::G1Affine
pub fn msm_with_cached_bases<C: CurveAffine>(
    coeffs: &[C::Scalar],
    device_bases: &midnight_bls12_381_cuda::PrecomputedBases,
) -> C::Curve {
    #[cfg(feature = "trace-msm")]
    let start = std::time::Instant::now();
    #[cfg(feature = "trace-msm")]
    eprintln!("[MSM-CACHED] Starting with {} points (using pre-uploaded GPU bases)", coeffs.len());
    
    // Verify we're using midnight_curves (BLS12-381)
    assert!(
        is_blst_available::<C>(),
        "MSM must use midnight_curves::G1Affine. Found: {}",
        std::any::type_name::<C>()
    );
    
    // Safe: we just verified the type
    let coeffs = unsafe { &*(coeffs as *const _ as *const [Fq]) };
    
    // Launch async MSM (enables better CUDA stream pipelining)
    #[cfg(feature = "trace-msm")]
    eprintln!("   [MSM-CACHED] Launching async MSM (non-blocking)");
    
    let ctx = get_msm_context();
    let handle = ctx.msm_with_device_bases_async(coeffs, device_bases)
        .expect("Async MSM launch failed");
    
    // CPU could do other work here while GPU uploads and computes
    
    let res = handle.wait()
        .expect("Async MSM wait failed");
    let result = unsafe { std::mem::transmute_copy(&res) };
    
    #[cfg(feature = "trace-msm")]
    {
        let elapsed = start.elapsed();
        eprintln!("✓  [MSM-CACHED] Completed in {:?} (async-internal pattern)", elapsed);
    }
    
    result
}

#[allow(unsafe_code)]
/// MSM using BLST multi_exp with optional GPU acceleration
/// 
/// # Architecture Notes
/// 
/// This function is the **fallback path** for MSMs without pre-cached GPU bases.
/// For SRS-based commitments (the hot path), use `msm_with_cached_bases()` instead,
/// which uses bases already uploaded to GPU memory.
/// 
/// **Call sites:**
/// - `MSMKZG::eval()` - Verification accumulator (small, BLST is fine)
/// - Fallback when `!should_use_gpu(size)` - Small MSMs use BLST
/// - Non-SRS MSMs - Rare, converts projective→affine each call
///
/// # GPU Acceleration
/// When compiled with `gpu` feature and size >= 16384 (K≥14):
/// - Uses ICICLE CUDA backend via GPU executor
/// - Converts projective bases to affine (overhead for non-cached bases)
/// - For cached SRS bases, prefer `msm_with_cached_bases()`
///
/// This function REQUIRES midnight_curves::G1Affine and will panic if used with other curves.
pub fn msm_specific<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C::Curve]) -> C::Curve {
    #[cfg(feature = "trace-msm")]
    let start = std::time::Instant::now();
    #[cfg(feature = "trace-msm")]
    eprintln!("[MSM] Starting multi-scalar multiplication with {} points", coeffs.len());
    
    // Check if we're using midnight_curves (BLS12-381 with BLST)
    // If not, fall back to generic implementation using parallelize
    if !is_blst_available::<C>() {
        #[cfg(feature = "trace-msm")]
        eprintln!("   [MSM] Using generic fallback MSM for {}", std::any::type_name::<C>());
        
        use group::Group;
        use crate::utils::arithmetic::parallelize;
        
        let num_threads = rayon::current_num_threads();
        let base_chunk_size = coeffs.len() / num_threads;
        let remainder = coeffs.len() % num_threads;
        
        let mut results = vec![C::Curve::identity(); num_threads];
        
        parallelize(&mut results, |results_chunk, chunk_idx| {
            // Calculate the range for this chunk
            let chunk_size = if chunk_idx < remainder {
                base_chunk_size + 1
            } else {
                base_chunk_size
            };
            
            let start = if chunk_idx < remainder {
                chunk_idx * (base_chunk_size + 1)
            } else {
                remainder * (base_chunk_size + 1) + (chunk_idx - remainder) * base_chunk_size
            };
            let end = start + chunk_size;
            
            if start >= coeffs.len() {
                return;
            }
            
            let mut acc = C::Curve::identity();
            for i in start..end.min(coeffs.len()) {
                acc = (acc + bases[i] * coeffs[i]).into();
            }
            results_chunk[0] = acc;
        });
        
        let result = results.iter().fold(C::Curve::identity(), |acc, r| acc + r);
        
        #[cfg(feature = "trace-msm")]
        {
            let elapsed = start.elapsed();
            eprintln!("✓  [MSM] Completed in {:?}", elapsed);
        }
        
        return result;
    }
    
    // Safe: we verified the type is midnight_curves::G1Affine
    let coeffs = unsafe { &*(coeffs as *const _ as *const [Fq]) };
    let bases = unsafe { &*(bases as *const _ as *const [G1Projective]) };
    
    #[cfg(feature = "gpu")]
    {
        use midnight_bls12_381_cuda::should_use_gpu;
        
        let size = bases.len();
        
        // Check if GPU should be used (respects MIDNIGHT_DEVICE and size threshold)
        // If false, fall through to BLST path
        if should_use_gpu(size) {
            // Get global MSM context (will initialize lazily if not done eagerly)
            let ctx = get_msm_context();
            
            #[cfg(feature = "trace-msm")]
            eprintln!("   [MSM] Using GPU backend ({} points)", size);
            
            // Convert projective bases to affine for GPU MSM
            let bases_affine: Vec<G1Affine> = bases.iter().map(|p| G1Affine::from(*p)).collect();
            
            match ctx.msm(coeffs, &bases_affine) {
                Ok(res) => {
                    let result = unsafe { std::mem::transmute_copy(&res) };
                    
                    #[cfg(feature = "trace-msm")]
                    {
                        let elapsed = start.elapsed();
                        eprintln!("✓  [MSM] Completed in {:?}", elapsed);
                    }
                    
                    return result;
                }
                Err(_e) => {
                    // GPU failed, fall back to BLST
                    #[cfg(feature = "trace-msm")]
                    eprintln!("   [MSM] GPU failed: {:?}, falling back to BLST", _e);
                }
            }
        }
        
        // Use BLST directly for small MSMs - no conversion overhead
        #[cfg(feature = "trace-msm")]
        eprintln!("   [MSM] Using BLST multi_exp (size: {} points)", size);
        
        let res = G1Projective::multi_exp(bases, coeffs);
        let result = unsafe { std::mem::transmute_copy(&res) };
        
        #[cfg(feature = "trace-msm")]
        {
            let elapsed = start.elapsed();
            eprintln!("✓  [MSM] Completed in {:?}", elapsed);
        }
        
        result
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        // CPU-only path using BLST multi_exp
        #[cfg(feature = "trace-msm")]
        eprintln!("   [MSM] Using BLST multi_exp (BLS12-381 optimized)");
        
        let res = G1Projective::multi_exp(bases, coeffs);
        let result = unsafe { std::mem::transmute_copy(&res) };
        
        #[cfg(feature = "trace-msm")]
        {
            let elapsed = start.elapsed();
            eprintln!("✓  [MSM] Completed in {:?}", elapsed);
        }
        
        result
    }
}

/// Two channel MSM accumulator
#[derive(Debug, Clone)]
pub struct DualMSM<E: Engine> {
    pub(crate) left: MSMKZG<E>,
    pub(crate) right: MSMKZG<E>,
}

/// A [DualMSM] split into left and right vectors of `(Scalar, Point)` tuples
pub type SplitDualMSM<'a, E> = (
    Vec<(&'a <E as Engine>::Fr, &'a <E as Engine>::G1)>,
    Vec<(&'a <E as Engine>::Fr, &'a <E as Engine>::G1)>,
);

impl<E: MultiMillerLoop + Debug> Default for DualMSM<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn default() -> Self {
        Self::init()
    }
}

impl<E: MultiMillerLoop> Guard<E::Fr, KZGCommitmentScheme<E>> for DualMSM<E>
where
    E::G1: Default + CurveExt<ScalarExt = E::Fr> + ProcessedSerdeObject,
    E::G1Affine: Default + CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn verify(
        self,
        params: &<KZGCommitmentScheme<E> as PolynomialCommitmentScheme<E::Fr>>::VerifierParameters,
    ) -> Result<(), Error> {
        self.check(params).then_some(()).ok_or(Error::OpeningError)
    }
}

impl<E: MultiMillerLoop + Debug> DualMSM<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    /// Create an empty two channel MSM accumulator instance
    pub fn init() -> Self {
        Self {
            left: MSMKZG::init(),
            right: MSMKZG::init(),
        }
    }

    /// Create a new two channel MSM accumulator instance
    pub fn new(left: MSMKZG<E>, right: MSMKZG<E>) -> Self {
        Self { left, right }
    }

    /// Split the [DualMSM] into `left` and `right`
    pub fn split(&self) -> SplitDualMSM<E> {
        let left = self.left.scalars.iter().zip(self.left.bases.iter()).collect();
        let right = self.right.scalars.iter().zip(self.right.bases.iter()).collect();
        (left, right)
    }

    /// Scale all scalars in the MSM by some scaling factor
    pub fn scale(&mut self, e: E::Fr) {
        self.left.scale(e);
        self.right.scale(e);
    }

    /// Add another multiexp into this one
    pub fn add_msm(&mut self, other: Self) {
        self.left.add_msm(&other.left);
        self.right.add_msm(&other.right);
    }

    /// Performs final pairing check with given verifier params and two channel
    /// linear combination
    pub fn check(self, params: &ParamsVerifierKZG<E>) -> bool {
        let left = if self.left.scalars.len() == 1 && self.left.scalars[0] == E::Fr::ONE {
            self.left.bases[0]
        } else {
            self.left.eval()
        };

        let right = self.right.eval();

        let (term_1, term_2) = (
            (&left.into(), &params.s_g2_prepared),
            (&right.into(), &params.n_g2_prepared),
        );
        let terms = &[term_1, term_2];

        bool::from(E::multi_miller_loop(&terms[..]).final_exponentiation().is_identity())
    }
}

// =============================================================================
// Async MSM API for Pipelined Commitments
// =============================================================================

#[cfg(feature = "gpu")]
/// Launch async MSM using pre-uploaded GPU bases
/// 
/// Returns a handle that can be waited on later, enabling CPU/GPU overlap
/// and pipelining of multiple MSM operations.
/// 
/// # Arguments
/// * `coeffs` - Scalars for MSM computation
/// * `device_bases` - Pre-uploaded GPU bases (from ParamsKZG::get_or_upload_gpu_bases)
/// 
/// # Returns
/// Handle that completes to C::Curve when waited
/// 
/// # Example
/// ```rust,ignore
/// // Launch multiple MSMs without waiting
/// let handles: Vec<_> = polynomials.iter()
///     .map(|poly| msm_with_cached_bases_async(&poly.values, &device_bases))
///     .collect::<Result<_, _>>()?;
/// 
/// // GPU computes all MSMs while CPU does other work
/// let metadata = prepare_proof_metadata();
/// 
/// // Wait for all commitments
/// let commitments: Vec<_> = handles.into_iter()
///     .map(|h| h.wait())
///     .collect::<Result<_, _>>()?;
/// ```
pub fn msm_with_cached_bases_async<C: CurveAffine>(
    coeffs: &[C::Scalar],
    device_bases: &midnight_bls12_381_cuda::PrecomputedBases,
) -> Result<midnight_bls12_381_cuda::msm::MsmHandle, crate::poly::Error> {
    #[cfg(feature = "trace-msm")]
    eprintln!("[MSM-ASYNC] Launching async MSM with {} points", coeffs.len());
    
    // Verify we're using midnight_curves (BLS12-381)
    assert!(
        is_blst_available::<C>(),
        "MSM must use midnight_curves::G1Affine. Found: {}",
        std::any::type_name::<C>()
    );
    
    // Safe: we just verified the type
    let coeffs = unsafe { &*(coeffs as *const _ as *const [Fq]) };
    
    let ctx = get_msm_context();
    ctx.msm_with_device_bases_async(coeffs, device_bases)
        .map_err(|_| crate::poly::Error::OpeningError)
}

#[cfg(feature = "gpu")]
/// Batch launch multiple async MSMs
/// 
/// More efficient than calling msm_with_cached_bases_async() in a loop because
/// it minimizes launch overhead and enables better GPU pipelining.
pub fn msm_batch_async<C: CurveAffine>(
    coeffs_batch: &[&[C::Scalar]],
    device_bases: &midnight_bls12_381_cuda::PrecomputedBases,
) -> Result<Vec<midnight_bls12_381_cuda::msm::MsmHandle>, crate::poly::Error> {
    coeffs_batch.iter()
        .map(|coeffs| msm_with_cached_bases_async::<C>(coeffs, device_bases))
        .collect()
}

// =============================================================================
// Batch MSM Operations
// =============================================================================

#[cfg(feature = "gpu")]
/// Compute multiple MSMs with shared bases in a single GPU kernel.
///
/// This is the **critical optimization** for PLONK provers: when computing
/// polynomial commitments, multiple MSMs share the same SRS bases but use
/// different scalar sets. This function batches them into a single kernel launch.
///
/// # Arguments
///
/// * `coeffs_batch` - Slice of coefficient slices, one per polynomial. **All must have same length.**
/// * `device_bases` - Pre-uploaded SRS bases on GPU (shared across all MSMs)
///
/// # Returns
///
/// Vector of commitments, one per polynomial in the batch.
///
/// # Panics
///
/// Panics if type is not midnight_curves::G1Affine
pub fn msm_batch_with_cached_bases<C: CurveAffine>(
    coeffs_batch: &[&[C::Scalar]],
    device_bases: &midnight_bls12_381_cuda::PrecomputedBases,
) -> Vec<C::Curve> {
    #[cfg(feature = "trace-msm")]
    let start = std::time::Instant::now();
    #[cfg(feature = "trace-msm")]
    eprintln!(
        "[BATCH-MSM] Starting batch of {} MSMs with {} points each",
        coeffs_batch.len(),
        coeffs_batch.first().map(|c| c.len()).unwrap_or(0)
    );

    if coeffs_batch.is_empty() {
        return Vec::new();
    }

    // Verify we're using midnight_curves (BLS12-381)
    assert!(
        is_blst_available::<C>(),
        "Batch MSM must use midnight_curves::G1Affine. Found: {}",
        std::any::type_name::<C>()
    );

    // Safe: we verified the type
    let coeffs_batch_fq: Vec<&[Fq]> = coeffs_batch
        .iter()
        .map(|coeffs| unsafe { &*(*coeffs as *const _ as *const [Fq]) })
        .collect();

    // Execute batch MSM
    let ctx = get_msm_context();
    let results = ctx
        .msm_batch_with_device_bases(&coeffs_batch_fq, device_bases)
        .expect("Batch MSM execution failed");

    #[cfg(feature = "trace-msm")]
    {
        let elapsed = start.elapsed();
        eprintln!(
            "✓  [BATCH-MSM] {} MSMs in {:?} ({:.2}ms per MSM, single kernel)",
            results.len(),
            elapsed,
            elapsed.as_secs_f64() * 1000.0 / results.len() as f64
        );
    }

    // Convert results to generic curve type
    results
        .into_iter()
        .map(|res| unsafe { std::mem::transmute_copy(&res) })
        .collect()
}

#[cfg(feature = "gpu")]
/// Async variant of batch MSM - launches computation without blocking.
///
/// Launch batch MSM asynchronously and do CPU work while GPU computes.
pub fn msm_batch_with_cached_bases_async<C: CurveAffine>(
    coeffs_batch: &[&[C::Scalar]],
    device_bases: &midnight_bls12_381_cuda::PrecomputedBases,
) -> Result<BatchMsmHandleWrapper<C>, crate::poly::Error> {
    if coeffs_batch.is_empty() {
        return Err(crate::poly::Error::OpeningError);
    }

    assert!(
        is_blst_available::<C>(),
        "Async batch MSM must use midnight_curves::G1Affine"
    );

    let coeffs_batch_fq: Vec<&[Fq]> = coeffs_batch
        .iter()
        .map(|coeffs| unsafe { &*(*coeffs as *const _ as *const [Fq]) })
        .collect();

    let ctx = get_msm_context();
    let handle = ctx
        .msm_batch_with_device_bases_async(&coeffs_batch_fq, device_bases)
        .map_err(|_| crate::poly::Error::OpeningError)?;

    Ok(BatchMsmHandleWrapper {
        handle,
        _phantom: std::marker::PhantomData,
    })
}

#[cfg(feature = "gpu")]
/// Wrapper for batch MSM async handle with type safety
#[derive(Debug)]
pub struct BatchMsmHandleWrapper<C: CurveAffine> {
    handle: midnight_bls12_381_cuda::BatchMsmHandle,
    _phantom: std::marker::PhantomData<C>,
}

#[cfg(feature = "gpu")]
impl<C: CurveAffine> BatchMsmHandleWrapper<C> {
    /// Get the number of MSMs in this batch
    pub fn batch_size(&self) -> usize {
        self.handle.batch_size()
    }

    /// Wait for batch computation to complete and retrieve all results.
    pub fn wait(self) -> Result<Vec<C::Curve>, crate::poly::Error> {
        let results = self
            .handle
            .wait()
            .map_err(|_| crate::poly::Error::OpeningError)?;

        Ok(results
            .into_iter()
            .map(|res| unsafe { std::mem::transmute_copy(&res) })
            .collect())
    }
}

// =============================================================================
// Pipelined Batch Operations
// =============================================================================

#[cfg(feature = "gpu")]
/// Pipeline multiple MSMs with CPU/GPU overlap.
///
/// This is more efficient than calling `msm_with_cached_bases()` in a loop
/// because it launches all GPU operations before waiting, enabling:
/// - Overlap of H2D transfers with kernel execution
/// - CPU work (proof metadata, next phase prep) during GPU compute
/// - Better GPU utilization through stream pipelining
///
/// # Performance
///
/// The benefit comes from overlapping data transfers and kernel execution.
///
/// # Arguments
///
/// * `coeffs_batch` - Slice of coefficient slices, one per MSM
/// * `device_bases` - Pre-uploaded GPU bases (shared across all MSMs)
///
/// # Returns
///
/// Vector of MSM results, same order as input batch
///
/// # Example
///
/// ```rust,ignore
/// use midnight_proofs::poly::kzg::msm::msm_batch_pipelined;
/// use midnight_curves::G1Affine;
///
/// // Commit to multiple polynomials efficiently
/// let poly_coeffs: Vec<&[Scalar]> = polynomials.iter()
///     .map(|p| &p.coeffs[..])
///     .collect();
///
/// let commitments = msm_batch_pipelined::<G1Affine>(&poly_coeffs, &device_bases);
/// ```
pub fn msm_batch_pipelined<C: CurveAffine>(
    coeffs_batch: &[&[C::Scalar]],
    device_bases: &midnight_bls12_381_cuda::PrecomputedBases,
) -> Vec<C::Curve> {
    if coeffs_batch.is_empty() {
        return Vec::new();
    }

    #[cfg(feature = "trace-msm")]
    let start = std::time::Instant::now();
    #[cfg(feature = "trace-msm")]
    eprintln!("[MSM-PIPELINE] Launching {} MSMs asynchronously", coeffs_batch.len());

    // Verify type
    assert!(
        is_blst_available::<C>(),
        "Pipelined MSM must use midnight_curves::G1Affine"
    );

    // Launch all MSMs without waiting
    let handles: Vec<_> = coeffs_batch
        .iter()
        .map(|coeffs| {
            let coeffs_fq = unsafe { &*(*coeffs as *const _ as *const [Fq]) };
            let ctx = get_msm_context();
            ctx.msm_with_device_bases_async(coeffs_fq, device_bases)
                .expect("Failed to launch async MSM")
        })
        .collect();

    #[cfg(feature = "trace-msm")]
    eprintln!("   [MSM-PIPELINE] All {} MSMs launched, GPU working...", handles.len());

    // CPU can do work here while GPU computes all MSMs
    // Example: prepare proof metadata, next phase setup, etc.

    // Wait for all results
    let results: Vec<C::Curve> = handles
        .into_iter()
        .map(|handle| {
            let res = handle.wait().expect("Failed to wait on async MSM");
            unsafe { std::mem::transmute_copy(&res) }
        })
        .collect();

    #[cfg(feature = "trace-msm")]
    {
        let elapsed = start.elapsed();
        eprintln!(
            "✓  [MSM-PIPELINE] {} commitments in {:?} ({:.2}ms each avg)",
            results.len(),
            elapsed,
            elapsed.as_secs_f64() * 1000.0 / results.len() as f64
        );
    }

    results
}
