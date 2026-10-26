use std::{any::TypeId, fmt::Debug};

#[cfg(feature = "gpu")]
use std::sync::OnceLock;

use ff::Field;
use group::{Curve, Group};
use itertools::izip;
use midnight_curves::{
    msm::msm_best,
    pairing::{Engine, MillerLoopResult, MultiMillerLoop},
    CurveAffine, Fq, G1Projective,
};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

#[cfg(feature = "gpu")]
use midnight_curves::G1Affine;

// GPU types via bridge module (single import point)
#[cfg(feature = "gpu")]
use crate::gpu_accel::{
    GpuMsmContext, PrecomputedBases, MsmHandle, should_use_gpu,
    is_g1_affine, try_as_fq_slice, projective_to_curve,
};

#[cfg(feature = "gpu")]
/// Global GPU MSM context instance
/// 
/// This is initialized on first use (lazy) or can be initialized eagerly
/// via init_gpu_backend() to avoid first-call latency.
static GLOBAL_MSM_CONTEXT: OnceLock<GpuMsmContext> = OnceLock::new();

#[cfg(feature = "gpu")]
/// Try to get or initialize the global GPU MSM context, returning `None` (for CPU fallback)
/// instead of panicking when the GPU/backend is unavailable (correctness fix C5).
pub fn try_get_msm_context() -> Option<&'static GpuMsmContext> {
    if let Some(ctx) = GLOBAL_MSM_CONTEXT.get() {
        return Some(ctx);
    }
    match GpuMsmContext::new() {
        Ok(ctx) => {
            // If another thread won the init race, `set` fails and we return the winner.
            let _ = GLOBAL_MSM_CONTEXT.set(ctx);
            GLOBAL_MSM_CONTEXT.get()
        }
        Err(_e) => {
            #[cfg(feature = "trace-msm")]
            eprintln!("[MSM] GPU MSM context unavailable: {:?}", _e);
            None
        }
    }
}

#[cfg(feature = "gpu")]
/// Get or initialize the global GPU MSM context.
///
/// Panics if the GPU is unavailable; hot paths that can fall back to CPU should prefer
/// [`try_get_msm_context`].
pub fn get_msm_context() -> &'static GpuMsmContext {
    try_get_msm_context().expect("Failed to create GPU MSM context")
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
        CommitmentLabel, Error,
    },
    utils::{
        arithmetic::{CurveExt, MSM},
        helpers::ProcessedSerdeObject,
    },
};

/// A multi-scalar multiplication in the polynomial commitment scheme.
/// For every i, term (bases_i, scalars_i) may be have an optional
/// label_i for debugging or other purposes.
#[derive(Clone, Default, Debug)]
pub struct MSMKZG<E: Engine> {
    pub(crate) scalars: Vec<E::Fr>,
    pub(crate) bases: Vec<E::G1>,
    pub(crate) labels: Vec<CommitmentLabel>,
}

impl<E: Engine> MSMKZG<E> {
    /// Create an empty MSM instance
    pub fn init() -> Self {
        MSMKZG {
            scalars: vec![],
            bases: vec![],
            labels: vec![],
        }
    }

    /// Create an MSM from various MSMs
    pub fn from_many(msms: Vec<Self>) -> Self {
        let len = msms.iter().map(|m| m.scalars.len()).sum();

        let mut scalars = Vec::with_capacity(len);
        let mut bases = Vec::with_capacity(len);
        let mut labels = Vec::with_capacity(len);

        for mut msm in msms {
            scalars.append(&mut msm.scalars);
            bases.append(&mut msm.bases);
            labels.append(&mut msm.labels);
        }

        Self {
            scalars,
            bases,
            labels,
        }
    }

    /// Create a new MSM from a given base (with scalar of 1).
    pub fn from_base(base: &E::G1) -> Self {
        MSMKZG {
            scalars: vec![E::Fr::ONE],
            bases: vec![*base],
            labels: vec![CommitmentLabel::NoLabel],
        }
    }
}

impl<E: Engine + Debug> MSMKZG<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    /// Evaluates the MSM to a single point and replaces all terms with that
    /// single point (scalar = 1, label = `NoLabel`).
    ///
    /// This mirrors `AssignedMsm::collapse` in the circuits crate.
    ///
    /// # Panics (in debug mode)
    ///
    /// If any term carries a label other than `NoLabel` or `Advice`.
    //
    // We only allow `NoLabel` or `Advice` because these types of labels are
    // not relevant for the `verifier_gadget` in `midnight-circuits` (at least for
    // now). Other types of labels may carry information that we do not want to lose
    // when "collapsing".
    pub fn collapse(&mut self) {
        debug_assert!(
            self.labels
                .iter()
                .all(|l| matches!(l, CommitmentLabel::NoLabel | CommitmentLabel::Advice(_))),
            "collapse: all labels must be NoLabel or Advice, found: {:?}",
            self.labels,
        );
        let point = self.eval();
        self.scalars = vec![E::Fr::ONE];
        self.bases = vec![point];
        self.labels = vec![CommitmentLabel::NoLabel];
    }
}

impl<E: Engine + Debug> MSM<E::G1Affine> for MSMKZG<E>
where
    E::G1Affine: CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
{
    fn append_term(&mut self, scalar: E::Fr, point: E::G1, label: CommitmentLabel) {
        self.scalars.push(scalar);
        self.bases.push(point);
        self.labels.push(label);
    }

    fn add_msm(&mut self, other: &Self) {
        self.scalars.reserve(other.scalars().len());
        self.scalars.extend_from_slice(&other.scalars());

        self.bases.reserve(other.bases().len());
        self.bases.extend_from_slice(&other.bases());

        self.labels.reserve(other.labels().len());
        self.labels.extend_from_slice(&other.labels());
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

    fn labels(&self) -> Vec<CommitmentLabel> {
        self.labels.clone()
    }
}

/// Check if BLST optimization is available for the given curve type
/// 
/// This should always return true for midnight_curves::G1Affine (BLS12-381).
/// If this returns false, msm_specific() uses the generic `msm_best` path.
#[inline]
pub fn is_blst_available<C: CurveAffine + 'static>() -> bool {
    // Use the dispatch helper for cleaner type checking
    #[cfg(feature = "gpu")]
    { is_g1_affine::<C>() }
    #[cfg(not(feature = "gpu"))]
    {
        use std::any::TypeId;
        TypeId::of::<C>() == TypeId::of::<midnight_curves::G1Affine>()
    }
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
pub fn msm_with_cached_bases<C: CurveAffine + 'static>(
    coeffs: &[C::Scalar],
    device_bases: &PrecomputedBases,
) -> C::Curve
where
    C::Scalar: ff::PrimeField + 'static,
{
    #[cfg(feature = "trace-msm")]
    let start = std::time::Instant::now();
    #[cfg(feature = "trace-msm")]
    eprintln!("[MSM-CACHED] Starting with {} points (using pre-uploaded GPU bases)", coeffs.len());

    let result = try_msm_with_cached_bases::<C>(coeffs, device_bases)
        .expect("GPU MSM with cached bases failed");

    #[cfg(feature = "trace-msm")]
    {
        let elapsed = start.elapsed();
        eprintln!("✓  [MSM-CACHED] Completed in {:?} (async-internal pattern)", elapsed);
    }

    result
}

#[cfg(feature = "gpu")]
/// Fallible MSM using pre-uploaded GPU bases.
///
/// Returns an error (for CPU fallback) instead of panicking when the GPU/backend is
/// unavailable or the MSM fails (correctness fix C5). This is the primitive used by the
/// commit hot path; [`msm_with_cached_bases`] is the panicking convenience wrapper.
pub fn try_msm_with_cached_bases<C: CurveAffine + 'static>(
    coeffs: &[C::Scalar],
    device_bases: &PrecomputedBases,
) -> Result<C::Curve, crate::poly::Error>
where
    C::Scalar: ff::PrimeField + 'static,
{
    // Verify we're using midnight_curves (BLS12-381) and convert coefficients
    let fq_coeffs = try_as_fq_slice(coeffs).ok_or(crate::poly::Error::OpeningError)?;

    let ctx = try_get_msm_context().ok_or(crate::poly::Error::OpeningError)?;
    let handle = ctx
        .msm_with_device_bases_async(fq_coeffs, device_bases)
        .map_err(|_| crate::poly::Error::OpeningError)?;
    let res = handle.wait().map_err(|_| crate::poly::Error::OpeningError)?;
    Ok(projective_to_curve::<C>(res))
}

#[allow(unsafe_code)]
/// Wrapper over the MSM function to use the blstrs underlying function, with
/// optional GPU acceleration.
///
/// # GPU acceleration (`gpu` feature)
///
/// This is the **non-cached** MSM path. SRS commitments (`commit()` /
/// `commit_lagrange()`) use bases pre-uploaded to GPU memory
/// (`try_msm_with_cached_bases()`) and only fall back to this function.
///
/// **Call sites:**
/// - `MSMKZG::eval()` - Verification accumulator (small, BLST is fine)
/// - Fallback when `!should_use_gpu(size)` - Small MSMs use BLST
/// - CPU fallback of the cached-bases commit paths
///
/// For BLS12-381 MSMs at or above the GPU threshold (`should_use_gpu`), the
/// projective bases are converted to affine and the MSM runs on the GPU. Any GPU
/// failure (or an unavailable GPU context) is reported through
/// `note_gpu_fallback` (fatal under `MIDNIGHT_GPU_STRICT`) and the MSM is then
/// computed on the CPU below.
pub fn msm_specific<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C::Curve]) -> C::Curve {
    #[cfg(feature = "trace-msm")]
    let start = std::time::Instant::now();
    #[cfg(feature = "trace-msm")]
    eprintln!("[MSM] Starting multi-scalar multiplication with {} points", coeffs.len());

    // We remove zeros (keep only non-zero coefficients)
    let (coeffs, bases): (Vec<C::Scalar>, Vec<C::Curve>) = coeffs
        .iter()
        .zip(bases)
        .filter(|(s, _)| !s.is_zero_vartime())
        .map(|(s, b)| (*s, *b))
        .unzip();

    if coeffs.is_empty() {
        return C::Curve::identity();
    }

    #[cfg(feature = "gpu")]
    {
        let size = coeffs.len();

        // Check if GPU should be used (respects MIDNIGHT_DEVICE and size threshold)
        // If false, fall through to the CPU path below.
        // Correctness fix C5: probe the GPU context WITHOUT panicking. On GPU
        // unavailability (`try_get_msm_context` -> None) or any GPU error, fall through to
        // the CPU path below. This makes `msm_specific` a safe fallback target for
        // the commit hot path even on a `gpu`-built binary running with no usable GPU.
        if is_blst_available::<C>() && should_use_gpu(size) {
            if let Some(ctx) = try_get_msm_context() {
                #[cfg(feature = "trace-msm")]
                eprintln!("   [MSM] Using GPU backend ({} points)", size);

                // Safe: we verified the type is midnight_curves::G1Affine
                let fq_coeffs = unsafe { &*(coeffs.as_slice() as *const _ as *const [Fq]) };
                let proj_bases =
                    unsafe { &*(bases.as_slice() as *const _ as *const [G1Projective]) };

                // Convert projective bases to affine for GPU MSM
                let bases_affine: Vec<G1Affine> =
                    proj_bases.iter().map(|p| G1Affine::from(*p)).collect();

                match ctx.msm(fq_coeffs, &bases_affine) {
                    Ok(res) => {
                        #[cfg(feature = "trace-msm")]
                        eprintln!("✓  [MSM] Completed in {:?}", start.elapsed());

                        return unsafe { std::mem::transmute_copy(&res) };
                    }
                    Err(e) => {
                        // GPU MSM was attempted and FAILED: make it observable +
                        // strict-fatal (was silent unless the trace-msm feature was
                        // on), then fall through to the CPU path below.
                        midnight_bls12_381_cuda::config::note_gpu_fallback(
                            "MSM (msm_specific)", &format!("{:?}", e));
                    }
                }
            } else {
                // GPU was expected (should_use_gpu true) but the MSM context is
                // unavailable — e.g. the backend .so failed to load or a wrong
                // ICICLE_BACKEND_INSTALL_DIR. This was a fully silent BLST
                // fallback (the backend-dir trap); make it observable +
                // strict-fatal before falling through to the CPU path.
                midnight_bls12_381_cuda::config::note_gpu_fallback(
                    "MSM (msm_specific): GPU context unavailable",
                    "try_get_msm_context() returned None");
            }
        }
    }

    // We empirically checked that for MSMs larger than 2**18, the blstrs
    // implementation regresses.
    let result = if coeffs.len() <= (2 << 18)
        && TypeId::of::<C>() == TypeId::of::<midnight_curves::G1Affine>()
    {
        #[cfg(feature = "trace-msm")]
        eprintln!("   [MSM] Using BLST multi_exp (size: {} points)", coeffs.len());

        // Safe: we just checked type
        let coeffs_slice = coeffs.as_slice();
        let bases_slice = bases.as_slice();
        let coeffs = unsafe { &*(coeffs_slice as *const _ as *const [Fq]) };
        let bases = unsafe { &*(bases_slice as *const _ as *const [G1Projective]) };
        let res = G1Projective::multi_exp(bases, coeffs);
        unsafe { std::mem::transmute_copy(&res) }
    } else {
        let mut affine_bases = vec![C::identity(); coeffs.len()];
        C::Curve::batch_normalize(&bases, &mut affine_bases);
        msm_best(&coeffs, &affine_bases)
    };

    #[cfg(feature = "trace-msm")]
    eprintln!("✓  [MSM] Completed in {:?}", start.elapsed());

    result
}

/// Two channel MSM accumulator
#[derive(Debug, Clone)]
pub struct DualMSM<E: Engine> {
    pub(crate) left: MSMKZG<E>,
    pub(crate) right: MSMKZG<E>,
}

/// A [DualMSM] split into left and right vectors of `(Scalar, Point)` tuples
pub type SplitDualMSM<'a, E> = (
    Vec<(
        &'a CommitmentLabel,
        &'a <E as Engine>::Fr,
        &'a <E as Engine>::G1,
    )>,
    Vec<(
        &'a CommitmentLabel,
        &'a <E as Engine>::Fr,
        &'a <E as Engine>::G1,
    )>,
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
    pub fn split(&self) -> SplitDualMSM<'_, E> {
        let left = izip!(
            self.left.labels.iter(),
            self.left.scalars.iter(),
            self.left.bases.iter()
        )
        .collect();
        let right = izip!(
            self.right.labels.iter(),
            self.right.scalars.iter(),
            self.right.bases.iter(),
        )
        .collect();
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
pub fn msm_with_cached_bases_async<C: CurveAffine + 'static>(
    coeffs: &[C::Scalar],
    device_bases: &PrecomputedBases,
) -> Result<MsmHandle, crate::poly::Error>
where
    C::Scalar: ff::PrimeField + 'static,
{
    #[cfg(feature = "trace-msm")]
    eprintln!("[MSM-ASYNC] Launching async MSM with {} points", coeffs.len());
    
    // Use dispatch helper for type-safe conversion. Errors (rather than
    // panics) so callers can fall back to CPU (fix C5).
    let fq_coeffs = try_as_fq_slice(coeffs).ok_or(crate::poly::Error::OpeningError)?;
    
    let ctx = try_get_msm_context().ok_or(crate::poly::Error::OpeningError)?;
    ctx.msm_with_device_bases_async(fq_coeffs, device_bases)
        .map_err(|_| crate::poly::Error::OpeningError)
}

#[cfg(feature = "gpu")]
/// Batch launch multiple async MSMs
/// 
/// More efficient than calling msm_with_cached_bases_async() in a loop because
/// it minimizes launch overhead and enables better GPU pipelining.
pub fn msm_batch_async<C: CurveAffine>(
    coeffs_batch: &[&[C::Scalar]],
    device_bases: &PrecomputedBases,
) -> Result<Vec<MsmHandle>, crate::poly::Error> {
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
pub fn msm_batch_with_cached_bases<C: CurveAffine + 'static>(
    coeffs_batch: &[&[C::Scalar]],
    device_bases: &PrecomputedBases,
) -> Vec<C::Curve>
where
    C::Scalar: ff::PrimeField + 'static,
{
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

    // Use dispatch helper for type-safe conversion of all batches
    let coeffs_batch_fq: Vec<&[Fq]> = coeffs_batch
        .iter()
        .map(|coeffs| try_as_fq_slice(coeffs)
            .expect("Batch MSM must use midnight_curves::G1Affine (BLS12-381)"))
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

    // Convert results to generic curve type using dispatch helper
    results
        .into_iter()
        .map(|res| projective_to_curve::<C>(res))
        .collect()
}

#[cfg(feature = "gpu")]
/// Async variant of batch MSM - launches computation without blocking.
///
/// Launch batch MSM asynchronously and do CPU work while GPU computes.
pub fn msm_batch_with_cached_bases_async<C: CurveAffine + 'static>(
    coeffs_batch: &[&[C::Scalar]],
    device_bases: &PrecomputedBases,
) -> Result<BatchMsmHandleWrapper<C>, crate::poly::Error>
where
    C::Scalar: ff::PrimeField + 'static,
{
    if coeffs_batch.is_empty() {
        return Err(crate::poly::Error::OpeningError);
    }

    // Use dispatch helper for type-safe conversion
    let coeffs_batch_fq: Vec<&[Fq]> = coeffs_batch
        .iter()
        .map(|coeffs| try_as_fq_slice(coeffs)
            .expect("Async batch MSM must use midnight_curves::G1Affine"))
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
    handle: crate::gpu_accel::BatchMsmHandle,
    _phantom: std::marker::PhantomData<C>,
}

#[cfg(feature = "gpu")]
impl<C: CurveAffine + 'static> BatchMsmHandleWrapper<C> {
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

        // Use dispatch helper for type-safe conversion
        Ok(results
            .into_iter()
            .map(|res| projective_to_curve::<C>(res))
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
pub fn msm_batch_pipelined<C: CurveAffine + 'static>(
    coeffs_batch: &[&[C::Scalar]],
    device_bases: &PrecomputedBases,
) -> Vec<C::Curve>
where
    C::Scalar: ff::PrimeField + 'static,
{
    if coeffs_batch.is_empty() {
        return Vec::new();
    }

    #[cfg(feature = "trace-msm")]
    let start = std::time::Instant::now();
    #[cfg(feature = "trace-msm")]
    eprintln!("[MSM-PIPELINE] Launching {} MSMs asynchronously", coeffs_batch.len());

    // Launch all MSMs without waiting, using dispatch helper for conversion
    let handles: Vec<_> = coeffs_batch
        .iter()
        .map(|coeffs| {
            let fq_coeffs = try_as_fq_slice(coeffs)
                .expect("Pipelined MSM must use midnight_curves::G1Affine");
            let ctx = get_msm_context();
            ctx.msm_with_device_bases_async(fq_coeffs, device_bases)
                .expect("Failed to launch async MSM")
        })
        .collect();

    #[cfg(feature = "trace-msm")]
    eprintln!("   [MSM-PIPELINE] All {} MSMs launched, GPU working...", handles.len());

    // CPU can do work here while GPU computes all MSMs
    // Example: prepare proof metadata, next phase setup, etc.

    // Wait for all results, using dispatch helper for conversion
    let results: Vec<C::Curve> = handles
        .into_iter()
        .map(|handle| {
            let res = handle.wait().expect("Failed to wait on async MSM");
            projective_to_curve::<C>(res)
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
