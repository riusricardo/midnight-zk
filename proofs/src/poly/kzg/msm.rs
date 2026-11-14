use std::{any::TypeId, fmt::Debug};

#[cfg(feature = "gpu")]
use std::sync::OnceLock;

use ff::Field;
use group::{Curve, Group};
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
/// Call this at application startup so the first proof request does not pay
/// for the backend's initialization. If the backend is already initialized,
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
/// Fallible MSM using pre-uploaded GPU bases.
///
/// Returns an error (for CPU fallback) instead of panicking when the GPU/backend is
/// unavailable or the MSM fails (correctness fix C5). This is the primitive used by the
/// commit hot path.
pub fn try_msm_with_cached_bases<C: CurveAffine + 'static>(
    coeffs: &[C::Scalar],
    device_bases: &PrecomputedBases,
) -> Result<C::Curve, crate::poly::Error>
where
    C::Scalar: ff::PrimeField + 'static,
{
    // Verify we're using midnight_curves (BLS12-381) and convert coefficients.
    // The curve check makes the `projective_to_curve` reinterpretation below
    // valid for any `C` (it only debug-asserts it).
    if !is_g1_affine::<C>() {
        return Err(crate::poly::Error::OpeningError);
    }
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
                let fq_coeffs = unsafe { &*(coeffs as *const _ as *const [Fq]) };
                let proj_bases = unsafe { &*(bases as *const _ as *const [G1Projective]) };

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

    let result = msm_cpu::<C>(coeffs, bases);

    #[cfg(feature = "trace-msm")]
    eprintln!("✓  [MSM] Completed in {:?}", start.elapsed());

    result
}

/// The CPU multi-scalar multiplication of [`msm_specific`], which never uses
/// the GPU: BLST's multi-exponentiation for midnight's G1, `msm_best`
/// otherwise and above 2^19 points.
#[allow(unsafe_code)]
pub(crate) fn msm_cpu<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C::Curve]) -> C::Curve {
    // We empirically checked that for MSMs larger than 2**18, the blstrs
    // implementation regresses.
    if coeffs.len() <= (2 << 18) && TypeId::of::<C>() == TypeId::of::<midnight_curves::G1Affine>() {
        #[cfg(feature = "trace-msm")]
        eprintln!("   [MSM] Using BLST multi_exp (size: {} points)", coeffs.len());

        // Safe: we just checked type
        let coeffs = unsafe { &*(coeffs as *const _ as *const [Fq]) };
        let bases = unsafe { &*(bases as *const _ as *const [G1Projective]) };
        let res = G1Projective::multi_exp(bases, coeffs);
        unsafe { std::mem::transmute_copy(&res) }
    } else {
        let mut affine_bases = vec![C::identity(); coeffs.len()];
        C::Curve::batch_normalize(bases, &mut affine_bases);
        msm_best(coeffs, &affine_bases)
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
// Async MSM API
// =============================================================================

#[cfg(feature = "gpu")]
/// Launch an MSM using pre-uploaded GPU bases
/// 
/// Returns a handle that completes to the result when waited.
/// The backend completes the MSM before the launch returns (its final window
/// combination runs on the host), so the handle defers only the copy of the
/// result back: it does not overlap CPU work with the MSM.
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
/// // Launch the MSMs
/// let handles: Vec<_> = polynomials.iter()
///     .map(|poly| msm_with_cached_bases_async(&poly.values, &device_bases))
///     .collect::<Result<_, _>>()?;
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
