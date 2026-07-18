use std::{fmt::Debug, io};

use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve, Group};
use midnight_curves::pairing::{Engine, MultiMillerLoop};
use rand_core::RngCore;

#[cfg(feature = "gpu")]
use std::sync::Arc;
#[cfg(feature = "gpu")]
use once_cell::sync::OnceCell;

// GPU types via bridge module (single import point)
#[cfg(feature = "gpu")]
use crate::gpu_accel::{
    PrecomputedBases, TypeConverter, DeviceVec, HostSlice, IcicleStream,
    IcicleDevice, icicle_set_device, ensure_backend_loaded,
    should_use_gpu, should_use_gpu_batch,
};

use crate::{
    poly::commitment::Params,
    utils::{
        arithmetic::{g_to_lagrange, parallelize},
        helpers::ProcessedSerdeObject,
        SerdeFormat,
    },
};

/// These are the public parameters for the polynomial commitment scheme.
/// 
/// # GPU Acceleration
/// 
/// When compiled with the `gpu` feature, SRS bases are cached in GPU memory:
/// - `g_gpu`: Coefficient form bases for `commit()`
/// - `g_lagrange_gpu`: Lagrange form bases for `commit_lagrange()`
/// 
/// Bases are uploaded lazily on first use via `get_or_upload_gpu_bases()`.
/// This one-time upload eliminates per-MSM conversion and transfer overhead,
/// providing ~40% speedup for large MSMs.
/// 
/// The caches use `Arc<OnceCell<...>>` for thread-safe lazy initialization
/// that survives `Clone` operations.
#[derive(Clone)]
pub struct ParamsKZG<E: Engine> {
    pub(crate) g: Vec<E::G1>,
    pub(crate) g_lagrange: Vec<E::G1>,
    pub(crate) g2: E::G2,
    pub(crate) s_g2: E::G2,
    
    /// Cached GPU bases for coefficient form commitments.
    /// Initialized lazily via `get_or_upload_gpu_bases()`.
    /// Uses Arc to survive Clone while sharing the same GPU memory.
    #[cfg(feature = "gpu")]
    pub(crate) g_gpu: Arc<OnceCell<PrecomputedBases>>,
    
    /// Cached GPU bases for Lagrange form commitments.
    /// Initialized lazily via `get_or_upload_gpu_lagrange_bases()`.
    #[cfg(feature = "gpu")]
    pub(crate) g_lagrange_gpu: Arc<OnceCell<PrecomputedBases>>,
}

impl<E: Engine> Debug for ParamsKZG<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParamsKZG")
            .field("g_len", &self.g.len())
            .field("g_lagrange_len", &self.g_lagrange.len())
            .field("g2", &self.g2)
            .field("s_g2", &self.s_g2)
            .finish()
    }
}

impl<E: Engine> Params for ParamsKZG<E> {
    fn max_k(&self) -> u32 {
        assert_eq!(self.g.len(), self.g_lagrange.len());

        self.g.len().ilog2()
    }

    fn downsize(&mut self, new_k: u32) {
        ParamsKZG::<E>::downsize(self, new_k)
    }
}

impl<E: Engine + Debug> ParamsKZG<E> {
    /// Downsize the current parameters to match a smaller `k`.
    pub fn downsize(&mut self, new_k: u32) {
        if self.max_k() == new_k {
            return;
        }

        let n = 1 << new_k;
        assert!(n < self.g_lagrange.len() as u32);
        self.g.truncate(n as usize);
        self.g_lagrange = g_to_lagrange(&self.g, new_k);
    }

    /// Initializes parameters for the curve, draws toxic secret from given rng.
    /// MUST NOT be used in production.
    pub fn unsafe_setup<R: RngCore>(k: u32, rng: R) -> Self {
        // Largest root of unity exponent of the Engine is `2^E::Fr::S`, so we can
        // only support FFTs of polynomials below degree `2^E::Fr::S`.
        assert!(k <= E::Fr::S);
        let n: u64 = 1 << k;

        // Calculate g = [G1, [s] G1, [s^2] G1, ..., [s^(n-1)] G1] in parallel.
        let g1 = E::G1::generator();
        let s = <E::Fr>::random(rng);

        let mut g = vec![E::G1::identity(); n as usize];
        parallelize(&mut g, |g, start| {
            let mut current_g: E::G1 = g1;
            current_g *= s.pow_vartime([start as u64]);
            for g in g.iter_mut() {
                *g = current_g;
                current_g *= s;
            }
        });

        let mut g_lagrange = vec![E::G1::identity(); n as usize];
        let mut root = E::Fr::ROOT_OF_UNITY;
        for _ in k..E::Fr::S {
            root = root.square();
        }
        let n_inv = E::Fr::from(n).invert().expect("inversion should be ok for n = 1<<k");
        let multiplier = (s.pow_vartime([n]) - E::Fr::ONE) * n_inv;
        parallelize(&mut g_lagrange, |g, start| {
            for (idx, g) in g.iter_mut().enumerate() {
                let offset = start + idx;
                let root_pow = root.pow_vartime([offset as u64]);
                let scalar = multiplier * root_pow * (s - root_pow).invert().unwrap();
                *g = g1 * scalar;
            }
        });

        let g2 = E::G2::generator();
        let s_g2 = g2 * s;

        Self {
            g,
            g_lagrange,
            g2,
            s_g2,
            #[cfg(feature = "gpu")]
            g_gpu: Arc::new(OnceCell::new()),
            #[cfg(feature = "gpu")]
            g_lagrange_gpu: Arc::new(OnceCell::new()),
        }
    }

    /// Initializes parameters for the curve through existing parameters
    /// k, g, g_lagrange (optional), g2, s_g2
    pub fn from_parts(
        &self,
        k: u32,
        g: Vec<E::G1>,
        g_lagrange: Option<Vec<E::G1>>,
        g2: E::G2,
        s_g2: E::G2,
    ) -> Self {
        Self {
            g_lagrange: match g_lagrange {
                Some(g_l) => g_l,
                None => g_to_lagrange(&g, k),
            },
            g,
            g2,
            s_g2,
            #[cfg(feature = "gpu")]
            g_gpu: Arc::new(OnceCell::new()),
            #[cfg(feature = "gpu")]
            g_lagrange_gpu: Arc::new(OnceCell::new()),
        }
    }

    /// Get or upload GPU bases for coefficient form (lazy initialization)
    /// 
    /// Following ingonyama-zk pattern: bases are uploaded ONCE and cached in GPU memory,
    /// eliminating per-MSM conversion and upload overhead.
    /// 
    /// # Montgomery Form Optimization
    /// 
    /// Bases are uploaded in Montgomery form (the internal representation of midnight-curves).
    /// This eliminates per-MSM D2D copy + Montgomery conversion in the CUDA backend.
    /// When using these bases, set `cfg.are_bases_montgomery_form = true`.
    #[cfg(feature = "gpu")]
    pub fn get_or_upload_gpu_bases(&self) -> &PrecomputedBases {
        self.g_gpu.get_or_init(|| {
            #[cfg(feature = "trace-msm")]
            eprintln!("[GPU] Uploading {} SRS bases to GPU in Montgomery form (one-time cost)...", self.g.len());
            
            #[cfg(feature = "trace-msm")]
            let start = std::time::Instant::now();
            
            // Ensure ICICLE backend is loaded (singleton - only loads once globally)
            ensure_backend_loaded().expect("Failed to load ICICLE backend");
            
            // CRITICAL: Set device before allocating GPU memory
            // This is needed because get_or_init() might be called from different thread contexts
            let device = IcicleDevice::new("CUDA", 0);
            icicle_set_device(&device).expect("Failed to set GPU device");
            
            // Convert G1Projective to G1Affine
            let mut bases_affine = vec![E::G1Affine::identity(); self.g.len()];
            E::G1::batch_normalize(&self.g, &mut bases_affine);
            
            // Zero-copy view as ICICLE points (Montgomery form preserved!)
            // midnight-curves stores Fp coordinates in Montgomery form internally,
            // which is exactly what ICICLE expects when are_bases_montgomery_form=true.
            let bases_midnight: &[midnight_curves::G1Affine] = unsafe {
                std::mem::transmute(bases_affine.as_slice())
            };
            let icicle_points = TypeConverter::g1_slice_as_icicle(bases_midnight);
            
            // Upload to GPU (Montgomery form - no conversion needed at MSM time!)
            let stream = IcicleStream::default();
            let mut device_bases = DeviceVec::device_malloc_async(icicle_points.len(), &stream)
                .expect("Failed to allocate GPU memory for bases");
            device_bases.copy_from_host_async(HostSlice::from_slice(icicle_points), &stream)
                .expect("Failed to upload bases to GPU");
            stream.synchronize().expect("Failed to synchronize GPU stream");
            
            #[cfg(feature = "trace-msm")]
            eprintln!("✓  [GPU] Bases uploaded in Montgomery form in {:?} (zero-copy MSM ready)", start.elapsed());
            
            // Wrap in PrecomputedBases (no precomputation, just normal upload)
            PrecomputedBases::new(device_bases, icicle_points.len())
        })
    }
    
    /// Get or upload GPU bases for Lagrange form (lazy initialization)
    /// 
    /// Same as `get_or_upload_gpu_bases()` but for Lagrange form bases.
    /// Bases are stored in Montgomery form for zero-copy MSM execution.
    #[cfg(feature = "gpu")]
    pub fn get_or_upload_gpu_lagrange_bases(&self) -> &PrecomputedBases {
        self.g_lagrange_gpu.get_or_init(|| {
            #[cfg(feature = "trace-msm")]
            eprintln!("[GPU] Uploading {} Lagrange bases to GPU in Montgomery form (one-time cost)...", self.g_lagrange.len());
            
            #[cfg(feature = "trace-msm")]
            let start = std::time::Instant::now();
            
            // Ensure ICICLE backend is loaded (singleton - only loads once globally)
            ensure_backend_loaded().expect("Failed to load ICICLE backend");
            
            // CRITICAL: Set device before allocating GPU memory
            let device = IcicleDevice::new("CUDA", 0);
            icicle_set_device(&device).expect("Failed to set GPU device");
            
            // Convert G1Projective to G1Affine
            let mut bases_affine = vec![E::G1Affine::identity(); self.g_lagrange.len()];
            E::G1::batch_normalize(&self.g_lagrange, &mut bases_affine);
            
            // Zero-copy view as ICICLE points (Montgomery form preserved!)
            let bases_midnight: &[midnight_curves::G1Affine] = unsafe {
                std::mem::transmute(bases_affine.as_slice())
            };
            let icicle_points = TypeConverter::g1_slice_as_icicle(bases_midnight);
            
            // Upload to GPU (Montgomery form - no conversion needed at MSM time!)
            let stream = IcicleStream::default();
            let mut device_bases = DeviceVec::device_malloc_async(icicle_points.len(), &stream)
                .expect("Failed to allocate GPU memory for Lagrange bases");
            device_bases.copy_from_host_async(HostSlice::from_slice(icicle_points), &stream)
                .expect("Failed to upload Lagrange bases to GPU");
            stream.synchronize().expect("Failed to synchronize GPU stream");
            
            #[cfg(feature = "trace-msm")]
            eprintln!("✓  [GPU] Lagrange bases uploaded in Montgomery form in {:?} (zero-copy MSM ready)", start.elapsed());
            
            // Wrap in PrecomputedBases (no precomputation, just normal upload)
            PrecomputedBases::new(device_bases, icicle_points.len())
        })
    }

    /// Commit to multiple Lagrange polynomials with GPU pipelining
    /// 
    /// This is the production-ready batch commit that provides optimal performance
    /// for multiple commitments by enabling GPU kernel pipelining.
    /// 
    /// Falls back to sequential sync commits if async is not beneficial.
    #[cfg(feature = "gpu")]
    pub fn commit_lagrange_batch(
        &self,
        polys: &[&crate::poly::Polynomial<E::Fr, crate::poly::LagrangeCoeff>],
    ) -> Vec<E::G1>
    where
        E::G1Affine: crate::utils::arithmetic::CurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
    {
        use crate::poly::kzg::msm::{msm_batch_async, msm_with_cached_bases};
        
        if polys.is_empty() {
            return vec![];
        }
        
        // Check if async/GPU is beneficial for this batch
        // Consider total work, not just individual MSM size
        let individual_size = polys[0].len();
        let batch_count = polys.len();
        let use_async = batch_count > 1 && should_use_gpu_batch(individual_size, batch_count);
        
        #[cfg(feature = "trace-kzg")]
        if batch_count > 1 {
            eprintln!(
                "[KZG::batch] {} polys × {} points = {} total work, use_async={}",
                batch_count, individual_size, batch_count * individual_size, use_async
            );
        }
        
        if use_async {
            #[cfg(feature = "trace-kzg")]
            eprintln!("[KZG::batch] Using async pipeline for {} Lagrange commits", polys.len());
            
            // Convert all polynomials to scalar vectors
            let scalars: Vec<Vec<E::Fr>> = polys.iter()
                .map(|poly| {
                    let mut s = Vec::with_capacity(poly.len());
                    s.extend(poly.iter());
                    s
                })
                .collect();
            
            let device_bases = self.get_or_upload_gpu_lagrange_bases();
            let scalar_refs: Vec<&[E::Fr]> = scalars.iter().map(|s| s.as_slice()).collect();
            
            // Launch all MSMs asynchronously
            match msm_batch_async::<E::G1Affine>(&scalar_refs, device_bases) {
                Ok(handles) => {
                    // Wait for all results and transmute to E::G1
                    return handles.into_iter()
                        .map(|h| {
                            let result = h.wait().expect("Async commit failed");
                            // Safe: E::G1 is midnight_curves::G1Projective for all supported curves
                            unsafe { std::mem::transmute_copy(&result) }
                        })
                        .collect();
                }
                Err(_) => {
                    // Fallback to sync
                    #[cfg(feature = "trace-kzg")]
                    eprintln!("[KZG::batch] Async failed, falling back to sync");
                }
            }
        }
        
        // Sync path (fallback or not beneficial)
        // Use polynomial directly via Deref - no allocation needed
        polys.iter()
            .map(|poly| {
                let scalars: &[E::Fr] = &***poly;
                let size = scalars.len();
                assert!(self.g_lagrange.len() >= size);
                
                if should_use_gpu(size) {
                    let device_bases = self.get_or_upload_gpu_lagrange_bases();
                    msm_with_cached_bases::<E::G1Affine>(scalars, device_bases)
                } else {
                    use crate::poly::kzg::msm::msm_specific;
                    msm_specific::<E::G1Affine>(scalars, &self.g_lagrange[0..size])
                }
            })
            .collect()
    }

    /// Returns the committed lagrange polynomials of these KZG params.
    pub fn g_lagrange(&self) -> &[E::G1] {
        &self.g_lagrange
    }

    /// Returns generator on G2
    pub fn g2(&self) -> E::G2 {
        self.g2
    }

    /// Returns first power of secret on G2
    pub fn s_g2(&self) -> E::G2 {
        self.s_g2
    }

    /// Writes parameters to buffer
    pub fn write_custom<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()>
    where
        E::G1: Curve + ProcessedSerdeObject,
        E::G2: Curve + ProcessedSerdeObject,
    {
        writer.write_all(&self.g.len().ilog2().to_le_bytes())?;
        for el in self.g.iter() {
            el.write(writer, format)?;
        }
        for el in self.g_lagrange.iter() {
            el.write(writer, format)?;
        }
        self.g2.write(writer, format)?;
        self.s_g2.write(writer, format)?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read_custom<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self>
    where
        E::G1: Curve + ProcessedSerdeObject,
        E::G2: Curve + ProcessedSerdeObject,
    {
        let mut k = [0u8; 4];
        reader.read_exact(&mut k[..])?;
        let k = u32::from_le_bytes(k);
        let n = 1 << k;

        let (g, g_lagrange) = match format {
            SerdeFormat::Processed => {
                use group::GroupEncoding;
                let load_points_from_file_parallelly =
                    |reader: &mut R| -> io::Result<Vec<Option<E::G1>>> {
                        let mut points_compressed =
                            vec![<<E as Engine>::G1 as GroupEncoding>::Repr::default(); n];
                        for points_compressed in points_compressed.iter_mut() {
                            reader.read_exact((*points_compressed).as_mut())?;
                        }

                        let mut points = vec![Option::<E::G1>::None; n];
                        parallelize(&mut points, |points, chunks| {
                            for (i, point) in points.iter_mut().enumerate() {
                                *point =
                                    Option::from(E::G1::from_bytes(&points_compressed[chunks + i]));
                            }
                        });
                        Ok(points)
                    };

                let g = load_points_from_file_parallelly(reader)?;
                let g: Vec<<E as Engine>::G1> = g
                    .iter()
                    .map(|point| point.ok_or_else(|| io::Error::other("invalid point encoding")))
                    .collect::<Result<_, _>>()?;
                let g_lagrange = load_points_from_file_parallelly(reader)?;
                let g_lagrange: Vec<<E as Engine>::G1> = g_lagrange
                    .iter()
                    .map(|point| point.ok_or_else(|| io::Error::other("invalid point encoding")))
                    .collect::<Result<_, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytes => {
                let g = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format))
                    .collect::<Result<Vec<_>, _>>()?;
                let g_lagrange = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format))
                    .collect::<Result<Vec<_>, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytesUnchecked => {
                // avoid try branching for performance
                let g = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format).unwrap())
                    .collect::<Vec<_>>();
                let g_lagrange = (0..n)
                    .map(|_| <E::G1 as ProcessedSerdeObject>::read(reader, format).unwrap())
                    .collect::<Vec<_>>();
                (g, g_lagrange)
            }
        };

        let g2 = E::G2::read(reader, format)?;
        let s_g2 = E::G2::read(reader, format)?;

        Ok(Self {
            g,
            g_lagrange,
            g2,
            s_g2,
            #[cfg(feature = "gpu")]
            g_gpu: Arc::new(OnceCell::new()),
            #[cfg(feature = "gpu")]
            g_lagrange_gpu: Arc::new(OnceCell::new()),
        })
    }
}

// TODO: see the issue at https://github.com/appliedzkp/halo2/issues/45
// So we probably need much smaller verifier key. However for new bases in g1
// should be in verifier keys.
/// KZG multi-open verification parameters
#[derive(Clone, Debug)]
pub struct ParamsVerifierKZG<E: MultiMillerLoop> {
    pub(crate) s_g2: E::G2,
    pub(crate) n_g2_prepared: E::G2Prepared,
    pub(crate) s_g2_prepared: E::G2Prepared,
}

impl<E: MultiMillerLoop + Debug> ParamsVerifierKZG<E>
where
    E::G2: Curve + ProcessedSerdeObject,
{
    /// Writes parameters to buffer
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        self.s_g2.write(writer, format)?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        let s_g2 = E::G2::read(reader, format)?;
        let s_g2_prepared = E::G2Prepared::from(s_g2.into());
        let n_g2_prepared = E::G2Prepared::from(-E::G2Affine::generator());

        Ok(Self {
            s_g2,
            n_g2_prepared,
            s_g2_prepared,
        })
    }
}

impl<E: MultiMillerLoop + Debug> ParamsKZG<E> {
    /// Consume the prover parameters into verifier parameters. Need to specify
    /// the size of public inputs.
    pub fn verifier_params(&self) -> ParamsVerifierKZG<E> {
        let n_g2_prepared = E::G2Prepared::from((-self.g2).into());
        let s_g2_prepared = E::G2Prepared::from(self.s_g2.into());
        ParamsVerifierKZG {
            s_g2: self.s_g2,
            n_g2_prepared,
            s_g2_prepared,
        }
    }
}

#[cfg(test)]
mod test {
    use rand_core::OsRng;

    use crate::{
        poly::{
            commitment::PolynomialCommitmentScheme,
            kzg::{params::ParamsKZG, KZGCommitmentScheme},
        },
        utils::SerdeFormat,
    };

    #[test]
    fn test_commit_lagrange() {
        const K: u32 = 6;

        use midnight_curves::{Bls12, Fq};

        use crate::poly::EvaluationDomain;

        let params: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let domain = EvaluationDomain::new(1, K);

        let mut a = domain.empty_lagrange();

        for (i, a) in a.iter_mut().enumerate() {
            *a = Fq::from(i as u64);
        }

        let b = domain.lagrange_to_coeff(a.clone());

        let tmp = KZGCommitmentScheme::commit_lagrange(&params, &a);
        let commitment = KZGCommitmentScheme::commit(&params, &b);

        assert_eq!(commitment, tmp);
    }

    #[test]
    fn test_parameter_serialisation_roundtrip() {
        const K: u32 = 4;

        use midnight_curves::Bls12;

        let params0: ParamsKZG<Bls12> = ParamsKZG::unsafe_setup(K, OsRng);
        let mut data = vec![];
        ParamsKZG::write_custom(&params0, &mut data, SerdeFormat::RawBytesUnchecked).unwrap();
        let params1 =
            ParamsKZG::<Bls12>::read_custom::<_>(&mut &data[..], SerdeFormat::RawBytesUnchecked)
                .unwrap();

        assert_eq!(params0.g.len(), params1.g.len());
        assert_eq!(params0.g_lagrange.len(), params1.g_lagrange.len());

        assert_eq!(params0.g, params1.g);
        assert_eq!(params0.g_lagrange, params1.g_lagrange);
        assert_eq!(params0.g2, params1.g2);
        assert_eq!(params0.s_g2, params1.s_g2);
    }
}
