/**
 * @file icicle_curve_api.cu
 * @brief Icicle-compatible MSM Backend Registration
 * 
 * This file registers our MSM implementation with Icicle's dispatcher system.
 * It uses the REGISTER_MSM_BACKEND macro to register at library load time.
 * 
 * The registration mechanism works as follows:
 * 1. This library is loaded via dlopen() by Icicle runtime
 * 2. Static initializers run the REGISTER_MSM_BACKEND macro
 * 3. The macro calls register_msm("CUDA", impl) 
 * 4. register_msm is resolved from libicicle_curve_bls12_381.so
 * 5. Our implementation is now available for the "CUDA" device type
 * 
 * MONTGOMERY FORM HANDLING:
 * =========================
 * Icicle's Rust bindings send field elements in NON-Montgomery form (via to_repr()).
 * 
 * CRITICAL INSIGHT:
 * - SCALARS are used for BIT EXTRACTION in MSM (to determine bucket indices)
 *   They must remain in STANDARD form - we extract bits from the actual value!
 *   Converting to Montgomery would make MSM compute (R*s mod r) * G instead of s * G
 * 
 * - POINTS use field arithmetic during bucket accumulation
 *   They need to be in MONTGOMERY form for efficient modular multiplication
 * 
 * - RESULTS are returned in standard form (converted from Montgomery)
 */

// Our minimal Icicle-compatible API declarations
#include "icicle_backend_api.cuh"

// Include field.cuh for Montgomery conversion functions
#include "field.cuh"
#include "point.cuh"

// Forward declarations only - avoid including full msm.cuh to prevent duplicate kernel instantiation
#include "msm_fwd.cuh"

using namespace bls12_381;

// =============================================================================
// Montgomery Conversion Kernels  
// =============================================================================

// NOTE: Scalars are NOT converted to Montgomery form!
// MSM extracts bits from scalars to determine bucket indices.
// Converting to Montgomery would change the bit pattern and compute wrong scalar multiplication.

// Convert affine points from standard form to Montgomery form
__global__ void points_to_montgomery_kernel(G1Affine* points, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Affine& p = points[idx];
    Fq x_mont, y_mont;
    field_to_montgomery(x_mont, p.x);
    field_to_montgomery(y_mont, p.y);
    p.x = x_mont;
    p.y = y_mont;
}

// Convert Jacobian projective result (Montgomery) to standard projective (standard form)
// for ICICLE which expects standard projective: x = X/Z, y = Y/Z
//
// Our MSM uses Jacobian: x = X/Z², y = Y/Z³
// ICICLE uses Standard: x = X/Z, y = Y/Z
//
// Conversion: compute affine (x, y) then return as (x, y, 1) in standard form
__global__ void jacobian_to_standard_projective_kernel(G1Projective* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    G1Projective& p = result[idx];
    
    // Check identity
    if (p.Z.is_zero()) {
        // Identity point: set to (0, 1, 0) in standard form
        // But first convert from Montgomery - identity in Montgomery is still identity
        p.X = Fq::zero();
        p.Y = Fq::one();  // This gives 1 in Montgomery form, need standard 1
        // Actually for identity we can just set all to zero except Y=1 in standard form
        for (int i = 0; i < 6; i++) {
            p.X.limbs[i] = 0;
            p.Y.limbs[i] = 0;
            p.Z.limbs[i] = 0;
        }
        p.Y.limbs[0] = 1;  // Y = 1 in standard form
        return;
    }
    
    // Compute affine coordinates from Jacobian: x = X/Z², y = Y/Z³
    Fq z_inv, z_inv_sq, z_inv_cu;
    field_inv(z_inv, p.Z);
    field_sqr(z_inv_sq, z_inv);
    field_mul(z_inv_cu, z_inv_sq, z_inv);
    
    Fq x_aff, y_aff;
    field_mul(x_aff, p.X, z_inv_sq);   // x = X/Z²
    field_mul(y_aff, p.Y, z_inv_cu);   // y = Y/Z³
    
    // Now x_aff, y_aff are in Montgomery form
    // Convert to standard form for ICICLE
    Fq x_std, y_std;
    field_from_montgomery(x_std, x_aff);
    field_from_montgomery(y_std, y_aff);
    
    // Return as standard projective with Z=1 (standard form 1)
    p.X = x_std;
    p.Y = y_std;
    // Z = 1 in standard form
    for (int i = 0; i < 6; i++) p.Z.limbs[i] = 0;
    p.Z.limbs[0] = 1;
}

// =============================================================================
// Debug Kernels (temporary)
// =============================================================================

__global__ void debug_print_scalar(const Fr* scalar, const char* label) {
    printf("[CUDA] %s: 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n",
           label,
           (unsigned long long)scalar[0].limbs[0], (unsigned long long)scalar[0].limbs[1],
           (unsigned long long)scalar[0].limbs[2], (unsigned long long)scalar[0].limbs[3]);
}

__global__ void debug_print_point(const G1Affine* point, const char* label) {
    printf("[CUDA] %s.x: 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n",
           label,
           (unsigned long long)point[0].x.limbs[0], (unsigned long long)point[0].x.limbs[1], 
           (unsigned long long)point[0].x.limbs[2], (unsigned long long)point[0].x.limbs[3], 
           (unsigned long long)point[0].x.limbs[4], (unsigned long long)point[0].x.limbs[5]);
    printf("[CUDA] %s.y: 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n",
           label,
           (unsigned long long)point[0].y.limbs[0], (unsigned long long)point[0].y.limbs[1], 
           (unsigned long long)point[0].y.limbs[2], (unsigned long long)point[0].y.limbs[3], 
           (unsigned long long)point[0].y.limbs[4], (unsigned long long)point[0].y.limbs[5]);
}

__global__ void debug_print_result(const G1Projective* point, const char* label) {
    printf("[CUDA] %s.X: 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n",
           label,
           (unsigned long long)point[0].X.limbs[0], (unsigned long long)point[0].X.limbs[1], 
           (unsigned long long)point[0].X.limbs[2], (unsigned long long)point[0].X.limbs[3], 
           (unsigned long long)point[0].X.limbs[4], (unsigned long long)point[0].X.limbs[5]);
    printf("[CUDA] %s.Y: 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n",
           label,
           (unsigned long long)point[0].Y.limbs[0], (unsigned long long)point[0].Y.limbs[1], 
           (unsigned long long)point[0].Y.limbs[2], (unsigned long long)point[0].Y.limbs[3], 
           (unsigned long long)point[0].Y.limbs[4], (unsigned long long)point[0].Y.limbs[5]);
    printf("[CUDA] %s.Z: 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx\n",
           label,
           (unsigned long long)point[0].Z.limbs[0], (unsigned long long)point[0].Z.limbs[1], 
           (unsigned long long)point[0].Z.limbs[2], (unsigned long long)point[0].Z.limbs[3], 
           (unsigned long long)point[0].Z.limbs[4], (unsigned long long)point[0].Z.limbs[5]);
}

// Print input BEFORE any conversion
__global__ void debug_print_raw_input(const void* data, int size_bytes, const char* label) {
    const unsigned char* bytes = (const unsigned char*)data;
    printf("[CUDA] %s (first 32 bytes): ", label);
    for (int i = 0; i < 32 && i < size_bytes; i++) {
        printf("%02x", bytes[i]);
    }
    printf("\n");
}

// =============================================================================
// MSM Implementation Wrapper  
// =============================================================================

static icicle::eIcicleError msm_cuda_impl(
    const icicle::Device& device,
    const icicle::icicle_scalar_t* scalars,
    const icicle::icicle_affine_t* bases,
    int msm_size,
    const icicle::MSMConfig& config,
    icicle::icicle_projective_t* results)
{
    (void)device; // Unused - we always use CUDA context
    
    // DEBUG: (disabled) Entry point logging
    // printf("[CUDA MSM] Entry: msm_size=%d, scalars_on_device=%d, points_on_device=%d, results_on_device=%d\n",
    //        msm_size, config.are_scalars_on_device, config.are_points_on_device, config.are_results_on_device);
    // printf("[CUDA MSM] Montgomery flags: scalars_mont=%d, points_mont=%d\n",
    //        config.are_scalars_montgomery_form, config.are_points_montgomery_form);
    
    // DEBUG: (disabled) Print raw input bytes (first scalar and first point)
    // if (msm_size > 0) {
    //     if (!config.are_scalars_on_device) {
    //         const unsigned char* scalar_bytes = (const unsigned char*)scalars;
    //         printf("[CUDA MSM] Raw scalar[0]: ");
    //         for (int i = 0; i < 32; i++) printf("%02x", scalar_bytes[i]);
    //         printf("\n");
    //     }
    //     if (!config.are_points_on_device) {
    //         const unsigned char* point_bytes = (const unsigned char*)bases;
    //         printf("[CUDA MSM] Raw point[0].x: ");
    //         for (int i = 0; i < 48; i++) printf("%02x", point_bytes[i]);
    //         printf("\n");
    //         printf("[CUDA MSM] Raw point[0].y: ");
    //         for (int i = 48; i < 96; i++) printf("%02x", point_bytes[i]);
    //         printf("\n");
    //     }
    // }
    
    if (msm_size == 0) {
        return icicle::eIcicleError::SUCCESS;
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    cudaError_t err;
    
    // We need to convert inputs to Montgomery form and outputs from Montgomery form.
    // To avoid modifying input data, we allocate temporary buffers.
    Fr* d_scalars = nullptr;
    G1Affine* d_bases = nullptr;
    G1Projective* d_result = nullptr;
    
    bool allocated_scalars = false;
    bool allocated_bases = false;
    bool allocated_result = false;
    
    // --- Handle scalars ---
    // IMPORTANT: Do NOT convert scalars to Montgomery form!
    // MSM extracts bits from scalars to determine bucket indices.
    // Scalars must remain in standard form so bits represent actual scalar value.
    if (config.are_scalars_on_device) {
        // Data already on device - use directly (no conversion needed)
        d_scalars = const_cast<Fr*>(reinterpret_cast<const Fr*>(scalars));
    } else {
        // Data on host - copy to device (no Montgomery conversion)
        err = cudaMalloc(&d_scalars, msm_size * sizeof(Fr));
        if (err != cudaSuccess) goto cleanup;
        allocated_scalars = true;
        
        err = cudaMemcpyAsync(d_scalars, scalars, msm_size * sizeof(Fr),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
    }
    
    // --- Handle bases (points) ---
    if (config.are_points_on_device) {
        if (!config.are_points_montgomery_form) {
            err = cudaMalloc(&d_bases, msm_size * sizeof(G1Affine));
            if (err != cudaSuccess) goto cleanup;
            allocated_bases = true;
            
            err = cudaMemcpyAsync(d_bases, bases, msm_size * sizeof(G1Affine),
                                  cudaMemcpyDeviceToDevice, stream);
            if (err != cudaSuccess) goto cleanup;
            
            // Convert to Montgomery form
            int threads = 256;
            int blocks = (msm_size + threads - 1) / threads;
            points_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(d_bases, msm_size);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        } else {
            d_bases = const_cast<G1Affine*>(reinterpret_cast<const G1Affine*>(bases));
        }
    } else {
        err = cudaMalloc(&d_bases, msm_size * sizeof(G1Affine));
        if (err != cudaSuccess) goto cleanup;
        allocated_bases = true;
        
        err = cudaMemcpyAsync(d_bases, bases, msm_size * sizeof(G1Affine),
                              cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) goto cleanup;
        
        if (!config.are_points_montgomery_form) {
            int threads = 256;
            int blocks = (msm_size + threads - 1) / threads;
            points_to_montgomery_kernel<<<blocks, threads, 0, stream>>>(d_bases, msm_size);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto cleanup;
        }
    }
    
    // --- Handle result ---
    if (config.are_results_on_device) {
        d_result = reinterpret_cast<G1Projective*>(results);
    } else {
        err = cudaMalloc(&d_result, sizeof(G1Projective));
        if (err != cudaSuccess) goto cleanup;
        allocated_result = true;
    }
    
    // --- Create modified config for internal MSM ---
    {
        // DEBUG: (disabled) Print first scalar and point after conversion
        // debug_print_scalar<<<1, 1, 0, stream>>>(d_scalars, "scalar[0] (standard form)");
        // debug_print_point<<<1, 1, 0, stream>>>(d_bases, "bases[0] after to_mont");
        // cudaStreamSynchronize(stream);
        
        icicle::MSMConfig internal_cfg = config;
        internal_cfg.are_scalars_on_device = true;
        internal_cfg.are_points_on_device = true;
        internal_cfg.are_results_on_device = true;
        // Scalars remain in standard form - MSM needs actual bit values
        internal_cfg.are_scalars_montgomery_form = false;  
        // Points are converted to Montgomery for field arithmetic
        internal_cfg.are_points_montgomery_form = true;
        
        // Call our MSM implementation
        err = msm::msm_cuda<Fr, G1Affine, G1Projective>(
            d_scalars, d_bases, msm_size, internal_cfg, d_result);
        if (err != cudaSuccess) goto cleanup;
        
        // DEBUG: (disabled) Print result after MSM (before conversion)
        // debug_print_result<<<1, 1, 0, stream>>>(d_result, "result after MSM (Jacobian/mont)");
        // cudaStreamSynchronize(stream);
    }
    
    // --- Convert result from Jacobian/Montgomery to Standard/Standard form ---
    // Our MSM uses Jacobian projective (x=X/Z², y=Y/Z³) in Montgomery form
    // ICICLE expects standard projective (x=X/Z, y=Y/Z) in standard form
    {
        jacobian_to_standard_projective_kernel<<<1, 1, 0, stream>>>(d_result, 1);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        
        // DEBUG: (disabled) Print result after conversion
        // debug_print_result<<<1, 1, 0, stream>>>(d_result, "result after jacobian_to_standard");
        // cudaStreamSynchronize(stream);
    }
    
    // --- Copy result back if needed ---
    if (!config.are_results_on_device) {
        err = cudaMemcpyAsync(results, d_result, sizeof(G1Projective),
                              cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Synchronize if not async
    if (!config.is_async) {
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) goto cleanup;
    }
    
cleanup:
    if (allocated_scalars && d_scalars) cudaFree(d_scalars);
    if (allocated_bases && d_bases) cudaFree(d_bases);
    if (allocated_result && d_result) cudaFree(d_result);
    
    return (err == cudaSuccess) ? icicle::eIcicleError::SUCCESS : icicle::eIcicleError::UNKNOWN_ERROR;
}

// Precompute bases wrapper
static icicle::eIcicleError msm_precompute_bases_cuda_impl(
    const icicle::Device& device,
    const icicle::icicle_affine_t* input_bases,
    int bases_size,
    const icicle::MSMConfig& config,
    icicle::icicle_affine_t* output_bases)
{
    (void)device; // Unused - we always use CUDA context
    
    // Our MSM doesn't require precomputation - just copy if needed
    if (output_bases != input_bases && config.are_points_on_device) {
        // Use Icicle's type size for the copy since that's what the caller expects
        // Icicle affine_t is 2 * 48 bytes = 96 bytes
        constexpr size_t ICICLE_AFFINE_SIZE = 96;  // 2 * 12 * 4 bytes (2 fields of 12 x 32-bit limbs)
        
        cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
        cudaError_t err = cudaMemcpyAsync(
            output_bases, input_bases, 
            bases_size * ICICLE_AFFINE_SIZE,
            cudaMemcpyDeviceToDevice, stream
        );
        if (!config.is_async) cudaStreamSynchronize(stream);
        return (err == cudaSuccess) ? icicle::eIcicleError::SUCCESS : icicle::eIcicleError::COPY_FAILED;
    }
    return icicle::eIcicleError::SUCCESS;
}

// =============================================================================
// Backend Registration
// =============================================================================

// Register our MSM implementation with the "CUDA" device type
// This uses Icicle's macro which creates a static initializer
REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND("CUDA", msm_precompute_bases_cuda_impl);
REGISTER_MSM_BACKEND("CUDA", msm_cuda_impl);

// TODO: Add G2 MSM registration when implemented
// #ifdef G2_ENABLED
// REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND("CUDA", msm_g2_precompute_bases_cuda_impl);
// REGISTER_MSM_G2_BACKEND("CUDA", msm_g2_cuda_impl);
// #endif
