/**
 * @file point_ops.cu
 * @brief Batch Point Operations for BLS12-381 G1 and G2
 * 
 * Provides batch operations on elliptic curve points for efficient parallel processing.
 * 
 * ARCHITECTURE:
 * =============
 * All kernels are defined in this file and called from this file only (self-contained).
 * This is required by CUDA's static library linking model.
 * 
 * Operations provided:
 * - Batch affine ↔ projective conversion
 * - Batch point addition
 * - Batch point doubling
 * - Batch point negation
 * - Batch scalar multiplication
 * 
 * Performance note: These kernels use naive per-element operations.
 * For MSM, use the specialized Pippenger implementation in msm.cu.
 */

#include "point.cuh"
#include "icicle_types.cuh"

namespace curve {

using namespace bls12_381;

// =============================================================================
// Batch Point Operations
// =============================================================================

/**
 * @brief Batch affine to projective conversion
 */
__global__ void batch_affine_to_projective_g1_kernel(
    G1Projective* output,
    const G1Affine* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = G1Projective::from_affine(input[idx]);
}

/**
 * @brief Batch projective to affine conversion
 */
__global__ void batch_projective_to_affine_g1_naive_kernel(
    G1Affine* output,
    const G1Projective* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = input[idx].to_affine();
}

/**
 * @brief Batch projective to affine using naive per-element inversion
 */
void batch_projective_to_affine_g1(
    G1Affine* d_output,
    const G1Projective* d_input,
    int size,
    cudaStream_t stream
) {
    // Optimal thread count for memory-bound point operations
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    batch_projective_to_affine_g1_naive_kernel<<<blocks, threads, 0, stream>>>(
        d_output, d_input, size
    );
}

/**
 * @brief Batch point addition
 */
__global__ void batch_add_g1_kernel(
    G1Projective* output,
    const G1Projective* lhs,
    const G1Projective* rhs,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    g1_add(output[idx], lhs[idx], rhs[idx]);
}

/**
 * @brief Batch point doubling
 */
__global__ void batch_double_g1_kernel(
    G1Projective* output,
    const G1Projective* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    g1_double(output[idx], input[idx]);
}

/**
 * @brief Batch point negation
 */
__global__ void batch_negate_g1_kernel(
    G1Projective* output,
    const G1Projective* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Projective p = input[idx];
    g1_neg(output[idx], p);
}

// =============================================================================
// GLV Endomorphism Constants for BLS12-381
// =============================================================================
// For BLS12-381 G1, the curve has an efficient endomorphism:
//   φ(P) = (β*x, y) where β is a cube root of unity in Fq
// 
// This allows decomposing scalar k = k1 + k2*λ where λ is the eigenvalue
// and both k1, k2 have ~128 bits (half the scalar size).
// 
// GLV speedup: 2 half-size scalar muls instead of 1 full-size → ~2x faster
//
// NOTE: GLV decomposition is not yet implemented. These constants are 
// provided for future optimization. Current scalar multiplication uses
// windowed method (w=4) which provides ~2x speedup over naive double-and-add.

// β = cube root of unity in Fq (in Montgomery form)
// β^3 = 1 mod p, β ≠ 1
// Verified: β in Montgomery form, matches ZETA_BASE from curves crate
__device__ __constant__ uint64_t GLV_BETA[6] = {
    0xcd03c9e48671f071ULL, 0x5dab22461fcda5d2ULL,
    0x587042afd3851b95ULL, 0x8eb60ebe01bacb9eULL,
    0x03f97d6e83d050d2ULL, 0x18f0206554638741ULL
};

// λ = eigenvalue of φ in Fr (φ(P) = λ*P for P in G1)
// λ = z^2 - 1 (where z = -0xd201000000010000)
// λ^2 + λ + 1 = 0 mod r
// Verified: Satisfies eigenvalue equation
__device__ __constant__ uint64_t GLV_LAMBDA[4] = {
    0x00000000ffffffffULL, 0xac45a4010001a402ULL,
    0x0ULL, 0x0ULL
};

// Decomposition lattice vectors for GLV (in regular form, not Montgomery)
// v1 = (v1_0, v1_1) = (λ, -1)
// v2 = (v2_0, v2_1) = (1, λ+1)
// These satisfy: v1_0 + λ*v1_1 ≡ 0 (mod r) and v2_0 + λ*v2_1 ≡ 0 (mod r)
// Verified: Both basis vectors satisfy lattice reduction properties
__device__ __constant__ uint64_t GLV_V1_0[2] = { 0x00000000ffffffffULL, 0xac45a4010001a402ULL }; // λ
__device__ __constant__ uint64_t GLV_V1_1[2] = { 0xffffffffffffffffULL, 0xffffffffffffffffULL }; // -1
__device__ __constant__ uint64_t GLV_V2_0[2] = { 0x1ULL, 0x0ULL }; // 1
__device__ __constant__ uint64_t GLV_V2_1[2] = { 0x0000000100000000ULL, 0xac45a4010001a402ULL }; // λ+1 = z^2

/**
 * @brief Apply GLV endomorphism: φ(P) = (β*x, y)
 * 
 * NOTE: This function is defined but not currently used in scalar multiplication.
 * Reserved for future GLV implementation.
 */
__device__ __forceinline__ void g1_endomorphism(G1Projective& result, const G1Projective& p) {
    // Load beta
    Fq beta;
    for (int i = 0; i < 6; i++) {
        beta.limbs[i] = GLV_BETA[i];
    }
    
    // φ(X, Y, Z) = (β*X, Y, Z)
    field_mul(result.X, p.X, beta);
    result.Y = p.Y;
    result.Z = p.Z;
}

/**
 * @brief Apply GLV endomorphism to affine point: φ(P) = (β*x, y)
 * 
 * NOTE: This function is defined but not currently used in scalar multiplication.
 * Reserved for future GLV implementation.
 */
__device__ __forceinline__ void g1_endomorphism_affine(G1Affine& result, const G1Affine& p) {
    Fq beta;
    for (int i = 0; i < 6; i++) {
        beta.limbs[i] = GLV_BETA[i];
    }
    
    field_mul(result.x, p.x, beta);
    result.y = p.y;
}

/**
 * @brief Windowed scalar multiplication with precomputation
 * 
 * Window size w=4 gives good balance: 16-element table, ~64 additions
 */
constexpr int SCALAR_MUL_WINDOW = 4;
constexpr int SCALAR_MUL_TABLE_SIZE = (1 << SCALAR_MUL_WINDOW); // 16

/**
 * @brief Batch scalar multiplication using windowed method
 * 
 * Uses windowed method with w=4 for efficient scalar multiplication.
 * 
 * NOTE: GLV decomposition is not yet implemented in this kernel.
 * The constants GLV_BETA and GLV_LAMBDA are defined for future optimization.
 * Current implementation uses standard windowed scalar multiplication.
 */
__global__ void batch_scalar_mul_g1_kernel(
    G1Projective* output,
    const G1Affine* bases,
    const Fr* scalars,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Affine base_affine = bases[idx];
    Fr scalar = scalars[idx];
    
    // Build precomputation table: table[i] = i * P for i = 0..15
    // Using affine coordinates to save memory
    G1Projective table[SCALAR_MUL_TABLE_SIZE];
    table[0] = G1Projective::identity();
    table[1] = G1Projective::from_affine(base_affine);
    
    // Compute 2P, 3P, ..., 15P
    for (int i = 2; i < SCALAR_MUL_TABLE_SIZE; i++) {
        if (i % 2 == 0) {
            // i = 2k, table[i] = 2 * table[k]
            g1_double(table[i], table[i/2]);
        } else {
            // i = 2k+1, table[i] = table[2k] + P
            g1_add_mixed(table[i], table[i-1], base_affine);
        }
    }
    
    G1Projective result = G1Projective::identity();
    
    // Process scalar in windows from MSB to LSB
    // Fr is 255 bits = 64 windows of 4 bits
    constexpr int SCALAR_BITS = Fr::LIMBS * 64;
    constexpr int NUM_WINDOWS = (SCALAR_BITS + SCALAR_MUL_WINDOW - 1) / SCALAR_MUL_WINDOW;
    
    for (int w = NUM_WINDOWS - 1; w >= 0; w--) {
        // Double window times
        for (int d = 0; d < SCALAR_MUL_WINDOW; d++) {
            g1_double(result, result);
        }
        
        // Extract window value
        int bit_offset = w * SCALAR_MUL_WINDOW;
        int limb_idx = bit_offset / 64;
        int bit_in_limb = bit_offset % 64;
        
        uint64_t window = scalar.limbs[limb_idx] >> bit_in_limb;
        if (bit_in_limb + SCALAR_MUL_WINDOW > 64 && limb_idx + 1 < Fr::LIMBS) {
            window |= (scalar.limbs[limb_idx + 1] << (64 - bit_in_limb));
        }
        window &= ((1ULL << SCALAR_MUL_WINDOW) - 1);
        
        int window_val = (int)window;
        
        // Add table[window_val] to result
        if (window_val != 0) {
            g1_add(result, result, table[window_val]);
        }
    }
    
    output[idx] = result;
}

// =============================================================================
// G2 Operations
// =============================================================================

/**
 * @brief G2 point doubling
 */
__device__ __forceinline__ void g2_double(G2Projective& r, const G2Projective& p) {
    if (p.is_identity()) {
        r = G2Projective::identity();
        return;
    }
    
    // a = X1^2
    Fq2 a = p.X * p.X;
    // b = Y1^2
    Fq2 b = p.Y * p.Y;
    // c = b^2
    Fq2 c = b * b;
    
    // d = 2 * ((X1 + b)^2 - a - c)
    Fq2 x_plus_b = p.X + b;
    Fq2 d = x_plus_b * x_plus_b - a - c;
    d = d + d;
    
    // e = 3 * a
    Fq2 e = a + a + a;
    
    // f = e^2
    Fq2 f = e * e;
    
    // Z3 = 2 * Y1 * Z1
    r.Z = p.Y * p.Z;
    r.Z = r.Z + r.Z;
    
    // X3 = f - 2 * d
    r.X = f - d - d;
    
    // Y3 = e * (d - X3) - 8 * c
    Fq2 c8 = c + c;
    c8 = c8 + c8;
    c8 = c8 + c8;
    r.Y = e * (d - r.X) - c8;
}

/**
 * @brief G2 point addition
 */
__device__ __forceinline__ void g2_add(G2Projective& r, const G2Projective& p, const G2Projective& q) {
    if (p.is_identity()) {
        r = q;
        return;
    }
    if (q.is_identity()) {
        r = p;
        return;
    }
    
    // z1z1 = Z1^2
    Fq2 z1z1 = p.Z * p.Z;
    // z2z2 = Z2^2
    Fq2 z2z2 = q.Z * q.Z;
    
    // u1 = X1 * Z2Z2
    Fq2 u1 = p.X * z2z2;
    // u2 = X2 * Z1Z1
    Fq2 u2 = q.X * z1z1;
    
    // s1 = Y1 * Z2 * Z2Z2
    Fq2 s1 = p.Y * q.Z * z2z2;
    // s2 = Y2 * Z1 * Z1Z1
    Fq2 s2 = q.Y * p.Z * z1z1;
    
    // h = u2 - u1
    Fq2 h = u2 - u1;
    // i = (2 * h)^2
    Fq2 i = h + h;
    i = i * i;
    
    // j = h * i
    Fq2 j = h * i;
    
    // rr = 2 * (s2 - s1)
    Fq2 rr = s2 - s1;
    rr = rr + rr;
    
    // v = u1 * i
    Fq2 v = u1 * i;
    
    // X3 = r^2 - j - 2 * v
    r.X = rr * rr - j - v - v;
    
    // Y3 = r * (v - X3) - 2 * s1 * j
    Fq2 s1j = s1 * j;
    r.Y = rr * (v - r.X) - s1j - s1j;
    
    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * h
    Fq2 z_sum = p.Z + q.Z;
    r.Z = (z_sum * z_sum - z1z1 - z2z2) * h;
}

/**
 * @brief G2 mixed addition (projective + affine)
 */
__device__ __forceinline__ void g2_add_mixed(G2Projective& r, const G2Projective& p, const G2Affine& q) {
    if (q.is_identity()) {
        r = p;
        return;
    }
    if (p.is_identity()) {
        r = G2Projective::from_affine(q);
        return;
    }
    
    // z1z1 = Z1^2
    Fq2 z1z1 = p.Z * p.Z;
    
    // u2 = X2 * Z1Z1
    Fq2 u2 = q.x * z1z1;
    
    // s2 = Y2 * Z1 * Z1Z1
    Fq2 s2 = q.y * p.Z * z1z1;
    
    // h = u2 - X1
    Fq2 h = u2 - p.X;
    // hh = h^2
    Fq2 hh = h * h;
    
    // i = 4 * hh
    Fq2 i = hh + hh;
    i = i + i;
    
    // j = h * i
    Fq2 j = h * i;
    
    // rr = 2 * (s2 - Y1)
    Fq2 rr = s2 - p.Y;
    rr = rr + rr;
    
    // v = X1 * i
    Fq2 v = p.X * i;
    
    // X3 = r^2 - j - 2 * v
    r.X = rr * rr - j - v - v;
    
    // Y3 = r * (v - X3) - 2 * Y1 * j
    Fq2 y1j = p.Y * j;
    r.Y = rr * (v - r.X) - y1j - y1j;
    
    // Z3 = (Z1 + h)^2 - Z1Z1 - hh
    Fq2 z_plus_h = p.Z + h;
    r.Z = z_plus_h * z_plus_h - z1z1 - hh;
}

/**
 * @brief Batch G2 operations
 */
__global__ void batch_affine_to_projective_g2_kernel(
    G2Projective* output,
    const G2Affine* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = G2Projective::from_affine(input[idx]);
}

/**
 * @brief Batch G2 projective to affine conversion
 */
__global__ void batch_projective_to_affine_g2_naive_kernel(
    G2Affine* output,
    const G2Projective* input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = input[idx].to_affine();
}

/**
 * @brief Batch G2 projective to affine using naive per-element inversion
 */
void batch_projective_to_affine_g2(
    G2Affine* d_output,
    const G2Projective* d_input,
    int size,
    cudaStream_t stream
) {
    // Optimal thread count for memory-bound point operations
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    batch_projective_to_affine_g2_naive_kernel<<<blocks, threads, 0, stream>>>(
        d_output, d_input, size
    );
}

} // namespace curve

// =============================================================================
// Constants for exported functions
// =============================================================================

namespace {
    // Optimal thread count for memory-bound point operations
    // 256 threads provides good occupancy on all GPU architectures (SM 5.0+)
    constexpr int POINT_OP_THREADS = 256;
    
    // Maximum batch size to prevent integer overflow and excessive memory usage
    // 2^26 = 64M points (reasonable upper limit for batch operations)
    constexpr int MAX_POINT_BATCH_SIZE = (1 << 26);
}

// =============================================================================
// Exported Symbols
// =============================================================================

extern "C" {

using namespace bls12_381;

/**
 * @brief Exported batch affine to projective for G1
 */
eIcicleError bls12_381_g1_affine_to_projective(
    const G1Affine* input,
    int size,
    const VecOpsConfig* config,
    G1Projective* output
) {
    // Input validation
    if (input == nullptr || output == nullptr || config == nullptr) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    if (size <= 0) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    if (size > MAX_POINT_BATCH_SIZE) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(config->stream);
    cudaError_t err;
    
    const G1Affine* d_input = input;
    G1Projective* d_output = output;
    
    bool need_alloc_input = !config->is_a_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    // Allocate and copy input if needed
    if (need_alloc_input) {
        err = cudaMalloc((void**)&d_input, size * sizeof(G1Affine));
        if (err != cudaSuccess) {
            return eIcicleError::ALLOCATION_FAILED;
        }
        
        err = cudaMemcpy((void*)d_input, input, size * sizeof(G1Affine), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree((void*)d_input);
            return eIcicleError::COPY_FAILED;
        }
    }
    
    // Allocate output if needed
    if (need_alloc_output) {
        err = cudaMalloc(&d_output, size * sizeof(G1Projective));
        if (err != cudaSuccess) {
            if (need_alloc_input) cudaFree((void*)d_input);
            return eIcicleError::ALLOCATION_FAILED;
        }
    }
    
    // Launch kernel with overflow-safe block calculation
    const int threads = POINT_OP_THREADS;
    const int64_t safe_blocks = ((int64_t)size + threads - 1) / threads;
    const int blocks = static_cast<int>(safe_blocks);
    
    curve::batch_affine_to_projective_g1_kernel<<<blocks, threads, 0, stream>>>(
        d_output, d_input, size
    );
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (need_alloc_output) cudaFree(d_output);
        if (need_alloc_input) cudaFree((void*)d_input);
        return eIcicleError::UNKNOWN_ERROR;
    }
    
    // Copy results back if needed
    if (need_alloc_output) {
        err = cudaMemcpy(output, d_output, size * sizeof(G1Projective), cudaMemcpyDeviceToHost);
        cudaFree(d_output);
        
        if (err != cudaSuccess) {
            if (need_alloc_input) cudaFree((void*)d_input);
            return eIcicleError::COPY_FAILED;
        }
    }
    
    if (need_alloc_input) {
        cudaFree((void*)d_input);
    }
    
    return eIcicleError::SUCCESS;
}

/**
 * @brief Exported batch projective to affine for G1
 */
eIcicleError bls12_381_g1_projective_to_affine(
    const G1Projective* input,
    int size,
    const VecOpsConfig* config,
    G1Affine* output
) {
    // Input validation
    if (input == nullptr || output == nullptr || config == nullptr) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    if (size <= 0) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    if (size > MAX_POINT_BATCH_SIZE) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(config->stream);
    cudaError_t err;
    
    const G1Projective* d_input = input;
    G1Affine* d_output = output;
    
    bool need_alloc_input = !config->is_a_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    // Allocate and copy input if needed
    if (need_alloc_input) {
        err = cudaMalloc((void**)&d_input, size * sizeof(G1Projective));
        if (err != cudaSuccess) {
            return eIcicleError::ALLOCATION_FAILED;
        }
        
        err = cudaMemcpy((void*)d_input, input, size * sizeof(G1Projective), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree((void*)d_input);
            return eIcicleError::COPY_FAILED;
        }
    }
    
    // Allocate output if needed
    if (need_alloc_output) {
        err = cudaMalloc(&d_output, size * sizeof(G1Affine));
        if (err != cudaSuccess) {
            if (need_alloc_input) cudaFree((void*)d_input);
            return eIcicleError::ALLOCATION_FAILED;
        }
    }
    
    curve::batch_projective_to_affine_g1(d_output, d_input, size, stream);
    
    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (need_alloc_output) cudaFree(d_output);
        if (need_alloc_input) cudaFree((void*)d_input);
        return eIcicleError::UNKNOWN_ERROR;
    }
    
    // Copy results back if needed
    if (need_alloc_output) {
        err = cudaMemcpy(output, d_output, size * sizeof(G1Affine), cudaMemcpyDeviceToHost);
        cudaFree(d_output);
        
        if (err != cudaSuccess) {
            if (need_alloc_input) cudaFree((void*)d_input);
            return eIcicleError::COPY_FAILED;
        }
    }
    
    if (need_alloc_input) {
        cudaFree((void*)d_input);
    }
    
    return eIcicleError::SUCCESS;
}

/**
 * @brief Exported batch projective to affine for G2
 */
eIcicleError bls12_381_g2_projective_to_affine(
    const G2Projective* input,
    int size,
    const VecOpsConfig* config,
    G2Affine* output
) {
    // Input validation
    if (input == nullptr || output == nullptr || config == nullptr) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    if (size <= 0) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    if (size > MAX_POINT_BATCH_SIZE) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(config->stream);
    cudaError_t err;
    
    const G2Projective* d_input = input;
    G2Affine* d_output = output;
    
    bool need_alloc_input = !config->is_a_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    // Allocate and copy input if needed
    if (need_alloc_input) {
        err = cudaMalloc((void**)&d_input, size * sizeof(G2Projective));
        if (err != cudaSuccess) {
            return eIcicleError::ALLOCATION_FAILED;
        }
        
        err = cudaMemcpy((void*)d_input, input, size * sizeof(G2Projective), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree((void*)d_input);
            return eIcicleError::COPY_FAILED;
        }
    }
    
    // Allocate output if needed
    if (need_alloc_output) {
        err = cudaMalloc(&d_output, size * sizeof(G2Affine));
        if (err != cudaSuccess) {
            if (need_alloc_input) cudaFree((void*)d_input);
            return eIcicleError::ALLOCATION_FAILED;
        }
    }
    
    curve::batch_projective_to_affine_g2(d_output, d_input, size, stream);
    
    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (need_alloc_output) cudaFree(d_output);
        if (need_alloc_input) cudaFree((void*)d_input);
        return eIcicleError::UNKNOWN_ERROR;
    }
    
    // Copy results back if needed
    if (need_alloc_output) {
        err = cudaMemcpy(output, d_output, size * sizeof(G2Affine), cudaMemcpyDeviceToHost);
        cudaFree(d_output);
        
        if (err != cudaSuccess) {
            if (need_alloc_input) cudaFree((void*)d_input);
            return eIcicleError::COPY_FAILED;
        }
    }
    
    if (need_alloc_input) {
        cudaFree((void*)d_input);
    }
    
    return eIcicleError::SUCCESS;
}

} // extern "C"
