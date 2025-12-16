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

/**
 * @brief Batch scalar multiplication
 */
__global__ void batch_scalar_mul_g1_kernel(
    G1Projective* output,
    const G1Affine* bases,
    const Fr* scalars,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    G1Projective result = G1Projective::identity();
    G1Projective base = G1Projective::from_affine(bases[idx]);
    Fr scalar = scalars[idx];
    
    // Double-and-add: process all 256 bits for safety
    for (int i = 0; i < 256; i++) {
        if ((scalar.limbs[i / 64] >> (i % 64)) & 1) {
            g1_add(result, result, base);
        }
        // Don't double after processing the last bit
        if (i < 255) {
            g1_double(base, base);
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

} // namespace curve

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
    cudaStream_t stream = static_cast<cudaStream_t>(config->stream);
    
    const G1Affine* d_input = input;
    G1Projective* d_output = output;
    
    bool need_alloc_input = !config->is_a_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    if (need_alloc_input) {
        cudaMalloc((void**)&d_input, size * sizeof(G1Affine));
        cudaMemcpy((void*)d_input, input, size * sizeof(G1Affine), cudaMemcpyHostToDevice);
    }
    if (need_alloc_output) {
        cudaMalloc(&d_output, size * sizeof(G1Projective));
    }
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    curve::batch_affine_to_projective_g1_kernel<<<blocks, threads, 0, stream>>>(
        d_output, d_input, size
    );
    
    if (need_alloc_output) {
        cudaMemcpy(output, d_output, size * sizeof(G1Projective), cudaMemcpyDeviceToHost);
        cudaFree(d_output);
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
    cudaStream_t stream = static_cast<cudaStream_t>(config->stream);
    
    const G1Projective* d_input = input;
    G1Affine* d_output = output;
    
    bool need_alloc_input = !config->is_a_on_device;
    bool need_alloc_output = !config->is_result_on_device;
    
    if (need_alloc_input) {
        cudaMalloc((void**)&d_input, size * sizeof(G1Projective));
        cudaMemcpy((void*)d_input, input, size * sizeof(G1Projective), cudaMemcpyHostToDevice);
    }
    if (need_alloc_output) {
        cudaMalloc(&d_output, size * sizeof(G1Affine));
    }
    
    curve::batch_projective_to_affine_g1(d_output, d_input, size, stream);
    
    if (need_alloc_output) {
        cudaMemcpy(output, d_output, size * sizeof(G1Affine), cudaMemcpyDeviceToHost);
        cudaFree(d_output);
    }
    if (need_alloc_input) {
        cudaFree((void*)d_input);
    }
    
    return eIcicleError::SUCCESS;
}

} // extern "C"
