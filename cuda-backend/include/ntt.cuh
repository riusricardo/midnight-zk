/**
 * @file ntt.cuh
 * @brief Number Theoretic Transform (NTT) implementation
 * 
 * Implements forward and inverse NTT over the BLS12-381 scalar field.
 * Uses Cooley-Tukey decimation-in-time algorithm.
 */

#pragma once

#include "field.cuh"
#include "icicle_types.cuh"
#include <cuda_runtime.h>

namespace ntt {

using namespace bls12_381;

// Maximum log domain size supported
constexpr int MAX_LOG_DOMAIN_SIZE = 33;

// =============================================================================
// NTT Domain Management
// =============================================================================

/**
 * @brief NTT Domain containing precomputed twiddle factors
 */
template<typename F>
class Domain {
public:
    F* twiddles;              // Twiddle factors (powers of omega)
    F* inv_twiddles;          // Inverse twiddle factors
    F domain_size_inv;        // 1/n for this domain
    int log_size;
    size_t size;
    
    // Global domain registry
    static Domain* domains[MAX_LOG_DOMAIN_SIZE];
    
    __host__ Domain() : twiddles(nullptr), inv_twiddles(nullptr), 
                        log_size(0), size(0) {}
    
    __host__ ~Domain() {
        if (twiddles) cudaFree(twiddles);
        if (inv_twiddles) cudaFree(inv_twiddles);
    }
    
    /**
     * @brief Get domain for given log size
     */
    static Domain* get_domain(int log_size) {
        if (log_size >= MAX_LOG_DOMAIN_SIZE) return nullptr;
        return domains[log_size];
    }
};

// =============================================================================
// NTT Kernels
// =============================================================================

/**
 * @brief Bit-reverse permutation kernel
 */
template<typename F>
__global__ void bit_reverse_kernel(
    F* output,
    const F* input,
    int n,
    int log_n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    // Compute bit-reversed index
    unsigned int rev = 0;
    unsigned int temp = idx;
    for (int i = 0; i < log_n; i++) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }
    
    output[rev] = input[idx];
}

/**
 * @brief Single butterfly stage kernel
 * 
 * Performs one stage of the Cooley-Tukey NTT butterfly.
 */
template<typename F>
__global__ void butterfly_kernel(
    F* data,
    const F* twiddles,
    int n,
    int m,           // Group size = 2^(stage+1)
    int half_m       // Half group size = 2^stage
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_butterflies = n / 2;
    
    if (idx >= num_butterflies) return;
    
    // Determine which butterfly group and position within group
    int group = idx / half_m;
    int pos = idx % half_m;
    
    // Indices of the two elements in the butterfly
    int i = group * m + pos;
    int j = i + half_m;
    
    // Twiddle factor index
    int twiddle_idx = pos * (n / m);
    
    F twiddle = twiddles[twiddle_idx];
    
    // Butterfly operation
    F u = data[i];
    F v = data[j] * twiddle;
    
    data[i] = u + v;
    data[j] = u - v;
}

/**
 * @brief Scale by n^{-1} for inverse NTT
 */
template<typename F>
__global__ void scale_kernel(
    F* output,
    const F* input,
    F scale_factor,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    output[idx] = input[idx] * scale_factor;
}

// =============================================================================
// Main NTT Entry Point
// =============================================================================

/**
 * @brief NTT using Cooley-Tukey algorithm
 * 
 * This matches the Icicle ntt_cuda signature.
 * Implementation is in field_backend.cu
 */
template<typename F>
eIcicleError ntt_cuda(
    const F* input,
    int size,
    NTTDir direction,
    const NTTConfig& config,
    F* output
);

/**
 * @brief Initialize NTT domain with root of unity
 */
template<typename F>
eIcicleError init_domain_cuda(
    const F& root_of_unity,
    const NTTInitDomainConfig& config
);

/**
 * @brief Release NTT domain resources
 */
template<typename F>
eIcicleError release_domain_cuda();

} // namespace ntt
