/**
 * @file ntt.cuh
 * @brief Number Theoretic Transform (NTT) - Header
 * 
 * This header provides:
 * - NTT Domain class for managing precomputed twiddle factors
 * - API declarations for NTT functions (implementations in field_backend.cu)
 * - Coset NTT helper kernels (template kernels used by coset operations)
 * 
 * ARCHITECTURE NOTE:
 * ==================
 * CUDA static libraries require kernels to be defined in the same compilation
 * unit that calls them. Therefore:
 * 
 * - Core NTT kernels (bit-reverse, butterfly, scale) are defined in field_backend.cu
 * - Only coset helper kernels are templates here (they get instantiated in field_backend.cu)
 * - DO NOT add __global__ kernel implementations here unless they are templates
 *   that will be instantiated by each .cu file that uses them
 * 
 * Algorithm: Cooley-Tukey decimation-in-time with optimizations:
 * - Shared memory kernel for small sizes (≤ 1024)
 * - Fused 2-stage butterfly for larger sizes (50% fewer kernel launches)
 */

#pragma once

#include "field.cuh"
#include "icicle_types.cuh"
#include <cuda_runtime.h>
#include <mutex>

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
    
    // Global domain registry - declared extern, defined in field_backend.cu
    static Domain* domains[MAX_LOG_DOMAIN_SIZE];
    static std::mutex domains_mutex;
    
    __host__ Domain() : twiddles(nullptr), inv_twiddles(nullptr), 
                        log_size(0), size(0) {}
    
    __host__ ~Domain() {
        if (twiddles) cudaFree(twiddles);
        if (inv_twiddles) cudaFree(inv_twiddles);
    }
    
    /**
     * @brief Get domain for given log size
     */
    static Domain* get_domain(int log_size);
    
    /**
     * @brief Set domain for given log size
     */
    static void set_domain(int log_size, Domain* domain);
};

// =============================================================================
// Coset NTT Helper Kernels (Template - instantiated in field_backend.cu)
// =============================================================================
// NOTE: Core NTT kernels (bit-reverse, butterfly, scale, fused) are defined
// directly in field_backend.cu to ensure proper CUDA device code linking.
// Only these coset helpers need to be templates since coset_ntt_cuda uses them.

/**
 * @brief Multiply by coset generator powers: output[i] = input[i] * g^i
 * 
 * Used for coset FFT: evaluates polynomial at g*omega^i instead of omega^i
 */
template<typename F>
__global__ void coset_mul_kernel(
    F* output,
    const F* input,
    F coset_gen,      // The coset generator g
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Compute g^idx by repeated squaring
    F power = F::one();
    F base = coset_gen;
    int exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            power = power * base;
        }
        base = base * base;
        exp >>= 1;
    }
    
    output[idx] = input[idx] * power;
}

/**
 * @brief Divide by coset generator powers: output[i] = input[i] * g^(-i)
 * 
 * Used after inverse coset FFT to convert back from coset domain
 */
template<typename F>
__global__ void coset_div_kernel(
    F* output,
    const F* input,
    F coset_gen_inv,  // The inverse of coset generator: g^(-1)
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Compute g^(-idx) = (g^(-1))^idx
    F power = F::one();
    F base = coset_gen_inv;
    int exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            power = power * base;
        }
        base = base * base;
        exp >>= 1;
    }
    
    output[idx] = input[idx] * power;
}

// =============================================================================
// NTT API Declarations (Implementations in field_backend.cu)
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
 * @brief Coset NTT: evaluates at g*omega^i
 * 
 * Multiplies input by coset powers, then applies regular NTT
 */
template<typename F>
eIcicleError coset_ntt_cuda(
    const F* input,
    int size,
    NTTDir direction,
    const F& coset_gen,
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
