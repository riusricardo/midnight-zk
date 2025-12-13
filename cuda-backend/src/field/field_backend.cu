/**
 * @file field_backend.cu
 * @brief Field backend exports for ICICLE compatibility
 * 
 */

#include "field.cuh"
#include "ntt.cuh"
#include "icicle_types.cuh"
#include <cuda_runtime.h>

using namespace bls12_381;

// =============================================================================
// Global NTT Domain Registry
// =============================================================================

// Instantiate domain registry for Fr (scalar field)
template<>
ntt::Domain<Fr>* ntt::Domain<Fr>::domains[MAX_LOG_DOMAIN_SIZE] = {nullptr};

// =============================================================================
// NTT Core Implementation
// =============================================================================

namespace ntt {

/**
 * @brief Forward NTT implementation
 */
template<typename F>
eIcicleError ntt_forward_impl(
    const F* input,
    int size,
    const NTTConfig& config,
    F* output
) {
    if (size == 0) return eIcicleError::SUCCESS;
    if ((size & (size - 1)) != 0) return eIcicleError::INVALID_ARGUMENT;
    
    int log_size = 0;
    while ((1 << log_size) < size) log_size++;
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    // Get domain
    Domain<F>* domain = Domain<F>::get_domain(log_size);
    if (!domain) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Allocate working buffer
    F* d_data = nullptr;
    bool need_alloc = !config.are_inputs_on_device;
    
    if (need_alloc) {
        cudaMalloc(&d_data, size * sizeof(F));
        cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyHostToDevice);
    } else if (input != output) {
        cudaMalloc(&d_data, size * sizeof(F));
        cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyDeviceToDevice);
    } else {
        d_data = const_cast<F*>(input);
    }
    
    // Bit reversal
    F* d_temp;
    cudaMalloc(&d_temp, size * sizeof(F));
    
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    bit_reverse_kernel<<<blocks, threads, 0, stream>>>(d_temp, d_data, size, log_size);
    cudaMemcpy(d_data, d_temp, size * sizeof(F), cudaMemcpyDeviceToDevice);
    cudaFree(d_temp);
    
    // Butterfly stages
    for (int s = 1; s <= log_size; s++) {
        int m = 1 << s;
        int half_m = m / 2;
        
        blocks = (size / 2 + threads - 1) / threads;
        butterfly_kernel<<<blocks, threads, 0, stream>>>(
            d_data, domain->twiddles, size, m, half_m
        );
        cudaStreamSynchronize(stream);
    }
    
    // Copy output
    if (config.are_outputs_on_device) {
        if (d_data != output) {
            cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToDevice);
        }
    } else {
        cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToHost);
    }
    
    if (need_alloc || (input != output && config.are_inputs_on_device)) {
        if (d_data != input) cudaFree(d_data);
    }
    
    return eIcicleError::SUCCESS;
}

/**
 * @brief Inverse NTT implementation
 */
template<typename F>
eIcicleError ntt_inverse_impl(
    const F* input,
    int size,
    const NTTConfig& config,
    F* output
) {
    if (size == 0) return eIcicleError::SUCCESS;
    if ((size & (size - 1)) != 0) return eIcicleError::INVALID_ARGUMENT;
    
    int log_size = 0;
    while ((1 << log_size) < size) log_size++;
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    Domain<F>* domain = Domain<F>::get_domain(log_size);
    if (!domain) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Allocate working buffer
    F* d_data = nullptr;
    bool need_alloc = !config.are_inputs_on_device;
    
    if (need_alloc) {
        cudaMalloc(&d_data, size * sizeof(F));
        cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyHostToDevice);
    } else if (input != output) {
        cudaMalloc(&d_data, size * sizeof(F));
        cudaMemcpy(d_data, input, size * sizeof(F), cudaMemcpyDeviceToDevice);
    } else {
        d_data = const_cast<F*>(input);
    }
    
    const int threads = 256;
    
    // Butterfly stages with inverse twiddles
    for (int s = log_size; s >= 1; s--) {
        int m = 1 << s;
        int half_m = m / 2;
        
        int blocks = (size / 2 + threads - 1) / threads;
        butterfly_kernel<<<blocks, threads, 0, stream>>>(
            d_data, domain->inv_twiddles, size, m, half_m
        );
        cudaStreamSynchronize(stream);
    }
    
    // Bit reversal
    F* d_temp;
    cudaMalloc(&d_temp, size * sizeof(F));
    
    int blocks = (size + threads - 1) / threads;
    bit_reverse_kernel<<<blocks, threads, 0, stream>>>(d_temp, d_data, size, log_size);
    
    // Scale by 1/n
    scale_kernel<<<blocks, threads, 0, stream>>>(
        d_data, d_temp, domain->domain_size_inv, size
    );
    
    cudaFree(d_temp);
    
    // Copy output
    if (config.are_outputs_on_device) {
        if (d_data != output) {
            cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToDevice);
        }
    } else {
        cudaMemcpy(output, d_data, size * sizeof(F), cudaMemcpyDeviceToHost);
    }
    
    if (need_alloc || (input != output && config.are_inputs_on_device)) {
        if (d_data != input) cudaFree(d_data);
    }
    
    return eIcicleError::SUCCESS;
}

/**
 * @brief Main NTT entry point
 */
template<typename F>
eIcicleError ntt_cuda_impl(
    const F* input,
    int size,
    NTTDir direction,
    const NTTConfig& config,
    F* output
) {
    if (direction == NTTDir::kForward) {
        return ntt_forward_impl(input, size, config, output);
    } else {
        return ntt_inverse_impl(input, size, config, output);
    }
}

// Explicit instantiations
template eIcicleError ntt_cuda_impl<Fr>(const Fr*, int, NTTDir, const NTTConfig&, Fr*);

} // namespace ntt

// =============================================================================
// NTT Domain Management
// =============================================================================

namespace ntt {

// Kernel to compute twiddle factors on GPU
__global__ void compute_twiddles_kernel(
    Fr* twiddles,
    Fr* inv_twiddles,
    Fr omega,
    Fr omega_inv,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Compute omega^idx using repeated squaring
    Fr pow = Fr::one();
    Fr base = omega;
    int exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            pow = pow * base;
        }
        base = base * base;
        exp >>= 1;
    }
    twiddles[idx] = pow;
    
    // Compute omega_inv^idx
    pow = Fr::one();
    base = omega_inv;
    exp = idx;
    
    while (exp > 0) {
        if (exp & 1) {
            pow = pow * base;
        }
        base = base * base;
        exp >>= 1;
    }
    inv_twiddles[idx] = pow;
}

// Kernel to compute omega^(2^k)
__global__ void square_field_kernel(Fr* result, const Fr* input, int iterations) {
    Fr val = *input;
    for (int i = 0; i < iterations; i++) {
        val = val * val;
    }
    *result = val;
}

// Kernel to compute inverse and size_inv
__global__ void compute_inv_kernel(Fr* omega_inv, Fr* size_inv, const Fr* omega, int size) {
    Fr inv;
    field_inv(inv, *omega);
    *omega_inv = inv;
    
    Fr size_field = Fr::from_int((uint64_t)size);
    field_inv(*size_inv, size_field);
}

/**
 * @brief Initialize NTT domain with given root of unity
 */
template<typename F>
eIcicleError init_domain_cuda_impl(
    const F& root_of_unity,
    const NTTInitDomainConfig& config
) {
    // Determine max log size from root of unity order
    // For BLS12-381 Fr, max is 2^32 (root has order 2^32)
    int max_log_size = 32;  // Default for BLS12-381
    
    if (config.max_log_size > 0 && config.max_log_size < max_log_size) {
        max_log_size = config.max_log_size;
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    // Copy root_of_unity to device
    F* d_root;
    cudaMalloc(&d_root, sizeof(F));
    cudaMemcpy(d_root, &root_of_unity, sizeof(F), cudaMemcpyHostToDevice);
    
    // Precompute twiddle factors for each domain size
    for (int log_size = 1; log_size <= max_log_size && log_size < MAX_LOG_DOMAIN_SIZE; log_size++) {
        int size = 1 << log_size;
        
        // Allocate domain
        Domain<F>* domain = new Domain<F>();
        domain->log_size = log_size;
        domain->size = size;
        
        // Allocate twiddle factor arrays
        cudaMalloc(&domain->twiddles, size * sizeof(F));
        cudaMalloc(&domain->inv_twiddles, size * sizeof(F));
        
        // Compute omega = root_of_unity^(2^(max_log - log_size)) on GPU
        F* d_omega;
        F* d_omega_inv;
        F* d_size_inv;
        cudaMalloc(&d_omega, sizeof(F));
        cudaMalloc(&d_omega_inv, sizeof(F));
        cudaMalloc(&d_size_inv, sizeof(F));
        
        int squarings = max_log_size - log_size;
        square_field_kernel<<<1, 1, 0, stream>>>(d_omega, d_root, squarings);
        cudaStreamSynchronize(stream);
        
        // Compute inverse and size_inv
        compute_inv_kernel<<<1, 1, 0, stream>>>(d_omega_inv, d_size_inv, d_omega, size);
        cudaStreamSynchronize(stream);
        
        // Copy omega values back to host to pass to twiddle kernel
        F h_omega, h_omega_inv;
        cudaMemcpy(&h_omega, d_omega, sizeof(F), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_omega_inv, d_omega_inv, sizeof(F), cudaMemcpyDeviceToHost);
        cudaMemcpy(&domain->domain_size_inv, d_size_inv, sizeof(F), cudaMemcpyDeviceToHost);
        
        // Compute all twiddles in parallel on GPU
        const int threads = 256;
        const int blocks = (size + threads - 1) / threads;
        compute_twiddles_kernel<<<blocks, threads, 0, stream>>>(
            domain->twiddles, domain->inv_twiddles, h_omega, h_omega_inv, size
        );
        cudaStreamSynchronize(stream);
        
        cudaFree(d_omega);
        cudaFree(d_omega_inv);
        cudaFree(d_size_inv);
        
        // Register domain
        Domain<F>::domains[log_size] = domain;
    }
    
    cudaFree(d_root);
    
    return eIcicleError::SUCCESS;
}

template eIcicleError init_domain_cuda_impl<Fr>(const Fr&, const NTTInitDomainConfig&);

/**
 * @brief Release NTT domain resources
 */
template<typename F>
eIcicleError release_domain_cuda_impl() {
    for (int i = 0; i < MAX_LOG_DOMAIN_SIZE; i++) {
        Domain<F>* domain = Domain<F>::domains[i];
        if (domain) {
            if (domain->twiddles) cudaFree(domain->twiddles);
            if (domain->inv_twiddles) cudaFree(domain->inv_twiddles);
            delete domain;
            Domain<F>::domains[i] = nullptr;
        }
    }
    return eIcicleError::SUCCESS;
}

template eIcicleError release_domain_cuda_impl<Fr>();

} // namespace ntt

// =============================================================================
// C++ Mangled Symbol Exports (matching ICICLE)
// =============================================================================

namespace ntt {

// These match the mangled symbols from the library
template<typename F>
eIcicleError ntt_cuda(
    const F* input,
    int size,
    NTTDir direction,
    const NTTConfig& config,
    F* output
) {
    return ntt_cuda_impl<F>(input, size, direction, config, output);
}

template<typename F>
eIcicleError init_domain_cuda(
    const F& root_of_unity,
    const NTTInitDomainConfig& config
) {
    return init_domain_cuda_impl<F>(root_of_unity, config);
}

template<typename F>
eIcicleError release_domain_cuda() {
    return release_domain_cuda_impl<F>();
}

// Explicit instantiations for symbol export
template eIcicleError ntt_cuda<Fr>(const Fr*, int, NTTDir, const NTTConfig&, Fr*);
template eIcicleError init_domain_cuda<Fr>(const Fr&, const NTTInitDomainConfig&);
template eIcicleError release_domain_cuda<Fr>();

} // namespace ntt
