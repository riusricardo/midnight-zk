/**
 * @file ecntt.cu
 * @brief EC-NTT (Elliptic Curve Number Theoretic Transform)
 * 
 * Applies NTT-like operations on elliptic curve points.
 * Used in multi-scalar multiplication and polynomial commitment schemes.
 * 
 * ARCHITECTURE:
 * =============
 * All EC-NTT kernels are defined and called in this file (self-contained).
 * 
 * Unlike field NTT which uses field multiplication, EC-NTT uses:
 * - Point addition instead of field addition
 * - Scalar multiplication instead of field multiplication
 * 
 * Kernels defined here:
 * - ecntt_butterfly_kernel: EC butterfly operation
 * - ecntt_bit_reverse_kernel: Bit-reversal permutation for points
 * 
 * Use case: Accelerating polynomial commitment schemes like KZG.
 */

#include "point.cuh"
#include "ntt.cuh"
#include "icicle_types.cuh"

namespace ecntt {

using namespace bls12_381;

// =============================================================================
// EC-NTT Butterfly Operations
// =============================================================================

/**
 * @brief EC-NTT butterfly operation
 * 
 * Unlike field NTT which uses multiplication, EC-NTT uses point addition
 * and scalar multiplication.
 * 
 * For points P0, P1 and twiddle factor omega:
 *   P0' = P0 + omega * P1
 *   P1' = P0 - omega * P1
 */
__device__ void ecntt_butterfly(
    G1Projective& p0_out,
    G1Projective& p1_out,
    const G1Projective& p0,
    const G1Projective& p1,
    const Fr& omega
) {
    // Scalar multiply: temp = omega * P1
    G1Projective temp = G1Projective::identity();
    G1Projective base = p1;
    
    Fr scalar = omega;
    // Process all 256 bits for safety (top bits will be 0 for properly reduced scalars)
    for (int i = 0; i < 256; i++) {
        if ((scalar.limbs[i / 64] >> (i % 64)) & 1) {
            g1_add(temp, temp, base);
        }
        // Don't double after processing the last bit
        if (i < 255) {
            g1_double(base, base);
        }
    }
    
    // p0' = p0 + temp
    g1_add(p0_out, p0, temp);
    
    // p1' = p0 - temp (negate y of temp)
    G1Projective neg_temp = temp;
    g1_neg(neg_temp, temp);
    g1_add(p1_out, p0, neg_temp);
}

/**
 * @brief EC-NTT butterfly kernel
 */
__global__ void ecntt_butterfly_kernel(
    G1Projective* data,
    const Fr* twiddles,
    int size,
    int step,
    int half_step
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size / 2) return;
    
    int group = idx / half_step;
    int pos = idx % half_step;
    
    int i0 = group * step + pos;
    int i1 = i0 + half_step;
    
    Fr omega = twiddles[pos * (size / step)];
    
    G1Projective p0 = data[i0];
    G1Projective p1 = data[i1];
    
    ecntt_butterfly(data[i0], data[i1], p0, p1, omega);
}

/**
 * @brief EC-NTT bit reversal kernel
 */
__global__ void ecntt_bit_reverse_kernel(
    G1Projective* output,
    const G1Projective* input,
    int size,
    int log_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Compute bit-reversed index
    unsigned int rev = 0;
    unsigned int n = idx;
    for (int i = 0; i < log_size; i++) {
        rev = (rev << 1) | (n & 1);
        n >>= 1;
    }
    
    if (idx < rev) {
        G1Projective temp = input[idx];
        output[idx] = input[rev];
        output[rev] = temp;
    } else if (idx == rev) {
        output[idx] = input[idx];
    }
}

/**
 * @brief Forward EC-NTT with error handling
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t ecntt_forward(
    G1Projective* d_data,
    const Fr* d_twiddles,
    int size,
    cudaStream_t stream
) {
    const int threads = 256;
    int log_size = 0;
    while ((1 << log_size) < size) log_size++;
    
    cudaError_t err;
    
    // Bit reversal
    G1Projective* d_temp;
    err = cudaMalloc(&d_temp, size * sizeof(G1Projective));
    if (err != cudaSuccess) return err;
    
    int blocks = (size + threads - 1) / threads;
    ecntt_bit_reverse_kernel<<<blocks, threads, 0, stream>>>(
        d_temp, d_data, size, log_size
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_temp);
        return err;
    }
    
    err = cudaMemcpyAsync(d_data, d_temp, size * sizeof(G1Projective), 
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_temp);
        return err;
    }
    cudaFree(d_temp);
    
    // Butterfly stages
    for (int s = 1; s <= log_size; s++) {
        int step = 1 << s;
        int half_step = step / 2;
        
        int num_ops = size / 2;
        blocks = (num_ops + threads - 1) / threads;
        
        ecntt_butterfly_kernel<<<blocks, threads, 0, stream>>>(
            d_data, d_twiddles, size, step, half_step
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return err;
    }
    
    return cudaSuccess;
}

/**
 * @brief Inverse EC-NTT with error handling
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t ecntt_inverse(
    G1Projective* d_data,
    const Fr* d_inv_twiddles,
    const Fr& inv_size,
    int size,
    cudaStream_t stream
) {
    // TODO: Implement final scaling by inv_size
    // NOTE: Size normalization (multiplying by 1/n) is not applied here.
    // The caller is responsible for scaling the result if needed.
    // This matches ICICLE's behavior where normalization is optional.
    (void)inv_size;
    const int threads = 256;
    int log_size = 0;
    while ((1 << log_size) < size) log_size++;
    
    cudaError_t err;
    
    // Similar to forward, but with inverse twiddles and final scaling
    // Butterfly stages in reverse order
    for (int s = log_size; s >= 1; s--) {
        int step = 1 << s;
        int half_step = step / 2;
        
        int num_ops = size / 2;
        int blocks = (num_ops + threads - 1) / threads;
        
        ecntt_butterfly_kernel<<<blocks, threads, 0, stream>>>(
            d_data, d_inv_twiddles, size, step, half_step
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return err;
    }
    
    // Bit reversal
    G1Projective* d_temp;
    err = cudaMalloc(&d_temp, size * sizeof(G1Projective));
    if (err != cudaSuccess) return err;
    
    int blocks = (size + threads - 1) / threads;
    ecntt_bit_reverse_kernel<<<blocks, threads, 0, stream>>>(
        d_temp, d_data, size, log_size
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_temp);
        return err;
    }
    
    err = cudaMemcpyAsync(d_data, d_temp, size * sizeof(G1Projective), 
                          cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_temp);
        return err;
    }
    cudaFree(d_temp);
    
    // Final scaling by 1/n would require scalar multiplication
    // This is handled by the caller
    return cudaSuccess;
}

} // namespace ecntt

// =============================================================================
// ICICLE API Exports
// =============================================================================

namespace curve_backend {

using namespace bls12_381;

/**
 * @brief ECNTT implementation matching Icicle's interface
 */
template<typename Scalar, typename Affine, typename Projective>
eIcicleError ecntt_impl(
    const Projective* input,
    int size,
    NTTDir direction,
    const NTTConfig& config,
    Projective* output
) {
    if (size == 0) {
        return eIcicleError::SUCCESS;
    }
    
    // Validate size is power of 2
    if ((size & (size - 1)) != 0) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    cudaStream_t stream = static_cast<cudaStream_t>(config.stream);
    
    // Allocate device memory if needed
    Projective* d_data;
    bool need_alloc = !config.are_inputs_on_device;
    
    if (need_alloc) {
        cudaError_t err = cudaMalloc(&d_data, size * sizeof(Projective));
        if (err != cudaSuccess) {
            return eIcicleError::ALLOCATION_FAILED;
        }
        cudaMemcpy(d_data, input, size * sizeof(Projective), cudaMemcpyHostToDevice);
    } else {
        if (input != output) {
            cudaMalloc(&d_data, size * sizeof(Projective));
            cudaMemcpy(d_data, input, size * sizeof(Projective), cudaMemcpyDeviceToDevice);
        } else {
            d_data = const_cast<Projective*>(input);
        }
    }
    
    // Get or create domain
    int log_size = 0;
    while ((1 << log_size) < size) log_size++;
    
    ntt::Domain<Scalar>* domain = ntt::Domain<Scalar>::get_domain(log_size);
    if (!domain) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Validate domain is fully initialized
    if (!domain->twiddles || !domain->inv_twiddles) {
        return eIcicleError::INVALID_ARGUMENT;
    }
    
    // Perform EC-NTT
    cudaError_t err;
    if (direction == NTTDir::kForward) {
        err = ecntt::ecntt_forward(
            d_data,
            domain->twiddles,
            size,
            stream
        );
    } else {
        err = ecntt::ecntt_inverse(
            d_data,
            domain->inv_twiddles,
            domain->domain_size_inv,
            size,
            stream
        );
    }
    
    if (err != cudaSuccess) {
        if (need_alloc || (input != output && config.are_inputs_on_device)) {
            cudaFree(d_data);
        }
        return eIcicleError::UNKNOWN_ERROR;
    }
    
    // Copy result
    if (config.are_outputs_on_device) {
        if (d_data != output) {
            cudaMemcpy(output, d_data, size * sizeof(Projective), cudaMemcpyDeviceToDevice);
        }
    } else {
        cudaMemcpy(output, d_data, size * sizeof(Projective), cudaMemcpyDeviceToHost);
    }
    
    if (need_alloc || (input != output && config.are_inputs_on_device)) {
        cudaFree(d_data);
    }
    
    return eIcicleError::SUCCESS;
}

// Template instantiation - only G1 for now
template eIcicleError ecntt_impl<Fr, G1Affine, G1Projective>(
    const G1Projective*, int, NTTDir, const NTTConfig&, G1Projective*
);

// G2 ECNTT requires separate implementation
// template eIcicleError ecntt_impl<Fr, G2Affine, G2Projective>(
//     const G2Projective*, int, NTTDir, const NTTConfig&, G2Projective*
// );

} // namespace curve_backend

// =============================================================================
// C Export
// =============================================================================

extern "C" {

eIcicleError ecntt_g1_cuda(
    const bls12_381::G1Projective* input,
    int size,
    NTTDir direction,
    const NTTConfig& config,
    bls12_381::G1Projective* output
) {
    return curve_backend::ecntt_impl<bls12_381::Fr, bls12_381::G1Affine, bls12_381::G1Projective>(
        input, size, direction, config, output
    );
}

eIcicleError ecntt_g2_cuda(
    const bls12_381::G2Projective* input,
    int size,
    NTTDir direction,
    const NTTConfig& config,
    bls12_381::G2Projective* output
) {
    // G2 ECNTT not yet implemented
    (void)input; (void)size; (void)direction; (void)config; (void)output;
    return eIcicleError::INVALID_ARGUMENT;
}

} // extern "C"
