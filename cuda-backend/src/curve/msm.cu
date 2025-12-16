/**
 * @file msm.cu
 * @brief Multi-Scalar Multiplication (MSM) - Template Instantiation
 * 
 * This file provides template instantiation for MSM and non-template helpers.
 * 
 * ARCHITECTURE:
 * =============
 * MSM kernels are defined in msm.cuh as templates. This is an exception to the
 * "kernels in same .cu file" rule because:
 * 
 * 1. MSM kernels are templates parameterized by curve type (G1, G2)
 * 2. Template kernels in headers get instantiated in each including .cu file
 * 3. This file explicitly instantiates the G1 version
 * 
 * The msm.cuh header contains:
 * - compute_bucket_indices_kernel: Decompose scalars into bucket indices
 * - accumulate_sorted_kernel: Sort-Reduce accumulation (constant-time, secure)
 * - parallel_bucket_reduction_kernel: Reduce bucket sums
 * - final_accumulation_kernel: Combine window results
 * 
 * SECURITY:
 * =========
 * MSM uses Sort-Reduce pattern for constant-time bucket accumulation.
 * This prevents timing side-channels that could leak scalar information.
 */

#include "msm.cuh"

// All MSM implementation is in the header for template reasons
// This file provides template instantiation and non-template helpers

namespace msm {

using namespace bls12_381;
using namespace icicle;

// Optimal C lookup table (precomputed from benchmarks)
static const int optimal_c_table[] = {
    // log2(size): optimal_c
    1,   // 2^0
    2,   // 2^1
    3,   // 2^2
    4,   // 2^3
    5,   // 2^4
    6,   // 2^5
    7,   // 2^6
    8,   // 2^7
    8,   // 2^8
    9,   // 2^9
    10,  // 2^10
    10,  // 2^11
    11,  // 2^12
    11,  // 2^13
    12,  // 2^14
    12,  // 2^15
    13,  // 2^16
    13,  // 2^17
    14,  // 2^18
    14,  // 2^19
    15,  // 2^20
    15,  // 2^21
    16,  // 2^22
    16,  // 2^23
};

int get_optimal_c_from_table(int msm_size) {
    if (msm_size <= 1) return 1;
    
    int log_size = 0;
    int temp = msm_size;
    while (temp > 1) {
        temp >>= 1;
        log_size++;
    }
    
    if (log_size >= 24) return 16;
    return optimal_c_table[log_size];
}

// Template instantiation for G1
template cudaError_t msm_cuda<Fr, G1Affine, G1Projective>(
    const Fr* scalars,
    const G1Affine* bases,
    int msm_size,
    const MSMConfig& config,
    G1Projective* result
);

} // namespace msm

// =============================================================================
// Icicle-Compatible C API Exports
// =============================================================================

extern "C" {

eIcicleError bls12_381_msm_cuda(
    const bls12_381::Fr* scalars,
    const bls12_381::G1Affine* bases,
    int msm_size,
    const MSMConfig* config,
    bls12_381::G1Projective* result)
{
    cudaError_t err = msm::msm_cuda<bls12_381::Fr, bls12_381::G1Affine, bls12_381::G1Projective>(
        scalars, bases, msm_size, *config, result
    );
    return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::UNKNOWN_ERROR;
}

eIcicleError bls12_381_msm_precompute_bases_cuda(
    const bls12_381::G1Affine* input_bases,
    int bases_size,
    const MSMConfig* config,
    bls12_381::G1Affine* output_bases)
{
    // Our MSM doesn't require precomputation - just copy if needed
    if (output_bases != input_bases && config->are_points_on_device) {
        cudaStream_t stream = config->stream;
        cudaError_t err = cudaMemcpyAsync(
            output_bases, input_bases, 
            bases_size * sizeof(bls12_381::G1Affine),
            cudaMemcpyDeviceToDevice, stream
        );
        if (!config->is_async) cudaStreamSynchronize(stream);
        return (err == cudaSuccess) ? eIcicleError::SUCCESS : eIcicleError::COPY_FAILED;
    }
    return eIcicleError::SUCCESS;
}

} // extern "C"
