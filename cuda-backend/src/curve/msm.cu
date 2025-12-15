/**
 * @file msm.cu
 * @brief MSM implementation
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
