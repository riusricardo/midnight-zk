/**
 * @file msm_fwd.cuh
 * @brief Forward declarations for MSM functions
 * 
 * Use this header when you need to call MSM functions without including
 * the full template definitions (to avoid multiple instantiation).
 */

#pragma once

#include "field.cuh"
#include "point.cuh"
#include "icicle_types.cuh"
#include <cuda_runtime.h>

namespace msm {

// Forward declaration of the MSM function
// The actual template is instantiated in msm.cu
template<typename S, typename A, typename P>
cudaError_t msm_cuda(
    const S* scalars,
    const A* bases,
    int msm_size,
    const MSMConfig& config,
    P* result
);

// Explicit instantiation declarations for BLS12-381 G1 and G2
// These tell the compiler the instantiations exist elsewhere (in msm.cu)
extern template cudaError_t msm_cuda<bls12_381::Fr, bls12_381::G1Affine, bls12_381::G1Projective>(
    const bls12_381::Fr* scalars,
    const bls12_381::G1Affine* bases,
    int msm_size,
    const MSMConfig& config,
    bls12_381::G1Projective* result
);

extern template cudaError_t msm_cuda<bls12_381::Fr, bls12_381::G2Affine, bls12_381::G2Projective>(
    const bls12_381::Fr* scalars,
    const bls12_381::G2Affine* bases,
    int msm_size,
    const MSMConfig& config,
    bls12_381::G2Projective* result
);

} // namespace msm
