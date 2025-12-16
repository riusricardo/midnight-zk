/**
 * @file icicle_curve_api.cu
 * @brief Thin C API wrapper for Icicle curve operations compatibility
 * 
 * This file provides the library constructor for dlopen() compatibility.
 * Most Icicle-compatible symbols are already exported by msm.cu.
 * 
 * Build: Compiled together with curve_backend.cu, msm.cu, etc.
 * into a single shared library.
 */

#include "field.cuh"
#include "point.cuh"
#include "icicle_types.cuh"

using namespace bls12_381;
using namespace icicle;

// =============================================================================
// Library Constructor
// =============================================================================

extern "C" {

// Library constructor - runs when loaded via dlopen()
// C API symbols (bls12_381_msm_cuda, etc.) are already in msm.cu
void __attribute__((constructor)) register_bls12_381_curve_backend() {
    // Icicle runtime discovers C API symbols via dlsym()
}

} // extern "C"
