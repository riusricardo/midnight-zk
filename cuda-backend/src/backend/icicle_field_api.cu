/**
 * @file icicle_field_api.cu
 * @brief Thin C API wrapper for Icicle compatibility
 * 
 * This file provides additional extern "C" exports and the library constructor.
 * Most Icicle-compatible symbols are already exported by field_backend.cu and vec_ops.cu.
 * 
 * Build: Compiled together with field_backend.cu, vec_ops.cu, params.cu
 * into a single shared library.
 */

#include "field.cuh"
#include "ntt.cuh"
#include "icicle_types.cuh"

using namespace bls12_381;
using namespace icicle;

// =============================================================================
// Additional C API Exports (symbols not already in field_backend.cu/vec_ops.cu)
// =============================================================================

extern "C" {

// Library constructor - runs when loaded via dlopen()
// This is the key entry point that Icicle's runtime looks for
void __attribute__((constructor)) register_bls12_381_field_backend() {
    // The C API symbols (bls12_381_ntt_cuda, bls12_381_vector_add, etc.)
    // are already exported by field_backend.cu and vec_ops.cu
    // Icicle runtime discovers them via dlsym()
}

} // extern "C"
