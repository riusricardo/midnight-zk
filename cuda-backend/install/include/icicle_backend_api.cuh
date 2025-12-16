/**
 * @file icicle_backend_api.cuh
 * @brief Icicle backend registration API
 * 
 * This file provides the registration functions and macros needed to
 * register our MSM/NTT implementations with Icicle's dispatcher.
 * 
 * The registration functions (register_msm, etc.) are resolved at runtime
 * from libicicle_curve_bls12_381.so when our backend is loaded via dlopen().
 * 
 * Type definitions are in icicle_types.cuh - this file only adds
 * registration-specific declarations.
 * 
 * IMPORTANT: The type declarations here must produce the EXACT same
 * C++ mangled symbols as Icicle's core library. This means we need
 * to use the same namespaces, template names, and struct names.
 */

#pragma once

// Get all Icicle-compatible type definitions
#include "icicle_types.cuh"
#include <functional>
#include <string>

// =============================================================================
// Forward declarations matching Icicle's type names EXACTLY for symbol mangling
// =============================================================================
// These MUST match Icicle's internal type names for correct symbol resolution

namespace bls12_381 {
    struct fp_config;  // Scalar field config
    struct fq_config;  // Base field config
    struct G1;         // G1 curve marker
}

// Template declarations matching Icicle's templates
// These are forward declarations only - we use them for type signatures
template<typename Config> class Field;
template<typename BaseField> class Affine;
template<typename BaseField, typename ScalarField, typename Gen> class Projective;

// =============================================================================
// Registration Function Declarations
// =============================================================================
// These are resolved at runtime from libicicle_curve_bls12_381.so
// The backend library has these as undefined symbols (U) that get
// resolved when loaded into a process that has the Icicle core loaded.
//
// CRITICAL: The function signatures MUST match Icicle's exactly for
// proper symbol name mangling. This includes:
// - Template instantiation types (Field<fp_config>, Affine<...>, etc.)
// - Namespace (icicle::)
// - Parameter order and types

namespace icicle {

// Type aliases matching Icicle's internal types for BLS12-381
using icicle_scalar_t = Field<bls12_381::fp_config>;
using icicle_point_field_t = Field<bls12_381::fq_config>;
using icicle_affine_t = Affine<icicle_point_field_t>;
using icicle_projective_t = Projective<icicle_point_field_t, icicle_scalar_t, bls12_381::G1>;

// MSM registration function types matching Icicle's exact signatures
using MsmImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_scalar_t* scalars,
    const icicle_affine_t* bases,
    int msm_size,
    const MSMConfig& config,
    icicle_projective_t* results)>;

using MsmPreComputeImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_affine_t* input_bases,
    int bases_size,
    const MSMConfig& config,
    icicle_affine_t* output_bases)>;

// These functions are provided by libicicle_curve_bls12_381.so
// They're declared here so we can call them at registration time
extern void register_msm(const std::string& deviceType, MsmImpl impl);
extern void register_msm_precompute_bases(const std::string& deviceType, MsmPreComputeImpl impl);

} // namespace icicle

// =============================================================================
// Registration Macros
// =============================================================================
// These create static initializers that run when the library is loaded

// Unique variable name generator
#define ICICLE_CONCAT_INNER(a, b) a##b
#define ICICLE_CONCAT(a, b) ICICLE_CONCAT_INNER(a, b)
#define ICICLE_UNIQUE(prefix) ICICLE_CONCAT(prefix, __COUNTER__)

#define REGISTER_MSM_BACKEND(DEVICE_TYPE, FUNC)                    \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_msm_) = []() -> bool {      \
            icicle::register_msm(DEVICE_TYPE, FUNC);               \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_MSM_PRE_COMPUTE_BASES_BACKEND(DEVICE_TYPE, FUNC)  \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_msm_pre_) = []() -> bool {  \
            icicle::register_msm_precompute_bases(DEVICE_TYPE, FUNC); \
            return true;                                            \
        }();                                                        \
    }
