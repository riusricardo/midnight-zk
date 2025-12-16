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
 * 
 * G1 vs G2 REGISTRATION:
 * ======================
 * G1 and G2 require separate registration functions because they operate
 * on different types:
 * - G1: Uses Fq base field, G1Affine/G1Projective points
 * - G2: Uses Fq2 extension field, G2Affine/G2Projective points
 * 
 * Icicle's dispatcher routes to the correct implementation based on
 * the registered device type and curve type.
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
    struct G2;         // G2 curve marker
}

// Template declarations matching Icicle's templates
// These are forward declarations only - we use them for type signatures
template<typename Config> class Field;
template<typename BaseField> class Affine;
template<typename BaseField, typename ScalarField, typename Gen> class Projective;

// G2 extension field type (Fq2)
template<typename BaseConfig, typename BaseField> class ComplexExtensionField;

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

// Type aliases matching Icicle's internal types for BLS12-381 G1
using icicle_scalar_t = Field<bls12_381::fp_config>;
using icicle_point_field_t = Field<bls12_381::fq_config>;
using icicle_affine_t = Affine<icicle_point_field_t>;
using icicle_projective_t = Projective<icicle_point_field_t, icicle_scalar_t, bls12_381::G1>;

// Type aliases for G2 (uses Fq2 extension field)
// Note: These may need adjustment based on actual Icicle symbol names
using icicle_g2_field_t = ComplexExtensionField<bls12_381::fq_config, icicle_point_field_t>;
using icicle_g2_affine_t = Affine<icicle_g2_field_t>;
using icicle_g2_projective_t = Projective<icicle_g2_field_t, icicle_scalar_t, bls12_381::G2>;

// =============================================================================
// G1 MSM Registration
// =============================================================================

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

// =============================================================================
// G2 MSM Registration
// =============================================================================

// G2 MSM registration function types
using MsmG2Impl = std::function<eIcicleError(
    const Device& device,
    const icicle_scalar_t* scalars,
    const icicle_g2_affine_t* bases,
    int msm_size,
    const MSMConfig& config,
    icicle_g2_projective_t* results)>;

using MsmG2PreComputeImpl = std::function<eIcicleError(
    const Device& device,
    const icicle_g2_affine_t* input_bases,
    int bases_size,
    const MSMConfig& config,
    icicle_g2_affine_t* output_bases)>;

// G2 registration functions (provided by libicicle_curve_bls12_381.so)
// Note: These may need to match actual Icicle function names
extern void register_msm_g2(const std::string& deviceType, MsmG2Impl impl);
extern void register_msm_g2_precompute_bases(const std::string& deviceType, MsmG2PreComputeImpl impl);

} // namespace icicle

// =============================================================================
// Registration Macros
// =============================================================================
// These create static initializers that run when the library is loaded

// Unique variable name generator
#define ICICLE_CONCAT_INNER(a, b) a##b
#define ICICLE_CONCAT(a, b) ICICLE_CONCAT_INNER(a, b)
#define ICICLE_UNIQUE(prefix) ICICLE_CONCAT(prefix, __COUNTER__)

// G1 MSM Registration
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

// G2 MSM Registration
// Note: These macros register G2 implementations. If Icicle doesn't
// have G2 registration functions yet, comment these out and use
// the G2 MSM through the C API directly.
#ifdef ICICLE_HAS_G2_REGISTRATION
#define REGISTER_MSM_G2_BACKEND(DEVICE_TYPE, FUNC)                 \
    namespace {                                                     \
        static bool ICICLE_UNIQUE(_reg_msm_g2_) = []() -> bool {   \
            icicle::register_msm_g2(DEVICE_TYPE, FUNC);            \
            return true;                                            \
        }();                                                        \
    }

#define REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND(DEVICE_TYPE, FUNC) \
    namespace {                                                      \
        static bool ICICLE_UNIQUE(_reg_msm_g2_pre_) = []() -> bool { \
            icicle::register_msm_g2_precompute_bases(DEVICE_TYPE, FUNC); \
            return true;                                             \
        }();                                                         \
    }
#else
// Stub macros when Icicle doesn't have G2 registration
// G2 MSM is still available through C API (bls12_381_g2_msm_cuda)
#define REGISTER_MSM_G2_BACKEND(DEVICE_TYPE, FUNC) \
    /* G2 Icicle registration disabled - use C API */
#define REGISTER_MSM_G2_PRE_COMPUTE_BASES_BACKEND(DEVICE_TYPE, FUNC) \
    /* G2 Icicle registration disabled - use C API */
#endif
