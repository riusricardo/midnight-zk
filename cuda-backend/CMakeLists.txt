cmake_minimum_required(VERSION 3.18)
project(icicle_backend_cuda_bls12_381 LANGUAGES CXX CUDA)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architectures (Turing through Blackwell)
# 75 = Turing (RTX 20xx)
# 80 = Ampere (A100)
# 86 = Ampere (RTX 30xx)
# 89 = Ada Lovelace (RTX 40xx)
# 100 = Hopper (H100)
# 120 = Blackwell (RTX 50xx)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 100 120)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# =============================================================================
# Redistributable Build Configuration
# =============================================================================
# Use static CUDA runtime for redistributability
# This embeds the CUDA runtime into the library, removing the need for
# users to have a specific CUDA version installed.
option(CUDA_STATIC_RUNTIME "Use static CUDA runtime" ON)

if(CUDA_STATIC_RUNTIME)
    set(CUDA_RUNTIME_LIB CUDA::cudart_static)
    message(STATUS "Using static CUDA runtime for redistributability")
else()
    set(CUDA_RUNTIME_LIB CUDA::cudart)
    message(STATUS "Using dynamic CUDA runtime")
endif()

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# For static runtime, we need to link pthread and dl
if(CUDA_STATIC_RUNTIME)
    find_package(Threads REQUIRED)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files for field library
# Note: ntt.cu was removed (all kernels were dead code, actual implementation is in field_backend.cu)
set(FIELD_SOURCES
    src/params.cu
    src/field/vec_ops.cu
    src/field/field_backend.cu
)

# Source files for curve library  
# Note: params.cu is already in field library, don't duplicate
set(CURVE_SOURCES
    src/curve/point_ops.cu
    src/curve/msm.cu
    src/curve/ecntt.cu
    src/curve/montgomery.cu
    src/curve/curve_backend.cu
)

# Note: msm_kernels.cu requires CUB library for sorting
# Add it conditionally if CUB is available
find_package(CUB QUIET)
if(CUB_FOUND)
    list(APPEND CURVE_SOURCES src/curve/msm_kernels.cu)
    message(STATUS "CUB found, enabling optimized MSM kernels")
endif()

# Field library - use STATIC for proper CUDA device code linking
add_library(icicle_backend_cuda_field_bls12_381 STATIC ${FIELD_SOURCES})
target_link_libraries(icicle_backend_cuda_field_bls12_381 
    ${CUDA_RUNTIME_LIB}
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
)
set_target_properties(icicle_backend_cuda_field_bls12_381 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC_RUNTIME}>,Static,Shared>
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "icicle_backend_cuda_field_bls12_381"
    # RPATH settings for redistributability
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
)

# Curve library - use STATIC for proper CUDA device code linking
add_library(icicle_backend_cuda_curve_bls12_381 STATIC ${CURVE_SOURCES})
target_link_libraries(icicle_backend_cuda_curve_bls12_381 
    icicle_backend_cuda_field_bls12_381
    ${CUDA_RUNTIME_LIB}
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
)
set_target_properties(icicle_backend_cuda_curve_bls12_381 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC_RUNTIME}>,Static,Shared>
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "icicle_backend_cuda_curve_bls12_381"
    # RPATH settings for redistributability
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN"
)

# Installation
install(TARGETS 
    icicle_backend_cuda_field_bls12_381 
    icicle_backend_cuda_curve_bls12_381
    LIBRARY DESTINATION lib/backend/bls12_381/cuda
)

# Tests
enable_testing()

add_executable(test_backend 
    tests/test_backend.cu
)
target_link_libraries(test_backend 
    icicle_backend_cuda_curve_bls12_381
    icicle_backend_cuda_field_bls12_381
    ${CUDA_RUNTIME_LIB}
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
)
set_target_properties(test_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC_RUNTIME}>,Static,Shared>
)

add_test(NAME backend_tests COMMAND test_backend)

# Comprehensive test suite with profiling
add_executable(test_vectors
    tests/test_vectors.cu
)
target_link_libraries(test_vectors
    icicle_backend_cuda_curve_bls12_381
    icicle_backend_cuda_field_bls12_381
    ${CUDA_RUNTIME_LIB}
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
)
set_target_properties(test_vectors PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC_RUNTIME}>,Static,Shared>
)

add_test(NAME vector_tests COMMAND test_vectors)

# Custom NTT Test
add_executable(test_ntt_custom
    tests/test_ntt_custom.cu
)
target_link_libraries(test_ntt_custom
    icicle_backend_cuda_curve_bls12_381
    icicle_backend_cuda_field_bls12_381
    ${CUDA_RUNTIME_LIB}
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
)
set_target_properties(test_ntt_custom PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC_RUNTIME}>,Static,Shared>
)

add_test(NAME ntt_custom_tests COMMAND test_ntt_custom)

# NTT Debug Test
add_executable(test_ntt_debug
    tests/test_ntt_debug.cu
)
target_link_libraries(test_ntt_debug
    icicle_backend_cuda_curve_bls12_381
    icicle_backend_cuda_field_bls12_381
    ${CUDA_RUNTIME_LIB}
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
)
set_target_properties(test_ntt_debug PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC_RUNTIME}>,Static,Shared>
)

# Roundtrip and Stress Tests
add_executable(test_roundtrip
    tests/test_roundtrip.cu
)
target_link_libraries(test_roundtrip
    icicle_backend_cuda_curve_bls12_381
    icicle_backend_cuda_field_bls12_381
    ${CUDA_RUNTIME_LIB}
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:Threads::Threads>
    $<$<BOOL:${CUDA_STATIC_RUNTIME}>:${CMAKE_DL_LIBS}>
)
set_target_properties(test_roundtrip PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RUNTIME_LIBRARY $<IF:$<BOOL:${CUDA_STATIC_RUNTIME}>,Static,Shared>
)

add_test(NAME roundtrip_tests COMMAND test_roundtrip)
