# BLS12-381 CUDA Security Audit Test Suite

This directory contains a comprehensive test suite designed to validate the security and correctness of the BLS12-381 CUDA cryptographic implementation. The tests are structured to meet the requirements of professional security audits.

## Overview

The test suite covers:

| Test File | Description | Critical for |
|-----------|-------------|--------------|
| `test_known_answer_vectors.cu` | Official specification constants and KAT | Compliance |
| `test_field_properties.cu` | Field arithmetic algebraic axioms | Correctness |
| `test_curve_operations.cu` | G1/G2 elliptic curve group laws | Correctness |
| `test_msm_security.cu` | Multi-Scalar Multiplication correctness | Performance |
| `test_ntt_security.cu` | Number Theoretic Transform properties | Performance |
| `test_security_edge_cases.cu` | Boundary conditions, constant-time ops | Security |

## Building

```bash
cd cuda-backend
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
```

## Running Tests

### Run All Tests
```bash
make test
# or
ctest --output-on-failure
```

### Run Security Audit Suite Only
```bash
make security-audit
```

### Run Individual Test
```bash
./test_known_answer_vectors
./test_field_properties
./test_curve_operations
./test_msm_security
./test_ntt_security
./test_security_edge_cases
```

## Test Categories

### 1. Known Answer Tests (`test_known_answer_vectors.cu`)

Validates implementation against official BLS12-381 specification values:

- **Field Moduli**: Fr (scalar field) and Fq (base field) modulus verification
- **Generator Points**: G1 and G2 generator coordinates
- **Montgomery Constants**: R, R², R⁻¹ for both fields
- **Subgroup Order**: Prime order r of the subgroups

**Why it matters**: Ensures the implementation uses correct cryptographic constants. Incorrect constants would make the implementation incompatible with the BLS12-381 standard.

### 2. Field Property Tests (`test_field_properties.cu`)

Verifies all field axioms are satisfied:

- **Additive Identity**: a + 0 = a
- **Multiplicative Identity**: a × 1 = a
- **Additive Inverse**: a + (-a) = 0
- **Multiplicative Inverse**: a × a⁻¹ = 1 (for a ≠ 0)
- **Commutativity**: a + b = b + a, a × b = b × a
- **Associativity**: (a + b) + c = a + (b + c)
- **Distributivity**: a × (b + c) = a×b + a×c
- **Double Negation**: -(-a) = a
- **Fermat's Little Theorem**: a^(p-1) = 1

**Why it matters**: Field arithmetic is the foundation of all cryptographic operations. Any violation of these axioms would lead to incorrect computations.

### 3. Curve Operation Tests (`test_curve_operations.cu`)

Tests elliptic curve group law compliance for G1 and G2:

- **Identity Element**: P + O = P = O + P
- **Inverse**: P + (-P) = O
- **Commutativity**: P + Q = Q + P
- **Associativity**: (P + Q) + R = P + (Q + R)
- **Doubling**: P + P = 2P
- **Mixed Addition**: Projective + Affine operations

**Why it matters**: Incorrect group operations would produce invalid signatures and proofs.

### 4. MSM Security Tests (`test_msm_security.cu`)

Tests Multi-Scalar Multiplication correctness:

- **Basic Cases**: MSM(1, G) = G, MSM(0, G) = O
- **All Zeros**: MSM([0,0,...], bases) = O
- **All Ones**: MSM([1,1,...], bases) = Σbases
- **Reference Comparison**: MSM vs naive double-and-add
- **Mixed Scalars**: Combination of zero and non-zero
- **Determinism**: Same input always produces same output

**Why it matters**: MSM is the performance-critical operation in zkSNARKs. Incorrect MSM would produce invalid proofs.

### 5. NTT Security Tests (`test_ntt_security.cu`)

Tests Number Theoretic Transform properties:

- **Roundtrip**: INTT(NTT(a)) = a
- **Linearity**: NTT(a+b) = NTT(a) + NTT(b)
- **Convolution**: NTT(a⊛b) = NTT(a)⊙NTT(b)
- **Zero Polynomial**: NTT(0) = 0
- **Constant Polynomial**: NTT([c,0,0,...]) = [c,c,c,...]
- **Various Sizes**: Power-of-2 domain sizes
- **Batch Processing**: Multiple polynomials at once

**Why it matters**: NTT is used for polynomial multiplication in zkSNARKs. Incorrect NTT would corrupt polynomial arithmetic.

### 6. Edge Case & Security Tests (`test_security_edge_cases.cu`)

Tests boundary conditions and security properties:

- **Zero Handling**: 0+0=0, 0×a=0, inv(0) handling
- **Constant-Time Selection**: cmov correctness for all conditions
- **Identity Handling**: Operations with identity point
- **Negation Properties**: P+(-P)=O, -(-P)=P
- **Scalar Edge Cases**: 0×P=O, 1×P=P
- **Invalid Input Handling**: Graceful error handling

**Why it matters**: Edge cases are common attack vectors. Constant-time operations prevent timing attacks.

## Security Properties Tested

### Constant-Time Operations

The following operations are tested for constant-time behavior:

| Function | Purpose |
|----------|---------|
| `field_cmov` | Conditional field element selection |
| `g1_cmov` | Conditional G1 point selection |
| `g2_cmov` | Conditional G2 point selection |

### Memory Safety

Tests verify:
- No buffer overflows in field operations
- Proper handling of edge cases without crashes
- Correct memory bounds in MSM bucket operations

### Algebraic Completeness

All test functions verify the complete algebraic structure:
- Field axioms (13 properties tested)
- Group axioms (5 properties tested per group)
- Homomorphism properties for NTT

## Test Framework

The test suite uses a custom `SecurityTestSuite` class defined in `security_audit_tests.cuh`:

```cpp
class SecurityTestSuite {
    void add_test(const char* name, const char* category, TestFunc func);
    bool run_all();
};
```

Each test returns `TestResult::PASSED`, `TestResult::FAILED`, or `TestResult::SKIPPED`.

## Adding New Tests

1. Include the header:
```cpp
#include "security_audit_tests.cuh"
```

2. Create a test function:
```cpp
TestResult test_my_feature() {
    // ... test logic ...
    if (condition_met) {
        return TestResult::PASSED;
    }
    std::cout << "\n    Error description";
    return TestResult::FAILED;
}
```

3. Register in main:
```cpp
suite.add_test("Test name", "Category", test_my_feature);
```

## CI Integration

For continuous integration, add to your pipeline:

```yaml
test:
  script:
    - mkdir build && cd build
    - cmake .. -DBUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
    - make -j$(nproc)
    - make security-audit
```

## Expected Output

```
GPU: NVIDIA GeForce RTX 5090 (SM 12.0)

Running Security Audit Tests...
====================================

[Fr Constants] Generator Point Verification .............. PASSED
[Fr Constants] Subgroup Order r .......................... PASSED
[Fr Modulus] Official BLS12-381 value .................... PASSED
...

====================================
Results: 45 passed, 0 failed, 0 skipped
====================================
```

## References

- [BLS12-381 Specification](https://hackmd.io/@benjaminion/bls12-381)
- [EIP-2537: BLS12-381 Precompiles](https://eips.ethereum.org/EIPS/eip-2537)
- [BLST Library Test Vectors](https://github.com/supranational/blst)
- [Arkworks BLS12-381](https://github.com/arkworks-rs/curves)
- [Zcash Protocol Specification](https://zips.z.cash/protocol/protocol.pdf)

## Contact

For security-related issues, please follow the security policy in SECURITY.md.
