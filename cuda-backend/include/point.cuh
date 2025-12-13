/**
 * @file point.cuh
 * @brief BLS12-381 Elliptic curve point types and operations
 * 
 * Implements G1 and G2 point operations in both affine and projective coordinates.
 */

#pragma once

#include "field.cuh"

namespace bls12_381 {

// =============================================================================
// Forward declarations
// =============================================================================

template<typename BaseField> struct Affine;
template<typename BaseField, typename ScalarField, typename CurveTag> struct Projective;
struct G1;
struct G2;

// =============================================================================
// Curve tags
// =============================================================================

struct G1 {};
struct G2 {};

// =============================================================================
// Complex extension field for G2 (Fq2 = Fq[u] / (u^2 + 1))
// =============================================================================

template<typename BaseConfig, typename BaseField>
struct ComplexExtensionField {
    BaseField c0;  // Real part
    BaseField c1;  // Imaginary part (coefficient of u)

    __host__ __device__ ComplexExtensionField() : c0(), c1() {}
    
    __host__ __device__ ComplexExtensionField(const BaseField& real, const BaseField& imag)
        : c0(real), c1(imag) {}

    __host__ __device__ static ComplexExtensionField zero() {
        return ComplexExtensionField();
    }

    __host__ __device__ static ComplexExtensionField one() {
        return ComplexExtensionField(BaseField::one(), BaseField::zero());
    }

    __host__ __device__ bool is_zero() const {
        return c0.is_zero() && c1.is_zero();
    }

    __host__ __device__ bool operator==(const ComplexExtensionField& other) const {
        return c0 == other.c0 && c1 == other.c1;
    }

    __host__ __device__ bool operator!=(const ComplexExtensionField& other) const {
        return !(*this == other);
    }
};

// Fq2 = Fq[u] / (u^2 + 1)
using Fq2 = ComplexExtensionField<fq_config, Fq>;

// Fq2 arithmetic
__device__ __forceinline__ void fq2_add(Fq2& result, const Fq2& a, const Fq2& b) {
    field_add(result.c0, a.c0, b.c0);
    field_add(result.c1, a.c1, b.c1);
}

__device__ __forceinline__ void fq2_sub(Fq2& result, const Fq2& a, const Fq2& b) {
    field_sub(result.c0, a.c0, b.c0);
    field_sub(result.c1, a.c1, b.c1);
}

__device__ __forceinline__ void fq2_mul(Fq2& result, const Fq2& a, const Fq2& b) {
    // (a0 + a1*u) * (b0 + b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
    // Using Karatsuba: 3 multiplications instead of 4
    Fq t0, t1, t2;
    
    field_mul(t0, a.c0, b.c0);           // a0 * b0
    field_mul(t1, a.c1, b.c1);           // a1 * b1
    
    Fq a_sum, b_sum;
    field_add(a_sum, a.c0, a.c1);        // a0 + a1
    field_add(b_sum, b.c0, b.c1);        // b0 + b1
    field_mul(t2, a_sum, b_sum);         // (a0 + a1)(b0 + b1)
    
    field_sub(result.c0, t0, t1);        // a0*b0 - a1*b1
    field_sub(result.c1, t2, t0);        // (a0+a1)(b0+b1) - a0*b0
    field_sub(result.c1, result.c1, t1); // ... - a1*b1
}

__device__ __forceinline__ void fq2_sqr(Fq2& result, const Fq2& a) {
    // (a0 + a1*u)^2 = (a0^2 - a1^2) + 2*a0*a1*u
    // Optimized: (a0 + a1)(a0 - a1) for real part
    Fq t0, t1, t2;
    
    field_add(t0, a.c0, a.c1);           // a0 + a1
    field_sub(t1, a.c0, a.c1);           // a0 - a1
    field_mul(result.c0, t0, t1);        // (a0+a1)(a0-a1) = a0^2 - a1^2
    
    field_mul(t2, a.c0, a.c1);           // a0 * a1
    field_add(result.c1, t2, t2);        // 2 * a0 * a1
}

__device__ __forceinline__ void fq2_neg(Fq2& result, const Fq2& a) {
    field_neg(result.c0, a.c0);
    field_neg(result.c1, a.c1);
}

__device__ __forceinline__ void fq2_inv(Fq2& result, const Fq2& a) {
    // (a0 + a1*u)^-1 = (a0 - a1*u) / (a0^2 + a1^2)
    // For u^2 = -1: norm = a0^2 + a1^2
    Fq t0, t1, norm, norm_inv;
    
    field_sqr(t0, a.c0);                 // a0^2
    field_sqr(t1, a.c1);                 // a1^2
    field_add(norm, t0, t1);             // a0^2 + a1^2
    field_inv(norm_inv, norm);           // 1 / (a0^2 + a1^2)
    
    field_mul(result.c0, a.c0, norm_inv);  // a0 / norm
    field_neg(t0, a.c1);
    field_mul(result.c1, t0, norm_inv);    // -a1 / norm
}

// Overload for compatibility with Projective::to_affine
__device__ __forceinline__ void field_inv(Fq2& result, const Fq2& a) {
    fq2_inv(result, a);
}

// Overload field_mul for Fq2 to work with Projective templates
__device__ __forceinline__ void field_mul(Fq2& result, const Fq2& a, const Fq2& b) {
    fq2_mul(result, a, b);
}

// Fq2 operators (for convenience)
__device__ __forceinline__ Fq2 operator+(const Fq2& a, const Fq2& b) {
    Fq2 result;
    fq2_add(result, a, b);
    return result;
}

__device__ __forceinline__ Fq2 operator-(const Fq2& a, const Fq2& b) {
    Fq2 result;
    fq2_sub(result, a, b);
    return result;
}

__device__ __forceinline__ Fq2 operator*(const Fq2& a, const Fq2& b) {
    Fq2 result;
    fq2_mul(result, a, b);
    return result;
}

__device__ __forceinline__ Fq2 operator-(const Fq2& a) {
    Fq2 result;
    fq2_neg(result, a);
    return result;
}

// Forward declaration for Projective
template<typename BaseField, typename ScalarField, typename CurveTag> struct Projective;

// =============================================================================
// Affine point representation
// =============================================================================

template<typename BaseField>
struct alignas(64) Affine {
    BaseField x;
    BaseField y;

    __host__ __device__ Affine() : x(), y() {}
    
    __host__ __device__ Affine(const BaseField& _x, const BaseField& _y) : x(_x), y(_y) {}

    __host__ __device__ static Affine identity() {
        // Point at infinity represented as (0, 0)
        return Affine();
    }

    __host__ __device__ bool is_identity() const {
        return x.is_zero() && y.is_zero();
    }

    __host__ __device__ bool operator==(const Affine& other) const {
        return x == other.x && y == other.y;
    }

    __host__ __device__ Affine neg() const {
        Affine result;
        result.x = x;
        result.y = -y;  // Use unary minus operator
        return result;
    }
    
    // Convert to projective coordinates (defined after Projective is complete)
    template<typename ScalarField, typename CurveTag>
    __device__ __forceinline__ Projective<BaseField, ScalarField, CurveTag> to_projective() const;
};

// Type aliases
using G1Affine = Affine<Fq>;
using G2Affine = Affine<Fq2>;

// =============================================================================
// Projective point representation (X, Y, Z) where x = X/Z, y = Y/Z
// =============================================================================

template<typename BaseField, typename ScalarField, typename CurveTag>
struct alignas(64) Projective {
    BaseField X;
    BaseField Y;
    BaseField Z;

    __host__ __device__ Projective() : X(), Y(), Z() {
        // Identity point has Z = 0
    }

    __host__ __device__ Projective(const BaseField& x, const BaseField& y, const BaseField& z)
        : X(x), Y(y), Z(z) {}

    __host__ __device__ static Projective identity() {
        Projective p;
        p.X = BaseField::zero();
        p.Y = BaseField::one();  // Convention: (0:1:0) for identity
        p.Z = BaseField::zero();
        return p;
    }

    __host__ __device__ static Projective from_affine(const Affine<BaseField>& p) {
        if (p.is_identity()) {
            return identity();
        }
        return Projective(p.x, p.y, BaseField::one());
    }

    __host__ __device__ bool is_identity() const {
        return Z.is_zero();
    }

    __host__ __device__ bool operator==(const Projective& other) const {
        if (is_identity() && other.is_identity()) return true;
        if (is_identity() || other.is_identity()) return false;
        
        // Compare X1*Z2 == X2*Z1 and Y1*Z2 == Y2*Z1
        BaseField lhs_x, rhs_x, lhs_y, rhs_y;
        field_mul(lhs_x, X, other.Z);
        field_mul(rhs_x, other.X, Z);
        field_mul(lhs_y, Y, other.Z);
        field_mul(rhs_y, other.Y, Z);
        
        return lhs_x == rhs_x && lhs_y == rhs_y;
    }
    
    // Convert to affine coordinates
    __device__ __forceinline__ Affine<BaseField> to_affine() const {
        if (is_identity()) {
            return Affine<BaseField>::identity();
        }
        BaseField z_inv;
        field_inv(z_inv, Z);
        BaseField ax, ay;
        field_mul(ax, X, z_inv);
        field_mul(ay, Y, z_inv);
        return Affine<BaseField>(ax, ay);
    }
};

// Type aliases
using G1Projective = Projective<Fq, Fr, G1>;
using G2Projective = Projective<Fq2, Fr, G2>;

// Implementation of Affine::to_projective
template<typename BaseField>
template<typename ScalarField, typename CurveTag>
__device__ __forceinline__ Projective<BaseField, ScalarField, CurveTag> 
Affine<BaseField>::to_projective() const {
    return Projective<BaseField, ScalarField, CurveTag>::from_affine(*this);
}

// Convenience methods for G1Affine and G2Affine
__device__ __forceinline__ G1Projective g1_affine_to_projective(const G1Affine& p) {
    return G1Projective::from_affine(p);
}

__device__ __forceinline__ G2Projective g2_affine_to_projective(const G2Affine& p) {
    return G2Projective::from_affine(p);
}

// =============================================================================
// G1 Point Operations
// =============================================================================

/**
 * @brief Double a projective point: result = 2 * P
 * 
 * Uses the doubling formula for short Weierstrass curves y^2 = x^3 + b
 * Cost: 1S + 5M + 8add
 */
__device__ __forceinline__ void g1_double(G1Projective& result, const G1Projective& p) {
    if (p.is_identity()) {
        result = p;
        return;
    }

    Fq t0, t1, t2, t3;
    
    // t0 = Y^2
    field_sqr(t0, p.Y);
    
    // t1 = 4 * X * Y^2
    field_mul(t1, p.X, t0);
    field_add(t1, t1, t1);
    field_add(t1, t1, t1);
    
    // t2 = 8 * Y^4
    field_sqr(t2, t0);
    field_add(t2, t2, t2);
    field_add(t2, t2, t2);
    field_add(t2, t2, t2);
    
    // t3 = 3 * X^2 (for a = 0)
    field_sqr(t3, p.X);
    Fq t3_2;
    field_add(t3_2, t3, t3);
    field_add(t3, t3_2, t3);
    
    // X3 = t3^2 - 2*t1
    field_sqr(result.X, t3);
    field_sub(result.X, result.X, t1);
    field_sub(result.X, result.X, t1);
    
    // Y3 = t3 * (t1 - X3) - t2
    field_sub(t0, t1, result.X);
    field_mul(result.Y, t3, t0);
    field_sub(result.Y, result.Y, t2);
    
    // Z3 = 2 * Y * Z
    field_mul(result.Z, p.Y, p.Z);
    field_add(result.Z, result.Z, result.Z);
}

/**
 * @brief Add two projective points: result = P + Q
 * 
 * Uses complete addition formula.
 */
__device__ __forceinline__ void g1_add(G1Projective& result, const G1Projective& p, const G1Projective& q) {
    if (p.is_identity()) {
        result = q;
        return;
    }
    if (q.is_identity()) {
        result = p;
        return;
    }

    Fq z1z1, z2z2, u1, u2, s1, s2, h, i, j, r, v;
    
    // Z1Z1 = Z1^2
    field_sqr(z1z1, p.Z);
    // Z2Z2 = Z2^2
    field_sqr(z2z2, q.Z);
    
    // U1 = X1 * Z2Z2
    field_mul(u1, p.X, z2z2);
    // U2 = X2 * Z1Z1
    field_mul(u2, q.X, z1z1);
    
    // S1 = Y1 * Z2 * Z2Z2
    field_mul(s1, p.Y, q.Z);
    field_mul(s1, s1, z2z2);
    // S2 = Y2 * Z1 * Z1Z1
    field_mul(s2, q.Y, p.Z);
    field_mul(s2, s2, z1z1);
    
    // H = U2 - U1
    field_sub(h, u2, u1);
    
    // Check for doubling case
    if (h.is_zero()) {
        Fq diff;
        field_sub(diff, s2, s1);
        if (diff.is_zero()) {
            g1_double(result, p);
            return;
        } else {
            // P + (-P) = Identity
            result = G1Projective::identity();
            return;
        }
    }
    
    // I = (2*H)^2
    field_add(i, h, h);
    field_sqr(i, i);
    
    // J = H * I
    field_mul(j, h, i);
    
    // r = 2 * (S2 - S1)
    field_sub(r, s2, s1);
    field_add(r, r, r);
    
    // V = U1 * I
    field_mul(v, u1, i);
    
    // X3 = r^2 - J - 2*V
    field_sqr(result.X, r);
    field_sub(result.X, result.X, j);
    field_sub(result.X, result.X, v);
    field_sub(result.X, result.X, v);
    
    // Y3 = r * (V - X3) - 2*S1*J
    field_sub(v, v, result.X);
    field_mul(result.Y, r, v);
    field_mul(s1, s1, j);
    field_add(s1, s1, s1);
    field_sub(result.Y, result.Y, s1);
    
    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    field_add(result.Z, p.Z, q.Z);
    field_sqr(result.Z, result.Z);
    field_sub(result.Z, result.Z, z1z1);
    field_sub(result.Z, result.Z, z2z2);
    field_mul(result.Z, result.Z, h);
}

/**
 * @brief Mixed addition: result = P + Q where Q is affine
 * 
 * More efficient than general addition when one point is affine.
 */
__device__ __forceinline__ void g1_add_mixed(G1Projective& result, const G1Projective& p, const G1Affine& q) {
    if (p.is_identity()) {
        result = G1Projective::from_affine(q);
        return;
    }
    if (q.is_identity()) {
        result = p;
        return;
    }

    Fq z1z1, u2, s2, h, hh, i, j, r, v;
    
    // Z1Z1 = Z1^2
    field_sqr(z1z1, p.Z);
    
    // U2 = X2 * Z1Z1 (U1 = X1)
    field_mul(u2, q.x, z1z1);
    
    // S2 = Y2 * Z1 * Z1Z1 (S1 = Y1)
    field_mul(s2, q.y, p.Z);
    field_mul(s2, s2, z1z1);
    
    // H = U2 - X1
    field_sub(h, u2, p.X);
    
    // Check for doubling case
    if (h.is_zero()) {
        Fq diff;
        field_sub(diff, s2, p.Y);
        if (diff.is_zero()) {
            g1_double(result, p);
            return;
        } else {
            result = G1Projective::identity();
            return;
        }
    }
    
    // HH = H^2
    field_sqr(hh, h);
    
    // I = 4 * HH
    field_add(i, hh, hh);
    field_add(i, i, i);
    
    // J = H * I
    field_mul(j, h, i);
    
    // r = 2 * (S2 - Y1)
    field_sub(r, s2, p.Y);
    field_add(r, r, r);
    
    // V = X1 * I
    field_mul(v, p.X, i);
    
    // X3 = r^2 - J - 2*V
    field_sqr(result.X, r);
    field_sub(result.X, result.X, j);
    field_sub(result.X, result.X, v);
    field_sub(result.X, result.X, v);
    
    // Y3 = r * (V - X3) - 2*Y1*J
    Fq t;
    field_sub(t, v, result.X);
    field_mul(result.Y, r, t);
    field_mul(t, p.Y, j);
    field_add(t, t, t);
    field_sub(result.Y, result.Y, t);
    
    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    field_add(result.Z, p.Z, h);
    field_sqr(result.Z, result.Z);
    field_sub(result.Z, result.Z, z1z1);
    field_sub(result.Z, result.Z, hh);
}

/**
 * @brief Negate a projective point
 */
__device__ __forceinline__ void g1_neg(G1Projective& result, const G1Projective& p) {
    result.X = p.X;
    field_neg(result.Y, p.Y);
    result.Z = p.Z;
}

/**
 * @brief Convert projective to affine: (X, Y, Z) -> (X/Z, Y/Z)
 */
__device__ __forceinline__ void g1_to_affine(G1Affine& result, const G1Projective& p) {
    if (p.is_identity()) {
        result = G1Affine::identity();
        return;
    }
    
    Fq z_inv, z_inv_sq;
    field_inv(z_inv, p.Z);
    field_sqr(z_inv_sq, z_inv);
    
    field_mul(result.x, p.X, z_inv_sq);
    field_mul(result.y, p.Y, z_inv_sq);
    field_mul(result.y, result.y, z_inv);
}

// =============================================================================
// G2 Point Operations
// =============================================================================

/**
 * @brief Double a G2 projective point: result = 2 * P
 * 
 * Uses the doubling formula for y^2 = x^3 + b where b = 4(1+i) for G2.
 * Cost: 1S + 5M + 8add (in Fq2)
 */
__device__ __forceinline__ void g2_double(G2Projective& result, const G2Projective& p) {
    if (p.is_identity()) {
        result = p;
        return;
    }

    Fq2 t0, t1, t2, t3;
    
    // t0 = Y^2
    fq2_sqr(t0, p.Y);
    
    // t1 = 4 * X * Y^2
    fq2_mul(t1, p.X, t0);
    fq2_add(t1, t1, t1);
    fq2_add(t1, t1, t1);
    
    // t2 = 8 * Y^4
    fq2_sqr(t2, t0);
    fq2_add(t2, t2, t2);
    fq2_add(t2, t2, t2);
    fq2_add(t2, t2, t2);
    
    // t3 = 3 * X^2 (for a = 0)
    fq2_sqr(t3, p.X);
    Fq2 t3_2;
    fq2_add(t3_2, t3, t3);
    fq2_add(t3, t3_2, t3);
    
    // X3 = t3^2 - 2*t1
    fq2_sqr(result.X, t3);
    fq2_sub(result.X, result.X, t1);
    fq2_sub(result.X, result.X, t1);
    
    // Y3 = t3 * (t1 - X3) - t2
    fq2_sub(t0, t1, result.X);
    fq2_mul(result.Y, t3, t0);
    fq2_sub(result.Y, result.Y, t2);
    
    // Z3 = 2 * Y * Z
    fq2_mul(result.Z, p.Y, p.Z);
    fq2_add(result.Z, result.Z, result.Z);
}

/**
 * @brief Add two G2 projective points: result = P + Q
 */
__device__ __forceinline__ void g2_add(G2Projective& result, const G2Projective& p, const G2Projective& q) {
    if (p.is_identity()) {
        result = q;
        return;
    }
    if (q.is_identity()) {
        result = p;
        return;
    }

    Fq2 z1z1, z2z2, u1, u2, s1, s2, h, i, j, r, v;
    
    // Z1Z1 = Z1^2
    fq2_sqr(z1z1, p.Z);
    // Z2Z2 = Z2^2
    fq2_sqr(z2z2, q.Z);
    
    // U1 = X1 * Z2Z2
    fq2_mul(u1, p.X, z2z2);
    // U2 = X2 * Z1Z1
    fq2_mul(u2, q.X, z1z1);
    
    // S1 = Y1 * Z2 * Z2Z2
    fq2_mul(s1, p.Y, q.Z);
    fq2_mul(s1, s1, z2z2);
    // S2 = Y2 * Z1 * Z1Z1
    fq2_mul(s2, q.Y, p.Z);
    fq2_mul(s2, s2, z1z1);
    
    // H = U2 - U1
    fq2_sub(h, u2, u1);
    
    // Check for doubling case
    if (h.is_zero()) {
        Fq2 diff;
        fq2_sub(diff, s2, s1);
        if (diff.is_zero()) {
            g2_double(result, p);
            return;
        } else {
            result = G2Projective::identity();
            return;
        }
    }
    
    // I = (2*H)^2
    fq2_add(i, h, h);
    fq2_sqr(i, i);
    
    // J = H * I
    fq2_mul(j, h, i);
    
    // r = 2 * (S2 - S1)
    fq2_sub(r, s2, s1);
    fq2_add(r, r, r);
    
    // V = U1 * I
    fq2_mul(v, u1, i);
    
    // X3 = r^2 - J - 2*V
    fq2_sqr(result.X, r);
    fq2_sub(result.X, result.X, j);
    fq2_sub(result.X, result.X, v);
    fq2_sub(result.X, result.X, v);
    
    // Y3 = r * (V - X3) - 2*S1*J
    fq2_sub(v, v, result.X);
    fq2_mul(result.Y, r, v);
    fq2_mul(s1, s1, j);
    fq2_add(s1, s1, s1);
    fq2_sub(result.Y, result.Y, s1);
    
    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    fq2_add(result.Z, p.Z, q.Z);
    fq2_sqr(result.Z, result.Z);
    fq2_sub(result.Z, result.Z, z1z1);
    fq2_sub(result.Z, result.Z, z2z2);
    fq2_mul(result.Z, result.Z, h);
}

/**
 * @brief Mixed addition for G2: result = P + Q where Q is affine
 */
__device__ __forceinline__ void g2_add_mixed(G2Projective& result, const G2Projective& p, const G2Affine& q) {
    if (p.is_identity()) {
        result = G2Projective::from_affine(q);
        return;
    }
    if (q.is_identity()) {
        result = p;
        return;
    }

    Fq2 z1z1, u2, s2, h, hh, i, j, r, v;
    
    // Z1Z1 = Z1^2
    fq2_sqr(z1z1, p.Z);
    
    // U2 = X2 * Z1Z1
    fq2_mul(u2, q.x, z1z1);
    
    // S2 = Y2 * Z1 * Z1Z1
    fq2_mul(s2, q.y, p.Z);
    fq2_mul(s2, s2, z1z1);
    
    // H = U2 - X1
    fq2_sub(h, u2, p.X);
    
    // Check for doubling case
    if (h.is_zero()) {
        Fq2 diff;
        fq2_sub(diff, s2, p.Y);
        if (diff.is_zero()) {
            g2_double(result, p);
            return;
        } else {
            result = G2Projective::identity();
            return;
        }
    }
    
    // HH = H^2
    fq2_sqr(hh, h);
    
    // I = 4 * HH
    fq2_add(i, hh, hh);
    fq2_add(i, i, i);
    
    // J = H * I
    fq2_mul(j, h, i);
    
    // r = 2 * (S2 - Y1)
    fq2_sub(r, s2, p.Y);
    fq2_add(r, r, r);
    
    // V = X1 * I
    fq2_mul(v, p.X, i);
    
    // X3 = r^2 - J - 2*V
    fq2_sqr(result.X, r);
    fq2_sub(result.X, result.X, j);
    fq2_sub(result.X, result.X, v);
    fq2_sub(result.X, result.X, v);
    
    // Y3 = r * (V - X3) - 2*Y1*J
    Fq2 t;
    fq2_sub(t, v, result.X);
    fq2_mul(result.Y, r, t);
    fq2_mul(t, p.Y, j);
    fq2_add(t, t, t);
    fq2_sub(result.Y, result.Y, t);
    
    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    fq2_add(result.Z, p.Z, h);
    fq2_sqr(result.Z, result.Z);
    fq2_sub(result.Z, result.Z, z1z1);
    fq2_sub(result.Z, result.Z, hh);
}

/**
 * @brief Negate a G2 projective point
 */
__device__ __forceinline__ void g2_neg(G2Projective& result, const G2Projective& p) {
    result.X = p.X;
    fq2_neg(result.Y, p.Y);
    result.Z = p.Z;
}

/**
 * @brief Convert G2 projective to affine
 */
__device__ __forceinline__ void g2_to_affine(G2Affine& result, const G2Projective& p) {
    if (p.is_identity()) {
        result = G2Affine::identity();
        return;
    }
    
    Fq2 z_inv, z_inv_sq;
    fq2_inv(z_inv, p.Z);
    fq2_sqr(z_inv_sq, z_inv);
    
    fq2_mul(result.x, p.X, z_inv_sq);
    fq2_mul(result.y, p.Y, z_inv_sq);
    fq2_mul(result.y, result.y, z_inv);
}

} // namespace bls12_381
