//! Type conversions between midnight-curves and ICICLE types
//!
//! ICICLE uses uncompressed point representations:
//! - ScalarField: 32 bytes (Fr field element)
//! - G1Affine: 96 bytes (x: 48 bytes Fq, y: 48 bytes Fq)
//! - G1Projective: 144 bytes (x, y, z: each 48 bytes Fq)
//! - G2Affine: 192 bytes (x: 96 bytes Fq2, y: 96 bytes Fq2)
//! - G2Projective: 288 bytes (x, y, z: each 96 bytes Fq2)
//!
//! Strategy: Direct field element conversion via bytes, avoiding point validation overhead

use midnight_curves::{Fq as Scalar, Fp, Fp2, G1Affine, G1Projective, G2Affine, G2Projective};

#[cfg(feature = "gpu")]
use icicle_bls12_381::curve::{
    ScalarField as IcicleScalar,
    BaseField as IcicleBaseField,
    G1Affine as IcicleG1Affine,
    G1Projective as IcicleG1Projective,
    G2BaseField as IcicleG2BaseField,
    G2Affine as IcicleG2Affine,
    G2Projective as IcicleG2Projective,
};
#[cfg(feature = "gpu")]
use group::prime::PrimeCurveAffine;
#[cfg(feature = "gpu")]
use icicle_core::ecntt::Projective as IcicleProjective;
#[cfg(feature = "gpu")]
use icicle_core::bignum::BigNum;
#[cfg(feature = "gpu")]
use halo2curves::CurveAffine;
#[cfg(feature = "gpu")]
use ff::PrimeField;

/// Type converter for midnight-curves <-> ICICLE types
#[derive(Debug)]
pub struct TypeConverter;

#[cfg(feature = "gpu")]
impl TypeConverter {
    /// Convert midnight Scalar (Fr) to ICICLE ScalarField
    #[inline]
    pub fn scalar_to_icicle(scalar: &Scalar) -> IcicleScalar {
        let bytes = scalar.to_repr();
        IcicleScalar::from_bytes_le(bytes.as_ref())
    }
    
    /// Convert slice of scalars - batched for efficiency
    pub fn scalar_slice_to_icicle_vec(scalars: &[Scalar]) -> Vec<IcicleScalar> {
        scalars.iter().map(|s| {
            let bytes = s.to_repr();
            IcicleScalar::from_bytes_le(bytes.as_ref())
        }).collect()
    }
    
    /// Convert midnight Fp (base field) to ICICLE BaseField
    #[inline]
    fn fp_to_icicle_base(fp: &Fp) -> IcicleBaseField {
        let bytes = fp.to_repr();
        IcicleBaseField::from_bytes_le(bytes.as_ref())
    }
    
    /// Convert midnight G1Affine to ICICLE G1Affine
    /// Extracts x,y coordinates and converts each field element
    #[inline]
    pub fn g1_affine_to_icicle(point: &G1Affine) -> IcicleG1Affine {
        
        // Get coordinates as Fp elements
        let coords = point.coordinates().unwrap();
        let x = Self::fp_to_icicle_base(coords.x());
        let y = Self::fp_to_icicle_base(coords.y());
        
        IcicleG1Affine { x, y }
    }
    
    /// Convert slice of G1Affine points
    pub fn g1_affine_slice_to_icicle_vec(points: &[G1Affine]) -> Vec<IcicleG1Affine> {
        points.iter().map(Self::g1_affine_to_icicle).collect()
    }
    
    /// Convert ICICLE BaseField back to midnight Fp
    #[inline]
    fn icicle_base_to_fp(icicle: &IcicleBaseField) -> Option<Fp> {
        let bytes = icicle.to_bytes_le();
        let byte_array: [u8; 48] = bytes.as_slice().try_into().ok()?;
        Fp::from_bytes_le(&byte_array).into_option()
    }
    
    /// Convert ICICLE G1Projective back to midnight G1Projective
    /// Converts to affine first (ICICLE provides to_affine), then reconstructs point
    pub fn icicle_to_g1_projective(icicle: &IcicleG1Projective) -> G1Projective {
        // Convert to affine coordinates
        let affine = icicle.to_affine();
        
        // Extract x, y and convert to midnight Fp
        let x = Self::icicle_base_to_fp(&affine.x)
            .expect("Invalid x coordinate from ICICLE");
        let y = Self::icicle_base_to_fp(&affine.y)
            .expect("Invalid y coordinate from ICICLE");
        
        // Reconstruct midnight affine point (from_xy validates the point is on curve)
        let midnight_affine = G1Affine::from_xy(x, y)
            .into_option()
            .expect("Invalid point from ICICLE");
        
        // Convert to projective
        G1Projective::from(midnight_affine)
    }
    
    // =========================================================================
    // G2 Conversions
    // =========================================================================
    // G2 uses the quadratic extension field Fp2 = Fp[u]/(u^2 + 1)
    // Each Fp2 element has two Fp components: c0 + c1*u
    // G2 point coordinates are in Fp2, so x = (x.c0, x.c1), y = (y.c0, y.c1)
    //
    // ICICLE uses little-endian byte order for field elements
    // midnight-curves uses blst which uses big-endian byte order
    
    /// Convert midnight Fp2 to ICICLE G2BaseField
    #[inline]
    fn fp2_to_icicle_g2_base(fp2: &Fp2) -> IcicleG2BaseField {
        // Fp2 has c0 and c1 components, each is an Fp (48 bytes little-endian via to_repr)
        let c0_bytes = fp2.c0().to_repr();
        let c1_bytes = fp2.c1().to_repr();
        
        // ICICLE expects 96 bytes: [c0 (48 bytes LE)][c1 (48 bytes LE)]
        let mut bytes = [0u8; 96];
        bytes[..48].copy_from_slice(c0_bytes.as_ref());
        bytes[48..].copy_from_slice(c1_bytes.as_ref());
        
        IcicleG2BaseField::from_bytes_le(&bytes)
    }
    
    /// Convert ICICLE G2BaseField back to midnight Fp2
    #[inline]
    fn icicle_g2_base_to_fp2(icicle: &IcicleG2BaseField) -> Option<Fp2> {
        let bytes = icicle.to_bytes_le();
        if bytes.len() != 96 {
            return None;
        }
        
        // Extract c0 and c1 (each 48 bytes, little-endian)
        let c0_bytes: [u8; 48] = bytes[..48].try_into().ok()?;
        let c1_bytes: [u8; 48] = bytes[48..96].try_into().ok()?;
        
        let c0 = Fp::from_bytes_le(&c0_bytes).into_option()?;
        let c1 = Fp::from_bytes_le(&c1_bytes).into_option()?;
        
        Some(Fp2::new(c0, c1))
    }

    /// Convert midnight G2Affine to ICICLE G2Affine
    #[inline]
    pub fn g2_affine_to_icicle(point: &G2Affine) -> IcicleG2Affine {
        // Get x and y coordinates as Fp2
        let coords = point.coordinates().unwrap();
        let x = Self::fp2_to_icicle_g2_base(coords.x());
        let y = Self::fp2_to_icicle_g2_base(coords.y());
        
        IcicleG2Affine { x, y }
    }
    
    /// Convert slice of G2Affine points
    pub fn g2_affine_slice_to_icicle_vec(points: &[G2Affine]) -> Vec<IcicleG2Affine> {
        points.iter().map(Self::g2_affine_to_icicle).collect()
    }
    
    /// Convert ICICLE G2Projective back to midnight G2Projective
    pub fn icicle_to_g2_projective(icicle: &IcicleG2Projective) -> G2Projective {
        // Convert to affine coordinates
        let affine = icicle.to_affine();
        
        // Extract x, y and convert to midnight Fp2
        let x = Self::icicle_g2_base_to_fp2(&affine.x)
            .expect("Invalid x coordinate from ICICLE");
        let y = Self::icicle_g2_base_to_fp2(&affine.y)
            .expect("Invalid y coordinate from ICICLE");
        
        // Reconstruct midnight affine point (from_xy validates the point is on curve)
        let midnight_affine = G2Affine::from_xy(x, y)
            .into_option()
            .expect("Invalid G2 point from ICICLE - point not on curve");
        
        // Convert to projective
        G2Projective::from(midnight_affine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[cfg(feature = "gpu")]
    #[test]
    fn test_scalar_conversion() {
        use ff::Field;
        let scalar = Scalar::ONE;
        let icicle = TypeConverter::scalar_to_icicle(&scalar);
        // Can't easily test roundtrip without implementing reverse conversion
        let _ = icicle; // Just verify it compiles and runs
    }
    
    #[cfg(feature = "gpu")]
    #[test]
    fn test_point_conversion() {
        use group::prime::PrimeCurveAffine;
        let point = G1Affine::generator();
        let icicle = TypeConverter::g1_affine_to_icicle(&point);
        let _ = icicle; // Just verify it compiles and runs
    }
}
