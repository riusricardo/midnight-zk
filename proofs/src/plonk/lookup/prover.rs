use std::{collections::HashMap, hash::Hash, iter};

use ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use group::ff::BatchInvert;
use rand_core::{CryptoRng, RngCore};

use super::{
    super::{circuit::Expression, Error, ProvingKey},
    Argument,
};
use crate::{
    plonk::evaluation::evaluate,
    poly::{
        commitment::PolynomialCommitmentScheme, Coeff, EvaluationDomain, LagrangeCoeff, Polynomial,
        ProverQuery, Rotation,
    },
    transcript::{Hashable, Transcript},
    utils::arithmetic::{eval_polynomial, parallelize},
};

#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct Permuted<F: PrimeField> {
    compressed_input_expression: Polynomial<F, LagrangeCoeff>,
    permuted_input_expression: Polynomial<F, LagrangeCoeff>,
    permuted_input_poly: Polynomial<F, Coeff>,
    compressed_table_expression: Polynomial<F, LagrangeCoeff>,
    permuted_table_expression: Polynomial<F, LagrangeCoeff>,
    permuted_table_poly: Polynomial<F, Coeff>,
}

#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct Committed<F: PrimeField> {
    pub(crate) permuted_input_poly: Polynomial<F, Coeff>,
    pub(crate) permuted_table_poly: Polynomial<F, Coeff>,
    pub(crate) product_poly: Polynomial<F, Coeff>,
}

/// Intermediate state after computing the product polynomial but before commitment.
/// This enables batch committing multiple lookup products together.
#[cfg_attr(feature = "bench-internal", derive(Clone))]
#[derive(Debug)]
pub(crate) struct ProductComputed<F: PrimeField> {
    pub(crate) permuted_input_poly: Polynomial<F, Coeff>,
    pub(crate) permuted_table_poly: Polynomial<F, Coeff>,
    /// Product polynomial in Lagrange form (ready for commitment)
    pub(crate) product_poly_lagrange: Polynomial<F, LagrangeCoeff>,
}

pub(crate) struct Evaluated<F: PrimeField> {
    constructed: Committed<F>,
}

impl<F: WithSmallOrderMulGroup<3> + Ord + Hash> Argument<F> {
    /// Given a Lookup with input expressions [A_0, A_1, ..., A_{m-1}] and table
    /// expressions [S_0, S_1, ..., S_{m-1}], this method
    /// - constructs A_compressed = \theta^{m-1} A_0 + theta^{m-2} A_1 + ... +
    ///   \theta A_{m-2} + A_{m-1} and S_compressed = \theta^{m-1} S_0 +
    ///   theta^{m-2} S_1 + ... + \theta S_{m-2} + S_{m-1},
    /// - permutes A_compressed and S_compressed using permute_expression_pair()
    ///   helper, obtaining A' and S', and
    /// - constructs `Permuted<C>` struct using permuted_input_value = A', and
    ///   permuted_table_expression = S'.
    ///
    /// The `Permuted<C>` struct is used to update the Lookup, and is then
    /// returned.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit_permuted<
        'a,
        'params: 'a,
        CS: PolynomialCommitmentScheme<F>,
        R: RngCore,
        T: Transcript,
    >(
        &self,
        pk: &ProvingKey<F, CS>,
        params: &'params CS::Parameters,
        domain: &EvaluationDomain<F>,
        theta: F,
        advice_values: &'a [Polynomial<F, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<F, LagrangeCoeff>],
        instance_values: &'a [Polynomial<F, LagrangeCoeff>],
        challenges: &'a [F],
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Permuted<F>, Error>
    where
        F: FromUniformBytes<64>,
        CS::Commitment: Hashable<T::Hash>,
    {
        // Closure to get values of expressions and compress them
        let compress_expressions = |expressions: &[Expression<F>]| {
            let compressed_expression = expressions
                .iter()
                .map(|expression| {
                    pk.vk.domain.lagrange_from_vec(evaluate(
                        expression,
                        domain.n as usize,
                        1,
                        fixed_values,
                        advice_values,
                        instance_values,
                        challenges,
                    ))
                })
                .fold(domain.empty_lagrange(), |acc, expression| {
                    acc * theta + &expression
                });
            compressed_expression
        };

        // Get values of input expressions involved in the lookup and compress them
        let compressed_input_expression = compress_expressions(&self.input_expressions);

        // Get values of table expressions involved in the lookup and compress them
        let compressed_table_expression = compress_expressions(&self.table_expressions);

        // Permute compressed (InputExpression, TableExpression) pair
        let (permuted_input_expression, permuted_table_expression) = permute_expression_pair(
            pk,
            domain,
            &mut rng,
            &compressed_input_expression,
            &compressed_table_expression,
        )?;

        // Closure to construct commitment to vector of values
        let commit_values = |values: &Polynomial<F, LagrangeCoeff>| {
            let poly = pk.vk.domain.lagrange_to_coeff(values.clone());
            let commitment = CS::commit_lagrange(params, values);
            (poly, commitment)
        };

        // Commit to permuted input expression
        let (permuted_input_poly, permuted_input_commitment) =
            commit_values(&permuted_input_expression);

        // Commit to permuted table expression
        let (permuted_table_poly, permuted_table_commitment) =
            commit_values(&permuted_table_expression);

        // Hash permuted input commitment
        transcript.write(&permuted_input_commitment)?;

        // Hash permuted table commitment
        transcript.write(&permuted_table_commitment)?;

        Ok(Permuted {
            compressed_input_expression,
            permuted_input_expression,
            permuted_input_poly,
            compressed_table_expression,
            permuted_table_expression,
            permuted_table_poly,
        })
    }
}

impl<F: WithSmallOrderMulGroup<3>> Permuted<F> {
    /// Compute the product polynomial without committing.
    /// This enables batch committing multiple lookup products together.
    /// 
    /// Returns `ProductComputed` which contains the product polynomial in Lagrange form.
    /// Call `ProductComputed::finalize` with a commitment to complete the process.
    pub(crate) fn compute_product<CS: PolynomialCommitmentScheme<F>>(
        self,
        pk: &ProvingKey<F, CS>,
        beta: F,
        gamma: F,
        mut rng: impl RngCore + CryptoRng,
    ) -> ProductComputed<F>
    where
        F: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
    {
        let blinding_factors = pk.vk.cs.blinding_factors();
        
        // Compute lookup product denominator
        let mut lookup_product = vec![F::ZERO; pk.vk.n() as usize];
        parallelize(&mut lookup_product, |lookup_product, start| {
            for ((lookup_product, permuted_input_value), permuted_table_value) in lookup_product
                .iter_mut()
                .zip(self.permuted_input_expression[start..].iter())
                .zip(self.permuted_table_expression[start..].iter())
            {
                *lookup_product = (beta + permuted_input_value) * &(gamma + permuted_table_value);
            }
        });

        // Batch invert denominators
        lookup_product.iter_mut().batch_invert();

        // Compute numerators
        parallelize(&mut lookup_product, |product, start| {
            for (i, product) in product.iter_mut().enumerate() {
                let i = i + start;
                *product *= &(self.compressed_input_expression[i] + &beta);
                *product *= &(self.compressed_table_expression[i] + &gamma);
            }
        });

        // Compute the product polynomial evaluations
        let z = iter::once(F::ONE)
            .chain(lookup_product)
            .scan(F::ONE, |state, cur| {
                *state *= &cur;
                Some(*state)
            })
            .take(pk.vk.n() as usize - blinding_factors)
            .chain((0..blinding_factors).map(|_| F::random(&mut rng)))
            .collect::<Vec<_>>();
        assert_eq!(z.len(), pk.vk.n() as usize);
        let z = pk.vk.domain.lagrange_from_vec(z);

        #[cfg(feature = "sanity-checks")]
        {
            let u = (pk.vk.n() as usize) - (blinding_factors + 1);
            assert_eq!(z[0], F::ONE);
            for i in 0..u {
                let mut left = z[i + 1];
                left *= &(beta + &self.permuted_input_expression[i]);
                left *= &(gamma + &self.permuted_table_expression[i]);

                let mut right = z[i];
                right *= &(self.compressed_input_expression[i] + &beta);
                right *= &(self.compressed_table_expression[i] + &gamma);

                assert_eq!(left, right);
            }
            assert_eq!(z[u], F::ONE);
        }

        ProductComputed {
            permuted_input_poly: self.permuted_input_poly,
            permuted_table_poly: self.permuted_table_poly,
            product_poly_lagrange: z,
        }
    }
}

impl<F: PrimeField> ProductComputed<F> {
    /// Finalize the lookup product with a pre-computed commitment.
    /// This is used when batch committing multiple lookups.
    pub(crate) fn finalize<CS: PolynomialCommitmentScheme<F>, T: Transcript>(
        self,
        pk: &ProvingKey<F, CS>,
        commitment: CS::Commitment,
        transcript: &mut T,
    ) -> Result<Committed<F>, Error>
    where
        F: WithSmallOrderMulGroup<3>,
        CS::Commitment: Hashable<T::Hash>,
    {
        // Convert to coefficient form
        let product_poly = pk.vk.domain.lagrange_to_coeff(self.product_poly_lagrange);
        
        // Write commitment to transcript
        transcript.write(&commitment)?;
        
        Ok(Committed {
            permuted_input_poly: self.permuted_input_poly,
            permuted_table_poly: self.permuted_table_poly,
            product_poly,
        })
    }
}

impl<F: WithSmallOrderMulGroup<3>> Committed<F> {
    pub(crate) fn evaluate<T: Transcript, CS: PolynomialCommitmentScheme<F>>(
        self,
        pk: &ProvingKey<F, CS>,
        x: F,
        transcript: &mut T,
    ) -> Result<Evaluated<F>, Error>
    where
        F: Hashable<T::Hash>,
    {
        let domain = &pk.vk.domain;
        let x_inv = domain.rotate_omega(x, Rotation::prev());
        let x_next = domain.rotate_omega(x, Rotation::next());

        let product_eval = eval_polynomial(&self.product_poly, x);
        let product_next_eval = eval_polynomial(&self.product_poly, x_next);
        let permuted_input_eval = eval_polynomial(&self.permuted_input_poly, x);
        let permuted_input_inv_eval = eval_polynomial(&self.permuted_input_poly, x_inv);
        let permuted_table_eval = eval_polynomial(&self.permuted_table_poly, x);

        // Hash each advice evaluation
        for eval in iter::empty()
            .chain(Some(product_eval))
            .chain(Some(product_next_eval))
            .chain(Some(permuted_input_eval))
            .chain(Some(permuted_input_inv_eval))
            .chain(Some(permuted_table_eval))
        {
            transcript.write(&eval)?;
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<F: WithSmallOrderMulGroup<3>> Evaluated<F> {
    pub(crate) fn open<'a, CS: PolynomialCommitmentScheme<F>>(
        &'a self,
        pk: &'a ProvingKey<F, CS>,
        x: F,
    ) -> impl Iterator<Item = ProverQuery<'a, F>> + Clone {
        let x_inv = pk.vk.domain.rotate_omega(x, Rotation::prev());
        let x_next = pk.vk.domain.rotate_omega(x, Rotation::next());

        iter::empty()
            // Open lookup product commitments at x
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.constructed.product_poly,
            }))
            // Open lookup input commitments at x
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.constructed.permuted_input_poly,
            }))
            // Open lookup table commitments at x
            .chain(Some(ProverQuery {
                point: x,
                poly: &self.constructed.permuted_table_poly,
            }))
            // Open lookup input commitments at x_inv
            .chain(Some(ProverQuery {
                point: x_inv,
                poly: &self.constructed.permuted_input_poly,
            }))
            // Open lookup product commitments at x_next
            .chain(Some(ProverQuery {
                point: x_next,
                poly: &self.constructed.product_poly,
            }))
    }
}

type ExpressionPair<F> = (Polynomial<F, LagrangeCoeff>, Polynomial<F, LagrangeCoeff>);

/// Given a vector of input values A and a vector of table values S,
/// this method permutes A and S to produce A' and S', such that:
/// - like values in A' are vertically adjacent to each other; and
/// - the first row in a sequence of like values in A' is the row that has the
///   corresponding value in S'.
///
/// This method returns (A', S') if no errors are encountered.
fn permute_expression_pair<F, CS: PolynomialCommitmentScheme<F>, R: RngCore>(
    pk: &ProvingKey<F, CS>,
    domain: &EvaluationDomain<F>,
    mut rng: R,
    input_expression: &Polynomial<F, LagrangeCoeff>,
    table_expression: &Polynomial<F, LagrangeCoeff>,
) -> Result<ExpressionPair<F>, Error>
where
    F: WithSmallOrderMulGroup<3> + Hash + Ord + FromUniformBytes<64>,
{
    let blinding_factors = pk.vk.cs.blinding_factors();
    let usable_rows = pk.vk.n() as usize - (blinding_factors + 1);

    let mut permuted_input_expression: Vec<F> = input_expression.to_vec();
    permuted_input_expression.truncate(usable_rows);

    // Sort input lookup expression values
    permuted_input_expression.sort();

    // A HashMap of each unique element in the table expression and its count
    let mut leftover_table_map = HashMap::<F, u32>::with_capacity(table_expression.len());
    table_expression.iter().take(usable_rows).for_each(|coeff| {
        *leftover_table_map.entry(*coeff).or_insert(0) += 1;
    });
    let mut permuted_table_coeffs = vec![F::ZERO; usable_rows];

    let mut repeated_input_rows = permuted_input_expression
        .iter()
        .zip(permuted_table_coeffs.iter_mut())
        .enumerate()
        .filter_map(|(row, (input_value, table_value))| {
            // If this is the first occurrence of `input_value` in the input expression
            if row == 0 || *input_value != permuted_input_expression[row - 1] {
                *table_value = *input_value;
                // Remove one instance of input_value from leftover_table_map
                if let Some(count) = leftover_table_map.get_mut(input_value) {
                    assert!(*count > 0);
                    *count -= 1;
                    None
                } else {
                    // Return error if input_value not found
                    Some(Err(Error::ConstraintSystemFailure))
                }
            // If input value is repeated
            } else {
                Some(Ok(row))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Populate permuted table at unfilled rows with leftover table elements
    for (coeff, count) in leftover_table_map.iter() {
        for _ in 0..*count {
            permuted_table_coeffs[repeated_input_rows.pop().unwrap()] = *coeff;
        }
    }
    assert!(repeated_input_rows.is_empty());

    permuted_input_expression.extend((0..(blinding_factors + 1)).map(|_| F::random(&mut rng)));
    permuted_table_coeffs.extend((0..(blinding_factors + 1)).map(|_| F::random(&mut rng)));
    assert_eq!(permuted_input_expression.len(), pk.vk.n() as usize);
    assert_eq!(permuted_table_coeffs.len(), pk.vk.n() as usize);

    #[cfg(feature = "sanity-checks")]
    {
        let mut last = None;
        for (a, b) in permuted_input_expression
            .iter()
            .zip(permuted_table_coeffs.iter())
            .take(usable_rows)
        {
            if *a != *b {
                assert_eq!(*a, last.unwrap());
            }
            last = Some(*a);
        }
    }

    Ok((
        domain.lagrange_from_vec(permuted_input_expression),
        domain.lagrange_from_vec(permuted_table_coeffs),
    ))
}
