use symbolica::atom::{Atom, AtomCore, AtomView, Symbol};
use symbolica::coefficient::CoefficientView;
use symbolica::domains::integer::{Integer, IntegerRing, Z};
use symbolica::domains::rational::Q;
use symbolica::domains::rational_polynomial::{RationalPolynomial, RationalPolynomialField};
use symbolica::symbol;
use symbolica::tensors::matrix::Matrix;
use symbolica::utils::resources;

use crate::error::OneLoopError;
use crate::masters::{MasterBasis, MasterIntegral, OneLoopMasters};
use crate::recurrence::{RecurrenceInput, ReductionOptions};
use crate::symbols::{S, validate_namespace};

mod ibp;

/// `Debug` so a `Result<Reduction, _>` can be unwrapped either way.
#[derive(Debug)]
pub struct Reduction {
    pub terms: Vec<(Atom, MasterIntegral)>,
}

impl Reduction {
    /// Reduce every coefficient to lowest terms
    pub fn simplify(mut self) -> Result<Self, OneLoopError> {
        for (coeff, _) in &mut self.terms {
            resources::checkpoint()?;
            *coeff = coeff.cancel();
        }
        resources::checkpoint()?;
        Ok(self)
    }
}

/// Suggested total-index ceiling for explicitly enabled work diagnostics.
/// Ordinary reduction has no such ceiling. Native recursion grows its stack
/// on the current thread and honors the enclosing RAM/time resource budget.
pub const MAX_TOTAL_INDEX: i32 = 32;

/// Reduce a one-loop integral family to the scalar masters `A0`/`B0`/`C0`/`D0`.
///
/// Returns an error for malformed input, an unsupported numerator, a resource
/// refusal, or a nonfinite result. Work-count diagnostics are disabled.
pub fn reduce(family: &RecurrenceInput) -> Result<Reduction, OneLoopError> {
    ReductionOptions::default().reduce(family)
}

impl ReductionOptions {
    pub(crate) fn validate_total(self, powers: &[i32], total: i128) -> Result<(), OneLoopError> {
        if let Some(limit) = self.diagnostic_max_total_index {
            if limit < 0 {
                return Err(OneLoopError::InvalidFamily(
                    "diagnostic_max_total_index must be nonnegative".into(),
                ));
            }
            if total > i128::from(limit) {
                return Err(OneLoopError::UnsupportedIndex {
                    found: powers.to_vec(),
                    max: limit,
                });
            }
        }
        Ok(())
    }

    /// Apply this execution policy to the ordinary recurrence implementation.
    pub fn reduce(self, family: &RecurrenceInput) -> Result<Reduction, OneLoopError> {
        resources::checkpoint()?;
        crate::record_usage();
        validate_family(family)?;
        if family.powers.iter().any(|&power| power < 0) {
            return Err(OneLoopError::InvalidFamily(format!(
                "unsupported propagator powers {:?}: recurrence indices must be nonnegative",
                family.powers
            )));
        }
        let total = family.powers.iter().map(|&power| i128::from(power)).sum();
        self.validate_total(&family.powers, total)?;
        check_numerator(&family.numerator, family.masses_squared.len())?;

        let merged = merge_coincident_lines(family)?;
        let family = merged.as_ref().unwrap_or(family);
        let reduction = if family.invariants.iter().any(|s| s.is_zero()) {
            reduce_regularized(family)?
        } else {
            reduce_core(family)?
        };
        if check_finite(&reduction).is_err() {
            // Preserve the existing exact fallback at the same degenerate point.
            if let Some(terms) = ibp::reduce_exact(family)? {
                let exact = Reduction { terms };
                check_finite(&exact)?;
                resources::checkpoint()?;
                return Ok(exact);
            }
        }
        check_finite(&reduction)?;
        resources::checkpoint()?;
        Ok(reduction)
    }
}

/// Grow only this thread's stack, retaining its ambient budget and Symbolica TLS.
fn with_stack<T>(work: impl FnOnce() -> Result<T, OneLoopError>) -> Result<T, OneLoopError> {
    resources::checkpoint()?;
    const RED_ZONE: usize = 1024 * 1024;
    const STACK_SIZE: usize = 8 * 1024 * 1024;
    if stacker::remaining_stack().is_none_or(|remaining| remaining < RED_ZONE) {
        resources::check_allocation(STACK_SIZE as u64)?;
    }
    let value = stacker::maybe_grow(RED_ZONE, STACK_SIZE, work)?;
    resources::checkpoint()?;
    Ok(value)
}

fn validate_family(family: &RecurrenceInput) -> Result<(), OneLoopError> {
    let invalid = |reason: String| Err(OneLoopError::InvalidFamily(reason));
    let n = family.masses_squared.len();
    if n == 0 {
        return invalid("no propagators".into());
    }
    if family.powers.len() != n {
        return invalid(format!(
            "{} exponents for {n} propagators",
            family.powers.len()
        ));
    }
    if family.invariants.len() != n * (n - 1) / 2 {
        return invalid(format!(
            "{} invariants for {n} propagators, expected {}",
            family.invariants.len(),
            n * (n - 1) / 2
        ));
    }
    for atom in family
        .masses_squared
        .iter()
        .chain(&family.invariants)
        .chain([&family.numerator])
    {
        validate_namespace(atom)?;
        if let Some(s) = atom
            .get_all_symbols(true)
            .iter()
            .find(|s| is_reserved_name(s.get_name()))
        {
            return invalid(format!(
                "`{}` is reserved for the reducer's own use",
                s.get_name()
            ));
        }
    }
    Ok(())
}

/// The reducer's scratch symbols. They are fixed global names, so an input
/// using one would silently be read as the reducer's own variable.
fn is_reserved_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("oneloopmaster::") else {
        return false;
    };
    let numbered = |prefix: &str| {
        rest.strip_prefix(prefix)
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    };
    rest == "reg_delta" || rest == "xll" || ["xq", "den", "routing_tmp_q"].into_iter().any(numbered)
}

/// Reject a numerator whose loop momentum is not a polynomial in `dot(k,k)`
/// and `dot(k,q_1..q_{N-1})` (`q_1..q_3` for a tadpole): anything else would
/// be carried into a coefficient as a constant, or, for a direction the family
/// lacks, projected against made-up Gram entries.
fn check_numerator(numerator: &Atom, n: usize) -> Result<(), OneLoopError> {
    let unsupported = |reason: String| Err(OneLoopError::UnsupportedNumerator { reason });
    let n_ext = if n == 1 { 3 } else { n - 1 };
    let (poly, vars) = loop_dots_to_vars(numerator, n_ext);
    if poly.contains_symbol(S.k) {
        return unsupported(format!(
            "loop momentum outside dot(k,k) and dot(k,q1..q{n_ext}): {numerator}"
        ));
    }
    let variables = vars.into_iter().map(Atom::var).collect::<Vec<_>>();
    validate_polynomial_exponents(&poly, &variables, "validating the loop numerator")
}

pub(crate) fn exponent_storage_error(
    operation: &str,
    value: impl std::fmt::Display,
) -> OneLoopError {
    OneLoopError::UnsupportedNumerator {
        reason: format!(
            "i32 exponent storage overflow while {operation}: {value} is not representable in this implementation's i32 exponent/index storage"
        ),
    }
}

pub(crate) fn checked_exponent_add(a: i32, b: i32, operation: &str) -> Result<i32, OneLoopError> {
    a.checked_add(b)
        .ok_or_else(|| exponent_storage_error(operation, i64::from(a) + i64::from(b)))
}

pub(crate) fn checked_index_subtract(
    a: i32,
    power: i32,
    operation: &str,
) -> Result<i32, OneLoopError> {
    a.checked_sub(power)
        .ok_or_else(|| exponent_storage_error(operation, i64::from(a) - i64::from(power)))
}

pub(crate) fn checked_nonnegative_exponent(
    exponent: AtomView<'_>,
    operation: &str,
) -> Result<i32, OneLoopError> {
    if let Ok(value) = i32::try_from(exponent)
        && value >= 0
    {
        return Ok(value);
    }
    let nonnegative_integer = match exponent {
        AtomView::Num(number) => match number.get_coeff_view() {
            CoefficientView::Natural(value, 1, 0, _) => value >= 0,
            CoefficientView::Large(real, imaginary) => {
                let real = real.to_rat();
                imaginary.is_zero() && real.is_integer() && !real.is_negative()
            }
            _ => false,
        },
        _ => false,
    };
    if nonnegative_integer {
        Err(exponent_storage_error(operation, exponent))
    } else {
        Err(OneLoopError::UnsupportedNumerator {
            reason: format!(
                "numerator must be polynomial in the loop scalar products: exponent {exponent} is not a nonnegative integer"
            ),
        })
    }
}

fn contains_polynomial_variable(expression: AtomView<'_>, variables: &[Atom]) -> bool {
    let mut found = false;
    expression.visitor(&mut |node| {
        found |= variables.iter().any(|variable| node == variable.as_view());
        !found
    });
    found
}

/// Check the exponent storage used by coefficient collection and `DotMono`.
/// Variables can be whole scalar-product atoms, rather than just symbols.
/// Ordinary Atom expansion precedes monomial inspection; no infallible
/// fixed-exponent polynomial conversion is used to validate its own input.
pub(crate) fn validate_polynomial_exponents(
    expression: &Atom,
    variables: &[Atom],
    operation: &str,
) -> Result<(), OneLoopError> {
    let mut error = None;
    expression.visitor(&mut |node| {
        if error.is_some() {
            return false;
        }
        if let AtomView::Pow(power) = node {
            let (base, exponent) = power.get_base_exp();
            if contains_polynomial_variable(exponent, variables) {
                error = Some(OneLoopError::UnsupportedNumerator {
                    reason: "numerator must be polynomial in the loop scalar products".into(),
                });
                return false;
            }
            if contains_polynomial_variable(base, variables)
                && let Err(failure) = checked_nonnegative_exponent(exponent, operation)
            {
                error = Some(failure);
                return false;
            }
        }
        true
    });
    if let Some(error) = error {
        return Err(error);
    }

    let expanded = expression.expand();
    let terms = match expanded.as_view() {
        AtomView::Add(sum) => sum.iter().collect::<Vec<_>>(),
        term => vec![term],
    };
    for term in terms {
        resources::checkpoint()?;
        let factors = match term {
            AtomView::Mul(product) => product.iter().collect::<Vec<_>>(),
            factor => vec![factor],
        };
        let mut degrees = vec![0i32; variables.len()];
        for factor in factors {
            let (base, exponent) = match factor {
                AtomView::Pow(power) => {
                    let (base, exponent) = power.get_base_exp();
                    if !contains_polynomial_variable(base, variables) {
                        continue;
                    }
                    (base, checked_nonnegative_exponent(exponent, operation)?)
                }
                factor => (factor, 1),
            };
            if let Some(index) = variables
                .iter()
                .position(|variable| variable.as_view() == base)
            {
                degrees[index] = checked_exponent_add(degrees[index], exponent, operation)?;
            } else if contains_polynomial_variable(factor, variables) {
                return Err(OneLoopError::UnsupportedNumerator {
                    reason: "numerator must be polynomial in the loop scalar products".into(),
                });
            }
        }
    }
    Ok(())
}

/// Merge lines that are the same denominator, `D_i^a D_j^b = D_i^(a+b)`. Equal
/// rows of the modified Cayley matrix mean equal masses, a zero invariant
/// between the two lines and equal invariants against every other line; a
/// scalar integral depends only on those, so the merge is exact. Unmerged, a
/// raised power divides by the vanishing `det(Y)` and the `delta` limit comes
/// out indeterminate (a dotted bubble at `p^2 = 0`, for instance). Not done
/// under a numerator, which sees the vectors: an on-shell leg is not zero.
fn merge_coincident_lines(
    family: &RecurrenceInput,
) -> Result<Option<RecurrenceInput>, OneLoopError> {
    let exps = &family.powers;
    if family.numerator != Atom::num(1) || exps.iter().all(|&e| e <= 1) {
        return Ok(None);
    }
    let n = exps.len();
    let masses: Vec<Atom> = family.masses_squared.clone();
    let invariants = &family.invariants;
    let y = modified_cayley(&masses, invariants);
    let same_row =
        |i: usize, j: usize| (y[i].iter().zip(&y[j])).all(|(a, b)| (a - b).expand().is_zero());
    // Each line's first line with an equal row; row equality is transitive.
    let rep: Vec<usize> = (0..n)
        .map(|j| (0..j).find(|&i| same_row(i, j)).unwrap_or(j))
        .collect();
    let keep: Vec<usize> = (0..n).filter(|&j| rep[j] == j).collect();
    if keep.len() == n {
        return Ok(None);
    }
    let lex = |i: usize, j: usize| invariants[i * n - i * (i + 1) / 2 + j - i - 1].clone();
    let mut out = family.clone();
    out.masses_squared = keep
        .iter()
        .map(|&i| family.masses_squared[i].clone())
        .collect();
    out.powers = keep
        .iter()
        .map(|&i| {
            (0..n).filter(|&j| rep[j] == i).try_fold(0i32, |sum, j| {
                sum.checked_add(exps[j]).ok_or_else(|| {
                    OneLoopError::InvalidFamily(
                        "a merged propagator index exceeds the i32 representation".into(),
                    )
                })
            })
        })
        .collect::<Result<_, _>>()?;
    out.invariants = keep
        .iter()
        .enumerate()
        .flat_map(|(a, &i)| keep[a + 1..].iter().map(move |&j| lex(i, j)))
        .collect();
    Ok(Some(out))
}

fn check_finite(reduction: &Reduction) -> Result<(), OneLoopError> {
    for (c, m) in &reduction.terms {
        let master = OneLoopMasters.symbol(m);
        if !c.is_finite() || !master.is_finite() {
            return Err(OneLoopError::NonFiniteResult {
                reason: format!("coefficient {c} of {master}"),
            });
        }
    }
    Ok(())
}

/// Substitute `delta -> 0` in every kinematic argument of a master.
fn master_at_zero(m: &MasterIntegral, delta: &Atom) -> MasterIntegral {
    let z = |a: &Atom| a.replace(delta.to_pattern()).with(Atom::Zero);
    match m {
        MasterIntegral::Tadpole { m_sq } => MasterIntegral::Tadpole { m_sq: z(m_sq) },
        MasterIntegral::Bubble { p_sq, m1_sq, m2_sq } => MasterIntegral::Bubble {
            p_sq: z(p_sq),
            m1_sq: z(m1_sq),
            m2_sq: z(m2_sq),
        },
        MasterIntegral::Triangle {
            p1_sq,
            p2_sq,
            p12_sq,
            m1_sq,
            m2_sq,
            m3_sq,
        } => MasterIntegral::Triangle {
            p1_sq: z(p1_sq),
            p2_sq: z(p2_sq),
            p12_sq: z(p12_sq),
            m1_sq: z(m1_sq),
            m2_sq: z(m2_sq),
            m3_sq: z(m3_sq),
        },
        MasterIntegral::Box {
            p1_sq,
            p2_sq,
            p3_sq,
            p4_sq,
            s,
            t,
            m1_sq,
            m2_sq,
            m3_sq,
            m4_sq,
        } => MasterIntegral::Box {
            p1_sq: z(p1_sq),
            p2_sq: z(p2_sq),
            p3_sq: z(p3_sq),
            p4_sq: z(p4_sq),
            s: z(s),
            t: z(t),
            m1_sq: z(m1_sq),
            m2_sq: z(m2_sq),
            m3_sq: z(m3_sq),
            m4_sq: z(m4_sq),
        },
    }
}

/// Off-shell regularization for on-shell massless legs
fn reduce_regularized(family: &RecurrenceInput) -> Result<Reduction, OneLoopError> {
    resources::checkpoint()?;
    let delta = Atom::var(symbol!("oneloopmaster::reg_delta"));
    let mut reg = family.clone();
    reg.invariants = family
        .invariants
        .iter()
        .map(|s| {
            if s.is_zero() {
                delta.clone()
            } else {
                s.clone()
            }
        })
        .collect();
    let reduced = reduce_core(&reg)?;
    let terms = reduced
        .terms
        .iter()
        .map(|(c, m)| {
            let c0 = c
                .together()
                .replace(delta.to_pattern())
                .with(Atom::Zero)
                .expand();
            (c0, master_at_zero(m, &delta))
        })
        .filter(|(c, _)| *c != Atom::Zero)
        .collect();
    resources::checkpoint()?;
    Ok(Reduction { terms })
}

fn reduce_core(family: &RecurrenceInput) -> Result<Reduction, OneLoopError> {
    resources::checkpoint()?;
    let inv = |i: usize| family.invariants.get(i).cloned().unwrap_or(Atom::Zero);
    let mass = |i: usize| family.masses_squared[i].clone();
    let exponents = &family.powers;

    // Every branch below indexes `invariants` by a hard-coded permutation of the C(n,2)
    // lexicographic pairwise slots, and `inv()` silently returns `Atom::Zero` past the end --
    // i.e. a short list is read as "these legs are on shell" rather than as an error. This
    // check used to guard only the N > 4 arm.
    {
        let n = family.masses_squared.len();
        let expected = n * (n - 1) / 2;
        assert_eq!(
            family.invariants.len(),
            expected,
            "an {n}-point family needs {expected} lexicographic pairwise invariants"
        );
    }

    let terms = match family.masses_squared.len() {
        1 => {
            let m_sq = mass(0);
            if family.numerator == Atom::num(1) {
                vec![(
                    tadpole_coefficient(exponents[0], &m_sq)?,
                    MasterIntegral::Tadpole { m_sq },
                )]
            } else {
                tadpole_numerator(&family.numerator, exponents[0], &m_sq)?
            }
        }
        2 => {
            let p_sq = inv(0);
            let m1_sq = mass(0);
            let m2_sq = mass(1);
            if family.numerator == Atom::num(1) {
                let m = [m1_sq, m2_sq];
                reduce_cayley(&modified_cayley(&m, &[p_sq]), &m, exponents)?
            } else {
                let (c_b0, c_a1, c_a2) = bubble_numerator(
                    &family.numerator,
                    exponents[0],
                    exponents[1],
                    &p_sq,
                    &m1_sq,
                    &m2_sq,
                )?;
                let mut terms = Vec::new();
                push_nonzero(
                    &mut terms,
                    c_b0,
                    MasterIntegral::Bubble {
                        p_sq,
                        m1_sq: m1_sq.clone(),
                        m2_sq: m2_sq.clone(),
                    },
                );
                push_nonzero(&mut terms, c_a1, MasterIntegral::Tadpole { m_sq: m1_sq });
                push_nonzero(&mut terms, c_a2, MasterIntegral::Tadpole { m_sq: m2_sq });
                terms
            }
        }
        3 => {
            if family.numerator == Atom::num(1) {
                // Triangle invariants (s1,s2,s3)
                let masses: Vec<Atom> = (0..3).map(mass).collect();
                let lex = [inv(0), inv(1), inv(2)];
                reduce_cayley(&modified_cayley(&masses, &lex), &masses, exponents)?
            } else {
                triangle_numerator(
                    &family.numerator,
                    exponents[0],
                    exponents[1],
                    exponents[2],
                    &inv(0),
                    &inv(2),
                    &inv(1),
                    &mass(0),
                    &mass(1),
                    &mass(2),
                )?
            }
        }
        4 => {
            if family.numerator == Atom::num(1) {
                // Box invariants (p1,p2,p3,p4,s,t)
                let masses: Vec<Atom> = (0..4).map(mass).collect();
                let lex = [inv(0), inv(1), inv(2), inv(3), inv(4), inv(5)];
                reduce_cayley(&modified_cayley(&masses, &lex), &masses, exponents)?
            } else {
                box_numerator(
                    &family.numerator,
                    exponents[0],
                    exponents[1],
                    exponents[2],
                    exponents[3],
                    &inv(0),
                    &inv(3),
                    &inv(5),
                    &inv(2),
                    &inv(1),
                    &inv(4),
                    &mass(0),
                    &mass(1),
                    &mass(2),
                    &mass(3),
                )?
            }
        }
        n => {
            // N > 4: van Neerven-Vermaseren / FJT-Tarasov reduction to boxes.
            let masses: Vec<Atom> = (0..n).map(mass).collect();
            if family.numerator != Atom::num(1) {
                ngon_numerator(&family.numerator, exponents, &family.invariants, &masses)?
            } else {
                let y = modified_cayley(&masses, &family.invariants);
                reduce_cayley(&y, &masses, exponents)?
            }
        }
    };

    resources::checkpoint()?;
    Ok(Reduction { terms })
}

fn tadpole_coefficient(exponent: i32, m_sq: &Atom) -> Result<Atom, OneLoopError> {
    if exponent <= 0 || *m_sq == Atom::Zero {
        return Ok(Atom::Zero);
    }
    let d = Atom::var(S.d);
    let mut coeff = Atom::num(1);
    for k in 1..i64::from(exponent) {
        resources::checkpoint()?;
        coeff = coeff * (&d - Atom::num(2 * k)) / (Atom::num(2 * k) * m_sq);
    }
    Ok(coeff)
}

fn push_nonzero(terms: &mut Vec<(Atom, MasterIntegral)>, coeff: Atom, master: MasterIntegral) {
    if coeff == Atom::Zero {
        return;
    }
    match terms.iter_mut().find(|(_, m)| *m == master) {
        Some(slot) => slot.0 = &slot.0 + &coeff,
        None => terms.push((coeff, master)),
    }
}

fn add_scaled(
    acc: &mut Vec<(Atom, MasterIntegral)>,
    coeff: &Atom,
    child: Vec<(Atom, MasterIntegral)>,
) {
    for (c, m) in child {
        let scaled = coeff * &c;
        if scaled == Atom::Zero {
            continue;
        }
        match acc.iter_mut().find(|pair| pair.1 == m) {
            Some(slot) => slot.0 = &slot.0 + &scaled,
            None => acc.push((scaled, m)),
        }
    }
}

// General symbolic determinant by cofactor expansion along the first row.
type Rp = RationalPolynomial<IntegerRing, u16>;

fn to_matrix(m: &[Vec<Atom>]) -> Matrix<RationalPolynomialField<IntegerRing, u16>> {
    let rows: Vec<Vec<Rp>> = m
        .iter()
        .map(|r| {
            r.iter()
                .map(|a| a.to_rational_polynomial::<_, _, u16>(&Q, &Z, None))
                .collect()
        })
        .collect();
    Matrix::from_nested_vec(rows, RationalPolynomialField::new(Z)).unwrap()
}

// Exact symbolic determinant
fn det(m: &[Vec<Atom>]) -> Atom {
    to_matrix(m).det().unwrap().to_expression()
}

// Inverse as an Atom matrix
fn matrix_inv(m: &[Vec<Atom>]) -> Vec<Vec<Atom>> {
    let inv = to_matrix(m).inv().unwrap();
    (0..inv.nrows())
        .map(|i| {
            (0..inv.ncols())
                .map(|j| inv[(i as u32, j as u32)].to_expression())
                .collect()
        })
        .collect()
}

// Modified Cayley matrix Y_ij = m_i^2 + m_j^2 - (r_i - r_j)^2 from the masses and the
// C(n,2) pairwise invariants (r_i - r_j)^2 in lexicographic (i<j) order.
fn modified_cayley(masses: &[Atom], pairwise: &[Atom]) -> Vec<Vec<Atom>> {
    let n = masses.len();
    let idx = |a: usize, b: usize| a * n - a * (a + 1) / 2 + (b - a - 1);
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        Atom::num(2) * &masses[i]
                    } else {
                        let (a, b) = (i.min(j), i.max(j));
                        &masses[i] + &masses[j] - &pairwise[idx(a, b)]
                    }
                })
                .collect()
        })
        .collect()
}

fn delete_row_col(m: &[Vec<Atom>], skip_row: usize, skip_col: usize) -> Vec<Vec<Atom>> {
    let n = m.len();
    let cols: Vec<usize> = (0..n).filter(|&c| c != skip_col).collect();
    (0..n)
        .filter(|&r| r != skip_row)
        .map(|r| cols.iter().map(|&c| m[r][c].clone()).collect())
        .collect()
}

// Rational null-space basis of `m`
fn nullspace(m: &[Vec<Atom>]) -> Vec<Vec<Atom>> {
    let rows = m.len();
    let cols = m.first().map_or(0, Vec::len);
    let mut a = m.to_vec();
    let mut pivot_of_col: Vec<Option<usize>> = vec![None; cols];
    let mut r = 0;
    for col in 0..cols {
        if r >= rows {
            break;
        }
        let Some(piv) = (r..rows).find(|&i| a[i][col] != Atom::Zero) else {
            continue;
        };
        a.swap(r, piv);
        let inv = Atom::num(1) / &a[r][col];
        for cell in a[r].iter_mut() {
            *cell = &*cell * &inv;
        }
        let pivot_row = a[r].clone();
        for (i, row) in a.iter_mut().enumerate() {
            if i != r && row[col] != Atom::Zero {
                let f = row[col].clone();
                for (cell, p) in row.iter_mut().zip(&pivot_row) {
                    *cell = &*cell - &(&f * p);
                }
            }
        }
        pivot_of_col[col] = Some(r);
        r += 1;
    }
    (0..cols)
        .filter(|&free| pivot_of_col[free].is_none())
        .map(|free| {
            let mut v = vec![Atom::Zero; cols];
            v[free] = Atom::num(1);
            for (col, &pr) in pivot_of_col.iter().enumerate() {
                if let Some(pr) = pr {
                    v[col] = -&a[pr][free];
                }
            }
            v
        })
        .collect()
}

// Degenerate reduction for det(Y) = 0
fn degenerate_coeffs(y: &[Vec<Atom>]) -> Vec<Atom> {
    let n = y.len();
    let mut yb = vec![vec![Atom::Zero; n + 1]; n + 1];
    for i in 0..n {
        for j in 0..n {
            yb[i][j] = y[i][j].clone();
        }
        yb[i][n] = Atom::num(1);
        yb[n][i] = Atom::num(1);
    }

    for v in nullspace(&yb) {
        if v[n] != Atom::Zero {
            return (0..n).map(|i| &v[i] / &v[n]).collect();
        }
    }
    panic!("degenerate Cayley reduction: exceptional kinematics");
}

// van Neerven-Vermaseren coefficients for I_N = sum_i c_i I_{N-1}^(i) (scalar, d=4):
// c_i = (YB^{-1})_{i0} / (YB^{-1})_{00}, YB the bordered Cayley matrix.
fn high_point_coeffs(y: &[Vec<Atom>]) -> Vec<Atom> {
    let det_y = det(y);
    if det_y == Atom::Zero {
        return degenerate_coeffs(y);
    }
    let n = y.len();
    let mut yb = vec![vec![Atom::Zero; n + 1]; n + 1];
    for i in 1..=n {
        yb[0][i] = Atom::num(1);
        yb[i][0] = Atom::num(1);
        for j in 1..=n {
            yb[i][j] = y[i - 1][j - 1].clone();
        }
    }
    (1..=n)
        .map(|col| {
            let minor = det(&delete_row_col(&yb, 0, col));
            let cofactor = if col % 2 == 0 { minor } else { -minor };
            cofactor / &det_y
        })
        .collect()
}

// Reduce an N-point (N>=4) from its Cayley matrix, masses, and powers.
fn emit_master(y: &[Vec<Atom>], masses: &[Atom]) -> Vec<(Atom, MasterIntegral)> {
    let pair = |a: usize, b: usize| (&masses[a] + &masses[b] - &y[a][b]).expand();
    let m = |i: usize| masses[i].clone();
    let master = match y.len() {
        2 => MasterIntegral::Bubble {
            p_sq: pair(0, 1),
            m1_sq: m(0),
            m2_sq: m(1),
        },
        3 => MasterIntegral::Triangle {
            p1_sq: pair(0, 1),
            p2_sq: pair(1, 2),
            p12_sq: pair(0, 2),
            m1_sq: m(0),
            m2_sq: m(1),
            m3_sq: m(2),
        },
        4 => MasterIntegral::Box {
            p1_sq: pair(0, 1),
            p2_sq: pair(1, 2),
            p3_sq: pair(2, 3),
            p4_sq: pair(0, 3),
            s: pair(0, 2),
            t: pair(1, 3),
            m1_sq: m(0),
            m2_sq: m(1),
            m3_sq: m(2),
            m4_sq: m(3),
        },
        _ => unreachable!("emit_master: scalar leaf must be a bubble/triangle/box"),
    };
    vec![(Atom::num(1), master)]
}

fn reduce_cayley(
    y: &[Vec<Atom>],
    masses: &[Atom],
    exponents: &[i32],
) -> Result<Masters, OneLoopError> {
    with_stack(|| reduce_cayley_inner(y, masses, exponents))
}

fn reduce_cayley_inner(
    y: &[Vec<Atom>],
    masses: &[Atom],
    exponents: &[i32],
) -> Result<Masters, OneLoopError> {
    let drop = |i: usize, v: &[Atom]| -> Vec<Atom> {
        v.iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, x)| x.clone())
            .collect()
    };
    let n = y.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        let c = tadpole_coefficient(exponents[0], &masses[0])?;
        return Ok(if c == Atom::Zero {
            Vec::new()
        } else {
            vec![(
                c,
                MasterIntegral::Tadpole {
                    m_sq: masses[0].clone(),
                },
            )]
        });
    }
    if exponents.contains(&0) {
        let z = exponents.iter().position(|&e| e == 0).unwrap();
        let sub_exp: Vec<i32> = exponents
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != z)
            .map(|(_, &e)| e)
            .collect();
        return reduce_cayley(&delete_row_col(y, z, z), &drop(z, masses), &sub_exp);
    }
    if exponents.iter().all(|&e| e == 1) {
        if n <= 4 {
            return Ok(emit_master(y, masses));
        }
        let ones = vec![1; n - 1];
        let mut terms = Vec::new();
        for (i, c_i) in high_point_coeffs(y).iter().enumerate() {
            add_scaled(
                &mut terms,
                c_i,
                reduce_cayley(&delete_row_col(y, i, i), &drop(i, masses), &ones)?,
            );
        }
        return Ok(terms);
    }

    let det_y = det(y);
    if det_y == Atom::Zero {
        let mut terms = Vec::new();
        for (l, cl) in degenerate_coeffs(y).iter().enumerate() {
            if exponents[l] == 0 {
                continue;
            }
            let mut a2 = exponents.to_vec();
            a2[l] -= 1;
            add_scaled(&mut terms, cl, reduce_cayley(y, masses, &a2)?);
        }
        return Ok(terms);
    }

    // dotted: FJT/Tarasov index-lowering
    let mut k = 0;
    for i in 1..n {
        if exponents[i] > exponents[k] {
            k = i;
        }
    }
    let mut a = exponents.to_vec();
    a[k] -= 1;
    let total: Atom = a.iter().map(|&power| Atom::num(i64::from(power))).sum();
    // inv[k][i] = adj[k][i]/det(y)
    let inv = matrix_inv(y);
    let den = Atom::num(i64::from(a[k]));

    let d = Atom::var(S.d);
    let mut diag = Atom::Zero;
    for i in 0..n {
        let factor = &d - &total - Atom::num(i64::from(a[i]));
        diag += &inv[k][i] * &factor;
    }
    let mut acc = Vec::new();
    add_scaled(&mut acc, &(diag / &den), reduce_cayley(y, masses, &a)?);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let num = Atom::num(i64::from(-a[j])) * &inv[k][i];
            let mut ch = a.clone();
            ch[j] = ch[j].checked_add(1).ok_or_else(|| {
                OneLoopError::InvalidFamily(
                    "a recurrence propagator index exceeds the i32 representation".into(),
                )
            })?;
            ch[i] -= 1;
            add_scaled(&mut acc, &(num / &den), reduce_cayley(y, masses, &ch)?);
        }
    }
    Ok(acc)
}

type DotMono = (Vec<i32>, Atom);

fn dot_ll() -> Atom {
    crate::symbols::scalar_product(&(Atom::var(S.k)), &(Atom::var(S.k)))
}
fn dot_lq(j: usize) -> Atom {
    let q = symbol!(format!("oneloopmaster::q{}", j + 1));
    crate::symbols::scalar_product(&(Atom::var(S.k)), &(Atom::var(q)))
}

/// Replace `dot(k,k)` and `dot(k,q_1..q_{n_ext})` by the variables `xll`,
/// `xq1..`, returned alongside.
fn loop_dots_to_vars(numerator: &Atom, n_ext: usize) -> (Atom, Vec<Symbol>) {
    let xll = symbol!("oneloopmaster::xll");
    let mut vars = vec![xll];
    let mut n = numerator
        .replace(dot_ll().to_pattern())
        .with(Atom::var(xll));
    for a in 0..n_ext {
        let xq = symbol!(format!("oneloopmaster::xq{}", a + 1));
        vars.push(xq);
        n = n.replace(dot_lq(a).to_pattern()).with(Atom::var(xq));
    }
    (n, vars)
}

fn numerator_to_monos(numerator: &Atom, n_ext: usize) -> Result<Vec<DotMono>, OneLoopError> {
    let (n, vars) = loop_dots_to_vars(numerator, n_ext);
    extract_monomials(&n, &vars)
}

// Monomials of a polynomial in `vars`
fn extract_monomials(expr: &Atom, vars: &[Symbol]) -> Result<Vec<DotMono>, OneLoopError> {
    let variables = vars.iter().copied().map(Atom::var).collect::<Vec<_>>();
    validate_polynomial_exponents(expr, &variables, "extracting numerator monomials")?;
    fn rec(
        expr: &Atom,
        vars: &[Symbol],
        idx: usize,
        exps: &mut Vec<i32>,
        out: &mut Vec<DotMono>,
    ) -> Result<(), OneLoopError> {
        with_stack(|| {
            if idx == vars.len() {
                if *expr != Atom::Zero {
                    out.push((exps.clone(), expr.clone()));
                }
                return Ok(());
            }
            let v = vars[idx];
            let vatom = Atom::var(v);
            // Peel off powers of `v` one at a time via Taylor coefficients:
            //   coeff of v^c = (1/c!) d^c/dv^c expr |_{v=0}
            let mut deriv = expr.clone();
            let mut factorial = Integer::from(1);
            let mut c = 0i32;
            loop {
                resources::checkpoint()?;
                // coefficient of v^c is (deriv with v->0) / factorial
                let at_zero = deriv.replace(vatom.to_pattern()).with(Atom::Zero);
                if at_zero != Atom::Zero {
                    let coeff = at_zero / Atom::num(factorial.clone());
                    exps[idx] = c;
                    rec(&coeff, vars, idx + 1, exps, out)?;
                }
                // step to next power
                let next = deriv.derivative(v);
                if next == Atom::Zero {
                    break;
                }
                deriv = next;
                c = checked_exponent_add(c, 1, "advancing the coefficient-extraction derivative")?;
                factorial *= Integer::from(c);
            }
            exps[idx] = 0;
            Ok(())
        })
    }
    let mut out = Vec::new();
    let mut exps = vec![0i32; vars.len()];
    rec(expr, vars, 0, &mut exps, &mut out)?;
    Ok(out)
}

// A direction in external-momentum space: integer coefficients over (q1,...,q_{n_ext}).
type Dir = Vec<i32>;

// Base Gram closure: (a,b) in {0,1,2} -> q_a . q_b at a kinematic point.
type GramFn = Box<dyn Fn(usize, usize) -> Atom>;
// A reduction result: a list of (coefficient, master integral).
type Masters = Vec<(Atom, MasterIntegral)>;
// Scalar reducer for a topology: propagator exponents -> masters.
type ScalarFn = Box<dyn Fn(&[i32]) -> Result<Masters, OneLoopError>>;
// Pinch routine: (kept lines, their exponents, residual dot-polynomial) -> masters.
type PinchFn = Box<dyn Fn(&[usize], &[i32], &[DotMono]) -> Result<Masters, OneLoopError>>;

// A topology level for arbitrary-rank numerator reduction.
struct Topo {
    // Reducible directions (RSP span), as coefficient vectors over (q1,q2,q3).
    // dot(k, d) for d in this span is reducible; everything else is an ISP.
    rsp_dirs: Vec<Dir>,
    // Base Gram q_a . q_b for a,b in {0,1,2} (q1,q2,q3) at this kinematic point.
    base_gram: GramFn,
    // RSP rule for dot(k,k): linear in den symbols.
    rule_ll: Atom,
    // RSP rule for dot(k, d) for each reducible direction d (same order as rsp_dirs).
    rule_lq: Vec<Atom>,
    // den symbols, one per surviving line, in order.
    den_syms: Vec<Symbol>,
    // Propagator exponents a_i of the surviving lines.
    a: Vec<i32>,
    // r_i: shift of each surviving line as a direction over (q1,q2,q3).
    r_coeffs: Vec<Dir>,
    // Mass^2 of each surviving line.
    masses: Vec<Atom>,
    // Reduce a pure propagator-shift (all exponents) of this topology to masters.
    scalar: ScalarFn,
    // Reduce a residual numerator on the sub-topology that KEEPS the given lines/exponents.
    pinch: PinchFn,
}

impl Topo {
    // Gram dot of two directions u, v over the external momenta: sum u_a v_b (q_a.q_b).
    fn dir_dot(&self, u: &Dir, v: &Dir) -> Atom {
        dir_dot_with(&*self.base_gram, u, v)
    }
}

// Solve the symbolic Gram system G * alpha = rhs (G_{ij} = rsp_dirs[i].rsp_dirs[j]) by
// Cramer's rule with the exact symbolic determinant, so det cancels downstream.
fn gram_solve(topo: &Topo, rhs: &[Atom]) -> Vec<Atom> {
    let dirs = &topo.rsp_dirs;
    let n = dirs.len();
    if n == 0 {
        return vec![];
    }
    let g: Vec<Vec<Atom>> = (0..n)
        .map(|i| (0..n).map(|j| topo.dir_dot(&dirs[i], &dirs[j])).collect())
        .collect();
    gram_solve_matrix(&g, rhs)
}

fn gram_solve_matrix(g: &[Vec<Atom>], rhs: &[Atom]) -> Vec<Atom> {
    let n = g.len();
    let b: Vec<Vec<Atom>> = rhs.iter().map(|x| vec![x.clone()]).collect();
    if let Ok(sol) = to_matrix(g).solve(&to_matrix(&b)) {
        return (0..n).map(|i| sol[(i as u32, 0)].to_expression()).collect();
    }
    let subset = independent_gram_subset(g);
    let sub_g: Vec<Vec<Atom>> = subset
        .iter()
        .map(|&i| subset.iter().map(|&j| g[i][j].clone()).collect())
        .collect();
    let sub_b: Vec<Vec<Atom>> = subset.iter().map(|&i| vec![rhs[i].clone()]).collect();
    let sub_sol = to_matrix(&sub_g)
        .solve(&to_matrix(&sub_b))
        .expect("maximal independent sub-Gram is invertible by construction");
    let mut c = vec![Atom::Zero; n];
    for (k, &i) in subset.iter().enumerate() {
        c[i] = sub_sol[(k as u32, 0)].to_expression();
    }
    c
}

// Greedy maximal subset of indices whose principal sub-Gram is non-singular
fn independent_gram_subset(g: &[Vec<Atom>]) -> Vec<usize> {
    let n = g.len();
    let mut chosen: Vec<usize> = Vec::new();
    for i in 0..n {
        let cand: Vec<usize> = chosen.iter().copied().chain(std::iter::once(i)).collect();
        let sub: Vec<Vec<Atom>> = cand
            .iter()
            .map(|&a| cand.iter().map(|&b| g[a][b].clone()).collect())
            .collect();
        if det(&sub) != Atom::Zero {
            chosen = cand;
        }
    }
    chosen
}

// All perfect matchings (pairings) of 0..m (m even).  Each pairing is a Vec of
// (i,j) index pairs.
fn pairings(items: &[usize]) -> Result<Vec<Vec<(usize, usize)>>, OneLoopError> {
    with_stack(|| {
        if items.is_empty() {
            return Ok(vec![vec![]]);
        }
        let first = items[0];
        let mut out = Vec::new();
        for i in 1..items.len() {
            let mut rest = Vec::with_capacity(items.len() - 2);
            for (j, &it) in items.iter().enumerate() {
                if j != 0 && j != i {
                    rest.push(it);
                }
            }
            for mut sub in pairings(&rest)? {
                resources::checkpoint()?;
                sub.insert(0, (first, items[i]));
                out.push(sub);
            }
        }
        Ok(out)
    })
}

// Step 1: PV transverse average.  Split each loop dot into a reducible part
// (par = sum alpha_ab dot(k,q_b)) plus an ISP part xi; average even xi products over
// pairings of the transverse metric in n_t = d-|rsp| dims.
#[allow(clippy::assign_op_pattern)]
fn isp_project(topo: &Topo, monos: &[DotMono]) -> Result<Vec<DotMono>, OneLoopError> {
    resources::checkpoint()?;
    let d = Atom::var(S.d);
    let n_rsp = topo.rsp_dirs.len();
    let n_out = n_rsp + 1; // output DotMono length: l.l slot + one per reducible direction
    let n_ext = topo
        .rsp_dirs
        .first()
        .map(Vec::len)
        .or_else(|| monos.first().map(|m| m.0.len() - 1))
        .unwrap_or(0); // input external dimension
    let nt = &d - Atom::num(n_rsp as i64);
    let unit = |a: usize| -> Dir {
        let mut e = vec![0i32; n_ext];
        e[a] = 1;
        e
    };

    // Reducible-span coeffs of v: exact integer decomposition (finite even for a singular
    // Gram), else the oblique Gram inverse (only reached for genuine ISPs on full-rank subs).
    let alpha_of = |v: &Dir| -> Vec<Atom> {
        if let Some(c) = span_coeffs(&topo.rsp_dirs, v) {
            return c;
        }
        let rhs: Vec<Atom> = topo.rsp_dirs.iter().map(|r| topo.dir_dot(r, v)).collect();
        gram_solve(topo, &rhs)
    };
    // Transverse metric (u.v)_perp = u.v - (rsp.u)^T G^{-1} (rsp.v).
    let perp_metric = |u: &Dir, v: &Dir| -> Atom {
        let alpha = alpha_of(u);
        let mut par = Atom::Zero;
        for (i, r) in topo.rsp_dirs.iter().enumerate() {
            par += &alpha[i] * topo.dir_dot(r, v);
        }
        topo.dir_dot(u, v) - par
    };
    // par(v): coefficients on the reducible loop dots dot(k, rsp_dirs[j]).
    let par_of = |v: &Dir| -> Vec<Atom> { alpha_of(v) };
    // A unit direction is reducible iff it lies in the integer span of rsp_dirs.
    let is_reducible = |v: &Dir| -> bool { dir_in_span(&topo.rsp_dirs, v) };

    // Only build the transverse metric (needs the Gram inverse) when a real ISP is present.
    let any_isp = monos
        .iter()
        .any(|(exps, _)| (0..n_ext).any(|a| exps[a + 1] > 0 && !is_reducible(&unit(a))));

    // L2perp = dot(k,k) - sum_{i,j} dot(k,rsp_i) Ginv_ij dot(k,rsp_j), keyed in the
    // OUTPUT convention (slot j+1 = dot(k, rsp_dirs[j])).
    let l2perp: Vec<DotMono> = if !any_isp {
        Vec::new()
    } else {
        let mut ll_slot = vec![0i32; n_out];
        ll_slot[0] = 1;
        let mut out: Vec<DotMono> = vec![(ll_slot, Atom::num(1))];
        for i in 0..n_rsp {
            let mut e = vec![Atom::Zero; n_rsp];
            e[i] = Atom::num(1);
            let ginv_col = gram_solve(topo, &e); // G^{-1} column i
            for (j, gij) in ginv_col.iter().enumerate() {
                let mut ek = vec![0i32; n_out];
                ek[i + 1] += 1;
                ek[j + 1] += 1;
                out.push((ek, Atom::num(-1) * gij));
            }
        }
        out
    };

    let mut result: Vec<DotMono> = Vec::new();
    for (exps, coeff) in monos {
        resources::checkpoint()?;
        let ll_pow = exps[0];
        // factors: the multiset of unit external directions from the q-dots.
        let mut factors: Vec<Dir> = Vec::new();
        for a in 0..n_ext {
            for _ in 0..exps[a + 1] {
                factors.push(unit(a));
            }
        }
        // base poly: ll^{ll_pow} (in output convention) times coeff.
        let mut base_e = vec![0i32; n_out];
        base_e[0] = ll_pow;
        let base_mono: Vec<DotMono> = vec![(base_e, coeff.clone())];
        // term state: (par-poly so far, xi-multiset of ISP directions).
        let mut terms: Vec<(Vec<DotMono>, Vec<Dir>)> = vec![(base_mono, Vec::new())];
        for fdir in &factors {
            let reducible = is_reducible(fdir);
            let par = par_of(fdir);
            let mut next: Vec<(Vec<DotMono>, Vec<Dir>)> = Vec::new();
            for (poly, xis) in &terms {
                // par branch
                let mut par_poly: Vec<DotMono> = Vec::new();
                for (j, c) in par.iter().enumerate() {
                    let mut ek = vec![0i32; n_out];
                    ek[j + 1] += 1;
                    par_poly.push((ek, c.clone()));
                }
                next.push((mono_mul(poly, &par_poly)?, xis.clone()));
                // xi branch (only if the direction is a genuine ISP)
                if !reducible {
                    let mut new_xis = xis.clone();
                    new_xis.push(fdir.clone());
                    next.push((poly.clone(), new_xis));
                }
            }
            terms = next;
        }
        for (poly, xis) in terms {
            let m = xis.len();
            if m % 2 == 1 {
                continue;
            }
            let kk = m / 2;
            let idxs: Vec<usize> = (0..m).collect();
            let mut s = Atom::Zero;
            for pr in pairings(&idxs)? {
                let mut term = Atom::num(1);
                for (i, j) in pr {
                    term *= perp_metric(&xis[i], &xis[j]);
                }
                s += term;
            }
            if s == Atom::Zero {
                continue;
            }
            let mut norm = Atom::num(1);
            for jj in 0..kk {
                norm = norm * (&nt + Atom::num(2 * jj as i64));
            }
            let avg_coeff = s / &norm;
            let mut piece = poly.clone();
            for _ in 0..kk {
                piece = mono_mul(&piece, &l2perp)?;
            }
            for (e, c) in &piece {
                push_mono(&mut result, e.clone(), &avg_coeff * c);
            }
        }
    }
    Ok(result)
}

fn dir_in_span(dirs: &[Dir], v: &Dir) -> bool {
    let to_i64 = |d: &Dir| -> Vec<i64> { d.iter().map(|&x| i64::from(x)).collect() };
    let mut basis: Vec<Vec<i64>> = Vec::new();
    for d in dirs {
        let mut row = to_i64(d);
        reduce_against(&mut row, &basis);
        if row.iter().any(|&x| x != 0) {
            basis.push(row);
        }
    }
    let mut row = to_i64(v);
    reduce_against(&mut row, &basis);
    row.iter().all(|&x| x == 0)
}

// Rational c with v = sum c_i dirs[i] (None if v is not in the coordinate span), by exact
// Gauss-Jordan.
fn span_coeffs(dirs: &[Dir], v: &Dir) -> Option<Vec<Atom>> {
    let n_ext = v.len();
    let n = dirs.len();
    if n == 0 {
        return v.iter().all(|&x| x == 0).then(Vec::new);
    }
    // Augmented rows = coordinates (n_ext), cols = n unknowns then rhs.
    let mut a: Vec<Vec<Atom>> = (0..n_ext)
        .map(|row| {
            let mut r: Vec<Atom> = dirs.iter().map(|d| Atom::num(i64::from(d[row]))).collect();
            r.push(Atom::num(i64::from(v[row])));
            r
        })
        .collect();
    let mut pivot_col = vec![usize::MAX; n_ext];
    let mut r = 0;
    for col in 0..n {
        let Some(piv) = (r..n_ext).find(|&i| a[i][col] != Atom::Zero) else {
            continue;
        };
        a.swap(r, piv);
        let pivot_row = a[r].clone();
        for (i, row) in a.iter_mut().enumerate() {
            if i != r && row[col] != Atom::Zero {
                let factor = &row[col] / &pivot_row[col];
                for (cell, p) in row.iter_mut().zip(&pivot_row) {
                    *cell = &*cell - &(&factor * p);
                }
            }
        }
        pivot_col[r] = col;
        r += 1;
        if r == n_ext {
            break;
        }
    }
    // Inconsistent row (all unknowns zero, rhs nonzero) => v not in the span.
    for row in &a {
        if row[..n].iter().all(|x| *x == Atom::Zero) && row[n] != Atom::Zero {
            return None;
        }
    }
    // Free (non-pivot) unknowns are set to zero; read each pivot off its row.
    let mut c = vec![Atom::Zero; n];
    for (row, &col) in a.iter().zip(&pivot_col) {
        if col != usize::MAX {
            c[col] = &row[n] / &row[col];
        }
    }
    Some(c)
}

// Clear each basis pivot from `row` via row <- b_piv*row - row_piv*b.
fn reduce_against(row: &mut [i64], basis: &[Vec<i64>]) {
    for b in basis {
        let piv = b.iter().position(|&x| x != 0).unwrap();
        if row[piv] != 0 {
            let (rp, bp) = (row[piv], b[piv]);
            for (rc, &bc) in row.iter_mut().zip(b) {
                *rc = bp * *rc - rp * bc;
            }
        }
    }
}

// Multiply two dot-polynomials.
fn mono_mul(a: &[DotMono], b: &[DotMono]) -> Result<Vec<DotMono>, OneLoopError> {
    let mut out: Vec<DotMono> = Vec::new();
    for (ea, ca) in a {
        for (eb, cb) in b {
            resources::checkpoint()?;
            let e = ea
                .iter()
                .zip(eb)
                .map(|(&x, &y)| checked_exponent_add(x, y, "multiplying numerator monomials"))
                .collect::<Result<Vec<_>, _>>()?;
            push_mono(&mut out, e, ca * cb);
        }
    }
    Ok(out)
}

fn push_mono(acc: &mut Vec<DotMono>, e: Vec<i32>, c: Atom) {
    if c == Atom::Zero {
        return;
    }
    match acc.iter_mut().find(|(ee, _)| *ee == e) {
        Some(slot) => slot.1 = &slot.1 + &c,
        None => acc.push((e, c)),
    }
}

// Gram dot of two directions using a base-Gram closure over the external momenta.
#[allow(clippy::needless_range_loop)]
fn dir_dot_with(gram: &dyn Fn(usize, usize) -> Atom, u: &Dir, v: &Dir) -> Atom {
    let mut out = Atom::Zero;
    for a in 0..u.len() {
        for b in 0..v.len() {
            if u[a] != 0 && v[b] != 0 {
                out += Atom::num(i64::from(u[a]) * i64::from(v[b])) * gram(a, b);
            }
        }
    }
    out
}

// Clone a base-Gram closure into a fresh boxed closure
fn clone_gram(
    gram: &dyn Fn(usize, usize) -> Atom,
    n_ext: usize,
) -> Box<dyn Fn(usize, usize) -> Atom> {
    let entries: Vec<Vec<Atom>> = (0..n_ext)
        .map(|a| (0..n_ext).map(|b| gram(a, b)).collect())
        .collect();
    Box::new(move |a: usize, b: usize| entries[a][b].clone())
}

// Apply the loop-momentum shift l -> l - r to a dot-polynomial numerator.
fn shift_numerator(
    monos: &[DotMono],
    r: &Dir,
    gram: &dyn Fn(usize, usize) -> Atom,
) -> Result<Vec<DotMono>, OneLoopError> {
    if r.iter().all(|&x| x == 0) {
        return Ok(monos.to_vec());
    }
    let n_ext = r.len();
    let n_dots = n_ext + 1;
    // dot(l,r) = sum_a r_a dot(l,q_a)
    let ll_shift: Vec<DotMono> = {
        let mut ll = vec![0i32; n_dots];
        ll[0] = 1;
        let mut v: Vec<DotMono> = vec![(ll, Atom::num(1))]; // dot(l,l)
        for a in 0..n_ext {
            if r[a] != 0 {
                let mut e = vec![0i32; n_dots];
                e[a + 1] = 1;
                push_mono(&mut v, e, Atom::num(-2 * i64::from(r[a])));
            }
        }
        push_mono(&mut v, vec![0i32; n_dots], dir_dot_with(gram, r, r));
        v
    };
    let lq_shift = |a: usize| -> Vec<DotMono> {
        let mut e = vec![0i32; n_dots];
        e[a + 1] = 1;
        let mut ra: Dir = vec![0i32; n_ext];
        ra[a] = 1;
        vec![
            (e, Atom::num(1)),
            (
                vec![0i32; n_dots],
                Atom::num(-1) * dir_dot_with(gram, r, &ra),
            ),
        ]
    };
    let mut out: Vec<DotMono> = Vec::new();
    for (e, c) in monos {
        // build the shifted product: ll_shift^{e0} * prod_a lq_shift(a)^{e_{a+1}}
        let mut prod: Vec<DotMono> = vec![(vec![0i32; n_dots], c.clone())];
        for _ in 0..e[0] {
            prod = mono_mul(&prod, &ll_shift)?;
        }
        for a in 0..n_ext {
            for _ in 0..e[a + 1] {
                prod = mono_mul(&prod, &lq_shift(a))?;
            }
        }
        for (pe, pc) in prod {
            push_mono(&mut out, pe, pc);
        }
    }
    Ok(out)
}

// Re-express the inverse propagator D_i = dot(k,k) + 2 dot(k,r_i) + (r_i.r_i - m_i) as a dot-polynomial.
fn di_as_dots(r_coeff: &[i32], mass: &Atom, gram: &dyn Fn(usize, usize) -> Atom) -> Vec<DotMono> {
    let n_ext = r_coeff.len();
    let n_dots = n_ext + 1;
    let mut ll = vec![0i32; n_dots];
    ll[0] = 1;
    let mut out: Vec<DotMono> = vec![(ll, Atom::num(1))]; // dot(k,k)
    for j in 0..n_ext {
        if r_coeff[j] != 0 {
            let mut e = vec![0i32; n_dots];
            e[j + 1] = 1;
            push_mono(&mut out, e, Atom::num(2 * i64::from(r_coeff[j])));
        }
    }
    // r_i . r_i = sum_{a,b} c_a c_b (q_a.q_b)
    let mut rr = Atom::Zero;
    for a in 0..n_ext {
        for b in 0..n_ext {
            if r_coeff[a] != 0 && r_coeff[b] != 0 {
                rr += Atom::num(i64::from(r_coeff[a]) * i64::from(r_coeff[b])) * gram(a, b);
            }
        }
    }
    push_mono(&mut out, vec![0i32; n_dots], rr - mass);
    out
}

// The generic engine: reduce a dot-polynomial numerator on topology `topo`.
#[allow(clippy::needless_range_loop)]
fn reduce_num(topo: &Topo, numerator_monos: &[DotMono]) -> Result<Masters, OneLoopError> {
    with_stack(|| reduce_num_inner(topo, numerator_monos))
}

fn reduce_num_inner(topo: &Topo, numerator_monos: &[DotMono]) -> Result<Masters, OneLoopError> {
    // Step 1: ISP projection (no-op for the top topology).
    let projected = isp_project(topo, numerator_monos)?;
    if projected.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2: RSP substitution (slot 0 -> rule_ll, slot j+1 -> rule_lq[j]) -> polynomial in den symbols.
    let n_lines = topo.den_syms.len();
    let mut n = Atom::Zero;
    for (e, c) in &projected {
        resources::checkpoint()?;
        let mut term = c.clone();
        for _ in 0..e[0] {
            term = &term * &topo.rule_ll;
        }
        for (j, rule) in topo.rule_lq.iter().enumerate() {
            for _ in 0..e[j + 1] {
                term = &term * rule;
            }
        }
        n += term;
    }

    // Step 3: extract den-monomials.
    let den_monos = extract_monomials(&n, &topo.den_syms)?;

    // Step 4: route each monomial.
    let mut acc: Vec<(Atom, MasterIntegral)> = Vec::new();
    for (cexp, coeff) in den_monos {
        resources::checkpoint()?;
        let mut b = topo.a.clone();
        for i in 0..n_lines {
            b[i] = checked_index_subtract(b[i], cexp[i], "applying numerator denominator powers")?;
        }
        if b.iter().all(|&x| x >= 0) {
            // pure propagator shift
            add_scaled(&mut acc, &coeff, (topo.scalar)(&b)?);
        } else {
            // pinch lines with b_i <= 0; residual prod D_i^{|b_i|} on survivors.
            let keep: Vec<usize> = (0..n_lines).filter(|&i| b[i] > 0).collect();
            let keep_exps: Vec<i32> = keep.iter().map(|&i| b[i]).collect();
            // build residual dot-polynomial (in this topology's external basis)
            let n_ext = topo.r_coeffs.first().map_or(0, |r| r.len());
            let mut residual: Vec<DotMono> = vec![(vec![0i32; n_ext + 1], Atom::num(1))];
            for (i, &power) in b.iter().enumerate().take(n_lines) {
                if power < 0 {
                    let di = di_as_dots(&topo.r_coeffs[i], &topo.masses[i], &*topo.base_gram);
                    for _ in 0..(-i64::from(power)) {
                        residual = mono_mul(&residual, &di)?;
                    }
                }
            }
            add_scaled(
                &mut acc,
                &coeff,
                (topo.pinch)(&keep, &keep_exps, &residual)?,
            );
        }
    }
    Ok(acc)
}

// --- Topology builders -----------------------------------------------------

fn den_symbol(i: usize) -> Symbol {
    symbol!(format!("oneloopmaster::den{}", i + 1))
}

// Reduce a tadpole numerator
fn tadpole_numerator(numerator: &Atom, a1: i32, m_sq: &Atom) -> Result<Masters, OneLoopError> {
    // A tadpole's loop momentum carries no external offset, so every external
    // contraction dot(k, q_i) is an irreducible scalar product.
    let n_ext = 3;
    let sym_gram: GramFn = Box::new(|a: usize, b: usize| -> Atom {
        let (i, j) = if a <= b { (a, b) } else { (b, a) };
        crate::symbols::scalar_product(
            &(Atom::var(symbol!(format!("oneloopmaster::q{}", i + 1)))),
            &(Atom::var(symbol!(format!("oneloopmaster::q{}", j + 1)))),
        )
    });
    reduce_num(
        &tadpole_topo(a1, m_sq, sym_gram, n_ext),
        &numerator_to_monos(numerator, n_ext)?,
    )
}

// Reduce a bubble with an arbitrary-rank numerator.
fn bubble_numerator(
    numerator: &Atom,
    a1: i32,
    a2: i32,
    p_sq: &Atom,
    m1_sq: &Atom,
    m2_sq: &Atom,
) -> Result<(Atom, Atom, Atom), OneLoopError> {
    let terms = reduce_num(
        &bubble_topo(a1, a2, p_sq, m1_sq, m2_sq),
        &numerator_to_monos(numerator, 3)?,
    )?;
    let mut c_b0 = Atom::Zero;
    let mut c_a1 = Atom::Zero;
    let mut c_a2 = Atom::Zero;
    for (c, m) in terms {
        match m {
            MasterIntegral::Bubble { .. } => c_b0 = &c_b0 + &c,
            MasterIntegral::Tadpole { m_sq } if m_sq == *m1_sq => c_a1 = &c_a1 + &c,
            MasterIntegral::Tadpole { m_sq } if m_sq == *m2_sq => c_a2 = &c_a2 + &c,
            _ => unreachable!("bubble numerator produced an unexpected master"),
        }
    }
    Ok((c_b0, c_a1, c_a2))
}

fn bubble_topo(a1: i32, a2: i32, p_sq: &Atom, m1_sq: &Atom, m2_sq: &Atom) -> Topo {
    // Top-level bubble: line 1 has zero shift, line 2 has shift q1.
    bubble_topo_general(
        a1,
        a2,
        vec![0, 0, 0],
        vec![1, 0, 0],
        m1_sq,
        m2_sq,
        p_sq.clone(),
        base_gram_box(
            p_sq,
            &Atom::Zero,
            &Atom::Zero,
            &Atom::Zero,
            &Atom::Zero,
            &Atom::Zero,
        ),
    )
}

// A general bubble sub-topology: lines r_a (reference) and r_b, with the parent Gram for ISP projection.
#[allow(clippy::too_many_arguments)]
fn bubble_topo_general(
    a1: i32,
    a2: i32,
    r_a: Dir,
    r_b: Dir,
    m_a: &Atom,
    m_b: &Atom,
    p_sq: Atom,
    base_gram: Box<dyn Fn(usize, usize) -> Atom>,
) -> Topo {
    let den1 = den_symbol(0);
    let den2 = den_symbol(1);
    let d1 = Atom::var(den1);
    let d2 = Atom::var(den2);
    let two = Atom::num(2);
    // Reducible direction w = r_b - r_a.
    let w: Dir = r_b.iter().zip(&r_a).map(|(b, a)| b - a).collect();
    // In the reference frame:
    //   l.l = D1 + m_a ;  l.w = (D2 - D1 - m_a + m_b - p_sq)/2
    let rule_ll = &d1 + m_a;
    let rule_w = (&d2 - &d1 - m_a + m_b - &p_sq) / &two;
    let (p1, m1a, m2a) = (p_sq.clone(), m_a.clone(), m_b.clone());

    let pinch_gram = clone_gram(&*base_gram, r_a.len());
    let pinch_r = vec![r_a.clone(), r_b.clone()];
    let pinch_masses = [m_a.clone(), m_b.clone()];
    Topo {
        rsp_dirs: vec![w],
        base_gram,
        rule_ll,
        rule_lq: vec![rule_w],
        den_syms: vec![den1, den2],
        a: vec![a1, a2],
        r_coeffs: vec![r_a, r_b],
        masses: vec![m_a.clone(), m_b.clone()],
        scalar: Box::new(move |b| {
            let m = [m1a.clone(), m2a.clone()];
            reduce_cayley(&modified_cayley(&m, std::slice::from_ref(&p1)), &m, b)
        }),
        // A bubble can pinch to a tadpole carrying a residual; route generically.
        pinch: Box::new(move |keep, exps, num| {
            pinch_to_subtopo(keep, exps, num, &pinch_r, &pinch_masses, &*pinch_gram)
        }),
    }
}

// Base Gram over (q1,q2,q3) from box invariants (triangle: p3=p4=t=0; bubble: only q1.q1).
fn base_gram_box(
    p1: &Atom,
    p2: &Atom,
    p3: &Atom,
    p4: &Atom,
    s: &Atom,
    t: &Atom,
) -> Box<dyn Fn(usize, usize) -> Atom> {
    let two = Atom::num(2);
    let q12 = (s - p1 - p2) / &two;
    let q23 = (t - p2 - p3) / &two;
    let q13 = (p4 - p1 - p2 - p3) / &two - &q12 - &q23;
    let (p1, p2, p3) = (p1.clone(), p2.clone(), p3.clone());
    Box::new(move |a: usize, b: usize| -> Atom {
        let (i, j) = if a <= b { (a, b) } else { (b, a) };
        match (i, j) {
            (0, 0) => p1.clone(),
            (1, 1) => p2.clone(),
            (2, 2) => p3.clone(),
            (0, 1) => q12.clone(),
            (1, 2) => q23.clone(),
            (0, 2) => q13.clone(),
            _ => unreachable!(),
        }
    })
}

// Base Gram q_a.q_b over n_ext external momenta from the C(n_ext+1,2)
fn base_gram_from_pairwise(n_ext: usize, pairwise: &[Atom]) -> Box<dyn Fn(usize, usize) -> Atom> {
    let n = n_ext + 1;
    let idx = |i: usize, j: usize| i * n - i * (i + 1) / 2 + (j - i - 1);
    let s = |i: usize, j: usize| -> Atom {
        if i == j {
            Atom::Zero
        } else {
            let (lo, hi) = (i.min(j), i.max(j));
            pairwise[idx(lo, hi)].clone()
        }
    };
    let gram: Vec<Vec<Atom>> = (0..n_ext)
        .map(|a| {
            (0..n_ext)
                .map(|b| {
                    if a == b {
                        s(a, a + 1)
                    } else {
                        (s(a + 1, b) + s(a, b + 1) - s(a + 1, b + 1) - s(a, b)) / Atom::num(2)
                    }
                })
                .collect()
        })
        .collect();
    Box::new(move |a: usize, b: usize| gram[a][b].clone())
}

// Reduce a triangle with an arbitrary-rank numerator.
#[allow(clippy::too_many_arguments)]
fn triangle_numerator(
    numerator: &Atom,
    a1: i32,
    a2: i32,
    a3: i32,
    s1: &Atom,
    s2: &Atom,
    s3: &Atom,
    m1: &Atom,
    m2: &Atom,
    m3: &Atom,
) -> Result<Masters, OneLoopError> {
    reduce_num(
        &triangle_topo(a1, a2, a3, s1, s2, s3, m1, m2, m3),
        &numerator_to_monos(numerator, 3)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn triangle_topo(
    a1: i32,
    a2: i32,
    a3: i32,
    s1: &Atom,
    s2: &Atom,
    s3: &Atom,
    m1: &Atom,
    m2: &Atom,
    m3: &Atom,
) -> Topo {
    triangle_topo_general(
        a1,
        a2,
        a3,
        [vec![0, 0, 0], vec![1, 0, 0], vec![1, 1, 0]],
        [m1.clone(), m2.clone(), m3.clone()],
        [s1.clone(), s2.clone(), s3.clone()],
        base_gram_box(s1, s2, &Atom::Zero, s3, s3, &Atom::Zero),
    )
}

// A general triangle sub-topology: lines r[0..3], invariants s, parent Gram for ISP projection.
#[allow(clippy::too_many_arguments)]
fn triangle_topo_general(
    a1: i32,
    a2: i32,
    a3: i32,
    r: [Dir; 3],
    masses: [Atom; 3],
    s: [Atom; 3],
    base_gram: Box<dyn Fn(usize, usize) -> Atom>,
) -> Topo {
    let den1 = den_symbol(0);
    let den2 = den_symbol(1);
    let den3 = den_symbol(2);
    let d1 = Atom::var(den1);
    let d2 = Atom::var(den2);
    let d3 = Atom::var(den3);
    let two = Atom::num(2);
    let (m1, m2, m3) = (masses[0].clone(), masses[1].clone(), masses[2].clone());
    let (s1, s2, s3) = (s[0].clone(), s[1].clone(), s[2].clone());
    // Reducible directions w1 = r[1]-r[0], w2 = r[2]-r[1]; RSP rules below.
    let w1: Dir = r[1].iter().zip(&r[0]).map(|(b, a)| b - a).collect();
    let w2: Dir = r[2].iter().zip(&r[1]).map(|(b, a)| b - a).collect();
    let rule_ll = &d1 + &m1;
    let rule_w1 = (&d2 - &d1 - &m1 + &m2 - &s1) / &two;
    let rule_w2 = (&d3 - &d2 - &m2 + &m3 + &s1 - &s3) / &two;
    let (ss1, ss2, ss3) = (s1.clone(), s2.clone(), s3.clone());
    let (sm1, sm2, sm3) = (m1.clone(), m2.clone(), m3.clone());
    let r_for_pinch = r.clone();
    let masses_for_pinch = [m1.clone(), m2.clone(), m3.clone()];
    // Pinch-to-bubble reuses this triangle's full Gram
    let gram_for_pinch = clone_gram(&*base_gram, r[0].len());
    Topo {
        rsp_dirs: vec![w1, w2],
        base_gram,
        rule_ll,
        rule_lq: vec![rule_w1, rule_w2],
        den_syms: vec![den1, den2, den3],
        a: vec![a1, a2, a3],
        r_coeffs: r.to_vec(),
        masses: vec![m1, m2, m3],
        scalar: Box::new(move |b| {
            let m = [sm1.clone(), sm2.clone(), sm3.clone()];
            // physics (s1,s2,s3) -> lexicographic Cayley order (s01, s02, s12)
            let lex = [ss1.clone(), ss3.clone(), ss2.clone()];
            reduce_cayley(&modified_cayley(&m, &lex), &m, b)
        }),
        pinch: Box::new(move |keep, exps, num| {
            pinch_to_subtopo(
                keep,
                exps,
                num,
                &r_for_pinch,
                &masses_for_pinch,
                &*gram_for_pinch,
            )
        }),
    }
}

fn pinch_to_subtopo(
    keep: &[usize],
    exps: &[i32],
    num: &[DotMono],
    all_r: &[Dir],
    all_masses: &[Atom],
    base_gram: &dyn Fn(usize, usize) -> Atom,
) -> Result<Masters, OneLoopError> {
    if keep.is_empty() {
        // No denominator survives
        return Ok(Vec::new());
    }
    let r_ref = all_r[keep[0]].clone();
    let n_ext = r_ref.len();
    // Shift the residual into the child's reference frame
    let shifted = shift_numerator(num, &r_ref, base_gram)?;
    // Surviving line shift-directions relative to the reference, and masses.
    let dirs: Vec<Dir> = keep
        .iter()
        .map(|&i| all_r[i].iter().zip(&r_ref).map(|(x, y)| x - y).collect())
        .collect();
    let kmasses: Vec<Atom> = keep.iter().map(|&i| all_masses[i].clone()).collect();

    match keep.len() {
        k if k >= 5 => {
            let topo = ngon_topo(exps, &dirs, &kmasses, clone_gram(base_gram, n_ext));
            reduce_num(&topo, &shifted)
        }
        4 => {
            let topo = box_topo_general(
                [exps[0], exps[1], exps[2], exps[3]],
                [
                    dirs[0].clone(),
                    dirs[1].clone(),
                    dirs[2].clone(),
                    dirs[3].clone(),
                ],
                [
                    kmasses[0].clone(),
                    kmasses[1].clone(),
                    kmasses[2].clone(),
                    kmasses[3].clone(),
                ],
                clone_gram(base_gram, n_ext),
            );
            reduce_num(&topo, &shifted)
        }
        3 => {
            // Triangle invariants of the kept triple.
            let w_ab = dirs[1].clone();
            let w_bc: Dir = dirs[2].iter().zip(&dirs[1]).map(|(a, b)| a - b).collect();
            let w_ac = dirs[2].clone();
            let s1 = dir_dot_with(base_gram, &w_ab, &w_ab);
            let s2 = dir_dot_with(base_gram, &w_bc, &w_bc);
            let s3 = dir_dot_with(base_gram, &w_ac, &w_ac);
            let topo = triangle_topo_general(
                exps[0],
                exps[1],
                exps[2],
                [dirs[0].clone(), dirs[1].clone(), dirs[2].clone()],
                [kmasses[0].clone(), kmasses[1].clone(), kmasses[2].clone()],
                [s1, s2, s3],
                clone_gram(base_gram, n_ext),
            );
            reduce_num(&topo, &shifted)
        }
        2 => {
            let w = dirs[1].clone();
            let p_sq = dir_dot_with(base_gram, &w, &w);
            let topo = bubble_topo_general(
                exps[0],
                exps[1],
                vec![0i32; n_ext],
                w,
                &kmasses[0],
                &kmasses[1],
                p_sq,
                clone_gram(base_gram, n_ext),
            );
            reduce_num(&topo, &shifted)
        }
        1 => {
            let topo = tadpole_topo(exps[0], &kmasses[0], clone_gram(base_gram, n_ext), n_ext);
            reduce_num(&topo, &shifted)
        }
        _ => unreachable!("pinch keeps 1..=4 lines"),
    }
}

// A tadpole sub-topology: one line, no reducible directions; every loop dot is an ISP.
fn tadpole_topo(
    a1: i32,
    mass: &Atom,
    base_gram: Box<dyn Fn(usize, usize) -> Atom>,
    n_ext: usize,
) -> Topo {
    let den1 = den_symbol(0);
    let d1 = Atom::var(den1);
    let rule_ll = &d1 + mass;
    let m = mass.clone();
    Topo {
        rsp_dirs: vec![],
        base_gram,
        rule_ll,
        rule_lq: vec![],
        den_syms: vec![den1],
        a: vec![a1],
        r_coeffs: vec![vec![0i32; n_ext]],
        masses: vec![mass.clone()],
        scalar: Box::new(move |b| {
            // A0 with exponent <= 0 is a scaleless integral (no scale) -> 0.
            if b[0] <= 0 {
                return Ok(Vec::new());
            }
            let coeff = tadpole_coefficient(b[0], &m)?;
            Ok(if coeff == Atom::Zero {
                Vec::new()
            } else {
                vec![(coeff, MasterIntegral::Tadpole { m_sq: m.clone() })]
            })
        }),
        pinch: Box::new(|_, _, _| Ok(Vec::new())),
    }
}

// Reduce a box with an arbitrary-rank numerator.
#[allow(clippy::too_many_arguments)]
fn box_numerator(
    numerator: &Atom,
    a1: i32,
    a2: i32,
    a3: i32,
    a4: i32,
    p1: &Atom,
    p2: &Atom,
    p3: &Atom,
    p4: &Atom,
    s: &Atom,
    t: &Atom,
    m1: &Atom,
    m2: &Atom,
    m3: &Atom,
    m4: &Atom,
) -> Result<Masters, OneLoopError> {
    reduce_num(
        &box_topo(a1, a2, a3, a4, p1, p2, p3, p4, s, t, m1, m2, m3, m4),
        &numerator_to_monos(numerator, 3)?,
    )
}

#[allow(clippy::too_many_arguments)]
fn box_topo(
    a1: i32,
    a2: i32,
    a3: i32,
    a4: i32,
    p1: &Atom,
    p2: &Atom,
    p3: &Atom,
    p4: &Atom,
    s: &Atom,
    t: &Atom,
    m1: &Atom,
    m2: &Atom,
    m3: &Atom,
    m4: &Atom,
) -> Topo {
    let den1 = den_symbol(0);
    let den2 = den_symbol(1);
    let den3 = den_symbol(2);
    let den4 = den_symbol(3);
    let d1 = Atom::var(den1);
    let d2 = Atom::var(den2);
    let d3 = Atom::var(den3);
    let d4 = Atom::var(den4);
    let two = Atom::num(2);
    // Reducible directions q1,q2,q3 (the full span); RSP rules below.
    let rule_ll = &d1 + m1;
    let rule_q1 = (&d2 - &d1 - m1 + m2 - p1) / &two;
    let rule_q2 = (&d3 - &d2 - m2 + m3 + p1 - s) / &two;
    let rule_q3 = (&d4 - &d3 - m3 + m4 - p4 + s) / &two;
    let base_gram = base_gram_box(p1, p2, p3, p4, s, t);
    let masses = [m1.clone(), m2.clone(), m3.clone(), m4.clone()];
    let r: [Dir; 4] = [vec![0, 0, 0], vec![1, 0, 0], vec![1, 1, 0], vec![1, 1, 1]];
    let (sp1, sp2, sp3, sp4) = (p1.clone(), p2.clone(), p3.clone(), p4.clone());
    let (ss, st) = (s.clone(), t.clone());
    let (sm1, sm2, sm3, sm4) = (m1.clone(), m2.clone(), m3.clone(), m4.clone());
    let pinch_gram = base_gram_box(p1, p2, p3, p4, s, t);
    let pinch_masses = masses.clone();
    Topo {
        rsp_dirs: vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]],
        base_gram,
        rule_ll,
        rule_lq: vec![rule_q1, rule_q2, rule_q3],
        den_syms: vec![den1, den2, den3, den4],
        a: vec![a1, a2, a3, a4],
        r_coeffs: r.to_vec(),
        masses: masses.to_vec(),
        scalar: Box::new(move |b| {
            let m = [sm1.clone(), sm2.clone(), sm3.clone(), sm4.clone()];
            // physics (p1,p2,p3,p4,s,t) -> lexicographic Cayley (s01,s02,s03,s12,s13,s23)
            let lex = [
                sp1.clone(),
                ss.clone(),
                sp4.clone(),
                sp2.clone(),
                st.clone(),
                sp3.clone(),
            ];
            reduce_cayley(&modified_cayley(&m, &lex), &m, b)
        }),
        pinch: Box::new(move |keep, exps, num| {
            pinch_to_subtopo(keep, exps, num, &r, &pinch_masses, &*pinch_gram)
        }),
    }
}

// A general box sub-topology: lines r[0..4], parent Gram for ISP projection and the six D0 invariants.
fn box_topo_general(
    exps: [i32; 4],
    r: [Dir; 4],
    masses: [Atom; 4],
    base_gram: Box<dyn Fn(usize, usize) -> Atom>,
) -> Topo {
    let dens: Vec<Symbol> = (0..4).map(den_symbol).collect();
    let d: Vec<Atom> = dens.iter().map(|&s| Atom::var(s)).collect();
    let two = Atom::num(2);
    let add = |u: &Dir, v: &Dir| -> Dir { u.iter().zip(v).map(|(a, b)| a + b).collect() };
    let sub = |u: &Dir, v: &Dir| -> Dir { u.iter().zip(v).map(|(a, b)| a - b).collect() };
    let w1 = sub(&r[1], &r[0]);
    let w2 = sub(&r[2], &r[1]);
    let w3 = sub(&r[3], &r[2]);
    // Box invariants (r_i - r_j)^2 from the parent Gram, in reduce_box's convention.
    let p1 = dir_dot_with(&*base_gram, &w1, &w1);
    let p2 = dir_dot_with(&*base_gram, &w2, &w2);
    let p3 = dir_dot_with(&*base_gram, &w3, &w3);
    let w12 = add(&w1, &w2);
    let w23 = add(&w2, &w3);
    let w123 = add(&w12, &w3);
    let s = dir_dot_with(&*base_gram, &w12, &w12);
    let t = dir_dot_with(&*base_gram, &w23, &w23);
    let p4 = dir_dot_with(&*base_gram, &w123, &w123);
    let (m1, m2, m3, m4) = (
        masses[0].clone(),
        masses[1].clone(),
        masses[2].clone(),
        masses[3].clone(),
    );
    // RSP rules, as in box_topo.
    let rule_ll = &d[0] + &m1;
    let rule_w1 = (&d[1] - &d[0] - &m1 + &m2 - &p1) / &two;
    let rule_w2 = (&d[2] - &d[1] - &m2 + &m3 + &p1 - &s) / &two;
    let rule_w3 = (&d[3] - &d[2] - &m3 + &m4 - &p4 + &s) / &two;
    let (sp1, sp2, sp3, sp4, ss, st) = (
        p1.clone(),
        p2.clone(),
        p3.clone(),
        p4.clone(),
        s.clone(),
        t.clone(),
    );
    let (sm1, sm2, sm3, sm4) = (m1.clone(), m2.clone(), m3.clone(), m4.clone());
    let r_for_pinch = r.clone();
    let masses_for_pinch = masses.clone();
    let gram_for_pinch = clone_gram(&*base_gram, r[0].len());
    Topo {
        rsp_dirs: vec![w1, w2, w3],
        base_gram,
        rule_ll,
        rule_lq: vec![rule_w1, rule_w2, rule_w3],
        den_syms: dens,
        a: exps.to_vec(),
        r_coeffs: r.to_vec(),
        masses: vec![m1, m2, m3, m4],
        scalar: Box::new(move |b| {
            let m = [sm1.clone(), sm2.clone(), sm3.clone(), sm4.clone()];
            let lex = [
                sp1.clone(),
                ss.clone(),
                sp4.clone(),
                sp2.clone(),
                st.clone(),
                sp3.clone(),
            ];
            reduce_cayley(&modified_cayley(&m, &lex), &m, b)
        }),
        pinch: Box::new(move |keep, exps, num| {
            pinch_to_subtopo(
                keep,
                exps,
                num,
                &r_for_pinch,
                &masses_for_pinch,
                &*gram_for_pinch,
            )
        }),
    }
}

// Reduce an N-gon (N >= 5) numerator
fn ngon_numerator(
    numerator: &Atom,
    exps: &[i32],
    invariants: &[Atom],
    masses: &[Atom],
) -> Result<Masters, OneLoopError> {
    let n = masses.len();
    let n_ext = n - 1;
    // Standard offsets r_i = q1 + ... + q_{i-1}, as directions over (q1..q_{n_ext}).
    let r_dirs: Vec<Dir> = (0..n)
        .map(|i| (0..n_ext).map(|k| (k < i) as i32).collect())
        .collect();
    reduce_num(
        &ngon_topo(
            exps,
            &r_dirs,
            masses,
            base_gram_from_pairwise(n_ext, invariants),
        ),
        &numerator_to_monos(numerator, n_ext)?,
    )
}

// A general K-point topology (K >= 5): RSP rules, reducible dirs, and Cayley invariants from the Gram.
fn ngon_topo(
    exps: &[i32],
    r_dirs: &[Dir],
    masses: &[Atom],
    base_gram: Box<dyn Fn(usize, usize) -> Atom>,
) -> Topo {
    let k = r_dirs.len();
    let n_ext = r_dirs[0].len();
    let dens: Vec<Symbol> = (0..k).map(den_symbol).collect();
    let d: Vec<Atom> = dens.iter().map(|&s| Atom::var(s)).collect();
    let two = Atom::num(2);
    let sub = |u: &Dir, v: &Dir| -> Dir { u.iter().zip(v).map(|(a, b)| a - b).collect() };
    // r_i^2 in the parent metric.
    let rr: Vec<Atom> = (0..k)
        .map(|i| dir_dot_with(&*base_gram, &r_dirs[i], &r_dirs[i]))
        .collect();
    let rule_ll = &d[0] + &masses[0];
    // l.q_a = [D_{a+1} - D_a - (r_{a+1}^2 - r_a^2) + m_{a+1} - m_a]/2, a = 0..k-1.
    let rule_lq: Vec<Atom> = (0..k - 1)
        .map(|a| (&d[a + 1] - &d[a] - (&rr[a + 1] - &rr[a]) + &masses[a + 1] - &masses[a]) / &two)
        .collect();
    // Reducible directions = the external momenta q_a = r_{a+1} - r_a.
    let rsp_dirs: Vec<Dir> = (0..k - 1)
        .map(|a| sub(&r_dirs[a + 1], &r_dirs[a]))
        .collect();

    let pairwise: Vec<Atom> = (0..k)
        .flat_map(|i| (i + 1..k).map(move |j| (i, j)))
        .map(|(i, j)| {
            let w = sub(&r_dirs[i], &r_dirs[j]);
            dir_dot_with(&*base_gram, &w, &w)
        })
        .collect();
    let smasses = masses.to_vec();
    let (r_for_pinch, pmasses) = (r_dirs.to_vec(), masses.to_vec());
    let gram_for_pinch = clone_gram(&*base_gram, n_ext);
    Topo {
        rsp_dirs,
        base_gram,
        rule_ll,
        rule_lq,
        den_syms: dens,
        a: exps.to_vec(),
        r_coeffs: r_dirs.to_vec(),
        masses: masses.to_vec(),
        scalar: Box::new(move |b| {
            reduce_cayley(&modified_cayley(&smasses, &pairwise), &smasses, b)
        }),
        pinch: Box::new(move |keep, exps, num| {
            pinch_to_subtopo(keep, exps, num, &r_for_pinch, &pmasses, &*gram_for_pinch)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_TOTAL_INDEX, dot_lq, reduce};
    use crate::masters::MasterIntegral;
    use crate::recurrence::RecurrenceInput;
    use crate::symbols::S;
    use symbolica::atom::{Atom, AtomCore};
    use symbolica::function;
    use symbolica::symbol;

    #[test]
    fn det_matches_a_known_value() {
        crate::ensure_symbolica_license();
        // M = [[2,1,0],[1,3,1],[0,1,2]], det 8
        let m = vec![
            vec![Atom::num(2), Atom::num(1), Atom::num(0)],
            vec![Atom::num(1), Atom::num(3), Atom::num(1)],
            vec![Atom::num(0), Atom::num(1), Atom::num(2)],
        ];
        assert_eq!(super::det(&m), Atom::num(8));
    }

    #[test]
    fn scalar_pentagon_reduces_to_five_boxes() {
        crate::ensure_symbolica_license();
        let masses: Vec<Atom> = [1, 2, 3, 4, 5].iter().map(|&x| Atom::num(x)).collect();
        let invariants: Vec<Atom> = [3, 5, 7, 9, 4, 6, 8, 5, 7, 6]
            .iter()
            .map(|&x| Atom::num(x))
            .collect();
        let r = reduce(&scalar_family(masses, invariants)).unwrap();
        assert_eq!(r.terms.len(), 5);
        assert!(
            r.terms
                .iter()
                .all(|(_, m)| matches!(m, MasterIntegral::Box { .. }))
        );
        // van Neerven-Vermaseren coefficients
        let want_c = [
            Atom::num(-1088) / Atom::num(639),
            Atom::num(-212) / Atom::num(639),
            Atom::num(-44) / Atom::num(213),
            Atom::num(-191) / Atom::num(639),
            Atom::num(-341) / Atom::num(639),
        ];
        for ((coeff, _), want) in r.terms.iter().zip(&want_c) {
            assert_eq!(coeff, want);
        }
        // pinching the last propagator leaves the box on propagators 1..4
        assert_eq!(
            r.terms[4].1,
            MasterIntegral::Box {
                p1_sq: Atom::num(3),
                p2_sq: Atom::num(4),
                p3_sq: Atom::num(5),
                p4_sq: Atom::num(7),
                s: Atom::num(5),
                t: Atom::num(6),
                m1_sq: Atom::num(1),
                m2_sq: Atom::num(2),
                m3_sq: Atom::num(3),
                m4_sq: Atom::num(4),
            }
        );
    }

    #[test]
    fn scalar_hexagon_recurses_down_to_boxes() {
        crate::ensure_symbolica_license();
        let masses: Vec<Atom> = [1, 2, 3, 4, 5, 6].iter().map(|&x| Atom::num(x)).collect();
        let invariants: Vec<Atom> = [3, 5, 7, 9, 11, 4, 6, 8, 10, 5, 7, 9, 6, 8, 7]
            .iter()
            .map(|&x| Atom::num(x))
            .collect();
        let r = reduce(&scalar_family(masses, invariants)).unwrap();
        assert!(!r.terms.is_empty());
        assert!(
            r.terms
                .iter()
                .all(|(_, m)| matches!(m, MasterIntegral::Box { .. }))
        );
    }

    #[test]
    fn heptagon_numerator_reduces_to_finite_masters() {
        crate::ensure_symbolica_license();
        let masses: Vec<Atom> = (1..=7).map(Atom::num).collect();
        let invariants: Vec<Atom> = [
            4, -8, -20, -28, -12, -22, -8, -16, -28, -20, -26, -16, -38, -20, -16, -32, 4, -22,
            -10, -10, -16,
        ]
        .iter()
        .map(|&x| Atom::num(x))
        .collect();
        for num in [super::dot_lq(0), &super::dot_lq(0) * &super::dot_lq(5)] {
            let mut fam = family(masses.clone(), invariants.clone(), vec![1; 7]);
            fam.numerator = num;
            let r = reduce(&fam).unwrap();
            assert!(!r.terms.is_empty());
            for (c, m) in &r.terms {
                assert!(c.to_string().is_ascii(), "non-finite heptagon coeff: {c}");
                assert!(matches!(
                    m,
                    MasterIntegral::Box { .. }
                        | MasterIntegral::Triangle { .. }
                        | MasterIntegral::Bubble { .. }
                        | MasterIntegral::Tadpole { .. }
                ));
            }
        }
    }

    #[test]
    #[ignore = "slow (~80s): dotted N>=7 FJT dimension-shift tree; verifies the gap-1 fix"]
    fn dotted_heptagon_reduces_to_finite_masters() {
        crate::ensure_symbolica_license();
        let masses: Vec<Atom> = (1..=7).map(Atom::num).collect();
        let invariants: Vec<Atom> = [
            4, -8, -20, -28, -12, -22, -8, -16, -28, -20, -26, -16, -38, -20, -16, -32, 4, -22,
            -10, -10, -16,
        ]
        .iter()
        .map(|&x| Atom::num(x))
        .collect();
        let r = reduce(&family(masses, invariants, vec![2, 1, 1, 1, 1, 1, 1])).unwrap();
        assert!(!r.terms.is_empty());
        for (c, m) in &r.terms {
            assert!(
                c.to_string().is_ascii(),
                "non-finite dotted-heptagon coeff: {c}"
            );
            assert!(matches!(
                m,
                MasterIntegral::Box { .. }
                    | MasterIntegral::Triangle { .. }
                    | MasterIntegral::Bubble { .. }
                    | MasterIntegral::Tadpole { .. }
            ));
        }
    }

    #[test]
    fn base_gram_from_pairwise_matches_the_box_gram() {
        crate::ensure_symbolica_license();
        // physics box invariants (p1,p2,p3,p4,s,t)
        let n = |x| Atom::num(x);
        let phys = super::base_gram_box(&n(2), &n(3), &n(4), &n(5), &n(6), &n(7));
        let lex: Vec<Atom> = [2, 6, 5, 3, 7, 4].iter().map(|&x| n(x)).collect();
        let general = super::base_gram_from_pairwise(3, &lex);
        for a in 0..3 {
            for b in 0..3 {
                assert_eq!(general(a, b), phys(a, b), "q{a}.q{b}");
            }
        }
    }

    #[test]
    fn dotted_pentagon_reduces_to_valid_masters() {
        crate::ensure_symbolica_license();
        let masses: Vec<Atom> = [1, 2, 3, 4, 5].iter().map(|&x| Atom::num(x)).collect();
        let invariants: Vec<Atom> = [3, 5, 7, 9, 4, 6, 8, 5, 7, 6]
            .iter()
            .map(|&x| Atom::num(x))
            .collect();
        // pentagon with one squared propagator
        let r = reduce(&family(masses, invariants, vec![2, 1, 1, 1, 1])).unwrap();
        assert!(!r.terms.is_empty());
        assert!(r.terms.iter().all(|(_, m)| matches!(
            m,
            MasterIntegral::Box { .. }
                | MasterIntegral::Triangle { .. }
                | MasterIntegral::Bubble { .. }
                | MasterIntegral::Tadpole { .. }
        )));
        assert!(
            r.terms
                .iter()
                .any(|(_, m)| matches!(m, MasterIntegral::Triangle { .. }))
        );
    }

    #[test]
    fn pentagon_with_numerator_reduces_to_valid_masters() {
        crate::ensure_symbolica_license();
        let masses: Vec<Atom> = [1, 2, 3, 4, 5].iter().map(|&x| Atom::num(x)).collect();
        let invariants: Vec<Atom> = [3, 5, 7, 9, 4, 6, 8, 5, 7, 6]
            .iter()
            .map(|&x| Atom::num(x))
            .collect();
        let mut fam = family(masses, invariants, vec![1, 1, 1, 1, 1]);
        // numerator = l . q4
        fam.numerator = super::dot_lq(3);
        let r = reduce(&fam).unwrap();
        assert!(!r.terms.is_empty());
        assert!(r.terms.iter().all(|(_, m)| matches!(
            m,
            MasterIntegral::Box { .. }
                | MasterIntegral::Triangle { .. }
                | MasterIntegral::Bubble { .. }
                | MasterIntegral::Tadpole { .. }
        )));
        assert!(
            r.terms
                .iter()
                .any(|(_, m)| matches!(m, MasterIntegral::Box { .. }))
        );
    }

    #[test]
    fn hexagon_numerator_reduces_to_finite_masters() {
        crate::ensure_symbolica_license();
        let masses: Vec<Atom> = [1, 2, 3, 4, 5, 6].iter().map(|&x| Atom::num(x)).collect();
        let invariants: Vec<Atom> = [
            -4, -10, -20, -30, -30, -10, -24, -30, -22, -18, -38, -28, -34, -14, -18,
        ]
        .iter()
        .map(|&x| Atom::num(x))
        .collect();
        // rank-1 (l.q5, boxes only) and rank-2 (l.q1^2, pinches to pentagons w/ residual)
        for num in [super::dot_lq(4), super::dot_lq(0) * super::dot_lq(0)] {
            let mut fam = family(masses.clone(), invariants.clone(), vec![1; 6]);
            fam.numerator = num;
            let r = reduce(&fam).unwrap();
            assert!(!r.terms.is_empty());
            for (c, m) in &r.terms {
                assert!(
                    c.to_string().is_ascii(),
                    "non-finite coeff (degenerate Gram): {c}"
                );
                assert!(matches!(
                    m,
                    MasterIntegral::Box { .. }
                        | MasterIntegral::Triangle { .. }
                        | MasterIntegral::Bubble { .. }
                        | MasterIntegral::Tadpole { .. }
                ));
            }
        }
    }

    fn family(masses: Vec<Atom>, invariants: Vec<Atom>, exponents: Vec<i32>) -> RecurrenceInput {
        RecurrenceInput {
            masses_squared: masses,
            invariants,
            powers: exponents,
            numerator: Atom::num(1),
        }
    }

    fn scalar_family(masses: Vec<Atom>, invariants: Vec<Atom>) -> RecurrenceInput {
        let n = masses.len();
        family(masses, invariants, vec![1; n])
    }

    // [p1, p2, p3, p4, s, t, m1, m2, m3, m4]
    fn box_syms() -> [Atom; 10] {
        [
            Atom::var(symbol!("oneloopmaster::p1")),
            Atom::var(symbol!("oneloopmaster::p2")),
            Atom::var(symbol!("oneloopmaster::p3")),
            Atom::var(symbol!("oneloopmaster::p4")),
            Atom::var(symbol!("oneloopmaster::sinv")),
            Atom::var(symbol!("oneloopmaster::tinv")),
            Atom::var(symbol!("oneloopmaster::m1sq")),
            Atom::var(symbol!("oneloopmaster::m2sq")),
            Atom::var(symbol!("oneloopmaster::m3sq")),
            Atom::var(symbol!("oneloopmaster::m4sq")),
        ]
    }

    #[test]
    fn scalar_tadpole_reduces_to_unit_a0() {
        crate::ensure_symbolica_license();
        let r = reduce(&scalar_family(vec![Atom::num(1)], vec![])).unwrap();
        assert_eq!(r.terms.len(), 1);
        let (coeff, master) = &r.terms[0];
        assert_eq!(*coeff, Atom::num(1));
        match master {
            MasterIntegral::Tadpole { m_sq } => assert_eq!(*m_sq, Atom::num(1)),
            other => panic!("expected a tadpole master, got {other:?}"),
        }
    }

    #[test]
    fn dotted_tadpole_reduces_with_recursion_coefficient() {
        crate::ensure_symbolica_license();
        let msq = Atom::var(symbol!("oneloopmaster::msq"));
        let r = reduce(&family(vec![msq.clone()], vec![], vec![2])).unwrap();
        assert_eq!(r.terms.len(), 1);
        let (coeff, master) = &r.terms[0];
        let d = Atom::var(S.d);
        let want = (&d - Atom::num(2)) / (Atom::num(2) * &msq);
        assert_eq!(*coeff, want);
        match master {
            MasterIntegral::Tadpole { m_sq } => assert_eq!(*m_sq, msq),
            other => panic!("expected a tadpole master, got {other:?}"),
        }
    }

    #[test]
    fn massless_dotted_tadpole_vanishes() {
        crate::ensure_symbolica_license();
        let r = reduce(&family(vec![Atom::Zero], vec![], vec![2])).unwrap();
        assert_eq!(r.terms.len(), 1);
        assert_eq!(r.terms[0].0, Atom::Zero);
    }

    #[test]
    fn tadpole_rank2_external_numerator_symmetric_averages() {
        // ∫ (k·q1)²/(k²-m²) = (q1²/d) ∫ k²/(k²-m²) — the symmetric (transverse)
        // average keeps the external invariant dot(q1,q1) symbolic.
        crate::ensure_symbolica_license();
        let msq = Atom::var(symbol!("oneloopmaster::msq"));
        let mut fam = family(vec![msq.clone()], vec![], vec![1]);
        fam.numerator = &dot_lq(0) * &dot_lq(0); // dot(k, q1)^2
        let r = reduce(&fam).unwrap();
        assert_eq!(r.terms.len(), 1, "expected one A0 term, got {:?}", r.terms);
        let (coeff, master) = &r.terms[0];
        match master {
            MasterIntegral::Tadpole { m_sq } => assert_eq!(*m_sq, msq),
            other => panic!("expected a tadpole master, got {other:?}"),
        }
        assert_ne!(
            *coeff,
            Atom::Zero,
            "rank-2 tadpole coefficient must be non-zero"
        );
        let s = coeff.to_string();
        assert!(
            s.contains("q1") && s.contains('d'),
            "coefficient should carry the symbolic dot(q1,q1) and 1/d: {s}"
        );
    }

    #[test]
    fn dotted_bubble_reduces_to_a_bubble_and_two_tadpoles() {
        crate::ensure_symbolica_license();
        let psq = Atom::var(S.psq);
        let m1 = Atom::var(symbol!("oneloopmaster::m1sq"));
        let m2 = Atom::var(symbol!("oneloopmaster::m2sq"));
        let r = reduce(&family(vec![m1, m2], vec![psq], vec![3, 1])).unwrap();
        assert_eq!(r.terms.len(), 3);
        assert!(matches!(r.terms[0].1, MasterIntegral::Bubble { .. }));
        assert!(matches!(r.terms[1].1, MasterIntegral::Tadpole { .. }));
        assert!(matches!(r.terms[2].1, MasterIntegral::Tadpole { .. }));
    }

    #[test]
    fn bubble_with_linear_numerator_reduces_to_masters() {
        crate::ensure_symbolica_license();
        let psq = Atom::var(S.psq);
        let m1 = Atom::var(symbol!("oneloopmaster::m1sq"));
        let m2 = Atom::var(symbol!("oneloopmaster::m2sq"));
        // numerator = l . p
        let fam = RecurrenceInput {
            masses_squared: vec![m1, m2],
            invariants: vec![psq],
            powers: vec![1, 1],
            numerator: crate::symbols::scalar_product(&(Atom::var(S.k)), &(Atom::var(S.q1))),
        };
        let r = reduce(&fam).unwrap();
        assert_eq!(r.terms.len(), 3);
        assert!(
            r.terms
                .iter()
                .any(|(_, m)| matches!(m, MasterIntegral::Bubble { .. }))
        );
        assert_eq!(
            r.terms
                .iter()
                .filter(|(_, m)| matches!(m, MasterIntegral::Tadpole { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn scalar_bubble_reduces_to_unit_b0() {
        crate::ensure_symbolica_license();
        let r = reduce(&scalar_family(
            vec![Atom::Zero, Atom::Zero],
            vec![Atom::var(S.psq)],
        ))
        .unwrap();
        assert_eq!(r.terms.len(), 1);
        let (coeff, master) = &r.terms[0];
        assert_eq!(*coeff, Atom::num(1));
        match master {
            MasterIntegral::Bubble { p_sq, m1_sq, m2_sq } => {
                assert_eq!(*p_sq, Atom::var(S.psq));
                assert_eq!(*m1_sq, Atom::Zero);
                assert_eq!(*m2_sq, Atom::Zero);
            }
            other => panic!("expected a bubble master, got {other:?}"),
        }
    }

    #[test]
    fn dotted_triangle_reduces_to_masters() {
        crate::ensure_symbolica_license();
        let s1 = Atom::var(symbol!("oneloopmaster::s1"));
        let s2 = Atom::var(symbol!("oneloopmaster::s2"));
        let s3 = Atom::var(symbol!("oneloopmaster::s3"));
        let m1 = Atom::var(symbol!("oneloopmaster::m1sq"));
        let m2 = Atom::var(symbol!("oneloopmaster::m2sq"));
        let m3 = Atom::var(symbol!("oneloopmaster::m3sq"));
        let r = reduce(&family(vec![m1, m2, m3], vec![s1, s2, s3], vec![2, 2, 2])).unwrap();
        let triangles = r
            .terms
            .iter()
            .filter(|(_, m)| matches!(m, MasterIntegral::Triangle { .. }))
            .count();
        let bubbles = r
            .terms
            .iter()
            .filter(|(_, m)| matches!(m, MasterIntegral::Bubble { .. }))
            .count();
        assert_eq!(triangles, 1);
        assert!(bubbles >= 1);
        assert!(r.terms.iter().all(|(_, m)| matches!(
            m,
            MasterIntegral::Triangle { .. }
                | MasterIntegral::Bubble { .. }
                | MasterIntegral::Tadpole { .. }
        )));
    }

    #[test]
    fn pinched_triangle_routes_to_the_right_bubble() {
        crate::ensure_symbolica_license();
        let s1 = Atom::var(symbol!("oneloopmaster::s1"));
        let s2 = Atom::var(symbol!("oneloopmaster::s2"));
        let s3 = Atom::var(symbol!("oneloopmaster::s3"));
        let m1 = Atom::var(symbol!("oneloopmaster::m1sq"));
        let m2 = Atom::var(symbol!("oneloopmaster::m2sq"));
        let m3 = Atom::var(symbol!("oneloopmaster::m3sq"));
        // third line pinched -> bubble of lines 1,2 carrying s1
        let r = reduce(&family(
            vec![m1.clone(), m2.clone(), m3],
            vec![s1.clone(), s2, s3],
            vec![1, 1, 0],
        ))
        .unwrap();
        assert!(r.terms.iter().any(|(_, m)| matches!(
            m,
            MasterIntegral::Bubble { p_sq, m1_sq, m2_sq }
                if *p_sq == s1 && *m1_sq == m1 && *m2_sq == m2
        )));
        assert!(
            r.terms
                .iter()
                .all(|(_, m)| !matches!(m, MasterIntegral::Triangle { .. }))
        );
    }

    #[test]
    fn triangle_with_linear_numerator_reduces_to_masters() {
        crate::ensure_symbolica_license();
        let s1 = Atom::var(symbol!("oneloopmaster::s1"));
        let s2 = Atom::var(symbol!("oneloopmaster::s2"));
        let s3 = Atom::var(symbol!("oneloopmaster::s3"));
        let m1 = Atom::var(symbol!("oneloopmaster::m1sq"));
        let m2 = Atom::var(symbol!("oneloopmaster::m2sq"));
        let m3 = Atom::var(symbol!("oneloopmaster::m3sq"));
        // numerator = 2*(l.q1) - (l.q2) + (l.l)
        let numerator = Atom::num(2)
            * crate::symbols::scalar_product(&(Atom::var(S.k)), &(Atom::var(S.q1)))
            - crate::symbols::scalar_product(&(Atom::var(S.k)), &(Atom::var(S.q2)))
            + crate::symbols::scalar_product(&(Atom::var(S.k)), &(Atom::var(S.k)));
        let fam = RecurrenceInput {
            masses_squared: vec![m1, m2, m3],
            invariants: vec![s1, s2, s3],
            powers: vec![1, 1, 1],
            numerator,
        };
        let r = reduce(&fam).unwrap();
        assert!(
            r.terms
                .iter()
                .any(|(_, m)| matches!(m, MasterIntegral::Triangle { .. }))
        );
        assert!(
            r.terms
                .iter()
                .any(|(_, m)| matches!(m, MasterIntegral::Bubble { .. }))
        );
        assert!(r.terms.iter().all(|(_, m)| matches!(
            m,
            MasterIntegral::Triangle { .. }
                | MasterIntegral::Bubble { .. }
                | MasterIntegral::Tadpole { .. }
        )));
    }

    #[test]
    fn scalar_triangle_reduces_to_unit_c0() {
        crate::ensure_symbolica_license();
        // Invariants are lexicographic pairwise (s01, s02, s12); the C0 legs come out in the
        // physical (adjacent, opposite) order (leg01, leg12, leg02) = (s01, s12, s02).
        let r = reduce(&scalar_family(
            vec![Atom::Zero, Atom::Zero, Atom::Zero],
            vec![Atom::num(1), Atom::num(2), Atom::num(3)],
        ))
        .unwrap();
        assert_eq!(r.terms.len(), 1);
        let (coeff, master) = &r.terms[0];
        assert_eq!(*coeff, Atom::num(1));
        match master {
            MasterIntegral::Triangle {
                p1_sq,
                p2_sq,
                p12_sq,
                ..
            } => {
                assert_eq!(*p1_sq, Atom::num(1));
                assert_eq!(*p2_sq, Atom::num(3));
                assert_eq!(*p12_sq, Atom::num(2));
            }
            other => panic!("expected a triangle master, got {other:?}"),
        }
    }

    #[test]
    fn gram_solve_matrix_handles_singular_gram() {
        crate::ensure_symbolica_license();

        let g = vec![
            vec![Atom::num(2), Atom::num(1), Atom::num(2)],
            vec![Atom::num(1), Atom::num(3), Atom::num(1)],
            vec![Atom::num(2), Atom::num(1), Atom::num(2)],
        ];
        let rhs = vec![Atom::num(5), Atom::num(4), Atom::num(5)];
        let c = super::gram_solve_matrix(&g, &rhs); // must not panic on the singular Gram
        for i in 0..3 {
            let mut lhs = Atom::Zero;
            for (j, cj) in c.iter().enumerate() {
                lhs += &g[i][j] * cj;
            }
            assert_eq!(lhs, rhs[i], "row {i}: G c != rhs");
        }
    }

    #[test]
    /// A *raised propagator power* at on-shell kinematics. The rank-2 case above
    /// goes through `isp_project`; this one goes through the dotted FJT branch,
    /// which divides by `det(y) ~ delta` in `matrix_inv`. The adjugate numerators
    /// carry a matching `delta`, so every coefficient is finite in the limit --
    /// but only once the terms are over a common denominator. Substituting into
    /// an `expand()`ed sum leaves a literal `delta/delta` that evaluates to the
    /// indeterminate `0/0`; `together()` before the substitution is what makes
    /// the cancellation happen. Nothing covered raised powers on-shell before.
    fn on_shell_raised_power_triangle_has_finite_coefficients() {
        crate::ensure_symbolica_license();
        let fam = RecurrenceInput {
            masses_squared: (0..3).map(|_| Atom::num(1)).collect(),
            invariants: vec![Atom::Zero, Atom::Zero, Atom::num(2) / Atom::num(5)],
            powers: vec![2, 1, 1],
            numerator: Atom::num(1),
        };
        let r = reduce(&fam).unwrap();
        assert_eq!(r.terms.len(), 3);
        // Every coefficient must be a finite rational function of `d`. The
        // failure this pins produced Symbolica's indeterminate glyph instead.
        for (c, m) in &r.terms {
            let text = c.to_string();
            assert!(
                !text.contains('\u{00bf}') && !text.contains('\u{29de}'),
                "indeterminate coefficient {text} on {m:?}"
            );
        }
        // The two bubbles enter with equal and opposite coefficients.
        let bub: Vec<&Atom> = r
            .terms
            .iter()
            .filter(|(_, m)| matches!(m, MasterIntegral::Bubble { .. }))
            .map(|(c, _)| c)
            .collect();
        assert_eq!(bub.len(), 2);
        assert_eq!((bub[0] + bub[1]).expand(), Atom::Zero);
    }

    #[test]
    fn on_shell_massless_triangle_rank2_regularizes() {
        crate::ensure_symbolica_license();
        let kq1 = crate::symbols::scalar_product(&(Atom::var(S.k)), &(Atom::var(S.q1)));
        let fam = RecurrenceInput {
            masses_squared: (0..3).map(|_| Atom::num(1)).collect(),
            invariants: vec![Atom::Zero, Atom::Zero, Atom::num(2) / Atom::num(5)],
            powers: vec![1, 1, 1],
            numerator: &kq1 * &kq1,
        };
        let r = reduce(&fam).unwrap(); // must NOT panic
        assert_eq!(
            r.terms.len(),
            2,
            "the C0 (coeff ~ delta^2) should drop, leaving 2 bubbles"
        );
        assert!(
            r.terms
                .iter()
                .all(|(_, m)| matches!(m, MasterIntegral::Bubble { .. }))
        );
        let coeffs: Vec<Atom> = r.terms.iter().map(|(c, _)| c.clone()).collect();
        assert!(coeffs.contains(&(Atom::num(1) / Atom::num(20))));
        assert!(coeffs.contains(&(Atom::num(-1) / Atom::num(20))));
    }

    #[test]
    fn mixed_dotted_and_numerator_reduces_to_masters() {
        crate::ensure_symbolica_license();
        let v = |s: &str| Atom::var(symbol!(format!("oneloopmaster::{s}")));
        let mut bub = family(vec![v("m1sq"), v("m2sq")], vec![v("psq")], vec![2, 1]);
        bub.numerator = super::dot_lq(0);
        let mut bx = family(
            vec![v("m1sq"), v("m2sq"), v("m3sq"), v("m4sq")],
            vec![v("p1"), v("p2"), v("p3"), v("p4"), v("sinv"), v("tinv")],
            vec![2, 1, 1, 1],
        );
        bx.numerator = super::dot_ll();
        for fam in [bub, bx] {
            let r = reduce(&fam).unwrap();
            assert!(!r.terms.is_empty());
            for (c, m) in &r.terms {
                assert!(c.to_string().is_ascii(), "non-finite mixed coeff: {c}");
                assert!(matches!(
                    m,
                    MasterIntegral::Box { .. }
                        | MasterIntegral::Triangle { .. }
                        | MasterIntegral::Bubble { .. }
                        | MasterIntegral::Tadpole { .. }
                ));
            }
        }
    }

    #[test]
    fn dotted_box_reduces_to_masters() {
        crate::ensure_symbolica_license();
        let p1 = Atom::var(symbol!("oneloopmaster::p1"));
        let p2 = Atom::var(symbol!("oneloopmaster::p2"));
        let p3 = Atom::var(symbol!("oneloopmaster::p3"));
        let p4 = Atom::var(symbol!("oneloopmaster::p4"));
        let s = Atom::var(symbol!("oneloopmaster::sinv"));
        let t = Atom::var(symbol!("oneloopmaster::tinv"));
        let m1 = Atom::var(symbol!("oneloopmaster::m1sq"));
        let m2 = Atom::var(symbol!("oneloopmaster::m2sq"));
        let m3 = Atom::var(symbol!("oneloopmaster::m3sq"));
        let m4 = Atom::var(symbol!("oneloopmaster::m4sq"));
        let r = reduce(&family(
            vec![m1, m2, m3, m4],
            vec![p1, p2, p3, p4, s, t],
            vec![2, 2, 1, 1],
        ))
        .unwrap();
        let boxes = r
            .terms
            .iter()
            .filter(|(_, m)| matches!(m, MasterIntegral::Box { .. }))
            .count();
        let triangles = r
            .terms
            .iter()
            .filter(|(_, m)| matches!(m, MasterIntegral::Triangle { .. }))
            .count();
        assert_eq!(boxes, 1);
        assert!(triangles >= 1);
        assert!(r.terms.iter().all(|(_, m)| matches!(
            m,
            MasterIntegral::Box { .. }
                | MasterIntegral::Triangle { .. }
                | MasterIntegral::Bubble { .. }
                | MasterIntegral::Tadpole { .. }
        )));
    }

    #[test]
    fn pinched_box_routes_to_a_triangle() {
        crate::ensure_symbolica_license();
        let [p1, p2, p3, p4, s, t, m1, m2, m3, m4] = box_syms();

        let r = reduce(&family(
            vec![m1.clone(), m2.clone(), m3.clone(), m4],
            vec![p1.clone(), p2.clone(), p3, p4.clone(), s, t],
            vec![1, 1, 1, 0],
        ))
        .unwrap();
        // lex invariants: pinching line 3 leaves triangle{0,1,2} with legs
        // (leg01, leg12, leg02) = (p1, p4, p2).
        assert!(r.terms.iter().any(|(_, m)| matches!(
            m,
            MasterIntegral::Triangle { p1_sq, p2_sq, p12_sq, m1_sq, m2_sq, m3_sq }
                if *p1_sq == p1 && *p2_sq == p4 && *p12_sq == p2
                    && *m1_sq == m1 && *m2_sq == m2 && *m3_sq == m3
        )));
        assert!(
            r.terms
                .iter()
                .all(|(_, m)| !matches!(m, MasterIntegral::Box { .. }))
        );
    }

    #[test]
    fn box_with_linear_numerator_reduces_to_masters() {
        crate::ensure_symbolica_license();
        let [p1, p2, p3, p4, s, t, m1, m2, m3, m4] = box_syms();
        // numerator = (l.q1) + 3*(l.q3) - 2*(l.l)
        let numerator = crate::symbols::scalar_product(&(Atom::var(S.k)), &(Atom::var(S.q1)))
            + Atom::num(3) * crate::symbols::scalar_product(&(Atom::var(S.k)), &(Atom::var(S.q3)))
            - Atom::num(2) * crate::symbols::scalar_product(&(Atom::var(S.k)), &(Atom::var(S.k)));
        let fam = RecurrenceInput {
            masses_squared: vec![m1, m2, m3, m4],
            invariants: vec![p1, p2, p3, p4, s, t],
            powers: vec![1, 1, 1, 1],
            numerator,
        };
        let r = reduce(&fam).unwrap();
        assert!(
            r.terms
                .iter()
                .any(|(_, m)| matches!(m, MasterIntegral::Box { .. }))
        );
        assert!(
            r.terms
                .iter()
                .any(|(_, m)| matches!(m, MasterIntegral::Triangle { .. }))
        );
        assert!(r.terms.iter().all(|(_, m)| matches!(
            m,
            MasterIntegral::Box { .. }
                | MasterIntegral::Triangle { .. }
                | MasterIntegral::Bubble { .. }
                | MasterIntegral::Tadpole { .. }
        )));
    }

    #[test]
    fn scalar_box_reduces_to_unit_d0() {
        crate::ensure_symbolica_license();
        let r = reduce(&scalar_family(
            vec![Atom::Zero, Atom::Zero, Atom::Zero, Atom::Zero],
            vec![
                Atom::num(1),
                Atom::num(2),
                Atom::num(3),
                Atom::num(4),
                Atom::num(5),
                Atom::num(6),
            ],
        ))
        .unwrap();
        assert_eq!(r.terms.len(), 1);
        let (coeff, master) = &r.terms[0];
        assert_eq!(*coeff, Atom::num(1));
        match master {
            // lex invariants (s01..s23) = [1..6]; the Mandelstam diagonals are s02=2, s13=5.
            MasterIntegral::Box { s, t, .. } => {
                assert_eq!(*s, Atom::num(2));
                assert_eq!(*t, Atom::num(5));
            }
            other => panic!("expected a box master, got {other:?}"),
        }
    }

    /// A negative index has no base case: `reduce_cayley` walks it further
    /// negative every level and overruns the stack, which aborts the process.
    #[test]
    fn rejects_a_negative_propagator_index() {
        crate::ensure_symbolica_license();
        let fam = family(
            vec![Atom::Zero, Atom::Zero],
            vec![Atom::var(S.psq)],
            vec![-1, 1],
        );
        let err = reduce(&fam).unwrap_err().to_string();
        assert!(err.contains("unsupported propagator powers"), "{err}");
    }

    #[test]
    fn rejects_obsolete_symbols_in_numerators_and_kinematics() {
        crate::ensure_symbolica_license();
        let mut fam = family(
            vec![Atom::num(1), Atom::num(1)],
            vec![Atom::var(S.psq)],
            vec![1, 1],
        );
        fam.numerator = function!(symbol!("oneloopreduce::dot"), S.k, S.q1);
        assert!(matches!(
            reduce(&fam),
            Err(crate::OneLoopError::ObsoleteSymbol { .. })
        ));
        fam.numerator = Atom::num(1);
        fam.masses_squared[0] = Atom::var(symbol!("oneloopreduce::m2"));
        assert!(matches!(
            reduce(&fam),
            Err(crate::OneLoopError::ObsoleteSymbol { .. })
        ));
        fam.masses_squared[0] = Atom::num(1);
        fam.invariants[0] = Atom::var(symbol!("oneloopreduce::psq"));
        assert!(matches!(
            reduce(&fam),
            Err(crate::OneLoopError::ObsoleteSymbol { .. })
        ));
    }

    /// An explicit diagnostic grant rejects work before entering recurrence.
    #[test]
    fn explicit_total_index_diagnostics_reject_before_recurrence() {
        crate::ensure_symbolica_license();
        let psq = Atom::var(S.psq);
        // The last one wraps to -2 under plain `sum`, so it also pins `checked_add`.
        for e in [vec![5000, 1], vec![i32::MAX, i32::MAX]] {
            let fam = family(vec![Atom::Zero, Atom::Zero], vec![psq.clone()], e);
            let err = crate::ReductionOptions::default()
                .with_diagnostics()
                .reduce(&fam)
                .unwrap_err()
                .to_string();
            assert!(err.contains(&format!("at most {MAX_TOTAL_INDEX}")), "{err}");
        }
    }

    /// A massless tadpole bottoms out without recursing, so it can sit exactly on
    /// the bound and pin the off-by-one that a real reduction is far too slow to.
    #[test]
    fn total_index_diagnostics_are_disabled_by_default_and_keep_their_boundary() {
        crate::ensure_symbolica_license();
        let at = family(vec![Atom::Zero], vec![], vec![MAX_TOTAL_INDEX]);
        let past = family(vec![Atom::Zero], vec![], vec![MAX_TOTAL_INDEX + 1]);
        assert!(reduce(&at).is_ok());
        assert!(reduce(&past).is_ok());
        let options = crate::ReductionOptions::default().with_diagnostics();
        assert!(options.reduce(&at).is_ok());
        assert!(options.reduce(&past).is_err());
        for power in [5000, i32::MAX] {
            assert!(reduce(&family(vec![Atom::Zero], vec![], vec![power])).is_ok());
        }
    }

    #[test]
    fn a_real_resource_refusal_aborts_uncapped_work_and_keeps_later_requests_healthy() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use symbolica::utils::resources::{ResourceBudget, ResourceLimitError, ResourceLimits};

        crate::ensure_symbolica_license();
        let input = family(vec![Atom::num(2)], vec![], vec![5000]);
        let polls = AtomicUsize::new(0);
        const LIMIT: u64 = 64 * 1024 * 1024;
        let budget = ResourceBudget::with_memory_monitor(
            ResourceLimits {
                max_process_rss_bytes: Some(LIMIT),
                ..Default::default()
            },
            move || {
                Ok(if polls.fetch_add(1, Ordering::Relaxed) < 40 {
                    0
                } else {
                    LIMIT + 1
                })
            },
        );
        let error = budget.run(|| reduce(&input)).unwrap_err();
        assert!(matches!(
            std::error::Error::source(&error).and_then(|cause| cause.downcast_ref::<ResourceLimitError>()),
            Some(ResourceLimitError::Memory { rss_bytes, additional_bytes: 0, limit_bytes })
                if *rss_bytes == LIMIT + 1 && *limit_bytes == LIMIT
        ));
        let healthy = family(vec![Atom::num(2)], vec![], vec![2]);
        assert!(!reduce(&healthy).unwrap().terms.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn growing_the_recurrence_stack_preserves_thread_and_resource_scope() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread::ThreadId;
        use symbolica::utils::resources::{ResourceBudget, ResourceLimitError, ResourceLimits};

        fn descend(depth: usize, thread: ThreadId) -> Result<(), crate::OneLoopError> {
            super::with_stack(|| {
                let frame = [depth as u8; 8192];
                assert_eq!(std::thread::current().id(), thread);
                if depth != 0 {
                    descend(depth - 1, thread)?;
                }
                std::hint::black_box(frame);
                Ok(())
            })
        }
        fn on_small_stack(thread: ThreadId) -> Result<(), crate::OneLoopError> {
            const INITIAL_STACK: usize = 2 * 1024 * 1024;
            symbolica::utils::resources::check_allocation(INITIAL_STACK as u64)?;
            // Fix the initial size even when the test runner has a large stack.
            stacker::grow(INITIAL_STACK, || descend(1200, thread))
        }
        let thread = std::thread::current().id();
        const LIMIT: u64 = 64 * 1024 * 1024;
        ResourceBudget::with_memory_monitor(
            ResourceLimits {
                max_process_rss_bytes: Some(LIMIT),
                ..Default::default()
            },
            || Ok(0),
        )
        .run(|| on_small_stack(thread))
        .unwrap();
        let polls = AtomicUsize::new(0);
        let error = ResourceBudget::with_memory_monitor(
            ResourceLimits {
                max_process_rss_bytes: Some(LIMIT),
                ..Default::default()
            },
            move || {
                Ok(if polls.fetch_add(1, Ordering::Relaxed) < 1000 {
                    0
                } else {
                    LIMIT + 1
                })
            },
        )
        .run(|| on_small_stack(thread))
        .unwrap_err();
        assert!(matches!(
            error,
            crate::OneLoopError::Resource(ResourceLimitError::Memory { .. })
        ));
    }

    /// Dotted lines still reduce, and to the right thing: the massless bubble has
    /// the closed form `G(a1,a2)/G(1,1)` from the Gamma-function ratio, so the
    /// coefficient can be checked against a value rather than just a shape.
    #[test]
    fn dotted_massless_bubbles_match_the_closed_form() {
        crate::ensure_symbolica_license();
        let psq = Atom::var(S.psq);
        let d = Atom::var(S.d);
        // (a1, a2) -> G(a1,a2)/G(1,1)
        let cases = [
            (vec![2, 1], -(&d - Atom::num(3)) / &psq),
            (
                vec![3, 2],
                (Atom::num(8) - &d) * (&d - Atom::num(3)) * (&d - Atom::num(5))
                    / (Atom::num(2) * &psq * &psq * &psq),
            ),
        ];
        for (exponents, want) in cases {
            let r = reduce(&family(
                vec![Atom::Zero, Atom::Zero],
                vec![psq.clone()],
                exponents.clone(),
            ))
            .unwrap()
            .simplify()
            .unwrap();
            assert_eq!(r.terms.len(), 1, "{exponents:?}");
            let (got, master) = &r.terms[0];
            assert_eq!(
                master,
                &MasterIntegral::Bubble {
                    p_sq: psq.clone(),
                    m1_sq: Atom::Zero,
                    m2_sq: Atom::Zero,
                }
            );
            assert_eq!((got - &want).expand(), Atom::Zero, "{exponents:?}: {got}");
        }
    }

    /// At zero momentum with equal masses the two lines of a bubble are the same
    /// denominator, so `[2,2]` and `[3,1]` are the power-4 tadpole. Both used to
    /// come back as an indeterminate coefficient reported as success.
    #[test]
    fn coincident_dotted_bubbles_are_the_power_four_tadpole() {
        crate::ensure_symbolica_license();
        let d = Atom::var(S.d);
        let msq = Atom::var(symbol!("oneloopmaster::msq"));
        for mass in [Atom::num(1), msq] {
            let m6 = &mass * &mass * &mass;
            let want = (&d - Atom::num(2)) * (&d - Atom::num(4)) * (&d - Atom::num(6))
                / (Atom::num(48) * &m6);
            let tadpole = reduce(&family(vec![mass.clone()], vec![], vec![4]))
                .unwrap()
                .simplify()
                .unwrap();
            assert_eq!((&tadpole.terms[0].0 - &want).expand(), Atom::Zero);
            for exponents in [vec![2, 2], vec![3, 1], vec![1, 3]] {
                let r = reduce(&family(
                    vec![mass.clone(), mass.clone()],
                    vec![Atom::Zero],
                    exponents.clone(),
                ))
                .unwrap()
                .simplify()
                .unwrap();
                assert_eq!(r.terms.len(), 1, "{exponents:?}");
                let (got, master) = &r.terms[0];
                assert_eq!(master, &MasterIntegral::Tadpole { m_sq: mass.clone() });
                assert_eq!((got - &want).expand(), Atom::Zero, "{exponents:?}: {got}");
            }
        }
    }

    /// Two lines of a triangle merge only if they agree against the third line
    /// too; `[0, s, s]` does, `[0, 0, 2/5]` (the on-shell regression above) does
    /// not, and must keep reducing as a triangle.
    #[test]
    fn coincident_lines_merge_only_when_every_invariant_agrees() {
        crate::ensure_symbolica_license();
        let s = Atom::var(S.psq);
        let one = || Atom::num(1);
        let merged = reduce(&family(
            vec![one(), one(), one()],
            vec![Atom::Zero, s.clone(), s.clone()],
            vec![2, 1, 1],
        ))
        .unwrap()
        .simplify()
        .unwrap();
        let bubble = reduce(&family(vec![one(), one()], vec![s.clone()], vec![3, 1]))
            .unwrap()
            .simplify()
            .unwrap();
        assert_eq!(merged.terms.len(), bubble.terms.len());
        for ((c1, m1), (c2, m2)) in merged.terms.iter().zip(&bubble.terms) {
            assert_eq!(m1, m2);
            assert_eq!((c1 - c2).expand(), Atom::Zero);
        }

        let kept = super::merge_coincident_lines(&family(
            vec![one(), one(), one()],
            vec![Atom::Zero, Atom::Zero, Atom::num(2) / Atom::num(5)],
            vec![2, 1, 1],
        ));
        assert!(kept.unwrap().is_none());
    }

    #[test]
    fn a_non_finite_reduction_is_an_error() {
        crate::ensure_symbolica_license();
        let bad = super::Reduction {
            terms: vec![(
                Atom::num(1) / Atom::Zero,
                MasterIntegral::Tadpole { m_sq: Atom::num(1) },
            )],
        };
        let err = super::check_finite(&bad).unwrap_err().to_string();
        assert!(err.contains("not finite"), "{err}");
        let fine = super::Reduction {
            terms: vec![(Atom::num(1), MasterIntegral::Tadpole { m_sq: Atom::num(1) })],
        };
        assert!(super::check_finite(&fine).is_ok());
    }

    fn with_numerator(mut fam: RecurrenceInput, numerator: Atom) -> RecurrenceInput {
        fam.numerator = numerator;
        fam
    }

    #[test]
    fn rejects_numerators_the_reducer_would_carry_into_a_coefficient() {
        crate::ensure_symbolica_license();
        let k = Atom::var(S.k);
        let dot = |a: &Atom, b: &Atom| crate::symbols::scalar_product(a, b);
        let q = |i: usize| Atom::var(symbol!(format!("oneloopmaster::q{i}")));
        let pol = Atom::var(symbol!("oneloopmaster::test_polarization"));
        let bubble = || family(vec![Atom::num(1); 2], vec![Atom::num(-2)], vec![1, 1]);
        let triangle = || {
            family(
                vec![Atom::num(1); 3],
                vec![Atom::num(-1), Atom::num(-2), Atom::num(-3)],
                vec![1, 1, 1],
            )
        };
        let cases = [
            ("dot(k,q4) on a bubble", bubble(), dot(&k, &q(4))),
            ("dot(k,q2) on a bubble", bubble(), dot(&k, &q(2))),
            ("dot(k,q3) on a triangle", triangle(), dot(&k, &q(3))),
            (
                "dot(k,eps)",
                bubble(),
                function!(symbol!("test::unsupported_dot"), &k, &pol),
            ),
            ("bare k", bubble(), k.clone()),
            ("1/dot(k,k)", bubble(), Atom::num(1) / dot(&k, &k)),
            (
                "dot(k,q1)^(1/2)",
                bubble(),
                dot(&k, &q(1)).pow(Atom::num(1) / Atom::num(2)),
            ),
        ];
        for (what, fam, num) in cases {
            let err = reduce(&with_numerator(fam, num)).unwrap_err();
            assert!(
                matches!(err, crate::OneLoopError::UnsupportedNumerator { .. }),
                "{what}: {err}"
            );
        }
    }

    #[test]
    fn accepts_numerators_with_external_scalar_prefactors() {
        crate::ensure_symbolica_license();
        let k = Atom::var(S.k);
        let d = Atom::var(S.d);
        let dot = |a: &Atom, b: &Atom| crate::symbols::scalar_product(a, b);
        let q = |i: usize| Atom::var(symbol!(format!("oneloopmaster::q{i}")));
        // A function of external momenta over a pole in d, times q1.q1.
        let prefactor = function!(symbol!("oneloopmaster::test_blob"), q(1)) / (&d - Atom::num(4))
            * dot(&q(1), &q(1));
        let bubble = family(vec![Atom::num(1); 2], vec![Atom::num(-2)], vec![1, 1]);
        let r = reduce(&with_numerator(bubble, &prefactor * dot(&k, &q(1)))).unwrap();
        assert!(!r.terms.is_empty());
        // A tadpole's Gram is symbolic, so all three directions are fine there.
        let tadpole = family(vec![Atom::num(1)], vec![], vec![1]);
        let r = reduce(&with_numerator(tadpole, dot(&k, &q(3)).pow(Atom::num(2)))).unwrap();
        assert_eq!(r.terms.len(), 1);
        // Degree twenty remains supported after removing the factorial ceiling.
        let tadpole = family(vec![Atom::num(1)], vec![], vec![1]);
        let r = reduce(&with_numerator(tadpole, dot(&k, &k).pow(Atom::num(20))));
        assert!(r.is_ok(), "{r:?}");
    }

    #[test]
    fn exact_coefficients_beyond_machine_factorials() {
        crate::ensure_symbolica_license();
        let x = symbol!("exact_numerator_test::x");
        let y = symbol!("exact_numerator_test::y");
        for degree in [21, 22] {
            let expression =
                Atom::num(3) * Atom::var(x).pow(Atom::num(degree)) * Atom::var(y).pow(Atom::num(2))
                    + Atom::num(5);
            let terms = super::extract_monomials(&expression, &[x, y])
                .unwrap()
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(terms.len(), 2);
            assert_eq!(terms[&vec![degree, 2]], Atom::num(3));
            assert_eq!(terms[&vec![0, 0]], Atom::num(5));
        }
    }

    #[test]
    fn massive_tadpole_numerator_degrees_twenty_one_and_twenty_two() {
        crate::ensure_symbolica_license();
        let mass_squared = Atom::num(2);
        for degree in [21, 22] {
            // Polynomial terms integrate to zero in dimensional regularization:
            // (k.k)^n / (k.k-m^2) = (m^2)^n / (k.k-m^2) + polynomial.
            let result = reduce(&with_numerator(
                family(vec![mass_squared.clone()], vec![], vec![1]),
                super::dot_ll().pow(Atom::num(degree)),
            ))
            .unwrap()
            .simplify()
            .unwrap();
            assert_eq!(result.terms.len(), 1);
            assert_eq!(result.terms[0].0, mass_squared.pow(Atom::num(degree)));
            assert!(matches!(&result.terms[0].1,
                MasterIntegral::Tadpole { m_sq } if *m_sq == mass_squared));
        }
    }

    #[test]
    fn transformed_denominator_degrees_combine_without_a_factorial_cap() {
        crate::ensure_symbolica_license();
        let topology = super::bubble_topo(1, 1, &Atom::num(5), &Atom::num(2), &Atom::num(3));
        for (second_degree, expected) in [(10, Atom::num((1, 1024))), (11, Atom::num((-1, 2048)))] {
            let input =
                super::dot_ll().pow(Atom::num(11)) * super::dot_lq(0).pow(Atom::num(second_degree));
            super::check_numerator(&input, 2).unwrap();
            // k.k = D1+2 and k.q = (D2-D1-4)/2. The leading D1
            // coefficient is independently (-1)^m / 2^m, at degree 11+m.
            let transformed = topology.rule_ll.pow(Atom::num(11))
                * topology.rule_lq[0].pow(Atom::num(second_degree));
            let terms = super::extract_monomials(&transformed, &topology.den_syms)
                .unwrap()
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(terms[&vec![11 + second_degree, 0]], expected);
        }
    }

    #[test]
    fn exponent_storage_boundaries_are_checked_without_large_reductions() {
        crate::ensure_symbolica_license();
        let x = Atom::var(symbol!("exponent_storage_test::x"));
        let y = Atom::var(symbol!("exponent_storage_test::y"));
        let largest = Atom::num(i32::MAX);
        assert_eq!(
            super::checked_nonnegative_exponent(largest.as_view(), "testing a literal").unwrap(),
            i32::MAX,
        );
        for exponent in [
            Atom::num(i64::from(i32::MAX) + 1),
            Atom::num(symbolica::domains::integer::Integer::factorial(50)),
        ] {
            let error =
                super::checked_nonnegative_exponent(exponent.as_view(), "testing a literal")
                    .unwrap_err();
            assert!(error.to_string().contains("i32 exponent storage overflow"));
            let error = super::validate_polynomial_exponents(
                &x.pow(&exponent),
                std::slice::from_ref(&x),
                "testing coefficient collection",
            )
            .unwrap_err();
            assert!(error.to_string().contains("i32 exponent storage overflow"));
        }
        // Total degree is not the coordinate storage bound: both powers fit.
        super::validate_polynomial_exponents(
            &(x.pow(&largest) * &y),
            &[x.clone(), y],
            "testing independent coordinates",
        )
        .unwrap();
        assert_eq!(
            super::mono_mul(
                &[(vec![i32::MAX], Atom::num(1))],
                &[(vec![0], Atom::num(1))]
            )
            .unwrap()[0]
                .0,
            vec![i32::MAX],
        );
        let error = super::mono_mul(
            &[(vec![i32::MAX], Atom::num(1))],
            &[(vec![1], Atom::num(1))],
        )
        .unwrap_err();
        assert!(error.to_string().contains("2147483648"));
        assert_eq!(
            super::checked_index_subtract(0, i32::MAX, "testing a shift").unwrap(),
            -i32::MAX
        );
        let error = super::checked_index_subtract(i32::MIN, 1, "testing a shift").unwrap_err();
        assert!(error.to_string().contains("-2147483649"));
    }

    #[test]
    fn coefficient_collection_checks_whole_atom_variables_and_nonpolynomial_powers() {
        crate::ensure_symbolica_license();
        let variable = function!(symbol!("coefficient_guard_test::dot"), 1, 2);
        super::validate_polynomial_exponents(
            &variable.pow(Atom::num(22)),
            std::slice::from_ref(&variable),
            "checking a scalar-product atom",
        )
        .unwrap();
        for expression in [
            variable.pow(Atom::num(-1)),
            variable.pow(Atom::num((1, 2))),
            Atom::num(2).pow(&variable),
        ] {
            let error = super::validate_polynomial_exponents(
                &expression,
                std::slice::from_ref(&variable),
                "checking polynomial syntax",
            )
            .unwrap_err();
            assert!(error.to_string().contains("must be polynomial"));
            assert!(!error.to_string().contains("storage overflow"));
        }
    }

    /// The reducer's scratch variables are fixed, interned names; an input using
    /// one would be silently read as the reducer's own variable.
    #[test]
    fn rejects_the_reducers_scratch_names_in_the_input() {
        crate::ensure_symbolica_license();
        let named = |n: &str| Atom::var(symbol!(format!("oneloopmaster::{n}")));
        let cases = [
            family(vec![named("reg_delta"); 2], vec![Atom::Zero], vec![1, 1]),
            with_numerator(family(vec![Atom::num(1)], vec![], vec![1]), named("xll")),
            with_numerator(family(vec![Atom::num(1)], vec![], vec![1]), named("xq2")),
            family(vec![Atom::num(1); 2], vec![named("den1")], vec![1, 1]),
            family(
                vec![named("routing_tmp_q1"), Atom::num(1)],
                vec![Atom::num(-2)],
                vec![1, 1],
            ),
        ];
        for fam in cases {
            let err = reduce(&fam).unwrap_err();
            assert!(
                matches!(err, crate::OneLoopError::InvalidFamily(..)),
                "{err}"
            );
        }
        // Names that merely share a prefix are the caller's to use.
        for n in ["xq", "dense", "reg_delta2", "xllx"] {
            let fam = family(vec![named(n); 2], vec![Atom::num(-2)], vec![1, 1]);
            assert!(reduce(&fam).is_ok(), "{n}");
        }
    }

    #[test]
    fn malformed_families_are_errors_not_panics() {
        crate::ensure_symbolica_license();
        let ok = || family(vec![Atom::num(1); 2], vec![Atom::num(-2)], vec![1, 1]);
        let breakages: [fn(&mut RecurrenceInput); 3] = [
            |f| {
                f.masses_squared.clear();
                f.invariants.clear();
                f.powers.clear();
            },
            |f| f.powers = vec![1],
            |f| f.invariants.push(Atom::num(3)),
        ];
        for (i, breakage) in breakages.into_iter().enumerate() {
            let mut fam = ok();
            breakage(&mut fam);
            let err = reduce(&fam).unwrap_err();
            assert!(
                matches!(err, crate::OneLoopError::InvalidFamily(..)),
                "{i}: {err}"
            );
        }
        assert!(reduce(&ok()).is_ok());
    }
}

#[cfg(test)]
mod high_point_tests {
    use super::{high_point_coeffs, modified_cayley};
    use symbolica::atom::{Atom, AtomCore};

    /// Pairwise invariants `(r_i - r_j)^2`, lexicographic, for offsets `r_i`
    /// given as vectors in a space with metric `diag(+1, -1, -1, ...)`.
    fn lex_invariants(r: &[Vec<i64>]) -> Vec<Atom> {
        let sq = |v: Vec<i64>| {
            v.iter()
                .enumerate()
                .map(|(a, x)| if a == 0 { x * x } else { -x * x })
                .sum::<i64>()
        };
        let mut out = Vec::new();
        for i in 0..r.len() {
            for j in i + 1..r.len() {
                out.push(Atom::num(sq(r[i]
                    .iter()
                    .zip(&r[j])
                    .map(|(a, b)| a - b)
                    .collect())));
            }
        }
        out
    }

    /// `B = sum_i c_i`. In `d` dimensions `I_N = sum_i c_i I_{N-1}^(i) +
    /// (N - d - 1) B I_N^(d+2)`; the reducer keeps only the sum.
    fn dropped_coefficient(r: &[Vec<i64>]) -> Atom {
        let masses: Vec<Atom> = (1..=r.len() as i64).map(Atom::num).collect();
        let y = modified_cayley(&masses, &lex_invariants(r));
        high_point_coeffs(&y)
            .iter()
            .fold(Atom::Zero, |acc, c| acc + c)
            .cancel()
    }

    const HEXAGON_4D: [[i64; 4]; 6] = [
        [0, 0, 0, 0],
        [3, 1, 0, 0],
        [5, 2, 1, 0],
        [7, 1, 3, 2],
        [4, -1, 2, 5],
        [2, 3, -2, 1],
    ];

    /// Six points in four dimensions: `B = 0`, so the step is exact in `d`.
    #[test]
    fn hexagon_step_is_exact_for_four_dimensional_kinematics() {
        crate::ensure_symbolica_license();
        let r: Vec<Vec<i64>> = HEXAGON_4D.iter().map(|v| v.to_vec()).collect();
        assert_eq!(dropped_coefficient(&r), Atom::Zero);
    }

    #[test]
    fn hexagon_step_is_not_exact_for_five_dimensional_kinematics() {
        crate::ensure_symbolica_license();
        let r: Vec<Vec<i64>> = HEXAGON_4D
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut v = v.to_vec();
                v.push([0, 0, 1, 3, 2, 5][i]);
                v
            })
            .collect();
        assert_ne!(dropped_coefficient(&r), Atom::Zero);
    }

    /// The pentagon's `(4 - d) B = 2 eps B` term is dropped; `I_5^(6-2eps)` is
    /// finite, so that is `O(eps)`.
    #[test]
    fn pentagon_step_drops_a_nonzero_order_eps_term() {
        crate::ensure_symbolica_license();
        let r: Vec<Vec<i64>> = HEXAGON_4D[..5].iter().map(|v| v.to_vec()).collect();
        assert_ne!(dropped_coefficient(&r), Atom::Zero);
    }
}

#[cfg(test)]
mod limit_tests {
    use super::{master_at_zero, reduce, reduce_core};
    use crate::masters::MasterIntegral;
    use crate::recurrence::RecurrenceInput;
    use crate::symbols::S;
    use symbolica::atom::{Atom, AtomCore};
    use symbolica::symbol;

    pub(super) fn n(x: i64) -> Atom {
        Atom::num(x)
    }

    pub(super) fn family(
        masses: Vec<Atom>,
        invariants: Vec<Atom>,
        exps: Vec<i32>,
        num: Atom,
    ) -> RecurrenceInput {
        RecurrenceInput {
            masses_squared: masses,
            invariants,
            powers: exps,
            numerator: num,
        }
    }

    fn combine(terms: Vec<(Atom, MasterIntegral)>) -> Vec<(Atom, MasterIntegral)> {
        let mut out: Vec<(Atom, MasterIntegral)> = Vec::new();
        for (c, m) in terms {
            match out.iter_mut().find(|(_, o)| *o == m) {
                Some(slot) => slot.0 = &slot.0 + &c,
                None => out.push((c, m)),
            }
        }
        out.retain(|(c, _)| !c.expand().is_zero());
        out
    }

    /// The on-shell limit with its own delta per zero invariant, taken one at a
    /// time, instead of `reduce`'s single shared delta.
    fn sequential_limit(fam: &RecurrenceInput) -> Vec<(Atom, MasterIntegral)> {
        let mut reg = fam.clone();
        let mut deltas = Vec::new();
        for s in reg.invariants.iter_mut().filter(|s| s.is_zero()) {
            let d = Atom::var(symbol!(format!(
                "oneloopmaster::test_delta{}",
                deltas.len()
            )));
            *s = d.clone();
            deltas.push(d);
        }
        let mut terms = reduce_core(&reg).unwrap().terms;
        for d in deltas.iter().rev() {
            terms = (terms.into_iter())
                .map(|(c, m)| {
                    let c = c
                        .together()
                        .replace(d.to_pattern())
                        .with(Atom::Zero)
                        .expand();
                    (c, master_at_zero(&m, d))
                })
                .collect();
        }
        combine(terms)
    }

    /// One shared delta is right only if the on-shell limit does not depend on
    /// how it is approached; check that against sending each leg on shell in turn.
    #[test]
    fn on_shell_limits_do_not_depend_on_the_path() {
        crate::ensure_symbolica_license();
        let k = Atom::var(S.k);
        let dot = |a: &Atom, b: &Atom| crate::symbols::scalar_product(a, b);
        let q = |i: usize| Atom::var(symbol!(format!("oneloopmaster::q{i}")));
        // Box lex order (01, 02, 03, 12, 13, 23) = (p1, s, p4, p2, t, p3).
        let two_legs = || vec![n(0), n(-7), n(-3), n(0), n(-5), n(-2)];
        let four_legs = || vec![n(0), n(-7), n(0), n(0), n(-5), n(0)];
        // Pentagon legs are 01, 12, 23, 34, 04; three of them on shell.
        let pentagon = || {
            vec![
                n(0),
                n(-7),
                n(-9),
                n(0),
                n(0),
                n(-5),
                n(-8),
                n(-2),
                n(-6),
                n(-3),
            ]
        };
        let cases = [
            family(vec![n(1); 4], two_legs(), vec![2, 1, 1, 1], n(1)),
            family(vec![n(1); 4], four_legs(), vec![1, 2, 1, 1], n(1)),
            family(
                vec![n(0); 4],
                four_legs(),
                vec![1; 4],
                dot(&k, &q(1)) * dot(&k, &q(2)),
            ),
            family(
                vec![n(1); 4],
                two_legs(),
                vec![1; 4],
                dot(&k, &q(1)) * dot(&k, &q(3)) * dot(&k, &k),
            ),
            family(
                vec![n(1); 3],
                vec![n(0), n(2) / n(5), n(0)],
                vec![2, 1, 1],
                n(1),
            ),
            family(
                vec![n(0); 3],
                vec![n(0), n(-3), n(0)],
                vec![1; 3],
                dot(&k, &q(1)).pow(n(2)),
            ),
            family(vec![n(1); 5], pentagon(), vec![1; 5], n(1)),
            family(vec![n(1); 5], pentagon(), vec![2, 1, 1, 1, 1], n(1)),
        ];
        for (i, fam) in cases.iter().enumerate() {
            let shared = combine(reduce(fam).unwrap().terms);
            let seq = sequential_limit(fam);
            assert_eq!(shared.len(), seq.len(), "case {i}");
            for (c, m) in &shared {
                let (c2, _) = seq.iter().find(|(_, m2)| m2 == m).unwrap();
                assert!((c - c2).expand().is_zero(), "case {i}, {m:?}: {c} vs {c2}");
            }
        }
    }
}
