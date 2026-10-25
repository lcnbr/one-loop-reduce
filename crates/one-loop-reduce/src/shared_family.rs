//! Adapt the shared HEP integral family to the private one-loop recurrence chart.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::{AtomicU64, Ordering},
};

use feynkit_graph::{IntegralFamily as SharedFamily, IntegralFamilyError};
use symbolica::utils::resources::{self, ResourceLimits};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    symbol,
};

use crate::{
    OneLoopError,
    recurrence::{RecurrenceInput, ReductionOptions},
    reduce::{
        Reduction, checked_exponent_add, checked_nonnegative_exponent, exponent_storage_error,
        validate_polynomial_exponents,
    },
    symbols::{S, scalar_product, validate_namespace},
};

fn invalid(message: impl ToString) -> OneLoopError {
    OneLoopError::InvalidFamily(message.to_string())
}

/// Reduce a shared family, given one signed power per denominator and a scalar
/// numerator. The dimension must be symbolic, so that `d`-dependent
/// coefficients keep their finite parts once expanded in epsilon.
/// Native partial-fraction work counts are unrestricted. Cooperative process-RSS
/// and elapsed-time checks honor any enclosing Symbolica resource budget;
/// resource failures preserve their typed cause rather than becoming invalid-input errors.
pub fn reduce_family(
    family: &SharedFamily,
    powers: &[i32],
    numerator: &Atom,
) -> Result<Reduction, OneLoopError> {
    ReductionOptions::default().reduce_family(family, powers, numerator)
}

impl ReductionOptions {
    pub fn reduce_family(
        self,
        family: &SharedFamily,
        powers: &[i32],
        numerator: &Atom,
    ) -> Result<Reduction, OneLoopError> {
        resources::checkpoint()?;
        if family.loop_momenta().len() != 1 {
            return Err(OneLoopError::UnsupportedLoopOrder {
                found: family.loop_momenta().len(),
            });
        }
        if powers.len() != family.denominators().len() {
            return Err(invalid(
                "supply one signed power per shared-family denominator",
            ));
        }
        let dimension = family.kinematics().dimension().to_symbolic();
        if !matches!(dimension.as_view(), AtomView::Var(_)) {
            return Err(invalid(
                "one-loop reduction requires a symbolic Kinematics dimension; use Kinematics(S(\"D\")) to retain epsilon-dependent finite terms",
            ));
        }
        for input in std::iter::once(numerator)
            .chain(family.denominators())
            .chain(family.loop_momenta())
            .chain(family.external_momenta())
            .chain(std::iter::once(&dimension))
        {
            resources::checkpoint()?;
            validate_namespace(input)?;
        }
        // Check actual loop-coordinate storage before negative powers expand.
        // Pure scalar denominators are coefficients, not recurrence indices.
        let mut positive_total = 0i128;
        for (denominator, power) in family.denominators().iter().zip(powers) {
            resources::checkpoint()?;
            if contains_loop(denominator, &family.loop_momenta()[0]) {
                if *power < 0 && power.checked_neg().is_none() {
                    return Err(exponent_storage_error(
                        "moving a negative propagator power into loop-coordinate storage",
                        -i64::from(*power),
                    ));
                }
                positive_total += i128::from((*power).max(0));
            }
        }
        self.validate_total(powers, positive_total)?;
        // Validate before dropping a scaleless sector, including hidden loop
        // dependence inside arbitrary functions or uncontracted tensors.
        validate_numerator(family, &family.kinematics().apply(numerator))?;
        let mut hygiene = ChartSymbols::new(family, numerator)?;
        let mut result = Vec::new();
        let partial_fractions = family
            .partial_fraction(powers, None, ResourceLimits::default())
            .map_err(|error| match error {
                IntegralFamilyError::Resource(error) => OneLoopError::Resource(error),
                error => invalid(error),
            })?;
        for (coefficient, powers) in partial_fractions {
            resources::checkpoint()?;
            let reduction = reduce_sector(family, &powers, numerator, &mut hygiene, self)?;
            let coefficient = hygiene.protect(&coefficient, &[]);
            result.extend(reduction.terms.into_iter().map(|(value, master)| {
                let value = (coefficient.clone() * value)
                    .replace(Atom::var(S.d))
                    .with(dimension.clone());
                (
                    hygiene.restore(&value),
                    master.map_arguments(|arg| hygiene.restore(arg)),
                )
            }));
        }
        let reduction = Reduction { terms: result }.simplify()?;
        resources::checkpoint()?;
        Ok(reduction)
    }
}

/// Keep user parameters independent from recurrence polynomial coordinates.
/// These substitutions never change the shared family or its assumptions.
struct ChartSymbols {
    nonce: u64,
    used: BTreeSet<Symbol>,
    forward: BTreeMap<Symbol, Atom>,
    backward: BTreeMap<Symbol, Atom>,
}

impl ChartSymbols {
    fn new(family: &SharedFamily, numerator: &Atom) -> Result<Self, OneLoopError> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let mut used = BTreeSet::new();
        let mut collect = |expression: &Atom| {
            expression.visitor(&mut |view| {
                if let Some(symbol) = view.get_symbol() {
                    used.insert(symbol);
                }
                true
            });
        };
        for expression in std::iter::once(numerator)
            .chain(family.denominators())
            .chain(family.loop_momenta())
            .chain(family.external_momenta())
        {
            collect(expression);
        }
        collect(&family.kinematics().apply(numerator));
        collect(&family.kinematics().dimension().to_symbolic());
        // Assumptions can introduce scalar parameters absent from denominators.
        for left in family.external_momenta() {
            for right in family.external_momenta() {
                collect(
                    &family
                        .kinematics()
                        .scalar_product(left, right)
                        .map_err(invalid)?,
                );
            }
        }
        Ok(Self {
            nonce: NEXT.fetch_add(1, Ordering::Relaxed),
            used,
            forward: BTreeMap::new(),
            backward: BTreeMap::new(),
        })
    }

    fn protect(&mut self, expression: &Atom, chart_products: &[Atom]) -> Atom {
        expression.replace_map(|view, _, output| {
            if chart_products
                .iter()
                .any(|product| product.as_view() == view)
            {
                // An identical output stops traversal into the Spenso scalar
                // product, whose dimension belongs to the recurrence.
                **output = view.to_owned();
            } else if let AtomView::Var(variable) = view {
                let symbol = variable.get_symbol();
                let Some(name) = symbol.get_name().strip_prefix("oneloopmaster::") else {
                    return;
                };
                let indexed = |prefix: &str| {
                    name.strip_prefix(prefix).is_some_and(|suffix| {
                        !suffix.is_empty() && suffix.bytes().all(|c| c.is_ascii_digit())
                    })
                };
                if !matches!(name, "d" | "k" | "xll" | "reg_delta")
                    && !indexed("q")
                    && !indexed("xq")
                    && !indexed("den")
                {
                    return;
                }
                let replacement = if let Some(replacement) = self.forward.get(&symbol) {
                    replacement.clone()
                } else {
                    let mut index = self.forward.len();
                    let placeholder = loop {
                        let candidate =
                            symbol!(format!("oneloopmaster::__input_{}_{}", self.nonce, index));
                        if self.used.insert(candidate) {
                            break candidate;
                        }
                        index += 1;
                    };
                    let replacement = Atom::var(placeholder);
                    self.forward.insert(symbol, replacement.clone());
                    self.backward.insert(placeholder, Atom::var(symbol));
                    replacement
                };
                **output = replacement;
            }
        })
    }

    fn restore(&self, expression: &Atom) -> Atom {
        expression.replace_map(|view, _, output| {
            if let AtomView::Var(variable) = view
                && let Some(original) = self.backward.get(&variable.get_symbol())
            {
                **output = original.clone();
            }
        })
    }
}

fn contains_loop(expression: &Atom, momentum: &Atom) -> bool {
    let mut found = false;
    expression.visitor(&mut |view| {
        found |= match (momentum.as_view(), view) {
            (AtomView::Var(p), _) => view.get_symbol() == Some(p.get_symbol()),
            (AtomView::Fun(p), AtomView::Fun(candidate)) => {
                p.get_symbol() == candidate.get_symbol()
                    && candidate.get_nargs() >= p.get_nargs()
                    && p.iter().zip(candidate.iter()).all(|(a, b)| a == b)
            }
            _ => false,
        };
        !found
    });
    found
}

fn monomial_powers(key: AtomView<'_>, variables: &[Atom]) -> Result<Vec<i32>, OneLoopError> {
    let mut powers = vec![0; variables.len()];
    let factors = match key {
        AtomView::Mul(mul) => mul.iter().collect::<Vec<_>>(),
        _ if key.is_one() => return Ok(powers),
        _ => vec![key],
    };
    for factor in factors {
        let (base, exponent) = match factor {
            AtomView::Pow(power) => {
                let (base, exponent) = power.get_base_exp();
                (
                    base,
                    checked_nonnegative_exponent(
                        exponent,
                        "reading a shared loop scalar-product exponent",
                    )?,
                )
            }
            _ => (factor, 1),
        };
        let index = variables
            .iter()
            .position(|variable| variable.as_view() == base)
            .ok_or_else(|| {
                invalid("numerator must be polynomial in the shared loop scalar products")
            })?;
        powers[index] = checked_exponent_add(
            powers[index],
            exponent,
            "combining shared loop scalar-product powers",
        )?;
    }
    Ok(powers)
}

fn validate_numerator(family: &SharedFamily, numerator: &Atom) -> Result<(), OneLoopError> {
    validate_polynomial_exponents(
        numerator,
        family.scalar_products(),
        "extracting the shared-family numerator",
    )?;
    for (key, coefficient) in numerator.coefficient_list::<i32>(family.scalar_products()) {
        monomial_powers(key.as_view(), family.scalar_products())?;
        if contains_loop(&coefficient, &family.loop_momenta()[0]) {
            return Err(invalid(
                "numerator contains unresolved loop momentum; contract tensors and use Kinematics.scalar_product",
            ));
        }
    }
    Ok(())
}

fn rank(expressions: &[Atom], variables: &[Atom]) -> Result<usize, OneLoopError> {
    if expressions.is_empty() || variables.is_empty() {
        return Ok(0);
    }
    Atom::system_to_matrix::<u16, _, _>(expressions, variables)
        .map(|(matrix, _)| matrix.rank())
        .map_err(invalid)
}

fn reduce_sector(
    family: &SharedFamily,
    powers: &[i32],
    numerator: &Atom,
    hygiene: &mut ChartSymbols,
    options: ReductionOptions,
) -> Result<Reduction, OneLoopError> {
    let kin = family.kinematics();
    let external = family.external_momenta();
    let loop_momentum = &family.loop_momenta()[0];
    let mut prefactor = Atom::num(1);
    let mut numerator = kin.apply(numerator);
    let mut shifts = Vec::new();
    let mut masses = Vec::new();
    let mut exponents = Vec::new();
    for (index, power) in powers.iter().copied().enumerate() {
        if power < 0 {
            // Widen before negating. Scalar-only coefficients are not restricted
            // by the loop-monomial exponent representation checked at ingress.
            let degree = -i64::from(power);
            numerator *= family.denominators()[index].pow(degree);
        } else if power > 0 {
            let Some(quadratic) = family.quadratic_denominator(index).map_err(invalid)? else {
                if contains_loop(&family.denominators()[index], loop_momentum) {
                    return Err(invalid(
                        "positive-power eikonal denominators are not supported by one-loop master reduction",
                    ));
                }
                prefactor *= family.denominators()[index].pow(-power);
                continue;
            };
            let shift = (quadratic.momentum() - loop_momentum).expand();
            for (_, coefficient) in shift.coefficient_list::<i32>(external) {
                if !coefficient.is_real().is_true() {
                    return Err(invalid(
                        "quadratic propagator shifts must have real coefficients",
                    ));
                }
            }
            prefactor *= quadratic.scale().pow(-power);
            shifts.push(shift);
            masses.push((-quadratic.remainder() / quadratic.scale()).cancel());
            exponents.push(power);
        }
    }
    validate_numerator(family, &numerator)?;
    if shifts.is_empty() {
        return Ok(Reduction { terms: Vec::new() });
    }
    let origin = shifts[0].clone();
    for shift in &mut shifts {
        *shift = (&*shift - &origin).expand();
    }
    let mut directions = shifts
        .windows(2)
        .map(|p| (&p[1] - &p[0]).expand())
        .collect::<Vec<_>>();
    // Extra numerator directions must retain their own Gram products. Add
    // zero-power auxiliary lines only to supply those coordinates; they never
    // introduce extra denominators into the represented integral.
    let mut current_rank = rank(&directions, external)?;
    for momentum in external {
        let mut trial = directions.clone();
        trial.push(momentum.clone());
        let next_rank = rank(&trial, external)?;
        if next_rank > current_rank {
            shifts.push((shifts.last().unwrap() + momentum).expand());
            directions.push(momentum.clone());
            masses.push(Atom::Zero);
            exponents.push(0);
            current_rank = next_rank;
        }
    }
    let backend_k = Atom::var(S.k);
    let backend_products = (1..=directions.len())
        .map(|i| {
            scalar_product(
                &backend_k,
                &Atom::var(symbol!(format!("oneloopmaster::q{i}"))),
            )
        })
        .collect::<Vec<_>>();
    let mut basis = Vec::new();
    let mut equations = Vec::new();
    for (index, direction) in directions.iter().enumerate() {
        let mut trial = basis.clone();
        trial.push(direction.clone());
        if rank(&trial, external)? > basis.len() {
            basis.push(direction.clone());
            equations.push(direction - &backend_products[index]);
        }
    }
    let images = if external.is_empty() {
        Vec::new()
    } else {
        let (matrix, rhs) =
            Atom::system_to_matrix::<u16, _, _>(&equations, external).map_err(invalid)?;
        matrix
            .solve(&rhs)
            .map_err(invalid)?
            .into_vec()
            .into_iter()
            .map(|value| value.to_expression().cancel())
            .collect::<Vec<_>>()
    };
    let coefficients = origin
        .coefficient_list::<i32>(external)
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let k_origin = external
        .iter()
        .zip(&images)
        .map(|(momentum, image)| coefficients.get(momentum).cloned().unwrap_or_default() * image)
        .sum::<Atom>();
    let mut rules = vec![
        scalar_product(&backend_k, &backend_k) - Atom::num(2) * k_origin
            + kin.scalar_product(&origin, &origin).map_err(invalid)?,
    ];
    for (momentum, image) in external.iter().zip(images) {
        rules.push(image - kin.scalar_product(&origin, momentum).map_err(invalid)?);
    }
    validate_polynomial_exponents(
        &numerator,
        family.scalar_products(),
        "transforming the shared-family numerator",
    )?;
    let mut transformed = Atom::Zero;
    for (key, coefficient) in numerator.coefficient_list::<i32>(family.scalar_products()) {
        let powers = monomial_powers(key.as_view(), family.scalar_products())?;
        let mut term = coefficient;
        for (rule, power) in rules.iter().zip(powers) {
            term *= rule.pow(power);
        }
        transformed += term;
    }
    let mut invariants = Vec::new();
    for (i, first) in shifts.iter().enumerate() {
        for second in &shifts[i + 1..] {
            let difference = first - second;
            invariants.push(
                kin.scalar_product(&difference, &difference)
                    .map_err(invalid)?,
            );
        }
    }
    // Preserve generated scalar products while shielding ordinary user scalars.
    let mut chart_products = backend_products;
    chart_products.push(scalar_product(&backend_k, &backend_k));
    let backend = RecurrenceInput {
        masses_squared: masses
            .iter()
            .map(|mass| hygiene.protect(mass, &[]))
            .collect(),
        invariants: invariants.iter().map(|s| hygiene.protect(s, &[])).collect(),
        powers: exponents,
        numerator: hygiene.protect(&transformed.expand(), &chart_products),
    };
    let prefactor = hygiene.protect(&prefactor, &[]);
    let mut result = options.reduce(&backend)?;
    for (coefficient, _) in &mut result.terms {
        *coefficient *= &prefactor;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MasterBasis, MasterIntegral, OneLoopMasters};
    use feynkit_kinematics::Kinematics;
    use symbolica::utils::resources::{ResourceBudget, ResourceLimitError};

    #[test]
    fn caller_resource_errors_preserve_their_type_and_healthy_reuse() {
        crate::ensure_symbolica_license();
        let dimension = Atom::var(symbol!("shared_resource_test::D"));
        let momentum = Atom::var(symbol!("shared_resource_test::ell"));
        let kin = Kinematics::in_dimension(&dimension)
            .unwrap()
            .with_momenta([momentum.clone()])
            .unwrap();
        let squared = kin.scalar_product(&momentum, &momentum).unwrap();
        let family =
            SharedFamily::new(vec![momentum], vec![], vec![squared - Atom::num(2)], &kin).unwrap();
        let budget = ResourceBudget::with_memory_monitor(
            ResourceLimits {
                max_process_rss_bytes: Some(1),
                ..ResourceLimits::default()
            },
            || Ok(2),
        );
        assert!(matches!(
            budget.run(|| reduce_family(&family, &[2], &Atom::one())),
            Err(OneLoopError::Resource(ResourceLimitError::Memory { .. }))
        ));
        let reduced = reduce_family(&family, &[2], &Atom::one()).unwrap();
        assert!(!reduced.terms.is_empty());
    }

    #[test]
    fn signed_power_storage_is_checked_before_expansion() {
        crate::ensure_symbolica_license();
        let dimension = Atom::var(symbol!("shared_reduction_test::D"));
        let momentum = Atom::var(symbol!("shared_reduction_test::ell"));
        let kin = Kinematics::in_dimension(&dimension)
            .unwrap()
            .with_momenta([momentum.clone()])
            .unwrap();
        let squared = kin.scalar_product(&momentum, &momentum).unwrap();
        let family =
            SharedFamily::new(vec![momentum], vec![], vec![squared - Atom::num(2)], &kin).unwrap();
        let error = reduce_family(&family, &[i32::MIN], &Atom::one()).unwrap_err();
        assert!(matches!(
            error,
            OneLoopError::UnsupportedNumerator { reason }
                if reason.contains("i32") && reason.contains("2147483648")
        ));
        for power in [-21, -22] {
            // Negative powers are polynomial insertions; without a remaining
            // propagator every resulting loop integral is scaleless.
            assert!(
                reduce_family(&family, &[power], &Atom::one())
                    .unwrap()
                    .terms
                    .is_empty()
            );
        }
        let diagnostic = ReductionOptions::default().with_diagnostics();
        for power in [33, i32::MAX] {
            assert!(
                diagnostic
                    .reduce_family(&family, &[power], &Atom::num(1))
                    .is_err()
            );
        }
        assert!(reduce_family(&family, &[33], &Atom::one()).is_ok());
    }

    #[test]
    fn shared_tadpole_accepts_exact_numerator_degrees_above_twenty() {
        crate::ensure_symbolica_license();
        let dimension = Atom::var(symbol!("shared_degree_test::D"));
        let momentum = Atom::var(symbol!("shared_degree_test::ell"));
        let kin = Kinematics::in_dimension(&dimension)
            .unwrap()
            .with_momenta([momentum.clone()])
            .unwrap();
        let squared = kin.scalar_product(&momentum, &momentum).unwrap();
        let family =
            SharedFamily::new(vec![momentum], vec![], vec![&squared - Atom::num(2)], &kin).unwrap();
        for degree in [21, 22] {
            let reduced = reduce_family(&family, &[1], &squared.pow(degree)).unwrap();
            assert_eq!(
                reduced.terms,
                vec![(
                    Atom::num(2).pow(degree),
                    MasterIntegral::Tadpole { m_sq: Atom::num(2) }
                )]
            );
        }
    }

    #[test]
    fn scalar_unit_denominator_accepts_the_minimum_signed_power() {
        crate::ensure_symbolica_license();
        let dimension = Atom::var(symbol!("scalar_power_test::D"));
        let momentum = Atom::var(symbol!("scalar_power_test::ell"));
        let kin = Kinematics::in_dimension(&dimension)
            .unwrap()
            .with_momenta([momentum.clone()])
            .unwrap();
        let squared = kin.scalar_product(&momentum, &momentum).unwrap();
        let family = SharedFamily::new(
            vec![momentum],
            vec![],
            vec![squared - Atom::num(2), Atom::one()],
            &kin,
        )
        .unwrap();
        // The unit factor has no loop-coordinate exponent and does not create
        // a large integer value or a large polynomial when its power is moved.
        let reduced = reduce_family(&family, &[1, i32::MIN], &Atom::one()).unwrap();
        assert_eq!(
            reduced.terms,
            vec![(Atom::one(), MasterIntegral::Tadpole { m_sq: Atom::num(2) })]
        );
    }

    #[test]
    fn unrepresentable_loop_exponent_is_rejected_before_polynomial_extraction() {
        crate::ensure_symbolica_license();
        let dimension = Atom::var(symbol!("shared_exponent_test::D"));
        let momentum = Atom::var(symbol!("shared_exponent_test::ell"));
        let kin = Kinematics::in_dimension(&dimension)
            .unwrap()
            .with_momenta([momentum.clone()])
            .unwrap();
        let squared = kin.scalar_product(&momentum, &momentum).unwrap();
        let family =
            SharedFamily::new(vec![momentum], vec![], vec![&squared - Atom::num(2)], &kin).unwrap();
        // A compact literal tests the actual boundary without differentiating
        // or expanding billions of terms.
        let numerator = squared.pow(i64::from(i32::MAX) + 1);
        let error = reduce_family(&family, &[1], &numerator).unwrap_err();
        assert!(matches!(
            error,
            OneLoopError::UnsupportedNumerator { reason }
                if reason.contains("i32") && reason.contains("2147483648")
        ));
    }

    #[test]
    fn user_parameters_do_not_capture_recurrence_coordinates_or_dimension() {
        crate::ensure_symbolica_license();
        let dimension = Atom::var(symbol!("shared_reduction_test::D"));
        let momentum = Atom::var(symbol!("shared_reduction_test::ell"));
        let kin = Kinematics::in_dimension(&dimension)
            .unwrap()
            .with_momenta([momentum.clone()])
            .unwrap();
        let squared = kin.scalar_product(&momentum, &momentum).unwrap();
        for name in ["d", "xll", "xq1", "den1", "reg_delta"] {
            let parameter = Atom::var(symbol!(format!("oneloopmaster::{name}")));
            let family = SharedFamily::new(
                vec![momentum.clone()],
                vec![],
                vec![&squared - Atom::num(2)],
                &kin,
            )
            .unwrap();
            let reduced = reduce_family(&family, &[2], &parameter).unwrap();
            let expression = reduced
                .terms
                .into_iter()
                .map(|(coefficient, master)| coefficient * OneLoopMasters.symbol(&master))
                .sum::<Atom>();
            let expected = &parameter * (&dimension - Atom::num(2)) / Atom::num(4)
                * OneLoopMasters.symbol(&MasterIntegral::Tadpole { m_sq: Atom::num(2) });
            assert!(
                (expression - expected).cancel().is_zero(),
                "scalar parameter {name}"
            );

            // Restoration also traverses master arguments, including masses.
            let family = SharedFamily::new(
                vec![momentum.clone()],
                vec![],
                vec![&squared - &parameter],
                &kin,
            )
            .unwrap();
            let reduced = reduce_family(&family, &[1], &Atom::num(1)).unwrap();
            assert_eq!(
                reduced.terms,
                vec![(Atom::num(1), MasterIntegral::Tadpole { m_sq: parameter })]
            );
        }
    }
}
