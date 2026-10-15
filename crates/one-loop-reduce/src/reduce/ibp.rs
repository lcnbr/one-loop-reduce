//! Exact reduction by Gaussian elimination of the IBP identities (Laporta),
//! at fixed kinematics with `d` symbolic. Far slower than the recursions, so
//! it is only the fallback where their on-shell limit does not exist: it needs
//! no regulator, selects an independent scalar basis at a degenerate point,
//! and sends integrals without a scale to zero.

use std::collections::{BTreeMap, HashMap};

use symbolica::atom::{Atom, AtomCore};
use symbolica::domains::integer::Z;
use symbolica::domains::rational::Q;

use super::{Rp, add_scaled, den_symbol, extract_monomials, loop_dots_to_vars};
use super::{modified_cayley, reduce_cayley};
use crate::masters::MasterIntegral;
use crate::recurrence::RecurrenceInput;
use crate::symbols::S;

/// An integral `I(a) = ∫ 1/(D_1^a_1 ... D_N^a_N)`; `a_i <= 0` is a numerator.
type Index = Vec<i32>;

/// Eliminate dotted/numerator integrals first. Among unit-power scalar
/// integrals retain the higher-point master: eliminating an IR triangle into
/// bubbles can introduce a 1/(d-4) absent from the original representation.
/// This preference reduces avoidable poles, without promising an epsilon-finite
/// basis for every family or arbitrary dimension-dependent numerator.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Complexity {
    Scalar(std::cmp::Reverse<usize>),
    NonScalar(usize),
}
type Key = (Complexity, i32, i32, Index);

fn key(a: &[i32]) -> Key {
    let lines = a.iter().filter(|&&e| e > 0).count();
    let dots = a.iter().filter(|&&e| e > 0).map(|&e| e - 1).sum();
    let numerator = a.iter().filter(|&&e| e < 0).map(|&e| -e).sum();
    let kind = if dots == 0 && numerator == 0 {
        Complexity::Scalar(std::cmp::Reverse(lines))
    } else {
        Complexity::NonScalar(lines)
    };
    (kind, dots, numerator, a.to_vec())
}

fn rp(a: &Atom) -> Rp {
    a.to_rational_polynomial(&Q, &Z, None)
}

/// A linear combination of integrals.
type Row = BTreeMap<Key, Rp>;

fn add_to(row: &mut Row, a: Index, c: Rp) {
    // Without a positive index there is no scale: the integral is zero.
    if c.is_zero() || a.iter().all(|&e| e <= 0) {
        return;
    }
    let k = key(&a);
    let sum = match row.remove(&k) {
        Some(old) => &old + &c,
        None => c,
    };
    if !sum.is_zero() {
        row.insert(k, sum);
    }
}

/// The IBP identity from `∂/∂l · (l + r_k)` on `I(a)`:
/// `(d - Σa) I(a) - Σ_i a_i I(a + e_i - e_k) - Σ_i a_i Y_ki I(a + e_i) = 0`.
fn identity(a: &[i32], k: usize, y: &[Vec<Rp>]) -> Row {
    let total: i32 = a.iter().sum();
    let mut row = Row::new();
    add_to(
        &mut row,
        a.to_vec(),
        rp(&(Atom::var(S.d) - Atom::num(i64::from(total)))),
    );
    for (i, &ai) in a.iter().enumerate().filter(|&(_, &ai)| ai != 0) {
        let ai = rp(&Atom::num(i64::from(ai)));
        let mut shifted = a.to_vec();
        shifted[i] += 1;
        add_to(&mut row, shifted.clone(), -(&ai * &y[k][i]));
        shifted[k] -= 1;
        add_to(&mut row, shifted, -ai);
    }
    row
}

/// Seeds: every sector, with up to `dots` extra powers and `num` numerator powers.
fn seeds(n: usize, dots: i32, num: i32) -> Vec<Index> {
    fn spread(slots: &[usize], total: i32, sign: i32, base: &mut Index, out: &mut Vec<Index>) {
        let Some((&first, rest)) = slots.split_first() else {
            if total == 0 {
                out.push(base.clone());
            }
            return;
        };
        for x in 0..=total {
            base[first] += sign * x;
            spread(rest, total - x, sign, base, out);
            base[first] -= sign * x;
        }
    }
    let mut out = Vec::new();
    for mask in 1..(1u32 << n) {
        let on = |i: usize| (mask & (1 << i)) != 0;
        let lines: Vec<usize> = (0..n).filter(|&i| on(i)).collect();
        let others: Vec<usize> = (0..n).filter(|&i| !on(i)).collect();
        for r in 0..=dots {
            for s in 0..=num {
                let mut base: Index = (0..n).map(|i| i32::from(on(i))).collect();
                let mut with_dots = Vec::new();
                spread(&lines, r, 1, &mut base, &mut with_dots);
                for mut b in with_dots {
                    spread(&others, s, -1, &mut b, &mut out);
                }
            }
        }
    }
    out.sort_by_key(|a| key(a));
    out
}

/// Substitute solved integrals until only unsolved ones remain.
fn reduce_row(mut row: Row, rules: &HashMap<Key, Row>) -> Row {
    while let Some(k) = row.keys().rev().find(|k| rules.contains_key(*k)).cloned() {
        let c = row.remove(&k).unwrap();
        for (j, cj) in &rules[&k] {
            add_to(&mut row, j.3.clone(), &c * cj);
        }
    }
    row
}

/// Turn every seed's identities into rules `integral -> simpler integrals`.
fn eliminate(n: usize, y: &[Vec<Rp>], dots: i32, num: i32) -> HashMap<Key, Row> {
    let mut rules = HashMap::new();
    for a in seeds(n, dots, num) {
        for k in 0..n {
            let mut row = reduce_row(identity(&a, k, y), &rules);
            if let Some((pivot, c)) = row.pop_last() {
                let rhs = row.into_iter().map(|(j, cj)| (j, -(&cj / &c))).collect();
                rules.insert(pivot, rhs);
            }
        }
    }
    rules
}

/// Reduce `family` exactly, or `None` for a tadpole (its numerator directions
/// are not inverse propagators) or if the elimination does not close.
pub(super) fn reduce_exact(family: &RecurrenceInput) -> Option<Vec<(Atom, MasterIntegral)>> {
    let n = family.masses_squared.len();
    if n < 2 {
        return None;
    }
    let masses: Vec<Atom> = family.masses_squared.clone();
    let invariants = &family.invariants;
    let y_atoms = modified_cayley(&masses, invariants);
    let y: Vec<Vec<Rp>> = y_atoms
        .iter()
        .map(|row| row.iter().map(rp).collect())
        .collect();

    // The numerator in inverse propagators, along the chain r_1 = 0,
    // r_i = q_1 + ... + q_{i-1}: k.k = D_1 + m_1^2 and
    // 2 k.r_i = D_i - D_1 - r_i^2 + m_i^2 - m_1^2, with r_i^2 = (r_1 - r_i)^2.
    let den: Vec<Atom> = (0..n).map(|i| Atom::var(den_symbol(i))).collect();
    let k_dot_r = |i: usize| match i {
        0 => Atom::Zero,
        _ => (&den[i] - &den[0] - &invariants[i - 1] + &masses[i] - &masses[0]) / Atom::num(2),
    };
    let (poly, vars) = loop_dots_to_vars(&family.numerator, n - 1);
    let mut numerator = poly
        .replace(Atom::var(vars[0]).to_pattern())
        .with(&den[0] + &masses[0]);
    for (j, xq) in vars.iter().enumerate().skip(1) {
        let k_dot_q = k_dot_r(j) - k_dot_r(j - 1);
        numerator = numerator.replace(Atom::var(*xq).to_pattern()).with(k_dot_q);
    }
    let den_syms: Vec<_> = (0..n).map(den_symbol).collect();
    let terms = extract_monomials(&numerator.expand(), &den_syms);
    let exps = &family.powers;
    let integrals: Vec<Index> = terms
        .iter()
        .map(|(powers, _)| exps.iter().zip(powers).map(|(a, p)| a - p).collect())
        .collect();

    // Reduce each integral to masters, widening the seeds if one does not close.
    let dots = integrals.iter().map(|a| key(a).1).max().unwrap_or(0);
    let num = integrals.iter().map(|a| key(a).2).max().unwrap_or(0);
    let reduced = (0..=2).find_map(|extra| {
        let rules = eliminate(n, &y, dots + extra, num + extra);
        integrals
            .iter()
            .map(|a| {
                let mut row = Row::new();
                add_to(&mut row, a.clone(), rp(&Atom::num(1)));
                let row = reduce_row(row, &rules);
                row.keys().all(|k| k.1 == 0 && k.2 == 0).then_some(row)
            })
            .collect::<Option<Vec<Row>>>()
    })?;

    // A master is a sector with every line to the power 1; `reduce_cayley`
    // names it (and takes a sector of five or more lines down to boxes).
    let mut out = Vec::new();
    for ((_, coefficient), row) in terms.iter().zip(reduced) {
        for (master, c) in row {
            let lines: Vec<usize> = (0..n).filter(|&i| master.3[i] > 0).collect();
            let sub = |m: &[Vec<Atom>]| -> Vec<Vec<Atom>> {
                lines
                    .iter()
                    .map(|&i| lines.iter().map(|&j| m[i][j].clone()).collect())
                    .collect()
            };
            let sub_masses: Vec<Atom> = lines.iter().map(|&i| masses[i].clone()).collect();
            let sub_y = sub(&y_atoms);
            // Reordering may leave a unit-power scaleless sub-sector among
            // the symbolic residuals. It is zero in dimensional regularization.
            if sub_masses.iter().all(Atom::is_zero) && sub_y.iter().flatten().all(Atom::is_zero) {
                continue;
            }
            let masters = reduce_cayley(&sub_y, &sub_masses, &vec![1; lines.len()]);
            add_scaled(&mut out, &(coefficient * c.to_expression()), masters);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::reduce_exact;
    use crate::masters::MasterIntegral;
    use crate::reduce::limit_tests::{family, n};
    use crate::reduce::reduce;
    use crate::symbols::S;
    use symbolica::atom::{Atom, AtomCore};
    use symbolica::symbol;

    /// Compare two reductions, writing `B0(0,m,m) = (d-2)/(2m^2) A0(m)` first:
    /// the exact solver sees that a zero-momentum equal-mass bubble is a
    /// tadpole, the recursion keeps it as a bubble.
    fn assert_same(what: &str, a: &[(Atom, MasterIntegral)], b: &[(Atom, MasterIntegral)]) {
        let normal = |t: &[(Atom, MasterIntegral)]| -> Vec<(Atom, MasterIntegral)> {
            let mut out = Vec::new();
            for (c, m) in t {
                let term = match m {
                    MasterIntegral::Bubble { p_sq, m1_sq, m2_sq }
                        if p_sq.is_zero() && m1_sq == m2_sq && !m1_sq.is_zero() =>
                    {
                        let factor = (Atom::var(S.d) - n(2)) / (n(2) * m1_sq);
                        (
                            c * factor,
                            MasterIntegral::Tadpole {
                                m_sq: m1_sq.clone(),
                            },
                        )
                    }
                    _ => (c.clone(), m.clone()),
                };
                crate::reduce::add_scaled(&mut out, &n(1), vec![term]);
            }
            out.retain(|(c, _)| !c.together().expand().is_zero());
            out
        };
        let (a, b) = (normal(a), normal(b));
        assert_eq!(a.len(), b.len(), "{what}: {a:?}\nvs {b:?}");
        for (c, m) in &a {
            let (c2, _) = b
                .iter()
                .find(|(_, m2)| m2 == m)
                .unwrap_or_else(|| panic!("{what}: {m:?}"));
            assert!(
                (c - c2).together().expand().is_zero(),
                "{what}, {m:?}: {c} vs {c2}"
            );
        }
    }

    fn dot(a: &Atom, b: &Atom) -> Atom {
        crate::symbols::scalar_product(a, b)
    }

    fn q(i: usize) -> Atom {
        Atom::var(symbol!(format!("oneloopmaster::q{i}")))
    }

    /// Away from degenerate points both methods are exact, so they must agree.
    #[test]
    fn matches_the_recursion_at_generic_kinematics() {
        crate::ensure_symbolica_license();
        let k = Atom::var(S.k);
        let box_inv = || vec![n(-2), n(-7), n(-3), n(-4), n(-5), n(-6)];
        let cases = [
            family(vec![n(1), n(2)], vec![n(-3)], vec![2, 1], n(1)),
            family(vec![n(0), n(0)], vec![n(-5)], vec![3, 2], n(1)),
            family(
                vec![n(1), n(2), n(3)],
                vec![n(-1), n(-2), n(-3)],
                vec![2, 1, 1],
                n(1),
            ),
            family(
                vec![n(1); 3],
                vec![n(-1), n(-2), n(-3)],
                vec![1; 3],
                dot(&k, &q(1)) * dot(&k, &q(2)),
            ),
            family(vec![n(0); 4], box_inv(), vec![2, 1, 1, 1], n(1)),
            family(
                vec![n(1); 4],
                box_inv(),
                vec![1; 4],
                dot(&k, &q(1)) * dot(&k, &k),
            ),
            family(vec![n(1); 4], box_inv(), vec![1, 2, 1, 1], dot(&k, &q(3))),
        ];
        for (i, fam) in cases.iter().enumerate() {
            let exact = reduce_exact(fam).unwrap();
            assert_same(&format!("case {i}"), &exact, &reduce(fam).unwrap().terms);
        }
    }

    /// With two on-shell legs a massless triangle and a bubble are dependent:
    /// `C0(0,0,s) = -2(d-3)/((d-4) s) B0(s,0,0)`, from the ratio of their
    /// Gamma-function closed forms.
    #[test]
    fn the_on_shell_massless_triangle_is_retained_without_a_coefficient_pole() {
        crate::ensure_symbolica_license();
        let s = Atom::var(S.psq);
        let d = Atom::var(S.d);
        let fam = family(vec![n(0); 3], vec![n(0), s.clone(), n(0)], vec![1; 3], n(1));
        let want = -(n(2) * (&d - n(3))) / ((&d - n(4)) * &s);
        let bubble = MasterIntegral::Bubble {
            p_sq: s,
            m1_sq: Atom::Zero,
            m2_sq: Atom::Zero,
        };
        let exact = reduce_exact(&fam).unwrap();
        assert_eq!(exact.len(), 1);
        assert!(matches!(exact[0].1, MasterIntegral::Triangle { .. }));
        assert!((exact[0].0.clone() - n(1)).cancel().is_zero());
        // Its pinched bubble is now C0/want, whose coefficient vanishes at d=4.
        let mut pinched = fam.clone();
        pinched.powers = vec![1, 0, 1];
        let reduced_bubble = reduce_exact(&pinched).unwrap();
        assert_eq!(reduced_bubble.len(), 1, "{reduced_bubble:?}");
        assert_eq!(reduced_bubble[0].1, exact[0].1);
        assert!((&reduced_bubble[0].0 * &want - n(1)).cancel().is_zero());
        assert!(
            reduced_bubble[0]
                .0
                .replace(d.to_pattern())
                .with(n(4))
                .is_zero()
        );
        // The ordinary recurrence remains a valid independent representation.
        assert_same(
            "pinched bubble",
            &reduce(&pinched).unwrap().terms,
            &[(n(1), bubble)],
        );
    }

    /// Where the shared-delta limit exists, the exact reduction at the on-shell
    /// point agrees with it -- an independent check of that limit.
    #[test]
    fn matches_the_on_shell_limit_where_it_exists() {
        crate::ensure_symbolica_license();
        let k = Atom::var(S.k);
        let two_legs = || vec![n(0), n(-7), n(-3), n(0), n(-5), n(-2)];
        let four_legs = || vec![n(0), n(-7), n(0), n(0), n(-5), n(0)];
        let cases = [
            family(vec![n(1); 4], two_legs(), vec![2, 1, 1, 1], n(1)),
            family(vec![n(1); 4], four_legs(), vec![1, 2, 1, 1], n(1)),
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
        ];
        for (i, fam) in cases.iter().enumerate() {
            assert_same(
                &format!("case {i}"),
                &reduce_exact(fam).unwrap(),
                &reduce(fam).unwrap().terms,
            );
        }
    }

    /// The two configurations the shared delta cannot do: `reduce` falls back
    /// to the exact solver and returns a finite result.
    #[test]
    fn reduces_where_the_on_shell_limit_does_not_exist() {
        crate::ensure_symbolica_license();
        let k = Atom::var(S.k);
        let cases = [
            family(
                vec![n(0); 4],
                vec![n(0), n(-7), n(-3), n(0), n(-5), n(-2)],
                vec![2, 1, 1, 1],
                n(1),
            ),
            family(
                vec![n(1); 3],
                vec![n(0), n(-3), n(0)],
                vec![2, 2, 1],
                dot(&k, &q(2)),
            ),
        ];
        for (i, fam) in cases.iter().enumerate() {
            let r = reduce(fam).unwrap_or_else(|e| panic!("case {i}: {e}"));
            assert_same(&format!("case {i}"), &r.terms, &reduce_exact(fam).unwrap());
            assert!(!r.terms.is_empty(), "case {i}");
        }
    }
}
