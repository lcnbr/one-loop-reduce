//! The recurrences' input in their own coordinates, built by `shared_family`
//! from a FeynKit family (the benchmarks build it directly).

use symbolica::atom::Atom;

/// Optional diagnostic stops for the existing one-loop recurrence.
/// RAM and elapsed time are controlled by the enclosing Symbolica resource scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReductionOptions {
    pub diagnostic_max_total_index: Option<i64>,
}

impl ReductionOptions {
    pub fn with_diagnostics(mut self) -> Self {
        self.diagnostic_max_total_index = Some(i64::from(crate::reduce::MAX_TOTAL_INDEX));
        self
    }
}

#[derive(Debug, Clone)]
pub struct RecurrenceInput {
    pub masses_squared: Vec<Atom>,
    /// Pairwise invariants `(r_i-r_j)^2` in lexicographic order.
    pub invariants: Vec<Atom>,
    /// Nonnegative powers of the normalized quadratic lines.
    pub powers: Vec<i32>,
    /// Scalar polynomial expressed in the recurrence coordinates.
    pub numerator: Atom,
}
