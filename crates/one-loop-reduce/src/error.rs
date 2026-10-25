use symbolica::utils::resources::ResourceLimitError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OneLoopError {
    /// A shared caller resource budget stopped the request.
    #[error("{0}")]
    Resource(#[from] ResourceLimitError),

    /// The input graph is not one loop.
    #[error("unsupported loop order: graph has {found} loops, only one-loop is supported")]
    UnsupportedLoopOrder { found: usize },

    /// A host-side graph (e.g. a gammaloop `Graph`, handed over through
    /// [`crate::bridge`]) could not be turned into recurrence coordinates.
    #[error("failed to extract integral family from graph: {reason}")]
    ExtractionFailed { reason: String },

    /// An explicitly enabled total-index diagnostic stopped the request.
    #[error(
        "propagator-index diagnostic for {found:?}: the positive powers may total at most {max}"
    )]
    UnsupportedIndex { found: Vec<i32>, max: i64 },

    /// A symbolic input uses the superseded reducer-only namespace.
    #[error(
        "obsolete reducer symbol {name}: use shared HEP families and \
         Kinematics.scalar_product; scalar master symbols belong to oneloopmaster"
    )]
    ObsoleteSymbol { name: String },

    /// A family the reducer cannot take: wrong shape, powers or kinematics.
    #[error("invalid one-loop integral family: {0}")]
    InvalidFamily(String),

    /// The numerator is not a polynomial in the loop scalar products the family supports.
    #[error("unsupported numerator: {reason}")]
    UnsupportedNumerator { reason: String },

    /// The reduction produced an indeterminate or infinite value.
    #[error("the reduction is not finite: {reason}")]
    NonFiniteResult { reason: String },

    /// Wraps an underlying Symbolica error surfaced during extraction.
    #[error("symbolica error: {0}")]
    Symbolica(String),
}
