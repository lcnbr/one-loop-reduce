//! One-loop IBP reduction of Feynman integrals to the four scalar master integrals.

#[doc(hidden)]
pub mod amplitude;
#[doc(hidden)]
pub mod bridge;
pub mod error;
pub mod masters;
#[doc(hidden)]
pub mod recurrence;
#[doc(hidden)]
pub mod reduce;
#[doc(hidden)]
pub mod routing;
pub mod shared_family;
pub mod symbols;
pub use shared_family::reduce_family;

pub use error::OneLoopError;
pub use masters::{MasterBasis, MasterIntegral, OneLoopMasters};
pub use recurrence::ReductionOptions;
pub use reduce::{MAX_TOTAL_INDEX, Reduction};

/// Activate the Symbolica license once per process, from `SYMBOLICA_LICENSE`.
///
/// Symbolica allows a single unlicensed instance per process and aborts when it
/// is touched from a second thread, so every Symbolica-using test calls this
/// first and the suite is run with `--test-threads=1`. With no key in the
/// environment the call is a no-op and Symbolica runs restricted, which is
/// enough for the test suite.
#[cfg(test)]
pub(crate) fn ensure_symbolica_license() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if let Ok(key) = std::env::var("SYMBOLICA_LICENSE") {
            let _ = symbolica::prelude::LicenseManager::set_license_key(&key);
        }
    });
}

static CITATIONS_USED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[inline]
pub(crate) fn record_usage() {
    use std::sync::atomic::Ordering;
    if !CITATIONS_USED.load(Ordering::Relaxed) {
        CITATIONS_USED.store(true, Ordering::Relaxed);
    }
}

/// Whether the reducer has run in this process; gates its citations.
pub fn was_used() -> bool {
    CITATIONS_USED.load(std::sync::atomic::Ordering::Relaxed)
}
