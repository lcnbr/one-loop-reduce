//! One-loop reduction of the existing shared HEP integral family.
//!
//! The input is Feynkit's `PyIntegralFamily`; scalar products and kinematics
//! retain their existing Symbolica objects. Only reduction results and scalar
//! master descriptors are registered by this module.

use std::panic;

use feynkit_py::PyIntegralFamily;
use oneloopreduce::masters::MasterIntegral as RsMasterIntegral;
pub use oneloopreduce::masters::OneLoopMasters;
use oneloopreduce::reduce::Reduction as RsReduction;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, Python, pyclass, pyfunction, pymethods};
use symbolica::api::python::{Citation, PythonExpression, SymbolicaCommunityModule};
use symbolica::atom::{Atom, AtomCore, Symbol};

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{
    define_stub_info_gatherer,
    derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods},
};

pub struct CommunityModule;

impl SymbolicaCommunityModule for CommunityModule {
    fn get_citations() -> Vec<Citation> {
        if !oneloopreduce::was_used() {
            return Vec::new();
        }
        let mut citations = vec![Citation {
            id: "https://github.com/ecavan/one-loop-reduce".into(),
            reference: "Elijah Cavan. one-loop-reduce (2026).".into(),
            bibtex: r#"@software{one_loop_reduce,
  author = {Cavan, Elijah},
  title = {{one-loop-reduce}},
  year = {2026},
  url = {https://github.com/ecavan/one-loop-reduce}
}"#
            .into(),
            reasons: vec!["Symbolic one-loop reduction to scalar master integrals.".into()],
            description: String::new(),
            relevance: None,
        }];
        // The methods the reducer implements: (key, DOI, authors, title, journal,
        // volume, pages, year, what it is used for).
        #[rustfmt::skip]
        let papers = [
            ("Passarino:1978jh", "10.1016/0550-3213(79)90234-7", "Passarino, G. and Veltman, M.",
             "One-loop corrections for $e^+e^-$ annihilation into $\\mu^+\\mu^-$ in the Weinberg model",
             "Nucl. Phys. B", "160", "151--207", "1979", "Tensor reduction."),
            ("vanNeerven:1983vr", "10.1016/0370-2693(84)90237-5", "van Neerven, W. L. and Vermaseren, J. A. M.",
             "Large loop integrals", "Phys. Lett. B", "137", "241--244", "1984",
             "Reduction of five or more propagators."),
            ("Fleischer:1999hq", "10.1016/S0550-3213(99)00678-1", "Fleischer, J. and Jegerlehner, F. and Tarasov, O. V.",
             "Algebraic reduction of one-loop Feynman graph amplitudes", "Nucl. Phys. B", "566", "423--440", "2000",
             "Lowering raised propagator powers."),
            ("Tarasov:1996br", "10.1103/PhysRevD.54.6479", "Tarasov, O. V.",
             "Connection between Feynman integrals having different values of the space-time dimension",
             "Phys. Rev. D", "54", "6479--6490", "1996", "Dimension-shift relations."),
            ("Chetyrkin:1981qh", "10.1016/0550-3213(81)90199-1", "Chetyrkin, K. G. and Tkachov, F. V.",
             "Integration by parts: The algorithm to calculate $\\beta$-functions in 4 loops",
             "Nucl. Phys. B", "192", "159--204", "1981", "Integration by parts."),
            ("Laporta:2000dsw", "10.1142/S0217751X00002159", "Laporta, S.",
             "High-precision calculation of multiloop Feynman integrals by difference equations",
             "Int. J. Mod. Phys. A", "15", "5087--5159", "2000", "Exact reduction at degenerate kinematics."),
        ];
        citations.extend(papers.map(
            |(key, doi, authors, title, journal, volume, pages, year, reason)| Citation {
                id: doi.into(),
                reference: format!("{authors}, {journal} {volume} ({year}) {pages}."),
                bibtex: format!(
                    "@article{{{key},\n  author = {{{authors}}},\n  title = {{{title}}},\n  \
                     journal = {{{journal}}},\n  volume = {{{volume}}},\n  pages = {{{pages}}},\n  \
                     year = {{{year}}},\n  doi = {{{doi}}}\n}}"
                ),
                reasons: vec![reason.into()],
                description: String::new(),
                relevance: None,
            },
        ));
        citations
    }

    fn get_name() -> String {
        "oneloopreduce".to_owned()
    }

    fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Reduction>()?;
        m.add_class::<MasterIntegral>()?;
        m.add_function(pyo3::wrap_pyfunction!(reduce, m)?)?;
        Ok(())
    }

    fn initialize(_py: Python) -> PyResult<()> {
        Ok(())
    }
}

/// Reduce a one-loop ``hep.IntegralFamily`` to scalar master integrals.
///
/// ``powers`` follows the family denominator order. Negative powers contribute
/// numerator factors and zero powers omit denominators. ``numerator`` is an
/// additional scalar expression written using ``family.kinematics.scalar_product``.
/// Exactly one loop and a symbolic dimension are required. Positive powers of
/// eikonal denominators and uncontracted loop tensors raise ``ValueError``.
///
/// Examples
/// --------
/// >>> from symbolica import S, E
/// >>> from symbolica.community import hepkit as hep
/// >>> from symbolica.community.hepkit import oneloop
/// >>> d, k, p, s = S("d", "k", "p", "s")
/// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
/// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
/// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
/// >>> reduction = oneloop.reduce(family, [1, 1])
/// >>> assert reduction.to_expression() == oneloop.B0(s, 0, 0, 1)
///
/// Parameters
/// ----------
/// family : IntegralFamily
///     One-loop denominator family with symbolic dimension.
/// powers : sequence[int]
///     One signed integer per denominator.
/// numerator : Expression or None, optional
///     Additional scalar numerator; None uses one.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.community.hepkit.oneloop")
)]
#[pyfunction]
#[pyo3(signature = (family, powers, *, numerator = None))]
pub fn reduce(
    family: &PyIntegralFamily,
    powers: Vec<i32>,
    numerator: Option<PythonExpression>,
) -> PyResult<Reduction> {
    let numerator = numerator
        .map(|value| value.expr)
        .unwrap_or_else(|| Atom::num(1));
    let shared = family.as_family();
    let reduced = catch_panic(|| oneloopreduce::reduce_family(shared, &powers, &numerator))
        .map_err(reduction_failed)?
        .map_err(|error| reduction_failed(error.to_string()))?;
    let dimension = shared
        .kinematics()
        .dimension()
        .to_symbolic()
        .get_symbol()
        .expect("reduce_family validated a symbolic dimension");
    Ok(Reduction {
        terms: reduced.terms,
        dimension,
    })
}

/// Run `f`, turning a panic into a message rather than unwinding across the FFI
/// boundary. Only this thread's panic output is silenced while it runs.
fn catch_panic<R>(f: impl FnOnce() -> R) -> Result<R, String> {
    use std::cell::Cell;
    use std::sync::Once;

    thread_local! {
        static SILENT: Cell<bool> = const { Cell::new(false) };
    }
    static INSTALL: Once = Once::new();

    INSTALL.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if !SILENT.with(|s| s.get()) {
                previous(info);
            }
        }));
    });

    SILENT.with(|s| s.set(true));
    let out = panic::catch_unwind(panic::AssertUnwindSafe(f));
    SILENT.with(|s| s.set(false));

    out.map_err(|payload| {
        payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "the reducer panicked".to_string())
    })
}

fn reduction_failed(message: String) -> pyo3::PyErr {
    PyValueError::new_err(format!("one-loop reduction failed: {message}"))
}

/// Check a symbolic input before returning one-loop expressions to Python.
pub fn validate_namespace(input: &Atom) -> PyResult<()> {
    oneloopreduce::symbols::validate_namespace(input)
        .map_err(|error| reduction_failed(error.to_string()))
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

/// A symbolic linear combination of scalar one-loop master integrals.
///
/// Created by ``oneloop.reduce``; there is no direct constructor. ``terms``
/// contains (coefficient, MasterIntegral) pairs with exact dependence on the
/// family's symbolic dimension. ``to_expression`` assembles a symbolic sum;
/// ``oneloop.reduction_coefficients`` expands it about d=4-2*eps for evaluation.
///
/// Examples
/// --------
/// >>> from symbolica import S, E
/// >>> from symbolica.community import hepkit as hep
/// >>> from symbolica.community.hepkit import oneloop
/// >>> d, k, p, s = S("d", "k", "p", "s")
/// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
/// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
/// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
/// >>> reduction = oneloop.reduce(family, [1, 1])
/// >>> assert len(reduction) == 1
/// >>> coefficient, master = reduction.terms[0]
/// >>> assert master.kind == "bubble"
/// >>> expression = reduction.to_expression()
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    name = "Reduction",
    module = "symbolica.community.hepkit.oneloop"
)]
#[derive(Clone)]
pub struct Reduction {
    terms: Vec<(Atom, RsMasterIntegral)>,
    dimension: Symbol,
}

impl Reduction {
    /// The shared family dimension to expand when forming Laurent coefficients.
    pub fn dimension_symbol(&self) -> Symbol {
        self.dimension
    }

    /// Borrow the exact coefficients and masters for a shared-kernel evaluator.
    pub fn terms_ref(&self) -> &[(Atom, RsMasterIntegral)] {
        &self.terms
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl Reduction {
    /// Symbolic dimension used in the unreduced family and its exact coefficients.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> assert reduction.dimension == d
    #[getter]
    fn dimension(&self) -> PythonExpression {
        Atom::var(self.dimension).into()
    }

    /// Linear-combination terms as (coefficient, MasterIntegral) pairs.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> bubbles = [(c, m) for c, m in reduction.terms if m.kind == "bubble"]
    /// >>> assert len(bubbles) == 1
    #[getter]
    fn terms(&self) -> Vec<(PythonExpression, MasterIntegral)> {
        self.terms
            .iter()
            .map(|(coefficient, master)| {
                (
                    coefficient.clone().into(),
                    MasterIntegral {
                        inner: master.clone(),
                    },
                )
            })
            .collect()
    }

    /// Assemble primitive scalar-master calls with the squared scale last.
    ///
    /// The default scale is exactly one. This builds a symbolic expression; use
    /// ``master_coefficients`` or ``reduction_coefficients`` to obtain Laurent
    /// coefficients with native evaluation hooks.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> mu2 = S("mu2")
    /// >>> expression = reduction.to_expression(mu2)
    /// >>> assert expression == oneloop.B0(s, 0, 0, mu2)
    ///
    /// Parameters
    /// ----------
    /// mu_squared : Expression or None, optional
    ///     Squared renormalization scale; None uses one.
    #[pyo3(signature = (mu_squared = None))]
    fn to_expression(&self, mu_squared: Option<PythonExpression>) -> PyResult<PythonExpression> {
        let mu_squared = mu_squared.map(|value| value.expr).unwrap_or(Atom::num(1));
        validate_namespace(&mu_squared)?;
        catch_panic(|| {
            self.terms
                .iter()
                .fold(Atom::Zero, |sum, (coefficient, master)| {
                    sum + coefficient * OneLoopMasters.symbol_with_scale(master, &mu_squared)
                })
        })
        .map(Into::into)
        .map_err(reduction_failed)
    }

    /// Cancel each rational coefficient to lowest terms and return a new reduction.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> simplified = reduction.simplify()
    /// >>> assert (simplified.to_expression() - reduction.to_expression()).together() == E("0")
    fn simplify(&self) -> PyResult<Reduction> {
        let terms = catch_panic(|| {
            RsReduction {
                terms: self.terms.clone(),
            }
            .simplify()
            .terms
        })
        .map_err(reduction_failed)?;
        Ok(Reduction {
            terms,
            dimension: self.dimension,
        })
    }

    /// Number of terms in this reduction; it can be zero for a vanishing integral.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> assert len(reduction) == len(reduction.terms)
    fn __len__(&self) -> usize {
        self.terms.len()
    }

    /// Display the number of retained master-integral terms.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> summary = repr(reduction)
    fn __repr__(&self) -> String {
        format!("Reduction({} terms)", self.terms.len())
    }
}

// ---------------------------------------------------------------------------
// MasterIntegral
// ---------------------------------------------------------------------------

/// A scalar tadpole, bubble, triangle or box returned in ``Reduction.terms``.
///
/// There is no direct constructor. ``arguments`` gives the invariants and
/// squared masses in primitive order; the squared renormalization scale is
/// supplied separately to ``to_expression``. The result is symbolic until
/// passed to the numerical or coefficient-evaluation API.
///
/// Examples
/// --------
/// >>> from symbolica import S, E
/// >>> from symbolica.community import hepkit as hep
/// >>> from symbolica.community.hepkit import oneloop
/// >>> d, k, p, s = S("d", "k", "p", "s")
/// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
/// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
/// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
/// >>> reduction = oneloop.reduce(family, [1, 1])
/// >>> coefficient, master = reduction.terms[0]
/// >>> assert master.kind == "bubble" and master.head == "B0"
/// >>> assert master.arguments == [s, E("0"), E("0")]
/// >>> assert master.to_expression() == oneloop.B0(s, 0, 0, 1)
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    frozen,
    from_py_object,
    eq,
    name = "MasterIntegral",
    module = "symbolica.community.hepkit.oneloop"
)]
#[derive(Clone, PartialEq)]
pub struct MasterIntegral {
    inner: RsMasterIntegral,
}

impl MasterIntegral {
    /// Borrow the scalar master for a shared-kernel evaluator.
    pub fn as_master(&self) -> &RsMasterIntegral {
        &self.inner
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl MasterIntegral {
    /// Topology name: "tadpole", "bubble", "triangle" or "box".
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> coefficient, master = reduction.terms[0]
    /// >>> assert master.kind == "bubble"
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            RsMasterIntegral::Tadpole { .. } => "tadpole",
            RsMasterIntegral::Bubble { .. } => "bubble",
            RsMasterIntegral::Triangle { .. } => "triangle",
            RsMasterIntegral::Box { .. } => "box",
        }
    }

    /// Primitive symbol name: "A0", "B0", "C0" or "D0".
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> coefficient, master = reduction.terms[0]
    /// >>> assert master.head == "B0"
    #[getter]
    fn head(&self) -> &'static str {
        match self.inner {
            RsMasterIntegral::Tadpole { .. } => "A0",
            RsMasterIntegral::Bubble { .. } => "B0",
            RsMasterIntegral::Triangle { .. } => "C0",
            RsMasterIntegral::Box { .. } => "D0",
        }
    }

    /// Kinematic arguments in primitive order, excluding the squared scale.
    ///
    /// A0 takes one squared mass; B0 takes an external invariant and two squared
    /// masses; C0 takes three invariants then three squared masses; D0 takes four
    /// external squared momenta, s12, s23, then four squared masses.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> coefficient, master = reduction.terms[0]
    /// >>> assert master.arguments == [s, E("0"), E("0")]
    #[getter]
    fn arguments(&self) -> Vec<PythonExpression> {
        self.inner
            .arguments()
            .into_iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Assemble primitive scalar-master calls with the squared scale last.
    ///
    /// The default scale is exactly one. This builds a symbolic expression; use
    /// ``master_coefficients`` or ``reduction_coefficients`` to obtain Laurent
    /// coefficients with native evaluation hooks.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> coefficient, master = reduction.terms[0]
    /// >>> mu2 = S("mu2")
    /// >>> expression = master.to_expression(mu2)
    /// >>> assert expression == oneloop.B0(s, 0, 0, mu2)
    ///
    /// Parameters
    /// ----------
    /// mu_squared : Expression or None, optional
    ///     Squared renormalization scale; None uses one.
    #[pyo3(signature = (mu_squared = None))]
    fn to_expression(&self, mu_squared: Option<PythonExpression>) -> PyResult<PythonExpression> {
        let mu_squared = mu_squared.map(|value| value.expr).unwrap_or(Atom::num(1));
        validate_namespace(&mu_squared)?;
        catch_panic(|| OneLoopMasters.symbol_with_scale(&self.inner, &mu_squared))
            .map(Into::into)
            .map_err(reduction_failed)
    }

    /// Display the primitive family and its kinematic arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import S, E
    /// >>> from symbolica.community import hepkit as hep
    /// >>> from symbolica.community.hepkit import oneloop
    /// >>> d, k, p, s = S("d", "k", "p", "s")
    /// >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    /// >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    /// ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    /// >>> reduction = oneloop.reduce(family, [1, 1])
    /// >>> coefficient, master = reduction.terms[0]
    /// >>> summary = repr(master)
    fn __repr__(&self) -> String {
        let args: Vec<String> = self
            .arguments()
            .into_iter()
            .map(|a| a.expr.to_string())
            .collect();
        format!("MasterIntegral({}({}))", self.head(), args.join(", "))
    }
}

#[cfg(feature = "python_stubgen")]
define_stub_info_gatherer!(stub_info);
