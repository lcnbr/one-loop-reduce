# one-loop-reduce

Reduces a one-loop `hep.IntegralFamily` to the scalar masters `A0`, `B0`, `C0` and
`D0` of [oneloopmaster](https://github.com/alphal00p/oneloopmaster), with coefficients
exact in the symbolic dimension. The family, its scalar products and kinematics are
FeynKit's own objects, shared with the other HEP backends; this package adds no family
type of its own.

## Python

It ships in the Symbolica community wheel as `symbolica.hepkit.oneloop`.

```python
from symbolica import E, S
from symbolica import hepkit as hep
from symbolica.hepkit import oneloop

D, k, p, q, m2, s1, s2, s = S("D", "k", "p", "q", "m2", "s1", "s2", "s")
kin = (hep.Kinematics(D, momenta=[k, p, q])
       .with_scalar_product(p, p, s1)
       .with_scalar_product(q, q, s2)
       .with_scalar_product(p, q, (s-s1-s2)/2))
family = hep.IntegralFamily(
    [k], [p, q],
    [kin.scalar_product(v, v) - m2 for v in [k, k+p, k+p+q]],
    kinematics=kin,
)
reduction = oneloop.reduce(family, [1, 1, 1], numerator=kin.scalar_product(k, p))
assert reduction.dimension == D
mu2 = S("mu2")
print(reduction.to_expression(mu_squared=mu2))
# [B0(s,m2,m2,mu2) - B0(s2,m2,m2,mu2)
#  - s1*C0(s1,s2,s,m2,m2,m2,mu2)] / 2
# Every master head above belongs to oneloopmaster::.
```


`powers` follows `family.denominators`: positive powers are propagators, zero omits a
line, and negative powers move the denominator into the numerator (this also covers
auxiliary entries from `family.complete()`).

Use a **symbolic dimension**, such as `hep.Kinematics(S("D"))`. Setting `D = 4` before
multiplying by the masters' Laurent series would lose the finite parts that come from
epsilon times a pole, so a concrete dimension is rejected.

`reduction.to_expression(mu_squared)` writes the result over oneloopmaster's
`A0(m², mu²)`, `B0(p², m0², m1², mu²)`, `C0(p1², p2², p3², m0², m1², m2², mu²)` and
`D0(p1², p2², p3², p4², s12, s23, m0², m1², m2², m3², mu²)`, scale last (default 1).
For numbers, `oneloop.reduction_coefficients(reduction, mu_squared=mu2)` expands in
epsilon about `D = 4` and returns `[finite, 1/ε, 1/ε²]` in oneloopmaster's
normalization, ready for `Expression.evaluate`:

```python
coefficients = oneloop.reduction_coefficients(reduction, mu_squared=mu2)
point = {m2: 2, s1: -1, s2: -2, s: -3, mu2: 1}
print([coefficient.evaluate(point) for coefficient in coefficients])
```

Exact master formulas and analytic-branch selection are oneloopmaster's; see its README.

## Limits

- One loop, and a numerator polynomial in the loop scalar products with tensors
  already contracted. Positive powers of eikonal (linear) denominators are
  unsupported; propagator shifts need real coefficients.
- Propagator powers total at most 32; numerator degree at most 20. These bound what
  is supported, not how fast: highly dotted families can be slow.
- With five or more propagators the reduction drops an `O(ε)` term and needs
  four-dimensional external kinematics.
- Where the regulated on-shell limit does not exist, an exact IBP solve at the
  degenerate point takes over.
- A coefficient with a pole at `D = 4` would need positive orders in ε from the
  masters, which oneloopmaster does not supply, so `reduction_coefficients` rejects it.

## How it works

`reduce_family` takes each quadratic denominator as `a_i (k + r_i)² + c_i` from
FeynKit, keeps the `a_i^(-power_i)` prefactor, shifts the numerator with the loop
momentum and computes the pairwise invariants from the shared kinematics. Numerator
directions not spanned by the propagators get power-zero auxiliary lines, so the full
external Gram matrix is kept; dependent denominators are partial-fractioned by
FeynKit. The reduction itself is closed-form one-loop recursions (tensor reduction,
index lowering, the five-point step), falling back to exact Laporta elimination at
degenerate points.

## Rust and builds

The entry point is
`oneloopreduce::reduce_family(&feynkit_graph::IntegralFamily, powers, numerator)`;
`OneLoopMasters.symbol_with_scale(&master, &mu_squared)` builds the master calls.

FeynKit (`alphal00p/gammaloop`, branch `feynkit`) and oneloopmaster (branch `main`)
are git dependencies on the same branches the consuming root uses, so Cargo links one
copy of each; `Cargo.lock` records the exact commits. Symbolica is the released 3.0.1;
consuming roots must resolve a single copy. WebAssembly builds leave out the native
numerical dependencies.

```sh
cargo build --workspace --all-targets
cargo test --workspace -- --test-threads=1   # restricted Symbolica: one thread
cargo fmt --all --check
```

The Python tests need a community wheel with FeynKit and oneloopmaster:
`SYMBOLICA_HIDE_BANNER=1 pytest python/tests/test_oneloopreduce.py`. The validation
record is in [STATUS.md](STATUS.md).

## Citations

Once a reduction has run, `symbolica.get_citations()` includes this package and the
methods it implements: Passarino–Veltman, van Neerven–Vermaseren,
Fleischer–Jegerlehner–Tarasov, Tarasov, Chetyrkin–Tkachov and Laporta. Importing alone
adds nothing; using the masters adds oneloopmaster's own references.
