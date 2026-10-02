# Status

Running record of where this repo is. Newest entries at the top.

---

## 2026-09-29 — FeynKit's IntegralFamily is the input

Ben Ruijl's community-3.0.1 patch (#1): `reduce_family` over FeynKit's
`IntegralFamily` replaces this crate's own family, propagator and `dot` types;
masters are oneloopmaster's native `A0..D0`; internal symbols moved to
`oneloopmaster::`, and `oneloopreduce::` inputs are rejected as obsolete. Symbolica
is crates.io 3.0.1, FeynKit and oneloopmaster are git dependencies on the branches
the community root uses, and `get_citations` credits this package and the methods it
implements. The reductions are unchanged: `golden_master` is byte-identical.

---

## 2026-09-28 — exact IBP fallback for on-shell limits that do not exist

The two configurations with no termwise `δ → 0` limit now reduce. At those points the
master basis shrinks — a massless triangle with two on-shell legs is a bubble times
`1/(d−4)` — which no regulator limit can produce. `reduce/ibp.rs` solves the IBP
identities by Laporta elimination at the exact kinematics, `d` symbolic, numerators
rewritten as inverse propagators; `reduce()` uses it only when the fast path is not
finite, so every other result is unchanged (`golden_master` byte-identical).

Checked: equal to the recursion at generic kinematics (7 families); the triangle-to-bubble
relation, from the Gamma-function closed forms; equal to the shared-`δ` result wherever
that exists, up to `B0(0,m,m) = (d−2)/(2m²) A0(m)`, which the solver reduces further.

---

## 2026-09-28 — the two open physics questions, answered by tests

**`N ≥ 5` at `d = 4`.** `I_N = Σ cᵢ I_{N−1} + (N − d − 1) B I_N^(d+2)`, `B = Σ cᵢ`; the
reducer keeps the sum. On explicit momenta: a hexagon in four dimensions has `B = 0`
(exact in `d`), in five `B ≠ 0`; a pentagon has `B ≠ 0`, so it drops `2ε B I₅^(6−2ε)`,
which is `O(ε)`. Nothing that multiplies the step has a pole at `d = 4` (branch-E
coefficients are polynomial in `d`, the Cayley and Gram algebra is `d`-free), so `O(ε⁰)`
is safe.

**The shared `δ`.** Compared against sending each on-shell leg to zero in turn: equal
for eight configurations (boxes, triangles, a dotted pentagon, ranks up to 3). Two have
no termwise limit at all and are now `NonFiniteResult` errors, pinned by a test: a
raised power in a massless box beside on-shell legs, and a massive triangle with raised
powers and a numerator.

**Invented Gram entries** (`triangle_topo`'s `q3`, `bubble_topo`'s `q2`, `q3`): unreachable
since the numerator check rejects `dot(k, q_j)` past the family's chain, and nothing else
reads them.

---

## 2026-09-28 — integration review: errors instead of wrong answers, `hep.oneloop`

The module was linked into symbolica-community as `symbolica.community.hepkit.oneloop`,
beside oneloopmaster, and a review (`COMMUNITY_INTEGRATION_REVIEW.md`, 2026-09-23)
came back with six findings and two patches.

1. **`⧞` returned as success.** A `[2,2]` or `[3,1]` bubble at `p² = 0`, equal
   masses: the two lines are one denominator, so `det(Y) ≡ 0` and the `δ` limit is
   indeterminate. Dotted scalar families now merge lines with equal Cayley rows
   (exact); any non-finite result is `NonFiniteResult`.
2. **Loop momentum passed through as a constant.** A numerator not polynomial in
   `dot(k,k)`, `dot(k,q₁…q_{N−1})` is `UnsupportedNumerator`. This also stops
   `dot(k,q₂)` on a bubble being projected against invented Gram entries.
3. **Scratch symbols captured user input.** `reg_delta`, `xll`, `xq<n>`, `den<n>`,
   `routing_tmp_q<n>` in the input are `InvalidFamily`.
4. **Panics on malformed input.** Shape checks up front, `InvalidFamily`.
5. **Contract.** `MAX_NUMERATOR_DEGREE = 20` is an error up front; the README says
   `N ≥ 5` drops an `O(ε)` term.
6. **Python checks in CI.** Open: needs `hep.oneloop` on symbolica-community `main`.

Patch A (the module move, `to_oneloopmaster()`, `native` without
`symbolica/default` so no allocator is imposed on the host) is applied, with the
lock update it lacked and a regenerated stub. Patch B is the host side; its
oneloopmaster-free part is symbolica-community PR #12.

Also, from `378d2fa`'s message: the move to Symbolica 3.0, and `together()` before
`δ → 0`, which fixed the raised-power triangle below.

---

## 2026-09-15 — the layout is flattened, two harnesses cut

`benchmarks/rust/` → `benchmarks/` (`git mv`, history follows). The `rust/`
level only ever distinguished the Rust examples from a `python/` sibling that
was deleted in `c816fde`; with no sibling it named nothing.

Two of the nine examples went with it. **`bench.rs`** (209 lines) swept every
topology tadpole→heptagon at ranks 0–4, scalar and dotted, and printed a status
string per case; the library suite now *asserts* that same grid
(`scalar_pentagon_reduces_to_five_boxes`, `scalar_hexagon_recurses_down_to_boxes`,
`heptagon_numerator_reduces_to_finite_masters`, the `dotted_*` family), and CI
enforces it, which printing never did. Its timing column duplicated
`time_reduce.rs`; its header claimed "(gitignored)" while the file was tracked.
**`frontier_map.rs`** (123 lines) mapped the on-shell-massless-leg frontier by
catching panics — and now prints `OK` for all twelve cells, because `reduce()`'s
regularized path erased the frontier it was built to find. `README.md` already
carries that frontier as prose (correctly labelled *unaided*, i.e.
pre-regularization) and `on_shell_massless_triangle_rank2_regularizes` pins the
cell that still matters.

`python/` was reviewed and **kept as-is** — see the note in `README.md`. The
four-level mirror is the literal destination inside symbolica-community, so
wiring the module up is `cp -r python/symbolica <root>/python/` with no path to
retype. It is also what `scripts/gen_stubs.sh` wants: pyo3-stub-gen derives its
output path from the registered dotted module name and writes
`python/symbolica/community/oneloopreduce.pyi` regardless, so a flat layout would
make the script build that tree, move one file out of it, and tear it back down
on every run.

31 files / 10548 lines → 29 / 10253. Excluding this file, which grew by the
entry you are reading: 10220 → 9880.

### Open, not fixed here (fixed in `378d2fa`; see the 2026-09-28 entry)

A raised-power massive triangle with two on-shell legs — exponents `[2,1,1]`,
invariants `[0, 0, 2/5]`, `m² = 1` — returns `Ok` with **four terms, two of whose
coefficients are Symbolica's indeterminate glyph**, with
`Created infinity by raising 0 to the power of -1` on stderr. The `1/δ` poles are
not cancelling on the raised-power path the way they do at rank 2. Found while
deciding whether `frontier_map` was still measuring anything: it reported that
case `OK`, since it only checked `Ok` vs `Err` and never looked at the
coefficients. Nothing in the suite covers raised powers at on-shell kinematics.

---

## 2026-09-15 — the prose is cut; here is where it went

The repo was about 45 % evidence and prose. That material is not deleted, it is
**relocated to an address**: it stays in gammaloop, where the work was done, and
in this repo's own history. Both are written down below so it can actually be
recovered.

### The archive

| | |
|---|---|
| Repo | `alphal00p/gammaloop` (`git@github.com:alphal00p/gammaloop.git`) |
| Branch | `oneloop`, PR **#86** (left open deliberately) |
| Commit | **`c0f597693ae75f1f4c4b2b0578f11fe0b92b4e77`** — *"oneloop benchmarks: cross-check masters against oneloopmaster"*, 2026-09-14 |
| Pushed | yes — `origin/oneloop` is at that exact SHA, 0 ahead / 0 behind |
| Docs | `crates/oneloop/docs/` — 12 files, including the five never present here (`05-app.md`, `06-benchmarks.md`, `07-benchmark-report.md`, `08-projector-traces.md`, `CHANGELOG.md`) |
| Benchmarks | `crates/oneloop/benchmarks/` — `python/` (13 scripts, 3 never present here), `rust/` (11 examples, incl. `reduce_regularized_draft.rs` and `symbolic_delta.rs` which are not here), `README.md`, `madloop_reference.md`, `MONDAY_AGENDA.md` |

```bash
git -C <gammaloop> show c0f597693:crates/oneloop/docs/04-frontier.md
git -C <gammaloop> show c0f597693:crates/oneloop/benchmarks/python/crosscheck.py
```

The gammaloop copy is the **superset** — it predates the cuts made here, so it is
the one to read.

### This repo's own history

Everything deleted is also one `git show` away without leaving this checkout.
The last commit that still contains it is **`4d6ca37`**, the commit that added
this record; `c4242ac` is the last commit of the repo as it stood before the cut
began. Either works as a `git show` prefix.

```bash
git show c4242ac:docs/00-summary.md                                   # the document map
git show c4242ac:crates/one-loop-reduce/benchmarks/README.md          # the harness guide
git show c4242ac:crates/one-loop-reduce/benchmarks/madloop_reference.md
git show c4242ac --stat                                               # the whole tree as it stood
git log --diff-filter=D --stat -1 -- docs/                            # the deletion itself
```

### What was cut, and what stayed

Deleted: `docs/` (8 files, 1604 lines), `benchmarks/python/` (10 scripts, 1570
lines), `benchmarks/README.md` (164) and `benchmarks/madloop_reference.md` (450).
**3788 lines, 20 files.**

Kept: the library (`crates/one-loop-reduce/src/`), the bindings
(`crates/one-loop-reduce-python/src/`), the nine Rust benchmark examples
(`benchmarks/rust/`), the Python facade and FFI test (`python/`), CI, scripts and
the two manifests. What is left is the mergeable artifact and nothing else.

Facts that the doc comments in `src/` reached for by filename now state the fact
directly instead of pointing at a file — the `gg→h` closed forms in `bridge.rs`,
the degenerate-Gram frontier note in `emit_reductions.rs`. No sentence lost its
content to the cut.

### Two Rust examples lost their driver

`box_reduce` and `wloop_reduce` were argv-driven emitters called by
`box_validate.py` and `wloop_assemble.py`. They take their kinematics as
integer numerator/denominator pairs and index `argv` unconditionally, so with no
arguments they panic on an empty vector. Both are kept — they are the only
in-repo exercise of the `δ`-regularized on-shell reduction — and their `//!`
headers now carry the exact invocation the deleted drivers used, so each still
runs standalone.

---

## 2026-09-15 — routing split out, CI, and the FFI boundary under test

### `src/routing.rs` (`a88e410`)

`bridge.rs` was doing two jobs. Everything that turns offset atoms into a propagator
chain — `offset_dirs`, `chain_order`, `chain_slots`, `relabel_numerator`,
`check_numerator_directions`, `invariants_from_offsets` — moved to a model-agnostic
`routing` module; the tensor-to-`dot` translation stayed. `GammaloopHeads` /
`GammaloopEdge` / `family_from_gammaloop` became `TensorHeads` / `LoopEdge` /
`family_from_tensor_numerator`, with `pub use` aliases under the old names.

`bridge.rs` 1569 → 1236 lines, `routing.rs` 422. Honest total: code +24 (a module
header and the alias block), tests +65 — four new and two moved. The new four cover
the real box LMB routing, `chain_slots`' refusals, `offset_dirs`, and the q1↔q2
relabel; none was reachable before, and `chain_order`'s justifying case had been
pinned only in gammaloop's own test suite.

gammaloop does **not** consume this repo: `crates/gammalooprs/Cargo.toml` points at a
vendored in-tree `crates/oneloop`. The renames cannot break it until someone
re-syncs, and the aliases keep that source-compatible when they do.

### CI — `.github/workflows/ci.yml`

`fmt --check`, `build`, `clippy -D warnings`, `cargo test -- --test-threads=1`,
matrixed over symbolica `main` and `dev` by `sed`-ing the root `[patch.crates-io]`
table.

**No secret is required** — superseded by `28b6a5e`, below. This entry originally
demanded an *environment* secret named `SYMBOLICA_LICENSE` under an environment
called `symbolica`, and failed the run at the first step until it existed. That was
reversed: CI now runs Symbolica *restricted* by default, which passes the suite
because `--test-threads=1` already assumes one instance on one thread, and logs a
`::notice::` naming the mode instead of failing. To run licensed, add a plain
**repository** secret — *Settings → Secrets and variables → Actions* — named
`SYMBOLICA_LICENSE`; the workflow reads `secrets.SYMBOLICA_LICENSE` with no
`environment:` key, so an environment-scoped secret would not reach it. The variable
name is confirmed in symbolica's `src/license.rs`; symbolica-community's own workflow
sets `SYMBOLICA_LICENSE_KEY`, which is read nowhere.

Standing the `dev` leg up found one real incompatibility: `LicenseManager` lives at
`symbolica::license::` on `main` and at the crate root on `dev`. Both expose it from
`symbolica::prelude`, so the twelve `ensure_symbolica_license` call sites now spell
it that way and the crate compiles against either branch. Verified locally, both
legs: `cargo test --workspace -- --test-threads=1` gives 70 + 7 passed, 1 ignored,
and clippy `-D warnings` and `fmt --check` are clean on each.

### `python/tests/test_oneloopreduce.py`

62 lines, plain pytest, no fixtures. The five things the Rust suite cannot reach:
the module imports and `initialize_module()` has run; `dot` still carries
`Symmetric, Linear` (the redefinition trap); a rank-one massive triangle reduces to
the right three coefficients, `Expression` in and out; a malformed family raises
`ValueError`; and the `MAX_TOTAL_INDEX` guard raises instead of aborting the
interpreter. Not wired into CI — it needs the module built into a
symbolica-community root.

**It has not been run.** The module built in the earlier session is gone; nothing
importable as `symbolica.community.oneloopreduce` exists on this machine and there
is no symbolica-community checkout to rebuild it from. The expected values were
taken from the Rust reducer directly, by probing `reduce()` on the same two
families — not guessed — but the file itself has never executed.

---

## 2026-09-15 — the unbounded recursion is fixed in the library

`reduce()` now returns `Result<Reduction, OneLoopError>` and refuses, before entering the
recursion, any target whose propagator indices are negative or total more than
`MAX_TOTAL_INDEX = 32`. The guard the Python constructor was carrying is gone; the binding
forwards the library's error, so the bound lives in one place and the Rust API is no longer
exposed.

### Why a bound rather than a deeper base case

`reduce_cayley` descends depth-first and every level drops either one unit of total index
or one propagator, so stack depth is bounded by `sum(exponents) + N` — and a negative index
never bottoms out at all. Overrunning the stack is an *abort*: `catch_unwind` cannot
intercept it and neither can pyo3's trampoline.

Measured on this reducer, macOS arm64:

| | |
|---|---|
| `reduce_cayley` frame, debug | 3408 bytes (exact, from stack-pointer deltas) |
| `reduce_cayley` frame, release | 560 bytes |
| overflow depth, 2 MiB thread stack, debug | 486 — SIGABRT |
| overflow depth, 8 MiB thread stack, debug | 2323 — SIGABRT |
| observed depth | exactly the total index, confirmed on bubble/triangle/N-gon |

Runtime hits the wall far sooner, because the tree branches `N(N-1)+1` ways per level. A
release-build dotted massless bubble: total index 11 → 0.05 s, 13 → 0.23 s, 15 → 1.84 s,
17 → 15.3 s, a factor ~2.9 per unit after that (≈ a day at 25). So 32 sits an order of
magnitude below the overflow floor and far above anything that would ever have returned —
it rejects only input that was never going to finish, and truncates nothing.

### Cost

`reduce`'s signature change touched 66 call sites across tests and the benchmark examples
(mechanical `.unwrap()`), and `amplitude()` became `Result` with it. That was worth it over
a second, checked entry point: one entry point means the abort is not reachable at all.

---

## 2026-09-14 — extracted from gammaloop, Python module works

The crate is standalone and the Symbolica-community binding runs end to end.

### What happened

Lifted out of `alphal00p/gammaloop` (`crates/oneloop`, PR #86) via `git subtree split`,
so all 49 original commits carry over with their authors and dates. Restructured into a
two-crate workspace, cut the gammaloop dependency, renamed the symbol namespace, and
added the Python module.

| | |
|---|---|
| Library | `crates/one-loop-reduce` — lib name `oneloopreduce` |
| Python module | `crates/one-loop-reduce-python` — registers as `symbolica.community.oneloopreduce` |
| Symbol namespace | `oneloopreduce::` (was `oneloop::`) |
| Deps | `symbolica 2.2`, `thiserror 2.0`. Nothing else. |

### Verified

Run on 2026-09-14, macOS arm64, Symbolica restricted mode (no `SYMBOLICA_LICENSE` set).

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` from clean | 0 errors, 49 s |
| `cargo test --workspace -- --test-threads=1` | 62 library + 11 binding, 0 failed, 1 ignored |
| ignored slow test (`dotted_heptagon…`) | passes in release, 4.8 s |
| `cargo clippy --workspace --all-targets` | 0 warnings |
| `cargo fmt --all --check` | clean |
| `cargo tree -d` | one symbolica, one numerica, one graphica |

Test-function names were diffed against gammaloop's `crates/oneloop/src`: 63 on both
sides, identical name for name. Nothing regressed through the rename.

### Physics still reproduces

The point of the extraction was that nothing about the physics should change. It didn't —
these are the same numbers gammaloop produces.

| Validation | Result |
|---|---|
| gg→H form factor `A_{1/2}(τ)`, six points | max rel. err **3.23e-13** |
| gg→H assembled \|M\|² vs MadLoop `9.3702613e-3` | ratio **0.9996** at α_s = 0.1114 |
| H→γγ, W loop, rank 6 | PASS, Γ = **9.102 keV** (SM LO ≈ 9.1) |
| RSP identity, 2000 random pentagons | PASS |
| Heptagon power-lowering, degenerate | PASS |
| Pentagon reduction, dotted pentagon | PASS |

Reproducing these requires `oneloop_bridge` (avh_olo) on the Python path and the
cross-engine driver scripts, which are no longer in this repo — see the archive
entry at the top of this file.

### The Python surface

`symbolica.community.oneloopreduce` — typed objects throughout, no strings.

```python
from symbolica import E
from symbolica.community.oneloopreduce import IntegralFamily, Propagator

fam = IntegralFamily(
    [Propagator(E("msq"))] * 3,
    [E("p1sq"), E("s"), E("p2sq")],
    numerator=E("oneloopreduce::dot(oneloopreduce::k, oneloopreduce::q1)"),
)
print(fam.reduce().simplify().to_expression())
# -1/2*p1sq*C0(p1sq,p2sq,s,msq,msq,msq) + 1/2*B0(s,msq,msq) - 1/2*B0(p2sq,msq,msq)
```

`Propagator`, `IntegralFamily`, `Reduction`, `MasterIntegral`. Coefficients come back as
Symbolica `Expression`s, so they compose with everything else in the shared kernel.

Verified against a symbolica-community-shaped root built with maturin — the module
imports, the symbols keep their `Symmetric, Linear` attributes, and the reduction above
runs in a real interpreter.

### Decisions

| Decision | Choice | Status |
|---|---|---|
| Repo location | Personal (`ecavan/one-loop-reduce`) | Ben's suggestion — "it is your project" |
| Name | `oneloopreduce` | **Needs Ben + Cedric to confirm.** Cedric's `oneloopmaster` currently returns `get_name() -> "oneloop"`; two modules with the same name silently clobber each other in `sys.modules`. |
| gammaloop PR #86 | Leave open | It gave the work access to gammaloop; theirs to close |
| `oneloop` branch in gammaloop | Keep | The deployed app needs `--reduce` on a buildable ref |
| History | Preserved via `subtree split` | 49 commits, original authors and dates |
| `symbolica` dependency | Plain version requirement, `"2.2"` | **Not** a git dep — see below |
| `Cargo.lock` | Tracked | The patch table points at a moving branch |

### Why symbolica must not be a git dependency

`[patch.crates-io]` at a consuming root only rewrites dependencies declared as
crates.io dependencies. Declare `symbolica = { git = … }` in a member crate and Cargo
treats it as a separate source, compiles a second copy of Symbolica, and produces two
disjoint global symbol tables — expressions from one cannot be used with the other, with
no error. Both members use `symbolica = "2.2"`; the git redirect lives only in the root
`[patch.crates-io]`, where it governs local builds and is inert when symbolica-community
consumes the crate.

Confirmed empirically: a simulated community root resolves to one symbolica, and
`cargo tree -d` reports no duplicates.

### Known issues

**`momentum` is not exposed on `Propagator`.** The field exists in the Rust struct but
`reduce` never reads it; external offsets come from `invariants`. Exposing a field the
reducer ignores would be a trap. `bridge.rs` is the only consumer.

### Deviations from the original plan

- `default-features = false` on symbolica does not build. Symbolica declares
  `numerica = { default-features = false }`, so dropping symbolica's defaults leaves
  numerica with neither an integer nor a float backend (28 errors). Naming a backend
  explicitly is not portable either — crates.io 2.2.0 spells them `gmp`/`no_gmp` while
  `symbolica-dev/symbolica@main` spells them `integer-gmp`/`float-mpfr`. Plain
  `symbolica = "2.2"` with defaults is the only spelling that resolves against both, and
  is what `example_extension` uses.
- `LicenseManager` is at `symbolica::license::LicenseManager` on `main`, not the crate
  root.

### For gammaloop

gammaloop's workspace requests symbolica's `gmp` feature, which no longer exists on
`symbolica-dev/symbolica@main` (it is `integer-gmp` there). That will break whoever moves
gammaloop off the `dev` branch.

### Not verified

- No build against the **real** `symbolica-community` — the check used a faithful but
  minimal root (no vakint/idenso/spynso3, and `set_python_integration_functions` was not
  called). Feature unification across all community modules at once is untested.
- abi3 build not exercised through the community root's `module` feature path.
- Restricted mode only; `SYMBOLICA_LICENSE` is not set here.
- The `.pyi` has not been run through a type checker.
- No Python test suite committed — the interpreter tests were run in a scratch directory.
- No CI in this repo yet.

### Next

1. Get the name confirmed by Ben and Cedric before anything is registered upstream.
2. ~~CI — GitHub Environment gating `SYMBOLICA_LICENSE`, matrix over symbolica `main`
   and `dev`~~ — done; **the `symbolica` environment secret still has to be created.**
3. ~~Commit the Python tests~~ — done, `python/tests/`; still never executed.
4. ~~Fix the recursion bound in `reduce.rs`~~ — done, `e9d58ab`.
5. ~~Promote the routing logic out of `bridge.rs` into a model-agnostic module~~ — done,
   `src/routing.rs`.
6. `reduce_diagram(FeynmanDiagram)` — most of `bridge.rs` deletes itself once FeynKit
   hands over `MomentumSignature.integer_coefficients()` directly, straight into
   `routing::chain_order`.
7. Sign the Ruijl Research CLA (commits must be authored from the signature email), then
   PR to `symbolica-dev/symbolica-community`.
