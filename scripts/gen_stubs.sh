#!/usr/bin/env bash
# Regenerate python/symbolica/community/hepkit/oneloop/__init__.pyi (reducer classes only).
#
# pyo3-stub-gen's `define_stub_info_gatherer!` hard-codes
# `$CARGO_MANIFEST_DIR/pyproject.toml` as the place it reads the module name and
# the Python source root from, and it needs a binary target to run in. Neither
# belongs in a crate that is only ever linked into symbolica-community, so this
# script scaffolds both, runs the generator, and takes them away again.
#
# Once the crate is wired into a symbolica-community checkout, that root's own
# `cargo run --bin stub_gen --features python_stubgen` regenerates this file
# along with every other community module's, and is the authoritative path.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
crate="$repo/crates/one-loop-reduce-python"

cleanup() {
    rm -f "$crate/pyproject.toml" "$crate/src/bin/stub_gen.rs"
    rmdir "$crate/src/bin" 2>/dev/null || true
    # Symbolica and FeynKit ship their own stubs.
    rm -f "$repo"/python/symbolica/core.pyi \
        "$repo"/python/symbolica/community/hepkit.pyi \
        "$repo"/python/symbolica/community/hepkit/__init__.pyi \
        "$repo"/python/symbolica/community/tensor.pyi
}
trap cleanup EXIT

cat > "$crate/pyproject.toml" <<'TOML'
[project]
name = "symbolica"
[tool.maturin]
module-name = "symbolica.core"
python-source = "../../python"
TOML

mkdir -p "$crate/src/bin"
cat > "$crate/src/bin/stub_gen.rs" <<'RUST'
fn main() -> pyo3_stub_gen::Result<()> {
    oneloopreduce_python::stub_info()?.generate()?;
    Ok(())
}
RUST

cargo run --manifest-path "$repo/Cargo.toml" \
    -p one-loop-reduce-python --features python_stubgen --bin stub_gen

mkdir -p "$repo/python/symbolica/community/hepkit/oneloop"
mv "$repo/python/symbolica/community/hepkit/oneloop.pyi" \
    "$repo/python/symbolica/community/hepkit/oneloop/__init__.pyi"

# Keep the owning generated stub free of trailing spaces, including doc lines.
python3 - "$repo/python/symbolica/community/hepkit/oneloop/__init__.pyi" <<'NORMALIZE_STUB'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
original = path.read_bytes()
normalized = re.sub(rb"[ \t]+(?=\r?$)", b"", original, flags=re.MULTILINE)
if normalized != original:
    path.write_bytes(normalized)
NORMALIZE_STUB

# The generator writes the canonical module's flat stub. The community host
# merges this reducer surface with the master evaluator's API in its own stub.
# It also re-emits Symbolica's and FeynKit's stubs, which cleanup removes.
echo "wrote python/symbolica/community/hepkit/oneloop/__init__.pyi"
