from __future__ import annotations

import subprocess
import sys

from support.paths import REPO_ROOT

import spectraxgk


def test_standalone_public_aliases_match_current_exports() -> None:
    assert spectraxgk.LinearExplicitTimeConfig is spectraxgk.ExplicitTimeConfig
    assert (
        spectraxgk.integrate_nonlinear_diagnostics
        is spectraxgk.integrate_nonlinear_explicit_diagnostics
    )


def test_root_import_keeps_pure_submodule_imports_dependency_light() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT / "src")!r})
import spectraxgk
assert "numpy" not in sys.modules
assert "jax" not in sys.modules
from spectraxgk.parallel.decomposition import build_independent_portfolio_decomposition
contract = build_independent_portfolio_decomposition(
    4, requested_shards=2, workload="independent_ky_scan"
)
assert contract.actual_shards == 2
assert "numpy" not in sys.modules
assert "jax" not in sys.modules
"""
    subprocess.run([sys.executable, "-S", "-c", script], check=True)
