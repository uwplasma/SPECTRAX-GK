from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def _compact(path: str) -> str:
    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


REQUIRED_PHRASES = {
    "docs/release_scope.rst": (
        "a scoped model-development and optimization-screening result",
        "No runtime/TOML absolute-flux predictor",
        "Solovev and shaped-pressure stress outliers outside the scoped claim",
        "W7-X TEM / kinetic-electron validation",
        "W7-X long-window zonal recurrence/damping closure",
        "selected QA optimized-equilibrium audit is the current scoped exception",
    ),
    "docs/verification_matrix.rst": (
        "Closed as scoped model-development result / failed promotion gate",
        "does not promote a runtime/TOML absolute-flux predictor",
        "W7-X zonal long-window recurrence/damping and W7-X TEM / kinetic-electron validation remain outside",
        "Production nonlinear optimization is promoted only for the selected optimized-equilibrium audit",
    ),
    "README.md": (
        "not a runtime/TOML universal absolute-flux predictor",
        "declared Solovev and shaped-pressure stress outliers",
        "W7-X zonal long-window recurrence/damping and W7-X TEM / kinetic-electron extensions are deferred",
        "converged post-transient heat-flux windows",
        "Sensitivity sweeps can use the same deterministic independent-work reconstruction, but they need a dedicated",
    ),
    "docs/performance.rst": (
        "Sensitivity sweeps are covered by",
        "before any speedup claim is promoted",
        "Communication-aware nonlinear domain decomposition remains",
    ),
    "docs/parallelization.rst": (
        "It is not a production nonlinear domain",
        "whole-state nonlinear sharding speedup",
    ),
    "docs/examples.rst": (
        "opt-in electrostatic linear-RHS identity artifact",
        "publication speedup claim",
    ),
}

FORBIDDEN_PHRASES = (
    "is a runtime/TOML absolute-flux predictor",
    "promotes a runtime/TOML absolute-flux predictor",
    "runtime/TOML absolute-flux predictor is accepted",
    "universal nonlinear transport model is promoted",
    "W7-X TEM / kinetic-electron validation is closed",
    "W7-X zonal long-window recurrence/damping closure is closed",
    "production nonlinear heat-flux stellarator optimization is release-ready",
    "nonlinear production optimization is release-ready",
    "optimized-equilibrium nonlinear heat-flux validation is closed",
    "production parallelization path for linear scans, quasilinear studies, sensitivity sweeps, and UQ ensembles",
    "production parallelization path for linear scans, quasilinear studies, sensitivity sweeps",
    "current production-parallelization identity artifact",
    "production nonlinear sharding speedup",
    "production nonlinear domain-decomposition speedup claim is closed",
    "broad multi-GPU nonlinear speedup claim",
)

COMPARISON_CODE_PATTERN = re.compile(
    r"\bGX\b|\bgx\b|gx_|_gx|GX-reference|comparison-code"
)
COMPARISON_ALLOWED_SOURCE_PREFIXES = (Path("src/spectraxgk/validation/benchmarks"),)


def test_claim_scope_pages_keep_required_quasilinear_boundaries() -> None:
    missing: list[str] = []
    for path, phrases in REQUIRED_PHRASES.items():
        text = _compact(path)
        missing.extend(f"{path}: {phrase}" for phrase in phrases if phrase not in text)

    assert not missing


def test_claim_scope_pages_avoid_promoted_unscoped_claims() -> None:
    violations: list[str] = []
    for path in REQUIRED_PHRASES:
        text = _compact(path)
        violations.extend(
            f"{path}: {phrase}" for phrase in FORBIDDEN_PHRASES if phrase in text
        )

    assert not violations


def test_core_source_avoids_comparison_code_terminology_outside_benchmarks() -> None:
    violations: list[str] = []
    source_root = ROOT / "src" / "spectraxgk"
    for path in source_root.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if any(
            rel.is_relative_to(prefix) for prefix in COMPARISON_ALLOWED_SOURCE_PREFIXES
        ):
            continue
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if COMPARISON_CODE_PATTERN.search(line):
                violations.append(f"{rel}:{line_no}: {line.strip()}")

    assert not violations
