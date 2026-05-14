from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _compact(path: str) -> str:
    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


REQUIRED_PHRASES = {
    "docs/release_scope.rst": (
        "accepted only as a scoped manuscript model-selection result",
        "does not promote a runtime/TOML absolute-flux predictor",
        "W7-X TEM / kinetic-electron validation",
        "W7-X long-window zonal recurrence/damping closure",
        "selected QA optimized-equilibrium audit is the current scoped exception",
    ),
    "docs/verification_matrix.rst": (
        "Closed as scoped model-selection result",
        "does not promote a runtime/TOML absolute-flux predictor",
        "W7-X zonal long-window recurrence/damping and W7-X TEM / kinetic-electron validation remain outside",
        "Production nonlinear optimization is promoted only for the selected optimized-equilibrium audit",
    ),
    "README.md": (
        "scoped manuscript model-selection candidate, not a runtime/TOML absolute-flux predictor",
        "W7-X zonal long-window recurrence/damping and W7-X TEM / kinetic-electron extensions are deferred",
        "converged post-transient heat-flux windows",
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
)


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
