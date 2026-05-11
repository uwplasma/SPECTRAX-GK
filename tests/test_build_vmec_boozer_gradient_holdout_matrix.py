"""Tests for the VMEC/Boozer multi-equilibrium gradient holdout matrix."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_gradient_holdout_matrix.py"
spec = importlib.util.spec_from_file_location(
    "build_vmec_boozer_gradient_holdout_matrix", SCRIPT
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_gate(
    path: Path,
    *,
    case: str,
    kind: str,
    passed: bool = True,
    source_scope: str = "mode21_vmec_boozer_state",
    mboz: int = 21,
    nboz: int = 21,
    extra_objectives: list[dict[str, object]] | None = None,
) -> None:
    objective_gates = [
        {
            "objective": "gamma",
            "passed": passed,
            "rel_error": 1.0e-3,
            "abs_error": 2.0e-4,
        },
        {
            "objective": "omega",
            "passed": True,
            "rel_error": 2.0e-3,
            "abs_error": 3.0e-4,
        },
    ]
    if extra_objectives is not None:
        objective_gates.extend(extra_objectives)
    path.write_text(
        json.dumps(
            {
                "kind": kind,
                "case_name": case,
                "passed": passed,
                "source_scope": source_scope,
                "mboz": mboz,
                "nboz": nboz,
                "surface_stencil_width": None,
                "objective_gates": objective_gates,
            }
        ),
        encoding="utf-8",
    )


def test_gradient_holdout_matrix_summarizes_passed_gates(tmp_path: Path) -> None:
    qh_freq = tmp_path / "qh_freq.json"
    qh_ql = tmp_path / "qh_ql.json"
    li_freq = tmp_path / "li_freq.json"
    li_ql = tmp_path / "li_ql.json"
    for path, case, kind in [
        (qh_freq, "nfp4_QH_warm_start", "frequency"),
        (qh_ql, "nfp4_QH_warm_start", "quasilinear"),
        (li_freq, "li383_low_res", "frequency"),
        (li_ql, "li383_low_res", "quasilinear"),
    ]:
        _write_gate(path, case=case, kind=kind)

    payload = mod.build_gradient_holdout_matrix(
        (
            ("QH", "frequency", qh_freq),
            ("QH", "quasilinear", qh_ql),
            ("Li383", "frequency", li_freq),
            ("Li383", "quasilinear", li_ql),
        )
    )

    assert payload["kind"] == "vmec_boozer_gradient_holdout_matrix"
    assert payload["passed"] is True
    assert payload["summary"]["n_cases"] == 2
    assert payload["summary"]["max_relative_error"] == 2.0e-3
    assert payload["rows"][0]["objectives"]["gamma"] is True


def test_gradient_holdout_matrix_writes_artifacts(tmp_path: Path) -> None:
    gate = tmp_path / "gate.json"
    _write_gate(gate, case="case", kind="frequency")
    payload = mod.build_gradient_holdout_matrix((("case", "frequency", gate),))

    paths = mod.write_gradient_holdout_matrix(payload, out=tmp_path / "matrix.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "matrix.json").read_text(encoding="utf-8"))
    assert saved["claim_level"] == (
        "multi_equilibrium_reduced_linear_quasilinear_and_nonlinear_window_estimator_gradient_gate_"
        "not_production_nonlinear_optimization"
    )


def test_gradient_holdout_matrix_requires_mode21_scope_and_mode_floor(
    tmp_path: Path,
) -> None:
    good_gate = tmp_path / "good.json"
    wrong_scope_gate = tmp_path / "wrong_scope.json"
    underresolved_gate = tmp_path / "underresolved.json"
    _write_gate(good_gate, case="nfp4_QH_warm_start", kind="frequency")
    _write_gate(
        wrong_scope_gate,
        case="nfp4_QH_warm_start",
        kind="quasilinear",
        source_scope="solver_ready_geometry_contract",
    )
    _write_gate(
        underresolved_gate,
        case="li383_low_res",
        kind="frequency",
        mboz=20,
        nboz=21,
    )

    wrong_scope_payload = mod.build_gradient_holdout_matrix(
        (
            ("QH", "frequency", good_gate),
            ("QH", "quasilinear", wrong_scope_gate),
        )
    )
    underresolved_payload = mod.build_gradient_holdout_matrix(
        (
            ("QH", "frequency", good_gate),
            ("Li383", "frequency", underresolved_gate),
        )
    )

    assert wrong_scope_payload["passed"] is False
    assert wrong_scope_payload["summary"]["all_gates_passed"] is True
    assert wrong_scope_payload["summary"]["all_mode21_source_scope"] is False
    assert underresolved_payload["passed"] is False
    assert underresolved_payload["summary"]["all_mboz_nboz_at_least_21"] is False


def test_gradient_holdout_matrix_tracks_estimator_as_reduced_claim(
    tmp_path: Path,
) -> None:
    estimator_gate = tmp_path / "estimator.json"
    _write_gate(
        estimator_gate,
        case="nfp4_QH_warm_start",
        kind="nonlinear-window estimator",
        extra_objectives=[
            {
                "objective": "nonlinear_window_heat_flux_mean",
                "passed": True,
                "rel_error": 2.5e-2,
                "abs_error": 1.0e-4,
            },
            {
                "objective": "nonlinear_window_heat_flux_cv",
                "passed": True,
                "rel_error": 1.5e-2,
                "abs_error": 1.0e-4,
            },
            {
                "objective": "nonlinear_window_heat_flux_trend",
                "passed": True,
                "rel_error": 2.0e-2,
                "abs_error": 1.0e-4,
            },
        ],
    )

    payload = mod.build_gradient_holdout_matrix(
        (("QH", "nonlinear-window estimator", estimator_gate),)
    )
    row = payload["rows"][0]

    assert payload["passed"] is True
    assert payload["claim_level"].endswith("not_production_nonlinear_optimization")
    assert payload["summary"]["gate_types"] == ["nonlinear-window estimator"]
    assert row["objectives"]["nonlinear_window_heat_flux_mean"] is True
    assert row["objectives"]["nonlinear_window_heat_flux_cv"] is True
    assert row["objectives"]["nonlinear_window_heat_flux_trend"] is True
    assert "optimized-equilibrium nonlinear transport" in payload["notes"]
