from __future__ import annotations

import json
from pathlib import Path

from tools.check_vmec_boozer_differentiability_claim import (
    build_vmec_boozer_differentiability_claim_guard,
)


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _minimal_artifacts(root: Path) -> None:
    _write_json(
        root,
        "docs/_static/vmec_boozer_parity_matrix.json",
        {
            "claim_level": "multi_equilibrium_zero_beta_equal_arc_parity_gate_not_full_transport_gradient_claim",
            "minimum_boozer_mode_count": 21,
            "summary": {
                "all_available": True,
                "all_equal_arc_passed": True,
                "n_cases": 2,
                "n_equal_arc_passed": 2,
            },
            "rows": [
                {
                    "case_name": "case_a",
                    "mboz": 21,
                    "nboz": 21,
                    "mode_floor_passed": True,
                    "production_parity_passed": False,
                    "status": "diagnostic_open",
                },
                {
                    "case_name": "case_b",
                    "mboz": 21,
                    "nboz": 21,
                    "mode_floor_passed": True,
                    "production_parity_passed": False,
                    "status": "diagnostic_open",
                },
            ],
        },
    )
    gradient_rows = []
    for case_name in ("case_a", "case_b"):
        for gate_type in ("frequency", "quasilinear", "nonlinear-window estimator"):
            gradient_rows.append(
                {
                    "case_name": case_name,
                    "gate_type": gate_type,
                    "mboz": 21,
                    "nboz": 21,
                    "passed": True,
                    "source_scope": "mode21_vmec_boozer_state",
                    "path": f"{case_name}_{gate_type}.json",
                }
            )
    _write_json(
        root,
        "docs/_static/vmec_boozer_gradient_holdout_matrix.json",
        {
            "passed": True,
            "claim_level": (
                "multi_equilibrium_reduced_linear_quasilinear_and_nonlinear_window_"
                "estimator_gradient_gate_not_production_nonlinear_optimization"
            ),
            "summary": {
                "all_gates_passed": True,
                "all_mboz_nboz_at_least_21": True,
                "all_mode21_source_scope": True,
            },
            "rows": gradient_rows,
        },
    )
    _write_json(
        root,
        "docs/_static/differentiable_geometry_bridge.json",
        {
            "vmec_jax_flux_tube_array_parity": {
                "production_parity_passed": False,
                "status": "diagnostic_open",
                "interpretation": "Direct tensor path is diagnostic; equal-arc Boozer path carries the claim.",
                "equal_arc_core_passed": True,
                "equal_arc_derivative_passed": True,
                "equal_arc_metric_passed": True,
                "equal_arc_drift_passed": True,
            },
            "vmec_jax_boozer_flux_tube": {"available": True},
            "booz_xform_flux_tube": {"available": True},
        },
    )
    _write_json(
        root,
        "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json",
        {
            "passed": True,
            "claim_level": "vmec_boozer_geometry_perturbed_startup_plumbing_fd_audit_not_transport_average",
            "vmec_boozer_startup_nonlinear_plumbing_fd_path_gate": True,
            "transport_average_gate": False,
            "production_nonlinear_window_gradient_gate": False,
            "vmec_boozer_production_nonlinear_observable_fd_path_gate": False,
        },
    )


def test_vmec_boozer_differentiability_claim_guard_accepts_scoped_artifacts(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is True
    assert not report["blockers"]
    assert (
        report["checks"]["differentiable_geometry_bridge"][
            "direct_tensor_gap_explicitly_scoped"
        ]
        is True
    )
    assert (
        report["checks"]["nonlinear_fd_audit"]["production_nonlinear_window_gradient_gate"]
        is False
    )


def test_vmec_boozer_differentiability_claim_guard_rejects_hidden_direct_gap(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    bridge_path = tmp_path / "docs/_static/differentiable_geometry_bridge.json"
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    bridge["vmec_jax_flux_tube_array_parity"]["status"] = "failed"
    bridge["vmec_jax_flux_tube_array_parity"]["interpretation"] = ""
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "direct_tensor_parity_gap_not_explicitly_scoped" in report["blockers"]


def test_vmec_boozer_differentiability_claim_guard_rejects_unscoped_nonlinear_claim(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    audit_path = tmp_path / "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["production_nonlinear_window_gradient_gate"] = True
    audit["transport_average_gate"] = True
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "startup_fd_audit_attempts_production_nonlinear_claim" in report["blockers"]


def test_vmec_boozer_differentiability_claim_guard_requires_mode21_gradient_scope(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gradient_path = tmp_path / "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
    gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
    gradient["rows"][0]["source_scope"] = "reduced_fixture"
    gradient_path.write_text(json.dumps(gradient), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "gradient_holdout_wrong_source_scope" in report["blockers"]
