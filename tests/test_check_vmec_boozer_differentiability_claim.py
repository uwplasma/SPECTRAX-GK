from __future__ import annotations

import json
from pathlib import Path

from tools.release.check_vmec_boozer_differentiability_claim import (
    build_vmec_boozer_differentiability_claim_guard,
)


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _objectives_for_gate(gate_type: str) -> tuple[dict[str, bool], dict[str, float]]:
    if gate_type == "frequency":
        names = ("gamma", "omega")
        rel_error = 1.0e-3
    elif gate_type == "quasilinear":
        names = (
            "gamma",
            "omega",
            "kperp_eff2",
            "linear_heat_flux_weight",
            "mixing_length_heat_flux_proxy",
        )
        rel_error = 2.0e-3
    elif gate_type == "nonlinear-window estimator":
        names = (
            "gamma",
            "omega",
            "kperp_eff2",
            "linear_heat_flux_weight",
            "mixing_length_heat_flux_proxy",
            "nonlinear_window_heat_flux_mean",
            "nonlinear_window_heat_flux_cv",
            "nonlinear_window_heat_flux_trend",
        )
        rel_error = 2.5e-2
    else:
        raise AssertionError(f"unexpected gate_type: {gate_type}")
    return {name: True for name in names}, {name: rel_error for name in names}


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
                "n_cases": 3,
                "n_equal_arc_passed": 3,
            },
            "rows": [
                {
                    "case_name": "case_a",
                    "available": True,
                    "family": "quasi-helical",
                    "equal_arc_all_passed": True,
                    "mboz": 21,
                    "nboz": 21,
                    "mode_floor_passed": True,
                    "production_parity_passed": False,
                    "status": "diagnostic_open",
                },
                {
                    "case_name": "case_b",
                    "available": True,
                    "family": "quasi-isodynamic",
                    "equal_arc_all_passed": True,
                    "mboz": 21,
                    "nboz": 21,
                    "mode_floor_passed": True,
                    "production_parity_passed": False,
                    "status": "diagnostic_open",
                },
                {
                    "case_name": "shaped_tokamak_pressure",
                    "available": True,
                    "family": "axisymmetric finite-beta",
                    "equal_arc_all_passed": True,
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
            objectives, objective_rel_error = _objectives_for_gate(gate_type)
            gradient_rows.append(
                {
                    "case_name": case_name,
                    "gate_type": gate_type,
                    "max_rel_error": max(objective_rel_error.values()),
                    "mboz": 21,
                    "nboz": 21,
                    "objective_rel_error": objective_rel_error,
                    "objectives": objectives,
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
    _write_json(
        root,
        "docs/_static/vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json",
        {
            "case_name": "shaped_tokamak_pressure",
            "eigenpair_gate": {"max_rel_error": 1.0e-8},
            "kind": "mode21_vmec_boozer_linear_frequency_gradient_gate",
            "linear_frequency_gradient_gate": True,
            "linear_growth_gradient_gate": True,
            "mboz": 21,
            "nboz": 21,
            "nonlinear_window_gradient_gate": False,
            "objective_gates": [
                {
                    "objective": "gamma",
                    "passed": True,
                    "rel_error": 0.0,
                },
                {
                    "objective": "omega",
                    "passed": True,
                    "rel_error": 1.0e-8,
                },
            ],
            "passed": True,
            "quasilinear_weight_gradient_gate": False,
            "source_scope": "mode21_vmec_boozer_state",
            "surface_stencil_width": 3,
        },
    )
    _write_json(
        root,
        "docs/_static/vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json",
        {
            "case_name": "shaped_tokamak_pressure",
            "eigenpair_gate": {"max_rel_error": 2.0e-4},
            "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
            "linear_frequency_gradient_gate": True,
            "linear_growth_gradient_gate": True,
            "mboz": 21,
            "nboz": 21,
            "nonlinear_window_gradient_gate": False,
            "objective_gates": [
                {
                    "objective": "gamma",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "omega",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "kperp_eff2",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "linear_heat_flux_weight",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "mixing_length_heat_flux_proxy",
                    "passed": True,
                    "rel_error": 2.0e-4,
                },
            ],
            "passed": True,
            "quasilinear_weight_gradient_gate": True,
            "source_scope": "mode21_vmec_boozer_state",
            "surface_stencil_width": 3,
        },
    )
    _write_json(
        root,
        "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json",
        {
            "case_name": "shaped_tokamak_pressure",
            "eigenpair_gate": {"max_rel_error": 2.5e-4},
            "kind": "mode21_vmec_boozer_nonlinear_window_gradient_gate",
            "linear_frequency_gradient_gate": True,
            "linear_growth_gradient_gate": True,
            "mboz": 21,
            "nboz": 21,
            "nonlinear_window_gradient_gate": True,
            "objective_gates": [
                {
                    "objective": "gamma",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "omega",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "kperp_eff2",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "linear_heat_flux_weight",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "mixing_length_heat_flux_proxy",
                    "passed": True,
                    "rel_error": 2.0e-4,
                },
                {
                    "objective": "nonlinear_window_heat_flux_mean",
                    "passed": True,
                    "rel_error": 2.0e-4,
                },
                {
                    "objective": "nonlinear_window_heat_flux_cv",
                    "passed": True,
                    "rel_error": 2.0e-4,
                },
                {
                    "objective": "nonlinear_window_heat_flux_trend",
                    "passed": True,
                    "rel_error": 2.0e-4,
                },
            ],
            "passed": True,
            "production_nonlinear_window_gradient_gate": False,
            "quasilinear_weight_gradient_gate": True,
            "source_scope": "mode21_vmec_boozer_state",
            "surface_stencil_width": 3,
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
    assert report["checks"]["parity_matrix"]["finite_beta_pressure_equal_arc_rows"] == [
        "shaped_tokamak_pressure"
    ]
    assert report["checks"]["finite_beta_frequency_gate"]["case_name"] == (
        "shaped_tokamak_pressure"
    )
    assert report["checks"]["finite_beta_quasilinear_gate"]["case_name"] == (
        "shaped_tokamak_pressure"
    )
    assert report["checks"]["finite_beta_nonlinear_window_gate"]["case_name"] == (
        "shaped_tokamak_pressure"
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


def test_vmec_boozer_differentiability_claim_guard_requires_finite_beta_parity(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    parity_path = tmp_path / "docs/_static/vmec_boozer_parity_matrix.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["rows"] = [
        row for row in parity["rows"] if row["case_name"] != "shaped_tokamak_pressure"
    ]
    parity_path.write_text(json.dumps(parity), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "parity_matrix_missing_required_family" in report["blockers"]
    assert "parity_matrix_missing_finite_beta_pressure_row" in report["blockers"]


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


def test_vmec_boozer_differentiability_claim_guard_requires_finite_beta_frequency_gate(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gate_path = tmp_path / "docs/_static/vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["case_name"] = "nfp4_QH_warm_start"
    gate["quasilinear_weight_gradient_gate"] = True
    gate["objective_gates"][1]["rel_error"] = 0.2
    gate["eigenpair_gate"]["max_rel_error"] = 0.2
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "finite_beta_frequency_gate_wrong_case" in report["blockers"]
    assert "finite_beta_frequency_gate_error_threshold_failed" in report["blockers"]
    assert (
        "finite_beta_frequency_gate_attempts_transport_gradient_claim"
        in report["blockers"]
    )


def test_vmec_boozer_differentiability_claim_guard_requires_finite_beta_ql_gate(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gate_path = tmp_path / "docs/_static/vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["case_name"] = "nfp4_QH_warm_start"
    gate["nonlinear_window_gradient_gate"] = True
    gate["objective_gates"] = [
        row
        for row in gate["objective_gates"]
        if row["objective"] != "mixing_length_heat_flux_proxy"
    ]
    gate["eigenpair_gate"]["max_rel_error"] = 0.2
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "finite_beta_quasilinear_gate_wrong_case" in report["blockers"]
    assert "finite_beta_quasilinear_gate_missing_objective" in report["blockers"]
    assert "finite_beta_quasilinear_gate_error_threshold_failed" in report["blockers"]
    assert (
        "finite_beta_quasilinear_gate_attempts_nonlinear_gradient_claim"
        in report["blockers"]
    )


def test_vmec_boozer_differentiability_claim_guard_requires_finite_beta_nonlinear_window_gate(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gate_path = (
        tmp_path
        / "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["case_name"] = "nfp4_QH_warm_start"
    gate["production_nonlinear_window_gradient_gate"] = True
    gate["objective_gates"] = [
        row
        for row in gate["objective_gates"]
        if row["objective"] != "nonlinear_window_heat_flux_trend"
    ]
    gate["eigenpair_gate"]["max_rel_error"] = 0.2
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "finite_beta_nonlinear_window_gate_wrong_case" in report["blockers"]
    assert (
        "finite_beta_nonlinear_window_gate_missing_objective" in report["blockers"]
    )
    assert (
        "finite_beta_nonlinear_window_gate_error_threshold_failed"
        in report["blockers"]
    )
    assert (
        "finite_beta_nonlinear_window_gate_attempts_production_transport_claim"
        in report["blockers"]
    )


def test_vmec_boozer_differentiability_claim_guard_requires_finite_beta_nonlinear_window_artifact(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gate_path = (
        tmp_path
        / "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json"
    )
    gate_path.unlink()

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "finite_beta_nonlinear_window_gate_unreadable" in report["blockers"]


def test_vmec_boozer_differentiability_claim_guard_blocks_malformed_finite_beta_nonlinear_window_gate(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gate_path = (
        tmp_path
        / "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["kind"] = "mode21_vmec_boozer_quasilinear_gradient_gate"
    gate["source_scope"] = "reduced_fixture"
    gate["mboz"] = 19
    gate["nboz"] = 21
    gate["passed"] = False
    gate["nonlinear_window_gradient_gate"] = False
    gate["objective_gates"][0]["passed"] = False
    gate["eigenpair_gate"].pop("max_rel_error")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "finite_beta_nonlinear_window_gate_failed" in report["blockers"]
    assert "finite_beta_nonlinear_window_gate_wrong_kind" in report["blockers"]
    assert "finite_beta_nonlinear_window_gate_wrong_source_scope" in report["blockers"]
    assert "finite_beta_nonlinear_window_gate_mode_floor_failed" in report["blockers"]
    assert "finite_beta_nonlinear_window_gate_objective_failed" in report["blockers"]
    assert (
        "finite_beta_nonlinear_window_gate_error_threshold_failed"
        in report["blockers"]
    )


def test_vmec_boozer_differentiability_claim_guard_requires_ql_objectives(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gradient_path = tmp_path / "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
    gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
    ql_row = next(row for row in gradient["rows"] if row["gate_type"] == "quasilinear")
    ql_row["objectives"].pop("mixing_length_heat_flux_proxy")
    ql_row["objective_rel_error"].pop("mixing_length_heat_flux_proxy")
    gradient_path.write_text(json.dumps(gradient), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "gradient_holdout_missing_required_objective" in report["blockers"]


def test_vmec_boozer_differentiability_claim_guard_rejects_large_objective_error(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gradient_path = tmp_path / "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
    gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
    estimator_row = next(
        row for row in gradient["rows"] if row["gate_type"] == "nonlinear-window estimator"
    )
    estimator_row["max_rel_error"] = 0.2
    estimator_row["objective_rel_error"]["nonlinear_window_heat_flux_mean"] = 0.2
    gradient_path.write_text(json.dumps(gradient), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "gradient_holdout_error_threshold_failed" in report["blockers"]
