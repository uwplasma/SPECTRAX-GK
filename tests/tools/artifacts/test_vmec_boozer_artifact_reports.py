"""Tests for VMEC/Boozer artifact reports and promotion gates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from support.paths import load_artifact_tool

from tools.artifacts import build_vmec_boozer_aggregate_holdout_gate as holdout_gate
from tools.artifacts import (
    build_vmec_boozer_aggregate_line_search_comparison as comparison_gate,
)
from tools.artifacts import build_vmec_boozer_aggregate_objective_gate as objective_gate
from tools.artifacts import (
    build_vmec_boozer_multi_point_objective_gate as multi_point_gate,
)
from tools.artifacts import (
    build_vmec_boozer_second_equilibrium_aggregate_gate as second_gate,
)



def _write_gradient_gate(
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
    mod = load_artifact_tool("build_vmec_boozer_gradient_holdout_matrix")
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
        _write_gradient_gate(path, case=case, kind=kind)

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
    mod = load_artifact_tool("build_vmec_boozer_gradient_holdout_matrix")
    gate = tmp_path / "gate.json"
    _write_gradient_gate(gate, case="case", kind="frequency")
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
    mod = load_artifact_tool("build_vmec_boozer_gradient_holdout_matrix")
    good_gate = tmp_path / "good.json"
    wrong_scope_gate = tmp_path / "wrong_scope.json"
    underresolved_gate = tmp_path / "underresolved.json"
    _write_gradient_gate(good_gate, case="nfp4_QH_warm_start", kind="frequency")
    _write_gradient_gate(
        wrong_scope_gate,
        case="nfp4_QH_warm_start",
        kind="quasilinear",
        source_scope="solver_ready_geometry_contract",
    )
    _write_gradient_gate(
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
    mod = load_artifact_tool("build_vmec_boozer_gradient_holdout_matrix")
    estimator_gate = tmp_path / "estimator.json"
    _write_gradient_gate(
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


def _fake_fd_geom(scale: float = 1.0):
    theta = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    ones = np.ones_like(theta)
    return SimpleNamespace(
        theta=theta,
        gradpar=lambda: 0.7 * scale,
        bmag_profile=scale * (1.0 + 0.05 * np.cos(theta)),
        bgrad_profile=scale * 0.05 * np.sin(theta),
        gds2_profile=scale * ones,
        gds21_profile=0.02 * scale * np.sin(theta),
        gds22_profile=scale * (1.0 + 0.03 * np.cos(theta)),
        cv_profile=0.1 * scale * np.cos(theta),
        gb_profile=0.1 * scale * np.cos(theta),
        cv0_profile=np.zeros_like(theta),
        gb0_profile=np.zeros_like(theta),
        jacobian_profile=ones / scale,
        grho_profile=scale * ones,
        q=1.4 * scale,
        s_hat=0.8 * scale,
        epsilon=0.18,
        R0=2.7,
        alpha=0.0,
        kxfac=1.0,
        theta_scale=1.0,
        nfp=4,
    )


def _synthetic_fd_run(label: str, perturbation: float, scale: float = 1.0) -> dict:
    mod = load_artifact_tool("build_vmec_boozer_nonlinear_window_fd_audit")
    time = np.linspace(0.0, 1.0, 8)
    heat = scale * (1.0 + 0.08 * time)
    return {
        "label": label,
        "perturbation": perturbation,
        "geometry_file_name": f"{label}.nc",
        "geometry_response": mod.geometry_response_metrics(
            _fake_fd_geom(), _fake_fd_geom(1.0 + abs(perturbation))
        ),
        "time": time.tolist(),
        "heat_flux": heat.tolist(),
        "window": mod.late_window_metrics(time, heat, tail_fraction=0.5),
    }


def test_geometry_response_metrics_reports_profile_and_scalar_changes() -> None:
    mod = load_artifact_tool("build_vmec_boozer_nonlinear_window_fd_audit")

    metrics = mod.geometry_response_metrics(_fake_fd_geom(), _fake_fd_geom(1.01))

    assert metrics["max_relative_change"] > 0.0
    assert metrics["per_profile"]["bmag"] > 0.0
    assert metrics["per_scalar"]["q"] > 0.0


def test_build_vmec_boozer_audit_payload_passes_conditioned_response() -> None:
    mod = load_artifact_tool("build_vmec_boozer_nonlinear_window_fd_audit")
    runs = [
        _synthetic_fd_run("minus", -1.0e-5, 0.80),
        _synthetic_fd_run("base", 0.0, 1.00),
        _synthetic_fd_run("plus", 1.0e-5, 1.25),
        _synthetic_fd_run("base_repeat", 0.0, 1.00),
    ]

    payload = mod.build_vmec_boozer_audit_payload(
        runs,
        case_name="nfp4_QH_warm_start",
        parameter_name="Rcos_r1_m1",
        perturbation_step=1.0e-5,
        tail_fraction=0.5,
        repeatability_rtol=1.0e-12,
        max_window_cv=0.1,
        max_window_trend=0.1,
        min_response_fraction=0.03,
        min_geometry_relative_change=1.0e-7,
    )

    assert payload["passed"] is True
    assert payload["vmec_boozer_startup_nonlinear_plumbing_fd_path_gate"] is True
    assert payload["transport_average_gate"] is False
    assert payload["vmec_boozer_production_nonlinear_observable_fd_path_gate"] is False
    assert payload["production_nonlinear_window_gradient_gate"] is False
    assert payload["gates"]["geometry_perturbation_resolved"] is True
    assert payload["metrics"]["central_fd_dq_dparameter"] > 0.0


def test_build_vmec_boozer_audit_payload_blocks_unresolved_geometry() -> None:
    mod = load_artifact_tool("build_vmec_boozer_nonlinear_window_fd_audit")
    runs = [
        _synthetic_fd_run("minus", -1.0e-10, 0.80),
        _synthetic_fd_run("base", 0.0, 1.00),
        _synthetic_fd_run("plus", 1.0e-10, 1.25),
        _synthetic_fd_run("base_repeat", 0.0, 1.00),
    ]

    payload = mod.build_vmec_boozer_audit_payload(
        runs,
        case_name="nfp4_QH_warm_start",
        parameter_name="Rcos_r1_m1",
        perturbation_step=1.0e-10,
        tail_fraction=0.5,
        repeatability_rtol=1.0e-12,
        max_window_cv=0.1,
        max_window_trend=0.1,
        min_response_fraction=0.03,
        min_geometry_relative_change=1.0e-7,
    )

    assert payload["passed"] is False
    assert payload["gates"]["geometry_perturbation_resolved"] is False


def test_fd_audit_main_writes_artifacts_without_running_solver(
    monkeypatch, tmp_path: Path
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_nonlinear_window_fd_audit")

    monkeypatch.setattr(
        mod,
        "_mode21_vmec_boozer_linear_context",
        lambda **_kwargs: {
            "parameter_names": ("Rcos_r1_m1",),
            "geometry_for": lambda _x: _fake_fd_geom(),
        },
    )

    def fake_run_vmec_boozer_window(*, label: str, perturbation: float, **_kwargs):
        scale = {"minus": 0.80, "base": 1.00, "plus": 1.25, "base_repeat": 1.00}[label]
        return _synthetic_fd_run(label, perturbation, scale)

    monkeypatch.setattr(mod, "run_vmec_boozer_window", fake_run_vmec_boozer_window)
    out = tmp_path / "audit.png"

    assert mod.main(["--out", str(out), "--tail-fraction", "0.5"]) == 0

    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".csv").exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["passed"] is True
    assert (
        meta["claim_level"]
        == "vmec_boozer_geometry_perturbed_startup_plumbing_fd_audit_not_transport_average"
    )
    assert meta["transport_average_gate"] is False


def _fake_nonlinear_gradient_payload() -> dict[str, object]:
    objective_names = [
        "gamma",
        "nonlinear_window_heat_flux_mean",
        "nonlinear_window_heat_flux_cv",
        "nonlinear_window_heat_flux_trend",
    ]
    return {
        "kind": "mode21_vmec_boozer_nonlinear_window_gradient_gate",
        "passed": True,
        "source_scope": "mode21_vmec_boozer_state",
        "parameter_names": ["Rcos_mid_surface_m1"],
        "objective_names": objective_names,
        "objective_gates": [
            {
                "objective": objective,
                "parameter": "Rcos_mid_surface_m1",
                "implicit": float(index + 1),
                "finite_difference": float(index + 1),
                "abs_error": 0.0,
                "rel_error": 0.0,
                "passed": True,
            }
            for index, objective in enumerate(objective_names)
        ],
        "eigenpair_gate": {
            "atol": 1.0e-6,
            "jacobian_implicit": [[1.0], [2.0], [3.0], [4.0]],
            "jacobian_fd": [[1.0], [2.0], [3.0], [4.0]],
        },
    }


def test_nonlinear_window_builder_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    mod = load_artifact_tool("build_vmec_boozer_gradient_gate")
    monkeypatch.setattr(
        mod,
        "mode21_vmec_boozer_nonlinear_window_gradient_report",
        lambda **_kwargs: _fake_nonlinear_gradient_payload(),
    )

    out = tmp_path / "vmec_boozer_nonlinear_window_gradient_gate.png"
    assert (
        mod.main(
            ["nonlinear-window", "--out", str(out), "--surface-stencil-width", "3"]
        )
        == 0
    )

    for suffix in (".png", ".pdf", ".json", ".csv"):
        assert out.with_suffix(suffix).exists()
    saved = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["kind"] == "mode21_vmec_boozer_nonlinear_window_gradient_gate"


def test_nonlinear_window_builder_json_only(capsys, monkeypatch) -> None:
    mod = load_artifact_tool("build_vmec_boozer_gradient_gate")
    monkeypatch.setattr(
        mod,
        "mode21_vmec_boozer_nonlinear_window_gradient_report",
        lambda **_kwargs: _fake_nonlinear_gradient_payload(),
    )

    assert (
        mod.main(["nonlinear-window", "--json-only", "--nonlinear-steps", "12"])
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["objective_names"][-1] == "nonlinear_window_heat_flux_trend"


def test_vmec_boozer_gradient_builder_frequency_and_quasilinear_json_only(
    capsys, monkeypatch
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_gradient_gate")
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_frequency(**kwargs: object) -> dict[str, object]:
        calls.append(("frequency", dict(kwargs)))
        return {
            "kind": "mode21_vmec_boozer_linear_frequency_gradient_gate",
            "passed": True,
            "objective_names": ["gamma", "omega"],
            "objective_gates": [],
        }

    def fake_quasilinear(**kwargs: object) -> dict[str, object]:
        calls.append(("quasilinear", dict(kwargs)))
        return {
            "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
            "passed": True,
            "objective_names": ["gamma", "omega", "ql_heat_flux"],
            "objective_gates": [],
        }

    monkeypatch.setattr(
        mod, "mode21_vmec_boozer_linear_frequency_gradient_report", fake_frequency
    )
    monkeypatch.setattr(
        mod, "mode21_vmec_boozer_quasilinear_gradient_report", fake_quasilinear
    )
    assert mod.main(["frequency", "--json-only", "--mboz", "21", "--nboz", "21"]) == 0
    frequency_payload = json.loads(capsys.readouterr().out)
    assert (
        frequency_payload["kind"]
        == "mode21_vmec_boozer_linear_frequency_gradient_gate"
    )

    assert mod.main(["quasilinear", "--json-only", "--fd-step", "2e-6"]) == 0
    quasilinear_payload = json.loads(capsys.readouterr().out)
    assert (
        quasilinear_payload["kind"] == "mode21_vmec_boozer_quasilinear_gradient_gate"
    )

    assert [name for name, _ in calls] == ["frequency", "quasilinear"]
    assert calls[0][1]["mboz"] == 21
    assert calls[1][1]["fd_step"] == 2.0e-6


def _fake_parity_report(**kwargs: object) -> dict[str, object]:
    assert int(kwargs["mboz"]) >= 21
    assert int(kwargs["nboz"]) >= 21
    return {
        "available": True,
        "case_name": kwargs["case_name"],
        "status": "diagnostic_open",
        "mboz": kwargs["mboz"],
        "nboz": kwargs["nboz"],
        "equal_arc_core_worst_normalized_max_abs": 4.0e-3,
        "equal_arc_core_worst_scalar_rel": 2.0e-3,
        "equal_arc_derivative_worst_normalized_max_abs": 2.0e-2,
        "equal_arc_metric_worst_normalized_max_abs": 3.0e-2,
        "equal_arc_drift_worst_normalized_max_abs": 7.0e-2,
        "equal_arc_core_tolerance": 1.0e-2,
        "equal_arc_derivative_tolerance": 3.0e-2,
        "equal_arc_metric_tolerance": 8.0e-2,
        "equal_arc_drift_tolerance": 8.0e-2,
        "equal_arc_core_passed": True,
        "equal_arc_derivative_passed": True,
        "equal_arc_metric_passed": True,
        "equal_arc_drift_passed": True,
        "production_parity_passed": False,
        "worst_core_normalized_max_abs": 2.0e-1,
        "worst_scalar_rel": 1.0e-3,
        "source_model": "vmec_jax:state->tensor-flux-tube vs imported-vmec-eik",
        "surface_index": 4,
        "torflux": 0.5,
        "alpha": 0.0,
    }


def _fake_parity_artifact_resolver(
    case_name: str,
) -> tuple[str | None, str | None, str | None]:
    if case_name in {"nfp1_QI", "nfp2_QI", "nfp4_QI_finite_beta"}:
        return f"/tmp/input.{case_name}", None, None
    return f"/tmp/input.{case_name}", "/dev/null", None


def test_build_parity_matrix_uses_mode21_floor_and_summarizes_rows() -> None:
    mod = load_artifact_tool("build_vmec_boozer_parity_matrix")
    cases = (
        mod.ParityCase("nfp4_QH_warm_start", "QH", "stellarator", 16),
        mod.ParityCase("nfp3_QI_fixed_resolution_final", "QI", "stellarator", 8),
    )

    payload = mod.build_parity_matrix(
        cases=cases,
        reporter=_fake_parity_report,
        artifact_resolver=_fake_parity_artifact_resolver,
    )

    assert payload["kind"] == "vmec_boozer_parity_matrix"
    assert payload["minimum_boozer_mode_count"] == 21
    assert payload["summary"]["n_cases"] == 2
    assert payload["summary"]["n_equal_arc_passed"] == 2
    assert payload["summary"]["all_equal_arc_passed"] is True
    assert all(row["mode_floor_passed"] for row in payload["rows"])
    assert payload["rows"][0][
        "equal_arc_drift_worst_normalized_max_abs"
    ] == pytest.approx(7.0e-2)
    assert payload["claim_level"].endswith("not_full_transport_gradient_claim")
    assert payload["rows"][0]["production_parity_passed"] is False
    assert payload["rows"][0]["sample_set_id"] == (
        "nfp4_QH_warm_start:ntheta=16:mboz=21:nboz=21"
    )
    provenance = payload["sample_set_provenance"]
    assert provenance["bounded_run"] is True
    assert provenance["external_vmec_solves_launched"] is False
    assert provenance["summary"]["n_total_sample_sets"] == 7
    assert provenance["summary"]["n_unique_sample_sets"] == 6
    assert provenance["summary"]["all_modes_at_or_above_floor"] is True
    assert (
        provenance["matrix_cases"][0]["sample_set_id"]
        == payload["rows"][0]["sample_set_id"]
    )
    assert provenance["matrix_cases"][0]["field_line_alpha"] == pytest.approx(0.0)
    assert provenance["matrix_cases"][0]["surface_index"] == 4
    assert provenance["matrix_cases"][0]["torflux"] == pytest.approx(0.5)
    qi_summary = payload["qi_seed_robustness"]["summary"]
    assert qi_summary["n_variants"] == 5
    assert qi_summary["n_passed"] == 2
    assert qi_summary["n_rejected"] == 3
    assert qi_summary["seed_robust_gate_passed"] is True
    assert qi_summary["full_declared_seed_campaign_passed"] is False
    assert qi_summary["evaluated_reference_gate_passed"] is True
    assert qi_summary["robustness_status"] == "artifact_limited_passed"
    assert qi_summary["artifact_reason_counts"]["missing_bundled_wout_reference"] == 3


def test_build_parity_matrix_rejects_underresolved_boozer_modes() -> None:
    mod = load_artifact_tool("build_vmec_boozer_parity_matrix")
    cases = (
        mod.ParityCase(
            "nfp3_QI_fixed_resolution_final", "QI", "stellarator", 8, mboz=20, nboz=21
        ),
    )

    with pytest.raises(ValueError, match="mboz and nboz"):
        mod.build_parity_matrix(
            cases=cases,
            reporter=_fake_parity_report,
            artifact_resolver=_fake_parity_artifact_resolver,
        )


def test_qi_seed_robustness_records_failed_mode21_variant() -> None:
    mod = load_artifact_tool("build_vmec_boozer_parity_matrix")

    def failing_report(**kwargs: object) -> dict[str, object]:
        report = _fake_parity_report(**kwargs)
        report["equal_arc_drift_worst_normalized_max_abs"] = 9.0e-2
        report["equal_arc_drift_passed"] = False
        return report

    payload = mod.build_parity_matrix(
        cases=(),
        qi_variants=(
            mod.ParityCase(
                "nfp3_QI_fixed_resolution_final",
                "QI",
                "quasi-isodynamic accepted reference",
                8,
            ),
        ),
        reporter=failing_report,
        artifact_resolver=_fake_parity_artifact_resolver,
    )

    qi = payload["qi_seed_robustness"]
    assert qi["summary"]["n_failed"] == 1
    assert qi["summary"]["seed_robust_gate_passed"] is False
    row = qi["rows"][0]
    assert row["qi_gate_status"] == "fragile_open"
    assert row["artifact_reason"] == "mode21_qi_tolerance_exceeded"
    assert row["equal_arc_drift_worst_normalized_max_abs"] == pytest.approx(9.0e-2)


def test_qi_seed_robustness_rejects_input_only_variants() -> None:
    mod = load_artifact_tool("build_vmec_boozer_parity_matrix")
    payload = mod.build_parity_matrix(
        cases=(),
        qi_variants=(
            mod.ParityCase(
                "nfp1_QI", "QI input nfp1", "quasi-isodynamic input variant", 8
            ),
        ),
        reporter=_fake_parity_report,
        artifact_resolver=_fake_parity_artifact_resolver,
    )

    qi = payload["qi_seed_robustness"]
    assert qi["summary"]["n_rejected"] == 1
    assert qi["summary"]["seed_robust_gate_passed"] is False
    row = qi["rows"][0]
    assert row["sample_set_id"] == "nfp1_QI:ntheta=8:mboz=21:nboz=21"
    assert row["qi_gate_status"] == "artifact_rejected"
    assert row["artifact_reason"] == "missing_bundled_wout_reference"
    assert "does not launch VMEC solves" in row["rejection_reason"]


def test_write_parity_matrix_artifacts_writes_companions(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_vmec_boozer_parity_matrix")
    payload = mod.build_parity_matrix(
        cases=(
            mod.ParityCase("shaped_tokamak_pressure", "tokamak", "axisymmetric", 8),
        ),
        reporter=_fake_parity_report,
        artifact_resolver=_fake_parity_artifact_resolver,
    )

    paths = mod.write_parity_matrix_artifacts(payload, out=tmp_path / "parity.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "parity.json").read_text(encoding="utf-8"))
    assert saved["summary"]["n_equal_arc_passed"] == 1
    assert saved["sample_set_provenance"]["summary"]["n_total_sample_sets"] == 6
    csv_text = (tmp_path / "parity.csv").read_text(encoding="utf-8")
    assert "sample_set_id" in csv_text
    assert "shaped_tokamak_pressure:ntheta=8:mboz=21:nboz=21" in csv_text
    assert "missing_bundled_wout_reference" in csv_text


def _write_holdout_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _holdout_manifest(path: Path) -> Path:
    return _write_holdout_json(
        path,
        {
            "kind": "external_vmec_holdout_config_manifest",
            "case": "qh_holdout",
            "transport_sample": {
                "vmec_file": "/tmp/wout_qh.nc",
                "torflux": 0.78,
                "alpha": 1.2,
                "ky": 0.2,
                "npol": 1.0,
            },
        },
    )


def _holdout_ensemble(path: Path, *, passed: bool) -> Path:
    return _write_holdout_json(
        path,
        {
            "kind": "nonlinear_window_ensemble_report",
            "case": "qh_holdout_replicated_window",
            "passed": passed,
            "gate_report": {"passed": passed},
            "window": {"tmin": 350.0, "tmax": 700.0},
            "statistics": {"ensemble_mean": 1.25, "combined_sem": 0.05},
        },
    )


def test_vmec_boozer_production_holdout_artifact_promotes_only_passed_ensemble(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_aggregate_holdout_gate")
    artifact = mod.build_vmec_boozer_production_holdout_artifact(
        transport_manifest=_holdout_manifest(tmp_path / "run_manifest.json"),
        ensemble_json=_holdout_ensemble(tmp_path / "ensemble.json", passed=True),
    )

    assert artifact["passed"] is True
    assert artifact["transport_average_gate"] is True
    assert artifact["promotion_gate"]["blockers"] == []
    assert (
        artifact["claim_level"]
        == "production_scope_vmec_boozer_heldout_nonlinear_transport_average"
    )
    sample = artifact["samples"][0]
    assert sample["surface"] == 0.78
    assert sample["torflux"] == 0.78
    assert sample["alpha"] == 1.2
    assert sample["selected_ky_index"] == "ky=0.2"
    assert artifact["holdout_samples"] == artifact["samples"]


def test_vmec_boozer_production_holdout_artifact_fails_closed_for_failed_ensemble(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_aggregate_holdout_gate")
    artifact = mod.build_vmec_boozer_production_holdout_artifact(
        transport_manifest=_holdout_manifest(tmp_path / "run_manifest.json"),
        ensemble_json=_holdout_ensemble(tmp_path / "ensemble.json", passed=False),
        case="explicit_case",
    )

    assert artifact["case"] == "explicit_case"
    assert artifact["passed"] is False
    assert artifact["transport_average_gate"] is False
    assert artifact["promotion_gate"]["blockers"] == [
        "replicated_nonlinear_window_ensemble_failed"
    ]


def test_vmec_boozer_production_holdout_artifact_main_writes_output(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_aggregate_holdout_gate")
    out = tmp_path / "holdout.json"
    result = mod.main(
        [
            "production",
            "--transport-manifest",
            str(_holdout_manifest(tmp_path / "run_manifest.json")),
            "--ensemble-json",
            str(_holdout_ensemble(tmp_path / "ensemble.json", passed=True)),
            "--out",
            str(out),
        ]
    )

    assert result == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert (
        saved["kind"]
        == "vmec_boozer_production_scope_heldout_nonlinear_transport_artifact"
    )
    assert saved["passed"] is True


# VMEC/Boozer aggregate artifact assertions
alpha_gate = holdout_gate
surface_gate = holdout_gate


def _assert_artifacts(paths: dict[str, str], *, json_key: str, csv_token: str) -> dict:
    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert json_key in payload
    assert csv_token in Path(paths["csv"]).read_text(encoding="utf-8")
    return payload


def _alpha_holdout_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_line_search_holdout_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "initial_delta": 0.0,
        "final_delta": -1.0e-8,
        "training_passed": True,
        "heldout_passed": True,
        "training_initial_objective": 0.9,
        "training_final_objective": 0.88,
        "training_relative_reduction": 0.0222222222,
        "heldout_initial_objective": 0.91,
        "heldout_final_objective": 0.909,
        "heldout_relative_reduction": 0.0010989011,
        "training_samples": [
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1},
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2},
        ],
        "heldout_samples": [
            {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1},
            {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2},
        ],
    }


def _surface_holdout_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_line_search_holdout_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "initial_delta": 0.0,
        "final_delta": 1.0e-8,
        "training_passed": True,
        "heldout_passed": True,
        "training_initial_objective": 0.8045688627,
        "training_final_objective": 0.8035154075,
        "training_relative_reduction": 0.0013093413,
        "heldout_initial_objective": 0.7205574540,
        "heldout_final_objective": 0.7202268146,
        "heldout_relative_reduction": 0.0004588662,
        "training_samples": [
            {"surface_index": 18, "alpha": 0.0, "selected_ky_index": 1},
            {"surface_index": 18, "alpha": 0.0, "selected_ky_index": 2},
        ],
        "heldout_samples": [
            {"surface_index": 19, "alpha": 0.0, "selected_ky_index": 1},
            {"surface_index": 19, "alpha": 0.0, "selected_ky_index": 2},
        ],
    }


def _objective_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "n_samples": 2,
        "base_value": 0.9,
        "minus_value": 0.85,
        "plus_value": 0.95,
        "central_derivative": 5.0,
        "response_abs": 0.1,
        "curvature_ratio": 0.02,
        "samples": [
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 1,
                "weight": 0.5,
            },
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 2,
                "weight": 0.5,
            },
        ],
        "minus_sample_values": [0.7, 1.0],
        "base_sample_values": [0.8, 1.0],
        "plus_sample_values": [0.9, 1.0],
    }


def _multi_point_payload() -> dict[str, object]:
    samples = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2, "weight": 0.25},
    ]
    payload = _objective_payload()
    payload.update(
        n_samples=len(samples),
        samples=samples,
        minus_sample_values=[0.70, 0.95, 0.75, 1.00],
        base_sample_values=[0.80, 1.00, 0.85, 0.95],
        plus_sample_values=[0.90, 1.05, 0.95, 0.90],
    )
    return payload


def _line_search_payload(
    objective: str = "quasilinear_flux",
    derivative: float = 5.0,
    initial: float = 0.90,
    final: float = 0.88,
) -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_line_search_report",
        "passed": True,
        "objective": objective,
        "reduction": "mean",
        "n_samples": 2,
        "accepted_steps": 1,
        "max_steps": 1,
        "initial_delta": 0.0,
        "final_delta": -1.0e-8 if derivative > 0.0 else 1.0e-8,
        "initial_objective": initial,
        "final_objective": final,
        "relative_reduction": (initial - final) / initial,
        "stop_reason": "max_steps",
        "samples": [
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 1,
                "weight": 0.5,
            },
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 2,
                "weight": 0.5,
            },
        ],
        "history": [
            {
                "step": 0,
                "delta": 0.0,
                "objective": initial,
                "central_derivative": derivative,
                "finite_difference_passed": True,
                "curvature_ratio": 0.02,
                "accepted": True,
                "candidate_delta": -1.0e-8 if derivative > 0.0 else 1.0e-8,
                "candidate_objective": final,
            }
        ],
    }


def _comparison_payload() -> dict[str, object]:
    reports = {
        "growth": _line_search_payload("growth", 2.0, 0.30, 0.29),
        "quasilinear_flux": _line_search_payload("quasilinear_flux", 5.0, 0.90, 0.88),
    }
    return {
        "kind": "vmec_boozer_aggregate_line_search_comparison",
        "passed": True,
        "case_name": "nfp4_QH_warm_start",
        "objectives": ["growth", "quasilinear_flux"],
        "reduction": "mean",
        "n_samples": 2,
        "same_sample_set": True,
        "all_line_searches_passed": True,
        "same_initial_update_direction": True,
        "final_delta_spread": 0.0,
        "relative_reduction_spread": 0.011,
        "rows": [
            {
                "objective": "growth",
                "passed": True,
                "n_samples": 2,
                "initial_objective": 0.30,
                "final_objective": 0.29,
                "absolute_reduction": 0.01,
                "relative_reduction": 0.0333333333,
                "initial_central_derivative": 2.0,
                "initial_update_direction": "negative_delta",
                "accepted_steps": 1,
                "max_steps": 1,
                "initial_delta": 0.0,
                "final_delta": -1.0e-8,
                "stop_reason": "max_steps",
            },
            {
                "objective": "quasilinear_flux",
                "passed": True,
                "n_samples": 2,
                "initial_objective": 0.90,
                "final_objective": 0.88,
                "absolute_reduction": 0.02,
                "relative_reduction": 0.0222222222,
                "initial_central_derivative": 5.0,
                "initial_update_direction": "negative_delta",
                "accepted_steps": 1,
                "max_steps": 1,
                "initial_delta": 0.0,
                "final_delta": -1.0e-8,
                "stop_reason": "max_steps",
            },
        ],
        "reports": reports,
    }


def _second_fd_payload() -> dict[str, object]:
    payload = _objective_payload()
    payload.update(
        case_name="li383_low_res",
        minus_value=9.80,
        base_value=9.79,
        plus_value=9.78,
        central_derivative=-1.0e5,
        response_abs=0.02,
        curvature_ratio=0.01,
    )
    return payload


def _second_line_payload() -> dict[str, object]:
    payload = _line_search_payload("quasilinear_flux", -1.0e5, 9.79, 9.78)
    payload.update(case_name="li383_low_res")
    return payload


def test_alpha_holdout_payload_uses_default_split(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _alpha_holdout_payload()

    monkeypatch.setattr(
        alpha_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = alpha_gate.build_vmec_boozer_aggregate_alpha_holdout_payload(
        ntheta=4, mboz=21, nboz=21
    )

    assert payload["artifact_kind"] == "vmec_boozer_aggregate_alpha_holdout_gate"
    assert payload["passed"] is True
    assert payload["holdout_split"]["training_alphas"] == [0.0]
    assert payload["holdout_split"]["holdout_alphas"] == [0.5]
    assert calls["training_selected_ky_indices"] == (1, 2)
    assert calls["holdout_selected_ky_indices"] == (1, 2)
    assert calls["mboz"] == 21
    assert calls["nboz"] == 21


def test_alpha_holdout_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _alpha_holdout_payload()

    monkeypatch.setattr(
        alpha_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    result = alpha_gate.main(
        [
            "alpha",
            "--out",
            str(tmp_path / "alpha_holdout.png"),
            "--holdout-alphas",
            "0.25",
            "--training-selected-ky-indices",
            "1",
            "2",
            "--json-only",
        ]
    )

    assert result == 0
    assert calls["holdout_alphas"] == (0.25,)
    assert calls["training_selected_ky_indices"] == (1, 2)


def test_surface_holdout_payload_contracts(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = surface_gate.build_vmec_boozer_aggregate_surface_holdout_payload(
        ntheta=4, mboz=21, nboz=21
    )

    assert payload["artifact_kind"] == "vmec_boozer_aggregate_surface_holdout_gate"
    assert payload["passed"] is True
    assert payload["blocked"] is False
    assert payload["blockers"] == []
    assert payload["holdout_split"]["training_surface_indices"] == [18]
    assert payload["holdout_split"]["holdout_surface_indices"] == [19]
    assert calls["training_surface_indices"] == (18,)
    assert calls["holdout_surface_indices"] == (19,)


def test_surface_holdout_payload_fails_closed_on_execution_error(monkeypatch) -> None:
    def fake_report(**_kwargs):  # noqa: ANN003, ANN202
        raise ValueError("surface_index is outside the VMEC metric radial grid")

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = surface_gate.build_vmec_boozer_aggregate_surface_holdout_payload(
        training_surface_indices=(18,), holdout_surface_indices=(99,)
    )

    assert payload["passed"] is False
    assert payload["blocked"] is True
    assert payload["blockers"] == ["surface_split_execution_failed"]
    assert payload["exception_type"] == "ValueError"
    assert "surface_index" in payload["exception_message"]


def test_surface_holdout_payload_rejects_non_holdout_surface_split(monkeypatch) -> None:
    calls: list[object] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = surface_gate.build_vmec_boozer_aggregate_surface_holdout_payload(
        training_surface_indices=(18,), holdout_surface_indices=(18,)
    )

    assert payload["passed"] is False
    assert payload["blocked"] is True
    assert payload["blockers"] == ["surface_split_not_held_out"]
    assert calls == []


def test_surface_holdout_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    result = surface_gate.main(
        [
            "surface",
            "--out",
            str(tmp_path / "surface_holdout.png"),
            "--training-surface-indices",
            "18",
            "--holdout-surface-indices",
            "19",
            "--json-only",
        ]
    )

    assert result == 0
    assert calls["training_surface_indices"] == (18,)
    assert calls["holdout_surface_indices"] == (19,)


def test_line_search_gate_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _line_search_payload()

    monkeypatch.setattr(
        objective_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_report,
    )

    result = objective_gate.main(
        [
            "line-search",
            "--out",
            str(tmp_path / "line_search.png"),
            "--selected-ky-indices",
            "1",
            "2",
            "--max-steps",
            "2",
            "--json-only",
        ]
    )

    assert result == 0
    assert calls["selected_ky_indices"] == (1, 2)
    assert calls["max_steps"] == 2
    assert calls["mboz"] == 21


def test_objective_gate_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _objective_payload()

    monkeypatch.setattr(
        objective_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    result = objective_gate.main(
        [
            "--out",
            str(tmp_path / "gate.png"),
            "--selected-ky-indices",
            "1",
            "2",
            "--surface-indices",
            "3",
            "5",
            "--json-only",
        ]
    )

    assert result == 0
    assert calls["surface_indices"] == (3, 5)
    assert calls["selected_ky_indices"] == (1, 2)
    assert calls["mboz"] == 21


def test_objective_gate_maps_physical_ky_and_torflux(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _objective_payload()

    monkeypatch.setattr(
        objective_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    result = objective_gate.main(
        [
            "--out",
            str(tmp_path / "gate.png"),
            "--ky-values",
            "0.1",
            "0.3",
            "0.5",
            "--json-only",
        ]
    )
    assert result == 0
    assert calls["selected_ky_indices"] == (1, 3, 5)
    assert calls["ny"] == 12
    assert calls["ly"] == pytest.approx(2.0 * np.pi / 0.1)

    result = objective_gate.main(
        [
            "--out",
            str(tmp_path / "gate2.png"),
            "--torflux-values",
            "0.5",
            "0.7",
            "--json-only",
        ]
    )
    assert result == 0
    assert calls["surface_indices"] == (None,)
    assert calls["torflux_values"] == (0.5, 0.7)

    with pytest.raises(ValueError, match="torflux-values or --surface-indices"):
        objective_gate.main(
            ["--surface-indices", "3", "--torflux-values", "0.5", "--json-only"]
        )


def test_physical_ky_annotation_adds_resolved_metadata() -> None:
    payload = _objective_payload()
    objective_gate._annotate_physical_ky_samples(
        payload,
        requested_ky_values=[0.1, 0.2],
        solver_grid_options={
            "selected_ky_indices": (1, 2),
            "resolved_ky_values": (0.100000001, 0.200000003),
        },
    )

    assert payload["samples"][0]["ky"] == pytest.approx(0.1)
    assert payload["samples"][0]["selected_ky"] == pytest.approx(0.100000001)
    assert payload["samples"][0]["ky_abs_error"] == pytest.approx(1.0e-9)
    assert payload["samples"][1]["ky"] == pytest.approx(0.2)


def test_comparison_report_uses_same_sample_set(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(dict(kwargs))
        objective = str(kwargs["objective"])
        if objective == "growth":
            return _line_search_payload(objective, 2.0, 0.30, 0.29)
        return _line_search_payload(objective, 5.0, 0.90, 0.88)

    monkeypatch.setattr(
        comparison_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_report,
    )

    payload = comparison_gate.build_vmec_boozer_aggregate_line_search_comparison_report(
        selected_ky_indices=(1, 2),
        surface_indices=(None,),
        alphas=(0.0,),
        max_steps=1,
        ntheta=4,
    )

    assert payload["passed"] is True
    assert payload["same_sample_set"] is True
    assert payload["same_initial_update_direction"] is True
    assert [call["objective"] for call in calls] == ["growth", "quasilinear_flux"]
    assert all(call["selected_ky_indices"] == (1, 2) for call in calls)
    assert all(call["ntheta"] == 4 for call in calls)


def test_comparison_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(dict(kwargs))
        return _line_search_payload(str(kwargs["objective"]), 1.0, 1.0, 0.9)

    monkeypatch.setattr(
        comparison_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_report,
    )

    result = comparison_gate.main(
        [
            "--out",
            str(tmp_path / "comparison.png"),
            "--selected-ky-indices",
            "1",
            "2",
            "--max-steps",
            "2",
            "--json-only",
        ]
    )

    assert result == 0
    assert [call["objective"] for call in calls] == ["growth", "quasilinear_flux"]
    assert all(call["selected_ky_indices"] == (1, 2) for call in calls)
    assert all(call["max_steps"] == 2 for call in calls)
    assert all(call["mboz"] == 21 for call in calls)


def test_multi_point_payload_contracts(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _multi_point_payload()

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    payload = multi_point_gate.build_vmec_boozer_multi_point_objective_payload(
        max_wall_seconds=0.0
    )

    assert payload["artifact_kind"] == "vmec_boozer_multi_point_objective_gate"
    assert payload["passed"] is True
    assert payload["claim_scope"].startswith("bounded finite-difference")
    assert payload["multi_point_coverage"]["multi_alpha_or_surface"] is True
    assert payload["multi_point_coverage"]["n_samples_requested"] == 4
    assert payload["bounded_runtime"]["max_samples"] == 8
    assert calls["surface_indices"] == (None,)
    assert calls["alphas"] == (0.0, 0.5)
    assert calls["selected_ky_indices"] == (1, 2)
    assert calls["mboz"] == 21
    assert calls["nboz"] == 21


def test_multi_point_payload_accepts_two_surfaces(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _multi_point_payload()

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    payload = multi_point_gate.build_vmec_boozer_multi_point_objective_payload(
        surface_indices=(3, 5),
        alphas=(0.0,),
        selected_ky_indices=(1,),
        max_wall_seconds=0.0,
    )

    assert payload["multi_point_coverage"]["surface_indices"] == [3, 5]
    assert payload["multi_point_coverage"]["n_samples_requested"] == 2
    assert calls["surface_indices"] == (3, 5)
    assert calls["alphas"] == (0.0,)
    assert calls["selected_ky_indices"] == (1,)


def test_multi_point_main_uses_report_and_bounds(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _multi_point_payload()

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    result = multi_point_gate.main(
        [
            "--out",
            str(tmp_path / "gate.png"),
            "--surface-indices",
            "3",
            "5",
            "--alphas",
            "0.0",
            "--selected-ky-indices",
            "1",
            "--json-only",
            "--max-wall-seconds",
            "0",
        ]
    )

    assert result == 0
    assert calls["surface_indices"] == (3, 5)
    assert calls["alphas"] == (0.0,)
    assert calls["selected_ky_indices"] == (1,)


@pytest.mark.parametrize(
    "argv",
    [
        ["--alphas", "0.0", "--selected-ky-indices", "1", "--json-only"],
        [
            "--surface-indices",
            "3",
            "5",
            "--alphas",
            "0.0",
            "0.5",
            "--selected-ky-indices",
            "1",
            "2",
            "3",
            "--max-samples",
            "8",
            "--json-only",
        ],
    ],
    ids=["single_point", "over_sample_bound"],
)
def test_multi_point_main_rejects_invalid_coverage(
    monkeypatch, argv: list[str]
) -> None:
    def fake_report(**_kwargs):  # noqa: ANN003, ANN202
        raise AssertionError("report should not run for invalid coverage")

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    with pytest.raises(SystemExit):
        multi_point_gate.main(argv)


def test_second_equilibrium_payload_passes_with_mode21_defaults(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_fd(**kwargs):  # noqa: ANN003, ANN202
        calls["fd"] = kwargs
        return _second_fd_payload()

    def fake_line(**kwargs):  # noqa: ANN003, ANN202
        calls["line"] = kwargs
        return _second_line_payload()

    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd,
    )
    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_line,
    )

    payload = second_gate.build_vmec_boozer_second_equilibrium_aggregate_payload(
        max_wall_seconds=0.0
    )

    assert payload["passed"] is True
    assert payload["feasible"] is True
    assert payload["case_name"] == "li383_low_res"
    assert payload["mode_bound"] == {
        "mboz": 21,
        "nboz": 21,
        "minimum_required": 21,
        "passed": True,
    }
    assert payload["coverage"]["selected_ky_indices"] == [1, 2]
    assert payload["finite_difference_summary"]["central_derivative"] == -1.0e5
    assert payload["line_search_summary"]["accepted_steps"] == 1
    assert calls["fd"]["mboz"] == 21
    assert calls["fd"]["nboz"] == 21
    assert calls["line"]["case_name"] == "li383_low_res"


def test_second_equilibrium_payload_fails_closed_on_backend_error(monkeypatch) -> None:
    def fake_fd(**_kwargs):  # noqa: ANN003, ANN202
        raise RuntimeError("vmec_jax example fixture missing")

    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd,
    )

    payload = second_gate.build_vmec_boozer_second_equilibrium_aggregate_payload(
        max_wall_seconds=0.0
    )

    assert payload["passed"] is False
    assert payload["feasible"] is False
    assert payload["blocker_type"] == "RuntimeError"
    assert payload["blocker_message"] == "vmec_jax example fixture missing"
    assert payload["mode_bound"]["passed"] is True


def test_second_equilibrium_json_only_uses_reports(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_fd(**kwargs):  # noqa: ANN003, ANN202
        calls["fd"] = kwargs
        return _second_fd_payload()

    def fake_line(**kwargs):  # noqa: ANN003, ANN202
        calls["line"] = kwargs
        return _second_line_payload()

    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd,
    )
    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_line,
    )

    result = second_gate.main(
        [
            "--case-name",
            "nfp3_QI_fixed_resolution_final",
            "--selected-ky-indices",
            "1",
            "2",
            "--max-wall-seconds",
            "0",
            "--json-only",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert calls["fd"]["case_name"] == "nfp3_QI_fixed_resolution_final"
    assert calls["line"]["selected_ky_indices"] == (1, 2)


@pytest.mark.parametrize(
    ("gate", "writer", "payload", "out_name", "json_key", "csv_token"),
    [
        (
            alpha_gate,
            "write_vmec_boozer_aggregate_alpha_holdout_artifacts",
            {
                **_alpha_holdout_payload(),
                "artifact_kind": "vmec_boozer_aggregate_alpha_holdout_gate",
                "wall_seconds": 1.25,
            },
            "alpha_holdout.png",
            "artifact_kind",
            "heldout",
        ),
        (
            surface_gate,
            "write_vmec_boozer_aggregate_surface_holdout_artifacts",
            {
                **_surface_holdout_payload(),
                "kind": "vmec_boozer_aggregate_surface_holdout_gate",
                "artifact_kind": "vmec_boozer_aggregate_surface_holdout_gate",
                "blocked": False,
                "blockers": [],
                "wall_seconds": 1.25,
                "holdout_split": {
                    "training_surface_indices": [18],
                    "training_alphas": [0.0],
                    "training_selected_ky_indices": [1, 2],
                    "holdout_surface_indices": [19],
                    "holdout_alphas": [0.0],
                    "holdout_selected_ky_indices": [1, 2],
                },
            },
            "surface_holdout.png",
            "artifact_kind",
            "heldout_surface",
        ),
        (
            objective_gate,
            "write_vmec_boozer_aggregate_line_search_artifacts",
            _line_search_payload(),
            "line_search.png",
            "passed",
            "candidate_objective",
        ),
        (
            comparison_gate,
            "write_vmec_boozer_aggregate_line_search_comparison_artifacts",
            _comparison_payload(),
            "comparison.png",
            "same_initial_update_direction",
            "initial_update_direction",
        ),
        (
            objective_gate,
            "write_vmec_boozer_aggregate_objective_artifacts",
            {
                **_objective_payload(),
                "samples": [
                    {
                        **_objective_payload()["samples"][0],
                        "torflux": 0.64,
                        "surface": 0.64,
                        "ky": 0.1,
                        "selected_ky": 0.1,
                        "ky_abs_error": 0.0,
                    },
                    _objective_payload()["samples"][1],
                ],
            },
            "aggregate_gate.png",
            "passed",
            "ky_abs_error",
        ),
        (
            multi_point_gate,
            "write_vmec_boozer_multi_point_objective_artifacts",
            multi_point_gate._annotate_payload(
                _multi_point_payload(),
                surfaces=(None,),
                alphas=(0.0, 0.5),
                selected_ky_indices=(1, 2),
                max_samples=8,
                max_wall_seconds=300.0,
                elapsed_wall_seconds=1.25,
            ),
            "multi_point_gate.png",
            "artifact_kind",
            "alpha",
        ),
        (
            second_gate,
            "write_vmec_boozer_second_equilibrium_aggregate_artifacts",
            {
                "kind": "vmec_boozer_second_equilibrium_aggregate_gate",
                "passed": True,
                "feasible": True,
                "case_name": "li383_low_res",
                "objective": "quasilinear_flux",
                "mode_bound": {
                    "mboz": 21,
                    "nboz": 21,
                    "minimum_required": 21,
                    "passed": True,
                },
                "sample_bound": {
                    "n_samples_requested": 2,
                    "max_samples": 4,
                    "passed": True,
                },
                "bounded_runtime": {
                    "max_wall_seconds": 300.0,
                    "elapsed_wall_seconds": 41.2,
                    "passed": True,
                },
                "finite_difference_passed": True,
                "line_search_passed": True,
                "finite_difference_summary": {
                    "minus_value": 9.80,
                    "base_value": 9.79,
                    "plus_value": 9.78,
                    "central_derivative": -1.0e5,
                    "response_abs": 0.02,
                    "curvature_ratio": 0.01,
                    "n_samples": 2,
                },
                "line_search_summary": {
                    "accepted_steps": 1,
                    "initial_objective": 9.79,
                    "final_objective": 9.78,
                    "relative_reduction": 1.0e-3,
                    "stop_reason": "max_steps",
                },
            },
            "second_gate.png",
            "kind",
            "fd_central_derivative",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_vmec_boozer_aggregate_writers(
    tmp_path: Path,
    gate,
    writer: str,
    payload: dict[str, object],
    out_name: str,
    json_key: str,
    csv_token: str,
) -> None:
    paths = getattr(gate, writer)(payload, out=tmp_path / out_name)
    _assert_artifacts(paths, json_key=json_key, csv_token=csv_token)
