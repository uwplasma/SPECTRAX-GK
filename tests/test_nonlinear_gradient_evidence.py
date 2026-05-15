from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from spectraxgk.nonlinear_gradient_evidence import (
    NonlinearTurbulenceGradientGapConfig,
    classify_gradient_artifact,
    load_json_artifact,
    nonlinear_turbulence_gradient_evidence_gap_report,
    nonlinear_turbulence_gradient_evidence_report,
    summarize_window_evidence,
)
from spectraxgk.quasilinear_window import (
    NonlinearWindowConvergenceConfig,
    nonlinear_window_convergence_report,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_nonlinear_turbulence_gradient_evidence.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("check_nonlinear_turbulence_gradient_evidence", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _window_report(offset: float, *, case: str) -> dict[str, object]:
    t = np.linspace(0.0, 240.0, 241)
    heat = 8.0 + offset + 0.02 * np.sin(2.0 * np.pi * t / 12.0)
    return nonlinear_window_convergence_report(
        t,
        heat,
        case=case,
        source_artifact=f"{case}.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            min_blocks=4,
            max_running_mean_rel_drift=0.01,
            max_terminal_mean_rel_delta=0.01,
            max_sem_rel=0.02,
        ),
    )


def _production_gradient() -> dict[str, object]:
    return {
        "kind": "nonlinear_turbulence_gradient_finite_difference_audit",
        "claim_level": "production_long_window_nonlinear_turbulence_gradient_evidence",
        "passed": True,
        "production_nonlinear_window_gradient_gate": True,
        "gradient": {
            "central": 1.25,
            "response_fraction": 0.08,
            "asymmetry_rel": 0.12,
        },
        "conditioning": {
            "condition_number": 42.0,
        },
        "uncertainty": {
            "gradient_sem_rel": 0.18,
        },
    }


def test_startup_fd_artifact_is_recorded_but_not_promoted() -> None:
    artifact = {
        "kind": "nonlinear_startup_window_finite_difference_audit",
        "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
        "passed": True,
        "transport_average_gate": False,
        "production_nonlinear_window_gradient_gate": False,
        "metrics": {
            "central_fd_dq_dtprim": 2.0,
            "response_fraction": 0.2,
            "derivative_asymmetry": 0.0,
        },
    }

    row = classify_gradient_artifact(artifact)

    assert row["artifact_passed"] is True
    assert row["qualifies_for_production_turbulence_gradient"] is False
    assert row["evidence_class"] == "startup_or_reduced_window_fd_not_production"
    assert "startup" in row["scope_blockers"]
    assert "transport_average_gate_false" in row["scope_blockers"]


def test_reduced_estimator_gradient_does_not_promote_even_with_replicates() -> None:
    reduced_gradient = {
        "kind": "mode21_vmec_boozer_nonlinear_window_gradient_gate",
        "claim_scope": "reduced nonlinear-window estimator gradient",
        "passed": True,
        "production_nonlinear_window_gradient_gate": False,
        "objective_gates": [
            {"objective": "nonlinear_window_heat_flux_mean", "finite_difference": 1.0, "implicit": 1.0}
        ],
        "conditioning": {
            "condition_number": 10.0,
            "response_fraction": 0.1,
            "asymmetry_rel": 0.1,
        },
        "uncertainty": {
            "gradient_sem_rel": 0.1,
        },
    }
    windows = [_window_report(-0.01, case="seed_1"), _window_report(0.01, case="seed_2")]

    report = nonlinear_turbulence_gradient_evidence_report(
        reduced_gradient,
        window_artifacts=windows,
    )

    assert report["passed"] is False
    assert report["window_evidence"]["passed"] is True
    assert report["gradient_artifact"]["qualifies_for_production_turbulence_gradient"] is False
    assert report["blockers"] == ["production_gradient_artifact"]
    assert report["evidence_gap"]["promotion_blocked"] is True
    assert report["evidence_gap"]["current_window_evidence_passed"] is True
    assert report["evidence_gap"]["required_campaign"]["required_runs"][0]["state"] == "minus_delta"
    assert report["evidence_gap"]["required_campaign"]["required_runs"][1]["state"] == "baseline"
    assert report["evidence_gap"]["required_campaign"]["required_runs"][2]["state"] == "plus_delta"
    audit = report["evidence_gap"]["required_campaign"]["finite_difference_audit"]
    assert audit["acceptance_gates"]["production_nonlinear_window_gradient_gate"] is True
    assert "central_gradient" in audit["required_metrics"]


def test_production_gradient_can_use_derived_replicated_window_summaries() -> None:
    windows = [
        _window_report(-0.02, case="seed_1"),
        _window_report(0.0, case="seed_2"),
        _window_report(0.02, case="dt_variant"),
    ]

    report = nonlinear_turbulence_gradient_evidence_report(
        _production_gradient(),
        window_artifacts=windows,
    )

    assert report["passed"] is True
    assert report["production_nonlinear_window_gradient_gate"] is True
    assert report["evidence_gap"]["passed"] is True
    assert report["evidence_gap"]["promotion_blocked"] is False
    assert report["window_evidence"]["derived_ensemble"]["source"] == "derived_from_window_summaries"
    assert report["gradient_artifact"]["conditioning"]["gradient_uncertainty_rel"] == 0.18


def test_gap_report_names_custom_paired_parameter_campaign() -> None:
    report = {
        "passed": False,
        "blockers": ["production_gradient_artifact"],
        "gradient_artifact": {
            "path": "gradient.json",
            "evidence_class": "startup_or_reduced_window_fd_not_production",
        },
        "window_evidence": {
            "passed": True,
            "ensemble_rows": [
                {"qualifies_for_replicated_long_window_uncertainty": True},
            ],
        },
    }

    gap = nonlinear_turbulence_gradient_evidence_gap_report(
        report,
        gap_config=NonlinearTurbulenceGradientGapConfig(
            case_slug="qa_ess_dqi",
            parameter_name="rbc_1_0",
            perturbation_fraction=0.02,
            analysis_tmin=500.0,
            analysis_tmax=900.0,
            minimum_tmax=900.0,
            minimum_grid="n96x96x64x48x48",
            replicate_labels=("seed41", "seed42", "dt0p04"),
        ),
    )

    assert gap["passed"] is False
    assert gap["promotion_blocked"] is True
    assert gap["qualifying_window_ensemble_count"] == 1
    assert gap["missing_evidence"][0]["current_artifact_path"] == "gradient.json"
    required_runs = gap["required_campaign"]["required_runs"]
    assert [row["state"] for row in required_runs] == [
        "minus_delta",
        "baseline",
        "plus_delta",
    ]
    assert required_runs[0]["parameter_multiplier"] == pytest.approx(0.98)
    assert required_runs[2]["parameter_multiplier"] == pytest.approx(1.02)
    assert required_runs[0]["run_contract"]["analysis_window"] == [500.0, 900.0]
    assert required_runs[0]["run_contract"]["minimum_grid"] == "n96x96x64x48x48"
    assert required_runs[0]["replicates"] == ["seed41", "seed42", "dt0p04"]


def test_production_gradient_fails_closed_without_uncertainty() -> None:
    gradient = _production_gradient()
    gradient.pop("uncertainty")
    windows = [_window_report(-0.02, case="seed_1"), _window_report(0.02, case="seed_2")]

    report = nonlinear_turbulence_gradient_evidence_report(
        gradient,
        window_artifacts=windows,
    )

    gradient_gates = {
        gate["metric"]: gate["passed"]
        for gate in report["gradient_artifact"]["gates"]
    }
    assert report["passed"] is False
    assert gradient_gates["gradient_uncertainty_bounded"] is False


def test_unscoped_nested_passed_artifact_with_bad_numbers_stays_blocked() -> None:
    artifact = {
        "kind": "long_window_gradient_candidate",
        "gate_report": {"passed": True},
        "objective_gates": [
            "not-a-row",
            {"finite_difference": "not-a-number", "implicit": "inf"},
        ],
        "gradient": {"central": "not-a-number"},
        "metrics": {"response_fraction": float("nan"), "derivative_asymmetry": "bad"},
        "conditioning": {"condition_number": "bad"},
        "uncertainty": {"sem_rel": None},
    }

    row = classify_gradient_artifact(artifact)

    assert row["artifact_passed"] is True
    assert row["evidence_class"] == "unscoped_gradient_or_fd_artifact_not_production"
    assert row["qualifies_for_production_turbulence_gradient"] is False
    assert row["conditioning"]["central_gradient"] is None
    assert {gate["metric"]: gate["passed"] for gate in row["gates"]}[
        "finite_gradient_estimate"
    ] is False


def test_explicit_nonlinear_turbulence_gradient_flag_can_promote_scope() -> None:
    artifact = _production_gradient()
    artifact.pop("production_nonlinear_window_gradient_gate")
    artifact["nonlinear_turbulence_gradient_gate"] = True

    row = classify_gradient_artifact(artifact)

    assert row["explicit_production_scope"] is True
    assert row["evidence_class"] == "production_long_window_turbulence_gradient_candidate"
    assert row["qualifies_for_production_turbulence_gradient"] is True


def test_window_evidence_handles_input_ensembles_unsupported_rows_and_path_mismatch() -> None:
    ensemble = {
        "kind": "nonlinear_window_ensemble_report",
        "passed": True,
        "statistics": {
            "n_reports": 3,
            "combined_sem_rel": 0.04,
            "mean_rel_spread": 0.05,
        },
    }
    malformed_ensemble = {
        "kind": "nonlinear_window_ensemble_report",
        "passed": True,
        "statistics": ["not", "a", "dict"],
    }
    unsupported = {"kind": "single_trace_debug_artifact", "promotion_gate": {"passed": True}}

    summary = summarize_window_evidence(
        [ensemble, malformed_ensemble, unsupported],
        paths=["ensemble.json", "bad_ensemble.json", "debug.json"],
    )

    assert summary["passed"] is True
    rows = summary["ensemble_rows"]
    assert rows[0]["source"] == "input_ensemble"
    assert rows[0]["qualifies_for_replicated_long_window_uncertainty"] is True
    assert rows[1]["statistics"] == {}
    assert rows[2]["source"] == "unsupported_window_artifact"
    assert rows[2]["passed"] is True

    with pytest.raises(ValueError, match="paths length"):
        summarize_window_evidence([ensemble], paths=["ensemble.json", "extra.json"])


def test_load_json_artifact_rejects_non_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain a JSON object"):
        load_json_artifact(path)


def test_cli_writes_report_and_can_fail_on_blocked(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gradient_path = tmp_path / "gradient.json"
    gradient_path.write_text(
        json.dumps(
            {
                "kind": "vmec_boozer_nonlinear_startup_finite_difference_audit",
                "claim_level": "vmec_boozer_geometry_perturbed_startup_plumbing_fd_audit_not_transport_average",
                "passed": True,
                "transport_average_gate": False,
                "production_nonlinear_window_gradient_gate": False,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "evidence.json"
    gap_out = tmp_path / "gap.json"

    rc = mod.main(
        [
            "--gradient-artifact",
            str(gradient_path),
            "--json-out",
            str(out),
            "--gap-json-out",
            str(gap_out),
            "--gap-case-slug",
            "qa_ess_gradient",
            "--gradient-parameter-name",
            "rbc_1_0",
            "--fail-on-blocked",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    gap = json.loads(gap_out.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["passed"] is False
    assert payload["blockers"] == [
        "production_gradient_artifact",
        "replicated_long_window_uncertainty",
    ]
    assert gap["promotion_blocked"] is True
    assert gap["required_campaign"]["case_slug"] == "qa_ess_gradient"
    assert gap["required_campaign"]["parameter_name"] == "rbc_1_0"
