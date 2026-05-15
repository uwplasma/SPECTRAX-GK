"""Additional fail-closed tests for quasilinear model-selection artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from spectraxgk.quasilinear_model_selection import (
    build_quasilinear_model_selection_status,
    build_quasilinear_model_selection_status_from_paths,
)


def _dataset_payload() -> dict[str, Any]:
    return {
        "kind": "quasilinear_dataset_sufficiency",
        "promotion_gate": {"passed": True, "blockers": []},
    }


def _candidate_payload() -> dict[str, Any]:
    return {
        "kind": "quasilinear_candidate_uncertainty_report",
        "promotion_gate": {
            "passed": True,
            "accepted_candidates": ["spectral_envelope_ridge"],
            "transport_mean_relative_error_gate": 0.35,
            "interval_coverage_gate": 0.75,
            "null_training_mean_mean_abs_relative_error": 0.82,
            "linear_weight_mean_abs_relative_error": 0.93,
        },
        "candidates": {
            "spectral_envelope_ridge": {
                "mean_abs_relative_error": 0.24,
                "prediction_interval_coverage": 0.86,
                "promotion_eligible": True,
                "eligibility_failures": [],
            }
        },
    }


def _calibration_report(
    *,
    claim_level: str = "calibration_dataset",
    passed: object = False,
    include_by_split: bool = True,
) -> dict[str, Any]:
    report = {
        "kind": "quasilinear_calibration_report",
        "claim_level": claim_level,
        "passed": passed,
    }
    if include_by_split:
        report["by_split"] = {
            "holdout": {"n": 3, "mean_abs_relative_error": 1.8}
        }
    return report


def _optimized_equilibrium_audit(*, passed: bool = True) -> dict[str, Any]:
    return {
        "kind": "production_nonlinear_turbulent_flux_optimization_guard",
        "claim_level": "production_nonlinear_optimization_promoted_by_replicated_transport_windows",
        "passed": True,
        "production_nonlinear_optimization_promoted": passed,
        "promotion_gate": {"passed": passed, "blockers": [] if passed else ["audit"]},
        "summary": {
            "qualifying_optimized_equilibrium_ensembles": 1 if passed else 0,
        },
        "optimized_equilibrium_artifacts": [
            {
                "path": "optimized_equilibrium_post_optimization_gate.json",
                "optimized_equilibrium_marker": True,
                "qualifies_for_production_optimization": passed,
            }
        ],
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_model_selection_path_inputs_reject_malformed_json_payloads(
    tmp_path: Path,
) -> None:
    dataset = _write_json(tmp_path / "dataset.json", _dataset_payload())
    candidate = _write_json(tmp_path / "candidate.json", _candidate_payload())
    calibration = _write_json(tmp_path / "calibration.json", _calibration_report())

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError):
        build_quasilinear_model_selection_status_from_paths(
            dataset_sufficiency=bad_json,
            candidate_uncertainty=candidate,
            calibration_reports=[calibration],
        )

    array_payload = _write_json(tmp_path / "array.json", [])
    with pytest.raises(ValueError, match="must contain a JSON object"):
        build_quasilinear_model_selection_status_from_paths(
            dataset_sufficiency=dataset,
            candidate_uncertainty=array_payload,
            calibration_reports=[calibration],
        )


def test_model_selection_fails_closed_when_required_candidate_metrics_are_missing() -> (
    None
):
    candidate = _candidate_payload()
    candidate["candidates"] = {"spectral_envelope_ridge": {"promotion_eligible": True}}

    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=candidate,
        calibration_reports=[_calibration_report()],
    )

    assert status["passed"] is False
    assert status["metrics"]["candidate_mean_abs_relative_error"] is None
    assert status["metrics"]["candidate_prediction_interval_coverage"] is None
    assert {
        "required_candidate_transport_error",
        "required_candidate_interval_coverage",
        "required_candidate_beats_training_mean_null",
        "required_candidate_beats_linear_weight",
    }.issubset(status["promotion_gate"]["blockers"])


def test_model_selection_fails_closed_on_absolute_flux_overclaim_without_holdout() -> (
    None
):
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[
            _calibration_report(
                claim_level="calibrated_absolute_flux",
                passed="yes",
                include_by_split=False,
            )
        ],
    )

    assert status["passed"] is False
    assert "absolute_flux_not_promoted" in status["promotion_gate"]["blockers"]
    assert (
        status["calibration_reports"][0]["holdout_mean_abs_relative_error"] is None
    )


def test_model_selection_fails_closed_for_calibration_report_missing_by_split() -> (
    None
):
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[_calibration_report(include_by_split=False)],
    )

    assert status["passed"] is False
    assert (
        "calibration_reports_have_holdout_metrics"
        in status["promotion_gate"]["blockers"]
    )
    assert (
        status["calibration_reports"][0]["holdout_mean_abs_relative_error"] is None
    )


def test_model_selection_path_wrapper_requires_path_payloads(tmp_path: Path) -> None:
    candidate = _write_json(tmp_path / "candidate.json", _candidate_payload())
    calibration = _write_json(tmp_path / "calibration.json", _calibration_report())

    with pytest.raises(TypeError, match="dataset_sufficiency must be a path"):
        build_quasilinear_model_selection_status_from_paths(
            dataset_sufficiency=_dataset_payload(),  # type: ignore[arg-type]
            candidate_uncertainty=candidate,
            calibration_reports=[calibration],
        )

    dataset = _write_json(tmp_path / "dataset.json", _dataset_payload())
    with pytest.raises(TypeError, match=r"calibration_reports\[0\] must be a path"):
        build_quasilinear_model_selection_status_from_paths(
            dataset_sufficiency=dataset,
            candidate_uncertainty=candidate,
            calibration_reports=[_calibration_report()],  # type: ignore[list-item]
        )

    with pytest.raises(
        TypeError, match=r"optimized_equilibrium_nonlinear_audits\[0\] must be a path"
    ):
        build_quasilinear_model_selection_status_from_paths(
            dataset_sufficiency=dataset,
            candidate_uncertainty=candidate,
            calibration_reports=[calibration],
            optimized_equilibrium_nonlinear_audits=[
                _optimized_equilibrium_audit()
            ],  # type: ignore[list-item]
        )


def test_model_selection_path_wrapper_accepts_optimized_equilibrium_audit(
    tmp_path: Path,
) -> None:
    dataset = _write_json(tmp_path / "dataset.json", _dataset_payload())
    candidate = _write_json(tmp_path / "candidate.json", _candidate_payload())
    calibration = _write_json(tmp_path / "calibration.json", _calibration_report())
    audit = _write_json(
        tmp_path / "optimized_audit.json", _optimized_equilibrium_audit()
    )

    status = build_quasilinear_model_selection_status_from_paths(
        dataset_sufficiency=dataset,
        candidate_uncertainty=candidate,
        calibration_reports=[calibration],
        optimized_equilibrium_nonlinear_audits=[audit],
        require_optimized_equilibrium_nonlinear_audit=True,
    )

    assert status["passed"] is True
    assert status["optimized_equilibrium_nonlinear_audits"][0]["artifact"] == str(
        audit
    )


def test_model_selection_fails_closed_for_supplied_failed_optimized_audit() -> None:
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[_calibration_report()],
        optimized_equilibrium_nonlinear_audits=[
            _optimized_equilibrium_audit(passed=False)
        ],
    )

    assert status["passed"] is False
    assert (
        "optimized_equilibrium_nonlinear_audit_qualified"
        in status["promotion_gate"]["blockers"]
    )
    assert (
        "optimized_equilibrium_audit_not_passed"
        in status["optimized_equilibrium_nonlinear_audits"][0]["blockers"]
    )


def test_model_selection_helpers_are_exported_at_package_top_level() -> None:
    import spectraxgk as sgk

    status = sgk.build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[_calibration_report()],
    )

    assert status["passed"] is True
    assert status["required_candidate"] == "spectral_envelope_ridge"
