"""Tests for quasilinear model-selection claim-boundary utilities."""

from __future__ import annotations

import json
from pathlib import Path

import spectraxgk
from spectraxgk.quasilinear_model_selection import (
    build_quasilinear_model_selection_status,
    build_quasilinear_model_selection_status_from_paths,
)


def _dataset_payload(*, passed: bool = True) -> dict:
    return {
        "kind": "quasilinear_dataset_sufficiency",
        "promotion_gate": {"passed": passed, "blockers": [] if passed else ["data"]},
    }


def _candidate_payload(*, accepted: bool = True) -> dict:
    accepted_candidates = ["spectral_envelope_ridge"] if accepted else []
    return {
        "kind": "quasilinear_candidate_uncertainty_report",
        "promotion_gate": {
            "passed": accepted,
            "accepted_candidates": accepted_candidates,
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


def _calibration_report(*, promoted: bool = False) -> dict:
    return {
        "kind": "quasilinear_calibration_report",
        "claim_level": (
            "calibrated_absolute_flux" if promoted else "calibration_dataset"
        ),
        "passed": promoted,
        "by_split": {
            "holdout": {
                "n": 5,
                "mean_abs_relative_error": 0.2 if promoted else 2.5,
            }
        },
    }


def test_model_selection_status_promotes_scoped_candidate_not_absolute_flux() -> None:
    status = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[_calibration_report()],
    )

    assert (
        spectraxgk.build_quasilinear_model_selection_status
        is build_quasilinear_model_selection_status
    )
    assert status["passed"] is True
    assert (
        status["claim_level"]
        == "scoped_candidate_model_selection_not_runtime_absolute_flux"
    )
    assert status["promotion_gate"]["blockers"] == []
    assert status["metrics"]["candidate_mean_abs_relative_error"] == 0.24
    assert status["calibration_reports"][0]["claim_level"] == "calibration_dataset"


def test_model_selection_status_fails_closed_for_overclaims_or_missing_skill() -> None:
    promoted = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(),
        calibration_reports=[_calibration_report(promoted=True)],
    )
    assert promoted["passed"] is False
    assert "absolute_flux_not_promoted" in promoted["promotion_gate"]["blockers"]

    missing_candidate = build_quasilinear_model_selection_status(
        dataset_sufficiency=_dataset_payload(),
        candidate_uncertainty=_candidate_payload(accepted=False),
        calibration_reports=[_calibration_report()],
    )
    assert missing_candidate["passed"] is False
    assert (
        "required_candidate_accepted"
        in missing_candidate["promotion_gate"]["blockers"]
    )
    assert (
        "candidate_uncertainty_passed"
        in missing_candidate["promotion_gate"]["blockers"]
    )


def test_model_selection_status_path_wrapper_preserves_source_artifacts(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.json"
    candidate = tmp_path / "candidate.json"
    calibration = tmp_path / "calibration.json"
    dataset.write_text(json.dumps(_dataset_payload()), encoding="utf-8")
    candidate.write_text(json.dumps(_candidate_payload()), encoding="utf-8")
    calibration.write_text(json.dumps(_calibration_report()), encoding="utf-8")

    status = build_quasilinear_model_selection_status_from_paths(
        dataset_sufficiency=dataset,
        candidate_uncertainty=candidate,
        calibration_reports=[calibration],
    )

    assert status["passed"] is True
    assert status["calibration_reports"][0]["artifact"] == str(calibration)
    assert (
        spectraxgk.build_quasilinear_model_selection_status_from_paths
        is build_quasilinear_model_selection_status_from_paths
    )
