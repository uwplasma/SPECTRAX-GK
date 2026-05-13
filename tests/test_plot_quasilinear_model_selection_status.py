"""Tests for the quasilinear model-selection status artifact writer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "plot_quasilinear_model_selection_status.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_quasilinear_model_selection_status", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps({"promotion_gate": {"passed": True, "blockers": []}}),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "promotion_gate": {
                    "passed": True,
                    "accepted_candidates": ["spectral_envelope_ridge"],
                    "transport_mean_relative_error_gate": 0.35,
                    "interval_coverage_gate": 0.75,
                    "null_training_mean_mean_abs_relative_error": 0.8,
                    "linear_weight_mean_abs_relative_error": 0.9,
                },
                "candidates": {
                    "spectral_envelope_ridge": {
                        "mean_abs_relative_error": 0.22,
                        "prediction_interval_coverage": 0.88,
                        "promotion_eligible": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "kind": "quasilinear_calibration_report",
                "claim_level": "calibration_dataset",
                "passed": False,
                "by_split": {"holdout": {"mean_abs_relative_error": 2.0}},
            }
        ),
        encoding="utf-8",
    )
    return dataset, candidate, calibration


def test_model_selection_status_tool_writes_replayable_artifacts(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    dataset, candidate, calibration = _write_inputs(tmp_path)
    out = tmp_path / "status.png"

    assert (
        mod.main(
            [
                "--dataset",
                str(dataset),
                "--candidate",
                str(candidate),
                "--calibration-report",
                str(calibration),
                "--out",
                str(out),
                "--no-pdf",
                "--dpi",
                "80",
            ]
        )
        == 0
    )

    assert out.exists()
    assert out.with_suffix(".csv").exists()
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert (
        payload["claim_level"]
        == "scoped_candidate_model_selection_not_runtime_absolute_flux"
    )
    assert "pdf" not in mod.write_model_selection_status_artifacts(
        payload,
        out=tmp_path / "status_again.png",
        title="status",
        dpi=80,
        write_pdf=False,
    )
