"""Tests for quasilinear absolute-flux promotion guardrails."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "check_quasilinear_promotion_guardrails.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_quasilinear_promotion_guardrails", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_doc(path: Path, text: str | None = None) -> None:
    path.write_text(
        text
        or (
            "This diagnostic is not a runtime/TOML absolute-flux predictor. "
            "Absolute-flux prediction not promoted.\n"
        ),
        encoding="utf-8",
    )


def _calibration_report(*, claim_level: str, passed: bool, holdout_error: float) -> dict:
    return {
        "kind": "quasilinear_calibration_report",
        "claim_level": claim_level,
        "passed": passed,
        "holdout_mean_rel_gate": 0.35,
        "metadata": {"calibration_policy": "one constant train/holdout"},
        "by_split": {
            "train": {"n": 1, "mean_abs_relative_error": 0.0},
            "holdout": {"n": 1, "mean_abs_relative_error": holdout_error},
        },
        "points": [
            {
                "case": "train",
                "split": "train",
                "geometry": "cyclone",
                "electron_model": "adiabatic",
                "saturation_rule": "mixing_length",
                "nonlinear_artifact": "tools_out/train.csv",
                "quasilinear_artifact": "docs/_static/train_spectrum.csv",
                "predicted_heat_flux": 1.0,
                "raw_predicted_heat_flux": 0.5,
                "calibration_scale": 2.0,
                "observed_heat_flux": 1.0,
                "observed_heat_flux_std": 0.1,
            },
            {
                "case": "holdout",
                "split": "holdout",
                "geometry": "miller",
                "electron_model": "adiabatic",
                "saturation_rule": "mixing_length",
                "nonlinear_artifact": "tools_out/holdout.csv",
                "quasilinear_artifact": "docs/_static/holdout_spectrum.csv",
                "predicted_heat_flux": 1.0,
                "raw_predicted_heat_flux": 0.5,
                "calibration_scale": 2.0,
                "observed_heat_flux": 1.0,
                "observed_heat_flux_std": 0.2,
            },
        ],
    }


def test_promoted_absolute_flux_requires_passed_holdout_gate_and_window_stats(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    payload = _calibration_report(
        claim_level="calibrated_absolute_flux",
        passed=True,
        holdout_error=0.7,
    )
    payload["points"][1]["observed_heat_flux_std"] = float("nan")
    report.write_text(json.dumps(payload), encoding="utf-8")
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed = {gate["metric"]: gate["detail"] for gate in audit["gate_report"]["gates"] if not gate["passed"]}
    assert "train_holdout_point_metadata" in failed
    assert "promoted_holdout_gate" in failed


def test_unpromoted_report_with_finite_metadata_passes_synthetic_guardrail(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc)

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is True
    assert audit["calibration_reports"][0]["n_train"] == 1
    assert audit["calibration_reports"][0]["n_holdout"] == 1


def test_docs_without_nonpromotion_marker_fail_scope_check(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _calibration_report(
                claim_level="calibration_dataset",
                passed=False,
                holdout_error=4.0,
            )
        ),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.rst"
    _write_doc(doc, "This section describes a calibrated absolute-flux predictor.\n")

    audit = mod.build_guardrail_audit([str(report)], [doc])

    assert audit["passed"] is False
    failed_metrics = {gate["metric"] for gate in audit["gate_report"]["gates"] if not gate["passed"]}
    assert f"doc_scope_marker:{doc}" in failed_metrics
    assert f"doc_no_absolute_flux_overclaim:{doc}" in failed_metrics


def test_tracked_quasilinear_promotion_guardrails_pass() -> None:
    mod = _load_tool_module()

    audit = mod.build_guardrail_audit(
        list(mod.DEFAULT_REPORT_PATTERNS),
        [str(path) for path in mod.DEFAULT_DOCS],
    )

    assert audit["passed"] is True
    assert audit["summary"]["n_calibration_reports"] == 4
    assert audit["summary"]["n_input_validation_reports"] >= 4
    assert audit["summary"]["n_promotion_gate_reports"] >= 4
