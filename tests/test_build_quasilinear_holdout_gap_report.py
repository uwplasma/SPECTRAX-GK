"""Tests for the quasilinear holdout gap report builder."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "build_quasilinear_holdout_gap_report.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_quasilinear_holdout_gap_report", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate(metric: str, observed: float, limit: float, *, passed: bool) -> dict:
    return {
        "abs_error": observed,
        "atol": limit,
        "metric": metric,
        "observed": observed,
        "passed": passed,
        "reference": 0.0,
        "rtol": 0.0,
    }


def _external_gate(path: Path, case: str, *, passed: bool, gates: list[dict]) -> Path:
    payload = {
        "case": case,
        "claim_level": "passed_grid_convergence_candidate_for_transport_holdout"
        if passed
        else "negative_grid_convergence_result_not_transport_validation",
        "gate_report": {"case": case, "gates": gates, "passed": passed},
        "kind": "external_vmec_nonlinear_grid_convergence_gate",
        "promotion_gate": {
            "passed": passed,
            "reason": f"{case} {'passed' if passed else 'failed'} synthetic gate",
        },
    }
    return _write_json(path, payload)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    model_selection = _write_json(
        tmp_path / "model_selection.json",
        {
            "accepted_candidates": ["spectral_envelope_ridge"],
            "claim_level": "scoped_candidate_model_selection_not_runtime_absolute_flux",
            "kind": "quasilinear_model_selection_status",
            "metrics": {
                "candidate_mean_abs_relative_error": 0.2,
                "candidate_prediction_interval_coverage": 0.9,
                "transport_mean_relative_error_gate": 0.35,
            },
            "passed": True,
            "required_candidate": "spectral_envelope_ridge",
        },
    )
    screening_skill = _write_json(
        tmp_path / "screening_skill.json",
        {
            "claim_level": "screening_correlation_model_development_not_absolute_flux_promotion",
            "gates": {
                "absolute_flux_promotion_passed": False,
                "accepted_absolute_flux_models": [],
                "accepted_holdout_screening_models": [],
                "accepted_screening_models": ["spectral_envelope_ridge"],
                "best_holdout_screening_model": "spectral_envelope_ridge",
                "best_screening_model": "spectral_envelope_ridge",
                "holdout_screening_correlation_passed": False,
                "screening_correlation_passed": True,
                "pairwise_order_gate": 0.75,
                "spearman_gate": 0.75,
            },
            "kind": "quasilinear_screening_skill",
            "models": [
                {
                    "holdout_mean_abs_relative_error": 0.25,
                    "holdout_pairwise_order_accuracy": 0.73,
                    "holdout_spearman": 0.71,
                    "mean_abs_relative_error": 0.29,
                    "model": "spectral_envelope_ridge",
                    "pairwise_order_accuracy": 0.79,
                    "spearman": 0.81,
                }
            ],
        },
    )
    train_holdout = _write_json(
        tmp_path / "train_holdout.json",
        {
            "by_split": {
                "holdout": {
                    "max_abs_relative_error": 0.5,
                    "mean_abs_relative_error": 0.5,
                    "n": 1,
                },
                "train": {"mean_abs_relative_error": 0.0, "n": 1},
            },
            "claim_level": "calibration_dataset",
            "holdout_mean_rel_gate": 0.35,
            "kind": "quasilinear_calibration_report",
            "observed_floor": 1e-12,
            "passed": False,
            "points": [
                {
                    "case": "train_external_window",
                    "geometry": "itermodel_external_vmec",
                    "nonlinear_artifact": "tools_out/train.csv",
                    "observed_heat_flux": 2.0,
                    "predicted_heat_flux": 2.0,
                    "quasilinear_artifact": "docs/_static/train_spectrum.csv",
                    "split": "train",
                },
                {
                    "case": "holdout_external_window",
                    "geometry": "dshape_external_vmec",
                    "nonlinear_artifact": "tools_out/holdout.csv",
                    "observed_heat_flux": 10.0,
                    "predicted_heat_flux": 15.0,
                    "quasilinear_artifact": "docs/_static/holdout_spectrum.csv",
                    "split": "holdout",
                },
            ],
        },
    )
    window_stats = _write_json(
        tmp_path / "window_stats.json",
        {
            "case_gate_passed": {"kbm_nonlinear_window": True},
            "case_gate_thresholds": {"kbm_nonlinear_window": 0.02},
            "cases": ["kbm_nonlinear_window"],
            "max_mean_rel_abs_by_case": {"kbm_nonlinear_window": 0.01},
        },
    )
    dataset = _write_json(
        tmp_path / "dataset.json",
        {
            "cases": [
                {
                    "case": "train_external_window",
                    "gate_case": "ITERModel external VMEC nonlinear t350 high-grid convergence",
                    "geometry": "itermodel_external_vmec",
                    "split": "train",
                },
                {
                    "case": "holdout_external_window",
                    "gate_case": "D-shaped tokamak external VMEC nonlinear t250 high-grid convergence",
                    "geometry": "dshape_external_vmec",
                    "split": "holdout",
                },
            ],
            "excluded_validated_nonlinear_cases": [
                {
                    "case": "kbm_nonlinear_window",
                    "gate_passed": True,
                    "reason": "electromagnetic nonlinear lane; electrostatic quasilinear channels are not promoted",
                }
            ],
            "input_validation": {
                "cases": [
                    {
                        "case": "train_external_window",
                        "gate_case": "ITERModel external VMEC nonlinear t350 high-grid convergence",
                        "passed": True,
                        "reason": "matched passed nonlinear summary gate",
                        "split": "train",
                    },
                    {
                        "case": "holdout_external_window",
                        "gate_case": "D-shaped tokamak external VMEC nonlinear t250 high-grid convergence",
                        "passed": True,
                        "reason": "matched passed nonlinear summary gate",
                        "split": "holdout",
                    },
                ]
            },
        },
    )
    pass_gate = [_gate("common_window_max_relative_slope_per_time", 0.001, 0.002, passed=True)]
    failed_gate = [
        _gate("common_window_max_relative_slope_per_time", 0.0022, 0.002, passed=False),
        _gate("common_window_max_heat_flux_cv", 0.18, 0.2, passed=True),
    ]
    _external_gate(
        tmp_path / "external_holdout.json",
        "D-shaped tokamak external VMEC nonlinear t250 high-grid convergence",
        passed=True,
        gates=pass_gate,
    )
    _external_gate(
        tmp_path / "external_train.json",
        "ITERModel external VMEC nonlinear t350 high-grid convergence",
        passed=True,
        gates=pass_gate,
    )
    _external_gate(
        tmp_path / "external_itermodel_audit.json",
        "ITERModel external VMEC independent audit t450 high-grid convergence",
        passed=True,
        gates=pass_gate,
    )
    _external_gate(
        tmp_path / "external_itermodel_t250.json",
        "ITERModel external VMEC nonlinear t250 high-grid convergence",
        passed=False,
        gates=failed_gate,
    )
    _external_gate(
        tmp_path / "external_dshape_older.json",
        "D-shaped tokamak external VMEC nonlinear grid convergence",
        passed=True,
        gates=pass_gate,
    )
    _external_gate(
        tmp_path / "external_shaped_failed.json",
        "Shaped tokamak external VMEC nonlinear t450 high-grid convergence",
        passed=False,
        gates=[
            _gate(
                "common_window_pairwise_heat_flux_symmetric_relative_difference",
                0.306,
                0.15,
                passed=False,
            )
        ],
    )
    return {
        "dataset": dataset,
        "model_selection": model_selection,
        "screening_skill": screening_skill,
        "train_holdout": train_holdout,
        "window_stats": window_stats,
    }


def test_gap_report_preserves_claim_boundary_and_ranks_next_holdout(tmp_path: Path) -> None:
    mod = _load_tool_module()
    paths = _write_inputs(tmp_path)
    external_gates = [
        mod._load_external_gate(path) for path in sorted(tmp_path.glob("external_*.json"))
    ]

    report = mod.build_holdout_gap_report(
        model_selection=json.loads(paths["model_selection"].read_text(encoding="utf-8")),
        train_holdout=json.loads(paths["train_holdout"].read_text(encoding="utf-8")),
        window_stats=json.loads(paths["window_stats"].read_text(encoding="utf-8")),
        external_gates=external_gates,
        dataset=json.loads(paths["dataset"].read_text(encoding="utf-8")),
        screening_skill=json.loads(paths["screening_skill"].read_text(encoding="utf-8")),
    )

    assert report["claim_level"] == "holdout_gap_report_no_absolute_flux_promotion"
    assert report["absolute_flux_promoted"] is False
    assert report["promotion_gate"]["passed"] is False
    assert "absolute_flux_predictor_not_promoted" in report["promotion_gate"]["blockers"]
    assert report["summary"]["n_admitted_holdouts"] == 1
    assert report["summary"]["n_training_references"] == 1
    assert report["admitted_holdouts"][0]["absolute_relative_error"] == 0.5
    assert report["admitted_holdouts"][0]["gate_passed"] is True
    assert any(row["case"] == "kbm_nonlinear_window" for row in report["excluded_candidates"])
    shaped = next(
        row
        for row in report["excluded_candidates"]
        if row["case"] == "Shaped tokamak external VMEC nonlinear t450 high-grid convergence"
    )
    assert shaped["geometry"] == "shaped_tokamak_external_vmec"
    assert shaped["status"] == "excluded_failed_external_gate"
    audit = next(
        row
        for row in report["excluded_candidates"]
        if row["case"] == "ITERModel external VMEC independent audit t450 high-grid convergence"
    )
    assert audit["geometry"] == "itermodel_external_vmec"
    assert audit["status"] == "excluded_same_family_training_audit"
    assert audit["eligible_for_next_candidate"] is False
    assert report["next_best_candidates"][0]["status"] == "training_reference_not_independent_holdout"
    assert report["next_actual_nonlinear_holdout_needed"]["preferred_family"] == "itermodel_external_vmec"
    screening = report["screening_skill_status"]
    assert screening["screening_correlation_passed"] is True
    assert screening["holdout_screening_correlation_passed"] is False
    assert screening["best_holdout_spearman"] == 0.71
    screening_requirements = report["screening_promotion_requirements"]
    assert screening_requirements["screening_promoted"] is False
    assert "heldout_screening_correlation_passed" in screening_requirements["blockers"]
    assert (
        "screening_requirement:heldout_screening_correlation_passed"
        in report["promotion_gate"]["blockers"]
    )
    nearest = report["next_actual_nonlinear_holdout_needed"]["nearest_tracked_gap"]
    assert nearest["case"] == "ITERModel external VMEC nonlinear t250 high-grid convergence"
    assert nearest["next_best_score"] == 1.1
    requirements = report["absolute_flux_promotion_requirements"]
    assert requirements["absolute_flux_promoted"] is False
    assert requirements["reconsideration_ready"] is False
    assert requirements["numeric_gap"]["holdout_mean_abs_relative_error"] == 0.5
    assert requirements["numeric_gap"]["holdout_mean_rel_gate"] == 0.35
    assert requirements["numeric_gap"]["error_factor_to_gate"] == 0.5 / 0.35
    assert (
        requirements["coverage_gap"]["additional_total_independent_holdouts_needed"]
        == 8
    )
    assert (
        requirements["coverage_gap"][
            "additional_external_vmec_holdout_families_needed"
        ]
        == 3
    )
    assert (
        requirements["coverage_gap"][
            "additional_nonaxisymmetric_external_vmec_holdout_families_needed"
        ]
        == 1
    )
    assert {
        "absolute_train_holdout_report_passed",
        "holdout_mean_abs_relative_error",
        "minimum_total_independent_holdouts",
        "minimum_external_vmec_holdout_families",
        "minimum_nonaxisymmetric_external_vmec_holdout_families",
    }.issubset(requirements["blockers"])
    assert any(
        row["case"] == "Shaped tokamak external VMEC nonlinear t450 high-grid convergence"
        for row in requirements["required_nonlinear_cases"]
    )


def test_gap_report_tool_writes_replayable_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    paths = _write_inputs(tmp_path)
    out = tmp_path / "gap.png"

    assert (
        mod.main(
            [
                "--model-selection",
                str(paths["model_selection"]),
                "--train-holdout",
                str(paths["train_holdout"]),
                "--nonlinear-window-statistics",
                str(paths["window_stats"]),
                "--screening-skill",
                str(paths["screening_skill"]),
                "--dataset-sufficiency",
                str(paths["dataset"]),
                "--external-gate-glob",
                str(tmp_path / "external_*.json"),
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
    assert not out.with_suffix(".pdf").exists()
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["absolute_flux_promoted"] is False
    assert payload["screening_skill_status"]["holdout_screening_correlation_passed"] is False
    assert (
        payload["absolute_flux_promotion_requirements"]["claim_boundary"]
        .lower()
        .startswith("this section defines the evidence needed")
    )
    rows = list(csv.DictReader(out.with_suffix(".csv").open(encoding="utf-8")))
    assert {row["section"] for row in rows} >= {
        "admitted_holdouts",
        "excluded_candidates",
        "next_best_candidates",
        "training_references",
    }
