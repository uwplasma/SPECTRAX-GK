"""Tests for quasilinear artifact plotting and model-development reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from support.paths import load_artifact_tool



# Quasilinear holdout gap report assertions
def _load_tool_module():
    return load_artifact_tool("build_quasilinear_holdout_gap_report")


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


def _external_gate(
    path: Path,
    case: str,
    *,
    passed: bool,
    gates: list[dict],
    claim_level: str | None = None,
) -> Path:
    payload = {
        "case": case,
        "claim_level": claim_level
        or (
            "passed_grid_convergence_candidate_for_transport_holdout"
            if passed
            else "negative_grid_convergence_result_not_transport_validation"
        ),
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
    pass_gate = [
        _gate("common_window_max_relative_slope_per_time", 0.001, 0.002, passed=True)
    ]
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


def test_gap_report_preserves_claim_boundary_and_ranks_next_holdout(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    paths = _write_inputs(tmp_path)
    external_gates = [
        mod._load_external_gate(path)
        for path in sorted(tmp_path.glob("external_*.json"))
    ]

    report = mod.build_holdout_gap_report(
        model_selection=json.loads(
            paths["model_selection"].read_text(encoding="utf-8")
        ),
        train_holdout=json.loads(paths["train_holdout"].read_text(encoding="utf-8")),
        window_stats=json.loads(paths["window_stats"].read_text(encoding="utf-8")),
        external_gates=external_gates,
        dataset=json.loads(paths["dataset"].read_text(encoding="utf-8")),
        screening_skill=json.loads(
            paths["screening_skill"].read_text(encoding="utf-8")
        ),
    )

    assert report["claim_level"] == "holdout_gap_report_no_absolute_flux_promotion"
    assert report["absolute_flux_promoted"] is False
    assert report["promotion_gate"]["passed"] is False
    assert (
        "absolute_flux_predictor_not_promoted" in report["promotion_gate"]["blockers"]
    )
    assert report["summary"]["n_admitted_holdouts"] == 1
    assert report["summary"]["n_training_references"] == 1
    assert report["admitted_holdouts"][0]["absolute_relative_error"] == 0.5
    assert report["admitted_holdouts"][0]["gate_passed"] is True
    assert any(
        row["case"] == "kbm_nonlinear_window" for row in report["excluded_candidates"]
    )
    shaped = next(
        row
        for row in report["excluded_candidates"]
        if row["case"]
        == "Shaped tokamak external VMEC nonlinear t450 high-grid convergence"
    )
    assert shaped["geometry"] == "shaped_tokamak_external_vmec"
    assert shaped["status"] == "excluded_failed_external_gate"
    audit = next(
        row
        for row in report["excluded_candidates"]
        if row["case"]
        == "ITERModel external VMEC independent audit t450 high-grid convergence"
    )
    assert audit["geometry"] == "itermodel_external_vmec"
    assert audit["status"] == "excluded_same_family_training_audit"
    assert audit["eligible_for_next_candidate"] is False
    assert (
        report["next_best_candidates"][0]["status"]
        == "training_reference_not_independent_holdout"
    )
    assert (
        report["next_actual_nonlinear_holdout_needed"]["preferred_family"]
        == "itermodel_external_vmec"
    )
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
    assert (
        nearest["case"]
        == "ITERModel external VMEC nonlinear t250 high-grid convergence"
    )
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
        requirements["coverage_gap"]["additional_external_vmec_holdout_families_needed"]
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
        row["case"]
        == "Shaped tokamak external VMEC nonlinear t450 high-grid convergence"
        for row in requirements["required_nonlinear_cases"]
    )


def test_gap_report_records_raw_passed_qh_gate_with_bad_claim_as_negative_evidence(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    paths = _write_inputs(tmp_path)
    pass_gate = [
        _gate("common_window_max_relative_slope_per_time", 0.001, 0.002, passed=True)
    ]
    qh_gate = _external_gate(
        tmp_path / "external_qh_raw_passed.json",
        "nfp4 QH external VMEC nonlinear high-grid convergence",
        passed=True,
        gates=pass_gate,
        claim_level="finite_high_grid_long_nonlinear_feasibility_not_yet_transport_validation",
    )
    external_gates = [mod._load_external_gate(qh_gate)]

    report = mod.build_holdout_gap_report(
        model_selection=json.loads(
            paths["model_selection"].read_text(encoding="utf-8")
        ),
        train_holdout=json.loads(paths["train_holdout"].read_text(encoding="utf-8")),
        window_stats=json.loads(paths["window_stats"].read_text(encoding="utf-8")),
        external_gates=external_gates,
        dataset=json.loads(paths["dataset"].read_text(encoding="utf-8")),
        screening_skill=json.loads(
            paths["screening_skill"].read_text(encoding="utf-8")
        ),
    )

    qh = next(
        row
        for row in report["excluded_candidates"]
        if row["case"] == "nfp4 QH external VMEC nonlinear high-grid convergence"
    )
    assert qh["status"] == "excluded_negative_external_evidence"
    assert qh["gate_passed"] is False
    assert qh["raw_gate_passed"] is True
    assert qh["promotion_gate_passed"] is True
    assert qh["claim_level_acceptable"] is False
    assert qh["negative_evidence"] is True
    assert qh["eligible_for_next_candidate"] is False
    assert qh["admission_blockers"] == ["claim_level_not_acceptable"]
    assert report["summary"]["n_external_gates_passed"] == 0
    assert report["summary"]["n_external_negative_evidence"] == 1
    assert all(
        row["case"] != "nfp4 QH external VMEC nonlinear high-grid convergence"
        for row in report["next_best_candidates"]
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
    assert (
        payload["screening_skill_status"]["holdout_screening_correlation_passed"]
        is False
    )
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


# Quasilinear calibration assertions
def test_plot_quasilinear_calibration_writes_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_calibration")
    report = {
        "kind": "quasilinear_calibration_report",
        "claim_level": "training_or_audit_only",
        "passed": False,
        "holdout_mean_rel_gate": 0.35,
        "observed_floor": 1.0e-12,
        "points": [
            {
                "case": "cyclone",
                "split": "audit",
                "predicted_heat_flux": 0.1,
                "observed_heat_flux": 1.0,
                "observed_heat_flux_std": 0.2,
            }
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    paths = mod.write_calibration_figure(
        report_path, out=tmp_path / "calibration.png", title="QL audit"
    )

    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    meta = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert meta["claim_level"] == "training_or_audit_only"
    assert meta["n_points"] == 1
    assert meta["mean_abs_relative_error"] == 0.9


def test_plot_quasilinear_calibration_handles_zero_prediction_on_log_axes(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_calibration")
    report = {
        "kind": "quasilinear_calibration_report",
        "claim_level": "calibration_dataset",
        "passed": False,
        "holdout_mean_rel_gate": 0.35,
        "observed_floor": 1.0e-12,
        "points": [
            {
                "case": "train",
                "split": "train",
                "predicted_heat_flux": 0.2,
                "observed_heat_flux": 0.3,
                "observed_heat_flux_std": 0.02,
            },
            {
                "case": "stable_holdout",
                "split": "holdout",
                "predicted_heat_flux": 0.0,
                "observed_heat_flux": 0.4,
                "observed_heat_flux_std": 0.5,
            },
        ],
    }
    report_path = tmp_path / "zero_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    paths = mod.write_calibration_figure(
        report_path, out=tmp_path / "zero_calibration.png", title="QL audit"
    )

    assert Path(paths["png"]).exists()
    meta = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert meta["n_points"] == 2
    assert meta["max_abs_relative_error"] == 1.0


def test_plot_quasilinear_calibration_uses_log_error_axis_for_wide_errors() -> None:
    mod = load_artifact_tool("plot_quasilinear_calibration")
    report = {
        "kind": "quasilinear_calibration_report",
        "claim_level": "calibration_dataset",
        "passed": False,
        "holdout_mean_rel_gate": 0.35,
        "observed_floor": 1.0e-12,
        "points": [
            {
                "case": "near",
                "split": "train",
                "predicted_heat_flux": 1.0,
                "observed_heat_flux": 1.0,
            },
            {
                "case": "far",
                "split": "holdout",
                "predicted_heat_flux": 1000.0,
                "observed_heat_flux": 1.0,
            },
        ],
    }

    fig = mod.calibration_figure(report, title="wide errors")
    try:
        assert fig.axes[1].get_xscale() == "log"
    finally:
        mod.plt.close(fig)


# Candidate regularization sweep assertions
def test_regularization_sweep_locks_tracked_near_miss() -> None:
    mod = load_artifact_tool("plot_quasilinear_candidate_regularization_sweep")

    report = mod.score_regularization_sweep(lambdas=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0))

    assert report["kind"] == "quasilinear_candidate_regularization_sweep"
    assert (
        report["claim_level"]
        == "spectral_envelope_regularization_audit_not_runtime_flux_predictor"
    )
    assert report["case_count"] == 12
    assert report["holdout_count"] == 10
    assert report["best_lambda"] == 0.5
    assert 0.68 < report["best_mean_abs_relative_error"] < 0.70
    assert 0.76 < report["best_holdout_mean_abs_relative_error"] < 0.77
    assert report["best_mean_abs_relative_error"] > report["transport_gate"]
    assert report["promotion_gate"]["passed"] is False
    assert report["promotion_gate"]["accepted_lambdas"] == []
    assert report["promotion_gate"]["blockers"] == [
        "best_regularization_transport_error_above_gate"
    ]
    assert len(report["rows"]) == 6


def test_regularization_sweep_writes_sidecars_and_cli_fails_closed(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_candidate_regularization_sweep")
    report = mod.score_regularization_sweep(lambdas=(0.2, 0.3))
    paths = mod.write_regularization_sweep_figure(
        report,
        out=tmp_path / "regularization.png",
        title="regularization",
        dpi=80,
        write_pdf=False,
    )

    assert Path(paths["png"]).exists()
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    assert "pdf" not in paths
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["promotion_gate"]["passed"] is False
    assert Path(paths["csv"]).read_text(encoding="utf-8").startswith("lambda,")

    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            str(
                root
                / "tools"
                / "artifacts"
                / "plot_quasilinear_candidate_regularization_sweep.py"
            ),
            "--out",
            str(tmp_path / "cli_regularization.png"),
            "--lambdas",
            "0.2,0.3",
            "--no-pdf",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert "promotion_passed=False" in completed.stdout
    assert (tmp_path / "cli_regularization.json").exists()


# Candidate uncertainty assertions
def _candidate_uncertainty_write_case(
    tmp_path: Path, name: str, *, observed: float, weight: float
) -> tuple[Path, Path]:
    spectrum = tmp_path / f"{name}_ql.csv"
    spectrum.write_text(
        "ky,gamma,kperp_eff2,heat_flux_weight_total,saturated_heat_flux_total\n"
        f"0.1,0.1,0.5,{weight},0.0\n"
        f"0.2,0.1,1.0,{0.5 * weight},0.0\n",
        encoding="utf-8",
    )
    diag = tmp_path / f"{name}_diag.csv"
    diag.write_text(f"t,heat_flux\n0.0,{observed}\n1.0,{observed}\n", encoding="utf-8")
    summary = tmp_path / f"{name}_summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": name,
                "spectrax": str(diag),
                "gate_report": {"case": name, "passed": True, "gates": []},
            }
        ),
        encoding="utf-8",
    )
    return spectrum, summary


def test_candidate_uncertainty_report_and_figure_are_replayable(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_candidate_uncertainty")
    cases = []
    for name, observed, weight in [("a", 3.0, 1.0), ("b", 6.0, 2.0), ("c", 9.0, 3.0)]:
        spectrum, summary = _candidate_uncertainty_write_case(
            tmp_path, name, observed=observed, weight=weight
        )
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, None))

    report = mod.build_candidate_uncertainty_report(
        tuple(cases), candidates=("linear_weight",)
    )
    paths = mod.write_candidate_uncertainty_figure(
        report,
        out=tmp_path / "candidate.png",
        title="Candidate",
        dpi=80,
        write_pdf=False,
    )

    assert report["kind"] == "quasilinear_candidate_uncertainty_report"
    assert report["input_validation"]["passed"] is True
    assert "linear_weight" in report["candidates"]
    assert report["promotion_gate"]["requires_interval_coverage"] is True
    assert Path(paths["png"]).exists()
    assert "pdf" not in paths
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["claim_level"] == "candidate_model_development_not_runtime_option"


def test_linear_state_ridge_candidate_reports_under_sampled_gate(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_candidate_uncertainty")
    cases = []
    for name, observed, weight in [
        ("a", 3.0, 1.0),
        ("b", 6.0, 2.0),
        ("c", 7.0, 2.4),
    ]:
        spectrum, summary = _candidate_uncertainty_write_case(
            tmp_path, name, observed=observed, weight=weight
        )
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, None))

    report = mod.build_candidate_uncertainty_report(
        tuple(cases), candidates=("linear_state_ridge",)
    )
    ridge = report["candidates"]["linear_state_ridge"]

    assert ridge["promotion_eligible"] is False
    assert ridge["eligibility_failures"] == ["insufficient_train_to_parameter_ratio"]
    assert report["promotion_gate"]["accepted_candidates"] == []
    assert report["promotion_gate"]["requires_candidate_eligibility"] is True
    assert ridge["rows"][0]["feature_names"] == list(mod.STATE_FEATURE_NAMES)


def test_candidate_uncertainty_parallel_workers_match_serial(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_candidate_uncertainty")
    cases = []
    for name, observed, weight in [
        ("a", 3.0, 1.0),
        ("b", 6.0, 2.0),
        ("c", 9.0, 3.0),
        ("d", 12.0, 4.0),
    ]:
        spectrum, summary = _candidate_uncertainty_write_case(
            tmp_path, name, observed=observed, weight=weight
        )
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, None))

    serial = mod.build_candidate_uncertainty_report(
        tuple(cases), candidates=("linear_weight",), workers=1
    )
    parallel = mod.build_candidate_uncertainty_report(
        tuple(cases), candidates=("linear_weight",), workers=3
    )

    assert parallel["parallel"]["workers"] == 3
    assert (
        parallel["null_training_mean_baseline"] == serial["null_training_mean_baseline"]
    )
    assert parallel["candidates"] == serial["candidates"]
    assert parallel["promotion_gate"] == serial["promotion_gate"]


def test_candidate_uncertainty_negative_controls_and_schema_guards(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_candidate_uncertainty")
    good_spectrum, good_summary = _candidate_uncertainty_write_case(
        tmp_path, "good", observed=3.0, weight=1.0
    )
    bad_spectrum = tmp_path / "bad_ql.csv"
    bad_spectrum.write_text(
        "ky,kperp_eff2,heat_flux_weight_total,saturated_heat_flux_total\n0.1,0.5,1.0,0.0\n",
        encoding="utf-8",
    )
    bad_summary = tmp_path / "bad_summary.json"
    bad_diag = tmp_path / "bad_diag.csv"
    bad_diag.write_text("t,heat_flux\n0.0,3.0\n1.0,3.0\n", encoding="utf-8")
    bad_summary.write_text(
        json.dumps(
            {"case": "bad", "spectrax": str(bad_diag), "gate_report": {"passed": True}}
        ),
        encoding="utf-8",
    )
    cases = (
        mod.SaturationCase(
            "good", "holdout", "good", good_spectrum, good_summary, None
        ),
        mod.SaturationCase("bad", "holdout", "bad", bad_spectrum, bad_summary, None),
    )

    with pytest.raises(ValueError, match="required column 'gamma'"):
        mod.build_candidate_uncertainty_report(
            cases, candidates=("linear_state_ridge",)
        )
    with pytest.raises(ValueError, match="unknown candidate"):
        mod.build_candidate_uncertainty_report(
            cases[:1], candidates=("not_a_candidate",)
        )


def test_candidate_uncertainty_can_be_run_as_unvalidated_development_audit(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_candidate_uncertainty")
    cases = []
    for name, observed, weight in [("a", 3.0, 1.0), ("b", 6.0, 2.0)]:
        spectrum, summary = _candidate_uncertainty_write_case(
            tmp_path, name, observed=observed, weight=weight
        )
        payload = json.loads(summary.read_text(encoding="utf-8"))
        payload["gate_report"]["passed"] = False
        summary.write_text(json.dumps(payload), encoding="utf-8")
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, None))

    report = mod.build_candidate_uncertainty_report(
        tuple(cases),
        candidates=("linear_weight",),
        require_validated_inputs=False,
    )

    assert report["input_validation"]["passed"] is None
    assert report["claim_level"] == "candidate_model_development_not_runtime_option"
    assert report["promotion_gate"]["requires_candidate_eligibility"] is True


def test_tracked_candidate_uncertainty_sidecar_is_fail_closed_near_miss() -> None:
    """Lock the scoped quasilinear model-development near miss to the artifact."""

    root = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (root / "docs/_static/quasilinear_candidate_uncertainty.json").read_text(
            encoding="utf-8"
        )
    )
    gate = payload["promotion_gate"]
    candidates = payload["candidates"]
    spectral = candidates["spectral_envelope_ridge"]
    linear = candidates["linear_weight"]
    state = candidates["linear_state_ridge"]
    null = payload["null_training_mean_baseline"]

    assert payload["claim_level"] == "candidate_model_development_not_runtime_option"
    assert gate["passed"] is False
    assert gate["accepted_candidates"] == []
    assert spectral["promotion_eligible"] is True
    assert (
        spectral["mean_abs_relative_error"] > gate["transport_mean_relative_error_gate"]
    )
    assert spectral["mean_abs_relative_error"] < linear["mean_abs_relative_error"]
    assert spectral["mean_abs_relative_error"] < null["mean_abs_relative_error"]
    assert spectral["prediction_interval_coverage"] >= gate["interval_coverage_gate"]
    assert state["promotion_eligible"] is True
    assert state["eligibility_failures"] == []
    assert state["mean_abs_relative_error"] > null["mean_abs_relative_error"]
    assert state["mean_abs_relative_error"] > linear["mean_abs_relative_error"]
    assert state["mean_abs_relative_error"] > spectral["mean_abs_relative_error"]


# Dataset sufficiency assertions
def _dataset_sufficiency_write_case(
    tmp_path: Path, name: str, *, gate_case: str, split: str, geometry: str
):
    spectrum = tmp_path / f"{name}_spectrum.csv"
    spectrum.write_text(
        "ky,gamma,kperp_eff2,heat_flux_weight_total\n0.1,0.2,1.0,2.0\n",
        encoding="utf-8",
    )
    summary = tmp_path / f"{name}_summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": gate_case,
                "gate_report": {"case": gate_case, "passed": True, "gates": []},
            }
        ),
        encoding="utf-8",
    )
    shape_gate = tmp_path / f"{name}_shape.json"
    shape_gate.write_text(json.dumps({"passed": True}), encoding="utf-8")
    return spectrum, summary, shape_gate, split, geometry


def test_dataset_sufficiency_blocks_under_sampled_quasilinear_promotion(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_dataset_sufficiency")
    cases = []
    for name, gate_case, split, geometry in [
        ("cyclone", "cyclone_nonlinear_long_window", "train", "cyclone"),
        ("miller", "cyclone_miller_nonlinear_window", "holdout", "cyclone_miller"),
        ("hsx", "hsx_nonlinear_window", "holdout", "hsx"),
        ("w7x", "w7x_nonlinear_window", "holdout", "w7x"),
    ]:
        spectrum, summary, shape_gate, split, geometry = (
            _dataset_sufficiency_write_case(
                tmp_path,
                name,
                gate_case=gate_case,
                split=split,
                geometry=geometry,
            )
        )
        cases.append(
            mod.SaturationCase(name, split, geometry, spectrum, summary, shape_gate)
        )
    nonlinear_index = tmp_path / "nonlinear_index.json"
    nonlinear_index.write_text(
        json.dumps(
            {
                "cases": [
                    "cyclone_nonlinear_long_window",
                    "cyclone_miller_nonlinear_window",
                    "hsx_nonlinear_window",
                    "w7x_nonlinear_window",
                    "kbm_nonlinear_window",
                ],
                "case_gate_passed": {"kbm_nonlinear_window": True},
                "case_gate_thresholds": {"kbm_nonlinear_window": 0.02},
            }
        ),
        encoding="utf-8",
    )
    candidate_gate = tmp_path / "candidate.json"
    candidate_gate.write_text(
        json.dumps({"kind": "candidate", "promotion_gate": {"passed": False}}),
        encoding="utf-8",
    )
    saturation_gate = tmp_path / "saturation.json"
    saturation_gate.write_text(
        json.dumps({"kind": "saturation", "promotion_gate": {"passed": False}}),
        encoding="utf-8",
    )

    report = mod.build_dataset_sufficiency_report(
        tuple(cases),
        nonlinear_index=nonlinear_index,
        candidate_gate=candidate_gate,
        saturation_gate=saturation_gate,
    )

    assert report["kind"] == "quasilinear_dataset_sufficiency"
    assert report["input_validation"]["passed"] is True
    assert report["promotion_gate"]["passed"] is False
    assert "minimum_total_electrostatic_cases" in report["promotion_gate"]["blockers"]
    assert "minimum_explicit_train_geometries" in report["promotion_gate"]["blockers"]
    assert (
        "downstream_candidate_skill_gates_not_passed"
        in report["promotion_gate"]["blockers"]
    )
    assert report["requirements"]["current_total_cases"] == 4
    assert report["candidate_requirements"][0]["data_volume_passed"] is True
    assert report["candidate_requirements"][-1]["candidate"] == "linear_state_ridge"
    assert report["candidate_requirements"][-1]["data_volume_passed"] is False
    assert (
        report["excluded_validated_nonlinear_cases"][0]["case"]
        == "kbm_nonlinear_window"
    )
    assert (
        "electromagnetic" in report["excluded_validated_nonlinear_cases"][0]["reason"]
    )


def test_dataset_sufficiency_writes_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_dataset_sufficiency")
    cases = []
    for idx, split in enumerate(
        ["train", "train", "holdout", "holdout", "holdout", "holdout"]
    ):
        spectrum, summary, shape_gate, split, geometry = (
            _dataset_sufficiency_write_case(
                tmp_path,
                f"case{idx}",
                gate_case=f"case{idx}_gate",
                split=split,
                geometry=f"geom{idx}",
            )
        )
        cases.append(
            mod.SaturationCase(
                f"case{idx}", split, geometry, spectrum, summary, shape_gate
            )
        )

    report = mod.build_dataset_sufficiency_report(
        tuple(cases),
        nonlinear_index=None,
        candidate_gate=None,
        saturation_gate=None,
    )
    paths = mod.write_dataset_sufficiency_figure(
        report,
        out=tmp_path / "dataset.png",
        title="Dataset sufficiency",
        dpi=80,
        write_pdf=False,
    )

    assert Path(paths["png"]).exists()
    assert Path(paths["json"]).exists()
    assert "pdf" not in paths
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert (
        payload["claim_level"]
        == "scoped_low_parameter_candidate_promotion_not_runtime_option"
    )


# Model-selection status assertions
def _model_selection_write_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
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


def _model_selection_write_optimized_audit(
    tmp_path: Path,
) -> Path:
    audit = tmp_path / "optimized_audit.json"
    audit.write_text(
        json.dumps(
            {
                "kind": "production_nonlinear_turbulent_flux_optimization_guard",
                "claim_level": "production_nonlinear_optimization_promoted_by_replicated_transport_windows",
                "passed": True,
                "production_nonlinear_optimization_promoted": True,
                "promotion_gate": {"passed": True, "blockers": []},
                "summary": {
                    "qualifying_optimized_equilibrium_ensembles": 1,
                },
                "optimized_equilibrium_artifacts": [
                    {
                        "path": "optimized_equilibrium_gate.json",
                        "optimized_equilibrium_marker": True,
                        "qualifies_for_production_optimization": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return audit


def test_model_selection_status_tool_writes_replayable_artifacts(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_model_selection_status")
    dataset, candidate, calibration = _model_selection_write_inputs(tmp_path)
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


def test_model_selection_status_tool_can_require_optimized_equilibrium_audit(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_model_selection_status")
    dataset, candidate, calibration = _model_selection_write_inputs(tmp_path)
    audit = _model_selection_write_optimized_audit(tmp_path)
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
                "--optimized-equilibrium-nonlinear-audit",
                str(audit),
                "--require-optimized-equilibrium-nonlinear-audit",
                "--out",
                str(out),
                "--no-pdf",
                "--dpi",
                "80",
            ]
        )
        == 0
    )

    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert (
        payload["absolute_flux_promotion"]["honest_status"]
        == "scoped_candidate_with_audited_optimized_equilibrium_evidence_not_universal_absolute_flux"
    )
    assert (
        payload["absolute_flux_promotion"]["universal_absolute_flux_promoted"] is False
    )


# Saturation-rule sweep assertions
def _saturation_rule_write_case(
    tmp_path: Path, name: str, *, observed: float, gamma_sign: float = 1.0
) -> tuple[Path, Path]:
    spectrum = tmp_path / f"{name}_ql.csv"
    spectrum.write_text(
        "ky,gamma,kperp_eff2,heat_flux_weight_total,saturated_heat_flux_total\n"
        f"0.1,{0.2 * gamma_sign},0.5,2.0,{0.8 if gamma_sign > 0.0 else 0.0}\n"
        f"0.2,{0.1 * gamma_sign},1.0,1.0,{0.1 if gamma_sign > 0.0 else 0.0}\n",
        encoding="utf-8",
    )
    diag = tmp_path / f"{name}_diag.csv"
    diag.write_text(f"t,heat_flux\n0.0,{observed}\n1.0,{observed}\n", encoding="utf-8")
    summary = tmp_path / f"{name}_summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": name,
                "spectrax": str(diag),
                "gate_report": {"case": name, "passed": True, "gates": []},
            }
        ),
        encoding="utf-8",
    )
    return spectrum, summary


def test_saturation_rule_sweep_fits_train_scale_and_scores_holdout(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_saturation_rule_sweep")
    train_spectrum, train_summary = _saturation_rule_write_case(
        tmp_path, "train", observed=9.0
    )
    holdout_spectrum, holdout_summary = _saturation_rule_write_case(
        tmp_path, "holdout", observed=4.5, gamma_sign=-1.0
    )
    cases = (
        mod.SaturationCase(
            case="train",
            split="train",
            geometry="cyclone",
            spectrum=train_spectrum,
            nonlinear_summary=train_summary,
        ),
        mod.SaturationCase(
            case="holdout",
            split="holdout",
            geometry="hsx",
            spectrum=holdout_spectrum,
            nonlinear_summary=holdout_summary,
        ),
    )

    report = mod.build_saturation_rule_sweep(cases)

    assert report["claim_level"] == "model_comparison_not_validated_transport"
    assert report["input_validation"]["passed"] is True
    assert report["holdout_relative_error_gate"] == pytest.approx(0.35)
    assert report["any_rule_holdout_gate_passed"] is False
    assert report["cases"][0]["shape_gate_status"] == "missing"
    assert report["cases"][0]["shape_passed"] is None
    assert report["rules"]["positive_mixing_length"]["scale"] == pytest.approx(10.0)
    assert report["rules"]["positive_mixing_length"]["holdout_gate_passed"] is False
    assert report["rules"]["positive_mixing_length"]["predicted_heat_flux"][
        0
    ] == pytest.approx(9.0)
    assert report["rules"]["positive_mixing_length"]["predicted_heat_flux"][
        1
    ] == pytest.approx(0.0)
    assert report["rules"]["linear_weight"]["scale"] == pytest.approx(3.0)
    assert report["rules"]["linear_weight"]["predicted_heat_flux"][1] == pytest.approx(
        9.0
    )
    assert report["null_training_mean_baseline"][
        "predicted_heat_flux"
    ] == pytest.approx([9.0, 9.0])
    assert report["null_training_mean_baseline"][
        "holdout_mean_abs_relative_error"
    ] == pytest.approx(1.0)
    assert report["promotion_gate"]["passed"] is False
    assert report["promotion_gate"]["accepted_rules"] == []


def test_saturation_rule_sweep_parallel_workers_match_serial(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_saturation_rule_sweep")
    cases = []
    for name, split, observed in [
        ("train", "train", 9.0),
        ("holdout_a", "holdout", 4.5),
        ("holdout_b", "holdout", 6.0),
    ]:
        spectrum, summary = _saturation_rule_write_case(
            tmp_path, name, observed=observed
        )
        cases.append(
            mod.SaturationCase(
                case=name,
                split=split,
                geometry=name,
                spectrum=spectrum,
                nonlinear_summary=summary,
            )
        )

    serial = mod.build_saturation_rule_sweep(tuple(cases), workers=1)
    parallel = mod.build_saturation_rule_sweep(tuple(cases), workers=2)

    assert parallel["parallel"]["workers"] == 2
    assert parallel["cases"] == serial["cases"]
    assert parallel["rules"] == serial["rules"]
    assert parallel["promotion_gate"] == serial["promotion_gate"]


def test_saturation_rule_sweep_writes_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_saturation_rule_sweep")
    spectrum, summary = _saturation_rule_write_case(tmp_path, "train", observed=9.0)
    cases = (
        mod.SaturationCase(
            case="train",
            split="train",
            geometry="cyclone",
            spectrum=spectrum,
            nonlinear_summary=summary,
        ),
    )
    report = mod.build_saturation_rule_sweep(cases)

    paths = mod.write_saturation_rule_sweep_figure(
        report, out=tmp_path / "sweep.png", title="Sweep"
    )

    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["kind"] == "quasilinear_saturation_rule_sweep"
    assert "null_training_mean_baseline" in payload
    assert "promotion_gate" in payload
    assert payload["cases"][0]["shape_gate_status"] == "missing"


def test_saturation_rule_sweep_rejects_failed_nonlinear_summary_gate(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_saturation_rule_sweep")
    spectrum, summary = _saturation_rule_write_case(tmp_path, "train", observed=9.0)
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["gate_report"]["passed"] = False
    summary.write_text(json.dumps(data), encoding="utf-8")
    cases = (
        mod.SaturationCase(
            case="train",
            split="train",
            geometry="cyclone",
            spectrum=spectrum,
            nonlinear_summary=summary,
        ),
    )

    with pytest.raises(ValueError, match="unvalidated nonlinear train/holdout input"):
        mod.build_saturation_rule_sweep(cases)


def test_saturation_rule_sweep_development_mode_and_input_schema_guards(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_saturation_rule_sweep")
    spectrum, summary = _saturation_rule_write_case(tmp_path, "holdout", observed=4.0)
    failed_payload = json.loads(summary.read_text(encoding="utf-8"))
    failed_payload["gate_report"]["passed"] = False
    summary.write_text(json.dumps(failed_payload), encoding="utf-8")
    cases = (
        mod.SaturationCase(
            case="holdout",
            split="holdout",
            geometry="hsx",
            spectrum=spectrum,
            nonlinear_summary=summary,
        ),
    )

    report = mod.nonlinear_input_validation_report(cases, required_splits=("train",))
    assert report["passed"] is True
    assert report["cases"][0]["reason"] == "not required split"

    with pytest.raises(ValueError, match="at least one train case"):
        mod.build_saturation_rule_sweep(cases, require_validated_inputs=False)

    bad_spectrum = tmp_path / "bad.csv"
    bad_spectrum.write_text(
        "ky,gamma,heat_flux_weight_total\n0.1,0.2,1.0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="required column 'kperp_eff2'"):
        mod.raw_rule_estimates(bad_spectrum)


def test_saturation_rule_sweep_records_shape_gate_metadata(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_saturation_rule_sweep")
    shape = tmp_path / "shape.json"
    shape.write_text(
        json.dumps(
            {
                "kind": "quasilinear_spectrum_shape_gate",
                "passed": True,
                "total_variation_distance": 0.12,
                "cosine_similarity": 0.97,
                "tv_gate": 0.2,
                "cosine_gate": 0.95,
            }
        ),
        encoding="utf-8",
    )

    payload = mod._shape_payload(shape)

    assert payload["shape_gate_status"] == "passed"
    assert payload["shape_tv_gate"] == pytest.approx(0.2)
    assert payload["shape_cosine_gate"] == pytest.approx(0.95)
    assert (
        mod._artifact_path(mod.ROOT / "docs/_static/example.json")
        == "docs/_static/example.json"
    )


# Screening-skill assertions
def test_screening_skill_keeps_spectral_envelope_fail_closed() -> None:
    module = load_artifact_tool("plot_quasilinear_screening_skill")
    report = module.build_report()
    models = {row["model"]: row for row in report["models"]}

    assert report["kind"] == "quasilinear_screening_skill"
    assert (
        report["claim_level"]
        == "screening_correlation_model_development_not_absolute_flux_promotion"
    )
    assert report["gates"]["accepted_screening_models"] == []
    assert report["gates"]["accepted_holdout_screening_models"] == []
    assert report["gates"]["best_screening_model"] == "spectral_envelope_ridge"
    assert report["gates"]["best_holdout_screening_model"] == "spectral_envelope_ridge"
    assert report["gates"]["mean_error_gate_models"] == []
    assert report["gates"]["accepted_absolute_flux_models"] == []
    assert report["gates"]["absolute_flux_promotion_passed"] is False
    assert report["gates"]["screening_correlation_passed"] is False
    assert report["gates"]["holdout_screening_correlation_passed"] is False

    spectral = models["spectral_envelope_ridge"]
    assert spectral["screening_gate_passed"] is False
    assert spectral["holdout_screening_gate_passed"] is False
    assert 0.62 < spectral["spearman"] < 0.75
    assert 0.68 < spectral["pairwise_order_accuracy"] < 0.75
    assert 0.55 < spectral["holdout_spearman"] < 0.75
    assert 0.65 < spectral["holdout_pairwise_order_accuracy"] < 0.75
    assert spectral["holdout_mean_abs_relative_error"] > 0.35

    assert models["positive_mixing_length"]["screening_gate_passed"] is False
    assert models["linear_weight"]["screening_gate_passed"] is False
    assert models["absolute_growth_mixing_length"]["screening_gate_passed"] is False


def test_screening_skill_writer_creates_sidecars(tmp_path: Path) -> None:
    module = load_artifact_tool("plot_quasilinear_screening_skill")
    report = module.build_report()
    paths = module.write_figure(
        report, out=tmp_path / "screening.png", title="test", dpi=80
    )

    for key in ("png", "pdf", "json", "csv"):
        assert Path(paths[key]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["gates"]["best_screening_model"] == "spectral_envelope_ridge"
    assert payload["gates"]["best_holdout_screening_model"] == "spectral_envelope_ridge"
    assert Path(paths["csv"]).read_text(encoding="utf-8").startswith("model,label")


# Shape-aware saturation assertions
def _shape_aware_write_case(
    tmp_path: Path,
    name: str,
    *,
    observed: float,
    nonlinear_dist: tuple[float, float, float],
) -> tuple[Path, Path, Path]:
    spectrum = tmp_path / f"{name}_ql.csv"
    spectrum.write_text(
        "ky,gamma,kperp_eff2,heat_flux_weight_total,saturated_heat_flux_total\n"
        "0.1,0.1,0.5,1.0,0.2\n"
        "0.2,0.1,0.5,1.0,0.2\n"
        "0.4,0.1,0.5,1.0,0.2\n",
        encoding="utf-8",
    )
    diag = tmp_path / f"{name}_diag.csv"
    diag.write_text(f"t,heat_flux\n0.0,{observed}\n1.0,{observed}\n", encoding="utf-8")
    summary = tmp_path / f"{name}_summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": name,
                "spectrax": str(diag),
                "gate_report": {"case": name, "passed": True, "gates": []},
            }
        ),
        encoding="utf-8",
    )
    ql_dist = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    shape = tmp_path / f"{name}_shape.json"
    shape.write_text(
        json.dumps(
            {
                "kind": "quasilinear_spectrum_shape_gate",
                "passed": True,
                "ky": [0.1, 0.2, 0.4],
                "quasilinear_distribution": ql_dist,
                "nonlinear_distribution": nonlinear_dist,
                "total_variation_distance": 0.1,
                "cosine_similarity": 0.98,
                "tv_gate": 0.2,
                "cosine_gate": 0.95,
            }
        ),
        encoding="utf-8",
    )
    return spectrum, summary, shape


def test_fit_power_law_shape_exponent_uses_case_intercepts(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_shape_aware_saturation")
    cases = []
    for name, dist in [
        ("a", (0.2, 0.3, 0.5)),
        ("b", (0.1, 0.3, 0.6)),
    ]:
        spectrum, summary, shape = _shape_aware_write_case(
            tmp_path, name, observed=1.0, nonlinear_dist=dist
        )
        cases.append(mod.SaturationCase(name, "train", name, spectrum, summary, shape))

    fit = mod.fit_power_law_shape_exponent(tuple(cases))

    assert fit["n_samples"] == 6
    assert fit["exponent"] > 0.0
    assert set(fit["used_cases"]) == {"a", "b"}


def test_shape_aware_report_and_figure_are_replayable(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_shape_aware_saturation")
    cases = []
    for name, observed, dist in [
        ("a", 1.0, (0.2, 0.3, 0.5)),
        ("b", 1.2, (0.2, 0.3, 0.5)),
        ("c", 0.8, (0.1, 0.3, 0.6)),
    ]:
        spectrum, summary, shape = _shape_aware_write_case(
            tmp_path, name, observed=observed, nonlinear_dist=dist
        )
        cases.append(
            mod.SaturationCase(name, "holdout", name, spectrum, summary, shape)
        )

    report = mod.build_shape_aware_saturation_report(tuple(cases))
    paths = mod.write_shape_aware_saturation_figure(
        report, out=tmp_path / "shape.png", title="Shape"
    )

    assert report["kind"] == "quasilinear_shape_aware_saturation_report"
    assert report["input_validation"]["passed"] is True
    assert len(report["leave_one_out"]) == 3
    assert report["holdout_relative_error_gate"] == pytest.approx(0.35)
    assert "null_training_mean_mean_abs_relative_error" in report["metrics"]
    assert "shape_aware_all_case_gate_passed" in report["metrics"]
    assert all(row["shape_gate_status"] == "passed" for row in report["cases"])
    assert report["promotion_gate"]["requires_beating_training_mean_null"] is True
    assert all(
        "null_training_mean_predicted_heat_flux" in row
        for row in report["leave_one_out"]
    )
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["claim_level"] == "leave_one_geometry_out_model_development"


def test_observed_flux_falls_back_to_tracked_calibration_points(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_shape_aware_saturation")
    case = mod.SaturationCase(
        "cyclone_long_window",
        "train",
        "cyclone",
        tmp_path / "unused_spectrum.csv",
        tmp_path / "missing_summary.json",
        None,
    )

    observed, observed_std = mod._observed_flux(case)

    assert observed == pytest.approx(6.6689961798985795)
    assert observed_std == pytest.approx(0.6486446261822355)


def test_shape_aware_report_rejects_missing_shape_gate(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_shape_aware_saturation")
    spectrum, summary, _shape = _shape_aware_write_case(
        tmp_path, "a", observed=1.0, nonlinear_dist=(0.2, 0.3, 0.5)
    )
    case = mod.SaturationCase("a", "train", "a", spectrum, summary, None)

    with pytest.raises(ValueError, match="missing a tracked shape-gate"):
        mod.build_shape_aware_saturation_report((case,))


def test_shape_aware_report_rejects_incomplete_shape_gate(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_shape_aware_saturation")
    spectrum, summary, shape = _shape_aware_write_case(
        tmp_path, "a", observed=1.0, nonlinear_dist=(0.2, 0.3, 0.5)
    )
    payload = json.loads(shape.read_text(encoding="utf-8"))
    payload.pop("tv_gate")
    shape.write_text(json.dumps(payload), encoding="utf-8")
    case = mod.SaturationCase("a", "train", "a", spectrum, summary, shape)

    with pytest.raises(ValueError, match="tv_gate"):
        mod.build_shape_aware_saturation_report((case,))


# Spectrum plotting assertions
def _spectrum_write_spectrum(
    path: Path,
) -> None:
    path.write_text(
        "\n".join(
            [
                "ky,gamma,omega,kperp_eff2,heat_flux_weight_total,particle_flux_weight_total,amplitude2,saturated_heat_flux_total,saturated_particle_flux_total",
                "0.2,0.1,-0.4,0.8,1.2,0.1,nan,nan,nan",
                "0.3,0.2,-0.5,1.0,1.5,0.2,0.4,0.6,0.08",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_quasilinear_spectrum_requires_columns(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_diagnostics")
    path = tmp_path / "spectrum.csv"
    _spectrum_write_spectrum(path)
    data = mod.load_quasilinear_spectrum(path)
    np.testing.assert_allclose(data["ky"], [0.2, 0.3])

    bad = tmp_path / "bad.csv"
    bad.write_text("ky,gamma\n0.2,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.load_quasilinear_spectrum(bad)


def test_write_quasilinear_spectrum_figure(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_diagnostics")
    path = tmp_path / "spectrum.csv"
    _spectrum_write_spectrum(path)
    out = tmp_path / "ql_spectrum.png"

    paths = mod.write_quasilinear_spectrum_figure(
        path, out=out, title="Test QL Spectrum"
    )

    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["json"]).exists()


# Spectrum-shape gate assertions
def _spectrum_shape_gate_write_netcdf(
    path: Path,
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", 3)
        root.createDimension("s", 1)
        root.createDimension("ky", 3)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = np.asarray([0.0, 1.0, 2.0])
        grids.createVariable("ky", "f8", ("ky",))[:] = np.asarray([0.1, 0.2, 0.3])
        diagnostics = root.createGroup("Diagnostics")
        values = np.asarray(
            [
                [[1.0, 2.0, 3.0]],
                [[2.0, 4.0, 6.0]],
                [[3.0, 6.0, 9.0]],
            ]
        )
        diagnostics.createVariable("HeatFlux_kyst", "f8", ("time", "s", "ky"))[:] = (
            values
        )


def test_quasilinear_spectrum_shape_gate_writes_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_diagnostics")
    spectrum = tmp_path / "ql.csv"
    np.savetxt(
        spectrum,
        np.asarray([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]]),
        delimiter=",",
        header="ky,heat_flux_weight_total",
        comments="",
    )
    nonlinear = tmp_path / "nl.nc"
    _spectrum_shape_gate_write_netcdf(nonlinear)

    report = mod.build_spectrum_shape_report(
        spectrum_csv=spectrum,
        nonlinear_netcdf=nonlinear,
        time_min=1.0,
        tv_gate=1.0e-12,
        cosine_gate=1.0 - 1.0e-12,
    )
    assert report["passed"] is True
    assert report["time_samples"] == 2
    assert report["total_variation_distance"] == pytest.approx(0.0)
    assert report["cosine_similarity"] == pytest.approx(1.0)

    paths = mod.write_spectrum_shape_figure(
        report, out=tmp_path / "shape.png", title="shape"
    )
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["json"]).exists()


def test_quasilinear_spectrum_shape_gate_rejects_missing_column(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_diagnostics")
    spectrum = tmp_path / "bad.csv"
    np.savetxt(
        spectrum,
        np.asarray([[0.1, 1.0]]),
        delimiter=",",
        header="ky,other",
        comments="",
    )
    nonlinear = tmp_path / "nl.nc"
    _spectrum_shape_gate_write_netcdf(nonlinear)

    with pytest.raises(ValueError, match="heat_flux_weight_total"):
        mod.build_spectrum_shape_report(
            spectrum_csv=spectrum, nonlinear_netcdf=nonlinear
        )


# Stellarator usefulness assertions
def test_stellarator_usefulness_report_keeps_claim_scoped() -> None:
    module = load_artifact_tool("plot_quasilinear_stellarator_usefulness")
    report = module.build_report()

    assert report["kind"] == "quasilinear_stellarator_usefulness"
    assert "not_runtime_absolute_flux_predictor" in report["claim_level"]
    assert report["models"]["spectral_envelope_ridge"]["accepted"] is False
    assert report["models"]["spectral_envelope_ridge"]["mean_abs_relative_error"] > 0.35
    assert report["models"]["positive_mixing_length"]["accepted"] is False
    assert (
        report["models"]["positive_mixing_length"]["holdout_mean_abs_relative_error"]
        > 1.0
    )
    assert "universal" in report["readme_sentence"]
    assert "rank-screening" in report["readme_sentence"]
    assert "frozen ledger" in report["readme_sentence"]
    assert "more converged nonlinear holdouts" not in " ".join(report["notes"])


def test_stellarator_rows_show_simple_rule_failure_and_scope_statuses() -> None:
    module = load_artifact_tool("plot_quasilinear_stellarator_usefulness")
    report = module.build_report()
    rows = {row["case"]: row for row in report["rows"]}

    for case in ("hsx_nonlinear_window", "w7x_nonlinear_window"):
        row = rows[case]
        assert row["observed_heat_flux"] > 0.0
        assert row["positive_mixing_length_prediction"] == 0.0
        assert row["positive_mixing_length_relative_error"] == 1.0
        assert row["spectral_envelope_ridge_relative_error"] < 0.35
        assert row["spectral_envelope_ridge_interval_contains_observed"] is True
        assert row["stellarator_family"] is True

    qa = report["stellarator_status"]["QA"]
    assert qa["baseline_heat_flux"] > qa["optimized_heat_flux"]
    assert qa["relative_reduction"] > 0.05
    assert "audit only" in qa["status"]

    qh = report["stellarator_status"]["QH"]
    assert qh["high_grid_gate_passed"] is False
    assert qh["least_window_pairwise_heat_flux_symmetric_relative_difference"] > 0.15
    assert "excluded" in qh["status"]


def test_stellarator_usefulness_writer_creates_sidecars(tmp_path: Path) -> None:
    module = load_artifact_tool("plot_quasilinear_stellarator_usefulness")
    report = module.build_report()
    out = tmp_path / "ql_usefulness.png"

    paths = module.write_figure(report, out=out, title="test", dpi=80)

    for key in ("png", "pdf", "json", "csv"):
        assert Path(paths[key]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["kind"] == "quasilinear_stellarator_usefulness"
    assert (
        Path(paths["csv"]).read_text(encoding="utf-8").startswith("case,label,geometry")
    )


# UQ ensemble scaling assertions
def _uq_ensemble_write_payload(path: Path, *, backend: str) -> None:
    payload = {
        "backend": backend,
        "claim_scope": "test",
        "grid": {"Nx": 1, "Ny": 8, "Nz": 8, "Nl": 2, "Nm": 3},
        "time": {"dt": 0.02, "steps": 10},
        "identity_passed": True,
        "rows": [
            {
                "requested_devices": 1,
                "actual_workers": 1,
                "timed_wall_s": 2.0,
                "strong_speedup_vs_1_device": 1.0,
                "parallel_efficiency": 1.0,
                "ensemble_mean_heat_flux_proxy": 1.2,
                "ensemble_std_heat_flux_proxy": 0.1,
                "max_heat_flux_proxy_rel_error": 0.0,
                "max_gamma_abs_error": 0.0,
                "identity_gate_pass": True,
                "error": None,
            },
            {
                "requested_devices": 2,
                "actual_workers": 2,
                "timed_wall_s": 1.1,
                "strong_speedup_vs_1_device": 1.8,
                "parallel_efficiency": 0.9,
                "ensemble_mean_heat_flux_proxy": 1.2,
                "ensemble_std_heat_flux_proxy": 0.1,
                "max_heat_flux_proxy_rel_error": 0.0,
                "max_gamma_abs_error": 0.0,
                "identity_gate_pass": True,
                "error": None,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_summary_combines_cpu_and_gpu_payloads(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_diagnostics")
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    _uq_ensemble_write_payload(cpu, backend="cpu")
    _uq_ensemble_write_payload(gpu, backend="gpu")

    summary = mod.load_summary([cpu, gpu])

    assert summary["identity_passed"] is True
    assert summary["kind"] == "quasilinear_uq_ensemble_scaling_combined"
    assert {row["backend"] for row in summary["rows"]} == {"cpu", "gpu"}
    assert all("Nx=1" in row["grid_label"] for row in summary["rows"])


def test_write_artifacts_creates_combined_outputs(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_quasilinear_diagnostics")
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    _uq_ensemble_write_payload(cpu, backend="cpu")
    _uq_ensemble_write_payload(gpu, backend="gpu")
    summary = mod.load_summary([cpu, gpu])

    paths = mod.write_artifacts(summary, tmp_path / "combined")

    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
