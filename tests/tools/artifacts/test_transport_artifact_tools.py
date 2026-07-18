"""Artifact maintainer tool contracts: transport artifact tools."""

from __future__ import annotations



# ---- test_nonlinear_artifact_reports.py ----

"""Tests for nonlinear artifact reports, gates, and performance panels."""


import csv
import json
from pathlib import Path

import numpy as np
import pytest

from spectraxgk.objectives.vmec_transport_admission import (
    VMECJAXNonlinearAuditPolicy,
    VMECJAXNonlinearCampaignPolicy,
    VMECJAXReducedPrelaunchPolicy,
)
from support.paths import load_artifact_tool


# Baseline-vs-optimized nonlinear audit assertions
def _build_baseline_optimized_nonlinear_audit_ensemble_payload(
    *, case: str, mean: float, sem: float = 0.1
) -> dict[str, object]:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
        "case": case,
        "comparison": f"{case}_seed_timestep_replicates",
        "passed": True,
        "gate_report": {"passed": True},
        "statistics": {
            "n_reports": 3,
            "ensemble_mean": mean,
            "combined_sem": sem,
            "combined_sem_rel": sem / abs(mean),
            "mean_spread": 0.2,
            "mean_rel_spread": 0.02,
        },
        "rows": [
            {"case": f"{case}_seed31", "late_mean": mean - 0.1, "sem": sem},
            {"case": f"{case}_seed32", "late_mean": mean + 0.1, "sem": sem},
            {"case": f"{case}_dt", "late_mean": mean, "sem": sem},
        ],
    }


def _build_baseline_optimized_nonlinear_audit_selected_audit_payload(
    optimized_path: Path,
) -> dict[str, object]:
    return {
        "kind": "production_nonlinear_turbulent_flux_optimization_guard",
        "production_nonlinear_optimization_promoted": True,
        "promotion_gate": {"passed": True, "blockers": []},
        "optimized_equilibrium_artifacts": [
            {
                "path": str(optimized_path),
                "qualifies_for_production_optimization": True,
            }
        ],
    }


def test_baseline_optimized_audit_writes_json_csv_and_png(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_baseline_optimized_nonlinear_audit")
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    selected = tmp_path / "selected.json"
    baseline.write_text(
        json.dumps(
            _build_baseline_optimized_nonlinear_audit_ensemble_payload(
                case="baseline", mean=20.0, sem=0.2
            )
        ),
        encoding="utf-8",
    )
    optimized.write_text(
        json.dumps(
            _build_baseline_optimized_nonlinear_audit_ensemble_payload(
                case="optimized_equilibrium_final", mean=12.0, sem=0.2
            )
        ),
        encoding="utf-8",
    )
    selected.write_text(
        json.dumps(
            _build_baseline_optimized_nonlinear_audit_selected_audit_payload(optimized)
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "audit.json"
    out_csv = tmp_path / "audit.csv"
    out_png = tmp_path / "audit.png"

    rc = mod.main(
        [
            "--baseline-ensemble",
            str(baseline),
            "--optimized-ensemble",
            str(optimized),
            "--selected-optimized-audit",
            str(selected),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--out-png",
            str(out_png),
            "--min-relative-reduction",
            "0.25",
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert out_csv.exists()
    assert out_png.exists()
    assert payload["passed"] is True
    assert payload["comparison"]["baseline_mean"] == 20.0
    assert payload["comparison"]["optimized_mean"] == 12.0
    assert payload["comparison"]["relative_reduction"] == 0.4
    assert payload["selected_optimized_audit"]["optimized_ensemble_selected"] is True


def test_baseline_optimized_audit_fails_closed_when_baseline_missing(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_baseline_optimized_nonlinear_audit")
    optimized = tmp_path / "optimized.json"
    optimized.write_text(
        json.dumps(
            _build_baseline_optimized_nonlinear_audit_ensemble_payload(
                case="optimized_equilibrium_final", mean=12.0
            )
        ),
        encoding="utf-8",
    )
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            _build_baseline_optimized_nonlinear_audit_selected_audit_payload(optimized)
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "audit.json"

    rc = mod.main(
        [
            "--baseline-ensemble",
            str(tmp_path / "missing_baseline.json"),
            "--optimized-ensemble",
            str(optimized),
            "--selected-optimized-audit",
            str(selected),
            "--out-json",
            str(out_json),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["passed"] is False
    assert payload["baseline_ensemble"]["present"] is False
    assert "baseline_ensemble_missing" in payload["blockers"]
    assert any("missing_baseline.json" in blocker for blocker in payload["blockers"])


def test_baseline_optimized_audit_rejects_unselected_optimized_audit(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_baseline_optimized_nonlinear_audit")
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    selected = tmp_path / "selected.json"
    baseline.write_text(
        json.dumps(
            _build_baseline_optimized_nonlinear_audit_ensemble_payload(
                case="baseline", mean=20.0
            )
        ),
        encoding="utf-8",
    )
    optimized.write_text(
        json.dumps(
            _build_baseline_optimized_nonlinear_audit_ensemble_payload(
                case="optimized_equilibrium_final", mean=12.0
            )
        ),
        encoding="utf-8",
    )
    selected.write_text(
        json.dumps(
            {
                "promotion_gate": {"passed": True},
                "optimized_equilibrium_artifacts": [
                    {
                        "path": str(tmp_path / "different_optimized.json"),
                        "qualifies_for_production_optimization": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "audit.json"

    rc = mod.main(
        [
            "--baseline-ensemble",
            str(baseline),
            "--optimized-ensemble",
            str(optimized),
            "--selected-optimized-audit",
            str(selected),
            "--out-json",
            str(out_json),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["passed"] is False
    assert "optimized_ensemble_not_selected_by_audit" in payload["blockers"]


# Matched nonlinear transport comparison assertions
def _build_matched_nonlinear_transport_comparison_ensemble(
    mean: float, sem: float, *, passed: bool = True
) -> dict:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "passed": passed,
        "statistics": {
            "ensemble_mean": mean,
            "combined_sem": sem,
            "mean_rel_spread": 0.03,
            "combined_sem_rel": sem / max(abs(mean), 1.0e-30),
        },
    }


def test_build_comparison_reports_relative_transport_reduction(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    report = load_artifact_tool(
        "build_matched_nonlinear_transport_comparison"
    ).build_comparison(
        baseline=_build_matched_nonlinear_transport_comparison_ensemble(12.0, 0.3),
        candidate=_build_matched_nonlinear_transport_comparison_ensemble(10.5, 0.4),
        baseline_artifact=baseline_path,
        candidate_artifact=candidate_path,
        case="qa_projected",
        min_relative_reduction=0.05,
    )

    assert report["passed"] is True
    assert report["statistics"]["absolute_reduction"] == 1.5
    assert report["statistics"]["relative_reduction"] == 0.125
    assert report["statistics"]["combined_uncertainty"] == 0.5
    assert all(gate["passed"] for gate in report["gates"])


def test_build_comparison_fails_when_candidate_not_lower_enough(tmp_path: Path) -> None:
    report = load_artifact_tool(
        "build_matched_nonlinear_transport_comparison"
    ).build_comparison(
        baseline=_build_matched_nonlinear_transport_comparison_ensemble(12.0, 0.3),
        candidate=_build_matched_nonlinear_transport_comparison_ensemble(11.8, 0.4),
        baseline_artifact=tmp_path / "baseline.json",
        candidate_artifact=tmp_path / "candidate.json",
        case="qa_projected",
        min_relative_reduction=0.05,
    )

    assert report["passed"] is False
    assert report["gates"][-1]["metric"] == "relative_transport_reduction"
    assert report["gates"][-1]["passed"] is False


def test_build_comparison_fails_closed_for_missing_ensemble_mean(
    tmp_path: Path,
) -> None:
    report = load_artifact_tool(
        "build_matched_nonlinear_transport_comparison"
    ).build_comparison(
        baseline={
            "kind": "nonlinear_window_ensemble_report",
            "passed": False,
            "statistics": {"ensemble_mean": None, "combined_sem": None},
        },
        candidate=_build_matched_nonlinear_transport_comparison_ensemble(10.5, 0.4),
        baseline_artifact=tmp_path / "baseline.json",
        candidate_artifact=tmp_path / "candidate.json",
        case="qa_outside_window",
        min_relative_reduction=0.0,
    )

    assert report["passed"] is False
    assert report["baseline"]["finite_mean"] is False
    assert report["statistics"]["relative_reduction"] is None
    assert "ensemble_mean is not finite" in report["gates"][0]["detail"]


def test_cli_writes_json_and_figure(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    out_json = tmp_path / "comparison.json"
    out_svg = tmp_path / "comparison.svg"
    baseline.write_text(
        json.dumps(_build_matched_nonlinear_transport_comparison_ensemble(12.0, 0.2)),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(_build_matched_nonlinear_transport_comparison_ensemble(10.8, 0.25)),
        encoding="utf-8",
    )

    assert (
        load_artifact_tool("build_matched_nonlinear_transport_comparison").main(
            [
                "--baseline-ensemble",
                str(baseline),
                "--candidate-ensemble",
                str(candidate),
                "--case",
                "qa_projected",
                "--min-relative-reduction",
                "0.05",
                "--out-json",
                str(out_json),
                "--out-figure",
                str(out_svg),
                "--fail-on-unpromoted",
            ]
        )
        == 0
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert (
        payload["claim_level"] == "matched_replicated_late_window_transport_comparison"
    )
    assert out_svg.exists()


def test_cli_writes_failure_json_and_figure_for_empty_window_ensemble(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    out_json = tmp_path / "comparison.json"
    out_svg = tmp_path / "comparison.svg"
    baseline.write_text(
        json.dumps(
            {
                "kind": "nonlinear_window_ensemble_report",
                "passed": False,
                "statistics": {"ensemble_mean": None, "combined_sem": None},
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(_build_matched_nonlinear_transport_comparison_ensemble(10.8, 0.25)),
        encoding="utf-8",
    )

    assert (
        load_artifact_tool("build_matched_nonlinear_transport_comparison").main(
            [
                "--baseline-ensemble",
                str(baseline),
                "--candidate-ensemble",
                str(candidate),
                "--case",
                "qa_outside_window",
                "--min-relative-reduction",
                "0.0",
                "--out-json",
                str(out_json),
                "--out-figure",
                str(out_svg),
            ]
        )
        == 0
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["statistics"]["relative_reduction"] is None
    assert out_svg.exists()


# Matched nonlinear transport matrix assertions
def _build_matched_nonlinear_transport_matrix_write_campaign(tmp_path: Path) -> Path:
    baseline = tmp_path / "baseline_wout.nc"
    candidate = tmp_path / "candidate_wout.nc"
    baseline.write_text("baseline vmec placeholder\n", encoding="utf-8")
    candidate.write_text("candidate vmec placeholder\n", encoding="utf-8")
    rc = load_artifact_tool("build_matched_nonlinear_transport_matrix").main(
        [
            "write",
            "--baseline-vmec-file",
            str(baseline),
            "--candidate-vmec-file",
            str(candidate),
            "--baseline-label",
            "strict_qa",
            "--candidate-label",
            "low_transport",
            "--case-prefix",
            "qa_matrix_test",
            "--out-dir",
            str(tmp_path / "campaign"),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--grid",
            "n8:8:8:8:8",
            "--horizons",
            "10,20",
            "--window-tmin",
            "10",
            "--window-tmax",
            "20",
            "--dt",
            "0.1",
            "--dt-variant",
            "0.08",
            "--seed-variant",
            "7",
            "--min-samples",
            "4",
            "--min-window-samples",
            "2",
            "--gpu-splits",
            "2",
        ]
    )
    assert rc == 0
    return tmp_path / "campaign" / "matched_transport_matrix_manifest.json"


def test_write_campaign_defaults_to_eighteen_point_transport_matrix(
    tmp_path: Path,
) -> None:
    manifest_path = _build_matched_nonlinear_transport_matrix_write_campaign(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["kind"] == "matched_nonlinear_transport_matrix_campaign"
    assert payload["config"]["sample_count"] == 18
    assert payload["coverage_gate"] == {
        "min_alphas": 2,
        "min_ky_values": 3,
        "min_surfaces": 3,
        "passed": True,
    }
    assert payload["config"]["surfaces"] == [0.45, 0.64, 0.78]
    assert payload["config"]["alphas"] == [0.0, 0.7853981633974483]
    assert payload["config"]["ky_values"] == [0.1, 0.3, 0.5]
    assert payload["config"]["seed_variants"] == [7]
    assert payload["config"]["dt_variants"] == [0.08]
    assert (
        payload["config"]["final_horizon_launch_locking"]
        == "per-output flock with mkdir fallback"
    )
    assert Path(payload["launch_scripts"]["staged_ladder_skip_existing"]).exists()
    assert Path(payload["launch_scripts"]["postprocess"]).exists()
    assert Path(
        payload["launch_scripts"]["final_horizon_direct_skip_existing"]
    ).exists()
    assert len(payload["launch_scripts"]["final_horizon_gpu_splits"]) == 2
    assert all(
        Path(path).exists()
        for path in payload["launch_scripts"]["final_horizon_gpu_splits"]
    )
    assert (
        "build_matched_nonlinear_transport_matrix.py report"
        in payload["aggregate_report"]["command"]
    )
    final_script = Path(
        payload["launch_scripts"]["final_horizon_direct_skip_existing"]
    ).read_text(encoding="utf-8")
    assert "_nonlinear_t20_" in final_script
    assert "_nonlinear_t10_" not in final_script
    assert "--steps 200" in final_script
    assert "tools/release/check_nonlinear_transport_gates.py target-time" in final_script
    assert "flock -n 9" in final_script
    assert "lock_dir=${lock_file}.d" in final_script
    assert "skip-locked" in final_script
    assert "skip-target-confirmed" in final_script
    assert "skip-target-confirmed-after-lock" in final_script
    assert "skip-existing" not in final_script
    gpu1_script = Path(
        payload["launch_scripts"]["final_horizon_gpu_splits"][1]
    ).read_text(encoding="utf-8")
    assert "export DEVICE=1" in gpu1_script

    first = payload["samples"][0]
    assert first["sample_id"] == "s0p45_a0_ky0p1"
    assert first["surface_torflux"] == 0.45
    assert first["alpha"] == 0.0
    assert first["ky"] == 0.1
    assert set(first["states"]) == {"baseline", "candidate"}
    toml = Path(first["states"]["baseline"]["state_manifest"]).parent / (
        "qa_matrix_test_strict_qa_s0p45_a0_ky0p1_nonlinear_t20_n8_seed7.toml"
    )
    text = toml.read_text(encoding="utf-8")
    assert "torflux = 0.45" in text
    assert "alpha = 0" in text
    assert "ky = 0.1" in text
    assert "random_seed = 7" in text


def test_report_passes_when_all_matrix_comparisons_pass(tmp_path: Path) -> None:
    manifest_path = _build_matched_nonlinear_transport_matrix_write_campaign(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, sample in enumerate(manifest["samples"]):
        comparison = Path(sample["comparison"]["json"])
        comparison.parent.mkdir(parents=True, exist_ok=True)
        comparison.write_text(
            json.dumps(
                {
                    "kind": "matched_nonlinear_transport_comparison",
                    "passed": True,
                    "baseline": {"ensemble_mean": 10.0 + 0.1 * index},
                    "candidate": {"ensemble_mean": 9.5 + 0.1 * index},
                    "statistics": {
                        "relative_reduction": 0.05,
                        "uncertainty_z_score": 3.5,
                    },
                }
            ),
            encoding="utf-8",
        )

    out_json = tmp_path / "matrix_report.json"
    out_png = tmp_path / "matrix_report.png"
    rc = load_artifact_tool("build_matched_nonlinear_transport_matrix").main(
        [
            "report",
            "--matrix-manifest",
            str(manifest_path),
            "--out-json",
            str(out_json),
            "--out-figure",
            str(out_png),
            "--min-pass-fraction",
            "1.0",
            "--min-mean-relative-reduction",
            "0.02",
            "--fail-on-blocked",
        ]
    )
    report = json.loads(out_json.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["passed"] is True
    assert report["summary"]["total_samples"] == 18
    assert report["summary"]["passed_samples"] == 18
    assert report["summary"]["mean_relative_reduction"] == pytest.approx(0.05)
    assert report["blockers"] == []
    assert out_png.exists()


# Nonlinear campaign admission assertions
def _build_nonlinear_campaign_admission_report_write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build_nonlinear_campaign_admission_report_prelaunch(
    *, passed: bool = True, cross_sample: bool = True
) -> dict:
    return {
        "kind": "vmec_jax_reduced_nonlinear_audit_prelaunch_report",
        "passed": passed,
        "blockers": []
        if passed
        else ["insufficient_reduced_margin_for_nonlinear_audit"],
        "objective_sample_summary": {
            "passed": True,
            "sample_count": 18,
            "blockers": [],
        },
        "reduced_cross_sample_statistics": {
            "available": cross_sample,
            "passed": True if cross_sample else None,
            "rows": [],
        },
    }


def _build_nonlinear_campaign_admission_report_landscape(
    *, passed: bool = True, reduction: float = 0.266, z_score: float = 18.0
) -> dict:
    selected = {
        "label": "+3% RBC(0,1)",
        "relative_reduction": reduction,
        "uncertainty_z_score": z_score,
        "combined_sem_rel": 0.0067,
        "n_reports": 3,
    }
    return {
        "kind": "nonlinear_landscape_admission_report",
        "passed": passed,
        "selected_candidate": selected if passed else None,
        "next_action": "use selected direction",
    }


def test_campaign_builder_loads_json_and_admits_rbc_like_evidence(
    tmp_path: Path,
) -> None:
    prelaunch = _build_nonlinear_campaign_admission_report_write(
        tmp_path / "prelaunch.json",
        _build_nonlinear_campaign_admission_report_prelaunch(),
    )
    landscape = _build_nonlinear_campaign_admission_report_write(
        tmp_path / "landscape.json",
        _build_nonlinear_campaign_admission_report_landscape(),
    )

    report = load_artifact_tool(
        "build_nonlinear_transport_admission"
    ).build_campaign_report(
        prelaunch_report=prelaunch,
        landscape_admission=landscape,
        policy=VMECJAXNonlinearCampaignPolicy(),
    )

    assert report["passed"] is True
    assert report["campaign_admitted"] is True
    assert report["artifacts"]["reduced_prelaunch_report"].endswith("prelaunch.json")
    assert report["selected_landscape_candidate"]["label"] == "+3% RBC(0,1)"


def test_campaign_cli_writes_blocked_report_and_returns_nonzero(tmp_path: Path) -> None:
    prelaunch = _build_nonlinear_campaign_admission_report_write(
        tmp_path / "prelaunch.json",
        _build_nonlinear_campaign_admission_report_prelaunch(cross_sample=False),
    )
    landscape = _build_nonlinear_campaign_admission_report_write(
        tmp_path / "landscape.json",
        _build_nonlinear_campaign_admission_report_landscape(
            reduction=0.01, z_score=0.2
        ),
    )
    out = tmp_path / "campaign.json"

    rc = load_artifact_tool("build_nonlinear_transport_admission").main(
        [
            "campaign",
            "--prelaunch-report",
            str(prelaunch),
            "--landscape-admission",
            str(landscape),
            "--out-json",
            str(out),
            "--fail-on-blocked",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["campaign_admitted"] is False
    assert "reduced_cross_sample_statistics_missing" in payload["blockers"]
    assert "selected_landscape_reduction_too_small" in payload["blockers"]


# Nonlinear landscape admission assertions
def _build_nonlinear_landscape_admission_report_ensemble(
    mean: float, sem: float, *, passed: bool = True, n_reports: int = 3
) -> dict:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "case": f"ensemble_mean_{mean}",
        "passed": passed,
        "statistics": {
            "ensemble_mean": mean,
            "combined_sem": sem,
            "combined_sem_rel": sem / abs(mean),
            "n_reports": n_reports,
        },
    }


def _build_nonlinear_landscape_admission_report_write(
    path: Path, payload: dict
) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_report_selects_best_uncertainty_resolved_candidate(
    tmp_path: Path,
) -> None:
    baseline = _build_nonlinear_landscape_admission_report_write(
        tmp_path / "baseline.json",
        _build_nonlinear_landscape_admission_report_ensemble(8.55, 0.12),
    )
    p3 = _build_nonlinear_landscape_admission_report_write(
        tmp_path / "p3.json",
        _build_nonlinear_landscape_admission_report_ensemble(6.27, 0.04),
    )
    p6 = _build_nonlinear_landscape_admission_report_write(
        tmp_path / "p6.json",
        _build_nonlinear_landscape_admission_report_ensemble(6.43, 0.04),
    )

    report = load_artifact_tool(
        "build_nonlinear_transport_admission"
    ).build_landscape_report(
        baseline_ensemble=baseline,
        candidate_ensembles=[("+3%", p3), ("+6%", p6)],
        policy=VMECJAXNonlinearAuditPolicy(minimum_uncertainty_z_score=2.0),
    )

    assert report["passed"] is True
    assert report["selected_candidate"]["label"] == "+3%"
    assert report["artifacts"]["baseline_ensemble"].endswith("baseline.json")
    assert report["artifacts"]["candidate_ensembles"][0]["label"] == "+3%"


def test_cli_writes_report_and_fails_closed_when_requested(tmp_path: Path) -> None:
    baseline = _build_nonlinear_landscape_admission_report_write(
        tmp_path / "baseline.json",
        _build_nonlinear_landscape_admission_report_ensemble(8.0, 0.5),
    )
    noisy = _build_nonlinear_landscape_admission_report_write(
        tmp_path / "noisy.json",
        _build_nonlinear_landscape_admission_report_ensemble(7.95, 0.5),
    )
    out_json = tmp_path / "admission.json"

    assert (
        load_artifact_tool("build_nonlinear_transport_admission").main(
            [
                "landscape",
                "--baseline-ensemble",
                str(baseline),
                "--candidate-ensemble",
                "noisy",
                str(noisy),
                "--min-relative-reduction",
                "0.02",
                "--min-uncertainty-z-score",
                "2.0",
                "--out-json",
                str(out_json),
                "--fail-on-no-admission",
            ]
        )
        == 1
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["selected_candidate"] is None
    assert (
        "insufficient_relative_reduction"
        in payload["candidates"][0]["admission_blockers"]
    )


# Nonlinear transport horizon audit assertions
def _build_nonlinear_transport_horizon_audit_write_json(
    root: Path, relative: str, payload: dict[str, object]
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_classify_record_separates_transport_startup_and_reduced() -> None:
    assert (
        load_artifact_tool("build_nonlinear_transport_horizon_audit").classify_record(
            {"gate_passed": True, "effective_tmax": 100.0}
        )
        == "release_transport_gate_passed"
    )
    assert (
        load_artifact_tool("build_nonlinear_transport_horizon_audit").classify_record(
            {
                "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
                "transport_average_gate": False,
                "effective_tmax": 0.6,
            }
        )
        == "short_or_startup_not_transport_average"
    )
    assert (
        load_artifact_tool("build_nonlinear_transport_horizon_audit").classify_record(
            {
                "kind": "stellarator_optimization_model",
                "claim_level": "reduced nonlinear estimator optimization",
                "effective_tmax": 90.0,
            }
        )
        == "reduced_estimator_not_transport_average"
    )
    assert (
        load_artifact_tool("build_nonlinear_transport_horizon_audit").classify_record(
            {
                "claim_level": "negative_grid_convergence_result_not_transport_validation",
                "effective_tmax": 150.0,
                "convergence_gate_passed": False,
            }
        )
        == "long_but_failed_convergence"
    )


def test_production_optimization_blockers_keep_transport_gates_as_prerequisites() -> (
    None
):
    transport_record = {
        "gate_passed": True,
        "effective_tmax": 100.0,
    }

    blockers = load_artifact_tool(
        "build_nonlinear_transport_horizon_audit"
    ).production_optimization_blockers(transport_record)

    assert "missing grid-convergence gate for optimized nonlinear objective" in blockers
    assert (
        "missing timestep-convergence gate for optimized nonlinear objective"
        in blockers
    )
    assert "missing seed/initial-condition uncertainty gate" in blockers
    assert "missing optimized-equilibrium nonlinear audit" in blockers

    ready_record = {
        **transport_record,
        "grid_convergence_gate_passed": True,
        "timestep_convergence_gate_passed": True,
        "seed_ensemble_gate_passed": True,
        "optimized_equilibrium_audit_passed": True,
    }
    assert (
        load_artifact_tool(
            "build_nonlinear_transport_horizon_audit"
        ).production_optimization_blockers(ready_record)
        == []
    )

    reduced_blockers = load_artifact_tool(
        "build_nonlinear_transport_horizon_audit"
    ).production_optimization_blockers(
        {
            "kind": "stellarator_optimization_model",
            "claim_level": "reduced nonlinear estimator optimization",
            "effective_tmax": 90.0,
        }
    )
    assert (
        "reduced estimator output is not an actual nonlinear transport average"
        in reduced_blockers
    )


def test_build_payload_marks_short_fd_audit_outside_transport_scope(
    tmp_path: Path,
) -> None:
    _build_nonlinear_transport_horizon_audit_write_json(
        tmp_path,
        "docs/_static/nonlinear_cyclone_gate_summary.json",
        {
            "case": "cyclone_nonlinear_long_window",
            "gate_passed": True,
            "tmax": 400.0,
            "spectrax": "missing.csv",
        },
    )
    _build_nonlinear_transport_horizon_audit_write_json(
        tmp_path,
        "docs/_static/nonlinear_window_fd_audit.json",
        {
            "kind": "nonlinear_startup_window_finite_difference_audit",
            "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
            "transport_average_gate": False,
            "passed": True,
            "metrics": {"max_tmax": 0.64},
        },
    )
    _build_nonlinear_transport_horizon_audit_write_json(
        tmp_path,
        "docs/_static/external_vmec_qh_nonlinear_t150_pilot.json",
        {
            "kind": "nonlinear_feasibility_pilot",
            "claim_level": "finite_reduced_grid_long_nonlinear_feasibility_not_grid_converged_transport_validation",
            "label": "QH pilot",
            "tmax": 150.0,
            "promotion_gate": {"passed": False, "reason": "not grid converged"},
        },
    )
    _build_nonlinear_transport_horizon_audit_write_json(
        tmp_path,
        "docs/_static/external_vmec_dshape_t250_high_grid_convergence_gate.json",
        {
            "kind": "external_vmec_nonlinear_grid_convergence_gate",
            "case": "D-shaped grid gate",
            "claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "gate_report": {"passed": True},
            "runs": [{"tmax": 250.0}, {"tmax": 250.0}],
        },
    )

    payload = load_artifact_tool(
        "build_nonlinear_transport_horizon_audit"
    ).build_payload(tmp_path)
    rows = {row["case"]: row for row in payload["records"]}

    assert (
        rows["cyclone_nonlinear_long_window"]["status"]
        == "release_transport_gate_passed"
    )
    assert (
        rows["cyclone_nonlinear_long_window"]["production_nonlinear_optimization_ready"]
        is False
    )
    assert (
        rows["Compact nonlinear FD startup audit"]["status"]
        == "short_or_startup_not_transport_average"
    )
    assert (
        "missing long post-transient nonlinear transport average"
        in rows["Compact nonlinear FD startup audit"][
            "production_nonlinear_optimization_blockers"
        ]
    )
    assert rows["QH pilot"]["status"] == "long_feasibility_pending_convergence"
    assert rows["D-shaped grid gate"]["status"] == "release_transport_gate_passed"
    assert rows["D-shaped grid gate"]["grid_convergence_gate_passed"] is True
    assert (
        "missing grid-convergence gate for optimized nonlinear objective"
        not in rows["D-shaped grid gate"]["production_nonlinear_optimization_blockers"]
    )
    assert (
        "missing timestep-convergence gate for optimized nonlinear objective"
        in rows["D-shaped grid gate"]["production_nonlinear_optimization_blockers"]
    )
    assert (
        "missing seed/initial-condition uncertainty gate"
        in rows["D-shaped grid gate"]["production_nonlinear_optimization_blockers"]
    )
    assert (
        "missing optimized-equilibrium nonlinear audit"
        in rows["D-shaped grid gate"]["production_nonlinear_optimization_blockers"]
    )
    assert payload["summary"]["release_transport_gate_passed"] == 2
    assert payload["summary"]["short_or_reduced_not_transport"] == 1
    assert payload["summary"]["production_nonlinear_optimization_ready"] == 0


# Nonlinear window finite-difference audit assertions
def _build_nonlinear_window_fd_audit_synthetic_run(
    label: str, tprim: float, scale: float = 1.0
) -> dict:
    time = np.linspace(0.0, 1.0, 8)
    heat = scale * (1.0 + 0.08 * time)
    return {
        "label": label,
        "tprim": tprim,
        "random_seed": 22,
        "time": time.tolist(),
        "heat_flux": heat.tolist(),
        "window": load_artifact_tool(
            "build_nonlinear_window_fd_audit"
        ).late_window_metrics(time, heat, tail_fraction=0.5),
    }


def test_late_window_metrics_reports_conditioning_quantities() -> None:
    mod = load_artifact_tool("build_nonlinear_window_fd_audit")
    time = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    heat = np.asarray([1.0, 1.1, 1.2, 1.3, 1.4])

    metrics = mod.late_window_metrics(time, heat, tail_fraction=0.4)

    assert metrics["start_index"] == 3
    assert metrics["n_samples"] == 2
    assert metrics["mean"] == 1.35
    assert metrics["cv"] > 0.0
    assert metrics["trend"] > 0.0


def test_build_audit_payload_passes_conditioned_monotone_runs() -> None:
    mod = load_artifact_tool("build_nonlinear_window_fd_audit")
    runs = [
        _build_nonlinear_window_fd_audit_synthetic_run("minus", 2.31, 0.90),
        _build_nonlinear_window_fd_audit_synthetic_run("base", 2.49, 1.00),
        _build_nonlinear_window_fd_audit_synthetic_run("plus", 2.67, 1.12),
        _build_nonlinear_window_fd_audit_synthetic_run("base_repeat", 2.49, 1.00),
    ]

    payload = mod.build_audit_payload(
        runs,
        base_tprim=2.49,
        perturbation_step=0.18,
        tail_fraction=0.5,
        repeatability_rtol=1.0e-12,
        max_window_cv=0.1,
        max_window_trend=0.1,
        min_response_fraction=0.03,
    )

    assert payload["passed"] is True
    assert payload["startup_nonlinear_plumbing_fd_path_gate"] is True
    assert payload["transport_average_gate"] is False
    assert payload["production_nonlinear_observable_fd_path_gate"] is False
    assert payload["production_nonlinear_window_gradient_gate"] is False
    assert payload["gates"]["monotonic_drive_response"] is True
    assert payload["metrics"]["central_fd_dq_dtprim"] > 0.0
    assert payload["transport_average_requirements"]["passed"] is False


def test_build_audit_payload_blocks_unresolved_response() -> None:
    mod = load_artifact_tool("build_nonlinear_window_fd_audit")
    runs = [
        _build_nonlinear_window_fd_audit_synthetic_run("minus", 2.31, 0.999),
        _build_nonlinear_window_fd_audit_synthetic_run("base", 2.49, 1.000),
        _build_nonlinear_window_fd_audit_synthetic_run("plus", 2.67, 1.001),
        _build_nonlinear_window_fd_audit_synthetic_run("base_repeat", 2.49, 1.000),
    ]

    payload = mod.build_audit_payload(
        runs,
        base_tprim=2.49,
        perturbation_step=0.18,
        tail_fraction=0.5,
        repeatability_rtol=1.0e-12,
        max_window_cv=0.1,
        max_window_trend=0.1,
        min_response_fraction=0.03,
    )

    assert payload["passed"] is False
    assert payload["gates"]["resolved_fd_response"] is False


def test_main_writes_artifacts_without_running_solver(
    monkeypatch, tmp_path: Path
) -> None:
    mod = load_artifact_tool("build_nonlinear_window_fd_audit")

    def fake_run_cyclone_window(*, label: str, tprim: float, **_kwargs):
        scale = {"minus": 0.90, "base": 1.00, "plus": 1.12, "base_repeat": 1.00}[label]
        time = np.linspace(0.0, 1.0, 8)
        heat = scale * (1.0 + 0.08 * time)
        return {
            "label": label,
            "tprim": tprim,
            "random_seed": 22,
            "time": time.tolist(),
            "heat_flux": heat.tolist(),
            "window": mod.late_window_metrics(time, heat, tail_fraction=0.5),
        }

    monkeypatch.setattr(mod, "run_cyclone_window", fake_run_cyclone_window)
    out = tmp_path / "audit.png"

    assert mod.main(["--out", str(out), "--tail-fraction", "0.5"]) == 0

    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".csv").exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["passed"] is True
    assert (
        meta["claim_level"]
        == "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average"
    )
    assert meta["transport_average_gate"] is False


# Reduced nonlinear audit prelaunch assertions
def _build_reduced_nonlinear_audit_prelaunch_report_landscape(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "vmec_boundary_transport_objective_landscape",
                "sample_set": {
                    "surfaces": [0.45, 0.64, 0.78],
                    "alphas": [0.0, 0.7853981633974483],
                    "ky_values": [0.1, 0.3, 0.5],
                },
                "rows": [
                    {
                        "label": "0",
                        "relative_fraction": 0.0,
                        "coefficient_value": 1.0,
                        "reduced_metrics": {
                            "nonlinear_window_heat_flux": 0.06558065223919245
                        },
                    },
                    {
                        "label": "p0p03",
                        "relative_fraction": 0.03,
                        "coefficient_value": 1.03,
                        "reduced_metrics": {
                            "nonlinear_window_heat_flux": 0.06251277500404685
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_report_from_landscape_selects_candidate_row(tmp_path: Path) -> None:
    report = load_artifact_tool(
        "build_nonlinear_transport_admission"
    ).build_prelaunch_report(
        landscape_json=_build_reduced_nonlinear_audit_prelaunch_report_landscape(
            tmp_path / "landscape.json"
        ),
        baseline_selector="0",
        candidate_selector="p0p03",
        metric_key="nonlinear_window_heat_flux",
        failed_reference_relative_reduction=0.022876,
        policy=VMECJAXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )

    assert report["passed"] is True
    assert report["selected_rows"]["candidate"]["label"] == "p0p03"
    assert report["relative_reduced_reduction"] > 0.046


def test_cli_writes_blocked_report(tmp_path: Path) -> None:
    landscape = _build_reduced_nonlinear_audit_prelaunch_report_landscape(
        tmp_path / "landscape.json"
    )
    out_json = tmp_path / "prelaunch.json"

    assert (
        load_artifact_tool("build_nonlinear_transport_admission").main(
            [
                "prelaunch",
                "--landscape-json",
                str(landscape),
                "--candidate-row",
                "p0p03",
                "--min-relative-reduction",
                "0.10",
                "--out-json",
                str(out_json),
                "--fail-on-blocked",
            ]
        )
        == 1
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "insufficient_reduced_margin_for_nonlinear_audit" in payload["blockers"]


def test_metric_mode_builds_negative_prelaunch_reference() -> None:
    report = load_artifact_tool(
        "build_nonlinear_transport_admission"
    ).build_prelaunch_metric_report(
        baseline_metric=0.08010670290,
        candidate_metric=0.07827418221,
        sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        metric_key="nonlinear_window_heat_flux",
        failed_reference_relative_reduction=0.022876,
        policy=VMECJAXReducedPrelaunchPolicy(
            minimum_relative_reduction=0.04,
            failed_reference_safety_factor=1.5,
        ),
    )

    assert report["passed"] is False
    assert (
        report["relative_reduced_reduction"]
        < report["required_relative_reduced_reduction"]
    )


# Laguerre nonlinear mode gate assertions
def test_write_laguerre_gate_csv_uses_lf_line_endings(tmp_path: Path) -> None:
    grid = {
        key: 1.0
        for key in load_artifact_tool("gate_laguerre_nonlinear_modes").DIAGNOSTIC_KEYS
    }
    spectral = {
        key: 1.0
        for key in load_artifact_tool("gate_laguerre_nonlinear_modes").DIAGNOSTIC_KEYS
    }
    grid["run_s"] = 2.0
    spectral["run_s"] = 1.0
    comparison = {
        f"{key}_rel_diff": 0.0
        for key in load_artifact_tool("gate_laguerre_nonlinear_modes").DIAGNOSTIC_KEYS
    }
    comparison["max_rel_diff"] = 0.0
    comparison["speedup_grid_over_spectral"] = 2.0

    out = tmp_path / "laguerre_gate.csv"
    load_artifact_tool("gate_laguerre_nonlinear_modes")._write_csv(
        out,
        [
            {
                "case": "cyclone",
                "status": "pass",
                "steps": 2,
                "dt": 0.05,
                "grid": grid,
                "spectral": spectral,
                "comparison": comparison,
            }
        ],
    )

    raw = out.read_bytes()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 2
    assert b"speedup_grid_over_spectral" in raw


# Nonlinear sharding production gate assertions
def _generate_nonlinear_sharding_production_gate_row(
    *, backend: str, speedup: float, identity: bool = True, active: bool = True
) -> dict:
    return {
        "backend": backend,
        "requested_devices": 2,
        "actual_devices": 2,
        "best_spec": "kx",
        "state_sharding_active": active,
        "identity_gate_pass": identity,
        "strong_speedup_vs_1_device": speedup,
        "max_abs_state_error": 0.0,
        "max_rel_state_error": 0.0,
        "error": None,
    }


def test_nonlinear_sharding_production_gate_defaults_to_tracked_inputs() -> None:
    mod = load_artifact_tool("generate_nonlinear_sharding_production_gate")
    args = mod.build_parser().parse_args([])

    assert args.inputs == mod.DEFAULT_INPUTS
    assert args.out_prefix == mod.DEFAULT_OUT_PREFIX
    assert args.min_speedup == mod.DEFAULT_MIN_SPEEDUP
    assert args.identity_atol == mod.DEFAULT_IDENTITY_ATOL
    assert args.identity_rtol == mod.DEFAULT_IDENTITY_RTOL
    assert args.required_backends == ("cpu", "gpu")


def test_nonlinear_sharding_production_gate_fails_closed_on_identity_without_speedup() -> (
    None
):
    mod = load_artifact_tool("generate_nonlinear_sharding_production_gate")

    summary = mod.evaluate_production_gate(
        [_generate_nonlinear_sharding_production_gate_row(backend="gpu", speedup=0.96)],
        required_backends=("gpu",),
        min_speedup=1.20,
        min_efficiency=0.50,
    )

    assert summary["gate_passed"] is False
    assert summary["production_speedup_claim_allowed"] is False
    assert summary["status"] == "diagnostic_only"
    assert summary["blockers"] == ["gpu_production_speedup_candidate_missing"]
    assert summary["rows"][0]["candidate_passed"] is False
    assert summary["rows"][0]["classification"] == "identity_preserving_regression"
    assert "speedup_below_threshold" in summary["rows"][0]["blockers"]
    assert summary["backend_blocker_report"]["gpu"] == {
        "row_count": 1,
        "candidate_row_count": 1,
        "passing_candidate_count": 0,
        "production_speedup_candidate_missing": True,
        "identity_evidence_complete": True,
        "active_identity_evidence_complete": True,
        "classification_counts": {"identity_preserving_regression": 1},
        "candidate_blocker_counts": {
            "parallel_efficiency_below_threshold": 1,
            "speedup_below_threshold": 1,
        },
        "primary_blockers": ["gpu_production_speedup_candidate_missing"],
        "claim_scope": (
            "Backend remains diagnostic unless at least one active candidate row "
            "has complete identity evidence and passes the speedup and efficiency gates."
        ),
    }


def test_nonlinear_sharding_production_gate_requires_identity_and_active_sharding() -> (
    None
):
    mod = load_artifact_tool("generate_nonlinear_sharding_production_gate")

    summary = mod.evaluate_production_gate(
        [
            _generate_nonlinear_sharding_production_gate_row(
                backend="gpu", speedup=1.6, identity=False
            ),
            _generate_nonlinear_sharding_production_gate_row(
                backend="gpu", speedup=1.8, active=False
            ),
        ],
        required_backends=("gpu",),
        min_speedup=1.20,
        min_efficiency=0.50,
    )

    assert summary["gate_passed"] is False
    assert "identity_gate_failed" in summary["rows"][0]["blockers"]
    assert summary["rows"][0]["classification"] == "identity_failed"
    assert "state_sharding_inactive" in summary["rows"][1]["blockers"]
    assert summary["rows"][1]["classification"] == "inactive_or_fallback"


def test_nonlinear_sharding_production_gate_passes_only_matching_backend_candidates() -> (
    None
):
    mod = load_artifact_tool("generate_nonlinear_sharding_production_gate")

    summary = mod.evaluate_production_gate(
        [
            _generate_nonlinear_sharding_production_gate_row(
                backend="cpu", speedup=1.3
            ),
            _generate_nonlinear_sharding_production_gate_row(
                backend="gpu", speedup=1.4
            ),
        ],
        required_backends=("cpu", "gpu"),
        min_speedup=1.20,
        min_efficiency=0.50,
    )

    assert summary["gate_passed"] is True
    assert summary["production_speedup_claim_allowed"] is True
    assert summary["status"] == "production_speedup_candidate"
    assert summary["best_candidates"]["gpu"]["strong_speedup_vs_1_device"] == 1.4
    assert summary["rows"][0]["classification"] == "production_candidate"
    assert summary["backend_summary"]["gpu"]["production_candidate_count"] == 1


def test_nonlinear_sharding_production_gate_loads_sweep_rows(tmp_path: Path) -> None:
    mod = load_artifact_tool("generate_nonlinear_sharding_production_gate")
    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps(
            {
                "backend": "gpu",
                "rows": [
                    _generate_nonlinear_sharding_production_gate_row(
                        backend="gpu", speedup=1.4
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = mod.load_rows([path])

    assert rows[0]["backend"] == "gpu"
    assert rows[0]["source"] == str(path)


def test_nonlinear_sharding_production_gate_fails_closed_on_missing_error_metrics() -> (
    None
):
    mod = load_artifact_tool("generate_nonlinear_sharding_production_gate")
    row = _generate_nonlinear_sharding_production_gate_row(backend="gpu", speedup=1.6)
    row.pop("max_abs_state_error")

    summary = mod.evaluate_production_gate(
        [row],
        required_backends=("gpu",),
        min_speedup=1.20,
        min_efficiency=0.50,
    )

    assert summary["gate_passed"] is False
    assert summary["rows"][0]["classification"] == "identity_failed"
    assert "identity_abs_error_missing" in summary["rows"][0]["blockers"]
    assert summary["rows"][0]["identity_abs_tolerance_fraction"] is None
    assert summary["identity_evidence_summary"]["gpu"]["finite_error_metric_count"] == 0
    assert (
        summary["identity_evidence_summary"]["gpu"]["identity_blocker_counts"][
            "identity_abs_error_missing"
        ]
        == 1
    )
    gpu_report = summary["backend_blocker_report"]["gpu"]
    assert gpu_report["production_speedup_candidate_missing"] is True
    assert gpu_report["identity_evidence_complete"] is False
    assert gpu_report["active_identity_evidence_complete"] is False
    assert "identity_evidence_incomplete" in gpu_report["primary_blockers"]
    assert "active_identity_evidence_incomplete" in gpu_report["primary_blockers"]
    assert gpu_report["candidate_blocker_counts"]["identity_abs_error_missing"] == 1


def test_nonlinear_sharding_production_gate_reports_identity_evidence_by_backend() -> (
    None
):
    mod = load_artifact_tool("generate_nonlinear_sharding_production_gate")
    cpu = _generate_nonlinear_sharding_production_gate_row(backend="cpu", speedup=1.3)
    cpu["max_abs_state_error"] = 5.0e-6
    cpu["max_rel_state_error"] = 2.0e-6
    gpu = _generate_nonlinear_sharding_production_gate_row(backend="gpu", speedup=1.4)
    gpu["max_abs_state_error"] = 2.0e-5
    gpu["max_rel_state_error"] = 1.0e-6

    summary = mod.evaluate_production_gate(
        [cpu, gpu],
        required_backends=("cpu", "gpu"),
        min_speedup=1.20,
        min_efficiency=0.50,
        identity_atol=1.0e-5,
        identity_rtol=1.0e-5,
    )

    cpu_evidence = summary["identity_evidence_summary"]["cpu"]
    gpu_evidence = summary["identity_evidence_summary"]["gpu"]

    assert cpu_evidence["row_count"] == 1
    assert cpu_evidence["identity_gate_pass_count"] == 1
    assert cpu_evidence["finite_error_metric_count"] == 1
    assert cpu_evidence["identity_within_tolerance_count"] == 1
    assert cpu_evidence["active_identity_within_tolerance_count"] == 1
    assert cpu_evidence["max_abs_tolerance_fraction"] == pytest.approx(0.5)
    assert cpu_evidence["max_rel_tolerance_fraction"] == pytest.approx(0.2)
    assert cpu_evidence["worst_finite_error_row"]["requested_devices"] == 2
    assert summary["rows"][0]["identity_abs_tolerance_fraction"] == pytest.approx(0.5)
    assert summary["rows"][0]["identity_rel_tolerance_fraction"] == pytest.approx(0.2)

    assert gpu_evidence["identity_within_tolerance_count"] == 0
    assert gpu_evidence["max_abs_tolerance_fraction"] == pytest.approx(2.0)
    assert (
        gpu_evidence["identity_blocker_counts"]["identity_abs_error_above_tolerance"]
        == 1
    )
    assert summary["rows"][1]["classification"] == "identity_failed"
    assert summary["gate_passed"] is False


def test_nonlinear_sharding_production_gate_classifies_reference_and_weak_scaling() -> (
    None
):
    mod = load_artifact_tool("generate_nonlinear_sharding_production_gate")
    reference = _generate_nonlinear_sharding_production_gate_row(
        backend="cpu", speedup=1.0
    )
    reference["requested_devices"] = 1
    reference["actual_devices"] = 1
    reference["state_sharding_active"] = False
    weak = _generate_nonlinear_sharding_production_gate_row(backend="cpu", speedup=1.3)
    weak["requested_devices"] = 4
    weak["actual_devices"] = 4

    summary = mod.evaluate_production_gate(
        [reference, weak],
        required_backends=("cpu",),
        min_speedup=1.20,
        min_efficiency=0.50,
    )

    assert summary["gate_passed"] is False
    assert summary["rows"][0]["classification"] == "reference_only"
    assert summary["rows"][1]["classification"] == "identity_only_insufficient_speedup"
    assert "parallel_efficiency_below_threshold" in summary["rows"][1]["blockers"]
    assert (
        summary["backend_summary"]["cpu"]["best_identity_preserving_row"][
            "requested_devices"
        ]
        == 4
    )
    assert summary["backend_blocker_report"]["cpu"]["candidate_row_count"] == 1
    assert summary["backend_blocker_report"]["cpu"]["candidate_blocker_counts"] == {
        "parallel_efficiency_below_threshold": 1
    }


# External VMEC nonlinear convergence gate assertions
def _plot_external_vmec_nonlinear_convergence_gate_write_pilot(
    tmp_path: Path, name: str, mean: float, *, slope: float = 0.0
) -> Path:
    t = np.linspace(0.0, 20.0, 21)
    heat_flux = mean + slope * t + 0.01 * np.sin(t)
    wphi = 2.0 + 0.02 * np.cos(t)
    csv_path = tmp_path / f"{name}.traces.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "heat_flux", "wphi", "wg"])
        for row in zip(t, heat_flux, wphi, np.ones_like(t), strict=True):
            writer.writerow([f"{float(value):.16e}" for value in row])
    late = t >= 10.0
    report = {
        "kind": "nonlinear_feasibility_pilot",
        "label": name,
        "csv": csv_path.name,
        "least_trending_window": {
            "tmin": float(t[late][0]),
            "tmax": float(t[late][-1]),
            "heat_flux_mean": float(np.mean(heat_flux[late])),
            "heat_flux_std": float(np.std(heat_flux[late])),
            "heat_flux_relative_slope_per_time": 0.0,
            "n_samples": int(np.count_nonzero(late)),
        },
    }
    json_path = tmp_path / f"{name}.json"
    json_path.write_text(json.dumps(report), encoding="utf-8")
    return json_path


def test_convergence_gate_passes_for_flat_nearby_traces(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_external_vmec_nonlinear_convergence_gate")
    first = _plot_external_vmec_nonlinear_convergence_gate_write_pilot(
        tmp_path, "n32", 1.0
    )
    second = _plot_external_vmec_nonlinear_convergence_gate_write_pilot(
        tmp_path, "n48", 1.05
    )

    paths = mod.write_convergence_gate(
        [first, second],
        out=tmp_path / "gate.png",
        labels=["n32", "n48"],
        case="synthetic external VMEC convergence",
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["promotion_gate"]["passed"] is True
    assert (
        payload["claim_level"]
        == "passed_grid_convergence_candidate_for_transport_holdout"
    )
    assert payload["promotion_gate"]["reason"].startswith(
        "synthetic external VMEC convergence passed"
    )
    assert payload["gate_report"]["passed"] is True
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["csv"]).exists()


def test_convergence_gate_fails_large_grid_shift(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_external_vmec_nonlinear_convergence_gate")
    first = _plot_external_vmec_nonlinear_convergence_gate_write_pilot(
        tmp_path, "n32", 1.0
    )
    second = _plot_external_vmec_nonlinear_convergence_gate_write_pilot(
        tmp_path, "n48", 1.6
    )

    paths = mod.write_convergence_gate(
        [first, second],
        out=tmp_path / "gate.png",
        labels=["n32", "n48"],
        case="synthetic external VMEC convergence",
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    failed = {
        gate["metric"] for gate in payload["gate_report"]["gates"] if not gate["passed"]
    }
    assert payload["passed"] is False
    assert payload["promotion_gate"]["passed"] is False
    assert (
        payload["claim_level"]
        == "negative_grid_convergence_result_not_transport_validation"
    )
    assert payload["promotion_gate"]["reason"].startswith(
        "synthetic external VMEC convergence is finite"
    )
    assert "common_window_pairwise_heat_flux_symmetric_relative_difference" in failed
    assert "least_window_pairwise_heat_flux_symmetric_relative_difference" in failed


# Nonlinear feasibility pilot assertions
def test_window_summaries_track_late_slope() -> None:
    mod = load_artifact_tool("build_nonlinear_validation_panels")
    t = np.linspace(0.0, 10.0, 11)
    heat = 2.0 + 0.1 * t
    wphi = 1.0 + 0.05 * t

    summaries = mod.window_summaries(t, heat, wphi, start_fractions=(0.5,))

    assert len(summaries) == 1
    assert summaries[0]["start_index"] == 5
    assert summaries[0]["n_samples"] == 6
    assert summaries[0]["heat_flux_slope"] == pytest.approx(0.1)
    assert summaries[0]["heat_flux_last"] == pytest.approx(3.0)


def test_write_pilot_panel_writes_replayable_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_nonlinear_validation_panels")
    t = np.linspace(0.0, 20.0, 21)
    trace = {
        "t": t,
        "heat_flux": 1.0 + np.sin(t / 3.0) * 0.1,
        "wphi": 0.5 + np.cos(t / 4.0) * 0.05,
        "wg": 2.0 + 0.01 * t,
    }

    paths = mod.write_pilot_panel(
        trace,
        out=tmp_path / "pilot.png",
        source="synthetic.out.nc",
        title="Synthetic pilot",
        label="synthetic",
        start_fractions=(0.5, 0.75),
    )

    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["csv"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["kind"] == "nonlinear_feasibility_pilot"
    assert payload["promotion_gate"]["passed"] is False
    assert len(payload["window_summaries"]) == 2


def test_window_summaries_validate_inputs() -> None:
    mod = load_artifact_tool("build_nonlinear_validation_panels")
    with pytest.raises(ValueError, match="same length"):
        mod.window_summaries([0, 1, 2], [1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="at least three samples"):
        mod.window_summaries([0, 1], [1, 2], [1, 2])


# Nonlinear RHS profile assertions
def test_build_summary_reports_dominant_kernel_and_speedups() -> None:
    payload = load_artifact_tool("plot_scaling_panels").build_rhs_profile_summary(
        {
            "CPU grid": {
                "field_solve": 1.0,
                "linear_rhs": 4.0,
                "nonlinear_bracket": 2.0,
                "full_rhs": 8.0,
            },
            "CPU spectral": {
                "field_solve": 1.0,
                "linear_rhs": 4.0,
                "nonlinear_bracket": 1.0,
                "full_rhs": 5.0,
            },
        }
    )

    assert payload["kind"] == "nonlinear_rhs_profile_summary"
    assert payload["rows"]["CPU grid"]["dominant_measured_kernel"] == "linear_rhs"
    assert payload["rows"]["CPU grid"]["linear_rhs_fraction_of_full_rhs"] == 0.5
    assert payload["spectral_speedups"]["cpu"]["full_rhs_grid_over_spectral"] == 1.6
    assert (
        payload["spectral_speedups"]["cpu"]["nonlinear_bracket_grid_over_spectral"]
        == 2.0
    )
    assert payload["fastest_full_rhs_label"] == "CPU spectral"


def test_write_summary_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    load_artifact_tool("plot_scaling_panels").write_rhs_profile_summary_json(
        {"kind": "x", "value": 1.0}, path
    )

    assert json.loads(path.read_text(encoding="utf-8")) == {"kind": "x", "value": 1.0}


def test_parse_input_arg_and_case_label() -> None:
    label, path = load_artifact_tool("plot_scaling_panels").parse_rhs_profile_input(
        "GPU spectral=docs/_static/example.csv"
    )
    payload = load_artifact_tool("plot_scaling_panels").build_rhs_profile_summary(
        {"GPU spectral": {"full_rhs": 2.0}}, case="larger_case"
    )

    assert label == "GPU spectral"
    assert str(path) == "docs/_static/example.csv"
    assert payload["case"] == "larger_case"
    assert (
        load_artifact_tool("plot_scaling_panels").rhs_profile_case_title(
            "cyclone_miller_benchmark_size"
        )
        == "Cyclone Miller benchmark-size case"
    )


# Nonlinear sharding strong-scaling assertions
def test_plot_scaling_panels_nonlinear_sharding_parser_defaults_to_large_inputs() -> (
    None
):
    mod = load_artifact_tool("plot_scaling_panels")

    args = mod.build_nonlinear_sharding_parser().parse_args([])

    assert args.inputs == mod.DEFAULT_NONLINEAR_SHARDING_INPUTS
    assert args.out_prefix == mod.DEFAULT_NONLINEAR_SHARDING_PREFIX


def test_plot_scaling_panels_nonlinear_sharding_loads_combined_rows(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_scaling_panels")
    payload = {
        "backend": "cpu",
        "grid": {"Nx": 24, "Ny_requested": 48, "Nz": 96, "Nl": 4, "Nm": 8},
        "identity_passed": True,
        "speedup_passed": False,
        "speedup_blockers": ["cpu_2devices_speedup_0.8_below_1"],
        "rows": [
            {
                "backend": "cpu",
                "requested_devices": 1,
                "actual_devices": 1,
                "best_spec": "kx",
                "state_sharding_active": False,
                "identity_gate_pass": True,
                "parallel_median_s": 2.0,
                "strong_speedup_vs_1_device": 1.0,
                "same_process_speedup": 1.1,
                "max_rel_state_error": 0.0,
                "error": None,
            }
        ],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary = mod.load_nonlinear_sharding_summary([path])

    assert summary["identity_passed"] is True
    assert summary["speedup_passed"] is False
    assert summary["status"] == "diagnostic_identity_only"
    assert summary["speedup_blockers"] == ["cpu:cpu_2devices_speedup_0.8_below_1"]
    assert summary["rows"][0]["grid_label"] == "Nx=24, Ny=48, Nz=96, Nl=4, Nm=8"
    assert summary["rows"][0]["source"] == str(path)


# Nonlinear window-statistics assertions
def _write_nonlinear_window_gate_summary(
    path: Path,
    *,
    case: str,
    include: bool = True,
    heat_flux: float = 0.03,
    wg: float = 0.02,
    wphi: float = 0.025,
) -> None:
    payload = {
        "case": case,
        "source": "synthetic runtime diagnostics",
        "gate_mean_rel": 0.10,
        "gate_passed": True,
        "summary": [
            {
                "metric": "Wg",
                "mean_rel_abs": wg,
                "max_rel_abs": 0.04,
                "final_rel": 0.01,
            },
            {
                "metric": "Wphi",
                "mean_rel_abs": wphi,
                "max_rel_abs": 0.05,
                "final_rel": -0.01,
            },
            {
                "metric": "HeatFlux",
                "mean_rel_abs": heat_flux,
                "max_rel_abs": 0.09,
                "final_rel": 0.02,
            },
        ],
    }
    if not include:
        payload["gate_index_include"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_window_rows_excludes_exploratory_and_uses_repo_relative_paths(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_nonlinear_validation_panels")
    old_root = mod.ROOT
    mod.ROOT = tmp_path
    try:
        _write_nonlinear_window_gate_summary(
            tmp_path / "nonlinear_cyclone_gate_summary.json",
            case="cyclone_nonlinear_long_window",
        )
        _write_nonlinear_window_gate_summary(
            tmp_path / "nonlinear_cyclone_short_gate_summary.json",
            case="cyclone_short_nonlinear_window",
            include=False,
        )

        rows = mod.load_window_rows(list(tmp_path.glob("*.json")))
    finally:
        mod.ROOT = old_root

    assert {row["case"] for row in rows} == {"cyclone_nonlinear_long_window"}
    assert {row["metric"] for row in rows} == {"Wg", "Wphi", "HeatFlux"}
    assert {row["artifact"] for row in rows} == {"nonlinear_cyclone_gate_summary.json"}
    assert {row["case_gate_mean_rel"] for row in rows} == {0.10}


def test_nonlinear_window_statistics_main_writes_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_nonlinear_validation_panels")
    _write_nonlinear_window_gate_summary(
        tmp_path / "nonlinear_cyclone_gate_summary.json",
        case="cyclone_nonlinear_long_window",
    )
    _write_nonlinear_window_gate_summary(
        tmp_path / "nonlinear_hsx_gate_summary.json",
        case="hsx_nonlinear_window",
        heat_flux=0.04,
    )
    _write_nonlinear_window_gate_summary(
        tmp_path / "nonlinear_cyclone_short_gate_summary.json",
        case="cyclone_short_nonlinear_window",
        include=False,
        heat_flux=0.5,
    )
    out = tmp_path / "panel.png"

    assert (
        mod.main(
            [
                "window-statistics",
                "--glob",
                str(tmp_path / "*.json"),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".csv").exists()
    assert out.with_suffix(".json").exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["n_cases"] == 2
    assert meta["n_rows"] == 6
    assert meta["all_cases_pass_gate"] is True
    assert meta["all_cases_pass_case_gates"] is True
    assert meta["case_gate_thresholds"]["hsx_nonlinear_window"] == 0.05
    assert "cyclone_short_nonlinear_window" not in set(meta["cases"])


def test_case_specific_window_gates_expose_tighter_release_thresholds(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_nonlinear_validation_panels")
    _write_nonlinear_window_gate_summary(
        tmp_path / "nonlinear_kbm_gate_summary.json",
        case="kbm_nonlinear_window",
        heat_flux=0.019,
        wg=0.01,
        wphi=0.015,
    )
    _write_nonlinear_window_gate_summary(
        tmp_path / "nonlinear_hsx_gate_summary.json",
        case="hsx_nonlinear_window",
        heat_flux=0.049,
    )
    _write_nonlinear_window_gate_summary(
        tmp_path / "nonlinear_cyclone_miller_gate_summary.json",
        case="cyclone_miller_nonlinear_window",
        heat_flux=0.094,
    )
    rows = mod.load_window_rows(list(tmp_path.glob("*.json")))
    by_case = {str(row["case"]): float(row["case_gate_mean_rel"]) for row in rows}

    assert by_case["kbm_nonlinear_window"] == 0.02
    assert by_case["hsx_nonlinear_window"] == 0.05
    assert by_case["cyclone_miller_nonlinear_window"] == 0.095

    out = tmp_path / "case_gates.json"
    mod.write_summary_json(
        rows, out, gate_threshold=0.10, patterns=[str(tmp_path / "*.json")]
    )
    meta = json.loads(out.read_text(encoding="utf-8"))
    assert meta["case_gate_passed"] == {
        "cyclone_miller_nonlinear_window": True,
        "kbm_nonlinear_window": True,
        "hsx_nonlinear_window": True,
    }


# ---- test_parallel_identity_gate_artifacts.py ----

from typing import Any


from spectraxgk.benchmarking.shared import CycloneScanResult
from spectraxgk.workflows.runtime.results import RuntimeLinearScanResult
from tools.artifacts import generate_electrostatic_parallel_gates as electrostatic_gates
from tools.artifacts import generate_linear_rhs_parallel_gates as linear_rhs_gates
from tools.artifacts import generate_parallel_identity_gate as parallel_identity_gate
from tools.artifacts import generate_velocity_parallel_gates as velocity_parallel_gates

hermite_exchange_gate = velocity_parallel_gates
velocity_reduce_gate = velocity_parallel_gates
hermite_ladder_gate = velocity_parallel_gates
periodic_gate = velocity_parallel_gates
field_reduce_gate = electrostatic_gates
diamagnetic_gate = electrostatic_gates
drift_gate = electrostatic_gates
streaming_gate = linear_rhs_gates
electrostatic_gate = linear_rhs_gates
slices_gate = linear_rhs_gates


class _VelocityPlan:
    def __init__(self, shape: tuple[int, ...], pattern: str) -> None:
        self.shape = shape
        self.pattern = pattern

    def to_dict(self) -> dict[str, object]:
        return {
            "state_shape": self.shape,
            "chunks": {"m": 2},
            "active_axes": ("m",),
            "communication_pattern": self.pattern,
        }


class _Grid:
    ky = np.asarray([0.0, 0.3])
    z = np.asarray([0.0, 1.0, 2.0, 3.0])


def _fake_devices(_kind: str | None = None) -> list[object]:
    return [object(), object()]


def _assert_standard_artifacts(
    paths: dict[str, str] | None, out_prefix: Path, csv_token: str
) -> None:
    json_path = Path(paths["json"]) if paths else out_prefix.with_suffix(".json")
    csv_path = Path(paths["csv"]) if paths else out_prefix.with_suffix(".csv")
    png_path = Path(paths["png"]) if paths else out_prefix.with_suffix(".png")
    pdf_path = Path(paths["pdf"]) if paths else out_prefix.with_suffix(".pdf")

    assert json.loads(json_path.read_text(encoding="utf-8"))["identity_passed"] is True
    assert csv_token in csv_path.read_text(encoding="utf-8")
    assert png_path.exists()
    assert pdf_path.exists()


def _runtime_scan(ky_values: np.ndarray, *, workers: int) -> RuntimeLinearScanResult:
    gamma = np.asarray(ky_values, dtype=float) + 1.0
    omega = -(np.asarray(ky_values, dtype=float) + 2.0)
    quasilinear = tuple(
        {
            "ky": float(ky),
            "gamma": float(gamma_i),
            "omega": float(omega_i),
            "kperp_eff2": 0.5 + float(ky),
            "heat_flux_weight_total": 2.0 * float(ky),
            "particle_flux_weight_total": 0.1 * float(ky),
            "amplitude2": 0.3 * float(ky),
            "saturated_heat_flux_total": 0.6 * float(ky),
            "saturated_particle_flux_total": 0.03 * float(ky),
        }
        for ky, gamma_i, omega_i in zip(ky_values, gamma, omega, strict=True)
    )
    return RuntimeLinearScanResult(
        ky=np.asarray(ky_values, dtype=float),
        gamma=gamma,
        omega=omega,
        quasilinear=quasilinear,
        parallel={
            "requested_workers": int(workers),
            "effective_workers": min(int(workers), len(ky_values)),
            "executor": "thread",
            "identity_contract": "test",
            "quasilinear_state_extraction": True,
        },
    )


def test_velocity_field_reduce_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 4, 2, 1, 1)
        return _VelocityPlan(shape, "field_reduce_broadcast")

    def fake_state(_shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.arange(8, dtype=jnp.float32).reshape((1, 4, 2, 1, 1))

    def fake_reduce(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.sum(state, axis=1)

    monkeypatch.setattr(velocity_reduce_gate, "_state", fake_state)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.velocity_field_reduce_reference", fake_reduce
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.velocity_field_reduce_shard_map", fake_reduce
    )

    summary = velocity_reduce_gate.build_velocity_field_reduce_gate(
        shape=(1, 4, 2, 1, 1), requested_devices=2, atol=1.0e-12, rtol=1.0e-10
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["rtol"] == 1.0e-10
    assert summary["max_allowed_error"] > summary["atol"]
    assert summary["max_rel_error"] == 0.0
    assert len(summary["rows"]) == 2


def test_hermite_exchange_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 4, 1, 1, 1)
        return _VelocityPlan(shape, "hermite_ghost_exchange")

    def fake_state(_shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.asarray([[[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]])

    def fake_reference(state):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.asarray([[[[[0.0]]], [[[1.0]]], [[[2.0]]], [[[3.0]]]]]), jnp.asarray(
            [[[[[2.0]]], [[[3.0]]], [[[4.0]]], [[[0.0]]]]]
        )

    monkeypatch.setattr(hermite_exchange_gate, "_state", fake_state)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_neighbor_reference", fake_reference
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_neighbor_shard_map",
        lambda state, plan, devices: fake_reference(state),
    )

    summary = hermite_exchange_gate.build_hermite_exchange_gate(
        shape=(1, 4, 1, 1, 1), requested_devices=2, atol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["plan"]["communication_pattern"] == "hermite_ghost_exchange"
    assert summary["max_lower_abs_error"] == 0.0
    assert summary["max_upper_abs_error"] == 0.0
    assert len(summary["rows"]) == 4


def test_hermite_streaming_ladder_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 4, 1, 1, 1)
        return _VelocityPlan(shape, "hermite_ghost_exchange+field_reduce_broadcast")

    def fake_state(_shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.asarray([[[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]])

    def fake_ladder(state, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.sqrt(state + 1.0)

    def fake_reduce(state, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.sum(state, axis=1)

    monkeypatch.setattr(hermite_ladder_gate, "_state", fake_state)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_streaming_ladder_reference", fake_ladder
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_streaming_ladder_shard_map",
        lambda state, plan, **kwargs: fake_ladder(state),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.velocity_field_reduce_reference", fake_reduce
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.velocity_field_reduce_shard_map",
        lambda state, plan, **kwargs: fake_reduce(state),
    )

    summary = hermite_ladder_gate.build_hermite_streaming_ladder_gate(
        shape=(1, 4, 1, 1, 1),
        requested_devices=2,
        vth=1.7,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["max_ladder_abs_error"] == 0.0
    assert summary["max_ladder_rel_error"] == 0.0
    assert summary["max_reduction_abs_error"] == 0.0
    assert len(summary["rows"]) == 4


def test_periodic_streaming_microkernel_gate_builds_identity_summary(
    monkeypatch,
) -> None:
    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 1, 4, 1, 1, 4)
        return _VelocityPlan(shape, "hermite_ghost_exchange+field_reduce_broadcast")

    def fake_state(shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.ones(shape, dtype=jnp.complex64), jnp.linspace(0.0, 1.0, shape[-1])

    def fake_streaming(state, **_kwargs):  # type: ignore[no-untyped-def]
        return 2.0 * state

    monkeypatch.setattr(periodic_gate, "_state", fake_state)
    monkeypatch.setattr(periodic_gate, "_production_streaming_term", fake_streaming)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.periodic_streaming_reference", fake_streaming
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.periodic_streaming_shard_map",
        lambda state, plan, **kwargs: fake_streaming(state),
    )

    summary = periodic_gate.build_periodic_streaming_microkernel_gate(
        shape=(1, 1, 4, 1, 1, 4),
        requested_devices=2,
        vth=1.7,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["max_sharded_abs_error"] == 0.0
    assert summary["max_sharded_rel_error"] == 0.0
    assert len(summary["rows"]) == 4


def test_electrostatic_field_reduce_gate_builds_identity_summary(monkeypatch) -> None:
    class FakeCache:
        Jl = 1.0
        mask0 = False

    class FakeParams:
        tau_e = 1.0
        charge_sign = 1.0
        density = 1.0
        tz = 1.0

    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            FakeCache(),
            FakeParams(),
            _Grid(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return state * 0.0, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    def fake_phi(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(field_reduce_gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr("spectraxgk.operators.linear.rhs.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan",
        lambda *_args, **_kwargs: _VelocityPlan(
            (1, 4, 2, 1, 4), "hermite_ghost_exchange+field_reduce_broadcast"
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.electrostatic_phi_shard_map", fake_phi
    )

    summary = field_reduce_gate.build_electrostatic_field_reduce_gate(
        requested_devices=2, nx=1, ny=2, nz=4, nl=1, nm=4, atol=1.0e-12, rtol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["phi_norm"] > 0.0
    assert len(summary["rows"]) == 4


def test_electrostatic_diamagnetic_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        class Cache:
            Jl = jnp.ones((1, 1, 2, 1, 4), dtype=jnp.float32)
            b = jnp.zeros((1, 2, 1, 4), dtype=jnp.float32)
            mask0 = jnp.zeros((2, 1, 4), dtype=bool)
            l4 = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
            ky = jnp.asarray([0.0, 0.3], dtype=jnp.float32)

        class Params:
            tau_e = 1.0
            charge_sign = 1.0
            density = 1.0
            tz = 1.0
            R_over_LTi = 6.9
            R_over_Ln = 2.2
            omega_star_scale = 1.0

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            Cache(),
            Params(),
            _Grid(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return 2.0 * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    def fake_phi(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(diamagnetic_gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr("spectraxgk.operators.linear.rhs.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan",
        lambda *_args, **_kwargs: _VelocityPlan(
            (1, 4, 2, 1, 4), "hermite_ghost_exchange+field_reduce_broadcast"
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.electrostatic_phi_shard_map", fake_phi
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.diamagnetic_drive_shard_map",
        lambda state, *_args, **_kwargs: 2.0 * state,
    )

    summary = diamagnetic_gate.build_electrostatic_diamagnetic_gate(
        requested_devices=2, nx=1, ny=2, nz=4, nl=1, nm=4, atol=1.0e-12, rtol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["max_phi_abs_error"] == 0.0
    assert summary["phi_norm"] > 0.0
    assert len(summary["rows"]) == 4


def test_electrostatic_drift_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        class Cache:
            Jl = jnp.ones((1, 1, 2, 1, 4), dtype=jnp.float32)
            mask0 = jnp.zeros((2, 1, 4), dtype=bool)
            bgrad = jnp.ones((4,), dtype=jnp.float32)
            m = jnp.ones((1, 4, 1, 1, 1), dtype=jnp.float32)
            sqrt_m = jnp.ones((1, 4, 1, 1, 1), dtype=jnp.float32)
            sqrt_m_p1 = jnp.ones((1, 4, 1, 1, 1), dtype=jnp.float32)
            cv_d = jnp.ones((2, 1, 4), dtype=jnp.float32)
            gb_d = jnp.ones((2, 1, 4), dtype=jnp.float32)

        setattr(Cache, "l", jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32))

        class Params:
            tau_e = 1.0
            charge_sign = 1.0
            density = 1.0
            tz = 1.0
            vth = 1.0
            omega_d_scale = 1.0

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            Cache(),
            Params(),
            _Grid(),
        )

    def fake_rhs(state, *_args, **kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        terms = kwargs.get("terms")
        scale = (
            float(getattr(terms, "mirror", 0.0))
            + float(getattr(terms, "curvature", 0.0))
            + float(getattr(terms, "gradb", 0.0))
        )
        return scale * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    def fake_phi(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(drift_gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr("spectraxgk.operators.linear.rhs.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr(
        "spectraxgk.operators.linear.moments.build_H", lambda state, *_args, **_kwargs: state
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan",
        lambda *_args, **_kwargs: _VelocityPlan(
            (1, 4, 2, 1, 4), "hermite_ghost_exchange+field_reduce_broadcast"
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.electrostatic_phi_shard_map", fake_phi
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.mirror_drift_shard_map",
        lambda state, *_args, **_kwargs: state,
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.curvature_gradb_drift_shard_map",
        lambda state, *_args, **_kwargs: 2.0 * state,
    )

    summary = drift_gate.build_electrostatic_drift_gate(
        requested_devices=2, nx=1, ny=2, nz=4, nl=1, nm=4, atol=1.0e-12, rtol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["phi_norm"] > 0.0
    assert {row["component"] for row in summary["rows"]} == {
        "mirror",
        "curvature_gradb",
        "total",
    }


def test_linear_rhs_streaming_gate_builds_identity_summary(monkeypatch) -> None:
    class FakeCache:
        kz = [0.0, 1.0, -2.0, -1.0]

    class FakeParams:
        vth = 1.0

    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            FakeCache(),
            FakeParams(),
            _Grid(),
            object(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return -2.0 * state, jnp.zeros((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(streaming_gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr("spectraxgk.operators.linear.rhs.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan",
        lambda *_args, **_kwargs: _VelocityPlan(
            (1, 4, 2, 1, 4), "hermite_ghost_exchange+field_reduce_broadcast"
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.periodic_streaming_shard_map",
        lambda state, *_args, **_kwargs: 2.0 * state,
    )

    summary = streaming_gate.build_linear_rhs_streaming_gate(
        requested_devices=2, nx=1, ny=2, nz=4, nl=1, nm=4, atol=1.0e-12, rtol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["phi_norm"] == 0.0
    assert len(summary["rows"]) == 4


@pytest.mark.parametrize(
    ("gate", "builder"),
    [
        (electrostatic_gate, "build_linear_rhs_streaming_electrostatic_gate"),
        (slices_gate, "build_linear_rhs_electrostatic_slices_gate"),
    ],
    ids=["streaming_electrostatic", "electrostatic_slices"],
)
def test_linear_rhs_electrostatic_routes_build_identity_summary(
    monkeypatch, gate: Any, builder: str
) -> None:
    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            object(),
            object(),
            _Grid(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return 3.0 * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", _fake_devices)
    monkeypatch.setattr("spectraxgk.operators.linear.rhs.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("spectraxgk.solvers.linear.parallel.linear_rhs_parallel_cached", fake_rhs)
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan",
        lambda *_args, **_kwargs: _VelocityPlan(
            (1, 4, 2, 1, 4), "hermite_ghost_exchange+field_reduce_broadcast"
        ),
    )

    summary = getattr(gate, builder)(
        requested_devices=2, nx=1, ny=2, nz=4, nl=1, nm=4, atol=1.0e-12, rtol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["max_phi_abs_error"] == 0.0
    assert summary["phi_norm"] > 0.0
    assert len(summary["rows"]) == 4


def test_parallel_ky_scan_gate_builds_identity_summary(monkeypatch) -> None:
    calls: list[int] = []

    def fake_scan(ky_values, *, ky_batch, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(int(ky_batch))
        result = CycloneScanResult(
            ky=np.asarray(ky_values, dtype=float),
            gamma=np.asarray([0.1, 0.2], dtype=float),
            omega=np.asarray([0.3, 0.4], dtype=float),
        )
        return result, 4.0 if ky_batch == 1 else 2.0

    monkeypatch.setattr(parallel_identity_gate, "_timed_cyclone_scan", fake_scan)
    summary = parallel_identity_gate.build_parallel_ky_scan_gate(
        ky_values=np.asarray([0.1, 0.2]),
        serial_batch=1,
        parallel_batch=2,
        gamma_rtol=1.0e-12,
        omega_atol=1.0e-12,
        steps=4,
        dt=0.1,
        nx=1,
        ny=4,
        nz=8,
        nlaguerre=2,
        nhermite=3,
    )

    assert calls == [1, 2]
    assert summary["identity_passed"] is True
    assert summary["observed_speedup"] == 2.0
    assert summary["max_gamma_rel_error"] == 0.0
    assert len(summary["rows"]) == 2


def test_logical_cpu_parallel_scan_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_devices(requested_devices: int):  # type: ignore[no-untyped-def]
        return [object()] * requested_devices

    def fake_scan(ky_values, *, batch_size, devices):  # type: ignore[no-untyped-def]
        assert devices
        ky = np.asarray(ky_values, dtype=float)
        return {
            "gamma": ky + 0.1,
            "omega": -ky,
            "kperp2": ky**2 + 0.08,
            "ql_proxy": (ky + 0.1) / (ky**2 + 0.08),
        }, 4.0 if batch_size == 1 else 2.0

    monkeypatch.setattr(parallel_identity_gate, "_select_devices", fake_devices)
    monkeypatch.setattr(parallel_identity_gate, "_timed_scan_model", fake_scan)

    summary = parallel_identity_gate.build_logical_cpu_parallel_scan_gate(
        ky_values=np.asarray([0.1, 0.2]),
        serial_batch=1,
        parallel_batch=2,
        requested_devices=2,
        gamma_rtol=1.0e-12,
        omega_atol=1.0e-12,
        ql_rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["observed_speedup"] == 2.0
    assert summary["device_parallel_config"]["strategy"] == "device_batch"
    assert summary["device_parallel_config"]["num_devices"] == 2
    assert summary["max_ql_rel_error"] == 0.0
    assert len(summary["rows"]) == 2


def test_quasilinear_runtime_parallel_gate_builds_identity_summary(monkeypatch) -> None:
    calls: list[int] = []

    def fake_timed_scan(_cfg, ky_values, *, workers, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(int(workers))
        return _runtime_scan(np.asarray(ky_values, dtype=float), workers=workers), (
            4.0 if workers == 1 else 2.0
        )

    monkeypatch.setattr(parallel_identity_gate, "_timed_runtime_scan", fake_timed_scan)
    summary = parallel_identity_gate.build_quasilinear_runtime_parallel_gate(
        ky_values=np.asarray([0.1, 0.2]),
        workers=2,
        rtol=1.0e-12,
        atol=1.0e-12,
        solver="krylov",
        nx=1,
        ny=8,
        nz=12,
        nlaguerre=2,
        nhermite=2,
    )

    assert calls == [1, 2]
    assert summary["identity_passed"] is True
    assert summary["observed_speedup"] == 2.0
    assert summary["serial_parallel_metadata"]["requested_workers"] == 2
    assert len(summary["rows"]) == 2
    assert summary["rows"][0]["heat_flux_weight_total_abs_error"] == 0.0


def _writer_name(prefix: str) -> str:
    if prefix == "parallel_gate":
        return "write_parallel_ky_scan_artifacts"
    if prefix == "logical_cpu_parallel_gate":
        return "write_logical_cpu_parallel_scan_artifacts"
    if prefix == "ql_parallel_gate":
        return "write_quasilinear_runtime_parallel_artifacts"
    return "write_artifacts"


@pytest.mark.parametrize(
    ("gate", "prefix", "summary", "csv_token"),
    [
        (
            velocity_reduce_gate,
            "velocity_field_reduce_gate",
            {
                "rows": [
                    {
                        "ky_index": 0,
                        "reduced_real": 1.0,
                        "reference_real": 1.0,
                        "abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-7,
                "max_allowed_error": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            hermite_exchange_gate,
            "hermite_exchange_gate",
            {
                "rows": [
                    {
                        "m": 0,
                        "center_real": 1.0,
                        "lower_real": 0.0,
                        "upper_real": 2.0,
                        "lower_reference_real": 0.0,
                        "upper_reference_real": 2.0,
                        "lower_abs_error": 0.0,
                        "upper_abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "identity_passed": True,
            },
            "lower_abs_error",
        ),
        (
            hermite_ladder_gate,
            "hermite_streaming_ladder_gate",
            {
                "rows": [
                    {
                        "m": 0,
                        "state_real": 1.0,
                        "ladder_real": 2.0,
                        "reference_real": 2.0,
                        "abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            periodic_gate,
            "periodic_streaming_microkernel_gate",
            {
                "rows": [
                    {
                        "m": 0,
                        "state_abs": 1.0,
                        "production_abs": 2.0,
                        "sharded_abs": 2.0,
                        "abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            field_reduce_gate,
            "electrostatic_field_reduce_gate",
            {
                "rows": [
                    {
                        "z_index": 0,
                        "serial_abs": 2.0,
                        "sharded_abs": 2.0,
                        "abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            diamagnetic_gate,
            "electrostatic_diamagnetic_gate",
            {
                "rows": [
                    {"m": 0, "serial_norm": 2.0, "sharded_norm": 2.0, "abs_error": 0.0}
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            drift_gate,
            "electrostatic_drift_gate",
            {
                "rows": [
                    {
                        "component": "mirror",
                        "serial_norm": 2.0,
                        "sharded_norm": 2.0,
                        "abs_error": 0.0,
                        "rel_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            streaming_gate,
            "linear_rhs_streaming_gate",
            {
                "rows": [
                    {
                        "m": 0,
                        "production_abs": 2.0,
                        "sharded_abs": 2.0,
                        "abs_error": 0.0,
                    }
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            electrostatic_gate,
            "linear_rhs_streaming_electrostatic_gate",
            {
                "rows": [
                    {"m": 0, "serial_abs": 2.0, "sharded_abs": 2.0, "abs_error": 0.0}
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            slices_gate,
            "linear_rhs_electrostatic_slices_gate",
            {
                "rows": [
                    {"m": 0, "serial_norm": 2.0, "sharded_norm": 2.0, "abs_error": 0.0}
                ],
                "atol": 1.0e-8,
                "rtol": 1.0e-8,
                "identity_passed": True,
            },
            "abs_error",
        ),
        (
            parallel_identity_gate,
            "parallel_gate",
            {
                "rows": [
                    {
                        "ky": 0.1,
                        "serial_gamma": 0.1,
                        "batched_gamma": 0.1,
                        "gamma_rel_error": 0.0,
                        "serial_omega": 0.2,
                        "batched_omega": 0.2,
                        "omega_abs_error": 0.0,
                    }
                ],
                "gamma_rtol": 1.0e-8,
                "omega_atol": 1.0e-8,
                "serial_elapsed_s": 2.0,
                "batched_elapsed_s": 1.0,
                "observed_speedup": 2.0,
                "identity_passed": True,
            },
            "gamma_rel_error",
        ),
        (
            parallel_identity_gate,
            "logical_cpu_parallel_gate",
            {
                "rows": [
                    {
                        "ky": 0.1,
                        "serial_gamma": 0.2,
                        "batched_gamma": 0.2,
                        "gamma_rel_error": 0.0,
                        "serial_omega": -0.1,
                        "batched_omega": -0.1,
                        "omega_abs_error": 0.0,
                        "serial_ql_proxy": 1.5,
                        "batched_ql_proxy": 1.5,
                        "ql_rel_error": 0.0,
                    }
                ],
                "gamma_rtol": 1.0e-8,
                "omega_atol": 1.0e-8,
                "ql_rtol": 1.0e-8,
                "serial_elapsed_s": 2.0,
                "batched_elapsed_s": 1.0,
                "observed_speedup": 2.0,
                "identity_passed": True,
            },
            "ql_rel_error",
        ),
        (
            parallel_identity_gate,
            "ql_parallel_gate",
            {
                "identity_passed": True,
                "observed_speedup": 2.0,
                "atol": 1.0e-12,
                "rows": [
                    {
                        "ky": 0.1,
                        "serial_heat_flux_weight_total": 0.2,
                        "parallel_heat_flux_weight_total": 0.2,
                        "heat_flux_weight_total_abs_error": 0.0,
                        "serial_saturated_heat_flux_total": 0.06,
                        "parallel_saturated_heat_flux_total": 0.06,
                        "saturated_heat_flux_total_abs_error": 0.0,
                    }
                ],
            },
            "heat_flux_weight_total_abs_error",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_parallel_identity_gate_writes_artifacts(
    tmp_path: Path, gate: Any, prefix: str, summary: dict[str, object], csv_token: str
) -> None:
    out = tmp_path / prefix
    paths = getattr(gate, _writer_name(prefix))(summary, out)
    _assert_standard_artifacts(paths, out, csv_token)
