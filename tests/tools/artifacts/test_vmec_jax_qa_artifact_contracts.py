from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from support.paths import REPO_ROOT, load_artifact_tool


ROOT = REPO_ROOT
STRATEGY_SCRIPT = ROOT / "tools" / "artifacts" / "build_qa_optimizer_strategy_report.py"
strategy_mod = load_artifact_tool("build_qa_optimizer_strategy_report")
candidate_mod = load_artifact_tool("build_vmec_jax_qa_transport_candidate_comparison")
status_mod = load_artifact_tool("build_vmec_jax_qa_transport_optimization_status")
full_sweep_mod = load_artifact_tool("build_vmec_jax_qa_full_sweep_panel")
gradient_mod = load_artifact_tool("build_vmec_jax_transport_gradient_diagnostic")


def _write_panel(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "qa_baseline_scipy",
                        "label": "QA baseline",
                        "gate_passed": True,
                        "diagnostic_gate_passed": True,
                        "gate_blockers": [],
                        "setup": {
                            "transport_kind": "nonlinear_window_heat_flux",
                            "spectrax_weight": 0.05,
                            "optimizer": {"method": "scipy"},
                        },
                        "history": {
                            "objective_initial": 100.0,
                            "objective_final": 0.01,
                            "aspect_final": 5.0,
                            "iota_final": 0.4102,
                            "qs_final": 1.0e-5,
                            "nfev": 48,
                        },
                    },
                    {
                        "case_id": "growth_from_strict_baseline",
                        "label": "growth from strict QA",
                        "gate_passed": False,
                        "diagnostic_gate_passed": True,
                        "gate_blockers": ["mean_iota"],
                        "setup": {
                            "transport_kind": "growth",
                            "spectrax_weight": 0.1,
                            "optimizer": {"method": "scalar_trust"},
                        },
                        "history": {
                            "objective_initial": 1.0,
                            "objective_final": 0.25,
                            "aspect_final": 5.004,
                            "iota_final": 0.4099,
                            "qs_final": 5.0e-4,
                            "transport_metric_final": 0.07,
                            "message": "radius too small",
                            "nfev": 16,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_landscape(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": "m0p1",
                        "relative_fraction": -0.1,
                        "coefficient_value": 0.9,
                        "reduced_metrics": {
                            "growth": 0.2,
                            "quasilinear_flux_mixing_length": 0.4,
                        },
                    },
                    {
                        "label": "0",
                        "relative_fraction": 0.0,
                        "coefficient_value": 1.0,
                        "reduced_metrics": {
                            "growth": 0.3,
                            "quasilinear_flux_mixing_length": 0.5,
                        },
                    },
                    {
                        "label": "p0p1",
                        "relative_fraction": 0.1,
                        "coefficient_value": 1.1,
                        "reduced_metrics": {
                            "growth": 0.4,
                            "quasilinear_flux_mixing_length": 0.6,
                        },
                    },
                ],
                "nonlinear_ensemble_points": [
                    {
                        "coefficient_value": 0.9,
                        "mean": 11.0,
                        "sem": 0.2,
                        "passed": True,
                    },
                    {
                        "coefficient_value": 1.0,
                        "mean": 10.0,
                        "sem": 0.1,
                        "passed": True,
                    },
                    {"coefficient_value": 1.1, "mean": 7.0, "sem": 0.3, "passed": True},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_strategy_report_keeps_nonlinear_optimization_fail_closed(
    tmp_path: Path,
) -> None:
    panel = tmp_path / "panel.json"
    landscape = tmp_path / "landscape.json"
    _write_panel(panel)
    _write_landscape(landscape)

    report = strategy_mod.build_report(panel, landscape)

    assert report["kind"] == "vmec_jax_qa_optimizer_strategy_report"
    assert (
        report["gates"]["deterministic_transport_rows_all_strict_gates_pass"] is False
    )
    assert report["gates"]["has_converged_long_window_landscape"] is True
    assert report["gates"]["has_admitted_long_window_landscape"] is False
    assert report["gates"]["has_material_landscape_reduction_direction"] is True
    assert report["gates"]["nonlinear_absolute_optimization_promoted"] is False
    assert report["cases"][1]["iota_shortfall"] > 0.0
    assert report["landscape"]["n_converged_nonlinear_points"] == 3
    assert report["landscape"]["best_point"]["label"] == "p0p1"
    assert "noise/convergence diagnostic" in report["claim_scope"]

    methods = {item["method"] for item in report["optimizer_recommendations"]}
    assert "vmec_jax_exact_discrete_adjoint_least_squares" in methods
    assert (
        "spsa_common_random_numbers_then_cma_es_or_bo_for_low_dimensional_projected_controls"
        in methods
    )


def test_strategy_report_exports_public_qa_transport_claim_boundaries(
    tmp_path: Path,
) -> None:
    panel = tmp_path / "panel.json"
    landscape = tmp_path / "landscape.json"
    _write_panel(panel)
    _write_landscape(landscape)

    report = strategy_mod.build_report(panel, landscape)
    boundaries = {row["transport_kind"]: row for row in report["claim_boundaries"]}

    assert set(boundaries) == {
        "growth",
        "quasilinear_flux",
        "nonlinear_window_heat_flux",
    }
    assert all(
        row["nonlinear_turbulent_flux_claim"] is False for row in boundaries.values()
    )
    assert "linear growth-rate residual" in boundaries["growth"]["claim_boundary"]
    assert (
        "not an absolute flux predictor"
        in boundaries["quasilinear_flux"]["claim_boundary"]
    )
    assert (
        "not a converged nonlinear transport average"
        in boundaries["nonlinear_window_heat_flux"]["claim_boundary"]
    )
    assert all(
        any("matched" in requirement for requirement in row["promotion_requires"])
        for row in boundaries.values()
    )

    readme = (ROOT / "examples" / "optimization" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "Claim boundary" in readme
    for kind, row in boundaries.items():
        script = ROOT / row["script"]
        text = script.read_text(encoding="utf-8")

        assert script.name in readme
        assert f'SPECTRAX_KIND = "{kind}"' in text
        assert "WRITE_LONG_NONLINEAR_AUDIT_CONFIGS = True" in text
        assert "RUN_LONG_NONLINEAR_AUDIT_COMMANDS = False" in text
        assert 'NONLINEAR_AUDIT_HORIZONS = "700,1100,1500"' in text
        assert "NONLINEAR_AUDIT_WINDOW_TMIN = 1100.0" in text
        assert "NONLINEAR_AUDIT_WINDOW_TMAX = 1500.0" in text
        assert "build_matched_nonlinear_transport_comparison.py" in text


def test_strategy_report_cli_writes_artifacts(tmp_path: Path) -> None:
    panel = tmp_path / "panel.json"
    landscape = tmp_path / "landscape.json"
    out_prefix = tmp_path / "strategy"
    _write_panel(panel)
    _write_landscape(landscape)

    completed = subprocess.run(
        [
            sys.executable,
            str(STRATEGY_SCRIPT),
            "--panel-json",
            str(panel),
            "--landscape-json",
            str(landscape),
            "--out-prefix",
            str(out_prefix),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    status = json.loads(completed.stdout)
    assert Path(status["out_json"]).exists()
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()



def test_default_candidate_sources_are_authoritative_sidecar_or_payload_fallback() -> (
    None
):
    assert "vmec_jax_qa_transport_authoritative_sidecar" in str(
        candidate_mod.DEFAULT_CONSTRAINTS_DIR
    )
    assert "vmec_jax_qa_transport_authoritative_sidecar" in str(
        candidate_mod.DEFAULT_TRANSPORT_DIR
    )
    assert "vmec_jax_qa_promotion_smoke" not in str(candidate_mod.DEFAULT_CONSTRAINTS_DIR)
    assert "vmec_jax_qa_promotion_smoke" not in str(candidate_mod.DEFAULT_TRANSPORT_DIR)
    assert (
        candidate_mod.DEFAULT_PAYLOAD_JSON.name
        == "vmec_jax_qa_transport_candidate_comparison.json"
    )


def test_load_or_build_payload_falls_back_to_tracked_payload_in_clean_clone(
    tmp_path: Path,
) -> None:
    payload_json = tmp_path / "candidate.json"
    expected = {
        "kind": "vmec_jax_qa_transport_candidate_comparison",
        "summary": {"from_payload": True},
    }
    payload_json.write_text(json.dumps(expected), encoding="utf-8")
    args = argparse.Namespace(
        constraints_dir=tmp_path / "missing_constraints",
        transport_dir=tmp_path / "missing_transport",
        payload_json=payload_json,
    )

    assert candidate_mod._load_or_build_payload(args) == expected


def _history(root: Path, *, qs: float = 0.02) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "history.json").write_text(
        json.dumps(
            {
                "aspect_initial": 6.1,
                "aspect_final": 6.0,
                "iota_initial": 0.42,
                "iota_final": 0.427,
                "qs_initial": 0.03,
                "qs_final": qs,
                "objective_initial": 2.0,
                "objective_final": 0.5,
                "history": [{"objective": 2.0}, {"objective": 0.5}],
                "nfev": 3,
                "success": True,
                "message": "ok",
                "total_wall_time_s": 1.0,
            }
        ),
        encoding="utf-8",
    )


def _solved_gate(root: Path, *, passed: bool = True, qs: float = 0.02) -> None:
    checks = {
        "aspect": {
            "value": 6.0,
            "target": 6.0,
            "absolute_error": 0.0,
            "absolute_tolerance": 0.05,
            "passed": True,
        },
        "mean_iota": {
            "value": 0.427,
            "minimum_abs": 0.41,
            "margin": 0.017,
            "passed": True,
        },
        "quasisymmetry": {
            "value": qs,
            "maximum": 0.05,
            "margin": 0.05 - qs,
            "source": "vmec_jax_state",
            "passed": qs <= 0.05,
        },
        "iota_profile": {
            "minimum_iotas_excluding_axis": 0.414,
            "minimum_iotaf": 0.412,
            "floor": 0.41,
            "source": "vmec_jax_state",
            "passed": True,
        },
    }
    if not passed:
        checks["quasisymmetry"]["passed"] = False
    (root / "solved_wout_gate.json").write_text(
        json.dumps(
            {
                "kind": "vmec_jax_solved_wout_candidate_gate",
                "passed": passed,
                "checks": checks,
                "next_action": "candidate may proceed" if passed else "do not promote",
            }
        ),
        encoding="utf-8",
    )


def _wout_reproducibility_gate(root: Path, *, passed: bool) -> None:
    (root / "wout_reproducibility_gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "checks": {
                    "mean_iota_reproducibility": {
                        "passed": passed,
                        "absolute_error": 0.0 if passed else 1.5e-3,
                        "absolute_tolerance": 5.0e-4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _rerun_wout_admission_gate(root: Path, *, passed: bool) -> None:
    (root / "rerun_wout_admission_gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "checks": {
                    "aspect": {"passed": passed},
                    "mean_iota": {"passed": passed},
                    "iota_profile": {"passed": passed},
                    "quasisymmetry": {"passed": passed},
                },
            }
        ),
        encoding="utf-8",
    )
    if passed:
        (root / "wout_final_rerun.nc").write_bytes(b"authoritative-rerun-wout")


def test_payload_admits_only_authoritative_solved_wout_gates(
    tmp_path: Path, monkeypatch
) -> None:
    constraints = tmp_path / "constraints"
    transport = tmp_path / "transport"
    _history(constraints)
    _history(transport)
    _solved_gate(constraints, passed=True)
    monkeypatch.setattr(
        candidate_mod,
        "_load_iota_profiles",
        lambda _root, *, wout_name="wout_final.nc": (
            np.asarray([0.0, 0.414, 0.427]),
            np.asarray([0.412, 0.421]),
        ),
    )

    payload = candidate_mod.build_payload(constraints, transport)
    branches = {branch["label"]: branch for branch in payload["branches"]}

    assert (
        payload["iota_gate_policy"]
        == "lower_bound_admission_not_exact_upstream_mean_iota_target"
    )
    assert payload["mean_iota_lower_bound"] == 0.41
    assert payload["iota_profile_floor"] == 0.41
    assert payload["legacy_target_iota_fields_are_lower_bounds"] is True
    assert payload["target_mean_iota"] == payload["mean_iota_lower_bound"]
    assert payload["target_iota_profile_floor"] == payload["iota_profile_floor"]
    assert (
        branches["QA constraints"]["admitted_for_long_window_nonlinear_audit"] is True
    )
    assert branches["QA constraints"]["gate_source"] == "solved_wout_gate.json"
    assert branches["QA + SPECTRAX-GK transport"]["gate_reported_passed"] is True
    assert (
        branches["QA + SPECTRAX-GK transport"][
            "admitted_for_long_window_nonlinear_audit"
        ]
        is False
    )
    assert branches["QA + SPECTRAX-GK transport"]["admission_blockers"] == [
        "non_authoritative_reconstructed_gate"
    ]
    assert payload["summary"]["transport_candidate_admitted"] is False
    assert (
        payload["summary"]["transport_optimization_status"]
        == "blocked_before_transport_claim"
    )
    assert payload["summary"]["all_branches_passed_solved_wout_gate"] is False


def test_payload_admits_transport_candidate_with_authoritative_gate(
    tmp_path: Path, monkeypatch
) -> None:
    constraints = tmp_path / "constraints"
    transport = tmp_path / "transport"
    _history(constraints)
    _history(transport)
    _solved_gate(constraints, passed=True)
    _solved_gate(transport, passed=True)
    monkeypatch.setattr(
        candidate_mod,
        "_load_iota_profiles",
        lambda _root, *, wout_name="wout_final.nc": (
            np.asarray([0.0, 0.414, 0.427]),
            np.asarray([0.412, 0.421]),
        ),
    )

    payload = candidate_mod.build_payload(constraints, transport)

    assert payload["summary"]["all_branches_passed_solved_wout_gate"] is True
    assert payload["summary"]["all_branches_have_authoritative_gate"] is True
    assert payload["summary"]["ready_for_long_window_nonlinear_audit"] == [
        "QA constraints",
        "QA + SPECTRAX-GK transport",
    ]
    assert payload["summary"]["transport_candidate_admitted"] is True


def test_failed_wout_reproducibility_gate_blocks_authoritative_transport_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    constraints = tmp_path / "constraints"
    transport = tmp_path / "transport"
    _history(constraints)
    _history(transport)
    _solved_gate(constraints, passed=True)
    _solved_gate(transport, passed=True)
    _wout_reproducibility_gate(transport, passed=False)
    monkeypatch.setattr(
        candidate_mod,
        "_load_iota_profiles",
        lambda _root, *, wout_name="wout_final.nc": (
            np.asarray([0.0, 0.414, 0.427]),
            np.asarray([0.412, 0.421]),
        ),
    )

    payload = candidate_mod.build_payload(constraints, transport)
    branches = {branch["label"]: branch for branch in payload["branches"]}
    transport_branch = branches["QA + SPECTRAX-GK transport"]

    assert transport_branch["gate_reported_passed"] is True
    assert transport_branch["gate_is_authoritative"] is True
    assert transport_branch["wout_reproducibility_gate_passed"] is False
    assert transport_branch["admitted_for_long_window_nonlinear_audit"] is False
    assert "wout_reproducibility_gate_failed" in transport_branch["admission_blockers"]
    assert payload["summary"]["transport_candidate_admitted"] is False
    assert payload["summary"]["all_branches_passed_solved_wout_gate"] is False


def test_authoritative_rerun_wout_gate_admits_transport_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    constraints = tmp_path / "constraints"
    transport = tmp_path / "transport"
    _history(constraints)
    _history(transport)
    _solved_gate(constraints, passed=True)
    _solved_gate(transport, passed=True)
    _wout_reproducibility_gate(transport, passed=False)
    _rerun_wout_admission_gate(transport, passed=True)
    monkeypatch.setattr(
        candidate_mod,
        "_load_iota_profiles",
        lambda _root, *, wout_name="wout_final.nc": (
            np.asarray([0.0, 0.414, 0.427]),
            np.asarray([0.412, 0.421]),
        ),
    )

    payload = candidate_mod.build_payload(constraints, transport)
    branches = {branch["label"]: branch for branch in payload["branches"]}
    transport_branch = branches["QA + SPECTRAX-GK transport"]

    assert transport_branch["wout_reproducibility_gate_passed"] is False
    assert transport_branch["rerun_wout_admission_gate_passed"] is True
    assert transport_branch["uses_authoritative_rerun_wout"] is True
    assert transport_branch["authoritative_wout"].endswith("wout_final_rerun.nc")
    assert transport_branch["admitted_for_long_window_nonlinear_audit"] is True
    assert (
        "wout_reproducibility_gate_failed" not in transport_branch["admission_blockers"]
    )
    assert payload["summary"]["transport_candidate_admitted"] is True


def test_candidate_comparison_plot_handles_normalized_gate_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    constraints = tmp_path / "constraints"
    transport = tmp_path / "transport"
    _history(constraints)
    _history(transport, qs=0.04)
    _solved_gate(constraints, passed=True)
    _solved_gate(transport, passed=False, qs=0.08)
    monkeypatch.setattr(
        candidate_mod,
        "_load_iota_profiles",
        lambda _root, *, wout_name="wout_final.nc": (
            np.asarray([0.0, 0.414, 0.427]),
            np.asarray([0.412, 0.421]),
        ),
    )
    payload = candidate_mod.build_payload(constraints, transport)
    out = tmp_path / "panel.png"

    candidate_mod.plot_payload(payload, out)

    assert out.exists()
    assert out.stat().st_size > 0



def _candidate(
    root: Path,
    *,
    passed: bool,
    metric: float | None,
    qs: float = 0.02,
    wout_reproducibility_passed: bool | None = None,
    rerun_wout_admission_passed: bool | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective_initial": 2.0,
        "objective_final": 1.0,
        "aspect_final": 6.0,
        "iota_final": 0.427,
        "qs_final": qs,
        "total_wall_time_s": 1.0,
    }
    if metric is not None:
        payload["transport_metric_final"] = metric
        payload["transport_metric_kind"] = "nonlinear_window_heat_flux"
    (root / "history.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "solved_wout_gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "checks": {
                    "aspect": {
                        "passed": True,
                        "absolute_tolerance": 0.05,
                        "absolute_error": 0.0,
                    },
                    "mean_iota": {"passed": True, "margin": 0.017},
                    "iota_profile": {
                        "passed": passed,
                        "minimum_iotas_excluding_axis": 0.414 if passed else 0.40,
                        "floor": 0.41,
                    },
                    "quasisymmetry": {
                        "passed": passed,
                        "margin": 0.05 - qs,
                    },
                },
                "next_action": "candidate may proceed" if passed else "do not promote",
            }
        ),
        encoding="utf-8",
    )
    if wout_reproducibility_passed is not None:
        (root / "wout_reproducibility_gate.json").write_text(
            json.dumps(
                {
                    "passed": wout_reproducibility_passed,
                    "checks": {
                        "mean_iota_reproducibility": {
                            "passed": wout_reproducibility_passed,
                            "absolute_error": 0.0
                            if wout_reproducibility_passed
                            else 1.5e-3,
                            "absolute_tolerance": 5.0e-4,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
    if rerun_wout_admission_passed is not None:
        (root / "rerun_wout_admission_gate.json").write_text(
            json.dumps(
                {
                    "passed": rerun_wout_admission_passed,
                    "checks": {
                        "aspect": {"passed": rerun_wout_admission_passed},
                        "mean_iota": {"passed": rerun_wout_admission_passed},
                        "iota_profile": {"passed": rerun_wout_admission_passed},
                        "quasisymmetry": {"passed": rerun_wout_admission_passed},
                    },
                }
            ),
            encoding="utf-8",
        )
        if rerun_wout_admission_passed:
            (root / "wout_final_rerun.nc").write_bytes(b"authoritative-rerun-wout")


def _supporting_artifacts(
    tmp_path: Path,
    *,
    nonlinear_claim_level: str = status_mod.EXPECTED_NONLINEAR_AUDIT_CLAIM_LEVEL,
) -> dict[str, Path]:
    line_search = tmp_path / "line_search.json"
    line_search.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "objective": "growth",
                        "passed": True,
                        "initial_objective": 2.0,
                        "final_objective": 1.8,
                        "relative_reduction": 0.1,
                        "initial_update_direction": "negative_delta",
                    },
                    {
                        "objective": "quasilinear_flux",
                        "passed": True,
                        "initial_objective": 3.0,
                        "final_objective": 2.4,
                        "relative_reduction": 0.2,
                        "initial_update_direction": "negative_delta",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ql_rule = tmp_path / "ql_rule.json"
    ql_rule.write_text(
        json.dumps(
            {
                "rules": {
                    "linear_weight": {
                        "label": "linear weight",
                        "holdout_mean_abs_relative_error": 0.8,
                        "holdout_gate_passed": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    ql_model = tmp_path / "ql_model.json"
    ql_model.write_text(
        json.dumps(
            {
                "passed": True,
                "required_candidate": "spectral_envelope_ridge",
                "metrics": {"candidate_mean_abs_relative_error": 0.2},
            }
        ),
        encoding="utf-8",
    )
    nonlinear = tmp_path / "nonlinear.json"
    nonlinear.write_text(
        json.dumps(
            {
                "passed": True,
                "claim_level": nonlinear_claim_level,
                "comparison": {
                    "baseline_mean": 12.0,
                    "optimized_mean": 9.0,
                    "relative_reduction": 0.25,
                    "uncertainty_separation_sigma": 4.0,
                },
            }
        ),
        encoding="utf-8",
    )
    landscape_admission = tmp_path / "landscape_admission.json"
    landscape_admission.write_text(
        json.dumps(
            {
                "kind": "nonlinear_landscape_admission_report",
                "passed": True,
                "claim_scope": "selected replicated nonlinear landscape admission",
                "next_action": "use selected direction",
                "policy": {
                    "minimum_relative_reduction": 0.05,
                    "minimum_sample_count": 18,
                },
                "selected_candidate": {
                    "admitted": True,
                    "relative_reduction": 0.12,
                    "admission_blockers": [],
                },
            }
        ),
        encoding="utf-8",
    )
    positive_prelaunch = tmp_path / "positive_prelaunch.json"
    positive_prelaunch.write_text(
        json.dumps(
            {
                "kind": "vmec_jax_reduced_nonlinear_audit_prelaunch_report",
                "passed": True,
                "claim_scope": "positive reduced prelaunch",
                "next_action": "launch replicated audit",
                "relative_reduced_reduction": 0.05,
                "required_relative_reduced_reduction": 0.04,
                "blockers": [],
                "objective_sample_summary": {"sample_count": 18},
            }
        ),
        encoding="utf-8",
    )
    negative_prelaunch = tmp_path / "negative_prelaunch.json"
    negative_prelaunch.write_text(
        json.dumps(
            {
                "kind": "vmec_jax_reduced_nonlinear_audit_prelaunch_report",
                "passed": False,
                "claim_scope": "negative reduced prelaunch",
                "next_action": "do not launch expensive audit",
                "relative_reduced_reduction": 0.02,
                "required_relative_reduced_reduction": 0.04,
                "blockers": ["insufficient_reduced_margin_for_nonlinear_audit"],
                "objective_sample_summary": {"sample_count": 18},
            }
        ),
        encoding="utf-8",
    )
    campaign_admission = tmp_path / "campaign_admission.json"
    campaign_admission.write_text(
        json.dumps(
            {
                "kind": "vmec_jax_nonlinear_campaign_admission_report",
                "passed": True,
                "claim_scope": "next nonlinear optimizer-campaign admission only",
                "next_action": "launch bounded campaign",
                "blockers": [],
                "policy": {"minimum_landscape_relative_reduction": 0.1},
                "selected_landscape_candidate": {"relative_reduction": 0.12},
                "gates": [
                    {
                        "metric": "reduced_objective_sample_coverage",
                        "passed": True,
                        "value": 18,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "line_search_json": line_search,
        "ql_rule_json": ql_rule,
        "ql_model_json": ql_model,
        "nonlinear_audit_json": nonlinear,
        "landscape_admission_json": landscape_admission,
        "positive_prelaunch_json": positive_prelaunch,
        "negative_prelaunch_json": negative_prelaunch,
        "campaign_admission_json": campaign_admission,
    }


def test_build_payload_separates_gate_failures_from_transport_metrics(
    tmp_path: Path,
) -> None:
    constraints = tmp_path / "constraints"
    direct = tmp_path / "direct"
    projected_base = tmp_path / "projected_base"
    projected_step = tmp_path / "projected_step"
    _candidate(constraints, passed=True, metric=None)
    _candidate(direct, passed=False, metric=None, qs=1.0)
    _candidate(projected_base, passed=True, metric=0.1)
    _candidate(projected_step, passed=True, metric=0.11)

    payload = status_mod.build_payload(
        constraints_dir=constraints,
        direct_transport_dir=direct,
        projected_baseline_dir=projected_base,
        projected_step_dir=projected_step,
        **_supporting_artifacts(tmp_path),
    )

    assert payload["summary"]["qa_baseline_gate_passed"] is True
    assert payload["summary"]["direct_scalar_transport_blocked"] is True
    assert payload["summary"]["projected_transport_gate_passed"] is True
    assert payload["summary"]["projected_transport_improved"] is False
    assert payload["summary"]["quasilinear_model_selection_passed"] is True
    assert payload["summary"]["simple_quasilinear_absolute_flux_promoted"] is False
    assert payload["summary"]["long_window_nonlinear_audit_passed"] is True
    assert payload["summary"]["nonlinear_prelaunch_policy_ready"] is True
    assert payload["summary"]["negative_reference_blocks_weak_margin"] is True
    assert (
        payload["summary"]["claim_evidence_level"]
        == "scoped_matched_replicated_nonlinear_audit"
    )
    assert (
        "direct_scalar_transport_branch_blocked"
        in payload["summary"]["claim_promotion_blockers"]
    )
    assert (
        "projected_transport_metric_not_improved"
        in payload["summary"]["claim_promotion_blockers"]
    )
    assert "direct scalar transport" in payload["summary"]["blocked_candidates"]


def test_status_plot_and_json_ready_handle_missing_transport_metric(
    tmp_path: Path,
) -> None:
    constraints = tmp_path / "constraints"
    direct = tmp_path / "direct"
    projected_base = tmp_path / "projected_base"
    projected_step = tmp_path / "projected_step"
    _candidate(constraints, passed=True, metric=None)
    _candidate(direct, passed=False, metric=None, qs=1.0)
    _candidate(projected_base, passed=True, metric=0.1)
    _candidate(projected_step, passed=True, metric=0.09)
    payload = status_mod.build_payload(
        constraints_dir=constraints,
        direct_transport_dir=direct,
        projected_baseline_dir=projected_base,
        projected_step_dir=projected_step,
        **_supporting_artifacts(tmp_path),
    )

    out = tmp_path / "status.png"
    status_mod.plot_payload(payload, out)
    cleaned = status_mod._json_ready(payload)

    assert out.exists()
    assert out.stat().st_size > 0
    assert cleaned["candidates"][0]["transport_metric_final"] is None
    json.dumps(cleaned, allow_nan=False)


def test_failed_wout_reproducibility_gate_blocks_status_admission(
    tmp_path: Path,
) -> None:
    constraints = tmp_path / "constraints"
    direct = tmp_path / "direct"
    projected_base = tmp_path / "projected_base"
    projected_step = tmp_path / "projected_step"
    _candidate(
        constraints,
        passed=True,
        metric=None,
        wout_reproducibility_passed=False,
    )
    _candidate(direct, passed=False, metric=None, qs=1.0)
    _candidate(projected_base, passed=True, metric=0.1)
    _candidate(projected_step, passed=True, metric=0.09)

    payload = status_mod.build_payload(
        constraints_dir=constraints,
        direct_transport_dir=direct,
        projected_baseline_dir=projected_base,
        projected_step_dir=projected_step,
        **_supporting_artifacts(tmp_path),
    )
    candidates = {candidate["label"]: candidate for candidate in payload["candidates"]}
    baseline = candidates["QA max_mode=5 baseline"]

    assert baseline["solved_wout_gate_passed"] is True
    assert baseline["wout_reproducibility_gate_passed"] is False
    assert baseline["passed_solved_wout_gate"] is False
    assert payload["summary"]["qa_baseline_gate_passed"] is False


def test_status_admits_explicit_authoritative_rerun_wout(tmp_path: Path) -> None:
    constraints = tmp_path / "constraints"
    direct = tmp_path / "direct"
    projected_base = tmp_path / "projected_base"
    projected_step = tmp_path / "projected_step"
    _candidate(
        constraints,
        passed=True,
        metric=None,
        wout_reproducibility_passed=False,
        rerun_wout_admission_passed=True,
    )
    _candidate(direct, passed=False, metric=None, qs=1.0)
    _candidate(projected_base, passed=True, metric=0.1)
    _candidate(projected_step, passed=True, metric=0.09)

    payload = status_mod.build_payload(
        constraints_dir=constraints,
        direct_transport_dir=direct,
        projected_baseline_dir=projected_base,
        projected_step_dir=projected_step,
        **_supporting_artifacts(tmp_path),
    )
    candidates = {candidate["label"]: candidate for candidate in payload["candidates"]}
    baseline = candidates["QA max_mode=5 baseline"]

    assert baseline["wout_reproducibility_gate_passed"] is False
    assert baseline["rerun_wout_admission_gate_passed"] is True
    assert baseline["uses_authoritative_rerun_wout"] is True
    assert baseline["passed_solved_wout_gate"] is True
    assert baseline["authoritative_wout"].endswith("wout_final_rerun.nc")
    assert payload["summary"]["qa_baseline_gate_passed"] is True


def test_nonlinear_audit_requires_expected_claim_level(tmp_path: Path) -> None:
    constraints = tmp_path / "constraints"
    direct = tmp_path / "direct"
    projected_base = tmp_path / "projected_base"
    projected_step = tmp_path / "projected_step"
    _candidate(constraints, passed=True, metric=None)
    _candidate(direct, passed=False, metric=None, qs=1.0)
    _candidate(projected_base, passed=True, metric=0.1)
    _candidate(projected_step, passed=True, metric=0.09)

    payload = status_mod.build_payload(
        constraints_dir=constraints,
        direct_transport_dir=direct,
        projected_baseline_dir=projected_base,
        projected_step_dir=projected_step,
        **_supporting_artifacts(
            tmp_path, nonlinear_claim_level="startup_window_observable"
        ),
    )
    audit = payload["long_window_nonlinear_audit"]

    assert audit["raw_passed"] is True
    assert audit["claim_level_matches_expected"] is False
    assert audit["passed"] is False
    assert payload["summary"]["long_window_nonlinear_audit_passed"] is False
    assert (
        payload["summary"]["claim_evidence_level"]
        == "nonlinear_campaign_prelaunch_ready"
    )
    assert (
        "nonlinear_audit_claim_level_mismatch"
        in payload["summary"]["claim_promotion_blockers"]
    )


def test_parse_args_accepts_campaign_admission_json(
    monkeypatch, tmp_path: Path
) -> None:
    campaign = tmp_path / "campaign.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_vmec_jax_qa_transport_optimization_status.py",
            "--campaign-admission-json",
            str(campaign),
        ],
    )

    args = status_mod._parse_args()

    assert args.campaign_admission_json == campaign



def _write_case(
    root: Path,
    *,
    objective_final: float,
    transport_metric: float | None = None,
    gate_passed: bool = True,
    completed: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    history = {
        "objective_initial": 4.0,
        "objective_final": objective_final,
        "aspect_initial": 8.8,
        "aspect_final": 5.1,
        "iota_initial": 0.1,
        "iota_final": 0.39,
        "qs_initial": 0.2,
        "qs_final": 0.07,
        "history": [
            {"objective": 4.0},
            {"objective": 2.0},
            {"objective": objective_final},
        ],
        "total_wall_time_s": 12.5,
        "success": True,
    }
    if transport_metric is not None:
        history["transport_metric_final"] = transport_metric
        history["transport_metric_kind"] = "growth"
    (root / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (root / "setup_summary.json").write_text(
        json.dumps(
            {
                "transport_kind": "growth",
                "constraints_only": False,
                "sample_set": {
                    "surfaces": [0.64],
                    "alphas": [0.0],
                    "ky_values": [0.3],
                    "n_samples": 1,
                },
                "spectrax_config": {
                    "ntheta": 8,
                    "mboz": 21,
                    "nboz": 21,
                    "n_laguerre": 2,
                    "n_hermite": 3,
                    "objective_transform": "log1p",
                    "objective_scale": 1.0,
                    "surface_chunk_size": 0,
                },
                "optimizer": {
                    "method": "scalar_trust",
                    "max_nfev": 70,
                    "inner_max_iter": 120,
                    "trial_max_iter": 120,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "solved_wout_gate.json").write_text(
        json.dumps(
            {
                "passed": gate_passed,
                "checks": {
                    "aspect": {"passed": True},
                    "mean_iota": {"passed": gate_passed},
                    "quasisymmetry": {"passed": gate_passed},
                },
            }
        ),
        encoding="utf-8",
    )
    campaign = (
        root.parent.parent
        if root.parent.name in {"runs", "runs_onepoint"}
        else root.parent
    )
    log_dir = campaign / (
        "logs_onepoint" if root.parent.name == "runs_onepoint" else "logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"[now] START {root.name} gpu=0"]
    if completed:
        lines.append(f"[now] END {root.name} rc=0")
    (log_dir / f"{root.name}.status").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_failed_wout_reproducibility_gate(root: Path) -> None:
    (root / "wout_reproducibility_gate.json").write_text(
        json.dumps(
            {
                "passed": False,
                "checks": {
                    "mean_iota_reproducibility": {
                        "passed": False,
                        "absolute_error": 1.5e-3,
                        "absolute_tolerance": 5.0e-4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_passed_rerun_wout_admission_gate(root: Path) -> None:
    (root / "rerun_wout_admission_gate.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "aspect": {"passed": True},
                    "mean_iota": {"passed": True},
                    "iota_profile": {"passed": True},
                    "quasisymmetry": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_payload_discovers_completed_cases_without_faking_q_traces(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "campaign"
    _write_case(run_root / "runs" / "qa_baseline_scipy", objective_final=1.5)
    _write_case(
        run_root / "runs" / "growth_scalar_trust",
        objective_final=0.8,
        transport_metric=0.25,
        gate_passed=False,
    )
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "logs" / "growth_scalar_trust.status").write_text(
        "[now] START growth_scalar_trust gpu=1\n",
        encoding="utf-8",
    )

    payload = full_sweep_mod.build_payload(run_root)

    assert payload["summary"]["n_cases"] == 2
    assert payload["summary"]["n_cases_with_nonlinear_q_traces"] == 0
    assert (
        payload["summary"]["nonlinear_transport_audit_status"]
        == "pending_for_this_sweep"
    )
    cases = {case["case_id"]: case for case in payload["cases"]}
    assert cases["growth_scalar_trust"]["history"]["transport_metric_final"] == 0.25
    assert "mean_iota" in cases["growth_scalar_trust"]["gate_blockers"]
    assert cases["growth_scalar_trust"]["diagnostic_gate_passed"] is False
    assert "quasisymmetry" in cases["growth_scalar_trust"]["diagnostic_gate_blockers"]
    assert cases["growth_scalar_trust"]["q_traces"] == []


def test_diagnostic_gate_accepts_only_iota_shortfall_above_floor() -> None:
    status, blockers = full_sweep_mod._diagnostic_gate_status(
        gate_passed=False,
        gate_blockers=["mean_iota"],
        iota_final=0.395,
    )

    assert status is True
    assert blockers == []

    status, blockers = full_sweep_mod._diagnostic_gate_status(
        gate_passed=False,
        gate_blockers=["mean_iota"],
        iota_final=0.37,
    )

    assert status is False
    assert blockers == ["mean_iota"]

    status, blockers = full_sweep_mod._diagnostic_gate_status(
        gate_passed=False,
        gate_blockers=["quasisymmetry"],
        iota_final=0.41,
    )

    assert status is False
    assert blockers == ["quasisymmetry"]


def test_q_trace_csv_is_used_only_when_present(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs" / "quasilinear_scalar_trust"
    _write_case(case, objective_final=0.6, transport_metric=0.1)
    (case / "audit_heat_flux_trace.csv").write_text(
        "t,heat_flux\n0.0,1.0\n1.0,2.0\n2.0,3.0\n",
        encoding="utf-8",
    )

    payload = full_sweep_mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert payload["summary"]["n_cases_with_nonlinear_q_traces"] == 1
    assert row["q_traces"][0]["late_window_mean"] == 3.0
    assert row["q_traces"][0]["late_window_tmin"] == 2.0


def test_compact_json_payload_keeps_q_trace_stats_without_dense_arrays(
    tmp_path: Path,
) -> None:
    case = tmp_path / "campaign" / "runs" / "qa_baseline_scipy"
    _write_case(case, objective_final=1.2)
    (case / "audit_heat_flux_trace.csv").write_text(
        "t,heat_flux\n0.0,1.0\n1.0,2.0\n2.0,3.0\n",
        encoding="utf-8",
    )

    payload = full_sweep_mod.build_payload(tmp_path / "campaign")
    compact = full_sweep_mod._compact_payload_for_json(payload)
    [trace] = compact["cases"][0]["q_traces"]

    assert trace["late_window_mean"] == 3.0
    assert trace["late_window_tmax"] == 2.0
    assert "t" not in trace
    assert "heat_flux" not in trace


def test_completed_wout_rows_include_reproducible_nonlinear_audit_command(
    tmp_path: Path,
) -> None:
    case = tmp_path / "campaign" / "runs" / "nonlinear_window_scalar_trust"
    _write_case(case, objective_final=0.4, transport_metric=0.08)
    (case / "wout_final.nc").write_bytes(b"not-a-real-netcdf-needed-for-command-test")

    payload = full_sweep_mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    command = row["recommended_nonlinear_audit_command"]
    assert command.startswith(
        "python3 tools/campaigns/write_optimized_equilibrium_transport_configs.py"
    )
    assert "vmec_qa_full_sweep_nonlinear_window_scalar_trust" in command
    assert "--horizons 700,1100,1500" in command
    assert "--window-tmin 1100 --window-tmax 1500" in command
    assert "--seed-variant 32 --seed-variant 33" in command
    assert "--dt-variant 0.04" in command


def test_strict_full_sweep_audit_status_detects_empty_strict_window(
    tmp_path: Path,
) -> None:
    audit_root = tmp_path / "optimized_equilibrium_replicates"
    audit_root.mkdir()
    for token in ("qa_baseline_scipy", "growth_from_strict_baseline"):
        (audit_root / f"vmec_qa_full_sweep_{token}_ensemble_gate.json").write_text(
            json.dumps(
                {
                    "case": token,
                    "passed": False,
                    "statistics": {
                        "n_finite_means": 0,
                        "ensemble_mean": None,
                    },
                }
            ),
            encoding="utf-8",
        )

    status = full_sweep_mod._strict_full_sweep_audit_status(audit_root)

    assert status["status"] == "failed_empty_strict_window"
    assert status["n_ensembles"] == 2
    assert status["n_empty_window_ensembles"] == 2


def test_iota_only_diagnostic_rows_are_audit_command_eligible(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs" / "growth_scalar_trust"
    _write_case(case, objective_final=0.5, transport_metric=0.12, gate_passed=False)
    (case / "wout_final.nc").write_bytes(b"diagnostic-wout")

    payload = full_sweep_mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert row["gate_passed"] is False
    assert row["gate_blockers"] == ["mean_iota", "quasisymmetry"]
    assert row["diagnostic_gate_passed"] is False
    assert row["recommended_nonlinear_audit_command"] is None

    gate = json.loads((case / "solved_wout_gate.json").read_text(encoding="utf-8"))
    gate["checks"]["quasisymmetry"]["passed"] = True
    (case / "solved_wout_gate.json").write_text(json.dumps(gate), encoding="utf-8")

    payload = full_sweep_mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert row["gate_passed"] is False
    assert row["gate_blockers"] == ["mean_iota"]
    assert row["diagnostic_gate_passed"] is True
    assert row["recommended_nonlinear_audit_command"] is not None


def test_failed_wout_reproducibility_gate_blocks_nonlinear_audit_promotion(
    tmp_path: Path,
) -> None:
    case = tmp_path / "campaign" / "runs" / "nonlinear_window_scalar_trust"
    _write_case(case, objective_final=0.4, transport_metric=0.08)
    _write_failed_wout_reproducibility_gate(case)
    (case / "wout_final.nc").write_bytes(b"not-a-real-netcdf-needed-for-command-test")

    payload = full_sweep_mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert row["solved_wout_gate_passed"] is True
    assert row["wout_reproducibility_gate_passed"] is False
    assert row["gate_passed"] is False
    assert row["recommended_nonlinear_audit_command"] is None
    assert "wout_reproducibility:mean_iota_reproducibility" in row["gate_blockers"]


def test_authoritative_rerun_wout_gate_selects_rerun_wout_for_audit(
    tmp_path: Path,
) -> None:
    case = tmp_path / "campaign" / "runs" / "qa_baseline_scipy"
    _write_case(case, objective_final=0.4, transport_metric=0.08)
    _write_failed_wout_reproducibility_gate(case)
    _write_passed_rerun_wout_admission_gate(case)
    (case / "wout_final.nc").write_bytes(b"optimizer-state-wout")
    (case / "wout_final_rerun.nc").write_bytes(b"authoritative-rerun-wout")

    payload = full_sweep_mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert row["gate_passed"] is True
    assert row["uses_authoritative_rerun_wout"] is True
    assert row["authoritative_wout"].endswith("wout_final_rerun.nc")
    assert row["authoritative_wout_source"] == "wout_final_rerun.nc"
    assert row["recommended_nonlinear_audit_command"] is not None
    assert "wout_final_rerun.nc" in row["recommended_nonlinear_audit_command"]
    assert row["gate_blockers"] == []
    assert row["gate_warnings"] == [
        "optimizer_state_wout_not_reproduced_authoritative_rerun_wout_used"
    ]


def test_in_progress_wout_is_not_promoted_to_completed_or_audit_ready(
    tmp_path: Path,
) -> None:
    case = tmp_path / "campaign" / "runs" / "quasilinear_scalar_trust"
    _write_case(case, objective_final=0.9, transport_metric=0.2, completed=False)
    (case / "wout_final.nc").write_bytes(b"partial-output")

    payload = full_sweep_mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert payload["summary"]["n_completed_wouts"] == 0
    assert row["run_completed"] is False
    assert row["recommended_nonlinear_audit_command"] is None


def test_runs_onepoint_root_uses_parent_status_directory(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs_onepoint" / "qa_baseline_scipy"
    _write_case(case, objective_final=1.1)
    (case / "wout_final.nc").write_bytes(b"fake-wout")

    payload = full_sweep_mod.build_payload(tmp_path / "campaign" / "runs_onepoint")
    [row] = payload["cases"]

    assert row["run_completed"] is True
    assert payload["summary"]["completed_case_ids"] == ["qa_baseline_scipy"]


def test_projected_child_without_status_is_complete_when_gate_and_wout_exist(
    tmp_path: Path,
) -> None:
    case = (
        tmp_path
        / "campaign"
        / "runs_onepoint"
        / "projected_guarded_ladder"
        / "transport_weight_0p0005"
    )
    _write_case(case, objective_final=0.9)
    (case / "wout_final.nc").write_bytes(b"fake-projected-wout")
    # Projected ladder children are tracked by the parent ladder status/log, not
    # by one status file per transport weight.
    for status in (tmp_path / "campaign" / "logs").glob(
        "transport_weight_0p0005.status"
    ):
        status.unlink()

    payload = full_sweep_mod.build_payload(tmp_path / "campaign" / "runs_onepoint")
    [row] = payload["cases"]

    assert row["case_id"] == "projected_guarded_ladder/transport_weight_0p0005"
    assert row["run_completed"] is True
    assert row["recommended_nonlinear_audit_command"] is not None


def test_plot_payload_handles_missing_wouts_and_writes_panel(tmp_path: Path) -> None:
    run_root = tmp_path / "campaign"
    _write_case(run_root / "runs" / "qa_baseline_scipy", objective_final=1.2)
    _write_case(
        run_root / "runs" / "nonlinear_window_scalar_trust",
        objective_final=0.4,
        transport_metric=0.08,
    )
    payload = full_sweep_mod.build_payload(run_root)
    out = tmp_path / "panel.png"

    full_sweep_mod.plot_payload(payload, out)
    cleaned = full_sweep_mod._json_ready(payload)

    assert out.exists()
    assert out.stat().st_size > 0
    assert cleaned["cases"][0]["history"]["transport_metric_final"] is None
    json.dumps(cleaned, allow_nan=False)


def test_iota_profile_plot_omits_vmec_axis_point() -> None:
    fig, ax = full_sweep_mod.plt.subplots()
    try:
        full_sweep_mod._plot_iota_profiles(
            ax,
            [
                {
                    "label": "baseline",
                    "iota_profile": {
                        "s": [0.0, 0.5, 1.0],
                        "iotas": [0.0, 0.39, 0.42],
                    },
                }
            ],
        )

        profile_line = next(line for line in ax.lines if line.get_label() == "baseline")

        assert list(profile_line.get_xdata()) == [0.5, 1.0]
        assert list(profile_line.get_ydata()) == [0.39, 0.42]
        assert "axis point omitted" in ax.get_title()
    finally:
        full_sweep_mod.plt.close(fig)


def test_payload_records_normalized_optimizer_comparison_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    _write_case(
        root / "runs" / "growth_scalar_trust",
        objective_final=0.8,
        transport_metric=0.25,
    )
    _write_case(
        root / "runs" / "growth_lbfgs_adjoint",
        objective_final=0.7,
        transport_metric=0.21,
    )
    setup = json.loads(
        (root / "runs" / "growth_lbfgs_adjoint" / "setup_summary.json").read_text()
    )
    setup["optimizer"]["method"] = "lbfgs_adjoint"
    (root / "runs" / "growth_lbfgs_adjoint" / "setup_summary.json").write_text(
        json.dumps(setup),
        encoding="utf-8",
    )

    payload = full_sweep_mod.build_payload(root)
    rows = {row["case_id"]: row for row in payload["cases"]}

    assert payload["summary"]["optimizer_methods"] == ["lbfgs_adjoint", "scalar_trust"]
    assert (
        rows["growth_scalar_trust"]["optimizer_comparison"]["method"] == "scalar_trust"
    )
    assert (
        rows["growth_lbfgs_adjoint"]["optimizer_comparison"]["method"]
        == "lbfgs_adjoint"
    )
    assert (
        rows["growth_scalar_trust"]["optimizer_comparison"]["comparison_fingerprint"]
        == rows["growth_lbfgs_adjoint"]["optimizer_comparison"][
            "comparison_fingerprint"
        ]
    )
    assert (
        "methods can be compared directly"
        in payload["summary"]["optimizer_comparison_policy"]
    )
    assert payload["summary"]["strict_nonlinear_audit_policy"]["window_tmin"] == 1100.0



def test_gradient_diagnostic_defaults_to_multisample_transport_contract(
    tmp_path: Path,
) -> None:
    args = gradient_mod._parse_args(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--out-json",
            str(tmp_path / "gradient.json"),
        ]
    )

    sample_set = gradient_mod._sample_set_from_args(args)
    summary = gradient_mod.transport_objective_sample_summary(sample_set)

    assert args.surfaces == gradient_mod.DEFAULT_TRANSPORT_SURFACES
    assert args.alphas == gradient_mod.DEFAULT_TRANSPORT_ALPHAS
    assert args.ky_values == gradient_mod.DEFAULT_TRANSPORT_KY_VALUES
    assert summary["passed"] is True
    assert summary["sample_count"] == 18


def test_gradient_diagnostic_fails_closed_for_underresolved_sample_set(
    tmp_path: Path, monkeypatch
) -> None:
    def unexpected_stage(_args):
        raise AssertionError(
            "under-resolved sample set should fail before VMEC-JAX stage construction"
        )

    monkeypatch.setattr(gradient_mod, "_build_stage", unexpected_stage)
    with pytest.raises(
        ValueError, match="under-resolved transport-gradient sample set"
    ):
        gradient_mod.main(
            [
                "--input",
                str(tmp_path / "input.final"),
                "--out-json",
                str(tmp_path / "gradient.json"),
                "--surfaces",
                "0.64",
                "--alphas",
                "0.0",
                "--ky-values",
                "0.3",
            ]
        )


def test_gradient_diagnostic_records_sample_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    fake_stage = SimpleNamespace(specs=[object(), object()], optimizer=object())

    def fake_stage_builder(_args):
        return fake_stage, {"setup_key": "setup_value"}

    def fake_report(_optimizer, **_kwargs):
        return {
            "kind": "vmec_jax_transport_gradient_diagnostic",
            "finite": True,
            "transport_sensitivity_detected": True,
        }

    def fake_write(report, out_json):
        Path(out_json).write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return Path(out_json)

    monkeypatch.setattr(gradient_mod, "_build_stage", fake_stage_builder)
    monkeypatch.setattr(gradient_mod, "build_boundary_transport_gradient_report", fake_report)
    monkeypatch.setattr(gradient_mod, "write_boundary_transport_gradient_report", fake_write)

    rc = gradient_mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--out-json",
            str(tmp_path / "gradient.json"),
        ]
    )

    payload = json.loads((tmp_path / "gradient.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["setup"] == {"setup_key": "setup_value"}
    assert payload["objective_sample_summary"]["passed"] is True
    assert payload["objective_sample_summary"]["sample_count"] == 18
    assert payload["nonlinear_audit_policy"]["recommended_ky_values"] == [0.1, 0.3, 0.5]


def test_gradient_diagnostic_fd_consistency_passes_for_matching_reverse_gradient(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class MatchingOptimizer:
        _specs = (SimpleNamespace(name="rc01"), SimpleNamespace(name="zs01"))

        def residual_fun(self, params):
            params_array = np.asarray(params, dtype=float)
            return np.asarray(
                [1.25 + 2.0 * params_array[0] - 3.0 * params_array[1]], dtype=float
            )

        def objective_and_gradient_fun(self, params):
            residual = float(self.residual_fun(params)[0])
            return 0.5 * residual**2, residual * np.asarray([2.0, -3.0], dtype=float)

    def fake_stage_builder(_args):
        return (
            SimpleNamespace(
                specs=MatchingOptimizer._specs,
                optimizer=MatchingOptimizer(),
            ),
            {"setup_key": "setup_value"},
        )

    monkeypatch.setattr(gradient_mod, "_build_stage", fake_stage_builder)

    rc = gradient_mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--out-json",
            str(tmp_path / "gradient.json"),
            "--fd-check-indices",
            "0,1",
            "--require-fd-consistency",
        ]
    )

    payload = json.loads((tmp_path / "gradient.json").read_text(encoding="utf-8"))
    fd = payload["finite_difference_consistency"]
    assert rc == 0
    assert fd["enabled"] is True
    assert fd["passed"] is True
    assert fd["blockers"] == []
    assert fd["rows"][0]["name"] == "rc01"
    assert fd["rows"][1]["name"] == "zs01"


def test_gradient_diagnostic_fd_consistency_reports_coefficient_conditioning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_file = tmp_path / "input.final"
    input_file.write_text(
        """
&INDATA
  RBC(1,1) = 2.0000000000000000E-05,  ZBS(1,1) = -1.0000000000000000E-02
/
""",
        encoding="utf-8",
    )

    class MatchingOptimizer:
        _specs = (
            SimpleNamespace(name="rc11", kind="rc", m=1, n=1),
            SimpleNamespace(name="zs11", kind="zs", m=1, n=1),
        )

        def residual_fun(self, params):
            params_array = np.asarray(params, dtype=float)
            return np.asarray(
                [1.0 + 2.0 * params_array[0] - 3.0 * params_array[1]], dtype=float
            )

        def objective_and_gradient_fun(self, params):
            residual = float(self.residual_fun(params)[0])
            return 0.5 * residual**2, residual * np.asarray([2.0, -3.0], dtype=float)

    def fake_stage_builder(_args):
        return (
            SimpleNamespace(
                specs=MatchingOptimizer._specs,
                optimizer=MatchingOptimizer(),
            ),
            {"setup_key": "setup_value"},
        )

    monkeypatch.setattr(gradient_mod, "_build_stage", fake_stage_builder)

    rc = gradient_mod.main(
        [
            "--input",
            str(input_file),
            "--out-json",
            str(tmp_path / "gradient.json"),
            "--fd-check-indices",
            "0,1",
            "--fd-check-step",
            "1e-4",
            "--fd-check-relative-step",
            "0.5",
            "--require-fd-consistency",
        ]
    )

    payload = json.loads((tmp_path / "gradient.json").read_text(encoding="utf-8"))
    fd = payload["finite_difference_consistency"]
    rows = fd["rows"]

    assert rc == 0
    assert fd["passed"] is True
    assert fd["relative_step"] == pytest.approx(0.5)
    assert fd["conditioning_warnings"] == ["fd_step_exceeds_input_coefficient:RBC(1,1)"]
    assert rows[0]["coefficient_label"] == "RBC(1,1)"
    assert rows[0]["input_coefficient_value"] == pytest.approx(2.0e-5)
    assert rows[0]["requested_step_to_input_coefficient_abs"] == pytest.approx(5.0)
    assert rows[0]["step"] == pytest.approx(1.0e-5)
    assert rows[0]["step_to_input_coefficient_abs"] == pytest.approx(0.5)
    assert rows[1]["coefficient_label"] == "ZBS(1,1)"
    assert rows[1]["requested_step_to_input_coefficient_abs"] == pytest.approx(1.0e-2)
    assert rows[1]["step"] == pytest.approx(1.0e-4)


def test_gradient_diagnostic_fd_consistency_fails_for_disconnected_reverse_gradient(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class DisconnectedOptimizer:
        _specs = (SimpleNamespace(name="rc01"), SimpleNamespace(name="zs01"))

        def residual_fun(self, params):
            params_array = np.asarray(params, dtype=float)
            return np.asarray(
                [1.25 + 2.0 * params_array[0] - 3.0 * params_array[1]], dtype=float
            )

        def objective_and_gradient_fun(self, params):
            residual = float(self.residual_fun(params)[0])
            return 0.5 * residual**2, np.zeros(2, dtype=float)

    def fake_stage_builder(_args):
        return (
            SimpleNamespace(
                specs=DisconnectedOptimizer._specs,
                optimizer=DisconnectedOptimizer(),
            ),
            {"setup_key": "setup_value"},
        )

    monkeypatch.setattr(gradient_mod, "_build_stage", fake_stage_builder)

    rc = gradient_mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--out-json",
            str(tmp_path / "gradient.json"),
            "--fd-check-indices",
            "0,1",
            "--require-fd-consistency",
        ]
    )

    payload = json.loads((tmp_path / "gradient.json").read_text(encoding="utf-8"))
    fd = payload["finite_difference_consistency"]
    assert rc == 3
    assert fd["enabled"] is True
    assert fd["passed"] is False
    assert "ad_fd_mismatch" in fd["blockers"]
    assert fd["max_abs_fd_cost_gradient"] > 0.0
    assert fd["rows"][0]["fd_cost_gradient"] == pytest.approx(2.5)
    assert fd["rows"][1]["fd_cost_gradient"] == pytest.approx(-3.75)


def test_gradient_diagnostic_surface_chunking_aggregates_raw_weighted_gradient(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeOptimizer:
        _specs = (SimpleNamespace(name="rc01"), SimpleNamespace(name="zs01"))

        def __init__(self, residual: float, residual_gradient: np.ndarray) -> None:
            self.residual = float(residual)
            self.residual_gradient = np.asarray(residual_gradient, dtype=float)

        def residual_fun(self, _params):
            return np.asarray([self.residual], dtype=float)

        def objective_and_gradient_fun(self, _params):
            return 0.5 * self.residual**2, self.residual * self.residual_gradient

    by_surface = {
        0.45: (1.0, np.asarray([2.0, 0.0])),
        0.64: (3.0, np.asarray([4.0, 0.0])),
        0.78: (5.0, np.asarray([6.0, 0.0])),
    }
    seen_surfaces: list[float] = []

    def fake_stage_builder(args):
        assert args.spectrax_objective_transform == "raw"
        assert args.transport_weight == 1.0
        assert len(args.surfaces) == 1
        surface = round(float(args.surfaces[0]), 2)
        seen_surfaces.append(surface)
        residual, gradient = by_surface[surface]
        return (
            SimpleNamespace(
                specs=FakeOptimizer._specs,
                optimizer=FakeOptimizer(residual, gradient),
            ),
            {"surfaces": [surface]},
        )

    monkeypatch.setattr(gradient_mod, "_build_stage", fake_stage_builder)

    rc = gradient_mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--out-json",
            str(tmp_path / "gradient.json"),
            "--surface-gradient-chunk-size",
            "1",
        ]
    )

    payload = json.loads((tmp_path / "gradient.json").read_text(encoding="utf-8"))
    expected_raw = (1.0 + 3.0 + 5.0) / 3.0
    expected_raw_gradient = (
        np.asarray([2.0, 0.0]) + np.asarray([4.0, 0.0]) + np.asarray([6.0, 0.0])
    ) / 3.0
    expected_residual = np.log1p(expected_raw)
    expected_residual_gradient = expected_raw_gradient / (1.0 + expected_raw)
    expected_cost_gradient = expected_residual * expected_residual_gradient

    assert rc == 0
    assert seen_surfaces == [0.45, 0.64, 0.78]
    assert payload["chunked_gradient"]["enabled"] is True
    assert payload["chunked_gradient"]["chunk_count"] == 3
    assert payload["chunked_gradient"]["raw_weighted_residual"] == pytest.approx(
        expected_raw
    )
    assert payload["chunked_gradient"][
        "raw_weighted_gradient_norm_l2"
    ] == pytest.approx(np.linalg.norm(expected_raw_gradient))
    assert payload["residual_norm_l2"] == pytest.approx(expected_residual)
    assert payload["gradient_norm_l2"] == pytest.approx(
        np.linalg.norm(expected_cost_gradient)
    )
    assert payload["top_gradient_components"][0]["name"] == "rc01"

