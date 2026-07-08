"""Tests for status, readiness, and closure artifact dashboards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from support.paths import REPO_ROOT, load_artifact_tool


ROOT = REPO_ROOT



# Manuscript readiness status assertions
def _build_manuscript_readiness_status_write_json(
    root: Path, relative: str, payload: dict[str, object]
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manuscript_status_closes_negative_ql_and_defers_zonal_tem(
    tmp_path: Path,
) -> None:
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_validated_calibration_inputs.json",
        {"passed": True},
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {
            "passed": False,
            "by_split": {"holdout": {"mean_abs_relative_error": 2.5}},
            "points": [
                {"case": "cyclone", "split": "train"},
                {"case": "w7x", "split": "holdout"},
                {"case": "hsx", "split": "holdout"},
            ],
        },
    )
    for name in (
        "quasilinear_saturation_rule_sweep",
        "quasilinear_shape_aware_saturation",
        "quasilinear_candidate_uncertainty",
    ):
        _build_manuscript_readiness_status_write_json(
            tmp_path,
            f"docs/_static/{name}.json",
            {
                "promotion_gate": {
                    "passed": False,
                    "null_training_mean_mean_abs_relative_error": 0.2,
                }
            },
        )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_dataset_sufficiency.json",
        {
            "promotion_gate": {
                "passed": False,
                "blockers": ["minimum_total_electrostatic_cases"],
            },
            "requirements": {
                "current_total_cases": 4,
                "min_total_electrostatic_cases": 6,
                "current_explicit_train_geometries": 1,
                "min_explicit_train_geometries": 2,
            },
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_promotion_guardrails.json",
        {"passed": True},
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/differentiable_geometry_bridge.json",
        {
            "sensitivity": {"max_abs_ad_fd_error": 1.0e-9},
            "uq": {"sensitivity_map_rank": 2},
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/vmec_boozer_parity_matrix.json",
        {
            "minimum_boozer_mode_count": 21,
            "summary": {"all_equal_arc_passed": True, "n_cases": 3},
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/stellarator_itg_optimization_comparison.json",
        {
            "results": [
                {
                    "initial_objective": 1.0,
                    "final_objective": 0.2,
                    "gradient_gate": {"passed": True},
                },
                {
                    "initial_objective": 2.0,
                    "final_objective": 0.5,
                    "gradient_gate": {"passed": True},
                },
            ]
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/stellarator_itg_optimization_uq.json",
        {
            "claim_level": "reduced_objective_uq_and_sensitivity_validation_not_full_vmec_gk_optimization",
            "all_gradient_gates_passed": True,
            "all_sensitivity_maps_full_rank": True,
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/nonlinear_sharding_profile_office_gpu.json",
        {
            "identity_gate_pass": True,
            "engineering_speedup": 0.8,
            "best_identity_preserving_candidate": {
                "spec": "kx",
                "engineering_speedup_median": 1.03,
            },
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/nonlinear_rhs_profile.json",
        {
            "fastest_full_rhs_label": "GPU spectral",
            "spectral_speedups": {
                "cpu": {
                    "full_rhs_grid_over_spectral": 1.11,
                    "nonlinear_bracket_grid_over_spectral": 1.66,
                },
                "gpu": {
                    "full_rhs_grid_over_spectral": 1.64,
                    "nonlinear_bracket_grid_over_spectral": 2.20,
                },
            },
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/nonlinear_rhs_profile_miller.json",
        {
            "rows": {
                "CPU grid": {"seconds": {"full_rhs": 0.32}},
                "GPU grid": {"seconds": {"full_rhs": 0.013}},
                "GPU spectral": {"seconds": {"full_rhs": 0.015}},
            }
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json",
        {
            "rows": {
                "W7-X CPU": {"seconds": {"full_rhs": 0.31}},
                "W7-X GPU": {"seconds": {"full_rhs": 0.027}},
                "HSX CPU": {"seconds": {"full_rhs": 0.31}},
                "HSX GPU": {"seconds": {"full_rhs": 0.027}},
            }
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/full_nonlinear_rhs_trace_summary.json",
        {"warm_seconds": 0.316},
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/full_nonlinear_rhs_trace_gpu_summary.json",
        {"warm_seconds": 0.0128, "hlo_token_counts": {"transpose": 32}},
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/solver_objective_gradient_gate.json",
        {
            "passed": True,
            "source_scope": "solver_ready_geometry_contract",
            "linear_growth_gradient_gate": True,
            "quasilinear_weight_gradient_gate": True,
        },
    )
    for name in (
        "vmec_boozer_solver_frequency_gradient_gate",
        "vmec_boozer_quasilinear_gradient_gate",
        "vmec_boozer_nonlinear_window_gradient_gate",
        "vmec_boozer_li383_nonlinear_window_gradient_gate",
    ):
        _build_manuscript_readiness_status_write_json(
            tmp_path,
            f"docs/_static/{name}.json",
            {
                "passed": True,
                "source_scope": "mode21_vmec_boozer_state",
                "nonlinear_window_gradient_gate": name.endswith(
                    "nonlinear_window_gradient_gate"
                ),
                "eigenpair_gate": {"max_rel_error": 1.0e-3},
            },
        )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/vmec_boozer_gradient_holdout_matrix.json",
        {
            "passed": True,
            "summary": {
                "n_cases": 2,
                "max_relative_error": 4.9e-3,
                "all_gates_passed": True,
            },
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/nonlinear_window_fd_audit.json",
        {
            "passed": True,
            "startup_nonlinear_plumbing_fd_path_gate": True,
            "transport_average_gate": False,
            "production_nonlinear_observable_fd_path_gate": False,
            "production_nonlinear_window_gradient_gate": False,
            "metrics": {
                "response_fraction": 0.11,
                "repeatability_relative_error": 0.0,
                "max_window_cv": 0.09,
                "max_window_trend": 0.31,
            },
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json",
        {
            "passed": True,
            "vmec_boozer_startup_nonlinear_plumbing_fd_path_gate": True,
            "transport_average_gate": False,
            "vmec_boozer_production_nonlinear_observable_fd_path_gate": False,
            "production_nonlinear_window_gradient_gate": False,
            "metrics": {
                "response_fraction": 0.04,
                "derivative_asymmetry": 2.8,
            },
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json",
        {
            "passed": True,
            "summary": {
                "common_pair_count": 21,
                "combined_response_uncertainty_rel": 0.311,
                "independent_response_mean": -0.356,
            },
        },
    )

    payload = load_artifact_tool(
        "build_manuscript_readiness_status"
    ).build_manuscript_readiness_payload(tmp_path)
    lanes = {lane["lane"]: lane for lane in payload["lanes"]}

    assert payload["summary"]["n_deferred"] == 2
    assert (
        lanes["Quasilinear diagnostics and saturation-model selection"]["status"]
        == "closed"
    )
    assert (
        lanes["Quasilinear diagnostics and saturation-model selection"]["claim_level"]
        == "validated_diagnostics_negative_absolute_flux_promotion"
    )
    assert (
        lanes["Quasilinear diagnostics and saturation-model selection"]["key_metrics"][
            "absolute_flux_promoted"
        ]
        is False
    )
    assert (
        lanes["Quasilinear diagnostics and saturation-model selection"]["key_metrics"][
            "dataset_sufficiency_promotion_passed"
        ]
        is False
    )
    assert (
        lanes["Quasilinear diagnostics and saturation-model selection"]["key_metrics"][
            "promotion_guardrails_passed"
        ]
        is True
    )
    assert (
        "docs/_static/quasilinear_promotion_guardrails.json"
        in lanes["Quasilinear diagnostics and saturation-model selection"][
            "primary_artifacts"
        ]
    )
    assert lanes["VMEC/Boozer differentiable geometry parity"]["status"] == "closed"
    assert (
        lanes["Reduced differentiable stellarator ITG optimization"]["status"]
        == "closed"
    )
    opt_lane = lanes["Reduced differentiable stellarator ITG optimization"]
    assert (
        "docs/_static/stellarator_itg_optimization_comparison.png"
        not in opt_lane["primary_artifacts"]
    )
    assert (
        "docs/_static/stellarator_itg_optimization_comparison.png"
        in opt_lane["supporting_artifacts"]
    )
    assert (
        "Do not use the reduced synthetic surface comparison" in opt_lane["next_action"]
    )
    assert lanes["Production solver-objective geometry gradients"]["status"] == "closed"
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "multi_equilibrium_gradient_holdout_matrix"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "full_vmec_boozer_reduced_nonlinear_window_gradient_gate"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "reduced_nonlinear_window_gradient_gate"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "multi_equilibrium_reduced_nonlinear_window_gradient_gate"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "production_nonlinear_window_gradient_gate"
        ]
        is False
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "startup_nonlinear_plumbing_fd_path_gate"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "startup_nonlinear_plumbing_response_fraction"
        ]
        == 0.11
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "nonlinear_transport_average_gate"
        ]
        is False
    )
    assert (
        "docs/_static/nonlinear_window_fd_audit.json"
        in lanes["Production solver-objective geometry gradients"]["primary_artifacts"]
    )
    assert (
        "docs/_static/nonlinear_window_fd_audit.png"
        in lanes["Production solver-objective geometry gradients"]["primary_artifacts"]
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "vmec_boozer_startup_nonlinear_plumbing_fd_path_gate"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "vmec_boozer_startup_nonlinear_response_fraction"
        ]
        == 0.04
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "vmec_boozer_nonlinear_transport_average_gate"
        ]
        is False
    )
    assert (
        "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json"
        in lanes["Production solver-objective geometry gradients"]["primary_artifacts"]
    )
    assert (
        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json"
        in lanes["Production solver-objective geometry gradients"]["primary_artifacts"]
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "variance_reduced_nonlinear_gradient_control_mean_passed"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "variance_reduced_nonlinear_gradient_common_pairs"
        ]
        == 21
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "variance_reduced_nonlinear_gradient_uncertainty_rel"
        ]
        == 0.311
    )
    assert lanes["W7-X zonal recurrence/damping"]["status"] == "deferred"
    assert lanes["TEM / kinetic-electron stellarator extension"]["status"] == "deferred"
    profiler = lanes["Profiler-backed nonlinear performance claims"]
    assert profiler["status"] == "closed"
    assert profiler["key_metrics"]["release_performance_gate"] is True
    assert "docs/_static/nonlinear_rhs_profile.json" in profiler["primary_artifacts"]
    assert (
        "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json"
        in profiler["primary_artifacts"]
    )
    assert profiler["key_metrics"]["best_identity_candidate"] == "kx"
    assert profiler["key_metrics"]["rhs_fastest_full_label"] == "GPU spectral"
    assert profiler["key_metrics"]["rhs_gpu_full_grid_over_spectral"] == 1.64
    assert profiler["key_metrics"]["rhs_cpu_bracket_grid_over_spectral"] == 1.66
    assert profiler["key_metrics"]["miller_gpu_grid_full_rhs"] == 0.013
    assert profiler["key_metrics"]["w7x_gpu_full_rhs"] == 0.027


def test_candidate_quasilinear_status_stays_scoped_not_runtime_flux_predictor(
    tmp_path: Path,
) -> None:
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_validated_calibration_inputs.json",
        {"passed": True},
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {
            "passed": False,
            "by_split": {"holdout": {"mean_abs_relative_error": 2.5}},
            "points": [
                {"case": "cyclone", "split": "train"},
                {"case": "w7x", "split": "holdout"},
            ],
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_saturation_rule_sweep.json",
        {"promotion_gate": {"passed": False}},
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_shape_aware_saturation.json",
        {"promotion_gate": {"passed": False}},
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_candidate_uncertainty.json",
        {
            "promotion_gate": {
                "passed": True,
                "accepted_candidates": ["spectral_envelope_ridge"],
            }
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_dataset_sufficiency.json",
        {
            "promotion_gate": {"passed": True},
            "requirements": {
                "current_total_cases": 7,
                "min_total_electrostatic_cases": 6,
            },
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_model_selection_status.json",
        {
            "promotion_gate": {"passed": True, "blockers": []},
            "metrics": {
                "candidate_mean_abs_relative_error": 0.24,
                "candidate_prediction_interval_coverage": 0.86,
            },
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_promotion_guardrails.json",
        {"passed": True},
    )

    payload = load_artifact_tool(
        "build_manuscript_readiness_status"
    ).build_manuscript_readiness_payload(tmp_path)
    lane = {row["lane"]: row for row in payload["lanes"]}[
        "Quasilinear diagnostics and saturation-model selection"
    ]

    assert lane["status"] == "closed"
    assert (
        lane["claim_level"]
        == "scoped_candidate_model_selection_not_runtime_flux_predictor"
    )
    assert lane["key_metrics"]["absolute_flux_promoted"] is False
    assert lane["key_metrics"]["promotion_guardrails_passed"] is True
    assert lane["key_metrics"]["accepted_uq_candidates"] == ["spectral_envelope_ridge"]
    assert lane["key_metrics"]["model_selection_status_passed"] is True
    assert lane["key_metrics"]["model_selection_candidate_mean_error"] == 0.24


def test_candidate_quasilinear_status_stays_open_without_promotion_guardrail(
    tmp_path: Path,
) -> None:
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_validated_calibration_inputs.json",
        {"passed": True},
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {
            "passed": False,
            "by_split": {"holdout": {"mean_abs_relative_error": 2.5}},
            "points": [
                {"case": "cyclone", "split": "train"},
                {"case": "w7x", "split": "holdout"},
            ],
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_saturation_rule_sweep.json",
        {"promotion_gate": {"passed": False}},
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_shape_aware_saturation.json",
        {"promotion_gate": {"passed": False}},
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_candidate_uncertainty.json",
        {
            "promotion_gate": {
                "passed": True,
                "accepted_candidates": ["spectral_envelope_ridge"],
            }
        },
    )
    _build_manuscript_readiness_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_dataset_sufficiency.json",
        {
            "promotion_gate": {"passed": True},
            "requirements": {"current_total_cases": 7},
        },
    )

    payload = load_artifact_tool(
        "build_manuscript_readiness_status"
    ).build_manuscript_readiness_payload(tmp_path)
    lane = {row["lane"]: row for row in payload["lanes"]}[
        "Quasilinear diagnostics and saturation-model selection"
    ]

    assert lane["status"] == "open"
    assert lane["key_metrics"]["promotion_guardrails_passed"] is False


def test_write_manuscript_readiness_artifacts_writes_all_formats(
    tmp_path: Path,
) -> None:
    payload = {
        "kind": "manuscript_readiness_status",
        "lanes": [
            {
                "lane": "Production solver-objective geometry gradients",
                "status": "open",
                "claim_level": "required",
                "primary_artifacts": [],
                "key_metrics": {},
                "next_action": "Add gates.",
            }
        ],
        "summary": {"active_fraction_closed": 0.0},
    }

    paths = load_artifact_tool(
        "build_manuscript_readiness_status"
    ).write_manuscript_readiness_artifacts(payload, out=tmp_path / "status.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "manuscript_readiness_status"


# Open research-lane status assertions
def _build_open_research_lane_status_write_json(
    root: Path, relative: str, payload: dict[str, object]
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_status_payload_keeps_open_lanes_scoped(tmp_path: Path) -> None:
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/w7x_zonal_reference_compare.json",
        {
            "validation_status": "open",
            "gate_report": {
                "gates": [
                    {"metric": "time_coverage", "passed": True},
                    {"metric": "residual_kx070", "passed": False},
                ]
            },
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/w7x_zonal_recurrence_sweep_kx070.json",
        {
            "rows": [
                {
                    "label": "low",
                    "mean_abs_error": 0.4,
                    "tail_std": 0.2,
                    "reference_tail_std": 0.05,
                },
                {
                    "label": "best",
                    "mean_abs_error": 0.1,
                    "tail_std": 0.1,
                    "reference_tail_std": 0.05,
                    "hermite_tail_at_tmax": 0.2,
                },
            ]
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/w7x_zonal_hypercollision_probe_kx070.json",
        {
            "validation_status": "open",
            "rows": [
                {
                    "label": "const nuhm0.01",
                    "mean_abs_error": 0.3,
                    "tail_std": 0.12,
                    "reference_tail_std": 0.03,
                    "hermite_tail_at_tmax": 0.22,
                    "free_energy_at_tmax_over_initial": 0.75,
                },
                {
                    "label": "const nuhm0.03",
                    "mean_abs_error": 0.28,
                    "tail_std": 0.13,
                    "reference_tail_std": 0.03,
                    "hermite_tail_at_tmax": 0.10,
                    "free_energy_at_tmax_over_initial": 0.60,
                },
            ],
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/w7x_zonal_mixedlm_resolution_kx070.json",
        {
            "rows": [
                {
                    "label": "Nl16 Nm64 mixedLM",
                    "mean_abs_error": 0.27,
                    "tail_std": 0.12,
                    "reference_tail_std": 0.03,
                    "hermite_tail_at_tmax": 1.5e-4,
                    "Nl": 16,
                    "Nm": 64,
                },
                {
                    "label": "Nl24 Nm96 mixedLM",
                    "mean_abs_error": 0.26,
                    "tail_std": 0.11,
                    "reference_tail_std": 0.03,
                    "hermite_tail_at_tmax": 1.2e-5,
                    "laguerre_tail_at_tmax": 8.2e-3,
                    "Nl": 24,
                    "Nm": 96,
                },
            ]
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/w7x_fluctuation_spectrum_panel.json",
        {
            "source_gate_passed": True,
            "time_samples": 8,
            "time_min": 1.0,
            "time_max": 4.0,
            "dominant_phi_ky": 0.2,
            "dominant_heat_flux_ky": 0.4,
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/w7x_tem_extension_status.json",
        {
            "rows": [
                {"lane": "W7-X nonlinear fluctuation spectrum", "status": "closed"},
                {"lane": "TEM / kinetic-electron linear parity", "status": "open"},
            ]
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_validated_calibration_inputs.json",
        {"passed": True},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {
            "passed": False,
            "points": [
                {"case": "cyclone", "split": "train"},
                {"case": "w7x", "split": "holdout"},
            ],
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/external_vmec_circular_t250_high_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_modified_high_grid_admission_gate.json",
        {"promotion_gate": {"passed": False}},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json",
        {
            "passed": True,
            "summary": {
                "common_pair_count": 21,
                "combined_response_uncertainty_rel": 0.311,
                "independent_response_mean": -0.356,
            },
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/external_vmec_dshape_t250_high_grid_convergence_gate.json",
        {"gate_report": {"passed": True}},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/external_vmec_itermodel_t350_high_grid_convergence_gate.json",
        {"gate_report": {"passed": True}},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/external_vmec_qh_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/external_vmec_qh_high_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/differentiable_geometry_bridge.json",
        {
            "sensitivity": {"max_abs_ad_fd_error": 1.0e-8},
            "geometry_inverse_design_report": {"final_residual_norm": 1.0e-12},
            "uq": {"sensitivity_map_rank": 2},
            "backend_info": {"vmec_jax_available": True},
            "booz_xform_jax_api_available": True,
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/nonlinear_sharding_profile_office_gpu.json",
        {
            "identity_gate_pass": True,
            "engineering_speedup": 0.8,
            "device_count": 2,
            "default_backend": "gpu",
            "best_identity_preserving_candidate": {
                "spec": "kx",
                "engineering_speedup_median": 1.03,
            },
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/nonlinear_rhs_profile.json",
        {
            "fastest_full_rhs_label": "GPU spectral",
            "spectral_speedups": {
                "cpu": {
                    "full_rhs_grid_over_spectral": 1.11,
                    "nonlinear_bracket_grid_over_spectral": 1.66,
                },
                "gpu": {
                    "full_rhs_grid_over_spectral": 1.64,
                    "nonlinear_bracket_grid_over_spectral": 2.20,
                },
            },
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/nonlinear_rhs_profile_miller.json",
        {
            "rows": {
                "CPU grid": {"seconds": {"full_rhs": 0.32}},
                "GPU grid": {"seconds": {"full_rhs": 0.013}},
                "GPU spectral": {"seconds": {"full_rhs": 0.015}},
            }
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json",
        {
            "rows": {
                "W7-X CPU": {"seconds": {"full_rhs": 0.31}},
                "W7-X GPU": {"seconds": {"full_rhs": 0.027}},
                "HSX CPU": {"seconds": {"full_rhs": 0.31}},
                "HSX GPU": {"seconds": {"full_rhs": 0.027}},
            }
        },
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/full_nonlinear_rhs_trace_summary.json",
        {"warm_seconds": 0.316},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/full_nonlinear_rhs_trace_gpu_summary.json",
        {"warm_seconds": 0.0128},
    )

    payload = load_artifact_tool(
        "build_open_research_lane_status"
    ).build_status_payload(tmp_path)
    lanes = {row["lane"]: row for row in payload["lanes"]}

    assert payload["summary"] == {
        "n_lanes": 5,
        "n_closed": 1,
        "n_partial": 2,
        "n_open": 2,
        "n_blocked": 0,
    }
    assert lanes["W7-X zonal long-window recurrence/damping"]["status"] == "open"
    assert lanes["W7-X zonal long-window recurrence/damping"]["key_metrics"][
        "failed_reference_gates"
    ] == ["residual_kx070"]
    assert (
        lanes["W7-X zonal long-window recurrence/damping"]["key_metrics"][
            "best_bounded_candidate"
        ]["label"]
        == "best"
    )
    hyper = lanes["W7-X zonal long-window recurrence/damping"]["key_metrics"][
        "best_constant_hypercollision_probe"
    ]
    assert hyper["label"] == "const nuhm0.03"
    assert hyper["validation_status"] == "open"
    mixed_lm = lanes["W7-X zonal long-window recurrence/damping"]["key_metrics"][
        "best_mixed_lm_resolution_probe"
    ]
    assert mixed_lm["label"] == "Nl24 Nm96 mixedLM"
    assert mixed_lm["tail_std_ratio"] == 0.11 / 0.03
    high_moment = lanes["W7-X zonal long-window recurrence/damping"]["key_metrics"][
        "stable_high_moment_mixed_lm_probe"
    ]
    assert high_moment["label"] == "Nl24 Nm96 mixedLM"
    assert high_moment["Nl"] == 24
    assert high_moment["Nm"] == 96
    assert high_moment["laguerre_tail"] == 8.2e-3
    assert (
        "docs/_static/w7x_zonal_mixedlm_resolution_kx070.json"
        in lanes["W7-X zonal long-window recurrence/damping"]["primary_artifacts"]
    )
    assert lanes["Scoped core quasilinear model-development diagnostic"][
        "claim_level"
    ] == ("diagnostic_calibration_dataset_not_absolute_flux")
    assert lanes["W7-X fluctuation spectrum and TEM/multi-flux extension"][
        "key_metrics"
    ]["open_extension_rows"] == ["TEM / kinetic-electron linear parity"]
    assert (
        lanes["Scoped core quasilinear model-development diagnostic"]["key_metrics"][
            "cth_like_external_vmec_converged"
        ]
        is False
    )
    assert (
        lanes["Scoped core quasilinear model-development diagnostic"]["key_metrics"][
            "cth_like_external_vmec_high_grid_admitted"
        ]
        is False
    )
    holdout_metrics = lanes["Scoped core quasilinear model-development diagnostic"][
        "key_metrics"
    ]
    assert holdout_metrics["circular_external_vmec_t250_converged"] is False
    assert holdout_metrics["qh_external_vmec_low_to_mid_grid_converged"] is False
    assert holdout_metrics["qh_external_vmec_mid_to_high_grid_converged"] is False
    assert holdout_metrics["dshape_external_vmec_t250_converged"] is True
    assert holdout_metrics["itermodel_external_vmec_t350_converged"] is True
    assert (
        holdout_metrics["variance_reduced_nonlinear_gradient_control_mean_passed"]
        is True
    )
    assert holdout_metrics["variance_reduced_nonlinear_gradient_common_pairs"] == 21
    assert (
        holdout_metrics["variance_reduced_nonlinear_gradient_uncertainty_rel"] == 0.311
    )
    profiler = lanes["Profiler-backed nonlinear hot-path optimization"]
    assert profiler["status"] == "closed"
    assert profiler["key_metrics"]["release_performance_gate"] is True
    assert "docs/_static/nonlinear_rhs_profile.json" in profiler["primary_artifacts"]
    assert (
        "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json"
        in profiler["primary_artifacts"]
    )
    assert profiler["key_metrics"]["best_identity_candidate"] == "kx"
    assert profiler["key_metrics"]["rhs_fastest_full_label"] == "GPU spectral"
    assert profiler["key_metrics"]["rhs_gpu_full_grid_over_spectral"] == 1.64
    assert profiler["key_metrics"]["rhs_gpu_bracket_grid_over_spectral"] == 2.20
    assert profiler["key_metrics"]["miller_gpu_grid_full_rhs"] == 0.013
    assert profiler["key_metrics"]["w7x_gpu_full_rhs"] == 0.027


def test_build_status_payload_accepts_cth_like_high_grid_admission(
    tmp_path: Path,
) -> None:
    """CTH-like can be admitted through the scoped high-grid gate without full-grid convergence."""

    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {"passed": False, "points": []},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _build_open_research_lane_status_write_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_modified_high_grid_admission_gate.json",
        {"promotion_gate": {"passed": True}},
    )

    payload = load_artifact_tool(
        "build_open_research_lane_status"
    ).build_status_payload(tmp_path)
    lanes = {row["lane"]: row for row in payload["lanes"]}
    metrics = lanes["Scoped core quasilinear model-development diagnostic"][
        "key_metrics"
    ]

    assert metrics["cth_like_external_vmec_full_grid_converged"] is False
    assert metrics["cth_like_external_vmec_high_grid_admitted"] is True
    assert metrics["cth_like_external_vmec_converged"] is True


def test_static_open_lane_status_keeps_deferred_w7x_zonal_and_tem_explicit() -> None:
    payload = load_artifact_tool(
        "build_open_research_lane_status"
    ).build_status_payload(ROOT)
    json.dumps(
        load_artifact_tool("build_open_research_lane_status")._json_clean(payload),
        allow_nan=False,
    )

    assert payload["kind"] == "open_research_lane_status"
    assert (
        payload["claim_scope"]
        == "post_v1_5_development_tracking_no_unvalidated_promotion"
    )
    assert set(payload["summary"]) == {
        "n_lanes",
        "n_closed",
        "n_partial",
        "n_open",
        "n_blocked",
    }
    for lane in payload["lanes"]:
        assert {
            "lane",
            "status",
            "claim_level",
            "primary_artifacts",
            "key_metrics",
            "next_action",
        }.issubset(lane)
        assert (
            lane["status"]
            in load_artifact_tool("build_open_research_lane_status").STATUS_ORDER
        )
        assert isinstance(lane["primary_artifacts"], list)
        assert isinstance(lane["key_metrics"], dict)

    lanes = {row["lane"]: row for row in payload["lanes"]}
    zonal = lanes["W7-X zonal long-window recurrence/damping"]
    zonal_metrics = zonal["key_metrics"]
    failed = set(zonal_metrics["failed_reference_gates"])

    assert zonal["status"] == "open"
    assert zonal["claim_level"] == "open_physical_closure_not_normalization"
    assert "residual_kx070" in failed
    assert any(metric.startswith("tail_envelope_std_kx") for metric in failed)
    assert (
        zonal_metrics["best_constant_hypercollision_probe"]["validation_status"]
        == "open"
    )
    assert zonal_metrics["stable_high_moment_mixed_lm_probe"]["Nl"] >= 24
    assert "docs/_static/w7x_zonal_reference_compare.json" in zonal["primary_artifacts"]
    assert (
        "docs/_static/w7x_zonal_mixedlm_resolution_kx070.json"
        in zonal["primary_artifacts"]
    )

    tem = lanes["W7-X fluctuation spectrum and TEM/multi-flux extension"]
    tem_metrics = tem["key_metrics"]
    time_min, time_max = tem_metrics["time_window"]
    open_rows = set(tem_metrics["open_extension_rows"])

    assert tem["status"] == "partial"
    assert tem["claim_level"] == "validated_simulation_spectrum_tem_extension_open"
    assert tem_metrics["time_samples"] > 0
    assert time_max > time_min >= 0.0
    assert {
        "TEM / kinetic-electron linear parity",
        "W7-X multi-flux-tube and multi-surface scan",
        "W7-X kinetic-electron/TEM nonlinear window",
    }.issubset(open_rows)
    assert "docs/_static/w7x_tem_extension_status.json" in tem["primary_artifacts"]

    holdouts = lanes["Scoped core quasilinear model-development diagnostic"]
    assert (
        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json"
        in holdouts["primary_artifacts"]
    )
    assert (
        holdouts["key_metrics"][
            "variance_reduced_nonlinear_gradient_control_mean_passed"
        ]
        is True
    )
    assert (
        holdouts["key_metrics"]["variance_reduced_nonlinear_gradient_common_pairs"]
        == 21
    )
    assert (
        holdouts["key_metrics"]["variance_reduced_nonlinear_gradient_uncertainty_rel"]
        < 0.5
    )


def test_write_status_artifacts_writes_all_formats(tmp_path: Path) -> None:
    payload = {
        "kind": "open_research_lane_status",
        "lanes": [
            {
                "lane": "Profiler-backed nonlinear hot-path optimization",
                "status": "partial",
                "claim_level": "profile_identity_artifact_no_speedup_claim",
                "primary_artifacts": ["profile.json"],
                "key_metrics": {"engineering_speedup": 0.75},
                "next_action": "Collect matched profiler traces.",
            }
        ],
        "summary": {
            "n_lanes": 1,
            "n_closed": 0,
            "n_partial": 1,
            "n_open": 0,
            "n_blocked": 0,
        },
    }

    paths = load_artifact_tool(
        "build_open_research_lane_status"
    ).write_status_artifacts(payload, out_png=tmp_path / "status.png")

    for path in paths.values():
        assert Path(path).exists()
    assert (
        json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["summary"][
            "n_partial"
        ]
        == 1
    )


# Parallelization completion status assertions
def _build_parallelization_completion_status_write_artifact(
    root: Path, name: str, payload: dict
) -> None:
    path = (
        root
        / "docs"
        / "_static"
        / load_artifact_tool("build_parallelization_completion_status").ARTIFACTS[name]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_parallelization_completion_status_scaling_payload(
    cpu_speedup: float, gpu_speedup: float
) -> dict:
    return {
        "kind": "combined",
        "claim_scope": (
            "solver-backed independent ky scan strong-scaling artifact for CPU processes and GPU workers; "
            "not a nonlinear domain-decomposition speedup claim"
        ),
        "identity_passed": True,
        "inputs": [{"backend": "cpu"}, {"backend": "gpu"}],
        "rows": [
            {
                "backend": "cpu",
                "requested_devices": 1,
                "strong_speedup_vs_1_device": 1.0,
                "identity_gate_pass": True,
            },
            {
                "backend": "cpu",
                "requested_devices": 8,
                "strong_speedup_vs_1_device": cpu_speedup,
                "identity_gate_pass": True,
            },
            {
                "backend": "gpu",
                "requested_devices": 1,
                "strong_speedup_vs_1_device": 1.0,
                "identity_gate_pass": True,
            },
            {
                "backend": "gpu",
                "requested_devices": 2,
                "strong_speedup_vs_1_device": gpu_speedup,
                "identity_gate_pass": True,
            },
        ],
    }


def _build_parallelization_completion_status_write_minimal_status_inputs(
    root: Path, *, cpu_speedup: float, gpu_speedup: float
) -> None:
    independent = _build_parallelization_completion_status_scaling_payload(
        cpu_speedup, gpu_speedup
    )
    independent["kind"] = "independent_ky_scan_scaling_combined"
    _build_parallelization_completion_status_write_artifact(
        root, "independent_ky_scan", independent
    )
    uq = _build_parallelization_completion_status_scaling_payload(
        cpu_speedup, gpu_speedup
    )
    uq["kind"] = "quasilinear_uq_ensemble_scaling_combined"
    uq["claim_scope"] = (
        "solver-backed quasilinear/UQ ensemble strong-scaling artifact for independent CPU processes "
        "and GPU workers; not a promoted absolute nonlinear heat-flux predictor"
    )
    _build_parallelization_completion_status_write_artifact(
        root, "quasilinear_uq_ensemble", uq
    )
    _build_parallelization_completion_status_write_artifact(
        root,
        "whole_state_nonlinear_sharding",
        {
            "kind": "nonlinear_sharding_strong_scaling_combined",
            "identity_passed": True,
            "claim_scope": "whole-state sharding engineering evidence, not a production speedup claim",
            "inputs": [{"backend": "cpu"}, {"backend": "gpu"}],
            "rows": [
                {
                    "backend": "cpu",
                    "requested_devices": 1,
                    "strong_speedup_vs_1_device": 1.0,
                    "identity_gate_pass": True,
                }
            ],
        },
    )
    _build_parallelization_completion_status_write_artifact(
        root,
        "fft_axis_domain",
        {
            "kind": "nonlinear_spectral_communication_identity_gate",
            "gate": {"identity_passed": True},
            "claim_scope": (
                "diagnostic nonlinear spectral communication, RHS, fixed-step integrator, "
                "pencil fused-bracket, and physical transport-window identity gate; "
                "no production distributed FFT routing or speedup claim"
            ),
            "rows": [{"identity_passed": True}],
        },
    )


def test_parallelization_completion_status_closes_production_lanes() -> None:
    status = load_artifact_tool("build_parallelization_completion_status").build_status(
        ROOT
    )
    production_gate = json.loads(
        (
            ROOT
            / "docs"
            / "_static"
            / "nonlinear_sharding_production_speedup_gate.json"
        ).read_text(encoding="utf-8")
    )

    assert status["passed"] is True
    assert status["production_completion_percent"] == 100.0
    assert status["independent_ensemble_provenance_gate"]["passed"] is True
    assert (
        status["independent_ensemble_provenance_gate"]["workload"]
        == "optimization_ensemble"
    )
    assert (
        status["independent_ensemble_provenance_gate"]["parallel_indices"]
        == status["independent_ensemble_provenance_gate"]["serial_indices"]
    )
    assert (
        status["independent_ensemble_provenance_gate"]["exception_metadata_passed"]
        is True
    )
    lanes = {lane["lane"]: lane for lane in status["lanes"]}
    assert lanes["independent_ky_scan"]["status"] == "production_closed"
    assert (
        lanes["independent_ky_scan"]["source_contract"]["claim_separation_passed"]
        is True
    )
    assert lanes["independent_ky_scan"]["source_contract"]["input_backends"] == [
        "cpu",
        "gpu",
    ]
    assert lanes["independent_ky_scan"]["best_speedups"]["cpu"] >= 5.0
    assert lanes["independent_ky_scan"]["best_speedups"]["gpu"] >= 1.5
    assert lanes["quasilinear_uq_ensemble"]["status"] == "production_closed"
    assert (
        lanes["whole_state_nonlinear_sharding"]["status"]
        == "diagnostic_closed_not_production"
    )
    assert (
        lanes["whole_state_nonlinear_sharding"]["source_contract"][
            "claim_separation_passed"
        ]
        is True
    )
    assert production_gate["production_speedup_claim_allowed"] is False
    assert production_gate["status"] == "diagnostic_only"
    assert "gpu_production_speedup_candidate_missing" in production_gate["blockers"]
    assert (
        lanes["whole_state_nonlinear_sharding"]["status"]
        == "diagnostic_closed_not_production"
    )
    assert lanes["fft_axis_domain"]["status"] == "diagnostic_identity_closed"
    assert "Whole-state nonlinear sharding" in status["claim_scope"]
    assert "exception metadata" in status["claim_scope"]


def test_parallelization_completion_status_rejects_weak_production_speedup(
    tmp_path: Path,
) -> None:
    _build_parallelization_completion_status_write_minimal_status_inputs(
        tmp_path, cpu_speedup=4.0, gpu_speedup=1.6
    )

    status = load_artifact_tool("build_parallelization_completion_status").build_status(
        tmp_path
    )

    assert status["passed"] is False
    assert status["production_completion_percent"] == 0.0
    assert {
        lane["status"]
        for lane in status["lanes"]
        if lane["claim_level"] == "production_parallelization"
    } == {"open"}


def test_parallelization_completion_status_rejects_ambiguous_claim_separation(
    tmp_path: Path,
) -> None:
    _build_parallelization_completion_status_write_minimal_status_inputs(
        tmp_path, cpu_speedup=6.0, gpu_speedup=1.7
    )
    path = (
        tmp_path
        / "docs"
        / "_static"
        / load_artifact_tool("build_parallelization_completion_status").ARTIFACTS[
            "whole_state_nonlinear_sharding"
        ]
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["claim_scope"] = "large nonlinear sharding production speedup artifact"
    path.write_text(json.dumps(payload), encoding="utf-8")

    status = load_artifact_tool("build_parallelization_completion_status").build_status(
        tmp_path
    )

    lanes = {lane["lane"]: lane for lane in status["lanes"]}
    assert status["passed"] is False
    assert lanes["whole_state_nonlinear_sharding"]["status"] == "open"
    assert lanes["whole_state_nonlinear_sharding"]["source_contract"][
        "missing_scope_phrases"
    ]


def test_parallelization_completion_status_writes_json_and_figures(
    tmp_path: Path,
) -> None:
    _build_parallelization_completion_status_write_minimal_status_inputs(
        tmp_path, cpu_speedup=6.0, gpu_speedup=1.7
    )
    status = load_artifact_tool("build_parallelization_completion_status").build_status(
        tmp_path
    )

    paths = load_artifact_tool(
        "build_parallelization_completion_status"
    ).write_artifacts(status, tmp_path / "parallelization_completion_status")

    for path in paths.values():
        assert Path(path).exists()


def test_parallelization_completion_status_script_runs_without_install(
    tmp_path: Path,
) -> None:
    _build_parallelization_completion_status_write_minimal_status_inputs(
        tmp_path, cpu_speedup=6.0, gpu_speedup=1.7
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = ""

    result = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "tools"
                / "artifacts"
                / "build_parallelization_completion_status.py"
            ),
            "--root",
            str(tmp_path),
            "--out-prefix",
            str(tmp_path / "status"),
            "--skip-figures",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "status.json").exists()


# Pre-manuscript closure status assertions
def _build_pre_manuscript_closure_status_write_json(
    root: Path, relative: str, payload: dict[str, object]
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_pre_manuscript_closure_status_write_all_pass_fixture(root: Path) -> None:
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {
            "passed": True,
            "by_split": {
                "train": {"n": 3, "mean_abs_relative_error": 0.12},
                "holdout": {"n": 10, "mean_abs_relative_error": 0.22},
            },
            "points": [],
        },
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/quasilinear_candidate_uncertainty.json",
        {"promotion_gate": {"passed": True, "accepted_candidates": ["candidate"]}},
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/quasilinear_model_selection_status.json",
        {
            "passed": True,
            "accepted_candidates": ["candidate"],
            "promotion_gate": {"passed": True, "blockers": []},
            "metrics": {
                "candidate_mean_abs_relative_error": 0.2,
                "candidate_prediction_interval_coverage": 0.9,
            },
        },
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/quasilinear_dataset_sufficiency.json",
        {
            "requirements": {
                "checks": {
                    "validated_input_gates": True,
                    "minimum_total_electrostatic_cases": True,
                    "minimum_holdout_geometries": True,
                    "minimum_explicit_train_geometries": True,
                }
            }
        },
    )
    _build_pre_manuscript_closure_status_write_json(
        root, "docs/_static/quasilinear_promotion_guardrails.json", {"passed": True}
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/quasilinear_holdout_gap_report.json",
        {"promotion_gate": {"passed": True, "blockers": []}},
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/quasilinear_error_anatomy.json",
        {
            "case_count": 12,
            "holdout_count": 10,
            "candidate_mean_abs_relative_error": 0.28,
            "promotion_gate": {
                "passed": False,
                "blockers": ["declared_stress_outliers_deferred"],
            },
            "frozen_ledger_policy": {"additional_holdout_collection_active": False},
            "core_portfolio_gate": {
                "passed": True,
                "transport_gate": 0.35,
                "interval_coverage_gate": 0.75,
                "core_case_count": 10,
                "core_holdout_count": 8,
                "core_mean_abs_relative_error": 0.28,
                "core_holdout_mean_abs_relative_error": 0.27,
                "core_max_abs_relative_error": 0.57,
                "core_prediction_interval_coverage": 1.0,
                "core_spearman": 0.74,
                "core_holdout_spearman": 0.73,
                "core_pairwise_order_accuracy": 0.75,
                "core_holdout_pairwise_order_accuracy": 0.75,
                "screening_gate_passed": False,
                "excluded_cases": [
                    {"case": "solovev_reference_repair_dt002_amp1em5_n48_t250"},
                    {
                        "case": "shaped_tokamak_pressure_external_vmec_t650_high_grid_window"
                    },
                ],
            },
        },
    )

    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/production_nonlinear_optimization_guard.json",
        {
            "passed": True,
            "summary": {
                "qualifying_matched_optimized_transport_audits": 3,
                "qualifying_optimized_equilibrium_ensembles": 3,
                "qualifying_replicated_holdout_ensembles": 4,
            },
        },
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/vmec_jax_qa_transport_optimization_status.json",
        {"summary": {"long_window_nonlinear_audit_passed": True}},
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/nonlinear_transport_matrix_portfolio.json",
        {
            "passed": True,
            "selected_family": "accepted_qa_ess",
            "selected_report": {
                "qualifies_for_broad_promotion": True,
                "summary": {
                    "total_samples": 18,
                    "completed_samples": 18,
                    "pass_fraction": 1.0,
                    "mean_relative_reduction": 0.03,
                },
            },
            "blockers": [],
        },
    )

    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/nonlinear_sharding_strong_scaling_large.json",
        {
            "identity_passed": True,
            "speedup_passed": True,
            "rows": [
                {"backend": "cpu", "speedup": 2.0},
                {"backend": "gpu", "speedup": 1.8},
            ],
        },
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/nonlinear_sharding_production_speedup_gate.json",
        {"passed": True},
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/nonlinear_domain_parallel_identity_gate.json",
        {"gate": {"identity_passed": True}},
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/nonlinear_spectral_communication_identity_gate.json",
        {"gate": {"identity_passed": True}},
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/nonlinear_spectral_domain_routing_profile.json",
        {
            "identity_passed": True,
            "strong_speedup_vs_serial": 1.8,
            "speedup_gate_passed": True,
            "production_speedup_claim_allowed": False,
            "work_model": {
                "production_speedup_feasible": True,
                "communication_to_owned_work_ratio": 0.25,
                "parallel_efficiency_ceiling": 0.8,
            },
        },
    )
    _build_pre_manuscript_closure_status_write_json(
        root, "docs/_static/parallel_decomposition_status.json", {"passed": True}
    )

    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/vmec_boozer_quasilinear_gradient_gate.json",
        {"passed": True},
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json",
        {"passed": True},
    )
    _build_pre_manuscript_closure_status_write_json(
        root, "docs/_static/vmec_boozer_gradient_holdout_matrix.json", {"passed": True}
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.json",
        {"passed": True},
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/vmec_boozer_aggregate_surface_holdout_gate.json",
        {"passed": True},
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.json",
        {"passed": True},
    )
    _build_pre_manuscript_closure_status_write_json(
        root,
        "docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json",
        {
            "passed": True,
            "promotion_gate": {"passed": True, "blockers": []},
            "holdout_artifacts": [{"qualifies_for_promotion": True}],
        },
    )


def test_current_repository_pre_manuscript_lanes_fail_closed() -> None:
    payload = load_artifact_tool(
        "build_pre_manuscript_closure_status"
    ).build_status_payload(ROOT)
    lanes = {lane["lane"]: lane for lane in payload["lanes"]}

    assert payload["kind"] == "pre_manuscript_closure_status"
    assert payload["summary"]["ready_for_manuscript_drafting"] is False
    assert len(lanes) == 4
    assert lanes["Scoped core quasilinear heat-flux diagnostic"]["passed"] is True
    assert lanes["Production nonlinear domain-decomposition speedup"]["passed"] is False
    broad_lane = lanes[
        "Broad end-to-end nonlinear turbulent-flux stellarator optimization"
    ]
    assert broad_lane["passed"] is False
    assert broad_lane["status"] == "partial"
    assert broad_lane["completion_percent"] == 94.0
    assert broad_lane["key_metrics"]["broad_matrix_portfolio_passed"] is False
    assert (
        "broad_nonlinear_transport_matrix_portfolio_missing_or_failed"
        in broad_lane["blockers"]
    )
    ql_lane = lanes["Scoped core quasilinear heat-flux diagnostic"]
    assert ql_lane["status"] == "closed"
    assert ql_lane["completion_percent"] == 100.0
    assert ql_lane["key_metrics"]["full_universal_promotion_passed"] is False
    assert ql_lane["key_metrics"]["core_case_count"] == 10
    assert ql_lane["key_metrics"]["core_holdout_count"] == 8
    assert 0.27 < ql_lane["key_metrics"]["core_mean_abs_relative_error"] < 0.29
    assert 0.27 < ql_lane["key_metrics"]["core_holdout_mean_abs_relative_error"] < 0.29
    assert ql_lane["key_metrics"]["core_prediction_interval_coverage"] == 1.0
    assert ql_lane["key_metrics"]["core_screening_gate_passed"] is False
    assert set(ql_lane["key_metrics"]["declared_stress_outliers"]) == {
        "solovev_reference_repair_dt002_amp1em5_n48_t250",
        "shaped_tokamak_pressure_external_vmec_t650_high_grid_window",
    }
    assert "universal absolute-flux runtime predictor" in ql_lane["next_action"]
    assert ql_lane["required_next_artifacts"] == []
    assert (
        "gpu_domain_speedup_below_1p5"
        in lanes["Production nonlinear domain-decomposition speedup"]["blockers"]
    )
    domain_lane = lanes["Production nonlinear domain-decomposition speedup"]
    assert domain_lane["completion_percent"] == 70.0
    assert domain_lane["key_metrics"]["routed_domain_timing_identity_passed"] is True
    assert (
        domain_lane["key_metrics"]["routed_domain_timing_speedup_gate_passed"] is False
    )
    assert domain_lane["key_metrics"]["routed_domain_work_model_present"] is True
    assert (
        domain_lane["key_metrics"]["routed_domain_work_model_speedup_feasible"] is False
    )


def test_all_pass_fixture_closes_pre_manuscript_dashboard(tmp_path: Path) -> None:
    _build_pre_manuscript_closure_status_write_all_pass_fixture(tmp_path)

    payload = load_artifact_tool(
        "build_pre_manuscript_closure_status"
    ).build_status_payload(tmp_path)

    assert payload["summary"]["ready_for_manuscript_drafting"] is True
    assert payload["summary"]["n_closed"] == 4
    assert payload["summary"]["mean_completion_percent"] == 100.0
    assert all(lane["passed"] for lane in payload["lanes"])
    vmec_lane = {lane["lane"]: lane for lane in payload["lanes"]}[
        "VMEC/Boozer holdout optimization"
    ]
    assert "closed for the current pre-manuscript gate" in vmec_lane["next_action"]
    assert all(
        "production-scope held-out surface or field-line artifact" not in item
        for item in vmec_lane["required_next_artifacts"]
    )


def test_write_pre_manuscript_artifacts(tmp_path: Path) -> None:
    _build_pre_manuscript_closure_status_write_all_pass_fixture(tmp_path)
    payload = load_artifact_tool(
        "build_pre_manuscript_closure_status"
    ).build_status_payload(tmp_path)

    paths = load_artifact_tool(
        "build_pre_manuscript_closure_status"
    ).write_status_artifacts(payload, out=tmp_path / "pre_status.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "pre_status.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "pre_manuscript_closure_status"


# Pre-manuscript closure runbook assertions
def _build_pre_manuscript_closure_runbook_write_json(
    root: Path, relative: str, payload: dict[str, object]
) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build_pre_manuscript_closure_runbook_write_screen(root: Path) -> Path:
    path = root / "screen.csv"
    path.write_text(
        "case,vmec_file,returncode,best_ky,best_gamma,best_omega,log\n"
        "existing_nc,wout_existing.nc,0,0.3,0.04,0.1,ok\n",
        encoding="utf-8",
    )
    return path


def test_pre_manuscript_runbook_fails_closed_but_lists_actions(tmp_path: Path) -> None:
    inventory = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path,
        "inventory.json",
        {
            "n_equilibria": 2,
            "rows": [
                {
                    "name": "wout_existing.nc",
                    "path": "wout_existing.nc",
                    "family": "axisymmetric",
                    "reference_scale_valid": True,
                    "candidate_score": 5.0,
                },
                {
                    "name": "wout_new_qh.nc",
                    "path": "wout_new_qh.nc",
                    "family": "quasi-helical",
                    "reference_scale_valid": True,
                    "candidate_score": 4.5,
                },
            ],
        },
    )
    screen = _build_pre_manuscript_closure_runbook_write_screen(tmp_path)
    external = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path,
        "external.json",
        {"passed": False, "launch_commands": [], "min_launch_gamma": 0.02},
    )
    optimizer = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path, "optimizer.json", {"entries": [{"status": "runnable"}]}
    )
    ladder = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path, "ladder.json", {"commands": [{"returncode": 0}]}
    )

    payload = load_artifact_tool(
        "build_pre_manuscript_closure_runbook"
    ).build_runbook_payload(
        root=tmp_path,
        inventory_path=inventory,
        screen_path=screen,
        external_runbook_path=external,
        optimizer_manifest_path=optimizer,
        ladder_status_path=ladder,
        office_root=Path("/office/repo"),
        audit_root=Path("tools_out/audits"),
    )

    holdout = payload["external_vmec_holdout_campaign"]
    assert payload["kind"] == "pre_manuscript_closure_runbook"
    assert holdout["status"] == "blocked_on_new_linear_screen"
    assert holdout["unscreened_candidates"][0]["name"] == "wout_new_qh.nc"
    assert payload["vmec_boozer_production_scope_artifacts"]["audit_commands"]
    heldout = payload["vmec_boozer_production_scope_artifacts"][
        "heldout_transport_commands"
    ]
    assert heldout[0]["transport_sample"]["alpha"] == 1.2
    assert "--torflux 0.78 --alpha 1.2" in heldout[0]["generate_configs_command"]
    assert "check_nonlinear_runtime_outputs.py" in heldout[0]["output_gate_command"]
    assert (
        "build_external_vmec_replicate_ensemble.py"
        in heldout[0]["build_ensemble_command"]
    )
    assert (
        "build_vmec_boozer_production_holdout_artifact.py"
        in heldout[0]["build_holdout_artifact_command"]
    )
    assert (
        "check_vmec_boozer_aggregate_holdout_gate.py"
        in heldout[0]["promotion_gate_command"]
    )
    assert "production_scope_vmec_boozer" in heldout[0]["claim_level"]
    assert (
        payload["vmec_boozer_production_scope_artifacts"]["office_seed_queue"][
            "launched"
        ]
        is True
    )
    assert "strict gates" in payload["claim_scope"]


def test_pre_manuscript_runbook_reports_launchable_external_holdout(
    tmp_path: Path,
) -> None:
    inventory = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path, "inventory.json", {"n_equilibria": 1, "rows": []}
    )
    screen = _build_pre_manuscript_closure_runbook_write_screen(tmp_path)
    external = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path,
        "external.json",
        {
            "passed": True,
            "launch_commands": [
                "python tools/campaigns/write_external_vmec_holdout_configs.py --case solovev"
            ],
            "min_launch_gamma": 0.02,
            "selected_new_family_candidate": {
                "case": "solovev_reference_nc",
                "best_gamma": 0.094,
            },
        },
    )
    optimizer = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path, "optimizer.json", {"entries": []}
    )
    ladder = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path, "ladder.json", {"commands": []}
    )

    payload = load_artifact_tool(
        "build_pre_manuscript_closure_runbook"
    ).build_runbook_payload(
        root=tmp_path,
        inventory_path=inventory,
        screen_path=screen,
        external_runbook_path=external,
        optimizer_manifest_path=optimizer,
        ladder_status_path=ladder,
        office_root=Path("/office/repo"),
        audit_root=Path("tools_out/audits"),
    )

    holdout = payload["external_vmec_holdout_campaign"]
    assert holdout["status"] == "launchable"
    assert holdout["selected_candidate"]["case"] == "solovev_reference_nc"
    assert holdout["launch_commands"]
    assert "Launch or harvest" in holdout["next_action"]
    assert payload["overall_next_actions"][1] == holdout["next_action"]


def test_pre_manuscript_runbook_marks_selected_external_holdout_harvested(
    tmp_path: Path,
) -> None:
    inventory = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path, "inventory.json", {"n_equilibria": 1, "rows": []}
    )
    screen = _build_pre_manuscript_closure_runbook_write_screen(tmp_path)
    external = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path,
        "external.json",
        {
            "passed": True,
            "launch_commands": [
                "python tools/campaigns/write_external_vmec_holdout_configs.py --case solovev"
            ],
            "selected_new_family_candidate": {
                "case": "solovev_reference_nc",
                "best_gamma": 0.094,
            },
        },
    )
    holdout_gap = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path,
        "holdout_gap.json",
        {
            "admitted_holdouts": [
                {
                    "case": "solovev_reference_repair_dt002_amp1em5_n48_t250",
                    "geometry": "solovev_external_vmec",
                    "gate_passed": True,
                    "status": "admitted_holdout",
                }
            ]
        },
    )
    optimizer = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path, "optimizer.json", {"entries": []}
    )
    ladder = _build_pre_manuscript_closure_runbook_write_json(
        tmp_path, "ladder.json", {"commands": []}
    )

    payload = load_artifact_tool(
        "build_pre_manuscript_closure_runbook"
    ).build_runbook_payload(
        root=tmp_path,
        inventory_path=inventory,
        screen_path=screen,
        external_runbook_path=external,
        holdout_gap_path=holdout_gap,
        optimizer_manifest_path=optimizer,
        ladder_status_path=ladder,
        office_root=Path("/office/repo"),
        audit_root=Path("tools_out/audits"),
    )

    holdout = payload["external_vmec_holdout_campaign"]
    assert holdout["status"] == "harvested_admitted"
    assert holdout["admitted_holdout"]["case"].startswith("solovev_reference_repair")
    assert holdout["launch_commands"] == []
    assert "already harvested" in holdout["next_action"]


def test_write_pre_manuscript_runbook_artifacts(tmp_path: Path) -> None:
    payload = {
        "external_vmec_holdout_campaign": {
            "status": "blocked_on_new_linear_screen",
            "next_action": "screen candidates",
        },
        "vmec_boozer_production_scope_artifacts": {
            "status": "launch_contracts_generated_on_office",
        },
        "nonlinear_optimization_audit_extension": {
            "status": "running_or_launchable",
            "acceptance": "long-window gates",
        },
        "nonlinear_domain_decomposition": {
            "status": "identity_route_extended_no_speedup_claim",
            "next_action": "add distributed routing",
        },
    }

    paths = load_artifact_tool(
        "build_pre_manuscript_closure_runbook"
    ).write_runbook_artifacts(payload, out=tmp_path / "runbook.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "runbook.json").read_text(encoding="utf-8"))
    assert (
        saved["external_vmec_holdout_campaign"]["status"]
        == "blocked_on_new_linear_screen"
    )
