"""Artifact maintainer tool contracts: stellarator artifact tools."""

from __future__ import annotations


# ---- test_status_readiness_artifacts.py ----

"""Tests for status, readiness, and closure artifact dashboards."""


import json
import os
from pathlib import Path
import subprocess
import sys

from support.paths import REPO_ROOT, load_artifact_tool


ROOT = REPO_ROOT


# Manuscript readiness status assertions
def _write_status_json(
    root: Path, relative: str, payload: dict[str, object]
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_nonlinear_performance_status_inputs(
    root: Path, *, include_device_metadata: bool = False
) -> None:
    sharding: dict[str, object] = {
        "identity_gate_pass": False,
        "engineering_speedup": 0.706,
        "best_identity_preserving_candidate": {
            "spec": None,
            "engineering_speedup_median": None,
        },
    }
    gpu_trace: dict[str, object] = {"warm_seconds": 0.0128}
    if include_device_metadata:
        sharding.update(device_count=2, default_backend="gpu")
    else:
        gpu_trace["hlo_token_counts"] = {"transpose": 32}

    payloads: dict[str, dict[str, object]] = {
        "nonlinear_sharding_profile_office_gpu_benchmark_grid": sharding,
        "nonlinear_rhs_profile": {
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
        "nonlinear_rhs_profile_miller": {
            "rows": {
                "CPU grid": {"seconds": {"full_rhs": 0.32}},
                "GPU grid": {"seconds": {"full_rhs": 0.013}},
                "GPU spectral": {"seconds": {"full_rhs": 0.015}},
            }
        },
        "nonlinear_rhs_profile_stellarator_runtime": {
            "rows": {
                "W7-X CPU": {"seconds": {"full_rhs": 0.31}},
                "W7-X GPU": {"seconds": {"full_rhs": 0.027}},
                "HSX CPU": {"seconds": {"full_rhs": 0.31}},
                "HSX GPU": {"seconds": {"full_rhs": 0.027}},
            }
        },
        "full_nonlinear_rhs_trace_summary": {"warm_seconds": 0.316},
        "full_nonlinear_rhs_trace_gpu_summary": gpu_trace,
    }
    for name, payload in payloads.items():
        _write_status_json(root, f"docs/_static/{name}.json", payload)


def test_manuscript_status_closes_negative_ql_and_defers_zonal_tem(
    tmp_path: Path,
) -> None:
    _write_status_json(
        tmp_path,
        "docs/_static/quasilinear_validated_calibration_inputs.json",
        {"passed": True},
    )
    _write_status_json(
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
        _write_status_json(
            tmp_path,
            f"docs/_static/{name}.json",
            {
                "promotion_gate": {
                    "passed": False,
                    "null_training_mean_mean_abs_relative_error": 0.2,
                }
            },
        )
    _write_status_json(
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
    _write_status_json(
        tmp_path,
        "docs/_static/quasilinear_promotion_guardrails.json",
        {"passed": True},
    )
    _write_status_json(
        tmp_path,
        "docs/_static/differentiable_geometry_bridge.json",
        {
            "sensitivity": {"max_abs_ad_fd_error": 1.0e-9},
            "uq": {"sensitivity_map_rank": 2},
        },
    )
    _write_status_json(
        tmp_path,
        "docs/_static/vmec_boozer_parity_matrix.json",
        {
            "minimum_boozer_mode_count": 21,
            "summary": {"all_equal_arc_passed": True, "n_cases": 3},
        },
    )
    _write_status_json(
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
    _write_status_json(
        tmp_path,
        "docs/_static/stellarator_itg_optimization_uq.json",
        {
            "claim_level": "reduced_objective_uq_and_sensitivity_validation_not_full_vmec_gk_optimization",
            "all_gradient_gates_passed": True,
            "all_sensitivity_maps_full_rank": True,
        },
    )
    _write_nonlinear_performance_status_inputs(tmp_path)
    _write_status_json(
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
        _write_status_json(
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
    _write_status_json(
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
    _write_status_json(
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
    _write_status_json(
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
    _write_status_json(
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
        "build_research_status"
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
    assert profiler["key_metrics"]["identity_gate_pass"] is False
    assert profiler["key_metrics"]["best_identity_candidate"] is None
    assert (
        "docs/_static/nonlinear_sharding_profile_office_gpu_benchmark_grid.json"
        in profiler["primary_artifacts"]
    )
    assert profiler["key_metrics"]["rhs_fastest_full_label"] == "GPU spectral"
    assert profiler["key_metrics"]["rhs_gpu_full_grid_over_spectral"] == 1.64
    assert profiler["key_metrics"]["rhs_cpu_bracket_grid_over_spectral"] == 1.66
    assert profiler["key_metrics"]["miller_gpu_grid_full_rhs"] == 0.013
    assert profiler["key_metrics"]["w7x_gpu_full_rhs"] == 0.027


def _write_candidate_quasilinear_status_inputs(
    root: Path,
    *,
    include_promotion_guardrail: bool,
) -> None:
    _write_status_json(
        root,
        "docs/_static/quasilinear_validated_calibration_inputs.json",
        {"passed": True},
    )
    _write_status_json(
        root,
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
    for name in (
        "quasilinear_saturation_rule_sweep",
        "quasilinear_shape_aware_saturation",
    ):
        _write_status_json(
            root,
            f"docs/_static/{name}.json",
            {"promotion_gate": {"passed": False}},
        )
    _write_status_json(
        root,
        "docs/_static/quasilinear_candidate_uncertainty.json",
        {
            "promotion_gate": {
                "passed": True,
                "accepted_candidates": ["spectral_envelope_ridge"],
            }
        },
    )
    _write_status_json(
        root,
        "docs/_static/quasilinear_dataset_sufficiency.json",
        {
            "promotion_gate": {"passed": True},
            "requirements": {
                "current_total_cases": 7,
                "min_total_electrostatic_cases": 6,
            },
        },
    )
    if not include_promotion_guardrail:
        return
    _write_status_json(
        root,
        "docs/_static/quasilinear_model_selection_status.json",
        {
            "promotion_gate": {"passed": True, "blockers": []},
            "metrics": {
                "candidate_mean_abs_relative_error": 0.24,
                "candidate_prediction_interval_coverage": 0.86,
            },
        },
    )
    _write_status_json(
        root,
        "docs/_static/quasilinear_promotion_guardrails.json",
        {"passed": True},
    )


def test_candidate_quasilinear_status_stays_scoped_not_runtime_flux_predictor(
    tmp_path: Path,
) -> None:
    _write_candidate_quasilinear_status_inputs(
        tmp_path,
        include_promotion_guardrail=True,
    )

    payload = load_artifact_tool(
        "build_research_status"
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
    _write_candidate_quasilinear_status_inputs(
        tmp_path,
        include_promotion_guardrail=False,
    )

    payload = load_artifact_tool(
        "build_research_status"
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
        "build_research_status"
    ).write_manuscript_readiness_artifacts(payload, out=tmp_path / "status.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "manuscript_readiness_status"


# Open research-lane status assertions
def test_build_status_payload_keeps_open_lanes_scoped(tmp_path: Path) -> None:
    _write_status_json(
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
    _write_status_json(
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
    _write_status_json(
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
    _write_status_json(
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
    _write_status_json(
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
    _write_status_json(
        tmp_path,
        "docs/_static/w7x_tem_extension_status.json",
        {
            "rows": [
                {"lane": "W7-X nonlinear fluctuation spectrum", "status": "closed"},
                {"lane": "TEM / kinetic-electron linear parity", "status": "open"},
            ]
        },
    )
    _write_status_json(
        tmp_path,
        "docs/_static/quasilinear_validated_calibration_inputs.json",
        {"passed": True},
    )
    _write_status_json(
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
    _write_status_json(
        tmp_path,
        "docs/_static/external_vmec_circular_t250_high_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _write_status_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _write_status_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_modified_high_grid_admission_gate.json",
        {"promotion_gate": {"passed": False}},
    )
    _write_status_json(
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
    _write_status_json(
        tmp_path,
        "docs/_static/external_vmec_dshape_t250_high_grid_convergence_gate.json",
        {"gate_report": {"passed": True}},
    )
    _write_status_json(
        tmp_path,
        "docs/_static/external_vmec_itermodel_t350_high_grid_convergence_gate.json",
        {"gate_report": {"passed": True}},
    )
    _write_status_json(
        tmp_path,
        "docs/_static/external_vmec_qh_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _write_status_json(
        tmp_path,
        "docs/_static/external_vmec_qh_high_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _write_status_json(
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
    _write_nonlinear_performance_status_inputs(
        tmp_path, include_device_metadata=True
    )

    payload = load_artifact_tool(
        "build_research_status"
    ).build_open_research_lane_payload(tmp_path)
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
    assert profiler["key_metrics"]["identity_gate_pass"] is False
    assert profiler["key_metrics"]["best_identity_candidate"] is None
    assert (
        "docs/_static/nonlinear_sharding_profile_office_gpu_benchmark_grid.json"
        in profiler["primary_artifacts"]
    )
    assert profiler["key_metrics"]["rhs_fastest_full_label"] == "GPU spectral"
    assert profiler["key_metrics"]["rhs_gpu_full_grid_over_spectral"] == 1.64
    assert profiler["key_metrics"]["rhs_gpu_bracket_grid_over_spectral"] == 2.20
    assert profiler["key_metrics"]["miller_gpu_grid_full_rhs"] == 0.013
    assert profiler["key_metrics"]["w7x_gpu_full_rhs"] == 0.027


def test_build_status_payload_accepts_cth_like_high_grid_admission(
    tmp_path: Path,
) -> None:
    """CTH-like can be admitted through the scoped high-grid gate without full-grid convergence."""

    _write_status_json(
        tmp_path,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {"passed": False, "points": []},
    )
    _write_status_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _write_status_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_modified_high_grid_admission_gate.json",
        {"promotion_gate": {"passed": True}},
    )

    payload = load_artifact_tool(
        "build_research_status"
    ).build_open_research_lane_payload(tmp_path)
    lanes = {row["lane"]: row for row in payload["lanes"]}
    metrics = lanes["Scoped core quasilinear model-development diagnostic"][
        "key_metrics"
    ]

    assert metrics["cth_like_external_vmec_full_grid_converged"] is False
    assert metrics["cth_like_external_vmec_high_grid_admitted"] is True
    assert metrics["cth_like_external_vmec_converged"] is True


def test_static_open_lane_status_keeps_deferred_w7x_zonal_and_tem_explicit() -> None:
    payload = load_artifact_tool(
        "build_research_status"
    ).build_open_research_lane_payload(ROOT)
    json.dumps(
        load_artifact_tool("build_research_status").json_clean(payload),
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
            in load_artifact_tool("build_research_status").OPEN_STATUS_ORDER
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
        "build_research_status"
    ).write_open_research_lane_artifacts(payload, out_png=tmp_path / "status.png")

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
def _write_pre_manuscript_closure_json(
    root: Path, relative: str, payload: dict[str, object]
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_pre_manuscript_closure_all_pass_fixture(root: Path) -> None:
    _write_pre_manuscript_closure_json(
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
    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/quasilinear_candidate_uncertainty.json",
        {"promotion_gate": {"passed": True, "accepted_candidates": ["candidate"]}},
    )
    _write_pre_manuscript_closure_json(
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
    _write_pre_manuscript_closure_json(
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
    _write_pre_manuscript_closure_json(
        root, "docs/_static/quasilinear_promotion_guardrails.json", {"passed": True}
    )
    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/quasilinear_holdout_gap_report.json",
        {"promotion_gate": {"passed": True, "blockers": []}},
    )
    _write_pre_manuscript_closure_json(
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

    _write_pre_manuscript_closure_json(
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
    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/vmec_jax_qa_transport_optimization_status.json",
        {"summary": {"long_window_nonlinear_audit_passed": True}},
    )
    _write_pre_manuscript_closure_json(
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

    _write_pre_manuscript_closure_json(
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
    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/nonlinear_sharding_production_speedup_gate.json",
        {"passed": True},
    )
    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/nonlinear_domain_parallel_identity_gate.json",
        {"gate": {"identity_passed": True}},
    )
    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/nonlinear_spectral_communication_identity_gate.json",
        {"gate": {"identity_passed": True}},
    )
    _write_pre_manuscript_closure_json(
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
    _write_pre_manuscript_closure_json(
        root, "docs/_static/parallel_decomposition_status.json", {"passed": True}
    )

    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/vmec_boozer_quasilinear_gradient_gate.json",
        {"passed": True},
    )
    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json",
        {"passed": True},
    )
    _write_pre_manuscript_closure_json(
        root, "docs/_static/vmec_boozer_gradient_holdout_matrix.json", {"passed": True}
    )
    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.json",
        {"passed": True},
    )
    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/vmec_boozer_aggregate_surface_holdout_gate.json",
        {"passed": True},
    )
    _write_pre_manuscript_closure_json(
        root,
        "docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.json",
        {"passed": True},
    )
    _write_pre_manuscript_closure_json(
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
        "build_research_status"
    ).build_pre_manuscript_closure_payload(ROOT)
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
    _write_pre_manuscript_closure_all_pass_fixture(tmp_path)

    payload = load_artifact_tool(
        "build_research_status"
    ).build_pre_manuscript_closure_payload(tmp_path)

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
    _write_pre_manuscript_closure_all_pass_fixture(tmp_path)
    payload = load_artifact_tool(
        "build_research_status"
    ).build_pre_manuscript_closure_payload(tmp_path)

    paths = load_artifact_tool(
        "build_research_status"
    ).write_pre_manuscript_closure_artifacts(payload, out=tmp_path / "pre_status.png")

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
        "build_research_status"
    ).build_pre_manuscript_runbook_payload(
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
    assert "check_nonlinear_transport_gates.py" in heldout[0]["output_gate_command"]
    assert (
        "build_external_vmec_replicate_ensemble.py"
        in heldout[0]["build_ensemble_command"]
    )
    assert (
        "build_vmec_boozer_aggregate_holdout_gate.py production"
        in heldout[0]["build_holdout_artifact_command"]
    )
    assert (
        "check_vmec_boozer_gates.py aggregate-holdout"
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
        "build_research_status"
    ).build_pre_manuscript_runbook_payload(
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
        "build_research_status"
    ).build_pre_manuscript_runbook_payload(
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
        "build_research_status"
    ).write_pre_manuscript_runbook_artifacts(payload, out=tmp_path / "runbook.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "runbook.json").read_text(encoding="utf-8"))
    assert (
        saved["external_vmec_holdout_campaign"]["status"]
        == "blocked_on_new_linear_screen"
    )


# ---- test_vmec_boozer_artifact_reports.py ----

"""Tests for VMEC/Boozer artifact reports and promotion gates."""


from types import SimpleNamespace

import numpy as np
import pytest


from tools.artifacts import build_vmec_boozer_aggregate_holdout_gate as holdout_gate
from tools.artifacts import build_vmec_boozer_aggregate_objective_gate as objective_gate

comparison_gate = objective_gate
multi_point_gate = objective_gate
second_gate = objective_gate


def _write_gradient_gate(
    path: Path,
    *,
    case: str,
    kind: str,
    passed: bool = True,
    source_scope: str = "mode21_vmec_boozer_state",
    mboz: int = 21,
    nboz: int = 21,
    extra_objectives: list[dict[str, object]] | None = None,
) -> None:
    objective_gates = [
        {
            "objective": "gamma",
            "passed": passed,
            "rel_error": 1.0e-3,
            "abs_error": 2.0e-4,
        },
        {
            "objective": "omega",
            "passed": True,
            "rel_error": 2.0e-3,
            "abs_error": 3.0e-4,
        },
    ]
    if extra_objectives is not None:
        objective_gates.extend(extra_objectives)
    path.write_text(
        json.dumps(
            {
                "kind": kind,
                "case_name": case,
                "passed": passed,
                "source_scope": source_scope,
                "mboz": mboz,
                "nboz": nboz,
                "surface_stencil_width": None,
                "objective_gates": objective_gates,
            }
        ),
        encoding="utf-8",
    )


def test_gradient_holdout_matrix_summarizes_passed_gates(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_vmec_boozer_gradient_holdout_matrix")
    qh_freq = tmp_path / "qh_freq.json"
    qh_ql = tmp_path / "qh_ql.json"
    li_freq = tmp_path / "li_freq.json"
    li_ql = tmp_path / "li_ql.json"
    for path, case, kind in [
        (qh_freq, "nfp4_QH_warm_start", "frequency"),
        (qh_ql, "nfp4_QH_warm_start", "quasilinear"),
        (li_freq, "li383_low_res", "frequency"),
        (li_ql, "li383_low_res", "quasilinear"),
    ]:
        _write_gradient_gate(path, case=case, kind=kind)

    payload = mod.build_gradient_holdout_matrix(
        (
            ("QH", "frequency", qh_freq),
            ("QH", "quasilinear", qh_ql),
            ("Li383", "frequency", li_freq),
            ("Li383", "quasilinear", li_ql),
        )
    )

    assert payload["kind"] == "vmec_boozer_gradient_holdout_matrix"
    assert payload["passed"] is True
    assert payload["summary"]["n_cases"] == 2
    assert payload["summary"]["max_relative_error"] == 2.0e-3
    assert payload["rows"][0]["objectives"]["gamma"] is True


def test_gradient_holdout_matrix_writes_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_vmec_boozer_gradient_holdout_matrix")
    gate = tmp_path / "gate.json"
    _write_gradient_gate(gate, case="case", kind="frequency")
    payload = mod.build_gradient_holdout_matrix((("case", "frequency", gate),))

    paths = mod.write_gradient_holdout_matrix(payload, out=tmp_path / "matrix.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "matrix.json").read_text(encoding="utf-8"))
    assert saved["claim_level"] == (
        "multi_equilibrium_reduced_linear_quasilinear_and_nonlinear_window_estimator_gradient_gate_"
        "not_production_nonlinear_optimization"
    )


def test_gradient_holdout_matrix_requires_mode21_scope_and_mode_floor(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_gradient_holdout_matrix")
    good_gate = tmp_path / "good.json"
    wrong_scope_gate = tmp_path / "wrong_scope.json"
    underresolved_gate = tmp_path / "underresolved.json"
    _write_gradient_gate(good_gate, case="nfp4_QH_warm_start", kind="frequency")
    _write_gradient_gate(
        wrong_scope_gate,
        case="nfp4_QH_warm_start",
        kind="quasilinear",
        source_scope="solver_ready_geometry_contract",
    )
    _write_gradient_gate(
        underresolved_gate,
        case="li383_low_res",
        kind="frequency",
        mboz=20,
        nboz=21,
    )

    wrong_scope_payload = mod.build_gradient_holdout_matrix(
        (
            ("QH", "frequency", good_gate),
            ("QH", "quasilinear", wrong_scope_gate),
        )
    )
    underresolved_payload = mod.build_gradient_holdout_matrix(
        (
            ("QH", "frequency", good_gate),
            ("Li383", "frequency", underresolved_gate),
        )
    )

    assert wrong_scope_payload["passed"] is False
    assert wrong_scope_payload["summary"]["all_gates_passed"] is True
    assert wrong_scope_payload["summary"]["all_mode21_source_scope"] is False
    assert underresolved_payload["passed"] is False
    assert underresolved_payload["summary"]["all_mboz_nboz_at_least_21"] is False


def test_gradient_holdout_matrix_tracks_estimator_as_reduced_claim(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_gradient_holdout_matrix")
    estimator_gate = tmp_path / "estimator.json"
    _write_gradient_gate(
        estimator_gate,
        case="nfp4_QH_warm_start",
        kind="nonlinear-window estimator",
        extra_objectives=[
            {
                "objective": "nonlinear_window_heat_flux_mean",
                "passed": True,
                "rel_error": 2.5e-2,
                "abs_error": 1.0e-4,
            },
            {
                "objective": "nonlinear_window_heat_flux_cv",
                "passed": True,
                "rel_error": 1.5e-2,
                "abs_error": 1.0e-4,
            },
            {
                "objective": "nonlinear_window_heat_flux_trend",
                "passed": True,
                "rel_error": 2.0e-2,
                "abs_error": 1.0e-4,
            },
        ],
    )

    payload = mod.build_gradient_holdout_matrix(
        (("QH", "nonlinear-window estimator", estimator_gate),)
    )
    row = payload["rows"][0]

    assert payload["passed"] is True
    assert payload["claim_level"].endswith("not_production_nonlinear_optimization")
    assert payload["summary"]["gate_types"] == ["nonlinear-window estimator"]
    assert row["objectives"]["nonlinear_window_heat_flux_mean"] is True
    assert row["objectives"]["nonlinear_window_heat_flux_cv"] is True
    assert row["objectives"]["nonlinear_window_heat_flux_trend"] is True
    assert "optimized-equilibrium nonlinear transport" in payload["notes"]


def _fake_fd_geom(scale: float = 1.0):
    theta = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    ones = np.ones_like(theta)
    return SimpleNamespace(
        theta=theta,
        gradpar=lambda: 0.7 * scale,
        bmag_profile=scale * (1.0 + 0.05 * np.cos(theta)),
        bgrad_profile=scale * 0.05 * np.sin(theta),
        gds2_profile=scale * ones,
        gds21_profile=0.02 * scale * np.sin(theta),
        gds22_profile=scale * (1.0 + 0.03 * np.cos(theta)),
        cv_profile=0.1 * scale * np.cos(theta),
        gb_profile=0.1 * scale * np.cos(theta),
        cv0_profile=np.zeros_like(theta),
        gb0_profile=np.zeros_like(theta),
        jacobian_profile=ones / scale,
        grho_profile=scale * ones,
        q=1.4 * scale,
        s_hat=0.8 * scale,
        epsilon=0.18,
        R0=2.7,
        alpha=0.0,
        kxfac=1.0,
        theta_scale=1.0,
        nfp=4,
    )


def _synthetic_fd_run(label: str, perturbation: float, scale: float = 1.0) -> dict:
    mod = load_artifact_tool("build_vmec_boozer_nonlinear_window_fd_audit")
    time = np.linspace(0.0, 1.0, 8)
    heat = scale * (1.0 + 0.08 * time)
    return {
        "label": label,
        "perturbation": perturbation,
        "geometry_file_name": f"{label}.nc",
        "geometry_response": mod.geometry_response_metrics(
            _fake_fd_geom(), _fake_fd_geom(1.0 + abs(perturbation))
        ),
        "time": time.tolist(),
        "heat_flux": heat.tolist(),
        "window": mod.late_window_metrics(time, heat, tail_fraction=0.5),
    }


def test_geometry_response_metrics_reports_profile_and_scalar_changes() -> None:
    mod = load_artifact_tool("build_vmec_boozer_nonlinear_window_fd_audit")

    metrics = mod.geometry_response_metrics(_fake_fd_geom(), _fake_fd_geom(1.01))

    assert metrics["max_relative_change"] > 0.0
    assert metrics["per_profile"]["bmag"] > 0.0
    assert metrics["per_scalar"]["q"] > 0.0


def test_build_vmec_boozer_audit_payload_passes_conditioned_response() -> None:
    mod = load_artifact_tool("build_vmec_boozer_nonlinear_window_fd_audit")
    runs = [
        _synthetic_fd_run("minus", -1.0e-5, 0.80),
        _synthetic_fd_run("base", 0.0, 1.00),
        _synthetic_fd_run("plus", 1.0e-5, 1.25),
        _synthetic_fd_run("base_repeat", 0.0, 1.00),
    ]

    payload = mod.build_vmec_boozer_audit_payload(
        runs,
        case_name="nfp4_QH_warm_start",
        parameter_name="Rcos_r1_m1",
        perturbation_step=1.0e-5,
        tail_fraction=0.5,
        repeatability_rtol=1.0e-12,
        max_window_cv=0.1,
        max_window_trend=0.1,
        min_response_fraction=0.03,
        min_geometry_relative_change=1.0e-7,
    )

    assert payload["passed"] is True
    assert payload["vmec_boozer_startup_nonlinear_plumbing_fd_path_gate"] is True
    assert payload["transport_average_gate"] is False
    assert payload["vmec_boozer_production_nonlinear_observable_fd_path_gate"] is False
    assert payload["production_nonlinear_window_gradient_gate"] is False
    assert payload["gates"]["geometry_perturbation_resolved"] is True
    assert payload["metrics"]["central_fd_dq_dparameter"] > 0.0


def test_build_vmec_boozer_audit_payload_blocks_unresolved_geometry() -> None:
    mod = load_artifact_tool("build_vmec_boozer_nonlinear_window_fd_audit")
    runs = [
        _synthetic_fd_run("minus", -1.0e-10, 0.80),
        _synthetic_fd_run("base", 0.0, 1.00),
        _synthetic_fd_run("plus", 1.0e-10, 1.25),
        _synthetic_fd_run("base_repeat", 0.0, 1.00),
    ]

    payload = mod.build_vmec_boozer_audit_payload(
        runs,
        case_name="nfp4_QH_warm_start",
        parameter_name="Rcos_r1_m1",
        perturbation_step=1.0e-10,
        tail_fraction=0.5,
        repeatability_rtol=1.0e-12,
        max_window_cv=0.1,
        max_window_trend=0.1,
        min_response_fraction=0.03,
        min_geometry_relative_change=1.0e-7,
    )

    assert payload["passed"] is False
    assert payload["gates"]["geometry_perturbation_resolved"] is False


def test_fd_audit_main_writes_artifacts_without_running_solver(
    monkeypatch, tmp_path: Path
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_nonlinear_window_fd_audit")

    monkeypatch.setattr(
        mod,
        "_mode21_vmec_boozer_linear_context",
        lambda **_kwargs: {
            "parameter_names": ("Rcos_r1_m1",),
            "geometry_for": lambda _x: _fake_fd_geom(),
        },
    )

    def fake_run_vmec_boozer_window(*, label: str, perturbation: float, **_kwargs):
        scale = {"minus": 0.80, "base": 1.00, "plus": 1.25, "base_repeat": 1.00}[label]
        return _synthetic_fd_run(label, perturbation, scale)

    monkeypatch.setattr(mod, "run_vmec_boozer_window", fake_run_vmec_boozer_window)
    out = tmp_path / "audit.png"

    assert mod.main(["--out", str(out), "--tail-fraction", "0.5"]) == 0

    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".csv").exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["passed"] is True
    assert (
        meta["claim_level"]
        == "vmec_boozer_geometry_perturbed_startup_plumbing_fd_audit_not_transport_average"
    )
    assert meta["transport_average_gate"] is False


def _fake_nonlinear_gradient_payload() -> dict[str, object]:
    objective_names = [
        "gamma",
        "nonlinear_window_heat_flux_mean",
        "nonlinear_window_heat_flux_cv",
        "nonlinear_window_heat_flux_trend",
    ]
    return {
        "kind": "mode21_vmec_boozer_nonlinear_window_gradient_gate",
        "passed": True,
        "source_scope": "mode21_vmec_boozer_state",
        "parameter_names": ["Rcos_mid_surface_m1"],
        "objective_names": objective_names,
        "objective_gates": [
            {
                "objective": objective,
                "parameter": "Rcos_mid_surface_m1",
                "implicit": float(index + 1),
                "finite_difference": float(index + 1),
                "abs_error": 0.0,
                "rel_error": 0.0,
                "passed": True,
            }
            for index, objective in enumerate(objective_names)
        ],
        "eigenpair_gate": {
            "atol": 1.0e-6,
            "jacobian_implicit": [[1.0], [2.0], [3.0], [4.0]],
            "jacobian_fd": [[1.0], [2.0], [3.0], [4.0]],
        },
    }


def test_nonlinear_window_builder_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    mod = load_artifact_tool("build_solver_objective_gradient_gate")
    monkeypatch.setattr(
        mod,
        "mode21_vmec_boozer_nonlinear_window_gradient_report",
        lambda **_kwargs: _fake_nonlinear_gradient_payload(),
    )

    out = tmp_path / "vmec_boozer_nonlinear_window_gradient_gate.png"
    assert (
        mod.main(
            [
                "vmec-boozer",
                "nonlinear-window",
                "--out",
                str(out),
                "--surface-stencil-width",
                "3",
            ]
        )
        == 0
    )

    for suffix in (".png", ".pdf", ".json", ".csv"):
        assert out.with_suffix(suffix).exists()
    saved = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["kind"] == "mode21_vmec_boozer_nonlinear_window_gradient_gate"


def test_nonlinear_window_builder_json_only(capsys, monkeypatch) -> None:
    mod = load_artifact_tool("build_solver_objective_gradient_gate")
    monkeypatch.setattr(
        mod,
        "mode21_vmec_boozer_nonlinear_window_gradient_report",
        lambda **_kwargs: _fake_nonlinear_gradient_payload(),
    )

    assert (
        mod.main(
            [
                "vmec-boozer",
                "nonlinear-window",
                "--json-only",
                "--nonlinear-steps",
                "12",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["objective_names"][-1] == "nonlinear_window_heat_flux_trend"


def test_vmec_boozer_gradient_builder_frequency_and_quasilinear_json_only(
    capsys, monkeypatch
) -> None:
    mod = load_artifact_tool("build_solver_objective_gradient_gate")
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_frequency(**kwargs: object) -> dict[str, object]:
        calls.append(("frequency", dict(kwargs)))
        return {
            "kind": "mode21_vmec_boozer_linear_frequency_gradient_gate",
            "passed": True,
            "objective_names": ["gamma", "omega"],
            "objective_gates": [],
        }

    def fake_quasilinear(**kwargs: object) -> dict[str, object]:
        calls.append(("quasilinear", dict(kwargs)))
        return {
            "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
            "passed": True,
            "objective_names": ["gamma", "omega", "ql_heat_flux"],
            "objective_gates": [],
        }

    monkeypatch.setattr(
        mod, "mode21_vmec_boozer_linear_frequency_gradient_report", fake_frequency
    )
    monkeypatch.setattr(
        mod, "mode21_vmec_boozer_quasilinear_gradient_report", fake_quasilinear
    )
    assert (
        mod.main(
            ["vmec-boozer", "frequency", "--json-only", "--mboz", "21", "--nboz", "21"]
        )
        == 0
    )
    frequency_payload = json.loads(capsys.readouterr().out)
    assert (
        frequency_payload["kind"] == "mode21_vmec_boozer_linear_frequency_gradient_gate"
    )

    assert (
        mod.main(["vmec-boozer", "quasilinear", "--json-only", "--fd-step", "2e-6"])
        == 0
    )
    quasilinear_payload = json.loads(capsys.readouterr().out)
    assert quasilinear_payload["kind"] == "mode21_vmec_boozer_quasilinear_gradient_gate"

    assert [name for name, _ in calls] == ["frequency", "quasilinear"]
    assert calls[0][1]["mboz"] == 21
    assert calls[1][1]["fd_step"] == 2.0e-6


def _fake_parity_report(**kwargs: object) -> dict[str, object]:
    assert int(kwargs["mboz"]) >= 21
    assert int(kwargs["nboz"]) >= 21
    return {
        "available": True,
        "case_name": kwargs["case_name"],
        "status": "diagnostic_open",
        "mboz": kwargs["mboz"],
        "nboz": kwargs["nboz"],
        "equal_arc_core_worst_normalized_max_abs": 4.0e-3,
        "equal_arc_core_worst_scalar_rel": 2.0e-3,
        "equal_arc_derivative_worst_normalized_max_abs": 2.0e-2,
        "equal_arc_metric_worst_normalized_max_abs": 3.0e-2,
        "equal_arc_drift_worst_normalized_max_abs": 7.0e-2,
        "equal_arc_core_tolerance": 1.0e-2,
        "equal_arc_derivative_tolerance": 3.0e-2,
        "equal_arc_metric_tolerance": 8.0e-2,
        "equal_arc_drift_tolerance": 8.0e-2,
        "equal_arc_core_passed": True,
        "equal_arc_derivative_passed": True,
        "equal_arc_metric_passed": True,
        "equal_arc_drift_passed": True,
        "production_parity_passed": False,
        "worst_core_normalized_max_abs": 2.0e-1,
        "worst_scalar_rel": 1.0e-3,
        "source_model": "vmec_jax:state->tensor-flux-tube vs imported-vmec-eik",
        "surface_index": 4,
        "torflux": 0.5,
        "alpha": 0.0,
    }


def _fake_parity_artifact_resolver(
    case_name: str,
) -> tuple[str | None, str | None, str | None]:
    if case_name in {"nfp1_QI", "nfp2_QI", "nfp4_QI_finite_beta"}:
        return f"/tmp/input.{case_name}", None, None
    return f"/tmp/input.{case_name}", "/dev/null", None


def test_build_parity_matrix_uses_mode21_floor_and_summarizes_rows() -> None:
    mod = load_artifact_tool("build_vmec_boozer_parity_matrix")
    cases = (
        mod.ParityCase("nfp4_QH_warm_start", "QH", "stellarator", 16),
        mod.ParityCase("nfp3_QI_fixed_resolution_final", "QI", "stellarator", 8),
    )

    payload = mod.build_parity_matrix(
        cases=cases,
        reporter=_fake_parity_report,
        artifact_resolver=_fake_parity_artifact_resolver,
    )

    assert payload["kind"] == "vmec_boozer_parity_matrix"
    assert payload["minimum_boozer_mode_count"] == 21
    assert payload["summary"]["n_cases"] == 2
    assert payload["summary"]["n_equal_arc_passed"] == 2
    assert payload["summary"]["all_equal_arc_passed"] is True
    assert all(row["mode_floor_passed"] for row in payload["rows"])
    assert payload["rows"][0][
        "equal_arc_drift_worst_normalized_max_abs"
    ] == pytest.approx(7.0e-2)
    assert payload["claim_level"].endswith("not_full_transport_gradient_claim")
    assert payload["rows"][0]["production_parity_passed"] is False
    assert payload["rows"][0]["sample_set_id"] == (
        "nfp4_QH_warm_start:ntheta=16:mboz=21:nboz=21"
    )
    provenance = payload["sample_set_provenance"]
    assert provenance["bounded_run"] is True
    assert provenance["external_vmec_solves_launched"] is False
    assert provenance["summary"]["n_total_sample_sets"] == 7
    assert provenance["summary"]["n_unique_sample_sets"] == 6
    assert provenance["summary"]["all_modes_at_or_above_floor"] is True
    assert (
        provenance["matrix_cases"][0]["sample_set_id"]
        == payload["rows"][0]["sample_set_id"]
    )
    assert provenance["matrix_cases"][0]["field_line_alpha"] == pytest.approx(0.0)
    assert provenance["matrix_cases"][0]["surface_index"] == 4
    assert provenance["matrix_cases"][0]["torflux"] == pytest.approx(0.5)
    qi_summary = payload["qi_seed_robustness"]["summary"]
    assert qi_summary["n_variants"] == 5
    assert qi_summary["n_passed"] == 2
    assert qi_summary["n_rejected"] == 3
    assert qi_summary["seed_robust_gate_passed"] is True
    assert qi_summary["full_declared_seed_campaign_passed"] is False
    assert qi_summary["evaluated_reference_gate_passed"] is True
    assert qi_summary["robustness_status"] == "artifact_limited_passed"
    assert qi_summary["artifact_reason_counts"]["missing_bundled_wout_reference"] == 3


def test_build_parity_matrix_rejects_underresolved_boozer_modes() -> None:
    mod = load_artifact_tool("build_vmec_boozer_parity_matrix")
    cases = (
        mod.ParityCase(
            "nfp3_QI_fixed_resolution_final", "QI", "stellarator", 8, mboz=20, nboz=21
        ),
    )

    with pytest.raises(ValueError, match="mboz and nboz"):
        mod.build_parity_matrix(
            cases=cases,
            reporter=_fake_parity_report,
            artifact_resolver=_fake_parity_artifact_resolver,
        )


def test_qi_seed_robustness_records_failed_mode21_variant() -> None:
    mod = load_artifact_tool("build_vmec_boozer_parity_matrix")

    def failing_report(**kwargs: object) -> dict[str, object]:
        report = _fake_parity_report(**kwargs)
        report["equal_arc_drift_worst_normalized_max_abs"] = 9.0e-2
        report["equal_arc_drift_passed"] = False
        return report

    payload = mod.build_parity_matrix(
        cases=(),
        qi_variants=(
            mod.ParityCase(
                "nfp3_QI_fixed_resolution_final",
                "QI",
                "quasi-isodynamic accepted reference",
                8,
            ),
        ),
        reporter=failing_report,
        artifact_resolver=_fake_parity_artifact_resolver,
    )

    qi = payload["qi_seed_robustness"]
    assert qi["summary"]["n_failed"] == 1
    assert qi["summary"]["seed_robust_gate_passed"] is False
    row = qi["rows"][0]
    assert row["qi_gate_status"] == "fragile_open"
    assert row["artifact_reason"] == "mode21_qi_tolerance_exceeded"
    assert row["equal_arc_drift_worst_normalized_max_abs"] == pytest.approx(9.0e-2)


def test_qi_seed_robustness_rejects_input_only_variants() -> None:
    mod = load_artifact_tool("build_vmec_boozer_parity_matrix")
    payload = mod.build_parity_matrix(
        cases=(),
        qi_variants=(
            mod.ParityCase(
                "nfp1_QI", "QI input nfp1", "quasi-isodynamic input variant", 8
            ),
        ),
        reporter=_fake_parity_report,
        artifact_resolver=_fake_parity_artifact_resolver,
    )

    qi = payload["qi_seed_robustness"]
    assert qi["summary"]["n_rejected"] == 1
    assert qi["summary"]["seed_robust_gate_passed"] is False
    row = qi["rows"][0]
    assert row["sample_set_id"] == "nfp1_QI:ntheta=8:mboz=21:nboz=21"
    assert row["qi_gate_status"] == "artifact_rejected"
    assert row["artifact_reason"] == "missing_bundled_wout_reference"
    assert "does not launch VMEC solves" in row["rejection_reason"]


def test_write_parity_matrix_artifacts_writes_companions(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_vmec_boozer_parity_matrix")
    payload = mod.build_parity_matrix(
        cases=(
            mod.ParityCase("shaped_tokamak_pressure", "tokamak", "axisymmetric", 8),
        ),
        reporter=_fake_parity_report,
        artifact_resolver=_fake_parity_artifact_resolver,
    )

    paths = mod.write_parity_matrix_artifacts(payload, out=tmp_path / "parity.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "parity.json").read_text(encoding="utf-8"))
    assert saved["summary"]["n_equal_arc_passed"] == 1
    assert saved["sample_set_provenance"]["summary"]["n_total_sample_sets"] == 6
    csv_text = (tmp_path / "parity.csv").read_text(encoding="utf-8")
    assert "sample_set_id" in csv_text
    assert "shaped_tokamak_pressure:ntheta=8:mboz=21:nboz=21" in csv_text
    assert "missing_bundled_wout_reference" in csv_text


def _write_holdout_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _holdout_manifest(path: Path) -> Path:
    return _write_holdout_json(
        path,
        {
            "kind": "external_vmec_holdout_config_manifest",
            "case": "qh_holdout",
            "transport_sample": {
                "vmec_file": "/tmp/wout_qh.nc",
                "torflux": 0.78,
                "alpha": 1.2,
                "ky": 0.2,
                "npol": 1.0,
            },
        },
    )


def _holdout_ensemble(path: Path, *, passed: bool) -> Path:
    return _write_holdout_json(
        path,
        {
            "kind": "nonlinear_window_ensemble_report",
            "case": "qh_holdout_replicated_window",
            "passed": passed,
            "gate_report": {"passed": passed},
            "window": {"tmin": 350.0, "tmax": 700.0},
            "statistics": {"ensemble_mean": 1.25, "combined_sem": 0.05},
        },
    )


def test_vmec_boozer_production_holdout_artifact_promotes_only_passed_ensemble(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_aggregate_holdout_gate")
    artifact = mod.build_vmec_boozer_production_holdout_artifact(
        transport_manifest=_holdout_manifest(tmp_path / "run_manifest.json"),
        ensemble_json=_holdout_ensemble(tmp_path / "ensemble.json", passed=True),
    )

    assert artifact["passed"] is True
    assert artifact["transport_average_gate"] is True
    assert artifact["promotion_gate"]["blockers"] == []
    assert (
        artifact["claim_level"]
        == "production_scope_vmec_boozer_heldout_nonlinear_transport_average"
    )
    sample = artifact["samples"][0]
    assert sample["surface"] == 0.78
    assert sample["torflux"] == 0.78
    assert sample["alpha"] == 1.2
    assert sample["selected_ky_index"] == "ky=0.2"
    assert artifact["holdout_samples"] == artifact["samples"]


def test_vmec_boozer_production_holdout_artifact_fails_closed_for_failed_ensemble(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_aggregate_holdout_gate")
    artifact = mod.build_vmec_boozer_production_holdout_artifact(
        transport_manifest=_holdout_manifest(tmp_path / "run_manifest.json"),
        ensemble_json=_holdout_ensemble(tmp_path / "ensemble.json", passed=False),
        case="explicit_case",
    )

    assert artifact["case"] == "explicit_case"
    assert artifact["passed"] is False
    assert artifact["transport_average_gate"] is False
    assert artifact["promotion_gate"]["blockers"] == [
        "replicated_nonlinear_window_ensemble_failed"
    ]


def test_vmec_boozer_production_holdout_artifact_main_writes_output(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_vmec_boozer_aggregate_holdout_gate")
    out = tmp_path / "holdout.json"
    result = mod.main(
        [
            "production",
            "--transport-manifest",
            str(_holdout_manifest(tmp_path / "run_manifest.json")),
            "--ensemble-json",
            str(_holdout_ensemble(tmp_path / "ensemble.json", passed=True)),
            "--out",
            str(out),
        ]
    )

    assert result == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert (
        saved["kind"]
        == "vmec_boozer_production_scope_heldout_nonlinear_transport_artifact"
    )
    assert saved["passed"] is True


# VMEC/Boozer aggregate artifact assertions
alpha_gate = holdout_gate
surface_gate = holdout_gate


def _assert_artifacts(paths: dict[str, str], *, json_key: str, csv_token: str) -> dict:
    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert json_key in payload
    assert csv_token in Path(paths["csv"]).read_text(encoding="utf-8")
    return payload


def _alpha_holdout_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_line_search_holdout_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "initial_delta": 0.0,
        "final_delta": -1.0e-8,
        "training_passed": True,
        "heldout_passed": True,
        "training_initial_objective": 0.9,
        "training_final_objective": 0.88,
        "training_relative_reduction": 0.0222222222,
        "heldout_initial_objective": 0.91,
        "heldout_final_objective": 0.909,
        "heldout_relative_reduction": 0.0010989011,
        "training_samples": [
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1},
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2},
        ],
        "heldout_samples": [
            {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1},
            {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2},
        ],
    }


def _surface_holdout_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_line_search_holdout_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "initial_delta": 0.0,
        "final_delta": 1.0e-8,
        "training_passed": True,
        "heldout_passed": True,
        "training_initial_objective": 0.8045688627,
        "training_final_objective": 0.8035154075,
        "training_relative_reduction": 0.0013093413,
        "heldout_initial_objective": 0.7205574540,
        "heldout_final_objective": 0.7202268146,
        "heldout_relative_reduction": 0.0004588662,
        "training_samples": [
            {"surface_index": 18, "alpha": 0.0, "selected_ky_index": 1},
            {"surface_index": 18, "alpha": 0.0, "selected_ky_index": 2},
        ],
        "heldout_samples": [
            {"surface_index": 19, "alpha": 0.0, "selected_ky_index": 1},
            {"surface_index": 19, "alpha": 0.0, "selected_ky_index": 2},
        ],
    }


def _objective_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "n_samples": 2,
        "base_value": 0.9,
        "minus_value": 0.85,
        "plus_value": 0.95,
        "central_derivative": 5.0,
        "response_abs": 0.1,
        "curvature_ratio": 0.02,
        "samples": [
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 1,
                "weight": 0.5,
            },
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 2,
                "weight": 0.5,
            },
        ],
        "minus_sample_values": [0.7, 1.0],
        "base_sample_values": [0.8, 1.0],
        "plus_sample_values": [0.9, 1.0],
    }


def _multi_point_payload() -> dict[str, object]:
    samples = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2, "weight": 0.25},
    ]
    payload = _objective_payload()
    payload.update(
        n_samples=len(samples),
        samples=samples,
        minus_sample_values=[0.70, 0.95, 0.75, 1.00],
        base_sample_values=[0.80, 1.00, 0.85, 0.95],
        plus_sample_values=[0.90, 1.05, 0.95, 0.90],
    )
    return payload


def _line_search_payload(
    objective: str = "quasilinear_flux",
    derivative: float = 5.0,
    initial: float = 0.90,
    final: float = 0.88,
) -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_line_search_report",
        "passed": True,
        "objective": objective,
        "reduction": "mean",
        "n_samples": 2,
        "accepted_steps": 1,
        "max_steps": 1,
        "initial_delta": 0.0,
        "final_delta": -1.0e-8 if derivative > 0.0 else 1.0e-8,
        "initial_objective": initial,
        "final_objective": final,
        "relative_reduction": (initial - final) / initial,
        "stop_reason": "max_steps",
        "samples": [
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 1,
                "weight": 0.5,
            },
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 2,
                "weight": 0.5,
            },
        ],
        "history": [
            {
                "step": 0,
                "delta": 0.0,
                "objective": initial,
                "central_derivative": derivative,
                "finite_difference_passed": True,
                "curvature_ratio": 0.02,
                "accepted": True,
                "candidate_delta": -1.0e-8 if derivative > 0.0 else 1.0e-8,
                "candidate_objective": final,
            }
        ],
    }


def _comparison_payload() -> dict[str, object]:
    reports = {
        "growth": _line_search_payload("growth", 2.0, 0.30, 0.29),
        "quasilinear_flux": _line_search_payload("quasilinear_flux", 5.0, 0.90, 0.88),
    }
    return {
        "kind": "vmec_boozer_aggregate_line_search_comparison",
        "passed": True,
        "case_name": "nfp4_QH_warm_start",
        "objectives": ["growth", "quasilinear_flux"],
        "reduction": "mean",
        "n_samples": 2,
        "same_sample_set": True,
        "all_line_searches_passed": True,
        "same_initial_update_direction": True,
        "final_delta_spread": 0.0,
        "relative_reduction_spread": 0.011,
        "rows": [
            {
                "objective": "growth",
                "passed": True,
                "n_samples": 2,
                "initial_objective": 0.30,
                "final_objective": 0.29,
                "absolute_reduction": 0.01,
                "relative_reduction": 0.0333333333,
                "initial_central_derivative": 2.0,
                "initial_update_direction": "negative_delta",
                "accepted_steps": 1,
                "max_steps": 1,
                "initial_delta": 0.0,
                "final_delta": -1.0e-8,
                "stop_reason": "max_steps",
            },
            {
                "objective": "quasilinear_flux",
                "passed": True,
                "n_samples": 2,
                "initial_objective": 0.90,
                "final_objective": 0.88,
                "absolute_reduction": 0.02,
                "relative_reduction": 0.0222222222,
                "initial_central_derivative": 5.0,
                "initial_update_direction": "negative_delta",
                "accepted_steps": 1,
                "max_steps": 1,
                "initial_delta": 0.0,
                "final_delta": -1.0e-8,
                "stop_reason": "max_steps",
            },
        ],
        "reports": reports,
    }


def _second_fd_payload() -> dict[str, object]:
    payload = _objective_payload()
    payload.update(
        case_name="li383_low_res",
        minus_value=9.80,
        base_value=9.79,
        plus_value=9.78,
        central_derivative=-1.0e5,
        response_abs=0.02,
        curvature_ratio=0.01,
    )
    return payload


def _second_line_payload() -> dict[str, object]:
    payload = _line_search_payload("quasilinear_flux", -1.0e5, 9.79, 9.78)
    payload.update(case_name="li383_low_res")
    return payload


def test_alpha_holdout_payload_uses_default_split(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _alpha_holdout_payload()

    monkeypatch.setattr(
        alpha_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = alpha_gate.build_vmec_boozer_aggregate_alpha_holdout_payload(
        ntheta=4, mboz=21, nboz=21
    )

    assert payload["artifact_kind"] == "vmec_boozer_aggregate_alpha_holdout_gate"
    assert payload["passed"] is True
    assert payload["holdout_split"]["training_alphas"] == [0.0]
    assert payload["holdout_split"]["holdout_alphas"] == [0.5]
    assert calls["training_selected_ky_indices"] == (1, 2)
    assert calls["holdout_selected_ky_indices"] == (1, 2)
    assert calls["mboz"] == 21
    assert calls["nboz"] == 21


def test_alpha_holdout_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _alpha_holdout_payload()

    monkeypatch.setattr(
        alpha_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    result = alpha_gate.main(
        [
            "alpha",
            "--out",
            str(tmp_path / "alpha_holdout.png"),
            "--holdout-alphas",
            "0.25",
            "--training-selected-ky-indices",
            "1",
            "2",
            "--json-only",
        ]
    )

    assert result == 0
    assert calls["holdout_alphas"] == (0.25,)
    assert calls["training_selected_ky_indices"] == (1, 2)


def test_surface_holdout_payload_contracts(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = surface_gate.build_vmec_boozer_aggregate_surface_holdout_payload(
        ntheta=4, mboz=21, nboz=21
    )

    assert payload["artifact_kind"] == "vmec_boozer_aggregate_surface_holdout_gate"
    assert payload["passed"] is True
    assert payload["blocked"] is False
    assert payload["blockers"] == []
    assert payload["holdout_split"]["training_surface_indices"] == [18]
    assert payload["holdout_split"]["holdout_surface_indices"] == [19]
    assert calls["training_surface_indices"] == (18,)
    assert calls["holdout_surface_indices"] == (19,)


def test_surface_holdout_payload_fails_closed_on_execution_error(monkeypatch) -> None:
    def fake_report(**_kwargs):  # noqa: ANN003, ANN202
        raise ValueError("surface_index is outside the VMEC metric radial grid")

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = surface_gate.build_vmec_boozer_aggregate_surface_holdout_payload(
        training_surface_indices=(18,), holdout_surface_indices=(99,)
    )

    assert payload["passed"] is False
    assert payload["blocked"] is True
    assert payload["blockers"] == ["surface_split_execution_failed"]
    assert payload["exception_type"] == "ValueError"
    assert "surface_index" in payload["exception_message"]


def test_surface_holdout_payload_rejects_non_holdout_surface_split(monkeypatch) -> None:
    calls: list[object] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = surface_gate.build_vmec_boozer_aggregate_surface_holdout_payload(
        training_surface_indices=(18,), holdout_surface_indices=(18,)
    )

    assert payload["passed"] is False
    assert payload["blocked"] is True
    assert payload["blockers"] == ["surface_split_not_held_out"]
    assert calls == []


def test_surface_holdout_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    result = surface_gate.main(
        [
            "surface",
            "--out",
            str(tmp_path / "surface_holdout.png"),
            "--training-surface-indices",
            "18",
            "--holdout-surface-indices",
            "19",
            "--json-only",
        ]
    )

    assert result == 0
    assert calls["training_surface_indices"] == (18,)
    assert calls["holdout_surface_indices"] == (19,)


@pytest.mark.parametrize(
    ("report_name", "payload_factory", "command_prefix", "command_args", "expected"),
    [
        pytest.param(
            "vmec_boozer_aggregate_scalar_objective_line_search_report",
            _line_search_payload,
            ["line-search"],
            ["--selected-ky-indices", "1", "2", "--max-steps", "2"],
            {"selected_ky_indices": (1, 2), "max_steps": 2, "mboz": 21},
            id="line-search",
        ),
        pytest.param(
            "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
            _objective_payload,
            [],
            [
                "--selected-ky-indices",
                "1",
                "2",
                "--surface-indices",
                "3",
                "5",
            ],
            {"surface_indices": (3, 5), "selected_ky_indices": (1, 2), "mboz": 21},
            id="finite-difference",
        ),
    ],
)
def test_objective_gate_main_uses_report(
    monkeypatch,
    tmp_path: Path,
    report_name: str,
    payload_factory,
    command_prefix: list[str],
    command_args: list[str],
    expected: dict[str, object],
) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return payload_factory()

    monkeypatch.setattr(objective_gate, report_name, fake_report)

    result = objective_gate.main(
        [
            *command_prefix,
            "--out",
            str(tmp_path / f"{report_name}.png"),
            *command_args,
            "--json-only",
        ]
    )

    assert result == 0
    for key, value in expected.items():
        assert calls[key] == value


def test_objective_gate_maps_physical_ky_and_torflux(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _objective_payload()

    monkeypatch.setattr(
        objective_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    result = objective_gate.main(
        [
            "--out",
            str(tmp_path / "gate.png"),
            "--ky-values",
            "0.1",
            "0.3",
            "0.5",
            "--json-only",
        ]
    )
    assert result == 0
    assert calls["selected_ky_indices"] == (1, 3, 5)
    assert calls["ny"] == 12
    assert calls["ly"] == pytest.approx(2.0 * np.pi / 0.1)

    result = objective_gate.main(
        [
            "--out",
            str(tmp_path / "gate2.png"),
            "--torflux-values",
            "0.5",
            "0.7",
            "--json-only",
        ]
    )
    assert result == 0
    assert calls["surface_indices"] == (None,)
    assert calls["torflux_values"] == (0.5, 0.7)

    with pytest.raises(ValueError, match="torflux-values or --surface-indices"):
        objective_gate.main(
            ["--surface-indices", "3", "--torflux-values", "0.5", "--json-only"]
        )


def test_physical_ky_annotation_adds_resolved_metadata() -> None:
    payload = _objective_payload()
    objective_gate._annotate_physical_ky_samples(
        payload,
        requested_ky_values=[0.1, 0.2],
        solver_grid_options={
            "selected_ky_indices": (1, 2),
            "resolved_ky_values": (0.100000001, 0.200000003),
        },
    )

    assert payload["samples"][0]["ky"] == pytest.approx(0.1)
    assert payload["samples"][0]["selected_ky"] == pytest.approx(0.100000001)
    assert payload["samples"][0]["ky_abs_error"] == pytest.approx(1.0e-9)
    assert payload["samples"][1]["ky"] == pytest.approx(0.2)


def test_comparison_report_uses_same_sample_set(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(dict(kwargs))
        objective = str(kwargs["objective"])
        if objective == "growth":
            return _line_search_payload(objective, 2.0, 0.30, 0.29)
        return _line_search_payload(objective, 5.0, 0.90, 0.88)

    monkeypatch.setattr(
        comparison_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_report,
    )

    payload = comparison_gate.build_vmec_boozer_aggregate_line_search_comparison_report(
        selected_ky_indices=(1, 2),
        surface_indices=(None,),
        alphas=(0.0,),
        max_steps=1,
        ntheta=4,
    )

    assert payload["passed"] is True
    assert payload["same_sample_set"] is True
    assert payload["same_initial_update_direction"] is True
    assert [call["objective"] for call in calls] == ["growth", "quasilinear_flux"]
    assert all(call["selected_ky_indices"] == (1, 2) for call in calls)
    assert all(call["ntheta"] == 4 for call in calls)


def test_comparison_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(dict(kwargs))
        return _line_search_payload(str(kwargs["objective"]), 1.0, 1.0, 0.9)

    monkeypatch.setattr(
        comparison_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_report,
    )

    result = comparison_gate.main(
        [
            "line-search-comparison",
            "--out",
            str(tmp_path / "comparison.png"),
            "--selected-ky-indices",
            "1",
            "2",
            "--max-steps",
            "2",
            "--json-only",
        ]
    )

    assert result == 0
    assert [call["objective"] for call in calls] == ["growth", "quasilinear_flux"]
    assert all(call["selected_ky_indices"] == (1, 2) for call in calls)
    assert all(call["max_steps"] == 2 for call in calls)
    assert all(call["mboz"] == 21 for call in calls)


def test_multi_point_payload_contracts(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _multi_point_payload()

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    payload = multi_point_gate.build_vmec_boozer_multi_point_objective_payload(
        max_wall_seconds=0.0
    )

    assert payload["artifact_kind"] == "vmec_boozer_multi_point_objective_gate"
    assert payload["passed"] is True
    assert payload["claim_scope"].startswith("bounded finite-difference")
    assert payload["multi_point_coverage"]["multi_alpha_or_surface"] is True
    assert payload["multi_point_coverage"]["n_samples_requested"] == 4
    assert payload["bounded_runtime"]["max_samples"] == 8
    assert calls["surface_indices"] == (None,)
    assert calls["alphas"] == (0.0, 0.5)
    assert calls["selected_ky_indices"] == (1, 2)
    assert calls["mboz"] == 21
    assert calls["nboz"] == 21


def test_multi_point_payload_accepts_two_surfaces(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _multi_point_payload()

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    payload = multi_point_gate.build_vmec_boozer_multi_point_objective_payload(
        surface_indices=(3, 5),
        alphas=(0.0,),
        selected_ky_indices=(1,),
        max_wall_seconds=0.0,
    )

    assert payload["multi_point_coverage"]["surface_indices"] == [3, 5]
    assert payload["multi_point_coverage"]["n_samples_requested"] == 2
    assert calls["surface_indices"] == (3, 5)
    assert calls["alphas"] == (0.0,)
    assert calls["selected_ky_indices"] == (1,)


def test_multi_point_main_uses_report_and_bounds(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _multi_point_payload()

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    result = multi_point_gate.main(
        [
            "multi-point",
            "--out",
            str(tmp_path / "gate.png"),
            "--surface-indices",
            "3",
            "5",
            "--alphas",
            "0.0",
            "--selected-ky-indices",
            "1",
            "--json-only",
            "--max-wall-seconds",
            "0",
        ]
    )

    assert result == 0
    assert calls["surface_indices"] == (3, 5)
    assert calls["alphas"] == (0.0,)
    assert calls["selected_ky_indices"] == (1,)


@pytest.mark.parametrize(
    "argv",
    [
        ["--alphas", "0.0", "--selected-ky-indices", "1", "--json-only"],
        [
            "--surface-indices",
            "3",
            "5",
            "--alphas",
            "0.0",
            "0.5",
            "--selected-ky-indices",
            "1",
            "2",
            "3",
            "--max-samples",
            "8",
            "--json-only",
        ],
    ],
    ids=["single_point", "over_sample_bound"],
)
def test_multi_point_main_rejects_invalid_coverage(
    monkeypatch, argv: list[str]
) -> None:
    def fake_report(**_kwargs):  # noqa: ANN003, ANN202
        raise AssertionError("report should not run for invalid coverage")

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    with pytest.raises(SystemExit):
        multi_point_gate.main(["multi-point", *argv])


def test_second_equilibrium_payload_passes_with_mode21_defaults(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_fd(**kwargs):  # noqa: ANN003, ANN202
        calls["fd"] = kwargs
        return _second_fd_payload()

    def fake_line(**kwargs):  # noqa: ANN003, ANN202
        calls["line"] = kwargs
        return _second_line_payload()

    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd,
    )
    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_line,
    )

    payload = second_gate.build_vmec_boozer_second_equilibrium_aggregate_payload(
        max_wall_seconds=0.0
    )

    assert payload["passed"] is True
    assert payload["feasible"] is True
    assert payload["case_name"] == "li383_low_res"
    assert payload["mode_bound"] == {
        "mboz": 21,
        "nboz": 21,
        "minimum_required": 21,
        "passed": True,
    }
    assert payload["coverage"]["selected_ky_indices"] == [1, 2]
    assert payload["finite_difference_summary"]["central_derivative"] == -1.0e5
    assert payload["line_search_summary"]["accepted_steps"] == 1
    assert calls["fd"]["mboz"] == 21
    assert calls["fd"]["nboz"] == 21
    assert calls["line"]["case_name"] == "li383_low_res"


def test_second_equilibrium_payload_fails_closed_on_backend_error(monkeypatch) -> None:
    def fake_fd(**_kwargs):  # noqa: ANN003, ANN202
        raise RuntimeError("vmec_jax example fixture missing")

    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd,
    )

    payload = second_gate.build_vmec_boozer_second_equilibrium_aggregate_payload(
        max_wall_seconds=0.0
    )

    assert payload["passed"] is False
    assert payload["feasible"] is False
    assert payload["blocker_type"] == "RuntimeError"
    assert payload["blocker_message"] == "vmec_jax example fixture missing"
    assert payload["mode_bound"]["passed"] is True


def test_second_equilibrium_json_only_uses_reports(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_fd(**kwargs):  # noqa: ANN003, ANN202
        calls["fd"] = kwargs
        return _second_fd_payload()

    def fake_line(**kwargs):  # noqa: ANN003, ANN202
        calls["line"] = kwargs
        return _second_line_payload()

    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd,
    )
    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_line,
    )

    result = second_gate.main(
        [
            "second-equilibrium",
            "--case-name",
            "nfp3_QI_fixed_resolution_final",
            "--selected-ky-indices",
            "1",
            "2",
            "--max-wall-seconds",
            "0",
            "--json-only",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert calls["fd"]["case_name"] == "nfp3_QI_fixed_resolution_final"
    assert calls["line"]["selected_ky_indices"] == (1, 2)


@pytest.mark.parametrize(
    ("gate", "writer", "payload", "out_name", "json_key", "csv_token"),
    [
        (
            alpha_gate,
            "write_vmec_boozer_aggregate_alpha_holdout_artifacts",
            {
                **_alpha_holdout_payload(),
                "artifact_kind": "vmec_boozer_aggregate_alpha_holdout_gate",
                "wall_seconds": 1.25,
            },
            "alpha_holdout.png",
            "artifact_kind",
            "heldout",
        ),
        (
            surface_gate,
            "write_vmec_boozer_aggregate_surface_holdout_artifacts",
            {
                **_surface_holdout_payload(),
                "kind": "vmec_boozer_aggregate_surface_holdout_gate",
                "artifact_kind": "vmec_boozer_aggregate_surface_holdout_gate",
                "blocked": False,
                "blockers": [],
                "wall_seconds": 1.25,
                "holdout_split": {
                    "training_surface_indices": [18],
                    "training_alphas": [0.0],
                    "training_selected_ky_indices": [1, 2],
                    "holdout_surface_indices": [19],
                    "holdout_alphas": [0.0],
                    "holdout_selected_ky_indices": [1, 2],
                },
            },
            "surface_holdout.png",
            "artifact_kind",
            "heldout_surface",
        ),
        (
            objective_gate,
            "write_vmec_boozer_aggregate_line_search_artifacts",
            _line_search_payload(),
            "line_search.png",
            "passed",
            "candidate_objective",
        ),
        (
            comparison_gate,
            "write_vmec_boozer_aggregate_line_search_comparison_artifacts",
            _comparison_payload(),
            "comparison.png",
            "same_initial_update_direction",
            "initial_update_direction",
        ),
        (
            objective_gate,
            "write_vmec_boozer_aggregate_objective_artifacts",
            {
                **_objective_payload(),
                "samples": [
                    {
                        **_objective_payload()["samples"][0],
                        "torflux": 0.64,
                        "surface": 0.64,
                        "ky": 0.1,
                        "selected_ky": 0.1,
                        "ky_abs_error": 0.0,
                    },
                    _objective_payload()["samples"][1],
                ],
            },
            "aggregate_gate.png",
            "passed",
            "ky_abs_error",
        ),
        (
            multi_point_gate,
            "write_vmec_boozer_multi_point_objective_artifacts",
            multi_point_gate._annotate_payload(
                _multi_point_payload(),
                surfaces=(None,),
                alphas=(0.0, 0.5),
                selected_ky_indices=(1, 2),
                max_samples=8,
                max_wall_seconds=300.0,
                elapsed_wall_seconds=1.25,
            ),
            "multi_point_gate.png",
            "artifact_kind",
            "alpha",
        ),
        (
            second_gate,
            "write_vmec_boozer_second_equilibrium_aggregate_artifacts",
            {
                "kind": "vmec_boozer_second_equilibrium_aggregate_gate",
                "passed": True,
                "feasible": True,
                "case_name": "li383_low_res",
                "objective": "quasilinear_flux",
                "mode_bound": {
                    "mboz": 21,
                    "nboz": 21,
                    "minimum_required": 21,
                    "passed": True,
                },
                "sample_bound": {
                    "n_samples_requested": 2,
                    "max_samples": 4,
                    "passed": True,
                },
                "bounded_runtime": {
                    "max_wall_seconds": 300.0,
                    "elapsed_wall_seconds": 41.2,
                    "passed": True,
                },
                "finite_difference_passed": True,
                "line_search_passed": True,
                "finite_difference_summary": {
                    "minus_value": 9.80,
                    "base_value": 9.79,
                    "plus_value": 9.78,
                    "central_derivative": -1.0e5,
                    "response_abs": 0.02,
                    "curvature_ratio": 0.01,
                    "n_samples": 2,
                },
                "line_search_summary": {
                    "accepted_steps": 1,
                    "initial_objective": 9.79,
                    "final_objective": 9.78,
                    "relative_reduction": 1.0e-3,
                    "stop_reason": "max_steps",
                },
            },
            "second_gate.png",
            "kind",
            "fd_central_derivative",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_vmec_boozer_aggregate_writers(
    tmp_path: Path,
    gate,
    writer: str,
    payload: dict[str, object],
    out_name: str,
    json_key: str,
    csv_token: str,
) -> None:
    paths = getattr(gate, writer)(payload, out=tmp_path / out_name)
    _assert_artifacts(paths, json_key=json_key, csv_token=csv_token)


# ---- test_vmec_jax_qa_artifact_contracts.py ----

import argparse

import pytest

from support.paths import REPO_ROOT


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
    assert "vmec_jax_qa_promotion_smoke" not in str(
        candidate_mod.DEFAULT_CONSTRAINTS_DIR
    )
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


def _candidate_comparison_payload(
    tmp_path: Path,
    monkeypatch,
    *,
    transport_gate_passed: bool | None = None,
    transport_history_qs: float = 0.02,
    transport_gate_qs: float = 0.02,
    reproducibility_passed: bool | None = None,
    rerun_admission_passed: bool | None = None,
) -> dict[str, object]:
    """Build the common two-branch QA candidate evidence fixture."""

    constraints = tmp_path / "constraints"
    transport = tmp_path / "transport"
    _history(constraints)
    _history(transport, qs=transport_history_qs)
    _solved_gate(constraints, passed=True)
    if transport_gate_passed is not None:
        _solved_gate(
            transport,
            passed=transport_gate_passed,
            qs=transport_gate_qs,
        )
    if reproducibility_passed is not None:
        _wout_reproducibility_gate(transport, passed=reproducibility_passed)
    if rerun_admission_passed is not None:
        _rerun_wout_admission_gate(transport, passed=rerun_admission_passed)
    monkeypatch.setattr(
        candidate_mod,
        "_load_iota_profiles",
        lambda _root, *, wout_name="wout_final.nc": (
            np.asarray([0.0, 0.414, 0.427]),
            np.asarray([0.412, 0.421]),
        ),
    )
    return candidate_mod.build_payload(constraints, transport)


def test_payload_admits_only_authoritative_solved_wout_gates(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _candidate_comparison_payload(tmp_path, monkeypatch)
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
    payload = _candidate_comparison_payload(
        tmp_path,
        monkeypatch,
        transport_gate_passed=True,
    )

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
    payload = _candidate_comparison_payload(
        tmp_path,
        monkeypatch,
        transport_gate_passed=True,
        reproducibility_passed=False,
    )
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
    payload = _candidate_comparison_payload(
        tmp_path,
        monkeypatch,
        transport_gate_passed=True,
        reproducibility_passed=False,
        rerun_admission_passed=True,
    )
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
    payload = _candidate_comparison_payload(
        tmp_path,
        monkeypatch,
        transport_gate_passed=False,
        transport_history_qs=0.04,
        transport_gate_qs=0.08,
    )
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


def _optimization_status_payload(
    tmp_path: Path,
    *,
    projected_step_metric: float = 0.09,
    baseline_reproducibility_passed: bool | None = None,
    baseline_rerun_admission_passed: bool | None = None,
    nonlinear_claim_level: str = status_mod.EXPECTED_NONLINEAR_AUDIT_CLAIM_LEVEL,
) -> dict[str, object]:
    """Build the common QA transport-optimization status portfolio."""

    constraints = tmp_path / "constraints"
    direct = tmp_path / "direct"
    projected_base = tmp_path / "projected_base"
    projected_step = tmp_path / "projected_step"
    _candidate(
        constraints,
        passed=True,
        metric=None,
        wout_reproducibility_passed=baseline_reproducibility_passed,
        rerun_wout_admission_passed=baseline_rerun_admission_passed,
    )
    _candidate(direct, passed=False, metric=None, qs=1.0)
    _candidate(projected_base, passed=True, metric=0.1)
    _candidate(projected_step, passed=True, metric=projected_step_metric)
    return status_mod.build_payload(
        constraints_dir=constraints,
        direct_transport_dir=direct,
        projected_baseline_dir=projected_base,
        projected_step_dir=projected_step,
        **_supporting_artifacts(
            tmp_path,
            nonlinear_claim_level=nonlinear_claim_level,
        ),
    )


def test_build_payload_separates_gate_failures_from_transport_metrics(
    tmp_path: Path,
) -> None:
    payload = _optimization_status_payload(
        tmp_path,
        projected_step_metric=0.11,
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
    payload = _optimization_status_payload(tmp_path)

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
    payload = _optimization_status_payload(
        tmp_path,
        baseline_reproducibility_passed=False,
    )
    candidates = {candidate["label"]: candidate for candidate in payload["candidates"]}
    baseline = candidates["QA max_mode=5 baseline"]

    assert baseline["solved_wout_gate_passed"] is True
    assert baseline["wout_reproducibility_gate_passed"] is False
    assert baseline["passed_solved_wout_gate"] is False
    assert payload["summary"]["qa_baseline_gate_passed"] is False


def test_status_admits_explicit_authoritative_rerun_wout(tmp_path: Path) -> None:
    payload = _optimization_status_payload(
        tmp_path,
        baseline_reproducibility_passed=False,
        baseline_rerun_admission_passed=True,
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
    payload = _optimization_status_payload(
        tmp_path,
        nonlinear_claim_level="startup_window_observable",
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
    monkeypatch.setattr(
        gradient_mod, "build_boundary_transport_gradient_report", fake_report
    )
    monkeypatch.setattr(
        gradient_mod, "write_boundary_transport_gradient_report", fake_write
    )

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


# ---- test_vmec_misc_artifact_reports.py ----

"""Tests for VMEC artifact reports outside the VMEC/Boozer aggregate suite."""


from netCDF4 import Dataset
import pytest

from support.paths import load_artifact_tool


# External VMEC replicate ensemble assertions
def _build_external_vmec_replicate_ensemble_write_output(
    path: Path, offset: float
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    t = np.linspace(0.0, 100.0, 101)
    q = 10.0 + offset + 0.02 * np.sin(2.0 * np.pi * t / 20.0)
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", t.size)
        root.createDimension("s", 1)
        grids = root.createGroup("Grids")
        diagnostics = root.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        diagnostics.createVariable("HeatFlux_st", "f8", ("time", "s"))[:, :] = q[
            :, None
        ]


def test_replicate_ensemble_tool_builds_trace_reports_and_plot(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_external_vmec_replicate_ensemble")
    outputs = [
        tmp_path / "demo_nonlinear_t100_n64_seed31.out.nc",
        tmp_path / "demo_nonlinear_t100_n64_seed32.out.nc",
        tmp_path / "demo_nonlinear_t100_n64_dt0p04.out.nc",
    ]
    for path, offset in zip(outputs, (-0.05, 0.05, 0.0)):
        _build_external_vmec_replicate_ensemble_write_output(path, offset)
    out_dir = tmp_path / "artifacts"

    rc = mod.main(
        [
            *[str(path) for path in outputs],
            "--out-dir",
            str(out_dir),
            "--case",
            "demo_replicate_window",
            "--tmin",
            "50",
            "--tmax",
            "100",
            "--baseline-seed",
            "22",
            "--baseline-dt",
            "0.05",
            "--artifact-prefix",
            "docs/_static/demo_replicates",
            "--bootstrap-samples",
            "32",
        ]
    )

    assert rc == 0
    readiness = json.loads((out_dir / "replicate_ensemble_readiness.json").read_text())
    ensemble = json.loads((out_dir / "replicate_ensemble_gate.json").read_text())
    summary = json.loads(
        (out_dir / "demo_nonlinear_t100_n64_seed31_transport_window.json").read_text()
    )
    assert readiness["passed"] is True
    assert ensemble["passed"] is True
    assert summary["nonlinear_artifact"] == "demo_nonlinear_t100_n64_seed31.out.nc"
    assert len(list(out_dir.glob("*_heat_flux_trace.csv"))) == 3
    assert (out_dir / "replicate_ensemble_gate.png").exists()
    assert readiness["observed_artifacts"][0]["source_artifact"].startswith(
        "docs/_static/demo_replicates/"
    )


def test_replicate_ensemble_tool_can_collect_failed_diagnostic_points(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_external_vmec_replicate_ensemble")
    outputs = [
        tmp_path / "diagnostic_nonlinear_t100_n64_seed31.out.nc",
        tmp_path / "diagnostic_nonlinear_t100_n64_seed32.out.nc",
        tmp_path / "diagnostic_nonlinear_t100_n64_dt0p04.out.nc",
    ]
    for path, offset in zip(outputs, (-4.0, 0.0, 4.0)):
        _build_external_vmec_replicate_ensemble_write_output(path, offset)

    common_args = [
        *[str(path) for path in outputs],
        "--case",
        "diagnostic_landscape_point",
        "--tmin",
        "50",
        "--tmax",
        "100",
        "--baseline-seed",
        "22",
        "--baseline-dt",
        "0.05",
        "--bootstrap-samples",
        "32",
        "--max-mean-rel-spread",
        "0.01",
    ]
    strict_dir = tmp_path / "strict"
    relaxed_dir = tmp_path / "relaxed"

    strict_rc = mod.main([*common_args, "--out-dir", str(strict_dir)])
    relaxed_rc = mod.main(
        [*common_args, "--out-dir", str(relaxed_dir), "--allow-failed-gates"]
    )

    assert strict_rc == 1
    assert relaxed_rc == 0
    ensemble = json.loads((relaxed_dir / "replicate_ensemble_gate.json").read_text())
    assert ensemble["passed"] is False
    assert (relaxed_dir / "replicate_ensemble_gate.png").exists()


def test_replicate_ensemble_tool_handles_requested_window_outside_trace(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_external_vmec_replicate_ensemble")
    outputs = [
        tmp_path / "short_nonlinear_t100_n64_seed31.out.nc",
        tmp_path / "short_nonlinear_t100_n64_seed32.out.nc",
        tmp_path / "short_nonlinear_t100_n64_dt0p04.out.nc",
    ]
    for path, offset in zip(outputs, (-0.05, 0.05, 0.0)):
        _build_external_vmec_replicate_ensemble_write_output(path, offset)
    out_dir = tmp_path / "outside_window"

    rc = mod.main(
        [
            *[str(path) for path in outputs],
            "--out-dir",
            str(out_dir),
            "--case",
            "outside_requested_window",
            "--tmin",
            "200",
            "--tmax",
            "300",
            "--bootstrap-samples",
            "16",
            "--allow-failed-gates",
        ]
    )

    assert rc == 0
    ensemble = json.loads((out_dir / "replicate_ensemble_gate.json").read_text())
    report = json.loads(
        next(
            (out_dir / "nonlinear_window_convergence_reports").glob("*seed31*")
        ).read_text()
    )
    assert ensemble["passed"] is False
    assert ensemble["statistics"]["n_finite_means"] == 0
    assert ensemble["statistics"]["ensemble_mean"] is None
    assert report["window"]["n_finite_late"] == 0
    assert (out_dir / "replicate_ensemble_gate.png").exists()


@pytest.mark.parametrize(
    ("filename", "baseline_dt", "expected"),
    [
        (
            "demo_nonlinear_t100_n64_seed32_dt0p04.out.nc",
            0.05,
            ("seed_timestep", "seed32_dt0p04", 32, 0.04),
        ),
        (
            "demo_nonlinear_t250_n48_dt0p01_gpu.out.nc",
            0.05,
            ("timestep", "dt0p01", 22, 0.01),
        ),
        (
            "solovev_reference_repair_dt002_amp1em5_n48_seed31.out.nc",
            0.02,
            ("seed", "seed31", 31, 0.02),
        ),
        (
            "solovev_reference_repair_dt002_amp1em5_n48_dt0p01_gpu.out.nc",
            0.02,
            ("timestep", "dt0p01", 22, 0.01),
        ),
    ],
    ids=["joint", "device_suffix", "protocol_dt_seed", "protocol_dt_timestep"],
)
def test_replicate_ensemble_tool_parses_variant_contract(
    tmp_path: Path,
    filename: str,
    baseline_dt: float,
    expected: tuple[str, str, int, float],
) -> None:
    mod = load_artifact_tool("build_external_vmec_replicate_ensemble")
    axis, label, seed, timestep = expected

    variant = mod._variant_from_path(
        tmp_path / filename,
        baseline_seed=22,
        baseline_dt=baseline_dt,
    )

    assert variant == {
        "variant_axis": axis,
        "variant_label": label,
        "seed": seed,
        "dt": timestep,
        "variant": {"seed": seed, "timestep": timestep},
    }


# VMEC-JAX boundary-chain collection assertions
def _build_vmec_jax_boundary_chain_collection_probe(
    path: Path, *, index: int, exact_ok: bool, growth_ok: bool
) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "vmec_jax_boundary_chain_probe",
                "index": index,
                "name": f"coeff{index}",
                "summary": {
                    "kind": "vmec_jax_boundary_chain_summary",
                    "finite": True,
                    "classification": (
                        "exact_fd_and_frozen_axis_replay_consistent"
                        if exact_ok
                        else "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive"
                    ),
                    "passes": {
                        "final_state_matches_exact_fd": True,
                        "frozen_axis_matches_exact_fd": exact_ok,
                        "frozen_axis_jvp_vjp_consistent": True,
                    },
                    "errors": {
                        "frozen_axis_vs_exact_fd_rel": 0.02 if exact_ok else 0.4,
                    },
                    "metrics": {
                        "exact_fd_cost_gradient": 0.1,
                        "frozen_axis_replay_cost_gradient": 0.1 if exact_ok else 0.2,
                    },
                },
                "growth_branch_locality": {
                    "enabled": True,
                    "passed": growth_ok,
                    "classification": (
                        "all_samples_dominant_growth_branch_locally_consistent"
                        if growth_ok
                        else "growth_branch_locality_failed_or_incomplete"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_collection_payload_counts_growth_branch_status(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _build_vmec_jax_boundary_chain_collection_probe(
        first, index=1, exact_ok=True, growth_ok=True
    )
    _build_vmec_jax_boundary_chain_collection_probe(
        second, index=2, exact_ok=False, growth_ok=False
    )

    payload = load_artifact_tool(
        "build_vmec_state_to_input_mapping_response"
    ).build_boundary_chain_collection_payload([first, second])

    assert payload["finite"] is True
    assert (
        payload["classification"]
        == "mixed_exact_fd_consistency_with_branch_sensitive_modes"
    )
    assert payload["counts"]["n_exact_fd_consistent"] == 1
    assert payload["counts"]["n_growth_branch_locality_checked"] == 2
    assert payload["counts"]["n_growth_branch_locality_passed"] == 1
    assert payload["rows"][0]["growth_branch_locality_passed"] is True
    assert payload["rows"][1]["growth_branch_locality_passed"] is False
    assert payload["probe_jsons"] == [str(first), str(second)]
    assert "not a nonlinear transport optimization claim" in payload["claim_scope"]


def test_build_collection_main_writes_json(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    out = tmp_path / "collection.json"
    _build_vmec_jax_boundary_chain_collection_probe(
        first, index=1, exact_ok=True, growth_ok=True
    )

    rc = load_artifact_tool("build_vmec_state_to_input_mapping_response").main(
        [
            "boundary-chain-collection",
            "--probe-json",
            str(first),
            "--out-json",
            str(out),
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["counts"]["n_total"] == 1
    assert payload["counts"]["n_growth_branch_locality_passed"] == 1


# VMEC optimization candidate-screen assertions
def _write_candidate_screen_spectrum(
    path: Path, rows: list[tuple[float, float, float, float, float]]
) -> None:
    lines = ["ky,gamma,omega,kperp_eff2,heat_flux_weight_total"]
    lines.extend(
        f"{ky},{gamma},{omega},{kperp},{heat}" for ky, gamma, omega, kperp, heat in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_candidate_screen_rejects_nonpositive_kperp_even_with_large_growth(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_nonlinear_transport_admission")
    spectrum = tmp_path / "bad.csv"
    _write_candidate_screen_spectrum(
        spectrum,
        [
            (0.1, 1.2, -0.5, -0.7, 0.1),
            (0.2, 0.8, -0.4, -0.1, 0.2),
            (0.3, 0.4, -0.2, -0.2, 0.3),
        ],
    )

    row = mod.summarize_linear_spectrum(label="bad_metric", spectrum_path=spectrum)

    assert row["passed"] is False
    assert row["status"] == "invalid_metric_nonpositive_kperp2"
    assert "nonpositive_effective_kperp2" in row["blockers"]
    assert row["max_gamma"] == 1.2


def test_candidate_screen_accepts_positive_metric_launch_candidate(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_nonlinear_transport_admission")
    spectrum = tmp_path / "good.csv"
    _write_candidate_screen_spectrum(
        spectrum,
        [
            (0.1, 0.01, -0.5, 0.7, 0.1),
            (0.2, 0.04, -0.4, 0.8, 0.2),
            (0.3, 0.03, -0.2, 0.9, 0.3),
        ],
    )

    report = mod.build_linear_screen_report([("good", spectrum)])

    assert report["passed"] is True
    assert report["n_launch_candidates"] == 1
    assert report["rows"][0]["status"] == "nonlinear_launch_candidate"


def test_candidate_screen_tool_writes_fail_closed_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_nonlinear_transport_admission")
    spectrum = tmp_path / "marginal.csv"
    _write_candidate_screen_spectrum(
        spectrum,
        [
            (0.1, -0.01, -0.5, 0.7, 0.1),
            (0.2, 0.01, -0.4, 0.8, 0.2),
            (0.3, 0.015, -0.2, 0.9, 0.3),
        ],
    )
    out = tmp_path / "screen.json"

    assert (
        mod.main(
            ["linear-screen", "--spectrum", f"marginal:{spectrum}", "--out", str(out)]
        )
        == 2
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["rows"][0]["status"] == "marginal_or_incomplete_screen"
    assert out.with_suffix(".csv").exists()


def test_candidate_screen_rejects_nonfinite_metric_rows(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_nonlinear_transport_admission")
    spectrum = tmp_path / "nonfinite.csv"
    _write_candidate_screen_spectrum(
        spectrum,
        [
            (0.1, 0.03, -0.5, 0.7, 0.1),
            (0.2, 0.04, -0.4, float("nan"), 0.2),
            (0.3, 0.03, -0.2, 0.9, float("inf")),
        ],
    )

    row = mod.summarize_linear_spectrum(
        label="nonfinite_metric", spectrum_path=spectrum
    )

    assert row["passed"] is False
    assert row["min_kperp_eff2"] is None
    assert row["max_heat_flux_weight_total"] is None
    assert "nonpositive_effective_kperp2" in row["blockers"]
    assert "nonfinite_heat_flux_weight" in row["blockers"]


# VMEC state-control bracket sweep assertions
def _build_vmec_state_control_bracket_sweep_status_gate(
    path: Path, *, alpha: float, parameter: str, response: float, passed: bool
) -> None:
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "blockers": [] if passed else ["fd_response_resolved"],
                "delta_parameter": alpha,
                "parameter_name": parameter,
                "config": {
                    "min_fd_response_fraction": 0.03,
                    "max_fd_asymmetry_rel": 0.5,
                    "max_gradient_uncertainty_rel": 0.5,
                },
                "metrics": {
                    "response_fraction": response,
                    "fd_asymmetry_rel": 0.2 if passed else 2.0,
                    "gradient_uncertainty_rel": 0.1 if passed else 4.0,
                    "baseline_window_mean": 1.0,
                    "plus_window_mean": 1.1,
                    "minus_window_mean": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_vmec_state_control_bracket_sweep_status(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_vmec_state_to_input_mapping_response")
    gate_a = tmp_path / "gate_a.json"
    gate_b = tmp_path / "gate_b.json"
    run_summary = tmp_path / "summary.json"
    out_prefix = tmp_path / "status"
    _build_vmec_state_control_bracket_sweep_status_gate(
        gate_a,
        alpha=0.003,
        parameter="state_control_rsin_mid_surface_m1",
        response=0.004,
        passed=False,
    )
    _build_vmec_state_control_bracket_sweep_status_gate(
        gate_b,
        alpha=0.01,
        parameter="state_control_zcos_mid_surface_m1",
        response=0.04,
        passed=True,
    )
    run_summary.write_text(
        json.dumps(
            {"successes": 36, "failures": [], "started_at": 1.0, "finished_at": 4.5}
        )
    )

    report = mod.build_bracket_sweep_status(
        [gate_b, gate_a], run_summary=run_summary, out_prefix=out_prefix
    )

    assert report["passed"] is False
    assert report["summary"]["central_fd_gates_passed"] == 1
    assert report["summary"]["central_fd_gates_total"] == 2
    assert report["summary"]["nonlinear_runs_completed"] == 36
    assert report["rows"][0]["alpha_delta"] == 0.003
    assert out_prefix.with_suffix(".json").exists()
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


# VMEC state-to-input mapping assertions
def _build_vmec_state_to_input_mapping_response_controls() -> list[dict[str, object]]:
    return [
        {"state_parameter": "Rsin_mid_surface_m1"},
        {"state_parameter": "Zcos_mid_surface_m1"},
    ]


def _build_vmec_state_to_input_mapping_response_directions() -> list[dict[str, object]]:
    return [
        {
            "coefficient": "RBC(1,1)",
            "coefficient_slug": "rbc_1_1",
            "delta_parameter": 0.1,
        },
        {
            "coefficient": "ZBS(1,1)",
            "coefficient_slug": "zbs_1_1",
            "delta_parameter": 0.2,
        },
    ]


def _build_vmec_state_to_input_mapping_response_sample(
    value: tuple[float, float],
) -> dict[str, float]:
    return {
        "Rsin_mid_surface_m1": value[0],
        "Zcos_mid_surface_m1": value[1],
    }


def test_mapping_report_fails_closed_for_zero_symmetric_response() -> None:
    mod = load_artifact_tool("build_vmec_state_to_input_mapping_response")
    samples = {
        "RBC(1,1)": {
            "baseline": _build_vmec_state_to_input_mapping_response_sample((0.0, 0.0)),
            "plus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, 0.0)
            ),
            "minus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, 0.0)
            ),
        },
        "ZBS(1,1)": {
            "baseline": _build_vmec_state_to_input_mapping_response_sample((0.0, 0.0)),
            "plus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, 0.0)
            ),
            "minus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, 0.0)
            ),
        },
    }

    report = mod.mapping_report_from_samples(
        case="zero",
        admitted_state_controls=_build_vmec_state_to_input_mapping_response_controls(),
        input_directions=_build_vmec_state_to_input_mapping_response_directions(),
        samples=samples,
    )

    assert report["passed"] is False
    assert report["jacobian"]["rank"] == 0
    assert report["jacobian"]["condition_number"] is None
    assert "zero_state_response" in report["blockers"]
    assert all(
        "state_control_not_observed" in row["blockers"] for row in report["controls"]
    )
    json.dumps(report, allow_nan=False)


def test_mapping_report_passes_conditioned_square_response() -> None:
    mod = load_artifact_tool("build_vmec_state_to_input_mapping_response")
    samples = {
        "RBC(1,1)": {
            "baseline": _build_vmec_state_to_input_mapping_response_sample((0.0, 0.0)),
            "plus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.1, 0.0)
            ),
            "minus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (-0.1, 0.0)
            ),
        },
        "ZBS(1,1)": {
            "baseline": _build_vmec_state_to_input_mapping_response_sample((0.0, 0.0)),
            "plus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, 0.2)
            ),
            "minus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, -0.2)
            ),
        },
    }

    report = mod.mapping_report_from_samples(
        case="identity",
        admitted_state_controls=_build_vmec_state_to_input_mapping_response_controls(),
        input_directions=_build_vmec_state_to_input_mapping_response_directions(),
        samples=samples,
    )

    assert report["passed"] is True
    assert report["jacobian"]["rank"] == 2
    assert report["jacobian"]["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert [row["passed"] for row in report["controls"]] == [True, True]


def test_mapping_report_rejects_missing_states_and_bad_deltas() -> None:
    mod = load_artifact_tool("build_vmec_state_to_input_mapping_response")
    directions = _build_vmec_state_to_input_mapping_response_directions()
    directions[0]["delta_parameter"] = float("nan")
    with pytest.raises(ValueError, match="delta_parameter"):
        mod.mapping_report_from_samples(
            case="bad",
            admitted_state_controls=_build_vmec_state_to_input_mapping_response_controls(),
            input_directions=directions,
            samples={},
        )

    with pytest.raises(ValueError, match="missing baseline"):
        mod.mapping_report_from_samples(
            case="missing",
            admitted_state_controls=_build_vmec_state_to_input_mapping_response_controls(),
            input_directions=_build_vmec_state_to_input_mapping_response_directions(),
            samples={
                "RBC(1,1)": {
                    "baseline": _build_vmec_state_to_input_mapping_response_sample(
                        (0.0, 0.0)
                    )
                }
            },
        )


# External VMEC time-horizon gate assertions
def _plot_external_vmec_time_horizon_gate_write_gate(
    tmp_path: Path, name: str, means: tuple[float, float], *, passed: bool = True
) -> Path:
    payload = {
        "kind": "external_vmec_nonlinear_grid_convergence_gate",
        "passed": passed,
        "common_window": {"max_pairwise_heat_flux_symmetric_relative_difference": 0.05},
        "least_windows": {"max_pairwise_heat_flux_symmetric_relative_difference": 0.04},
        "runs": [
            {
                "label": "n64",
                "common_window": {"heat_flux_mean": means[0]},
                "least_trending_window": {"heat_flux_mean": means[0] * 0.99},
            },
            {
                "label": "n80",
                "common_window": {"heat_flux_mean": means[1]},
                "least_trending_window": {"heat_flux_mean": means[1] * 1.01},
            },
        ],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_time_horizon_gate_passes_for_stable_high_grid_means(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_external_vmec_nonlinear_convergence_gate")
    first = _plot_external_vmec_time_horizon_gate_write_gate(
        tmp_path, "t250", (10.0, 10.4)
    )
    second = _plot_external_vmec_time_horizon_gate_write_gate(
        tmp_path, "t350", (10.2, 10.5)
    )

    paths = mod.write_time_horizon_panel(
        [(250.0, first), (350.0, second)],
        out=tmp_path / "horizon.png",
        case="synthetic time horizon",
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["kind"] == "external_vmec_time_horizon_gate"
    assert payload["passed"] is True
    assert payload["promotion_gate"]["passed"] is False
    assert (
        payload["claim_level"]
        == "passed_high_grid_time_horizon_candidate_not_replicated_holdout"
    )
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["csv"]).exists()


@pytest.mark.parametrize(
    ("means", "second_passed", "failed_metric", "claim_level"),
    [
        pytest.param(
            ((10.0, 10.0), (14.0, 14.0)),
            True,
            "common_window_time_horizon_relative_change",
            "negative_time_horizon_result_not_transport_validation",
            id="large-horizon-shift",
        ),
        pytest.param(
            ((10.0, 10.4), (10.2, 10.5)),
            False,
            "failed_grid_gate_count",
            None,
            id="failed-input-grid-gate",
        ),
    ],
)
def test_time_horizon_gate_fails_closed(
    tmp_path: Path,
    means: tuple[tuple[float, float], tuple[float, float]],
    second_passed: bool,
    failed_metric: str,
    claim_level: str | None,
) -> None:
    mod = load_artifact_tool("plot_external_vmec_nonlinear_convergence_gate")
    first = _plot_external_vmec_time_horizon_gate_write_gate(tmp_path, "t250", means[0])
    second = _plot_external_vmec_time_horizon_gate_write_gate(
        tmp_path, "t350", means[1], passed=second_passed
    )

    payload = mod.build_time_horizon_payload(
        [(250.0, first), (350.0, second)],
        case="synthetic time horizon",
    )

    failed = {
        gate["metric"] for gate in payload["gate_report"]["gates"] if not gate["passed"]
    }
    assert payload["passed"] is False
    assert failed_metric in failed
    if claim_level is not None:
        assert payload["claim_level"] == claim_level


# VMEC-JAX equilibrium inventory assertions
def _plot_vmec_jax_equilibrium_inventory_write_wout(
    path: Path,
    *,
    nfp: int,
    ntor: int,
    aspect: float,
    iota_edge: float,
    aminor: float = 0.3,
    rmajor: float = 1.2,
    volume: float = 2.0,
) -> None:
    with Dataset(path, "w") as ds:
        ds.createDimension("radius", 3)
        ds.createVariable("nfp", "i4").assignValue(nfp)
        ds.createVariable("ns", "i4").assignValue(3)
        ds.createVariable("mpol", "i4").assignValue(4)
        ds.createVariable("ntor", "i4").assignValue(ntor)
        ds.createVariable("aspect", "f8").assignValue(aspect)
        ds.createVariable("Aminor_p", "f8").assignValue(aminor)
        ds.createVariable("Rmajor_p", "f8").assignValue(rmajor)
        ds.createVariable("volume_p", "f8").assignValue(volume)
        ds.createVariable("betatotal", "f8").assignValue(0.01)
        iota = ds.createVariable("iotaf", "f8", ("radius",))
        iota[:] = [0.4, 0.5, iota_edge]
        pres = ds.createVariable("presf", "f8", ("radius",))
        pres[:] = [1.0, 0.5, 0.0]


def test_vmec_jax_inventory_report_and_figure_are_replayable(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_vmec_jax_equilibrium_inventory")
    _plot_vmec_jax_equilibrium_inventory_write_wout(
        tmp_path / "wout_circular_tokamak.nc", nfp=1, ntor=0, aspect=3.0, iota_edge=0.25
    )
    _plot_vmec_jax_equilibrium_inventory_write_wout(
        tmp_path / "wout_nfp4_QH_warm_start.nc",
        nfp=4,
        ntor=2,
        aspect=7.0,
        iota_edge=-1.1,
    )

    report = mod.build_inventory(tmp_path)
    paths = mod.write_inventory_figure(report, out=tmp_path / "inventory.png")

    assert report["kind"] == "vmec_jax_equilibrium_inventory"
    assert report["claim_level"] == "equilibrium_selection_not_transport_validation"
    assert report["n_equilibria"] == 2
    assert report["family_counts"]["axisymmetric"] == 1
    assert "wout_nfp4_QH_warm_start.nc" in report["recommended_next_linear_portfolio"]
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["rows"][0]["validation_role"].startswith("external_vmec_fixture")


def test_vmec_jax_inventory_defers_degenerate_reference_scales(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_vmec_jax_equilibrium_inventory")
    _plot_vmec_jax_equilibrium_inventory_write_wout(
        tmp_path / "wout_nfp4_QH_warm_start.nc",
        nfp=4,
        ntor=2,
        aspect=7.0,
        iota_edge=-1.1,
    )
    _plot_vmec_jax_equilibrium_inventory_write_wout(
        tmp_path / "wout_LandremanPaul2021_QA_lowres.nc",
        nfp=2,
        ntor=8,
        aspect=0.0,
        iota_edge=0.4,
        aminor=0.0,
        rmajor=0.0,
        volume=0.0,
    )

    report = mod.build_inventory(tmp_path)
    degenerate = next(
        row
        for row in report["rows"]
        if row["name"] == "wout_LandremanPaul2021_QA_lowres.nc"
    )

    assert degenerate["reference_scale_valid"] is False
    assert (
        degenerate["geometry_contract_status"]
        == "deferred_degenerate_vmec_reference_scale"
    )
    assert degenerate["candidate_score"] == 0.0
    assert (
        "wout_LandremanPaul2021_QA_lowres.nc"
        not in report["recommended_next_linear_portfolio"]
    )
