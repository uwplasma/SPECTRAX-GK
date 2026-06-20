#!/usr/bin/env python3
"""Build a manuscript-scope readiness dashboard.

This dashboard is intentionally narrower than ``open_research_lane_status``:
W7-X zonal recurrence and TEM/kinetic-electron extensions are marked as
deferred by scope, while the quasilinear and stellarator-optimization lanes are
tracked against the claims that can honestly appear in the current manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # type: ignore[import-untyped]  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "manuscript_readiness_status.png"

STATUS_ORDER = {"closed": 0, "partial": 1, "open": 2, "deferred": 3, "blocked": 4}
STATUS_COLORS = {
    "closed": "#2a9d8f",
    "partial": "#e9c46a",
    "open": "#f4a261",
    "deferred": "#8d99ae",
    "blocked": "#d1495b",
}


def _read_json(root: Path, relative: str) -> dict[str, Any] | None:
    path = root / relative
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{relative} must contain a JSON object")
    return payload


def _finite_float(value: object, default: float | None = None) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _count_points(report: dict[str, Any] | None, split: str) -> int:
    if not report:
        return 0
    points = report.get("points", [])
    if not isinstance(points, list):
        return 0
    return sum(
        1 for point in points if isinstance(point, dict) and point.get("split") == split
    )


def _profile_seconds(
    payload: dict[str, Any] | None, label: str, kernel: str
) -> float | None:
    rows = {} if payload is None else payload.get("rows", {})
    if not isinstance(rows, dict):
        return None
    row = rows.get(label, {})
    if not isinstance(row, dict):
        return None
    seconds = row.get("seconds", {})
    if not isinstance(seconds, dict):
        return None
    return _finite_float(seconds.get(kernel))


def _all_optimization_objectives_passed(payload: dict[str, Any] | None) -> bool:
    results = [] if payload is None else payload.get("results", [])
    if not isinstance(results, list) or not results:
        return False
    for result in results:
        if not isinstance(result, dict):
            return False
        if not bool(result.get("gradient_gate", {}).get("passed", False)):
            return False
        initial = _finite_float(result.get("initial_objective"))
        final = _finite_float(result.get("final_objective"))
        if initial is None or final is None or final >= initial:
            return False
    return True


def _optimization_reduction_summary(
    payload: dict[str, Any] | None,
) -> dict[str, float | int | None]:
    results = [] if payload is None else payload.get("results", [])
    if not isinstance(results, list) or not results:
        return {
            "n_objectives": 0,
            "best_reduction_factor": None,
            "worst_reduction_factor": None,
        }
    factors = []
    for result in results:
        if not isinstance(result, dict):
            continue
        initial = _finite_float(result.get("initial_objective"))
        final = _finite_float(result.get("final_objective"))
        if initial is None or final is None or initial <= 0.0:
            continue
        factors.append(final / initial)
    return {
        "n_objectives": len(results),
        "best_reduction_factor": None if not factors else float(min(factors)),
        "worst_reduction_factor": None if not factors else float(max(factors)),
    }


def _optimization_uq_gate_passed(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    return bool(
        payload.get("all_gradient_gates_passed", False)
        and payload.get("all_sensitivity_maps_full_rank", False)
        and payload.get("claim_level")
        == "reduced_objective_uq_and_sensitivity_validation_not_full_vmec_gk_optimization"
    )


def build_manuscript_readiness_payload(root: Path = ROOT) -> dict[str, Any]:
    """Return a JSON-ready manuscript-scope readiness payload."""

    root = Path(root)
    ql_inputs = _read_json(
        root, "docs/_static/quasilinear_validated_calibration_inputs.json"
    )
    ql_holdout = _read_json(
        root, "docs/_static/quasilinear_stellarator_train_holdout_report.json"
    )
    ql_sweep = _read_json(root, "docs/_static/quasilinear_saturation_rule_sweep.json")
    ql_shape = _read_json(root, "docs/_static/quasilinear_shape_aware_saturation.json")
    ql_uq = _read_json(root, "docs/_static/quasilinear_candidate_uncertainty.json")
    ql_dataset = _read_json(root, "docs/_static/quasilinear_dataset_sufficiency.json")
    ql_model_status = _read_json(
        root, "docs/_static/quasilinear_model_selection_status.json"
    )
    ql_guardrails = _read_json(
        root, "docs/_static/quasilinear_promotion_guardrails.json"
    )
    dshape_replicate = _read_json(
        root,
        "docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.json",
    )
    circular_replicate = _read_json(
        root,
        "docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json",
    )
    geom = _read_json(root, "docs/_static/differentiable_geometry_bridge.json")
    geom_matrix = _read_json(root, "docs/_static/vmec_boozer_parity_matrix.json")
    opt = _read_json(root, "docs/_static/stellarator_itg_optimization_comparison.json")
    opt_uq = _read_json(root, "docs/_static/stellarator_itg_optimization_uq.json")
    solver_grad = _read_json(root, "docs/_static/solver_objective_gradient_gate.json")
    vmec_solver_grad = _read_json(
        root, "docs/_static/vmec_boozer_solver_frequency_gradient_gate.json"
    )
    vmec_ql_grad = _read_json(
        root, "docs/_static/vmec_boozer_quasilinear_gradient_gate.json"
    )
    vmec_nl_window_grad = _read_json(
        root, "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json"
    )
    vmec_li383_nl_window_grad = _read_json(
        root,
        "docs/_static/vmec_boozer_li383_nonlinear_window_gradient_gate.json",
    )
    vmec_gradient_matrix = _read_json(
        root, "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
    )
    nonlinear_fd_audit = _read_json(root, "docs/_static/nonlinear_window_fd_audit.json")
    vmec_nonlinear_fd_audit = _read_json(
        root, "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json"
    )
    production_nl_guard = _read_json(
        root, "docs/_static/production_nonlinear_optimization_guard.json"
    )
    baseline_optimized_audit = _read_json(
        root, "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json"
    )
    nonlinear_control_mean_gate = _read_json(
        root,
        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json",
    )
    profile = _read_json(
        root, "docs/_static/nonlinear_sharding_profile_office_gpu.json"
    )
    rhs_profile = _read_json(root, "docs/_static/nonlinear_rhs_profile.json")
    rhs_miller = _read_json(root, "docs/_static/nonlinear_rhs_profile_miller.json")
    rhs_stellarator = _read_json(
        root, "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json"
    )
    full_rhs_trace_cpu = _read_json(
        root, "docs/_static/full_nonlinear_rhs_trace_summary.json"
    )
    full_rhs_trace_gpu = _read_json(
        root, "docs/_static/full_nonlinear_rhs_trace_gpu_summary.json"
    )

    ql_inputs_passed = bool((ql_inputs or {}).get("passed", False))
    ql_holdout_promoted = bool((ql_holdout or {}).get("passed", False))
    ql_holdout_mean = _finite_float(
        (ql_holdout or {})
        .get("by_split", {})
        .get("holdout", {})
        .get("mean_abs_relative_error")
    )
    ql_sweep_gate = (
        (ql_sweep or {}).get("promotion_gate", {})
        if isinstance((ql_sweep or {}).get("promotion_gate", {}), dict)
        else {}
    )
    ql_shape_gate = (
        (ql_shape or {}).get("promotion_gate", {})
        if isinstance((ql_shape or {}).get("promotion_gate", {}), dict)
        else {}
    )
    ql_uq_gate = (
        (ql_uq or {}).get("promotion_gate", {})
        if isinstance((ql_uq or {}).get("promotion_gate", {}), dict)
        else {}
    )
    ql_dataset_gate = (
        (ql_dataset or {}).get("promotion_gate", {})
        if isinstance((ql_dataset or {}).get("promotion_gate", {}), dict)
        else {}
    )
    ql_model_status_gate = (
        (ql_model_status or {}).get("promotion_gate", {})
        if isinstance((ql_model_status or {}).get("promotion_gate", {}), dict)
        else {}
    )
    ql_candidate_promoted = (
        bool(ql_uq_gate.get("passed", False))
        and bool(ql_dataset_gate.get("passed", False))
        and bool(ql_model_status_gate.get("passed", False))
    )
    ql_guardrails_passed = bool((ql_guardrails or {}).get("passed", False))
    ql_dataset_requirements = (
        (ql_dataset or {}).get("requirements", {})
        if isinstance((ql_dataset or {}).get("requirements", {}), dict)
        else {}
    )
    dshape_replicate_stats = (
        (dshape_replicate or {}).get("statistics", {})
        if isinstance((dshape_replicate or {}).get("statistics", {}), dict)
        else {}
    )
    circular_replicate_stats = (
        (circular_replicate or {}).get("statistics", {})
        if isinstance((circular_replicate or {}).get("statistics", {}), dict)
        else {}
    )
    ql_negative_closed = bool(
        ql_inputs_passed
        and ql_holdout is not None
        and ql_sweep is not None
        and ql_shape is not None
        and ql_uq is not None
        and ql_dataset is not None
        and not ql_holdout_promoted
        and not bool(ql_sweep_gate.get("passed", False))
        and not bool(ql_shape_gate.get("passed", False))
        and not bool(ql_uq_gate.get("passed", False))
        and not bool(ql_dataset_gate.get("passed", False))
        and ql_guardrails_passed
    )

    matrix_summary = (
        (geom_matrix or {}).get("summary", {})
        if isinstance((geom_matrix or {}).get("summary", {}), dict)
        else {}
    )
    geom_matrix_passed = bool(matrix_summary.get("all_equal_arc_passed", False))
    geom_uq = (
        (geom or {}).get("uq", {})
        if isinstance((geom or {}).get("uq", {}), dict)
        else {}
    )
    geom_sensitivity = (
        (geom or {}).get("sensitivity", {})
        if isinstance((geom or {}).get("sensitivity", {}), dict)
        else {}
    )
    geom_bridge_passed = (
        _finite_float(geom_sensitivity.get("max_abs_ad_fd_error")) is not None
        and int(geom_uq.get("sensitivity_map_rank", 0)) >= 2
    )

    opt_reductions = _optimization_reduction_summary(opt)
    opt_reduced_objectives_passed = _all_optimization_objectives_passed(opt)
    opt_uq_passed = _optimization_uq_gate_passed(opt_uq)
    solver_gradient_passed = bool((solver_grad or {}).get("passed", False))
    solver_gradient_source = str((solver_grad or {}).get("source_scope", "missing"))
    solver_gradient_full_vmec_frequency = (
        bool((vmec_solver_grad or {}).get("passed", False))
        and str((vmec_solver_grad or {}).get("source_scope", "missing"))
        == "mode21_vmec_boozer_state"
    )
    solver_gradient_full_vmec_quasilinear = (
        bool((vmec_ql_grad or {}).get("passed", False))
        and str((vmec_ql_grad or {}).get("source_scope", "missing"))
        == "mode21_vmec_boozer_state"
    )
    solver_gradient_reduced_nonlinear_window = (
        bool((vmec_nl_window_grad or {}).get("passed", False))
        and str((vmec_nl_window_grad or {}).get("source_scope", "missing"))
        == "mode21_vmec_boozer_state"
    )
    solver_gradient_reduced_nonlinear_window_multi_equilibrium = bool(
        solver_gradient_reduced_nonlinear_window
        and bool((vmec_li383_nl_window_grad or {}).get("passed", False))
        and str((vmec_li383_nl_window_grad or {}).get("source_scope", "missing"))
        == "mode21_vmec_boozer_state"
    )
    gradient_matrix_summary = (
        (vmec_gradient_matrix or {}).get("summary", {})
        if isinstance((vmec_gradient_matrix or {}).get("summary", {}), dict)
        else {}
    )
    solver_gradient_multi_equilibrium = bool(
        (vmec_gradient_matrix or {}).get("passed", False)
    )
    nonlinear_fd_metrics = (
        (nonlinear_fd_audit or {}).get("metrics", {})
        if isinstance((nonlinear_fd_audit or {}).get("metrics", {}), dict)
        else {}
    )
    startup_nonlinear_plumbing_fd_path_gate = bool(
        (nonlinear_fd_audit or {}).get("startup_nonlinear_plumbing_fd_path_gate", False)
    )
    nonlinear_transport_average_gate = bool(
        (nonlinear_fd_audit or {}).get("transport_average_gate", False)
    )
    production_nonlinear_observable_fd_path_gate = bool(
        (nonlinear_fd_audit or {}).get(
            "production_nonlinear_observable_fd_path_gate", False
        )
    )
    vmec_nonlinear_fd_metrics = (
        (vmec_nonlinear_fd_audit or {}).get("metrics", {})
        if isinstance((vmec_nonlinear_fd_audit or {}).get("metrics", {}), dict)
        else {}
    )
    vmec_boozer_startup_nonlinear_plumbing_fd_path_gate = bool(
        (vmec_nonlinear_fd_audit or {}).get(
            "vmec_boozer_startup_nonlinear_plumbing_fd_path_gate", False
        )
    )
    vmec_boozer_nonlinear_transport_average_gate = bool(
        (vmec_nonlinear_fd_audit or {}).get("transport_average_gate", False)
    )
    vmec_boozer_production_nonlinear_observable_fd_path_gate = bool(
        (vmec_nonlinear_fd_audit or {}).get(
            "vmec_boozer_production_nonlinear_observable_fd_path_gate", False
        )
    )
    production_nl_guard_summary = (
        (production_nl_guard or {}).get("summary", {})
        if isinstance((production_nl_guard or {}).get("summary", {}), dict)
        else {}
    )
    production_nonlinear_optimization_safe_to_release = bool(
        (production_nl_guard or {}).get("safe_to_release", False)
    )
    production_nonlinear_optimization_promoted = bool(
        (production_nl_guard or {}).get(
            "production_nonlinear_optimization_promoted", False
        )
    )
    baseline_optimized_comparison = (
        (baseline_optimized_audit or {}).get("comparison", {})
        if isinstance((baseline_optimized_audit or {}).get("comparison", {}), dict)
        else {}
    )
    nonlinear_control_mean_summary = (
        (nonlinear_control_mean_gate or {}).get("summary", {})
        if isinstance((nonlinear_control_mean_gate or {}).get("summary", {}), dict)
        else {}
    )
    matched_baseline_optimized_audit_passed = bool(
        (baseline_optimized_audit or {}).get("passed", False)
    )
    solver_gradient_closed = bool(
        solver_gradient_passed
        and solver_gradient_full_vmec_frequency
        and solver_gradient_full_vmec_quasilinear
        and solver_gradient_multi_equilibrium
    )
    solver_gradient_status = (
        "closed"
        if solver_gradient_closed
        else ("partial" if solver_gradient_passed else "open")
    )
    profile_best = (
        (profile or {}).get("best_identity_preserving_candidate", {})
        if isinstance(
            (profile or {}).get("best_identity_preserving_candidate", {}), dict
        )
        else {}
    )
    rhs_speedups = (
        (rhs_profile or {}).get("spectral_speedups", {})
        if isinstance((rhs_profile or {}).get("spectral_speedups", {}), dict)
        else {}
    )
    rhs_cpu = (
        rhs_speedups.get("cpu", {})
        if isinstance(rhs_speedups.get("cpu", {}), dict)
        else {}
    )
    rhs_gpu = (
        rhs_speedups.get("gpu", {})
        if isinstance(rhs_speedups.get("gpu", {}), dict)
        else {}
    )
    miller_cpu_grid_full = _profile_seconds(rhs_miller, "CPU grid", "full_rhs")
    miller_gpu_grid_full = _profile_seconds(rhs_miller, "GPU grid", "full_rhs")
    miller_gpu_spectral_full = _profile_seconds(rhs_miller, "GPU spectral", "full_rhs")
    w7x_cpu_full = _profile_seconds(rhs_stellarator, "W7-X CPU", "full_rhs")
    w7x_gpu_full = _profile_seconds(rhs_stellarator, "W7-X GPU", "full_rhs")
    hsx_cpu_full = _profile_seconds(rhs_stellarator, "HSX CPU", "full_rhs")
    hsx_gpu_full = _profile_seconds(rhs_stellarator, "HSX GPU", "full_rhs")
    full_rhs_cpu_warm = _finite_float((full_rhs_trace_cpu or {}).get("warm_seconds"))
    full_rhs_gpu_warm = _finite_float((full_rhs_trace_gpu or {}).get("warm_seconds"))
    full_rhs_gpu_transposes = _finite_float(
        (full_rhs_trace_gpu or {}).get("hlo_token_counts", {}).get("transpose")
        if isinstance((full_rhs_trace_gpu or {}).get("hlo_token_counts", {}), dict)
        else None
    )
    release_performance_closed = bool(
        (profile or {}).get("identity_gate_pass", False)
        and miller_cpu_grid_full is not None
        and miller_gpu_grid_full is not None
        and miller_gpu_spectral_full is not None
        and w7x_cpu_full is not None
        and w7x_gpu_full is not None
        and hsx_cpu_full is not None
        and hsx_gpu_full is not None
        and full_rhs_cpu_warm is not None
        and full_rhs_gpu_warm is not None
        and miller_gpu_grid_full < miller_cpu_grid_full
        and w7x_gpu_full < w7x_cpu_full
        and hsx_gpu_full < hsx_cpu_full
        and full_rhs_gpu_warm < full_rhs_cpu_warm
    )

    lanes: list[dict[str, Any]] = [
        {
            "lane": "Quasilinear diagnostics and saturation-model selection",
            "status": "closed"
            if ql_guardrails_passed and (ql_negative_closed or ql_candidate_promoted)
            else "open",
            "claim_level": (
                "validated_diagnostics_negative_absolute_flux_promotion"
                if ql_negative_closed
                else (
                    "scoped_candidate_model_selection_not_runtime_flux_predictor"
                    if ql_candidate_promoted
                    else "validated_diagnostics_negative_absolute_flux_promotion"
                )
            ),
            "primary_artifacts": [
                "docs/_static/quasilinear_validated_calibration_inputs.json",
                "docs/_static/quasilinear_stellarator_train_holdout_report.json",
                "docs/_static/quasilinear_saturation_rule_sweep.json",
                "docs/_static/quasilinear_shape_aware_saturation.json",
                "docs/_static/quasilinear_candidate_uncertainty.json",
                "docs/_static/quasilinear_dataset_sufficiency.json",
                "docs/_static/quasilinear_model_selection_status.json",
                "docs/_static/quasilinear_promotion_guardrails.json",
                "docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.json",
                "docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json",
            ],
            "key_metrics": {
                "validated_inputs_passed": ql_inputs_passed,
                "promotion_guardrails_passed": ql_guardrails_passed,
                "train_points": _count_points(ql_holdout, "train"),
                "holdout_points": _count_points(ql_holdout, "holdout"),
                "absolute_flux_promoted": ql_holdout_promoted,
                "holdout_mean_abs_relative_error": ql_holdout_mean,
                "simple_rule_promotion_passed": bool(
                    ql_sweep_gate.get("passed", False)
                ),
                "shape_aware_promotion_passed": bool(
                    ql_shape_gate.get("passed", False)
                ),
                "uq_candidate_promotion_passed": bool(ql_uq_gate.get("passed", False)),
                "dataset_sufficiency_promotion_passed": bool(
                    ql_dataset_gate.get("passed", False)
                ),
                "model_selection_status_passed": bool(
                    ql_model_status_gate.get("passed", False)
                ),
                "model_selection_candidate_mean_error": _finite_float(
                    (ql_model_status or {})
                    .get("metrics", {})
                    .get("candidate_mean_abs_relative_error")
                    if isinstance((ql_model_status or {}).get("metrics", {}), dict)
                    else None
                ),
                "model_selection_interval_coverage": _finite_float(
                    (ql_model_status or {})
                    .get("metrics", {})
                    .get("candidate_prediction_interval_coverage")
                    if isinstance((ql_model_status or {}).get("metrics", {}), dict)
                    else None
                ),
                "accepted_uq_candidates": ql_uq_gate.get("accepted_candidates", []),
                "dataset_current_total_cases": ql_dataset_requirements.get(
                    "current_total_cases"
                ),
                "dataset_min_total_cases": ql_dataset_requirements.get(
                    "min_total_electrostatic_cases"
                ),
                "dataset_current_train_geometries": ql_dataset_requirements.get(
                    "current_explicit_train_geometries"
                ),
                "dataset_min_train_geometries": ql_dataset_requirements.get(
                    "min_explicit_train_geometries"
                ),
                "dataset_blockers": ql_dataset_gate.get("blockers", []),
                "null_training_mean_error": _finite_float(
                    ql_uq_gate.get("null_training_mean_mean_abs_relative_error")
                ),
                "dshape_replicate_passed": bool((dshape_replicate or {}).get("passed", False)),
                "dshape_replicate_mean_rel_spread": _finite_float(
                    dshape_replicate_stats.get("mean_rel_spread")
                ),
                "dshape_replicate_combined_sem_rel": _finite_float(
                    dshape_replicate_stats.get("combined_sem_rel")
                ),
                "circular_replicate_passed": bool((circular_replicate or {}).get("passed", False)),
                "circular_replicate_mean_rel_spread": _finite_float(
                    circular_replicate_stats.get("mean_rel_spread")
                ),
                "circular_replicate_combined_sem_rel": _finite_float(
                    circular_replicate_stats.get("combined_sem_rel")
                ),
            },
            "next_action": (
                "Document the accepted richer candidate with scoped wording and keep the legacy one-scalar and "
                "shape-only diagnostics as negative transfer controls."
                if ql_guardrails_passed and ql_candidate_promoted
                else "Use these as manuscript-grade diagnostics and negative model-selection results; do not expose an "
                "absolute-flux runtime model until a future candidate beats the null baseline on held-out nonlinear data."
            ),
        },
        {
            "lane": "VMEC/Boozer differentiable geometry parity",
            "status": "closed"
            if geom_bridge_passed and geom_matrix_passed
            else "partial",
            "claim_level": "zero_beta_equal_arc_geometry_bridge_closed",
            "primary_artifacts": [
                "docs/_static/differentiable_geometry_bridge.json",
                "docs/_static/vmec_boozer_parity_matrix.json",
            ],
            "key_metrics": {
                "bridge_ad_fd_available": geom_bridge_passed,
                "matrix_all_equal_arc_passed": geom_matrix_passed,
                "matrix_n_cases": matrix_summary.get("n_cases"),
                "minimum_boozer_mode_count": (geom_matrix or {}).get(
                    "minimum_boozer_mode_count"
                ),
            },
            "next_action": (
                "Keep finite-beta pressure-correction drift parity as a future extension unless the manuscript claims "
                "finite-beta transport-gradient optimization."
            ),
        },
        {
            "lane": "Reduced differentiable stellarator ITG optimization",
            "status": "closed"
            if opt_reduced_objectives_passed and opt_uq_passed
            else "open",
            "claim_level": "reduced_objective_optimization_closed_not_full_production_vmec_gk",
            "primary_artifacts": [
                "docs/_static/stellarator_itg_optimization_uq.json",
                "docs/_static/stellarator_itg_optimization_uq.png",
                "docs/_static/production_nonlinear_optimization_guard.json",
                "docs/_static/production_nonlinear_optimization_guard.png",
                "docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.json",
                "docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.png",
                "docs/_static/qa_no_ess_reference_replicates/qa_no_ess_reference_t700_ensemble_gate.json",
                "docs/_static/qa_no_ess_reference_replicates/qa_no_ess_reference_t700_ensemble_gate.png",
                "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json",
                "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.png",
            ],
            "supporting_artifacts": [
                "docs/_static/stellarator_itg_optimization_comparison.json",
                "docs/_static/stellarator_itg_optimization_comparison.png",
            ],
            "key_metrics": {
                **opt_reductions,
                "all_objective_gradient_gates_passed": opt_reduced_objectives_passed,
                "weighted_residual_uq_gate_passed": opt_uq_passed,
                "production_nonlinear_optimization_guard_safe": (
                    production_nonlinear_optimization_safe_to_release
                ),
                "production_nonlinear_optimization_promoted": (
                    production_nonlinear_optimization_promoted
                ),
                "production_nonlinear_replicated_holdout_ensembles": (
                    production_nl_guard_summary.get(
                        "qualifying_replicated_holdout_ensembles"
                    )
                ),
                "production_nonlinear_optimized_equilibrium_ensembles": (
                    production_nl_guard_summary.get(
                        "qualifying_optimized_equilibrium_ensembles"
                    )
                ),
                "matched_qa_no_ess_to_optimized_audit_passed": (
                    matched_baseline_optimized_audit_passed
                ),
                "matched_qa_no_ess_relative_reduction": (
                    baseline_optimized_comparison.get("relative_reduction")
                ),
                "matched_qa_no_ess_uncertainty_separation_sigma": (
                    baseline_optimized_comparison.get(
                        "uncertainty_separation_sigma"
                    )
                ),
            },
            "next_action": (
                "Use the UQ, production-guard, replicated-ensemble, and matched-audit artifacts as the current "
                "optimization-evidence figures. Do not use the reduced synthetic surface comparison as the solved-geometry "
                "optimization figure. The matched QA no-ESS to optimized QA/ESS audit is closed, but nonlinear-window "
                "VMEC/Boozer/GK gradients and matched audits across broader geometry families remain future requirements "
                "before claiming broad end-to-end stellarator heat-flux optimization."
            ),
        },
        {
            "lane": "Production solver-objective geometry gradients",
            "status": solver_gradient_status,
            "claim_level": "required_for_full_end_to_end_stellarator_optimization_claim",
            "primary_artifacts": [
                path
                for path, payload in (
                    ("docs/_static/solver_objective_gradient_gate.json", solver_grad),
                    (
                        "docs/_static/vmec_boozer_solver_frequency_gradient_gate.json",
                        vmec_solver_grad,
                    ),
                    (
                        "docs/_static/vmec_boozer_quasilinear_gradient_gate.json",
                        vmec_ql_grad,
                    ),
                    (
                        "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json",
                        vmec_nl_window_grad,
                    ),
                    (
                        "docs/_static/vmec_boozer_li383_nonlinear_window_gradient_gate.json",
                        vmec_li383_nl_window_grad,
                    ),
                    (
                        "docs/_static/vmec_boozer_gradient_holdout_matrix.json",
                        vmec_gradient_matrix,
                    ),
                    ("docs/_static/nonlinear_window_fd_audit.json", nonlinear_fd_audit),
                    ("docs/_static/nonlinear_window_fd_audit.png", nonlinear_fd_audit),
                    (
                        "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json",
                        vmec_nonlinear_fd_audit,
                    ),
                    (
                        "docs/_static/vmec_boozer_nonlinear_window_fd_audit.png",
                        vmec_nonlinear_fd_audit,
                    ),
                    (
                        "docs/_static/production_nonlinear_optimization_guard.json",
                        production_nl_guard,
                    ),
                    (
                        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json",
                        nonlinear_control_mean_gate,
                    ),
                    (
                        "docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.json",
                        production_nl_guard,
                    ),
                )
                if payload
            ],
            "key_metrics": {
                "source_scope": solver_gradient_source,
                "solver_ready_gradient_gate": solver_gradient_passed,
                "full_vmec_boozer_frequency_gradient_gate": solver_gradient_full_vmec_frequency,
                "full_vmec_boozer_quasilinear_gradient_gate": solver_gradient_full_vmec_quasilinear,
                "full_vmec_boozer_reduced_nonlinear_window_gradient_gate": solver_gradient_reduced_nonlinear_window,
                "multi_equilibrium_reduced_nonlinear_window_gradient_gate": (
                    solver_gradient_reduced_nonlinear_window_multi_equilibrium
                ),
                "multi_equilibrium_gradient_holdout_matrix": solver_gradient_multi_equilibrium,
                "multi_equilibrium_gradient_cases": gradient_matrix_summary.get(
                    "n_cases"
                ),
                "multi_equilibrium_gradient_max_rel_error": _finite_float(
                    gradient_matrix_summary.get("max_relative_error")
                ),
                "linear_growth_gradient_gate": bool(
                    (solver_grad or {}).get("linear_growth_gradient_gate", False)
                ),
                "quasilinear_weight_gradient_gate": bool(
                    (solver_grad or {}).get("quasilinear_weight_gradient_gate", False)
                ),
                "vmec_boozer_frequency_rel_error": _finite_float(
                    (vmec_solver_grad or {})
                    .get("eigenpair_gate", {})
                    .get("max_rel_error")
                ),
                "vmec_boozer_quasilinear_rel_error": _finite_float(
                    (vmec_ql_grad or {}).get("eigenpair_gate", {}).get("max_rel_error")
                ),
                "vmec_boozer_reduced_nonlinear_window_rel_error": _finite_float(
                    (vmec_nl_window_grad or {})
                    .get("eigenpair_gate", {})
                    .get("max_rel_error")
                ),
                "vmec_boozer_li383_reduced_nonlinear_window_rel_error": _finite_float(
                    (vmec_li383_nl_window_grad or {})
                    .get("eigenpair_gate", {})
                    .get("max_rel_error")
                ),
                "reduced_nonlinear_window_gradient_gate": bool(
                    (vmec_nl_window_grad or solver_grad or {}).get(
                        "nonlinear_window_gradient_gate", False
                    )
                ),
                "startup_nonlinear_plumbing_fd_path_gate": startup_nonlinear_plumbing_fd_path_gate,
                "startup_nonlinear_plumbing_response_fraction": _finite_float(
                    nonlinear_fd_metrics.get("response_fraction")
                ),
                "startup_nonlinear_plumbing_repeatability_rel_error": _finite_float(
                    nonlinear_fd_metrics.get("repeatability_relative_error")
                ),
                "startup_nonlinear_plumbing_max_window_cv": _finite_float(
                    nonlinear_fd_metrics.get("max_window_cv")
                ),
                "startup_nonlinear_plumbing_max_window_trend": _finite_float(
                    nonlinear_fd_metrics.get("max_window_trend")
                ),
                "nonlinear_transport_average_gate": nonlinear_transport_average_gate,
                "production_nonlinear_observable_fd_path_gate": production_nonlinear_observable_fd_path_gate,
                "vmec_boozer_startup_nonlinear_plumbing_fd_path_gate": (
                    vmec_boozer_startup_nonlinear_plumbing_fd_path_gate
                ),
                "vmec_boozer_startup_nonlinear_response_fraction": _finite_float(
                    vmec_nonlinear_fd_metrics.get("response_fraction")
                ),
                "vmec_boozer_startup_nonlinear_derivative_asymmetry": _finite_float(
                    vmec_nonlinear_fd_metrics.get("derivative_asymmetry")
                ),
                "vmec_boozer_nonlinear_transport_average_gate": vmec_boozer_nonlinear_transport_average_gate,
                "vmec_boozer_production_nonlinear_observable_fd_path_gate": (
                    vmec_boozer_production_nonlinear_observable_fd_path_gate
                ),
                "production_nonlinear_window_gradient_gate": False,
                "production_nonlinear_optimization_guard_safe": (
                    production_nonlinear_optimization_safe_to_release
                ),
                "production_nonlinear_optimization_promoted": (
                    production_nonlinear_optimization_promoted
                ),
                "optimized_equilibrium_replicated_transport_ensembles": (
                    production_nl_guard_summary.get(
                        "qualifying_optimized_equilibrium_ensembles"
                    )
                ),
                "variance_reduced_nonlinear_gradient_control_mean_passed": bool(
                    (nonlinear_control_mean_gate or {}).get("passed", False)
                ),
                "variance_reduced_nonlinear_gradient_common_pairs": (
                    nonlinear_control_mean_summary.get("common_pair_count")
                ),
                "variance_reduced_nonlinear_gradient_uncertainty_rel": _finite_float(
                    nonlinear_control_mean_summary.get("combined_response_uncertainty_rel")
                ),
                "variance_reduced_nonlinear_gradient_response_mean": _finite_float(
                    nonlinear_control_mean_summary.get("independent_response_mean")
                ),
            },
            "next_action": (
                "Full VMEC/Boozer eigenfrequency, quasilinear, and multi-equilibrium reduced nonlinear-window "
                "estimator gradients are closed. The compact nonlinear FD audits are tracked only as startup "
                "plumbing checks; they do not validate transport heat-flux averages. The selected optimized-equilibrium "
                "long-window audit is closed; VMEC/Boozer turbulence gradients, local-gradient conditioning, and "
                "broader multi-surface baseline-to-optimized audits remain required before full nonlinear stellarator-"
                "optimization claims."
            ),
        },
        {
            "lane": "Profiler-backed nonlinear performance claims",
            "status": (
                "closed"
                if release_performance_closed
                else (
                    "partial"
                    if bool((profile or {}).get("identity_gate_pass", False))
                    else "open"
                )
            ),
            "claim_level": (
                "release_performance_artifacts_closed_no_broad_speedup_claim"
                if release_performance_closed
                else "identity_gate_present_no_new_speedup_claim"
            ),
            "primary_artifacts": [
                "docs/_static/nonlinear_sharding_profile_office_gpu.json",
                "docs/_static/nonlinear_rhs_profile.json",
                "docs/_static/nonlinear_rhs_profile_miller.json",
                "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json",
                "docs/_static/full_nonlinear_rhs_trace_summary.json",
                "docs/_static/full_nonlinear_rhs_trace_gpu_summary.json",
            ],
            "key_metrics": {
                "release_performance_gate": release_performance_closed,
                "identity_gate_pass": bool(
                    (profile or {}).get("identity_gate_pass", False)
                ),
                "engineering_speedup": _finite_float(
                    (profile or {}).get("engineering_speedup")
                ),
                "best_identity_candidate": profile_best.get("spec"),
                "best_identity_candidate_speedup": _finite_float(
                    profile_best.get("engineering_speedup_median")
                ),
                "rhs_fastest_full_label": (rhs_profile or {}).get(
                    "fastest_full_rhs_label"
                ),
                "rhs_cpu_full_grid_over_spectral": _finite_float(
                    rhs_cpu.get("full_rhs_grid_over_spectral")
                ),
                "rhs_gpu_full_grid_over_spectral": _finite_float(
                    rhs_gpu.get("full_rhs_grid_over_spectral")
                ),
                "rhs_cpu_bracket_grid_over_spectral": _finite_float(
                    rhs_cpu.get("nonlinear_bracket_grid_over_spectral")
                ),
                "rhs_gpu_bracket_grid_over_spectral": _finite_float(
                    rhs_gpu.get("nonlinear_bracket_grid_over_spectral")
                ),
                "miller_cpu_grid_full_rhs": miller_cpu_grid_full,
                "miller_gpu_grid_full_rhs": miller_gpu_grid_full,
                "miller_gpu_spectral_full_rhs": miller_gpu_spectral_full,
                "w7x_cpu_full_rhs": w7x_cpu_full,
                "w7x_gpu_full_rhs": w7x_gpu_full,
                "hsx_cpu_full_rhs": hsx_cpu_full,
                "hsx_gpu_full_rhs": hsx_gpu_full,
                "full_trace_cpu_warm_seconds": full_rhs_cpu_warm,
                "full_trace_gpu_warm_seconds": full_rhs_gpu_warm,
                "full_trace_gpu_transpose_count": full_rhs_gpu_transposes,
            },
            "next_action": (
                "Release performance evidence is closed for scoped CPU/GPU profiler artifacts; keep broad "
                "production nonlinear speedup and domain-decomposition claims for the next manuscript/science lane."
                if release_performance_closed
                else "Keep runtime claims conservative until fresh CPU/GPU profiler artifacts support a speedup."
            ),
        },
        {
            "lane": "W7-X zonal recurrence/damping",
            "status": "deferred",
            "claim_level": "deferred_out_of_current_manuscript_scope",
            "primary_artifacts": ["docs/_static/w7x_zonal_reference_compare.json"],
            "key_metrics": {},
            "next_action": "Post-manuscript lane.",
        },
        {
            "lane": "TEM / kinetic-electron stellarator extension",
            "status": "deferred",
            "claim_level": "deferred_out_of_current_manuscript_scope",
            "primary_artifacts": ["docs/_static/w7x_tem_extension_status.json"],
            "key_metrics": {},
            "next_action": "Post-manuscript lane.",
        },
    ]

    active = [lane for lane in lanes if str(lane["status"]) != "deferred"]
    return {
        "kind": "manuscript_readiness_status",
        "claim_scope": (
            "current_manuscript_excludes_w7x_zonal_recurrence_and_tem_kinetic_electron_extension; "
            "quasilinear absolute-flux models are validated as diagnostics/model-selection results only"
        ),
        "status_order": STATUS_ORDER,
        "lanes": lanes,
        "summary": {
            "n_lanes": len(lanes),
            "n_active": len(active),
            "n_closed": sum(1 for lane in lanes if str(lane["status"]) == "closed"),
            "n_partial": sum(1 for lane in lanes if str(lane["status"]) == "partial"),
            "n_open": sum(1 for lane in lanes if str(lane["status"]) == "open"),
            "n_deferred": sum(1 for lane in lanes if str(lane["status"]) == "deferred"),
            "n_blocked": sum(1 for lane in lanes if str(lane["status"]) == "blocked"),
            "active_fraction_closed": (
                sum(1 for lane in active if str(lane["status"]) == "closed")
                / len(active)
                if active
                else 0.0
            ),
        },
    }


def write_manuscript_readiness_artifacts(
    payload: dict[str, Any], *, out: str | Path = DEFAULT_OUT
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the manuscript readiness payload."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fieldnames = ["lane", "status", "claim_level", "primary_artifacts", "next_action"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for lane in payload["lanes"]:
            writer.writerow(
                {
                    "lane": lane["lane"],
                    "status": lane["status"],
                    "claim_level": lane["claim_level"],
                    "primary_artifacts": ";".join(lane["primary_artifacts"]),
                    "next_action": lane["next_action"],
                }
            )

    set_plot_style()
    lanes = list(payload["lanes"])
    y = np.arange(len(lanes))
    values = [STATUS_ORDER[str(lane["status"])] for lane in lanes]
    bar_values = [max(0.12, float(value)) for value in values]
    colors = [STATUS_COLORS[str(lane["status"])] for lane in lanes]
    labels = [textwrap.fill(str(lane["lane"]), width=34) for lane in lanes]

    fig, ax = plt.subplots(figsize=(12.0, 6.1))
    ax.barh(y, bar_values, color=colors, edgecolor="#333333", alpha=0.95)
    ax.scatter(
        values, y, s=52, color=colors, edgecolor="#222222", linewidth=0.7, zorder=3
    )
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 4.2)
    ax.set_xticks([0, 1, 2, 3, 4], ["closed", "partial", "open", "deferred", "blocked"])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Manuscript readiness: scoped physics and optimization lanes")
    for yi, lane, value in zip(y, lanes, values, strict=True):
        metric = ""
        km = lane.get("key_metrics", {})
        if str(lane["lane"]).startswith("Quasilinear"):
            replicated = sum(
                1
                for key in ("dshape_replicate_passed", "circular_replicate_passed")
                if km.get(key)
            )
            metric = (
                f"dataset: {km.get('dataset_current_total_cases')}/{km.get('dataset_min_total_cases')}; "
                f"holdouts: {km.get('holdout_points')}; "
                f"replicated: {replicated}; "
                f"absolute flux promoted: {km.get('absolute_flux_promoted')}"
            )
        elif str(lane["lane"]).startswith("VMEC/Boozer"):
            metric = f"mode floor: {km.get('minimum_boozer_mode_count')}; cases: {km.get('matrix_n_cases')}"
        elif str(lane["lane"]).startswith("Reduced"):
            worst = km.get("worst_reduction_factor")
            metric = (
                "gradient gates passed"
                if worst is None
                else f"worst final/initial: {float(worst):.2f}"
            )
        elif str(lane["lane"]).startswith("Production"):
            max_err = km.get("multi_equilibrium_gradient_max_rel_error")
            err_text = "n/a" if max_err is None else f"{float(max_err):.1e}"
            cv_pairs = km.get("variance_reduced_nonlinear_gradient_common_pairs")
            cv_uncertainty = km.get("variance_reduced_nonlinear_gradient_uncertainty_rel")
            cv_text = (
                "CV gate: n/a"
                if cv_pairs is None or cv_uncertainty is None
                else f"CV gate: {cv_pairs} pairs, u={float(cv_uncertainty):.2f}"
            )
            metric = (
                f"solver-ready: {km.get('solver_ready_gradient_gate')}; "
                f"holdouts: {km.get('multi_equilibrium_gradient_cases')}; "
                f"max err: {err_text}; "
                f"{cv_text}"
            )
        elif str(lane["lane"]).startswith("Profiler"):
            speed = km.get("engineering_speedup")
            best = km.get("best_identity_candidate")
            best_speed = km.get("best_identity_candidate_speedup")
            rhs_gpu_speed = km.get("rhs_gpu_full_grid_over_spectral")
            primary = (
                "primary: n/a" if speed is None else f"primary: {float(speed):.2f}x"
            )
            best_text = (
                "" if best_speed is None else f"; best {best}: {float(best_speed):.2f}x"
            )
            rhs_text = (
                ""
                if rhs_gpu_speed is None
                else f"; RHS GPU split: {float(rhs_gpu_speed):.2f}x"
            )
            metric = primary + best_text + rhs_text
        text_x = max(float(value) + 0.06, 0.18)
        ax.text(
            min(text_x, 4.05), float(yi), metric, va="center", ha="left", fontsize=8.0
        )

    summary = payload.get("summary", {})
    caption = (
        f"Active closed fraction: {float(summary.get('active_fraction_closed', 0.0)):.0%}. "
        "Deferred lanes are explicitly outside the current manuscript scope; open lanes are next priorities."
    )
    fig.text(0.5, 0.025, caption, fontsize=8.3, color="#333333", ha="center")
    fig.subplots_adjust(left=0.34, right=0.97, top=0.90, bottom=0.14)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_manuscript_readiness_payload(Path(args.root))
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_manuscript_readiness_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
