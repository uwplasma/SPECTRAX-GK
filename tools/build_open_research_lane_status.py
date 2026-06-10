#!/usr/bin/env python3
"""Build a machine-readable status summary for open research validation lanes."""

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

from spectraxgk.plotting import set_plot_style  # type: ignore[import-untyped]  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "_static" / "open_research_lane_status.png"

STATUS_ORDER = {"closed": 0, "partial": 1, "open": 2, "blocked": 3}
STATUS_COLORS = {
    "closed": "#2a9d8f",
    "partial": "#e9c46a",
    "open": "#f4a261",
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


def _gate_failures(gate_report: dict[str, Any] | None) -> list[str]:
    if not gate_report:
        return []
    failures: list[str] = []
    for gate in gate_report.get("gates", []):
        if isinstance(gate, dict) and not bool(gate.get("passed", False)):
            failures.append(str(gate.get("metric", "unknown")))
    return failures


def _best_recurrence_candidate(payload: dict[str, Any] | None) -> dict[str, Any]:
    rows = [] if payload is None else payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return {"label": None, "mean_abs_error": None, "tail_std_ratio": None, "hermite_tail": None}
    finite_rows = [row for row in rows if isinstance(row, dict) and _finite_float(row.get("mean_abs_error")) is not None]
    if not finite_rows:
        return {"label": None, "mean_abs_error": None, "tail_std_ratio": None, "hermite_tail": None}
    best = min(finite_rows, key=lambda row: float(row["mean_abs_error"]))
    tail_std = _finite_float(best.get("tail_std"), 0.0) or 0.0
    ref_tail_std = _finite_float(best.get("reference_tail_std"), 0.0) or 0.0
    ratio = None if ref_tail_std <= 0.0 else tail_std / ref_tail_std
    return {
        "label": str(best.get("label", "unknown")),
        "mean_abs_error": _finite_float(best.get("mean_abs_error")),
        "tail_std_ratio": ratio,
        "hermite_tail": _finite_float(best.get("hermite_tail_at_tmax")),
        "source_path": best.get("source_path"),
    }


def _highest_moment_candidate(payload: dict[str, Any] | None) -> dict[str, Any]:
    rows = [] if payload is None else payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return {"label": None, "mean_abs_error": None, "tail_std_ratio": None, "hermite_tail": None}
    finite_rows = [row for row in rows if isinstance(row, dict) and _finite_float(row.get("mean_abs_error")) is not None]
    if not finite_rows:
        return {"label": None, "mean_abs_error": None, "tail_std_ratio": None, "hermite_tail": None}
    best = max(
        finite_rows,
        key=lambda row: (
            int(_finite_float(row.get("Nl"), 0.0) or 0.0),
            int(_finite_float(row.get("Nm"), 0.0) or 0.0),
        ),
    )
    tail_std = _finite_float(best.get("tail_std"), 0.0) or 0.0
    ref_tail_std = _finite_float(best.get("reference_tail_std"), 0.0) or 0.0
    ratio = None if ref_tail_std <= 0.0 else tail_std / ref_tail_std
    return {
        "label": str(best.get("label", "unknown")),
        "Nl": int(_finite_float(best.get("Nl"), 0.0) or 0.0),
        "Nm": int(_finite_float(best.get("Nm"), 0.0) or 0.0),
        "mean_abs_error": _finite_float(best.get("mean_abs_error")),
        "tail_std_ratio": ratio,
        "hermite_tail": _finite_float(best.get("hermite_tail_at_tmax")),
        "laguerre_tail": _finite_float(best.get("laguerre_tail_at_tmax")),
        "source_path": best.get("source_path"),
    }


def _best_hypercollision_probe(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    rows = [] if payload is None else payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return None
    finite_rows = [row for row in rows if isinstance(row, dict) and _finite_float(row.get("mean_abs_error")) is not None]
    if not finite_rows:
        return None
    best = min(finite_rows, key=lambda row: float(row["mean_abs_error"]))
    tail_std = _finite_float(best.get("tail_std"), 0.0) or 0.0
    ref_tail_std = _finite_float(best.get("reference_tail_std"), 0.0) or 0.0
    ratio = None if ref_tail_std <= 0.0 else tail_std / ref_tail_std
    validation_status = None if payload is None else payload.get("validation_status")
    return {
        "label": str(best.get("label", "unknown")),
        "mean_abs_error": _finite_float(best.get("mean_abs_error")),
        "tail_std_ratio": ratio,
        "hermite_tail": _finite_float(best.get("hermite_tail_at_tmax")),
        "free_energy_ratio": _finite_float(best.get("free_energy_at_tmax_over_initial")),
        "source_path": best.get("source_path"),
        "validation_status": validation_status,
    }


def _holdout_counts(report: dict[str, Any] | None) -> tuple[int, int, list[str]]:
    if report is None:
        return 0, 0, []
    points = report.get("points", [])
    if not isinstance(points, list):
        return 0, 0, []
    train = 0
    holdout = 0
    names: list[str] = []
    for point in points:
        if not isinstance(point, dict):
            continue
        split = str(point.get("split", ""))
        if split == "train":
            train += 1
        if split == "holdout":
            holdout += 1
            names.append(str(point.get("case", "unknown")))
    return train, holdout, names


def _profile_seconds(payload: dict[str, Any] | None, label: str, kernel: str) -> float | None:
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


def build_status_payload(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return a JSON-ready status payload for active research lanes."""

    root = Path(root)
    zonal_ref = _read_json(root, "docs/_static/w7x_zonal_reference_compare.json")
    zonal_recurrence = _read_json(root, "docs/_static/w7x_zonal_recurrence_sweep_kx070.json")
    zonal_hypercollision = _read_json(root, "docs/_static/w7x_zonal_hypercollision_probe_kx070.json")
    zonal_mixed_lm_resolution = _read_json(root, "docs/_static/w7x_zonal_mixedlm_resolution_kx070.json")
    fluct = _read_json(root, "docs/_static/w7x_fluctuation_spectrum_panel.json")
    ql_inputs = _read_json(root, "docs/_static/quasilinear_validated_calibration_inputs.json")
    ql_report = _read_json(root, "docs/_static/quasilinear_stellarator_train_holdout_report.json")
    ql_uq = _read_json(root, "docs/_static/quasilinear_candidate_uncertainty.json")
    ql_dataset = _read_json(root, "docs/_static/quasilinear_dataset_sufficiency.json")
    ql_model_status = _read_json(root, "docs/_static/quasilinear_model_selection_status.json")
    production_nl_guard = _read_json(root, "docs/_static/production_nonlinear_optimization_guard.json")
    baseline_optimized_audit = _read_json(root, "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json")
    nonlinear_control_mean_gate = _read_json(
        root,
        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json",
    )
    circular_t250_gate = _read_json(root, "docs/_static/external_vmec_circular_t250_high_grid_convergence_gate.json")
    circular_t700_replicate = _read_json(
        root,
        "docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json",
    )
    dshape_gate = _read_json(root, "docs/_static/external_vmec_dshape_t250_high_grid_convergence_gate.json")
    dshape_replicate = _read_json(
        root,
        "docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.json",
    )
    itermodel_gate = _read_json(root, "docs/_static/external_vmec_itermodel_t350_high_grid_convergence_gate.json")
    updown_gate = _read_json(root, "docs/_static/external_vmec_updown_asym_t450_high_grid_convergence_gate.json")
    qh_gate = _read_json(root, "docs/_static/external_vmec_qh_grid_convergence_gate.json")
    qh_high_gate = _read_json(root, "docs/_static/external_vmec_qh_high_grid_convergence_gate.json")
    cth_gate = _read_json(root, "docs/_static/external_vmec_cth_like_grid_convergence_gate.json")
    cth_high_grid_admission = _read_json(
        root,
        "docs/_static/external_vmec_cth_like_modified_high_grid_admission_gate.json",
    )
    geom = _read_json(root, "docs/_static/differentiable_geometry_bridge.json")
    geom_matrix = _read_json(root, "docs/_static/vmec_boozer_parity_matrix.json")
    profile = _read_json(root, "docs/_static/nonlinear_sharding_profile_office_gpu.json")
    rhs_profile = _read_json(root, "docs/_static/nonlinear_rhs_profile.json")
    rhs_miller = _read_json(root, "docs/_static/nonlinear_rhs_profile_miller.json")
    rhs_stellarator = _read_json(root, "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json")
    full_rhs_trace_cpu = _read_json(root, "docs/_static/full_nonlinear_rhs_trace_summary.json")
    full_rhs_trace_gpu = _read_json(root, "docs/_static/full_nonlinear_rhs_trace_gpu_summary.json")

    zonal_failures = _gate_failures(zonal_ref.get("gate_report") if zonal_ref else None)
    best_recurrence = _best_recurrence_candidate(zonal_recurrence)
    best_hypercollision = _best_hypercollision_probe(zonal_hypercollision)
    best_mixed_lm_resolution = _best_recurrence_candidate(zonal_mixed_lm_resolution)
    high_moment_mixed_lm_resolution = _highest_moment_candidate(zonal_mixed_lm_resolution)
    zonal_status = "closed" if zonal_ref and not zonal_failures and zonal_ref.get("validation_status") == "closed" else "open"
    w7x_tem_extension = _read_json(root, "docs/_static/w7x_tem_extension_status.json")
    w7x_tem_rows = (w7x_tem_extension or {}).get("rows", [])
    w7x_tem_open = [
        row.get("lane", "unknown")
        for row in w7x_tem_rows
        if isinstance(row, dict) and str(row.get("status")) == "open"
    ]

    train_count, holdout_count, holdout_names = _holdout_counts(ql_report)
    ql_report_passed = bool(ql_report.get("passed", False)) if ql_report else False
    ql_uq_gate = (ql_uq or {}).get("promotion_gate", {}) if isinstance((ql_uq or {}).get("promotion_gate", {}), dict) else {}
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
    ql_passed = bool(
        ql_report_passed
        or (
            ql_uq_gate.get("passed", False)
            and ql_dataset_gate.get("passed", False)
            and ql_model_status_gate.get("passed", False)
        )
    )
    circular_t250_passed = bool((circular_t250_gate or {}).get("gate_report", {}).get("passed", False))
    circular_t700_replicate_passed = bool((circular_t700_replicate or {}).get("passed", False))
    dshape_passed = bool((dshape_gate or {}).get("gate_report", {}).get("passed", False))
    dshape_replicate_passed = bool((dshape_replicate or {}).get("passed", False))
    production_nl_guard_summary = (
        (production_nl_guard or {}).get("summary", {})
        if isinstance((production_nl_guard or {}).get("summary", {}), dict)
        else {}
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
    itermodel_passed = bool((itermodel_gate or {}).get("gate_report", {}).get("passed", False))
    updown_passed = bool((updown_gate or {}).get("gate_report", {}).get("passed", False))
    qh_passed = bool((qh_gate or {}).get("gate_report", {}).get("passed", False))
    qh_high_passed = bool((qh_high_gate or {}).get("gate_report", {}).get("passed", False))
    cth_full_grid_passed = bool((cth_gate or {}).get("gate_report", {}).get("passed", False))
    cth_high_grid_admitted = bool(
        (cth_high_grid_admission or {}).get("promotion_gate", {}).get("passed", False)
    )
    cth_passed = cth_full_grid_passed or cth_high_grid_admitted

    geom_sensitivity = (geom or {}).get("sensitivity", {}) if isinstance((geom or {}).get("sensitivity", {}), dict) else {}
    geom_vmec_metric = (
        (geom or {}).get("vmec_jax_metric_tensor", {})
        if isinstance((geom or {}).get("vmec_jax_metric_tensor", {}), dict)
        else {}
    )
    geom_vmec_field_line = (
        (geom or {}).get("vmec_jax_field_line_tensor", {})
        if isinstance((geom or {}).get("vmec_jax_field_line_tensor", {}), dict)
        else {}
    )
    geom_vmec_flux_tube = (
        (geom or {}).get("vmec_jax_flux_tube", {})
        if isinstance((geom or {}).get("vmec_jax_flux_tube", {}), dict)
        else {}
    )
    geom_vmec_array_parity = (
        (geom or {}).get("vmec_jax_flux_tube_array_parity", {})
        if isinstance((geom or {}).get("vmec_jax_flux_tube_array_parity", {}), dict)
        else {}
    )
    geom_vmec_state = (
        (geom or {}).get("vmec_jax_boozer_flux_tube", {})
        if isinstance((geom or {}).get("vmec_jax_boozer_flux_tube", {}), dict)
        else {}
    )
    geom_vmec_flux_tube_sensitivity = (
        geom_vmec_flux_tube.get("sensitivity", {})
        if isinstance(geom_vmec_flux_tube.get("sensitivity", {}), dict)
        else {}
    )
    geom_vmec_state_sensitivity = (
        geom_vmec_state.get("sensitivity", {}) if isinstance(geom_vmec_state.get("sensitivity", {}), dict) else {}
    )
    geom_inverse = (geom or {}).get("geometry_inverse_design_report", {})
    geom_uq = (geom or {}).get("uq", {})
    geom_max_abs = _finite_float(geom_sensitivity.get("max_abs_ad_fd_error"))
    geom_inverse_res = _finite_float(geom_inverse.get("final_residual_norm")) if isinstance(geom_inverse, dict) else None
    geom_rank = int(geom_uq.get("sensitivity_map_rank", 0)) if isinstance(geom_uq, dict) else 0
    geom_vmec_metric_abs = _finite_float(geom_vmec_metric.get("max_abs_ad_fd_error"))
    geom_vmec_metric_rel = _finite_float(geom_vmec_metric.get("max_rel_ad_fd_error"))
    geom_vmec_field_line_abs = _finite_float(geom_vmec_field_line.get("max_abs_ad_fd_error"))
    geom_vmec_field_line_rel = _finite_float(geom_vmec_field_line.get("max_rel_ad_fd_error"))
    geom_vmec_flux_tube_abs = _finite_float(geom_vmec_flux_tube_sensitivity.get("max_abs_ad_fd_error"))
    geom_vmec_flux_tube_rel = _finite_float(geom_vmec_flux_tube_sensitivity.get("max_rel_ad_fd_error"))
    geom_vmec_array_parity_worst = _finite_float(geom_vmec_array_parity.get("worst_core_normalized_max_abs"))
    geom_vmec_array_parity_passed = bool(geom_vmec_array_parity.get("production_parity_passed", False))
    geom_vmec_equal_arc_core_worst = _finite_float(
        geom_vmec_array_parity.get("equal_arc_core_worst_normalized_max_abs")
    )
    geom_vmec_equal_arc_core_scalar = _finite_float(geom_vmec_array_parity.get("equal_arc_core_worst_scalar_rel"))
    geom_vmec_equal_arc_derivative_worst = _finite_float(
        geom_vmec_array_parity.get("equal_arc_derivative_worst_normalized_max_abs")
    )
    geom_vmec_equal_arc_metric_worst = _finite_float(
        geom_vmec_array_parity.get("equal_arc_metric_worst_normalized_max_abs")
    )
    geom_vmec_equal_arc_drift_worst = _finite_float(
        geom_vmec_array_parity.get("equal_arc_drift_worst_normalized_max_abs")
    )
    geom_vmec_equal_arc_core_passed = bool(geom_vmec_array_parity.get("equal_arc_core_passed", False))
    geom_vmec_equal_arc_derivative_passed = bool(geom_vmec_array_parity.get("equal_arc_derivative_passed", False))
    geom_vmec_equal_arc_metric_passed = bool(geom_vmec_array_parity.get("equal_arc_metric_passed", False))
    geom_vmec_equal_arc_drift_passed = bool(geom_vmec_array_parity.get("equal_arc_drift_passed", False))
    geom_vmec_state_abs = _finite_float(geom_vmec_state_sensitivity.get("max_abs_ad_fd_error"))
    geom_vmec_state_rel = _finite_float(geom_vmec_state_sensitivity.get("max_rel_ad_fd_error"))
    geom_matrix_summary = geom_matrix.get("summary", {}) if isinstance(geom_matrix, dict) else {}
    geom_matrix_rows = geom_matrix.get("rows", []) if isinstance(geom_matrix, dict) else []
    geom_matrix_all_equal_arc_passed = bool(geom_matrix_summary.get("all_equal_arc_passed", False))
    geom_matrix_min_mode = int(geom_matrix.get("minimum_boozer_mode_count", 0)) if isinstance(geom_matrix, dict) else 0
    geom_matrix_limiting_drift = None
    if isinstance(geom_matrix_rows, list):
        finite_drift_rows = [
            row
            for row in geom_matrix_rows
            if isinstance(row, dict)
            and _finite_float(row.get("equal_arc_drift_worst_normalized_max_abs")) is not None
        ]
        if finite_drift_rows:
            limiting = max(
                finite_drift_rows,
                key=lambda row: float(row["equal_arc_drift_worst_normalized_max_abs"]),
            )
            geom_matrix_limiting_drift = {
                "case_name": limiting.get("case_name"),
                "value": _finite_float(limiting.get("equal_arc_drift_worst_normalized_max_abs")),
            }

    profile_identity = bool((profile or {}).get("identity_gate_pass", False))
    profile_speedup = _finite_float((profile or {}).get("engineering_speedup"))
    profile_best = (
        (profile or {}).get("best_identity_preserving_candidate", {})
        if isinstance((profile or {}).get("best_identity_preserving_candidate", {}), dict)
        else {}
    )
    rhs_speedups = (
        (rhs_profile or {}).get("spectral_speedups", {})
        if isinstance((rhs_profile or {}).get("spectral_speedups", {}), dict)
        else {}
    )
    rhs_cpu = rhs_speedups.get("cpu", {}) if isinstance(rhs_speedups.get("cpu", {}), dict) else {}
    rhs_gpu = rhs_speedups.get("gpu", {}) if isinstance(rhs_speedups.get("gpu", {}), dict) else {}
    miller_cpu_grid_full = _profile_seconds(rhs_miller, "CPU grid", "full_rhs")
    miller_gpu_grid_full = _profile_seconds(rhs_miller, "GPU grid", "full_rhs")
    miller_gpu_spectral_full = _profile_seconds(rhs_miller, "GPU spectral", "full_rhs")
    w7x_cpu_full = _profile_seconds(rhs_stellarator, "W7-X CPU", "full_rhs")
    w7x_gpu_full = _profile_seconds(rhs_stellarator, "W7-X GPU", "full_rhs")
    hsx_cpu_full = _profile_seconds(rhs_stellarator, "HSX CPU", "full_rhs")
    hsx_gpu_full = _profile_seconds(rhs_stellarator, "HSX GPU", "full_rhs")
    full_rhs_cpu_warm = _finite_float((full_rhs_trace_cpu or {}).get("warm_seconds"))
    full_rhs_gpu_warm = _finite_float((full_rhs_trace_gpu or {}).get("warm_seconds"))
    release_performance_closed = bool(
        profile_identity
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
            "lane": "W7-X zonal long-window recurrence/damping",
            "status": zonal_status,
            "claim_level": "open_physical_closure_not_normalization",
            "primary_artifacts": [
                "docs/_static/w7x_zonal_reference_compare.json",
                "docs/_static/w7x_zonal_recurrence_sweep_kx070.json",
                "docs/_static/w7x_zonal_hypercollision_probe_kx070.json",
                "docs/_static/w7x_zonal_mixedlm_resolution_kx070.json",
            ],
            "key_metrics": {
                "failed_reference_gates": zonal_failures,
                "best_bounded_candidate": best_recurrence,
                "best_constant_hypercollision_probe": best_hypercollision,
                "best_mixed_lm_resolution_probe": best_mixed_lm_resolution,
                "stable_high_moment_mixed_lm_probe": high_moment_mixed_lm_resolution,
            },
            "next_action": (
                "Move beyond constant Hermite and mixed Laguerre-Hermite damping: test a closure/operator that improves "
                "trace error, late-window envelope, and moment-tail gates together without losing high-moment stability."
            ),
        },
        {
            "lane": "W7-X fluctuation spectrum and TEM/multi-flux extension",
            "status": "partial" if bool(fluct and fluct.get("source_gate_passed")) else "open",
            "claim_level": "validated_simulation_spectrum_tem_extension_open",
            "primary_artifacts": [
                "docs/_static/w7x_fluctuation_spectrum_panel.json",
                "docs/_static/w7x_tem_extension_status.json",
                "docs/_static/tem_branch_parity_audit.json",
                "docs/_static/tem_mismatch_table.csv",
            ],
            "key_metrics": {
                "time_samples": (fluct or {}).get("time_samples"),
                "time_window": [(fluct or {}).get("time_min"), (fluct or {}).get("time_max")],
                "dominant_phi_ky": (fluct or {}).get("dominant_phi_ky"),
                "dominant_heat_flux_ky": (fluct or {}).get("dominant_heat_flux_ky"),
                "open_extension_rows": w7x_tem_open,
            },
            "next_action": (
                "Add W7-X multi-alpha/multi-surface ITG and kinetic-electron density-gradient/TEM scans before "
                "broad stellarator-validation claims."
            ),
        },
        {
            "lane": "Nonlinear holdouts for quasilinear absolute-flux promotion",
            "status": "closed" if ql_passed else "open",
            "claim_level": (
                "diagnostic_calibration_dataset_not_absolute_flux"
                if not ql_passed
                else (
                    "calibrated_absolute_flux"
                    if ql_report_passed
                    else "scoped_candidate_model_selection_not_absolute_flux"
                )
            ),
            "primary_artifacts": [
                "docs/_static/quasilinear_validated_calibration_inputs.json",
                "docs/_static/quasilinear_stellarator_train_holdout_report.json",
                "docs/_static/quasilinear_candidate_uncertainty.json",
                "docs/_static/quasilinear_dataset_sufficiency.json",
                "docs/_static/quasilinear_model_selection_status.json",
                "docs/_static/external_vmec_circular_t250_high_grid_convergence_gate.json",
                "docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json",
                "docs/_static/external_vmec_dshape_t250_high_grid_convergence_gate.json",
                "docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.json",
                "docs/_static/production_nonlinear_optimization_guard.json",
                "docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.json",
                "docs/_static/qa_no_ess_reference_replicates/qa_no_ess_reference_t700_ensemble_gate.json",
                "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json",
                "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json",
                "docs/_static/external_vmec_itermodel_t350_high_grid_convergence_gate.json",
                "docs/_static/external_vmec_updown_asym_t450_high_grid_convergence_gate.json",
                "docs/_static/external_vmec_qh_grid_convergence_gate.json",
                "docs/_static/external_vmec_qh_high_grid_convergence_gate.json",
                "docs/_static/external_vmec_cth_like_grid_convergence_gate.json",
                "docs/_static/external_vmec_cth_like_modified_high_grid_admission_gate.json",
            ],
            "key_metrics": {
                "validated_inputs_passed": bool((ql_inputs or {}).get("passed", False)),
                "train_points": train_count,
                "holdout_points": holdout_count,
                "holdout_cases": holdout_names,
                "calibration_report_passed": ql_report_passed,
                "uq_candidate_promotion_passed": bool(ql_uq_gate.get("passed", False)),
                "uq_accepted_candidates": ql_uq_gate.get("accepted_candidates", []),
                "dataset_sufficiency_promotion_passed": bool(ql_dataset_gate.get("passed", False)),
                "model_selection_status_passed": bool(ql_model_status_gate.get("passed", False)),
                "model_selection_candidate_mean_error": _finite_float(
                    (ql_model_status or {})
                    .get("metrics", {})
                    .get("candidate_mean_abs_relative_error")
                    if isinstance((ql_model_status or {}).get("metrics", {}), dict)
                    else None
                ),
                "circular_external_vmec_t250_converged": circular_t250_passed,
                "circular_external_vmec_t700_replicated": circular_t700_replicate_passed,
                "circular_external_vmec_t700_replicate_mean_rel_spread": _finite_float(
                    (circular_t700_replicate or {}).get("statistics", {}).get("mean_rel_spread")
                    if isinstance((circular_t700_replicate or {}).get("statistics", {}), dict)
                    else None
                ),
                "circular_external_vmec_t700_replicate_sem_rel": _finite_float(
                    (circular_t700_replicate or {}).get("statistics", {}).get("combined_sem_rel")
                    if isinstance((circular_t700_replicate or {}).get("statistics", {}), dict)
                    else None
                ),
                "dshape_external_vmec_t250_converged": dshape_passed,
                "dshape_external_vmec_t250_replicated": dshape_replicate_passed,
                "dshape_external_vmec_t250_replicate_mean_rel_spread": _finite_float(
                    (dshape_replicate or {}).get("statistics", {}).get("mean_rel_spread")
                    if isinstance((dshape_replicate or {}).get("statistics", {}), dict)
                    else None
                ),
                "dshape_external_vmec_t250_replicate_sem_rel": _finite_float(
                    (dshape_replicate or {}).get("statistics", {}).get("combined_sem_rel")
                    if isinstance((dshape_replicate or {}).get("statistics", {}), dict)
                    else None
                ),
                "production_nonlinear_optimization_guard_safe": bool(
                    (production_nl_guard or {}).get("safe_to_release", False)
                ),
                "production_nonlinear_optimization_promoted": bool(
                    (production_nl_guard or {}).get(
                        "production_nonlinear_optimization_promoted", False
                    )
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
                "matched_qa_no_ess_to_optimized_audit_passed": bool(
                    (baseline_optimized_audit or {}).get("passed", False)
                ),
                "matched_qa_no_ess_relative_reduction": (
                    baseline_optimized_comparison.get("relative_reduction")
                ),
                "matched_qa_no_ess_uncertainty_separation_sigma": (
                    baseline_optimized_comparison.get(
                        "uncertainty_separation_sigma"
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
                "itermodel_external_vmec_t350_converged": itermodel_passed,
                "updown_asym_external_vmec_t450_converged": updown_passed,
                "qh_external_vmec_low_to_mid_grid_converged": qh_passed,
                "qh_external_vmec_mid_to_high_grid_converged": qh_high_passed,
                "cth_like_external_vmec_full_grid_converged": cth_full_grid_passed,
                "cth_like_external_vmec_high_grid_admitted": cth_high_grid_admitted,
                "cth_like_external_vmec_converged": cth_passed,
            },
            "next_action": (
                "Document the accepted richer candidate and matched QA no-ESS to optimized QA/ESS audit with scoped "
                "wording; keep QH excluded until its common-window and grid-refinement gates pass, and keep "
                "CTH-like scoped to high-grid admission rather than full n48/n64/n80 convergence."
                if ql_passed
                else "Use the admitted D-shaped, circular, ITERModel, up-down asymmetric, and high-grid CTH-like "
                "external-VMEC holdouts as negative transfer constraints while developing richer saturation models; "
                "keep QH excluded until its common-window and grid-refinement gates pass."
            ),
        },
        {
            "lane": "vmec_jax / booz_xform_jax differentiable geometry bridge",
            "status": "partial" if geom_max_abs is not None and geom_rank >= 2 else "open",
            "claim_level": "contract_gradient_gate_not_full_stellarator_optimization",
            "primary_artifacts": [
                "docs/_static/differentiable_geometry_bridge.json",
                "docs/_static/vmec_boozer_parity_matrix.json",
            ],
            "key_metrics": {
                "max_abs_ad_fd_error": geom_max_abs,
                "vmec_metric_tensor_max_abs_ad_fd_error": geom_vmec_metric_abs,
                "vmec_metric_tensor_max_rel_ad_fd_error": geom_vmec_metric_rel,
                "vmec_field_line_tensor_max_abs_ad_fd_error": geom_vmec_field_line_abs,
                "vmec_field_line_tensor_max_rel_ad_fd_error": geom_vmec_field_line_rel,
                "vmec_tensor_flux_tube_max_abs_ad_fd_error": geom_vmec_flux_tube_abs,
                "vmec_tensor_flux_tube_max_rel_ad_fd_error": geom_vmec_flux_tube_rel,
                "vmec_tensor_vs_eik_array_parity_worst_core_norm": geom_vmec_array_parity_worst,
                "vmec_tensor_vs_eik_array_parity_passed": geom_vmec_array_parity_passed,
                "vmec_boozer_equal_arc_core_worst_norm": geom_vmec_equal_arc_core_worst,
                "vmec_boozer_equal_arc_core_worst_scalar_rel": geom_vmec_equal_arc_core_scalar,
                "vmec_boozer_equal_arc_bgrad_worst_norm": geom_vmec_equal_arc_derivative_worst,
                "vmec_boozer_equal_arc_metric_worst_norm": geom_vmec_equal_arc_metric_worst,
                "vmec_boozer_equal_arc_drift_worst_norm": geom_vmec_equal_arc_drift_worst,
                "vmec_boozer_equal_arc_core_passed": geom_vmec_equal_arc_core_passed,
                "vmec_boozer_equal_arc_bgrad_passed": geom_vmec_equal_arc_derivative_passed,
                "vmec_boozer_equal_arc_metric_passed": geom_vmec_equal_arc_metric_passed,
                "vmec_boozer_equal_arc_drift_passed": geom_vmec_equal_arc_drift_passed,
                "vmec_boozer_matrix_minimum_mode_count": geom_matrix_min_mode,
                "vmec_boozer_matrix_all_equal_arc_passed": geom_matrix_all_equal_arc_passed,
                "vmec_boozer_matrix_n_cases": geom_matrix_summary.get("n_cases"),
                "vmec_boozer_matrix_limiting_drift": geom_matrix_limiting_drift,
                "vmec_state_boozer_flux_tube_max_abs_ad_fd_error": geom_vmec_state_abs,
                "vmec_state_boozer_flux_tube_max_rel_ad_fd_error": geom_vmec_state_rel,
                "inverse_residual_norm": geom_inverse_res,
                "sensitivity_rank": geom_rank,
                "vmec_jax_available": (geom or {}).get("backend_info", {}).get("vmec_jax_available"),
                "booz_xform_jax_api_available": (geom or {}).get("booz_xform_jax_api_available"),
            },
            "next_action": (
                "Generalize the now-matched zero-beta Boozer equal-arc core/metric/drift convention across "
                "finite-beta pressure corrections and solver-objective geometry gradients for growth-rate/quasilinear gates."
            ),
        },
        {
            "lane": "Profiler-backed nonlinear hot-path optimization",
            "status": "closed" if release_performance_closed else ("partial" if profile_identity else "open"),
            "claim_level": (
                "release_hot_path_localization_closed_future_optimization_scoped"
                if release_performance_closed
                else "profile_identity_artifact_no_speedup_claim"
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
                "identity_gate_pass": profile_identity,
                "engineering_speedup": profile_speedup,
                "best_identity_candidate": profile_best.get("spec"),
                "best_identity_candidate_speedup": _finite_float(profile_best.get("engineering_speedup_median")),
                "rhs_fastest_full_label": (rhs_profile or {}).get("fastest_full_rhs_label"),
                "rhs_cpu_full_grid_over_spectral": _finite_float(rhs_cpu.get("full_rhs_grid_over_spectral")),
                "rhs_gpu_full_grid_over_spectral": _finite_float(rhs_gpu.get("full_rhs_grid_over_spectral")),
                "rhs_cpu_bracket_grid_over_spectral": _finite_float(
                    rhs_cpu.get("nonlinear_bracket_grid_over_spectral")
                ),
                "rhs_gpu_bracket_grid_over_spectral": _finite_float(
                    rhs_gpu.get("nonlinear_bracket_grid_over_spectral")
                ),
                "device_count": (profile or {}).get("device_count"),
                "backend": (profile or {}).get("default_backend"),
                "miller_cpu_grid_full_rhs": miller_cpu_grid_full,
                "miller_gpu_grid_full_rhs": miller_gpu_grid_full,
                "miller_gpu_spectral_full_rhs": miller_gpu_spectral_full,
                "w7x_cpu_full_rhs": w7x_cpu_full,
                "w7x_gpu_full_rhs": w7x_gpu_full,
                "hsx_cpu_full_rhs": hsx_cpu_full,
                "hsx_gpu_full_rhs": hsx_gpu_full,
                "full_trace_cpu_warm_seconds": full_rhs_cpu_warm,
                "full_trace_gpu_warm_seconds": full_rhs_gpu_warm,
            },
            "next_action": (
                "Release hot-path localization is closed with matched CPU/GPU profiler artifacts and conservative "
                "claim scope; continue larger-state algorithmic optimization as a future science/performance lane."
                if release_performance_closed
                else "Collect matched CPU/GPU profiler traces and optimize only persistent nonlinear bracket/field-solve "
                "hot paths; do not publish speedup claims until fresh profiler artifacts pass identity gates."
            ),
        },
    ]

    return {
        "kind": "open_research_lane_status",
        "claim_scope": "post_v1_5_development_tracking_no_unvalidated_promotion",
        "status_order": STATUS_ORDER,
        "lanes": lanes,
        "summary": {
            "n_lanes": len(lanes),
            "n_closed": sum(1 for lane in lanes if lane["status"] == "closed"),
            "n_partial": sum(1 for lane in lanes if lane["status"] == "partial"),
            "n_open": sum(1 for lane in lanes if lane["status"] == "open"),
            "n_blocked": sum(1 for lane in lanes if lane["status"] == "blocked"),
        },
    }


def write_status_artifacts(payload: dict[str, Any], *, out_png: Path = DEFAULT_OUT) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF artifacts for the lane-status payload."""

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_png.with_suffix(".json")
    out_csv = out_png.with_suffix(".csv")
    out_pdf = out_png.with_suffix(".pdf")

    out_json.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = ["lane", "status", "claim_level", "primary_artifacts", "next_action"]
    with out_csv.open("w", newline="") as handle:
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
    lanes = payload["lanes"]
    y = np.arange(len(lanes))
    colors = [STATUS_COLORS.get(str(lane["status"]), "#777777") for lane in lanes]
    values = [STATUS_ORDER.get(str(lane["status"]), 3) for lane in lanes]
    labels = [textwrap.fill(str(lane["lane"]), width=38) for lane in lanes]

    fig, ax = plt.subplots(figsize=(11.5, 5.9))
    ax.barh(y, values, color=colors, edgecolor="#333333", alpha=0.95)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 3.2)
    ax.set_xticks([0, 1, 2, 3], ["closed", "partial", "open", "blocked"])
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("Open research lanes: executable status and claim scope")
    ax.grid(axis="x", alpha=0.25)
    for yi, lane, value in zip(y, lanes, values, strict=True):
        metric = ""
        key_metrics = lane.get("key_metrics", {})
        if lane["lane"].startswith("W7-X zonal"):
            failed = key_metrics.get("failed_reference_gates", [])
            mixed = key_metrics.get("best_mixed_lm_resolution_probe", {})
            tail_ratio = mixed.get("tail_std_ratio") if isinstance(mixed, dict) else None
            metric = f"failed gates: {len(failed)}"
            if tail_ratio is not None:
                metric += f"; mixed-LM tail ratio: {tail_ratio:.2f}"
        elif lane["lane"].startswith("W7-X fluctuation"):
            metric = f"samples: {key_metrics.get('time_samples')}"
        elif lane["lane"].startswith("Nonlinear holdouts"):
            replicated = sum(
                1
                for key in (
                    "dshape_external_vmec_t250_replicated",
                    "circular_external_vmec_t700_replicated",
                )
                if key_metrics.get(key)
            )
            cv_pairs = key_metrics.get("variance_reduced_nonlinear_gradient_common_pairs")
            cv_uncertainty = key_metrics.get("variance_reduced_nonlinear_gradient_uncertainty_rel")
            cv_text = ""
            if cv_pairs is not None and cv_uncertainty is not None:
                cv_text = f"; CV pairs: {cv_pairs}, u={float(cv_uncertainty):.2f}"
            metric = (
                f"holdouts: {key_metrics.get('holdout_points')}, replicated: {replicated}, "
                f"promoted: {key_metrics.get('calibration_report_passed')}{cv_text}"
            )
        elif lane["lane"].startswith("vmec_jax"):
            field_line_abs = key_metrics.get("vmec_field_line_tensor_max_abs_ad_fd_error")
            flux_tube_rel = key_metrics.get("vmec_tensor_flux_tube_max_rel_ad_fd_error")
            array_worst = key_metrics.get("vmec_tensor_vs_eik_array_parity_worst_core_norm")
            metric_abs = key_metrics.get("vmec_metric_tensor_max_abs_ad_fd_error")
            state_abs = key_metrics.get("vmec_state_boozer_flux_tube_max_abs_ad_fd_error")
            if flux_tube_rel is not None and array_worst is not None:
                metric = f"VMEC flux-tube AD-FD: {flux_tube_rel:.1e} rel; arrays: {array_worst:.1e}"
            elif flux_tube_rel is not None:
                metric = f"VMEC flux-tube AD-FD: {flux_tube_rel:.1e} rel"
            elif field_line_abs is not None:
                metric = f"VMEC field-line AD-FD: {field_line_abs:.1e}"
            elif metric_abs is not None:
                metric = f"VMEC metric AD-FD: {metric_abs:.1e}"
            elif state_abs is None:
                metric = f"AD-FD max: {key_metrics.get('max_abs_ad_fd_error'):.1e}"
            else:
                metric = f"VMEC-state AD-FD: {state_abs:.1e}"
        elif lane["lane"].startswith("Profiler"):
            speed = key_metrics.get("engineering_speedup")
            best = key_metrics.get("best_identity_candidate")
            best_speed = key_metrics.get("best_identity_candidate_speedup")
            rhs_gpu_speed = key_metrics.get("rhs_gpu_full_grid_over_spectral")
            metric = "primary: n/a" if speed is None else f"primary: {speed:.2f}x"
            if best_speed is not None:
                metric += f"; best {best}: {best_speed:.2f}x"
            if rhs_gpu_speed is not None:
                metric += f"; RHS GPU split: {rhs_gpu_speed:.2f}x"
        ax.text(min(value + 0.06, 3.05), float(yi), metric, va="center", ha="left", fontsize=8.2)

    caption = (
        "Partial means a bounded diagnostic/gate exists, but the broader manuscript claim remains scoped. "
        "Open means no promotion until the listed physics or profiler gate passes."
    )
    fig.text(0.5, 0.025, caption, fontsize=8.2, color="#333333", ha="center")
    fig.subplots_adjust(left=0.35, right=0.97, top=0.90, bottom=0.14)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return {"png": str(out_png), "pdf": str(out_pdf), "json": str(out_json), "csv": str(out_csv)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_status_payload(Path(args.root))
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_status_artifacts(payload, out_png=Path(args.out))
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
