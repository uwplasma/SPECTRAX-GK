#!/usr/bin/env python3
"""Build research-status, manuscript-readiness, and closure dashboards.

This command consolidates the status-dashboard artifact builders that used to
live as separate one-panel scripts. Use subcommands to keep the public workflow
explicit while keeping artifact tooling easier to navigate:

* ``open-lanes``: broad open research lane status dashboard.
* ``manuscript-readiness``: scoped manuscript readiness dashboard.
* ``pre-manuscript-closure``: strict pre-manuscript closure dashboard.
* ``runbook``: actionable closure runbook.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
import textwrap
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # type: ignore[import-untyped]  # noqa: E402


# ---------------------------------------------------------------------------
# open status implementation, migrated from the former open-lane dashboard script.
# ---------------------------------------------------------------------------


_open_REPO_ROOT = Path(__file__).resolve().parents[2]
_open_DEFAULT_OUT = (
    _open_REPO_ROOT / "docs" / "_static" / "open_research_lane_status.png"
)

_open_STATUS_ORDER = {"closed": 0, "partial": 1, "open": 2, "blocked": 3}
_open_STATUS_COLORS = {
    "closed": "#2a9d8f",
    "partial": "#e9c46a",
    "open": "#f4a261",
    "blocked": "#d1495b",
}


def _open__read_json(root: Path, relative: str) -> dict[str, Any] | None:
    path = root / relative
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{relative} must contain a JSON object")
    return payload


def _open__finite_float(value: object, default: float | None = None) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _open__json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _open__json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_open__json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _open__json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _open__gate_failures(gate_report: dict[str, Any] | None) -> list[str]:
    if not gate_report:
        return []
    failures: list[str] = []
    for gate in gate_report.get("gates", []):
        if isinstance(gate, dict) and not bool(gate.get("passed", False)):
            failures.append(str(gate.get("metric", "unknown")))
    return failures


def _open__best_recurrence_candidate(payload: dict[str, Any] | None) -> dict[str, Any]:
    rows = [] if payload is None else payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return {
            "label": None,
            "mean_abs_error": None,
            "tail_std_ratio": None,
            "hermite_tail": None,
        }
    finite_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and _open__finite_float(row.get("mean_abs_error")) is not None
    ]
    if not finite_rows:
        return {
            "label": None,
            "mean_abs_error": None,
            "tail_std_ratio": None,
            "hermite_tail": None,
        }
    best = min(finite_rows, key=lambda row: float(row["mean_abs_error"]))
    tail_std = _open__finite_float(best.get("tail_std"), 0.0) or 0.0
    ref_tail_std = _open__finite_float(best.get("reference_tail_std"), 0.0) or 0.0
    ratio = None if ref_tail_std <= 0.0 else tail_std / ref_tail_std
    return {
        "label": str(best.get("label", "unknown")),
        "mean_abs_error": _open__finite_float(best.get("mean_abs_error")),
        "tail_std_ratio": ratio,
        "hermite_tail": _open__finite_float(best.get("hermite_tail_at_tmax")),
        "source_path": best.get("source_path"),
    }


def _open__highest_moment_candidate(payload: dict[str, Any] | None) -> dict[str, Any]:
    rows = [] if payload is None else payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return {
            "label": None,
            "mean_abs_error": None,
            "tail_std_ratio": None,
            "hermite_tail": None,
        }
    finite_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and _open__finite_float(row.get("mean_abs_error")) is not None
    ]
    if not finite_rows:
        return {
            "label": None,
            "mean_abs_error": None,
            "tail_std_ratio": None,
            "hermite_tail": None,
        }
    best = max(
        finite_rows,
        key=lambda row: (
            int(_open__finite_float(row.get("Nl"), 0.0) or 0.0),
            int(_open__finite_float(row.get("Nm"), 0.0) or 0.0),
        ),
    )
    tail_std = _open__finite_float(best.get("tail_std"), 0.0) or 0.0
    ref_tail_std = _open__finite_float(best.get("reference_tail_std"), 0.0) or 0.0
    ratio = None if ref_tail_std <= 0.0 else tail_std / ref_tail_std
    return {
        "label": str(best.get("label", "unknown")),
        "Nl": int(_open__finite_float(best.get("Nl"), 0.0) or 0.0),
        "Nm": int(_open__finite_float(best.get("Nm"), 0.0) or 0.0),
        "mean_abs_error": _open__finite_float(best.get("mean_abs_error")),
        "tail_std_ratio": ratio,
        "hermite_tail": _open__finite_float(best.get("hermite_tail_at_tmax")),
        "laguerre_tail": _open__finite_float(best.get("laguerre_tail_at_tmax")),
        "source_path": best.get("source_path"),
    }


def _open__best_hypercollision_probe(
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    rows = [] if payload is None else payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return None
    finite_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and _open__finite_float(row.get("mean_abs_error")) is not None
    ]
    if not finite_rows:
        return None
    best = min(finite_rows, key=lambda row: float(row["mean_abs_error"]))
    tail_std = _open__finite_float(best.get("tail_std"), 0.0) or 0.0
    ref_tail_std = _open__finite_float(best.get("reference_tail_std"), 0.0) or 0.0
    ratio = None if ref_tail_std <= 0.0 else tail_std / ref_tail_std
    validation_status = None if payload is None else payload.get("validation_status")
    return {
        "label": str(best.get("label", "unknown")),
        "mean_abs_error": _open__finite_float(best.get("mean_abs_error")),
        "tail_std_ratio": ratio,
        "hermite_tail": _open__finite_float(best.get("hermite_tail_at_tmax")),
        "free_energy_ratio": _open__finite_float(
            best.get("free_energy_at_tmax_over_initial")
        ),
        "source_path": best.get("source_path"),
        "validation_status": validation_status,
    }


def _open__holdout_counts(report: dict[str, Any] | None) -> tuple[int, int, list[str]]:
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


def _open__profile_seconds(
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
    return _open__finite_float(seconds.get(kernel))


def _open_build_status_payload(root: Path = _open_REPO_ROOT) -> dict[str, Any]:
    """Return a JSON-ready status payload for active research lanes."""

    root = Path(root)
    zonal_ref = _open__read_json(root, "docs/_static/w7x_zonal_reference_compare.json")
    zonal_recurrence = _open__read_json(
        root, "docs/_static/w7x_zonal_recurrence_sweep_kx070.json"
    )
    zonal_hypercollision = _open__read_json(
        root, "docs/_static/w7x_zonal_hypercollision_probe_kx070.json"
    )
    zonal_mixed_lm_resolution = _open__read_json(
        root, "docs/_static/w7x_zonal_mixedlm_resolution_kx070.json"
    )
    fluct = _open__read_json(root, "docs/_static/w7x_fluctuation_spectrum_panel.json")
    ql_inputs = _open__read_json(
        root, "docs/_static/quasilinear_validated_calibration_inputs.json"
    )
    ql_report = _open__read_json(
        root, "docs/_static/quasilinear_stellarator_train_holdout_report.json"
    )
    ql_uq = _open__read_json(
        root, "docs/_static/quasilinear_candidate_uncertainty.json"
    )
    ql_dataset = _open__read_json(
        root, "docs/_static/quasilinear_dataset_sufficiency.json"
    )
    ql_model_status = _open__read_json(
        root, "docs/_static/quasilinear_model_selection_status.json"
    )
    ql_error_anatomy = _open__read_json(
        root, "docs/_static/quasilinear_error_anatomy.json"
    )
    production_nl_guard = _open__read_json(
        root, "docs/_static/production_nonlinear_optimization_guard.json"
    )
    baseline_optimized_audit = _open__read_json(
        root, "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json"
    )
    nonlinear_control_mean_gate = _open__read_json(
        root,
        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json",
    )
    circular_t250_gate = _open__read_json(
        root, "docs/_static/external_vmec_circular_t250_high_grid_convergence_gate.json"
    )
    circular_t700_replicate = _open__read_json(
        root,
        "docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json",
    )
    dshape_gate = _open__read_json(
        root, "docs/_static/external_vmec_dshape_t250_high_grid_convergence_gate.json"
    )
    dshape_replicate = _open__read_json(
        root,
        "docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.json",
    )
    itermodel_gate = _open__read_json(
        root,
        "docs/_static/external_vmec_itermodel_t350_high_grid_convergence_gate.json",
    )
    updown_gate = _open__read_json(
        root,
        "docs/_static/external_vmec_updown_asym_t450_high_grid_convergence_gate.json",
    )
    qh_gate = _open__read_json(
        root, "docs/_static/external_vmec_qh_grid_convergence_gate.json"
    )
    qh_high_gate = _open__read_json(
        root, "docs/_static/external_vmec_qh_high_grid_convergence_gate.json"
    )
    cth_gate = _open__read_json(
        root, "docs/_static/external_vmec_cth_like_grid_convergence_gate.json"
    )
    cth_high_grid_admission = _open__read_json(
        root,
        "docs/_static/external_vmec_cth_like_modified_high_grid_admission_gate.json",
    )
    geom = _open__read_json(root, "docs/_static/differentiable_geometry_bridge.json")
    geom_matrix = _open__read_json(root, "docs/_static/vmec_boozer_parity_matrix.json")
    profile = _open__read_json(
        root, "docs/_static/nonlinear_sharding_profile_office_gpu.json"
    )
    rhs_profile = _open__read_json(root, "docs/_static/nonlinear_rhs_profile.json")
    rhs_miller = _open__read_json(
        root, "docs/_static/nonlinear_rhs_profile_miller.json"
    )
    rhs_stellarator = _open__read_json(
        root, "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json"
    )
    full_rhs_trace_cpu = _open__read_json(
        root, "docs/_static/full_nonlinear_rhs_trace_summary.json"
    )
    full_rhs_trace_gpu = _open__read_json(
        root, "docs/_static/full_nonlinear_rhs_trace_gpu_summary.json"
    )

    zonal_failures = _open__gate_failures(
        zonal_ref.get("gate_report") if zonal_ref else None
    )
    best_recurrence = _open__best_recurrence_candidate(zonal_recurrence)
    best_hypercollision = _open__best_hypercollision_probe(zonal_hypercollision)
    best_mixed_lm_resolution = _open__best_recurrence_candidate(
        zonal_mixed_lm_resolution
    )
    high_moment_mixed_lm_resolution = _open__highest_moment_candidate(
        zonal_mixed_lm_resolution
    )
    zonal_status = (
        "closed"
        if zonal_ref
        and not zonal_failures
        and zonal_ref.get("validation_status") == "closed"
        else "open"
    )
    w7x_tem_extension = _open__read_json(
        root, "docs/_static/w7x_tem_extension_status.json"
    )
    w7x_tem_rows = (w7x_tem_extension or {}).get("rows", [])
    w7x_tem_open = [
        row.get("lane", "unknown")
        for row in w7x_tem_rows
        if isinstance(row, dict) and str(row.get("status")) == "open"
    ]

    train_count, holdout_count, holdout_names = _open__holdout_counts(ql_report)
    ql_report_passed = bool(ql_report.get("passed", False)) if ql_report else False
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
    ql_core_gate = (
        (ql_error_anatomy or {}).get("core_portfolio_gate", {})
        if isinstance((ql_error_anatomy or {}).get("core_portfolio_gate", {}), dict)
        else {}
    )
    ql_core_passed = bool(ql_core_gate.get("passed", False))
    ql_passed = bool(
        ql_report_passed
        or ql_core_passed
        or (
            ql_uq_gate.get("passed", False)
            and ql_dataset_gate.get("passed", False)
            and ql_model_status_gate.get("passed", False)
        )
    )
    circular_t250_passed = bool(
        (circular_t250_gate or {}).get("gate_report", {}).get("passed", False)
    )
    circular_t700_replicate_passed = bool(
        (circular_t700_replicate or {}).get("passed", False)
    )
    dshape_passed = bool(
        (dshape_gate or {}).get("gate_report", {}).get("passed", False)
    )
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
    itermodel_passed = bool(
        (itermodel_gate or {}).get("gate_report", {}).get("passed", False)
    )
    updown_passed = bool(
        (updown_gate or {}).get("gate_report", {}).get("passed", False)
    )
    qh_passed = bool((qh_gate or {}).get("gate_report", {}).get("passed", False))
    qh_high_passed = bool(
        (qh_high_gate or {}).get("gate_report", {}).get("passed", False)
    )
    cth_full_grid_passed = bool(
        (cth_gate or {}).get("gate_report", {}).get("passed", False)
    )
    cth_high_grid_admitted = bool(
        (cth_high_grid_admission or {}).get("promotion_gate", {}).get("passed", False)
    )
    cth_passed = cth_full_grid_passed or cth_high_grid_admitted

    geom_sensitivity = (
        (geom or {}).get("sensitivity", {})
        if isinstance((geom or {}).get("sensitivity", {}), dict)
        else {}
    )
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
        geom_vmec_state.get("sensitivity", {})
        if isinstance(geom_vmec_state.get("sensitivity", {}), dict)
        else {}
    )
    geom_inverse = (geom or {}).get("geometry_inverse_design_report", {})
    geom_uq = (geom or {}).get("uq", {})
    geom_max_abs = _open__finite_float(geom_sensitivity.get("max_abs_ad_fd_error"))
    geom_inverse_res = (
        _open__finite_float(geom_inverse.get("final_residual_norm"))
        if isinstance(geom_inverse, dict)
        else None
    )
    geom_rank = (
        int(geom_uq.get("sensitivity_map_rank", 0)) if isinstance(geom_uq, dict) else 0
    )
    geom_vmec_metric_abs = _open__finite_float(
        geom_vmec_metric.get("max_abs_ad_fd_error")
    )
    geom_vmec_metric_rel = _open__finite_float(
        geom_vmec_metric.get("max_rel_ad_fd_error")
    )
    geom_vmec_field_line_abs = _open__finite_float(
        geom_vmec_field_line.get("max_abs_ad_fd_error")
    )
    geom_vmec_field_line_rel = _open__finite_float(
        geom_vmec_field_line.get("max_rel_ad_fd_error")
    )
    geom_vmec_flux_tube_abs = _open__finite_float(
        geom_vmec_flux_tube_sensitivity.get("max_abs_ad_fd_error")
    )
    geom_vmec_flux_tube_rel = _open__finite_float(
        geom_vmec_flux_tube_sensitivity.get("max_rel_ad_fd_error")
    )
    geom_vmec_array_parity_worst = _open__finite_float(
        geom_vmec_array_parity.get("worst_core_normalized_max_abs")
    )
    geom_vmec_array_parity_passed = bool(
        geom_vmec_array_parity.get("production_parity_passed", False)
    )
    geom_vmec_equal_arc_core_worst = _open__finite_float(
        geom_vmec_array_parity.get("equal_arc_core_worst_normalized_max_abs")
    )
    geom_vmec_equal_arc_core_scalar = _open__finite_float(
        geom_vmec_array_parity.get("equal_arc_core_worst_scalar_rel")
    )
    geom_vmec_equal_arc_derivative_worst = _open__finite_float(
        geom_vmec_array_parity.get("equal_arc_derivative_worst_normalized_max_abs")
    )
    geom_vmec_equal_arc_metric_worst = _open__finite_float(
        geom_vmec_array_parity.get("equal_arc_metric_worst_normalized_max_abs")
    )
    geom_vmec_equal_arc_drift_worst = _open__finite_float(
        geom_vmec_array_parity.get("equal_arc_drift_worst_normalized_max_abs")
    )
    geom_vmec_equal_arc_core_passed = bool(
        geom_vmec_array_parity.get("equal_arc_core_passed", False)
    )
    geom_vmec_equal_arc_derivative_passed = bool(
        geom_vmec_array_parity.get("equal_arc_derivative_passed", False)
    )
    geom_vmec_equal_arc_metric_passed = bool(
        geom_vmec_array_parity.get("equal_arc_metric_passed", False)
    )
    geom_vmec_equal_arc_drift_passed = bool(
        geom_vmec_array_parity.get("equal_arc_drift_passed", False)
    )
    geom_vmec_state_abs = _open__finite_float(
        geom_vmec_state_sensitivity.get("max_abs_ad_fd_error")
    )
    geom_vmec_state_rel = _open__finite_float(
        geom_vmec_state_sensitivity.get("max_rel_ad_fd_error")
    )
    geom_matrix_summary = (
        geom_matrix.get("summary", {}) if isinstance(geom_matrix, dict) else {}
    )
    geom_matrix_rows = (
        geom_matrix.get("rows", []) if isinstance(geom_matrix, dict) else []
    )
    geom_matrix_all_equal_arc_passed = bool(
        geom_matrix_summary.get("all_equal_arc_passed", False)
    )
    geom_matrix_min_mode = (
        int(geom_matrix.get("minimum_boozer_mode_count", 0))
        if isinstance(geom_matrix, dict)
        else 0
    )
    geom_matrix_limiting_drift = None
    if isinstance(geom_matrix_rows, list):
        finite_drift_rows = [
            row
            for row in geom_matrix_rows
            if isinstance(row, dict)
            and _open__finite_float(row.get("equal_arc_drift_worst_normalized_max_abs"))
            is not None
        ]
        if finite_drift_rows:
            limiting = max(
                finite_drift_rows,
                key=lambda row: float(row["equal_arc_drift_worst_normalized_max_abs"]),
            )
            geom_matrix_limiting_drift = {
                "case_name": limiting.get("case_name"),
                "value": _open__finite_float(
                    limiting.get("equal_arc_drift_worst_normalized_max_abs")
                ),
            }

    profile_identity = bool((profile or {}).get("identity_gate_pass", False))
    profile_speedup = _open__finite_float((profile or {}).get("engineering_speedup"))
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
    miller_cpu_grid_full = _open__profile_seconds(rhs_miller, "CPU grid", "full_rhs")
    miller_gpu_grid_full = _open__profile_seconds(rhs_miller, "GPU grid", "full_rhs")
    miller_gpu_spectral_full = _open__profile_seconds(
        rhs_miller, "GPU spectral", "full_rhs"
    )
    w7x_cpu_full = _open__profile_seconds(rhs_stellarator, "W7-X CPU", "full_rhs")
    w7x_gpu_full = _open__profile_seconds(rhs_stellarator, "W7-X GPU", "full_rhs")
    hsx_cpu_full = _open__profile_seconds(rhs_stellarator, "HSX CPU", "full_rhs")
    hsx_gpu_full = _open__profile_seconds(rhs_stellarator, "HSX GPU", "full_rhs")
    full_rhs_cpu_warm = _open__finite_float(
        (full_rhs_trace_cpu or {}).get("warm_seconds")
    )
    full_rhs_gpu_warm = _open__finite_float(
        (full_rhs_trace_gpu or {}).get("warm_seconds")
    )
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
            "status": "partial"
            if bool(fluct and fluct.get("source_gate_passed"))
            else "open",
            "claim_level": "validated_simulation_spectrum_tem_extension_open",
            "primary_artifacts": [
                "docs/_static/w7x_fluctuation_spectrum_panel.json",
                "docs/_static/w7x_tem_extension_status.json",
                "docs/_static/tem_branch_parity_audit.json",
                "docs/_static/tem_mismatch_table.csv",
            ],
            "key_metrics": {
                "time_samples": (fluct or {}).get("time_samples"),
                "time_window": [
                    (fluct or {}).get("time_min"),
                    (fluct or {}).get("time_max"),
                ],
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
            "lane": "Scoped core quasilinear model-development diagnostic",
            "status": "closed" if ql_passed else "open",
            "claim_level": (
                "diagnostic_calibration_dataset_not_absolute_flux"
                if not ql_passed
                else (
                    "calibrated_absolute_flux"
                    if ql_report_passed
                    else "scoped_core_absolute_flux_diagnostic_not_universal_predictor"
                    if ql_core_passed
                    else "scoped_candidate_model_selection_not_absolute_flux"
                )
            ),
            "primary_artifacts": [
                "docs/_static/quasilinear_validated_calibration_inputs.json",
                "docs/_static/quasilinear_stellarator_train_holdout_report.json",
                "docs/_static/quasilinear_candidate_uncertainty.json",
                "docs/_static/quasilinear_error_anatomy.json",
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
                "dataset_sufficiency_promotion_passed": bool(
                    ql_dataset_gate.get("passed", False)
                ),
                "model_selection_status_passed": bool(
                    ql_model_status_gate.get("passed", False)
                ),
                "core_portfolio_passed": ql_core_passed,
                "core_mean_abs_relative_error": _open__finite_float(
                    ql_core_gate.get("core_mean_abs_relative_error")
                ),
                "core_holdout_mean_abs_relative_error": _open__finite_float(
                    ql_core_gate.get("core_holdout_mean_abs_relative_error")
                ),
                "core_prediction_interval_coverage": _open__finite_float(
                    ql_core_gate.get("core_prediction_interval_coverage")
                ),
                "core_screening_gate_passed": bool(
                    ql_core_gate.get("screening_gate_passed", False)
                ),
                "model_selection_candidate_mean_error": _open__finite_float(
                    (ql_model_status or {})
                    .get("metrics", {})
                    .get("candidate_mean_abs_relative_error")
                    if isinstance((ql_model_status or {}).get("metrics", {}), dict)
                    else None
                ),
                "circular_external_vmec_t250_converged": circular_t250_passed,
                "circular_external_vmec_t700_replicated": circular_t700_replicate_passed,
                "circular_external_vmec_t700_replicate_mean_rel_spread": _open__finite_float(
                    (circular_t700_replicate or {})
                    .get("statistics", {})
                    .get("mean_rel_spread")
                    if isinstance(
                        (circular_t700_replicate or {}).get("statistics", {}), dict
                    )
                    else None
                ),
                "circular_external_vmec_t700_replicate_sem_rel": _open__finite_float(
                    (circular_t700_replicate or {})
                    .get("statistics", {})
                    .get("combined_sem_rel")
                    if isinstance(
                        (circular_t700_replicate or {}).get("statistics", {}), dict
                    )
                    else None
                ),
                "dshape_external_vmec_t250_converged": dshape_passed,
                "dshape_external_vmec_t250_replicated": dshape_replicate_passed,
                "dshape_external_vmec_t250_replicate_mean_rel_spread": _open__finite_float(
                    (dshape_replicate or {})
                    .get("statistics", {})
                    .get("mean_rel_spread")
                    if isinstance((dshape_replicate or {}).get("statistics", {}), dict)
                    else None
                ),
                "dshape_external_vmec_t250_replicate_sem_rel": _open__finite_float(
                    (dshape_replicate or {})
                    .get("statistics", {})
                    .get("combined_sem_rel")
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
                    baseline_optimized_comparison.get("uncertainty_separation_sigma")
                ),
                "variance_reduced_nonlinear_gradient_control_mean_passed": bool(
                    (nonlinear_control_mean_gate or {}).get("passed", False)
                ),
                "variance_reduced_nonlinear_gradient_common_pairs": (
                    nonlinear_control_mean_summary.get("common_pair_count")
                ),
                "variance_reduced_nonlinear_gradient_uncertainty_rel": _open__finite_float(
                    nonlinear_control_mean_summary.get(
                        "combined_response_uncertainty_rel"
                    )
                ),
                "variance_reduced_nonlinear_gradient_response_mean": _open__finite_float(
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
                "Use the closed scoped-core QL diagnostic for examples and optimization screening; keep declared "
                "stress outliers deferred until a new saturation-physics lane is opened."
                if ql_core_passed
                else "Document the accepted richer candidate and matched QA no-ESS to optimized QA/ESS audit with scoped "
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
            "status": "partial"
            if geom_max_abs is not None and geom_rank >= 2
            else "open",
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
                "vmec_jax_available": (geom or {})
                .get("backend_info", {})
                .get("vmec_jax_available"),
                "booz_xform_jax_api_available": (geom or {}).get(
                    "booz_xform_jax_api_available"
                ),
            },
            "next_action": (
                "Generalize the now-matched zero-beta Boozer equal-arc core/metric/drift convention across "
                "finite-beta pressure corrections and solver-objective geometry gradients for growth-rate/quasilinear gates."
            ),
        },
        {
            "lane": "Profiler-backed nonlinear hot-path optimization",
            "status": "closed"
            if release_performance_closed
            else ("partial" if profile_identity else "open"),
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
                "best_identity_candidate_speedup": _open__finite_float(
                    profile_best.get("engineering_speedup_median")
                ),
                "rhs_fastest_full_label": (rhs_profile or {}).get(
                    "fastest_full_rhs_label"
                ),
                "rhs_cpu_full_grid_over_spectral": _open__finite_float(
                    rhs_cpu.get("full_rhs_grid_over_spectral")
                ),
                "rhs_gpu_full_grid_over_spectral": _open__finite_float(
                    rhs_gpu.get("full_rhs_grid_over_spectral")
                ),
                "rhs_cpu_bracket_grid_over_spectral": _open__finite_float(
                    rhs_cpu.get("nonlinear_bracket_grid_over_spectral")
                ),
                "rhs_gpu_bracket_grid_over_spectral": _open__finite_float(
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
        "status_order": _open_STATUS_ORDER,
        "lanes": lanes,
        "summary": {
            "n_lanes": len(lanes),
            "n_closed": sum(1 for lane in lanes if lane["status"] == "closed"),
            "n_partial": sum(1 for lane in lanes if lane["status"] == "partial"),
            "n_open": sum(1 for lane in lanes if lane["status"] == "open"),
            "n_blocked": sum(1 for lane in lanes if lane["status"] == "blocked"),
        },
    }


def _open_write_status_artifacts(
    payload: dict[str, Any], *, out_png: Path = _open_DEFAULT_OUT
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF artifacts for the lane-status payload."""

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_png.with_suffix(".json")
    out_csv = out_png.with_suffix(".csv")
    out_pdf = out_png.with_suffix(".pdf")

    out_json.write_text(
        json.dumps(_open__json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    colors = [_open_STATUS_COLORS.get(str(lane["status"]), "#777777") for lane in lanes]
    values = [_open_STATUS_ORDER.get(str(lane["status"]), 3) for lane in lanes]
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
            tail_ratio = (
                mixed.get("tail_std_ratio") if isinstance(mixed, dict) else None
            )
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
            cv_pairs = key_metrics.get(
                "variance_reduced_nonlinear_gradient_common_pairs"
            )
            cv_uncertainty = key_metrics.get(
                "variance_reduced_nonlinear_gradient_uncertainty_rel"
            )
            cv_text = ""
            if cv_pairs is not None and cv_uncertainty is not None:
                cv_text = f"; CV pairs: {cv_pairs}, u={float(cv_uncertainty):.2f}"
            metric = (
                f"holdouts: {key_metrics.get('holdout_points')}, replicated: {replicated}, "
                f"promoted: {key_metrics.get('calibration_report_passed')}{cv_text}"
            )
        elif lane["lane"].startswith("vmec_jax"):
            field_line_abs = key_metrics.get(
                "vmec_field_line_tensor_max_abs_ad_fd_error"
            )
            flux_tube_rel = key_metrics.get("vmec_tensor_flux_tube_max_rel_ad_fd_error")
            array_worst = key_metrics.get(
                "vmec_tensor_vs_eik_array_parity_worst_core_norm"
            )
            metric_abs = key_metrics.get("vmec_metric_tensor_max_abs_ad_fd_error")
            state_abs = key_metrics.get(
                "vmec_state_boozer_flux_tube_max_abs_ad_fd_error"
            )
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
        ax.text(
            min(value + 0.06, 3.05),
            float(yi),
            metric,
            va="center",
            ha="left",
            fontsize=8.2,
        )

    caption = (
        "Partial means a bounded diagnostic/gate exists, but the broader manuscript claim remains scoped. "
        "Open means no promotion until the listed physics or profiler gate passes."
    )
    fig.text(0.5, 0.025, caption, fontsize=8.2, color="#333333", ha="center")
    fig.subplots_adjust(left=0.35, right=0.97, top=0.90, bottom=0.14)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return {
        "png": str(out_png),
        "pdf": str(out_pdf),
        "json": str(out_json),
        "csv": str(out_csv),
    }


# ---------------------------------------------------------------------------
# readiness status implementation, migrated from the former manuscript-readiness dashboard script.
# ---------------------------------------------------------------------------


_readiness_ROOT = Path(__file__).resolve().parents[2]
_readiness_DEFAULT_OUT = (
    _readiness_ROOT / "docs" / "_static" / "manuscript_readiness_status.png"
)

_readiness_STATUS_ORDER = {
    "closed": 0,
    "partial": 1,
    "open": 2,
    "deferred": 3,
    "blocked": 4,
}
_readiness_STATUS_COLORS = {
    "closed": "#2a9d8f",
    "partial": "#e9c46a",
    "open": "#f4a261",
    "deferred": "#8d99ae",
    "blocked": "#d1495b",
}


def _readiness__read_json(root: Path, relative: str) -> dict[str, Any] | None:
    path = root / relative
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{relative} must contain a JSON object")
    return payload


def _readiness__finite_float(
    value: object, default: float | None = None
) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _readiness__json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _readiness__json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_readiness__json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _readiness__json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _readiness__count_points(report: dict[str, Any] | None, split: str) -> int:
    if not report:
        return 0
    points = report.get("points", [])
    if not isinstance(points, list):
        return 0
    return sum(
        1 for point in points if isinstance(point, dict) and point.get("split") == split
    )


def _readiness__profile_seconds(
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
    return _readiness__finite_float(seconds.get(kernel))


def _readiness__all_optimization_objectives_passed(
    payload: dict[str, Any] | None,
) -> bool:
    results = [] if payload is None else payload.get("results", [])
    if not isinstance(results, list) or not results:
        return False
    for result in results:
        if not isinstance(result, dict):
            return False
        if not bool(result.get("gradient_gate", {}).get("passed", False)):
            return False
        initial = _readiness__finite_float(result.get("initial_objective"))
        final = _readiness__finite_float(result.get("final_objective"))
        if initial is None or final is None or final >= initial:
            return False
    return True


def _readiness__optimization_reduction_summary(
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
        initial = _readiness__finite_float(result.get("initial_objective"))
        final = _readiness__finite_float(result.get("final_objective"))
        if initial is None or final is None or initial <= 0.0:
            continue
        factors.append(final / initial)
    return {
        "n_objectives": len(results),
        "best_reduction_factor": None if not factors else float(min(factors)),
        "worst_reduction_factor": None if not factors else float(max(factors)),
    }


def _readiness__optimization_uq_gate_passed(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    return bool(
        payload.get("all_gradient_gates_passed", False)
        and payload.get("all_sensitivity_maps_full_rank", False)
        and payload.get("claim_level")
        == "reduced_objective_uq_and_sensitivity_validation_not_full_vmec_gk_optimization"
    )


def _readiness_build_manuscript_readiness_payload(
    root: Path = _readiness_ROOT,
) -> dict[str, Any]:
    """Return a JSON-ready manuscript-scope readiness payload."""

    root = Path(root)
    ql_inputs = _readiness__read_json(
        root, "docs/_static/quasilinear_validated_calibration_inputs.json"
    )
    ql_holdout = _readiness__read_json(
        root, "docs/_static/quasilinear_stellarator_train_holdout_report.json"
    )
    ql_sweep = _readiness__read_json(
        root, "docs/_static/quasilinear_saturation_rule_sweep.json"
    )
    ql_shape = _readiness__read_json(
        root, "docs/_static/quasilinear_shape_aware_saturation.json"
    )
    ql_uq = _readiness__read_json(
        root, "docs/_static/quasilinear_candidate_uncertainty.json"
    )
    ql_dataset = _readiness__read_json(
        root, "docs/_static/quasilinear_dataset_sufficiency.json"
    )
    ql_model_status = _readiness__read_json(
        root, "docs/_static/quasilinear_model_selection_status.json"
    )
    ql_guardrails = _readiness__read_json(
        root, "docs/_static/quasilinear_promotion_guardrails.json"
    )
    dshape_replicate = _readiness__read_json(
        root,
        "docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.json",
    )
    circular_replicate = _readiness__read_json(
        root,
        "docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json",
    )
    geom = _readiness__read_json(
        root, "docs/_static/differentiable_geometry_bridge.json"
    )
    geom_matrix = _readiness__read_json(
        root, "docs/_static/vmec_boozer_parity_matrix.json"
    )
    opt = _readiness__read_json(
        root, "docs/_static/stellarator_itg_optimization_comparison.json"
    )
    opt_uq = _readiness__read_json(
        root, "docs/_static/stellarator_itg_optimization_uq.json"
    )
    solver_grad = _readiness__read_json(
        root, "docs/_static/solver_objective_gradient_gate.json"
    )
    vmec_solver_grad = _readiness__read_json(
        root, "docs/_static/vmec_boozer_solver_frequency_gradient_gate.json"
    )
    vmec_ql_grad = _readiness__read_json(
        root, "docs/_static/vmec_boozer_quasilinear_gradient_gate.json"
    )
    vmec_nl_window_grad = _readiness__read_json(
        root, "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json"
    )
    vmec_li383_nl_window_grad = _readiness__read_json(
        root,
        "docs/_static/vmec_boozer_li383_nonlinear_window_gradient_gate.json",
    )
    vmec_gradient_matrix = _readiness__read_json(
        root, "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
    )
    nonlinear_fd_audit = _readiness__read_json(
        root, "docs/_static/nonlinear_window_fd_audit.json"
    )
    vmec_nonlinear_fd_audit = _readiness__read_json(
        root, "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json"
    )
    production_nl_guard = _readiness__read_json(
        root, "docs/_static/production_nonlinear_optimization_guard.json"
    )
    baseline_optimized_audit = _readiness__read_json(
        root, "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json"
    )
    nonlinear_control_mean_gate = _readiness__read_json(
        root,
        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json",
    )
    profile = _readiness__read_json(
        root, "docs/_static/nonlinear_sharding_profile_office_gpu.json"
    )
    rhs_profile = _readiness__read_json(root, "docs/_static/nonlinear_rhs_profile.json")
    rhs_miller = _readiness__read_json(
        root, "docs/_static/nonlinear_rhs_profile_miller.json"
    )
    rhs_stellarator = _readiness__read_json(
        root, "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json"
    )
    full_rhs_trace_cpu = _readiness__read_json(
        root, "docs/_static/full_nonlinear_rhs_trace_summary.json"
    )
    full_rhs_trace_gpu = _readiness__read_json(
        root, "docs/_static/full_nonlinear_rhs_trace_gpu_summary.json"
    )

    ql_inputs_passed = bool((ql_inputs or {}).get("passed", False))
    ql_holdout_promoted = bool((ql_holdout or {}).get("passed", False))
    ql_holdout_mean = _readiness__finite_float(
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
        _readiness__finite_float(geom_sensitivity.get("max_abs_ad_fd_error"))
        is not None
        and int(geom_uq.get("sensitivity_map_rank", 0)) >= 2
    )

    opt_reductions = _readiness__optimization_reduction_summary(opt)
    opt_reduced_objectives_passed = _readiness__all_optimization_objectives_passed(opt)
    opt_uq_passed = _readiness__optimization_uq_gate_passed(opt_uq)
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
    miller_cpu_grid_full = _readiness__profile_seconds(
        rhs_miller, "CPU grid", "full_rhs"
    )
    miller_gpu_grid_full = _readiness__profile_seconds(
        rhs_miller, "GPU grid", "full_rhs"
    )
    miller_gpu_spectral_full = _readiness__profile_seconds(
        rhs_miller, "GPU spectral", "full_rhs"
    )
    w7x_cpu_full = _readiness__profile_seconds(rhs_stellarator, "W7-X CPU", "full_rhs")
    w7x_gpu_full = _readiness__profile_seconds(rhs_stellarator, "W7-X GPU", "full_rhs")
    hsx_cpu_full = _readiness__profile_seconds(rhs_stellarator, "HSX CPU", "full_rhs")
    hsx_gpu_full = _readiness__profile_seconds(rhs_stellarator, "HSX GPU", "full_rhs")
    full_rhs_cpu_warm = _readiness__finite_float(
        (full_rhs_trace_cpu or {}).get("warm_seconds")
    )
    full_rhs_gpu_warm = _readiness__finite_float(
        (full_rhs_trace_gpu or {}).get("warm_seconds")
    )
    full_rhs_gpu_transposes = _readiness__finite_float(
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
                "train_points": _readiness__count_points(ql_holdout, "train"),
                "holdout_points": _readiness__count_points(ql_holdout, "holdout"),
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
                "model_selection_candidate_mean_error": _readiness__finite_float(
                    (ql_model_status or {})
                    .get("metrics", {})
                    .get("candidate_mean_abs_relative_error")
                    if isinstance((ql_model_status or {}).get("metrics", {}), dict)
                    else None
                ),
                "model_selection_interval_coverage": _readiness__finite_float(
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
                "null_training_mean_error": _readiness__finite_float(
                    ql_uq_gate.get("null_training_mean_mean_abs_relative_error")
                ),
                "dshape_replicate_passed": bool(
                    (dshape_replicate or {}).get("passed", False)
                ),
                "dshape_replicate_mean_rel_spread": _readiness__finite_float(
                    dshape_replicate_stats.get("mean_rel_spread")
                ),
                "dshape_replicate_combined_sem_rel": _readiness__finite_float(
                    dshape_replicate_stats.get("combined_sem_rel")
                ),
                "circular_replicate_passed": bool(
                    (circular_replicate or {}).get("passed", False)
                ),
                "circular_replicate_mean_rel_spread": _readiness__finite_float(
                    circular_replicate_stats.get("mean_rel_spread")
                ),
                "circular_replicate_combined_sem_rel": _readiness__finite_float(
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
                    baseline_optimized_comparison.get("uncertainty_separation_sigma")
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
                "multi_equilibrium_gradient_max_rel_error": _readiness__finite_float(
                    gradient_matrix_summary.get("max_relative_error")
                ),
                "linear_growth_gradient_gate": bool(
                    (solver_grad or {}).get("linear_growth_gradient_gate", False)
                ),
                "quasilinear_weight_gradient_gate": bool(
                    (solver_grad or {}).get("quasilinear_weight_gradient_gate", False)
                ),
                "vmec_boozer_frequency_rel_error": _readiness__finite_float(
                    (vmec_solver_grad or {})
                    .get("eigenpair_gate", {})
                    .get("max_rel_error")
                ),
                "vmec_boozer_quasilinear_rel_error": _readiness__finite_float(
                    (vmec_ql_grad or {}).get("eigenpair_gate", {}).get("max_rel_error")
                ),
                "vmec_boozer_reduced_nonlinear_window_rel_error": _readiness__finite_float(
                    (vmec_nl_window_grad or {})
                    .get("eigenpair_gate", {})
                    .get("max_rel_error")
                ),
                "vmec_boozer_li383_reduced_nonlinear_window_rel_error": _readiness__finite_float(
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
                "startup_nonlinear_plumbing_response_fraction": _readiness__finite_float(
                    nonlinear_fd_metrics.get("response_fraction")
                ),
                "startup_nonlinear_plumbing_repeatability_rel_error": _readiness__finite_float(
                    nonlinear_fd_metrics.get("repeatability_relative_error")
                ),
                "startup_nonlinear_plumbing_max_window_cv": _readiness__finite_float(
                    nonlinear_fd_metrics.get("max_window_cv")
                ),
                "startup_nonlinear_plumbing_max_window_trend": _readiness__finite_float(
                    nonlinear_fd_metrics.get("max_window_trend")
                ),
                "nonlinear_transport_average_gate": nonlinear_transport_average_gate,
                "production_nonlinear_observable_fd_path_gate": production_nonlinear_observable_fd_path_gate,
                "vmec_boozer_startup_nonlinear_plumbing_fd_path_gate": (
                    vmec_boozer_startup_nonlinear_plumbing_fd_path_gate
                ),
                "vmec_boozer_startup_nonlinear_response_fraction": _readiness__finite_float(
                    vmec_nonlinear_fd_metrics.get("response_fraction")
                ),
                "vmec_boozer_startup_nonlinear_derivative_asymmetry": _readiness__finite_float(
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
                "variance_reduced_nonlinear_gradient_uncertainty_rel": _readiness__finite_float(
                    nonlinear_control_mean_summary.get(
                        "combined_response_uncertainty_rel"
                    )
                ),
                "variance_reduced_nonlinear_gradient_response_mean": _readiness__finite_float(
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
                "engineering_speedup": _readiness__finite_float(
                    (profile or {}).get("engineering_speedup")
                ),
                "best_identity_candidate": profile_best.get("spec"),
                "best_identity_candidate_speedup": _readiness__finite_float(
                    profile_best.get("engineering_speedup_median")
                ),
                "rhs_fastest_full_label": (rhs_profile or {}).get(
                    "fastest_full_rhs_label"
                ),
                "rhs_cpu_full_grid_over_spectral": _readiness__finite_float(
                    rhs_cpu.get("full_rhs_grid_over_spectral")
                ),
                "rhs_gpu_full_grid_over_spectral": _readiness__finite_float(
                    rhs_gpu.get("full_rhs_grid_over_spectral")
                ),
                "rhs_cpu_bracket_grid_over_spectral": _readiness__finite_float(
                    rhs_cpu.get("nonlinear_bracket_grid_over_spectral")
                ),
                "rhs_gpu_bracket_grid_over_spectral": _readiness__finite_float(
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
        "status_order": _readiness_STATUS_ORDER,
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


def _readiness_write_manuscript_readiness_artifacts(
    payload: dict[str, Any], *, out: str | Path = _readiness_DEFAULT_OUT
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the manuscript readiness payload."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_readiness__json_clean(payload), indent=2, sort_keys=True) + "\n",
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
    values = [_readiness_STATUS_ORDER[str(lane["status"])] for lane in lanes]
    bar_values = [max(0.12, float(value)) for value in values]
    colors = [_readiness_STATUS_COLORS[str(lane["status"])] for lane in lanes]
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
            cv_uncertainty = km.get(
                "variance_reduced_nonlinear_gradient_uncertainty_rel"
            )
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


# ---------------------------------------------------------------------------
# closure status implementation, migrated from the former pre-manuscript closure dashboard script.
# ---------------------------------------------------------------------------


_closure_REPO_ROOT = Path(__file__).resolve().parents[2]
_closure_DEFAULT_OUT = (
    _closure_REPO_ROOT / "docs" / "_static" / "pre_manuscript_closure_status.png"
)

_closure_STATUS_COLORS = {
    "closed": "#2a9d8f",
    "partial": "#e9c46a",
    "open": "#f4a261",
    "blocked": "#d1495b",
}
_closure_STATUS_ORDER = {"closed": 0, "partial": 1, "open": 2, "blocked": 3}

_closure_QL_ABSOLUTE_ERROR_GATE = 0.35
_closure_MIN_BROAD_MATCHED_OPTIMIZATION_AUDITS = 3
_closure_MIN_BROAD_OPTIMIZED_EQUILIBRIUM_ENSEMBLES = 3
_closure_MIN_BROAD_REPLICATED_HOLDOUT_ENSEMBLES = 3
_closure_MIN_DOMAIN_CPU_SPEEDUP = 1.5
_closure_MIN_DOMAIN_GPU_SPEEDUP = 1.5


def _closure__read_json(root: Path, relative: str) -> dict[str, Any] | None:
    path = root / relative
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{relative} must contain a JSON object")
    return payload


def _closure__finite_float(value: object, default: float | None = None) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _closure__json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _closure__json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_closure__json_clean(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return _closure__json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _closure__as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _closure__as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _closure__gate_bool(
    payload: dict[str, Any] | None, *path: str, default: bool = False
) -> bool:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return bool(current) if current is not None else default


def _closure__count_split(report: dict[str, Any] | None, split: str) -> int:
    points = _closure__as_list((report or {}).get("points"))
    return sum(
        1
        for point in points
        if isinstance(point, dict) and str(point.get("split")) == split
    )


def _closure__ratio_score(value: int, target: int, weight: float) -> float:
    if target <= 0:
        return 0.0
    return min(float(value) / float(target), 1.0) * weight


def _closure__bool_score(passed: bool, weight: float) -> float:
    return weight if passed else 0.0


def _closure__normalize_blockers(blockers: list[str]) -> list[str]:
    """Return stable, human-readable blocker identifiers."""

    replacements = {
        "dataset_sufficiency_passed": "dataset_sufficiency_gate_failed",
        "candidate_uncertainty_passed": "candidate_uncertainty_gate_failed",
        "required_candidate_accepted": "required_candidate_not_accepted",
        "required_candidate_transport_error": "required_candidate_transport_error_gate_failed",
        "passed_holdout_surface_or_field_line_artifact": "production_scope_heldout_surface_or_field_line_artifact_missing",
    }
    normalized = [replacements.get(str(item), str(item)) for item in blockers]
    return sorted(set(normalized))


def _closure__max_speedup(rows: list[Any], backend: str) -> float | None:
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("backend", "")).lower() != backend.lower():
            continue
        speed = _closure__finite_float(row.get("speedup"))
        if speed is None:
            speed = _closure__finite_float(row.get("warm_speedup"))
        if speed is not None:
            values.append(speed)
    return max(values) if values else None


def _closure__lane_status(passed: bool, blockers: list[str], completion: float) -> str:
    if passed:
        return "closed"
    if blockers and completion < 55.0:
        return "blocked"
    return "partial" if completion >= 35.0 else "open"


def _closure__scoped_core_ql_lane(root: Path) -> dict[str, Any]:
    anatomy = _closure__read_json(root, "docs/_static/quasilinear_error_anatomy.json")
    ql_report = _closure__read_json(
        root, "docs/_static/quasilinear_stellarator_train_holdout_report.json"
    )
    ql_model = _closure__read_json(
        root, "docs/_static/quasilinear_model_selection_status.json"
    )
    ql_dataset = _closure__read_json(
        root, "docs/_static/quasilinear_dataset_sufficiency.json"
    )
    ql_guardrails = _closure__read_json(
        root, "docs/_static/quasilinear_promotion_guardrails.json"
    )

    core_gate = _closure__as_dict((anatomy or {}).get("core_portfolio_gate"))
    full_promotion_gate = _closure__as_dict((anatomy or {}).get("promotion_gate"))
    report_by_split = _closure__as_dict((ql_report or {}).get("by_split"))
    holdout_stats = _closure__as_dict(report_by_split.get("holdout"))
    train_stats = _closure__as_dict(report_by_split.get("train"))
    model_metrics = _closure__as_dict((ql_model or {}).get("metrics"))
    dataset_checks = _closure__as_dict(
        _closure__as_dict(
            _closure__as_dict((ql_dataset or {}).get("requirements")).get("checks")
        )
    )
    frozen_policy = _closure__as_dict((anatomy or {}).get("frozen_ledger_policy"))

    full_case_count = int((anatomy or {}).get("case_count") or 0)
    full_holdout_count = int((anatomy or {}).get("holdout_count") or 0)
    train = int(train_stats.get("n") or _closure__count_split(ql_report, "train"))
    holdouts = int(
        holdout_stats.get("n") or _closure__count_split(ql_report, "holdout")
    )
    core_count = int(core_gate.get("core_case_count") or 0)
    core_holdouts = int(core_gate.get("core_holdout_count") or 0)
    excluded_cases = _closure__as_list(core_gate.get("excluded_cases"))
    excluded_names = [
        str(item.get("case")) for item in excluded_cases if isinstance(item, dict)
    ]
    core_mean_error = _closure__finite_float(
        core_gate.get("core_mean_abs_relative_error")
    )
    core_holdout_error = _closure__finite_float(
        core_gate.get("core_holdout_mean_abs_relative_error")
    )
    core_max_error = _closure__finite_float(
        core_gate.get("core_max_abs_relative_error")
    )
    core_coverage = _closure__finite_float(
        core_gate.get("core_prediction_interval_coverage")
    )
    core_spearman = _closure__finite_float(core_gate.get("core_spearman"))
    core_holdout_spearman = _closure__finite_float(
        core_gate.get("core_holdout_spearman")
    )
    core_pairwise = _closure__finite_float(
        core_gate.get("core_pairwise_order_accuracy")
    )
    core_holdout_pairwise = _closure__finite_float(
        core_gate.get("core_holdout_pairwise_order_accuracy")
    )
    transport_gate = (
        _closure__finite_float(
            core_gate.get("transport_gate"), _closure_QL_ABSOLUTE_ERROR_GATE
        )
        or _closure_QL_ABSOLUTE_ERROR_GATE
    )
    interval_gate = (
        _closure__finite_float(core_gate.get("interval_coverage_gate"), 0.75) or 0.75
    )

    dataset_volume = (
        core_count >= 10 and core_holdouts >= 8 and train >= 2 and holdouts >= 10
    )
    validated_inputs = bool(dataset_checks.get("validated_input_gates", False)) or bool(
        ql_report
    )
    declared_outliers_recorded = len(excluded_names) >= 2
    core_transport_passed = bool(
        core_mean_error is not None
        and core_holdout_error is not None
        and core_mean_error <= transport_gate
        and core_holdout_error <= transport_gate
    )
    core_coverage_passed = bool(
        core_coverage is not None and core_coverage >= interval_gate
    )
    core_gate_passed = bool(core_gate.get("passed", False))
    guardrails_present = bool(ql_guardrails) and bool(anatomy)
    frozen_policy_present = (
        frozen_policy.get("additional_holdout_collection_active") is False
    )
    full_universal_promoted = bool(full_promotion_gate.get("passed", False))

    blockers: list[str] = []
    if not anatomy:
        blockers.append("quasilinear_error_anatomy_missing")
    if not core_gate_passed:
        blockers.extend(
            str(item) for item in _closure__as_list(core_gate.get("blockers"))
        )
        blockers.append("scoped_core_portfolio_gate_failed")
    if not dataset_volume:
        blockers.append("scoped_core_dataset_volume_below_gate")
    if not declared_outliers_recorded:
        blockers.append("declared_outlier_exclusion_record_missing")
    if not validated_inputs:
        blockers.append("validated_input_gate_missing")

    completion = (
        _closure__bool_score(bool(anatomy), 12.0)
        + _closure__bool_score(validated_inputs, 10.0)
        + _closure__bool_score(dataset_volume, 16.0)
        + _closure__bool_score(declared_outliers_recorded, 12.0)
        + _closure__bool_score(core_transport_passed, 22.0)
        + _closure__bool_score(core_coverage_passed, 10.0)
        + _closure__bool_score(guardrails_present, 8.0)
        + _closure__bool_score(frozen_policy_present, 10.0)
    )
    passed = bool(
        not blockers
        and core_gate_passed
        and core_transport_passed
        and core_coverage_passed
    )

    required_next_artifacts = (
        []
        if passed
        else [
            "a regenerated quasilinear residual-anatomy artifact with a passing declared core-portfolio gate",
            "explicit excluded-stress-case records for every case removed from the scoped core claim",
            "dataset and input-validation artifacts showing the core portfolio is not a single-geometry fit",
        ]
    )

    return {
        "lane": "Scoped core quasilinear heat-flux diagnostic",
        "status": _closure__lane_status(passed, blockers, completion),
        "passed": passed,
        "completion_percent": round(completion, 1),
        "claim_level": (
            "scoped_core_portfolio_absolute_flux_diagnostic_closed"
            if passed
            else "scoped_core_portfolio_absolute_flux_diagnostic_incomplete"
        ),
        "primary_artifacts": [
            "docs/_static/quasilinear_error_anatomy.json",
            "docs/_static/quasilinear_error_anatomy.png",
            "docs/_static/quasilinear_candidate_uncertainty.json",
            "docs/_static/quasilinear_model_selection_status.json",
            "docs/_static/quasilinear_dataset_sufficiency.json",
            "docs/_static/quasilinear_promotion_guardrails.json",
        ],
        "key_metrics": {
            "train_cases": train,
            "holdout_cases": holdouts,
            "full_case_count": full_case_count,
            "full_holdout_count": full_holdout_count,
            "full_candidate_mean_abs_relative_error": _closure__finite_float(
                (anatomy or {}).get("candidate_mean_abs_relative_error")
            ),
            "full_universal_promotion_passed": full_universal_promoted,
            "transport_mean_relative_error_gate": transport_gate,
            "core_case_count": core_count,
            "core_holdout_count": core_holdouts,
            "core_mean_abs_relative_error": core_mean_error,
            "core_holdout_mean_abs_relative_error": core_holdout_error,
            "core_max_abs_relative_error": core_max_error,
            "core_prediction_interval_coverage": core_coverage,
            "core_interval_coverage_gate": interval_gate,
            "core_spearman": core_spearman,
            "core_holdout_spearman": core_holdout_spearman,
            "core_pairwise_order_accuracy": core_pairwise,
            "core_holdout_pairwise_order_accuracy": core_holdout_pairwise,
            "core_screening_gate_passed": bool(
                core_gate.get("screening_gate_passed", False)
            ),
            "candidate_mean_abs_relative_error": _closure__finite_float(
                model_metrics.get("candidate_mean_abs_relative_error")
            ),
            "declared_stress_outliers": excluded_names,
        },
        "blockers": _closure__normalize_blockers(blockers),
        "required_next_artifacts": required_next_artifacts,
        "next_action": (
            "Use the passing scoped core QL portfolio for examples, model-development figures, and optimization-screening "
            "diagnostics; keep the declared stress outliers as deferred saturation-physics evidence before any universal "
            "absolute-flux runtime predictor is promoted."
            if passed
            else "Regenerate the residual-anatomy artifact and close the declared core-portfolio QL gate before using it in examples."
        ),
    }


def _closure__broad_nonlinear_optimization_lane(root: Path) -> dict[str, Any]:
    guard = _closure__read_json(
        root, "docs/_static/production_nonlinear_optimization_guard.json"
    )
    vmec_holdout = _closure__read_json(
        root, "docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json"
    )
    qa_status = _closure__read_json(
        root, "docs/_static/vmec_jax_qa_transport_optimization_status.json"
    )
    matrix_portfolio = _closure__read_json(
        root, "docs/_static/nonlinear_transport_matrix_portfolio.json"
    )

    summary = _closure__as_dict((guard or {}).get("summary"))
    matched = int(summary.get("qualifying_matched_optimized_transport_audits") or 0)
    optimized = int(summary.get("qualifying_optimized_equilibrium_ensembles") or 0)
    replicated = int(summary.get("qualifying_replicated_holdout_ensembles") or 0)
    scoped_guard_passed = bool((guard or {}).get("passed", False))
    holdout_promotion_passed = bool((vmec_holdout or {}).get("passed", False))
    qa_long_window_anchor = bool(
        _closure__as_dict((qa_status or {}).get("summary")).get(
            "long_window_nonlinear_audit_passed", False
        )
    )
    broad_matrix_portfolio_passed = bool((matrix_portfolio or {}).get("passed", False))

    blockers: list[str] = []
    if matched < _closure_MIN_BROAD_MATCHED_OPTIMIZATION_AUDITS:
        blockers.append("need_at_least_three_matched_optimized_transport_audits")
    if optimized < _closure_MIN_BROAD_OPTIMIZED_EQUILIBRIUM_ENSEMBLES:
        blockers.append("need_at_least_three_optimized_equilibrium_ensembles")
    if replicated < _closure_MIN_BROAD_REPLICATED_HOLDOUT_ENSEMBLES:
        blockers.append("need_at_least_three_replicated_holdout_ensembles")
    if not holdout_promotion_passed:
        blockers.append("vmec_boozer_production_scope_holdout_missing")
    if not scoped_guard_passed:
        blockers.append("scoped_production_nonlinear_guard_failed")
    if not broad_matrix_portfolio_passed:
        blockers.append("broad_nonlinear_transport_matrix_portfolio_missing_or_failed")

    scoped_completion = (
        _closure__bool_score(scoped_guard_passed, 25.0)
        + _closure__bool_score(qa_long_window_anchor, 10.0)
        + _closure__ratio_score(
            matched, _closure_MIN_BROAD_MATCHED_OPTIMIZATION_AUDITS, 20.0
        )
        + _closure__ratio_score(
            optimized, _closure_MIN_BROAD_OPTIMIZED_EQUILIBRIUM_ENSEMBLES, 15.0
        )
        + _closure__ratio_score(
            replicated, _closure_MIN_BROAD_REPLICATED_HOLDOUT_ENSEMBLES, 15.0
        )
        + _closure__bool_score(holdout_promotion_passed, 15.0)
    )
    completion = min(scoped_completion, 100.0) * 0.94 + _closure__bool_score(
        broad_matrix_portfolio_passed, 6.0
    )
    passed = bool(not blockers)

    return {
        "lane": "Broad end-to-end nonlinear turbulent-flux stellarator optimization",
        "status": _closure__lane_status(passed, blockers, completion),
        "passed": passed,
        "completion_percent": round(completion, 1),
        "claim_level": (
            "broad_nonlinear_turbulent_flux_optimization_ready"
            if passed
            else "scoped_positive_audits_pending_broad_matrix_portfolio"
        ),
        "primary_artifacts": [
            "docs/_static/production_nonlinear_optimization_guard.json",
            "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json",
            "docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.json",
            "docs/_static/vmec_jax_qa_transport_optimization_status.json",
            "docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json",
            "docs/_static/nonlinear_transport_matrix_portfolio.json",
        ],
        "key_metrics": {
            "scoped_guard_passed": scoped_guard_passed,
            "qa_long_window_anchor_passed": qa_long_window_anchor,
            "qualifying_matched_optimized_transport_audits": matched,
            "min_qualifying_matched_optimized_transport_audits": _closure_MIN_BROAD_MATCHED_OPTIMIZATION_AUDITS,
            "qualifying_optimized_equilibrium_ensembles": optimized,
            "min_qualifying_optimized_equilibrium_ensembles": _closure_MIN_BROAD_OPTIMIZED_EQUILIBRIUM_ENSEMBLES,
            "qualifying_replicated_holdout_ensembles": replicated,
            "min_qualifying_replicated_holdout_ensembles": _closure_MIN_BROAD_REPLICATED_HOLDOUT_ENSEMBLES,
            "vmec_boozer_holdout_promotion_passed": holdout_promotion_passed,
            "broad_matrix_portfolio_passed": broad_matrix_portfolio_passed,
        },
        "blockers": _closure__normalize_blockers(blockers),
        "required_next_artifacts": [
            "passing nonlinear_transport_matrix_portfolio.json built from at least one three-surface, two-field-line, three-ky matched matrix report",
            "matched long post-transient baseline-vs-optimized audits for at least three independent optimized equilibria",
            "surface/alpha held-out VMEC/Boozer production-scope nonlinear transport artifact",
            "replicated seed/timestep ensembles for each optimized equilibrium",
            "running-mean, block/SEM, spread, and finite-flux gates for all promoted windows",
        ],
        "next_action": (
            "Copy the passing broad matrix portfolio into docs/_static before manuscript-level nonlinear "
            "optimization promotion; scoped matched audits remain supporting evidence only."
            if not passed
            else "Use the selected broad matrix family as the manuscript-level nonlinear turbulent-flux optimization evidence."
        ),
    }


def _closure__domain_decomposition_lane(root: Path) -> dict[str, Any]:
    combined = _closure__read_json(
        root, "docs/_static/nonlinear_sharding_strong_scaling_large.json"
    )
    production = _closure__read_json(
        root, "docs/_static/nonlinear_sharding_production_speedup_gate.json"
    )
    domain_identity = _closure__read_json(
        root, "docs/_static/nonlinear_domain_parallel_identity_gate.json"
    )
    spectral_identity = _closure__read_json(
        root, "docs/_static/nonlinear_spectral_communication_identity_gate.json"
    )
    routed_profile = _closure__read_json(
        root, "docs/_static/nonlinear_spectral_domain_routing_profile.json"
    )
    decomposition_status = _closure__read_json(
        root, "docs/_static/parallel_decomposition_status.json"
    )

    rows = _closure__as_list((combined or {}).get("rows"))
    cpu_speedup = _closure__max_speedup(rows, "cpu")
    gpu_speedup = _closure__max_speedup(rows, "gpu")
    identity_passed = bool((combined or {}).get("identity_passed", False))
    strong_scaling_speedup_passed = bool((combined or {}).get("speedup_passed", False))
    production_passed = (
        bool((production or {}).get("passed", False))
        or str((production or {}).get("status", "")).lower() == "production_speedup"
    )
    domain_identity_passed = _closure__gate_bool(
        domain_identity, "gate", "identity_passed"
    )
    spectral_identity_passed = _closure__gate_bool(
        spectral_identity, "gate", "identity_passed"
    )
    routed_profile_identity_passed = bool(
        (routed_profile or {}).get("identity_passed", False)
    )
    routed_profile_speedup = _closure__finite_float(
        (routed_profile or {}).get("strong_speedup_vs_serial")
    )
    routed_profile_speedup_passed = bool(
        (routed_profile or {}).get("speedup_gate_passed", False)
    )
    routed_profile_work_model = _closure__as_dict(
        (routed_profile or {}).get("work_model")
    )
    routed_profile_work_model_present = bool(routed_profile_work_model)
    routed_profile_work_model_feasible = bool(
        routed_profile_work_model.get("production_speedup_feasible", False)
    )
    routed_profile_communication_ratio = _closure__finite_float(
        routed_profile_work_model.get("communication_to_owned_work_ratio")
    )
    routed_profile_efficiency_ceiling = _closure__finite_float(
        routed_profile_work_model.get("parallel_efficiency_ceiling")
    )
    decomposition_contract_passed = bool(
        (decomposition_status or {}).get("passed", False)
    )
    cpu_speedup_passed = (
        cpu_speedup is not None and cpu_speedup >= _closure_MIN_DOMAIN_CPU_SPEEDUP
    )
    gpu_speedup_passed = (
        gpu_speedup is not None and gpu_speedup >= _closure_MIN_DOMAIN_GPU_SPEEDUP
    )

    blockers: list[str] = []
    if not production_passed:
        blockers.append("production_speedup_gate_not_passed")
    if not strong_scaling_speedup_passed:
        blockers.extend(
            str(item)
            for item in _closure__as_list((combined or {}).get("speedup_blockers"))
        )
        blockers.append("combined_strong_scaling_speedup_not_passed")
    if not gpu_speedup_passed:
        blockers.append("gpu_domain_speedup_below_1p5")
    if not cpu_speedup_passed:
        blockers.append("cpu_domain_speedup_below_1p5")

    completion = (
        _closure__bool_score(domain_identity_passed, 15.0)
        + _closure__bool_score(spectral_identity_passed, 15.0)
        + _closure__bool_score(identity_passed, 15.0)
        + _closure__bool_score(routed_profile_identity_passed, 10.0)
        + _closure__bool_score(routed_profile_work_model_present, 5.0)
        + _closure__bool_score(decomposition_contract_passed, 10.0)
        + _closure__bool_score(cpu_speedup_passed, 10.0)
        + _closure__bool_score(gpu_speedup_passed, 10.0)
        + _closure__bool_score(strong_scaling_speedup_passed, 7.5)
        + _closure__bool_score(production_passed, 2.5)
    )
    passed = bool(
        domain_identity_passed
        and spectral_identity_passed
        and identity_passed
        and routed_profile_identity_passed
        and cpu_speedup_passed
        and gpu_speedup_passed
        and strong_scaling_speedup_passed
        and production_passed
    )

    return {
        "lane": "Production nonlinear domain-decomposition speedup",
        "status": _closure__lane_status(passed, blockers, completion),
        "passed": passed,
        "completion_percent": round(completion, 1),
        "claim_level": "identity_and_profiler_diagnostic_not_production_speedup"
        if not passed
        else "production_nonlinear_domain_speedup_ready",
        "primary_artifacts": [
            "docs/_static/nonlinear_domain_parallel_identity_gate.json",
            "docs/_static/nonlinear_spectral_communication_identity_gate.json",
            "docs/_static/nonlinear_spectral_domain_routing_profile.json",
            "docs/_static/nonlinear_sharding_strong_scaling_large.json",
            "docs/_static/nonlinear_sharding_production_speedup_gate.json",
            "docs/_static/parallel_decomposition_status.json",
        ],
        "key_metrics": {
            "domain_identity_passed": domain_identity_passed,
            "spectral_identity_passed": spectral_identity_passed,
            "combined_identity_passed": identity_passed,
            "routed_domain_timing_identity_passed": routed_profile_identity_passed,
            "routed_domain_timing_speedup": routed_profile_speedup,
            "routed_domain_timing_speedup_gate_passed": routed_profile_speedup_passed,
            "routed_domain_work_model_present": routed_profile_work_model_present,
            "routed_domain_work_model_speedup_feasible": routed_profile_work_model_feasible,
            "routed_domain_communication_to_owned_work_ratio": routed_profile_communication_ratio,
            "routed_domain_parallel_efficiency_ceiling": routed_profile_efficiency_ceiling,
            "parallel_decomposition_contract_passed": decomposition_contract_passed,
            "cpu_best_speedup": cpu_speedup,
            "cpu_speedup_gate": _closure_MIN_DOMAIN_CPU_SPEEDUP,
            "gpu_best_speedup": gpu_speedup,
            "gpu_speedup_gate": _closure_MIN_DOMAIN_GPU_SPEEDUP,
            "strong_scaling_speedup_passed": strong_scaling_speedup_passed,
            "production_gate_passed": production_passed,
        },
        "blockers": _closure__normalize_blockers(blockers),
        "required_next_artifacts": [
            "real communication-aware nonlinear domain decomposition in the production RHS/integrator path",
            "transport-window identity gates comparing serial and decomposed trajectories on a nonlinear case",
            "large-grid CPU and multi-GPU strong-scaling artifacts with speedup >= 1.5 on each backend",
            "profiler traces proving the communication overhead is below the saved RHS work",
        ],
        "next_action": (
            "Keep independent-work batching as the production path; for nonlinear domains, implement a real decomposed "
            "RHS/integrator route and require identity plus CPU/GPU speedup before any manuscript claim."
        ),
    }


def _closure__vmec_boozer_holdout_lane(root: Path) -> dict[str, Any]:
    promotion = _closure__read_json(
        root, "docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json"
    )
    alpha = _closure__read_json(
        root, "docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.json"
    )
    surface = _closure__read_json(
        root, "docs/_static/vmec_boozer_aggregate_surface_holdout_gate.json"
    )
    second_eq = _closure__read_json(
        root, "docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.json"
    )
    gradient_matrix = _closure__read_json(
        root, "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
    )
    ql_grad = _closure__read_json(
        root, "docs/_static/vmec_boozer_quasilinear_gradient_gate.json"
    )
    nonlinear_window_grad = _closure__read_json(
        root, "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json"
    )

    promotion_passed = bool((promotion or {}).get("passed", False))
    alpha_passed = bool((alpha or {}).get("passed", False))
    surface_passed = bool((surface or {}).get("passed", False))
    second_eq_passed = bool((second_eq or {}).get("passed", False))
    gradient_matrix_passed = bool((gradient_matrix or {}).get("passed", False))
    ql_grad_passed = bool((ql_grad or {}).get("passed", False))
    nonlinear_grad_passed = bool((nonlinear_window_grad or {}).get("passed", False))
    promotion_gate = _closure__as_dict((promotion or {}).get("promotion_gate"))
    holdout_artifacts = _closure__as_list((promotion or {}).get("holdout_artifacts"))
    qualifying_production_holdouts = sum(
        1
        for item in holdout_artifacts
        if isinstance(item, dict) and bool(item.get("qualifies_for_promotion", False))
    )
    blockers = [str(item) for item in _closure__as_list(promotion_gate.get("blockers"))]
    if (
        not promotion_passed
        and "aggregate_holdout_promotion_gate_failed" not in blockers
    ):
        blockers.append("aggregate_holdout_promotion_gate_failed")
    if qualifying_production_holdouts <= 0:
        blockers.append("no_production_scope_heldout_surface_or_alpha_artifact")

    completion = (
        _closure__bool_score(ql_grad_passed, 12.0)
        + _closure__bool_score(nonlinear_grad_passed, 12.0)
        + _closure__bool_score(gradient_matrix_passed, 14.0)
        + _closure__bool_score(alpha_passed, 13.0)
        + _closure__bool_score(surface_passed, 13.0)
        + _closure__bool_score(second_eq_passed, 14.0)
        + _closure__ratio_score(qualifying_production_holdouts, 1, 10.0)
        + _closure__bool_score(promotion_passed, 12.0)
    )
    passed = bool(promotion_passed and qualifying_production_holdouts >= 1)
    required_next_artifacts = (
        [
            "VMEC/Boozer nonlinear transport-gradient or robust finite-difference gate on the held-out split",
            "second-equilibrium heldout nonlinear transport validation before broad geometry-optimization claims",
            "same-WOUT provenance linking optimizer state, Boozer transform, SPECTRAX-GK input, and nonlinear audit output",
        ]
        if passed
        else [
            "production-scope held-out surface or field-line artifact using long post-transient nonlinear transport, not reduced growth/QL objectives",
            "VMEC/Boozer nonlinear transport-gradient or robust finite-difference gate on the held-out split",
            "second-equilibrium heldout nonlinear transport validation before broad geometry-optimization claims",
            "same-WOUT provenance linking optimizer state, Boozer transform, SPECTRAX-GK input, and nonlinear audit output",
        ]
    )
    next_action = (
        "VMEC/Boozer held-out nonlinear transport is closed for the current pre-manuscript gate; extend to nonlinear "
        "transport-gradient and second-equilibrium nonlinear transport before broader optimization claims."
        if passed
        else (
            "Promote the existing reduced alpha/surface/second-equilibrium gates only as plumbing; add a true production-scope "
            "heldout nonlinear transport artifact before claiming VMEC/Boozer optimization closure."
        )
    )

    return {
        "lane": "VMEC/Boozer holdout optimization",
        "status": _closure__lane_status(passed, blockers, completion),
        "passed": passed,
        "completion_percent": round(completion, 1),
        "claim_level": "reduced_holdout_plumbing_not_production_optimization"
        if not passed
        else "vmec_boozer_holdout_optimization_ready",
        "primary_artifacts": [
            "docs/_static/vmec_boozer_quasilinear_gradient_gate.json",
            "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json",
            "docs/_static/vmec_boozer_gradient_holdout_matrix.json",
            "docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.json",
            "docs/_static/vmec_boozer_aggregate_surface_holdout_gate.json",
            "docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.json",
            "docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json",
        ],
        "key_metrics": {
            "vmec_boozer_quasilinear_gradient_gate_passed": ql_grad_passed,
            "vmec_boozer_nonlinear_window_gradient_gate_passed": nonlinear_grad_passed,
            "gradient_holdout_matrix_passed": gradient_matrix_passed,
            "reduced_alpha_holdout_passed": alpha_passed,
            "reduced_surface_holdout_passed": surface_passed,
            "second_equilibrium_reduced_gate_passed": second_eq_passed,
            "qualifying_production_holdout_artifacts": qualifying_production_holdouts,
            "promotion_gate_passed": promotion_passed,
            "promotion_gate_blockers": _closure__as_list(
                promotion_gate.get("blockers")
            ),
        },
        "blockers": _closure__normalize_blockers(blockers),
        "required_next_artifacts": required_next_artifacts,
        "next_action": next_action,
    }


def _closure_build_status_payload(root: Path = _closure_REPO_ROOT) -> dict[str, Any]:
    """Return a JSON-ready strict pre-manuscript closure payload."""

    root = Path(root)
    lanes = [
        _closure__scoped_core_ql_lane(root),
        _closure__broad_nonlinear_optimization_lane(root),
        _closure__domain_decomposition_lane(root),
        _closure__vmec_boozer_holdout_lane(root),
    ]
    ready = all(bool(lane["passed"]) for lane in lanes)
    mean_completion = float(
        np.mean([float(lane["completion_percent"]) for lane in lanes])
    )
    return {
        "kind": "pre_manuscript_closure_status",
        "claim_scope": (
            "strict manuscript-blocking closure gates; the QL lane closes only at the declared core-portfolio "
            "scope, while broad nonlinear optimization and nonlinear speedup still require production evidence"
        ),
        "status_order": _closure_STATUS_ORDER,
        "lanes": lanes,
        "summary": {
            "ready_for_manuscript_drafting": ready,
            "n_lanes": len(lanes),
            "n_closed": sum(1 for lane in lanes if lane["status"] == "closed"),
            "n_partial": sum(1 for lane in lanes if lane["status"] == "partial"),
            "n_open": sum(1 for lane in lanes if lane["status"] == "open"),
            "n_blocked": sum(1 for lane in lanes if lane["status"] == "blocked"),
            "mean_completion_percent": round(mean_completion, 1),
            "blocking_lanes": [lane["lane"] for lane in lanes if not lane["passed"]],
        },
    }


def _closure_write_status_artifacts(
    payload: dict[str, Any], *, out: Path = _closure_DEFAULT_OUT
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the closure payload."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    csv_path = out.with_suffix(".csv")
    pdf_path = out.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_closure__json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "lane",
        "status",
        "completion_percent",
        "claim_level",
        "blockers",
        "next_action",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for lane in payload["lanes"]:
            writer.writerow(
                {
                    "lane": lane["lane"],
                    "status": lane["status"],
                    "completion_percent": lane["completion_percent"],
                    "claim_level": lane["claim_level"],
                    "blockers": ";".join(lane["blockers"]),
                    "next_action": lane["next_action"],
                }
            )

    set_plot_style()
    lanes = list(payload["lanes"])
    y = np.arange(len(lanes))
    completion = [float(lane["completion_percent"]) for lane in lanes]
    colors = [
        _closure_STATUS_COLORS.get(str(lane["status"]), "#777777") for lane in lanes
    ]
    labels = [textwrap.fill(str(lane["lane"]), width=34) for lane in lanes]

    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.barh(y, completion, color=colors, edgecolor="#333333", alpha=0.95)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("strict pre-manuscript closure (%)")
    ax.set_title("Pre-manuscript closure status: scoped and production gates")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    for yi, lane, value in zip(y, lanes, completion, strict=True):
        blockers = lane.get("blockers", [])
        blocker_text = "closed" if lane.get("passed") else f"{len(blockers)} blocker(s)"
        ax.text(
            min(value + 1.2, 98.0),
            float(yi),
            f"{value:.1f}% | {lane['status']} | {blocker_text}",
            va="center",
            ha="left" if value < 82 else "right",
            fontsize=8.0,
            color="#222222",
        )
    summary = payload.get("summary", {})
    caption = (
        f"Ready for manuscript drafting: {summary.get('ready_for_manuscript_drafting')}. "
        f"Mean strict closure: {float(summary.get('mean_completion_percent', 0.0)):.1f}%. "
        "Closed scoped diagnostics do not imply universal stress-case or production-speedup promotion."
    )
    fig.text(0.5, 0.025, caption, fontsize=8.4, color="#333333", ha="center")
    fig.subplots_adjust(left=0.34, right=0.97, top=0.86, bottom=0.18)
    fig.savefig(out, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "png": str(out),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


_closure_DEFAULT_RUNBOOK_OUT = (
    _closure_REPO_ROOT / "docs" / "_static" / "pre_manuscript_closure_runbook.png"
)
_closure_DEFAULT_INVENTORY = (
    _closure_REPO_ROOT / "docs" / "_static" / "vmec_jax_equilibrium_inventory.json"
)
_closure_DEFAULT_SCREEN = (
    _closure_REPO_ROOT
    / "docs"
    / "_static"
    / "external_vmec_candidate_linear_screen.csv"
)
_closure_DEFAULT_EXTERNAL_RUNBOOK = (
    _closure_REPO_ROOT / "docs" / "_static" / "external_vmec_next_holdout_runbook.json"
)
_closure_DEFAULT_HOLDOUT_GAP = (
    _closure_REPO_ROOT / "docs" / "_static" / "quasilinear_holdout_gap_report.json"
)
_closure_DEFAULT_OPTIMIZER_MANIFEST = (
    _closure_REPO_ROOT
    / "docs"
    / "_static"
    / "vmec_jax_qa_optimizer_comparison_manifest.json"
)
_closure_DEFAULT_LADDER_STATUS = (
    _closure_REPO_ROOT
    / "docs"
    / "_static"
    / "vmec_jax_qa_optimizer_ladder_resume_status.json"
)
_closure_DEFAULT_OFFICE_ROOT = Path(
    "/home/rjorge/spectrax_optimizer_ladder_20260609/SPECTRAX-GK"
)
_closure_DEFAULT_AUDIT_ROOT = Path("tools_out/pre_manuscript_nonlinear_audits")
_closure_MIN_LINEAR_LAUNCH_GAMMA = 0.02


def _closure__read_json_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _closure__slug(value: object) -> str:
    text = str(value).lower()
    text = text.removeprefix("wout_").removesuffix(".nc").removesuffix("_nc")
    return re.sub(r"[^a-z0-9]+", "", text)


def _closure__screen_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _closure__screened_slugs(rows: list[dict[str, str]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        for key in ("case", "vmec_file", "source", "geometry"):
            raw = row.get(key, "")
            if raw:
                out.add(_closure__slug(Path(raw).name))
                out.add(_closure__slug(raw))
    return out


def _closure__screen_expansion_candidates(
    *,
    inventory: dict[str, Any],
    screen_rows: list[dict[str, str]],
    max_candidates: int,
) -> list[dict[str, Any]]:
    screened = _closure__screened_slugs(screen_rows)
    rows = [row for row in inventory.get("rows", []) if isinstance(row, dict)]
    candidates: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name", ""))
        if not name or _closure__slug(name) in screened:
            continue
        if not bool(row.get("reference_scale_valid", False)):
            continue
        family = str(row.get("family", ""))
        if family in {"axisymmetric"} and any(
            "tokamak" in str(row.get("name", "")).lower() for _ in (0,)
        ):
            priority_note = "axisymmetric reserve candidate; use only if stellarator candidates fail screen"
        else:
            priority_note = "independent VMEC candidate requiring linear screen before nonlinear launch"
        candidates.append(
            {
                "name": name,
                "path": row.get("path", name),
                "family": family,
                "aspect": row.get("aspect"),
                "iota_edge": row.get("iota_edge"),
                "candidate_score": row.get("candidate_score", 0.0),
                "priority_note": priority_note,
                "next_required_gate": "linear_ky_screen_with_gamma_ge_0p02_before_nonlinear_holdout_launch",
            }
        )
    candidates.sort(
        key=lambda item: (-float(item.get("candidate_score") or 0.0), str(item["name"]))
    )
    return candidates[:max_candidates]


def _closure__candidate_admitted_holdout(
    selected: dict[str, Any], holdout_gap: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the admitted nonlinear holdout matching a selected runbook row."""

    selected_slugs = {
        _closure__slug(value)
        for key in ("case", "family", "geometry", "vmec_file")
        if (value := selected.get(key))
    }
    selected_slugs.discard("")
    if not selected_slugs:
        return None
    for row in holdout_gap.get("admitted_holdouts", []):
        if not isinstance(row, dict):
            continue
        if not bool(row.get("gate_passed", False)):
            continue
        row_slugs = {
            _closure__slug(value)
            for key in ("case", "case_label", "geometry", "nonlinear_artifact")
            if (value := row.get(key))
        }
        row_slugs.discard("")
        if any(
            row_slug == selected_slug
            or row_slug.startswith(selected_slug)
            or selected_slug.startswith(row_slug)
            for selected_slug in selected_slugs
            for row_slug in row_slugs
        ):
            return row
    return None


def _closure__optimizer_audit_commands(
    *,
    office_root: Path,
    audit_root: Path,
    names: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        wout = (
            office_root
            / "tools_out"
            / "vmec_jax_qa_optimizer_ladder_20260609"
            / "runs"
            / name
            / "wout_final_rerun.nc"
        )
        out_dir = audit_root / name
        case = f"premanuscript_{name}"
        generate = (
            "python3 tools/campaigns/write_optimized_equilibrium_transport_configs.py "
            f"--vmec-file {wout.relative_to(office_root).as_posix()} "
            f"--case {case} "
            f"--out-dir {out_dir.as_posix()} "
            "--horizons 700,1100,1500 --grid n64:64:64:40:40 "
            "--torflux 0.64 --alpha 0.0 --npol 1.0 "
            "--window-tmin 1100 --window-tmax 1500 "
            "--dt-variant 0.04 --seed-variant 32 --seed-variant 33"
        )
        seed32 = (
            "CUDA_VISIBLE_DEVICES=0 python3 -m spectraxgk.cli run-runtime-nonlinear "
            f"--config {out_dir / f'{case}_nonlinear_t1500_n64_seed32.toml'} "
            "--steps 30000 --no-progress"
        )
        seed33 = (
            "CUDA_VISIBLE_DEVICES=1 python3 -m spectraxgk.cli run-runtime-nonlinear "
            f"--config {out_dir / f'{case}_nonlinear_t1500_n64_seed33.toml'} "
            "--steps 30000 --no-progress"
        )
        dt_variant = (
            "CUDA_VISIBLE_DEVICES=${DEVICE:-0} python3 -m spectraxgk.cli run-runtime-nonlinear "
            f"--config {out_dir / f'{case}_nonlinear_t1500_n64_dt0p04.toml'} "
            "--steps 37500 --no-progress"
        )
        rows.append(
            {
                "rank": idx + 1,
                "optimizer_case": name,
                "office_root": office_root.as_posix(),
                "wout": wout.relative_to(office_root).as_posix(),
                "manifest": (out_dir / "run_manifest.json").as_posix(),
                "generate_configs_command": generate,
                "seed_launch_commands": [seed32, seed33],
                "dt_variant_followup_command": dt_variant,
                "window": [1100.0, 1500.0],
                "claim_level": "launch_contract_or_running_audit_not_transport_promotion",
            }
        )
    return rows


def _closure__vmec_boozer_holdout_transport_commands(
    *,
    office_root: Path,
    out_root: Path,
) -> list[dict[str, Any]]:
    """Return production-scope held-out VMEC/Boozer nonlinear launch contracts."""

    wout = Path("/home/rjorge/src/vmec_jax/examples/data/wout_nfp4_QH_warm_start.nc")
    case = "vmec_boozer_qh_torflux078_alpha120_holdout"
    out_dir = out_root / case
    generate = (
        "python3 tools/campaigns/write_optimized_equilibrium_transport_configs.py "
        f"--vmec-file {wout.as_posix()} "
        f"--case {case} "
        f"--out-dir {out_dir.as_posix()} "
        "--horizons 250,350,450,700 --grid n64:64:64:40:40 "
        "--torflux 0.78 --alpha 1.2 --npol 1.0 --ky 0.2 "
        "--window-tmin 350 --window-tmax 700 "
        "--dt-variant 0.04 --seed-variant 31 --seed-variant 32"
    )
    outputs = [
        out_dir / f"{case}_nonlinear_t700_n64_seed31.out.nc",
        out_dir / f"{case}_nonlinear_t700_n64_seed32.out.nc",
        out_dir / f"{case}_nonlinear_t700_n64_dt0p04.out.nc",
    ]
    artifact_dir = office_root / "docs" / "_static" / "vmec_boozer_holdout_transport"
    ensemble_json = artifact_dir / f"{case}_ensemble_gate.json"
    readiness_json = artifact_dir / f"{case}_readiness.json"
    ensemble_png = artifact_dir / f"{case}_ensemble_gate.png"
    holdout_json = artifact_dir / f"{case}_production_holdout.json"
    output_gate_json = artifact_dir / f"{case}_output_gate.json"
    output_gate_command = (
        "python3 tools/release/check_nonlinear_runtime_outputs.py "
        + " ".join(path.as_posix() for path in outputs)
        + " --min-samples 200 --tmin 350 --tmax 700 --min-window-samples 80 "
        f"--min-abs-window-mean 0.0001 --json-out {output_gate_json.as_posix()}"
    )
    build_ensemble_command = (
        "python3 tools/artifacts/build_external_vmec_replicate_ensemble.py "
        + " ".join(path.as_posix() for path in outputs)
        + f" --out-dir {artifact_dir.as_posix()}"
        + f" --case {case}_replicated_nonlinear_window"
        + " --tmin 350 --tmax 700"
        + " --artifact-prefix docs/_static/vmec_boozer_holdout_transport"
        + f" --readiness-json {readiness_json.name}"
        + f" --ensemble-json {ensemble_json.name}"
        + f" --out-png {ensemble_png.name}"
    )
    build_holdout_artifact_command = (
        "python3 tools/artifacts/build_vmec_boozer_aggregate_holdout_gate.py production "
        f"--transport-manifest {(out_dir / 'run_manifest.json').as_posix()} "
        f"--ensemble-json {ensemble_json.as_posix()} "
        f"--case {case} --out {holdout_json.as_posix()}"
    )
    promotion_gate_command = (
        "python3 tools/release/check_vmec_boozer_aggregate_holdout_gate.py "
        "--holdout-artifact docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.json "
        "--holdout-artifact docs/_static/vmec_boozer_aggregate_surface_holdout_gate.json "
        f"--holdout-artifact {holdout_json.as_posix()} "
        "--nonlinear-ensemble-artifact docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.json "
        "--nonlinear-ensemble-artifact docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json "
        f"--nonlinear-ensemble-artifact {ensemble_json.as_posix()} "
        "--json-out docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json"
    )
    return [
        {
            "case": case,
            "wout": wout.as_posix(),
            "transport_sample": {
                "torflux": 0.78,
                "alpha": 1.2,
                "ky": 0.2,
                "npol": 1.0,
                "role": "heldout_surface_field_line_transport",
            },
            "manifest": (out_dir / "run_manifest.json").as_posix(),
            "generate_configs_command": generate,
            "direct_full_horizon_launch_commands": [
                (
                    f"CUDA_VISIBLE_DEVICES={device} python3 -m spectraxgk.cli run-runtime-nonlinear "
                    f"--config {out_dir / f'{case}_nonlinear_t700_n64_{variant}.toml'} "
                    f"--steps {steps} --no-progress"
                )
                for device, variant, steps in (
                    (0, "seed31", 14000),
                    (1, "seed32", 14000),
                    ("${DEVICE:-0}", "dt0p04", 17500),
                )
            ],
            "expected_outputs": [path.as_posix() for path in outputs],
            "output_gate_command": output_gate_command,
            "build_ensemble_command": build_ensemble_command,
            "build_holdout_artifact_command": build_holdout_artifact_command,
            "promotion_gate_command": promotion_gate_command,
            "postprocess_artifacts": {
                "output_gate": output_gate_json.as_posix(),
                "ensemble_json": ensemble_json.as_posix(),
                "readiness_json": readiness_json.as_posix(),
                "ensemble_png": ensemble_png.as_posix(),
                "production_holdout_json": holdout_json.as_posix(),
            },
            "window": [350.0, 700.0],
            "claim_level": (
                "production_scope_vmec_boozer_surface_field_line_launch_contract_not_transport_promotion"
            ),
        }
    ]


def _closure_build_runbook_payload(
    *,
    root: Path = _closure_REPO_ROOT,
    inventory_path: Path = _closure_DEFAULT_INVENTORY,
    screen_path: Path = _closure_DEFAULT_SCREEN,
    external_runbook_path: Path = _closure_DEFAULT_EXTERNAL_RUNBOOK,
    holdout_gap_path: Path | None = None,
    optimizer_manifest_path: Path = _closure_DEFAULT_OPTIMIZER_MANIFEST,
    ladder_status_path: Path = _closure_DEFAULT_LADDER_STATUS,
    office_root: Path = _closure_DEFAULT_OFFICE_ROOT,
    audit_root: Path = _closure_DEFAULT_AUDIT_ROOT,
    max_screen_candidates: int = 8,
) -> dict[str, Any]:
    """Return a JSON-ready pre-manuscript action runbook."""

    status = _closure_build_status_payload(root)
    inventory = _closure__read_json_path(inventory_path)
    screen_rows = _closure__screen_rows(screen_path)
    external_runbook = _closure__read_json_path(external_runbook_path)
    holdout_gap = _closure__read_json_path(
        holdout_gap_path or (root / "docs/_static/quasilinear_holdout_gap_report.json")
    )
    optimizer_manifest = _closure__read_json_path(optimizer_manifest_path)
    ladder_status = _closure__read_json_path(ladder_status_path)
    screen_candidates = _closure__screen_expansion_candidates(
        inventory=inventory,
        screen_rows=screen_rows,
        max_candidates=max_screen_candidates,
    )
    optimizer_names = (
        "growth_scalar_trust_from_strict_baseline",
        "growth_lbfgs_adjoint_from_strict_baseline",
        "quasilinear_scalar_trust_from_strict_baseline",
    )
    optimizer_audits = _closure__optimizer_audit_commands(
        office_root=office_root,
        audit_root=audit_root,
        names=optimizer_names,
    )
    heldout_transport = _closure__vmec_boozer_holdout_transport_commands(
        office_root=office_root,
        out_root=office_root / "tools_out" / "vmec_boozer_holdout_transport",
    )
    external_has_launch = bool(external_runbook.get("passed", False)) and bool(
        external_runbook.get("launch_commands")
    )
    external_next_action = (
        "Launch or harvest the selected nonlinear holdout campaign, then admit it only through "
        "grid/window convergence, replicated post-transient transport, and QL recalibration gates."
        if external_has_launch
        else (
            "Run a linear ky screen on the listed unscreened VMEC candidates; only candidates "
            "with gamma >= 0.02 and valid flux-tube metrics may enter the nonlinear holdout runbook."
        )
    )
    selected_external = (
        external_runbook.get("selected_new_family_candidate")
        or external_runbook.get("selected_preferred_family_audit")
        or {}
    )
    admitted_external = _closure__candidate_admitted_holdout(
        selected_external, holdout_gap
    )
    if admitted_external:
        external_status = "harvested_admitted"
        external_next_action = (
            "Selected external-VMEC candidate is already harvested and admitted as nonlinear "
            "holdout evidence; use the holdout-gap report to choose the next independent "
            "candidate rather than relaunching this case."
        )
        external_launch_commands: list[Any] = []
    else:
        external_status = (
            "blocked_on_new_linear_screen" if not external_has_launch else "launchable"
        )
        external_launch_commands = external_runbook.get("launch_commands", [])
    lanes = {
        str(lane["lane"]): lane
        for lane in status.get("lanes", [])
        if isinstance(lane, dict)
    }
    ql_lane = lanes.get("Scoped core quasilinear heat-flux diagnostic", {})
    ql_scoped_closed = bool(ql_lane.get("passed", False))
    if ql_scoped_closed:
        external_status = "frozen_scoped_ql_closed"
        external_next_action = (
            "No additional QL holdout launches are active for this tranche: use the closed scoped-core "
            "diagnostic and keep declared stress outliers deferred until a new saturation-physics lane is opened."
        )
        external_launch_commands = []
    domain_lane = lanes.get("Production nonlinear domain-decomposition speedup", {})
    payload = {
        "kind": "pre_manuscript_closure_runbook",
        "claim_scope": (
            "actionable campaign runbook only; generated or launched commands do not promote "
            "absolute quasilinear, broad nonlinear optimization, VMEC/Boozer optimization, or "
            "production nonlinear parallel speedup claims without the strict gates"
        ),
        "status_summary": status.get("summary", {}),
        "external_vmec_holdout_campaign": {
            "status": external_status,
            "external_runbook": external_runbook_path.relative_to(root).as_posix(),
            "external_runbook_passed": bool(external_runbook.get("passed", False)),
            "holdout_gap_report": (
                holdout_gap_path
                or (root / "docs/_static/quasilinear_holdout_gap_report.json")
            )
            .relative_to(root)
            .as_posix(),
            "min_launch_gamma": float(
                external_runbook.get(
                    "min_launch_gamma", _closure_MIN_LINEAR_LAUNCH_GAMMA
                )
            ),
            "screen_rows": len(screen_rows),
            "inventory_equilibria": int(inventory.get("n_equilibria", 0) or 0),
            "selected_candidate": selected_external,
            "admitted_holdout": admitted_external,
            "launch_commands": external_launch_commands,
            "unscreened_candidates": screen_candidates,
            "next_action": external_next_action,
        },
        "vmec_boozer_production_scope_artifacts": {
            "status": "launch_contracts_generated_on_office",
            "purpose": (
                "three matched long-window optimized-equilibrium nonlinear audits for broad "
                "VMEC/Boozer and nonlinear turbulent-flux optimization evidence, plus a held-out "
                "VMEC/Boozer surface/field-line nonlinear transport audit"
            ),
            "optimizer_manifest": optimizer_manifest_path.relative_to(root).as_posix(),
            "optimizer_manifest_entries": len(
                optimizer_manifest.get("entries", []) or []
            ),
            "ladder_status": ladder_status_path.relative_to(root).as_posix(),
            "ladder_commands": len(ladder_status.get("commands", []) or []),
            "audit_commands": optimizer_audits,
            "heldout_transport_commands": heldout_transport,
            "office_seed_queue": {
                "launched": True,
                "pids": [3402448, 3402449],
                "queues": [
                    "GPU0 seed32: growth_scalar_trust -> growth_lbfgs_adjoint -> quasilinear_scalar_trust",
                    "GPU1 seed33: growth_scalar_trust -> growth_lbfgs_adjoint -> quasilinear_scalar_trust",
                ],
                "dt_variant_policy": "launch dt=0.04 variants after seed outputs are finite and logs show no NaNs",
            },
        },
        "nonlinear_optimization_audit_extension": {
            "status": "running_or_launchable",
            "required_count": 3,
            "candidate_count": len(optimizer_audits),
            "window": [1100.0, 1500.0],
            "acceptance": (
                "each candidate must pass finite-output, long-window sample count, running-mean/block-SEM, "
                "seed/timestep ensemble, and matched baseline-vs-optimized transport gates"
            ),
        },
        "nonlinear_domain_decomposition": {
            "status": "identity_route_extended_no_speedup_claim",
            "completion_percent": domain_lane.get("completion_percent"),
            "artifact": "docs/_static/nonlinear_spectral_communication_identity_gate.json",
            "new_gate": "logical_sharded_nonlinear_spectral_integrator_identity",
            "next_action": (
                "Replace the local logical tile route with real device communication/distributed FFT routing, "
                "then require serial-vs-decomposed transport-window identity and CPU/GPU speedup >= 1.5."
            ),
        },
    }
    payload["overall_next_actions"] = [
        "Harvest office t=1500 seed audit logs and outputs; launch dt=0.04 variants only if seed outputs are finite.",
        external_next_action,
        (
            "Use the closed scoped-core QL diagnostic for examples and optimization screening; "
            "do not launch additional QL holdouts for this release tranche."
        ),
        "Keep nonlinear decomposition as identity-only until production distributed routing and profiler speedup gates pass.",
    ]
    return _closure__json_clean(payload)


def _closure_write_runbook_artifacts(
    payload: dict[str, Any], *, out: Path = _closure_DEFAULT_RUNBOOK_OUT
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the action runbook."""

    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    csv_path = out.with_suffix(".csv")
    pdf_path = out.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = [
        {
            "lane": "external_vmec_holdout_campaign",
            "status": payload["external_vmec_holdout_campaign"]["status"],
            "next_action": payload["external_vmec_holdout_campaign"]["next_action"],
        },
        {
            "lane": "vmec_boozer_production_scope_artifacts",
            "status": payload["vmec_boozer_production_scope_artifacts"]["status"],
            "next_action": "harvest launched office t=1500 audits and build ensemble gates",
        },
        {
            "lane": "nonlinear_optimization_audit_extension",
            "status": payload["nonlinear_optimization_audit_extension"]["status"],
            "next_action": payload["nonlinear_optimization_audit_extension"][
                "acceptance"
            ],
        },
        {
            "lane": "nonlinear_domain_decomposition",
            "status": payload["nonlinear_domain_decomposition"]["status"],
            "next_action": payload["nonlinear_domain_decomposition"]["next_action"],
        },
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("lane", "status", "next_action"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    labels = [row["lane"].replace("_", "\n") for row in rows]
    status_to_score = {
        "launchable": 0.75,
        "harvested_admitted": 0.85,
        "running_or_launchable": 0.72,
        "launch_contracts_generated_on_office": 0.68,
        "identity_route_extended_no_speedup_claim": 0.62,
        "blocked_on_new_linear_screen": 0.35,
    }
    scores = [status_to_score.get(row["status"], 0.5) * 100.0 for row in rows]
    colors = ["#d89c32" if "blocked" in row["status"] else "#2f7f5f" for row in rows]
    fig, ax = plt.subplots(figsize=(12.2, 5.4))
    x = np.arange(len(rows))
    ax.bar(x, scores, color=colors, edgecolor="#333333")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("actionability score (%)")
    ax.set_title("Pre-manuscript action runbook")
    ax.grid(axis="y", alpha=0.25)
    for xi, score, row in zip(x, scores, rows, strict=True):
        ax.text(
            xi,
            score + 2.0,
            row["status"],
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=12,
        )
    fig.text(
        0.5,
        0.035,
        "Actionability is not claim closure: strict promotion gates remain authoritative.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.34)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": str(out),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


# Public aliases for tests, docs tooling, and campaign finalizers. The names are
# intentionally domain-specific to avoid another ambiguous ``build_status`` API.
OPEN_STATUS_ORDER = _open_STATUS_ORDER
OPEN_STATUS_COLORS = _open_STATUS_COLORS
MANUSCRIPT_READINESS_STATUS_ORDER = _readiness_STATUS_ORDER
MANUSCRIPT_READINESS_STATUS_COLORS = _readiness_STATUS_COLORS
PRE_MANUSCRIPT_CLOSURE_STATUS_ORDER = _closure_STATUS_ORDER
PRE_MANUSCRIPT_CLOSURE_STATUS_COLORS = _closure_STATUS_COLORS

build_open_research_lane_payload = _open_build_status_payload
write_open_research_lane_artifacts = _open_write_status_artifacts
build_manuscript_readiness_payload = _readiness_build_manuscript_readiness_payload
write_manuscript_readiness_artifacts = _readiness_write_manuscript_readiness_artifacts
build_pre_manuscript_closure_payload = _closure_build_status_payload
write_pre_manuscript_closure_artifacts = _closure_write_status_artifacts
build_pre_manuscript_runbook_payload = _closure_build_runbook_payload
write_pre_manuscript_runbook_artifacts = _closure_write_runbook_artifacts
json_clean = _open__json_clean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = False

    open_parser = subparsers.add_parser(
        "open-lanes", help="Build the broad open research lane status dashboard."
    )
    open_parser.add_argument("--root", type=Path, default=_open_REPO_ROOT)
    open_parser.add_argument("--out", type=Path, default=_open_DEFAULT_OUT)
    open_parser.add_argument("--json-only", action="store_true")

    readiness_parser = subparsers.add_parser(
        "manuscript-readiness", help="Build the scoped manuscript readiness dashboard."
    )
    readiness_parser.add_argument("--root", type=Path, default=_readiness_ROOT)
    readiness_parser.add_argument("--out", type=Path, default=_readiness_DEFAULT_OUT)
    readiness_parser.add_argument("--json-only", action="store_true")

    closure_parser = subparsers.add_parser(
        "pre-manuscript-closure",
        help="Build the strict pre-manuscript closure dashboard.",
    )
    closure_parser.add_argument("--root", type=Path, default=_closure_REPO_ROOT)
    closure_parser.add_argument("--out", type=Path, default=_closure_DEFAULT_OUT)
    closure_parser.add_argument("--json-only", action="store_true")

    runbook_parser = subparsers.add_parser(
        "runbook", help="Build the actionable pre-manuscript closure runbook."
    )
    runbook_parser.add_argument(
        "--out", type=Path, default=_closure_DEFAULT_RUNBOOK_OUT
    )
    runbook_parser.add_argument(
        "--inventory", type=Path, default=_closure_DEFAULT_INVENTORY
    )
    runbook_parser.add_argument("--screen", type=Path, default=_closure_DEFAULT_SCREEN)
    runbook_parser.add_argument(
        "--external-runbook", type=Path, default=_closure_DEFAULT_EXTERNAL_RUNBOOK
    )
    runbook_parser.add_argument("--holdout-gap", type=Path, default=None)
    runbook_parser.add_argument(
        "--optimizer-manifest", type=Path, default=_closure_DEFAULT_OPTIMIZER_MANIFEST
    )
    runbook_parser.add_argument(
        "--ladder-status", type=Path, default=_closure_DEFAULT_LADDER_STATUS
    )
    runbook_parser.add_argument(
        "--office-root", type=Path, default=_closure_DEFAULT_OFFICE_ROOT
    )
    runbook_parser.add_argument(
        "--audit-root", type=Path, default=_closure_DEFAULT_AUDIT_ROOT
    )
    runbook_parser.add_argument("--max-screen-candidates", type=int, default=8)
    return parser


def _print_paths(paths: dict[str, str]) -> int:
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    command = args.command or "open-lanes"

    if command == "open-lanes":
        payload = build_open_research_lane_payload(Path(args.root))
        if args.json_only:
            print(json.dumps(_open__json_clean(payload), indent=2, sort_keys=True))
            return 0
        return _print_paths(
            write_open_research_lane_artifacts(payload, out_png=Path(args.out))
        )

    if command == "manuscript-readiness":
        payload = build_manuscript_readiness_payload(Path(args.root))
        if args.json_only:
            print(json.dumps(_readiness__json_clean(payload), indent=2, sort_keys=True))
            return 0
        return _print_paths(write_manuscript_readiness_artifacts(payload, out=args.out))

    if command == "pre-manuscript-closure":
        payload = build_pre_manuscript_closure_payload(Path(args.root))
        if args.json_only:
            print(json.dumps(_closure__json_clean(payload), indent=2, sort_keys=True))
            return 0
        return _print_paths(
            write_pre_manuscript_closure_artifacts(payload, out=args.out)
        )

    if command == "runbook":
        payload = build_pre_manuscript_runbook_payload(
            inventory_path=args.inventory,
            screen_path=args.screen,
            external_runbook_path=args.external_runbook,
            holdout_gap_path=args.holdout_gap,
            optimizer_manifest_path=args.optimizer_manifest,
            ladder_status_path=args.ladder_status,
            office_root=args.office_root,
            audit_root=args.audit_root,
            max_screen_candidates=int(args.max_screen_candidates),
        )
        return _print_paths(
            write_pre_manuscript_runbook_artifacts(payload, out=args.out)
        )

    raise AssertionError(f"unhandled command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
