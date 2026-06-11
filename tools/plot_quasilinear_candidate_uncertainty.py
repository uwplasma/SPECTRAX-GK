#!/usr/bin/env python3
"""Score quasilinear saturation candidates with leave-one-out uncertainty intervals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.parallel import independent_map  # noqa: E402

from plot_quasilinear_saturation_rule_sweep import (  # noqa: E402
    DEFAULT_CASES,
    SaturationCase,
    raw_rule_estimates,
    require_validated_nonlinear_inputs,
)
from plot_quasilinear_shape_aware_saturation import (  # noqa: E402
    _observed_flux,
    fit_power_law_shape_exponent,
    shape_aware_raw_estimate,
)


CANDIDATE_LABELS = {
    "linear_weight": r"$\hat Q$ calibrated",
    "shape_power_law": r"shape power law",
    "spectral_envelope_ridge": r"spectral-envelope ridge",
    "linear_state_ridge": r"linear-state ridge",
}

STATE_FEATURE_NAMES = (
    "log_linear_weight",
    "log_abs_growth_mixing_length",
    "unstable_weight_fraction",
    "log_weighted_ky_centroid",
)

SPECTRAL_ENVELOPE_FEATURE_NAMES = (
    "log_positive_ky_centroid",
    "ky_weighted_std",
)


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


def _fit_scale(raw: np.ndarray, observed: np.ndarray, *, floor: float) -> float:
    finite = np.isfinite(raw) & np.isfinite(observed) & (np.abs(raw) > floor)
    if not np.any(finite):
        return float("nan")
    denom = float(np.dot(raw[finite], raw[finite]))
    if denom <= floor:
        return float("nan")
    return float(np.dot(raw[finite], observed[finite]) / denom)


def _linear_raw(case: SaturationCase) -> float:
    return float(raw_rule_estimates(case.spectrum)["linear_weight"])


def _load_spectrum_table(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    return data


def _required_column(data: np.ndarray, path: Path, column: str) -> np.ndarray:
    names = set(data.dtype.names or ())
    if column not in names:
        raise ValueError(f"{path} is missing required column '{column}'")
    return np.asarray(data[column], dtype=float)


def _state_feature_vector(case: SaturationCase, *, floor: float) -> np.ndarray:
    """Return linear-only branch/state features for transport-candidate scoring."""

    spectrum = Path(case.spectrum)
    data = _load_spectrum_table(spectrum)
    ky = _required_column(data, spectrum, "ky")
    gamma = _required_column(data, spectrum, "gamma")
    weight = np.maximum(_required_column(data, spectrum, "heat_flux_weight_total"), 0.0)
    finite = np.isfinite(ky) & np.isfinite(gamma) & np.isfinite(weight) & (ky > 0.0)
    if not np.any(finite):
        raise ValueError(f"{spectrum} contains no finite linear-state samples")
    ky = ky[finite]
    gamma = gamma[finite]
    weight = weight[finite]
    total_weight = float(np.sum(weight))
    rules = raw_rule_estimates(spectrum, floor=floor)
    if total_weight > floor:
        unstable_fraction = float(np.sum(weight[gamma > 0.0]) / total_weight)
        weighted_ky = float(np.sum(ky * weight) / total_weight)
    else:
        unstable_fraction = 0.0
        weighted_ky = float(np.mean(ky))
    return np.asarray(
        [
            math.log(max(float(rules["linear_weight"]), floor)),
            math.log(max(float(rules["absolute_growth_mixing_length"]), floor)),
            unstable_fraction,
            math.log(max(weighted_ky, floor)),
        ],
        dtype=float,
    )


def _spectral_envelope_feature_vector(case: SaturationCase, *, floor: float) -> np.ndarray:
    """Return reduced spectrum-shape features for bounded absolute-flux modeling."""

    spectrum = Path(case.spectrum)
    data = _load_spectrum_table(spectrum)
    ky = _required_column(data, spectrum, "ky")
    gamma = _required_column(data, spectrum, "gamma")
    weight = np.maximum(_required_column(data, spectrum, "heat_flux_weight_total"), 0.0)
    finite = np.isfinite(ky) & np.isfinite(gamma) & np.isfinite(weight) & (ky > 0.0)
    if not np.any(finite):
        raise ValueError(f"{spectrum} contains no finite linear-state samples")
    ky = ky[finite]
    gamma = gamma[finite]
    weight = weight[finite]
    positive = np.maximum(gamma, 0.0)
    positive_weight = positive if np.any(positive > floor) else np.ones_like(ky)
    positive_ky_centroid = float(np.sum(ky * positive_weight) / max(float(np.sum(positive_weight)), floor))
    if np.sum(weight) > floor:
        weight_mean_ky = float(np.sum(ky * weight) / float(np.sum(weight)))
        ky_weighted_std = math.sqrt(
            max(float(np.sum(weight * (ky - weight_mean_ky) ** 2) / float(np.sum(weight))), 0.0)
        )
    else:
        ky_weighted_std = 0.0
    return np.asarray(
        [
            math.log(max(positive_ky_centroid, floor)),
            ky_weighted_std,
        ],
        dtype=float,
    )


def _state_feature_matrix(cases: tuple[SaturationCase, ...], *, floor: float) -> np.ndarray:
    return np.vstack([_state_feature_vector(case, floor=floor) for case in cases])


def _spectral_envelope_feature_matrix(cases: tuple[SaturationCase, ...], *, floor: float) -> np.ndarray:
    return np.vstack([_spectral_envelope_feature_vector(case, floor=floor) for case in cases])


def _ridge_loglinear_holdout_row(
    *,
    case: SaturationCase,
    train_cases: tuple[SaturationCase, ...],
    features_all: np.ndarray,
    feature_names: tuple[str, ...],
    observed_all: np.ndarray,
    holdout_idx: int,
    train_indices: list[int],
    observed_floor: float,
    interval_z: float,
    ridge_lambda: float = 1.0,
    condition_gate: float = 1.0e6,
    min_train_to_parameter_ratio: float = 2.0,
) -> dict[str, Any]:
    """Fit a ridge log-linear model on training cases and score one holdout."""

    x_train_raw = np.asarray(features_all[train_indices], dtype=float)
    x_hold_raw = np.asarray(features_all[holdout_idx], dtype=float)
    feature_mean = np.mean(x_train_raw, axis=0)
    feature_scale = np.std(x_train_raw, axis=0, ddof=0)
    feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
    x_train = (x_train_raw - feature_mean) / feature_scale
    x_hold = (x_hold_raw - feature_mean) / feature_scale
    design = np.column_stack([np.ones(x_train.shape[0]), x_train])
    hold_design = np.concatenate([[1.0], x_hold])
    y_train = np.log(np.maximum(observed_all[train_indices], observed_floor))
    penalty = np.eye(design.shape[1]) * ridge_lambda
    penalty[0, 0] = 0.0
    normal_matrix = design.T @ design + penalty
    coef = np.linalg.solve(normal_matrix, design.T @ y_train)
    train_log_pred = design @ coef
    residual = y_train - train_log_pred
    residual_mean = float(np.mean(residual))
    residual_sigma_fit = float(np.std(residual, ddof=1)) if residual.size > 1 else 0.0
    parameters = design.shape[1]
    train_to_parameter_ratio = float(len(train_indices) / parameters)
    eligibility_failures = []
    if train_to_parameter_ratio < min_train_to_parameter_ratio:
        eligibility_failures.append("insufficient_train_to_parameter_ratio")
    condition_number = float(np.linalg.cond(normal_matrix))
    if condition_number > condition_gate:
        eligibility_failures.append("ill_conditioned_normal_matrix")
    # Under-sampled fits get a conservative residual floor so intervals do not imply false precision.
    residual_sigma = max(residual_sigma_fit, 1.0 if eligibility_failures else 0.0)
    holdout_log_pred = float(hold_design @ coef + residual_mean)
    predicted = float(math.exp(holdout_log_pred))
    lo = float(math.exp(holdout_log_pred - interval_z * residual_sigma))
    hi = float(math.exp(holdout_log_pred + interval_z * residual_sigma))
    holdout_observed = float(observed_all[holdout_idx])
    rel_error = abs(predicted - holdout_observed) / max(abs(holdout_observed), observed_floor)
    return {
        "holdout_case": case.case,
        "train_cases": [item.case for item in train_cases],
        "feature_names": list(feature_names),
        "feature_values": x_hold_raw.tolist(),
        "coefficients": coef.tolist(),
        "ridge_lambda": float(ridge_lambda),
        "condition_number": condition_number,
        "condition_gate": float(condition_gate),
        "n_train": int(len(train_indices)),
        "n_parameters": int(parameters),
        "train_to_parameter_ratio": train_to_parameter_ratio,
        "min_train_to_parameter_ratio": float(min_train_to_parameter_ratio),
        "promotion_eligible": not eligibility_failures,
        "eligibility_failures": eligibility_failures,
        "predicted_heat_flux": predicted,
        "observed_heat_flux": holdout_observed,
        "absolute_relative_error": float(rel_error),
        "prediction_interval_low": lo,
        "prediction_interval_high": hi,
        "prediction_interval_contains_observed": bool(lo <= holdout_observed <= hi),
        "train_log_residual_mean": residual_mean,
        "train_log_residual_sigma": residual_sigma,
        "train_log_residual_sigma_fit": residual_sigma_fit,
    }


def _candidate_raw_values(
    candidate: str,
    cases: tuple[SaturationCase, ...],
    *,
    train_cases: tuple[SaturationCase, ...],
    passed_shape_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if candidate == "linear_weight":
        return np.asarray([_linear_raw(case) for case in cases], dtype=float), {}
    if candidate == "shape_power_law":
        fit = fit_power_law_shape_exponent(train_cases, passed_only=passed_shape_only)
        exponent = float(fit["exponent"])
        return (
            np.asarray([shape_aware_raw_estimate(case.spectrum, exponent=exponent) for case in cases], dtype=float),
            {"shape_fit": fit, "exponent": exponent},
        )
    raise ValueError(f"unknown candidate {candidate!r}")


def _candidate_holdout_payload(task: dict[str, Any]) -> dict[str, Any]:
    """Score all candidate models for one leave-one-geometry-out holdout."""

    cases = tuple(task["cases"])
    candidates = tuple(task["candidates"])
    observed = np.asarray(task["observed"], dtype=float)
    holdout_idx = int(task["holdout_idx"])
    observed_floor = float(task["observed_floor"])
    passed_shape_only = bool(task["passed_shape_only"])
    interval_z = float(task["interval_z"])
    features_all = task.get("features_all")
    envelope_features_all = task.get("envelope_features_all")
    holdout_case = cases[holdout_idx]
    train_indices = [idx for idx in range(len(cases)) if idx != holdout_idx]
    train_cases = tuple(cases[idx] for idx in train_indices)
    train_observed = observed[train_indices]
    holdout_observed = float(observed[holdout_idx])
    null_predicted = float(np.mean(train_observed))
    null_error = abs(null_predicted - holdout_observed) / max(abs(holdout_observed), observed_floor)
    null_row = {
        "holdout_case": holdout_case.case,
        "predicted_heat_flux": null_predicted,
        "observed_heat_flux": holdout_observed,
        "absolute_relative_error": float(null_error),
    }

    candidate_rows: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate == "linear_state_ridge":
            if features_all is None:
                raise AssertionError("linear_state_ridge requires precomputed features")
            candidate_rows[candidate] = _ridge_loglinear_holdout_row(
                case=holdout_case,
                train_cases=train_cases,
                features_all=np.asarray(features_all, dtype=float),
                feature_names=STATE_FEATURE_NAMES,
                observed_all=observed,
                holdout_idx=holdout_idx,
                train_indices=train_indices,
                observed_floor=observed_floor,
                interval_z=interval_z,
            )
            continue
        if candidate == "spectral_envelope_ridge":
            if envelope_features_all is None:
                raise AssertionError("spectral_envelope_ridge requires precomputed features")
            candidate_rows[candidate] = _ridge_loglinear_holdout_row(
                case=holdout_case,
                train_cases=train_cases,
                features_all=np.asarray(envelope_features_all, dtype=float),
                feature_names=SPECTRAL_ENVELOPE_FEATURE_NAMES,
                observed_all=observed,
                holdout_idx=holdout_idx,
                train_indices=train_indices,
                observed_floor=observed_floor,
                interval_z=interval_z,
                ridge_lambda=0.3,
            )
            continue
        raw_all, metadata = _candidate_raw_values(
            candidate,
            cases,
            train_cases=train_cases,
            passed_shape_only=passed_shape_only,
        )
        train_raw = raw_all[train_indices]
        scale = _fit_scale(train_raw, train_observed, floor=observed_floor)
        train_predicted = scale * train_raw
        holdout_predicted = float(scale * raw_all[holdout_idx])
        train_residual = np.log((train_observed + observed_floor) / np.maximum(train_predicted, observed_floor))
        residual_mean = float(np.mean(train_residual))
        residual_sigma = float(np.std(train_residual, ddof=1)) if train_residual.size > 1 else 0.0
        lo = holdout_predicted * math.exp(residual_mean - interval_z * residual_sigma)
        hi = holdout_predicted * math.exp(residual_mean + interval_z * residual_sigma)
        if lo > hi:
            lo, hi = hi, lo
        rel_error = abs(holdout_predicted - holdout_observed) / max(abs(holdout_observed), observed_floor)
        candidate_rows[candidate] = {
            "holdout_case": holdout_case.case,
            "train_cases": [case.case for case in train_cases],
            "scale": float(scale),
            "raw_estimate": float(raw_all[holdout_idx]),
            "predicted_heat_flux": holdout_predicted,
            "observed_heat_flux": holdout_observed,
            "absolute_relative_error": float(rel_error),
            "prediction_interval_low": float(lo),
            "prediction_interval_high": float(hi),
            "prediction_interval_contains_observed": bool(lo <= holdout_observed <= hi),
            "train_log_residual_mean": residual_mean,
            "train_log_residual_sigma": residual_sigma,
            **metadata,
        }
    return {"holdout_idx": holdout_idx, "null_row": null_row, "candidate_rows": candidate_rows}


def build_candidate_uncertainty_report(
    cases: tuple[SaturationCase, ...] = DEFAULT_CASES,
    *,
    candidates: tuple[str, ...] = (
        "linear_weight",
        "spectral_envelope_ridge",
        "linear_state_ridge",
    ),
    observed_floor: float = 1.0e-12,
    passed_shape_only: bool = True,
    interval_z: float = 1.96,
    transport_gate: float = 0.35,
    interval_coverage_gate: float = 0.75,
    require_validated_inputs: bool = True,
    workers: int = 1,
    parallel_executor: str = "thread",
) -> dict[str, Any]:
    """Build a leave-one-geometry-out uncertainty report for candidate models."""

    input_validation = (
        require_validated_nonlinear_inputs(cases)
        if require_validated_inputs
        else {"kind": "quasilinear_model_input_validation", "passed": None, "required": False}
    )
    observed = np.asarray([_observed_flux(case)[0] for case in cases], dtype=float)
    features_all = _state_feature_matrix(cases, floor=observed_floor) if "linear_state_ridge" in candidates else None
    envelope_features_all = (
        _spectral_envelope_feature_matrix(cases, floor=observed_floor)
        if "spectral_envelope_ridge" in candidates
        else None
    )
    candidate_rows: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in candidates}
    tasks = [
        {
            "holdout_idx": holdout_idx,
            "cases": cases,
            "candidates": candidates,
            "observed": observed,
            "features_all": features_all,
            "envelope_features_all": envelope_features_all,
            "observed_floor": observed_floor,
            "passed_shape_only": passed_shape_only,
            "interval_z": interval_z,
        }
        for holdout_idx in range(len(cases))
    ]
    holdout_payloads = independent_map(
        _candidate_holdout_payload,
        tasks,
        workers=workers,
        executor=parallel_executor,
    )
    holdout_payloads = sorted(holdout_payloads, key=lambda payload: int(payload["holdout_idx"]))
    null_rows = [payload["null_row"] for payload in holdout_payloads]
    for payload in holdout_payloads:
        for candidate in candidates:
            candidate_rows[candidate].append(payload["candidate_rows"][candidate])

    null_errors = np.asarray([row["absolute_relative_error"] for row in null_rows], dtype=float)
    null_mean = float(np.nanmean(null_errors))
    linear_mean = None
    candidates_report = {}
    for candidate, rows in candidate_rows.items():
        errors = np.asarray([row["absolute_relative_error"] for row in rows], dtype=float)
        coverage = float(np.mean([bool(row["prediction_interval_contains_observed"]) for row in rows]))
        mean_error = float(np.nanmean(errors))
        promotion_eligible = all(bool(row.get("promotion_eligible", True)) for row in rows)
        eligibility_failures = sorted(
            {
                str(failure)
                for row in rows
                for failure in row.get("eligibility_failures", [])
            }
        )
        if candidate == "linear_weight":
            linear_mean = mean_error
        candidates_report[candidate] = {
            "label": CANDIDATE_LABELS.get(candidate, candidate),
            "mean_abs_relative_error": mean_error,
            "max_abs_relative_error": float(np.nanmax(errors)),
            "prediction_interval_coverage": coverage,
            "promotion_eligible": promotion_eligible,
            "eligibility_failures": eligibility_failures,
            "rows": rows,
        }

    accepted = []
    for candidate, payload in candidates_report.items():
        mean_error = float(payload["mean_abs_relative_error"])
        beats_linear = linear_mean is None or candidate == "linear_weight" or mean_error < linear_mean
        if (
            bool(payload.get("promotion_eligible", True))
            and mean_error <= transport_gate
            and mean_error < null_mean
            and beats_linear
            and float(payload["prediction_interval_coverage"]) >= interval_coverage_gate
        ):
            accepted.append(candidate)

    return {
        "kind": "quasilinear_candidate_uncertainty_report",
        "claim_level": "candidate_model_development_not_runtime_option",
        "observed_floor": float(observed_floor),
        "passed_shape_only": bool(passed_shape_only),
        "interval_z": float(interval_z),
        "transport_gate": float(transport_gate),
        "interval_coverage_gate": float(interval_coverage_gate),
        "input_validation": input_validation,
        "parallel": {
            "workers": int(workers),
            "executor": str(parallel_executor),
            "identity_contract": "parallel holdout rows preserve serial holdout ordering",
        },
        "null_training_mean_baseline": {
            "mean_abs_relative_error": null_mean,
            "max_abs_relative_error": float(np.nanmax(null_errors)),
            "rows": null_rows,
        },
        "candidates": candidates_report,
        "promotion_gate": {
            "passed": bool(accepted),
            "accepted_candidates": accepted,
            "requires_beating_training_mean_null": True,
            "requires_beating_linear_weight_baseline": True,
            "requires_interval_coverage": True,
            "requires_candidate_eligibility": True,
            "transport_mean_relative_error_gate": float(transport_gate),
            "interval_coverage_gate": float(interval_coverage_gate),
            "null_training_mean_mean_abs_relative_error": null_mean,
            "linear_weight_mean_abs_relative_error": linear_mean,
        },
        "notes": (
            "Candidate predictions are leave-one-geometry-out. Prediction intervals are "
            "estimated from training log residuals only. Candidates remain unavailable as "
            "runtime saturation rules unless promotion_gate.passed is true."
        ),
    }


def write_candidate_uncertainty_figure(
    report: dict[str, Any],
    *,
    out: str | Path,
    title: str,
    dpi: int = 220,
    write_pdf: bool = True,
) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a candidate uncertainty report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    candidates = dict(report["candidates"])
    null_rows = list(report["null_training_mean_baseline"]["rows"])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    ax0, ax1 = axes
    colors = {
        "linear_weight": "#6b7280",
        "shape_power_law": "#0f4c81",
        "spectral_envelope_ridge": "#b45309",
        "linear_state_ridge": "#047857",
    }
    markers = {"linear_weight": "o", "shape_power_law": "s", "spectral_envelope_ridge": "D", "linear_state_ridge": "^"}
    observed_all = np.asarray([row["observed_heat_flux"] for row in null_rows], dtype=float)
    positive = [observed_all[observed_all > 0.0]]
    for payload in candidates.values():
        pred = np.asarray([row["predicted_heat_flux"] for row in payload["rows"]], dtype=float)
        lo = np.asarray([row["prediction_interval_low"] for row in payload["rows"]], dtype=float)
        hi = np.asarray([row["prediction_interval_high"] for row in payload["rows"]], dtype=float)
        positive.extend([pred[pred > 0.0], lo[lo > 0.0], hi[hi > 0.0]])
    flat = np.concatenate([arr for arr in positive if arr.size])
    lim_lo = float(np.min(flat)) * 0.45
    lim_hi = float(np.max(flat)) * 1.8
    ax0.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="0.25", linestyle="--", linewidth=1.3, label="1:1")
    for candidate, payload in candidates.items():
        rows = list(payload["rows"])
        observed = np.asarray([row["observed_heat_flux"] for row in rows], dtype=float)
        predicted = np.asarray([row["predicted_heat_flux"] for row in rows], dtype=float)
        low = np.asarray([row["prediction_interval_low"] for row in rows], dtype=float)
        high = np.asarray([row["prediction_interval_high"] for row in rows], dtype=float)
        yerr = np.vstack([np.maximum(predicted - low, 0.0), np.maximum(high - predicted, 0.0)])
        ax0.errorbar(
            observed,
            predicted,
            yerr=yerr,
            fmt=markers.get(candidate, "o"),
            markersize=6,
            capsize=3,
            color=colors.get(candidate, "0.3"),
            label=payload["label"],
            linewidth=1.2,
        )
    short_labels = {
        "cyclone_long_window": "Cyclone",
        "cyclone_miller_long_window": "Cyclone Miller",
        "hsx_nonlinear_window": "HSX",
        "w7x_nonlinear_window": "W7-X",
        "dshape_external_vmec_t250_window": "D-shaped",
        "itermodel_external_vmec_t350_window": "ITERModel",
        "updown_asym_external_vmec_t450_window": "up-down VMEC",
        "circular_external_vmec_t450_window": "circular",
        "cth_like_external_vmec_t700_high_grid_ensemble": "CTH-like",
        "shaped_tokamak_pressure_external_vmec_t650_high_grid_window": "shaped-pressure",
    }
    if "spectral_envelope_ridge" in candidates:
        spectral_rows = list(candidates["spectral_envelope_ridge"]["rows"])
        label_indices = sorted(
            range(len(spectral_rows)),
            key=lambda idx: float(spectral_rows[idx]["absolute_relative_error"]),
            reverse=True,
        )[:5]
        label_offsets = ((12, 16), (12, -20), (-62, 18), (-58, -18), (12, 28))
        for idx, offset in zip(label_indices, label_offsets, strict=True):
            row = spectral_rows[idx]
            label = short_labels.get(str(row["holdout_case"]), str(row["holdout_case"]))
            ax0.annotate(
                label,
                (float(row["observed_heat_flux"]), float(row["predicted_heat_flux"])),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
                arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.6},
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "0.85",
                    "alpha": 0.78,
                },
            )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(lim_lo, lim_hi)
    ax0.set_ylim(lim_lo, lim_hi)
    ax0.set_xlabel("observed nonlinear heat-flux window")
    ax0.set_ylabel("candidate prediction with 95% interval")
    ax0.set_title("Leave-one-geometry-out predictions")
    ax0.grid(True, which="both", alpha=0.24)
    ax0.legend(loc="best", fontsize=8)

    labels = []
    mean_errors = []
    coverages = []
    promotion_eligible = []
    for candidate, payload in candidates.items():
        labels.append(candidate.replace("_", "\n"))
        mean_errors.append(float(payload["mean_abs_relative_error"]))
        coverages.append(float(payload["prediction_interval_coverage"]))
        promotion_eligible.append(bool(payload.get("promotion_eligible", True)))
    labels.append("train-mean\nnull")
    mean_errors.append(float(report["null_training_mean_baseline"]["mean_abs_relative_error"]))
    coverages.append(float("nan"))
    promotion_eligible.append(True)
    x = np.arange(len(labels))
    bar_colors = [colors.get(candidate, "0.3") for candidate in candidates]
    bar_colors.append("#b45309")
    bars = ax1.bar(x, mean_errors, color=bar_colors)
    for bar, eligible in zip(bars, promotion_eligible, strict=True):
        if not eligible:
            bar.set_hatch("//")
            bar.set_edgecolor("black")
            bar.set_linewidth(0.8)
    ax1.axhline(report["transport_gate"], color="#c2410c", linestyle="--", linewidth=1.4, label="0.35 transport gate")
    ax1.set_yscale("log")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("mean absolute relative error")
    ax1.set_title("Promotion metrics")
    ax1.set_ylim(min(mean_errors) * 0.7, max(mean_errors) * 1.45)
    ax1.grid(True, axis="y", alpha=0.24)
    for xpos, err, cov, eligible in zip(x, mean_errors, coverages, promotion_eligible, strict=True):
        text = f"{err:.2g}" if not math.isfinite(cov) else f"{err:.2g}\ncoverage {cov:.2g}"
        if not eligible:
            text = f"{err:.2g}\nineligible"
        ax1.text(xpos, err / 1.08, text, ha="center", va="top", fontsize=8, color="white", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, y=1.03, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    paths = {"png": str(out_path)}
    if write_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        paths["pdf"] = str(pdf_path)
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["json"] = str(json_path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--out", default=str(root / "docs/_static/quasilinear_candidate_uncertainty.png"))
    parser.add_argument("--title", default="Quasilinear candidate uncertainty gate")
    parser.add_argument("--include-all-shape-gates", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Independent holdout workers.")
    parser.add_argument("--parallel-executor", choices=("thread", "process"), default="thread")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_candidate_uncertainty_report(
        passed_shape_only=not args.include_all_shape_gates,
        workers=args.workers,
        parallel_executor=args.parallel_executor,
    )
    paths = write_candidate_uncertainty_figure(report, out=args.out, title=args.title)
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    print(
        "promotion_gate_passed={passed} accepted_candidates={accepted}".format(
            passed=report["promotion_gate"]["passed"],
            accepted=",".join(report["promotion_gate"]["accepted_candidates"]) or "none",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
