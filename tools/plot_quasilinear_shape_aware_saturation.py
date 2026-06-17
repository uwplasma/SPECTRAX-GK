#!/usr/bin/env python3
"""Fit and score a low-dimensional shape-aware quasilinear saturation model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import sys

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.quasilinear import shape_aware_power_law_objective  # noqa: E402
from spectraxgk.validation.quasilinear.calibration import calibration_point_from_nonlinear_window_summary  # noqa: E402

from plot_quasilinear_saturation_rule_sweep import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_HOLDOUT_RELATIVE_ERROR_GATE,
    SaturationCase,
    _artifact_path,
    require_validated_nonlinear_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPE_CASES = tuple(case for case in DEFAULT_CASES if case.shape_gate is not None)


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


def _load_table(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    return data


def _required_column(data: np.ndarray, path: Path, column: str) -> np.ndarray:
    names = set(data.dtype.names or ())
    if column not in names:
        raise ValueError(f"{path} is missing required column '{column}'")
    return np.asarray(data[column], dtype=float)


def _shape_gate_payload(case: SaturationCase) -> dict[str, Any]:
    if case.shape_gate is None or not Path(case.shape_gate).exists():
        raise ValueError(f"{case.case} is missing a tracked shape-gate JSON")
    data = json.loads(Path(case.shape_gate).read_text(encoding="utf-8"))
    required = (
        "kind",
        "passed",
        "ky",
        "quasilinear_distribution",
        "nonlinear_distribution",
        "total_variation_distance",
        "cosine_similarity",
        "tv_gate",
        "cosine_gate",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"{case.shape_gate} is missing '{key}'")
    if data["kind"] != "quasilinear_spectrum_shape_gate":
        raise ValueError(f"{case.shape_gate} is not a quasilinear spectrum-shape gate")
    return data


def fit_power_law_shape_exponent(
    cases: tuple[SaturationCase, ...],
    *,
    passed_only: bool = False,
    floor: float = 1.0e-300,
) -> dict[str, Any]:
    """Fit ``nonlinear_shape / quasilinear_shape ~ C_case * ky**exponent``.

    Each training case receives its own intercept, while the exponent is shared.
    The fitted exponent is therefore a shape-transfer parameter, not an
    absolute-flux scale.
    """

    xs: list[float] = []
    ys: list[float] = []
    groups: list[int] = []
    used_cases: list[str] = []
    for group, case in enumerate(cases):
        payload = _shape_gate_payload(case)
        if passed_only and not bool(payload.get("passed", False)):
            continue
        ky = np.asarray(payload["ky"], dtype=float)
        ql = np.asarray(payload["quasilinear_distribution"], dtype=float)
        nl = np.asarray(payload["nonlinear_distribution"], dtype=float)
        mask = (ky > 0.0) & (ql > floor) & (nl > floor) & np.isfinite(ky) & np.isfinite(ql) & np.isfinite(nl)
        if not np.any(mask):
            continue
        ky_use = ky[mask]
        x = np.log(ky_use / np.exp(np.mean(np.log(ky_use))))
        y = np.log(nl[mask] / ql[mask])
        xs.extend(float(v) for v in x)
        ys.extend(float(v) for v in y)
        groups.extend([group] * len(x))
        used_cases.append(case.case)

    if len(used_cases) < 1 or len(xs) < 2:
        raise ValueError("not enough shape samples to fit a power-law envelope")
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    group_arr = np.asarray(groups, dtype=int)
    n_groups = max(groups) + 1
    design = np.zeros((x_arr.size, n_groups + 1), dtype=float)
    design[np.arange(x_arr.size), group_arr] = 1.0
    design[:, -1] = x_arr
    coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    residual = y_arr - design @ coef
    return {
        "exponent": float(coef[-1]),
        "used_cases": used_cases,
        "n_samples": int(x_arr.size),
        "rms_log_shape_residual": float(np.sqrt(np.mean(residual**2))),
        "passed_only": bool(passed_only),
    }


def shape_aware_raw_estimate(spectrum_csv: str | Path, *, exponent: float) -> float:
    """Return ``sum Qhat(ky) * (ky/geomean(ky))**exponent`` from a spectrum CSV."""

    path = Path(spectrum_csv)
    data = _load_table(path)
    ky = _required_column(data, path, "ky")
    weight = np.maximum(_required_column(data, path, "heat_flux_weight_total"), 0.0)
    finite = (ky > 0.0) & np.isfinite(ky) & np.isfinite(weight)
    if not np.any(finite):
        raise ValueError(f"{path} contains no finite positive ky/weight samples")
    ky = ky[finite]
    weight = weight[finite]
    features = np.stack([np.zeros_like(weight), np.ones_like(weight), weight], axis=-1)
    return float(np.sum(np.asarray(shape_aware_power_law_objective(features, ky, exponent=exponent))))


def _tracked_observed_flux(case: SaturationCase) -> tuple[float, float | None] | None:
    """Return observed flux from tracked calibration sidecars when raw traces are absent.

    Several long nonlinear diagnostics live under ignored ``tools_out`` paths to
    keep the repository small.  CI and source distributions still need to
    replay candidate-model audits from tracked evidence, so the plotting tools
    fall back to the compact calibration-point sidecar instead of requiring the
    raw trace to be present.
    """

    candidates = (
        ROOT / "docs/_static/quasilinear_stellarator_train_holdout_points.json",
        ROOT / "docs/_static/quasilinear_stellarator_train_holdout_report.json",
    )
    aliases = {
        "cth_like_external_vmec_t700_high_grid_ensemble": "cth_like_external_vmec_t700_high_grid_window",
    }
    wanted = {case.case, aliases.get(case.case, case.case)}
    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        points = payload if isinstance(payload, list) else payload.get("points", [])
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, dict) or point.get("case") not in wanted:
                continue
            observed = point.get("observed_heat_flux")
            if observed is None:
                continue
            observed_std = point.get("observed_heat_flux_std")
            return (
                float(observed),
                None if observed_std is None else float(observed_std),
            )
    return None


def _observed_flux(case: SaturationCase) -> tuple[float, float | None]:
    try:
        point = calibration_point_from_nonlinear_window_summary(
            case.nonlinear_summary,
            predicted_heat_flux=1.0,
            split=case.split,
            saturation_rule="shape_aware_power_law",
            geometry=case.geometry,
            electron_model="adiabatic",
            quasilinear_artifact=str(case.spectrum),
        )
    except FileNotFoundError:
        tracked = _tracked_observed_flux(case)
        if tracked is None:
            raise
        return tracked
    return point.observed_heat_flux, point.observed_heat_flux_std


def _fit_scale(raw: np.ndarray, observed: np.ndarray, *, floor: float) -> float:
    finite = np.isfinite(raw) & np.isfinite(observed) & (np.abs(raw) > floor)
    if not np.any(finite):
        return float("nan")
    denom = float(np.dot(raw[finite], raw[finite]))
    if denom <= floor:
        return float("nan")
    return float(np.dot(raw[finite], observed[finite]) / denom)


def build_shape_aware_saturation_report(
    cases: tuple[SaturationCase, ...] = DEFAULT_SHAPE_CASES,
    *,
    observed_floor: float = 1.0e-12,
    passed_shape_only: bool = False,
    require_validated_inputs: bool = True,
    holdout_relative_error_gate: float = DEFAULT_HOLDOUT_RELATIVE_ERROR_GATE,
) -> dict[str, Any]:
    """Run leave-one-case-out validation for the power-law shape-aware model."""

    if holdout_relative_error_gate <= 0.0:
        raise ValueError("holdout_relative_error_gate must be positive")
    input_validation = (
        require_validated_nonlinear_inputs(cases)
        if require_validated_inputs
        else {"kind": "quasilinear_model_input_validation", "passed": None, "required": False}
    )
    case_rows = []
    for case in cases:
        observed, observed_std = _observed_flux(case)
        shape_payload = _shape_gate_payload(case)
        case_rows.append(
            {
                "case": case.case,
                "geometry": case.geometry,
                "spectrum": _artifact_path(case.spectrum),
                "nonlinear_summary": _artifact_path(case.nonlinear_summary),
                "shape_gate": _artifact_path(case.shape_gate),
                "shape_gate_kind": shape_payload.get("kind"),
                "shape_gate_status": "passed" if bool(shape_payload.get("passed", False)) else "failed",
                "shape_passed": bool(shape_payload.get("passed", False)),
                "shape_tv": shape_payload.get("total_variation_distance"),
                "shape_cosine": shape_payload.get("cosine_similarity"),
                "shape_tv_gate": shape_payload.get("tv_gate"),
                "shape_cosine_gate": shape_payload.get("cosine_gate"),
                "observed_heat_flux": float(observed),
                "observed_heat_flux_std": None if observed_std is None else float(observed_std),
            }
        )

    observed_arr = np.asarray([row["observed_heat_flux"] for row in case_rows], dtype=float)
    loo_rows = []
    for holdout_idx, holdout_case in enumerate(cases):
        train_cases = tuple(case for i, case in enumerate(cases) if i != holdout_idx)
        fit = fit_power_law_shape_exponent(train_cases, passed_only=passed_shape_only)
        exponent = float(fit["exponent"])
        train_raw = np.asarray([shape_aware_raw_estimate(case.spectrum, exponent=exponent) for case in train_cases])
        train_observed = np.asarray([case_rows[i]["observed_heat_flux"] for i in range(len(cases)) if i != holdout_idx])
        scale = _fit_scale(train_raw, train_observed, floor=observed_floor)
        holdout_raw = shape_aware_raw_estimate(holdout_case.spectrum, exponent=exponent)
        predicted = float(scale * holdout_raw)
        observed = float(observed_arr[holdout_idx])
        rel_error = abs(predicted - observed) / max(abs(observed), observed_floor)
        baseline_raw_train = np.asarray([shape_aware_raw_estimate(case.spectrum, exponent=0.0) for case in train_cases])
        baseline_scale = _fit_scale(baseline_raw_train, train_observed, floor=observed_floor)
        baseline_predicted = float(baseline_scale * shape_aware_raw_estimate(holdout_case.spectrum, exponent=0.0))
        baseline_rel_error = abs(baseline_predicted - observed) / max(abs(observed), observed_floor)
        null_predicted = float(np.mean(train_observed))
        null_rel_error = abs(null_predicted - observed) / max(abs(observed), observed_floor)
        loo_rows.append(
            {
                "holdout_case": holdout_case.case,
                "train_cases": [case.case for case in train_cases],
                "exponent": exponent,
                "scale": scale,
                "predicted_heat_flux": predicted,
                "observed_heat_flux": observed,
                "absolute_relative_error": float(rel_error),
                "baseline_linear_weight_predicted_heat_flux": baseline_predicted,
                "baseline_linear_weight_absolute_relative_error": float(baseline_rel_error),
                "null_training_mean_predicted_heat_flux": null_predicted,
                "null_training_mean_absolute_relative_error": float(null_rel_error),
                "shape_fit": fit,
            }
        )

    shape_errors = np.asarray([row["absolute_relative_error"] for row in loo_rows], dtype=float)
    baseline_errors = np.asarray([row["baseline_linear_weight_absolute_relative_error"] for row in loo_rows], dtype=float)
    null_errors = np.asarray([row["null_training_mean_absolute_relative_error"] for row in loo_rows], dtype=float)
    all_fit = fit_power_law_shape_exponent(cases, passed_only=passed_shape_only)
    shape_mean = float(np.nanmean(shape_errors))
    baseline_mean = float(np.nanmean(baseline_errors))
    null_mean = float(np.nanmean(null_errors))
    transport_gate = float(holdout_relative_error_gate)
    return {
        "kind": "quasilinear_shape_aware_saturation_report",
        "claim_level": "leave_one_geometry_out_model_development",
        "observed_floor": float(observed_floor),
        "holdout_relative_error_gate": float(holdout_relative_error_gate),
        "passed_shape_only": bool(passed_shape_only),
        "input_validation": input_validation,
        "all_case_shape_fit": all_fit,
        "metrics": {
            "shape_aware_mean_abs_relative_error": shape_mean,
            "shape_aware_max_abs_relative_error": float(np.nanmax(shape_errors)),
            "shape_aware_all_case_gate_passed": bool(np.all(shape_errors <= transport_gate)),
            "baseline_linear_weight_mean_abs_relative_error": baseline_mean,
            "baseline_linear_weight_max_abs_relative_error": float(np.nanmax(baseline_errors)),
            "baseline_linear_weight_all_case_gate_passed": bool(np.all(baseline_errors <= transport_gate)),
            "null_training_mean_mean_abs_relative_error": null_mean,
            "null_training_mean_max_abs_relative_error": float(np.nanmax(null_errors)),
            "null_training_mean_all_case_gate_passed": bool(np.all(null_errors <= transport_gate)),
        },
        "promotion_gate": {
            "passed": bool(shape_mean <= transport_gate and shape_mean < baseline_mean and shape_mean < null_mean),
            "transport_mean_relative_error_gate": transport_gate,
            "requires_beating_linear_weight_baseline": True,
            "requires_beating_training_mean_null": True,
            "shape_aware_mean_abs_relative_error": shape_mean,
            "baseline_linear_weight_mean_abs_relative_error": baseline_mean,
            "null_training_mean_mean_abs_relative_error": null_mean,
        },
        "cases": case_rows,
        "leave_one_out": loo_rows,
        "notes": (
            "The power-law exponent is fitted from training nonlinear spectrum-shape gates only; "
            "the held-out nonlinear shape is not used for that held-out prediction. This is a "
            "model-development diagnostic, not a validated transport claim."
        ),
    }


def write_shape_aware_saturation_figure(report: dict[str, Any], *, out: str | Path, title: str) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a shape-aware saturation report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(report["leave_one_out"])
    labels = [str(row["holdout_case"]) for row in rows]
    short_labels = [label.replace("_long_window", "").replace("_nonlinear_window", "") for label in labels]
    observed = np.asarray([row["observed_heat_flux"] for row in rows], dtype=float)
    predicted = np.asarray([row["predicted_heat_flux"] for row in rows], dtype=float)
    baseline = np.asarray([row["baseline_linear_weight_predicted_heat_flux"] for row in rows], dtype=float)
    null = np.asarray([row["null_training_mean_predicted_heat_flux"] for row in rows], dtype=float)
    shape_err = np.asarray([row["absolute_relative_error"] for row in rows], dtype=float)
    baseline_err = np.asarray([row["baseline_linear_weight_absolute_relative_error"] for row in rows], dtype=float)
    null_err = np.asarray([row["null_training_mean_absolute_relative_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    ax0, ax1 = axes
    positive = np.concatenate(
        [observed[observed > 0.0], predicted[predicted > 0.0], baseline[baseline > 0.0], null[null > 0.0]]
    )
    lo = float(np.min(positive)) * 0.6
    hi = float(np.max(positive)) * 1.7
    ax0.plot([lo, hi], [lo, hi], color="0.25", linestyle="--", linewidth=1.5, label="1:1")
    ax0.scatter(observed, baseline, s=70, facecolors="none", edgecolors="#6b7280", linewidth=1.5, label="linear-weight LOO")
    ax0.scatter(observed, null, s=65, marker="^", color="#b45309", edgecolor="white", linewidth=0.8, label="train-mean null")
    ax0.scatter(observed, predicted, s=75, color="#0f4c81", edgecolor="white", linewidth=0.8, label="shape-aware LOO")
    for label, xval, yval in zip(short_labels, observed, predicted, strict=True):
        ax0.annotate(label, (xval, yval), xytext=(5, 4), textcoords="offset points", fontsize=7)
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_xlabel("observed nonlinear heat-flux window")
    ax0.set_ylabel("leave-one-out prediction")
    ax0.set_title("Absolute flux")
    ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, which="both", alpha=0.24)

    y = np.arange(len(labels))
    height = 0.25
    ax1.barh(y - height, baseline_err, height=height, color="#9ca3af", label="linear-weight baseline")
    ax1.barh(y, null_err, height=height, color="#b45309", label="train-mean null")
    ax1.barh(y + height, shape_err, height=height, color="#0f4c81", label="shape-aware")
    gate = float(report.get("holdout_relative_error_gate", DEFAULT_HOLDOUT_RELATIVE_ERROR_GATE))
    ax1.axvline(gate, color="#c2410c", linestyle="--", linewidth=1.5, label=f"{gate:.2g} gate")
    ax1.set_xscale("log")
    ax1.set_yticks(y, short_labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("absolute relative error")
    ax1.set_title("Leave-one-geometry-out errors")
    ax1.grid(True, axis="x", alpha=0.24)
    ax1.legend(loc="upper right", fontsize=8)

    fig.suptitle(title, y=1.03, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--out", default=str(root / "docs/_static/quasilinear_shape_aware_saturation.png"))
    parser.add_argument("--title", default="Shape-aware quasilinear saturation diagnostic")
    parser.add_argument(
        "--passed-shape-only",
        action="store_true",
        help="Fit shape exponent using only training cases whose shape gate passed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_shape_aware_saturation_report(passed_shape_only=args.passed_shape_only)
    paths = write_shape_aware_saturation_figure(report, out=args.out, title=args.title)
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    print(
        "shape_aware_mean_abs_relative_error={shape:.6g} "
        "baseline_mean_abs_relative_error={base:.6g} "
        "null_mean_abs_relative_error={null:.6g}".format(
            shape=report["metrics"]["shape_aware_mean_abs_relative_error"],
            base=report["metrics"]["baseline_linear_weight_mean_abs_relative_error"],
            null=report["metrics"]["null_training_mean_mean_abs_relative_error"],
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
