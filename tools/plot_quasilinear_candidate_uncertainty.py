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

from plot_quasilinear_saturation_rule_sweep import DEFAULT_CASES, SaturationCase, raw_rule_estimates  # noqa: E402
from plot_quasilinear_shape_aware_saturation import (  # noqa: E402
    _observed_flux,
    fit_power_law_shape_exponent,
    shape_aware_raw_estimate,
)


CANDIDATE_LABELS = {
    "linear_weight": r"$\hat Q$ calibrated",
    "shape_power_law": r"shape power law",
}


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


def build_candidate_uncertainty_report(
    cases: tuple[SaturationCase, ...] = DEFAULT_CASES,
    *,
    candidates: tuple[str, ...] = ("linear_weight", "shape_power_law"),
    observed_floor: float = 1.0e-12,
    passed_shape_only: bool = True,
    interval_z: float = 1.96,
    transport_gate: float = 0.35,
    interval_coverage_gate: float = 0.75,
) -> dict[str, Any]:
    """Build a leave-one-geometry-out uncertainty report for candidate models."""

    observed = np.asarray([_observed_flux(case)[0] for case in cases], dtype=float)
    null_rows = []
    candidate_rows: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in candidates}
    for holdout_idx, holdout_case in enumerate(cases):
        train_indices = [idx for idx in range(len(cases)) if idx != holdout_idx]
        train_cases = tuple(cases[idx] for idx in train_indices)
        train_observed = observed[train_indices]
        holdout_observed = float(observed[holdout_idx])
        null_predicted = float(np.mean(train_observed))
        null_error = abs(null_predicted - holdout_observed) / max(abs(holdout_observed), observed_floor)
        null_rows.append(
            {
                "holdout_case": holdout_case.case,
                "predicted_heat_flux": null_predicted,
                "observed_heat_flux": holdout_observed,
                "absolute_relative_error": float(null_error),
            }
        )

        for candidate in candidates:
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
            candidate_rows[candidate].append(
                {
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
            )

    null_errors = np.asarray([row["absolute_relative_error"] for row in null_rows], dtype=float)
    null_mean = float(np.nanmean(null_errors))
    linear_mean = None
    candidates_report = {}
    for candidate, rows in candidate_rows.items():
        errors = np.asarray([row["absolute_relative_error"] for row in rows], dtype=float)
        coverage = float(np.mean([bool(row["prediction_interval_contains_observed"]) for row in rows]))
        mean_error = float(np.nanmean(errors))
        if candidate == "linear_weight":
            linear_mean = mean_error
        candidates_report[candidate] = {
            "label": CANDIDATE_LABELS.get(candidate, candidate),
            "mean_abs_relative_error": mean_error,
            "max_abs_relative_error": float(np.nanmax(errors)),
            "prediction_interval_coverage": coverage,
            "rows": rows,
        }

    accepted = []
    for candidate, payload in candidates_report.items():
        mean_error = float(payload["mean_abs_relative_error"])
        beats_linear = linear_mean is None or candidate == "linear_weight" or mean_error < linear_mean
        if (
            mean_error <= transport_gate
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


def write_candidate_uncertainty_figure(report: dict[str, Any], *, out: str | Path, title: str) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a candidate uncertainty report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    candidates = dict(report["candidates"])
    null_rows = list(report["null_training_mean_baseline"]["rows"])
    case_labels = [row["holdout_case"].replace("_long_window", "").replace("_nonlinear_window", "") for row in null_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    ax0, ax1 = axes
    colors = {"linear_weight": "#6b7280", "shape_power_law": "#0f4c81"}
    markers = {"linear_weight": "o", "shape_power_law": "s"}
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
    for label, xval, yval in zip(
        case_labels,
        observed_all,
        [row["predicted_heat_flux"] for row in candidates[next(iter(candidates))]["rows"]],
        strict=True,
    ):
        ax0.annotate(label, (xval, yval), xytext=(4, 4), textcoords="offset points", fontsize=7)
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
    for candidate, payload in candidates.items():
        labels.append(candidate.replace("_", "\n"))
        mean_errors.append(float(payload["mean_abs_relative_error"]))
        coverages.append(float(payload["prediction_interval_coverage"]))
    labels.append("train-mean\nnull")
    mean_errors.append(float(report["null_training_mean_baseline"]["mean_abs_relative_error"]))
    coverages.append(float("nan"))
    x = np.arange(len(labels))
    ax1.bar(x, mean_errors, color=["#6b7280", "#0f4c81", "#b45309"][: len(labels)])
    ax1.axhline(report["transport_gate"], color="#c2410c", linestyle="--", linewidth=1.4, label="0.35 transport gate")
    ax1.set_yscale("log")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("mean absolute relative error")
    ax1.set_title("Promotion metrics")
    ax1.set_ylim(min(mean_errors) * 0.7, max(mean_errors) * 1.45)
    ax1.grid(True, axis="y", alpha=0.24)
    for xpos, err, cov in zip(x, mean_errors, coverages, strict=True):
        text = f"{err:.2g}" if not math.isfinite(cov) else f"{err:.2g}\ncoverage {cov:.2g}"
        ax1.text(xpos, err / 1.08, text, ha="center", va="top", fontsize=8, color="white", fontweight="bold")
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
    parser.add_argument("--out", default=str(root / "docs/_static/quasilinear_candidate_uncertainty.png"))
    parser.add_argument("--title", default="Quasilinear candidate uncertainty gate")
    parser.add_argument("--include-all-shape-gates", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_candidate_uncertainty_report(passed_shape_only=not args.include_all_shape_gates)
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
