#!/usr/bin/env python3
"""Audit spectral-envelope ridge regularization for QL candidate promotion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402

from plot_quasilinear_candidate_uncertainty import (  # noqa: E402
    SPECTRAL_ENVELOPE_FEATURE_NAMES,
    _observed_flux,
    _ridge_loglinear_holdout_row,
    _spectral_envelope_feature_matrix,
)
from plot_quasilinear_saturation_rule_sweep import DEFAULT_CASES, SaturationCase  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "quasilinear_candidate_regularization_sweep.png"
DEFAULT_LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0)


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


def score_regularization_sweep(
    cases: tuple[SaturationCase, ...] = DEFAULT_CASES,
    *,
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS,
    observed_floor: float = 1.0e-12,
    interval_z: float = 1.96,
    transport_gate: float = 0.35,
    interval_coverage_gate: float = 0.75,
) -> dict[str, Any]:
    """Score spectral-envelope ridge lambda values using LOO predictions."""

    if not lambdas:
        raise ValueError("at least one lambda is required")
    if any(value < 0.0 or not math.isfinite(float(value)) for value in lambdas):
        raise ValueError("lambda values must be finite and non-negative")
    observed = np.asarray([_observed_flux(case)[0] for case in cases], dtype=float)
    features = _spectral_envelope_feature_matrix(cases, floor=observed_floor)
    rows: list[dict[str, Any]] = []
    for lambda_value in lambdas:
        loo_rows = []
        for holdout_idx, case in enumerate(cases):
            train_indices = [idx for idx in range(len(cases)) if idx != holdout_idx]
            loo_rows.append(
                _ridge_loglinear_holdout_row(
                    case=case,
                    train_cases=tuple(cases[idx] for idx in train_indices),
                    features_all=features,
                    feature_names=SPECTRAL_ENVELOPE_FEATURE_NAMES,
                    observed_all=observed,
                    holdout_idx=holdout_idx,
                    train_indices=train_indices,
                    observed_floor=observed_floor,
                    interval_z=interval_z,
                    ridge_lambda=float(lambda_value),
                )
            )
        errors = np.asarray([row["absolute_relative_error"] for row in loo_rows], dtype=float)
        holdout_mask = np.asarray([case.split == "holdout" for case in cases], dtype=bool)
        coverage = float(np.mean([bool(row["prediction_interval_contains_observed"]) for row in loo_rows]))
        mean_error = float(np.mean(errors))
        holdout_mean_error = float(np.mean(errors[holdout_mask])) if np.any(holdout_mask) else mean_error
        row = {
            "lambda": float(lambda_value),
            "mean_abs_relative_error": mean_error,
            "holdout_mean_abs_relative_error": holdout_mean_error,
            "max_abs_relative_error": float(np.max(errors)),
            "prediction_interval_coverage": coverage,
            "promotion_eligible": bool(all(bool(item.get("promotion_eligible", True)) for item in loo_rows)),
            "transport_gate_passed": bool(mean_error <= transport_gate),
            "holdout_transport_gate_passed": bool(holdout_mean_error <= transport_gate),
            "coverage_gate_passed": bool(coverage >= interval_coverage_gate),
            "worst_case": cases[int(np.argmax(errors))].case,
        }
        rows.append(row)
    best = min(rows, key=lambda row: float(row["mean_abs_relative_error"]))
    accepted = [
        row
        for row in rows
        if row["promotion_eligible"]
        and row["transport_gate_passed"]
        and row["coverage_gate_passed"]
    ]
    return {
        "kind": "quasilinear_candidate_regularization_sweep",
        "claim_level": "spectral_envelope_regularization_audit_not_runtime_flux_predictor",
        "candidate": "spectral_envelope_ridge",
        "feature_names": list(SPECTRAL_ENVELOPE_FEATURE_NAMES),
        "observed_floor": float(observed_floor),
        "interval_z": float(interval_z),
        "transport_gate": float(transport_gate),
        "interval_coverage_gate": float(interval_coverage_gate),
        "case_count": len(cases),
        "holdout_count": sum(1 for case in cases if case.split == "holdout"),
        "rows": rows,
        "best_lambda": float(best["lambda"]),
        "best_mean_abs_relative_error": float(best["mean_abs_relative_error"]),
        "best_holdout_mean_abs_relative_error": float(best["holdout_mean_abs_relative_error"]),
        "best_prediction_interval_coverage": float(best["prediction_interval_coverage"]),
        "promotion_gate": {
            "passed": bool(accepted),
            "accepted_lambdas": [float(row["lambda"]) for row in accepted],
            "requires_transport_gate": True,
            "requires_interval_coverage": True,
            "best_lambda": float(best["lambda"]),
            "best_mean_abs_relative_error": float(best["mean_abs_relative_error"]),
            "transport_mean_relative_error_gate": float(transport_gate),
            "blockers": [] if accepted else ["best_regularization_transport_error_above_gate"],
        },
        "notes": (
            "This is a local sensitivity audit for the existing spectral-envelope ridge candidate. "
            "It does not tune lambda on the holdout set for promotion; it verifies whether the near-miss "
            "is robust to plausible regularization changes."
        ),
    }


def write_regularization_sweep_figure(
    report: dict[str, Any],
    *,
    out: str | Path = DEFAULT_OUT,
    title: str = "Quasilinear spectral-envelope regularization audit",
    dpi: int = 180,
    write_pdf: bool = True,
) -> dict[str, str]:
    """Write PNG/PDF/CSV/JSON artifacts for the regularization audit."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(report["rows"])
    lambdas = np.asarray([row["lambda"] for row in rows], dtype=float)
    x = np.arange(len(lambdas))
    full = np.asarray([row["mean_abs_relative_error"] for row in rows], dtype=float)
    holdout = np.asarray([row["holdout_mean_abs_relative_error"] for row in rows], dtype=float)
    max_error = np.asarray([row["max_abs_relative_error"] for row in rows], dtype=float)
    coverage = np.asarray([row["prediction_interval_coverage"] for row in rows], dtype=float)
    gate = float(report["transport_gate"])
    cov_gate = float(report["interval_coverage_gate"])

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    ax0, ax1 = axes
    ax0.plot(x, full, marker="o", linewidth=2.0, label="all cases")
    ax0.plot(x, holdout, marker="s", linewidth=2.0, label="holdout only")
    ax0.plot(x, max_error, marker="^", linewidth=1.5, alpha=0.8, label="max case error")
    ax0.axhline(gate, color="#c2410c", linestyle="--", linewidth=1.4, label=f"{gate:.2f} gate")
    ax0.set_xticks(x, [f"{value:g}" for value in lambdas], rotation=35, ha="right")
    ax0.set_yscale("log")
    ax0.set_xlabel("ridge regularization lambda")
    ax0.set_ylabel("absolute relative error")
    ax0.set_title("Transport-error sensitivity")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(frameon=False, fontsize=8)

    ax1.plot(x, coverage, marker="D", linewidth=2.0, color="#0f766e")
    ax1.axhline(cov_gate, color="#c2410c", linestyle="--", linewidth=1.4, label=f"coverage gate {cov_gate:.2f}")
    ax1.set_xticks(x, [f"{value:g}" for value in lambdas], rotation=35, ha="right")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("ridge regularization lambda")
    ax1.set_ylabel("prediction-interval coverage")
    ax1.set_title("Uncertainty coverage")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)
    best_text = (
        f"best lambda = {report['best_lambda']:g}\n"
        f"best mean error = {report['best_mean_abs_relative_error']:.3f}\n"
        f"promotion = {'PASS' if report['promotion_gate']['passed'] else 'FAIL'}"
    )
    ax1.text(0.03, 0.06, best_text, transform=ax1.transAxes, ha="left", va="bottom", fontsize=9,
             bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9})
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    outputs = {"png": str(out_path)}
    if write_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs["pdf"] = str(pdf_path)
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "lambda",
            "mean_abs_relative_error",
            "holdout_mean_abs_relative_error",
            "max_abs_relative_error",
            "prediction_interval_coverage",
            "promotion_eligible",
            "transport_gate_passed",
            "coverage_gate_passed",
            "worst_case",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    outputs.update({"json": str(json_path), "csv": str(csv_path)})
    return outputs


def _parse_lambdas(raw: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one lambda is required")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("lambda values must be finite and non-negative")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", default="Quasilinear spectral-envelope regularization audit")
    parser.add_argument("--lambdas", type=_parse_lambdas, default=DEFAULT_LAMBDAS)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-pdf", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = score_regularization_sweep(lambdas=tuple(args.lambdas))
    outputs = write_regularization_sweep_figure(
        report,
        out=args.out,
        title=str(args.title),
        dpi=int(args.dpi),
        write_pdf=not bool(args.no_pdf),
    )
    print(f"saved {outputs['png']}")
    print(f"saved {outputs['json']}")
    print(
        "best_lambda={lam:g} best_mean_abs_relative_error={err:.6g} promotion_passed={passed}".format(
            lam=report["best_lambda"],
            err=report["best_mean_abs_relative_error"],
            passed=report["promotion_gate"]["passed"],
        )
    )
    return 0 if report["promotion_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
