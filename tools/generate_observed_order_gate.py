#!/usr/bin/env python3
"""Generate an observed-order gate report from a convergence CSV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from spectraxgk.benchmarking import (
    estimate_observed_order,
    gate_report_to_dict,
    observed_order_gate_report,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "docs" / "_static" / "cyclone_resolution_subset.csv"
DEFAULT_JSON = REPO_ROOT / "docs" / "_static" / "cyclone_resolution_observed_order.json"
DEFAULT_PNG = REPO_ROOT / "docs" / "_static" / "cyclone_resolution_observed_order.png"


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


def load_convergence_series(
    csv_path: Path,
    *,
    step_column: str | None,
    resolution_column: str | None,
    error_column: str,
    absolute_error: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Load ``h`` and error arrays from a convergence table."""

    if (step_column is None) == (resolution_column is None):
        raise ValueError("Specify exactly one of step_column or resolution_column.")
    table = pd.read_csv(csv_path)
    required = {error_column, step_column or resolution_column}
    missing = sorted(col for col in required if col not in table.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")

    if step_column is not None:
        h = np.asarray(table[step_column], dtype=float)
        step_source = step_column
    else:
        resolution = np.asarray(table[resolution_column], dtype=float)
        if np.any(resolution <= 0.0):
            raise ValueError("resolution values must be positive")
        h = 1.0 / resolution
        step_source = f"1/{resolution_column}"

    err = np.asarray(table[error_column], dtype=float)
    if absolute_error:
        err = np.abs(err)

    order = np.argsort(h)[::-1]
    h = h[order]
    err = err[order]
    selected = table.iloc[order].copy()
    selected["effective_step"] = h
    selected["error_for_gate"] = err
    selected["step_source"] = step_source
    rows = _json_clean(selected.to_dict(orient="records"))
    return h, err, rows


def build_summary(
    csv_path: Path,
    *,
    step_column: str | None,
    resolution_column: str | None,
    error_column: str,
    case: str,
    source: str,
    min_order: float,
    min_pairwise_order: float | None,
    max_final_error: float | None,
    absolute_error: bool = True,
) -> dict[str, object]:
    h, err, rows = load_convergence_series(
        csv_path,
        step_column=step_column,
        resolution_column=resolution_column,
        error_column=error_column,
        absolute_error=absolute_error,
    )
    metrics = estimate_observed_order(h, err)
    report = observed_order_gate_report(
        metrics,
        case=case,
        source=source,
        min_asymptotic_order=min_order,
        min_pairwise_order=min_pairwise_order,
        max_final_error=max_final_error,
    )
    payload = {
        "case": case,
        "source": source,
        "csv": str(csv_path),
        "error_column": error_column,
        "absolute_error": bool(absolute_error),
        "step_sizes": metrics.step_sizes.tolist(),
        "errors": metrics.errors.tolist(),
        "pairwise_orders": metrics.orders.tolist(),
        "asymptotic_order": metrics.asymptotic_order,
        "min_pairwise_order": float(np.min(metrics.orders)),
        "final_error": float(metrics.errors[-1]),
        "gate_report": gate_report_to_dict(report),
        "gate_passed": bool(report.passed),
        "rows": rows,
    }
    return _json_clean(payload)


def write_observed_order_plot(
    summary: dict[str, object],
    out_png: Path,
    *,
    title: str,
    min_order: float,
) -> None:
    h = np.asarray(summary["step_sizes"], dtype=float)
    err = np.asarray(summary["errors"], dtype=float)
    asymptotic_order = float(summary["asymptotic_order"])
    min_pairwise_order = float(summary["min_pairwise_order"])
    final_error = float(summary["final_error"])
    gate_status = "passed" if bool(summary["gate_passed"]) else "open"

    fig, ax = plt.subplots(figsize=(5.2, 3.7), constrained_layout=True)
    ax.loglog(h, err, "o-", color="#1b6ca8", lw=2.0, ms=6, label="measured error")
    ref = err[-1] * (h / h[-1]) ** float(min_order)
    ax.loglog(h, ref, "--", color="#8f2d2d", lw=1.8, label=f"order {min_order:g} guide")
    ax.invert_xaxis()
    ax.set_xlabel("effective step size h (coarse to fine)")
    ax.set_ylabel("absolute error")
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.03,
        0.04,
        (
            f"final-pair order = {asymptotic_order:.2f}\n"
            f"min pairwise order = {min_pairwise_order:.2f}\n"
            f"final error = {final_error:.3g}\n"
            f"gate: {gate_status}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.85", "alpha": 0.92},
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write an observed-order gate report from a convergence CSV."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--step-column", default=None)
    group.add_argument("--resolution-column", default=None)
    parser.add_argument("--error-column", default="rel_gamma")
    parser.add_argument("--case", default="cyclone_resolution_observed_order")
    parser.add_argument("--source", default="tracked Cyclone resolution subset")
    parser.add_argument("--min-order", type=float, default=1.0)
    parser.add_argument(
        "--min-pairwise-order",
        type=float,
        default=0.0,
        help="Optional floor for every pairwise observed order; set negative to disable.",
    )
    parser.add_argument("--max-final-error", type=float, default=0.05)
    parser.add_argument(
        "--signed-error",
        action="store_true",
        help="Use signed errors instead of absolute values.",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--title", default="Cyclone Resolution Convergence")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolution_column = args.resolution_column if args.resolution_column is not None else "Nm"
    min_pairwise_order = (
        None
        if args.min_pairwise_order is None or float(args.min_pairwise_order) < 0.0
        else float(args.min_pairwise_order)
    )
    summary = build_summary(
        args.csv,
        step_column=args.step_column,
        resolution_column=resolution_column if args.step_column is None else None,
        error_column=args.error_column,
        case=args.case,
        source=args.source,
        min_order=float(args.min_order),
        min_pairwise_order=min_pairwise_order,
        max_final_error=args.max_final_error,
        absolute_error=not bool(args.signed_error),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if not args.no_plot:
        write_observed_order_plot(summary, args.out_png, title=args.title, min_order=float(args.min_order))
    print(f"Wrote {args.out_json}")
    if not args.no_plot:
        print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
