#!/usr/bin/env python3
"""Plot quasilinear-vs-nonlinear calibration report points."""

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


SPLIT_COLORS = {
    "train": "#0f4c81",
    "holdout": "#2a9d8f",
    "audit": "#c44e52",
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


def load_calibration_report(path: str | Path) -> dict[str, Any]:
    """Load and validate a quasilinear calibration report."""

    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("kind") != "quasilinear_calibration_report":
        raise ValueError(f"{path} is not a quasilinear calibration report")
    points = report.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("calibration report must contain a non-empty points list")
    required = {"case", "split", "predicted_heat_flux", "observed_heat_flux"}
    for point in points:
        missing = required - set(point)
        if missing:
            raise ValueError(f"calibration point is missing keys: {sorted(missing)}")
    return report


def _point_arrays(points: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray([float(p["predicted_heat_flux"]) for p in points], dtype=float)
    observed = np.asarray([float(p["observed_heat_flux"]) for p in points], dtype=float)
    std = np.asarray(
        [
            np.nan if p.get("observed_heat_flux_std") is None else float(p["observed_heat_flux_std"])
            for p in points
        ],
        dtype=float,
    )
    return predicted, observed, std


def calibration_figure(
    report: dict[str, Any],
    *,
    title: str = "Quasilinear calibration audit",
) -> plt.Figure:
    """Create a publication-facing calibration/audit figure."""

    points = list(report["points"])
    predicted, observed, std = _point_arrays(points)
    finite = np.isfinite(predicted) & np.isfinite(observed)
    if not np.any(finite):
        raise ValueError("no finite calibration points to plot")

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    ax = axes[0]
    residual_ax = axes[1]

    pos = finite & (predicted > 0.0) & (observed > 0.0)
    if np.any(pos):
        lo = float(min(np.min(predicted[pos]), np.min(observed[pos]))) * 0.7
        hi = float(max(np.max(predicted[pos]), np.max(observed[pos]))) * 1.4
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1.6, label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    else:
        lo = float(min(np.min(predicted[finite]), np.min(observed[finite]), 0.0))
        hi = float(max(np.max(predicted[finite]), np.max(observed[finite]), 1.0))
        ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1.6, label="1:1")

    labels = []
    rel_errors = []
    colors = []
    for point, pred, obs, err in zip(points, predicted, observed, std, strict=True):
        if not np.isfinite(pred) or not np.isfinite(obs):
            continue
        split = str(point["split"])
        color = SPLIT_COLORS.get(split, "#6c757d")
        ax.errorbar(
            pred,
            obs,
            yerr=None if not np.isfinite(err) else err,
            marker="o",
            markersize=8,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=3.0,
            linestyle="None",
            label=split,
        )
        labels.append(str(point["case"]))
        denom = max(abs(obs), float(report.get("observed_floor", 1.0e-12)))
        rel_errors.append(abs(pred - obs) / denom)
        colors.append(color)

    ax.set_title("Absolute flux comparison")
    ax.set_xlabel("quasilinear estimate")
    ax.set_ylabel("nonlinear window mean")
    handles, legend_labels = ax.get_legend_handles_labels()
    legend = dict(zip(legend_labels, handles, strict=False))
    ax.legend(legend.values(), legend.keys())
    ax.grid(True, which="both", alpha=0.22)

    y = np.arange(len(labels))
    residual_ax.barh(y, rel_errors, color=colors, edgecolor="white", linewidth=0.8)
    gate = float(report.get("holdout_mean_rel_gate", 0.35))
    residual_ax.axvline(gate, color="#c2410c", linestyle="--", linewidth=1.6, label="holdout gate")
    residual_ax.set_yticks(y, labels)
    residual_ax.invert_yaxis()
    residual_ax.set_xlabel("absolute relative error")
    residual_ax.set_title(f"Claim level: {report.get('claim_level', 'unknown')}")
    residual_ax.legend(loc="lower right", frameon=True, framealpha=0.92)
    residual_ax.grid(True, axis="x", alpha=0.22)

    fig.suptitle(title, y=1.03, fontsize=14, fontweight="bold")
    return fig


def write_calibration_figure(report_path: str | Path, *, out: str | Path, title: str) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a calibration report."""

    report = load_calibration_report(report_path)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = calibration_figure(report, title=title)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    points = report["points"]
    predicted, observed, _std = _point_arrays(points)
    denom = np.maximum(np.abs(observed), float(report.get("observed_floor", 1.0e-12)))
    rel = np.abs(predicted - observed) / denom
    meta_path = out_path.with_suffix(".json")
    meta = {
        "kind": "quasilinear_calibration_figure",
        "source": str(report_path),
        "png": str(out_path),
        "pdf": str(pdf_path),
        "n_points": int(len(points)),
        "claim_level": report.get("claim_level"),
        "passed": bool(report.get("passed", False)),
        "max_abs_relative_error": float(np.nanmax(rel)),
        "mean_abs_relative_error": float(np.nanmean(rel)),
    }
    meta_path.write_text(json.dumps(_json_clean(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(meta_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Quasilinear calibration report JSON")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Quasilinear calibration audit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_calibration_figure(args.report, out=args.out, title=args.title)
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
