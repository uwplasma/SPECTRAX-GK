#!/usr/bin/env python3
"""Audit the reduced QA low-turbulence nonlinear time horizon.

The main comparison panel uses a reduced differentiable nonlinear envelope at a
fixed gradient. This helper keeps that claim honest by re-evaluating the tracked
optimized designs at progressively longer horizons and comparing the late-window
means to the longest horizon in the sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.objectives.qa_low_turbulence import (  # noqa: E402
    QALowTurbulenceConfig,
    qa_low_turbulence_heat_flux_trace,
    qa_low_turbulence_window_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON = ROOT / "docs" / "_static" / "qa_low_turbulence_comparison.json"
DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "qa_low_turbulence_time_horizon_audit"
COLORS = {
    "qa_constraints": "#244c66",
    "qa_plus_nonlinear_heat_flux": "#b45f2a",
}
LABELS = {
    "qa_constraints": "QA constraints",
    "qa_plus_nonlinear_heat_flux": "QA + reduced NL Q",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=[150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0],
        help="time horizons t v_ti/a to audit; longest is the reference",
    )
    parser.add_argument("--nonlinear-dt", type=float, default=0.20)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--cv-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--trend-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--half-window-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--pdf", action="store_true", help="write a PDF companion")
    return parser.parse_args()


def _load_designs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    designs: list[dict[str, Any]] = []
    for result in payload["results"]:
        designs.append(
            {
                "design_name": str(result["design_name"]),
                "final_params": [float(x) for x in result["final_params"]],
            }
        )
    return designs


def _window_row(
    *,
    design_name: str,
    params: Sequence[float],
    horizon: float,
    nonlinear_dt: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray | int]]:
    steps = int(round(float(horizon) / float(nonlinear_dt)))
    cfg = QALowTurbulenceConfig(nonlinear_dt=float(nonlinear_dt), nonlinear_steps=steps)
    times, heat_flux = qa_low_turbulence_heat_flux_trace(
        params,
        cfg,
        density_gradient=cfg.fixed_density_gradient,
        temperature_gradient=cfg.fixed_temperature_gradient,
    )
    window = qa_low_turbulence_window_metrics(
        times,
        heat_flux,
        tail_fraction=cfg.nonlinear_tail_fraction,
    )
    times_np = np.asarray(times, dtype=float)
    heat_flux_np = np.asarray(heat_flux, dtype=float)
    start = int(window["start_index"])
    tail = heat_flux_np[start:]
    split = max(1, tail.size // 2)
    mean = float(window["mean"])
    half_change = abs(float(np.mean(tail[split:])) - float(np.mean(tail[:split]))) / max(
        abs(mean),
        1.0e-14,
    )
    return (
        {
            "design_name": design_name,
            "t_end": float(times_np[-1]),
            "window_start": float(times_np[start]),
            "mean": mean,
            "cv": float(window["cv"]),
            "trend": float(window["trend"]),
            "half_window_relative_mean_change": float(half_change),
        },
        {"times": times_np, "heat_flux": heat_flux_np, "start_index": start},
    )


def build_time_horizon_payload(
    comparison_json: Path = DEFAULT_COMPARISON,
    *,
    horizons: Sequence[float] = (150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0),
    nonlinear_dt: float = 0.20,
    relative_tolerance: float = 1.0e-3,
    cv_tolerance: float = 1.0e-3,
    trend_tolerance: float = 1.0e-3,
    half_window_tolerance: float = 1.0e-3,
) -> dict[str, Any]:
    """Return a JSON-ready time-horizon audit payload."""

    sorted_horizons = sorted(float(h) for h in horizons)
    if len(sorted_horizons) < 2:
        raise ValueError("at least two horizons are required")
    designs = _load_designs(comparison_json)
    rows: list[dict[str, Any]] = []
    traces: dict[str, list[dict[str, np.ndarray | int]]] = {}
    for design in designs:
        design_rows: list[dict[str, Any]] = []
        design_traces: list[dict[str, np.ndarray | int]] = []
        for horizon in sorted_horizons:
            row, trace = _window_row(
                design_name=design["design_name"],
                params=design["final_params"],
                horizon=horizon,
                nonlinear_dt=nonlinear_dt,
            )
            design_rows.append(row)
            design_traces.append(trace)
        reference_mean = float(design_rows[-1]["mean"])
        t400_row = min(design_rows, key=lambda row: abs(float(row["t_end"]) - 400.0))
        t400_mean = float(t400_row["mean"])
        for row in design_rows:
            row["relative_change_vs_t400"] = (float(row["mean"]) - t400_mean) / max(
                abs(t400_mean),
                1.0e-14,
            )
            row["relative_change_vs_reference"] = (float(row["mean"]) - reference_mean) / max(
                abs(reference_mean),
                1.0e-14,
            )
            row["reference_t_end"] = float(design_rows[-1]["t_end"])
            rows.append(row)
        traces[design["design_name"]] = design_traces

    metrics: dict[str, Any] = {}
    for design in designs:
        name = design["design_name"]
        design_rows = [row for row in rows if row["design_name"] == name]
        t400_row = min(design_rows, key=lambda row: abs(float(row["t_end"]) - 400.0))
        passed = bool(
            abs(float(t400_row["relative_change_vs_reference"])) <= float(relative_tolerance)
            and float(t400_row["cv"]) <= float(cv_tolerance)
            and float(t400_row["trend"]) <= float(trend_tolerance)
            and float(t400_row["half_window_relative_mean_change"]) <= float(half_window_tolerance)
        )
        metrics[name] = {
            "t400_relative_change_vs_reference": float(t400_row["relative_change_vs_reference"]),
            "t400_cv": float(t400_row["cv"]),
            "t400_trend": float(t400_row["trend"]),
            "t400_half_window_relative_mean_change": float(
                t400_row["half_window_relative_mean_change"]
            ),
            "relative_tolerance": float(relative_tolerance),
            "cv_tolerance": float(cv_tolerance),
            "trend_tolerance": float(trend_tolerance),
            "half_window_tolerance": float(half_window_tolerance),
            "passed": passed,
            "recommendation": (
                "t=400 is sufficient for the reduced envelope"
                if passed
                else "extend the reduced envelope beyond t=400"
            ),
        }
    return {
        "kind": "qa_low_turbulence_time_horizon_audit",
        "comparison_json": str(comparison_json),
        "horizons": sorted_horizons,
        "reference_t_end": sorted_horizons[-1],
        "rows": rows,
        "metrics": metrics,
        "passed": all(bool(metric["passed"]) for metric in metrics.values()),
        "claim_level": "reduced nonlinear-envelope time-horizon audit, not a full-GK convergence claim",
        "scope_notes": [
            "The audit reuses the tracked optimized reduced QA designs and varies only the reduced nonlinear-envelope horizon.",
            "Production nonlinear turbulent-flux claims still require long post-transient replicated SPECTRAX-GK audits.",
        ],
        "_plot_traces": traces,
    }


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(payload: dict[str, Any], path: Path) -> None:
    set_plot_style()
    traces = payload["_plot_traces"]
    rows = payload["rows"]
    horizons = payload["horizons"]
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)

    ax = axs[0, 0]
    for name, series in traces.items():
        trace = series[-1]
        times = np.asarray(trace["times"], dtype=float)
        q = np.asarray(trace["heat_flux"], dtype=float)
        start = int(trace["start_index"])
        ax.plot(times, q, color=COLORS[name], lw=2.0, label=LABELS[name])
        ax.axvline(400.0, color=COLORS[name], ls="--", alpha=0.65)
        ax.axvspan(times[start], times[-1], color=COLORS[name], alpha=0.06)
    ax.set_title("Fixed-gradient reduced-envelope traces to reference horizon")
    ax.set_xlabel(r"$t v_{ti}/a$")
    ax.set_ylabel(r"$Q_{\rm env}(t)$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axs[0, 1]
    for name in traces:
        design_rows = [row for row in rows if row["design_name"] == name]
        ax.plot(
            [row["t_end"] for row in design_rows],
            [row["mean"] for row in design_rows],
            marker="o",
            color=COLORS[name],
            label=LABELS[name],
        )
    ax.set_title("Late-window mean vs horizon")
    ax.set_xlabel(r"$t_{end} v_{ti}/a$")
    ax.set_ylabel(r"late-window $\langle Q_{\rm env}\rangle$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axs[1, 0]
    for name in traces:
        design_rows = [row for row in rows if row["design_name"] == name]
        ax.semilogy(
            [row["t_end"] for row in design_rows],
            [abs(row["relative_change_vs_reference"]) for row in design_rows],
            marker="o",
            color=COLORS[name],
            label=LABELS[name],
        )
    ax.axhline(1.0e-3, color="black", ls=":", lw=1.0, label="0.1%")
    ax.set_title("Horizon error relative to reference mean")
    ax.set_xlabel(r"$t_{end} v_{ti}/a$")
    ax.set_ylabel("relative mean difference")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)

    ax = axs[1, 1]
    x = np.arange(len(horizons))
    width = 0.18
    for i, name in enumerate(traces):
        design_rows = [row for row in rows if row["design_name"] == name]
        ax.bar(
            x + (i - 0.5) * width,
            [row["half_window_relative_mean_change"] for row in design_rows],
            width=width,
            color=COLORS[name],
            alpha=0.85,
            label=LABELS[name],
        )
    ax.axhline(0.02, color="black", ls="--", lw=1.0, label="long-window gate 2%")
    ax.set_xticks(x, [f"{h:g}" for h in horizons])
    ax.set_title("Half-window drift gate")
    ax.set_xlabel(r"$t_{end} v_{ti}/a$")
    ax.set_ylabel("relative first/second-half mean change")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("QA low-turbulence reduced nonlinear time-horizon audit", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_artifacts(payload: dict[str, Any], out_prefix: Path, *, write_pdf: bool = False) -> dict[str, str]:
    payload_to_write = dict(payload)
    payload_to_write.pop("_plot_traces", None)
    paths = {
        "json": str(out_prefix.with_suffix(".json")),
        "csv": str(out_prefix.with_suffix(".csv")),
        "png": str(out_prefix.with_suffix(".png")),
    }
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(
        json.dumps(payload_to_write, separators=(",", ":")),
        encoding="utf-8",
    )
    _write_csv(payload["rows"], out_prefix.with_suffix(".csv"))
    _plot(payload, out_prefix.with_suffix(".png"))
    if write_pdf:
        _plot(payload, out_prefix.with_suffix(".pdf"))
        paths["pdf"] = str(out_prefix.with_suffix(".pdf"))
    return paths


def main() -> int:
    args = _parse_args()
    payload = build_time_horizon_payload(
        args.comparison_json,
        horizons=args.horizons,
        nonlinear_dt=float(args.nonlinear_dt),
        relative_tolerance=float(args.relative_tolerance),
        cv_tolerance=float(args.cv_tolerance),
        trend_tolerance=float(args.trend_tolerance),
        half_window_tolerance=float(args.half_window_tolerance),
    )
    paths = write_artifacts(payload, args.out_prefix, write_pdf=bool(args.pdf))
    print(json.dumps({"passed": payload["passed"], "metrics": payload["metrics"], "paths": paths}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
