#!/usr/bin/env python3
"""Plot windowed nonlinear GX/SPECTRAX-GK diagnostic agreement.

This script consumes the tracked ``nonlinear_*_gate_summary.json`` files
written by ``tools/compare_gx_nonlinear_diagnostics.py`` and turns their
windowed mismatch statistics into a manuscript-facing summary panel. It does
not rerun simulations; it visualizes the frozen release-window gate metadata.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOB = str(ROOT / "docs" / "_static" / "nonlinear_*_gate_summary.json")
DEFAULT_OUT = ROOT / "docs" / "_static" / "nonlinear_window_statistics.png"
DEFAULT_METRICS = ("Phi2", "Wg", "Wphi", "Wapar", "HeatFlux", "ParticleFlux")
CASE_LABELS = {
    "cyclone_nonlinear_long_window": "Cyclone",
    "cyclone_miller_nonlinear_window": "Cyclone Miller",
    "hsx_nonlinear_window": "HSX",
    "kbm_nonlinear_window": "KBM",
    "w7x_nonlinear_window": "W7-X",
}
CASE_ORDER = {
    "cyclone_nonlinear_long_window": 0,
    "cyclone_miller_nonlinear_window": 1,
    "kbm_nonlinear_window": 2,
    "w7x_nonlinear_window": 3,
    "hsx_nonlinear_window": 4,
}
METRIC_COLORS = {
    "Phi2": "#7b2cbf",
    "Wg": "#0f4c81",
    "Wphi": "#2a9d8f",
    "Wapar": "#6c757d",
    "HeatFlux": "#c44e52",
    "ParticleFlux": "#f4a261",
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


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _repo_relative_pattern(pattern: str) -> str:
    repo = str(ROOT.resolve())
    raw = str(pattern)
    return raw[len(repo) + 1 :] if raw.startswith(repo + "/") else raw


def _case_label(case: str) -> str:
    return CASE_LABELS.get(case, case.replace("_nonlinear_window", "").replace("_", " ").title())


def load_window_rows(paths: list[Path], metrics: tuple[str, ...] = DEFAULT_METRICS) -> list[dict[str, object]]:
    """Load per-case/per-diagnostic windowed mismatch rows from gate summaries."""

    allowed = set(metrics)
    rows: list[dict[str, object]] = []
    for path in sorted(set(paths)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} does not contain a JSON object")
        if data.get("gate_index_include") is False:
            continue
        case = str(data.get("case", path.stem))
        summary = data.get("summary", [])
        if not isinstance(summary, list):
            raise ValueError(f"{path} has a non-list summary")
        for item in summary:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric", ""))
            if metric not in allowed:
                continue
            rows.append(
                {
                    "case": case,
                    "case_label": _case_label(case),
                    "metric": metric,
                    "mean_rel_abs": float(item.get("mean_rel_abs", np.nan)),
                    "max_rel_abs": float(item.get("max_rel_abs", np.nan)),
                    "final_rel": float(item.get("final_rel", np.nan)),
                    "gate_mean_rel": float(data.get("gate_mean_rel", np.nan)),
                    "gate_passed": bool(data.get("gate_passed", False)),
                    "source": str(data.get("source", "")),
                    "artifact": _repo_relative_path(path),
                }
            )
    return sorted(rows, key=lambda row: (CASE_ORDER.get(str(row["case"]), 999), str(row["metric"])))


def _metric_offsets(metrics: list[str]) -> dict[str, float]:
    if len(metrics) == 1:
        return {metrics[0]: 0.0}
    offsets = np.linspace(-0.26, 0.26, len(metrics))
    return {metric: float(offset) for metric, offset in zip(metrics, offsets, strict=True)}


def window_statistics_figure(
    rows: list[dict[str, object]],
    *,
    gate_threshold: float = 0.10,
    title: str = "Windowed nonlinear diagnostic agreement",
) -> plt.Figure:
    """Create the two-panel windowed nonlinear statistics summary figure."""

    if not rows:
        raise ValueError("no nonlinear window rows to plot")
    set_plot_style()
    cases = sorted({str(row["case"]) for row in rows}, key=lambda case: CASE_ORDER.get(case, 999))
    labels = [_case_label(case) for case in cases]
    metrics = [metric for metric in DEFAULT_METRICS if any(str(row["metric"]) == metric for row in rows)]
    offsets = _metric_offsets(metrics)
    y_base = {case: idx for idx, case in enumerate(cases)}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4.3, 0.62 * len(cases) + 1.6)), constrained_layout=True)
    panels = (
        (axes[0], "mean_rel_abs", "Mean relative mismatch", gate_threshold),
        (axes[1], "max_rel_abs", "Maximum relative mismatch", None),
    )
    for ax, key, xlabel, threshold in panels:
        for metric in metrics:
            xs = []
            ys = []
            for row in rows:
                if str(row["metric"]) != metric:
                    continue
                value = float(row[key])
                if not np.isfinite(value):
                    continue
                xs.append(value)
                ys.append(y_base[str(row["case"])] + offsets[metric])
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=54,
                    color=METRIC_COLORS.get(metric, "#333333"),
                    edgecolor="white",
                    linewidth=0.7,
                    label=metric,
                    zorder=3,
                )
        if threshold is not None:
            ax.axvline(float(threshold), color="#c2410c", linestyle="--", linewidth=1.7, label="release gate")
        ax.set_yticks(range(len(cases)), labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", alpha=0.25)
        ax.grid(False, axis="y")
        ax.set_xlim(left=-0.005)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_title("Release-window gate statistic")
    axes[1].set_title("Transient/envelope sensitivity")
    handles, labels_ = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels_, handles, strict=False))
    axes[1].legend(by_label.values(), by_label.keys(), loc="lower right", frameon=True, framealpha=0.92)
    fig.suptitle(title, y=1.03, fontsize=14, fontweight="bold")
    return fig


def write_rows_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write the plotted rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "case_label",
        "metric",
        "mean_rel_abs",
        "max_rel_abs",
        "final_rel",
        "gate_mean_rel",
        "gate_passed",
        "source",
        "artifact",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(rows: list[dict[str, object]], path: Path, *, gate_threshold: float, patterns: list[str]) -> None:
    """Write machine-readable metadata for the plotted nonlinear window panel."""

    cases = sorted({str(row["case"]) for row in rows}, key=lambda case: CASE_ORDER.get(case, 999))
    metrics = sorted({str(row["metric"]) for row in rows})
    max_mean_by_case = {
        case: max(
            float(row["mean_rel_abs"])
            for row in rows
            if str(row["case"]) == case and np.isfinite(float(row["mean_rel_abs"]))
        )
        for case in cases
    }
    payload = {
        "n_cases": len(cases),
        "n_rows": len(rows),
        "cases": cases,
        "metrics": metrics,
        "gate_threshold": float(gate_threshold),
        "all_cases_pass_gate": all(value <= float(gate_threshold) for value in max_mean_by_case.values()),
        "max_mean_rel_abs_by_case": max_mean_by_case,
        "patterns": [_repo_relative_pattern(pattern) for pattern in patterns],
        "rows": rows,
        "notes": (
            "This artifact summarizes frozen nonlinear release-window gate JSONs. "
            "It excludes exploratory summaries with gate_index_include=false and does not rerun simulations."
        ),
    }
    path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", action="append", dest="patterns", default=None, help="Input gate-summary JSON glob.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output PNG path.")
    parser.add_argument("--gate-threshold", type=float, default=0.10, help="Mean-relative release gate threshold.")
    parser.add_argument("--title", default="Windowed nonlinear diagnostic agreement", help="Figure title.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    patterns = args.patterns or [DEFAULT_GLOB]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(item) for item in glob.glob(pattern, recursive=True))
    rows = load_window_rows(paths)
    fig = window_statistics_figure(rows, gate_threshold=float(args.gate_threshold), title=str(args.title))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    write_rows_csv(rows, args.out.with_suffix(".csv"))
    write_summary_json(rows, args.out.with_suffix(".json"), gate_threshold=float(args.gate_threshold), patterns=patterns)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out.with_suffix('.pdf')}")
    print(f"Wrote {args.out.with_suffix('.csv')}")
    print(f"Wrote {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
