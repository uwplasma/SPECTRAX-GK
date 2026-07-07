#!/usr/bin/env python3
"""Summarize replicated nonlinear-window spread from ensemble JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.diagnostics.nonlinear_replicates import (  # noqa: E402
    NonlinearReplicateSpreadConfig,
    nonlinear_replicate_spread_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ensembles", nargs="+", type=Path, help="Nonlinear ensemble JSON files."
    )
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--case", default="nonlinear_replicate_spread_diagnostic")
    parser.add_argument("--max-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--value-floor", type=float, default=1.0e-12)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a JSON object")
    return payload


def _candidate_paths(raw: object, *, ensemble_path: Path) -> list[Path]:
    if not isinstance(raw, str) or not raw:
        return []
    path = Path(raw)
    if path.is_absolute():
        return [path]
    return [
        ROOT / path,
        ensemble_path.parent / path,
        ensemble_path.parent / path.name,
    ]


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _convergence_path(summary_path: Path) -> Path | None:
    candidate = (
        summary_path.parent
        / "nonlinear_window_convergence_reports"
        / f"{summary_path.stem}.convergence.json"
    )
    if candidate.exists():
        return candidate
    sibling = summary_path.with_suffix(".convergence.json")
    return sibling if sibling.exists() else None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _enrich_ensemble(payload: dict[str, Any], *, ensemble_path: Path) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return payload
    for row in rows:
        if not isinstance(row, dict):
            continue
        summary_path = _first_existing(
            _candidate_paths(row.get("summary_artifact"), ensemble_path=ensemble_path)
        )
        if summary_path is not None:
            summary = _read_json(summary_path)
            for key in ("variant_label", "variant_axis", "variant", "seed", "dt"):
                if key in summary and key not in row:
                    row[key] = summary[key]
            convergence_path = _convergence_path(summary_path)
            if convergence_path is not None:
                convergence = _read_json(convergence_path)
                stats = convergence.get("statistics")
                if isinstance(stats, dict):
                    row["window_statistics"] = stats
                row["convergence_artifact"] = _display_path(convergence_path)
    return payload


def _write_csv(report: dict[str, Any], out_csv: Path) -> None:
    rows = list(report.get("replicate_rows", []))
    fieldnames = [
        "state",
        "index",
        "variant_label",
        "variant_axis",
        "late_mean",
        "sem",
        "ensemble_mean",
        "relative_delta",
        "running_mean_rel_drift",
        "terminal_mean_rel_delta",
        "sem_rel",
        "n_blocks",
        "passed",
        "promotion_ready",
        "source_artifact",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_png(report: dict[str, Any], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    replicate_rows = list(report.get("replicate_rows", []))
    state_rows = {str(row["state"]): row for row in report.get("state_rows", [])}
    states = list(dict.fromkeys(str(row["state"]) for row in replicate_rows))
    offsets = {"seed": -0.24, "timestep": 0.24, "seed_timestep": 0.0, "unknown": 0.0}
    colors = {
        "seed": "#2563eb",
        "timestep": "#d97706",
        "seed_timestep": "#4b5563",
        "unknown": "#6b7280",
    }

    set_plot_style()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 4.8), constrained_layout=True)
    plotted_values: list[float] = []
    for state_index, state in enumerate(states):
        group = [row for row in replicate_rows if str(row["state"]) == state]
        for local_index, row in enumerate(group):
            axis = str(row.get("variant_axis") or "unknown")
            offset = offsets.get(axis, 0.0) + 0.06 * (
                local_index - (len(group) - 1) / 2
            )
            x = state_index + offset
            mean = row.get("late_mean")
            if mean is None:
                continue
            sem = 0.0 if row.get("sem") is None else float(row["sem"])
            plotted_values.extend([float(mean) - sem, float(mean) + sem])
            ax.errorbar(
                [x],
                [float(mean)],
                yerr=[sem],
                fmt="o",
                ms=6.0,
                capsize=3.0,
                color=colors.get(axis, colors["unknown"]),
                label=axis if axis not in ax.get_legend_handles_labels()[1] else None,
            )
            ax.text(
                x,
                float(mean) + max(sem, 0.05),
                str(row.get("variant_label", "")),
                rotation=45,
                ha="left",
                va="bottom",
                fontsize=6.5,
            )
        state_summary = state_rows.get(state, {})
        ensemble_mean = state_summary.get("ensemble_mean")
        if ensemble_mean is not None:
            ax.hlines(
                float(ensemble_mean),
                state_index - 0.36,
                state_index + 0.36,
                color="0.2",
                lw=1.2,
                ls="--",
            )
        if state_summary.get("classification") != "passed_replicate_spread_gate":
            ax.axvspan(
                state_index - 0.45,
                state_index + 0.45,
                color="#fee2e2",
                alpha=0.35,
                lw=0,
            )

    ax.set_xticks(np.arange(len(states)), states)
    if plotted_values:
        ymin = min(plotted_values)
        ymax = max(plotted_values)
        pad = max(0.25, 0.16 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("post-transient ion heat flux")
    ax.set_title("Replicated nonlinear-window spread diagnostic")
    ax.grid(True, axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="best", fontsize=8)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensembles = [
        _enrich_ensemble(_read_json(path), ensemble_path=path)
        for path in args.ensembles
    ]
    report = nonlinear_replicate_spread_report(
        ensembles,
        case=args.case,
        config=NonlinearReplicateSpreadConfig(
            max_mean_rel_spread=args.max_mean_rel_spread,
            value_floor=args.value_floor,
        ),
    )
    out_json = args.out_prefix.with_suffix(".json")
    out_csv = args.out_prefix.with_suffix(".csv")
    out_png = args.out_prefix.with_suffix(".png")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(report, out_csv)
    _write_png(report, out_png)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
