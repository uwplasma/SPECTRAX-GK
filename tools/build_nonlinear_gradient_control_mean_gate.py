#!/usr/bin/env python3
"""Build the independent control-mean gate for a nonlinear-gradient CV screen."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.nonlinear_gradient.evidence import load_json_artifact  # noqa: E402
from spectraxgk.validation.nonlinear_gradient.followup import (  # noqa: E402
    NonlinearGradientControlMeanGateConfig,
    nonlinear_gradient_control_mean_gate,
)

DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "nonlinear_gradient_control_mean_gate"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "plus_mean",
        "minus_mean",
        "control_mean_sample",
        "response_sample",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in report.get("pair_rows", []):
            if isinstance(row, dict):
                writer.writerow({key: row.get(key) for key in fieldnames})


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _plot(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    summary = report.get("summary", {})
    rows = [row for row in report.get("pair_rows", []) if isinstance(row, dict)]
    labels = [str(row.get("label", idx)) for idx, row in enumerate(rows)]
    controls = np.asarray([_float_or_nan(row.get("control_mean_sample")) for row in rows])
    responses = np.asarray([_float_or_nan(row.get("response_sample")) for row in rows])
    budget_labels = ["residual", "control mean", "combined"]
    budget = np.asarray(
        [
            _float_or_nan(summary.get("residual_uncertainty_rel")),
            _float_or_nan(summary.get("control_contribution_sem"))
            / max(abs(_float_or_nan(summary.get("paired_response_mean"))), 1.0e-12),
            _float_or_nan(summary.get("combined_response_uncertainty_rel")),
        ]
    )

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].bar(x, controls, color="#72b7b2", edgecolor="0.25", linewidth=0.5)
    if np.isfinite(controls).any():
        axes[0].axhline(float(np.nanmean(controls)), color="#1b4d4a", lw=1.3, label="mean")
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("0.5 * (Q_plus + Q_minus)")
    axes[0].set_title("Independent control samples")
    axes[0].legend(frameon=False)

    axes[1].bar(x, responses, color="#4c78a8", edgecolor="0.25", linewidth=0.5)
    axes[1].axhline(0.0, color="0.3", lw=0.9)
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylabel("Q_plus - Q_minus")
    axes[1].set_title("Held-out response samples")

    axes[2].bar(np.arange(3), budget, color=["#54a24b", "#f58518", "#b279a2"], edgecolor="0.25")
    axes[2].axhline(0.5, color="#d62728", ls="--", lw=1.2, label="target")
    axes[2].set_xticks(np.arange(3), budget_labels, rotation=20, ha="right")
    axes[2].set_ylabel("relative uncertainty")
    axes[2].set_title("Combined CV uncertainty")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Independent control-mean gate: " + ("passed" if report.get("passed") else "blocked"), fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variance-report", required=True, type=Path)
    parser.add_argument("--plus-ensemble", required=True, type=Path)
    parser.add_argument("--minus-ensemble", required=True, type=Path)
    parser.add_argument("--case", default="nonlinear_gradient_control_mean_gate")
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--target-response-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--min-control-mean-pairs", type=int, default=4)
    parser.add_argument("--allow-failed-state-ensembles", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = nonlinear_gradient_control_mean_gate(
        load_json_artifact(args.variance_report),
        plus_ensemble=load_json_artifact(args.plus_ensemble),
        minus_ensemble=load_json_artifact(args.minus_ensemble),
        plus_path=_repo_relative(args.plus_ensemble),
        minus_path=_repo_relative(args.minus_ensemble),
        case=args.case,
        candidate_name=args.candidate_name,
        config=NonlinearGradientControlMeanGateConfig(
            target_response_uncertainty_rel=float(args.target_response_uncertainty_rel),
            min_control_mean_pairs=int(args.min_control_mean_pairs),
            require_state_ensembles_passed=not bool(args.allow_failed_state_ensembles),
        ),
    )
    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, report)
    _plot(png_path, report)
    print(json.dumps({"passed": report["passed"], "blockers": report["blockers"], "json": _repo_relative(json_path)}, indent=2))
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
