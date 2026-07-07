#!/usr/bin/env python3
"""Write an independent control-mean campaign plan for a nonlinear-gradient gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.diagnostics.nonlinear_gradient_evidence import load_json_artifact  # noqa: E402
from tools.campaigns.nonlinear_gradient_followup import (  # noqa: E402
    NonlinearGradientControlVariateCampaignConfig,
)
from tools.campaigns.nonlinear_gradient_followup import (  # noqa: E402
    nonlinear_gradient_control_variate_campaign_plan,
)

DEFAULT_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_control_variate_campaign_plan"
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_index",
        "variant_label",
        "plus_state",
        "minus_state",
        "control_observable",
        "response_observable",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in report.get("planned_pairs", []):
            if isinstance(row, dict):
                writer.writerow({key: row.get(key) for key in fieldnames})


def _plot(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from spectraxgk.artifacts.plotting import set_plot_style

    summary = report.get("summary", {})
    raw = summary.get("raw_response_uncertainty_rel")
    residual = summary.get("residual_uncertainty_rel")
    combined = summary.get("predicted_combined_uncertainty_rel")
    required_pairs = summary.get("required_independent_control_mean_pairs") or 0
    planned_runs = summary.get("planned_new_run_count") or 0
    current_pairs = summary.get("current_common_pair_count") or 0

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0), constrained_layout=True)

    labels = ["raw paired", "CV residual", "projected total"]
    values = np.asarray(
        [
            float(raw) if raw is not None else np.nan,
            float(residual) if residual is not None else np.nan,
            float(combined) if combined is not None else np.nan,
        ],
        dtype=float,
    )
    axes[0].bar(
        np.arange(len(labels)),
        values,
        color=["#4c78a8", "#54a24b", "#f58518"],
        edgecolor="0.25",
    )
    axes[0].axhline(0.5, color="#d62728", ls="--", lw=1.2, label="target")
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    axes[0].set_ylabel("relative uncertainty")
    axes[0].set_title("Uncertainty budget")
    axes[0].legend(frameon=False)

    axes[1].bar(
        [0, 1, 2],
        [float(current_pairs), float(required_pairs), float(planned_runs)],
        color=["#4c78a8", "#54a24b", "#f58518"],
        edgecolor="0.25",
    )
    axes[1].set_xticks([0, 1, 2], ["existing\npairs", "new control\npairs", "new runs"])
    axes[1].set_ylabel("count")
    axes[1].set_title("Bounded launch size")

    pair_count = int(required_pairs)
    preview = min(pair_count, 12)
    y = np.ones(preview)
    axes[2].bar(np.arange(preview), y, color="#72b7b2", edgecolor="0.25")
    if pair_count > preview:
        axes[2].text(
            preview - 0.5,
            0.55,
            f"+{pair_count - preview} more",
            ha="right",
            va="center",
            fontsize=10,
        )
    axes[2].set_xticks(np.arange(preview), [f"{idx + 1}" for idx in range(preview)])
    axes[2].set_yticks([])
    axes[2].set_xlabel("independent matched pair index")
    axes[2].set_title("Control-mean pair plan")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    title = (
        str(report.get("action", "control variate campaign"))
        .replace("_", " ")
        .capitalize()
    )
    fig.suptitle(title, fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variance_report", type=Path)
    parser.add_argument(
        "--case", default="nonlinear_gradient_control_variate_campaign_plan"
    )
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--target-response-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--sem-safety-factor", type=float, default=1.10)
    parser.add_argument("--min-control-mean-pairs", type=int, default=4)
    parser.add_argument("--max-control-mean-pairs", type=int, default=32)
    parser.add_argument("--first-new-seed", type=int, default=34)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = nonlinear_gradient_control_variate_campaign_plan(
        load_json_artifact(args.variance_report),
        case=args.case,
        candidate_name=args.candidate_name,
        config=NonlinearGradientControlVariateCampaignConfig(
            target_response_uncertainty_rel=float(args.target_response_uncertainty_rel),
            sem_safety_factor=float(args.sem_safety_factor),
            min_control_mean_pairs=int(args.min_control_mean_pairs),
            max_control_mean_pairs=int(args.max_control_mean_pairs),
            first_new_seed=int(args.first_new_seed),
        ),
    )
    out_prefix = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(csv_path, report)
    _plot(png_path, report)
    print(
        json.dumps(
            {"action": report["action"], "json": _repo_relative(json_path)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
