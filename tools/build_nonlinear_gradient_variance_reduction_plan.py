#!/usr/bin/env python3
"""Build a paired-seed/control-variate plan for a nonlinear-gradient artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.nonlinear_gradient.evidence import load_json_artifact  # noqa: E402
from spectraxgk.validation.nonlinear_gradient.followup_core import (  # noqa: E402
    NonlinearGradientVarianceReductionConfig,
)
from spectraxgk.validation.nonlinear_gradient.followup_variance import (  # noqa: E402
    nonlinear_gradient_variance_reduction_plan,
)


DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "nonlinear_gradient_variance_reduction_plan"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _label(payload: dict[str, Any], path: Path) -> str:
    parameter = payload.get("parameter_name")
    if isinstance(parameter, str) and parameter:
        return f"{parameter}:{path.stem.removesuffix('_central_fd_gradient_gate')}"
    return path.stem


def _write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "plus_mean",
        "minus_mean",
        "baseline_mean",
        "plus_minus_difference",
        "plus_baseline_difference",
        "baseline_minus_difference",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in report.get("pair_rows", []):
            writer.writerow({key: row.get(key) for key in fieldnames})


def _plot(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from spectraxgk.artifacts.plotting import set_plot_style

    rows = list(report.get("pair_rows", []))
    labels = [str(row.get("label")) for row in rows]
    differences = np.asarray([float(row.get("plus_minus_difference", np.nan)) for row in rows], dtype=float)
    summary = report.get("summary", {})
    paired_mean = summary.get("paired_response_mean")
    paired_sem = summary.get("paired_response_sem")

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].bar(x, differences, color="#4c78a8", edgecolor="0.25", linewidth=0.6)
    if paired_mean is not None:
        axes[0].axhline(float(paired_mean), color="#d62728", lw=1.5, label="paired mean")
    axes[0].axhline(0.0, color="0.3", lw=0.8)
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Q_plus - Q_minus")
    axes[0].set_title("Matched-pair response samples")
    axes[0].legend(frameon=False)

    states = report.get("variance_reduction", {}).get("state_rows", [])
    state_labels = [str(row.get("state")) for row in states]
    spreads = np.asarray([float(row.get("mean_rel_spread", np.nan)) for row in states], dtype=float)
    axes[1].bar(np.arange(len(states)), spreads, color="#f58518", edgecolor="0.25", linewidth=0.6)
    axes[1].axhline(0.15, color="#d62728", ls="--", lw=1.2, label="spread gate")
    axes[1].set_xticks(np.arange(len(states)), state_labels)
    axes[1].set_ylabel("mean relative spread")
    axes[1].set_title("Replicate variance limiter")
    axes[1].legend(frameon=False)

    candidates = list(report.get("control_variate_candidates", []))
    candidate_labels = [str(row.get("name", idx)).replace("_common_mode", "") for idx, row in enumerate(candidates)]
    raw_rel = summary.get("paired_response_uncertainty_rel")
    adjusted_rel = np.asarray(
        [float(row.get("adjusted_response_uncertainty_rel", np.nan)) for row in candidates],
        dtype=float,
    )
    axes[2].bar(np.arange(len(candidates)), adjusted_rel, color="#54a24b", edgecolor="0.25", linewidth=0.6)
    if raw_rel is not None:
        axes[2].axhline(float(raw_rel), color="0.35", lw=1.2, label="paired raw")
    axes[2].axhline(0.5, color="#d62728", ls="--", lw=1.2, label="target")
    axes[2].set_xticks(np.arange(len(candidates)), candidate_labels, rotation=25, ha="right")
    axes[2].set_ylabel("relative uncertainty")
    axes[2].set_title("Apparent control-variate screen")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    rel = summary.get("paired_response_uncertainty_rel")
    fig.suptitle(
        "Nonlinear-gradient variance-reduction plan"
        + (f"\npaired SEM={paired_sem:.3g}, relative uncertainty={rel:.3g}" if paired_sem is not None and rel is not None else ""),
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--case", default="nonlinear_gradient_variance_reduction_plan")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--max-paired-response-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-control-variate-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--min-control-variate-sem-reduction", type=float, default=0.25)
    parser.add_argument(
        "--allow-sample-control-mean",
        action="store_true",
        help=(
            "Allow sample-centered control variates to pass. Omit for production planning, "
            "where a known or independently estimated control mean is required."
        ),
    )
    parser.add_argument("--sem-safety-factor", type=float, default=1.10)
    parser.add_argument("--min-common-pairs", type=int, default=2)
    parser.add_argument("--max-extra-paired-seeds", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = load_json_artifact(args.artifact)
    report = nonlinear_gradient_variance_reduction_plan(
        payload,
        path=_repo_relative(args.artifact),
        label=_label(payload, args.artifact),
        case=args.case,
        config=NonlinearGradientVarianceReductionConfig(
            max_paired_response_uncertainty_rel=args.max_paired_response_uncertainty_rel,
            max_control_variate_uncertainty_rel=args.max_control_variate_uncertainty_rel,
            min_control_variate_sem_reduction=args.min_control_variate_sem_reduction,
            require_known_control_mean=not bool(args.allow_sample_control_mean),
            sem_safety_factor=args.sem_safety_factor,
            min_common_pairs=args.min_common_pairs,
            max_extra_paired_seeds=args.max_extra_paired_seeds,
        ),
    )
    out_prefix = args.out_prefix
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(csv_path, report)
    _plot(png_path, report)
    print(
        json.dumps(
            {
                "json": _repo_relative(json_path),
                "passed": report["passed"],
                "action": report["action"],
                "recommendation": report["recommendation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
