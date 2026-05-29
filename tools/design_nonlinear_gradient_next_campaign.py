#!/usr/bin/env python3
"""Design the next nonlinear-gradient campaign from failed FD artifacts."""

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

from spectraxgk.nonlinear_gradient_evidence import load_json_artifact  # noqa: E402
from spectraxgk.nonlinear_gradient_followup import (  # noqa: E402
    NonlinearGradientCandidateDesignConfig,
    nonlinear_gradient_candidate_design_report,
)

DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "nonlinear_gradient_next_campaign_design"


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
        "rank",
        "label",
        "parameter_name",
        "action",
        "response_fraction",
        "fd_asymmetry_rel",
        "gradient_uncertainty_rel",
        "uncertainty_required_bracket_scale",
        "locality_safe_bracket_scale_limit",
        "bracket_only_feasible",
        "current_replicates_per_state",
        "variance_limiting_state",
        "max_mean_rel_spread",
        "failed_spread_states",
        "estimated_required_replicates_at_locality_limit",
        "estimated_extra_replicates_at_locality_limit",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for rank, row in enumerate(report["candidates"], start=1):
            metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
            variance = row.get("variance_reduction", {}) if isinstance(row.get("variance_reduction"), dict) else {}
            writer.writerow(
                {
                    "rank": rank,
                    "label": row.get("label"),
                    "parameter_name": row.get("parameter_name"),
                    "action": row.get("action"),
                    "response_fraction": metrics.get("response_fraction"),
                    "fd_asymmetry_rel": metrics.get("fd_asymmetry_rel"),
                    "gradient_uncertainty_rel": metrics.get("gradient_uncertainty_rel"),
                    "uncertainty_required_bracket_scale": row.get("uncertainty_required_bracket_scale"),
                    "locality_safe_bracket_scale_limit": row.get("locality_safe_bracket_scale_limit"),
                    "bracket_only_feasible": row.get("bracket_only_feasible"),
                    "current_replicates_per_state": row.get("current_replicates_per_state"),
                    "variance_limiting_state": variance.get("limiting_state"),
                    "max_mean_rel_spread": variance.get("max_mean_rel_spread"),
                    "failed_spread_states": ";".join(variance.get("failed_spread_states", []))
                    if isinstance(variance.get("failed_spread_states"), list)
                    else "",
                    "estimated_required_replicates_at_locality_limit": row.get(
                        "estimated_required_replicates_at_locality_limit"
                    ),
                    "estimated_extra_replicates_at_locality_limit": row.get(
                        "estimated_extra_replicates_at_locality_limit"
                    ),
                }
            )


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _plot(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.plotting import set_plot_style

    rows = list(report["candidates"])
    labels = [str(row.get("parameter_name") or row.get("label") or idx) for idx, row in enumerate(rows)]
    x = np.arange(len(rows))
    uncertainty = np.asarray([_float_or_nan(row.get("metrics", {}).get("gradient_uncertainty_rel")) for row in rows])
    asymmetry = np.asarray([_float_or_nan(row.get("metrics", {}).get("fd_asymmetry_rel")) for row in rows])
    needed_scale = np.asarray([_float_or_nan(row.get("uncertainty_required_bracket_scale")) for row in rows])
    locality_scale = np.asarray([_float_or_nan(row.get("locality_safe_bracket_scale_limit")) for row in rows])
    extra = np.asarray([_float_or_nan(row.get("estimated_extra_replicates_at_locality_limit")) for row in rows])
    cfg = report["config"]

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    axes[0].bar(x - 0.18, uncertainty, width=0.36, label="uncertainty", color="#386cb0")
    axes[0].bar(x + 0.18, asymmetry, width=0.36, label="asymmetry", color="#fdb462")
    axes[0].axhline(float(cfg["max_gradient_uncertainty_rel"]), color="#386cb0", ls="--", lw=1.2)
    axes[0].axhline(float(cfg["max_fd_asymmetry_rel"]), color="#b55d00", ls="--", lw=1.2)
    axes[0].set_title("Current FD gate margins")
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].set_ylabel("relative metric")
    axes[0].legend(frameon=False)

    width = 0.36
    axes[1].bar(x - width / 2, needed_scale, width=width, color="#7fc97f", label="needed for uncertainty")
    axes[1].bar(x + width / 2, locality_scale, width=width, color="#beaed4", label="locality-safe limit")
    axes[1].axhline(1.0, color="0.3", lw=0.9)
    axes[1].set_title("Bracket feasibility")
    axes[1].set_xticks(x, labels, rotation=22, ha="right")
    axes[1].set_ylabel("bracket scale")
    axes[1].legend(frameon=False)

    axes[2].bar(x, extra, color="#fb8072", edgecolor="0.25", linewidth=0.6)
    axes[2].axhline(float(cfg["max_extra_replicates_per_state"]), color="0.25", ls="--", lw=1.2)
    axes[2].set_title("Extra replicas after locality cap")
    axes[2].set_xticks(x, labels, rotation=22, ha="right")
    axes[2].set_ylabel("replicas per state")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Next nonlinear-gradient campaign design", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="+", type=Path)
    parser.add_argument("--case", default="qa_ess_nonlinear_gradient_next_campaign_design")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument("--sem-safety-factor", type=float, default=1.10)
    parser.add_argument("--max-extra-replicates-per-state", type=int, default=4)
    parser.add_argument("--max-checked-bracket-scale", type=float, default=1.50)
    parser.add_argument("--locality-safety-factor", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = [load_json_artifact(path) for path in args.artifact]
    report = nonlinear_gradient_candidate_design_report(
        artifacts,
        paths=[_repo_relative(path) for path in args.artifact],
        labels=[_label(payload, path) for payload, path in zip(artifacts, args.artifact)],
        case=args.case,
        config=NonlinearGradientCandidateDesignConfig(
            max_gradient_uncertainty_rel=args.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=args.max_fd_asymmetry_rel,
            min_fd_response_fraction=args.min_fd_response_fraction,
            sem_safety_factor=args.sem_safety_factor,
            max_extra_replicates_per_state=args.max_extra_replicates_per_state,
            max_checked_bracket_scale=args.max_checked_bracket_scale,
            locality_safety_factor=args.locality_safety_factor,
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
    print(json.dumps({"json": _repo_relative(json_path), "passed": report["passed"], "next_action": report["next_action"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
