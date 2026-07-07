#!/usr/bin/env python3
"""Build a long-window nonlinear turbulence-gradient finite-difference gate."""

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

from spectraxgk.diagnostics.nonlinear_gradient_evidence import (  # noqa: E402
    NonlinearTurbulenceGradientFiniteDifferenceConfig,
    load_json_artifact,
    nonlinear_turbulence_gradient_finite_difference_report,
)


DEFAULT_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_turbulence_gradient_central_fd_gate"
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _format_parameter_label(value: object) -> str:
    """Return a compact plot label for long VMEC profile-direction names."""

    name = str(value or "parameter")
    if name == "profile_direction_zbs_1_1_zbs_1_0_rbc_1_1":
        return "profile direction\nZBS(1,1), ZBS(1,0), RBC(1,1)"
    return name.replace("_", " ") if len(name) <= 32 else name.replace("_", " ", 3)


def write_artifacts(report: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = []
    source_ensembles = report.get("source_ensembles", {})
    if isinstance(source_ensembles, dict):
        for state in ("minus", "baseline", "plus"):
            row = dict(source_ensembles.get(state, {}))
            row["state"] = state
            rows.append(row)
    fieldnames = [
        "state",
        "case",
        "path",
        "passed",
        "ensemble_mean",
        "combined_sem",
        "combined_sem_rel",
        "mean_rel_spread",
        "n_reports",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    labels = ["minus", "baseline", "plus"]
    means = np.asarray(
        [
            _float_or_nan(source_ensembles.get(label, {}).get("ensemble_mean"))
            for label in labels
        ]
    )
    sems = np.asarray(
        [
            _float_or_nan(source_ensembles.get(label, {}).get("combined_sem"))
            for label in labels
        ]
    )
    x = np.arange(len(labels))
    axes[0].errorbar(x, means, yerr=sems, fmt="o", capsize=4, lw=1.8, color="#276b8e")
    axes[0].plot(x, means, "-", lw=1.3, color="#276b8e", alpha=0.65)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("late-window heat flux")
    axes[0].set_title("Matched nonlinear windows")

    metrics = report.get("metrics", {})
    gradient_labels = ["backward", "central", "forward"]
    gradient_values = np.asarray(
        [
            _float_or_nan(metrics.get("backward_gradient")),
            _float_or_nan(metrics.get("central_gradient")),
            _float_or_nan(metrics.get("forward_gradient")),
        ]
    )
    colors = ["#8c6d31", "#3f6f8f", "#b65f2a"]
    axes[1].bar(gradient_labels, gradient_values, color=colors, alpha=0.88)
    axes[1].axhline(0.0, color="0.25", lw=0.9)
    axes[1].set_ylabel(
        f"dQ/dp\n{_format_parameter_label(report.get('parameter_name', 'parameter'))}"
    )
    status = "passed" if bool(report.get("passed", False)) else "blocked"
    axes[1].set_title(f"Central FD gate: {status}")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plus", type=Path, required=True)
    parser.add_argument("--minus", type=Path, required=True)
    parser.add_argument("--delta-parameter", type=float, required=True)
    parser.add_argument(
        "--parameter-name", default="vmec_state_control_or_profile_gradient"
    )
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--min-window-reports", type=int, default=2)
    parser.add_argument("--max-window-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--max-window-combined-sem-rel", type=float, default=0.25)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-condition-number", type=float, default=1.0e8)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = nonlinear_turbulence_gradient_finite_difference_report(
        baseline=load_json_artifact(args.baseline),
        plus=load_json_artifact(args.plus),
        minus=load_json_artifact(args.minus),
        baseline_path=_repo_relative(args.baseline),
        plus_path=_repo_relative(args.plus),
        minus_path=_repo_relative(args.minus),
        delta_parameter=float(args.delta_parameter),
        parameter_name=str(args.parameter_name),
        config=NonlinearTurbulenceGradientFiniteDifferenceConfig(
            min_window_reports=int(args.min_window_reports),
            max_window_mean_rel_spread=float(args.max_window_mean_rel_spread),
            max_window_combined_sem_rel=float(args.max_window_combined_sem_rel),
            max_gradient_uncertainty_rel=float(args.max_gradient_uncertainty_rel),
            max_fd_asymmetry_rel=float(args.max_fd_asymmetry_rel),
            max_fd_condition_number=float(args.max_fd_condition_number),
            min_fd_response_fraction=float(args.min_fd_response_fraction),
        ),
    )
    paths = write_artifacts(report, Path(args.out_prefix))
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "blockers": report["blockers"],
                "paths": paths,
            },
            indent=2,
        )
    )
    return 1 if bool(args.fail_on_blocked) and not bool(report["passed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
