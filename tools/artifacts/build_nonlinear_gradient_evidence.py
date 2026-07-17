#!/usr/bin/env python3
"""Build nonlinear-gradient evidence artifacts from replicated transport data.

Subcommands:

* ``finite-difference`` builds the matched three-state central-difference gate.
* ``rank-candidates`` ranks finite-difference candidates without planning runs.
* ``bracket-sweep`` summarizes perturbation-locality and uncertainty evidence.
* ``variance-plan`` summarizes paired plus/minus nonlinear-gradient evidence and
  plans the control-variate campaign needed to reduce uncertainty.
* ``control-mean`` gates the independent control-mean campaign once the
  plus/minus ensembles have been run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.diagnostics.nonlinear_gradient_evidence import (  # noqa: E402
    NonlinearTurbulenceGradientBracketSweepConfig,
    NonlinearTurbulenceGradientCandidateRankingConfig,
    NonlinearTurbulenceGradientFiniteDifferenceConfig,
    load_json_artifact,
    nonlinear_turbulence_gradient_bracket_sweep_report,
    nonlinear_turbulence_gradient_candidate_ranking_report,
    nonlinear_turbulence_gradient_finite_difference_report,
)
from spectraxgk.diagnostics.nonlinear_gradient_statistics import (  # noqa: E402
    NonlinearGradientControlMeanGateConfig,
    NonlinearGradientVarianceReductionConfig,
    nonlinear_gradient_control_mean_gate,
    nonlinear_gradient_variance_reduction_plan,
)


DEFAULT_VARIANCE_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_variance_reduction_plan"
)
DEFAULT_CONTROL_MEAN_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_control_mean_gate"
)
DEFAULT_FINITE_DIFFERENCE_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_turbulence_gradient_central_fd_gate"
)
DEFAULT_BRACKET_SWEEP_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_bracket_sweep"
)


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


def _candidate_artifact_label(payload: dict[str, Any], path: Path) -> str:
    parameter = payload.get("parameter_name")
    if isinstance(parameter, str) and parameter:
        return parameter
    return path.stem.removesuffix("_central_fd_gradient_gate")


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _format_parameter_label(value: object) -> str:
    """Return a compact plot label for long VMEC profile-direction names."""

    name = str(value or "parameter")
    if name == "profile_direction_zbs_1_1_zbs_1_0_rbc_1_1":
        return "profile direction\nZBS(1,1), ZBS(1,0), RBC(1,1)"
    return name.replace("_", " ") if len(name) <= 32 else name.replace("_", " ", 3)


def _write_finite_difference_artifacts(
    report: dict[str, Any], out_prefix: Path
) -> dict[str, str]:
    """Write the central finite-difference JSON, table, and figure."""

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

    source_ensembles = report.get("source_ensembles", {})
    rows = []
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
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
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
    axes[1].bar(
        gradient_labels,
        gradient_values,
        color=["#8c6d31", "#3f6f8f", "#b65f2a"],
        alpha=0.88,
    )
    axes[1].axhline(0.0, color="0.25", lw=0.9)
    axes[1].set_ylabel(
        f"dQ/dp\n{_format_parameter_label(report.get('parameter_name', 'parameter'))}"
    )
    status = "passed" if bool(report.get("passed", False)) else "blocked"
    axes[1].set_title(f"Central FD gate: {status}")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def _write_variance_csv(path: Path, report: dict[str, Any]) -> None:
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


def _write_control_mean_csv(path: Path, report: dict[str, Any]) -> None:
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


def _plot_variance(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    rows = list(report.get("pair_rows", []))
    labels = [str(row.get("label")) for row in rows]
    differences = np.asarray(
        [float(row.get("plus_minus_difference", np.nan)) for row in rows], dtype=float
    )
    summary = report.get("summary", {})
    paired_mean = summary.get("paired_response_mean")
    paired_sem = summary.get("paired_response_sem")

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].bar(x, differences, color="#4c78a8", edgecolor="0.25", linewidth=0.6)
    if paired_mean is not None:
        axes[0].axhline(
            float(paired_mean), color="#d62728", lw=1.5, label="paired mean"
        )
    axes[0].axhline(0.0, color="0.3", lw=0.8)
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Q_plus - Q_minus")
    axes[0].set_title("Matched-pair response samples")
    axes[0].legend(frameon=False)

    states = report.get("variance_reduction", {}).get("state_rows", [])
    state_labels = [str(row.get("state")) for row in states]
    spreads = np.asarray(
        [float(row.get("mean_rel_spread", np.nan)) for row in states], dtype=float
    )
    axes[1].bar(
        np.arange(len(states)),
        spreads,
        color="#f58518",
        edgecolor="0.25",
        linewidth=0.6,
    )
    axes[1].axhline(0.15, color="#d62728", ls="--", lw=1.2, label="spread gate")
    axes[1].set_xticks(np.arange(len(states)), state_labels)
    axes[1].set_ylabel("mean relative spread")
    axes[1].set_title("Replicate variance limiter")
    axes[1].legend(frameon=False)

    candidates = list(report.get("control_variate_candidates", []))
    candidate_labels = [
        str(row.get("name", idx)).replace("_common_mode", "")
        for idx, row in enumerate(candidates)
    ]
    raw_rel = summary.get("paired_response_uncertainty_rel")
    adjusted_rel = np.asarray(
        [
            float(row.get("adjusted_response_uncertainty_rel", np.nan))
            for row in candidates
        ],
        dtype=float,
    )
    axes[2].bar(
        np.arange(len(candidates)),
        adjusted_rel,
        color="#54a24b",
        edgecolor="0.25",
        linewidth=0.6,
    )
    if raw_rel is not None:
        axes[2].axhline(float(raw_rel), color="0.35", lw=1.2, label="paired raw")
    axes[2].axhline(0.5, color="#d62728", ls="--", lw=1.2, label="target")
    axes[2].set_xticks(
        np.arange(len(candidates)), candidate_labels, rotation=25, ha="right"
    )
    axes[2].set_ylabel("relative uncertainty")
    axes[2].set_title("Apparent control-variate screen")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    rel = summary.get("paired_response_uncertainty_rel")
    fig.suptitle(
        "Nonlinear-gradient variance-reduction plan"
        + (
            f"\npaired SEM={paired_sem:.3g}, relative uncertainty={rel:.3g}"
            if paired_sem is not None and rel is not None
            else ""
        ),
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_control_mean(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    summary = report.get("summary", {})
    rows = [row for row in report.get("pair_rows", []) if isinstance(row, dict)]
    labels = [str(row.get("label", idx)) for idx, row in enumerate(rows)]
    controls = np.asarray(
        [_float_or_nan(row.get("control_mean_sample")) for row in rows]
    )
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
        axes[0].axhline(
            float(np.nanmean(controls)), color="#1b4d4a", lw=1.3, label="mean"
        )
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("0.5 * (Q_plus + Q_minus)")
    axes[0].set_title("Independent control samples")
    axes[0].legend(frameon=False)

    axes[1].bar(x, responses, color="#4c78a8", edgecolor="0.25", linewidth=0.5)
    axes[1].axhline(0.0, color="0.3", lw=0.9)
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].set_ylabel("Q_plus - Q_minus")
    axes[1].set_title("Held-out response samples")

    axes[2].bar(
        np.arange(3), budget, color=["#54a24b", "#f58518", "#b279a2"], edgecolor="0.25"
    )
    axes[2].axhline(0.5, color="#d62728", ls="--", lw=1.2, label="target")
    axes[2].set_xticks(np.arange(3), budget_labels, rotation=20, ha="right")
    axes[2].set_ylabel("relative uncertainty")
    axes[2].set_title("Combined CV uncertainty")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(
        "Independent control-mean gate: "
        + ("passed" if report.get("passed") else "blocked"),
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _bracket_sweep_label(payload: dict[str, Any], path: Path) -> str:
    parameter = payload.get("parameter_name")
    delta = payload.get("delta_parameter")
    if isinstance(parameter, str) and parameter:
        delta_value = _float_or_nan(delta)
        if not np.isfinite(delta_value):
            return parameter
        return f"{parameter}:delta={delta_value:.4g}"
    return path.stem


def _write_bracket_sweep_artifacts(
    report: dict[str, Any],
    out_prefix: Path,
    *,
    write_pdf: bool = True,
) -> dict[str, str]:
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
    fieldnames = [
        "label",
        "path",
        "parameter_name",
        "delta_parameter",
        "passed",
        "central_gradient",
        "response_fraction",
        "fd_asymmetry_rel",
        "gradient_uncertainty_rel",
        "paired_gradient_uncertainty_rel",
        "paired_same_sign_fraction",
        "repeated_bracket_stable",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in report.get("brackets", []):
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            writer.writerow(
                {
                    "label": row.get("label"),
                    "path": row.get("path"),
                    "parameter_name": row.get("parameter_name"),
                    "delta_parameter": row.get("delta_parameter"),
                    "passed": row.get("passed"),
                    "central_gradient": metrics.get("central_gradient"),
                    "response_fraction": metrics.get("response_fraction"),
                    "fd_asymmetry_rel": metrics.get("fd_asymmetry_rel"),
                    "gradient_uncertainty_rel": metrics.get("gradient_uncertainty_rel"),
                    "paired_gradient_uncertainty_rel": metrics.get(
                        "paired_gradient_uncertainty_rel"
                    ),
                    "paired_same_sign_fraction": metrics.get(
                        "paired_same_sign_fraction"
                    ),
                    "repeated_bracket_stable": metrics.get("repeated_bracket_stable"),
                }
            )

    set_plot_style()
    rows = [row for row in report.get("brackets", []) if isinstance(row, dict)]
    deltas = np.asarray([_float_or_nan(row.get("delta_parameter")) for row in rows])
    gradients = np.asarray(
        [_float_or_nan(row.get("metrics", {}).get("central_gradient")) for row in rows]
    )
    response = np.asarray(
        [_float_or_nan(row.get("metrics", {}).get("response_fraction")) for row in rows]
    )
    asymmetry = np.asarray(
        [_float_or_nan(row.get("metrics", {}).get("fd_asymmetry_rel")) for row in rows]
    )
    uncertainty = np.asarray(
        [
            _float_or_nan(row.get("metrics", {}).get("gradient_uncertainty_rel"))
            for row in rows
        ]
    )
    labels = [str(row.get("label", "")) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), constrained_layout=True)
    marker_colors = [
        "#24727f" if bool(row.get("passed", False)) else "#b45f2a" for row in rows
    ]
    axes[0].scatter(deltas, gradients, c=marker_colors, s=70, zorder=3)
    axes[0].plot(deltas, gradients, color="#46545f", lw=1.2, alpha=0.65)
    axes[0].axhline(0.0, color="0.25", lw=0.8)
    axes[0].set_xlabel("perturbation amplitude")
    axes[0].set_ylabel("central dQ/dp")
    axes[0].set_title("Gradient locality sweep")

    axes[1].plot(deltas, response, "o-", color="#24727f", label="response")
    axes[1].axhline(
        float(report["config"]["min_fd_response_fraction"]),
        color="#24727f",
        ls="--",
        lw=1.0,
        alpha=0.65,
    )
    axes[1].set_xlabel("perturbation amplitude")
    axes[1].set_ylabel("response fraction")
    axes[1].set_title("Resolved response")

    axes[2].plot(deltas, asymmetry, "o-", color="#7b4c9a", label="FD asymmetry")
    axes[2].plot(deltas, uncertainty, "s-", color="#b45f2a", label="uncertainty")
    axes[2].axhline(
        float(report["config"]["max_fd_asymmetry_rel"]),
        color="#7b4c9a",
        ls="--",
        lw=1.0,
        alpha=0.55,
    )
    axes[2].axhline(
        float(report["config"]["max_gradient_uncertainty_rel"]),
        color="#b45f2a",
        ls="--",
        lw=1.0,
        alpha=0.55,
    )
    axes[2].set_xlabel("perturbation amplitude")
    axes[2].set_ylabel("relative metric")
    axes[2].set_title("Locality and uncertainty")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    for delta, gradient, label in zip(deltas, gradients, labels):
        if np.isfinite(delta) and np.isfinite(gradient):
            axes[0].annotate(
                label.split(":")[-1],
                (delta, gradient),
                xytext=(3, 4),
                textcoords="offset points",
                fontsize=7,
            )
    fig.suptitle(str(report.get("recommendation", "")), fontsize=10)
    fig.savefig(png_path, dpi=220)
    paths = {
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path),
    }
    if write_pdf:
        fig.savefig(pdf_path)
        paths["pdf"] = str(pdf_path)
    plt.close(fig)
    return paths


def _add_rank_candidates_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "rank-candidates", help="rank nonlinear-gradient finite-difference candidates"
    )
    parser.add_argument("artifact", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-condition-number", type=float, default=1.0e8)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument(
        "--campaign-context",
        choices=("single_control_screen", "overdetermined_followup"),
        default="single_control_screen",
    )
    parser.add_argument("--fail-on-no-promotable", action="store_true")
    parser.set_defaults(func=_run_rank_candidates)


def _add_bracket_sweep_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "bracket-sweep", help="summarize a same-control perturbation sweep"
    )
    parser.add_argument("artifact", nargs="+", type=Path)
    parser.add_argument(
        "--json-out-prefix", type=Path, default=DEFAULT_BRACKET_SWEEP_OUT_PREFIX
    )
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-condition-number", type=float, default=1.0e8)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument(
        "--max-repeated-bracket-uncertainty-rel", type=float, default=0.75
    )
    parser.add_argument(
        "--min-repeated-bracket-same-sign-fraction", type=float, default=0.80
    )
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--fail-on-no-promotable", action="store_true")
    parser.set_defaults(func=_run_bracket_sweep)


def _add_variance_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "variance-plan", help="build a paired-seed variance-reduction plan"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--case", default="nonlinear_gradient_variance_reduction_plan")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_VARIANCE_OUT_PREFIX)
    parser.add_argument(
        "--max-paired-response-uncertainty-rel", type=float, default=0.50
    )
    parser.add_argument(
        "--max-control-variate-uncertainty-rel", type=float, default=0.50
    )
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
    parser.set_defaults(func=_run_variance_plan)


def _add_finite_difference_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "finite-difference", help="build the matched central finite-difference gate"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plus", type=Path, required=True)
    parser.add_argument("--minus", type=Path, required=True)
    parser.add_argument("--delta-parameter", type=float, required=True)
    parser.add_argument(
        "--parameter-name", default="vmec_state_control_or_profile_gradient"
    )
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_FINITE_DIFFERENCE_OUT_PREFIX
    )
    parser.add_argument("--min-window-reports", type=int, default=2)
    parser.add_argument("--max-window-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--max-window-combined-sem-rel", type=float, default=0.25)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-condition-number", type=float, default=1.0e8)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument("--fail-on-blocked", action="store_true")
    parser.set_defaults(func=_run_finite_difference)


def _add_control_mean_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "control-mean", help="gate an independent control-mean campaign"
    )
    parser.add_argument("--variance-report", required=True, type=Path)
    parser.add_argument("--plus-ensemble", required=True, type=Path)
    parser.add_argument("--minus-ensemble", required=True, type=Path)
    parser.add_argument("--case", default="nonlinear_gradient_control_mean_gate")
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_CONTROL_MEAN_OUT_PREFIX
    )
    parser.add_argument("--target-response-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--min-control-mean-pairs", type=int, default=4)
    parser.add_argument("--allow-failed-state-ensembles", action="store_true")
    parser.set_defaults(func=_run_control_mean)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_finite_difference_parser(subparsers)
    _add_rank_candidates_parser(subparsers)
    _add_bracket_sweep_parser(subparsers)
    _add_variance_parser(subparsers)
    _add_control_mean_parser(subparsers)
    return parser


def _run_finite_difference(args: argparse.Namespace) -> int:
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
    paths = _write_finite_difference_artifacts(report, Path(args.out_prefix))
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


def _run_rank_candidates(args: argparse.Namespace) -> int:
    artifacts = [load_json_artifact(path) for path in args.artifact]
    report = nonlinear_turbulence_gradient_candidate_ranking_report(
        artifacts,
        paths=[_repo_relative(path) for path in args.artifact],
        labels=[
            _candidate_artifact_label(payload, path)
            for payload, path in zip(artifacts, args.artifact)
        ],
        config=NonlinearTurbulenceGradientCandidateRankingConfig(
            max_gradient_uncertainty_rel=args.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=args.max_fd_asymmetry_rel,
            max_fd_condition_number=args.max_fd_condition_number,
            min_fd_response_fraction=args.min_fd_response_fraction,
            campaign_context=args.campaign_context,
        ),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is None:
        print(text)
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {args.json_out}")
    if args.fail_on_no_promotable and not bool(report.get("passed", False)):
        print(
            "no nonlinear-gradient candidate passes production gates", file=sys.stderr
        )
        return 1
    return 0


def _run_bracket_sweep(args: argparse.Namespace) -> int:
    payloads = [load_json_artifact(path) for path in args.artifact]
    report = nonlinear_turbulence_gradient_bracket_sweep_report(
        payloads,
        labels=[
            _bracket_sweep_label(payload, path)
            for payload, path in zip(payloads, args.artifact)
        ],
        paths=[_repo_relative(path) for path in args.artifact],
        config=NonlinearTurbulenceGradientBracketSweepConfig(
            max_gradient_uncertainty_rel=float(args.max_gradient_uncertainty_rel),
            max_fd_asymmetry_rel=float(args.max_fd_asymmetry_rel),
            max_fd_condition_number=float(args.max_fd_condition_number),
            min_fd_response_fraction=float(args.min_fd_response_fraction),
            max_repeated_bracket_uncertainty_rel=float(
                args.max_repeated_bracket_uncertainty_rel
            ),
            min_repeated_bracket_same_sign_fraction=float(
                args.min_repeated_bracket_same_sign_fraction
            ),
        ),
    )
    paths = _write_bracket_sweep_artifacts(
        report, Path(args.json_out_prefix), write_pdf=not bool(args.no_pdf)
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "recommendation": report["recommendation"],
                "paths": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_no_promotable and not report["passed"] else 0


def _run_variance_plan(args: argparse.Namespace) -> int:
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
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_variance_csv(csv_path, report)
    _plot_variance(png_path, report)
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


def _run_control_mean(args: argparse.Namespace) -> int:
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
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_control_mean_csv(csv_path, report)
    _plot_control_mean(png_path, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "blockers": report["blockers"],
                "json": _repo_relative(json_path),
            },
            indent=2,
        )
    )
    return 0 if bool(report["passed"]) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
