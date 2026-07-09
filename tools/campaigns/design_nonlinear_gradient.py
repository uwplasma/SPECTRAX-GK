#!/usr/bin/env python3
"""Design nonlinear-gradient campaign artifacts from admitted diagnostics.

This command groups the nonlinear-gradient design entry points that used to live
as separate one-off scripts. The report models remain in
``tools.campaigns.nonlinear_gradient_followup``; this file owns only CLI
argument parsing, CSV/JSON writing, and publication-figure rendering.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.diagnostics.nonlinear_gradient_evidence import load_json_artifact  # noqa: E402
from tools.campaigns.nonlinear_gradient_followup import (  # noqa: E402
    NonlinearGradientCandidateDesignConfig,
    NonlinearGradientCompositeControlConfig,
    NonlinearGradientQLSeedScreenConfig,
    NonlinearGradientStateControlRunbookConfig,
    nonlinear_gradient_candidate_design_report,
    nonlinear_gradient_composite_control_report,
    nonlinear_gradient_ql_seed_screen_report,
    nonlinear_gradient_state_control_runbook_report,
)

DEFAULT_NEXT_CAMPAIGN_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_next_campaign_design"
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _candidate_artifact_label(payload: dict[str, Any], path: Path) -> str:
    parameter = payload.get("parameter_name")
    if isinstance(parameter, str) and parameter:
        return f"{parameter}:{path.stem.removesuffix('_central_fd_gradient_gate')}"
    return path.stem


def _write_next_campaign_csv(path: Path, report: dict[str, Any]) -> None:
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
            metrics = (
                row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
            )
            variance = (
                row.get("variance_reduction", {})
                if isinstance(row.get("variance_reduction"), dict)
                else {}
            )
            writer.writerow(
                {
                    "rank": rank,
                    "label": row.get("label"),
                    "parameter_name": row.get("parameter_name"),
                    "action": row.get("action"),
                    "response_fraction": metrics.get("response_fraction"),
                    "fd_asymmetry_rel": metrics.get("fd_asymmetry_rel"),
                    "gradient_uncertainty_rel": metrics.get("gradient_uncertainty_rel"),
                    "uncertainty_required_bracket_scale": row.get(
                        "uncertainty_required_bracket_scale"
                    ),
                    "locality_safe_bracket_scale_limit": row.get(
                        "locality_safe_bracket_scale_limit"
                    ),
                    "bracket_only_feasible": row.get("bracket_only_feasible"),
                    "current_replicates_per_state": row.get(
                        "current_replicates_per_state"
                    ),
                    "variance_limiting_state": variance.get("limiting_state"),
                    "max_mean_rel_spread": variance.get("max_mean_rel_spread"),
                    "failed_spread_states": ";".join(
                        variance.get("failed_spread_states", [])
                    )
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


def _plot_next_campaign(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    rows = list(report["candidates"])
    labels = [
        str(row.get("parameter_name") or row.get("label") or idx)
        for idx, row in enumerate(rows)
    ]
    x = np.arange(len(rows))
    uncertainty = np.asarray(
        [
            _float_or_nan(row.get("metrics", {}).get("gradient_uncertainty_rel"))
            for row in rows
        ]
    )
    asymmetry = np.asarray(
        [_float_or_nan(row.get("metrics", {}).get("fd_asymmetry_rel")) for row in rows]
    )
    needed_scale = np.asarray(
        [_float_or_nan(row.get("uncertainty_required_bracket_scale")) for row in rows]
    )
    locality_scale = np.asarray(
        [_float_or_nan(row.get("locality_safe_bracket_scale_limit")) for row in rows]
    )
    extra = np.asarray(
        [
            _float_or_nan(row.get("estimated_extra_replicates_at_locality_limit"))
            for row in rows
        ]
    )
    cfg = report["config"]

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    axes[0].bar(x - 0.18, uncertainty, width=0.36, label="uncertainty", color="#386cb0")
    axes[0].bar(x + 0.18, asymmetry, width=0.36, label="asymmetry", color="#fdb462")
    axes[0].axhline(
        float(cfg["max_gradient_uncertainty_rel"]), color="#386cb0", ls="--", lw=1.2
    )
    axes[0].axhline(
        float(cfg["max_fd_asymmetry_rel"]), color="#b55d00", ls="--", lw=1.2
    )
    axes[0].set_title("Current FD gate margins")
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].set_ylabel("relative metric")
    axes[0].legend(frameon=False)

    width = 0.36
    axes[1].bar(
        x - width / 2,
        needed_scale,
        width=width,
        color="#7fc97f",
        label="needed for uncertainty",
    )
    axes[1].bar(
        x + width / 2,
        locality_scale,
        width=width,
        color="#beaed4",
        label="locality-safe limit",
    )
    axes[1].axhline(1.0, color="0.3", lw=0.9)
    axes[1].set_title("Bracket feasibility")
    axes[1].set_xticks(x, labels, rotation=22, ha="right")
    axes[1].set_ylabel("bracket scale")
    axes[1].legend(frameon=False)

    axes[2].bar(x, extra, color="#fb8072", edgecolor="0.25", linewidth=0.6)
    axes[2].axhline(
        float(cfg["max_extra_replicates_per_state"]), color="0.25", ls="--", lw=1.2
    )
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


def build_next_campaign_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="+", type=Path)
    parser.add_argument(
        "--case", default="qa_ess_nonlinear_gradient_next_campaign_design"
    )
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_NEXT_CAMPAIGN_OUT_PREFIX)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument("--sem-safety-factor", type=float, default=1.10)
    parser.add_argument("--max-extra-replicates-per-state", type=int, default=4)
    parser.add_argument("--max-checked-bracket-scale", type=float, default=1.50)
    parser.add_argument("--locality-safety-factor", type=float, default=0.95)
    return parser


def main_next_campaign(argv: list[str] | None = None) -> int:
    args = build_next_campaign_parser().parse_args(argv)
    artifacts = [load_json_artifact(path) for path in args.artifact]
    report = nonlinear_gradient_candidate_design_report(
        artifacts,
        paths=[_repo_relative(path) for path in args.artifact],
        labels=[
            _candidate_artifact_label(payload, path) for payload, path in zip(artifacts, args.artifact)
        ],
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
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_next_campaign_csv(csv_path, report)
    _plot_next_campaign(png_path, report)
    print(
        json.dumps(
            {
                "json": _repo_relative(json_path),
                "passed": report["passed"],
                "next_action": report["next_action"],
            },
            indent=2,
        )
    )
    return 0


DEFAULT_COMPOSITE_CONTROL_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_composite_control_design"
)


def _write_composite_control_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "parameter_name",
        "coefficient",
        "admissible_for_composite_direction",
        "blockers",
        "central_gradient",
        "descent_gradient",
        "response_fraction",
        "fd_asymmetry_rel",
        "gradient_uncertainty_rel",
        "same_sign_fraction",
        "control_weight",
        "control_argument",
    ]
    weights = {
        row["source_label"]: row
        for row in report.get("controls", [])
        if isinstance(row, dict)
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["candidates"]:
            metrics = (
                row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
            )
            control = weights.get(row.get("label"), {})
            writer.writerow(
                {
                    "label": row.get("label"),
                    "parameter_name": row.get("parameter_name"),
                    "coefficient": row.get("coefficient"),
                    "admissible_for_composite_direction": row.get(
                        "admissible_for_composite_direction"
                    ),
                    "blockers": ";".join(row.get("blockers", [])),
                    "central_gradient": metrics.get("central_gradient"),
                    "descent_gradient": metrics.get("descent_gradient"),
                    "response_fraction": metrics.get("response_fraction"),
                    "fd_asymmetry_rel": metrics.get("fd_asymmetry_rel"),
                    "gradient_uncertainty_rel": metrics.get("gradient_uncertainty_rel"),
                    "same_sign_fraction": metrics.get("same_sign_fraction"),
                    "control_weight": control.get("weight"),
                    "control_argument": control.get("control_argument"),
                }
            )


def _plot_composite_control(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    rows = list(report["candidates"])
    labels = [str(row.get("parameter_name") or idx) for idx, row in enumerate(rows)]
    x = np.arange(len(rows))
    gradients = np.asarray(
        [_float_or_nan(row.get("metrics", {}).get("central_gradient")) for row in rows]
    )
    descent = np.asarray(
        [_float_or_nan(row.get("metrics", {}).get("descent_gradient")) for row in rows]
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
    weights_by_label = {
        row["source_label"]: row
        for row in report.get("controls", [])
        if isinstance(row, dict)
    }
    weights = np.asarray(
        [
            _float_or_nan(weights_by_label.get(row.get("label"), {}).get("weight"))
            for row in rows
        ]
    )
    admissible = np.asarray(
        [bool(row.get("admissible_for_composite_direction")) for row in rows]
    )
    cfg = report["config"]

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    colors = np.where(admissible, "#4daf4a", "#bdbdbd")
    axes[0].bar(
        x - 0.18, gradients, width=0.36, color="#377eb8", label="central gradient"
    )
    axes[0].bar(x + 0.18, descent, width=0.36, color="#e41a1c", label="descent sign")
    axes[0].axhline(0.0, color="0.25", lw=0.9)
    axes[0].set_title("Measured long-window gradient")
    axes[0].set_xticks(x, labels, rotation=24, ha="right")
    axes[0].set_ylabel("dQ / dcontrol")
    axes[0].legend(frameon=False)

    axes[1].bar(x - 0.18, asymmetry, width=0.36, color="#ff7f00", label="asymmetry")
    axes[1].bar(x + 0.18, uncertainty, width=0.36, color="#984ea3", label="uncertainty")
    axes[1].axhline(
        float(cfg["max_fd_asymmetry_rel"]), color="#ff7f00", ls="--", lw=1.2
    )
    axes[1].axhline(
        float(cfg["max_gradient_uncertainty_rel"]), color="#984ea3", ls="--", lw=1.2
    )
    axes[1].set_title("Composite admission gates")
    axes[1].set_xticks(x, labels, rotation=24, ha="right")
    axes[1].set_ylabel("relative gate metric")
    axes[1].legend(frameon=False)

    axes[2].bar(x, weights, color=colors, edgecolor="0.25", linewidth=0.6)
    axes[2].axhline(0.0, color="0.25", lw=0.9)
    axes[2].set_title("Recommended normalized controls")
    axes[2].set_xticks(x, labels, rotation=24, ha="right")
    axes[2].set_ylabel("VMEC input weight")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Nonlinear-gradient composite control design", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_composite_control_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="+", type=Path)
    parser.add_argument("--case", default="qa_ess_nonlinear_gradient_composite_control")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_COMPOSITE_CONTROL_OUT_PREFIX)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=1.00)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument("--min-same-sign-fraction", type=float, default=0.80)
    parser.add_argument("--min-controls", type=int, default=2)
    parser.add_argument("--default-relative-delta", type=float, default=0.02)
    parser.add_argument("--max-weight-abs", type=float, default=1.0)
    return parser


def main_composite_control(argv: list[str] | None = None) -> int:
    args = build_composite_control_parser().parse_args(argv)
    artifacts = [load_json_artifact(path) for path in args.artifact]
    report = nonlinear_gradient_composite_control_report(
        artifacts,
        paths=[_repo_relative(path) for path in args.artifact],
        labels=[
            _candidate_artifact_label(payload, path) for payload, path in zip(artifacts, args.artifact)
        ],
        case=args.case,
        config=NonlinearGradientCompositeControlConfig(
            max_gradient_uncertainty_rel=args.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=args.max_fd_asymmetry_rel,
            min_fd_response_fraction=args.min_fd_response_fraction,
            min_same_sign_fraction=args.min_same_sign_fraction,
            min_controls=args.min_controls,
            default_relative_delta=args.default_relative_delta,
            max_weight_abs=args.max_weight_abs,
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
    _write_composite_control_csv(csv_path, report)
    _plot_composite_control(png_path, report)
    print(
        json.dumps(
            {
                "json": _repo_relative(json_path),
                "passed": report["passed"],
                "next_action": report["next_action"],
            },
            indent=2,
        )
    )
    return 0


DEFAULT_QL_SEED_SCREEN_OUT_PREFIX = ROOT / "docs" / "_static" / "nonlinear_gradient_ql_seed_screen"


def _ql_seed_label(payload: dict[str, Any], path: Path) -> str:
    case = payload.get("case_name")
    if isinstance(case, str) and case:
        return f"{case}:{path.stem}"
    return path.stem


def _write_ql_seed_screen_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "state_parameter",
        "state_control_family",
        "admitted_for_nonlinear_screen",
        "blockers",
        "primary_objective",
        "n_accepted_rows",
        "n_cases",
        "dominant_sensitivity_sign",
        "descent_direction_sign",
        "sign_consistency_fraction",
        "mean_abs_sensitivity",
        "state_control_argument",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in report["controls"]:
            writer.writerow(
                {
                    "state_parameter": row.get("state_parameter"),
                    "state_control_family": row.get("state_control_family"),
                    "admitted_for_nonlinear_screen": row.get(
                        "admitted_for_nonlinear_screen"
                    ),
                    "blockers": ";".join(row.get("blockers", [])),
                    "primary_objective": row.get("primary_objective"),
                    "n_accepted_rows": row.get("n_accepted_rows"),
                    "n_cases": row.get("n_cases"),
                    "dominant_sensitivity_sign": row.get("dominant_sensitivity_sign"),
                    "descent_direction_sign": row.get("descent_direction_sign"),
                    "sign_consistency_fraction": row.get("sign_consistency_fraction"),
                    "mean_abs_sensitivity": row.get("mean_abs_sensitivity"),
                    "state_control_argument": row.get("state_control_argument"),
                }
            )


def _plot_ql_seed_screen(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    objective_rows = list(report["objective_rows"])
    controls = list(report["controls"])
    row_labels = [
        f"{row.get('case_name', idx)}\n{row.get('objective', '')}"
        for idx, row in enumerate(objective_rows)
    ]
    row_x = np.arange(len(objective_rows))
    sensitivities = np.asarray(
        [
            _float_or_nan(row.get("metrics", {}).get("implicit_sensitivity"))
            for row in objective_rows
        ]
    )
    row_pass = np.asarray(
        [bool(row.get("accepted_objective_gate")) for row in objective_rows]
    )
    control_labels = [
        str(row.get("state_parameter") or idx) for idx, row in enumerate(controls)
    ]
    control_x = np.arange(len(controls))
    sign_fraction = np.asarray(
        [_float_or_nan(row.get("sign_consistency_fraction")) for row in controls]
    )
    n_cases = np.asarray([_float_or_nan(row.get("n_cases")) for row in controls])
    direction = np.asarray(
        [_float_or_nan(row.get("descent_direction_sign")) for row in controls]
    )
    admitted = np.asarray(
        [bool(row.get("admitted_for_nonlinear_screen")) for row in controls]
    )
    cfg = report["config"]

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    colors = np.where(row_pass, "#4daf4a", "#bdbdbd")
    axes[0].bar(row_x, sensitivities, color=colors, edgecolor="0.25", linewidth=0.5)
    axes[0].axhline(0.0, color="0.25", lw=0.9)
    axes[0].set_title("QL/linear objective sensitivities")
    axes[0].set_ylabel("implicit sensitivity")
    axes[0].set_xticks(row_x, row_labels, rotation=28, ha="right")

    width = 0.36
    axes[1].bar(
        control_x - width / 2,
        sign_fraction,
        width=width,
        color="#377eb8",
        label="sign consistency",
    )
    axes[1].bar(
        control_x + width / 2, n_cases, width=width, color="#ff7f00", label="case count"
    )
    axes[1].axhline(
        float(cfg["min_sign_consistency"]), color="#377eb8", ls="--", lw=1.1
    )
    axes[1].axhline(
        float(cfg["min_cases_per_control"]), color="#ff7f00", ls="--", lw=1.1
    )
    axes[1].set_title("Cross-artifact admission")
    axes[1].set_xticks(control_x, control_labels, rotation=24, ha="right")
    axes[1].legend(frameon=False)

    colors = np.where(admitted, "#4daf4a", "#bdbdbd")
    axes[2].bar(control_x, direction, color=colors, edgecolor="0.25", linewidth=0.5)
    axes[2].axhline(0.0, color="0.25", lw=0.9)
    axes[2].set_title("Recommended descent signs")
    axes[2].set_xticks(control_x, control_labels, rotation=24, ha="right")
    axes[2].set_ylabel("state-control sign")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("QL-seeded nonlinear-gradient control screen", fontsize=15)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_ql_seed_screen_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="+", type=Path)
    parser.add_argument("--case", default="qa_ess_nonlinear_gradient_ql_seed_screen")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_QL_SEED_SCREEN_OUT_PREFIX)
    parser.add_argument(
        "--target-objective", action="append", dest="target_objectives", default=None
    )
    parser.add_argument("--primary-objective", default="mixing_length_heat_flux_proxy")
    parser.add_argument("--min-distinct-controls", type=int, default=2)
    parser.add_argument("--min-cases-per-control", type=int, default=2)
    parser.add_argument("--min-sign-consistency", type=float, default=0.75)
    parser.add_argument("--max-objective-rel-error", type=float, default=0.02)
    parser.add_argument("--min-abs-sensitivity", type=float, default=1.0e-12)
    artifact_group = parser.add_mutually_exclusive_group()
    artifact_group.add_argument(
        "--require-artifact-passed",
        dest="require_artifact_passed",
        action="store_true",
        default=NonlinearGradientQLSeedScreenConfig().require_artifact_passed,
        help="Require the whole source artifact to pass, not only selected objective rows.",
    )
    artifact_group.add_argument(
        "--allow-failed-artifacts",
        dest="require_artifact_passed",
        action="store_false",
        help="Allow selected objective rows from artifacts that failed unrelated gates.",
    )
    return parser


def main_ql_seed_screen(argv: list[str] | None = None) -> int:
    args = build_ql_seed_screen_parser().parse_args(argv)
    target_objectives = tuple(
        args.target_objectives
        or NonlinearGradientQLSeedScreenConfig().target_objectives
    )
    artifacts = [load_json_artifact(path) for path in args.artifact]
    report = nonlinear_gradient_ql_seed_screen_report(
        artifacts,
        paths=[_repo_relative(path) for path in args.artifact],
        labels=[
            _ql_seed_label(payload, path) for payload, path in zip(artifacts, args.artifact)
        ],
        case=args.case,
        config=NonlinearGradientQLSeedScreenConfig(
            target_objectives=target_objectives,
            primary_objective=args.primary_objective,
            min_distinct_controls=args.min_distinct_controls,
            min_cases_per_control=args.min_cases_per_control,
            min_sign_consistency=args.min_sign_consistency,
            max_objective_rel_error=args.max_objective_rel_error,
            min_abs_sensitivity=args.min_abs_sensitivity,
            require_artifact_passed=bool(args.require_artifact_passed),
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
    _write_ql_seed_screen_csv(csv_path, report)
    _plot_ql_seed_screen(png_path, report)
    print(
        json.dumps(
            {
                "json": _repo_relative(json_path),
                "passed": report["passed"],
                "next_action": report["next_action"],
            },
            indent=2,
        )
    )
    return 0


DEFAULT_STATE_CONTROL_RUNBOOK_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_state_control_runbook"
)


def _write_state_control_runbook_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "state_parameter",
        "mapping_ready",
        "blockers",
        "input_control_argument",
        "condition_number",
        "relative_residual",
        "short_bracket_command_fragment",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in report["controls"]:
            writer.writerow(
                {
                    "state_parameter": row.get("state_parameter"),
                    "mapping_ready": row.get("mapping_ready"),
                    "blockers": ";".join(row.get("blockers", [])),
                    "input_control_argument": row.get("input_control_argument"),
                    "condition_number": row.get("condition_number"),
                    "relative_residual": row.get("relative_residual"),
                    "short_bracket_command_fragment": row.get(
                        "short_bracket_command_fragment"
                    ),
                }
            )


def _plot_state_control_runbook(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    controls = list(report["controls"])
    labels = [
        str(row.get("state_parameter") or index) for index, row in enumerate(controls)
    ]
    ready = np.asarray([1.0 if row.get("mapping_ready") else 0.0 for row in controls])
    condition = np.asarray(
        [
            float(row.get("condition_number"))
            if row.get("condition_number") is not None
            else np.nan
            for row in controls
        ]
    )
    residual = np.asarray(
        [
            float(row.get("relative_residual"))
            if row.get("relative_residual") is not None
            else np.nan
            for row in controls
        ]
    )
    x = np.arange(len(controls))
    cfg = report["config"]

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    axes[0].bar(
        x, ready, color=np.where(ready > 0.5, "#4daf4a", "#bdbdbd"), edgecolor="0.25"
    )
    for index, value in enumerate(ready):
        if value <= 0.5:
            axes[0].text(
                index,
                0.05,
                "blocked",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="0.25",
            )
    axes[0].set_ylim(0.0, 1.15)
    axes[0].set_title("Launch mapping status")
    axes[0].set_ylabel("ready")
    axes[0].set_xticks(x, labels, rotation=24, ha="right")

    condition_limit = float(cfg["max_mapping_condition_number"])
    condition_plot_state_control_runbook = np.where(np.isfinite(condition), condition, condition_limit * 10.0)
    axes[1].bar(x, condition_plot_state_control_runbook, color="#377eb8", edgecolor="0.25")
    for index, value in enumerate(condition):
        if not np.isfinite(value):
            axes[1].text(
                index,
                condition_limit * 10.0,
                "missing/inf",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )
    axes[1].axhline(condition_limit, color="0.25", ls="--", lw=1.1)
    axes[1].set_yscale("log")
    axes[1].set_title("Mapping condition")
    axes[1].set_xticks(x, labels, rotation=24, ha="right")

    axes[2].bar(x, residual, color="#ff7f00", edgecolor="0.25")
    axes[2].axhline(
        float(cfg["max_mapping_relative_residual"]), color="0.25", ls="--", lw=1.1
    )
    axes[2].set_title("Mapping residual")
    axes[2].set_xticks(x, labels, rotation=24, ha="right")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("VMEC-state to input-control nonlinear-gradient runbook", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_state_control_runbook_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ql_seed_screen", type=Path)
    parser.add_argument("--mapping-artifact", action="append", type=Path, default=[])
    parser.add_argument(
        "--case", default="qa_ess_nonlinear_gradient_state_control_runbook"
    )
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_STATE_CONTROL_RUNBOOK_OUT_PREFIX)
    parser.add_argument("--min-mapped-controls", type=int, default=2)
    parser.add_argument("--max-mapping-condition-number", type=float, default=1.0e6)
    parser.add_argument("--max-mapping-relative-residual", type=float, default=0.10)
    parser.add_argument("--default-relative-delta", type=float, default=0.02)
    parser.add_argument(
        "--allow-unpassed-mapping-artifacts",
        action="store_true",
        help="Use only numerical mapping thresholds; otherwise mapping rows must also set passed=true.",
    )
    return parser


def main_state_control_runbook(argv: list[str] | None = None) -> int:
    args = build_state_control_runbook_parser().parse_args(argv)
    ql_seed_screen = load_json_artifact(args.ql_seed_screen)
    mappings = [load_json_artifact(path) for path in args.mapping_artifact]
    report = nonlinear_gradient_state_control_runbook_report(
        ql_seed_screen,
        mapping_artifacts=mappings,
        case=args.case,
        config=NonlinearGradientStateControlRunbookConfig(
            min_mapped_controls=int(args.min_mapped_controls),
            max_mapping_condition_number=float(args.max_mapping_condition_number),
            max_mapping_relative_residual=float(args.max_mapping_relative_residual),
            default_relative_delta=float(args.default_relative_delta),
            require_mapping_passed=not bool(args.allow_unpassed_mapping_artifacts),
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
    _write_state_control_runbook_csv(csv_path, report)
    _plot_state_control_runbook(png_path, report)
    print(
        json.dumps(
            {
                "json": _repo_relative(json_path),
                "passed": report["passed"],
                "next_action": report["next_action"],
            },
            indent=2,
        )
    )
    return 0

SUBCOMMANDS: dict[str, Callable[[list[str] | None], int]] = {
    "next-campaign": main_next_campaign,
    "composite-control": main_composite_control,
    "ql-seed-screen": main_ql_seed_screen,
    "state-control-runbook": main_state_control_runbook,
}


def build_dispatch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=sorted(SUBCOMMANDS),
        help="Nonlinear-gradient design artifact to build.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help"}:
        build_dispatch_parser().parse_args(tokens)
        return 0
    command, rest = tokens[0], tokens[1:]
    try:
        handler = SUBCOMMANDS[command]
    except KeyError:
        build_dispatch_parser().parse_args([command])
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
