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

from spectraxgk.diagnostics.nonlinear_gradient_evidence import (  # noqa: E402
    NonlinearTurbulenceGradientBracketSweepConfig,
    NonlinearTurbulenceGradientCandidateRankingConfig,
    load_json_artifact,
    nonlinear_turbulence_gradient_bracket_sweep_report,
    nonlinear_turbulence_gradient_candidate_ranking_report,
)
from tools.campaigns.nonlinear_gradient_followup import (  # noqa: E402
    NonlinearGradientCandidateDesignConfig,
    NonlinearGradientCompositeControlConfig,
    NonlinearGradientControlVariateCampaignConfig,
    NonlinearGradientFollowupConfig,
    NonlinearGradientQLSeedScreenConfig,
    NonlinearGradientStateControlRunbookConfig,
    nonlinear_gradient_candidate_design_report,
    nonlinear_gradient_composite_control_report,
    nonlinear_gradient_control_variate_campaign_plan,
    nonlinear_gradient_followup_plan,
    nonlinear_gradient_ql_seed_screen_report,
    nonlinear_gradient_state_control_runbook_report,
)
from tools.campaigns.write_nonlinear_turbulence_gradient_campaign import (  # noqa: E402
    DEFAULT_DT_VARIANT,
    DEFAULT_GRID,
    DEFAULT_HORIZONS,
    DEFAULT_SEEDS,
    DEFAULT_WINDOW,
    PYTHON_CMD,
)
from tools.campaigns.write_vmec_boundary_perturbation_inputs import (  # noqa: E402
    CoefficientSpec,
    _json_clean as _vmec_json_clean,
    _parse_coefficient_spec,
    write_perturbation_inputs,
)

DEFAULT_NEXT_CAMPAIGN_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_next_campaign_design"
)


def _repo_relative(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _candidate_artifact_label(payload: dict[str, Any], path: Path) -> str:
    parameter = payload.get("parameter_name")
    if isinstance(parameter, str) and parameter:
        return f"{parameter}:{path.stem.removesuffix('_central_fd_gradient_gate')}"
    return path.stem


def _resolve_artifact_path(raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path if path.exists() else None
    for candidate in (ROOT / path, path):
        if candidate.exists():
            return candidate
    return None


def _hydrate_source_ensembles(payload: dict[str, Any]) -> dict[str, Any]:
    """Load compact source-ensemble rows from tracked ensemble artifacts."""

    source_ensembles = payload.get("source_ensembles")
    if not isinstance(source_ensembles, dict):
        return payload
    hydrated: dict[str, Any] = {}
    changed = False
    for state, raw in source_ensembles.items():
        if not isinstance(raw, dict):
            hydrated[state] = raw
            continue
        row = dict(raw)
        if isinstance(row.get("rows"), list):
            hydrated[state] = row
            continue
        ensemble_path = _resolve_artifact_path(row.get("path"))
        if ensemble_path is not None:
            ensemble = load_json_artifact(ensemble_path)
            if isinstance(ensemble.get("rows"), list):
                row["rows"] = ensemble["rows"]
                changed = True
        hydrated[state] = row
    if not changed:
        return payload
    return {**payload, "source_ensembles": hydrated}


def build_followup_plan_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan bounded nonlinear-gradient follow-up runs."
    )
    parser.add_argument(
        "artifact",
        nargs="+",
        type=Path,
        help="Production central-FD gradient JSON artifacts to inspect.",
    )
    parser.add_argument("--case", default="nonlinear_turbulence_gradient_followup")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument(
        "--sem-safety-factor",
        type=float,
        default=1.10,
        help="Safety factor applied to the ideal 1/sqrt(N) replica estimate.",
    )
    parser.add_argument("--max-extra-replicates-per-state", type=int, default=4)
    parser.add_argument("--default-nominal-timestep", type=float, default=0.05)
    return parser


def main_followup_plan(argv: list[str] | None = None) -> int:
    args = build_followup_plan_parser().parse_args(argv)
    artifacts = [
        _hydrate_source_ensembles(load_json_artifact(path)) for path in args.artifact
    ]
    report = nonlinear_gradient_followup_plan(
        artifacts,
        paths=[_repo_relative(path) for path in args.artifact],
        labels=[
            _candidate_artifact_label(payload, path)
            for payload, path in zip(artifacts, args.artifact)
        ],
        case=args.case,
        config=NonlinearGradientFollowupConfig(
            max_gradient_uncertainty_rel=args.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=args.max_fd_asymmetry_rel,
            min_fd_response_fraction=args.min_fd_response_fraction,
            sem_safety_factor=args.sem_safety_factor,
            max_extra_replicates_per_state=args.max_extra_replicates_per_state,
            default_nominal_timestep=args.default_nominal_timestep,
        ),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {args.json_out}")
    else:
        print(text)
    return 0


def build_rank_candidates_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank failed nonlinear turbulence-gradient candidates."
    )
    parser.add_argument(
        "artifact",
        nargs="+",
        type=Path,
        help="Central finite-difference candidate JSON artifacts to rank.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-condition-number", type=float, default=1.0e8)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument(
        "--campaign-context",
        choices=("single_control_screen", "overdetermined_followup"),
        default="single_control_screen",
        help=(
            "Recommendation context. Use overdetermined_followup when the input "
            "candidates are the result of a completed multi-control follow-up."
        ),
    )
    parser.add_argument(
        "--fail-on-no-promotable",
        action="store_true",
        help="Return nonzero unless at least one candidate already passes all production gates.",
    )
    return parser


def main_rank_candidates(argv: list[str] | None = None) -> int:
    args = build_rank_candidates_parser().parse_args(argv)
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
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {args.json_out}")
    else:
        print(text)
    if args.fail_on_no_promotable and not bool(report.get("passed", False)):
        print(
            "no nonlinear turbulence-gradient candidate passes production gates",
            file=sys.stderr,
        )
        return 1
    return 0


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


DEFAULT_BRACKET_SWEEP_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_bracket_sweep"
)


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


def build_bracket_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize a same-control nonlinear-gradient bracket sweep."
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
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Write JSON/CSV/PNG only. Useful for tracked documentation previews.",
    )
    parser.add_argument("--fail-on-no-promotable", action="store_true")
    return parser


def main_bracket_sweep(argv: list[str] | None = None) -> int:
    args = build_bracket_sweep_parser().parse_args(argv)
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
    return 1 if bool(args.fail_on_no_promotable) and not bool(report["passed"]) else 0


DEFAULT_CONTROL_VARIATE_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_control_variate_campaign_plan"
)


def _write_control_variate_csv(path: Path, report: dict[str, Any]) -> None:
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


def _plot_control_variate(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def build_control_variate_campaign_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write an independent control-mean campaign plan."
    )
    parser.add_argument("variance_report", type=Path)
    parser.add_argument(
        "--case", default="nonlinear_gradient_control_variate_campaign_plan"
    )
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_CONTROL_VARIATE_OUT_PREFIX
    )
    parser.add_argument("--target-response-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--sem-safety-factor", type=float, default=1.10)
    parser.add_argument("--min-control-mean-pairs", type=int, default=4)
    parser.add_argument("--max-control-mean-pairs", type=int, default=32)
    parser.add_argument("--first-new-seed", type=int, default=34)
    return parser


def main_control_variate_campaign(argv: list[str] | None = None) -> int:
    args = build_control_variate_campaign_parser().parse_args(argv)
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
    _write_control_variate_csv(csv_path, report)
    _plot_control_variate(png_path, report)
    print(
        json.dumps(
            {"action": report["action"], "json": _repo_relative(json_path)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_NEXT_CAMPAIGN_OUT_PREFIX
    )
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
            _candidate_artifact_label(payload, path)
            for payload, path in zip(artifacts, args.artifact)
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
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_COMPOSITE_CONTROL_OUT_PREFIX
    )
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
            _candidate_artifact_label(payload, path)
            for payload, path in zip(artifacts, args.artifact)
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


DEFAULT_QL_SEED_SCREEN_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_ql_seed_screen"
)


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
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_QL_SEED_SCREEN_OUT_PREFIX
    )
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
            _ql_seed_label(payload, path)
            for payload, path in zip(artifacts, args.artifact)
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
    condition_plot_state_control_runbook = np.where(
        np.isfinite(condition), condition, condition_limit * 10.0
    )
    axes[1].bar(
        x, condition_plot_state_control_runbook, color="#377eb8", edgecolor="0.25"
    )
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
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_STATE_CONTROL_RUNBOOK_OUT_PREFIX
    )
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


DEFAULT_OVERDETERMINED_OUT_DIR = (
    ROOT / "tools_out" / "overdetermined_nonlinear_gradient_campaign"
)


def _coefficient_slug(spec: CoefficientSpec) -> str:
    return str(spec.slug).replace("__", "_")


def _fd_artifact(case: str, coefficient: CoefficientSpec) -> Path:
    slug = _coefficient_slug(coefficient)
    return (
        ROOT
        / "docs"
        / "_static"
        / f"{case}_{slug}_nonlinear_gradient_{slug}_central_fd_gradient_gate.json"
    )


def _nonlinear_campaign_command(
    *,
    case: str,
    coefficient: CoefficientSpec,
    vmec_manifest: dict[str, Any],
    campaign_out_dir: Path,
    horizons: str,
    grid: str,
    window_tmin: float,
    window_tmax: float,
    ky: float,
    dt: float,
    dt_variant: float,
    baseline_seed: int,
    seed_variants: tuple[int, ...],
    nl: int,
    nm: int,
) -> str:
    slug = _coefficient_slug(coefficient)
    state_wouts = vmec_manifest["expected_wout_files"]
    command = (
        f"{PYTHON_CMD} tools/campaigns/write_nonlinear_turbulence_gradient_campaign.py "
        f"--baseline-vmec-file {state_wouts['baseline']} "
        f"--plus-vmec-file {state_wouts['plus_delta']} "
        f"--minus-vmec-file {state_wouts['minus_delta']} "
        f"--case {case}_{slug}_nonlinear_gradient "
        f"--parameter-name {slug} "
        f"--delta-parameter {float(vmec_manifest['delta_parameter']):.16e} "
        f"--out-dir {_repo_relative(campaign_out_dir / slug)} "
        f"--horizons {horizons} "
        f"--grid {grid} "
        f"--window-tmin {float(window_tmin):.12g} "
        f"--window-tmax {float(window_tmax):.12g} "
        f"--ky {float(ky):.16g} "
        f"--dt {float(dt):.12g} "
        f"--dt-variant {float(dt_variant):.12g} "
        f"--baseline-seed {int(baseline_seed)} "
        f"--Nl {int(nl)} --Nm {int(nm)}"
    )
    for seed in seed_variants:
        command += f" --seed-variant {int(seed)}"
    return command


def _expected_nonlinear_campaign_manifest(
    *,
    campaign_out_dir: Path,
    coefficient: CoefficientSpec,
) -> Path:
    return (
        campaign_out_dir
        / _coefficient_slug(coefficient)
        / "gradient_campaign_manifest.json"
    )


def _load_previous_ranking(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def write_overdetermined_campaign(
    *,
    baseline_input: Path,
    out_dir: Path,
    case: str,
    coefficients: tuple[CoefficientSpec, ...],
    relative_delta: float,
    vmec_command: str = "vmec_jax",
    campaign_out_dir: Path | None = None,
    previous_ranking: Path | None = None,
    horizons: str = DEFAULT_HORIZONS,
    grid: str = DEFAULT_GRID,
    window_tmin: float = DEFAULT_WINDOW[0],
    window_tmax: float = DEFAULT_WINDOW[1],
    ky: float = 0.47619047619047616,
    dt: float = 0.05,
    dt_variant: float = DEFAULT_DT_VARIANT,
    baseline_seed: int = 22,
    seed_variants: tuple[int, ...] = DEFAULT_SEEDS,
    nl: int = 4,
    nm: int = 8,
) -> dict[str, Any]:
    """Write a multi-control nonlinear turbulence-gradient campaign manifest."""

    if len(coefficients) < 2:
        raise ValueError(
            "overdetermined nonlinear-gradient campaigns require at least two controls"
        )
    if relative_delta <= 0.0:
        raise ValueError("relative_delta must be positive")
    if len({_coefficient_slug(spec) for spec in coefficients}) != len(coefficients):
        raise ValueError("coefficient list contains duplicate controls")

    out_dir.mkdir(parents=True, exist_ok=True)
    campaign_root = campaign_out_dir or (out_dir / "nonlinear_campaigns")
    previous = _load_previous_ranking(previous_ranking)
    controls: list[dict[str, Any]] = []
    fd_artifacts: list[Path] = []
    for coefficient in coefficients:
        slug = _coefficient_slug(coefficient)
        control_case = f"{case}_{slug}"
        control_out_dir = out_dir / slug
        vmec_manifest = write_perturbation_inputs(
            baseline_input=baseline_input,
            out_dir=control_out_dir,
            case=control_case,
            coefficient=coefficient,
            relative_delta=relative_delta,
            vmec_command=vmec_command,
        )
        fd_json = _fd_artifact(case, coefficient)
        fd_artifacts.append(fd_json)
        controls.append(
            {
                "coefficient": coefficient.label,
                "coefficient_slug": slug,
                "case": control_case,
                "vmec_manifest": _repo_relative(vmec_manifest["manifest"]),
                "delta_parameter": float(vmec_manifest["delta_parameter"]),
                "state_input_files": vmec_manifest["state_input_files"],
                "expected_wout_files": vmec_manifest["expected_wout_files"],
                "vmec_run_commands": vmec_manifest["vmec_run_commands"],
                "nonlinear_campaign_command_after_vmec_runs": _nonlinear_campaign_command(
                    case=case,
                    coefficient=coefficient,
                    vmec_manifest=vmec_manifest,
                    campaign_out_dir=campaign_root,
                    horizons=horizons,
                    grid=grid,
                    window_tmin=window_tmin,
                    window_tmax=window_tmax,
                    ky=ky,
                    dt=dt,
                    dt_variant=dt_variant,
                    baseline_seed=baseline_seed,
                    seed_variants=seed_variants,
                    nl=nl,
                    nm=nm,
                ),
                "expected_nonlinear_campaign_manifest": _repo_relative(
                    _expected_nonlinear_campaign_manifest(
                        campaign_out_dir=campaign_root,
                        coefficient=coefficient,
                    )
                ),
                "expected_fd_artifact": _repo_relative(fd_json),
            }
        )

    ranking_json = (
        ROOT
        / "docs"
        / "_static"
        / f"{case}_overdetermined_nonlinear_gradient_candidate_ranking.json"
    )
    ranking_command = (
        f"{PYTHON_CMD} tools/campaigns/design_nonlinear_gradient.py rank-candidates "
        + " ".join(_repo_relative(path) for path in fd_artifacts)
        + f" --json-out {_repo_relative(ranking_json)}"
        + " --campaign-context overdetermined_followup"
        + " --fail-on-no-promotable"
    )
    manifest = {
        "kind": "overdetermined_nonlinear_turbulence_gradient_campaign_manifest",
        "claim_level": "multi_control_profile_gradient_launch_plan_not_simulation_claim",
        "case": str(case),
        "baseline_input": baseline_input,
        "relative_delta": float(relative_delta),
        "control_count": len(controls),
        "controls": controls,
        "previous_ranking": None
        if previous is None
        else {
            "path": _repo_relative(previous_ranking)
            if previous_ranking is not None
            else None,
            "passed": bool(previous.get("passed", False)),
            "recommendation": str(previous.get("recommendation", "")),
            "best_candidate": previous.get("best_candidate"),
        },
        "run_contract": {
            "same_numerics_except_parameter": True,
            "overdetermined_controls": True,
            "horizons": horizons,
            "grid": grid,
            "analysis_window": [float(window_tmin), float(window_tmax)],
            "ky": float(ky),
            "dt": float(dt),
            "dt_variant": float(dt_variant),
            "replicates": [f"seed{seed}" for seed in seed_variants]
            + [f"dt{float(dt_variant):.12g}".replace(".", "p").replace("-", "m")],
        },
        "promotion_contract": {
            "claim_boundary": (
                "This manifest only launches an overdetermined campaign. "
                "Production nonlinear turbulence-gradient evidence still requires "
                "real re-equilibrated VMEC files, matched long post-transient nonlinear "
                "replicates, per-control central-FD gates, and a ranking/evidence report "
                "with at least one control passing all locality and uncertainty gates."
            ),
            "expected_fd_artifacts": [_repo_relative(path) for path in fd_artifacts],
            "candidate_ranking_json": _repo_relative(ranking_json),
            "candidate_ranking_command": ranking_command,
        },
    }
    manifest_path = out_dir / "overdetermined_nonlinear_gradient_campaign_manifest.json"
    manifest_path.write_text(
        json.dumps(_vmec_json_clean(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest"] = manifest_path
    return manifest


def build_overdetermined_campaign_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-input", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OVERDETERMINED_OUT_DIR)
    parser.add_argument("--case", required=True)
    parser.add_argument("--coefficient", action="append", required=True)
    parser.add_argument("--relative-delta", type=float, default=0.05)
    parser.add_argument("--vmec-command", default="vmec_jax")
    parser.add_argument("--campaign-out-dir", type=Path)
    parser.add_argument("--previous-ranking", type=Path)
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--grid", default=DEFAULT_GRID)
    parser.add_argument("--window-tmin", type=float, default=DEFAULT_WINDOW[0])
    parser.add_argument("--window-tmax", type=float, default=DEFAULT_WINDOW[1])
    parser.add_argument("--ky", type=float, default=0.47619047619047616)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--dt-variant", type=float, default=DEFAULT_DT_VARIANT)
    parser.add_argument("--baseline-seed", type=int, default=22)
    parser.add_argument("--seed-variant", action="append", type=int, default=None)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    return parser


def main_overdetermined_campaign(argv: list[str] | None = None) -> int:
    args = build_overdetermined_campaign_parser().parse_args(argv)
    manifest = write_overdetermined_campaign(
        baseline_input=Path(args.baseline_input),
        out_dir=Path(args.out_dir),
        case=str(args.case),
        coefficients=tuple(_parse_coefficient_spec(raw) for raw in args.coefficient),
        relative_delta=float(args.relative_delta),
        vmec_command=str(args.vmec_command),
        campaign_out_dir=args.campaign_out_dir,
        previous_ranking=args.previous_ranking,
        horizons=str(args.horizons),
        grid=str(args.grid),
        window_tmin=float(args.window_tmin),
        window_tmax=float(args.window_tmax),
        ky=float(args.ky),
        dt=float(args.dt),
        dt_variant=float(args.dt_variant),
        baseline_seed=int(args.baseline_seed),
        seed_variants=tuple(args.seed_variant or DEFAULT_SEEDS),
        nl=int(args.Nl),
        nm=int(args.Nm),
    )
    print(
        json.dumps(
            {
                "manifest": _repo_relative(manifest["manifest"]),
                "case": manifest["case"],
                "control_count": manifest["control_count"],
                "ranking_command": manifest["promotion_contract"][
                    "candidate_ranking_command"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


SUBCOMMANDS: dict[str, Callable[[list[str] | None], int]] = {
    "bracket-sweep": main_bracket_sweep,
    "control-variate-campaign": main_control_variate_campaign,
    "next-campaign": main_next_campaign,
    "composite-control": main_composite_control,
    "followup-plan": main_followup_plan,
    "overdetermined-campaign": main_overdetermined_campaign,
    "ql-seed-screen": main_ql_seed_screen,
    "rank-candidates": main_rank_candidates,
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
