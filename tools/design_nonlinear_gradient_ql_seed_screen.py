#!/usr/bin/env python3
"""Screen QL/linear VMEC-state sensitivities before nonlinear-gradient campaigns."""

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
from spectraxgk.validation.nonlinear_gradient.followup_core import (  # noqa: E402
    NonlinearGradientQLSeedScreenConfig,
)
from spectraxgk.validation.nonlinear_gradient.followup_ql_seed import (  # noqa: E402
    nonlinear_gradient_ql_seed_screen_report,
)

DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "nonlinear_gradient_ql_seed_screen"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _label(payload: dict[str, Any], path: Path) -> str:
    case = payload.get("case_name")
    if isinstance(case, str) and case:
        return f"{case}:{path.stem}"
    return path.stem


def _write_csv(path: Path, report: dict[str, Any]) -> None:
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
                    "admitted_for_nonlinear_screen": row.get("admitted_for_nonlinear_screen"),
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

    from spectraxgk.artifacts.plotting import set_plot_style

    objective_rows = list(report["objective_rows"])
    controls = list(report["controls"])
    row_labels = [
        f"{row.get('case_name', idx)}\n{row.get('objective', '')}"
        for idx, row in enumerate(objective_rows)
    ]
    row_x = np.arange(len(objective_rows))
    sensitivities = np.asarray([
        _float_or_nan(row.get("metrics", {}).get("implicit_sensitivity"))
        for row in objective_rows
    ])
    row_pass = np.asarray([bool(row.get("accepted_objective_gate")) for row in objective_rows])
    control_labels = [str(row.get("state_parameter") or idx) for idx, row in enumerate(controls)]
    control_x = np.arange(len(controls))
    sign_fraction = np.asarray([_float_or_nan(row.get("sign_consistency_fraction")) for row in controls])
    n_cases = np.asarray([_float_or_nan(row.get("n_cases")) for row in controls])
    direction = np.asarray([_float_or_nan(row.get("descent_direction_sign")) for row in controls])
    admitted = np.asarray([bool(row.get("admitted_for_nonlinear_screen")) for row in controls])
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
    axes[1].bar(control_x - width / 2, sign_fraction, width=width, color="#377eb8", label="sign consistency")
    axes[1].bar(control_x + width / 2, n_cases, width=width, color="#ff7f00", label="case count")
    axes[1].axhline(float(cfg["min_sign_consistency"]), color="#377eb8", ls="--", lw=1.1)
    axes[1].axhline(float(cfg["min_cases_per_control"]), color="#ff7f00", ls="--", lw=1.1)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="+", type=Path)
    parser.add_argument("--case", default="qa_ess_nonlinear_gradient_ql_seed_screen")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--target-objective", action="append", dest="target_objectives", default=None)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_objectives = tuple(args.target_objectives or NonlinearGradientQLSeedScreenConfig().target_objectives)
    artifacts = [load_json_artifact(path) for path in args.artifact]
    report = nonlinear_gradient_ql_seed_screen_report(
        artifacts,
        paths=[_repo_relative(path) for path in args.artifact],
        labels=[_label(payload, path) for payload, path in zip(artifacts, args.artifact)],
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
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(csv_path, report)
    _plot(png_path, report)
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


if __name__ == "__main__":
    raise SystemExit(main())
