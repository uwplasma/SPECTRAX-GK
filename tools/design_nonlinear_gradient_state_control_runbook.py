#!/usr/bin/env python3
"""Build a fail-closed runbook for VMEC-state nonlinear-gradient controls."""

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
    NonlinearGradientStateControlRunbookConfig,
)
from spectraxgk.validation.nonlinear_gradient.followup_state_runbook import (  # noqa: E402
    nonlinear_gradient_state_control_runbook_report,
)

DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "nonlinear_gradient_state_control_runbook"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_csv(path: Path, report: dict[str, Any]) -> None:
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
                    "short_bracket_command_fragment": row.get("short_bracket_command_fragment"),
                }
            )


def _plot(path: Path, report: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    controls = list(report["controls"])
    labels = [str(row.get("state_parameter") or index) for index, row in enumerate(controls)]
    ready = np.asarray([1.0 if row.get("mapping_ready") else 0.0 for row in controls])
    condition = np.asarray([
        float(row.get("condition_number")) if row.get("condition_number") is not None else np.nan
        for row in controls
    ])
    residual = np.asarray([
        float(row.get("relative_residual")) if row.get("relative_residual") is not None else np.nan
        for row in controls
    ])
    x = np.arange(len(controls))
    cfg = report["config"]

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    axes[0].bar(x, ready, color=np.where(ready > 0.5, "#4daf4a", "#bdbdbd"), edgecolor="0.25")
    for index, value in enumerate(ready):
        if value <= 0.5:
            axes[0].text(index, 0.05, "blocked", ha="center", va="bottom", fontsize=8.0, color="0.25")
    axes[0].set_ylim(0.0, 1.15)
    axes[0].set_title("Launch mapping status")
    axes[0].set_ylabel("ready")
    axes[0].set_xticks(x, labels, rotation=24, ha="right")

    condition_limit = float(cfg["max_mapping_condition_number"])
    condition_plot = np.where(np.isfinite(condition), condition, condition_limit * 10.0)
    axes[1].bar(x, condition_plot, color="#377eb8", edgecolor="0.25")
    for index, value in enumerate(condition):
        if not np.isfinite(value):
            axes[1].text(index, condition_limit * 10.0, "missing/inf", ha="center", va="bottom", fontsize=8.0)
    axes[1].axhline(condition_limit, color="0.25", ls="--", lw=1.1)
    axes[1].set_yscale("log")
    axes[1].set_title("Mapping condition")
    axes[1].set_xticks(x, labels, rotation=24, ha="right")

    axes[2].bar(x, residual, color="#ff7f00", edgecolor="0.25")
    axes[2].axhline(float(cfg["max_mapping_relative_residual"]), color="0.25", ls="--", lw=1.1)
    axes[2].set_title("Mapping residual")
    axes[2].set_xticks(x, labels, rotation=24, ha="right")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("VMEC-state to input-control nonlinear-gradient runbook", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ql_seed_screen", type=Path)
    parser.add_argument("--mapping-artifact", action="append", type=Path, default=[])
    parser.add_argument("--case", default="qa_ess_nonlinear_gradient_state_control_runbook")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
