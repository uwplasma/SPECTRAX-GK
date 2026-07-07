#!/usr/bin/env python3
"""Design a composite VMEC-boundary control for nonlinear-gradient follow-up."""

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

from spectraxgk.validation.nonlinear_gradient.evidence import load_json_artifact  # noqa: E402
from spectraxgk.validation.nonlinear_gradient.followup_composite import (  # noqa: E402
    nonlinear_gradient_composite_control_report,
)
from spectraxgk.validation.nonlinear_gradient.followup_core import (  # noqa: E402
    NonlinearGradientCompositeControlConfig,
)

DEFAULT_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_composite_control_design"
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


def _write_csv(path: Path, report: dict[str, Any]) -> None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="+", type=Path)
    parser.add_argument("--case", default="qa_ess_nonlinear_gradient_composite_control")
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=1.00)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument("--min-same-sign-fraction", type=float, default=0.80)
    parser.add_argument("--min-controls", type=int, default=2)
    parser.add_argument("--default-relative-delta", type=float, default=0.02)
    parser.add_argument("--max-weight-abs", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = [load_json_artifact(path) for path in args.artifact]
    report = nonlinear_gradient_composite_control_report(
        artifacts,
        paths=[_repo_relative(path) for path in args.artifact],
        labels=[
            _label(payload, path) for payload, path in zip(artifacts, args.artifact)
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
