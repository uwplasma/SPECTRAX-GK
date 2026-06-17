#!/usr/bin/env python3
"""Summarize a same-control nonlinear gradient perturbation-amplitude sweep."""

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

from spectraxgk.validation.nonlinear_gradient.evidence import (  # noqa: E402
    NonlinearTurbulenceGradientBracketSweepConfig,
    load_json_artifact,
    nonlinear_turbulence_gradient_bracket_sweep_report,
)


DEFAULT_OUT_PREFIX = ROOT / "docs" / "_static" / "nonlinear_gradient_bracket_sweep"


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
    return out if np.isfinite(out) else float("nan")


def _artifact_label(payload: dict[str, Any], path: Path) -> str:
    parameter = payload.get("parameter_name")
    delta = payload.get("delta_parameter")
    if isinstance(parameter, str) and parameter:
        delta_value = _float_or_nan(delta)
        if not np.isfinite(delta_value):
            return parameter
        return f"{parameter}:delta={delta_value:.4g}"
    return path.stem


def write_artifacts(
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

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
                    "paired_same_sign_fraction": metrics.get("paired_same_sign_fraction"),
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
        [_float_or_nan(row.get("metrics", {}).get("gradient_uncertainty_rel")) for row in rows]
    )
    labels = [str(row.get("label", "")) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8), constrained_layout=True)
    marker_colors = ["#24727f" if bool(row.get("passed", False)) else "#b45f2a" for row in rows]
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
            axes[0].annotate(label.split(":")[-1], (delta, gradient), xytext=(3, 4), textcoords="offset points", fontsize=7)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="+", type=Path)
    parser.add_argument("--json-out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-condition-number", type=float, default=1.0e8)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument("--max-repeated-bracket-uncertainty-rel", type=float, default=0.75)
    parser.add_argument("--min-repeated-bracket-same-sign-fraction", type=float, default=0.80)
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Write JSON/CSV/PNG only. Useful for tracked documentation previews.",
    )
    parser.add_argument("--fail-on-no-promotable", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payloads = [load_json_artifact(path) for path in args.artifact]
    report = nonlinear_turbulence_gradient_bracket_sweep_report(
        payloads,
        labels=[_artifact_label(payload, path) for payload, path in zip(payloads, args.artifact)],
        paths=[_repo_relative(path) for path in args.artifact],
        config=NonlinearTurbulenceGradientBracketSweepConfig(
            max_gradient_uncertainty_rel=float(args.max_gradient_uncertainty_rel),
            max_fd_asymmetry_rel=float(args.max_fd_asymmetry_rel),
            max_fd_condition_number=float(args.max_fd_condition_number),
            min_fd_response_fraction=float(args.min_fd_response_fraction),
            max_repeated_bracket_uncertainty_rel=float(args.max_repeated_bracket_uncertainty_rel),
            min_repeated_bracket_same_sign_fraction=float(
                args.min_repeated_bracket_same_sign_fraction
            ),
        ),
    )
    paths = write_artifacts(report, Path(args.json_out_prefix), write_pdf=not bool(args.no_pdf))
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


if __name__ == "__main__":
    raise SystemExit(main())
