#!/usr/bin/env python3
"""Compare growth and QL aggregate VMEC/Boozer line searches on one sample set."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from spectraxgk.objectives.vmec_boozer_line_search import (  # noqa: E402
    vmec_boozer_aggregate_scalar_objective_line_search_report,
)
from tools.artifacts.build_solver_objective_gradient_gate import _json_clean  # noqa: E402
from tools.artifacts.build_vmec_boozer_aggregate_objective_gate import _surface_indices  # noqa: E402

DEFAULT_OUT = (
    ROOT / "docs" / "_static" / "vmec_boozer_aggregate_line_search_comparison.png"
)
DEFAULT_OBJECTIVES = ("growth", "quasilinear_flux")


def _finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _first_history_row(report: dict[str, object]) -> dict[str, object]:
    history = report.get("history", [])
    if isinstance(history, list) and history and isinstance(history[0], dict):
        return history[0]
    return {}


def _line_search_direction(derivative: float) -> str:
    if not math.isfinite(derivative) or derivative == 0.0:
        return "stationary_or_unresolved"
    return "negative_delta" if derivative > 0.0 else "positive_delta"


def _sample_key(sample: object) -> tuple[object, ...]:
    if not isinstance(sample, dict):
        return (repr(sample),)
    return (
        sample.get("surface_index"),
        _finite_float(sample.get("alpha")),
        sample.get("selected_ky_index"),
        _finite_float(sample.get("weight")),
    )


def _samples_match(reports: dict[str, dict[str, object]]) -> bool:
    keys: list[tuple[tuple[object, ...], ...]] = []
    for report in reports.values():
        samples = report.get("samples", [])
        if not isinstance(samples, list) or not samples:
            return False
        keys.append(tuple(_sample_key(sample) for sample in samples))
    return bool(keys) and all(key == keys[0] for key in keys[1:])


def _summarize_report(objective: str, report: dict[str, object]) -> dict[str, object]:
    first = _first_history_row(report)
    derivative = _finite_float(first.get("central_derivative"))
    initial = _finite_float(report.get("initial_objective"))
    final = _finite_float(report.get("final_objective"))
    reduction = _finite_float(report.get("relative_reduction"), default=math.nan)
    return {
        "objective": str(objective),
        "passed": bool(report.get("passed", False)),
        "n_samples": int(report.get("n_samples", 0) or 0),
        "initial_objective": initial,
        "final_objective": final,
        "absolute_reduction": initial - final
        if math.isfinite(initial) and math.isfinite(final)
        else math.nan,
        "relative_reduction": reduction,
        "initial_central_derivative": derivative,
        "initial_update_direction": _line_search_direction(derivative),
        "accepted_steps": int(report.get("accepted_steps", 0) or 0),
        "max_steps": int(report.get("max_steps", 0) or 0),
        "initial_delta": _finite_float(report.get("initial_delta")),
        "final_delta": _finite_float(report.get("final_delta")),
        "stop_reason": str(report.get("stop_reason", "")),
    }


def build_vmec_boozer_aggregate_line_search_comparison_report(
    *,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    case_name: str = "nfp4_QH_warm_start",
    reduction: str = "mean",
    surface_indices: tuple[int | None, ...] = (None,),
    alphas: tuple[float, ...] = (0.0,),
    selected_ky_indices: tuple[int, ...] = (1, 2),
    radial_index: int | None = None,
    mode_index: int = 1,
    perturbation_step: float = 1.0e-7,
    update_step: float = 1.0e-8,
    max_steps: int = 1,
    min_improvement: float = 0.0,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    """Run aggregate line searches for each objective and compare summaries."""

    objective_tuple = tuple(str(item) for item in objectives)
    if len(objective_tuple) < 2:
        raise ValueError("objectives must contain at least two entries")
    reports: dict[str, dict[str, object]] = {}
    for objective in objective_tuple:
        reports[objective] = vmec_boozer_aggregate_scalar_objective_line_search_report(
            case_name=case_name,
            objective=objective,  # type: ignore[arg-type]
            reduction=reduction,  # type: ignore[arg-type]
            surface_indices=surface_indices,
            alphas=alphas,
            selected_ky_indices=selected_ky_indices,
            radial_index=radial_index,
            mode_index=mode_index,
            perturbation_step=perturbation_step,
            update_step=update_step,
            max_steps=max_steps,
            min_improvement=min_improvement,
            response_atol=response_atol,
            max_curvature_ratio=max_curvature_ratio,
            **kwargs,
        )

    rows = [
        _summarize_report(objective, reports[objective])
        for objective in objective_tuple
    ]
    same_samples = _samples_match(reports)
    directions = [str(row["initial_update_direction"]) for row in rows]
    same_direction = bool(
        directions and all(direction == directions[0] for direction in directions[1:])
    )
    all_line_searches_passed = all(bool(row["passed"]) for row in rows)
    final_deltas = [_finite_float(row["final_delta"]) for row in rows]
    delta_spread = (
        float(max(final_deltas) - min(final_deltas))
        if all(math.isfinite(item) for item in final_deltas)
        else None
    )
    reductions = [_finite_float(row["relative_reduction"]) for row in rows]
    finite_reductions = [item for item in reductions if math.isfinite(item)]
    reduction_spread = (
        float(max(finite_reductions) - min(finite_reductions))
        if finite_reductions
        else None
    )

    return {
        "kind": "vmec_boozer_aggregate_line_search_comparison",
        "passed": bool(all_line_searches_passed and same_samples),
        "source_scope": "mode21_vmec_boozer_state_multi_point",
        "claim_scope": (
            "side-by-side one-parameter line-search comparison for reduced aggregate growth "
            "and quasilinear proxy objectives on the same QH VMEC/Boozer sample set; "
            "not a nonlinear turbulent transport or optimizer-convergence claim"
        ),
        "case_name": str(case_name),
        "objectives": list(objective_tuple),
        "reduction": str(reduction),
        "n_samples": int(rows[0].get("n_samples", 0) or 0) if rows else 0,
        "same_sample_set": same_samples,
        "all_line_searches_passed": all_line_searches_passed,
        "same_initial_update_direction": same_direction,
        "initial_update_directions": {
            str(row["objective"]): str(row["initial_update_direction"]) for row in rows
        },
        "final_delta_spread": delta_spread,
        "relative_reduction_spread": reduction_spread,
        "rows": rows,
        "reports": reports,
        "notes": (
            "Direction agreement is recorded as a diagnostic, not a pass/fail criterion beyond "
            "requiring both underlying line-search reports to be valid on the same samples."
        ),
        "options": {
            key: value
            for key, value in kwargs.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }


def write_vmec_boozer_aggregate_line_search_comparison_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for an aggregate line-search comparison."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = payload.get("rows", [])
    row_list = rows if isinstance(rows, list) else []
    fieldnames = [
        "objective",
        "passed",
        "n_samples",
        "initial_objective",
        "final_objective",
        "absolute_reduction",
        "relative_reduction",
        "initial_central_derivative",
        "initial_update_direction",
        "accepted_steps",
        "max_steps",
        "initial_delta",
        "final_delta",
        "stop_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in row_list:
            if isinstance(row, dict):
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    labels = [
        str(row.get("objective", index))
        for index, row in enumerate(row_list)
        if isinstance(row, dict)
    ]
    initial = np.asarray(
        [
            _finite_float(row.get("initial_objective"))
            for row in row_list
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    final = np.asarray(
        [
            _finite_float(row.get("final_objective"))
            for row in row_list
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    rel = np.asarray(
        [
            _finite_float(row.get("relative_reduction"))
            for row in row_list
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    derivative = np.asarray(
        [
            _finite_float(row.get("initial_central_derivative"))
            for row in row_list
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    final_delta = np.asarray(
        [
            _finite_float(row.get("final_delta"))
            for row in row_list
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    finite_initial = np.where(
        np.isfinite(initial) & (np.abs(initial) > 0.0), initial, np.nan
    )
    normalized_final = final / finite_initial

    set_plot_style()
    fig = plt.figure(figsize=(13.6, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05])
    ax_obj = fig.add_subplot(gs[0, 0])
    ax_dir = fig.add_subplot(gs[0, 1])
    ax_meta = fig.add_subplot(gs[0, 2])
    x = np.arange(max(len(labels), 1))
    width = 0.34
    if labels:
        ax_obj.bar(
            x - width / 2.0,
            np.ones_like(x, dtype=float),
            width,
            label="initial",
            color="#94d2bd",
            edgecolor="#1f2937",
        )
        ax_obj.bar(
            x + width / 2.0,
            normalized_final,
            width,
            label="final / initial",
            color="#005f73",
            edgecolor="#1f2937",
        )
        for xi, value in zip(x, rel, strict=True):
            text = f"{value:.2%}" if math.isfinite(float(value)) else "n/a"
            ax_obj.text(xi, 1.03, text, ha="center", va="bottom", fontsize=8)
    ax_obj.set_xticks(x, labels or ["objective"], rotation=15, ha="right")
    ax_obj.set_ylabel("normalized objective")
    ax_obj.set_title("Line-search outcome")
    ax_obj.grid(axis="y", alpha=0.25)
    ax_obj.legend(frameon=False, fontsize=8)

    signs = np.sign(derivative)
    colors = [
        "#ca6702" if sign > 0.0 else "#0a9396" if sign < 0.0 else "#6b7280"
        for sign in signs
    ]
    if labels:
        ax_dir.bar(x, final_delta, color=colors, edgecolor="#1f2937")
        for xi, deriv in zip(x, derivative, strict=True):
            if math.isfinite(float(deriv)):
                ax_dir.text(
                    xi,
                    0.0,
                    f"dJ/dx={deriv:.2g}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=90,
                )
    ax_dir.axhline(0.0, color="#111827", linewidth=1.0)
    ax_dir.set_xticks(x, labels or ["objective"], rotation=15, ha="right")
    ax_dir.set_ylabel("final VMEC coefficient delta")
    ax_dir.set_title("Initial descent direction")
    ax_dir.grid(axis="y", alpha=0.25)

    passed = bool(payload.get("passed"))
    status = "passed" if passed else "open"
    summary_lines = [
        f"status: {status}",
        f"case: {payload.get('case_name')}",
        f"reduction: {payload.get('reduction')}",
        f"samples: {payload.get('n_samples')}",
        f"same samples: {payload.get('same_sample_set')}",
        f"same direction: {payload.get('same_initial_update_direction')}",
        f"delta spread: {payload.get('final_delta_spread')}",
        f"rel. reduction spread: {payload.get('relative_reduction_spread')}",
    ]
    ax_meta.axis("off")
    ax_meta.set_title("Comparison scope")
    ax_meta.text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.2,
        transform=ax_meta.transAxes,
    )
    ax_meta.text(
        0.02,
        0.22,
        "Growth and QL proxy line searches are run with\n"
        "the same QH surfaces, field lines, ky samples,\n"
        "VMEC coefficient, and finite-difference controls.\n"
        "This compares reduced directions/results only; it\n"
        "does not validate nonlinear turbulent transport.",
        va="top",
        ha="left",
        fontsize=8.3,
        transform=ax_meta.transAxes,
    )
    fig.suptitle(f"VMEC/Boozer aggregate growth-vs-QL line search: {status}", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.20, wspace=0.35)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "png": str(out_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument("--objectives", nargs="+", default=list(DEFAULT_OBJECTIVES))
    parser.add_argument(
        "--reduction", choices=["mean", "weighted_mean", "max"], default="mean"
    )
    parser.add_argument("--surface-indices", nargs="*", type=int, default=[])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--selected-ky-indices", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--perturbation-step", type=float, default=1.0e-7)
    parser.add_argument("--update-step", type=float, default=1.0e-8)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--max-curvature-ratio", type=float, default=5.0)
    parser.add_argument("--response-atol", type=float, default=0.0)
    parser.add_argument("--ntheta", type=int, default=4)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--surface-stencil-width", type=int, default=3)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_vmec_boozer_aggregate_line_search_comparison_report(
        objectives=tuple(args.objectives),
        case_name=args.case_name,
        reduction=args.reduction,
        surface_indices=_surface_indices(args.surface_indices),
        alphas=tuple(args.alphas),
        selected_ky_indices=tuple(args.selected_ky_indices),
        radial_index=args.radial_index,
        mode_index=args.mode_index,
        perturbation_step=args.perturbation_step,
        update_step=args.update_step,
        max_steps=args.max_steps,
        min_improvement=args.min_improvement,
        response_atol=args.response_atol,
        max_curvature_ratio=args.max_curvature_ratio,
        ntheta=args.ntheta,
        mboz=args.mboz,
        nboz=args.nboz,
        surface_stencil_width=None
        if args.surface_stencil_width <= 0
        else args.surface_stencil_width,
        n_laguerre=args.n_laguerre,
        n_hermite=args.n_hermite,
        nx=args.nx,
        ny=args.ny,
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_vmec_boozer_aggregate_line_search_comparison_artifacts(
        payload, out=args.out
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
