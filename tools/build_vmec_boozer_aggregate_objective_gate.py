#!/usr/bin/env python3
"""Build the multi-point VMEC/Boozer aggregate-objective FD artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.solver_objective_gradients import (  # noqa: E402
    solver_grid_options_from_ky_values,
    vmec_boozer_aggregate_scalar_objective_finite_difference_report,
)
from tools.build_solver_objective_gradient_gate import _json_clean  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_aggregate_objective_gate.png"


def _surface_indices(raw: list[int] | None) -> tuple[int | None, ...]:
    if not raw:
        return (None,)
    return tuple(int(item) for item in raw)


def write_vmec_boozer_aggregate_objective_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for an aggregate-objective payload."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    samples = payload.get("samples", [])
    base_values = payload.get("base_sample_values", [])
    minus_values = payload.get("minus_sample_values", [])
    plus_values = payload.get("plus_sample_values", [])
    rows = samples if isinstance(samples, list) else []
    fieldnames = [
        "sample",
        "surface_index",
        "torflux",
        "surface",
        "alpha",
        "ky",
        "selected_ky_index",
        "selected_ky",
        "ky_abs_error",
        "weight",
        "minus_value",
        "base_value",
        "plus_value",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            sample = row if isinstance(row, dict) else {}
            writer.writerow(
                {
                    "sample": index,
                    "surface_index": sample.get("surface_index", ""),
                    "torflux": sample.get("torflux", ""),
                    "surface": sample.get("surface", ""),
                    "alpha": sample.get("alpha", ""),
                    "ky": sample.get("ky", ""),
                    "selected_ky_index": sample.get("selected_ky_index", ""),
                    "selected_ky": sample.get("selected_ky", ""),
                    "ky_abs_error": sample.get("ky_abs_error", ""),
                    "weight": sample.get("weight", ""),
                    "minus_value": minus_values[index] if isinstance(minus_values, list) and index < len(minus_values) else "",
                    "base_value": base_values[index] if isinstance(base_values, list) and index < len(base_values) else "",
                    "plus_value": plus_values[index] if isinstance(plus_values, list) and index < len(plus_values) else "",
                }
            )

    base = np.asarray(base_values, dtype=float)
    minus = np.asarray(minus_values, dtype=float)
    plus = np.asarray(plus_values, dtype=float)
    labels = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        surface = row.get("surface_index", "mid")
        if row.get("torflux") not in (None, ""):
            surface_label = f"{float(row['torflux']):.3g}"
        else:
            surface_label = "mid" if surface is None else str(surface)
        ky_label = (
            f"ky={float(row['ky']):.3g}"
            if "ky" in row and row.get("ky") not in (None, "")
            else f"ky#{int(row.get('selected_ky_index', 0))}"
        )
        labels.append(
            f"s={surface_label}, a={float(row.get('alpha', 0.0)):.2g}, "
            f"{ky_label}"
        )
    if not labels:
        labels = [str(index) for index in range(base.size)]

    set_plot_style()
    fig, (ax_values, ax_summary) = plt.subplots(1, 2, figsize=(12.2, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
    x = np.arange(base.size)
    width = 0.24
    ax_values.bar(x - width, minus, width, label=r"$x-h$", color="#8ecae6", edgecolor="#202020", linewidth=0.4)
    ax_values.bar(x, base, width, label=r"$x$", color="#219ebc", edgecolor="#202020", linewidth=0.4)
    ax_values.bar(x + width, plus, width, label=r"$x+h$", color="#023047", edgecolor="#202020", linewidth=0.4)
    ax_values.set_xticks(x, labels, rotation=18, ha="right")
    ax_values.set_ylabel(str(payload.get("objective", "objective")))
    ax_values.set_title("Per-sample scalar objective")
    ax_values.grid(axis="y", alpha=0.25)
    ax_values.legend(frameon=False, fontsize=8)

    passed = bool(payload.get("passed"))
    status = "passed" if passed else "open"
    summary_lines = [
        f"status: {status}",
        f"objective: {payload.get('objective')}",
        f"reduction: {payload.get('reduction')}",
        f"samples: {payload.get('n_samples')}",
        f"base: {float(payload.get('base_value', float('nan'))):.6g}",
        f"central FD: {float(payload.get('central_derivative', float('nan'))):.6g}",
        f"response: {float(payload.get('response_abs', float('nan'))):.3e}",
        f"curvature ratio: {float(payload.get('curvature_ratio', float('nan'))):.3e}",
    ]
    ax_summary.axis("off")
    ax_summary.set_title("Aggregate FD gate")
    ax_summary.text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=ax_summary.transAxes,
    )
    ax_summary.text(
        0.02,
        0.16,
        "Finite-difference sensitivity through the in-memory\n"
        "VMEC/Boozer/SPECTRAX-GK value path. This is a\n"
        "multi-point reduced linear/QL objective gate, not a\n"
        "nonlinear turbulent transport optimization claim.",
        va="top",
        ha="left",
        fontsize=8.3,
        transform=ax_summary.transAxes,
    )
    fig.suptitle(f"VMEC/Boozer multi-point aggregate-objective gate: {status}")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22, wspace=0.28)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path), "csv": str(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument("--objective", default="quasilinear_flux")
    parser.add_argument("--reduction", choices=["mean", "weighted_mean", "max"], default="mean")
    parser.add_argument("--surface-indices", nargs="*", type=int, default=[])
    parser.add_argument(
        "--torflux-values",
        nargs="*",
        type=float,
        default=[],
        help="Optional physical normalized toroidal-flux samples. Cannot be combined with --surface-indices.",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--selected-ky-indices", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--ky-values",
        nargs="*",
        type=float,
        default=[],
        help="Optional physical ky*rho_i values. When set, selected indices, Ly, and Ny are inferred.",
    )
    parser.add_argument(
        "--ky-base",
        type=float,
        default=None,
        help="Base ky spacing for --ky-values; defaults to the smallest requested ky.",
    )
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--perturbation-step", type=float, default=1.0e-7)
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


def _annotate_physical_ky_samples(
    payload: dict[str, object],
    *,
    requested_ky_values: list[float],
    solver_grid_options: dict[str, object],
) -> None:
    index_to_requested = {
        int(index): float(ky)
        for index, ky in zip(
            solver_grid_options["selected_ky_indices"],
            requested_ky_values,
            strict=True,
        )
    }
    index_to_resolved = {
        int(index): float(ky)
        for index, ky in zip(
            solver_grid_options["selected_ky_indices"],
            solver_grid_options["resolved_ky_values"],
            strict=True,
        )
    }
    rows = payload.get("samples")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        selected = int(row.get("selected_ky_index", 0))
        if selected not in index_to_requested:
            continue
        requested = index_to_requested[selected]
        resolved = index_to_resolved[selected]
        row["ky"] = requested
        row["selected_ky"] = resolved
        row["ky_abs_error"] = abs(resolved - requested)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torflux_values and args.surface_indices:
        raise ValueError("use --torflux-values or --surface-indices, not both")
    selected_ky_indices = tuple(args.selected_ky_indices)
    solver_grid_options: dict[str, object] = {}
    objective_kwargs: dict[str, object] = {
        "ntheta": args.ntheta,
        "mboz": args.mboz,
        "nboz": args.nboz,
        "surface_stencil_width": None if args.surface_stencil_width <= 0 else args.surface_stencil_width,
        "n_laguerre": args.n_laguerre,
        "n_hermite": args.n_hermite,
        "nx": args.nx,
        "ny": args.ny,
    }
    if args.ky_values:
        solver_grid_options = solver_grid_options_from_ky_values(
            tuple(args.ky_values),
            ky_base=args.ky_base,
            min_ny=args.ny,
        )
        selected_ky_indices = tuple(int(item) for item in solver_grid_options["selected_ky_indices"])
        objective_kwargs["ny"] = int(solver_grid_options["ny"])
        objective_kwargs["ly"] = float(solver_grid_options["ly"])

    payload = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
        case_name=args.case_name,
        objective=args.objective,
        reduction=args.reduction,
        surface_indices=(None,) if args.torflux_values else _surface_indices(args.surface_indices),
        torflux_values=tuple(args.torflux_values) if args.torflux_values else None,
        alphas=tuple(args.alphas),
        selected_ky_indices=selected_ky_indices,
        radial_index=args.radial_index,
        mode_index=args.mode_index,
        perturbation_step=args.perturbation_step,
        response_atol=args.response_atol,
        max_curvature_ratio=args.max_curvature_ratio,
        **objective_kwargs,
    )
    if solver_grid_options:
        requested_ky_values = [float(item) for item in args.ky_values]
        _annotate_physical_ky_samples(
            payload,
            requested_ky_values=requested_ky_values,
            solver_grid_options=solver_grid_options,
        )
        payload["requested_ky_values"] = requested_ky_values
        payload["ky_values"] = requested_ky_values
        payload["solver_grid_options_from_ky_values"] = solver_grid_options
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_vmec_boozer_aggregate_objective_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
