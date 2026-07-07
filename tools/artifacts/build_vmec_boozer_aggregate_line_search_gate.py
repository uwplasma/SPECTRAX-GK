#!/usr/bin/env python3
"""Build the multi-point VMEC/Boozer aggregate-objective line-search artifact."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from spectraxgk.objectives.vmec_boozer_line_search import (  # noqa: E402
    vmec_boozer_aggregate_scalar_objective_line_search_report,
)
from tools.artifacts.build_solver_objective_gradient_gate import _json_clean  # noqa: E402
from tools.artifacts.build_vmec_boozer_aggregate_objective_gate import _surface_indices  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_aggregate_line_search_gate.png"


def write_vmec_boozer_aggregate_line_search_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for an aggregate line-search payload."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    history = payload.get("history", [])
    rows = history if isinstance(history, list) else []
    fieldnames = [
        "step",
        "delta",
        "objective",
        "central_derivative",
        "curvature_ratio",
        "accepted",
        "candidate_delta",
        "candidate_objective",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    step_labels = [
        str(row.get("step", index))
        for index, row in enumerate(rows)
        if isinstance(row, dict)
    ]
    objectives = np.asarray(
        [float(row.get("objective", np.nan)) for row in rows if isinstance(row, dict)],
        dtype=float,
    )
    candidates = np.asarray(
        [
            float(row.get("candidate_objective", np.nan))
            for row in rows
            if isinstance(row, dict)
        ],
        dtype=float,
    )
    curvature = np.asarray(
        [
            float(row.get("curvature_ratio", np.nan))
            for row in rows
            if isinstance(row, dict)
        ],
        dtype=float,
    )

    set_plot_style()
    fig, (ax_obj, ax_meta) = plt.subplots(
        1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )
    x = np.arange(max(len(objectives), 1))
    if objectives.size:
        ax_obj.plot(x, objectives, marker="o", lw=2.0, color="#005f73", label="current")
    if candidates.size:
        ax_obj.plot(
            x, candidates, marker="s", lw=1.8, color="#ca6702", label="candidate"
        )
    ax_obj.set_xticks(x, step_labels or ["0"])
    ax_obj.set_xlabel("line-search step")
    ax_obj.set_ylabel(str(payload.get("objective", "objective")))
    ax_obj.set_title("Aggregate objective decrease")
    ax_obj.grid(alpha=0.25)
    ax_obj.legend(frameon=False, fontsize=8)

    if curvature.size and np.all(np.isfinite(curvature)):
        ax_curve = ax_obj.twinx()
        ax_curve.plot(
            x,
            curvature,
            marker="^",
            lw=1.3,
            ls="--",
            color="#6a4c93",
            label="curvature",
        )
        ax_curve.set_ylabel("curvature ratio")
        ax_curve.tick_params(axis="y", labelcolor="#6a4c93")

    passed = bool(payload.get("passed"))
    status = "passed" if passed else "open"
    summary_lines = [
        f"status: {status}",
        f"objective: {payload.get('objective')}",
        f"reduction: {payload.get('reduction')}",
        f"samples: {payload.get('n_samples')}",
        f"accepted: {payload.get('accepted_steps')}/{payload.get('max_steps')}",
        f"initial: {float(payload.get('initial_objective', float('nan'))):.6g}",
        f"final: {float(payload.get('final_objective', float('nan'))):.6g}",
        f"rel. reduction: {payload.get('relative_reduction')}",
        f"stop: {payload.get('stop_reason')}",
    ]
    ax_meta.axis("off")
    ax_meta.set_title("Line-search gate")
    ax_meta.text(
        0.02,
        0.95,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.4,
        transform=ax_meta.transAxes,
    )
    ax_meta.text(
        0.02,
        0.14,
        "Every attempted update must pass the aggregate\n"
        "finite-difference curvature gate and decrease the\n"
        "multi-point reduced objective. This is optimizer\n"
        "control-flow evidence, not a nonlinear transport claim.",
        va="top",
        ha="left",
        fontsize=8.3,
        transform=ax_meta.transAxes,
    )
    fig.suptitle(f"VMEC/Boozer aggregate-objective line-search gate: {status}")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.18, wspace=0.28)
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
    parser.add_argument("--objective", default="quasilinear_flux")
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
    payload = vmec_boozer_aggregate_scalar_objective_line_search_report(
        case_name=args.case_name,
        objective=args.objective,
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
    paths = write_vmec_boozer_aggregate_line_search_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
