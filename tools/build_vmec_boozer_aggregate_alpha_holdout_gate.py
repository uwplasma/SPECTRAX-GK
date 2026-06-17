#!/usr/bin/env python3
"""Build the VMEC/Boozer aggregate alpha-heldout line-search gate artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from spectraxgk.objectives.vmec_boozer_line_search import (  # noqa: E402
    vmec_boozer_aggregate_line_search_holdout_report,
)
from tools.build_solver_objective_gradient_gate import _json_clean  # noqa: E402
from tools.build_vmec_boozer_aggregate_objective_gate import _surface_indices  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_aggregate_alpha_holdout_gate.png"


def _finite_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _write_csv(payload: dict[str, object], csv_path: Path) -> None:
    rows = [
        {
            "split": "training",
            "passed": payload.get("training_passed"),
            "initial_objective": payload.get("training_initial_objective"),
            "final_objective": payload.get("training_final_objective"),
            "relative_reduction": payload.get("training_relative_reduction"),
            "n_samples": len(payload.get("training_samples", []))
            if isinstance(payload.get("training_samples"), list)
            else "",
        },
        {
            "split": "heldout",
            "passed": payload.get("heldout_passed"),
            "initial_objective": payload.get("heldout_initial_objective"),
            "final_objective": payload.get("heldout_final_objective"),
            "relative_reduction": payload.get("heldout_relative_reduction"),
            "n_samples": len(payload.get("heldout_samples", []))
            if isinstance(payload.get("heldout_samples"), list)
            else "",
        },
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_vmec_boozer_aggregate_alpha_holdout_payload(
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: str = "quasilinear_flux",
    reduction: str = "mean",
    training_surface_indices: tuple[int | None, ...] = (None,),
    training_alphas: tuple[float, ...] = (0.0,),
    training_selected_ky_indices: tuple[int, ...] = (1, 2),
    holdout_surface_indices: tuple[int | None, ...] = (None,),
    holdout_alphas: tuple[float, ...] = (0.5,),
    holdout_selected_ky_indices: tuple[int, ...] = (1, 2),
    **kwargs: object,
) -> dict[str, object]:
    """Run the reduced aggregate line search and held-out alpha audit."""

    start = time.perf_counter()
    payload = vmec_boozer_aggregate_line_search_holdout_report(
        case_name=case_name,
        objective=objective,  # type: ignore[arg-type]
        reduction=reduction,  # type: ignore[arg-type]
        training_surface_indices=training_surface_indices,
        training_alphas=training_alphas,
        training_selected_ky_indices=training_selected_ky_indices,
        holdout_surface_indices=holdout_surface_indices,
        holdout_alphas=holdout_alphas,
        holdout_selected_ky_indices=holdout_selected_ky_indices,
        **kwargs,
    )
    annotated = dict(payload)
    annotated["artifact_kind"] = "vmec_boozer_aggregate_alpha_holdout_gate"
    annotated["builder"] = "tools/build_vmec_boozer_aggregate_alpha_holdout_gate.py"
    annotated["wall_seconds"] = time.perf_counter() - start
    annotated["claim_scope"] = (
        "reduced aggregate VMEC/Boozer/SPECTRAX-GK line-search split with a "
        "held-out field-line alpha; this is reduced growth/quasilinear "
        "objective evidence, not a nonlinear turbulent transport claim"
    )
    annotated["holdout_split"] = {
        "training_surface_indices": [None if item is None else int(item) for item in training_surface_indices],
        "training_alphas": [float(item) for item in training_alphas],
        "training_selected_ky_indices": [int(item) for item in training_selected_ky_indices],
        "holdout_surface_indices": [None if item is None else int(item) for item in holdout_surface_indices],
        "holdout_alphas": [float(item) for item in holdout_alphas],
        "holdout_selected_ky_indices": [int(item) for item in holdout_selected_ky_indices],
    }
    annotated["next_action"] = (
        "Repeat on a held-out surface and at least one second equilibrium before "
        "promoting reduced optimizer figures; production nonlinear optimization "
        "still requires converged nonlinear transport audits."
    )
    return annotated


def write_vmec_boozer_aggregate_alpha_holdout_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the alpha-heldout gate."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(payload, csv_path)

    labels = ["training", "held-out alpha"]
    initial = np.asarray(
        [
            _finite_float(payload.get("training_initial_objective")),
            _finite_float(payload.get("heldout_initial_objective")),
        ],
        dtype=float,
    )
    final = np.asarray(
        [
            _finite_float(payload.get("training_final_objective")),
            _finite_float(payload.get("heldout_final_objective")),
        ],
        dtype=float,
    )
    reductions = np.asarray(
        [
            _finite_float(payload.get("training_relative_reduction")),
            _finite_float(payload.get("heldout_relative_reduction")),
        ],
        dtype=float,
    )
    normalized_final = final / np.where(np.abs(initial) > 0.0, initial, np.nan)

    set_plot_style()
    fig = plt.figure(figsize=(12.0, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15])
    ax_obj = fig.add_subplot(gs[0, 0])
    ax_red = fig.add_subplot(gs[0, 1])
    ax_meta = fig.add_subplot(gs[0, 2])
    x = np.arange(2)
    width = 0.34
    ax_obj.bar(x - width / 2.0, np.ones_like(x, dtype=float), width, label="initial", color="#94d2bd", edgecolor="#1f2937")
    ax_obj.bar(x + width / 2.0, normalized_final, width, label="final / initial", color="#005f73", edgecolor="#1f2937")
    ax_obj.set_xticks(x, labels)
    ax_obj.set_ylabel("normalized objective")
    ax_obj.set_title("Split objective")
    ax_obj.grid(axis="y", alpha=0.25)
    ax_obj.legend(frameon=False, fontsize=8)

    ax_red.bar(x, reductions, color=["#0a9396", "#ee9b00"], edgecolor="#1f2937")
    for xi, reduction in zip(x, reductions, strict=True):
        if np.isfinite(reduction):
            ax_red.text(xi, reduction, f"{reduction:.2e}", ha="center", va="bottom", fontsize=8)
    ax_red.axhline(0.0, color="#111827", linewidth=1.0)
    ax_red.set_xticks(x, labels)
    ax_red.set_ylabel("relative reduction")
    ax_red.set_title("Generalization check")
    ax_red.grid(axis="y", alpha=0.25)

    passed = bool(payload.get("passed"))
    status = "passed" if passed else "open"
    summary_lines = [
        f"status: {status}",
        f"objective: {payload.get('objective')}",
        f"final delta: {payload.get('final_delta')}",
        f"training passed: {payload.get('training_passed')}",
        f"heldout passed: {payload.get('heldout_passed')}",
        f"train rel.: {payload.get('training_relative_reduction')}",
        f"heldout rel.: {payload.get('heldout_relative_reduction')}",
        f"wall seconds: {_finite_float(payload.get('wall_seconds')):.2f}",
    ]
    ax_meta.axis("off")
    ax_meta.set_title("Claim boundary")
    ax_meta.text(0.02, 0.95, "\n".join(summary_lines), va="top", ha="left", family="monospace", fontsize=9.0, transform=ax_meta.transAxes)
    ax_meta.text(
        0.02,
        0.20,
        "The accepted QH coefficient update is trained on\n"
        "alpha=0 and evaluated on a held-out alpha=0.5\n"
        "with the same ky samples. This is a reduced\n"
        "linear/quasilinear split gate, not a nonlinear\n"
        "transport optimization claim.",
        va="top",
        ha="left",
        fontsize=8.2,
        transform=ax_meta.transAxes,
    )
    fig.suptitle(f"VMEC/Boozer aggregate alpha-heldout gate: {status}", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.18, wspace=0.34)
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
    parser.add_argument("--training-surface-indices", nargs="*", type=int, default=[])
    parser.add_argument("--training-alphas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--training-selected-ky-indices", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--holdout-surface-indices", nargs="*", type=int, default=[])
    parser.add_argument("--holdout-alphas", nargs="+", type=float, default=[0.5])
    parser.add_argument("--holdout-selected-ky-indices", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--perturbation-step", type=float, default=1.0e-7)
    parser.add_argument("--update-step", type=float, default=1.0e-8)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--min-holdout-improvement", type=float, default=0.0)
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
    payload = build_vmec_boozer_aggregate_alpha_holdout_payload(
        case_name=args.case_name,
        objective=args.objective,
        reduction=args.reduction,
        training_surface_indices=_surface_indices(args.training_surface_indices),
        training_alphas=tuple(args.training_alphas),
        training_selected_ky_indices=tuple(args.training_selected_ky_indices),
        holdout_surface_indices=_surface_indices(args.holdout_surface_indices),
        holdout_alphas=tuple(args.holdout_alphas),
        holdout_selected_ky_indices=tuple(args.holdout_selected_ky_indices),
        radial_index=args.radial_index,
        mode_index=args.mode_index,
        perturbation_step=args.perturbation_step,
        update_step=args.update_step,
        max_steps=args.max_steps,
        min_improvement=args.min_improvement,
        min_holdout_improvement=args.min_holdout_improvement,
        response_atol=args.response_atol,
        max_curvature_ratio=args.max_curvature_ratio,
        ntheta=args.ntheta,
        mboz=args.mboz,
        nboz=args.nboz,
        surface_stencil_width=None if args.surface_stencil_width <= 0 else args.surface_stencil_width,
        n_laguerre=args.n_laguerre,
        n_hermite=args.n_hermite,
        nx=args.nx,
        ny=args.ny,
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_vmec_boozer_aggregate_alpha_holdout_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
