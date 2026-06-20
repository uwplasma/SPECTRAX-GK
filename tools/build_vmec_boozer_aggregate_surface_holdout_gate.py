#!/usr/bin/env python3
"""Build the VMEC/Boozer aggregate surface-heldout line-search gate artifact."""

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

DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_aggregate_surface_holdout_gate.png"


def _finite_float(value: object, default: float = np.nan) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _surface_tuple(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    result = tuple(int(item) for item in values)
    if not result:
        raise ValueError("surface holdout splits must contain at least one explicit surface_index")
    return result


def _same_surface_set(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return {int(item) for item in left} == {int(item) for item in right}


def _holdout_surface_blockers(
    payload: dict[str, object],
    *,
    training_surface_indices: tuple[int, ...],
    holdout_surface_indices: tuple[int, ...],
) -> list[str]:
    blockers: list[str] = []
    if _same_surface_set(training_surface_indices, holdout_surface_indices):
        blockers.append("surface_split_not_held_out")
    if not bool(payload.get("training_passed")):
        blockers.append("training_line_search_not_accepted")
    if not bool(payload.get("heldout_passed")):
        blockers.append("heldout_surface_objective_not_reduced")
    if not bool(payload.get("passed")):
        blockers.append("aggregate_surface_holdout_gate_not_passed")
    return sorted(set(blockers))


def _blocked_payload(
    *,
    case_name: str,
    objective: str,
    reduction: str,
    training_surface_indices: tuple[int, ...],
    training_alphas: tuple[float, ...],
    training_selected_ky_indices: tuple[int, ...],
    holdout_surface_indices: tuple[int, ...],
    holdout_alphas: tuple[float, ...],
    holdout_selected_ky_indices: tuple[int, ...],
    blocker: str,
    exc: BaseException | None = None,
    wall_seconds: float = 0.0,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "vmec_boozer_aggregate_surface_holdout_gate",
        "artifact_kind": "vmec_boozer_aggregate_surface_holdout_gate",
        "builder": "tools/build_vmec_boozer_aggregate_surface_holdout_gate.py",
        "passed": False,
        "blocked": True,
        "blockers": [blocker],
        "case_name": case_name,
        "objective": objective,
        "reduction": reduction,
        "wall_seconds": float(wall_seconds),
        "holdout_split": {
            "training_surface_indices": [int(item) for item in training_surface_indices],
            "training_alphas": [float(item) for item in training_alphas],
            "training_selected_ky_indices": [int(item) for item in training_selected_ky_indices],
            "holdout_surface_indices": [int(item) for item in holdout_surface_indices],
            "holdout_alphas": [float(item) for item in holdout_alphas],
            "holdout_selected_ky_indices": [int(item) for item in holdout_selected_ky_indices],
        },
        "claim_scope": (
            "closed blocker for a true surface_index VMEC/Boozer aggregate reduced-objective "
            "line-search split; no alpha-only or ky-only substitute is promoted"
        ),
        "next_action": "Fix fixture/API surface_index support or choose valid distinct interior surfaces, then rerun this gate.",
    }
    if exc is not None:
        payload["exception_type"] = type(exc).__name__
        payload["exception_message"] = str(exc)
    return payload


def build_vmec_boozer_aggregate_surface_holdout_payload(
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: str = "quasilinear_flux",
    reduction: str = "mean",
    training_surface_indices: tuple[int, ...] = (18,),
    training_alphas: tuple[float, ...] = (0.0,),
    training_selected_ky_indices: tuple[int, ...] = (1, 2),
    holdout_surface_indices: tuple[int, ...] = (19,),
    holdout_alphas: tuple[float, ...] = (0.0,),
    holdout_selected_ky_indices: tuple[int, ...] = (1, 2),
    **kwargs: object,
) -> dict[str, object]:
    """Run a reduced aggregate line search with a distinct held-out surface."""

    train_surfaces = _surface_tuple(training_surface_indices)
    holdout_surfaces = _surface_tuple(holdout_surface_indices)
    training_alpha_values = tuple(float(item) for item in training_alphas)
    holdout_alpha_values = tuple(float(item) for item in holdout_alphas)
    training_ky_values = tuple(int(item) for item in training_selected_ky_indices)
    holdout_ky_values = tuple(int(item) for item in holdout_selected_ky_indices)
    start = time.perf_counter()

    if _same_surface_set(train_surfaces, holdout_surfaces):
        return _blocked_payload(
            case_name=case_name,
            objective=objective,
            reduction=reduction,
            training_surface_indices=train_surfaces,
            training_alphas=training_alpha_values,
            training_selected_ky_indices=training_ky_values,
            holdout_surface_indices=holdout_surfaces,
            holdout_alphas=holdout_alpha_values,
            holdout_selected_ky_indices=holdout_ky_values,
            blocker="surface_split_not_held_out",
        )

    try:
        payload = vmec_boozer_aggregate_line_search_holdout_report(
            case_name=case_name,
            objective=objective,  # type: ignore[arg-type]
            reduction=reduction,  # type: ignore[arg-type]
            training_surface_indices=train_surfaces,
            training_alphas=training_alpha_values,
            training_selected_ky_indices=training_ky_values,
            holdout_surface_indices=holdout_surfaces,
            holdout_alphas=holdout_alpha_values,
            holdout_selected_ky_indices=holdout_ky_values,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - this lane must fail closed with JSON.
        return _blocked_payload(
            case_name=case_name,
            objective=objective,
            reduction=reduction,
            training_surface_indices=train_surfaces,
            training_alphas=training_alpha_values,
            training_selected_ky_indices=training_ky_values,
            holdout_surface_indices=holdout_surfaces,
            holdout_alphas=holdout_alpha_values,
            holdout_selected_ky_indices=holdout_ky_values,
            blocker="surface_split_execution_failed",
            exc=exc,
            wall_seconds=time.perf_counter() - start,
        )

    annotated: dict[str, object] = dict(payload)
    annotated["kind"] = "vmec_boozer_aggregate_surface_holdout_gate"
    annotated["source_report_kind"] = payload.get("kind")
    annotated["artifact_kind"] = "vmec_boozer_aggregate_surface_holdout_gate"
    annotated["builder"] = "tools/build_vmec_boozer_aggregate_surface_holdout_gate.py"
    annotated["blocked"] = not bool(payload.get("passed"))
    annotated["blockers"] = _holdout_surface_blockers(
        payload,
        training_surface_indices=train_surfaces,
        holdout_surface_indices=holdout_surfaces,
    )
    annotated["wall_seconds"] = time.perf_counter() - start
    annotated["holdout_split"] = {
        "training_surface_indices": [int(item) for item in train_surfaces],
        "training_alphas": [float(item) for item in training_alpha_values],
        "training_selected_ky_indices": [int(item) for item in training_ky_values],
        "holdout_surface_indices": [int(item) for item in holdout_surfaces],
        "holdout_alphas": [float(item) for item in holdout_alpha_values],
        "holdout_selected_ky_indices": [int(item) for item in holdout_ky_values],
    }
    annotated["claim_scope"] = (
        "true surface_index-heldout reduced aggregate VMEC/Boozer/SPECTRAX-GK "
        "line-search split; this is reduced growth/quasilinear objective evidence, "
        "not a nonlinear turbulent transport claim"
    )
    annotated["next_action"] = (
        "Repeat on additional held-out surfaces and a second equilibrium before promoting "
        "reduced optimizer figures; production claims still require nonlinear transport audits."
    )
    return annotated


def _write_csv(payload: dict[str, object], csv_path: Path) -> None:
    split = payload.get("holdout_split") if isinstance(payload.get("holdout_split"), dict) else {}
    assert isinstance(split, dict)
    rows = [
        {
            "split": "training",
            "surface_indices": " ".join(str(item) for item in split.get("training_surface_indices", [])),
            "passed": payload.get("training_passed"),
            "initial_objective": payload.get("training_initial_objective"),
            "final_objective": payload.get("training_final_objective"),
            "relative_reduction": payload.get("training_relative_reduction"),
            "n_samples": len(payload.get("training_samples", []))
            if isinstance(payload.get("training_samples"), list)
            else "",
        },
        {
            "split": "heldout_surface",
            "surface_indices": " ".join(str(item) for item in split.get("holdout_surface_indices", [])),
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


def write_vmec_boozer_aggregate_surface_holdout_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the surface-heldout gate."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(payload, csv_path)

    labels = ["training", "held-out surface"]
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
    ax_obj.bar(x - width / 2.0, np.ones_like(x, dtype=float), width, label="initial", color="#e9d8a6", edgecolor="#1f2937")
    ax_obj.bar(x + width / 2.0, normalized_final, width, label="final / initial", color="#0a9396", edgecolor="#1f2937")
    ax_obj.set_xticks(x, labels)
    ax_obj.set_ylabel("normalized objective")
    ax_obj.set_title("Surface split objective")
    ax_obj.grid(axis="y", alpha=0.25)
    ax_obj.legend(frameon=False, fontsize=8)

    ax_red.bar(x, reductions, color=["#005f73", "#ca6702"], edgecolor="#1f2937")
    for xi, reduction in zip(x, reductions, strict=True):
        if np.isfinite(reduction):
            ax_red.text(xi, reduction, f"{reduction:.2e}", ha="center", va="bottom", fontsize=8)
    ax_red.axhline(0.0, color="#111827", linewidth=1.0)
    ax_red.set_xticks(x, labels)
    ax_red.set_ylabel("relative reduction")
    ax_red.set_title("Held-out surface check")
    ax_red.grid(axis="y", alpha=0.25)

    passed = bool(payload.get("passed"))
    blocked = bool(payload.get("blocked"))
    status = "passed" if passed else "blocked" if blocked else "open"
    split = payload.get("holdout_split") if isinstance(payload.get("holdout_split"), dict) else {}
    assert isinstance(split, dict)
    summary_lines = [
        f"status: {status}",
        f"objective: {payload.get('objective')}",
        f"train s: {split.get('training_surface_indices')}",
        f"holdout s: {split.get('holdout_surface_indices')}",
        f"final delta: {payload.get('final_delta')}",
        f"training passed: {payload.get('training_passed')}",
        f"heldout passed: {payload.get('heldout_passed')}",
        f"train rel.: {payload.get('training_relative_reduction')}",
        f"heldout rel.: {payload.get('heldout_relative_reduction')}",
        f"blockers: {payload.get('blockers')}",
        f"wall seconds: {_finite_float(payload.get('wall_seconds')):.2f}",
    ]
    ax_meta.axis("off")
    ax_meta.set_title("Claim boundary")
    ax_meta.text(0.02, 0.95, "\n".join(summary_lines), va="top", ha="left", family="monospace", fontsize=8.2, transform=ax_meta.transAxes)
    ax_meta.text(
        0.02,
        0.15,
        "The QH coefficient update is trained on one\n"
        "VMEC/Boozer radial surface_index and audited\n"
        "on a distinct held-out surface_index with the\n"
        "same alpha and ky samples. This is a reduced\n"
        "linear/quasilinear gate, not nonlinear transport.",
        va="top",
        ha="left",
        fontsize=8.0,
        transform=ax_meta.transAxes,
    )
    fig.suptitle(f"VMEC/Boozer aggregate surface-heldout gate: {status}", y=0.98)
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
    parser.add_argument("--training-surface-indices", nargs="+", type=int, default=[18])
    parser.add_argument("--training-alphas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--training-selected-ky-indices", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--holdout-surface-indices", nargs="+", type=int, default=[19])
    parser.add_argument("--holdout-alphas", nargs="+", type=float, default=[0.0])
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
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_vmec_boozer_aggregate_surface_holdout_payload(
        case_name=args.case_name,
        objective=args.objective,
        reduction=args.reduction,
        training_surface_indices=tuple(args.training_surface_indices),
        training_alphas=tuple(args.training_alphas),
        training_selected_ky_indices=tuple(args.training_selected_ky_indices),
        holdout_surface_indices=tuple(args.holdout_surface_indices),
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
    else:
        paths = write_vmec_boozer_aggregate_surface_holdout_artifacts(payload, out=args.out)
        print(json.dumps(paths, indent=2, sort_keys=True))
    if args.fail_on_blocked and not bool(payload.get("passed")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
