#!/usr/bin/env python3
"""Build a second-equilibrium reduced VMEC/Boozer aggregate gate artifact."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
import signal
import sys
import time
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.solver_objective_gradients import (  # noqa: E402
    vmec_boozer_aggregate_scalar_objective_finite_difference_report,
    vmec_boozer_aggregate_scalar_objective_line_search_report,
)
from tools.build_solver_objective_gradient_gate import _json_clean  # noqa: E402
from tools.build_vmec_boozer_multi_point_objective_gate import (  # noqa: E402
    _float_tuple,
    _int_tuple,
    _sample_count,
    _surface_indices,
    _validate_sample_bound,
)

DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_second_equilibrium_aggregate_gate.png"
DEFAULT_CASE_NAME = "li383_low_res"
DEFAULT_ALPHAS = (0.0,)
DEFAULT_SELECTED_KY_INDICES = (1, 2)
DEFAULT_MAX_SAMPLES = 4
DEFAULT_MAX_WALL_SECONDS = 300.0


def _as_public_surfaces(surfaces: Sequence[int | None]) -> list[int | None]:
    return [None if item is None else int(item) for item in surfaces]


@contextmanager
def _wall_time_limit(seconds: float) -> Iterator[None]:
    """Bound optional-backend gate generation on Unix-like CI hosts."""

    seconds_float = float(seconds)
    if seconds_float <= 0.0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(
            f"second-equilibrium VMEC/Boozer aggregate gate exceeded {seconds_float:g} s"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds_float)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _mode_bound(mboz: int, nboz: int) -> dict[str, object]:
    return {
        "mboz": int(mboz),
        "nboz": int(nboz),
        "minimum_required": 21,
        "passed": bool(int(mboz) >= 21 and int(nboz) >= 21),
    }


def _fail_closed_payload(
    *,
    case_name: str,
    blocker: BaseException,
    elapsed_wall_seconds: float,
    max_wall_seconds: float,
    mboz: int,
    nboz: int,
    surfaces: Sequence[int | None],
    alphas: Sequence[float],
    selected_ky_indices: Sequence[int],
    max_samples: int,
) -> dict[str, object]:
    sample_count = _sample_count(tuple(surfaces), tuple(alphas), tuple(selected_ky_indices))
    return {
        "kind": "vmec_boozer_second_equilibrium_aggregate_gate",
        "builder": "tools/build_vmec_boozer_second_equilibrium_aggregate_gate.py",
        "passed": False,
        "feasible": False,
        "case_name": str(case_name),
        "source_scope": "mode21_vmec_boozer_state_second_equilibrium_aggregate",
        "claim_scope": (
            "second-equilibrium reduced aggregate VMEC/Boozer/SPECTRAX-GK "
            "finite-difference or line-search feasibility at mboz=nboz>=21; "
            "not a nonlinear turbulent transport optimization claim"
        ),
        "blocker_type": type(blocker).__name__,
        "blocker_message": str(blocker),
        "mode_bound": _mode_bound(mboz, nboz),
        "sample_bound": {
            "n_samples_requested": int(sample_count),
            "max_samples": int(max_samples),
            "passed": bool(int(sample_count) <= int(max_samples)),
        },
        "bounded_runtime": {
            "max_wall_seconds": float(max_wall_seconds),
            "elapsed_wall_seconds": float(elapsed_wall_seconds),
            "passed": bool(float(elapsed_wall_seconds) <= float(max_wall_seconds)),
        },
        "coverage": {
            "surface_indices": _as_public_surfaces(surfaces),
            "alphas": [float(item) for item in alphas],
            "selected_ky_indices": [int(item) for item in selected_ky_indices],
        },
        "next_action": (
            "Resolve the recorded fixture, memory, optional-backend API, or runtime "
            "blocker before using this second equilibrium in aggregate optimization gates."
        ),
    }


def build_vmec_boozer_second_equilibrium_aggregate_payload(
    *,
    case_name: str = DEFAULT_CASE_NAME,
    objective: str = "quasilinear_flux",
    reduction: str = "mean",
    surface_indices: Sequence[int | None] | None = None,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    selected_ky_indices: Sequence[int] = DEFAULT_SELECTED_KY_INDICES,
    radial_index: int | None = None,
    mode_index: int = 1,
    perturbation_step: float = 1.0e-7,
    update_step: float = 1.0e-8,
    max_steps: int = 1,
    min_improvement: float = 0.0,
    max_curvature_ratio: float = 5.0,
    response_atol: float = 0.0,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = 3,
    n_laguerre: int = 2,
    n_hermite: int = 3,
    nx: int = 1,
    ny: int = 6,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS,
) -> dict[str, object]:
    """Return a pass/fail aggregate gate payload for a non-QH equilibrium."""

    surfaces = _surface_indices(surface_indices)
    alpha_values = _float_tuple(alphas, name="alphas")
    ky_indices = _int_tuple(selected_ky_indices, name="selected_ky_indices")
    sample_count = _sample_count(surfaces, alpha_values, ky_indices)
    _validate_sample_bound(sample_count, max_samples=int(max_samples))
    mode_bound = _mode_bound(mboz, nboz)
    if not mode_bound["passed"]:
        raise ValueError("second-equilibrium aggregate gate requires mboz,nboz >= 21")

    start = time.perf_counter()
    common_kwargs: dict[str, Any] = {
        "case_name": str(case_name),
        "objective": str(objective),
        "reduction": str(reduction),
        "surface_indices": surfaces,
        "alphas": alpha_values,
        "selected_ky_indices": ky_indices,
        "radial_index": radial_index,
        "mode_index": int(mode_index),
        "perturbation_step": float(perturbation_step),
        "response_atol": float(response_atol),
        "max_curvature_ratio": float(max_curvature_ratio),
        "ntheta": int(ntheta),
        "mboz": int(mboz),
        "nboz": int(nboz),
        "surface_stencil_width": surface_stencil_width,
        "n_laguerre": int(n_laguerre),
        "n_hermite": int(n_hermite),
        "nx": int(nx),
        "ny": int(ny),
    }
    try:
        with _wall_time_limit(float(max_wall_seconds)):
            fd_report = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
                **common_kwargs,
            )
            line_search_report = vmec_boozer_aggregate_scalar_objective_line_search_report(
                **common_kwargs,
                update_step=float(update_step),
                max_steps=int(max_steps),
                min_improvement=float(min_improvement),
            )
    except Exception as exc:  # noqa: BLE001 - the artifact must fail closed with exact blocker metadata.
        elapsed = time.perf_counter() - start
        return _fail_closed_payload(
            case_name=str(case_name),
            blocker=exc,
            elapsed_wall_seconds=elapsed,
            max_wall_seconds=float(max_wall_seconds),
            mboz=int(mboz),
            nboz=int(nboz),
            surfaces=surfaces,
            alphas=alpha_values,
            selected_ky_indices=ky_indices,
            max_samples=int(max_samples),
        )

    elapsed = time.perf_counter() - start
    runtime_passed = bool(float(max_wall_seconds) <= 0.0 or elapsed <= float(max_wall_seconds))
    fd_passed = bool(fd_report.get("passed", False))
    line_search_passed = bool(line_search_report.get("passed", False))
    return {
        "kind": "vmec_boozer_second_equilibrium_aggregate_gate",
        "builder": "tools/build_vmec_boozer_second_equilibrium_aggregate_gate.py",
        "passed": bool(fd_passed and line_search_passed and runtime_passed),
        "feasible": bool(fd_passed or line_search_passed),
        "case_name": str(case_name),
        "source_scope": "mode21_vmec_boozer_state_second_equilibrium_aggregate",
        "claim_scope": (
            "second-equilibrium reduced aggregate VMEC/Boozer/SPECTRAX-GK "
            "finite-difference and one-parameter line-search gate at mboz=nboz>=21; "
            "not a nonlinear turbulent transport optimization claim"
        ),
        "objective": str(objective),
        "reduction": str(reduction),
        "coverage": {
            "surface_indices": _as_public_surfaces(surfaces),
            "alphas": [float(item) for item in alpha_values],
            "selected_ky_indices": [int(item) for item in ky_indices],
            "n_samples_requested": int(sample_count),
        },
        "mode_bound": mode_bound,
        "sample_bound": {
            "n_samples_requested": int(sample_count),
            "max_samples": int(max_samples),
            "passed": True,
        },
        "bounded_runtime": {
            "max_wall_seconds": float(max_wall_seconds),
            "elapsed_wall_seconds": float(elapsed),
            "passed": runtime_passed,
        },
        "finite_difference_passed": fd_passed,
        "line_search_passed": line_search_passed,
        "finite_difference_summary": {
            "base_value": fd_report.get("base_value"),
            "minus_value": fd_report.get("minus_value"),
            "plus_value": fd_report.get("plus_value"),
            "central_derivative": fd_report.get("central_derivative"),
            "response_abs": fd_report.get("response_abs"),
            "curvature_ratio": fd_report.get("curvature_ratio"),
            "n_samples": fd_report.get("n_samples"),
        },
        "line_search_summary": {
            "accepted_steps": line_search_report.get("accepted_steps"),
            "initial_objective": line_search_report.get("initial_objective"),
            "final_objective": line_search_report.get("final_objective"),
            "relative_reduction": line_search_report.get("relative_reduction"),
            "stop_reason": line_search_report.get("stop_reason"),
            "n_samples": line_search_report.get("n_samples"),
        },
        "finite_difference_report": fd_report,
        "line_search_report": line_search_report,
        "next_action": (
            "Use this only as a second-equilibrium reduced aggregate optimizer-plumbing "
            "gate. Promotion still requires held-out surface or field-line evidence and "
            "separate nonlinear transport-window validation."
        ),
    }


def write_vmec_boozer_second_equilibrium_aggregate_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the second-equilibrium gate."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fd = payload.get("finite_difference_summary", {})
    ls = payload.get("line_search_summary", {})
    mode_bound = payload.get("mode_bound", {})
    runtime = payload.get("bounded_runtime", {})
    sample_bound = payload.get("sample_bound", {})
    fd_dict = fd if isinstance(fd, dict) else {}
    ls_dict = ls if isinstance(ls, dict) else {}
    mode_dict = mode_bound if isinstance(mode_bound, dict) else {}
    runtime_dict = runtime if isinstance(runtime, dict) else {}
    sample_dict = sample_bound if isinstance(sample_bound, dict) else {}
    fieldnames = [
        "case_name",
        "passed",
        "feasible",
        "mboz",
        "nboz",
        "n_samples",
        "elapsed_wall_seconds",
        "fd_passed",
        "fd_base_value",
        "fd_central_derivative",
        "fd_response_abs",
        "fd_curvature_ratio",
        "line_search_passed",
        "accepted_steps",
        "initial_objective",
        "final_objective",
        "relative_reduction",
        "stop_reason",
        "blocker_type",
        "blocker_message",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "case_name": payload.get("case_name", ""),
                "passed": payload.get("passed", ""),
                "feasible": payload.get("feasible", ""),
                "mboz": mode_dict.get("mboz", ""),
                "nboz": mode_dict.get("nboz", ""),
                "n_samples": sample_dict.get("n_samples_requested", fd_dict.get("n_samples", "")),
                "elapsed_wall_seconds": runtime_dict.get("elapsed_wall_seconds", ""),
                "fd_passed": payload.get("finite_difference_passed", ""),
                "fd_base_value": fd_dict.get("base_value", ""),
                "fd_central_derivative": fd_dict.get("central_derivative", ""),
                "fd_response_abs": fd_dict.get("response_abs", ""),
                "fd_curvature_ratio": fd_dict.get("curvature_ratio", ""),
                "line_search_passed": payload.get("line_search_passed", ""),
                "accepted_steps": ls_dict.get("accepted_steps", ""),
                "initial_objective": ls_dict.get("initial_objective", ""),
                "final_objective": ls_dict.get("final_objective", ""),
                "relative_reduction": ls_dict.get("relative_reduction", ""),
                "stop_reason": ls_dict.get("stop_reason", ""),
                "blocker_type": payload.get("blocker_type", ""),
                "blocker_message": payload.get("blocker_message", ""),
            }
        )

    set_plot_style()
    fig, (ax_values, ax_meta) = plt.subplots(1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.2, 1.0]})
    passed = bool(payload.get("passed"))
    status = "passed" if passed else "blocked"
    if fd_dict:
        values = np.asarray(
            [
                float(fd_dict.get("minus_value", np.nan)),
                float(fd_dict.get("base_value", np.nan)),
                float(fd_dict.get("plus_value", np.nan)),
            ],
            dtype=float,
        )
        ax_values.bar([0, 1, 2], values, color=["#90be6d", "#277da1", "#f3722c"], edgecolor="#202020", linewidth=0.5)
        ax_values.set_xticks([0, 1, 2], ["x-h", "x", "x+h"])
        ax_values.set_ylabel(str(payload.get("objective", "objective")))
        ax_values.set_title("Aggregate finite-difference values")
        ax_values.grid(axis="y", alpha=0.25)
    else:
        ax_values.axis("off")
        ax_values.text(0.05, 0.55, str(payload.get("blocker_message", "blocked")), va="center", ha="left", wrap=True)
        ax_values.set_title("Fail-closed blocker")

    summary_lines = [
        f"status: {status}",
        f"case: {payload.get('case_name')}",
        f"mboz/nboz: {mode_dict.get('mboz')}/{mode_dict.get('nboz')}",
        f"samples: {sample_dict.get('n_samples_requested')}",
        f"elapsed: {float(runtime_dict.get('elapsed_wall_seconds', float('nan'))):.3g} s",
        f"FD passed: {payload.get('finite_difference_passed', False)}",
        f"line passed: {payload.get('line_search_passed', False)}",
        f"FD deriv: {float(fd_dict.get('central_derivative', float('nan'))):.6g}",
        f"curvature: {float(fd_dict.get('curvature_ratio', float('nan'))):.3e}",
        f"rel. red.: {ls_dict.get('relative_reduction')}",
    ]
    if not passed and payload.get("blocker_type"):
        summary_lines.extend([f"blocker: {payload.get('blocker_type')}", str(payload.get("blocker_message"))])
    ax_meta.axis("off")
    ax_meta.set_title("Second-equilibrium aggregate gate")
    ax_meta.text(0.02, 0.95, "\n".join(summary_lines), va="top", ha="left", family="monospace", fontsize=9.2, transform=ax_meta.transAxes)
    ax_meta.text(
        0.02,
        0.13,
        "Reduced VMEC/Boozer aggregate FD and line-search\n"
        "evidence on a non-QH fixture. This checks bounded\n"
        "optimizer plumbing, not nonlinear transport validity.",
        va="top",
        ha="left",
        fontsize=8.3,
        transform=ax_meta.transAxes,
    )
    fig.suptitle(f"VMEC/Boozer second-equilibrium aggregate gate: {status}")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, wspace=0.28)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path), "csv": str(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case-name", default=DEFAULT_CASE_NAME)
    parser.add_argument("--objective", default="quasilinear_flux")
    parser.add_argument("--reduction", choices=["mean", "weighted_mean", "max"], default="mean")
    parser.add_argument("--surface-indices", nargs="*", type=int, default=[])
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--selected-ky-indices", nargs="+", type=int, default=list(DEFAULT_SELECTED_KY_INDICES))
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
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--max-wall-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_vmec_boozer_second_equilibrium_aggregate_payload(
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
        surface_stencil_width=None if args.surface_stencil_width <= 0 else args.surface_stencil_width,
        n_laguerre=args.n_laguerre,
        n_hermite=args.n_hermite,
        nx=args.nx,
        ny=args.ny,
        max_samples=args.max_samples,
        max_wall_seconds=args.max_wall_seconds,
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0 if bool(payload.get("passed")) else 1
    paths = write_vmec_boozer_second_equilibrium_aggregate_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0 if bool(payload.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
