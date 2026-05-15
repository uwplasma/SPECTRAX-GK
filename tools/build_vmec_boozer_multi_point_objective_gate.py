#!/usr/bin/env python3
"""Build a bounded multi-alpha/surface VMEC/Boozer objective gate artifact."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.solver_objective_gradients import (  # noqa: E402
    vmec_boozer_aggregate_scalar_objective_finite_difference_report,
)
from tools.build_solver_objective_gradient_gate import _json_clean  # noqa: E402
from tools.build_vmec_boozer_aggregate_objective_gate import (  # noqa: E402
    write_vmec_boozer_aggregate_objective_artifacts,
)

DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_multi_point_objective_gate.png"
DEFAULT_ALPHAS = (0.0, 0.5)
DEFAULT_SELECTED_KY_INDICES = (1, 2)
DEFAULT_MAX_SAMPLES = 8
DEFAULT_MAX_WALL_SECONDS = 300.0


def _surface_indices(raw: Sequence[int | None] | None) -> tuple[int | None, ...]:
    if raw is None or len(raw) == 0:
        return (None,)
    return tuple(None if item is None else int(item) for item in raw)


def _float_tuple(raw: Sequence[float] | None, *, name: str) -> tuple[float, ...]:
    if raw is None or len(raw) == 0:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(float(item) for item in raw)


def _int_tuple(raw: Sequence[int] | None, *, name: str) -> tuple[int, ...]:
    if raw is None or len(raw) == 0:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(int(item) for item in raw)


def _unique_count(values: Sequence[object]) -> int:
    return len(set(values))


def _validate_multi_alpha_or_surface(
    surfaces: tuple[int | None, ...],
    alphas: tuple[float, ...],
) -> None:
    if _unique_count(surfaces) >= 2 or _unique_count(alphas) >= 2:
        return
    raise ValueError(
        "multi-point VMEC/Boozer gate requires at least two alphas or two "
        "surface indices"
    )


def _sample_count(
    surfaces: tuple[int | None, ...],
    alphas: tuple[float, ...],
    selected_ky_indices: tuple[int, ...],
) -> int:
    return len(surfaces) * len(alphas) * len(selected_ky_indices)


def _validate_sample_bound(sample_count: int, *, max_samples: int) -> None:
    if int(max_samples) < 1:
        raise ValueError("max_samples must be at least 1")
    if int(sample_count) > int(max_samples):
        raise ValueError(
            f"requested {sample_count} aggregate samples, exceeding "
            f"--max-samples={int(max_samples)}"
        )


@contextmanager
def _wall_time_limit(seconds: float) -> Iterator[None]:
    """Bound artifact generation on Unix-like CI hosts."""

    seconds_float = float(seconds)
    if seconds_float <= 0.0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(
            f"VMEC/Boozer multi-point objective gate exceeded {seconds_float:g} s"
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


def _annotate_payload(
    payload: dict[str, object],
    *,
    surfaces: tuple[int | None, ...],
    alphas: tuple[float, ...],
    selected_ky_indices: tuple[int, ...],
    max_samples: int,
    max_wall_seconds: float,
    elapsed_wall_seconds: float,
) -> dict[str, object]:
    sample_count = _sample_count(surfaces, alphas, selected_ky_indices)
    multi_alpha_or_surface = _unique_count(surfaces) >= 2 or _unique_count(alphas) >= 2
    annotated = dict(payload)
    annotated["artifact_kind"] = "vmec_boozer_multi_point_objective_gate"
    annotated["builder"] = "tools/build_vmec_boozer_multi_point_objective_gate.py"
    annotated["claim_scope"] = (
        "bounded finite-difference sensitivity of a reduced linear/quasilinear "
        "VMEC/Boozer/SPECTRAX-GK objective over multiple fixed field-line "
        "alphas and/or surfaces; not a nonlinear turbulent transport claim"
    )
    annotated["multi_point_coverage"] = {
        "surface_indices": [None if item is None else int(item) for item in surfaces],
        "alphas": [float(item) for item in alphas],
        "selected_ky_indices": [int(item) for item in selected_ky_indices],
        "n_surfaces": len(surfaces),
        "n_alphas": len(alphas),
        "n_selected_ky": len(selected_ky_indices),
        "n_samples_requested": sample_count,
        "multi_alpha_or_surface": bool(multi_alpha_or_surface),
    }
    annotated["bounded_runtime"] = {
        "max_samples": int(max_samples),
        "max_wall_seconds": float(max_wall_seconds),
        "elapsed_wall_seconds": float(elapsed_wall_seconds),
    }
    annotated["passed"] = bool(annotated.get("passed", False)) and bool(
        multi_alpha_or_surface
    )
    annotated["next_action"] = (
        "Use this artifact only for reduced linear/quasilinear objective "
        "plumbing across fixed VMEC/Boozer field-line or surface samples. "
        "Nonlinear transport optimization still requires separate long-window "
        "nonlinear gates."
    )
    return annotated


def build_vmec_boozer_multi_point_objective_payload(
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: str = "quasilinear_flux",
    reduction: str = "mean",
    surface_indices: Sequence[int | None] | None = None,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    selected_ky_indices: Sequence[int] = DEFAULT_SELECTED_KY_INDICES,
    radial_index: int | None = None,
    mode_index: int = 1,
    perturbation_step: float = 1.0e-7,
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
    """Build a bounded multi-alpha/surface aggregate objective payload."""

    surfaces = _surface_indices(surface_indices)
    alpha_values = _float_tuple(alphas, name="alphas")
    ky_indices = _int_tuple(selected_ky_indices, name="selected_ky_indices")
    _validate_multi_alpha_or_surface(surfaces, alpha_values)
    sample_count = _sample_count(surfaces, alpha_values, ky_indices)
    _validate_sample_bound(sample_count, max_samples=int(max_samples))

    start = time.perf_counter()
    with _wall_time_limit(float(max_wall_seconds)):
        payload = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
            case_name=case_name,
            objective=objective,
            reduction=reduction,
            surface_indices=surfaces,
            alphas=alpha_values,
            selected_ky_indices=ky_indices,
            radial_index=radial_index,
            mode_index=mode_index,
            perturbation_step=perturbation_step,
            response_atol=response_atol,
            max_curvature_ratio=max_curvature_ratio,
            ntheta=ntheta,
            mboz=mboz,
            nboz=nboz,
            surface_stencil_width=surface_stencil_width,
            n_laguerre=n_laguerre,
            n_hermite=n_hermite,
            nx=nx,
            ny=ny,
        )
    elapsed = time.perf_counter() - start
    return _annotate_payload(
        payload,
        surfaces=surfaces,
        alphas=alpha_values,
        selected_ky_indices=ky_indices,
        max_samples=int(max_samples),
        max_wall_seconds=float(max_wall_seconds),
        elapsed_wall_seconds=elapsed,
    )


def write_vmec_boozer_multi_point_objective_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the multi-point gate payload."""

    return write_vmec_boozer_aggregate_objective_artifacts(payload, out=out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument(
        "--objective",
        choices=[
            "growth",
            "gamma",
            "frequency",
            "omega",
            "kperp_eff2",
            "linear_heat_flux_weight",
            "linear_particle_flux_weight",
            "quasilinear_flux",
            "mixing_length_heat_flux_proxy",
        ],
        default="quasilinear_flux",
    )
    parser.add_argument(
        "--reduction",
        choices=["mean", "weighted_mean", "max"],
        default="mean",
    )
    parser.add_argument("--surface-indices", nargs="*", type=int, default=[])
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument(
        "--selected-ky-indices",
        nargs="+",
        type=int,
        default=list(DEFAULT_SELECTED_KY_INDICES),
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
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=DEFAULT_MAX_WALL_SECONDS,
        help="Set <=0 to disable the Unix wall-clock timeout.",
    )
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = build_vmec_boozer_multi_point_objective_payload(
            case_name=args.case_name,
            objective=args.objective,
            reduction=args.reduction,
            surface_indices=_surface_indices(args.surface_indices),
            alphas=tuple(args.alphas),
            selected_ky_indices=tuple(args.selected_ky_indices),
            radial_index=args.radial_index,
            mode_index=args.mode_index,
            perturbation_step=args.perturbation_step,
            response_atol=args.response_atol,
            max_curvature_ratio=args.max_curvature_ratio,
            ntheta=args.ntheta,
            mboz=args.mboz,
            nboz=args.nboz,
            surface_stencil_width=(
                None if args.surface_stencil_width <= 0 else args.surface_stencil_width
            ),
            n_laguerre=args.n_laguerre,
            n_hermite=args.n_hermite,
            nx=args.nx,
            ny=args.ny,
            max_samples=args.max_samples,
            max_wall_seconds=args.max_wall_seconds,
        )
    except (TimeoutError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_vmec_boozer_multi_point_objective_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
