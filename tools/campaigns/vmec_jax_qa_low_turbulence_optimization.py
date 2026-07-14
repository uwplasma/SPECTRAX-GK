#!/usr/bin/env python3
"""Run a configurable current-VMEC-JAX QA plus SPECTRAX-GK optimization.

The public examples remain the pedagogical entry point. This campaign driver
adds command-line configuration and machine-readable outputs without wrapping
or emulating VMEC-JAX's former optimizer API. Growth objectives can use the
implicit equilibrium Jacobian; eigenvector-weighted objectives use finite
differences and remain screening diagnostics until long nonlinear audits pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_SURFACES = (0.45, 0.64, 0.78)
DEFAULT_ALPHAS = (0.0, float(np.pi / 4.0))
DEFAULT_KY_VALUES = (0.10, 0.30, 0.50)


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in raw.split(",") if item.strip())
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("expected finite comma-separated values")
    return values


def _int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in raw.split(",") if item.strip())
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("mode schedule must contain positive integers")
    return values


def _default_input() -> Path:
    import vmec_jax as vj

    return (
        Path(vj.__file__).resolve().parents[1] / "examples/data/input.minimal_seed_nfp2"
    )


def _surface_index(surface: float, ns: int) -> int:
    if int(ns) < 5:
        raise ValueError("VMEC state needs at least five radial surfaces")
    if not 0.0 < float(surface) < 1.0:
        raise ValueError("transport surfaces must lie strictly inside (0, 1)")
    return int(np.clip(round(float(surface) * (int(ns) - 1)), 2, int(ns) - 2))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("output_QA_transport"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--constraints-only", action="store_true")
    parser.add_argument(
        "--transport-kind",
        choices=("growth", "quasilinear_flux", "nonlinear_window_heat_flux"),
        default="growth",
    )
    parser.add_argument("--surfaces", type=_float_tuple, default=DEFAULT_SURFACES)
    parser.add_argument("--alphas", type=_float_tuple, default=DEFAULT_ALPHAS)
    parser.add_argument("--ky-values", type=_float_tuple, default=DEFAULT_KY_VALUES)
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--r-over-lt", type=float, default=6.9)
    parser.add_argument("--r-over-ln", type=float, default=2.2)
    parser.add_argument("--transport-weight", type=float, default=0.01)
    parser.add_argument(
        "--objective-transform", choices=("raw", "scaled", "log1p"), default="log1p"
    )
    parser.add_argument("--objective-scale", type=float, default=1.0)
    parser.add_argument("--target-aspect", type=float, default=6.0)
    parser.add_argument("--target-iota", type=float, default=0.42)
    parser.add_argument("--aspect-weight", type=float, default=1.0)
    parser.add_argument("--iota-weight", type=float, default=10.0)
    parser.add_argument("--qs-weight", type=float, default=1.0)
    parser.add_argument("--seed-perturbation", type=float, default=0.01)
    parser.add_argument("--mode-schedule", type=_int_tuple, default=(1, 2, 3, 4, 5))
    parser.add_argument("--max-nfev", type=int, default=2000)
    parser.add_argument("--ftol", type=float, default=1.0e-6)
    parser.add_argument(
        "--jacobian",
        choices=("auto", "implicit", "finite-difference"),
        default="auto",
    )
    parser.add_argument("--disable-ess", action="store_true")
    parser.add_argument("--solver-device", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--make-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.input is None:
        args.input = _default_input()
    if int(args.ntheta) < 8:
        parser.error("--ntheta must be at least 8")
    if int(args.n_laguerre) < 1 or int(args.n_hermite) < 1:
        parser.error("velocity-space resolutions must be positive")
    if float(args.transport_weight) < 0.0:
        parser.error("--transport-weight must be non-negative")
    if not np.isfinite(args.objective_scale) or float(args.objective_scale) <= 0.0:
        parser.error("--objective-scale must be finite and positive")
    if not np.isfinite(args.ftol) or float(args.ftol) <= 0.0:
        parser.error("--ftol must be finite and positive")
    if int(args.max_nfev) < 1:
        parser.error("--max-nfev must be positive")
    if not np.isfinite(args.seed_perturbation):
        parser.error("--seed-perturbation must be finite")
    if args.jacobian == "implicit" and args.transport_kind != "growth":
        parser.error("implicit Jacobians are currently supported only for growth")
    return args


def _objective_transform(value: jnp.ndarray, kind: str, scale: float) -> jnp.ndarray:
    scaled = value / jnp.asarray(scale, dtype=value.dtype)
    if kind == "raw":
        return value
    if kind == "scaled":
        return scaled
    return jnp.log1p(jnp.maximum(scaled, 0.0))


def _transport_objective(args: argparse.Namespace):
    from vmec_jax.core import turbulence

    def objective(state: Any, runtime: Any) -> jnp.ndarray:
        ns = int(state.R_cos.shape[0])
        values = []
        for surface in args.surfaces:
            for alpha in args.alphas:
                for ky in args.ky_values:
                    options = {
                        "s_index": _surface_index(surface, ns),
                        "alpha": float(alpha),
                        "ntheta": int(args.ntheta),
                        "selected_ky_index": 1,
                        "n_laguerre": int(args.n_laguerre),
                        "n_hermite": int(args.n_hermite),
                        "ly": 2.0 * np.pi / float(ky),
                        "r_over_lt": float(args.r_over_lt),
                        "r_over_ln": float(args.r_over_ln),
                    }
                    if args.transport_kind == "growth":
                        value = turbulence.turbulent_growth_rate(
                            state, runtime, **options
                        )
                    elif args.transport_kind == "quasilinear_flux":
                        value = turbulence.quasilinear_flux_proxy(
                            state, runtime, **options
                        )
                    else:
                        value = turbulence.nonlinear_heat_flux_proxy(
                            state, runtime, **options
                        )
                    values.append(value)
        mean = jnp.mean(jnp.stack(values))
        return _objective_transform(
            mean, args.objective_transform, args.objective_scale
        )

    return objective


def _summary(args: argparse.Namespace, *, jacobian: str | None) -> dict[str, Any]:
    objectives = ["quasisymmetry", "aspect_ratio", "mean_iota"]
    if not args.constraints_only:
        objectives.append(args.transport_kind)
    return {
        "kind": "qa_transport_optimization_setup",
        "api": "current_vmec_jax_opt_least_squares",
        "input": str(args.input),
        "outdir": str(args.outdir),
        "objectives": objectives,
        "transport_kind": None if args.constraints_only else args.transport_kind,
        "sample_set": {
            "surfaces": list(args.surfaces),
            "alphas": list(args.alphas),
            "ky_values": list(args.ky_values),
            "n_samples": len(args.surfaces) * len(args.alphas) * len(args.ky_values),
        },
        "targets": {"aspect": args.target_aspect, "mean_iota": args.target_iota},
        "mode_schedule": list(args.mode_schedule),
        "jacobian": jacobian,
        "max_nfev_per_stage": args.max_nfev,
        "claim_scope": (
            "optimizer output is screening evidence; nonlinear transport reduction requires "
            "matched converged replicated post-transient audits"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    jacobian = (
        "implicit"
        if args.jacobian == "auto" and args.transport_kind == "growth"
        else None
        if args.jacobian in ("auto", "finite-difference")
        else "implicit"
    )
    summary = _summary(args, jacobian=jacobian)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "setup_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0

    import vmec_jax as vj
    from vmec_jax import optimize as opt

    inp = vj.VmecInput.from_file(args.input)
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor + 1, 1] += float(args.seed_perturbation)
    zbs[inp.ntor + 1, 1] += float(args.seed_perturbation)
    inp = dataclasses.replace(inp, rbc=rbc, zbs=zbs)
    initial = opt.solve_equilibrium(inp)
    qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), 1, 0)
    transport = _transport_objective(args)
    objective_terms = [
        (qs, 0.0, float(args.qs_weight)),
        (opt.aspect_ratio, float(args.target_aspect), float(args.aspect_weight)),
        (opt.mean_iota, float(args.target_iota), float(args.iota_weight)),
    ]
    if not args.constraints_only:
        objective_terms.append((transport, 0.0, float(args.transport_weight)))

    stages = []
    for mode in args.mode_schedule:
        result = opt.least_squares(
            objective_terms,
            inp,
            max_mode=int(mode),
            jac=jacobian,
            use_ess=not args.disable_ess,
            verbose=1,
            max_nfev=int(args.max_nfev),
            ftol=float(args.ftol),
            xtol=1.0e-10,
            solve_kwargs={}
            if args.solver_device is None
            else {"device": args.solver_device},
        )
        inp = result.input
        stages.append(
            {
                "max_mode": int(mode),
                "cost": float(result.cost),
                "nfev": int(result.nfev),
            }
        )

    final = result.equilibrium or opt.solve_equilibrium(inp)
    initial_transport = (
        None
        if args.constraints_only
        else float(transport(initial.state, initial.runtime))
    )
    final_transport = (
        None if args.constraints_only else float(transport(final.state, final.runtime))
    )
    history = {
        "stages": stages,
        "aspect_initial": float(opt.aspect_ratio(initial.state, initial.runtime)),
        "aspect_final": float(opt.aspect_ratio(final.state, final.runtime)),
        "iota_initial": float(opt.mean_iota(initial.state, initial.runtime)),
        "iota_final": float(opt.mean_iota(final.state, final.runtime)),
        "qs_initial": float(qs.total(initial)),
        "qs_final": float(qs.total(final)),
        "transport_objective_initial": initial_transport,
        "transport_objective_final": final_transport,
        "transport_kind": summary["transport_kind"],
    }
    (args.outdir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    inp.to_indata(args.outdir / "input.final")
    initial_wout = vj.write_wout(args.outdir / "wout_initial.nc", initial.wout)
    final_wout = vj.write_wout(args.outdir / "wout_final.nc", final.wout)
    if args.make_plots:
        vj.plot_wout(final_wout, args.outdir)
    print(json.dumps(history, indent=2))
    print(f"wrote {initial_wout}\nwrote {final_wout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
