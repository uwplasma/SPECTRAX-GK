#!/usr/bin/env python3
"""Evaluate a SPECTRAX-GK VMEC-JAX transport metric without optimizing.

This tool is intentionally separate from the optimization driver.  It builds a
solved VMEC-JAX state from an input deck, evaluates one SPECTRAX-GK transport
objective on that state, and writes a history-compatible JSON sidecar.  Use it
for baseline/candidate admission bookkeeping before launching expensive matched
nonlinear audits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk import (  # noqa: E402
    StellaratorITGSampleSet,
    VMECJAXSpectraxTransportObjective,
    VMECJAXTransportObjectiveConfig,
)
from spectraxgk.vmec_jax_transport_objective import VMECJAXTransportObjectiveTransform  # noqa: E402


DEFAULT_SURFACES = (0.45, 0.64, 0.78)
DEFAULT_ALPHAS = (0.0, 0.7853981633974483)
DEFAULT_KY_VALUES = (0.10, 0.30, 0.50)


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if not all(np.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("values must be finite")
    return values


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_report(
    *,
    input_path: Path,
    max_mode: int,
    min_vmec_mode: int,
    transport_kind: str,
    sample_set: StellaratorITGSampleSet,
    config: VMECJAXTransportObjectiveConfig,
    metric: float,
    solver_device: str | None,
    inner_max_iter: int,
    inner_ftol: float,
    trial_max_iter: int,
    trial_ftol: float,
) -> dict[str, Any]:
    """Return a JSON-safe metric report."""

    return {
        "kind": "vmec_jax_spectrax_transport_metric_eval",
        "claim_scope": (
            "SPECTRAX-GK transport metric evaluated on a supplied VMEC-JAX input; "
            "this is not an optimization and does not imply nonlinear turbulent-flux promotion"
        ),
        "input": str(input_path),
        "max_mode": int(max_mode),
        "min_vmec_mode": int(min_vmec_mode),
        "transport_kind": str(transport_kind),
        "transport_metric_kind": str(transport_kind),
        "transport_objective_final": float(metric),
        "spectrax_objective_final": float(metric),
        "transport_metric_final": float(metric),
        "transport_objective_source": "vmec_jax_input_final_state_eval_only",
        "sample_set": sample_set.to_dict(),
        "spectrax_config": {
            "ntheta": int(config.ntheta),
            "mboz": int(config.mboz),
            "nboz": int(config.nboz),
            "n_laguerre": int(config.n_laguerre),
            "n_hermite": int(config.n_hermite),
            "objective_transform": str(config.objective_transform),
            "objective_scale": float(config.objective_scale),
            "surface_chunk_size": int(config.surface_chunk_size),
            "gradient_scope": config.gradient_scope,
        },
        "solver": {
            "device": solver_device,
            "inner_max_iter": int(inner_max_iter),
            "inner_ftol": float(inner_ftol),
            "trial_max_iter": int(trial_max_iter),
            "trial_ftol": float(trial_ftol),
        },
        "next_action": (
            "use this metric only for reduced-objective admission; matched long-window "
            "nonlinear audits remain required for turbulent-flux claims"
        ),
    }


def evaluate_metric(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the configured transport metric on the supplied VMEC input."""

    import vmec_jax as vj
    import vmec_jax.optimization_workflow as workflow

    sample_set = StellaratorITGSampleSet(
        surfaces=tuple(float(x) for x in args.surfaces),
        alphas=tuple(float(x) for x in args.alphas),
        ky_values=tuple(float(x) for x in args.ky_values),
    )
    config = VMECJAXTransportObjectiveConfig(
        kind=args.transport_kind,
        sample_set=sample_set,
        ntheta=int(args.ntheta),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        n_laguerre=int(args.n_laguerre),
        n_hermite=int(args.n_hermite),
        objective_transform=cast(VMECJAXTransportObjectiveTransform, str(args.spectrax_objective_transform)),
        objective_scale=float(args.spectrax_objective_scale),
        surface_chunk_size=int(args.surface_chunk_size),
    )
    objective = VMECJAXSpectraxTransportObjective(config=config)
    vmec = vj.FixedBoundaryVMEC.from_input(
        args.input,
        max_mode=int(args.max_mode),
        min_vmec_mode=int(args.min_vmec_mode),
        output_dir=args.outdir,
    )
    problem = vj.LeastSquaresProblem.from_tuples([(objective.J, 0.0, 1.0)])
    stage = workflow.build_fixed_boundary_objective_stage(
        vmec.cfg,
        vmec.indata,
        stage_mode=int(args.max_mode),
        objectives=problem.objective_terms,
        include=vmec.include,
        fix=vmec.fix,
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
        solver_device=args.solver_device,
    )
    params0 = np.zeros(len(stage.specs), dtype=float)
    # The public VMEC-JAX workflow currently exposes final states through the
    # optimizer path.  For eval-only admission bookkeeping, solve the supplied
    # boundary once and call the SPECTRAX objective directly; no boundary update
    # or least-squares step is taken.
    state = stage.optimizer._solve_forward(params0, trial=False)  # noqa: SLF001
    metric = float(np.asarray(objective.J(stage.ctx, state)))
    if not np.isfinite(metric):
        raise RuntimeError(f"non-finite SPECTRAX transport metric: {metric!r}")
    return build_report(
        input_path=Path(args.input),
        max_mode=int(args.max_mode),
        min_vmec_mode=int(args.min_vmec_mode),
        transport_kind=str(args.transport_kind),
        sample_set=sample_set,
        config=config,
        metric=metric,
        solver_device=args.solver_device,
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Solved VMEC input deck, typically input.final")
    parser.add_argument("--out-json", required=True, type=Path, help="JSON sidecar to write")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "tools_out" / "vmec_jax_transport_metric_eval",
        help="Scratch VMEC-JAX output directory used while building the state",
    )
    parser.add_argument("--max-mode", type=int, default=5)
    parser.add_argument("--min-vmec-mode", type=int, default=7)
    parser.add_argument(
        "--transport-kind",
        choices=("growth", "quasilinear_flux", "nonlinear_window_heat_flux"),
        default="growth",
    )
    parser.add_argument("--surfaces", type=_float_tuple, default=DEFAULT_SURFACES)
    parser.add_argument("--alphas", type=_float_tuple, default=DEFAULT_ALPHAS)
    parser.add_argument("--ky-values", type=_float_tuple, default=DEFAULT_KY_VALUES)
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument(
        "--surface-chunk-size",
        type=int,
        default=0,
        help="Evaluate the transport objective in surface chunks before applying the scalar transform",
    )
    parser.add_argument(
        "--spectrax-objective-transform",
        choices=("raw", "scaled", "log1p"),
        default="log1p",
    )
    parser.add_argument("--spectrax-objective-scale", type=float, default=1.0)
    parser.add_argument("--inner-max-iter", type=int, default=120)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-9)
    parser.add_argument("--trial-max-iter", type=int, default=120)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-9)
    parser.add_argument("--solver-device", choices=("cpu", "gpu"), default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_metric(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(_json_safe(report), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
