#!/usr/bin/env python3
"""Build a VMEC-JAX/SPECTRAX-GK boundary transport-gradient diagnostic.

This tool is meant to be run on a solved VMEC input deck, typically the
``input.final`` from a constraints-only QA optimization.  It rebuilds the same
active boundary basis, assembles a transport-only SPECTRAX-GK objective, and
reports whether the local boundary gradient is measurable before launching more
expensive transport-weight ladders or long-window nonlinear audits.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import cast

import numpy as np

from spectraxgk.stellarator_optimization import StellaratorITGSampleSet
from spectraxgk.vmec_jax_transport_gradient import (
    build_boundary_transport_gradient_report,
    write_boundary_transport_gradient_report,
)
from spectraxgk.vmec_jax_transport_objective import (
    VMECJAXSpectraxTransportObjective,
    VMECJAXTransportObjectiveConfig,
    VMECJAXTransportObjectiveKind,
    VMECJAXTransportObjectiveTransform,
)


ROOT = Path(__file__).resolve().parents[1]


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Solved VMEC input deck, e.g. input.final")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "docs" / "_static" / "vmec_jax_transport_gradient_diagnostic.json",
        help="Output JSON diagnostic path",
    )
    parser.add_argument("--max-mode", type=int, default=5, help="Active boundary max mode")
    parser.add_argument("--min-vmec-mode", type=int, default=7, help="VMEC resolution floor")
    parser.add_argument("--surfaces", type=_float_tuple, default=(0.64,))
    parser.add_argument("--alphas", type=_float_tuple, default=(0.0,))
    parser.add_argument("--ky-values", type=_float_tuple, default=(0.30,))
    parser.add_argument(
        "--transport-kind",
        choices=("growth", "quasilinear_flux", "nonlinear_window_heat_flux"),
        default="nonlinear_window_heat_flux",
    )
    parser.add_argument("--transport-weight", type=float, default=1.0)
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
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
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--sensitivity-atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--include-jacobian",
        action="store_true",
        help="Also materialize dense residual-Jacobian norms; slower than the scalar-gradient path",
    )
    parser.add_argument(
        "--require-sensitive",
        action="store_true",
        help="Exit 2 if the local transport-gradient norm is below --sensitivity-atol",
    )
    return parser.parse_args(argv)


def _build_stage(args: argparse.Namespace):
    vj = importlib.import_module("vmec_jax")
    build_stage = getattr(vj, "build_fixed_boundary_objective_stage", None)
    if build_stage is None:  # vmec_jax releases may not re-export workflow helpers.
        workflow = importlib.import_module("vmec_jax.optimization_workflow")
        build_stage = getattr(workflow, "build_fixed_boundary_objective_stage")

    sample_set = StellaratorITGSampleSet(
        surfaces=tuple(float(x) for x in args.surfaces),
        alphas=tuple(float(x) for x in args.alphas),
        ky_values=tuple(float(x) for x in args.ky_values),
    )
    config = VMECJAXTransportObjectiveConfig(
        kind=cast(VMECJAXTransportObjectiveKind, str(args.transport_kind)),
        sample_set=sample_set,
        ntheta=int(args.ntheta),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        n_laguerre=int(args.n_laguerre),
        n_hermite=int(args.n_hermite),
        objective_transform=cast(VMECJAXTransportObjectiveTransform, str(args.spectrax_objective_transform)),
        objective_scale=float(args.spectrax_objective_scale),
    )
    transport = VMECJAXSpectraxTransportObjective(config=config)
    vmec = vj.FixedBoundaryVMEC.from_input(
        args.input,
        max_mode=int(args.max_mode),
        min_vmec_mode=int(args.min_vmec_mode),
        output_dir=Path(args.out_json).parent,
    )
    problem = vj.LeastSquaresProblem.from_tuples(
        [(transport.J, 0.0, float(args.transport_weight))]
    )
    stage = build_stage(
        vmec.cfg,
        vmec.indata,
        stage_mode=int(args.max_mode),
        objectives=problem.objective_terms,
        include=vmec.include,
        fix=vmec.fix,
        project_input_boundary_to_max_mode=bool(vmec.project_input_boundary_to_max_mode),
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
        solver_device=args.solver_device,
    )
    setup = {
        "input": str(args.input),
        "max_mode": int(args.max_mode),
        "min_vmec_mode": int(args.min_vmec_mode),
        "transport_kind": str(args.transport_kind),
        "transport_weight": float(args.transport_weight),
        "sample_set": sample_set.to_dict(),
        "spectrax_config": {
            "ntheta": int(args.ntheta),
            "mboz": int(args.mboz),
            "nboz": int(args.nboz),
            "n_laguerre": int(args.n_laguerre),
            "n_hermite": int(args.n_hermite),
            "objective_transform": str(args.spectrax_objective_transform),
            "objective_scale": float(args.spectrax_objective_scale),
            "gradient_scope": config.gradient_scope,
        },
        "solver_device": args.solver_device,
    }
    return stage, setup


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stage, setup = _build_stage(args)
    params = np.zeros(len(stage.specs), dtype=float)
    report = build_boundary_transport_gradient_report(
        stage.optimizer,
        params=params,
        label="vmec_jax_transport_gradient",
        top_n=int(args.top_n),
        sensitivity_atol=float(args.sensitivity_atol),
        include_jacobian=bool(args.include_jacobian),
    )
    report["setup"] = setup
    write_boundary_transport_gradient_report(report, args.out_json)
    print(json.dumps(report, indent=2, allow_nan=False))
    if bool(args.require_sensitive) and not bool(report["transport_sensitivity_detected"]):
        return 2
    return 0 if bool(report["finite"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
