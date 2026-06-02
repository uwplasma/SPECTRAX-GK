#!/usr/bin/env python3
"""Write VMEC input decks for a projected transport-gradient line search."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import cast

import numpy as np

from spectraxgk.stellarator_optimization import StellaratorITGSampleSet
from spectraxgk.vmec_jax_transport_line_search import (
    projected_line_search_input_manifest,
    sparse_descent_direction_from_gradient_report,
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
    parser.add_argument("--input", type=Path, required=True, help="Solved VMEC input deck")
    parser.add_argument("--gradient-json", type=Path, required=True, help="Transport-gradient diagnostic JSON")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for generated candidate inputs")
    parser.add_argument("--steps", type=_float_tuple, default=(2.5e-4, 5.0e-4, 1.0e-3, 2.0e-3))
    parser.add_argument("--top-n", type=int, default=12, help="Number of ranked gradient components to use")
    parser.add_argument("--max-mode", type=int, default=5)
    parser.add_argument("--min-vmec-mode", type=int, default=7)
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
    return parser.parse_args(argv)


def _build_stage(args: argparse.Namespace):
    vj = importlib.import_module("vmec_jax")
    build_stage = getattr(vj, "build_fixed_boundary_objective_stage", None)
    if build_stage is None:
        workflow = importlib.import_module("vmec_jax.optimization_workflow")
        build_stage = getattr(workflow, "build_fixed_boundary_objective_stage")

    config = VMECJAXTransportObjectiveConfig(
        kind=cast(VMECJAXTransportObjectiveKind, str(args.transport_kind)),
        sample_set=StellaratorITGSampleSet(
            surfaces=tuple(float(x) for x in args.surfaces),
            alphas=tuple(float(x) for x in args.alphas),
            ky_values=tuple(float(x) for x in args.ky_values),
        ),
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
        output_dir=args.outdir,
    )
    problem = vj.LeastSquaresProblem.from_tuples([(transport.J, 0.0, float(args.transport_weight))])
    return build_stage(
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


def _label_for_step(step: float) -> str:
    return f"step_{step:.1e}".replace(".", "p").replace("-", "m")


def _replay_command(args: argparse.Namespace, input_path: Path, outdir: Path) -> list[str]:
    return [
        "python",
        "examples/optimization/vmec_jax_qa_low_turbulence_optimization.py",
        "--input",
        str(input_path),
        "--outdir",
        str(outdir),
        "--constraints-only",
        "--disable-mode-continuation",
        "--max-mode",
        str(int(args.max_mode)),
        "--min-vmec-mode",
        str(int(args.min_vmec_mode)),
        "--mboz",
        str(int(args.mboz)),
        "--nboz",
        str(int(args.nboz)),
        "--transport-kind",
        str(args.transport_kind),
        "--spectrax-objective-transform",
        str(args.spectrax_objective_transform),
        "--surfaces",
        ",".join(str(float(x)) for x in args.surfaces),
        "--alphas",
        ",".join(str(float(x)) for x in args.alphas),
        "--ky-values",
        ",".join(str(float(x)) for x in args.ky_values),
        "--iota-objective",
        "floor",
        "--target-aspect",
        "6.0",
        "--min-iota",
        "0.41",
        "--iota-profile-floor",
        "0.41",
        "--max-nfev",
        "1",
        "--save-final-outputs",
        "--allow-failed-solved-wout-gate",
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = json.loads(args.gradient_json.read_text(encoding="utf-8"))
    direction = sparse_descent_direction_from_gradient_report(
        report,
        top_n=int(args.top_n),
    )
    manifest = projected_line_search_input_manifest(
        report,
        steps=tuple(float(x) for x in args.steps),
        top_n=int(args.top_n),
    )
    stage = _build_stage(args)
    if len(stage.specs) != int(direction.size):
        raise ValueError(
            f"stage has {len(stage.specs)} parameters but gradient report has {direction.size}"
        )
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw_step in args.steps:
        step = float(raw_step)
        label = _label_for_step(step)
        case_dir = args.outdir / label
        solve_dir = case_dir / "solve"
        case_dir.mkdir(parents=True, exist_ok=True)
        input_path = case_dir / "input.gradient_step"
        stage.optimizer.save_input(input_path, step * np.asarray(direction, dtype=float))
        rows.append(
            {
                "label": label,
                "step": step,
                "input": str(input_path),
                "solve_dir": str(solve_dir),
                "replay_command": _replay_command(args, input_path, solve_dir),
            }
        )
    manifest["input"] = str(args.input)
    manifest["gradient_json"] = str(args.gradient_json)
    manifest["rows"] = rows
    manifest_path = args.outdir / "projected_line_search_inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
