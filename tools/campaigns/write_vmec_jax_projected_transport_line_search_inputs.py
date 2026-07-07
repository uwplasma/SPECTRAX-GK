#!/usr/bin/env python3
"""Write VMEC input decks for a projected transport-gradient line search."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import cast

import numpy as np

from spectraxgk.objectives.stellarator import StellaratorITGSampleSet
from spectraxgk.objectives.vmec_transport_admission import (
    VMECJAXNonlinearAuditPolicy,
)
from spectraxgk.objectives.vmec_transport_admission import (
    transport_objective_sample_summary,
)
from spectraxgk.objectives.vmec_transport_line_search import (
    projected_line_search_input_manifest,
    sparse_descent_direction_from_gradient_report,
)
from spectraxgk.objectives.vmec_transport import (
    VMECJAXSpectraxTransportObjective,
    VMECJAXTransportObjectiveConfig,
    VMECJAXTransportObjectiveKind,
    VMECJAXTransportObjectiveTransform,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_POLICY = VMECJAXNonlinearAuditPolicy()
DEFAULT_TRANSPORT_SURFACES = DEFAULT_AUDIT_POLICY.recommended_surfaces
DEFAULT_TRANSPORT_ALPHAS = DEFAULT_AUDIT_POLICY.recommended_alphas
DEFAULT_TRANSPORT_KY_VALUES = DEFAULT_AUDIT_POLICY.recommended_ky_values


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Solved VMEC input deck"
    )
    parser.add_argument(
        "--gradient-json",
        type=Path,
        required=True,
        help="Transport-gradient diagnostic JSON",
    )
    parser.add_argument(
        "--boundary-chain-collection-json",
        type=Path,
        default=None,
        help=(
            "Optional VMEC-JAX boundary-chain collection JSON. When supplied, "
            "projected directions use only coefficients that pass frozen-axis "
            "replay and exact finite-difference agreement."
        ),
    )
    parser.add_argument(
        "--allow-boundary-chain-branch-sensitive",
        action="store_true",
        help=(
            "Diagnostic only: admit internally transposed but exact-FD branch-sensitive "
            "coefficients from the boundary-chain collection."
        ),
    )
    parser.add_argument(
        "--allow-ungated-boundary-chain",
        action="store_true",
        help="Diagnostic only: generate projected inputs without a boundary-chain collection gate.",
    )
    parser.add_argument(
        "--require-growth-branch-locality",
        action="store_true",
        help=(
            "Require every admitted coefficient to have an explicit passing "
            "SPECTRAX growth-branch locality block in the boundary-chain collection."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory for generated candidate inputs",
    )
    parser.add_argument(
        "--steps", type=_float_tuple, default=(2.5e-4, 5.0e-4, 1.0e-3, 2.0e-3)
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of ranked gradient components to use",
    )
    parser.add_argument("--max-mode", type=int, default=5)
    parser.add_argument("--min-vmec-mode", type=int, default=7)
    parser.add_argument(
        "--surfaces", type=_float_tuple, default=DEFAULT_TRANSPORT_SURFACES
    )
    parser.add_argument("--alphas", type=_float_tuple, default=DEFAULT_TRANSPORT_ALPHAS)
    parser.add_argument(
        "--ky-values", type=_float_tuple, default=DEFAULT_TRANSPORT_KY_VALUES
    )
    parser.add_argument(
        "--allow-underresolved-sample-set",
        action="store_true",
        help="Permit exploratory single-point projected searches; production admission fails closed by default",
    )
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
        "--surface-chunk-size",
        type=int,
        default=0,
        help=(
            "Chunk the transport residual by surfaces in the generated VMEC-JAX "
            "stage. This is useful for diagnostics/evaluation, but it does not "
            "by itself make the full VMEC-JAX optimizer memory-safe."
        ),
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
    parser.add_argument("--python-executable", default="python3")
    parser.add_argument("--save-rerun-wouts", action="store_true")
    parser.add_argument("--require-rerun-wout-gate", action="store_true")
    parser.add_argument("--admit-authoritative-rerun-wout", action="store_true")
    parser.add_argument("--target-aspect", type=float, default=6.0)
    parser.add_argument("--min-iota", type=float, default=0.41)
    parser.add_argument(
        "--iota-objective", choices=("target", "floor"), default="floor"
    )
    parser.add_argument("--iota-profile-floor", type=float, default=0.41)
    parser.add_argument(
        "--disable-iota-profile-floor",
        action="store_true",
        help="Forward the upstream MeanIota-only convention to replay commands.",
    )
    parser.add_argument("--solved-wout-gate-aspect-atol", type=float, default=5.0e-2)
    parser.add_argument("--solved-wout-gate-min-abs-iota", type=float, default=None)
    parser.add_argument("--solved-wout-gate-qs-max", type=float, default=5.0e-2)
    parser.add_argument("--solved-wout-gate-profile-floor", type=float, default=None)
    return parser.parse_args(argv)


def _sample_set_from_args(args: argparse.Namespace) -> StellaratorITGSampleSet:
    return StellaratorITGSampleSet(
        surfaces=tuple(float(x) for x in args.surfaces),
        alphas=tuple(float(x) for x in args.alphas),
        ky_values=tuple(float(x) for x in args.ky_values),
    )


def _build_stage(args: argparse.Namespace):
    vj = importlib.import_module("vmec_jax")
    build_stage = getattr(vj, "build_fixed_boundary_objective_stage", None)
    if build_stage is None:
        workflow = importlib.import_module("vmec_jax.optimization_workflow")
        build_stage = getattr(workflow, "build_fixed_boundary_objective_stage")

    config = VMECJAXTransportObjectiveConfig(
        kind=cast(VMECJAXTransportObjectiveKind, str(args.transport_kind)),
        sample_set=_sample_set_from_args(args),
        ntheta=int(args.ntheta),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        n_laguerre=int(args.n_laguerre),
        n_hermite=int(args.n_hermite),
        objective_transform=cast(
            VMECJAXTransportObjectiveTransform, str(args.spectrax_objective_transform)
        ),
        objective_scale=float(args.spectrax_objective_scale),
        surface_chunk_size=int(args.surface_chunk_size),
    )
    transport = VMECJAXSpectraxTransportObjective(config=config)
    vmec = vj.FixedBoundaryVMEC.from_input(
        args.input,
        max_mode=int(args.max_mode),
        min_vmec_mode=int(args.min_vmec_mode),
        output_dir=args.outdir,
    )
    problem = vj.LeastSquaresProblem.from_tuples(
        [(transport.J, 0.0, float(args.transport_weight))]
    )
    return build_stage(
        vmec.cfg,
        vmec.indata,
        stage_mode=int(args.max_mode),
        objectives=problem.objective_terms,
        include=vmec.include,
        fix=vmec.fix,
        project_input_boundary_to_max_mode=bool(
            vmec.project_input_boundary_to_max_mode
        ),
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
        solver_device=args.solver_device,
    )


def _label_for_step(step: float) -> str:
    return f"step_{step:.1e}".replace(".", "p").replace("-", "m")


def _replay_command(
    args: argparse.Namespace, input_path: Path, outdir: Path
) -> list[str]:
    command = [
        str(args.python_executable),
        "tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py",
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
        "--ntheta",
        str(int(args.ntheta)),
        "--n-laguerre",
        str(int(args.n_laguerre)),
        "--n-hermite",
        str(int(args.n_hermite)),
        "--surface-chunk-size",
        str(int(args.surface_chunk_size)),
        "--transport-kind",
        str(args.transport_kind),
        "--spectrax-objective-transform",
        str(args.spectrax_objective_transform),
        "--spectrax-objective-scale",
        str(float(args.spectrax_objective_scale)),
        "--surfaces",
        ",".join(str(float(x)) for x in args.surfaces),
        "--alphas",
        ",".join(str(float(x)) for x in args.alphas),
        "--ky-values",
        ",".join(str(float(x)) for x in args.ky_values),
        "--iota-objective",
        str(args.iota_objective),
        "--target-aspect",
        str(float(args.target_aspect)),
        "--min-iota",
        str(float(args.min_iota)),
        "--solved-wout-gate-aspect-atol",
        str(float(args.solved_wout_gate_aspect_atol)),
        "--solved-wout-gate-min-abs-iota",
        str(
            float(args.solved_wout_gate_min_abs_iota)
            if args.solved_wout_gate_min_abs_iota is not None
            else float(args.min_iota)
        ),
        "--solved-wout-gate-qs-max",
        str(float(args.solved_wout_gate_qs_max)),
        "--max-nfev",
        "1",
        "--save-final-outputs",
        "--allow-failed-solved-wout-gate",
    ]
    if bool(args.disable_iota_profile_floor):
        command.append("--disable-iota-profile-floor")
    else:
        command.extend(["--iota-profile-floor", str(float(args.iota_profile_floor))])
        if args.solved_wout_gate_profile_floor is not None:
            command.extend(
                [
                    "--solved-wout-gate-profile-floor",
                    str(float(args.solved_wout_gate_profile_floor)),
                ]
            )
    if args.solver_device is not None:
        command.extend(["--solver-device", str(args.solver_device)])
    if bool(args.save_rerun_wouts):
        command.append("--save-rerun-wouts")
    if bool(args.require_rerun_wout_gate):
        command.append("--require-rerun-wout-gate")
    if bool(args.admit_authoritative_rerun_wout):
        command.append("--admit-authoritative-rerun-wout")
    return command


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = json.loads(args.gradient_json.read_text(encoding="utf-8"))
    boundary_chain_collection = (
        None
        if args.boundary_chain_collection_json is None
        else json.loads(args.boundary_chain_collection_json.read_text(encoding="utf-8"))
    )
    if boundary_chain_collection is None and not bool(
        args.allow_ungated_boundary_chain
    ):
        raise ValueError(
            "projected VMEC boundary updates require --boundary-chain-collection-json; "
            "use --allow-ungated-boundary-chain only for diagnostic replay"
        )
    require_boundary_chain_exact_fd = not bool(
        args.allow_boundary_chain_branch_sensitive
    )
    direction = sparse_descent_direction_from_gradient_report(
        report,
        top_n=int(args.top_n),
        boundary_chain_collection=boundary_chain_collection,
        require_boundary_chain_exact_fd=require_boundary_chain_exact_fd,
        require_growth_branch_locality=bool(args.require_growth_branch_locality),
    )
    manifest = projected_line_search_input_manifest(
        report,
        steps=tuple(float(x) for x in args.steps),
        top_n=int(args.top_n),
        boundary_chain_collection=boundary_chain_collection,
        require_boundary_chain_exact_fd=require_boundary_chain_exact_fd,
        require_growth_branch_locality=bool(args.require_growth_branch_locality),
    )
    sample_set = _sample_set_from_args(args)
    sample_summary = transport_objective_sample_summary(sample_set)
    if not bool(sample_summary["passed"]) and not bool(
        args.allow_underresolved_sample_set
    ):
        raise ValueError(
            "under-resolved transport objective sample set; use the default multi-sample set "
            "or pass --allow-underresolved-sample-set for exploratory diagnostics"
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
        stage.optimizer.save_input(
            input_path, step * np.asarray(direction, dtype=float)
        )
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
    if args.boundary_chain_collection_json is not None:
        manifest["boundary_chain_collection_json"] = str(
            args.boundary_chain_collection_json
        )
    manifest["transport_objective_sample_set"] = sample_set.to_dict()
    manifest["objective_sample_summary"] = sample_summary
    manifest["nonlinear_audit_policy"] = DEFAULT_AUDIT_POLICY.to_dict()
    manifest["rows"] = rows
    manifest_path = args.outdir / "projected_line_search_inputs.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
