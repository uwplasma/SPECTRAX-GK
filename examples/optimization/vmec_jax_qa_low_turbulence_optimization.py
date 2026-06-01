#!/usr/bin/env python
"""Run a VMEC-JAX QA optimization with a SPECTRAX-GK ITG transport residual.

This script mirrors the ``vmec_jax`` ``QA_optimization.py`` workflow: VMEC-JAX
controls the fixed-boundary equilibrium, quasisymmetry, aspect ratio, and mean
iota target, while SPECTRAX-GK supplies an extra transport objective evaluated from
the in-memory VMEC/Boozer flux-tube bridge.

The default transport residual is a trace-safe reduced nonlinear-window
objective built from SPECTRAX-GK linear ITG rows. Growth-only runs use
eigenvalue AD; quasilinear and nonlinear-window runs combine that solver
growth rate with differentiable geometry-level transport weights. Long
post-transient nonlinear SPECTRAX-GK transport audits should still be run on
the resulting candidate before making a production turbulent-flux claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectraxgk import (  # noqa: E402
    StellaratorITGSampleSet,
    VMECJAXSpectraxTransportObjective,
    VMECJAXTransportObjectiveConfig,
)


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _default_input_file() -> Path:
    try:
        import vmec_jax as vj

        package_root = Path(vj.__file__).resolve().parents[1]
        candidate = package_root / "examples" / "data" / "input.nfp2_QA_omnigenity"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    local = ROOT / "examples" / "vmec" / "input.LandremanPaul2021_QA_lowres"
    if local.exists():
        return local
    return local


class SignedIotaProfileFloor:
    """Smooth one-sided floor for the solved VMEC rotational-transform profile."""

    name = "iota_profile_floor"

    def __init__(self, floor: float, *, softness: float = 1.0e-3):
        self.floor = float(floor)
        self.softness = float(softness)

    def J(self, ctx, state):
        import vmec_jax as vj

        _chips, iotas, _iotaf = vj.equilibrium_iota_profiles_from_state(
            state=state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.signgs,
        )
        iotas = jnp.asarray(iotas, dtype=jnp.float64)
        profile = iotas[1:] if int(iotas.shape[0]) > 1 else iotas
        softness = jnp.maximum(jnp.asarray(self.softness, dtype=profile.dtype), jnp.asarray(1.0e-12))
        shortfall = (jnp.asarray(self.floor, dtype=profile.dtype) - profile) / softness
        residual = softness * jax_softplus(shortfall)
        return jnp.sqrt(jnp.mean(jnp.square(residual)))

    def to_objective_term(self, *, target, residual_weight: float):
        del target
        import vmec_jax as vj

        return vj.ObjectiveTerm(
            self.name,
            self.J,
            target=0.0,
            weight=residual_weight,
            track_iota=True,
            metadata={"iota_profile_floor": self.floor},
        )


def jax_softplus(x):
    """Stable softplus without importing ``jax.nn`` at module import time."""

    return jnp.logaddexp(x, jnp.asarray(0.0, dtype=jnp.asarray(x).dtype))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_default_input_file(), help="VMEC input deck")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "_static" / "vmec_jax_qa_low_turbulence_opt",
        help="Output directory for VMEC-JAX optimization artifacts",
    )
    parser.add_argument("--max-mode", type=int, default=1, help="Active boundary max mode")
    parser.add_argument("--min-vmec-mode", type=int, default=3, help="VMEC resolution floor")
    parser.add_argument(
        "--use-simple-seed",
        action="store_true",
        help="Mirror the upstream VMEC-JAX QA example by rebuilding the input as a simple omnigeneity seed",
    )
    parser.add_argument(
        "--simple-seed-perturbation",
        type=float,
        default=1.0e-5,
        help="Small active-mode seed amplitude used with --use-simple-seed",
    )
    parser.add_argument("--target-aspect", type=float, default=6.0, help="Aspect-ratio target")
    parser.add_argument("--min-iota", type=float, default=0.41, help="Mean-iota target or floor value")
    parser.add_argument(
        "--iota-objective",
        choices=("target", "floor"),
        default="target",
        help="Use the original VMEC-JAX QA MeanIota target objective or a one-sided AbsMeanIotaFloor objective",
    )
    parser.add_argument("--aspect-weight", type=float, default=1.0)
    parser.add_argument(
        "--iota-floor-weight",
        type=float,
        default=10_000.0,
        help="Weight for the mean-iota target or absolute-mean-iota floor objective",
    )
    parser.add_argument(
        "--iota-profile-floor",
        type=float,
        default=0.41,
        help="Signed lower bound applied to the solved iota profile, excluding the axis",
    )
    parser.add_argument("--iota-profile-floor-weight", type=float, default=10_000.0)
    parser.add_argument(
        "--disable-iota-profile-floor",
        action="store_true",
        help="Disable the profile-level iota floor residual; useful only for reproducing the original MeanIota-only QA setup",
    )
    parser.add_argument("--qs-weight", type=float, default=1.0)
    parser.add_argument(
        "--constraints-only",
        action="store_true",
        help="Run only the VMEC-JAX QA/aspect/iota objective blocks, without the SPECTRAX-GK transport residual",
    )
    parser.add_argument(
        "--spectrax-weight",
        type=float,
        default=0.05,
        help="Least-squares weight for the SPECTRAX-GK transport residual",
    )
    parser.add_argument(
        "--transport-kind",
        choices=("growth", "quasilinear_flux", "nonlinear_window_heat_flux"),
        default="nonlinear_window_heat_flux",
    )
    parser.add_argument("--surfaces", type=_float_tuple, default=(0.64,))
    parser.add_argument("--alphas", type=_float_tuple, default=(0.0,))
    parser.add_argument("--ky-values", type=_float_tuple, default=(0.30,))
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--n-laguerre", type=int, default=2)
    parser.add_argument("--n-hermite", type=int, default=3)
    parser.add_argument("--max-nfev", type=int, default=70)
    parser.add_argument("--continuation-nfev", type=int, default=25)
    parser.add_argument("--inner-max-iter", type=int, default=120)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-9)
    parser.add_argument("--trial-max-iter", type=int, default=120)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-9)
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "VMEC-JAX optimizer method. Defaults to scipy for constraints-only "
            "runs and scalar_trust when the SPECTRAX-GK transport residual is enabled."
        ),
    )
    parser.add_argument(
        "--solver-device",
        choices=("cpu", "gpu"),
        default=None,
        help="Force the VMEC-JAX solve onto a specific backend; by default JAX chooses",
    )
    parser.add_argument(
        "--scipy-tr-solver",
        choices=("exact", "lsmr"),
        default="exact",
        help="SciPy trust-region solver used when --method=scipy",
    )
    parser.add_argument(
        "--scipy-lsmr-maxiter",
        type=int,
        default=None,
        help="Optional LSMR iteration cap used when --scipy-tr-solver=lsmr",
    )
    parser.add_argument("--ftol", type=float, default=1.0e-5)
    parser.add_argument("--gtol", type=float, default=1.0e-5)
    parser.add_argument("--xtol", type=float, default=1.0e-6)
    parser.add_argument("--save-stage-wouts", action="store_true", help="Write per-stage WOUT files")
    parser.add_argument("--save-final-outputs", action="store_true", help="Keep VMEC-JAX final-output side files")
    parser.add_argument("--make-plots", action="store_true", help="Generate VMEC-JAX boundary/|B|/history plots")
    parser.add_argument("--dry-run", action="store_true", help="Assemble objectives and stop before solving")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import vmec_jax as vj

    max_mode = int(args.max_mode)
    min_vmec_mode = int(args.min_vmec_mode)
    if args.use_simple_seed:
        min_vmec_mode = max(min_vmec_mode, max_mode + 2)
    use_mode_continuation = max_mode > 1 and not bool(args.use_simple_seed)

    input_file = Path(args.input)
    if args.use_simple_seed:
        input_file = vj.prepare_simple_omnigenity_seed_input(
            input_file,
            args.outdir,
            max_mode=max_mode,
            min_vmec_mode=min_vmec_mode,
            enabled=True,
            perturbation=float(args.simple_seed_perturbation),
        )

    sample_set = StellaratorITGSampleSet(
        surfaces=tuple(float(x) for x in args.surfaces),
        alphas=tuple(float(x) for x in args.alphas),
        ky_values=tuple(float(x) for x in args.ky_values),
    )
    spectrax_config = VMECJAXTransportObjectiveConfig(
        kind=args.transport_kind,
        sample_set=sample_set,
        ntheta=int(args.ntheta),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        n_laguerre=int(args.n_laguerre),
        n_hermite=int(args.n_hermite),
    )
    transport = VMECJAXSpectraxTransportObjective(config=spectrax_config)

    vmec = vj.FixedBoundaryVMEC.from_input(
        input_file,
        max_mode=max_mode,
        min_vmec_mode=min_vmec_mode,
        output_dir=args.outdir,
    )
    aspect = vj.AspectRatio()
    iota_objective = (
        vj.MeanIota()
        if args.iota_objective == "target"
        else vj.AbsMeanIotaFloor(float(args.min_iota), softness=1.0e-3)
    )
    iota_profile_floor = SignedIotaProfileFloor(float(args.iota_profile_floor), softness=1.0e-3)
    qs = vj.QuasisymmetryRatioResidual(
        helicity_m=1,
        helicity_n=0,
        surfaces=np.arange(0.0, 1.01, 0.1),
    )
    objective_tuples = [
        (aspect.J, float(args.target_aspect), float(args.aspect_weight)),
        (qs.J, 0.0, float(args.qs_weight)),
    ]
    if args.iota_objective == "target":
        objective_tuples.insert(1, (iota_objective.J, float(args.min_iota), float(args.iota_floor_weight)))
    else:
        objective_tuples.insert(1, (iota_objective.J, 0.0, float(args.iota_floor_weight)))
    if not args.disable_iota_profile_floor:
        objective_tuples.insert(2, (iota_profile_floor.J, 0.0, float(args.iota_profile_floor_weight)))
    if not args.constraints_only:
        objective_tuples.append((transport.J, 0.0, float(args.spectrax_weight)))
    problem = vj.LeastSquaresProblem.from_tuples(objective_tuples)
    optimizer_method = str(args.method or ("scipy" if args.constraints_only else "scalar_trust"))
    summary = {
        "kind": "vmec_jax_qa_low_turbulence_optimization",
        "input": str(input_file),
        "requested_input": str(args.input),
        "use_simple_seed": bool(args.use_simple_seed),
        "simple_seed_perturbation": float(args.simple_seed_perturbation),
        "max_mode": max_mode,
        "min_vmec_mode": min_vmec_mode,
        "use_mode_continuation": use_mode_continuation,
        "outdir": str(args.outdir),
        "target_aspect": float(args.target_aspect),
        "min_iota": float(args.min_iota),
        "iota_objective": str(args.iota_objective),
        "iota_profile_floor": None
        if args.disable_iota_profile_floor
        else float(args.iota_profile_floor),
        "iota_profile_floor_weight": 0.0
        if args.disable_iota_profile_floor
        else float(args.iota_profile_floor_weight),
        "spectrax_weight": float(args.spectrax_weight),
        "constraints_only": bool(args.constraints_only),
        "transport_kind": args.transport_kind,
        "sample_set": sample_set.to_dict(),
        "spectrax_config": {
            "ntheta": int(args.ntheta),
            "mboz": int(args.mboz),
            "nboz": int(args.nboz),
            "n_laguerre": int(args.n_laguerre),
            "n_hermite": int(args.n_hermite),
            "gradient_scope": spectrax_config.gradient_scope,
        },
        "optimizer": {
            "method": optimizer_method,
            "max_nfev": int(args.max_nfev),
            "continuation_nfev": int(args.continuation_nfev),
            "inner_max_iter": int(args.inner_max_iter),
            "inner_ftol": float(args.inner_ftol),
            "trial_max_iter": int(args.trial_max_iter),
            "trial_ftol": float(args.trial_ftol),
            "solver_device": args.solver_device,
            "scipy_tr_solver": args.scipy_tr_solver,
            "scipy_lsmr_maxiter": args.scipy_lsmr_maxiter,
        },
        "objectives": list(problem.objective_names),
        "claim_scope": (
            "vmec_jax fixed-boundary QA optimization"
            + (
                " without SPECTRAX-GK transport residual; "
                if args.constraints_only
                else " with SPECTRAX-GK transport residual; "
            )
            + "production nonlinear flux claims require matched long-window SPECTRAX-GK audits"
        ),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "setup_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0

    result = vj.least_squares_solve(
        vmec,
        problem,
        stage_modes=vj.qs_stage_modes(
            max_mode=int(args.max_mode),
            use_mode_continuation=use_mode_continuation,
            continuation_nfev=int(args.continuation_nfev),
        ),
        max_nfev=int(args.max_nfev),
        continuation_nfev=int(args.continuation_nfev),
        method=optimizer_method,
        ftol=float(args.ftol),
        gtol=float(args.gtol),
        xtol=float(args.xtol),
        use_ess=True,
        ess_alpha=1.2,
        label="QA optimization with SPECTRAX-GK transport residual",
        use_mode_continuation=use_mode_continuation,
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
        solver_device=args.solver_device,
        scipy_tr_solver=str(args.scipy_tr_solver),
        scipy_lsmr_maxiter=args.scipy_lsmr_maxiter,
        save_stage_inputs=True,
        save_stage_wouts=bool(args.save_stage_wouts),
        save_final_outputs=bool(args.save_final_outputs),
    )
    saved = vj.save_optimization_result(result, output_dir=args.outdir)
    print("\nFinal VMEC-JAX diagnostics:")
    print(f"  aspect ratio: {result.history['aspect_final']:.6g}")
    print(f"  mean iota:    {result.history['iota_final']:.6g}")
    print(f"  QS residual:  {result.history['qs_final']:.6e}")
    print(f"  objective:    {result.history['objective_final']:.6e}")
    print("\nFiles:")
    for name, path in saved.as_dict().items():
        print(f"  {name}: {path}")
    if args.make_plots:
        plot_paths = {
            "boundary_comparison": vj.plot_3d_boundary_comparison(
                saved.initial_wout,
                saved.final_wout,
                outdir=args.outdir,
            ),
            "lcfs_boozer_bmag": vj.plot_boozer_lcfs_bmag_comparison(
                saved.initial_wout,
                saved.final_wout,
                outdir=args.outdir,
            ),
            "objective_history": vj.plot_objective_history(
                saved.history,
                outdir=args.outdir,
            ),
        }
        print("\nPlot files:")
        for name, path in plot_paths.items():
            print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
