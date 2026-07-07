#!/usr/bin/env python
"""Run a VMEC-JAX QA optimization with a SPECTRAX-GK ITG transport residual.

This script mirrors the ``vmec_jax`` ``QA_optimization.py`` workflow: VMEC-JAX
controls the fixed-boundary equilibrium, quasisymmetry, aspect ratio, and mean
iota target, while SPECTRAX-GK supplies an extra transport objective evaluated from
the in-memory VMEC/Boozer flux-tube bridge.

The default transport residual is a trace-safe nonlinear-window screening
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
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spectraxgk import (  # noqa: E402
    StellaratorITGSampleSet,
    VMECJAXSpectraxTransportObjective,
    VMECJAXTransportObjectiveConfig,
)
from spectraxgk.validation.stellarator.candidate_gate import (  # noqa: E402
    build_authoritative_wout_candidate_gate,
    build_solved_vmec_candidate_gate,
    build_wout_reproducibility_gate,
)
from spectraxgk.objectives.vmec_transport import VMECJAXTransportObjectiveTransform  # noqa: E402


DEFAULT_TRANSPORT_SURFACES = (0.45, 0.64, 0.78)
DEFAULT_TRANSPORT_ALPHAS = (0.0, 0.7853981633974483)
DEFAULT_TRANSPORT_KY_VALUES = (0.10, 0.30, 0.50)


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _default_input_file() -> Path:
    try:
        import vmec_jax as vj  # type: ignore[import-not-found]

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


def _default_simple_seed_input_file() -> Path:
    try:
        import vmec_jax as vj  # type: ignore[import-not-found]

        package_root = Path(vj.__file__).resolve().parents[1]
        candidate = package_root / "examples" / "data" / "input.minimal_seed_nfp2"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    return ROOT / "examples" / "vmec" / "input.minimal_seed_nfp2"


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
        softness = jnp.maximum(
            jnp.asarray(self.softness, dtype=profile.dtype), jnp.asarray(1.0e-12)
        )
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


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(np.asarray(value))
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _transport_metric_from_result(
    transport: VMECJAXSpectraxTransportObjective,
    result: Any,
) -> dict[str, Any]:
    """Evaluate the SPECTRAX-GK transport residual on the final VMEC-JAX state."""

    optimizer = getattr(result, "final_optimizer", None)
    state = getattr(result, "final_state", None)
    if optimizer is None or state is None:
        return {
            "transport_objective_final": None,
            "transport_objective_source": "missing_final_vmec_jax_state",
        }
    try:
        static = getattr(optimizer, "_static")
        ctx = SimpleNamespace(
            static=static,
            indata=getattr(optimizer, "_indata"),
            signgs=int(getattr(optimizer, "_signgs")),
            flux=getattr(optimizer, "_flux"),
            pressure=jnp.zeros_like(jnp.asarray(getattr(static, "s"))),
        )
        value = _finite_float_or_none(transport.J(ctx, state))
        return {
            "transport_objective_final": value,
            "spectrax_objective_final": value,
            "transport_metric_final": value,
            "transport_objective_source": "final_vmec_jax_state",
            "transport_metric_kind": transport.config.kind,
            "transport_metric_transform": transport.config.objective_transform,
            "transport_metric_scale": float(transport.config.objective_scale),
        }
    except Exception as exc:
        return {
            "transport_objective_final": None,
            "transport_objective_source": "final_vmec_jax_state_error",
            "transport_objective_error": f"{type(exc).__name__}: {exc}",
        }


def _update_history_with_transport_metric(
    history_path: Path, metric: dict[str, Any]
) -> None:
    """Persist transport-only metric fields into VMEC-JAX ``history.json``."""

    if not history_path.exists():
        return
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return
    for key, value in metric.items():
        if value is not None:
            payload[key] = value
    history = payload.get("history")
    if isinstance(history, list) and history and isinstance(history[-1], dict):
        for key, value in metric.items():
            if value is not None:
                history[-1][key] = value
    history_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=_default_input_file(), help="VMEC input deck"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "_static" / "vmec_jax_qa_low_turbulence_opt",
        help="Output directory for VMEC-JAX optimization artifacts",
    )
    parser.add_argument(
        "--max-mode", type=int, default=1, help="Active boundary max mode"
    )
    parser.add_argument(
        "--min-vmec-mode", type=int, default=3, help="VMEC resolution floor"
    )
    parser.add_argument(
        "--use-simple-seed",
        action="store_true",
        help="Mirror the upstream VMEC-JAX QA example by rebuilding the input as a simple omnigeneity seed",
    )
    parser.add_argument(
        "--disable-mode-continuation",
        action="store_true",
        help="Optimize the requested max-mode branch directly when restarting from an existing VMEC input",
    )
    parser.add_argument(
        "--simple-seed-perturbation",
        type=float,
        default=1.0e-5,
        help="Small active-mode seed amplitude used with --use-simple-seed",
    )
    parser.add_argument(
        "--target-aspect", type=float, default=6.0, help="Aspect-ratio target"
    )
    parser.add_argument(
        "--min-iota", type=float, default=0.41, help="Mean-iota target or floor value"
    )
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
        "--strict-upstream-qa-baseline",
        action="store_true",
        help=(
            "Use the VMEC-JAX QA_optimization.py simple-seed max-mode-5 baseline "
            "with tighter admission tolerances for paper-facing solved-WOUT gates"
        ),
    )
    parser.add_argument(
        "--strict-iota-admission-buffer",
        type=float,
        default=2.0e-4,
        help=(
            "Small iota target buffer used only by --strict-upstream-qa-baseline; "
            "the solved-WOUT gate remains at --solved-wout-gate-min-abs-iota or 0.41"
        ),
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
    parser.add_argument(
        "--surfaces",
        type=_float_tuple,
        default=DEFAULT_TRANSPORT_SURFACES,
        help="Comma-separated normalized toroidal-flux samples for the transport residual",
    )
    parser.add_argument(
        "--alphas",
        type=_float_tuple,
        default=DEFAULT_TRANSPORT_ALPHAS,
        help="Comma-separated field-line labels for the transport residual",
    )
    parser.add_argument(
        "--ky-values",
        type=_float_tuple,
        default=DEFAULT_TRANSPORT_KY_VALUES,
        help="Comma-separated physical ky*rho_i values for the transport residual",
    )
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
            "Evaluate the SPECTRAX-GK transport residual in surface chunks before applying the scalar transform. "
            "This is useful for eval-only metrics and chunked gradient diagnostics; the full VMEC-JAX "
            "optimizer can still be limited by its final-state cotangent memory path."
        ),
    )
    parser.add_argument(
        "--spectrax-objective-transform",
        choices=("raw", "scaled", "log1p"),
        default="log1p",
        help="Transform applied to the SPECTRAX-GK transport residual before VMEC-JAX least squares",
    )
    parser.add_argument(
        "--spectrax-objective-scale",
        type=float,
        default=1.0,
        help="Positive scale used by --spectrax-objective-transform=scaled/log1p",
    )
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
    parser.add_argument(
        "--disable-ess",
        action="store_true",
        help="Disable VMEC-JAX ESS high-mode scaling; enabled by default to mirror the upstream QA example",
    )
    parser.add_argument(
        "--ess-alpha",
        type=float,
        default=1.2,
        help="VMEC-JAX ESS high-mode scaling exponent",
    )
    parser.add_argument(
        "--save-stage-wouts", action="store_true", help="Write per-stage WOUT files"
    )
    parser.add_argument(
        "--save-final-outputs",
        action="store_true",
        help="Keep VMEC-JAX final-output side files",
    )
    parser.add_argument(
        "--save-rerun-wouts",
        action="store_true",
        help="Write fresh VMEC rerun WOUTs from input.initial/input.final and gate input/WOUT reproducibility",
    )
    parser.add_argument(
        "--require-rerun-wout-gate",
        action="store_true",
        help="Fail the driver if input.final does not reproduce wout_final.nc within the rerun-WOUT gate",
    )
    parser.add_argument(
        "--admit-authoritative-rerun-wout",
        action="store_true",
        help=(
            "Allow driver success when rerun-WOUT reproducibility fails but "
            "the deterministic wout_final_rerun.nc passes its own solved-equilibrium gate; "
            "downstream transport must then use wout_final_rerun.nc as authoritative."
        ),
    )
    parser.add_argument("--wout-repro-mean-iota-atol", type=float, default=5.0e-4)
    parser.add_argument("--wout-repro-aspect-atol", type=float, default=1.0e-6)
    parser.add_argument("--wout-repro-profile-atol", type=float, default=5.0e-4)
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Generate VMEC-JAX boundary/|B|/history plots",
    )
    parser.add_argument(
        "--solved-wout-gate-aspect-atol",
        type=float,
        default=5.0e-2,
        help="Absolute aspect-ratio tolerance for the solved-candidate gate",
    )
    parser.add_argument(
        "--solved-wout-gate-min-abs-iota",
        type=float,
        default=None,
        help="Minimum accepted absolute mean iota; defaults to --min-iota",
    )
    parser.add_argument(
        "--solved-wout-gate-qs-max",
        type=float,
        default=5.0e-2,
        help="Maximum accepted final quasisymmetry residual before nonlinear audits",
    )
    parser.add_argument(
        "--solved-wout-gate-profile-floor",
        type=float,
        default=None,
        help="Solved iota-profile floor for the gate; defaults to --iota-profile-floor when enabled",
    )
    parser.add_argument(
        "--allow-failed-solved-wout-gate",
        action="store_true",
        help="Write failed solved-candidate gate artifacts but exit successfully for exploratory sweeps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble objectives and stop before solving",
    )
    args = parser.parse_args()
    if args.use_simple_seed and Path(args.input) == _default_input_file():
        # Mirror vmec_jax/examples/optimization/QA_optimization.py: the simple
        # seed branch starts from input.minimal_seed_nfp2, not the warm-start
        # QA input that is used when --use-simple-seed is disabled.
        args.input = _default_simple_seed_input_file()
    if args.strict_upstream_qa_baseline:
        gate_min_iota = (
            float(args.solved_wout_gate_min_abs_iota)
            if args.solved_wout_gate_min_abs_iota is not None
            else 0.41
        )
        args.constraints_only = True
        args.use_simple_seed = True
        args.disable_mode_continuation = True
        args.disable_iota_profile_floor = True
        args.input = _default_simple_seed_input_file()
        args.max_mode = 5
        args.min_vmec_mode = 7
        args.target_aspect = 5.0
        args.min_iota = gate_min_iota + max(
            float(args.strict_iota_admission_buffer), 0.0
        )
        args.solved_wout_gate_min_abs_iota = gate_min_iota
        args.iota_objective = "target"
        args.aspect_weight = 1.0
        args.iota_floor_weight = 10_000.0
        args.qs_weight = 1.0
        args.method = args.method or "scipy"
        args.scipy_tr_solver = "exact"
        args.save_rerun_wouts = True
        args.require_rerun_wout_gate = True
        # The upstream script's 70/1e-6 settings can terminate a few 1e-5 below
        # the strict iota admission gate. The preset keeps the same objective
        # and ESS recipe but avoids the premature outer step termination while
        # staying bounded enough for workstation/GPU artifact generation.
        args.max_nfev = max(int(args.max_nfev), 80)
        args.continuation_nfev = max(int(args.continuation_nfev), 25)
        args.inner_max_iter = max(int(args.inner_max_iter), 180)
        args.trial_max_iter = max(int(args.trial_max_iter), 180)
        args.inner_ftol = min(float(args.inner_ftol), 1.0e-10)
        args.trial_ftol = min(float(args.trial_ftol), 1.0e-10)
        args.ftol = min(float(args.ftol), 1.0e-5)
        args.gtol = min(float(args.gtol), 1.0e-5)
        args.xtol = min(float(args.xtol), 1.0e-8)
    return args


def main() -> int:
    args = _parse_args()
    import vmec_jax as vj

    max_mode = int(args.max_mode)
    min_vmec_mode = int(args.min_vmec_mode)
    if args.use_simple_seed:
        min_vmec_mode = max(min_vmec_mode, max_mode + 2)
    use_mode_continuation = (
        max_mode > 1
        and not bool(args.use_simple_seed)
        and not bool(args.disable_mode_continuation)
    )

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
        objective_transform=cast(
            VMECJAXTransportObjectiveTransform, str(args.spectrax_objective_transform)
        ),
        objective_scale=float(args.spectrax_objective_scale),
        surface_chunk_size=int(args.surface_chunk_size),
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
    iota_profile_floor = SignedIotaProfileFloor(
        float(args.iota_profile_floor), softness=1.0e-3
    )
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
        objective_tuples.insert(
            1, (iota_objective.J, float(args.min_iota), float(args.iota_floor_weight))
        )
    else:
        objective_tuples.insert(
            1, (iota_objective.J, 0.0, float(args.iota_floor_weight))
        )
    if not args.disable_iota_profile_floor:
        objective_tuples.insert(
            2, (iota_profile_floor.J, 0.0, float(args.iota_profile_floor_weight))
        )
    if not args.constraints_only:
        objective_tuples.append((transport.J, 0.0, float(args.spectrax_weight)))
    problem = vj.LeastSquaresProblem.from_tuples(objective_tuples)
    optimizer_method = str(
        args.method or ("scipy" if args.constraints_only else "scalar_trust")
    )
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
        "strict_upstream_qa_baseline": bool(args.strict_upstream_qa_baseline),
        "strict_iota_admission_buffer": (
            float(args.strict_iota_admission_buffer)
            if args.strict_upstream_qa_baseline
            else 0.0
        ),
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
            "objective_transform": str(args.spectrax_objective_transform),
            "objective_scale": float(args.spectrax_objective_scale),
            "surface_chunk_size": int(args.surface_chunk_size),
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
            "ftol": float(args.ftol),
            "gtol": float(args.gtol),
            "xtol": float(args.xtol),
            "use_ess": not bool(args.disable_ess),
            "ess_alpha": float(args.ess_alpha),
            "strict_upstream_qa_baseline": bool(args.strict_upstream_qa_baseline),
            "save_rerun_wouts": bool(args.save_rerun_wouts),
            "require_rerun_wout_gate": bool(args.require_rerun_wout_gate),
            "admit_authoritative_rerun_wout": bool(args.admit_authoritative_rerun_wout),
            "wout_repro_mean_iota_atol": float(args.wout_repro_mean_iota_atol),
            "wout_repro_aspect_atol": float(args.wout_repro_aspect_atol),
            "wout_repro_profile_atol": float(args.wout_repro_profile_atol),
        },
        "solved_wout_gate_policy": {
            "aspect_atol": float(args.solved_wout_gate_aspect_atol),
            "min_abs_mean_iota": (
                float(args.solved_wout_gate_min_abs_iota)
                if args.solved_wout_gate_min_abs_iota is not None
                else float(args.min_iota)
            ),
            "qs_residual_max": float(args.solved_wout_gate_qs_max),
            "iota_profile_floor": None
            if args.disable_iota_profile_floor
            else (
                float(args.solved_wout_gate_profile_floor)
                if args.solved_wout_gate_profile_floor is not None
                else float(args.iota_profile_floor)
            ),
            "allow_failed_gate": bool(args.allow_failed_solved_wout_gate),
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
    summary["optimizer_comparison"] = {
        "schema_version": 1,
        "method": optimizer_method,
        "comparison_class": (
            "constraints_only_qa"
            if args.constraints_only
            else f"spectraxgk_transport_{args.transport_kind}"
        ),
        "transport_kind": None if args.constraints_only else str(args.transport_kind),
        "sample_set_fingerprint": {
            "surfaces": [float(x) for x in sample_set.surfaces],
            "alphas": [float(x) for x in sample_set.alphas],
            "ky_values": [float(x) for x in sample_set.ky_values],
            "n_samples": int(sample_set.n_samples),
            "ntheta": int(args.ntheta),
            "mboz": int(args.mboz),
            "nboz": int(args.nboz),
            "n_laguerre": int(args.n_laguerre),
            "n_hermite": int(args.n_hermite),
            "objective_transform": str(args.spectrax_objective_transform),
            "objective_scale": float(args.spectrax_objective_scale),
            "surface_chunk_size": int(args.surface_chunk_size),
        },
        "optimizer_budget": {
            "max_nfev": int(args.max_nfev),
            "continuation_nfev": int(args.continuation_nfev),
            "inner_max_iter": int(args.inner_max_iter),
            "trial_max_iter": int(args.trial_max_iter),
            "inner_ftol": float(args.inner_ftol),
            "trial_ftol": float(args.trial_ftol),
            "ftol": float(args.ftol),
            "gtol": float(args.gtol),
            "xtol": float(args.xtol),
            "use_ess": not bool(args.disable_ess),
            "ess_alpha": float(args.ess_alpha),
        },
        "solved_equilibrium_targets": {
            "target_aspect": float(args.target_aspect),
            "min_iota": float(args.min_iota),
            "iota_objective": str(args.iota_objective),
            "strict_upstream_qa_baseline": bool(args.strict_upstream_qa_baseline),
        },
        "nonlinear_promotion_policy": {
            "claim_boundary": (
                "optimizer output is screening evidence until matched long post-transient "
                "nonlinear audits, seed/timestep ensembles, and grid/window convergence pass"
            ),
            "recommended_horizons": "700,1100,1500",
            "recommended_window_tmin": 1100.0,
            "recommended_window_tmax": 1500.0,
            "recommended_seed_variants": [32, 33],
            "recommended_dt_variant": 0.04,
        },
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "setup_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
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
        use_ess=not bool(args.disable_ess),
        ess_alpha=float(args.ess_alpha),
        label=(
            "QA constraints-only optimization"
            if args.constraints_only
            else "QA optimization with SPECTRAX-GK transport residual"
        ),
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
    transport_metric = (
        {
            "transport_objective_final": None,
            "transport_objective_source": "constraints_only_skipped",
            "transport_metric_final": None,
        }
        if args.constraints_only
        else _transport_metric_from_result(transport, result)
    )
    for key, value in transport_metric.items():
        if value is not None:
            result.history[key] = value
    _update_history_with_transport_metric(
        args.outdir / "history.json", transport_metric
    )
    print("\nFinal VMEC-JAX diagnostics:")
    print(f"  aspect ratio: {result.history['aspect_final']:.6g}")
    print(f"  mean iota:    {result.history['iota_final']:.6g}")
    print(f"  QS residual:  {result.history['qs_final']:.6e}")
    print(f"  objective:    {result.history['objective_final']:.6e}")
    transport_value = _finite_float_or_none(
        transport_metric.get("transport_objective_final")
    )
    if transport_value is not None:
        print(f"  transport:    {transport_value:.6e}")
    print("\nFiles:")
    for name, path in saved.as_dict().items():
        print(f"  {name}: {path}")
    gate_profile_floor = (
        None
        if args.disable_iota_profile_floor
        else (
            float(args.solved_wout_gate_profile_floor)
            if args.solved_wout_gate_profile_floor is not None
            else float(args.iota_profile_floor)
        )
    )
    rerun_gate_report = None
    rerun_wout_admission_report = None
    if args.save_rerun_wouts:
        rerun_paths = {}
        if Path(saved.initial_input).exists():
            initial_rerun_wout = args.outdir / "wout_initial_rerun.nc"
            initial_rerun = vj.run_fixed_boundary(
                str(saved.initial_input), verbose=False
            )
            vj.write_wout_from_fixed_boundary_run(
                str(initial_rerun_wout), initial_rerun
            )
            rerun_paths["initial_rerun_wout"] = str(initial_rerun_wout)
        final_rerun_wout = args.outdir / "wout_final_rerun.nc"
        final_rerun = vj.run_fixed_boundary(str(saved.final_input), verbose=False)
        vj.write_wout_from_fixed_boundary_run(str(final_rerun_wout), final_rerun)
        rerun_paths["final_rerun_wout"] = str(final_rerun_wout)
        rerun_gate_report = build_wout_reproducibility_gate(
            saved.final_wout,
            final_rerun_wout,
            target_aspect=float(args.target_aspect),
            aspect_atol=float(args.solved_wout_gate_aspect_atol),
            min_abs_mean_iota=(
                float(args.solved_wout_gate_min_abs_iota)
                if args.solved_wout_gate_min_abs_iota is not None
                else float(args.min_iota)
            ),
            iota_profile_floor=gate_profile_floor,
            mean_iota_repro_atol=float(args.wout_repro_mean_iota_atol),
            aspect_repro_atol=float(args.wout_repro_aspect_atol),
            profile_repro_atol=float(args.wout_repro_profile_atol),
        )
        rerun_gate_report["rerun_paths"] = rerun_paths
        rerun_gate_path = args.outdir / "wout_reproducibility_gate.json"
        rerun_gate_path.write_text(
            json.dumps(rerun_gate_report, indent=2, allow_nan=False), encoding="utf-8"
        )
        print("\nWOUT reproducibility gate:")
        print(f"  passed: {rerun_gate_report['passed']}")
        print(f"  file:   {rerun_gate_path}")
        rerun_wout_admission_report = build_authoritative_wout_candidate_gate(
            final_rerun_wout,
            target_aspect=float(args.target_aspect),
            aspect_atol=float(args.solved_wout_gate_aspect_atol),
            min_abs_mean_iota=(
                float(args.solved_wout_gate_min_abs_iota)
                if args.solved_wout_gate_min_abs_iota is not None
                else float(args.min_iota)
            ),
            qs_residual_max=float(args.solved_wout_gate_qs_max),
            iota_profile_floor=gate_profile_floor,
        )
        rerun_wout_admission_path = args.outdir / "rerun_wout_admission_gate.json"
        rerun_wout_admission_path.write_text(
            json.dumps(rerun_wout_admission_report, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print("\nAuthoritative rerun-WOUT admission gate:")
        print(f"  passed: {rerun_wout_admission_report['passed']}")
        print(f"  file:   {rerun_wout_admission_path}")
    gate_report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=float(args.target_aspect),
        aspect_atol=float(args.solved_wout_gate_aspect_atol),
        min_abs_mean_iota=(
            float(args.solved_wout_gate_min_abs_iota)
            if args.solved_wout_gate_min_abs_iota is not None
            else float(args.min_iota)
        ),
        qs_residual_max=float(args.solved_wout_gate_qs_max),
        iota_profile_floor=gate_profile_floor,
    )
    gate_path = args.outdir / "solved_wout_gate.json"
    gate_path.write_text(
        json.dumps(gate_report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("\nSolved candidate gate:")
    print(f"  passed: {gate_report['passed']}")
    print(f"  file:   {gate_path}")
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
                mbooz=int(args.mboz),
                nbooz=int(args.nboz),
            ),
            "objective_history": vj.plot_objective_history(
                saved.history,
                outdir=args.outdir,
            ),
        }
        print("\nPlot files:")
        for name, path in plot_paths.items():
            print(f"  {name}: {path}")
    if not bool(gate_report["passed"]) and not bool(args.allow_failed_solved_wout_gate):
        return 2
    if (
        args.require_rerun_wout_gate
        and rerun_gate_report is not None
        and not bool(rerun_gate_report["passed"])
        and not bool(args.allow_failed_solved_wout_gate)
    ):
        if bool(args.admit_authoritative_rerun_wout) and bool(
            (rerun_wout_admission_report or {}).get("passed", False)
        ):
            return 0
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
