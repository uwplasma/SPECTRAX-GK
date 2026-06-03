#!/usr/bin/env python3
"""Probe VMEC-JAX/SPECTRAX-GK boundary-gradient chain conventions.

This diagnostic is intentionally narrower than
``build_vmec_jax_transport_gradient_diagnostic.py``.  It decomposes one active
boundary coefficient into:

* raw plus/minus exact-solve finite differences;
* frozen-axis initial-state finite differences, matching the VMEC-JAX optimizer
  derivative convention;
* VMEC-JAX exact-tape JVP and VJP contractions; and
* the SPECTRAX-GK final-state cotangent dotted into exact final-state finite
  differences.

Use it when sparse AD/FD checks fail and the next question is whether the
mismatch is a SPECTRAX final-state cotangent issue, a VMEC-JAX tape transpose
issue, or a raw finite-difference branch/convergence issue.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_vmec_jax_transport_gradient_diagnostic import _build_stage, _sample_set_from_args  # noqa: E402
from spectraxgk.vmec_jax_boundary_chain import build_boundary_chain_summary  # noqa: E402
from spectraxgk.vmec_jax_transport_objective import (  # noqa: E402
    VMECJAXTransportObjectiveConfig,
    _reference_wout_from_context,
    vmec_jax_transport_growth_branch_locality_report_from_states,
)
from vmec_jax.discrete_adjoint import (  # noqa: E402
    checkpoint_tape_state_jvp_columns,
    checkpoint_tape_state_vjp,
)
from vmec_jax.init_guess import initial_guess_from_boundary  # noqa: E402
from vmec_jax.state import pack_state, unpack_state  # noqa: E402


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        drop = {"resume_state", "adjoint_step_trace"}
        return {str(k): _safe_json(v) for k, v in value.items() if str(k) not in drop}
    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            return (
                [_safe_json(v) for v in value[:5]]
                + [f"... {len(value) - 10} omitted ..."]
                + [_safe_json(v) for v in value[-5:]]
            )
        return [_safe_json(v) for v in value]
    if hasattr(value, "shape"):
        arr = np.asarray(value)
        if arr.size > 20:
            return {
                "shape": [int(item) for item in arr.shape],
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
            }
        return arr.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=float).reshape(-1)))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--index", type=int, default=28)
    parser.add_argument("--step", type=float, default=1.0e-4)
    parser.add_argument("--max-mode", type=int, default=5)
    parser.add_argument("--min-vmec-mode", type=int, default=7)
    parser.add_argument("--surfaces", type=_float_tuple, default=(0.64,))
    parser.add_argument("--alphas", type=_float_tuple, default=(0.0,))
    parser.add_argument("--ky-values", type=_float_tuple, default=(0.3,))
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
    parser.add_argument("--inner-max-iter", type=int, default=500)
    parser.add_argument("--inner-ftol", type=float, default=1.0e-10)
    parser.add_argument("--trial-max-iter", type=int, default=500)
    parser.add_argument("--trial-ftol", type=float, default=1.0e-10)
    parser.add_argument("--solver-device", choices=("cpu", "gpu"), default=None)
    parser.add_argument("--exact-relative-tolerance", type=float, default=1.0e-1)
    parser.add_argument("--internal-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--include-growth-branch-locality",
        action="store_true",
        help="Also compare dominant-growth eigenbranch locality at base/plus/minus final states",
    )
    parser.add_argument("--branch-gap-floor", type=float, default=1.0e-8)
    parser.add_argument("--branch-slope-rtol", type=float, default=1.0e-2)
    parser.add_argument("--branch-slope-atol", type=float, default=1.0e-8)
    parser.add_argument(
        "--branch-locality-max-samples",
        type=int,
        default=0,
        help="Limit branch-locality samples for debugging; 0 checks the full configured sample set",
    )
    return parser.parse_args(argv)


def _stage_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        input=args.input,
        out_json=args.out_json,
        max_mode=int(args.max_mode),
        min_vmec_mode=int(args.min_vmec_mode),
        surfaces=tuple(float(x) for x in args.surfaces),
        alphas=tuple(float(x) for x in args.alphas),
        ky_values=tuple(float(x) for x in args.ky_values),
        transport_kind=str(args.transport_kind),
        transport_weight=float(args.transport_weight),
        ntheta=int(args.ntheta),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        n_laguerre=int(args.n_laguerre),
        n_hermite=int(args.n_hermite),
        spectrax_objective_transform=str(args.spectrax_objective_transform),
        spectrax_objective_scale=float(args.spectrax_objective_scale),
        inner_max_iter=int(args.inner_max_iter),
        inner_ftol=float(args.inner_ftol),
        trial_max_iter=int(args.trial_max_iter),
        trial_ftol=float(args.trial_ftol),
        solver_device=args.solver_device,
    )


def _objective_and_cotangent_from_packed(opt: Any, packed: Any) -> tuple[Any, Any]:
    factory = getattr(
        opt._residuals_fn,
        "_state_objective_value_and_cotangent_from_packed",
        None,
    )
    if factory is not None:
        cost, cotangent = factory(packed, opt._layout)
        return jax.block_until_ready(cost), jax.block_until_ready(cotangent)
    state = unpack_state(packed, opt._layout)
    residuals = jnp.asarray(opt._residuals_fn(state), dtype=jnp.float64)
    cost = 0.5 * jnp.vdot(residuals, residuals)

    def residuals_from_packed(packed_state: Any) -> Any:
        return jnp.asarray(
            opt._residuals_fn(unpack_state(packed_state, opt._layout)),
            dtype=jnp.float64,
        )

    _, vjp = jax.vjp(residuals_from_packed, packed)
    cotangent = vjp(residuals)[0]
    return jax.block_until_ready(cost), jax.block_until_ready(cotangent)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stage, setup = _build_stage(_stage_args(args))
    opt = stage.optimizer
    params0 = np.zeros(len(stage.specs), dtype=float)
    if int(args.index) < 0 or int(args.index) >= params0.size:
        raise IndexError(
            f"--index {args.index} outside active parameter count {params0.size}"
        )
    direction = np.zeros_like(params0)
    direction[int(args.index)] = 1.0
    plus = params0 + float(args.step) * direction
    minus = params0 - float(args.step) * direction

    def solve(
        params: np.ndarray, *, jvp_only: bool = False
    ) -> tuple[Any, dict[str, Any], Any]:
        state, payload = opt._solve_exact_with_tape(
            jnp.asarray(params, dtype=jnp.float64),
            return_payload=True,
            jvp_only=bool(jvp_only),
        )
        packed = opt._packed_final_from_exact_payload(state, payload)
        return (
            state,
            payload,
            jax.block_until_ready(jnp.asarray(packed, dtype=jnp.float64)),
        )

    def initial_packed(params: np.ndarray) -> Any:
        state0 = opt._initial_state_from_params(
            jnp.asarray(params, dtype=jnp.float64),
            profile_name="boundary_chain_initial",
        )
        return jax.block_until_ready(jnp.asarray(pack_state(state0), dtype=jnp.float64))

    def frozen_axis_initial_packed(
        params: np.ndarray, axis_override: Mapping[str, Any]
    ) -> Any:
        frozen_axis = {
            key: jnp.asarray(value, dtype=jnp.float64)
            for key, value in axis_override.items()
        }
        boundary = opt._boundary_from_params(jnp.asarray(params, dtype=jnp.float64))
        state0 = initial_guess_from_boundary(
            opt._static,
            boundary,
            opt._indata,
            vmec_project=True,
            axis_override=frozen_axis,
        )
        return jax.block_until_ready(jnp.asarray(pack_state(state0), dtype=jnp.float64))

    _state_base, payload_base, packed_base = solve(params0, jvp_only=False)
    _state_plus, payload_plus, packed_plus = solve(plus, jvp_only=True)
    _state_minus, payload_minus, packed_minus = solve(minus, jvp_only=True)
    tape = payload_base["tape"]
    cost_base, final_cotangent = _objective_and_cotangent_from_packed(opt, packed_base)
    cost_plus, _ = _objective_and_cotangent_from_packed(opt, packed_plus)
    cost_minus, _ = _objective_and_cotangent_from_packed(opt, packed_minus)
    final_cotangent = jnp.nan_to_num(
        jnp.asarray(final_cotangent, dtype=jnp.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    raw_initial_fd = (initial_packed(plus) - initial_packed(minus)) / (
        2.0 * float(args.step)
    )
    frozen_axis_initial_fd = (
        frozen_axis_initial_packed(plus, payload_base["axis_override"])
        - frozen_axis_initial_packed(minus, payload_base["axis_override"])
    ) / (2.0 * float(args.step))
    tangent_columns = opt._initial_tangent_columns(
        jnp.asarray(params0, dtype=jnp.float64),
        payload_base["axis_override"],
        profile_prefix="boundary_chain",
    )
    frozen_axis_linear_tangent = jax.block_until_ready(
        jnp.asarray(tangent_columns[int(args.index)], dtype=jnp.float64)
    )

    def replay(initial_tangent: Any) -> Any:
        out = checkpoint_tape_state_jvp_columns(
            tape=tape,
            static=opt._static,
            initial_tangents=jnp.asarray(initial_tangent, dtype=jnp.float64).reshape(
                1, -1
            ),
            rebuild_preconditioner=True,
        )[0]
        return jax.block_until_ready(jnp.asarray(out, dtype=jnp.float64))

    final_fd = (packed_plus - packed_minus) / (2.0 * float(args.step))
    tape_jvp_raw_initial = replay(raw_initial_fd)
    tape_jvp_frozen_axis_fd = replay(frozen_axis_initial_fd)
    tape_jvp_frozen_axis_linear = replay(frozen_axis_linear_tangent)
    initial_cotangent = checkpoint_tape_state_vjp(
        tape=tape,
        static=opt._static,
        final_cotangent=final_cotangent,
        rebuild_preconditioner=True,
    )
    initial_cotangent = jax.block_until_ready(
        jnp.nan_to_num(
            jnp.asarray(initial_cotangent, dtype=jnp.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )

    exact_fd_cost_gradient = float((cost_plus - cost_minus) / (2.0 * float(args.step)))
    frozen_jvp = float(jnp.vdot(final_cotangent, tape_jvp_frozen_axis_fd))
    frozen_vjp = float(jnp.vdot(initial_cotangent, frozen_axis_initial_fd))
    result: dict[str, Any] = {
        "kind": "vmec_jax_boundary_chain_probe",
        "setup": setup,
        "index": int(args.index),
        "step": float(args.step),
        "name": getattr(
            stage.specs[int(args.index)], "name", str(stage.specs[int(args.index)])
        ),
        "base_cost": float(cost_base),
        "plus_cost": float(cost_plus),
        "minus_cost": float(cost_minus),
        "exact_fd_cost_gradient": exact_fd_cost_gradient,
        "final_cot_dot_exact_final_fd": float(jnp.vdot(final_cotangent, final_fd)),
        "final_cot_dot_tape_jvp_raw_initial_fd": float(
            jnp.vdot(final_cotangent, tape_jvp_raw_initial)
        ),
        "final_cot_dot_tape_jvp_frozen_axis_fd": frozen_jvp,
        "final_cot_dot_tape_jvp_frozen_axis_linear": float(
            jnp.vdot(final_cotangent, tape_jvp_frozen_axis_linear)
        ),
        "initial_cot_dot_raw_initial_fd": float(
            jnp.vdot(initial_cotangent, raw_initial_fd)
        ),
        "initial_cot_dot_frozen_axis_fd": frozen_vjp,
        "initial_cot_dot_frozen_axis_linear": float(
            jnp.vdot(initial_cotangent, frozen_axis_linear_tangent)
        ),
        "exact_final_fd_norm": _norm(final_fd),
        "tape_jvp_final_raw_initial_norm": _norm(tape_jvp_raw_initial),
        "tape_jvp_final_frozen_axis_fd_norm": _norm(tape_jvp_frozen_axis_fd),
        "tape_jvp_final_frozen_axis_linear_norm": _norm(tape_jvp_frozen_axis_linear),
        "raw_initial_fd_norm": _norm(raw_initial_fd),
        "frozen_axis_initial_fd_norm": _norm(frozen_axis_initial_fd),
        "frozen_axis_linear_initial_norm": _norm(frozen_axis_linear_tangent),
        "final_cot_norm": _norm(final_cotangent),
        "initial_cot_norm": _norm(initial_cotangent),
        "base_tape_diagnostics": _safe_json(getattr(tape, "diagnostics", {})),
        "plus_tape_diagnostics": _safe_json(
            getattr(payload_plus["tape"], "diagnostics", {})
        ),
        "minus_tape_diagnostics": _safe_json(
            getattr(payload_minus["tape"], "diagnostics", {})
        ),
    }
    result["summary"] = build_boundary_chain_summary(
        exact_fd_cost_gradient=exact_fd_cost_gradient,
        final_cot_dot_exact_final_fd=result["final_cot_dot_exact_final_fd"],
        frozen_axis_replay_cost_gradient=frozen_jvp,
        frozen_axis_vjp_cost_gradient=frozen_vjp,
        raw_initial_replay_cost_gradient=result[
            "final_cot_dot_tape_jvp_raw_initial_fd"
        ],
        raw_initial_fd_norm=result["raw_initial_fd_norm"],
        frozen_axis_initial_fd_norm=result["frozen_axis_initial_fd_norm"],
        exact_relative_tolerance=float(args.exact_relative_tolerance),
        internal_relative_tolerance=float(args.internal_relative_tolerance),
        absolute_tolerance=float(args.absolute_tolerance),
    )
    if bool(args.include_growth_branch_locality):
        ctx = getattr(stage, "ctx", None)
        if ctx is None:
            result["growth_branch_locality"] = {
                "enabled": True,
                "passed": False,
                "classification": "stage_context_unavailable",
                "blockers": ["stage_context_unavailable"],
            }
        else:
            config = VMECJAXTransportObjectiveConfig(
                kind=cast(Any, args.transport_kind),
                sample_set=_sample_set_from_args(_stage_args(args)),
                ntheta=int(args.ntheta),
                mboz=int(args.mboz),
                nboz=int(args.nboz),
                n_laguerre=int(args.n_laguerre),
                n_hermite=int(args.n_hermite),
                objective_transform=cast(Any, str(args.spectrax_objective_transform)),
                objective_scale=float(args.spectrax_objective_scale),
            )
            result["growth_branch_locality"] = (
                vmec_jax_transport_growth_branch_locality_report_from_states(
                    _state_base,
                    _state_plus,
                    _state_minus,
                    ctx.static,
                    ctx.indata,
                    _reference_wout_from_context(ctx),
                    config,
                    step=float(args.step),
                    gap_floor=float(args.branch_gap_floor),
                    slope_rtol=float(args.branch_slope_rtol),
                    slope_atol=float(args.branch_slope_atol),
                    max_samples=int(args.branch_locality_max_samples),
                )
            )
    else:
        result["growth_branch_locality"] = {
            "enabled": False,
            "passed": False,
            "classification": "growth_branch_locality_not_requested",
            "blockers": ["growth_branch_locality_not_requested"],
        }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if bool(result["summary"]["finite"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
