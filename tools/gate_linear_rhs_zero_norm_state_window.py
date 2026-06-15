#!/usr/bin/env python3
"""Gate profiler-zero linear RHS terms across a bounded state window.

The linear RHS term profiler can report zero norms for some terms on the
default initial condition. This gate checks whether those terms remain
identity-safe when the state contains resolved parallel variation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import apply_imported_geometry_grid_defaults
from spectraxgk.grids import build_spectral_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.linear import build_linear_cache
from spectraxgk.runtime import (
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from spectraxgk.terms.assembly import assemble_rhs_cached, assemble_rhs_terms_cached
from spectraxgk.terms.config import TermConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"),
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--kx", type=float, default=None)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--z-mode", type=int, default=1)
    parser.add_argument("--z-wave-amplitude", type=float, default=1.0e-3)
    parser.add_argument("--kick-dt", type=float, default=1.0e-3)
    parser.add_argument("--identity-threshold", type=float, default=1.0e-10)
    parser.add_argument("--activation-threshold", type=float, default=1.0e-8)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("docs/_static/linear_rhs_zero_norm_state_window_gate.json"),
    )
    return parser.parse_args()


def _block_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        jax.block_until_ready(leaf)


def _float_norm(arr: jnp.ndarray) -> float:
    return float(np.asarray(jnp.linalg.norm(arr)))


def _relative_error(reference: jnp.ndarray, candidate: jnp.ndarray) -> float:
    ref_norm = _float_norm(reference)
    diff_norm = _float_norm(reference - candidate)
    return diff_norm / max(ref_norm, 1.0e-30)


def _z_variation_norm(state: jnp.ndarray) -> float:
    mean_z = jnp.mean(state, axis=-1, keepdims=True)
    return _float_norm(state - mean_z)


def _inject_z_wave(
    state: jnp.ndarray,
    *,
    ky_index: int,
    kx_index: int,
    amplitude: float,
    z_mode: int,
) -> jnp.ndarray:
    """Inject a deterministic resolved parallel wave into one Hermite mode."""

    state = jnp.asarray(state)
    Nz = state.shape[-1]
    Nm = state.shape[-4] if state.ndim == 6 else state.shape[-4]
    m_index = min(max(1, Nm - 1), 3)
    z = jnp.arange(Nz, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * float(z_mode) * z / float(Nz)
    wave = amplitude * jnp.exp(1j * phase).astype(state.dtype)
    perturbation = jnp.zeros_like(state)
    if state.ndim == 6:
        perturbation = perturbation.at[:, 0, m_index, ky_index, kx_index, :].set(wave)
    elif state.ndim == 5:
        perturbation = perturbation.at[0, m_index, ky_index, kx_index, :].set(wave)
    else:  # pragma: no cover - upstream assembly validates this before normal use.
        raise ValueError("state must have 5 or 6 dimensions")
    return state + perturbation


def _term_skip_config(term_cfg: TermConfig, term: str) -> TermConfig:
    if term == "collisions":
        return replace(term_cfg, collisions=0.0)
    if term == "hypercollisions":
        return replace(term_cfg, hypercollisions=0.0)
    raise ValueError(f"unsupported zero-norm skip candidate: {term}")


def _evaluate_state(
    name: str,
    state: jnp.ndarray,
    *,
    cache: Any,
    params: Any,
    term_cfg: TermConfig,
    candidates: tuple[str, ...] = ("collisions", "hypercollisions"),
) -> dict[str, Any]:
    rhs, _fields, contrib = assemble_rhs_terms_cached(
        state,
        cache,
        params,
        terms=term_cfg,
        use_custom_vjp=False,
    )
    full_rhs, _ = assemble_rhs_cached(
        state,
        cache,
        params,
        terms=term_cfg,
        use_custom_vjp=False,
    )
    _block_tree((rhs, full_rhs, contrib))
    term_norms = {key: _float_norm(value) for key, value in contrib.items()}
    skip_errors: dict[str, float] = {}
    for candidate in candidates:
        skipped_rhs, _ = assemble_rhs_cached(
            state,
            cache,
            params,
            terms=_term_skip_config(term_cfg, candidate),
            use_custom_vjp=False,
        )
        _block_tree(skipped_rhs)
        skip_errors[candidate] = _relative_error(full_rhs, skipped_rhs)
    return {
        "state": name,
        "state_norm": _float_norm(state),
        "z_variation_norm": _z_variation_norm(state),
        "term_norms": term_norms,
        "relative_skip_errors": skip_errors,
    }


def _build_summary(
    rows: list[dict[str, Any]],
    *,
    config: str,
    ky: float,
    kx: float | None,
    nl: int,
    nm: int,
    identity_threshold: float,
    activation_threshold: float,
) -> dict[str, Any]:
    candidate_names = ("collisions", "hypercollisions")
    candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidate_names:
        skip_errors = [float(row["relative_skip_errors"][candidate]) for row in rows]
        term_norms = [float(row["term_norms"][candidate]) for row in rows]
        initial_skip_error = skip_errors[0] if skip_errors else math.nan
        max_skip_error = max(skip_errors, default=math.nan)
        max_term_norm = max(term_norms, default=math.nan)
        active_states = [
            str(row["state"])
            for row in rows
            if float(row["relative_skip_errors"][candidate]) > activation_threshold
            or float(row["term_norms"][candidate]) > activation_threshold
        ]
        candidates[candidate] = {
            "initial_relative_skip_error": initial_skip_error,
            "max_relative_skip_error": max_skip_error,
            "max_term_norm": max_term_norm,
            "active_states": active_states,
            "safe_to_disable_over_window": bool(max_skip_error <= identity_threshold),
        }

    collisions_identity_pass = bool(
        candidates["collisions"]["safe_to_disable_over_window"]
    )
    hyper_initial_zero = bool(
        candidates["hypercollisions"]["initial_relative_skip_error"]
        <= identity_threshold
    )
    hyper_rejected = bool(
        candidates["hypercollisions"]["max_relative_skip_error"] > activation_threshold
    )
    return {
        "kind": "linear_rhs_zero_norm_state_window_gate",
        "case": Path(config).stem,
        "config": config,
        "backend": jax.default_backend(),
        "ky": float(ky),
        "kx": None if kx is None else float(kx),
        "Nl": int(nl),
        "Nm": int(nm),
        "identity_threshold": float(identity_threshold),
        "activation_threshold": float(activation_threshold),
        "rows": rows,
        "candidates": candidates,
        "passed": collisions_identity_pass and hyper_initial_zero and hyper_rejected,
        "policy": (
            "A term may only be disabled by a fast path when its relative skip error stays below "
            "the identity threshold over the whole sampled state window. A profiler-zero initial "
            "state is insufficient evidence for nonlinear states with resolved parallel variation."
        ),
        "recommendations": {
            "collisions": (
                "This Cyclone window supports a zero-collision fast path only when all collision "
                "frequencies are zero and the same identity gate passes."
            ),
            "hypercollisions": (
                "Do not skip kz hypercollisions based on the initial-state zero norm; the z-varying "
                "state activates the contribution."
            ),
        },
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    cfg, _ = load_runtime_from_toml(args.config)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
    term_cfg = build_runtime_term_config(cfg)
    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=args.ky,
        kx_target=args.kx,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    state0 = jnp.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=kx_index,
            Nl=args.Nl,
            Nm=args.Nm,
            nspecies=len(cfg.species),
        )
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    z_wave = _inject_z_wave(
        state0,
        ky_index=int(ky_index),
        kx_index=int(kx_index),
        amplitude=float(args.z_wave_amplitude),
        z_mode=int(args.z_mode),
    )
    rhs0, _ = assemble_rhs_cached(
        state0, cache, params, terms=term_cfg, use_custom_vjp=False
    )
    rhs_wave, _ = assemble_rhs_cached(
        z_wave, cache, params, terms=term_cfg, use_custom_vjp=False
    )
    _block_tree((rhs0, rhs_wave))
    states = [
        ("initial", state0),
        ("initial_linear_kick", state0 + args.kick_dt * rhs0),
        ("z_wave", z_wave),
        ("z_wave_linear_kick", z_wave + args.kick_dt * rhs_wave),
    ]
    rows = [
        _evaluate_state(name, state, cache=cache, params=params, term_cfg=term_cfg)
        for name, state in states
    ]
    payload = _build_summary(
        rows,
        config=str(args.config),
        ky=args.ky,
        kx=args.kx,
        nl=args.Nl,
        nm=args.Nm,
        identity_threshold=args.identity_threshold,
        activation_threshold=args.activation_threshold,
    )
    _write_json(payload, args.out_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"saved {args.out_json}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
