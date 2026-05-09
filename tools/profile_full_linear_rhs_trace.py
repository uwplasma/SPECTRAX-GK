#!/usr/bin/env python3
"""Profile and trace the fused full-linear-RHS graph for runtime TOML cases."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    from tools._profiler_options import make_profile_options
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _profiler_options import make_profile_options

from spectraxgk.geometry import apply_gx_geometry_grid_defaults
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
from spectraxgk.terms.assembly import _is_static_zero, assemble_rhs_cached


HLO_TOKENS = (
    "fusion",
    "fft",
    "reduce",
    "gather",
    "scatter",
    "transpose",
    "broadcast",
    "reshape",
    "slice",
    "concatenate",
    "convert",
    "multiply",
    "add",
    "subtract",
    "divide",
    "select",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml"),
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--kx", type=float, default=None)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--state", choices=("initial", "z_wave"), default="initial")
    parser.add_argument("--z-mode", type=int, default=1)
    parser.add_argument("--z-wave-amplitude", type=float, default=1.0e-3)
    parser.add_argument("--summary-json", type=Path, default=Path("docs/_static/full_linear_rhs_trace_summary.json"))
    parser.add_argument("--hlo-out", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--memory-profile", type=Path, default=None)
    parser.add_argument("--python-tracer-level", type=int, default=0)
    parser.add_argument("--host-tracer-level", type=int, default=0)
    return parser.parse_args()


def _block_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        try:
            jax.block_until_ready(leaf)
        except TypeError:
            continue


def _time_call(fn, *args) -> tuple[float, Any]:
    t0 = time.perf_counter()
    out = fn(*args)
    _block_tree(out)
    return time.perf_counter() - t0, out


def _z_variation_norm(state: jnp.ndarray) -> float:
    mean_z = jnp.mean(state, axis=-1, keepdims=True)
    return float(np.asarray(jnp.linalg.norm(state - mean_z)))


def _inject_z_wave(
    state: jnp.ndarray,
    *,
    ky_index: int,
    kx_index: int,
    amplitude: float,
    z_mode: int,
) -> jnp.ndarray:
    """Inject a deterministic parallel wave so linked-z paths are active."""

    state = jnp.asarray(state)
    nz = state.shape[-1]
    nm = state.shape[-4]
    m_index = min(max(1, nm - 1), 3)
    z = jnp.arange(nz, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * float(z_mode) * z / float(nz)
    wave = amplitude * jnp.exp(1j * phase).astype(state.dtype)
    perturbation = jnp.zeros_like(state)
    if state.ndim == 6:
        perturbation = perturbation.at[:, 0, m_index, ky_index, kx_index, :].set(wave)
    elif state.ndim == 5:
        perturbation = perturbation.at[0, m_index, ky_index, kx_index, :].set(wave)
    else:  # pragma: no cover - runtime state builder controls dimensionality.
        raise ValueError("state must have 5 or 6 dimensions")
    return state + perturbation


def _hlo_token_counts(hlo_text: str, tokens: tuple[str, ...] = HLO_TOKENS) -> dict[str, int]:
    """Count coarse HLO tokens used for trace triage."""

    lower = hlo_text.lower()
    return {token: lower.count(token) for token in tokens}


def _build_summary(
    *,
    config: str,
    backend: str,
    nl: int,
    nm: int,
    repeats: int,
    state: str,
    z_variation_norm: float,
    compile_execute_seconds: float,
    warm_seconds: float,
    rhs_norm: float,
    phi_norm: float,
    hlo_text: str,
    trace_dir: Path | None,
    memory_profile: Path | None,
    hlo_out: Path | None,
    force_electrostatic_fields: bool,
) -> dict[str, Any]:
    """Build a machine-readable full-linear-RHS trace summary."""

    return {
        "kind": "full_linear_rhs_trace_summary",
        "case": Path(config).stem,
        "config": config,
        "backend": backend,
        "Nl": int(nl),
        "Nm": int(nm),
        "repeats": int(repeats),
        "state": state,
        "z_variation_norm": float(z_variation_norm),
        "compile_execute_seconds": float(compile_execute_seconds),
        "warm_seconds": float(warm_seconds),
        "rhs_norm": float(rhs_norm),
        "phi_norm": float(phi_norm),
        "hlo_line_count": len(hlo_text.splitlines()),
        "hlo_bytes": len(hlo_text.encode("utf-8")),
        "hlo_token_counts": _hlo_token_counts(hlo_text),
        "trace_dir": None if trace_dir is None else str(trace_dir),
        "memory_profile": None if memory_profile is None else str(memory_profile),
        "hlo_out": None if hlo_out is None else str(hlo_out),
        "force_electrostatic_fields": bool(force_electrostatic_fields),
        "claim_scope": (
            "Full fused linear-RHS graph triage for one runtime state. Use this to choose "
            "kernel-level optimization targets; do not treat it as a standalone runtime claim."
        ),
    }


def _write_summary_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    cfg, _ = load_runtime_from_toml(args.config)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
    term_cfg = build_runtime_term_config(cfg)
    linear_terms = replace(term_cfg, nonlinear=0.0)
    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=args.ky,
        kx_target=args.kx,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    g0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=args.Nl,
        Nm=args.Nm,
        nspecies=len(cfg.species),
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    g0 = jnp.asarray(g0)
    if args.state == "z_wave":
        g0 = _inject_z_wave(
            g0,
            ky_index=int(ky_index),
            kx_index=int(kx_index),
            amplitude=float(args.z_wave_amplitude),
            z_mode=int(args.z_mode),
        )

    force_electrostatic_fields = _is_static_zero(linear_terms.apar) and _is_static_zero(linear_terms.bpar)
    rhs_fn = jax.jit(
        lambda state: assemble_rhs_cached(
            state,
            cache,
            params,
            terms=linear_terms,
            force_electrostatic_fields=force_electrostatic_fields,
        )
    )
    compile_execute_seconds, first_out = _time_call(rhs_fn, g0)
    rhs, fields = first_out

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(
            str(args.trace_dir),
            profiler_options=make_profile_options(
                python_tracer_level=int(args.python_tracer_level),
                host_tracer_level=int(args.host_tracer_level),
            ),
        )
    try:
        warm_t0 = time.perf_counter()
        for _ in range(int(args.repeats)):
            rhs, fields = rhs_fn(g0)
            _block_tree((rhs, fields))
        warm_seconds = (time.perf_counter() - warm_t0) / float(args.repeats)
    finally:
        if args.trace_dir is not None:
            jax.profiler.stop_trace()

    if args.memory_profile is not None:
        args.memory_profile.parent.mkdir(parents=True, exist_ok=True)
        jax.profiler.save_device_memory_profile(str(args.memory_profile))

    hlo_text = rhs_fn.lower(g0).compiler_ir(dialect="hlo").as_hlo_text()
    if args.hlo_out is not None:
        args.hlo_out.parent.mkdir(parents=True, exist_ok=True)
        args.hlo_out.write_text(hlo_text, encoding="utf-8")

    summary = _build_summary(
        config=str(args.config),
        backend=jax.default_backend(),
        nl=int(args.Nl),
        nm=int(args.Nm),
        repeats=int(args.repeats),
        state=str(args.state),
        z_variation_norm=_z_variation_norm(g0),
        compile_execute_seconds=float(compile_execute_seconds),
        warm_seconds=float(warm_seconds),
        rhs_norm=float(np.asarray(jnp.linalg.norm(rhs))),
        phi_norm=float(np.asarray(jnp.linalg.norm(fields.phi))),
        hlo_text=hlo_text,
        trace_dir=args.trace_dir,
        memory_profile=args.memory_profile,
        hlo_out=args.hlo_out,
        force_electrostatic_fields=force_electrostatic_fields,
    )
    _write_summary_json(summary, args.summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
