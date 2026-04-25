#!/usr/bin/env python3
"""Profile fixed-step nonlinear state sharding with a numerical identity gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.nonlinear import integrate_nonlinear_cached
from spectraxgk.sharded_integrators import integrate_nonlinear_sharded
from spectraxgk.sharding import resolve_state_sharding
from spectraxgk.terms.config import TermConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "nonlinear_sharding_profile.json"


def _block_until_ready(value: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _build_problem(args: argparse.Namespace) -> tuple[jnp.ndarray, Any, LinearParams]:
    grid_cfg = GridConfig(
        Nx=int(args.nx),
        Ny=int(args.ny),
        Nz=int(args.nz),
        Lx=6.0,
        Ly=6.0,
        boundary="periodic",
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    shape = (int(args.nl), int(args.nm), grid.ky.size, grid.kx.size, grid.z.size)
    G0 = jnp.zeros(shape, dtype=jnp.complex64)
    G0 = G0.at[0, 0, 0, 0, :].set(jnp.asarray(args.amplitude, dtype=G0.dtype))
    cache = build_linear_cache(grid, geom, params, int(args.nl), int(args.nm))
    return G0, cache, params


def _time_call(fn: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    out = _block_until_ready(fn())
    return out, time.perf_counter() - start


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--method", default="rk2")
    parser.add_argument("--sharding", default="auto")
    parser.add_argument("--laguerre-mode", default="grid")
    parser.add_argument("--amplitude", type=float, default=1.0e-4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    G0, cache, params = _build_problem(args)
    G0_host = np.asarray(G0)
    terms = TermConfig(nonlinear=1.0, collisions=0.0, hypercollisions=0.0, apar=0.0, bpar=0.0)
    state_sharding = resolve_state_sharding(G0, args.sharding)

    def serial_run():
        G_final, _fields = integrate_nonlinear_cached(
            jnp.asarray(G0_host),
            cache,
            params,
            dt=float(args.dt),
            steps=int(args.steps),
            method=str(args.method),
            terms=terms,
            gx_real_fft=True,
            laguerre_mode=str(args.laguerre_mode),
        )
        return G_final

    def sharded_run():
        return integrate_nonlinear_sharded(
            jnp.asarray(G0_host),
            cache,
            params,
            dt=float(args.dt),
            steps=int(args.steps),
            method=str(args.method),
            terms=terms,
            state_sharding=state_sharding,
            gx_real_fft=True,
            laguerre_mode=str(args.laguerre_mode),
            return_fields=False,
        )

    _block_until_ready(serial_run())
    _block_until_ready(sharded_run())
    serial_final, serial_s = _time_call(serial_run)
    sharded_final, sharded_s = _time_call(sharded_run)

    err = np.asarray(sharded_final - serial_final)
    scale = max(float(np.max(np.abs(np.asarray(serial_final)))), 1.0e-30)
    max_abs = float(np.max(np.abs(err)))
    max_rel = float(max_abs / scale)
    payload = {
        "case": "cyclone_nonlinear_fixed_step",
        "device_count": int(jax.device_count()),
        "devices": [str(device) for device in jax.devices()],
        "state_shape": list(map(int, G0.shape)),
        "state_sharding_requested": str(args.sharding),
        "state_sharding_active": state_sharding is not None,
        "dt": float(args.dt),
        "steps": int(args.steps),
        "method": str(args.method),
        "laguerre_mode": str(args.laguerre_mode),
        "serial_warm_s": serial_s,
        "sharded_warm_s": sharded_s,
        "engineering_speedup": float(serial_s / sharded_s) if sharded_s > 0.0 else None,
        "max_abs_state_error": max_abs,
        "max_rel_state_error": max_rel,
        "identity_gate_pass": bool(max_abs <= 1.0e-5 and max_rel <= 1.0e-5),
        "claim_scope": (
            "Profiler/identity artifact for fixed-step nonlinear state sharding. "
            "Do not use as a published runtime claim without a larger matched CPU/GPU sweep."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["identity_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
