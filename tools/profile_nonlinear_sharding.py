#!/usr/bin/env python3
"""Profile fixed-step nonlinear state sharding with a numerical identity gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import statistics
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


def _time_repeated(fn: Any, *, warmups: int, repeats: int) -> tuple[Any, list[float]]:
    for _ in range(int(warmups)):
        _block_until_ready(fn())
    last: Any = None
    times: list[float] = []
    for _ in range(int(repeats)):
        last, elapsed = _time_call(fn)
        times.append(float(elapsed))
    return last, times


def _time_stats(times: list[float]) -> dict[str, float]:
    if not times:
        raise ValueError("at least one timing sample is required")
    return {
        "min": float(min(times)),
        "median": float(statistics.median(times)),
        "mean": float(statistics.fmean(times)),
        "max": float(max(times)),
        "std": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
    }


def _sharding_specs(primary: str, extra: str | None) -> list[str]:
    raw = extra if extra is not None else primary
    specs: list[str] = []
    for item in str(raw).split(","):
        key = item.strip()
        if key and key not in specs:
            specs.append(key)
    if primary not in specs:
        specs.insert(0, primary)
    return specs


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
    parser.add_argument(
        "--sharding-options",
        default=None,
        help=(
            "Comma-separated candidate nonlinear state-decomposition axes to gate. "
            "Use auto,kx for the release-gated path; z is exploratory FFT-axis decomposition."
        ),
    )
    parser.add_argument("--laguerre-mode", default="grid")
    parser.add_argument("--amplitude", type=float, default=1.0e-4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Optional JAX profiler trace directory. The trace is not a runtime claim; it localizes hot paths.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    G0, cache, params = _build_problem(args)
    G0_host = np.asarray(G0)
    terms = TermConfig(nonlinear=1.0, collisions=0.0, hypercollisions=0.0, apar=0.0, bpar=0.0)
    sharding_specs = _sharding_specs(str(args.sharding), args.sharding_options)

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

    def make_sharded_run(spec: str):
        state_sharding = resolve_state_sharding(G0, spec)

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

        return state_sharding, sharded_run

    trace_status: dict[str, Any] = {"requested": args.trace_dir is not None, "path": None, "error": None}
    sharded_fns: dict[str, Any] = {}
    sharded_state_active: dict[str, bool] = {}
    for spec in sharding_specs:
        state_sharding, fn = make_sharded_run(spec)
        sharded_fns[spec] = fn
        sharded_state_active[spec] = state_sharding is not None

    if args.trace_dir is not None:
        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_status["path"] = str(trace_dir)
        try:
            jax.profiler.start_trace(str(trace_dir))
            _block_until_ready(serial_run())
            for fn in sharded_fns.values():
                _block_until_ready(fn())
            jax.profiler.stop_trace()
        except Exception as exc:  # pragma: no cover - profiler availability is platform-specific.
            trace_status["error"] = repr(exc)
            try:
                jax.profiler.stop_trace()
            except Exception:
                pass

    serial_final, serial_times = _time_repeated(serial_run, warmups=int(args.warmups), repeats=int(args.repeats))
    serial_stats = _time_stats(serial_times)

    scale = max(float(np.max(np.abs(np.asarray(serial_final)))), 1.0e-30)
    sharded_results: dict[str, dict[str, Any]] = {}
    identity_passes: list[bool] = []
    for spec, fn in sharded_fns.items():
        try:
            sharded_final, sharded_times = _time_repeated(fn, warmups=int(args.warmups), repeats=int(args.repeats))
            sharded_stats = _time_stats(sharded_times)
            err = np.asarray(sharded_final - serial_final)
            max_abs = float(np.max(np.abs(err)))
            max_rel = float(max_abs / scale)
            identity_pass = bool(max_abs <= 1.0e-5 and max_rel <= 1.0e-5)
            sharded_results[spec] = {
                "state_sharding_active": bool(sharded_state_active[spec]),
                "times_s": sharded_times,
                "stats_s": sharded_stats,
                "engineering_speedup_median": float(serial_stats["median"] / sharded_stats["median"])
                if sharded_stats["median"] > 0.0
                else None,
                "max_abs_state_error": max_abs,
                "max_rel_state_error": max_rel,
                "identity_gate_pass": identity_pass,
                "error": None,
            }
        except Exception as exc:
            identity_pass = False
            sharded_results[spec] = {
                "state_sharding_active": bool(sharded_state_active[spec]),
                "times_s": [],
                "stats_s": None,
                "engineering_speedup_median": None,
                "max_abs_state_error": None,
                "max_rel_state_error": None,
                "identity_gate_pass": False,
                "error": repr(exc),
            }
        identity_passes.append(identity_pass)

    primary = sharded_results[str(args.sharding)]
    payload = {
        "case": "cyclone_nonlinear_fixed_step",
        "device_count": int(jax.device_count()),
        "devices": [str(device) for device in jax.devices()],
        "default_backend": str(jax.default_backend()),
        "state_shape": list(map(int, G0.shape)),
        "state_sharding_requested": str(args.sharding),
        "state_sharding_active": bool(primary["state_sharding_active"]),
        "sharding_options": sharding_specs,
        "dt": float(args.dt),
        "steps": int(args.steps),
        "method": str(args.method),
        "laguerre_mode": str(args.laguerre_mode),
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "serial_times_s": serial_times,
        "serial_stats_s": serial_stats,
        "sharded_results": sharded_results,
        "profiler_trace": trace_status,
        "serial_warm_s": serial_stats["median"],
        "sharded_warm_s": primary["stats_s"]["median"] if primary["stats_s"] else None,
        "engineering_speedup": primary["engineering_speedup_median"],
        "max_abs_state_error": primary["max_abs_state_error"],
        "max_rel_state_error": primary["max_rel_state_error"],
        "identity_gate_pass": bool(all(identity_passes)),
        "claim_scope": (
            "Profiler/identity artifact for fixed-step nonlinear state sharding and candidate "
            "state-axis decompositions. Do not use as a published runtime claim without a larger "
            "matched CPU/GPU sweep and profiler trace review."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["identity_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
