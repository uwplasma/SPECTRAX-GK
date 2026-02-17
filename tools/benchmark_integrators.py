"""Compare runtime and host memory for custom vs diffrax integrators."""

from __future__ import annotations

import argparse
import time
import tracemalloc

import jax
import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase
from spectraxgk.diffrax_integrators import integrate_linear_diffrax
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, integrate_linear


def _measure(func, *, repeat: int) -> tuple[float, float]:
    times = []
    peaks = []
    for _ in range(repeat):
        tracemalloc.start()
        t0 = time.perf_counter()
        out = func()
        jax.tree_util.tree_map(lambda x: jax.block_until_ready(x), out)
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(t1 - t0)
        peaks.append(peak / 1.0e6)
    return float(sum(times) / len(times)), float(max(peaks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark custom vs diffrax integrators.")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--diffrax-solver", default="Tsit5")
    parser.add_argument("--diffrax-adaptive", action="store_true")
    parser.add_argument("--diffrax-rtol", type=float, default=1.0e-3)
    parser.add_argument("--diffrax-atol", type=float, default=1.0e-6)
    parser.add_argument("--diffrax-max-steps", type=int, default=20000)
    args = parser.parse_args()

    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    steps = int(round(cfg.time.t_max / cfg.time.dt))
    dt = float(cfg.time.dt)

    G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    G0 = G0.at[0, 0, 0, 0, :].set(1.0e-3 + 0.0j)

    def run_custom():
        return integrate_linear(
            G0,
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            method=cfg.time.method,
        )

    def run_diffrax():
        return integrate_linear_diffrax(
            G0,
            grid,
            geom,
            params,
            dt=dt,
            steps=steps,
            method=args.diffrax_solver,
            adaptive=args.diffrax_adaptive,
            rtol=args.diffrax_rtol,
            atol=args.diffrax_atol,
            max_steps=args.diffrax_max_steps,
            progress_bar=False,
            jit=False,
        )

    for _ in range(max(args.warmup, 0)):
        _ = run_custom()
        _ = run_diffrax()

    custom_time, custom_peak = _measure(run_custom, repeat=args.repeat)
    diffrax_time, diffrax_peak = _measure(run_diffrax, repeat=args.repeat)

    print("Cyclone defaults benchmark")
    print(f"steps={steps} dt={dt} method={cfg.time.method}")
    print(f"custom:  time={custom_time:.3f}s  host_peak={custom_peak:.2f} MB")
    print(f"diffrax: time={diffrax_time:.3f}s  host_peak={diffrax_peak:.2f} MB")
    print("Note: host_peak is Python tracemalloc (device memory not captured).")


if __name__ == "__main__":
    main()
