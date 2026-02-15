"""Profile cached vs uncached linear RHS evaluation."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, build_linear_cache, linear_rhs, linear_rhs_cached


def time_call(fn, G, repeats: int) -> float:
    for _ in range(2):
        fn(G).block_until_ready()
    start = time.perf_counter()
    for _ in range(repeats):
        fn(G).block_until_ready()
    end = time.perf_counter()
    return (end - start) / repeats


def main() -> None:
    cfg = CycloneBaseCase(grid=GridConfig(Nx=16, Ny=16, Nz=32, Lx=62.8, Ly=62.8))
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        omega_d_scale=0.32,
        omega_star_scale=1.0,
    )

    Nl, Nm = 2, 4
    G0 = jnp.zeros((Nl, Nm, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    rhs_uncached = jax.jit(lambda G: linear_rhs(G, grid, geom, params)[0])
    rhs_cached = jax.jit(lambda G: linear_rhs_cached(G, cache, params)[0])

    uncached_time = time_call(rhs_uncached, G0, repeats=20)
    cached_time = time_call(rhs_cached, G0, repeats=20)
    speedup = uncached_time / cached_time if cached_time > 0 else float("inf")

    print(f"uncached_s={uncached_time:.6f}")
    print(f"cached_s={cached_time:.6f}")
    print(f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
