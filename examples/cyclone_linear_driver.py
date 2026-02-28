#!/usr/bin/env python3
"""JAX/JIT-ready Cyclone linear driver (minimal example)."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
)
from spectraxgk.config import CycloneBaseCase
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, build_linear_cache, _integrate_linear_cached


def main() -> None:
    cfg = CycloneBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    ky_target = 0.3
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - ky_target)))
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        omega_d_scale=CYCLONE_OMEGA_D_SCALE,
        omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
        rho_star=CYCLONE_RHO_STAR,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
    )

    Nl, Nm = 6, 8
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    rng = np.random.default_rng(0)
    G0 = (1.0e-6 * rng.normal(size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size))).astype(
        np.complex64
    )
    G0 = jnp.asarray(G0)

    @jax.jit
    def run_linear(G_init: jnp.ndarray) -> jnp.ndarray:
        _G_final, phi_t = _integrate_linear_cached(
            G_init,
            cache,
            params,
            dt=0.02,
            steps=50,
            method="rk2",
            sample_stride=5,
        )
        return phi_t

    phi_t = run_linear(G0)
    phi2 = jnp.mean(jnp.abs(phi_t) ** 2)
    print(f"mean |phi|^2 = {float(phi2):.6e}")

    # Autodiff-ready: differentiate a simple objective with respect to init amplitude.
    def objective(amp: float) -> jnp.ndarray:
        return jnp.mean(jnp.abs(run_linear(amp * G0)) ** 2)

    grad = jax.grad(objective)(1.0)
    print(f"d objective / d amp = {float(grad):.6e}")


if __name__ == "__main__":
    main()
