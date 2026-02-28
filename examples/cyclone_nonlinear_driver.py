#!/usr/bin/env python3
"""JAX/JIT-ready Cyclone nonlinear driver (minimal example)."""

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
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.nonlinear import integrate_nonlinear_cached
from spectraxgk.terms.config import TermConfig


def main() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
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

    Nl, Nm = 4, 6
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    term_cfg = TermConfig(nonlinear=1.0, apar=0.0, bpar=0.0)

    rng = np.random.default_rng(1)
    G0 = (1.0e-4 * rng.normal(size=(1, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size))).astype(
        np.complex64
    )
    G0 = jnp.asarray(G0)

    @jax.jit
    def run_nonlinear(G_init: jnp.ndarray) -> jnp.ndarray:
        _G_final, fields = integrate_nonlinear_cached(
            G_init,
            cache,
            params,
            dt=0.02,
            steps=20,
            method="rk2",
            terms=term_cfg,
            gx_real_fft=True,
        )
        return fields.phi

    phi = run_nonlinear(G0)
    phi2 = jnp.mean(jnp.abs(phi) ** 2)
    print(f"final |phi|^2 = {float(phi2):.6e}")

    def objective(amp: float) -> jnp.ndarray:
        return jnp.mean(jnp.abs(run_nonlinear(amp * G0)) ** 2)

    grad = jax.grad(objective)(1.0)
    print(f"d objective / d amp = {float(grad):.6e}")


if __name__ == "__main__":
    main()
