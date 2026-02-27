"""Tests for k_perp hyperdiffusion term."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from spectraxgk.config import GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.terms.assembly import assemble_rhs_cached
from spectraxgk.terms.config import TermConfig


def test_hyperdiffusion_damps_high_k() -> None:
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=4, Lx=6.0, Ly=6.0)
    grid = build_spectral_grid(grid_cfg)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, B0=1.0, alpha=0.0)
    params = LinearParams(D_hyper=0.5, p_hyper_kperp=2.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
    G = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)

    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=1.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=0.0,
    )
    dG, _ = assemble_rhs_cached(G, cache, params, terms=terms)
    dG = dG[0, 0]
    kx = jnp.asarray(grid.kx)
    ky = jnp.asarray(grid.ky)
    kperp2 = (ky[:, None] ** 2 + kx[None, :] ** 2).astype(float)
    mask = np.asarray(grid.dealias_mask, dtype=bool)
    kperp2_np = np.array(kperp2)
    kperp2_np[~mask] = np.nan
    low_idx = np.unravel_index(np.nanargmin(kperp2_np), kperp2_np.shape)
    high_idx = np.unravel_index(np.nanargmax(kperp2_np), kperp2_np.shape)
    low = jnp.abs(dG[low_idx[0], low_idx[1], 0])
    high = jnp.abs(dG[high_idx[0], high_idx[1], 0])
    assert high > low
