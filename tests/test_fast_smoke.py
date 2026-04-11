"""Fast smoke tests that exercise core linear/nonlinear paths."""

import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, integrate_linear
from spectraxgk.nonlinear import integrate_nonlinear


def _tiny_case():
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    return cfg, grid, geom, params


def test_linear_smoke_fast():
    """Tiny linear run should return finite fields."""
    _cfg, grid, geom, params = _tiny_case()
    G0 = jnp.ones((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64) * 1.0e-6
    _, phi_t = integrate_linear(G0, grid, geom, params, dt=0.1, steps=2, method="rk2")
    assert jnp.isfinite(phi_t).all()


def test_nonlinear_smoke_fast():
    """Tiny nonlinear run should complete without NaNs."""
    _cfg, grid, geom, params = _tiny_case()
    G0 = jnp.ones((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64) * 1.0e-6
    _, fields = integrate_nonlinear(G0, grid, geom, params, dt=0.1, steps=2, method="rk2")
    assert jnp.isfinite(fields.phi).all()
