"""Config-driven runner tests."""

import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase, GridConfig, TimeConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams
from spectraxgk.runners import integrate_linear_from_config


def test_integrate_linear_from_config():
    """TimeConfig should map into the linear integrator."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", use_diffrax=False)
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear_from_config(G, grid, geom, params, cfg.time)
    assert phi_t.shape[0] == 2
