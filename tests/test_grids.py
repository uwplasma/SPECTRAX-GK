"""Spectral grid construction tests."""

import jax.numpy as jnp

from spectraxgk.config import GridConfig
from spectraxgk.grids import build_spectral_grid


def test_build_spectral_grid_shapes():
    """Grid arrays should have consistent shapes."""
    cfg = GridConfig(Nx=8, Ny=6, Nz=4, Lx=2.0, Ly=3.0)
    grid = build_spectral_grid(cfg)
    assert grid.kx.shape == (cfg.Nx,)
    assert grid.ky.shape == (cfg.Ny,)
    assert grid.z.shape == (cfg.Nz,)
    assert grid.kx_grid.shape == (cfg.Ny, cfg.Nx)
    assert grid.ky_grid.shape == (cfg.Ny, cfg.Nx)
    assert grid.dealias_mask.shape == (cfg.Ny, cfg.Nx)


def test_build_spectral_grid_spacing():
    """Fourier spacing should match 2*pi/L for each direction."""
    cfg = GridConfig(Nx=8, Ny=6, Nz=4, Lx=2.0, Ly=3.0)
    grid = build_spectral_grid(cfg)
    dkx = grid.kx[1] - grid.kx[0]
    dky = grid.ky[1] - grid.ky[0]
    assert jnp.isclose(dkx, 2.0 * jnp.pi / cfg.Lx)
    assert jnp.isclose(dky, 2.0 * jnp.pi / cfg.Ly)
