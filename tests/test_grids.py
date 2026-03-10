"""Spectral grid construction tests."""

import jax.numpy as jnp

from spectraxgk.config import GridConfig
from spectraxgk.grids import (
    SpectralGrid,
    build_spectral_grid,
    gx_real_fft_kx,
    gx_real_fft_ky,
    select_gx_real_fft_ky_grid,
)


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


def test_spectral_grid_tree_roundtrip():
    """SpectralGrid pytree should round-trip through flatten/unflatten."""
    cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=2.0, Ly=2.0)
    grid = build_spectral_grid(cfg)
    children, aux = grid.tree_flatten()
    grid2 = SpectralGrid.tree_unflatten(aux, children)
    assert jnp.allclose(grid2.kx, grid.kx)
    assert jnp.allclose(grid2.ky, grid.ky)
    assert jnp.allclose(grid2.z, grid.z)


def test_grid_config_y0_and_ntheta():
    """Field-aligned grid inputs should map to expected ky and z spacing."""
    cfg = GridConfig(Nx=4, Ny=12, Nz=4, Lx=2.0, Ly=3.0, y0=20.0, ntheta=8, nperiod=2)
    grid = build_spectral_grid(cfg)
    assert grid.z.shape[0] == 8 * 3
    dz = grid.z[1] - grid.z[0]
    assert jnp.isclose(dz, 2.0 * jnp.pi / 8.0)
    dky = grid.ky[1] - grid.ky[0]
    assert jnp.isclose(dky, 1.0 / 20.0)


def test_grid_config_ntheta_default_zp():
    """ntheta without nperiod should default to Zp=1."""
    cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=2.0, Ly=3.0, ntheta=6)
    grid = build_spectral_grid(cfg)
    assert grid.z.shape[0] == 6
    assert jnp.isclose(grid.z[0], -jnp.pi)
    dz = grid.z[1] - grid.z[0]
    assert jnp.isclose(dz, 2.0 * jnp.pi / 6.0)


def test_grid_config_explicit_zp():
    """Explicit Zp should override nperiod when provided."""
    cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=2.0, Ly=3.0, ntheta=5, zp=3)
    grid = build_spectral_grid(cfg)
    assert grid.z.shape[0] == 15
    assert jnp.isclose(grid.z[0], -jnp.pi * 3.0)


def test_gx_real_fft_wavenumbers_match_gx_native_layout():
    """GX real-FFT helpers should expose positive Nyquist multipliers."""

    cfg = GridConfig(Nx=4, Ny=10, Nz=4, Lx=2.0, Ly=20.0)
    grid = build_spectral_grid(cfg)
    dkx = 2.0 * jnp.pi / cfg.Lx
    dky = 2.0 * jnp.pi / cfg.Ly
    assert jnp.allclose(gx_real_fft_kx(grid.kx), jnp.asarray([0.0, dkx, 2.0 * dkx, -dkx]))
    assert jnp.allclose(
        gx_real_fft_ky(grid.ky),
        jnp.asarray([0.0, dky, 2.0 * dky, 3.0 * dky, 4.0 * dky, 5.0 * dky]),
    )


def test_select_gx_real_fft_ky_grid_uses_explicit_positive_dump_values():
    """GX dump grids should not inherit the negative Nyquist sign from fftfreq order."""

    cfg = GridConfig(Nx=4, Ny=6, Nz=4, Lx=2.0, Ly=6.0)
    grid = build_spectral_grid(cfg)
    gx_ky = jnp.asarray([0.0, 2.0 * jnp.pi / cfg.Ly, 2.0 * 2.0 * jnp.pi / cfg.Ly, 3.0 * 2.0 * jnp.pi / cfg.Ly])
    gx_grid = select_gx_real_fft_ky_grid(grid, gx_ky)

    assert jnp.allclose(gx_grid.ky, gx_ky)
    assert jnp.all(gx_grid.ky >= 0.0)
    assert jnp.allclose(gx_grid.kx, gx_real_fft_kx(grid.kx))
    assert gx_grid.dealias_mask.shape == (gx_ky.shape[0], cfg.Nx)
    assert jnp.allclose(gx_grid.ky_grid[:, 0], gx_ky)
