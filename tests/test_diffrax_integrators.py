"""Diffrax integrator smoke tests."""

import pytest
import jax.numpy as jnp

diffrax = pytest.importorskip("diffrax")
pytest.importorskip("equinox")

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.diffrax_integrators import integrate_linear_diffrax, integrate_nonlinear_diffrax
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams
from spectraxgk.terms.config import TermConfig


def test_integrate_linear_diffrax_runs():
    """Diffrax explicit integrator should run on a tiny grid."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="Tsit5",
        progress_bar=False,
        adaptive=True,
        rtol=1.0e-3,
        atol=1.0e-6,
        max_steps=20000,
        jit=False,
    )
    assert phi_t.shape[0] == 2


def test_integrate_nonlinear_diffrax_imex_runs():
    """Diffrax IMEX integrator should run on a tiny grid."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=1.0)
    _, fields_t = integrate_nonlinear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="KenCarp4",
        terms=terms,
        progress_bar=False,
        adaptive=True,
        rtol=1.0e-3,
        atol=1.0e-6,
        max_steps=20000,
        jit=False,
    )
    assert fields_t.phi.shape[0] == 2
