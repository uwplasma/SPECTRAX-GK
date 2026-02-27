"""Nonlinear integrator tests."""

import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.nonlinear import (
    build_nonlinear_imex_operator,
    integrate_nonlinear,
    integrate_nonlinear_gx_diagnostics,
    integrate_nonlinear_imex_cached,
)
from spectraxgk.terms.config import TermConfig


def test_integrate_nonlinear_checkpoint_runs():
    """Checkpointed nonlinear integration should run on a tiny grid."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=1.0)
    _, fields_t = integrate_nonlinear(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="rk4",
        terms=terms,
        checkpoint=True,
    )
    assert fields_t.phi.shape[0] == 2


def test_nonlinear_imex_reuses_prebuilt_operator():
    """Prebuilt IMEX operator should be reusable for the same state shape."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    terms = TermConfig(nonlinear=0.0)
    op = build_nonlinear_imex_operator(
        G,
        cache,
        params,
        dt=0.05,
        terms=terms,
        implicit_preconditioner="damping",
    )
    G_out, fields_t = integrate_nonlinear_imex_cached(
        G,
        cache,
        params,
        dt=0.05,
        steps=2,
        terms=terms,
        implicit_operator=op,
    )
    assert G_out.shape == G.shape
    assert fields_t.phi.shape[0] == 2


def test_integrate_nonlinear_gx_diagnostics_shapes():
    """GX-style nonlinear diagnostics should return time-series arrays."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=0.0)
    t, diag = integrate_nonlinear_gx_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=3,
        method="rk3",
        terms=terms,
    )
    assert t.shape[0] == 3
    assert diag.energy_t.shape[0] == 3
