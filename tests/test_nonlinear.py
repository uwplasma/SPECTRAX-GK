"""Nonlinear integrator tests."""

import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams
from spectraxgk.nonlinear import integrate_nonlinear
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
