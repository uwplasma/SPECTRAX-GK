"""Fast example smoke tests."""

import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase, GridConfig, TimeConfig
from spectraxgk.geometry import SAlphaGeometry, sample_flux_tube_geometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams
from spectraxgk.runners import integrate_linear_from_config, integrate_nonlinear_from_config
from spectraxgk.terms.config import TermConfig


def test_example_smoke_diffrax():
    """Run a short diffrax solve via the config runner."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(
        t_max=0.2,
        dt=0.1,
        use_diffrax=True,
        diffrax_solver="Tsit5",
        diffrax_adaptive=True,
        diffrax_rtol=1.0e-3,
        diffrax_atol=1.0e-6,
        diffrax_max_steps=20000,
        progress_bar=False,
    )
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear_from_config(G, grid, geom, params, cfg.time)
    assert phi_t.shape[0] == 2


def test_example_smoke_nonlinear_scan():
    """Run a tiny nonlinear scan over two seeds."""
    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", use_diffrax=False)
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    terms = TermConfig(nonlinear=1.0)

    for seed in (0, 1):
        G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
        G = G.at[0, 0, seed, 0, :].set(1.0e-3 + 0.0j)
        _, fields_t = integrate_nonlinear_from_config(
            G,
            grid,
            geom,
            params,
            cfg.time,
            terms=terms,
        )
        assert fields_t.phi.shape[0] == 2


def test_example_smoke_nonlinear_scan_with_sampled_geometry():
    """The nonlinear config runner should accept the sampled geometry contract."""

    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", use_diffrax=False)
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = sample_flux_tube_geometry(SAlphaGeometry.from_config(cfg.geometry), grid.z)
    params = LinearParams()
    terms = TermConfig(nonlinear=1.0)

    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    G = G.at[0, 0, 0, 0, :].set(1.0e-3 + 0.0j)
    _, fields_t = integrate_nonlinear_from_config(
        G,
        grid,
        geom,
        params,
        cfg.time,
        terms=terms,
    )

    assert fields_t.phi.shape[0] == 2
