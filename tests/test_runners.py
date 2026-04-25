"""Config-driven runner tests."""

import jax.numpy as jnp
import pytest

import spectraxgk.runners as runners
from spectraxgk.config import CycloneBaseCase, GridConfig, TimeConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams
from spectraxgk.runners import integrate_linear_from_config, integrate_nonlinear_from_config
from spectraxgk.terms.config import FieldState


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


def test_integrate_nonlinear_from_config_routes_fixed_step_state_sharding(monkeypatch):
    """Non-diffrax nonlinear runs should honor TimeConfig.state_sharding."""

    captured = {}

    def fake_cache(grid, geom, params, nl, nm):
        captured["cache_shape"] = (nl, nm)
        return "cache"

    def fake_sharded(G0, cache, params, **kwargs):
        captured["kwargs"] = kwargs
        captured["cache"] = cache
        return G0 + 1.0, FieldState(phi=jnp.ones((2, 1, 1, 1), dtype=G0.dtype))

    monkeypatch.setattr(runners, "resolve_state_sharding", lambda G0, spec: "mesh" if spec else None)
    monkeypatch.setattr(runners, "build_linear_cache", fake_cache)
    monkeypatch.setattr(runners, "integrate_nonlinear_sharded", fake_sharded)

    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", use_diffrax=False, state_sharding="ky")
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)

    G_out, fields = integrate_nonlinear_from_config(G, grid, geom, params, cfg.time)

    assert captured["cache_shape"] == (2, 2)
    assert captured["cache"] == "cache"
    assert captured["kwargs"]["state_sharding"] == "mesh"
    assert captured["kwargs"]["return_fields"] is True
    assert G_out.shape == G.shape
    assert fields.phi.shape[0] == 2


def test_integrate_nonlinear_from_config_rejects_ungated_z_state_sharding():
    """The config path should not expose exploratory z-FFT sharding as release-grade."""

    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2", use_diffrax=False, state_sharding="z")
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)

    with pytest.raises(ValueError, match="z FFT axis"):
        integrate_nonlinear_from_config(G, grid, geom, params, cfg.time)
