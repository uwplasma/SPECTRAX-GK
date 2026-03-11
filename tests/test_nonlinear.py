"""Nonlinear integrator tests."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry, ensure_flux_tube_geometry_data
from spectraxgk.gx_integrators import _gx_linear_omega_max
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.nonlinear import (
    _apply_collision_split,
    _collision_damping,
    build_nonlinear_imex_operator,
    integrate_nonlinear,
    integrate_nonlinear_gx_diagnostics,
    integrate_nonlinear_gx_diagnostics_state,
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
        method="sspx3",
        terms=terms,
    )
    assert t.shape[0] == 3
    assert diag.energy_t.shape[0] == 3
    assert diag.heat_flux_species_t is not None
    assert diag.particle_flux_species_t is not None
    assert np.asarray(diag.heat_flux_species_t).shape == (3, 1)
    assert np.asarray(diag.particle_flux_species_t).shape == (3, 1)
    assert np.isfinite(np.asarray(diag.dt_mean))
    assert np.isfinite(np.asarray(diag.dt_t)).all()


def test_integrate_nonlinear_imex_gx_diagnostics_shapes():
    """IMEX nonlinear diagnostics should return time-series arrays."""

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
        dt=0.05,
        steps=2,
        method="imex",
        terms=terms,
    )
    assert t.shape[0] == 2
    assert diag.energy_t.shape[0] == 2
    assert diag.heat_flux_species_t is not None
    assert diag.particle_flux_species_t is not None
    assert np.asarray(diag.heat_flux_species_t).shape == (2, 1)
    assert np.asarray(diag.particle_flux_species_t).shape == (2, 1)


def test_integrate_nonlinear_collision_split_sts():
    """Collision split with STS scheme should run and remain finite."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=0.0, collisions=1.0)
    _t, diag = integrate_nonlinear_gx_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="rk2",
        terms=terms,
        collision_split=True,
        collision_scheme="sts",
    )
    assert np.isfinite(np.asarray(diag.Wg_t)).all()


def test_nonlinear_collision_split_does_not_double_count_explicit_collisions():
    """Explicit nonlinear diagnostics path should remove split collisions from the RHS."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(nu=0.2)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    G = jnp.asarray(
        np.linspace(1.0, 1.0 + 2 * 2 * 2 * 2 * 4 - 1, 2 * 2 * 2 * 2 * 4, dtype=np.float32).reshape(2, 2, 2, 2, 4),
        dtype=jnp.complex64,
    )
    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        nonlinear=0.0,
        apar=0.0,
        bpar=0.0,
    )

    _t, _diag, G_final, _fields = integrate_nonlinear_gx_diagnostics_state(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="rk3",
        cache=cache,
        terms=terms,
        collision_split=True,
        collision_scheme="exp",
    )

    damping = _collision_damping(cache, params, terms, jnp.float32, squeeze_species=True)
    expected = _apply_collision_split(G, damping, jnp.asarray(0.05, dtype=jnp.float32), "exp")

    np.testing.assert_allclose(np.asarray(G_final), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6)


def test_nonlinear_gx_adaptive_default_dt_max_matches_gx():
    """Adaptive nonlinear GX diagnostics should clamp dt to dt when dt_max is unset."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=0.0)
    _t, diag = integrate_nonlinear_gx_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=3,
        method="rk3",
        terms=terms,
        fixed_dt=False,
        dt_max=None,
        cfl=10.0,
    )
    dt_t = np.asarray(diag.dt_t, dtype=float)
    assert dt_t.size > 0
    assert np.nanmax(dt_t) <= 0.05 + 1.0e-6


def test_nonlinear_gx_adaptive_dt_includes_linear_frequency_cap():
    """Adaptive nonlinear dt should honor the GX linear CFL estimate even with zero nonlinear drive."""

    grid_cfg = GridConfig(Nx=8, Ny=8, Nz=16, Lx=20.0, Ly=20.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    params = LinearParams(R_over_LTi=3.0, R_over_Ln=1.0)
    G = jnp.zeros((2, 4, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=0.0)

    cfl = 0.5
    cfl_fac = 1.73
    dt0 = 0.1
    cache = build_linear_cache(grid, geom_eff, params, Nl=2, Nm=4)
    linear_omega = _gx_linear_omega_max(
        grid,
        geom_eff,
        params,
        nl=int(cache.l.shape[0]),
        nm=int(cache.m.shape[1]),
        include_diamagnetic_drive=False,
    )
    expected_dt = cfl_fac * cfl / float(np.sum(linear_omega))

    _t, diag = integrate_nonlinear_gx_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=dt0,
        steps=2,
        method="rk3",
        terms=terms,
        fixed_dt=False,
        dt_max=dt0,
        cfl=cfl,
        cfl_fac=cfl_fac,
    )

    dt_t = np.asarray(diag.dt_t, dtype=float)
    assert dt_t.size > 0
    assert dt_t[0] == pytest.approx(expected_dt, rel=1.0e-5, abs=1.0e-8)
    assert dt_t[0] < dt0


@pytest.mark.parametrize("method", ["rk3", "imex"])
def test_nonlinear_gx_gamma_omega_use_previous_step_not_previous_diagnostic(method: str):
    """GX-style nonlinear gamma/omega should be invariant to diagnostics_stride."""

    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    shape = (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)
    terms = TermConfig(nonlinear=0.0)

    t_dense, diag_dense = integrate_nonlinear_gx_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.02,
        steps=4,
        method=method,
        terms=terms,
        sample_stride=1,
        diagnostics_stride=1,
    )
    t_sparse, diag_sparse = integrate_nonlinear_gx_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.02,
        steps=4,
        method=method,
        terms=terms,
        sample_stride=1,
        diagnostics_stride=2,
    )

    assert np.allclose(np.asarray(t_dense)[::2], np.asarray(t_sparse))
    assert np.allclose(np.asarray(diag_dense.gamma_t)[::2], np.asarray(diag_sparse.gamma_t))
    assert np.allclose(np.asarray(diag_dense.omega_t)[::2], np.asarray(diag_sparse.omega_t))


def test_nonlinear_imex_gx_diagnostics_match_operator_dtype_under_x64():
    """IMEX GX diagnostics should keep the scan state dtype aligned with the implicit operator."""

    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)

    with jax.enable_x64():
        grid = build_spectral_grid(cfg.grid)
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = LinearParams()
        shape = (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)

        _t, diag = integrate_nonlinear_gx_diagnostics(
            G,
            grid,
            geom,
            params,
            dt=0.02,
            steps=2,
            method="imex",
            terms=TermConfig(nonlinear=0.0),
            sample_stride=1,
            diagnostics_stride=1,
        )

    assert np.isfinite(np.asarray(diag.gamma_t)).all()
    assert np.isfinite(np.asarray(diag.omega_t)).all()


@pytest.mark.parametrize("method", ["rk3", "sspx3"])
def test_nonlinear_gx_state_diagnostics_can_freeze_one_mode(method: str):
    """Fixed-mode projection should preserve a selected Fourier mode exactly."""

    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    shape = (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)

    _t, _diag, G_final, _fields = integrate_nonlinear_gx_diagnostics_state(
        G,
        grid,
        geom,
        params,
        dt=0.02,
        steps=3,
        method=method,
        terms=TermConfig(nonlinear=1.0, collisions=0.0, hypercollisions=0.0),
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
    )

    assert np.allclose(np.asarray(G_final)[..., 1, 0, :], np.asarray(G)[..., 1, 0, :])
