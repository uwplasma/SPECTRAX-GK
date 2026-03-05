import numpy as np

import jax.numpy as jnp
from dataclasses import replace

from spectraxgk.benchmarks import CycloneBaseCase, _build_initial_condition
from spectraxgk.config import InitializationConfig
from spectraxgk.diagnostics import (
    gx_Wapar_krehm,
    gx_Wg,
    gx_Wphi_krehm,
    gx_energy_total,
    gx_heat_flux,
    gx_heat_flux_species,
    gx_particle_flux,
    gx_particle_flux_species,
    gx_volume_factors,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.analysis import select_ky_index
from spectraxgk.linear import build_linear_cache, LinearParams, LinearTerms
from spectraxgk.gx_integrators import (
    GXTimeConfig,
    _gx_growth_rate_step,
    integrate_linear_gx_diagnostics,
)


def _small_setup():
    cfg = CycloneBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), 0.2)
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
    )
    cache = build_linear_cache(grid, geom, params, 4, 4)
    return cfg, grid, geom, params, cache


def test_gx_energy_components_finite():
    cfg, grid, geom, params, cache = _small_setup()
    vol_fac, flux_fac = gx_volume_factors(geom, grid)
    G0 = _build_initial_condition(grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init)
    _, fields = cache, None
    # Build dummy fields from RHS for consistent shapes
    from spectraxgk.terms.assembly import assemble_rhs_cached

    _dG, fields = assemble_rhs_cached(G0, cache, params, terms=LinearTerms())
    phi = fields.phi
    apar = fields.apar if fields.apar is not None else jnp.zeros_like(phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(phi)

    Wg = gx_Wg(G0, grid, params, vol_fac)
    Wphi = gx_Wphi_krehm(phi, grid, params, vol_fac)
    Wapar = gx_Wapar_krehm(apar, grid)
    heat = gx_heat_flux(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    pflux = gx_particle_flux(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    energy = gx_energy_total(Wg, Wphi, Wapar)

    assert np.isfinite(np.asarray(Wg))
    assert np.isfinite(np.asarray(Wphi))
    assert np.isfinite(np.asarray(Wapar))
    assert np.isfinite(np.asarray(heat))
    assert np.isfinite(np.asarray(pflux))
    assert np.isfinite(np.asarray(energy))
    assert energy == Wg + Wphi + Wapar


def test_gx_species_flux_sums_to_total():
    cfg, grid, geom, params, cache = _small_setup()
    _vol_fac, flux_fac = gx_volume_factors(geom, grid)
    G0 = _build_initial_condition(grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init)
    from spectraxgk.terms.assembly import assemble_rhs_cached

    _dG, fields = assemble_rhs_cached(G0, cache, params, terms=LinearTerms())
    phi = fields.phi
    apar = fields.apar if fields.apar is not None else jnp.zeros_like(phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(phi)

    heat_s = gx_heat_flux_species(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    pflux_s = gx_particle_flux_species(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    heat = gx_heat_flux(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    pflux = gx_particle_flux(G0, phi, apar, bpar, cache, grid, params, flux_fac)

    assert heat_s.shape == (1,)
    assert pflux_s.shape == (1,)
    assert np.allclose(np.asarray(jnp.sum(heat_s)), np.asarray(heat))
    assert np.allclose(np.asarray(jnp.sum(pflux_s)), np.asarray(pflux))


def test_gx_init_all_scaling_matches_reference():
    cfg = CycloneBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), float(grid_full.ky[1]))
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    init_cfg = InitializationConfig(
        init_field="all",
        init_amp=1.0,
        gaussian_init=False,
    )
    G0 = _build_initial_condition(grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=init_cfg)
    base = 1.0 + 1.0j
    # density (l=0,m=0) should be unscaled
    assert np.allclose(G0[0, 0, 0, 0, 0], base)
    # tpar (l=0,m=2) scaled by 1/sqrt(2)
    assert np.allclose(G0[0, 2, 0, 0, 0], base / np.sqrt(2.0))
    # qpar (l=0,m=3) scaled by 1/sqrt(6)
    assert np.allclose(G0[0, 3, 0, 0, 0], base / np.sqrt(6.0))


def test_integrate_linear_gx_diagnostics_shapes():
    cfg, grid, geom, params, cache = _small_setup()
    G0 = _build_initial_condition(grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init)
    time_cfg = GXTimeConfig(dt=0.01, t_max=0.1, sample_stride=1, fixed_dt=True)

    t, phi_t, gamma_t, omega_t, diag = integrate_linear_gx_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=LinearTerms(),
        jit=False,
    )
    assert t.shape[0] == phi_t.shape[0] == gamma_t.shape[0] == omega_t.shape[0]
    assert diag.t.shape[0] == t.shape[0]
    assert diag.Wg_t.shape[0] == t.shape[0]
    assert diag.Wphi_t.shape[0] == t.shape[0]
    assert diag.Wapar_t.shape[0] == t.shape[0]
    assert diag.heat_flux_t.shape[0] == t.shape[0]
    assert diag.particle_flux_t.shape[0] == t.shape[0]


def test_gx_energy_drift_small_no_drive():
    cfg, grid, geom, params, cache = _small_setup()
    params = replace(params, R_over_Ln=0.0, R_over_LTi=0.0, R_over_LTe=0.0, nu=0.0)
    G0 = _build_initial_condition(grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init)
    time_cfg = GXTimeConfig(dt=0.01, t_max=0.2, sample_stride=1, fixed_dt=True)

    _, _, _, _, diag = integrate_linear_gx_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=LinearTerms(streaming=1.0, mirror=0.0, curvature=0.0, gradb=0.0, diamagnetic=0.0, collisions=0.0, hypercollisions=0.0, end_damping=0.0, apar=0.0, bpar=0.0),
        jit=False,
    )
    energy = np.asarray(diag.energy_t)
    assert np.all(np.isfinite(energy))
    if energy.size > 1:
        rel = np.abs((energy[-1] - energy[0]) / max(abs(energy[0]), 1.0e-12))
        assert rel < 0.05


def test_gx_growth_rate_step_matches_real_imag_validity_mask():
    """GX growth-rate kernel should require non-zero real and imaginary parts."""

    phi_prev = jnp.asarray([[[1.0 + 1.0j, 1.0 + 1.0j]]], dtype=jnp.complex64)
    phi_now_invalid = jnp.asarray([[[2.0 + 0.0j, 2.0 + 0.0j]]], dtype=jnp.complex64)
    mask = jnp.asarray([[True]])
    gamma_bad, omega_bad = _gx_growth_rate_step(
        phi_now_invalid, phi_prev, 0.1, z_index=0, mask=mask
    )
    assert np.allclose(np.asarray(gamma_bad), 0.0)
    assert np.allclose(np.asarray(omega_bad), 0.0)

    phi_now_valid = jnp.asarray([[[2.0 + 2.0j, 2.0 + 2.0j]]], dtype=jnp.complex64)
    gamma_ok, omega_ok = _gx_growth_rate_step(phi_now_valid, phi_prev, 0.1, z_index=0, mask=mask)
    assert np.isfinite(np.asarray(gamma_ok)).all()
    assert np.isfinite(np.asarray(omega_ok)).all()
    assert not np.allclose(np.asarray(gamma_ok), 0.0)


def test_gx_growth_rate_step_validity_depends_on_current_phi_only():
    """GX kernel checks real/imag nonzero on current phi only."""

    phi_prev = jnp.asarray([[[1.0 + 0.0j, 1.0 + 0.0j]]], dtype=jnp.complex64)
    phi_now = jnp.asarray([[[2.0 + 2.0j, 2.0 + 2.0j]]], dtype=jnp.complex64)
    mask = jnp.asarray([[True]])
    gamma, omega = _gx_growth_rate_step(phi_now, phi_prev, 0.1, z_index=0, mask=mask)
    assert np.isfinite(np.asarray(gamma)).all()
    assert np.isfinite(np.asarray(omega)).all()
    assert not np.allclose(np.asarray(gamma), 0.0)


def test_linear_gx_adaptive_default_dt_max_matches_gx():
    """When dt_max is unset, adaptive GX path should clamp to dt."""

    cfg, grid, geom, params, cache = _small_setup()
    G0 = _build_initial_condition(grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init)
    time_cfg = GXTimeConfig(
        dt=0.01,
        t_max=0.05,
        sample_stride=1,
        fixed_dt=False,
        dt_max=None,
        cfl=10.0,
    )
    _t, _phi_t, _gamma_t, _omega_t, diag = integrate_linear_gx_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=LinearTerms(),
        jit=False,
    )
    dt_t = np.asarray(diag.dt_t, dtype=float)
    assert dt_t.size > 0
    assert np.nanmax(dt_t) <= float(time_cfg.dt) + 1.0e-12
