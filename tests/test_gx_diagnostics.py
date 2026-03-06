import numpy as np

import jax.numpy as jnp
from dataclasses import replace

from spectraxgk.benchmarks import CycloneBaseCase, _build_initial_condition
from spectraxgk.config import InitializationConfig
from spectraxgk.diagnostics import (
    _jl_family,
    _gx_fac_mask_nonzero,
    gx_Wapar,
    gx_Wg,
    gx_Wphi,
    gx_energy_total,
    gx_heat_flux,
    gx_heat_flux_species,
    gx_particle_flux,
    gx_particle_flux_species,
    gx_volume_factors,
)
from spectraxgk.gyroaverage import gamma0
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.analysis import select_ky_index
from spectraxgk.linear import build_linear_cache, LinearParams, LinearTerms
from spectraxgk.gx_integrators import (
    GXTimeConfig,
    _gx_growth_rate_step,
    integrate_linear_gx_diagnostics,
)
from spectraxgk.species import Species, build_linear_params


def test_gx_flux_fac_nonzero_matches_positive_ky_convention() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(replace(cfg.grid, Ny=8, Nx=4))
    fac = np.asarray(_gx_fac_mask_nonzero(grid, use_dealias=False))
    ky = np.asarray(grid.ky, dtype=float)
    pos = ky > 0.0
    assert np.allclose(fac[pos], 1.0)
    assert np.allclose(fac[~pos], 0.0)


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
    Wphi = gx_Wphi(phi, cache, params, vol_fac)
    Wapar = gx_Wapar(apar, cache, vol_fac)
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


def test_gx_standard_field_energies_match_geometry_weighted_formula():
    _cfg, grid, geom, params, cache = _small_setup()
    vol_fac, _flux_fac = gx_volume_factors(geom, grid)
    phi = jnp.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    apar = (1.0 + 2.0j) * jnp.ones_like(phi)

    fac = np.where(np.asarray(grid.ky, dtype=float)[:, None] == 0.0, 1.0, 2.0)
    fac = fac * np.asarray(grid.dealias_mask, dtype=float)
    weight = fac[:, :, None] * np.asarray(vol_fac, dtype=float)[None, None, :]
    phi2 = np.abs(np.asarray(phi)) ** 2
    apar2 = np.abs(np.asarray(apar)) ** 2

    rho = np.atleast_1d(np.asarray(params.rho, dtype=float))
    wphi_expected = 0.0
    for rho_s in rho:
        b = np.asarray(cache.kperp2, dtype=float) * (rho_s * rho_s)
        wphi_expected += 0.5 * np.sum(phi2 * (1.0 - np.asarray(gamma0(b))) * weight)

    bmag2 = np.asarray(cache.bmag, dtype=float)[None, None, :] ** 2 if cache.kperp2_bmag else 1.0
    wapar_expected = 0.5 * np.sum(
        apar2 * np.asarray(cache.kperp2, dtype=float) * bmag2 * weight
    )

    assert np.allclose(np.asarray(gx_Wphi(phi, cache, params, vol_fac)), wphi_expected)
    assert np.allclose(np.asarray(gx_Wapar(apar, cache, vol_fac)), wapar_expected)


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


def test_gx_jl_family_preserves_species_axis() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None))
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0),
            Species(
                charge=-1.0,
                mass=0.00027,
                density=1.0,
                temperature=1.0,
                tprim=1.0,
                fprim=1.0,
            ),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    Jl, JlB, Jfac = _jl_family(cache)

    assert np.asarray(Jl).shape == np.asarray(cache.Jl).shape
    assert np.asarray(JlB).shape == np.asarray(cache.JlB).shape
    assert np.asarray(Jfac).shape == np.asarray(cache.Jl).shape
    assert np.allclose(np.asarray(Jl), np.asarray(cache.Jl))
    assert np.allclose(np.asarray(JlB), np.asarray(cache.JlB))


def test_gx_particle_flux_species_matches_manual_multispecies_formula() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None))
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0),
            Species(
                charge=-1.0,
                mass=0.00027,
                density=1.0,
                temperature=1.0,
                tprim=1.0,
                fprim=1.0,
            ),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    _vol_fac, flux_fac = gx_volume_factors(geom, grid)

    shape = (2, 3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)
    field_base = np.arange(grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32).reshape(
        grid.ky.size, grid.kx.size, grid.z.size
    )
    phi = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    apar = 0.3 * phi
    bpar = -0.2 * phi

    got = np.asarray(
        gx_particle_flux_species(
            G,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=False,
        )
    )

    fac = np.asarray(_gx_fac_mask_nonzero(grid, use_dealias=False), dtype=np.float32)[:, :, None]
    flx = np.asarray(flux_fac, dtype=np.float32)[None, None, :]
    ky = np.asarray(grid.ky, dtype=np.float32)[:, None, None]
    vphi = 1.0j * ky * np.asarray(phi)
    vapar = 1.0j * ky * np.asarray(apar)
    vbpar = 1.0j * ky * np.asarray(bpar)
    Jl = np.asarray(cache.Jl)
    JlB = np.asarray(cache.JlB)
    dens = np.asarray(params.density, dtype=np.float32)
    vth = np.asarray(params.vth, dtype=np.float32)
    tz = np.asarray(params.tz, dtype=np.float32)
    G_np = np.asarray(G)

    expected = []
    for s in range(2):
        G0 = G_np[s, :, 0, ...]
        G1 = G_np[s, :, 1, ...]
        n_bar = np.sum(Jl[s] * G0, axis=0)
        u_bar = np.sum(Jl[s] * G1, axis=0)
        uB_bar = np.sum(JlB[s] * G0, axis=0)
        fg = np.conj(vphi) * n_bar - vth[s] * np.conj(vapar) * u_bar + tz[s] * np.conj(vbpar) * uB_bar
        expected.append(np.sum((fg * 2.0 * flx * fac).real) * dens[s])

    assert np.allclose(got, np.asarray(expected), rtol=1.0e-6, atol=1.0e-6)


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
