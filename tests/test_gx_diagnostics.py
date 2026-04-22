import numpy as np

import jax.numpy as jnp
from dataclasses import replace
import pytest
import spectraxgk.gx_integrators as gx_integrators
from types import SimpleNamespace

from spectraxgk.benchmarks import CycloneBaseCase, _build_initial_condition
from spectraxgk.config import InitializationConfig
from spectraxgk.diagnostics import (
    _jl_family,
    _gx_fac_mask_nonzero,
    gx_Wapar,
    gx_Wg,
    gx_Wphi,
    total_energy,
    gx_heat_flux,
    gx_heat_flux_resolved_species,
    gx_heat_flux_split_resolved_species,
    gx_heat_flux_split_species,
    gx_heat_flux_species,
    gx_particle_flux,
    gx_particle_flux_resolved_species,
    gx_particle_flux_split_resolved_species,
    gx_particle_flux_split_species,
    gx_particle_flux_species,
    gx_phi_zonal_mode_kxt,
    gx_volume_factors,
    gx_turbulent_heating,
    gx_turbulent_heating_resolved_species,
    gx_turbulent_heating_species,
)
from spectraxgk.gyroaverage import gamma0
from spectraxgk.geometry import SAlphaGeometry, sample_flux_tube_geometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.analysis import select_ky_index
from spectraxgk.linear import (
    build_linear_cache,
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.gx_integrators import (
    ExplicitTimeConfig,
    _gx_growth_mask,
    _gx_linear_omega_max,
    _gx_growth_rate_step,
    _rk4_step,
    integrate_linear_gx_diagnostics,
)
from spectraxgk.species import Species, build_linear_params
from spectraxgk.terms.assembly import assemble_rhs_cached
from spectraxgk.terms.config import FieldState


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
    energy = total_energy(Wg, Wphi, Wapar)

    assert np.isfinite(np.asarray(Wg))
    assert np.isfinite(np.asarray(Wphi))
    assert np.isfinite(np.asarray(Wapar))
    assert np.isfinite(np.asarray(heat))
    assert np.isfinite(np.asarray(pflux))
    assert np.isfinite(np.asarray(energy))
    assert energy == Wg + Wphi + Wapar


def test_gx_volume_factors_accept_sampled_geometry_contract():
    _cfg, grid, geom, _params, _cache = _small_setup()
    sampled = sample_flux_tube_geometry(geom, grid.z)

    vol_ref, flux_ref = gx_volume_factors(geom, grid)
    vol_s, flux_s = gx_volume_factors(sampled, grid)

    assert np.allclose(np.asarray(vol_s), np.asarray(vol_ref))
    assert np.allclose(np.asarray(flux_s), np.asarray(flux_ref))


def test_gx_phi_zonal_mode_kxt_recovers_signed_zonal_average() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(replace(cfg.grid, Ny=8, Nx=4))
    geom = SAlphaGeometry.from_config(cfg.geometry)
    vol_fac, _flux_fac = gx_volume_factors(geom, grid)
    phi = jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    zonal_profile = (0.3 - 0.1j) + (0.2 + 0.05j) * jnp.cos(grid.z)
    phi = phi.at[0, 1, :].set(zonal_profile)

    out = gx_phi_zonal_mode_kxt(phi, grid, vol_fac)

    expected = jnp.sum(zonal_profile * vol_fac)
    assert np.allclose(np.asarray(out[1]), np.asarray(expected))
    assert np.allclose(np.asarray(out[0]), 0.0)


def test_gx_volume_factors_use_grho_for_flux_weights():
    _cfg, grid, geom, _params, _cache = _small_setup()
    sampled = sample_flux_tube_geometry(geom, grid.z)
    sampled = replace(
        sampled,
        jacobian_profile=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        grho_profile=jnp.asarray([1.0, 2.0, 1.0, 2.0]),
        theta=jnp.asarray(grid.z[:4]),
    )
    grid_small = replace(grid, z=grid.z[:4])

    vol_fac, flux_fac = gx_volume_factors(sampled, grid_small)

    jac = np.array([1.0, 2.0, 3.0, 4.0])
    grho = np.array([1.0, 2.0, 1.0, 2.0])
    assert np.allclose(np.asarray(vol_fac), jac / np.sum(jac))
    assert np.allclose(np.asarray(flux_fac), jac / np.sum(jac * grho))


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


def test_gx_flux_channel_splits_sum_to_total_multispecies() -> None:
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
    cache = build_linear_cache(grid, geom, params, 3, 4)
    _vol_fac, flux_fac = gx_volume_factors(geom, grid)

    shape = (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)
    field_base = np.arange(grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32).reshape(
        grid.ky.size, grid.kx.size, grid.z.size
    )
    phi = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    apar = 0.3 * phi
    bpar = -0.2 * phi

    heat = np.asarray(gx_heat_flux_species(G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False))
    heat_es, heat_apar, heat_bpar = (
        np.asarray(arr)
        for arr in gx_heat_flux_split_species(
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
    pflux = np.asarray(gx_particle_flux_species(G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False))
    pflux_es, pflux_apar, pflux_bpar = (
        np.asarray(arr)
        for arr in gx_particle_flux_split_species(
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

    np.testing.assert_allclose(heat, heat_es + heat_apar + heat_bpar, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(pflux, pflux_es + pflux_apar + pflux_bpar, rtol=1.0e-6, atol=1.0e-6)

    heat_total = gx_heat_flux_resolved_species(G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False)
    heat_split = gx_heat_flux_split_resolved_species(G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False)
    pflux_total = gx_particle_flux_resolved_species(G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False)
    pflux_split = gx_particle_flux_split_resolved_species(G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False)

    for total_arr, split_arrs in ((heat_total, heat_split), (pflux_total, pflux_split)):
        for idx in range(len(total_arr)):
            combined = np.asarray(split_arrs[0][idx]) + np.asarray(split_arrs[1][idx]) + np.asarray(split_arrs[2][idx])
            np.testing.assert_allclose(np.asarray(total_arr[idx]), combined, rtol=1.0e-6, atol=1.0e-6)


def test_gx_turbulent_heating_zero_for_steady_state() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None))
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0),
            Species(charge=-1.0, mass=0.00027, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, 3, 4)
    vol_fac, _flux_fac = gx_volume_factors(geom, grid)

    shape = (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 0.5), dtype=jnp.complex64)
    field_base = np.arange(grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32).reshape(
        grid.ky.size, grid.kx.size, grid.z.size
    )
    phi = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    apar = 0.2 * phi
    bpar = -0.1 * phi

    heat_species = gx_turbulent_heating_species(
        G,
        G,
        phi,
        apar,
        bpar,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        vol_fac,
        0.05,
        use_dealias=False,
    )
    heat_total = gx_turbulent_heating(
        G,
        G,
        phi,
        apar,
        bpar,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        vol_fac,
        0.05,
        use_dealias=False,
    )

    np.testing.assert_allclose(np.asarray(heat_species), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(heat_total), 0.0, atol=1.0e-7)


def test_gx_turbulent_heating_resolved_sums_to_species_total() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None))
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = build_linear_params(
        [
            Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0),
            Species(charge=-1.0, mass=0.00027, density=1.0, temperature=1.0, tprim=1.0, fprim=1.0),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, 3, 4)
    vol_fac, _flux_fac = gx_volume_factors(geom, grid)

    shape = (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G_old = jnp.asarray(base + 1.0j * (base + 0.5), dtype=jnp.complex64)
    G = 1.03 * G_old + (0.02 - 0.01j)
    field_base = np.arange(grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32).reshape(
        grid.ky.size, grid.kx.size, grid.z.size
    )
    phi_old = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    phi = 1.01 * phi_old + (0.03 + 0.02j)
    apar_old = 0.2 * phi_old
    apar = 0.2 * phi
    bpar_old = -0.1 * phi_old
    bpar = -0.1 * phi + (0.01 - 0.02j)

    heat_species = gx_turbulent_heating_species(
        G,
        G_old,
        phi,
        apar,
        bpar,
        phi_old,
        apar_old,
        bpar_old,
        cache,
        grid,
        params,
        vol_fac,
        0.05,
        use_dealias=False,
    )
    heat_st, heat_kxst, heat_kyst, heat_kxkyst, heat_zst = gx_turbulent_heating_resolved_species(
        G,
        G_old,
        phi,
        apar,
        bpar,
        phi_old,
        apar_old,
        bpar_old,
        cache,
        grid,
        params,
        vol_fac,
        0.05,
        use_dealias=False,
    )

    np.testing.assert_allclose(np.asarray(heat_st), np.asarray(heat_species), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(heat_kxst).sum(axis=1), np.asarray(heat_species), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(heat_kyst).sum(axis=1), np.asarray(heat_species), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(heat_kxkyst).sum(axis=(1, 2)), np.asarray(heat_species), rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(heat_zst).sum(axis=1), np.asarray(heat_species), rtol=1.0e-5, atol=1.0e-6)
    assert np.max(np.abs(np.asarray(heat_species))) > 0.0


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
    time_cfg = ExplicitTimeConfig(dt=0.01, t_max=0.1, sample_stride=1, fixed_dt=True)

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


def test_integrate_linear_gx_diagnostics_honors_rk3_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, grid, geom, params, cache = _small_setup()
    G0 = _build_initial_condition(grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init)
    calls: list[str] = []

    def _fake_step(G, cache, params, term_cfg, dt, *, method):
        calls.append(str(method))
        _dG, fields = assemble_rhs_cached(G, cache, params, terms=term_cfg)
        return G, fields

    monkeypatch.setattr(gx_integrators, "_linear_explicit_step", _fake_step)

    time_cfg = ExplicitTimeConfig(dt=0.01, t_max=0.01, method="rk3", sample_stride=1, fixed_dt=True)
    integrate_linear_gx_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=LinearTerms(),
        jit=False,
    )

    assert calls
    assert set(calls) == {"rk3"}


def test_linear_explicit_step_applies_gx_post_step_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SimpleNamespace(
        dealias_mask=jnp.asarray([[True, True], [True, False]]),
        ky=jnp.asarray([0.0, 0.2]),
        kx=jnp.asarray([0.0, 0.1]),
    )
    G0 = jnp.ones((1, 1, 2, 2, 1), dtype=jnp.complex64)
    seen_states: list[np.ndarray] = []

    def _fake_assemble_rhs(state, _cache, _params, *, terms, dt=None):
        seen_states.append(np.asarray(state))
        return jnp.zeros_like(state), FieldState(phi=state[0, 0])

    monkeypatch.setattr(gx_integrators, "assemble_rhs_cached", _fake_assemble_rhs)

    G_next, fields = gx_integrators._linear_explicit_step(
        G0,
        cache,
        object(),
        linear_terms_to_term_config(LinearTerms()),
        0.1,
        method="euler",
    )

    expected = np.ones((2, 2, 1), dtype=np.complex64)
    expected[0, 0, 0] = 0.0
    expected[1, 1, 0] = 0.0

    assert len(seen_states) == 2
    assert np.allclose(seen_states[0][0, 0], 1.0)
    assert np.allclose(np.asarray(G_next[0, 0]), expected)
    assert np.allclose(np.asarray(seen_states[-1][0, 0]), expected)
    assert np.allclose(np.asarray(fields.phi), expected)


def test_gx_energy_drift_small_no_drive():
    cfg, grid, geom, params, cache = _small_setup()
    params = replace(params, R_over_Ln=0.0, R_over_LTi=0.0, R_over_LTe=0.0, nu=0.0)
    G0 = _build_initial_condition(grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init)
    time_cfg = ExplicitTimeConfig(dt=0.01, t_max=0.2, sample_stride=1, fixed_dt=True)

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


def test_gx_growth_rate_step_max_uses_per_step_peak():
    """GX max-mode diagnostics should follow each step's peak-z sample."""

    phi_prev = jnp.asarray([[[1.0 + 1.0j, 8.0 + 8.0j]]], dtype=jnp.complex64)
    phi_now = jnp.asarray([[[9.0 + 9.0j, 2.0 + 2.0j]]], dtype=jnp.complex64)
    mask = jnp.asarray([[True]])
    gamma, omega = _gx_growth_rate_step(
        phi_now,
        phi_prev,
        0.1,
        z_index=0,
        mask=mask,
        mode_method="max",
    )
    ratio = (9.0 + 9.0j) / (8.0 + 8.0j)
    assert np.allclose(np.asarray(gamma), np.log(np.abs(ratio)) / 0.1)
    assert np.allclose(np.asarray(omega), -np.angle(ratio) / 0.1)


def test_gx_growth_mask_promotes_single_selected_nonzonal_slice() -> None:
    mask = _gx_growth_mask(
        jnp.asarray([-0.01]),
        jnp.asarray([0.0]),
        jnp.asarray([[False]]),
    )
    assert np.asarray(mask).item() is True


def test_gx_growth_mask_keeps_single_selected_zonal_slice_masked() -> None:
    mask = _gx_growth_mask(
        jnp.asarray([0.0]),
        jnp.asarray([0.0]),
        jnp.asarray([[False]]),
    )
    assert np.asarray(mask).item() is False


def test_rk4_step_uses_runtime_scaled_end_damping_once() -> None:
    cfg, grid, geom, params, cache = _small_setup()
    params = replace(params, damp_ends_amp=0.5)
    G0 = _build_initial_condition(grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init)
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=1.0,
        apar=0.0,
        bpar=0.0,
    )
    term_cfg = linear_terms_to_term_config(terms)
    dt = 0.2

    G_step, fields_step = _rk4_step(G0, cache, params, term_cfg, dt)

    def rhs(state: jnp.ndarray) -> jnp.ndarray:
        dG, _fields = assemble_rhs_cached(state, cache, params, terms=term_cfg)
        return dG

    k1 = rhs(G0)
    k2 = rhs(G0 + 0.5 * dt * k1)
    k3 = rhs(G0 + 0.5 * dt * k2)
    k4 = rhs(G0 + dt * k3)
    G_manual = G0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    _dG_manual, fields_manual = assemble_rhs_cached(G_manual, cache, params, terms=term_cfg)

    assert np.allclose(np.asarray(G_step), np.asarray(G_manual), rtol=1.0e-6, atol=1.0e-6)
    assert np.allclose(
        np.asarray(fields_step.phi), np.asarray(fields_manual.phi), rtol=1.0e-6, atol=1.0e-6
    )


def test_linear_gx_adaptive_default_dt_max_matches_gx():
    """When dt_max is unset, adaptive GX path should clamp to dt."""

    cfg, grid, geom, params, cache = _small_setup()
    G0 = _build_initial_condition(grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init)
    time_cfg = ExplicitTimeConfig(
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


def test_gx_linear_omega_max_preserves_selected_ky_mode():
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

    omega_sel = _gx_linear_omega_max(grid, geom, params, 4, 4)

    assert omega_sel[1] > 0.0
