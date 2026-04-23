"""Linear operator tests for the flux-tube electrostatic model."""

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.integration

from spectraxgk.config import CycloneBaseCase, GridConfig, GeometryConfig
from spectraxgk.geometry import SAlphaGeometry, sample_flux_tube_geometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import (
    _build_linked_fft_maps,
    _integrate_linear_cached,
    _x64_enabled,
    LinearCache,
    LinearParams,
    LinearTerms,
    apply_hermite_v,
    apply_laguerre_x,
    build_H,
    _build_implicit_operator,
    build_linear_cache,
    compute_b,
    diamagnetic_drive_coeffs,
    energy_operator,
    grad_z_periodic,
    integrate_linear,
    linear_rhs,
    linear_rhs_cached,
    quasineutrality_phi,
    streaming_term,
)
from spectraxgk.terms.operators import grad_z_linked_fft
from spectraxgk.gyroaverage import J_l_all
from spectraxgk.linear_krylov import dominant_eigenpair
from spectraxgk.terms.linear_terms import collisions_contribution
from spectraxgk.terms.assembly import assemble_rhs_terms_cached
from spectraxgk.terms.config import TermConfig


def test_grad_z_periodic_sine():
    """Centered periodic derivative should differentiate a sine wave."""
    z = jnp.linspace(0.0, 2.0 * jnp.pi, 64, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.sin(z)
    df = grad_z_periodic(f, dz)
    assert jnp.allclose(df, jnp.cos(z), atol=2.0e-2)


def test_build_linked_fft_maps_keeps_real_fft_positive_ky_modes():
    kx = np.array([0.0], dtype=float)
    ky = np.array([0.0, 0.01, 0.02], dtype=float)
    linked_indices, linked_kz = _build_linked_fft_maps(
        kx=kx,
        ky=ky,
        y0=100.0,
        jtwist=2,
        dz=(2.0 * np.pi) / 32.0,
        nz=32,
        real_dtype=jnp.float32,
        ky_mode=np.array([0, 1, 2], dtype=int),
    )

    assert len(linked_indices) == 1
    assert np.array_equal(np.asarray(linked_indices[0]), np.array([[0], [1], [2]], dtype=np.int32))
    assert np.asarray(linked_kz[0]).shape == (32,)


def test_build_linear_cache_zero_shat_periodic_uses_periodic_fft_without_end_damping():
    from spectraxgk.geometry import SlabGeometry, apply_geometry_grid_defaults
    from spectraxgk.config import GeometryConfig
    from spectraxgk.grids import select_real_fft_ky_grid
    from spectraxgk.species import Species, build_linear_params

    geom = SlabGeometry.from_config(GeometryConfig(model="slab", s_hat=1.0e-8, zero_shat=True))
    grid_cfg = apply_geometry_grid_defaults(
        geom,
        GridConfig(Nx=1, Ny=7, Nz=32, Lx=2.0 * np.pi, Ly=200.0 * np.pi, boundary="linked", y0=100.0),
    )
    grid_full = build_spectral_grid(grid_cfg)
    grid = select_real_fft_ky_grid(grid_full, np.array([0.0, 0.01, 0.02], dtype=np.float32))
    params = build_linear_params(
        (Species(charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0),),
        tau_e=0.0,
        kpar_scale=float(geom.gradpar()),
        beta=0.01,
        fapar=1.0,
    )

    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)

    assert cache.use_twist_shift is False
    assert cache.jtwist == 0
    assert len(cache.linked_indices) == 0
    assert np.allclose(np.asarray(cache.damp_profile), 0.0)
    assert cache.linked_damp_profile.size == 0


def test_build_linear_cache_single_selected_ky_ignores_nonlinear_dealias_mask():
    grid_cfg = GridConfig(Nx=1, Ny=16, Nz=32, Lx=2.0 * np.pi, Ly=0.628, boundary="periodic", y0=0.2)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid_full = build_spectral_grid(cfg.grid)
    grid = select_ky_grid(grid_full, 6)  # ky = 30 on this grid; masked by 2/3 in the full nonlinear mesh
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)

    assert np.asarray(grid.dealias_mask).item() is False
    assert float(np.nanmax(np.asarray(cache.kperp2))) > 0.0


def test_linked_fft_derivative_matches_periodic_for_one_link_chains():
    rng = np.random.default_rng(0)
    f = (
        rng.normal(size=(2, 3, 1, 32)) + 1j * rng.normal(size=(2, 3, 1, 32))
    ).astype(np.complex64)
    dz = (2.0 * np.pi) / 32.0
    linked_indices, linked_kz = _build_linked_fft_maps(
        kx=np.array([0.0], dtype=float),
        ky=np.array([0.0, 0.01, 0.02], dtype=float),
        y0=100.0,
        jtwist=2,
        dz=dz,
        nz=32,
        real_dtype=jnp.float32,
        ky_mode=np.array([0, 1, 2], dtype=int),
    )

    df_periodic = grad_z_periodic(jnp.asarray(f), dz=dz)
    df_linked = grad_z_linked_fft(
        jnp.asarray(f),
        dz=dz,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_full_cover=True,
        linked_inverse_permutation=jnp.arange(3, dtype=jnp.int32),
        linked_use_gather=True,
        linked_gather_map=jnp.arange(3, dtype=jnp.int32),
        linked_gather_mask=jnp.ones(3, dtype=bool),
    )

    assert jnp.allclose(df_linked, df_periodic, rtol=1.0e-6, atol=2.0e-6)


def test_compute_b_shape_and_value():
    """b should match k_perp^2 for s-alpha geometry."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    b = compute_b(grid, geom, rho=1.0)
    assert b.shape == (cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)
    kx0 = grid.kx[0]
    ky0 = grid.ky[0]
    theta0 = grid.z[0]
    kx_eff = kx0 + geom.s_hat * ky0 * theta0
    assert jnp.isclose(b[0, 0, 0], kx_eff * kx_eff + ky0 * ky0)


def test_quasineutrality_simple():
    """Quasineutrality should reduce to a simple ratio for a single mode."""
    Nl, Nm, Ny, Nx, Nz = 2, 2, 1, 1, 1
    b = jnp.array([[[0.5]]])
    Jl_single = J_l_all(b, l_max=Nl - 1)
    Jl = Jl_single[None, ...]
    G = jnp.zeros((1, Nl, Nm, Ny, Nx, Nz))
    G = G.at[0, 0, 0, 0, 0, 0].set(2.0)
    phi = quasineutrality_phi(
        G,
        Jl,
        tau_e=1.0,
        charge=jnp.array([1.0]),
        density=jnp.array([1.0]),
        tz=jnp.array([1.0]),
    )
    den = 1.0 + 1.0 - jnp.sum(Jl_single[:, 0, 0, 0] ** 2)
    assert jnp.isclose(phi[0, 0, 0], Jl_single[0, 0, 0, 0] * 2.0 / den)


def test_quasineutrality_charge_sign():
    """Charge sign should flip the quasineutrality solution."""
    Nl, Nm, Ny, Nx, Nz = 2, 1, 1, 1, 4
    Jl = jnp.ones((1, Nl, Ny, Nx, Nz))
    G = jnp.zeros((1, Nl, Nm, Ny, Nx, Nz))
    G = G.at[0, 0, 0, 0, 0, :].set(1.0)
    phi_pos = quasineutrality_phi(
        G,
        Jl,
        tau_e=1.0,
        charge=jnp.array([1.0]),
        density=jnp.array([1.0]),
        tz=jnp.array([1.0]),
    )
    phi_neg = quasineutrality_phi(
        G,
        Jl,
        tau_e=1.0,
        charge=jnp.array([-1.0]),
        density=jnp.array([1.0]),
        tz=jnp.array([-1.0]),
    )
    assert jnp.allclose(phi_pos, -phi_neg)


def test_build_H_adds_phi_to_m0():
    """H should add J_l phi only to the m=0 Hermite index."""
    G = jnp.zeros((1, 2, 2, 1, 1, 1))
    Jl = jnp.ones((1, 2, 1, 1, 1))
    phi = jnp.array([[[3.0]]])
    H = build_H(G, Jl, phi, tz=jnp.array([1.0]))
    assert jnp.allclose(H[0, :, 0, 0, 0, 0], 3.0)
    assert jnp.allclose(H[0, :, 1, 0, 0, 0], 0.0)


def test_build_H_adds_apar_to_m1():
    """Apar enters H at m=1 with GX sign convention."""
    G = jnp.zeros((1, 2, 2, 1, 1, 1))
    Jl = jnp.ones((1, 2, 1, 1, 1))
    phi = jnp.zeros((1, 1, 1))
    apar = jnp.ones((1, 1, 1))
    H = build_H(G, Jl, phi, tz=jnp.array([1.0]), apar=apar, vth=jnp.array([2.0]))
    assert jnp.allclose(H[0, :, 1, 0, 0, 0], -2.0)


def test_collisions_include_gx_conservation_correction():
    G = jnp.zeros((1, 1, 3, 1, 1, 1), dtype=jnp.complex64)
    G = G.at[0, 0, 0, 0, 0, 0].set(2.0 + 0.0j)
    G = G.at[0, 0, 1, 0, 0, 0].set(3.0 + 0.0j)
    G = G.at[0, 0, 2, 0, 0, 0].set(5.0 + 0.0j)
    Jl = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32)
    JlB = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32)
    H = G
    out = collisions_contribution(
        H,
        G=G,
        Jl=Jl,
        JlB=JlB,
        b=jnp.full((1, 1, 1, 1), 4.0, dtype=jnp.float32),
        nu=jnp.array([0.5], dtype=jnp.float32),
        collision_lam=jnp.zeros((1, 1, 3, 1, 1, 1), dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    assert jnp.allclose(out[0, 0, 0, 0, 0, 0], 4.0)
    assert jnp.allclose(out[0, 0, 1, 0, 0, 0], 1.5)
    assert jnp.allclose(out[0, 0, 2, 0, 0, 0], 5.0)


def test_build_H_adds_bpar_to_m0():
    """Bpar term should enter H at m=0 with J_l + J_{l-1}."""
    G = jnp.zeros((1, 3, 2, 1, 1, 1))
    Jl = jnp.ones((1, 3, 1, 1, 1))
    JlB = Jl + jnp.pad(Jl[:, :-1, ...], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
    phi = jnp.zeros((1, 1, 1))
    bpar = jnp.ones((1, 1, 1))
    H = build_H(G, Jl, phi, tz=jnp.array([1.0]), bpar=bpar, JlB=JlB)
    assert jnp.allclose(H[0, :, 0, 0, 0, 0], JlB[0, :, 0, 0, 0])


def test_linear_cache_bessel_bmag_power_scales_b():
    grid_cfg = GridConfig(Nx=1, Ny=1, Nz=8, Lx=6.0, Ly=6.0)
    grid = build_spectral_grid(grid_cfg)
    geom_base = SAlphaGeometry.from_config(GeometryConfig(R0=2.77778))
    geom_bmag = SAlphaGeometry.from_config(
        GeometryConfig(R0=2.77778, kperp2_bmag=False, bessel_bmag_power=1.0)
    )
    params = LinearParams()
    cache_base = build_linear_cache(grid, geom_base, params, Nl=2, Nm=2)
    cache_bmag = build_linear_cache(grid, geom_bmag, params, Nl=2, Nm=2)
    bmag = geom_bmag.bmag(jnp.asarray(grid.z))
    ratio = cache_bmag.b / cache_base.b
    expected = (1.0 / bmag)[None, None, None, :]
    mask = jnp.isfinite(ratio)
    assert jnp.allclose(ratio[mask], expected[mask], rtol=1.0e-6, atol=1.0e-8)


def test_build_linear_cache_accepts_sampled_geometry_contract():
    grid_cfg = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    grid = build_spectral_grid(grid_cfg)
    geom = SAlphaGeometry.from_config(GeometryConfig(R0=2.77778, epsilon=0.18))
    sampled = sample_flux_tube_geometry(geom, jnp.asarray(grid.z))
    params = LinearParams()
    cache_geom = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    cache_sampled = build_linear_cache(grid, sampled, params, Nl=2, Nm=2)
    assert jnp.allclose(cache_sampled.kperp2, cache_geom.kperp2)
    assert jnp.allclose(cache_sampled.omega_d, cache_geom.omega_d)
    assert jnp.allclose(cache_sampled.bmag, cache_geom.bmag)


def test_build_linear_cache_restores_linked_end_damping_on_full_fft_grid():
    grid = build_spectral_grid(
        GridConfig(
            Nx=8,
            Ny=8,
            Nz=8,
            Lx=62.8,
            Ly=2.0 * np.pi,
            boundary="linked",
            y0=1.0,
        )
    )
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    cache = build_linear_cache(grid, geom, LinearParams(), Nl=2, Nm=2)

    profile = np.asarray(cache.linked_damp_profile, dtype=float)
    assert profile.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert np.max(profile) > 0.0

    pos_rows = np.flatnonzero(np.max(profile, axis=(1, 2)) > 0.0)
    pos_rows = pos_rows[pos_rows > 0]
    assert pos_rows.size > 0
    ky_idx = int(pos_rows[0])
    kx_idx = int(np.flatnonzero(np.max(profile[ky_idx], axis=-1) > 0.0)[0])
    mirror_ky = (-ky_idx) % int(grid.ky.size)
    mirror_kx = 0 if kx_idx == 0 else int(grid.kx.size - kx_idx)
    assert np.allclose(profile[mirror_ky, mirror_kx], profile[ky_idx, kx_idx])


def test_build_linear_cache_keeps_linked_end_damping_on_selected_positive_ky_grid():
    grid_full = build_spectral_grid(
        GridConfig(
            Nx=1,
            Ny=16,
            Nz=96,
            Lx=62.8,
            Ly=20.0 * np.pi,
            boundary="linked",
            y0=10.0,
            ntheta=32,
            nperiod=2,
        )
    )
    ky_idx = int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    assert float(grid_full.ky[ky_idx]) > 0.0
    grid = select_ky_grid(grid_full, ky_idx)
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    cache = build_linear_cache(grid, geom, LinearParams(), Nl=16, Nm=48)

    profile = np.asarray(cache.linked_damp_profile, dtype=float)
    assert profile.shape == (1, grid.kx.size, grid.z.size)
    assert np.max(profile) > 0.0
    assert int(np.asarray(grid.ky_mode)[0]) > 0


def test_assemble_rhs_terms_scales_linked_end_damping_by_step_dt():
    grid_full = build_spectral_grid(
        GridConfig(
            Nx=1,
            Ny=16,
            Nz=96,
            Lx=62.8,
            Ly=20.0 * np.pi,
            boundary="linked",
            y0=10.0,
            ntheta=32,
            nperiod=2,
        )
    )
    ky_idx = int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    grid = select_ky_grid(grid_full, ky_idx)
    geom = SAlphaGeometry.from_config(GeometryConfig(s_hat=0.8))
    params = LinearParams(damp_ends_amp=0.1, damp_ends_widthfrac=0.125)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
    G = jnp.ones((2, 4, 1, 1, 96), dtype=jnp.complex64)
    term_cfg = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=1.0,
        apar=0.0,
        bpar=0.0,
    )

    _rhs_raw, _fields_raw, contrib_raw = assemble_rhs_terms_cached(G, cache, params, terms=term_cfg)
    _rhs_dt, _fields_dt, contrib_dt = assemble_rhs_terms_cached(G, cache, params, terms=term_cfg, dt=0.2)

    end_raw = np.asarray(contrib_raw["end_damping"])
    end_dt = np.asarray(contrib_dt["end_damping"])
    mask = np.abs(end_raw) > 1.0e-12
    assert np.any(mask)
    assert np.allclose(end_dt[mask], end_raw[mask] / 0.2, rtol=1.0e-6, atol=1.0e-6)


def test_streaming_zero_for_constant_z():
    """Streaming should vanish for z-constant fields."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.0,
        nu_hyper=0.0,
        nu_hyper_l=0.0,
        nu_hyper_m=0.0,
        nu_hyper_lm=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )

    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    G = G.at[:, 1:, ...].set(1.0)
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    dG, _phi = linear_rhs(G, grid, geom, params, terms=terms)
    assert jnp.allclose(dG, 0.0)


def test_linear_rhs_shapes():
    """RHS and potential should have consistent shapes."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    dG, phi = linear_rhs(G, grid, geom, params)
    assert dG.shape == G.shape
    assert phi.shape == (cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)


def test_linear_param_validation():
    """Invalid parameters should be rejected in checked paths."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    with pytest.raises(ValueError):
        compute_b(grid, geom, rho=0.0)
    with pytest.raises(ValueError):
        quasineutrality_phi(
            G[None, ...],
            jnp.ones((1, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)),
            tau_e=-1.0,
            charge=jnp.array([1.0]),
            density=jnp.array([1.0]),
            tz=jnp.array([1.0]),
        )
    with pytest.raises(ValueError):
        streaming_term(G, dz=1.0, vth=0.0)
    with pytest.raises(ValueError):
        grad_z_periodic(G, dz=0.0)
    with pytest.raises(ValueError):
        linear_rhs(G.reshape(2, 3, -1), grid, geom, LinearParams())


def test_streaming_term_zero():
    """Zero fields should return zero streaming."""
    H = jnp.zeros((2, 3, 1, 1, 8))
    out = streaming_term(H, dz=1.0, vth=1.0)
    assert jnp.allclose(out, 0.0)


def test_integrate_linear_shapes():
    """Integrator should return a time series of phi with expected length."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear(G, grid, geom, params, dt=0.1, steps=3, method="rk4")
    assert phi_t.shape[0] == 3


def test_integrate_linear_progress_with_sample_stride_gt_one():
    """Sampled progress reporting must compute diagnostics before emitting callbacks."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))

    _, phi_t = integrate_linear(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=4,
        method="rk2",
        sample_stride=2,
        show_progress=True,
    )

    assert phi_t.shape[0] == 2


def test_integrate_linear_methods():
    """Explicit and IMEX paths should run without error."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    for method in ("euler", "rk2", "imex", "semi-implicit", "sspx3"):
        _, phi_t = integrate_linear(G, grid, geom, params, dt=0.1, steps=2, method=method)
        assert phi_t.shape[0] == 2


def test_integrate_linear_with_cache():
    """Integrate with a precomputed cache path."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    cache = build_linear_cache(grid, geom, params, G.shape[0], G.shape[1])
    _, phi_t = integrate_linear(G, grid, geom, params, dt=0.1, steps=2, method="rk4", cache=cache)
    assert phi_t.shape[0] == 2


def test_integrate_linear_checkpoint_runs():
    """Checkpointed integration should run on a tiny grid."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear(G, grid, geom, params, dt=0.1, steps=2, method="rk4", checkpoint=True)
    assert phi_t.shape[0] == 2


def test_integrate_linear_invalid_method():
    """Invalid integrator names should raise a ValueError."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    with pytest.raises(ValueError):
        integrate_linear(G, grid, geom, params, dt=0.1, steps=2, method="rk5")


def test_linear_cache_matches_rhs():
    """Cached RHS should match the direct RHS for the same inputs."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    cache = build_linear_cache(grid, geom, params, G.shape[0], G.shape[1])
    dG0, phi0 = linear_rhs(G, grid, geom, params)
    dG1, phi1 = linear_rhs_cached(G, cache, params)
    assert jnp.allclose(dG0, dG1)
    assert jnp.allclose(phi0, phi1)


def test_linear_cache_tree_roundtrip():
    """LinearCache pytree should round-trip through flatten/unflatten."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    children, aux = cache.tree_flatten()
    cache2 = LinearCache.tree_unflatten(aux, children)
    assert jnp.allclose(cache2.Jl, cache.Jl)
    assert jnp.allclose(cache2.omega_d, cache.omega_d)
    assert jnp.allclose(cache2.lb_lam, cache.lb_lam)
    assert jnp.allclose(cache2.hyper_ratio, cache.hyper_ratio)


def test_build_linear_cache_multispecies():
    """Cache should support multiple species arrays."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(rho=jnp.array([1.0, 0.5]))
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    assert cache.Jl.shape[0] == 2


def test_linear_rhs_multispecies_shapes():
    """Multispecies RHS should return a matching shape."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        charge_sign=jnp.array([1.0, -1.0]),
        density=jnp.array([1.0, 1.0]),
        mass=jnp.array([1.0, 0.001]),
        temp=jnp.array([1.0, 1.0]),
        vth=jnp.array([1.0, 1.0]),
        rho=jnp.array([1.0, 0.5]),
        tz=jnp.array([1.0, -1.0]),
        R_over_Ln=jnp.array([0.0, 0.0]),
        R_over_LTi=jnp.array([0.0, 0.0]),
        )


def test_implicit_preconditioner_hermite_line_shape_and_finite():
    """Hermite-line preconditioner should run and preserve shape/dtype."""

    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=16, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        omega_d_scale=0.2,
        omega_star_scale=0.55,
        rho_star=0.9,
        kpar_scale=float(geom.gradpar()),
    )
    Nl, Nm = 4, 6
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    G0 = jnp.zeros((1, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=base_dtype)
    _G, _shape, size, _dt_val, precond_op, _matvec, _squeeze = _build_implicit_operator(
        G0,
        cache,
        params,
        dt=0.1,
        terms=LinearTerms(),
        implicit_preconditioner="hermite-line",
    )
    x = jnp.ones((size,), dtype=base_dtype)
    y = precond_op(x)
    assert y.shape == (size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))
    assert jnp.all(jnp.isfinite(jnp.imag(y)))


def test_shift_invert_preconditioner_hermite_line_runs():
    """Shift-invert Krylov path should run with the Hermite-line preconditioner."""

    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=16, Lx=6.28, Ly=6.28)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        kpar_scale=float(geom.gradpar()),
    )
    Nl, Nm = 4, 8
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    v0 = jnp.ones((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=base_dtype)
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    eig, _vec = dominant_eigenpair(
        v0,
        cache,
        params,
        terms,
        method="shift_invert",
        krylov_dim=4,
        restarts=1,
        shift=0.5j,
        shift_tol=1.0e-2,
        shift_maxiter=20,
        shift_restart=10,
        shift_solve_method="batched",
        shift_preconditioner="hermite-line",
    )
    assert jnp.isfinite(jnp.real(eig))
    assert jnp.isfinite(jnp.imag(eig))


def test_linear_cache_rho_star_scales_ky():
    """rho_star should scale cached ky for normalization control."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(rho_star=2.0)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    assert jnp.allclose(cache.ky, grid.ky * 2.0)


def test_linear_rhs_cached_invalid_shape():
    """Cached RHS should reject invalid shapes."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    with pytest.raises(ValueError):
        linear_rhs_cached(jnp.zeros((2, 3, 4)), cache, params)


def test_jit_path_handles_tracers():
    """JIT tracing should exercise the tracer-safe validation path."""
    import jax
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))

    @jax.jit
    def _run(G_in):
        return linear_rhs(G_in, grid, geom, params)[0]

    out = _run(G)
    assert out.shape == G.shape


def test_integrate_linear_implicit_runs():
    """Implicit path should run on a tiny grid."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(nu=0.1)
    G = jnp.zeros((1, 1, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    _, phi_t = integrate_linear(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=1,
        method="implicit",
        implicit_iters=2,
        implicit_relax=0.5,
    )
    assert phi_t.shape[0] == 1


def test_apply_hermite_v_simple():
    """Hermite v operator should map a single mode to neighbors."""
    G = jnp.zeros((1, 3, 1, 1, 1))
    G = G.at[0, 1, 0, 0, 0].set(1.0)
    out = apply_hermite_v(G)
    assert jnp.isclose(out[0, 0, 0, 0, 0], 1.0)
    assert jnp.isclose(out[0, 2, 0, 0, 0], jnp.sqrt(2.0))


def test_apply_laguerre_x_simple():
    """Laguerre x operator should reproduce the three-term recurrence."""
    G = jnp.zeros((3, 1, 1, 1, 1))
    G = G.at[1, 0, 0, 0, 0].set(1.0)
    out = apply_laguerre_x(G)
    assert jnp.isclose(out[0, 0, 0, 0, 0], -1.0)
    assert jnp.isclose(out[1, 0, 0, 0, 0], 3.0)
    assert jnp.isclose(out[2, 0, 0, 0, 0], -2.0)


def test_energy_operator_and_drive_coeffs():
    """Energy and drive coefficient helpers should return consistent shapes."""
    G = jnp.zeros((2, 3, 1, 1, 1))
    energy = energy_operator(G, coeff_const=1.0, coeff_par=0.5, coeff_perp=1.0)
    assert energy.shape == G.shape
    coeffs = diamagnetic_drive_coeffs(
        2, 3, eta_i=jnp.array(0.0), coeff_const=1.0, coeff_par=0.5, coeff_perp=1.0
    )
    assert coeffs.shape == (2, 3)
    assert jnp.isclose(coeffs[0, 0], 1.0)
    assert jnp.allclose(coeffs[1:, :], 0.0)


def test_gx_mirror_curvature_activation():
    """Drift/mirror terms should activate when omega_d_scale is nonzero."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    G = jnp.zeros((1, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    G = G.at[0, 1, 1, 0, :].set(1.0 + 0.0j)

    params_off = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache_off = build_linear_cache(grid, geom, params_off, G.shape[0], G.shape[1])
    terms_off = LinearTerms(streaming=0.0, mirror=0.0, curvature=0.0, gradb=0.0, diamagnetic=0.0,
                            collisions=0.0, hypercollisions=0.0, end_damping=0.0, apar=0.0)
    dG_off, _phi_off = linear_rhs_cached(G, cache_off, params_off, terms=terms_off)
    assert jnp.allclose(dG_off, 0.0)

    params_on = LinearParams(
        omega_d_scale=1.0,
        omega_star_scale=0.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache_on = build_linear_cache(grid, geom, params_on, G.shape[0], G.shape[1])
    terms_on = LinearTerms(streaming=0.0, mirror=1.0, curvature=1.0, gradb=1.0, diamagnetic=0.0,
                           collisions=0.0, hypercollisions=0.0, end_damping=0.0, apar=0.0)
    dG_on, _phi_on = linear_rhs_cached(G, cache_on, params_on, terms=terms_on)
    assert jnp.max(jnp.abs(dG_on)) > 0.0


def test_gx_diamagnetic_drive_populates_m2():
    """Diamagnetic drive should populate the m=2 component when enabled."""
    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    ky_index = 1
    G = G.at[0, 0, ky_index, 0, :].set(1.0 + 0.0j)

    params_off = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache_off = build_linear_cache(grid, geom, params_off, G.shape[0], G.shape[1])
    terms_off = LinearTerms(streaming=0.0, mirror=0.0, curvature=0.0, gradb=0.0, diamagnetic=0.0,
                            collisions=0.0, hypercollisions=0.0, end_damping=0.0, apar=0.0)
    dG_off, _phi_off = linear_rhs_cached(G, cache_off, params_off, terms=terms_off)
    assert jnp.allclose(dG_off[:, 2, ...], 0.0)

    params_on = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=1.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache_on = build_linear_cache(grid, geom, params_on, G.shape[0], G.shape[1])
    terms_on = LinearTerms(streaming=0.0, mirror=0.0, curvature=0.0, gradb=0.0, diamagnetic=1.0,
                           collisions=0.0, hypercollisions=0.0, end_damping=0.0, apar=0.0)
    dG_on, _phi_on = linear_rhs_cached(G, cache_on, params_on, terms=terms_on)
    assert jnp.max(jnp.abs(dG_on[:, 2, ...])) > 0.0
    assert jnp.allclose(dG_on[:, 1, ...], 0.0)


def test_gx_drive_vanishes_for_ky_zero():
    """Diamagnetic drive should vanish for the ky=0 mode."""
    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    G = G.at[0, 0, 0, 0, :].set(1.0 + 0.0j)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=1.0,
        kpar_scale=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    cache = build_linear_cache(grid, geom, params, G.shape[0], G.shape[1])
    terms = LinearTerms(streaming=0.0, mirror=0.0, curvature=0.0, gradb=0.0, diamagnetic=1.0,
                        collisions=0.0, hypercollisions=0.0, end_damping=0.0, apar=0.0)
    dG, _phi = linear_rhs_cached(G, cache, params, terms=terms)
    assert jnp.allclose(dG, 0.0)


def test_rho_star_scales_cache_ky():
    """rho_star should scale the cached ky values."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(rho_star=2.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
    assert jnp.allclose(cache.ky, 2.0 * grid.ky)


def test_shift_axis_noop():
    """shift_axis should return the input when offset is zero."""
    from spectraxgk.linear import shift_axis

    arr = jnp.arange(6.0).reshape(2, 3)
    out = shift_axis(arr, 0, axis=0)
    assert jnp.allclose(out, arr)


def test_apar_streaming_coupling_changes_rhs():
    """Finite beta should modify streaming via Apar coupling."""
    grid_cfg = GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    z = grid.z
    G = G.at[0, 1, 1, 0, :].set(jnp.sin(z) + 0.0j)

    params_base = LinearParams(
        kpar_scale=1.0,
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        beta=0.0,
        fapar=0.0,
    )
    params_beta = LinearParams(
        kpar_scale=1.0,
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        beta=1.0,
        fapar=1.0,
    )
    cache = build_linear_cache(grid, geom, params_beta, G.shape[0], G.shape[1])
    dG0, _phi0 = linear_rhs_cached(G, cache, params_base, terms=LinearTerms())
    dG1, _phi1 = linear_rhs_cached(G, cache, params_beta, terms=LinearTerms())
    assert jnp.max(jnp.abs(dG1 - dG0)) > 0.0
