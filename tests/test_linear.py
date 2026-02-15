"""Linear operator tests for the flux-tube electrostatic model."""

import jax.numpy as jnp
import pytest

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import (
    LinearCache,
    LinearParams,
    apply_hermite_v,
    apply_laguerre_x,
    build_H,
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
from spectraxgk.gyroaverage import J_l_all


def test_grad_z_periodic_sine():
    """Centered periodic derivative should differentiate a sine wave."""
    z = jnp.linspace(0.0, 2.0 * jnp.pi, 64, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.sin(z)
    df = grad_z_periodic(f, dz)
    assert jnp.allclose(df, jnp.cos(z), atol=2.0e-2)


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
    Jl = J_l_all(b, l_max=Nl - 1)
    G = jnp.zeros((Nl, Nm, Ny, Nx, Nz))
    G = G.at[0, 0, 0, 0, 0].set(2.0)
    phi = quasineutrality_phi(G, Jl, tau_e=1.0)
    den = 1.0 + 1.0 - jnp.sum(Jl[:, 0, 0, 0] ** 2)
    assert jnp.isclose(phi[0, 0, 0], Jl[0, 0, 0, 0] * 2.0 / den)


def test_build_H_adds_phi_to_m0():
    """H should add J_l phi only to the m=0 Hermite index."""
    G = jnp.zeros((2, 2, 1, 1, 1))
    Jl = jnp.ones((2, 1, 1, 1))
    phi = jnp.array([[[3.0]]])
    H = build_H(G, Jl, phi)
    assert jnp.allclose(H[:, 0, 0, 0, 0], 3.0)
    assert jnp.allclose(H[:, 1, 0, 0, 0], 0.0)


def test_streaming_zero_for_constant_z():
    """Streaming should vanish for z-constant fields."""
    grid_cfg = GridConfig(Nx=8, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(omega_d_scale=0.0, omega_star_scale=0.0)

    G = jnp.zeros((2, 3, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    G = G.at[:, 1:, ...].set(1.0)
    dG, _phi = linear_rhs(G, grid, geom, params)
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
        quasineutrality_phi(G, jnp.ones((2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)), tau_e=0.0)
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


def test_integrate_linear_methods():
    """Euler and RK2 paths should run without error."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    for method in ("euler", "rk2"):
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
    cache = build_linear_cache(grid, geom, params, G.shape[0])
    _, phi_t = integrate_linear(G, grid, geom, params, dt=0.1, steps=2, method="rk4", cache=cache)
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
    cache = build_linear_cache(grid, geom, params, G.shape[0])
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
    cache = build_linear_cache(grid, geom, params, Nl=2)
    children, aux = cache.tree_flatten()
    cache2 = LinearCache.tree_unflatten(aux, children)
    assert jnp.allclose(cache2.Jl, cache.Jl)
    assert jnp.allclose(cache2.omega_d, cache.omega_d)


def test_linear_rhs_cached_invalid_shape():
    """Cached RHS should reject invalid shapes."""
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2)
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
