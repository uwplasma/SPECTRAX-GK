"""Linear operator tests for the flux-tube electrostatic model."""

import jax.numpy as jnp
import pytest

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import (
    LinearParams,
    build_H,
    compute_b,
    grad_z_periodic,
    integrate_linear,
    linear_rhs,
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
    params = LinearParams()

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
    _, phi_t = integrate_linear(G, grid, geom, params, dt=0.1, steps=3)
    assert phi_t.shape[0] == 3


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
