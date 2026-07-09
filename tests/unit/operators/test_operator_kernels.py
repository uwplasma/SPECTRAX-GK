"""Focused operator-kernel and reduced full-operator regression checks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.benchmarks import run_cyclone_scan
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.core.grid import build_spectral_grid
from spectraxgk.core.velocity import J_l_all, gamma0, sum_Jl2
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache
from spectraxgk.operators import hermite_streaming
from spectraxgk.terms.assembly import assemble_rhs_cached
from spectraxgk.terms.config import TermConfig


def test_streaming_zero_kpar() -> None:
    """Hermite streaming should vanish at k_parallel = 0."""
    state = jnp.ones((2, 4))

    derivative = hermite_streaming(state, kpar=0.0, vth=1.0)

    assert jnp.allclose(derivative, 0.0)


def test_streaming_energy_conservation() -> None:
    """Hermite streaming is skew-adjoint for constant k_parallel."""
    state = jnp.sin(jnp.arange(12, dtype=jnp.float32)).reshape(3, 4)

    derivative = hermite_streaming(state, kpar=0.7, vth=1.2)
    energy_rate = jnp.real(jnp.sum(jnp.conj(state) * derivative))

    assert jnp.isclose(energy_rate, 0.0, atol=1e-6)


def test_streaming_known_case() -> None:
    """Simple Hermite ladder behavior should match a hand calculation."""
    state = jnp.array([[1.0, 2.0, 0.0]])

    derivative = hermite_streaming(state, kpar=1.0, vth=1.0)

    assert jnp.isclose(derivative[0, 0], -1j * 2.0)
    assert jnp.isclose(derivative[0, 1], -1j * 1.0)
    assert jnp.isclose(derivative[0, 2], -1j * (jnp.sqrt(2.0) * 2.0))


def test_streaming_invalid_shape() -> None:
    """Hermite axis length must be positive."""
    with pytest.raises(ValueError):
        hermite_streaming(jnp.zeros((2, 0)), kpar=1.0, vth=1.0)


def test_linear_operator_package_reexports_streaming_kernel() -> None:
    from spectraxgk.operators.linear import hermite_streaming as package_streaming
    from spectraxgk.operators.linear.moments import hermite_streaming as streaming_impl

    assert hermite_streaming is package_streaming
    assert package_streaming is streaming_impl


def test_gamma0_basic_properties() -> None:
    """Gamma0 should be 1 at b=0 and decrease with b."""
    b = jnp.array([0.0, 1.0, 4.0])

    values = gamma0(b)

    assert jnp.isclose(values[0], 1.0)
    assert values[1] < values[0]
    assert values[2] < values[1]


def test_gamma0_small_b_series() -> None:
    """Small-b series should be accurate to O(b^2)."""
    b = jnp.array(1.0e-3)

    value = gamma0(b)
    approx = 1.0 - b + 0.75 * b * b

    assert jnp.isclose(value, approx, rtol=1e-6, atol=1e-12)


def test_jl_shape_and_j0() -> None:
    """J_l should return the correct array shape and J0 factor."""
    b = jnp.array([0.0, 0.5])

    values = J_l_all(b, l_max=3)

    assert values.shape == (4, 2)
    assert jnp.allclose(values[0], jnp.exp(-0.5 * b))


def test_jl_zero_b_is_one() -> None:
    """At b=0 only the l=0 coefficient is nonzero."""
    b = jnp.array(0.0)

    values = J_l_all(b, l_max=4)

    assert jnp.isclose(values[0], 1.0)
    assert jnp.allclose(values[1:], 0.0)


def test_jl_large_b_decay() -> None:
    """Large b should suppress J0 through the exponential factor."""
    j0 = J_l_all(jnp.array(10.0), l_max=0)[0]

    assert j0 < 1.0e-2


def test_jl_large_b_stays_finite() -> None:
    """Large-b ETG coefficients should underflow cleanly."""
    b = jnp.array([100.0, 500.0, 2000.0], dtype=jnp.float32)

    values = J_l_all(b, l_max=23)

    assert jnp.all(jnp.isfinite(values))


def test_jl_first_order_coeff() -> None:
    """J1 should match the analytic (-b/2) * exp(-b/2) coefficient."""
    b = jnp.array(0.6)

    values = J_l_all(b, l_max=1)
    expected = -0.5 * b * jnp.exp(-0.5 * b)

    assert jnp.isclose(values[1], expected)


def test_sum_jl2_monotone_in_lmax() -> None:
    """Truncated sum of J_l^2 should increase with l_max."""
    b = jnp.array([0.2, 1.5])

    s2 = sum_Jl2(b, l_max=2)
    s5 = sum_Jl2(b, l_max=5)

    assert jnp.all(s5 >= s2)


def test_jl_invalid_lmax() -> None:
    """Negative l_max should raise a ValueError."""
    with pytest.raises(ValueError):
        J_l_all(jnp.array(0.0), l_max=-1)


def test_hyperdiffusion_damps_high_k() -> None:
    """k_perp hyperdiffusion should damp high-k modes more strongly."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=4, Lx=6.0, Ly=6.0)
    grid = build_spectral_grid(grid_cfg)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, B0=1.0)
    params = LinearParams(D_hyper=0.5, p_hyper_kperp=2.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
    state = jnp.ones(
        (1, 1, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=jnp.complex64,
    )
    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=1.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=0.0,
    )

    derivative, _ = assemble_rhs_cached(state, cache, params, terms=terms)
    derivative = derivative[0, 0]
    kperp2 = (
        jnp.asarray(grid.ky)[:, None] ** 2 + jnp.asarray(grid.kx)[None, :] ** 2
    ).astype(float)
    mask = np.asarray(grid.dealias_mask, dtype=bool)
    kperp2_np = np.asarray(kperp2).copy()
    kperp2_np[~mask] = np.nan
    low_idx = np.unravel_index(np.nanargmin(kperp2_np), kperp2_np.shape)
    high_idx = np.unravel_index(np.nanargmax(kperp2_np), kperp2_np.shape)

    low = jnp.abs(derivative[low_idx[0], low_idx[1], 0])
    high = jnp.abs(derivative[high_idx[0], high_idx[1], 0])

    assert high > low


def test_full_operator_scan_relaxed() -> None:
    """Full operator should produce finite scans on a field-aligned grid."""
    grid = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    cfg = CycloneBaseCase(grid=grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    ky_values = np.array([0.2, 0.3, 0.4])
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
    )

    scan = run_cyclone_scan(
        ky_values,
        cfg=cfg,
        Nl=4,
        Nm=8,
        steps=200,
        dt=0.02,
        method="rk4",
        terms=LinearTerms(),
        params=params,
    )

    for gamma, omega in zip(scan.gamma, scan.omega, strict=True):
        assert np.isfinite(gamma)
        assert np.isfinite(omega)
        assert abs(gamma) < 50.0
        assert abs(omega) < 100.0
