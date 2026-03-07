"""Geometry helper tests."""

import jax.numpy as jnp

from spectraxgk.config import GeometryConfig, GridConfig
from spectraxgk.geometry import SAlphaGeometry, sample_flux_tube_geometry
from spectraxgk.grids import build_spectral_grid


def test_kperp2_matches_s_alpha():
    """k_perp^2 should match the s-alpha formula for kx(theta)."""
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.0)
    kx0 = jnp.array(0.0)
    ky = jnp.array(1.0)
    theta = jnp.array([0.0, 2.0])
    kperp2 = geom.k_perp2(kx0, ky, theta)
    assert jnp.allclose(kperp2[0], 1.0)
    assert jnp.allclose(kperp2[1], 5.0)


def test_geometry_from_config():
    """Geometry config should map cleanly into the geometry class."""
    cfg = GeometryConfig(q=1.7, s_hat=0.9, epsilon=0.2, R0=3.0, B0=2.0, alpha=0.1)
    geom = SAlphaGeometry.from_config(cfg)
    assert geom.q == 1.7
    assert geom.R0 == 3.0
    assert geom.alpha == 0.1


def test_bmag_and_omega_d_shapes():
    """Magnetic field and drift frequency should have consistent shapes."""
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    theta = jnp.array([0.0])
    bmag = geom.bmag(theta)
    assert jnp.isclose(bmag[0], 1.0 / (1.0 + geom.epsilon))
    assert jnp.isclose(geom.gradpar(), jnp.abs(1.0 / (geom.q * geom.R0)))

    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0))
    omega_d = geom.omega_d(grid.kx, grid.ky, grid.z)
    assert omega_d.shape == (grid.ky.size, grid.kx.size, grid.z.size)


def test_metric_and_drift_coeffs_at_midplane():
    """Metric and drift coefficients should reduce cleanly at theta=0."""
    geom = SAlphaGeometry(q=1.4, s_hat=0.7, epsilon=0.0, R0=2.0, alpha=0.2)
    theta = jnp.array([0.0])
    gds2, gds21, gds22 = geom.metric_coeffs(theta)
    assert jnp.isclose(gds2[0], 1.0)
    assert jnp.isclose(gds21[0], 0.0)
    assert jnp.isclose(gds22, geom.s_hat * geom.s_hat)

    cv, gb, cv0, gb0 = geom.drift_coeffs(theta)
    expected = geom.drift_scale * (1.0 / geom.R0)
    assert jnp.isclose(cv[0], expected)
    assert jnp.isclose(gb[0], cv[0])
    assert jnp.isclose(cv0[0], 0.0)
    assert jnp.isclose(gb0[0], 0.0)

    cv_d, gb_d = geom.drift_components(jnp.array([0.0]), jnp.array([1.0]), theta)
    assert cv_d.shape == (1, 1, 1)
    assert gb_d.shape == (1, 1, 1)
    bgrad = geom.bgrad(theta)
    assert jnp.isfinite(bgrad[0])


def test_kx_effective_shear_shift():
    """kx_effective should include the s-alpha shear shift."""
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.0, alpha=0.5)
    kx0 = jnp.array([0.2])
    ky = jnp.array([0.3])
    theta = jnp.array([1.0])
    kx_eff = geom.kx_effective(kx0, ky, theta)
    shear = geom.s_hat * theta - geom.alpha * jnp.sin(theta)
    assert jnp.isclose(kx_eff[0], kx0[0] - shear[0] * ky[0])


def test_geometry_tree_roundtrip():
    """Geometry pytree should round-trip through flatten/unflatten."""
    geom = SAlphaGeometry(q=1.5, s_hat=0.8, epsilon=0.2, R0=3.0, B0=1.8, alpha=0.1)
    children, aux = geom.tree_flatten()
    geom2 = SAlphaGeometry.tree_unflatten(aux, children)
    assert geom2.q == geom.q
    assert geom2.s_hat == geom.s_hat
    assert geom2.epsilon == geom.epsilon
    assert geom2.R0 == geom.R0
    assert geom2.B0 == geom.B0
    assert geom2.alpha == geom.alpha


def test_sampled_flux_tube_geometry_matches_salpha_profiles():
    """Sampled geometry data should preserve the analytic s-alpha profiles."""
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, alpha=0.1)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 17)
    sampled = sample_flux_tube_geometry(geom, theta)

    assert jnp.allclose(sampled.bmag(theta), geom.bmag(theta))
    assert jnp.allclose(sampled.bgrad(theta), geom.bgrad(theta))
    gds2_s, gds21_s, gds22_s = sampled.metric_coeffs(theta)
    gds2_g, gds21_g, gds22_g = geom.metric_coeffs(theta)
    assert jnp.allclose(gds2_s, gds2_g)
    assert jnp.allclose(gds21_s, gds21_g)
    assert jnp.allclose(gds22_s, jnp.full_like(theta, gds22_g))

    kx = jnp.array([0.0, 0.2])
    ky = jnp.array([0.1, 0.3])
    theta_b = theta[None, None, :]
    assert jnp.allclose(
        sampled.k_perp2(kx[None, :, None], ky[:, None, None], theta_b),
        geom.k_perp2(kx[None, :, None], ky[:, None, None], theta_b),
    )
