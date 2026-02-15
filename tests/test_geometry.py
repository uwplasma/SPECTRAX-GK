"""Geometry helper tests."""

import jax.numpy as jnp

from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.config import GeometryConfig


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
