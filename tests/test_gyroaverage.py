"""Gyroaveraging coefficient tests."""

import jax.numpy as jnp
import pytest

from spectraxgk.gyroaverage import J_l_all, gamma0, sum_Jl2


def test_gamma0_basic_properties():
    """Gamma0 should be 1 at b=0 and decrease with b."""
    b = jnp.array([0.0, 1.0, 4.0])
    g = gamma0(b)
    assert jnp.isclose(g[0], 1.0)
    assert g[1] < g[0]
    assert g[2] < g[1]


def test_gamma0_small_b_series():
    """Small-b series should be accurate to O(b^2)."""
    b = jnp.array(1.0e-3)
    g = gamma0(b)
    approx = 1.0 - b + 0.75 * b * b
    assert jnp.isclose(g, approx, rtol=1e-6, atol=1e-12)


def test_Jl_shape_and_J0():
    """J_l should return the correct array shape and J0 factor."""
    b = jnp.array([0.0, 0.5])
    Jl = J_l_all(b, l_max=3)
    assert Jl.shape == (4, 2)
    assert jnp.allclose(Jl[0], jnp.exp(-0.5 * b))


def test_Jl_zero_b_is_one():
    """At b=0 only the l=0 coefficient is nonzero."""
    b = jnp.array(0.0)
    Jl = J_l_all(b, l_max=4)
    assert jnp.isclose(Jl[0], 1.0)
    assert jnp.allclose(Jl[1:], 0.0)


def test_Jl_large_b_decay():
    """Large b should suppress J0 via the exponential factor."""
    b = jnp.array(10.0)
    J0 = J_l_all(b, l_max=0)[0]
    assert J0 < 1.0e-2


def test_Jl_first_order_coeff():
    """J1 should match the analytic (-b/2) * exp(-b/2) coefficient."""
    b = jnp.array(0.6)
    Jl = J_l_all(b, l_max=1)
    expected = -0.5 * b * jnp.exp(-0.5 * b)
    assert jnp.isclose(Jl[1], expected)


def test_sumJl2_monotone_in_lmax():
    """Truncated sum of J_l^2 should increase with l_max."""
    b = jnp.array([0.2, 1.5])
    s2 = sum_Jl2(b, l_max=2)
    s5 = sum_Jl2(b, l_max=5)
    assert jnp.all(s5 >= s2)


def test_Jl_invalid_lmax():
    """Negative l_max should raise a ValueError."""
    with pytest.raises(ValueError):
        J_l_all(jnp.array(0.0), l_max=-1)
