
import jax.numpy as jnp
import pytest

from spectraxgk.gyroaverage import J_l_all, gamma0, sum_Jl2


def test_gamma0_basic_properties():
    b = jnp.array([0.0, 1.0, 4.0])
    g = gamma0(b)
    assert jnp.isclose(g[0], 1.0)
    assert g[1] < g[0]
    assert g[2] < g[1]


def test_gamma0_small_b_series():
    b = jnp.array(1.0e-3)
    g = gamma0(b)
    approx = 1.0 - b + 0.75 * b * b
    assert jnp.isclose(g, approx, rtol=1e-6, atol=1e-12)


def test_Jl_shape_and_J0():
    b = jnp.array([0.0, 0.5])
    Jl = J_l_all(b, l_max=3)
    assert Jl.shape == (4, 2)
    assert jnp.allclose(Jl[0], jnp.exp(-0.5 * b))


def test_Jl_zero_b_is_one():
    b = jnp.array(0.0)
    Jl = J_l_all(b, l_max=4)
    assert jnp.allclose(Jl, jnp.ones((5,)))


def test_Jl_large_b_decay():
    b = jnp.array(10.0)
    J0 = J_l_all(b, l_max=0)[0]
    assert J0 < 1.0e-2


def test_sumJl2_monotone_in_lmax():
    b = jnp.array([0.2, 1.5])
    s2 = sum_Jl2(b, l_max=2)
    s5 = sum_Jl2(b, l_max=5)
    assert jnp.all(s5 >= s2)


def test_Jl_invalid_lmax():
    with pytest.raises(ValueError):
        J_l_all(jnp.array(0.0), l_max=-1)
