import jax.numpy as jnp

from spectraxgk.basis import hermite_normed, laguerre


def test_hermite_orthonormality():
    n_max = 4
    x = jnp.linspace(-6.0, 6.0, 4001)
    dx = x[1] - x[0]
    h = hermite_normed(x, n_max)
    w = jnp.exp(-x * x)
    gram = jnp.einsum("ix,jx,x->ij", h, h, w) * dx
    assert jnp.allclose(gram, jnp.eye(n_max + 1), atol=2e-2)


def test_laguerre_orthonormality():
    l_max = 4
    x = jnp.linspace(0.0, 40.0, 8001)
    dx = x[1] - x[0]
    l = laguerre(x, l_max)
    w = jnp.exp(-x)
    gram = jnp.einsum("ix,jx,x->ij", l, l, w) * dx
    assert jnp.allclose(gram, jnp.eye(l_max + 1), atol=2e-2)


def test_basis_invalid_inputs():
    import pytest
    from spectraxgk.basis import hermite_ladder_coeffs

    with pytest.raises(ValueError):
        hermite_normed(jnp.array([0.0]), -1)
    with pytest.raises(ValueError):
        laguerre(jnp.array([0.0]), -1)
    with pytest.raises(ValueError):
        hermite_ladder_coeffs(-1)


def test_basis_zero_order():
    x = jnp.array([0.0, 1.0])
    h0 = hermite_normed(x, 0)
    l0 = laguerre(x, 0)
    assert h0.shape == (1, 2)
    assert l0.shape == (1, 2)
