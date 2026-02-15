
import jax.numpy as jnp
import pytest

from spectraxgk.operators import hermite_streaming


def test_streaming_zero_kpar():
    G = jnp.ones((2, 4))
    dG = hermite_streaming(G, kpar=0.0, vth=1.0)
    assert jnp.allclose(dG, 0.0)


def test_streaming_energy_conservation():
    G = jnp.sin(jnp.arange(12, dtype=jnp.float32)).reshape(3, 4)
    dG = hermite_streaming(G, kpar=0.7, vth=1.2)
    energy_rate = jnp.real(jnp.sum(jnp.conj(G) * dG))
    assert jnp.isclose(energy_rate, 0.0, atol=1e-6)


def test_streaming_known_case():
    G = jnp.array([[1.0, 2.0, 0.0]])
    dG = hermite_streaming(G, kpar=1.0, vth=1.0)
    assert jnp.isclose(dG[0, 0], -1j * 2.0)
    assert jnp.isclose(dG[0, 1], -1j * 1.0)
    assert jnp.isclose(dG[0, 2], -1j * (jnp.sqrt(2.0) * 2.0))


def test_streaming_invalid_shape():
    with pytest.raises(ValueError):
        hermite_streaming(jnp.zeros((2, 0)), kpar=1.0, vth=1.0)
