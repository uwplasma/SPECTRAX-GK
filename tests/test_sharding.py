import jax
import jax.numpy as jnp
import pytest

from spectraxgk.sharding import resolve_state_sharding


def _state_5d():
    return jnp.zeros((2, 2, 4, 1, 8), dtype=jnp.complex64)


def _state_6d():
    return jnp.zeros((1, 2, 2, 4, 1, 8), dtype=jnp.complex64)


def test_state_sharding_disabled():
    G0 = _state_5d()
    assert resolve_state_sharding(G0, None) is None
    assert resolve_state_sharding(G0, "none") is None
    assert resolve_state_sharding(G0, "off") is None


def test_state_sharding_invalid():
    G0 = _state_5d()
    with pytest.raises(ValueError):
        resolve_state_sharding(G0, "banana")


def test_state_sharding_single_device_noop():
    G0 = _state_6d()
    sharding = resolve_state_sharding(G0, "ky", devices=[jax.devices()[0]])
    assert sharding is None
