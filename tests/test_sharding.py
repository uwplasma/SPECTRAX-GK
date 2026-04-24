import jax
import jax.numpy as jnp
import pytest

import spectraxgk.sharding as sharding_mod
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


def test_state_sharding_builds_partition_specs_with_fake_mesh(monkeypatch):
    class FakeNamedSharding:
        def __init__(self, mesh, spec):
            self.mesh = mesh
            self.spec = spec

    monkeypatch.setattr(sharding_mod, "_mesh_from_devices", lambda devices, axis_name: f"mesh:{axis_name}")
    monkeypatch.setattr(sharding_mod, "NamedSharding", FakeNamedSharding)

    ky_sharding = resolve_state_sharding(_state_5d(), "auto", axis_name="batch", devices=[object(), object()])
    species_sharding = resolve_state_sharding(_state_6d(), "species", axis_name="batch", devices=[object(), object()])

    assert ky_sharding.mesh == "mesh:batch"
    assert ky_sharding.spec == sharding_mod.PartitionSpec(None, None, "batch", None, None)
    assert species_sharding.spec == sharding_mod.PartitionSpec("batch", None, None, None, None, None)

    with pytest.raises(ValueError, match="Cannot shard"):
        resolve_state_sharding(_state_5d(), "species", devices=[object(), object()])
    with pytest.raises(ValueError, match="5 or 6 dimensions"):
        resolve_state_sharding(jnp.zeros((2, 2)), "ky", devices=[object(), object()])
