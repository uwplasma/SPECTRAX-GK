from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from spectraxgk.sharded_integrators import integrate_linear_sharded


def test_integrate_linear_sharded_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError):
        integrate_linear_sharded(jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64), None, None, dt=0.1, steps=0)


def test_integrate_linear_sharded_runs_with_mocked_pjit(monkeypatch) -> None:
    calls = {"rhs": 0, "shard": 0, "put": 0}

    def fake_rhs(G, cache, params, terms=None, dt=None):
        calls["rhs"] += 1
        return jnp.ones_like(G), None

    monkeypatch.setattr("spectraxgk.sharded_integrators.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("spectraxgk.sharded_integrators.pjit", lambda fn, **kwargs: fn)
    monkeypatch.setattr(
        "spectraxgk.sharded_integrators.jax.lax.with_sharding_constraint",
        lambda state, sharding: calls.__setitem__("shard", calls["shard"] + 1) or state,
    )
    monkeypatch.setattr(
        "spectraxgk.sharded_integrators.jax.device_put",
        lambda state, sharding: calls.__setitem__("put", calls["put"] + 1) or state,
    )

    G0 = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64)
    out = integrate_linear_sharded(G0, SimpleNamespace(), SimpleNamespace(), dt=0.5, steps=2, state_sharding="mesh")
    assert out.shape == G0.shape
    assert calls["rhs"] >= 2
    assert calls["put"] == 1
    assert calls["shard"] >= 3
