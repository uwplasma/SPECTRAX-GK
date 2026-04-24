from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk.gx_integrators as gx
from spectraxgk.terms.config import FieldState


def _cache() -> SimpleNamespace:
    return SimpleNamespace(
        ky=jnp.asarray([0.0, 0.3]),
        kx=jnp.asarray([0.0, 0.2]),
        dealias_mask=jnp.asarray([[True, True], [True, True]]),
    )


def test_gx_lowlevel_array_and_maximum_helpers() -> None:
    empty_grid = SimpleNamespace(ky=np.asarray([]), kx=np.asarray([0.0]), z=np.asarray([0.0]), ky_mode=None)
    kx, ky, kz = gx._gx_k_arrays(empty_grid)
    assert kx.tolist() == [0.0]
    assert ky.size == 0
    assert kz.tolist() == [0.0]

    sliced_grid = SimpleNamespace(
        ky=np.asarray([0.3]),
        kx=np.asarray([0.0]),
        z=np.asarray([0.0, 1.0, 2.0, 3.0]),
        ky_mode=np.asarray([3]),
    )
    _kx, ky, kz = gx._gx_k_arrays(sliced_grid)
    np.testing.assert_allclose(ky, [0.3])
    np.testing.assert_allclose(kz, [0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])

    assert gx._gx_laguerre_vmax(0) == 0.0
    assert gx._gx_eta_max(np.asarray([]), np.asarray([])) == 0.0
    assert gx._gx_eta_max(np.asarray([2.0]), np.asarray([0.0])) == pytest.approx(1.0e6)


def test_gx_growth_rate_step_max_mode_and_invalid_method() -> None:
    phi_prev = jnp.asarray([[[1.0 + 1.0j, 2.0 + 0.5j]]])
    phi_now = jnp.asarray([[[2.0 + 2.0j, 3.0 + 4.0j]]])
    mask = jnp.asarray([[True]])

    gamma, omega = gx._gx_growth_rate_step(phi_now, phi_prev, 0.5, z_index=0, mask=mask, mode_method="max")

    assert gamma.shape == (1, 1)
    assert omega.shape == (1, 1)
    assert np.isfinite(np.asarray(gamma[0, 0]))
    with pytest.raises(ValueError, match="mode_method"):
        gx._gx_growth_rate_step(phi_now, phi_prev, 0.5, z_index=0, mask=mask, mode_method="bad")


@pytest.mark.parametrize("method", ["euler", "rk2", "rk3_classic", "rk3", "rk3_gx", "rk4", "sspx3", "k10"])
def test_linear_explicit_step_methods_match_scalar_linear_amplification(monkeypatch, method: str) -> None:
    rate = 0.2 - 0.1j

    def fake_assemble(state, cache, params, terms=None, dt=None):
        return rate * state, FieldState(phi=jnp.sum(state, axis=0))

    monkeypatch.setattr(gx, "assemble_rhs_cached", fake_assemble)
    G0 = jnp.ones((1, 1, 2, 2, 1), dtype=jnp.complex64)

    G1, fields = gx._linear_explicit_step(G0, _cache(), object(), object(), 0.05, method=method)

    assert G1.shape == G0.shape
    assert fields.phi.shape == (1, 2, 2, 1)
    assert np.all(np.isfinite(np.asarray(G1)))


def test_linear_explicit_step_rejects_unknown_method(monkeypatch) -> None:
    monkeypatch.setattr(
        gx,
        "assemble_rhs_cached",
        lambda state, cache, params, terms=None, dt=None: (state, FieldState(phi=jnp.sum(state, axis=0))),
    )

    with pytest.raises(ValueError, match="GX linear method"):
        gx._linear_explicit_step(
            jnp.ones((1, 1, 2, 2, 1), dtype=jnp.complex64),
            _cache(),
            object(),
            object(),
            0.05,
            method="bad",
        )
