"""Tests for modular nonlinear scan integrator utilities."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectraxgk.terms.config import FieldState
from spectraxgk.terms.integrators import integrate_nonlinear
from spectraxgk.terms.nonlinear import exb_nonlinear_contribution, placeholder_nonlinear_contribution


def _linear_rhs(rate: complex):
    def rhs_fn(G: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        dG = rate * G
        phi = jnp.sum(G, axis=0)
        return dG, FieldState(phi=phi)

    return rhs_fn


@pytest.mark.parametrize(
    ("method", "one_step_factor"),
    [
        ("euler", lambda a: 1.0 + a),
        ("rk2", lambda a: 1.0 + a + 0.5 * a * a),
        ("rk3", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
        ("rk4", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0 + (a * a * a * a) / 24.0),
    ],
)
def test_integrate_nonlinear_methods_match_linear_amplification(method, one_step_factor) -> None:
    dt = 0.1
    steps = 4
    rate = 0.3 - 0.2j
    G0 = jnp.asarray([[1.0 + 0.0j, 0.5 + 0.25j]], dtype=jnp.complex64)
    G_final, fields = integrate_nonlinear(
        _linear_rhs(rate),
        G0,
        dt,
        steps,
        method=method,
        checkpoint=True,
    )
    a = rate * dt
    expected = (one_step_factor(a) ** steps) * G0
    assert G_final.shape == G0.shape
    assert fields.phi.shape[0] == steps
    assert jnp.allclose(G_final, expected, rtol=3.0e-3, atol=3.0e-3)


def test_integrate_nonlinear_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        integrate_nonlinear(_linear_rhs(0.1 + 0.0j), jnp.ones((2, 2), dtype=jnp.complex64), 0.1, 4, method="bad")


def test_nonlinear_placeholders() -> None:
    G = jnp.ones((3, 4, 1), dtype=jnp.complex64)
    out = placeholder_nonlinear_contribution(G, weight=jnp.asarray(2.0))
    assert jnp.allclose(out, 0.0)
    exb = exb_nonlinear_contribution(
        G,
        phi=jnp.ones((3, 4, 1), dtype=jnp.complex64),
        dealias_mask=jnp.ones((3, 4), dtype=bool),
        kx_grid=jnp.ones((3, 4), dtype=jnp.float32),
        ky_grid=jnp.ones((3, 4), dtype=jnp.float32),
        weight=jnp.asarray(1.0),
        gx_real_fft=False,
    )
    assert exb.shape == G.shape
