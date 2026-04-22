"""Tests for modular nonlinear scan integrator utilities."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.terms.config import FieldState
from spectraxgk.terms.integrators import integrate_nonlinear
from spectraxgk.terms.nonlinear import exb_nonlinear_contribution, placeholder_nonlinear_contribution

_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


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
        ("rk3_gx", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
        ("rk3_classic", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
        ("rk4", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0 + (a * a * a * a) / 24.0),
        ("sspx3", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
        ],
        )

def test_integrate_nonlinear_methods_match_linear_amplification(method, one_step_factor) -> None:
    dt = 0.1
    steps = 4
    rate = 0.3 - 0.2j
    G0 = jnp.asarray([[1.0 + 0.0j, 0.5 + 0.25j]], dtype=jnp.complex64)
    G0_ref = jnp.array(G0)
    G_final, fields = integrate_nonlinear(
        _linear_rhs(rate),
        G0,
        dt,
        steps,
        method=method,
        checkpoint=True,
    )
    a = rate * dt
    expected = (one_step_factor(a) ** steps) * G0_ref
    assert G_final.shape == G0.shape
    assert fields.phi.shape[0] == steps
    assert jnp.allclose(G_final, expected, rtol=3.0e-3, atol=3.0e-3)


def test_integrate_nonlinear_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        integrate_nonlinear(_linear_rhs(0.1 + 0.0j), jnp.ones((2, 2), dtype=jnp.complex64), 0.1, 4, method="bad")


def test_integrate_nonlinear_projects_each_stage() -> None:
    def rhs_fn(G: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        flipped = jnp.flip(G, axis=-2)
        return 1j * flipped, FieldState(phi=jnp.sum(G, axis=0))

    def projector(G: jnp.ndarray) -> jnp.ndarray:
        pos = G[..., :3, :]
        neg = jnp.conj(pos[..., 1:2, :])[..., ::-1, :]
        return jnp.concatenate([pos, neg], axis=-2)

    G0 = jnp.asarray([[1.0 + 0.0j], [2.0 + 1.0j], [-3.0 + 0.5j], [7.0 - 2.0j]], dtype=jnp.complex64)
    G_final, _fields = integrate_nonlinear(
        rhs_fn,
        G0,
        0.1,
        2,
        method="rk4",
        project_state=projector,
    )
    assert jnp.allclose(G_final[..., 3, :], jnp.conj(G_final[..., 1, :]))


def test_integrate_nonlinear_rk3_alias_matches_gx_variant() -> None:
    G0 = jnp.asarray([[1.0 + 0.0j, 0.5 + 0.25j]], dtype=jnp.complex64)
    out_rk3, _ = integrate_nonlinear(_linear_rhs(0.3 - 0.2j), jnp.array(G0), 0.1, 3, method="rk3")
    out_gx, _ = integrate_nonlinear(_linear_rhs(0.3 - 0.2j), jnp.array(G0), 0.1, 3, method="rk3_gx")
    assert jnp.allclose(out_rk3, out_gx)


@pytest.mark.parametrize(
    ("method", "expected_order", "min_observed_order"),
    [
        ("rk2", 2.0, 1.75),
        ("rk3", 3.0, 2.6),
        ("rk4", 4.0, 3.3),
        ("sspx3", 3.0, 2.6),
    ],
)
def test_integrate_nonlinear_observed_order_against_exact_solution(
    method: str,
    expected_order: float,
    min_observed_order: float,
) -> None:
    rate = -1.1 + 0.7j
    t_final = 0.8
    G0 = jnp.asarray([[1.0 + 0.1j, -0.4 + 0.2j]], dtype=jnp.complex64)
    exact = np.exp(rate * t_final) * np.asarray(G0)

    errors: list[float] = []
    dts: list[float] = []
    for steps in (2, 4, 8, 16):
        dt = t_final / steps
        out, _ = integrate_nonlinear(_linear_rhs(rate), jnp.array(G0), dt, steps, method=method)
        err = float(np.max(np.abs(np.asarray(out) - exact)))
        errors.append(err)
        dts.append(dt)

    observed_orders = [
        np.log(errors[i] / errors[i + 1]) / np.log(dts[i] / dts[i + 1])
        for i in range(len(errors) - 1)
        if errors[i] > 0.0 and errors[i + 1] > 0.0
    ]
    assert observed_orders, "expected non-zero errors to estimate convergence order"
    assert observed_orders[-1] >= min_observed_order
    assert observed_orders[-1] <= expected_order + 0.6


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
