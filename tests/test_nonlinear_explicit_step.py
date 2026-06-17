from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.solvers.nonlinear.explicit import (
    advance_explicit_nonlinear_state,
    integrate_cached_explicit_scan,
)


def _constant_rhs(value: float):
    def rhs_fn(G_state):
        return jnp.ones_like(G_state) * value, None

    return rhs_fn


def test_advance_explicit_nonlinear_state_euler_projects_and_preserves_dtype() -> None:
    G = jnp.asarray([1.0], dtype=jnp.float32)
    dG = jnp.asarray([3.0], dtype=jnp.float32)

    out = advance_explicit_nonlinear_state(
        G,
        dG,
        jnp.asarray(0.1, dtype=jnp.float32),
        method="euler",
        rhs_fn=_constant_rhs(0.0),
        project_state=lambda state: 2.0 * state,
        state_dtype=jnp.float32,
    )

    np.testing.assert_allclose(np.asarray(out), [2.6], rtol=1e-6)
    assert out.dtype == jnp.float32


@pytest.mark.parametrize("method", ["rk2", "rk3", "rk3_heun", "rk3_classic", "rk4"])
def test_advance_explicit_nonlinear_state_rk_methods_match_constant_rhs(
    method: str,
) -> None:
    G = jnp.asarray([1.0], dtype=jnp.float32)
    dG = jnp.asarray([2.0], dtype=jnp.float32)

    out = advance_explicit_nonlinear_state(
        G,
        dG,
        jnp.asarray(0.1, dtype=jnp.float32),
        method=method,
        rhs_fn=_constant_rhs(2.0),
        project_state=lambda state: state,
        state_dtype=jnp.float32,
    )

    np.testing.assert_allclose(np.asarray(out), [1.2], rtol=1e-6)


@pytest.mark.parametrize("method", ["sspx3", "k10"])
def test_advance_explicit_nonlinear_state_extended_methods_are_finite(
    method: str,
) -> None:
    G = jnp.asarray([1.0], dtype=jnp.float32)
    dG = jnp.asarray([0.5], dtype=jnp.float32)

    out = advance_explicit_nonlinear_state(
        G,
        dG,
        jnp.asarray(0.05, dtype=jnp.float32),
        method=method,
        rhs_fn=_constant_rhs(0.5),
        project_state=lambda state: state,
        state_dtype=jnp.float32,
    )

    assert out.shape == G.shape
    assert np.all(np.isfinite(np.asarray(out)))


def test_advance_explicit_nonlinear_state_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        advance_explicit_nonlinear_state(
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.asarray(0.1, dtype=jnp.float32),
            method="bogus",
            rhs_fn=_constant_rhs(0.0),
            project_state=lambda state: state,
            state_dtype=jnp.float32,
        )


def test_integrate_cached_explicit_scan_forwards_scan_policy() -> None:
    captured: dict[str, object] = {}
    G0 = jnp.asarray([1.0], dtype=jnp.float32)

    def rhs_fn(G):
        return jnp.ones_like(G), "fields"

    def project_state(G):
        return G + 2.0

    def scan_fn(rhs, G, dt, steps, **kwargs):
        captured["rhs"] = rhs
        captured["G"] = G
        captured["dt"] = dt
        captured["steps"] = steps
        captured.update(kwargs)
        dG, fields = rhs(G)
        return kwargs["project_state"](G + dt * steps * dG), fields

    G_out, fields = integrate_cached_explicit_scan(
        G0,
        0.25,
        4,
        method="rk4",
        rhs_fn=rhs_fn,
        scan_fn=scan_fn,
        checkpoint=True,
        project_state=project_state,
        show_progress=True,
    )

    np.testing.assert_allclose(np.asarray(G_out), [4.0], rtol=1e-6)
    assert fields == "fields"
    assert captured["rhs"] is rhs_fn
    assert captured["G"] is G0
    assert captured["dt"] == 0.25
    assert captured["steps"] == 4
    assert captured["method"] == "rk4"
    assert captured["checkpoint"] is True
    assert captured["project_state"] is project_state
    assert captured["show_progress"] is True
