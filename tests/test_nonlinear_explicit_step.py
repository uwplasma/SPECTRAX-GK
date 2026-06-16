from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.solvers.nonlinear.explicit import advance_explicit_nonlinear_state


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
