from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from spectraxgk.solvers.nonlinear.imex import (
    advance_imex_nonlinear_state,
    imex_fixed_point_guess,
    solve_imex_step,
)


def test_imex_fixed_point_guess_applies_linear_predictor_iterations() -> None:
    def linear_rhs(g, *_args, **_kwargs):
        return g, None

    out = imex_fixed_point_guess(
        jnp.asarray([0.0], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        linear_rhs_fn=linear_rhs,
        cache=SimpleNamespace(),
        params=SimpleNamespace(),
        linear_cfg=SimpleNamespace(),
        external_phi=None,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        implicit_iters=2,
        implicit_relax=1.0,
    )

    np.testing.assert_allclose(np.asarray(out), [1.1], rtol=1e-6)


def test_solve_imex_step_identity_system_returns_rhs_shape() -> None:
    def linear_rhs(g, *_args, **_kwargs):
        return jnp.zeros_like(g), None

    G_rhs = jnp.asarray([[2.0]], dtype=jnp.float32)
    out = solve_imex_step(
        jnp.zeros_like(G_rhs),
        G_rhs,
        linear_rhs_fn=linear_rhs,
        cache=SimpleNamespace(),
        params=SimpleNamespace(),
        linear_cfg=SimpleNamespace(),
        external_phi=None,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        implicit_iters=0,
        implicit_relax=1.0,
        matvec=lambda flat: flat,
        shape=tuple(G_rhs.shape),
        implicit_tol=1.0e-8,
        implicit_maxiter=20,
        implicit_restart=5,
        implicit_solve_method="batched",
    )

    np.testing.assert_allclose(np.asarray(out), np.asarray(G_rhs), rtol=1e-6)


def test_advance_imex_nonlinear_state_default_method_solves_rhs() -> None:
    calls: list[float] = []

    def nonlinear_term(g):
        return 2.0 * g

    def solve_step(g_in, rhs):
        calls.append(float(np.asarray(g_in[0])))
        return rhs + 1.0

    out = advance_imex_nonlinear_state(
        jnp.asarray([1.0], dtype=jnp.float32),
        dt_val=jnp.asarray(0.25, dtype=jnp.float32),
        method="imex",
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        project_state=lambda g: g,
    )

    np.testing.assert_allclose(np.asarray(out), [2.5], rtol=1e-6)
    assert calls == [1.0]


def test_advance_imex_nonlinear_state_sspx3_matches_constant_rhs_step() -> None:
    def nonlinear_term(g):
        return jnp.ones_like(g) * 2.0

    def solve_step(_g_in, rhs):
        return rhs

    out = advance_imex_nonlinear_state(
        jnp.asarray([1.0], dtype=jnp.float32),
        dt_val=jnp.asarray(0.25, dtype=jnp.float32),
        method="sspx3",
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        project_state=lambda g: g,
    )

    np.testing.assert_allclose(np.asarray(out), [1.5], rtol=1e-6)
