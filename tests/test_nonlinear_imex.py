from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from spectraxgk.nonlinear_imex import imex_fixed_point_guess, solve_imex_step


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
