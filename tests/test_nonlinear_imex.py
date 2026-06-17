from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from spectraxgk.solvers.nonlinear.imex import (
    advance_imex_nonlinear_state,
    imex_fixed_point_guess,
    integrate_cached_imex_scan,
    make_imex_nonlinear_term,
    make_imex_solve_step,
    solve_imex_step,
)
from spectraxgk.terms.config import FieldState


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


def test_make_imex_nonlinear_term_forwards_injected_kernels() -> None:
    seen: dict[str, object] = {}

    def fields_fn(*_args, **_kwargs):
        return "fields"

    def contribution_fn(*_args, **_kwargs):
        return "contribution"

    def nonlinear_kernel(G, cache, params, terms, **kwargs):
        seen["G"] = G
        seen["cache"] = cache
        seen["params"] = params
        seen["terms"] = terms
        seen["fields_fn"] = kwargs["fields_fn"]
        seen["contribution_fn"] = kwargs["nonlinear_contribution_fn"]
        seen["real_dtype"] = kwargs["real_dtype"]
        seen["external_phi"] = kwargs["external_phi"]
        seen["compressed_real_fft"] = kwargs["compressed_real_fft"]
        seen["laguerre_mode"] = kwargs["laguerre_mode"]
        return G + 2.0

    cache = SimpleNamespace(name="cache")
    params = SimpleNamespace(name="params")
    terms = SimpleNamespace(name="terms")
    term = make_imex_nonlinear_term(
        cache,
        params,
        terms,
        real_dtype=jnp.float32,
        external_phi=3.0,
        compressed_real_fft=False,
        laguerre_mode="spectral",
        fields_fn=fields_fn,
        nonlinear_term_fn=nonlinear_kernel,
        nonlinear_contribution_fn=contribution_fn,
    )
    G = jnp.asarray([1.0], dtype=jnp.float32)

    out = term(G)

    np.testing.assert_allclose(np.asarray(out), [3.0])
    assert seen["G"] is G
    assert seen["cache"] is cache
    assert seen["params"] is params
    assert seen["terms"] is terms
    assert seen["fields_fn"] is fields_fn
    assert seen["contribution_fn"] is contribution_fn
    assert seen["real_dtype"] is jnp.float32
    assert seen["external_phi"] == 3.0
    assert seen["compressed_real_fft"] is False
    assert seen["laguerre_mode"] == "spectral"


def test_make_imex_solve_step_forwards_solver_policy() -> None:
    seen: dict[str, object] = {}

    def solve_step_fn(G_in, G_rhs, **kwargs):
        seen["G_in"] = G_in
        seen["G_rhs"] = G_rhs
        seen.update(kwargs)
        return G_rhs + 4.0

    def linear_rhs_fn(g, *_args, **_kwargs):
        return g, None

    def matvec(flat):
        return flat

    def precond(flat):
        return flat
    solve_step = make_imex_solve_step(
        linear_rhs_fn=linear_rhs_fn,
        cache=SimpleNamespace(name="cache"),
        params=SimpleNamespace(name="params"),
        linear_cfg=SimpleNamespace(name="linear"),
        external_phi=None,
        dt_val=jnp.asarray(0.2, dtype=jnp.float32),
        implicit_iters=3,
        implicit_relax=0.5,
        matvec=matvec,
        shape=(1,),
        implicit_tol=1.0e-5,
        implicit_maxiter=7,
        implicit_restart=2,
        implicit_solve_method="batched",
        precond_op=precond,
        solve_step_fn=solve_step_fn,
    )
    G_in = jnp.asarray([1.0], dtype=jnp.float32)
    G_rhs = jnp.asarray([2.0], dtype=jnp.float32)

    out = solve_step(G_in, G_rhs)

    np.testing.assert_allclose(np.asarray(out), [6.0])
    assert seen["G_in"] is G_in
    assert seen["G_rhs"] is G_rhs
    assert seen["linear_rhs_fn"] is linear_rhs_fn
    assert seen["matvec"] is matvec
    assert seen["precond_op"] is precond
    assert seen["implicit_iters"] == 3
    assert seen["implicit_relax"] == 0.5
    assert seen["shape"] == (1,)
    assert seen["implicit_tol"] == 1.0e-5
    assert seen["implicit_maxiter"] == 7
    assert seen["implicit_restart"] == 2
    assert seen["implicit_solve_method"] == "batched"


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


def test_integrate_cached_imex_scan_owns_cached_scan_policy(monkeypatch) -> None:
    G0 = jnp.zeros((1,), dtype=jnp.complex64)
    fields = FieldState(phi=jnp.zeros((1,), dtype=jnp.complex64), apar=None, bpar=None)
    build_calls: list[dict[str, object]] = []
    nonlinear_calls: list[str] = []
    linear_calls: list[str] = []

    def build_operator_fn(G, cache, params, dt, **kwargs):
        build_calls.append(kwargs)
        return SimpleNamespace(
            shape=tuple(G.shape),
            dt_val=jnp.asarray(dt, dtype=jnp.float32),
            precond_op=None,
            matvec=lambda x: x,
            squeeze_species=False,
            state_dtype=G.dtype,
        )

    def linear_rhs_fn(G, *_args, **_kwargs):
        linear_calls.append("linear")
        return jnp.zeros_like(G), fields

    def nonlinear_kernel(G, cache, params, terms, **kwargs):
        del cache, params, terms
        assert kwargs["fields_fn"] is fields_fn
        assert kwargs["nonlinear_contribution_fn"] is contribution_fn
        assert kwargs["compressed_real_fft"] is False
        assert kwargs["laguerre_mode"] == "spectral"
        nonlinear_calls.append("nonlinear")
        return jnp.ones_like(G)

    def fields_fn(*_args, **_kwargs):
        return fields

    def contribution_fn(*_args, **_kwargs):
        return jnp.asarray(0.0, dtype=jnp.float32)

    monkeypatch.setattr(
        "spectraxgk.solvers.nonlinear.imex.jax.scipy.sparse.linalg.gmres",
        lambda matvec, rhs, **kwargs: (rhs, SimpleNamespace(success=True)),
    )

    G_out, fields_t = integrate_cached_imex_scan(
        G0,
        SimpleNamespace(name="cache"),
        SimpleNamespace(name="params"),
        0.2,
        2,
        term_cfg=SimpleNamespace(name="terms"),
        linear_cfg=SimpleNamespace(name="linear"),
        linear_rhs_fn=linear_rhs_fn,
        build_operator_fn=build_operator_fn,
        fields_fn=fields_fn,
        nonlinear_term_fn=nonlinear_kernel,
        nonlinear_contribution_fn=contribution_fn,
        implicit_iters=0,
        compressed_real_fft=False,
        laguerre_mode="spectral",
    )

    assert build_calls
    assert build_calls[0]["terms"].name == "linear"
    np.testing.assert_allclose(np.asarray(G_out), np.asarray([0.4]), rtol=1e-6)
    assert fields_t.phi.shape == (2, 1)
    assert nonlinear_calls == ["nonlinear"]
    assert linear_calls
