from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.solvers.nonlinear.explicit import (
    advance_explicit_nonlinear_state,
    integrate_cached_explicit_scan,
    run_explicit_diagnostic_scan,
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


def test_run_explicit_diagnostic_scan_dense_path_runs_all_steps() -> None:
    def step_fn(carry, idx):
        G, G_prev, fields_prev, diag_prev, t_prev, dt_prev = carry
        del G_prev, fields_prev, diag_prev
        G_next = G + 1
        t_next = t_prev + dt_prev
        diag = G_next + idx
        return (G_next, G_next, G_next, diag, t_next, dt_prev), (
            diag,
            t_next,
            dt_prev,
        )

    G_final, (diag, t, dt_series) = run_explicit_diagnostic_scan(
        step_fn,
        (
            jnp.asarray(0),
            jnp.asarray(0),
            jnp.asarray(0),
            jnp.asarray(0),
            jnp.asarray(0.0),
            jnp.asarray(0.5),
        ),
        steps=3,
        stride=1,
        sampled_scan=False,
        checkpoint=False,
        sampled_scan_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sampled path should not run")
        ),
    )

    assert int(G_final) == 3
    np.testing.assert_allclose(np.asarray(diag), [1, 3, 5])
    np.testing.assert_allclose(np.asarray(t), [0.5, 1.0, 1.5])
    np.testing.assert_allclose(np.asarray(dt_series), [0.5, 0.5, 0.5])


def test_run_explicit_diagnostic_scan_sampled_path_forwards_policy() -> None:
    seen: dict[str, object] = {}

    def step_fn(carry, idx):
        del idx
        return carry, carry[-3:]

    initial = (
        jnp.asarray(1),
        jnp.asarray(2),
        jnp.asarray(3),
        jnp.asarray(4),
        jnp.asarray(5),
        jnp.asarray(6),
    )

    def sampled_scan_fn(step, carry, **kwargs):
        seen["step"] = step
        seen["carry"] = carry
        seen.update(kwargs)
        return carry, (jnp.asarray([7]), jnp.asarray([8]), jnp.asarray([9]))

    G_final, scan_diag_out = run_explicit_diagnostic_scan(
        step_fn,
        initial,
        steps=5,
        stride=2,
        sampled_scan=True,
        checkpoint=False,
        sampled_scan_fn=sampled_scan_fn,
    )

    assert int(G_final) == 1
    assert seen["step"] is step_fn
    assert seen["carry"] is initial
    assert seen["steps"] == 5
    assert seen["stride"] == 2
    np.testing.assert_allclose(np.asarray(scan_diag_out[0]), [7])
