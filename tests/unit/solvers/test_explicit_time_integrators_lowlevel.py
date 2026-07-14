from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk.solvers.time.explicit as eti
from spectraxgk.solvers.time.explicit_steps import _linear_explicit_stage_update
from spectraxgk.terms.config import FieldState


def _cache() -> SimpleNamespace:
    return SimpleNamespace(
        ky=jnp.asarray([0.0, 0.3]),
        kx=jnp.asarray([0.0, 0.2]),
        dealias_mask=jnp.asarray([[True, True], [True, True]]),
    )


def test_explicit_time_lowlevel_array_and_maximum_helpers() -> None:
    empty_grid = SimpleNamespace(
        ky=np.asarray([]), kx=np.asarray([0.0]), z=np.asarray([0.0]), ky_mode=None
    )
    kx, ky, kz = eti._cfl_wavenumber_arrays(empty_grid)
    assert kx.tolist() == [0.0]
    assert ky.size == 0
    assert kz.tolist() == [0.0]

    sliced_grid = SimpleNamespace(
        ky=np.asarray([0.3]),
        kx=np.asarray([0.0]),
        z=np.asarray([0.0, 1.0, 2.0, 3.0]),
        ky_mode=np.asarray([3]),
    )
    _kx, ky, kz = eti._cfl_wavenumber_arrays(sliced_grid)
    np.testing.assert_allclose(ky, [0.3])
    np.testing.assert_allclose(kz, [0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])

    assert eti._laguerre_velocity_max(0) == 0.0
    assert eti._gradient_ratio_max(np.asarray([]), np.asarray([])) == 0.0
    assert eti._gradient_ratio_max(
        np.asarray([2.0]), np.asarray([0.0])
    ) == pytest.approx(1.0e6)


def test_instantaneous_growth_rate_step_max_mode_and_invalid_method() -> None:
    phi_prev = jnp.asarray([[[1.0 + 1.0j, 2.0 + 0.5j]]])
    phi_now = jnp.asarray([[[2.0 + 2.0j, 3.0 + 4.0j]]])
    mask = jnp.asarray([[True]])

    gamma, omega = eti._instantaneous_growth_rate_step(
        phi_now, phi_prev, 0.5, z_index=0, mask=mask, mode_method="max"
    )

    assert gamma.shape == (1, 1)
    assert omega.shape == (1, 1)
    assert np.isfinite(np.asarray(gamma[0, 0]))
    with pytest.raises(ValueError, match="mode_method"):
        eti._instantaneous_growth_rate_step(
            phi_now, phi_prev, 0.5, z_index=0, mask=mask, mode_method="bad"
        )


@pytest.mark.parametrize(
    "method", ["euler", "rk2", "rk3_classic", "rk3", "rk3_heun", "rk4", "sspx3", "k10"]
)
def test_linear_explicit_step_methods_match_scalar_linear_amplification(
    monkeypatch, method: str
) -> None:
    rate = 0.2 - 0.1j

    def fake_assemble(state, cache, params, terms=None, dt=None):
        return rate * state, FieldState(phi=jnp.sum(state, axis=0))

    monkeypatch.setattr(eti, "assemble_rhs_cached", fake_assemble)
    G0 = jnp.ones((1, 1, 2, 2, 1), dtype=jnp.complex64)

    G1, fields = eti._linear_explicit_step(
        G0, _cache(), object(), object(), 0.05, method=method
    )

    assert G1.shape == G0.shape
    assert fields.phi.shape == (1, 2, 2, 1)
    assert np.all(np.isfinite(np.asarray(G1)))


def test_linear_explicit_step_rejects_unknown_method(monkeypatch) -> None:
    monkeypatch.setattr(
        eti,
        "assemble_rhs_cached",
        lambda state, cache, params, terms=None, dt=None: (
            state,
            FieldState(phi=jnp.sum(state, axis=0)),
        ),
    )

    with pytest.raises(ValueError, match="explicit linear method"):
        eti._linear_explicit_step(
            jnp.ones((1, 1, 2, 2, 1), dtype=jnp.complex64),
            _cache(),
            object(),
            object(),
            0.05,
            method="bad",
        )


@pytest.mark.parametrize(("method", "expected_calls"), [("sspx3", 3), ("k10", 10)])
def test_self_staging_explicit_methods_do_not_evaluate_unused_rhs(
    method: str, expected_calls: int
) -> None:
    calls = 0

    def rhs(state):
        nonlocal calls
        calls += 1
        return 0.2 * state

    state = jnp.asarray([1.0])
    result = _linear_explicit_stage_update(
        state, jnp.asarray(0.1), method_key=method, rhs=rhs
    )

    assert calls == expected_calls
    assert np.all(np.isfinite(np.asarray(result)))


def test_explicit_from_config_preserves_adaptive_controls(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(eti, "build_linear_cache", lambda *_args: "cache")

    def fake_integrate(_state, _grid, cache, _params, _geom, config, *_args, **kwargs):
        captured.update(cache=cache, config=config, kwargs=kwargs)
        return np.asarray([0.1]), np.ones((1, 1, 1, 2)), None, None, None

    monkeypatch.setattr(eti, "integrate_linear_explicit_diagnostics", fake_integrate)
    time_cfg = SimpleNamespace(
        dt=0.02, t_max=2.0, sample_stride=3, fixed_dt=False,
        dt_min=1.0e-6, dt_max=0.04, cfl=0.7, method="rk2", cfl_fac=None,
        use_dealias_mask=True,
    )
    t, phi = eti.integrate_linear_explicit_from_config(
        jnp.ones((1,)), object(), object(), object(), time_cfg,
        Nl=2, Nm=3, z_index=1, show_progress=True,
    )

    config = captured["config"]
    assert config.dt == pytest.approx(0.02)
    assert config.t_max == pytest.approx(2.0)
    assert config.sample_stride == 3
    assert config.fixed_dt is False
    assert config.use_dealias_mask is True
    assert config.dt_max == pytest.approx(0.04)
    assert config.cfl == pytest.approx(0.7)
    assert captured["cache"] == "cache"
    assert captured["kwargs"]["show_progress"] is True
    np.testing.assert_allclose(t, [0.1])
    assert phi.shape == (1, 1, 1, 2)
