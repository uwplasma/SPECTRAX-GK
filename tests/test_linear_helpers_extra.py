from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.linear import (
    LinearTerms,
    _as_species_array,
    _build_implicit_operator,
    _build_linked_end_damping_profile,
    _check_nonnegative,
    _check_positive,
    _resolve_implicit_preconditioner,
    _signed_to_index,
)


def test_linear_validation_helpers_scalar_and_array() -> None:
    _check_positive(1.0, "x")
    _check_nonnegative(0.0, "x")
    _check_positive(jnp.asarray([1.0, 2.0]), "arr")
    _check_nonnegative(jnp.asarray([0.0, 2.0]), "arr")

    with pytest.raises(ValueError):
        _check_positive(0.0, "x")
    with pytest.raises(ValueError):
        _check_nonnegative(-1.0, "x")
    with pytest.raises(ValueError):
        _check_positive(jnp.asarray([1.0, 0.0]), "arr")
    with pytest.raises(ValueError):
        _check_nonnegative(jnp.asarray([0.0, -1.0]), "arr")


def test_as_species_array_and_preconditioner_resolution() -> None:
    np.testing.assert_allclose(np.asarray(_as_species_array(2.0, 3, "nu")), [2.0, 2.0, 2.0])
    np.testing.assert_allclose(np.asarray(_as_species_array(jnp.asarray([1.0, 2.0]), 2, "nu")), [1.0, 2.0])
    with pytest.raises(ValueError):
        _as_species_array(jnp.asarray([1.0, 2.0]), 3, "nu")

    assert _resolve_implicit_preconditioner(None) == "auto"
    assert _resolve_implicit_preconditioner("  Damping ") == "damping"
    fn = lambda x: x
    assert _resolve_implicit_preconditioner(fn) is fn


def test_signed_to_index_and_linked_end_damping_profile() -> None:
    assert _signed_to_index(0, 3) == 0
    assert _signed_to_index(1, 3) == 1
    assert _signed_to_index(-1, 3) == 2
    assert _signed_to_index(-3, 3) == -1

    linked = (jnp.asarray([[1, 4]], dtype=jnp.int32),)
    profile = _build_linked_end_damping_profile(
        linked_indices=linked,
        ny=3,
        nx=2,
        nz=4,
        widthfrac=0.5,
        ky_mode=np.asarray([0, 1, -1], dtype=np.int32),
    )
    assert profile.shape == (3, 2, 4)
    assert np.max(profile) > 0.0
    assert np.all(profile[0] == 0.0)

    empty = _build_linked_end_damping_profile(
        linked_indices=(),
        ny=2,
        nx=2,
        nz=2,
        widthfrac=0.5,
    )
    assert np.allclose(empty, 0.0)

    with pytest.raises(ValueError):
        _build_linked_end_damping_profile(
            linked_indices=linked,
            ny=3,
            nx=2,
            nz=4,
            widthfrac=0.5,
            ky_mode=np.asarray([0, 1], dtype=np.int32),
        )


def test_build_implicit_operator_handles_species_squeeze(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        l=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
        m=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
        cv_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        gb_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        bgrad=jnp.ones((2,), dtype=jnp.float32),
        sqrt_m_ladder=jnp.ones((2,), dtype=jnp.float32),
        sqrt_p=jnp.ones((2,), dtype=jnp.float32),
        kz=jnp.array([0.0, 1.0], dtype=jnp.float32),
    )
    params = SimpleNamespace(
        nu=0.1,
        tz=1.0,
        vth=1.0,
        omega_d_scale=1.0,
        kpar_scale=1.0,
    )
    monkeypatch.setattr(
        "spectraxgk.linear.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), None),
    )

    G, shape, size, dt_val, precond_op, matvec, squeeze_species = _build_implicit_operator(
        G0,
        cache,
        params,
        dt=0.2,
        terms=LinearTerms(),
        implicit_preconditioner="damping",
    )

    assert squeeze_species is True
    assert shape == (1, 2, 2, 1, 1, 2)
    assert size == 8
    assert G.shape == shape
    assert np.isfinite(np.asarray(precond_op(G))).all()
    assert np.isfinite(np.asarray(matvec(G))).all()
    assert float(dt_val) == pytest.approx(0.2)
