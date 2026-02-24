"""Validation helper and package-lazy-import tests for term modules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectraxgk.terms as term_pkg
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.validation import _check_nonnegative, _check_positive


def test_scalar_validation_checks() -> None:
    _check_positive(1.0, "x")
    _check_nonnegative(0.0, "x")
    with pytest.raises(ValueError):
        _check_positive(0.0, "x")
    with pytest.raises(ValueError):
        _check_nonnegative(-1.0, "x")


def test_array_validation_checks() -> None:
    _check_positive(jnp.asarray([1.0, 2.0]), "arr")
    _check_nonnegative(jnp.asarray([0.0, 2.0]), "arr")
    with pytest.raises(ValueError):
        _check_positive(jnp.asarray([1.0, 0.0]), "arr")
    with pytest.raises(ValueError):
        _check_nonnegative(jnp.asarray([1.0, -0.1]), "arr")


def test_validation_skips_tracer_runtime_checks() -> None:
    @jax.jit
    def f(x: jnp.ndarray) -> jnp.ndarray:
        _check_positive(x, "x")
        _check_nonnegative(x, "x")
        return x + 1.0

    out = f(jnp.asarray(0.0))
    assert float(out) == 1.0


def test_terms_package_lazy_exports() -> None:
    assert callable(term_pkg.assemble_rhs)
    assert callable(term_pkg.assemble_rhs_cached)
    assert callable(term_pkg.assemble_rhs_cached_jit)
    assert callable(term_pkg.integrate_nonlinear)
    with pytest.raises(AttributeError):
        _ = term_pkg.not_a_real_symbol


def test_terms_config_pytrees_roundtrip() -> None:
    cfg = TermConfig(streaming=0.5, nonlinear=0.25, bpar=0.0)
    leaves, treedef = jax.tree_util.tree_flatten(cfg)
    cfg_rt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert cfg_rt == cfg

    state = FieldState(phi=jnp.ones((2, 2)), apar=jnp.zeros((2, 2)), bpar=None)
    leaves_s, tree_s = jax.tree_util.tree_flatten(state)
    state_rt = jax.tree_util.tree_unflatten(tree_s, leaves_s)
    assert jnp.allclose(state_rt.phi, state.phi)
    assert jnp.allclose(state_rt.apar, state.apar)
    assert state_rt.bpar is None
