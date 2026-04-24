from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.benchmarking import estimate_observed_order
from spectraxgk.gyroaverage import J_l_all
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    _is_tracer,
    _as_species_array,
    _build_implicit_operator,
    _build_end_damping_profile_array,
    _build_gyroaverage_cache_arrays,
    _build_linked_end_damping_profile,
    _build_low_rank_moment_cache_arrays,
    _check_nonnegative,
    _check_positive,
    _integrate_linear_cached_impl,
    _resolve_implicit_preconditioner,
    _signed_to_index,
    _integrate_linear_implicit_cached,
    integrate_linear,
    integrate_linear_diagnostics,
    lenard_bernstein_eigenvalues,
    linear_terms_to_term_config,
    term_config_to_linear_terms,
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

    def fn(x):
        return x

    assert _resolve_implicit_preconditioner(fn) is fn


def test_is_tracer_and_lenard_bernstein_eigenvalues() -> None:
    assert _is_tracer(1.0) is False
    traced_flag = jax.jit(lambda x: jnp.asarray(1 if _is_tracer(x) else 0, dtype=jnp.int32))(1.0)
    assert int(traced_flag) == 1

    expected = np.asarray([[0.0, 0.3, 0.6], [0.7, 1.0, 1.3]], dtype=np.float32)
    got = np.asarray(lenard_bernstein_eigenvalues(2, 3, nu_hermite=0.3, nu_laguerre=0.7), dtype=np.float32)
    np.testing.assert_allclose(got, expected)


def test_low_rank_moment_and_damping_cache_match_expected_shapes_and_values() -> None:
    params = LinearParams(nu_hermite=0.3, nu_laguerre=0.7, p_hyper=2, p_hyper_l=3, p_hyper_m=4)
    cache = _build_low_rank_moment_cache_arrays(2, 3, params, jnp.float32)

    expected_lb = np.asarray([[0.0, 0.3, 0.6], [0.7, 1.0, 1.3]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(cache["lb_lam"]), expected_lb, rtol=1e-6)
    assert cache["lb_lam"].shape == (2, 3)
    assert cache["hyper_ratio"].shape == (2, 3, 1, 1, 1)
    assert cache["sqrt_p"].shape == (1, 1, 3, 1, 1, 1)
    assert cache["mask_const"].dtype == jnp.bool_

    periodic = np.asarray(_build_end_damping_profile_array(8, 0.25, "periodic", jnp.float32))
    linked = np.asarray(_build_end_damping_profile_array(8, 0.25, "linked", jnp.float32))
    np.testing.assert_allclose(periodic, np.zeros(8, dtype=np.float32))
    assert linked[0] > 0.0
    assert linked[-1] > 0.0


def test_gyroaverage_cache_helper_matches_species_vmap_convention() -> None:
    b = jnp.asarray(
        [
            [[[0.0, 0.2], [0.4, 0.6]]],
            [[[0.1, 0.3], [0.5, 0.7]]],
        ],
        dtype=jnp.float32,
    )
    Jl, JlB = _build_gyroaverage_cache_arrays(b, Nl=3, real_dtype=jnp.float32)
    expected = jax.vmap(lambda bs: J_l_all(bs, l_max=2))(b).astype(jnp.float32)

    np.testing.assert_allclose(np.asarray(Jl), np.asarray(expected), rtol=1e-6)
    assert Jl.shape == (2, 3, 1, 2, 2)
    assert JlB.shape == Jl.shape
    np.testing.assert_allclose(np.asarray(JlB[:, 0]), np.asarray(Jl[:, 0]), rtol=1e-6)


def test_linear_params_and_terms_roundtrip() -> None:
    params = LinearParams(charge_sign=jnp.asarray([1.0, -1.0]), nu=jnp.asarray([0.1, 0.2]), beta=0.3)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(restored.charge_sign), [1.0, -1.0])
    np.testing.assert_allclose(np.asarray(restored.nu), [0.1, 0.2])
    assert float(restored.beta) == pytest.approx(0.3)

    terms = LinearTerms(apar=0.0, bpar=0.0, hyperdiffusion=1.0)
    term_cfg = linear_terms_to_term_config(terms, nonlinear=0.25)
    assert float(term_cfg.nonlinear) == pytest.approx(0.25)
    assert term_config_to_linear_terms(term_cfg) == terms


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


def test_build_implicit_operator_preconditioner_aliases_and_errors(monkeypatch) -> None:
    G0 = jnp.zeros((1, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        l=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        m=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        cv_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        gb_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        bgrad=jnp.ones((2,), dtype=jnp.float32),
        sqrt_m_ladder=jnp.ones((2,), dtype=jnp.float32),
        sqrt_p=jnp.ones((2,), dtype=jnp.float32),
        kz=jnp.array([0.0, 1.0], dtype=jnp.float32),
        linked_indices=(),
        linked_kz=(),
        use_twist_shift=False,
    )
    params = SimpleNamespace(
        nu=jnp.asarray([0.1], dtype=jnp.float32),
        tz=jnp.asarray([1.0], dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
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

    size = int(np.prod(np.asarray(G0.shape)))
    x = jnp.ones((size,), dtype=G0.dtype)
    for key in ("pas-coarse", "hermite-line-coarse", "identity"):
        _G, _shape, _size, _dt_val, precond_op, _matvec, _squeeze = _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.2,
            terms=LinearTerms(),
            implicit_preconditioner=key,
        )
        y = precond_op(x)
        assert y.shape == x.shape
        assert np.isfinite(np.asarray(y)).all()

    with pytest.raises(ValueError):
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.2,
            terms=LinearTerms(),
            implicit_preconditioner="not-a-preconditioner",
        )


def test_integrate_linear_wrapper_routes_methods(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr("spectraxgk.linear.build_linear_cache", lambda *args, **kwargs: "cache")
    monkeypatch.setattr(
        "spectraxgk.linear._integrate_linear_cached",
        lambda *args, **kwargs: calls.append(("cached", kwargs["method"])) or ("G", "phi"),
    )
    monkeypatch.setattr(
        "spectraxgk.linear._integrate_linear_cached_donate",
        lambda *args, **kwargs: calls.append(("donate", kwargs["method"])) or ("Gd", "phid"),
    )
    monkeypatch.setattr(
        "spectraxgk.linear._integrate_linear_implicit_cached",
        lambda *args, **kwargs: calls.append(("implicit", "implicit")) or ("Gi", "phii"),
    )

    assert integrate_linear(G0, grid, geom, params, dt=0.1, steps=2, method="semi-implicit") == ("G", "phi")
    assert integrate_linear(G0, grid, geom, params, dt=0.1, steps=2, method="rk2", donate=True) == ("Gd", "phid")
    assert integrate_linear(G0, grid, geom, params, dt=0.1, steps=2, method="implicit") == ("Gi", "phii")
    assert ("cached", "imex") in calls
    assert ("donate", "rk2") in calls
    assert ("implicit", "implicit") in calls

    with pytest.raises(ValueError):
        integrate_linear(G0, grid, geom, params, dt=0.1, steps=2, sample_stride=0)
    with pytest.raises(ValueError):
        integrate_linear(G0, grid, geom, params, dt=0.1, steps=3, sample_stride=2)
    with pytest.raises(ValueError):
        integrate_linear(jnp.zeros((2, 2)), grid, geom, params, dt=0.1, steps=2)


def test_integrate_linear_cached_impl_invalid_and_sampled(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(lb_lam=jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.float32))
    params = SimpleNamespace(nu=0.0)
    monkeypatch.setattr(
        "spectraxgk.linear.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), jnp.zeros((1, 1, 2), dtype=jnp.complex64)),
    )

    with pytest.raises(ValueError):
        _integrate_linear_cached_impl(G0, cache, params, dt=0.1, steps=2, method="bad")

    G_out, phi_t = _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt=0.1,
        steps=4,
        method="euler",
        sample_stride=2,
    )
    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2


def test_integrate_linear_implicit_cached_sampled_path(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace()
    params = SimpleNamespace()

    monkeypatch.setattr(
        "spectraxgk.linear._build_implicit_operator",
        lambda *args, **kwargs: (
            G0,
            G0.shape,
            G0.size,
            jnp.asarray(0.1, dtype=jnp.float32),
            (lambda x: x),
            (lambda x: x),
            False,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.gmres",
        lambda matvec, rhs, **kwargs: (rhs, SimpleNamespace(success=True)),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), jnp.ones((1, 1, 2), dtype=jnp.complex64)),
    )

    G_out, phi_t = _integrate_linear_implicit_cached(
        G0,
        cache,
        params,
        dt=0.1,
        steps=4,
        sample_stride=2,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2


def test_integrate_linear_diagnostics_validates_and_records_energy(monkeypatch) -> None:
    G0 = jnp.zeros((1, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    cache = SimpleNamespace(
        lb_lam=jnp.zeros((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((1, 2, 1, 1, 2), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), jnp.ones((1, 1, 2), dtype=jnp.complex64)),
    )

    with pytest.raises(ValueError):
        integrate_linear_diagnostics(G0, grid, geom, params, dt=0.1, steps=2, sample_stride=0, cache=cache)
    with pytest.raises(ValueError):
        integrate_linear_diagnostics(G0, grid, geom, params, dt=0.1, steps=3, sample_stride=2, cache=cache)

    G_out, phi_t, density_t, hl_t = integrate_linear_diagnostics(
        G0,
        grid,
        geom,
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=4,
        method="rk2",
        cache=cache,
        sample_stride=2,
        species_index=0,
        record_hl_energy=True,
    )
    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape[0] == 2
    assert hl_t.shape[0] == 2


def test_integrate_linear_diagnostics_builds_cache_and_uses_imex2(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.zeros((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((1, 2, 1, 1, 2), dtype=jnp.float32),
    )
    build_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        "spectraxgk.linear.build_linear_cache",
        lambda grid, geom, params, Nl, Nm: build_calls.append((Nl, Nm)) or cache,
    )
    monkeypatch.setattr(
        "spectraxgk.linear.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), jnp.ones((1, 1, 2), dtype=jnp.complex64)),
    )

    G_out, phi_t, density_t = integrate_linear_diagnostics(
        G0,
        object(),
        object(),
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=2,
        method="imex2",
        cache=None,
        sample_stride=1,
        species_index=None,
        record_hl_energy=False,
    )

    assert build_calls == [(2, 2)]
    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape[0] == 2


def test_integrate_linear_diagnostics_imex_sampled_multispecies(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.zeros_like(G), jnp.ones((1, 1, 2), dtype=jnp.complex64)),
    )

    G_out, phi_t, density_t = integrate_linear_diagnostics(
        G0,
        object(),
        object(),
        SimpleNamespace(nu=jnp.asarray([0.1, 0.2], dtype=jnp.float32)),
        dt=0.1,
        steps=4,
        method="imex",
        cache=cache,
        sample_stride=2,
        species_index=None,
        record_hl_energy=False,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape == (2, 1, 1, 2)


def test_integrate_linear_diagnostics_species_none_and_5d_density_paths(monkeypatch) -> None:
    cache6 = SimpleNamespace(
        lb_lam=jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
    )
    cache5 = SimpleNamespace(
        lb_lam=jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((2, 1, 1, 2), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "spectraxgk.linear.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), jnp.ones((1, 1, 2), dtype=jnp.complex64)),
    )

    G6 = jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    G6_out, phi6_t, density6_t = integrate_linear_diagnostics(
        G6,
        object(),
        object(),
        SimpleNamespace(nu=jnp.asarray([0.0, 0.0])),
        dt=0.1,
        steps=2,
        method="sspx3",
        cache=cache6,
        sample_stride=1,
        species_index=None,
        record_hl_energy=False,
    )
    assert G6_out.shape == G6.shape
    assert phi6_t.shape[0] == 2
    assert density6_t.shape[0] == 2

    G5 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    G5_out, phi5_t, density5_t = integrate_linear_diagnostics(
        G5,
        object(),
        object(),
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=2,
        method="rk4",
        cache=cache5,
        sample_stride=1,
        species_index=0,
        record_hl_energy=False,
    )
    assert G5_out.shape == G5.shape
    assert phi5_t.shape[0] == 2
    assert density5_t.shape[0] == 2


@pytest.mark.parametrize(
    ("method", "rate", "damping", "min_order"),
    [
        ("rk2", -0.7 + 0.3j, 0.0, 1.75),
        ("rk4", -0.7 + 0.3j, 0.0, 3.2),
        ("sspx3", -0.7 + 0.3j, 0.0, 2.6),
        ("imex", -0.7 + 0.3j, 0.8, 0.95),
        ("imex2", -0.7 + 0.3j, 0.0, 1.9),
    ],
)
def test_integrate_linear_cached_impl_observed_order_against_exact_solution(
    monkeypatch,
    method: str,
    rate: complex,
    damping: float,
    min_order: float,
) -> None:
    cache = SimpleNamespace(lb_lam=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32))
    G0 = jnp.asarray([[[[[1.0 + 0.25j]]]]], dtype=jnp.complex64)
    params = LinearParams(nu=0.0)
    monkeypatch.setattr(
        "spectraxgk.linear.hypercollision_damping",
        lambda cache, params, dtype: jnp.ones_like(cache.lb_lam, dtype=dtype) * damping,
    )
    monkeypatch.setattr(
        "spectraxgk.linear.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.asarray(rate * G, dtype=G.dtype),
            jnp.asarray(G[0, 0], dtype=G.dtype),
        ),
    )

    t_final = 0.8
    step_sizes: list[float] = []
    errors: list[float] = []
    exact = np.exp(rate * t_final) * np.asarray(G0)
    for steps in (1, 2, 4, 8):
        dt = t_final / steps
        G_out, _phi_t = _integrate_linear_cached_impl(
            G0,
            cache,
            params,
            dt=dt,
            steps=steps,
            method=method,
        )
        errors.append(float(np.max(np.abs(np.asarray(G_out) - exact))))
        step_sizes.append(dt)

    metrics = estimate_observed_order(np.array(step_sizes), np.array(errors))
    assert metrics.asymptotic_order >= min_order
