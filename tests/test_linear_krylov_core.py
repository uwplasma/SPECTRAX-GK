"""Focused unit/regression tests for matrix-free Krylov utilities."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache, linear_terms_to_term_config
from spectraxgk import linear_krylov as lk


def _tiny_krylov_setup(*, linked: bool = False):
    grid_cfg = GridConfig(
        Nx=4 if linked else 2,
        Ny=4 if linked else 2,
        Nz=8,
        Lx=6.0,
        Ly=6.0,
        boundary="linked" if linked else "periodic",
        y0=20.0 if linked else None,
        jtwist=1 if linked else None,
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.01,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    Nl, Nm = 2, 4
    cache = build_linear_cache(grid, geom, params, Nl=Nl, Nm=Nm)
    v0 = jnp.ones((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64) * (1.0 + 0.1j)
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    term_cfg = linear_terms_to_term_config(terms)
    return grid, cache, params, v0, term_cfg, terms


def test_mode_family_and_target_selection_helpers() -> None:
    assert lk._mode_family_sign("cyclone") == 1
    assert lk._mode_family_sign("etg") == -1
    assert lk._mode_family_sign("other") == 0
    real = jnp.asarray([-0.1, 0.05, 0.08])
    imag = jnp.asarray([-1.0, 1.9, 2.2])
    mask = jnp.asarray([True, True, False])
    idx = lk._select_by_target(
        real,
        imag,
        mask,
        omega_scale=jnp.asarray(1.0),
        omega_target_factor=2.0,
        omega_sign=1,
        fallback_idx=jnp.asarray(0),
    )
    assert int(idx) == 1


def test_select_by_overlap_prefers_reference_branch() -> None:
    V = jnp.asarray([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]])
    eigvecs = jnp.eye(2, dtype=jnp.complex64)
    v_ref = jnp.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    mask = jnp.asarray([True, True])
    idx = lk._select_by_overlap(eigvecs, V, v_ref, mask, fallback_idx=jnp.asarray(1))
    assert int(idx) == 0
    idx_fallback = lk._select_by_overlap(
        eigvecs, V, v_ref, jnp.asarray([False, False]), fallback_idx=jnp.asarray(1)
    )
    assert int(idx_fallback) == 1


def test_build_shift_invert_preconditioner_modes() -> None:
    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    sigma = jnp.asarray(0.1j, dtype=v0.dtype)

    precond, op = lk._build_shift_invert_precond(v0, cache, params, term_cfg, sigma, None)
    assert precond is None and op is None
    precond, op = lk._build_shift_invert_precond(v0, cache, params, term_cfg, sigma, "unknown")
    assert precond is None and op is None

    precond, op = lk._build_shift_invert_precond(v0, cache, params, term_cfg, sigma, "damping")
    assert precond is not None and op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))

    precond, op = lk._build_shift_invert_precond(v0, cache, params, term_cfg, sigma, "hermite-line")
    assert op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))

    precond, op = lk._build_shift_invert_precond(v0, cache, params, term_cfg, sigma, "hermite-line-coarse")
    assert op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))


def test_build_shift_invert_preconditioner_linked_branch() -> None:
    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=True)
    sigma = jnp.asarray(0.2j, dtype=v0.dtype)
    precond, op = lk._build_shift_invert_precond(v0, cache, params, term_cfg, sigma, "hermite-line")
    assert op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))


@pytest.mark.parametrize("method", ["power", "propagator", "arnoldi"])
def test_dominant_eigenpair_methods_produce_finite_values(method: str) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    eig, vec = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method=method,
        krylov_dim=4,
        restarts=1,
        power_iters=4,
        power_dt=0.05,
    )
    assert vec.shape == v0.shape
    assert jnp.isfinite(jnp.real(eig))
    assert jnp.isfinite(jnp.imag(eig))


def test_dominant_eigenpair_shift_invert_sources_and_errors() -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    for source in ("propagator", "target", "power"):
        eig, vec = lk.dominant_eigenpair(
            v0,
            cache,
            params,
            terms=terms,
            method="shift_invert",
            shift=None,
            shift_source=source,
            shift_preconditioner="damping",
            krylov_dim=4,
            restarts=1,
            shift_maxiter=15,
            shift_restart=10,
            power_iters=4,
            power_dt=0.05,
        )
        assert vec.shape == v0.shape
        assert jnp.isfinite(jnp.real(eig))
    with pytest.raises(ValueError):
        lk.dominant_eigenpair(v0, cache, params, terms=terms, method="bad")


@pytest.mark.parametrize("shift_source", ["propagator", "power"])
def test_dominant_eigenpair_explicit_shift_uses_requested_seed_source(
    monkeypatch: pytest.MonkeyPatch,
    shift_source: str,
) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    captured: dict[str, jnp.ndarray] = {}
    seed = jnp.full_like(v0, 3.0 + 0.0j)

    def _fake_shift(v_init, v_ref, *_args, sigma, **_kwargs):
        captured["v_init"] = v_init
        captured["v_ref"] = v_ref
        captured["sigma"] = sigma
        return jnp.asarray(0.4 + 0.2j, dtype=v0.dtype), jnp.full_like(v0, 5.0 + 0.0j)

    monkeypatch.setattr(lk, "dominant_eigenpair_shift_invert_cached", _fake_shift)
    if shift_source == "propagator":
        monkeypatch.setattr(
            lk,
            "dominant_eigenpair_propagator_cached",
            lambda *args, **kwargs: (jnp.asarray(0.1 + 0.0j, dtype=v0.dtype), seed),
        )
    else:
        monkeypatch.setattr(
            lk,
            "dominant_eigenpair_power",
            lambda *args, **kwargs: (jnp.asarray(0.1 + 0.0j, dtype=v0.dtype), seed),
        )

    eig, vec = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        shift=0.2 - 1.1j,
        shift_source=shift_source,
        shift_selection="shift",
        krylov_dim=4,
        restarts=1,
        shift_maxiter=15,
        shift_restart=10,
        power_iters=4,
        power_dt=0.05,
    )

    assert jnp.allclose(captured["sigma"], jnp.asarray(0.2 - 1.1j, dtype=v0.dtype))
    assert jnp.allclose(captured["v_init"], seed)
    assert jnp.allclose(eig, jnp.asarray(0.4 + 0.2j, dtype=v0.dtype))
    assert jnp.allclose(vec, 5.0 + 0.0j)


def test_dominant_eigenpair_explicit_shift_defaults_to_reference_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    captured: dict[str, jnp.ndarray] = {}
    v_ref = jnp.full_like(v0, 7.0 + 0.0j)

    def _fake_shift(v_init, v_ref_in, *_args, sigma, **_kwargs):
        captured["v_init"] = v_init
        captured["v_ref"] = v_ref_in
        captured["sigma"] = sigma
        return jnp.asarray(0.4 + 0.2j, dtype=v0.dtype), jnp.full_like(v0, 5.0 + 0.0j)

    monkeypatch.setattr(lk, "dominant_eigenpair_shift_invert_cached", _fake_shift)

    lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        shift=0.2 - 1.1j,
        shift_source="target",
        shift_selection="shift",
        v_ref=v_ref,
        krylov_dim=4,
        restarts=1,
        shift_maxiter=15,
        shift_restart=10,
        power_iters=4,
        power_dt=0.05,
    )

    assert jnp.allclose(captured["sigma"], jnp.asarray(0.2 - 1.1j, dtype=v0.dtype))
    assert jnp.allclose(captured["v_init"], v_ref)
    assert jnp.allclose(captured["v_ref"], v_ref)


def test_shift_invert_fallback_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    _grid, cache, params, v0, term_cfg, terms = _tiny_krylov_setup(linked=False)

    def fake_shift(*args, **kwargs):
        return jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=v0.dtype), jnp.ones_like(v0)

    def fake_prop(*args, **kwargs):
        return jnp.asarray(1.0 + 0.2j, dtype=v0.dtype), jnp.full_like(v0, 2.0 + 0.0j)

    def fake_arnoldi(*args, **kwargs):
        return jnp.asarray(0.5 + 0.1j, dtype=v0.dtype), jnp.full_like(v0, 3.0 + 0.0j)

    def fake_power(*args, **kwargs):
        return jnp.asarray(0.2 + 0.05j, dtype=v0.dtype), jnp.full_like(v0, 4.0 + 0.0j)

    monkeypatch.setattr(lk, "dominant_eigenpair_shift_invert_cached", fake_shift)
    monkeypatch.setattr(lk, "dominant_eigenpair_propagator_cached", fake_prop)
    monkeypatch.setattr(lk, "dominant_eigenpair_cached", fake_arnoldi)
    monkeypatch.setattr(lk, "dominant_eigenpair_power", fake_power)
    monkeypatch.setattr(lk, "_omega_scale", lambda *_args, **_kwargs: jnp.asarray(1.0))

    eig_p, vec_p = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        fallback_method="propagator",
        fallback_real_floor=0.0,
    )
    assert jnp.allclose(eig_p, jnp.asarray(1.0 + 0.2j, dtype=v0.dtype))
    assert jnp.allclose(vec_p, 2.0 + 0.0j)

    eig_a, vec_a = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        fallback_method="arnoldi",
        fallback_real_floor=0.0,
    )
    assert jnp.allclose(eig_a, jnp.asarray(0.5 + 0.1j, dtype=v0.dtype))
    assert jnp.allclose(vec_a, 3.0 + 0.0j)

    eig_w, vec_w = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        fallback_method="power",
        fallback_real_floor=0.0,
    )
    assert jnp.allclose(eig_w, jnp.asarray(0.2 + 0.05j, dtype=v0.dtype))
    assert jnp.allclose(vec_w, 4.0 + 0.0j)

    # no fallback branch should return the (nan) shift-invert value directly.
    eig_n, _vec_n = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        fallback_method="none",
        fallback_real_floor=0.0,
    )
    assert not jnp.isfinite(jnp.real(eig_n))

    # dominant_eigenvalue wrapper should reuse dominant_eigenpair.
    monkeypatch.setattr(
        lk,
        "dominant_eigenpair",
        lambda *args, **kwargs: (jnp.asarray(0.7 + 0.1j), jnp.ones_like(v0)),
    )
    eig_val = lk.dominant_eigenvalue(v0, cache, params, terms=terms, krylov_dim=4, restarts=1)
    assert jnp.allclose(eig_val, jnp.asarray(0.7 + 0.1j))
