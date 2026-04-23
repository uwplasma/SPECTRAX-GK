from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.nonlinear import (
    _apply_collision_split,
    _collision_damping,
    _gx_nonlinear_omega_components,
    _gx_omega_mode_mask,
    _integrate_nonlinear_gx_diagnostics_impl,
    _make_fixed_mode_projector,
    _make_hermitian_projector,
    _pack_resolved_diagnostics,
    build_nonlinear_imex_operator,
    integrate_nonlinear,
    integrate_nonlinear_cached,
    integrate_nonlinear_gx_diagnostics,
    integrate_nonlinear_gx_diagnostics_state,
    integrate_nonlinear_imex_cached,
    integrate_nonlinear_imex_gx_diagnostics,
)
from spectraxgk.terms.config import FieldState, TermConfig


def test_pack_resolved_diagnostics_and_fixed_mode_projector() -> None:
    resolved = tuple(np.full((1,), i, dtype=float) for i in range(57))
    packed = _pack_resolved_diagnostics(resolved)
    np.testing.assert_allclose(packed.Phi2_kxt, [0.0])
    np.testing.assert_allclose(packed.Wapar_zst, [20.0])
    np.testing.assert_allclose(packed.TurbulentHeating_zst, [56.0])

    projector = _make_fixed_mode_projector(
        jnp.arange(24, dtype=jnp.float32).reshape(1, 3, 2, 4),
        ky_index=1,
        kx_index=0,
    )
    G = jnp.zeros((1, 3, 2, 4), dtype=jnp.float32)
    out = projector(G)
    np.testing.assert_allclose(np.asarray(out[..., 1:2, 0:1, :]), np.asarray(jnp.arange(24, dtype=jnp.float32).reshape(1, 3, 2, 4)[..., 1:2, 0:1, :]))
    assert _make_fixed_mode_projector(None, ky_index=0, kx_index=0) is None


def test_make_hermitian_projector_and_mode_mask() -> None:
    ky = np.array([0.0, 0.2, -0.2, -0.4], dtype=float)
    projector = _make_hermitian_projector(ky, nx=3)
    state = jnp.zeros((1, 4, 3, 2), dtype=jnp.complex64)
    state = state.at[..., 0:3, :, :].set(1.0 + 2.0j)
    out = projector(state)
    assert out.shape == state.shape
    np.testing.assert_allclose(np.asarray(out[..., 3, :, :]), np.asarray(jnp.conj(out[..., 1, [0, 2, 1], :])))

    no_project = _make_hermitian_projector(np.array([0.0, 0.2], dtype=float), nx=1)
    same = no_project(state[..., :2, :1, :])
    np.testing.assert_allclose(np.asarray(same), np.asarray(state[..., :2, :1, :]))

    grid = SimpleNamespace(ky=np.array([0.0, 0.2, -0.2, -0.4]), kx=np.array([0.0, 0.5]), dealias_mask=np.array([[True, False], [True, True], [True, True], [False, True]]))
    cache = SimpleNamespace(ky=jnp.asarray(grid.ky))
    mask = _gx_omega_mode_mask(grid, cache, gx_real_fft=True)
    assert mask.shape == (4, 2)
    assert bool(mask[0, 0]) is True
    assert bool(mask[3, 1]) is False


def test_collision_damping_and_imex_operator_builder(monkeypatch) -> None:
    cache = SimpleNamespace(
        lb_lam=jnp.ones((2, 2, 1, 1, 1), dtype=jnp.float32),
    )
    params = SimpleNamespace(nu=0.1)
    term_cfg = TermConfig(collisions=0.5, hypercollisions=2.0)
    monkeypatch.setattr("spectraxgk.nonlinear.hypercollision_damping", lambda cache, params, dtype: jnp.ones_like(cache.lb_lam, dtype=dtype) * 3.0)
    damp = _collision_damping(cache, params, term_cfg, jnp.float32, squeeze_species=False)
    np.testing.assert_allclose(np.asarray(damp), 0.5 * 0.1 + 2.0 * 3.0)

    cache6 = SimpleNamespace(lb_lam=jnp.ones((1, 2, 2, 1, 1, 1), dtype=jnp.float32))
    monkeypatch.setattr("spectraxgk.nonlinear.hypercollision_damping", lambda cache, params, dtype: jnp.ones_like(cache.lb_lam, dtype=dtype))
    squeezed = _collision_damping(cache6, SimpleNamespace(nu=0.4), TermConfig(collisions=1.0, hypercollisions=1.0), jnp.float32, squeeze_species=True)
    assert squeezed.shape == (2, 2, 1, 1, 1)

    cache_low_rank = SimpleNamespace(
        lb_lam=jnp.ones((2, 2), dtype=jnp.float32),
        b=jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.hypercollision_damping",
        lambda cache, params, dtype: jnp.ones((1, 2, 2, 1, 1, 1), dtype=dtype),
    )
    squeezed_low_rank = _collision_damping(
        cache_low_rank,
        SimpleNamespace(nu=jnp.asarray([0.4], dtype=jnp.float32)),
        TermConfig(collisions=1.0, hypercollisions=1.0),
        jnp.float32,
        squeeze_species=True,
    )
    assert squeezed_low_rank.shape == (2, 2, 1, 1, 1)
    np.testing.assert_allclose(np.asarray(squeezed_low_rank), 1.4)

    monkeypatch.setattr(
        "spectraxgk.nonlinear._build_implicit_operator",
        lambda *args, **kwargs: (
            jnp.zeros((1, 2, 2, 1, 1, 1), dtype=jnp.complex64),
            (1, 2, 2, 1, 1, 1),
            4,
            jnp.asarray(0.1, dtype=jnp.float32),
            None,
            lambda x: x,
            True,
        ),
    )
    op = build_nonlinear_imex_operator(
        jnp.zeros((2, 2, 1, 1, 1), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
    )
    assert op.shape == (1, 2, 2, 1, 1, 1)
    assert op.squeeze_species is True


def test_build_nonlinear_imex_operator_forwards_preconditioner(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_build(G0, cache, params, dt, terms, implicit_preconditioner):
        captured["preconditioner"] = implicit_preconditioner
        captured["terms"] = terms
        return (
            jnp.zeros((1, 2, 2, 1, 1, 1), dtype=jnp.complex64),
            (1, 2, 2, 1, 1, 1),
            4,
            jnp.asarray(0.2, dtype=jnp.float32),
            lambda x: x,
            lambda x: x,
            False,
        )

    monkeypatch.setattr("spectraxgk.nonlinear._build_implicit_operator", _fake_build)
    op = build_nonlinear_imex_operator(
        jnp.zeros((1, 2, 2, 1, 1, 1), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.2,
        terms=TermConfig(nonlinear=1.0),
        implicit_preconditioner="identity",
    )
    assert captured["preconditioner"] == "identity"
    assert op.shape == (1, 2, 2, 1, 1, 1)


def test_gx_nonlinear_omega_components_zero_and_finite() -> None:
    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    cache = build_linear_cache(grid, geom, LinearParams(), Nl=2, Nm=2)

    zeros = FieldState(
        phi=jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64),
        apar=None,
        bpar=None,
    )
    ox, oy = _gx_nonlinear_omega_components(
        zeros,
        grid,
        cache,
        gx_real_fft=False,
        kx_max=1.0,
        ky_max=1.0,
        kxfac=1.0,
        vpar_max=1.0,
        muB_max=1.0,
    )
    assert float(ox) == pytest.approx(0.0)
    assert float(oy) == pytest.approx(0.0)

    phi = jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64).at[1, 1, 0].set(1.0 + 0.0j)
    fields = FieldState(phi=phi, apar=0.5 * phi, bpar=0.25 * phi)
    ox, oy = _gx_nonlinear_omega_components(
        fields,
        grid,
        cache,
        gx_real_fft=False,
        kx_max=1.0,
        ky_max=1.0,
        kxfac=1.0,
        vpar_max=1.0,
        muB_max=1.0,
    )
    assert np.isfinite(float(ox))
    assert np.isfinite(float(oy))
    assert float(ox) >= 0.0
    assert float(oy) >= 0.0


def test_apply_collision_split_and_nonlinear_wrapper_routing(monkeypatch) -> None:
    G = jnp.ones((2, 2, 1, 1, 1), dtype=jnp.complex64)
    damping = jnp.ones_like(G.real)
    implicit = _apply_collision_split(G, damping, jnp.asarray(0.1, dtype=jnp.float32), "implicit")
    exp = _apply_collision_split(G, damping, jnp.asarray(0.1, dtype=jnp.float32), "exp")
    assert np.all(np.isfinite(np.asarray(implicit)))
    assert np.all(np.isfinite(np.asarray(exp)))
    with pytest.raises(ValueError):
        _apply_collision_split(G, damping, jnp.asarray(0.1, dtype=jnp.float32), "bad")

    monkeypatch.setattr(
        "spectraxgk.nonlinear.integrate_nonlinear_imex_cached",
        lambda *args, **kwargs: ("imex", "fields"),
    )
    assert integrate_nonlinear_cached(
        G,
        SimpleNamespace(ky=jnp.asarray([0.0, 0.2]), kx=jnp.asarray([0.0]), Jl=None, JlB=None, laguerre_to_grid=None, laguerre_to_spectral=None, laguerre_roots=None, laguerre_j0=None, laguerre_j1_over_alpha=None, b=None, dealias_mask=None, kxfac=1.0),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="semi-implicit",
    ) == ("imex", "fields")

    captured: dict[str, object] = {}

    def _fake_scan(rhs_fn, G0, dt, steps, **kwargs):
        captured["project_state"] = kwargs.get("project_state")
        return G0, FieldState(phi=jnp.zeros((4, 2, 2), dtype=jnp.complex64), apar=None, bpar=None)

    monkeypatch.setattr("spectraxgk.nonlinear.integrate_nonlinear_scan", _fake_scan)
    out_G, out_fields = integrate_nonlinear_cached(
        jnp.zeros((1, 4, 2, 2), dtype=jnp.complex64),
        SimpleNamespace(
            ky=jnp.asarray([0.0, 0.2, -0.2, -0.4]),
            kx=jnp.asarray([0.0, 0.5]),
            Jl=None,
            JlB=None,
            laguerre_to_grid=None,
            laguerre_to_spectral=None,
            laguerre_roots=None,
            laguerre_j0=None,
            laguerre_j1_over_alpha=None,
            b=None,
            dealias_mask=jnp.ones((4, 2), dtype=bool),
            kxfac=1.0,
        ),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="rk2",
        gx_real_fft=True,
    )
    assert out_G.shape == (1, 4, 2, 2)
    assert captured["project_state"] is not None
    assert out_fields.phi.shape == (4, 2, 2)


def test_integrate_nonlinear_builds_cache_and_rejects_bad_shape(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr("spectraxgk.nonlinear.ensure_flux_tube_geometry_data", lambda geom, z: "geom_eff")
    monkeypatch.setattr(
        "spectraxgk.nonlinear.build_linear_cache",
        lambda grid, geom, params, Nl, Nm: calls.append((Nl, Nm)) or "cache",
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.integrate_nonlinear_cached",
        lambda G0, cache, params, dt, steps, **kwargs: ("G_out", "fields_out"),
    )

    assert integrate_nonlinear(
        jnp.zeros((2, 3, 1, 1, 4), dtype=jnp.complex64),
        SimpleNamespace(z=np.array([-1.0, 0.0, 1.0, 2.0])),
        object(),
        object(),
        dt=0.1,
        steps=2,
    ) == ("G_out", "fields_out")
    assert calls == [(2, 3)]

    with pytest.raises(ValueError):
        integrate_nonlinear(jnp.zeros((2, 2), dtype=jnp.complex64), SimpleNamespace(z=np.array([0.0])), object(), object(), dt=0.1, steps=2)


def test_nonlinear_gx_diagnostics_route_and_state_reject_imex(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.nonlinear.integrate_nonlinear_imex_gx_diagnostics",
        lambda *args, **kwargs: ("t_imex", "diag_imex"),
    )
    assert integrate_nonlinear_gx_diagnostics(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="semi-implicit",
    ) == ("t_imex", "diag_imex")

    with pytest.raises(ValueError):
        integrate_nonlinear_gx_diagnostics_state(
            jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            dt=0.1,
            steps=2,
            method="imex",
        )


def test_integrate_nonlinear_gx_diagnostics_explicit_and_state_routes(monkeypatch) -> None:
    payload = ("t_explicit", "diag_explicit", "G_final", "fields_final")
    monkeypatch.setattr(
        "spectraxgk.nonlinear._integrate_nonlinear_gx_diagnostics_impl",
        lambda *args, **kwargs: payload,
    )

    out = integrate_nonlinear_gx_diagnostics(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="rk3",
    )
    assert out == ("t_explicit", "diag_explicit")

    out_state = integrate_nonlinear_gx_diagnostics_state(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="rk3",
    )
    assert out_state == payload


def test_integrate_nonlinear_gx_diagnostics_forwarding_contracts(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_impl(*args, **kwargs):
        captured.update(kwargs)
        return ("t", "diag", "G_final", "fields_final")

    monkeypatch.setattr("spectraxgk.nonlinear._integrate_nonlinear_gx_diagnostics_impl", _fake_impl)

    out = integrate_nonlinear_gx_diagnostics(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="rk4",
        fixed_dt=False,
        dt_min=1.0e-4,
        dt_max=0.2,
        cfl=0.7,
        cfl_fac=0.5,
        collision_split=True,
        collision_scheme="exp",
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
    )

    assert out == ("t", "diag")
    assert captured["fixed_dt"] is False
    assert captured["collision_split"] is True
    assert captured["collision_scheme"] == "exp"
    assert captured["fixed_mode_ky_index"] == 1
    assert captured["fixed_mode_kx_index"] == 0

    captured.clear()
    out_state = integrate_nonlinear_gx_diagnostics_state(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="rk4",
        fixed_dt=False,
        fixed_mode_ky_index=0,
        fixed_mode_kx_index=1,
    )
    assert out_state == ("t", "diag", "G_final", "fields_final")
    assert captured["fixed_dt"] is False
    assert captured["fixed_mode_ky_index"] == 0
    assert captured["fixed_mode_kx_index"] == 1


def test_integrate_nonlinear_gx_diagnostics_imex_forwarding_contracts(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_imex(*args, **kwargs):
        captured.update(kwargs)
        return ("t_imex", "diag_imex")

    monkeypatch.setattr("spectraxgk.nonlinear.integrate_nonlinear_imex_gx_diagnostics", _fake_imex)

    out = integrate_nonlinear_gx_diagnostics(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="semi-implicit",
        collision_split=True,
        collision_scheme="exp",
        implicit_preconditioner="identity",
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
        show_progress=True,
    )

    assert out == ("t_imex", "diag_imex")
    assert captured["collision_split"] is True
    assert captured["collision_scheme"] == "exp"
    assert captured["implicit_preconditioner"] == "identity"
    assert captured["fixed_mode_ky_index"] == 1
    assert captured["fixed_mode_kx_index"] == 0
    assert captured["show_progress"] is True


def test_explicit_gx_diagnostics_impl_applies_fixed_mode_collision_and_stride(monkeypatch) -> None:
    grid = SimpleNamespace(
        ky=np.array([0.0, 0.2], dtype=float),
        kx=np.array([0.0], dtype=float),
        z=np.array([0.0, 1.0], dtype=float),
        dealias_mask=np.ones((2, 1), dtype=bool),
    )
    cache = SimpleNamespace(
        ky=jnp.asarray(grid.ky),
        kx=jnp.asarray(grid.kx),
        kxfac=1.0,
        l=jnp.asarray([0], dtype=jnp.int32),
        m=jnp.asarray([[0]], dtype=jnp.int32),
        lb_lam=jnp.ones((1, 1, 1, 2, 1, 2), dtype=jnp.float32),
    )
    params = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]), nu=0.2)
    phi = jnp.ones((2, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=phi, apar=None, bpar=None)

    def _resolved_tuple():
        return (
            jnp.asarray(1.0),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )

    def _split_flux_tuple():
        return (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        )

    monkeypatch.setattr("spectraxgk.nonlinear.ensure_flux_tube_geometry_data", lambda geom, z: geom)
    monkeypatch.setattr(
        "spectraxgk.nonlinear.gx_volume_factors",
        lambda geom, grid: (jnp.ones((grid.z.size,), dtype=jnp.float32), jnp.asarray(1.0)),
    )
    monkeypatch.setattr("spectraxgk.nonlinear._gx_omega_mode_mask", lambda grid, cache, **kwargs: jnp.ones((2, 1), dtype=bool))
    monkeypatch.setattr("spectraxgk.nonlinear._gx_linear_omega_max", lambda *args, **kwargs: np.array([0.0, 0.0, 0.0], dtype=float))
    monkeypatch.setattr("spectraxgk.nonlinear._gx_laguerre_vmax", lambda nl: 0.0)
    monkeypatch.setattr(
        "spectraxgk.nonlinear.nonlinear_rhs_cached",
        lambda G, cache, params, terms, **kwargs: (jnp.ones_like(G), fields),
    )
    monkeypatch.setattr("spectraxgk.nonlinear.compute_fields_cached", lambda *args, **kwargs: fields)

    def _fake_growth(phi, phi_prev, dt_step, z_index, mask):
        return (
            jnp.ones((2, 1), dtype=jnp.float32) * 2.0,
            jnp.ones((2, 1), dtype=jnp.float32) * -3.0,
        )

    monkeypatch.setattr("spectraxgk.nonlinear._gx_growth_rate_step", _fake_growth)
    monkeypatch.setattr("spectraxgk.nonlinear.gx_phi2_resolved", lambda *args, **kwargs: _resolved_tuple())
    monkeypatch.setattr(
        "spectraxgk.nonlinear.gx_Wg_resolved",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.gx_Wphi_resolved",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.gx_Wapar_resolved",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.gx_heat_flux_resolved_species",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.gx_heat_flux_split_resolved_species",
        lambda *args, **kwargs: (_split_flux_tuple(), _split_flux_tuple(), _split_flux_tuple()),
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.gx_particle_flux_resolved_species",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.gx_particle_flux_split_resolved_species",
        lambda *args, **kwargs: (_split_flux_tuple(), _split_flux_tuple(), _split_flux_tuple()),
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.gx_turbulent_heating_resolved_species",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr("spectraxgk.nonlinear._collision_damping", lambda *args, **kwargs: jnp.ones((1, 1, 2, 1, 2), dtype=jnp.float32))

    def _fake_collision_split(G_state, damping, dt_local, scheme):
        assert scheme == "exp"
        return G_state + 5.0

    monkeypatch.setattr("spectraxgk.nonlinear._apply_collision_split", _fake_collision_split)

    G0 = jnp.zeros((1, 1, 2, 1, 2), dtype=jnp.complex64)
    G0 = G0.at[..., 1:2, 0:1, :].set(7.0 + 0.0j)

    t, diag, G_final, fields_final = _integrate_nonlinear_gx_diagnostics_impl(
        G0,
        grid,
        SimpleNamespace(),
        params,
        dt=0.1,
        steps=3,
        method="euler",
        cache=cache,
        terms=TermConfig(collisions=1.0, hypercollisions=0.0),
        sample_stride=1,
        diagnostics_stride=2,
        collision_split=True,
        collision_scheme="exp",
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
    )

    np.testing.assert_allclose(np.asarray(t), [0.1, 0.3])
    np.testing.assert_allclose(np.asarray(diag.gamma_t), [2.0, 2.0])
    np.testing.assert_allclose(np.asarray(G_final[..., 1:2, 0:1, :]), 7.0)
    assert np.all(np.asarray(G_final[..., 0:1, 0:1, :]) > 0.0)
    assert fields_final.phi.shape == (2, 1, 2)


def test_integrate_nonlinear_imex_gx_diagnostics_rejects_bad_shape(monkeypatch) -> None:
    monkeypatch.setattr("spectraxgk.nonlinear.ensure_flux_tube_geometry_data", lambda geom, z: geom)
    with pytest.raises(ValueError):
        integrate_nonlinear_imex_gx_diagnostics(
            jnp.zeros((2, 2), dtype=jnp.complex64),
            SimpleNamespace(z=np.array([0.0])),
            object(),
            object(),
            dt=0.1,
            steps=2,
        )


def test_integrate_nonlinear_imex_cached_shape_mismatch_and_zero_nonlinear(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    implicit_operator = SimpleNamespace(
        shape=(1, 2, 2, 1, 1, 2),
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        precond_op=lambda x: x,
        matvec=lambda x: x,
        squeeze_species=False,
        state_dtype=jnp.complex64,
    )
    with pytest.raises(ValueError):
        integrate_nonlinear_imex_cached(
            G0,
            SimpleNamespace(),
            SimpleNamespace(),
            dt=0.1,
            steps=2,
            implicit_operator=implicit_operator,
        )

    cache = SimpleNamespace(
        Jl=None,
        JlB=None,
        sqrt_m=None,
        sqrt_m_p1=None,
        kx_grid=None,
        ky_grid=None,
        dealias_mask=None,
        kxfac=1.0,
        laguerre_to_grid=None,
        laguerre_to_spectral=None,
        laguerre_roots=None,
        laguerre_j0=None,
        laguerre_j1_over_alpha=None,
        b=None,
    )
    good_operator = SimpleNamespace(
        shape=G0.shape,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        precond_op=lambda x: x,
        matvec=lambda x: x,
        squeeze_species=False,
        state_dtype=jnp.complex64,
    )

    gmres_calls: list[int] = []

    monkeypatch.setattr(
        "spectraxgk.nonlinear.jax.scipy.sparse.linalg.gmres",
        lambda matvec, rhs, **kwargs: (gmres_calls.append(rhs.size) or rhs, SimpleNamespace(success=True)),
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.assemble_rhs_cached_jit",
        lambda G, cache, params, terms, **kwargs: (
            jnp.zeros_like(G),
            FieldState(phi=jnp.zeros((1, 1, 2), dtype=jnp.complex64), apar=None, bpar=None),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.nonlinear.compute_fields_cached",
        lambda G, cache, params, terms=None: (_ for _ in ()).throw(AssertionError("nonlinear path should stay off")),
    )

    G_out, fields_t = integrate_nonlinear_imex_cached(
        G0,
        cache,
        SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
        dt=0.1,
        steps=2,
        terms=TermConfig(nonlinear=0.0),
        implicit_operator=good_operator,
    )

    assert gmres_calls
    assert G_out.shape == G0.shape
    assert fields_t.phi.shape[0] == 2
