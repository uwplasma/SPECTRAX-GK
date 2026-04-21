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
    _collision_damping,
    _gx_nonlinear_omega_components,
    _gx_omega_mode_mask,
    _make_fixed_mode_projector,
    _make_hermitian_projector,
    _pack_resolved_diagnostics,
    build_nonlinear_imex_operator,
)
from spectraxgk.terms.config import FieldState, TermConfig


def test_pack_resolved_diagnostics_and_fixed_mode_projector() -> None:
    resolved = tuple(np.full((1,), i, dtype=float) for i in range(56))
    packed = _pack_resolved_diagnostics(resolved)
    np.testing.assert_allclose(packed.Phi2_kxt, [0.0])
    np.testing.assert_allclose(packed.Wapar_zst, [19.0])
    np.testing.assert_allclose(packed.TurbulentHeating_zst, [55.0])

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
