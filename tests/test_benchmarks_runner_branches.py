from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    KBMBaseCase,
    KineticElectronBaseCase,
    TEMBaseCase,
)
from spectraxgk.linear import LinearTerms
from spectraxgk.benchmarks import (
    KrylovConfig,
    compare_cyclone_to_reference,
    run_cyclone_linear,
    run_cyclone_scan,
    run_etg_linear,
    run_etg_scan,
    run_kinetic_scan,
    run_kinetic_linear,
    run_kbm_beta_scan,
    run_kbm_linear,
    run_kbm_scan,
    run_tem_scan,
    run_tem_linear,
)


def _grid_full():
    return SimpleNamespace(
        ky=np.array([0.0, 0.3], dtype=float),
        kx=np.array([0.0], dtype=float),
        z=np.array([-1.0, 0.0, 1.0], dtype=float),
        dealias_mask=np.array([[True], [True]], dtype=bool),
    )


def _grid_sel():
    return SimpleNamespace(
        ky=np.array([0.3], dtype=float),
        kx=np.array([0.0], dtype=float),
        z=np.array([-1.0, 0.0, 1.0], dtype=float),
        dealias_mask=np.array([[True]], dtype=bool),
    )


def _select_grid_dynamic(grid, idx):
    ky_idx = np.atleast_1d(np.asarray(idx, dtype=int))
    return SimpleNamespace(
        ky=np.asarray(grid.ky)[ky_idx],
        kx=np.asarray(grid.kx),
        z=np.asarray(grid.z),
        dealias_mask=np.ones((ky_idx.size, np.asarray(grid.kx).size), dtype=bool),
    )


def _fake_initial_condition(grid, *args, **kwargs):
    nl = int(kwargs.get("Nl", 2))
    nm = int(kwargs.get("Nm", 2))
    return np.zeros(
        (
            nl,
            nm,
            np.asarray(grid.ky).size,
            np.asarray(grid.kx).size,
            np.asarray(grid.z).size,
        ),
        dtype=np.complex64,
    )


def test_run_cyclone_linear_auto_falls_back_to_krylov(monkeypatch) -> None:
    status: list[str] = []
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear",
        lambda *args, **kwargs: (
            np.arange(3),
            np.ones((3, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._select_fit_signal_auto",
        lambda *args, **kwargs: (np.ones(3, dtype=np.complex64), "phi", -0.1, 0.2),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_explicit",
        lambda *args, **kwargs: (
            np.array([0.0, 1.0]),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.instantaneous_growth_rate_from_phi",
        lambda *args, **kwargs: (0.3, -0.2, None, None, 0.5),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.dominant_eigenpair",
        lambda *args, **kwargs: (
            0.4 + 0.1j,
            np.ones((2, 2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.compute_fields_cached",
        lambda *args, **kwargs: SimpleNamespace(
            phi=np.ones((1, 1, 3), dtype=np.complex64)
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.linear_terms_to_term_config",
        lambda terms: object(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_cyclone_linear(
        cfg=CycloneBaseCase(),
        solver="auto",
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=3,
        fit_signal="auto",
        status_callback=status.append,
    )

    assert result.gamma == 0.3
    assert result.omega == -0.2
    assert any("building spectral grid" in msg for msg in status)


def test_run_cyclone_linear_krylov_uses_gx_seed_and_branch_guard(monkeypatch) -> None:
    status: list[str] = []
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_explicit",
        lambda *args, **kwargs: (
            np.array([0.0, 1.0, 2.0]),
            np.ones((3, 1, 1, 3), dtype=np.complex64),
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.instantaneous_growth_rate_from_phi",
        lambda *args, **kwargs: (0.45, -0.25, None, None, 1.0),
    )

    def _fake_eigenpair(*args, **kwargs):
        assert kwargs["shift"] == pytest.approx(complex(0.45, 0.25))
        return 0.01 + 1.2j, np.ones((2, 2, 1, 1, 3), dtype=np.complex64)

    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.dominant_eigenpair", _fake_eigenpair
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.compute_fields_cached",
        lambda *args, **kwargs: SimpleNamespace(
            phi=np.ones((1, 1, 3), dtype=np.complex64)
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.linear_terms_to_term_config",
        lambda terms: object(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_cyclone_linear(
        cfg=CycloneBaseCase(),
        solver="krylov",
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=3,
        status_callback=status.append,
    )

    assert result.gamma == pytest.approx(0.45)
    assert result.omega == pytest.approx(0.25)
    assert result.t.tolist() == [0.0]
    assert any("estimating frequency seed" in msg for msg in status)
    assert any("Krylov solve complete" in msg for msg in status)


def test_run_cyclone_linear_krylov_uses_reduced_seed_after_primary_failure(
    monkeypatch,
) -> None:
    status: list[str] = []
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_explicit",
        lambda *args, **kwargs: (
            np.array([0.0, 1.0, 2.0]),
            np.ones((3, 1, 1, 3), dtype=np.complex64),
            None,
            None,
        ),
    )
    seed_values = iter(
        [ValueError("primary seed failed"), (0.33, -0.18, None, None, 1.0)]
    )

    def _fake_growth(*args, **kwargs):
        value = next(seed_values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.instantaneous_growth_rate_from_phi", _fake_growth
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.dominant_eigenpair",
        lambda *args, **kwargs: (
            0.02 + 0.9j,
            np.ones((2, 2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.compute_fields_cached",
        lambda *args, **kwargs: SimpleNamespace(
            phi=np.ones((1, 1, 3), dtype=np.complex64)
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.linear_terms_to_term_config",
        lambda terms: object(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_cyclone_linear(
        cfg=CycloneBaseCase(),
        solver="krylov",
        Nl=4,
        Nm=4,
        dt=0.1,
        steps=3,
        status_callback=status.append,
    )

    assert result.gamma == pytest.approx(0.33)
    assert result.omega == pytest.approx(0.18)
    assert any("retrying reduced Hermite-Laguerre seed" in msg for msg in status)


def test_run_cyclone_linear_reference_aligned_time_path_uses_reference_contract(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def _fake_reference_integrator(*args, **kwargs):
        captured["time_cfg"] = args[5]
        captured["mode_method"] = kwargs["mode_method"]
        return (
            np.array([0.0, 0.1, 0.2]),
            np.ones((3, 1, 1, 3), dtype=np.complex64),
            None,
            None,
        )

    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_explicit",
        _fake_reference_integrator,
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.instantaneous_growth_rate_from_phi",
        lambda *args, **kwargs: (0.31, -0.17, None, None, 0.1),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_cyclone_linear(
        cfg=CycloneBaseCase(),
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        solver="time",
        reference_aligned=True,
        mode_method="project",
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
    )

    assert captured["mode_method"] == "z_index"
    assert captured["time_cfg"].fixed_dt is True
    assert captured["time_cfg"].sample_stride == 1
    assert result.gamma == pytest.approx(0.31)
    assert result.omega == pytest.approx(-0.17)


def test_run_etg_linear_streaming_density_path(monkeypatch) -> None:
    cfg0 = ETGBaseCase()
    cfg = replace(
        cfg0,
        time=replace(
            cfg0.time,
            use_diffrax=True,
            dt=0.1,
            t_max=1.0,
            sample_stride=1,
            diffrax_max_steps=8,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._resolve_streaming_window",
        lambda *args, **kwargs: (0.2, 0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.integrate_linear_diffrax_streaming",
        lambda *args, **kwargs: (
            np.ones((1, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.array([0.25]),
            np.array([-0.15]),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.compute_fields_cached",
        lambda *args, **kwargs: SimpleNamespace(
            phi=np.ones((1, 1, 3), dtype=np.complex64)
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.linear_terms_to_term_config", lambda terms: object()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_etg_linear(
        cfg=cfg,
        solver="time",
        params=SimpleNamespace(charge_sign=np.array([-1.0]), rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=5,
        fit_signal="density",
        streaming_fit=True,
    )

    np.testing.assert_allclose(result.t, [0.8])
    assert result.gamma == 0.25
    assert result.omega == -0.15


@pytest.mark.parametrize("use_diffrax", [True, False])
def test_run_etg_linear_honors_explicit_time_config_density_paths(
    monkeypatch, use_diffrax: bool
) -> None:
    cfg0 = ETGBaseCase()
    cfg = replace(
        cfg0,
        time=replace(
            cfg0.time,
            use_diffrax=use_diffrax,
            dt=0.2,
            t_max=0.8,
            sample_stride=2,
            diffrax_max_steps=8,
        ),
    )
    called: dict[str, object] = {}
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    phi_t = np.ones((3, 1, 1, 3), dtype=np.complex64)
    density_t = 2.0 * phi_t

    def _fake_diffrax(*args, **kwargs):
        called["path"] = "diffrax"
        assert kwargs["dt"] == pytest.approx(0.2)
        assert kwargs["steps"] == 4
        assert kwargs["sample_stride"] == 2
        return np.zeros((3, 1, 2, 2, 1, 1, 3), dtype=np.complex64), (phi_t, density_t)

    def _fake_diagnostics(*args, **kwargs):
        called["path"] = "diagnostics"
        assert kwargs["dt"] == pytest.approx(0.2)
        assert kwargs["steps"] == 4
        assert kwargs["sample_stride"] == 2
        return (
            np.zeros((3, 1, 2, 2, 1, 1, 3), dtype=np.complex64),
            phi_t,
            density_t,
        )

    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.integrate_linear_diffrax", _fake_diffrax
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.integrate_linear_diagnostics", _fake_diagnostics
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._select_fit_signal",
        lambda *args, **kwargs: np.array(
            [1.0 + 0.0j, 2.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex64
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.29, -0.14, 0.0, 0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_etg_linear(
        cfg=cfg,
        time_cfg=cfg.time,
        solver="time",
        params=SimpleNamespace(charge_sign=np.array([-1.0]), rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        fit_signal="density",
    )

    assert called["path"] == ("diffrax" if use_diffrax else "diagnostics")
    np.testing.assert_allclose(result.t, [0.0, 0.4, 0.8])
    assert result.gamma == 0.29
    assert result.omega == -0.14


def test_run_etg_linear_explicit_time_config_phi_path_uses_config_integrator(
    monkeypatch,
) -> None:
    cfg0 = ETGBaseCase()
    cfg = replace(
        cfg0,
        time=replace(
            cfg0.time, use_diffrax=False, dt=0.15, t_max=0.45, sample_stride=1
        ),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def _fake_from_config(*args, **kwargs):
        captured["time_cfg"] = args[4]
        return (
            np.zeros((3, 1, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.ones((3, 1, 1, 3), dtype=np.complex64),
        )

    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.integrate_linear_from_config", _fake_from_config
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._select_fit_signal",
        lambda *args, **kwargs: np.array(
            [1.0 + 0.0j, 1.5 + 0.0j, 2.25 + 0.0j], dtype=np.complex64
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.33, -0.12, 0.0, 0.45),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_etg_linear(
        cfg=cfg,
        time_cfg=cfg.time,
        solver="time",
        params=SimpleNamespace(charge_sign=np.array([-1.0]), rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        fit_signal="phi",
    )

    assert captured["time_cfg"].dt == pytest.approx(0.15)
    np.testing.assert_allclose(result.t, [0.0, 0.15, 0.3])
    assert result.gamma == 0.33
    assert result.omega == -0.12


def test_run_kbm_linear_explicit_time_uses_omega_series_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_flux_tube_geometry",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.apply_geometry_grid_defaults", lambda geom, grid: grid
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.integrate_linear_explicit_diagnostics",
        lambda *args, **kwargs: (
            np.array([0.0, 1.0]),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            np.array([[[0.1, 0.2]]]),
            np.array([[[0.0, -0.3]]]),
            None,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.instantaneous_growth_rate_from_phi",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no fit")),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.windowed_growth_rate_from_omega_series",
        lambda *args, **kwargs: (0.35, -0.22, None, None),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_kbm_linear(
        cfg=KBMBaseCase(),
        solver="explicit_time",
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        mode_method="z_index",
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
        ky_target=0.3,
    )

    assert result.gamma == 0.35
    assert result.omega == -0.22


def test_run_kbm_beta_scan_multi_target_resolves_near_marginal_branch(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._two_species_params",
        lambda *args, **kwargs: SimpleNamespace(rho_star=1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    target_calls: list[float] = []
    eigs = iter([0.0 + 0.2j, 0.12 + 1.1j])

    def _fake_eigenpair(G0, *args, **kwargs):
        target_calls.append(float(kwargs["omega_target_factor"]))
        return next(eigs), np.zeros_like(np.asarray(G0))

    monkeypatch.setattr("spectraxgk.validation.benchmarks.kbm.dominant_eigenpair", _fake_eigenpair)

    scan = run_kbm_beta_scan(
        np.array([0.015]),
        cfg=KBMBaseCase(),
        solver="krylov",
        krylov_cfg=KrylovConfig(
            method="shift_invert",
            mode_family="kbm",
            shift_selection="targeted",
            omega_sign=-1,
        ),
        kbm_target_factors=(0.7, 1.5),
        kbm_beta_transition=float("nan"),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=1,
    )

    assert target_calls == [0.7, 1.5]
    np.testing.assert_allclose(scan.ky, [0.015])
    np.testing.assert_allclose(scan.gamma, [0.12])
    np.testing.assert_allclose(scan.omega, [-1.1])


def test_compare_cyclone_to_reference_handles_zero_reference() -> None:
    result = SimpleNamespace(gamma=0.2, omega=-0.1, ky=0.3)
    reference = SimpleNamespace(
        ky=np.array([0.3]),
        gamma=np.array([0.0]),
        omega=np.array([0.0]),
    )
    comparison = compare_cyclone_to_reference(result, reference)
    assert np.isnan(comparison.rel_gamma)
    assert np.isnan(comparison.rel_omega)


def test_run_cyclone_scan_krylov_mode_follow(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_explicit",
        lambda *args, **kwargs: (
            np.array([0.0, 1.0]),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.instantaneous_growth_rate_from_phi",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no seed")),
    )
    vals = iter(
        [
            (0.3 + 0.1j, np.ones((2, 2, 1, 1, 3), dtype=np.complex64)),
            (0.4 + 0.2j, np.ones((2, 2, 1, 1, 3), dtype=np.complex64)),
        ]
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.dominant_eigenpair",
        lambda *args, **kwargs: next(vals),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_cyclone_scan(
        np.array([0.2, 0.3]),
        cfg=CycloneBaseCase(),
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        solver="krylov",
        Nl=2,
        Nm=2,
        mode_follow=True,
    )
    np.testing.assert_allclose(scan.gamma, [0.3, 0.4])
    np.testing.assert_allclose(scan.omega, [-0.1, -0.2])


def test_run_cyclone_scan_plain_krylov_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    vals = iter(
        [
            (0.21 + 0.09j, np.zeros((2, 2, 1, 1, 3), dtype=np.complex64)),
            (0.31 - 0.11j, np.zeros((2, 2, 1, 1, 3), dtype=np.complex64)),
        ]
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.dominant_eigenpair",
        lambda *args, **kwargs: next(vals),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_cyclone_scan(
        np.array([0.2, 0.3]),
        cfg=CycloneBaseCase(),
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        solver="krylov",
        mode_follow=False,
        Nl=2,
        Nm=2,
        ky_batch=1,
    )

    np.testing.assert_allclose(scan.gamma, [0.21, 0.31])
    np.testing.assert_allclose(scan.omega, [-0.09, 0.11])


def test_run_cyclone_scan_diffrax_streaming_time_config_batch(monkeypatch) -> None:
    cfg0 = CycloneBaseCase()
    cfg = replace(
        cfg0,
        time=replace(
            cfg0.time,
            use_diffrax=True,
            dt=0.1,
            t_max=0.2,
            sample_stride=1,
            diffrax_max_steps=8,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", _select_grid_dynamic
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._resolve_streaming_window",
        lambda *args, **kwargs: (0.0, 0.2),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_diffrax_streaming",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 1, 3), dtype=np.complex64),
            np.array([0.12, 0.18]),
            np.array([-0.04, -0.06]),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_cyclone_scan(
        np.array([0.2, 0.3]),
        cfg=cfg,
        time_cfg=cfg.time,
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        solver="time",
        fit_signal="phi",
        streaming_fit=True,
        Nl=2,
        Nm=2,
        ky_batch=2,
        mode_follow=False,
    )

    np.testing.assert_allclose(scan.gamma, [0.12, 0.18])
    np.testing.assert_allclose(scan.omega, [-0.04, -0.06])


def test_run_cyclone_scan_time_config_auto_signal_fallback(monkeypatch) -> None:
    cfg0 = CycloneBaseCase()
    cfg = replace(
        cfg0,
        time=replace(cfg0.time, use_diffrax=False, dt=0.1, t_max=0.2, sample_stride=1),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_from_config",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
            (
                np.ones((2, 1, 1, 3), dtype=np.complex64),
                2.0 * np.ones((2, 1, 1, 3), dtype=np.complex64),
            ),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._select_fit_signal_auto",
        lambda *args, **kwargs: (np.ones(2, dtype=np.complex64), "phi", -0.2, -0.05),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.run_cyclone_linear",
        lambda *args, **kwargs: SimpleNamespace(gamma=0.44, omega=-0.22),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_cyclone_scan(
        np.array([0.3]),
        cfg=cfg,
        time_cfg=cfg.time,
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        solver="auto",
        fit_signal="auto",
        reference_aligned=False,
        require_positive=True,
        Nl=2,
        Nm=2,
        ky_batch=1,
        mode_follow=False,
    )

    np.testing.assert_allclose(scan.gamma, [0.44])
    np.testing.assert_allclose(scan.omega, [-0.22])


def test_run_kinetic_linear_time_density_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.integrate_linear_diagnostics",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            2.0 * np.ones((2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear._select_fit_signal",
        lambda *args, **kwargs: np.array([1.0 + 0.0j, 2.0 + 0.0j]),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.12, -0.05, 0.0, 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_kinetic_linear(
        solver="time",
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
        fit_signal="density",
        auto_window=True,
    )
    assert result.gamma == 0.12
    assert result.omega == -0.05


def test_run_kinetic_linear_krylov_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.dominant_eigenpair",
        lambda *args, **kwargs: (
            0.31 + 0.27j,
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.compute_fields_cached",
        lambda *args, **kwargs: SimpleNamespace(
            phi=np.ones((1, 1, 3), dtype=np.complex64)
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.linear_terms_to_term_config",
        lambda terms: object(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_kinetic_linear(
        solver="krylov",
        params=SimpleNamespace(charge_sign=np.array([1.0, -1.0]), rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
    )

    assert result.gamma == 0.31
    assert result.omega == -0.27
    assert result.t.tolist() == [0.0]


@pytest.mark.parametrize(
    ("use_diffrax", "fit_signal"),
    [
        (True, "density"),
        (False, "density"),
        (False, "phi"),
    ],
)
def test_run_kinetic_linear_time_config_branches(
    monkeypatch, use_diffrax: bool, fit_signal: str
) -> None:
    cfg0 = KineticElectronBaseCase()
    cfg = replace(
        cfg0,
        time=replace(
            cfg0.time,
            use_diffrax=use_diffrax,
            dt=0.1,
            t_max=0.2,
            sample_stride=1,
            diffrax_max_steps=8,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.integrate_linear_from_config",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.integrate_linear_diagnostics",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            2.0 * np.ones((2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear._select_fit_signal",
        lambda *args, **kwargs: np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.16, -0.08, 0.0, 0.2),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_kinetic_linear(
        cfg=cfg,
        time_cfg=cfg.time,
        solver="time",
        fit_signal=fit_signal,
        params=SimpleNamespace(charge_sign=np.array([1.0, -1.0]), rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
    )

    assert result.gamma == 0.16
    assert result.omega == -0.08


def test_run_tem_linear_rejects_invalid_fit_signal_and_time_density(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    with np.testing.assert_raises(ValueError):
        run_tem_linear(
            solver="time",
            fit_signal="auto",
            params=SimpleNamespace(rho_star=1.0),
            terms=LinearTerms(),
            Nl=2,
            Nm=2,
            dt=0.1,
            steps=2,
        )

    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.integrate_linear_diagnostics",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            3.0 * np.ones((2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.extract_mode_time_series",
        lambda arr, sel, method: np.array([1.0 + 0.0j, 3.0 + 0.0j]),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.2, -0.1, 0.0, 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )
    result = run_tem_linear(
        solver="time",
        fit_signal="density",
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
    )
    assert result.gamma == 0.2
    assert result.omega == -0.1


def test_run_cyclone_scan_auto_explicit_time_falls_back_to_krylov(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._apply_reference_hypercollisions",
        lambda params, **kwargs: params,
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_explicit",
        lambda *args, **kwargs: (
            np.array([0.0, 1.0]),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.instantaneous_growth_rate_from_phi",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no fit")),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.dominant_eigenpair",
        lambda *args, **kwargs: (
            0.25 + 0.4j,
            np.ones((2, 2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_cyclone_scan(
        np.array([0.3]),
        cfg=CycloneBaseCase(),
        solver="auto",
        reference_aligned=True,
        Nl=2,
        Nm=2,
    )

    np.testing.assert_allclose(scan.gamma, [0.25])
    np.testing.assert_allclose(scan.omega, [0.4])


def test_run_cyclone_scan_explicit_time_reselects_branch_with_previous_frequency(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_explicit",
        lambda *args, **kwargs: (
            np.array([0.0, 1.0]),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            None,
            None,
        ),
    )
    growth_values = iter(
        [
            (0.10, -0.20, None, None, 1.0),
            (0.12, 0.30, None, None, 1.0),
            (0.11, 0.10, None, None, 1.0),
        ]
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.instantaneous_growth_rate_from_phi",
        lambda *args, **kwargs: next(growth_values),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.dominant_eigenpair",
        lambda *args, **kwargs: (
            0.13 - 0.38j,
            np.ones((2, 2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_cyclone_scan(
        np.array([0.2, 0.3, 0.4]),
        cfg=CycloneBaseCase(),
        solver="explicit_time",
        reference_aligned=True,
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=np.array([0.1, 0.1, 0.1]),
        steps=np.array([2, 2, 2]),
    )

    np.testing.assert_allclose(scan.omega, [0.2, 0.3, 0.38])
    np.testing.assert_allclose(scan.gamma, [0.10, 0.12, 0.13])


def test_run_cyclone_scan_empty_explicit_time_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )

    scan = run_cyclone_scan(
        np.asarray([]),
        cfg=CycloneBaseCase(),
        solver="explicit_time",
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
    )

    assert scan.ky.size == 0
    assert scan.gamma.size == 0
    assert scan.omega.size == 0


def test_run_kbm_scan_forwards_per_mode_arrays(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_kbm_beta_scan(**kwargs):
        calls.append(kwargs)
        ky = float(kwargs["ky_target"])
        return SimpleNamespace(gamma=np.array([ky + 1.0]), omega=np.array([-ky - 2.0]))

    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.run_kbm_beta_scan", _fake_run_kbm_beta_scan
    )

    scan = run_kbm_scan(
        np.array([0.2, 0.4]),
        beta_value=1.0e-4,
        dt=np.array([0.1, 0.2]),
        steps=np.array([3, 4]),
        tmin=np.array([0.0, 1.0]),
        tmax=np.array([2.0, 3.0]),
    )

    assert len(calls) == 2
    assert calls[0]["dt"] == 0.1
    assert calls[1]["dt"] == 0.2
    assert calls[0]["steps"] == 3
    assert calls[1]["steps"] == 4
    assert calls[0]["tmin"] == 0.0
    assert calls[1]["tmin"] == 1.0
    assert calls[0]["tmax"] == 2.0
    assert calls[1]["tmax"] == 3.0
    np.testing.assert_allclose(scan.gamma, [1.2, 1.4])
    np.testing.assert_allclose(scan.omega, [-2.2, -2.4])


def test_run_kbm_scan_uses_cfg_beta_and_sequence_pick(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_run_kbm_beta_scan(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(gamma=np.array([3.0]), omega=np.array([-4.0]))

    cfg = replace(KBMBaseCase(), model=replace(KBMBaseCase().model, beta=2.5e-3))
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.run_kbm_beta_scan", _fake_run_kbm_beta_scan
    )

    scan = run_kbm_scan(
        np.array([0.15, 0.35]),
        cfg=cfg,
        dt=[0.05, 0.1],
        steps=(5, 6),
        tmin=[0.2, 0.4],
        tmax=(0.8, 1.2),
    )

    assert len(calls) == 2
    assert calls[0]["betas"][0] == pytest.approx(2.5e-3)
    assert calls[1]["dt"] == 0.1
    assert calls[0]["steps"] == 5
    assert calls[1]["tmin"] == 0.4
    assert calls[0]["tmax"] == 0.8
    np.testing.assert_allclose(scan.gamma, [3.0, 3.0])
    np.testing.assert_allclose(scan.omega, [-4.0, -4.0])


def test_run_kbm_beta_scan_rejects_invalid_species_indices() -> None:
    with pytest.raises(ValueError):
        run_kbm_beta_scan(np.array([1.0e-4]), init_species_index=-1)
    with pytest.raises(ValueError):
        run_kbm_beta_scan(np.array([1.0e-4]), density_species_index=2)


def test_run_kbm_linear_rejects_invalid_fit_and_species_indices(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_flux_tube_geometry",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.apply_geometry_grid_defaults", lambda geom, grid: grid
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_spectral_grid", lambda cfg: _grid_full()
    )
    with pytest.raises(ValueError, match="fit_signal"):
        run_kbm_linear(
            cfg=KBMBaseCase(),
            fit_signal="bad",
            params=SimpleNamespace(rho_star=1.0),
            terms=LinearTerms(),
            Nl=2,
            Nm=2,
        )
    with pytest.raises(ValueError, match="init_species_index"):
        run_kbm_linear(
            cfg=KBMBaseCase(),
            init_species_index=2,
            params=SimpleNamespace(rho_star=1.0),
            terms=LinearTerms(),
            Nl=2,
            Nm=2,
        )
    with pytest.raises(ValueError, match="density_species_index"):
        run_kbm_linear(
            cfg=KBMBaseCase(),
            density_species_index=-1,
            params=SimpleNamespace(rho_star=1.0),
            terms=LinearTerms(),
            Nl=2,
            Nm=2,
        )


def test_run_kbm_beta_scan_auto_krylov_invalid_growth_falls_back_to_time(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._two_species_params",
        lambda *args, **kwargs: SimpleNamespace(rho_star=1.0, nu=0.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.select_kbm_solver_auto",
        lambda *args, **kwargs: "krylov",
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.dominant_eigenpair",
        lambda *args, **kwargs: (
            -0.1 + 0.2j,
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.integrate_linear_diagnostics",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
            2.0 * np.ones((2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._select_fit_signal",
        lambda *args, **kwargs: np.array([1.0 + 0.0j, 2.0 + 0.0j]),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.15, -0.07, 0.0, 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_kbm_beta_scan(
        np.array([1.0e-4]),
        solver="auto",
        fit_signal="density",
        reference_aligned=False,
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
    )

    np.testing.assert_allclose(scan.gamma, [0.15])
    np.testing.assert_allclose(scan.omega, [-0.07])


def test_run_kbm_beta_scan_explicit_time_diagnostic_fallback_ladder(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._two_species_params",
        lambda *args, **kwargs: SimpleNamespace(rho_star=1.0, nu=0.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.select_kbm_solver_auto",
        lambda *args, **kwargs: "explicit_time",
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    cases = iter(
        [
            {
                "t": np.array([0.0, 1.0]),
                "growth": (0.21, -0.08, None, None, 0.5),
                "omega_series": None,
                "fit": None,
                "method": "z_index",
                "expected": (0.21, -0.08),
            },
            {
                "t": np.array([0.0, 1.0]),
                "growth": ValueError("phi fit failed"),
                "omega_series": (0.22, -0.09, None, None),
                "fit": None,
                "method": "max",
                "expected": (0.22, -0.09),
            },
            {
                "t": np.array([0.0, 1.0]),
                "growth": ValueError("phi fit failed"),
                "omega_series": ValueError("omega fit failed"),
                "fit": (0.23, -0.10, 0.0, 1.0),
                "method": "z_index",
                "expected": (0.23, -0.10),
            },
            {
                "t": np.array([0.0, 1.0]),
                "growth": None,
                "omega_series": None,
                "fit": (0.24, -0.11, 0.0, 1.0),
                "method": "project",
                "expected": (0.24, -0.11),
            },
            {
                "t": np.array([0.0]),
                "growth": None,
                "omega_series": None,
                "fit": None,
                "method": "z_index",
                "expected": (np.nan, np.nan),
            },
        ]
    )

    def _fake_integrate(*args, **kwargs):
        case = current["case"]
        return (
            case["t"],
            np.ones((len(case["t"]), 1, 1, 3), dtype=np.complex64),
            np.ones((1, 1, len(case["t"])), dtype=float),
            -np.ones((1, 1, len(case["t"])), dtype=float),
            None,
        )

    def _fake_growth(*args, **kwargs):
        value = current["case"]["growth"]
        if isinstance(value, Exception):
            raise value
        return value

    def _fake_omega_series(*args, **kwargs):
        value = current["case"]["omega_series"]
        if isinstance(value, Exception):
            raise value
        return value

    def _fake_fit(*args, **kwargs):
        return current["case"]["fit"]

    current: dict[str, object] = {}
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.integrate_linear_explicit_diagnostics",
        _fake_integrate,
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.instantaneous_growth_rate_from_phi", _fake_growth
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.windowed_growth_rate_from_omega_series",
        _fake_omega_series,
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kbm.extract_mode_time_series",
        lambda *args, **kwargs: np.ones(2, dtype=np.complex64),
    )
    monkeypatch.setattr("spectraxgk.validation.benchmarks.kbm.fit_growth_rate_auto", _fake_fit)

    observed_gamma = []
    observed_omega = []
    for case in cases:
        current["case"] = case
        scan = run_kbm_beta_scan(
            np.array([1.0e-4]),
            solver="explicit_time",
            fit_signal="phi",
            mode_method=case["method"],
            reference_aligned=True,
            Nl=2,
            Nm=2,
            dt=0.1,
            steps=2,
        )
        observed_gamma.append(scan.gamma[0])
        observed_omega.append(scan.omega[0])

    np.testing.assert_allclose(observed_gamma[:4], [0.21, 0.22, 0.23, 0.24])
    np.testing.assert_allclose(observed_omega[:4], [-0.08, -0.09, -0.10, -0.11])
    assert np.isnan(observed_gamma[-1])
    assert np.isnan(observed_omega[-1])


def test_run_kinetic_scan_diffrax_streaming_density_batch(monkeypatch) -> None:
    cfg0 = KineticElectronBaseCase()
    cfg = replace(
        cfg0,
        time=replace(
            cfg0.time,
            use_diffrax=True,
            diffrax_adaptive=False,
            dt=0.1,
            t_max=0.2,
            sample_stride=1,
            diffrax_max_steps=8,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.select_ky_grid", _select_grid_dynamic
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._kinetic_reference_init_cfg",
        lambda init_cfg, *, reference_aligned: init_cfg,
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 2, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.integrate_linear_diffrax_streaming",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 2, 1, 3), dtype=np.complex64),
            np.array([0.11, 0.22], dtype=float),
            np.array([-0.03, -0.04], dtype=float),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_kinetic_scan(
        np.array([0.2, 0.3]),
        cfg=cfg,
        time_cfg=cfg.time,
        solver="time",
        params=SimpleNamespace(rho_star=1.0, nu=0.0),
        terms=LinearTerms(),
        fit_signal="density",
        Nl=2,
        Nm=2,
        ky_batch=2,
        dt=0.1,
        steps=2,
        reference_aligned=False,
    )

    np.testing.assert_allclose(scan.ky, [0.2, 0.3])
    np.testing.assert_allclose(scan.gamma, [0.11, 0.22])
    np.testing.assert_allclose(scan.omega, [-0.03, -0.04])


def test_run_kinetic_scan_plain_krylov_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._kinetic_reference_init_cfg",
        lambda init_cfg, *, reference_aligned: init_cfg,
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    vals = iter(
        [
            (0.41 + 0.13j, np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64)),
            (0.51 - 0.17j, np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64)),
        ]
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.dominant_eigenpair",
        lambda *args, **kwargs: next(vals),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_kinetic_scan(
        np.array([0.2, 0.3]),
        solver="krylov",
        params=SimpleNamespace(rho_star=1.0, nu=0.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        ky_batch=1,
        reference_aligned=False,
    )

    np.testing.assert_allclose(scan.gamma, [0.41, 0.51])
    np.testing.assert_allclose(scan.omega, [-0.13, 0.17])


def test_run_kinetic_scan_time_config_density_mode_only(monkeypatch) -> None:
    cfg0 = KineticElectronBaseCase()
    cfg = replace(
        cfg0,
        time=replace(cfg0.time, use_diffrax=False, dt=0.1, t_max=0.2, sample_stride=1),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.select_ky_grid", _select_grid_dynamic
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._kinetic_reference_init_cfg",
        lambda init_cfg, *, reference_aligned: init_cfg,
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.integrate_linear_from_config",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 2, 1, 3), dtype=np.complex64),
            np.array(
                [
                    [1.0 + 0.0j, 2.0 + 0.0j],
                    [2.0 + 0.0j, 4.0 + 0.0j],
                ],
                dtype=np.complex64,
            ),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.fit_growth_rate_auto",
        lambda t, signal, **kwargs: (
            float(np.real(signal[0])) / 10.0,
            -0.09,
            0.0,
            float(t[-1]),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_kinetic_scan(
        np.array([0.2, 0.3]),
        cfg=cfg,
        time_cfg=cfg.time,
        solver="time",
        fit_signal="density",
        mode_only=True,
        params=SimpleNamespace(rho_star=1.0, nu=0.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        ky_batch=2,
        reference_aligned=False,
    )

    np.testing.assert_allclose(scan.gamma, [0.1, 0.2])
    np.testing.assert_allclose(scan.omega, [-0.09, -0.09])


def test_run_kinetic_scan_rejects_invalid_batch_and_species_indices() -> None:
    with pytest.raises(ValueError):
        run_kinetic_scan(np.array([0.2]), ky_batch=0)
    with pytest.raises(ValueError):
        run_kinetic_scan(np.array([0.2]), init_species_index=2)
    with pytest.raises(ValueError):
        run_kinetic_scan(np.array([0.2]), density_species_index=-1)


def test_run_kinetic_linear_rejects_invalid_species_indices(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_linear.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    params = SimpleNamespace(rho_star=1.0)
    with pytest.raises(ValueError, match="init_species_index"):
        run_kinetic_linear(
            cfg=KineticElectronBaseCase(),
            params=params,
            terms=LinearTerms(),
            init_species_index=-1,
            Nl=2,
            Nm=2,
        )
    with pytest.raises(ValueError, match="density_species_index"):
        run_kinetic_linear(
            cfg=KineticElectronBaseCase(),
            params=params,
            terms=LinearTerms(),
            density_species_index=2,
            Nl=2,
            Nm=2,
        )


def test_run_kinetic_scan_diffrax_mode_only_phi_uses_z_index_save_mode(
    monkeypatch,
) -> None:
    cfg0 = KineticElectronBaseCase()
    cfg = replace(
        cfg0,
        time=replace(
            cfg0.time,
            use_diffrax=True,
            diffrax_adaptive=False,
            dt=0.1,
            t_max=0.2,
            sample_stride=1,
        ),
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def _fake_integrate(*args, **kwargs):
        captured["mode_method"] = kwargs["mode_method"]
        captured["save_mode"] = kwargs["save_mode"]
        return (
            np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.array([[1.0 + 0.0j], [2.0 + 0.0j]], dtype=np.complex64),
        )

    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.integrate_linear_from_config", _fake_integrate
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.11, -0.03, 0.0, 0.1),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.kinetic_scan._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_kinetic_scan(
        np.array([0.2]),
        cfg=cfg,
        time_cfg=cfg.time,
        solver="time",
        fit_signal="phi",
        mode_method="project",
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
        ky_batch=1,
        streaming_fit=False,
        reference_aligned=False,
    )

    assert captured["mode_method"] == "z_index"
    assert captured["save_mode"] is not None
    np.testing.assert_allclose(scan.gamma, [0.11])
    np.testing.assert_allclose(scan.omega, [-0.03])


def test_run_tem_scan_time_config_mode_only_extracts_columns(monkeypatch) -> None:
    cfg0 = TEMBaseCase()
    cfg = replace(
        cfg0,
        time=replace(cfg0.time, use_diffrax=False, dt=0.1, t_max=0.2, sample_stride=1),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr("spectraxgk.validation.benchmarks.tem.select_ky_grid", _select_grid_dynamic)
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 2, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.integrate_linear_from_config",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 2, 1, 3), dtype=np.complex64),
            np.array(
                [
                    [1.0 + 0.0j, 2.0 + 0.0j],
                    [2.0 + 0.0j, 4.0 + 0.0j],
                ],
                dtype=np.complex64,
            ),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.fit_growth_rate_auto",
        lambda t, signal, **kwargs: (
            float(np.real(signal[0])) / 10.0,
            -0.2,
            0.0,
            float(t[-1]),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_tem_scan(
        np.array([0.2, 0.3]),
        cfg=cfg,
        time_cfg=cfg.time,
        solver="time",
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
        ky_batch=2,
        sample_stride=1,
    )

    np.testing.assert_allclose(scan.ky, [0.2, 0.3])
    np.testing.assert_allclose(scan.gamma, [0.1, 0.2])
    np.testing.assert_allclose(scan.omega, [-0.2, -0.2])


def test_run_tem_scan_plain_krylov_branch(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    vals = iter(
        [
            (0.24 + 0.15j, np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64)),
            (0.34 - 0.19j, np.zeros((2, 2, 2, 1, 1, 3), dtype=np.complex64)),
        ]
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.dominant_eigenpair",
        lambda *args, **kwargs: next(vals),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_tem_scan(
        np.array([0.2, 0.3]),
        solver="krylov",
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        ky_batch=1,
    )

    np.testing.assert_allclose(scan.gamma, [0.24, 0.34])
    np.testing.assert_allclose(scan.omega, [-0.15, 0.19])


def test_run_tem_scan_diffrax_streaming_batch(monkeypatch) -> None:
    cfg0 = TEMBaseCase()
    cfg = replace(
        cfg0,
        time=replace(
            cfg0.time,
            use_diffrax=True,
            diffrax_adaptive=False,
            dt=0.1,
            t_max=0.2,
            sample_stride=1,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr("spectraxgk.validation.benchmarks.tem.select_ky_grid", _select_grid_dynamic)
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem._resolve_streaming_window",
        lambda *args, **kwargs: (0.0, 0.2),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.tem.integrate_linear_diffrax_streaming",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2, 2, 1, 3), dtype=np.complex64),
            np.array([0.14, 0.16]),
            np.array([-0.05, -0.07]),
        ),
    )

    scan = run_tem_scan(
        np.array([0.2, 0.3]),
        cfg=cfg,
        time_cfg=cfg.time,
        solver="time",
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        ky_batch=2,
        streaming_fit=True,
    )

    np.testing.assert_allclose(scan.gamma, [0.14, 0.16])
    np.testing.assert_allclose(scan.omega, [-0.05, -0.07])


def test_run_tem_scan_rejects_invalid_batch_and_species_indices() -> None:
    with pytest.raises(ValueError):
        run_tem_scan(np.array([0.2]), ky_batch=0)
    with pytest.raises(ValueError):
        run_tem_scan(np.array([0.2]), init_species_index=-1)
    with pytest.raises(ValueError):
        run_tem_scan(np.array([0.2]), density_species_index=2)


def test_run_etg_scan_rejects_invalid_batch_and_fit_signal() -> None:
    with pytest.raises(ValueError):
        run_etg_scan(np.array([1.0]), ky_batch=0)
    with pytest.raises(ValueError):
        run_etg_scan(np.array([1.0]), fit_signal="bad")


def test_run_etg_scan_auto_solver_uses_time_with_zero_reference_fallback(
    monkeypatch,
) -> None:
    cfg0 = ETGBaseCase()
    cfg = replace(
        cfg0,
        time=replace(cfg0.time, use_diffrax=False, dt=0.1, t_max=0.2, sample_stride=1),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._build_initial_condition",
        lambda *args, **kwargs: np.zeros((1, 2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.integrate_linear",
        lambda *args, **kwargs: (
            np.zeros((2, 1, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.ones((2, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.fit_growth_rate",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("forced fallback")),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.18, -0.06, 0.0, 0.2),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_etg_scan(
        np.array([2.0]),
        cfg=cfg,
        solver="auto",
        params=SimpleNamespace(charge_sign=np.array([-1.0]), rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=2,
        fit_signal="phi",
        auto_window=False,
        tmin=0.0,
        tmax=0.0,
    )

    np.testing.assert_allclose(scan.gamma, [0.18])
    np.testing.assert_allclose(scan.omega, [-0.06])


def test_run_etg_scan_diffrax_streaming_density_batch(monkeypatch) -> None:
    cfg0 = ETGBaseCase()
    cfg = replace(
        cfg0,
        time=replace(
            cfg0.time,
            use_diffrax=True,
            diffrax_adaptive=False,
            dt=0.1,
            t_max=0.2,
            sample_stride=1,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr("spectraxgk.validation.benchmarks.etg.select_ky_grid", _select_grid_dynamic)
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._resolve_streaming_window",
        lambda *args, **kwargs: (0.0, 0.2),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.integrate_linear_diffrax_streaming",
        lambda *args, **kwargs: (
            np.zeros((2, 1, 2, 2, 1, 3), dtype=np.complex64),
            np.array([0.23, 0.27]),
            np.array([-0.13, -0.17]),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_etg_scan(
        np.array([2.0, 3.0]),
        cfg=cfg,
        time_cfg=cfg.time,
        solver="time",
        params=SimpleNamespace(charge_sign=np.array([-1.0]), rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        fit_signal="density",
        ky_batch=2,
        streaming_fit=True,
    )

    np.testing.assert_allclose(scan.gamma, [0.23, 0.27])
    np.testing.assert_allclose(scan.omega, [-0.13, -0.17])


def test_run_etg_scan_time_config_auto_signal_fallback_to_krylov(monkeypatch) -> None:
    cfg0 = ETGBaseCase()
    cfg = replace(
        cfg0,
        time=replace(cfg0.time, use_diffrax=False, dt=0.1, t_max=0.2, sample_stride=1),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._build_initial_condition", _fake_initial_condition
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.integrate_linear_from_config",
        lambda *args, **kwargs: (
            np.zeros((2, 1, 2, 2, 1, 1, 3), dtype=np.complex64),
            (
                np.ones((2, 1, 1, 3), dtype=np.complex64),
                2.0 * np.ones((2, 1, 1, 3), dtype=np.complex64),
            ),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._select_fit_signal_auto",
        lambda *args, **kwargs: (
            np.ones(2, dtype=np.complex64),
            "density",
            -0.3,
            -0.12,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg.run_etg_linear",
        lambda *args, **kwargs: SimpleNamespace(gamma=0.39, omega=-0.21),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.etg._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_etg_scan(
        np.array([2.0]),
        cfg=cfg,
        time_cfg=cfg.time,
        solver="auto",
        params=SimpleNamespace(charge_sign=np.array([-1.0]), rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        fit_signal="auto",
        require_positive=True,
        ky_batch=1,
    )

    np.testing.assert_allclose(scan.gamma, [0.39])
    np.testing.assert_allclose(scan.omega, [-0.21])


def test_run_cyclone_linear_time_cached_uses_integrate_linear_output(
    monkeypatch,
) -> None:
    cfg0 = CycloneBaseCase()
    cfg = replace(
        cfg0,
        time=replace(cfg0.time, use_diffrax=False, dt=0.1, t_max=0.3, sample_stride=1),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear",
        lambda *args, **kwargs: (
            np.zeros((3, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.ones((3, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._select_fit_signal",
        lambda *args, **kwargs: np.array(
            [1.0 + 0.0j, 2.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex64
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.21, -0.07, 0.0, 0.2),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_cyclone_linear(
        cfg=cfg,
        solver="time",
        fit_signal="phi",
        reference_aligned=False,
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=3,
    )

    np.testing.assert_allclose(result.t, [0.0, 0.1, 0.2])
    assert result.gamma == 0.21
    assert result.omega == -0.07


def test_run_cyclone_linear_time_config_auto_signal_records_density(
    monkeypatch,
) -> None:
    cfg0 = CycloneBaseCase()
    time_cfg = replace(cfg0.time, use_diffrax=False, dt=0.2, t_max=0.6, sample_stride=2)
    status: list[str] = []
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def _fake_from_config(*args, **kwargs):
        assert kwargs["save_field"] == "phi+density"
        assert kwargs["density_species_index"] == 0
        phi = np.ones((3, 1, 1, 3), dtype=np.complex64)
        density = 2.0 * phi
        return np.zeros((3, 2, 2, 1, 1, 3), dtype=np.complex64), (phi, density)

    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_from_config", _fake_from_config
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._select_fit_signal_auto",
        lambda *args, **kwargs: (
            np.ones(3, dtype=np.complex64),
            "density",
            np.nan,
            np.nan,
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_cyclone_linear(
        cfg=cfg0,
        time_cfg=time_cfg,
        solver="time",
        fit_signal="auto",
        reference_aligned=False,
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=3,
        sample_stride=2,
        status_callback=status.append,
    )

    np.testing.assert_allclose(result.t, [0.0, 0.2, 0.4])
    assert result.gamma == 0.0
    assert result.omega == 0.0
    assert any("automatic fit selected signal 'density'" in msg for msg in status)


def test_run_cyclone_linear_density_fit_uses_diagnostics_integrator(
    monkeypatch,
) -> None:
    cfg0 = CycloneBaseCase()
    cfg = replace(cfg0, time=replace(cfg0.time, use_diffrax=False))
    status: list[str] = []
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def _fake_diagnostics(*args, **kwargs):
        assert kwargs["species_index"] == 0
        assert kwargs["record_hl_energy"] is False
        phi = np.ones((3, 1, 1, 3), dtype=np.complex64)
        density = 3.0 * phi
        return np.zeros((3, 2, 2, 1, 1, 3), dtype=np.complex64), phi, density

    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear_diagnostics", _fake_diagnostics
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._select_fit_signal",
        lambda *args, **kwargs: np.array(
            [1.0 + 0.0j, 3.0 + 0.0j, 9.0 + 0.0j], dtype=np.complex64
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.31, -0.13, 0.0, 0.2),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    result = run_cyclone_linear(
        cfg=cfg,
        solver="time",
        fit_signal="density",
        reference_aligned=False,
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=3,
        sample_stride=1,
        status_callback=status.append,
    )

    assert result.gamma == 0.31
    assert result.omega == -0.13
    assert any("explicit diagnostics integrator" in msg for msg in status)


def test_run_cyclone_scan_time_cached_uses_integrate_linear_output(monkeypatch) -> None:
    cfg0 = CycloneBaseCase()
    cfg = replace(
        cfg0,
        time=replace(cfg0.time, use_diffrax=False, dt=0.1, t_max=0.3, sample_stride=1),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_spectral_grid", lambda cfg: _grid_full()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.select_ky_grid", lambda grid, idx: _grid_sel()
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.SAlphaGeometry.from_config",
        lambda cfg: SimpleNamespace(gradpar=lambda: 1.0, s_hat=0.8),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.build_linear_cache",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.integrate_linear",
        lambda *args, **kwargs: (
            np.zeros((3, 2, 2, 1, 1, 3), dtype=np.complex64),
            np.ones((3, 1, 1, 3), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.extract_mode_time_series",
        lambda arr, sel, method: np.array(
            [1.0 + 0.0j, 2.0 + 0.0j, 4.0 + 0.0j], dtype=np.complex64
        ),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.19, -0.05, 0.0, 0.2),
    )
    monkeypatch.setattr(
        "spectraxgk.validation.benchmarks.cyclone._normalize_growth_rate",
        lambda g, o, params, norm: (g, o),
    )

    scan = run_cyclone_scan(
        np.array([0.3]),
        cfg=cfg,
        solver="time",
        fit_signal="phi",
        reference_aligned=False,
        params=SimpleNamespace(rho_star=1.0),
        terms=LinearTerms(),
        Nl=2,
        Nm=2,
        dt=0.1,
        steps=3,
        ky_batch=1,
    )

    np.testing.assert_allclose(scan.ky, [0.3])
    np.testing.assert_allclose(scan.gamma, [0.19])
    np.testing.assert_allclose(scan.omega, [-0.05])
