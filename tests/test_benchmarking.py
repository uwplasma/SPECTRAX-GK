from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk.analysis import ModeSelection
from spectraxgk.benchmarking import (
    estimate_observed_order,
    late_time_linear_metrics,
    normalize_eigenfunction,
    run_linear_scan,
    run_scan_and_mode,
    windowed_nonlinear_metrics,
)
from spectraxgk.benchmarks import LinearRunResult, LinearScanResult
from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.runtime import RuntimeLinearResult, RuntimeNonlinearResult


def test_normalize_eigenfunction_uses_nearest_zero() -> None:
    eig = np.array([2.0 + 0.0j, 4.0 + 0.0j, 8.0 + 0.0j])
    z = np.array([-1.0, 0.1, 0.9])

    out = normalize_eigenfunction(eig, z)

    np.testing.assert_allclose(out, eig / eig[1])


def test_normalize_eigenfunction_leaves_zero_scale_unchanged() -> None:
    eig = np.array([1.0 + 0.0j, 0.0 + 0.0j, 3.0 + 0.0j])
    z = np.array([-1.0, 0.0, 1.0])

    out = normalize_eigenfunction(eig, z)

    np.testing.assert_allclose(out, eig)


def test_run_linear_scan_applies_resolution_and_krylov_policies() -> None:
    calls: list[dict[str, object]] = []

    def fake_run_linear_fn(**kwargs):
        calls.append(kwargs)
        ky = float(kwargs["ky_target"])
        return SimpleNamespace(gamma=ky + 1.0, omega=-ky, ky=ky + 0.01)

    result = run_linear_scan(
        ky_values=np.array([0.1, 0.3]),
        run_linear_fn=fake_run_linear_fn,
        cfg=object(),
        Nl=2,
        Nm=3,
        dt=np.array([0.01, 0.02]),
        steps=np.array([10, 20]),
        method="rk4",
        solver="time",
        krylov_cfg="base",
        window_kw={"window_fraction": 0.5},
        tmin=np.array([1.0, 2.0]),
        tmax=np.array([3.0, 4.0]),
        auto_window=False,
        run_kwargs={"tag": "ok"},
        resolution_policy=lambda ky: (4, 5) if ky < 0.2 else (6, 7),
        krylov_policy=lambda ky: f"kcfg-{ky:.1f}",
    )

    assert isinstance(result, LinearScanResult)
    np.testing.assert_allclose(result.ky, [0.11, 0.31])
    np.testing.assert_allclose(result.gamma, [1.1, 1.3])
    np.testing.assert_allclose(result.omega, [-0.1, -0.3])
    assert calls[0]["Nl"] == 4
    assert calls[0]["Nm"] == 5
    assert calls[0]["krylov_cfg"] == "kcfg-0.1"
    assert calls[0]["tmin"] == 1.0
    assert calls[1]["Nl"] == 6
    assert calls[1]["Nm"] == 7
    assert calls[1]["krylov_cfg"] == "kcfg-0.3"
    assert calls[1]["tmax"] == 4.0
    assert calls[1]["tag"] == "ok"


def test_run_scan_and_mode_uses_selected_ky_and_fit_window(monkeypatch) -> None:
    selection = ModeSelection(ky_index=0, kx_index=0, z_index=1)
    run = LinearRunResult(
        t=np.array([0.0, 1.0, 2.0]),
        phi_t=np.ones((3, 1, 1, 3), dtype=np.complex128),
        gamma=0.4,
        omega=-0.2,
        ky=0.3,
        selection=selection,
    )
    calls: list[dict[str, object]] = []

    def fake_linear_fn(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setattr(
        "spectraxgk.benchmarking.extract_mode_time_series",
        lambda phi_t, sel, method: np.array([1.0 + 0.0j, 2.0 + 0.0j, 4.0 + 0.0j]),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarking.fit_growth_rate_auto",
        lambda t, signal, **kwargs: (0.5, -0.1, 0.25, 1.75),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarking.extract_eigenfunction",
        lambda phi_t, t, selection, z, method, tmin, tmax: np.array([1.0, 2.0, 3.0]),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarking.build_spectral_grid",
        lambda _grid: SimpleNamespace(z=np.array([-1.0, 0.0, 1.0])),
    )
    cfg = SimpleNamespace(grid=object())

    result = run_scan_and_mode(
        ky_values=np.array([0.1, 0.3]),
        scan_fn=None,
        linear_fn=fake_linear_fn,
        cfg=cfg,
        Nl=2,
        Nm=2,
        dt=np.array([0.1, 0.2]),
        steps=np.array([5, 6]),
        method="rk2",
        solver="time",
        mode_solver="krylov",
        krylov_cfg="kcfg",
        window_kw={"window_fraction": 0.5},
        resolution_policy=lambda ky: (3, 4) if ky < 0.2 else (5, 6),
    )

    assert result.ky_selected == 0.3
    np.testing.assert_allclose(result.scan.gamma, [0.4, 0.4])
    np.testing.assert_allclose(result.eigenfunction, [1.0, 2.0, 3.0])
    assert result.tmin == 0.25
    assert result.tmax == 1.75
    assert calls[0]["solver"] == "time"
    assert calls[1]["solver"] == "time"
    assert calls[2]["solver"] == "krylov"
    assert calls[2]["Nl"] == 5
    assert calls[2]["Nm"] == 6
    assert calls[2]["ky_target"] == 0.3


def test_run_scan_and_mode_short_trace_skips_fit(monkeypatch) -> None:
    run = LinearRunResult(
        t=np.array([0.0]),
        phi_t=np.ones((1, 1, 1, 2), dtype=np.complex128),
        gamma=0.2,
        omega=0.1,
        ky=0.2,
        selection=ModeSelection(ky_index=0, kx_index=0),
    )

    monkeypatch.setattr(
        "spectraxgk.benchmarking.build_spectral_grid",
        lambda _grid: SimpleNamespace(z=np.array([-1.0, 1.0])),
    )
    monkeypatch.setattr(
        "spectraxgk.benchmarking.extract_eigenfunction",
        lambda *args, **kwargs: np.array([1.0, -1.0]),
    )

    result = run_scan_and_mode(
        ky_values=np.array([0.2]),
        scan_fn=None,
        linear_fn=lambda **kwargs: run,
        cfg=SimpleNamespace(grid=object()),
        Nl=1,
        Nm=1,
        dt=0.1,
        steps=2,
        method="rk2",
        solver="time",
        mode_solver="time",
        krylov_cfg=None,
        window_kw={"window_fraction": 0.5},
        select_ky=lambda scan: float(scan.ky[0]),
    )

    assert result.tmin is None
    assert result.tmax is None
    np.testing.assert_allclose(result.eigenfunction, [1.0, -1.0])


def test_late_time_linear_metrics_from_linear_run_result() -> None:
    gamma = 0.35
    omega = -0.18
    t = np.linspace(0.0, 4.0, 9)
    z_profile = np.array([1.0, 0.5 - 0.25j])
    signal = np.exp((gamma - 1j * omega) * t)
    phi_t = signal[:, None, None, None] * z_profile[None, None, None, :]
    run = LinearRunResult(
        t=t,
        phi_t=phi_t,
        gamma=gamma,
        omega=omega,
        ky=0.3,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        gamma_t=np.full_like(t, gamma, dtype=float),
        omega_t=np.full_like(t, omega, dtype=float),
    )

    metrics = late_time_linear_metrics(run, tail_fraction=0.5)

    assert metrics.signal_source == "phi_t:project"
    assert metrics.nsamples == 5
    assert metrics.gamma_fit == pytest.approx(gamma, rel=1.0e-3)
    assert metrics.omega_fit == pytest.approx(omega, rel=1.0e-3)
    assert metrics.gamma_tail_mean == pytest.approx(gamma)
    assert metrics.omega_tail_mean == pytest.approx(omega)
    assert metrics.tmin == pytest.approx(2.0)
    assert metrics.tmax == pytest.approx(4.0)


def test_late_time_linear_metrics_runtime_signal_and_scalar_fallback() -> None:
    gamma = 0.22
    omega = -0.07
    t = np.linspace(0.0, 2.0, 5)
    signal = np.exp((gamma - 1j * omega) * t)
    runtime = RuntimeLinearResult(
        ky=0.2,
        gamma=gamma,
        omega=omega,
        selection=ModeSelection(ky_index=0, kx_index=0),
        t=t,
        signal=signal,
    )

    metrics = late_time_linear_metrics(runtime, tail_fraction=0.4)
    assert metrics.signal_source == "signal"
    assert metrics.gamma_fit == pytest.approx(gamma, rel=1.0e-3)
    assert metrics.omega_fit == pytest.approx(omega, rel=1.0e-3)
    assert metrics.nsamples == 2

    scalar_only = late_time_linear_metrics(SimpleNamespace(gamma=0.1, omega=-0.2))
    assert scalar_only.signal_source == "scalar"
    assert scalar_only.gamma_fit == pytest.approx(0.1)
    assert scalar_only.omega_fit == pytest.approx(-0.2)
    assert scalar_only.nsamples == 1


def test_windowed_nonlinear_metrics_from_runtime_result() -> None:
    diagnostics = SimulationDiagnostics(
        t=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        dt_t=np.full(5, 0.1),
        dt_mean=np.full(5, 0.1),
        gamma_t=np.zeros(5),
        omega_t=np.zeros(5),
        Wg_t=np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
        Wphi_t=np.array([0.5, 0.75, 1.0, 1.25, 1.5]),
        Wapar_t=np.zeros(5),
        heat_flux_t=np.array([0.0, 0.2, 0.4, 0.6, 0.8]),
        particle_flux_t=np.zeros(5),
        energy_t=np.zeros(5),
        phi_mode_t=np.array([0.0, 1.0 + 0.0j, 1.0 + 1.0j, 2.0 + 0.0j, 2.0 + 1.0j]),
    )
    result = RuntimeNonlinearResult(t=np.asarray(diagnostics.t), diagnostics=diagnostics)

    metrics = windowed_nonlinear_metrics(result, start_fraction=0.6)

    assert metrics.nsamples == 2
    assert metrics.tmin == pytest.approx(3.0)
    assert metrics.tmax == pytest.approx(4.0)
    assert metrics.heat_flux_mean == pytest.approx(0.7)
    assert metrics.wphi_mean == pytest.approx(1.375)
    assert metrics.wg_mean == pytest.approx(2.75)
    assert metrics.phi_mode_envelope_max == pytest.approx(np.sqrt(5.0))


def test_windowed_nonlinear_metrics_rejects_missing_or_empty_diagnostics() -> None:
    with pytest.raises(ValueError):
        windowed_nonlinear_metrics(RuntimeNonlinearResult(t=np.array([]), diagnostics=None))

    bad = SimulationDiagnostics(
        t=np.array([0.0, 1.0]),
        dt_t=np.full(2, 0.1),
        dt_mean=np.full(2, 0.1),
        gamma_t=np.zeros(2),
        omega_t=np.zeros(2),
        Wg_t=np.array([np.nan, np.nan]),
        Wphi_t=np.array([1.0, 2.0]),
        Wapar_t=np.zeros(2),
        heat_flux_t=np.array([1.0, 2.0]),
        particle_flux_t=np.zeros(2),
        energy_t=np.zeros(2),
    )
    with pytest.raises(ValueError):
        windowed_nonlinear_metrics(bad)


def test_estimate_observed_order_returns_asymptotic_pairwise_orders() -> None:
    step_sizes = np.array([0.4, 0.2, 0.1, 0.05])
    errors = 3.0 * step_sizes**2

    metrics = estimate_observed_order(step_sizes, errors)

    np.testing.assert_allclose(metrics.orders, [2.0, 2.0, 2.0], atol=1.0e-12)
    assert metrics.asymptotic_order == pytest.approx(2.0)

    with pytest.raises(ValueError):
        estimate_observed_order(np.array([0.1]), np.array([0.01]))
    with pytest.raises(ValueError):
        estimate_observed_order(np.array([0.2, 0.2]), np.array([0.1, 0.025]))
