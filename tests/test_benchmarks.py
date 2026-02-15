"""Benchmark utilities and reference data tests."""

import numpy as np
import pytest

from spectraxgk.analysis import fit_growth_rate
from spectraxgk.benchmarks import (
    compare_cyclone_to_reference,
    load_cyclone_reference,
    run_cyclone_linear,
    run_cyclone_scan,
)
from spectraxgk.config import CycloneBaseCase, GridConfig


def test_load_cyclone_reference():
    """Reference CSV must load and match known Cyclone values."""
    ref = load_cyclone_reference()
    assert ref.ky.shape == ref.omega.shape == ref.gamma.shape
    assert ref.ky.size == 12
    idx = int(np.argmin(np.abs(ref.ky - 0.3)))
    assert np.isclose(ref.ky[idx], 0.3)
    assert np.isclose(ref.omega[idx], 0.28199035, rtol=1e-6)
    assert np.isclose(ref.gamma[idx], 0.09301763, rtol=1e-6)


def test_fit_growth_rate_exact():
    """Exact exponential signal should be recovered by the fitter."""
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.12
    omega = 0.34
    signal = np.exp((gamma + 1j * omega) * t)
    g_fit, w_fit = fit_growth_rate(t, signal)
    assert np.isclose(g_fit, gamma, rtol=1e-3, atol=1e-3)
    assert np.isclose(w_fit, omega, rtol=1e-3, atol=1e-3)


def test_fit_growth_rate_window():
    """Windowing should not bias the fit for a pure exponential."""
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.08
    omega = 0.21
    signal = np.exp((gamma + 1j * omega) * t)
    g_fit, w_fit = fit_growth_rate(t, signal, tmin=5.0)
    assert np.isclose(g_fit, gamma, rtol=1e-3, atol=1e-3)
    assert np.isclose(w_fit, omega, rtol=1e-3, atol=1e-3)


def test_fit_growth_rate_tmax():
    """A tmax cut should still recover the correct growth rate."""
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.06
    omega = 0.18
    signal = np.exp((gamma + 1j * omega) * t)
    g_fit, w_fit = fit_growth_rate(t, signal, tmax=7.0)
    assert np.isclose(g_fit, gamma, rtol=1e-3, atol=1e-3)
    assert np.isclose(w_fit, omega, rtol=1e-3, atol=1e-3)


def test_fit_growth_rate_invalid():
    """The fitter should reject malformed inputs."""
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([0.0, 1.0]), np.array([1.0 + 0j]))
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([[0.0, 1.0]]), np.array([1.0 + 0j, 2.0 + 0j]))
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([0.0, 1.0]), np.array([[1.0 + 0j, 2.0 + 0j]]))
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([0.0, 1.0]), np.array([1.0 + 0j, 2.0 + 0j]), tmin=2.0)


def test_run_cyclone_linear_shapes():
    """Smoke test for the Cyclone linear runner on a tiny grid."""
    grid = GridConfig(Nx=8, Ny=8, Nz=16, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, steps=5, dt=0.1, method="rk4")
    assert result.phi_t.shape[0] == 5
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_run_cyclone_linear_defaults():
    """Default cfg/params path should run without error."""
    grid = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, steps=3, dt=0.1, method="rk2")
    assert result.phi_t.shape[0] == 3


def test_run_cyclone_linear_full_operator_smoke():
    """Full operator path should execute without NaNs on a tiny run."""
    grid = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, steps=3, dt=0.1, method="rk2", operator="full")
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_cyclone_scan_and_compare():
    """Scan helper should return arrays and comparison should report errors."""
    grid = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    ky_values = np.array([0.2, 0.3])
    scan = run_cyclone_scan(ky_values, cfg=cfg, steps=3, dt=0.1, method="euler")
    assert scan.ky.shape == ky_values.shape
    ref = load_cyclone_reference()
    result = run_cyclone_linear(cfg=cfg, steps=3, dt=0.1, ky_target=0.3, method="euler")
    comparison = compare_cyclone_to_reference(result, ref)
    assert comparison.ky > 0.0
    assert np.isfinite(comparison.rel_gamma)


def test_cyclone_physics_regression():
    """Cyclone growth rates should track published values at ky rho_i = 0.3."""
    grid = GridConfig(Nx=8, Ny=12, Nz=24, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, ky_target=0.3, steps=300, dt=0.02, tmin=3.0, method="rk4")
    ref = load_cyclone_reference()
    idx = int(np.argmin(np.abs(ref.ky - 0.3)))
    assert np.isclose(result.gamma, ref.gamma[idx], rtol=0.25)
    assert np.isclose(result.omega, ref.omega[idx], rtol=0.25)


def test_cyclone_scan_regression():
    """Reduced ky scan should remain within reference trends."""
    grid = GridConfig(Nx=8, Ny=12, Nz=24, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    ky_values = np.array([0.3, 0.4])
    scan = run_cyclone_scan(ky_values, cfg=cfg, steps=300, dt=0.02, tmin=3.0, method="rk4")
    ref = load_cyclone_reference()
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        assert np.isclose(gamma, ref.gamma[idx], rtol=1.3)
        assert np.isclose(omega, ref.omega[idx], rtol=0.6)
