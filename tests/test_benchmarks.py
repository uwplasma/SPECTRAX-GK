"""Benchmark utilities and reference data tests."""

import numpy as np
import pytest

from spectraxgk.analysis import fit_growth_rate
from spectraxgk.benchmarks import load_cyclone_reference, run_cyclone_linear
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
    result = run_cyclone_linear(cfg=cfg, steps=5, dt=0.1)
    assert result.phi_t.shape[0] == 5
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)


def test_run_cyclone_linear_defaults():
    """Default cfg/params path should run without error."""
    grid = GridConfig(Nx=6, Ny=6, Nz=8, Lx=62.8, Ly=62.8)
    cfg = CycloneBaseCase(grid=grid)
    result = run_cyclone_linear(cfg=cfg, steps=3, dt=0.1)
    assert result.phi_t.shape[0] == 3
