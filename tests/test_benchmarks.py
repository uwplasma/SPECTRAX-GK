import numpy as np
import pytest

from spectraxgk.benchmarks import fit_growth_rate, load_cyclone_reference


def test_load_cyclone_reference():
    ref = load_cyclone_reference()
    assert ref.ky.shape == ref.omega.shape == ref.gamma.shape
    assert ref.ky.size == 12
    idx = int(np.argmin(np.abs(ref.ky - 0.3)))
    assert np.isclose(ref.ky[idx], 0.3)
    assert np.isclose(ref.omega[idx], 0.28199035, rtol=1e-6)
    assert np.isclose(ref.gamma[idx], 0.09301763, rtol=1e-6)


def test_fit_growth_rate_exact():
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.12
    omega = 0.34
    signal = np.exp((gamma + 1j * omega) * t)
    g_fit, w_fit = fit_growth_rate(t, signal)
    assert np.isclose(g_fit, gamma, rtol=1e-3, atol=1e-3)
    assert np.isclose(w_fit, omega, rtol=1e-3, atol=1e-3)


def test_fit_growth_rate_window():
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.08
    omega = 0.21
    signal = np.exp((gamma + 1j * omega) * t)
    g_fit, w_fit = fit_growth_rate(t, signal, tmin=5.0)
    assert np.isclose(g_fit, gamma, rtol=1e-3, atol=1e-3)
    assert np.isclose(w_fit, omega, rtol=1e-3, atol=1e-3)


def test_fit_growth_rate_tmax():
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.06
    omega = 0.18
    signal = np.exp((gamma + 1j * omega) * t)
    g_fit, w_fit = fit_growth_rate(t, signal, tmax=7.0)
    assert np.isclose(g_fit, gamma, rtol=1e-3, atol=1e-3)
    assert np.isclose(w_fit, omega, rtol=1e-3, atol=1e-3)


def test_fit_growth_rate_invalid():
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([0.0, 1.0]), np.array([1.0 + 0j]))
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([[0.0, 1.0]]), np.array([1.0 + 0j, 2.0 + 0j]))
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([0.0, 1.0]), np.array([[1.0 + 0j, 2.0 + 0j]]))
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([0.0, 1.0]), np.array([1.0 + 0j, 2.0 + 0j]), tmin=2.0)
