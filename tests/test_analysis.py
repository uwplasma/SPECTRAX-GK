"""Analysis helper tests for mode extraction and fit windows."""

import numpy as np

from spectraxgk.analysis import (
    ModeSelection,
    extract_mode,
    extract_mode_time_series,
    fit_growth_rate_auto,
    select_fit_window,
)


def test_extract_mode_time_series_methods():
    """Mode extraction should work for z_index, max, and svd modes."""
    t = np.linspace(0.0, 1.0, 64)
    gamma = 0.1
    omega = 0.2
    ts = np.exp((gamma + 1j * omega) * t)
    spatial = np.linspace(1.0, 2.0, 4)
    data = ts[:, None] * spatial[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=1)

    z_series = extract_mode_time_series(phi_t, sel, method="z_index")
    assert np.allclose(z_series, ts * spatial[1])

    max_series = extract_mode_time_series(phi_t, sel, method="max")
    assert np.allclose(max_series, ts * spatial[-1])

    svd_series = extract_mode_time_series(phi_t, sel, method="svd")
    ratio = svd_series / ts
    ratio_norm = ratio / ratio[0]
    assert np.allclose(ratio_norm, np.ones_like(ratio_norm), atol=1.0e-6)

    direct = extract_mode(phi_t, sel)
    assert np.allclose(direct, ts * spatial[1])

    try:
        extract_mode_time_series(phi_t, sel, method="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mode method should raise ValueError")


def test_select_fit_window_and_auto_fit():
    """Auto window should favor the clean exponential region."""
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.12
    omega = 0.3
    signal = np.exp((gamma + 1j * omega) * t)
    signal = signal.copy()
    signal[:40] *= 1.0 + 0.2 * np.sin(5.0 * t[:40])

    tmin, tmax = select_fit_window(t, signal, window_fraction=0.3, min_points=20)
    assert tmax > tmin
    assert tmin >= t[20]

    g_fit, w_fit, _tmin, _tmax = fit_growth_rate_auto(
        t, signal, window_fraction=0.3, min_points=20
    )
    assert np.isclose(g_fit, gamma, rtol=1e-2, atol=1e-2)
    assert np.isclose(w_fit, omega, rtol=1e-2, atol=1e-2)

    try:
        select_fit_window(np.array([[0.0, 1.0]]), signal)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid t shape should raise ValueError")

    try:
        select_fit_window(t, np.array([[0.0, 1.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid signal shape should raise ValueError")

    try:
        select_fit_window(t[:5], signal[:3])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched length should raise ValueError")

    try:
        select_fit_window(t[:1], signal[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("too-short signal should raise ValueError")

    try:
        select_fit_window(t[:2], signal[:2], window_fraction=0.1, min_points=1)
    except ValueError:
        pass
    else:
        raise AssertionError("window too short should raise ValueError")

    flat_signal = np.ones_like(signal)
    _ = select_fit_window(t, flat_signal, window_fraction=0.3, min_points=20)

    g_fit2, w_fit2, _tmin2, _tmax2 = fit_growth_rate_auto(t, signal, tmin=2.0)
    assert np.isfinite(g_fit2)
    assert np.isfinite(w_fit2)
