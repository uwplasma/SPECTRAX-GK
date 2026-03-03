"""Analysis helper tests for mode extraction and fit windows."""

import numpy as np
import pytest

from spectraxgk.analysis import (
    ModeSelection,
    extract_mode,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate_auto,
    gx_growth_rate_from_omega_series,
    select_fit_window,
)


def test_extract_mode_time_series_methods():
    """Mode extraction should work for z_index, max, and svd modes."""
    t = np.linspace(0.0, 1.0, 64)
    gamma = 0.1
    omega = 0.2
    ts = np.exp((gamma - 1j * omega) * t)
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

    project_series = extract_mode_time_series(phi_t, sel, method="project")
    proj_ratio = project_series / ts
    proj_norm = proj_ratio / proj_ratio[0]
    assert np.allclose(proj_norm, np.ones_like(proj_norm), atol=1.0e-6)

    direct = extract_mode(phi_t, sel)
    assert np.allclose(direct, ts * spatial[1])

    try:
        extract_mode_time_series(phi_t, sel, method="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mode method should raise ValueError")


def test_extract_mode_svd_fallback_nan():
    """SVD mode extraction should fall back when NaNs are present."""
    t = np.linspace(0.0, 1.0, 16)
    gamma = 0.1
    omega = 0.2
    ts = np.exp((gamma - 1j * omega) * t)
    data = ts[:, None] * np.array([1.0, np.nan])[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    svd_series = extract_mode_time_series(phi_t, sel, method="svd")
    proj_series = extract_mode_time_series(phi_t, sel, method="project")
    assert np.allclose(svd_series, proj_series, equal_nan=True)


def test_extract_mode_svd_fallback_linalg(monkeypatch):
    """SVD mode extraction should fall back on LinAlgError."""
    t = np.linspace(0.0, 1.0, 16)
    gamma = 0.1
    omega = 0.2
    ts = np.exp((gamma - 1j * omega) * t)
    data = ts[:, None] * np.linspace(1.0, 2.0, 4)[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    def _bad_svd(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(np.linalg, "svd", _bad_svd)
    svd_series = extract_mode_time_series(phi_t, sel, method="svd")
    proj_series = extract_mode_time_series(phi_t, sel, method="project")
    assert np.allclose(svd_series, proj_series)


def test_select_fit_window_and_auto_fit():
    """Auto window should favor the clean exponential region."""
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.12
    omega = 0.3
    signal = np.exp((gamma - 1j * omega) * t)
    signal = signal.copy()
    signal[:40] *= 1.0 + 0.2 * np.sin(5.0 * t[:40])

    tmin, tmax = select_fit_window(t, signal, window_fraction=0.3, min_points=20)
    assert tmax > tmin
    assert tmin >= t[20]

    tmin2, tmax2 = select_fit_window(
        t,
        signal,
        window_fraction=0.3,
        min_points=20,
        start_fraction=0.5,
        growth_weight=1.0,
        require_positive=True,
    )
    assert tmax2 > tmin2
    assert tmin2 >= t[int(0.5 * t.shape[0])]

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

    decaying = np.exp((-0.2 - 1j * omega) * t)
    tmin3, tmax3 = select_fit_window(
        t,
        decaying,
        window_fraction=0.3,
        min_points=20,
        start_fraction=0.1,
        growth_weight=0.5,
        require_positive=True,
    )
    assert tmax3 > tmin3

    g_fit2, w_fit2, _tmin2, _tmax2 = fit_growth_rate_auto(t, signal, tmin=2.0)
    assert np.isfinite(g_fit2)
    assert np.isfinite(w_fit2)

    try:
        select_fit_window(t, signal, start_fraction=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid start_fraction should raise ValueError")

    try:
        select_fit_window(t, signal, growth_weight=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative growth_weight should raise ValueError")


def test_extract_eigenfunction_svd_and_snapshot():
    """Eigenfunction extraction should recover the spatial mode."""
    t = np.linspace(0.0, 1.0, 64)
    gamma = 0.2
    omega = 0.4
    ts = np.exp((gamma - 1j * omega) * t)
    mode = np.array([1.0, 2.0, 3.0, 4.0])
    data = ts[:, None] * mode[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    svd_mode = extract_eigenfunction(phi_t, t, sel, method="svd")
    snapshot_mode = extract_eigenfunction(phi_t, t, sel, method="snapshot")
    assert np.allclose(svd_mode / svd_mode[0], mode / mode[0])
    assert np.allclose(snapshot_mode / snapshot_mode[0], mode / mode[0])


def test_extract_eigenfunction_invalid():
    """Eigenfunction extraction should validate inputs."""
    t = np.linspace(0.0, 1.0, 8)
    phi_t = np.zeros((8, 1, 1, 4))
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t[None, ...], t, sel)
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t[None, :], sel)
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t[:-1], sel)
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t, sel, method="bad")
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t, sel, tmin=2.0, tmax=3.0)
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t, sel, z=np.zeros(3))


def test_extract_eigenfunction_z_normalization():
    """Eigenfunction should normalize to theta=0 when z is provided."""
    t = np.linspace(0.0, 1.0, 8)
    z = np.array([-1.0, 0.0, 1.0, 2.0])
    mode = np.array([2.0, 4.0, 6.0, 8.0])
    phi_t = (np.ones((t.size, 1, 1, mode.size)) * mode[None, None, None, :])
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    out = extract_eigenfunction(phi_t, t, sel, z=z, method="snapshot")
    assert np.isclose(out[1], 1.0)


def test_extract_eigenfunction_z_zero_fallback():
    """If theta=0 value is zero, normalization should fall back to max."""
    t = np.linspace(0.0, 1.0, 8)
    z = np.array([-1.0, 0.0, 1.0, 2.0])
    mode = np.array([1.0, 0.0, 2.0, 3.0])
    phi_t = (np.ones((t.size, 1, 1, mode.size)) * mode[None, None, None, :])
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    out = extract_eigenfunction(phi_t, t, sel, z=z, method="snapshot")
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_extract_eigenfunction_nan_fallback():
    """SVD eigenfunction extraction should fall back on NaNs."""
    t = np.linspace(0.0, 1.0, 16)
    ts = np.exp((0.1 - 1j * 0.2) * t)
    data = ts[:, None] * np.array([1.0, np.nan])[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    svd_mode = extract_eigenfunction(phi_t, t, sel, method="svd")
    snap_mode = extract_eigenfunction(phi_t, t, sel, method="snapshot")
    assert np.allclose(svd_mode, snap_mode, equal_nan=True)


def test_extract_eigenfunction_linalg_fallback(monkeypatch):
    """SVD eigenfunction extraction should fall back on LinAlgError."""
    t = np.linspace(0.0, 1.0, 16)
    ts = np.exp((0.1 - 1j * 0.2) * t)
    data = ts[:, None] * np.linspace(1.0, 2.0, 4)[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    def _bad_svd(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(np.linalg, "svd", _bad_svd)
    svd_mode = extract_eigenfunction(phi_t, t, sel, method="svd")
    snap_mode = extract_eigenfunction(phi_t, t, sel, method="snapshot")
    assert np.allclose(svd_mode, snap_mode)


def test_extract_eigenfunction_zero_signal():
    """Zero signals should return a finite eigenfunction."""
    t = np.linspace(0.0, 1.0, 8)
    phi_t = np.zeros((8, 1, 1, 4))
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    mode = extract_eigenfunction(phi_t, t, sel, method="svd")
    assert np.all(np.isfinite(mode))


def test_gx_growth_rate_from_omega_series():
    """GX omega-series averaging should select the requested (ky, kx) branch."""

    gamma_t = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.3, 0.4], [0.5, 0.6]],
            [[0.4, 0.5], [0.6, 0.7]],
        ],
        dtype=float,
    )
    omega_t = -2.0 * gamma_t
    sel = ModeSelection(ky_index=1, kx_index=0, z_index=0)

    g, w, gs, ws = gx_growth_rate_from_omega_series(gamma_t, omega_t, sel, navg_fraction=0.5)
    assert np.allclose(gs, np.array([0.3, 0.4, 0.5, 0.6]))
    assert np.allclose(ws, np.array([-0.6, -0.8, -1.0, -1.2]))
    assert np.isclose(g, np.mean([0.5, 0.6]))
    assert np.isclose(w, np.mean([-1.0, -1.2]))

    g_last, w_last, _gs, _ws = gx_growth_rate_from_omega_series(
        gamma_t, omega_t, sel, use_last=True
    )
    assert np.isclose(g_last, 0.6)
    assert np.isclose(w_last, -1.2)
