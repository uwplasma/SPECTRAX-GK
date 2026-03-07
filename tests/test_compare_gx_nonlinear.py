"""Tests for the GX nonlinear comparison tool."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def test_compare_gx_nonlinear_loads_restart_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "restart.csv"
    t = np.linspace(5.0, 6.0, 4)
    data = np.column_stack(
        [
            t,
            np.linspace(0.1, 0.2, 4),   # gamma
            np.linspace(-0.3, -0.2, 4), # omega
            np.linspace(1.0, 2.0, 4),   # Wg
            np.linspace(3.0, 4.0, 4),   # Wphi
            np.linspace(5.0, 6.0, 4),   # Wapar
            np.linspace(7.0, 8.0, 4),   # heat flux
            np.linspace(9.0, 10.0, 4),  # particle flux
        ]
    )
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="t,gamma,omega,Wg,Wphi,Wapar,heat_flux,particle_flux",
        comments="",
    )

    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod

        loaded = mod._load_spectrax(csv_path)
    finally:
        sys.path.remove(str(tools_dir))

    assert np.allclose(loaded["t"], t)
    assert np.allclose(loaded["gamma"], data[:, 1])
    assert np.allclose(loaded["omega"], data[:, 2])
    assert np.allclose(loaded["Wg"], data[:, 3])
    assert np.allclose(loaded["Wphi"], data[:, 4])
    assert np.allclose(loaded["Wapar"], data[:, 5])
    assert np.allclose(loaded["heat"], data[:, 6])
    assert np.allclose(loaded["pflux"], data[:, 7])


def test_compare_gx_nonlinear_late_stats_handle_decorrelated_saturated_traces() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear as mod
    finally:
        sys.path.remove(str(tools_dir))

    t_sp = np.linspace(0.0, 400.0, 3201)
    t_gx = np.linspace(0.0, 400.0, 801)
    phase = 0.9
    sp = 20.0 + 3.0 * np.sin(0.35 * t_sp)
    gx = 20.0 + 3.0 * np.sin(0.35 * t_gx + phase)

    pointwise = mod._relative_error_window(t_sp, sp, mod._interp(t_sp, t_gx, gx), tmin=20.0)
    _, _, stats = mod._stats_relative_errors(t_sp, sp, t_gx, gx, tmin=20.0)

    assert pointwise > 0.05
    assert stats["rel_mean"] < 1.0e-3
    assert stats["rel_std"] < 1.0e-2
