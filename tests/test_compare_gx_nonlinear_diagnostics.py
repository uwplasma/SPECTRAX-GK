"""Tests for GX vs SPECTRAX nonlinear diagnostics comparison tool."""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_minimal_gx_nc(path: Path, ntime: int = 5) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    with Dataset(path, "w") as root:
        root.createDimension("time", ntime)
        root.createDimension("species", 2)
        grids = root.createGroup("Grids")
        diags = root.createGroup("Diagnostics")

        tvar = grids.createVariable("time", "f8", ("time",))
        tvar[:] = np.linspace(0.0, 1.0, ntime)

        phi2 = diags.createVariable("Phi2_t", "f8", ("time",))
        phi2[:] = np.linspace(0.1, 0.2, ntime)

        for name in ["Wg_st", "Wphi_st", "HeatFlux_st", "ParticleFlux_st"]:
            var = diags.createVariable(name, "f8", ("time", "species"))
            series = np.linspace(0.1, 0.2, ntime)[:, None]
            var[:, :] = np.concatenate([series, 2.0 * series], axis=1)

        wapar = diags.createVariable("Wapar_st", "f8", ("time", "species"))
        wapar[:, :] = np.repeat(np.linspace(0.3, 0.4, ntime)[:, None], 2, axis=1)


def _write_minimal_spectrax_nc(path: Path, ntime: int = 5) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    with Dataset(path, "w") as root:
        root.createDimension("time", ntime)
        root.createDimension("species", 2)
        grids = root.createGroup("Grids")
        diags = root.createGroup("Diagnostics")

        tvar = grids.createVariable("time", "f8", ("time",))
        tvar[:] = np.linspace(0.0, 1.0, ntime)

        phi2 = diags.createVariable("Phi2_t", "f8", ("time",))
        phi2[:] = np.linspace(0.2, 0.3, ntime)

        for name in ["Wg_st", "Wphi_st", "HeatFlux_st", "ParticleFlux_st"]:
            var = diags.createVariable(name, "f8", ("time", "species"))
            series = np.linspace(0.2, 0.3, ntime)[:, None]
            var[:, :] = np.concatenate([series, 2.0 * series], axis=1)

        wapar = diags.createVariable("Wapar_st", "f8", ("time", "species"))
        wapar[:, :] = np.repeat(np.linspace(0.4, 0.5, ntime)[:, None], 2, axis=1)


def _write_minimal_spectrax_csv(path: Path, ntime: int = 5) -> None:
    t = np.linspace(0.0, 1.0, ntime)
    data = np.column_stack(
        [
            t,
            np.zeros_like(t),  # gamma
            np.zeros_like(t),  # omega
            np.linspace(0.1, 0.2, ntime),  # Wg
            np.linspace(0.2, 0.3, ntime),  # Wphi
            np.linspace(0.3, 0.4, ntime),  # Wapar
            np.linspace(0.6, 0.9, ntime),  # energy
            np.linspace(0.01, 0.02, ntime),  # heat flux
            np.linspace(0.03, 0.04, ntime),  # particle flux
        ]
    )
    header = "t,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def test_compare_gx_nonlinear_diagnostics_plot(tmp_path: Path) -> None:
    pytest.importorskip("netCDF4")
    os.environ.setdefault("MPLBACKEND", "Agg")

    gx_path = tmp_path / "gx.out.nc"
    sp_path = tmp_path / "spectrax.csv"
    out_path = tmp_path / "diag_compare.png"
    summary_path = tmp_path / "diag_compare.summary.json"

    _write_minimal_gx_nc(gx_path)
    _write_minimal_spectrax_csv(sp_path)

    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear_diagnostics as mod

        argv = [
            "compare_gx_nonlinear_diagnostics.py",
            "--gx",
            str(gx_path),
            "--spectrax",
            str(sp_path),
            "--tmin",
            "0.25",
            "--tmax",
            "1.0",
            "--out",
            str(out_path),
            "--summary-json",
            str(summary_path),
            "--summary-case",
            "cyclone_nonlinear_window",
            "--summary-source",
            "minimal GX fixture",
            "--gate-mean-rel",
            "2.0",
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            assert mod.main() == 0
        finally:
            sys.argv = old_argv
    finally:
        sys.path.remove(str(tools_dir))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gate_mean_rel"] == 2.0
    assert summary["case"] == "cyclone_nonlinear_window"
    assert summary["source"] == "minimal GX fixture"
    assert summary["tmin"] == 0.25
    assert summary["tmax"] == 1.0
    assert summary["gate_report"]["case"] == "cyclone_nonlinear_window"
    assert summary["gate_report"]["source"] == "minimal GX fixture"
    assert {row["metric"] for row in summary["summary"]} >= {"Wg", "Wphi", "HeatFlux"}
    assert isinstance(summary["gate_passed"], bool)
    assert "Infinity" not in summary_path.read_text(encoding="utf-8")
    assert "NaN" not in summary_path.read_text(encoding="utf-8")


def test_compare_gx_nonlinear_diagnostics_uses_single_species_wapar(tmp_path: Path) -> None:
    pytest.importorskip("netCDF4")

    gx_path = tmp_path / "gx.out.nc"
    _write_minimal_gx_nc(gx_path)

    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear_diagnostics as mod

        loaded = mod._load_gx_diag(gx_path)
    finally:
        sys.path.remove(str(tools_dir))

    t = np.linspace(0.0, 1.0, 5)
    assert np.allclose(loaded["Wg"], 3.0 * np.linspace(0.1, 0.2, 5))
    assert np.allclose(loaded["Wphi"], 3.0 * np.linspace(0.1, 0.2, 5))
    assert np.allclose(loaded["heat_flux"], 3.0 * np.linspace(0.1, 0.2, 5))
    assert np.allclose(loaded["particle_flux"], 3.0 * np.linspace(0.1, 0.2, 5))
    assert np.allclose(loaded["Wapar"], np.linspace(0.3, 0.4, 5))
    assert np.allclose(loaded["t"], t)


def test_compare_gx_nonlinear_diagnostics_loads_spectrax_out_nc(tmp_path: Path) -> None:
    pytest.importorskip("netCDF4")

    spectrax_path = tmp_path / "spectrax.out.nc"
    _write_minimal_spectrax_nc(spectrax_path)

    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear_diagnostics as mod

        loaded = mod._load_spectrax(spectrax_path)
    finally:
        sys.path.remove(str(tools_dir))

    t = np.linspace(0.0, 1.0, 5)
    assert np.allclose(loaded["t"], t)
    assert np.allclose(loaded["phi2"], np.linspace(0.2, 0.3, 5))
    assert np.allclose(loaded["Wg"], 3.0 * np.linspace(0.2, 0.3, 5))
    assert np.allclose(loaded["Wphi"], 3.0 * np.linspace(0.2, 0.3, 5))
    assert np.allclose(loaded["heat_flux"], 3.0 * np.linspace(0.2, 0.3, 5))
    assert np.allclose(loaded["particle_flux"], 3.0 * np.linspace(0.2, 0.3, 5))
    assert np.allclose(loaded["Wapar"], np.linspace(0.4, 0.5, 5))


def test_compare_gx_nonlinear_diagnostics_interp_summary() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear_diagnostics as mod

        mean_rel, max_rel, final_rel = mod._interp_summary(
            np.array([0.0, 1.0, 2.0]),
            np.array([2.0, 4.0, 6.0]),
            np.array([0.0, 2.0]),
            np.array([1.0, 3.0]),
        )
    finally:
        sys.path.remove(str(tools_dir))

    assert np.isclose(mean_rel, 1.0)
    assert np.isclose(max_rel, 1.0)
    assert np.isclose(final_rel, 1.0)


def test_compare_gx_nonlinear_diagnostics_apply_time_window() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear_diagnostics as mod

        series = {
            "t": np.array([0.0, 1.0, 2.0, 3.0]),
            "Wg": np.array([10.0, 11.0, 12.0, 13.0]),
        }
        windowed = mod._apply_time_window(series, tmin=1.0, tmax=2.0)
    finally:
        sys.path.remove(str(tools_dir))

    assert np.allclose(windowed["t"], [1.0, 2.0])
    assert np.allclose(windowed["Wg"], [11.0, 12.0])
