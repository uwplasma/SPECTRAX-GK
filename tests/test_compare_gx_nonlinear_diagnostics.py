"""Tests for GX vs SPECTRAX nonlinear diagnostics comparison tool."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_minimal_gx_nc(path: Path, ntime: int = 5) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    with Dataset(path, "w") as root:
        root.createDimension("time", ntime)
        root.createDimension("species", 1)
        grids = root.createGroup("Grids")
        diags = root.createGroup("Diagnostics")

        tvar = grids.createVariable("time", "f8", ("time",))
        tvar[:] = np.linspace(0.0, 1.0, ntime)

        for name in [
            "Phi2_t",
            "Wg_st",
            "Wphi_st",
            "Wapar_st",
            "HeatFlux_st",
            "ParticleFlux_st",
        ]:
            var = diags.createVariable(name, "f8", ("time", "species"))
            var[:, :] = np.linspace(0.1, 0.2, ntime)[:, None]


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
            "--out",
            str(out_path),
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
