"""Tests for the W7-X fluctuation-spectrum panel generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_w7x_fluctuation_spectrum_panel.py"
    spec = importlib.util.spec_from_file_location("plot_w7x_fluctuation_spectrum_panel", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_w7x_synthetic_output(path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    time = np.linspace(0.0, 30.0, 10)
    ky = np.array([0.0, 0.1, 0.2])
    kx = np.array([-0.2, 0.0, 0.2, 0.4])
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", time.size)
        root.createDimension("ky", ky.size)
        root.createDimension("kx", kx.size)
        root.createDimension("s", 1)
        root.createDimension("ri", 2)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = time
        grids.createVariable("ky", "f8", ("ky",))[:] = ky
        grids.createVariable("kx", "f8", ("kx",))[:] = kx
        diag = root.createGroup("Diagnostics")
        envelope = 1.0 + 0.2 * np.sin(0.5 * time)
        phi2 = np.outer(envelope, np.array([0.02, 1.0, 0.4]))
        diag.createVariable("Phi2_kyt", "f8", ("time", "ky"))[:] = phi2
        phi2_map = np.zeros((time.size, ky.size, kx.size))
        for t_idx, scale in enumerate(envelope):
            phi2_map[t_idx] = scale * np.outer(np.array([0.02, 1.0, 0.4]), np.array([0.1, 0.2, 1.0, 0.3]))
        diag.createVariable("Phi2_kxkyt", "f8", ("time", "ky", "kx"))[:] = phi2_map
        wphi = np.zeros((time.size, 1, ky.size))
        wphi[:, 0, :] = np.outer(1.0 + 0.1 * np.cos(0.4 * time), np.array([0.01, 0.5, 1.5]))
        diag.createVariable("Wphi_kyst", "f8", ("time", "s", "ky"))[:] = wphi
        heat = np.zeros((time.size, 1, ky.size))
        heat[:, 0, :] = np.outer(1.0 + 0.05 * np.sin(0.3 * time), np.array([0.0, 2.0, 0.7]))
        diag.createVariable("HeatFlux_kyst", "f8", ("time", "s", "ky"))[:] = heat
        zonal = np.zeros((time.size, kx.size, 2))
        zonal[:, 2, 0] = np.cos(0.6 * time)
        zonal[:, 2, 1] = np.sin(0.6 * time)
        diag.createVariable("Phi_zonal_mode_kxt", "f8", ("time", "kx", "ri"))[:] = zonal


def test_w7x_fluctuation_spectrum_report_and_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out_nc = tmp_path / "w7x.out.nc"
    _write_w7x_synthetic_output(out_nc)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"gate_passed": True, "gate_report": {"passed": True, "case": "W7-X"}}))

    report = mod.build_w7x_fluctuation_spectrum_report(
        nonlinear=out_nc,
        gate_summary=gate,
        time_min=3.0,
        time_max=28.0,
    )
    paths = mod.write_w7x_fluctuation_spectrum_artifacts(report, out=tmp_path / "panel.png")

    assert report["claim_level"] == "validated_nonlinear_simulation_spectrum_not_experimental_validation"
    assert report["gate_index_include"] is False
    assert report["source_gate_passed"] is True
    assert report["dominant_phi_ky"] == pytest.approx(0.1)
    assert report["dominant_heat_flux_ky"] == pytest.approx(0.1)
    assert report["dominant_zonal_kx"] == pytest.approx(0.2)
    assert np.sum(report["phi2_ky_distribution"]) == pytest.approx(1.0)
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()


def test_w7x_fluctuation_spectrum_rejects_failed_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out_nc = tmp_path / "w7x.out.nc"
    _write_w7x_synthetic_output(out_nc)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"gate_passed": False, "gate_report": {"passed": False, "case": "W7-X"}}))

    with pytest.raises(ValueError, match="did not pass"):
        mod.build_w7x_fluctuation_spectrum_report(nonlinear=out_nc, gate_summary=gate)
