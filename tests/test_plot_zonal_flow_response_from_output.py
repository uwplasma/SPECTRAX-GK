from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np


def _load_tool_module():
    path = Path("/Users/rogeriojorge/local/SPECTRAX-GK/tools/plot_zonal_flow_response_from_output.py")
    spec = importlib.util.spec_from_file_location("plot_zonal_flow_response_from_output", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_zonal_flow_response_from_output_main(tmp_path, monkeypatch) -> None:
    mod = _load_tool_module()

    data_path = tmp_path / "diag.out.nc"
    with nc.Dataset(data_path, "w") as ds:
        ds.createDimension("time", 5)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.linspace(0.0, 4.0, 5)
        diag.createVariable("Phi2_zonal_t", "f8", ("time",))[:] = np.array([1.0, 0.7, 0.55, 0.45, 0.4])

    out = tmp_path / "zf_from_output.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(mod.__file__),
            str(data_path),
            "--out",
            str(out),
        ],
    )

    assert mod.main() == 0
    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".csv").exists()
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["variable"] == "Phi2_zonal_t"
    assert meta["initial_policy"] == "window_abs_mean"
    assert "peak_fit_count" in meta
    assert "zonal-energy proxy" in meta["notes"]


def test_plot_zonal_flow_response_from_output_complex_mode_history(tmp_path, monkeypatch) -> None:
    mod = _load_tool_module()

    data_path = tmp_path / "diag.out.nc"
    with nc.Dataset(data_path, "w") as ds:
        ds.createDimension("time", 5)
        ds.createDimension("kx", 2)
        ds.createDimension("ri", 2)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.linspace(0.0, 4.0, 5)
        raw = np.zeros((5, 2, 2), dtype=float)
        raw[:, 1, 0] = np.array([0.0, -0.4, -0.2, 0.1, 0.05])
        raw[:, 1, 1] = np.array([1.0, 0.6, 0.3, -0.2, -0.1])
        diag.createVariable("Phi_zonal_mode_kxt", "f8", ("time", "kx", "ri"))[:] = raw

    out = tmp_path / "zf_signed.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(mod.__file__),
            str(data_path),
            "--var",
            "Phi_zonal_mode_kxt",
            "--kx-index",
            "1",
            "--align-phase",
            "--component",
            "real",
            "--out",
            str(out),
        ],
    )

    assert mod.main() == 0
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["variable"] == "Phi_zonal_mode_kxt"
    assert meta["initial_policy"] == "window_abs_mean"
    assert "peak_fit_count" in meta
