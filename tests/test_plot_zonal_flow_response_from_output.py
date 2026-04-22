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
    assert "zonal-energy proxy" in meta["notes"]
