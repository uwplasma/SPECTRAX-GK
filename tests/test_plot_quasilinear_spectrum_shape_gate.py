"""Tests for the quasilinear/nonlinear spectrum-shape gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_quasilinear_spectrum_shape_gate.py"
    spec = importlib.util.spec_from_file_location("plot_quasilinear_spectrum_shape_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_netcdf(path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", 3)
        root.createDimension("s", 1)
        root.createDimension("ky", 3)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = np.asarray([0.0, 1.0, 2.0])
        grids.createVariable("ky", "f8", ("ky",))[:] = np.asarray([0.1, 0.2, 0.3])
        diagnostics = root.createGroup("Diagnostics")
        values = np.asarray(
            [
                [[1.0, 2.0, 3.0]],
                [[2.0, 4.0, 6.0]],
                [[3.0, 6.0, 9.0]],
            ]
        )
        diagnostics.createVariable("HeatFlux_kyst", "f8", ("time", "s", "ky"))[:] = values


def test_quasilinear_spectrum_shape_gate_writes_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum = tmp_path / "ql.csv"
    np.savetxt(
        spectrum,
        np.asarray([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]]),
        delimiter=",",
        header="ky,heat_flux_weight_total",
        comments="",
    )
    nonlinear = tmp_path / "nl.nc"
    _write_netcdf(nonlinear)

    report = mod.build_spectrum_shape_report(
        spectrum_csv=spectrum,
        nonlinear_netcdf=nonlinear,
        time_min=1.0,
        tv_gate=1.0e-12,
        cosine_gate=1.0 - 1.0e-12,
    )
    assert report["passed"] is True
    assert report["time_samples"] == 2
    assert report["total_variation_distance"] == pytest.approx(0.0)
    assert report["cosine_similarity"] == pytest.approx(1.0)

    paths = mod.write_spectrum_shape_figure(report, out=tmp_path / "shape.png", title="shape")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["json"]).exists()


def test_quasilinear_spectrum_shape_gate_rejects_missing_column(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum = tmp_path / "bad.csv"
    np.savetxt(spectrum, np.asarray([[0.1, 1.0]]), delimiter=",", header="ky,other", comments="")
    nonlinear = tmp_path / "nl.nc"
    _write_netcdf(nonlinear)

    with pytest.raises(ValueError, match="heat_flux_weight_total"):
        mod.build_spectrum_shape_report(spectrum_csv=spectrum, nonlinear_netcdf=nonlinear)
