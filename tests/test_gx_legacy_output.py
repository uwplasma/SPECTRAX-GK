"""Tests for legacy GX grouped-output readers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from inspect_gx_legacy_cetg import build_parser
from spectraxgk.gx_legacy_output import load_gx_legacy_cetg_output


def test_load_gx_legacy_cetg_output_reads_grouped_contract(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "cetg_smoke.nc"
    with Dataset(path, "w") as root:
        root.createDimension("time", 3)
        root.createDimension("ky", 2)
        root.createDimension("kx", 4)
        root.createDimension("kz", 5)
        root.createDimension("x", 8)
        root.createDimension("y", 8)
        root.createDimension("s", 1)
        root.createVariable("time", "f8", ("time",))[:] = np.array([0.0, 0.5, 1.0])
        root.createVariable("ky", "f8", ("ky",))[:] = np.array([0.1, 0.2])
        root.createVariable("kx", "f8", ("kx",))[:] = np.array([-0.2, -0.1, 0.0, 0.1])
        root.createVariable("kz", "f8", ("kz",))[:] = np.linspace(-1.0, 1.0, 5)
        root.createVariable("x", "f8", ("x",))[:] = np.linspace(0.0, 1.0, 8)
        root.createVariable("y", "f8", ("y",))[:] = np.linspace(0.0, 1.0, 8)
        spectra = root.createGroup("Spectra")
        fluxes = root.createGroup("Fluxes")
        spectra.createVariable("W", "f8", ("time",))[:] = np.array([1.0, 2.0, 3.0])
        spectra.createVariable("Phi2t", "f8", ("time",))[:] = np.array([0.5, 0.6, 0.7])
        fluxes.createVariable("qflux", "f8", ("time", "s"))[:] = np.array([[1.0], [2.0], [4.0]])
        fluxes.createVariable("pflux", "f8", ("time", "s"))[:] = np.array([[0.1], [0.2], [0.4]])

    out = load_gx_legacy_cetg_output(path)
    assert out.time.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert out.ky.tolist() == pytest.approx([0.1, 0.2])
    assert out.kx.shape == (4,)
    assert out.kz.shape == (5,)
    assert out.W.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert out.Phi2.tolist() == pytest.approx([0.5, 0.6, 0.7])
    assert out.qflux.shape == (3, 1)
    assert out.pflux.shape == (3, 1)


def test_inspect_gx_legacy_cetg_parser_accepts_json_flag() -> None:
    args = build_parser().parse_args(["/tmp/cetg_smoke.nc", "--json"])
    assert args.gx_nc == Path("/tmp/cetg_smoke.nc")
    assert args.json is True
