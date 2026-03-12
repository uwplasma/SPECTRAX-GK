"""Tests for legacy GX grouped-output readers."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from inspect_gx_legacy_cetg import build_parser
from spectraxgk.gx_legacy_output import (
    expand_gx_legacy_positive_ky_state,
    load_gx_legacy_cetg_output,
    load_gx_legacy_cetg_restart,
)


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


def test_load_gx_legacy_cetg_output_reconstructs_totals_from_spectra(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "cetg_fill.nc"
    fill = 9.969209968386869e36
    with Dataset(path, "w") as root:
        root.createDimension("time", 2)
        root.createDimension("ky", 2)
        root.createDimension("kx", 3)
        root.createDimension("kz", 4)
        root.createDimension("x", 4)
        root.createDimension("y", 4)
        root.createDimension("s", 1)
        root.createVariable("time", "f8", ("time",))[:] = np.array([0.0, 1.0])
        root.createVariable("ky", "f8", ("ky",))[:] = np.array([0.1, 0.2])
        root.createVariable("kx", "f8", ("kx",))[:] = np.array([-0.1, 0.0, 0.1])
        root.createVariable("kz", "f8", ("kz",))[:] = np.linspace(-1.0, 1.0, 4)
        root.createVariable("x", "f8", ("x",))[:] = np.linspace(0.0, 1.0, 4)
        root.createVariable("y", "f8", ("y",))[:] = np.linspace(0.0, 1.0, 4)
        spectra = root.createGroup("Spectra")
        fluxes = root.createGroup("Fluxes")
        spectra.createVariable("W", "f8", ("time",))[:] = np.array([fill, fill])
        spectra.createVariable("Phi2t", "f8", ("time",))[:] = np.array([fill, fill])
        spectra.createVariable("Wkxst", "f8", ("time", "s", "kx"))[:] = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
        spectra.createVariable("Phi2kxt", "f8", ("time", "kx"))[:] = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        fluxes.createVariable("qflux", "f8", ("time", "s"))[:] = np.array([[1.0], [2.0]])
        fluxes.createVariable("pflux", "f8", ("time", "s"))[:] = np.array([[fill], [fill]])

    out = load_gx_legacy_cetg_output(path)

    assert out.W.tolist() == pytest.approx([6.0, 15.0])
    assert out.Phi2.tolist() == pytest.approx([6.0, 3.0])
    assert np.allclose(out.pflux, 0.0)


def test_inspect_gx_legacy_cetg_parser_accepts_json_flag() -> None:
    args = build_parser().parse_args(["/tmp/cetg_smoke.nc", "--json"])
    assert args.gx_nc == Path("/tmp/cetg_smoke.nc")
    assert args.json is True


def test_load_gx_legacy_cetg_restart_maps_compressed_kx_layout(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "cetg.restart.nc"
    with Dataset(path, "w") as root:
        root.createDimension("Nspecies", 1)
        root.createDimension("Nm", 1)
        root.createDimension("Nl", 2)
        root.createDimension("Nz", 2)
        root.createDimension("Nkx", 3)
        root.createDimension("Nky", 2)
        root.createDimension("ri", 2)
        root.createVariable("time", "f8", ())[:] = 1.25
        G = root.createVariable("G", "f8", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri"))
        raw = np.zeros((1, 1, 2, 2, 3, 2, 2), dtype=float)
        raw[0, 0, 0, 0, 0, 0] = (10.0, 1.0)
        raw[0, 0, 0, 0, 1, 1] = (20.0, 2.0)
        raw[0, 0, 1, 1, 2, 1] = (30.0, 3.0)
        G[:] = raw

    out = load_gx_legacy_cetg_restart(path, nx_full=4, ny_full=4)

    assert out.time == pytest.approx(1.25)
    assert out.state_active.shape == (1, 2, 1, 2, 3, 2)
    assert out.state_active[0, 0, 0, 0, 0, 0] == pytest.approx(10.0 + 1.0j)
    assert out.state_active[0, 0, 0, 1, 1, 0] == pytest.approx(20.0 + 2.0j)
    assert out.state_active[0, 1, 0, 1, 2, 1] == pytest.approx(30.0 + 3.0j)
    assert out.state_positive_ky.shape == (1, 2, 1, 3, 4, 2)
    assert out.state_positive_ky[0, 0, 0, 0, 0, 0] == pytest.approx(10.0 + 1.0j)
    assert out.state_positive_ky[0, 0, 0, 1, 1, 0] == pytest.approx(20.0 + 2.0j)
    assert out.state_positive_ky[0, 1, 0, 1, 3, 1] == pytest.approx(30.0 + 3.0j)
    assert out.state_positive_ky[0, 0, 0, 2, 0, 0] == pytest.approx(0.0)


def test_expand_gx_legacy_positive_ky_state_builds_full_hermitian_ky_grid() -> None:
    pos = np.zeros((1, 2, 1, 3, 4, 1), dtype=np.complex64)
    pos[0, 0, 0, 1, 1, 0] = 1.0 + 2.0j
    full = expand_gx_legacy_positive_ky_state(pos, ny_full=4)

    assert full.shape == (1, 2, 1, 4, 4, 1)
    assert full[0, 0, 0, 1, 1, 0] == pytest.approx(1.0 + 2.0j)
    assert full[0, 0, 0, 3, 3, 0] == pytest.approx(1.0 - 2.0j)
