from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk.geometry.miller import derm, dermv, generate_miller_eik, nperiod_data_extend, reflect_n_append
from spectraxgk.geometry.vmec import generate_vmec_eik


class _FakeDataset:
    def __init__(self, path, mode):
        self.path = Path(path)
        self.mode = mode
        self.dimensions: dict[str, int] = {}
        self.closed = False

    def createDimension(self, name, size):
        self.dimensions[name] = size

    def close(self):
        self.closed = True


def test_standalone_miller_helpers_and_generator(monkeypatch, tmp_path) -> None:
    arr = np.array([1.0, 2.0, 4.0, 7.0])
    np.testing.assert_allclose(derm(arr, "l", "e"), [[0.0, 3.0, 5.0, 0.0]])
    out = dermv(arr, np.array([0.0, 1.0, 2.0, 4.0]), "l", "o")
    assert out.shape == (1, 4)
    np.testing.assert_allclose(nperiod_data_extend(np.array([0.0, 1.0, 2.0]), 2).shape, (7,))
    np.testing.assert_allclose(reflect_n_append(np.array([0.0, 1.0, 2.0]), "o"), [-2.0, -1.0, 0.0, 1.0, 2.0])

    datasets: list[_FakeDataset] = []
    monkeypatch.setitem(sys.modules, "netCDF4", SimpleNamespace(Dataset=lambda path, mode: datasets.append(_FakeDataset(path, mode)) or datasets[-1]))
    out_path = tmp_path / "miller.nc"
    generate_miller_eik(
        {
            "Dimensions": {"ntheta": 16, "nperiod": 1},
            "Geometry": {"rhoc": 0.5, "q": 1.4, "s_hat": 0.8, "R0": 3.0},
        },
        out_path,
    )
    assert datasets[0].path == out_path
    assert datasets[0].dimensions["z"] == 16
    assert datasets[0].closed is True


def test_standalone_vmec_generator(monkeypatch, tmp_path) -> None:
    datasets: list[_FakeDataset] = []
    monkeypatch.setitem(sys.modules, "netCDF4", SimpleNamespace(Dataset=lambda path, mode: datasets.append(_FakeDataset(path, mode)) or datasets[-1]))
    monkeypatch.setitem(sys.modules, "booz_xform_jax", SimpleNamespace(Booz_xform=object))
    out_path = tmp_path / "vmec.nc"
    generate_vmec_eik({"Geometry": {}, "Dimensions": {}}, out_path)
    assert datasets[0].path == out_path
    assert datasets[0].dimensions["z"] == 16
    assert datasets[0].closed is True


def test_standalone_vmec_generator_falls_back_to_booz_xform(monkeypatch, tmp_path) -> None:
    datasets: list[_FakeDataset] = []
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "netCDF4":
            return SimpleNamespace(Dataset=lambda path, mode: datasets.append(_FakeDataset(path, mode)) or datasets[-1])
        if name == "booz_xform_jax":
            raise ImportError("booz_xform_jax unavailable")
        if name == "booz_xform":
            return SimpleNamespace(Boozer=object)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    out_path = tmp_path / "vmec_fallback.nc"
    generate_vmec_eik({"Geometry": {}, "Dimensions": {}}, out_path)

    assert datasets[0].path == out_path
    assert datasets[0].dimensions["z"] == 16
    assert datasets[0].closed is True


def test_standalone_vmec_generator_requires_backend(monkeypatch, tmp_path) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "netCDF4":
            return SimpleNamespace(Dataset=lambda path, mode: _FakeDataset(path, mode))
        if name in {"booz_xform_jax", "booz_xform"}:
            raise ImportError(f"{name} unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Either booz_xform or booz_xform_jax is required"):
        generate_vmec_eik({"Geometry": {}, "Dimensions": {}}, tmp_path / "vmec_missing_backend.nc")


def test_standalone_vmec_generator_requires_netcdf4(monkeypatch, tmp_path) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "netCDF4":
            raise ImportError("netCDF4 unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="netCDF4 is required"):
        generate_vmec_eik({"Geometry": {}, "Dimensions": {}}, tmp_path / "vmec_missing_netcdf.nc")
