"""Tests for vmec_jax equilibrium inventory tooling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from netCDF4 import Dataset


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_vmec_jax_equilibrium_inventory.py"
    spec = importlib.util.spec_from_file_location("plot_vmec_jax_equilibrium_inventory", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_wout(path: Path, *, nfp: int, ntor: int, aspect: float, iota_edge: float) -> None:
    with Dataset(path, "w") as ds:
        ds.createDimension("radius", 3)
        ds.createVariable("nfp", "i4").assignValue(nfp)
        ds.createVariable("ns", "i4").assignValue(3)
        ds.createVariable("mpol", "i4").assignValue(4)
        ds.createVariable("ntor", "i4").assignValue(ntor)
        ds.createVariable("aspect", "f8").assignValue(aspect)
        ds.createVariable("Aminor_p", "f8").assignValue(0.3)
        ds.createVariable("Rmajor_p", "f8").assignValue(1.2)
        ds.createVariable("volume_p", "f8").assignValue(2.0)
        ds.createVariable("betatotal", "f8").assignValue(0.01)
        iota = ds.createVariable("iotaf", "f8", ("radius",))
        iota[:] = [0.4, 0.5, iota_edge]
        pres = ds.createVariable("presf", "f8", ("radius",))
        pres[:] = [1.0, 0.5, 0.0]


def test_vmec_jax_inventory_report_and_figure_are_replayable(tmp_path: Path) -> None:
    mod = _load_tool_module()
    _write_wout(tmp_path / "wout_circular_tokamak.nc", nfp=1, ntor=0, aspect=3.0, iota_edge=0.25)
    _write_wout(tmp_path / "wout_nfp4_QH_warm_start.nc", nfp=4, ntor=2, aspect=7.0, iota_edge=-1.1)

    report = mod.build_inventory(tmp_path)
    paths = mod.write_inventory_figure(report, out=tmp_path / "inventory.png")

    assert report["kind"] == "vmec_jax_equilibrium_inventory"
    assert report["claim_level"] == "equilibrium_selection_not_transport_validation"
    assert report["n_equilibria"] == 2
    assert report["family_counts"]["axisymmetric"] == 1
    assert "wout_nfp4_QH_warm_start.nc" in report["recommended_next_linear_portfolio"]
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["rows"][0]["validation_role"].startswith("external_vmec_fixture")
