from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from netCDF4 import Dataset

from tools.build_vmec_boundary_transport_landscape import (
    _parse_float_list,
    _write_scan_inputs,
)
from tools.patch_vmec_jax_wout_metadata import patch_wout
from tools.write_vmec_boundary_perturbation_inputs import _parse_coefficient_spec


def _coefficient_value(text: str, name: str) -> float:
    match = re.search(rf"{re.escape(name)}\s*=\s*([+\-\d.Ee]+)", text)
    if match is None:
        raise AssertionError(f"missing {name}")
    return float(match.group(1))


def test_landscape_scan_inputs_patch_selected_boundary_coefficient(tmp_path: Path) -> None:
    baseline = tmp_path / "input.final"
    baseline.write_text(
        "\n".join(
            [
                "&INDATA",
                "  RBC(0,0) = 1.0000000000000000E+00",
                "  RBC(0,1) = 2.0000000000000000E-01",
                "  ZBS(0,1) = 1.0000000000000000E-01",
                "/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    base, rows = _write_scan_inputs(
        baseline_input=baseline,
        coefficient=_parse_coefficient_spec("RBC(0,1)"),
        fractions=(-0.1, 0.0, 0.1),
        out_dir=tmp_path / "scan",
    )

    assert base == 0.2
    assert [row["relative_fraction"] for row in rows] == [-0.1, 0.0, 0.1]
    patched = [Path(row["input_path"]).read_text(encoding="utf-8") for row in rows]
    assert _coefficient_value(patched[0], "RBC(0,1)") == 0.18000000000000002
    assert _coefficient_value(patched[1], "RBC(0,1)") == 0.2
    assert _coefficient_value(patched[2], "RBC(0,1)") == 0.22000000000000003
    assert all("ZBS(0,1) = 1.0000000000000000E-01" in text for text in patched)


def test_parse_float_list_rejects_empty_lists() -> None:
    assert _parse_float_list("0.45,0.64") == (0.45, 0.64)

    try:
        _parse_float_list(" , ")
    except Exception as exc:
        assert "expected at least one" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("empty float list was accepted")


def test_patch_vmec_jax_wout_metadata_fills_zero_scalars(tmp_path: Path) -> None:
    path = tmp_path / "wout_test.nc"
    with Dataset(path, "w") as ds:
        ds.createDimension("mn", 2)
        ds.createDimension("ns", 2)
        ds.createVariable("xm", "f8", ("mn",))[:] = np.asarray([0.0, 1.0])
        ds.createVariable("xn", "f8", ("mn",))[:] = np.asarray([0.0, 0.0])
        rmnc = ds.createVariable("rmnc", "f8", ("ns", "mn"))
        rmnc[:, :] = np.asarray([[1.0, 0.1], [1.0, 0.2]])
        for name in ("Aminor_p", "Rmajor_p", "aspect", "volume_p"):
            ds.createVariable(name, "f8").assignValue(0.0)

    report = patch_wout(path, ntheta=64, nphi=4)

    assert set(report["patched"]) == {"Aminor_p", "Rmajor_p", "aspect", "volume_p"}
    assert report["after"]["Aminor_p"] > 0.0
    assert report["after"]["Rmajor_p"] > 0.0
    assert report["after"]["aspect"] > 0.0
    with Dataset(path) as ds:
        assert float(ds.variables["Aminor_p"][:]) == report["after"]["Aminor_p"]
