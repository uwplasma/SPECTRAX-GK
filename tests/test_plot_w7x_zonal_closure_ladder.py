from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_w7x_zonal_closure_ladder.py"
    spec = importlib.util.spec_from_file_location("plot_w7x_zonal_closure_ladder", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_reference(path: Path, *, kx: float = 0.07) -> None:
    t = np.linspace(0.0, 20.0, 21)
    rows = []
    for code, offset in (("stella", -0.01), ("GENE", 0.01)):
        for time_value in t:
            rows.append(
                {
                    "kx_rhoi": kx,
                    "code": code,
                    "t_vti_over_a": time_value,
                    "response": 0.2 + np.exp(-0.2 * time_value) + offset,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_output(path: Path, *, kx: float = 0.07, offset: float = 0.02) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 20.0, 21)
    kx_grid = np.array([-kx, 0.0, kx])
    response = 0.2 + np.exp(-0.2 * t) + offset
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", t.size)
        ds.createDimension("kx", kx_grid.size)
        ds.createDimension("ri", 2)
        ds.createDimension("s", 1)
        ds.createDimension("m", 8)
        ds.createDimension("l", 4)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        grids.createVariable("kx", "f8", ("kx",))[:] = kx_grid
        phi = np.zeros((t.size, kx_grid.size, 2), dtype=float)
        phi[:, 2, 0] = response / response[0]
        diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = phi
        wg = np.ones((t.size, 1, 8, 4), dtype=float)
        wg[:, 0, -2:, :] *= 3.0
        diag.createVariable("Wg_lmst", "f8", ("time", "s", "m", "l"))[:] = wg


def test_w7x_zonal_closure_ladder_builds_rows_and_main(tmp_path: Path) -> None:
    mod = _load_tool_module()
    reference = tmp_path / "reference.csv"
    out_nc = tmp_path / "run" / "w7x_test4_kx070.out.nc"
    _write_reference(reference)
    _write_output(out_nc)

    reference_t, reference_y = mod.load_reference_trace(reference, 0.07)
    rows, traces = mod.build_rows(
        [("synthetic", "paper", out_nc)],
        reference_t=reference_t,
        reference_y=reference_y,
        kx=0.07,
        t_compare=20.0,
        tail_fraction=0.3,
    )

    assert len(rows) == 1
    assert rows[0]["label"] == "synthetic"
    assert rows[0]["mean_abs_error"] < 1.0
    assert rows[0]["hermite_tail_last"] > 0.0
    assert "synthetic" in traces

    out_png = tmp_path / "closure.png"
    rc = mod.main(
        [
            "--reference-traces",
            str(reference),
            "--run",
            "synthetic",
            "paper",
            str(out_nc),
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False
    assert payload["rows"][0]["family"] == "paper"
