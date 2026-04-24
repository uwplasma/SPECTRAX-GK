from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import netCDF4 as nc
import numpy as np


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_w7x_zonal_moment_tail_audit.py"
    spec = importlib.util.spec_from_file_location("plot_w7x_zonal_moment_tail_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_output(path: Path, *, kx_target: float = 0.07, nm: int = 8, nl: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 20.0, 11)
    kx = np.array([-kx_target, 0.0, kx_target])
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", t.size)
        ds.createDimension("kx", kx.size)
        ds.createDimension("ri", 2)
        ds.createDimension("s", 1)
        ds.createDimension("m", nm)
        ds.createDimension("l", nl)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        grids.createVariable("kx", "f8", ("kx",))[:] = kx
        wg = np.ones((t.size, 1, nm, nl), dtype=float)
        wg[:, 0, -2:, :] *= np.linspace(1.0, 5.0, t.size)[:, None, None]
        wg[:, 0, :, -1:] *= 2.0
        diag.createVariable("Wg_lmst", "f8", ("time", "s", "m", "l"))[:] = wg
        phi = np.zeros((t.size, kx.size, 2), dtype=float)
        phi[:, 2, 0] = 1.0 + 0.1 * np.sin(t)
        diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = phi


def test_w7x_zonal_moment_tail_loads_rows_and_main(tmp_path: Path) -> None:
    mod = _load_tool_module()
    run_dir = tmp_path / "run"
    _write_output(run_dir / "w7x_test4_kx070.out.nc")

    rows, heatmap = mod.load_audit_rows(
        [("synthetic", run_dir)],
        kx_values=(0.07,),
        tail_fraction=0.3,
        hermite_tail_fraction=0.25,
        laguerre_tail_fraction=0.25,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "synthetic"
    assert row["kx_index"] == 2
    assert row["hermite_tail_last"] > 0.0
    assert row["laguerre_tail_last"] > 0.0
    assert heatmap is None

    out_png = tmp_path / "audit.png"
    rc = mod.main(
        [
            "--run",
            "synthetic",
            str(run_dir),
            "--kx-values",
            "0.07",
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
    assert payload["rows"][0]["Nm"] == 8
