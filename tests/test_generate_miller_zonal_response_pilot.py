from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np


def _load_tool_module():
    path = Path("/Users/rogeriojorge/local/SPECTRAX-GK/tools/generate_miller_zonal_response_pilot.py")
    spec = importlib.util.spec_from_file_location("generate_miller_zonal_response_pilot", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_miller_zonal_response_pilot_main(tmp_path, monkeypatch) -> None:
    mod = _load_tool_module()

    config = tmp_path / "pilot.toml"
    config.write_text(
        """
[grid]
Nx = 4
Ny = 6
Nz = 8
Lx = 6.28
Ly = 6.28
boundary = "periodic"

[time]
t_max = 1.0
dt = 0.1
method = "rk2"
diagnostics = true
sample_stride = 1

[geometry]
model = "miller"
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1.0e-4
init_single = true

[physics]
adiabatic_electrons = true
nonlinear = false

[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[run]
ky = 0.0
kx = 0.1
Nl = 2
Nm = 2
dt = 0.1
steps = 10
sample_stride = 1
diagnostics = true
""".strip()
    )

    out_bundle = tmp_path / "pilot.out.nc"
    out_png = tmp_path / "pilot.png"

    def _fake_run(cfg, *, out, **kwargs):
        path = Path(out)
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("time", 6)
            ds.createDimension("kx", 3)
            ds.createDimension("ri", 2)
            grids = ds.createGroup("Grids")
            diag = ds.createGroup("Diagnostics")
            grids.createVariable("time", "f8", ("time",))[:] = np.linspace(0.0, 5.0, 6)
            grids.createVariable("kx", "f8", ("kx",))[:] = np.array([-0.1, 0.0, 0.1])
            raw = np.zeros((6, 3, 2), dtype=float)
            raw[:, 2, 0] = np.array([1.0, 0.6, 0.35, 0.2, 0.12, 0.1])
            raw[:, 2, 1] = np.array([0.2, 0.12, 0.08, 0.03, 0.02, 0.01])
            diag.createVariable("Phi_zonal_mode_kxt", "f8", ("time", "kx", "ri"))[:] = raw
        return object(), {"out": str(path)}

    monkeypatch.setattr(mod, "run_runtime_nonlinear_with_artifacts", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(mod.__file__),
            "--config",
            str(config),
            "--out-bundle",
            str(out_bundle),
            "--out-png",
            str(out_png),
        ],
    )

    assert mod.main() == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    assert out_png.with_suffix(".csv").exists()
    meta = json.loads(out_png.with_suffix(".json").read_text())
    assert meta["variable"] == "Phi_zonal_mode_kxt"
    assert meta["kx_selected"] == 0.1
    assert "density-seeded shaped-Miller pilot" in meta["notes"]
