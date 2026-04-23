from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np


def _load_tool_module():
    path = Path("/Users/rogeriojorge/local/SPECTRAX-GK/tools/generate_w7x_zonal_response_panel.py")
    spec = importlib.util.spec_from_file_location("generate_w7x_zonal_response_panel", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_w7x_zonal_response_panel_main(tmp_path, monkeypatch) -> None:
    mod = _load_tool_module()

    config = tmp_path / "w7x_test4.toml"
    config.write_text(
        """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 0.0
fprim = 0.0
kinetic = true

[grid]
Nx = 6
Ny = 4
Nz = 32
Lx = 125.66370614359172
Ly = 62.8
boundary = "linked"
nperiod = 4

[time]
t_max = 10.0
dt = 0.1
method = "rk4"
sample_stride = 1
diagnostics = true
fixed_dt = true

[geometry]
model = "vmec"
vmec_file = "$W7X_VMEC_FILE"
torflux = 0.64
alpha = 0.0
R0 = 5.485

[init]
init_field = "density"
init_amp = 1.0e-6
gaussian_init = true
gaussian_width = 0.5
init_single = true

[physics]
adiabatic_electrons = true
nonlinear = false
collisions = false
hypercollisions = false

[run]
ky = 0.0
kx = 0.05
Nl = 4
Nm = 8
dt = 0.1
steps = 100
sample_stride = 1
diagnostics = true
""".strip()
    )

    out_dir = tmp_path / "w7x_out"
    out_png = tmp_path / "w7x_panel.png"

    def _fake_run(cfg, *, out, kx_target, **kwargs):
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0.0, 10.0, 41)
        signal = np.exp(-0.18 * t) * np.cos(1.35 * t) + 0.12
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("time", t.size)
            ds.createDimension("kx", 3)
            ds.createDimension("ri", 2)
            grids = ds.createGroup("Grids")
            diag = ds.createGroup("Diagnostics")
            grids.createVariable("time", "f8", ("time",))[:] = t
            grids.createVariable("kx", "f8", ("kx",))[:] = np.array([-float(kx_target), 0.0, float(kx_target)])
            raw = np.zeros((t.size, 3, 2), dtype=float)
            raw[:, 2, 0] = signal
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
            "--out-dir",
            str(out_dir),
            "--out-png",
            str(out_png),
        ],
    )

    assert mod.main() == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    assert out_png.with_suffix(".csv").exists()
    meta = json.loads(out_png.with_suffix(".json").read_text())
    assert meta["initial_policy"] == "first_abs"
    assert meta["damping_method"] == "branchwise_extrema"
    assert meta["frequency_method"] == "hilbert_phase"
    assert meta["validation_status"] == "open"
    assert len(meta["cases"]) == 4
    assert meta["literature_reference"]["test"] == 4
    assert meta["literature_reference"]["flux_tube"] == "bean"
    assert "slower stellarator-specific oscillation" in meta["notes"]
    assert "manuscript-policy inference" in meta["notes"]
