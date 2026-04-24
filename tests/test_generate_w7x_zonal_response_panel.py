from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "generate_w7x_zonal_response_panel.py"
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
init_field = "phi"
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
    run_calls = []

    def _fake_run(cfg, *, out, kx_target, **kwargs):
        run_calls.append((float(kx_target), cfg.grid, dict(kwargs)))
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
            diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = raw
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
            "--dt",
            "0.2",
            "--steps",
            "80",
            "--sample-stride",
            "2",
            "--Nl",
            "6",
            "--Nm",
            "10",
            "--show-progress",
        ],
    )

    assert mod.main() == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    assert out_png.with_suffix(".csv").exists()
    meta = json.loads(out_png.with_suffix(".json").read_text())
    assert meta["initial_policy"] == "first_abs"
    assert meta["initial_normalization"] == "init_amp"
    assert meta["initial_level_override"] == 1.0e-6
    assert meta["damping_method"] == "branchwise_extrema"
    assert meta["frequency_method"] == "hilbert_phase"
    assert meta["validation_status"] == "open"
    assert len(meta["cases"]) == 4
    assert meta["literature_reference"]["test"] == 4
    assert meta["literature_reference"]["flux_tube"] == "bean"
    assert meta["literature_reference"]["observable"] == "unweighted line-averaged electrostatic potential"
    assert "slower stellarator-specific oscillation" in meta["notes"]
    assert "maximum initial potential amplitude" in meta["notes"]
    assert "manuscript-policy inference" in meta["notes"]
    assert "digitized-reference gate" in meta["notes"]
    assert meta["runtime"] == {
        "dt": 0.2,
        "steps": 80,
        "sample_stride": 2,
        "diagnostics": True,
        "show_progress": True,
        "expected_tmax": 16.0,
        "Nl": 6,
        "Nm": 10,
    }
    assert len(run_calls) == 4
    for kx_target, grid, kwargs in run_calls:
        assert grid.boundary == "periodic"
        assert grid.non_twist is True
        assert grid.jtwist is None
        assert np.isclose(grid.Lx, 2.0 * np.pi / kx_target)
        assert kwargs["dt"] == 0.2
        assert kwargs["steps"] == 80
        assert kwargs["sample_stride"] == 2
        assert kwargs["show_progress"] is True
        assert kwargs["Nl"] == 6
        assert kwargs["Nm"] == 10


def test_generate_w7x_zonal_response_formats_unresolved_damping() -> None:
    mod = _load_tool_module()

    assert mod._finite_or_none(float("nan")) is None
    assert mod._format_metric(None) == "not fitted"
    assert mod._format_metric(float("nan")) == "not fitted"
    assert mod._format_metric(1.23456) == "1.235"
