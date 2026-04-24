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
        run_calls.append((float(kx_target), cfg.grid, cfg.time.nstep_restart, cfg.output, dict(kwargs)))
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
            "--time-scale",
            "3",
            "--checkpoint-steps",
            "20",
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
    assert out_png.with_suffix(".traces.csv").exists()
    meta = json.loads(out_png.with_suffix(".json").read_text())
    assert meta["summary_csv"].endswith("w7x_panel.csv")
    assert meta["traces_csv"].endswith("w7x_panel.traces.csv")
    assert meta["initial_policy"] == "first_abs"
    assert meta["initial_normalization"] == "line_first"
    assert meta["initial_level_override"] is None
    assert meta["damping_method"] == "branchwise_extrema"
    assert meta["frequency_method"] == "hilbert_phase"
    assert meta["validation_status"] == "open"
    assert len(meta["cases"]) == 4
    assert meta["literature_reference"]["test"] == 4
    assert meta["literature_reference"]["flux_tube"] == "bean"
    assert meta["literature_reference"]["observable"] == "unweighted line-averaged electrostatic potential"
    assert "t=0 line-average" in meta["literature_reference"]["normalization"]
    assert "slower stellarator-specific oscillation" in meta["notes"]
    assert "default --initial-normalization=line_first" in meta["notes"]
    assert "clipped initial portion of Fig. 11" in meta["notes"]
    assert "manuscript-policy inference" in meta["notes"]
    assert "digitized-reference gate" in meta["notes"]
    assert meta["runtime"] == {
        "dt": 0.2,
        "steps": 80,
        "sample_stride": 2,
        "checkpoint_steps": 20,
        "resume_output": False,
        "time_scale": 3.0,
        "diagnostics": True,
        "show_progress": True,
        "expected_tmax": 16.0,
        "Nl": 6,
        "Nm": 10,
    }
    assert len(run_calls) == 4
    trace = np.loadtxt(out_dir / "w7x_test4_kx050.csv", delimiter=",", skiprows=1)
    assert np.isclose(trace[-1, 0], 30.0)
    combined = np.genfromtxt(out_png.with_suffix(".traces.csv"), delimiter=",", names=True)
    assert combined.size == 4 * 41
    assert np.isclose(np.max(combined["t_reference"]), 30.0)
    assert "response_normalized" in combined.dtype.names
    for kx_target, grid, nstep_restart, output, kwargs in run_calls:
        assert grid.boundary == "periodic"
        assert grid.non_twist is True
        assert grid.jtwist is None
        assert np.isclose(grid.Lx, 2.0 * np.pi / kx_target)
        assert nstep_restart == 20
        assert output.restart_if_exists is False
        assert output.append_on_restart is True
        assert output.save_for_restart is True
        assert kwargs["dt"] == 0.2
        assert kwargs["steps"] == 80
        assert kwargs["sample_stride"] == 2
        assert kwargs["show_progress"] is True
        assert kwargs["Nl"] == 6
        assert kwargs["Nm"] == 10


def test_generate_w7x_zonal_response_panel_resume_output(tmp_path, monkeypatch) -> None:
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

[grid]
Nx = 6
Ny = 4
Nz = 16
Lx = 125.66370614359172
Ly = 62.8
boundary = "linked"

[time]
t_max = 1.0
dt = 0.1
method = "rk4"
sample_stride = 1
diagnostics = true
fixed_dt = true

[geometry]
model = "s-alpha"
R0 = 5.485

[init]
init_field = "phi"
init_amp = 1.0e-6
gaussian_init = true
init_single = true

[physics]
adiabatic_electrons = true
nonlinear = false

[run]
ky = 0.0
Nl = 2
Nm = 4
dt = 0.1
steps = 4
sample_stride = 1
diagnostics = true
""".strip()
    )

    out_dir = tmp_path / "w7x_out"
    out_png = tmp_path / "w7x_panel.png"
    seen = []

    def _fake_run(cfg, *, out, kx_target, **_kwargs):
        seen.append((cfg.output.restart_if_exists, cfg.output.append_on_restart, Path(cfg.output.path), Path(out)))
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0.0, 1.0, 8)
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("time", t.size)
            ds.createDimension("kx", 3)
            ds.createDimension("ri", 2)
            grids = ds.createGroup("Grids")
            diag = ds.createGroup("Diagnostics")
            grids.createVariable("time", "f8", ("time",))[:] = t
            grids.createVariable("kx", "f8", ("kx",))[:] = np.array([-float(kx_target), 0.0, float(kx_target)])
            raw = np.zeros((t.size, 3, 2), dtype=float)
            raw[:, 2, 0] = 1.0 + 0.01 * t
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
            "--kx-values",
            "0.07",
            "--resume-output",
        ],
    )

    assert mod.main() == 0
    assert seen == [(True, True, out_dir / "w7x_test4_kx070.out.nc", out_dir / "w7x_test4_kx070.out.nc")]
    meta = json.loads(out_png.with_suffix(".json").read_text())
    assert meta["runtime"]["resume_output"] is True
    assert meta["runtime"]["time_scale"] == 1.0


def test_generate_w7x_zonal_response_formats_unresolved_damping() -> None:
    mod = _load_tool_module()

    assert mod._finite_or_none(float("nan")) is None
    assert mod._format_metric(None) == "not fitted"
    assert mod._format_metric(float("nan")) == "not fitted"
    assert mod._format_metric(1.23456) == "1.235"
