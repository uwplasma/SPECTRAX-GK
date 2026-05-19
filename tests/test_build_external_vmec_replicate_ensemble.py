from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "build_external_vmec_replicate_ensemble.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_external_vmec_replicate_ensemble", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_output(path: Path, offset: float) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    t = np.linspace(0.0, 100.0, 101)
    q = 10.0 + offset + 0.02 * np.sin(2.0 * np.pi * t / 20.0)
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", t.size)
        root.createDimension("s", 1)
        grids = root.createGroup("Grids")
        diagnostics = root.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        diagnostics.createVariable("HeatFlux_st", "f8", ("time", "s"))[:, :] = q[:, None]


def test_replicate_ensemble_tool_builds_trace_reports_and_plot(tmp_path: Path) -> None:
    mod = _load_tool_module()
    outputs = [
        tmp_path / "demo_nonlinear_t100_n64_seed31.out.nc",
        tmp_path / "demo_nonlinear_t100_n64_seed32.out.nc",
        tmp_path / "demo_nonlinear_t100_n64_dt0p04.out.nc",
    ]
    for path, offset in zip(outputs, (-0.05, 0.05, 0.0)):
        _write_output(path, offset)
    out_dir = tmp_path / "artifacts"

    rc = mod.main(
        [
            *[str(path) for path in outputs],
            "--out-dir",
            str(out_dir),
            "--case",
            "demo_replicate_window",
            "--tmin",
            "50",
            "--tmax",
            "100",
            "--baseline-seed",
            "22",
            "--baseline-dt",
            "0.05",
            "--artifact-prefix",
            "docs/_static/demo_replicates",
            "--bootstrap-samples",
            "32",
        ]
    )

    assert rc == 0
    readiness = json.loads((out_dir / "replicate_ensemble_readiness.json").read_text())
    ensemble = json.loads((out_dir / "replicate_ensemble_gate.json").read_text())
    summary = json.loads(
        (out_dir / "demo_nonlinear_t100_n64_seed31_transport_window.json").read_text()
    )
    assert readiness["passed"] is True
    assert ensemble["passed"] is True
    assert summary["nonlinear_artifact"] == "demo_nonlinear_t100_n64_seed31.out.nc"
    assert len(list(out_dir.glob("*_heat_flux_trace.csv"))) == 3
    assert (out_dir / "replicate_ensemble_gate.png").exists()
    assert (
        readiness["observed_artifacts"][0]["source_artifact"]
        .startswith("docs/_static/demo_replicates/")
    )


def test_replicate_ensemble_tool_parses_joint_seed_timestep_variant(tmp_path: Path) -> None:
    mod = _load_tool_module()
    variant = mod._variant_from_path(
        tmp_path / "demo_nonlinear_t100_n64_seed32_dt0p04.out.nc",
        baseline_seed=22,
        baseline_dt=0.05,
    )

    assert variant == {
        "variant_axis": "seed_timestep",
        "variant_label": "seed32_dt0p04",
        "seed": 32,
        "dt": 0.04,
        "variant": {"seed": 32, "timestep": 0.04},
    }
