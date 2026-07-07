from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "artifacts"
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
        diagnostics.createVariable("HeatFlux_st", "f8", ("time", "s"))[:, :] = q[
            :, None
        ]


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
    assert readiness["observed_artifacts"][0]["source_artifact"].startswith(
        "docs/_static/demo_replicates/"
    )


def test_replicate_ensemble_tool_can_collect_failed_diagnostic_points(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    outputs = [
        tmp_path / "diagnostic_nonlinear_t100_n64_seed31.out.nc",
        tmp_path / "diagnostic_nonlinear_t100_n64_seed32.out.nc",
        tmp_path / "diagnostic_nonlinear_t100_n64_dt0p04.out.nc",
    ]
    for path, offset in zip(outputs, (-4.0, 0.0, 4.0)):
        _write_output(path, offset)

    common_args = [
        *[str(path) for path in outputs],
        "--case",
        "diagnostic_landscape_point",
        "--tmin",
        "50",
        "--tmax",
        "100",
        "--baseline-seed",
        "22",
        "--baseline-dt",
        "0.05",
        "--bootstrap-samples",
        "32",
        "--max-mean-rel-spread",
        "0.01",
    ]
    strict_dir = tmp_path / "strict"
    relaxed_dir = tmp_path / "relaxed"

    strict_rc = mod.main([*common_args, "--out-dir", str(strict_dir)])
    relaxed_rc = mod.main(
        [*common_args, "--out-dir", str(relaxed_dir), "--allow-failed-gates"]
    )

    assert strict_rc == 1
    assert relaxed_rc == 0
    ensemble = json.loads((relaxed_dir / "replicate_ensemble_gate.json").read_text())
    assert ensemble["passed"] is False
    assert (relaxed_dir / "replicate_ensemble_gate.png").exists()


def test_replicate_ensemble_tool_handles_requested_window_outside_trace(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    outputs = [
        tmp_path / "short_nonlinear_t100_n64_seed31.out.nc",
        tmp_path / "short_nonlinear_t100_n64_seed32.out.nc",
        tmp_path / "short_nonlinear_t100_n64_dt0p04.out.nc",
    ]
    for path, offset in zip(outputs, (-0.05, 0.05, 0.0)):
        _write_output(path, offset)
    out_dir = tmp_path / "outside_window"

    rc = mod.main(
        [
            *[str(path) for path in outputs],
            "--out-dir",
            str(out_dir),
            "--case",
            "outside_requested_window",
            "--tmin",
            "200",
            "--tmax",
            "300",
            "--bootstrap-samples",
            "16",
            "--allow-failed-gates",
        ]
    )

    assert rc == 0
    ensemble = json.loads((out_dir / "replicate_ensemble_gate.json").read_text())
    report = json.loads(
        next(
            (out_dir / "nonlinear_window_convergence_reports").glob("*seed31*")
        ).read_text()
    )
    assert ensemble["passed"] is False
    assert ensemble["statistics"]["n_finite_means"] == 0
    assert ensemble["statistics"]["ensemble_mean"] is None
    assert report["window"]["n_finite_late"] == 0
    assert (out_dir / "replicate_ensemble_gate.png").exists()


def test_replicate_ensemble_tool_parses_joint_seed_timestep_variant(
    tmp_path: Path,
) -> None:
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


def test_replicate_ensemble_tool_parses_timestep_variant_with_device_suffix(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    variant = mod._variant_from_path(
        tmp_path / "demo_nonlinear_t250_n48_dt0p01_gpu.out.nc",
        baseline_seed=22,
        baseline_dt=0.05,
    )

    assert variant == {
        "variant_axis": "timestep",
        "variant_label": "dt0p01",
        "seed": 22,
        "dt": 0.01,
        "variant": {"seed": 22, "timestep": 0.01},
    }


def test_replicate_ensemble_tool_ignores_protocol_dt_in_case_slug(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()

    seed_variant = mod._variant_from_path(
        tmp_path / "solovev_reference_repair_dt002_amp1em5_n48_seed31.out.nc",
        baseline_seed=22,
        baseline_dt=0.02,
    )
    timestep_variant = mod._variant_from_path(
        tmp_path / "solovev_reference_repair_dt002_amp1em5_n48_dt0p01_gpu.out.nc",
        baseline_seed=22,
        baseline_dt=0.02,
    )

    assert seed_variant == {
        "variant_axis": "seed",
        "variant_label": "seed31",
        "seed": 31,
        "dt": 0.02,
        "variant": {"seed": 31, "timestep": 0.02},
    }
    assert timestep_variant == {
        "variant_axis": "timestep",
        "variant_label": "dt0p01",
        "seed": 22,
        "dt": 0.01,
        "variant": {"seed": 22, "timestep": 0.01},
    }
