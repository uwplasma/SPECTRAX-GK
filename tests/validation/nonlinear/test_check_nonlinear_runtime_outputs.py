from __future__ import annotations

from pathlib import Path

from support.paths import REPO_ROOT, load_release_tool

import numpy as np
import pytest


ROOT = REPO_ROOT
SCRIPT = ROOT / "tools" / "release" / "check_nonlinear_runtime_outputs.py"


def _load_tool_module():
    return load_release_tool("check_nonlinear_runtime_outputs")


def _write_nonlinear_output(
    path: Path,
    *,
    time: np.ndarray | None = None,
    heat: np.ndarray | None = None,
    include_heat: bool = True,
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    t = np.asarray([0.0, 5.0, 10.0] if time is None else time, dtype=float)
    q = np.asarray([1.0, 2.0, 3.0] if heat is None else heat, dtype=float)
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", t.size)
        root.createDimension("s", 1)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = t
        diagnostics = root.createGroup("Diagnostics")
        if include_heat:
            diagnostics.createVariable("HeatFlux_st", "f8", ("time", "s"))[:, :] = q[
                :, None
            ]


def test_validate_output_accepts_grouped_runtime_netcdf(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out = tmp_path / "run.out.nc"
    _write_nonlinear_output(out)

    row = mod.validate_output(
        out, min_samples=3, tmin=5.0, tmax=10.0, min_window_samples=2
    )

    assert row["passed"] is True
    assert row["samples"] == 3
    assert row["window"]["mean_heat_flux"] == pytest.approx(2.5)


def test_validate_output_fails_closed_for_missing_diagnostics(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out = tmp_path / "restart_like.out.nc"
    _write_nonlinear_output(out, include_heat=False)

    row = mod.validate_output(out, min_samples=2)

    assert row["passed"] is False
    assert any("HeatFlux_st" in failure for failure in row["failures"])


def test_check_outputs_reports_required_window_failures(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out = tmp_path / "short.out.nc"
    _write_nonlinear_output(out, time=np.asarray([0.0, 5.0, 10.0]), heat=np.ones(3))

    payload = mod.check_outputs(
        [out],
        heat_flux_variable=mod.DEFAULT_HEAT_FLUX_VARIABLE,
        min_samples=2,
        tmin=20.0,
        tmax=30.0,
        tmax_atol=None,
        min_window_samples=2,
        min_abs_window_mean=None,
    )

    assert payload["passed"] is False
    assert payload["summary"] == {"outputs": 1, "passed": 0, "failed": 1}
    assert {
        "does_not_reach_required_tmax",
        "too_few_window_samples",
    }.issubset(set(payload["rows"][0]["failures"]))


def test_validate_output_tolerates_fixed_step_time_roundoff(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out = tmp_path / "rounded.out.nc"
    _write_nonlinear_output(
        out,
        time=np.asarray([896.0, 898.0, 899.99]),
        heat=np.asarray([2.0, 2.1, 2.2]),
    )

    row = mod.validate_output(out, min_samples=3, tmin=896.0, tmax=900.0)

    assert row["passed"] is True
    assert row["tmax_atol"] == pytest.approx(0.25 * np.median([2.0, 1.99]))


def test_validate_output_can_enforce_strict_tmax_tolerance(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out = tmp_path / "rounded.out.nc"
    _write_nonlinear_output(out, time=np.asarray([0.0, 2.0, 899.99]), heat=np.ones(3))

    row = mod.validate_output(out, min_samples=3, tmax=900.0, tmax_atol=1.0e-4)

    assert row["passed"] is False
    assert "does_not_reach_required_tmax" in row["failures"]


def test_main_writes_json_and_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_tool_module()
    out = tmp_path / "run.out.nc"
    report = tmp_path / "report.json"
    _write_nonlinear_output(out)

    rc = mod.main([str(out), "--json-out", str(report), "--min-samples", "3"])

    assert rc == 0
    assert report.exists()
    assert '"failed": 0' in capsys.readouterr().out
