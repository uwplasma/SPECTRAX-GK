from __future__ import annotations

from support.paths import REPO_ROOT, load_campaign_tool
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = REPO_ROOT


def _load_tool_module():
    return load_campaign_tool("postprocess_nonlinear_gradient_control_mean_campaign")


def _write_output(path: Path, mean: float, *, tmax: float = 100.0) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    time = np.linspace(0.0, tmax, 101)
    heat = mean + 0.002 * np.sin(2.0 * np.pi * time / 20.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", time.size)
        root.createDimension("s", 1)
        grids = root.createGroup("Grids")
        diagnostics = root.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = time
        diagnostics.createVariable("HeatFlux_st", "f8", ("time", "s"))[:, :] = heat[
            :, None
        ]


def _make_campaign(
    tmp_path: Path, *, seeds: tuple[int, ...] = (31, 32, 33, 34)
) -> Path:
    campaign = tmp_path / "control_mean_campaign"
    for idx, seed in enumerate(seeds):
        common = 10.0 + 0.001 * idx
        _write_output(
            campaign
            / "nonlinear_campaign"
            / "plus_delta"
            / f"demo_plus_t100_n64_seed{seed}.out.nc",
            common + 0.2,
        )
        _write_output(
            campaign
            / "nonlinear_campaign"
            / "minus_delta"
            / f"demo_minus_t100_n64_seed{seed}.out.nc",
            common - 0.2,
        )
    return campaign


def test_postprocess_control_mean_campaign_discovers_common_pairs(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    campaign = _make_campaign(tmp_path, seeds=(31, 33))
    extra = (
        campaign
        / "nonlinear_campaign"
        / "plus_delta"
        / "demo_plus_t100_n64_seed35.out.nc"
    )
    _write_output(extra, 10.2)

    matched = mod.discover_matched_outputs(campaign)

    assert matched["common_seeds"] == [31, 33]
    assert matched["plus_completed"] == [31, 33, 35]
    assert matched["minus_completed"] == [31, 33]


def test_postprocess_control_mean_campaign_accepts_stride_rounded_final_time(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    campaign = _make_campaign(tmp_path, seeds=(31,))
    rounded_plus = (
        campaign
        / "nonlinear_campaign"
        / "plus_delta"
        / "demo_plus_t100_n64_seed32.out.nc"
    )
    rounded_minus = (
        campaign
        / "nonlinear_campaign"
        / "minus_delta"
        / "demo_minus_t100_n64_seed32.out.nc"
    )
    _write_output(rounded_plus, 10.2, tmax=99.2)
    _write_output(rounded_minus, 9.8, tmax=99.2)

    matched = mod.discover_matched_outputs(campaign, min_tmax=99.0)

    assert matched["common_seeds"] == [31, 32]


def test_postprocess_control_mean_campaign_ignores_partial_outputs(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    campaign = _make_campaign(tmp_path, seeds=(31, 33))
    partial_plus = (
        campaign
        / "nonlinear_campaign"
        / "plus_delta"
        / "demo_plus_t100_n64_seed34.out.nc"
    )
    partial_minus = (
        campaign
        / "nonlinear_campaign"
        / "minus_delta"
        / "demo_minus_t100_n64_seed34.out.nc"
    )
    _write_output(partial_plus, 10.2, tmax=50.0)
    _write_output(partial_minus, 9.8, tmax=50.0)

    matched = mod.discover_matched_outputs(campaign, min_tmax=100.0)

    assert matched["common_seeds"] == [31, 33]
    assert matched["plus_completed"] == [31, 33]
    assert matched["minus_completed"] == [31, 33]


def test_postprocess_control_mean_campaign_status_only_reports_readiness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_tool_module()
    campaign = _make_campaign(tmp_path, seeds=(31, 32))
    partial_plus = (
        campaign
        / "nonlinear_campaign"
        / "plus_delta"
        / "demo_plus_t100_n64_seed33.out.nc"
    )
    partial_minus = (
        campaign
        / "nonlinear_campaign"
        / "minus_delta"
        / "demo_minus_t100_n64_seed33.out.nc"
    )
    _write_output(partial_plus, 10.2, tmax=50.0)
    _write_output(partial_minus, 9.8, tmax=50.0)

    status = mod.discover_campaign_status(campaign, min_tmax=99.0, min_common_pairs=3)

    assert status["common_pair_count"] == 2
    assert status["ready_for_strict_postprocess"] is False
    assert status["states"]["plus_delta"]["partial_outputs"][0]["seed"] == 33

    rc = mod.main(
        [
            "--campaign-dir",
            str(campaign),
            "--status-only",
            "--min-common-pairs",
            "2",
            "--tmax",
            "100",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready_for_strict_postprocess"] is True
    assert payload["common_seeds"] == [31, 32]


def test_postprocess_control_mean_campaign_status_deduplicates_horizon_tomls(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    campaign = _make_campaign(tmp_path, seeds=(31, 32))
    for state in ("plus_delta", "minus_delta"):
        folder = campaign / "nonlinear_campaign" / state
        for horizon in (300, 500, 700, 900):
            (folder / f"demo_{state}_t{horizon}_seed31.toml").write_text(
                "title = 'demo'\n"
            )
            (folder / f"demo_{state}_t{horizon}_seed32.toml").write_text(
                "title = 'demo'\n"
            )

    status = mod.discover_campaign_status(campaign, min_tmax=99.0, min_common_pairs=2)

    assert status["ready_for_strict_postprocess"] is True
    assert status["states"]["plus_delta"]["planned_count"] == 2
    assert status["states"]["minus_delta"]["planned_count"] == 2
    assert status["states"]["plus_delta"]["planned_seeds"] == [31, 32]
    assert status["states"]["minus_delta"]["planned_seeds"] == [31, 32]


def test_postprocess_control_mean_campaign_builds_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    campaign = _make_campaign(tmp_path)
    out_root = tmp_path / "artifacts"

    rc = mod.main(
        [
            "--campaign-dir",
            str(campaign),
            "--out-root",
            str(out_root),
            "--case-prefix",
            "demo_control_mean",
            "--tmin",
            "50",
            "--tmax",
            "100",
            "--min-common-pairs",
            "4",
            "--min-control-mean-pairs",
            "4",
            "--bootstrap-samples",
            "32",
        ]
    )

    assert rc == 0
    gate = json.loads((out_root / "demo_control_mean_gate.json").read_text())
    assert gate["passed"] is True
    assert gate["summary"]["common_pair_count"] == 4
    assert (out_root / "demo_control_mean_gate.png").exists()


def test_postprocess_control_mean_campaign_fails_closed_without_pairs(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    campaign = _make_campaign(tmp_path, seeds=(31,))

    rc = mod.main(
        [
            "--campaign-dir",
            str(campaign),
            "--out-root",
            str(tmp_path / "artifacts"),
            "--min-common-pairs",
            "2",
        ]
    )

    assert rc == 2
