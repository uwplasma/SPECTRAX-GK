from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from spectraxgk.quasilinear_window import (
    NonlinearWindowConvergenceConfig,
    nonlinear_window_convergence_report,
    nonlinear_window_stats_promotion_ready,
)


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "check_nonlinear_window_convergence.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_nonlinear_window_convergence", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _saturated_trace() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 200.0, 201)
    heat = 4.0 + 0.04 * np.sin(2.0 * np.pi * t / 10.0)
    return t, heat


def test_converged_saturated_transport_window_passes_with_finite_uncertainty() -> None:
    t, heat = _saturated_trace()

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="synthetic_saturated_itg",
        source_artifact="synthetic.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            min_blocks=4,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )

    assert report["passed"] is True
    assert report["statistics"]["late_mean"] == pytest.approx(4.0, abs=5.0e-3)
    assert np.isfinite(report["statistics"]["block_bootstrap_sem"])
    assert np.isfinite(report["statistics"]["sem"])
    assert report["window"]["transient_cutoff"] == pytest.approx(100.0)
    ready, failures = nonlinear_window_stats_promotion_ready(report)
    assert ready is True
    assert failures == []


def test_transient_only_trace_fails_running_mean_gate() -> None:
    t = np.linspace(0.0, 120.0, 121)
    heat = 0.05 * t

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="ramping_transient",
        source_artifact="ramp.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=32,
            max_running_mean_rel_drift=0.05,
            max_sem_rel=1.0,
        ),
    )

    failed = {gate["metric"] for gate in report["gates"] if not gate["passed"]}
    assert report["passed"] is False
    assert "running_mean_drift" in failed


def test_small_window_and_nan_late_window_fail() -> None:
    t = np.linspace(0.0, 10.0, 11)
    heat = np.ones_like(t)
    small = nonlinear_window_convergence_report(
        t,
        heat,
        case="small_window",
        source_artifact="small.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=16,
            min_blocks=4,
        ),
    )
    small_failed = {gate["metric"] for gate in small["gates"] if not gate["passed"]}
    assert "finite_sample_count" in small_failed

    t2, heat2 = _saturated_trace()
    heat2[-3] = np.nan
    nan_report = nonlinear_window_convergence_report(
        t2,
        heat2,
        case="nan_late_window",
        source_artifact="nan.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
        ),
    )
    nan_failed = {gate["metric"] for gate in nan_report["gates"] if not gate["passed"]}
    assert nan_report["passed"] is False
    assert "finite_late_window" in nan_failed


def test_check_nonlinear_window_convergence_tool_writes_json(tmp_path: Path) -> None:
    mod = _load_tool_module()
    t, heat = _saturated_trace()
    csv = tmp_path / "trace.csv"
    csv.write_text(
        "t,heat_flux\n"
        + "\n".join(f"{ti:.8g},{qi:.12g}" for ti, qi in zip(t, heat, strict=True))
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "window.json"

    assert (
        mod.main(
            [
                "--csv",
                str(csv),
                "--out-json",
                str(out),
                "--min-samples",
                "64",
                "--max-sem-rel",
                "0.02",
                "--max-running-mean-rel-drift",
                "0.02",
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["provenance"]["source_artifact"] == str(csv)


def test_nonlinear_window_script_imports_before_editable_install() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": ""}

    completed = subprocess.run(
        [
            sys.executable,
            "tools/check_nonlinear_window_convergence.py",
            "--help",
        ],
        cwd=root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Check nonlinear late-window convergence metadata" in completed.stdout
