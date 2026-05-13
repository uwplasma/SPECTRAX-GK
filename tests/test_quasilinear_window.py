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
    nonlinear_window_convergence_from_csv,
    nonlinear_window_convergence_from_summary,
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


def test_window_report_supports_explicit_bounds_and_deterministic_blocks() -> None:
    t, heat = _saturated_trace()

    report = nonlinear_window_convergence_report(
        t,
        heat,
        case="bounded_saturated_itg",
        source_artifact="bounded.csv",
        config=NonlinearWindowConvergenceConfig(
            tmin=50.0,
            tmax=150.0,
            transient_fraction=0.25,
            block_size=8,
            bootstrap_samples=0,
            min_samples=40,
            min_blocks=4,
            max_running_mean_rel_drift=0.03,
            max_sem_rel=0.03,
        ),
    )

    assert report["passed"] is False
    assert report["window"]["selected_tmin"] == pytest.approx(50.0)
    assert report["window"]["selected_tmax"] == pytest.approx(150.0)
    assert report["statistics"]["block_size"] == 8
    assert report["statistics"]["block_bootstrap_sem"] is None
    failed = {gate["metric"] for gate in report["gates"] if not gate["passed"]}
    assert failed == {"block_bootstrap_sem"}
    # With no bootstrap samples, the diagnostic SEM falls back to sample/block SEM,
    # but promotion still fails closed because bootstrap uncertainty was requested.
    assert np.isfinite(report["statistics"]["sem"])


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


def test_nonlinear_window_config_and_input_validation_are_fail_closed() -> None:
    t = np.linspace(0.0, 10.0, 11)
    heat = np.ones_like(t)

    bad_configs = [
        (NonlinearWindowConvergenceConfig(tmin=np.nan), "tmin must be finite"),
        (NonlinearWindowConvergenceConfig(tmax=np.inf), "tmax must be finite"),
        (NonlinearWindowConvergenceConfig(tmin=5.0, tmax=5.0), "tmin must be less"),
        (NonlinearWindowConvergenceConfig(transient_fraction=1.0), "transient_fraction"),
        (NonlinearWindowConvergenceConfig(min_samples=1), "min_samples"),
        (NonlinearWindowConvergenceConfig(min_blocks=1), "min_blocks"),
        (NonlinearWindowConvergenceConfig(block_size=0), "block_size"),
        (NonlinearWindowConvergenceConfig(bootstrap_samples=-1), "bootstrap_samples"),
        (
            NonlinearWindowConvergenceConfig(max_running_mean_rel_drift=-1.0),
            "running_mean",
        ),
        (NonlinearWindowConvergenceConfig(max_sem_rel=-1.0), "max_sem_rel"),
        (NonlinearWindowConvergenceConfig(value_floor=0.0), "value_floor"),
    ]
    for config, message in bad_configs:
        with pytest.raises(ValueError, match=message):
            nonlinear_window_convergence_report(t, heat, config=config)

    with pytest.raises(ValueError, match="same length"):
        nonlinear_window_convergence_report([0.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="must not be empty"):
        nonlinear_window_convergence_report([], [])
    with pytest.raises(ValueError, match="time contains non-finite"):
        nonlinear_window_convergence_report([0.0, np.nan], [1.0, 1.0])
    with pytest.raises(ValueError, match="selected nonlinear window is empty"):
        nonlinear_window_convergence_report([1.0], [1.0])


def test_nonlinear_window_csv_and_summary_loaders_cover_artifact_contracts(
    tmp_path: Path,
) -> None:
    csv = tmp_path / "diagnostics.csv"
    t, heat = _saturated_trace()
    csv.write_text(
        "time,q_i\n"
        + "\n".join(f"{ti:.8g},{qi:.12g}" for ti, qi in zip(t, heat, strict=True))
        + "\n",
        encoding="utf-8",
    )
    config = NonlinearWindowConvergenceConfig(
        transient_fraction=0.5,
        min_samples=64,
        max_running_mean_rel_drift=0.02,
        max_sem_rel=0.02,
    )

    from_csv = nonlinear_window_convergence_from_csv(
        csv,
        time_column="time",
        value_column="q_i",
        case="csv_case",
        config=config,
        summary_artifact="summary.json",
    )

    assert from_csv["passed"] is True
    assert from_csv["case"] == "csv_case"
    assert from_csv["observable"] == "q_i"
    assert from_csv["provenance"]["summary_artifact"] == "summary.json"

    summary = tmp_path / "window_summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": "summary_case",
                "spectrax": "diagnostics.csv",
                "tmin": 50.0,
                "tmax": 200.0,
            }
        ),
        encoding="utf-8",
    )
    from_summary = nonlinear_window_convergence_from_summary(
        summary,
        time_column="time",
        value_column="q_i",
        config=config,
    )

    assert from_summary["case"] == "summary_case"
    assert from_summary["provenance"]["summary_artifact"] == str(summary)
    assert from_summary["provenance"]["source_artifact"].endswith("diagnostics.csv")

    missing_column = tmp_path / "missing_column.csv"
    missing_column.write_text("time,other\n0.0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="observable column"):
        nonlinear_window_convergence_from_csv(
            missing_column,
            time_column="time",
            value_column="q_i",
        )

    bad_summary = tmp_path / "bad_summary.json"
    bad_summary.write_text(json.dumps({"other": "diagnostics.csv"}), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostics source"):
        nonlinear_window_convergence_from_summary(bad_summary)

    txt_summary = tmp_path / "txt_summary.json"
    txt_summary.write_text(json.dumps({"spectrax": "trace.txt"}), encoding="utf-8")
    (tmp_path / "trace.txt").write_text("not,csv\n", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="diagnostics CSV"):
        nonlinear_window_convergence_from_summary(txt_summary)


def test_nonlinear_window_promotion_ready_reports_all_missing_contracts() -> None:
    ready, failures = nonlinear_window_stats_promotion_ready(None)
    assert ready is False
    assert failures == ["missing nonlinear_window_stats object"]

    ready, failures = nonlinear_window_stats_promotion_ready(
        {
            "kind": "wrong",
            "passed": False,
            "provenance": {},
            "statistics": {"late_mean": 1.0},
            "window": {"transient_fraction": 0.0, "n_finite_late": 0},
            "gate_report": {"passed": False},
        }
    )

    assert ready is False
    assert "unexpected nonlinear_window_stats kind" in failures
    assert "nonlinear window convergence report did not pass" in failures
    assert "missing nonlinear source_artifact provenance" in failures
    assert "missing/non-finite statistics.sem" in failures
    assert "missing/non-finite window.late_tmin" in failures
    assert "missing declared transient cutoff policy" in failures
    assert "window has no finite late samples" in failures
    assert "missing passed gate_report" in failures
