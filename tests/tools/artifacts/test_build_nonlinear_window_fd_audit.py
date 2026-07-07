from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "artifacts"
        / "build_nonlinear_window_fd_audit.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_nonlinear_window_fd_audit", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_run(label: str, tprim: float, scale: float = 1.0) -> dict:
    time = np.linspace(0.0, 1.0, 8)
    heat = scale * (1.0 + 0.08 * time)
    return {
        "label": label,
        "tprim": tprim,
        "random_seed": 22,
        "time": time.tolist(),
        "heat_flux": heat.tolist(),
        "window": _load_tool_module().late_window_metrics(
            time, heat, tail_fraction=0.5
        ),
    }


def test_late_window_metrics_reports_conditioning_quantities() -> None:
    mod = _load_tool_module()
    time = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    heat = np.asarray([1.0, 1.1, 1.2, 1.3, 1.4])

    metrics = mod.late_window_metrics(time, heat, tail_fraction=0.4)

    assert metrics["start_index"] == 3
    assert metrics["n_samples"] == 2
    assert metrics["mean"] == 1.35
    assert metrics["cv"] > 0.0
    assert metrics["trend"] > 0.0


def test_build_audit_payload_passes_conditioned_monotone_runs() -> None:
    mod = _load_tool_module()
    runs = [
        _synthetic_run("minus", 2.31, 0.90),
        _synthetic_run("base", 2.49, 1.00),
        _synthetic_run("plus", 2.67, 1.12),
        _synthetic_run("base_repeat", 2.49, 1.00),
    ]

    payload = mod.build_audit_payload(
        runs,
        base_tprim=2.49,
        perturbation_step=0.18,
        tail_fraction=0.5,
        repeatability_rtol=1.0e-12,
        max_window_cv=0.1,
        max_window_trend=0.1,
        min_response_fraction=0.03,
    )

    assert payload["passed"] is True
    assert payload["startup_nonlinear_plumbing_fd_path_gate"] is True
    assert payload["transport_average_gate"] is False
    assert payload["production_nonlinear_observable_fd_path_gate"] is False
    assert payload["production_nonlinear_window_gradient_gate"] is False
    assert payload["gates"]["monotonic_drive_response"] is True
    assert payload["metrics"]["central_fd_dq_dtprim"] > 0.0
    assert payload["transport_average_requirements"]["passed"] is False


def test_build_audit_payload_blocks_unresolved_response() -> None:
    mod = _load_tool_module()
    runs = [
        _synthetic_run("minus", 2.31, 0.999),
        _synthetic_run("base", 2.49, 1.000),
        _synthetic_run("plus", 2.67, 1.001),
        _synthetic_run("base_repeat", 2.49, 1.000),
    ]

    payload = mod.build_audit_payload(
        runs,
        base_tprim=2.49,
        perturbation_step=0.18,
        tail_fraction=0.5,
        repeatability_rtol=1.0e-12,
        max_window_cv=0.1,
        max_window_trend=0.1,
        min_response_fraction=0.03,
    )

    assert payload["passed"] is False
    assert payload["gates"]["resolved_fd_response"] is False


def test_main_writes_artifacts_without_running_solver(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_tool_module()

    def fake_run_cyclone_window(*, label: str, tprim: float, **_kwargs):
        scale = {"minus": 0.90, "base": 1.00, "plus": 1.12, "base_repeat": 1.00}[label]
        time = np.linspace(0.0, 1.0, 8)
        heat = scale * (1.0 + 0.08 * time)
        return {
            "label": label,
            "tprim": tprim,
            "random_seed": 22,
            "time": time.tolist(),
            "heat_flux": heat.tolist(),
            "window": mod.late_window_metrics(time, heat, tail_fraction=0.5),
        }

    monkeypatch.setattr(mod, "run_cyclone_window", fake_run_cyclone_window)
    out = tmp_path / "audit.png"

    assert mod.main(["--out", str(out), "--tail-fraction", "0.5"]) == 0

    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".csv").exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["passed"] is True
    assert (
        meta["claim_level"]
        == "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average"
    )
    assert meta["transport_average_gate"] is False
