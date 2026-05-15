"""Tests for external-VMEC nonlinear convergence gate plotting."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_external_vmec_nonlinear_convergence_gate.py"
    spec = importlib.util.spec_from_file_location("plot_external_vmec_nonlinear_convergence_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pilot(tmp_path: Path, name: str, mean: float, *, slope: float = 0.0) -> Path:
    t = np.linspace(0.0, 20.0, 21)
    heat_flux = mean + slope * t + 0.01 * np.sin(t)
    wphi = 2.0 + 0.02 * np.cos(t)
    csv_path = tmp_path / f"{name}.traces.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "heat_flux", "wphi", "wg"])
        for row in zip(t, heat_flux, wphi, np.ones_like(t), strict=True):
            writer.writerow([f"{float(value):.16e}" for value in row])
    late = t >= 10.0
    report = {
        "kind": "nonlinear_feasibility_pilot",
        "label": name,
        "csv": csv_path.name,
        "least_trending_window": {
            "tmin": float(t[late][0]),
            "tmax": float(t[late][-1]),
            "heat_flux_mean": float(np.mean(heat_flux[late])),
            "heat_flux_std": float(np.std(heat_flux[late])),
            "heat_flux_relative_slope_per_time": 0.0,
            "n_samples": int(np.count_nonzero(late)),
        },
    }
    json_path = tmp_path / f"{name}.json"
    json_path.write_text(json.dumps(report), encoding="utf-8")
    return json_path


def test_convergence_gate_passes_for_flat_nearby_traces(tmp_path: Path) -> None:
    mod = _load_tool_module()
    first = _write_pilot(tmp_path, "n32", 1.0)
    second = _write_pilot(tmp_path, "n48", 1.05)

    paths = mod.write_convergence_gate(
        [first, second],
        out=tmp_path / "gate.png",
        labels=["n32", "n48"],
        case="synthetic external VMEC convergence",
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["promotion_gate"]["passed"] is True
    assert payload["claim_level"] == "passed_grid_convergence_candidate_for_transport_holdout"
    assert payload["promotion_gate"]["reason"].startswith("synthetic external VMEC convergence passed")
    assert payload["gate_report"]["passed"] is True
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["csv"]).exists()


def test_convergence_gate_fails_large_grid_shift(tmp_path: Path) -> None:
    mod = _load_tool_module()
    first = _write_pilot(tmp_path, "n32", 1.0)
    second = _write_pilot(tmp_path, "n48", 1.6)

    paths = mod.write_convergence_gate(
        [first, second],
        out=tmp_path / "gate.png",
        labels=["n32", "n48"],
        case="synthetic external VMEC convergence",
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    failed = {gate["metric"] for gate in payload["gate_report"]["gates"] if not gate["passed"]}
    assert payload["passed"] is False
    assert payload["promotion_gate"]["passed"] is False
    assert payload["claim_level"] == "negative_grid_convergence_result_not_transport_validation"
    assert payload["promotion_gate"]["reason"].startswith("synthetic external VMEC convergence is finite")
    assert "common_window_pairwise_heat_flux_symmetric_relative_difference" in failed
    assert "least_window_pairwise_heat_flux_symmetric_relative_difference" in failed
