"""Tests for external-VMEC time-horizon gate plotting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_external_vmec_time_horizon_gate.py"
    spec = importlib.util.spec_from_file_location("plot_external_vmec_time_horizon_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_gate(tmp_path: Path, name: str, means: tuple[float, float], *, passed: bool = True) -> Path:
    payload = {
        "kind": "external_vmec_nonlinear_grid_convergence_gate",
        "passed": passed,
        "common_window": {"max_pairwise_heat_flux_symmetric_relative_difference": 0.05},
        "least_windows": {"max_pairwise_heat_flux_symmetric_relative_difference": 0.04},
        "runs": [
            {
                "label": "n64",
                "common_window": {"heat_flux_mean": means[0]},
                "least_trending_window": {"heat_flux_mean": means[0] * 0.99},
            },
            {
                "label": "n80",
                "common_window": {"heat_flux_mean": means[1]},
                "least_trending_window": {"heat_flux_mean": means[1] * 1.01},
            },
        ],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_time_horizon_gate_passes_for_stable_high_grid_means(tmp_path: Path) -> None:
    mod = _load_tool_module()
    first = _write_gate(tmp_path, "t250", (10.0, 10.4))
    second = _write_gate(tmp_path, "t350", (10.2, 10.5))

    paths = mod.write_time_horizon_panel(
        [(250.0, first), (350.0, second)],
        out=tmp_path / "horizon.png",
        case="synthetic time horizon",
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["kind"] == "external_vmec_time_horizon_gate"
    assert payload["passed"] is True
    assert payload["promotion_gate"]["passed"] is False
    assert payload["claim_level"] == "passed_high_grid_time_horizon_candidate_not_replicated_holdout"
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["csv"]).exists()


def test_time_horizon_gate_fails_large_horizon_shift(tmp_path: Path) -> None:
    mod = _load_tool_module()
    first = _write_gate(tmp_path, "t250", (10.0, 10.0))
    second = _write_gate(tmp_path, "t350", (14.0, 14.0))

    payload = mod.build_time_horizon_payload(
        [(250.0, first), (350.0, second)],
        case="synthetic time horizon",
    )

    failed = {gate["metric"] for gate in payload["gate_report"]["gates"] if not gate["passed"]}
    assert payload["passed"] is False
    assert payload["claim_level"] == "negative_time_horizon_result_not_transport_validation"
    assert "common_window_time_horizon_relative_change" in failed


def test_time_horizon_gate_fails_when_input_grid_gate_failed(tmp_path: Path) -> None:
    mod = _load_tool_module()
    first = _write_gate(tmp_path, "t250", (10.0, 10.4))
    second = _write_gate(tmp_path, "t350", (10.2, 10.5), passed=False)

    payload = mod.build_time_horizon_payload(
        [(250.0, first), (350.0, second)],
        case="synthetic time horizon",
    )

    failed = {gate["metric"] for gate in payload["gate_report"]["gates"] if not gate["passed"]}
    assert payload["passed"] is False
    assert "failed_grid_gate_count" in failed
