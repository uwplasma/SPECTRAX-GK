from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "generate_nonlinear_sharding_production_gate.py"
    spec = importlib.util.spec_from_file_location("generate_nonlinear_sharding_production_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(*, backend: str, speedup: float, identity: bool = True, active: bool = True) -> dict:
    return {
        "backend": backend,
        "requested_devices": 2,
        "actual_devices": 2,
        "best_spec": "kx",
        "state_sharding_active": active,
        "identity_gate_pass": identity,
        "strong_speedup_vs_1_device": speedup,
        "max_abs_state_error": 0.0,
        "max_rel_state_error": 0.0,
        "error": None,
    }


def test_nonlinear_sharding_production_gate_defaults_to_tracked_inputs() -> None:
    mod = _load_tool_module()
    args = mod.build_parser().parse_args([])

    assert args.inputs == mod.DEFAULT_INPUTS
    assert args.out_prefix == mod.DEFAULT_OUT_PREFIX
    assert args.min_speedup == mod.DEFAULT_MIN_SPEEDUP
    assert args.identity_atol == mod.DEFAULT_IDENTITY_ATOL
    assert args.identity_rtol == mod.DEFAULT_IDENTITY_RTOL
    assert args.required_backends == ("cpu", "gpu")


def test_nonlinear_sharding_production_gate_fails_closed_on_identity_without_speedup() -> None:
    mod = _load_tool_module()

    summary = mod.evaluate_production_gate(
        [_row(backend="gpu", speedup=0.96)],
        required_backends=("gpu",),
        min_speedup=1.20,
        min_efficiency=0.50,
    )

    assert summary["gate_passed"] is False
    assert summary["production_speedup_claim_allowed"] is False
    assert summary["status"] == "diagnostic_only"
    assert summary["blockers"] == ["gpu_production_speedup_candidate_missing"]
    assert summary["rows"][0]["candidate_passed"] is False
    assert summary["rows"][0]["classification"] == "identity_preserving_regression"
    assert "speedup_below_threshold" in summary["rows"][0]["blockers"]


def test_nonlinear_sharding_production_gate_requires_identity_and_active_sharding() -> None:
    mod = _load_tool_module()

    summary = mod.evaluate_production_gate(
        [
            _row(backend="gpu", speedup=1.6, identity=False),
            _row(backend="gpu", speedup=1.8, active=False),
        ],
        required_backends=("gpu",),
        min_speedup=1.20,
        min_efficiency=0.50,
    )

    assert summary["gate_passed"] is False
    assert "identity_gate_failed" in summary["rows"][0]["blockers"]
    assert summary["rows"][0]["classification"] == "identity_failed"
    assert "state_sharding_inactive" in summary["rows"][1]["blockers"]
    assert summary["rows"][1]["classification"] == "inactive_or_fallback"


def test_nonlinear_sharding_production_gate_passes_only_matching_backend_candidates() -> None:
    mod = _load_tool_module()

    summary = mod.evaluate_production_gate(
        [_row(backend="cpu", speedup=1.3), _row(backend="gpu", speedup=1.4)],
        required_backends=("cpu", "gpu"),
        min_speedup=1.20,
        min_efficiency=0.50,
    )

    assert summary["gate_passed"] is True
    assert summary["production_speedup_claim_allowed"] is True
    assert summary["status"] == "production_speedup_candidate"
    assert summary["best_candidates"]["gpu"]["strong_speedup_vs_1_device"] == 1.4
    assert summary["rows"][0]["classification"] == "production_candidate"
    assert summary["backend_summary"]["gpu"]["production_candidate_count"] == 1


def test_nonlinear_sharding_production_gate_loads_sweep_rows(tmp_path: Path) -> None:
    mod = _load_tool_module()
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps({"backend": "gpu", "rows": [_row(backend="gpu", speedup=1.4)]}), encoding="utf-8")

    rows = mod.load_rows([path])

    assert rows[0]["backend"] == "gpu"
    assert rows[0]["source"] == str(path)


def test_nonlinear_sharding_production_gate_fails_closed_on_missing_error_metrics() -> None:
    mod = _load_tool_module()
    row = _row(backend="gpu", speedup=1.6)
    row.pop("max_abs_state_error")

    summary = mod.evaluate_production_gate(
        [row],
        required_backends=("gpu",),
        min_speedup=1.20,
        min_efficiency=0.50,
    )

    assert summary["gate_passed"] is False
    assert summary["rows"][0]["classification"] == "identity_failed"
    assert "identity_abs_error_missing" in summary["rows"][0]["blockers"]


def test_nonlinear_sharding_production_gate_classifies_reference_and_weak_scaling() -> None:
    mod = _load_tool_module()
    reference = _row(backend="cpu", speedup=1.0)
    reference["requested_devices"] = 1
    reference["actual_devices"] = 1
    reference["state_sharding_active"] = False
    weak = _row(backend="cpu", speedup=1.3)
    weak["requested_devices"] = 4
    weak["actual_devices"] = 4

    summary = mod.evaluate_production_gate(
        [reference, weak],
        required_backends=("cpu",),
        min_speedup=1.20,
        min_efficiency=0.50,
    )

    assert summary["gate_passed"] is False
    assert summary["rows"][0]["classification"] == "reference_only"
    assert summary["rows"][1]["classification"] == "identity_only_insufficient_speedup"
    assert "parallel_efficiency_below_threshold" in summary["rows"][1]["blockers"]
    assert (
        summary["backend_summary"]["cpu"]["best_identity_preserving_row"]["requested_devices"]
        == 4
    )
