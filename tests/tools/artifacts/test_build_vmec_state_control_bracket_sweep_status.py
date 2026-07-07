from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "tools" / "artifacts" / "build_vmec_state_control_bracket_sweep_status.py"
)


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "build_vmec_state_control_bracket_sweep_status", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gate(
    path: Path, *, alpha: float, parameter: str, response: float, passed: bool
) -> None:
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "blockers": [] if passed else ["fd_response_resolved"],
                "delta_parameter": alpha,
                "parameter_name": parameter,
                "config": {
                    "min_fd_response_fraction": 0.03,
                    "max_fd_asymmetry_rel": 0.5,
                    "max_gradient_uncertainty_rel": 0.5,
                },
                "metrics": {
                    "response_fraction": response,
                    "fd_asymmetry_rel": 0.2 if passed else 2.0,
                    "gradient_uncertainty_rel": 0.1 if passed else 4.0,
                    "baseline_window_mean": 1.0,
                    "plus_window_mean": 1.1,
                    "minus_window_mean": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_vmec_state_control_bracket_sweep_status(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gate_a = tmp_path / "gate_a.json"
    gate_b = tmp_path / "gate_b.json"
    run_summary = tmp_path / "summary.json"
    out_prefix = tmp_path / "status"
    _gate(
        gate_a,
        alpha=0.003,
        parameter="state_control_rsin_mid_surface_m1",
        response=0.004,
        passed=False,
    )
    _gate(
        gate_b,
        alpha=0.01,
        parameter="state_control_zcos_mid_surface_m1",
        response=0.04,
        passed=True,
    )
    run_summary.write_text(
        json.dumps(
            {"successes": 36, "failures": [], "started_at": 1.0, "finished_at": 4.5}
        )
    )

    report = mod.build_bracket_sweep_status(
        [gate_b, gate_a], run_summary=run_summary, out_prefix=out_prefix
    )

    assert report["passed"] is False
    assert report["summary"]["central_fd_gates_passed"] == 1
    assert report["summary"]["central_fd_gates_total"] == 2
    assert report["summary"]["nonlinear_runs_completed"] == 36
    assert report["rows"][0]["alpha_delta"] == 0.003
    assert out_prefix.with_suffix(".json").exists()
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()
