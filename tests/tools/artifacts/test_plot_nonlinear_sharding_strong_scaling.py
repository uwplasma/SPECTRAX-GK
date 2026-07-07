from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "plot_nonlinear_sharding_strong_scaling.py"
    spec = importlib.util.spec_from_file_location("plot_nonlinear_sharding_strong_scaling", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_nonlinear_sharding_strong_scaling_parser_defaults_to_large_inputs() -> None:
    mod = _load_tool_module()

    args = mod.build_parser().parse_args([])

    assert args.inputs == mod.DEFAULT_INPUTS
    assert args.out_prefix == mod.DEFAULT_PREFIX


def test_plot_nonlinear_sharding_strong_scaling_loads_combined_rows(tmp_path: Path) -> None:
    mod = _load_tool_module()
    payload = {
        "backend": "cpu",
        "grid": {"Nx": 24, "Ny_requested": 48, "Nz": 96, "Nl": 4, "Nm": 8},
        "identity_passed": True,
        "speedup_passed": False,
        "speedup_blockers": ["cpu_2devices_speedup_0.8_below_1"],
        "rows": [
            {
                "backend": "cpu",
                "requested_devices": 1,
                "actual_devices": 1,
                "best_spec": "kx",
                "state_sharding_active": False,
                "identity_gate_pass": True,
                "parallel_median_s": 2.0,
                "strong_speedup_vs_1_device": 1.0,
                "same_process_speedup": 1.1,
                "max_rel_state_error": 0.0,
                "error": None,
            }
        ],
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary = mod.load_summary([path])

    assert summary["identity_passed"] is True
    assert summary["speedup_passed"] is False
    assert summary["status"] == "diagnostic_identity_only"
    assert summary["speedup_blockers"] == ["cpu:cpu_2devices_speedup_0.8_below_1"]
    assert summary["rows"][0]["grid_label"] == "Nx=24, Ny=48, Nz=96, Nl=4, Nm=8"
    assert summary["rows"][0]["source"] == str(path)
