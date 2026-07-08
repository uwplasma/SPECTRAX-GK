from __future__ import annotations

import json
from pathlib import Path

from support.paths import load_artifact_tool


def _load_tool_module():
    return load_artifact_tool("plot_independent_ky_scan_scaling")


def test_plot_independent_ky_scan_scaling_parser_defaults_to_large_inputs() -> None:
    mod = _load_tool_module()

    args = mod.build_parser().parse_args([])

    assert args.inputs == mod.DEFAULT_INPUTS
    assert args.out_prefix == mod.DEFAULT_PREFIX


def test_plot_independent_ky_scan_scaling_loads_rows(tmp_path: Path) -> None:
    mod = _load_tool_module()
    payload = {
        "backend": "cpu",
        "grid": {"Nx": 1, "Ny": 128, "Nz": 96, "Nl": 4, "Nm": 8},
        "time": {"steps": 240},
        "identity_passed": True,
        "rows": [
            {
                "requested_devices": 1,
                "actual_workers": 1,
                "timed_wall_s": 2.0,
                "strong_speedup_vs_1_device": 1.0,
                "parallel_efficiency": 1.0,
                "max_gamma_rel_error": 0.0,
                "max_omega_abs_error": 0.0,
                "identity_gate_pass": True,
                "error": None,
            }
        ],
    }
    path = tmp_path / "cpu.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary = mod.load_summary([path])

    assert summary["identity_passed"] is True
    assert summary["rows"][0]["backend"] == "cpu"
    assert summary["rows"][0]["grid_label"] == "Nx=1, Ny=128, Nz=96, Nl=4, Nm=8"
