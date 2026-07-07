from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "artifacts"
        / "plot_quasilinear_uq_ensemble_scaling.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_quasilinear_uq_ensemble_scaling", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_payload(path: Path, *, backend: str) -> None:
    payload = {
        "backend": backend,
        "claim_scope": "test",
        "grid": {"Nx": 1, "Ny": 8, "Nz": 8, "Nl": 2, "Nm": 3},
        "time": {"dt": 0.02, "steps": 10},
        "identity_passed": True,
        "rows": [
            {
                "requested_devices": 1,
                "actual_workers": 1,
                "timed_wall_s": 2.0,
                "strong_speedup_vs_1_device": 1.0,
                "parallel_efficiency": 1.0,
                "ensemble_mean_heat_flux_proxy": 1.2,
                "ensemble_std_heat_flux_proxy": 0.1,
                "max_heat_flux_proxy_rel_error": 0.0,
                "max_gamma_abs_error": 0.0,
                "identity_gate_pass": True,
                "error": None,
            },
            {
                "requested_devices": 2,
                "actual_workers": 2,
                "timed_wall_s": 1.1,
                "strong_speedup_vs_1_device": 1.8,
                "parallel_efficiency": 0.9,
                "ensemble_mean_heat_flux_proxy": 1.2,
                "ensemble_std_heat_flux_proxy": 0.1,
                "max_heat_flux_proxy_rel_error": 0.0,
                "max_gamma_abs_error": 0.0,
                "identity_gate_pass": True,
                "error": None,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_summary_combines_cpu_and_gpu_payloads(tmp_path: Path) -> None:
    mod = _load_tool_module()
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    _write_payload(cpu, backend="cpu")
    _write_payload(gpu, backend="gpu")

    summary = mod.load_summary([cpu, gpu])

    assert summary["identity_passed"] is True
    assert summary["kind"] == "quasilinear_uq_ensemble_scaling_combined"
    assert {row["backend"] for row in summary["rows"]} == {"cpu", "gpu"}
    assert all("Nx=1" in row["grid_label"] for row in summary["rows"])


def test_write_artifacts_creates_combined_outputs(tmp_path: Path) -> None:
    mod = _load_tool_module()
    cpu = tmp_path / "cpu.json"
    gpu = tmp_path / "gpu.json"
    _write_payload(cpu, backend="cpu")
    _write_payload(gpu, backend="gpu")
    summary = mod.load_summary([cpu, gpu])

    paths = mod.write_artifacts(summary, tmp_path / "combined")

    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
