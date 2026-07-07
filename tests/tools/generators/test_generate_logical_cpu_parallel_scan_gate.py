from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools import generate_logical_cpu_parallel_scan_gate as gate


def test_logical_cpu_parallel_scan_gate_builds_identity_summary(monkeypatch) -> None:
    def fake_devices(requested_devices: int):  # type: ignore[no-untyped-def]
        return [object()] * requested_devices

    def fake_scan(ky_values, *, batch_size, devices):  # type: ignore[no-untyped-def]
        assert devices
        ky = np.asarray(ky_values, dtype=float)
        return {
            "gamma": ky + 0.1,
            "omega": -ky,
            "kperp2": ky**2 + 0.08,
            "ql_proxy": (ky + 0.1) / (ky**2 + 0.08),
        }, 4.0 if batch_size == 1 else 2.0

    monkeypatch.setattr(gate, "_select_devices", fake_devices)
    monkeypatch.setattr(gate, "_timed_scan_model", fake_scan)

    summary = gate.build_logical_cpu_parallel_scan_gate(
        ky_values=np.asarray([0.1, 0.2]),
        serial_batch=1,
        parallel_batch=2,
        requested_devices=2,
        gamma_rtol=1.0e-12,
        omega_atol=1.0e-12,
        ql_rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["observed_speedup"] == 2.0
    assert summary["device_parallel_config"]["strategy"] == "device_batch"
    assert summary["device_parallel_config"]["num_devices"] == 2
    assert summary["max_ql_rel_error"] == 0.0
    assert len(summary["rows"]) == 2


def test_logical_cpu_parallel_scan_gate_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {
                "ky": 0.1,
                "serial_gamma": 0.2,
                "batched_gamma": 0.2,
                "gamma_rel_error": 0.0,
                "serial_omega": -0.1,
                "batched_omega": -0.1,
                "omega_abs_error": 0.0,
                "serial_ql_proxy": 1.5,
                "batched_ql_proxy": 1.5,
                "ql_rel_error": 0.0,
            }
        ],
        "gamma_rtol": 1.0e-8,
        "omega_atol": 1.0e-8,
        "ql_rtol": 1.0e-8,
        "serial_elapsed_s": 2.0,
        "batched_elapsed_s": 1.0,
        "observed_speedup": 2.0,
        "identity_passed": True,
    }
    out = tmp_path / "logical_cpu_parallel_gate"
    paths = gate.write_artifacts(summary, out)

    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["identity_passed"] is True
    assert "ql_rel_error" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
