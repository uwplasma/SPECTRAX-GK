from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectraxgk.benchmarks import CycloneScanResult
from tools import generate_parallel_ky_scan_gate as gate


def test_parallel_ky_scan_gate_builds_identity_summary(monkeypatch) -> None:
    calls: list[int] = []

    def fake_scan(ky_values, *, ky_batch, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(int(ky_batch))
        result = CycloneScanResult(
            ky=np.asarray(ky_values, dtype=float),
            gamma=np.asarray([0.1, 0.2], dtype=float),
            omega=np.asarray([0.3, 0.4], dtype=float),
        )
        return result, 4.0 if ky_batch == 1 else 2.0

    monkeypatch.setattr(gate, "_timed_cyclone_scan", fake_scan)
    summary = gate.build_parallel_ky_scan_gate(
        ky_values=np.asarray([0.1, 0.2]),
        serial_batch=1,
        parallel_batch=2,
        gamma_rtol=1.0e-12,
        omega_atol=1.0e-12,
        steps=4,
        dt=0.1,
        nx=1,
        ny=4,
        nz=8,
        nlaguerre=2,
        nhermite=3,
    )

    assert calls == [1, 2]
    assert summary["identity_passed"] is True
    assert summary["observed_speedup"] == 2.0
    assert summary["max_gamma_rel_error"] == 0.0
    assert len(summary["rows"]) == 2


def test_parallel_ky_scan_gate_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {
                "ky": 0.1,
                "serial_gamma": 0.1,
                "batched_gamma": 0.1,
                "gamma_rel_error": 0.0,
                "serial_omega": 0.2,
                "batched_omega": 0.2,
                "omega_abs_error": 0.0,
            }
        ],
        "gamma_rtol": 1.0e-8,
        "omega_atol": 1.0e-8,
        "serial_elapsed_s": 2.0,
        "batched_elapsed_s": 1.0,
        "observed_speedup": 2.0,
        "identity_passed": True,
    }
    out = tmp_path / "parallel_gate"
    gate.write_artifacts(summary, out)

    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["identity_passed"] is True
    assert "gamma_rel_error" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert out.with_suffix(".png").exists()
    assert out.with_suffix(".pdf").exists()
