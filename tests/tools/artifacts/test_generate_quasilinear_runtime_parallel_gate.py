from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectraxgk.workflows.runtime.results import RuntimeLinearScanResult
from tools.artifacts import generate_quasilinear_runtime_parallel_gate as gate


def _scan(ky_values: np.ndarray, *, workers: int) -> RuntimeLinearScanResult:
    ql = []
    gamma = np.asarray(ky_values, dtype=float) + 1.0
    omega = -(np.asarray(ky_values, dtype=float) + 2.0)
    for ky, gamma_i, omega_i in zip(ky_values, gamma, omega, strict=True):
        ql.append(
            {
                "ky": float(ky),
                "gamma": float(gamma_i),
                "omega": float(omega_i),
                "kperp_eff2": 0.5 + float(ky),
                "heat_flux_weight_total": 2.0 * float(ky),
                "particle_flux_weight_total": 0.1 * float(ky),
                "amplitude2": 0.3 * float(ky),
                "saturated_heat_flux_total": 0.6 * float(ky),
                "saturated_particle_flux_total": 0.03 * float(ky),
            }
        )
    return RuntimeLinearScanResult(
        ky=np.asarray(ky_values, dtype=float),
        gamma=gamma,
        omega=omega,
        quasilinear=tuple(ql),
        parallel={
            "requested_workers": int(workers),
            "effective_workers": min(int(workers), len(ky_values)),
            "executor": "thread",
            "identity_contract": "test",
            "quasilinear_state_extraction": True,
        },
    )


def test_quasilinear_runtime_parallel_gate_builds_identity_summary(monkeypatch) -> None:
    calls: list[int] = []

    def fake_timed_scan(_cfg, ky_values, *, workers, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(int(workers))
        return _scan(
            np.asarray(ky_values, dtype=float), workers=workers
        ), 4.0 if workers == 1 else 2.0

    monkeypatch.setattr(gate, "_timed_runtime_scan", fake_timed_scan)
    summary = gate.build_quasilinear_runtime_parallel_gate(
        ky_values=np.asarray([0.1, 0.2]),
        workers=2,
        rtol=1.0e-12,
        atol=1.0e-12,
        solver="krylov",
        nx=1,
        ny=8,
        nz=12,
        nlaguerre=2,
        nhermite=2,
    )

    assert calls == [1, 2]
    assert summary["identity_passed"] is True
    assert summary["observed_speedup"] == 2.0
    assert summary["serial_parallel_metadata"]["requested_workers"] == 2
    assert len(summary["rows"]) == 2
    assert summary["rows"][0]["heat_flux_weight_total_abs_error"] == 0.0


def test_quasilinear_runtime_parallel_gate_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "identity_passed": True,
        "observed_speedup": 2.0,
        "atol": 1.0e-12,
        "rows": [
            {
                "ky": 0.1,
                "serial_heat_flux_weight_total": 0.2,
                "parallel_heat_flux_weight_total": 0.2,
                "heat_flux_weight_total_abs_error": 0.0,
                "serial_saturated_heat_flux_total": 0.06,
                "parallel_saturated_heat_flux_total": 0.06,
                "saturated_heat_flux_total_abs_error": 0.0,
            }
        ],
    }
    out = tmp_path / "ql_parallel_gate"
    paths = gate.write_artifacts(summary, out)

    assert (
        json.loads(Path(paths["json"]).read_text(encoding="utf-8"))["identity_passed"]
        is True
    )
    assert "heat_flux_weight_total_abs_error" in Path(paths["csv"]).read_text(
        encoding="utf-8"
    )
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
