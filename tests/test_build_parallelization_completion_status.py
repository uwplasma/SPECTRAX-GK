from __future__ import annotations

import json
from pathlib import Path

from tools.build_parallelization_completion_status import ARTIFACTS
from tools.build_parallelization_completion_status import build_status
from tools.build_parallelization_completion_status import write_artifacts


ROOT = Path(__file__).resolve().parents[1]


def _write_artifact(root: Path, name: str, payload: dict) -> None:
    path = root / "docs" / "_static" / ARTIFACTS[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scaling_payload(cpu_speedup: float, gpu_speedup: float) -> dict:
    return {
        "kind": "combined",
        "identity_passed": True,
        "rows": [
            {
                "backend": "cpu",
                "requested_devices": 1,
                "strong_speedup_vs_1_device": 1.0,
                "identity_gate_pass": True,
            },
            {
                "backend": "cpu",
                "requested_devices": 8,
                "strong_speedup_vs_1_device": cpu_speedup,
                "identity_gate_pass": True,
            },
            {
                "backend": "gpu",
                "requested_devices": 1,
                "strong_speedup_vs_1_device": 1.0,
                "identity_gate_pass": True,
            },
            {
                "backend": "gpu",
                "requested_devices": 2,
                "strong_speedup_vs_1_device": gpu_speedup,
                "identity_gate_pass": True,
            },
        ],
    }


def _write_minimal_status_inputs(root: Path, *, cpu_speedup: float, gpu_speedup: float) -> None:
    _write_artifact(root, "independent_ky_scan", _scaling_payload(cpu_speedup, gpu_speedup))
    _write_artifact(root, "quasilinear_uq_ensemble", _scaling_payload(cpu_speedup, gpu_speedup))
    _write_artifact(
        root,
        "whole_state_nonlinear_sharding",
        {
            "identity_passed": True,
            "claim_scope": "engineering evidence, not a production speedup claim",
            "rows": [
                {
                    "backend": "cpu",
                    "requested_devices": 1,
                    "strong_speedup_vs_1_device": 1.0,
                    "identity_gate_pass": True,
                }
            ],
        },
    )
    _write_artifact(
        root,
        "fft_axis_domain",
        {
            "gate": {"identity_passed": True},
            "claim_scope": "diagnostic only, no production routing or speedup claim",
            "rows": [{"identity_passed": True}],
        },
    )


def test_parallelization_completion_status_closes_production_lanes() -> None:
    status = build_status(ROOT)

    assert status["passed"] is True
    assert status["production_completion_percent"] == 100.0
    lanes = {lane["lane"]: lane for lane in status["lanes"]}
    assert lanes["independent_ky_scan"]["status"] == "production_closed"
    assert lanes["independent_ky_scan"]["best_speedups"]["cpu"] >= 5.0
    assert lanes["independent_ky_scan"]["best_speedups"]["gpu"] >= 1.5
    assert lanes["quasilinear_uq_ensemble"]["status"] == "production_closed"
    assert lanes["whole_state_nonlinear_sharding"]["status"] == "diagnostic_closed_not_production"
    assert lanes["fft_axis_domain"]["status"] == "diagnostic_identity_closed"
    assert "Whole-state nonlinear sharding" in status["claim_scope"]


def test_parallelization_completion_status_rejects_weak_production_speedup(tmp_path: Path) -> None:
    _write_minimal_status_inputs(tmp_path, cpu_speedup=4.0, gpu_speedup=1.6)

    status = build_status(tmp_path)

    assert status["passed"] is False
    assert status["production_completion_percent"] == 0.0
    assert {lane["status"] for lane in status["lanes"] if lane["claim_level"] == "production_parallelization"} == {
        "open"
    }


def test_parallelization_completion_status_writes_json_and_figures(tmp_path: Path) -> None:
    _write_minimal_status_inputs(tmp_path, cpu_speedup=6.0, gpu_speedup=1.7)
    status = build_status(tmp_path)

    paths = write_artifacts(status, tmp_path / "parallelization_completion_status")

    for path in paths.values():
        assert Path(path).exists()
