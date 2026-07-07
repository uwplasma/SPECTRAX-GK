from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from tools.build_parallelization_completion_status import ARTIFACTS
from tools.build_parallelization_completion_status import build_status
from tools.build_parallelization_completion_status import write_artifacts


ROOT = Path(__file__).resolve().parents[3]


def _write_artifact(root: Path, name: str, payload: dict) -> None:
    path = root / "docs" / "_static" / ARTIFACTS[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scaling_payload(cpu_speedup: float, gpu_speedup: float) -> dict:
    return {
        "kind": "combined",
        "claim_scope": (
            "solver-backed independent ky scan strong-scaling artifact for CPU processes and GPU workers; "
            "not a nonlinear domain-decomposition speedup claim"
        ),
        "identity_passed": True,
        "inputs": [{"backend": "cpu"}, {"backend": "gpu"}],
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
    independent = _scaling_payload(cpu_speedup, gpu_speedup)
    independent["kind"] = "independent_ky_scan_scaling_combined"
    _write_artifact(root, "independent_ky_scan", independent)
    uq = _scaling_payload(cpu_speedup, gpu_speedup)
    uq["kind"] = "quasilinear_uq_ensemble_scaling_combined"
    uq["claim_scope"] = (
        "solver-backed quasilinear/UQ ensemble strong-scaling artifact for independent CPU processes "
        "and GPU workers; not a promoted absolute nonlinear heat-flux predictor"
    )
    _write_artifact(root, "quasilinear_uq_ensemble", uq)
    _write_artifact(
        root,
        "whole_state_nonlinear_sharding",
        {
            "kind": "nonlinear_sharding_strong_scaling_combined",
            "identity_passed": True,
            "claim_scope": "whole-state sharding engineering evidence, not a production speedup claim",
            "inputs": [{"backend": "cpu"}, {"backend": "gpu"}],
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
            "kind": "nonlinear_spectral_communication_identity_gate",
            "gate": {"identity_passed": True},
            "claim_scope": (
                "diagnostic nonlinear spectral communication, RHS, fixed-step integrator, "
                "pencil fused-bracket, and physical transport-window identity gate; "
                "no production distributed FFT routing or speedup claim"
            ),
            "rows": [{"identity_passed": True}],
        },
    )


def test_parallelization_completion_status_closes_production_lanes() -> None:
    status = build_status(ROOT)
    production_gate = json.loads(
        (
            ROOT
            / "docs"
            / "_static"
            / "nonlinear_sharding_production_speedup_gate.json"
        ).read_text(encoding="utf-8")
    )

    assert status["passed"] is True
    assert status["production_completion_percent"] == 100.0
    assert status["independent_ensemble_provenance_gate"]["passed"] is True
    assert (
        status["independent_ensemble_provenance_gate"]["workload"]
        == "optimization_ensemble"
    )
    assert (
        status["independent_ensemble_provenance_gate"]["parallel_indices"]
        == status["independent_ensemble_provenance_gate"]["serial_indices"]
    )
    assert (
        status["independent_ensemble_provenance_gate"][
            "exception_metadata_passed"
        ]
        is True
    )
    lanes = {lane["lane"]: lane for lane in status["lanes"]}
    assert lanes["independent_ky_scan"]["status"] == "production_closed"
    assert lanes["independent_ky_scan"]["source_contract"]["claim_separation_passed"] is True
    assert lanes["independent_ky_scan"]["source_contract"]["input_backends"] == ["cpu", "gpu"]
    assert lanes["independent_ky_scan"]["best_speedups"]["cpu"] >= 5.0
    assert lanes["independent_ky_scan"]["best_speedups"]["gpu"] >= 1.5
    assert lanes["quasilinear_uq_ensemble"]["status"] == "production_closed"
    assert lanes["whole_state_nonlinear_sharding"]["status"] == "diagnostic_closed_not_production"
    assert lanes["whole_state_nonlinear_sharding"]["source_contract"]["claim_separation_passed"] is True
    assert production_gate["production_speedup_claim_allowed"] is False
    assert production_gate["status"] == "diagnostic_only"
    assert "gpu_production_speedup_candidate_missing" in production_gate["blockers"]
    assert (
        lanes["whole_state_nonlinear_sharding"]["status"]
        == "diagnostic_closed_not_production"
    )
    assert lanes["fft_axis_domain"]["status"] == "diagnostic_identity_closed"
    assert "Whole-state nonlinear sharding" in status["claim_scope"]
    assert "exception metadata" in status["claim_scope"]


def test_parallelization_completion_status_rejects_weak_production_speedup(tmp_path: Path) -> None:
    _write_minimal_status_inputs(tmp_path, cpu_speedup=4.0, gpu_speedup=1.6)

    status = build_status(tmp_path)

    assert status["passed"] is False
    assert status["production_completion_percent"] == 0.0
    assert {lane["status"] for lane in status["lanes"] if lane["claim_level"] == "production_parallelization"} == {
        "open"
    }


def test_parallelization_completion_status_rejects_ambiguous_claim_separation(tmp_path: Path) -> None:
    _write_minimal_status_inputs(tmp_path, cpu_speedup=6.0, gpu_speedup=1.7)
    path = tmp_path / "docs" / "_static" / ARTIFACTS["whole_state_nonlinear_sharding"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["claim_scope"] = "large nonlinear sharding production speedup artifact"
    path.write_text(json.dumps(payload), encoding="utf-8")

    status = build_status(tmp_path)

    lanes = {lane["lane"]: lane for lane in status["lanes"]}
    assert status["passed"] is False
    assert lanes["whole_state_nonlinear_sharding"]["status"] == "open"
    assert lanes["whole_state_nonlinear_sharding"]["source_contract"]["missing_scope_phrases"]


def test_parallelization_completion_status_writes_json_and_figures(tmp_path: Path) -> None:
    _write_minimal_status_inputs(tmp_path, cpu_speedup=6.0, gpu_speedup=1.7)
    status = build_status(tmp_path)

    paths = write_artifacts(status, tmp_path / "parallelization_completion_status")

    for path in paths.values():
        assert Path(path).exists()


def test_parallelization_completion_status_script_runs_without_install(tmp_path: Path) -> None:
    _write_minimal_status_inputs(tmp_path, cpu_speedup=6.0, gpu_speedup=1.7)
    env = dict(os.environ)
    env["PYTHONPATH"] = ""

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_parallelization_completion_status.py"),
            "--root",
            str(tmp_path),
            "--out-prefix",
            str(tmp_path / "status"),
            "--skip-figures",
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "status.json").exists()
