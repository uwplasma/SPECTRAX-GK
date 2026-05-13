from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "docs" / "_static" / name).read_text(encoding="utf-8"))


def _assert_scaling_artifact(payload: dict) -> None:
    assert payload["identity_passed"] is True
    assert "independent ky" in payload["claim_scope"].lower()
    assert payload["grid"] == {"Nl": 4, "Nm": 8, "Nx": 1, "Ny": 128, "Nz": 96}
    assert payload["time"]["steps"] == 240
    assert len(payload["ky"]) == 64
    for row in payload["rows"]:
        assert row["identity_gate_pass"] is True
        assert row["max_gamma_rel_error"] == 0.0
        assert row["max_omega_abs_error"] == 0.0
        assert row["strong_speedup_vs_1_device"] > 0.0


def test_large_cpu_independent_ky_scaling_artifact_is_identity_gated() -> None:
    payload = _load("independent_ky_scan_scaling_cpu_large.json")

    _assert_scaling_artifact(payload)
    assert payload["backend"] == "cpu"
    assert [row["requested_devices"] for row in payload["rows"]] == [1, 2, 4, 8]
    assert payload["rows"][-1]["strong_speedup_vs_1_device"] > 7.0


def test_large_gpu_independent_ky_scaling_artifact_is_identity_gated() -> None:
    payload = _load("independent_ky_scan_scaling_gpu_large.json")

    _assert_scaling_artifact(payload)
    assert payload["backend"] == "gpu"
    assert [row["requested_devices"] for row in payload["rows"]] == [1, 2]
    assert payload["rows"][-1]["strong_speedup_vs_1_device"] > 1.8


def test_combined_independent_ky_scaling_artifact_tracks_cpu_and_gpu() -> None:
    payload = _load("independent_ky_scan_scaling_large.json")

    assert payload["identity_passed"] is True
    assert {row["backend"] for row in payload["rows"]} == {"cpu", "gpu"}
    assert "not a nonlinear domain-decomposition" in payload["claim_scope"]
