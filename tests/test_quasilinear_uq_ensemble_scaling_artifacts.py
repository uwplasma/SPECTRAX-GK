from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"


def _load(name: str) -> dict:
    return json.loads((STATIC / name).read_text(encoding="utf-8"))


def test_quasilinear_uq_cpu_large_artifact_has_identity_and_speedup() -> None:
    data = _load("quasilinear_uq_ensemble_scaling_cpu_large.json")

    assert data["identity_passed"] is True
    assert data["grid"] == {"Nx": 1, "Ny": 96, "Nz": 64, "Nl": 3, "Nm": 6}
    assert data["time"]["steps"] == 2000
    assert data["time"]["fit_start_fraction"] == 0.5
    assert data["time"]["fit_end_fraction"] == 0.95
    assert data["gradients"] == [2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
    assert data["ky"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert [row["requested_devices"] for row in data["rows"]] == [1, 2, 4, 8]
    assert all(row["identity_gate_pass"] for row in data["rows"])
    assert min(row["ensemble_mean_heat_flux_proxy"] for row in data["rows"]) > 1.0
    assert data["rows"][-1]["actual_workers"] == 6
    assert data["rows"][-1]["strong_speedup_vs_1_device"] > 5.0


def test_quasilinear_uq_gpu_large_artifact_has_identity_and_speedup() -> None:
    data = _load("quasilinear_uq_ensemble_scaling_gpu_large.json")

    assert data["identity_passed"] is True
    assert data["grid"] == {"Nx": 1, "Ny": 96, "Nz": 64, "Nl": 3, "Nm": 6}
    assert [row["requested_devices"] for row in data["rows"]] == [1, 2]
    assert all(row["identity_gate_pass"] for row in data["rows"])
    assert min(row["ensemble_mean_heat_flux_proxy"] for row in data["rows"]) > 1.0
    assert data["rows"][-1]["strong_speedup_vs_1_device"] > 1.5


def test_quasilinear_uq_combined_artifact_tracks_both_backends() -> None:
    data = _load("quasilinear_uq_ensemble_scaling_large.json")

    assert data["identity_passed"] is True
    assert data["kind"] == "quasilinear_uq_ensemble_scaling_combined"
    assert {row["backend"] for row in data["rows"]} == {"cpu", "gpu"}
    assert "not a promoted absolute nonlinear heat-flux predictor" in data["claim_scope"]
