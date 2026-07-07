from __future__ import annotations

import json

from support.paths import REPO_ROOT


ROOT = REPO_ROOT


def _load_static_json(name: str) -> dict:
    return json.loads((ROOT / "docs" / "_static" / name).read_text(encoding="utf-8"))


def _assert_identity_artifact(payload: dict) -> None:
    assert payload["identity_gate_pass"] is True
    assert payload["sharding_options"] == ["auto", "kx"]
    assert "Do not use as a published runtime claim" in payload["claim_scope"]
    for axis in payload["sharding_options"]:
        result = payload["sharded_results"][axis]
        assert result["identity_gate_pass"] is True
        assert result["error"] is None
        assert result["max_abs_state_error"] == 0.0
        assert result["max_rel_state_error"] == 0.0
        assert result["diagnostic_identity_gate_pass"] is True
        assert result["max_abs_phi_error"] == 0.0
        assert result["max_rel_phi_error"] == 0.0
        assert result["max_abs_rhs_error"] == 0.0
        assert result["max_rel_rhs_error"] == 0.0


def test_local_nonlinear_sharding_profile_is_identity_gated() -> None:
    payload = _load_static_json("nonlinear_sharding_profile.json")

    _assert_identity_artifact(payload)
    assert payload["default_backend"] == "cpu"
    assert payload["state_sharding_active"] is False


def test_office_gpu_nonlinear_sharding_profile_is_active_and_identity_gated() -> None:
    payload = _load_static_json("nonlinear_sharding_profile_office_gpu.json")

    _assert_identity_artifact(payload)
    assert payload["default_backend"] == "gpu"
    assert payload["device_count"] >= 2
    assert payload["state_sharding_active"] is True
    assert payload["profiler_trace"]["requested"] is True


def test_device_z_transport_window_profiles_are_identity_gated_and_scoped() -> None:
    cpu = _load_static_json("nonlinear_device_z_pencil_transport_cpu4_profile.json")
    gpu = _load_static_json("nonlinear_device_z_pencil_transport_gpu2_profile.json")

    for payload in (cpu, gpu):
        assert payload["kind"] == "nonlinear_device_z_pencil_transport_window_profile"
        assert (
            "not yet a full production nonlinear turbulent-transport solve"
            in payload["claim_scope"]
        )
        assert payload["summary"]["all_active_identity_passed"] is True
        assert payload["summary"]["full_solver_speedup_claim_allowed"] is False
        assert payload["shape"] == [4, 16, 96, 96, 32]
        assert payload["steps"] == 4
        active_rows = [
            row for row in payload["rows"] if row["active"] and row["device_count"] > 1
        ]
        assert active_rows
        for row in active_rows:
            assert row["identity_passed"] is True
            assert row["transport_window_identity_passed"] is True
            assert row["final_state_max_abs_error"] <= payload["atol"]
            assert row["physical_flux_trace_max_abs_error"] <= payload["atol"]
            assert row["transport_window_report"]["identity_passed"] is True

    assert cpu["backend"] == "cpu"
    assert cpu["summary"]["transport_window_speedup_claim_allowed"] is True
    assert cpu["summary"]["max_speedup_vs_serial"] >= 1.5
    assert cpu["hlo"]["device_4"]["all_to_all"] == 0
    assert cpu["hlo"]["device_4"]["collective_permute"] == 0
    assert cpu["trace"]["requested"] is True

    assert gpu["backend"] == "gpu"
    assert gpu["summary"]["transport_window_speedup_claim_allowed"] is False
    assert gpu["rows"][1]["blocked_reasons"] == ["speedup_below_gate"]
    assert gpu["hlo"]["device_2"]["all_to_all"] == 0
    assert gpu["hlo"]["device_2"]["collective_permute"] == 0
    assert gpu["trace"]["requested"] is True
