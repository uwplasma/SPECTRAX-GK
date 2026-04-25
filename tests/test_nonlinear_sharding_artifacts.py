from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
