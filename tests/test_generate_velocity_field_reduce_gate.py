from __future__ import annotations

import json
from pathlib import Path

from tools import generate_velocity_field_reduce_gate as gate


def test_velocity_field_reduce_gate_builds_identity_summary(monkeypatch) -> None:
    class FakePlan:
        def to_dict(self):
            return {
                "state_shape": (1, 4, 2, 1, 1),
                "chunks": {"m": 2},
                "active_axes": ("m",),
                "communication_pattern": "field_reduce_broadcast",
            }

    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 4, 2, 1, 1)
        return FakePlan()

    def fake_state(_shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.arange(8, dtype=jnp.float32).reshape((1, 4, 2, 1, 1))

    def fake_reduce(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.sum(state, axis=1)

    monkeypatch.setattr(gate, "_state", fake_state)
    monkeypatch.setattr("jax.devices", lambda _kind=None: [object(), object()])
    monkeypatch.setattr("spectraxgk.velocity_sharding.build_velocity_sharding_plan", fake_build_plan)
    monkeypatch.setattr("spectraxgk.velocity_sharding.velocity_field_reduce_reference", fake_reduce)
    monkeypatch.setattr("spectraxgk.velocity_sharding.velocity_field_reduce_shard_map", fake_reduce)

    summary = gate.build_velocity_field_reduce_gate(shape=(1, 4, 2, 1, 1), requested_devices=2, atol=1.0e-12)

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert len(summary["rows"]) == 2


def test_velocity_field_reduce_gate_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {
                "ky_index": 0,
                "reduced_real": 1.0,
                "reference_real": 1.0,
                "abs_error": 0.0,
            }
        ],
        "atol": 1.0e-8,
        "identity_passed": True,
    }
    out = tmp_path / "velocity_field_reduce_gate"
    paths = gate.write_artifacts(summary, out)

    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["identity_passed"] is True
    assert "abs_error" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
