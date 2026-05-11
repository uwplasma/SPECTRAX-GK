from __future__ import annotations

import json
from pathlib import Path

from tools import generate_periodic_streaming_microkernel_gate as gate


def test_periodic_streaming_microkernel_gate_builds_identity_summary(monkeypatch) -> None:
    class FakePlan:
        def to_dict(self):
            return {
                "state_shape": (1, 1, 4, 1, 1, 4),
                "chunks": {"m": 2},
                "active_axes": ("m",),
                "communication_pattern": "hermite_ghost_exchange+field_reduce_broadcast",
            }

    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 1, 4, 1, 1, 4)
        return FakePlan()

    def fake_state(shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.ones(shape, dtype=jnp.complex64), jnp.linspace(0.0, 1.0, shape[-1])

    def fake_streaming(state, **_kwargs):  # type: ignore[no-untyped-def]
        return 2.0 * state

    monkeypatch.setattr(gate, "_state", fake_state)
    monkeypatch.setattr(gate, "_production_streaming_term", fake_streaming)
    monkeypatch.setattr("jax.devices", lambda _kind=None: [object(), object()])
    monkeypatch.setattr("spectraxgk.velocity_sharding.build_velocity_sharding_plan", fake_build_plan)
    monkeypatch.setattr("spectraxgk.velocity_sharding.periodic_streaming_reference", fake_streaming)
    monkeypatch.setattr("spectraxgk.velocity_sharding.periodic_streaming_shard_map", lambda state, plan, **kwargs: fake_streaming(state))

    summary = gate.build_periodic_streaming_microkernel_gate(
        shape=(1, 1, 4, 1, 1, 4),
        requested_devices=2,
        vth=1.7,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["max_sharded_abs_error"] == 0.0
    assert summary["max_sharded_rel_error"] == 0.0
    assert len(summary["rows"]) == 4


def test_periodic_streaming_microkernel_gate_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {
                "m": 0,
                "state_abs": 1.0,
                "production_abs": 2.0,
                "sharded_abs": 2.0,
                "abs_error": 0.0,
            }
        ],
        "atol": 1.0e-8,
        "rtol": 1.0e-8,
        "identity_passed": True,
    }
    out = tmp_path / "periodic_streaming_microkernel_gate"
    paths = gate.write_artifacts(summary, out)

    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["identity_passed"] is True
    assert "abs_error" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
