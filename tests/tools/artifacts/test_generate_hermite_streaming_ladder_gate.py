from __future__ import annotations

import json
from pathlib import Path

from tools.artifacts import generate_hermite_streaming_ladder_gate as gate


def test_hermite_streaming_ladder_gate_builds_identity_summary(monkeypatch) -> None:
    class FakePlan:
        def to_dict(self):
            return {
                "state_shape": (1, 4, 1, 1, 1),
                "chunks": {"m": 2},
                "active_axes": ("m",),
                "communication_pattern": "hermite_ghost_exchange+field_reduce_broadcast",
            }

    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 4, 1, 1, 1)
        return FakePlan()

    def fake_state(_shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.asarray([[[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]])

    def fake_ladder(state, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.sqrt(state + 1.0)

    def fake_reduce(state, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.sum(state, axis=1)

    monkeypatch.setattr(gate, "_state", fake_state)
    monkeypatch.setattr("jax.devices", lambda _kind=None: [object(), object()])
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_streaming_ladder_reference", fake_ladder
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_streaming_ladder_shard_map",
        lambda state, plan, **kwargs: fake_ladder(state),
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.velocity_field_reduce_reference", fake_reduce
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.velocity_field_reduce_shard_map",
        lambda state, plan, **kwargs: fake_reduce(state),
    )

    summary = gate.build_hermite_streaming_ladder_gate(
        shape=(1, 4, 1, 1, 1),
        requested_devices=2,
        vth=1.7,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["max_ladder_abs_error"] == 0.0
    assert summary["max_ladder_rel_error"] == 0.0
    assert summary["max_reduction_abs_error"] == 0.0
    assert len(summary["rows"]) == 4


def test_hermite_streaming_ladder_gate_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {
                "m": 0,
                "state_real": 1.0,
                "ladder_real": 2.0,
                "reference_real": 2.0,
                "abs_error": 0.0,
            }
        ],
        "atol": 1.0e-8,
        "rtol": 1.0e-8,
        "identity_passed": True,
    }
    out = tmp_path / "hermite_streaming_ladder_gate"
    paths = gate.write_artifacts(summary, out)

    assert (
        json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))[
            "identity_passed"
        ]
        is True
    )
    assert "abs_error" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
