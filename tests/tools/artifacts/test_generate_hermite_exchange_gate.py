from __future__ import annotations

import json
from pathlib import Path

from tools.artifacts import generate_hermite_exchange_gate as gate


def test_hermite_exchange_gate_builds_identity_summary(monkeypatch) -> None:
    class FakePlan:
        def to_dict(self):
            return {
                "state_shape": (1, 4, 1, 1, 1),
                "chunks": {"m": 2},
                "active_axes": ("m",),
                "communication_pattern": "hermite_ghost_exchange",
            }

    def fake_build_plan(shape, **_kwargs):  # type: ignore[no-untyped-def]
        assert shape == (1, 4, 1, 1, 1)
        return FakePlan()

    def fake_state(_shape):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.asarray([[[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]])

    def fake_reference(state):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.asarray([[[[[0.0]]], [[[1.0]]], [[[2.0]]], [[[3.0]]]]]), jnp.asarray(
            [[[[[2.0]]], [[[3.0]]], [[[4.0]]], [[[0.0]]]]]
        )

    monkeypatch.setattr(gate, "_state", fake_state)
    monkeypatch.setattr("jax.devices", lambda _kind=None: [object(), object()])
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.build_velocity_sharding_plan", fake_build_plan
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_neighbor_reference", fake_reference
    )
    monkeypatch.setattr(
        "spectraxgk.parallel.velocity.hermite_neighbor_shard_map",
        lambda state, plan, devices: fake_reference(state),
    )

    summary = gate.build_hermite_exchange_gate(
        shape=(1, 4, 1, 1, 1), requested_devices=2, atol=1.0e-12
    )

    assert summary["identity_passed"] is True
    assert summary["plan"]["communication_pattern"] == "hermite_ghost_exchange"
    assert summary["max_lower_abs_error"] == 0.0
    assert summary["max_upper_abs_error"] == 0.0
    assert len(summary["rows"]) == 4


def test_hermite_exchange_gate_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {
                "m": 0,
                "center_real": 1.0,
                "lower_real": 0.0,
                "upper_real": 2.0,
                "lower_reference_real": 0.0,
                "upper_reference_real": 2.0,
                "lower_abs_error": 0.0,
                "upper_abs_error": 0.0,
            }
        ],
        "atol": 1.0e-8,
        "identity_passed": True,
    }
    out = tmp_path / "hermite_exchange_gate"
    paths = gate.write_artifacts(summary, out)

    assert (
        json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))[
            "identity_passed"
        ]
        is True
    )
    assert "lower_abs_error" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
