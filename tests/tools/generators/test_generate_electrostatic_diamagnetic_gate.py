from __future__ import annotations

import json
from pathlib import Path

from tools import generate_electrostatic_diamagnetic_gate as gate


def test_electrostatic_diamagnetic_gate_builds_identity_summary(monkeypatch) -> None:
    class FakePlan:
        def to_dict(self):
            return {
                "state_shape": (1, 4, 2, 1, 4),
                "chunks": {"m": 2},
                "active_axes": ("m",),
                "communication_pattern": "hermite_ghost_exchange+field_reduce_broadcast",
            }

    class FakeGrid:
        import numpy as np

        ky = np.asarray([0.0, 0.3])
        z = np.asarray([0.0, 1.0, 2.0, 3.0])

    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        class Cache:
            Jl = jnp.ones((1, 1, 2, 1, 4), dtype=jnp.float32)
            mask0 = jnp.zeros((2, 1, 4), dtype=bool)
            l4 = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
            ky = jnp.asarray([0.0, 0.3], dtype=jnp.float32)

        class Params:
            tau_e = 1.0
            charge_sign = 1.0
            density = 1.0
            tz = 1.0
            R_over_LTi = 6.9
            R_over_Ln = 2.2
            omega_star_scale = 1.0

        return jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64), Cache(), Params(), FakeGrid()

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return 2.0 * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    def fake_phi(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", lambda _kind=None: [object(), object()])
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("spectraxgk.parallel.velocity.build_velocity_sharding_plan", lambda *_args, **_kwargs: FakePlan())
    monkeypatch.setattr("spectraxgk.parallel.velocity.electrostatic_phi_shard_map", fake_phi)
    monkeypatch.setattr("spectraxgk.parallel.velocity.diamagnetic_drive_shard_map", lambda state, *_args, **_kwargs: 2.0 * state)

    summary = gate.build_electrostatic_diamagnetic_gate(
        requested_devices=2,
        nx=1,
        ny=2,
        nz=4,
        nl=1,
        nm=4,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["max_phi_abs_error"] == 0.0
    assert summary["phi_norm"] > 0.0
    assert len(summary["rows"]) == 4


def test_electrostatic_diamagnetic_gate_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {
                "m": 0,
                "serial_norm": 2.0,
                "sharded_norm": 2.0,
                "abs_error": 0.0,
            }
        ],
        "atol": 1.0e-8,
        "rtol": 1.0e-8,
        "identity_passed": True,
    }
    out = tmp_path / "electrostatic_diamagnetic_gate"
    paths = gate.write_artifacts(summary, out)

    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["identity_passed"] is True
    assert "abs_error" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
