from __future__ import annotations

import json
from pathlib import Path

from tools import generate_linear_rhs_electrostatic_slices_gate as gate


def test_linear_rhs_electrostatic_slices_gate_builds_identity_summary(monkeypatch) -> None:
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

        return jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64), object(), object(), FakeGrid()

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return 3.0 * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(gate, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", lambda _kind=None: [object(), object()])
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("spectraxgk.linear.linear_rhs_parallel_cached", fake_rhs)
    monkeypatch.setattr("spectraxgk.parallel.velocity.build_velocity_sharding_plan", lambda *_args, **_kwargs: FakePlan())

    summary = gate.build_linear_rhs_electrostatic_slices_gate(
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


def test_linear_rhs_electrostatic_slices_gate_writes_artifacts(tmp_path: Path) -> None:
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
    out = tmp_path / "linear_rhs_electrostatic_slices_gate"
    paths = gate.write_artifacts(summary, out)

    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["identity_passed"] is True
    assert "abs_error" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
