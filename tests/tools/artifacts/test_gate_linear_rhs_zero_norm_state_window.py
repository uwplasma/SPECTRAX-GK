from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "artifacts" / "gate_linear_rhs_zero_norm_state_window.py"
spec = importlib.util.spec_from_file_location(
    "gate_linear_rhs_zero_norm_state_window", SCRIPT
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_inject_z_wave_targets_resolved_hermite_mode() -> None:
    state = jnp.zeros((2, 4, 3, 2, 5), dtype=jnp.complex64)

    out = mod._inject_z_wave(state, ky_index=1, kx_index=0, amplitude=0.25, z_mode=1)

    assert jnp.linalg.norm(out[0, 3, 1, 0]) > 0.0
    assert jnp.linalg.norm(out.at[0, 3, 1, 0, :].set(0.0)) == 0.0


def test_build_summary_accepts_collision_skip_and_rejects_hypercollision_skip() -> None:
    rows = [
        {
            "state": "initial",
            "term_norms": {"collisions": 0.0, "hypercollisions": 0.0},
            "relative_skip_errors": {"collisions": 0.0, "hypercollisions": 0.0},
        },
        {
            "state": "z_wave",
            "term_norms": {"collisions": 0.0, "hypercollisions": 2.0e-4},
            "relative_skip_errors": {"collisions": 0.0, "hypercollisions": 3.0e-3},
        },
    ]

    payload = mod._build_summary(
        rows,
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml",
        ky=0.3,
        kx=None,
        nl=4,
        nm=8,
        identity_threshold=1.0e-10,
        activation_threshold=1.0e-8,
    )

    assert payload["kind"] == "linear_rhs_zero_norm_state_window_gate"
    assert payload["passed"] is True
    assert payload["candidates"]["collisions"]["safe_to_disable_over_window"] is True
    assert (
        payload["candidates"]["hypercollisions"]["safe_to_disable_over_window"] is False
    )
    assert payload["candidates"]["hypercollisions"]["active_states"] == ["z_wave"]
