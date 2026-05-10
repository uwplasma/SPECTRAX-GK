from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "profile_linear_rhs_terms.py"
spec = importlib.util.spec_from_file_location("profile_linear_rhs_terms", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_build_summary_reports_dominant_and_zero_norm_terms() -> None:
    payload = mod._build_summary(
        [
            {"term": "field_solve", "seconds": 0.1, "norm": 1.0},
            {"term": "streaming", "seconds": 0.2, "norm": 2.0},
            {"term": "collisions", "seconds": 0.4, "norm": 0.0},
            {"term": "linked_abs_kz", "seconds": 0.3, "norm": 1.0e-16},
            {"term": "full_linear_rhs", "seconds": 1.2, "norm": 3.0},
        ],
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml",
        ky=0.3,
        kx=None,
        nl=4,
        nm=8,
        repeats=5,
        backend="cpu",
        state="z_wave",
        z_variation_norm=0.25,
    )

    assert payload["kind"] == "linear_rhs_terms_profile_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear"
    assert payload["state"] == "z_wave"
    assert payload["z_variation_norm"] == 0.25
    assert payload["dominant_measured_term"] == "collisions"
    assert payload["dominant_nonzero_norm_term"] == "streaming"
    assert payload["zero_norm_terms_by_time"][0]["term"] == "collisions"
    assert payload["full_over_sum_independently_measured_components"] == 1.2
    assert "initial state only" in payload["claim_scope"]


def test_write_summary_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    mod._write_summary_json({"kind": "linear", "value": 2.0}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"kind": "linear", "value": 2.0}


def test_inject_z_wave_adds_parallel_variation() -> None:
    state = jnp.zeros((1, 4, 3, 2, 1, 5), dtype=jnp.complex64)

    out = mod._inject_z_wave(state, ky_index=1, kx_index=0, amplitude=0.2, z_mode=1)

    assert mod._z_variation_norm(out) > 0.0
    assert jnp.linalg.norm(out[0, 0, 2, 1, 0]) > 0.0


def test_hypercollision_kz_source_uses_nu_hyper_m_path() -> None:
    state = jnp.zeros((1, 2, 4, 1, 1, 4), dtype=jnp.complex64)
    state = state.at[0, 0, 3, 0, 0, :].set(jnp.asarray([0.0, 1.0, 0.0, -1.0], dtype=jnp.complex64))
    mask_kz = jnp.ones((2, 4, 1, 1, 1), dtype=bool)
    m_pow = jnp.ones((2, 4, 1, 1, 1), dtype=jnp.float32)

    source = mod._hypercollision_kz_source(
        state,
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(1.0, dtype=jnp.float32),
        nu_hyper_m=jnp.asarray(1.0, dtype=jnp.float32),
        m_norm_kz_factor=jnp.asarray(0.5, dtype=jnp.float32),
        vth=jnp.asarray([2.0], dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        mask_kz=mask_kz,
        m_pow=m_pow,
    )

    assert jnp.linalg.norm(source) > 0.0
    assert jnp.allclose(source[0, 0, 3, 0, 0, 1], -2.3 + 0.0j)


def test_tracked_miller_profile_is_active_state_artifact() -> None:
    path = ROOT / "docs" / "_static" / "linear_rhs_terms_profile_miller_cpu.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["kind"] == "linear_rhs_terms_profile_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["state"] == "z_wave_linear_kick"
    assert payload["full_linear_rhs_seconds"] > 0.0
    assert payload["rows"]["streaming"]["norm"] > 0.0
    assert payload["dominant_nonzero_norm_term"] == "streaming"
