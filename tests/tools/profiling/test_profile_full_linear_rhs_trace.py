from __future__ import annotations

from support.paths import load_profiling_tool
import json
from pathlib import Path

import jax.numpy as jnp


mod = load_profiling_tool("profile_full_linear_rhs_trace")


def test_hlo_token_counts_are_coarse_but_stable() -> None:
    hlo = """
    ROOT fusion.1 = f32[2] fusion(arg), kind=kLoop
    fft.2 = c64[2] fft(arg), fft_type=FFT
    scatter.3 = f32[2] scatter(arg)
    """
    counts = mod._hlo_token_counts(hlo)

    assert counts["fusion"] >= 1
    assert counts["fft"] >= 2
    assert counts["scatter"] >= 1
    assert counts["gather"] == 0


def test_build_summary_contains_trace_metadata() -> None:
    payload = mod._build_summary(
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml",
        backend="cpu",
        nl=4,
        nm=8,
        repeats=3,
        state="z_wave",
        z_variation_norm=0.2,
        compile_execute_seconds=1.5,
        warm_seconds=0.1,
        rhs_norm=2.0,
        phi_norm=3.0,
        hlo_text="ROOT add.1 = f32[] add(a, b)\n",
        trace_dir=Path("tools_out/trace"),
        memory_profile=Path("tools_out/memory.prof"),
        hlo_out=Path("tools_out/hlo.txt"),
        force_electrostatic_fields=True,
        source="spectraxgk.linear.linear_rhs_cached",
    )

    assert payload["kind"] == "full_linear_rhs_trace_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["backend"] == "cpu"
    assert payload["warm_seconds"] == 0.1
    assert payload["hlo_token_counts"]["add"] >= 1
    assert payload["force_electrostatic_fields"] is True
    assert payload["source"] == "spectraxgk.linear.linear_rhs_cached"
    assert payload["trace_dir"] == "tools_out/trace"
    assert "kernel-level optimization targets" in payload["claim_scope"]


def test_write_summary_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    mod._write_summary_json({"kind": "full", "value": 2.0}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "kind": "full",
        "value": 2.0,
    }


def test_inject_z_wave_adds_parallel_variation() -> None:
    state = jnp.zeros((1, 4, 3, 2, 1, 5), dtype=jnp.complex64)

    out = mod._inject_z_wave(state, ky_index=1, kx_index=0, amplitude=0.2, z_mode=1)

    assert mod._z_variation_norm(out) > 0.0
    assert jnp.linalg.norm(out[0, 0, 2, 1, 0]) > 0.0
