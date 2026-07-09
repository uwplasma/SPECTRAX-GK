from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp

from support.paths import load_profiling_tool
from tools.profiling._profiler_options import make_profile_options
from tools.profiling.profile_linear_cache_build import build_low_rank_moment_cache
from tools.profiling.profile_runtime_startup import (
    PhaseTiming,
    _write_phase_csv,
    _write_phase_json,
)

linear_trace = load_profiling_tool("profile_full_linear_rhs_trace")
nonlinear_trace = load_profiling_tool("profile_full_nonlinear_rhs_trace")


def test_make_profile_options_defaults_disable_python_and_host_tracers() -> None:
    opts = make_profile_options()
    assert opts.python_tracer_level == 0
    assert opts.host_tracer_level == 0


def test_make_profile_options_accepts_explicit_levels() -> None:
    opts = make_profile_options(python_tracer_level=1, host_tracer_level=2)
    assert opts.python_tracer_level == 1
    assert opts.host_tracer_level == 2


def test_profile_linear_cache_uses_low_rank_moment_factors() -> None:
    params = SimpleNamespace(
        nu_hermite=0.5,
        nu_laguerre=0.25,
        p_hyper=4,
        p_hyper_l=3,
        p_hyper_m=5,
        p_hyper_lm=2,
    )

    cache = build_low_rank_moment_cache(
        nl=3, nm=4, params=params, real_dtype=jnp.float32
    )

    assert cache["lb_lam"].shape == (3, 4)
    assert cache["collision_lam"].shape == (0,)
    assert cache["hyper_ratio"].shape == (3, 4, 1, 1, 1)
    assert cache["sqrt_p"].shape == (1, 1, 4, 1, 1, 1)
    assert cache["mask_const"].dtype == jnp.bool_


def test_profile_runtime_startup_writes_csv_and_json(tmp_path: Path) -> None:
    phases = [
        PhaseTiming(phase="a", seconds=1.25, note="first"),
        PhaseTiming(phase="b", seconds=2.75, note="second"),
    ]
    csv_path = tmp_path / "startup.csv"
    json_path = tmp_path / "startup.json"

    _write_phase_csv(csv_path, phases)
    _write_phase_json(json_path, phases, {"config": "case.toml", "device_count": 1})

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "phase,seconds,note" in csv_text
    assert "a,1.25,first" in csv_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["config"] == "case.toml"
    assert payload["startup_total_s"] == 4.0
    assert payload["phases"][1]["phase"] == "b"


def test_full_linear_trace_hlo_token_counts_are_coarse_but_stable() -> None:
    hlo = """
    ROOT fusion.1 = f32[2] fusion(arg), kind=kLoop
    fft.2 = c64[2] fft(arg), fft_type=FFT
    scatter.3 = f32[2] scatter(arg)
    """
    counts = linear_trace._hlo_token_counts(hlo)

    assert counts["fusion"] >= 1
    assert counts["fft"] >= 2
    assert counts["scatter"] >= 1
    assert counts["gather"] == 0


def test_full_linear_trace_summary_contains_metadata() -> None:
    payload = linear_trace._build_summary(
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


def test_full_linear_trace_summary_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    linear_trace._write_summary_json({"kind": "full", "value": 2.0}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "kind": "full",
        "value": 2.0,
    }


def test_full_linear_trace_inject_z_wave_adds_parallel_variation() -> None:
    state = jnp.zeros((1, 4, 3, 2, 1, 5), dtype=jnp.complex64)

    out = linear_trace._inject_z_wave(
        state, ky_index=1, kx_index=0, amplitude=0.2, z_mode=1
    )

    assert linear_trace._z_variation_norm(out) > 0.0
    assert jnp.linalg.norm(out[0, 0, 2, 1, 0]) > 0.0


def test_full_nonlinear_trace_summary_contains_metadata() -> None:
    payload = nonlinear_trace._build_summary(
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml",
        backend="gpu",
        nl=4,
        nm=8,
        repeats=5,
        state="initial",
        laguerre_mode="grid",
        compressed_real_fft=True,
        z_variation_norm=0.0,
        compile_execute_seconds=2.0,
        warm_seconds=0.01,
        rhs_norm=1.0,
        phi_norm=0.1,
        apar_norm=0.0,
        bpar_norm=0.0,
        hlo_text="ROOT multiply.1 = f32[] multiply(a, b)\nfft.2 = c64[] fft(c)\n",
        trace_dir=Path("tools_out/nonlinear_trace"),
        memory_profile=Path("tools_out/nonlinear.prof"),
        hlo_out=Path("tools_out/nonlinear.hlo.txt"),
        electrostatic_specialized=True,
    )

    assert payload["kind"] == "full_nonlinear_rhs_trace_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["backend"] == "gpu"
    assert payload["laguerre_mode"] == "grid"
    assert payload["compressed_real_fft"] is True
    assert payload["hlo_token_counts"]["multiply"] >= 1
    assert payload["hlo_token_counts"]["fft"] >= 1
    assert payload["electrostatic_specialized"] is True
    assert payload["trace_dir"] == "tools_out/nonlinear_trace"
    assert "transport runtime claim" in payload["claim_scope"]


def test_full_nonlinear_trace_field_norm_handles_missing_em_fields() -> None:
    assert nonlinear_trace._field_norm(None) == 0.0
