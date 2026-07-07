from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "profile_full_nonlinear_rhs_trace.py"
spec = importlib.util.spec_from_file_location("profile_full_nonlinear_rhs_trace", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_build_summary_contains_nonlinear_trace_metadata() -> None:
    payload = mod._build_summary(
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


def test_field_norm_handles_missing_em_fields() -> None:
    assert mod._field_norm(None) == 0.0
