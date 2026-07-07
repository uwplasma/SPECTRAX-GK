from __future__ import annotations

import importlib.util
from pathlib import Path

import jax


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "profiling" / "profile_device_z_pencil_transport_window.py"
spec = importlib.util.spec_from_file_location(
    "profile_device_z_pencil_transport_window",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_auto_z_chunk_size_records_fft_batch_pressure_model(monkeypatch) -> None:
    real_devices = tuple(jax.devices())
    monkeypatch.setattr(mod.jax, "devices", lambda: [real_devices[0]])

    payload = mod.build_profile(
        shape=(1, 1, 4, 4, 8),
        device_counts=(1, 2),
        steps=1,
        dt=0.001,
        warmups=0,
        repeats=1,
        observable_repeats=0,
        atol=5.0e-6,
        rtol=1.0e-4,
        min_speedup=1.5,
        z_chunk_size=None,
        auto_z_chunk_size=True,
        max_fft_batch_count=4,
        observable_mode="host_gather",
        trace_dir=None,
        trace_device_count=None,
        hlo_prefix=None,
    )

    model = payload["fft_batch_pressure_model"]
    assert payload["z_chunk_size"] == 1
    assert payload["auto_z_chunk_size"] is True
    assert model["chunking_required"] is True
    assert model["suggested_z_chunk_size"] == 1
    assert model["chunked_fft_batch_count"] == 4
    assert payload["rows"][1]["blocked_reasons"] == ["not_enough_devices"]
    assert payload["observable_repeats"] == 0
    assert payload["observable_mode"] == "host_gather"
    assert payload["rows"][0]["timing_scope"] == "compute_only_final_state_update"
    assert payload["rows"][1]["observable_gate_stats_s"] == {}


def test_observable_repeats_rejects_negative_values(monkeypatch) -> None:
    real_devices = tuple(jax.devices())
    monkeypatch.setattr(mod.jax, "devices", lambda: [real_devices[0]])

    try:
        mod.build_profile(
            shape=(1, 1, 4, 4, 8),
            device_counts=(1,),
            steps=1,
            dt=0.001,
            warmups=0,
            repeats=1,
            observable_repeats=-1,
            atol=5.0e-6,
            rtol=1.0e-4,
            min_speedup=1.5,
            z_chunk_size=None,
            auto_z_chunk_size=False,
            max_fft_batch_count=4,
            observable_mode="host_gather",
            trace_dir=None,
            trace_device_count=None,
            hlo_prefix=None,
        )
    except ValueError as exc:
        assert "observable_repeats" in str(exc)
    else:  # pragma: no cover - defensive assertion path.
        raise AssertionError("negative observable_repeats should fail")
