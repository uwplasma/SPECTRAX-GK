from __future__ import annotations

from support.paths import load_profiling_tool

import jax


mod = load_profiling_tool("profile_device_z_pencil_transport_window")


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


def test_rhs_mode_records_rhs_identity_gate(monkeypatch) -> None:
    real_devices = tuple(jax.devices())
    monkeypatch.setattr(mod.jax, "devices", lambda: [real_devices[0]])

    payload = mod.build_rhs_profile(
        shape=(1, 1, 4, 4, 8),
        device_counts=(1, 2),
        warmups=0,
        repeats=1,
        atol=5.0e-6,
        rtol=1.0e-4,
        min_speedup=1.5,
    )

    assert payload["kind"] == "nonlinear_device_z_pencil_rhs_profile"
    assert payload["rows"][0]["rhs_max_abs_error"] == 0.0
    assert payload["rows"][1]["blocked_reasons"] == ["not_enough_devices"]
    assert payload["summary"]["status"] == "skipped_no_multidevice"


def test_parser_defaults_to_transport_window_mode() -> None:
    args = mod.build_parser().parse_args([])

    assert args.mode == "transport-window"
    assert args.out_prefix is None


def test_main_dispatches_rhs_mode_to_rhs_profiler(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_build_rhs_profile(**kwargs):
        calls["profile"] = kwargs
        return {
            "kind": "nonlinear_device_z_pencil_rhs_profile",
            "summary": {"status": "fake_rhs"},
        }

    def fake_write_artifacts(summary, out_prefix):
        calls["write"] = (summary, out_prefix)

    monkeypatch.setattr(mod, "build_rhs_profile", fake_build_rhs_profile)
    monkeypatch.setattr(mod, "write_artifacts", fake_write_artifacts)

    rc = mod.main(
        [
            "--mode",
            "rhs",
            "--shape",
            "1,1,4,4,8",
            "--device-counts",
            "1",
            "--warmups",
            "0",
            "--repeats",
            "1",
        ]
    )

    assert rc == 0
    assert calls["profile"] == {
        "shape": (1, 1, 4, 4, 8),
        "device_counts": (1,),
        "warmups": 0,
        "repeats": 1,
        "atol": 5.0e-6,
        "rtol": 1.0e-4,
        "min_speedup": 1.5,
    }
    summary, out_prefix = calls["write"]  # type: ignore[misc]
    assert summary["summary"]["status"] == "fake_rhs"
    assert out_prefix == mod.DEFAULT_RHS_OUT_PREFIX


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
