from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "profiling"
        / "profile_nonlinear_sharding_sweep.py"
    )
    spec = importlib.util.spec_from_file_location(
        "profile_nonlinear_sharding_sweep", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_nonlinear_sharding_sweep_parser_defaults_to_bounded_artifact() -> None:
    mod = _load_tool_module()

    args = mod.build_parser().parse_args([])

    assert args.out_prefix == mod.DEFAULT_PREFIX
    assert args.backend == "cpu"
    assert args.devices == [1, 2]
    assert args.sharding_options == "auto,kx"
    assert args.timeout_s == 300.0
    assert args.office_gpu_xlarge is False


def test_profile_nonlinear_sharding_sweep_office_gpu_preset_is_canonical() -> None:
    mod = _load_tool_module()

    args = mod.apply_profile_preset(
        mod.build_parser().parse_args(["--office-gpu-xlarge"])
    )

    assert args.backend == "gpu"
    assert args.devices == [1, 2]
    assert (args.nx, args.ny, args.nz, args.nl, args.nm, args.steps) == (
        48,
        96,
        128,
        4,
        8,
        12,
    )
    assert args.sharding_options == "auto,kx"
    assert args.out_prefix == mod.OFFICE_GPU_XLARGE_PREFIX
    assert args.trace is True


def test_profile_nonlinear_sharding_sweep_device_env_is_backend_specific() -> None:
    mod = _load_tool_module()

    cpu_env = mod._device_env({"XLA_FLAGS": "--foo=bar"}, backend="cpu", devices=4)
    replaced_cpu_env = mod._device_env(
        {"XLA_FLAGS": "--foo=bar --xla_force_host_platform_device_count=8"},
        backend="cpu",
        devices=2,
    )
    gpu_env = mod._device_env({}, backend="gpu", devices=2)

    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert "--xla_force_host_platform_device_count=4" in cpu_env["XLA_FLAGS"]
    assert (
        "--xla_force_host_platform_device_count=8" not in replaced_cpu_env["XLA_FLAGS"]
    )
    assert "--xla_force_host_platform_device_count=2" in replaced_cpu_env["XLA_FLAGS"]
    assert "--foo=bar" in replaced_cpu_env["XLA_FLAGS"]
    assert gpu_env["JAX_PLATFORMS"] == "cuda"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert gpu_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_profile_nonlinear_sharding_sweep_row_selects_fastest_identity_candidate() -> (
    None
):
    mod = _load_tool_module()
    payload = {
        "source_contract_version": 1,
        "backend": "gpu",
        "device_count": 2,
        "default_backend": "gpu",
        "sharding_axis": "kx",
        "profile_command": "python tools/profiling/profile_nonlinear_sharding.py --sharding kx",
        "profile_command_argv": [
            "python",
            "tools/profiling/profile_nonlinear_sharding.py",
            "--sharding",
            "kx",
        ],
        "source_artifact": "/tmp/profile.json",
        "software_versions": {
            "python": "3.11.0",
            "spectraxgk": "test",
            "jax": "0.test",
            "jaxlib": "0.test",
            "numpy": "2.test",
        },
        "timing_warmup_repeat": {"warmups": 1, "repeats": 3},
        "state_shape": [4, 8, 17, 32, 64],
        "state_sharding_requested": "auto",
        "serial_stats_s": {"median": 10.0},
        "best_identity_preserving_candidate": {"spec": "kx"},
        "sharded_results": {
            "auto": {
                "state_sharding_active": True,
                "stats_s": {"median": 8.0},
                "identity_gate_pass": True,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
                "error": None,
            },
            "kx": {
                "state_sharding_active": True,
                "stats_s": {"median": 5.0},
                "identity_gate_pass": True,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
                "error": None,
            },
        },
    }

    row = mod._row_from_payload(payload, requested_devices=2)

    assert row["best_spec"] == "kx"
    assert row["parallel_median_s"] == 5.0
    assert row["same_process_speedup"] == 2.0
    assert row["identity_gate_pass"] is True
    assert row["source_contract_version"] == 1
    assert row["profile_command"].startswith(
        "python tools/profiling/profile_nonlinear_sharding.py"
    )
    assert row["profile_command_argv"][-2:] == ["--sharding", "kx"]
    assert row["source_artifact"] == "/tmp/profile.json"
    assert row["software_versions"]["spectraxgk"] == "test"
    assert row["timing_warmup_repeat"] == {"warmups": 1, "repeats": 3}
    assert row["profile_backend"] == "gpu"
    assert row["profile_device_count"] == 2
    assert row["profile_sharding_axis"] == "kx"


def test_profile_nonlinear_sharding_sweep_json_clean_replaces_nonfinite() -> None:
    mod = _load_tool_module()

    cleaned = mod._json_clean({"bad": math.inf, "ok": 1.0})

    assert cleaned == {"bad": None, "ok": 1.0}


def test_profile_nonlinear_sharding_sweep_records_timeout_rows(monkeypatch) -> None:
    mod = _load_tool_module()

    def _raise_timeout(*_args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="profile",
            timeout=float(kwargs["timeout"]),
            output="stdout tail",
            stderr="stderr tail",
        )

    monkeypatch.setattr(mod.subprocess, "run", _raise_timeout)

    summary = mod.run_sweep(
        backend="cpu",
        devices=[2],
        nx=4,
        ny=4,
        nz=4,
        nl=1,
        nm=1,
        dt=0.02,
        steps=1,
        method="rk2",
        sharding="auto",
        sharding_options="auto,kx",
        laguerre_mode="grid",
        warmups=0,
        repeats=1,
        timeout_s=0.5,
        trace=False,
    )

    assert summary["identity_passed"] is False
    assert summary["speedup_passed"] is False
    assert summary["status"] == "diagnostic_identity_only"
    assert summary["rows"][0]["parallel_median_s"] is None
    assert "timed out" in summary["rows"][0]["error"]
    assert "stderr tail" in summary["rows"][0]["error"]
    assert summary["speedup_blockers"] == ["cpu_2devices_identity_failed"]


def test_profile_nonlinear_sharding_sweep_marks_identity_only_slowdown(
    monkeypatch,
) -> None:
    mod = _load_tool_module()

    def _fake_run(cmd, **_kwargs):
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        device_count = 2 if "2devices" in out_json.name else 1
        spec = "kx" if device_count == 2 else "auto"
        median = 20.0 if device_count == 2 else 10.0
        payload = {
            "device_count": device_count,
            "default_backend": "gpu",
            "state_shape": [1],
            "state_sharding_requested": "auto",
            "serial_stats_s": {"median": 10.0},
            "best_identity_preserving_candidate": {"spec": spec},
            "sharded_results": {
                spec: {
                    "state_sharding_active": device_count > 1,
                    "stats_s": {"median": median},
                    "identity_gate_pass": True,
                    "max_abs_state_error": 0.0,
                    "max_rel_state_error": 0.0,
                    "error": None,
                }
            },
        }
        out_json.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    summary = mod.run_sweep(
        backend="gpu",
        devices=[1, 2],
        nx=4,
        ny=4,
        nz=4,
        nl=1,
        nm=1,
        dt=0.02,
        steps=1,
        method="rk2",
        sharding="auto",
        sharding_options="auto,kx",
        laguerre_mode="grid",
        warmups=0,
        repeats=1,
        timeout_s=1.0,
        trace=False,
    )

    assert summary["identity_passed"] is True
    assert summary["speedup_passed"] is False
    assert summary["status"] == "diagnostic_identity_only"
    assert summary["rows"][1]["strong_speedup_vs_1_device"] == 0.5
    assert summary["speedup_blockers"] == ["gpu_2devices_speedup_0.5_below_1"]


def test_profile_nonlinear_sharding_sweep_preserves_failed_profile_json(
    monkeypatch,
) -> None:
    mod = _load_tool_module()

    def _fake_run(cmd, **_kwargs):
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        payload = {
            "device_count": 4,
            "default_backend": "cpu",
            "state_shape": [4, 8, 17, 32, 64],
            "state_sharding_requested": "auto",
            "serial_stats_s": {"median": 10.0},
            "best_identity_preserving_candidate": {
                "spec": None,
                "identity_gate_pass": False,
                "state_sharding_active": False,
                "engineering_speedup_median": None,
            },
            "sharded_results": {
                "auto": {
                    "state_sharding_active": True,
                    "stats_s": None,
                    "identity_gate_pass": False,
                    "max_abs_state_error": None,
                    "max_rel_state_error": None,
                    "error": "skipped: cpu_whole_state_pjit_sharding_unsafe_for_fft_layout",
                    "skip_reason": "cpu_whole_state_pjit_sharding_unsafe_for_fft_layout",
                }
            },
        }
        out_json.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 2, "profile json written", "")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    summary = mod.run_sweep(
        backend="cpu",
        devices=[4],
        nx=4,
        ny=4,
        nz=4,
        nl=1,
        nm=1,
        dt=0.02,
        steps=1,
        method="rk2",
        sharding="auto",
        sharding_options="auto",
        laguerre_mode="grid",
        warmups=0,
        repeats=1,
        timeout_s=1.0,
        trace=False,
    )

    assert summary["identity_passed"] is False
    assert summary["rows"][0]["profile_returncode"] == 2
    assert summary["rows"][0]["state_sharding_active"] is True
    assert "unsafe_for_fft_layout" in summary["rows"][0]["error"]
    assert summary["profiles"]["4"]["profile_returncode"] == 2
