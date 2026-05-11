from __future__ import annotations

import importlib.util
import math
from pathlib import Path


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "profile_nonlinear_sharding_sweep.py"
    spec = importlib.util.spec_from_file_location("profile_nonlinear_sharding_sweep", path)
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


def test_profile_nonlinear_sharding_sweep_device_env_is_backend_specific() -> None:
    mod = _load_tool_module()

    cpu_env = mod._device_env({"XLA_FLAGS": "--foo=bar"}, backend="cpu", devices=4)
    gpu_env = mod._device_env({}, backend="gpu", devices=2)

    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert "--xla_force_host_platform_device_count=4" in cpu_env["XLA_FLAGS"]
    assert gpu_env["JAX_PLATFORMS"] == "cuda"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert gpu_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_profile_nonlinear_sharding_sweep_row_selects_fastest_identity_candidate() -> None:
    mod = _load_tool_module()
    payload = {
        "device_count": 2,
        "default_backend": "gpu",
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


def test_profile_nonlinear_sharding_sweep_json_clean_replaces_nonfinite() -> None:
    mod = _load_tool_module()

    cleaned = mod._json_clean({"bad": math.inf, "ok": 1.0})

    assert cleaned == {"bad": None, "ok": 1.0}
