from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "profile_nonlinear_sharding.py"
    spec = importlib.util.spec_from_file_location("profile_nonlinear_sharding", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_nonlinear_sharding_parser_defaults_to_tracked_artifact() -> None:
    mod = _load_tool_module()
    args = mod.build_parser().parse_args([])

    assert args.out_json == mod.DEFAULT_OUT
    assert args.sharding == "auto"
    assert args.sharding_options is None
    assert args.method == "rk2"
    assert args.warmups == 1
    assert args.repeats == 3


def test_profile_nonlinear_sharding_helpers_report_stats_and_unique_specs() -> None:
    mod = _load_tool_module()

    stats = mod._time_stats([3.0, 1.0, 2.0])

    assert stats["min"] == 1.0
    assert stats["median"] == 2.0
    assert stats["mean"] == 2.0
    assert stats["max"] == 3.0
    assert mod._sharding_specs("auto", "ky,kx,ky,z") == ["auto", "ky", "kx", "z"]
