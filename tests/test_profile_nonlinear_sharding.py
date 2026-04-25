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
    assert args.method == "rk2"
