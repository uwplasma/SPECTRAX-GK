from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _load_tool():
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear_rk4_stage as mod
    finally:
        sys.path.remove(str(tools_dir))
    return mod


def test_compare_gx_nonlinear_rk4_stage_parser_accepts_core_args() -> None:
    mod = _load_tool()

    args = mod.build_parser().parse_args(
        [
            "--gx-dir",
            "gx_dump",
            "--config",
            "runtime.toml",
            "--partial-call",
            "1",
        ]
    )

    assert args.gx_dir == Path("gx_dump")
    assert args.config == Path("runtime.toml")
    assert args.partial_call == 1


def test_nonlinear_partial_stage_targets_map_calls() -> None:
    mod = _load_tool()

    stages = mod.NonlinearRK4StageStates(
        k1_linear=np.array([1.0]),
        k1_nonlinear=np.array([2.0]),
        k1_total=np.array([3.0]),
        k2_linear=np.array([4.0]),
        k2_nonlinear=np.array([5.0]),
        k2_total=np.array([6.0]),
        k3_linear=np.array([7.0]),
        k3_nonlinear=np.array([8.0]),
        k3_total=np.array([9.0]),
        k4_linear=np.array([10.0]),
        k4_nonlinear=np.array([11.0]),
        k4_total=np.array([12.0]),
        g2=np.array([13.0]),
        g3=np.array([14.0]),
        g4=np.array([15.0]),
    )

    g2, l2, n2, t2 = mod._partial_stage_targets(1, stages)
    g3, l3, n3, t3 = mod._partial_stage_targets(2, stages)
    g4, l4, n4, t4 = mod._partial_stage_targets(3, stages)

    assert np.array_equal(g2, np.array([13.0]))
    assert np.array_equal(l2, np.array([4.0]))
    assert np.array_equal(n2, np.array([5.0]))
    assert np.array_equal(t2, np.array([6.0]))
    assert np.array_equal(g3, np.array([14.0]))
    assert np.array_equal(l3, np.array([7.0]))
    assert np.array_equal(n3, np.array([8.0]))
    assert np.array_equal(t3, np.array([9.0]))
    assert np.array_equal(g4, np.array([15.0]))
    assert np.array_equal(l4, np.array([10.0]))
    assert np.array_equal(n4, np.array([11.0]))
    assert np.array_equal(t4, np.array([12.0]))

    with pytest.raises(ValueError, match="partial_call"):
        mod._partial_stage_targets(4, stages)
