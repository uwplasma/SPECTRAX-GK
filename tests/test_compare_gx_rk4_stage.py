from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


def _load_tool():
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rk4_stage as mod
    finally:
        sys.path.remove(str(tools_dir))
    return mod


def test_partial_stage_targets_map_calls() -> None:
    mod = _load_tool()

    stages = mod.RK4StageStates(
        k1=np.array([1.0]),
        k2=np.array([2.0]),
        k3=np.array([3.0]),
        k4=np.array([4.0]),
        g1=np.array([10.0]),
        g2=np.array([20.0]),
        g3=np.array([30.0]),
        g_next=np.array([40.0]),
    )

    g1, rhs1 = mod._partial_stage_targets(1, stages)
    g2, rhs2 = mod._partial_stage_targets(2, stages)
    g3, rhs3 = mod._partial_stage_targets(3, stages)

    assert np.array_equal(g1, np.array([10.0]))
    assert np.array_equal(rhs1, np.array([2.0]))
    assert np.array_equal(g2, np.array([20.0]))
    assert np.array_equal(rhs2, np.array([3.0]))
    assert np.array_equal(g3, np.array([30.0]))
    assert np.array_equal(rhs3, np.array([4.0]))

    with pytest.raises(ValueError, match="partial_call"):
        mod._partial_stage_targets(4, stages)


def test_compute_stage_states_reconstructs_rk4(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_tool()

    def _fake_rhs(state, _cache, _params, *, terms):
        return 2.0 * state, terms

    monkeypatch.setattr(mod, "assemble_rhs_cached", _fake_rhs)

    stages = mod._compute_stage_states(
        np.array([1.0], dtype=np.float64),
        cache=None,
        params=None,
        term_cfg=None,
        dt=0.5,
    )

    assert np.allclose(stages.k1, np.array([2.0]))
    assert np.allclose(stages.g1, np.array([1.5]))
    assert np.allclose(stages.k2, np.array([3.0]))
    assert np.allclose(stages.g2, np.array([1.75]))
    assert np.allclose(stages.k3, np.array([3.5]))
    assert np.allclose(stages.g3, np.array([2.75]))
    assert np.allclose(stages.k4, np.array([5.5]))
    assert np.allclose(stages.g_next, np.array([2.7083333333333335]))
