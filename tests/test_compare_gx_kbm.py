"""Tests for the deterministic GX KBM comparison harness."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd


def test_compare_gx_kbm_checkpoints_partial_rows(tmp_path: Path) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    out = tmp_path / "kbm.csv"
    rows = [
        {"ky": 0.2, "solver": "gx_time", "gamma": 1.0},
        {"ky": 0.3, "solver": "gx_time", "gamma": 2.0},
    ]

    mod._write_rows(out, rows[:1])
    first = pd.read_csv(out)
    assert list(first["ky"]) == [0.2]

    mod._write_rows(out, rows)
    second = pd.read_csv(out)
    assert list(second["ky"]) == [0.2, 0.3]
    assert list(second["gamma"]) == [1.0, 2.0]


def test_compare_gx_kbm_continuation_score_prefers_overlap() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    smooth = mod._candidate_objective(
        rel_gamma=0.10,
        rel_omega=0.10,
        eig_overlap_gx=0.80,
        eig_overlap_prev=0.95,
        gamma_weight=1.0,
        omega_weight=1.0,
        gx_overlap_weight=1.0,
        prev_overlap_weight=2.0,
    )
    jump = mod._candidate_objective(
        rel_gamma=0.08,
        rel_omega=0.08,
        eig_overlap_gx=0.82,
        eig_overlap_prev=0.20,
        gamma_weight=1.0,
        omega_weight=1.0,
        gx_overlap_weight=1.0,
        prev_overlap_weight=2.0,
    )

    assert smooth < jump


def test_compare_gx_kbm_run_candidate_uses_gx_shift_for_krylov(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    captured: dict[str, object] = {}

    def _fake_run_kbm_linear(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(gamma=0.1, omega=0.2)

    monkeypatch.setattr(mod, "run_kbm_linear", _fake_run_kbm_linear)

    args = SimpleNamespace(
        time_fit_signal="auto",
        Nl=16,
        Nm=48,
        dt=0.01,
        steps=4000,
        method="rk4",
        mode_method="z_index",
        no_auto_window=False,
        tmin=None,
        tmax=None,
        sample_stride=1,
        krylov_gx_shift=True,
    )

    result = mod._run_candidate(
        args,
        cfg=object(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="krylov",
        gx_gamma=0.219,
        gx_omega=1.141,
    )

    assert result.gamma == 0.1
    krylov_cfg = captured["krylov_cfg"]
    assert krylov_cfg is not None
    assert krylov_cfg.shift == complex(0.219, -1.141)
    assert krylov_cfg.shift_source == "propagator"
    assert krylov_cfg.omega_sign == 0
    assert krylov_cfg.omega_target_factor == 0.0
    assert krylov_cfg.shift_selection == "shift"


def test_compare_gx_kbm_run_candidate_skips_gx_shift_for_non_krylov(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    captured: dict[str, object] = {}

    def _fake_run_kbm_linear(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(gamma=0.1, omega=0.2)

    monkeypatch.setattr(mod, "run_kbm_linear", _fake_run_kbm_linear)

    args = SimpleNamespace(
        time_fit_signal="auto",
        Nl=16,
        Nm=48,
        dt=0.01,
        steps=4000,
        method="rk4",
        mode_method="z_index",
        no_auto_window=False,
        tmin=None,
        tmax=None,
        sample_stride=1,
        krylov_gx_shift=True,
    )

    mod._run_candidate(
        args,
        cfg=object(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="gx_time",
        gx_gamma=0.219,
        gx_omega=1.141,
    )

    assert captured["krylov_cfg"] is None
