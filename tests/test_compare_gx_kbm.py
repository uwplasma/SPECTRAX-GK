"""Tests for the deterministic GX KBM comparison harness."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
    from spectraxgk.benchmarks import LinearRunResult

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
        krylov_gx_shift_source="target",
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
    assert krylov_cfg.shift_source == "target"
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
        krylov_gx_shift_source="target",
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


def test_compare_gx_kbm_run_candidate_honors_mode_method_override(monkeypatch) -> None:
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
        mode_method="project",
        no_auto_window=False,
        tmin=None,
        tmax=None,
        sample_stride=1,
        krylov_gx_shift=False,
        krylov_gx_shift_source="target",
    )

    mod._run_candidate(
        args,
        cfg=object(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="gx_time",
        mode_method_override="max",
        gx_gamma=0.219,
        gx_omega=1.141,
    )

    assert captured["mode_method"] == "max"


def test_compare_gx_kbm_run_candidate_cached_reuses_gx_time_trajectory(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    calls: list[str] = []

    def _fake_run_candidate(*_args, **_kwargs):
        calls.append("run")
        return SimpleNamespace(
            gamma=0.1,
            omega=0.2,
            t=[0.0, 1.0],
            phi_t=[[[[1.0 + 0.0j]]], [[[2.0 + 0.0j]]]],
            selection=SimpleNamespace(ky_index=0, kx_index=0, z_index=0),
        )

    def _fake_recompute(args, result, *, mode_method):
        calls.append(f"recompute:{mode_method}")
        return SimpleNamespace(
            gamma=1.1 if mode_method == "project" else 1.2,
            omega=-2.1 if mode_method == "project" else -2.2,
            t=result.t,
            phi_t=result.phi_t,
            selection=result.selection,
        )

    monkeypatch.setattr(mod, "_run_candidate", _fake_run_candidate)
    monkeypatch.setattr(mod, "_recompute_time_history_growth", _fake_recompute)

    args = SimpleNamespace(
        mode_method="project",
        dt=0.01,
        steps=4000,
        method="rk4",
    )
    cache: dict[tuple[object, ...], object] = {}

    result_project = mod._run_candidate_cached(
        args,
        cfg=object(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="gx_time",
        mode_method_override="project",
        result_cache=cache,
        gx_time_ref=None,
        gx_gamma=0.2,
        gx_omega=-1.0,
    )
    result_max = mod._run_candidate_cached(
        args,
        cfg=object(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="gx_time",
        mode_method_override="max",
        result_cache=cache,
        gx_time_ref=None,
        gx_gamma=0.2,
        gx_omega=-1.0,
    )

    assert calls == ["run", "recompute:project", "recompute:max"]
    assert result_project.gamma == 1.1
    assert result_max.gamma == 1.2


def test_compare_gx_kbm_recompute_on_gx_time_grid(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))
    from spectraxgk.benchmarks import LinearRunResult

    captured: dict[str, object] = {}

    def _fake_recompute(_args, result, *, mode_method: str):
        captured["t"] = np.asarray(result.t)
        captured["phi_shape"] = np.asarray(result.phi_t).shape
        captured["mode_method"] = mode_method
        return result

    monkeypatch.setattr(mod, "_recompute_time_history_growth", _fake_recompute)

    result = LinearRunResult(
        t=np.array([0.0, 1.0, 2.0], dtype=float),
        phi_t=np.array([[[[1.0 + 0.0j]]], [[[2.0 + 0.0j]]], [[[3.0 + 0.0j]]]], dtype=np.complex128),
        gamma=0.0,
        omega=0.0,
        ky=0.3,
        selection=SimpleNamespace(ky_index=0, kx_index=0, z_index=0),
    )
    gx_time = np.array([0.0, 0.5, 1.5], dtype=float)

    sampled = mod._recompute_time_history_growth_on_grid(
        SimpleNamespace(),
        result,
        mode_method="project",
        t_ref=gx_time,
    )

    assert np.allclose(captured["t"], gx_time)
    assert captured["phi_shape"] == (3, 1, 1, 1)
    assert captured["mode_method"] == "project"
    assert np.allclose(np.asarray(sampled.phi_t)[:, 0, 0, 0].real, [1.0, 1.5, 2.5])


def test_compare_gx_kbm_run_candidate_allows_shift_source_override(monkeypatch) -> None:
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
        krylov_gx_shift_source="propagator",
    )

    mod._run_candidate(
        args,
        cfg=object(),
        ky_value=0.3,
        beta_value=0.015,
        solver_name="krylov",
        gx_gamma=0.219,
        gx_omega=1.141,
    )

    krylov_cfg = captured["krylov_cfg"]
    assert krylov_cfg is not None
    assert krylov_cfg.shift_source == "propagator"


def test_compare_gx_kbm_parser_defaults_to_project_mode() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(["--gx", "kbm.out.nc"])

    assert args.mode_method == "project"
    assert args.steps is None
    assert args.branch_policy == "continuation"
    assert args.branch_solvers == "gx_time@project,gx_time@svd,gx_time@max,gx_time@z_index,krylov,time"


def test_compare_gx_kbm_candidate_row_captures_branch_metrics() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    result = SimpleNamespace(gamma=0.8, omega=-1.5)
    row = mod._candidate_row(
        ky=0.3,
        solver="gx_time",
        result=result,
        gx_gamma=1.0,
        gx_omega=-2.0,
        eig_overlap_gx=0.9,
        eig_rel_l2=0.1,
        eig_overlap_prev=0.8,
        branch_score=0.42,
        selected=True,
    )

    assert row["ky"] == 0.3
    assert row["solver"] == "gx_time"
    assert row["gamma"] == 0.8
    assert row["omega"] == -1.5
    assert row["rel_gamma"] == pytest.approx(0.2)
    assert row["rel_omega"] == pytest.approx(0.25)
    assert row["eig_overlap_gx"] == 0.9
    assert row["eig_rel_l2"] == 0.1
    assert row["eig_overlap_prev"] == 0.8
    assert row["branch_score"] == 0.42
    assert row["selected"] is True


def test_compare_gx_kbm_parse_candidate_spec_supports_mode_override() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    solver, mode_method, label = mod._parse_candidate_spec("gx_time@max")

    assert solver == "gx_time"
    assert mode_method == "max"
    assert label == "gx_time@max"


def test_compare_gx_kbm_parse_candidate_spec_without_override() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    solver, mode_method, label = mod._parse_candidate_spec("krylov")

    assert solver == "krylov"
    assert mode_method is None
    assert label == "krylov"


def test_compare_gx_kbm_loads_npz_reference(tmp_path: Path) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    path = tmp_path / "kbm_ref.npz"
    np.savez(
        path,
        time=np.array([0.1, 0.2], dtype=float),
        ky=np.array([0.1, 0.2, 0.3], dtype=float),
        omega_series=np.zeros((2, 3, 2), dtype=float),
        beta=np.asarray(0.015),
        q=np.asarray(1.4),
        shat=np.asarray(0.8),
        eps=np.asarray(0.18),
        rmaj=np.asarray(2.77778),
        theta=np.array([-1.0, 0.0, 1.0], dtype=float),
        phi_modes=np.array(
            [
                [1.0 + 0.0j, 2.0 + 0.0j, 1.0 + 0.0j],
                [2.0 + 0.0j, 4.0 + 0.0j, 2.0 + 0.0j],
                [1.0j, 2.0j, 1.0j],
            ],
            dtype=np.complex128,
        ),
    )

    time, ky, omega, beta, q, shat, eps, rmaj = mod._load_gx_omega_gamma(path)
    theta, mode = mod._load_gx_eigenfunction(path, 0.2)

    assert np.allclose(time, np.array([0.1, 0.2]))
    assert np.allclose(ky, np.array([0.1, 0.2, 0.3]))
    assert omega.shape == (2, 3, 2)
    assert beta == pytest.approx(0.015)
    assert q == pytest.approx(1.4)
    assert shat == pytest.approx(0.8)
    assert eps == pytest.approx(0.18)
    assert rmaj == pytest.approx(2.77778)
    assert np.allclose(theta, np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(mode, np.array([0.5 + 0.0j, 1.0 + 0.0j, 0.5 + 0.0j]))


def test_compare_gx_kbm_npz_zero_geometry_scalars_fall_back_to_defaults(tmp_path: Path) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    path = tmp_path / "kbm_ref_zero_geom.npz"
    np.savez(
        path,
        time=np.array([0.1, 0.2], dtype=float),
        ky=np.array([0.1, 0.2, 0.3], dtype=float),
        omega_series=np.zeros((2, 3, 2), dtype=float),
        beta=np.asarray(0.015),
        q=np.asarray(0.0),
        shat=np.asarray(0.0),
        eps=np.asarray(0.0),
        rmaj=np.asarray(0.0),
        theta=np.array([-1.0, 0.0, 1.0], dtype=float),
        phi_modes=np.ones((3, 3), dtype=np.complex128),
    )

    _time, _ky, _omega, beta, q, shat, eps, rmaj = mod._load_gx_omega_gamma(path)

    assert beta == pytest.approx(0.015)
    assert q is None
    assert shat is None
    assert eps is None
    assert rmaj is None
