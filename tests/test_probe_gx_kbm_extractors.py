"""Tests for the single-trajectory KBM extractor probe."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from spectraxgk.analysis import ModeSelection
from spectraxgk.config import GridConfig


def _load_module():
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import probe_gx_kbm_extractors as mod
    finally:
        sys.path.remove(str(tools_dir))
    return mod


def test_probe_gx_kbm_extractors_save_load_roundtrip(tmp_path: Path) -> None:
    mod = _load_module()
    result = SimpleNamespace(
        t=np.array([0.0, 1.0], dtype=float),
        phi_t=np.array([[[[1.0 + 0.0j]]], [[[2.0 + 1.0j]]]], dtype=np.complex64),
        gamma_t=np.array([0.1, 0.2], dtype=float),
        omega_t=np.array([0.3, 0.4], dtype=float),
        ky=0.3,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
    )
    path = mod._save_trajectory(tmp_path / "traj.npz", result)
    restored = mod._load_trajectory(path)
    assert np.allclose(restored.t, result.t)
    assert np.allclose(restored.phi_t, result.phi_t)
    assert np.allclose(restored.gamma_t, result.gamma_t)
    assert np.allclose(restored.omega_t, result.omega_t)
    assert restored.selection == result.selection
    assert restored.ky == result.ky


def test_probe_gx_kbm_extractors_main_reuses_cached_trajectory(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_module()
    traj_dir = tmp_path / "traj"
    out = tmp_path / "probe.csv"
    result = SimpleNamespace(
        t=np.array([0.0, 1.0], dtype=float),
        phi_t=np.array([[[[1.0 + 0.0j]]], [[[2.0 + 1.0j]]]], dtype=np.complex64),
        gamma_t=np.array([0.1, 0.2], dtype=float),
        omega_t=np.array([0.3, 0.4], dtype=float),
        ky=0.3,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
    )
    mod._save_trajectory(mod._trajectory_path(traj_dir, 0.3), result)

    gx_time = np.array([0.0, 1.0], dtype=float)
    gx_ky = np.array([0.3], dtype=float)
    gx_omega_series = np.array([[[1.0, 2.0]], [[1.0, 2.0]]], dtype=float)

    monkeypatch.setattr(
        mod,
        "_load_gx_omega_gamma",
        lambda _path: (gx_time, gx_ky, gx_omega_series, 0.01, 1.4, 0.8, 0.18, 2.77778),
    )
    monkeypatch.setattr(
        mod,
        "_build_cfg",
        lambda **_kwargs: SimpleNamespace(
            grid=GridConfig(
                Nx=1,
                Ny=7,
                Nz=96,
                Lx=62.8,
                Ly=62.8,
                boundary="linked",
                y0=10.0,
                ntheta=32,
                nperiod=2,
            )
        ),
    )
    monkeypatch.setattr(
        mod,
        "_recompute_time_history_growth_on_grid",
        lambda _args, res, *, mode_method, t_ref: SimpleNamespace(
            t=t_ref,
            phi_t=res.phi_t,
            selection=res.selection,
            gamma=1.0 if mode_method == "project" else 1.5,
            omega=2.0 if mode_method == "project" else 2.5,
            gamma_t=res.gamma_t,
            omega_t=res.omega_t,
        ),
    )
    monkeypatch.setattr(
        mod,
        "_mode_metrics",
        lambda _result, **_kwargs: (None, None, 0.9, 0.1, float("nan")),
    )

    def _unexpected_run(**_kwargs):
        raise AssertionError("run_kbm_linear should not be called when reusing cached trajectories")

    monkeypatch.setattr(mod, "run_kbm_linear", _unexpected_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_gx_kbm_extractors.py",
            "--gx",
            str(tmp_path / "dummy.nc"),
            "--out",
            str(out),
            "--trajectory-dir",
            str(traj_dir),
            "--reuse-trajectory",
            "--ky",
            "0.3",
            "--mode-methods",
            "project,max",
        ],
    )

    mod.main()

    table = pd.read_csv(out)
    assert list(table["solver"]) == ["gx_time@project", "gx_time@max"]
    assert np.allclose(table["gamma"], [1.0, 1.5])
    assert np.allclose(table["omega"], [2.0, 2.5])
