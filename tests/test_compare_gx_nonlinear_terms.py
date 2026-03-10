"""Tests for GX nonlinear term comparison tooling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_compare_gx_nonlinear_terms_parser_accepts_runtime_config() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(
        [
            "--gx-dir",
            "gx_dump",
            "--gx-out",
            "gx.out.nc",
            "--config",
            "runtime.toml",
            "--ky",
            "0.4",
        ]
    )

    assert args.gx_dir == Path("gx_dump")
    assert args.gx_out == Path("gx.out.nc")
    assert args.config == Path("runtime.toml")
    assert args.ky == 0.4


def test_build_runtime_compare_context_overrides_grid_from_dump(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_nonlinear_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    cfg = SimpleNamespace(grid=SimpleNamespace(Nx=8, Ny=8, Nz=8, y0=None))
    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "load_runtime_from_toml", lambda _path: (cfg, None))
    monkeypatch.setattr(
        mod,
        "replace",
        lambda obj, **updates: SimpleNamespace(**(obj.__dict__ | updates)),
    )

    def _fake_build_runtime_geometry(cfg_use):
        captured["cfg_use"] = cfg_use
        return "geom"

    monkeypatch.setattr(mod, "build_runtime_geometry", _fake_build_runtime_geometry)
    monkeypatch.setattr(mod, "apply_gx_geometry_grid_defaults", lambda _geom, grid: grid)
    grid_obj = SimpleNamespace(ky=np.array([0.0, 0.2, -0.2]), kx=np.array([0.0]), z=np.array([0.0, 1.0]))
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_obj)
    monkeypatch.setattr(mod, "build_runtime_linear_params", lambda *_args, **_kwargs: "params")
    monkeypatch.setattr(mod, "build_runtime_term_config", lambda _cfg: "terms")

    cfg_use, geom, grid, params, term_cfg = mod._build_runtime_compare_context(
        Path("runtime.toml"),
        nx=3,
        ny_full=6,
        nz=5,
        nl=2,
        nm=4,
        ky_vals_nyc=np.array([0.2, 0.4], dtype=float),
        y0_override=None,
    )

    assert cfg_use.grid.Nx == 3
    assert cfg_use.grid.Ny == 6
    assert cfg_use.grid.Nz == 5
    assert cfg_use.grid.y0 == 5.0
    assert captured["cfg_use"] is cfg_use
    assert geom == "geom"
    assert grid is grid_obj
    assert params == "params"
    assert term_cfg == "terms"
