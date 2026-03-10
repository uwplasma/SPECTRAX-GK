"""Tests for exact-state nonlinear runtime window comparison tooling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset


def test_compare_gx_runtime_window_parser_requires_core_args() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_runtime_window as mod
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
            "--time-index-start",
            "10",
            "--time-index-stop",
            "11",
        ]
    )

    assert args.gx_dir == Path("gx_dump")
    assert args.gx_out == Path("gx.out.nc")
    assert args.config == Path("runtime.toml")
    assert args.time_index_start == 10
    assert args.time_index_stop == 11


def test_compare_gx_runtime_window_parser_accepts_optional_ky_and_steps() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_runtime_window as mod
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
            "--time-index-start",
            "10",
            "--time-index-stop",
            "11",
            "--steps",
            "50",
            "--ky",
            "0.3",
        ]
    )
    assert args.steps == 50
    assert args.ky == 0.3


def test_compare_gx_runtime_window_writes_csv(tmp_path: Path, monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_runtime_window as mod
    finally:
        sys.path.remove(str(tools_dir))

    gx_out = tmp_path / "gx.out.nc"
    with Dataset(gx_out, "w") as ds:
        ds.createDimension("l", 1)
        ds.createDimension("m", 1)
        ds.createDimension("s", 1)
        ds.createDimension("time", 2)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        time = grids.createVariable("time", "f8", ("time",))
        time[:] = [1.0, 2.0]
        for name, value in {
            "Wg_kyst": [1.0, 2.0],
            "Wphi_kyst": [3.0, 4.0],
            "Wapar_kyst": [0.0, 0.0],
            "HeatFlux_kyst": [5.0, 6.0],
            "ParticleFlux_kyst": [0.0, 0.0],
        }.items():
            var = diag.createVariable(name, "f8", ("time", "s"))
            var[:] = np.asarray(value, dtype=float)[:, None]
        heat_s = diag.createVariable("HeatFlux_st", "f8", ("time", "s"))
        heat_s[:] = [[5.0], [6.0]]
        pflux_s = diag.createVariable("ParticleFlux_st", "f8", ("time", "s"))
        pflux_s[:] = [[0.0], [0.0]]

    gx_dir = tmp_path / "gx_dump"
    gx_dir.mkdir()
    np.ones(2, dtype=np.complex64).tofile(gx_dir / "diag_state_G_s0_t0.bin")
    np.ones(2, dtype=np.complex64).tofile(gx_dir / "diag_state_G_s0_t1.bin")
    np.ones(2, dtype=np.complex64).tofile(gx_dir / "diag_state_phi_t0.bin")
    np.ones(2, dtype=np.complex64).tofile(gx_dir / "diag_state_phi_t1.bin")
    np.array([0.0], dtype=np.float32).tofile(gx_dir / "diag_state_kx_t0.bin")
    np.array([0.1], dtype=np.float32).tofile(gx_dir / "diag_state_ky_t0.bin")

    monkeypatch.setattr(
        mod,
        "replace",
        lambda obj, **updates: SimpleNamespace(**(obj.__dict__ | updates)),
    )
    monkeypatch.setattr(
        mod,
        "load_runtime_from_toml",
        lambda _path: (
            SimpleNamespace(
                grid=SimpleNamespace(Nx=1, Ny=1, Nz=2, y0=10.0),
                time=SimpleNamespace(
                    dt=0.1,
                    method="rk3",
                    nonlinear_dealias=False,
                    laguerre_nonlinear_mode="grid",
                    fixed_dt=False,
                    dt_min=1.0e-6,
                    dt_max=None,
                    cfl=1.0,
                    cfl_fac=1.73,
                    collision_split=False,
                    collision_scheme="implicit",
                    implicit_restart=20,
                    implicit_solve_method="gmres",
                    implicit_preconditioner=None,
                ),
                run=SimpleNamespace(ky=0.1),
                normalization=SimpleNamespace(flux_scale=1.0, wphi_scale=1.0),
                init=SimpleNamespace(init_file=None, init_file_scale=1.0, init_file_mode="replace"),
            ),
            None,
        ),
    )
    monkeypatch.setattr(mod, "build_runtime_geometry", lambda _cfg: object())
    monkeypatch.setattr(mod, "apply_gx_geometry_grid_defaults", lambda _geom, grid: grid)
    grid_full = SimpleNamespace(ky=np.array([0.1]), kx=np.array([0.0]), z=np.array([0.0, 1.0]))
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_full)
    monkeypatch.setattr(mod, "select_gx_real_fft_ky_grid", lambda grid, _ky: grid)
    monkeypatch.setattr(mod, "ensure_flux_tube_geometry_data", lambda geom, _theta: geom)
    monkeypatch.setattr(mod, "build_runtime_linear_params", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod, "build_linear_cache", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(mod, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        mod,
        "_load_real_vector_auto",
        lambda path: np.array([0.0], dtype=np.float32) if "kx" in path.name else np.array([0.1], dtype=np.float32),
    )
    monkeypatch.setattr(mod, "_load_species_state", lambda *_args, **_kwargs: np.ones((1, 1, 1, 1, 1, 2), dtype=np.complex64))
    monkeypatch.setattr(mod, "_load_field", lambda *_args, **_kwargs: np.ones((1, 1, 2), dtype=np.complex64))
    monkeypatch.setattr(mod, "_maybe_load_field", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "compute_fields_cached",
        lambda *_args, **_kwargs: SimpleNamespace(
            phi=np.ones((1, 1, 2), dtype=np.complex64),
            apar=None,
            bpar=None,
        ),
    )
    monkeypatch.setattr(mod, "_summary", lambda *_args, **_kwargs: None)

    diag = SimpleNamespace(
        t=np.array([1.0], dtype=float),
        dt_t=np.array([0.02], dtype=float),
        gamma_t=np.array([0.0], dtype=float),
        omega_t=np.array([0.0], dtype=float),
        Wg_t=np.array([2.0], dtype=float),
        Wphi_t=np.array([4.0], dtype=float),
        Wapar_t=np.array([0.0], dtype=float),
        energy_t=np.array([6.0], dtype=float),
        heat_flux_t=np.array([6.0], dtype=float),
        particle_flux_t=np.array([0.0], dtype=float),
        heat_flux_species_t=np.array([[6.0]], dtype=float),
        particle_flux_species_t=np.array([[0.0]], dtype=float),
    )
    monkeypatch.setattr(
        mod,
        "run_runtime_nonlinear",
        lambda *_args, **_kwargs: SimpleNamespace(
            diagnostics=diag,
        ),
    )

    out_csv = tmp_path / "window.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_gx_runtime_window.py",
            "--gx-dir",
            str(gx_dir),
            "--gx-out",
            str(gx_out),
            "--config",
            str(tmp_path / "runtime.toml"),
            "--time-index-start",
            "0",
            "--time-index-stop",
            "1",
            "--steps",
            "50",
            "--out",
            str(out_csv),
        ],
    )

    mod.main()

    text = out_csv.read_text()
    assert "time_index_start,time_index_stop" in text
    assert "Wg" in text
