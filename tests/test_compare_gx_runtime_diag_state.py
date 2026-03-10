"""Tests for runtime-configured GX late-time diagnostic-state comparison tooling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset


def test_compare_gx_runtime_diag_state_parser_requires_core_args() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_runtime_diag_state as mod
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
            "--time-index",
            "10",
        ]
    )

    assert args.gx_dir == Path("gx_dump")
    assert args.gx_out == Path("gx.out.nc")
    assert args.config == Path("runtime.toml")
    assert args.time_index == 10


def test_compare_gx_runtime_diag_state_builds_positive_ky_grid_and_writes_csv(
    tmp_path: Path, monkeypatch
) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_runtime_diag_state as mod
    finally:
        sys.path.remove(str(tools_dir))

    gx_out = tmp_path / "gx.out.nc"
    with Dataset(gx_out, "w") as ds:
        ds.createDimension("l", 1)
        ds.createDimension("m", 1)
        ds.createDimension("s", 1)
        ds.createDimension("kx", 1)
        ds.createDimension("ky", 2)
        ds.createDimension("theta", 2)
        ds.createDimension("time", 1)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        ky = grids.createVariable("ky", "f8", ("ky",))
        ky[:] = [0.1, 0.2]
        kx = grids.createVariable("kx", "f8", ("kx",))
        kx[:] = [0.0]
        time = grids.createVariable("time", "f8", ("time",))
        time[:] = [3.0]
        for name, value in {
            "Wg_kyst": 1.5,
            "Wphi_kyst": 2.5,
            "Wapar_kyst": 0.0,
            "HeatFlux_kyst": 3.5,
            "ParticleFlux_kyst": 0.0,
        }.items():
            var = diag.createVariable(name, "f8", ("time", "s", "ky"))
            var[:] = value
        heat_s = diag.createVariable("HeatFlux_st", "f8", ("time", "s"))
        heat_s[:] = [[3.5]]
        pflux_s = diag.createVariable("ParticleFlux_st", "f8", ("time", "s"))
        pflux_s[:] = [[0.0]]

    gx_dir = tmp_path / "gx_dump"
    gx_dir.mkdir()
    np.arange(4, dtype=np.complex64).tofile(gx_dir / "diag_state_G_s0_t0.bin")
    np.arange(4, dtype=np.complex64).tofile(gx_dir / "diag_state_phi_t0.bin")
    np.arange(4, dtype=np.float32).tofile(gx_dir / "diag_state_kperp2_t0.bin")
    np.arange(2, dtype=np.float32).tofile(gx_dir / "diag_state_fluxfac_t0.bin")
    np.array([0.0], dtype=np.float32).tofile(gx_dir / "diag_state_kx_t0.bin")
    np.array([0.1, 0.2], dtype=np.float32).tofile(gx_dir / "diag_state_ky_t0.bin")

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
                grid=SimpleNamespace(Nx=1, Ny=4, Nz=2, y0=10.0),
                species=[object()],
                normalization=SimpleNamespace(flux_scale=1.0, wphi_scale=1.0),
            ),
            None,
        ),
    )
    monkeypatch.setattr(mod, "build_runtime_geometry", lambda _cfg: object())
    monkeypatch.setattr(mod, "ensure_flux_tube_geometry_data", lambda geom, _theta: geom)
    monkeypatch.setattr(mod, "apply_gx_geometry_grid_defaults", lambda _geom, grid: grid)
    grid_full = SimpleNamespace(ky=np.array([0.0, 0.1, 0.2, -0.1]), kx=np.array([0.0]), z=np.array([0.0, 1.0]))
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_full)
    grid_pos = SimpleNamespace(ky=np.array([0.1, 0.2]), kx=np.array([0.0]), z=np.array([0.0, 1.0]))
    captured: dict[str, object] = {}

    def _fake_select_gx_real_fft_ky_grid(grid, ky_vals):
        captured["grid"] = grid
        captured["ky_vals"] = np.asarray(ky_vals)
        return grid_pos

    monkeypatch.setattr(mod, "select_gx_real_fft_ky_grid", _fake_select_gx_real_fft_ky_grid)
    monkeypatch.setattr(mod, "build_runtime_linear_params", lambda *_args, **_kwargs: object())
    cache = SimpleNamespace(kperp2=np.arange(4, dtype=np.float32).reshape(2, 1, 2), bmag=np.ones(2), kperp2_bmag=False)
    monkeypatch.setattr(mod, "build_linear_cache", lambda *_args, **_kwargs: cache)
    monkeypatch.setattr(mod, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        mod,
        "compute_fields_cached",
        lambda *_args, **_kwargs: SimpleNamespace(phi=np.arange(4, dtype=np.complex64).reshape(2, 1, 2), apar=None, bpar=None),
    )
    monkeypatch.setattr(mod, "gx_volume_factors", lambda *_args, **_kwargs: (np.array([0.4, 0.6]), np.array([0.0, 1.0])))
    monkeypatch.setattr(mod, "gx_Wg", lambda *_args, **_kwargs: 1.5)
    monkeypatch.setattr(mod, "gx_Wphi", lambda *_args, **_kwargs: 2.5)
    monkeypatch.setattr(mod, "gx_Wapar", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(mod, "gx_heat_flux", lambda *_args, **_kwargs: 3.5)
    monkeypatch.setattr(mod, "gx_particle_flux", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(mod, "gx_heat_flux_species", lambda *_args, **_kwargs: np.array([3.5]))
    monkeypatch.setattr(mod, "gx_particle_flux_species", lambda *_args, **_kwargs: np.array([0.0]))

    summaries: list[str] = []
    monkeypatch.setattr(mod, "_summary", lambda name, *_args, **_kwargs: summaries.append(name))

    out_csv = tmp_path / "diag.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_gx_runtime_diag_state.py",
            "--gx-dir",
            str(gx_dir),
            "--gx-out",
            str(gx_out),
            "--config",
            str(tmp_path / "runtime.toml"),
            "--time-index",
            "0",
            "--out",
            str(out_csv),
        ],
    )

    mod.main()

    assert captured["grid"] is grid_full
    assert np.array_equal(captured["ky_vals"], np.array([0.1, 0.2], dtype=np.float32))
    assert summaries == ["kperp2", "fluxfac", "kx", "ky", "phi"]
    text = out_csv.read_text()
    assert "metric,gx_out,spectrax_dump" in text
    assert "Wg" in text
