"""Tests for runtime-configured GX startup-state comparison tooling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset


def test_compare_gx_runtime_startup_select_ky_block_slices_third_to_last_axis() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_runtime_startup as mod
    finally:
        sys.path.remove(str(tools_dir))

    arr = np.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
    sliced = mod._select_ky_block(arr, 1)

    assert sliced.shape == (2, 3, 1, 5, 6)
    assert np.array_equal(sliced[:, :, 0, :, :], arr[:, :, 1, :, :])


def test_compare_gx_runtime_startup_infers_full_ny_from_positive_ky() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_runtime_startup as mod
    finally:
        sys.path.remove(str(tools_dir))

    assert mod._full_ny_from_positive_ky(np.array([0.1, 0.2, 0.3, 0.4])) == 10


def test_compare_gx_runtime_startup_parser_requires_core_args() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_runtime_startup as mod
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
            "0.3",
        ]
    )

    assert args.gx_dir == Path("gx_dump")
    assert args.gx_out == Path("gx.out.nc")
    assert args.config == Path("runtime.toml")
    assert args.ky == 0.3


def test_compare_gx_runtime_startup_builds_full_grid_before_slicing(
    tmp_path: Path, monkeypatch
) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_runtime_startup as mod
    finally:
        sys.path.remove(str(tools_dir))

    gx_out = tmp_path / "gx.out.nc"
    with Dataset(gx_out, "w") as ds:
        grids = ds.createGroup("Grids")
        grids.createDimension("ky", 2)
        ky = grids.createVariable("ky", "f8", ("ky",))
        ky[:] = [0.1, 0.2]

    monkeypatch.setattr(
        mod,
        "_load_shape",
        lambda _path: {"nspec": 1, "nl": 1, "nm": 1, "nyc": 2, "nx": 1, "nz": 2},
    )
    monkeypatch.setattr(mod, "_load_bin", lambda *_args, **_kwargs: np.zeros(4, dtype=np.complex64))
    gx_g = np.arange(4, dtype=np.complex64).reshape(1, 1, 1, 2, 1, 2)
    gx_phi = np.arange(4, dtype=np.complex64).reshape(2, 1, 2)
    monkeypatch.setattr(mod, "_reshape_gx", lambda *_args, **_kwargs: gx_g)
    monkeypatch.setattr(mod, "_load_field", lambda *_args, **_kwargs: gx_phi)
    monkeypatch.setattr(
        mod,
        "replace",
        lambda obj, **updates: SimpleNamespace(**(obj.__dict__ | updates)),
    )
    monkeypatch.setattr(mod, "load_runtime_from_toml", lambda _path: (SimpleNamespace(grid=SimpleNamespace(Nx=1, Ny=4, Nz=2, y0=10.0), species=[object()]), None))
    monkeypatch.setattr(mod, "build_runtime_geometry", lambda _cfg: object())
    monkeypatch.setattr(mod, "apply_gx_geometry_grid_defaults", lambda _geom, grid: grid)
    grid_full = SimpleNamespace(ky=np.array([0.1, 0.2]), kx=np.array([0.0]))
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_full)
    monkeypatch.setattr(mod, "build_runtime_linear_params", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod, "_species_to_linear", lambda _species: [object()])

    captured: dict[str, object] = {}

    def _fake_build_initial_condition(grid, _geom, _cfg, *, ky_index, kx_index, Nl, Nm, nspecies):
        captured["grid"] = grid
        captured["ky_index"] = ky_index
        captured["kx_index"] = kx_index
        captured["Nl"] = Nl
        captured["Nm"] = Nm
        captured["nspecies"] = nspecies
        return np.arange(4, dtype=np.complex64).reshape(1, 1, 1, 2, 1, 2)

    monkeypatch.setattr(mod, "_build_initial_condition", _fake_build_initial_condition)
    monkeypatch.setattr(mod, "build_linear_cache", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        mod,
        "compute_fields_cached",
        lambda *_args, **_kwargs: SimpleNamespace(phi=np.arange(4, dtype=np.complex64).reshape(2, 1, 2), apar=None),
    )

    summaries: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def _fake_summary(name, ref, test):
        summaries.append((name, tuple(np.asarray(ref).shape), tuple(np.asarray(test).shape)))

    monkeypatch.setattr(mod, "_summary", _fake_summary)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_gx_runtime_startup.py",
            "--gx-dir",
            str(tmp_path),
            "--gx-out",
            str(gx_out),
            "--config",
            str(tmp_path / "runtime.toml"),
            "--ky",
            "0.2",
        ],
    )

    mod.main()

    assert captured["grid"] is grid_full
    assert captured["ky_index"] == 1
    assert captured["kx_index"] == 0
    assert ("g_state", (1, 1, 1, 1, 1, 2), (1, 1, 1, 1, 1, 2)) in summaries
    assert ("phi", (1, 1, 2), (1, 1, 2)) in summaries
