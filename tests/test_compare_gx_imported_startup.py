"""Tests for imported-geometry GX startup-state comparison tooling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset


def test_compare_gx_imported_startup_parser_requires_core_args() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_imported_startup as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(
        [
            "--gx-dir",
            "gx_dump",
            "--gx-out",
            "gx.out.nc",
            "--geometry-file",
            "geom.eik.nc",
            "--gx-input",
            "run.in",
            "--ky",
            "0.3",
        ]
    )

    assert args.gx_dir == Path("gx_dump")
    assert args.gx_out == Path("gx.out.nc")
    assert args.geometry_file == Path("geom.eik.nc")
    assert args.gx_input == Path("run.in")
    assert args.ky == 0.3


def test_compare_gx_imported_startup_builds_full_grid_before_slicing(
    tmp_path: Path, monkeypatch
) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_imported_startup as mod
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
        "_load_gx_input_contract",
        lambda _path: SimpleNamespace(
            boundary="linked",
            nperiod=2,
            ntheta=4,
            species=(object(),),
            tau_e=1.0,
            beta=0.0,
        ),
    )
    monkeypatch.setattr(mod, "_select_geometry_source", lambda gx_out, geom, _contract: geom)
    monkeypatch.setattr(mod, "load_gx_geometry_netcdf", lambda _path: SimpleNamespace(gradpar=lambda: 1.0))
    monkeypatch.setattr(mod, "apply_gx_geometry_grid_defaults", lambda _geom, grid: grid)
    grid_full = SimpleNamespace(ky=np.array([0.0, 0.1, 0.2, -0.1]), kx=np.array([0.0]))
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_full)
    captured: dict[str, object] = {}

    def _fake_build_imported_initial_condition(**kwargs):
        captured["grid"] = kwargs["grid"]
        captured["ky_index"] = kwargs["ky_index"]
        captured["kx_index"] = kwargs["kx_index"]
        return np.arange(8, dtype=np.complex64).reshape(1, 1, 1, 4, 1, 2)

    monkeypatch.setattr(mod, "_build_imported_initial_condition", _fake_build_imported_initial_condition)
    monkeypatch.setattr(mod, "build_linear_params", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod, "build_linear_cache", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        mod,
        "compute_fields_cached",
        lambda *_args, **_kwargs: SimpleNamespace(phi=np.arange(8, dtype=np.complex64).reshape(4, 1, 2), apar=None),
    )

    summaries: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    monkeypatch.setattr(
        mod,
        "_summary",
        lambda name, ref, test: summaries.append((name, tuple(np.asarray(ref).shape), tuple(np.asarray(test).shape))),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_gx_imported_startup.py",
            "--gx-dir",
            str(tmp_path),
            "--gx-out",
            str(gx_out),
            "--geometry-file",
            str(tmp_path / "geom.eik.nc"),
            "--gx-input",
            str(tmp_path / "run.in"),
            "--ky",
            "0.2",
        ],
    )

    mod.main()

    assert captured["grid"] is grid_full
    assert captured["ky_index"] == 2
    assert captured["kx_index"] == 0
    assert ("g_state", (1, 1, 1, 1, 1, 2), (1, 1, 1, 1, 1, 2)) in summaries
    assert ("phi", (1, 1, 2), (1, 1, 2)) in summaries
