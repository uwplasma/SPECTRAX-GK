"""Contracts for active reference-comparison helper scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from support.paths import load_comparison_tool


def test_imported_window_parser_accepts_required_args() -> None:
    mod = load_comparison_tool("compare_gx_imported_window")
    args = mod.build_parser().parse_args(
        [
            "--gx-dir",
            "/tmp/gx",
            "--gx-out",
            "/tmp/run.out.nc",
            "--gx-input",
            "/tmp/run.in",
            "--geometry-file",
            "/tmp/run.eik.nc",
            "--time-index-start",
            "0",
            "--time-index-stop",
            "1",
        ]
    )
    assert args.gx_dir == Path("/tmp/gx")
    assert args.gx_out == Path("/tmp/run.out.nc")
    assert args.gx_input == Path("/tmp/run.in")
    assert args.geometry_file == Path("/tmp/run.eik.nc")
    assert args.time_index_start == 0
    assert args.time_index_stop == 1


def test_startup_select_ky_block_and_parser_contracts() -> None:
    mod = load_comparison_tool("compare_gx_startup")
    arr = np.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)

    sliced = mod._select_ky_block(arr, 2)

    assert sliced.shape == (2, 3, 1, 5, 6)
    assert np.array_equal(sliced[:, :, 0, :, :], arr[:, :, 2, :, :])

    args = mod.build_parser().parse_args(
        [
            "--gx-dir",
            "gx_dump",
            "--gx-out",
            "gx.out.nc",
            "--case",
            "kbm",
            "--ky",
            "0.3",
            "--Ny",
            "16",
            "--Nz",
            "96",
            "--Nl",
            "16",
            "--Nm",
            "48",
        ]
    )
    assert args.case == "kbm"
    assert args.gx_dir == Path("gx_dump")
    assert args.gx_out == Path("gx.out.nc")
    assert args.ky == 0.3


def test_nonlinear_comparison_loaders_and_late_window_stats(tmp_path: Path) -> None:
    mod = load_comparison_tool("compare_gx_nonlinear")
    csv_path = tmp_path / "restart.csv"
    t = np.linspace(5.0, 6.0, 4)
    data = np.column_stack(
        [
            t,
            np.linspace(0.1, 0.2, 4),
            np.linspace(-0.3, -0.2, 4),
            np.linspace(1.0, 2.0, 4),
            np.linspace(3.0, 4.0, 4),
            np.linspace(5.0, 6.0, 4),
            np.linspace(7.0, 8.0, 4),
            np.linspace(9.0, 10.0, 4),
        ]
    )
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        header="t,gamma,omega,Wg,Wphi,Wapar,heat_flux,particle_flux",
        comments="",
    )

    loaded = mod._load_spectrax(csv_path)

    assert np.allclose(loaded["t"], t)
    assert np.allclose(loaded["gamma"], data[:, 1])
    assert np.allclose(loaded["omega"], data[:, 2])
    assert np.allclose(loaded["Wg"], data[:, 3])
    assert np.allclose(loaded["Wphi"], data[:, 4])
    assert np.allclose(loaded["Wapar"], data[:, 5])
    assert np.allclose(loaded["heat"], data[:, 6])
    assert np.allclose(loaded["pflux"], data[:, 7])

    t_sp = np.linspace(0.0, 400.0, 3201)
    t_ref = np.linspace(0.0, 400.0, 801)
    phase = 0.9
    spectrax = 20.0 + 3.0 * np.sin(0.35 * t_sp)
    reference = 20.0 + 3.0 * np.sin(0.35 * t_ref + phase)

    pointwise = mod._relative_error_window(
        t_sp, spectrax, mod._interp(t_sp, t_ref, reference), tmin=20.0
    )
    _, _, stats = mod._stats_relative_errors(
        t_sp, spectrax, t_ref, reference, tmin=20.0
    )

    assert pointwise > 0.05
    assert stats["rel_mean"] < 1.0e-3
    assert stats["rel_std"] < 1.0e-2

    t_round = np.array([0.10000000149011612, 0.2], dtype=float)
    y = np.array([1.0, 2.0], dtype=float)
    mask = mod._window_mask(t_round, y, tmax=0.1)
    rel = mod._relative_error_window(t_round, y, y, tmax=0.1)
    abs_err = mod._absolute_error_window(t_round, y, y, tmax=0.1)

    assert np.array_equal(mask, np.array([True, False]))
    assert rel == 0.0
    assert abs_err == 0.0
