"""Contracts for reference-comparison helper scripts."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from support.paths import load_comparison_tool
from spectraxgk.config import GridConfig
from spectraxgk.core.grid import build_spectral_grid


def test_fieldsolve_dump_loads_two_block_complex_layout(tmp_path: Path) -> None:
    mod = load_comparison_tool("compare_gx_fieldsolve_dump")
    path = tmp_path / "field_nbar.bin"
    blocks = [
        np.arange(6, dtype=np.float32).astype(np.complex64),
        (10 + np.arange(6, dtype=np.float32)).astype(np.complex64),
    ]
    np.concatenate(blocks).astype(np.complex64).tofile(path)

    out = mod._load_complex_packed_fields(path, nyc=1, nx=1, nz=6)

    assert len(out) == 2
    np.testing.assert_allclose(out[0].reshape(-1), blocks[0])
    np.testing.assert_allclose(out[1].reshape(-1), blocks[1])


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


def test_linear_rk4_stage_reconstruction_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = load_comparison_tool("compare_gx_rk4_stage")
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

    def _fake_rhs(state, _cache, _params, *, terms):
        return 2.0 * state, terms

    monkeypatch.setattr(mod, "assemble_rhs_cached", _fake_rhs)
    computed = mod._compute_stage_states(
        np.array([1.0], dtype=np.float64),
        cache=None,
        params=None,
        term_cfg=None,
        dt=0.5,
    )

    assert np.allclose(computed.k1, np.array([2.0]))
    assert np.allclose(computed.g1, np.array([1.5]))
    assert np.allclose(computed.k2, np.array([3.0]))
    assert np.allclose(computed.g2, np.array([1.75]))
    assert np.allclose(computed.k3, np.array([3.5]))
    assert np.allclose(computed.g3, np.array([2.75]))
    assert np.allclose(computed.k4, np.array([5.5]))
    assert np.allclose(computed.g_next, np.array([2.7083333333333335]))


def test_nonlinear_rk4_stage_parser_and_partial_stage_contracts() -> None:
    mod = load_comparison_tool("compare_gx_nonlinear_rk4_stage")
    args = mod.build_parser().parse_args(
        ["--gx-dir", "gx_dump", "--config", "runtime.toml", "--partial-call", "1"]
    )
    assert args.gx_dir == Path("gx_dump")
    assert args.config == Path("runtime.toml")
    assert args.partial_call == 1

    stages = mod.NonlinearRK4StageStates(
        k1_linear=np.array([1.0]),
        k1_nonlinear=np.array([2.0]),
        k1_total=np.array([3.0]),
        k2_linear=np.array([4.0]),
        k2_nonlinear=np.array([5.0]),
        k2_total=np.array([6.0]),
        k3_linear=np.array([7.0]),
        k3_nonlinear=np.array([8.0]),
        k3_total=np.array([9.0]),
        k4_linear=np.array([10.0]),
        k4_nonlinear=np.array([11.0]),
        k4_total=np.array([12.0]),
        g2=np.array([13.0]),
        g3=np.array([14.0]),
        g4=np.array([15.0]),
    )

    g2, l2, n2, t2 = mod._partial_stage_targets(1, stages)
    g3, l3, n3, t3 = mod._partial_stage_targets(2, stages)
    g4, l4, n4, t4 = mod._partial_stage_targets(3, stages)

    assert np.array_equal(g2, np.array([13.0]))
    assert np.array_equal(l2, np.array([4.0]))
    assert np.array_equal(n2, np.array([5.0]))
    assert np.array_equal(t2, np.array([6.0]))
    assert np.array_equal(g3, np.array([14.0]))
    assert np.array_equal(l3, np.array([7.0]))
    assert np.array_equal(n3, np.array([8.0]))
    assert np.array_equal(t3, np.array([9.0]))
    assert np.array_equal(g4, np.array([15.0]))
    assert np.array_equal(l4, np.array([10.0]))
    assert np.array_equal(n4, np.array([11.0]))
    assert np.array_equal(t4, np.array([12.0]))
    with pytest.raises(ValueError, match="partial_call"):
        mod._partial_stage_targets(4, stages)


def test_imported_bigfield_linear_parser_grid_and_sparse_growth_contracts() -> None:
    mod = load_comparison_tool("compare_gx_imported_bigfield_linear")
    args = mod.build_parser().parse_args(
        [
            "--gx-big",
            "/tmp/run.big.nc",
            "--geometry-file",
            "/tmp/geom.nc",
            "--gx-input",
            "/tmp/run.in",
            "--ky",
            "0.3",
            "--sample-step-stride",
            "2",
            "--max-samples",
            "16",
            "--sample-window",
            "tail",
        ]
    )
    assert args.gx_big == Path("/tmp/run.big.nc")
    assert args.geometry_file == Path("/tmp/geom.nc")
    assert args.gx_input == Path("/tmp/run.in")
    assert args.ky == 0.3
    assert args.sample_window == "tail"

    grid = build_spectral_grid(GridConfig(Nx=4, Ny=8, Nz=8, y0=10.0))
    reduced, ky_idx = mod._reduce_linear_grid_to_target_ky(grid, 2, init_single=False)
    assert reduced.ky.shape == (1,)
    assert jnp.asarray(reduced.kx_grid).shape[0] == 1
    assert ky_idx == 0

    reduced, ky_idx = mod._reduce_linear_grid_to_target_ky(grid, 2, init_single=True)
    assert reduced.ky.shape == grid.ky.shape
    assert ky_idx == 2

    t = np.asarray([0.0, 2.0, 5.0], dtype=float)
    gamma_ref = 0.3
    omega_ref = -0.8
    signal = np.exp((gamma_ref - 1j * omega_ref) * t)
    gamma, omega, gamma_t, omega_t, t_mid = mod._growth_rate_from_signal_sparse(signal, t)
    assert np.isclose(gamma, gamma_ref)
    assert np.isclose(omega, omega_ref)
    assert gamma_t.shape == (2,)
    assert omega_t.shape == (2,)
    assert t_mid.shape == (2,)


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
    t_gx = np.linspace(0.0, 400.0, 801)
    phase = 0.9
    sp = 20.0 + 3.0 * np.sin(0.35 * t_sp)
    gx = 20.0 + 3.0 * np.sin(0.35 * t_gx + phase)

    pointwise = mod._relative_error_window(
        t_sp, sp, mod._interp(t_sp, t_gx, gx), tmin=20.0
    )
    _, _, stats = mod._stats_relative_errors(t_sp, sp, t_gx, gx, tmin=20.0)

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
