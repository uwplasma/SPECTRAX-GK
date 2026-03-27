from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_gx_imported_bigfield_linear import (
    _growth_rate_from_signal_sparse,
    _reduce_linear_grid_to_target_ky,
    build_parser,
)
from spectraxgk.analysis import ModeSelection
from spectraxgk.config import GridConfig
from spectraxgk.grids import build_spectral_grid


def test_compare_gx_imported_bigfield_linear_parser_accepts_tail_window() -> None:
    args = build_parser().parse_args(
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


def test_reduce_linear_grid_to_target_ky_slices_to_one_mode() -> None:
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=8, Nz=8, y0=10.0))
    reduced, ky_idx = _reduce_linear_grid_to_target_ky(grid, 2, init_single=False)
    assert reduced.ky.shape == (1,)
    assert jnp.asarray(reduced.kx_grid).shape[0] == 1
    assert ky_idx == 0


def test_reduce_linear_grid_to_target_ky_preserves_init_single_layout() -> None:
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=8, Nz=8, y0=10.0))
    reduced, ky_idx = _reduce_linear_grid_to_target_ky(grid, 2, init_single=True)
    assert reduced.ky.shape == grid.ky.shape
    assert ky_idx == 2


def test_growth_rate_from_signal_sparse_uses_direct_ratio_for_two_intervals() -> None:
    t = np.asarray([0.0, 2.0, 5.0], dtype=float)
    gamma_ref = 0.3
    omega_ref = -0.8
    signal = np.exp((gamma_ref - 1j * omega_ref) * t)
    gamma, omega, gamma_t, omega_t, t_mid = _growth_rate_from_signal_sparse(signal, t)
    assert np.isclose(gamma, gamma_ref)
    assert np.isclose(omega, omega_ref)
    assert gamma_t.shape == (2,)
    assert omega_t.shape == (2,)
    assert t_mid.shape == (2,)
