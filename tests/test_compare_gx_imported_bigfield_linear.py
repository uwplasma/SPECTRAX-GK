from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_gx_imported_bigfield_linear import _reduce_linear_grid_to_target_ky, build_parser
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
