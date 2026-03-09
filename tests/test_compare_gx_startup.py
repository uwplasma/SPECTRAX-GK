"""Tests for GX startup-state comparison tooling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def test_compare_gx_startup_select_ky_block_slices_third_to_last_axis() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_startup as mod
    finally:
        sys.path.remove(str(tools_dir))

    arr = np.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
    sliced = mod._select_ky_block(arr, 2)

    assert sliced.shape == (2, 3, 1, 5, 6)
    assert np.array_equal(sliced[:, :, 0, :, :], arr[:, :, 2, :, :])


def test_compare_gx_startup_parser_requires_core_args() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_startup as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(
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
