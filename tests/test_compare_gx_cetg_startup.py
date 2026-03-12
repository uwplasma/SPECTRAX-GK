"""Tests for legacy GX cETG startup comparison tooling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _load_module():
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_cetg_startup as mod
    finally:
        sys.path.remove(str(tools_dir))
    return mod


def test_compare_gx_cetg_startup_parser_requires_core_args() -> None:
    mod = _load_module()

    parser = mod.build_parser()
    args = parser.parse_args(
        [
            "--gx-nc",
            "cetg_init.nc",
            "--gx-restart",
            "cetg_init.restart.nc",
            "--config",
            "runtime.toml",
        ]
    )

    assert args.gx_nc == Path("cetg_init.nc")
    assert args.gx_restart == Path("cetg_init.restart.nc")
    assert args.config == Path("runtime.toml")


def test_compare_gx_cetg_startup_match_output_kx_indices_tracks_sorted_active_grid() -> None:
    mod = _load_module()

    kx_full = np.array([0.0, 0.5, 1.0, -1.0, -0.5], dtype=float)
    kx_out = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)

    idx = mod._match_output_kx_indices(kx_full, kx_out)

    assert np.array_equal(idx, np.array([3, 4, 0, 1, 2], dtype=np.int32))


def test_compare_gx_cetg_startup_active_state_uses_gx_memory_order() -> None:
    mod = _load_module()

    full = np.zeros((1, 2, 1, 4, 8, 2), dtype=np.complex64)
    full[0, 0, 0, 1, 0, 0] = 1.0 + 0.0j
    full[0, 0, 0, 1, 1, 0] = 2.0 + 0.0j
    full[0, 0, 0, 1, 6, 0] = 3.0 + 0.0j
    full[0, 0, 0, 1, 7, 0] = 4.0 + 0.0j
    restart = mod.GXLegacyCetgRestart(
        time=0.0,
        state_active=np.zeros((1, 2, 1, 3, 5, 2), dtype=np.complex64),
        state_positive_ky=np.zeros((1, 2, 1, 3, 8, 2), dtype=np.complex64),
        nakx_active=5,
        naky_active=3,
    )

    active = mod._active_state_from_full_grid(full, restart)

    assert active.shape == (1, 2, 1, 3, 5, 2)
    assert active[0, 0, 0, 1, 0, 0] == 1.0 + 0.0j
    assert active[0, 0, 0, 1, 1, 0] == 2.0 + 0.0j
    assert active[0, 0, 0, 1, 3, 0] == 3.0 + 0.0j
    assert active[0, 0, 0, 1, 4, 0] == 4.0 + 0.0j
