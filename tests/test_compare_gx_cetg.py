"""Tests for the legacy GX cETG comparison tool."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_gx_cetg import build_parser


def test_compare_gx_cetg_parser_accepts_core_args() -> None:
    args = build_parser().parse_args(
        [
            "--gx-nc",
            "/tmp/cetg_smoke.nc",
            "--config",
            "/tmp/runtime_cetg.toml",
            "--sample-stride",
            "4",
        ]
    )

    assert args.gx_nc == Path("/tmp/cetg_smoke.nc")
    assert args.config == Path("/tmp/runtime_cetg.toml")
    assert args.sample_stride == 4
