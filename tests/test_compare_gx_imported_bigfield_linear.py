from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_gx_imported_bigfield_linear import build_parser


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
