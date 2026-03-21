from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_gx_imported_window import build_parser


def test_compare_gx_imported_window_parser_accepts_required_args() -> None:
    args = build_parser().parse_args(
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
