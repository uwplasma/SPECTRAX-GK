from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_gx_imported_growth_dump import build_parser


def test_compare_gx_imported_growth_dump_parser_accepts_required_paths() -> None:
    args = build_parser().parse_args(
        [
            "--gx-dir-start",
            "/tmp/start",
            "--gx-dir-stop",
            "/tmp/stop",
            "--gx-out",
            "/tmp/run.out.nc",
            "--gx-input",
            "/tmp/run.in",
            "--geometry-file",
            "/tmp/geom.nc",
            "--time-index-start",
            "10",
            "--time-index-stop",
            "11",
        ]
    )
    assert args.gx_dir_start == Path("/tmp/start")
    assert args.gx_dir_stop == Path("/tmp/stop")
    assert args.gx_out == Path("/tmp/run.out.nc")
    assert args.gx_input == Path("/tmp/run.in")
    assert args.geometry_file == Path("/tmp/geom.nc")
    assert args.time_index_start == 10
    assert args.time_index_stop == 11
