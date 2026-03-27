from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from compare_gx_imported_growth_dump import _load_growth_dt, build_parser


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


def test_load_growth_dt_accepts_float64_scalar(tmp_path: Path) -> None:
    path = tmp_path / "diag_growth_dt_t45.bin"
    import numpy as np

    np.asarray([2.5e-4], dtype=np.float64).tofile(path)
    assert _load_growth_dt(path) == 2.5e-4
