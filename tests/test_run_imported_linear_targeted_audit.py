from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_imported_linear_targeted_audit import build_parser


def test_run_imported_linear_targeted_audit_parser_defaults(tmp_path: Path) -> None:
    out = tmp_path / "combined.csv"
    args = build_parser().parse_args(
        [
            "--gx",
            "gx.out.nc",
            "--geometry-file",
            "geom.nc",
            "--out",
            str(out),
        ]
    )
    assert args.gx == Path("gx.out.nc")
    assert args.geometry_file == Path("geom.nc")
    assert args.out == out
    assert args.max_kys is None
