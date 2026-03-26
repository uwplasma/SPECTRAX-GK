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
    assert args.Nl is None
    assert args.Nm is None
    assert args.sample_step_stride == 1
    assert args.max_samples is None
    assert args.sample_window == "head"


def test_run_imported_linear_targeted_audit_parser_accepts_inner_cache_controls(tmp_path: Path) -> None:
    out = tmp_path / "combined.csv"
    args = build_parser().parse_args(
        [
            "--gx",
            "gx.out.nc",
            "--geometry-file",
            "geom.nc",
            "--out",
            str(out),
            "--sample-step-stride",
            "4",
            "--max-samples",
            "16",
            "--sample-window",
            "tail",
            "--reuse-cache",
        ]
    )
    assert args.sample_step_stride == 4
    assert args.max_samples == 16
    assert args.sample_window == "tail"
    assert args.reuse_cache is True


def test_run_imported_linear_targeted_audit_parser_accepts_project_mode_method(tmp_path: Path) -> None:
    out = tmp_path / "combined.csv"
    args = build_parser().parse_args(
        [
            "--gx",
            "gx.out.nc",
            "--geometry-file",
            "geom.nc",
            "--out",
            str(out),
            "--mode-method",
            "project",
        ]
    )
    assert args.mode_method == "project"
