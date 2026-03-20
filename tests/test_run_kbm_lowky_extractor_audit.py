from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_kbm_lowky_extractor_audit import build_parser


def test_run_kbm_lowky_extractor_audit_parser_defaults(tmp_path: Path) -> None:
    out = tmp_path / "branch.csv"
    args = build_parser().parse_args(["--gx", "kbm.out.nc", "--out", str(out)])
    assert args.gx == Path("kbm.out.nc")
    assert args.out == out
    assert args.ky == "0.3,0.4"
