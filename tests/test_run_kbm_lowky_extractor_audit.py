from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_kbm_lowky_extractor_audit import build_parser
import run_kbm_lowky_extractor_audit as mod


def test_run_kbm_lowky_extractor_audit_parser_defaults(tmp_path: Path) -> None:
    out = tmp_path / "branch.csv"
    args = build_parser().parse_args(["--gx", "kbm.out.nc", "--out", str(out)])
    assert args.gx == Path("kbm.out.nc")
    assert args.out == out
    assert args.ky == "0.3,0.4"
    assert args.gx_input is None


def test_run_kbm_lowky_uses_compare_gx_kbm_for_salpha(tmp_path: Path) -> None:
    gx = tmp_path / "kbm_salpha_correct.out.nc"
    gx.write_text("")
    gx_input = tmp_path / "kbm_salpha.in"
    gx_input.write_text("[Geometry]\ngeo_option = 's-alpha'\n")
    args = build_parser().parse_args(["--gx", str(gx), "--gx-input", str(gx_input), "--out", str(tmp_path / "out.csv")])
    cmd = mod._build_command(args, here=Path("/toolroot"), gx=gx, gx_input=gx_input, gx_big=tmp_path / "gx.big.nc")
    assert cmd[1] == "/toolroot/compare_gx_kbm.py"
    assert "--candidate-out" in cmd


def test_run_kbm_lowky_uses_imported_audit_for_miller(tmp_path: Path) -> None:
    gx = tmp_path / "kbm_miller_correct.out.nc"
    gx.write_text("")
    gx_input = tmp_path / "kbm_miller.in"
    gx_input.write_text("[Geometry]\ngeo_option = 'miller'\n")
    args = build_parser().parse_args(
        [
            "--gx",
            str(gx),
            "--gx-input",
            str(gx_input),
            "--out",
            str(tmp_path / "out.csv"),
            "--sample-step-stride",
            "2",
            "--max-samples",
            "16",
        ]
    )
    cmd = mod._build_command(args, here=Path("/toolroot"), gx=gx, gx_input=gx_input, gx_big=tmp_path / "gx.big.nc")
    assert cmd[1] == "/toolroot/run_imported_linear_targeted_audit.py"
    assert "--geometry-file" in cmd
    assert "--sample-step-stride" in cmd
    assert "--max-samples" in cmd
