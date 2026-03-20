from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_vmec_roundtrip_gate import build_parser


def test_run_vmec_roundtrip_gate_parser_requires_manifest(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    ns = build_parser().parse_args(["--manifest", "lanes.toml", "--outdir", str(outdir)])
    assert ns.manifest == Path("lanes.toml")
    assert ns.outdir == outdir
