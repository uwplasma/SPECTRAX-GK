"""Parser smoke test for the exact-state audit orchestrator."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_exact_state_audit import build_parser


def test_run_exact_state_audit_parser_accepts_core_args() -> None:
    args = build_parser().parse_args(
        [
            "--manifest",
            "/tmp/lanes.toml",
            "--lane",
            "cyclone_miller",
            "--outdir",
            "/tmp/out",
        ]
    )
    assert args.manifest == Path("/tmp/lanes.toml")
    assert args.lane == "cyclone_miller"
    assert args.outdir == Path("/tmp/out")

