from __future__ import annotations

import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_restart_parity_gate import build_parser


def test_run_restart_parity_gate_parser_accepts_core_args() -> None:
    parser = build_parser()

    args = parser.parse_args(["--manifest", "tools/restart_gate_lanes.office.toml", "--lane", "kbm_salpha"])

    assert args.manifest == Path("tools/restart_gate_lanes.office.toml")
    assert args.lane == "kbm_salpha"
