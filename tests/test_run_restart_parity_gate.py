from __future__ import annotations

import sys
from pathlib import Path
import tomllib


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_restart_parity_gate import _resolve_manifest_path, build_parser


def test_run_restart_parity_gate_parser_accepts_core_args() -> None:
    parser = build_parser()

    args = parser.parse_args(["--manifest", "tools/restart_gate_lanes.office.toml", "--lane", "kbm_salpha"])

    assert args.manifest == Path("tools/restart_gate_lanes.office.toml")
    assert args.lane == "kbm_salpha"


def test_restart_gate_office_manifest_w7x_config_resolves_to_real_example() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "tools" / "restart_gate_lanes.office.toml"
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    config = data["lane"]["w7x_vmec"]["config"]
    resolved = _resolve_manifest_path(config, manifest_dir=manifest.parent)
    assert resolved == (repo / "examples" / "nonlinear" / "non-axisymmetric" / "runtime_w7x_nonlinear_vmec_geometry.toml")
    assert resolved.is_file()


def test_restart_gate_office_manifest_kbm_config_resolves_to_real_example() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "tools" / "restart_gate_lanes.office.toml"
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    config = data["lane"]["kbm_salpha"]["config"]
    resolved = _resolve_manifest_path(config, manifest_dir=manifest.parent)
    assert resolved == (repo / "examples" / "nonlinear" / "axisymmetric" / "runtime_kbm_nonlinear_t100.toml")
    assert resolved.is_file()
