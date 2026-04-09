from __future__ import annotations

from pathlib import Path
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_device_parity_gate import _resolve_manifest_path, build_parser


def test_run_device_parity_gate_parser_requires_manifest(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    ns = build_parser().parse_args(["--manifest", "lanes.toml", "--outdir", str(outdir)])
    assert ns.manifest == Path("lanes.toml")
    assert ns.outdir == outdir


def test_device_parity_office_manifest_w7x_config_resolves_to_real_example() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "tools" / "device_parity_lanes.office.toml"
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    config = data["lane"]["w7x_vmec"]["config"]
    resolved = _resolve_manifest_path(config, manifest_dir=manifest.parent)
    assert resolved == (repo / "examples" / "nonlinear" / "non-axisymmetric" / "runtime_w7x_nonlinear_vmec_geometry.toml")
    assert resolved.is_file()
