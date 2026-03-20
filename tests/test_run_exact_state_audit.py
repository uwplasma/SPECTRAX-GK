"""Parser smoke test for the exact-state audit orchestrator."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_exact_state_audit import _resolve_manifest_path, build_parser


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


def test_resolve_manifest_path_handles_relative_and_env_paths(tmp_path: Path, monkeypatch) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    monkeypatch.setenv("SPECTRAX_AUDIT_ROOT", str(tmp_path / "audit_root"))

    rel = _resolve_manifest_path("../configs/lane.toml", manifest_dir=manifest_dir)
    env_rel = _resolve_manifest_path("$SPECTRAX_AUDIT_ROOT/dumps", manifest_dir=manifest_dir)

    assert rel == (tmp_path / "configs" / "lane.toml").resolve()
    assert env_rel == (tmp_path / "audit_root" / "dumps").resolve()
