"""Parser smoke test for the exact-state audit orchestrator."""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from run_exact_state_audit import _resolve_manifest_path, _tool_env, build_parser


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


def test_exact_state_office_manifest_w7x_config_resolves_to_real_example() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "tools" / "exact_state_lanes.office.toml"
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    config = data["lane"]["w7x_vmec"]["config"]
    resolved = _resolve_manifest_path(config, manifest_dir=manifest.parent)
    assert resolved == (repo / "examples" / "nonlinear" / "non-axisymmetric" / "runtime_w7x_nonlinear_vmec_geometry.toml")
    assert resolved.is_file()


def test_exact_state_office_manifest_kbm_config_resolves_to_real_example() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "tools" / "exact_state_lanes.office.toml"
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    config = data["lane"]["kbm_salpha"]["config"]
    resolved = _resolve_manifest_path(config, manifest_dir=manifest.parent)
    assert resolved == (repo / "examples" / "nonlinear" / "axisymmetric" / "runtime_kbm_nonlinear_gx_t100.toml")
    assert resolved.is_file()


def test_exact_state_office_manifest_kbm_has_late_diag_state_lane() -> None:
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "tools" / "exact_state_lanes.office.toml"
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    diag_state = data["lane"]["kbm_salpha"]["diag_state"]
    assert diag_state["time_index"] == 130
    assert "kbm_diag_t130" in diag_state["gx_dir"]


def test_tool_env_prepends_absolute_repo_pythonpath(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYTHONPATH", "src:.")
    env = _tool_env(tmp_path)
    parts = env["PYTHONPATH"].split(":")
    assert parts[0] == str((tmp_path / "src").resolve())
    assert parts[1] == str(tmp_path.resolve())
    assert parts[2:] == ["src", "."]
