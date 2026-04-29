"""Tests for the release-artifact manifest checker."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import textwrap


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "check_release_artifact_manifest.py"
    spec = importlib.util.spec_from_file_location("check_release_artifact_manifest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path: Path, *, sha: str, size: int) -> Path:
    manifest = tmp_path / "release_artifacts.toml"
    manifest.write_text(
        textwrap.dedent(
            f"""
            [policy]
            release_series = "test"
            default_destination = "GitHub Releases"
            status = "planned"

            [[artifacts]]
            path = "panel.png"
            size_bytes = {size}
            sha256 = "{sha}"
            action = "move_to_release"
            artifact_type = "panel"
            release_asset_name = "panel.png"
            reason = "test panel"
            preview_strategy = "test preview"
            replay_command = "python make_panel.py"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_release_artifact_manifest_validates_size_and_sha(tmp_path: Path) -> None:
    mod = _load_tool_module()
    payload = b"panel"
    (tmp_path / "panel.png").write_bytes(payload)
    manifest = _manifest(tmp_path, sha=hashlib.sha256(payload).hexdigest(), size=len(payload))

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is True
    assert report["move_to_release_bytes"] == len(payload)


def test_release_artifact_manifest_fails_on_sha_mismatch(tmp_path: Path) -> None:
    mod = _load_tool_module()
    payload = b"panel"
    (tmp_path / "panel.png").write_bytes(payload)
    manifest = _manifest(tmp_path, sha="0" * 64, size=len(payload))

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is False
    assert any("sha256" in failure for failure in report["failures"])
