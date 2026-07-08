"""Tests for the release-artifact manifest checker."""

from __future__ import annotations

from support.paths import load_release_tool
import hashlib
from pathlib import Path
import textwrap


def _load_tool_module():
    return load_release_tool("check_release_artifact_manifest")


def _manifest(
    tmp_path: Path, *, sha: str, size: int, action: str = "move_to_release"
) -> Path:
    release_fields = (
        '\nrelease_tag = "v-test"\nrelease_url = "https://example.test/download/panel.png"'
        if action == "move_to_release"
        else ""
    )
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
            action = "{action}"
            artifact_type = "panel"
            release_asset_name = "panel.png"
            reason = "test panel"
            preview_strategy = "test preview"
            replay_command = "python make_panel.py"
            {release_fields}
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
    manifest = _manifest(
        tmp_path, sha=hashlib.sha256(payload).hexdigest(), size=len(payload)
    )

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is True
    assert report["move_to_release_bytes"] == len(payload)


def test_release_artifact_manifest_accepts_kept_preview_action(tmp_path: Path) -> None:
    mod = _load_tool_module()
    payload = b"preview"
    (tmp_path / "panel.png").write_bytes(payload)
    manifest = _manifest(
        tmp_path,
        sha=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        action="keep_preview_in_repo",
    )

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is True
    assert report["move_to_release_bytes"] == 0
    assert report["artifacts"][0]["action"] == "keep_preview_in_repo"


def test_release_artifact_manifest_fails_on_sha_mismatch(tmp_path: Path) -> None:
    mod = _load_tool_module()
    payload = b"panel"
    (tmp_path / "panel.png").write_bytes(payload)
    manifest = _manifest(tmp_path, sha="0" * 64, size=len(payload))

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is False
    assert any("sha256" in failure for failure in report["failures"])


def test_release_artifact_manifest_accepts_uploaded_release_asset(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    payload = b"panel"
    manifest = _manifest(
        tmp_path, sha=hashlib.sha256(payload).hexdigest(), size=len(payload)
    )

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is True
    assert report["move_to_release_bytes"] == len(payload)
    assert report["artifacts"][0]["exists"] is False
    assert report["artifacts"][0]["release_tag"] == "v-test"
    assert report["artifacts"][0]["release_url"].endswith("/panel.png")


def test_release_artifact_manifest_requires_url_for_missing_moved_asset(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    payload = b"panel"
    manifest = _manifest(
        tmp_path, sha=hashlib.sha256(payload).hexdigest(), size=len(payload)
    )
    text = manifest.read_text(encoding="utf-8")
    text = "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("release_tag", "release_url"))
    )
    manifest.write_text(text + "\n", encoding="utf-8")

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is False
    assert any("does not exist" in failure for failure in report["failures"])
