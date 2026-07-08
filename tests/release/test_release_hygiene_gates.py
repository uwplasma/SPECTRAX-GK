from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import textwrap

import pytest

from support.paths import load_release_tool
from tools.artifacts.build_technical_release_status import (
    LANES,
    build_technical_release_status,
)
from tools.release.check_release_version import (
    ReleaseVersionError,
    default_tag_from_github_env,
    normalize_tag,
    read_project_version,
    read_source_version,
    validate_release_version,
)


def _write_version_files(
    root: Path, *, project: str = "1.2.3", source: str = "1.2.3"
) -> None:
    (root / "src" / "spectraxgk").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "spectraxgk"
            version = "{project}"
            """
        ).strip(),
        encoding="utf-8",
    )
    (root / "src" / "spectraxgk" / "_version.py").write_text(
        f'__version__ = "{source}"\n',
        encoding="utf-8",
    )


def test_release_version_accepts_matching_project_source_and_tag(
    tmp_path: Path,
) -> None:
    _write_version_files(tmp_path, project="2.0.1", source="2.0.1")

    report = validate_release_version(
        root=tmp_path,
        tag="refs/tags/v2.0.1",
        require_tag=True,
        pypi_versions={"1.5.0", "2.0.0"},
    )

    assert report["project_version"] == "2.0.1"
    assert report["source_version"] == "2.0.1"
    assert report["tag"] == "v2.0.1"
    assert report["checked_pypi"] is True


def test_release_version_rejects_source_pyproject_mismatch(tmp_path: Path) -> None:
    _write_version_files(tmp_path, project="2.0.1", source="2.0.0")

    with pytest.raises(ReleaseVersionError, match="_version.py"):
        validate_release_version(root=tmp_path)


def test_release_version_rejects_wrong_or_missing_tag(tmp_path: Path) -> None:
    _write_version_files(tmp_path, project="2.0.1", source="2.0.1")

    with pytest.raises(ReleaseVersionError, match="expected 'v2.0.1'"):
        validate_release_version(root=tmp_path, tag="v2.0.0", require_tag=True)
    with pytest.raises(ReleaseVersionError, match="requires a tag"):
        validate_release_version(root=tmp_path, tag=None, require_tag=True)


def test_release_version_rejects_duplicate_pypi_version(tmp_path: Path) -> None:
    _write_version_files(tmp_path, project="2.0.1", source="2.0.1")

    with pytest.raises(ReleaseVersionError, match="already exists on PyPI"):
        validate_release_version(
            root=tmp_path, tag="v2.0.1", require_tag=True, pypi_versions={"2.0.1"}
        )


def test_release_version_readers_and_tag_normalization(tmp_path: Path) -> None:
    _write_version_files(tmp_path, project="3.4.5", source="3.4.5")

    assert read_project_version(tmp_path) == "3.4.5"
    assert read_source_version(tmp_path) == "3.4.5"
    assert normalize_tag("refs/tags/v3.4.5") == "v3.4.5"
    assert normalize_tag("") is None


def test_default_tag_from_github_env_ignores_branch_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    monkeypatch.setenv("GITHUB_REF_TYPE", "branch")

    assert default_tag_from_github_env() is None

    monkeypatch.setenv("GITHUB_REF_NAME", "v2.0.1")
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")

    assert default_tag_from_github_env() == "v2.0.1"


def _release_artifact_manifest(
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
    mod = load_release_tool("check_release_artifact_manifest")
    payload = b"panel"
    (tmp_path / "panel.png").write_bytes(payload)
    manifest = _release_artifact_manifest(
        tmp_path, sha=hashlib.sha256(payload).hexdigest(), size=len(payload)
    )

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is True
    assert report["move_to_release_bytes"] == len(payload)


def test_release_artifact_manifest_accepts_kept_preview_action(tmp_path: Path) -> None:
    mod = load_release_tool("check_release_artifact_manifest")
    payload = b"preview"
    (tmp_path / "panel.png").write_bytes(payload)
    manifest = _release_artifact_manifest(
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
    mod = load_release_tool("check_release_artifact_manifest")
    payload = b"panel"
    (tmp_path / "panel.png").write_bytes(payload)
    manifest = _release_artifact_manifest(tmp_path, sha="0" * 64, size=len(payload))

    report = mod.check_release_artifact_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is False
    assert any("sha256" in failure for failure in report["failures"])


def test_release_artifact_manifest_accepts_uploaded_release_asset(
    tmp_path: Path,
) -> None:
    mod = load_release_tool("check_release_artifact_manifest")
    payload = b"panel"
    manifest = _release_artifact_manifest(
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
    mod = load_release_tool("check_release_artifact_manifest")
    payload = b"panel"
    manifest = _release_artifact_manifest(
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


def _init_size_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "small.txt").write_text("small\n", encoding="utf-8")
    (tmp_path / "large.bin").write_bytes(b"0" * 64)
    subprocess.run(["git", "add", "small.txt", "large.bin"], cwd=tmp_path, check=True)


def test_repository_size_manifest_passes_for_allowed_large_file(tmp_path: Path) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    _init_size_repo(tmp_path)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        textwrap.dedent(
            """
            [policy]
            max_tracked_total_bytes = 1000
            max_unlisted_tracked_file_bytes = 32

            [[allowed_large_files]]
            path = "large.bin"
            max_bytes = 128
            reason = "test fixture"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    report = mod.check_repository_size_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is True
    assert report["unlisted_large_files"] == []


def test_repository_size_manifest_fails_for_unlisted_large_file(tmp_path: Path) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    _init_size_repo(tmp_path)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        "[policy]\nmax_tracked_total_bytes = 1000\nmax_unlisted_tracked_file_bytes = 32\n",
        encoding="utf-8",
    )

    report = mod.check_repository_size_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is False
    assert report["unlisted_large_files"] == [{"path": "large.bin", "bytes": 64}]
    assert any("large.bin" in failure for failure in report["failures"])


def test_repository_size_report_separates_tracked_and_local_roots(
    tmp_path: Path,
) -> None:
    mod = load_release_tool("audit_repository_size")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "docs" / "_static").mkdir(parents=True)
    (tmp_path / "tools_out").mkdir()
    (tmp_path / "src" / "small.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "docs" / "_static" / "panel.png").write_bytes(b"0" * 128)
    (tmp_path / "tools_out" / "scratch.nc").write_bytes(b"1" * 256)
    subprocess.run(
        ["git", "add", "src/small.py", "docs/_static/panel.png"],
        cwd=tmp_path,
        check=True,
    )

    report = mod.build_repository_size_report(tmp_path, top_n=1)

    assert report["kind"] == "repository_size_audit"
    assert report["tracked_file_count"] == 2
    assert report["largest_tracked_files"][0]["path"] == "docs/_static/panel.png"
    assert report["tracked_by_category"]["docs/_static"] == 128
    local = {row["path"]: row for row in report["local_artifact_roots"]}
    assert local["tools_out"]["bytes"] == 256


def _write_minimal_release_status_tree(root: Path) -> None:
    text_by_path: dict[Path, list[str]] = {}
    for checks in LANES.values():
        for check in checks:
            path = root / check.path
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() in {".png", ".pdf"}:
                path.write_bytes(b"artifact")
                continue
            text_by_path.setdefault(path, []).append(check.snippet or "present")
    for path, snippets in text_by_path.items():
        path.write_text("\n".join(snippets) + "\n", encoding="utf-8")


def test_technical_release_status_passes_complete_evidence_tree(tmp_path: Path) -> None:
    _write_minimal_release_status_tree(tmp_path)

    report = build_technical_release_status(tmp_path)

    assert report["passed"] is True
    assert report["technical_release_completion_percent"] == 100.0
    assert not report["failed_required"]
    assert set(report["lanes"]) == set(LANES)


def test_technical_release_status_reports_missing_required_evidence(
    tmp_path: Path,
) -> None:
    _write_minimal_release_status_tree(tmp_path)
    (tmp_path / "docs" / "parallelization.rst").unlink()

    report = build_technical_release_status(tmp_path)

    assert report["passed"] is False
    assert report["technical_release_completion_percent"] < 100.0
    assert any(
        "parallelization_release_surface" in item for item in report["failed_required"]
    )
