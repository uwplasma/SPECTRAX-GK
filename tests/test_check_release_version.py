from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tools.check_release_version import (
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
