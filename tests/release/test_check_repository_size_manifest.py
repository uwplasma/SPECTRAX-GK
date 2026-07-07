"""Tests for the repository-size manifest gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap


def _load_tool_module(name: str):
    tools_dir = Path(__file__).resolve().parents[2] / "tools" / "release"
    sys.path.insert(0, str(tools_dir))
    path = tools_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _init_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "small.txt").write_text("small\n", encoding="utf-8")
    (tmp_path / "large.bin").write_bytes(b"0" * 64)
    subprocess.run(["git", "add", "small.txt", "large.bin"], cwd=tmp_path, check=True)


def test_repository_size_manifest_passes_for_allowed_large_file(tmp_path: Path) -> None:
    mod = _load_tool_module("check_repository_size_manifest")
    _init_repo(tmp_path)
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
    mod = _load_tool_module("check_repository_size_manifest")
    _init_repo(tmp_path)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        "[policy]\nmax_tracked_total_bytes = 1000\nmax_unlisted_tracked_file_bytes = 32\n",
        encoding="utf-8",
    )

    report = mod.check_repository_size_manifest(root=tmp_path, manifest=manifest)

    assert report["passed"] is False
    assert report["unlisted_large_files"] == [{"path": "large.bin", "bytes": 64}]
    assert any("large.bin" in failure for failure in report["failures"])
