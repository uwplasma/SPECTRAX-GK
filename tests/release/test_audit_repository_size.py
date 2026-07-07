"""Tests for the repository-size audit helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "release"
        / "audit_repository_size.py"
    )
    spec = importlib.util.spec_from_file_location("audit_repository_size", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_size_report_separates_tracked_and_local_roots(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
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
