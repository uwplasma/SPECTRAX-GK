"""Tests for checked-in documentation preview compression."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from PIL import Image


def _load_tool_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "compress_docs_previews.py"
    spec = importlib.util.spec_from_file_location("compress_docs_previews", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compress_docs_previews_skips_release_manifest_paths(tmp_path: Path) -> None:
    mod = _load_tool_module()
    static = tmp_path / "docs" / "_static"
    static.mkdir(parents=True)
    keep = static / "keep.png"
    trim = static / "trim.png"
    Image.new("RGBA", (128, 64), (255, 255, 255, 255)).save(keep)
    Image.new("RGBA", (128, 64), (200, 220, 255, 255)).save(trim)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        f"""
[[artifacts]]
path = "{keep.as_posix()}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    reports = mod.compress_docs_previews(
        static_dir=static,
        manifest=manifest,
        min_bytes=1,
        max_width=32,
        colors=8,
    )

    by_name = {Path(row["path"]).name: row for row in reports}
    assert by_name["keep.png"]["skipped"] is True
    assert by_name["trim.png"]["skipped"] is False
    assert Image.open(keep).size == (128, 64)
    assert Image.open(trim).size == (32, 16)


def test_compress_png_preview_dry_run_does_not_modify_file(tmp_path: Path) -> None:
    mod = _load_tool_module()
    path = tmp_path / "panel.png"
    Image.new("RGBA", (64, 32), (255, 255, 255, 255)).save(path)
    before = path.read_bytes()

    report = mod.compress_png_preview(path, max_width=16, colors=8, dry_run=True)

    assert report["dry_run"] is True
    assert report["saved_bytes"] == 0
    assert path.read_bytes() == before
