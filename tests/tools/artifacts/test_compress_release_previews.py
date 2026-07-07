"""Tests for release-preview compression tooling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from PIL import Image


def _load_tool_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "compress_release_previews.py"
    spec = importlib.util.spec_from_file_location("compress_release_previews", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compress_png_preview_reduces_dimensions_and_reports_hash(tmp_path: Path) -> None:
    mod = _load_tool_module()
    path = tmp_path / "panel.png"
    image = Image.new("RGBA", (64, 32), (255, 255, 255, 255))
    image.save(path)

    report = mod.compress_png_preview(path, max_width=16, colors=8)

    assert report["original_dimensions"] == (64, 32)
    assert report["preview_dimensions"] == (16, 8)
    assert report["after_size_bytes"] > 0
    assert report["after_sha256"] != report["before_sha256"]
    assert Image.open(path).size == (16, 8)
