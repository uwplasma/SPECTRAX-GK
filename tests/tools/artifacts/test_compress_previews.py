"""Tests for documentation and release preview compression tooling."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from support.paths import load_artifact_tool


def _load_tool_module():
    return load_artifact_tool("compress_previews")


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
        f'''
[[artifacts]]
path = "{keep.as_posix()}"
'''.strip()
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


def test_release_preview_targets_and_compression_use_manifest(tmp_path: Path) -> None:
    mod = _load_tool_module()
    panel = tmp_path / "panel.png"
    ignored = tmp_path / "ignored.png"
    Image.new("RGBA", (64, 32), (255, 255, 255, 255)).save(panel)
    Image.new("RGBA", (64, 32), (0, 0, 0, 255)).save(ignored)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        f'''
[[artifacts]]
path = "{panel.as_posix()}"
action = "keep_preview_in_repo"
preview_strategy = "compressed_preview"

[[artifacts]]
path = "{ignored.as_posix()}"
action = "keep_raw"
preview_strategy = "none"
'''.strip()
        + "\n",
        encoding="utf-8",
    )

    reports = mod.compress_release_previews(
        manifest=manifest,
        max_width=16,
        colors=8,
    )

    assert [Path(row["path"]).name for row in reports] == ["panel.png"]
    report = reports[0]
    assert report["original_dimensions"] == (64, 32)
    assert report["preview_dimensions"] == (16, 8)
    assert report["after_size_bytes"] > 0
    assert report["after_sha256"] != report["before_sha256"]
    assert Image.open(panel).size == (16, 8)
    assert Image.open(ignored).size == (64, 32)


def test_cli_supports_docs_and_release_modes(tmp_path: Path, capsys) -> None:
    mod = _load_tool_module()
    static = tmp_path / "docs" / "_static"
    static.mkdir(parents=True)
    panel = static / "panel.png"
    Image.new("RGBA", (64, 32), (255, 255, 255, 255)).save(panel)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        f'''
[[artifacts]]
path = "{panel.as_posix()}"
action = "keep_preview_in_repo"
preview_strategy = "preview"
'''.strip()
        + "\n",
        encoding="utf-8",
    )

    assert (
        mod.main(
            [
                "--mode",
                "docs",
                "--static-dir",
                str(static),
                "--manifest",
                str(manifest),
                "--min-bytes",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "total_saved=0" in capsys.readouterr().out
    assert (
        mod.main(
            [
                "--mode",
                "release",
                "--manifest",
                str(manifest),
                "--max-width",
                "32",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "panel.png" in capsys.readouterr().out
