#!/usr/bin/env python3
"""Compress large checked-in documentation PNG previews.

This keeps Git-hosted documentation lightweight while leaving raw solver outputs,
JSON/CSV evidence, and manifest-pinned release previews untouched by default.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tomllib
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_DIR = ROOT / "docs" / "_static"
DEFAULT_RELEASE_MANIFEST = ROOT / "tools" / "release_artifact_manifest.toml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_manifest_paths(manifest: str | Path = DEFAULT_RELEASE_MANIFEST) -> set[Path]:
    """Return repo-relative paths pinned by the release-artifact manifest."""

    path = Path(manifest)
    if not path.exists():
        return set()
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    paths = set()
    for item in data.get("artifacts", []):
        raw = item.get("path")
        if isinstance(raw, str) and raw:
            paths.add(Path(raw))
    return paths


def compress_png_preview(path: str | Path, *, max_width: int = 1800, colors: int = 192, dry_run: bool = False) -> dict[str, Any]:
    """Compress one PNG preview and return a JSON-ready report."""

    target = Path(path)
    before_size = target.stat().st_size
    before_sha = _sha256(target)
    image = Image.open(target).convert("RGBA")
    original_dimensions = image.size
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    rgb = background.convert("RGB")
    report: dict[str, Any] = {
        "path": str(target),
        "original_dimensions": original_dimensions,
        "preview_dimensions": image.size,
        "before_size_bytes": before_size,
        "before_sha256": before_sha,
        "dry_run": dry_run,
    }
    if dry_run:
        report.update({"after_size_bytes": before_size, "after_sha256": before_sha, "saved_bytes": 0})
        return report
    original_bytes = target.read_bytes()
    palette = rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    palette.save(target, optimize=True)
    after_size = target.stat().st_size
    if after_size >= before_size:
        target.write_bytes(original_bytes)
        after_size = before_size
    after_sha = _sha256(target)
    report.update(
        {
            "after_size_bytes": after_size,
            "after_sha256": after_sha,
            "saved_bytes": before_size - after_size,
        }
    )
    return report


def compress_docs_previews(
    *,
    static_dir: str | Path = DEFAULT_STATIC_DIR,
    manifest: str | Path = DEFAULT_RELEASE_MANIFEST,
    min_bytes: int = 300_000,
    max_width: int = 1800,
    colors: int = 192,
    dry_run: bool = False,
    include_manifest_paths: bool = False,
) -> list[dict[str, Any]]:
    """Compress large docs PNG previews, skipping release-manifest paths by default."""

    static = Path(static_dir)
    manifest_rel = release_manifest_paths(manifest) if not include_manifest_paths else set()
    reports = []
    for path in sorted(static.glob("*.png")):
        if path.stat().st_size < min_bytes:
            continue
        try:
            rel = path.resolve().relative_to(ROOT)
        except ValueError:
            rel = path
        if rel in manifest_rel or path in manifest_rel:
            reports.append(
                {
                    "path": str(path),
                    "skipped": True,
                    "reason": "release_artifact_manifest",
                    "before_size_bytes": path.stat().st_size,
                    "after_size_bytes": path.stat().st_size,
                    "saved_bytes": 0,
                }
            )
            continue
        report = compress_png_preview(path, max_width=max_width, colors=colors, dry_run=dry_run)
        report["skipped"] = False
        reports.append(report)
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-dir", default=str(DEFAULT_STATIC_DIR), help="docs/_static directory to scan.")
    parser.add_argument("--manifest", default=str(DEFAULT_RELEASE_MANIFEST), help="Release manifest paths to skip by default.")
    parser.add_argument("--min-bytes", type=int, default=300_000, help="Only compress PNGs at or above this size.")
    parser.add_argument("--max-width", type=int, default=1800, help="Maximum preview width in pixels.")
    parser.add_argument("--colors", type=int, default=192, help="Palette colors for compressed previews.")
    parser.add_argument("--dry-run", action="store_true", help="Report candidate previews without modifying files.")
    parser.add_argument(
        "--include-release-manifest-paths",
        action="store_true",
        help="Also compress files pinned by tools/release_artifact_manifest.toml.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = compress_docs_previews(
        static_dir=args.static_dir,
        manifest=args.manifest,
        min_bytes=int(args.min_bytes),
        max_width=int(args.max_width),
        colors=int(args.colors),
        dry_run=bool(args.dry_run),
        include_manifest_paths=bool(args.include_release_manifest_paths),
    )
    total_saved = 0
    for report in reports:
        if report.get("skipped"):
            print(f"skip {report['path']}: {report['reason']}")
            continue
        saved = int(report["saved_bytes"])
        total_saved += saved
        print(
            "{path}: {before} -> {after} bytes, saved {saved}".format(
                path=report["path"],
                before=report["before_size_bytes"],
                after=report["after_size_bytes"],
                saved=saved,
            )
        )
    print(f"total_saved={total_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
