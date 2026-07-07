#!/usr/bin/env python3
"""Compress lightweight PNG previews for documentation and release manifests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tomllib
from typing import Any, Literal

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_DIR = ROOT / "docs" / "_static"
DEFAULT_RELEASE_MANIFEST = ROOT / "tools" / "release_artifact_manifest.toml"
Mode = Literal["docs", "release"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def release_manifest_paths(
    manifest: str | Path = DEFAULT_RELEASE_MANIFEST,
) -> set[Path]:
    """Return repo-relative artifact paths pinned by the release manifest."""

    path = Path(manifest)
    if not path.exists():
        return set()
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    paths: set[Path] = set()
    for item in data.get("artifacts", []):
        raw = item.get("path")
        if isinstance(raw, str) and raw:
            paths.add(Path(raw))
    return paths


def release_preview_targets(
    manifest: str | Path = DEFAULT_RELEASE_MANIFEST,
) -> list[Path]:
    """Return repo-relative release-manifest PNG preview targets."""

    path = Path(manifest)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    targets: list[Path] = []
    for item in data.get("artifacts", []):
        rel = Path(str(item.get("path", "")))
        action = str(item.get("action", ""))
        preview_strategy = str(item.get("preview_strategy", ""))
        if rel.suffix.lower() != ".png":
            continue
        if action not in {"move_to_release", "keep_preview_in_repo"}:
            continue
        if "preview" in preview_strategy.lower() or action == "keep_preview_in_repo":
            targets.append(rel)
    return targets


def compress_png_preview(
    path: str | Path,
    *,
    max_width: int = 1800,
    colors: int = 192,
    dry_run: bool = False,
    restore_if_larger: bool = True,
) -> dict[str, Any]:
    """Compress one PNG preview and return size/checksum metadata."""

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
        report.update(
            {
                "after_size_bytes": before_size,
                "after_sha256": before_sha,
                "saved_bytes": 0,
            }
        )
        return report
    original_bytes = target.read_bytes() if restore_if_larger else None
    palette = rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    palette.save(target, optimize=True)
    after_size = target.stat().st_size
    if restore_if_larger and original_bytes is not None and after_size >= before_size:
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
    manifest_rel = (
        release_manifest_paths(manifest) if not include_manifest_paths else set()
    )
    reports: list[dict[str, Any]] = []
    for path in sorted(static.glob("*.png")):
        if path.stat().st_size < min_bytes:
            continue
        rel = _repo_relative(path)
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
        report = compress_png_preview(
            path,
            max_width=max_width,
            colors=colors,
            dry_run=dry_run,
            restore_if_larger=True,
        )
        report["skipped"] = False
        reports.append(report)
    return reports


def compress_release_previews(
    *,
    manifest: str | Path = DEFAULT_RELEASE_MANIFEST,
    max_width: int = 2200,
    colors: int = 192,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Compress PNG previews selected by the release artifact manifest."""

    reports: list[dict[str, Any]] = []
    for rel in release_preview_targets(manifest):
        path = rel if rel.is_absolute() else ROOT / rel
        report = compress_png_preview(
            path,
            max_width=max_width,
            colors=colors,
            dry_run=dry_run,
            restore_if_larger=False,
        )
        report["manifest_path"] = str(rel)
        report["skipped"] = False
        reports.append(report)
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("docs", "release"),
        default="docs",
        help="docs scans docs/_static; release follows tools/release_artifact_manifest.toml.",
    )
    parser.add_argument(
        "--static-dir",
        default=str(DEFAULT_STATIC_DIR),
        help="docs/_static directory to scan in docs mode.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_RELEASE_MANIFEST),
        help="Release artifact manifest used for skip or target selection.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=300_000,
        help="Docs mode only: only compress PNGs at or above this size.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Maximum preview width. Defaults to 1800 for docs and 2200 for release.",
    )
    parser.add_argument("--colors", type=int, default=192, help="Palette colors.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report without modifying files."
    )
    parser.add_argument(
        "--include-release-manifest-paths",
        action="store_true",
        help="Docs mode only: also compress files pinned by the release manifest.",
    )
    return parser


def _print_reports(reports: list[dict[str, Any]], *, release: bool) -> None:
    total_saved = 0
    for report in reports:
        if report.get("skipped"):
            print(f"skip {report['path']}: {report['reason']}")
            continue
        total_saved += int(report["saved_bytes"])
        label = (
            report.get("manifest_path", report["path"]) if release else report["path"]
        )
        print(
            "{path}: {before} -> {after} bytes, saved {saved}".format(
                path=label,
                before=report["before_size_bytes"],
                after=report["after_size_bytes"],
                saved=report["saved_bytes"],
            )
        )
    print(f"total_saved={total_saved}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode: Mode = args.mode
    if mode == "release":
        reports = compress_release_previews(
            manifest=args.manifest,
            max_width=2200 if args.max_width is None else int(args.max_width),
            colors=int(args.colors),
            dry_run=bool(args.dry_run),
        )
        _print_reports(reports, release=True)
        return 0
    reports = compress_docs_previews(
        static_dir=args.static_dir,
        manifest=args.manifest,
        min_bytes=int(args.min_bytes),
        max_width=1800 if args.max_width is None else int(args.max_width),
        colors=int(args.colors),
        dry_run=bool(args.dry_run),
        include_manifest_paths=bool(args.include_release_manifest_paths),
    )
    _print_reports(reports, release=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
