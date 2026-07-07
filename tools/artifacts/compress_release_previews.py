#!/usr/bin/env python3
"""Compress release-manifest PNG previews in place for lightweight docs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tomllib

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "tools/release_artifact_manifest.toml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preview_targets(manifest: Path) -> list[Path]:
    with manifest.open("rb") as fh:
        data = tomllib.load(fh)
    targets = []
    for item in data.get("artifacts", []):
        path = Path(str(item.get("path", "")))
        action = str(item.get("action", ""))
        preview_strategy = str(item.get("preview_strategy", ""))
        if path.suffix.lower() == ".png" and action in {
            "move_to_release",
            "keep_preview_in_repo",
        }:
            if (
                "preview" in preview_strategy.lower()
                or action == "keep_preview_in_repo"
            ):
                targets.append(path)
    return targets


def compress_png_preview(
    path: str | Path, *, max_width: int = 2200, colors: int = 192
) -> dict[str, object]:
    """Compress a PNG figure preview in place and return size/checksum metadata."""

    target = Path(path)
    before_size = target.stat().st_size
    before_sha = _sha256(target)
    image = Image.open(target).convert("RGBA")
    original_size = image.size
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    rgb = background.convert("RGB")
    palette = rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    palette.save(target, optimize=True)
    return {
        "path": str(target),
        "original_dimensions": original_size,
        "preview_dimensions": image.size,
        "before_size_bytes": before_size,
        "after_size_bytes": target.stat().st_size,
        "before_sha256": before_sha,
        "after_sha256": _sha256(target),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST), help="Release artifact manifest."
    )
    parser.add_argument(
        "--max-width", type=int, default=2200, help="Maximum preview width in pixels."
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=192,
        help="Palette colors for compressed previews.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    for rel in _preview_targets(manifest):
        result = compress_png_preview(
            ROOT / rel, max_width=args.max_width, colors=args.colors
        )
        print(
            "{path}: {before} -> {after} bytes".format(
                path=rel,
                before=result["before_size_bytes"],
                after=result["after_size_bytes"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
