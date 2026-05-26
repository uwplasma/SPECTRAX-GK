#!/usr/bin/env python3
"""Compress tracked PNG figure artifacts with bounded visual error.

The default mode is intentionally conservative for documentation plots:
it converts RGBA/RGB PNGs to an indexed 256-colour PNG only when the
candidate is smaller and the pixel-space RMSE/max-channel-difference gates
pass. The script prints a CSV report so release-size changes are auditable.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


@dataclass
class Result:
    path: str
    before: int
    after: int
    saved: int
    saved_percent: float
    rmse: float
    max_channel_diff: int
    action: str


def _tracked_pngs() -> list[Path]:
    files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    return [Path(path) for path in files if path.lower().endswith(".png")]


def _candidate_png(data: bytes, colors: int) -> tuple[bytes, float, int]:
    image = Image.open(BytesIO(data)).convert("RGBA")
    quantized = image.quantize(
        colors=colors,
        method=Image.Quantize.FASTOCTREE,
        dither=Image.Dither.NONE,
    )
    restored = quantized.convert("RGBA")
    diff = ImageChops.difference(image, restored)
    stat = ImageStat.Stat(diff)
    rmse = math.sqrt(sum(value * value for value in stat.rms) / len(stat.rms))
    max_channel_diff = max(channel[1] for channel in diff.getextrema())

    out = BytesIO()
    quantized.save(out, format="PNG", optimize=True, compress_level=9)
    return out.getvalue(), rmse, int(max_channel_diff)


def compress_one(
    path: Path,
    *,
    colors: int,
    max_rmse: float,
    max_channel_diff: int,
    min_saving_percent: float,
    dry_run: bool,
) -> Result:
    before_data = path.read_bytes()
    before = len(before_data)
    try:
        after_data, rmse, channel_diff = _candidate_png(before_data, colors)
    except Exception as exc:  # pragma: no cover - defensive release utility path
        return Result(str(path), before, before, 0, 0.0, math.nan, -1, f"error:{exc}")

    after = len(after_data)
    saved = before - after
    saved_percent = 100.0 * saved / before if before else 0.0
    allowed = (
        saved > 0
        and saved_percent >= min_saving_percent
        and rmse <= max_rmse
        and channel_diff <= max_channel_diff
    )
    if allowed:
        if not dry_run:
            path.write_bytes(after_data)
        action = "compressed" if not dry_run else "would_compress"
    else:
        after = before
        saved = 0
        saved_percent = 0.0
        action = "skipped"
    return Result(str(path), before, after, saved, saved_percent, rmse, channel_diff, action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="PNG files to compress; defaults to tracked PNGs")
    parser.add_argument("--colors", type=int, default=256, help="Indexed PNG palette size")
    parser.add_argument("--max-rmse", type=float, default=3.0, help="Maximum RGBA pixel RMSE")
    parser.add_argument("--max-channel-diff", type=int, default=80, help="Maximum per-channel pixel difference")
    parser.add_argument("--min-saving-percent", type=float, default=2.0, help="Minimum file-size saving to rewrite")
    parser.add_argument("--dry-run", action="store_true", help="Report without modifying files")
    parser.add_argument("--report", type=Path, default=None, help="Optional CSV report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.paths or _tracked_pngs()
    results = [
        compress_one(
            path,
            colors=args.colors,
            max_rmse=args.max_rmse,
            max_channel_diff=args.max_channel_diff,
            min_saving_percent=args.min_saving_percent,
            dry_run=args.dry_run,
        )
        for path in paths
    ]

    total_before = sum(result.before for result in results)
    total_after = sum(result.after for result in results)
    changed = sum(1 for result in results if result.action in {"compressed", "would_compress"})
    print(
        f"PNG compression: {changed}/{len(results)} files, "
        f"{total_before / 1024 / 1024:.2f} -> {total_after / 1024 / 1024:.2f} MiB, "
        f"saved {(total_before - total_after) / 1024 / 1024:.2f} MiB"
    )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(Result.__annotations__))
            writer.writeheader()
            for result in results:
                writer.writerow(result.__dict__)
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
