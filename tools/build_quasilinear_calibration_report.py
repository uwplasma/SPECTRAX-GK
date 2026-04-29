#!/usr/bin/env python3
"""Build a quasilinear calibration/holdout report from JSON points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from spectraxgk.quasilinear_calibration import (
    quasilinear_calibration_report,
    write_quasilinear_calibration_report,
)


def _load_points(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "points" in data:
        data = data["points"]
    if not isinstance(data, list):
        raise TypeError("input JSON must be a list of calibration points or an object with a 'points' list")
    return [dict(item) for item in data]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, help="JSON list of calibration points")
    parser.add_argument("--out", required=True, help="Output calibration report JSON")
    parser.add_argument("--saturation-rule", default="mixing_length")
    parser.add_argument("--holdout-mean-rel-gate", type=float, default=0.35)
    parser.add_argument("--observed-floor", type=float, default=1.0e-12)
    parser.add_argument("--version", default="0.1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    points = _load_points(Path(args.points))
    report = quasilinear_calibration_report(
        points,
        saturation_rule=args.saturation_rule,
        version=args.version,
        holdout_mean_rel_gate=args.holdout_mean_rel_gate,
        observed_floor=args.observed_floor,
        metadata={"source_points": str(Path(args.points))},
    )
    out = write_quasilinear_calibration_report(args.out, report)
    print(f"saved {out}")
    print(f"claim_level={report['claim_level']} passed={report['passed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
