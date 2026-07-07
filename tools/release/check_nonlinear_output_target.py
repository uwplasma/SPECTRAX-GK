#!/usr/bin/env python3
"""Check whether one nonlinear NetCDF output bundle reached a target time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    from .check_matched_nonlinear_transport_matrix_progress import (
        _bundle_paths,
        _read_output_tmax,
        _repo_relative,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_matched_nonlinear_transport_matrix_progress import (
        _bundle_paths,
        _read_output_tmax,
        _repo_relative,
    )


def build_report(
    *,
    output: Path,
    target_time: float,
    time_tolerance: float,
) -> dict[str, object]:
    bundle = _bundle_paths(output)
    present = {key: path.exists() for key, path in bundle.items()}
    bundle_complete = all(present.values())
    output_tmax = _read_output_tmax(bundle["out"])
    target_confirmed = bool(
        bundle_complete
        and output_tmax is not None
        and float(output_tmax) >= float(target_time) - float(time_tolerance)
    )
    return {
        "kind": "nonlinear_output_target_time_check",
        "output": _repo_relative(output),
        "bundle": {key: _repo_relative(path) for key, path in bundle.items()},
        "present": present,
        "bundle_complete": bundle_complete,
        "output_tmax": output_tmax,
        "target_time": float(target_time),
        "time_tolerance": float(time_tolerance),
        "target_time_confirmed": target_confirmed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-time", required=True, type=float)
    parser.add_argument("--time-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        output=args.output,
        target_time=float(args.target_time),
        time_tolerance=float(args.time_tolerance),
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["target_time_confirmed"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
