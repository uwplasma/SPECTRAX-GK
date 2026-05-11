#!/usr/bin/env python
"""Compare growth, quasilinear-flux, and nonlinear-flux QA optimization outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stellarator_itg_plotting import write_comparison_artifacts  # noqa: E402
from spectraxgk import compare_stellarator_itg_objectives  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "stellarator_itg_optimization_comparison",
        help="Output base path without extension.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent objective workers; preserves serial result ordering.",
    )
    parser.add_argument(
        "--parallel-executor",
        choices=("thread", "process"),
        default="thread",
        help="Executor for independent objective workers.",
    )
    parser.add_argument(
        "--finite-difference-workers",
        type=int,
        default=1,
        help="Thread workers for finite-difference gradient-gate columns inside each objective.",
    )
    args = parser.parse_args()

    payload = compare_stellarator_itg_objectives(
        workers=args.workers,
        parallel_executor=args.parallel_executor,
        finite_difference_workers=args.finite_difference_workers,
    )
    write_comparison_artifacts(payload, args.out)
    print(f"comparison artifacts={args.out}")


if __name__ == "__main__":
    main()
