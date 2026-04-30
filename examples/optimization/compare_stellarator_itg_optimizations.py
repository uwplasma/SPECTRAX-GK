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
    args = parser.parse_args()

    payload = compare_stellarator_itg_objectives()
    write_comparison_artifacts(payload, args.out)
    print(f"comparison artifacts={args.out}")


if __name__ == "__main__":
    main()
