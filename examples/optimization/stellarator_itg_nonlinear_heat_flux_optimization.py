#!/usr/bin/env python
"""Optimize a QA max-mode-1 stellarator for a reduced nonlinear ITG window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stellarator_itg_plotting import write_result_artifacts  # noqa: E402
from spectraxgk import optimize_stellarator_itg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "stellarator_itg_nonlinear_optimization",
        help="Output base path without extension.",
    )
    parser.add_argument(
        "--finite-difference-workers",
        type=int,
        default=1,
        help="Thread workers for finite-difference gradient-gate columns.",
    )
    args = parser.parse_args()

    result = optimize_stellarator_itg(
        "nonlinear_heat_flux",
        finite_difference_workers=args.finite_difference_workers,
    )
    write_result_artifacts(
        result,
        args.out,
        title="QA stellarator optimization for a reduced nonlinear ITG window",
    )
    trace = result.nonlinear_trace or {}
    final_window = trace.get("final_window", {})
    print(
        "reduced nonlinear-window optimization: "
        f"objective {result.initial_objective:.4e} -> {result.final_objective:.4e}, "
        f"AD/FD gate={result.gradient_gate['passed']}, "
        f"tail CV={final_window.get('cv', float('nan')):.3e}, "
        f"tail trend={final_window.get('trend', float('nan')):.3e}, "
        f"artifacts={args.out}"
    )


if __name__ == "__main__":
    main()
