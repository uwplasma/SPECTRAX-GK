#!/usr/bin/env python3
"""Check nonlinear late-window convergence metadata without rerunning a solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from spectraxgk.quasilinear_window import (
    NonlinearWindowConvergenceConfig,
    nonlinear_window_convergence_from_csv,
    nonlinear_window_convergence_from_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Diagnostics CSV with time trace.")
    source.add_argument(
        "--summary",
        type=Path,
        help="Window summary JSON containing a diagnostics source path.",
    )
    parser.add_argument("--diagnostics-source", default="spectrax")
    parser.add_argument("--time-column", default="t")
    parser.add_argument("--value-column", default="heat_flux")
    parser.add_argument("--case", default=None)
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--transient-fraction", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--max-running-mean-rel-drift", type=float, default=0.15)
    parser.add_argument("--max-sem-rel", type=float, default=0.25)
    parser.add_argument("--value-floor", type=float, default=1.0e-12)
    parser.add_argument(
        "--allow-nonfinite",
        action="store_true",
        help="Ignore non-finite samples inside the late window instead of failing.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def _config(args: argparse.Namespace) -> NonlinearWindowConvergenceConfig:
    return NonlinearWindowConvergenceConfig(
        tmin=args.tmin,
        tmax=args.tmax,
        transient_fraction=args.transient_fraction,
        min_samples=args.min_samples,
        min_blocks=args.min_blocks,
        block_size=args.block_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        max_running_mean_rel_drift=args.max_running_mean_rel_drift,
        max_sem_rel=args.max_sem_rel,
        value_floor=args.value_floor,
        require_all_finite=not args.allow_nonfinite,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config(args)
    if args.csv is not None:
        report = nonlinear_window_convergence_from_csv(
            args.csv,
            time_column=args.time_column,
            value_column=args.value_column,
            case=args.case,
            config=cfg,
        )
    else:
        report = nonlinear_window_convergence_from_summary(
            args.summary,
            diagnostics_source=args.diagnostics_source,
            time_column=args.time_column,
            value_column=args.value_column,
            case=args.case,
            config=cfg,
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    stats = report["statistics"]
    print(f"wrote {args.out_json}")
    print(
        "nonlinear_window_converged={passed} late_mean={mean} sem={sem}".format(
            passed=report["passed"],
            mean=stats["late_mean"],
            sem=stats["sem"],
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
