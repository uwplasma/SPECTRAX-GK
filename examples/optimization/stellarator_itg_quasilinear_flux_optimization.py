#!/usr/bin/env python
"""Optimize a QA max-mode-1 stellarator for reduced quasilinear ITG heat flux."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _stellarator_itg_plotting import write_portfolio_gate_artifacts, write_result_artifacts  # noqa: E402
from spectraxgk import (  # noqa: E402
    StellaratorITGOptimizationConfig,
    StellaratorITGSampleSet,
    optimize_stellarator_itg,
    stellarator_itg_portfolio_gate_payload,
)


PORTFOLIO_OBJECTIVES = ("growth", "quasilinear_flux")


def _float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _portfolio_out_path(out_base: Path) -> Path:
    return out_base.with_name(f"{out_base.name}_portfolio_gate")


def _sample_set_from_args(args: argparse.Namespace) -> StellaratorITGSampleSet:
    defaults = StellaratorITGSampleSet()
    return StellaratorITGSampleSet(
        surfaces=defaults.surfaces if args.surfaces is None else args.surfaces,
        alphas=defaults.alphas if args.alphas is None else args.alphas,
        ky_values=defaults.ky_values if args.ky_values is None else args.ky_values,
    )


def _write_portfolio_artifacts(args: argparse.Namespace, result: object) -> Path:
    sample_set = _sample_set_from_args(args)
    params = np.asarray(result.final_params, dtype=float)
    cfg = StellaratorITGOptimizationConfig(**result.config)
    payload = stellarator_itg_portfolio_gate_payload(
        params,
        PORTFOLIO_OBJECTIVES,
        cfg,
        sample_set,
        objective_weights=args.objective_weights,
        finite_difference_workers=args.finite_difference_workers,
    )
    payload["optimization_objective_kind"] = result.objective_kind
    payload["optimized_params"] = params.tolist()
    payload["optimization_initial_params"] = [
        float(value) for value in np.asarray(result.initial_params, dtype=float)
    ]
    out = _portfolio_out_path(args.out)
    write_portfolio_gate_artifacts(payload, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "stellarator_itg_quasilinear_optimization",
        help="Output base path without extension.",
    )
    parser.add_argument(
        "--finite-difference-workers",
        type=int,
        default=1,
        help="Thread workers for finite-difference gradient-gate columns.",
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Also write a reduced multi-surface/alpha/ky growth+QL portfolio gate at the optimized point.",
    )
    parser.add_argument(
        "--surfaces",
        type=_float_tuple,
        default=None,
        help="Comma-separated normalized flux surfaces for --portfolio.",
    )
    parser.add_argument(
        "--alphas",
        type=_float_tuple,
        default=None,
        help="Comma-separated field-line alpha values for --portfolio.",
    )
    parser.add_argument(
        "--ky-values",
        type=_float_tuple,
        default=None,
        help="Comma-separated ky*rho_i values for --portfolio.",
    )
    parser.add_argument(
        "--objective-weights",
        type=_float_tuple,
        default=None,
        help="Optional comma-separated weights for portfolio objectives: growth,quasilinear_flux.",
    )
    args = parser.parse_args()
    if args.objective_weights is not None and len(args.objective_weights) != len(PORTFOLIO_OBJECTIVES):
        parser.error("--objective-weights must provide two values: growth,quasilinear_flux")

    result = optimize_stellarator_itg("quasilinear_flux", finite_difference_workers=args.finite_difference_workers)
    write_result_artifacts(
        result,
        args.out,
        title="QA stellarator optimization for small quasilinear ITG heat flux",
    )
    portfolio_out = _write_portfolio_artifacts(args, result) if args.portfolio else None
    print(
        "quasilinear-flux optimization: "
        f"objective {result.initial_objective:.4e} -> {result.final_objective:.4e}, "
        f"AD/FD gate={result.gradient_gate['passed']}, artifacts={args.out}"
        + ("" if portfolio_out is None else f", portfolio_artifacts={portfolio_out}")
    )


if __name__ == "__main__":
    main()
