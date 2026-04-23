#!/usr/bin/env python3
"""Plot a zonal-flow response trace and save reviewer-facing metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectraxgk.benchmarking import zonal_flow_response_metrics
from spectraxgk.plotting import zonal_flow_response_figure

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV with columns t,response")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "zonal_flow_response.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.3,
        help="Late-time fraction used for the residual window.",
    )
    parser.add_argument(
        "--initial-fraction",
        type=float,
        default=0.1,
        help="Leading fraction used to normalize the initial amplitude.",
    )
    parser.add_argument(
        "--initial-policy",
        choices=("window_abs_mean", "first_abs"),
        default="window_abs_mean",
        help="Initial normalization convention for the response metrics.",
    )
    parser.add_argument(
        "--title",
        default="Zonal-flow response",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=float)
    names = set(data.dtype.names or ())
    if {"t", "response"} - names:
        raise ValueError("CSV must contain columns t,response")

    t = np.asarray(data["t"], dtype=float)
    response = np.asarray(data["response"], dtype=float)
    metrics = zonal_flow_response_metrics(
        t,
        response,
        tail_fraction=float(args.tail_fraction),
        initial_fraction=float(args.initial_fraction),
        initial_policy=str(args.initial_policy),
    )
    fig, _axes = zonal_flow_response_figure(t, response, metrics=metrics, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    if args.out.suffix.lower() != ".pdf":
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")

    metrics_out = args.out.with_suffix(".json")
    metrics_out.write_text(
        json.dumps(
            {
                "initial_level": metrics.initial_level,
                "initial_policy": metrics.initial_policy,
                "residual_level": metrics.residual_level,
                "residual_std": metrics.residual_std,
                "response_rms": metrics.response_rms,
                "gam_frequency": metrics.gam_frequency,
                "gam_damping_rate": metrics.gam_damping_rate,
                "peak_count": metrics.peak_count,
                "tmin": metrics.tmin,
                "tmax": metrics.tmax,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
