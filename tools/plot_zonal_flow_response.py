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
        "--peak-fit-max-peaks",
        type=int,
        default=None,
        help="Optional maximum number of early envelope peaks to use for the GAM damping fit.",
    )
    parser.add_argument(
        "--damping-fit-mode",
        choices=("combined_envelope", "branchwise_extrema"),
        default="combined_envelope",
        help="Damping-fit convention used for the GAM envelope.",
    )
    parser.add_argument(
        "--frequency-fit-mode",
        choices=("peak_spacing", "hilbert_phase"),
        default="peak_spacing",
        help="Frequency-fit convention used for the GAM oscillation.",
    )
    parser.add_argument("--fit-window-tmin", type=float, default=None, help="Optional lower fit-window bound.")
    parser.add_argument("--fit-window-tmax", type=float, default=None, help="Optional upper fit-window bound.")
    parser.add_argument(
        "--hilbert-trim-fraction",
        type=float,
        default=0.2,
        help="Fraction trimmed from both ends of a Hilbert-phase fit window.",
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
        peak_fit_max_peaks=args.peak_fit_max_peaks,
        damping_fit_mode=str(args.damping_fit_mode),
        frequency_fit_mode=str(args.frequency_fit_mode),
        fit_window_tmin=args.fit_window_tmin,
        fit_window_tmax=args.fit_window_tmax,
        hilbert_trim_fraction=float(args.hilbert_trim_fraction),
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
                "damping_method": metrics.damping_method,
                "frequency_method": metrics.frequency_method,
                "peak_count": metrics.peak_count,
                "peak_fit_count": metrics.peak_fit_count,
                "tmin": metrics.tmin,
                "tmax": metrics.tmax,
                "fit_tmin": metrics.fit_tmin,
                "fit_tmax": metrics.fit_tmax,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
