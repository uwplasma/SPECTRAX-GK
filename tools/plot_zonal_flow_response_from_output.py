#!/usr/bin/env python3
"""Extract a zonal-response series from ``out.nc`` and render the metrics panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectraxgk.benchmarking import load_diagnostic_time_series, zonal_flow_response_metrics
from spectraxgk.plotting import zonal_flow_response_figure

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="GX-style out.nc file")
    parser.add_argument(
        "--var",
        default="Phi2_zonal_t",
        help="Diagnostics variable to extract. Use Phi_zonal_mode_kxt for the signed zonal mode history.",
    )
    parser.add_argument("--kx-index", type=int, default=None, help="Select kx index for 2D time-by-kx diagnostics.")
    parser.add_argument(
        "--component",
        choices=("real", "imag", "abs", "complex"),
        default="real",
        help="Component to extract from complex diagnostics.",
    )
    parser.add_argument(
        "--align-phase",
        action="store_true",
        help="Rotate complex diagnostics so the first nonzero sample is real and positive.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "zonal_flow_response_from_output.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV path for the extracted time series. Defaults next to --out.",
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
        default=None,
        help="Optional figure title. Defaults to '<var> response'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    series = load_diagnostic_time_series(
        args.output,
        variable=args.var,
        kx_index=args.kx_index,
        component=args.component,
        align_phase=bool(args.align_phase),
    )
    if np.iscomplexobj(series.values):
        raise ValueError("zonal-response plotting requires a real-valued extracted series; choose component real/imag/abs")
    metrics = zonal_flow_response_metrics(
        series.t,
        series.values,
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

    title = args.title or f"{args.var} response"
    fig, _axes = zonal_flow_response_figure(series.t, series.values, metrics=metrics, title=title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    if args.out.suffix.lower() != ".pdf":
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")

    csv_out = args.csv_out if args.csv_out is not None else args.out.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        csv_out,
        np.column_stack([series.t, series.values]),
        delimiter=",",
        header="t,response",
        comments="",
    )

    meta_out = args.out.with_suffix(".json")
    meta_out.write_text(
        json.dumps(
            {
                "source_path": series.source_path,
                "variable": series.variable,
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
                "notes": (
                    "Phi2_zonal_t is a zonal-energy proxy. For manuscript-grade Rosenbluth-Hinton/GAM work, "
                    "prefer Phi_zonal_mode_kxt with --kx-index and --align-phase."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
