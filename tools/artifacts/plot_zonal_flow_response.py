#!/usr/bin/env python3
"""Render zonal-flow response panels from CSV data or saved diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spectraxgk.artifacts.plotting import zonal_flow_response_figure
from spectraxgk.benchmarks import load_diagnostic_time_series, zonal_flow_response_metrics

ROOT = Path(__file__).resolve().parents[2]


def _add_metric_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--fit-window-tmin", type=float, default=None)
    parser.add_argument("--fit-window-tmax", type=float, default=None)
    parser.add_argument(
        "--hilbert-trim-fraction",
        type=float,
        default=0.2,
        help="Fraction trimmed from both ends of a Hilbert-phase fit window.",
    )


def _metric_payload(metrics) -> dict[str, object]:
    return {
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
    }


def _metrics_from_args(args: argparse.Namespace, t: np.ndarray, response: np.ndarray):
    return zonal_flow_response_metrics(
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


def _save_panel(
    *,
    t: np.ndarray,
    response: np.ndarray,
    out: Path,
    title: str,
    metrics,
) -> None:
    fig, _axes = zonal_flow_response_figure(t, response, metrics=metrics, title=title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_csv(args: argparse.Namespace) -> int:
    data = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=float)
    names = set(data.dtype.names or ())
    if {"t", "response"} - names:
        raise ValueError("CSV must contain columns t,response")

    t = np.asarray(data["t"], dtype=float)
    response = np.asarray(data["response"], dtype=float)
    metrics = _metrics_from_args(args, t, response)
    _save_panel(t=t, response=response, out=args.out, title=args.title, metrics=metrics)
    _write_json(args.out.with_suffix(".json"), _metric_payload(metrics))
    return 0


def _run_output(args: argparse.Namespace) -> int:
    series = load_diagnostic_time_series(
        args.output,
        variable=args.var,
        kx_index=args.kx_index,
        component=args.component,
        align_phase=bool(args.align_phase),
    )
    if np.iscomplexobj(series.values):
        raise ValueError(
            "zonal-response plotting requires a real-valued extracted series; choose component real/imag/abs"
        )

    metrics = _metrics_from_args(args, series.t, series.values)
    title = args.title or f"{args.var} response"
    _save_panel(
        t=series.t,
        response=series.values,
        out=args.out,
        title=title,
        metrics=metrics,
    )

    csv_out = args.csv_out if args.csv_out is not None else args.out.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        csv_out,
        np.column_stack([series.t, series.values]),
        delimiter=",",
        header="t,response",
        comments="",
    )

    payload = {
        "source_path": series.source_path,
        "variable": series.variable,
        **_metric_payload(metrics),
        "notes": (
            "Phi2_zonal_t is a zonal-energy proxy. For manuscript-grade Rosenbluth-Hinton/GAM work, "
            "prefer Phi_zonal_mode_kxt with --kx-index and --align-phase."
        ),
    }
    _write_json(args.out.with_suffix(".json"), payload)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    csv_parser = subparsers.add_parser("csv", help="Plot a t,response CSV file.")
    csv_parser.add_argument("csv", type=Path, help="CSV with columns t,response")
    csv_parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "zonal_flow_response.png",
        help="Output figure path.",
    )
    csv_parser.add_argument("--title", default="Zonal-flow response")
    _add_metric_args(csv_parser)
    csv_parser.set_defaults(func=_run_csv)

    output_parser = subparsers.add_parser(
        "output", help="Extract a response series from a saved diagnostics file."
    )
    output_parser.add_argument(
        "output", type=Path, help="SPECTRAX-GK out.nc diagnostics file"
    )
    output_parser.add_argument(
        "--var",
        default="Phi2_zonal_t",
        help="Diagnostics variable to extract. Use Phi_zonal_mode_kxt for signed mode history.",
    )
    output_parser.add_argument(
        "--kx-index",
        type=int,
        default=None,
        help="Select kx index for 2D time-by-kx diagnostics.",
    )
    output_parser.add_argument(
        "--component",
        choices=("real", "imag", "abs", "complex"),
        default="real",
        help="Component to extract from complex diagnostics.",
    )
    output_parser.add_argument(
        "--align-phase",
        action="store_true",
        help="Rotate complex diagnostics so the first nonzero sample is real and positive.",
    )
    output_parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "zonal_flow_response_from_output.png",
        help="Output figure path.",
    )
    output_parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV path for the extracted time series. Defaults next to --out.",
    )
    output_parser.add_argument("--title", default=None)
    _add_metric_args(output_parser)
    output_parser.set_defaults(func=_run_output)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
