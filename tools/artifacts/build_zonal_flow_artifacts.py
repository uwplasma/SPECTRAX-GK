#!/usr/bin/env python3
"""Build zonal-response figures and optimization-row artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spectraxgk.artifacts.nonlinear_diagnostics import (  # noqa: E402
    load_diagnostic_time_series,
)
from spectraxgk.artifacts.plotting import (  # noqa: E402
    set_plot_style,
    zonal_flow_response_figure,
)
from spectraxgk.diagnostics.zonal_validation import (  # noqa: E402
    zonal_flow_response_metrics,
)
from spectraxgk.objectives.zonal import (  # noqa: E402
    ZonalFlowObjectiveConfig,
    zonal_flow_objective_artifact_from_records,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "docs" / "_static" / "w7x_zonal_response_panel.csv"
DEFAULT_COMPARISON = ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.csv"
DEFAULT_OUT_JSON = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.json"
DEFAULT_OUT_CSV = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.csv"
DEFAULT_OUT_PNG = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.png"


def _add_response_metric_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tail-fraction", type=float, default=0.3)
    parser.add_argument("--initial-fraction", type=float, default=0.1)
    parser.add_argument(
        "--initial-policy",
        choices=("window_abs_mean", "first_abs"),
        default="window_abs_mean",
    )
    parser.add_argument("--peak-fit-max-peaks", type=int, default=None)
    parser.add_argument(
        "--damping-fit-mode",
        choices=("combined_envelope", "branchwise_extrema"),
        default="combined_envelope",
    )
    parser.add_argument(
        "--frequency-fit-mode",
        choices=("peak_spacing", "hilbert_phase"),
        default="peak_spacing",
    )
    parser.add_argument("--fit-window-tmin", type=float, default=None)
    parser.add_argument("--fit-window-tmax", type=float, default=None)
    parser.add_argument("--hilbert-trim-fraction", type=float, default=0.2)


def _response_metrics(args: argparse.Namespace, t: np.ndarray, response: np.ndarray):
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


def _response_payload(metrics) -> dict[str, object]:
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


def _write_response_panel(
    *, t: np.ndarray, response: np.ndarray, out: Path, title: str, metrics
) -> None:
    fig, _axes = zonal_flow_response_figure(
        t, response, metrics=metrics, title=title
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_response_csv_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a t,response CSV artifact.")
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "zonal_flow_response.png",
    )
    parser.add_argument("--title", default="Zonal-flow response")
    _add_response_metric_args(parser)
    return parser


def _build_response_output_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a saved zonal diagnostic.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--var", default="Phi2_zonal_t")
    parser.add_argument("--kx-index", type=int, default=None)
    parser.add_argument(
        "--component", choices=("real", "imag", "abs", "complex"), default="real"
    )
    parser.add_argument("--align-phase", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "zonal_flow_response_from_output.png",
    )
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--title", default=None)
    _add_response_metric_args(parser)
    return parser


def _main_response_csv(argv: list[str]) -> int:
    args = _build_response_csv_parser().parse_args(argv)
    data = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=float)
    if {"t", "response"} - set(data.dtype.names or ()):
        raise ValueError("CSV must contain columns t,response")
    t = np.asarray(data["t"], dtype=float)
    response = np.asarray(data["response"], dtype=float)
    metrics = _response_metrics(args, t, response)
    _write_response_panel(
        t=t, response=response, out=args.out, title=args.title, metrics=metrics
    )
    _write_json(args.out.with_suffix(".json"), _response_payload(metrics))
    return 0


def _main_response_output(argv: list[str]) -> int:
    args = _build_response_output_parser().parse_args(argv)
    series = load_diagnostic_time_series(
        args.output,
        variable=args.var,
        kx_index=args.kx_index,
        component=args.component,
        align_phase=bool(args.align_phase),
    )
    if np.iscomplexobj(series.values):
        raise ValueError(
            "zonal-response plotting requires a real extracted component"
        )
    metrics = _response_metrics(args, series.t, series.values)
    _write_response_panel(
        t=series.t,
        response=series.values,
        out=args.out,
        title=args.title or f"{args.var} response",
        metrics=metrics,
    )
    csv_out = args.csv_out or args.out.with_suffix(".csv")
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
        **_response_payload(metrics),
        "notes": (
            "Phi2_zonal_t is a zonal-energy proxy. Prefer Phi_zonal_mode_kxt "
            "with a selected kx and phase alignment for signed response studies."
        ),
    }
    _write_json(args.out.with_suffix(".json"), payload)
    return 0


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "nan", "none", "null"}:
            return None
        value = stripped
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    return scalar


def _kx_key(value: Any) -> float:
    scalar = _optional_float(value)
    if scalar is None:
        raise ValueError(f"missing finite kx value: {value!r}")
    return round(float(scalar), 10)


def _comparison_by_kx(path: Path | None) -> dict[float, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    table = _read_csv(path)
    rows: dict[float, dict[str, str]] = {}
    for row in table:
        if "kx" in row:
            rows[_kx_key(row["kx"])] = row
        elif "kx_target" in row:
            rows[_kx_key(row["kx_target"])] = row
    return rows


def _tail_std_ratio(row: dict[str, str] | None) -> float | None:
    if row is None:
        return None
    direct = _optional_float(row.get("tail_std_ratio"))
    if direct is not None:
        return direct
    tail_std = _optional_float(row.get("tail_std"))
    reference_tail_std = _optional_float(row.get("reference_tail_std"))
    if tail_std is None or reference_tail_std is None or reference_tail_std <= 0.0:
        return None
    return tail_std / reference_tail_std


def _recurrence_value(
    *,
    summary_row: dict[str, str],
    comparison_row: dict[str, str] | None,
    source: str,
) -> float | None:
    if source == "residual_std":
        return _optional_float(summary_row.get("residual_std"))
    if source == "tail_std":
        return (
            None
            if comparison_row is None
            else _optional_float(comparison_row.get("tail_std"))
        )
    if source == "tail_std_ratio":
        return _tail_std_ratio(comparison_row)
    if source != "auto":
        raise ValueError(f"unknown recurrence source: {source}")
    ratio = _tail_std_ratio(comparison_row)
    if ratio is not None:
        return ratio
    return _optional_float(summary_row.get("residual_std"))


def records_from_w7x_summary(
    summary_csv: Path,
    *,
    comparison_csv: Path | None = None,
    recurrence_source: str = "auto",
) -> list[dict[str, object]]:
    """Return normalized zonal-objective records from the W7-X summary CSV."""

    summary = _read_csv(summary_csv)
    comparison = _comparison_by_kx(comparison_csv)
    records: list[dict[str, object]] = []
    for row in summary:
        kx = _kx_key(row.get("kx_target", row.get("kx")))
        comparison_row = comparison.get(kx)
        recurrence = _recurrence_value(
            summary_row=row,
            comparison_row=comparison_row,
            source=recurrence_source,
        )
        records.append(
            {
                "surface": _optional_float(row.get("surface")) or 0.0,
                "alpha": _optional_float(row.get("alpha")) or 0.0,
                "kx": float(kx),
                "residual_level": row.get("residual_level"),
                "damping_rate": row.get("gam_damping_rate", row.get("damping_rate")),
                "linear_growth_rate": row.get("linear_growth_rate", 0.0),
                "recurrence_amplitude": recurrence,
            }
        )
    return records


def _write_row_csv(path: Path, payload: dict[str, object]) -> None:
    rows = list(payload["row_table"])
    if not rows:
        raise ValueError("cannot write an empty zonal-flow objective table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "surface",
        "alpha",
        "kx",
        "residual_level",
        "damping_rate",
        "linear_growth_rate",
        "recurrence_amplitude",
        "inverse_residual",
        "growth_over_residual",
        "sample_objective",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def _plot_payload(path: Path, payload: dict[str, object]) -> None:
    set_plot_style()
    table = list(payload["row_table"])
    kx = np.asarray([float(row["kx"]) for row in table], dtype=float)
    order = np.argsort(kx)
    labels = [f"{kx[index]:.2f}" for index in order]
    x = np.arange(order.size)
    metrics = {
        "Residual response\n(higher is better)": [
            float(table[index]["residual_level"]) for index in order
        ],
        "Damping penalty\n(lower is better)": [
            float(table[index]["damping_rate"]) for index in order
        ],
        "Recurrence/tail penalty\n(lower is better)": [
            float(table[index]["recurrence_amplitude"]) for index in order
        ],
        "Weighted sample objective\n(lower is better)": [
            float(table[index]["sample_objective"]) for index in order
        ],
    }
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for ax, (title, values), color in zip(
        axes.ravel(), metrics.items(), colors, strict=True
    ):
        ax.bar(x, values, color=color, alpha=0.86, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_xlabel(r"$k_x \rho_i$")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Zonal-flow objective row gate", fontsize=15)
    status = "promotion-ready" if payload["promotion_ready"] else "diagnostic only"
    fig.text(
        0.5,
        0.01,
        (
            f"Status: {status}; missing damping rows: {payload['missing_damping_count']}; "
            f"claim: {payload['claim_level']}"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _parse_objective_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument(
        "--recurrence-source",
        choices=("auto", "residual_std", "tail_std", "tail_std_ratio"),
        default="auto",
        help="Late-envelope recurrence metric used for the fourth objective column.",
    )
    parser.add_argument(
        "--missing-damping-policy",
        choices=("zero", "fail"),
        default="zero",
        help=(
            "Use 'fail' for promoted physics gates. The default 'zero' writes a "
            "diagnostic W7-X row artifact while preserving promotion_ready=false."
        ),
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--damping-weight", type=float, default=1.0)
    parser.add_argument("--growth-over-residual-weight", type=float, default=0.0)
    parser.add_argument("--recurrence-weight", type=float, default=0.25)
    parser.add_argument("--residual-floor", type=float, default=1.0e-6)
    parser.add_argument(
        "--claim-level",
        default="diagnostic_zonal_objective_row_producer_not_promoted_w7x_optimization_claim",
    )
    return parser.parse_args(argv)


def _main_objective_gate(argv: list[str]) -> int:
    args = _parse_objective_args(argv)
    records = records_from_w7x_summary(
        args.summary_csv,
        comparison_csv=args.comparison_csv,
        recurrence_source=args.recurrence_source,
    )
    config = ZonalFlowObjectiveConfig(
        residual_weight=args.residual_weight,
        damping_weight=args.damping_weight,
        growth_over_residual_weight=args.growth_over_residual_weight,
        recurrence_weight=args.recurrence_weight,
        residual_floor=args.residual_floor,
    )
    payload = zonal_flow_objective_artifact_from_records(
        records,
        config=config,
        missing_damping_policy=args.missing_damping_policy,
        claim_level=args.claim_level,
        source_paths=[
            _repo_relative(args.summary_csv),
            _repo_relative(args.comparison_csv),
        ],
    )
    payload["input_summary_csv"] = _repo_relative(args.summary_csv)
    payload["input_comparison_csv"] = _repo_relative(args.comparison_csv)
    payload["recurrence_source"] = args.recurrence_source
    payload["validation_status"] = (
        "closed" if payload["promotion_ready"] else "diagnostic"
    )
    payload["gate_index_include"] = False
    payload["notes"] = [
        "This artifact verifies the row-production contract for zonal-flow optimization objectives.",
        "W7-X rows with missing GAM damping remain diagnostic and are not promoted to an optimization claim.",
        "Use --missing-damping-policy=fail for closed QA/QH/Miller-style promotion gates.",
    ]
    json.dumps(payload, allow_nan=False)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_row_csv(args.out_csv, payload)
    _plot_payload(args.out_png, payload)
    print(
        "wrote zonal-flow objective gate "
        f"samples={payload['sample_count']} promotion_ready={payload['promotion_ready']} "
        f"json={_repo_relative(args.out_json)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        print(
            "usage: build_zonal_flow_artifacts.py "
            "{response-csv,response-output,objective-gate} ..."
        )
        return 2
    command, rest = tokens[0], tokens[1:]
    if command == "response-csv":
        return _main_response_csv(rest)
    if command == "response-output":
        return _main_response_output(rest)
    if command == "objective-gate":
        return _main_objective_gate(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
