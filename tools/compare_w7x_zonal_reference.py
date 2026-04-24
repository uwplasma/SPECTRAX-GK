#!/usr/bin/env python3
"""Compare W7-X zonal-response traces against digitized stella/GENE references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectraxgk.benchmarking import evaluate_scalar_gate, gate_report, gate_report_to_dict
from spectraxgk.plotting import set_plot_style
from spectraxgk.zonal_validation import (
    kx_token,
    load_w7x_combined_trace_csv,
    load_w7x_trace_csv,
    normalize_trace,
    reference_mean_trace,
    reference_residual_table,
    reference_time_limits,
    tail_trace_metrics,
    w7x_trace_path,
)


ROOT = Path(__file__).resolve().parents[1]
KX_VALUES = (0.05, 0.07, 0.10, 0.30)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectrax-summary",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_response_panel.csv",
        help="SPECTRAX-GK zonal summary CSV written by generate_w7x_zonal_response_panel.py.",
    )
    parser.add_argument(
        "--spectrax-trace-dir",
        type=Path,
        default=None,
        help="Optional directory containing per-kx w7x_test4_kxNNN.csv trace files.",
    )
    parser.add_argument(
        "--spectrax-traces",
        type=Path,
        default=None,
        help=(
            "Optional combined trace CSV written next to the W7-X response panel. "
            "This is mutually exclusive with --spectrax-trace-dir."
        ),
    )
    parser.add_argument(
        "--reference-traces",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.csv",
        help="Digitized stella/GENE main trace CSV.",
    )
    parser.add_argument(
        "--reference-residuals",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized_residuals.csv",
        help="Digitized stella/GENE inset residual CSV.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.csv",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.json",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.png",
    )
    parser.add_argument("--residual-atol", type=float, default=0.02)
    parser.add_argument("--residual-rtol", type=float, default=0.10)
    parser.add_argument("--coverage-fraction", type=float, default=0.98)
    parser.add_argument("--tail-fraction", type=float, default=0.10)
    parser.add_argument("--envelope-atol", type=float, default=0.03)
    parser.add_argument(
        "--trace-normalization",
        choices=("summary_initial_level", "first_nonzero"),
        default="summary_initial_level",
        help=(
            "Normalization for optional trace-shape gates. summary_initial_level "
            "keeps the envelope metric consistent with the residual normalization "
            "recorded by generate_w7x_zonal_response_panel.py."
        ),
    )
    parser.add_argument("--gate-index-include", action="store_true")
    return parser.parse_args(argv)


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_spectrax_summary(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"kx_target", "residual_level", "residual_std", "tmax"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return table


def _optional_trace_metrics(
    *,
    trace_dir: Path | None,
    combined_trace_csv: Path | None,
    reference_traces: pd.DataFrame,
    kx: float,
    tail_fraction: float,
    initial_level: float | None,
) -> dict[str, float | int | None]:
    if trace_dir is None and combined_trace_csv is None:
        return {
            "trace_available": 0,
            "tail_std": None,
            "reference_tail_std": None,
            "tail_mean_abs_error": None,
            "tail_max_abs_error": None,
        }
    if combined_trace_csv is not None:
        if not combined_trace_csv.exists():
            return {
                "trace_available": 0,
                "tail_std": None,
                "reference_tail_std": None,
                "tail_mean_abs_error": None,
                "tail_max_abs_error": None,
            }
        t_raw, y_raw = load_w7x_combined_trace_csv(combined_trace_csv, kx)
    else:
        if trace_dir is None:
            raise ValueError("trace_dir is unexpectedly None")
        path = w7x_trace_path(trace_dir, kx)
        if not path.exists():
            return {
                "trace_available": 0,
                "tail_std": None,
                "reference_tail_std": None,
                "tail_mean_abs_error": None,
                "tail_max_abs_error": None,
            }
        t_raw, y_raw = load_w7x_trace_csv(path)
    t_obs, y_obs = normalize_trace(t_raw, y_raw, initial_level=initial_level)
    ref_t, ref_y = reference_mean_trace(reference_traces, kx)
    return {
        "trace_available": 1,
        **tail_trace_metrics(
            t_obs=t_obs,
            y_obs=y_obs,
            t_ref=ref_t,
            y_ref=ref_y,
            tail_fraction=float(tail_fraction),
        ),
    }


def build_comparison(
    *,
    spectrax_summary: Path,
    reference_traces: Path,
    reference_residuals: Path,
    spectrax_trace_dir: Path | None = None,
    spectrax_traces: Path | None = None,
    residual_atol: float = 0.02,
    residual_rtol: float = 0.10,
    coverage_fraction: float = 0.98,
    tail_fraction: float = 0.10,
    envelope_atol: float = 0.03,
    trace_normalization: str = "summary_initial_level",
):
    if spectrax_trace_dir is not None and spectrax_traces is not None:
        raise ValueError("spectrax_trace_dir and spectrax_traces are mutually exclusive")
    summary = _load_spectrax_summary(spectrax_summary)
    ref_traces = pd.read_csv(reference_traces)
    ref_residuals = reference_residual_table(reference_residuals)
    ref_limits = reference_time_limits(ref_traces)
    ref = pd.merge(ref_residuals, ref_limits, on="kx", how="inner")
    rows: list[dict[str, object]] = []
    gates = []
    for kx in KX_VALUES:
        obs_matches = summary[np.isclose(summary["kx_target"], float(kx))]
        ref_matches = ref[np.isclose(ref["kx"], float(kx))]
        if obs_matches.empty:
            raise ValueError(f"missing SPECTRAX summary row for kx={kx}")
        if ref_matches.empty:
            raise ValueError(f"missing reference row for kx={kx}")
        obs = obs_matches.iloc[0]
        ref_row = ref_matches.iloc[0]
        residual_ref = float(ref_row["reference_residual"])
        residual_atol_eff = float(residual_atol) + float(ref_row["reference_code_spread"])
        residual_gate = evaluate_scalar_gate(
            f"residual_kx{kx_token(kx)}",
            float(obs["residual_level"]),
            residual_ref,
            atol=residual_atol_eff,
            rtol=float(residual_rtol),
            notes=(
                "Residual compared with the mean of digitized stella/GENE inset values; "
                "absolute tolerance includes the inter-code spread."
            ),
        )
        coverage_ratio = min(float(obs["tmax"]) / float(ref_row["reference_tmax"]), 1.0)
        coverage_gate = evaluate_scalar_gate(
            f"time_coverage_kx{kx_token(kx)}",
            coverage_ratio,
            1.0,
            atol=1.0 - float(coverage_fraction),
            rtol=0.0,
            notes=f"Passes when SPECTRAX reaches at least {coverage_fraction:.0%} of the digitized reference window.",
        )
        gates.extend([coverage_gate, residual_gate])
        trace_initial_level = None
        trace_source_provided = spectrax_trace_dir is not None or spectrax_traces is not None
        if str(trace_normalization).strip().lower().replace("-", "_") == "summary_initial_level":
            if trace_source_provided and "initial_level" not in obs.index:
                raise ValueError(
                    "summary_initial_level trace normalization requires an initial_level column "
                    "in the SPECTRAX summary CSV"
                )
            trace_initial_level = float(obs["initial_level"]) if trace_source_provided else None
        elif str(trace_normalization).strip().lower().replace("-", "_") != "first_nonzero":
            raise ValueError("trace_normalization must be one of {'summary_initial_level', 'first_nonzero'}")
        trace_metrics = _optional_trace_metrics(
            trace_dir=spectrax_trace_dir,
            combined_trace_csv=spectrax_traces,
            reference_traces=ref_traces,
            kx=kx,
            tail_fraction=float(tail_fraction),
            initial_level=trace_initial_level,
        )
        if trace_metrics["tail_std"] is not None and trace_metrics["reference_tail_std"] is not None:
            gates.append(
                evaluate_scalar_gate(
                    f"tail_envelope_std_kx{kx_token(kx)}",
                    float(trace_metrics["tail_std"]),
                    float(trace_metrics["reference_tail_std"]),
                    atol=float(envelope_atol),
                    rtol=0.0,
                    notes="Late-window oscillation envelope compared against digitized stella/GENE mean trace.",
                )
            )
        rows.append(
            {
                "kx": float(kx),
                "spectrax_residual": float(obs["residual_level"]),
                "spectrax_residual_std": float(obs["residual_std"]),
                "spectrax_tmax": float(obs["tmax"]),
                "reference_residual": residual_ref,
                "reference_min": float(ref_row["reference_min"]),
                "reference_max": float(ref_row["reference_max"]),
                "reference_tmax": float(ref_row["reference_tmax"]),
                "coverage_ratio": float(coverage_ratio),
                "residual_abs_error": float(abs(float(obs["residual_level"]) - residual_ref)),
                "residual_atol_effective": residual_atol_eff,
                **trace_metrics,
            }
        )
    report = gate_report(
        "w7x_zonal_response_reference",
        "digitized stella/GENE W7-X test-4 Fig. 11",
        tuple(gates),
    )
    return pd.DataFrame(rows), report


def _write_plot(rows: pd.DataFrame, out_png: Path) -> None:
    set_plot_style()
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.fill_between(
        x,
        np.asarray(rows["reference_min"], dtype=float),
        np.asarray(rows["reference_max"], dtype=float),
        color="#8ecae6",
        alpha=0.45,
        label="digitized stella/GENE band",
    )
    ax.plot(x, rows["reference_residual"], color="#1d4e89", marker="o", linewidth=2.0, label="reference mean")
    ax.plot(x, rows["spectrax_residual"], color="#c2410c", marker="s", linewidth=2.0, label="SPECTRAX-GK")
    ax.set_xticks(x, [f"{value:.2f}" for value in rows["kx"]])
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel("late residual")
    ax.set_title("Residual Gate")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(x, rows["coverage_ratio"], color="#2a9d55", alpha=0.85)
    ax.axhline(0.98, color="#c2410c", linestyle="--", linewidth=1.5, label="98% gate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x, [f"{value:.2f}" for value in rows["kx"]])
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel("covered reference window")
    ax.set_title("Time Coverage")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows, report = build_comparison(
        spectrax_summary=args.spectrax_summary,
        reference_traces=args.reference_traces,
        reference_residuals=args.reference_residuals,
        spectrax_trace_dir=args.spectrax_trace_dir,
        spectrax_traces=args.spectrax_traces,
        residual_atol=float(args.residual_atol),
        residual_rtol=float(args.residual_rtol),
        coverage_fraction=float(args.coverage_fraction),
        tail_fraction=float(args.tail_fraction),
        envelope_atol=float(args.envelope_atol),
        trace_normalization=str(args.trace_normalization),
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_csv, index=False)
    _write_plot(rows, args.out_png)
    payload = {
        "case": "w7x_zonal_response_reference",
        "validation_status": "closed" if report.passed else "open",
        "gate_index_include": bool(args.gate_index_include),
        "gate_report": gate_report_to_dict(report),
        "spectrax_summary": _repo_relative(args.spectrax_summary),
        "spectrax_trace_dir": None if args.spectrax_trace_dir is None else _repo_relative(args.spectrax_trace_dir),
        "spectrax_traces": None if args.spectrax_traces is None else _repo_relative(args.spectrax_traces),
        "reference_traces": _repo_relative(args.reference_traces),
        "reference_residuals": _repo_relative(args.reference_residuals),
        "comparison_csv": _repo_relative(args.out_csv),
        "comparison_png": _repo_relative(args.out_png),
        "trace_normalization": str(args.trace_normalization),
        "notes": (
            "This gate compares SPECTRAX-GK W7-X test-4 zonal-flow residuals and, when trace CSVs are available, "
            "late-window oscillation envelopes against digitized stella/GENE Figure 11 references. The current "
            "paper-normalized long-window artifact closes the time-coverage gates, but residuals remain open at "
            "three wavelengths and the late-window envelope gates remain open. This is tracked as a physics/numerics "
            "closure lane rather than a documentation-only mismatch."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_png}")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
