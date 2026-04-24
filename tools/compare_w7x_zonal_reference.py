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


def _kx_token(kx: float) -> str:
    return f"{int(round(1000.0 * float(kx))):03d}"


def _reference_residual_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"kx_rhoi", "code", "residual_median"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    rows: list[dict[str, float]] = []
    for kx, group in table.groupby("kx_rhoi"):
        medians = np.asarray(group["residual_median"], dtype=float)
        if medians.size < 1:
            continue
        center = float(np.mean(medians))
        spread = float(np.max(np.abs(medians - center))) if medians.size > 1 else 0.0
        rows.append(
            {
                "kx": float(kx),
                "reference_residual": center,
                "reference_code_spread": spread,
                "reference_min": float(np.min(medians)),
                "reference_max": float(np.max(medians)),
            }
        )
    return pd.DataFrame(rows).sort_values("kx").reset_index(drop=True)


def _reference_time_limits(trace_table: pd.DataFrame) -> pd.DataFrame:
    required = {"kx_rhoi", "t_vti_over_a"}
    missing = required.difference(trace_table.columns)
    if missing:
        raise ValueError(f"reference trace table missing columns: {sorted(missing)}")
    rows = []
    for kx, group in trace_table.groupby("kx_rhoi"):
        t = np.asarray(group["t_vti_over_a"], dtype=float)
        rows.append({"kx": float(kx), "reference_tmax": float(np.nanmax(t)), "reference_tmin": float(np.nanmin(t))})
    return pd.DataFrame(rows)


def _load_spectrax_summary(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"kx_target", "residual_level", "residual_std", "tmax"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return table


def _trace_path(trace_dir: Path, kx: float) -> Path:
    return trace_dir / f"w7x_test4_kx{_kx_token(kx)}.csv"


def _normalize_trace(t: np.ndarray, y: np.ndarray, *, initial_level: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(t)
    t_sorted = np.asarray(t, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    finite = np.isfinite(t_sorted) & np.isfinite(y_sorted)
    t_sorted = t_sorted[finite]
    y_sorted = y_sorted[finite]
    if t_sorted.size == 0:
        raise ValueError("trace is empty after finite filtering")
    if initial_level is None:
        nz = np.flatnonzero(np.abs(y_sorted) > 1.0e-30)
        scale = float(abs(y_sorted[nz[0]])) if nz.size else 1.0
    else:
        scale = float(initial_level)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("trace normalization level must be finite and positive")
    return t_sorted, y_sorted / scale


def _optional_trace_metrics(
    *,
    trace_dir: Path | None,
    reference_traces: pd.DataFrame,
    kx: float,
    tail_fraction: float,
    initial_level: float | None,
) -> dict[str, float | int | None]:
    if trace_dir is None:
        return {
            "trace_available": 0,
            "tail_std": None,
            "reference_tail_std": None,
            "tail_mean_abs_error": None,
            "tail_max_abs_error": None,
        }
    path = _trace_path(trace_dir, kx)
    if not path.exists():
        return {
            "trace_available": 0,
            "tail_std": None,
            "reference_tail_std": None,
            "tail_mean_abs_error": None,
            "tail_max_abs_error": None,
        }
    trace = pd.read_csv(path)
    if not {"t", "phi_zonal_real"}.issubset(trace.columns):
        raise ValueError(f"{path} must contain t,phi_zonal_real columns")
    t_obs, y_obs = _normalize_trace(
        np.asarray(trace["t"], dtype=float),
        np.asarray(trace["phi_zonal_real"], dtype=float),
        initial_level=initial_level,
    )
    ref_subset = reference_traces[np.isclose(reference_traces["kx_rhoi"], float(kx))]
    ref_pivot = ref_subset.pivot_table(index="t_vti_over_a", columns="code", values="response", aggfunc="mean").sort_index()
    ref_t = np.asarray(ref_pivot.index, dtype=float)
    ref_y = np.asarray(ref_pivot.mean(axis=1), dtype=float)
    ref_tmax = float(np.nanmax(ref_t))
    tail_start = ref_tmax - float(tail_fraction) * (ref_tmax - float(np.nanmin(ref_t)))
    mask = (t_obs >= tail_start) & (t_obs <= ref_tmax)
    if not np.any(mask):
        return {
            "trace_available": 1,
            "tail_std": None,
            "reference_tail_std": None,
            "tail_mean_abs_error": None,
            "tail_max_abs_error": None,
        }
    ref_interp = np.interp(t_obs[mask], ref_t, ref_y)
    diff = y_obs[mask] - ref_interp
    ref_tail = ref_y[ref_t >= tail_start]
    return {
        "trace_available": 1,
        "tail_std": float(np.std(y_obs[mask])),
        "reference_tail_std": float(np.std(ref_tail)),
        "tail_mean_abs_error": float(np.mean(np.abs(diff))),
        "tail_max_abs_error": float(np.max(np.abs(diff))),
    }


def build_comparison(
    *,
    spectrax_summary: Path,
    reference_traces: Path,
    reference_residuals: Path,
    spectrax_trace_dir: Path | None = None,
    residual_atol: float = 0.02,
    residual_rtol: float = 0.10,
    coverage_fraction: float = 0.98,
    tail_fraction: float = 0.10,
    envelope_atol: float = 0.03,
    trace_normalization: str = "summary_initial_level",
):
    summary = _load_spectrax_summary(spectrax_summary)
    ref_traces = pd.read_csv(reference_traces)
    ref_residuals = _reference_residual_table(reference_residuals)
    ref_limits = _reference_time_limits(ref_traces)
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
            f"residual_kx{_kx_token(kx)}",
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
            f"time_coverage_kx{_kx_token(kx)}",
            coverage_ratio,
            1.0,
            atol=1.0 - float(coverage_fraction),
            rtol=0.0,
            notes=f"Passes when SPECTRAX reaches at least {coverage_fraction:.0%} of the digitized reference window.",
        )
        gates.extend([coverage_gate, residual_gate])
        trace_initial_level = None
        if str(trace_normalization).strip().lower().replace("-", "_") == "summary_initial_level":
            if spectrax_trace_dir is not None and "initial_level" not in obs.index:
                raise ValueError(
                    "summary_initial_level trace normalization requires an initial_level column "
                    "in the SPECTRAX summary CSV"
                )
            trace_initial_level = None if spectrax_trace_dir is None else float(obs["initial_level"])
        elif str(trace_normalization).strip().lower().replace("-", "_") != "first_nonzero":
            raise ValueError("trace_normalization must be one of {'summary_initial_level', 'first_nonzero'}")
        trace_metrics = _optional_trace_metrics(
            trace_dir=spectrax_trace_dir,
            reference_traces=ref_traces,
            kx=kx,
            tail_fraction=float(tail_fraction),
            initial_level=trace_initial_level,
        )
        if trace_metrics["tail_std"] is not None and trace_metrics["reference_tail_std"] is not None:
            gates.append(
                evaluate_scalar_gate(
                    f"tail_envelope_std_kx{_kx_token(kx)}",
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
        "reference_traces": _repo_relative(args.reference_traces),
        "reference_residuals": _repo_relative(args.reference_residuals),
        "comparison_csv": _repo_relative(args.out_csv),
        "comparison_png": _repo_relative(args.out_png),
        "trace_normalization": str(args.trace_normalization),
        "notes": (
            "This gate compares SPECTRAX-GK W7-X test-4 zonal-flow residuals and, when trace CSVs are available, "
            "late-window oscillation envelopes against digitized stella/GENE Figure 11 references. The current "
            "long-window Gaussian-potential artifact closes the residual and time-coverage gates, but the overall "
            "status remains open while the late-window envelope mismatch is tracked as a velocity-space recurrence "
            "and closure follow-up."
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
