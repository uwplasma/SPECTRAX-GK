#!/usr/bin/env python3
"""Plot the W7-X test-4 zonal-response validation contract audit.

This script does not run simulations. It combines the tracked SPECTRAX-GK
W7-X zonal response artifacts with digitized stella/GENE Fig. 11 data from the
González-Jerez et al. W7-X benchmark paper. The output is deliberately
paper-facing but marked as an open audit until the residual and late-envelope
gates close under the literature normalization convention.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.zonal_validation import load_w7x_combined_trace_csv, reference_mean_trace  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "w7x_zonal_contract_audit.png"
TRACE_OVERLAY_KX = (0.07, 0.30)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-traces",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.csv",
        help="Digitized stella/GENE Fig. 11 trace CSV.",
    )
    parser.add_argument(
        "--reference-residuals",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized_residuals.csv",
        help="Digitized stella/GENE Fig. 11 inset residual CSV.",
    )
    parser.add_argument(
        "--spectrax-summary",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_response_panel.csv",
        help="SPECTRAX-GK W7-X zonal summary CSV.",
    )
    parser.add_argument(
        "--spectrax-traces",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_response_panel.traces.csv",
        help="Combined SPECTRAX-GK W7-X zonal trace CSV from generate_w7x_zonal_response_panel.py.",
    )
    parser.add_argument(
        "--compare-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.csv",
        help="Residual/time/envelope comparison CSV from compare_w7x_zonal_reference.py.",
    )
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT, help="Output PNG path.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output audit CSV path.")
    parser.add_argument("--out-json", type=Path, default=None, help="Output audit JSON path.")
    parser.add_argument("--residual-rtol", type=float, default=0.10)
    parser.add_argument("--envelope-atol", type=float, default=0.03)
    return parser.parse_args(argv)


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def load_audit_rows(
    compare_csv: Path,
    *,
    residual_rtol: float = 0.10,
    envelope_atol: float = 0.03,
) -> list[dict[str, object]]:
    """Build a compact audit table from the tracked W7-X comparison CSV."""

    table = pd.read_csv(compare_csv)
    required = {
        "kx",
        "spectrax_residual",
        "reference_residual",
        "reference_min",
        "reference_max",
        "coverage_ratio",
        "residual_abs_error",
        "residual_atol_effective",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{compare_csv} missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for _, item in table.sort_values("kx").iterrows():
        ref = float(item["reference_residual"])
        residual_tol = float(item["residual_atol_effective"]) + float(residual_rtol) * abs(ref)
        tail_std = float(item["tail_std"]) if "tail_std" in table.columns and pd.notna(item["tail_std"]) else np.nan
        reference_tail_std = (
            float(item["reference_tail_std"])
            if "reference_tail_std" in table.columns and pd.notna(item["reference_tail_std"])
            else np.nan
        )
        tail_ratio = tail_std / reference_tail_std if np.isfinite(tail_std) and reference_tail_std > 0.0 else np.nan
        rows.append(
            {
                "kx": float(item["kx"]),
                "spectrax_residual": float(item["spectrax_residual"]),
                "reference_residual": ref,
                "reference_min": float(item["reference_min"]),
                "reference_max": float(item["reference_max"]),
                "residual_abs_error": float(item["residual_abs_error"]),
                "residual_tolerance": residual_tol,
                "tail_std": tail_std,
                "reference_tail_std": reference_tail_std,
                "tail_std_ratio": tail_ratio,
                "coverage_ratio": float(item["coverage_ratio"]),
                "residual_gate_passed": bool(float(item["residual_abs_error"]) <= residual_tol),
                "tail_gate_passed": bool(
                    np.isfinite(tail_std)
                    and np.isfinite(reference_tail_std)
                    and abs(tail_std - reference_tail_std) <= float(envelope_atol)
                ),
            }
        )
    return rows


def audit_figure(
    rows: list[dict[str, object]],
    reference_traces: pd.DataFrame,
    spectrax_traces: Path,
    *,
    overlay_kx: tuple[float, ...] = TRACE_OVERLAY_KX,
) -> plt.Figure:
    """Create the four-panel W7-X zonal-response contract audit figure."""

    if not rows:
        raise ValueError("no W7-X audit rows to plot")
    set_plot_style()
    kx = np.asarray([float(row["kx"]) for row in rows])
    x = np.arange(len(rows))
    xlabels = [f"{value:.2f}" for value in kx]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.fill_between(
        x,
        np.asarray([float(row["reference_min"]) for row in rows]),
        np.asarray([float(row["reference_max"]) for row in rows]),
        color="#8ecae6",
        alpha=0.42,
        label="digitized stella/GENE band",
    )
    ax.plot(x, [float(row["reference_residual"]) for row in rows], marker="o", linewidth=2.2, color="#1d4e89", label="reference mean")
    ax.plot(x, [float(row["spectrax_residual"]) for row in rows], marker="s", linewidth=2.2, color="#c2410c", label="SPECTRAX-GK")
    ax.set_xticks(x, xlabels)
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel("late residual")
    ax.set_title("Residual level")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    width = 0.34
    ref_tail = np.asarray([float(row["reference_tail_std"]) for row in rows], dtype=float)
    obs_tail = np.asarray([float(row["tail_std"]) for row in rows], dtype=float)
    ax.bar(x - width / 2, ref_tail, width=width, color="#1d4e89", alpha=0.78, label="reference")
    ax.bar(x + width / 2, obs_tail, width=width, color="#c2410c", alpha=0.78, label="SPECTRAX-GK")
    ax.set_yscale("log")
    ax.set_xticks(x, xlabels)
    ax.set_xlabel(r"$k_x \rho_i$")
    ax.set_ylabel("late-window standard deviation")
    ax.set_title("Late envelope")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    for ax, this_kx in zip(axes[1, :], overlay_kx, strict=True):
        ref_t, ref_y = reference_mean_trace(reference_traces, this_kx)
        obs_t, obs_y = load_w7x_combined_trace_csv(spectrax_traces, this_kx, normalized=True)
        ax.plot(ref_t, ref_y, color="#1d4e89", linewidth=2.0, label="digitized mean")
        ax.plot(obs_t, obs_y, color="#c2410c", linewidth=1.7, alpha=0.92, label="SPECTRAX-GK")
        ax.set_xlabel(r"$t v_{ti}/a$")
        ax.set_ylabel(r"$\phi_z/\phi_z(0)$")
        ax.set_title(fr"Trace overlay, $k_x \rho_i={this_kx:.2f}$")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("W7-X test-4 zonal-response contract audit", y=1.02, fontsize=14, fontweight="bold")
    return fig


def write_rows(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    *,
    rows: list[dict[str, object]],
    out_json: Path,
    out_csv: Path,
    out_png: Path,
    args: argparse.Namespace,
) -> None:
    payload = {
        "case": "w7x_zonal_response_contract_audit",
        "validation_status": "open",
        "gate_index_include": False,
        "reference": "Gonzalez-Jerez et al., J. Plasma Phys. 88, 905880310 (2022), W7-X test 4",
        "reference_contract": {
            "observable": "unweighted line-averaged electrostatic potential",
            "normalization": "line-averaged potential normalized to its t=0 line-average value",
            "kx_rhoi": [float(row["kx"]) for row in rows],
        },
        "all_residual_gates_pass": all(bool(row["residual_gate_passed"]) for row in rows),
        "all_tail_gates_pass": all(bool(row["tail_gate_passed"]) for row in rows),
        "audit_csv": _repo_relative(out_csv),
        "audit_png": _repo_relative(out_png),
        "reference_traces": _repo_relative(args.reference_traces),
        "reference_residuals": _repo_relative(args.reference_residuals),
        "spectrax_summary": _repo_relative(args.spectrax_summary),
        "spectrax_traces": _repo_relative(args.spectrax_traces),
        "compare_csv": _repo_relative(args.compare_csv),
        "rows": rows,
        "notes": (
            "This figure is a diagnostic audit, not a closed validation gate. It intentionally uses "
            "the paper-text line-first normalization rather than the clipped initial value visible in "
            "the published figure. The current artifact reaches the reference time windows, but the "
            "residual and late-envelope mismatches remain open physics/numerics work."
        ),
    }
    out_json.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_csv = args.out_csv or args.out_png.with_suffix(".csv")
    out_json = args.out_json or args.out_png.with_suffix(".json")
    rows = load_audit_rows(
        args.compare_csv,
        residual_rtol=float(args.residual_rtol),
        envelope_atol=float(args.envelope_atol),
    )
    reference_traces = pd.read_csv(args.reference_traces)
    fig = audit_figure(rows, reference_traces, args.spectrax_traces)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=240, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_rows(rows, out_csv)
    write_metadata(rows=rows, out_json=out_json, out_csv=out_csv, out_png=args.out_png, args=args)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
