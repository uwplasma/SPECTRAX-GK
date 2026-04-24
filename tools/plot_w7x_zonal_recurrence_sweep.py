#!/usr/bin/env python3
"""Plot a bounded W7-X zonal recurrence sweep with separated closure factors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import netCDF4 as nc
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "w7x_zonal_recurrence_sweep_kx070.png"
DEFAULT_REFERENCE = ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.csv"
DEFAULT_RUNS = (
    (
        "Nl8 Nm32 none",
        "moment_resolution",
        "none",
        ROOT / "tools_out" / "zonal_response" / "w7x_test4_line_gaussian_probe" / "w7x_test4_kx070.out.nc",
    ),
    (
        "Nl12 Nm48 none",
        "moment_resolution",
        "none",
        ROOT / "tools_out" / "zonal_response" / "w7x_recurrence_nl12_nm48_none_t100" / "w7x_test4_kx070.out.nc",
    ),
    (
        "Nl16 Nm64 none",
        "moment_resolution",
        "none",
        ROOT / "tools_out" / "zonal_response" / "w7x_publication_nl16_nm64_dt005_t100" / "w7x_test4_kx070.out.nc",
    ),
    (
        "Nl16 Nm64 none baseline",
        "closure_source",
        "none",
        ROOT / "tools_out" / "zonal_response" / "w7x_publication_nl16_nm64_dt005_t100" / "w7x_test4_kx070.out.nc",
    ),
    (
        "Nl16 Nm64 const",
        "closure_source",
        "const",
        ROOT
        / "tools_out"
        / "zonal_response"
        / "w7x_closure_probe_nl16_nm64_nuhm003_const_t100"
        / "w7x_test4_kx070.out.nc",
    ),
    (
        "Nl16 Nm64 kz",
        "closure_source",
        "kz",
        ROOT / "tools_out" / "zonal_response" / "w7x_recurrence_nl16_nm64_kz003_t100" / "w7x_test4_kx070.out.nc",
    ),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-traces", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--kx", type=float, default=0.07)
    parser.add_argument("--analysis-tmax", type=float, default=100.0, help="Common analysis cutoff for all runs.")
    parser.add_argument("--tail-fraction", type=float, default=0.30)
    parser.add_argument(
        "--run",
        nargs=4,
        action="append",
        metavar=("LABEL", "SWEEP", "CLOSURE_SOURCE", "OUT_NC"),
        help="Run label, sweep family, closure source, and out.nc path. Can be repeated.",
    )
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
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


def load_reference_trace(path: Path, kx: float) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(path)
    required = {"kx_rhoi", "code", "t_vti_over_a", "response"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    subset = table[np.isclose(table["kx_rhoi"], float(kx))]
    if subset.empty:
        raise ValueError(f"{path} has no reference trace for kx={kx}")
    pivot = subset.pivot_table(index="t_vti_over_a", columns="code", values="response", aggfunc="mean").sort_index()
    return np.asarray(pivot.index, dtype=float), np.asarray(pivot.mean(axis=1), dtype=float)


def _selected_kx_index(kx_grid: np.ndarray, target_kx: float) -> tuple[int, float]:
    grid = np.asarray(kx_grid, dtype=float)
    positive = np.flatnonzero(grid > 0.0)
    if positive.size:
        idx = int(positive[np.argmin(np.abs(grid[positive] - float(target_kx)))])
    else:
        idx = int(np.argmin(np.abs(grid - float(target_kx))))
    return idx, float(grid[idx])


def load_run(
    *,
    label: str,
    sweep: str,
    closure_source: str,
    path: Path,
    kx: float,
    reference_t: np.ndarray,
    reference_y: np.ndarray,
    analysis_tmax: float,
    tail_fraction: float,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load one recurrence run and compute common-window scalar metrics."""

    with nc.Dataset(path) as ds:
        t = np.asarray(ds["Grids/time"][:], dtype=float)
        kx_grid = np.asarray(ds["Grids/kx"][:], dtype=float)
        phi = np.asarray(ds["Diagnostics/Phi_zonal_line_kxt"][:, :, 0], dtype=float)
        wg_lm = np.asarray(ds["Diagnostics/Wg_lmst"][:, 0, :, :], dtype=float)
    kx_index, kx_selected = _selected_kx_index(kx_grid, kx)
    raw = phi[:, kx_index]
    nz = np.flatnonzero(np.abs(raw) > 1.0e-30)
    initial_level = float(abs(raw[nz[0]])) if nz.size else 1.0
    response = raw / initial_level
    tmax = min(float(np.nanmax(t)), float(analysis_tmax), float(np.nanmax(reference_t)))
    mask = (t >= 0.0) & (t <= tmax)
    if not np.any(mask):
        raise ValueError(f"{path} has no samples before analysis_tmax={analysis_tmax}")
    ref_interp = np.interp(t[mask], reference_t, reference_y)
    diff = response[mask] - ref_interp
    tail_start = tmax - float(tail_fraction) * (tmax - float(np.nanmin(t[mask])))
    tail_mask = mask & (t >= tail_start)
    total = np.sum(wg_lm, axis=(1, 2))
    safe = np.maximum(total, 1.0e-300)
    nm = int(wg_lm.shape[1])
    nl = int(wg_lm.shape[2])
    m_tail = min(nm, max(4, int(np.ceil(0.125 * nm))))
    l_tail = min(nl, max(2, int(np.ceil(0.25 * nl))))
    hermite_tail = np.sum(wg_lm[:, -m_tail:, :], axis=(1, 2)) / safe
    laguerre_tail = np.sum(wg_lm[:, :, -l_tail:], axis=(1, 2)) / safe
    last_idx = int(np.flatnonzero(mask)[-1])
    row = {
        "label": str(label),
        "sweep": str(sweep),
        "closure_source": str(closure_source),
        "source_path": _repo_relative(path),
        "kx_target": float(kx),
        "kx_selected": float(kx_selected),
        "kx_index": int(kx_index),
        "analysis_tmax": float(tmax),
        "raw_tmax": float(np.nanmax(t)),
        "n_samples_used": int(np.count_nonzero(mask)),
        "Nl": nl,
        "Nm": nm,
        "initial_level": initial_level,
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "tail_mean": float(np.mean(response[tail_mask])),
        "tail_std": float(np.std(response[tail_mask])),
        "tail_abs_max": float(np.max(np.abs(response[tail_mask]))),
        "reference_tail_std": float(np.std(ref_interp[t[mask] >= tail_start])),
        "hermite_tail_at_tmax": float(hermite_tail[last_idx]),
        "laguerre_tail_at_tmax": float(laguerre_tail[last_idx]),
        "free_energy_at_tmax_over_initial": float(total[last_idx] / safe[0]),
    }
    trace = {"t": t[mask], "response": response[mask]}
    return row, trace


def build_sweep(
    runs: list[tuple[str, str, str, Path]],
    *,
    reference_t: np.ndarray,
    reference_y: np.ndarray,
    kx: float,
    analysis_tmax: float,
    tail_fraction: float,
) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    traces: dict[str, dict[str, np.ndarray]] = {}
    for label, sweep, closure_source, path in runs:
        if not path.exists():
            continue
        row, trace = load_run(
            label=label,
            sweep=sweep,
            closure_source=closure_source,
            path=path,
            kx=kx,
            reference_t=reference_t,
            reference_y=reference_y,
            analysis_tmax=analysis_tmax,
            tail_fraction=tail_fraction,
        )
        rows.append(row)
        traces[str(label)] = trace
    return rows, traces


def recurrence_figure(
    rows: list[dict[str, object]],
    traces: dict[str, dict[str, np.ndarray]],
    reference_t: np.ndarray,
    reference_y: np.ndarray,
    *,
    analysis_tmax: float,
) -> plt.Figure:
    if not rows:
        raise ValueError("no recurrence sweep rows to plot")
    set_plot_style()
    label_palette = ["#0f4c81", "#2a9d8f", "#4f46e5", "#6b7280", "#c2410c", "#7b2cbf"]
    label_colors = {str(row["label"]): label_palette[idx % len(label_palette)] for idx, row in enumerate(rows)}
    plot_ids = {str(row["label"]): chr(ord("A") + idx) for idx, row in enumerate(rows)}
    display_labels = {str(row["label"]): f"{plot_ids[str(row['label'])]}  {row['label']}" for row in rows}
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)
    for axis, sweep, title in (
        (axes[0, 0], "moment_resolution", "Moment-resolution sweep"),
        (axes[0, 1], "closure_source", "Closure-source sweep"),
    ):
        ref_mask = reference_t <= float(analysis_tmax)
        axis.plot(reference_t[ref_mask], reference_y[ref_mask], color="#111827", linewidth=2.4, label="stella/GENE mean")
        for row in rows:
            if str(row["sweep"]) != sweep:
                continue
            trace = traces[str(row["label"])]
            color = label_colors[str(row["label"])]
            axis.plot(
                trace["t"],
                trace["response"],
                linewidth=1.85,
                alpha=0.9,
                color=color,
                label=display_labels[str(row["label"])],
            )
        axis.set_xlabel(r"$t v_{ti}/a$")
        axis.set_ylabel(r"$\phi_z/\phi_z(0)$")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=True, framealpha=0.92, fontsize=8)

    ax = axes[1, 0]
    labels = [display_labels[str(row["label"])] for row in rows]
    y = np.arange(len(rows))
    bar_colors = [label_colors[str(row["label"])] for row in rows]
    ax.barh(y - 0.18, [float(row["mean_abs_error"]) for row in rows], height=0.34, color=bar_colors, alpha=0.85, label="mean abs error")
    ax.barh(y + 0.18, [float(row["tail_std"]) for row in rows], height=0.34, color="#6b7280", alpha=0.72, label="tail std")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("paper-normalized metric")
    ax.set_title("Reference mismatch and recurrence envelope")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    scatter = ax.scatter(
        [float(row["hermite_tail_at_tmax"]) for row in rows],
        [float(row["tail_std"]) for row in rows],
        c=[float(row["mean_abs_error"]) for row in rows],
        s=100,
        cmap="viridis_r",
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    offsets = [(5, 6), (5, -13), (5, 8), (-24, -18), (6, 12), (-12, 10)]
    for row, offset in zip(rows, offsets, strict=False):
        ax.annotate(
            plot_ids[str(row["label"])],
            (float(row["hermite_tail_at_tmax"]), float(row["tail_std"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xlabel("Hermite-tail fraction at analysis cutoff")
    ax.set_ylabel("late-window std")
    ax.set_title("Moment-tail vs recurrence envelope")
    ax.margins(x=0.14, y=0.12)
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("mean abs error")
    fig.suptitle(r"W7-X test-4 bounded recurrence sweep, $k_x\rho_i=0.07$", y=1.02, fontsize=14, fontweight="bold")
    return fig


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    *,
    rows: list[dict[str, object]],
    out_json: Path,
    out_csv: Path,
    out_png: Path,
    runs: list[tuple[str, str, str, Path]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "case": "w7x_zonal_response_recurrence_sweep_kx070",
        "validation_status": "open",
        "gate_index_include": False,
        "reference": "Gonzalez-Jerez et al., J. Plasma Phys. 88, 905880310 (2022), W7-X test 4",
        "audit_csv": _repo_relative(out_csv),
        "audit_png": _repo_relative(out_png),
        "reference_traces": _repo_relative(args.reference_traces),
        "kx": float(args.kx),
        "analysis_tmax": float(args.analysis_tmax),
        "tail_fraction": float(args.tail_fraction),
        "runs": [
            {"label": label, "sweep": sweep, "closure_source": source, "path": _repo_relative(path)}
            for label, sweep, source, path in runs
            if path.exists()
        ],
        "rows": rows,
        "notes": (
            "This is an open recurrence audit. It separates moment-resolution changes from closure-source "
            "changes at fixed paper-facing Gaussian width and line-average normalization. The analysis cutoff "
            "keeps all rows on the same bounded window, so long existing runs and new shorter probes can be "
            "compared without mixing tail horizons."
        ),
    }
    out_json.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _runs_from_args(args: argparse.Namespace) -> list[tuple[str, str, str, Path]]:
    if args.run:
        return [(str(label), str(sweep), str(source), Path(path)) for label, sweep, source, path in args.run]
    return [(label, sweep, source, path) for label, sweep, source, path in DEFAULT_RUNS]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runs = _runs_from_args(args)
    reference_t, reference_y = load_reference_trace(args.reference_traces, float(args.kx))
    rows, traces = build_sweep(
        runs,
        reference_t=reference_t,
        reference_y=reference_y,
        kx=float(args.kx),
        analysis_tmax=float(args.analysis_tmax),
        tail_fraction=float(args.tail_fraction),
    )
    if not rows:
        raise ValueError("no W7-X out.nc files found for the requested recurrence sweep")
    out_csv = args.out_csv or args.out_png.with_suffix(".csv")
    out_json = args.out_json or args.out_png.with_suffix(".json")
    fig = recurrence_figure(rows, traces, reference_t, reference_y, analysis_tmax=float(args.analysis_tmax))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=240, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    write_csv(rows, out_csv)
    write_json(rows=rows, out_json=out_json, out_csv=out_csv, out_png=args.out_png, runs=runs, args=args)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
