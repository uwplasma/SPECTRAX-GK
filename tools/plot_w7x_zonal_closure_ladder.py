#!/usr/bin/env python3
"""Plot a bounded W7-X zonal-response closure ladder for ``kx rho_i=0.07``.

This script compares selected existing SPECTRAX-GK W7-X test-4 outputs against
the digitized stella/GENE trace. It is intended to make the open
recurrence/closure investigation reproducible: moment resolution, weak closure
knobs, and non-contract initializer audits are displayed separately from the
paper-facing validation contract.
"""

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
DEFAULT_OUT = ROOT / "docs" / "_static" / "w7x_zonal_closure_ladder_kx070.png"
DEFAULT_REFERENCE = ROOT / "docs" / "_static" / "w7x_zonal_reference_digitized.csv"
DEFAULT_KX = 0.07
DEFAULT_RUNS = (
    (
        "paper: Nl8 Nm32 long",
        "paper",
        ROOT / "tools_out" / "zonal_response" / "w7x_test4_line_gaussian_probe" / "w7x_test4_kx070.out.nc",
    ),
    (
        "paper: Nl16 Nm64 t100",
        "paper-resolution",
        ROOT / "tools_out" / "zonal_response" / "w7x_publication_nl16_nm64_dt005_t100" / "w7x_test4_kx070.out.nc",
    ),
    (
        "paper: Nl16 Nm64 t180",
        "paper-resolution",
        ROOT / "tools_out" / "zonal_response" / "w7x_stability_probe_dt005_nl16_nm64_kx070_t200" / "w7x_test4_kx070.out.nc",
    ),
    (
        "weak closure: nu_hyper_m=1e-2",
        "closure-audit",
        ROOT / "tools_out" / "zonal_response" / "w7x_test4_line_hyper_nu1e2" / "w7x_test4_kx070.out.nc",
    ),
    (
        "weak closure: Nl16 Nm64 nu_hyper_m=3e-3",
        "closure-audit",
        ROOT
        / "tools_out"
        / "zonal_response"
        / "w7x_closure_probe_nl16_nm64_nuhm003_const_t100"
        / "w7x_test4_kx070.out.nc",
    ),
    (
        "non-contract: width=4 high moment",
        "initializer-audit",
        ROOT / "tools_out" / "zonal_response" / "w7x_width4_nl16_nm64_dt005_kx070_t100" / "w7x_test4_kx070.out.nc",
    ),
    (
        "non-contract: width=4 Nl24 Nm96",
        "initializer-audit",
        ROOT / "tools_out" / "zonal_response" / "w7x_width4_nl24_nm96_dt0025_kx070_t25" / "w7x_test4_kx070.out.nc",
    ),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-traces", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--kx", type=float, default=DEFAULT_KX)
    parser.add_argument("--t-compare", type=float, default=200.0, help="Early-window trace-comparison horizon.")
    parser.add_argument("--tail-fraction", type=float, default=0.30, help="Late-window fraction for trace std.")
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        metavar=("LABEL", "FAMILY", "OUT_NC"),
        help="Run label, family, and out.nc path. Can be repeated.",
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


def load_reference_trace(reference_traces: Path, kx: float) -> tuple[np.ndarray, np.ndarray]:
    """Load the digitized mean stella/GENE W7-X zonal trace for one ``kx``."""

    table = pd.read_csv(reference_traces)
    required = {"kx_rhoi", "code", "t_vti_over_a", "response"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{reference_traces} missing columns: {sorted(missing)}")
    subset = table[np.isclose(table["kx_rhoi"], float(kx))]
    if subset.empty:
        raise ValueError(f"{reference_traces} has no reference trace for kx={kx}")
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


def load_variant_trace(path: Path, *, kx: float) -> dict[str, object]:
    """Load one SPECTRAX-GK W7-X zonal output trace and moment-tail diagnostics."""

    with nc.Dataset(path) as ds:
        t = np.asarray(ds["Grids/time"][:], dtype=float)
        kx_grid = np.asarray(ds["Grids/kx"][:], dtype=float)
        phi = np.asarray(ds["Diagnostics/Phi_zonal_line_kxt"][:, :, 0], dtype=float)
        wg_lm = (
            np.asarray(ds["Diagnostics/Wg_lmst"][:, 0, :, :], dtype=float)
            if "Wg_lmst" in ds["Diagnostics"].variables
            else None
        )
    kx_index, kx_selected = _selected_kx_index(kx_grid, kx)
    raw = phi[:, kx_index]
    nz = np.flatnonzero(np.abs(raw) > 1.0e-30)
    initial_level = float(abs(raw[nz[0]])) if nz.size else 1.0
    response = raw / initial_level
    out: dict[str, object] = {
        "t": t,
        "response": response,
        "kx_selected": kx_selected,
        "kx_index": int(kx_index),
        "initial_level": initial_level,
        "tmax": float(np.nanmax(t)),
        "n_time": int(t.size),
        "Nm": None,
        "Nl": None,
        "hermite_tail_last": None,
        "laguerre_tail_last": None,
        "free_energy_last_over_first": None,
    }
    if wg_lm is not None:
        total = np.sum(wg_lm, axis=(1, 2))
        safe = np.maximum(total, 1.0e-300)
        nm = int(wg_lm.shape[1])
        nl = int(wg_lm.shape[2])
        m_tail = max(4, int(np.ceil(0.125 * nm)))
        l_tail = max(2, int(np.ceil(0.25 * nl)))
        out.update(
            {
                "Nm": nm,
                "Nl": nl,
                "hermite_tail_last": float(np.sum(wg_lm[-1, -m_tail:, :]) / safe[-1]),
                "laguerre_tail_last": float(np.sum(wg_lm[-1, :, -l_tail:]) / safe[-1]),
                "free_energy_last_over_first": float(total[-1] / safe[0]),
            }
        )
    return out


def build_rows(
    runs: list[tuple[str, str, Path]],
    *,
    reference_t: np.ndarray,
    reference_y: np.ndarray,
    kx: float,
    t_compare: float,
    tail_fraction: float,
) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    """Build scalar closure-ladder metrics and keep plotted traces."""

    rows: list[dict[str, object]] = []
    traces: dict[str, dict[str, np.ndarray]] = {}
    for label, family, path in runs:
        if not path.exists():
            continue
        loaded = load_variant_trace(path, kx=kx)
        t = np.asarray(loaded["t"], dtype=float)
        response = np.asarray(loaded["response"], dtype=float)
        compare_tmax = min(float(t_compare), float(np.nanmax(t)), float(np.nanmax(reference_t)))
        mask = (t >= 0.0) & (t <= compare_tmax)
        if not np.any(mask):
            raise ValueError(f"{path} has no samples in the comparison window")
        reference_interp = np.interp(t[mask], reference_t, reference_y)
        diff = response[mask] - reference_interp
        tail_start = float(np.nanmax(t)) - float(tail_fraction) * (float(np.nanmax(t)) - float(np.nanmin(t)))
        tail_mask = t >= tail_start
        rows.append(
            {
                "label": label,
                "family": family,
                "source_path": _repo_relative(path),
                "kx_target": float(kx),
                "kx_selected": float(loaded["kx_selected"]),
                "tmax": float(loaded["tmax"]),
                "n_time": int(loaded["n_time"]),
                "Nm": loaded["Nm"],
                "Nl": loaded["Nl"],
                "mean_abs_error": float(np.mean(np.abs(diff))),
                "max_abs_error": float(np.max(np.abs(diff))),
                "final_window_tmax": float(compare_tmax),
                "tail_mean": float(np.mean(response[tail_mask])),
                "tail_std": float(np.std(response[tail_mask])),
                "tail_abs_max": float(np.max(np.abs(response[tail_mask]))),
                "hermite_tail_last": loaded["hermite_tail_last"],
                "laguerre_tail_last": loaded["laguerre_tail_last"],
                "free_energy_last_over_first": loaded["free_energy_last_over_first"],
            }
        )
        traces[label] = {"t": t, "response": response}
    return rows, traces


def closure_ladder_figure(
    rows: list[dict[str, object]],
    traces: dict[str, dict[str, np.ndarray]],
    reference_t: np.ndarray,
    reference_y: np.ndarray,
    *,
    t_compare: float,
) -> plt.Figure:
    """Create a four-panel closure-ladder figure."""

    if not rows:
        raise ValueError("no W7-X closure-ladder rows to plot")
    set_plot_style()
    labels = [str(row["label"]) for row in rows]
    families = [str(row["family"]) for row in rows]
    plot_ids = [chr(ord("A") + idx) for idx in range(len(rows))]
    plot_labels = [
        label.replace("paper: ", "")
        .replace("weak closure: ", "weak: ")
        .replace("non-contract: ", "audit: ")
        .replace("high moment", "Nl16 Nm64")
        for label in labels
    ]
    plot_labels = [f"{plot_id}  {label}" for plot_id, label in zip(plot_ids, plot_labels, strict=True)]
    family_colors = {
        "paper": "#0f4c81",
        "paper-resolution": "#2a9d8f",
        "closure-audit": "#c2410c",
        "initializer-audit": "#7b2cbf",
    }
    colors = [family_colors.get(family, "#555555") for family in families]
    y_pos = np.arange(len(rows))

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    ref_mask = reference_t <= float(t_compare)
    ax.plot(reference_t[ref_mask], reference_y[ref_mask], color="#111827", linewidth=2.4, label="digitized stella/GENE mean")
    for row, color, plot_id in zip(rows, colors, plot_ids, strict=True):
        trace = traces[str(row["label"])]
        t = np.asarray(trace["t"], dtype=float)
        response = np.asarray(trace["response"], dtype=float)
        mask = t <= float(t_compare)
        ax.plot(t[mask], response[mask], linewidth=1.65, alpha=0.88, color=color, label=plot_id)
    ax.set_xlabel(r"$t v_{ti}/a$")
    ax.set_ylabel(r"$\phi_z/\phi_z(0)$")
    ax.set_title(r"Trace overlay, $k_x \rho_i = 0.07$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, framealpha=0.90, fontsize=7, ncol=4, loc="lower left")

    ax = axes[0, 1]
    ax.barh(y_pos, [float(row["mean_abs_error"]) for row in rows], color=colors, alpha=0.85)
    ax.set_yticks(y_pos, plot_labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"mean $|\Delta \phi_z|$ to reference")
    ax.set_title("Reference-trace mismatch")
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1, 0]
    ax.barh(y_pos, [float(row["tail_std"]) for row in rows], color=colors, alpha=0.85)
    ax.set_yticks(y_pos, plot_labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"late-window std$(\phi_z/\phi_z(0))$")
    ax.set_title("Late-envelope response")
    ax.grid(True, axis="x", alpha=0.25)

    ax = axes[1, 1]
    h = np.asarray([np.nan if row["hermite_tail_last"] is None else float(row["hermite_tail_last"]) for row in rows])
    laguerre_tail = np.asarray(
        [np.nan if row["laguerre_tail_last"] is None else float(row["laguerre_tail_last"]) for row in rows]
    )
    err = np.asarray([float(row["mean_abs_error"]) for row in rows])
    scatter = ax.scatter(h, laguerre_tail, c=err, s=90, cmap="viridis_r", edgecolor="white", linewidth=0.9, zorder=3)
    for plot_id, hx, ly in zip(plot_ids, h, laguerre_tail, strict=True):
        if np.isfinite(hx) and np.isfinite(ly):
            ax.annotate(plot_id, (hx, ly), xytext=(5, 4), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.set_xlabel("final Hermite-tail fraction")
    ax.set_ylabel("final Laguerre-tail fraction")
    ax.set_title("Moment-tail state")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"mean $|\Delta \phi_z|$")
    fig.suptitle("W7-X test-4 zonal closure ladder", y=1.02, fontsize=14, fontweight="bold")
    return fig


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    *,
    rows: list[dict[str, object]],
    out_json: Path,
    out_csv: Path,
    out_png: Path,
    runs: list[tuple[str, str, Path]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "case": "w7x_zonal_response_closure_ladder_kx070",
        "validation_status": "open",
        "gate_index_include": False,
        "reference": "Gonzalez-Jerez et al., J. Plasma Phys. 88, 905880310 (2022), W7-X test 4",
        "audit_csv": _repo_relative(out_csv),
        "audit_png": _repo_relative(out_png),
        "reference_traces": _repo_relative(args.reference_traces),
        "kx": float(args.kx),
        "t_compare": float(args.t_compare),
        "tail_fraction": float(args.tail_fraction),
        "runs": [
            {"label": label, "family": family, "path": _repo_relative(path)}
            for label, family, path in runs
            if path.exists()
        ],
        "rows": rows,
        "notes": (
            "This is an open closure ladder, not a validation gate. It separates paper-contract resolution "
            "changes from weak-closure and non-contract initializer audits. Weak closure reduces the late "
            "envelope in existing probes but does not close the early trace mismatch, so it should not be "
            "used as a hidden validation setting."
        ),
    }
    out_json.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _runs_from_args(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    if args.run:
        return [(str(label), str(family), Path(path)) for label, family, path in args.run]
    return [(label, family, path) for label, family, path in DEFAULT_RUNS]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runs = _runs_from_args(args)
    reference_t, reference_y = load_reference_trace(args.reference_traces, float(args.kx))
    rows, traces = build_rows(
        runs,
        reference_t=reference_t,
        reference_y=reference_y,
        kx=float(args.kx),
        t_compare=float(args.t_compare),
        tail_fraction=float(args.tail_fraction),
    )
    if not rows:
        raise ValueError("no existing W7-X out.nc files found for the requested closure ladder")
    out_csv = args.out_csv or args.out_png.with_suffix(".csv")
    out_json = args.out_json or args.out_png.with_suffix(".json")
    fig = closure_ladder_figure(rows, traces, reference_t, reference_y, t_compare=float(args.t_compare))
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
