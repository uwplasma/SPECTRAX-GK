#!/usr/bin/env python3
"""Build the tracked GX-vs-SPECTRAX README summary panel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_secondary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ky", "kx", "gamma_gx", "gamma_sp", "rel_gamma", "omega_gx", "omega_sp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    if "abs_omega" not in df.columns:
        df["abs_omega"] = (df["omega_sp"] - df["omega_gx"]).abs()
    return df.sort_values(["ky", "kx"]).reset_index(drop=True)


def _secondary_table_rows(df: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in df.itertuples(index=False):
        rows.append(
            [
                f"({row.ky:.2f}, {row.kx:+.2f})",
                f"{row.gamma_gx:.6f}",
                f"{row.gamma_sp:.6f}",
                f"{row.rel_gamma:.2e}",
                f"{row.omega_gx:.2e}",
                f"{row.omega_sp:.2e}",
                f"{row.abs_omega:.2e}",
            ]
        )
    return rows


def _load_imported_linear(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "ky",
        "mean_abs_omega",
        "mean_rel_omega",
        "mean_abs_gamma",
        "mean_rel_gamma",
        "mean_rel_Wg",
        "mean_rel_Wphi",
        "mean_rel_Wapar",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df.sort_values("ky").reset_index(drop=True)


def _linear_table_rows(df: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in df.itertuples(index=False):
        rows.append(
            [
                f"{row.ky:.3f}",
                f"{row.mean_abs_omega:.2e}",
                f"{row.mean_abs_gamma:.2e}",
                f"{row.mean_rel_Wg:.2e}",
                f"{row.mean_rel_Wphi:.2e}",
            ]
        )
    return rows


def _load_cetg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "W_spectrax",
        "W_gx",
        "Phi2_spectrax",
        "Phi2_gx",
        "qflux_spectrax",
        "qflux_gx",
        "pflux_spectrax",
        "pflux_gx",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df


def _mean_rel(lhs: np.ndarray, rhs: np.ndarray, floor_fraction: float = 1.0e-8) -> float:
    lhs_f = np.asarray(lhs, dtype=float)
    rhs_f = np.asarray(rhs, dtype=float)
    finite = np.isfinite(lhs_f) & np.isfinite(rhs_f)
    if not np.any(finite):
        return float("nan")
    rhs_sel = rhs_f[finite]
    floor = max(float(np.nanmax(np.abs(rhs_sel))) * floor_fraction, 1.0e-30)
    denom = np.maximum(np.abs(rhs_sel), floor)
    return float(np.nanmean(np.abs(lhs_f[finite] - rhs_sel) / denom))


def _cetg_table_rows(df: pd.DataFrame) -> list[list[str]]:
    metrics = [
        ("W", _mean_rel(np.asarray(df["W_spectrax"], dtype=float), np.asarray(df["W_gx"], dtype=float))),
        (
            "Phi2",
            _mean_rel(np.asarray(df["Phi2_spectrax"], dtype=float), np.asarray(df["Phi2_gx"], dtype=float)),
        ),
        (
            "qflux",
            _mean_rel(np.asarray(df["qflux_spectrax"], dtype=float), np.asarray(df["qflux_gx"], dtype=float)),
        ),
        (
            "pflux",
            _mean_rel(np.asarray(df["pflux_spectrax"], dtype=float), np.asarray(df["pflux_gx"], dtype=float)),
        ),
    ]
    return [[name, f"{value:.2e}"] for name, value in metrics]


def _plot_imported_linear(ax: Axes, df: pd.DataFrame, title: str) -> None:
    ky = np.asarray(df["ky"], dtype=float)
    ax.plot(ky, np.asarray(df["mean_abs_omega"], dtype=float), marker="o", linewidth=2.0, label="abs ω")
    ax.plot(ky, np.asarray(df["mean_abs_gamma"], dtype=float), marker="s", linewidth=2.0, label="abs γ")
    ax.plot(ky, np.asarray(df["mean_rel_Wg"], dtype=float), marker="^", linewidth=2.0, linestyle="--", label="rel Wg")
    ax.plot(
        ky,
        np.asarray(df["mean_rel_Wphi"], dtype=float),
        marker="d",
        linewidth=2.0,
        linestyle="--",
        label="rel Wphi",
    )
    if "mean_rel_Wapar" in df.columns and np.any(np.asarray(df["mean_rel_Wapar"], dtype=float) > 0.0):
        ax.plot(
            ky,
            np.asarray(df["mean_rel_Wapar"], dtype=float),
            marker="x",
            linewidth=1.8,
            linestyle=":",
            label="rel Wapar",
        )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("ky")
    ax.set_yscale("log")
    ax.set_ylabel("error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper left")


def _plot_secondary(ax: Axes, df: pd.DataFrame, title: str) -> None:
    modes = [f"({row.ky:.2f},{row.kx:+.2f})" for row in df.itertuples(index=False)]
    x = np.arange(len(modes), dtype=float)
    ax.plot(x, np.asarray(df["gamma_gx"], dtype=float), marker="o", linewidth=2.0, label="γ GX")
    ax.plot(x, np.asarray(df["gamma_sp"], dtype=float), marker="s", linewidth=2.0, linestyle="--", label="γ S")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("gamma")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=25, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax_omega = cast(Axes, ax.twinx())
    ax_omega.bar(
        x,
        np.asarray(df["abs_omega"], dtype=float),
        width=0.55,
        color="0.75",
        alpha=0.5,
        label="|Δω|",
    )
    ax_omega.set_ylabel("|Δω|")
    handles_l, labels_l = ax.get_legend_handles_labels()
    handles_r, labels_r = ax_omega.get_legend_handles_labels()
    ax.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8, frameon=False, loc="upper left")


def _plot_cetg(ax: Axes, df: pd.DataFrame, title: str) -> None:
    t = np.asarray(df["t"], dtype=float)
    metric_specs = [
        ("W", "W_spectrax", "W_gx", "#1f77b4"),
        ("Phi2", "Phi2_spectrax", "Phi2_gx", "#ff7f0e"),
        ("qflux", "qflux_spectrax", "qflux_gx", "#2ca02c"),
    ]
    for label, sp_col, gx_col, color in metric_specs:
        sp = np.asarray(df[sp_col], dtype=float)
        gx = np.asarray(df[gx_col], dtype=float)
        finite = np.isfinite(sp) & np.isfinite(gx) & (sp > 0.0) & (gx > 0.0)
        if not np.any(finite):
            continue
        ax.plot(t[finite], gx[finite], color=color, linewidth=2.0, label=f"{label} GX")
        ax.plot(t[finite], sp[finite], color=color, linewidth=1.8, linestyle="--", label=f"{label} S")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t")
    ax.set_yscale("log")
    ax.set_ylabel("signal")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cyclone-kbm-panel",
        type=Path,
        default=STATIC / "gx_cyclone_kbm_panel.png",
        help="Existing Cyclone/KBM GX comparison panel.",
    )
    parser.add_argument(
        "--w7x-panel",
        type=Path,
        default=STATIC / "nonlinear_w7x_diag_compare_t200.png",
        help="Tracked nonlinear W7-X comparison figure.",
    )
    parser.add_argument(
        "--hsx-panel",
        type=Path,
        default=STATIC / "hsx_nonlinear_compare_t50_true.png",
        help="Tracked nonlinear HSX comparison figure.",
    )
    parser.add_argument(
        "--secondary-csv",
        type=Path,
        default=STATIC / "secondary_gx_out_compare.csv",
        help="Tracked secondary comparison CSV.",
    )
    parser.add_argument(
        "--w7x-linear-csv",
        type=Path,
        default=STATIC / "w7x_linear_t2_scan.csv",
        help="Tracked linear W7-X imported-geometry comparison CSV.",
    )
    parser.add_argument(
        "--hsx-linear-csv",
        type=Path,
        default=STATIC / "hsx_linear_t2_scan.csv",
        help="Tracked linear HSX imported-geometry comparison CSV.",
    )
    parser.add_argument(
        "--cetg-csv",
        type=Path,
        default=STATIC / "cetg_gx_compare.csv",
        help="Tracked legacy GX cETG comparison CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=STATIC / "gx_summary_panel.png",
        help="Output PNG path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cyclone_kbm = _resolve(args.cyclone_kbm_panel)
    w7x_panel = _resolve(args.w7x_panel)
    hsx_panel = _resolve(args.hsx_panel)
    secondary_csv = _resolve(args.secondary_csv)
    w7x_linear_csv = _resolve(args.w7x_linear_csv)
    hsx_linear_csv = _resolve(args.hsx_linear_csv)
    cetg_csv = _resolve(args.cetg_csv)
    out = _resolve(args.out)

    secondary = _load_secondary(secondary_csv)
    w7x_linear = _load_imported_linear(w7x_linear_csv)
    hsx_linear = _load_imported_linear(hsx_linear_csv)
    cetg = _load_cetg(cetg_csv)

    fig = plt.figure(figsize=(18, 20), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1.0, 0.8, 0.8])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(mpimg.imread(cyclone_kbm))
    ax0.set_title("Cyclone / KBM", fontsize=14, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(mpimg.imread(w7x_panel))
    ax1.set_title("W7-X Nonlinear VMEC", fontsize=14, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(mpimg.imread(hsx_panel))
    ax2.set_title("HSX Nonlinear VMEC", fontsize=14, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2, 0])
    _plot_imported_linear(ax3, w7x_linear, "W7-X Linear VMEC")

    ax4 = fig.add_subplot(gs[2, 1])
    _plot_imported_linear(ax4, hsx_linear, "HSX Linear VMEC")

    ax5 = fig.add_subplot(gs[3, 0])
    _plot_secondary(ax5, secondary, "Secondary Slab (GX kh01a.out.nc)")

    ax6 = fig.add_subplot(gs[3, 1])
    _plot_cetg(ax6, cetg, "cETG (legacy GX)")

    fig.suptitle("GX-Aligned Validation Summary", fontsize=18, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
