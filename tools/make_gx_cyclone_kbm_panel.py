#!/usr/bin/env python3
"""Build the tracked tokamak GX-vs-SPECTRAX publication panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

try:
    from tools.make_gx_summary_panel import STATIC, _autocrop_image, _resolve
except ModuleNotFoundError:  # Allows PYTHONPATH=tools execution in tests.
    from make_gx_summary_panel import STATIC, _autocrop_image, _resolve


REQUIRED_LINEAR_COLUMNS = {"ky", "gamma", "omega", "gamma_gx", "omega_gx"}


def _load_linear_mismatch(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_LINEAR_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df.sort_values("ky").reset_index(drop=True)


def _mean_rel(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_f = np.asarray(lhs, dtype=float)
    rhs_f = np.asarray(rhs, dtype=float)
    finite = np.isfinite(lhs_f) & np.isfinite(rhs_f)
    if not np.any(finite):
        return float("nan")
    rhs_sel = rhs_f[finite]
    floor = max(float(np.nanmax(np.abs(rhs_sel))) * 1.0e-12, 1.0e-30)
    denom = np.maximum(np.abs(rhs_sel), floor)
    return float(np.nanmean(np.abs(lhs_f[finite] - rhs_sel) / denom))


def _plot_linear_metric(ax: Axes, df: pd.DataFrame, *, metric: str, title: str) -> None:
    ky = np.asarray(df["ky"], dtype=float)
    gx = np.asarray(df[f"{metric}_gx"], dtype=float)
    sp = np.asarray(df[metric], dtype=float)
    ax.plot(ky, gx, marker="o", linewidth=2.2, color="#111111", label="GX")
    ax.plot(ky, sp, marker="s", linewidth=2.2, color="#d1495b", linestyle="--", label="SPECTRAX")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    rel = _mean_rel(sp, gx)
    ax.text(
        0.03,
        0.97,
        f"mean rel err = {rel:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )
    ax.legend(frameon=False, fontsize=8, loc="best")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cyclone-linear",
        type=Path,
        default=STATIC / "cyclone_miller_linear_mismatch.csv",
        help="Tracked clean-mainline Cyclone Miller linear mismatch CSV.",
    )
    parser.add_argument(
        "--kbm-linear",
        type=Path,
        default=STATIC / "kbm_gx_mismatch.csv",
        help="Tracked KBM linear mismatch CSV.",
    )
    parser.add_argument(
        "--cyclone-nonlinear-panel",
        type=Path,
        default=STATIC / "nonlinear_cyclone_miller_diag_compare_t122.png",
        help="Tracked clean-mainline Cyclone Miller nonlinear comparison figure.",
    )
    parser.add_argument(
        "--kbm-nonlinear-panel",
        type=Path,
        default=STATIC / "nonlinear_kbm_diag_compare_t400_stats.png",
        help="Tracked KBM nonlinear comparison figure.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=STATIC / "gx_cyclone_kbm_panel.png",
        help="Output tokamak panel path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cyclone_linear = _load_linear_mismatch(_resolve(args.cyclone_linear))
    kbm_linear = _load_linear_mismatch(_resolve(args.kbm_linear))
    cyclone_img = _autocrop_image(mpimg.imread(_resolve(args.cyclone_nonlinear_panel)), pad_pixels=8)
    kbm_img = _autocrop_image(mpimg.imread(_resolve(args.kbm_nonlinear_panel)), pad_pixels=8)
    out = _resolve(args.out)

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[0.95, 1.25])

    _plot_linear_metric(fig.add_subplot(gs[0, 0]), cyclone_linear, metric="gamma", title="Cyclone (Miller) Linear γ")
    _plot_linear_metric(fig.add_subplot(gs[0, 1]), cyclone_linear, metric="omega", title="Cyclone (Miller) Linear ω")
    _plot_linear_metric(fig.add_subplot(gs[0, 2]), kbm_linear, metric="gamma", title="KBM Linear γ")
    _plot_linear_metric(fig.add_subplot(gs[0, 3]), kbm_linear, metric="omega", title="KBM Linear ω")

    ax_c = fig.add_subplot(gs[1, :2])
    ax_c.imshow(cyclone_img)
    ax_c.set_title("Cyclone (Miller) Nonlinear", fontsize=14, fontweight="bold")
    ax_c.axis("off")

    ax_k = fig.add_subplot(gs[1, 2:])
    ax_k.imshow(kbm_img)
    ax_k.set_title("KBM Nonlinear", fontsize=14, fontweight="bold")
    ax_k.axis("off")

    fig.suptitle("Tokamak GX Validation: Cyclone and KBM", fontsize=17, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
