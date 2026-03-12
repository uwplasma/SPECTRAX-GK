#!/usr/bin/env python3
"""Build the tracked GX-vs-SPECTRAX README summary panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
    ax3.axis("off")
    ax3.set_title("W7-X Linear VMEC", fontsize=14, fontweight="bold")
    table = ax3.table(
        cellText=_linear_table_rows(w7x_linear),
        colLabels=["ky", "abs ω", "abs γ", "rel Wg", "rel Wphi"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    ax4.set_title("HSX Linear VMEC", fontsize=14, fontweight="bold")
    table = ax4.table(
        cellText=_linear_table_rows(hsx_linear),
        colLabels=["ky", "abs ω", "abs γ", "rel Wg", "rel Wphi"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    ax5 = fig.add_subplot(gs[3, 0])
    ax5.axis("off")
    ax5.set_title("Secondary Slab (GX kh01a.out.nc)", fontsize=14, fontweight="bold")
    table = ax5.table(
        cellText=_secondary_table_rows(secondary),
        colLabels=["(ky,kx)", "γ GX", "γ S", "rel γ", "ω GX", "ω S", "|Δω|"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    ax5.text(
        0.5,
        0.03,
        "Gamma now uses the longest leading finite selected-mode window.\n"
        "Rows come from a real GX kh01a.out.nc reference.\n"
        "Tiny secondary omega remains under audit; absolute error is more informative than relative error here.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis("off")
    ax6.set_title("cETG (legacy GX)", fontsize=14, fontweight="bold")
    table = ax6.table(
        cellText=_cetg_table_rows(cetg),
        colLabels=["metric", "mean rel err"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    ax6.text(
        0.5,
        0.06,
        "Short-horizon cETG comparison uses the legacy grouped GX output lane.\n"
        "This is the honest reduced-model reference until a native panel plot is added.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.suptitle("GX-Aligned Validation Summary", fontsize=18, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
