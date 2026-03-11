#!/usr/bin/env python3
"""Build the tracked GX-vs-SPECTRAX README summary panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
    out = _resolve(args.out)

    secondary = _load_secondary(secondary_csv)

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(mpimg.imread(cyclone_kbm))
    ax0.set_title("Cyclone / KBM", fontsize=14, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(mpimg.imread(w7x_panel))
    ax1.set_title("W7-X Nonlinear VMEC", fontsize=14, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(mpimg.imread(hsx_panel))
    ax2.set_title("HSX Nonlinear VMEC", fontsize=14, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.set_title("Secondary Slab (GX kh01a.out.nc)", fontsize=14, fontweight="bold")
    table = ax3.table(
        cellText=_secondary_table_rows(secondary),
        colLabels=["(ky,kx)", "γ GX", "γ S", "rel γ", "ω GX", "ω S", "|Δω|"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    ax3.text(
        0.5,
        0.03,
        "Gamma now uses the longest leading finite selected-mode window.\n"
        "Rows come from a real GX kh01a.out.nc reference.\n"
        "Tiny secondary omega remains under audit; absolute error is more informative than relative error here.",
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
