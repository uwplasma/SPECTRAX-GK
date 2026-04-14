#!/usr/bin/env python3
"""Build the publication-facing GX validation panel for the main parity cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

try:
    from tools.make_gx_summary_panel import (
        STATIC,
        _autocrop_image,
        _load_imported_linear,
        _plot_imported_linear,
        _resolve,
    )
except ModuleNotFoundError:  # Allows PYTHONPATH=tools execution in tests.
    from make_gx_summary_panel import (
        STATIC,
        _autocrop_image,
        _load_imported_linear,
        _plot_imported_linear,
        _resolve,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cyclone-kbm-panel",
        type=Path,
        default=STATIC / "gx_cyclone_kbm_panel.png",
        help="Tracked Cyclone/KBM tokamak panel.",
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
        "--etg-panel",
        type=Path,
        default=STATIC / "etg_fullgk_pilot_compare_dt1e4_gaussian_match.png",
        help="Tracked full-GK ETG nonlinear pilot comparison figure.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=STATIC / "gx_publication_panel.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--pdf-out",
        type=Path,
        default=STATIC / "gx_publication_panel.pdf",
        help="Optional PDF output path for publication workflows.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cyclone_kbm = _resolve(args.cyclone_kbm_panel)
    w7x_panel = _resolve(args.w7x_panel)
    hsx_panel = _resolve(args.hsx_panel)
    w7x_linear_csv = _resolve(args.w7x_linear_csv)
    hsx_linear_csv = _resolve(args.hsx_linear_csv)
    etg_panel = _resolve(args.etg_panel)
    out = _resolve(args.out)
    pdf_out = _resolve(args.pdf_out)

    w7x_linear = _load_imported_linear(w7x_linear_csv)
    hsx_linear = _load_imported_linear(hsx_linear_csv)

    cyclone_img = _autocrop_image(mpimg.imread(cyclone_kbm), pad_pixels=4)
    w7x_img = _autocrop_image(mpimg.imread(w7x_panel), pad_pixels=4)
    hsx_img = _autocrop_image(mpimg.imread(hsx_panel), pad_pixels=4)
    etg_img = _autocrop_image(mpimg.imread(etg_panel), pad_pixels=4)

    fig = plt.figure(figsize=(20.5, 23.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.55, 1.15, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(cyclone_img)
    ax0.set_title("Tokamak Benchmarks: Cyclone and KBM", fontsize=15, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(w7x_img)
    ax1.set_title("W7-X Nonlinear VMEC", fontsize=14, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(hsx_img)
    ax2.set_title("HSX Nonlinear VMEC", fontsize=14, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2, 0])
    _plot_imported_linear(ax3, w7x_linear, "W7-X Linear VMEC")

    ax4 = fig.add_subplot(gs[2, 1])
    _plot_imported_linear(ax4, hsx_linear, "HSX Linear VMEC")

    ax5 = fig.add_subplot(gs[3, :])
    ax5.imshow(etg_img)
    ax5.set_title("ETG Nonlinear Short-Window Pilot", fontsize=14, fontweight="bold")
    ax5.axis("off")

    fig.suptitle("GX-Aligned Validation: Publication Panel", fontsize=18, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, facecolor="white")
    fig.savefig(pdf_out, facecolor="white")
    print(f"saved {out}")
    print(f"saved {pdf_out}")


if __name__ == "__main__":
    main()
