#!/usr/bin/env python3
"""Build the compact benchmark atlas used by the README and docs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"

REF_COLOR = "#111827"
SP_COLOR = "#1d4ed8"
GRID_COLOR = "#cbd5e1"
TITLE_COLOR = "#0f172a"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _autocrop_image(
    image: np.ndarray,
    *,
    white_threshold: float = 0.985,
    pad_pixels: int = 14,
) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return arr
    rgb = arr[..., :3]
    non_white = np.any(rgb < white_threshold, axis=2)
    rows = np.where(np.any(non_white, axis=1))[0]
    cols = np.where(np.any(non_white, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return arr
    r0 = max(int(rows[0]) - pad_pixels, 0)
    r1 = min(int(rows[-1]) + pad_pixels + 1, arr.shape[0])
    c0 = max(int(cols[0]) - pad_pixels, 0)
    c1 = min(int(cols[-1]) + pad_pixels + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def _load_image(path: Path, *, pad_pixels: int = 14) -> np.ndarray:
    return _autocrop_image(mpimg.imread(path), pad_pixels=pad_pixels)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"saved {path.with_suffix('.pdf')}")


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.45, color=GRID_COLOR, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155", labelsize=9)


def _plot_overlay_case(
    ax_gamma: plt.Axes,
    ax_omega: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    xcol: str,
    gamma_ref: str,
    gamma_sp: str,
    omega_ref: str,
    omega_sp: str,
    x_label: str,
) -> None:
    x = np.asarray(df[xcol], dtype=float)
    gamma_ref_vals = np.asarray(df[gamma_ref], dtype=float)
    gamma_sp_vals = np.asarray(df[gamma_sp], dtype=float)
    omega_ref_vals = np.asarray(df[omega_ref], dtype=float)
    omega_sp_vals = np.asarray(df[omega_sp], dtype=float)

    _style_axis(ax_gamma)
    _style_axis(ax_omega)

    ax_gamma.plot(x, gamma_ref_vals, color=REF_COLOR, marker="o", linewidth=2.0, label="Reference")
    ax_gamma.plot(x, gamma_sp_vals, color=SP_COLOR, marker="s", linewidth=2.0, label="SPECTRAX-GK")
    ax_gamma.set_title(title, fontsize=12, color=TITLE_COLOR, fontweight="bold")
    ax_gamma.set_ylabel(r"Growth rate $\gamma$")

    ax_omega.plot(x, omega_ref_vals, color=REF_COLOR, marker="o", linewidth=2.0, label="Reference")
    ax_omega.plot(x, omega_sp_vals, color=SP_COLOR, marker="s", linewidth=2.0, label="SPECTRAX-GK")
    ax_omega.set_ylabel(r"Frequency $\omega$")
    ax_omega.set_xlabel(x_label)

    if x.size == 1:
        for ax in (ax_gamma, ax_omega):
            center = float(x[0])
            width = max(abs(center) * 0.4, 0.02)
            ax.set_xlim(center - width, center + width)

    denom_gamma = np.maximum(np.abs(gamma_ref_vals), 1.0e-12)
    denom_omega = np.maximum(np.abs(omega_ref_vals), 1.0e-12)
    max_rel_gamma = float(np.max(np.abs(gamma_sp_vals - gamma_ref_vals) / denom_gamma))
    max_rel_omega = float(np.max(np.abs(omega_sp_vals - omega_ref_vals) / denom_omega))
    ax_gamma.text(
        0.03,
        0.97,
        f"max rel γ = {max_rel_gamma:.2%}\nmax rel ω = {max_rel_omega:.2%}",
        transform=ax_gamma.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.25"},
    )


def _plot_kaw_case(ax_gamma: plt.Axes, ax_omega: plt.Axes, df: pd.DataFrame) -> None:
    _plot_overlay_case(
        ax_gamma,
        ax_omega,
        df,
        title="KAW Exact Diagnostic Window",
        xcol="ky",
        gamma_ref="gamma_ref",
        gamma_sp="gamma_spectrax",
        omega_ref="omega_ref",
        omega_sp="omega_spectrax",
        x_label=r"$k_y \rho_i$",
    )
    row = df.iloc[0]
    ax_omega.text(
        0.03,
        0.12,
        (
            "window errors\n"
            f"free energy = {row['rel_free_energy']:.2e}\n"
            f"electrostatic field energy = {row['rel_electrostatic_energy']:.2e}\n"
            f"magnetic field energy = {row['rel_magnetic_energy']:.2e}"
        ),
        transform=ax_omega.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.25"},
    )


def _plot_exact_growth_case(
    ax_gamma: plt.Axes,
    ax_omega: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    gamma_ref: str,
    gamma_sp: str,
    omega_ref: str,
    omega_sp: str,
    rel_gamma: str | None = None,
    rel_omega: str | None = None,
    note_title: str = "window errors",
) -> None:
    _plot_overlay_case(
        ax_gamma,
        ax_omega,
        df,
        title=title,
        xcol="ky",
        gamma_ref=gamma_ref,
        gamma_sp=gamma_sp,
        omega_ref=omega_ref,
        omega_sp=omega_sp,
        x_label=r"$k_y \rho_i$",
    )
    row = df.iloc[0]
    lines = [note_title]
    if rel_gamma is not None and rel_gamma in row:
        lines.append(f"rel γ = {row[rel_gamma]:.2e}")
    if rel_omega is not None and rel_omega in row:
        lines.append(f"rel ω = {row[rel_omega]:.2e}")
    ax_omega.text(
        0.03,
        0.12,
        "\n".join(lines),
        transform=ax_omega.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.25"},
    )


def _image_tile(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=12, color=TITLE_COLOR, fontweight="bold")
    ax.axis("off")


def _build_imported_linear_panel(path: Path) -> None:
    w7x = pd.read_csv(STATIC / "w7x_linear_t2_scan.csv").sort_values("ky")
    hsx = pd.read_csv(STATIC / "hsx_linear_t2_scan.csv").sort_values("ky")
    miller = pd.read_csv(STATIC / "cyclone_miller_linear_mismatch.csv").sort_values("ky")
    kaw = pd.read_csv(STATIC / "kaw_exact_growth_dump.csv").sort_values("ky")

    fig, axes = plt.subplots(2, 4, figsize=(18, 7), constrained_layout=True)

    _plot_overlay_case(
        axes[0, 0],
        axes[1, 0],
        w7x,
        title="W7-X VMEC",
        xcol="ky",
        gamma_ref="gamma_ref_last",
        gamma_sp="gamma_last",
        omega_ref="omega_ref_last",
        omega_sp="omega_last",
        x_label=r"$k_y \rho_i$",
    )
    _plot_overlay_case(
        axes[0, 1],
        axes[1, 1],
        hsx,
        title="HSX VMEC",
        xcol="ky",
        gamma_ref="gamma_ref_last",
        gamma_sp="gamma_last",
        omega_ref="omega_ref_last",
        omega_sp="omega_last",
        x_label=r"$k_y \rho_i$",
    )
    _plot_overlay_case(
        axes[0, 2],
        axes[1, 2],
        miller,
        title="Cyclone Miller",
        xcol="ky",
        gamma_ref="gamma_gx",
        gamma_sp="gamma",
        omega_ref="omega_gx",
        omega_sp="omega",
        x_label=r"$k_y \rho_i$",
    )
    _plot_kaw_case(axes[0, 3], axes[1, 3], kaw)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Imported Geometry and Exact-Diagnostic Linear Benchmarks", fontsize=16, fontweight="bold")
    _save(fig, path)


def _build_extended_linear_panel(path: Path) -> None:
    kinetic = pd.read_csv(STATIC / "kinetic_mismatch_table.csv").sort_values("ky")
    tem = pd.read_csv(STATIC / "tem_mismatch_table.csv").sort_values("ky")
    miller = pd.read_csv(STATIC / "kbm_miller_exact_growth_dump.csv").sort_values("ky")

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.2), constrained_layout=True)
    _plot_overlay_case(
        axes[0, 0],
        axes[1, 0],
        kinetic,
        title="Cyclone Kinetic Electrons",
        xcol="ky",
        gamma_ref="gamma_ref",
        gamma_sp="gamma_spectrax",
        omega_ref="omega_ref",
        omega_sp="omega_spectrax",
        x_label=r"$k_y \rho_i$",
    )
    _plot_overlay_case(
        axes[0, 1],
        axes[1, 1],
        tem,
        title="TEM Stress Lane",
        xcol="ky",
        gamma_ref="gamma_ref",
        gamma_sp="gamma_spectrax",
        omega_ref="omega_ref",
        omega_sp="omega_spectrax",
        x_label=r"$k_y \rho_i$",
    )
    _plot_exact_growth_case(
        axes[0, 2],
        axes[1, 2],
        miller,
        title="KBM Miller Late Growth Window",
        gamma_ref="gamma_gx_dump",
        gamma_sp="gamma_sp_dump",
        omega_ref="omega_gx_dump",
        omega_sp="omega_sp_dump",
        rel_gamma="rel_gamma_sp_vs_gx_dump",
        rel_omega="rel_omega_sp_vs_gx_dump",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Extended Linear Stress Matrix", fontsize=16, fontweight="bold")
    _save(fig, path)


def _build_core_linear_atlas(path: Path, imported_panel_path: Path) -> None:
    cyclone = _load_image(STATIC / "cyclone_comparison.png")
    etg = _load_image(STATIC / "etg_gs2_stella_comparison.png")
    kbm = _load_image(STATIC / "kbm_gs2_stella_comparison.png")
    imported = _load_image(imported_panel_path, pad_pixels=6)

    fig, axes = plt.subplots(2, 2, figsize=(18, 13), constrained_layout=True)
    _image_tile(axes[0, 0], cyclone, "Cyclone ITG Cross-Code Scan")
    _image_tile(axes[0, 1], etg, "ETG Cross-Code Scan")
    _image_tile(axes[1, 0], kbm, "KBM Cross-Code Scan")
    _image_tile(axes[1, 1], imported, "Imported Geometry and Exact-Diagnostic Scans")
    fig.suptitle("Core Linear Benchmark Atlas", fontsize=18, fontweight="bold")
    _save(fig, path)


def _build_core_nonlinear_atlas(path: Path) -> None:
    cyclone = _load_image(STATIC / "nonlinear_cyclone_miller_diag_compare_t122.png", pad_pixels=10)
    kbm = _load_image(STATIC / "nonlinear_kbm_diag_compare_t400_stats.png", pad_pixels=10)
    w7x = _load_image(STATIC / "nonlinear_w7x_diag_compare_t200.png", pad_pixels=10)
    hsx = _load_image(STATIC / "hsx_nonlinear_compare_t50_true.png", pad_pixels=10)

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    _image_tile(axes[0, 0], cyclone, "Cyclone Nonlinear Time Trace (Miller Geometry)")
    _image_tile(axes[0, 1], kbm, "KBM Nonlinear Time Trace")
    _image_tile(axes[1, 0], w7x, "W7-X Nonlinear Time Trace")
    _image_tile(axes[1, 1], hsx, "HSX Nonlinear Time Trace")
    fig.suptitle("Core Nonlinear Benchmark Atlas", fontsize=18, fontweight="bold")
    _save(fig, path)


def _build_readme_panel(path: Path, core_linear_path: Path, core_nonlinear_path: Path) -> None:
    linear = _load_image(core_linear_path, pad_pixels=6)
    nonlinear = _load_image(core_nonlinear_path, pad_pixels=6)

    fig, axes = plt.subplots(2, 1, figsize=(18, 22), constrained_layout=True)
    _image_tile(axes[0], linear, "Linear Benchmarks")
    _image_tile(axes[1], nonlinear, "Nonlinear Benchmarks")
    fig.suptitle("SPECTRAX-GK Benchmark Atlas", fontsize=20, fontweight="bold")
    _save(fig, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imported-linear-out",
        type=Path,
        default=STATIC / "benchmark_imported_linear_panel.png",
        help="Output path for imported-geometry and exact-diagnostic linear scans.",
    )
    parser.add_argument(
        "--extended-linear-out",
        type=Path,
        default=STATIC / "benchmark_extended_linear_panel.png",
        help="Output path for extended linear stress scans.",
    )
    parser.add_argument(
        "--core-linear-out",
        type=Path,
        default=STATIC / "benchmark_core_linear_atlas.png",
        help="Output path for the core linear atlas.",
    )
    parser.add_argument(
        "--core-nonlinear-out",
        type=Path,
        default=STATIC / "benchmark_core_nonlinear_atlas.png",
        help="Output path for the core nonlinear atlas.",
    )
    parser.add_argument(
        "--readme-out",
        type=Path,
        default=STATIC / "benchmark_readme_panel.png",
        help="Output path for the README benchmark panel.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    imported_linear_out = _resolve(args.imported_linear_out)
    extended_linear_out = _resolve(args.extended_linear_out)
    core_linear_out = _resolve(args.core_linear_out)
    core_nonlinear_out = _resolve(args.core_nonlinear_out)
    readme_out = _resolve(args.readme_out)

    _build_imported_linear_panel(imported_linear_out)
    _build_extended_linear_panel(extended_linear_out)
    _build_core_linear_atlas(core_linear_out, imported_linear_out)
    _build_core_nonlinear_atlas(core_nonlinear_out)
    _build_readme_panel(readme_out, core_linear_out, core_nonlinear_out)


if __name__ == "__main__":
    main()
