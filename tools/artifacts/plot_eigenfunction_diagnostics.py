#!/usr/bin/env python3
"""Plot eigenfunction diagnostics used by benchmark and manuscript artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.artifacts.plotting import (  # noqa: E402
    eigenfunction_overlap_summary_figure,
    eigenfunction_reference_overlay_figure,
)
from spectraxgk.diagnostics.modes import load_eigenfunction_reference_bundle  # noqa: E402

DEFAULT_OVERLAP_CSV = (
    ROOT / "docs" / "_static" / "comparison" / "kbm_reference_candidates.csv"
)
DEFAULT_OVERLAP_OUT = ROOT / "docs" / "_static" / "kbm_eigenfunction_overlap_summary.png"
DEFAULT_REFERENCE_OVERLAY_OUT = (
    ROOT / "docs" / "_static" / "eigenfunction_reference_overlay.png"
)


def load_spectrax_eigenfunction_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a complex eigenfunction from the documented CSV artifact schema."""

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if (
        "z" not in data.dtype.names
        or "eigen_real" not in data.dtype.names
        or "eigen_imag" not in data.dtype.names
    ):
        raise ValueError(
            "eigenfunction CSV must contain z,eigen_real,eigen_imag columns"
        )
    theta = np.asarray(data["z"], dtype=float)
    mode = np.asarray(data["eigen_real"], dtype=float) + 1j * np.asarray(
        data["eigen_imag"], dtype=float
    )
    return theta, mode


def plot_reference_overlay(
    *,
    reference: Path,
    spectrax: Path,
    out: Path = DEFAULT_REFERENCE_OVERLAY_OUT,
    title: str = "Eigenfunction overlay against frozen reference",
) -> dict[str, str]:
    """Write a raw phase-aligned eigenfunction overlay figure."""

    bundle = load_eigenfunction_reference_bundle(reference)
    theta, mode = load_spectrax_eigenfunction_csv(spectrax)
    fig, _axes = eigenfunction_reference_overlay_figure(
        theta,
        mode,
        bundle.theta,
        bundle.mode,
        title=title,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    if out.suffix.lower() != ".pdf":
        fig.savefig(pdf, bbox_inches="tight")
    return {"png_or_pdf": str(out), "pdf": str(pdf)}


def plot_overlap_summary(
    *,
    csv: Path = DEFAULT_OVERLAP_CSV,
    out: Path = DEFAULT_OVERLAP_OUT,
    title: str = "KBM eigenfunction overlap against reference",
) -> dict[str, str]:
    """Write an overlap/relative-L2 summary figure from a candidate table."""

    df = pd.read_csv(csv)
    if "selected" in df.columns:
        df = df[df["selected"].astype(bool)]
    needed = {"ky", "eig_overlap_gx", "eig_rel_l2"}
    missing = needed.difference(df.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")
    df = df[np.isfinite(df["eig_overlap_gx"]) & np.isfinite(df["eig_rel_l2"])]
    df = df.sort_values("ky")
    fig, _axes = eigenfunction_overlap_summary_figure(
        np.asarray(df["ky"], dtype=float),
        np.asarray(df["eig_overlap_gx"], dtype=float),
        np.asarray(df["eig_rel_l2"], dtype=float),
        title=str(title),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    return {"png": str(out), "pdf": str(pdf)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    overlay = subcommands.add_parser(
        "reference-overlay",
        help="Plot a raw phase-aligned eigenfunction overlay against a reference bundle.",
    )
    overlay.add_argument("reference", type=Path, help="Frozen reference bundle (.npz).")
    overlay.add_argument("spectrax", type=Path, help="Eigenfunction CSV artifact.")
    overlay.add_argument("--out", type=Path, default=DEFAULT_REFERENCE_OVERLAY_OUT)
    overlay.add_argument(
        "--title",
        default="Eigenfunction overlay against frozen reference",
        help="Figure title.",
    )

    summary = subcommands.add_parser(
        "overlap-summary",
        help="Plot overlap and relative-L2 summary from a candidate table.",
    )
    summary.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_OVERLAP_CSV,
        help="Input CSV with ky/eig_overlap_gx/eig_rel_l2 columns.",
    )
    summary.add_argument("--out", type=Path, default=DEFAULT_OVERLAP_OUT)
    summary.add_argument(
        "--title",
        default="KBM eigenfunction overlap against reference",
        help="Figure title.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "reference-overlay":
        plot_reference_overlay(
            reference=args.reference,
            spectrax=args.spectrax,
            out=args.out,
            title=args.title,
        )
        return 0
    if args.command == "overlap-summary":
        plot_overlap_summary(csv=args.csv, out=args.out, title=args.title)
        return 0
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
