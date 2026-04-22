"""Plot a raw phase-aligned eigenfunction overlay against a frozen reference bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spectraxgk.benchmarking import load_eigenfunction_reference_bundle
from spectraxgk.plotting import eigenfunction_reference_overlay_figure

ROOT = Path(__file__).resolve().parents[1]


def _load_spectrax_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if "z" not in data.dtype.names or "eigen_real" not in data.dtype.names or "eigen_imag" not in data.dtype.names:
        raise ValueError("eigenfunction CSV must contain z,eigen_real,eigen_imag columns")
    theta = np.asarray(data["z"], dtype=float)
    mode = np.asarray(data["eigen_real"], dtype=float) + 1j * np.asarray(data["eigen_imag"], dtype=float)
    return theta, mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Frozen reference eigenfunction bundle (.npz).")
    parser.add_argument("spectrax", type=Path, help="SPECTRAX-GK eigenfunction CSV artifact.")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "eigenfunction_reference_overlay.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--title",
        default="Eigenfunction overlay against frozen reference",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_eigenfunction_reference_bundle(args.reference)
    theta, mode = _load_spectrax_csv(args.spectrax)
    fig, _axes = eigenfunction_reference_overlay_figure(
        theta,
        mode,
        bundle.theta,
        bundle.mode,
        title=args.title,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    if args.out.suffix.lower() != ".pdf":
        fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
