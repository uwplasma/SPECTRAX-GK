#!/usr/bin/env python3
"""Freeze a GX ``.big.nc`` eigenfunction into a compact reference bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from spectraxgk.benchmarking import save_eigenfunction_reference_bundle


def _normalize_mode(theta: np.ndarray, mode: np.ndarray) -> np.ndarray:
    finite = np.isfinite(mode)
    if not np.any(finite):
        return np.zeros_like(mode)
    idx0 = int(np.argmin(np.abs(theta)))
    ref = mode[idx0]
    if not np.isfinite(ref) or abs(ref) < 1.0e-14:
        idx = int(np.nanargmax(np.abs(np.where(finite, mode, 0.0))))
        ref = mode[idx]
    if not np.isfinite(ref) or abs(ref) < 1.0e-14:
        scale = float(np.nanmax(np.abs(np.where(finite, mode, 0.0))))
        return mode if scale <= 0.0 else mode / scale
    return mode / ref


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gx_big", type=Path, help="GX .big.nc file with Diagnostics/Phi.")
    parser.add_argument("--ky", type=float, required=True, help="Target ky value.")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz bundle path.")
    parser.add_argument("--case", required=True, help="Case label stored in the bundle metadata.")
    parser.add_argument("--source", default="GX", help="Reference code label stored in the bundle.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Dataset(args.gx_big, "r")
    theta = np.asarray(root.groups["Grids"].variables["theta"][:], dtype=float)
    ky = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)
    phi = root.groups["Diagnostics"].variables["Phi"]
    ky_idx = int(np.argmin(np.abs(ky - float(args.ky))))
    raw = np.asarray(phi[-1, ky_idx, 0, :, :], dtype=float)
    root.close()
    mode = _normalize_mode(theta, raw[:, 0] + 1j * raw[:, 1])
    out = save_eigenfunction_reference_bundle(
        args.out,
        theta=theta,
        mode=mode,
        source=str(args.source),
        case=str(args.case),
        metadata={"ky": float(ky[ky_idx]), "gx_big": str(args.gx_big)},
    )
    print(out)


if __name__ == "__main__":
    main()
