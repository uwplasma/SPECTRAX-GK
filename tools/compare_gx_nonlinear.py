#!/usr/bin/env python3
"""Compare GX nonlinear diagnostics against SPECTRAX-GK CSV output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def _read_diag_series(group, name: str) -> np.ndarray:
    var = group.variables[name][:]
    if var.ndim == 3:
        # time, species, ky
        return np.sum(var, axis=(1, 2))
    if var.ndim == 2:
        return np.sum(var, axis=1)
    if var.ndim == 1:
        return np.asarray(var)
    raise ValueError(f"Unexpected shape for {name}: {var.shape}")


def _load_gx(path: Path) -> dict[str, np.ndarray]:
    root = Dataset(path, "r")
    diag = root.groups["Diagnostics"]
    t = np.asarray(root.groups["Grids"].variables["time"][:], dtype=float)
    out = {
        "t": t,
        "Wg": _read_diag_series(diag, "Wg_kyst"),
        "Wphi": _read_diag_series(diag, "Wphi_kyst"),
        "Wapar": _read_diag_series(diag, "Wapar_kyst"),
        "heat": _read_diag_series(diag, "HeatFlux_kyst"),
        "pflux": _read_diag_series(diag, "ParticleFlux_kyst"),
    }
    root.close()
    return out


def _load_spectrax(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {
        "t": data[:, 0],
        "gamma": data[:, 1],
        "omega": data[:, 2],
        "Wg": data[:, 3],
        "Wphi": data[:, 4],
        "Wapar": data[:, 5],
        "energy": data[:, 6],
        "heat": data[:, 7],
        "pflux": data[:, 8],
    }


def _interp(target_t: np.ndarray, source_t: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    if source_t.size == 0:
        return np.zeros_like(target_t)
    return np.interp(target_t, source_t, source_y)


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(b), 1.0e-12)
    return float(np.nanmean(np.abs(a - b) / denom))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx", type=Path, required=True, help="GX .out.nc file")
    parser.add_argument("--spectrax", type=Path, required=True, help="SPECTRAX diagnostics CSV")
    parser.add_argument("--out", type=Path, default=Path("docs/_static/nonlinear_cyclone_compare.png"))
    args = parser.parse_args()

    gx = _load_gx(args.gx)
    sp = _load_spectrax(args.spectrax)
    t = sp["t"]

    gx_interp = {key: _interp(t, gx["t"], gx[key]) for key in ["Wg", "Wphi", "Wapar", "heat", "pflux"]}

    print(f"Wg rel error: {_relative_error(sp['Wg'], gx_interp['Wg']):.3e}")
    print(f"Wphi rel error: {_relative_error(sp['Wphi'], gx_interp['Wphi']):.3e}")
    print(f"Wapar rel error: {_relative_error(sp['Wapar'], gx_interp['Wapar']):.3e}")
    print(f"Heat flux rel error: {_relative_error(sp['heat'], gx_interp['heat']):.3e}")
    print(f"Particle flux rel error: {_relative_error(sp['pflux'], gx_interp['pflux']):.3e}")

    fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
    ax[0].plot(t, sp["Wphi"], label="SPECTRAX Wphi")
    ax[0].plot(t, gx_interp["Wphi"], label="GX Wphi", linestyle="--")
    ax[0].set_ylabel("Wphi")
    ax[0].legend(frameon=False)
    ax[1].plot(t, sp["heat"], label="SPECTRAX Q")
    ax[1].plot(t, gx_interp["heat"], label="GX Q", linestyle="--")
    ax[1].set_ylabel("Heat flux")
    ax[1].set_xlabel("t")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
