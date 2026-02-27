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
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    denom = np.maximum(np.abs(b[mask]), 1.0e-12)
    return float(np.nanmean(np.abs(a[mask] - b[mask]) / denom))


def _window_mean(t: np.ndarray, y: np.ndarray, frac: float) -> float:
    if t.size == 0:
        return float("nan")
    t_min = t.min() + (1.0 - frac) * (t.max() - t.min())
    mask = (t >= t_min) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(y[mask]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx", type=Path, required=True, help="GX .out.nc file")
    parser.add_argument("--spectrax", type=Path, required=True, help="SPECTRAX diagnostics CSV")
    parser.add_argument("--out", type=Path, default=Path("docs/_static/nonlinear_cyclone_compare.png"))
    parser.add_argument(
        "--avg-fraction",
        type=float,
        default=0.2,
        help="Fraction of late-time window for averaged metrics.",
    )
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
    if 0.0 < args.avg_fraction <= 1.0:
        wg_sp = _window_mean(t, sp["Wg"], args.avg_fraction)
        wg_gx = _window_mean(t, gx_interp["Wg"], args.avg_fraction)
        wphi_sp = _window_mean(t, sp["Wphi"], args.avg_fraction)
        wphi_gx = _window_mean(t, gx_interp["Wphi"], args.avg_fraction)
        q_sp = _window_mean(t, sp["heat"], args.avg_fraction)
        q_gx = _window_mean(t, gx_interp["heat"], args.avg_fraction)
        print(f"Late-time mean Wg: SPECTRAX={wg_sp:.3e}, GX={wg_gx:.3e}")
        print(f"Late-time mean Wphi: SPECTRAX={wphi_sp:.3e}, GX={wphi_gx:.3e}")
        print(f"Late-time mean Q: SPECTRAX={q_sp:.3e}, GX={q_gx:.3e}")

    fig, ax = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
    mask = np.isfinite(sp["Wphi"]) & np.isfinite(gx_interp["Wphi"])
    ax[0].plot(t[mask], sp["Wphi"][mask], label="SPECTRAX Wphi")
    ax[0].plot(t[mask], gx_interp["Wphi"][mask], label="GX Wphi", linestyle="--")
    ax[0].set_ylabel("Wphi")
    ax[0].legend(frameon=False)
    mask_h = np.isfinite(sp["heat"]) & np.isfinite(gx_interp["heat"])
    ax[1].plot(t[mask_h], sp["heat"][mask_h], label="SPECTRAX Q")
    ax[1].plot(t[mask_h], gx_interp["heat"][mask_h], label="GX Q", linestyle="--")
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
