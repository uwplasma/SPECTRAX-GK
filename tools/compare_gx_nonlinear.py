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


def _load_gx(path: Path, *, ky_target: float = 0.3) -> dict[str, np.ndarray]:
    root = Dataset(path, "r")
    diag = root.groups["Diagnostics"]
    grids = root.groups["Grids"]
    t = np.asarray(grids.variables["time"][:], dtype=float)
    out = {
        "t": t,
        "Wg": _read_diag_series(diag, "Wg_kyst"),
        "Wphi": _read_diag_series(diag, "Wphi_kyst"),
        "Wapar": _read_diag_series(diag, "Wapar_kyst"),
        "heat": _read_diag_series(diag, "HeatFlux_kyst"),
        "pflux": _read_diag_series(diag, "ParticleFlux_kyst"),
    }
    if "omega_kxkyt" in diag.variables and "ky" in grids.variables and "kx" in grids.variables:
        ky = np.asarray(grids.variables["ky"][:], dtype=float)
        kx = np.asarray(grids.variables["kx"][:], dtype=float)
        iky = int(np.argmin(np.abs(ky - float(ky_target))))
        ikx = int(np.argmin(np.abs(kx)))
        om = np.asarray(diag.variables["omega_kxkyt"][:, iky, ikx, 0], dtype=float)
        gm = np.asarray(diag.variables["omega_kxkyt"][:, iky, ikx, 1], dtype=float)
        out["omega"] = om
        out["gamma"] = gm
    root.close()
    return out


def _load_spectrax(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] == 9:
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
    if data.shape[1] == 10:
        # Runtime CLI format: t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat,pflux
        return {
            "t": data[:, 0],
            "gamma": data[:, 2],
            "omega": data[:, 3],
            "Wg": data[:, 4],
            "Wphi": data[:, 5],
            "Wapar": data[:, 6],
            "energy": data[:, 7],
            "heat": data[:, 8],
            "pflux": data[:, 9],
        }
    raise ValueError(f"Unsupported SPECTRAX CSV shape: {data.shape}")


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


def _absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(np.abs(a[mask] - b[mask])))


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
    parser.add_argument("--ky", type=float, default=0.3, help="ky mode used for GX omega/gamma extraction")
    args = parser.parse_args()

    gx = _load_gx(args.gx, ky_target=args.ky)
    sp = _load_spectrax(args.spectrax)
    t = sp["t"]

    gx_interp = {key: _interp(t, gx["t"], gx[key]) for key in ["Wg", "Wphi", "Wapar", "heat", "pflux"]}
    if "gamma" in gx and "omega" in gx:
        gx_interp["gamma"] = _interp(t, gx["t"], gx["gamma"])
        gx_interp["omega"] = _interp(t, gx["t"], gx["omega"])

    print(f"Wg rel error: {_relative_error(sp['Wg'], gx_interp['Wg']):.3e}")
    print(f"Wg abs error: {_absolute_error(sp['Wg'], gx_interp['Wg']):.3e}")
    print(f"Wphi rel error: {_relative_error(sp['Wphi'], gx_interp['Wphi']):.3e}")
    print(f"Wphi abs error: {_absolute_error(sp['Wphi'], gx_interp['Wphi']):.3e}")
    print(f"Wapar rel error: {_relative_error(sp['Wapar'], gx_interp['Wapar']):.3e}")
    print(f"Wapar abs error: {_absolute_error(sp['Wapar'], gx_interp['Wapar']):.3e}")
    print(f"Heat flux rel error: {_relative_error(sp['heat'], gx_interp['heat']):.3e}")
    print(f"Heat flux abs error: {_absolute_error(sp['heat'], gx_interp['heat']):.3e}")
    print(f"Particle flux rel error: {_relative_error(sp['pflux'], gx_interp['pflux']):.3e}")
    print(f"Particle flux abs error: {_absolute_error(sp['pflux'], gx_interp['pflux']):.3e}")
    if "gamma" in gx_interp and "omega" in gx_interp:
        print(f"Gamma rel error: {_relative_error(sp['gamma'], gx_interp['gamma']):.3e}")
        print(f"Gamma abs error: {_absolute_error(sp['gamma'], gx_interp['gamma']):.3e}")
        print(f"Omega rel error: {_relative_error(sp['omega'], gx_interp['omega']):.3e}")
        print(f"Omega abs error: {_absolute_error(sp['omega'], gx_interp['omega']):.3e}")
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

    fig, ax = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True)
    mask = np.isfinite(sp["Wphi"]) & np.isfinite(gx_interp["Wphi"])
    ax[0].plot(t[mask], sp["Wphi"][mask], label="SPECTRAX Wphi")
    ax[0].plot(t[mask], gx_interp["Wphi"][mask], label="GX Wphi", linestyle="--")
    ax[0].set_ylabel("Wphi")
    ax[0].legend(frameon=False)
    if "gamma" in gx_interp and "omega" in gx_interp:
        mask_g = np.isfinite(sp["gamma"]) & np.isfinite(gx_interp["gamma"])
        mask_o = np.isfinite(sp["omega"]) & np.isfinite(gx_interp["omega"])
        ax[1].plot(t[mask_g], sp["gamma"][mask_g], label="SPECTRAX gamma")
        ax[1].plot(t[mask_g], gx_interp["gamma"][mask_g], label="GX gamma", linestyle="--")
        ax[1].plot(t[mask_o], sp["omega"][mask_o], label="SPECTRAX omega")
        ax[1].plot(t[mask_o], gx_interp["omega"][mask_o], label="GX omega", linestyle="--")
        ax[1].set_ylabel("gamma / omega")
        ax[1].legend(frameon=False, ncol=2, fontsize=8)
    mask_h = np.isfinite(sp["heat"]) & np.isfinite(gx_interp["heat"])
    ax[2].plot(t[mask_h], sp["heat"][mask_h], label="SPECTRAX Q")
    ax[2].plot(t[mask_h], gx_interp["heat"][mask_h], label="GX Q", linestyle="--")
    ax[2].set_ylabel("Heat flux")
    ax[2].set_xlabel("t")
    ax[2].legend(frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
