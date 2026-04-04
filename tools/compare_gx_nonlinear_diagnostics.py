#!/usr/bin/env python3
"""Compare GX vs SPECTRAX nonlinear diagnostics for Cyclone runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def _reduce_species_time(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 2:
        if name.startswith("Wapar"):
            return arr[:, 0]
        return np.sum(arr, axis=1)
    return arr


def _load_spectrax_csv(path: Path) -> dict[str, np.ndarray]:
    named = np.genfromtxt(path, delimiter=",", names=True)
    if isinstance(named, np.ndarray) and named.dtype.names:
        names = set(named.dtype.names)
        if {"t", "Wg", "Wphi", "Wapar", "energy", "heat_flux", "particle_flux"}.issubset(names):
            out = {
                "t": np.asarray(named["t"], dtype=float),
                "Wg": np.asarray(named["Wg"], dtype=float),
                "Wphi": np.asarray(named["Wphi"], dtype=float),
                "Wapar": np.asarray(named["Wapar"], dtype=float),
                "energy": np.asarray(named["energy"], dtype=float),
                "heat_flux": np.asarray(named["heat_flux"], dtype=float),
                "particle_flux": np.asarray(named["particle_flux"], dtype=float),
            }
            if "gamma" in names:
                out["gamma"] = np.asarray(named["gamma"], dtype=float)
            if "omega" in names:
                out["omega"] = np.asarray(named["omega"], dtype=float)
            return out

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
            "heat_flux": data[:, 7],
            "particle_flux": data[:, 8],
        }
    if data.shape[1] >= 10:
        return {
            "t": data[:, 0],
            "gamma": data[:, 2],
            "omega": data[:, 3],
            "Wg": data[:, 4],
            "Wphi": data[:, 5],
            "Wapar": data[:, 6],
            "energy": data[:, 7],
            "heat_flux": data[:, 8],
            "particle_flux": data[:, 9],
        }
    raise ValueError(f"unsupported SPECTRAX CSV shape {data.shape}")


def _load_spectrax_diag(path: Path) -> dict[str, np.ndarray]:
    root = Dataset(path, "r")
    diag = root.groups["Diagnostics"]
    grid = root.groups["Grids"]
    out = {
        "t": np.asarray(grid.variables["time"][:], dtype=float),
        "phi2": np.asarray(diag.variables["Phi2_t"][:], dtype=float),
        "Wg": _reduce_species_time(np.asarray(diag.variables["Wg_st"][:], dtype=float), "Wg_st"),
        "Wphi": _reduce_species_time(np.asarray(diag.variables["Wphi_st"][:], dtype=float), "Wphi_st"),
        "Wapar": _reduce_species_time(np.asarray(diag.variables["Wapar_st"][:], dtype=float), "Wapar_st"),
        "heat_flux": _reduce_species_time(np.asarray(diag.variables["HeatFlux_st"][:], dtype=float), "HeatFlux_st"),
        "particle_flux": _reduce_species_time(
            np.asarray(diag.variables["ParticleFlux_st"][:], dtype=float), "ParticleFlux_st"
        ),
    }
    out["energy"] = out["Wg"] + out["Wphi"] + out["Wapar"]
    root.close()
    return out


def _load_gx_diag(path: Path) -> dict[str, np.ndarray]:
    root = Dataset(path, "r")
    diag = root.groups["Diagnostics"]
    grid = root.groups["Grids"]
    t = np.asarray(grid.variables["time"][:], dtype=float)
    out = {
        "t": t,
        "phi2": np.asarray(diag.variables["Phi2_t"][:], dtype=float),
        "Wg": _reduce_species_time(np.asarray(diag.variables["Wg_st"][:], dtype=float), "Wg_st"),
        "Wphi": _reduce_species_time(np.asarray(diag.variables["Wphi_st"][:], dtype=float), "Wphi_st"),
        "Wapar": _reduce_species_time(np.asarray(diag.variables["Wapar_st"][:], dtype=float), "Wapar_st"),
        "heat_flux": _reduce_species_time(np.asarray(diag.variables["HeatFlux_st"][:], dtype=float), "HeatFlux_st"),
        "particle_flux": _reduce_species_time(
            np.asarray(diag.variables["ParticleFlux_st"][:], dtype=float), "ParticleFlux_st"
        ),
    }
    out["energy"] = out["Wg"] + out["Wphi"] + out["Wapar"]
    root.close()
    return out


def _load_spectrax(path: Path) -> dict[str, np.ndarray]:
    if path.suffix == ".nc":
        return _load_spectrax_diag(path)
    return _load_spectrax_csv(path)


def _interp_summary(
    ref_t: np.ndarray,
    ref_y: np.ndarray,
    cmp_t: np.ndarray,
    cmp_y: np.ndarray,
) -> tuple[float, float, float]:
    cmp_interp = np.interp(ref_t, cmp_t, cmp_y)
    denom = np.maximum(np.abs(cmp_interp), 1.0e-30)
    rel = np.abs((ref_y - cmp_interp) / denom)
    final_rel = float((ref_y[-1] - cmp_interp[-1]) / denom[-1])
    return float(np.mean(rel)), float(np.max(rel)), final_rel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx", type=Path, required=True, help="GX .out.nc file with diagnostics")
    parser.add_argument("--spectrax", type=Path, required=True, help="SPECTRAX nonlinear CSV or GX-style .out.nc")
    parser.add_argument("--tmax", type=float, default=None, help="Optional max time for plotting")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/_static/nonlinear_cyclone_diag_compare.png"),
        help="Output figure path",
    )
    args = parser.parse_args()

    gx = _load_gx_diag(args.gx)
    sp = _load_spectrax(args.spectrax)

    if args.tmax is not None:
        gx_mask = gx["t"] <= args.tmax
        sp_mask = sp["t"] <= args.tmax
        for key in gx:
            gx[key] = gx[key][gx_mask]
        for key in sp:
            sp[key] = sp[key][sp_mask]

    fig, axes = plt.subplots(3, 2, figsize=(9.5, 8.5), sharex=True)
    axes = axes.ravel()

    axes[0].plot(gx["t"], gx["Wg"], label="GX", lw=2)
    axes[0].plot(sp["t"], sp["Wg"], label="SPECTRAX-GK", lw=2)
    axes[0].set_ylabel("Wg")
    axes[0].legend(frameon=False)

    axes[1].plot(gx["t"], gx["Wphi"], label="GX", lw=2)
    axes[1].plot(sp["t"], sp["Wphi"], label="SPECTRAX-GK", lw=2)
    axes[1].set_ylabel("Wphi")

    axes[2].plot(gx["t"], gx["Wapar"], label="GX", lw=2)
    axes[2].plot(sp["t"], sp["Wapar"], label="SPECTRAX-GK", lw=2)
    axes[2].set_ylabel("Wapar")

    axes[3].plot(gx["t"], gx["energy"], label="GX", lw=2)
    axes[3].plot(sp["t"], sp["energy"], label="SPECTRAX-GK", lw=2)
    axes[3].set_ylabel("Wtot")

    axes[4].plot(gx["t"], gx["heat_flux"], label="GX", lw=2)
    axes[4].plot(sp["t"], sp["heat_flux"], label="SPECTRAX-GK", lw=2)
    axes[4].set_ylabel("Heat flux")
    axes[4].set_xlabel("t")

    axes[5].plot(gx["t"], gx["particle_flux"], label="GX", lw=2)
    axes[5].plot(sp["t"], sp["particle_flux"], label="SPECTRAX-GK", lw=2)
    axes[5].set_ylabel("Particle flux")
    axes[5].set_xlabel("t")

    fig.suptitle("Nonlinear Cyclone diagnostics: GX vs SPECTRAX-GK", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")

    summary_pairs = []
    if "phi2" in sp and "phi2" in gx:
        summary_pairs.append(("Phi2", sp["phi2"], gx["phi2"]))
    summary_pairs.extend(
        [
            ("Wg", sp["Wg"], gx["Wg"]),
            ("Wphi", sp["Wphi"], gx["Wphi"]),
            ("Wapar", sp["Wapar"], gx["Wapar"]),
            ("HeatFlux", sp["heat_flux"], gx["heat_flux"]),
            ("ParticleFlux", sp["particle_flux"], gx["particle_flux"]),
        ]
    )
    print("metric mean_rel_abs max_rel_abs final_rel")
    for name, sp_y, gx_y in summary_pairs:
        mean_rel, max_rel, final_rel = _interp_summary(sp["t"], sp_y, gx["t"], gx_y)
        print(f"{name} {mean_rel:.6e} {max_rel:.6e} {final_rel:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
