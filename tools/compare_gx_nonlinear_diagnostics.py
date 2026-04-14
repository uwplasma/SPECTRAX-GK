#!/usr/bin/env python3
"""Compare GX vs SPECTRAX nonlinear diagnostics for Cyclone runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


def _reduce_species_resolved(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"expected resolved array with shape (time, species, mode), got {arr.shape} for {name}")
    if name.startswith("Wapar"):
        return np.asarray(arr[:, 0, :], dtype=float)
    return np.asarray(np.sum(arr, axis=1), dtype=float)


def _load_resolved_diag(path: Path) -> dict[str, np.ndarray]:
    root = Dataset(path, "r")
    diag = root.groups["Diagnostics"]
    grid = root.groups["Grids"]
    out = {
        "t": np.asarray(grid.variables["time"][:], dtype=float),
        "kx": np.asarray(grid.variables["kx"][:], dtype=float),
        "ky": np.asarray(grid.variables["ky"][:], dtype=float),
        "Wphi_kx": _reduce_species_resolved(np.asarray(diag.variables["Wphi_kxst"][:], dtype=float), "Wphi_kxst"),
        "Wphi_ky": _reduce_species_resolved(np.asarray(diag.variables["Wphi_kyst"][:], dtype=float), "Wphi_kyst"),
        "HeatFlux_kx": _reduce_species_resolved(
            np.asarray(diag.variables["HeatFlux_kxst"][:], dtype=float),
            "HeatFlux_kxst",
        ),
    }
    root.close()
    return out


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


def _interp_mode_summary(
    ref_t: np.ndarray,
    ref_y: np.ndarray,
    cmp_t: np.ndarray,
    cmp_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ref_y.ndim != 2 or cmp_y.ndim != 2:
        raise ValueError(f"expected resolved 2D arrays, got {ref_y.shape} and {cmp_y.shape}")
    if ref_y.shape[1] != cmp_y.shape[1]:
        raise ValueError(f"resolved mode count mismatch: {ref_y.shape[1]} vs {cmp_y.shape[1]}")
    mean_rel = np.zeros(ref_y.shape[1], dtype=float)
    max_rel = np.zeros(ref_y.shape[1], dtype=float)
    final_rel = np.zeros(ref_y.shape[1], dtype=float)
    for idx in range(ref_y.shape[1]):
        cmp_interp = np.interp(ref_t, cmp_t, cmp_y[:, idx])
        denom = np.maximum(np.abs(cmp_interp), 1.0e-30)
        rel = np.abs((ref_y[:, idx] - cmp_interp) / denom)
        mean_rel[idx] = float(np.mean(rel))
        max_rel[idx] = float(np.max(rel))
        final_rel[idx] = float((ref_y[-1, idx] - cmp_interp[-1]) / denom[-1])
    return mean_rel, max_rel, final_rel


def _write_resolved_audit(
    *,
    gx_path: Path,
    sp_path: Path,
    out_png: Path,
    out_csv: Path | None,
    tmax: float | None,
) -> None:
    gx = _load_resolved_diag(gx_path)
    sp = _load_resolved_diag(sp_path)

    if tmax is not None:
        gx_mask = gx["t"] <= tmax
        sp_mask = sp["t"] <= tmax
        gx["t"] = gx["t"][gx_mask]
        sp["t"] = sp["t"][sp_mask]
        for key in ("Wphi_kx", "Wphi_ky", "HeatFlux_kx"):
            gx[key] = gx[key][gx_mask, :]
            sp[key] = sp[key][sp_mask, :]

    rows: list[dict[str, float | str | int]] = []
    audit_specs = (
        ("Wphi_kx", "kx", gx["kx"]),
        ("Wphi_ky", "ky", gx["ky"]),
        ("HeatFlux_kx", "kx", gx["kx"]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    for ax, (name, axis_name, coords) in zip(axes, audit_specs):
        mean_rel, max_rel, final_rel = _interp_mode_summary(sp["t"], sp[name], gx["t"], gx[name])
        coord_vals = np.asarray(coords, dtype=float)
        ax.plot(coord_vals, mean_rel, marker="o", linewidth=2.0, label="mean rel")
        ax.plot(coord_vals, max_rel, marker="s", linewidth=1.8, linestyle="--", label="max rel")
        ax.plot(coord_vals, np.abs(final_rel), marker="^", linewidth=1.6, linestyle=":", label="|final rel|")
        ax.set_title(name.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_xlabel(axis_name)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        if axis_name == "kx":
            ax.set_ylabel("relative error")
        worst_idx = int(np.argmax(mean_rel))
        ax.axvline(coord_vals[worst_idx], color="0.6", linewidth=1.0, linestyle=":")
        ax.text(
            coord_vals[worst_idx],
            mean_rel[worst_idx],
            f"worst {axis_name}={coord_vals[worst_idx]:.3g}",
            fontsize=8,
            ha="left",
            va="bottom",
        )
        for idx, coord in enumerate(coord_vals):
            rows.append(
                {
                    "metric": name,
                    "axis": axis_name,
                    "mode_index": idx,
                    "mode_value": float(coord),
                    "mean_rel": float(mean_rel[idx]),
                    "max_rel": float(max_rel[idx]),
                    "final_rel": float(final_rel[idx]),
                }
            )
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Resolved nonlinear diagnostic audit: GX vs SPECTRAX-GK", fontsize=14, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    print(f"saved {out_png}")
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"saved {out_csv}")


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
    parser.add_argument(
        "--title",
        type=str,
        default="Nonlinear diagnostics: GX vs SPECTRAX-GK",
        help="Figure title.",
    )
    parser.add_argument(
        "--resolved-out",
        type=Path,
        default=None,
        help="Optional resolved-diagnostic audit figure (requires both inputs as .nc)",
    )
    parser.add_argument(
        "--resolved-csv",
        type=Path,
        default=None,
        help="Optional CSV summary for the resolved-diagnostic audit",
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

    gx_style = {"color": "#111827", "linewidth": 2.1, "linestyle": "--", "zorder": 2}
    sp_style = {"color": "#2563eb", "linewidth": 2.3, "marker": "o", "markevery": max(len(sp["t"]) // 14, 1), "ms": 3.2, "zorder": 3}
    fig, axes = plt.subplots(3, 2, figsize=(10.6, 8.9), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(gx["t"], gx["Wg"], label="GX", **gx_style)
    axes[0].plot(sp["t"], sp["Wg"], label="SPECTRAX-GK", **sp_style)
    axes[0].set_ylabel("Wg")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    axes[1].plot(gx["t"], gx["Wphi"], label="GX", **gx_style)
    axes[1].plot(sp["t"], sp["Wphi"], label="SPECTRAX-GK", **sp_style)
    axes[1].set_ylabel("Wphi")

    axes[2].plot(gx["t"], gx["Wapar"], label="GX", **gx_style)
    axes[2].plot(sp["t"], sp["Wapar"], label="SPECTRAX-GK", **sp_style)
    axes[2].set_ylabel("Wapar")

    axes[3].plot(gx["t"], gx["energy"], label="GX", **gx_style)
    axes[3].plot(sp["t"], sp["energy"], label="SPECTRAX-GK", **sp_style)
    axes[3].set_ylabel("Wtot")

    axes[4].plot(gx["t"], gx["heat_flux"], label="GX", **gx_style)
    axes[4].plot(sp["t"], sp["heat_flux"], label="SPECTRAX-GK", **sp_style)
    axes[4].set_ylabel("Heat flux")
    axes[4].set_xlabel("t")

    axes[5].plot(gx["t"], gx["particle_flux"], label="GX", **gx_style)
    axes[5].plot(sp["t"], sp["particle_flux"], label="SPECTRAX-GK", **sp_style)
    axes[5].set_ylabel("Particle flux")
    axes[5].set_xlabel("t")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(args.title, fontsize=13, fontweight="bold")
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
    if args.resolved_out is not None:
        if args.gx.suffix != ".nc" or args.spectrax.suffix != ".nc":
            raise ValueError("--resolved-out requires both --gx and --spectrax to be .nc files")
        _write_resolved_audit(
            gx_path=args.gx,
            sp_path=args.spectrax,
            out_png=args.resolved_out,
            out_csv=args.resolved_csv,
            tmax=args.tmax,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
