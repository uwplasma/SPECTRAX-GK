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
        if name.startswith("Wapar_"):
            return np.asarray(var[:, 0, :], dtype=float).sum(axis=1)
        return np.sum(var, axis=(1, 2))
    if var.ndim == 2:
        if name.startswith("Wapar_"):
            return np.asarray(var[:, 0], dtype=float)
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
    if "HeatFlux_st" in diag.variables:
        out["heat_s"] = np.asarray(diag.variables["HeatFlux_st"][:], dtype=float)
    if "ParticleFlux_st" in diag.variables:
        out["pflux_s"] = np.asarray(diag.variables["ParticleFlux_st"][:], dtype=float)
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
                "heat": np.asarray(named["heat_flux"], dtype=float),
                "pflux": np.asarray(named["particle_flux"], dtype=float),
            }
            if "gamma" in names:
                out["gamma"] = np.asarray(named["gamma"], dtype=float)
            if "omega" in names:
                out["omega"] = np.asarray(named["omega"], dtype=float)
            heat_species_cols = sorted(
                [name for name in names if name.startswith("heat_flux_s")],
                key=lambda key: int(key.removeprefix("heat_flux_s")),
            )
            if heat_species_cols:
                out["heat_s"] = np.column_stack([np.asarray(named[key], dtype=float) for key in heat_species_cols])
            pflux_species_cols = sorted(
                [name for name in names if name.startswith("particle_flux_s")],
                key=lambda key: int(key.removeprefix("particle_flux_s")),
            )
            if pflux_species_cols:
                out["pflux_s"] = np.column_stack([np.asarray(named[key], dtype=float) for key in pflux_species_cols])
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
            "heat": data[:, 7],
            "pflux": data[:, 8],
        }
    if data.shape[1] == 8:
        # Restart/exact-state format: t,gamma,omega,Wg,Wphi,Wapar,heat,pflux
        return {
            "t": data[:, 0],
            "gamma": data[:, 1],
            "omega": data[:, 2],
            "Wg": data[:, 3],
            "Wphi": data[:, 4],
            "Wapar": data[:, 5],
            "heat": data[:, 6],
            "pflux": data[:, 7],
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


def _relative_error(a: np.ndarray, b: np.ndarray, *, eps_rel: float = 1.0e-8) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    b_sel = b[mask]
    scale = float(np.nanmax(np.abs(b_sel)))
    denom_floor = max(eps_rel * scale, 1.0e-30)
    denom = np.maximum(np.abs(b_sel), denom_floor)
    return float(np.nanmean(np.abs(a[mask] - b_sel) / denom))


def _relative_error_window(
    t: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    eps_rel: float = 1.0e-8,
) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if tmin is not None:
        mask = mask & (t >= float(tmin))
    if tmax is not None:
        mask = mask & (t <= float(tmax))
    if not np.any(mask):
        return float("nan")
    b_sel = b[mask]
    scale = float(np.nanmax(np.abs(b_sel)))
    denom_floor = max(eps_rel * scale, 1.0e-30)
    denom = np.maximum(np.abs(b_sel), denom_floor)
    return float(np.nanmean(np.abs(a[mask] - b_sel) / denom))


def _absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(np.abs(a[mask] - b[mask])))


def _absolute_error_window(
    t: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if tmin is not None:
        mask = mask & (t >= float(tmin))
    if tmax is not None:
        mask = mask & (t <= float(tmax))
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(np.abs(a[mask] - b[mask])))


def _pass_tol(rel: float, abs_err: float, rtol: float, atol: float | None) -> bool:
    if np.isfinite(rel) and rel <= rtol:
        return True
    if atol is not None and np.isfinite(abs_err) and abs_err <= atol:
        return True
    return False


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
    parser.add_argument(
        "--early-tmax",
        type=float,
        default=0.1,
        help="Early-time window end used for strict parity checks.",
    )
    parser.add_argument(
        "--late-tmin",
        type=float,
        default=1.0,
        help="Late-time window start used for relaxed nonlinear checks.",
    )
    parser.add_argument("--rtol-early-Wg", type=float, default=0.05)
    parser.add_argument("--rtol-early-heat", type=float, default=0.2)
    parser.add_argument("--rtol-early-pflux", type=float, default=0.3)
    parser.add_argument("--rtol-late-Wg", type=float, default=0.5)
    parser.add_argument("--rtol-late-heat", type=float, default=1.0)
    parser.add_argument("--rtol-late-pflux", type=float, default=1.0)
    parser.add_argument("--atol-early-Wg", type=float, default=None)
    parser.add_argument("--atol-early-heat", type=float, default=None)
    parser.add_argument("--atol-early-pflux", type=float, default=None)
    parser.add_argument("--atol-late-Wg", type=float, default=None)
    parser.add_argument("--atol-late-heat", type=float, default=None)
    parser.add_argument("--atol-late-pflux", type=float, default=None)
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
    if "heat_s" in sp and "heat_s" in gx:
        ns = min(sp["heat_s"].shape[1], gx["heat_s"].shape[1])
        gx_heat_s = np.column_stack([_interp(t, gx["t"], gx["heat_s"][:, i]) for i in range(ns)])
        gx_pflux_s = np.column_stack([_interp(t, gx["t"], gx["pflux_s"][:, i]) for i in range(ns)])
        for i in range(ns):
            print(f"Heat flux s{i} rel error: {_relative_error(sp['heat_s'][:, i], gx_heat_s[:, i]):.3e}")
            print(f"Particle flux s{i} rel error: {_relative_error(sp['pflux_s'][:, i], gx_pflux_s[:, i]):.3e}")

    early = {
        "Wg": _relative_error_window(t, sp["Wg"], gx_interp["Wg"], tmax=args.early_tmax),
        "heat": _relative_error_window(t, sp["heat"], gx_interp["heat"], tmax=args.early_tmax),
        "pflux": _relative_error_window(t, sp["pflux"], gx_interp["pflux"], tmax=args.early_tmax),
    }
    early_abs = {
        "Wg": _absolute_error_window(t, sp["Wg"], gx_interp["Wg"], tmax=args.early_tmax),
        "heat": _absolute_error_window(t, sp["heat"], gx_interp["heat"], tmax=args.early_tmax),
        "pflux": _absolute_error_window(t, sp["pflux"], gx_interp["pflux"], tmax=args.early_tmax),
    }
    late = {
        "Wg": _relative_error_window(t, sp["Wg"], gx_interp["Wg"], tmin=args.late_tmin),
        "heat": _relative_error_window(t, sp["heat"], gx_interp["heat"], tmin=args.late_tmin),
        "pflux": _relative_error_window(t, sp["pflux"], gx_interp["pflux"], tmin=args.late_tmin),
    }
    late_abs = {
        "Wg": _absolute_error_window(t, sp["Wg"], gx_interp["Wg"], tmin=args.late_tmin),
        "heat": _absolute_error_window(t, sp["heat"], gx_interp["heat"], tmin=args.late_tmin),
        "pflux": _absolute_error_window(t, sp["pflux"], gx_interp["pflux"], tmin=args.late_tmin),
    }
    print(
        "Early-window relative errors "
        f"(t<= {args.early_tmax:g}): "
        f"Wg={early['Wg']:.3e}, heat={early['heat']:.3e}, pflux={early['pflux']:.3e}"
    )
    print(
        "Early-window absolute errors "
        f"(t<= {args.early_tmax:g}): "
        f"Wg={early_abs['Wg']:.3e}, heat={early_abs['heat']:.3e}, pflux={early_abs['pflux']:.3e}"
    )
    print(
        "Late-window relative errors "
        f"(t>= {args.late_tmin:g}): "
        f"Wg={late['Wg']:.3e}, heat={late['heat']:.3e}, pflux={late['pflux']:.3e}"
    )
    print(
        "Late-window absolute errors "
        f"(t>= {args.late_tmin:g}): "
        f"Wg={late_abs['Wg']:.3e}, heat={late_abs['heat']:.3e}, pflux={late_abs['pflux']:.3e}"
    )
    early_ok = (
        _pass_tol(early["Wg"], early_abs["Wg"], args.rtol_early_Wg, args.atol_early_Wg)
        and _pass_tol(early["heat"], early_abs["heat"], args.rtol_early_heat, args.atol_early_heat)
        and _pass_tol(early["pflux"], early_abs["pflux"], args.rtol_early_pflux, args.atol_early_pflux)
    )
    late_ok = (
        _pass_tol(late["Wg"], late_abs["Wg"], args.rtol_late_Wg, args.atol_late_Wg)
        and _pass_tol(late["heat"], late_abs["heat"], args.rtol_late_heat, args.atol_late_heat)
        and _pass_tol(late["pflux"], late_abs["pflux"], args.rtol_late_pflux, args.atol_late_pflux)
    )
    print(f"Tolerance check: early={'PASS' if early_ok else 'FAIL'} late={'PASS' if late_ok else 'FAIL'}")

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
