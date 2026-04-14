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


def _window_mask(
    t: np.ndarray,
    y: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> np.ndarray:
    def _bound_tol(bound: float) -> float:
        # Runtime CSV times often carry float32 roundoff (e.g. 0.10000000149).
        # Treat those samples as lying on the requested comparison window.
        return max(1.0e-12, 1.0e-8 * max(1.0, abs(bound)))

    mask = np.isfinite(y)
    if tmin is not None:
        tmin_f = float(tmin)
        mask = mask & (t >= (tmin_f - _bound_tol(tmin_f)))
    if tmax is not None:
        tmax_f = float(tmax)
        mask = mask & (t <= (tmax_f + _bound_tol(tmax_f)))
    return mask


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
    mask = _window_mask(t, a, tmin=tmin, tmax=tmax) & _window_mask(t, b, tmin=tmin, tmax=tmax)
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
    mask = _window_mask(t, a, tmin=tmin, tmax=tmax) & _window_mask(t, b, tmin=tmin, tmax=tmax)
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
    mask = _window_mask(t, y, tmin=t_min)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(y[mask]))


def _window_stats(
    t: np.ndarray,
    y: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> dict[str, float]:
    mask = _window_mask(t, y, tmin=tmin, tmax=tmax)
    if not np.any(mask):
        return {"mean": float("nan"), "std": float("nan"), "rms": float("nan")}
    values = np.asarray(y[mask], dtype=float)
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "rms": float(np.sqrt(np.nanmean(values**2))),
    }


def _scalar_relative_error(a: float, b: float, *, eps_rel: float = 1.0e-8) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    scale = max(abs(a), abs(b), 1.0)
    denom = max(abs(b), eps_rel * scale, 1.0e-30)
    return float(abs(a - b) / denom)


def _stats_relative_errors(
    sp_t: np.ndarray,
    sp_y: np.ndarray,
    gx_t: np.ndarray,
    gx_y: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    sp_stats = _window_stats(sp_t, sp_y, tmin=tmin, tmax=tmax)
    gx_stats = _window_stats(gx_t, gx_y, tmin=tmin, tmax=tmax)
    rel = {name: _scalar_relative_error(sp_stats[name], gx_stats[name]) for name in ("mean", "std", "rms")}
    abs_err = {name: float(abs(sp_stats[name] - gx_stats[name])) for name in ("mean", "std", "rms")}
    return sp_stats, gx_stats, {"rel_mean": rel["mean"], "rel_std": rel["std"], "rel_rms": rel["rms"], "abs_mean": abs_err["mean"], "abs_std": abs_err["std"], "abs_rms": abs_err["rms"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx", type=Path, required=True, help="GX .out.nc file")
    parser.add_argument(
        "--gx-early",
        type=Path,
        default=None,
        help="Optional dense GX .out.nc file used only for early-time checks.",
    )
    parser.add_argument("--spectrax", type=Path, required=True, help="SPECTRAX diagnostics CSV")
    parser.add_argument("--out", type=Path, default=Path("docs/_static/nonlinear_cyclone_compare.png"))
    parser.add_argument(
        "--title",
        type=str,
        default="Nonlinear diagnostics: GX vs SPECTRAX-GK",
        help="Figure title.",
    )
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
    parser.add_argument(
        "--late-mode",
        choices=["pointwise", "stats"],
        default="pointwise",
        help="Use pointwise interpolation or native-grid window statistics for late-time checks.",
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
    gx_early = _load_gx(args.gx_early, ky_target=args.ky) if args.gx_early is not None else gx
    sp = _load_spectrax(args.spectrax)
    t = sp["t"]

    gx_interp = {key: _interp(t, gx["t"], gx[key]) for key in ["Wg", "Wphi", "Wapar", "heat", "pflux"]}
    gx_early_interp = {key: _interp(t, gx_early["t"], gx_early[key]) for key in ["Wg", "Wphi", "Wapar", "heat", "pflux"]}
    if "gamma" in gx and "omega" in gx:
        gx_interp["gamma"] = _interp(t, gx["t"], gx["gamma"])
        gx_interp["omega"] = _interp(t, gx["t"], gx["omega"])
    if "gamma" in gx_early and "omega" in gx_early:
        gx_early_interp["gamma"] = _interp(t, gx_early["t"], gx_early["gamma"])
        gx_early_interp["omega"] = _interp(t, gx_early["t"], gx_early["omega"])

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
    if args.late_mode == "stats":
        print("Note: full-trace relative errors above are pointwise interpolation metrics; long-horizon acceptance uses the late-window native-grid statistics below.")
    if 0.0 < args.avg_fraction <= 1.0:
        wg_sp = _window_mean(sp["t"], sp["Wg"], args.avg_fraction)
        wg_gx = _window_mean(gx["t"], gx["Wg"], args.avg_fraction)
        wphi_sp = _window_mean(sp["t"], sp["Wphi"], args.avg_fraction)
        wphi_gx = _window_mean(gx["t"], gx["Wphi"], args.avg_fraction)
        q_sp = _window_mean(sp["t"], sp["heat"], args.avg_fraction)
        q_gx = _window_mean(gx["t"], gx["heat"], args.avg_fraction)
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
        "Wg": _relative_error_window(t, sp["Wg"], gx_early_interp["Wg"], tmax=args.early_tmax),
        "heat": _relative_error_window(t, sp["heat"], gx_early_interp["heat"], tmax=args.early_tmax),
        "pflux": _relative_error_window(t, sp["pflux"], gx_early_interp["pflux"], tmax=args.early_tmax),
    }
    early_abs = {
        "Wg": _absolute_error_window(t, sp["Wg"], gx_early_interp["Wg"], tmax=args.early_tmax),
        "heat": _absolute_error_window(t, sp["heat"], gx_early_interp["heat"], tmax=args.early_tmax),
        "pflux": _absolute_error_window(t, sp["pflux"], gx_early_interp["pflux"], tmax=args.early_tmax),
    }
    if args.late_mode == "pointwise":
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
    else:
        late_stats = {
            field: _stats_relative_errors(sp["t"], sp[field], gx["t"], gx[field], tmin=args.late_tmin)
            for field in ["Wg", "Wphi", "Wapar", "heat", "pflux"]
        }
        for field, (sp_stats, gx_stats, errs) in late_stats.items():
            print(
                "Late-window stats "
                f"{field} (t>= {args.late_tmin:g}): "
                f"mean SPECTRAX={sp_stats['mean']:.3e} GX={gx_stats['mean']:.3e} rel={errs['rel_mean']:.3e}; "
                f"std SPECTRAX={sp_stats['std']:.3e} GX={gx_stats['std']:.3e} rel={errs['rel_std']:.3e}; "
                f"rms SPECTRAX={sp_stats['rms']:.3e} GX={gx_stats['rms']:.3e} rel={errs['rel_rms']:.3e}"
            )
        late = {
            "Wg": max(late_stats["Wg"][2]["rel_mean"], late_stats["Wg"][2]["rel_std"]),
            "heat": max(late_stats["heat"][2]["rel_mean"], late_stats["heat"][2]["rel_std"]),
            "pflux": max(late_stats["pflux"][2]["rel_mean"], late_stats["pflux"][2]["rel_std"]),
        }
        late_abs = {
            "Wg": max(late_stats["Wg"][2]["abs_mean"], late_stats["Wg"][2]["abs_std"]),
            "heat": max(late_stats["heat"][2]["abs_mean"], late_stats["heat"][2]["abs_std"]),
            "pflux": max(late_stats["pflux"][2]["abs_mean"], late_stats["pflux"][2]["abs_std"]),
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
    late_label = "Late-window relative errors" if args.late_mode == "pointwise" else "Late-window summary relative errors"
    late_abs_label = "Late-window absolute errors" if args.late_mode == "pointwise" else "Late-window summary absolute errors"
    print(
        f"{late_label} "
        f"(t>= {args.late_tmin:g}): "
        f"Wg={late['Wg']:.3e}, heat={late['heat']:.3e}, pflux={late['pflux']:.3e}"
    )
    print(
        f"{late_abs_label} "
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

    gx_style = {"color": "#111827", "linewidth": 2.2, "linestyle": "--", "zorder": 2}
    sp_style = {"color": "#2563eb", "linewidth": 2.4, "marker": "o", "markevery": max(len(t) // 12, 1), "ms": 3.8, "zorder": 3}
    fig, ax = plt.subplots(3, 1, figsize=(8.2, 8.6), sharex=True, constrained_layout=True)
    mask = np.isfinite(sp["Wphi"]) & np.isfinite(gx_interp["Wphi"])
    ax[0].plot(t[mask], sp["Wphi"][mask], label="SPECTRAX-GK", **sp_style)
    ax[0].plot(t[mask], gx_interp["Wphi"][mask], label="GX", **gx_style)
    ax[0].set_ylabel("Wphi")
    ax[0].legend(frameon=False, ncol=2, loc="upper right")
    if "gamma" in gx_interp and "omega" in gx_interp:
        mask_g = np.isfinite(sp["gamma"]) & np.isfinite(gx_interp["gamma"])
        mask_o = np.isfinite(sp["omega"]) & np.isfinite(gx_interp["omega"])
        ax[1].plot(t[mask_g], sp["gamma"][mask_g], label="SPECTRAX γ", **sp_style)
        ax[1].plot(t[mask_g], gx_interp["gamma"][mask_g], label="GX γ", **gx_style)
        ax[1].plot(
            t[mask_o],
            sp["omega"][mask_o],
            label="SPECTRAX ω",
            color="#059669",
            linewidth=2.0,
            marker="s",
            markevery=max(len(t) // 12, 1),
            ms=3.5,
            zorder=3,
        )
        ax[1].plot(t[mask_o], gx_interp["omega"][mask_o], label="GX ω", color="#f97316", linewidth=2.0, linestyle=":")
        ax[1].set_ylabel("gamma / omega")
        ax[1].legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    mask_h = np.isfinite(sp["heat"]) & np.isfinite(gx_interp["heat"])
    ax[2].plot(t[mask_h], sp["heat"][mask_h], label="SPECTRAX-GK", **sp_style)
    ax[2].plot(t[mask_h], gx_interp["heat"][mask_h], label="GX", **gx_style)
    ax[2].set_ylabel("Heat flux")
    ax[2].set_xlabel("t")
    ax[2].legend(frameon=False, ncol=2, loc="upper right")
    for axis in ax:
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(args.title, fontsize=13, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
