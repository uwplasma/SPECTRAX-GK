#!/usr/bin/env python3
"""Build a GX-vs-SPECTRAX Cyclone/KBM linear+nonlinear comparison panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from spectraxgk.benchmarks import (
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    _two_species_params,
    run_cyclone_linear,
    run_kinetic_linear,
)
from spectraxgk.config import (
    CycloneBaseCase,
    GridConfig,
    InitializationConfig,
    KineticElectronBaseCase,
    KineticElectronModelConfig,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearTerms


def _normalize_mode(theta: np.ndarray, mode: np.ndarray) -> np.ndarray:
    finite = np.isfinite(mode)
    if not np.any(finite):
        return np.zeros_like(mode)
    idx0 = int(np.argmin(np.abs(theta)))
    ref = mode[idx0]
    if not np.isfinite(ref) or abs(ref) < 1.0e-14:
        j = int(np.nanargmax(np.abs(np.where(finite, mode, 0.0))))
        ref = mode[j]
    if not np.isfinite(ref) or abs(ref) < 1.0e-14:
        return mode
    return mode / ref


def _load_gx_eigenfunction(path: Path, ky_target: float) -> tuple[np.ndarray, np.ndarray]:
    root = Dataset(path, "r")
    grids = root.groups["Grids"]
    diag = root.groups["Diagnostics"]
    theta = np.asarray(grids.variables["theta"][:], dtype=float)
    ky = np.asarray(grids.variables["ky"][:], dtype=float)
    ky_idx = int(np.argmin(np.abs(ky - float(ky_target))))
    phi = np.asarray(diag.variables["Phi"][-1, ky_idx, 0, :, :], dtype=float)
    root.close()
    mode = phi[:, 0] + 1j * phi[:, 1]
    return theta, _normalize_mode(theta, mode)


def _cyclone_spectrax_eigenfunction(ky_target: float) -> tuple[np.ndarray, np.ndarray]:
    cfg = CycloneBaseCase(
        grid=GridConfig(
            Nx=1,
            Ny=24,
            Nz=96,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=20.0,
            ntheta=32,
            nperiod=2,
        )
    )
    res = run_cyclone_linear(
        ky_target=float(ky_target),
        cfg=cfg,
        Nl=16,
        Nm=8,
        solver="krylov",
        fit_signal="phi",
        diagnostic_norm="gx",
        gx_parity=True,
    )
    theta = np.asarray(build_spectral_grid(cfg.grid).z, dtype=float)
    mode = np.asarray(
        res.phi_t[-1, res.selection.ky_index, res.selection.kx_index, :],
        dtype=np.complex128,
    )
    return theta, _normalize_mode(theta, mode)


def _kbm_spectrax_eigenfunction(ky_target: float) -> tuple[np.ndarray, np.ndarray]:
    cfg = KineticElectronBaseCase(
        grid=GridConfig(
            Nx=1,
            Ny=12,
            Nz=96,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            ntheta=32,
            nperiod=2,
        ),
        model=KineticElectronModelConfig(
            R_over_LTi=2.49,
            R_over_LTe=2.49,
            R_over_Ln=0.8,
            Te_over_Ti=1.0,
            mass_ratio=3670.0,
            nu_i=0.0,
            nu_e=0.0,
            beta=0.015,
        ),
        init=InitializationConfig(init_field="all", init_amp=1.0e-10, gaussian_init=True),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = _two_species_params(
        cfg.model,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=KBM_OMEGA_D_SCALE,
        omega_star_scale=KBM_OMEGA_STAR_SCALE,
        rho_star=KBM_RHO_STAR,
        beta_override=0.015,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        nhermite=8,
    )
    res = run_kinetic_linear(
        ky_target=float(ky_target),
        cfg=cfg,
        params=params,
        Nl=4,
        Nm=8,
        solver="krylov",
        terms=LinearTerms(bpar=0.0),
        fit_signal="phi",
        init_species_index=1,
        density_species_index=1,
        diagnostic_norm="gx",
    )
    theta = np.asarray(build_spectral_grid(cfg.grid).z, dtype=float)
    mode = np.asarray(
        res.phi_t[-1, res.selection.ky_index, res.selection.kx_index, :],
        dtype=np.complex128,
    )
    return theta, _normalize_mode(theta, mode)


def _load_linear_scan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"ky", "gamma", "omega", "gamma_gx", "omega_gx"}
    if not req.issubset(df.columns):
        raise ValueError(f"{path} missing expected columns {req}")
    return df.sort_values("ky").reset_index(drop=True)


def _load_spectrax_diag(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    # Legacy tool CSV: t,gamma,omega,Wg,Wphi,Wapar,energy,heat,pflux (9 cols)
    if data.shape[1] == 9:
        return {
            "t": np.asarray(data[:, 0], dtype=float),
            "gamma": np.asarray(data[:, 1], dtype=float),
            "omega": np.asarray(data[:, 2], dtype=float),
            "Wg": np.asarray(data[:, 3], dtype=float),
            "Wphi": np.asarray(data[:, 4], dtype=float),
            "Wapar": np.asarray(data[:, 5], dtype=float),
            "energy": np.asarray(data[:, 6], dtype=float),
            "heat": np.asarray(data[:, 7], dtype=float),
            "pflux": np.asarray(data[:, 8], dtype=float),
        }
    # CLI/runtime CSV: t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat,pflux (10 cols)
    if data.shape[1] == 10:
        return {
            "t": np.asarray(data[:, 0], dtype=float),
            "gamma": np.asarray(data[:, 2], dtype=float),
            "omega": np.asarray(data[:, 3], dtype=float),
            "Wg": np.asarray(data[:, 4], dtype=float),
            "Wphi": np.asarray(data[:, 5], dtype=float),
            "Wapar": np.asarray(data[:, 6], dtype=float),
            "energy": np.asarray(data[:, 7], dtype=float),
            "heat": np.asarray(data[:, 8], dtype=float),
            "pflux": np.asarray(data[:, 9], dtype=float),
        }
    raise ValueError(f"Unsupported diagnostics CSV format in {path} with shape {data.shape}")


def _series_ky(arr: np.ndarray, ky_idx: int) -> np.ndarray:
    if arr.ndim == 1:
        return np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        if ky_idx < arr.shape[1]:
            return np.asarray(arr[:, ky_idx], dtype=float)
        return np.asarray(np.sum(arr, axis=1), dtype=float)
    if arr.ndim == 3:
        # (t, species, ky)
        return np.asarray(np.sum(arr[:, :, ky_idx], axis=1), dtype=float)
    raise ValueError(f"unsupported array shape {arr.shape}")


def _load_gx_nonlinear(path: Path, ky_target: float) -> dict[str, np.ndarray]:
    root = Dataset(path, "r")
    grids = root.groups["Grids"]
    diag = root.groups["Diagnostics"]
    t = np.asarray(grids.variables["time"][:], dtype=float)
    ky = np.asarray(grids.variables["ky"][:], dtype=float)
    ky_idx = int(np.argmin(np.abs(ky - float(ky_target))))

    if "Wphi_st" in diag.variables:
        wphi_arr = np.asarray(diag.variables["Wphi_st"][:], dtype=float)
        if wphi_arr.ndim == 2:
            wphi = np.sum(wphi_arr, axis=1)
        else:
            wphi = wphi_arr
    else:
        wphi = _series_ky(np.asarray(diag.variables["Wphi_kyst"][:], dtype=float), ky_idx)

    if "HeatFlux_st" in diag.variables:
        heat_arr = np.asarray(diag.variables["HeatFlux_st"][:], dtype=float)
        if heat_arr.ndim == 2:
            heat = np.sum(heat_arr, axis=1)
        else:
            heat = heat_arr
    else:
        heat = _series_ky(np.asarray(diag.variables["HeatFlux_kyst"][:], dtype=float), ky_idx)

    out = {
        "t": t,
        "Wphi": np.asarray(wphi, dtype=float),
        "heat": np.asarray(heat, dtype=float),
    }

    if "omega_kxkyt" in diag.variables:
        omega_kxkyt = np.asarray(diag.variables["omega_kxkyt"][:], dtype=float)
        # Match CLI/runtime CSV reduction: mean over all (ky, kx) entries.
        out["omega"] = np.nanmean(omega_kxkyt[..., 0], axis=(1, 2))
        out["gamma"] = np.nanmean(omega_kxkyt[..., 1], axis=(1, 2))
    else:
        phi2 = np.asarray(diag.variables["Phi2_t"][:], dtype=float)
        gamma = np.full_like(t, np.nan)
        if t.size > 1:
            dt = np.diff(t)
            ratio = np.maximum(phi2[1:] / np.maximum(phi2[:-1], 1.0e-30), 1.0e-30)
            gamma[1:] = 0.5 * np.log(ratio) / np.maximum(dt, 1.0e-12)
        out["gamma"] = gamma
        out["omega"] = np.full_like(t, np.nan)
    root.close()
    return out


def _plot_linear_row(
    ax_row: np.ndarray,
    title: str,
    theta_sp: np.ndarray,
    mode_sp: np.ndarray,
    theta_gx: np.ndarray,
    mode_gx: np.ndarray,
    scan: pd.DataFrame,
) -> None:
    ax = ax_row[0]
    ax.plot(theta_sp, np.real(mode_sp), color="tab:blue", lw=1.8, label="SPECTRAX Re")
    ax.plot(theta_sp, np.imag(mode_sp), color="tab:orange", lw=1.8, ls="--", label="SPECTRAX Im")
    ax.plot(theta_gx, np.real(mode_gx), color="tab:green", lw=1.6, alpha=0.8, label="GX Re")
    ax.plot(theta_gx, np.imag(mode_gx), color="tab:red", lw=1.6, ls="--", alpha=0.8, label="GX Im")
    ax.set_ylabel(f"{title}\n$\\phi(\\theta)/\\phi(0)$")
    ax.set_xlabel(r"$\theta$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = ax_row[1]
    ax.plot(scan["ky"], scan["gamma"], "o-", lw=1.8, label="SPECTRAX")
    ax.plot(scan["ky"], scan["gamma_gx"], "s--", lw=1.8, label="GX")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\gamma$")
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = ax_row[2]
    ax.plot(scan["ky"], scan["omega"], "o-", lw=1.8, label="SPECTRAX")
    ax.plot(scan["ky"], scan["omega_gx"], "s--", lw=1.8, label="GX")
    ax.set_xscale("log")
    ax.set_ylabel(r"$\omega$")
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = ax_row[3]
    rel_g = np.abs((scan["gamma"] - scan["gamma_gx"]) / np.maximum(np.abs(scan["gamma_gx"]), 1.0e-12))
    rel_o = np.abs((scan["omega"] - scan["omega_gx"]) / np.maximum(np.abs(scan["omega_gx"]), 1.0e-12))
    ax.plot(scan["ky"], rel_g, "o-", lw=1.8, label=r"$|\Delta\gamma|/|\gamma_{GX}|$")
    ax.plot(scan["ky"], rel_o, "s--", lw=1.8, label=r"$|\Delta\omega|/|\omega_{GX}|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Relative error (γ, ω)")
    ax.set_xlabel(r"$k_y \rho_i$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def _plot_nonlinear_row(
    ax_row: np.ndarray,
    title: str,
    gx: dict[str, np.ndarray],
    sp: dict[str, np.ndarray],
) -> None:
    t_max = float(min(np.nanmax(gx["t"]), np.nanmax(sp["t"])))
    mask_t_gx = np.asarray(gx["t"] <= t_max, dtype=bool)
    mask_t_sp = np.asarray(sp["t"] <= t_max, dtype=bool)

    ax = ax_row[0]
    ax.plot(gx["t"][mask_t_gx], gx["Wphi"][mask_t_gx], lw=1.8, label="GX")
    ax.plot(sp["t"][mask_t_sp], sp["Wphi"][mask_t_sp], lw=1.8, label="SPECTRAX")
    ax.set_ylabel(f"{title}\n$W_\\phi$")
    ax.set_xlabel("t")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = ax_row[1]
    mask_gx = mask_t_gx & np.isfinite(gx["gamma"])
    mask_sp = mask_t_sp & np.isfinite(sp["gamma"])
    ax.plot(gx["t"][mask_gx], gx["gamma"][mask_gx], lw=1.8, label="GX")
    ax.plot(sp["t"][mask_sp], sp["gamma"][mask_sp], lw=1.8, label="SPECTRAX")
    ax.set_ylabel(r"$\gamma(t)$")
    ax.set_xlabel("t")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = ax_row[2]
    mask_gx = mask_t_gx & np.isfinite(gx["omega"])
    mask_sp = mask_t_sp & np.isfinite(sp["omega"])
    ax.plot(gx["t"][mask_gx], gx["omega"][mask_gx], lw=1.8, label="GX")
    ax.plot(sp["t"][mask_sp], sp["omega"][mask_sp], lw=1.8, label="SPECTRAX")
    ax.set_ylabel(r"$\omega(t)$")
    ax.set_xlabel("t")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = ax_row[3]
    ax.plot(gx["t"][mask_t_gx], gx["heat"][mask_t_gx], lw=1.8, label="GX")
    ax.plot(sp["t"][mask_t_sp], sp["heat"][mask_t_sp], lw=1.8, label="SPECTRAX")
    ax.set_ylabel("Heat flux")
    ax.set_xlabel("t")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cyclone-linear",
        type=Path,
        default=Path("docs/_static/cyclone_gx_mismatch.csv"),
    )
    parser.add_argument(
        "--kbm-linear",
        type=Path,
        default=Path("docs/_static/kbm_gx_mismatch.csv"),
    )
    parser.add_argument(
        "--gx-cyclone-linear-big",
        type=Path,
        default=Path(".cache/gx/itg_salpha_adiabatic_electrons.big.nc"),
    )
    parser.add_argument(
        "--gx-kbm-linear-big",
        type=Path,
        default=Path(".cache/gx/kbm_salpha.big.nc"),
    )
    parser.add_argument(
        "--gx-cyclone-nonlinear",
        type=Path,
        default=Path(".cache/gx/cyclone_salpha_nonlinear_omega.out.nc"),
    )
    parser.add_argument(
        "--gx-kbm-nonlinear",
        type=Path,
        default=Path(".cache/gx/kbm_salpha_nonlinear.out.nc"),
    )
    parser.add_argument(
        "--spectrax-cyclone-nonlinear",
        type=Path,
        default=Path(".cache/spectrax/cyclone_nonlinear_diag.csv"),
    )
    parser.add_argument(
        "--spectrax-kbm-nonlinear",
        type=Path,
        default=Path(".cache/spectrax/kbm_nonlinear_diag.csv"),
    )
    parser.add_argument("--cyclone-ky", type=float, default=0.3)
    parser.add_argument("--kbm-ky", type=float, default=0.3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/_static/gx_cyclone_kbm_panel.png"),
    )
    args = parser.parse_args()

    cycl_scan = _load_linear_scan(args.cyclone_linear)
    cycl_scan = cycl_scan[cycl_scan["ky"] <= 0.5 + 1.0e-12].reset_index(drop=True)
    kbm_scan = _load_linear_scan(args.kbm_linear)

    theta_gx_c, mode_gx_c = _load_gx_eigenfunction(args.gx_cyclone_linear_big, args.cyclone_ky)
    theta_sp_c, mode_sp_c = _cyclone_spectrax_eigenfunction(args.cyclone_ky)
    theta_gx_k, mode_gx_k = _load_gx_eigenfunction(args.gx_kbm_linear_big, args.kbm_ky)
    theta_sp_k, mode_sp_k = _kbm_spectrax_eigenfunction(args.kbm_ky)

    gx_nl_c = _load_gx_nonlinear(args.gx_cyclone_nonlinear, args.cyclone_ky)
    gx_nl_k = _load_gx_nonlinear(args.gx_kbm_nonlinear, args.kbm_ky)
    sp_nl_c = _load_spectrax_diag(args.spectrax_cyclone_nonlinear)
    sp_nl_k = _load_spectrax_diag(args.spectrax_kbm_nonlinear)

    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    _plot_linear_row(
        axes[0],
        "Cyclone linear",
        theta_sp_c,
        mode_sp_c,
        theta_gx_c,
        mode_gx_c,
        cycl_scan,
    )
    _plot_nonlinear_row(axes[1], "Cyclone nonlinear", gx_nl_c, sp_nl_c)
    _plot_linear_row(
        axes[2],
        "KBM linear",
        theta_sp_k,
        mode_sp_k,
        theta_gx_k,
        mode_gx_k,
        kbm_scan,
    )
    _plot_nonlinear_row(axes[3], "KBM nonlinear", gx_nl_k, sp_nl_k)

    col_titles = [
        "Eigenfunction / field",
        "Growth rate",
        "Frequency",
        "Relative error (linear γ,ω) / Heat flux (nonlinear)",
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)
    fig.suptitle("SPECTRAX-GK vs GX: Cyclone and KBM (linear + nonlinear)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
