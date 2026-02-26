#!/usr/bin/env python3
"""Compare GX linear outputs against SPECTRAX-GK for the Cyclone base case."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from spectraxgk.analysis import ModeSelection, gx_growth_rate_from_phi, select_ky_index
from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    _apply_gx_hypercollisions,
    _build_initial_condition,
    _midplane_index,
    run_cyclone_scan,
)
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.gx_integrators import GXTimeConfig, integrate_linear_gx
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache


def _load_gx_omega_gamma(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Dataset(path, "r")
    try:
        grids = root.groups["Grids"]
        diagnostics = root.groups["Diagnostics"]
    except KeyError as exc:
        raise ValueError(f"{path} missing expected GX groups") from exc

    ky = np.asarray(grids.variables["ky"][:], dtype=float)
    omega_kxkyt = np.asarray(diagnostics.variables["omega_kxkyt"][:], dtype=float)
    if omega_kxkyt.ndim != 4 or omega_kxkyt.shape[-1] != 2:
        raise ValueError(f"unexpected omega_kxkyt shape: {omega_kxkyt.shape}")

    # final time slice, kx=0, ri axis = (omega, gamma)
    omega_last = omega_kxkyt[-1, :, 0, :]
    omega = omega_last[:, 0]
    gamma = omega_last[:, 1]

    root.close()
    mask = ky > 0.0
    return ky[mask], gamma[mask], omega[mask]


def _build_cyclone_cfg(
    ny: int,
    *,
    ntheta: int,
    nperiod: int,
    y0: float,
    drift_scale: float,
) -> CycloneBaseCase:
    grid = GridConfig(
        Nx=1,
        Ny=ny,
        Nz=ntheta * (2 * nperiod - 1),
        Lx=62.8,
        Ly=62.8,
        y0=y0,
        ntheta=ntheta,
        nperiod=nperiod,
        boundary="linked",
    )
    cfg = CycloneBaseCase(grid=grid)
    return replace(cfg, geometry=replace(cfg.geometry, drift_scale=drift_scale))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GX Cyclone output against SPECTRAX-GK.")
    parser.add_argument("--gx", type=Path, required=True, help="Path to GX .out.nc file")
    parser.add_argument("--Nl", type=int, default=48)
    parser.add_argument("--Nm", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--method", type=str, default="imex2")
    parser.add_argument("--solver", type=str, default="auto")
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--y0", type=float, default=20.0)
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument(
        "--ky",
        type=str,
        default="",
        help="Comma-separated ky values to compare (default: all from GX output)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV path for mismatch table")
    args = parser.parse_args()

    gx_ky, gx_gamma, gx_omega = _load_gx_omega_gamma(args.gx)
    if args.ky:
        ky_req = np.asarray([float(k.strip()) for k in args.ky.split(",") if k.strip()])
        if ky_req.size == 0:
            raise ValueError("No ky values parsed from --ky")
        idx = [int(np.argmin(np.abs(gx_ky - k))) for k in ky_req]
        gx_ky = gx_ky[idx]
        gx_gamma = gx_gamma[idx]
        gx_omega = gx_omega[idx]
    cfg = _build_cyclone_cfg(
        args.ny, ntheta=args.ntheta, nperiod=args.nperiod, y0=args.y0, drift_scale=args.drift_scale
    )

    if args.solver == "gx_time":
        geom = SAlphaGeometry.from_config(cfg.geometry)
        grid_full = build_spectral_grid(cfg.grid)
        gammas = []
        omegas = []
        for ky_val in gx_ky:
            ky_index = select_ky_index(np.asarray(grid_full.ky), float(ky_val))
            grid = select_ky_grid(grid_full, ky_index)
            params = LinearParams(
                R_over_Ln=cfg.model.R_over_Ln,
                R_over_LTi=cfg.model.R_over_LTi,
                R_over_LTe=cfg.model.R_over_LTe,
                omega_d_scale=CYCLONE_OMEGA_D_SCALE,
                omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
                rho_star=CYCLONE_RHO_STAR,
                kpar_scale=float(geom.gradpar()),
                nu=cfg.model.nu_i,
                damp_ends_amp=0.1,
                damp_ends_widthfrac=1.0 / 8.0,
            )
            params = _apply_gx_hypercollisions(params, nhermite=args.Nm)
            cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
            G0 = _build_initial_condition(
                grid,
                geom,
                ky_index=0,
                kx_index=0,
                Nl=args.Nl,
                Nm=args.Nm,
                init_cfg=cfg.init,
            )
            time_cfg = GXTimeConfig(dt=args.dt, t_max=args.dt * float(args.steps), sample_stride=1, fixed_dt=True)
            t, phi_t, _g_t, _o_t = integrate_linear_gx(
                G0,
                grid,
                cache,
                params,
                geom,
                time_cfg,
                terms=LinearTerms(),
                mode_method="z_index",
            )
            sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
            gamma, omega, _g, _o, _t_mid = gx_growth_rate_from_phi(
                phi_t, t, sel, navg_fraction=0.5, mode_method="z_index"
            )
            gammas.append(gamma)
            omegas.append(omega)
        scan = type("CycloneScan", (), {"ky": gx_ky, "gamma": np.asarray(gammas), "omega": np.asarray(omegas)})()
    else:
        scan = run_cyclone_scan(
            gx_ky,
            cfg=cfg,
            Nl=args.Nl,
            Nm=args.Nm,
            dt=args.dt,
            steps=args.steps,
            method=args.method,
            solver=args.solver,
            fit_signal="auto",
            diagnostic_norm="gx",
        )

    rel_gamma = (scan.gamma - gx_gamma) / gx_gamma
    rel_omega = (scan.omega - gx_omega) / gx_omega
    table = pd.DataFrame(
        {
            "ky": gx_ky,
            "gamma": scan.gamma,
            "omega": scan.omega,
            "gamma_gx": gx_gamma,
            "omega_gx": gx_omega,
            "rel_gamma": rel_gamma,
            "rel_omega": rel_omega,
        }
    )
    with pd.option_context("display.max_rows", None):
        print(table)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
