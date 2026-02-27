#!/usr/bin/env python3
"""Compare GX linear outputs against SPECTRAX-GK for the Cyclone base case."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from spectraxgk.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate_auto,
    gx_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    CYCLONE_KRYLOV_DEFAULT,
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
    parser.add_argument(
        "--ny",
        type=int,
        default=None,
        help="Full Ny (FFT grid). If omitted, inferred from GX nky.",
    )
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--y0", type=float, default=20.0)
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument(
        "--gx-last",
        action="store_true",
        help="Use GX-style final-step omega/gamma (instead of time-average).",
    )
    parser.add_argument(
        "--gx-fixed-dt",
        action="store_true",
        help="Force fixed dt for GX-style integrator (default: adaptive dt).",
    )
    parser.add_argument(
        "--ky",
        type=str,
        default="",
        help="Comma-separated ky values to compare (default: all from GX output)",
    )
    parser.add_argument(
        "--mode-follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ky continuation (overlap selection) in Krylov scans.",
    )
    parser.add_argument(
        "--krylov-method",
        type=str,
        default=None,
        help="Override Krylov method (power, propagator, shift_invert).",
    )
    parser.add_argument("--krylov-dim", type=int, default=None, help="Override Krylov dimension.")
    parser.add_argument("--krylov-restarts", type=int, default=None, help="Override Krylov restarts.")
    parser.add_argument("--krylov-power-dt", type=float, default=None, help="Override Krylov power dt.")
    parser.add_argument(
        "--krylov-omega-target",
        type=float,
        default=None,
        help="Override omega_target_factor for Krylov.",
    )
    parser.add_argument(
        "--krylov-shift-precond",
        type=str,
        default=None,
        help="Override shift-invert preconditioner.",
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
    if args.ny is None:
        nky = int(len(gx_ky))
        ny = 3 * (nky - 1) + 1
    else:
        ny = int(args.ny)
    cfg = _build_cyclone_cfg(
        ny, ntheta=args.ntheta, nperiod=args.nperiod, y0=args.y0, drift_scale=args.drift_scale
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
            time_cfg = GXTimeConfig(
                dt=args.dt,
                t_max=args.dt * float(args.steps),
                sample_stride=1,
                fixed_dt=args.gx_fixed_dt,
            )
            t, phi_t, gamma_t, omega_t = integrate_linear_gx(
                G0,
                grid,
                cache,
                params,
                geom,
                time_cfg,
                terms=LinearTerms(),
                mode_method="z_index",
            )
            if args.gx_last:
                if gamma_t.size == 0:
                    raise ValueError("GX time integrator returned no growth-rate samples")
                gamma_series = np.asarray(gamma_t)[:, 0, 0]
                omega_series = np.asarray(omega_t)[:, 0, 0]
                gamma = float(gamma_series[-1])
                omega = float(omega_series[-1])
            else:
                sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
                try:
                    gamma, omega, _g, _o, _t_mid = gx_growth_rate_from_phi(
                        phi_t, t, sel, navg_fraction=0.5, mode_method="z_index"
                    )
                except ValueError:
                    print(
                        "gx_growth_rate_from_phi: z_index failed; falling back to mode_method='max'"
                    )
                    try:
                        gamma, omega, _g, _o, _t_mid = gx_growth_rate_from_phi(
                            phi_t, t, sel, navg_fraction=0.5, mode_method="max"
                        )
                    except ValueError:
                        print(
                            "gx_growth_rate_from_phi: max failed; falling back to loglinear fit"
                        )
                        signal = extract_mode_time_series(phi_t, sel, method="max")
                        gamma, omega, *_ = fit_growth_rate_auto(
                            t, signal, window_method="loglinear"
                        )
            gammas.append(gamma)
            omegas.append(omega)
        scan = type("CycloneScan", (), {"ky": gx_ky, "gamma": np.asarray(gammas), "omega": np.asarray(omegas)})()
    else:
        krylov_cfg = None
        if args.solver.lower() == "krylov":
            krylov_cfg = CYCLONE_KRYLOV_DEFAULT
            if args.krylov_method is not None:
                krylov_cfg = replace(krylov_cfg, method=str(args.krylov_method))
            if args.krylov_dim is not None:
                krylov_cfg = replace(krylov_cfg, krylov_dim=int(args.krylov_dim))
            if args.krylov_restarts is not None:
                krylov_cfg = replace(krylov_cfg, restarts=int(args.krylov_restarts))
            if args.krylov_power_dt is not None:
                krylov_cfg = replace(krylov_cfg, power_dt=float(args.krylov_power_dt))
            if args.krylov_omega_target is not None:
                krylov_cfg = replace(krylov_cfg, omega_target_factor=float(args.krylov_omega_target))
            if args.krylov_shift_precond is not None:
                krylov_cfg = replace(krylov_cfg, shift_preconditioner=str(args.krylov_shift_precond))

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
            mode_follow=bool(args.mode_follow),
            krylov_cfg=krylov_cfg,
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
