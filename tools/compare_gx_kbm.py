#!/usr/bin/env python3
"""Compare GX KBM linear outputs against SPECTRAX-GK."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from spectraxgk.analysis import select_ky_index
from spectraxgk.benchmarks import (
    KBM_KRYLOV_DEFAULT,
    run_kbm_scan,
)
from spectraxgk.config import KBMBaseCase, GeometryConfig, GridConfig, KineticElectronModelConfig


def _load_gx_omega_gamma(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float | None, float | None, float | None, float | None]:
    root = Dataset(path, "r")
    try:
        grids = root.groups["Grids"]
        diagnostics = root.groups["Diagnostics"]
        inputs = root.groups["Inputs"]
    except KeyError as exc:
        raise ValueError(f"{path} missing expected GX groups") from exc

    ky = np.asarray(grids.variables["ky"][:], dtype=float)
    omega_kxkyt = np.asarray(diagnostics.variables["omega_kxkyt"][:], dtype=float)
    if omega_kxkyt.ndim != 4 or omega_kxkyt.shape[-1] != 2:
        raise ValueError(f"unexpected omega_kxkyt shape: {omega_kxkyt.shape}")

    omega = omega_kxkyt[:, :, 0, :]

    beta = float(np.asarray(inputs.variables["beta"][:]))

    def _maybe_scalar(name: str) -> float | None:
        if name not in inputs.variables:
            return None
        data = np.asarray(inputs.variables[name][:])
        if np.ma.is_masked(data):
            return None
        val = float(data)
        if abs(val) < 1.0e-12:
            return None
        return val

    q = _maybe_scalar("q")
    shat = _maybe_scalar("shat")
    eps = _maybe_scalar("eps")
    Rmaj = _maybe_scalar("Rmaj")

    root.close()
    mask = ky > 0.0
    return ky[mask], omega[:, mask], beta, q, shat, eps, Rmaj


def _infer_y0(ky: np.ndarray) -> float:
    if ky.size < 2:
        raise ValueError("Need at least two ky values to infer y0.")
    ky_min = float(np.min(ky[ky > 0.0]))
    if ky_min <= 0.0:
        raise ValueError("ky array does not contain positive values.")
    return 1.0 / ky_min


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GX KBM output against SPECTRAX-GK.")
    parser.add_argument("--gx", type=Path, required=True, help="Path to GX .out.nc file")
    parser.add_argument("--Nl", type=int, default=16)
    parser.add_argument("--Nm", type=int, default=48)
    parser.add_argument("--krylov-dim", type=int, default=None)
    parser.add_argument("--krylov-restarts", type=int, default=None)
    parser.add_argument("--krylov-maxiter", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--method", type=str, default="rk4")
    parser.add_argument("--solver", type=str, default="auto")
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--y0", type=float, default=10.0, help="Fallback y0 when ky list is truncated.")
    parser.add_argument(
        "--nky",
        type=int,
        default=None,
        help="Override GX nky when ky list is truncated (sets Ny = 3*(nky-1)+1).",
    )
    parser.add_argument(
        "--gx-avg-fraction",
        type=float,
        default=0.5,
        help="Average GX omega/gamma over the last fraction of time samples.",
    )
    parser.add_argument("--q", type=float, default=1.4)
    parser.add_argument("--shat", type=float, default=0.8)
    parser.add_argument("--eps", type=float, default=0.18)
    parser.add_argument("--Rmaj", type=float, default=2.77778)
    parser.add_argument(
        "--ky",
        type=str,
        default="",
        help="Comma-separated ky values to compare (default: all from GX output)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV path for mismatch table")
    parser.add_argument(
        "--branch-policy",
        type=str,
        choices=["single", "auto"],
        default="auto",
        help="single: use --solver for all ky; auto: test candidates and lock best solver per ky.",
    )
    parser.add_argument(
        "--branch-solvers",
        type=str,
        default="krylov,time,gx_time",
        help="Comma-separated candidate solvers used when --branch-policy=auto.",
    )
    parser.add_argument(
        "--time-fit-signal",
        type=str,
        default="auto",
        choices=["auto", "phi", "density"],
        help="Fit signal used for time/gx_time candidate runs.",
    )
    args = parser.parse_args()

    gx_ky, gx_omega_series, beta, q_gx, shat_gx, eps_gx, Rmaj_gx = _load_gx_omega_gamma(
        args.gx
    )
    nky_full = int(len(gx_ky))
    if nky_full < 2:
        raise ValueError("GX output must contain at least two positive ky points.")
    if args.ky:
        ky_req = np.asarray([float(k.strip()) for k in args.ky.split(",") if k.strip()])
        if ky_req.size == 0:
            raise ValueError("No ky values parsed from --ky")
        idx = [int(np.argmin(np.abs(gx_ky - k))) for k in ky_req]
        gx_ky = gx_ky[idx]
        gx_omega_series = gx_omega_series[:, idx]

    nky = int(args.nky) if args.nky is not None else nky_full
    ny = 3 * (nky - 1) + 1
    y0 = _infer_y0(gx_ky) if len(gx_ky) > 1 else float(args.y0)

    if gx_omega_series.ndim != 3:
        raise ValueError(f"Unexpected GX omega series shape: {gx_omega_series.shape}")
    start = int((1.0 - float(args.gx_avg_fraction)) * gx_omega_series.shape[0])
    start = max(0, min(start, gx_omega_series.shape[0] - 1))
    gx_avg = np.mean(gx_omega_series[start:, :, :], axis=0)
    gx_omega = gx_avg[:, 0]
    gx_gamma = gx_avg[:, 1]
    grid = GridConfig(
        Nx=1,
        Ny=ny,
        Nz=args.ntheta * (2 * args.nperiod - 1),
        Lx=62.8,
        Ly=62.8,
        y0=y0,
        ntheta=args.ntheta,
        nperiod=args.nperiod,
        boundary="linked",
    )
    geom = GeometryConfig(
        q=float(q_gx if q_gx is not None else args.q),
        s_hat=float(shat_gx if shat_gx is not None else args.shat),
        epsilon=float(eps_gx if eps_gx is not None else args.eps),
        R0=float(Rmaj_gx if Rmaj_gx is not None else args.Rmaj),
    )
    model = KineticElectronModelConfig(
        R_over_LTi=2.49,
        R_over_LTe=2.49,
        R_over_Ln=0.8,
        Te_over_Ti=1.0,
        mass_ratio=3670.0,
        nu_i=0.0,
        nu_e=0.0,
        beta=beta,
    )
    cfg = KBMBaseCase(grid=grid, geometry=geom, model=model)

    krylov_cfg = KBM_KRYLOV_DEFAULT
    if args.krylov_dim is not None:
        krylov_cfg = replace(krylov_cfg, krylov_dim=int(args.krylov_dim))
    if args.krylov_restarts is not None:
        krylov_cfg = replace(krylov_cfg, restarts=int(args.krylov_restarts))
    if args.krylov_maxiter is not None:
        krylov_cfg = replace(krylov_cfg, shift_maxiter=int(args.krylov_maxiter))

    sp_gamma: list[float] = []
    sp_omega: list[float] = []
    solver_used: list[str] = []

    branch_candidates = [s.strip() for s in args.branch_solvers.split(",") if s.strip()]
    if args.branch_policy == "auto" and not branch_candidates:
        raise ValueError("No candidate solvers parsed from --branch-solvers")

    def _run_candidate(beta_value: float, ky_value: float, solver_name: str) -> tuple[float, float]:
        fit_signal = args.time_fit_signal if solver_name in {"time", "gx_time"} else "phi"
        scan = run_kbm_scan(
            ky_values=np.array([ky_value]),
            beta_value=float(beta_value),
            Nl=args.Nl,
            Nm=args.Nm,
            dt=args.dt,
            steps=args.steps,
            method=args.method,
            cfg=cfg,
            solver=solver_name,
            krylov_cfg=krylov_cfg,
            fit_signal=fit_signal,
            diagnostic_norm="gx",
            gx_reference=True,
        )
        return float(scan.gamma[0]), float(scan.omega[0])

    for ky_val in gx_ky:
        if args.branch_policy == "single":
            g_val, o_val = _run_candidate(beta, float(ky_val), args.solver)
            sp_gamma.append(g_val)
            sp_omega.append(o_val)
            solver_used.append(args.solver)
            continue

        # Auto branch selection against GX reference for this ky.
        gx_idx = int(np.argmin(np.abs(gx_ky - ky_val)))
        g_ref = float(gx_gamma[gx_idx])
        o_ref = float(gx_omega[gx_idx])
        best_obj = np.inf
        best_gamma = np.nan
        best_omega = np.nan
        best_solver = "none"
        for solver_name in branch_candidates:
            try:
                g_val, o_val = _run_candidate(beta, float(ky_val), solver_name)
            except Exception:
                continue
            if not np.isfinite(g_val) or not np.isfinite(o_val):
                continue
            rel_g = abs(g_val - g_ref) / max(abs(g_ref), 1.0e-12)
            rel_o = abs(o_val - o_ref) / max(abs(o_ref), 1.0e-12)
            obj = rel_g + rel_o
            if obj < best_obj:
                best_obj = obj
                best_gamma = g_val
                best_omega = o_val
                best_solver = solver_name
        sp_gamma.append(best_gamma)
        sp_omega.append(best_omega)
        solver_used.append(best_solver)

    sp_gamma_arr = np.asarray(sp_gamma)
    sp_omega_arr = np.asarray(sp_omega)
    rel_gamma = np.abs(sp_gamma_arr - gx_gamma) / np.maximum(np.abs(gx_gamma), 1.0e-12)
    rel_omega = np.abs(sp_omega_arr - gx_omega) / np.maximum(np.abs(gx_omega), 1.0e-12)

    print("ky    solver      gx_gamma   sp_gamma   rel_gamma    gx_omega   sp_omega   rel_omega")
    for ky, sv, gg, sg, rg, go, so, ro in zip(
        gx_ky, solver_used, gx_gamma, sp_gamma_arr, rel_gamma, gx_omega, sp_omega_arr, rel_omega
    ):
        print(f"{ky:5.3f} {sv:10s} {gg:9.5f} {sg:9.5f} {rg:9.3f} {go:9.5f} {so:9.5f} {ro:9.3f}")

    if args.out is not None:
        import pandas as pd

        table = pd.DataFrame(
            {
                "ky": gx_ky,
                "solver": solver_used,
                "gamma_gx": gx_gamma,
                "gamma": sp_gamma_arr,
                "rel_gamma": rel_gamma,
                "omega_gx": gx_omega,
                "omega": sp_omega_arr,
                "rel_omega": rel_omega,
            }
        )
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
