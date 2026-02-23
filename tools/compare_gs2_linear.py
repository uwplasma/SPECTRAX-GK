#!/usr/bin/env python3
"""Compare GS2 linear outputs against SPECTRAX-GK on matching ky points."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from spectraxgk.benchmarks import run_cyclone_linear, run_kinetic_linear
from spectraxgk.config import (
    CycloneBaseCase,
    GeometryConfig,
    GridConfig,
    KineticElectronBaseCase,
    KineticElectronModelConfig,
    ModelConfig,
)


def _load_gs2_omega_gamma(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path)
    if "omega_average" not in ds:
        raise ValueError(f"{path} does not contain omega_average")
    omega_avg = np.asarray(ds["omega_average"])
    if omega_avg.ndim != 4 or omega_avg.shape[-1] != 2:
        raise ValueError(f"unexpected omega_average shape in {path}: {omega_avg.shape}")
    ky = np.asarray(ds["ky"])
    # final time, kx=0 branch: ri axis stores (omega, gamma)
    final = omega_avg[-1, :, 0, :]
    omega_ref = final[:, 0]
    gamma_ref = final[:, 1]
    return ky, gamma_ref, omega_ref


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gs2-out",
        action="append",
        required=True,
        help="Path to GS2 .out.nc file. Can be passed multiple times.",
    )
    p.add_argument("--case", choices=("cyclone", "kinetic"), default="cyclone")
    p.add_argument("--out-csv", type=Path, default=Path("docs/_static/gs2_linear_mismatch.csv"))
    p.add_argument("--solver", default="krylov", choices=("krylov", "time"))
    p.add_argument("--Nl", type=int, default=16)
    p.add_argument("--Nm", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--method", default="imex2")
    p.add_argument("--q", type=float, default=1.44)
    p.add_argument("--s-hat", type=float, default=0.8)
    p.add_argument("--epsilon", type=float, default=0.18)
    p.add_argument("--R0", type=float, default=2.77778)
    p.add_argument("--R-over-LTi", type=float, default=6.9)
    p.add_argument("--R-over-Ln", type=float, default=2.2)
    p.add_argument("--R-over-LTe", type=float, default=0.0)
    p.add_argument("--nu-i", type=float, default=0.0)
    p.add_argument("--nu-e", type=float, default=0.0)
    p.add_argument("--mass-ratio", type=float, default=3670.0)
    p.add_argument("--Te-over-Ti", type=float, default=1.0)
    p.add_argument("--Ny", type=int, default=16)
    p.add_argument("--Nz", type=int, default=64)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.case == "cyclone":
        base_cfg = CycloneBaseCase(
            grid=GridConfig(Nx=1, Ny=args.Ny, Nz=args.Nz, Lx=62.8, Ly=62.8, ntheta=12, nperiod=1),
            geometry=GeometryConfig(
                q=args.q,
                s_hat=args.s_hat,
                epsilon=args.epsilon,
                R0=args.R0,
                B0=1.0,
                alpha=0.0,
            ),
            model=ModelConfig(
                R_over_LTi=args.R_over_LTi,
                R_over_LTe=args.R_over_LTe,
                R_over_Ln=args.R_over_Ln,
                nu_i=args.nu_i,
            ),
        )
    else:
        base_cfg = KineticElectronBaseCase(
            grid=GridConfig(Nx=1, Ny=args.Ny, Nz=args.Nz, Lx=62.8, Ly=62.8, y0=10.0, ntheta=32, nperiod=2),
            geometry=GeometryConfig(
                q=args.q,
                s_hat=args.s_hat,
                epsilon=args.epsilon,
                R0=args.R0,
                B0=1.0,
                alpha=0.0,
            ),
            model=KineticElectronModelConfig(
                R_over_LTi=args.R_over_LTi,
                R_over_LTe=args.R_over_LTe,
                R_over_Ln=args.R_over_Ln,
                Te_over_Ti=args.Te_over_Ti,
                mass_ratio=args.mass_ratio,
                nu_i=args.nu_i,
                nu_e=args.nu_e,
                beta=1.0e-5,
            ),
        )

    rows: list[dict[str, float | str]] = []
    for gs2_file in args.gs2_out:
        path = Path(gs2_file)
        ky_vals, gamma_ref, omega_ref = _load_gs2_omega_gamma(path)
        if args.verbose:
            print(f"[GS2] {path}: {len(ky_vals)} ky points")
        for idx, ky in enumerate(ky_vals):
            if args.case == "cyclone":
                result = run_cyclone_linear(
                    cfg=replace(base_cfg),
                    ky_target=float(ky),
                    Nl=args.Nl,
                    Nm=args.Nm,
                    dt=args.dt,
                    steps=args.steps,
                    method=args.method,
                    solver=args.solver,
                )
            else:
                result = run_kinetic_linear(
                    cfg=replace(base_cfg),
                    ky_target=float(ky),
                    Nl=args.Nl,
                    Nm=args.Nm,
                    dt=args.dt,
                    steps=args.steps,
                    method=args.method,
                    solver=args.solver,
                )
            g_ref = float(gamma_ref[idx])
            w_ref = float(omega_ref[idx])
            rel_g = np.nan if g_ref == 0.0 else (result.gamma - g_ref) / g_ref
            rel_w = np.nan if w_ref == 0.0 else (result.omega - w_ref) / w_ref
            rows.append(
                {
                    "source": str(path),
                    "ky": float(ky),
                    "gamma_ref": g_ref,
                    "omega_ref": w_ref,
                    "gamma_spectrax": float(result.gamma),
                    "omega_spectrax": float(result.omega),
                    "rel_gamma": float(rel_g),
                    "rel_omega": float(rel_w),
                }
            )
            if args.verbose:
                print(
                    f"  ky={ky:.6g} gamma={result.gamma:.6g} omega={result.omega:.6g} "
                    f"| ref gamma={g_ref:.6g} omega={w_ref:.6g}"
                )

    df = pd.DataFrame(rows).sort_values(["source", "ky"]).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"saved mismatch table: {args.out_csv}")
    if not df.empty:
        print(
            "mean(|rel_gamma|)={:.3%} max(|rel_gamma|)={:.3%} "
            "mean(|rel_omega|)={:.3%} max(|rel_omega|)={:.3%}".format(
                float(np.nanmean(np.abs(df["rel_gamma"]))),
                float(np.nanmax(np.abs(df["rel_gamma"]))),
                float(np.nanmean(np.abs(df["rel_omega"]))),
                float(np.nanmax(np.abs(df["rel_omega"]))),
            )
        )


if __name__ == "__main__":
    main()
