#!/usr/bin/env python3
"""Compare GS2 linear outputs against SPECTRAX-GK on matching ky points."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from spectraxgk.analysis import ModeSelection, gx_growth_rate_from_phi, select_ky_index
from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
    _apply_gx_hypercollisions,
    _build_initial_condition,
    _electron_only_params,
    _two_species_params,
    run_cyclone_linear,
    run_etg_linear,
    run_kinetic_linear,
)
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    ETGModelConfig,
    GeometryConfig,
    GridConfig,
    KineticElectronBaseCase,
    KineticElectronModelConfig,
    ModelConfig,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.gx_integrators import GXTimeConfig, integrate_linear_gx
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache


def _load_gs2_omega_gamma(
    path: Path,
    *,
    gamma_scale: float,
    omega_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path)
    if "omega_average" not in ds:
        raise ValueError(f"{path} does not contain omega_average")
    omega_avg = np.asarray(ds["omega_average"])
    if omega_avg.ndim != 4 or omega_avg.shape[-1] != 2:
        raise ValueError(f"unexpected omega_average shape in {path}: {omega_avg.shape}")
    ky = np.asarray(ds["ky"])
    # final time, kx=0 branch: ri axis stores (omega, gamma)
    final = omega_avg[-1, :, 0, :]
    omega_ref = final[:, 0] * omega_scale
    gamma_ref = final[:, 1] * gamma_scale
    return ky, gamma_ref, omega_ref


def _build_cyclone_params(cfg: CycloneBaseCase, geom: SAlphaGeometry, Nm: int) -> LinearParams:
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        omega_d_scale=CYCLONE_OMEGA_D_SCALE,
        omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
        rho_star=CYCLONE_RHO_STAR,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    return _apply_gx_hypercollisions(params, nhermite=Nm)


def _run_cyclone_gx(
    *,
    cfg: CycloneBaseCase,
    ky: float,
    Nl: int,
    Nm: int,
    dt: float,
    steps: int,
    navg_fraction: float,
    sample_stride: int,
) -> tuple[float, float]:
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky)
    grid = select_ky_grid(grid_full, ky_index)
    params = _build_cyclone_params(cfg, geom, Nm)
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    init_cfg = getattr(cfg, "init", None)
    G0 = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=init_cfg,
    )
    terms = LinearTerms(bpar=0.0) if getattr(cfg.model, "adiabatic_ions", False) else LinearTerms()
    t, phi_t, _gamma_t, _omega_t = integrate_linear_gx(
        G0,
        grid,
        cache,
        params,
        geom,
        GXTimeConfig(dt=dt, t_max=dt * float(steps), sample_stride=max(1, sample_stride)),
        terms=terms,
        mode_method="z_index",
    )
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=grid.z.size // 2)
    gamma, omega, _g_t, _w_t, _t_mid = gx_growth_rate_from_phi(
        np.asarray(phi_t),
        np.asarray(t),
        sel,
        navg_fraction=navg_fraction,
        mode_method="z_index",
    )
    return float(gamma), float(omega)


def _run_etg_gx(
    *,
    cfg: ETGBaseCase,
    ky: float,
    Nl: int,
    Nm: int,
    dt: float,
    steps: int,
    navg_fraction: float,
    sample_stride: int,
) -> tuple[float, float]:
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky)
    grid = select_ky_grid(grid_full, ky_index)
    if getattr(cfg.model, "adiabatic_ions", False):
        params = _electron_only_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=ETG_OMEGA_D_SCALE,
            omega_star_scale=ETG_OMEGA_STAR_SCALE,
            rho_star=ETG_RHO_STAR,
            nhermite=Nm,
        )
        terms = LinearTerms(bpar=0.0)
    else:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=ETG_OMEGA_D_SCALE,
            omega_star_scale=ETG_OMEGA_STAR_SCALE,
            rho_star=ETG_RHO_STAR,
            nhermite=Nm,
        )
        terms = LinearTerms()
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    init_cfg = getattr(cfg, "init", None)
    G0 = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=init_cfg,
    )
    t, phi_t, _gamma_t, _omega_t = integrate_linear_gx(
        G0,
        grid,
        cache,
        params,
        geom,
        GXTimeConfig(dt=dt, t_max=dt * float(steps), sample_stride=max(1, sample_stride)),
        terms=terms,
        mode_method="z_index",
    )
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=grid.z.size // 2)
    gamma, omega, _g_t, _w_t, _t_mid = gx_growth_rate_from_phi(
        np.asarray(phi_t),
        np.asarray(t),
        sel,
        navg_fraction=navg_fraction,
        mode_method="z_index",
    )
    return float(gamma), float(omega)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gs2-out",
        action="append",
        required=True,
        help="Path to GS2 .out.nc file. Can be passed multiple times.",
    )
    p.add_argument("--case", choices=("cyclone", "etg", "kinetic"), default="cyclone")
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
    # Keep defaults aligned with the SPECTRAX Cyclone/GX-balanced benchmark case.
    p.add_argument("--R-over-LTi", type=float, default=2.49)
    p.add_argument("--R-over-Ln", type=float, default=0.8)
    p.add_argument(
        "--R-over-LTe",
        type=float,
        default=None,
        help="Electron temperature gradient. If omitted: cyclone=0, ETG/kinetic=R-over-LTi.",
    )
    p.add_argument("--nu-i", type=float, default=0.0)
    p.add_argument("--nu-e", type=float, default=0.0)
    p.add_argument("--mass-ratio", type=float, default=3670.0)
    p.add_argument("--Te-over-Ti", type=float, default=1.0)
    p.add_argument("--Ny", type=int, default=16)
    p.add_argument("--Nz", type=int, default=64)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--spectrax-integrator",
        choices=("benchmark", "gx"),
        default="gx",
        help="SPECTRAX path: benchmark wrappers or GX-style RK4 growth extraction.",
    )
    p.add_argument("--spectrax-navg-frac", type=float, default=0.3)
    p.add_argument("--sample-stride", type=int, default=20)
    p.add_argument("--ref-gamma-scale", type=float, default=1.0)
    p.add_argument("--ref-omega-scale", type=float, default=1.0)
    p.add_argument("--fit-signal", choices=("phi", "density"), default="density")
    return p


def main() -> None:
    args = build_parser().parse_args()
    r_over_lte = float(args.R_over_LTi) if args.R_over_LTe is None else float(args.R_over_LTe)
    if args.case == "cyclone":
        base_cfg = CycloneBaseCase(
            grid=GridConfig(Nx=1, Ny=args.Ny, Nz=args.Nz, Lx=62.8, Ly=62.8, ntheta=32, nperiod=2),
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
                R_over_LTe=0.0,
                R_over_Ln=args.R_over_Ln,
                nu_i=args.nu_i,
            ),
        )
        run_linear_fn = run_cyclone_linear
    elif args.case == "etg":
        base_cfg = ETGBaseCase(
            grid=GridConfig(Nx=1, Ny=args.Ny, Nz=args.Nz, Lx=6.28, Ly=6.28, ntheta=32, nperiod=2),
            geometry=GeometryConfig(
                q=args.q,
                s_hat=args.s_hat,
                epsilon=args.epsilon,
                R0=args.R0,
                B0=1.0,
                alpha=0.0,
            ),
            model=ETGModelConfig(
                R_over_LTi=args.R_over_LTi,
                R_over_LTe=r_over_lte,
                R_over_Ln=args.R_over_Ln,
                Te_over_Ti=args.Te_over_Ti,
                mass_ratio=args.mass_ratio,
                nu_i=args.nu_i,
                nu_e=args.nu_e,
                beta=1.0e-5,
                adiabatic_ions=True,
            ),
        )
        run_linear_fn = run_etg_linear
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
                R_over_LTe=r_over_lte,
                R_over_Ln=args.R_over_Ln,
                Te_over_Ti=args.Te_over_Ti,
                mass_ratio=args.mass_ratio,
                nu_i=args.nu_i,
                nu_e=args.nu_e,
                beta=1.0e-5,
            ),
        )
        run_linear_fn = run_kinetic_linear

    rows: list[dict[str, float | str]] = []
    for gs2_file in args.gs2_out:
        path = Path(gs2_file)
        ky_vals, gamma_ref, omega_ref = _load_gs2_omega_gamma(
            path,
            gamma_scale=args.ref_gamma_scale,
            omega_scale=args.ref_omega_scale,
        )
        if args.verbose:
            print(f"[GS2] {path}: {len(ky_vals)} ky points")
        for idx, ky in enumerate(ky_vals):
            if args.spectrax_integrator == "gx" and args.case == "cyclone":
                gamma_sp, omega_sp = _run_cyclone_gx(
                    cfg=replace(base_cfg),
                    ky=float(ky),
                    Nl=args.Nl,
                    Nm=args.Nm,
                    dt=args.dt,
                    steps=args.steps,
                    navg_fraction=args.spectrax_navg_frac,
                    sample_stride=args.sample_stride,
                )
            elif args.spectrax_integrator == "gx" and args.case == "etg":
                gamma_sp, omega_sp = _run_etg_gx(
                    cfg=replace(base_cfg),
                    ky=float(ky),
                    Nl=args.Nl,
                    Nm=args.Nm,
                    dt=args.dt,
                    steps=args.steps,
                    navg_fraction=args.spectrax_navg_frac,
                    sample_stride=args.sample_stride,
                )
            else:
                run_kwargs = {
                    "cfg": replace(base_cfg),
                    "ky_target": float(ky),
                    "Nl": args.Nl,
                    "Nm": args.Nm,
                    "dt": args.dt,
                    "steps": args.steps,
                    "method": args.method,
                    "solver": args.solver,
                }
                if args.case == "etg":
                    run_kwargs["fit_signal"] = args.fit_signal
                result = run_linear_fn(**run_kwargs)
                gamma_sp = float(result.gamma)
                omega_sp = float(result.omega)
            g_ref = float(gamma_ref[idx])
            w_ref = float(omega_ref[idx])
            rel_g = np.nan if g_ref == 0.0 else (gamma_sp - g_ref) / g_ref
            rel_w = np.nan if w_ref == 0.0 else (omega_sp - w_ref) / w_ref
            rows.append(
                {
                    "source": str(path),
                    "ky": float(ky),
                    "gamma_ref": g_ref,
                    "omega_ref": w_ref,
                    "gamma_spectrax": gamma_sp,
                    "omega_spectrax": omega_sp,
                    "rel_gamma": float(rel_g),
                    "rel_omega": float(rel_w),
                }
            )
            if args.verbose:
                print(
                    f"  ky={ky:.6g} gamma={gamma_sp:.6g} omega={omega_sp:.6g} "
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
