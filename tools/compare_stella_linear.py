#!/usr/bin/env python3
"""Compare stella linear outputs against SPECTRAX-GK on matching ky points."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
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


def _load_stella_omega_gamma(
    path: Path,
    navg_frac: float,
    *,
    gamma_scale: float,
    omega_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path)
    if "omega" not in ds:
        raise ValueError(f"{path} does not contain omega")
    if "ky" not in ds:
        raise ValueError(f"{path} does not contain ky")

    omega_da = ds["omega"]
    omega_arr = np.asarray(omega_da)
    dims = list(omega_da.dims)

    if "ri" in dims:
        comp_axis = dims.index("ri")
    else:
        size2_axes = [i for i, s in enumerate(omega_arr.shape) if s == 2]
        if not size2_axes:
            raise ValueError(f"could not infer omega component axis in {path} with shape {omega_arr.shape}")
        comp_axis = size2_axes[-1]
    omega_arr = np.moveaxis(omega_arr, comp_axis, -1)
    dims = [d for i, d in enumerate(dims) if i != comp_axis] + ["ri"]

    if "t" not in dims:
        raise ValueError(f"could not infer time axis for omega in {path}")
    time_axis = dims.index("t")

    if "ky" not in dims:
        raise ValueError(f"could not infer ky axis for omega in {path}")
    ky_axis = dims.index("ky")

    kx_idx = 0
    if "kx" in dims:
        kx_vals = np.asarray(ds["kx"])
        kx_axis = dims.index("kx")
        kx_idx = int(np.argmin(np.abs(kx_vals)))
        omega_arr = np.take(omega_arr, kx_idx, axis=kx_axis)
        dims.pop(kx_axis)
        if ky_axis > kx_axis:
            ky_axis -= 1
        if time_axis > kx_axis:
            time_axis -= 1

    if omega_arr.ndim != 3:
        raise ValueError(
            f"unexpected omega rank after kx selection in {path}: shape={omega_arr.shape}, dims={dims}"
        )

    ky_vals = np.asarray(ds["ky"], dtype=float)
    gamma_ref = np.zeros_like(ky_vals, dtype=float)
    omega_ref = np.zeros_like(ky_vals, dtype=float)

    for j in range(ky_vals.size):
        time_ky = np.take(omega_arr, j, axis=ky_axis)
        # time_ky shape: (time, 2)
        if time_axis != 0:
            time_ky = np.moveaxis(time_ky, time_axis, 0)
        omega_t = np.asarray(time_ky[:, 0], dtype=float)
        gamma_t = np.asarray(time_ky[:, 1], dtype=float)
        finite = np.isfinite(omega_t) & np.isfinite(gamma_t)
        if not np.any(finite):
            omega_ref[j] = np.nan
            gamma_ref[j] = np.nan
            continue
        omega_f = omega_t[finite]
        gamma_f = gamma_t[finite]
        n = max(1, int(np.floor(navg_frac * omega_f.size)))
        omega_ref[j] = float(np.mean(omega_f[-n:])) * omega_scale
        gamma_ref[j] = float(np.mean(gamma_f[-n:])) * gamma_scale

    _ = kx_idx  # keep explicit for readability/debugging if needed later
    return ky_vals, gamma_ref, omega_ref


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
    omega_d_scale: float,
    omega_star_scale: float,
    hypercollisions: float,
) -> tuple[float, float]:
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky)
    grid = select_ky_grid(grid_full, ky_index)
    if getattr(cfg.model, "adiabatic_ions", False):
        params = _electron_only_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=omega_d_scale,
            omega_star_scale=omega_star_scale,
            rho_star=ETG_RHO_STAR,
            nhermite=Nm,
        )
        terms = LinearTerms(apar=0.0, bpar=0.0, hypercollisions=hypercollisions)
    else:
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=omega_d_scale,
            omega_star_scale=omega_star_scale,
            rho_star=ETG_RHO_STAR,
            nhermite=Nm,
        )
        terms = LinearTerms(apar=0.0, bpar=0.0, hypercollisions=hypercollisions)
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
    charge = np.atleast_1d(np.asarray(params.charge_sign))
    ns = int(charge.size)
    electron_index = int(np.argmin(charge))
    G0_species = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0_species[electron_index] = np.asarray(G0, dtype=np.complex64)
    G0 = jnp.asarray(G0_species)
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
        "--stella-out",
        action="append",
        required=True,
        help="Path to stella .out.nc file. Can be passed multiple times.",
    )
    p.add_argument("--case", choices=("cyclone", "etg", "kinetic"), default="cyclone")
    p.add_argument(
        "--out-csv", type=Path, default=Path("docs/_static/stella_linear_mismatch.csv")
    )
    p.add_argument("--solver", default="krylov", choices=("krylov", "time"))
    p.add_argument("--Nl", type=int, default=16)
    p.add_argument("--Nm", type=int, default=8)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--method", default="imex2")
    p.add_argument("--q", type=float, default=1.4)
    p.add_argument("--s-hat", type=float, default=0.8)
    p.add_argument("--epsilon", type=float, default=0.18)
    p.add_argument("--R0", type=float, default=2.77778)
    p.add_argument("--R-over-LTi", type=float, default=2.49)
    p.add_argument("--R-over-Ln", type=float, default=0.8)
    p.add_argument("--R-over-Ln-i", type=float, default=None)
    p.add_argument("--R-over-Ln-e", type=float, default=None)
    p.add_argument(
        "--etg-adiabatic-ions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use adiabatic ions in ETG runs. Default false for GS2/stella two-species alignment.",
    )
    p.add_argument(
        "--etg-ion-R-over-LTi",
        type=float,
        default=None,
        help="Ion temperature gradient for ETG two-species runs; defaults to 0 when ions are kinetic.",
    )
    p.add_argument("--etg-omega-d-scale", type=float, default=0.4)
    p.add_argument("--etg-omega-star-scale", type=float, default=0.8)
    p.add_argument("--etg-hypercollisions", type=float, default=1.0)
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
    p.add_argument("--beta", type=float, default=1.0e-5)
    p.add_argument("--stella-navg-frac", type=float, default=0.3)
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
    p.add_argument("--verbose", action="store_true")
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
        ln_i = args.R_over_Ln_i
        ln_e = args.R_over_Ln_e
        ion_r_over_lti = args.etg_ion_R_over_LTi
        if not args.etg_adiabatic_ions:
            if ln_i is None:
                ln_i = 0.0
            if ln_e is None:
                ln_e = args.R_over_Ln
            if ion_r_over_lti is None:
                ion_r_over_lti = 0.0
        if ion_r_over_lti is None:
            ion_r_over_lti = args.R_over_LTi
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
                R_over_LTi=float(ion_r_over_lti),
                R_over_LTe=r_over_lte,
                R_over_Ln=args.R_over_Ln,
                R_over_Lni=ln_i,
                R_over_Lne=ln_e,
                Te_over_Ti=args.Te_over_Ti,
                mass_ratio=args.mass_ratio,
                nu_i=args.nu_i,
                nu_e=args.nu_e,
                beta=args.beta,
                adiabatic_ions=bool(args.etg_adiabatic_ions),
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
                beta=args.beta,
            ),
        )
        run_linear_fn = run_kinetic_linear

    rows: list[dict[str, float | str]] = []
    for stella_file in args.stella_out:
        path = Path(stella_file)
        ky_vals, gamma_ref, omega_ref = _load_stella_omega_gamma(
            path,
            args.stella_navg_frac,
            gamma_scale=args.ref_gamma_scale,
            omega_scale=args.ref_omega_scale,
        )
        if args.verbose:
            print(f"[stella] {path}: {len(ky_vals)} ky points")
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
                    omega_d_scale=args.etg_omega_d_scale,
                    omega_star_scale=args.etg_omega_star_scale,
                    hypercollisions=args.etg_hypercollisions,
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
                    geom = SAlphaGeometry.from_config(base_cfg.geometry)
                    if bool(args.etg_adiabatic_ions):
                        params = _electron_only_params(
                            base_cfg.model,
                            kpar_scale=float(geom.gradpar()),
                            omega_d_scale=args.etg_omega_d_scale,
                            omega_star_scale=args.etg_omega_star_scale,
                            rho_star=ETG_RHO_STAR,
                            nhermite=args.Nm,
                        )
                        etg_terms = LinearTerms(
                            apar=0.0,
                            bpar=0.0,
                            hypercollisions=args.etg_hypercollisions,
                        )
                    else:
                        params = _two_species_params(
                            base_cfg.model,
                            kpar_scale=float(geom.gradpar()),
                            omega_d_scale=args.etg_omega_d_scale,
                            omega_star_scale=args.etg_omega_star_scale,
                            rho_star=ETG_RHO_STAR,
                            nhermite=args.Nm,
                        )
                        etg_terms = LinearTerms(
                            apar=0.0,
                            bpar=0.0,
                            hypercollisions=args.etg_hypercollisions,
                        )
                    run_kwargs["params"] = params
                    run_kwargs["terms"] = etg_terms
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
