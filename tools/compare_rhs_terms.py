#!/usr/bin/env python3
"""Compare term-by-term linear RHS contributions for a single ky case."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from spectraxgk.benchmarks import (
    ETGBaseCase,
    KineticElectronBaseCase,
    TEMBaseCase,
    KBMBaseCase,
    CycloneBaseCase,
    _build_initial_condition,
    _electron_only_params,
    _two_species_params,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
    Kinetic_OMEGA_D_SCALE,
    Kinetic_OMEGA_STAR_SCALE,
    Kinetic_RHO_STAR,
    TEM_OMEGA_D_SCALE,
    TEM_OMEGA_STAR_SCALE,
    TEM_RHO_STAR,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
)
from spectraxgk.config import GridConfig, ETGModelConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearTerms, build_linear_cache
from spectraxgk.terms.config import TermConfig
from spectraxgk.terms.assembly import assemble_rhs_cached


@dataclass(frozen=True)
class CaseInfo:
    name: str
    cfg: object
    params: object
    init_species_index: int


def _build_case(args) -> CaseInfo:
    case = args.case.lower()
    if case == "cyclone":
        cfg = CycloneBaseCase(
            grid=GridConfig(
                Nx=args.Nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=args.Lx,
                Ly=args.Ly,
                boundary=args.boundary,
                y0=args.y0,
                ntheta=args.ntheta,
                nperiod=args.nperiod,
            )
        )
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=CYCLONE_OMEGA_D_SCALE,
            omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
            rho_star=CYCLONE_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=args.Nm,
        )
        return CaseInfo("cyclone", cfg, params, init_species_index=0)
    if case == "etg":
        model = ETGModelConfig(R_over_LTe=args.R_over_LTe, adiabatic_ions=args.adiabatic_ions)
        cfg = ETGBaseCase(
            grid=GridConfig(
                Nx=args.Nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=args.Lx,
                Ly=args.Ly,
                boundary=args.boundary,
                y0=args.y0,
                ntheta=args.ntheta,
                nperiod=args.nperiod,
            ),
            model=model,
        )
        geom = SAlphaGeometry.from_config(cfg.geometry)
        if args.adiabatic_ions:
            params = _electron_only_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=args.Nm,
            )
            init_species_index = 0
        else:
            params = _two_species_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
                nhermite=args.Nm,
            )
            init_species_index = 1
        return CaseInfo("etg", cfg, params, init_species_index)
    if case == "kinetic":
        cfg = KineticElectronBaseCase(
            grid=GridConfig(
                Nx=args.Nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=args.Lx,
                Ly=args.Ly,
                boundary=args.boundary,
                y0=args.y0,
                ntheta=args.ntheta,
                nperiod=args.nperiod,
            )
        )
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=Kinetic_OMEGA_D_SCALE,
            omega_star_scale=Kinetic_OMEGA_STAR_SCALE,
            rho_star=Kinetic_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=args.Nm,
        )
        return CaseInfo("kinetic", cfg, params, init_species_index=1)
    if case == "tem":
        cfg = TEMBaseCase(
            grid=GridConfig(
                Nx=args.Nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=args.Lx,
                Ly=args.Ly,
                boundary=args.boundary,
                y0=args.y0,
                ntheta=args.ntheta,
                nperiod=args.nperiod,
            )
        )
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=TEM_OMEGA_D_SCALE,
            omega_star_scale=TEM_OMEGA_STAR_SCALE,
            rho_star=TEM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=args.Nm,
        )
        return CaseInfo("tem", cfg, params, init_species_index=1)
    if case == "kbm":
        cfg = KBMBaseCase(
            grid=GridConfig(
                Nx=args.Nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=args.Lx,
                Ly=args.Ly,
                boundary=args.boundary,
                y0=args.y0,
                ntheta=args.ntheta,
                nperiod=args.nperiod,
            )
        )
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            beta_override=args.beta,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=args.Nm,
        )
        return CaseInfo("kbm", cfg, params, init_species_index=1)
    raise ValueError(f"Unknown case '{args.case}'")


def _term_config(**weights: float) -> TermConfig:
    return TermConfig(
        streaming=weights.get("streaming", 0.0),
        mirror=weights.get("mirror", 0.0),
        curvature=weights.get("curvature", 0.0),
        gradb=weights.get("gradb", 0.0),
        diamagnetic=weights.get("diamagnetic", 0.0),
        collisions=weights.get("collisions", 0.0),
        hypercollisions=weights.get("hypercollisions", 0.0),
        end_damping=weights.get("end_damping", 0.0),
        apar=weights.get("apar", 0.0),
        bpar=weights.get("bpar", 0.0),
        nonlinear=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="etg", choices=["cyclone", "etg", "kinetic", "tem", "kbm"])
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.015)
    parser.add_argument("--Nl", type=int, default=24)
    parser.add_argument("--Nm", type=int, default=12)
    parser.add_argument("--Nx", type=int, default=1)
    parser.add_argument("--Ny", type=int, default=24)
    parser.add_argument("--Nz", type=int, default=96)
    parser.add_argument("--Lx", type=float, default=62.8)
    parser.add_argument("--Ly", type=float, default=62.8)
    parser.add_argument("--y0", type=float, default=20.0)
    parser.add_argument("--boundary", default="linked")
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--adiabatic-ions", action="store_true")
    parser.add_argument("--R_over_LTe", type=float, default=2.49)
    args = parser.parse_args()

    case = _build_case(args)
    grid_full = build_spectral_grid(case.cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - args.ky)))
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(case.cfg.geometry)
    cache = build_linear_cache(grid, geom, case.params, args.Nl, args.Nm)

    ns = int(np.atleast_1d(np.asarray(case.params.charge_sign)).shape[0])
    G0 = np.zeros((ns, args.Nl, args.Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=args.Nl,
        Nm=args.Nm,
        init_cfg=case.cfg.init,
    )
    G0[case.init_species_index] = np.asarray(G0_single, dtype=np.complex64)
    G0_jax = jnp.asarray(G0)

    term_names = [
        "streaming",
        "mirror",
        "curvature",
        "gradb",
        "diamagnetic",
        "collisions",
        "hypercollisions",
        "end_damping",
        "apar",
        "bpar",
    ]
    full_cfg = _term_config(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=1.0,
        hypercollisions=1.0,
        end_damping=1.0,
        apar=1.0,
        bpar=1.0,
    )
    dG_total, _fields = assemble_rhs_cached(G0_jax, cache, case.params, terms=full_cfg)
    total_norm = float(np.linalg.norm(np.asarray(dG_total)))
    total_max = float(np.max(np.abs(np.asarray(dG_total))))

    print(f"Case: {case.name} ky={float(grid.ky[0]):.6g} Nl={args.Nl} Nm={args.Nm}")
    print(f"Total RHS norm: {total_norm:.6e}  max|RHS|: {total_max:.6e}")
    print("term                    norm            max|RHS|        frac(norm)")
    for name in term_names:
        cfg = _term_config(**{name: 1.0})
        dG, _ = assemble_rhs_cached(G0_jax, cache, case.params, terms=cfg)
        dG_np = np.asarray(dG)
        norm = float(np.linalg.norm(dG_np))
        max_abs = float(np.max(np.abs(dG_np)))
        frac = norm / total_norm if total_norm != 0.0 else np.nan
        print(f"{name:18s} {norm:12.5e} {max_abs:12.5e} {frac:12.5e}")


if __name__ == "__main__":
    main()
