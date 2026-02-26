#!/usr/bin/env python3
"""Dump term-by-term RHS contributions for a single ky point."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    Kinetic_OMEGA_D_SCALE,
    Kinetic_OMEGA_STAR_SCALE,
    Kinetic_RHO_STAR,
    TEM_OMEGA_D_SCALE,
    TEM_OMEGA_STAR_SCALE,
    TEM_RHO_STAR,
    CycloneBaseCase,
    ETGBaseCase,
    KBMBaseCase,
    KineticElectronBaseCase,
    TEMBaseCase,
    _apply_gx_hypercollisions,
    _build_initial_condition,
    _electron_only_params,
    _two_species_params,
)
from spectraxgk.config import ETGModelConfig, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache
from spectraxgk.terms.assembly import assemble_rhs_terms_cached
from spectraxgk.terms.config import TermConfig


def _case_config(name: str, args) -> tuple[object, object, int, float, float, float]:
    case = name.lower()
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
        geom = SAlphaGeometry.from_config(replace(cfg.geometry, drift_scale=args.drift_scale))
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
        params = _apply_gx_hypercollisions(params, nhermite=args.Nm)
        return cfg, params, 0, CYCLONE_OMEGA_D_SCALE, CYCLONE_OMEGA_STAR_SCALE, CYCLONE_RHO_STAR
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
        geom = SAlphaGeometry.from_config(replace(cfg.geometry, drift_scale=args.drift_scale))
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
        return cfg, params, init_species_index, ETG_OMEGA_D_SCALE, ETG_OMEGA_STAR_SCALE, ETG_RHO_STAR
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
        geom = SAlphaGeometry.from_config(replace(cfg.geometry, drift_scale=args.drift_scale))
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
        return cfg, params, 1, Kinetic_OMEGA_D_SCALE, Kinetic_OMEGA_STAR_SCALE, Kinetic_RHO_STAR
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
        geom = SAlphaGeometry.from_config(replace(cfg.geometry, drift_scale=args.drift_scale))
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
        return cfg, params, 1, TEM_OMEGA_D_SCALE, TEM_OMEGA_STAR_SCALE, TEM_RHO_STAR
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
        geom = SAlphaGeometry.from_config(replace(cfg.geometry, drift_scale=args.drift_scale))
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
            nhermite=args.Nm,
        )
        return cfg, params, 0, KBM_OMEGA_D_SCALE, KBM_OMEGA_STAR_SCALE, KBM_RHO_STAR
    raise ValueError(f"Unknown case '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=str, default="cyclone")
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nx", type=int, default=1)
    parser.add_argument("--Ny", type=int, default=24)
    parser.add_argument("--Nz", type=int, default=96)
    parser.add_argument("--Lx", type=float, default=62.8)
    parser.add_argument("--Ly", type=float, default=62.8)
    parser.add_argument("--boundary", type=str, default="linked")
    parser.add_argument("--y0", type=float, default=20.0)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--Nl", type=int, default=48)
    parser.add_argument("--Nm", type=int, default=16)
    parser.add_argument("--drift-scale", type=float, default=1.0)
    parser.add_argument("--adiabatic-ions", action="store_true")
    parser.add_argument("--R_over_LTe", type=float, default=6.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg, params, _init_species_index, *_ = _case_config(args.case, args)
    geom = SAlphaGeometry.from_config(replace(cfg.geometry, drift_scale=args.drift_scale))
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(args.ky))))
    grid = select_ky_grid(grid_full, ky_index)

    G0 = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=args.Nl,
        Nm=args.Nm,
        init_cfg=cfg.init,
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    term_cfg = TermConfig()
    rhs_total, fields, contrib = assemble_rhs_terms_cached(G0, cache, params, terms=term_cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        rhs_total=np.asarray(rhs_total),
        phi=np.asarray(fields.phi),
        apar=np.asarray(fields.apar) if fields.apar is not None else None,
        bpar=np.asarray(fields.bpar) if fields.bpar is not None else None,
        streaming=np.asarray(contrib["streaming"]),
        mirror=np.asarray(contrib["mirror"]),
        curvature=np.asarray(contrib["curvature"]),
        gradb=np.asarray(contrib["gradb"]),
        diamagnetic=np.asarray(contrib["diamagnetic"]),
        collisions=np.asarray(contrib["collisions"]),
        hypercollisions=np.asarray(contrib["hypercollisions"]),
        end_damping=np.asarray(contrib["end_damping"]),
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
