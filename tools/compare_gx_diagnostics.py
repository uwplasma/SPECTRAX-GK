#!/usr/bin/env python3
"""Compare GX diagnostic time series against SPECTRAX-GK for a single ky."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from spectraxgk.analysis import gx_growth_rate_from_phi, select_ky_index, ModeSelection
from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    _apply_gx_hypercollisions,
    _build_initial_condition,
    _midplane_index,
)
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.gx_integrators import GXTimeConfig, integrate_linear_gx_diagnostics
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache


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


def _read_diag_series(group, name: str, ky_idx: int) -> np.ndarray:
    var = group.variables[name][:]
    if var.ndim == 3:
        # time, species, ky
        return np.sum(var[:, :, ky_idx], axis=1)
    if var.ndim == 2:
        return np.sum(var[:, :], axis=1)
    raise ValueError(f"Unexpected shape for {name}: {var.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GX diagnostics vs SPECTRAX-GK.")
    parser.add_argument("--gx", type=Path, required=True, help="Path to GX .out.nc file")
    parser.add_argument("--ky", type=float, default=0.55, help="ky to compare")
    parser.add_argument("--Nl", type=int, default=16)
    parser.add_argument("--Nm", type=int, default=48)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()

    root = Dataset(args.gx, "r")
    ky_all = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)
    mask = ky_all > 0.0
    ky_all = ky_all[mask]
    ky_idx = int(np.argmin(np.abs(ky_all - args.ky)))
    ky_val = float(ky_all[ky_idx])

    diag = root.groups["Diagnostics"]
    gx_Wg = _read_diag_series(diag, "Wg_kyst", ky_idx)
    gx_Wphi = _read_diag_series(diag, "Wphi_kyst", ky_idx)
    gx_Wapar = _read_diag_series(diag, "Wapar_kyst", ky_idx)
    gx_heat = _read_diag_series(diag, "HeatFlux_kyst", ky_idx)
    gx_pflux = _read_diag_series(diag, "ParticleFlux_kyst", ky_idx)
    root.close()

    ny = 3 * (len(ky_all) - 1) + 1
    cfg = _build_cyclone_cfg(ny, ntheta=32, nperiod=2, y0=20.0, drift_scale=1.0)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_val)
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
        grid, geom, ky_index=0, kx_index=0, Nl=args.Nl, Nm=args.Nm, init_cfg=cfg.init
    )
    gx_time_cfg = GXTimeConfig(dt=args.dt, t_max=args.dt * args.steps, sample_stride=1, fixed_dt=True)

    t, phi_t, _g_t, _o_t, diag_spec = integrate_linear_gx_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        gx_time_cfg,
        terms=LinearTerms(),
        mode_method="z_index",
    )

    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    gamma, omega, *_ = gx_growth_rate_from_phi(
        phi_t, t, sel, navg_fraction=0.5, mode_method="z_index"
    )

    gx_energy = gx_Wg + gx_Wphi + gx_Wapar
    sp_energy = np.asarray(diag_spec.energy_t)

    def _drift(arr: np.ndarray) -> float:
        if arr.size < 2:
            return 0.0
        return float((arr[-1] - arr[0]) / max(abs(arr[0]), 1.0e-12))

    print(f"ky={ky_val:.3f} gamma={gamma:.6e} omega={omega:.6e}")
    print(f"GX energy drift: {_drift(gx_energy):.3e}")
    print(f"SPECTRAX energy drift: {_drift(sp_energy):.3e}")
    print(f"GX heat flux avg: {np.mean(gx_heat):.6e}")
    print(f"SPECTRAX heat flux avg: {np.mean(diag_spec.heat_flux_t):.6e}")
    print(f"GX particle flux avg: {np.mean(gx_pflux):.6e}")
    print(f"SPECTRAX particle flux avg: {np.mean(diag_spec.particle_flux_t):.6e}")


if __name__ == "__main__":
    main()
