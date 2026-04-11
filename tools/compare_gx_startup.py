#!/usr/bin/env python3
"""Compare a GX startup field-solve dump against SPECTRAX initial conditions."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from tools.compare_gx_rhs_terms import _infer_y0, _load_bin, _load_field, _load_shape, _reshape_gx, _summary
from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    CycloneBaseCase,
    KBMBaseCase,
    _apply_gx_hypercollisions,
    _build_initial_condition,
    _two_species_params,
)
from spectraxgk.config import GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.terms.assembly import compute_fields_cached
from spectraxgk.terms.config import TermConfig


def _select_ky_block(arr: np.ndarray, ky_idx: int) -> np.ndarray:
    if arr.ndim < 3:
        raise ValueError("Expected an array with a ky axis in the third-to-last position")
    slicer = [slice(None)] * arr.ndim
    slicer[-3] = slice(ky_idx, ky_idx + 1)
    return arr[tuple(slicer)]


def _build_case_setup(
    *,
    case: str,
    nx: int,
    ny: int,
    nz: int,
    y0: float,
    ntheta: int | None,
    nperiod: int | None,
    Nm: int,
) -> tuple[CycloneBaseCase | KBMBaseCase, SAlphaGeometry, LinearParams]:
    if case == "cyclone":
        cfg_c = CycloneBaseCase(
            grid=GridConfig(
                Nx=nx,
                Ny=ny,
                Nz=nz,
                Lx=62.8,
                Ly=62.8,
                boundary="linked",
                y0=y0,
                ntheta=None,
                nperiod=None,
            )
        )
        geom = SAlphaGeometry.from_config(cfg_c.geometry)
        params = LinearParams(
            R_over_Ln=cfg_c.model.R_over_Ln,
            R_over_LTi=cfg_c.model.R_over_LTi,
            R_over_LTe=cfg_c.model.R_over_LTe,
            omega_d_scale=CYCLONE_OMEGA_D_SCALE,
            omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
            rho_star=CYCLONE_RHO_STAR,
            kpar_scale=float(geom.gradpar()),
            nu=cfg_c.model.nu_i,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
        params = _apply_gx_hypercollisions(params, nhermite=Nm)
        return cfg_c, geom, params

    cfg_k = KBMBaseCase(
        grid=GridConfig(
            Nx=nx,
            Ny=ny,
            Nz=nz,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=y0,
            ntheta=32 if ntheta is None else int(ntheta),
            nperiod=2 if nperiod is None else int(nperiod),
        )
    )
    geom = SAlphaGeometry.from_config(cfg_k.geometry)
    params = _two_species_params(
        cfg_k.model,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=KBM_OMEGA_D_SCALE,
        omega_star_scale=KBM_OMEGA_STAR_SCALE,
        rho_star=KBM_RHO_STAR,
        nhermite=Nm,
    )
    return cfg_k, geom, params


def _build_startup_state(
    *,
    case: str,
    cfg: CycloneBaseCase | KBMBaseCase,
    geom: SAlphaGeometry,
    Nl: int,
    Nm: int,
    ky_target: float,
) -> tuple[np.ndarray, np.ndarray, SAlphaGeometry]:
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(ky_target))))
    grid = select_ky_grid(grid_full, ky_index)
    g_single = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg.init,
    )
    if case == "kbm":
        g0 = np.zeros((2, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
        g0[1] = np.asarray(g_single, dtype=np.complex64)
    else:
        g0 = np.asarray(g_single, dtype=np.complex64)[None, ...]
    return g0, np.asarray(grid.ky, dtype=float), geom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX field dump binaries")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file for ky metadata")
    parser.add_argument("--case", choices=["cyclone", "kbm"], required=True)
    parser.add_argument("--ky", type=float, required=True)
    parser.add_argument("--Ny", type=int, required=True)
    parser.add_argument("--Nz", type=int, required=True)
    parser.add_argument("--Nl", type=int, required=True)
    parser.add_argument("--Nm", type=int, required=True)
    parser.add_argument("--ntheta", type=int, default=None)
    parser.add_argument("--nperiod", type=int, default=None)
    parser.add_argument("--y0", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    shape = _load_shape(args.gx_dir / "field_shape.txt")
    nspec = shape["nspec"]
    nl = shape["nl"]
    nm = shape["nm"]
    nyc = shape["nyc"]
    nx = shape["nx"]
    nz = shape["nz"]
    gx_shape = (nspec, nl, nm, nyc, nx, nz)

    gx_g = _reshape_gx(
        _load_bin(args.gx_dir / "field_g_state.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_phi = _load_field(args.gx_dir / "field_phi.bin", nyc, nx, nz)
    gx_apar = None
    gx_apar_path = args.gx_dir / "field_apar.bin"
    if gx_apar_path.exists():
        gx_apar = _load_field(gx_apar_path, nyc, nx, nz)

    with Dataset(args.gx_out, "r") as root:
        ky_vals = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)

    y0_use = float(args.y0) if args.y0 is not None else _infer_y0(ky_vals)
    ky_idx = int(np.argmin(np.abs(ky_vals - float(args.ky))))

    cfg, geom, params = _build_case_setup(
        case=args.case,
        nx=nx,
        ny=args.Ny,
        nz=args.Nz,
        y0=y0_use,
        ntheta=args.ntheta,
        nperiod=args.nperiod,
        Nm=args.Nm,
    )
    sp_g, _ky_grid, _geom = _build_startup_state(
        case=args.case,
        cfg=cfg,
        geom=geom,
        Nl=args.Nl,
        Nm=args.Nm,
        ky_target=float(args.ky),
    )
    grid = select_ky_grid(build_spectral_grid(cfg.grid), int(np.argmin(np.abs(np.asarray(build_spectral_grid(cfg.grid).ky) - float(args.ky)))))
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    term_cfg = TermConfig(bpar=0.0) if args.case == "kbm" else TermConfig()
    sp_fields = compute_fields_cached(jnp.asarray(sp_g), cache, params, terms=term_cfg)

    gx_g_slice = _select_ky_block(gx_g, ky_idx)
    gx_phi_slice = _select_ky_block(gx_phi, ky_idx)
    _summary("g_state", gx_g_slice.astype(np.complex64), sp_g.astype(np.complex64))
    _summary("phi", gx_phi_slice.astype(np.complex64), np.asarray(sp_fields.phi, dtype=np.complex64))

    if gx_apar is not None and sp_fields.apar is not None:
        gx_apar_slice = _select_ky_block(gx_apar, ky_idx)
        _summary("apar", gx_apar_slice.astype(np.complex64), np.asarray(sp_fields.apar, dtype=np.complex64))


if __name__ == "__main__":
    main()
