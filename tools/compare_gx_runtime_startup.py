#!/usr/bin/env python3
"""Compare a GX startup field dump against a runtime-configured SPECTRAX setup."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from compare_gx_rhs_terms import _infer_y0, _load_bin, _load_field, _load_shape, _reshape_gx, _summary
from spectraxgk.geometry import apply_gx_geometry_grid_defaults
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.linear import build_linear_cache
from spectraxgk.runtime import (
    _build_initial_condition,
    _species_to_linear,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from spectraxgk.terms.assembly import compute_fields_cached


def _select_ky_block(arr: np.ndarray, ky_idx: int) -> np.ndarray:
    if arr.ndim < 3:
        raise ValueError("Expected an array with a ky axis in the third-to-last position")
    slicer = [slice(None)] * arr.ndim
    slicer[-3] = slice(ky_idx, ky_idx + 1)
    return arr[tuple(slicer)]


def _full_ny_from_positive_ky(ky_vals: np.ndarray) -> int:
    ky_arr = np.asarray(ky_vals, dtype=float)
    if ky_arr.ndim != 1 or ky_arr.size == 0:
        raise ValueError("ky_vals must be a non-empty 1D array")
    return int(3 * (ky_arr.size - 1) + 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX field dump binaries")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file for ky metadata")
    parser.add_argument("--config", type=Path, required=True, help="Runtime TOML config used by SPECTRAX")
    parser.add_argument("--ky", type=float, required=True, help="ky value to compare")
    parser.add_argument("--kx-target", type=float, default=0.0, help="kx target within the selected ky block")
    parser.add_argument("--y0", type=float, default=None, help="Optional y0 override; defaults to GX ky metadata")
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

    cfg, _data = load_runtime_from_toml(args.config)
    y0_use = float(args.y0) if args.y0 is not None else _infer_y0(ky_vals)
    full_ny = _full_ny_from_positive_ky(ky_vals)
    cfg_use = replace(
        cfg,
        grid=replace(
            cfg.grid,
            Nx=int(nx),
            Ny=int(full_ny),
            Nz=int(nz),
            y0=float(y0_use),
        ),
    )

    geom = build_runtime_geometry(cfg_use)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg_use.grid)
    grid_full = build_spectral_grid(grid_cfg)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(args.ky))))
    grid = select_ky_grid(grid_full, ky_index)
    kx_index = int(np.argmin(np.abs(np.asarray(grid.kx, dtype=float) - float(args.kx_target))))

    params = build_runtime_linear_params(cfg_use, Nm=nm, geom=geom)
    g0 = _build_initial_condition(
        grid,
        geom,
        cfg_use,
        ky_index=0,
        kx_index=kx_index,
        Nl=nl,
        Nm=nm,
        nspecies=len(_species_to_linear(cfg_use.species)),
    )
    cache = build_linear_cache(grid, geom, params, nl, nm)
    term_cfg = build_runtime_term_config(cfg_use)
    sp_fields = compute_fields_cached(jnp.asarray(g0), cache, params, terms=term_cfg)

    ky_idx_gx = int(np.argmin(np.abs(ky_vals - float(args.ky))))
    gx_g_slice = _select_ky_block(gx_g, ky_idx_gx)
    gx_phi_slice = _select_ky_block(gx_phi, ky_idx_gx)
    _summary("g_state", gx_g_slice.astype(np.complex64), np.asarray(g0, dtype=np.complex64))
    _summary("phi", gx_phi_slice.astype(np.complex64), np.asarray(sp_fields.phi, dtype=np.complex64))

    if gx_apar is not None and sp_fields.apar is not None:
        gx_apar_slice = _select_ky_block(gx_apar, ky_idx_gx)
        _summary("apar", gx_apar_slice.astype(np.complex64), np.asarray(sp_fields.apar, dtype=np.complex64))


if __name__ == "__main__":
    main()
