#!/usr/bin/env python3
"""Compare a GX startup field dump against an imported-geometry SPECTRAX setup."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from compare_gx_imported_linear import (
    _build_imported_initial_condition,
    _load_gx_input_contract,
    _resolve_imported_real_fft_ny,
    _resolve_imported_boundary,
    _resolve_internal_geometry_source,
)
from compare_gx_rhs_terms import _infer_y0, _load_bin, _load_field, _load_shape, _reshape_gx, _summary
from compare_gx_runtime_startup import _select_ky_block
from spectraxgk.config import GeometryConfig
from spectraxgk.geometry import SlabGeometry, apply_gx_geometry_grid_defaults, load_gx_geometry_netcdf
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import build_linear_cache
from spectraxgk.species import build_linear_params
from spectraxgk.terms.assembly import compute_fields_cached


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX field dump binaries")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file for ky metadata")
    parser.add_argument(
        "--geometry-file",
        type=Path,
        required=True,
        help="GX/VMEC geometry file used for the imported SPECTRAX run",
    )
    parser.add_argument("--gx-input", type=Path, required=True, help="GX input file used to build the imported setup")
    parser.add_argument("--ky", type=float, required=True, help="ky value to compare")
    parser.add_argument("--kx-target", type=float, default=0.0, help="kx target within the selected ky block")
    parser.add_argument("--y0", type=float, default=None, help="Optional y0 override; defaults to GX ky metadata")
    return parser


def _resolve_startup_dump_layout(gx_dir: Path) -> Tuple[Path, Path, Path, Optional[Path]]:
    """Return the shape/g_state/phi/apar paths for a GX startup-style dump."""

    field_shape = gx_dir / "field_shape.txt"
    if field_shape.exists():
        field_g = gx_dir / "field_g_state.bin"
        field_phi = gx_dir / "field_phi.bin"
        field_apar = gx_dir / "field_apar.bin"
        return field_shape, field_g, field_phi, field_apar if field_apar.exists() else None

    rhs_shape = gx_dir / "rhs_terms_shape.txt"
    if rhs_shape.exists():
        rhs_g = gx_dir / "g_state.bin"
        rhs_phi = gx_dir / "phi.bin"
        rhs_apar = gx_dir / "apar.bin"
        return rhs_shape, rhs_g, rhs_phi, rhs_apar if rhs_apar.exists() else None

    # Keep the historical default for tests and lightweight ad hoc runs.
    return field_shape, gx_dir / "field_g_state.bin", gx_dir / "field_phi.bin", None


def main() -> None:
    args = build_parser().parse_args()

    shape_path, gx_g_path, gx_phi_path, gx_apar_path = _resolve_startup_dump_layout(args.gx_dir)
    shape = _load_shape(shape_path)
    nspec = shape["nspec"]
    nl = shape["nl"]
    nm = shape["nm"]
    nyc = shape["nyc"]
    nx = shape["nx"]
    nz = shape["nz"]
    gx_shape = (nspec, nl, nm, nyc, nx, nz)

    gx_g = _reshape_gx(
        _load_bin(gx_g_path, gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_phi = _load_field(gx_phi_path, nyc, nx, nz)
    gx_apar = None
    if gx_apar_path is not None:
        gx_apar = _load_field(gx_apar_path, nyc, nx, nz)

    with Dataset(args.gx_out, "r") as root:
        ky_vals = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)

    gx_contract = _load_gx_input_contract(args.gx_input)
    y0_use = float(args.y0) if args.y0 is not None else _infer_y0(ky_vals)
    ny_full = _resolve_imported_real_fft_ny(ky_vals, gx_contract)
    if getattr(gx_contract, "geo_option", "s-alpha") == "slab":
        geom = SlabGeometry.from_config(
            GeometryConfig(
                model="slab",
                s_hat=float(getattr(gx_contract, "s_hat", 0.0)),
                zero_shat=bool(getattr(gx_contract, "zero_shat", False)),
            )
        )
    else:
        geom = load_gx_geometry_netcdf(_resolve_internal_geometry_source(geometry_file=args.geometry_file, runtime_config=None))
    grid_cfg = apply_gx_geometry_grid_defaults(
        geom,
        gx_contract_to_grid(
            gx_contract=gx_contract,
            nx=int(nx),
            ny=int(ny_full),
            nz=int(nz),
            y0=float(y0_use),
        ),
    )
    grid_full = build_spectral_grid(grid_cfg)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(args.ky))))
    kx_index = int(np.argmin(np.abs(np.asarray(grid_full.kx, dtype=float) - float(args.kx_target))))

    g0 = _build_imported_initial_condition(
        grid=grid_full,
        geom=geom,
        gx_contract=gx_contract,
        species=gx_contract.species,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=nl,
        Nm=nm,
    )
    params = build_linear_params(
        gx_contract.species,
        tau_e=float(gx_contract.tau_e),
        kpar_scale=float(geom.gradpar()),
        beta=float(gx_contract.beta),
    )
    cache = build_linear_cache(grid_full, geom, params, nl, nm)
    sp_fields = compute_fields_cached(jnp.asarray(g0), cache, params)

    ky_idx_gx = int(np.argmin(np.abs(ky_vals - float(args.ky))))
    gx_g_slice = _select_ky_block(gx_g, ky_idx_gx)
    gx_phi_slice = _select_ky_block(gx_phi, ky_idx_gx)
    sp_g_slice = _select_ky_block(np.asarray(g0, dtype=np.complex64), ky_index)
    sp_phi_slice = _select_ky_block(np.asarray(sp_fields.phi, dtype=np.complex64), ky_index)
    _summary("g_state", gx_g_slice.astype(np.complex64), sp_g_slice)
    _summary("phi", gx_phi_slice.astype(np.complex64), sp_phi_slice)

    if gx_apar is not None and sp_fields.apar is not None:
        gx_apar_slice = _select_ky_block(gx_apar, ky_idx_gx)
        sp_apar_slice = _select_ky_block(np.asarray(sp_fields.apar, dtype=np.complex64), ky_index)
        _summary("apar", gx_apar_slice.astype(np.complex64), sp_apar_slice)


def gx_contract_to_grid(*, gx_contract, nx: int, ny: int, nz: int, y0: float):
    from spectraxgk.config import GridConfig

    boundary = _resolve_imported_boundary(
        str(gx_contract.boundary),
        zero_shat=bool(getattr(gx_contract, "zero_shat", False)),
    )
    lx = 2.0 * np.pi * float(y0) if boundary == "periodic" else 62.8

    return GridConfig(
        Nx=int(nx),
        Ny=int(ny),
        Nz=int(nz),
        Lx=lx,
        Ly=2.0 * np.pi * float(y0),
        boundary=boundary,
        y0=float(y0),
        nperiod=max(1, int(gx_contract.nperiod)),
        ntheta=max(1, int(gx_contract.ntheta)),
    )


if __name__ == "__main__":
    main()
