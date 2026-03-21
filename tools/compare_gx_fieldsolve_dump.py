#!/usr/bin/env python3
"""Compare GX field-solver dump coefficients against SPECTRAX on the same state."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from compare_gx_rhs_terms import (
    _build_imported_compare_context,
    _infer_y0,
    _load_bin,
    _load_shape,
    _reshape_gx,
    _summary,
)
from spectraxgk.grids import select_ky_grid
from spectraxgk.linear import _as_species_array, build_linear_cache


def _load_real_field(path: Path, nyc: int, nx: int, nz: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    expected = nyc * nx * nz
    if raw.size != expected:
        raise ValueError(f"{path} size {raw.size} does not match expected {expected}")
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    return raw[idxyz.ravel()].reshape(nyc, nx, nz)


def _load_complex_packed_fields(path: Path, nyc: int, nx: int, nz: int, nfields: int) -> list[np.ndarray]:
    raw = np.fromfile(path, dtype=np.complex64)
    nR = nyc * nx * nz
    expected = nfields * nR
    if raw.size != expected:
        raise ValueError(f"{path} size {raw.size} does not match expected {expected}")
    out: list[np.ndarray] = []
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    for i in range(nfields):
        block = raw[i * nR : (i + 1) * nR]
        out.append(block[idxyz.ravel()].reshape(nyc, nx, nz))
    return out


def _select_ky_block(arr: np.ndarray, ky_idx: int) -> np.ndarray:
    slicer = [slice(None)] * arr.ndim
    slicer[-3] = slice(ky_idx, ky_idx + 1)
    return arr[tuple(slicer)]


def _fieldsolve_factors(G: np.ndarray, cache, params) -> dict[str, np.ndarray]:
    real_dtype = jnp.float32
    G_j = jnp.asarray(G, dtype=jnp.complex64)
    ns = int(G_j.shape[0])
    charge = _as_species_array(params.charge_sign, ns, "charge_sign").astype(real_dtype)
    density = _as_species_array(params.density, ns, "density").astype(real_dtype)
    mass = _as_species_array(params.mass, ns, "mass").astype(real_dtype)
    temp = _as_species_array(params.temp, ns, "temp").astype(real_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)

    Jl = cache.Jl
    JlB = cache.JlB
    bmag = cache.bmag
    kperp2 = cache.kperp2

    beta = jnp.asarray(params.beta, dtype=real_dtype)
    apar_beta_scale = jnp.asarray(params.apar_beta_scale, dtype=real_dtype)
    ampere_g0_scale = jnp.asarray(params.ampere_g0_scale, dtype=real_dtype)
    bpar_beta_scale = jnp.asarray(params.bpar_beta_scale, dtype=real_dtype)
    tau_e = jnp.asarray(params.tau_e, dtype=real_dtype)

    Gm1 = G_j[:, :, 1, ...]
    Gm0 = G_j[:, :, 0, ...]

    nbar = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * jnp.sum(Jl * Gm0, axis=1),
        axis=0,
    )
    bmag_inv2 = 1.0 / (bmag * bmag)
    bpar_beta = bpar_beta_scale * beta
    jperpbar = jnp.sum(
        (-bpar_beta)
        * density[:, None, None, None]
        * temp[:, None, None, None]
        * bmag_inv2[None, None, :]
        * jnp.sum(JlB * Gm0, axis=1),
        axis=0,
    )
    g0 = jnp.sum(Jl * Jl, axis=1)
    g01 = jnp.sum(Jl * JlB, axis=1)
    g11 = jnp.sum(JlB * JlB, axis=1)
    qneut = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * jnp.where(tz == 0.0, 0.0, 1.0 / tz)[:, None, None, None]
        * (1.0 - g0),
        axis=0,
    )
    qphi = tau_e + qneut
    qb = -jnp.sum(density[:, None, None, None] * charge[:, None, None, None] * g01, axis=0)
    aphi = bpar_beta * jnp.sum(
        density[:, None, None, None] * charge[:, None, None, None] * g01,
        axis=0,
    ) * bmag_inv2[None, None, :]
    ab = 1.0 + bpar_beta * jnp.sum(
        density[:, None, None, None] * temp[:, None, None, None] * g11,
        axis=0,
    ) * bmag_inv2[None, None, :]
    jparbar = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * vth[:, None, None, None]
        * jnp.sum(Jl * Gm1, axis=1),
        axis=0,
    )
    jparbar = apar_beta_scale * beta * jparbar
    bmag2 = bmag[None, None, :] * bmag[None, None, :]
    use_bmag = jnp.asarray(getattr(cache, "kperp2_bmag", True), dtype=real_dtype)
    ampere_kperp2 = kperp2 * (use_bmag * bmag2 + (1.0 - use_bmag))
    ampere_par = ampere_kperp2 + ampere_g0_scale * beta * jnp.sum(
        density[:, None, None, None]
        * (charge * charge / mass)[:, None, None, None]
        * g0,
        axis=0,
    )

    return {
        "nbar": np.asarray(nbar, dtype=np.complex64),
        "jparbar": np.asarray(jparbar, dtype=np.complex64),
        "jperpbar": np.asarray(jperpbar, dtype=np.complex64),
        "qneutFacPhi": np.asarray(qphi, dtype=np.float32),
        "qneutFacBpar": np.asarray(qb, dtype=np.float32),
        "ampereParFac": np.asarray(ampere_par, dtype=np.float32),
        "amperePerpFacPhi": np.asarray(aphi, dtype=np.float32),
        "amperePerpFacBpar": np.asarray(ab, dtype=np.float32),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gx-dir", type=Path, required=True, help="GX field dump directory containing field_*.bin")
    p.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc used to infer ky/y0 metadata")
    p.add_argument("--gx-input", type=Path, required=True, help="GX input file for the imported geometry contract")
    p.add_argument("--geometry-file", type=Path, required=True, help="Imported geometry source used by SPECTRAX")
    p.add_argument("--ky", type=float, required=True, help="ky slice to compare")
    return p


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
    gx_nbar_blocks = _load_complex_packed_fields(args.gx_dir / "field_nbar.bin", nyc, nx, nz, 3)
    gx_factors = {
        "nbar": gx_nbar_blocks[0],
        "jparbar": gx_nbar_blocks[1],
        "jperpbar": gx_nbar_blocks[2],
        "qneutFacPhi": _load_real_field(args.gx_dir / "field_qneutFacPhi.bin", nyc, nx, nz),
        "qneutFacBpar": _load_real_field(args.gx_dir / "field_qneutFacBpar.bin", nyc, nx, nz),
        "ampereParFac": _load_real_field(args.gx_dir / "field_ampereParFac.bin", nyc, nx, nz),
        "amperePerpFacPhi": _load_real_field(args.gx_dir / "field_amperePerpFacPhi.bin", nyc, nx, nz),
        "amperePerpFacBpar": _load_real_field(args.gx_dir / "field_amperePerpFacBpar.bin", nyc, nx, nz),
    }

    from netCDF4 import Dataset

    with Dataset(args.gx_out, "r") as root:
        ky_vals = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)

    _cfg, geom, grid_full, params, _term_cfg = _build_imported_compare_context(
        args.gx_out,
        args.gx_input,
        args.geometry_file,
        nx=nx,
        nz=nz,
        nm=nm,
        ky_vals=ky_vals,
        y0_override=_infer_y0(ky_vals),
    )
    ky_idx = int(np.argmin(np.abs(ky_vals - float(args.ky))))
    ky_index_full = int(np.argmin(np.abs(np.asarray(grid_full.ky, dtype=float) - float(args.ky))))
    grid = select_ky_grid(grid_full, ky_index_full)
    cache = build_linear_cache(grid, geom, params, nl, nm)
    gx_g_slice = _select_ky_block(gx_g, ky_idx)
    sp_factors = _fieldsolve_factors(gx_g_slice, cache, params)

    for name in (
        "nbar",
        "jparbar",
        "jperpbar",
        "qneutFacPhi",
        "qneutFacBpar",
        "ampereParFac",
        "amperePerpFacPhi",
        "amperePerpFacBpar",
    ):
        gx_slice = _select_ky_block(gx_factors[name], ky_idx)
        sp_slice = np.asarray(sp_factors[name], dtype=np.complex64)
        _summary(name, np.asarray(gx_slice).astype(np.complex64), np.asarray(sp_slice).astype(np.complex64))


if __name__ == "__main__":
    main()
