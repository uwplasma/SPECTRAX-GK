#!/usr/bin/env python3
"""Compare a GX late-time diagnostic-state dump against a runtime-configured SPECTRAX setup."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from compare_gx_rhs_terms import _infer_y0, _load_field, _reshape_gx, _summary
from spectraxgk.diagnostics import (
    gx_Wapar,
    gx_Wg,
    gx_Wphi,
    gx_heat_flux,
    gx_heat_flux_species,
    gx_particle_flux,
    gx_particle_flux_species,
    gx_volume_factors,
)
from spectraxgk.geometry import apply_gx_geometry_grid_defaults, ensure_flux_tube_geometry_data
from spectraxgk.grids import build_spectral_grid, select_gx_real_fft_ky_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.linear import build_linear_cache
from spectraxgk.runtime import (
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from spectraxgk.terms.assembly import compute_fields_cached


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


def _load_real_vector(path: Path, size: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != size:
        raise ValueError(f"{path} size {raw.size} does not match expected {size}")
    return raw.astype(np.float32)


def _load_real_vector_auto(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")
    return raw.astype(np.float32)


def _load_species_state(
    gx_dir: Path,
    *,
    nspec: int,
    nl: int,
    nm: int,
    nyc: int,
    nx: int,
    nz: int,
    time_index: int,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    n_expected = nl * nm * nyc * nx * nz
    for ispec in range(nspec):
        path = gx_dir / f"diag_state_G_s{ispec}_t{time_index}.bin"
        raw = np.fromfile(path, dtype=np.complex64)
        if raw.size != n_expected:
            raise ValueError(f"{path} size {raw.size} does not match expected {n_expected}")
        pieces.append(raw)
    return _reshape_gx(
        np.stack(pieces, axis=0),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )


def _maybe_load_field(path: Path, nyc: int, nx: int, nz: int) -> np.ndarray | None:
    if not path.exists():
        return None
    return _load_field(path, nyc, nx, nz)


def _gx_diag_scalar(group, name: str, time_index: int) -> float:
    arr = np.asarray(group.variables[name][time_index], dtype=float)
    return float(np.sum(arr))


def _gx_diag_species(group, name: str, time_index: int) -> np.ndarray | None:
    if name not in group.variables:
        return None
    arr = np.asarray(group.variables[name][time_index], dtype=float)
    return np.atleast_1d(np.squeeze(arr))


def _diag_row(
    G: np.ndarray,
    phi: np.ndarray,
    apar: np.ndarray,
    bpar: np.ndarray,
    *,
    cache,
    grid,
    params,
    vol_fac: jnp.ndarray,
    flux_fac: jnp.ndarray,
    flux_scale: float,
    wphi_scale: float,
) -> dict[str, float | np.ndarray]:
    G_j = jnp.asarray(G)
    phi_j = jnp.asarray(phi)
    apar_j = jnp.asarray(apar)
    bpar_j = jnp.asarray(bpar)
    return {
        "Wg": float(gx_Wg(G_j, grid, params, vol_fac)),
        "Wphi": float(gx_Wphi(phi_j, cache, params, vol_fac, wphi_scale=wphi_scale)),
        "Wapar": float(gx_Wapar(apar_j, cache, vol_fac)),
        "heat": float(gx_heat_flux(G_j, phi_j, apar_j, bpar_j, cache, grid, params, flux_fac, flux_scale=flux_scale)),
        "pflux": float(
            gx_particle_flux(G_j, phi_j, apar_j, bpar_j, cache, grid, params, flux_fac, flux_scale=flux_scale)
        ),
        "heat_s": np.asarray(
            gx_heat_flux_species(
                G_j,
                phi_j,
                apar_j,
                bpar_j,
                cache,
                grid,
                params,
                flux_fac,
                flux_scale=flux_scale,
            ),
            dtype=float,
        ),
        "pflux_s": np.asarray(
            gx_particle_flux_species(
                G_j,
                phi_j,
                apar_j,
                bpar_j,
                cache,
                grid,
                params,
                flux_fac,
                flux_scale=flux_scale,
            ),
            dtype=float,
        ),
    }


def _rel_err(test: float, ref: float) -> float:
    denom = max(abs(ref), 1.0e-30)
    return float(abs(test - ref) / denom)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX diag_state dump binaries")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file for dimensions/diagnostics")
    parser.add_argument("--config", type=Path, required=True, help="Runtime TOML config used by SPECTRAX")
    parser.add_argument("--time-index", type=int, required=True, help="GX diagnostic time index to compare")
    parser.add_argument("--y0", type=float, default=None, help="Optional y0 override; defaults to GX ky metadata")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV summary output")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with Dataset(args.gx_out, "r") as root:
        diag = root.groups["Diagnostics"]
        nl = int(root.dimensions["l"].size)
        nm = int(root.dimensions["m"].size)
        nspec = int(root.dimensions["s"].size)
        if args.time_index < 0 or args.time_index >= int(root.dimensions["time"].size):
            raise ValueError(f"time_index={args.time_index} outside [0, {int(root.dimensions['time'].size) - 1}]")
        gx_diag = {
            "Wg": _gx_diag_scalar(diag, "Wg_kyst", args.time_index),
            "Wphi": _gx_diag_scalar(diag, "Wphi_kyst", args.time_index),
            "Wapar": _gx_diag_scalar(diag, "Wapar_kyst", args.time_index),
            "heat": _gx_diag_scalar(diag, "HeatFlux_kyst", args.time_index),
            "pflux": _gx_diag_scalar(diag, "ParticleFlux_kyst", args.time_index),
        }
        gx_heat_s = _gx_diag_species(diag, "HeatFlux_st", args.time_index)
        gx_pflux_s = _gx_diag_species(diag, "ParticleFlux_st", args.time_index)
        t_val = float(np.asarray(root.groups["Grids"].variables["time"][args.time_index], dtype=float))

    gx_kx = _load_real_vector_auto(args.gx_dir / f"diag_state_kx_t{args.time_index}.bin")
    gx_ky = _load_real_vector_auto(args.gx_dir / f"diag_state_ky_t{args.time_index}.bin")
    nyc = int(gx_ky.size)
    nx = int(gx_kx.size)
    phi_raw = np.fromfile(args.gx_dir / f"diag_state_phi_t{args.time_index}.bin", dtype=np.complex64)
    if phi_raw.size % max(nyc * nx, 1) != 0:
        raise ValueError(
            f"diag_state_phi_t{args.time_index}.bin size {phi_raw.size} is not divisible by nyc*nx={nyc*nx}"
        )
    nz = int(phi_raw.size // (nyc * nx))

    gx_G = _load_species_state(
        args.gx_dir,
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
        time_index=args.time_index,
    )
    gx_phi = _load_field(args.gx_dir / f"diag_state_phi_t{args.time_index}.bin", nyc, nx, nz)
    gx_apar = _maybe_load_field(args.gx_dir / f"diag_state_apar_t{args.time_index}.bin", nyc, nx, nz)
    gx_bpar = _maybe_load_field(args.gx_dir / f"diag_state_bpar_t{args.time_index}.bin", nyc, nx, nz)
    gx_kperp2 = _load_real_field(args.gx_dir / f"diag_state_kperp2_t{args.time_index}.bin", nyc, nx, nz)
    gx_fluxfac = _load_real_vector(args.gx_dir / f"diag_state_fluxfac_t{args.time_index}.bin", nz)

    cfg, _data = load_runtime_from_toml(args.config)
    y0_use = float(args.y0) if args.y0 is not None else _infer_y0(gx_ky)
    full_ny = int(2 * (nyc - 1))
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
    grid = select_gx_real_fft_ky_grid(grid_full, gx_ky.astype(np.float32))
    params = build_runtime_linear_params(cfg_use, Nm=nm, geom=geom)
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    cache = build_linear_cache(grid, geom_eff, params, nl, nm)
    term_cfg = build_runtime_term_config(cfg_use)
    sp_fields = compute_fields_cached(jnp.asarray(gx_G), cache, params, terms=term_cfg)

    vol_fac, flux_fac = gx_volume_factors(geom_eff, grid)

    sp_phi = np.asarray(sp_fields.phi, dtype=np.complex64)
    sp_apar = (
        np.asarray(sp_fields.apar, dtype=np.complex64)
        if sp_fields.apar is not None
        else np.zeros_like(gx_phi, dtype=np.complex64)
    )
    sp_bpar = (
        np.asarray(sp_fields.bpar, dtype=np.complex64)
        if sp_fields.bpar is not None
        else np.zeros_like(gx_phi, dtype=np.complex64)
    )
    gx_apar_use = gx_apar if gx_apar is not None else np.zeros_like(gx_phi, dtype=np.complex64)
    gx_bpar_use = gx_bpar if gx_bpar is not None else np.zeros_like(gx_phi, dtype=np.complex64)

    _summary("kperp2", gx_kperp2.astype(np.float32), np.asarray(cache.kperp2, dtype=np.float32))
    _summary("fluxfac", gx_fluxfac.astype(np.float32), np.asarray(flux_fac, dtype=np.float32))
    if gx_kx.shape == grid.kx.shape:
        _summary("kx", gx_kx.astype(np.float32), np.asarray(grid.kx, dtype=np.float32))
    if gx_ky.shape == grid.ky.shape:
        _summary("ky", gx_ky.astype(np.float32), np.asarray(grid.ky, dtype=np.float32))
    _summary("phi", gx_phi.astype(np.complex64), sp_phi)
    if gx_apar is not None or sp_fields.apar is not None:
        _summary("apar", gx_apar_use.astype(np.complex64), sp_apar)
    if gx_bpar is not None or sp_fields.bpar is not None:
        _summary("bpar", gx_bpar_use.astype(np.complex64), sp_bpar)

    flux_scale = float(cfg_use.normalization.flux_scale)
    wphi_scale = float(cfg_use.normalization.wphi_scale)
    sp_dump = _diag_row(
        gx_G,
        gx_phi,
        gx_apar_use,
        gx_bpar_use,
        cache=cache,
        grid=grid,
        params=params,
        vol_fac=vol_fac,
        flux_fac=flux_fac,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
    )
    sp_solve = _diag_row(
        gx_G,
        sp_phi,
        sp_apar,
        sp_bpar,
        cache=cache,
        grid=grid,
        params=params,
        vol_fac=vol_fac,
        flux_fac=flux_fac,
        flux_scale=flux_scale,
        wphi_scale=wphi_scale,
    )

    print(f"time_index={args.time_index} t={t_val:.8f}")
    print("metric     gx_out        spectrax_dump  rel_dump      spectrax_solve rel_solve")
    rows: list[dict[str, float | str]] = []
    for key in ("Wg", "Wphi", "Wapar", "heat", "pflux"):
        gx_val = float(gx_diag[key])
        dump_val = float(sp_dump[key])
        solve_val = float(sp_solve[key])
        rel_dump = _rel_err(dump_val, gx_val)
        rel_solve = _rel_err(solve_val, gx_val)
        rows.append(
            {
                "time_index": float(args.time_index),
                "t": t_val,
                "metric": key,
                "gx_out": gx_val,
                "spectrax_dump": dump_val,
                "rel_dump": rel_dump,
                "spectrax_solve": solve_val,
                "rel_solve": rel_solve,
            }
        )
        print(
            f"{key:8s} {gx_val: .6e} {dump_val: .6e} {rel_dump: .3e} "
            f"{solve_val: .6e} {rel_solve: .3e}"
        )

    if gx_heat_s is not None:
        print("heat_flux_species gx_out=", np.asarray(gx_heat_s, dtype=float), "spectrax_dump=", sp_dump["heat_s"])
    if gx_pflux_s is not None:
        print("particle_flux_species gx_out=", np.asarray(gx_pflux_s, dtype=float), "spectrax_dump=", sp_dump["pflux_s"])

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
