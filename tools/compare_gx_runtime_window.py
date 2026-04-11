#!/usr/bin/env python3
"""Compare one exact-state nonlinear evolution window against GX runtime diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from tools.compare_gx_rhs_terms import _infer_y0, _summary
from tools.compare_gx_runtime_diag_state import _gx_diag_scalar, _gx_diag_species, _load_field, _load_real_vector_auto, _load_species_state, _maybe_load_field
from spectraxgk.geometry import apply_gx_geometry_grid_defaults, ensure_flux_tube_geometry_data
from spectraxgk.grids import build_spectral_grid, select_gx_real_fft_ky_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.linear import build_linear_cache
from spectraxgk.runtime import build_runtime_geometry, build_runtime_linear_params, build_runtime_term_config, run_runtime_nonlinear
from spectraxgk.terms.assembly import compute_fields_cached


def _rel_err(test: float, ref: float) -> float:
    denom = max(abs(ref), 1.0e-30)
    return float(abs(test - ref) / denom)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX diag_state dump binaries")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file for dimensions/diagnostics")
    parser.add_argument("--config", type=Path, required=True, help="Runtime TOML config used by SPECTRAX")
    parser.add_argument("--time-index-start", type=int, required=True, help="GX diagnostic time index for the start state")
    parser.add_argument("--time-index-stop", type=int, required=True, help="GX diagnostic time index for the target state")
    parser.add_argument("--steps", type=int, default=None, help="Optional maximum step count override for the runtime audit")
    parser.add_argument("--ky", type=float, default=None, help="Optional ky to use for gamma/omega diagnostics")
    parser.add_argument("--y0", type=float, default=None, help="Optional y0 override; defaults to GX ky metadata")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV summary output")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with Dataset(args.gx_out, "r") as root:
        diag = root.groups["Diagnostics"]
        grids = root.groups["Grids"]
        nl = int(root.dimensions["l"].size)
        nm = int(root.dimensions["m"].size)
        nspec = int(root.dimensions["s"].size)
        ntime = int(root.dimensions["time"].size)
        if args.time_index_start < 0 or args.time_index_start >= ntime:
            raise ValueError(f"time_index_start={args.time_index_start} outside [0, {ntime - 1}]")
        if args.time_index_stop < 0 or args.time_index_stop >= ntime:
            raise ValueError(f"time_index_stop={args.time_index_stop} outside [0, {ntime - 1}]")
        t_vals = np.asarray(grids.variables["time"][:], dtype=float)
        gx_diag_stop = {
            "Wg": _gx_diag_scalar(diag, "Wg_kyst", args.time_index_stop),
            "Wphi": _gx_diag_scalar(diag, "Wphi_kyst", args.time_index_stop),
            "Wapar": _gx_diag_scalar(diag, "Wapar_kyst", args.time_index_stop),
            "heat": _gx_diag_scalar(diag, "HeatFlux_kyst", args.time_index_stop),
            "pflux": _gx_diag_scalar(diag, "ParticleFlux_kyst", args.time_index_stop),
        }
        gx_heat_s = _gx_diag_species(diag, "HeatFlux_st", args.time_index_stop)
        gx_pflux_s = _gx_diag_species(diag, "ParticleFlux_st", args.time_index_stop)
        t_start = float(t_vals[args.time_index_start])
        t_stop = float(t_vals[args.time_index_stop])
        dt_window = float(t_stop - t_start)

    gx_kx = _load_real_vector_auto(args.gx_dir / f"diag_state_kx_t{args.time_index_start}.bin")
    gx_ky = _load_real_vector_auto(args.gx_dir / f"diag_state_ky_t{args.time_index_start}.bin")
    nyc = int(gx_ky.size)
    nx = int(gx_kx.size)
    phi_raw = np.fromfile(args.gx_dir / f"diag_state_phi_t{args.time_index_start}.bin", dtype=np.complex64)
    if phi_raw.size % max(nyc * nx, 1) != 0:
        raise ValueError(
            f"diag_state_phi_t{args.time_index_start}.bin size {phi_raw.size} is not divisible by nyc*nx={nyc*nx}"
        )
    nz = int(phi_raw.size // (nyc * nx))

    gx_G_start = _load_species_state(
        args.gx_dir,
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
        time_index=args.time_index_start,
    )
    gx_phi_start = _load_field(args.gx_dir / f"diag_state_phi_t{args.time_index_start}.bin", nyc, nx, nz)
    gx_apar_start = _maybe_load_field(args.gx_dir / f"diag_state_apar_t{args.time_index_start}.bin", nyc, nx, nz)
    gx_bpar_start = _maybe_load_field(args.gx_dir / f"diag_state_bpar_t{args.time_index_start}.bin", nyc, nx, nz)

    cfg, _data = load_runtime_from_toml(args.config)
    y0_use = float(args.y0) if args.y0 is not None else _infer_y0(gx_ky)
    full_ny = int(2 * (nyc - 1))
    cfg_use = replace(
        cfg,
        grid=replace(cfg.grid, Nx=int(nx), Ny=int(full_ny), Nz=int(nz), y0=float(y0_use)),
    )

    geom = build_runtime_geometry(cfg_use)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg_use.grid)
    grid_full = build_spectral_grid(grid_cfg)
    grid = select_gx_real_fft_ky_grid(grid_full, gx_ky.astype(np.float32))
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    params = build_runtime_linear_params(cfg_use, Nm=nm, geom=geom)
    cache = build_linear_cache(grid, geom_eff, params, nl, nm)
    term_cfg = build_runtime_term_config(cfg_use)

    sp_fields_start = compute_fields_cached(jnp.asarray(gx_G_start, dtype=jnp.complex64), cache, params, terms=term_cfg)
    sp_phi_start = np.asarray(sp_fields_start.phi, dtype=np.complex64)
    sp_apar_start = (
        np.asarray(sp_fields_start.apar, dtype=np.complex64)
        if sp_fields_start.apar is not None
        else np.zeros_like(gx_phi_start, dtype=np.complex64)
    )
    sp_bpar_start = (
        np.asarray(sp_fields_start.bpar, dtype=np.complex64)
        if sp_fields_start.bpar is not None
        else np.zeros_like(gx_phi_start, dtype=np.complex64)
    )
    gx_apar_start_use = gx_apar_start if gx_apar_start is not None else np.zeros_like(gx_phi_start, dtype=np.complex64)
    gx_bpar_start_use = gx_bpar_start if gx_bpar_start is not None else np.zeros_like(gx_phi_start, dtype=np.complex64)

    _summary("start_g_state", gx_G_start.astype(np.complex64), np.asarray(gx_G_start, dtype=np.complex64))
    _summary("start_phi", gx_phi_start.astype(np.complex64), sp_phi_start)
    if gx_apar_start is not None or sp_fields_start.apar is not None:
        _summary("start_apar", gx_apar_start_use.astype(np.complex64), sp_apar_start)
    if gx_bpar_start is not None or sp_fields_start.bpar is not None:
        _summary("start_bpar", gx_bpar_start_use.astype(np.complex64), sp_bpar_start)

    ky_target = (
        float(args.ky)
        if args.ky is not None
        else float(next((val for val in np.asarray(grid.ky, dtype=float) if abs(val) > 0.0), 0.0))
    )
    cfg_run = replace(
        cfg_use,
        time=replace(
            cfg_use.time,
            t_max=dt_window,
            sample_stride=1,
            diagnostics_stride=1,
        ),
        init=replace(
            cfg_use.init,
            init_file=str(args.gx_dir / f"diag_state_G_s0_t{args.time_index_start}.bin"),
            init_file_scale=1.0,
            init_file_mode="replace",
        ),
    )
    result = run_runtime_nonlinear(
        cfg_run,
        ky_target=ky_target,
        Nl=nl,
        Nm=nm,
        steps=args.steps,
        diagnostics=True,
    )
    if result.diagnostics is None:
        raise RuntimeError("runtime window audit requires diagnostics output")
    diag_use = result.diagnostics
    t_arr = np.asarray(diag_use.t, dtype=float)
    idx_match = int(np.argmin(np.abs(t_arr - dt_window))) if t_arr.size else 0
    t_match = float(t_arr[idx_match]) if t_arr.size else 0.0
    sp_diag_stop = {
        "Wg": float(np.asarray(diag_use.Wg_t)[idx_match]),
        "Wphi": float(np.asarray(diag_use.Wphi_t)[idx_match]),
        "Wapar": float(np.asarray(diag_use.Wapar_t)[idx_match]),
        "heat": float(np.asarray(diag_use.heat_flux_t)[idx_match]),
        "pflux": float(np.asarray(diag_use.particle_flux_t)[idx_match]),
    }
    sp_diag_species: dict[str, np.ndarray] = {}
    if diag_use.heat_flux_species_t is not None:
        heat_s = np.asarray(diag_use.heat_flux_species_t)
        sp_diag_species["heat_s"] = np.asarray(heat_s[idx_match], dtype=float)
    if diag_use.particle_flux_species_t is not None:
        pflux_s = np.asarray(diag_use.particle_flux_species_t)
        sp_diag_species["pflux_s"] = np.asarray(pflux_s[idx_match], dtype=float)

    print(
        f"time_index_start={args.time_index_start} t_start={t_start:.8f} "
        f"time_index_stop={args.time_index_stop} t_stop={t_stop:.8f} "
        f"delta_t={dt_window:.8f} steps={args.steps if args.steps is not None else -1} "
        f"spectrax_t_match={t_match:.8f}"
    )
    print("metric     gx_stop       spectrax      rel")
    rows: list[dict[str, float | str]] = []
    for key in ("Wg", "Wphi", "Wapar", "heat", "pflux"):
        gx_val = float(gx_diag_stop[key])
        sp_val = sp_diag_stop[key]
        rel = _rel_err(sp_val, gx_val)
        rows.append(
            {
                "time_index_start": float(args.time_index_start),
                "time_index_stop": float(args.time_index_stop),
                "t_start": t_start,
                "t_stop": t_stop,
                "delta_t": dt_window,
                "steps": float(args.steps) if args.steps is not None else -1.0,
                "spectrax_t": t_match,
                "metric": key,
                "gx_stop": gx_val,
                "spectrax": sp_val,
                "rel": rel,
            }
        )
        print(f"{key:8s} {gx_val: .6e} {sp_val: .6e} {rel: .3e}")
    if gx_heat_s is not None:
        print(
            "heat_flux_species gx_stop=",
            np.asarray(gx_heat_s, dtype=float),
            "spectrax=",
            sp_diag_species.get("heat_s", np.asarray([])),
        )
    if gx_pflux_s is not None:
        print(
            "particle_flux_species gx_stop=",
            np.asarray(gx_pflux_s, dtype=float),
            "spectrax=",
            sp_diag_species.get("pflux_s", np.asarray([])),
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
