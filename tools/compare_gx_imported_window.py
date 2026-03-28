#!/usr/bin/env python3
"""Compare one exact imported-linear evolution window against GX diag_state dumps."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from compare_gx_imported_linear import (
    _build_imported_linear_terms,
    _gx_has_uniform_linear_dt,
    _gx_term_config,
    _infer_gx_linear_dt,
    _load_gx_input_contract,
    _resolve_imported_boundary,
    _resolve_imported_real_fft_ny,
    _select_geometry_source,
)
from compare_gx_rhs_terms import _infer_y0, _summary
from compare_gx_runtime_diag_state import (
    _load_field,
    _load_real_vector_auto,
    _load_species_state,
    _maybe_load_field,
)
from spectraxgk.benchmarks import _apply_gx_hypercollisions
from spectraxgk.config import GeometryConfig, GridConfig, resolve_cfl_fac
from spectraxgk.diagnostics import gx_Wapar, gx_Wg, gx_Wphi, gx_volume_factors
from spectraxgk.geometry import SlabGeometry, apply_gx_geometry_grid_defaults, load_gx_geometry_netcdf
from spectraxgk.grids import build_spectral_grid, select_gx_real_fft_ky_grid
from spectraxgk.gx_integrators import GXTimeConfig, _gx_linear_omega_max, _linear_explicit_step
from spectraxgk.linear import LinearTerms, build_linear_cache
from spectraxgk.species import build_linear_params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX diag_state dump binaries")
    p.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file for dimensions and times")
    p.add_argument("--gx-input", type=Path, required=True, help="GX input file describing the imported contract")
    p.add_argument("--geometry-file", type=Path, required=True, help="Imported geometry file used by SPECTRAX")
    p.add_argument("--time-index-start", type=int, required=True, help="GX diag_state start index")
    p.add_argument("--time-index-stop", type=int, required=True, help="GX diag_state stop index")
    p.add_argument("--out", type=Path, default=None, help="Optional CSV summary output")
    return p


def _rel_err(test: np.ndarray, ref: np.ndarray) -> float:
    ref_abs = np.abs(ref)
    denom = np.maximum(ref_abs, 1.0e-30)
    return float(np.max(np.abs(test - ref) / denom))


def _gx_phi2_total(phi: jnp.ndarray, vol_fac: jnp.ndarray) -> float:
    return float(jnp.sum(jnp.abs(phi) ** 2 * vol_fac[None, None, :]))


def main() -> None:
    args = build_parser().parse_args()

    gx_contract = _load_gx_input_contract(args.gx_input)
    with Dataset(args.gx_out, "r") as root:
        nl = int(root.dimensions["l"].size)
        nm = int(root.dimensions["m"].size)
        nspec = int(root.dimensions["s"].size)
        gx_time = np.asarray(root.groups["Grids"].variables["time"][:], dtype=float)
    if args.time_index_start < 0 or args.time_index_start >= gx_time.size:
        raise ValueError(f"time-index-start={args.time_index_start} outside [0, {gx_time.size - 1}]")
    if args.time_index_stop < 0 or args.time_index_stop >= gx_time.size:
        raise ValueError(f"time-index-stop={args.time_index_stop} outside [0, {gx_time.size - 1}]")
    if args.time_index_stop <= args.time_index_start:
        raise ValueError("time-index-stop must be greater than time-index-start")

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
    gx_G_stop = _load_species_state(
        args.gx_dir,
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
        time_index=args.time_index_stop,
    )
    gx_phi_stop = _load_field(args.gx_dir / f"diag_state_phi_t{args.time_index_stop}.bin", nyc, nx, nz)
    gx_apar_stop = _maybe_load_field(args.gx_dir / f"diag_state_apar_t{args.time_index_stop}.bin", nyc, nx, nz)
    gx_bpar_stop = _maybe_load_field(args.gx_dir / f"diag_state_bpar_t{args.time_index_stop}.bin", nyc, nx, nz)

    y0 = float(gx_contract.y0) if np.isfinite(float(gx_contract.y0)) else _infer_y0(gx_ky)
    ny_full = _resolve_imported_real_fft_ny(gx_ky, gx_contract)
    if gx_contract.geo_option == "slab":
        geom = SlabGeometry.from_config(
            GeometryConfig(model="slab", s_hat=float(gx_contract.s_hat), zero_shat=bool(gx_contract.zero_shat))
        )
    else:
        geom = load_gx_geometry_netcdf(_select_geometry_source(args.gx_out, args.geometry_file, gx_contract))

    boundary_eff = _resolve_imported_boundary(gx_contract.boundary, zero_shat=bool(gx_contract.zero_shat))
    lx = 2.0 * np.pi * y0 if boundary_eff == "periodic" else 62.8
    grid_cfg = apply_gx_geometry_grid_defaults(
        geom,
        GridConfig(
            Nx=int(nx),
            Ny=int(ny_full),
            Nz=int(nz),
            Lx=lx,
            Ly=2.0 * np.pi * y0,
            boundary=boundary_eff,
            y0=y0,
            nperiod=max(1, int(gx_contract.nperiod)),
            ntheta=max(1, int(gx_contract.ntheta)),
        ),
    )
    grid_full = build_spectral_grid(grid_cfg)
    grid = select_gx_real_fft_ky_grid(grid_full, gx_ky.astype(np.float32))

    params = build_linear_params(
        gx_contract.species,
        tau_e=float(gx_contract.tau_e),
        kpar_scale=float(geom.gradpar()),
        beta=float(gx_contract.beta),
        fapar=float(gx_contract.fapar),
    )
    terms = _build_imported_linear_terms(gx_contract)
    if gx_contract.hypercollisions:
        params = _apply_gx_hypercollisions(params, nhermite=nm)
    params = replace(
        params,
        D_hyper=float(gx_contract.D_hyper),
        damp_ends_amp=float(gx_contract.damp_ends_amp),
        damp_ends_widthfrac=float(gx_contract.damp_ends_widthfrac),
    )

    cache = build_linear_cache(grid, geom, params, nl, nm)
    dt = _infer_gx_linear_dt(gx_time, gx_contract)
    time_cfg = GXTimeConfig(
        dt=dt,
        t_max=float(gx_time[args.time_index_stop] - gx_time[args.time_index_start]),
        method=str(gx_contract.scheme),
        sample_stride=max(1, int(gx_contract.nwrite)),
        fixed_dt=bool((gx_contract.dt is not None) or _gx_has_uniform_linear_dt(gx_time, gx_contract)),
        cfl_fac=resolve_cfl_fac(str(gx_contract.scheme), None),
    )
    dt_min = float(time_cfg.dt_min)
    dt_max = float(time_cfg.dt_max) if time_cfg.dt_max is not None else float(time_cfg.dt)

    G = jnp.asarray(gx_G_start, dtype=jnp.complex64)
    omega_max = _gx_linear_omega_max(grid, geom, params, nl, nm)
    wmax = float(np.sum(omega_max))
    t = 0.0
    target = float(time_cfg.t_max)

    def _step(G_state, cache_state, params_state, term_cfg_state, dt_state):
        return _linear_explicit_step(
            G_state,
            cache_state,
            params_state,
            term_cfg_state,
            dt_state,
            method=time_cfg.method,
        )

    stepper = jax.jit(_step, donate_argnums=(0,))
    term_cfg = _gx_term_config(terms)
    step_count = 0
    while t < target - 1.0e-12:
        dt_step = float(time_cfg.dt)
        if not time_cfg.fixed_dt and wmax > 0.0:
            dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
            dt_step = min(max(dt_guess, dt_min), dt_max)
        remaining = target - t
        if dt_step > remaining:
            dt_step = max(remaining, dt_min)
        G, fields = stepper(G, cache, params, term_cfg, dt_step)
        t += dt_step
        step_count += 1

    sp_G_stop = np.asarray(G, dtype=np.complex64)
    sp_phi_stop = np.asarray(fields.phi, dtype=np.complex64)
    sp_apar_stop = (
        np.asarray(fields.apar, dtype=np.complex64)
        if fields.apar is not None
        else np.zeros_like(gx_phi_stop, dtype=np.complex64)
    )
    sp_bpar_stop = (
        np.asarray(fields.bpar, dtype=np.complex64)
        if fields.bpar is not None
        else np.zeros_like(gx_phi_stop, dtype=np.complex64)
    )
    gx_apar_stop_use = gx_apar_stop if gx_apar_stop is not None else np.zeros_like(gx_phi_stop, dtype=np.complex64)
    gx_bpar_stop_use = gx_bpar_stop if gx_bpar_stop is not None else np.zeros_like(gx_phi_stop, dtype=np.complex64)

    print(
        f"time_index_start={args.time_index_start} t_start={gx_time[args.time_index_start]:.8f} "
        f"time_index_stop={args.time_index_stop} t_stop={gx_time[args.time_index_stop]:.8f} "
        f"delta_t={target:.8f} steps={step_count} t_match={t:.8f}"
    )
    _summary("start_phi", gx_phi_start.astype(np.complex64), gx_phi_start.astype(np.complex64))
    if gx_apar_start is not None:
        _summary("start_apar", gx_apar_start.astype(np.complex64), gx_apar_start.astype(np.complex64))
    _summary("stop_g_state", gx_G_stop.astype(np.complex64), sp_G_stop)
    _summary("stop_phi", gx_phi_stop.astype(np.complex64), sp_phi_stop)
    if gx_apar_stop is not None or fields.apar is not None:
        _summary("stop_apar", gx_apar_stop_use.astype(np.complex64), sp_apar_stop)
    if gx_bpar_stop is not None or fields.bpar is not None:
        _summary("stop_bpar", gx_bpar_stop_use.astype(np.complex64), sp_bpar_stop)

    vol_fac, _flux_fac = gx_volume_factors(geom, grid)
    gx_Wg_stop = float(gx_Wg(jnp.asarray(gx_G_stop), grid, params, vol_fac))
    sp_Wg_stop = float(gx_Wg(jnp.asarray(sp_G_stop), grid, params, vol_fac))
    gx_Wphi_stop = float(gx_Wphi(jnp.asarray(gx_phi_stop), cache, params, vol_fac))
    sp_Wphi_stop = float(gx_Wphi(jnp.asarray(sp_phi_stop), cache, params, vol_fac))
    gx_Wapar_stop = float(gx_Wapar(jnp.asarray(gx_apar_stop_use), cache, vol_fac))
    sp_Wapar_stop = float(gx_Wapar(jnp.asarray(sp_apar_stop), cache, vol_fac))
    gx_Phi2_stop = _gx_phi2_total(jnp.asarray(gx_phi_stop), vol_fac)
    sp_Phi2_stop = _gx_phi2_total(jnp.asarray(sp_phi_stop), vol_fac)

    rows = [
        {"metric": "g_state", "rel": _rel_err(sp_G_stop, gx_G_stop)},
        {"metric": "phi", "rel": _rel_err(sp_phi_stop, gx_phi_stop)},
        {"metric": "apar", "rel": _rel_err(sp_apar_stop, gx_apar_stop_use)},
        {"metric": "bpar", "rel": _rel_err(sp_bpar_stop, gx_bpar_stop_use)},
        {"metric": "Wg", "gx_stop": gx_Wg_stop, "spectrax": sp_Wg_stop, "rel": abs(sp_Wg_stop - gx_Wg_stop) / max(abs(gx_Wg_stop), 1.0e-30)},
        {"metric": "Wphi", "gx_stop": gx_Wphi_stop, "spectrax": sp_Wphi_stop, "rel": abs(sp_Wphi_stop - gx_Wphi_stop) / max(abs(gx_Wphi_stop), 1.0e-30)},
        {"metric": "Wapar", "gx_stop": gx_Wapar_stop, "spectrax": sp_Wapar_stop, "rel": abs(sp_Wapar_stop - gx_Wapar_stop) / max(abs(gx_Wapar_stop), 1.0e-30)},
        {"metric": "Phi2", "gx_stop": gx_Phi2_stop, "spectrax": sp_Phi2_stop, "rel": abs(sp_Phi2_stop - gx_Phi2_stop) / max(abs(gx_Phi2_stop), 1.0e-30)},
    ]
    print("metric     rel")
    for row in rows[:4]:
        print(f"{row['metric']:8s} {float(row['rel']): .3e}")
    print("diag       gx_stop       spectrax      rel")
    for row in rows[4:]:
        print(f"{row['metric']:8s} {float(row['gx_stop']): .6e} {float(row['spectrax']): .6e} {float(row['rel']): .3e}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
