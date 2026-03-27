#!/usr/bin/env python3
"""Compare one exact imported-linear GX omega_kxkyt interval against late diag_state dumps."""

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
    _gx_has_uniform_linear_dt,
    _gx_term_config,
    _infer_gx_linear_dt,
    _load_gx_input_contract,
    _resolve_imported_boundary,
    _resolve_imported_real_fft_ny,
    _select_geometry_source,
)
from compare_gx_rhs_terms import _infer_y0
from compare_gx_runtime_diag_state import (
    _load_field,
    _load_real_vector_auto,
    _load_species_state,
    _maybe_load_field,
)
from spectraxgk.benchmarks import _apply_gx_hypercollisions
from spectraxgk.config import GeometryConfig, GridConfig, resolve_cfl_fac
from spectraxgk.geometry import SlabGeometry, apply_gx_geometry_grid_defaults, load_gx_geometry_netcdf
from spectraxgk.grids import build_spectral_grid, select_gx_real_fft_ky_grid
from spectraxgk.gx_integrators import (
    GXTimeConfig,
    _gx_growth_rate_step,
    _gx_linear_omega_max,
    _gx_midplane_index,
    _linear_explicit_step,
)
from spectraxgk.linear import LinearTerms, build_linear_cache
from spectraxgk.species import build_linear_params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gx-dir-start", type=Path, required=True, help="Directory containing the start diag_state_* dump set.")
    p.add_argument("--gx-dir-stop", type=Path, required=True, help="Directory containing the stop diag_state_* dump set.")
    p.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file for times and omega_kxkyt.")
    p.add_argument("--gx-input", type=Path, required=True, help="GX input file describing the imported contract.")
    p.add_argument("--geometry-file", type=Path, required=True, help="Imported geometry file used by SPECTRAX.")
    p.add_argument("--time-index-start", type=int, required=True, help="GX diagnostic start index.")
    p.add_argument("--time-index-stop", type=int, required=True, help="GX diagnostic stop index.")
    p.add_argument("--ky", type=float, default=None, help="Optional ky value to score. Defaults to the smallest positive ky.")
    p.add_argument("--kx", type=float, default=0.0, help="Optional kx value to score. Defaults to 0.")
    p.add_argument("--out", type=Path, default=None, help="Optional CSV output path.")
    return p


def _gx_growth_pair(phi_now: np.ndarray, phi_prev: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    z_index = _gx_midplane_index(phi_now.shape[-1])
    phi_now_j = jnp.asarray(phi_now)
    phi_prev_j = jnp.asarray(phi_prev)
    mask = jnp.ones(phi_now.shape[:2], dtype=bool)
    gamma, omega = _gx_growth_rate_step(
        phi_now_j,
        phi_prev_j,
        dt,
        z_index=z_index,
        mask=mask,
        mode_method="z_index",
    )
    return np.asarray(gamma, dtype=float), np.asarray(omega, dtype=float)


def _select_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def main() -> None:
    args = build_parser().parse_args()

    gx_contract = _load_gx_input_contract(args.gx_input)
    with Dataset(args.gx_out, "r") as root:
        gx_time = np.asarray(root.groups["Grids"].variables["time"][:], dtype=float)
        gx_omega = np.asarray(root.groups["Diagnostics"].variables["omega_kxkyt"][:], dtype=float)
        nl = int(root.dimensions["l"].size)
        nm = int(root.dimensions["m"].size)
        nspec = int(root.dimensions["s"].size)
    if args.time_index_stop <= args.time_index_start:
        raise ValueError("time-index-stop must be greater than time-index-start")

    gx_kx = _load_real_vector_auto(args.gx_dir_start / f"diag_state_kx_t{args.time_index_start}.bin")
    gx_ky = _load_real_vector_auto(args.gx_dir_start / f"diag_state_ky_t{args.time_index_start}.bin")
    nyc = int(gx_ky.size)
    nx = int(gx_kx.size)
    phi_raw = np.fromfile(args.gx_dir_start / f"diag_state_phi_t{args.time_index_start}.bin", dtype=np.complex64)
    if phi_raw.size % max(nyc * nx, 1) != 0:
        raise ValueError(
            f"diag_state_phi_t{args.time_index_start}.bin size {phi_raw.size} is not divisible by nyc*nx={nyc*nx}"
        )
    nz = int(phi_raw.size // (nyc * nx))

    gx_G_start = _load_species_state(
        args.gx_dir_start,
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
        time_index=args.time_index_start,
    )
    gx_phi_start = _load_field(args.gx_dir_start / f"diag_state_phi_t{args.time_index_start}.bin", nyc, nx, nz)
    gx_phi_stop = _load_field(args.gx_dir_stop / f"diag_state_phi_t{args.time_index_stop}.bin", nyc, nx, nz)
    gx_apar_stop = _maybe_load_field(args.gx_dir_stop / f"diag_state_apar_t{args.time_index_stop}.bin", nyc, nx, nz)
    gx_bpar_stop = _maybe_load_field(args.gx_dir_stop / f"diag_state_bpar_t{args.time_index_stop}.bin", nyc, nx, nz)

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
    terms = LinearTerms()
    if gx_contract.hypercollisions:
        params = _apply_gx_hypercollisions(params, nhermite=nm)
    params = replace(
        params,
        D_hyper=float(gx_contract.D_hyper),
        damp_ends_amp=float(gx_contract.damp_ends_amp),
        damp_ends_widthfrac=float(gx_contract.damp_ends_widthfrac),
    )
    terms = replace(
        terms,
        hypercollisions=1.0 if gx_contract.hypercollisions else 0.0,
        hyperdiffusion=1.0 if gx_contract.hyper else 0.0,
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
    _ = (gx_apar_stop, gx_bpar_stop, sp_apar_stop, sp_bpar_stop)

    gamma_gx_dump, omega_gx_dump = _gx_growth_pair(gx_phi_stop, gx_phi_start, target)
    gamma_sp_dump, omega_sp_dump = _gx_growth_pair(sp_phi_stop, gx_phi_start, target)

    ky_target = float(args.ky) if args.ky is not None else float(np.min(gx_ky[gx_ky > 0.0]))
    ky_idx = _select_index(gx_ky, ky_target)
    kx_idx = _select_index(gx_kx, float(args.kx))

    row = {
        "time_index_start": int(args.time_index_start),
        "time_index_stop": int(args.time_index_stop),
        "t_start": float(gx_time[args.time_index_start]),
        "t_stop": float(gx_time[args.time_index_stop]),
        "delta_t": float(target),
        "ky": float(gx_kx.size and gx_ky[ky_idx]),
        "kx": float(gx_kx[kx_idx]),
        "omega_out": float(gx_omega[args.time_index_stop, ky_idx, kx_idx, 0]),
        "gamma_out": float(gx_omega[args.time_index_stop, ky_idx, kx_idx, 1]),
        "omega_gx_dump": float(omega_gx_dump[ky_idx, kx_idx]),
        "gamma_gx_dump": float(gamma_gx_dump[ky_idx, kx_idx]),
        "omega_sp_dump": float(omega_sp_dump[ky_idx, kx_idx]),
        "gamma_sp_dump": float(gamma_sp_dump[ky_idx, kx_idx]),
    }
    row["abs_omega_out_vs_gx_dump"] = abs(row["omega_out"] - row["omega_gx_dump"])
    row["abs_gamma_out_vs_gx_dump"] = abs(row["gamma_out"] - row["gamma_gx_dump"])
    row["abs_omega_sp_vs_gx_dump"] = abs(row["omega_sp_dump"] - row["omega_gx_dump"])
    row["abs_gamma_sp_vs_gx_dump"] = abs(row["gamma_sp_dump"] - row["gamma_gx_dump"])
    row["rel_omega_sp_vs_gx_dump"] = row["abs_omega_sp_vs_gx_dump"] / max(abs(row["omega_gx_dump"]), 1.0e-12)
    row["rel_gamma_sp_vs_gx_dump"] = row["abs_gamma_sp_vs_gx_dump"] / max(abs(row["gamma_gx_dump"]), 1.0e-12)

    df = pd.DataFrame([row])
    print(df.to_string(index=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
