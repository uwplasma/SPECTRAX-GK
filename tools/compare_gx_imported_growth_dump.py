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

from tools.compare_gx_imported_linear import (
    _build_imported_initial_condition,
    _build_imported_linear_terms,
    _gx_has_uniform_linear_dt,
    _gx_term_config,
    _infer_gx_linear_dt,
    _load_gx_input_contract,
    _resolve_imported_boundary,
    _resolve_imported_real_fft_ny,
    _resolve_internal_geometry_source,
)
from tools.compare_gx_rhs_terms import _infer_y0
from tools.compare_gx_runtime_diag_state import (
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
    p.add_argument(
        "--gx-restart-start",
        type=Path,
        default=None,
        help="Optional GX restart.nc file holding the exact start distribution state for late-window replay.",
    )
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


def _load_growth_dt(path: Path) -> float:
    raw64 = np.fromfile(path, dtype=np.float64)
    if raw64.size == 1:
        return float(raw64[0])
    raw32 = np.fromfile(path, dtype=np.float32)
    if raw32.size == 1:
        return float(raw32[0])
    raise ValueError(f"unexpected growth dt payload in {path}")


def _gx_active_kx_count(nx_full: int) -> int:
    return 1 + 2 * ((int(nx_full) - 1) // 3)


def _gx_active_ky_count(ny_full: int) -> int:
    return 1 + ((int(ny_full) - 1) // 3)


def _expand_gx_restart_state_to_full_positive_ky(
    state_active: np.ndarray,
    *,
    ny_full: int,
    nx_full: int,
) -> np.ndarray:
    state_active = np.asarray(state_active, dtype=np.complex64)
    if state_active.ndim != 6:
        raise ValueError(f"restart state must have rank 6, got {state_active.shape}")
    nspec, nl, nm, naky, nakx, nz = state_active.shape
    nyc_full = int(ny_full) // 2 + 1
    expected_naky = _gx_active_ky_count(int(ny_full))
    expected_nakx = _gx_active_kx_count(int(nx_full))
    if naky != expected_naky:
        raise ValueError(f"restart Nky={naky} does not match ny_full={ny_full} (expected {expected_naky})")
    if nakx != expected_nakx:
        raise ValueError(f"restart Nkx={nakx} does not match nx_full={nx_full} (expected {expected_nakx})")

    out = np.zeros((nspec, nl, nm, nyc_full, int(nx_full), nz), dtype=np.complex64)
    split = 1 + ((int(nx_full) - 1) // 3)
    out[..., :naky, :split, :] = state_active[..., :split, :]
    if int(nx_full) > 1:
        for i in range(2 * int(nx_full) // 3 + 1, int(nx_full)):
            it = i - 2 * int(nx_full) // 3 + ((int(nx_full) - 1) // 3)
            out[..., :naky, i, :] = state_active[..., it, :]
    return out


def _load_gx_restart_state(path: Path) -> np.ndarray:
    with Dataset(path, "r") as root:
        if "G" not in root.variables:
            raise ValueError(f"restart file {path} does not contain variable 'G'")
        raw = np.asarray(root.variables["G"][:], dtype=float)
    if raw.ndim != 7 or raw.shape[-1] != 2:
        raise ValueError(f"unexpected GX restart G shape {raw.shape}")
    state = raw[..., 0] + 1j * raw[..., 1]
    # GX restart layout: (species, m, l, z, kx, ky) -> SPECTRAX: (species, l, m, ky, kx, z)
    return np.asarray(np.transpose(state, (0, 2, 1, 5, 4, 3)), dtype=np.complex64)


def _load_gx_restart_time(path: Path) -> float:
    with Dataset(path, "r") as root:
        if "time" not in root.variables:
            raise ValueError(f"restart file {path} does not contain variable 'time'")
        return float(np.asarray(root.variables["time"][:], dtype=float).reshape(-1)[0])


def main() -> None:
    args = build_parser().parse_args()

    gx_contract = _load_gx_input_contract(args.gx_input)
    with Dataset(args.gx_out, "r") as root:
        gx_time = np.asarray(root.groups["Grids"].variables["time"][:], dtype=float)
        gx_omega = np.asarray(root.groups["Diagnostics"].variables["omega_kxkyt"][:], dtype=float)
        nl = int(root.dimensions["l"].size)
        nm = int(root.dimensions["m"].size)
        nspec = int(root.dimensions["s"].size)
    growth_phi_prev_path = args.gx_dir_stop / f"diag_growth_phi_prev_t{args.time_index_stop}.bin"
    growth_phi_path = args.gx_dir_stop / f"diag_growth_phi_t{args.time_index_stop}.bin"
    growth_dt_path = args.gx_dir_stop / f"diag_growth_dt_t{args.time_index_stop}.bin"
    growth_kx_path = args.gx_dir_stop / f"diag_growth_kx_t{args.time_index_stop}.bin"
    growth_ky_path = args.gx_dir_stop / f"diag_growth_ky_t{args.time_index_stop}.bin"
    stop_has_growth = all(
        path.exists()
        for path in (growth_phi_prev_path, growth_phi_path, growth_dt_path, growth_kx_path, growth_ky_path)
    )
    start_has_state = (args.gx_dir_start / f"diag_state_G_s0_t{args.time_index_start}.bin").exists()
    if stop_has_growth:
        if args.time_index_stop < args.time_index_start:
            raise ValueError("time-index-stop must be >= time-index-start in growth-dump mode")
    elif args.time_index_stop <= args.time_index_start:
        raise ValueError("time-index-stop must be greater than time-index-start")

    if stop_has_growth:
        gx_kx = _load_real_vector_auto(growth_kx_path)
        gx_ky = _load_real_vector_auto(growth_ky_path)
    else:
        gx_kx = _load_real_vector_auto(args.gx_dir_start / f"diag_state_kx_t{args.time_index_start}.bin")
        gx_ky = _load_real_vector_auto(args.gx_dir_start / f"diag_state_ky_t{args.time_index_start}.bin")
    nyc = int(gx_ky.size)
    nx = int(gx_kx.size)
    phi_seed_path = growth_phi_prev_path if stop_has_growth else args.gx_dir_start / f"diag_state_phi_t{args.time_index_start}.bin"
    phi_raw = np.fromfile(phi_seed_path, dtype=np.complex64)
    if phi_raw.size % max(nyc * nx, 1) != 0:
        raise ValueError(f"{phi_seed_path.name} size {phi_raw.size} is not divisible by nyc*nx={nyc*nx}")
    nz = int(phi_raw.size // (nyc * nx))

    if stop_has_growth:
        gx_phi_start = _load_field(growth_phi_prev_path, nyc, nx, nz)
        gx_phi_stop = _load_field(growth_phi_path, nyc, nx, nz)
        target = _load_growth_dt(growth_dt_path)
        gx_apar_stop = None
        gx_bpar_stop = None
        gx_G_start = None
        restart_time = None
        restart_state_active = None
        if args.gx_restart_start is not None:
            restart_state_active = _load_gx_restart_state(args.gx_restart_start)
            restart_time = _load_gx_restart_time(args.gx_restart_start)
        elif start_has_state:
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
    else:
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
        geom = load_gx_geometry_netcdf(_resolve_internal_geometry_source(geometry_file=args.geometry_file, runtime_config=None))

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

    if stop_has_growth and args.gx_restart_start is not None:
        if restart_state_active is None:
            raise ValueError("restart_state_active must be available for restart-based growth replay")
        gx_G_start = _expand_gx_restart_state_to_full_positive_ky(
            restart_state_active,
            ny_full=ny_full,
            nx_full=int(nx),
        )
        gx_G_start = np.asarray(gx_G_start, dtype=np.complex64) * np.complex64(gx_contract.restart_scale)
        if gx_contract.restart_with_perturb:
            gx_G_start = gx_G_start + np.asarray(
                _build_imported_initial_condition(
                    grid=grid,
                    geom=geom,
                    gx_contract=gx_contract,
                    species=gx_contract.species,
                    ky_index=0,
                    kx_index=0,
                    Nl=nl,
                    Nm=nm,
                ),
                dtype=np.complex64,
            )

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
        t_max=float(gx_time[args.time_index_stop] - gx_time[args.time_index_start]) if not stop_has_growth else float(target),
        method=str(gx_contract.scheme),
        sample_stride=max(1, int(gx_contract.nwrite)),
        fixed_dt=bool((gx_contract.dt is not None) or _gx_has_uniform_linear_dt(gx_time, gx_contract)),
        cfl_fac=resolve_cfl_fac(str(gx_contract.scheme), None),
    )
    dt_min = float(time_cfg.dt_min)
    dt_max = float(time_cfg.dt_max) if time_cfg.dt_max is not None else float(time_cfg.dt)

    gamma_gx_dump, omega_gx_dump = _gx_growth_pair(gx_phi_stop, gx_phi_start, target)
    if stop_has_growth and gx_G_start is None:
        gamma_sp_dump, omega_sp_dump = gamma_gx_dump, omega_gx_dump
    else:
        G = jnp.asarray(gx_G_start, dtype=jnp.complex64)
        omega_max = _gx_linear_omega_max(grid, geom, params, nl, nm)
        wmax = float(np.sum(omega_max))
        t = 0.0
        phi_prev_step = gx_phi_start

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
        if stop_has_growth and args.gx_restart_start is not None:
            if restart_time is None:
                raise ValueError("restart_time must be available for restart-based growth replay")
            target = float(gx_time[args.time_index_stop] - restart_time)
            step_dt = float(_load_growth_dt(growth_dt_path))
            if step_dt <= 0.0:
                raise ValueError("growth dump dt must be > 0")
            nsteps_float = target / step_dt
            nsteps = int(np.rint(nsteps_float))
            if nsteps < 1 or not np.isclose(target, step_dt * nsteps, rtol=1.0e-6, atol=1.0e-10):
                raise ValueError(
                    "restart-based growth replay requires a uniform late window; "
                    f"got target={target:.12g}, step_dt={step_dt:.12g}, ratio={nsteps_float:.12g}"
                )
            for _ in range(nsteps):
                G, fields = stepper(G, cache, params, term_cfg, step_dt)
                t += step_dt
                if t < target - 0.5 * step_dt:
                    phi_prev_step = np.asarray(fields.phi, dtype=np.complex64)
            gamma_sp_dump, omega_sp_dump = _gx_growth_pair(np.asarray(fields.phi, dtype=np.complex64), phi_prev_step, step_dt)
        else:
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
            gamma_sp_dump, omega_sp_dump = _gx_growth_pair(sp_phi_stop, gx_phi_start, target)

    ky_target = float(args.ky) if args.ky is not None else float(np.min(gx_ky[gx_ky > 0.0]))
    ky_idx = _select_index(gx_ky, ky_target)
    kx_idx = _select_index(gx_kx, float(args.kx))

    row = {
        "time_index_start": int(args.time_index_start),
        "time_index_stop": int(args.time_index_stop),
        "t_start": float(gx_time[args.time_index_start]),
        "t_restart_start": (float(restart_time) if stop_has_growth and args.gx_restart_start is not None else np.nan),
        "t_stop": float(gx_time[args.time_index_stop]),
        "delta_t": float(target),
        "compare_mode": (
            "growth_dump"
            if stop_has_growth and gx_G_start is None
            else (
                "growth_restart_replay"
                if stop_has_growth and args.gx_restart_start is not None
                else ("growth_replay" if stop_has_growth else "state_replay")
            )
        ),
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
