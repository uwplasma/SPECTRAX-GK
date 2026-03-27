#!/usr/bin/env python3
"""Compare GX big-field linear Phi(time,z) against imported SPECTRAX on matching samples."""

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
    _build_imported_initial_condition,
    _build_sample_steps,
    _gx_has_uniform_linear_dt,
    _gx_linear_omega_max,
    _gx_term_config,
    _infer_gx_linear_dt,
    _infer_y0,
    _load_gx_input_contract,
    _resolve_imported_boundary,
    _resolve_imported_real_fft_ny,
    _select_geometry_source,
)
from spectraxgk.analysis import ModeSelection, gx_growth_rate_from_phi, select_ky_index
from spectraxgk.benchmarks import _apply_gx_hypercollisions
from spectraxgk.config import GeometryConfig, GridConfig, resolve_cfl_fac
from spectraxgk.geometry import SlabGeometry, apply_gx_geometry_grid_defaults, load_gx_geometry_netcdf
from spectraxgk.grids import build_spectral_grid, select_gx_real_fft_ky_grid
from spectraxgk.gx_integrators import GXTimeConfig, _linear_explicit_step, _gx_midplane_index
from spectraxgk.linear import LinearTerms, build_linear_cache
from spectraxgk.species import build_linear_params
from spectraxgk.terms.assembly import assemble_rhs_cached


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gx-big", type=Path, required=True, help="GX .big.nc file containing Diagnostics/Phi.")
    p.add_argument("--geometry-file", type=Path, required=True, help="Imported geometry file passed to SPECTRAX.")
    p.add_argument("--gx-input", type=Path, required=True, help="GX input file used to build the imported contract.")
    p.add_argument("--ky", type=float, required=True, help="ky value to score.")
    p.add_argument("--kx", type=float, default=0.0, help="kx value to score.")
    p.add_argument("--mode-method", choices=("z_index", "max", "project", "svd"), default="project")
    p.add_argument("--sample-step-stride", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--sample-window", choices=("head", "tail"), default="tail")
    p.add_argument("--out", type=Path, default=None)
    return p


def _load_gx_big_phi(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Dataset(path, "r")
    try:
        grids = root.groups["Grids"]
        diag = root.groups["Diagnostics"]
        time = np.asarray(grids.variables["time"][:], dtype=float)
        ky = np.asarray(grids.variables["ky"][:], dtype=float)
        kx = np.asarray(grids.variables["kx"][:], dtype=float)
        theta = np.asarray(grids.variables["theta"][:], dtype=float)
        phi_ri = np.asarray(diag.variables["Phi"][:], dtype=float)
    finally:
        root.close()
    phi = phi_ri[..., 0] + 1j * phi_ri[..., 1]
    return time, ky, kx, theta, np.asarray(phi, dtype=np.complex64)


def _integrate_phi_samples(
    *,
    G0: jnp.ndarray,
    grid,
    geom,
    cache,
    params,
    time_cfg: GXTimeConfig,
    terms: LinearTerms,
    ky_index: int,
    kx_index: int,
    sample_times: np.ndarray,
) -> np.ndarray:
    G = jnp.asarray(G0)
    t = 0.0
    target_times = np.asarray(sample_times, dtype=float)
    phi_samples: list[np.ndarray] = []
    target_idx = 0
    omega_max = _gx_linear_omega_max(grid, geom, params, G.shape[-5], G.shape[-4])
    wmax = float(np.sum(omega_max))
    dt_min = float(time_cfg.dt_min)
    dt_max = float(time_cfg.dt_max) if time_cfg.dt_max is not None else float(time_cfg.dt)

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

    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg)
    if target_times.size > 0 and target_times[0] <= 1.0e-14:
        phi_samples.append(np.asarray(fields0.phi)[ky_index : ky_index + 1, kx_index : kx_index + 1, :])
        target_idx = 1

    while target_idx < target_times.size:
        dt_step = float(time_cfg.dt)
        if not time_cfg.fixed_dt and wmax > 0.0:
            dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
            dt_step = min(max(dt_guess, dt_min), dt_max)
        remaining = float(target_times[target_idx] - t)
        if dt_step > remaining:
            dt_step = max(remaining, dt_min)
        G, fields = stepper(G, cache, params, term_cfg, dt_step)
        t += dt_step
        if t >= target_times[target_idx] - 1.0e-12:
            phi_samples.append(np.asarray(fields.phi)[ky_index : ky_index + 1, kx_index : kx_index + 1, :])
            target_idx += 1

    return np.asarray(phi_samples, dtype=np.complex64)


def main() -> None:
    args = build_parser().parse_args()
    gx_time, gx_ky, gx_kx, gx_theta, gx_phi = _load_gx_big_phi(args.gx_big)
    sample_steps = _build_sample_steps(
        gx_time,
        sample_step_stride=int(args.sample_step_stride),
        max_samples=args.max_samples,
        sample_window=str(args.sample_window),
    )
    sample_times = np.asarray(gx_time[sample_steps], dtype=float)
    ky_idx_big = select_ky_index(gx_ky, float(args.ky))
    kx_idx_big = int(np.argmin(np.abs(gx_kx - float(args.kx))))
    gx_phi_sel = np.asarray(gx_phi[sample_steps, ky_idx_big : ky_idx_big + 1, kx_idx_big : kx_idx_big + 1, :], dtype=np.complex64)

    gx_contract = _load_gx_input_contract(args.gx_input)
    y0 = float(gx_contract.y0) if np.isfinite(float(gx_contract.y0)) else _infer_y0(gx_ky)
    ny_full = _resolve_imported_real_fft_ny(gx_ky, gx_contract)
    if gx_contract.geo_option == "slab":
        geom = SlabGeometry.from_config(
            GeometryConfig(model="slab", s_hat=float(gx_contract.s_hat), zero_shat=bool(gx_contract.zero_shat))
        )
    else:
        geom = load_gx_geometry_netcdf(_select_geometry_source(args.gx_big, args.geometry_file, gx_contract))

    boundary_eff = _resolve_imported_boundary(gx_contract.boundary, zero_shat=bool(gx_contract.zero_shat))
    lx = 2.0 * np.pi * y0 if boundary_eff == "periodic" else 62.8
    grid_cfg = apply_gx_geometry_grid_defaults(
        geom,
        GridConfig(
            Nx=max(1, int(gx_kx.size)),
            Ny=int(ny_full),
            Nz=int(gx_theta.size),
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
    ky_idx_local = select_ky_index(np.asarray(grid.ky), float(args.ky))
    kx_idx_local = int(np.argmin(np.abs(np.asarray(grid.kx) - float(args.kx))))

    nl = int(gx_contract.nlaguerre)
    nm = int(gx_contract.nhermite)
    G0 = _build_imported_initial_condition(
        grid=grid,
        geom=geom,
        gx_contract=gx_contract,
        species=tuple(gx_contract.species),
        ky_index=ky_idx_local,
        kx_index=kx_idx_local,
        Nl=nl,
        Nm=nm,
    )
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
        t_max=float(sample_times[-1]),
        method=str(gx_contract.scheme),
        sample_stride=max(1, int(gx_contract.nwrite)),
        fixed_dt=bool((gx_contract.dt is not None) or _gx_has_uniform_linear_dt(gx_time, gx_contract)),
        cfl_fac=resolve_cfl_fac(str(gx_contract.scheme), None),
    )
    sp_phi_sel = _integrate_phi_samples(
        G0=G0,
        grid=grid,
        geom=geom,
        cache=cache,
        params=params,
        time_cfg=time_cfg,
        terms=terms,
        ky_index=ky_idx_local,
        kx_index=kx_idx_local,
        sample_times=sample_times,
    )

    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_gx_midplane_index(int(gx_theta.size)))
    gx_gamma, gx_omega, gx_gamma_t, gx_omega_t, _gx_tmid = gx_growth_rate_from_phi(
        gx_phi_sel,
        sample_times,
        sel,
        use_last=False,
        mode_method=str(args.mode_method),
    )
    sp_gamma, sp_omega, sp_gamma_t, sp_omega_t, _sp_tmid = gx_growth_rate_from_phi(
        sp_phi_sel,
        sample_times,
        sel,
        use_last=False,
        mode_method=str(args.mode_method),
    )

    row = {
        "ky": float(args.ky),
        "kx": float(args.kx),
        "mode_method": str(args.mode_method),
        "sample_count": int(sample_times.size),
        "gx_gamma": float(gx_gamma),
        "sp_gamma": float(sp_gamma),
        "rel_gamma": abs(float(sp_gamma) - float(gx_gamma)) / max(abs(float(gx_gamma)), 1.0e-12),
        "gx_omega": float(gx_omega),
        "sp_omega": float(sp_omega),
        "rel_omega": abs(float(sp_omega) - float(gx_omega)) / max(abs(float(gx_omega)), 1.0e-12),
        "last_gamma_gx": float(np.asarray(gx_gamma_t)[-1]),
        "last_gamma_sp": float(np.asarray(sp_gamma_t)[-1]),
        "last_omega_gx": float(np.asarray(gx_omega_t)[-1]),
        "last_omega_sp": float(np.asarray(sp_omega_t)[-1]),
    }
    df = pd.DataFrame([row])
    print(df.to_string(index=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
