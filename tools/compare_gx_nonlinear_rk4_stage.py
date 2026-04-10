#!/usr/bin/env python3
"""Compare GX nonlinear RK4 partial-stage dumps against SPECTRAX reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from compare_gx_rhs_terms import _load_bin, _load_field, _load_shape, _reshape_gx, _summary
from spectraxgk.geometry import apply_gx_geometry_grid_defaults, ensure_flux_tube_geometry_data
from spectraxgk.grids import build_spectral_grid, gx_real_fft_ky, select_gx_real_fft_ky_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.linear import build_linear_cache
from spectraxgk.runtime import build_runtime_geometry, build_runtime_linear_params, build_runtime_term_config
from spectraxgk.terms.assembly import assemble_rhs_terms_cached, compute_fields_cached
from spectraxgk.terms.nonlinear import nonlinear_em_contribution


@dataclass(frozen=True)
class NonlinearRK4StageStates:
    k1_linear: np.ndarray
    k1_nonlinear: np.ndarray
    k1_total: np.ndarray
    k2_linear: np.ndarray
    k2_nonlinear: np.ndarray
    k2_total: np.ndarray
    k3_linear: np.ndarray
    k3_nonlinear: np.ndarray
    k3_total: np.ndarray
    k4_linear: np.ndarray
    k4_nonlinear: np.ndarray
    k4_total: np.ndarray
    g2: np.ndarray
    g3: np.ndarray
    g4: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX startup and rk4_partial dumps")
    parser.add_argument("--config", type=Path, required=True, help="Runtime TOML config used by SPECTRAX")
    parser.add_argument("--partial-call", type=int, choices=[1, 2, 3], default=1, help="Which RK4 partial state to compare")
    parser.add_argument("--dt", type=float, default=None, help="Optional dt override; defaults to the runtime config")
    return parser


def _split_rhs(
    G: np.ndarray,
    *,
    cache,
    params,
    term_cfg,
    gx_real_fft: bool,
    laguerre_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    linear_total, fields, _contrib = assemble_rhs_terms_cached(jnp.asarray(G), cache, params, terms=term_cfg)
    nonlinear = jnp.zeros_like(linear_total)
    if float(term_cfg.nonlinear) != 0.0:
        real_dtype = jnp.real(jnp.empty((), dtype=linear_total.dtype)).dtype
        nonlinear = nonlinear_em_contribution(
            jnp.asarray(G),
            phi=fields.phi,
            apar=fields.apar,
            bpar=fields.bpar,
            Jl=cache.Jl,
            JlB=cache.JlB,
            tz=jnp.asarray(params.tz),
            vth=jnp.asarray(params.vth),
            sqrt_m=cache.sqrt_m,
            sqrt_m_p1=cache.sqrt_m_p1,
            kx_grid=cache.kx_grid,
            ky_grid=cache.ky_grid,
            dealias_mask=cache.dealias_mask,
            kxfac=cache.kxfac,
            weight=jnp.asarray(term_cfg.nonlinear, dtype=real_dtype),
            apar_weight=float(term_cfg.apar),
            bpar_weight=float(term_cfg.bpar),
            laguerre_to_grid=cache.laguerre_to_grid,
            laguerre_to_spectral=cache.laguerre_to_spectral,
            laguerre_roots=cache.laguerre_roots,
            laguerre_j0=cache.laguerre_j0,
            laguerre_j1_over_alpha=cache.laguerre_j1_over_alpha,
            b=cache.b,
            gx_real_fft=gx_real_fft,
            laguerre_mode=laguerre_mode,
        )
    total = jnp.asarray(linear_total) + jnp.asarray(nonlinear)
    return (
        np.asarray(linear_total),
        np.asarray(nonlinear),
        np.asarray(total),
        fields,
    )


def _compute_stage_states(
    G0: np.ndarray,
    *,
    cache,
    params,
    term_cfg,
    dt: float,
    gx_real_fft: bool,
    laguerre_mode: str,
) -> NonlinearRK4StageStates:
    k1_lin, k1_nl, k1_tot, _fields0 = _split_rhs(
        G0,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        gx_real_fft=gx_real_fft,
        laguerre_mode=laguerre_mode,
    )
    G2 = np.asarray(G0) + 0.5 * float(dt) * k1_tot
    k2_lin, k2_nl, k2_tot, _fields2 = _split_rhs(
        G2,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        gx_real_fft=gx_real_fft,
        laguerre_mode=laguerre_mode,
    )
    G3 = np.asarray(G0) + 0.5 * float(dt) * k2_tot
    k3_lin, k3_nl, k3_tot, _fields3 = _split_rhs(
        G3,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        gx_real_fft=gx_real_fft,
        laguerre_mode=laguerre_mode,
    )
    G4 = np.asarray(G0) + float(dt) * k3_tot
    k4_lin, k4_nl, k4_tot, _fields4 = _split_rhs(
        G4,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        gx_real_fft=gx_real_fft,
        laguerre_mode=laguerre_mode,
    )
    return NonlinearRK4StageStates(
        k1_linear=k1_lin,
        k1_nonlinear=k1_nl,
        k1_total=k1_tot,
        k2_linear=k2_lin,
        k2_nonlinear=k2_nl,
        k2_total=k2_tot,
        k3_linear=k3_lin,
        k3_nonlinear=k3_nl,
        k3_total=k3_tot,
        k4_linear=k4_lin,
        k4_nonlinear=k4_nl,
        k4_total=k4_tot,
        g2=G2,
        g3=G3,
        g4=G4,
    )


def _partial_stage_targets(partial_call: int, stages: NonlinearRK4StageStates) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if partial_call == 1:
        return stages.g2, stages.k2_linear, stages.k2_nonlinear, stages.k2_total
    if partial_call == 2:
        return stages.g3, stages.k3_linear, stages.k3_nonlinear, stages.k3_total
    if partial_call == 3:
        return stages.g4, stages.k4_linear, stages.k4_nonlinear, stages.k4_total
    raise ValueError("partial_call must be 1, 2, or 3")


def main() -> None:
    args = build_parser().parse_args()

    shape = _load_shape(args.gx_dir / "field_shape.txt")
    partial_shape = _load_shape(args.gx_dir / "rk4_partial_shape.txt")
    if partial_shape != shape:
        raise ValueError("field_shape.txt and rk4_partial_shape.txt do not match")
    nspec = shape["nspec"]
    nl = shape["nl"]
    nm = shape["nm"]
    nyc = shape["nyc"]
    nx = shape["nx"]
    nz = shape["nz"]
    gx_shape = (nspec, nl, nm, nyc, nx, nz)

    gx_g0 = _reshape_gx(
        _load_bin(args.gx_dir / "field_g_state.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_k1_linear = _reshape_gx(
        _load_bin(args.gx_dir / "rhs_linear.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_phi0 = _load_field(args.gx_dir / "field_phi.bin", nyc, nx, nz)
    gx_apar0 = _load_field(args.gx_dir / "field_apar.bin", nyc, nx, nz)

    gx_partial_g = _reshape_gx(
        _load_bin(args.gx_dir / "rk4_partial_g_state.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_partial_linear = _reshape_gx(
        _load_bin(args.gx_dir / "rk4_partial_rhs_linear.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_partial_nonlinear = _reshape_gx(
        _load_bin(args.gx_dir / "rk4_partial_rhs_nonlinear.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_partial_total = _reshape_gx(
        _load_bin(args.gx_dir / "rk4_partial_rhs_total.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_partial_phi = _load_field(args.gx_dir / "rk4_partial_phi.bin", nyc, nx, nz)
    gx_partial_apar = _load_field(args.gx_dir / "rk4_partial_apar.bin", nyc, nx, nz)

    cfg, _data = load_runtime_from_toml(args.config)
    ny_full = 2 * (nyc - 1)
    cfg_use = replace(
        cfg,
        grid=replace(cfg.grid, Nx=int(nx), Ny=int(ny_full), Nz=int(nz)),
    )
    geom = build_runtime_geometry(cfg_use)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg_use.grid)
    grid_full = build_spectral_grid(grid_cfg)
    grid = select_gx_real_fft_ky_grid(grid_full, gx_real_fft_ky(grid_full.ky))
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    params = build_runtime_linear_params(cfg_use, Nm=nm, geom=geom)
    term_cfg = build_runtime_term_config(cfg_use)
    cache = build_linear_cache(grid, geom_eff, params, nl, nm)

    dt = float(cfg_use.time.dt if args.dt is None else args.dt)
    stages = _compute_stage_states(
        gx_g0,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        dt=dt,
        gx_real_fft=bool(cfg_use.time.gx_real_fft),
        laguerre_mode=str(cfg_use.time.laguerre_nonlinear_mode),
    )
    sp_phi0 = np.asarray(compute_fields_cached(jnp.asarray(gx_g0), cache, params, terms=term_cfg).phi)
    sp_apar0 = np.asarray(compute_fields_cached(jnp.asarray(gx_g0), cache, params, terms=term_cfg).apar)

    sp_partial_g, sp_partial_linear, sp_partial_nonlinear, sp_partial_total = _partial_stage_targets(
        int(args.partial_call), stages
    )
    sp_partial_fields = compute_fields_cached(jnp.asarray(sp_partial_g), cache, params, terms=term_cfg)
    sp_partial_phi = np.asarray(sp_partial_fields.phi)
    sp_partial_apar = np.asarray(sp_partial_fields.apar)

    print("Startup comparison")
    _summary("g_state", gx_g0.astype(np.complex64), np.asarray(gx_g0, dtype=np.complex64))
    _summary("phi", gx_phi0.astype(np.complex64), sp_phi0.astype(np.complex64))
    _summary("apar", gx_apar0.astype(np.complex64), sp_apar0.astype(np.complex64))
    _summary("k1_linear", gx_k1_linear.astype(np.complex64), stages.k1_linear.astype(np.complex64))

    print(f"Partial call {int(args.partial_call)} comparison")
    _summary("partial_g", gx_partial_g.astype(np.complex64), sp_partial_g.astype(np.complex64))
    _summary("partial_phi", gx_partial_phi.astype(np.complex64), sp_partial_phi.astype(np.complex64))
    _summary("partial_apar", gx_partial_apar.astype(np.complex64), sp_partial_apar.astype(np.complex64))
    _summary("partial_rhs_linear", gx_partial_linear.astype(np.complex64), sp_partial_linear.astype(np.complex64))
    _summary("partial_rhs_nonlinear", gx_partial_nonlinear.astype(np.complex64), sp_partial_nonlinear.astype(np.complex64))
    _summary("partial_rhs_total", gx_partial_total.astype(np.complex64), sp_partial_total.astype(np.complex64))


if __name__ == "__main__":
    main()
