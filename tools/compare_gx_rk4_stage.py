#!/usr/bin/env python3
"""Compare GX linear RK4 stage dumps against SPECTRAX stage reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from tools.compare_gx_rhs_terms import _infer_y0, _load_bin, _load_field, _load_shape, _reshape_gx, _summary
from tools.compare_gx_startup import _build_case_setup, _build_startup_state, _select_ky_block
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import build_linear_cache
from spectraxgk.terms.assembly import assemble_rhs_cached, compute_fields_cached
from spectraxgk.terms.config import TermConfig


@dataclass(frozen=True)
class RK4StageStates:
    k1: np.ndarray
    k2: np.ndarray
    k3: np.ndarray
    k4: np.ndarray
    g1: np.ndarray
    g2: np.ndarray
    g3: np.ndarray
    g_next: np.ndarray


def _term_config(case: str) -> TermConfig:
    if case == "kbm":
        return TermConfig(bpar=0.0)
    return TermConfig()


def _compute_stage_states(
    g0: np.ndarray,
    *,
    cache,
    params,
    term_cfg: TermConfig,
    dt: float,
) -> RK4StageStates:
    g0_j = jnp.asarray(g0)
    dt_j = jnp.asarray(float(dt))

    k1, _fields0 = assemble_rhs_cached(g0_j, cache, params, terms=term_cfg)
    g1 = g0_j + 0.5 * dt_j * k1
    k2, _fields1 = assemble_rhs_cached(g1, cache, params, terms=term_cfg)
    g2 = g0_j + 0.5 * dt_j * k2
    k3, _fields2 = assemble_rhs_cached(g2, cache, params, terms=term_cfg)
    g3 = g0_j + dt_j * k3
    k4, _fields3 = assemble_rhs_cached(g3, cache, params, terms=term_cfg)
    g_next = g0_j + (dt_j / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return RK4StageStates(
        k1=np.asarray(k1),
        k2=np.asarray(k2),
        k3=np.asarray(k3),
        k4=np.asarray(k4),
        g1=np.asarray(g1),
        g2=np.asarray(g2),
        g3=np.asarray(g3),
        g_next=np.asarray(g_next),
    )


def _partial_stage_targets(partial_call: int, stages: RK4StageStates) -> tuple[np.ndarray, np.ndarray]:
    if partial_call == 1:
        return stages.g1, stages.k2
    if partial_call == 2:
        return stages.g2, stages.k3
    if partial_call == 3:
        return stages.g3, stages.k4
    raise ValueError("partial_call must be 1, 2, or 3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory containing GX RK4 dump binaries")
    parser.add_argument(
        "--gx-out",
        type=Path,
        default=None,
        help="Optional GX .out.nc file used only to infer ky metadata and y0 when --y0 is omitted.",
    )
    parser.add_argument("--case", choices=["cyclone", "kbm"], required=True)
    parser.add_argument("--ky", type=float, required=True)
    parser.add_argument("--Ny", type=int, required=True)
    parser.add_argument("--Nz", type=int, required=True)
    parser.add_argument("--Nl", type=int, required=True)
    parser.add_argument("--Nm", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--partial-call", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--ntheta", type=int, default=None)
    parser.add_argument("--nperiod", type=int, default=None)
    parser.add_argument("--y0", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ky_vals = None
    if args.gx_out is not None:
        from netCDF4 import Dataset

        with Dataset(args.gx_out, "r") as root:
            ky_vals = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)

    if args.y0 is not None:
        y0_use = float(args.y0)
    elif ky_vals is not None:
        y0_use = _infer_y0(ky_vals)
    else:
        raise ValueError("Either --y0 or --gx-out is required to infer the ky grid")

    shape_stage0 = _load_shape(args.gx_dir / "rk4_stage0_shape.txt")
    nspec = shape_stage0["nspec"]
    nl = shape_stage0["nl"]
    nm = shape_stage0["nm"]
    nyc = shape_stage0["nyc"]
    nx = shape_stage0["nx"]
    nz = shape_stage0["nz"]
    gx_shape = (nspec, nl, nm, nyc, nx, nz)

    gx_stage0_g = _reshape_gx(
        _load_bin(args.gx_dir / "rk4_stage0_g_state.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_stage0_rhs = _reshape_gx(
        _load_bin(args.gx_dir / "rk4_stage0_rhs_linear.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_stage0_phi = _load_field(args.gx_dir / "rk4_stage0_phi.bin", nyc, nx, nz)

    shape_partial = _load_shape(args.gx_dir / "rk4_partial_shape.txt")
    if shape_partial != shape_stage0:
        raise ValueError("rk4_stage0_shape.txt and rk4_partial_shape.txt do not match")
    gx_partial_g = _reshape_gx(
        _load_bin(args.gx_dir / "rk4_partial_g_state.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_partial_rhs = _reshape_gx(
        _load_bin(args.gx_dir / "rk4_partial_rhs_linear.bin", gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    gx_partial_phi = _load_field(args.gx_dir / "rk4_partial_phi.bin", nyc, nx, nz)

    cfg, geom, params = _build_case_setup(
        case=args.case,
        nx=nx,
        ny=args.Ny,
        nz=args.Nz,
        y0=y0_use,
        ntheta=args.ntheta,
        nperiod=args.nperiod,
        Nm=args.Nm,
    )
    sp_g0, _ky_grid, _geom = _build_startup_state(
        case=args.case,
        cfg=cfg,
        geom=geom,
        Nl=args.Nl,
        Nm=args.Nm,
        ky_target=float(args.ky),
    )

    grid_full = build_spectral_grid(cfg.grid)
    ky_idx = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(args.ky))))
    grid = select_ky_grid(grid_full, int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(args.ky)))))
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    term_cfg = _term_config(args.case)
    stages = _compute_stage_states(sp_g0, cache=cache, params=params, term_cfg=term_cfg, dt=float(args.dt))

    sp_phi0 = np.asarray(compute_fields_cached(jnp.asarray(sp_g0), cache, params, terms=term_cfg).phi)
    sp_g_partial, sp_rhs_partial = _partial_stage_targets(int(args.partial_call), stages)
    sp_phi_partial = np.asarray(
        compute_fields_cached(jnp.asarray(sp_g_partial), cache, params, terms=term_cfg).phi
    )

    gx_stage0_g_slice = _select_ky_block(gx_stage0_g, ky_idx)
    gx_stage0_rhs_slice = _select_ky_block(gx_stage0_rhs, ky_idx)
    gx_stage0_phi_slice = _select_ky_block(gx_stage0_phi, ky_idx)
    gx_partial_g_slice = _select_ky_block(gx_partial_g, ky_idx)
    gx_partial_rhs_slice = _select_ky_block(gx_partial_rhs, ky_idx)
    gx_partial_phi_slice = _select_ky_block(gx_partial_phi, ky_idx)

    print("Stage-0 startup comparison")
    _summary("stage0_g", gx_stage0_g_slice.astype(np.complex64), sp_g0.astype(np.complex64))
    _summary("stage0_phi", gx_stage0_phi_slice.astype(np.complex64), sp_phi0.astype(np.complex64))
    _summary("k1_linear", gx_stage0_rhs_slice.astype(np.complex64), stages.k1.astype(np.complex64))

    print(f"Partial call {int(args.partial_call)} comparison")
    _summary("partial_g", gx_partial_g_slice.astype(np.complex64), sp_g_partial.astype(np.complex64))
    _summary("partial_phi", gx_partial_phi_slice.astype(np.complex64), sp_phi_partial.astype(np.complex64))
    _summary("next_rhs", gx_partial_rhs_slice.astype(np.complex64), sp_rhs_partial.astype(np.complex64))


if __name__ == "__main__":
    main()
