#!/usr/bin/env python3
"""Compare GX RHS term dumps against SPECTRAX term contributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    CycloneBaseCase,
    _apply_gx_hypercollisions,
    _build_initial_condition,
)
from spectraxgk.config import GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache, linear_terms_to_term_config
from spectraxgk.terms.assembly import assemble_rhs_terms_cached
from spectraxgk.terms.config import TermConfig


def _load_shape(path: Path) -> dict[str, int]:
    data: dict[str, int] = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            data[parts[0]] = int(parts[1])
    return data


def _load_bin(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.complex64)
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"{path} size {raw.size} does not match expected {shape}")
    return raw.reshape(shape)


def _summary(label: str, ref: np.ndarray, test: np.ndarray) -> None:
    diff = test - ref
    rel = np.where(np.abs(ref) > 0, diff / ref, np.nan)
    print(f"{label:12s} max|diff|={np.nanmax(np.abs(diff)):.3e} max|rel|={np.nanmax(np.abs(rel)):.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory with rhs_stream.bin, rhs_linear.bin")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file to map ky indices")
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nl", type=int, default=48)
    parser.add_argument("--Nm", type=int, default=16)
    parser.add_argument("--Ny", type=int, default=24)
    parser.add_argument("--Nz", type=int, default=96)
    parser.add_argument("--y0", type=float, default=20.0)
    args = parser.parse_args()

    shape_path = args.gx_dir / "rhs_terms_shape.txt"
    stream_path = args.gx_dir / "rhs_stream.bin"
    linear_path = args.gx_dir / "rhs_linear.bin"
    if not shape_path.exists() or not stream_path.exists() or not linear_path.exists():
        raise FileNotFoundError("Missing GX rhs dump files")

    shape = _load_shape(shape_path)
    nspec = shape.get("nspec", 1)
    nl = shape.get("nl", args.Nl)
    nm = shape.get("nm", args.Nm)
    nyc = shape.get("nyc", args.Ny // 2 + 1)
    nx = shape.get("nx", 1)
    nz = shape.get("nz", args.Nz)
    gx_shape = (nspec, nl, nm, nyc, nx, nz)

    gx_stream = _load_bin(stream_path, gx_shape)
    gx_linear = _load_bin(linear_path, gx_shape)

    with Dataset(args.gx_out, "r") as root:
        ky_vals = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)
    ky_idx = int(np.argmin(np.abs(ky_vals - float(args.ky))))
    gx_stream = gx_stream[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_linear = gx_linear[:, :, :, ky_idx : ky_idx + 1, :, :]

    cfg = CycloneBaseCase(
        grid=GridConfig(
            Nx=1,
            Ny=args.Ny,
            Nz=args.Nz,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=args.y0,
            ntheta=32,
            nperiod=2,
        )
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(args.ky))))
    grid = select_ky_grid(grid_full, ky_index)
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        omega_d_scale=CYCLONE_OMEGA_D_SCALE,
        omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
        rho_star=CYCLONE_RHO_STAR,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    params = _apply_gx_hypercollisions(params, nhermite=args.Nm)
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    G0 = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=args.Nl,
        Nm=args.Nm,
        init_cfg=cfg.init,
    )
    term_cfg = TermConfig(hypercollisions=0.0, end_damping=0.0)
    _rhs_total, _fields, contrib = assemble_rhs_terms_cached(G0, cache, params, terms=term_cfg)
    spectrax_stream = np.asarray(contrib["streaming"])[None, ...]
    spectrax_linear = (
        np.asarray(contrib["mirror"])
        + np.asarray(contrib["curvature"])
        + np.asarray(contrib["gradb"])
        + np.asarray(contrib["diamagnetic"])
        + np.asarray(contrib["collisions"])
    )
    spectrax_linear = spectrax_linear[None, ...]

    _summary("streaming", gx_stream, spectrax_stream)
    _summary("linear", gx_linear, spectrax_linear)


if __name__ == "__main__":
    main()
