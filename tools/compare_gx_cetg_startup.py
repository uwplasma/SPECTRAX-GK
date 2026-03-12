#!/usr/bin/env python3
"""Compare legacy GX cETG startup state and raw field dump against SPECTRAX."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from compare_gx_rhs_terms import _summary
from spectraxgk.cetg import build_cetg_model_params, cetg_fields
from spectraxgk.geometry import apply_gx_geometry_grid_defaults
from spectraxgk.gx_legacy_output import GXLegacyCetgRestart, load_gx_legacy_cetg_restart
from spectraxgk.grids import build_spectral_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.runtime import _build_initial_condition, build_runtime_geometry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-nc", required=True, type=Path, help="Legacy GX cETG startup NetCDF file")
    parser.add_argument("--gx-restart", required=True, type=Path, help="Legacy GX cETG restart file")
    parser.add_argument("--config", required=True, type=Path, help="SPECTRAX cETG runtime config")
    return parser


def _runtime_initial_state(config_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cfg, _data = load_runtime_from_toml(config_path)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=1,
            kx_index=0,
            Nl=2,
            Nm=1,
            nspecies=1,
        ),
        dtype=np.complex64,
    )
    params = build_cetg_model_params(cfg, geom, Nl=2, Nm=1)
    phi0 = np.asarray(cetg_fields(jnp.asarray(g0), grid, params, apply_kz_dealias=False).phi, dtype=np.complex64)
    return g0, phi0


def _active_state_from_full_grid(full_state: np.ndarray, restart: GXLegacyCetgRestart) -> np.ndarray:
    state = np.asarray(full_state, dtype=np.complex64)
    nx_full = int(state.shape[-2])
    ny_active = int(restart.naky_active)
    nx_active = int(restart.nakx_active)
    active = np.zeros((1, 2, 1, ny_active, nx_active, state.shape[-1]), dtype=np.complex64)
    nx_pos = 1 + (nx_full - 1) // 3
    active[:, :, :, :, :nx_pos, :] = state[:, :, :, :ny_active, :nx_pos, :]
    for i in range(2 * nx_full // 3 + 1, nx_full):
        it = i - 2 * nx_full // 3 + ((nx_full - 1) // 3)
        active[:, :, :, :, it, :] = state[:, :, :, :ny_active, i, :]
    return active


def _match_output_kx_indices(kx_full: np.ndarray, kx_out: np.ndarray) -> np.ndarray:
    full = np.asarray(kx_full, dtype=float)
    out = np.asarray(kx_out, dtype=float)
    idx = [int(np.argmin(np.abs(full - val))) for val in out]
    return np.asarray(idx, dtype=np.int32)


def _load_startup_phi(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Dataset(path, "r") as root:
        kx = np.asarray(root.variables["kx"][:], dtype=float)
        raw = np.asarray(root.groups["Special"].variables["Phi_z"][:], dtype=float)
    if raw.ndim != 4 or raw.shape[-1] != 2:
        raise ValueError(f"Legacy GX Special/Phi_z has unsupported shape {raw.shape}")
    return kx, raw[..., 0] + 1j * raw[..., 1]


def main() -> None:
    args = build_parser().parse_args()

    cfg, _data = load_runtime_from_toml(args.config)
    gx_restart = load_gx_legacy_cetg_restart(
        args.gx_restart,
        nx_full=int(cfg.grid.Nx),
        ny_full=int(cfg.grid.Ny),
    )
    sp_g0, sp_phi0 = _runtime_initial_state(args.config)

    gx_kx, gx_phi = _load_startup_phi(args.gx_nc)
    sp_active = _active_state_from_full_grid(sp_g0, gx_restart)
    _summary("g_state", gx_restart.state_active.astype(np.complex64), sp_active)

    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(apply_gx_geometry_grid_defaults(geom, cfg.grid))
    kx_idx = _match_output_kx_indices(np.asarray(grid.kx, dtype=float), gx_kx)
    sp_phi_sorted = sp_phi0[: gx_phi.shape[0], kx_idx, :]
    _summary("phi", gx_phi.astype(np.complex64), sp_phi_sorted)


if __name__ == "__main__":
    main()
