#!/usr/bin/env python3
"""Compare legacy GX cETG restart/state diagnostics against grouped output."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from spectraxgk.cetg import _compute_cetg_diag, _gx_midplane_index, build_cetg_model_params, cetg_fields
from spectraxgk.gx_legacy_output import (
    expand_gx_legacy_positive_ky_state,
    load_gx_legacy_cetg_output,
    load_gx_legacy_cetg_restart,
)
from spectraxgk.grids import build_spectral_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.runtime import build_runtime_geometry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-nc", required=True, type=Path, help="Legacy GX grouped cETG NetCDF file")
    parser.add_argument("--gx-restart", required=True, type=Path, help="Legacy GX cETG restart NetCDF file")
    parser.add_argument("--config", required=True, type=Path, help="SPECTRAX cETG runtime config")
    return parser


def _kz_filter(arr: np.ndarray) -> np.ndarray:
    if arr.shape[-1] <= 1:
        return arr
    kz_frac = np.fft.fftfreq(arr.shape[-1])
    mask = (np.abs(kz_frac) < (1.0 / 3.0)).astype(float)
    return np.fft.ifft(np.fft.fft(arr, axis=-1) * mask, axis=-1)


def _phi2_qflux_from_gx_positive_state(
    state_positive_ky: np.ndarray,
    ky_active: np.ndarray,
    *,
    tau_fac: float,
    pressure: float,
    dealias_kz: bool,
) -> tuple[float, float, np.ndarray]:
    state = np.asarray(state_positive_ky, dtype=np.complex128)
    pos = state[0, :, 0]
    density = pos[0, : ky_active.size]
    temperature = pos[1, : ky_active.size]
    phi = -float(tau_fac) * density
    if dealias_kz:
        phi = _kz_filter(phi)
    nz = phi.shape[-1]
    vol = np.ones((nz,), dtype=float) / float(nz)
    flux = np.ones((nz,), dtype=float) / float(nz)
    fac = np.where(np.arange(ky_active.size) == 0, 1.0, 2.0)[:, None, None]
    phi2 = float(np.sum(0.5 * np.abs(phi) ** 2 * fac * vol[None, None, :]))
    vphi_r = -1j * ky_active[:, None, None] * phi
    qflux = float(np.sum(np.real(np.conj(vphi_r) * temperature) * 2.0 * flux[None, None, :]) * pressure)
    return phi2, qflux, phi


def _load_special_phi(path: Path) -> np.ndarray | None:
    with Dataset(path, "r") as root:
        special = root.groups.get("Special")
        if special is None or "Phi_z" not in special.variables:
            return None
        raw = np.asarray(special.variables["Phi_z"][:], dtype=float)
    if raw.ndim != 4 or raw.shape[-1] != 2:
        raise ValueError(f"Legacy GX Special/Phi_z has unsupported shape {raw.shape}")
    return raw[..., 0] + 1j * raw[..., 1]


def _rms_rel(ref: np.ndarray, test: np.ndarray) -> float:
    ref_arr = np.asarray(ref)
    test_arr = np.asarray(test)
    denom = np.sqrt(np.mean(np.abs(ref_arr) ** 2)) + 1.0e-30
    return float(np.sqrt(np.mean(np.abs(test_arr - ref_arr) ** 2)) / denom)


def main() -> int:
    args = build_parser().parse_args()
    gx = load_gx_legacy_cetg_output(args.gx_nc)
    cfg, _data = load_runtime_from_toml(args.config)

    restart = load_gx_legacy_cetg_restart(
        args.gx_restart,
        nx_full=int(cfg.grid.Nx),
        ny_full=int(cfg.grid.Ny),
    )

    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    params = build_cetg_model_params(cfg, geom, Nl=2, Nm=1)

    phi2_restart_raw, qflux_restart_raw, phi_restart_raw = _phi2_qflux_from_gx_positive_state(
        restart.state_positive_ky,
        gx.ky,
        tau_fac=params.tau_fac,
        pressure=params.pressure,
        dealias_kz=params.dealias_kz,
    )

    state_full = expand_gx_legacy_positive_ky_state(restart.state_positive_ky, ny_full=int(cfg.grid.Ny))
    state_full_jax = jnp.asarray(state_full)
    fields = cetg_fields(state_full_jax, grid, params)
    mask = jnp.broadcast_to(jnp.asarray(grid.dealias_mask, dtype=bool), (grid.ky.size, grid.kx.size))
    diag = _compute_cetg_diag(
        state_full_jax,
        fields,
        fields.phi,
        jnp.asarray(1.0, dtype=jnp.float32),
        grid,
        params,
        mask=mask,
        z_index=_gx_midplane_index(grid.z.size),
        omega_ky_index=None,
        omega_kx_index=None,
    )
    _gamma, _omega, W_s, Phi2_s, _Wapar, q_s, p_s, _qs_s, _ps_s, _phi_mode = diag

    phi_special = _load_special_phi(args.gx_nc)
    phi2_special = None
    qflux_special = None
    phi_special_vs_restart = None
    if phi_special is not None:
        nz = phi_special.shape[-1]
        vol = np.ones((nz,), dtype=float) / float(nz)
        flux = np.ones((nz,), dtype=float) / float(nz)
        fac = np.where(np.arange(gx.ky.size) == 0, 1.0, 2.0)[:, None, None]
        phi2_special = float(np.sum(0.5 * np.abs(phi_special) ** 2 * fac * vol[None, None, :]))
        temp = np.asarray(restart.state_positive_ky[0, 1, 0, : gx.ky.size], dtype=np.complex128)
        vphi_r = -1j * gx.ky[:, None, None] * phi_special
        qflux_special = float(np.sum(np.real(np.conj(vphi_r) * temp) * 2.0 * flux[None, None, :]) * params.pressure)
        phi_special_vs_restart = _rms_rel(phi_special, phi_restart_raw)

    print(f"gx_out_final:     W={float(gx.W[-1]):.9g} Phi2={float(gx.Phi2[-1]):.9g} qflux={float(gx.qflux[-1, 0]):.9g}")
    print(
        "restart_raw:      "
        f"t={restart.time:.9g} Phi2={phi2_restart_raw:.9g} qflux={qflux_restart_raw:.9g}"
    )
    print(
        "spectrax_restart: "
        f"W={float(np.asarray(W_s)):.9g} Phi2={float(np.asarray(Phi2_s)):.9g} "
        f"qflux={float(np.asarray(q_s)):.9g} pflux={float(np.asarray(p_s)):.9g}"
    )
    if phi2_special is not None and qflux_special is not None:
        print(
            "gx_field_dump:    "
            f"Phi2={phi2_special:.9g} qflux={qflux_special:.9g} "
            f"phi_rms_rel_vs_restart={phi_special_vs_restart:.9g}"
        )
    else:
        print("gx_field_dump:    not present in grouped file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
