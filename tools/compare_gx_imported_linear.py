#!/usr/bin/env python3
"""Compare a GX linear run against SPECTRAX-GK using imported GX/VMEC geometry."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from spectraxgk.benchmarks import _apply_gx_hypercollisions, _build_initial_condition
from spectraxgk.config import GridConfig, InitializationConfig
from spectraxgk.geometry import apply_gx_geometry_grid_defaults, load_gx_geometry_netcdf
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.analysis import select_ky_index
from spectraxgk.gx_integrators import GXTimeConfig, integrate_linear_gx_diagnostics
from spectraxgk.linear import LinearTerms, build_linear_cache
from spectraxgk.species import Species, build_linear_params


def _load_gx_reference(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Dataset(path, "r")
    try:
        grids = root.groups["Grids"]
        diag = root.groups["Diagnostics"]
        time = np.asarray(grids.variables["time"][:], dtype=float)
        ky = np.asarray(grids.variables["ky"][:], dtype=float)
        omega = np.asarray(diag.variables["omega_kxkyt"][:], dtype=float)
        Wg = np.asarray(diag.variables["Wg_kyst"][:, 0, :], dtype=float)
        Wphi = np.asarray(diag.variables["Wphi_kyst"][:, 0, :], dtype=float)
        Wapar = np.asarray(diag.variables["Wapar_kyst"][:, 0, :], dtype=float)
    finally:
        root.close()
    return time, ky, omega, Wg, Wphi, Wapar


def _infer_real_fft_ny(ky: np.ndarray) -> int:
    positive = ky[ky > 0.0]
    if positive.size == 0:
        raise ValueError("GX reference does not contain positive ky modes")
    return int(3 * (positive.size - 1) + 1)


def _infer_y0(ky: np.ndarray) -> float:
    positive = ky[ky > 0.0]
    if positive.size == 0:
        raise ValueError("GX reference does not contain positive ky modes")
    return float(1.0 / np.min(positive))


def _mean_rel_error(lhs: np.ndarray, rhs: np.ndarray, *, floor_fraction: float) -> float:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    scale = max(float(np.nanmax(np.abs(rhs))), 1.0e-12)
    floor = floor_fraction * scale
    denom = np.maximum(np.abs(rhs), floor)
    return float(np.mean(np.abs(lhs - rhs) / denom))


def _run_single_ky(
    *,
    ky_target: float,
    geom,
    grid_full,
    params,
    time_cfg: GXTimeConfig,
    init_cfg: InitializationConfig,
    Nl: int,
    Nm: int,
    sample_steps: np.ndarray,
    mode_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    G0 = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=init_cfg,
    )
    _t, _phi_t, gamma_t, omega_t, diag = integrate_linear_gx_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=LinearTerms(),
        mode_method=mode_method,
        jit=True,
    )
    gamma = np.asarray(gamma_t)[sample_steps, 0, 0]
    omega = np.asarray(omega_t)[sample_steps, 0, 0]
    Wg = np.asarray(diag.Wg_t)[sample_steps]
    Wphi = np.asarray(diag.Wphi_t)[sample_steps]
    Wapar = np.asarray(diag.Wapar_t)[sample_steps]
    return gamma, omega, Wg, Wphi, Wapar


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GX linear diagnostics against SPECTRAX-GK using imported GX/VMEC geometry."
    )
    parser.add_argument("--gx", type=Path, required=True, help="Path to the GX .out.nc file")
    parser.add_argument(
        "--geometry-file",
        type=Path,
        required=True,
        help="Path to the GX/VMEC geometry file (for example *.eik.nc)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--ky", type=float, nargs="*", default=None, help="Specific ky values to compare")
    parser.add_argument("--Nl", type=int, default=8)
    parser.add_argument("--Nm", type=int, default=16)
    parser.add_argument("--tprim", type=float, default=3.0)
    parser.add_argument("--fprim", type=float, default=1.0)
    parser.add_argument("--tau-e", type=float, default=1.0, dest="tau_e")
    parser.add_argument("--damp-ends-amp", type=float, default=0.1)
    parser.add_argument("--damp-ends-widthfrac", type=float, default=1.0 / 8.0)
    parser.add_argument("--mode-method", choices=("z_index", "max"), default="z_index")
    parser.add_argument(
        "--rel-floor-fraction",
        type=float,
        default=1.0e-2,
        help="Relative-error floor as a fraction of the peak reference magnitude for each series",
    )
    args = parser.parse_args()

    gx_time, gx_ky, gx_omega, gx_Wg, gx_Wphi, gx_Wapar = _load_gx_reference(args.gx)
    positive_ky = gx_ky[gx_ky > 0.0]
    ky_values = positive_ky if args.ky is None or len(args.ky) == 0 else np.asarray(args.ky, dtype=float)
    dt = float(gx_time[0])
    sample_steps = np.rint(gx_time / dt).astype(int) - 1

    geom = load_gx_geometry_netcdf(args.geometry_file)
    grid_cfg = GridConfig(
        Nx=1,
        Ny=_infer_real_fft_ny(gx_ky),
        Nz=int(np.asarray(geom.theta).size),
        Lx=62.8,
        Ly=62.8,
        boundary="linked",
        y0=_infer_y0(gx_ky),
    )
    grid_full = build_spectral_grid(apply_gx_geometry_grid_defaults(geom, grid_cfg))

    params = build_linear_params(
        [
            Species(
                charge=1.0,
                mass=1.0,
                density=1.0,
                temperature=1.0,
                tprim=float(args.tprim),
                fprim=float(args.fprim),
            )
        ],
        tau_e=float(args.tau_e),
        kpar_scale=float(geom.gradpar()),
    )
    params = _apply_gx_hypercollisions(params, nhermite=args.Nm)
    params = replace(
        params,
        damp_ends_amp=float(args.damp_ends_amp) / dt,
        damp_ends_widthfrac=float(args.damp_ends_widthfrac),
    )
    init_cfg = InitializationConfig(
        gaussian_init=True,
        init_field="density",
        init_amp=1.0e-10,
    )
    time_cfg = GXTimeConfig(
        dt=dt,
        t_max=float(gx_time[-1]),
        sample_stride=1,
        fixed_dt=True,
    )

    rows: list[dict[str, float]] = []
    for ky_target in ky_values:
        ky_idx = int(np.argmin(np.abs(gx_ky - float(ky_target))))
        gamma, omega, Wg, Wphi, Wapar = _run_single_ky(
            ky_target=float(ky_target),
            geom=geom,
            grid_full=grid_full,
            params=params,
            time_cfg=time_cfg,
            init_cfg=init_cfg,
            Nl=args.Nl,
            Nm=args.Nm,
            sample_steps=sample_steps,
            mode_method=args.mode_method,
        )
        omega_ref = gx_omega[:, ky_idx, 0, 0]
        gamma_ref = gx_omega[:, ky_idx, 0, 1]
        rows.append(
            {
                "ky": float(ky_target),
                "mean_abs_omega": float(np.mean(np.abs(omega - omega_ref))),
                "mean_rel_omega": _mean_rel_error(
                    omega, omega_ref, floor_fraction=float(args.rel_floor_fraction)
                ),
                "mean_abs_gamma": float(np.mean(np.abs(gamma - gamma_ref))),
                "mean_rel_gamma": _mean_rel_error(
                    gamma, gamma_ref, floor_fraction=float(args.rel_floor_fraction)
                ),
                "mean_rel_Wg": _mean_rel_error(Wg, gx_Wg[:, ky_idx], floor_fraction=1.0e-6),
                "mean_rel_Wphi": _mean_rel_error(Wphi, gx_Wphi[:, ky_idx], floor_fraction=1.0e-6),
                "mean_rel_Wapar": _mean_rel_error(Wapar, gx_Wapar[:, ky_idx], floor_fraction=1.0e-6),
                "omega_last": float(omega[-1]),
                "omega_ref_last": float(omega_ref[-1]),
                "gamma_last": float(gamma[-1]),
                "gamma_ref_last": float(gamma_ref[-1]),
            }
        )

    df = pd.DataFrame(rows).sort_values("ky").reset_index(drop=True)
    print(df.to_string(index=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
