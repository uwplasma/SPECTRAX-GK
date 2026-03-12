#!/usr/bin/env python3
"""Compare a GX linear run against SPECTRAX-GK using imported GX/VMEC geometry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from spectraxgk.io import load_toml
from spectraxgk.linear import LinearTerms, build_linear_cache
from spectraxgk.species import Species, build_linear_params


@dataclass(frozen=True)
class GXInputContract:
    Nx: int
    Ny: int
    nperiod: int
    ntheta: int
    boundary: str
    y0: float
    species: tuple[Species, ...]
    tau_e: float
    beta: float


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


def _load_gx_input_contract(path: Path) -> GXInputContract:
    data = load_toml(path)
    dims = data.get("Dimensions", {})
    domain = data.get("Domain", {})
    physics = data.get("Physics", {})
    species_raw = data.get("species", {})
    boltz = data.get("Boltzmann", {})

    nspecies = int(dims.get("nspecies", 1))

    def _species_array(name: str, *, default: float) -> np.ndarray:
        raw = species_raw.get(name)
        if raw is None:
            return np.full(nspecies, default, dtype=float)
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            arr = np.full(nspecies, float(arr), dtype=float)
        if arr.size < nspecies:
            raise ValueError(f"{path} species.{name} has {arr.size} entries, expected at least {nspecies}")
        return np.asarray(arr[:nspecies], dtype=float)

    charge = _species_array("z", default=1.0)
    mass = _species_array("mass", default=1.0)
    dens = _species_array("dens", default=1.0)
    temp = _species_array("temp", default=1.0)
    tprim = _species_array("tprim", default=0.0)
    fprim = _species_array("fprim", default=0.0)
    nu = _species_array("vnewk", default=0.0)
    species = tuple(
        Species(
            charge=float(charge[i]),
            mass=float(mass[i]),
            density=float(dens[i]),
            temperature=float(temp[i]),
            tprim=float(tprim[i]),
            fprim=float(fprim[i]),
            nu=float(nu[i]),
        )
        for i in range(nspecies)
    )

    add_boltz = bool(boltz.get("add_Boltzmann_species", False))
    boltz_type = str(boltz.get("Boltzmann_type", "")).strip().lower()
    tau_e = float(boltz.get("tau_fac", 0.0)) if add_boltz and boltz_type == "electrons" else 0.0

    return GXInputContract(
        Nx=int(dims.get("nkx", dims.get("nx", 1))),
        Ny=int(dims.get("nky", dims.get("ny", 0))),
        nperiod=int(dims.get("nperiod", 1)),
        ntheta=int(dims.get("ntheta", 0)),
        boundary=str(domain.get("boundary", "linked")),
        y0=float(domain["y0"]) if "y0" in domain else float("nan"),
        species=species,
        tau_e=tau_e,
        beta=float(physics.get("beta", 0.0)),
    )


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


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--gx-input",
        type=Path,
        default=None,
        help="Optional GX input file used to infer boundary/grid defaults for imported geometry cases.",
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    gx_time, gx_ky, gx_omega, gx_Wg, gx_Wphi, gx_Wapar = _load_gx_reference(args.gx)
    positive_ky = gx_ky[gx_ky > 0.0]
    ky_values = positive_ky if args.ky is None or len(args.ky) == 0 else np.asarray(args.ky, dtype=float)
    dt = float(gx_time[0])
    sample_steps = np.rint(gx_time / dt).astype(int) - 1

    geom = load_gx_geometry_netcdf(args.geometry_file)
    boundary = "linked"
    y0 = _infer_y0(gx_ky)
    nx = 1
    ny = _infer_real_fft_ny(gx_ky)
    nperiod = 1
    ntheta = int(np.asarray(geom.theta).size)
    species = [
        Species(
            charge=1.0,
            mass=1.0,
            density=1.0,
            temperature=1.0,
            tprim=float(args.tprim),
            fprim=float(args.fprim),
        )
    ]
    tau_e = float(args.tau_e)
    beta = 0.0
    if args.gx_input is not None:
        gx_contract = _load_gx_input_contract(args.gx_input)
        boundary = str(gx_contract.boundary)
        y0 = float(gx_contract.y0) if np.isfinite(float(gx_contract.y0)) else y0
        nx = max(1, int(gx_contract.Nx))
        ny_input = int(gx_contract.Ny)
        ny = ny_input if ny_input > 0 else ny
        nperiod = max(1, int(gx_contract.nperiod))
        ntheta_in = int(gx_contract.ntheta)
        if ntheta_in > 0:
            ntheta = ntheta_in
        species = list(gx_contract.species)
        tau_e = float(gx_contract.tau_e)
        beta = float(gx_contract.beta)

    grid_cfg = GridConfig(
        Nx=nx,
        Ny=ny,
        Nz=int(np.asarray(geom.theta).size),
        Lx=62.8,
        Ly=2.0 * np.pi * y0,
        boundary=boundary,
        y0=y0,
        nperiod=nperiod,
        ntheta=ntheta,
    )
    grid_full = build_spectral_grid(apply_gx_geometry_grid_defaults(geom, grid_cfg))

    params = build_linear_params(
        species,
        tau_e=tau_e,
        kpar_scale=float(geom.gradpar()),
        beta=beta,
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
