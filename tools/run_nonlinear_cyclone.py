#!/usr/bin/env python3
"""Run a nonlinear Cyclone case with GX-style diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from spectraxgk.benchmarks import CYCLONE_NORMALIZATION
from spectraxgk.config import CycloneBaseCase, GeometryConfig, GridConfig, InitializationConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.nonlinear import integrate_nonlinear_gx_diagnostics
from spectraxgk.species import Species, build_linear_params
from spectraxgk.terms.config import TermConfig


def _build_initial_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    amp: float,
) -> np.ndarray:
    noise = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return (amp * noise).astype(np.complex64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--ntheta", type=int, default=24)
    parser.add_argument("--nperiod", type=int, default=1)
    parser.add_argument("--y0", type=float, default=28.2)
    parser.add_argument("--Lx", type=float, default=2.0 * np.pi * 28.2)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--Nl", type=int, default=8)
    parser.add_argument("--Nm", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", type=float, default=1.0e-3)
    parser.add_argument("--tprim", type=float, default=2.49)
    parser.add_argument("--fprim", type=float, default=0.8)
    parser.add_argument("--rho-star", type=float, default=CYCLONE_NORMALIZATION.rho_star)
    parser.add_argument("--method", type=str, default="rk3", choices=("euler", "rk2", "rk3", "rk4"))
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--use-dealias-mask", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("docs/_static/nonlinear_cyclone_diag.csv"))
    args = parser.parse_args()

    grid_cfg = GridConfig(
        Nx=args.nx,
        Ny=args.ny,
        Nz=args.ntheta * args.nperiod,
        Lx=args.Lx,
        Ly=args.Lx,
        boundary="linked",
        y0=args.y0,
        ntheta=args.ntheta,
        nperiod=args.nperiod,
    )
    grid = build_spectral_grid(grid_cfg)

    geom_cfg = GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778)
    geom = SAlphaGeometry.from_config(geom_cfg)

    ion = Species(
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        tprim=float(args.tprim),
        fprim=float(args.fprim),
        nu=0.0,
    )

    params = build_linear_params(
        [ion],
        tau_e=1.0,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=CYCLONE_NORMALIZATION.omega_d_scale,
        omega_star_scale=CYCLONE_NORMALIZATION.omega_star_scale,
        rho_star=float(args.rho_star),
        beta=0.0,
        fapar=0.0,
        nu_hyper=0.0,
        p_hyper=4.0,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.0,
    )

    init_cfg = InitializationConfig(
        gaussian_init=True,
        gaussian_width=0.5,
        init_field="density",
    )
    rng = np.random.default_rng(args.seed)
    G0 = np.zeros((1, args.Nl, args.Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0[:, 0, 0, ...] = _build_initial_noise(rng, G0[:, 0, 0, ...].shape, args.amp)
    G0 = jnp.asarray(G0)

    term_cfg = TermConfig(nonlinear=1.0, apar=0.0, bpar=0.0)

    t, diag = integrate_nonlinear_gx_diagnostics(
        G0,
        grid,
        geom,
        params,
        dt=float(args.dt),
        steps=int(args.steps),
        method=str(args.method),
        terms=term_cfg,
        sample_stride=int(args.sample_stride),
        use_dealias_mask=bool(args.use_dealias_mask),
    )

    gamma_t = np.asarray(diag.gamma_t)
    omega_t = np.asarray(diag.omega_t)
    if gamma_t.ndim > 1:
        axes = tuple(range(1, gamma_t.ndim))
        gamma_t = np.mean(gamma_t, axis=axes)
        omega_t = np.mean(omega_t, axis=axes)

    data = np.column_stack(
        [
            np.asarray(t),
            gamma_t,
            omega_t,
            np.asarray(diag.Wg_t),
            np.asarray(diag.Wphi_t),
            np.asarray(diag.Wapar_t),
            np.asarray(diag.energy_t),
            np.asarray(diag.heat_flux_t),
            np.asarray(diag.particle_flux_t),
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.out,
        data,
        delimiter=",",
        header="t,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux",
        comments="",
    )
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
