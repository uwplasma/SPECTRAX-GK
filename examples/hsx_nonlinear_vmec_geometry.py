#!/usr/bin/env python3
"""Run a nonlinear ITG case on HSX using a VMEC ``wout`` file."""

from __future__ import annotations

import argparse

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.runtime import run_runtime_nonlinear
from spectraxgk.runtime_config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)


def build_hsx_nonlinear_cfg(
    vmec_file: str,
    *,
    geometry_file: str | None,
    gx_repo: str | None,
    torflux: float,
    alpha: float,
    npol: float,
    dt: float,
    t_max: float,
) -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(
            Nx=96,
            Ny=96,
            Nz=48,
            Lx=62.8,
            Ly=62.8,
            boundary="fix aspect",
            y0=21.0,
            ntheta=48,
            nperiod=1,
        ),
        time=TimeConfig(
            t_max=t_max,
            dt=dt,
            method="rk3",
            use_diffrax=False,
            fixed_dt=False,
            sample_stride=50,
            diagnostics_stride=50,
            cfl=1.0,
        ),
        geometry=GeometryConfig(
            model="vmec",
            vmec_file=vmec_file,
            geometry_file=geometry_file,
            gx_repo=gx_repo,
            torflux=torflux,
            alpha=alpha,
            npol=npol,
        ),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-3,
            gaussian_init=False,
            init_single=False,
        ),
        species=(
            RuntimeSpeciesConfig(
                name="ion",
                charge=1.0,
                mass=1.0,
                density=1.0,
                temperature=1.0,
                tprim=3.0,
                fprim=1.0,
                nu=0.01,
            ),
        ),
        physics=RuntimePhysicsConfig(
            linear=False,
            nonlinear=True,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            use_apar=False,
            use_bpar=False,
            beta=0.0,
            collisions=False,
            hypercollisions=True,
        ),
        collisions=RuntimeCollisionConfig(
            damp_ends_amp=0.1,
            damp_ends_widthfrac=1.0 / 8.0,
            D_hyper=0.05,
        ),
        normalization=RuntimeNormalizationConfig(contract="kinetic", diagnostic_norm="gx"),
        terms=RuntimeTermsConfig(
            apar=0.0,
            bpar=0.0,
            end_damping=1.0,
            hypercollisions=1.0,
            hyperdiffusion=1.0,
            nonlinear=1.0,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the HSX nonlinear ITG VMEC example.")
    parser.add_argument("--vmec-file", required=True, help="Path to the VMEC wout file")
    parser.add_argument("--geometry-file", default=None, help="Optional output/reuse path for the generated *.eik.nc file")
    parser.add_argument("--gx-repo", default=None, help="Optional GX repository path for gx_geo_vmec.py")
    parser.add_argument("--torflux", type=float, default=0.64, help="Normalized toroidal flux surface label")
    parser.add_argument("--alpha", type=float, default=0.0, help="Field-line alpha label")
    parser.add_argument("--npol", type=float, default=1.0, help="Number of poloidal turns")
    parser.add_argument("--ky", type=float, default=1.0 / 21.0, help="Target ky mode for diagnostics")
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1, help="Maximum time step")
    parser.add_argument("--t-max", type=float, default=200.0, help="Final time")
    parser.add_argument("--steps", type=int, default=None, help="Optional explicit step-count override")
    args = parser.parse_args()

    cfg = build_hsx_nonlinear_cfg(
        args.vmec_file,
        geometry_file=args.geometry_file,
        gx_repo=args.gx_repo,
        torflux=float(args.torflux),
        alpha=float(args.alpha),
        npol=float(args.npol),
        dt=float(args.dt),
        t_max=float(args.t_max),
    )
    steps = int(args.steps) if args.steps is not None else int(round(float(args.t_max) / float(args.dt)))
    result = run_runtime_nonlinear(
        cfg,
        ky_target=float(args.ky),
        Nl=int(args.Nl),
        Nm=int(args.Nm),
        dt=float(args.dt),
        steps=steps,
    )
    if result.diagnostics is None or result.ky_selected is None:
        raise RuntimeError("Nonlinear runtime did not produce GX diagnostics")
    print(
        "ky={:.6f} Wg={:.8e} Wphi={:.8e} heat={:.8e} pflux={:.8e}".format(
            float(result.ky_selected),
            float(result.diagnostics.Wg_t[-1]),
            float(result.diagnostics.Wphi_t[-1]),
            float(result.diagnostics.heat_flux_t[-1]),
            float(result.diagnostics.particle_flux_t[-1]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
