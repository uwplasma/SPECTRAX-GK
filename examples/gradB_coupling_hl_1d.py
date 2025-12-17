#!/usr/bin/env python3
import numpy as np
from spectraxgk._simulation_multispecies import simulation_multispecies
from spectraxgk.plot_multispecies import plot_multispecies


def main():
    # 1D in space, but keep Laguerre + Hermite moments
    Nx, Ny, Nz = 1, 1, 128
    Nl, Nh = 6, 32

    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=0.0, vth=1.0, nu=0.0, Upar=0.0),
        dict(name="e",   q=-1.0, T=1.0, n0=1.0, rho=0.0, vth=1.0, nu=0.0, Upar=0.0),
    ]

    out = simulation_multispecies(
        input_parameters=dict(
            Lz=2*np.pi,
            t_max=20.0,

            enable_streaming=True,
            enable_gradB_parallel=True,   # <-- turns on ∇∥ ln B couplings
            B_eps=0.15,                   # B(z)=1+eps*cos(...)
            B_mode=1,

            enable_nonlinear=False,
            enable_collisions=False,
            enforce_reality=True,

            # excite kz=1
            nx0=0, ny0=0, nz0=1,
            pert_amp=1e-6,

            # perturb only electrons to seed φ
            perturb_species=["e"],

            lambda_D=0.5,
            species=species,
        ),
        Nx=Nx, Ny=Ny, Nz=Nz,
        Nl=Nl, Nh=Nh,
        timesteps=400,
        dt=0.05,
        adaptive_time_step=True,
        save="diagnostics",
        save_every=1,
        progress=True,
        prefer_rich=True,
        diag_config=dict(
            save_phi_k_line=True,
            save_density_k_line=True,
            save_hermite_spectrum=True,
            ky=0, kx=0, kz=Nz//2 + 1,
        ),
    )

    plot_dir = plot_multispecies(out, outdir="plots", prefix="gradB_coupling_1d", show=False)
    print("Wrote plots to:", plot_dir)


if __name__ == "__main__":
    main()
