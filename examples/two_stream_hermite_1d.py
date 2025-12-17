#!/usr/bin/env python3
import os
# For higher-accuracy runs:
# os.environ.setdefault("SPECTRAX_X64", "1")
# os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from spectraxgk._simulation_multispecies import simulation_multispecies
from spectraxgk.plot_multispecies import plot_multispecies


def main():
    Nx, Ny, Nz = 1, 1, 128
    Nl, Nh = 1, 64

    species = [
        dict(name="ion",   q=+1.0, T=1.0, n0=1.0, rho=0.0, vth=1.0, nu=1, Upar=0.0),
        dict(name="e+",    q=-1.0, T=1.0, n0=0.5, rho=0.0, vth=1.0, nu=1, Upar=+1.2),
        dict(name="e-",    q=-1.0, T=1.0, n0=0.5, rho=0.0, vth=1.0, nu=1, Upar=-1.2),
    ]

    out = simulation_multispecies(
        input_parameters=dict(
            Lz=2*np.pi,
            t_max=30.0,

            enable_streaming=True,
            enable_gradB_parallel=False,   # Hermite-only (no Laguerre coupling)
            enable_nonlinear=False,
            enable_collisions=False,
            enforce_reality=True,

            # excite kz=1
            nx0=0, ny0=0, nz0=1,
            pert_amp=1e-6,

            # IMPORTANT: do NOT perturb ions -> avoid exact charge cancellation
            perturb_species=["e+", "e-"],

            # IMPORTANT: allow k_perp=0 restoring (otherwise den=0)
            lambda_D=0.5,

            species=species,
        ),
        Nx=Nx, Ny=Ny, Nz=Nz,
        Nl=Nl, Nh=Nh,
        timesteps=600,
        dt=0.05,
        adaptive_time_step=False,
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

    t = np.asarray(out["time"])
    phi = np.asarray(out["phi_rms"])
    print("phi_rms(t0,tend) =", phi[0], phi[-1])

    a = np.log(np.maximum(phi, 1e-300))
    i0, i1 = len(t)//3, 2*len(t)//3
    gamma = np.polyfit(t[i0:i1], a[i0:i1], 1)[0]
    print("Estimated growth rate gamma ~", gamma)

    plot_dir = plot_multispecies(out, outdir="plots", prefix="two_stream_1d", show=False)
    print("Wrote plots to:", plot_dir)


if __name__ == "__main__":
    main()
