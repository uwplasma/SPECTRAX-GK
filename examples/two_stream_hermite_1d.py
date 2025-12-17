#!/usr/bin/env python3
import os
# Uncomment for higher-accuracy runs (must be set before importing jax/spectraxgk):
# os.environ.setdefault("SPECTRAX_X64", "1")
# os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import jax
import jax.numpy as jnp

from spectraxgk._simulation_multispecies import simulation_multispecies

def main():
    # 1D spatial: only z Fourier; set Nx=Ny=1 so kx=ky=0 and Nl=1 (no mu/Laguerre needed)
    Nx, Ny, Nz = 1, 1, 128
    Nl, Nh = 1, 64

    # Two-stream: two kinetic electron beams drifting ±U, plus an ion background.
    # Normalizations here are dimensionless; choose U ~ O(1) to see growth.
    species = [
        dict(name="ion",   q=+1.0, T=1.0, n0=1.0, rho=0.0, vth=1.0, nu=0.0, Upar=0.0),
        dict(name="e+",    q=-1.0, T=1.0, n0=0.5, rho=0.0, vth=1.0, nu=0.0, Upar=+1.2),
        dict(name="e-",    q=-1.0, T=1.0, n0=0.5, rho=0.0, vth=1.0, nu=0.0, Upar=-1.2),
    ]

    out = simulation_multispecies(
        input_parameters=dict(
            # 1D domain
            Lz=2*np.pi,
            t_max=30.0,

            # physics toggles
            enable_streaming=True,
            enable_nonlinear=False,   # 1D two-stream is electrostatic/parallel; ExB not relevant with kx=ky=0
            enable_collisions=False,
            enforce_reality=True,

            # excite kz=1 density perturbation in ALL species (small)
            nz0=1,
            pert_amp=1e-6,

            # species list
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
    )

    t = np.asarray(out["time"])
    phi = np.asarray(out["phi_rms"])
    print("phi_rms(t0,tend) =", phi[0], phi[-1])

    # crude growth rate estimate from middle window
    a = np.log(np.maximum(phi, 1e-300))
    i0, i1 = len(t)//3, 2*len(t)//3
    gamma = np.polyfit(t[i0:i1], a[i0:i1], 1)[0]
    print("Estimated growth rate gamma ~", gamma)

if __name__ == "__main__":
    main()
