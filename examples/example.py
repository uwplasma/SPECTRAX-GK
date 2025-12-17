# examples/example.py
from spectraxgk import simulation, plot

out = simulation(
    input_parameters={
        "t_max": 150.0,
        "nu": 0.01,
        "pert_amp": 1e-3,
        # toggles you can flip for physics tests:
        "enable_streaming": True,
        "enable_nonlinear": True,
        "enable_collisions": True,
        "enforce_reality": True,
    },
    Nx=15, Ny=15, Nz=5,
    Nh=12, Nl=4,
    timesteps=200,
    dt=1e-2,
    adaptive_time_step=False,   # for reproducibility / debugging
    save="diagnostics",
    save_every=1,
    probe=dict(
        ky=15 // 2,
        kx=15 // 2 + 1,
        kz=5 // 2,
        lmax=4,
        mmax=12,
    ),
)

plot(out)
