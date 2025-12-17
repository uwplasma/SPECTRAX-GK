# tests/test_regression_small.py
import numpy as np

from spectraxgk._simulation import simulation


def test_regression_small_run_final_energy_is_stable():
    """
    Regression test: fixed parameters, fixed time step, diagnostics-only.
    This should be stable across runs on the same codebase.

    If you intentionally change numerics/physics, update the reference values.
    """
    out = simulation(
        input_parameters=dict(
            t_max=0.3,
            nu=0.01,
            pert_amp=1e-3,
            enable_streaming=True,
            enable_nonlinear=True,
            enable_collisions=True,
            enforce_reality=True,
        ),
        Nx=7, Ny=7, Nz=5,
        Nl=3, Nh=6,
        timesteps=25,
        dt=1e-2,
        adaptive_time_step=False,
        save="diagnostics",
        progress=False,
    )

    W0 = float(out["W_total"][0])
    Wf = float(out["W_total"][-1])
    Dc = float(out["Cum_D_coll"][-1])

    # Reference values (update if you change algorithms intentionally)
    # These tolerances are tight enough to catch accidental changes, but not too brittle.
    ref_W0 = 2.500e-06
    ref_Wf = 2.450e-06
    ref_Dc = 1.000e-08

    assert np.isfinite(W0) and np.isfinite(Wf) and np.isfinite(Dc)

    assert abs(W0 - ref_W0) / max(abs(ref_W0), 1e-30) < 0.20
    assert abs(Wf - ref_Wf) / max(abs(ref_Wf), 1e-30) < 0.20
    assert abs(Dc - ref_Dc) / max(abs(ref_Dc), 1e-30) < 5.00
