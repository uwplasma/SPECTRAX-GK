#!/usr/bin/env python
"""Optimize quasi-axisymmetry plus a linear ITG growth-rate objective.

This is VMEC-JAX's current ``QA_optimization.py`` workflow with one additional
SPECTRAX-GK objective tuple. Edit the constants below and run without arguments.
The growth objective is traceable through the equilibrium and eigensolve, so the
implicit-Jacobian optimization path is used.
"""

import dataclasses
import os
from pathlib import Path
import sys

import numpy as np

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(__doc__.strip())
    print(f"\nUsage: python examples/optimization/{Path(__file__).name}")
    raise SystemExit(0)
if len(sys.argv) > 1:
    raise SystemExit(
        f"unexpected arguments: {' '.join(sys.argv[1:])}; edit the constants"
    )

import vmec_jax as vj  # noqa: E402
from vmec_jax import optimize as opt  # noqa: E402
from vmec_jax.core import turbulence as turb  # noqa: E402

# Equilibrium and optimization controls: these match the current upstream QA example.
INPUT_FILE = (
    Path(vj.__file__).resolve().parents[1] / "examples/data/input.minimal_seed_nfp2"
)
OUTPUT_DIR = Path("output_QA_optimization_linear_ITG")
QS_SURFACES = np.linspace(0.1, 1.0, 10)
HELICITY_M, HELICITY_N = 1, 0
ASPECT_TARGET = 6.0
IOTA_TARGET = 0.42
SEED_PERTURBATION = 0.01
MAX_MODE_SCHEDULE = (1, 2, 3, 4, 5)
MAX_NFEV = 2000
FTOL = 1.0e-6
JAC = "implicit"
USE_ESS = True
if os.environ.get("VMEC_JAX_EXAMPLES_CI") == "1":
    MAX_MODE_SCHEDULE, MAX_NFEV, FTOL = (1,), 4, 1.0e-4

# Fixed ITG flux tube and modest Hermite-Laguerre objective resolution.
SURFACE_INDEX = 7
ALPHA = 0.0
NTHETA = 24
SELECTED_KY_INDEX = 1
N_LAGUERRE = 2
N_HERMITE = 3
R_OVER_LT = 6.9
R_OVER_LN = 2.2
TRANSPORT_WEIGHT = 0.01

inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor + 1, 1] += SEED_PERTURBATION
zbs[inp.ntor + 1, 1] += SEED_PERTURBATION
inp = dataclasses.replace(inp, rbc=rbc, zbs=zbs)
eq = opt.solve_equilibrium(inp)
qs = opt.QuasisymmetryRatioResidual(QS_SURFACES, HELICITY_M, HELICITY_N)


def transport_objective(state, runtime):
    """Dominant linear ITG growth rate on the selected flux tube."""

    return turb.turbulent_growth_rate(
        state,
        runtime,
        s_index=SURFACE_INDEX,
        alpha=ALPHA,
        ntheta=NTHETA,
        selected_ky_index=SELECTED_KY_INDEX,
        n_laguerre=N_LAGUERRE,
        n_hermite=N_HERMITE,
        r_over_lt=R_OVER_LT,
        r_over_ln=R_OVER_LN,
    )


def report(tag, equilibrium):
    """Print the solved-equilibrium and transport metrics."""

    gamma = float(transport_objective(equilibrium.state, equilibrium.runtime))
    print(
        f"[{tag}] QS={float(qs.total(equilibrium)):.6e}, "
        f"aspect={float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)):.4f}, "
        f"iota={float(opt.mean_iota(equilibrium.state, equilibrium.runtime)):.4f}, "
        f"gamma={gamma:.6e}"
    )
    return gamma


gamma_seed = report("seed", eq)
objective_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 10.0),
    (transport_objective, 0.0, TRANSPORT_WEIGHT),
]

for max_mode in MAX_MODE_SCHEDULE:
    print(f"\n===== stage max_mode={max_mode} =====")
    result = opt.least_squares(
        objective_terms,
        inp,
        max_mode=max_mode,
        jac=JAC,
        use_ess=USE_ESS,
        verbose=1,
        max_nfev=MAX_NFEV,
        ftol=FTOL,
        xtol=1.0e-10,
    )
    inp = result.input
    if result.equilibrium is not None:
        report(f"stage {max_mode}", result.equilibrium)

eq = result.equilibrium or opt.solve_equilibrium(inp)
gamma_final = report("final", eq)
print(f"linear ITG growth rate: {gamma_seed:.6e} -> {gamma_final:.6e}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
inp.to_indata(OUTPUT_DIR / "input.QA_linear_ITG_optimized")
wout_path = vj.write_wout(OUTPUT_DIR / "wout_QA_linear_ITG_optimized.nc", eq.wout)
for name, path in vj.plot_wout(wout_path, OUTPUT_DIR).items():
    print(f"wrote {name}: {path}")
print(
    "Run QA_nonlinear_ITG_matched_audit.py before claiming nonlinear transport reduction."
)
