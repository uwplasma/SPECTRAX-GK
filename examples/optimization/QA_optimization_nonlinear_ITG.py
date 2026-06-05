#!/usr/bin/env python
"""Quasi-axisymmetric VMEC-JAX optimization with SPECTRAX-GK nonlinear-window heat-flux screening objective.

This script intentionally mirrors VMEC-JAX
``examples/optimization/QA_optimization.py``. The QA/aspect/iota objective
block is unchanged; the only physics addition is the final SPECTRAX-GK
transport tuple in ``objective_tuples``. Edit the constants below directly,
as in the upstream VMEC-JAX example.
"""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

SCRIPT_NAME = Path(__file__).name
USAGE = f"""\
Usage:
  python examples/optimization/{SCRIPT_NAME}

This example intentionally follows vmec_jax/examples/optimization/QA_optimization.py:
edit the constants near the top of the file, then run the script with no
arguments. It appends one SPECTRAX-GK nonlinear-window heat-flux screening
objective tuple to the standard QA/aspect/iota objective list.
"""

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(__doc__.strip())
    print()
    print(USAGE)
    raise SystemExit(0)
if len(sys.argv) > 1:
    unknown = " ".join(sys.argv[1:])
    raise SystemExit(
        f"{SCRIPT_NAME} is configured by editing constants in the file; "
        f"unexpected arguments: {unknown!r}. Use --help for usage."
    )

SPECTRAX_ROOT = Path(__file__).resolve().parents[2]
if str(SPECTRAX_ROOT) not in sys.path:
    sys.path.insert(0, str(SPECTRAX_ROOT))

import vmec_jax as vj  # noqa: E402
from vmec_jax._compat import enable_x64  # noqa: E402

from spectraxgk import (  # noqa: E402
    StellaratorITGSampleSet,
    VMECJAXSpectraxTransportObjective,
    VMECJAXTransportObjectiveConfig,
)


enable_x64(True)

DATA_DIR = Path(vj.__file__).resolve().parents[1] / "examples" / "data"

# Problem parameters. These are intended to be edited directly, as in the
# VMEC-JAX and SIMSOPT examples.
WARM_START_INPUT_FILE = DATA_DIR / "input.nfp2_QA_omnigenity"
SIMPLE_SEED_INPUT_FILE = DATA_DIR / "input.minimal_seed_nfp2"
OUTPUT_DIR = Path("results/qa_opt/spectrax_nonlinear_window")
MAX_MODE = 5
MIN_VMEC_MODE = MAX_MODE + 2
USE_SIMPLE_SEED = True  # Start from near-circular RBC(0,0), RBC(0,1), ZBS(0,1).
SIMPLE_SEED_PERTURBATION = 1.0e-5  # Tiny active modes keep derivatives away from exactly zero.
INPUT_FILE = SIMPLE_SEED_INPUT_FILE if USE_SIMPLE_SEED else WARM_START_INPUT_FILE
INPUT_FILE = vj.prepare_simple_omnigenity_seed_input(
    INPUT_FILE,
    OUTPUT_DIR,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    enabled=USE_SIMPLE_SEED,
    perturbation=SIMPLE_SEED_PERTURBATION,
)
USE_MODE_CONTINUATION = not USE_SIMPLE_SEED
MAX_NFEV = 70
CONTINUATION_NFEV = 25
STAGE_MODES = vj.qs_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
)

# Optimizer parameters.
METHOD = "scalar_trust"  # SPECTRAX transport uses custom VJP; dense scipy exact asks for JVP columns.
SCIPY_TR_SOLVER = "exact"  # For METHOD="scipy": "lsmr" is memory-light; "exact" is dense.
SCIPY_LSMR_MAXITER = None  # For scipy_matrix_free, None uses vmec_jax's bounded cap of 4.
FTOL = 1.0e-5  # Relative cost-reduction tolerance for the outer optimizer.
GTOL = 1.0e-5  # Gradient optimality tolerance for the outer optimizer.
XTOL = 1.0e-6  # Step-size tolerance for the outer optimizer.
# Budget probes on the aspect-5 max_mode=4/5 lane show 120/1e-9 matches the
# conservative 180/1e-9 trajectory while avoiding the too-loose 60/1e-6 branch.
INNER_MAX_ITER = 120  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
INNER_FTOL = 1.0e-9  # Accepted-point VMEC tolerance; 0 uses FTOL from the input deck.
TRIAL_MAX_ITER = 120  # Trial-point VMEC iterations; 0 follows the accepted/input budget.
TRIAL_FTOL = 1.0e-9  # Trial-point VMEC tolerance; 0 follows the accepted/input tolerance.
SOLVER_DEVICE = None  # None uses JAX default; set "cpu" or "gpu" to force one backend.
USE_ESS = True  # Set False for an unscaled trust-region solve.
ALPHA = 1.2  # ESS high-mode scaling strength.
# Common alternatives:
# METHOD = "lbfgs_adjoint"
# METHOD = "scipy"  # Pure VMEC-JAX QA only; transport objectives need scalar-adjoint AD.
# USE_SIMPLE_SEED = False
# USE_MODE_CONTINUATION = False
# STAGE_MODES = [MAX_MODE]
# USE_ESS = False

# Output controls.
SAVE_STAGE_INPUTS = True  # Keep per-stage input decks for continuation/debugging.
SAVE_STAGE_WOUTS = False  # Set True to also write per-stage WOUT files.
MAKE_PLOTS = True

# Post-optimization nonlinear ITG audit controls. The example writes the same
# long-window config manifest used by the production promotion pipeline. Flip
# RUN_LONG_NONLINEAR_AUDIT_COMMANDS to True only on a workstation/GPU node.
WRITE_LONG_NONLINEAR_AUDIT_CONFIGS = True
RUN_LONG_NONLINEAR_AUDIT_COMMANDS = False
NONLINEAR_AUDIT_OUT_DIR = OUTPUT_DIR / "nonlinear_transport_audit_configs"
NONLINEAR_AUDIT_HORIZONS = "700"
NONLINEAR_AUDIT_WINDOW_TMIN = 350.0
NONLINEAR_AUDIT_WINDOW_TMAX = 700.0
NONLINEAR_AUDIT_GRID = "n64:64:64:40:40"
NONLINEAR_AUDIT_KY = 0.47619047619047616
NONLINEAR_AUDIT_DT = 0.05
NONLINEAR_AUDIT_DT_VARIANT = 0.04
NONLINEAR_AUDIT_SEED_VARIANTS = (31, 32)

# Physics targets and least-squares objective weights. These are SIMSOPT-style
# tuple weights, so vmec_jax minimizes sqrt(weight) * (J - target).
TARGET_ASPECT = 5.0
TARGET_IOTA = 0.41
HELICITY_M = 1
HELICITY_N = 0
SURFACES = np.arange(0.0, 1.01, 0.1)
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 10_000.0
QS_WEIGHT = 1.0

# SPECTRAX-GK transport objective. Keep this weight small while tuning so the
# upstream QA/aspect/iota objective remains the dominant solved-equilibrium gate.
# The default sample set matches the 18-point nonlinear-audit prelaunch policy:
# three surfaces, two field-line labels, and three grid-compatible ky values.
SPECTRAX_KIND = "nonlinear_window_heat_flux"
SPECTRAX_WEIGHT = 0.0025
SPECTRAX_OBJECTIVE_TRANSFORM = "log1p"
SPECTRAX_OBJECTIVE_SCALE = 1.0
SPECTRAX_SURFACES = (0.45, 0.64, 0.78)
SPECTRAX_ALPHAS = (0.0, np.pi / 4.0)
SPECTRAX_KY_VALUES = (0.10, 0.30, 0.50)
SPECTRAX_NTHETA = 24
SPECTRAX_MBOZ = 21
SPECTRAX_NBOZ = 21
SPECTRAX_N_LAGUERRE = 2
SPECTRAX_N_HERMITE = 3


# Optimizable VMEC object.
vmec = vj.FixedBoundaryVMEC.from_input(
    INPUT_FILE,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    output_dir=OUTPUT_DIR,
)


# Objective function. Add new terms by appending another
# (objective.J, target, weight) tuple.
aspect = vj.AspectRatio()
iota = vj.MeanIota()
qs = vj.QuasisymmetryRatioResidual(
    helicity_m=HELICITY_M,
    helicity_n=HELICITY_N,
    surfaces=SURFACES,
)
transport_sample_set = StellaratorITGSampleSet(
    surfaces=SPECTRAX_SURFACES,
    alphas=SPECTRAX_ALPHAS,
    ky_values=SPECTRAX_KY_VALUES,
)
transport = VMECJAXSpectraxTransportObjective(
    config=VMECJAXTransportObjectiveConfig(
        kind=SPECTRAX_KIND,
        sample_set=transport_sample_set,
        ntheta=SPECTRAX_NTHETA,
        mboz=SPECTRAX_MBOZ,
        nboz=SPECTRAX_NBOZ,
        n_laguerre=SPECTRAX_N_LAGUERRE,
        n_hermite=SPECTRAX_N_HERMITE,
        objective_transform=SPECTRAX_OBJECTIVE_TRANSFORM,
        objective_scale=SPECTRAX_OBJECTIVE_SCALE,
    ),
    name="spectraxgk_nonlinear_window_heat_flux",
)
objective_tuples = [
    (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
    (iota.J, TARGET_IOTA, IOTA_WEIGHT),
    (qs.J, 0.0, QS_WEIGHT),
    (transport.J, 0.0, SPECTRAX_WEIGHT),
    # Optional:
    # (vj.LgradB(threshold=0.30, smooth_penalty=1.0e-3).J, 0.0, 0.01),
    # (vj.MagneticWell(minimum=0.0).J, 0.0, 1.0),
    # Finite-beta examples can also add:
    # (vj.VolavgB().J, TARGET_VOLAVGB, VOLAVGB_WEIGHT),
    # (vj.BetaTotal().J, TARGET_BETA, BETA_WEIGHT),
    # (vj.DMerc(minimum=0.0, softness=1.0e-3).J, 0.0, DMERC_WEIGHT),
    # (vj.JDotB(surfaces=(0.25, 0.50, 0.75)).J, 0.0, JDOTB_WEIGHT),
    # (vj.BDotB(surfaces=(0.25, 0.50, 0.75)).J, TARGET_BDOTB, BDOTB_WEIGHT),
    # (vj.BDotGradV(surfaces=(0.25, 0.50, 0.75)).J, TARGET_BDOTGRADV, BDOTGRADV_WEIGHT),
    # (vj.ToroidalCurrent(surfaces=(0.25, 0.50, 0.75)).J, TARGET_TORCUR, TORCUR_WEIGHT),
    # (vj.ToroidalCurrentGradient(surfaces=(0.25, 0.50, 0.75)).J, TARGET_TORCUR_PRIME, TORCUR_PRIME_WEIGHT),
    # (vj.RedlBootstrapMismatch(helicity_n=HELICITY_N, ne_coeffs=NE_COEFFS, Te_coeffs=TE_COEFFS, surfaces=(0.25, 0.50, 0.75)).J, 0.0, BOOTSTRAP_WEIGHT),
    # (vj.BVector(s_index=-1).J, TARGET_B_VECTOR, B_VECTOR_WEIGHT),
    # (vj.JVector(surfaces=(0.25, 0.50, 0.75)).J, TARGET_J_VECTOR, J_VECTOR_WEIGHT),
]
problem = vj.LeastSquaresProblem.from_tuples(objective_tuples)

print("\nAssembled least-squares problem:")
print(f"  objectives: {', '.join(problem.objective_names)}")
print(f"  scalar terms: {problem.scalar_objective_names}")
print(f"  SPECTRAX-GK transport kind: {SPECTRAX_KIND} (nonlinear-window heat-flux screening)")
print(f"  SPECTRAX-GK sample set: s={SPECTRAX_SURFACES}, alpha={SPECTRAX_ALPHAS}, ky={SPECTRAX_KY_VALUES}")
print(f"  SPECTRAX-GK tuple weight: {SPECTRAX_WEIGHT:g}")

# Optimization.
# The solve call only receives optimizer, continuation, device, and output
# controls. Physics targets stay in objective_tuples above.
result = vj.least_squares_solve(
    vmec,
    problem,
    stage_modes=STAGE_MODES,
    max_nfev=MAX_NFEV,
    continuation_nfev=CONTINUATION_NFEV,
    method=METHOD,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    use_ess=USE_ESS,
    ess_alpha=ALPHA,
    label=f"QA optimization + SPECTRAX-GK nonlinear-window heat-flux screening (max_mode={MAX_MODE})",
    use_mode_continuation=USE_MODE_CONTINUATION,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    save_stage_inputs=SAVE_STAGE_INPUTS,
    save_stage_wouts=SAVE_STAGE_WOUTS,
    save_final_outputs=False,
)

# Results are plain Python objects. The call below only saves the standard
# artifacts; diagnostics and plots remain explicit in this script.
history = result.history
objective_history = result.objective_history
timing = result.timing_summary
result_summary = result.summary

saved_paths = vj.save_optimization_result(result, output_dir=OUTPUT_DIR)

print("\nFinal diagnostics from result.history:")
print(f"  stages:           {result_summary['stage_modes']}")
print(f"  aspect ratio:     {history['aspect_final']:.6g}")
print(f"  mean iota:        {history['iota_final']:.6g}")
print(f"  QS objective:     {history['qs_final']:.6e}")
print(f"  total objective:  {history['objective_final']:.6e}")
print(f"  wall time:        {timing['total_wall_time_s']:.2f} s")
print(f"  objective samples: {objective_history[:5]} ... {objective_history[-3:]}")

print("\nFiles saved from result objects:")
for name, path in saved_paths.as_dict().items():
    print(f"  {name}: {path}")

wout_final = vj.load_wout(saved_paths.final_wout)
theta, zeta, b_lcfs = vj.vmecplot2_bmag_grid(
    wout_final,
    s_index=-1,
    ntheta=64,
    nzeta=64,
    zeta_max=2.0 * np.pi / float(wout_final.nfp),
)
print("\nLCFS |B| data from vmecplot2_bmag_grid:")
print(f"  theta grid: {theta.shape}, zeta grid: {zeta.shape}, B grid: {b_lcfs.shape}")
print(f"  Bmin/Bmax:  {np.min(b_lcfs):.6g} / {np.max(b_lcfs):.6g}")

if MAKE_PLOTS:
    # Plotting is a normal post-processing block; add or remove entries here
    # instead of relying on hidden plotting side effects from the solve.
    print("\nGenerating initial-vs-final LCFS |B| contour comparison in Boozer coordinates:")
    plot_paths = {
        "boundary_comparison": vj.plot_3d_boundary_comparison(
            saved_paths.initial_wout,
            saved_paths.final_wout,
            outdir=OUTPUT_DIR,
        ),
        "initial_vs_final_lcfs_boozer_bmag_contours": vj.plot_boozer_lcfs_bmag_comparison(
            saved_paths.initial_wout,
            saved_paths.final_wout,
            outdir=OUTPUT_DIR,
        ),
        "objective_history": vj.plot_objective_history(
            saved_paths.history,
            outdir=OUTPUT_DIR,
        ),
    }
    print("\nPlot files selected by this script:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

if WRITE_LONG_NONLINEAR_AUDIT_CONFIGS:
    print("\nWriting long-window initial/final nonlinear ITG audit configs:")
    audit_manifests = {}
    for state_label, wout_path in (
        ("initial", saved_paths.initial_wout),
        ("final", saved_paths.final_wout),
    ):
        audit_out = NONLINEAR_AUDIT_OUT_DIR / state_label
        command = [
            sys.executable,
            str(SPECTRAX_ROOT / "tools" / "write_optimized_equilibrium_transport_configs.py"),
            "--vmec-file",
            str(wout_path),
            "--case",
            f"{OUTPUT_DIR.name}_{state_label}",
            "--out-dir",
            str(audit_out),
            "--horizons",
            NONLINEAR_AUDIT_HORIZONS,
            "--window-tmin",
            f"{NONLINEAR_AUDIT_WINDOW_TMIN:.12g}",
            "--window-tmax",
            f"{NONLINEAR_AUDIT_WINDOW_TMAX:.12g}",
            "--grid",
            NONLINEAR_AUDIT_GRID,
            "--ky",
            f"{NONLINEAR_AUDIT_KY:.16g}",
            "--dt",
            f"{NONLINEAR_AUDIT_DT:.16g}",
            "--dt-variant",
            f"{NONLINEAR_AUDIT_DT_VARIANT:.16g}",
        ]
        for seed in NONLINEAR_AUDIT_SEED_VARIANTS:
            command.extend(["--seed-variant", str(int(seed))])
        subprocess.run(command, cwd=SPECTRAX_ROOT, check=True)
        manifest = audit_out / "run_manifest.json"
        audit_manifests[state_label] = manifest
        print(f"  {state_label}: {manifest}")

    if RUN_LONG_NONLINEAR_AUDIT_COMMANDS:
        print("\nLaunching nonlinear ITG audit commands from generated manifests:")
        audit_ensembles = {}
        for state_label, manifest in audit_manifests.items():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for command in payload.get("launch_commands", []):
                print(f"  [{state_label}] {command}")
                subprocess.run(command, cwd=SPECTRAX_ROOT, shell=True, check=True)
            promotion = payload.get("promotion_contract", {})
            build_ensemble_command = promotion.get("build_ensemble_command")
            if build_ensemble_command:
                print(f"  [{state_label}] {build_ensemble_command}")
                subprocess.run(build_ensemble_command, cwd=SPECTRAX_ROOT, shell=True, check=True)
                audit_ensembles[state_label] = SPECTRAX_ROOT / str(promotion["ensemble_json"])
        if {"initial", "final"}.issubset(audit_ensembles):
            comparison_json = NONLINEAR_AUDIT_OUT_DIR / "initial_final_matched_nonlinear_audit.json"
            comparison_png = NONLINEAR_AUDIT_OUT_DIR / "initial_final_matched_nonlinear_audit.png"
            comparison_command = [
                sys.executable,
                str(SPECTRAX_ROOT / "tools" / "build_matched_nonlinear_transport_comparison.py"),
                "--baseline-ensemble",
                str(audit_ensembles["initial"]),
                "--candidate-ensemble",
                str(audit_ensembles["final"]),
                "--case",
                f"{OUTPUT_DIR.name}_initial_to_final_nonlinear_transport",
                "--min-relative-reduction",
                "0.0",
                "--out-json",
                str(comparison_json),
                "--out-figure",
                str(comparison_png),
            ]
            subprocess.run(comparison_command, cwd=SPECTRAX_ROOT, check=True)
            print(f"\nInitial-vs-final nonlinear audit figure: {comparison_png}")
    else:
        print("\nNonlinear audit configs were written but not launched.")
        print("Set RUN_LONG_NONLINEAR_AUDIT_COMMANDS = True in this script to run them")
        print("or launch the commands listed in each run_manifest.json on a GPU node.")
        print("When launched from this script, it also builds the initial-vs-final Q(t) audit plot.")
