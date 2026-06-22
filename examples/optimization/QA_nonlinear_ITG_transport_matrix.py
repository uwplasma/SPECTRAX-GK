#!/usr/bin/env python
"""Write a broad matched nonlinear ITG transport matrix for a QA candidate.

This example is the broad-claim companion to
``QA_nonlinear_ITG_matched_audit.py``.  A single matched audit can show that
one optimized equilibrium reduces one long-window nonlinear heat-flux trace.
This script writes the larger paper-facing campaign: three surfaces, two
field-line labels, and three ``k_y`` values, each with seed/timestep replicated
post-transient nonlinear windows.

Edit the constants below after solving a strict VMEC-JAX QA baseline and a
transport-aware candidate.  The script writes SPECTRAX-GK input files plus
launch/postprocess shell scripts; it does not run the simulations.
"""

from pathlib import Path
import sys


SCRIPT_NAME = Path(__file__).name
USAGE = f"""\
Usage:
  python examples/optimization/{SCRIPT_NAME}

This script is configured by editing constants near the top of the file. It
writes a matched baseline-vs-candidate nonlinear transport matrix campaign and
the GPU split launch scripts needed to run it on a multi-GPU node.
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

from tools.build_matched_nonlinear_transport_matrix import main as write_matrix  # noqa: E402


# Replace these with the solved-WOUT files from the strict QA baseline and the
# transport-aware candidate.  The paths below match the default output layout of
# the VMEC-JAX-style optimization examples.
BASELINE_VMEC_FILE = SPECTRAX_ROOT / "results/qa_opt/strict_qa_baseline/wout_final.nc"
CANDIDATE_VMEC_FILE = SPECTRAX_ROOT / "results/qa_opt/nonlinear_ITG/wout_final.nc"
BASELINE_LABEL = "strict_qa"
CANDIDATE_LABEL = "low_transport_candidate"

CASE_PREFIX = "qa_low_transport_matrix"
OUTPUT_DIR = Path("results/qa_opt/nonlinear_transport_matrix")
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"

# Broad nonlinear turbulent-flux optimization evidence uses the same sample
# coverage as the current paper-facing gate.
SURFACES = "0.45,0.64,0.78"
ALPHAS = "0.0,pi/4"
KY_VALUES = "0.10,0.30,0.50"

GRID = "n64:64:64:40:40"
HORIZONS = "700,1100,1500"
WINDOW_TMIN = 1100.0
WINDOW_TMAX = 1500.0
DT = 0.05
SEED_VARIANTS = (31, 32)
DT_VARIANTS = (0.04,)
GPU_SPLITS = 2

MIN_RELATIVE_REDUCTION = 0.02
MIN_PASS_FRACTION = 1.0
MIN_MEAN_RELATIVE_REDUCTION = 0.02
MAX_MEAN_REL_SPREAD = 0.15
MAX_COMBINED_SEM_REL = 0.25


argv = [
    "write",
    "--baseline-vmec-file",
    str(BASELINE_VMEC_FILE),
    "--candidate-vmec-file",
    str(CANDIDATE_VMEC_FILE),
    "--baseline-label",
    BASELINE_LABEL,
    "--candidate-label",
    CANDIDATE_LABEL,
    "--case-prefix",
    CASE_PREFIX,
    "--out-dir",
    str(OUTPUT_DIR),
    "--artifact-dir",
    str(ARTIFACT_DIR),
    "--surfaces",
    SURFACES,
    "--alphas",
    ALPHAS,
    "--ky-values",
    KY_VALUES,
    "--grid",
    GRID,
    "--horizons",
    HORIZONS,
    "--dt",
    str(DT),
    "--window-tmin",
    str(WINDOW_TMIN),
    "--window-tmax",
    str(WINDOW_TMAX),
    "--min-relative-reduction",
    str(MIN_RELATIVE_REDUCTION),
    "--min-pass-fraction",
    str(MIN_PASS_FRACTION),
    "--min-mean-relative-reduction",
    str(MIN_MEAN_RELATIVE_REDUCTION),
    "--max-mean-rel-spread",
    str(MAX_MEAN_REL_SPREAD),
    "--max-combined-sem-rel",
    str(MAX_COMBINED_SEM_REL),
    "--gpu-splits",
    str(GPU_SPLITS),
]
for seed in SEED_VARIANTS:
    argv.extend(["--seed-variant", str(seed)])
for dt_variant in DT_VARIANTS:
    argv.extend(["--dt-variant", str(dt_variant)])

raise SystemExit(write_matrix(argv))
