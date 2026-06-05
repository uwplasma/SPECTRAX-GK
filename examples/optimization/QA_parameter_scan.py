#!/usr/bin/env python
"""VMEC-JAX QA boundary-parameter scan with SPECTRAX-GK ITG objectives.

This example scans one VMEC boundary coefficient, here ``RBC(1,1)``, and plots
linear growth, quasilinear heat-flux, and reduced nonlinear-window diagnostics.
Optional replicated long-window nonlinear heat-flux diagnostics can be overlaid
when ensemble sidecars are available. It is intentionally configured by editing
constants below, matching the style of VMEC-JAX example scripts instead of using
a command-line driver wrapper.
"""

from pathlib import Path
import subprocess
import sys

SCRIPT_NAME = Path(__file__).name
USAGE = f"""\
Usage:
  python examples/optimization/{SCRIPT_NAME}

Edit the constants near the top of the file to change the scanned coefficient,
scan fractions, reduced objective sample set, or nonlinear ensemble overlays.
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

# Scan controls. These defaults reproduce the tracked paper-facing RBC(1,1)
# diagnostic from the strict QA baseline. Set EVALUATE_REDUCED = True to rerun
# the deterministic reduced metrics instead of reusing the tracked audit JSON.
BASELINE_INPUT = (
    SPECTRAX_ROOT
    / "tools_out/vmec_jax_qa_full_sweep_20260605/runs/qa_baseline_scipy/input.final"
)
COEFFICIENT = "RBC(1,1)"
FRACTIONS = "-0.50,-0.45,-0.40,-0.35,-0.30,-0.25,-0.20,-0.15,-0.10,-0.05,0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50"
EVALUATE_REDUCED = False
REUSE_REDUCED_JSON = SPECTRAX_ROOT / "docs/_static/vmec_boundary_transport_landscape_rbc11.json"
OUT_PREFIX = SPECTRAX_ROOT / "results/qa_opt/parameter_scan/qa_parameter_scan_rbc11"

# SPECTRAX-GK reduced objective settings. The tracked paper-facing landscape
# uses one representative sample so the coefficient response can be regenerated
# quickly before selecting candidates for expensive nonlinear audits.
SURFACES = "0.64"
ALPHAS = "0.0"
KY_VALUES = "0.30"
NTHETA = 16
MBOZ = 21
NBOZ = 21
N_LAGUERRE = 1
N_HERMITE = 2
SOLVER_DEVICE = None  # Set "cpu" or "gpu" to force a backend.

# Optional replicated nonlinear overlays. These should be long-window
# t=[350,700] or longer ensemble gates produced from concrete VMEC WOUTs, not
# reduced diagnostics.
NONLINEAR_ENSEMBLES: tuple[str, ...] = ()

command = [
    sys.executable,
    str(SPECTRAX_ROOT / "tools" / "build_vmec_boundary_transport_landscape.py"),
    "--baseline-input",
    str(BASELINE_INPUT),
    "--coefficient",
    COEFFICIENT,
    f"--fractions={FRACTIONS}",
    "--out-prefix",
    str(OUT_PREFIX),
    "--surfaces",
    SURFACES,
    "--alphas",
    ALPHAS,
    "--ky-values",
    KY_VALUES,
    "--ntheta",
    str(NTHETA),
    "--mboz",
    str(MBOZ),
    "--nboz",
    str(NBOZ),
    "--n-laguerre",
    str(N_LAGUERRE),
    "--n-hermite",
    str(N_HERMITE),
]
if EVALUATE_REDUCED:
    command.append("--evaluate-reduced")
else:
    command.extend(["--reuse-reduced-json", str(REUSE_REDUCED_JSON)])
if SOLVER_DEVICE is not None:
    command.extend(["--solver-device", str(SOLVER_DEVICE)])
for ensemble in NONLINEAR_ENSEMBLES:
    command.extend(["--nonlinear-ensemble", ensemble])

print("Running QA transport parameter scan:")
print("  " + " ".join(str(item) for item in command))
subprocess.run(command, cwd=SPECTRAX_ROOT, check=True)
print(f"\nWrote scan panel: {OUT_PREFIX.with_suffix('.png')}")
print(f"Wrote scan JSON:  {OUT_PREFIX.with_suffix('.json')}")
