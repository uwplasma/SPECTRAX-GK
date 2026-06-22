#!/usr/bin/env python
"""Matched long-window nonlinear ITG audit for a QA transport candidate.

This example is the production-evidence companion to
``QA_optimization_nonlinear_ITG.py``. The optimizer script can propose a
transport-aware VMEC equilibrium, but the nonlinear turbulent-flux claim is
accepted only after matched baseline and optimized equilibria each pass
replicated post-transient SPECTRAX-GK nonlinear windows.

Edit the constants below to point at new baseline/candidate ensemble JSON
sidecars after running the long-window nonlinear campaign. The bundled default
rebuilds the tracked no-ESS reference versus optimized QA/ESS audit.
"""

from pathlib import Path
import json
import sys


SCRIPT_NAME = Path(__file__).name
USAGE = f"""\
Usage:
  python examples/optimization/{SCRIPT_NAME}

This script is configured by editing constants near the top of the file. It
does not launch nonlinear simulations; it consumes already-replicated ensemble
sidecars from long post-transient SPECTRAX-GK runs and writes a matched
baseline-vs-optimized audit figure plus JSON/CSV sidecars.
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

from tools.build_baseline_optimized_nonlinear_audit import (  # noqa: E402
    build_audit,
    write_audit_artifacts,
)


# Replace these with new ensemble-gate JSON files after a new long-window QA
# campaign. Each ensemble must be built from post-transient nonlinear heat-flux
# windows with seed and timestep replicates, not from startup or reduced-window
# optimizer residuals.
BASELINE_ENSEMBLE = (
    SPECTRAX_ROOT
    / "docs"
    / "_static"
    / "qa_no_ess_reference_replicates"
    / "qa_no_ess_reference_t700_ensemble_gate.json"
)
OPTIMIZED_ENSEMBLE = (
    SPECTRAX_ROOT
    / "docs"
    / "_static"
    / "optimized_equilibrium_replicates"
    / "optimized_equilibrium_replicate_t700_ensemble_gate.json"
)
SELECTED_OPTIMIZED_AUDIT = (
    SPECTRAX_ROOT / "docs" / "_static" / "production_nonlinear_optimization_guard.json"
)

CASE = "qa_no_ess_to_optimized_nonlinear_transport_audit"
MIN_RELATIVE_REDUCTION = 0.02
REQUIRE_UNCERTAINTY_SEPARATION = True
OUTPUT_DIR = Path("results/qa_opt/nonlinear_matched_audit")
OUT_JSON = OUTPUT_DIR / "qa_nonlinear_ITG_matched_audit.json"
OUT_CSV = OUTPUT_DIR / "qa_nonlinear_ITG_matched_audit.csv"
OUT_PNG = OUTPUT_DIR / "qa_nonlinear_ITG_matched_audit.png"


report = build_audit(
    baseline_path=BASELINE_ENSEMBLE,
    optimized_path=OPTIMIZED_ENSEMBLE,
    selected_optimized_audit_path=SELECTED_OPTIMIZED_AUDIT,
    case=CASE,
    min_relative_reduction=MIN_RELATIVE_REDUCTION,
    require_uncertainty_separation=REQUIRE_UNCERTAINTY_SEPARATION,
)
write_audit_artifacts(report, out_json=OUT_JSON, out_csv=OUT_CSV, out_png=OUT_PNG)

summary = {
    "passed": report["passed"],
    "relative_reduction": report["comparison"]["relative_reduction"],
    "uncertainty_separation_sigma": report["comparison"][
        "uncertainty_separation_sigma"
    ],
    "out_json": str(OUT_JSON),
    "out_png": str(OUT_PNG),
}
print(json.dumps(summary, indent=2, sort_keys=True))

if not bool(report["passed"]):
    raise SystemExit(1)
