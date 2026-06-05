# Optimization Examples

This directory is reserved for actual VMEC-JAX QA stellarator optimization workflows with SPECTRAX-GK transport objectives.

## VMEC-JAX-Style QA Transport Scripts

Use these when the goal is a real VMEC-JAX QA optimization with the upstream high-weight iota target preserved and one SPECTRAX-GK transport objective appended to the VMEC-JAX objective tuple list:

```bash
python examples/optimization/QA_optimization_linear_ITG.py
python examples/optimization/QA_optimization_quasilinear_ITG.py
python examples/optimization/QA_optimization_nonlinear_ITG.py
python examples/optimization/QA_parameter_scan.py
```

Each script deliberately follows the structure of `vmec_jax/examples/optimization/QA_optimization.py`: constants are visible at the top level, the objective blocks are assembled in `objective_tuples`, and there is no argparse `main()` wrapper. The only supported command-line argument is `--help`; any other argument fails before a `results/` directory can be created.

```python
MAX_MODE = 5
TARGET_ASPECT = 5.0
TARGET_IOTA = 0.41
IOTA_WEIGHT = 10_000.0
objective_tuples = [
    (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
    (iota.J, TARGET_IOTA, IOTA_WEIGHT),
    (qs.J, 0.0, QS_WEIGHT),
    (transport.J, 0.0, SPECTRAX_WEIGHT),
]
```

Keep `SPECTRAX_WEIGHT` small while tuning. The QA, aspect-ratio, and iota constraints must remain the dominant solved-equilibrium gate before any final WOUT is sent to long-window SPECTRAX-GK nonlinear transport audits.

The transport scripts default to `METHOD = "scalar_trust"`. SPECTRAX-GK transport residuals include reverse-mode custom-VJP components, while the pure VMEC-JAX dense `scipy`/`exact` least-squares path requests forward-mode JVP columns. For publication work, use a two-stage workflow:

1. Solve and verify the upstream VMEC-JAX QA baseline first.
2. Restart/refine from that solved input or WOUT with a small SPECTRAX-GK transport weight.
3. Gate the result with AD/finite-difference checks, solved-WOUT aspect/iota/QS checks, Boozer/geometry diagnostics, and matched long post-transient nonlinear heat-flux audits.

Running one script is not a transport-optimization success claim, and is not,
by itself, a nonlinear turbulent-flux optimization success claim.

The optimization scripts write long-window initial/final nonlinear ITG audit
config manifests after the VMEC-JAX solve. These audits are not launched by
default; edit `RUN_LONG_NONLINEAR_AUDIT_COMMANDS = True` inside the script to
run them and build the initial-vs-final nonlinear `Q(t)` comparison plot, or
run the commands from the generated `run_manifest.json` on a GPU node.

`QA_parameter_scan.py` scans `RBC(0,1)` by default and regenerates the noisy
linear/quasilinear/nonlinear objective landscape with replicated nonlinear
error bars from tracked ensemble gates.

## Campaign Tooling

The user-facing files in this directory are intentionally edited through
top-level constants. Argparse-heavy dry-runs, guarded transport-weight ladders,
solved-WOUT admission gates, and bounded optimizer-budget checks live under
`tools/` so the examples remain close to VMEC-JAX `QA_optimization.py`:

```bash
python tools/vmec_jax_qa_low_turbulence_optimization.py --dry-run
```

A typical constraints-only branch is:

```bash
python tools/vmec_jax_qa_low_turbulence_optimization.py \
  --constraints-only \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --mboz 21 \
  --nboz 21 \
  --make-plots \
  --outdir runs/qa_constraints_only
```

For paper-facing sweeps, prefer the strict upstream baseline preset:

```bash
python tools/vmec_jax_qa_low_turbulence_optimization.py \
  --strict-upstream-qa-baseline \
  --solver-device gpu \
  --outdir runs/qa_baseline_strict_upstream
```

This uses the same upstream simple seed, `MAX_MODE = 5`, ESS scaling, and
aspect/iota/QS objective tuples as `vmec_jax/examples/optimization/QA_optimization.py`,
but tightens the outer step tolerance and budget so the final WOUT is admitted
by the strict solved-equilibrium gate. The preset keeps the gate at
`iota >= 0.41` and uses a small default optimizer target buffer
(`target iota = 0.4102`). A baseline that stops just below the gate should be
refined with this preset, not accepted by loosening the gate.

The current tracked strict-baseline evidence is summarized in
`docs/_static/vmec_jax_qa_strict_baseline/summary.json`: exact SciPy/ESS,
`nfev=39`, aspect `5.000154`, mean iota `0.4101997`, QS residual `2.60e-4`,
and a passed solved-WOUT gate. It is a constraints-only reference; rerun
matched SPECTRAX-GK nonlinear audits before comparing transport candidates
against this stricter WOUT.

A transport-aware branch should start from a solved baseline and use a small transport weight:

```bash
python tools/vmec_jax_qa_low_turbulence_optimization.py \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --mboz 21 \
  --nboz 21 \
  --make-plots \
  --outdir runs/qa_transport_refinement \
  --spectrax-weight 0.005 \
  --transport-kind growth \
  --surfaces 0.45,0.64,0.78 \
  --alphas 0.0,0.7853981633974483 \
  --ky-values 0.10,0.30,0.50
```

Use `growth` first because it is the cheapest differentiable transport target.
The sample set above is the admission-grade default used by the public
VMEC-JAX-style scripts; one-point samples are acceptable only for explicit
debugging. Promote to quasilinear or nonlinear transport only after the
geometry gates and finite-difference gradient checks pass. Nonlinear
turbulent-flux claims require long post-transient replicated SPECTRAX-GK
audits, not startup windows.

## Expected Outputs

Solved runs write optimizer history, final VMEC input/WOUT files when available, SPECTRAX-GK transport diagnostics, and `solved_wout_gate.json`. The gate fails closed if the final equilibrium violates the aspect, iota, or quasisymmetry constraints. This is intentional: a transport residual reduction that breaks the QA equilibrium is not an accepted optimized stellarator.
