# Optimization Examples

This directory is reserved for actual VMEC-JAX QA stellarator optimization workflows with SPECTRAX-GK transport objectives.

## VMEC-JAX-Style QA Transport Scripts

Use these when the goal is a real VMEC-JAX QA optimization with the upstream high-weight iota target preserved and one SPECTRAX-GK transport objective appended to the VMEC-JAX objective tuple list:

```bash
python examples/optimization/QA_optimization_with_growth_rate.py
python examples/optimization/QA_optimization_with_quasilinear_flux.py
python examples/optimization/QA_optimization_with_nonlinear_heat_flux.py
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

## Configurable QA Driver

For dry-runs, guarded transport-weight ladders, solved-WOUT admission gates, and bounded optimizer-budget checks, use:

```bash
python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py --dry-run
```

A typical constraints-only branch is:

```bash
python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py \
  --constraints-only \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --mboz 21 \
  --nboz 21 \
  --make-plots \
  --outdir runs/qa_constraints_only
```

A transport-aware branch should start from a solved baseline and use a small transport weight:

```bash
python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --mboz 21 \
  --nboz 21 \
  --make-plots \
  --outdir runs/qa_transport_refinement \
  --spectrax-weight 0.005 \
  --transport-kind growth \
  --surfaces 0.64 \
  --alphas 0.0 \
  --ky-values 0.30
```

Use `growth` first because it is the cheapest differentiable transport target. Promote to quasilinear or nonlinear transport only after the geometry gates and finite-difference gradient checks pass. Nonlinear turbulent-flux claims require long post-transient replicated SPECTRAX-GK audits, not startup windows.

## Expected Outputs

Solved runs write optimizer history, final VMEC input/WOUT files when available, SPECTRAX-GK transport diagnostics, and `solved_wout_gate.json`. The gate fails closed if the final equilibrium violates the aspect, iota, or quasisymmetry constraints. This is intentional: a transport residual reduction that breaks the QA equilibrium is not an accepted optimized stellarator.
