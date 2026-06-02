# Optimization Examples

This directory contains small, runnable optimization examples. The QA
low-turbulence scripts are intentionally split into two levels:

- Reduced, fast figure generation for the README/manuscript panel.
- VMEC-JAX fixed-boundary QA optimization with an optional SPECTRAX-GK transport
  residual for researchers who want to experiment with solved equilibria.

## Rebuild The README QA Figure

From the repository root:

```bash
python tools/build_qa_low_turbulence_comparison.py --pdf
python tools/build_qa_low_turbulence_time_horizon_audit.py --pdf
```

Outputs:

- `docs/_static/qa_low_turbulence_comparison.png`
- `docs/_static/qa_low_turbulence_comparison.pdf`
- `docs/_static/qa_low_turbulence_comparison.json`
- `docs/_static/qa_low_turbulence_comparison.summary.csv`
- `docs/_static/qa_low_turbulence_comparison.scan.csv`
- `docs/_static/qa_low_turbulence_time_horizon_audit.png`

The figure is a reduced differentiable optimization-plumbing artifact. It is
useful for teaching, debugging, and manuscript layout, but it is not a final
solved-VMEC nonlinear turbulent heat-flux optimization claim. The trace in the
figure is a smooth reduced envelope (`dE/dt = 2 gamma E - alpha E^2`,
`Q_env = W_i E`), not a chaotic SPECTRAX-GK nonlinear heat-flux trace. The
QA-only branch in that figure is also not the final WOUT from upstream
`vmec_jax/examples/optimization/QA_optimization.py`; use the solved-boundary
commands below when you need that baseline.

## VMEC-JAX QA + SPECTRAX-GK Transport Objective

This path requires the optional `vmec_jax` and `booz_xform_jax` packages. It is
for solved-boundary VMEC-JAX experimentation, not for the fast default install
path.

The easiest entry point is:

```bash
python examples/optimization/QA_optimization_with_nonlinear_heat_flux.py \
  --dry-run \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7
```

That command assembles the VMEC-JAX QA objective blocks and writes a setup
summary without solving. The canonical implementation is
`examples/optimization/vmec_jax_qa_low_turbulence_optimization.py`.

Run a QA-only baseline:

```bash
python examples/optimization/QA_optimization_with_nonlinear_heat_flux.py \
  --constraints-only \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --make-plots \
  --outdir runs/qa_constraints_only
```

Run the transport-aware design:

```bash
python examples/optimization/QA_optimization_with_nonlinear_heat_flux.py \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --make-plots \
  --outdir runs/qa_plus_reduced_nonlinear_heat_flux \
  --spectrax-weight 0.05 \
  --transport-kind nonlinear_window_heat_flux \
  --surfaces 0.64 \
  --alphas 0.0 \
  --ky-values 0.3
```

On a GPU node, append `--solver-device gpu`; otherwise JAX will use the
available default backend. The QA-only branch defaults to the upstream
VMEC-JAX `scipy` optimizer. The transport-aware branch defaults to
`scalar_trust` because dense exact SciPy Jacobians are memory-heavy for the
SPECTRAX-GK transport residual; override `--method` only when you have a
specific optimizer/memory reason.

The recommended solved-boundary commands above mirror the upstream VMEC-JAX QA
script structure: use a simple omnigeneity seed, optimize active boundary modes
through `max_mode=5` without mode continuation, and use the high-weight
`MeanIota` target `iota = 0.41`. The SPECTRAX-GK low-turbulence study targets
aspect ratio `A=6` and adds a signed solved-profile floor `iota(s) >= 0.41`;
the upstream VMEC-JAX example itself targets `A=5` and does not include that
profile-floor gate. To reproduce the upstream A=5 baseline, run:

```bash
VMEC_JAX_ROOT=/path/to/vmec_jax
python examples/optimization/QA_optimization_with_nonlinear_heat_flux.py \
  --constraints-only \
  --input "$VMEC_JAX_ROOT/examples/data/input.minimal_seed_nfp2" \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --target-aspect 5.0 \
  --min-iota 0.41 \
  --iota-objective target \
  --disable-iota-profile-floor \
  --method scipy \
  --scipy-tr-solver exact \
  --mboz 21 \
  --nboz 21 \
  --make-plots \
  --outdir runs/qa_constraints_only_upstream_a5
```

For the A=6 low-turbulence branch, use `mboz=nboz=21` and append a small
SPECTRAX-GK transport residual. The profile-floor gate must be checked from the
final WOUT; a passed mean-iota target is not sufficient. For quick local
validation before launching a longer solve, run a bounded growth-only smoke:

```bash
python examples/optimization/QA_optimization_with_nonlinear_heat_flux.py \
  --outdir /tmp/spectraxgk_vmec_jax_qa_scalar_smoke \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --method scalar_trust \
  --max-nfev 1 \
  --continuation-nfev 1 \
  --inner-max-iter 3 \
  --trial-max-iter 3 \
  --inner-ftol 1e-4 \
  --trial-ftol 1e-4 \
  --ntheta 4 \
  --n-laguerre 1 \
  --n-hermite 1 \
  --transport-kind growth \
  --surfaces 0.64 \
  --alphas 0.0 \
  --ky-values 0.3
```

When refining an already optimized `input.final`, add
`--disable-mode-continuation` so the driver optimizes the requested `max_mode`
branch directly instead of rebuilding the lower-mode continuation ladder. This
is useful for profile-floor and transport-weight sweeps after a solved WOUT has
already passed the basic aspect/iota/QS gates.

After a real QA-only and transport-aware pair is produced, the next scientific
step is not another reduced figure. It is a matched long-window SPECTRAX-GK
nonlinear transport audit of both final WOUT files, with running-average,
block-statistics, and seed/replicate checks.
