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
solved-VMEC nonlinear turbulent heat-flux optimization claim.

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
  --outdir runs/qa_constraints_only
```

Run the transport-aware design:

```bash
python examples/optimization/QA_optimization_with_nonlinear_heat_flux.py \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --outdir runs/qa_plus_reduced_nonlinear_heat_flux \
  --spectrax-weight 0.05 \
  --transport-kind nonlinear_window_heat_flux \
  --surfaces 0.64 \
  --alphas 0.0 \
  --ky-values 0.3
```

The recommended solved-boundary commands above mirror the upstream VMEC-JAX QA
script: use a simple omnigeneity seed, optimize active boundary modes through
`max_mode=5`, target aspect ratio `A=6`, use the original high-weight
`MeanIota` target `iota = 0.41`, add a signed solved-profile floor
`iota(s) >= 0.41`, use `mboz=nboz=21`, and append a small SPECTRAX-GK transport
residual. The profile-floor gate must be checked from the final WOUT; a passed
mean-iota target is not sufficient. For quick local validation before launching
a longer solve, run a bounded growth-only smoke:

```bash
python examples/optimization/QA_optimization_with_nonlinear_heat_flux.py \
  --outdir /tmp/spectraxgk_vmec_jax_qa_scalar_smoke \
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

After a real QA-only and transport-aware pair is produced, the next scientific
step is not another reduced figure. It is a matched long-window SPECTRAX-GK
nonlinear transport audit of both final WOUT files, with running-average,
block-statistics, and seed/replicate checks.
