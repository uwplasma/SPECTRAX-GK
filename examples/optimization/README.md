# Optimization Examples

This directory contains small, runnable optimization examples. The main
SPECTRAX-GK examples are intentionally split into three transport objectives:

- small linear ITG growth rate;
- small quasilinear ITG heat-flux proxy;
- small reduced nonlinear-window ITG heat-flux envelope.

Each script follows the editable structure of VMEC-JAX
`examples/optimization/QA_optimization.py`: visible problem constants, explicit
QA/aspect/iota residuals, one transport residual, optimizer settings,
AD/finite-difference gates, and explicit artifact generation. The compact
package API remains available, but these examples avoid hiding the workflow
behind a single `optimize_stellarator_itg(...)` call.

The examples are split into two levels:

- Reduced, fast figure generation for README/manuscript development panels.
- VMEC-JAX fixed-boundary QA optimization with an optional SPECTRAX-GK transport
  residual for researchers who want to experiment with solved equilibria.

## Three Reduced ITG Optimization Examples

From the repository root:

```bash
python examples/optimization/stellarator_itg_growth_optimization.py
python examples/optimization/stellarator_itg_quasilinear_flux_optimization.py
python examples/optimization/stellarator_itg_nonlinear_heat_flux_optimization.py
python examples/optimization/compare_stellarator_itg_optimizations.py
```

Outputs:

- `docs/_static/stellarator_itg_growth_optimization.{json,png,pdf}`
- `docs/_static/stellarator_itg_quasilinear_optimization.{json,png,pdf}`
- `docs/_static/stellarator_itg_nonlinear_optimization.{json,png,pdf}`
- `docs/_static/stellarator_itg_optimization_comparison.{json,png,pdf}`

The combined panel compares objective histories, reduced nonlinear heat-flux
responses, fixed-gradient heat-flux traces, and reduced LCFS/Boozer `|B|`
diagnostics for the three optimized controls. The LCFS maps use a 72-by-72
angular grid and a shared `jet` colormap. These are reduced max-mode-1
diagnostics for optimization plumbing; they are not solved VMEC surfaces and
they are not production nonlinear turbulent heat-flux claims.

## Rebuild The Reduced QA Low-Turbulence Figure

From the repository root:

```bash
python tools/build_qa_low_turbulence_comparison.py
python tools/build_qa_low_turbulence_time_horizon_audit.py
```

Outputs:

- `docs/_static/qa_low_turbulence_comparison.png`
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

Every solved run now writes `solved_wout_gate.json` and exits nonzero unless
the final equilibrium satisfies the aspect, mean-iota, solved iota-profile, and
quasisymmetry gates. This is intentional: a transport-weight refinement that
reduces a reduced SPECTRAX-GK residual but breaks the QA/iota constraints is not
ready for a long nonlinear turbulent-flux audit. For exploratory sweeps that
should keep failed candidates, pass `--allow-failed-solved-wout-gate`.

For transport-weight ladders, use the fail-closed runner instead of manually
choosing from `history.json`:

```bash
python tools/run_vmec_jax_guarded_transport_ladder.py \
  --constraints-dir runs/qa_constraints_only \
  --outdir runs/qa_transport_ladder \
  --weights 0.0005,0.001,0.0025,0.005 \
  --driver-args "--max-mode 5 --min-vmec-mode 7 --mboz 21 --nboz 21"
```

The runner restarts every candidate from the QA `input.final`, writes a summary
JSON, keeps failed candidates for diagnosis, and promotes only the largest
transport weight whose `solved_wout_gate.json` passes. If none pass, the
QA-only WOUT remains the only candidate ready for a matched long-window
nonlinear SPECTRAX-GK audit.

After a real QA-only and transport-aware pair is produced, the next scientific
step is not another reduced figure. It is a matched long-window SPECTRAX-GK
nonlinear transport audit of both final WOUT files, with running-average,
block-statistics, and seed/replicate checks.
