# Optimization Examples

This directory contains two classes of optimization examples.

- VMEC-JAX-style solved-boundary scripts that intentionally mirror upstream
  `vmec_jax/examples/optimization/QA_optimization.py` and only append one
  SPECTRAX-GK transport tuple to `objective_tuples`.
- Reduced max-mode-1 scripts that run quickly and regenerate documentation
  figures for AD/finite-difference, UQ, plotting, and objective-plumbing gates.

## VMEC-JAX-Style QA Transport Scripts

Use these when the goal is a real VMEC-JAX QA optimization with the upstream
high-weight iota target preserved:

```bash
python examples/optimization/QA_optimization_with_growth_rate.py
python examples/optimization/QA_optimization_with_quasilinear_flux.py
python examples/optimization/QA_optimization_with_nonlinear_heat_flux.py
```

Each script keeps the VMEC-JAX constants visible near the top:

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

Keep `SPECTRAX_WEIGHT` small while tuning. The QA/aspect/iota terms must remain
the dominant solved-equilibrium gate before any final WOUT is sent to expensive
long-window nonlinear transport audits.
The scripts are configured by editing constants, not command-line flags. They
accept only `--help`; any other argument fails before an optimization is
launched so accidental flags do not create `results/` artifacts.

The scripts default to `METHOD = "scalar_trust"` because the SPECTRAX-GK
transport residual contains reverse-mode custom-VJP pieces. The pure VMEC-JAX
QA script can use dense `scipy`/`exact`, but that path asks for forward-mode
JVP columns and is not the right default once the SPECTRAX-GK transport tuple is
active. For research runs, use a two-stage workflow: solve and verify the
upstream QA baseline first, then refine from that solved input/WOUT with
transport weight, AD/finite-difference gradient gates, and matched nonlinear
audits. Running one of these scripts is not a transport-optimization success
claim until those gates pass.

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
diagnostics for the three optimized controls. These are reduced max-mode-1
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
python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py \
  --dry-run \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7
```

That command assembles the VMEC-JAX QA objective blocks and writes a setup
summary without solving. The canonical implementation is
`examples/optimization/vmec_jax_qa_low_turbulence_optimization.py`.

Run an aspect-6 constraints-only branch:

```bash
python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py \
  --constraints-only \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --make-plots \
  --outdir runs/qa_constraints_only
```

Run the transport-aware design:

```bash
python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py \
  --use-simple-seed \
  --max-mode 5 \
  --min-vmec-mode 7 \
  --make-plots \
  --outdir runs/qa_plus_reduced_nonlinear_heat_flux \
  --spectrax-weight 0.05 \
  --transport-kind nonlinear_window_heat_flux \
  --surfaces 0.45,0.64,0.78 \
  --alphas 0.0,0.7853981633974483 \
  --ky-values 0.10,0.30,0.50
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
python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py \
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

The candidate-comparison JSON uses `mean_iota_lower_bound` and
`iota_profile_floor` for this A=6 admission policy. Any legacy `target_*` iota
fields in that JSON are compatibility aliases for lower-bound gates, not the
upstream VMEC-JAX QA script's exact mean-iota target.

```bash
python examples/optimization/vmec_jax_qa_low_turbulence_optimization.py \
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
transport weight whose `solved_wout_gate.json` passes and whose explicit
transport metric improves relative to the admitted QA baseline. If none pass,
the QA-only WOUT remains the only candidate ready for a matched long-window
nonlinear SPECTRAX-GK audit. Single-surface/single-`alpha`/single-`ky`
transport runs are useful for smoke tests and diagnostics only; they are not
admission evidence for the production nonlinear audit queue.

After a real QA-only and transport-aware pair is produced, the next scientific
step is not another reduced figure. It is a matched long-window SPECTRAX-GK
nonlinear transport audit of both final WOUT files, with running-average,
block-statistics, and seed/replicate checks.
