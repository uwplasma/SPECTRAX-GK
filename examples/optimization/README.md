# Optimization Examples

This directory is reserved for actual VMEC-JAX QA stellarator optimization workflows with SPECTRAX-GK transport objectives.

## VMEC-JAX-Style QA Transport Scripts

Use these when the goal is a real VMEC-JAX QA optimization with the upstream high-weight iota target preserved and one SPECTRAX-GK transport objective appended to the VMEC-JAX objective tuple list:

```bash
python examples/optimization/QA_optimization_linear_ITG.py
python examples/optimization/QA_optimization_quasilinear_ITG.py
python examples/optimization/QA_optimization_nonlinear_ITG.py
python examples/optimization/QA_nonlinear_ITG_matched_audit.py
python examples/optimization/QA_nonlinear_ITG_transport_matrix.py
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

## How To Modify The Optimization Examples

The production examples are meant to be edited in-place, not wrapped by hidden
driver APIs. Keep the upstream QA/aspect/iota block intact and change only the
top-level constants needed for the scientific question:

| What to change | Constants or files | Notes |
| --- | --- | --- |
| Optimizer algorithm | `METHOD`, `SCIPY_TR_SOLVER`, `USE_ESS`, `ALPHA`, `MAX_NFEV`, `INNER_MAX_ITER`, `INNER_FTOL` | Use `scalar_trust` or `lbfgs_adjoint` for SPECTRAX-GK transport objectives. Use dense `scipy`/`exact` mainly for constraints-only QA baselines because it asks for forward-mode JVP columns. |
| Geometry or VMEC seed | `WARM_START_INPUT_FILE`, `SIMPLE_SEED_INPUT_FILE`, `INPUT_FILE`, `MAX_MODE`, `MIN_VMEC_MODE`, `USE_SIMPLE_SEED` | Point these to another VMEC-JAX input deck when studying a different QA/QH/QI family. For matrix audits, edit `BASELINE_VMEC_FILE` and `CANDIDATE_VMEC_FILE` in `QA_nonlinear_ITG_transport_matrix.py` to use solved WOUT files. |
| Transport objective | `SPECTRAX_KIND`, `SPECTRAX_WEIGHT`, `SPECTRAX_OBJECTIVE_TRANSFORM`, `SPECTRAX_OBJECTIVE_SCALE` | Supported example objectives are `growth`, `quasilinear_flux`, and `nonlinear_window_heat_flux`. Treat the nonlinear-window objective as a differentiable screening residual, not as a saturated turbulent-flux average. |
| Physics sample set | `SPECTRAX_SURFACES`, `SPECTRAX_ALPHAS`, `SPECTRAX_KY_VALUES`, `SPECTRAX_NTHETA`, `SPECTRAX_MBOZ`, `SPECTRAX_NBOZ` | Keep `mboz,nboz >= 21` for VMEC/Boozer transport rows. The default sample set matches the broad matrix gate: three surfaces, two field-line labels, and three `k_y` values. |
| Extra equilibrium objectives | Append another `(objective.J, target, weight)` tuple to `objective_tuples` | Examples include magnetic well, `LgradB`, finite-beta, bootstrap-current, or current-profile terms from VMEC-JAX. Keep weights explicit so transport changes cannot silently weaken the QA/iota/aspect gate. |
| Production nonlinear audit | `NONLINEAR_AUDIT_*` constants or `QA_nonlinear_ITG_transport_matrix.py` | A production claim requires long post-transient replicated windows over `t=[1100,1500]`, followed by the matched audit or broad matrix portfolio gate. Do not promote optimizer residuals or startup traces. |

For a new geometry family, first run a constraints-only QA/QH/QI equilibrium
solve, save the admitted WOUT, and only then add a small SPECTRAX-GK transport
weight. For a new objective function, add it as one explicit tuple in
`objective_tuples`, run AD/finite-difference checks, and keep the long-window
nonlinear audit separate from the differentiable optimizer residual.

Use the following claim boundaries when citing these scripts or their generated
sidecars:

| Script | Differentiable objective | Claim boundary |
| --- | --- | --- |
| `QA_optimization_linear_ITG.py` | Linear ITG growth-rate residual | Trace-safe VMEC-JAX plus SPECTRAX-GK objective-refinement evidence only; not a quasilinear calibration and not a nonlinear heat-flux reduction claim. |
| `QA_optimization_quasilinear_ITG.py` | Electrostatic quasilinear heat-flux residual | Screening/model-development evidence only; not an absolute flux predictor and not a nonlinear turbulent-flux optimization claim. |
| `QA_optimization_nonlinear_ITG.py` | Reduced nonlinear-window heat-flux screening residual | Startup/window-estimator evidence only; not a converged nonlinear transport average and not a nonlinear turbulent-flux optimization success claim. |
| `QA_nonlinear_ITG_matched_audit.py` | Matched replicated nonlinear heat-flux ensemble comparison | Production-evidence audit for already-run long post-transient baseline/candidate ensembles; this is the gate that accepts or rejects a nonlinear turbulent-flux reduction. |
| `QA_nonlinear_ITG_transport_matrix.py` | Multi-surface, multi-field-line, multi-`k_y` matched nonlinear matrix writer | Broad nonlinear turbulent-flux optimization launch contract; this writes the campaign and promotion scripts but the claim passes only after every long-window ensemble and matched comparison passes. |
| `QA_parameter_scan.py` | `RBC(1,1)` linear/QL landscape plus concrete nonlinear sidecars | Landscape and noise/convergence diagnostics only; reduced/startup nonlinear-window diagnostics are excluded from optimization-promotion claims. |

The optimization scripts write strict long-window initial/final nonlinear ITG
audit config manifests after the VMEC-JAX solve. The current promotion policy
uses staged horizons `700,1100,1500`, averages only over `t=[1100,1500]`, and
replicates the final window with seed and timestep variants. These audits are
not launched by default; edit `RUN_LONG_NONLINEAR_AUDIT_COMMANDS = True` inside
the script to run them and build the initial-vs-final nonlinear `Q(t)`
comparison plot, or run the commands from the generated `run_manifest.json` on
a GPU node.

After those long-window runs finish, edit the ensemble paths at the top of
`QA_nonlinear_ITG_matched_audit.py` and run:

```bash
python examples/optimization/QA_nonlinear_ITG_matched_audit.py
```

The script consumes only accepted ensemble-gate JSON sidecars, applies the
matched baseline-vs-optimized reduction and uncertainty-separation gates, and
writes `results/qa_opt/nonlinear_matched_audit/qa_nonlinear_ITG_matched_audit.{json,csv,png}`.
Use this path for production turbulent-flux evidence; do not cite optimizer
residuals, startup traces, or reduced nonlinear-window values as saturated heat
flux reductions.

For a broad nonlinear turbulent-flux optimization claim, use the matched matrix
campaign instead of a single audit. Edit
`QA_nonlinear_ITG_transport_matrix.py` to point at the solved baseline and
candidate WOUT files, then run:

```bash
python examples/optimization/QA_nonlinear_ITG_transport_matrix.py
```

That example writes the same commands as the lower-level tool invocation below.
The default matrix is the current paper-facing gate: three surfaces, two
field-line labels, and three `k_y` values, with seed/timestep replicated
nonlinear windows over `t=[1100,1500]`:

```bash
python tools/build_matched_nonlinear_transport_matrix.py write \
  --baseline-vmec-file /path/to/baseline/wout_final.nc \
  --candidate-vmec-file /path/to/candidate/wout_final.nc \
  --baseline-label strict_qa \
  --candidate-label low_transport \
  --case-prefix qa_low_transport_matrix \
  --out-dir tools_out/qa_low_transport_matrix \
  --artifact-dir tools_out/qa_low_transport_matrix/artifacts \
  --gpu-splits 2

./tools_out/qa_low_transport_matrix/run_matrix_final_horizon_gpu0.sh
./tools_out/qa_low_transport_matrix/run_matrix_final_horizon_gpu1.sh
python tools/check_matched_nonlinear_transport_matrix_progress.py \
  --matrix-manifest tools_out/qa_low_transport_matrix/matched_transport_matrix_manifest.json \
  --out-json tools_out/qa_low_transport_matrix/artifacts/progress.json
./tools_out/qa_low_transport_matrix/run_matrix_postprocess.sh
```

The generated aggregate report passes only after every sample has completed
its baseline and candidate ensemble gates and the matched reductions satisfy
the configured pass-fraction and mean-reduction policy.
Run the progress checker before postprocessing; a checkpointed output can have
all three NetCDF bundle files present while its recorded `Grids/time` is still
below the final transport window. The generated final-horizon launch scripts
use `tools/check_nonlinear_output_target.py` before skipping an existing file,
and wrap each output in a per-output lock (`flock` with a `mkdir` fallback), so
interrupted runs are safe to relaunch without manually deleting partial
checkpoint bundles and future split-worker launches do not race on the same
NetCDF bundle.

When more than one candidate family is available, select the release claim
with the portfolio gate rather than by hand:

```bash
python tools/check_nonlinear_transport_matrix_portfolio.py \
  --matrix-report accepted_qa_ess=tools_out/qa_ess_matrix/artifacts/qa_ess_matrix_report.json \
  --matrix-report projected_0p001=tools_out/projected_0p001_matrix/artifacts/projected_0p001_matrix_report.json \
  --excluded-comparison strict_growth=docs/_static/vmec_qa_t1500_baseline_to_growth_comparison.json \
  --excluded-comparison strict_quasilinear=docs/_static/vmec_qa_t1500_baseline_to_quasilinear_comparison.json \
  --excluded-comparison strict_nonlinear_window=docs/_static/vmec_qa_t1500_baseline_to_nonlinear_window_comparison.json \
  --out-json tools_out/nonlinear_transport_matrix_portfolio.json \
  --out-figure tools_out/nonlinear_transport_matrix_portfolio.png
```

The portfolio report promotes only a passing broad matrix family. The strict
`t=1500` growth, quasilinear, and nonlinear-window rows are retained as
negative-transfer evidence and never counted toward broad nonlinear
turbulent-flux optimization promotion.

`QA_parameter_scan.py` scans `RBC(1,1)` from `-75%` to `+75%` by default and
regenerates the linear/quasilinear objective landscape. The top panel includes
linear growth and every shipped electrostatic quasilinear heat-flux rule on the
same multi-surface, multi-field-line, multi-`k_y` sample set used by the
optimization scripts. The lower panel is reserved for true post-transient
nonlinear heat-flux means; add it only through long-window ensemble sidecars
from concrete nonlinear outputs, not reduced/startup nonlinear-window
diagnostics.

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

For optimizer-comparison campaigns, generate commands from one manifest rather
than by hand-editing launch scripts:

```bash
python tools/write_vmec_jax_optimizer_comparison_manifest.py \
  --campaign-root tools_out/vmec_jax_qa_optimizer_comparison_campaign \
  --out-json docs/_static/vmec_jax_qa_optimizer_comparison_manifest.json
```

The manifest emits the strict baseline command, matched `scipy`, `scalar_trust`,
and `lbfgs_adjoint` transport-refinement commands, plus SPSA/CMA/BO outer-loop
contracts with deterministic metric-evaluation and nonlinear-audit templates.
Use the generated `comparison_fingerprint` to keep optimizer comparisons scoped
to identical sample sets, moment resolution, objective transforms, budgets, and
strict long-window audit policies.

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
audits, grid/window convergence, and matched baseline-vs-optimized reductions,
not startup or reduced-window objectives.

## Expected Outputs

Solved runs write optimizer history, final VMEC input/WOUT files when available, SPECTRAX-GK transport diagnostics, and `solved_wout_gate.json`. The strict gate fails closed if the final equilibrium violates the aspect, `|iota| >= 0.41`, or quasisymmetry constraints. For exploratory transport rows, a small iota shortfall can still be retained as `diag-ok` evidence when `|iota| >= 0.39`, aspect and QS remain acceptable, and the result is clearly described as a diagnostic candidate rather than an accepted optimized stellarator. A nonlinear turbulent-flux claim still requires matched long post-transient nonlinear audits.
