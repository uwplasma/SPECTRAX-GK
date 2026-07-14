# Optimization Examples

These examples use the current VMEC-JAX equilibrium/optimization API and
SPECTRAX-GK turbulence objectives without an intermediate file or legacy
optimizer adapter.

## QA Transport Optimizations

Each script follows `vmec_jax/examples/optimization/QA_optimization.py`: it
starts from the same perturbed circular seed, continues boundary modes 1 through
5, and retains the QA, aspect-ratio 6, and mean-iota 0.42 objective terms. The
only added tuple is one SPECTRAX-GK ITG observable:

```bash
python examples/optimization/QA_optimization_linear_ITG.py
python examples/optimization/QA_optimization_quasilinear_ITG.py
python examples/optimization/QA_optimization_nonlinear_ITG.py
```

The examples are deliberately configured through readable constants rather
than argument parsers or hidden high-level drivers. Their core is the current
VMEC-JAX least-squares interface:

```python
objective_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 10.0),
    (transport_objective, 0.0, TRANSPORT_WEIGHT),
]
result = opt.least_squares(
    objective_terms,
    inp,
    max_mode=max_mode,
    jac=JAC,
    use_ess=True,
)
```

The fixed flux-tube controls are `SURFACE_INDEX`, `ALPHA`, `NTHETA`,
`SELECTED_KY_INDEX`, `N_LAGUERRE`, `N_HERMITE`, `R_OVER_LT`, and `R_OVER_LN`.
Change them explicitly when moving the objective to another surface, field
line, wavenumber, velocity resolution, or drive.

| Script | Objective and derivative route | Claim boundary |
| --- | --- | --- |
| `QA_optimization_linear_ITG.py` | Dominant linear ITG growth rate; implicit equilibrium Jacobian plus differentiable eigensolve | Linear microstability optimization at the selected flux tube |
| `QA_optimization_quasilinear_ITG.py` | Mixing-length heat-flux proxy; finite-difference outer Jacobian because it uses a nonsymmetric eigenvector | Screening/model-development evidence, not universal absolute flux |
| `QA_optimization_nonlinear_ITG.py` | Smooth reduced nonlinear-window proxy; finite-difference outer Jacobian | Candidate generation only, not a saturated turbulent-flux average |

Current JAX supports the growth objective in the implicit path. The QL and
reduced nonlinear objectives use dominant eigenvectors, for which the required
nonsymmetric eigenvector derivative is unavailable; those two examples
therefore set `JAC = None` honestly rather than silently dropping a gradient.

All scripts write an optimized input deck, WOUT file, and standard VMEC-JAX
plots. They do not embed a long campaign launcher. To evaluate a candidate,
use the separate reproducible audit examples:

```bash
python examples/optimization/QA_nonlinear_ITG_matched_audit.py
python examples/optimization/QA_nonlinear_ITG_transport_matrix.py
python examples/optimization/QA_parameter_scan.py
```

Set `BASELINE_VMEC_FILE` and `CANDIDATE_VMEC_FILE` in the matrix example to
the solved WOUT files being compared; the matched-audit example similarly
points to their accepted ensemble sidecars.

A nonlinear reduction is promoted only from converged, replicated,
post-transient windows. The current production policy uses staged horizons
`700,1100,1500`, averages over `t=[1100,1500]`, and includes independent seed
and timestep variants. Optimizer residuals, startup traces, and reduced
nonlinear-window proxies are not saturated heat-flux evidence.

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
python tools/artifacts/build_matched_nonlinear_transport_matrix.py write \
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
python tools/release/check_nonlinear_transport_gates.py matrix-progress \
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
use `tools/release/check_nonlinear_transport_gates.py target-time` before skipping an existing file,
and wrap each output in a per-output lock (`flock` with a `mkdir` fallback), so
interrupted runs are safe to relaunch without manually deleting partial
checkpoint bundles and future split-worker launches do not race on the same
NetCDF bundle.

When more than one candidate family is available, select the release claim
with the portfolio gate rather than by hand:

```bash
python tools/release/check_nonlinear_transport_gates.py matrix-portfolio \
  --matrix-report accepted_qa_ess=tools_out/qa_ess_matrix/artifacts/qa_ess_matrix_report.json \
  --matrix-report projected_0p001=tools_out/projected_0p001_matrix/artifacts/projected_0p001_matrix_report.json \
  --matrix-report projected_0p0005=tools_out/projected_0p0005_matrix/artifacts/projected_0p0005_matrix_report.json \
  --excluded-comparison strict_growth=docs/_static/vmec_qa_t1500_baseline_to_growth_comparison.json \
  --excluded-comparison strict_quasilinear=docs/_static/vmec_qa_t1500_baseline_to_quasilinear_comparison.json \
  --excluded-comparison strict_nonlinear_window=docs/_static/vmec_qa_t1500_baseline_to_nonlinear_window_comparison.json \
  --out-json tools_out/nonlinear_transport_matrix_portfolio.json \
  --out-figure tools_out/nonlinear_transport_matrix_portfolio.png
```

The portfolio report promotes only a passing broad matrix family. The strict
`t=1500` growth, quasilinear, and nonlinear-window rows are retained as
negative-transfer evidence and never counted toward broad nonlinear
turbulent-flux optimization promotion. The tracked max-mode-5 release campaign
failed this gate for accepted QA/ESS and both projected-weight families; keep
those reports as negative evidence and do not run the finalizer/importer unless
a future portfolio JSON has `passed=true`.

After the portfolio gate passes, finalize the release-facing docs and status
panels with the fail-closed wrapper:

```bash
python tools/campaigns/finalize_nonlinear_transport_matrix_release.py \
  --portfolio-json tools_out/nonlinear_transport_matrix_portfolio.json \
  --portfolio-figure tools_out/nonlinear_transport_matrix_portfolio.png \
  --matrix-report-json selected_family=tools_out/selected_matrix/artifacts/selected_matrix_report.json \
  --matrix-report-figure selected_family=tools_out/selected_matrix/artifacts/selected_matrix_report.png
```

This wrapper imports only a passing portfolio into `docs/_static` and then
regenerates the manuscript-readiness and pre-manuscript closure dashboards.

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
python tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py --dry-run
```

A typical constraints-only branch is:

```bash
python tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py \
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
python tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py \
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
python tools/campaigns/write_vmec_jax_optimizer_comparison_manifest.py \
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
python tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py \
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
