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
plots. They do not embed a long campaign launcher.

A nonlinear reduction is promoted only from converged, replicated,
post-transient windows. The current production policy uses staged horizons
`700,1100,1500`, averages over `t=[1100,1500]`, and includes independent seed
and timestep variants. Optimizer residuals, startup traces, and reduced
nonlinear-window proxies are not saturated heat-flux evidence.

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


## Expected Outputs

Solved runs write optimizer history, final VMEC input/WOUT files when available, SPECTRAX-GK transport diagnostics, and `solved_wout_gate.json`. The strict gate fails closed if the final equilibrium violates the aspect, `|iota| >= 0.41`, or quasisymmetry constraints. For exploratory transport rows, a small iota shortfall can still be retained as `diag-ok` evidence when `|iota| >= 0.39`, aspect and QS remain acceptable, and the result is clearly described as a diagnostic candidate rather than an accepted optimized stellarator. A nonlinear turbulent-flux claim still requires matched long post-transient nonlinear audits.
