# SPECTRAX-GK Active Plan and Running Log

Last updated: 2026-06-04
Active repository: `uwplasma/SPECTRAX-GK`
Current public baseline: `main` at v1.6.0
Historical planning archive: private repo `rogeriojorge/spectraxgk_plan`

This file is the public active plan and concise running log. Keep it short,
dated, and tied to reproducible artifacts, tests, figures, and gates. Detailed
historical logs live outside the release repository so clones stay small.

## Current Release Status

- CI/CD: green on `main` at `9aebb53`.
  - GitHub Actions run: `26968506370`.
  - Result: 59 successful jobs, 1 skipped.
  - Wide package coverage gate passed at exactly the required release level:
    `TOTAL 18064 stmts, 882 miss, 95%`.
- Repository-size policy: tracked payload must stay below 50 MB. This active
  plan replaces the old 531 KB historical log to restore edit headroom.
- Release posture: technically shippable; manuscript-level strict QA nonlinear
  turbulence-optimization claim remains open until a matched rerun-WOUT
  candidate improves the reduced transport metric and passes long-window audits.

## Active Lanes

| Lane | Status | Current gate |
| --- | ---: | --- |
| CI/CD, release infrastructure, package coverage | 100% | Green CI, 95% package-wide coverage |
| Rerun-WOUT admission and artifact policy | 100% | Explicit authoritative rerun-WOUT path implemented and tested |
| Strict QA candidate screening | 95% | Top-12 projected edge candidate passes rerun-WOUT gates and reduces the 18-point metric by 2.29% |
| Strict nonlinear turbulent-flux optimization evidence | 80% | Matched t=700 baseline/candidate nonlinear audit is launched on office; no turbulent-flux claim until it passes |
| Docs/readme/release hygiene | 90% | Needs final sync after the next admitted or failed scientific tranche |
| Performance/parallelization release lane | 95% | Independent-work parallel paths are release-ready; nonlinear domain sharding remains research/development |

Deferred post-release/manuscript extensions unless explicitly reprioritized:
W7-X zonal long-window recurrence/damping, W7-X TEM/multi-flux-tube extension,
and production nonlinear domain decomposition claims.

## Strict QA Baseline Convention

The max-mode-5 VMEC-JAX QA baseline is now handled under an explicit
rerun-WOUT-authoritative convention.

Primary office artifact:
`/home/rjorge/tmp/spectrax_strict_qa_rerun_gate_bd85fae`

Optimizer-state solved WOUT:

- `nfev = 39`, wall time `706.95 s`.
- Aspect: `5.000154379`.
- Mean iota: `0.410199722`.
- QS residual: `2.60098e-4`.
- Solved-equilibrium gate: passed.

Deterministic rerun WOUT from `input.final`:

- File: `wout_final_rerun.nc`.
- Aspect: `5.000154379`.
- Mean iota: `0.411691350`.
- Profile minima: `0.402859 / 0.402619`.
- QS residual: `1.849256e-4`.
- Rerun-WOUT admission gate: passed.
- Reproducibility gate relative to optimizer-state WOUT: failed, because the
  optimizer-state and fixed-input rerun equilibria are measurably different.

Policy: downstream transport plots, reduced metrics, and nonlinear audit TOMLs
may use `wout_final_rerun.nc` only when `rerun_wout_admission_gate.json` passes
and the optimizer-state drift remains visible in the artifact metadata. Failed
rerun reproducibility alone must not silently promote optimizer-state WOUTs.

## Reduced Transport Admission Metric

Baseline reduced metric under the strict rerun-WOUT convention uses the
18-point admission sample:

- `s = (0.45, 0.64, 0.78)`.
- `alpha = (0, pi/4)`.
- `k_y rho_i = (0.10, 0.30, 0.50)`.
- `mboz = nboz = 21`.

Strict baseline reduced metrics:

- Growth: `0.03657107649`.
- Quasilinear flux: `0.1230452010`.
- Nonlinear-window reduced heat flux: `0.08010670290`.

These are admission metrics only. They do not claim an absolute quasilinear
flux predictor or a converged nonlinear turbulent heat-flux reduction.

## Completed Recent Work

- Added and tested `build_wout_reproducibility_gate` and
  `build_authoritative_wout_candidate_gate`.
- Updated VMEC-JAX/SPECTRAX-GK artifact builders so failed rerun reproducibility
  remains fail-closed unless an explicit rerun-WOUT admission gate passes.
- Added downstream support for explicitly authoritative rerun WOUTs in full
  sweep, optimization-status, and candidate-comparison artifacts.
- Rerun-gated the older aspect-5 projected candidates with weights `5e-4` and
  `1e-3`; both fail strict admission because deterministic rerun mean iota is
  about `0.39849`.
- Added `tools/evaluate_vmec_jax_spectrax_transport_metric.py` for eval-only
  SPECTRAX-GK transport metrics from solved VMEC-JAX inputs/WOUTs.
- Added memory-safe surface chunking for reduced metric evaluation and gradient
  diagnostics. This is valid for chunked evaluations, but full reverse-mode
  VMEC-JAX optimization at the 18-point, `mboz=nboz=21` setting still OOMs on
  16 GB GPUs.
- Produced a chunked strict-baseline nonlinear-window gradient on office:
  `/home/rjorge/tmp/spectrax_strict_transport_gradient_bfb55e6/transport_gradient.json`.
- Produced a boundary-chain collection for the strict baseline:
  `/home/rjorge/tmp/spectrax_strict_boundary_chain_top_cpu_bfb55e6/boundary_chain_top2_collection.json`.
  The top-two CPU replay verifies the frozen-axis convention; only the `rc24`
  direction passes growth-branch locality.
- Updated projected line-search tooling to forward strict rerun-WOUT flags and
  use `python3` in replay commands.
- Added coverage tests for candidate gates and projected transport line-search
  edge cases, restoring the wide package coverage gate to 95% in CI.

## Negative Candidate Evidence

### Scalar-Trust One-Point Candidate

Artifact:
`/home/rjorge/tmp/spectrax_strict_rerun_authoritative_transport_iota0p423_onepoint_18157e0`

Result: failed physically and should not be continued.

- Aspect: `1.8249358625`.
- Mean iota: `0.0660699321`.
- QS residual: `5.686562`.
- Transport metric: `0.0250488`.
- Solved gate: failed.
- Rerun admission: failed.

Conclusion: unconstrained scalar-trust transport objectives can reduce the proxy
metric by destroying equilibrium constraints. Future candidate generation must
stay projection/admission gated.

### One-Coefficient Projected Line Search

Forward projected line-search artifact:
`/home/rjorge/tmp/spectrax_strict_rerun_authoritative_projected_line_search_35b55fd`

Reverse projected line-search artifact:
`/home/rjorge/tmp/spectrax_strict_rerun_authoritative_projected_line_search_reverse_35b55fd`

Both used the strict baseline input, strict gradient, top-two boundary-chain
collection, rerun-WOUT gates, and the same 18-point nonlinear-window metric.
All replayed optimizer-state solved gates fail slightly on mean iota, but all
rerun-WOUT admissions pass.

Forward metrics:

- Step `1e-5`: `0.08069043127911753`.
- Step `2.5e-5`: `0.08068612823580769`.
- Step `5e-5`: `0.08067914558393591`.
- Step `1e-4`: `0.08033196895045838`.
- Step `2.5e-4`: `0.08030227976488954`.

Reverse metrics:

- Step `1e-5`: `0.08011064875203953`.
- Step `2.5e-5`: `0.08011658641173851`.
- Step `5e-5`: `0.08012673090579224`.
- Step `1e-4`: `0.08015250260264938`.
- Step `2.5e-4`: `0.08024770409371546`.

Baseline metric: `0.08010670290`.

Conclusion: the one-coefficient projected direction fails closed in both signs.
No long nonlinear audit should be launched from these candidates.

## Immediate Next Steps

1. Monitor the matched long-window nonlinear audit on office:
   `/home/rjorge/tmp/spectrax_strict_matched_nonlinear_audit_top12_edge_0d887d3`.
   It contains strict baseline and top-12 edge candidate `t=700`, `n64`,
   seed/timestep replicate runs with post-transient window `350..700`.
2. After all six nonlinear outputs exist, build baseline and candidate
   replicated-window ensemble gates, then run
   `tools/build_matched_nonlinear_transport_comparison.py` on the two
   ensembles. Promote only if the matched nonlinear comparison is separated
   from uncertainty overlap.
3. If the matched nonlinear audit fails or is uncertainty-overlapped, record the
   top-12 edge candidate as reduced-objective-only evidence and redesign the
   transport metric or variance-reduction campaign before spending more long
   GPU time.
4. Keep CI green after each tranche: fast unit shards, coverage aggregation,
   repository-size gate, docs links, and package build.

## Release Hygiene Rules

- Do not track large transient artifacts, old figures, office scratch outputs,
  or generated demo products. Keep release artifacts small and reproducible.
- Any new tracked figure must be compressed and checked against the repository
  size policy before commit.
- Any promoted nonlinear transport claim must include matched baseline/candidate
  windows, seed or timestep replicates, running-mean convergence, SEM/block
  uncertainty, and an acceptance gate separated from uncertainty overlap.
- Any autodiff or optimization claim must include finite-difference or tangent
  checks and conditioning diagnostics for the differentiated observable.
- Sparse comparison-code mentions are allowed only for validation/benchmarking;
  file names and user-facing examples should remain SPECTRAX-GK-native.

## Running Log

### 2026-06-04

- CI passed on `main` at `9aebb53` after targeted gate and line-search coverage
  additions restored package-wide coverage to 95%.
- Trimmed this active plan from the old public historical running log to the
  current release/science lanes so the repository stays below the 50 MB tracked
  payload limit.
- CI passed again on `main` at `0d887d3` after the plan trim.
- Ran a strict rerun-WOUT boundary-chain campaign on office:
  `/home/rjorge/tmp/spectrax_strict_boundary_chain_top12_cpu_0d887d3`.
  All 12 leading gradient coefficients are finite, frozen-axis convention
  verified, and growth-branch-locality passing; 4 are exact-FD consistent.
- Ran top-6 and top-12 projected line searches under the strict
  rerun-authoritative convention. Top-6 best was `step=5e-4`, metric
  `0.07987162077`, a `0.293%` reduction from the strict baseline
  `0.08010670290`. Top-12 best in the regular sweep was `step=1e-3`, metric
  `0.07941291648`, a `0.866%` reduction; larger `1.5e-3` and `2e-3` failed
  iota admission.
- Ran a top-12 edge scan. `step=1.25e-3` passes rerun-WOUT admission with mean
  iota `0.41001918798`, QS residual `0.01257245066`, and 18-point reduced
  metric `0.07827418221`, a `2.2876%` reduction from baseline. This is a
  reduced-objective admission result only.
- Wrote matched long-window nonlinear audit configs for the strict baseline and
  the top-12 edge candidate under
  `/home/rjorge/tmp/spectrax_strict_matched_nonlinear_audit_top12_edge_0d887d3`.
  Launched six `t=700`, `n64`, post-transient `350..700` runs on office with
  two-way GPU concurrency. The runtime's `10000`-step log entry is the first
  checkpoint chunk; the CLI invocation passes the required manifest step counts
  (`14000` for `dt=0.05`, `17500` for `dt=0.04`).
