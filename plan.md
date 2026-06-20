# SPECTRAX-GK Completion Plan

This file is the active engineering plan for the differentiable-refactor and
research-grade validation branch.  Older chronological logs are intentionally not
kept in this working-tree file to keep the repository light; detailed history is
available from git commits on `codex/differentiable-refactor-plan`.

## Mission

Make SPECTRAX-GK a research-grade gyrokinetic code that is:

- accurate and benchmarked against independent reference calculations where
  relevant;
- end-to-end differentiable from Python for optimization, sensitivity analysis,
  inverse design, and uncertainty quantification;
- performant on CPU/GPU with low memory footprint and clear runtime progress;
- simple to use from the executable and Python APIs;
- simple to maintain, with domain-oriented modules, short auditable functions,
  docstrings/comments where useful, and no legacy compatibility paths that make
  current validated workflows harder to understand;
- documented with equations, algorithms, examples, validation gates, benchmark
  provenance, and publication-ready figures where claims are made.

## Current Branch Policy

- Branch: `codex/differentiable-refactor-plan`.
- Keep one draft PR until this plan is complete; do not split into new PRs.
- Commit and push frequently after focused, gated changes.
- Do not add large outputs or transient artifacts to git.
- Keep the tracked repository below the repository-size manifest cap.
- Use `ssh office` for GPU-backed parity, profiling, and long simulation runs
  only when local evidence is insufficient.
- References to other codes should appear only in benchmark/comparison context;
  source names should describe the physics or numerical method.

## Completion Gates

The plan is not complete until the following evidence exists and passes:

1. **Refactor/testability**
   - Long public functions are split into clear private stages or focused domain
     modules.
   - New structure reduces cognitive load rather than creating many thin files.
   - Public APIs and documented workflows remain stable unless intentionally
     simplified.
   - Focused tests cover each refactored policy boundary.

2. **Differentiability**
   - Python workflows have AD/FD or tangent checks for the differentiated
     observables they expose.
   - VMEC/Boozer bridge paths have geometry parity and gradient gates before any
     optimization claim.
   - Differentiable APIs avoid side effects and non-JAX branches where gradients
     are expected.

3. **Physics validation and benchmarks**
   - Linear, nonlinear, quasilinear, stellarator, tokamak, geometry, and
     transport gates are documented with their scope.
   - Claims distinguish release-level examples, model-development diagnostics,
     and manuscript-grade production evidence.
   - Any comparison with external reference codes is reproducible and scoped.

4. **Performance and memory**
   - Runtime/memory figures are regenerated only from fresh measured artifacts.
   - No new speedup claim is made without profiler-backed evidence and numerical
     identity gates.
   - Parallelization claims remain limited to validated decompositions.

5. **Coverage and CI/CD**
   - Package-wide coverage gate remains at or above 95%.
   - Fast local shards, docs build, package build, lint, type checks, and release
     workflow checks pass before release.
   - Coverage badge/workflow reflects computed package-wide coverage.

6. **Docs/readme/examples**
   - README is concise and points to docs for full derivations and validation.
   - Docs include equations, numerical algorithms, diagnostics, examples,
     validation matrix, and release scope.
   - Examples are runnable, educational, and do not require hidden local files.
   - Publication figures are polished, readable, scoped, and reproducible.

## Active Lane Status

Percentages are engineering estimates, not completion claims.

| Lane | Status | Next Required Evidence |
| --- | ---: | --- |
| Refactor/testability | 99.9% | Explicit linear/nonlinear diagnostic integration, imported geometry loading, VMEC/Boozer core metric/drift assembly, VMEC field-line numerics, nonlinear identity-gate/device-z report builders, independent-work provenance helpers, validation scan runners, and differentiability/objective report hotspots are closed for this checkpoint. |
| Package coverage/release infrastructure | 97% | Confirm latest CI; rerun package-wide coverage shard before release. |
| Runtime/performance infrastructure | 97% | Regenerate panels only from fresh artifacts; profile before speedup claims. |
| Differentiable VMEC/Boozer plumbing | 98% | Keep geometry parity/gradient gates current; broaden only with passed holdouts. |
| Quasilinear model-development | 99% | Keep scoped screening claims; do not promote universal absolute flux without gates. |
| Nonlinear turbulent-flux optimization evidence | 91% | Require long post-transient matched transport windows for production claims. |
| Production nonlinear domain decomposition | 84% | Identity-gated decomposed RHS/integrator/device-z helpers are clearer; CPU/GPU profiling and production speedup evidence still required before claims. |
| Docs/readme/release polish | 95% | Final pass after refactor and performance artifacts settle. |

## Current Refactor Queue

Prioritize behavior-preserving cleanup that makes tests and validation easier.

1. Validation/benchmark scan runners:
   - Closed for this checkpoint; reopen only if a new complexity or testability hotspot appears.
2. Nonlinear transport/optimization reports:
   - Continue only if new hotspots appear after the next scan.
3. Differentiability/objective reports:
   - Closed for this checkpoint; reopen only for new duplicated payload or gate logic.
4. Core numerics/geometry hotspots, only with stronger local gates:
   - `solvers/time/explicit.py` closed for this checkpoint; reopen only if new loop-policy duplication appears.
   - `solvers/nonlinear/diagnostics.py` closed for this checkpoint; reopen only if explicit diagnostic option plumbing grows again.
   - `geometry/flux_tube.py` closed for this checkpoint; reopen only if imported NetCDF schema handling grows again.
   - `geometry/vmec_boozer_core.py` closed for this checkpoint; reopen only if Boozer metric/drift staging grows again.
   - `geometry_backends/vmec_fieldline_numerics.py` closed for this checkpoint; reopen only if VMEC field-line metric/drift or flux-surface-average staging grows again.
5. Parallel/performance hotspots, only with identity gates:
   - `operators/nonlinear/domain_decomposition.py` closed for this checkpoint; reopen only if local domain trace/report policy grows again.
   - `operators/nonlinear/spectral_identity_integrator.py` closed for this checkpoint; reopen only if spectral transport-window trace/report policy grows again.
   - `operators/nonlinear/device_z.py` closed for this checkpoint; reopen only if device-sharding setup, RHS, or transport-window routing policy grows again.
   - `parallel/independent.py` closed for this checkpoint; reopen only if independent-work provenance or ordered-map execution policy grows again.

## Recent Checkpoint

Recent behavior-preserving refactor commits on this branch include:

- this checkpoint: Cyclone scan time-branch fitting now uses explicit run,
  fit, output, and per-batch routing objects; focused Cyclone benchmark branch
  tests and benchmark-scan tests passed locally.
- this checkpoint: kinetic-electron scan batching now uses explicit run, fit,
  output, and per-batch routing objects; focused kinetic scan branch tests and
  benchmark kinetic smoke tests passed locally.
- this checkpoint: ETG scan time-path integration and saved-fit appending now
  use explicit batch/fit contexts with staged streaming, configured-history,
  unconfigured-history, direct-fit, auto-fit, and Krylov-fallback helpers;
  focused ETG scan branch tests and benchmark ETG scan tests passed locally.
- this checkpoint: KBM single-point linear dispatch now carries one explicit
  run-options object through explicit-time, Krylov, and saved-time solver-path
  helpers; focused KBM branch tests and the broader KBM benchmark subset passed
  locally.
- this checkpoint: mode-21 VMEC/Boozer gradient reports now share context,
  observable-vector, sensitivity-gate, and payload-assembly helpers while
  preserving injected test hooks; focused gradient-gate tests passed locally.
- this checkpoint: VMEC/Boozer line-search reports now share scalar/aggregate
  probe builders, common payload assembly, and explicit held-out training/probe
  helpers; focused line-search tests passed locally.
- this checkpoint: reduced QA low-turbulence comparison artifacts now split
  optimized-design generation, per-design diagnostics, gate booleans,
  comparison metrics, and differentiability-plumbing metadata into focused
  helpers; focused QA payload/artifact tests passed locally.
- this checkpoint: explicit linear initial-value integration now stages method
  validation, adaptive CFL timing, JIT stepper construction, sample-history
  collection, progress emission, and array packaging behind the stable
  `integrate_linear_explicit` facade; focused explicit/runtime tests passed
  locally.
- this checkpoint: explicit nonlinear runtime diagnostics now pack the broad
  public signature into one private options object, build state/policy/closure
  components in named stages, and keep scan finalization isolated; focused
  nonlinear runtime-diagnostics tests passed locally.
- this checkpoint: imported flux-tube NetCDF loading now separates schema
  selection, scalar/profile reads, terminal-theta inference, bgrad/drift/Jacobian
  conversion, and `FluxTubeGeometryData` packing; focused imported-geometry
  runtime tests passed locally.
- this checkpoint: VMEC/Boozer equal-arc core assembly now separates
  differential geometry evaluation, raw metric coefficients, raw curvature-drift
  coefficients, equal-arc packing, and final state-to-profile orchestration;
  focused differentiable-geometry Boozer tests passed locally.
- this checkpoint: VMEC field-line numerics now split Hegna-Nakajima
  curvature, normalized metric profiles, magnetic-drift profiles,
  gradient-vector packing, and flux-surface HNGC averaging into focused stages;
  Ruff, mypy, focused VMEC field-line helper tests, and repository/refactor
  manifests passed locally.
- this checkpoint: nonlinear domain and spectral transport-window identity gates
  now separate trace collection, trace-error scoring, fail-closed blockers, and
  report packing while preserving diagnostic-only claim scope; focused nonlinear
  parallel tests passed locally.
- this checkpoint: device-z nonlinear spectral routes now separate sharding
  setup, fail-closed blockers, sharded RHS execution, transport-window sampling,
  and final report packing while preserving single-device fallback behavior;
  focused nonlinear parallel tests passed locally.
- this checkpoint: independent-work parallel execution now separates indexed
  payload collection, reconstruction contracts, identity reports, exception
  provenance, ordered executor routing, and metadata packing while preserving
  ordered-thread/process behavior; focused parallel tests passed locally.
- `53c99703` Refactor stellarator transport prelaunch report.
- `f39eda6f` Refactor nonlinear optimization guard orchestration.
- `726ccdab` Refactor nonlinear replicate spread diagnostics.
- `2885a231` Refactor nonlinear gradient state runbook.
- `f293792d` Refactor nonlinear gradient QL seed screening.
- `fd7e6344` Refactor nonlinear gradient evidence gap report.
- `110544cf` Refactor nonlinear gradient candidate design.
- `658cd0cd` Refactor nonlinear replicate followup planning.
- `6115f94f` Refactor runtime scan orchestration stages.
- `74e5115f` Refactor cached linear integrator stages.
- `ad126cac` Refactor electromagnetic field solve stages.
- `9bc5d905` Refactor VMEC Boozer finite difference gates.

Latest local gates for these tranches included focused pytest shards, Ruff, mypy,
`py_compile`, Sphinx build, differentiable-refactor manifest, repository-size
manifest, and `git diff --check`.

## Verification Cadence

For each focused tranche:

1. Inspect current worktree and branch state.
2. Identify the smallest behavior-preserving boundary that improves readability,
   testability, performance clarity, or claim safety.
3. Refactor with no schema or public-behavior changes unless intentionally
   planned.
4. Run focused tests that cover the touched behavior.
5. Run Ruff, mypy, `py_compile`, `git diff --check`.
6. Run repository-size manifest before commits that touch docs, examples,
   figures, or logs.
7. Run Sphinx when docs or API-doc-visible modules change.
8. Commit and push.
9. Check CI for real failures after it has had time to run; do not waste time
   polling queued jobs.

## Deferred or Scoped Claims

These remain explicit until stronger evidence exists:

- Universal absolute quasilinear flux prediction is scoped as model development
  unless holdout gates prove otherwise.
- Full production nonlinear turbulent-flux stellarator optimization requires
  long post-transient matched baseline-vs-optimized windows across broader
  surfaces/field lines/geometries.
- Nonlinear domain-decomposition speedup requires serial-vs-decomposed identity
  gates and profiler-backed CPU/GPU scaling artifacts.
- W7-X zonal long-window recurrence/damping and W7-X TEM/multi-flux-tube
  extensions are post-release science lanes unless explicitly reprioritized.

## Immediate Next Steps

1. Confirm CI result from the latest push and fix only concrete failures.
2. Continue the refactor queue with the remaining core numerics/geometry and
   identity-gated parallel/performance hotspots.
3. Preserve repository-size margin before adding more docs or figures.
4. Use `office` GPUs for the next simulation/performance lane that needs them.
