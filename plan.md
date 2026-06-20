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
| Package coverage/release infrastructure | 98.5% | Latest local technical-release gates and targeted wide-coverage shards pass; tracked runtime-summary logs are pruned to compact provenance; confirm the queued CI run and rerun/verify the full package-wide coverage combine before release. |
| Runtime/performance infrastructure | 97.5% | Regenerate panels only from fresh artifacts; profiler logs remain available in ignored local log roots while tracked summaries keep compact digests; profile before speedup claims. |
| Differentiable VMEC/Boozer plumbing | 98% | Keep geometry parity/gradient gates current; broaden only with passed holdouts. |
| Quasilinear model-development | 99% | Keep scoped screening claims; do not promote universal absolute flux without gates. |
| Nonlinear turbulent-flux optimization evidence | 91% | Require long post-transient matched transport windows for production claims. |
| Production nonlinear domain decomposition | 88% | Identity-gated decomposed RHS/integrator/device-z helpers are clearer; refreshed CPU and two-GPU transport-window profiling is identity-clean, including a longer two-GPU window after the compute-route fix, but the GPU route remains just below the speedup gate and end-to-end production speedup evidence is still required before claims. |
| Docs/readme/release polish | 97% | Release guardrails and docs status artifacts are current; tracked docs evidence is slimmer; final pass after CI and any remaining refactor/performance artifacts settle. |

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

- this checkpoint: Cyclone reference-aligned explicit-time scan now separates
  per-ky point preparation, explicit trace fitting, branch-reselection
  predicates, Krylov reselection fallback, and previous-frequency continuation
  updates; focused Cyclone scan branch tests, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: kinetic-electron ky-scan orchestration now separates
  setup/control resolution, growth-window policy construction, runtime option
  packing, batch execution, and public result assembly from the scan wrapper;
  focused kinetic scan branch tests, mypy, architecture, repository-size, and
  diff hygiene passed locally.
- this checkpoint: Cyclone time-scan branches now separate shared run-control
  packing, history-fit policy packing, batch execution, and public result
  assembly from the signature-heavy scan entry point; focused Cyclone branch
  tests, mypy, architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: TEM single-run time integration now separates fit-signal
  validation, sampling/time-config resolution, integration routing,
  signal extraction, window fitting, and public result packing; focused TEM
  branch tests, mypy, architecture, repository-size, and diff hygiene passed
  locally.
- this checkpoint: Cyclone single-mode time integration now separates resolved
  time/fit controls and the saved-trace branch from the public entry point while
  preserving the reference-aligned explicit branch; focused Cyclone branch tests,
  mypy, architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: full linear runtime dispatch now separates Krylov
  eigenvector finalization and auto-solver fallback acceptance from the public
  workflow entry point; focused linear runtime integration tests, mypy,
  architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: bounded Boozer spectral sensitivity reporting now separates
  fail-closed availability payloads, demo Boozer input construction, spectral
  objective evaluation, derivative computation, and success payload assembly;
  focused differentiable-geometry bridge tests, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: nonlinear turbulence-gradient conditioning now separates
  schema-variant metric extraction, production gate-row construction, and
  public payload packing; the full nonlinear-gradient evidence test file, mypy,
  architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: VMEC-JAX transport-gradient diagnostics now separate active
  parameter resolution, optimizer residual/objective/gradient evaluation,
  sensitivity classification, base report packing, and optional Jacobian
  diagnostics; focused VMEC transport-gradient tests, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: implicit eigenvalue branch-locality diagnostics now separate
  tolerance validation, spectrum loading, per-side branch selection,
  dominant-vs-nearest slope comparison, classification, and payload packing;
  focused solver-objective gradient tests, mypy, architecture, repository-size,
  and diff hygiene passed locally.
- this checkpoint: fixed-beta KBM ky-scan orchestration now separates fixed-beta
  case resolution, scan option packing, per-ky beta-scan dispatch, mutable row
  accumulation, and public result assembly; focused KBM scan branch tests, mypy,
  architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: quasilinear calibration-window ingestion now separates
  replicated-ensemble admission, CSV/NetCDF heat-flux extraction, convergence
  reporting, note construction, and calibration-point packing; focused
  quasilinear calibration tests, mypy, architecture, repository-size, and diff
  hygiene passed locally.
- this checkpoint: quasilinear model-selection status construction now
  separates artifact loading, candidate/claim-boundary context creation,
  gate-row assembly, metric projection, and scoped-claim payload packing; the
  full quasilinear model-selection/guardrail test files, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: the no-input executable demo now separates bundled-source
  resolution, run-setting normalization, terminal progress/introduction,
  solver dispatch, signal/eigenfunction extraction, plot writing, artifact
  writing, and result reporting; the full CLI test file, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: linear runtime artifact writing now separates scan/runtime
  path targets, summary construction, quasilinear spectrum column projection,
  timeseries/eigenfunction/state writers, and quasilinear bundle merging while
  preserving saved-output schemas; the combined runtime artifact and CLI test
  files, mypy, architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: the internal VMEC-to-imported-geometry pipeline now
  separates request validation, theta-grid construction, flux-tube cut policy,
  beta-prime evaluation, field-line solve dispatch, profile packing, and
  atomic NetCDF emission; the full VMEC backend/eik test pair, mypy,
  architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: matched nonlinear optimized-transport reporting now
  separates schema variant extraction, reduction/uncertainty metric resolution,
  baseline/candidate/selection qualification flags, and blocker construction;
  the full nonlinear transport optimization test file, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: diagnostic sharded nonlinear integration now separates the
  explicit Euler/RK2/RK3/RK4/SSPX3 update table from sharding/projector setup
  and scan wiring; the full sharded-integrator test file, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: fixed-ky KBM beta-scan orchestration now separates scan
  option packing, per-beta sample dispatch, Krylov continuation updates,
  mutable beta/growth/frequency accumulation, and public result assembly;
  focused KBM branch tests, selected marker-overridden KBM benchmark tests,
  mypy, architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: nonlinear turbulence-gradient composite-control reporting
  now separates config validation, candidate metadata normalization, row
  construction, descent-control scaling, and launch-plan text construction; the
  full nonlinear-gradient follow-up test file, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: differentiated objective-portfolio sensitivity reporting
  now separates portfolio weight packing, row/scalar objective function
  construction, contract validation, scalar/row AD-FD gates, and final
  conditioning/covariance payload assembly; focused stellarator/portfolio
  sensitivity tests, mypy, architecture, repository-size, and diff hygiene
  passed locally.
- this checkpoint: observable-gradient AD/FD validation now separates
  conditioning-array validation, SVD/norm statistics, worst-entry extraction,
  finite-difference step metadata, derivative construction, gate evaluation,
  and report assembly. The tangent AD/FD acceptance gate now uses the same
  entrywise absolute-or-relative tolerance semantics as the Jacobian gate,
  fixing a zero-reference tangent regression; differentiable-geometry and
  full differentiable-geometry-bridge shards, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: differentiable solver-objective evaluation now shares one
  solver-geometry context builder and explicit operator-matrix route across
  matrix, growth-rate, and full objective-vector evaluators. Dominant-branch
  extraction and linear/quasilinear transport-weight computation are separate
  helpers, reducing duplication in the stellarator optimization objective path;
  the full solver-objective-gradient test file, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: VMEC-JAX flux-tube array parity reporting now separates
  validated parity options, backend availability checks, direct/imported
  geometry loading, production metric evaluation, equal-arc parity evaluation,
  exception payloads, and final report packing while preserving the public
  parity schema and thresholds; focused flux-tube parity tests, the full
  differentiable-geometry-bridge shard, mypy, architecture, repository-size,
  and diff hygiene passed locally.
- this checkpoint: nonlinear NetCDF artifact writing now separates bundle path
  resolution, required diagnostics validation, output-grid/geometry layout,
  dimension creation, grid-variable writing, primary ``.out.nc`` writing, and
  optional restart/``.big.nc`` writing while preserving the saved-output schema;
  direct nonlinear NetCDF artifact tests, saved-output plotting tests, mypy,
  architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: Cyclone Krylov ky-scan branch-following now separates
  per-ky point preparation, explicit/reduced seed resolution, dominant
  eigenpair solve dispatch, raw continuation-state updates, and normalized
  output writing; focused Cyclone scan branch tests, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: VMEC/Boozer mode-21 linear context construction now
  separates backend/example loading, VMEC state-parameter resolution,
  compact linear grid/state-shape setup, geometry closure creation, and
  solver/cache/matrix closure construction; focused solver-gradient module
  tests, mypy, architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: reduced zonal-flow record normalization now separates
  per-record metric parsing, optional damping/recurrence accounting, axis
  construction, duplicate-aware tensor fill, and finite tensor completeness
  checks; focused zonal objective tests, mypy, architecture, repository-size,
  and diff hygiene passed locally.
- this checkpoint: reduced zonal-flow objective artifact assembly now separates
  normalized record tensor packing, objective row/reduced-value evaluation,
  promotion-claim metadata, axes/metric serialization, and final JSON payload
  construction while preserving the public artifact schema; focused zonal
  objective tests, mypy, architecture, repository-size, and diff hygiene passed
  locally.
- this checkpoint: zonal-response plotting now separates trace validation,
  metric extraction, branchwise extrema overlays, normalized-response rendering,
  envelope-fit rendering, annotation text, and axis styling while preserving the
  public ``zonal_flow_response_figure`` contract; the full plotting test file,
  mypy, architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: VMEC/Boozer transport-candidate admission now separates
  candidate annotation, baseline selection/state construction, metric
  improvement gates, admitted-candidate collection, promotion selection, and
  report payload assembly while preserving the existing long-window audit
  policy; the full VMEC-JAX transport-admission test file, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: KBM single-point linear paths now separate explicit
  integrator policy construction, projected-rate fallbacks, direct trace
  fitting, Krylov target eigenpair evaluation, multi-target branch selection,
  and final result packaging; focused KBM branch tests, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: Cyclone scan public orchestration now builds one execution
  options object and delegates Krylov, reference-explicit, and saved-time
  solver-path selection to a focused dispatcher; focused Cyclone scan branch
  tests, broader Cyclone benchmark-runner tests, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: TEM scan batching now separates runtime scan options,
  scalar/fixed-width ky iteration, branch routing, and mutable output
  accumulation from the public scan wrapper; focused TEM scan tests, broader
  TEM benchmark-runner tests, mypy, architecture, repository-size, and diff
  hygiene passed locally.
- this checkpoint: ETG scan orchestration now separates runtime scan options
  and mutable batch accumulation from the public scan wrapper while preserving
  Krylov continuation state; focused ETG runner branch tests, benchmark runner
  branch tests, mypy, architecture, repository-size, and diff hygiene passed
  locally.
- this checkpoint: VMEC/Boozer boundary-chain summary classification now
  separates scalar metric collection, required-finite detection, nonfinite
  fail-closed payloads, and finite classification payloads; the full boundary
  chain test file, mypy, architecture, repository-size, and diff hygiene passed
  locally.
- this checkpoint: reduced stellarator ITG optimization now separates initial
  parameter validation, Adam state/update, optimization history, objective
  AD/FD gate, residual covariance metadata, and nonlinear-window trace payloads;
  the full stellarator optimization test file, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: geometry inverse-design reporting now separates input
  validation, selected-observable construction, and Gauss-Newton iteration from
  the AD/FD/UQ payload assembly; focused differentiable-geometry tests, mypy,
  architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: quasilinear transport diagnostics now separate option
  validation, amplitude-normalized linear flux weights, species labels,
  saturation-rule application, and claim-scope metadata; focused quasilinear
  tests, the AD derivative gate, mypy, architecture, repository-size, and diff
  hygiene passed locally.
- this checkpoint: VMEC flux-tube cut/remap code now separates field-line
  sample extraction, interpolation, crossing selection, aspect-cut root solving,
  and public cut orchestration; full VMEC backend helper tests, mypy,
  architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: external-VMEC holdout runbook generation now separates
  gap-report context extraction, candidate ranking, launch-command construction,
  and fail-closed acceptance-gate policy; focused external-holdout/runbook tests,
  mypy, architecture, repository-size, and diff hygiene passed locally.
- this checkpoint: Cyclone single-mode time-path routing now uses an explicit
  saved-trace object plus separate fit-policy construction and integrator
  routing helpers; focused Cyclone linear time-path tests, mypy, architecture,
  repository-size, and diff hygiene passed locally.
- this checkpoint: Cyclone scan time-branch fitting now uses explicit run,
  fit, output, and per-batch routing objects; focused Cyclone benchmark branch
  tests and benchmark-scan tests passed locally.
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
- this checkpoint: post-refactor device-z transport-window profiling refreshed
  the logical-CPU artifact on the ``(4,16,96,96,32)`` workload; serial-vs-sharded
  identity passed for final state, free energy, field energy, physical flux, and
  bracket RMS, and the fixed-window micro-route reached ``1.61x`` on two CPU
  devices and ``3.13x`` on four while full-solver production speedup remains
  blocked until GPU/end-to-end gates pass.
- this checkpoint: the matching two-GPU device-z transport-window profile was
  rerun on ``office`` after the compute-route identity fix; identity passed
  with maximum final-state absolute error ``7.45e-9`` and speedup improved to
  ``1.48x``, but the route remains below the configured ``1.5x`` production
  speedup gate.
- this checkpoint: follow-up two-GPU exploratory profiles on ``office`` showed
  that simply enlarging the diagnostic workload is not sufficient for a
  production nonlinear speedup claim: ``(4,16,96,96,64)`` reached only
  ``1.41x`` and ``(4,16,128,128,32)`` reached only ``1.29x``. A separate
  16-step ``(4,16,96,96,32)`` run exposed an instrumentation-path final-state
  mismatch, which is now fixed by comparing final states on the compute-only
  jitted route used by the profiler while keeping scalar traces on the
  instrumented path.
- this checkpoint: local release-hygiene gates passed for release artifacts,
  package architecture, release readiness, validation/coverage manifest,
  technical release status, parallelization completion status, quasilinear
  guardrails, and VMEC/Boozer differentiability claims. Bounded wide-coverage
  shards 9 and 10 also passed under the documented 48-shard workflow; full
  package-wide coverage remains a CI/combine gate before release.
- this checkpoint: release-scope tests now guard core source modules against
  comparison-code terminology outside the validation benchmark package, keeping
  benchmark/reference-code names from leaking back into the main solver,
  geometry, objective, runtime, and artifact APIs. The guard is also routed
  through the fast release-artifacts CI shard so regressions are caught before
  the full wide-coverage combine.
- this checkpoint: runtime/memory benchmark summaries now store compact
  stdout/stderr byte counts and SHA-256 digests instead of embedding full
  process logs in tracked JSON artifacts. The tracked release summary shrank
  from about ``987 kB`` to about ``58 kB`` while preserving the numeric rows
  needed to regenerate the runtime/memory panel; full logs remain in ignored
  ``tools_out/runtime_memory_logs`` during local benchmark execution.
- this checkpoint: cETG reduced nonlinear runtime orchestration now separates
  setup, adaptive chunked execution, and fixed-step execution behind the stable
  public workflow function. The public nonlinear cETG entry point shrank from
  the largest active workflow hotspot to a small dispatcher while preserving
  the same integrator, dependency injection, progress messages, and runtime
  result schema.
- this checkpoint: cETG reduced linear runtime orchestration now mirrors that
  staged structure with explicit setup, time-series integration, growth/frequency
  fitting, and result-packing helpers. The public linear cETG workflow remains
  API-compatible but is no longer responsible for grid construction, coefficient
  construction, integration, fitting, and output assembly in one block.
- this checkpoint: runtime linear diagnostic fitting now separates input
  normalization, automatic phi/density candidate scoring, explicit-signal
  fitting, eigenfunction extraction, and result packing. A focused regression
  test covers automatic selection of the higher-scored density channel so the
  executable fit-signal policy remains stable after the split.
- this checkpoint: nonlinear adaptive-CFL frequency evaluation now separates
  rFFT-compressed gradients, full-complex gradients, electromagnetic Apar/Bpar
  speed accumulation, and final x/y frequency reduction. The public helper keeps
  the same CPU/GPU batching policy and is covered by zero/finite and spectral
  gradient CFL tests.
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
- this checkpoint: linear dissipation assembly now separates species collision
  frequency validation, Laguerre collision-base construction, moment-restoring
  collision corrections, constant hypercollision damping, parallel
  hypercollision source construction, and linked/nonlinked ``|k_z|`` application.
  The linear RHS profiler now reuses the production hypercollision source helper
  and guards linked-only profiling inputs explicitly, reducing profiler/solver
  drift while preserving the tested formulas.

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
