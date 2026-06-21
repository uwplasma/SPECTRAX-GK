# SPECTRAX-GK Completion Plan

This file is the active engineering plan after the `v1.6.7` differentiable
architecture/refactor release.  Older chronological logs are intentionally not
kept in this working-tree file to keep the repository light; detailed history is
available from git commits and release notes.

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

- Branch: `main` after the merged `v1.6.7` refactor release.
- Use focused commits for post-release tranches; create a branch/PR only when
  the tranche is large enough that review isolation is useful.
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
| Refactor/testability | 99.9% | Core numerics, diagnostics, geometry, validation, objective, and parallel hotspots touched in the `v1.6.7` release are closed for that checkpoint. Remaining large functions are mostly validation/report/artifact orchestration; handle them only when they improve developer usability, tested policy boundaries, or release evidence. |
| Package coverage/release infrastructure | 100% for `v1.6.7` | PR CI, post-merge CI, release workflow, GitHub release, and PyPI publish passed for `v1.6.7`. Keep the gate active for subsequent commits. |
| Runtime/performance infrastructure | 97.5% | Current release claims are scoped to tracked runtime/memory and profiler artifacts. No additional speedup claim should be added without fresh identity-gated profiler evidence. |
| Differentiable VMEC/Boozer plumbing | 98% | Keep geometry parity/gradient gates current; broaden only with passed holdouts. |
| Quasilinear model-development | 99% | Keep scoped screening claims; do not promote universal absolute flux without gates. |
| Nonlinear turbulent-flux optimization evidence | 91% | Require long post-transient matched transport windows for production claims. |
| Production nonlinear domain decomposition | 88% | Identity-gated decomposed RHS/integrator/device-z helpers are clearer; refreshed CPU and two-GPU transport-window profiling is identity-clean, including a longer two-GPU window after the compute-route fix, but the GPU route remains just below the speedup gate and end-to-end production speedup evidence is still required before claims. |
| Docs/readme/release polish | 100% for `v1.6.7` | Release guardrails and docs status artifacts are current for the shipped release. Future docs changes must preserve scoped claims and reproducible figure provenance. |

## Current Refactor Queue

Prioritize behavior-preserving cleanup that makes tests and validation easier.

### 2026-06-20 Refactor and Release Audit

The current package has a domain-oriented structure and no root-prefix modules
left under the architecture manifest, but it is still a large scientific code:
roughly 359 package Python files and 103k source lines.  That size is now mostly
from validation/reporting breadth rather than unresolved core-runtime
spaghetti.  The most important remaining large-file clusters are:

- validation benchmark orchestration (`validation/benchmarks/*`), especially
  Cyclone, ETG, KBM, kinetic-electron, and TEM scan drivers;
- VMEC/Boozer geometry reports and backend numerics
  (`geometry/vmec_boozer_core.py`, `geometry_backends/vmec_fieldline_numerics.py`,
  `geometry/vmec_flux_tube_reports.py`);
- linear-cache construction and nonlinear term assembly
  (`operators/linear/cache_builder.py`, `terms/nonlinear.py`);
- workflow/report orchestration (`workflows/reduced_models.py`,
  `workflows/runtime/*`, nonlinear IMEX diagnostics, and validation reports).

Release decision now closed: `v1.6.7` shipped with architecture,
repository-size, technical-release, release-readiness, package-build, focused
release-scope, full PR CI, post-merge CI, and PyPI/GitHub release gates passing.
A broad pre-release collapse was intentionally deferred because the remaining
hotspots are mostly validation/report drivers rather than release-blocking core
runtime issues.

Efficient next refactor after `v1.6.7`:

1. Keep the domain packages but stop creating new one-off modules unless they
   become shared extension points.
2. Consolidate single-use validation/report helpers back into their nearest
   domain package when that reduces navigation cost.
3. Split only functions above roughly 100 lines when the split exposes a tested
   policy boundary, removes duplicated logic, or makes a differentiable path
   side-effect-free.
4. Prefer short public facades plus private helpers over compatibility aliases
   or legacy paths.
5. Rename remaining comparison-code terminology only where it is not explicitly
   a benchmark/comparison artifact.
6. Preserve profiler and numerical-identity gates before changing nonlinear
   RHS, field solve, or geometry kernels.

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

- this checkpoint: the public Cyclone scan wrapper now routes through a typed
  request object and small fit-policy/setup/execution-option stages, preserving
  the public API while removing duplicated scan-control construction from the
  wrapper; targeted Cyclone scan branch/entrypoint tests, mypy, Ruff,
  architecture, repository-size, and differentiable-refactor manifests passed
  locally.
- this checkpoint: Cyclone time-scan branch orchestration now routes public
  scan arguments through a typed private input bundle, removing duplicated
  keyword forwarding between the public wrapper and shared run/fit controls;
  the full benchmark runner branch test file, mypy, Ruff, architecture,
  repository-size, and differentiable-refactor manifests passed locally.
- this checkpoint: zonal-flow response metrics now separate normalized-window
  construction from peak/envelope frequency fitting, leaving residual,
  Rosenbluth-Hinton/GAM normalization, branchwise damping, and invalid-input
  behavior unchanged; focused benchmarking/zonal validation tests, mypy, Ruff,
  architecture, repository-size, and differentiable-refactor manifests passed
  locally.
- this checkpoint: quasilinear calibration reports now separate report-control
  validation, calibration-point normalization, physical value checks,
  optional train-scale application, split metrics, claim-level selection, and
  public payload assembly; the full quasilinear calibration test file, mypy,
  Ruff, quasilinear-promotion guardrails, architecture, repository-size, and
  differentiable-refactor manifests passed locally.
- this checkpoint: VMEC field-line metric assembly now separates coordinate
  gradient construction, HNGC field-line integrals, shear/pressure correction
  factors, local-shear assembly, and metric/drift packing; the full VMEC
  geometry helper test file, mypy, Ruff, architecture, repository-size, and
  differentiable-refactor manifests passed locally.
- this checkpoint: VMEC/Boozer aggregate line-search holdout reports now
  separate injected report dependencies, held-out improvement validation,
  train/holdout report execution, and public payload assembly from the public
  report function; focused aggregate holdout tests, mypy, Ruff, architecture,
  repository-size, and differentiable-refactor manifests passed locally.
- this checkpoint: VMEC/Boozer scalar objective finite-difference reports now
  separate injected dependency resolution, scalar point evaluation, three-point
  FD triplet construction, diagnostic calculation, and public payload assembly
  from the report function; focused solver-gradient module tests, focused
  scalar FD tests, mypy, Ruff, architecture, repository-size, and
  differentiable-refactor manifests passed locally.
- this checkpoint: VMEC/Boozer mode-21 quasilinear and reduced nonlinear-window
  gradient gates now share an explicit enriched linear context helper, and the
  nonlinear-window gate separates observable construction, objective-pass
  classification, and public config payload assembly from the report function;
  focused solver-gradient gate tests, focused mypy, Ruff, architecture,
  repository-size, and differentiable-refactor manifests passed locally.
- this checkpoint: nonlinear-gradient composite-control candidate rows now
  separate canonical metric extraction, per-condition gate evaluation, blocker
  construction, and JSON metric payload assembly from the row constructor; the
  full nonlinear-gradient follow-up test file, focused mypy, Ruff,
  architecture, repository-size, and differentiable-refactor manifests passed
  locally.
- this checkpoint: VMEC/Boozer reduced objective-portfolio artifact guards now
  separate canonical artifact input extraction, portfolio coverage checks,
  full objective-table checks, scalar reducer validation, and final promotion
  gate assembly from the public report function; the dedicated portfolio guard
  and stellarator objective portfolio tests, focused mypy, Ruff, architecture,
  repository-size, and differentiable-refactor manifests passed locally.
- this checkpoint: nonlinear-gradient replicated-window evidence now separates
  input artifact row classification, unsupported-artifact handling,
  convergence-window row construction, derived ensemble construction,
  qualifying-row selection, and gate assembly from the public summary function;
  the full nonlinear-gradient evidence test file, focused mypy, Ruff,
  architecture, repository-size, and differentiable-refactor manifests passed
  locally.
- this checkpoint: nonlinear runtime artifact handoff now separates run-option
  packing, one-chunk execution/finite validation, optional artifact writing,
  checkpoint-advance policy, and the checkpoint loop from the public handoff
  entry point; the full runtime artifact/helper tests, focused mypy, Ruff,
  architecture, repository-size, and differentiable-refactor manifests passed
  locally.
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
- this checkpoint: linked streaming now separates Hermite ladder construction,
  electrostatic/electromagnetic field-drive injection, safe inverse-temperature
  handling, and periodic-vs-linked parallel derivative application. The public
  streaming contribution keeps the same schema and numerical route, while the
  full comparison shard and disabled-field assembly regressions passed locally.
- this checkpoint: diamagnetic linear drive assembly now separates Laguerre
  gradient profiles, scalar drift-frequency factors, electrostatic/magnetic
  compression drives, and Apar temperature-profile drives. The comparison
  regression shard and the production-vs-sharded diamagnetic drive identity
  check passed locally, with the multi-device logical-device case skipped on
  this single-device backend as expected.
- this checkpoint: the nonlinear scan integrator now separates method
  validation, projection, RHS casting, optional progress callbacks, and each
  explicit update family (Euler, RK2, RK3/SSP-like variants, RK4, and K10)
  into traceable helpers. The public jitted scan API and field-history schema
  are unchanged; method-amplification, observed-order, projection, progress,
  nonlinear smoke, architecture, repository-size, and diff-hygiene gates passed
  locally.
- this checkpoint: explicit adaptive-CFL frequency bounds now separate spectral
  grid bounds, species/gradient scales, effective geometry/non-twist scales,
  radial drift, binormal/diamagnetic drift, electron pressure, and parallel
  streaming/Alfven guards. The public ``_linear_frequency_bound`` policy and
  return schema are unchanged; static checks, selected-ky CFL preservation,
  nonlinear CFL component gates, integration-marked adaptive-dt caps,
  architecture, repository-size, and diff-hygiene gates passed locally.
- this checkpoint: opt-in parallel linear RHS dispatch now separates route
  normalization, serial aliasing, auto-backend resolution, Hermite-axis
  admission checks, and each gated velocity backend (streaming-only,
  electrostatic streaming, and electrostatic slices). The public dispatch
  semantics and error messages are preserved; focused dispatch tests, helper
  branch tests, serial-vs-parallel identity gates, architecture, repository-size,
  and diff-hygiene gates passed locally.
- this checkpoint: linked-boundary linear-cache construction now separates kx
  link maps, linked FFT/gather metadata, linked end-damping profile construction,
  and final dictionary packing. Twist-shift and periodic/no-twist behavior are
  unchanged; selected-ky linked FFT, periodic zero-shear, linked non-twist cache,
  geometry-grid twist-shift, static, architecture, repository-size, and
  diff-hygiene gates passed locally.
- this checkpoint: linked FFT map construction now separates active-mode
  selection, left/right neighbor-map construction, directional chain counting,
  chain-length grouping, full-grid kx index restoration, linked chain index
  packing, and linked kz construction. The public linked-map schema is
  unchanged; linked-map unit tests, linked FFT operator identity/gather tests,
  integration-marked selected-ky and one-link derivative tests, linked-cache
  contracts, static checks, architecture, repository-size, and diff-hygiene
  gates passed locally.
- this checkpoint: log-linear growth-window selection now separates immutable
  option validation, search-state preparation, candidate scoring, fallback
  window selection, and positive-growth retry policy. The public
  ``select_fit_window_loglinear`` API and fit-window semantics are unchanged;
  focused log-linear validation/fallback tests, the full analysis diagnostic
  shard, static checks, architecture, repository-size, and diff-hygiene gates
  passed locally.
- this checkpoint: shift-invert Krylov mode extraction now separates GMRES
  operator/preconditioner construction, transformed Ritz-spectrum recovery,
  frequency/sign masks, nearest-shift fallback selection, growth/target/overlap
  selection, and per-restart vector reconstruction. The jitted
  ``dominant_eigenpair_shift_invert_cached`` signature and numerical policy are
  unchanged; Krylov core tests, the integration-marked Hermite-line
  shift-invert run, benchmark shift-policy helper tests, static checks,
  architecture, repository-size, and diff-hygiene gates passed locally.
- this checkpoint: plain Arnoldi and IMEX-propagator Krylov extraction now share
  frequency/sign mask construction, target/overlap mode selection, and Ritz
  vector reconstruction helpers while preserving their distinct fallback
  policies. The public jitted solver signatures are unchanged; the full Krylov
  core shard, integration-marked shift-invert smoke run, static checks,
  architecture, repository-size, and diff-hygiene gates passed locally.
- this checkpoint: fixed-step IMEX nonlinear diagnostic integration now uses
  explicit preparation, runtime, diagnostic, and scan option bundles around the
  compatibility entry point. The public
  ``integrate_imex_nonlinear_diagnostics_impl`` signature and scan behavior are
  unchanged; nonlinear IMEX unit tests, nonlinear package-export tests, static
  checks, architecture, repository-size, and diff-hygiene gates passed locally.
- this checkpoint: standalone Miller weighted finite differences now separate
  centered nonuniform-grid stencils, one-sided odd-parity endpoints, 1D
  theta/radial contracts, and 2D theta/radial contracts. The legacy public
  ``dermv`` shape and endpoint behavior are unchanged; Miller low-level tests,
  standalone Miller generator tests, internal Miller backend tests, runtime
  Miller config/helper tests, static checks, architecture, repository-size, and
  diff-hygiene gates passed locally.
- this checkpoint: streaming Diffrax linear integration now separates fit-window
  and monitored-mode options, solve/JIT/sharding options, packed-state
  preparation, Diffrax term construction, solve execution, and output
  finalization. The public ``integrate_linear_diffrax_streaming`` signature and
  streamed growth/frequency semantics are unchanged; focused Diffrax streaming
  tests, benchmark streaming-hook tests, static checks, architecture,
  repository-size, and diff-hygiene gates passed locally.
- this checkpoint: VMEC-state differentiability reports now separate Boozer
  context loading, VMEC-to-Boozer mapping closure construction, Boozer
  flux-tube report packing, and VMEC field-line tensor payload construction.
  Public optional-backend report APIs and fail-closed behavior are unchanged;
  VMEC/Boozer optional-backend report tests, facade and refactor-manifest tests,
  AD/FD helper tests, static checks, architecture, repository-size, and
  diff-hygiene gates passed locally.
- this checkpoint: Boozer-transform differentiable bridge helpers now separate
  Boozer output preparation, field-line sampling, smooth metric/drift closure
  construction, flux-tube mapping packing, two-parameter demo input
  construction, fail-closed sensitivity reports, and successful report packing.
  The public Boozer mapping and AD/FD sensitivity report contracts are
  unchanged; focused Boozer bridge tests, facade tests, static checks,
  architecture, repository-size, and diff-hygiene gates passed locally.
- this checkpoint: linear hypercollision contribution assembly now separates
  inactive-operator early exits, static ``k_z`` branch skips, and the
  linked-field-line parallel hypercollision contribution from the public
  physics API. The formula, linked FFT routing, and zero-operator behavior are
  unchanged; direct hypercollision formula/skip/link tests, profiling helper
  tests, static checks, architecture, repository-size, and diff-hygiene gates
  passed locally.
- this checkpoint: fixed-step linear integration dispatch now separates
  sampling validation, cache construction, public method alias normalization,
  implicit solver routing, non-serial explicit routing, and serial/donated
  explicit routing. Public ``integrate_linear`` behavior is unchanged;
  non-integration wrapper/cached-implementation routing tests, static checks,
  architecture, repository-size, and diff-hygiene gates passed locally. The
  small-grid ``tests/test_linear.py`` nodes remain integration-marked and are
  not part of the bounded default local shard.
- this checkpoint: implicit linear integration now separates fixed-point GMRES
  warm-start construction, one-step GMRES solve routing, and field diagnostic
  evaluation from the scan driver. Preconditioner construction, solve
  tolerances, sampled output semantics, and public behavior are unchanged;
  implicit preconditioner tests, sampled implicit-step tests, static checks,
  architecture, repository-size, and diff-hygiene gates passed locally.
- this checkpoint: nonlinear electromagnetic bracket entry points now share a
  prepared-path helper for state normalization, dealiased field preparation,
  Laguerre-grid availability, and electrostatic-vs-electromagnetic routing.
  Public contribution/component APIs and spectral/Laguerre semantics are
  unchanged; the full nonlinear bracket shard, static checks, architecture,
  repository-size, and diff-hygiene gates passed locally.
- this checkpoint: linked-field-line FFT operators now separate input
  validation, flattened linked-chain state preparation, per-chain FFT updates,
  covered-row discovery, gather/full-cover/scatter reconstruction paths, and
  real-FFT conjugate restoration. Numerical routing for linked derivatives and
  ``|k_z|`` hypercollision remains unchanged; linked-operator tests, linked
  hypercollision regressions, static checks, architecture, repository-size, and
  diff-hygiene gates passed locally.
- this checkpoint: velocity-space parallel field and diamagnetic-drive helpers
  now share single-species state/plan validation, Laguerre-weight normalization,
  and Hermite-shard mesh/spec construction. Reference fallbacks, logical-device
  shard-map behavior, and error policy are unchanged; the full velocity-sharding
  shard, static checks, architecture, repository-size, and diff-hygiene gates
  passed locally.
- this checkpoint: whole-state nonlinear sharded integration now separates
  state/dt/method setup, Hermitian projection and sharding constraints, nonlinear
  RHS closure construction, explicit scan-step construction, device placement,
  and final-only vs field-history dispatch. Public diagnostic sharded-integrator
  behavior is unchanged; sharded integrator tests, static checks, architecture,
  repository-size, and diff-hygiene gates passed locally.
- this checkpoint: cached nonlinear IMEX scans now separate implicit-operator
  resolution, state rank/dtype normalization, nonlinear-term and GMRES step
  construction, checkpointed scan dispatch, and single-species output squeezing.
  Public cached-IMEX behavior is unchanged; focused IMEX tests, cached nonlinear
  forwarding tests, static checks, and local repository gates passed.
- this checkpoint: linear explicit stepping now separates the reusable staged
  RHS closure, RK3/RK4/SSPX3/K10 formulas, method dispatch, post-step spectral
  masking, and final field solve. Public explicit-step behavior is unchanged;
  scalar amplification, mask, static, and repository gates passed locally.
- this checkpoint: nonlinear explicit stepping now separates RK/SSP/K10 state
  updates, adaptive diagnostic time-step advancement, optional collision split,
  diagnostic selection, and progress emission. Public explicit nonlinear
  behavior is unchanged; explicit-step tests, package export tests, static
  checks, and local repository gates passed.
- this checkpoint: turbulent-heating diagnostics now separate species-axis
  normalization, safe time increments, masked field derivatives, moment
  reconstruction, and per-species heating assembly. Diagnostic formulas are
  unchanged; zero-field, steady-state, resolved-sum, zero-dt, static, and local
  repository gates passed.
- this checkpoint: linear cache k_perp/drift construction now separates NTFT
  metric assembly, standard geometry assembly, and dealias masking. Cache output
  policy is unchanged; current cache helper tests, static checks, collection
  audit for non-collected legacy linear tests, and local repository gates passed.

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

## Actual Open Lanes

Release-blocking technical lanes for `v1.6.7` are closed.  Remaining lanes are
post-release research/performance/completeness lanes:

- Universal absolute quasilinear heat-flux prediction remains explicitly not
  promoted; current release scope is model-development and screening guardrails.
- Full production nonlinear turbulent-flux stellarator optimization still needs
  broader long post-transient matched windows across surfaces/field lines.
- Production nonlinear domain-decomposition speedup still needs an end-to-end
  GPU speedup gate; current artifacts support diagnostic/identity claims only.
- W7-X zonal long-window recurrence/damping and W7-X TEM/multi-flux-tube
  extensions remain post-release science lanes.

## Post-Release Refactor Audit

The current refactor target is not "more files".  The package already has the
domain layout needed for maintainability; the next release should reduce
navigation cost and stabilize extension boundaries.

Current audited structure:

- package Python files: roughly 359;
- package source lines: roughly 103k;
- no blocked root-prefix modules remain under the architecture manifest;
- remaining long functions are mostly 100-123 lines and are concentrated in
  validation scans, calibration/report writers, artifact handoff, geometry
  metric construction, and objective/gradient reports.

Efficient refactor policy for the next version:

1. Split a function only when the split creates a reusable tested policy
   boundary, removes duplicated logic, or makes an AD path side-effect-free.
2. Avoid new one-off modules unless the new module is an extension boundary
   such as geometry backends, objectives, solvers, or validation gates.
3. Consolidate single-use helper modules back into nearby domain files when that
   improves developer navigation without increasing public-function length.
4. Keep short public facades and private implementation helpers; remove legacy
   aliases and examples when they describe workflows we no longer support.
5. Continue replacing non-benchmark comparison-code terminology with physical
   or numerical terminology.
6. Keep performance changes separate from cleanup unless an identity gate and
   profiler artifact are produced in the same tranche.

Highest-value remaining refactor tranches:

| Tranche | Scope | Done When |
| --- | --- | --- |
| Artifact/report orchestration | NetCDF writers, runtime artifact handoff, report payload builders | Schemas unchanged, focused artifact tests pass, each long writer is staged into grid/data/metadata helpers. |
| Validation scan drivers | Cyclone/ETG/KBM/TEM scan wrappers still above 100 lines | Scan controls, per-point execution, fitting, and result packing are separated with focused branch tests. |
| Objective/gradient reports | VMEC/Boozer FD, line-search, portfolio, nonlinear-window reports | Pure objective evaluation, finite-difference/tangent checks, and plotting/report assembly are separated. |
| Geometry metric construction | VMEC field-line metric coefficients and Boozer bridge reports | Metric, drift, interpolation, and parity/gradient gates are independently testable. |
| Naming cleanup | non-benchmark `gx`, `runtime_`, and legacy reduced-model naming | Names describe physics or numerics; benchmark-only references remain scoped. |

## Next-Version Closure Plan

The next version should be cut only after these finite gates are satisfied:

1. **Technical gates**
   - local fast tests for touched modules pass;
   - `ruff`, focused `mypy`, architecture, repository-size, and release-status
     checks pass;
   - package-wide coverage remains above 95% in CI;
   - docs build if README/docs/source API changed.
2. **Refactor gates**
   - no new source bloat from transient helpers;
   - no schema-breaking artifact change unless docs/tests are updated in the
     same commit;
   - no new compatibility layer for old examples or legacy output names.
3. **Science gates**
   - quasilinear claims remain screening/model-development unless held-out
     nonlinear transport-error gates pass;
   - nonlinear turbulent optimization claims require long post-transient
     replicated transport windows;
   - VMEC/Boozer optimization claims require geometry parity and gradient gates.
4. **Performance gates**
   - nonlinear domain-decomposition claims require serial-vs-decomposed
     numerical identity and CPU/GPU profiler-backed speedup;
   - runtime/memory panel refreshes must be from fresh measured artifacts.
5. **Release gates**
   - update version only after the above gates pass on the pushed commit;
   - tag and release from `main`;
   - verify GitHub release workflow and PyPI publish.

## Immediate Next Steps

1. Continue low-risk refactor tranches in remaining artifact/validation/report
   hotspots, with focused tests and no schema changes.
2. Resume performance work only with identity gates and profiler evidence before
   updating runtime or speedup claims.
3. Resume science lanes with long-window nonlinear evidence before promoting
   stronger quasilinear or turbulent-optimization claims.
4. Use `office` GPUs only for a specific post-release science/performance lane
   or if a GPU-specific gate is needed.

## Latest Refactor Log

- 2026-06-20: split `run_cyclone_linear` into a stable public facade plus a
  typed `_CycloneLinearRequest`, request-to-fit-policy assembly, status routing,
  and private request runner.  This preserved the public API and Cyclone result
  schema while dropping the public wrapper from 108 lines to 41 lines.
- Focused gates passed for this tranche: Ruff, mypy, Cyclone linear branch
  tests, explicit integration invalid-option entrypoint test, differentiable
  refactor manifest, architecture manifest, repository-size manifest, and
  `git diff --check`.
- 2026-06-20: split `run_cyclone_time_path` into a stable public facade plus a
  typed `_CycloneTimePathRequest`, request-to-control assembly, and private
  reference-aligned/saved-time dispatch.  This preserved the time-path branch
  behavior while removing the largest remaining validation-driver hotspot.
- Focused gates passed for the time-path tranche: Ruff, mypy, Cyclone linear
  branch tests, differentiable refactor manifest, architecture manifest,
  repository-size manifest, and `git diff --check`.
- 2026-06-20: split the ETG ky-scan driver into a stable public facade, typed
  `_ETGScanRequest`, request-to-setup/runtime assembly, and batch execution that
  consumes `_ETGScanRuntimeOptions` plus the scan accumulator.  This removed the
  two ETG scan hotspots while preserving scan result schemas and solver branch
  behavior.
- Focused gates passed for the ETG scan tranche: Ruff, mypy, ETG scan branch
  tests, differentiable refactor manifest, architecture manifest,
  repository-size manifest, and `git diff --check`.
- 2026-06-20: split `run_tem_linear` into a stable public facade plus
  `_TEMLinearRequest`, setup/state helpers, and private Krylov/time dispatch.
  TEM scan species-index validation now reuses the same helper.  This preserved
  TEM result schemas and solver branch behavior while removing the TEM linear
  wrapper hotspot.
- Focused gates passed for the TEM tranche: Ruff, mypy, TEM branch tests,
  differentiable refactor manifest, architecture manifest, repository-size
  manifest, and `git diff --check`.
- 2026-06-20: split `run_kbm_linear` into a stable public facade plus
  `_KBMLinearRequest`, request-to-setup/options assembly, and private solver
  dispatch.  Existing KBM setup/state/solver helpers were reused, preserving
  solver selection and output schemas.
- Focused gates passed for the KBM linear tranche: Ruff, mypy, KBM linear branch
  tests, differentiable refactor manifest, architecture manifest,
  repository-size manifest, and `git diff --check`.
- 2026-06-20: split kinetic-electron scan control assembly into a typed
  `_KineticScanControlRequest`, request construction from the public wrapper,
  and a shorter setup/run/fit-control resolver.  This preserved
  `run_kinetic_scan` inputs and scan output schema while removing the kinetic
  control hotspot.
- Focused gates passed for the kinetic scan tranche: Ruff, mypy, kinetic scan
  branch tests, differentiable refactor manifest, architecture manifest,
  repository-size manifest, and `git diff --check`.
- 2026-06-20: split `run_kbm_beta_scan` into a stable public facade plus a
  `_KBMBetaScanRequest` and private request runner.  Existing beta-scan setup,
  fit-policy, option, and per-beta loop helpers were reused, preserving fixed-ky
  beta-scan behavior and output schema.
- Focused gates passed for the KBM beta tranche: Ruff, mypy, KBM beta branch
  tests, differentiable refactor manifest, architecture manifest,
  repository-size manifest, and `git diff --check`.
- 2026-06-20: split `run_etg_linear` into a stable public facade plus
  `_ETGLinearRequest` and a private request runner.  The Krylov-first auto
  solver policy, time-path fallback, fit-signal validation, and output schema
  are unchanged.
- Focused gates passed for the ETG linear tranche: Ruff, mypy, ETG linear branch
  tests, differentiable refactor manifest, architecture manifest,
  repository-size manifest, release-readiness check, and `git diff --check`.
- 2026-06-20: split nonlinear transport audit and landscape admission report
  builders into explicit metric extraction, blocker evaluation, candidate-row
  assembly, selection, and next-action helpers.  This preserves the
  VMEC/Boozer nonlinear turbulent-flux optimization evidence schemas while
  making the fail-closed policy boundaries smaller and easier to test.
- Focused gates passed for the transport-admission tranche: Ruff, focused mypy,
  transport admission tests, landscape/audit builder-script tests,
  architecture manifest, repository-size manifest, differentiable-refactor
  manifest, release-readiness check, and `git diff --check`.
- 2026-06-20: split nonlinear-gradient same-control bracket-sweep evidence
  helpers into evidence-config construction, conditioning metric extraction,
  margin scoring, repeated-bracket stability, gate-name extraction, and staged
  recommendation predicates.  This preserves the bracket-sweep artifact schema
  and facade re-exports while making the differentiable turbulent-flux
  optimization evidence policy easier to audit.
- Focused gates passed for the nonlinear-gradient bracket tranche: Ruff,
  focused mypy, full nonlinear-gradient evidence tests, architecture manifest,
  repository-size manifest, differentiable-refactor manifest,
  release-readiness check, and `git diff --check`.
- 2026-06-20: split nonlinear spectral identity reporting into shared abs/rel
  error summaries, identity-pass policy helpers, chunk/tile normalization, and
  RHS report construction.  This preserves the serial-vs-logical-shard identity
  report schemas while making the nonlinear domain-decomposition performance
  gate easier to inspect before any future speedup claim.
- Focused gates passed for the nonlinear spectral identity tranche: Ruff,
  focused mypy, nonlinear spectral communication/parallel tests, architecture
  manifest, repository-size manifest, differentiable-refactor manifest,
  release-readiness check, and `git diff --check`.
- Remaining source functions at or above 100 lines: 0.
- Remaining source functions at or above 90 lines: 20.
- Remaining source functions at or above 80 lines: 71.

## Latest Release Log

- `v1.6.8` was merged to `main`, tagged, published to PyPI, and released on
  GitHub on 2026-06-20.
- Release readiness, repository-size, architecture, release-artifact,
  technical-release, package-build, Twine metadata, docs build, GitHub release,
  and PyPI publish gates passed.
- PyPI lists `spectraxgk==1.6.8` with wheel and source distribution artifacts.
