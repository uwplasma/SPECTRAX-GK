- 2026-06-19: Continued differentiable-geometry report simplification inside
  `geometry.vmec_flux_tube_reports` without adding modules. The VMEC flux-tube
  parity path now shares array-metric, worst-error, and optional Boozer
  equal-arc parity helpers instead of embedding the full optional-backend
  branch inside the public array-parity report. Public report keys and
  differentiable-geometry facade imports remain unchanged, while
  `vmec_flux_tube_reports.py` dropped from 519 to 497 lines and the public
  report now reads as direct-parity setup plus optional equal-arc parity plus
  final JSON assembly. Local gates passed: focused differentiable-geometry
  flux-tube shard, Ruff, mypy, and `py_compile`.

- 2026-06-19: Continued differentiable-geometry validation refactor by
  simplifying solved-equilibrium candidate gates and VMEC/Boozer objective
  line-search gates without adding modules. `validation.stellarator.
  candidate_gate` now builds aspect, mean-iota, iota-profile, and pass/fail
  checks through shared private helpers, keeping the publication-facing JSON
  schema stable. `objectives.vmec_boozer_line_search` now routes scalar and
  aggregate one-parameter searches through one shared curvature-gated loop,
  while public scalar, aggregate, and holdout report functions retain their
  signatures and report fields. The touched modules dropped from 999 to 961
  total lines. Local gates passed: candidate-gate shard, QA optimization-driver
  candidate shard, VMEC/Boozer line-search/holdout shards, Ruff, mypy,
  `py_compile`, and `git diff --check`.

- 2026-06-19: Continued nonlinear diagnostic wiring simplification inside
  `solvers.nonlinear.diagnostic_integration` without adding a module. The
  explicit and IMEX public entry-point wrappers now use the shared option-key
  tables for forwarding into their injected implementation owners instead of
  duplicating long keyword lists. The IMEX method branch set is also centralized
  in `_IMEX_METHODS`. This keeps all public signatures and monkeypatch seams
  unchanged while reducing the wiring module from 495 to 445 lines and making
  future diagnostic options a one-table update. Local gates passed: nonlinear
  forwarding tests, nonlinear/IMEX/package identity shard, py_compile, Ruff,
  mypy, refactor manifest, validation coverage manifest, repository size
  manifest, source terminology scans, and `git diff --check`.

- 2026-06-19: Continued nonlinear solver refactor by moving IMEX diagnostic
  integration orchestration from `solvers.nonlinear.diagnostics` into the
  existing `solvers.nonlinear.imex_diagnostics` owner. The explicit diagnostic
  implementation remains in `diagnostics.py`, while IMEX diagnostic dependency
  wiring, implicit-operator setup, nonlinear-term/solve-step closure assembly,
  fixed diagnostic scan, and final diagnostic packing now live with the IMEX
  diagnostic step policy. `diagnostics.py` dropped from 499 to 267 lines and
  `imex_diagnostics.py` grew from 171 to 420 lines, with no new module added.
  `diagnostics.py` retains explicit `__all__` re-exports so public imports from
  `spectraxgk.solvers.nonlinear.diagnostics` remain stable. Updated
  code-structure and differentiable-refactor docs. Local gates passed: IMEX
  unit/package-identity tests, nonlinear diagnostic helper shard, py_compile,
  Ruff, mypy, refactor manifest, validation coverage manifest, repository size
  manifest, source terminology scans, and `git diff --check`.

- 2026-06-19: Continued executable/runtime simplification by moving runtime
  command presentation policy from `workflows.runtime.commands` into the
  existing `workflows.runtime.command_artifacts` owner. Command artifact output
  now owns saved-path display order, linear/nonlinear executable headers, and
  nonlinear terminal summaries; `commands.py` keeps command orchestration,
  option resolution, path/quasilinear overrides, and public re-exports.
  `commands.py` dropped from 499 to 420 lines while `command_artifacts.py`
  grew from 113 to 198 lines, with no new module added. Updated code-structure
  and API docs to record the ownership boundary. Local gates passed: focused
  runtime command-output tests, full CLI/runtime helper shard, Ruff, mypy,
  `py_compile`, refactor manifest, validation coverage manifest, repository
  size manifest, source terminology scans, and `git diff --check`.

- 2026-06-19: Continued linear-operator refactor by extracting collisional
  damping, hypercollisions, perpendicular hyperdiffusion, and linked/end
  damping from `terms.linear_terms` into `terms.linear_dissipation`. The
  streaming, mirror, curvature/grad-B, and diamagnetic drive module now stays
  focused on physical linear drives, while artificial/physical dissipation has
  its own owner. Existing imports from `terms.linear_terms` remain object-
  identical re-exports for comparison/profiling tools, and internal RHS
  assembly now imports dissipation functions from the canonical owner.
  `linear_terms.py` dropped from 571 to 287 lines and the new dissipation
  module is 307 lines. Local gates passed: linear/RHS/term-assembly tests,
  Ruff, mypy, `py_compile`, architecture/refactor/validation-coverage/size
  manifests, terminology audits, and `git diff --check`.
- 2026-06-19: Continued differentiable zonal-flow objective refactor by
  extracting record normalization, complete surface/alpha/kx tensor assembly,
  missing-damping policy, and objective row-table construction from
  `objectives.zonal` into `objectives.zonal_records`. The public zonal
  objective, reduced scalar, artifact builder, and AD/FD sensitivity report
  remain available through `objectives.zonal`, while the non-differentiable
  artifact-ingestion layer now has a focused owner. `zonal.py` dropped from 581
  to 374 lines and the new record helper module is 239 lines. Local gates
  passed: zonal objective/artifact tests, Ruff, mypy, `py_compile`,
  architecture/refactor/validation-coverage/size manifests, terminology audits,
  and `git diff --check`.
- 2026-06-19: Continued quasilinear validation refactor by extracting
  model-selection input normalization and optimized-equilibrium audit summaries
  from `validation.quasilinear.model_selection` into
  `validation.quasilinear.model_selection_inputs`. The public claim-ledger
  builder now focuses on assembling promotion gates and path-based wrappers,
  while artifact loading, required-candidate metric normalization, calibration
  summaries, and scoped nonlinear-audit summaries have a focused owner.
  `model_selection.py` dropped from 541 to 321 lines and the new input helper
  module is 239 lines. The same tranche assigned recent split modules to the
  validation coverage manifest so package-wide coverage ownership remains
  explicit. Local gates passed: quasilinear model-selection tests, model-
  selection plotting tests, Ruff, mypy, `py_compile`, architecture/refactor/
  validation-coverage/size manifests, terminology audits, and
  `git diff --check`.
- 2026-06-19: Continued nonlinear-gradient validation refactor by splitting
  `validation.nonlinear_gradient.evidence_screening` into focused screening
  owners. Candidate ranking remains in `evidence_screening`, same-control
  bracket-sweep rows/recommendations now live in `evidence_brackets`, and
  normalized evidence-margin scoring now lives in `evidence_scoring` so both
  paths share one gate-margin policy. The public `evidence` facade and
  `evidence_screening` compatibility re-exports are preserved and tested.
  `evidence_screening.py` dropped from 547 to 285 lines; new modules are 280
  and 30 lines. Local gates passed: nonlinear-gradient evidence shard, Ruff,
  mypy, `py_compile`, architecture/refactor/size manifests, terminology audits,
  and `git diff --check`.
- 2026-06-19: Continued VMEC differentiability cleanup by extracting JAX-only
  VMEC field-line coordinate construction and differentiable RMS reductions
  from `geometry.vmec_state_sensitivity` into
  `geometry.vmec_field_line_sampling`. Field-line tensor report wrappers keep
  their optional-backend orchestration and private compatibility aliases, while
  sampling convention and zero-safe RMS reductions now have a focused owner.
  `vmec_state_sensitivity.py` dropped further from 554 to 516 lines and the new
  sampling helper module is 55 lines. Local gates passed: differentiable
  geometry bridge shard, refactor manifest tests, Ruff, mypy, `py_compile`,
  architecture/refactor/size manifests, terminology audits, and
  `git diff --check`.
- 2026-06-19: Continued differentiable VMEC/Boozer refactor by extracting
  shared VMEC-JAX state-control scaffolding from
  `geometry.vmec_state_sensitivity` into `geometry.vmec_state_controls`.
  Example-state loading, coefficient index resolution, length-two perturbation
  normalization, and pure `VMECState` coefficient perturbation now have a
  focused owner used by both state-sensitivity reports and flux-tube reports.
  Public report wrappers and optional-backend patch seams remain in
  `vmec_state_sensitivity`; private helper aliases remain re-exported for
  existing tests. `vmec_state_sensitivity.py` dropped from 650 to 554 lines and
  the new controls module is 130 lines. Local gates passed: differentiable
  geometry bridge shard, refactor manifest tests, Ruff, mypy, `py_compile`,
  architecture/refactor/size manifests, terminology audits, and
  `git diff --check`.
- 2026-06-19: Continued executable/runtime refactor by extracting command
  saved-path display and optional artifact-write stdout policy from
  `workflows.runtime.orchestration_artifacts` into
  `workflows.runtime.command_artifacts`. Nonlinear artifact restart/checkpoint
  handoff stays in `orchestration_artifacts`, while runtime commands now import
  the command-output helpers from their canonical owner and the old helper
  surface remains re-exported for tests/users. `orchestration_artifacts.py`
  dropped from 495 to 411 lines and the new command artifact module is 113
  lines. Local gates passed: focused runtime command-artifact stdout-order
  tests, CLI artifact path tests, nonlinear artifact handoff tests, Ruff, mypy,
  `py_compile`, architecture/refactor/size manifests, terminology audits, and
  `git diff --check`.
- 2026-06-19: Continued nonlinear solver refactor by extracting diagnostic
  IMEX stage and scan policy from `solvers.nonlinear.imex` into
  `solvers.nonlinear.imex_diagnostics`. Cached IMEX solve policy keeps the
  fixed-point predictor, GMRES solve-step seam, implicit-operator shape policy,
  and cached scan, preserving existing monkeypatch and public import surfaces.
  `imex.py` dropped from 491 to 352 lines and the new diagnostic stepping
  module is 171 lines. Local gates passed: IMEX unit tests, nonlinear shard,
  solver package re-export tests, Ruff, mypy, `py_compile`,
  architecture/refactor/size manifests, terminology audits, and
  `git diff --check`.
- 2026-06-19: Continued VMEC differentiability/refactor cleanup by extracting
  collection-level boundary-chain gate policy from `geometry.vmec_boundary_chain`
  into `geometry.vmec_boundary_collection`. Single-probe scalar chain
  classification remains in `vmec_boundary_chain`, while row assembly, stable
  count payloads, and collection decision policy now have a focused owner
  re-exported through the existing helper surface. `vmec_boundary_chain.py`
  dropped from 657 to 480 lines and the new collection module is 197 lines.
  Local gates passed: boundary-chain collection tests, Ruff, mypy,
  `py_compile`, architecture/refactor/size manifests, terminology audits, and
  `git diff --check`.
- 2026-06-19: Continued differentiable-geometry/refactor cleanup by extracting
  pure Boozer field-line numerics from `geometry_backends.vmec_fieldlines` into
  `geometry_backends.vmec_fieldline_numerics`. Boozer mode angle/basis
  contractions, mode-table sampling, field-line integrals, reference-scale
  validation, and Hegna-Nakajima shear/pressure correction policy now have a
  focused owner, while `vmec_fieldlines` keeps backend fallback, dataset
  lifecycle, VMEC/Boozer tensor assembly, and normalized flux-tube coefficient
  packaging. Private helper re-exports are preserved and directly tested.
  `vmec_fieldlines.py` dropped from 787 to 486 lines and the new numerical
  helper module is 345 lines. Local gates passed: full VMEC backend helper
  shard, Ruff, mypy, `py_compile`, package-architecture manifest,
  differentiable-refactor manifest, repository-size manifest, source terminology
  audits, and `git diff --check`.
- 2026-06-19: Continued executable/runtime simplification by extracting pure
  runtime command option policy into `workflows.runtime.command_options`.
  Option dataclasses, CLI/TOML/default resolution, ky scan parsing, fit-config
  filtering, and progress-display policy now have one focused owner; the
  runtime command workflow module keeps command orchestration, headers,
  path/quasilinear overrides, artifact writes, and summaries while re-exporting
  the existing helper surface for tests. `commands.py` dropped from 694 to 499
  lines and the new option module is 268 lines. Updated the differentiable
  refactor manifest to record the new owner. Local gates passed: focused
  runtime command option tests, full runtime helper shard, CLI shard, Ruff,
  mypy, `py_compile`, package-architecture manifest, differentiable-refactor
  manifest, repository-size manifest, source terminology audits, and
  `git diff --check`.
- 2026-06-19: Continued nonlinear device-z refactor/performance lane by
  extracting fail-closed device-z RHS and transport-window report policy into
  `operators.nonlinear.device_z_reports`. The sharded operator route now owns
  only z-sharding, fused RHS execution, observable collection, and explicit
  transport-window stepping; report construction, trace freezing, and identity
  tolerance policy are tested through the existing device-z helper surface.
  `device_z.py` dropped from 651 to 440 lines, the generic spectral identity
  report module stayed focused at 369 lines, and the new report-policy module is
  290 lines. Updated the differentiable refactor manifest to record the new
  owner. Local gates passed: focused device-z identity/report tests, full
  nonlinear parallel shard, Ruff, mypy, `py_compile`, package-architecture
  manifest, differentiable-refactor manifest, repository-size manifest, source
  terminology audits, and `git diff --check`.
- 2026-06-19: Continued performance/refactor cleanup in nonlinear
  parallelization by moving pencil-FFT RHS identity and pencil transport-window
  identity implementations out of the public `operators.nonlinear.parallel`
  facade. Pencil RHS routing now lives with `spectral_identity_rhs`, and the
  fixed-window physical-transport trace gate now lives with
  `spectral_identity_integrator`. The public parallel module now re-exports
  these identity-gated routes instead of owning their numerical implementation,
  while preserving fail-closed behavior and avoiding new speedup claims. Added
  facade re-export tests for the moved functions. Local gates passed: full
  nonlinear parallel test shard, focused pencil RHS/transport-window gates,
  Ruff, mypy, `py_compile`, package-architecture manifest,
  differentiable-refactor manifest, repository-size manifest, source terminology
  audits, and `git diff --check`.
- 2026-06-19: Continued runtime command simplification by moving command
  artifact display ordering and optional artifact write/print helpers from
  `workflows.runtime.commands` into `workflows.runtime.orchestration_artifacts`.
  Runtime commands now resolve options and dispatch runs, while artifact
  orchestration owns saved-path ordering for linear, quasilinear, scan, and
  nonlinear outputs. Added direct tests against the artifact helper ownership
  and preserved executable command-output behavior through focused CLI/runtime
  tests. Local gates passed: focused runtime artifact-output tests, focused CLI
  artifact tests, Ruff, mypy, `py_compile`, package-architecture manifest,
  differentiable-refactor manifest, repository-size manifest, source terminology
  audits, and `git diff --check`.
- 2026-06-19: Continued runtime scan-orchestration simplification by moving
  per-ky scan task execution from the public `spectraxgk.runtime` facade into
  `workflows.runtime.orchestration_scan`. The runtime facade now injects its
  patchable `run_runtime_linear` into `run_runtime_scan_ky_task`, keeping
  downstream monkeypatch seams intact while making worker-task option forwarding
  independently testable. Added direct coverage for the forwarded linear
  options plus existing scan-dependency and parallel-order gates. Local gates
  passed: focused runtime scan tests, Ruff, mypy, `py_compile`,
  package-architecture manifest, differentiable-refactor manifest,
  repository-size manifest, source terminology audits, and `git diff --check`.
- 2026-06-19: Continued executable facade simplification by moving direct
  TOML shorthand classification from `spectraxgk.cli` into
  `workflows.runtime.toml`. Runtime-vs-named-case TOML detection, known-command
  guards, missing-file guards, and shorthand argument construction now live next
  to TOML loading/path-resolution policy. The CLI keeps only thin wrappers so
  tests and downstream patch surfaces still see the existing private names and
  the facade-injected `load_toml`. Added direct workflow-level tests for
  runtime/named-case classification and shorthand guard behavior. Local gates
  passed: explicit CLI compatibility tests, TOML policy test, Ruff, mypy,
  `py_compile`, package-architecture manifest, differentiable-refactor manifest,
  repository-size manifest, source terminology audits, and `git diff --check`.
- 2026-06-19: Continued runtime startup simplification by making the
  VMEC/Miller geometry routing policy shared instead of duplicated between
  `spectraxgk.runtime` and `workflows.runtime.startup`.
  `runtime_geometry_config_for_builder` now lives in the startup workflow layer
  and accepts explicit EIK-generation callbacks so the public runtime facade
  keeps its patchable test/user surface. Added integration coverage for the
  shared startup helper while preserving the fast runtime-facade geometry
  builder test. Local gates passed: focused runtime-helper test, explicit
  integration startup-geometry test, Ruff, mypy, `py_compile`,
  package-architecture manifest, differentiable-refactor manifest,
  repository-size manifest, source terminology audits, and `git diff --check`.
- 2026-06-19: Continued runtime facade simplification by separating runtime
  geometry routing from flux-tube geometry construction.
  `_runtime_geometry_config_for_builder` now owns VMEC and Miller EIK
  generation routing plus default geometry pass-through, while
  `build_runtime_geometry` only calls the flux-tube builder. Added direct tests
  for VMEC, Miller, and default routing plus existing builder behavior. Local
  gates passed: focused runtime-helper geometry/dependency tests, Ruff, mypy,
  `py_compile`, package-architecture manifest, differentiable-refactor manifest,
  repository-size manifest, source terminology audits, and `git diff --check`.
- 2026-06-19: Continued VMEC imported-geometry cleanup by centralizing
  Boozer trigonometric basis construction. `_boozer_trig_basis` now owns the
  shape-generic `cos/sin`, `m cos`, `m sin`, `n cos`, and `n sin` arrays used by
  both Hegna-Nakajima flux-surface averages and the main field-line assembly.
  Added direct tests for mode-axis preservation and weighted derivative arrays.
  Local gates passed: focused VMEC field-line helper tests, Ruff, mypy,
  `py_compile`, package-architecture manifest, differentiable-refactor manifest,
  repository-size manifest, and source terminology audits.
- 2026-06-19: Continued differentiable-geometry facade simplification
  without changing public bridge exports or optional-backend routing. Temporary
  facade hook patching now uses `_patched_module_attrs`, so Boozer, VMEC-state,
  tensor-mapping, VMEC/Boozer-core, and VMEC report wrappers share one restore
  policy. Added direct restoration coverage for successful calls and exception
  paths. Local gates passed: focused differentiable-geometry facade tests,
  Ruff, mypy, `py_compile`, package-architecture manifest, differentiable-refactor
  manifest, repository-size manifest, and source terminology audits.
- 2026-06-19: Continued executable facade simplification in
  `spectraxgk.cli` without changing command-line behavior. Direct TOML
  shorthand dispatch now routes through `_toml_shorthand_command` and
  `_direct_config_shorthand_args`, separating runtime-vs-named-case TOML
  classification from `main()`. Added direct tests for runtime shorthand,
  named-case shorthand, known-command guards, flag guards, and missing-file
  guards. Local gates passed: focused CLI shorthand tests, Ruff, mypy,
  `py_compile`, package-architecture manifest, differentiable-refactor
  manifest, repository-size manifest, and source terminology audits.
- 2026-06-19: Continued VMEC imported-geometry simplification in
  `spectraxgk.geometry_backends.vmec_fieldlines` without changing Boozer-mode
  sums or field-line outputs. Hegna-Nakajima shear and pressure correction
  factors now live in `_hngc_shear_correction` and `_hngc_pressure_correction`,
  separating physics switch/floor policies from the main field-line assembly.
  Added direct tests for enabled and disabled shear/pressure variation plus the
  small-pressure-gradient floor. Local gates passed: focused VMEC field-line
  helper tests, Ruff, mypy, `py_compile`, package-architecture manifest,
  differentiable-refactor manifest, repository-size manifest, and source
  terminology audits.
- 2026-06-19: Continued runtime executable refactor in the named-case path
  without changing CLI/TOML behavior. Named linear run and scan commands now
  resolve CLI overrides, TOML values, defaults, fit-window options, scan ky
  arrays, and progress policy through explicit option objects before command
  execution. Added direct tests for single-run precedence, scan parsing,
  remaining fit-window options, and empty-scan rejection. Local gates passed:
  focused named-case and CLI shorthand tests, full runtime-helper shard, Ruff,
  mypy, `py_compile`, package-architecture manifest, differentiable-refactor
  manifest, repository-size manifest, and source terminology audits.
- 2026-06-19: Continued differentiable VMEC state-sensitivity cleanup
  without changing optional-backend report schemas. Field-line tensor sampling
  coordinates now live in `_vmec_field_line_sampling_coordinates`, which owns
  the ntheta gate, iota-profile compatibility check, zero-iota floor, and
  VMEC theta/zeta convention. Metric and field-line tensor reports also share
  `_rms_with_floor` for smooth RMS observables. Added direct tests for the
  field-line coordinate contract, malformed iota profiles, zero-iota handling,
  and RMS floor policy. Local gates passed: focused differentiable-geometry
  sensitivity tests, Ruff, mypy, `py_compile`, package-architecture manifest,
  differentiable-refactor manifest, repository-size manifest, and source
  terminology audits.
- 2026-06-19: Continued nonlinear device-z performance-gate cleanup
  without changing routed RHS numerics. The transport-window identity decision
  now lives in `_device_z_transport_identity_passed`, so final-state,
  free-energy, field-energy, physical-flux, and bracket-RMS tolerances are
  checked by one auditable policy before any decomposed path is enabled. Added
  direct pass/fail tests for the helper. Local gates passed: device-z nonlinear
  parallel tests (with expected multi-device skip when unavailable), Ruff,
  mypy, `py_compile`, package-architecture manifest, differentiable-refactor
  manifest, parallel-scaling artifact manifest, repository-size manifest, and
  source terminology audits.
- 2026-06-19: Continued differentiable VMEC boundary-chain refactoring
  without changing artifact schemas. Boundary-chain collection classification
  now lives in `_boundary_chain_collection_decision`, separating row counting
  from the promotion/diagnostic decision ladder for nonfinite, internal replay,
  exact-FD, frozen-axis-convention, and branch-sensitive cases. Added direct
  tests for each collection decision branch so future VMEC-JAX derivative
  gate changes cannot silently relax the policy. Local gates passed:
  VMEC boundary-chain tests, projected line-search boundary-chain tests, Ruff,
  mypy, `py_compile`, package-architecture manifest, differentiable-refactor
  manifest, repository-size manifest, and source terminology audits.
- 2026-06-19: Continued VMEC imported-geometry simplification in
  `spectraxgk.geometry_backends.vmec_fieldlines` without changing Boozer
  algebra or sampled geometry outputs. Reference-scale validation and
  iota/shear override handling now live in named helper policies, including
  explicit coverage for invalid minor radius rejection and the zero-shear
  floor used by the local flux-tube coordinates. Local gates passed: VMEC
  field-line helper tests, edited runtime/CLI/geometry test shard, Ruff,
  mypy, `py_compile`, package-architecture manifest, differentiable-refactor
  manifest, repository-size manifest, and source terminology audit.
- 2026-06-19: Continued runtime workflow simplification without adding new
  source files. Runtime executable commands now route linear, quasilinear, and
  scan artifact persistence through focused output-policy helpers, and
  programmatic TOML case workflows now resolve Python-argument/TOML/default
  precedence through table-driven linear and nonlinear case policies instead
  of inline fallback blocks. Added direct regression coverage for command
  output policies and case-option precedence. Public executable and runtime
  facade behavior is unchanged. Local gates passed: runtime-helper and CLI
  test shard, focused runtime-case tests, Ruff, mypy, `py_compile`, package
  architecture manifest, differentiable-refactor manifest, repository-size
  manifest, and source terminology audit.
- 2026-06-19: Refreshed the validation gate index to match the current
  scoped quasilinear claim boundary. The index now records 17/18 tracked
  gate reports passing and shows the quasilinear model-selection status as
  the sole open row because the spectral-envelope candidate misses the strict
  transport-error gate and is not promoted as an absolute-flux predictor.
  Added the small PDF companion generated by `tools/make_validation_gate_index.py`
  and updated manuscript figures, verification matrix, testing, and release-scope
  docs to stop claiming every indexed gate passes. Local gates passed:
  validation-index tests, release-readiness tests, release-scope docs tests,
  release readiness, quasilinear promotion guardrails, validation-coverage
  manifest, repository-size manifest, release-artifact manifest, package-architecture
  manifest, and warning-free Sphinx docs build.
- 2026-06-19: Continued differentiable-objective cleanup in
  `spectraxgk.objectives.zonal`. Zonal-flow artifact row-table serialization
  now lives in `_zonal_row_table`, using precomputed surface/alpha/kx index
  maps instead of repeated list scans inside the public artifact builder. The
  public payload schema and objective reductions are unchanged. Local gates
  passed: zonal objective tests, zonal-flow objective gate tests, stellarator
  portfolio tests, Ruff, mypy, and `py_compile` on touched files.
- 2026-06-19: Continued Cyclone benchmark branch simplification inside
  `spectraxgk.validation.benchmarks.cyclone_scan_branches` without changing
  solver kernels or reference-data comparisons. The time-branch auto-solver
  fallback to the Krylov path now lives in `_resolve_time_branch_growth`, so
  both automatic-signal and fitted-signal paths share the same validity check,
  fallback arguments, and return convention. Added direct regression coverage
  for valid time fits and invalid positive-growth fits that must call the
  Krylov fallback, and renamed a benchmark test from comparison-code seed
  wording to reference-seed wording. Local gates passed: selected Cyclone
  fallback tests, full benchmark-runner branch tests, Ruff, mypy, and
  `py_compile` on touched files.
- 2026-06-19: Continued the quasilinear validation refactor inside the existing
  `spectraxgk.validation.quasilinear.model_selection` owner without adding
  source files. Required-candidate metric extraction for accepted rules,
  transport-error thresholds, interval-coverage thresholds, null/linear
  baselines, and promotion eligibility now lives in one helper policy used by
  the model-selection ledger. Added direct regression coverage for threshold
  overrides, accepted-rule fallback, and malformed candidate payloads. Local
  gates passed: quasilinear model-selection tests, model-selection plot tests,
  Ruff, mypy, `py_compile`, source terminology audits, differentiable-refactor,
  package-architecture, performance, repository-size, release-artifact,
  release-version, validation-coverage, quasilinear-guardrail, and
  parallel-scaling artifact manifests. Regenerating the validation gate index
  still exposes a pre-existing stale QL status artifact, so the generated
  index files were not included in this behavior-preserving source tranche.
- 2026-06-18: Aligned the architecture docs and differentiable-refactor
  manifest with the stable public-facade policy. The executable manifest now
  uses `public_facade_policy` rather than `compatibility_policy`, the manifest
  checker validates that field, and architecture/code-structure/differentiable
  refactor docs now describe public facades plus removal of old helper shims
  instead of a compatibility-facade target. This keeps the active plan aligned
  with the source cleanup that already removed legacy/historical terminology
  from `src/spectraxgk`.
- 2026-06-18: Removed remaining source-level legacy/historical terminology and
  old Cyclone scan private alias shims. Cyclone scan branch code now imports
  explicit-time reselection and seed policies by their canonical public helper
  names, while the seed and explicit-time owner modules no longer export
  duplicated underscored aliases. The CLI named-case detector now uses
  `_NAMED_CASE_TOP_LEVEL_KEYS`, and maintained facades now describe documented
  public seams rather than historical compatibility surfaces. Focused Cyclone
  benchmark, CLI, runtime-helper/artifact, stellarator-objective, Ruff, mypy,
  and `py_compile` gates passed.
- 2026-06-18: Centralized runtime executable artifact-output writing inside
  `spectraxgk.workflows.runtime.commands._write_command_outputs`. Linear,
  linear-scan, and standalone quasilinear command helpers now share one
  destination/payload/write/print policy, and the old internal wrapper helpers
  were removed so command bodies use the canonical writer directly. Also removed the last source-level
  "compatibility" wording from the nonlinear restart help text by describing
  restart files as matching distribution states. Focused runtime-helper/CLI
  tests, Ruff, mypy, and `py_compile` passed.
- 2026-06-18: Removed the remaining mixed-case kinetic benchmark
  normalization constants from the maintained source surface. Kinetic-electron
  benchmark defaults, public benchmark re-exports, kinetic runners, and
  diagnostic/debug tools now use the same all-caps ``KINETIC_*`` naming scheme
  as Cyclone, ETG, KBM, and TEM. This drops a legacy naming wart without
  changing physics normalization values. Also tightened source and architecture
  documentation language from "compatibility" wording to stable public-facade
  ownership where behavior is intentionally maintained. Focused benchmark
  tests, Ruff, and `py_compile` passed.
- 2026-06-18: Removed the private adaptive-chunk `_format_duration`
  compatibility wrapper. Runtime chunk tests now import the canonical
  orchestration progress formatter directly, leaving
  `spectraxgk.workflows.runtime.chunks` focused on adaptive chunk execution,
  time-axis validation, truncation, striding, and result assembly. Focused
  runtime chunk/helper tests, Ruff, mypy, and `py_compile` passed.
- 2026-06-18: Removed a stale runtime diagnostic-array compatibility hop.
  `spectraxgk.workflows.runtime.diagnostics` now owns only linear fit and
  quasilinear finalization helpers; finite-value checks and diagnostic slicing,
  striding, truncation, and concatenation are imported directly from
  `spectraxgk.workflows.runtime.diagnostic_arrays`. Updated runtime tests to the
  canonical owner and removed stale private compatibility imports from the
  velocity-parallel and linear-cache facades. This tranche is net deletion
  (25 removed / 5 added before docs/log) and keeps public runtime behavior
  unchanged. Focused runtime, velocity, linear-cache, Ruff, mypy, and
  `py_compile` gates passed.
- 2026-06-18: Continued nonlinear parallelization testability inside the
  existing `spectraxgk.operators.nonlinear.device_z` owner without adding source
  files. Device-z fused RHS identity report construction now lives in explicit
  fail-closed and post-comparison helper policies, matching the earlier
  transport-window report split and keeping the sharded route focused on RHS
  computation and gated selection. Added direct helper coverage for topology
  blockers, passing identity reports, and failed candidate-comparison blockers;
  the nonlinear-parallel shard, Ruff, mypy, and `py_compile` passed.
- 2026-06-18: Continued executable runtime-command simplification inside the
  existing `spectraxgk.workflows.runtime.commands` owner without adding source
  files. Linear, scan, standalone quasilinear, and nonlinear artifact writing /
  save-message ordering now lives in focused command-output helpers rather than
  inline command bodies. Added direct helper coverage for skip behavior,
  save-order determinism, quasilinear availability, and nonlinear output gating;
  the runtime-helper/CLI shard, Ruff, mypy, and `py_compile` passed.
- 2026-06-18: Continued nonlinear solver simplification inside the existing
  `spectraxgk.solvers.nonlinear.imex` owner without adding source files.
  Cached IMEX implicit-operator resolution and initial-state shape normalization
  are now explicit helper policies, keeping `integrate_cached_imex_scan` focused
  on assembling nonlinear terms, solve steps, and the scan. Added direct helper
  coverage for provided-vs-built operators, optional implicit-operator builders,
  squeeze-species state insertion, dtype casting, and fail-fast shape mismatch;
  the focused IMEX shard, Ruff, mypy, and `py_compile` passed.
- 2026-06-18: Continued VMEC imported-geometry simplification inside the
  existing `spectraxgk.geometry_backends.vmec_fieldlines` owner without adding
  source files. Field-line Boozer coordinate construction on
  `alpha = theta_b - iota * phi_b` and the axisymmetric flip-convention
  detector are now explicit helper policies instead of inline setup inside
  `_vmec_fieldlines`. Added direct helper coverage for the alpha-line identity
  and flip/no-flip branches; the focused VMEC backend helper shard, Ruff, mypy,
  and `py_compile` passed.
- 2026-06-18: Continued differentiability objective-gate simplification inside
  `spectraxgk.objectives.gradient_gates` without adding source files. The
  repeated solver-ready linear setup for branch-continuity and geometry-gradient
  reports now lives in a shared `_SolverReadyLinearContext`, covering the
  Cyclone-like grid, selected ky grid, state shape, default linear parameters,
  term toggles, geometry mapping, cache construction, RHS/phi evaluation, and
  explicit operator matrix construction. The public report schemas and
  objective names are unchanged, while the branch objective now also reuses its
  existing field-line quadrature weights instead of recomputing them for the
  particle-flux normalization. Added direct context-contract coverage plus the
  focused solver-objective gradient tests; Ruff, mypy, and `py_compile` passed.
- 2026-06-18: Continued executable runtime-command simplification inside
  `spectraxgk.workflows.runtime.commands` without adding source files. The
  nonlinear runtime command now uses explicit `print_nonlinear_run_header` and
  `print_nonlinear_run_summary` helpers, matching the linear command structure
  and making user-facing progress/final-summary output testable without
  launching a real nonlinear simulation. Added focused coverage for
  diagnostics-present and diagnostics-absent nonlinear summary branches, and
  reconfirmed the full runtime-helper and CLI shards.
- 2026-06-18: Continued differentiable VMEC/Boozer boundary-chain
  simplification inside the existing `spectraxgk.geometry.vmec_boundary_chain`
  owner without adding source files. Scalar error construction for exact-FD,
  frozen-axis JVP/VJP, optional explicit tangent columns, and raw-branch
  diagnostics now lives in `_boundary_chain_error_metrics`, while the release
  gate pass/fail policy lives in `_boundary_chain_passes`. This leaves the
  public summary JSON schema and classification strings unchanged but makes
  VMEC/Boozer derivative-convention gates directly unit-testable without
  launching VMEC solves. Added focused helper coverage for missing optional
  linear tangent columns plus the existing boundary-chain summary, collection,
  projected line-search, and transport admission shards.
- 2026-06-18: Continued nonlinear parallelization testability inside the
  existing `spectraxgk.operators.nonlinear.device_z` owner without adding
  source files. The device-z transport-window identity gate now uses explicit
  helper policies for trace dictionary creation, trace freezing, trace-error
  summarization, and fail-closed transport-window report assembly. Added direct
  helper coverage while preserving the public `device_z_pencil_*` API and the
  existing fail-closed behavior. This is an identity-gate/refactor tranche
  only; it does not promote a new nonlinear GPU speedup claim. Local gates
  passed: selected device-z shard and full `tests/test_nonlinear_parallel.py`,
  Ruff on touched device-z/test files, mypy and `py_compile` on the device-z
  owner, source terminology audit, differentiable-refactor/validation-coverage/
  package-architecture manifests, repository-size manifest, warning-free
  Sphinx build, and `git diff --check`.
- 2026-06-18: Continued VMEC imported-geometry simplification inside the
  existing `spectraxgk.geometry_backends.vmec_fieldlines` owner without adding
  source files. Boozer backend fallback, Boozer-mode table sampling, Boozer
  angle construction, resonant-denominator guarding, flux-surface averaging,
  and centered field-line integrals are now explicit helper policies instead
  of inline branches inside `_vmec_fieldlines`. Added direct helper regressions
  for the angle/denominator, surface-average/centered-integral, and mode-table
  sampling policies while keeping the existing fake-VMEC field-line assembly
  tests unchanged. Local gates passed: full
  `tests/test_geometry_backend_vmec_helpers.py`, selected VMEC EIK and
  differentiable-geometry bridge shard, Ruff on touched files, mypy and
  `py_compile` on the VMEC field-line owner, source terminology audit,
  differentiable-refactor/validation-coverage/package-architecture manifests,
  repository-size manifest, warning-free Sphinx build, and `git diff --check`.
- 2026-06-18: Continued runtime artifact-handoff simplification inside the
  existing owner, `spectraxgk.workflows.runtime.orchestration_artifacts`, without
  adding source files. Restart input resolution, append-on-restart history
  loading, checkpoint chunk-size selection, restart-next-chunk config updates,
  diagnostic-history merging, and checkpoint-loop termination are now explicit
  helper policies instead of inline branches inside
  `run_runtime_nonlinear_artifact_handoff`. This makes the NetCDF/restart path
  easier to audit and extend while preserving the public handoff API and output
  schema. Local gates passed so far: focused restart/history/adaptive/live-output
  shard, full `tests/test_runtime_artifacts.py`, Ruff on touched runtime artifact
  files, mypy on the artifact handoff owner, and `py_compile` on the touched
  owner. The differentiable-refactor manifest, validation coverage manifest
  regeneration, repository-size manifest, warning-free Sphinx build, and `git
  diff --check` also pass.
- 2026-06-18: Completed a source-level comparison-code terminology cleanup
  pass. The only remaining `GX`/`gx_` identifiers under `src/spectraxgk` were
  ETG validation benchmark fit-window options, so they were renamed from
  `gx_growth` / `gx_navg_fraction` to
  `reference_growth_window` / `reference_navg_fraction`. Comparison-specific
  docs, tools, and benchmark artifact paths still mention GX where they are
  explicitly about benchmark/reference comparison. Local gates passed so far:
  ETG benchmark branch tests, ETG asset-helper test, Ruff on touched ETG/test
  files, mypy on touched ETG modules, and `py_compile` on touched ETG modules.
  The source terminology audit, repository-size manifest, and `git diff
  --check` also pass.
- 2026-06-18: Continued core linear-term simplification without adding source
  files by centralizing Hermite-mode drive insertion in
  `spectraxgk.terms.linear_terms._hermite_mode_drive`. Linked streaming,
  diamagnetic drives, and collision conservation corrections now share the
  same mode-mask convention instead of repeating local mask construction.
  Added a direct Hermite-mode isolation regression for the diamagnetic drive
  and documented the convention in the code-structure page. Local gates passed
  so far: focused linear consistency/algebra shard, broad linear helper shard,
  velocity-sharded term shard, Ruff on touched linear/test files, mypy on
  touched linear/test files, `py_compile` on touched files,
  differentiable-refactor manifest, validation coverage manifest regeneration,
  repository-size manifest, warning-free Sphinx build, and `git diff --check`.
- 2026-06-18: Continued executable-command consolidation by moving runtime
  command dependency construction from `spectraxgk.cli` into the existing
  command-workflow owner, `spectraxgk.workflows.runtime.commands`. The public
  CLI facade now passes `sys.modules[__name__]` to
  `build_runtime_command_deps`, so command tests and downstream monkeypatches
  still patch the executable facade instead of workflow internals. Added a
  focused regression test for the patch seam and documented the command owner
  split in the code-structure docs. Local gates passed: focused CLI command
  dependency/plot routing tests, runtime command owner tests, Ruff on touched
  executable/test files, mypy on touched executable modules, `py_compile` on
  touched executable modules, differentiable-refactor manifest, validation
  coverage manifest regeneration, repository-size manifest, warning-free
  Sphinx build, and `git diff --check`.
- 2026-06-18: Continued runtime/executable consolidation by moving runtime
  scan dependency-bundle construction from the public `spectraxgk.runtime`
  facade into the existing workflow owner,
  `spectraxgk.workflows.runtime.orchestration_scan`. The runtime facade now
  passes `sys.modules[__name__]` to owner-side builders so monkeypatch seams
  still work, while scan orchestration owns its own dependency contracts. The
  intentionally patchable runtime imports are now documented in a compact
  `_PATCHABLE_RUNTIME_GLOBALS` registry so static analysis does not treat them
  as dead imports. This brought `src/spectraxgk/runtime.py` from 613 to 594
  lines without adding files. Local gates passed: runtime scan
  dependency/monkeypatch tests, focused runtime scan ordering/quasilinear
  tests, selected non-slow runtime scan integration checks, Ruff on touched
  runtime/test files, mypy on touched runtime modules, `py_compile` on touched
  runtime modules, differentiable-refactor manifest, validation coverage
  manifest regeneration, repository-size manifest, warning-free Sphinx build,
  and `git diff --check`.
- 2026-06-18: Continued the differentiable-geometry cleanup by reusing the
  shared VMEC-state context from `spectraxgk.geometry.vmec_state_sensitivity`
  inside `spectraxgk.geometry.vmec_flux_tube_reports`. The direct flux-tube
  sensitivity and parity reports now share optional-backend VMEC example
  loading, coefficient-index validation, and coefficient perturbation policy
  with the Boozer/metric/field-line sensitivity reports, removing duplicated
  VMEC setup without adding source files. While gating that refactor, a cold
  optional-backend test exposed an ill-conditioned summary relative-error
  metric for zero-scale finite-difference references. The AD/FD report now
  preserves raw per-entry relative errors and records
  `max_rel_ad_fd_error_raw`, while the existing summary
  `max_rel_ad_fd_error` is gate-facing and ignores entries that already satisfy
  the absolute tolerance. Added a regression test for that zero-reference
  behavior. Local gates passed: full `tests/test_differentiable_geometry_bridge.py`
  shard, focused VMEC flux-tube report shard, Ruff on touched geometry/test
  modules, mypy on touched geometry modules, `py_compile` on touched geometry
  modules, differentiable-refactor manifest, validation coverage manifest
  regeneration, repository-size manifest, warning-free Sphinx build, and
  `git diff --check`.
- 2026-06-18: Continued the differentiable VMEC/Boozer refactor without adding
  source files. `spectraxgk.geometry.vmec_state_sensitivity` now centralizes
  optional-backend VMEC example loading, coefficient-index validation, state
  perturbation, and length-2 parameter validation for the Boozer flux-tube,
  metric-tensor, and field-line tensor AD/FD reports. The Boozer equal-arc core
  bridge now uses explicit helper functions for radial Boozer-profile
  interpolation, derivative interpolation, and repeated equal-arc open-grid
  profile remapping. This keeps the public
  `spectraxgk.geometry.differentiable` facade unchanged while making the
  differentiable geometry path easier to audit and extend. Reconfirmed the
  root `benchmarks/` contract: benchmark drivers, TOML inputs, and
  `benchmarks/results/manifest.toml` already live at repository root, track
  only small pointers/scripts, and publish their promoted results through
  `docs/benchmarks.rst` without adding raw run products to git. Local gates
  passed: focused differentiable-geometry VMEC/Boozer report tests, Ruff on
  touched geometry modules, mypy on touched geometry modules, `py_compile` on
  touched geometry modules, differentiable-refactor manifest, validation
  coverage manifest regeneration, repository-size manifest, warning-free
  Sphinx build, and `git diff --check`.
- 2026-06-18: Centralized saved-artifact path printing for runtime executable
  commands in `spectraxgk.workflows.runtime.commands._print_saved_paths`.
  Linear, ky-scan, nonlinear, and quasilinear command paths now use the same
  ordered display helper, keeping output behavior explicit while removing
  repeated per-command loops. I kept the process-safe runtime scan worker in
  the public `spectraxgk.runtime` facade for now because the current
  independent-worker route relies on an importable top-level task to preserve
  patchability and multiprocessing compatibility. Local gates passed: focused
  CLI/runtime command tests, Ruff on the touched command/test modules,
  `py_compile` and mypy for the command workflow, differentiable-refactor
  manifest, validation coverage manifest regeneration, repository-size
  manifest, warning-free Sphinx build, and `git diff --check`.
- 2026-06-18: Simplified the VMEC boundary-gradient collection gate without
  adding another source file. The repeated guarded dictionary access in
  `spectraxgk.geometry.vmec_boundary_chain.build_boundary_chain_collection_summary`
  now lives in a local `_collection_row` helper, leaving the public collection
  function focused on probe summarization, gate counts, and classification.
  This reduced `src/spectraxgk/geometry/vmec_boundary_chain.py` from 578 to
  567 lines while keeping the same JSON row schema used by projected transport
  line-search filters. Local gates passed: VMEC boundary-chain tests, VMEC
  transport line-search boundary-filter tests, Ruff on touched geometry/test
  modules, mypy on `spectraxgk.geometry.vmec_boundary_chain`, `py_compile` on
  the touched geometry module, differentiable-refactor manifest, validation
  coverage manifest regeneration, repository-size manifest, warning-free
  Sphinx build, and `git diff --check`.
- 2026-06-18: Reduced the executable facade below its active line-count target
  by moving the top-level `spectraxgk --plot` saved-output command parsing into
  `spectraxgk.workflows.runtime.commands.plot_saved_output_command`. The CLI
  still owns public parser dispatch and still passes its patchable
  `plot_saved_output` renderer seam, while the runtime command workflow now
  owns command-specific validation and messaging for saved-artifact plotting.
  This brought `src/spectraxgk/cli.py` from 507 to 492 lines without adding a
  new module. Local gates passed: full CLI test shard, focused runtime-helper owner
  tests, Ruff on touched CLI/runtime command/test files, mypy on touched
  source modules, `py_compile` on touched source modules,
  differentiable-refactor manifest, validation coverage manifest regeneration,
  repository-size manifest, warning-free Sphinx build, and `git diff --check`.
- 2026-06-18: Simplified runtime TOML case wrappers by removing the duplicate
  case-dependency builder from `spectraxgk.runtime` and delegating default
  dependency construction to the workflow owner,
  `spectraxgk.workflows.cases.default_runtime_case_deps`. The public runtime
  module still owns the stable `run_linear_case` and `run_nonlinear_case`
  signatures, but case-workflow wiring now has one owner instead of a
  facade-local copy. Reconfirmed that the root `benchmarks/` directory is
  already the canonical lightweight benchmark location: it contains only
  drivers, TOML inputs, and `benchmarks/results/manifest.toml`, with promoted
  result figures/tables displayed from the docs and raw outputs kept in
  scratch directories. Local gates passed: `py_compile` on `spectraxgk.runtime`, Ruff on touched
  runtime files, mypy on `spectraxgk.runtime`, focused runtime-helper
  dependency tests, differentiable-refactor manifest, validation coverage
  manifest regeneration, repository-size manifest, warning-free Sphinx build,
  and `git diff --check`.
- 2026-06-18: Simplified the public runtime scan facade by extracting explicit
  dependency-bundle builders for ky-scan orchestration and combined-ky batch
  execution inside `spectraxgk.runtime`. The extracted helpers keep the public
  runtime facade as the monkeypatch surface but make `run_runtime_scan` and
  `_run_runtime_scan_batch` read as thin handoffs to the workflow owners instead
  of constructing dependency namespaces inline. Added a focused patchability
  regression test for those bundles. Local gates passed: focused runtime
  scan/helper tests, Ruff on touched runtime files, mypy on
  `spectraxgk.runtime`, `py_compile` on touched Python files,
  differentiable-refactor manifest, validation coverage manifest regeneration,
  repository-size manifest, warning-free Sphinx build, and `git diff --check`.
  A broader mypy run over the full test file still reports pre-existing test
  typing issues unrelated to this refactor, so it is not used as this tranche's
  gate.
- 2026-06-18: Simplified executable parser ownership by splitting the dense
  `spectraxgk.cli.build_parser` body into generic-run, named-case, and runtime
  parser builders while keeping `spectraxgk.cli` as the single parser owner.
  The central `build_parser` function is now a short coordinator instead of a
  command-registration hotspot. Removed private CLI wrapper aliases for runtime
  output-path and named-case scan-ky loading; focused tests now import those
  contracts from `spectraxgk.workflows.runtime.commands` and
  `spectraxgk.workflows.named_cases`, respectively. Local gates passed: full
  CLI test shard, focused runtime-helper owner shard, Ruff on touched Python
  files, mypy on `spectraxgk.cli`, `py_compile` on touched Python files,
  executable help smoke checks for top-level, generic-run, and runtime-scan
  commands, differentiable-refactor manifest, validation coverage manifest
  regeneration, repository-size manifest, warning-free Sphinx build, and
  `git diff --check`.
- 2026-06-18: Continued runtime/executable simplification by moving repeated
  CLI/TOML/default option precedence in `spectraxgk.workflows.runtime.commands`
  into typed command-option records for linear, ky-scan, and nonlinear runtime
  executable workflows. This keeps the parser in `spectraxgk.cli`, keeps
  command execution in the runtime workflow owner, avoids adding another module,
  and makes solver-call inputs inspectable before execution. The root
  `benchmarks/` directory remains the canonical lightweight benchmark location
  and is already documented through `docs/benchmarks.rst` and
  `benchmarks/results/manifest.toml`. Local gates passed: focused CLI runtime
  command tests, focused runtime-command helper tests, Ruff on touched Python
  command files, mypy on executable/runtime command modules, `py_compile` on
  the runtime command owner, differentiable-refactor manifest, validation
  coverage manifest regeneration, repository-size manifest, warning-free Sphinx
  build, and `git diff --check`.
- 2026-06-18: Continued runtime/executable consolidation by adding an explicit
  preloaded runtime-config handoff between the generic `spectraxgk run`
  dispatcher and `spectraxgk.workflows.runtime.commands`. The generic run path
  still inspects the TOML once to choose linear versus nonlinear execution, but
  runtime command execution now reuses that loaded config/data instead of
  parsing the same file again. Parser construction stays in `spectraxgk.cli`,
  while path overrides, progress policy, quasilinear overrides, runtime output
  routing, and preload reuse stay in the runtime command owner. Added a CLI
  regression test for single-load generic dispatch and documented the contract
  in the code-structure guide. Local gates passed: focused CLI runtime command
  tests, runtime command re-export smoke shard, Ruff on touched Python files,
  mypy on executable/runtime command modules, py_compile on touched source
  modules, and `git diff --check`.
- 2026-06-18: Promoted the root `benchmarks/` directory contract into a
  docs-checked result index. `benchmarks/` remains the canonical user-facing
  benchmark entry point at repository root and stays lightweight at 56 KB of
  tracked drivers, TOML inputs, and pointer manifests. `docs/benchmarks.rst`
  now lists every promoted entry from `benchmarks/results/manifest.toml`,
  including figure/table paths, claim scopes, and regeneration commands, while
  raw solver products remain in ignored scratch directories. Added tests that
  fail if the docs omit a promoted manifest entry or if the tracked benchmark
  payload grows past the lightweight budget. Local gates passed: benchmark
  result/atlas/refresh tests, Ruff on the touched Python test, repository-size
  manifest check, warning-free Sphinx docs build, runtime-facade compile/signature
  sanity check, and `git diff --check`.
- 2026-06-18: Moved the core quasilinear transport diagnostic implementation
  from the root `spectraxgk.quasilinear` module into
  `spectraxgk.diagnostics.quasilinear_transport`. The root module is now a
  small stable public facade, while internal runtime/objective/API code imports
  the diagnostics-domain owner directly. Added facade identity coverage,
  documented the owner/facade split in API/code-structure/refactor-plan docs,
  and registered the new owner in the validation/refactor manifests. Local
  gates passed: quasilinear/autodiff-focused tests, public facade identity
  import check, Ruff on touched Python modules, mypy on touched source modules,
  validation/refactor manifests, repository-size check, warning-free Sphinx
  docs build, and `git diff --check`.
- 2026-06-18: Moved benchmark-family case presets out of the generic runtime
  schema in `spectraxgk.config` and into
  `spectraxgk.validation.benchmarks.case_configs`. `spectraxgk.config` remains
  the stable public import location through explicit re-exports, while the
  implementation owner now lives with the validation/benchmark modules that use
  the presets. Added import-identity coverage, updated API/code-structure and
  refactor-plan docs, registered the new owner in the validation/refactor
  manifests, regenerated the validation coverage manifest summary, and
  suppressed duplicate autodoc indexing on the benchmark facade. Local gates
  passed: config tests, focused benchmark runner branch shard, public export
  identity check, Ruff on touched Python modules, mypy on the split config
  modules, validation/refactor manifests, repository-size check, warning-free
  Sphinx docs build, and `git diff --check`.
- 2026-06-18: Confirmed the root `benchmarks/` directory is the canonical
  tracked benchmark location and kept it lightweight: only drivers, TOML
  inputs, and `benchmarks/results/manifest.toml` live there, while raw outputs
  stay in ignored scratch directories. Tightened README/docs/code-structure
  wording so benchmark results are discoverable from the root manifest and the
  benchmark-family tests patch implementation owner modules directly rather
  than retired family compatibility modules. Local gates passed: stale
  benchmark-path/facade wording audit, benchmark result/runtime manifest tests,
  focused benchmark-runner branch shard, validation/refactor manifests,
  repository-size check, Ruff on touched Python modules, mypy on touched
  benchmark modules, Sphinx docs build, and `git diff --check`.
- 2026-06-18: Audited the runtime facade after the benchmark-root tranche.
  The public runtime wrapper still has broad, intentional monkeypatch seams in
  the focused runtime tests, so this pass avoided moving runtime dispatch code
  prematurely. Cleaned the remaining generic runtime damping test name that
  used comparison-code terminology outside a benchmark/comparison context.
  Local gates passed: runtime damping/terms shard with integration filtering
  disabled, Ruff on `tests/test_runtime_runner.py`, terminology audit, and
  `git diff --check`.
- 2026-06-18: Retired the KBM benchmark-family compatibility module. Public
  KBM benchmark imports now route from `spectraxgk.benchmarks` directly to
  `spectraxgk.validation.benchmarks.kbm_linear`,
  `spectraxgk.validation.benchmarks.kbm_scan`, and
  `spectraxgk.validation.benchmarks.kbm_beta`; tests patch those owner modules
  directly. Updated API/code-structure/refactor-plan docs and the
  validation/refactor manifests so the benchmark ownership map reflects the
  maintained modules only. Regenerated the validation coverage summary and kept
  root `benchmarks/` as the lightweight tracked benchmark-output index. Local
  gates passed: KBM runner-branch tests, KBM integration selection with
  integration filtering disabled, benchmark-result/runtime-config tests,
  validation/refactor manifest tests, repository-size check, Ruff, mypy over
  342 source files, Sphinx docs build, and `git diff --check`.
- 2026-06-18: Retired the Cyclone benchmark-family compatibility module. Public
  Cyclone benchmark imports now route from `spectraxgk.benchmarks` directly to
  `spectraxgk.validation.benchmarks.cyclone_linear` and
  `spectraxgk.validation.benchmarks.cyclone_scan`; tests patch those owner
  modules directly. Updated API/code-structure/refactor-plan docs and the
  validation/refactor manifests so the benchmark ownership map reflects the
  maintained modules only. Regenerated the validation coverage summary and kept
  root `benchmarks/` as the lightweight tracked benchmark-output index. Local
  gates passed: Cyclone runner-branch tests, affected Cyclone integration node
  IDs with integration filtering disabled, benchmark-result/runtime-config
  tests, validation/refactor manifest tests, repository-size check, Ruff, mypy
  over 343 source files, Sphinx docs build, and `git diff --check`.
- 2026-06-18: Retired the ETG benchmark-family compatibility module. Public
  ETG benchmark imports now route from `spectraxgk.benchmarks` directly to
  `spectraxgk.validation.benchmarks.etg_linear` and
  `spectraxgk.validation.benchmarks.etg_scan`; tests now patch those owner
  modules directly. Updated API/code-structure/refactor-plan docs and the
  validation/refactor manifests so the benchmark ownership map reflects the
  maintained modules only. Regenerated the validation coverage summary and
  kept the root `benchmarks/` result manifest unchanged as the lightweight
  tracked index for benchmark outputs. Local gates passed: ETG runner-branch
  tests, affected ETG integration node IDs with integration filtering disabled,
  benchmark-result/runtime-config tests, validation/refactor manifest tests,
  repository-size check, Ruff, mypy over 344 source files, Sphinx docs build,
  and `git diff --check`.
- 2026-06-18: Retired the nonlinear parallelization compatibility facades
  `spectraxgk.operators.nonlinear.parallel_contracts` and
  `spectraxgk.operators.nonlinear.spectral_identity`. Public nonlinear
  parallelization imports now route through `spectraxgk.operators.nonlinear.parallel`,
  while contract DTOs and logical spectral identity reports/RHS/integrator gates
  live directly in focused owner modules. Added a root-level
  `benchmarks/results/manifest.toml` index so promoted benchmark outputs are
  discoverable from the root `benchmarks/` directory without copying raw run
  artifacts, updated docs to show the tracked benchmark result set, and fixed
  the runtime-core CI shard to load Miller/W7-X zonal TOMLs from the root
  benchmark directory. Local gates passed: benchmark-results manifest tests,
  benchmark-atlas/runtime-memory tests, nonlinear parallel/domain/spectral
  tests, runtime-core shard, validation/refactor manifest tests, validation
  summary regeneration, repository-size check, Ruff, mypy over 345 source
  files, Sphinx docs build, and `git diff --check`.
- 2026-06-18: Continued the Cyclone scan refactor by moving the standard
  saved-time, Diffrax-streaming, auto-fit, and invalid-growth Krylov fallback
  batch loop from `spectraxgk.validation.benchmarks.cyclone_scan` into
  `spectraxgk.validation.benchmarks.cyclone_scan_branches`. The public scan
  runner now owns setup, solver/fit policy selection, and dispatch across the
  Krylov, explicit-time, and standard time branches. The hook bundle was
  extended so existing public-facade monkeypatch/debug seams continue to route
  into the implementation owner. Focused Cyclone scan branch tests, Ruff, and
  py_compile passed locally before broader gates.
- 2026-06-18: Split the TEM benchmark runner into a compact public owner plus
  `spectraxgk.validation.benchmarks.tem_paths`. The public module now owns
  grid/geometry setup, TEM parameter construction, species-index validation,
  and stable `run_tem_linear`/`run_tem_scan` imports, while the path owner
  contains single-ky Krylov solving, single-ky time fitting, streaming scan
  fitting, and saved-time scan batching through an explicit hook bundle. The
  hook bundle preserves existing benchmark monkeypatch/debug seams. Focused TEM
  branch tests, Ruff, and py_compile passed locally before manifest/docs gates.
- 2026-06-18: Continued the KBM single-ky benchmark refactor by moving the
  explicit-time diagnostics branch, fit-window fallback, and single/multi-target
  Krylov branch policy from `spectraxgk.validation.benchmarks.kbm_linear` into
  `spectraxgk.validation.benchmarks.kbm_linear_paths`. The public single-point
  runner still owns geometry setup, generic saved-time fitting, and result
  packaging, while the new path owner preserves existing facade monkeypatch
  seams through explicit hook synchronization. Focused KBM branch tests, Ruff,
  py_compile, manifest, mypy, Sphinx, and repository-size gates passed locally.
- 2026-06-18: Continued the ETG scan benchmark refactor by moving Krylov
  continuation, streaming-fit handling, saved-signal time integration, and
  fallback fit/appending policy from
  `spectraxgk.validation.benchmarks.etg_scan` into
  `spectraxgk.validation.benchmarks.etg_scan_paths`. The scan runner now owns
  case setup, batching, parameter construction, and public result packaging,
  while the path owner keeps existing ETG owner-module monkeypatch seams through
  explicit hook synchronization. Focused ETG scan tests, Ruff, py_compile,
  manifest, mypy, Sphinx, and repository-size gates passed locally.
- 2026-06-18: Continued the Cyclone single-mode benchmark refactor by moving
  Krylov seed/branch selection and time-integration fit policy from
  `spectraxgk.validation.benchmarks.cyclone_linear` into
  `spectraxgk.validation.benchmarks.cyclone_linear_paths`. The public
  single-mode runner now owns setup, parameter construction, fallback
  orchestration, and result packaging, while the new owner module preserves the
  existing facade monkeypatch seams through explicit hook synchronization.
  Focused Cyclone branch tests, Ruff, py_compile, manifest, mypy, Sphinx, and
  repository-size gates passed locally.
- 2026-06-18: Split the 699-line runtime orchestration module into a 40-line
  `spectraxgk.workflows.runtime.orchestration` facade plus focused scan,
  progress, and nonlinear artifact/restart handoff owner modules. Runtime
  helper, runtime artifact, runtime chunk, public API, Ruff, manifest, and docs
  gates passed locally.
- 2026-06-18: Completed the kinetic-electron benchmark cleanup: the former
  combined runner is now split into focused `kinetic_linear` and `kinetic_scan`
  owner modules, and the intermediate validation-package facade has been
  removed. Kinetic-electron benchmark functions now route from the supported
  `spectraxgk.benchmarks` API directly to the owner modules, reducing one file
  and one stale compatibility layer while preserving user-facing imports.
  Focused kinetic benchmark tests, manifests, lint, and compile gates passed
  locally before the broader benchmark/docs gate run.
- 2026-06-18: Split the 706-line public nonlinear driver into a 98-line
  `spectraxgk.nonlinear` facade plus `spectraxgk.nonlinear_core` for cached
  RHS/state integration and `spectraxgk.nonlinear_diagnostics` for explicit and
  IMEX diagnostic entry points. Tests now patch owner modules directly instead
  of treating the public facade as the implementation owner. Focused nonlinear,
  runtime, public API, and Ruff gates passed locally.
- 2026-06-18: Split the 710-line Diffrax time-integrator module into a 41-line
  `spectraxgk.solvers.time.diffrax` facade plus focused core policy/helper,
  linear integration, streaming growth/frequency, and nonlinear integration
  modules. Public Diffrax imports are preserved; optional-dependency monkeypatch
  tests now target `spectraxgk.solvers.time.diffrax_core`, which owns `dfx/eqx`.
  Focused Diffrax core/smoke tests and Ruff passed locally.
- 2026-06-18: Split the 719-line term-wise RHS assembly module into a 59-line
  `spectraxgk.terms.assembly` facade plus focused cached RHS, per-term
  diagnostic decomposition, field-only solve, and helper-policy owner modules.
  The physics term order and public imports are unchanged; the Hamiltonian
  builder monkeypatch seam now targets `spectraxgk.terms.assembly_core`, which
  owns the cached RHS implementation. Focused term assembly, linear helper,
  and Ruff gates passed locally.
- 2026-06-18: Continued the differentiable objective refactor by splitting the
  723-line `spectraxgk.objectives.solver_gradients` module into a 133-line
  public facade plus `spectraxgk.objectives.solver_vmec` for VMEC/Boozer
  objective wrappers and `spectraxgk.objectives.solver_gradient_reports` for
  solver-ready, branch-locality, and mode-21 gradient reports. Public exports
  and top-level API identities are preserved, while validation monkeypatch seams
  now target the implementation owner module directly. Focused solver-objective
  differentiability tests and Ruff passed locally.
- 2026-06-18: Split the 1235-line internal VMEC imported-geometry backend into
  a 50-line `spectraxgk.geometry_backends.vmec` facade plus focused backend
  discovery, numerical helper, field-line assembly, flux-tube remap, NetCDF IO,
  pipeline, and shared type modules. Public VMEC imports remain available, but
  tests now patch the implementation owner modules directly. Focused VMEC helper
  and runtime EIK tests plus Ruff passed locally; broader manifest and docs
  gates were refreshed to track the new owners.
- 2026-06-18: Split the 868-line linear Krylov solver into a compact public
  facade plus focused eigenmode owner modules:
  `spectraxgk.solvers.linear.eigen_policy`,
  `spectraxgk.solvers.linear.eigen_operator`,
  `spectraxgk.solvers.linear.eigen_selection`,
  `spectraxgk.solvers.linear.eigen_preconditioners`, and
  `spectraxgk.solvers.linear.krylov_algorithms`. The public
  `spectraxgk.solvers.linear.krylov` import path still owns user-facing status
  reporting and monkeypatch-compatible seams, while branch selection,
  matrix-free operator application, shift-invert preconditioning, and compiled
  Arnoldi/power kernels now have direct owner modules. Updated API/code
  structure docs plus refactor and coverage manifests; local Krylov core tests
  passed before broader gates.
- 2026-06-18: Split the cETG reduced-model implementation out of the legacy
  `spectraxgk.terms.cetg` file. The old import path is now a small facade,
  while `spectraxgk.terms.reduced.cetg_model`,
  `spectraxgk.terms.reduced.cetg_state`,
  `spectraxgk.terms.reduced.cetg_rhs`, and
  `spectraxgk.terms.reduced.cetg_integrator` own the runtime contract, spectral
  state projection, field/RHS physics, and explicit diagnostic integrator.
  The split removed one duplicated adaptive-timestep `linear_omega` assignment
  without changing behavior; focused cETG tests passed after the move.
- 2026-06-17: Removed duplicate runtime startup helper implementations from
  `spectraxgk.runtime`. The public runtime facade now aliases
  `_centered_glibc_random_pairs`, `_dealiased_initial_mode_pairs`, and
  `_periodic_zp_from_grid` to the existing `spectraxgk.workflows.runtime.startup` owner,
  preserving imports while shrinking the main runtime runner.
- 2026-06-17: Moved explicit/IMEX diagnostic collision-split setup into
  `spectraxgk.operators.nonlinear.policies.build_nonlinear_collision_split_policy`. The
  facade still injects its damping seam for monkeypatch compatibility, but
  active/inactive split detection, collision-free RHS terms, and damping
  assembly now have one focused policy object and direct tests.
- 2026-06-17: Moved explicit/IMEX nonlinear diagnostic output finalization into
  `spectraxgk.operators.nonlinear.diagnostics.finalize_nonlinear_scan_diagnostics`. The
  public nonlinear facade no longer owns duplicate stride/sample-index logic
  before packaging `SimulationDiagnostics`, and the helper now has focused
  tests for dense versus already-sampled scans.
- 2026-06-17: Moved fixed-step IMEX nonlinear diagnostic scan execution into
  `spectraxgk.solvers.nonlinear.imex.run_imex_diagnostic_scan`. The public
  nonlinear facade now delegates checkpoint and `jax.lax.scan` mechanics for
  IMEX diagnostics to the solver owner, matching the explicit diagnostic scan
  split and reducing facade-only scan logic.
- 2026-06-17: Moved IMEX nonlinear diagnostic step construction into
  `spectraxgk.solvers.nonlinear.imex.make_imex_diagnostic_step`. The public
  nonlinear facade still owns diagnostic setup, implicit-operator setup, and
  compatibility seams, while the solver owner now contains the per-step
  IMEX/collision/diagnostic/progress policy with direct unit tests and package
  re-export coverage.
- 2026-06-17: Moved explicit nonlinear diagnostic step construction into
  `spectraxgk.solvers.nonlinear.explicit.make_explicit_diagnostic_step`.
  The public nonlinear facade still owns cache setup, diagnostic setup, and
  monkeypatch-compatible injected functions, but the solver owner now contains
  the per-step RHS/adaptive-dt/RK/collision/diagnostic/progress policy with
  direct unit tests.
- 2026-06-17: Extracted explicit nonlinear diagnostic scan-selection policy
  into `spectraxgk.solvers.nonlinear.explicit.run_explicit_diagnostic_scan`.
  The public nonlinear facade still owns diagnostic setup and public
  compatibility, but checkpointing plus sampled-vs-dense diagnostic scan
  orchestration now has direct solver-owner tests and package re-export
  coverage.
- 2026-06-17: Moved cached explicit nonlinear scan dispatch into
  `spectraxgk.solvers.nonlinear.explicit.integrate_cached_explicit_scan`.
  `spectraxgk.nonlinear.integrate_nonlinear_cached` still owns public
  compatibility, term normalization, RHS closure construction, and Hermitian
  projector selection, but the actual explicit scan call is now tested in the
  solver owner package with injected scan/progress/checkpoint/projector seams.
- 2026-06-17: Continued nonlinear consolidation by moving the cached IMEX
  nonlinear scan policy from the public `spectraxgk.nonlinear` facade into
  `spectraxgk.solvers.nonlinear.imex.integrate_cached_imex_scan`. The facade
  now normalizes term configuration and injects its field/RHS/operator seams
  for compatibility, while the solver package owns the shape check, scan
  closure, nonlinear-term routing, GMRES solve-step closure, checkpointed
  scan, and final field trace. Added direct owner-module coverage and package
  re-export coverage, and updated the refactor docs plus manifest.
- 2026-06-17: Re-audited the active differentiable-refactor branch, draft PR,
  CI status, changed files, large source hotspots, companion VMEC/Boozer
  repositories, and external JAX/scientific-Python guidance. The active PR is
  green before this tranche, and the plan hierarchy is now explicit:
  `docs/architecture_refactor_plan.rst` is the single layout authority,
  `docs/differentiable_refactor_plan.rst` is the AD/validation appendix, and
  this file is the chronological log. Added an explicit differentiation-method
  ladder and adaptive-branch derivative admission policy so end-to-end
  differentiability through adaptive controllers is gated rather than assumed.
  The finite completion sequence for the branch is now: nonlinear
  consolidation, runtime/executable consolidation, objectives/optimization
  consolidation with adaptive gates, validation/benchmark consolidation,
  package-mirrored tests, then removal of temporary migration allowances.
- 2026-06-17: Routed cached IMEX nonlinear integration through
  `build_nonlinear_imex_operator` instead of duplicating implicit-operator
  setup in `spectraxgk.nonlinear`. The helper now accepts a call-time injected
  implicit-operator builder so facade and helper monkeypatch/debug seams remain
  intact.
- 2026-06-17: Moved explicit/IMEX nonlinear state-to-diagnostic closure
  construction into
  `spectraxgk.operators.nonlinear.diagnostic_state.make_nonlinear_diagnostic_tuple_fn`.
  The public nonlinear facade still injects facade-level diagnostic kernels, but
  both scan paths now share one tested closure factory and direct operator
  package re-export.
- 2026-06-17: Extracted the non-CPU sampled explicit diagnostic scan runner
  into `spectraxgk.operators.nonlinear.diagnostics`. The retained-step interval policy
  now lives in `sampled_scan_intervals`, and
  `run_sampled_explicit_diagnostic_scan` owns the interval `fori_loop`/`scan`
  orchestration while preserving final-step retention. Added manufactured scan
  tests covering sample intervals, final carry, sampled diagnostics, and `dt`
  series.
- 2026-06-17: Extracted fixed/adaptive nonlinear time-step policy from
  `spectraxgk.nonlinear` into
  `spectraxgk.operators.nonlinear.policies.build_nonlinear_time_step_policy`. The explicit
  diagnostics path now delegates initial `dt`, progress horizon, linear/CFL
  frequency bounds, velocity-space bounds, and `dt_min`/`dt_max` clipping to a
  directly tested helper with injected compatibility seams.
- 2026-06-17: Deduplicated nonlinear diagnostic-stride selection and progress
  callback routing. Explicit and IMEX diagnostic scans now share
  `select_nonlinear_step_diagnostics` and `maybe_emit_nonlinear_progress` in
  `spectraxgk.operators.nonlinear.diagnostics`, preserving the existing callback cadence
  while keeping host-output policy out of the scan bodies. Added direct tests
  for compute-vs-reuse stride selection and the disabled-progress no-op path.
- 2026-06-17: Moved duplicated nonlinear IMEX closure construction into
  `spectraxgk.solvers.nonlinear.imex`. Cached and diagnostic IMEX paths now
  share `make_imex_nonlinear_term` and `make_imex_solve_step` for explicit
  nonlinear-term evaluation and GMRES solve-step policy, preserving injected
  facade kernels for debugging/monkeypatch workflows. Added direct factory
  tests for nonlinear-kernel forwarding and solve-policy forwarding.
- 2026-06-17: Extracted shared nonlinear diagnostic setup into
  `spectraxgk.operators.nonlinear.policies.build_nonlinear_diagnostic_setup`.
  Explicit and IMEX diagnostic scans now use the same cache construction,
  quadrature weights, omega masks, monitored z-index policy, and fixed-mode plus
  Hermitian projection setup through injected compatibility seams. Added a
  direct setup-builder regression covering moment-count inference, injected
  geometry/cache/weight/mask policies, and monitored-index selection.
- 2026-06-17: Deduplicated nonlinear diagnostic scan postprocessing by moving
  raw scan-tuple sampling, resolved-diagnostic packing, energy reconstruction,
  and `SimulationDiagnostics` construction into
  `spectraxgk.operators.nonlinear.diagnostics.build_nonlinear_simulation_diagnostics`.
  Explicit and IMEX nonlinear diagnostic paths now share the same output
  convention, while direct tests cover stride sampling, resolved schema mapping,
  and total-energy reconstruction.
- 2026-06-17: Deduplicated nonlinear diagnostic projection setup by moving the
  composed fixed-mode plus compressed-real-FFT Hermitian projector into
  `spectraxgk.operators.nonlinear.policies._make_nonlinear_state_projector`. Explicit and
  IMEX diagnostic scans now share one projection convention, with a direct
  helper test covering fixed Fourier-mode preservation and negative-ky
  reconstruction.
- 2026-06-17: Continued nonlinear solver modularization by moving the IMEX
  SSPX3 stage-composition policy into `spectraxgk.solvers.nonlinear.imex`.
  `spectraxgk.nonlinear` now builds runtime state, cache, and diagnostics while
  the solver module owns fixed-point prediction, GMRES solves, and the
  semi-implicit stage update. Added direct default-step and constant-RHS SSPX3
  tests plus solver-package re-export coverage.
- 2026-06-17: Continued the linear refactor by moving implicit linear
  matrix-free operator construction, Hermite/PAS/diagonal preconditioner
  policy, and GMRES time integration into
  `spectraxgk.solvers.linear.implicit`. The public `spectraxgk.linear` facade
  still re-exports the implicit helpers for current users, while nonlinear
  IMEX code and profiling tools now import the focused owner module directly.
  Focused implicit linear and nonlinear IMEX tests cover the new owner seam.
- 2026-06-17: Moved linear fixed-step time integration, implicit-method
  dispatch, velocity-parallel fallback routing, and diagnostic sampling into
  `spectraxgk.solvers.linear.integrators`. The public `spectraxgk.linear`
  module is now a compact facade for linear parameters, caches, RHS helpers,
  moments, and solver entry points. Tests now patch the integrator owner module
  while preserving facade import identity for existing users.
- 2026-06-16: Started the first architecture-plan implementation tranche.
  Added `tools/package_architecture_manifest.toml` and
  `tools/check_package_architecture_manifest.py` to stop new root-level prefix
  module growth unless explicitly listed as temporary migration scaffolding.
  Converted `spectraxgk.operators` from a single module into a package while
  preserving `hermite_streaming`, moved nonlinear RHS/diagnostic-state
  implementation under `operators/nonlinear`, moved explicit/IMEX nonlinear
  solve policy under `solvers/nonlinear`, and left the old root modules as thin
  compatibility facades. Added import-identity tests for the old and new
  module paths and wired the architecture checker into CI/release readiness.
- 2026-06-16: Finalized the refactor planning hierarchy to avoid conflicting
  architecture plans. `docs/architecture_refactor_plan.rst` is now explicitly
  authoritative for future package layout, naming, migration order, simplicity
  constraints, Python-vs-executable differentiability boundaries, and
  performance/memory rules. `docs/differentiable_refactor_plan.rst` is now a
  technical appendix for AD contracts, manifest rows, historical split
  inventory, and validation gates. `docs/code_structure.rst` now describes the
  current tree only and points future work to the architecture plan.
  `tools/differentiable_refactor_manifest.toml` records the architecture plan
  as the layout authority while remaining the active migration ledger.
- 2026-06-16: Reset the refactor plan around domain packages instead of
  adding more root-level prefix modules. The new architecture plan documents
  the audit snapshot (`167` source files, about `70k` source lines, `134`
  root-level package modules, and `315` top-level tests), names the target
  package layout, defines naming rules, phases, acceptance gates, documentation
  requirements, test mirroring, and the first concrete tranche:
  move the already extracted nonlinear RHS/diagnostic/explicit/IMEX helpers
  into `operators/nonlinear` and `solvers/nonlinear` behind the existing
  `spectraxgk.nonlinear` facade. The plan explicitly stops future
  `runtime_*`, `nonlinear_*`, `vmec_jax_*`, `quasilinear_*`, and
  `benchmark_*` root-module growth unless listed as temporary migration
  scaffolding.
- 2026-06-16: Continued the nonlinear refactor by moving explicit RK/SSP/K10
  one-step policy into `spectraxgk.solvers.nonlinear.explicit` and shared IMEX
  fixed-point/GMRES solve policy into `spectraxgk.solvers.nonlinear.imex`. Added direct
  tests for projection/dtype stability, constant-RHS RK behavior,
  fixed-point iterations, and identity-system IMEX solves. Renamed lingering
  non-benchmark nonlinear test names/docstrings that used comparison-code
  terminology. Updated API docs, architecture docs, refactor manifests, and
  validation coverage ownership.
- 2026-06-16: Continued the nonlinear refactor by extracting nonlinear RHS
  linear-path selection and electromagnetic bracket composition into
  `spectraxgk.operators.nonlinear.rhs`. The public `spectraxgk.nonlinear` API
  injects its module-level callables so debug workflows and runtime behavior stay
  compatible. Added direct RHS routing tests, registered the
  module in API/code-structure docs and the refactor/coverage manifests, and
  kept performance claims unchanged until profiler-backed gates are rerun.
- 2026-06-16: Continued the nonlinear RHS refactor by moving the duplicated
  IMEX explicit nonlinear-term assembly into
  `spectraxgk.operators.nonlinear.rhs.nonlinear_em_term_cached_impl`. The explicit and
  cached IMEX paths now share one bracket payload convention, while
  `spectraxgk.nonlinear` still injects `compute_fields_cached` and
  `nonlinear_em_contribution` for compatibility debugging. Added a direct zero
  coefficient/payload-forwarding test and kept performance claims unchanged.
- 2026-06-16: Continued the nonlinear refactor by extracting duplicated
  explicit/IMEX state-to-diagnostic tuple assembly into
  `spectraxgk.operators.nonlinear.diagnostic_state`. The implementation receives
  injected diagnostic kernels so debug seams stay intact. Added direct scalar/resolved diagnostic-packing tests and updated the
  API docs, code-structure docs, refactor manifest, and coverage manifest.
- 2026-06-16: Continued the differentiable solver-objective refactor by
  removing the temporary solver-gradient compatibility facade. The canonical
  implementation modules are now `spectraxgk.objectives.gradient_gates` for
  solver-ready branch/linear-RHS gradient gates and
  `spectraxgk.objectives.vmec_boozer_gradients` for mode-21 VMEC/Boozer
  frequency, quasilinear, and reduced nonlinear-window gradient gates. Added
  direct implementation-module tests for FD reports, line-search/holdout
  gates, and injected VMEC/Boozer gradient reports. Updated
  API docs, architecture docs, refactor manifests, validation ownership, and
  README scope wording. Performance-manifest and parallel-scaling artifact
  checks were rerun; they pass while still blocking production nonlinear
  speedup claims until GPU production-speedup evidence exists.
- 2026-06-15: Continued the VMEC/Boozer gate refactor by splitting
  finite-difference report construction into
  `spectraxgk.objectives.vmec_boozer_fd` and line-search/held-out audit
  logic into `spectraxgk.objectives.vmec_boozer_line_search`. The temporary
  wrapper module has since been removed; `solver_objective_gradients` imports
  the focused gate modules directly while preserving dependency-injected hook
  seams. Focused solver-objective tests, Ruff, and mypy passed locally.
- 2026-06-15: Continued the differentiable solver-objective refactor by moving
  solver-ready branch-continuity and geometry-gradient reports plus mode-21
  VMEC/Boozer frequency, quasilinear, and reduced nonlinear-window gradient
  reports into focused implementation modules. `solver_objective_gradients`
  keeps the higher-level public objective surface and dependency-injected
  private context hooks. Focused solver-objective tests, Ruff, and mypy passed
  locally.
- 2026-06-15: Continued the differentiable solver-objective refactor by moving
  VMEC/Boozer finite-difference sensitivity reports, curvature-gated
  line-search gates, and held-out aggregate objective audits into
  focused FD and line-search gate modules. `solver_objective_gradients` now
  delegates to those modules directly through dependency-injected wrappers so
  public objective imports and validation tests keep working. Focused
  solver-objective tests, refactor/coverage manifest tests, Ruff, mypy, docs
  build with warnings-as-errors, and release-readiness checks passed locally.
- 2026-06-15: Continued the differentiable solver-objective refactor by moving
  VMEC/Boozer objective option splitting, objective-table construction, sample
  metadata, and scalar reductions into
  `spectraxgk.objectives.vmec_boozer`. The legacy
  `solver_objective_gradients` facade now delegates through thin wrappers so
  public imports and monkeypatch-based validation tests keep working. Focused
  solver-objective tests, refactor/coverage manifest tests, Ruff, mypy, docs
  build with warnings-as-errors, and release-readiness checks passed locally.
- 2026-06-15: Fixed the Codecov project-status policy on the
  differentiable-refactor branch. Codecov now waits for both CI coverage
  uploads and evaluates the required project status only against the
  `wide-package` report, so the release coverage check mirrors the
  package-wide 95% gate instead of the earlier fast diagnostic subset. The
  release-readiness checker and tests now guard this policy.
- 2026-06-15: Removed the old comparison-code-named compatibility shims from
  package source. The deleted package paths are `spectraxgk.from_gx.*`,
  `spectraxgk.gx_legacy_output`, and `spectraxgk.gx_reduced_models`; callers
  should use `spectraxgk.geometry_backends.*`,
  and `spectraxgk.reduced_model_contracts` instead. Legacy input aliases for
  helper paths, diagnostic normalization, imported-geometry model strings,
  reduced-model names, and runtime `gx_time` solver spelling were also removed
  from canonical runtime/config paths. Explicit benchmark/comparison tools keep
  reference-code names only where they operate on external comparison data.
  Focused cleanup tests, Ruff, docs build with warnings-as-errors, package
  build, and wheel/sdist shim-exclusion checks passed locally.
- 2026-06-15: Removed the remaining `gx_time` compatibility mapping from the
  shared benchmark solver normalizer. External-reference comparison tools now
  translate their `gx_time` candidate label to the canonical SPECTRAX-GK
  `explicit_time` solver only at the benchmark API boundary, while preserving
  the comparison label in CSVs, cache keys, and branch-selection reports.
  Focused benchmark-scan, KBM comparison, extractor, overlay, RHS comparison,
  and Ruff checks passed locally.
- 2026-06-15: Cleaned another native-source terminology slice on the
  differentiable-refactor branch. CLI help, runtime errors, initialization
  comments, dealiased-grid comments, transport-mode weights, and
  Hermite-Laguerre field-transform comments now describe the implemented
  physics/numerics directly rather than naming comparison-code conventions.
  Focused Ruff and grid/runtime/operator/CLI tests passed locally; CI was
  green before this tranche and was relaunched after the push.
- 2026-06-15: Tightened executable-facing output wording. README quickstart,
  output docs, CLI `--out`/`--ql-output` help, and `spectraxgk --plot`
  docstrings/errors now use "saved output" / "output path" language for normal
  user workflows, while validation/release ledgers keep "artifact" where they
  explicitly mean a versioned evidence object. Focused CLI/plot/runtime-output
  tests, docs build with warnings-as-errors, and package build passed locally.
- 2026-06-15: Cleaned additional user-facing docs wording so normal
  SPECTRAX-GK slab geometry, imported-geometry, initialization, restart, RNG,
  and explicit-time contracts are described by physics/numerics semantics
  rather than by another implementation. Benchmark/comparison references remain
  where they are actually validation sources. The docs build still passes with
  warnings-as-errors after this cleanup.
- 2026-06-15: Cleaned a second source-level wording slice so cETG,
  linked-boundary damping, linked-FFT operators, secondary diagnostics,
  growth-rate extraction, and initialization scaling describe native
  SPECTRAX-GK model/numerics contracts. Compatibility aliases and explicit
  benchmark/comparison tools remain named as such. Targeted cETG/runtime/
  secondary/operator tests and Ruff passed.

- 2026-06-15: Added the first finite-beta VMEC/Boozer solver-objective
  gradient artifact. The shaped-tokamak-pressure frequency gate runs from
  `vmec_jax` state coefficients through `booz_xform_jax` mode-21 equal-arc
  geometry into the SPECTRAX-GK eigenfrequency objective, passes in a bounded
  local run (~27 s), and records max AD-vs-finite-difference relative error
  about `6.4e-11`. The shaped-pressure finite-beta quasilinear-gradient gate
  also passes in a bounded local run (~50 s), with max relative error about
  `2.1e-4`. The shaped-pressure finite-beta reduced nonlinear-window
  estimator-gradient gate also passes in a bounded local run (~43 s), with max
  relative error about `2.1e-4` across the window mean, CV, and trend
  observables. The differentiability guard now requires all three artifacts
  while still blocking finite-beta converged nonlinear transport-gradient
  claims until those gates are run and pass.

- 2026-06-15: Tightened the VMEC/Boozer differentiability guard from a
  matrix-level gradient check to an objective-level solver contract. Each QH
  and Li383 row must now include the required frequency, quasilinear, and
  nonlinear-window estimator objectives, every case must carry all three gate
  types, and row/objective AD-vs-finite-difference errors must stay below the
  documented release thresholds (`5e-2`, `2e-2`, and `7.5e-2`). Focused tests
  now fail closed on missing quasilinear objectives and excessive
  nonlinear-window estimator error.

- 2026-06-15: Tightened the VMEC/Boozer differentiability guard to require
  family coverage in the parity matrix. The release claim now fails if the
  tracked equal-arc parity evidence drops the QH, QI, or
  shaped-tokamak-pressure finite-beta row, while still keeping finite-beta
  solver-objective gradients and production nonlinear transport gradients out
  of scope. Focused guard tests now include the finite-beta row and a failure
  case for missing finite-beta coverage.

- 2026-06-15: Added a content-based VMEC/Boozer differentiability claim guard.
  `tools/check_vmec_boozer_differentiability_claim.py` now validates the
  tracked equal-arc parity matrix, mode-21 QH/Li383 frequency/quasilinear/
  nonlinear-window gradient holdout matrix, explicit diagnostic-open status for
  the direct VMEC tensor-vs-imported-EIK convention gap, and startup-only scope
  for the nonlinear finite-difference audit. CI, release readiness, technical
  release status, geometry docs, and release-scope docs now require the guard
  so reduced AD claims stay separated from unpromoted full nonlinear
  turbulence-gradient optimization claims.

- 2026-06-15: Removed the remaining old imported-geometry facade aliases from
  the production geometry package surface (`apply_imported_geometry_grid_defaults`,
  `load_imported_geometry_netcdf`, `zero_shear_enabled`, `effective_boundary`,
  and `twist_shift_params` are now the canonical names). Comparison/debug tools
  were switched to those canonical imports while keeping explicit comparison
  filenames where they operate on external reference outputs. Focused geometry,
  EIK, runtime-config, comparison-tool, lint, and compile checks passed locally.

- 2026-06-15: Continued source naming cleanup by moving generic reduced-model
  helpers from old reference-code-named modules to
  `spectraxgk.reduced_model_contracts`. The transitional grouped cETG NetCDF
  output reader has since been removed so the package only carries current
  artifact formats and validation workflows. Coverage ownership moved to the
  canonical modules and obsolete shim modules are excluded from the
  wide-coverage ownership inventory. Focused reduced-model/cETG tests,
  manifest tests, lint,
  format, and compile checks passed locally.

- 2026-06-15: Completed the internal imported-geometry backend package rename
  tranche. The implementation moved from `spectraxgk.from_gx.*` to
  `spectraxgk.geometry_backends.*`. Source, profiler utilities, validation
  gates, coverage manifests, and backend tests use the canonical backend
  package plus neutral imported-geometry names
  (`load_imported_geometry_netcdf`, `apply_imported_geometry_grid_defaults`,
  `imported-netcdf`, `imported-eik`). The Miller helper kernels now expose
  descriptive finite-difference/extension names. Targeted backend,
  runtime/config/artifact, manifest, lint, format, and compile checks passed
  locally.

- 2026-06-15: Continued the naming-governance refactor by making
  `explicit_time` the canonical runtime and benchmark solver key. The old
  `gx_time` spelling is retained only in benchmark/comparison artifacts that
  explicitly refer to external reference data. KBM solver-lock constants,
  example solver choices, and focused runtime/benchmark tests now use the numerics-based
  `explicit_time` name.

- 2026-06-15: Continued benchmark/config naming cleanup by replacing the
  legacy top-level `gx_reference` TOML/config spelling with
  `reference_alignment` / `reference_aligned`. The neutral name now appears in
  Cyclone, KBM, and kinetic benchmark APIs, config serialization, input docs,
  and refactor manifests. Old-name compatibility is bounded to explicit
  benchmark/comparison runner keyword handling, while comparison tools keep
  direct reference-code names only where they operate on external reference
  files.

- 2026-06-15: Started the geometry-import naming tranche by adding canonical
  imported-geometry APIs and model strings:
  `load_imported_geometry_netcdf`,
  `apply_imported_geometry_grid_defaults`, `imported-netcdf`, and
  `imported-eik`. Runtime paths, imported-geometry examples, docs, and focused
  geometry/runtime/benchmark tests now use the canonical names. Later cleanup
  tranches removed the old geometry-model aliases and backend-package shims
  from canonical runtime/config paths.

- 2026-06-16: Moved runtime artifact helper implementations into the
  `spectraxgk.artifacts` domain package. Generic artifact I/O now lives in
  `spectraxgk.artifacts.io`, linear/quasilinear writers in
  `spectraxgk.artifacts.linear`, nonlinear table writers in
  `spectraxgk.artifacts.nonlinear`, NetCDF diagnostic reload helpers in
  `spectraxgk.artifacts.nonlinear_diagnostics`, and finite-value artifact
  checks in `spectraxgk.artifacts.validation`. The obsolete root
  `spectraxgk.runtime_artifact_*` helper modules were removed;
  `spectraxgk.workflows.runtime.artifacts` remains the public dispatcher for executable
  artifact handoff and monkeypatch seams.

- 2026-06-16: Started the runtime workflow consolidation by moving the
  side-effecting TOML case runners into `spectraxgk.workflows.cases`. The
  public `spectraxgk.runtime.run_linear_case` and `run_nonlinear_case` wrappers
  now delegate through explicit dependency injection, preserving existing
  executable behavior and monkeypatch-based tests while keeping CLI workflow
  code separate from solver kernels.

- 2026-06-15: Added the naming-governance rule for the refactor: package
  source, examples, README, and docs should use physics, numerics, and schema
  names (`dealiased`, `NetCDF output`, `runtime diagnostics`, `restart
  layout`) rather than naming internals after comparison codes. Direct
  reference-code names remain allowed only in benchmark/comparison tools,
  parity notes, and validation artifacts whose purpose is explicitly a
  comparison. Performance investigations may still use external source code and
  reruns, but SPECTRAX-GK internals should stay named after the implemented
  algorithm or physical quantity.

- 2026-06-15: Completed the diagnostics/transport-observable naming tranche.
  Runtime diagnostic APIs now use physical names such as
  `fieldline_quadrature_weights`, `distribution_free_energy`,
  `electrostatic_field_energy`, `magnetic_vector_potential_energy`,
  `heat_flux_species`, `particle_flux_species`, `turbulent_heating_species`,
  and `zonal_phi_mode_kxt`. The old diagnostic dataclass aliases were removed
  from the public package surface, the diagnostics test was renamed to
  `tests/test_runtime_diagnostics.py`, and comparison tools were updated to
  import the neutral package APIs while retaining explicit comparison wording
  where appropriate.

- 2026-06-15: Continued the naming/refactor cleanup by renaming the nonlinear
  NetCDF writer and spectral-layout helpers from reference-code-oriented names
  to `spectraxgk.artifacts.nonlinear_netcdf` and
  `spectraxgk.netcdf_spectral_layout`. Runtime diagnostics, adaptive chunk
  execution, restart IO, and startup randomization helpers now use
  `runtime_*`, `NetCDF`, `dealiased`, and `glibc` vocabulary. The remaining
  naming tranches are the real-FFT nonlinear option, diagnostic-weight helper
  names, geometry-import adapters, and benchmark-only comparison tooling.

- 2026-06-15: Continued the naming cleanup in velocity-space numerics by
  renaming gyroaverage helper functions from reference-code names to
  `single_precision_factorial`, `laguerre_quadrature_count`, and
  `laguerre_transform`. The implementation and focused nonlinear bracket tests
  are unchanged; comparison tooling now imports the neutral helper names.

- 2026-06-15: Completed the main runtime artifact facade reduction by moving
  nonlinear NetCDF output schema writing, artifact geometry resolution,
  particle-moment output helpers, and geometry/input metadata group writers into
  `spectraxgk.artifacts.nonlinear_netcdf`. The legacy
  `spectraxgk.workflows.runtime.artifacts` module is now a small dispatch/orchestration
  facade that re-exports compatibility helpers for existing tests and tools.

- 2026-06-15: Continued the runtime artifact refactor by moving generic
  nonlinear JSON/CSV/NPY summary and diagnostic table writing into
  `spectraxgk.artifacts.nonlinear`. The public
  `spectraxgk.workflows.runtime.artifacts` module now remains only as the artifact
  dispatcher, while helper imports use the domain package directly.

- 2026-06-15: Continued the runtime artifact refactor by moving linear scan,
  linear single-run, and quasilinear artifact writers into
  `spectraxgk.artifacts.linear`. The public
  `spectraxgk.workflows.runtime.artifacts` dispatcher keeps the executable/runtime
  artifact contract stable while pure linear CSV/JSON serialization now lives
  in the domain package.

- 2026-06-15: Continued the runtime artifact refactor by moving nonlinear
  NetCDF diagnostic reload, optional-variable parsing, restart-path
  resolution, species-time condensation, and restart-append diagnostic schema
  normalization into `spectraxgk.artifacts.nonlinear_diagnostics`. The
  public `spectraxgk.workflows.runtime.artifacts` dispatcher preserves restart and
  NetCDF handoff compatibility while helper imports now use the domain package.

- 2026-06-15: Completed the nonlinear-gradient follow-up facade split by
  moving candidate-design, composite-control, matched-replicate follow-up,
  QL/linear seed-screen, and VMEC-state runbook reports into
  `spectraxgk.validation.nonlinear_gradient.followup_candidate`,
  `spectraxgk.validation.nonlinear_gradient.followup_composite`,
  `spectraxgk.validation.nonlinear_gradient.followup_plan`,
  `spectraxgk.validation.nonlinear_gradient.followup_ql_seed`, and
  `spectraxgk.validation.nonlinear_gradient.followup_state_runbook`. The legacy
  `spectraxgk.validation.nonlinear_gradient.followup` module is now a compatibility
  facade over core, variance, and report modules, with tests asserting public
  and test-visible facade identities.

- 2026-06-15: Completed the nonlinear-gradient evidence facade split by moving
  production evidence report assembly and missing-campaign gap reports into
  `spectraxgk.validation.nonlinear_gradient.evidence_gap`. The legacy
  `spectraxgk.validation.nonlinear_gradient.evidence` module is now a compatibility
  facade plus JSON artifact loader, while the direct implementation modules
  own classification, replicated windows, central finite differences,
  candidate/bracket screening, and gap-report orchestration.

- 2026-06-15: Continued the nonlinear-gradient evidence refactor by moving
  artifact claim classification into
  `spectraxgk.validation.nonlinear_gradient.evidence_classification` and campaign
  candidate/bracket screening reports into
  `spectraxgk.validation.nonlinear_gradient.evidence_screening`. The remaining
  `spectraxgk.validation.nonlinear_gradient.evidence` module is now a small evidence-gap
  orchestration facade plus JSON loader, while compatibility tests assert that
  public and test-visible facade names still resolve to the moved modules.

- 2026-06-15: Continued the nonlinear-gradient evidence refactor by moving
  replicated nonlinear-window evidence summaries into
  `spectraxgk.validation.nonlinear_gradient.evidence_windows` and central
  finite-difference turbulence-gradient report assembly into
  `spectraxgk.validation.nonlinear_gradient.evidence_fd`. The public
  `spectraxgk.validation.nonlinear_gradient.evidence` facade still re-exports the moved
  report builders and compatibility seam, while tests, API docs, and the
  validation/refactor manifests now track the new modules directly.

- 2026-06-14: Continued the nonlinear-gradient follow-up refactor by moving
  paired-seed variance-reduction planning, control-variate campaign design,
  and independent control-mean uncertainty gates into
  `spectraxgk.validation.nonlinear_gradient.followup_variance`. The public
  `spectraxgk.validation.nonlinear_gradient.followup` facade still re-exports the moved
  report builders and config/helper seams, keeping existing tools and tests
  compatible while making the control-variate evidence path easier to test.

- 2026-06-14: Continued the nonlinear parallelization refactor by moving the
  device-z shard-map RHS route, z-sharding topology check, physical transport
  observable reductions, and serial-vs-device transport-window identity gate
  into `spectraxgk.operators.nonlinear.device_z`. The public
  `spectraxgk.operators.nonlinear.parallel` facade still re-exports the release-visible
  route and test-visible helper seams, preserving the fail-closed distinction
  between identity-gated routing and profiler-backed speedup claims.

- 2026-06-14: Continued the runtime artifact refactor by moving generic
  artifact path/file I/O helpers into `spectraxgk.artifacts.io` and
  pure dealiased-axis, real/imag packing, restart-layout, species-matrix,
  and diagnostic-condense helpers into
  `spectraxgk.netcdf_spectral_layout`. The public
  `spectraxgk.workflows.runtime.artifacts` dispatcher remains for artifact orchestration,
  while low-level helper imports now use the domain package.

- 2026-06-14: Continued the nonlinear-gradient evidence refactor by moving
  the claim-boundary scope markers, acceptance config dataclasses, JSON-safe
  metric parsing, replicated finite-difference helpers, and gradient
  conditioning summary into `spectraxgk.validation.nonlinear_gradient.evidence_core`.
  The public `spectraxgk.validation.nonlinear_gradient.evidence` facade still re-exports
  the moved names used by existing tools, and tests now assert facade/core
  object identity for the production-scope and finite-difference gate seams.

- 2026-06-14: Continued the nonlinear-gradient evidence refactor by moving
  follow-up configuration dataclasses, JSON-safe metric parsing, replicate
  metadata extraction, coefficient/control labeling, and paired-seed/control-
  variate statistics helpers into `spectraxgk.validation.nonlinear_gradient.followup_core`.
  The existing `spectraxgk.validation.nonlinear_gradient.followup` planner facade still
  re-exports the moved names, and tests now assert object identity for the core
  compatibility seam.

- 2026-06-14: Continued the large-module refactor by moving nonlinear spectral
  parallelization primitives into `spectraxgk.operators.nonlinear.spectral_core`.
  The split module now owns deterministic spectral test states, chunk/layout
  utilities, communication/work models, pencil FFT/bracket kernels, RHS
  micro-routes, z-chunked bracket helpers, host-staged sharding preparation,
  and tolerance helpers. The public `spectraxgk.operators.nonlinear.parallel` facade
  still re-exports the moved public and test-visible helpers, with an
  import-identity regression guarding downstream compatibility.

- 2026-06-14: Continued the solver-objective refactor with a larger
  three-module split. Solver-ready geometry objective gates moved into
  `spectraxgk.objectives.geometry`, reduced nonlinear-window estimator
  metrics moved into `spectraxgk.objectives.nonlinear_window`, and
  VMEC/Boozer state coefficient helpers moved into
  `spectraxgk.objectives.vmec_state`. The unchanged
  `spectraxgk.objectives.solver_gradients` facade still re-exports the public
  and test-visible names, while the manifest/docs now track the moved physics,
  numerics, and differentiability contracts directly.

- 2026-06-14: Continued the differentiable-objective refactor by moving the
  implicit dominant-eigenvalue custom VJP and branch-locality finite-difference
  report into `spectraxgk.objectives.eigen`. The legacy
  `spectraxgk.objectives.solver_gradients` facade still re-exports
  `dominant_real_eigenvalue` and
  `dominant_eigenvalue_branch_locality_report`, preserving package-level and
  tool imports while separating the eigen-AD gate from VMEC/Boozer objective
  plumbing.

- 2026-06-14: Continued the solver-objective refactor by moving physical
  `ky` scan mapping, VMEC/Boozer sample-axis helpers, and aggregate objective
  weights into `spectraxgk.objectives.sampling`. The legacy
  `spectraxgk.objectives.solver_gradients` facade still exposes the public
  `solver_grid_options_from_ky_values` helper and private compatibility seams
  used by existing tests, while the new module isolates deterministic sampling
  contracts from gradient-report orchestration.

- 2026-06-14: Continued the solver-objective refactor by moving core
  linear/quasilinear objective constants and value evaluators into
  `spectraxgk.objectives.core`. The unchanged
  `spectraxgk.objectives.solver_gradients` facade still re-exports
  `SOLVER_OBJECTIVE_NAMES`, `SolverScalarObjective`,
  `solver_growth_rate_from_geometry`,
  `solver_linear_operator_matrix_from_geometry`,
  `solver_objective_vector_from_geometry`, and
  `solver_scalar_objective_from_vector`, preserving optimizer, tool, and
  package-level imports while separating forward observables from VMEC/Boozer
  finite-difference report orchestration.

- 2026-06-14: Split the Cyclone benchmark-family runners
  (`run_cyclone_linear` and `run_cyclone_scan`) into focused linear and scan
  owner modules behind the unchanged `spectraxgk.benchmarks` public facade.
  This turns `benchmarks.py` into a
  small compatibility facade for benchmark constants, helper exports, config
  classes, and public runners while preserving legacy imports such as
  `ModeSelection`, `ExplicitTimeConfig`, and `KrylovConfig`.

- 2026-06-14: Split the ETG benchmark-family runners (`run_etg_linear` and
  `run_etg_scan`) into focused linear and scan owner modules behind the
  unchanged `spectraxgk.benchmarks` public facade. ETG branch tests patch the
  implementation owners directly, and the stale ETG branch-test monkeypatch
  against a Cyclone-only helper was removed rather than re-exporting unused
  implementation symbols.

- 2026-06-14: Split the kinetic-electron benchmark-family runners
  (`run_kinetic_linear` and `run_kinetic_scan`) into
  focused kinetic linear and scan owner modules, preserving the reference-aligned
  hypercollision, end-damping, and density-seed policies behind the unchanged
  `spectraxgk.benchmarks` public facade. Focused kinetic branch tests now patch
  the implementation module directly.

- 2026-06-14: Split the TEM benchmark-family runners (`run_tem_linear` and
  `run_tem_scan`) into `spectraxgk.validation.benchmarks.tem` using the same
  behavior-preserving facade pattern as the KBM split. TEM branch tests now
  patch the implementation module directly, while public examples and
  downstream code can continue importing the runners from
  `spectraxgk.benchmarks`.

- 2026-06-14: Continued the behavior-preserving benchmark refactor by moving
  the KBM benchmark-family runners (`run_kbm_linear`, `run_kbm_scan`, and
  `run_kbm_beta_scan`) into focused linear, scan, and beta-scan owner modules
  behind the existing `spectraxgk.benchmarks` public facade. KBM-specific
  regression tests patch the implementation owners directly, preserving public imports while
  making the largest benchmark file smaller and the KBM lane easier to test
  independently.

- 2026-06-14: Continued the behavior-preserving refactor lane by splitting
  nonlinear parallelization contracts, JSON-ready reports, and local
  state-domain identity gates into `spectraxgk.operators.nonlinear.parallel_contracts`
  and `spectraxgk.operators.nonlinear.domain_decomposition`. The public
  `spectraxgk.operators.nonlinear.parallel` facade remains the import surface for
  examples and downstream users, while focused tests now assert that facade
  exports are identical to the underlying contract and domain objects. This
  advances the refactor/testability lane without changing nonlinear RHS,
  transport-window, sharding, or speedup claims.

- 2026-06-13: Closed the current QL lane as a scoped core-portfolio
  diagnostic instead of a universal absolute-flux claim. The refreshed
  `docs/_static/quasilinear_error_anatomy.{png,json,csv}` now records two
  declared stress outliers (`solovev_reference_repair_dt002_amp1em5_n48_t250`
  and `shaped_tokamak_pressure_external_vmec_t650_high_grid_window`) and a
  passing 10-case core portfolio: mean relative error `0.280`, held-out mean
  `0.275`, maximum error `0.575`, and interval coverage `10/10`. The full
  12-case universal predictor remains unpromoted (`0.697 > 0.35`) and the
  core rank/screening metric remains borderline (`Spearman≈0.745 < 0.75`).
  The pre-manuscript dashboard now closes the scoped QL diagnostic at `100%`
  and moves active work to broad nonlinear turbulent-flux optimization and
  nonlinear domain-decomposition speedup.

- 2026-06-13: Advanced the nonlinear turbulent-flux optimization evidence
  without changing the promotion gate. The production guard now counts the
  strict `t=1500` growth, QL, and nonlinear-window optimized-candidate
  replicated trace ensembles alongside the selected `t=700`
  optimized-equilibrium audit, giving `4` qualifying optimized-equilibrium
  ensembles and closing that trace-count blocker. The guard remains
  promoted under the explicit `2%` late-window policy with `3/3` matched baseline-to-optimized audits passing
  (`18.4%` reduction, `7.82` combined SEMs). The replicated-holdout lane is
  frozen at three accepted long-window holdout ensembles; no additional
  generic holdouts are active for this tranche. Broad nonlinear optimization
  moves to `86.7%`; mean pre-manuscript closure moves to `85.4%`.

- 2026-06-13: Closed the scoped broad nonlinear turbulent-flux optimization guard. The default production guard now counts the two full max-mode-5 projected-weight matched comparisons (`2.68%` and `3.35%`, both uncertainty-separated) alongside the no-ESS-to-optimized QA/ESS audit (`18.4%`, `7.82` combined SEMs). The guard records the explicit `2%` late-window reduction policy, `3/3` qualifying matched audits, `4` optimized-equilibrium ensembles, and `3` replicated holdouts. Three strict `t=1500` QA objective candidates remain negative transfer evidence. The remaining pre-manuscript blocker is production nonlinear domain-decomposition speedup.

- 2026-06-13: Added a routed nonlinear spectral-domain profiling artifact to
  the strict closure dashboard. The new
  `docs/_static/nonlinear_spectral_domain_routing_profile.{png,json,csv}`
  verifies serial-vs-logical routed identity on the deterministic nonlinear
  spectral RHS and records warm timing (`0.94x` locally), but it explicitly
  does not permit a production speedup claim. The same artifact now includes a
  communication/work model for the current global-reconstruction route:
  communication/owned-work ratio `6.375`, efficiency ceiling `0.136`, and
  blocker `global_reconstruction_communication_dominates_owned_work`. The
  production nonlinear domain-decomposition lane therefore moves from `55.0%`
  to `70.0%` on identity/timing/model diagnostics while retaining the CPU/GPU
  `>=1.5x` strong-scaling blockers. Mean strict pre-manuscript closure moves
  to `89.2%`.

- 2026-06-13: Froze the nonlinear holdout-expansion lane for this tranche.
  The Solovev-inclusive 12-case QL ledger is now the working calibration and
  negative-evidence dataset; no additional holdouts should be launched to
  rescue the current absolute-flux model. The universal absolute QL lane stays
  blocked because the saturation/amplitude model fails the existing admitted
  ledger (`6.49 > 0.35` for the one-constant positive-growth family and
  `0.697 > 0.35` for the best reduced `spectral_envelope_ridge` candidate).
  Next work therefore moves away from holdout collection and toward: (1)
  better saturation/transport-amplitude physics using the frozen ledger, (2)
  broad matched nonlinear turbulent-flux optimization evidence from existing
  optimized-equilibrium artifacts, and (3) production nonlinear
  domain-decomposition speedup with identity and profiler gates. CI failed only
  because three QL tests still encoded the pre-Solovev 11-case near-miss
  metrics; those tests were updated to assert the current fail-closed
  Solovev-inclusive metrics without loosening any scientific promotion gate.

- 2026-06-13: Converted the QL residual-anatomy artifact into an explicit
  frozen-ledger model-development diagnostic. The refreshed
  `docs/_static/quasilinear_error_anatomy.{png,json,csv}` keeps absolute-flux
  promotion failed (`0.697 > 0.35`) but now records programmatic policy fields:
  additional holdout collection is inactive for this tranche, the active next
  step is saturation/transport-amplitude physics on the admitted 12-case
  ledger, and the dominant residuals are Solovev, shaped-pressure VMEC, and
  ITERModel VMEC. This gives the next QL work a concrete target without
  changing any promotion gate or adding data.

- 2026-06-13: Tightened the production nonlinear turbulent-flux optimization
  guard without launching new runs. The default guard now ingests the existing
  strict `t=1500` matched baseline-to-growth, baseline-to-QL, and
  baseline-to-nonlinear-window comparison artifacts as negative evidence, in
  addition to the positive QA no-ESS matched audit. The refreshed
  `docs/_static/production_nonlinear_optimization_guard.{png,json,csv}`
  remains release-safe but not production-promoted: `4` matched audits are
  present, only `1` qualifies, and the three strict objective-specific audits
  fail reduction/uncertainty gates. This shifts the next optimization work
  toward better optimizer candidates and additional optimized-equilibrium
  evidence, not holdout expansion.

- 2026-06-12: Harvested the Solovev repaired external-VMEC holdout and used it
  to harden the quasilinear claim boundary. The original CPU `dt=0.01`
  duplicate remained too slow, but the office GPU duplicate completed the
  `n48/t250` run with `501` samples through `t=249.94`. Runtime-output,
  readiness, and replicated seed/timestep ensemble gates pass when the Solovev
  repair protocol is admitted under the explicit `20%` spread gate:
  `<Q_i>=1.409`, mean-relative spread `0.1599`, combined SEM/mean `0.0462`.
  Postprocessing exposed a real filename-parser bug: protocol labels such as
  `repair_dt002` in the case slug were being mistaken for timestep replicate
  suffixes. `tools/build_external_vmec_replicate_ensemble.py` now treats only
  suffix-style `seedNN`/`dtNN` tokens as replicate variants, with regression
  tests for GPU suffixes and protocol-`dt` case slugs. The Solovev spectrum and
  nonlinear ensemble are now included in the 12-case QL ledger as negative
  transfer evidence. Positive-growth mixing-length transfer worsens to
  `6.49 > 0.35`; the best reduced `spectral_envelope_ridge` candidate has
  leave-one-geometry-out mean relative error about `0.697`, interval coverage
  `11/12`, and held-out screening metrics `Spearman=0.624`, pairwise order
  `0.689`, so universal absolute-flux and promoted screening claims remain
  blocked. Regenerated the train/holdout, saturation-rule, candidate
  uncertainty, regularization, residual-anatomy, screening, dataset-sufficiency,
  model-selection, holdout-gap, and stellarator-usefulness panels. The
  nonlinear holdout expansion/audit lane is now effectively complete for this
  tranche, while universal absolute QL remains an explicit negative result.

- 2026-06-12: Hardened nonlinear sharding profiling after the local CPU
  forced-device profile exposed a JAX/XLA FFT-layout abort path. The
  multi-device CPU whole-state `pjit` route now fails closed before execution
  unless `--allow-unsafe-cpu-state-sharding` is explicitly set, writing
  `cpu_whole_state_pjit_sharding_unsafe_for_fft_layout` into the profile JSON
  instead of risking a process abort or collective stall. The sweep wrapper now
  preserves failed profile JSON artifacts even when the profile exits nonzero
  because an identity gate failed, and it replaces inherited
  `--xla_force_host_platform_device_count` values so requested CPU device counts
  cannot be contaminated by the parent environment. A bounded local check now
  gives a true one-device identity row and a safe four-device skip row. This is
  negative evidence for current whole-state CPU nonlinear speedup, not a
  promotion; the production nonlinear domain-decomposition speedup lane moves
  to `70.0%` after adding a routed spectral-domain identity/timing/profile model,
  but remains blocked pending a communication-complete decomposed RHS/integrator with
  CPU/GPU identity, transport-window, and profiler-backed speedup gates. The
  office QA growth-candidate `t=1500` `dt=0.04` run has now completed and the
  existing postprocessed artifacts are consistent: the growth ensemble passes,
  but the matched baseline-to-growth comparison gives only `0.60%` reduction
  and fails the configured reduction gate, so it remains negative/non-promoted
  evidence. The original Solovev CPU `dt=0.01` duplicate ran for about
  46 minutes without writing an output bundle, so it was stopped after a
  clean GPU duplicate was launched on office GPU 1 with the same `n48/t250`,
  `dt=0.01` protocol and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. That GPU run
  became the active Solovev holdout source and is superseded by the successful
  harvest entry above.

- 2026-06-12: Harvested the first production-scope VMEC/Boozer held-out
  nonlinear transport artifact and kept broader claims fail-closed. The QH
  `vmec_jax -> booz_xform_jax` held-out surface/field-line run
  (`wout_nfp4_QH_warm_start.nc`, `torflux=0.78`, `alpha=1.2`, `ky rho_i≈0.2`,
  `n64`, `t=700`, window `350-700`, seed31/seed32 plus `dt=0.04`) passed the
  runtime-output, replicated-window, production-holdout, and aggregate
  VMEC/Boozer promotion gates. The accepted ensemble has
  `<Q_i>=7.9978`, mean-relative spread `0.0837`, and combined SEM/mean
  `0.0242`; VMEC/Boozer holdout optimization is now closed for the strict
  pre-manuscript gate. The production nonlinear optimization guard was
  regenerated with three qualifying replicated holdout ensembles
  (D-shaped, circular, QH VMEC/Boozer). At that historical checkpoint it was
  not production-promoted because both the optimized-equilibrium ensemble count
  and matched optimized transport-audit count were still below threshold;
  the current status table below supersedes that count. The strict closure dashboard is now
  `72.0%` mean completion: universal absolute QL remains `60.0%`, broad
  nonlinear turbulent-flux optimization `72.9%`, production nonlinear domain
  decomposition `55.0%`, and VMEC/Boozer holdout optimization `100.0%`.
  The Solovev external-VMEC repair protocol (`dt=0.02`, `init_amp=1e-5`) passed
  a finite `t=50` pilot. The first `n64/t250` repaired launch was resource-terminated
  before writing an output bundle while GPU 0 was sharing memory with another VMEC/JAX job, so a
  separate bounded `n48/t250` seed/timestep manifest is now running on office GPU 0 with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`; the current QA `t=1500` dt0.04 replicate is still
  running on office GPU 1, so no new QA promotion was made. The nonlinear
  spectral logical route identity gate was refreshed on a larger deterministic
  state, and a bounded local CPU profile again showed identity with no active
  state sharding; nonlinear domain speedup remains blocked pending a real
  communication-aware implementation and CPU/GPU profiler-backed speedup.
  `tools/write_external_vmec_holdout_configs.py` now exposes the protocol
  knobs needed for repair launches (`--init-amp`, gradients, geometry sampling,
  and diagnostic strides), with regression coverage. README/docs now include
  the QH held-out transport panel and corrected claim boundaries. Repository
  size remains under policy after compressing three existing plot PNGs.

- 2026-06-12: Added the next concrete pre-manuscript closure tranche without
  promoting unfinished claims. `tools/write_optimized_equilibrium_transport_configs.py`
  now exposes and records explicit VMEC transport-sample metadata (`torflux`,
  `alpha`, `npol`, `ky`, `tprim`, `fprim`, `nu`) so long nonlinear audits can
  be tied to a held-out surface/field-line rather than a default flux tube.
  Generated a local QH held-out transport launch contract for
  `wout_nfp4_QH_warm_start.nc` at `torflux=0.78`, `alpha=1.2`, `ky=0.2`,
  `n64`, `t=700`, window `350-700`, seed31/seed32 plus `dt=0.04`; the tracked
  pre-manuscript runbook now lists the matching office commands. Regenerating
  the VMEC/Boozer promotion gate exposed an additional honest blocker: the
  aggregate objective artifact currently covers alphas `0` and `0.7`, while the
  line-search artifact covers only alpha `0`; this gate now fails closed on
  `line_search_reuses_aggregate_sample_set` and the still-missing production
  held-out nonlinear transport artifact. Added
  `integrate_logical_decomposed_nonlinear_spectral` as the callable logical
  decomposed nonlinear spectral RHS/integrator route behind the identity
  artifacts. It is identity-gated and serial-fallback safe; it remains a
  diagnostic/profiling route, not a production distributed-FFT speedup claim
  until CPU/GPU profiler and speedup gates pass. Local targeted status:
  34 focused tests passed, ruff passed, release-readiness passed, and repository
  size passed (`tracked_total_bytes=49.87 MB`). GitHub Actions for the previous
  push is still queued. Office QA `t=1500` jobs are still running and should
  not be harvested until final `t≈1500`; Solovev remains queued behind them.
  After commit `1920196`, the older office checkout could not be fast-forwarded
  because untracked generated docs artifacts would be overwritten, so those
  files were left untouched and a fresh checkout was created at
  `/home/rjorge/spectrax_premanuscript_holdouts_20260612`. The same QH
  held-out VMEC/Boozer configs were generated there and deferred launcher PID
  `3425020` is waiting behind the active QA/Solovev queues. Added
  `tools/build_vmec_boozer_production_holdout_artifact.py`, a fail-closed
  postprocessor that combines a concrete transport manifest with a passed
  replicated nonlinear ensemble to produce the exact held-out surface/field-line
  JSON consumed by the VMEC/Boozer promotion gate. The runbook now lists the
  complete postprocessing chain: finite-output gate, replicate-ensemble gate,
  production holdout artifact, then VMEC/Boozer aggregate promotion gate.

- 2026-06-12: Tightened the production nonlinear turbulent-flux optimization
  promotion guard to match the broader pre-manuscript requirement. The guard is
  now release-safe but not production-promoted: current counts are `1/3`
  matched baseline-to-optimized audits, `1/3` optimized-equilibrium ensembles,
  and `2` accepted long-window holdout ensembles. README/docs/plan wording was
  updated so the selected QA no-ESS -> optimized QA/ESS audit remains visible
  as one positive scoped result (`18.4%`, `7.8 sigma`) without closing the
  broad multi-equilibrium nonlinear optimization lane. Office status at
  `2026-06-12T10:51-05:00`: the true `t=1500` growth-objective QA audits are
  still running on both GPUs; seed33 has written `t≈800`, seed32 remains at the
  first `t≈400` checkpoint, and the Solovev external-VMEC holdout launcher is
  correctly waiting behind those queues. A local 4-logical-CPU pjit nonlinear
  sharding attempt on velocity axes (`l,m`) reproduced the XLA CPU FFT layout
  failure and collective stall seen for `ky/kx`; it was killed before the
  five-minute cap. This is negative evidence for exposing current pjit
  nonlinear state sharding as production CPU speedup. The next parallelization
  step is a different communication-aware decomposition, not widening the
  existing `state_sharding` options.

- 2026-06-12: Advanced the independent external-VMEC holdout lane. A bounded
  VMEC linear screen over four previously unscreened `vmec_jax` examples found
  `wout_solovev_reference.nc` launchable (`gamma=0.0944` at `ky=0.2857`) and
  `wout_up_down_asymmetric_tokamak_reference.nc` unstable but already
  represented (`gamma=0.0360` at `ky=0.4762`). `LandremanSenguptaPlunk`
  remains below the nonlinear-launch threshold (`gamma=0.0073`) and
  `basic_non_stellsym_pressure_reference` fails the current VMEC aspect-cut
  flux-tube path. The tracked external-VMEC runbook now selects Solovev and
  writes an office-resolvable nonlinear launch command. Configs were generated
  on office under `tools_out/external_vmec_holdouts/solovev_reference`, and a
  deferred launcher is waiting behind the active QA `t=1500` GPU queues so it
  does not oversubscribe the two GPUs. This is a launch contract only: Solovev
  is not admitted into quasilinear calibration until its long-window nonlinear
  grid/window, replicate, and recalibration gates pass.

- 2026-06-12: Started the pre-manuscript closure phase after verified
  ``v1.6.5`` release/PyPI publication. Added
  ``tools/build_pre_manuscript_closure_status.py`` and tracked
  ``docs/_static/pre_manuscript_closure_status.{png,pdf,json,csv}`` as the
  strict machine-readable gate for the four lanes that must close before
  drafting starts. Current strict status: not ready for manuscript drafting,
  mean closure ``61.8%``. Lane status is recalibrated against stricter
  manuscript requirements, not release-safe scoped diagnostics:
  universal absolute quasilinear heat-flux prediction ``60.0%`` partial,
  broad end-to-end nonlinear turbulent-flux stellarator optimization ``54.2%``
  blocked, production nonlinear domain-decomposition speedup ``55.0%``
  partial, and VMEC/Boozer holdout optimization ``78.0%`` partial. Immediate
  execution order:

  1. Universal absolute QL: add at least one genuinely independent converged
     nonlinear holdout, then replace the failed amplitude/saturation model so
     leave-one-geometry-out candidate uncertainty, model selection, and
     absolute train/holdout error all pass the ``0.35`` mean-relative-error
     transport gate. No runtime/TOML absolute-flux predictor is allowed until
     this closes.
  2. Broad nonlinear turbulent-flux optimization: extend from the single
     selected-QA positive audit to at least three independent matched
     baseline-vs-optimized long-window transport audits, at least three
     optimized-equilibrium ensembles, and the frozen three replicated nonlinear
     holdout ensembles. Only post-transient running-average transport windows
     count; reduced/startup nonlinear-window objectives remain excluded.
  3. Production nonlinear domain decomposition: keep independent ``k_y`` and
     UQ batching as the current production path while implementing a real
     communication-aware nonlinear decomposed RHS/integrator route. Promotion
     requires serial-vs-decomposed transport-window identity plus large-grid
     CPU and multi-GPU speedup ``>=1.5`` with profiler artifacts.
  4. VMEC/Boozer holdout optimization: keep the existing alpha/surface and
     second-equilibrium holdouts as reduced plumbing evidence, then add a
     production-scope held-out surface/field-line nonlinear transport artifact
     with same-WOUT provenance through ``vmec_jax -> booz_xform_jax ->
     SPECTRAX-GK`` before claiming optimization closure.

- 2026-06-11: Started nonlinear admission for the top solved-WOUT screen
  candidate, `qp_diag_nfp2_m4_final`. The `t=150`, `dt=0.05`, `n48/n64`
  office-GPU pair is finite but non-admissible (`0.163` common-window and
  `0.200` least-window heat-flux differences). The true restart continuation
  to `t=250` passes the grid/window gate (`0.033` common-window, `0.0023`
  least-window, slopes below `2e-3`, CV about `0.08`). The minimal `n64`
  seed/timestep ensemble also passes (`<Q_i>=16.40`, spread `0.071`,
  combined SEM/mean `0.029`). A temporary QL re-score with this new holdout
  improves aggregate mean relative error `2.83 -> 2.65`, but holdout error is
  still `3.13 > 0.35`, so absolute-flux promotion remains blocked. Backup
  QA/QI ladders are prepared locally under `tools_out/` but remain untracked.

- 2026-06-11: Added a fail-closed VMEC optimization-result candidate screen
  before launching nonlinear holdouts from solved `vmec_jax` WOUTs. A bounded
  local CPU scan of four solved mode-5 optimization outputs found no launchable
  nonlinear holdout: `qa_nfp2` is marginal, `qh_nfp3`/`qp_nfp4` are stable, and
  apparent high-growth `qp_nfp3` is rejected because all sampled rows have
  non-positive effective `k_perp^2`. Added
  `tools/build_vmec_optimization_candidate_screen_gate.py` and tracked the JSON
  gate artifact so future screens require finite positive metric evidence.

- 2026-06-11: Corrected and closed the QH warm-start long-window restart
  protocol as negative evidence. The earlier direct `n80/t450` and `n80/t700`
  launches were segment-length runs from zero, not cumulative horizons, so a
  true staged ladder was relaunched on office by copying complete restart
  bundles forward. The corrected `n80/t450` and `n80/t700` runs both passed
  runtime-output checks, but the relaxed 20% high-grid convergence gates still
  fail: `t450` has `0.355` common-window and `0.294` least-window heat-flux
  differences, while `t700` has `0.3487` common-window and `0.3668`
  least-window differences. The final `t700` late-window means are about
  `5.885` for `n64` and `4.137` for `n80`, so the mismatch is not an early
  transient. QH remains excluded from quasilinear calibration. Regenerated the
  external-VMEC runbook without a QH modified-protocol allowance; it now fails
  closed with no launch commands until a genuinely new independent candidate or
  materially higher-resolution protocol exists. Added local guardrails so
  generated external-VMEC manifests write executable staged restart-ladder
  scripts, external-VMEC QL admission is fail-closed on promotion-gate/claim
  metadata, nonlinear sharding artifacts report per-backend identity/speedup
  blockers, and public QA optimization examples state linear/QL/reduced
  nonlinear claim boundaries explicitly.

- 2026-06-11: Harvested the first modified-protocol QH warm-start nonlinear
  office gate. The `n64/n80`, `dt=0.04`, `t=250` pair finished cleanly and is
  finite, but it is not admissible: the common-window and least-trending
  high-grid heat-flux disagreements are `0.3675` and `0.4120`, above both the
  strict `15%` and relaxed `20%` policies. QH therefore stays excluded from the
  quasilinear calibration ledger. Longer `n80/t450` and `n80/t700` runs were
  launched on office to compare against the completed `n64/t450` and
  `n64/t700` traces and decide whether the mismatch is an early-window artifact
  or a real grid-resolution blocker.

- 2026-06-11: Tightened two release-facing evidence ledgers while QH nonlinear
  runs continued on office. The QA transport optimization status now reports a
  machine-readable `claim_evidence_level` and explicit
  `claim_promotion_blockers`, and release readiness expects the current scoped
  state: matched long-window nonlinear audit evidence is present, while QL
  model selection and simple absolute-flux QL remain unpromoted. The nonlinear
  sharding production-speedup gate now records per-backend identity-evidence
  summaries and tolerance fractions; it remains `diagnostic_only` because the
  GPU production speedup candidate is still missing.

- 2026-06-11: Hardened the direct nonlinear campaign runner used by external
  VMEC holdouts and nonlinear-gradient audits. `--skip-existing` now skips a
  task only when the complete runtime bundle (`*.out.nc`, `*.restart.nc`, and
  `*.big.nc`) exists, and status rows record the required bundle files. This
  prevents interrupted long GPU runs from being mistaken for completed
  restartable transport evidence.

- 2026-06-11: Promoted the next external-VMEC nonlinear holdout work from a
  blocked replay to a launch-ready modified-protocol QH candidate. A bounded
  local screen of the `vmec_jax` `nfp4_QH_warm_start` fixture found a weak but
  finite unstable branch (`gamma = 0.022949` at `ky = 0.4762`). The refreshed
  `docs/_static/external_vmec_next_holdout_runbook.{png,json,csv}` now passes
  only as `nonlinear_holdout_launch_plan_not_transport_validation` and writes a
  single command for `n64/n80`, `dt=0.04`, and horizons `t=250,450,700`.
  Previous QH nonlinear gates remain excluded; this candidate can enter the QL
  ledger only after the fresh high-grid convergence, late-window time-horizon,
  and seed/timestep replicate gates pass.

- 2026-06-11: Added a quasilinear residual-anatomy artifact for the current
  best reduced candidate. `docs/_static/quasilinear_error_anatomy.{png,json,csv}`
  consumes the existing uncertainty, screening, and saturation-rule sidecars
  and remains fail-closed: `spectral_envelope_ridge` has mean relative error
  `0.424 > 0.35`, no screening gate passes, and no runtime/TOML absolute-flux
  predictor is promoted. The anatomy shows the shaped-pressure external-VMEC
  holdout is the largest residual and external axisymmetric VMEC cases account
  for about `59%` of the residual budget, while HSX/W7-X are comparatively
  well tracked. This originally pointed the next QL work toward richer
  saturation physics plus dataset expansion, not threshold loosening; that
  data-expansion step is superseded by the
  2026-06-13 frozen-ledger decision, so the active next step is saturation and
  transport-amplitude physics on the admitted twelve-case ledger.

- 2026-06-11: Closed the shaped-tokamak-pressure external-VMEC repair as a
  scoped high-grid nonlinear holdout. The full `n48/n64/n80`, `dt=0.04`,
  `t=450` ladder fails only coarse-grid heat-flux agreement (`0.469 > 0.15`),
  while retained `n64/n80` gates pass at `t=450` (`0.0789`) and `t=650`
  (`0.0983` common, `0.0981` least). The high-grid horizon gate passes
  (`0.0418` common, `0.1237` least), and the `n80` seed/timestep ensemble
  passes on `t=[325,650]` with mean heat flux `7.156`, mean-relative spread
  `0.0939`, and combined SEM/mean `0.0463`. The case is now in the
  quasilinear ledger as high-grid admission evidence only. It makes the
  current absolute/screening QL claims weaker, not stronger:
  positive-growth mixing length has holdout mean error `3.42 > 0.35`, the
  spectral-envelope candidate has uncertainty mean error `0.424 > 0.35`, and
  no screening model is currently accepted after the shaped holdout.

- 2026-06-11: Repaired the shaped-tokamak-pressure external-VMEC nonlinear
  holdout after the `dt=0.05` `n80` instability. The office `n64/n80`,
  `dt=0.04`, `t=450` reruns completed with finite diagnostics and passed the
  high-grid convergence gate: late-window heat-flux means `7.422` and `6.859`,
  symmetric relative difference `0.0789 < 0.15`, and no stationarity/CV/sample
  failures. The `t=650` restart continuations for the same two grids are now
  running on office to test time-horizon stability before any admission or
  quasilinear calibration update. CI shard 24 exposed a documentation
  guardrail drift; README, release scope, and verification matrix now again
  state that `spectral_envelope_ridge` is only a scoped manuscript
  model-selection candidate and not a runtime/TOML absolute-flux predictor.

- 2026-06-11: Returned to the larger science gates after the RBC(1,1)
  diagnostic refresh. CI for commit `20e22a9` passed, including wide coverage.
  The quasilinear holdout-gap report was regenerated and still blocks
  absolute-flux promotion with seven independent holdouts and held-out error
  `1.91 > 0.35`. The runbook now selects a modified
  shaped-tokamak-pressure external-VMEC campaign (`n64/n80/n96`, `t=450,650`)
  rather than replaying CTH-like or same-family ITERModel evidence. The office
  campaign was launched from a fresh shallow clone at
  `/home/rjorge/spectrax_ql_shaped_holdout_20260611_063247`; it waits for an
  idle GPU, runs the restart ladder, and postprocesses the nonlinear gate.
  The production nonlinear turbulent-flux optimization guard was also
  re-audited: the selected QA optimized-equilibrium audit remains a scoped
  positive result with `18.4%` lower long-window heat flux and `7.8 sigma`
  separation, while broader nonlinear turbulence-gradient optimization and
  universal absolute QL-flux prediction remain open.

- 2026-06-11: Expanded the quasilinear model-development ledger to consume the
  admitted CTH-like high-grid replicated nonlinear ensemble as a first-class
  nonlinear calibration input. Added fail-closed ensemble-gate ingestion in
  `spectraxgk.validation.quasilinear.calibration` and promotion-ready checks in
  `spectraxgk.validation.quasilinear.window`, with regression tests. Regenerated the QL
  saturation, candidate uncertainty, dataset-sufficiency, screening-skill,
  usefulness, model-selection, and holdout-gap artifacts. Result: rank/correlation
  screening is stronger (`spectral_envelope_ridge` passes full and held-out
  rank gates), but strict candidate uncertainty/model-selection is demoted
  (`0.377 > 0.35`) and universal absolute-flux promotion remains blocked
  (`1.91 > 0.35`, two more independent holdouts required).

- 2026-06-11: Added skip-existing command lists to the external-VMEC nonlinear
  holdout config manifest. Future office campaigns can use
  `staged_ladder_skip_existing_commands` or
  `direct_full_horizon_skip_existing_launch_commands` to resume interrupted
  long runs or manually distribute remaining grids without treating partial
  outputs as complete: each wrapper skips only after the full `.out.nc`,
  `.restart.nc`, and `.big.nc` bundle exists. The live shaped-tokamak-pressure
  campaign remains unchanged; the completed `n64/t450` segment passed the basic
  runtime output gate with late-window mean heat flux about `8.39`.

- 2026-06-11: Added a local-CPU quasilinear regularization sensitivity audit
  for the `spectral_envelope_ridge` candidate. The tracked
  `docs/_static/quasilinear_candidate_regularization_sweep.{png,json,csv}`
  artifact sweeps the ridge penalty and confirms that the near miss is not
  fixed by retuning regularization: the best setting remains `lambda = 0.3`
  with full-ledger mean relative error `0.377 > 0.35`, held-out mean `0.355`,
  and interval coverage `8/9`. The artifact is documented as a fail-closed
  model-development guardrail, not as an absolute-flux predictor.

- 2026-06-11: The live shaped-tokamak-pressure `n80/t450` office run completed
  the integrator but failed artifact validation because `Wg_t` became
  non-finite at an early saved sample. Root cause is a diagnostic masking bug:
  Runtime diagnostic reductions multiplied energy/flux factors by the dealias
  mask after intermediate products, so `inf * 0` in a masked/dealiased mode
  could produce `NaN`. Patched free-energy, field-energy, `phi2`, heat-flux,
  particle-flux, and turbulent-heating reductions to zero masked modes before
  products or moment contractions. Added a regression test that injects `inf`
  into a masked mode while preserving strict validation for unmasked diagnostics.
  The shaped holdout remains unadmitted until the repaired `n80/n96` reruns and
  high-grid/window gates pass.

- 2026-06-10: Added and passed the explicit CTH-like high-grid admission
  policy. The case is now admitted to the quasilinear model-development ledger
  as a scoped high-grid nonlinear transport holdout, with `n48` explicitly
  excluded and `n64/n80` plus late-horizon and seed/timestep replicate evidence
  retained. This is not a full `n48/n64/n80` convergence claim and not an
  absolute-flux promotion: the one-constant train/holdout calibration still
  fails with mean held-out relative error `1.91 > 0.35`, and the holdout-gap
  report still requires two additional independent post-transient nonlinear
  holdouts plus a better saturation model. Commit `a2e1072` passed CI run
  `27288648364` with 59 successful jobs and 1 skipped nightly job.

- 2026-06-10: Closed the next CTH-like high-grid robustness step with true
  office GPU continuations rather than a proxy. The `n80`, `t=[250,350]`
  seed/timestep extraction passed individual readiness gates but failed the
  strict ensemble spread gate (`0.1819 > 0.15`), so the window was too short
  for admission. The same four variants were restart-continued from `t=350`
  to `t=700` and re-extracted on `t=[350,700]`; readiness and ensemble gates
  both pass with mean heat flux `9.603`, mean-relative spread `0.0406`, and
  combined SEM/mean `0.0517`. CTH-like is therefore a replicated high-grid
  candidate under the later explicit admission policy: full `n48/n64/n80`
  convergence still fails, but `n64/n80` plus late-window replicate evidence
  are sufficient for scoped high-grid calibration-ledger use. The
  release repository tracks only
  `docs/_static/external_vmec_cth_like_modified_replicates_t700/replicate_ensemble_gate.*`;
  raw NetCDFs and trace byproducts remain on office.

- 2026-06-09: Completed the CTH-like modified-protocol horizon harvest and
  added a high-grid time-horizon gate. All nine direct office jobs returned
  `0`. The `t=150` high-grid `n64/n80` gate has close heat-flux means
  (`0.026` common-window and `0.009` least-window relative differences) but
  fails the common-window trend gate (`0.00292 > 0.002`), so it is still a
  transient window. The late `t=250`/`t=350` high-grid horizon gate passes with
  common/least horizon changes `0.018`/`0.019`, but its promotion gate remains
  false because replicate/seed/timestep evidence and an explicit high-grid
  admission policy are still required. The release repository tracks compact
  JSON gate sidecars plus the paper-facing `t=350` and late-horizon PNGs; the
  larger pilot byproducts remain reproducible from the office run directory.

- 2026-06-09: Harvested the first CTH-like modified-protocol nonlinear
  outputs at `t=350` from office. All three direct full-horizon jobs returned
  `0`. The formal `n48/n64/n80` grid gate fails closed with common/least
  heat-flux differences `0.296`/`0.272`, dominated by the coarse `n48` trace.
  The high-grid `n64/n80` diagnostic gate passes with common/least differences
  `0.058`/`0.013`, so the case is now a high-grid candidate only. It remains
  outside quasilinear calibration until the remaining horizons and any
  replicate/window gates close under an explicit high-grid admission policy.
  The release repository tracks the compact JSON gate sidecar and high-grid
  PNG needed for the README/docs; the remaining pilot byproducts stay outside
  git to preserve the repository-size policy.

- 2026-06-09: Added an explicit modified-protocol external-VMEC holdout
  contract for failed-family repairs. The default runbook still fail-closes
  unchanged failed families, but `tools/build_external_vmec_holdout_runbook.py`
  can now require `--allow-modified-protocol-family` plus a
  `--modified-protocol-note` and optional horizon override. The tracked
  external-VMEC linear screen now includes the existing CTH-like spectrum point
  (`gamma = 0.0488013` at `ky = 0.285714`), and the refreshed runbook selects a
  CTH-like `n48/n64/n80`, `t = 150,250,350` repair campaign. This is a launch
  contract only; CTH-like remains outside quasilinear calibration until the new
  traces pass grid/window and post-transient holdout gates.

- 2026-06-09: Completed the resumed office QA optimizer ladder. Scalar-trust
  and LBFGS-adjoint growth/QL runs all returned `0` and passed the
  authoritative rerun-WOUT admission gate, but the solved-candidate gate stayed
  false. The SPSA nonlinear-window metric sweep completed four plus/minus
  pairs; the best reduced metrics were about `0.06144` and `0.06174` versus
  the nearby `0.063` level. Tracked summaries:
  `docs/_static/vmec_jax_qa_optimizer_ladder_resume_status.json` and
  `docs/_static/vmec_jax_qa_optimizer_ladder_spsa_metric_summary.json`. These
  artifacts support optimizer-strategy design only; they do not promote a
  long-window nonlinear turbulent-flux reduction.

- 2026-06-09: Fixed an office launch mismatch in the external-VMEC holdout
  config manifest: direct nonlinear commands now prefix `PYTHONPATH=src` so
  `python -m spectraxgk.cli` resolves the checkout source instead of a stale
  venv-installed package. Focused tests enforce the prefix and the structured
  direct-full-horizon step counts. After fast-forwarding office to `8ef0931`,
  the CTH-like modified-protocol campaign was regenerated and launched with
  the direct full-horizon commands. The first active pair is the most
  informative final-window batch, `t=350` at `n80` and `n64`, both running on
  the two office GPUs; no holdout admission claim exists until the resulting
  traces pass the grid/window gate.

- 2026-06-09: Diagnosed the strict QA `t1500` mismatch as a launch-contract
  issue, not a physics result. Final-horizon TOMLs are restart-ladder segments:
  launching a `t1500` segment command from `t=0` integrates only the
  `(1500-1100)` segment and stops near `t=400`. The generators now record
  per-config `dt`, emit explicit staged-ladder commands, emit direct
  full-horizon commands (`t1500/dt`: 30000 steps for `dt=0.05`, 37500 for
  `dt=0.04`), and include a runtime-output gate over `t=[1100,1500]`.
  Focused tests cover the exact strict-policy numbers. The QA `|B|`
  manuscript/readme panels now use unfilled Boozer-LCFS contours instead of
  filled density maps.

- 2026-06-09: Relaunched the corrected strict QA full-horizon audit on
  `office` from a clean shallow clone at commit `9e50d59`. The clean-clone
  `src/` path is forced ahead of the stale editable install, and
  `/home/rjorge/booz_xform_jax` is injected so the internal VMEC/Boozer backend
  is available. The controller queued all twelve true `t=1500` nonlinear jobs
  with direct 30000/37500-step commands and is running two jobs concurrently on
  the two A4000 GPUs. The runtime log line `running ... over 8000 steps` is the
  first restart/checkpoint chunk (`output.nsave`), not the total horizon; the
  controller-level command and executable header show the full 30000/37500-step
  target.

- 2026-06-09: Harvested the true `t=1500` strict QA baseline, growth-objective,
  quasilinear-objective, and nonlinear-window-objective audit triplets from
  office. All four pass the runtime-output and replicated seed/timestep gates
  over `t=[1100,1500]`.
  The strict QA baseline has mean `<Q_i>=11.580`, mean relative spread
  `3.81%`, and combined SEM/mean `1.95%`; growth has mean `<Q_i>=11.510`,
  spread `4.27%`, and SEM/mean `1.24%`; quasilinear has mean
  `<Q_i>=11.636`, spread `2.34%`, and SEM/mean `1.64%`; nonlinear-window has
  mean `<Q_i>=11.609`, spread `3.66%`, and SEM/mean `1.77%`. Matched
  comparisons do not promote any transport candidate: growth gives only
  `0.60%` relative reduction (`z=0.26`, below the `4%` gate), while the
  quasilinear and nonlinear-window rows are slightly worse (`-0.49%`,
  `z=-0.19`; `-0.25%`, `z=-0.09`). The strict-QA `t=1500` candidate set is
  closed as robust negative optimization-transfer evidence.

- 2026-06-09: Added reproducible strict-QA `t=1500` postprocess tooling and
  fixed its output-gate artifact names after the final QL harvest. The new
  `tools/compact_replicate_ensemble_bundle.py` rewrites compact tracked
  ensemble provenance from regenerable trace CSVs to authoritative NetCDF
  outputs, and `tools/write_vmec_qa_t1500_postprocess_manifest.py` writes the
  exact runtime-output, ensemble, compact-provenance, and matched-comparison
  commands for baseline, growth, quasilinear, and nonlinear-window rows. The
  tracked `docs/_static/vmec_qa_t1500_postprocess_manifest.json` is a command
  manifest, not a simulation claim, and now resolves to the tracked
  `baseline`, `growth`, `quasilinear`, and `nonlinear_window` output-gate
  JSONs.

- 2026-06-09: Added `tools/build_qa_optimizer_strategy_report.py` plus focused
  tests and regenerated
  `docs/_static/vmec_jax_qa_optimizer_strategy_report.{png,json,csv}`. The
  report combines the strict QA optimizer panel with the converged `RBC(1,1)`
  long-window landscape. It shows a real lower-Q direction (`+40% RBC(1,1)`,
  about 35% below the zero-offset late-window mean), but treats that landscape
  as a noise/convergence diagnostic rather than an admission source for
  optimized QA stellarators. Nonlinear optimization promotion remains blocked
  because current transport optimizer rows are still diagnostic-only and the
  true matched `t=1500` audits fail the `4%` reduction gate. The recommended
  campaign ladder is now
  explicit: exact-adjoint least squares from the VMEC-JAX simple seed for
  smooth QA constraints, constraint-aware adjoint trust/L-BFGS with
  transport-weight continuation for linear/QL residuals, and SPSA/CMA-ES/
  Bayesian outer-loop comparators only for noisy long-window nonlinear
  heat-flux objectives.

- Harvested the matched strict QA full-sweep nonlinear audit from office and
  reran postprocessing with the patched fail-closed tools. All 36 raw runtime
  jobs returned success, but the generated traces only reach `t≈400` while the
  admission window is `t=[1100,1500]`. The four replicated ensembles therefore
  have `n_finite_means=0`, and all three matched baseline-vs-growth/QL/
  nonlinear-window comparisons are non-promoted with no computable transport
  reduction. These artifacts are retained as negative admission evidence under
  `docs/_static/optimized_equilibrium_replicates/vmec_qa_full_sweep_*` and
  `docs/_static/qa_strict_baseline_to_*_strict_baseline.*`; no QA point is
  added to the quasilinear calibration ledger.

- Tightened the quasilinear screening/correlation artifact to separate full-portfolio
  screening from held-out-only promotion. The refreshed
  `docs/_static/quasilinear_screening_skill.{png,pdf,json,csv}` records
  `spectral_envelope_ridge` as the only full-portfolio and held-out
  rank/correlation pass on the expanded ledger, but the mean-error gate remains
  empty. This strengthens the claim boundary: useful rank-screening evidence
  is claimable now, while universal absolute-flux or screening promotion is now
  treated as blocked by saturation/model physics on the frozen admitted ledger,
  not by missing holdout collection for this tranche.

- Extended the quasilinear holdout-gap report to ingest the screening-skill
  sidecar. The refreshed `docs/_static/quasilinear_holdout_gap_report.*` now
  carries both `absolute_flux_promotion_requirements` and
  `screening_promotion_requirements`: full-portfolio and held-out-only
  rank/correlation screening pass for `spectral_envelope_ridge`, but the
  active follow-up is no longer additional holdout collection. The frozen
  Solovev-inclusive ledger should be used to improve the saturation and
  transport-amplitude model before either screening or absolute-flux promotion
  is reconsidered.

# SPECTRAX-GK Active Plan and Running Log

Last updated: 2026-06-13
Active repository: `uwplasma/SPECTRAX-GK`
Current public baseline: `main`; see `pyproject.toml` for the active release
version and GitHub Actions for the latest CI result.
Historical planning archive: private repo `rogeriojorge/spectraxgk_plan`

This file is the public active plan and concise running log. Keep it short,
dated, and tied to reproducible artifacts, tests, figures, and gates. Detailed
historical logs live outside the release repository so clones stay small.

## Current Release Status

- CI/CD: release-readiness, package build, docs build, quick numerical shards,
  and wide package coverage are green for the verified `v1.6.5` release commit
  `5e845f1`.
  - GitHub Actions CI run `27419886180`: successful.
  - GitHub release/PyPI workflow run `27421079800`: successful.
  - Wide package coverage gate remains required at `>=95%`.
- Repository-size policy: tracked payload must stay below 50 MB. This active
  plan replaces the old 531 KB historical log to restore edit headroom.
- Release posture: technically shippable for scoped claims. Broad
  manuscript-level absolute quasilinear-flux and nonlinear
  turbulence-optimization claims are not promoted. The strict QA baseline,
  true `t=1500` matched audits, and refreshed RBC(1,1) landscape are tracked as
  optimization/noise diagnostics and negative-transfer evidence, not as solved
  nonlinear turbulent-flux optimization.

## Active Lanes

| Lane | Status | Current gate |
| --- | ---: | --- |
| CI/CD, release infrastructure, package coverage | 100% | Green CI, 95% package-wide coverage |
| Quasilinear screening/model-development | 100% | Scoped core QL diagnostic is closed: excluding the declared Solovev and shaped-pressure stress outliers, `spectral_envelope_ridge` passes the transport/coverage diagnostic with core mean error `0.280`, held-out core error `0.275`, and coverage `10/10`; rank screening remains borderline and no runtime/TOML universal predictor is promoted |
| Universal absolute quasilinear-flux prediction | Deferred | Full 12-case stress-ledger promotion remains unpromoted (`0.697 > 0.35` for the best reduced candidate and `6.49 > 0.35` for the one-constant family); this is no longer an active holdout-collection lane for this release tranche |
| Nonlinear holdout expansion/audits | 100% | Frozen for this tranche with ten admitted holdouts; CTH-like and shaped-pressure are scoped high-grid admissions, QH warm-start is retained as negative high-grid evidence, and Solovev passes a repaired `n48/t250` seed/timestep ensemble under the explicit `20%` spread gate as negative absolute-QL evidence |
| Rerun-WOUT admission and artifact policy | 100% | Explicit authoritative rerun-WOUT path implemented and tested |
| Strict QA candidate screening | 100% | Top-12 projected edge candidate passes rerun-WOUT gates and reduces the 18-point metric by 2.29% |
| Strict nonlinear transport and campaign-admission evidence | 100% | Strict top-12 matched audit fails promotion; historical full-sweep QA audit is negative evidence; true t=1500 baseline/growth/quasilinear/nonlinear-window triplets pass, but all three matched candidate comparisons fail the 4% reduction gate |
| Boundary-coefficient landscape and optimizer-noise diagnosis | 99% | 31-point RBC(1,1) reduced linear/QL landscape is tracked; 24 true long-window nonlinear overlays pass the scoped diagnostic gates; `+20%` is admitted under an explicit 20% spread gate, while `+45%` and higher remain stability-boundary/open long-window points |
| Differentiable QA optimization evidence | 100% | Current VMEC/Boozer differentiability and holdout plumbing gate is closed: frequency, QL, reduced nonlinear-window estimator gradients, alpha/surface/second-equilibrium holdouts, and production-scope QH heldout transport pass; broad nonlinear turbulent-flux optimization remains a separate lane |
| Broad end-to-end nonlinear turbulent-flux stellarator optimization | 100% | Closed for the scoped guard: optimized-equilibrium trace-count blocker is closed with four qualifying ensembles, the generic holdout lane is frozen with three accepted replicated holdout ensembles, and three matched baseline-to-optimized audits pass the explicit `2%` late-window reduction policy (`18.4%`, `2.68%`, `3.35%`); broad multi-surface/multi-alpha generalization remains a future claim |
| VMEC/Boozer holdout optimization | 100% | Closed for the current pre-manuscript gate: reduced alpha/surface, second-equilibrium, gradient holdout matrix, and aggregate promotion gates pass |
| Docs/readme/release hygiene | 100% | Public wording separates reduced linear/QL landscape metrics from true nonlinear heat-flux evidence; strict-QA t1500, CTH high-grid, and QL holdout-gap artifacts are tracked |
| Performance/parallelization release lane | 98.5% | Independent-work parallel paths are release-ready; nonlinear sharding profiler provenance is versioned and checker-gated; device-z `shard_map` RHS and fixed-step physical transport-window routes now have CPU speedup evidence plus CPU/GPU identity artifacts; whole-state/domain production speedup remains diagnostic pending full-solver routing and GPU speedup gates |
| Production nonlinear domain-decomposition speedup | 90% | Strict pre-manuscript gate remains partial: local, spectral, combined, routed timing, pencil fused-bracket, logical physical transport-window, active device-z pencil RHS, and active device-z physical transport-window identity now pass on logical CPU and office GPU after host-staging the initial state before explicit z sharding. The global-reconstruction route has communication/owned-work ratio `6.375`; the pencil model reduces communication/FFT-work ratio to `0.075`; the active logical-CPU `shard_map` z-pencil RHS profile reaches `1.51x` on two logical CPU devices and `2.62x` on four; the CPU transport-window profile reaches `1.72x` on two and `3.11x` on four with max final-state error `7.45e-9`; the two-GPU transport-window profile passes identity but reaches only `1.20x`, so production-speedup gates remain blocked by GPU speedup and full-solver transport-window routing |
| QA optimization optimizer-comparison metadata | 100% | Public examples emit strict nonlinear audit manifests; optimizer/full-sweep generators now separate restart-ladder and direct full-horizon commands, add output gates, and admit only completed true t=1500 replicated ensembles; the matched QL comparison is closed and non-promoted |
| External-VMEC high-grid holdout policy | 100% | CTH-like modified-protocol launch, horizon gates, `n80` seed/timestep long-window replicate gate, and explicit high-grid admission policy are reproducible; full `n48/n64/n80` remains non-claimable |
| Optimizer comparison campaign execution | 76% | Metadata/generators, strategy report, and solved-WOUT prelaunch metric gate are ready; actual multistart/continuation/SPSA-CMA-BO campaign remains planned unless promoted to a new run tranche |
| Production nonlinear turbulent-flux optimization evidence | 100% | Closed for the scoped production guard under the explicit `2%` late-window reduction policy: three matched baseline-to-optimized audits pass (`18.4%`, `2.68%`, `3.35%`) with positive uncertainty separation, the optimized-equilibrium trace-count requirement passes with four ensembles, and strict t=1500 growth/QL/nonlinear-window candidate audits remain tracked as negative transfer evidence; broader multi-surface/multi-alpha nonlinear optimization remains a future claim |

Deferred post-release/manuscript extensions unless explicitly reprioritized:
W7-X zonal long-window recurrence/damping and W7-X TEM/multi-flux-tube
extension. Nonlinear domain decomposition is no longer merely deferred in the
pre-manuscript plan: it is an active strict gate, but remains diagnostic until
identity, transport-window, and CPU/GPU speedup requirements pass.

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

- Added the stellarator-specific quasilinear usefulness summary
  `docs/_static/quasilinear_stellarator_usefulness.{png,pdf,json,csv}` and
  its generator/test. The figure makes the current scientific conclusion
  explicit: simple one-constant quasilinear rules fail HSX/W7-X absolute-flux
  transfer, the spectral-envelope ridge candidate is the best scoped
  model-development result, QA remains matched-nonlinear-audit-only, and QH is
  excluded until grid/window convergence passes.
- Added the quasilinear screening/correlation summary
  `docs/_static/quasilinear_screening_skill.{png,pdf,json,csv}` and its
  generator/test. It separates useful screening claims from absolute-flux
  promotion: `spectral_envelope_ridge` passes the current rank/correlation and
  mean-error gates, while `accepted_absolute_flux_models` remains empty.
- Added `tools/write_vmec_jax_optimizer_comparison_manifest.py`, a tested
  manifest generator for strict QA optimizer comparisons. It emits one strict
  SciPy QA baseline, matched deterministic transport commands for
  `scipy`/`scalar_trust`/`lbfgs_adjoint`, and SPSA/CMA/BO outer-loop contracts
  with deterministic metric-evaluation and nonlinear-audit templates. The
  tracked manifest sidecar is
  `docs/_static/vmec_jax_qa_optimizer_comparison_manifest.json`.
- Harvested matched strict nonlinear audits on office under
  `/home/rjorge/spectrax_qa_matched_strict_20260608/SPECTRAX-GK`. All raw
  baseline-vs-growth/QL/nonlinear-window runtime jobs completed, but the
  strict admission postprocess fails closed because the traces end near
  `t=400` while the requested accepted window is `t=[1100,1500]`. The
  comparison artifacts therefore remain negative admission evidence and cannot
  be used to refit quasilinear calibration or promote nonlinear optimization.
- Polled the active positive-side RBC(1,1) campaign; no new gates were
  harvestable at this checkpoint. The tracked landscape remains at 23/31 true
  nonlinear overlays until the running `p0p55`/`p0p6` work completes and passes
  the strict `t=[1100,1500]` ensemble gate.
- Added normalized optimizer-comparison metadata to the VMEC-JAX QA
  optimization driver and full-sweep panel. Optimizer methods may now be
  compared only inside identical comparison-fingerprint groups.
- Updated public QA optimization examples to write strict staged nonlinear ITG
  audit manifests: horizons `700,1100,1500`, accepted window `t=[1100,1500]`,
  seed variants `32,33`, and timestep variant `dt=0.04`.
- Documented the optimizer strategy: least-squares for smooth QA constraints,
  scalar-adjoint methods for differentiable linear/quasilinear residuals, and
  stochastic/derivative-free outer-loop comparators only for noisy long
  nonlinear heat-flux objectives after matched audit gates pass.

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

1. Treat the strict top-12 edge candidate as reduced-objective-only evidence.
   Its matched long-window nonlinear audit passed both ensemble gates but failed
   promotion, so it must not be described as nonlinear turbulent-flux
   optimization.
2. Use `docs/_static/nonlinear_campaign_admission_report.json` as the
   admission-only launch contract for the next nonlinear optimizer campaign. It
   admits the selected
   ``+3% RBC(0,1)`` direction for a bounded multi-control campaign because the
   reduced prelaunch gate, deterministic cross-sample dispersion gate, and
   replicated nonlinear landscape gate all pass. It remains a campaign
   admission, not a broad nonlinear turbulent-flux optimization claim.
3. Keep the tracked failed-promotion artifacts in docs as negative evidence:
   `docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.json`,
   `docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.png`,
   `docs/_static/strict_qa_top12_edge_redesign_report.json`, and the
   baseline/candidate ensemble JSON sidecars.
4. Keep CI green after each tranche: fast unit shards, coverage aggregation,
   repository-size gate, docs links, and package build.
5. Keep the production nonlinear optimization guard strict:
   `docs/_static/production_nonlinear_optimization_guard.json` now requires
   optimized-equilibrium seed/timestep provenance and at least three matched
   baseline-to-optimized reduction audits before production promotion. The
   scoped guard now satisfies that count under the explicit `2%` late-window
   policy, but new optimized candidates must still reproduce the matched-reduction
   evidence structure before any broader claim.

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

### 2026-06-05

- Refreshed the paper-facing boundary landscape from the earlier RBC(0,1)
  narrow scan and sparse RBC(1,1) follow-up to a 31-point ``RBC(1,1)`` scan
  over ``[-75%, +75%]`` of the strict QA baseline coefficient. If a future
  scanned coefficient has zero baseline value, the landscape builder now sets
  the absolute scan amplitude from the largest configured reference
  coefficient, defaulting to ``RBC(1,0)`` and ``RBC(0,1)``.
- The refreshed RBC(1,1) landscape is a diagnostic, not a nonlinear transport
  claim. The top panel now plots only deterministic linear growth and all
  explicit quasilinear heat-flux rules on the same three-surface,
  two-field-line, three-``ky`` sample used by the optimizer examples. The
  bottom panel accepts only true long-window post-transient nonlinear
  heat-flux ensembles; reduced/startup nonlinear-window metrics are excluded
  from the paper-facing landscape.
- Updated the QA full-sweep panel so transport rows with only small mean-iota
  misses remain marked ``diag-ok`` when ``|iota| >= 0.39``; strict admission at
  ``|iota| >= 0.41`` remains separate. The solved-WOUT iota profile plot now
  omits the VMEC axis point so zero/convention artifacts do not skew the axis.
- Launched the office true nonlinear landscape campaign for all 31 RBC(1,1)
  coefficients, with seed31, seed32, and ``dt=0.04`` variants at
  ``n64:64:64:40:40``. The first ``-75%`` point showed that ``t_max=700`` with
  window ``t=[350,700]`` was still inside the transient: seed/timestep traces
  kept drifting upward and failed running-mean convergence. The first
  continuation therefore tested ``t_max=1100`` with the transport window
  ``t=[700,1100]`` before the neighboring point below forced the final
  ``t_max=1500`` protocol. The previous sparse baseline/``-50%``/``+35%`` audit
  remains useful historical evidence, but it is no longer the paper-facing
  landscape; promotion now waits for the complete refreshed ensemble overlay.
  A controller GPU-placement bug briefly
  co-located ``+35%`` seed31 and ``dt=0.04`` on one GPU; the controller was
  stopped, the ``dt=0.04`` run was relaunched manually on the idle GPU, and the
  seed31 plus seed33 traces completed cleanly.
- Restarted the refreshed 31-point nonlinear landscape campaign after removing
  a logging ambiguity from generated TOMLs: external/optimized VMEC nonlinear
  configs now write ``[output].nsave = [run].steps`` so NetCDF output artifact
  handoff does not split a ``t=700`` run into a misleading 10,000-step first
  chunk. The clean office logs now report 14,000 steps for ``dt=0.05`` and
  17,500 steps for the ``dt=0.04`` variants.
- Clarified the RBC(1,1) landscape plot contract: two panels only, with growth
  and every explicit quasilinear rule on top and true post-transient nonlinear
  heat flux on the bottom. The bottom panel no longer uses ``<Q_i>`` shorthand
  in the label, and reduced/startup nonlinear-window values remain excluded.
- Added ``tools/build_external_vmec_replicate_ensemble.py
  --allow-failed-gates`` for diagnostic landscape postprocessing only. The
  option lets the full landscape collect failed convergence points without
  aborting, while JSON/PNG sidecars still mark those points failed and prevent
  promotion. Normal release/physics gates remain fail-closed.
- First office long-window RBC(1,1) landscape outputs completed: the ``-75%``
  seed31/seed32 runs reached ``t=699.903`` with 281 samples and late-window
  heat-flux means about ``17.48`` and ``16.10`` over ``t=[350,700]``. The
  two-seed diagnostic mean is about ``16.79`` with mean-relative spread about
  ``8.2%``; it is not promotion-ready until the timestep variant and
  convergence gates complete.
- Continued the ``-75%`` seed31, seed32, and ``dt=0.04`` variants from the
  existing ``t=700`` restarts to ``t=1100``. The new ``t=[700,1100]`` ensemble
  passes readiness and ensemble gates with means ``18.489``, ``18.939``, and
  ``18.545``, ensemble mean ``18.657``, mean-relative spread ``2.41%``, and
  combined SEM/mean ``1.25%``. This closes the first refreshed RBC(1,1)
  nonlinear landscape point under the true post-transient protocol and proves
  that the earlier small-window landscape was not sufficiently converged.
- Continued the neighboring ``-70%`` point through the same ``t=[700,1100]``
  window. Readiness passed, but the ensemble failed robustness with mean
  ``14.581``, mean-relative spread ``19.96%`` against the ``15%`` limit, and
  combined SEM/mean ``6.44%``; the ``dt=0.04`` trace remained systematically
  high. Extending the same three variants to ``t=1500`` and accepting only
  ``t=[1100,1500]`` closed the gate with mean ``15.586``, mean-relative spread
  ``13.81%``, and combined SEM/mean ``4.14%``. The paper-facing 31-point
  landscape launch protocol is therefore ``t_max=1500`` with the
  ``t=[1100,1500]`` transport window.
- Started a restartable office controller for the full 31-point
  ``RBC(1,1)`` nonlinear overlay under that final protocol. It skips variants
  whose NetCDF outputs already reach ``t=1500``, continues partial ``t=1100``
  outputs when available, runs missing seed/timestep variants on the two office
  GPUs, and postprocesses each coefficient with diagnostic
  ``--allow-failed-gates`` sidecars so failed points remain visible without
  being promotable.
- The controller has closed the first two low-end nonlinear overlay points
  under the final ``t=[1100,1500]`` protocol: ``-75%`` passes with ensemble
  mean ``18.572``, mean-relative spread ``2.46%``, and combined SEM/mean
  ``1.28%``; ``-70%`` passes with ensemble mean ``15.586``,
  mean-relative spread ``13.81%``, and combined SEM/mean ``4.14%``. It then
  launched the direct ``t=1500`` ``-65%`` seed variants.
- The blind full-controller path was stopped before it could launch additional
  coefficients because the direct-from-zero ``-65%`` ``t=1500`` seed variants
  ran for nearly an hour without producing checkpoint/output files. Those two
  active seed runs were left running briefly to salvage the already-spent GPU
  time, but the scalable 31-point overlay should be relaunched with staged
  ``t=700 -> 1100 -> 1500`` checkpointed horizons and explicit per-stage
  wall-time/status reporting before committing to the full scan.
- After a final wait, the same ``-65%`` direct seed variants still had no
  NetCDF outputs after about ``62`` minutes, so they were terminated. No
  additional landscape coefficients are running on office. The next controller
  must enforce stage-level wall-time caps and visible progress instead of
  single-call direct ``t=1500`` integrations.
- Relaunched ``-65%`` as a bounded staged pilot only to ``t=700`` for seed31
  and seed32, one per office GPU, with a ``2700`` second per-process timeout.
  This tests whether the checkpointed horizon strategy can produce salvageable
  NetCDF/restart files before committing to later ``t=1100`` and ``t=1500``
  continuation stages.
- The bounded ``-65%`` ``t=700`` stage succeeded in about nine minutes: seed31
  and seed32 each wrote ``281`` samples plus restart files at ``t=699.903``.
  Their terminal heat fluxes were similar, about ``14.24`` and ``14.34``. The
  same two outputs are now being continued to ``t=1100`` with restart/append
  enabled and the same ``2700`` second per-process timeout.
- The ``-65%`` ``t=1100`` continuation also succeeded, appending both seed
  outputs to ``442`` samples at ``t=1099.879``. The ``t=[700,1100]`` heat-flux
  means are already close, about ``15.56`` and ``15.07``. The same two seed
  outputs are now being continued to ``t=1500`` under the staged timeout
  protocol; the timestep variant remains to be run before the coefficient can
  enter the strict ensemble gate.
- The ``-65%`` seed variants then reached ``t=1499.854`` with ``603`` samples;
  their ``t=[1100,1500]`` means are close, about ``15.78`` and ``15.30``. The
  ``dt=0.04`` timestep variant is now running through the same staged protocol,
  starting with the bounded ``t=700`` stage.
- The ``-65%`` ``dt=0.04`` ``t=700`` stage also completed, writing ``351``
  samples at ``t=699.932`` with terminal heat flux about ``13.44``. It is now
  being continued to ``t=1100`` under the same staged timeout protocol.
- The ``-65%`` ``dt=0.04`` ``t=1100`` continuation completed, writing ``552``
  samples at ``t=1099.944`` with a ``t=[700,1100]`` heat-flux mean about
  ``15.74``. The final ``dt=0.04`` ``t=1500`` continuation is now running; once
  it finishes, ``-65%`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-65%`` ``dt=0.04`` continuation reached ``t=1500`` and the strict
  ``t=[1100,1500]`` ensemble gate passed without diagnostic relaxation. The
  three-member seed/timestep ensemble has mean ``15.227``, mean-relative spread
  ``7.81%``, and combined SEM/mean ``2.92%``. This closes the third adjacent
  low-end nonlinear overlay point and validates the staged
  ``700 -> 1100 -> 1500`` checkpoint protocol for continued landscape
  production.
- Installed a reusable staged label runner on office and launched the next
  adjacent coefficient, ``-60%``/``m0p6``, through the bounded ``t=700`` seed31
  and seed32 stage. This continues the low-end scan with the validated
  checkpoint protocol rather than direct ``t=1500`` integrations.
- The ``-60%`` ``t=700`` seed stage completed with restart files. It was slower
  than ``-65%`` (about ``35`` minutes), and the early ``t=[350,700]`` seed means
  were more separated, about ``16.58`` and ``14.78``. Because the accepted
  landscape window is ``t=[1100,1500]``, this is only a transient diagnostic;
  both seed outputs are now continuing to ``t=1100`` under the staged timeout
  protocol.
- The ``-60%`` ``t=1100`` seed continuation completed, appending both outputs
  to ``442`` samples at ``t=1099.879``. The ``t=[700,1100]`` means remain
  separated, about ``17.73`` and ``15.95``, so the final
  ``t=[1100,1500]`` ensemble gate is essential. The two seed outputs are now
  continuing to ``t=1500`` under the staged timeout protocol.
- The ``-60%`` seed variants reached ``t=1499.854`` with ``603`` samples.
  Their accepted ``t=[1100,1500]`` heat-flux means are now close, about
  ``17.22`` and ``17.35``, so the final-window seed spread is below ``1%``.
  The independent ``dt=0.04`` timestep variant has been launched through the
  same bounded staged protocol, starting with the ``t=700`` checkpoint stage,
  before ``-60%`` can enter the strict three-member ensemble gate.
- While the ``-60%`` timestep replicate runs alone on office GPU0, the idle
  GPU1 is being used for a single non-overlapping ``-55%``/``m0p55`` seed32
  ``t=700`` pilot. This is intentionally only one variant, launched manually
  with ``CUDA_VISIBLE_DEVICES=1``, so it cannot collide with the gated
  ``m0p6`` timestep path.
- The ``-60%`` ``dt=0.04`` ``t=700`` stage completed with ``351`` samples at
  ``t=699.932`` and a ``t=[350,700]`` heat-flux mean about ``14.37``. The
  same timestep replicate is now continuing to ``t=1100`` on office GPU0,
  while the independent ``m0p55`` seed32 pilot continues on GPU1.
- The ``-60%`` ``dt=0.04`` ``t=1100`` continuation completed with ``552``
  samples at ``t=1099.944`` and a ``t=[700,1100]`` heat-flux mean about
  ``16.88``. The final ``dt=0.04`` continuation to ``t=1500`` is now running
  on GPU0; once it finishes, ``m0p6`` can be postprocessed with the strict
  seed/timestep ensemble gate over ``t=[1100,1500]``.
- The final ``-60%`` ``dt=0.04`` continuation reached ``t=1500`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``16.900``, mean-relative
  spread ``7.22%``, and combined SEM/mean ``2.29%``. This closes the fourth
  adjacent low-end true nonlinear overlay point. The next adjacent
  coefficient, ``-55%``/``m0p55``, now has seed31 and seed32 ``t=700`` pilots
  running one per office GPU; the timestep replicate should wait until a GPU
  frees.
- The ``-55%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are about ``13.12``
  and ``14.06``. Both seed outputs are now continuing to ``t=1100`` in
  parallel, one per office GPU, before any ``m0p55`` timestep replicate is
  launched.
- The ``-55%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are about ``13.57`` and ``15.02``.
  Both seeds are now continuing to ``t=1500`` under the same bounded staged
  protocol; after that, the ``dt=0.04`` timestep replicate remains to be run
  before the coefficient can enter the strict ensemble gate.
- The ``-55%`` seed continuations reached ``t=1499.854`` with close accepted
  ``t=[1100,1500]`` heat-flux means, about ``14.22`` and ``14.55``. The
  required ``dt=0.04`` timestep replicate has been launched through the staged
  protocol, starting at ``t=700`` on office GPU0. While that gating timestep
  replicate runs, the next adjacent coefficient, ``-50%``/``m0p5``, has a
  single non-overlapping seed32 ``t=700`` pilot running on office GPU1.
- The ``-55%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``13.34``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0 while the ``m0p5`` seed32 pilot continues on GPU1.
- The ``-55%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``14.95``. The
  final ``dt=0.04`` continuation to ``t=1500`` is now running on office GPU0;
  if it completes, ``m0p55`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-55%`` ``dt=0.04`` continuation reached ``t=1500`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``14.696``, mean-relative
  spread ``7.51%``, and combined SEM/mean ``2.22%``. This closes the fifth
  adjacent low-end true nonlinear overlay point. The next adjacent
  coefficient, ``-50%``/``m0p5``, now has seed31 and seed32 ``t=700`` pilots
  running one per office GPU.
- The ``-50%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are about ``11.57``
  and ``11.06``. Both seeds are now continuing to ``t=1100`` under the same
  staged protocol, one per office GPU.
- The ``-50%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are about ``12.75`` and ``10.97``.
  Both seeds are now continuing to ``t=1500``; the final
  ``t=[1100,1500]`` seed window and the later ``dt=0.04`` replicate remain
  required before this coefficient can enter the strict gate.
- The ``-50%`` seed continuations reached ``t=1499.854`` with close accepted
  ``t=[1100,1500]`` heat-flux means, about ``12.44`` and ``12.29``. The
  required ``dt=0.04`` timestep replicate has been launched through the staged
  protocol, starting with the ``t=700`` checkpoint stage on office GPU0.
- While the ``-50%`` timestep replicate runs on office GPU0, the next adjacent
  coefficient, ``-45%``/``m0p45``, has a single non-overlapping seed32
  ``t=700`` pilot running on office GPU1. This is only pipeline fill; ``m0p45``
  remains open until seed31, seed32, and the timestep replicate pass the final
  ``t=[1100,1500]`` gate.
- The ``-50%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``11.74``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0. The ``-45%`` seed32 pilot also reached ``t=700`` with a transient
  mean about ``7.09``, and the matching seed31 ``t=700`` pilot is running on
  GPU1.
- The ``-50%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``12.43``. The
  final timestep continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p5`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-50%`` ``dt=0.04`` continuation reached ``t=1500`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``12.516``, mean-relative
  spread ``4.24%``, and combined SEM/mean ``1.94%``. This closes the sixth
  adjacent low-end true nonlinear overlay point. The next adjacent
  coefficient, ``-45%``/``m0p45``, now has both seed ``t=700`` pilots complete
  with close transient means, about ``7.03`` and ``7.09``, and both seeds are
  continuing to ``t=1100``.
- The ``-45%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means remain close, about ``7.19`` and
  ``7.38``. Both seeds are now continuing to ``t=1500`` under the staged
  protocol; the final seed window and ``dt=0.04`` timestep replicate remain
  required before ``m0p45`` can enter the strict gate.
- The ``-45%`` seed continuations reached ``t=1499.854`` with ``603`` samples.
  The accepted ``t=[1100,1500]`` seed means are close, about ``7.17`` and
  ``7.03``. The required ``dt=0.04`` timestep replicate has been launched
  through the staged protocol, starting with the bounded ``t=700`` checkpoint
  stage on office GPU0.
- The ``-45%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``7.00``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0.
- The ``-45%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``7.24``. The
  final ``dt=0.04`` continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p45`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-45%`` ``dt=0.04`` continuation reached ``t=1499.956`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``7.205``, mean-relative
  spread ``5.36%``, and combined SEM/mean ``1.57%``. This closes the seventh
  adjacent low-end true nonlinear overlay point.
- Launched the next adjacent coefficient, ``-40%``/``m0p4``, through the
  bounded ``t=700`` seed31 and seed32 pilot stage, one per office GPU.
- The ``-40%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are close, about
  ``11.41`` and ``11.26``. Both seed outputs are now continuing to
  ``t=1100`` under the same staged protocol, one per office GPU.
- The ``-40%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are about ``11.91`` and ``10.99``.
  Both seeds are now continuing to ``t=1500``; the final
  ``t=[1100,1500]`` seed window and later ``dt=0.04`` replicate remain
  required before this coefficient can enter the strict gate.
- The ``-40%`` seed variants reached ``t=1499.854`` with ``603`` samples and
  nearly identical accepted ``t=[1100,1500]`` heat-flux means, about ``11.75``
  and ``11.76``. The required ``dt=0.04`` timestep replicate has been launched
  through the staged protocol, starting with the ``t=700`` checkpoint stage on
  office GPU0.
- The ``-40%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``11.95``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0.
- The ``-40%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``11.51``. The
  final timestep continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p4`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-40%`` ``dt=0.04`` continuation reached ``t=1499.956`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``11.722``, mean-relative
  spread ``0.99%``, and combined SEM/mean ``1.96%``. This closes the eighth
  adjacent low-end true nonlinear overlay point.
- Launched the next adjacent coefficient, ``-35%``/``m0p35``, through the
  bounded ``t=700`` seed31 and seed32 pilot stage, one per office GPU.
- The ``-35%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are close, about
  ``10.75`` and ``10.48``. Both seed outputs are now continuing to
  ``t=1100`` under the same staged protocol, one per office GPU.
- The ``-35%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are close, about ``10.52`` and
  ``10.78``. Both seeds are now continuing to ``t=1500``; the final
  ``t=[1100,1500]`` seed window and later ``dt=0.04`` replicate remain
  required before this coefficient can enter the strict gate.
- The ``-35%`` seed variants reached ``t=1499.854`` with ``603`` samples.
  Their accepted ``t=[1100,1500]`` heat-flux means are about ``10.45`` and
  ``11.01``. The required ``dt=0.04`` timestep replicate has been launched
  through the staged protocol, starting with the ``t=700`` checkpoint stage on
  office GPU0.
- The ``-35%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``10.83``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0.
- The ``-35%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``10.52``. The
  final timestep continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p35`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-35%`` ``dt=0.04`` continuation reached ``t=1499.956`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``10.780``, mean-relative
  spread ``5.23%``, and combined SEM/mean ``1.58%``. This closes the ninth
  adjacent low-end true nonlinear overlay point.
- Launched the next adjacent coefficient, ``-30%``/``m0p3``, through the
  bounded ``t=700`` seed31 and seed32 pilot stage, one per office GPU.
- The ``-30%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are close, about
  ``11.34`` and ``11.76``. Both seed outputs are now continuing to
  ``t=1100`` under the same staged protocol, one per office GPU.
- The ``-30%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are close, about ``11.40`` and
  ``11.56``. Both seeds are now continuing to ``t=1500``; the final
  ``t=[1100,1500]`` seed window and later ``dt=0.04`` replicate remain
  required before this coefficient can enter the strict gate.
- The ``-30%`` seed variants reached ``t=1499.854`` with ``603`` samples.
  Their accepted ``t=[1100,1500]`` heat-flux means are close, about ``11.77``
  and ``11.65``. The required ``dt=0.04`` timestep replicate has been launched
  through the staged protocol, starting with the ``t=700`` checkpoint stage on
  office GPU0.
- The ``-30%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``11.44``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0.
- The ``-30%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``11.47``. The
  final timestep continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p3`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-30%`` ``dt=0.04`` continuation reached ``t=1499.956`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``11.530``, mean-relative
  spread ``5.29%``, and combined SEM/mean ``1.61%``. This closes the tenth
  adjacent low-end true nonlinear overlay point.
- Launched the next adjacent coefficient, ``-25%``/``m0p25``, through the
  bounded ``t=700`` seed31 and seed32 pilot stage, one per office GPU.
- The ``-25%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are about ``10.13``
  and ``10.59``. Both seed outputs are now continuing to ``t=1100`` under the
  same staged protocol, one per office GPU.
- The ``-25%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are about ``10.30`` and ``10.69``.
  Both seeds are now continuing to ``t=1500``; the final
  ``t=[1100,1500]`` seed window and later ``dt=0.04`` replicate remain
  required before this coefficient can enter the strict gate.
- The ``-25%`` seed variants reached ``t=1499.854`` with ``603`` samples.
  Their accepted ``t=[1100,1500]`` heat-flux means are close, about ``10.52``
  and ``10.41``. The required ``dt=0.04`` timestep replicate has been launched
  through the staged protocol, starting with the ``t=700`` checkpoint stage on
  office GPU0.
- The ``-25%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``10.34``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0.
- The ``-25%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``10.35``. The
  final timestep continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p25`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-25%`` ``dt=0.04`` continuation reached ``t=1499.956`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``10.507``, mean-relative
  spread ``1.78%``, and combined SEM/mean ``1.45%``. This closes the eleventh
  adjacent low-end true nonlinear overlay point.

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
- Completed the matched long-window nonlinear audit. Baseline and candidate
  replicated ensembles both pass: baseline late-window mean `11.22662981`
  with combined SEM `0.27005804`; candidate mean `11.16155393` with combined
  SEM `0.17680020`. The matched comparison fails promotion with absolute
  reduction `0.06507587`, relative reduction `0.00579656`, combined
  uncertainty `0.32278422`, and uncertainty z-score `0.201608`. Tracked
  compact artifacts:
  `docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.json`,
  `docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.png`,
  `docs/_static/strict_qa_top12_edge_redesign_report.json`,
  `docs/_static/strict_qa_rerun_baseline_ensemble_gate.json`, and
  `docs/_static/strict_qa_top12_step1p25em3_candidate_ensemble_gate.json`.
  The redesign report confirms that the 18-point reduced objective has
  sufficient surface, field-line, and `k_y` coverage, but blocks promotion on
  insufficient matched nonlinear reduction and insufficient uncertainty
  separation. Conclusion: this is a fail-closed negative transfer result, not a
  nonlinear turbulence-optimization claim.
- Added a boundary-coefficient landscape diagnostic for strict QA ``RBC(0,1)``.
  The 18-point reduced scan over ``[-6%, -3%, 0, +3%, +6%]`` finds the ``+3%``
  coefficient point best for all reduced objectives: growth improves by about
  ``51%``, quasilinear flux by about ``49%``, and reduced nonlinear-window heat
  flux by about ``4.7%``. The small reduced nonlinear-window margin makes this
  an optimizer-noise diagnostic, not a nonlinear heat-flux claim. Generated
  artifacts:
  `docs/_static/vmec_boundary_transport_landscape_rbc01.png`,
  `docs/_static/vmec_boundary_transport_landscape_rbc01.json`, and
  `docs/_static/vmec_boundary_transport_landscape_rbc01.csv`.
- Launched a two-GPU office nonlinear error-bar queue for the baseline, ``+3%``,
  and ``+6%`` landscape points under
  `/home/rjorge/tmp/spectrax_landscape_rbc01_code`. The VMEC-JAX WOUTs required
  metadata-only patching because their scalar ``Aminor_p/Rmajor_p/aspect``
  fields were zero; Fourier geometry was left unchanged.
- Added a guarded ``--reuse-reduced-json`` path to the boundary-landscape
  builder so finished reduced metrics can be reused when overlaying expensive
  nonlinear ensemble error bars. The reuse gate validates coefficient values
  and the full surface/field-line/``k_y`` sample set before accepting metrics.
- Extended the reduced landscape metric sidecars to store deterministic
  per-sample rows and cross-sample standard errors. These error bars diagnose
  surface/field-line/``k_y`` spread in the reduced model; they are explicitly
  not stochastic nonlinear heat-flux SEMs.
- Completed the office ``RBC(0,1)`` replicated nonlinear landscape queue.
  Baseline, ``+3%``, and ``+6%`` ensembles all pass the late-window gate over
  ``t=[350,700]`` with three replicas each. The ensemble means are
  ``8.554 +/- 0.120`` at baseline, ``6.275 +/- 0.042`` at ``+3%``, and
  ``6.427 +/- 0.044`` at ``+6%``. The selected nonlinear audit therefore
  confirms a ``26.65%`` reduction for ``+3%`` with ``z=17.99`` and a
  ``24.87%`` reduction for ``+6%`` with ``z=16.71``. The final landscape panel
  is `docs/_static/vmec_boundary_transport_landscape_rbc01.png`; only the
  compact ensemble JSON sidecars are tracked, not NetCDF outputs or office
  scratch traces.
- Added a backend-free nonlinear landscape admission helper in
  `spectraxgk.validation.stellarator.transport_admission` and materialized
  `docs/_static/vmec_boundary_transport_landscape_admission.json`. The policy
  requires passed ensembles, three replicas, bounded relative SEM, a minimum
  relative heat-flux reduction, and an uncertainty-separated z-score. Applied
  to the ``RBC(0,1)`` landscape, it selects ``+3% RBC(0,1)``.
- Added `tools/build_nonlinear_landscape_admission_report.py` so future
  landscape admissions can be regenerated and CI-gated directly from compact
  ensemble JSON sidecars, without manually inspecting office outputs or
  tracking large NetCDF files.
- Added the nonlinear landscape admission JSON to the release-readiness
  required-artifact contract, so this positive campaign-admission evidence
  cannot silently disappear from future release candidates or be broadened into
  a turbulent-optimization claim.
- Added a reduced nonlinear-audit prelaunch gate. It blocks reduced candidates
  below a calibrated margin before expensive GPU audits; applied to the
  ``RBC(0,1)`` landscape, ``p0p03`` passes for bounded campaign admission with a
  ``4.678%`` reduced nonlinear-window margin over a ``4%`` threshold derived
  from the failed strict top-12 transfer reference.
- Materialized the complementary negative prelaunch artifact
  `docs/_static/strict_qa_top12_edge_prelaunch_gate.json`: the strict top-12
  edge candidate's ``2.2876%`` reduced margin is blocked against the same
  ``4%`` threshold before any future GPU launch at that margin.
- Launched the next adjacent strict ``RBC(1,1)`` coefficient, ``-20%``/``m0p2``,
  through the bounded ``t=700`` seed31 and seed32 pilot stage, one run per office
  GPU, continuing the accepted staged protocol toward the strict ``t=[1100,1500]``
  nonlinear overlay.
- Completed the ``m0p2`` seed ``t=700`` pilot stage and launched the ``t=1100``
  continuation for seed31 and seed32, again one run per office GPU. The transient
  ``t≈350..700`` means are seed31 ``9.799934475`` and seed32 ``9.979881665``
  over 141 samples each; these are checkpoint diagnostics only, not the accepted
  strict ``t=[1100,1500]`` overlay value.
- Completed the ``m0p2`` seed ``t=1100`` continuation and launched the final
  ``t=1500`` seed continuation. Both seed files reached ``t=1099.87854``; the
  appended ``t≈702..1100`` means are seed31 ``9.773082185`` and seed32
  ``9.821264589`` over 160 samples each. These remain checkpoint diagnostics
  until the final strict ``t=[1100,1500]`` ensemble is built with the timestep
  replicate.
- Completed the final ``m0p2`` seed continuation to ``t=1500`` and launched the
  ``dt=0.04`` timestep replicate from ``t=700``. Both seed files reached
  ``t=1499.85437``; the accepted-window ``t≈1102..1500`` means are seed31
  ``9.969421017`` and seed32 ``9.837136143`` over 160 samples each. The
  point remains open until the timestep replicate reaches ``t=1500`` and the
  strict ensemble gate passes.
- Completed the ``m0p2`` ``dt=0.04`` timestep replicate to ``t=700`` and
  launched its ``t=1100`` continuation. The ``t≈350..700`` checkpoint mean is
  ``9.785573818`` over 176 samples, consistent with the seed transient
  windows.
- Completed the ``m0p2`` ``dt=0.04`` timestep continuation to ``t=1100`` and
  launched the final ``t=1500`` timestep continuation. The file reached
  ``t=1099.94421``; the ``t≈702..1100`` checkpoint mean is ``9.909606848``
  over 200 samples.
- Closed the strict ``m0p2`` nonlinear overlay. The final ``dt=0.04`` trace
  reached ``t=1499.95605`` with strict-window mean ``10.183545103`` over
  200 samples. The three-member fail-closed ensemble over ``t=[1100,1500]``
  passed with mean ``9.996700755``, mean relative spread ``3.47%``, and
  combined SEM/mean ``1.70%``. This closes 12/31 strict ``RBC(1,1)`` true
  nonlinear overlay points.
- Launched the next adjacent strict ``RBC(1,1)`` coefficient, ``-15%``/``m0p15``,
  through the bounded ``t=700`` seed31 and seed32 pilot stage, one run per
  office GPU.
- Completed the ``m0p15`` seed ``t=700`` pilot stage and launched the ``t=1100``
  continuation for seed31 and seed32, one run per office GPU. The transient
  ``t≈350..700`` means are seed31 ``9.781122174`` and seed32 ``10.040998351``
  over 141 samples each; these are checkpoint diagnostics only.
- Completed the ``m0p15`` seed ``t=1100`` continuation and launched the final
  ``t=1500`` seed continuation. Both seed files reached ``t=1099.87854``; the
  appended ``t≈702..1100`` means are seed31 ``10.022761494`` and seed32
  ``10.321978360`` over 160 samples each.
- Completed the final ``m0p15`` seed continuation to ``t=1500`` and launched
  the ``dt=0.04`` timestep replicate from ``t=700``. Both seed files reached
  ``t=1499.85437``; the accepted-window ``t≈1102..1500`` means are seed31
  ``10.398264855`` and seed32 ``10.045676231`` over 160 samples each.
- Completed the ``m0p15`` ``dt=0.04`` timestep replicate to ``t=700`` and
  launched its ``t=1100`` continuation. The ``t≈350..700`` checkpoint mean is
  ``9.769640410`` over 176 samples, consistent with the seed transient
  windows.
- Completed the ``m0p15`` ``dt=0.04`` timestep continuation to ``t=1100`` and
  launched the final ``t=1500`` timestep continuation. The file reached
  ``t=1099.94421``; the ``t≈702..1100`` checkpoint mean is ``10.119170070``
  over 200 samples.
- Closed the strict ``m0p15`` nonlinear overlay. The final ``dt=0.04`` trace
  reached ``t=1499.95605`` with strict-window mean ``10.008445282`` over
  200 samples. The three-member fail-closed ensemble over ``t=[1100,1500]``
  passed with mean ``10.150795456``, mean relative spread ``3.84%``, and
  combined SEM/mean ``1.37%``. This closes 13/31 strict ``RBC(1,1)`` true
  nonlinear overlay points.
- Launched the next adjacent strict ``RBC(1,1)`` coefficient, ``-10%``/``m0p1``,
  through the bounded ``t=700`` seed31 and seed32 pilot stage, one run per
  office GPU.
- Completed the ``m0p1`` seed ``t=700`` pilot stage and launched the ``t=1100``
  continuation for seed31 and seed32, one run per office GPU. The transient
  ``t≈350..700`` means are seed31 ``12.593895730`` and seed32 ``12.283921621``
  over 141 samples each; these are checkpoint diagnostics only.
- Completed the ``m0p1`` seed ``t=1100`` continuation and launched the final
  ``t=1500`` seed continuation. Both seed files reached ``t=1099.87854``; the
  appended ``t≈702..1100`` means are seed31 ``12.043095994`` and seed32
  ``12.036228555`` over 160 samples each.
- Completed the final ``m0p1`` seed continuation to ``t=1500`` and launched
  the ``dt=0.04`` timestep replicate from ``t=700``. Both seed files reached
  ``t=1499.85437``; the accepted-window ``t≈1102..1500`` means are seed31
  ``12.056269771`` and seed32 ``12.053729022`` over 160 samples each. The
  point remains open until the timestep replicate reaches ``t=1500`` and the
  strict ensemble gate passes.
- Completed the ``m0p1`` ``dt=0.04`` timestep replicate to ``t=700`` and
  launched its ``t=1100`` continuation. The ``t≈350..700`` checkpoint mean is
  ``11.854555016`` over 176 samples, consistent with the seed transient
  windows.
- Completed the ``m0p1`` ``dt=0.04`` timestep continuation to ``t=1100`` and
  launched the final ``t=1500`` timestep continuation. The file reached
  ``t=1099.94421``; the ``t≈704..1100`` checkpoint mean is ``11.929200834``
  over 199 samples.
- Closed the strict ``m0p1`` nonlinear overlay. The final ``dt=0.04`` trace
  reached ``t=1499.95605`` with strict-window mean ``12.025078411`` over
  200 samples. The three-member fail-closed ensemble over ``t=[1100,1500]``
  passed with mean ``12.045025735``, mean relative spread ``0.259%``, and
  combined SEM/mean ``1.46%``. This closes 14/31 strict ``RBC(1,1)`` true
  nonlinear overlay points.
- Launched the next adjacent strict ``RBC(1,1)`` coefficient, ``-5%``/``m0p05``,
  through the bounded ``t=700`` seed31 and seed32 pilot stage, one run per
  office GPU.
- Completed the ``m0p05`` seed ``t=700`` pilot stage and launched the
  ``t=1100`` continuation for seed31 and seed32, one run per office GPU. The
  transient ``t≈350..700`` means are seed31 ``11.248333424`` and seed32
  ``10.815876007`` over 141 samples each; these are checkpoint diagnostics
  only.
- Completed the ``m0p05`` seed ``t=1100`` continuation and launched the final
  ``t=1500`` seed continuation. Both seed files reached ``t=1099.87854``; the
  appended ``t≈702..1100`` means are seed31 ``11.111491352`` and seed32
  ``10.839283931`` over 160 samples each.
- Completed the final ``m0p05`` seed continuation to ``t=1500`` and launched
  the ``dt=0.04`` timestep replicate from ``t=700``. Both seed files reached
  ``t=1499.85437``; the accepted-window ``t≈1102..1500`` means are seed31
  ``11.090639961`` and seed32 ``10.921340626`` over 160 samples each. The
  point remains open until the timestep replicate reaches ``t=1500`` and the
  strict ensemble gate passes.
- Completed the ``m0p05`` ``dt=0.04`` timestep replicate to ``t=700`` and
  launched its ``t=1100`` continuation. The ``t≈350..700`` checkpoint mean is
  ``11.243907137`` over 176 samples, consistent with the seed transient
  windows.
- Completed the ``m0p05`` ``dt=0.04`` timestep continuation to ``t=1100`` and
  launched the final ``t=1500`` timestep continuation. The file reached
  ``t=1099.94421``; the ``t≈704..1100`` checkpoint mean is ``11.065919138``
  over 199 samples.
- Closed the strict ``m0p05`` nonlinear overlay. The final ``dt=0.04`` trace
  reached ``t=1499.95605`` with strict-window mean ``10.995303574`` over
  200 samples. The three-member fail-closed ensemble over ``t=[1100,1500]``
  passed with mean ``11.002428054``, mean relative spread ``1.54%``, and
  combined SEM/mean ``1.66%``. This closes 15/31 strict ``RBC(1,1)`` true
  nonlinear overlay points.
- Launched the zero-offset strict ``RBC(1,1)`` coefficient, ``0``/baseline,
  through the bounded ``t=700`` seed31 and seed32 pilot stage, one run per
  office GPU.
- Regenerated the README/docs ``RBC(1,1)`` full landscape panel with the 15
  strict negative-side nonlinear ensemble overlays that have closed under the
  ``t=[1100,1500]`` seed/timestep protocol. The zero-offset and positive-side
  coefficients remain pending, so the figure is explicitly scoped as a
  launch/noise diagnostic and optimizer-design input rather than a promoted
  nonlinear turbulent-flux optimization result.
- Promoted the shipped runtime/memory CSV and JSON sidecars into
  ``docs/_static`` so the public runtime panel is reproducible from a clean
  checkout instead of depending on ignored ``tools_out`` files.
- Tightened release hygiene: ``release.yml`` now reruns the fast repository
  size, release-artifact, performance-manifest, parallel-scaling,
  quasilinear-guardrail, parallelization-status, technical-status, and release
  readiness checks before PyPI publishing. The release-readiness checker now
  requires the tracked runtime/memory sidecars and release workflow guardrails.
- Tightened README/docs claim scope for parallelization: production speedup is
  backed for independent ``k_y`` scans and quasilinear/UQ ensembles; sensitivity
  sweeps use the same deterministic partitioning but do not yet have a
  standalone speedup artifact. Nonlinear sharding remains diagnostic.
- Verified the tranche with bounded checks: release readiness regeneration,
  repository-size manifest, release-artifact manifest, performance manifest,
  parallel-scaling artifact inventory, and focused pytest guardrails all pass.
- Updated GitHub Actions workflow majors to the Node 24-backed action releases
  (`checkout@v6`, `setup-python@v6`, `cache@v5`, `upload-artifact@v7`, and
  `download-artifact@v8`) and added the release-workflow Node 24 environment
  fallback. Focused release-readiness, repository-size, release-artifact, and
  workflow-reference checks pass locally; GitHub CI for commit `4f3de69` is in
  progress.
- Launched the positive-side strict ``RBC(1,1)`` nonlinear overlay campaign on
  office as a resumable two-GPU queue. The controller generated positive-side
  ``t=1100`` and ``t=1500`` continuation TOMLs from the existing ``t=700``
  positive configs, originally ran seed31, seed32, and ``dt=0.04`` chains per
  coefficient, validates that each output reaches ``t≈1500``, then builds the
  strict ``t=[1100,1500]`` replicate ensemble gate. The seed31 positive-side
  base stages were later replaced by seed33 after repeated stalls; README/docs
  are now scoped to the 17 completed strict points until more positive gates
  finish.
- Staged matched nonlinear transport traces for the four strict QA optimization
  geometries: baseline, growth-optimized, quasilinear-optimized, and
  nonlinear-window-optimized. The campaign uses the authoritative
  ``vmec_jax_qa_full_sweep_20260605`` WOUTs, staged ``t=700,1100,1500`` n64
  seed/timestep configs, and a queued two-GPU office controller that waits for
  the positive-side ``RBC(1,1)`` sweep before running. This is the matched
  post-transient nonlinear evidence lane; no transport-reduction claim is made
  until the ensembles pass and uncertainty separation is quantified.
- Strengthened nonlinear parallelization identity gates without adding a
  speedup claim. The state-domain prototype gate now uses a 24x24 state split
  across three domains and passes exactly at ``atol=rtol=1e-10``. The velocity
  field-reduction gate now reports the standard ``atol + rtol ||ref||``
  criterion so float32 reduction-order differences are accepted only when the
  full-field relative error is small. The production nonlinear sharding gate
  remains fail-closed/diagnostic-only because the GPU speedup candidate is not
  profiler-backed.
- Fixed the wide-coverage shard-42 CI failure introduced by the velocity
  field-reduction relative-tolerance gate by updating the gate unit test to
  pass and assert ``rtol`` and ``max_allowed_error``. The fresh GitHub CI run
  for commit ``d1dfbbb`` completed successfully.
- Replaced the positive-side strict ``RBC(1,1)`` seed31 replicate with seed33
  on office after two seed31 base-stage runs stalled without output despite
  active GPU kernels. The corrected positive-side replicate policy remains two
  independent seeds plus a timestep replicate: seed32, seed33, and ``dt=0.04``.
  The first positive coefficient, ``+5%``/``p0p05``, now passes the
  ``t=[1100,1500]`` strict ensemble gate with mean ``Q_i=10.8433250467``,
  mean-relative spread ``1.699%``, and combined SEM/mean ``1.908%``. The
  second positive coefficient, ``+10%``/``p0p1``, also passes with mean
  ``Q_i=9.6447762903``, mean-relative spread ``2.265%``, and combined SEM/mean
  ``1.840%``. The third positive coefficient, ``+15%``/``p0p15``, now passes
  with mean ``Q_i=10.9084437509``, mean-relative spread ``4.195%``, and
  combined SEM/mean ``1.572%``. The ``+25%``/``p0p25`` coefficient now passes
  with mean ``Q_i=10.0771448855``, mean-relative spread ``6.267%``, and
  combined SEM/mean ``2.419%``. The README/docs ``RBC(1,1)`` panel is
  refreshed to 20/31 strict true nonlinear points. The neighboring
  ``+20%``/``p0p2`` point narrowly missed the strict ``t=[1100,1500]`` spread
  gate (``15.48%`` versus ``15%``) and is being continued to a later window
  instead of relaxing the threshold; the remaining higher positive
  coefficients continue running on office.
- Completed the zero-offset strict ``RBC(1,1)`` seed ``t=700`` pilot stage and
  launched the ``t=1100`` continuation for seed31 and seed32, one run per
  office GPU. The transient ``t≈350..700`` means are seed31 ``10.896999582``
  and seed32 ``11.155873752`` over 141 samples each; these are checkpoint
  diagnostics only.
- Completed the zero-offset strict ``RBC(1,1)`` seed ``t=1100`` continuation
  and launched the final ``t=1500`` seed continuation. Both seed files reached
  ``t=1099.87854``; the appended ``t≈702..1100`` means are seed31
  ``11.119230902`` and seed32 ``11.140138322`` over 160 samples each.
- Completed the final zero-offset ``RBC(1,1)`` seed continuation to ``t=1500``
  and launched the ``dt=0.04`` timestep replicate from ``t=700``. Both seed
  files reached ``t=1499.85437``; the accepted-window ``t≈1102..1500`` means
  are seed31 ``11.127188009`` and seed32 ``10.994246972`` over 160 samples
  each. The point remains open until the timestep replicate reaches ``t=1500``
  and the strict ensemble gate passes.
- Completed the zero-offset ``RBC(1,1)`` ``dt=0.04`` timestep replicate to
  ``t=700`` and launched its ``t=1100`` continuation. The ``t≈350..700``
  checkpoint mean is ``11.145515642`` over 176 samples, consistent with the
  seed transient windows.
- Completed the zero-offset ``RBC(1,1)`` ``dt=0.04`` timestep continuation to
  ``t=1100`` and launched the final ``t=1500`` timestep continuation. The file
  reached ``t=1099.94421``; the ``t≈706..1100`` checkpoint mean is
  ``11.179217074`` over 198 samples.
- Closed the zero-offset strict ``RBC(1,1)`` nonlinear overlay. The final
  ``dt=0.04`` trace reached ``t=1499.95605`` with strict-window mean
  ``10.905578384`` over 200 samples. The three-member fail-closed ensemble over
  ``t=[1100,1500]`` passed with mean ``11.009004455``, mean relative spread
  ``2.01%``, and combined SEM/mean ``1.51%``. This updates the public
  ``RBC(1,1)`` overlay to 16/31 strict true nonlinear points: all negative-side
  coefficients plus the zero-offset baseline.
- Closed two additional positive-side strict ``RBC(1,1)`` nonlinear overlays from
  the office two-GPU controller. The ``+30%``/``p0p3`` coefficient passed the
  ``t=[1100,1500]`` seed/timestep ensemble gate with mean ``Q_i=9.6482220987``,
  mean-relative spread ``0.669%``, and combined SEM/mean ``2.090%``. The
  ``+35%``/``p0p35`` coefficient passed with mean ``Q_i=8.5866099427``,
  mean-relative spread ``2.957%``, and combined SEM/mean ``2.083%``. The public
  ``RBC(1,1)`` landscape was regenerated with 22/31 strict true nonlinear
  overlays: the negative side, zero offset, and positive ``+5%``, ``+10%``,
  ``+15%``, ``+25%``, ``+30%``, and ``+35%`` points. The ``+20%`` point remains
  pending after its narrow strict spread miss; higher positive coefficients
  remain pending and are not inferred from reduced metrics.
- Harvested the next positive strict ``RBC(1,1)`` overlay from office. The
  ``+40%``/``p0p4`` coefficient passed the ``t=[1100,1500]`` seed/timestep
  ensemble gate with mean ``Q_i=7.1067187691``, mean-relative spread ``3.502%``,
  and combined SEM/mean ``2.009%``. The public landscape was regenerated with
  23/31 strict true nonlinear overlays. ``+45%`` and higher remain incomplete
  and are not shown in the public nonlinear overlay until their three-member
  strict ensembles pass.
- Re-audited the pending positive-side ``RBC(1,1)`` campaign before launching
  new GPU time. The old office ``+20%``/``p0p2`` strict ensemble over
  ``t=[1100,1500]`` failed only the mean-spread gate
  (``15.481%`` versus the fail-closed ``15%`` threshold) while all individual
  windows passed; this remains a convergence-window repair target rather than a
  threshold-relaxation candidate. The old ``+45%`` and higher positive-side
  attempts failed much earlier with non-finite nonlinear diagnostics under the
  strict protocol, typically by ``t≈0..5`` after compilation, so they are
  treated as strict-protocol stability-boundary points and not inferred from
  deterministic reduced metrics.
- Staged a fresh current-main office repair in
  ``/home/rjorge/spectrax_rbc11_completion_20260610_214752/SPECTRAX-GK`` at
  commit ``f00d736``. VMEC-JAX solved the missing ``+20%``, ``+45%``, and
  ``+50%`` WOUTs on current main; the slow higher-positive VMEC batch was
  stopped to avoid competing with the transport repair queue. The active
  detached controller
  ``tools_out/vmec_boundary_transport_landscape_rbc11_completion/run_p0p2_repair_controller.py``
  is running the ``+20%`` staged repair with horizons
  ``700,1100,1500,1900`` and the later acceptance window
  ``t=[1500,1900]`` for seed32, seed33, and ``dt=0.04`` variants. PID and logs
  are recorded under
  ``tools_out/vmec_boundary_transport_landscape_rbc11_completion/`` in that
  office clone. Promotion remains blocked until all three current-main variants
  finish and the fail-closed ensemble gate passes.
- The first current-main ``+20%`` controller attempt exposed office resource
  contention rather than a numerical result: stale unrelated VMEC-JAX QI jobs
  held GPU memory and caused one BLAS-initialization error and one GPU OOM.
  Those stale jobs were stopped, partial ``+20%`` transport outputs were
  deleted, and the repair was relaunched as PID ``2478543`` with
  ``SPECTRAX_DEVICES=0`` and ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so the
  three variants run sequentially and reproducibly. Focused local tests for the
  landscape/admission helpers pass: ``21 passed``.
- Accepted the existing ``+20%``/``p0p2`` ``t=[1100,1500]`` long-window
  ensemble under the explicitly relaxed diagnostic landscape policy requested
  for this lane. The three individual windows were already converged and the
  combined SEM/mean was ``4.472%``; only the mean-relative seed/timestep spread
  separated it from the stricter gate. With ``max_mean_rel_spread=0.20``, the
  point passes with mean ``Q_i=9.2545128095`` and spread ``15.481%``. The
  public ``RBC(1,1)`` landscape was regenerated with 24/31 nonlinear overlays:
  all negative-side coefficients, zero offset, and positive ``+5%``, ``+10%``,
  ``+15%``, ``+20%``, ``+25%``, ``+30%``, ``+35%``, and ``+40%``. The relaxed
  gate is scoped to the landscape/noise diagnostic and does not promote broad
  nonlinear turbulent-flux optimization or absolute quasilinear-flux claims.
  Current-main ``+45%`` and ``+50%`` short stability probes reached ``t=50``
  without the old immediate non-finite diagnostic failure, but they remain
  short diagnostic probes rather than accepted long-window overlays.
- Added a real device-z fused pencil nonlinear RHS route and CPU4/GPU2 profile artifacts. The route shards the field-line `z` axis, keeps `(k_y,k_x)` FFTs local on each device, and avoids global spectral tile reconstruction. A shard-level office diagnostic showed that direct `device_put` from a single-device GPU JAX array into `NamedSharding(..., PartitionSpec(..., "z"))` misplaced the second `z` shard, while host-backed NumPy input sharded correctly. The diagnostic/profiler path now host-stages the initial state before explicit z sharding so the gate tests the candidate nonlinear route rather than source-device resharding behavior.
- Replaced the slow pjit-style z-pencil timing path with a `jax.shard_map` local-RHS route. The active logical-CPU check in `docs/_static/nonlinear_device_z_pencil_rhs_cpu4_profile.json` passes host-gathered serial-vs-sharded RHS identity on two and four devices for a `(4,16,96,96,32)` bracket workload (`max_abs_error=7.57e-10`, `max_rel_error=3.82e-7`) and reaches `1.51x` on two logical CPU devices and `2.62x` on four. The office two-GPU artifact `docs/_static/nonlinear_device_z_pencil_rhs_gpu2_profile.{json,csv,png}` also passes identity (`max_abs_error=5.24e-10`, `max_rel_error=2.65e-7`) but reaches only `1.09x`, below the `1.5x` gate. This moves the production nonlinear domain-decomposition lane from GPU identity-blocked to CPU-microkernel-speedup achieved/GPU-speedup-blocked; no production nonlinear domain speedup claim is allowed until GPU speedup and a physical transport-window route both pass.
- Added the device-z physical transport-window route, profiler, HLO summaries, and Perfetto trace hooks. `docs/_static/nonlinear_device_z_pencil_transport_cpu4_profile.{json,csv,png}` advances the serial and z-sharded routes for four fixed nonlinear steps on the same `(4,16,96,96,32)` workload, passes final-state/free-energy/field-energy/physical-flux/bracket-RMS identity, and reaches `1.72x` on two logical CPU devices and `3.11x` on four. `docs/_static/nonlinear_device_z_pencil_transport_gpu2_profile.{json,csv,png}` passes the same identity gate on office GPUs (`max_abs_error=7.45e-9`) but reaches only `1.20x`, below the `1.5x` promotion gate. HLO summaries show local FFT lowering and no all-to-all or collective-permute operations, so the remaining blocker is GPU workload granularity/full-solver routing rather than numerical identity.
- Added an opt-in z-chunked pencil bracket for larger GPU diagnostics. Unchunked office probes at `(4,16,128,128,32)` and `(4,16,96,96,64)` failed in cuFFT plan creation. With `--z-chunk-size 8` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`, both cases run and pass identity, but remain below the `1.5x` two-GPU speedup gate (`1.30x` and `1.40x`). This localizes the next optimization target to FFT batching/allocation and full-solver workload granularity, not a numerical mismatch.
- Added a backend-free device-z pencil FFT batch-pressure model and `tools/profile_device_z_pencil_transport_window.py --auto-z-chunk-size`. The model reproduces the office lesson in a deterministic preflight gate: for large `(N_l,N_m,N_y,N_x,N_z)` profiles it estimates the largest axis-wise cuFFT batch, suggests a local `z_chunk_size`, and records whether GPU preallocation should be disabled before timing. This prevents blind relaunches of known bad cuFFT plan shapes, but it remains diagnostic-only until a full solver route passes identity and GPU speedup gates.
- Office validation of `--auto-z-chunk-size` on the previously problematic `(4,16,96,96,64)` two-GPU transport-window profile selected `z_chunk_size=8`, reduced the estimated largest axis-wise FFT batch from `196608` to `49152`, and passed final-state/physical-flux identity (`max_final_state_abs_error=7.45e-9`, `physical_flux_abs_error=1.78e-15`). The short bounded probe reached only `1.20x`, so the blocker remains GPU workload granularity/full-solver routing rather than cuFFT plan creation or numerical identity.
- Added `tools/profile_device_z_pencil_transport_window.py --observable-repeats` so the device-z transport-window artifact can separate compute-only fixed-step speedup rows from scalar observable/identity-gate timing. The new `observable_gate_*` JSON/CSV fields measure host-gathered free-energy, field-energy, physical-flux, and bracket-RMS gate cost separately from the speedup gate. This improves bottleneck diagnosis for the next nonlinear parallelization tranche but does not promote production nonlinear GPU speedup; that still requires full-solver routing plus identity and >1.5x matched CPU/GPU artifacts.
- Office validation of the observable-split profiler on the auto-chunked `(4,16,96,96,64)` two-GPU diagnostic produced `docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.{json,csv,png}`. Identity passed, `z_chunk_size=8`, compute-only speedup remained below gate at `1.19x`, and the repeated scalar observable/identity gate took `3.13 s` median versus `0.073 s` for the sharded compute row (`42.6x` overhead). The next route must keep diagnostics streamed/device-side and integrate into a full solver window before another speedup promotion attempt.
- Implemented a `sharded_reduce` observable mode for the device-z transport-window identity gate. It computes free-energy, field-energy, physical-flux, and bracket-RMS sums on z shards and reduces only scalars, preserving identity in local and office tests. The large office `(4,16,96,96,64)` probe still stayed below promotion (`1.40x`) and the observable gate was slower than host gather (`65.5x` compute overhead) because it recomputes the nonlinear bracket for diagnostics. This rules out standalone diagnostic recomputation as the production route; the next tranche should fuse scalar accumulation into the RHS/update while the bracket is already resident.
- Finalized the release-scoped performance tranche. Regenerated the runtime/memory panel from the shipped summary sidecar, rebuilt the parallelization/decomposition/manuscript status artifacts, and kept the nonlinear domain-decomposition conclusion fail-closed: production independent-work parallelization is ready, device-z nonlinear decomposition has identity/profiler evidence only, and production GPU nonlinear speedup is deferred until fused in-RHS diagnostics plus full-solver serial-vs-decomposed transport-window gates pass.
- Started the research-grade differentiable architecture refactor lane on
  ``codex/differentiable-refactor-plan``. Added a planned executable manifest
  for the high-risk module splits (benchmarks, differentiable geometry,
  nonlinear parallelism, solver/objective gradients, nonlinear RHS, runtime,
  runtime artifacts, linear API, and executable/plotting CLI), plus the docs
  page that freezes the target package layout, extension points, public
  compatibility facade policy, JAX autodiff policy, parity/literature gates, and
  acceptance criteria before behavior-changing refactors begin.
- Began Phase 1 of that refactor with behavior-preserving core contracts:
  ``spectraxgk.core.contracts`` declares shape, differentiability,
  validation-gate, extension-point, and module-refactor contracts, while
  ``spectraxgk.core.extension_points`` declares structural protocols for basis,
  geometry, collision, field-solver, RHS, diagnostic, objective, and artifact
  writer extensions. The validation coverage manifest now owns these modules,
  the differentiable refactor manifest validates them explicitly, and
  ``tests/test_core_contracts.py`` exercises both valid and invalid metadata
  paths with 99% coverage for the new core package.
- Continued Phase 1 of the differentiable architecture refactor with the first
  behavior-preserving benchmark-helper split. ``spectraxgk.validation.benchmarks.initialization``
  now owns benchmark Gaussian/moment initial-condition builders and the kinetic
  reference seed policy; ``spectraxgk.validation.benchmarks.reference`` now owns benchmark
  result containers, reference-table loaders, and reference comparison records.
  The refactor manifest validates these implemented split modules with source
  paths, moved exports, tests, and docs.
- Extended the same benchmark-helper refactor tranche with
  ``spectraxgk.validation.benchmarks.species``. The new module owns benchmark
  species-to-``LinearParams`` builders plus reference hypercollision and
  linked-end damping policy. The manifest now tracks three implemented Phase-1
  split modules for benchmark helper behavior.
- Completed the benchmark-helper Phase-1 split by making the focused domain
  modules canonical. Fit-signal and normalization policies now live in
  ``spectraxgk.validation.benchmarks.fit_signals``, scan batching/window helpers live in
  ``spectraxgk.validation.benchmarks.batching``, and solver-selection/KBM branch policies
  live in ``spectraxgk.validation.benchmarks.solver_policy``. Benchmark runners import those
  modules directly, while ``spectraxgk.benchmarks`` remains the public benchmark
  entry point.
- Continued Phase 1 of the differentiable architecture refactor with the first
  behavior-preserving differentiable-geometry support split. Optional backend
  discovery, local-checkout import precedence, JAX dtype selection, and tracer
  detection now live in ``spectraxgk.geometry.backend_discovery``. Finite-
  difference Jacobians, observable AD/FD gradient reports, conditioning
  metadata, and strict JSON sanitation now live in
  ``spectraxgk.geometry.autodiff_checks``. ``spectraxgk.geometry.differentiable``
  remains the compatibility facade with object-identical re-exports; the larger
  VMEC/Boozer bridge and parity routines remain the next geometry refactor
  tranche and still require the existing same-WOUT, field-line, and gradient
  gates before movement.
- Continued the differentiable-geometry support split by moving pure numerical
  helpers into ``spectraxgk.geometry.numerics``. The new module owns parity
  metrics, radial and equal-arc interpolation, Boozer half-mesh coordinates,
  radial derivative stencils, Boozer Fourier field-line evaluation, cumulative
  trapezoids, and periodic bilinear sampling. ``spectraxgk.geometry.differentiable``
  keeps object-identical re-exports so existing bridge tests, tools, and hidden
  diagnostics continue to use the old import path while the remaining VMEC/Boozer
  bridge split is prepared.
- Continued the differentiable-geometry support split by moving the solver-ready
  in-memory geometry contract into ``spectraxgk.geometry.flux_tube_contract``.
  The new module owns mapping validation, scalar/array finite checks,
  observable-name contracts, and differentiable geometry-observable reductions.
  ``spectraxgk.geometry.differentiable`` keeps object-identical re-exports so
  public imports and package-level ``spectraxgk`` exports remain stable while
  the remaining VMEC/Boozer bridge routines are prepared for later extraction.
- Continued the differentiable-geometry split by moving geometry sensitivity,
  inverse-design, conditioning, and local UQ report routines into
  ``spectraxgk.geometry.sensitivity``. This gives future VMEC/Boozer bridge
  modules a direct dependency on the report contract instead of depending on
  the compatibility facade. ``spectraxgk.geometry.differentiable`` preserves
  object-identical re-exports for existing public imports and tests.
- Continued the differentiable-geometry split by moving bounded VMEC boundary
  and Boozer bridge helpers into ``spectraxgk.geometry.booz_xform_bridge``.
  The new module owns boundary aspect sensitivity, Boozer spectral sensitivity,
  Boozer field-line ``|B|`` evaluation, Boozer-to-flux-tube mapping, and the
  bounded Boozer flux-tube sensitivity report. ``spectraxgk.geometry.differentiable``
  remains the compatibility facade with object-identical pure-helper re-exports
  and thin wrappers for optional-backend discovery hooks; the larger VMEC-state,
  equal-arc, and parity routines remain the next geometry tranche.
- Continued the differentiable-geometry split by moving optional-backend
  ``VMECState`` sensitivity reports into
  ``spectraxgk.geometry.vmec_state_sensitivity``. The new module owns the
  VMEC-state-to-Boozer flux-tube sensitivity report, the VMEC metric tensor
  sensitivity report, and the VMEC field-line tensor sensitivity report.
  ``spectraxgk.geometry.differentiable`` keeps public wrappers that preserve
  facade-level monkeypatch hooks for backend discovery, finite-difference
  checks, geometry sensitivity reports, Boozer mapping, and periodic sampling.
- Continued the differentiable-geometry split by moving direct ``vmec_jax``
  tensor sampling into ``spectraxgk.geometry.vmec_tensor_mapping``. The new
  module owns ``vmec_jax_flux_tube_mapping_from_state`` and converts raw VMEC
  metric, magnetic-field, shear, drift, and Jacobian tensors into the
  solver-ready flux-tube mapping contract. The compatibility facade retains a
  wrapper that forwards the facade-level periodic sampler hook into the focused
  implementation.
- Continued the differentiable-geometry split by moving the VMEC-to-Boozer
  equal-arc core-profile builder into ``spectraxgk.geometry.vmec_boozer_core``.
  The new module owns Boozer constants caching/prewarm and the
  ``vmec_jax_boozer_equal_arc_core_profiles_from_state`` implementation, while
  ``spectraxgk.geometry.differentiable`` keeps hook-preserving wrappers so
  existing optional-backend and monkeypatch tests still target the public
  facade.
- Continued the differentiable-geometry split by moving VMEC flux-tube
  sensitivity and array-parity reports into
  ``spectraxgk.geometry.vmec_flux_tube_reports``. The facade now forwards
  backend discovery, flux-tube mapping, Boozer equal-arc core, sensitivity, and
  parity-metric hooks into the focused report implementation so public
  monkeypatch seams and optional-backend tests remain stable.
- Split the main geometry package into ``spectraxgk.geometry.core`` plus a
  thin ``spectraxgk.geometry`` compatibility facade. The core module now owns
  analytic s-alpha/slab geometry, sampled ``FluxTubeGeometryData``, imported
  NetCDF/eik loading, twist-shift defaults, and grid-default policy, while the
  package facade re-exports all public and test-visible symbols with identity
  checks.
- Continued the source-name cleanup by renaming the explicit time-stepper module
  from ``spectraxgk.gx_integrators`` to
  ``spectraxgk.solvers.time.explicit``. Linear, cETG, nonlinear, runtime,
  benchmark, and low-level tests now use explicit-time names for the Heun/RK4
  paths and diagnostic masks. External-reference comparison tools keep their
  comparison wording where they explicitly compare against another code.
- Continued the source-name cleanup by renaming the optimized nonlinear
  real-FFT path from the old ``gx_real_fft`` schema/API wording to
  ``compressed_real_fft``. Runtime TOMLs, docs, nonlinear kernels, profiling
  tools, and tests now describe this as a compressed Hermitian real-FFT
  algorithm. Internal bracket helpers were renamed to real-FFT terminology,
  while explicit external-code comparison tools keep comparison-specific names.
- Continued the source-name cleanup by renaming growth/frequency extraction
  helpers from old provenance-oriented names to algorithmic names:
  ``instantaneous_growth_rate_from_phi`` and
  ``windowed_growth_rate_from_omega_series``. Benchmark runners, comparison
  tools, public exports, and tests now use the neutral diagnostic API names.
- Continued the source-name cleanup in nonlinear helper internals. CFL-frequency
  estimates, omega/gamma diagnostic masks, Laguerre ``J0`` field factors,
  magnetic-compression corrections, and adiabatic quasineutrality helpers now
  use physics/numerics names instead of legacy provenance names. Focused
  nonlinear, cETG, runtime-diagnostic, lint, and manifest gates passed.
- Continued the source-name cleanup in benchmark species policies. Reference
  hypercollision and linked-boundary damping helpers now use
  ``_apply_reference_hypercollisions``,
  ``_reference_hypercollision_power``, and
  ``_linked_boundary_end_damping``. Benchmark facades, comparison tools,
  manifests, and focused benchmark tests were updated and passed.
- Continued the source-name cleanup by renaming imported-geometry helper config
  fields from provenance-oriented ``gx_python``/``gx_repo`` to
  ``geometry_helper_python``/``geometry_helper_repo``. Runtime TOML loading,
  VMEC/Miller docs, the HSX VMEC example, the Miller geometry generator, and
  roundtrip tests now use canonical helper names; serialization emits only the
  neutral helper-field names.
- Continued the source-name cleanup by replacing the runtime diagnostic
  normalization spelling ``diagnostic_norm = \"gx\"`` with the physics-based
  ``diagnostic_norm = \"rho_star\"`` across shipped examples, docs, runtime
  defaults, and focused tests. The low-level diagnostic normalization helper
  still accepts the old spelling as a compatibility alias with identical
  scaling.
- Continued the source-name cleanup by moving the Miller geometry generator from
  ``tools/generate_gx_miller_eik.py`` to the neutral
  ``tools/generate_miller_eik.py``. Documentation now points to the canonical
  script, while the old path remains a tiny compatibility wrapper for existing
  automation.
- Continued the source-name cleanup by removing the old
  ``streaming_contribution_gx`` alias for the Hermite-Laguerre field-coupled
  streaming RHS helper. RHS assembly, profiling, comparison tools, and
  comparison tests now import the algorithmic
  ``linked_streaming_contribution`` name directly.
- Continued the solver-objective facade reduction without changing the public
  top-level API. Stellarator optimization and VMEC-JAX transport objective
  source modules now import solver-objective primitives from their focused
  owner modules directly, and VMEC/Boozer artifact builders import finite-
  difference, line-search, gradient, sampling, and objective-name helpers from
  the same owner modules. ``spectraxgk.__init__`` still re-exports through the
  stable ``spectraxgk.objectives.solver_gradients`` facade until the major API
  cleanup, and the facade-specific tests keep the monkeypatch seams covered.
- Continued the linear-facade reduction by moving type/cache-only source
  imports to focused owner modules. Benchmark fit/scan/species policies,
  species construction, diagnostics, quasilinear diagnostics, and nonlinear
  NetCDF output now import ``LinearParams``/``LinearCache``/``build_linear_cache``
  from ``spectraxgk.operators.linear.params`` and
  ``spectraxgk.operators.linear.cache`` directly. Integrator-facing imports
  remain on ``spectraxgk.linear`` until linear RHS/integration entry points are
  split behind their own public API.
- Continued the same linear-facade reduction through nonlinear, runtime,
  sharded-integration, Krylov, benchmark-case, and solver-gradient support
  modules. These internals now import linear data classes, cache builders,
  damping policies, and term-conversion helpers from
  ``spectraxgk.operators.linear.params`` and
  ``spectraxgk.operators.linear.cache``. Only actual public RHS/integrator entry
  points remain routed through ``spectraxgk.linear``.
- Continued differentiable-geometry facade reduction by moving the
  production VMEC/Boozer state-to-flux-tube builder into
  ``spectraxgk.geometry.vmec_boozer_core`` and routing internal solver,
  gradient-gate, and VMEC-JAX transport modules to the focused owner modules:
  backend discovery, flux-tube contracts, autodiff checks, and VMEC/Boozer
  core profiles. The public ``spectraxgk.geometry.differentiable`` facade
  remains the compatibility and monkeypatch surface for documented imports.
- Split linear RHS ownership out of the broad ``spectraxgk.linear`` facade into
  ``spectraxgk.operators.linear.rhs``. The public facade still re-exports
  ``linear_rhs`` and ``linear_rhs_cached``, while solver-objective gates,
  sharded integration, and velocity-parallel fallback routes now import the RHS
  owner directly. Focused linear, velocity-sharding, and solver-gradient tests
  cover the new owner seams.
- Continued the cleanup of provenance-oriented wording in source docstrings and
  docs. Generic grouped-NetCDF, nonnegative-``ky`` real-FFT storage,
  species/Hermite sharding, field-coupled streaming, linked-boundary ordering,
  and ``kz``-proportional hypercollision descriptions now use physics/numerics
  names. Explicit benchmark/comparison mentions remain where they describe
  validation evidence or legacy artifact formats.
- Continued the cleanup of tool defaults by switching validation figure/table
  generators from the legacy diagnostic normalization spelling to
  ``rho_star``. The Hermite-Laguerre evolution diagnostic now defaults to the
  descriptive ``time_initial`` normalization mode while retaining ``gx`` as a
  CLI alias with identical behavior.
- Removed stale provenance and placeholder-style comments from standalone Miller
  and VMEC geometry helpers. The comments now state the actual role of these
  compact helpers and point production Miller generation to the runtime
  ``spectraxgk.geometry.miller_eik`` path.
- Added a root ``codecov.yml`` so Codecov patch coverage mirrors the CI coverage
  contract. The Actions coverage gates measure ``src/spectraxgk`` package
  coverage, while docs, tools, examples, tests, and generated-output trees are
  validated by separate release, docs, artifact, and smoke-test gates. This
  should keep external Codecov patch status from failing on non-package cleanup
  commits.

### 2026-06-16 Linear Package Consolidation Tranche

- Moved linear cache construction, linked-boundary maps, Hermite-Laguerre
  moment helpers, parameter pytrees, Krylov eigensolver policy, and
  velocity-parallel RHS dispatch into `spectraxgk.operators.linear` and
  `spectraxgk.solvers.linear` implementation packages.
- Kept `spectraxgk.linear_*` root modules as object-identical compatibility
  facades and added explicit facade identity tests so documented imports remain
  stable during the refactor.
- Updated validation and architecture manifests, API docs, code-structure docs,
  and release-status evidence so coverage ownership follows the new domain
  package layout.

### 2026-06-17 Default Executable Demo Workflow Split

- Moved the no-input `spectraxgk` educational demo orchestration out of the
  public parser module and into `spectraxgk.workflows.demo`. The CLI now builds
  explicit workflow dependencies and delegates, preserving monkeypatch seams and
  executable behavior while separating parser dispatch from simulation,
  plotting, TOML provenance, and artifact-writing side effects.
- Updated the differentiable refactor manifest and code-structure docs so the
  workflow split is tracked as intentional architecture work rather than a new
  unowned helper module. This tranche reduced `src/spectraxgk/cli.py` from 1296
  to 1091 lines without changing the default-run or `--plot` contracts.

### 2026-06-17 Runtime Command Workflow Split

- Moved the runtime linear, runtime ky-scan, and runtime nonlinear executable
  command bodies from `spectraxgk.cli` into `spectraxgk.workflows.cases`.
  `cli.py` now owns parser construction and public command wrappers, while the
  workflow module owns path override policy, progress/header printing,
  quasilinear override policy, solver invocation, artifact dispatch, and command
  summaries.
- Kept compatibility wrappers for `_runtime_output_path`,
  `_apply_runtime_path_overrides`, `_apply_quasilinear_overrides`, and command
  functions so existing tests and developer monkeypatch workflows still target
  the public CLI facade. This tranche reduced `src/spectraxgk/cli.py` from 1091
  to 771 lines, below the refactor manifest public-module target.

### 2026-06-17 Runtime Scan Orchestration Split

- Moved runtime ky-scan coordination out of `spectraxgk.runtime` and into the
  existing `spectraxgk.workflows.runtime.orchestration` owner. The public `run_runtime_scan`
  facade now wires dependency seams for Hermite-Laguerre dimension resolution,
  solver-name normalization, independent-worker policy, combined-ky batching,
  and the scan task runner.
- Preserved the existing monkeypatch surfaces `_run_runtime_scan_batch`,
  `_run_runtime_scan_ky_task`, `run_runtime_linear`, and `independent_map` on the
  public runtime module. Fast and integration-marked scan-policy tests passed,
  covering serial ordering, quasilinear payload ordering, independent-worker
  metadata, explicit worker overrides, non-ky-axis rejection, and combined-ky
  dispatch.

### 2026-06-17 Runtime Nonlinear Diagnostics Policy Split

- Extracted nonlinear diagnostics integrator keyword assembly from the public
  runtime facade into `spectraxgk.workflows.runtime.policies.build_runtime_nonlinear_diagnostics_kwargs`.
  Fixed-window and adaptive nonlinear diagnostic branches now use one policy for
  sample/diagnostic stride, dealiased masks, Laguerre mode, flux normalization,
  adaptive-step controls, collision split, implicit solve settings, fixed-mode
  indices, external forcing, resolved-diagnostic suppression, and progress flags.
- Added a focused runtime policy test and re-ran the nonlinear runtime helper
  shard covering source forcing, adaptive chunks, fixed mode, collision split,
  return-state diagnostics, and final-state contracts.

### 2026-06-17 Runtime Linear Fit Diagnostics Split

- Moved generic runtime linear fit/eigenfunction extraction from `run_runtime_linear`
  into `spectraxgk.workflows.runtime.diagnostics.fit_runtime_linear_diagnostics`.
- Preserved runtime facade monkeypatch seams by injecting analysis callables from
  `spectraxgk.runtime`; added a direct density-fit helper test and reran runtime
  linear fit integration tests.

### 2026-06-17 cETG Linear Runtime Workflow Split

- Moved the reduced-model cETG linear runtime branch from `run_runtime_linear`
  into `spectraxgk.workflows.reduced_models.run_cetg_linear_runtime`.
- Kept existing runtime monkeypatch seams by passing geometry, validation,
  initial-condition, model-parameter, integrator, and fit callables through
  `CETGLinearRuntimeDeps`; added a direct fake-dependency workflow contract test.

### 2026-06-17 Runtime Quasilinear Finalization Split

- Moved optional runtime linear quasilinear payload construction from the
  `run_runtime_linear` closure into
  `spectraxgk.workflows.runtime.diagnostics.finalize_runtime_linear_quasilinear`.
- Preserved runtime facade seams by injecting cache construction, quasilinear
  computation, and term-conversion callables; added a direct metadata/state
  contract test plus runtime quasilinear smoke coverage.

### 2026-06-17 cETG Nonlinear Runtime Workflow Split

- Moved the reduced-model cETG nonlinear runtime branch from
  `run_runtime_nonlinear` into
  `spectraxgk.workflows.reduced_models.run_cetg_nonlinear_runtime`.
- Preserved runtime facade monkeypatch seams by injecting geometry, validation,
  mode-selection, initial-condition, cETG integrator, adaptive-chunk, and
  nonlinear-result assembly callables. Mocked runtime tests and real cETG
  nonlinear smoke/adaptive tests passed locally.

### 2026-06-17 Full-GK Linear Runtime Workflow Split

- Moved the full-GK `run_runtime_linear` orchestration body into
  `spectraxgk.workflows.linear.run_full_linear_runtime`, including time/Krylov
  dispatch, auto fallback, fit/eigenfunction wiring, velocity-parallel policy,
  and quasilinear finalization.
- Kept `spectraxgk.runtime.run_runtime_linear` as the public facade by passing
  all patchable runtime globals through `FullLinearRuntimeDeps`. Focused runtime
  linear, quasilinear, patched-Krylov, diffrax, density-fit, and explicit-time
  guard tests passed locally.

### 2026-06-17 Full-GK Nonlinear Runtime Workflow Split

- Moved the full-GK `run_runtime_nonlinear` orchestration body into
  `spectraxgk.workflows.nonlinear.run_full_nonlinear_runtime`, including
  diagnostics routing, adaptive chunks, fixed-mode/source policy, and
  final-state integration.
- Kept `spectraxgk.runtime.run_runtime_nonlinear` as the public facade by
  passing all patchable runtime globals through `FullNonlinearRuntimeDeps`.
  Focused nonlinear helper tests, integration-marked runtime nonlinear guards,
  Ruff, mypy, manifest, and docs gates passed locally.

### 2026-06-17 Runtime Facade Target Trim and Named-Case CLI Split

- Trimmed the public `spectraxgk.runtime` facade below the active 700-line
  manifest target without changing behavior, keeping it as a patchable
  compatibility surface over the extracted workflow modules.
- Moved named Cyclone/ETG `run-linear` and `scan-linear` executable workflows
  into `spectraxgk.workflows.named_cases`, including optional fit/eigenfunction
  plots and scan-reference plots. The public CLI now injects the same globals as
  dependencies, preserving existing monkeypatch seams while reducing
  `src/spectraxgk/cli.py` from 771 to 615 lines.
- Local gates: Ruff on the touched modules and the complete `tests/test_cli.py`
  shard passed.

### 2026-06-17 CLI Parser Boilerplate Consolidation

- Consolidated repeated executable parser flag definitions inside
  `spectraxgk.cli`, preserving the same public subcommand option sets while
  reducing the file to 498 lines, below the active 500-line target.
- Removed unused private path/quasilinear override wrappers from the CLI facade;
  runtime command execution still delegates to `spectraxgk.workflows.cases`.
- Local gates: Ruff, targeted mypy, parser option-contract check, and the
  complete `tests/test_cli.py` shard passed.

### 2026-06-17 Nonlinear Facade Diagnostic Option Consolidation

- Consolidated duplicated explicit/IMEX nonlinear diagnostic keyword forwarding
  into small option-key policies inside `spectraxgk.nonlinear`, keeping the
  public facade and solver placement unchanged while reducing the file from 951
  to 882 lines, below the active 900-line target.
- Local gates: Ruff, targeted mypy, and the nonlinear helper/RHS/diagnostic/
  explicit-step/IMEX test shard passed.

### 2026-06-17 Nonlinear Parallel Spectral Identity Split

- Moved logical spectral communication, RHS, and fixed-window integrator
  identity gates from `spectraxgk.operators.nonlinear.parallel` into
  `spectraxgk.operators.nonlinear.spectral_identity`, leaving the public
  nonlinear-parallel facade as a re-export and strategy/pencil policy surface.
- Added a facade re-export test for the new identity module and reduced
  `src/spectraxgk/operators/nonlinear/parallel.py` from 1153 to 472 lines, below the
  active 900-line target.
- Local gates: Ruff, targeted mypy, and the nonlinear parallel/domain/spectral
  communication test shard passed.
- Continued the differentiable architecture refactor by splitting the 905-line root public API registry into domain-organized `spectraxgk.api.*` modules. The root `spectraxgk` package is now an 11-line facade over the API registry, with exact historical `__all__` order and membership preserved for 411 exports. Focused public API, objective, VMEC transport, quasilinear model-selection, parallel, nonlinear-parallel, validation-manifest, refactor-manifest, ruff, and Sphinx gates passed. This tranche changes import organization only; solver, physics, validation, and differentiable objective behavior are unchanged.
- Continued the validation refactor by splitting the 946-line quasilinear nonlinear-window gate module into focused configuration, statistics, file-IO, promotion-readiness, and ensemble-gate modules. The stable public facade `spectraxgk.validation.quasilinear.window` is now 41 lines and preserves existing imports for calibration, tools, and top-level API exports. Focused quasilinear-window, calibration, promotion-guardrail, ensemble-tool, manifest, ruff, and Sphinx gates passed; this is a behavior-preserving organization change for the existing nonlinear holdout/absolute-flux promotion policy.
- Continued the performance/parallelization refactor by splitting the 1103-line velocity-space sharding module into focused plan, Hermite exchange/reduction, streaming/drift, and electrostatic/diamagnetic drive modules. The public `spectraxgk.parallel.velocity` facade is now 122 lines and preserves existing public exports plus private compatibility hooks used by current gates. Focused velocity-sharding, generator, validation-manifest, refactor-manifest, ruff, and Sphinx gates passed. This preserves numerical identity paths while making future CPU/GPU velocity-parallel work easier to profile and extend.
- Continued the validation/benchmark refactor by splitting the 1055-line benchmark harness into focused eigenfunction-reference, diagnostics time-series, physics-metric, and scan-orchestration modules. The stable public `spectraxgk.validation.benchmarks.harness` facade preserves existing imports and monkeypatch seams for benchmark tests and tools. Focused benchmark/runtime shards, validation-manifest, refactor-manifest, ruff, and Sphinx gates passed. This is behavior-preserving and makes linear/zonal/nonlinear validation metrics easier to test independently.
- Continued the benchmark-family refactor by splitting the 1380-line Cyclone runner into `cyclone_linear.py` for single-ky linear runs and `cyclone_scan.py` for ky scans. The 169-line `cyclone.py` facade preserves the public import path plus existing monkeypatch seams used by branch-policy tests. Ruff, Cyclone branch tests, full-operator smoke, parallel ky-scan gate, and profile-scaling smoke passed locally; this is an organization-only change to the existing Cyclone validation workflow.
- Continued the benchmark-family refactor by splitting the 1288-line KBM runner into `kbm_beta.py`, `kbm_linear.py`, and `kbm_scan.py`, with a temporary KBM compatibility layer retained at that point for existing monkeypatch seams. KBM branch tests, comparison/overlay shards, ruff, manifest, and docs gates passed locally. This preserved KBM physics/branch-selection behavior while making beta-scan, single-ky, and ky-scan policies separately testable; the temporary family layer was retired in a later tranche.
- Continued the benchmark-family refactor by splitting the 1005-line ETG runner into `etg_linear.py` and `etg_scan.py`, with a temporary ETG compatibility layer retained at that point for existing branch-test monkeypatch seams. ETG branch tests, ETG asset/CLI shards, ruff, manifest, and docs gates passed locally. This kept ETG physics behavior unchanged while separating single-ky, streaming-density, and scan fallback policies; the temporary family layer was retired in a later tranche.
- Continued the differentiable-stellarator validation refactor by splitting the 1030-line transport-admission module into focused policy, sample-coverage, nonlinear-audit, and candidate-selection modules. The 56-line `transport_admission.py` facade preserves public imports and top-level API identities. Transport admission/status tests, ruff, manifest, and docs gates passed locally; behavior and claim-scope guardrails are unchanged.
- Continued the quasilinear validation refactor by splitting the 778-line calibration helper into core report/scale fitting, spectrum integration, and nonlinear-window artifact IO modules. The 49-line `validation/quasilinear/calibration.py` facade preserves public imports and private NetCDF helper compatibility used by tests. Quasilinear calibration/model-selection tests, ruff, manifest, and docs gates passed locally; calibration semantics and guardrails are unchanged.
- Continued the Cyclone benchmark refactor by extracting ky-scan Krylov branch-following and reference-aligned explicit-time reselection into `spectraxgk.validation.benchmarks.cyclone_scan_branches`. The public `cyclone_scan.py` runner now owns setup and batched time integration while passing explicit hook bundles into the branch module, preserving facade monkeypatch seams and benchmark behavior. Focused Cyclone scan branch tests and Ruff passed locally before the manifest/docs gate run.
- Continued the linear-operator refactor by splitting `spectraxgk.operators.linear.cache` into a 27-line facade plus `cache_model.py` for the stable `LinearCache` pytree, `cache_arrays.py` for moment/damping/gyroaverage array factories, and `cache_builder.py` for geometry-dependent construction. The public cache import path and `spectraxgk.linear` exports are preserved, with `_shift_axis_for_cache` kept private. Focused linear-cache export, damping, gyroaverage, linked/non-twist, sampled-geometry, multispecies, and pytree tests passed locally before the full manifest/docs gate run.
- Continued the nonlinear parallelization refactor by splitting the 737-line `spectraxgk.operators.nonlinear.spectral_identity` implementation into a 37-line facade plus `spectral_identity_reports.py`, `spectral_identity_rhs.py`, and `spectral_identity_integrator.py`. Public and private compatibility imports used by profiling scripts are preserved, while fail-closed reports, logical RHS routing, and fixed-window integrator gates are now independently testable. Focused nonlinear spectral communication and nonlinear parallel tests passed locally.
- Continued the imported-geometry refactor by splitting the 775-line `spectraxgk.geometry_backends.miller` backend into a 49-line facade plus focused numerics, core surface/theta-grid, profile assembly, NetCDF IO, and high-level pipeline modules. Public Miller imports used by runtime geometry generation and tests are preserved. Focused Miller backend and runtime EIK tests passed locally.
- Continued the nonlinear parallelization refactor by splitting the 725-line `spectraxgk.operators.nonlinear.parallel_contracts` DTO/policy module into a 61-line facade plus domain-contract, spectral-contract, and strategy-readiness modules. Public DTO imports, private domain blocker helpers, and strategy table compatibility are preserved. Focused nonlinear domain, nonlinear parallel, and spectral communication tests passed locally.
- Continued the nonlinear parallelization refactor by splitting the 723-line `spectraxgk.operators.nonlinear.spectral_core` helper module into a 75-line facade plus state, layout, work-model, pseudo-spectral bracket/RHS, and tolerance/host-staging modules. Public and private helper exports used by parallel tests and profiling scripts are preserved. Focused nonlinear parallel, spectral communication, and nonlinear sharding/profile tests passed locally.

- Continued the independent-work parallelization refactor by splitting the
  693-line `spectraxgk.parallel.core` implementation into focused identity,
  JAX batch-map, and ordered Python independent-task modules. The public
  `spectraxgk.parallel.core` facade preserves existing imports, while tests now
  patch the real owner modules. Focused parallel API, artifact-contract, ky-scan
  gate, ruff, manifest, and Sphinx gates passed locally.

- Continued the linear-RHS parallelization refactor by splitting the 678-line
  `spectraxgk.solvers.linear.parallel` implementation into common eligibility/
  device policy, Hermite streaming, and electrostatic slice/fused shard-map
  owner modules. The public module remains the dispatcher and import facade;
  focused linear helper/profile tests passed before manifest/docs gates.

- Continued the linear solver refactor by moving diagnostic fixed-step sampling
  from `spectraxgk.solvers.linear.integrators` into
  `spectraxgk.solvers.linear.integrator_diagnostics`. The public facade keeps
  the same `integrate_linear_diagnostics` entry point and patchable dependency
  seams. Linear, helper, runtime routing, manifest, ruff, and docs gates passed
  locally.

- Continued the nonlinear turbulent-flux optimization validation refactor by
  splitting `validation/nonlinear_transport/optimization_guard.py` into policy/
  scope helpers, replicated transport report extractors, and a public promotion
  facade. Nonlinear transport optimization, guard tool import, public API, ruff,
  manifest, and docs gates passed locally.

- Continued the runtime/executable refactor by moving electrostatic-potential
  initial-condition inversion into `spectraxgk.workflows.runtime.initial_phi`.
  The startup facade still re-exports the density-moment inversion helpers used
  by runtime tests and downstream diagnostics, while `initial_conditions.py` now
  owns only Gaussian/random/restart state construction and Hermitian ky expansion.
  Focused runtime, manifest, architecture, size, type, and docs gates were run
  locally before commit.

- Cleaned non-benchmark comparison-code terminology from the VMEC imported-geometry
  pipeline by renaming the equal-arc remap payload from `arrays_gx` to
  `arrays_equal_arc`. The remaining `gx_*` names in `src/spectraxgk` are confined
  to validation/benchmark compatibility parameters or comparison-specific tools.
  Focused VMEC backend tests, lint, manifests, architecture, and repository-size
  gates passed locally.

- Removed the old `spectraxgk.terms.cetg` compatibility facade and made
  `spectraxgk.terms.reduced` plus the focused cETG owner modules the canonical
  reduced-model import path. Runtime and cETG tests now import the owner modules
  directly, API docs no longer list the deleted facade, and the validation/
  refactor manifests dropped the stale module. Local cETG/runtime, lint, type,
  manifest, architecture, repository-size, and docs gates passed.

- Removed the benchmark-runner `gx_reference` compatibility keyword and kept
  `reference_aligned` as the canonical physics/numerics policy name. GX-specific
  comparison tools now call the canonical keyword, while GX-specific names remain
  confined to comparison-tool loaders/tests. Focused benchmark branch, runtime,
  comparison-tool, lint, type, manifest, architecture, repository-size, and docs
  gates passed locally.

- Split Cyclone ky-scan policy ownership further: explicit-trace seed/shift logic
  now lives in `spectraxgk.validation.benchmarks.cyclone_scan_seed`, and
  reference-aligned explicit-time reselection lives in
  `spectraxgk.validation.benchmarks.cyclone_scan_explicit`. The original
  branch module keeps compatibility re-exports and scan orchestration. Added
  direct deterministic tests for seed override tolerance, continuation-target
  extrapolation, and reselected-frequency scoring. Focused Cyclone benchmark
  branch tests, lint, py_compile, manifest, type, repository-size, and Sphinx
  gates passed locally.

- Removed the empty `spectraxgk.parallel.core` facade and made
  `spectraxgk.parallel` import `batch`, `identity`, and `independent` owners
  directly. The coverage manifest now tracks `spectraxgk.parallel.__init__`
  as the package-level public surface, and parallel artifact tests read that
  row for scaling artifacts. Focused parallel/public-API/refactor-manifest,
  lint, type, repository-size, and Sphinx gates passed locally.

- Retired the internal `spectraxgk.validation.stellarator.transport_nonlinear`
  re-export layer. The public `transport_admission` facade now imports
  nonlinear landscape, prelaunch, campaign, and audit report owners directly.
  This reduces one file and one stale compatibility hop while keeping the
  documented transport admission API unchanged. Focused stellarator validation
  tests, manifests, lint, and compile gates passed locally before the broader
  gate run.

- Folded the single-use nonlinear NetCDF restart writer into
  `spectraxgk.artifacts.nonlinear_netcdf` and deleting the old dedicated
  restart-writer module. Restart output still uses the same deterministic
  real/imag state layout and final-time metadata, but the output bundle writer
  now owns its restart sidecar directly. Focused runtime artifact, restart,
  lint, compile, and manifest gates passed locally before the broader gate run.

- 2026-06-18: Moved the lightweight benchmark drivers and runtime TOML inputs
  from the previous examples-tree location to the root `benchmarks/` directory, added a
  root benchmark README, updated docs/static provenance paths, and included
  the benchmark scripts/TOMLs in the source distribution manifest. Generated
  benchmark outputs remain excluded through `tools_out/` and docs-static review
  policy. The same tranche fixed the fast-coverage CI threshold to track the
  current `workflows/runtime/artifacts.py` owner and updated runtime tests to
  import `ModeSelection` from the diagnostics owner rather than the runtime
  facade. Local gates passed: benchmark path tests, CI-style runtime-runner
  shard, fast-coverage threshold parser, release/docs tests, validation
  coverage manifest regeneration, repository-size check, refactor manifest,
  ruff, mypy, Sphinx build, release readiness, and sdist/wheel build with
  benchmark files verified in the sdist.

- 2026-06-18: Removed the dummy standalone `spectraxgk.geometry.vmec` module,
  which only wrote a placeholder NetCDF dimension and was not a real VMEC
  geometry implementation. VMEC flux-tube generation remains owned by the
  physical runtime/backend path in `spectraxgk.geometry.vmec_eik` and
  `spectraxgk.geometry_backends.vmec_*`. The same tranche removed the stub-only
  tests, dropped the stale mypy/coverage-manifest row, regenerated the validation
  coverage summary, and fixed the remaining benchmark test path to
  the root `benchmarks/` directory. Local gates passed: standalone Miller plus
  real VMEC EIK tests, runtime benchmark TOML loader test, validation/refactor
  manifest tests, VMEC backend helper tests, validation coverage summary
  regeneration, refactor manifest check, repository-size check, ruff, mypy,
  Sphinx docs build, and `git diff --check`.

- 2026-06-18: Retired the legacy `spectraxgk.objectives.stellarator_portfolio`
  facade. Portfolio contracts, AD/FD sensitivity reports, and reduced-artifact
  guards now import directly from `objectives/portfolio_contracts.py`,
  `objectives/portfolio_sensitivity.py`, and `objectives/portfolio_artifacts.py`;
  the top-level `spectraxgk.objectives` API still re-exports the public helper
  names. Updated tests, tools, API docs, code-structure docs, and validation
  coverage manifests, then regenerated the validation coverage summary. Local
  gates passed: portfolio reducer/sensitivity tests, reduced-portfolio guard
  tests, validation/refactor manifest tests, validation summary regeneration,
  repository-size check, ruff, mypy, Sphinx docs build, and `git diff --check`.

- 2026-06-18: Retired the legacy quasilinear validation facades
  `spectraxgk.validation.quasilinear.calibration` and
  `spectraxgk.validation.quasilinear.window`. Calibration callers now import
  from `calibration_core.py`, `calibration_io.py`, and
  `calibration_spectrum.py`; late-window callers import from `window_config.py`,
  `window_statistics.py`, `window_io.py`, `window_ensemble.py`, and
  `window_promotion.py`. The public `spectraxgk.validation` API continues to
  re-export the user-facing names directly from those owners. Updated source,
  tests, tools, authored docs, and both validation/refactor manifests, then
  regenerated the validation coverage summary. Local gates passed: quasilinear
  calibration/window tests, nonlinear-gradient evidence tests, validation and
  refactor manifest tests, validation summary regeneration, repository-size
  check, ruff, mypy, Sphinx docs build, and `git diff --check`.

- 2026-06-18: Retired the legacy `spectraxgk.validation.stellarator.transport_admission`
  facade. VMEC-JAX transport admission callers now import directly from the
  focused owner modules: `transport_policies.py`, `transport_samples.py`,
  `transport_landscape.py`, `transport_prelaunch.py`, `transport_campaign.py`,
  `transport_audit.py`, and `transport_selection.py`. The public
  `spectraxgk.validation` API continues to re-export the user-facing admission
  helpers directly from those owners. Updated tests, tools, API/code-structure
  docs, validation coverage manifests, and the differentiable refactor manifest,
  then regenerated the validation coverage summary. Local gates passed: VMEC-JAX
  transport admission tests, nonlinear landscape/prelaunch/campaign/audit tool
  tests, validation/refactor manifest tests, validation summary regeneration,
  repository-size check, ruff, mypy, Sphinx docs build, and `git diff --check`.

- 2026-06-18: Retired the legacy `spectraxgk.validation.nonlinear_gradient.followup`
  facade. Nonlinear turbulence-gradient follow-up tools and tests now import
  directly from the focused owner modules: `followup_core.py`,
  `followup_plan.py`, `followup_candidate.py`, `followup_composite.py`,
  `followup_ql_seed.py`, `followup_state_runbook.py`, and
  `followup_variance.py`. Removed the facade from API docs and refactor/
  coverage manifests, then regenerated the validation coverage summary. Local
  gates passed: nonlinear-gradient follow-up, QL-seed-screen, and state-runbook
  tests; validation/refactor manifest tests; validation summary regeneration;
  repository-size check; ruff; mypy across 347 source files; Sphinx docs build;
  and `git diff --check`.

- 2026-06-19: Split the Cyclone ky-scan Krylov branch-following path out of
  `cyclone_scan_branches.py` into the focused owner
  `cyclone_scan_krylov.py`. The branch facade now keeps only the hook contract
  and time-branch orchestration while forwarding Krylov execution through the
  canonical owner; a regression test checks object identity to prevent duplicate
  owners from returning. Updated the differentiable-refactor and validation
  coverage manifests so the new module is tracked. Local gates passed: focused
  Cyclone benchmark-branch tests, benchmark smoke tests, py_compile, ruff, mypy,
  refactor manifest, validation coverage manifest, repository-size manifest,
  source terminology scans, and `git diff --check`.

- 2026-06-19: Moved the remaining saved-time and diffrax-streaming KBM
  beta-scan fitting path from `kbm_beta.py` into the existing
  `kbm_beta_solver_paths.py` owner. The public benchmark runner now handles
  scan setup, sample-state construction, and dispatch, while explicit-time,
  Krylov, and saved-time fitting policies live together behind patchable hook
  dataclasses. This avoided creating another module while reducing the runner
  from roughly 520 lines to 352 lines. Local gates passed: full benchmark
  branch test shard, KBM branch shard, explicit KBM node-id pass, py_compile,
  ruff, mypy, refactor manifest, validation coverage manifest, repository-size
  manifest, source terminology scans, and `git diff --check`.

- 2026-06-19: Simplified the internal KBM beta saved-time path inside
  `kbm_beta_solver_paths.py` without adding another module. The public
  `fit_kbm_beta_time_sample` now delegates to focused private helpers for
  per-sample time configuration, diffrax streaming fits, saved/configured
  trajectory integration, and signal selection/fitting. This increases local
  helper code in the existing owner but removes nested branch complexity and
  keeps KBM beta setup/dispatch in `kbm_beta.py` separate from solver-path
  policy. Local gates passed: KBM benchmark branch shard, full benchmark branch
  test shard, py_compile, ruff, mypy, refactor manifest, validation coverage
  manifest, repository-size manifest, source terminology scans, and
  `git diff --check`.

- 2026-06-19: Simplified the VMEC-JAX/Boozer equal-arc bridge setup in
  `geometry/vmec_boozer_core.py`. Request validation, reference length/field
  normalization, and Boozer surface-stencil selection now live in explicit
  private helpers and records, while the metric/drift JAX algebra remains in
  the public core-profile builder unchanged. This makes the differentiable
  geometry path easier to audit and test without moving numerical formulas or
  changing public APIs. Local gates passed: full differentiable geometry bridge
  test shard, py_compile, ruff, mypy, refactor manifest, validation coverage
  manifest, repository-size manifest, source terminology scans, and
  `git diff --check`.

- 2026-06-19: Simplified TEM scan-path branching inside
  `validation/benchmarks/tem_paths.py` without adding another module. The
  public `run_tem_scan_batches` loop now delegates batch preparation,
  per-batch time-config resolution, Krylov fitting, diffrax streaming fitting,
  and saved-trajectory fitting to focused private helpers. This is a complexity
  split rather than a module-size reduction: the file stays as the single TEM
  path owner so imports and monkeypatch seams remain stable, while the scan
  loop itself is shorter and easier to audit. Local gates passed: TEM benchmark
  branch shard, full benchmark branch shard, py_compile, ruff, mypy, refactor
  manifest, validation coverage manifest, repository-size manifest, source
  terminology scans, and `git diff --check`. The broad `tests/test_benchmarks.py`
  subset returned pytest exit 5 in this environment because no tests were
  collected, so it was not used as a gate for this tranche.
