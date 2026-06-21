# SPECTRAX-GK Final Closure Plan

This is the active execution plan after the `v1.6.9` compact release
checkpoint and the first post-release source-simplification passes. Detailed
chronological history is intentionally
kept in git commits, release notes, docs artifacts, and CI logs rather than in
this root file, so the repository stays easy to read and maintain.

## One-Sentence Plan

Ship SPECTRAX-GK as a compact, domain-organized, JAX-native gyrokinetic code
with a small public API, parity with GX only in explicit benchmark/comparison
lanes, a small and easy-to-manage file structure, physics-anchored tests and
documentation, differentiable Python workflows for optimization/UQ, and measured
low-runtime/low-memory CPU/GPU execution shown near the top of the README.

## Authority Map

- `plan.md`: active execution plan, current status, finite closure checklist,
  and short work log.
- `docs/architecture_refactor_plan.rst`: authoritative architecture and naming
  plan.
- `docs/differentiable_refactor_plan.rst`: differentiability-method appendix and
  migration traceability.
- `docs/code_structure.rst`: current source-tree map, not the target design.
- `docs/release_scope.rst`: claim ledger for README, docs, release notes, and
  manuscript language.
- `tools/*manifest*.toml`: executable gates for architecture, repository size,
  validation coverage, performance, quasilinear claims, parallelization,
  differentiability, and release readiness.

If these conflict, resolve in this order: release-scope claim ledger for claims,
architecture plan for layout/naming, executable manifests for gates, and this
file for execution priority.

## Current Audit Snapshot

Last audited: 2026-06-21 on `main`.

- Latest released tag: `v1.6.9`.
- Latest source-simplification commit audited here:
  `a64d2dfe Simplify reduced cETG integration policies`.
- Latest pushed plan-audit commit: `24419653 Record velocity sharded RHS
  refactor audit`; worktree was clean and synchronized with `origin/main` at
  the start of this final plan review.
- Latest completed green CI before the newest simplification queue:
  `ea506e11 Simplify VMEC flux tube parity report packing`. Later
  simplification runs may be superseded/cancelled by newer pushes; the latest
  non-superseded `main` CI run is the required green check before release
  tagging.
- Release workflow `27906940479` passed for tag `v1.6.9`; PyPI lists the
  `spectraxgk-1.6.9` wheel and sdist, and GitHub Release `v1.6.9` is latest.
- Package shape: 357 Python files under `src/spectraxgk`, about 106.4k source
  lines, 9 root facade modules, and no blocked root-prefix modules under the
  architecture manifest.
- Test shape: 316 Python test files, about 92.6k test lines. The wide
  sharded CI path remains the authority for the package-wide coverage badge.
- Largest package-internal navigation costs are now concentrated in
  `validation`, `objectives`, `solvers`, and `operators`; the refactor target
  is fewer compatibility seams and clearer package ownership, not more files.
- Function length: 0 source functions at or above 90 lines; 40 functions in the
  80-89 line range and 118 functions at or above 70 lines after the linear
  workflow, nonlinear IMEX diagnostic dispatch, linear implicit
  preconditioner, velocity-sharded electrostatic RHS route, and nonlinear
  electromagnetic dispatch, nonlinear timestep-policy, and reduced-cETG
  integration-policy simplifications.
- Current high-value simplification targets are concentrated in validation
  benchmark path/report orchestration, VMEC/Boozer and QA objective assembly,
  runtime TOML loading, and linear dissipation policy. These should be handled
  inside existing domain packages, not by adding new root modules.
- Tests: package-wide CI coverage gate remains at or above 95% through the wide
  coverage path.
- README: runtime/memory comparison panel is visible immediately after
  Highlights, before current claim scope, and lists W7-X/HSX rows.
- Benchmarks: benchmark scripts and result manifest live at root `benchmarks/`;
  raw transient outputs stay out of git.
- Repository size: large local files are ignored scratch/build/env artifacts.
  The tracked runtime/memory figure and metadata are small; the release-size
  gate remains the source of truth.
- Current worktree status before this plan edit: clean on `main` and
  synchronized with `origin/main`.

## Results Already Closed

- Root-level prefix sprawl has been removed from the public package surface.
- The executable quickstart works with `spectraxgk`, writes reproducible default
  inputs in the current directory, and supports `spectraxgk --plot` for saved
  linear/nonlinear outputs.
- Runtime command options were consolidated without changing public behavior.
- Runtime command-artifact display helpers were folded into the runtime artifact
  orchestration owner, reducing package source files from 358 to 357 without
  changing executable behavior.
- Gradient-validation report assembly now has explicit Jacobian, tangent, and
  conditioning-gate helpers, keeping AD/FD report schema stable while reducing
  long differentiability plumbing functions.
- VMEC/Boozer objective-table assembly now has explicit surface-geometry,
  row-metadata, and physical-``k_y`` metadata helpers, preserving the
  differentiable table contract while reducing the main table builder.
- VMEC flux-tube parity report packing now goes through an explicit
  result/options helper, preserving the public parity report schema while
  shortening the geometry validation wrapper.
- The README runtime/memory panel was restored near the top and tied to measured
  artifact provenance.
- The root `benchmarks/` directory now contains lightweight drivers, TOMLs, and
  a small result manifest.
- Release engineering for `v1.6.9` is closed: PyPI/GitHub release passed, CI
  coverage is wide-sharded, and package/docs builds pass.
- The refreshed nonlinear sharding xlarge GPU artifact is identity-consistent
  and documents the current whole-state route as diagnostic-only; production
  parallelization remains the independent-work path.
- The full-GK linear runtime workflow now delegates fit-policy construction and
  solver-branch dispatch to focused same-file helpers, preserving the public
  runtime API while shortening the orchestration function named in this plan.
- The nonlinear IMEX diagnostic integrator now uses one option-bundle helper
  instead of four single-use option constructors, preserving the scan/core
  contract while removing navigation-only helpers and shortening the public
  integration wrapper named in this plan.
- The linear implicit GMRES preconditioner selector now routes through explicit
  named policies for diagonal, damping, parallel-streaming, coarse, and
  Hermite-line preconditioners. This preserves the public implicit operator
  contract while reducing the selector itself from the 80-89 line band to a
  small dispatcher.
- The velocity-sharded electrostatic linear RHS now has named fused-kernel and
  serial term-contribution helpers. This preserves the existing public
  dispatch and monkeypatch contract while shortening the fused and serial
  route bodies out of the 80-89 line band.
- The nonlinear electromagnetic contribution/component APIs now share a
  same-file bracket context and Laguerre-vs-spectral dispatch helpers. Public
  signatures and diagnostic payload schemas are unchanged, while both public
  route bodies are now out of the 80-89 line band.
- The nonlinear timestep-policy builder now assembles explicit limits, CFL
  bounds, and the adaptive update closure through named same-file helpers.
  Fixed/adaptive timestep semantics and public policy fields are unchanged.
- The reduced cETG diagnostic integrator now routes explicit method stages and
  diagnostic observable assembly through named same-file helpers. The public
  cETG integration entry point and diagnostic tuple/schema are unchanged.

## Open Lanes And Priority

Percentages are engineering status estimates, not scientific claims.

| Priority | Lane | Status | Closure Evidence |
| --- | --- | ---: | --- |
| P0 | Plan/docs/readme consistency | 99% | This file is the active plan; update docs/readme only when evidence-backed artifacts change. |
| P0 | Release hygiene | 98% | Last pushed commit is green; next release waits for a clean worktree, bounded local gates, and CI. |
| P1 | Source simplification and naming | 99% | No new root-prefix modules, non-benchmark comparison-code terminology removed or justified, fewer navigation-only helpers, tests updated. |
| P1 | Refactor/testability | 99% | High-value 70-89 line functions reduced only where it exposes a real policy boundary or removes duplication. |
| P1 | Package coverage and physics tests | 100% gate, 96% margin | Wide package coverage stays >=95%; new tests remain physics/numerics/autodiff/regression tests rather than smoke-only coverage. |
| P2 | Differentiable Python workflows | 99% scoped | AD/FD, tangent, conditioning, or covariance gates exist for every promoted differentiated observable. |
| P2 | VMEC/Boozer differentiable geometry | 99% scoped | Geometry parity and gradient gates pass for promoted rows; broad optimization claims stay scoped. |
| P2 | Performance and memory | 97% scoped | Runtime/memory panel remains measured; no new speedup claim without profiler artifacts and numerical identity gates. |
| P2 | Production parallelization | 95% scoped | Independent-work parallelization is production; refreshed whole-state nonlinear sharding is identity-correct but slower than serial, so nonlinear domain decomposition remains diagnostic. |
| P3 | Quasilinear model development | 99% scoped | Diagnostics and screening claims documented; universal absolute-flux prediction remains unpromoted. |
| P3 | Nonlinear turbulent-flux optimization | 91% scoped | Long post-transient matched audits exist for scoped cases; broad optimized stellarator turbulence claim remains unpromoted. |
| P4 | Deferred W7-X/TEM science | deferred | W7-X zonal long-window recurrence, W7-X TEM/multi-flux-tube, and W7-X fluctuation extensions remain post-release unless explicitly reopened. |

## Priority Order From Here

The remaining path is finite and release-oriented. The goal is not more
incremental file splitting; it is to make the existing domain packages easier
to read, test, and extend while preserving validated behavior.

1. **Confirm the current checkpoint.** Let the newest non-superseded CI run
   finish, fix only real failures, and keep cancelled runs caused by newer
   pushes out of the blocker list.
2. **Keep the README performance figure in place.** The runtime/memory figure
   is restored near the top of the README, after Highlights and before claim
   scope; refresh it only from new measured CPU/GPU artifacts with hardware,
   wall-time, memory, and W7-X/HSX rows.
3. **Finish one compact source-simplification batch.** Work in existing domain
   packages only: solvers/operators first, then validation/benchmarks,
   objectives/geometry, and workflows/artifacts. Acceptance criteria are no new
   root modules, no functions at or above 90 lines, a lower 80-89 line count
   only when the extraction exposes a real physics/numerics policy, and no
   public behavior change.
4. **Clean terminology and public surfaces.** Keep GX wording only in explicit
   benchmark/comparison artifacts; rename source/test names that describe
   current SPECTRAX-GK physics or numerics rather than comparison contracts.
5. **Mirror tests and docs after each tranche.** Tests should move toward
   package-aligned behavior names, and docs should describe the stable workflow
   rather than migration history.
6. **Release only from measured, gated evidence.** New parity, nonlinear
   optimization, quasilinear, differentiability, or speedup claims need
   reproducible artifacts, physics gates, and docs updates before README
   promotion; otherwise they remain scoped or deferred.

## Final Prioritized Steps

### 1. Freeze the current checkpoint

Goal: preserve the green `main` state before any new release or science push.

- Keep the README runtime/memory panel where it is: after Highlights and before
  current claim scope.
- Do not regenerate runtime/memory figures unless fresh CPU/GPU measured
  artifacts are produced.
- Keep benchmark/comparison references to GX only in validation, benchmark,
  performance, and comparison artifacts.
- Run bounded local gates for any plan/docs/readme-only edits:
  - `python tools/check_package_architecture_manifest.py`
  - `python tools/check_repository_size_manifest.py`
  - `python tools/check_release_readiness.py --out-json /tmp/spectrax_release_readiness.json`
  - `python -m pytest -q tests/test_check_release_readiness.py tests/test_check_repository_size_manifest.py tests/test_check_release_version.py --maxfail=1`
  - `python -m ruff check src tests tools benchmarks`
  - `python -m sphinx -b html docs /tmp/spectrax_docs_plan_build`
  - `git diff --check`
- Confirm the latest non-superseded CI run is green before tagging.

### 2. Finish source simplification without file sprawl

Goal: make the code easier to navigate without increasing the number of files or
introducing another wave of thin modules.

- Do not add new root-level modules.
- Do not split code just to reduce line counts. Split only when the result is a
  named physics/numerics policy, a differentiability boundary, or a testable
  artifact contract.
- Consolidate single-use helpers back into their nearest domain owner when that
  reduces navigation cost and keeps functions under the architecture gate.
- The explicitly named high-value functions from this checkpoint are now below
  the 80-89 line band. Continue only with package-internal consolidation that
  removes duplication or clarifies a real physics/numerics policy boundary.
- Next consolidation targets are package-internal:
  - `validation/benchmarks`: keep benchmark case families but reduce duplicated
    path/branch/report boilerplate.
  - `objectives`: group VMEC/Boozer, QA-transport, and generic solver-gradient
    policies behind smaller public exports.
  - `solvers` and `operators`: keep hot kernels close to JIT boundaries and
    remove wrappers that only rename arguments.
  - `tests`: keep package-wide coverage but move large legacy-style test names
    toward physics/numerics behavior names.
- Rename non-benchmark comparison-code terminology in source/tests only where it
  is not explicitly a benchmark/comparison contract.
- Remove legacy or compatibility-only examples that are not part of the current
  documented workflow.

Domain-batched execution order:

1. **Solvers/operators.** Finish the high-impact policy functions still in the
   80-89 line band, especially linear implicit preconditioner selection,
   velocity-sharded electrostatic RHS routing, nonlinear electromagnetic
   contribution assembly, IMEX core assembly, and reduced-model explicit
   stepping. Keep hot JIT kernels close to their caches and avoid wrapper-only
   moves.
2. **Validation/benchmarks.** Consolidate repeated fit-window, path, branch,
   and report-packing policies in TEM, ETG, KBM, kinetic-electron, and Cyclone
   benchmark families. Keep comparison-code terminology only in benchmark and
   comparison contracts.
3. **Objectives/geometry.** Consolidate QA low-turbulence, VMEC/Boozer line
   search, sensitivity, inverse-design, and gradient-report assembly so public
   optimization APIs expose physics objectives, not migration-era helper
   names.
4. **Workflows/artifacts.** Tighten runtime TOML loading, runtime scan batch
   orchestration, plotting dispatch, restart/artifact writing, and default-demo
   paths. Preserve executable behavior, progress output, and `--plot`.
5. **Tests/docs.** After source movement is stable, migrate large top-level
   tests into package-aligned behavior groups and update docs to match the
   current public surfaces.

### 3. Keep tests physics-anchored and coverage stable

Goal: keep the 95% package-wide gate green while improving scientific value.

- Add tests only when they protect an equation, numerical method, diagnostic
  convention, artifact schema, autodiff contract, restart behavior, or known
  regression.
- Keep local tests bounded; use CI wide shards for package-wide coverage.
- Keep optional office/GPU and external-code runs out of the default local test
  suite, behind manifests and explicit commands.
- Maintain validation-coverage manifest ownership for any moved module.

### 4. Close differentiable-code guarantees for promoted workflows

Goal: Python research APIs are differentiable and testable, while executable
paths remain user-friendly and fast.

- Keep pure solver/objective functions free of file I/O, plotting, terminal
  progress, subprocess calls, global mutable state, and host callbacks.
- Use native JAX AD for smooth fixed-step/reduced workflows, implicit eigenpair
  differentiation for isolated linear branches, and implicit/adjoint methods
  only where finite-difference/tangent gates pass.
- Keep executable adaptive/progress paths separate from differentiable Python
  objectives.
- VMEC/Boozer optimization claims require geometry parity, gradient gates, and
  claim-scope entries before README promotion.

### 5. Preserve physics validation and scoped parity

Goal: claims stay reviewer-proof and reproducible.

- Keep validated linear/nonlinear atlas cases and release gates intact.
- Re-run GX on `ssh office` only for touched parity lanes, refreshed public
  comparison figures, or suspected numerical regressions.
- Keep unpromoted lanes visible but scoped:
  - universal absolute quasilinear flux prediction;
  - broad nonlinear turbulent-flux stellarator optimization;
  - production nonlinear domain-decomposition speedup;
  - W7-X zonal long-window recurrence/damping;
  - W7-X TEM/multi-flux-tube and fluctuation-spectrum extensions.

### 6. Keep performance claims measured

Goal: low runtime and memory footprint are documented only where measured.

- Runtime/memory panel stays in README and docs, backed by:
  - `docs/_static/runtime_memory_benchmark.png`
  - `docs/_static/runtime_memory_results_ship_refresh.csv`
  - `docs/_static/runtime_memory_summary_ship_refresh.json`
- Refresh the panel only after fresh measured CPU/GPU artifacts include wall
  time, peak memory, hardware/backend metadata, and W7-X/HSX rows.
- Make no nonlinear speedup claim without serial-vs-decomposed identity gates,
  profiler artifacts, and end-to-end CPU/GPU timing.

### 7. Release sequence

Goal: ship the next version only from clean, green `main`.

1. Confirm latest CI is green.
2. Run bounded local release gates.
3. Bump version in `pyproject.toml` and `src/spectraxgk/_version.py`.
4. Run release-version tests and package/docs build.
5. Commit and push the version bump.
6. Confirm CI for the bump commit.
7. Tag `vX.Y.Z` and push the tag.
8. Verify the GitHub release workflow and PyPI publish path.

## Release Blocking Checklist

- [ ] `main` clean and up to date with `origin/main`.
- [ ] Latest non-superseded CI success confirmed.
- [x] `plan.md`, `README.md`, `docs/architecture_refactor_plan.rst`,
      `docs/code_structure.rst`, and `docs/release_scope.rst` agree on scope.
- [x] README runtime/memory panel remains near the top and tied to measured
      artifacts.
- [x] No new large tracked artifacts or raw simulation outputs.
- [ ] Architecture, repository-size, release-readiness, docs, package, and
      bounded tests pass locally for the final release candidate.
- [ ] Version bump and tag are pushed only after green gates.

## Short Work Log

- 2026-06-21: Audited `main` after `bbd515f0`; latest CI completed
  successfully. README runtime/memory panel is already restored near the top,
  benchmark root directory exists with a small result manifest, and the package
  has 358 Python files, about 106k source lines, 9 root facades, and no source
  function at or above 90 lines.
- 2026-06-21: Replaced the oversized root plan log with this finite closure
  plan. Detailed history remains available through git commits and release/docs
  artifacts.

- 2026-06-21: Consolidated `workflows/runtime/command_artifacts.py` into
  `workflows/runtime/orchestration_artifacts.py`, updated docs/manifests/tests,
  and reduced package source count to 357 files while preserving runtime command
  saved-output and stdout behavior.
- 2026-06-21: Green CI confirmed for `3e8dd615`; prepared the `v1.6.9`
  release checkpoint without regenerating runtime/memory artifacts.
- 2026-06-21: Released `v1.6.9` from `494b9ea2`. CI run `27906389241` and
  release workflow `27906940479` passed; PyPI published the wheel/sdist and the
  GitHub Release entry is latest.
- 2026-06-21: Split differentiable gradient-validation report assembly into
  focused Jacobian, tangent, and conditioning-gate helpers inside
  `geometry/autodiff_checks.py`. Targeted differentiability tests passed; source
  functions in the 80-89 line band dropped from 54 to 52 with no new files.
- 2026-06-21: Split VMEC/Boozer objective-table assembly into local
  surface-geometry, row-metadata, and physical-`k_y` metadata helpers. Focused
  VMEC/Boozer table-contract tests passed; source functions in the 80-89 line
  band dropped from 52 to 51 with no new files.
- 2026-06-21: Added a result/options packer for VMEC flux-tube parity reports.
  Focused geometry bridge and claim-check tests passed; source functions in the
  80-89 line band dropped from 51 to 50 with no new files.
- 2026-06-21: Refreshed the office two-RTX-A4000 nonlinear whole-state
  sharding xlarge artifact with the identity-preserving `auto` route. The gate
  passes final-state identity but fails production speedup fail-closed
  (`0.586x`), so docs record it as diagnostic negative evidence and README
  speedup wording remains unchanged.
- 2026-06-21: Split full-GK linear runtime orchestration in
  `workflows/linear.py` into same-file fit-policy and solver-branch helpers.
  The public runtime signature and behavior stay unchanged; selected
  runtime-linear integration tests, ruff, mypy, and compileall passed, and the
  80-89 line function count dropped from 50 to 49.
- 2026-06-21: Consolidated nonlinear IMEX diagnostic option packing in
  `solvers/nonlinear/imex_diagnostics.py` into one same-file option bundle.
  Focused IMEX helper/public wrapper/facade tests, ruff, mypy, and compileall
  passed; the 80-89 line function count dropped from 49 to 48.
- 2026-06-21: Re-audited the active plan after the latest source
  simplification commits. The final closure path is now domain-batched:
  solvers/operators, validation/benchmarks, objectives/geometry,
  workflows/artifacts, then package-aligned tests/docs. The README
  runtime/memory panel remains a required top-of-README artifact and must be
  refreshed only from measured CPU/GPU evidence.
- 2026-06-21: Simplified the linear implicit preconditioner policy in
  `solvers/linear/implicit.py` into named canonical-alias and preconditioner
  application helpers. Focused implicit/preconditioner tests, nonlinear IMEX
  forwarding tests, ruff, mypy, compileall, architecture, repository-size, and
  release-readiness checks passed; the 80-89 line function count dropped from
  48 to 47 and the >=70 count dropped from 126 to 125.
- 2026-06-21: Simplified the velocity-sharded electrostatic linear RHS in
  `solvers/linear/parallel_electrostatic.py` by extracting the fused
  electrostatic RHS kernel and serial streaming/mirror/curvature/diamagnetic
  contribution helpers. Focused parallel-dispatch and velocity-sharding tests,
  ruff, mypy, compileall, architecture, repository-size, and
  release-readiness checks passed; the 80-89 line count dropped from 47 to 45
  and the >=70 count dropped from 125 to 123.
- 2026-06-21: Simplified nonlinear electromagnetic dispatch in
  `terms/nonlinear.py` by adding a bracket-context helper and shared
  contribution/component dispatch helpers. Public nonlinear EM signatures,
  Laguerre/spectral routing, and diagnostic payload schemas stay unchanged.
  Focused nonlinear E×B/electromagnetic and cached RHS tests passed; ruff,
  mypy, compileall, architecture, repository-size, and release-readiness checks
  passed; the 80-89 line count dropped from 45 to 43 and the >=70 count
  dropped from 123 to 121.
- 2026-06-21: Simplified nonlinear timestep-policy assembly in
  `operators/nonlinear/policies.py` by extracting explicit timestep-limit,
  CFL-bound, and update-closure helpers inside the existing nonlinear-operator
  package. Focused fixed/adaptive timestep and nonlinear CFL-frequency tests
  passed; ruff, mypy, compileall, architecture, repository-size, and
  release-readiness checks passed; the 80-89 line count dropped from 43 to 42
  and the >=70 count dropped from 121 to 120.
- 2026-06-21: Simplified reduced cETG integration policies in
  `terms/reduced/cetg_integrator.py` by extracting explicit method-stage
  helpers, mode-diagnostic selection, and energy/flux diagnostic assembly.
  The full cETG test file passed, along with ruff, mypy, compileall,
  architecture, repository-size, and release-readiness checks; the 80-89 line
  count dropped from 42 to 40 and the >=70 count dropped from 120 to 118.
