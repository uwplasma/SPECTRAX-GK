# SPECTRAX-GK Final Closure Plan

This is the active execution plan after the `v1.6.9` compact release
checkpoint following the differentiable architecture/refactor work. Detailed chronological history is intentionally
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
- Latest pushed release commit audited here:
  `494b9ea2 Bump version to 1.6.9`.
- Latest completed green CI for the release commit: `494b9ea2`. Quick shards,
  mypy, docs/package build, fast coverage, all 48 wide-coverage shards, and
  wide coverage combine passed.
- Release workflow `27906940479` passed for tag `v1.6.9`; PyPI lists the
  `spectraxgk-1.6.9` wheel and sdist, and GitHub Release `v1.6.9` is latest.
- Package shape: 357 Python files under `src/spectraxgk`, about 106.6k source
  lines, 9 root facade modules, and no blocked root-prefix modules under the
  architecture manifest.
- Function length: 0 source functions at or above 90 lines; 54 functions in the
  80-89 line range and 129 functions at or above 70 lines.
- Tests: package-wide CI coverage gate remains at or above 95% through the wide
  coverage path.
- README: runtime/memory comparison panel is visible immediately after
  Highlights, before current claim scope, and lists W7-X/HSX rows.
- Benchmarks: benchmark scripts and result manifest live at root `benchmarks/`;
  raw transient outputs stay out of git.
- Repository size: large local files are ignored scratch/build/env artifacts.
  The tracked runtime/memory figure and metadata are small; the release-size
  gate remains the source of truth.

## Results Already Closed

- Root-level prefix sprawl has been removed from the public package surface.
- The executable quickstart works with `spectraxgk`, writes reproducible default
  inputs in the current directory, and supports `spectraxgk --plot` for saved
  linear/nonlinear outputs.
- Runtime command options were consolidated without changing public behavior.
- Runtime command-artifact display helpers were folded into the runtime artifact
  orchestration owner, reducing package source files from 358 to 357 without
  changing executable behavior.
- The README runtime/memory panel was restored near the top and tied to measured
  artifact provenance.
- The root `benchmarks/` directory now contains lightweight drivers, TOMLs, and
  a small result manifest.
- Release engineering for `v1.6.9` is closed: PyPI/GitHub release passed, CI
  coverage is wide-sharded, and package/docs builds pass.

## Open Lanes And Priority

Percentages are engineering status estimates, not scientific claims.

| Priority | Lane | Status | Closure Evidence |
| --- | --- | ---: | --- |
| P0 | Plan/docs/readme consistency | 100% | This file, architecture docs, README, and release-scope docs agree on current status and claim scope. |
| P0 | Release hygiene | 100% | Clean `main`, green CI, bounded local release gates, version bump, tag, release workflow, PyPI publish. |
| P1 | Source simplification and naming | 96% | No new root-prefix modules, non-benchmark comparison-code terminology removed or justified, fewer navigation-only helpers, tests updated. |
| P1 | Refactor/testability | 97% | High-value 70-89 line functions reduced only where it exposes a real policy boundary or removes duplication. |
| P1 | Package coverage and physics tests | 100% gate, 96% margin | Wide package coverage stays >=95%; new tests remain physics/numerics/autodiff/regression tests rather than smoke-only coverage. |
| P2 | Differentiable Python workflows | 98% scoped | AD/FD, tangent, conditioning, or covariance gates exist for every promoted differentiated observable. |
| P2 | VMEC/Boozer differentiable geometry | 98% scoped | Geometry parity and gradient gates pass for promoted rows; broad optimization claims stay scoped. |
| P2 | Performance and memory | 97% scoped | Runtime/memory panel remains measured; no new speedup claim without profiler artifacts and numerical identity gates. |
| P2 | Production parallelization | 94% scoped | Independent-work parallelization is production; nonlinear domain decomposition remains diagnostic until identity and speedup gates pass. |
| P3 | Quasilinear model development | 99% scoped | Diagnostics and screening claims documented; universal absolute-flux prediction remains unpromoted. |
| P3 | Nonlinear turbulent-flux optimization | 91% scoped | Long post-transient matched audits exist for scoped cases; broad optimized stellarator turbulence claim remains unpromoted. |
| P4 | Deferred W7-X/TEM science | deferred | W7-X zonal long-window recurrence, W7-X TEM/multi-flux-tube, and W7-X fluctuation extensions remain post-release unless explicitly reopened. |

## Priority Order From Here

The remaining path is intentionally short and release-oriented:

1. **Keep the README performance figure in place.** The runtime/memory figure is
   already restored near the top of the README; refresh it only from new
   measured CPU/GPU artifacts with hardware, wall-time, memory, and W7-X/HSX
   rows.
2. **Resume refactoring only after the release checkpoint.** Further source
   simplification should reduce navigation cost inside the largest domain
   packages without adding new root files or compatibility shims.
3. **Resume science/performance lanes only with gates.** New parity, nonlinear
   optimization, quasilinear, differentiability, or speedup claims need
   reproducible artifacts, physics gates, and docs updates before README
   promotion.

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

### 2. Finish source simplification without file sprawl

Goal: make the code easier to navigate without increasing the number of files or
introducing another wave of thin modules.

- Do not add new root-level modules.
- Do not split code just to reduce line counts. Split only when the result is a
  named physics/numerics policy, a differentiability boundary, or a testable
  artifact contract.
- Consolidate single-use helpers back into their nearest domain owner when that
  reduces navigation cost and keeps functions under the architecture gate.
- Prioritize the current high-value 80-89 line functions only if touched by a
  feature or bugfix:
  - `workflows/linear.py::run_full_linear_runtime`
  - `solvers/nonlinear/imex_diagnostics.py::integrate_imex_nonlinear_diagnostics_impl`
  - `objectives/vmec_boozer.py::vmec_boozer_solver_objective_table_with_metadata_from_state`
  - `geometry/vmec_flux_tube_reports.py::vmec_jax_flux_tube_array_parity_report`
  - `geometry/autodiff_checks.py::_gradient_gate_data`
- Rename non-benchmark comparison-code terminology in source/tests only where it
  is not explicitly a benchmark/comparison contract.
- Remove legacy or compatibility-only examples that are not part of the current
  documented workflow.

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

- [x] `main` clean and up to date with `origin/main`.
- [x] Latest CI success confirmed.
- [x] `plan.md`, `README.md`, `docs/architecture_refactor_plan.rst`,
      `docs/code_structure.rst`, and `docs/release_scope.rst` agree on scope.
- [x] README runtime/memory panel remains near the top and tied to measured
      artifacts.
- [x] No new large tracked artifacts or raw simulation outputs.
- [x] Architecture, repository-size, release-readiness, docs, package, and
      bounded tests pass locally.
- [x] Version bump and tag are pushed only after green gates.

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
