# SPECTRAX-GK Final Closure Plan

This is the active execution plan after the `v1.6.9` release checkpoint and the
post-release source-simplification passes. Detailed chronology belongs in git
commits, docs artifacts, release notes, and CI logs; this root file stays short,
actionable, and finite.

## One-Sentence Plan

Ship SPECTRAX-GK as a compact, domain-organized, JAX-native gyrokinetic code with a small public API, parity with GX only in explicit benchmark/comparison lanes, a small and easy-to-manage source tree, physics-anchored tests, complete README/docs, differentiable Python workflows, and measured low-runtime/low-memory CPU/GPU execution with the runtime/memory comparison panel kept near the top of the README.

## Authority Map

- `plan.md`: active execution priorities, acceptance gates, and rolling status.
- `docs/architecture_refactor_plan.rst`: target package layout, naming policy,
  and what counts as a useful refactor.
- `docs/differentiable_refactor_plan.rst`: differentiability strategy,
  supported AD/FD/implicit-adjoint paths, and migration traceability.
- `docs/code_structure.rst`: current source-tree map.
- `docs/release_scope.rst`: claim ledger for README, docs, releases, and papers.
- `tools/*manifest*.toml`: executable gates for architecture, repo size,
  validation coverage, performance, quasilinear claims, parallelization,
  differentiability, and release readiness.

Conflict resolution order: release-scope claims, architecture plan, executable
manifests, then this execution file.

## Current Audit Snapshot

Last audited: 2026-06-22 on `main`.

- Latest released tag: `v1.6.9`.
- Current source-simplification head:
  `ed432628 Simplify linear implicit integration scan`.
- Worktree at audit start: local `main` contains the linear implicit
  source-simplification tranche and this plan update; push both after the
  bounded local gates pass.
- Latest CI state at audit: the newest pushed `main` run for `d7b77801` was
  in progress, and the latest completed non-superseded run for `fe77d967`
  passed. Verify the newest non-superseded run after pushing this tranche, but
  do not spend time watching runs cancelled or superseded by newer pushes.
- Package shape: 357 tracked Python files under `src/spectraxgk`, 316 tracked
  Python tests, 9 root facade modules, 21 required domain packages, and zero
  blocked root-prefix modules.
- Size and docs shape: about 107,800 source lines, 92,600 test lines, a
  1,287-line README, and large but organized documentation pages for testing,
  quasilinear modeling, stellarator optimization, performance, manuscript
  figures, and code structure. The next documentation pass should tighten and
  cross-link rather than add more broad narrative by default.
- Function-length audit: 0 source functions at or above 90 lines, 16 functions
  in the 80-89 line band, and 99 functions at or above 70 lines. Long classes
  remain mostly dataclass/config containers, not oversized algorithms.
- Source-tree audit: function size is controlled, but 357 source files is still
  broad. The remaining refactor work must prefer consolidation of single-use
  slices and clearer domain ownership over adding more files.
- Repository-size audit: architecture and size manifests pass. Tracked content
  is about 49 MB with no unlisted large tracked files. The large local checkout
  footprint is dominated by ignored `.venv`, caches, `docs/_build`, `dist`, and
  `tools_out`.
- README status: installation, executable quickstart, `spectraxgk --plot`, claim
  scope, QA optimization scope, VMEC examples, and the runtime/memory comparison
  panel are present. The runtime/memory panel is already restored after
  Highlights and before claim scope.
- Documentation status: domain docs exist for theory, numerics, operators,
  geometry, benchmarks, testing, performance, parallelization, quasilinear
  transport, differentiability, and stellarator optimization. The next docs
  work is tightening, not expanding by default.
- Benchmark layout: lightweight benchmark drivers and manifests live at root
  `benchmarks/`; raw transient outputs stay ignored/out of git.

## Closed Results

- `v1.6.9` release passed GitHub release/PyPI publication.
- Executable quickstart works with `spectraxgk`, writes reproducible default
  inputs in the current directory, emits progress/ETA for longer runs, and
  supports `spectraxgk --plot` for supported saved linear/nonlinear outputs.
- README runtime/memory comparison panel was restored near the top and tied to
  measured artifacts.
- Root benchmark directory was added with lightweight drivers and a small result
  manifest.
- Source simplification has removed root-level prefix sprawl and kept the public
  package surface domain-organized.
- Recent refactors simplified runtime artifact display, gradient-validation
  reports, VMEC/Boozer objective tables, VMEC flux-tube parity packing, linear
  runtime dispatch, nonlinear IMEX diagnostics, linear implicit preconditioner
  routing, velocity-sharded electrostatic RHS routing, nonlinear electromagnetic
  dispatch, nonlinear timestep policy, reduced cETG integration policy,
  benchmark diagnostic loading, linear hypercollision routing, reduced QA
  core-feature assembly, quasilinear transport payload assembly, QA
  low-turbulence envelope tracing, geometry inverse-design report assembly, and
  runtime TOML loading, runtime linear diagnostic fitting, and linear
  time-series integration dispatch, and QA low-turbulence optimizer state
  assembly, Cyclone scan setup policy resolution, ETG scan time-batch context
  packing, TEM path/scan policy packing, cETG linear runtime fitting, and
  late-time linear metrics signal/tail-stat assembly, and nonlinear runtime
  diagnostic/final-state result routing, linear diffrax setup bundling, and
  VMEC transport table row assembly, and VMEC/Boozer line-search step
  candidate/stop routing, and VMEC state-sensitivity report runner/payload
  routing, and linear implicit sample/scan orchestration.
- Package-wide coverage remains gated by wide CI shards at or above 95%.
- Independent-work parallelization is the production path; nonlinear domain
  decomposition is identity-tested diagnostic evidence only until speedup gates
  pass.

## Open Lanes

Percentages are engineering progress estimates, not scientific claims.

| Priority | Lane | Status | Closure Evidence |
| --- | --- | ---: | --- |
| P0 | CI/release hygiene | 99% | Latest completed non-superseded CI green; queued head run must be checked once, then fixed only if it fails. |
| P0 | README/docs/plan consistency | 99% | README runtime/memory panel visible after Highlights; docs and claim scope agree; this plan is the single execution authority. |
| P1 | Source simplification and naming | 99.75% | No new root modules, zero functions >=90 lines, 16 functions in the 80-89 band, and remaining work is file/navigation consolidation rather than more splits. |
| P1 | Refactor/testability | 99.55% | Remaining 80-89 line functions reduced only when they expose real physics/numerics policy boundaries, remove duplication, or consolidate single-use wrappers. |
| P1 | Package coverage and physics tests | 100% gate | Wide package coverage stays >=95%; new tests protect equations, numerics, diagnostics, AD contracts, artifacts, or regressions. |
| P2 | Runtime/memory and performance claims | 98% scoped | README panel remains measured; refresh only from new CPU/GPU artifacts with hardware, wall time, memory, and W7-X/HSX rows. |
| P2 | Differentiable Python workflows | 99% scoped | Promoted observables have AD/FD, tangent, conditioning, covariance, or implicit-differentiation gates. |
| P2 | VMEC/Boozer differentiable geometry | 99% scoped | Promoted geometry rows have parity and gradient gates; broad optimization claims stay scoped. |
| P2 | Production parallelization | 95% scoped | Independent ky/batch/UQ work is production; nonlinear domain decomposition remains diagnostic until identity plus speedup gates pass. |
| P3 | Quasilinear model development | 99% scoped | Diagnostics/screening documented; universal absolute-flux prediction remains unpromoted. |
| P3 | Nonlinear turbulent-flux optimization | 91% scoped | Long post-transient matched audits exist for scoped cases; broad optimized-stellarator turbulence claim remains unpromoted. |
| P4 | Deferred W7-X/TEM science | deferred | W7-X zonal long-window recurrence, W7-X TEM/multi-flux-tube, and fluctuation-spectrum extensions are post-release unless reopened. |

## Prioritized Closure Steps

### 1. Confirm the checkpoint

Goal: preserve a clean, green `main` before another release or science push.

- Check the latest non-superseded CI run; fix real failures, ignore runs
  cancelled by newer pushes.
- Run bounded local gates for plan/docs/readme-only edits:
  - `python tools/check_package_architecture_manifest.py`
  - `python tools/check_repository_size_manifest.py`
  - `python tools/check_release_readiness.py --out-json /tmp/spectrax_release_readiness.json`
  - `python -m pytest -q tests/test_check_release_readiness.py tests/test_check_repository_size_manifest.py tests/test_check_release_version.py --maxfail=1`
  - `python -m ruff check src tests tools benchmarks`
  - `python -m sphinx -b html docs /tmp/spectrax_docs_plan_build`
  - `git diff --check`

### 2. Keep the README runtime/memory panel evidence-backed

Goal: the README again shows the runtime and memory comparison between
SPECTRAX-GK CPU/GPU and the reference benchmark backend, without unsupported
speedup claims.

- Keep `docs/_static/runtime_memory_benchmark.png` in the README immediately
  after Highlights.
- Keep the provenance files:
  - `docs/_static/runtime_memory_results_ship_refresh.csv`
  - `docs/_static/runtime_memory_summary_ship_refresh.json`
  - `docs/_static/runtime_memory_benchmark.png`
- Refresh the figure only from new measured CPU/GPU artifacts that include
  hardware/backend metadata, cold/warm wall time where relevant, peak memory,
  and W7-X/HSX rows.
- Do not claim nonlinear domain-decomposition speedup until profiler-backed
  serial-vs-decomposed identity and speedup gates pass.

### 3. Finish source simplification without file sprawl

Goal: make the code simpler to navigate and extend while keeping the source tree
small enough that new developers can find the physics, numerics, diagnostics,
and workflows without following wrapper chains.

- Do not add new root modules or migration-era compatibility facades.
- Do not increase the 357-file source count unless the same tranche deletes or
  consolidates at least as many files and the net navigation cost decreases.
- Target the next release candidate with zero functions >=90 lines, fewer than
  25 functions in the 80-89 line band, and a non-increasing source-file count;
  the stretch goal is to remove 10-20 single-use internal files without moving
  stable public facades.
- Treat oversized documentation/API pages the same way as oversized source
  files: tighten indexes, claim ledgers, and cross-links before adding more
  text, and move examples toward canonical scripts rather than duplicating
  workflow prose.
- Split only when the result names a real physics model, numerical policy,
  differentiability boundary, or artifact contract.
- Consolidate single-use wrappers into their domain owner when that lowers
  navigation cost and preserves tests.
- Keep public facade modules (`linear.py`, `nonlinear.py`, `runtime.py`,
  `quasilinear.py`, `benchmarks.py`) as stable user entry points only.
- Remove or rename non-benchmark GX/comparison terminology in source/tests when
  it actually describes SPECTRAX-GK physics or numerics.

Execution order from the current state:

1. `geometry` and `objectives`: finish the VMEC/Boozer sensitivity and
   transport-report consolidation because those files still carry the largest
   adjacent report-assembly wrappers and are central to differentiability.
2. `validation/benchmarks`: reduce repeated scan/path/report boilerplate in TEM,
   ETG, KBM, Cyclone, and kinetic-electron benchmark families, but keep GX
   terminology only where the code is explicitly a benchmark/comparison lane.
3. `workflows`: keep runtime plotting dispatch, restart/artifact writing, and
   default-demo paths stable; touch them only for user-visible defects or to
   consolidate existing owner modules without changing `--plot` or progress.
4. `diagnostics`: simplify quasilinear-state extraction and transport-rule
   aggregation without changing schemas.
5. `solvers` and `operators`: touch hot kernels only when the boundary improves
   performance, clarity, or differentiability without hurting JIT caching.

### 4. Keep tests fast, physics-anchored, and above 95% coverage

Goal: maintain reviewer-grade confidence without unbounded local test runs.

- Add or keep tests only when they protect an equation, numerical method,
  diagnostic convention, artifact schema, restart behavior, differentiability
  contract, or known regression.
- Keep local test selections under five minutes; use CI matrix shards for wide
  coverage.
- Keep office/GPU and external-code runs out of default local tests; route them
  through explicit benchmark/validation manifests.
- Update coverage manifests whenever files move or responsibilities change.

### 5. Preserve differentiable research APIs and fast executable paths

Goal: Python workflows remain end-to-end differentiable where promoted, while
executable workflows remain fast, interactive, and user-friendly.

- Keep solver/objective kernels pure: no file I/O, plotting, terminal progress,
  subprocess calls, global mutable state, or host callbacks inside differentiated
  objectives.
- Use native JAX AD for smooth fixed-step/reduced workflows, implicit eigenpair
  differentiation for isolated linear branches, and implicit/adjoint methods
  only where FD/tangent gates pass.
- Keep adaptive/progress executable paths separate from differentiable Python
  objective paths.
- VMEC/Boozer optimization promotion requires geometry parity, gradient gates,
  conditioning diagnostics, and release-scope entries.

### 6. Preserve validation scope and reference-code parity

Goal: keep claims reproducible and honest.

- Keep validated linear/nonlinear atlas cases and release gates intact.
- Use GX only in explicit benchmark/comparison lanes: parity reruns, reference
  dumps, benchmark figures, and source-code algorithm checks.
- Do not leave GX naming in core source or user-facing workflow names unless the
  object is truly a comparison artifact.
- Keep unpromoted lanes scoped in README/docs instead of promoting them by
  implication: universal absolute QL flux, broad nonlinear turbulent-flux
  optimization, production nonlinear domain-decomposition speedup, and W7-X/TEM
  extensions.

### 7. Release sequence

Goal: ship the next version from a clean, green, measured state.

1. Confirm latest CI is green.
2. Run bounded local release gates.
3. Verify README/docs/plan claim consistency.
4. Verify repository-size manifest and no raw artifacts are tracked.
5. Bump version in `pyproject.toml` and `src/spectraxgk/_version.py`.
6. Run release-version tests and package/docs build.
7. Commit and push the version bump.
8. Confirm CI for the bump commit.
9. Tag `vX.Y.Z`, push the tag, and verify GitHub release plus PyPI publish.

## Immediate Next Tranche

1. Check the queued CI result and fix only real failures after the local
   simplification commits are pushed; do not wait on superseded runs.
2. Run the bounded local release gates after this plan update, then commit and
   push this plan refactor.
3. Take one final source-simplification tranche only if it removes a real
   navigation or policy boundary problem. Best current candidates after the
   VMEC state-sensitivity cleanup are
   `objectives/vmec_boozer_line_search.py::vmec_boozer_aggregate_line_search_holdout_report`,
   `solvers/time/diffrax_nonlinear.py::integrate_nonlinear_diffrax`,
   or benchmark scan/report helpers that still duplicate fit-window and
   branch-selection policies.
4. Audit non-benchmark `GX`/comparison terminology in source and tests; rename
   only cases that describe native SPECTRAX-GK physics or numerics rather than
   an explicit comparison artifact.
5. Keep the README runtime/memory figure in place. Refresh it only by launching
   a new measured CPU/GPU sweep with hardware/backend metadata, W7-X/HSX rows,
   peak memory, and reference-backend rows.
6. After CI is green, choose between a patch release that ships the current
   simplification state or reopening scoped science lanes explicitly in
   `docs/release_scope.rst`.

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

## Rolling Log

- 2026-06-22: Re-audited the final-closure state after `042f8d21`. Worktree was
  clean and synced with `origin/main`; the newest CI run was queued and the
  latest completed non-superseded run was green. Architecture and repository-size
  manifests passed. Source audit found 357 Python files, about 107,800 source
  lines, 316 test files, about 92,600 test lines, 0 functions >=90 lines, 20
  functions in the 80-89 line band, and 103 functions >=70 lines. README already
  has the runtime/memory panel immediately after Highlights; the plan now treats
  keeping that measured panel as a release invariant and prioritizes
  geometry/objective consolidation plus docs tightening over further file
  proliferation.
- 2026-06-22: Simplified VMEC state-sensitivity reports in
  `geometry/vmec_state_sensitivity.py` by routing optional-backend discovery,
  fail-closed exception handling, metadata packing, and payload assembly through
  shared in-file contracts. Public report names and JSON keys are unchanged.
  Focused VMEC differentiable-geometry tests passed (`5 passed`), facade and
  differentiable-refactor manifest checks passed (`3 passed`), along with ruff,
  mypy for the touched module, compileall, architecture, repository-size, and
  release-readiness checks. The 80-89 line function count dropped from 20 to 17
  and the >=70 count dropped from 103 to 100 without adding source files.
- 2026-06-22: Simplified linear implicit integration in
  `solvers/linear/implicit.py` by extracting sample-cadence validation,
  per-step GMRES solve construction, and saved-output scan orchestration from
  `_integrate_linear_implicit_cached`. The matrix-free operator,
  preconditioner selection, checkpoint policy, saved `phi` cadence, and public
  helper exports are unchanged. Focused linear implicit tests passed, along
  with ruff, mypy for the touched module, compileall, architecture,
  repository-size, release-readiness, release-version tests, and diff-hygiene
  checks. The 80-89 line function count dropped from 17 to 16 and the >=70
  count dropped from 100 to 99 without adding source files.
- Older 2026-06-21 source-simplification tranche details are preserved in the
  corresponding git commits; this root plan keeps only the current checkpoint
  and latest evidence to avoid becoming a second changelog.
