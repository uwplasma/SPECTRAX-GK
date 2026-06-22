# SPECTRAX-GK Final Closure Plan

This is the single active execution plan after the `v1.6.9` release checkpoint
and the post-release simplification passes. Chronology belongs in git history,
CI logs, release notes, and validation artifacts; this file defines the finite
steps needed to ship the next polished version.

## One-Sentence Plan

Finish SPECTRAX-GK as a compact, domain-organized, JAX-native gyrokinetic code with a small and easy-to-manage source tree, explicit GX parity only in benchmark/comparison lanes, physics-anchored tests and documentation, an accurate README with the runtime/memory GX-vs-SPECTRAX-GK panel near the top, differentiable Python research workflows, and low-runtime/low-memory CPU/GPU execution backed only by measured artifacts.

## Current State

Last audited: 2026-06-22 on `main`.

- Latest released tag: `v1.6.9`.
- Current head: `c0c35068 Record twist-shift policy simplification`.
- Git state at audit: clean local `main`, synced with `origin/main`.
- CI state at audit: newest head run was queued; latest completed
  non-superseded run was green. Check the head run once before release, but do
  not spend time watching superseded/cancelled runs.
- Source tree: 357 tracked Python source files under `src/spectraxgk`, 9 public
  root facades, and domain packages for API, artifacts, core, diagnostics,
  geometry, geometry backends, objectives, operators, parallel, solvers, terms,
  validation, and workflows.
- Function-size audit from the latest source pass: zero source functions at or
  above 90 lines, 7 functions in the 80-89 line band, and 92 functions at or
  above 70 lines.
- Tests: 316 tracked Python test files; wide CI coverage gate remains at or
  above 95% package-wide coverage.
- Docs/readme: README, docs, examples, benchmarks, release scope, architecture,
  differentiability, performance, validation, and code-structure docs exist.
  The next pass should tighten, cross-link, and remove stale claims rather than
  add broad narrative by default.
- Repository footprint: tracked files total about 49 MB; no tracked files above
  2 MB. Large local checkout size is from ignored/generated artifacts such as
  `.venv`, caches, `docs/_build`, `dist`, and `tools_out`, not tracked release
  content.
- README performance surface: `docs/_static/runtime_memory_benchmark.png` is
  restored after Highlights and before claim scope, with measured CPU/GPU and
  GX/reference rows including W7-X and HSX.
- Benchmark layout: lightweight benchmark drivers and manifests live at root
  `benchmarks/`; raw transient outputs remain ignored/out of git.

## Authority Map

- `plan.md`: active priorities, gates, and release sequence.
- `docs/architecture_refactor_plan.rst`: target package layout, naming policy,
  and what counts as useful simplification.
- `docs/differentiable_refactor_plan.rst`: differentiability strategy and AD
  gate requirements.
- `docs/code_structure.rst`: current source-tree map.
- `docs/release_scope.rst`: claim ledger for README, docs, releases, and papers.
- `tools/*manifest*.toml`: executable gates for architecture, size,
  validation, performance, differentiability, artifacts, and release readiness.

Conflict order: release-scope claims, architecture plan, executable manifests,
then this plan.

## Closed Results

- `v1.6.9` release passed GitHub release and PyPI publication.
- `spectraxgk` executable quickstart works, emits progress/ETA, writes a
  reproducible default input in the current directory, and supports
  `spectraxgk --plot` for supported saved linear/nonlinear outputs.
- Runtime/memory comparison panel is back near the top of the README and tied
  to measured artifacts.
- Root-level prefix sprawl was removed; stable public facades now sit over
  domain packages.
- Recent refactors simplified runtime, solver setup, nonlinear Diffrax/IMEX,
  validation reports, VMEC/Boozer gates, nonlinear-gradient/report paths,
  quasilinear optimized-equilibrium audit inputs, KBM beta Krylov sample policy,
  linear explicit dispatch, and twist-shift cache policy without adding new
  public behavior.
- Package-wide coverage gate is maintained by CI shards at or above 95%.
- Production parallelization claims are limited to independent ky/batch/UQ
  work. Nonlinear domain decomposition remains diagnostic until stronger gates
  pass.

## Open Lanes and Closure Gates

| Priority | Lane | Current status | Closure gate |
| --- | --- | ---: | --- |
| P0 | CI/release hygiene | 99% | Latest head CI green, bounded local release gates pass, version bump/tag publish cleanly. |
| P0 | README/docs/plan consistency | 99% | README references current figures only; docs, release scope, and plan agree on promoted and deferred claims. |
| P1 | Source simplification and naming | 99.9% | No new root modules, source-file count non-increasing, zero functions >=90 lines, and remaining long helpers either simplified or justified. |
| P1 | Refactor/testability | 99.7% | Tests map to domain ownership; no migration-era wrappers or stale compatibility paths remain in examples/docs. |
| P1 | Package coverage and physics tests | 100% gate | Wide package coverage stays >=95%; new tests are physics, numerics, artifact, AD, or regression gates, not smoke-only scaffolds. |
| P2 | Runtime/memory and performance claims | 98% scoped | README panel uses measured artifacts with hardware/backend metadata; new speedup claims require identity plus profiler gates. |
| P2 | Differentiable Python workflows | 99% scoped | Promoted observables have AD/FD/tangent/conditioning/covariance or implicit-differentiation checks. |
| P2 | VMEC/Boozer differentiable geometry | 99% scoped | Promoted geometry/optimization rows have parity and gradient gates; broad optimization claims remain scoped. |
| P2 | Production parallelization | 95% scoped | Independent-work paths are production; nonlinear domain decomposition stays diagnostic until full transport-window identity and CPU/GPU speedup pass. |
| P3 | Quasilinear model development | 99% scoped | Screening/model-development diagnostics are documented; universal absolute-flux predictor remains unpromoted unless held-out gates pass. |
| P3 | Nonlinear turbulent-flux optimization | 91% scoped | Long post-transient matched audits support scoped examples; broad optimized-stellarator turbulence claims remain unpromoted. |
| P4 | W7-X/TEM extensions | deferred | W7-X zonal recurrence, W7-X TEM/multi-flux-tube, and fluctuation-spectrum panels are post-release unless explicitly reopened. |

## Prioritized Execution Plan

### 1. Freeze the checkpoint

Goal: keep `main` clean before further refactor or release work.

1. Check the newest non-superseded CI run once.
2. Fix real failures only; ignore cancelled runs superseded by newer pushes.
3. Run bounded local gates after any plan/docs/readme or source tranche:
   - `python tools/check_package_architecture_manifest.py`
   - `python tools/check_repository_size_manifest.py`
   - `python tools/check_release_readiness.py --out-json /tmp/spectrax_release_readiness.json`
   - `python -m pytest -q tests/test_check_release_readiness.py tests/test_check_repository_size_manifest.py tests/test_check_release_version.py --maxfail=1`
   - `python -m ruff check src tests tools benchmarks`
   - `python -m sphinx -b html docs /tmp/spectrax_docs_plan_build`
   - `git diff --check`

### 2. Lock the README runtime/memory panel

Goal: the README clearly shows measured CPU/GPU runtime and memory against GX
benchmark rows without unsupported speedup claims.

1. Keep `docs/_static/runtime_memory_benchmark.png` immediately after
   Highlights.
2. Keep the figure provenance files tracked:
   - `docs/_static/runtime_memory_summary_ship_refresh.json`
   - `docs/_static/runtime_memory_results_ship_refresh.csv`
   - `docs/_static/runtime_memory_benchmark.png`
3. Refresh the panel only from new measured CPU/GPU artifacts with hardware,
   backend, wall-time, peak-memory, and W7-X/HSX rows.
4. Keep nonlinear domain-decomposition speedup out of the README until identity
   and profiler gates pass.

### 3. Finish source simplification without adding file sprawl

Goal: make the code easier to navigate and extend while keeping stable public
facades.

1. Do not add new root modules or migration-era compatibility facades.
2. Do not increase the 357-file source count unless the same tranche deletes or
   consolidates at least as many files and lowers navigation cost.
3. Keep public facades (`linear.py`, `nonlinear.py`, `runtime.py`,
   `quasilinear.py`, `benchmarks.py`, and `cli.py`) as user entry points only.
4. Consolidate single-use internal wrappers into domain owners when this lowers
   navigation cost and preserves tests.
5. Rename non-benchmark GX/comparison terminology to physics or numerics names;
   keep GX naming only in explicit benchmark/comparison tools, tests, docs, and
   plots.
6. Next source candidates, in priority order:
   - `geometry_backends/vmec_fieldlines.py::_sample_fieldline_boozer_state`
   - `geometry/flux_tube_contract.py::flux_tube_geometry_from_mapping`
   - `workflows/runtime/orchestration_scan.py::run_runtime_scan_batch`
   - benchmark scan/report helpers that duplicate fit-window, branch-selection,
     or report-packing policies

### 4. Keep tests fast and physics-anchored

Goal: keep confidence high without unbounded local runs.

1. Local test selections should stay under five minutes.
2. Wide coverage remains a CI-matrix responsibility.
3. New tests must protect equations, numerical convergence, diagnostic
   conventions, artifact schemas, restart behavior, differentiability
   contracts, or known regressions.
4. Office/GPU/GX reruns stay in explicit benchmark/validation manifests, not
   default local tests.

### 5. Preserve differentiability and fast executable paths

Goal: Python research workflows are differentiable where promoted, while the
executable remains fast and informative.

1. Keep solver/objective kernels pure: no file I/O, plotting, subprocesses,
   terminal progress, host callbacks, or global mutable state inside
   differentiated objectives.
2. Use native JAX AD for smooth fixed-step/reduced workflows, implicit eigenpair
   differentiation for isolated linear branches, and implicit/adjoint methods
   only after FD/tangent gates pass.
3. Keep adaptive/progress executable paths separate from differentiable Python
   objective paths.
4. VMEC/Boozer optimization promotion requires geometry parity, gradient gates,
   conditioning diagnostics, and release-scope entries.

### 6. Preserve validation scope and GX parity

Goal: claims remain reproducible and honest.

1. Keep validated linear/nonlinear atlas cases and release gates intact.
2. Use GX only for explicit comparison lanes: parity reruns, reference dumps,
   benchmark figures, and algorithm checks.
3. Do not leave GX naming in core source or user workflows unless the object is
   truly a comparison artifact.
4. Keep deferred lanes scoped in README/docs: universal absolute QL flux,
   broad nonlinear turbulent-flux optimization, production nonlinear domain
   decomposition, W7-X zonal recurrence, W7-X TEM/multi-flux-tube, and W7-X
   fluctuation spectra.

### 7. Release sequence

Goal: ship the next version from a clean, green, measured state.

1. Confirm head CI is green.
2. Run bounded local release gates.
3. Verify README/docs/plan claim consistency.
4. Verify repository-size manifest and no raw outputs are tracked.
5. Bump version in `pyproject.toml` and `src/spectraxgk/_version.py`.
6. Run package, docs, release-version, and artifact checks.
7. Commit and push the version bump.
8. Confirm CI for the bump commit.
9. Tag `vX.Y.Z`, push the tag, and verify GitHub release plus PyPI publish.

## Immediate Next Tranche

1. Run the bounded local gates for this plan refactor and commit/push it.
2. Check the queued CI run once after the push; fix only real failures.
3. Take one final source-simplification tranche from the priority list above,
   preferring an in-file simplification or consolidation that does not increase
   source-file count.
4. Audit non-benchmark GX terminology in `src`, `tests`, `docs`, and README;
   rename only native-code references, not benchmark/comparison artifacts.
5. If no real code-quality blockers remain, move directly to release readiness:
   docs/readme consistency pass, package build, version bump, tag, and publish.
