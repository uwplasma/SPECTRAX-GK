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
- Latest implementation checkpoint: `fd8fd816 Simplify kinetic benchmark
  dispatch`; plan-only checkpoints may follow it.
- Git state at audit: clean local `main`, synced with `origin/main` after the
  implementation checkpoint.
- CI state at audit: newest head run is pending; the previous Miller helper
  mypy failure is fixed. Check the head run once before release, but do not
  spend time watching superseded/cancelled runs.
- Source tree: 357 tracked Python source files under `src/spectraxgk`, 9 public
  root facades, and domain packages for API, artifacts, core, diagnostics,
  geometry, geometry backends, objectives, operators, parallel, solvers, terms,
  validation, and workflows.
- Function-size audit from the latest source pass: zero source functions at or
  above 90 lines, zero functions in the 80-89 line band, and 86 functions at or
  above 70 lines. The former 80-89 line public benchmark entry point
  `validation/benchmarks/kinetic_linear.py::run_kinetic_linear` is now a
  79-line setup/dispatch wrapper with the same public call surface and artifact
  schema.
- Tests: 316 tracked Python test files; wide CI coverage gate remains at or
  above 95% package-wide coverage.
- Docs/readme: README, docs, examples, benchmarks, release scope, architecture,
  differentiability, performance, validation, and code-structure docs exist.
  The next pass should tighten, cross-link, and remove stale claims rather than
  add broad narrative by default.
- Repository footprint: tracked files total 48,423,613 bytes after trimming
  unreferenced duplicate stellarator optimization PDFs; no tracked files above
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
  linear explicit dispatch, twist-shift cache policy, duplicate optimization
  artifacts, solver-ready flux-tube geometry packing, VMEC/Boozer field-line
  sampling assembly, runtime scan batch orchestration, Cyclone time-batch
  result branching, Miller straight-theta rebuild staging, and objective
  portfolio sensitivity gates without adding new public behavior. The kinetic
  benchmark entry point now packs time/fit controls into typed internal request
  objects, reducing public dispatch complexity while preserving the stable API.
- A non-benchmark terminology audit found no `GX` tokens in Python source
  outside the release-scope documentation test. Remaining `reference_aligned`
  identifiers are benchmark API terms; native operator/solver comments now use
  benchmark-compatible numerical wording instead.
- Package-wide coverage gate is maintained by CI shards at or above 95%.
- Production parallelization claims are limited to independent ky/batch/UQ
  work. Nonlinear domain decomposition remains diagnostic until stronger gates
  pass.

## Open Lanes and Closure Gates

| Priority | Lane | Current status | Closure gate |
| --- | --- | ---: | --- |
| P0 | CI/release hygiene | 99% | Latest head CI green, bounded local release gates pass, version bump/tag publish cleanly. |
| P0 | README/docs/plan consistency | 99% | README references current figures only; docs, release scope, and plan agree on promoted and deferred claims. |
| P1 | Source simplification and naming | 100% scoped | No new root modules, source-file count non-increasing, zero functions >=90 lines, zero functions in the 80-89 band, and non-benchmark comparison-code terminology audited. |
| P1 | Refactor/testability | 99.9% | Tests map to domain ownership; no migration-era wrappers or stale compatibility paths remain in examples/docs. |
| P1 | Package coverage and physics tests | 100% gate | Wide package coverage stays >=95%; new tests are physics, numerics, artifact, AD, or regression gates, not smoke-only scaffolds. |
| P2 | Runtime/memory and performance claims | 98% scoped | README panel uses measured artifacts with hardware/backend metadata; new speedup claims require identity plus profiler gates. |
| P2 | Differentiable Python workflows | 99% scoped | Promoted observables have AD/FD/tangent/conditioning/covariance or implicit-differentiation checks. |
| P2 | VMEC/Boozer differentiable geometry | 99% scoped | Promoted geometry/optimization rows have parity and gradient gates; broad optimization claims remain scoped. |
| P2 | Production parallelization | 95% scoped | Independent-work paths are production; nonlinear domain decomposition stays diagnostic until full transport-window identity and CPU/GPU speedup pass. |
| P3 | Quasilinear model development | 99% scoped | Screening/model-development diagnostics are documented; universal absolute-flux predictor remains unpromoted unless held-out gates pass. |
| P3 | Nonlinear turbulent-flux optimization | 94% scoped | Long post-transient matched audits plus a user-facing matched-audit example support scoped examples; broad multi-surface optimized-stellarator turbulence claims remain unpromoted. |
| P4 | W7-X/TEM extensions | deferred | W7-X zonal recurrence, W7-X TEM/multi-flux-tube, and fluctuation-spectrum panels are post-release unless explicitly reopened. |

## Prioritized Execution Plan

The remaining work is deliberately ordered to finish release-critical technical
lanes before reopening science lanes. Do not start new large validation
campaigns until the codebase, documentation, and release gates below are
stable.

### 1. Freeze the checkpoint and CI

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

### 2. Lock the README/runtime-memory claim surface

Goal: the README clearly shows measured CPU/GPU runtime and memory against GX
benchmark rows without unsupported speedup claims.

1. Keep `docs/_static/runtime_memory_benchmark.png` immediately after
   Highlights and before the claim-scope ledger.
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
   - stop line-count-driven source churn unless a real duplicated policy or
     stale compatibility path is found;
   - benchmark scan/report helpers that duplicate fit-window, branch-selection,
     or report-packing policies;
   - docs/examples references to stale migration-era paths, if any remain.

### 4. Complete naming and documentation consistency

Goal: make the repository understandable to new users and contributors.

1. Keep GX mentions only in benchmark/comparison context: parity reruns,
   reference plots, validation tables, and comparison documentation.
2. Rename native-code wording such as "GX-style" or "GX-reference" to the
   underlying numerical or physical convention when it is not explicitly a
   benchmark artifact.
3. Keep docs organized by domain: theory, numerics, geometry, validation,
   performance, examples, differentiability, and release scope.
4. Do not expand the README with long derivations. README should show install,
   quickstart, runtime/memory panel, main validation/optimization figures, and
   claim boundaries; detailed equations and derivations belong in docs.

### 5. Keep tests fast and physics-anchored

Goal: keep confidence high without unbounded local runs.

1. Local test selections should stay under five minutes.
2. Wide coverage remains a CI-matrix responsibility.
3. New tests must protect equations, numerical convergence, diagnostic
   conventions, artifact schemas, restart behavior, differentiability
   contracts, or known regressions.
4. Office/GPU/GX reruns stay in explicit benchmark/validation manifests, not
   default local tests.

### 6. Preserve differentiability and fast executable paths

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

### 6a. Finalize nonlinear turbulent-flux optimization scope

Goal: close the release lane without overstating the science claim.

1. Keep the three VMEC-JAX-style QA optimizer examples as candidate generators:
   growth, quasilinear flux, and nonlinear-window heat-flux screening.
2. Use `examples/optimization/QA_nonlinear_ITG_matched_audit.py` as the
   production-evidence example: it consumes accepted long-window baseline and
   optimized ensemble sidecars and writes the matched reduction audit.
3. Use `examples/optimization/QA_nonlinear_ITG_transport_matrix.py` as the
   broad-evidence example: it writes the three-surface, two-field-line,
   three-`k_y` matched nonlinear matrix and the GPU split launch scripts from a
   solved baseline/candidate WOUT pair.
4. Promote a new low-turbulence stellarator only when the matched audit passes:
   both ensembles qualify, the optimized post-transient mean is lower by the
   configured threshold, and the difference is uncertainty separated.
5. Treat the current positive evidence as scoped: no-ESS-to-optimized QA/ESS
   plus two projected-weight max-mode-5 audits pass; strict `t=1500`
   growth/QL/nonlinear-window candidates are negative transfer evidence.
6. Use `tools/build_matched_nonlinear_transport_matrix.py write` directly only
   for scripted campaigns; it is the lower-level helper used by the example.
   The default matrix is `s=(0.45,0.64,0.78)`, `alpha=(0,pi/4)`, and
   `k_y rho_i=(0.10,0.30,0.50)`, with seed/timestep replicated fixed-step
   nonlinear windows over `t=[1100,1500]`.
7. Run the generated staged-ladder script on office/GPU, then run the generated
   postprocess script. The companion `report` subcommand promotes the matrix
   only if the completed matched comparisons satisfy the configured
   pass-fraction and mean-reduction gates.
8. Use `tools/check_nonlinear_transport_matrix_portfolio.py` when multiple
   selected families have matrix reports. It selects a passing broad family,
   records strict `t=1500` growth/QL/nonlinear-window rows only as excluded
   negative-transfer evidence, and blocks release promotion if no broad matrix
   report passes.
9. Defer broad nonlinear turbulent-flux optimization claims until that
   multi-surface, multi-field-line, multi-`k_y` matrix passes for the selected
   optimization family. Single-point positive audits remain scoped evidence.

Current launch log:

- `2026-06-22`: generated the accepted QA/ESS max-mode-5 matrix on office from
  `/home/rjorge/vmec_jax_autoscalar_20260601/examples/optimization/results/qs_ess_sweep/gpu/continuation/qa/mode5/no_ess/wout_final.nc`
  to
  `/home/rjorge/vmec_jax_autoscalar_20260601/examples/optimization/results/qs_ess_sweep/gpu/continuation/qa/mode5/ess/wout_final.nc`.
  The campaign lives under
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix` with artifacts in
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix_artifacts`.
- The first trial exposed a misleading checkpoint log line (`8000` steps for
  each `t=1500` command). This is checkpoint chunking, not a horizon cap; the
  strengthened local regression
  `tests/test_runtime_artifacts.py::test_runtime_orchestration_handoff_chunks_and_restarts`
  now verifies that checkpointed nonlinear artifact handoff accumulates all
  requested steps.
- Relaunched final-horizon direct queues on office at commit `770386d2`:
  GPU0 PID `255406`, GPU1 PID `255407`. Each queue has 54 independent
  `t=1500` jobs. After completion, run
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix/run_matrix_postprocess.sh`
  and inspect
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix_artifacts/qa_mode5_ess_matrix_matrix_report.json`.
- Staged, but did not launch, the two projected max-mode-5 follow-up families:
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/projected_0p0005_matrix`
  and
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/projected_0p001_matrix`.
  They use the one-point QA baseline WOUT against the tracked projected
  transport-weight `5e-4` and `1e-3` WOUTs; both generated manifests pass the
  same 18-point coverage gate.
- Added `tools/check_matched_nonlinear_transport_matrix_progress.py` to avoid
  counting checkpoint bundles as final outputs. The first office progress
  report at
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix_artifacts/qa_mode5_ess_matrix_progress.json`
  found `2/108` complete bundles and `0/108` outputs confirmed at `t=1500`, so
  postprocessing is not ready yet.
- `2026-06-22`: added the non-invasive bundle-only progress path
  (`--skip-time-check`) and `tools/check_nonlinear_output_target.py` so active
  NetCDF files do not need to be read during office polling and interrupted
  checkpoint bundles are not skipped on relaunch. The latest safe office poll
  found `11/108` bundle-complete outputs; a target-time check of non-active
  bundles confirmed `10` true final-horizon outputs at `t≈1500`, with both
  GPU queues still active. Postprocessing remains blocked until all `108`
  outputs are target-confirmed.
- `2026-06-23`: resumed from the active office queues and fast-forwarded the
  office clone to `5e94a51d`. The non-invasive progress report now finds
  `95/108` bundle-complete outputs under `--skip-time-check`. The
  remaining rows are all on the outer surface/second field-line label
  (`s=0.78`, `alpha=pi/4`), mainly `k_y rho_i=0.30,0.50`, plus one candidate
  `k_y rho_i=0.10` replicate. Both office GPUs remain saturated, so
  postprocessing is still blocked until the last `13` bundles finish.
- `2026-06-23`: while the office matrix continued, local release hygiene checks
  passed: package-architecture manifest, repository-size manifest,
  release-readiness check, Ruff over `src tests examples tools`, and the
  focused nonlinear matrix test shard (`23 passed`).
- `2026-06-23`: installed a bounded office watcher at
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/watch_qa_mode5_ess_matrix_postprocess.sh`.
  It uses a lock file, polls active queues every five minutes with
  `--skip-time-check`, switches to the full target-time progress check only
  after the queue processes exit, and then runs
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix/run_matrix_postprocess.sh`
  exactly once when `ready_for_postprocess=true`. The watcher log is
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix_artifacts/qa_mode5_ess_matrix_watch.log`.
- `2026-06-23`: verified the fallback projected max-mode-5 matrix families are
  staged but not launched:
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/projected_0p0005_matrix`
  and
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/projected_0p001_matrix`.
  Each has a valid 18-sample manifest and GPU split scripts. Keep them idle
  until the accepted QA/ESS matrix either passes or fails; launching extra CPU
  duplicates while the old GPU queue scripts are active is unsafe because those
  scripts do not coordinate target-aware per-output locks with external
  workers.
- `2026-06-23`: the watcher second poll still found `95/108`
  bundle-complete outputs under `--skip-time-check`. The active rows are
  `qa_mode5_ess_s0p78_a0p785398_ky0p1_seed31` and
  `qa_mode5_no_ess_s0p78_a0p785398_ky0p3_seed31`; both remain GPU-bound. The
  focused matrix/report/target tests passed (`8 passed`) and the optimization
  example test shard passed (`29 passed`).
- `2026-06-23`: fixed `tools/check_nonlinear_output_target.py` so direct
  `python3 tools/check_nonlinear_output_target.py ...` execution works from a
  repo checkout, matching the generated target-aware relaunch scripts. Also
  tightened `tools/check_matched_nonlinear_transport_matrix_progress.py` so
  `--skip-time-check` is explicitly bundle-only: it never marks
  `ready_for_postprocess=true` or labels outputs target-confirmed without
  reading their NetCDF time. A direct check of the long active
  `qa_mode5_ess_s0p78_a0p785398_ky0p1_seed31` output found
  `tmax≈1199.93`, so the GPU0 process is still useful work rather than a
  duplicate final output. The latest non-invasive poll has `96/108`
  bundle-complete outputs; full target confirmation remains deferred until the
  queue exits and the watcher runs the non-skip progress check.
- `2026-06-23`: installed a second office watcher at
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/watch_nonlinear_matrix_portfolio_gate.sh`.
  It waits for
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix_artifacts/qa_mode5_ess_matrix_matrix_report.json`
  and then runs `tools/check_nonlinear_transport_matrix_portfolio.py` with the
  accepted QA/ESS report, the two staged projected max-mode-5 report paths,
  and the strict growth/QL/nonlinear-window negative-transfer comparisons as
  excluded evidence. Its output will be
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix_artifacts/nonlinear_transport_matrix_portfolio.{json,png}`.
  This watcher does not launch simulations; it only automates the gate after
  accepted-matrix postprocessing finishes.
- CI for code head `5e94a51d` passed, and local gates passed after the
  plan-only progress commits. Do not watch superseded/cancelled runs while the
  office final-horizon matrix is still executing.
- `2026-06-23`: latest accepted-matrix office poll on `main` commit
  `6563b191` still has GPU0/GPU1 queues active. The bundle-only progress
  report now correctly labels `--skip-time-check` as non-promotional:
  `96/108` bundles are present, `0/108` are target-confirmed under the skipped
  NetCDF-time path, and `ready_for_postprocess=false`. Direct target checks of
  the active rows found `tmax≈1199.93` for the candidate
  `s=0.78, alpha=pi/4, k_y rho_i=0.10, seed31` row and `tmax≈799.95` for the
  baseline `s=0.78, alpha=pi/4, k_y rho_i=0.30, seed31` row, so both processes
  are still useful long-window work rather than duplicate completed outputs.
- `2026-06-23`: regenerated the idle projected fallback families on office
  from the current target-aware matrix generator:
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/projected_0p0005_matrix`
  and
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/projected_0p001_matrix`.
  Their GPU split scripts now call `tools/check_nonlinear_output_target.py`
  and print `skip-target-confirmed` only for outputs whose recorded time
  reaches `t=1500` within the generated timestep tolerance. They are ready as
  fallback launch scripts, but remain idle until the accepted QA/ESS matrix
  either passes or fails.
- `2026-06-23`: tightened the pre-manuscript closure dashboard so the broad
  nonlinear turbulent-flux optimization lane cannot close from scoped matched
  audits alone. `tools/build_pre_manuscript_closure_status.py` now requires a
  passing `docs/_static/nonlinear_transport_matrix_portfolio.json`, so the
  tracked dashboard reports this lane as `94%` and blocked only by the missing
  broad matrix portfolio while the accepted QA/ESS office matrix is still
  running.
- `2026-06-25`: audited the completed office accepted QA/ESS matrix. The
  full target-time progress check passed with `108/108` outputs confirmed at
  `t=1500` and `ready_for_postprocess=true`; the postprocess report exists at
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix_artifacts/qa_mode5_ess_matrix_matrix_report.json`.
  The aggregate broad matrix gate failed: `9/18` samples passed, mean relative
  reduction was `9.18%`, and the blocker was `pass_fraction 0.5 < 1`. The
  portfolio gate therefore selected no family and wrote
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/matrix_artifacts/nonlinear_transport_matrix_portfolio.json`
  with blocker `no candidate family passed the broad matrix gate`.
- `2026-06-25`: launched exactly one target-aware fallback family:
  `projected_0p001`. The scripts
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/projected_0p001_matrix/run_matrix_final_horizon_gpu0.sh`
  and `run_matrix_final_horizon_gpu1.sh` are running on office and use
  `tools/check_nonlinear_output_target.py` before every output. The companion
  watcher
  `/home/rjorge/spectrax_nonlinear_matrix_20260622/watch_projected_0p001_matrix_postprocess_and_portfolio.sh`
  will switch from bundle-only polling to full target-time verification after
  the queue exits, run matrix postprocess only if `ready_for_postprocess=true`,
  and rerun the portfolio gate with accepted QA/ESS plus projected `0p001`.
  Do not launch `projected_0p0005` unless this single fallback completes and
  still fails the portfolio gate.
- `2026-06-25`: rechecked the active `projected_0p001` queue with the full
  target-time progress checker, not just bundle presence. Both GPU queue
  scripts and the watcher are still running; the current status is `2/108`
  complete bundles, `0/108` outputs confirmed at `t=1500`, and
  `ready_for_postprocess=false`. The two present bundles are therefore
  intermediate/in-progress outputs and must not be postprocessed or promoted.
- `2026-06-25`: while the office fallback queue continued, reran bounded local
  release checks. `tools/check_release_readiness.py`,
  `tools/check_release_artifact_manifest.py`, `tools/check_release_version.py`,
  `tools/check_repository_size_manifest.py`,
  `tools/check_package_architecture_manifest.py`,
  `tools/check_performance_optimization_manifest.py`, and
  `tools/check_parallel_scaling_artifacts.py` passed. The matrix-focused
  pytest shard passed (`14 passed`), the package build produced the `1.6.9`
  wheel/sdist, and the Sphinx HTML documentation build passed. The only
  remaining release-critical item in this tranche is the nonlinear matrix
  portfolio outcome from office.
- `2026-06-25`: latest `main` CI for commit `a18149a7` passed. Rechecked the
  office `projected_0p001` fallback with full target-time validation after the
  queue had advanced for several hours. The queue is still active on both GPUs,
  but it now has real long-window progress: `23/108` bundles are present and
  `21/108` outputs are confirmed at `t=1500`. The matrix remains
  `ready_for_postprocess=false`, so no report, closure artifact, version bump,
  or release tag should be created yet.
- `2026-06-25`: rechecked office after the latest continuation. Both
  `projected_0p001` GPU queues and the postprocess watcher are still active,
  both GPUs are saturated, and the full target-time progress checker reports
  `24/108` bundles present with `22/108` outputs confirmed at `t=1500`.
  Two active rows are still below target (`tmax≈1199.93` and `tmax≈799.95`),
  so the fallback is progressing but remains non-promotional. The latest
  `main` CI run for commit `49f04c2b` has no failing jobs at this checkpoint:
  `28` jobs have succeeded, `8` are in progress, and `22` wide-coverage jobs
  are queued.
- `2026-06-25`: rechecked the running `projected_0p001` fallback again. Both
  active rows now report `tmax≈1199.93`, but only `22/108` outputs are
  target-confirmed and the matrix is still `ready_for_postprocess=false`.
  The already-running office scripts are target-aware but not lock-aware, so
  no extra CPU workers were launched against the same output tree. Updated
  `tools/build_matched_nonlinear_transport_matrix.py` so newly generated
  final-horizon scripts guard each output with a per-output `flock` lock and
  an atomic-directory fallback; this makes any future regenerated fallback
  family safe for split workers or relaunches without output races.
- `2026-06-25`: rechecked the same office `projected_0p001` fallback after
  the lock-safe generator commit. The current office checkout is still the
  pre-lock run context and both active GPU processes remain healthy, so the
  office tree was intentionally left untouched. Full target-time progress now
  reports `23/108` outputs confirmed at `t=1500`, `24/108` bundles present,
  and `ready_for_postprocess=false`; the running candidate row remains at
  `tmax≈1199.93`. Head CI for commit `3a570f7c` is queued/in progress with no
  visible failures at this checkpoint.
- `2026-06-25`: added a fail-closed release importer,
  `tools/import_nonlinear_transport_matrix_portfolio.py`, so passing portfolio
  artifacts are copied into `docs/_static` only after
  `nonlinear_transport_matrix_portfolio_gate.passed=true`. The importer and
  portfolio tests pass, Ruff passes, release-readiness/artifact/repository-size
  checks pass, and the Sphinx docs build passes. This removes the remaining
  manual-copy ambiguity after the office matrix gate.
- `2026-06-25`: tightened the public optimization example contract in
  `examples/optimization/README.md`. The examples now explicitly document the
  top-level knobs for optimizer algorithm, VMEC seed/WOUT geometry, SPECTRAX-GK
  objective kind, sample set, extra VMEC-JAX objective tuples, and production
  long-window nonlinear audit policy. The VMEC-JAX QA exact-script test shard
  now guards those user-facing customization instructions.
- `2026-06-25`: added
  `tools/finalize_nonlinear_transport_matrix_release.py`, a fail-closed wrapper
  for the post-portfolio release path. Once office writes a passing portfolio,
  this tool imports the selected family artifacts, rejects blocked portfolios,
  and regenerates the manuscript-readiness plus pre-manuscript closure panels.
  It is documented in `examples/optimization/README.md` next to the broad
  nonlinear transport matrix workflow.
- `2026-06-25`: rechecked the active office `projected_0p001` fallback after
  the importer/examples commits. Both GPU queue scripts and the watcher are
  still active; both GPUs are saturated. Full target-time progress reports
  `24/108` outputs confirmed at `t=1500`, `26/108` bundles present, and
  `ready_for_postprocess=false`. Two present bundles are intermediate outputs
  (`tmax≈400` and `tmax≈800`), so no postprocess/import/release action is
  allowed yet.

### 7. Preserve validation scope and GX parity

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

### 8. Release sequence

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

1. Let the single launched `projected_0p001` fallback finish on office. Use
   only non-invasive bundle-only polling while queue processes are active.
2. After both projected queue scripts exit, run the full target-time progress
   check. If `ready_for_postprocess=true`, let the watcher or manual
   postprocess build
   `qa_projected_weight_0p001_matrix_matrix_report.{json,png}`.
3. Run or inspect the portfolio gate. If projected `0p001` passes, copy the
   matrix and portfolio artifacts into `docs/_static` with
   `tools/import_nonlinear_transport_matrix_portfolio.py`, update README/docs
   and release scope with the final broad nonlinear turbulent-flux optimization
   evidence, then run bounded release gates. The importer is fail-closed and
   refuses blocked portfolio JSON, so manual copying cannot accidentally
   promote a failed broad nonlinear transport matrix.
4. If projected `0p001` fails, keep the broad nonlinear optimization claim
   blocked and decide whether to defer the claim or launch the remaining
   `projected_0p0005` fallback as a separate tranche.
5. After the nonlinear optimization portfolio is resolved or explicitly
   deferred in release scope, finish the final
   README/docs/release-scope consistency pass, package build, version bump,
   tag, and publish.
