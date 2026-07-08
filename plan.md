# SPECTRAX-GK Consolidation And Performance Refactor Plan

This is the active plan for the next large refactor. It replaces the previous
campaign-log-oriented plan with a finite implementation plan focused on a
smaller, clearer, faster, research-grade codebase.

## One-Sentence Plan

Refactor SPECTRAX-GK into a compact JAX-native gyrokinetic package with at most
100 installable source files, fewer than 100 test files, fewer than 100 tool
scripts, validation code moved out of the runtime package, legacy or
non-promoted functionality removed from `main`, a clearer examples and
benchmarks layout, physics-anchored tests, explicit benchmark-only comparison
references, and profiler-backed CPU/GPU performance improvements while
preserving validated solver behavior and public user workflows.

## Authoritative Complexity-Reduction Plan

This section is the current decision authority. Older progress-log sections below
are retained only for audit history; if they conflict with this section, this
section wins.

### File Retention Rule

Every retained file must have all four fields below. Files that fail the rule are
deleted from `main` or moved to a draft experiment branch/PR.

| Field | Allowed answer | Not allowed |
| --- | --- | --- |
| Owner | runtime package, example, root benchmark, release/tooling, docs artifact, test, active campaign | historical branch, local probe, unused manuscript scratch |
| Reason | promoted user workflow, reusable physics/numerics API, reproducible benchmark, CI/release gate, documented figure/table, active long-run campaign | convenience wrapper, preserved old branch behavior, unreferenced output builder |
| Test owner | unit/integration/validation/tool/release test family or explicit manual office benchmark | no test and no documented manual gate |
| Destination | `src`, `examples`, `benchmarks`, `tools`, `tests`, `docs/_static`, external draft PR | installable validation/campaign package, raw output directory, duplicated tool script |

### Current Findings

- Obsolete branches are not the problem in this checkout: local and remote branch
  state is only `main`/`origin/main`; old experimental PRs #4-#6 are closed.
- Tracked repository data is not the blocker: no tracked file is above 2 MB; the
  large local footprint is ignored generated state (`tools_out`, `docs/_build`,
  `.mypy_cache`, `.venv`).
- Complexity is concentrated in code topology: the oversized public benchmark
  facade, `tools/artifacts`, `tests/tools`, large runtime/benchmark tests, and
  one-campaign/one-artifact tooling.
- `benchmarks/` is already small enough and should remain a root-level user and
  developer benchmark-driver layer. The confusing benchmark code is the
  large public `spectraxgk.benchmarks` implementation that absorbed the retired
  validation benchmark package.
- The completed nonlinear-gradient, nonlinear-transport, and stellarator
  extractions are the model for future moves: reusable diagnostics stay in
  `src/spectraxgk/diagnostics`, campaign planning moves to `tools/campaigns`,
  release decisions stay in `tools/release`, and installable validation
  packages disappear.

### 2026-07-07 Simplification Reset

The previous refactor tranches successfully removed the installable validation
package, but they also left too many small source, test, and tool files. From
this point forward, the default action is **merge, delete, or move out of the
installable package**, not split into more modules.

Current tracked audit:

| Area | Current tracked state | Main issue | Next action |
| --- | ---: | --- | --- |
| Branches | `main`, `origin/main` only | no branch cleanup needed | keep experiments in one draft PR, not on `main` |
| Tracked large files | none above 2 MB | local size comes from ignored caches/output | keep release artifact audit fail-closed |
| Source package | 277 Python files, 100,402 LOC | benchmark facade, many tiny geometry/objective/operator shards | consolidate by domain and move benchmark-only workflows out of `src` |
| Tests | 241 Python files, 96,720 LOC | one-file-per-tool and monkeypatch-heavy branch tests | table-driven contract families with shared fixtures |
| Tools | 239 Python files, 100,526 LOC | one-script-per-artifact/campaign/status | subcommand-style drivers plus manifest data |
| Root benchmarks | 12 Python files, 1,589 LOC | role is acceptable but results are under-documented | keep at root and document outputs in docs |
| Docs static | 1,572 files, about 38.5 MiB | many historical evidence files | prune by README/docs/release-manifest reference graph |

Near-term implementation rule:

- A source refactor tranche must reduce the total installed source file count or
  remove a larger obsolete source path. New internal modules are allowed only
  when they replace multiple existing files or isolate a measured JIT/performance
  boundary.
- A test refactor tranche must reduce test file count or test LOC while keeping
  the same physics, numerical, release, and coverage gates.
- A tool refactor tranche must replace multiple one-off scripts with a family
  command or manifest-driven entry point.
- Benchmark/comparison references to other codes are allowed only in
  `benchmarks/`, `tools/comparison/`, `tests/tools/comparison/`, benchmark docs,
  and explicitly labeled comparison figures/tables. Current naming blockers to
  classify are `tools/comparison/run_reference_linear_stress_matrix.py`,
  `tools/comparison/fixtures/etg_ky25_reference.in`, and `tools/comparison/fixtures/etg_runtime_ky15_reference.in`.

Authoritative domain consolidation:

| Domain | Keep | Merge/delete/move |
| --- | --- | --- |
| `benchmarks/` root | runnable benchmark drivers and compact TOML inputs | raw outputs, branch-policy implementation, duplicate test wrappers |
| `spectraxgk.benchmarks` | small public facade and stable result contracts | case-family branch histories, long policy ladders, comparison-code details |
| `tools/artifacts` | one artifact-driver family plus reusable plot/table helpers | one-file-per-panel builders not referenced by docs or release manifests |
| `tools/campaigns` | active long-run campaign launch/postprocess commands | stale fallback launchers, completed probes, comparison-code launchers |
| `tests/tools` | one file per tool family contract | one file per command parser when behavior is table-driven |
| `tests/validation` | physics gates, benchmark contracts, and accepted tolerance ledgers | source-topology mirrors and monkeypatch forests for deleted legacy paths |
| `geometry` | all Miller/VMEC/Boozer providers and differentiability checks | separate `geometry_backends` namespace after imports are migrated |
| `operators` | linear/nonlinear RHS kernels, fields, brackets, gyroaverages, collisions, dissipation | duplicate `terms` ownership unless kept as a documented mathematical API |
| `objectives` | differentiable objective families for linear, QL, nonlinear-window, VMEC/Boozer, zonal | tiny `vmec_*` and portfolio shards that only wrap another family owner |

Refactor order that minimizes risk:

1. **Benchmark facade shrink:** keep only stable API/result contracts in
   `spectraxgk.benchmarks`; move case implementation policies into root
   benchmark drivers or delete unsupported branch history. This targets the
   13,211-line file without adding a larger source package forest.
2. **Test tree collapse:** introduce shared fake-runner fixtures and convert the
   largest runtime/benchmark/tool tests into parametrized family contracts.
3. **Tool command consolidation:** create manifest-driven artifact/campaign
   command families, then delete single-panel or stale fallback scripts that are
   not referenced by current docs/readme/release manifests.
4. **Geometry namespace merge:** move `geometry_backends` implementation under
   `geometry`, preserving geometry parity and AD/FD gates.
5. **Operator ownership cleanup:** remove dual `terms`/`operators` ambiguity by
   choosing one public mathematical-kernel namespace and updating docs/tests.
6. **Objective family merge:** consolidate VMEC/Boozer, line-search, FD,
   gradient-gate, and portfolio wrappers into fewer objective-family modules.
7. **Performance pass:** only after ownership cleanup, profile quickstart,
   linear scan setup/RHS, nonlinear RHS/bracket/field solve, diagnostics IO, and
   VMEC/Boozer transforms; each speed claim needs before/after timing, memory
   where practical, and numerical/physics identity gates.

### Repository Role Model

The repository should have one obvious destination for every retained file. If a
file does not fit one of these roles, it does not belong on `main`.

| Location | Role | Keep examples | Remove or move |
| --- | --- | --- | --- |
| `src/spectraxgk` | installed user and developer API | kernels, solvers, geometry providers, diagnostics, objectives, executable workflows | benchmark-case branch logic, manuscript campaigns, raw comparison scripts, one-off artifact builders |
| `examples` | small runnable user tutorials | default run, plotting, geometry input, linear/QL/nonlinear optimization examples | long campaigns, parameter sweeps, unreleased research probes |
| `benchmarks` | reproducible benchmark drivers | compact scripts that run a documented physics/performance benchmark from inputs | duplicate test wrappers, generated outputs, hidden policy code imported by the package |
| `tools` | maintainer commands | artifact builders, release checks, campaign launchers, profiling, external-code comparison utilities | single-panel scripts that can be a manifest entry, stale fallback launchers, scratch dumps |
| `tests` | automated correctness gates | table-driven unit/integration/validation/tool/release contracts | one-file-per-helper mirrors, monkeypatch forests, redundant branch tests |
| `docs/_static` | curated evidence for current docs/readme claims | compressed figures, compact JSON summaries, CSV tables referenced by docs or release manifests | stale pilot traces, negative scratch probes, unreferenced historical panels |

This role model resolves the `benchmarks` versus `tools` confusion:

- `benchmarks/` answers "what should a researcher run to reproduce a benchmark?"
- `tools/` answers "what should a maintainer run to build evidence, profile, or release?"
- `tests/` answers "what must CI run automatically and quickly?"
- `examples/` answers "what should a new user copy and modify?"

### Deletion And Quarantine Policy

Every consolidation tranche must run this triage before adding or moving files:

1. **Delete from `main`** if the file is unreferenced, obsolete, a completed
   pilot/probe, a stale reduced-window artifact, or a compatibility shim for an
   unsupported workflow.
2. **Move to a draft experiment PR** if the file is scientifically plausible but
   not validated, not documented, or too expensive/noisy for release gates.
3. **Fold into an existing owner** if the file only forwards to another module,
   contains one case variant, or differs only by constants/manifest data.
4. **Keep as a separate file** only when it owns a stable public concept,
   a performance-critical JIT boundary, or a documented testable physics model.

No new source package file is allowed unless it removes at least one old source
file or isolates a JIT/performance boundary that is measured in the profiler
manifest. No new test file is allowed unless it replaces larger duplicated tests
or introduces a genuinely new gate family.

### Complexity Reduction Targets

These targets are intentionally aggressive but realistic:

| Area | Current audit | Release target | How to get there |
| --- | ---: | ---: | --- |
| Installed source files | 275 | <= 150 near term, <= 100 final | delete `validation`, merge tiny geometry/objective/operator shards, remove legacy facades |
| Installed source LOC | 101k | <= 70k near term, <= 50k final | fold branch-specific benchmark code, remove compatibility paths, prefer data tables over code branches |
| Test files | 243 | <= 150 near term, <= 100 final | table-driven fixtures, one file per contract family, merge repeated artifact/comparison tests |
| Test LOC | 96k | <= 60k near term, <= 40k final | replace monkeypatch forests with reusable fake runners and parametrized contracts |
| Tool scripts | 247 | <= 150 near term, <= 100 final | manifest-driven artifact/campaign builders, merge one-panel status scripts |
| Docs static files | 1605 | reference-graph curated | keep only docs/readme/release-manifest referenced evidence |

The immediate milestone is not a cosmetic move. It is a measurable shrink:
delete the installable benchmark-validation package, reduce the largest runtime
and benchmark test files, and convert one-off artifact builders to
manifest-driven commands.

### Obsolete And Experimental-Code Audit Rules

Use these searches before each release candidate and record the result in this
plan:

```bash
git ls-files | rg -i '(^|/)(scratch|tmp|temp|probe|experimental|draft|old|legacy|deprecated|debug|dump|transient|prototype|smoke|scaffold|synthetic)'
git ls-files | rg -v '(^benchmarks/|^tools/comparison/|^tests/tools/comparison/|^docs/.*(benchmark|comparison)|^README.md|^plan.md)' | xargs rg -n '\bGX\b|gx_|_gx|Gx'
python tools/release/audit_repository_size.py --json-out docs/_static/repository_size_audit.json
python tools/release/check_release_artifact_manifest.py
```

Allowed outcomes:

- comparison-code names may remain only in explicit benchmark/comparison context;
- "legacy", "probe", "dump", "synthetic", and "smoke" names are blockers on
  `main` unless a release gate explicitly justifies them;
- local ignored output such as `tools_out`, `docs/_build`, caches, and
  `__pycache__` must stay untracked and may be deleted locally before release.

### Performance Bottleneck Ownership

Performance work should happen after ownership cleanup so measurements point to
one responsible module. The current bottleneck map is:

| Path | Likely bottleneck | Owner after refactor | Required gate before claim |
| --- | --- | --- | --- |
| Default executable run | compile latency, progress/reporting, setup overhead | `workflows` + `solvers` | wall-time, progress output, same linear fit |
| Linear scans | repeated geometry/cache setup, batched RHS/eigensolve throughput | `operators/linear`, `solvers/linear`, `parallel` | serial-vs-batched identity and scan parity |
| Nonlinear RHS | spectral bracket, field solve, diagnostics materialization | `operators/nonlinear`, `solvers/nonlinear` | serial-vs-decomposed identity on transport window |
| VMEC/Boozer geometry | interpolation and coordinate transforms | `geometry` | geometry parity plus AD/FD gradient gate |
| Output/plotting | NetCDF and large diagnostic writes | `artifacts`, `workflows` | artifact schema identity and reduced IO timing |

Every runtime improvement must include: equivalent workload, before/after wall
time, memory when practical, numerical identity or physics metric, and a profiler
artifact if it supports a public speed claim.

### 2026-07-07 Final Topology Audit And Refactor Direction

This audit supersedes older count snapshots below where they conflict.

Current tracked state after the latest consolidation:

- Branches: only `main` and `origin/main`. There are no obsolete local or remote
  branches to prune in this checkout. Future experiments should live in one
  draft PR until the full refactor plan is complete, not in multiple active
  branches.
- Tracked files: 2,510. Tracked generated Python cache files: 0. Tracked NetCDF
  files: 0. The local large files visible in simple filesystem scans are ignored
  cache/build/output directories, not tracked release content.
- Tracked docs/static evidence: 1,572 files and about 34.7 MiB. This is useful
  publication/validation evidence, but it must be pruned by reference graph:
  keep only README/docs/release-manifest referenced figures, compact JSON
  summaries, and benchmark evidence that backs a current claim.
- Root `benchmarks/`: 12 Python files and about 1.6k LOC. This is already a
  clear reproducibility layer and should stay at the repository root. Do not
  merge it into `tools/` or `examples/`.
- Installed package: 277 Python files and about 100.4k LOC. The main blockers
  are now the oversized benchmark facade, `objectives` (41 files),
  `geometry_backends` (18 files), dual `terms`/`operators` ownership, and
  large runtime/artifact facades.
- Tests: 243 Python files and about 96.7k LOC. The blocker is not coverage; it
  is one-file-per-tool and monkeypatch-heavy historical branch testing. The
  largest targets are runtime integration, benchmark branch tests, artifact
  tool tests, and comparison-tool tests.
- Tools: 247 Python files and about 100.7k LOC. Tool folders are conceptually
  right (`artifacts`, `campaigns`, `comparison`, `profiling`, `release`), but
  each still has too many single-panel, single-campaign, or single-status entry
  points.

External design anchors for the refactor:

- JAX kernels must stay pure-function-first: all differentiable state passes in
  arguments and returns values; side effects belong in CLI, tool, and artifact
  layers. See the JAX sharp-bits guidance on pure functions:
  https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
- Avoid shape-changing branches inside hot JIT paths. Static configuration,
  pytrees, and fixed-shape batch policies should own branch selection before a
  compiled kernel is called. See JAX control-flow and pytree guidance:
  https://docs.jax.dev/en/latest/control-flow.html and
  https://docs.jax.dev/en/latest/pytrees.html
- Compile latency is a real user-facing performance cost. The CLI can use
  non-differentiable progress/reporting paths, persistent compilation cache, and
  ahead-of-time lowering where useful; Python APIs should expose differentiable
  kernels and objective functions without printing or writing files. See:
  https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- Developer workflows should be task-family based, not script-count based. The
  Scientific Python development guide recommends explicit task runners for
  recurring docs/tests/release tasks:
  https://learn.scientific-python.org/development/guides/tasks/

Final package target layout:

| Domain | Target role | File-count target | Consolidation rule |
| --- | --- | ---: | --- |
| `core` | grids, species, normalizations, typed state | 6-8 | small stable dataclasses and shape contracts only |
| `geometry` | Miller, VMEC, Boozer, field-line and differentiable geometry providers | 12-16 | absorb `geometry_backends`; keep provider facades, merge tiny backend shards |
| `operators` | linear/nonlinear RHS kernels, fields, brackets, gyroaverages, dissipation | 14-18 | either absorb `terms` or make `terms` a documented mathematical API, not both |
| `solvers` | time stepping, Krylov/eigen, explicit/IMEX policies | 14-18 | keep solver policy separate from physics kernels; no benchmark branch logic |
| `diagnostics` | growth/frequency, transport windows, QL calibration, validation gates | 14-18 | reusable metrics only; campaign-specific ledgers move to tools/docs artifacts |
| `objectives` | differentiable linear, QL, nonlinear-window, VMEC/Boozer, zonal objectives | 12-18 | merge tiny `vmec_*`, `qa_*`, and portfolio shards by objective family |
| `parallel` | independent-work batching and validated decomposition kernels | 5-8 | only identity-gated production paths, no experimental speedup wrappers |
| `workflows` | runtime orchestration, TOML, plotting, restart, public executable flow | 8-12 | non-differentiable side effects stay here, not in kernels/objectives |
| `artifacts` | NetCDF, plotting, restart, documented output contracts | 8-12 | merge by artifact type; no one-file-per-output-section helpers |
| `benchmarks` facade | documented public benchmark API only | 1-3 | no installable `validation.benchmarks` package behind it |

Final test layout:

| Test domain | Target | Consolidation rule |
| --- | ---: | --- |
| `tests/unit` | 35-45 files | one file per stable domain contract, not per helper module |
| `tests/integration/runtime` | 5-7 files | config, progress, execution, artifacts, restart, plotting |
| `tests/validation` | 15-20 files | physics gates and benchmark contracts; no mirror of source package topology |
| `tests/tools` | 10-15 files | artifact, campaign, comparison, profiling, release family tests |
| `tests/release` | 5-8 files | architecture, coverage, docs/static, package build, release guardrails |

Final tool layout:

| Tool family | Target form | Keep/delete rule |
| --- | --- | --- |
| `tools/artifacts` | one artifact driver plus family-specific plot/table modules | keep only builders referenced by README/docs/release manifests |
| `tools/campaigns` | manifest writers and campaign launchers for active documented campaigns | delete or move stale probes, fallback launchers, and old campaign scaffolds |
| `tools/comparison` | explicit external-code comparison utilities | comparison-code names are allowed here only |
| `tools/profiling` | profilers tied to performance manifest metrics | no speed claim without profiler artifact and identity/physics gate |
| `tools/release` | CI/release guard commands | keep fail-closed gates, merge repeated JSON/status checks |

Performance bottlenecks to address only after ownership cleanup:

1. Quickstart compile latency and default-run progress output.
2. Linear cache construction and linear RHS repeated setup.
3. Nonlinear bracket/RHS kernels and field solves.
4. Diagnostic materialization and NetCDF/restart IO.
5. VMEC/Boozer conversion and geometry interpolation.
6. Nonlinear domain-decomposition communication.

Each performance claim must include an equivalent-workload before/after timing,
memory measurement where practical, and a numerical-identity or physics gate.

Execution order for the remaining refactor:

1. Finish benchmark-validation exit: move public benchmark runners behind
   `spectraxgk.benchmarks`, move internal case workflows to root `benchmarks`
   or tests, delete `src/spectraxgk/validation`.
2. Contract benchmark and runtime tests: replace monkeypatch-string forests with
   reusable fake runner contracts, keeping the same physics and branch gates.
3. Merge tool script forests by family: start with artifact builders, then
   campaign writers, then release status gates.
4. Prune docs/static artifacts by reference graph: referenced claim evidence
   stays; stale pilot/probe/reduced-window companions are removed or moved to
   release assets.
5. Merge `geometry_backends` into `geometry` and update API/docs/tests in the
   same commit.
6. Resolve `terms` versus `operators`: either rename `terms` to explicit
   `operators` ownership or keep `terms` as a documented mathematical API and
   remove duplicate operator entry points.
7. Collapse `objectives` by objective family, preserving differentiability
   gates and VMEC/Boozer finite-difference checks.
8. Run profiler-backed performance tranches only after the owning code path is
   stable.

### 2026-07-07 Re-Audit Decisions

The latest inventory pass after the profiler consolidation found 2,526 tracked
files. No tracked byte-size problem remains: the largest tracked file is below
1 MiB, and generated caches/build trees are ignored. The remaining problem is
maintainer complexity, not clone size.

| Area | Current evidence | Decision |
| --- | --- | --- |
| Branches | only `main` and `origin/main` | no branch cleanup needed; keep future experimental work in draft PRs, not `main` |
| `benchmarks/` | 18 tracked files, 12 Python files, about 1k lines | keep as the root reproducibility layer; do not merge with `tools/` or `examples/` |
| `tools/` | 247 Python files, 19 profilers, 122 artifact builders | keep family folders but merge script forests into manifest-driven entry points |
| `tests/` | 243 Python files; largest runtime and benchmark tests are 2k-4k lines | reduce by parametrizing repeated branch/tool tests, not by splitting monoliths into more files |
| `src/spectraxgk/validation` | 13 files, all benchmark-related | close this package next by moving benchmark policy to a clearer benchmark-cases owner or root benchmark drivers |
| comparison-code names | comparison tooling is correctly explicit, but some unit/runtime tests still use comparison-code terminology for numerical conventions | rename to physical/numerical names outside explicit comparison/benchmark context |

The next work should avoid cosmetic moves. A file move is allowed only when it
also removes a duplicated entry point, shrinks an installable validation surface,
or clarifies a public user workflow.

### Next Consolidation Tranches

1. **Benchmark validation exit from `src/spectraxgk/validation`.**
   - Keep `spectraxgk.benchmarks` as the public facade for documented benchmark
     helpers.
   - Rename implementation ownership away from `validation.benchmarks` because
     these modules are benchmark-case workflows, not runtime validation logic.
   - Collapse shared request, fit, scan, reference-data, and branch-selection
     code before moving files; do not create a new 16-file package with a
     different name.
   - Gate with `tests/validation/benchmarks`, public API import tests, CLI
     quickstart tests, and a comparison-terminology scan.

2. **Benchmark and runtime test contraction.**
   - Convert repeated monkeypatch-heavy branch tests into table-driven cases
     with reusable fake runners and fake field/fit outputs.
   - Target the largest files first:
     `tests/integration/runtime/test_runtime_runner.py`,
     `tests/validation/benchmarks/test_benchmarks_runner_branches.py`,
     `tests/integration/runtime/test_runtime_helpers.py`, and
     `tests/integration/runtime/test_cli.py`.
   - Preserve coverage by testing contracts: configuration normalization,
     progress reporting, geometry input, initialization, restart, artifact
     writing, branch selection, and plotting.

3. **Tool script forest contraction.**
   - `tools/artifacts`: merge builders that only differ by labels, manifests, or
     input paths into family drivers with a TOML/JSON manifest.
   - `tools/campaigns`: keep only active long-run launch/postprocess scripts;
     delete stale probes and move future experiments to draft PRs.
   - `tools/comparison`: keep explicit comparison-code names here only; do not
     import comparison terminology into runtime package names.
   - `tools/profiling`: keep profiler entry points only when tied to a manifest
     metric and artifact path.

4. **Source-domain contraction after validation exits.**
   - Merge `geometry_backends` into `geometry` around provider/back-end
     contracts.
   - Fold `terms` into `operators` unless a term module is a documented
     mathematical API.
   - Shrink root/public API facades to documented imports only.
   - Consolidate NetCDF/TOML/plot/restart helpers by artifact contract, not by
     historical output section.

5. **Performance work tied to refactor ownership.**
   - Profile the hot owner before changing it; no new runtime claim without a
     before/after artifact and identity or physics gate.
   - Current hot owners are linear cache construction, nonlinear bracket/RHS,
     field solves, diagnostic materialization, VMEC/Boozer conversion, restart
     IO, and nonlinear domain-decomposition communication.

### Consolidation Priorities

1. **Finish validation leaving the package.**
   - Keep `validation.nonlinear_gradient`, `validation.nonlinear_transport`,
     and `validation.stellarator` closed; do not recreate installable campaign
     subpackages for those lanes.
   - Replace `validation.benchmarks` with a small public benchmark facade plus
     root `benchmarks/` drivers and validation tests.

2. **Collapse tests by contract, not by script.**
   - Replace one-tool-one-test files with parametrized family tests for artifact
     builders, campaign writers, comparison tools, release gates, profiling
     tools, runtime behavior, and benchmark policies.
   - Split large monoliths only when the split creates stable contracts:
     configuration, progress/live output, execution, restart, artifacts, plots,
     benchmark fitting, branch selection, scans, and external comparisons.
   - Delete tests that only preserve retired compatibility behavior.

3. **Collapse tools by capability.**
   - Keep five tool families only: `release`, `artifacts`, `profiling`,
     `comparison`, and `campaigns`.
   - Within each family, merge case-specific builders into manifest-driven
     entry points. A script that only changes labels, paths, or one figure name
     should not remain a separate maintained executable.
   - Move inactive campaign launchers, probe builders, and status reporters to a
     draft PR or delete them.

4. **Clarify examples and benchmarks.**
   - `examples/` teaches promoted user workflows and should stay small, tested,
     and runnable.
   - `benchmarks/` contains small reproducible benchmark drivers and manifests.
   - Long campaigns and generated outputs never belong in either directory.

5. **Shrink package domains after validation exits.**
   - Merge `geometry_backends` into `geometry`.
   - Merge `terms` into `operators` unless a term is a user-facing mathematical
     API.
   - Reduce root/API facades to documented public imports only.
   - Consolidate artifact/IO/runtime helpers around NetCDF, TOML, plotting, and
     restart contracts instead of one file per output section.

6. **Performance refactor follows ownership cleanup.**
   - Keep hot JAX kernels stable and close to their domain owner.
   - Profile before/after any speed claim. The active bottleneck list is: quick
     start compile latency, linear cache/RHS, nonlinear field solve/bracket/RHS,
     diagnostic materialization, VMEC/Boozer conversion, restart/artifact IO,
     and nonlinear domain-decomposition communication.
   - Add speedups only with numerical-identity gates, physics gates, timing,
     memory, and profiler artifacts for equivalent workloads.

### Finite Execution Order

| Step | Target reduction | Gate before commit |
| --- | --- | --- |
| 1 | Completed: nonlinear-gradient extraction | focused nonlinear-gradient tests, manifests, import smoke, `git diff --check` |
| 2 | Completed: `validation.nonlinear_transport` extraction | optimization guard/report tests, release guard tests, stale import scan |
| 3 | Completed: `validation.stellarator` extraction | stellarator validation tests, VMEC/Boozer artifact tests, docs import build check |
| 4 | Replace `validation.benchmarks` internals with root benchmark drivers and package facade | benchmark validation tests, public facade import tests, comparison terminology scan |
| 5 | Merge tool artifact/status builders by manifest family | tool-family tests, docs/static reference audit |
| 6 | Reorganize large tests into contract families | fast test shard under 5 minutes, wide coverage >=95% |
| 7 | Collapse source domains (`terms`, `geometry_backends`, root facades, IO/artifacts) | unit/integration/validation shards, API docs build |
| 8 | Run profiler-backed performance tranches | before/after profiler artifacts and identity/physics gates |
| 9 | Final docs/readme/release pass | release checks, package build, docs build, CI green |

### Benchmark, Tool, And Test Consolidation Design

This is the detailed implementation plan for the remaining topology blocker.
The intent is not to keep moving files around; the intent is to remove
unpromoted code paths, preserve the small public API, and make each directory
mean one thing.

Directory roles:

- `src/spectraxgk`: installed runtime, numerics, diagnostics, objectives,
  plotting, and the small public benchmark facade. It must not contain campaign
  launchers, historical branch ladders, one-off comparison scripts, or long-run
  policy matrices.
- `benchmarks`: root-level reproducible benchmark drivers and manifests that a
  developer or reviewer can run from a clone. These are not imported by the
  runtime package and do not contain raw outputs.
- `tools`: developer/release executables only. A tool either builds documented
  artifacts, launches an explicit campaign, profiles a workload, compares
  against an external code, or checks release gates. One-off status/probe
  scripts are deleted or moved to a draft experiment branch.
- `tests`: assertions only. Tests should be grouped by contract and parametrized;
  they should not mirror every tool filename or preserve retired private helper
  behavior.
- `examples`: promoted user workflows only. Long campaigns, reduced scaffolds,
  and unpublished probes are removed from `main` or moved to root benchmark/tool
  workflows.

Benchmark extraction plan:

1. **Freeze the public benchmark API.** Keep `spectraxgk.benchmarks` as the only
   supported installed benchmark import. The stable API is result containers,
   reference loaders, promoted `run_*` benchmark runners, and documented config
   classes. Private underscored helpers currently re-exported through
   `spectraxgk.benchmarks` are not a compatibility target unless a public doc or
   example uses them.
2. **Completed: move case config dataclasses out of validation.** `config.py`
   now owns the Cyclone, ETG, kinetic-electron, KBM, and TEM benchmark presets
   directly, and `spectraxgk.benchmarks` re-exports them from that physical
   owner.
3. **Move reusable diagnostics out of benchmarks.** Time-series loading,
   late-window fitting, eigenfunction comparison, observed-order checks,
   branch-continuity metrics, heat-flux convergence, nonlinear-window metrics,
   and zonal-response metrics now live in `diagnostics`. Public wrappers can
   remain in `spectraxgk.benchmarks` only where documented.
4. **Collapse branch-history modules.** Case-specific branch modules such as
   `*_paths`, `*_branches`, `*_explicit`, `*_krylov`, and seed-selection helpers
   are either folded into one parametrized root benchmark driver or removed if
   they only preserve historical private branches. Tests should assert the
   supported branch policy, not patch dozens of private hooks.
5. **Keep package runners small.** Promoted installed runners should call the
   shared solver/config/diagnostic APIs directly. If a runner needs a long
   campaign matrix, that matrix belongs in root `benchmarks` or `tools/campaigns`
   and the package runner should expose only a simple documented workload.
6. **Remove `src/spectraxgk/validation/benchmarks`.** After the facade imports
   no validation modules, delete the package and update API docs, architecture
   docs, release manifests, and stale import scans in the same commit.

Next benchmark-validation tranches, in order:

1. **Shared helper ownership.** Move or delete the small helper modules that are
   not benchmark cases. Completed in this class: fit-signal selection now lives
   in `diagnostics/growth_rates.py`. Completed in this class: scan batching now lives in
   `validation/benchmarks/defaults.py`. Completed in this class: solver-selection policy now lives in
   `validation/benchmarks/defaults.py`. Completed in this class: reference containers and CSV loaders now live in
   `validation/benchmarks/defaults.py`. Completed in this class: benchmark species/parameter policy now lives in
   `validation/benchmarks/defaults.py`. Completed in this class: benchmark initialization helpers now live in
   `validation/benchmarks/defaults.py`. Completed in this class: benchmark scan/mode orchestration now lives in
   `spectraxgk.benchmarks`. Remaining helper ownership is now concentrated in `defaults.py`. Reusable
   physics or numerics helpers go to `diagnostics`, `workflows`, or `config`; historical
   benchmark-only policy moves to root `benchmarks/` or the tests that assert
   it.
2. **Branch/path module collapse.** Fold `*_paths.py`, `*_branches.py`,
   `*_explicit.py`, `*_krylov.py`, and seed-selection modules into the case
   runner or root benchmark driver only when the branch remains promoted. Delete
   branches that only preserve old private patch seams.
3. **Case-runner migration.** Move promoted Cyclone, ETG, KBM, TEM, and kinetic
   benchmark runners behind `spectraxgk.benchmarks` or root `benchmarks/`
   drivers with documented configuration objects. The installed package should
   not expose internal validation package paths.
4. **Benchmark-test consolidation.** Replace the 2979-line
   `tests/validation/benchmarks/test_benchmarks_runner_branches.py` with
   parametrized contract tests for setup normalization, fit-signal selection,
   branch policy, scan-window policy, and reference loading. Keep physics gates;
   drop tests that only assert retired private function names.
5. **Documentation cleanup.** Remove direct API documentation for
   `spectraxgk.validation.benchmarks.*` modules except the temporary public
   facade during migration. Benchmark documentation should point users to
   `spectraxgk.benchmarks` and root `benchmarks/`.

Benchmark tranche gates:

- `pytest -q tests/validation/benchmarks tests/unit/diagnostics tests/integration/runtime/test_cli.py --maxfail=1`
- `python tools/release/check_validation_coverage_manifest.py --out-json docs/_static/validation_coverage_manifest_summary.json`
- `python tools/release/check_differentiable_refactor_manifest.py --out-json /tmp/spectrax_diff_benchmark_snapshot.json`
- `python tools/release/check_package_architecture_manifest.py --out-json /tmp/spectrax_arch_benchmark_snapshot.json`
- stale import scan for `spectraxgk.validation.benchmarks`
- public import smoke for `spectraxgk.benchmarks` and `spectraxgk.api.benchmarks`

Test consolidation plan:

1. Split `tests/integration/runtime/test_runtime_runner.py` by runtime contract:
   config parsing, progress/live output, linear execution, nonlinear execution,
   restart, artifacts, and plotting. Delete assertions for legacy behavior that
   the current executable no longer supports.
2. Replace `tests/validation/benchmarks/test_benchmarks_runner_branches.py`
   with parametrized setup, fit, branch-policy, scan-policy, and reference-loader
   tests. Patching private functions is allowed only for true dependency
   isolation; it should not define the API.
3. Merge one-wrapper artifact tests by family: benchmark/runtime, nonlinear
   transport, quasilinear, VMEC/Boozer, W7-X/zonal, release status/readiness, and
   generic plotting.
4. Keep coverage physics-driven: late-window growth/frequency, observed-order
   convergence, eigenfunction phase alignment, nonlinear transport windows,
   quasilinear calibration/screening, differentiable geometry gradients, and
   serial-vs-parallel identity gates remain explicit.

Tool consolidation plan:

1. Reduce `tools/artifacts` from one-script-per-panel to manifest-driven
   builders. Target families are benchmark atlas/runtime, quasilinear model
   development, nonlinear transport, VMEC/Boozer optimization, W7-X/zonal,
   readiness/status, and generic plotting.
2. Keep comparison-code scripts only in `tools/comparison` and only when they
   are actively used for benchmark comparisons. No comparison-code terminology
   should appear in core source names except in explicit benchmark/comparison
   context.
3. Keep profiling tools only if they produce profiler-backed timing/memory
   artifacts for an equivalent workload. Delete stale profiling wrappers that
   duplicate current profiling entry points.
4. Every retained tool gets one test owner or one documented manual/office gate.

Performance simplification plan:

1. Do not optimize around validation scaffolding. Finish ownership cleanup first
   so profiles measure runtime kernels, not campaign/report overhead.
2. Profile these hot paths with CPU and GPU separately before claiming speedups:
   JIT/startup latency, linear cache/RHS, nonlinear field solve and Poisson
   bracket, diagnostics materialization, NetCDF/restart IO, VMEC/Boozer
   conversion, and nonlinear decomposition communication.
3. Prefer simplifications that reduce allocations and Python dispatch: fewer
   transient pytrees, cached geometry/operator data, fused nonlinear bracket
   paths, optional diagnostic thinning, and bounded live-output formatting.
4. Require numerical-identity and physics gates before any speed claim:
   serial-vs-parallel identity, conserved/diagnostic window agreement, runtime,
   memory, and profiler artifact for the same workload.

### Success Metrics

- `src/spectraxgk`: <=100 Python files, with `spectraxgk.validation` removed or
  reduced to a tiny public metric facade.
- `tests`: <=100 Python files, organized by domain contracts rather than tool
  wrappers, with package-wide coverage >=95%.
- `tools`: <=100 Python files, with no undocumented one-off probe/status/build
  scripts.
- `benchmarks`: root-level, small, documented, no raw outputs.
- `examples`: promoted, runnable, documented workflows only.
- Comparison-code terminology appears only in explicit benchmark/comparison
  contexts.
- README/docs explain where new physics, diagnostics, objectives, examples,
  tests, benchmarks, and tools belong.


## Current Audited State

Last audited: 2026-07-07 on `main`.

- Audited baseline head for the current topology plan:
  `b503c0d4 Move quasilinear model selection into diagnostics`.
- Latest reachable release tag at the audit: `v1.6.10`; the audited baseline
  was three commits after that tag.
- Git state at audit: clean `main`, tracking `origin/main`.
- Latest GitHub release workflow and PyPI publish for `v1.6.10` passed. CI for
  the post-release refactor commits must be rechecked before the next tag; the
  `b503c0d4` CI run was in progress at the latest audit.
- Active local/remote branches: only `main` and `origin/main`.
- Stale detached worktree metadata for old local investigations was pruned on
  2026-07-07; the only remaining worktree is this `main` checkout.
- Closed obsolete experimental PRs remain closed: #4, #5, and #6.
- Tracked repository size is acceptable for now; no tracked file is above 2 MB.
  The largest tracked file is `docs/_static/qa_low_turbulence_comparison.json`
  at about 0.94 MiB.
- Current topology counts:
  - `src/spectraxgk`: 275 Python files after extracting nonlinear-gradient, nonlinear-transport, stellarator validation subpackages, benchmark case presets, benchmark eigenfunction diagnostics, benchmark time-series/window diagnostics, benchmark zonal-response metrics, benchmark trace/window metrics, benchmark fit-signal helpers, benchmark scan-batching helpers, benchmark solver-policy helpers, benchmark reference loaders, benchmark species policies, benchmark initialization helpers, benchmark scan/mode orchestration, benchmark scan-window policy, the secondary slab workflow, benchmark scan/eigenfunction harness helpers, the KBM fixed-beta ky-scan wrapper, and the kinetic-electron, ETG, KBM, TEM, and Cyclone benchmark runners, then deleting `src/spectraxgk/validation`.
  - `tests`: 243 Python files, including the shared `tests/support/paths.py`
    helper; only `conftest.py` remains at the flat `tests/` root.
  - `tools`: 247 Python files after purpose-folder moves, nonlinear-transport follow-up relocation, deletion of obsolete unreferenced tool scripts, and consolidation of the device-z RHS profiler into the transport-window profiler.
  - `examples`: 42 Python files after retiring the cETG example.
  - `benchmarks`: 18 tracked files, 12 Python files, about 1k lines.
- The repository inventory now leaves no installable validation files. The broader test/tool/doc-artifact
  inventory remains in the `keep-or-merge` and `keep-and-consolidate` queues,
  and the new benchmark priority is to split the oversized public facade into
  cleaner benchmark-data, fit-policy, and case-runner owners without reviving
  `spectraxgk.validation`.
- Source-package Python file counts by domain:
  - `validation`: 0 files; package removed.
  - `objectives`: 41 files.
  - `operators`: 34 files.
  - `solvers`: 34 files.
  - `geometry`: 25 files.
  - `workflows`: 24 files.
  - `terms`: 16 files.
  - `artifacts`: 18 files.
  - `geometry_backends`: 18 files.
  - `diagnostics`: 25 files.
  - `parallel`: 12 files.
  - `api`: 11 files.
  - root facades: 9 files.
  - `core`: 6 files.
  - `utils`: 2 files.
- Source-function size is no longer the main problem: there are currently no
  source functions at or above 90 lines and none in the 80-89 line band.
- No obsolete active branches are present in this clone: only `main` and
  `origin/main` exist.
- No tracked `__pycache__`, `.pyc`, `.DS_Store`, or file above 1 MB is present.
  Large local files are generated outputs under ignored locations such as
  `tools_out`, `.venv`, `docs/_build`, and `.mypy_cache`.
- The main maintainability problem is repository topology: validation, campaign,
  plotting, profiling, release, comparison, and debug code has grown to roughly
  the same size as the solver and is spread across flat `tests/` and `tools/`
  namespaces.
- The architecture manifest now treats `spectraxgk.validation` as a temporary
  facade, not a permanent family of required installable validation packages.

Latest focused audit for this tranche:

- Flat topology is no longer the blocker: `tests/` has zero flat `test_*.py`
  files, and `tools/` has zero flat scripts except `tools/__init__.py`.
- The remaining code-size problem is family sprawl:
  - `tests/tools/artifacts`: 26 artifact-family tests after the linear-validation, parallel-identity, VMEC/Boozer aggregate, VMEC/Boozer report, quasilinear plotting, W7-X/zonal panel, nonlinear report, status/readiness, and VMEC miscellaneous consolidations.
  - `tools/artifacts`: 122 figure/table/status/gate builders after deleting unreferenced nonlinear-parallel/compression scripts.
- `src/spectraxgk/validation`: 25 installable benchmark-validation files.
  - `tests/integration/runtime/test_runtime_runner.py`: about 4.2k lines,
    mostly preserving historical runtime branches in one file.
  - `tests/validation/benchmarks/test_benchmarks_runner_branches.py`: about
    3.0k lines, mostly patching case-specific benchmark branches.
- `benchmarks/` itself is not bloated. It is small and should stay at the
  repository root as the lightweight benchmark-driver layer. The confusing part
  is that benchmark harness and validation campaign code still lives under
  `src/spectraxgk/validation`.
- No stale local branches need pruning in this checkout. Obsolete experimental
  work should be removed from `main` by moving code to a draft experiment PR or
  deleting it, not by keeping dead branches around.
- The next 10x simplification cannot come from more micro-splitting. It must
  come from deleting non-promoted paths, merging parametrizable tools/tests, and
  moving campaign validation out of the installable package.

## 2026-07-07 Planning Reset: What Actually Needs To Change

This section supersedes older migration notes whenever there is a conflict. The
current tree no longer has a flat-test or flat-tool problem. It has a product
boundary problem: package code, validation campaigns, artifact builders,
profilers, benchmark drivers, and historical branch tests are all treated as if
they were equally central to the runtime library.

Current measured topology after extracting nonlinear-gradient,
nonlinear-transport, quasilinear, stellarator validation families, benchmark
case presets, benchmark eigenfunction diagnostics, and benchmark time-series/
window diagnostics, and benchmark zonal-response metrics:

| Area | Files / lines | Main issue |
| --- | ---: | --- |
| `src/spectraxgk` | 275 Python files, about 100.2k LOC | oversized benchmark facade plus many public/internal facades |
| `src/spectraxgk/validation` | 0 Python files | removed; do not reintroduce installable validation campaigns |
| `tests` | 243 Python files, about 96.7k LOC | one-file-per-tool suites and historical branch monoliths are hard to maintain |
| `tools` | 247 Python scripts, about 100.7k LOC | many scripts differ by case labels, artifact names, or campaign paths, but obsolete zero-reference scripts are being removed |
| `tools/artifacts` | 122 Python scripts, about 52.5k LOC | figure/status/gate builders should be manifest-driven families, not one script per panel |
| `benchmarks` | 12 Python files, about 1.6k LOC | already small; keep as root-level reproducible benchmark entry points |
| `examples` | 42 Python files, about 6.2k LOC | keep only promoted pedagogical workflows; move long campaigns and reduced scaffolds out |

Branch and PR state:

- There are no obsolete local or remote branches in this clone: only `main` and
  `origin/main` are present.
- Closed experimental PRs should stay closed. If an experiment is still useful,
  port only the specific, validated idea into the main refactor; do not
  resurrect historical branches or keep stale compatibility code in `main`.
- Future exploratory code belongs in draft PRs or external scratch branches
  until it is promoted by tests, docs, and validation artifacts.

Obsolete and experimental-code concentration:

- Current solver kernels are not the main obsolete-code source. Keyword and
  ownership scans put most `pilot`, `probe`, `reduced`, `legacy`, and
  compatibility residue in docs, examples, tests/tools, `tools/artifacts`,
  `tools/campaigns`, and `src/spectraxgk/validation`.
- The first deletion candidates are files that only build historical,
  non-promoted, reduced-window, pilot, or probe artifacts and are not referenced
  by README, docs, release manifests, or current tests.
- Compatibility aliases should be kept only for documented public user imports.
  Test-only compatibility should be removed with the behavior it preserves.

Benchmarks, tools, and examples are now defined as follows:

- `examples/` teaches a user how to run promoted workflows. It should be small,
  runnable, documented, and free of hidden campaign machinery.
- `benchmarks/` contains small, reproducible accuracy/runtime benchmark drivers
  and benchmark manifests. It should not contain raw outputs, docs figures, or
  long campaign launch forests.
- `tools/` is repository machinery: release gates, artifact builders,
  comparison utilities, profiling reproducers, and active long-run campaign
  launch/postprocess scripts. A tool must either be called by CI/release/docs or
  be explicitly listed as an active campaign.
- `src/spectraxgk/` is the installable library and executable. It should not
  contain manuscript campaign policy, plotting scripts, long-run launchers, or
  benchmark-branch history.

The required consolidation path is therefore:

1. **Move campaign validation out of `src` before further solver reshaping.**
   The `spectraxgk.benchmarks` public facade currently imports from
   `spectraxgk.validation.benchmarks`, so this must be staged: define the small
   public benchmark API first, then relocate case-specific branch runners,
   holdout ledgers, and campaign policy to root `benchmarks/`, `tools/`, or
   `tests/validation`.
2. **Merge tools by capability, not by file prefix.** Artifact builders should
   become manifest-driven builders for families such as linear validation,
   nonlinear transport, W7-X/zonal, VMEC/Boozer, quasilinear, release status,
   and performance panels. The target is fewer maintained entry points, not one
   mega-tool with unreadable modes.
3. **Merge tests by physical contract.** The current highest-value test work is
   replacing historical branch tests with parametrized contracts:
   benchmark-fit policy, runtime progress/output policy, nonlinear diagnostics,
   validation gates, artifact schema, and release policy.
4. **Delete non-promoted examples and docs artifacts.** Reduced/synthetic or
   short-window scaffolds stay only if they validate a promoted step and are
   labeled as such. Otherwise they move to a draft experiment branch or are
   deleted from `main`.
5. **Only then consolidate source domains.** After validation exits the package,
   merge adjacent runtime domains where names become simpler: `terms` into
   `operators`, `geometry_backends` into `geometry`, package validation metrics
   into `diagnostics`/`artifacts`, and root facades into a minimal public API.
6. **Use profiler evidence for performance changes.** The known bottlenecks are
   default-demo latency, linear cache construction, linear RHS, nonlinear
   bracket/field solve, diagnostic materialization, VMEC/Boozer conversion, and
   nonlinear decomposition communication. Each speedup claim needs before/after
   profiler artifacts plus numerical-identity or physics gates.

Single active priority queue from the current audit:

1. **Validation out of package.** Do this before more source reshaping. First
   move shared validation metrics and nonlinear-gradient evidence/follow-up
   helpers to `diagnostics`, `objectives`, `tools/campaigns`, or
   `tools/release`; leave only a tiny stable facade if a documented import
   needs it. Then stage the benchmark family after the public
   `spectraxgk.benchmarks` facade is made independent of case-history modules.
2. **Tool families, not script forests.** Consolidate `tools/artifacts` by
   family: VMEC/Boozer, quasilinear, nonlinear transport, W7-X/zonal,
   benchmark/runtime panels, release-status/readiness, and generic plotting
   utilities. Each family should have one manifest-driven entry point plus
   small internal helpers, not one script per figure.
3. **Test families, not one-wrapper tests.** Merge `tests/tools/*` into
   parametrized family tests and split only the two large historical monoliths
   by behavior contract: runtime config/progress/output/execution/restart/plot
   and benchmark setup/fit/branch/scan policy. Do not create more one-case
   files.
4. **Docs/static artifact audit.** Keep images/CSV/JSON only when referenced by
   README, docs, or a release/artifact manifest. Delete stale pilot/probe and
   reduced-window companions before generating new figures.
5. **Source-domain collapse.** After validation moves, merge overlapping
   package domains in this order: `terms` into `operators`,
   `geometry_backends` into `geometry`, runtime/artifact IO into a smaller IO
   surface, broad `api/*` re-exports into thin documented facades, then
   objectives by physical workflow.
6. **Performance pass after topology shrink.** Profile quickstart, linear
   cache/RHS, nonlinear RHS/bracket/field solve, diagnostics streaming,
   VMEC/Boozer in-memory geometry, and parallel execution. Make speedup claims
   only when the physical workload, numerical-identity gate, profiler artifact,
   wall time, and memory evidence all match.

This queue is the active plan. Older sections below are retained as rationale
and progress log; where they conflict with this queue, this queue wins.

Every file should pass this keep test:

1. Does it implement promoted runtime/library functionality?
2. Does it teach a current user workflow?
3. Does it drive a small reproducible benchmark?
4. Does it enforce a CI/release gate?
5. Does it build a README/docs/manuscript artifact that is referenced?
6. Does it compare against an external code or reference in an explicit
   benchmark context?
7. Does it reproduce a profiler result cited by performance docs?
8. Does it launch or postprocess an active documented campaign?
9. Does it test a physics, numerics, API, artifact, or release contract that
   still belongs to the promoted product?

If the answer is no, the file leaves `main`. If the answer is yes but the file
duplicates another file's policy, merge it. If the answer is yes but the file is
installed campaign machinery, move it out of `src`.

### Validation Package Exit Plan

The validation package is the largest installable-code blocker. A fresh
import-aware scan, excluding generated `docs/_build` files, gives this staged
migration map:

| Validation family | Files / LOC | Current role | Target owner |
| --- | ---: | --- | --- |
| `validation.benchmarks` | 24 files, about 15.2k LOC | benchmark case runners, fit policy, branch ladders, and the current implementation behind `spectraxgk.benchmarks` | small public `spectraxgk.benchmarks` facade plus root `benchmarks/` drivers and `tests/validation/benchmarks` policy tests |
| `validation.nonlinear_gradient` | closed | evidence and gate diagnostics consolidated into `diagnostics.nonlinear_gradient_evidence`; follow-up/campaign planning consolidated into `tools/campaigns/nonlinear_gradient_followup.py`; one obsolete Cyclone campaign helper deleted to keep tool count non-regressing | keep closed; do not recreate an installable nonlinear-gradient validation package |
| quasilinear validation family | closed | all reusable diagnostics moved to `diagnostics`; holdout admission moved to release tooling | keep closed; do not recreate an installable quasilinear validation package |
| `validation.stellarator` | closed | candidate gates moved to `objectives.vmec_candidate_admission`; transport admission policy and sample coverage moved to `objectives.vmec_transport_admission`; nonlinear transport report diagnostics moved to `diagnostics.stellarator_transport_reports` | keep closed; do not recreate an installable stellarator validation package |
| `validation.nonlinear_transport` | closed | optimization promotion diagnostics consolidated into `diagnostics.nonlinear_transport_optimization`; replicate-spread diagnostics consolidated into `diagnostics.nonlinear_replicates`; follow-up launch planning moved to `tools/campaigns/nonlinear_replicate_followup.py` | keep closed; do not recreate an installable nonlinear-transport validation package |
| shared validation metrics | 7 files, about 2.3k LOC | autodiff covariance checks, finite-difference helpers, gate reports/types, zonal summaries, external-holdout ledgers | tiny `validation` facade only for stable public metrics, with most reusable math moved to `diagnostics`/`objectives` |

Staged extraction order:

1. **Shared metrics first.** Move `gate_types`, `gate_reports`, `gates`,
   `autodiff`, `autodiff_finite_difference`, and `zonal` into physically named
   `diagnostics.validation`, `diagnostics.zonal`, or `objectives.autodiff`
   owners. Keep a tiny compatibility facade only for documented user imports.
2. **Quasilinear and nonlinear-window metrics.** Move pure window statistics
   and calibration math into `diagnostics.transport_windows` and
   `diagnostics.quasilinear_validation`; move promotion/checker policy to
   `tools/release`.
3. **Nonlinear transport and stellarator campaign policy.** Move campaign
   selection, prelaunch, and follow-up report code to `tools/campaigns` or
   `tests/validation`; keep only promoted objective/diagnostic math in package
   code.
4. **Benchmark family.** Define the stable public `spectraxgk.benchmarks`
   surface, then move case-specific branch histories and long benchmark-policy
   implementations into root `benchmarks/` and validation tests. This is last
   because the public facade currently depends on these modules.
5. **Delete the validation package as a campaign namespace.** When all importers
   use canonical owners, leave at most `src/spectraxgk/validation/__init__.py`
   and a small stable facade, or remove the package entirely in the next major
   API cleanup.

Each family move must update `docs/api.rst`, `docs/code_structure.rst`,
`tools/validation_coverage_manifest.toml`, direct importer tests, and release
checks in the same commit. Do not add new files to `src/spectraxgk/validation`
during this process.

## Hard Targets For The Refactor

These targets are release gates for this refactor. They are intentionally strict
because the goal is not another incremental split, but a much smaller and more
usable codebase.

| Area | Current | Target | Requirement |
| --- | ---: | ---: | --- |
| Installable source Python files | 288 | <= 100 | Move validation/campaign code out of `src`; consolidate domain modules. |
| Test Python files | 243 | < 100 | Reorganize and parametrize tests by domain; merge one-file-per-script tests. |
| Tool Python files | 247 | < 100 | Keep release gates, artifact builders, profilers, and comparison entry points only. |
| Root public facades | 9 | <= 8 | Keep only user-facing facades; no new root prefix modules. |
| `src/spectraxgk/validation` package | 13 | 0-5 | Remove installable validation campaigns; keep only tiny public metric helpers if necessary. |
| Legacy/non-promoted paths | many | 0 promoted by accident | Delete from `main` or move to a draft PR/experiment branch. |
| Default local test runtime | variable | < 5 min | Keep local gates bounded; long physics campaigns stay explicit. |
| Wide package coverage | >= 95% gate | >= 95% | Preserve or improve coverage after consolidation. |
| README/docs claims | scoped | current and scoped | No unsupported speedup, optimization, or validation claims. |

A file-count reduction that merely hides code in larger files is not acceptable.
A successful consolidation must also reduce duplicated policy, redundant tests,
patch-heavy test design, broad public exports, and unclear ownership.

## Refined Refactor Strategy

The next version should be a simpler product, not a compatibility museum. The
repository should keep only promoted solver features, reproducible examples,
release gates, benchmark/comparison tooling, and documentation that matches
those workflows. Everything else should either be deleted from `main` or moved
to a draft PR/experiment branch outside the release path.

The highest-impact reductions are now clear:

| Lane | Current issue | Required action | Expected impact |
| --- | --- | --- | --- |
| Validation in `src` | 69 installable files, many are campaign/report builders | Move benchmark/campaign code to `benchmarks/`, `tools/campaigns`, or `tests/validation`; keep only reusable metrics or public facades | Largest source-file reduction and cleaner runtime imports |
| Tool-family sprawl | 247 scripts, with 122 artifact builders and many case-specific status/check/report tools | Merge by capability with manifest-driven modes; delete unowned probes/debug scripts | Fewer maintenance entry points and clearer release/artifact ownership |
| Test-family sprawl | 246 files, including runtime and benchmark branch monoliths plus one-file-per-tool wrappers | Merge by physical contract and shared fixtures; parametrize tool-family tests | Lower navigation cost without lowering coverage |
| Retired cETG/reduced-model residue | Source implementation is gone, but unsupported-config tests/docs still mention it intentionally | Keep only fail-closed input validation and remove all historical cETG tutorial/research scaffolding | Prevents a deleted model from shaping the new architecture |
| Reduced/synthetic optimization artifacts | Still appear in docs/tests as historical scaffolding | Keep only if they validate a promoted step; otherwise move out of README/docs and then out of main | Prevents confusing claims and reduces examples/tests |
| Comparison-code terminology | Some source/test names use external-code names for physical conventions | Rename to physical/numerical names except explicit benchmark/comparison tools/docs | Cleaner clean-room library surface |
| Validation API breadth | `api/validation.py` exposes many campaign objects as public API | Shrink public validation to stable metrics/gates; move campaign policies to tools/tests | Smaller API and less backward-compatibility burden |
| Benchmarks vs tools vs examples | Roles are still mixed in docs and imports | Root `benchmarks/` owns benchmark drivers; `examples/` owns tutorials; `tools/` owns maintenance/artifacts | New developers can find the right place quickly |

The plan is deliberately deletion-first:

1. Delete or move non-promoted legacy paths before merging tiny modules.
2. Move campaign code out of `src` before reshaping kernels.
3. Collapse tests and tools by family before changing solver internals.
4. Refactor hot kernels only behind profiler and identity gates.
5. Update docs/readme only after the file layout and claims are true.

## Repository Ownership Map

This ownership map is the rule for deciding whether a file stays in `main`,
moves, merges, or is deleted. If a file does not fit one of these roles, it
should not remain in the release branch.

| Location | Owner role | What belongs there | What does not belong there |
| --- | --- | --- | --- |
| `src/spectraxgk/core` | Data contracts | Grids, species, velocity basis, typed contracts, small extension points | Campaign policy, plotting, file-system workflows |
| `src/spectraxgk/geometry` | Geometry API | Analytic, Miller, VMEC/Boozer, differentiable geometry contracts, in-memory JAX adapters | Long-run validation reports or one-off VMEC campaigns |
| `src/spectraxgk/operators` | Physics operators | Linear/nonlinear RHS kernels, field solve pieces, dissipation, gyroaverages | Runtime orchestration, benchmark policies |
| `src/spectraxgk/solvers` | Numerical methods | Explicit/IMEX/time policies, eigensolvers, Krylov methods, differentiable solve wrappers | Case-specific benchmark branches |
| `src/spectraxgk/diagnostics` | Physics observables | Growth/frequency windows, transport, free-energy, quasilinear formulas, statistics | Publication-panel builders |
| `src/spectraxgk/objectives` | Differentiable objectives | Pure JAX objective functions, FD/AD checks, line-search utilities | Subprocess runs, plotting, local campaign launchers |
| `src/spectraxgk/parallel` | Parallel execution | Independent-work batching, identity-gated decompositions, device utilities | Performance claims without profiler artifacts |
| `src/spectraxgk/workflows` | User workflows | TOML/runtime/executable orchestration, progress output, plotting command routing | Validation campaign code that is not a user workflow |
| `benchmarks/` | Benchmark drivers | Small reproducible benchmark entry points and manifests | Raw outputs, long campaign trees, figures, one-off tools |
| `examples/` | User education | Runnable scripts that teach the promoted API and physics | Release machinery, hidden campaign postprocessing |
| `tools/release` | CI/release gates | Deterministic checks used by CI or release readiness | Plot generation or exploratory campaign logic |
| `tools/comparison` | External-code comparisons | Explicit parity/reference utilities and benchmark-only external-code names | Runtime library conventions or general helper names |
| `tools/artifacts` | Figure/table builders | README/docs/manuscript figure generators with stable inputs | Long simulation launchers or raw campaign outputs |
| `tools/campaigns` | Active long-run campaigns | Launch/postprocess scripts for current documented campaigns | Stale probes, old fallback launchers, accepted artifacts |
| `tools/profiling` | Reproducible profiling | CPU/GPU profiler reproducers, Perfetto/XLA dump drivers | Unsupported speedup claims |
| `tests/unit` | Small correctness tests | Fast physics/numerics/unit checks with shared fixtures | End-to-end executable runs or one-test-per-tool wrappers |
| `tests/integration` | User workflows | Executable, runtime, plotting, artifact, example integration tests | Long research campaigns |
| `tests/validation` | Literature gates | Physics gates, convergence checks, benchmark comparisons, promotion guards | Smoke tests without physical assertions |
| `tests/tools` | Maintenance tooling | Parametrized tests for release/artifact/campaign/comparison/profiling tools | Hundreds of one-file-per-script tests |

## Finite Refactor Execution Plan

The refactor should now proceed in six bounded waves. Each wave must reduce
ambiguity, not just move files.

1. **Tools ownership and deletion wave.**
   - Finish moving explicit comparison utilities to `tools/comparison`.
   - Move figure/table builders to `tools/artifacts`.
   - Move active launch/postprocess scripts to `tools/campaigns`.
   - Move profiler reproducers to `tools/profiling`.
   - Delete or move out of `main` probes and local-only scripts unless docs,
     tests, or release manifests prove they are active.
   - Replace case-specific duplicate builders with manifest-driven commands
     where the only differences are case name, paths, or plotted labels.

2. **Test topology wave.**
   - Keep the flat root closed except for `conftest.py`; do not add new flat
     root tests.
   - Merge one-file-per-script tool tests into family suites under
     `tests/tools/{release,artifacts,campaigns,comparison,profiling}`.
   - Split the largest historical monoliths by physical/runtime contract rather
     than by old branch history.
   - Keep 95% package-wide coverage by replacing duplicated tests with
     parameterization and shared physical fixtures, not by deleting coverage.
   - Enforce the local fast shard under five minutes; long nonlinear and GPU
     campaigns remain explicit validation jobs.

3. **Validation-out-of-package wave.**
   - Empty `src/spectraxgk/validation` except for at most a tiny stable metrics
     facade if still needed by users.
   - Move benchmark runners to root `benchmarks/` or `tools/campaigns`.
   - Move promotion/admission policies that are release checks to
     `tools/release` or `tests/validation`.
   - Keep literature-anchored metrics in `src/spectraxgk/diagnostics` only when
     they are reusable physics observables rather than campaign policy.

4. **Source simplification wave.**
   - Merge small or prefix-driven modules into domain files with physical names.
   - Keep public facades small and stable: `linear`, `nonlinear`,
     `quasilinear`, `geometry`, `objectives`, `parallel`, `runtime`, and
     `artifacts`.
   - Remove backward-compatible aliases for retired behavior unless they are
     necessary for the documented executable or Python API.
   - Continue renaming comparison-code terminology to physical/numerical names
     except inside `tools/comparison`, comparison tests, and benchmark docs.

5. **Performance and differentiability wave.**
   - Define a small set of stable JIT boundaries for linear RHS, nonlinear RHS,
     field solve, diagnostics reduction, and geometry sampling.
   - Keep differentiable Python objectives pure and in-memory; keep executable
     progress/plotting/file I/O outside differentiated code.
   - For each optimization, add identity gates first, then profiler artifacts,
     then performance claims.
   - Focus first on cold-start/default-demo latency, linear cache reuse,
     nonlinear diagnostic streaming, and avoiding shape-changing wrappers.

6. **Docs, README, and release wave.**
   - Rebuild `docs/code_structure.rst` as a concise developer map rather than
     a migration diary.
   - Keep `benchmarks/`, `examples/`, and `tools/` roles explicit in README and
     docs.
   - Remove docs/static artifacts that are not referenced by README/docs or
     release manifests.
   - Run fast package tests, coverage gate, docs build, package build, release
     checks, repository-size check, and architecture manifest before tagging.

## Latest Complexity Audit And Consolidation Decisions

Audited on 2026-07-07 after commit
`4b57ef41 Consolidate preview compression tooling`:

- Branch and PR hygiene is clean. This clone has only `main` and `origin/main`;
  obsolete experimental PRs #4, #5, and #6 remain closed; PR #7 is merged. The
  current CI run for `4b57ef41` is in progress and earlier runs were cancelled by
  newer pushes, so the next check is to inspect that run after more work rather
  than polling continuously.
- The active topology is `src/spectraxgk`: 275 Python files,
  `tests`: 243 Python files, `tools`: 247 Python files, `examples`: 42 Python
  files, and `benchmarks`: 12 Python files. The recent artifact-test
  consolidations reduced `tests/tools/artifacts` from 94 to 26 files while
  preserving focused artifact gates.
- `docs/_static` is the largest tracked data footprint by count and size:
  1,572 tracked files and about 36.4 MiB, mostly compressed PNG/JSON/CSV
  validation artifacts. This is acceptable for release size, but every retained
  artifact must be referenced by README/docs or a release/artifact manifest.
- No tracked raw NetCDF, HDF5, pickle, or large generated outputs are present.
  The only tracked binary array bundles are the two small reference-mode NPZ
  files under `docs/_static/reference_modes`.
- The largest structural source offender is still
  `src/spectraxgk/validation`: 69 installable Python files and about 28.1k
  Python lines.
  It mixes benchmark branch policies, nonlinear-gradient campaigns,
  quasilinear ledgers, and stellarator campaign gates with the runtime package.
- The largest test-maintenance offenders are
  `tests/integration/runtime/test_runtime_runner.py` at about 4.2k lines,
  `tests/validation/benchmarks/test_benchmarks_runner_branches.py` at about
  3.0k lines, plus several 1.3k-2.3k line aggregate tests. The problem is now
  not flat roots, but giant historical-branch files and one-family-per-script
  preservation.
- The largest tool-maintenance offenders remain `tools/artifacts` with
  125 scripts, `tools/campaigns` with 48 scripts, `tools/comparison` with
  34 scripts, `tools/profiling` with 22 scripts, and `tools/release` with
  29 scripts. The next tool work should merge duplicate builders/checkers
  inside these folders, not create more folders.
- `benchmarks/` is already correctly at the root and small. It should stay as
  the lightweight benchmark-driver layer; long campaign launchers and generated
  outputs should not move into it.
- Keyword scans show that `probe`, `pilot`, `synthetic`, `reduced`, `legacy`,
  and comparison-code terms are concentrated in docs, examples, tests/tools,
  `tools/artifacts`, `tools/campaigns`, and `src/spectraxgk/validation`.
  Current solver kernels are less affected, which means cleanup should focus on
  claims, examples, campaign scaffolding, and validation packaging first.
- Current profiler manifests identify the real performance-risk zones as
  cold-start compilation, linear cache construction, linear RHS kernels,
  nonlinear bracket/field-solve/full-RHS throughput, diagnostic
  materialization, VMEC/Boozer sampling and conversion, and nonlinear domain
  decomposition communication. No new speedup claim should be made without
  fresh profiler and numerical-identity evidence.

Decision rules from this audit:

1. If a source file only supports one manuscript/campaign artifact, move it out
   of `src` or delete it.
2. If a tool differs from another tool only by case name, figure name, or
   manifest path, merge it into a manifest-driven builder/checker.
3. If a test only checks a single tool wrapper, fold it into a parametrized
   tool-family test.
4. If a docs/static artifact is not referenced by README/docs or a release
   manifest, delete it or regenerate it only in release artifacts.
5. If a name uses comparison-code terminology for a physical convention or
   numerical method, rename it unless the file is explicitly a benchmark or
   comparison utility.
6. If a feature is not promoted, validated, documented, and tested, it should
   not stay in the main runtime path.
7. If a refactor would only make fewer but much larger files, reject it. The
   intended outcome is fewer concepts, fewer duplicated policies, fewer public
   aliases, and clearer domain ownership.

Immediate execution sequence from this audited state:

1. Finish artifact-test and artifact-tool consolidation down to capability
   families. The next target is to keep `tests/tools/artifacts` below 30 while
   moving repeated artifact schemas into manifest-driven `tools/artifacts`
   builders.
2. Split or parametrize the two largest historical branch tests:
   `tests/integration/runtime/test_runtime_runner.py` and
   `tests/validation/benchmarks/test_benchmarks_runner_branches.py`. The split
   should produce behavior contracts, not more case-history files.
3. Start validation-out-of-package with `src/spectraxgk/validation/benchmarks`.
   Move benchmark harnesses to root `benchmarks/` or tests, and keep only
   reusable diagnostics in `src`.
4. Audit `docs/_static` references and delete unreferenced pilot/probe artifacts
   before adding new figures.
5. Rename comparison-code terminology in source/tests where it is not an
   explicit benchmark/comparison context.
6. Only after topology and ownership shrink, run profiler-backed hot-path
   refactors: default demo latency, linear cache/RHS, nonlinear RHS/bracket,
   diagnostic streaming, and VMEC/Boozer in-memory geometry.
7. Rebuild `docs/code_structure.rst` and `docs/architecture_refactor_plan.rst`
   as concise developer guides after the first validation-out-of-package move
   lands.

## Final Consolidation Model

The refactor should now optimize for a small number of domain concepts rather
than a small number of arbitrary files. The target product shape is:

- **Runtime package**: promoted gyrokinetic solver kernels, geometry contracts,
  diagnostics, differentiable objectives, parallel execution policies, IO, and
  user workflows.
- **Examples**: runnable educational scripts that teach the promoted API.
- **Benchmarks**: small reproducible drivers and manifests for validation and
  performance, not raw outputs or long campaigns.
- **Tools**: repository-maintenance entry points only: release gates, artifact
  builders, profiling reproducers, comparison utilities, and active campaigns.
- **Tests**: domain-organized, parametrized suites that protect physics,
  numerics, executable workflows, artifact schemas, and repository policy.

Files are retained only when they have a current owner and at least one of
these roles:

1. promoted library/runtime functionality;
2. documented user example;
3. reproducible benchmark driver;
4. CI/release gate;
5. reviewed docs/readme artifact builder;
6. explicit external-reference comparison utility;
7. profiler reproducer linked from performance docs;
8. active long-run campaign with a documented claim or pending gate;
9. test fixture or validation gate with a physics/numerics assertion.

Everything else leaves `main`: delete it, move it to a draft experiment branch,
or turn it into a documented benchmark/example before it is retained. This is
the main mechanism for reducing code lines without hiding complexity in larger
files.

### Planned High-Impact Tranches

1. **Finish flat-tool cleanup.** Completed in this tranche: artifact/reference
   helpers moved to `tools/artifacts`, active RHS/ky diagnostics moved to
   `tools/comparison`, the VMEC metadata patcher moved to `tools/campaigns`,
   and two no-owner probes left `main`. Target achieved: zero flat tool scripts
   except `tools/__init__.py`.
2. **Collapse tools by capability.** Merge duplicated artifact/status builders
   into manifest-driven builders where only case names, labels, or output paths
   differ. Target: `tools/` below 180 scripts before source moves, then below
   100 before release.
3. **Reorganize tests by domain.** Three topology moves relocated 138 tests into
   `tests/unit`, `tests/integration`, `tests/validation`, existing tool
   folders, and `tests/release`. Flat root tests are now closed except for
   `conftest.py`; the next test move is merging one-file-per-script tool tests
   into parametrized family tests. Target: fewer than 180 tests before
   validation extraction, then fewer than 100.
4. **Move validation campaigns out of the installable package.** Keep reusable
   physics metrics in `diagnostics` or a tiny `validation` facade; move campaign
   launchers, report builders, and holdout ledgers to `benchmarks`, `tools`, or
   `tests/validation`. Target: `src/spectraxgk/validation` from 69 files to at
   most 5.
5. **Consolidate source domains after imports settle.** Merge `terms` into
   `operators` where appropriate, combine `geometry_backends` into `geometry`,
   shrink `api/*` to documented facades, and collapse over-split objective and
   workflow modules by physical contract. Target: 60-90 source files without
   creating giant miscellaneous modules.
6. **Profile-backed performance pass.** Only after topology stabilizes, run
   before/after profiles for default demo latency, linear cache/RHS, nonlinear
   bracket/field solve, diagnostic streaming, VMEC/Boozer sampling, and
   parallel execution. Every performance claim needs an identity gate and a
   profiler artifact.

### Benchmarks, Tools, And Examples Boundary

- If a file teaches a user workflow, it belongs in `examples/`.
- If a file measures accuracy or runtime in a small reproducible way, it belongs
  in `benchmarks/`.
- If a file builds docs/readme artifacts, enforces release policy, profiles
  kernels, launches long campaigns, or compares against an external code, it
  belongs under a purpose folder in `tools/`.
- Raw NetCDF outputs, profiler traces, scratch logs, and campaign directories
  stay ignored or are attached to releases, not tracked.

This boundary should be enforced before large source refactors so that new
developers do not have to infer whether a script is an example, a test, a
benchmark, or a release-only artifact builder.

### Performance Bottleneck Reduction Plan

The architecture work should actively reduce runtime and memory risk:

- Keep hot JIT boundaries coarse and stable: geometry sampling, linear cache
  construction, linear RHS, nonlinear RHS/bracket, field solve, and diagnostics
  reductions.
- Remove shape-changing wrappers and migration shims around hot kernels.
- Cache static geometry, gyroaverage, drift, and field-solve coefficients.
- Stream/reduce nonlinear diagnostics by default; full histories are explicit
  opt-ins.
- Keep differentiable objective functions pure and in-memory; executable
  progress, plotting, TOML emission, and file I/O stay outside AD paths.
- Treat nonlinear domain decomposition as diagnostic until serial-vs-decomposed
  identity and profiler-backed CPU/GPU speedup both pass on equivalent physical
  transport windows.

## Immediate Obsolete/Experimental Candidates

These are candidates, not automatic deletions. Each must be checked for imports,
docs references, tests, and promoted claims before removal.

- cETG/reduced-model runtime implementation has been retired from `main`.
  Remaining references must be limited to fail-closed input validation,
  progress-log history, and explicit notes that the path is not promoted.
- `examples/theory_and_demos/reduced_stellarator_itg/` and synthetic
  low-turbulence plotting paths. Default action: keep only if explicitly
  documented as a pedagogical differentiability demo; otherwise move out of
  release examples.
- `tools/probe_*`, `tools/design_*`, `tools/write_*` campaign generators, and
  one-off `build_*status*` scripts that are not referenced by current docs,
  release gates, or accepted artifacts. Default action: delete or move to
  `tools/campaigns` only if still active.
- `docs/_static` CSV/JSON artifacts whose only purpose is an obsolete
  comparison/debug panel. Default action: keep only reviewed publication/readme
  artifacts; remove stale companions when docs no longer reference them.
- Tests whose only assertion is preserving deleted compatibility behavior.
  Default action: remove with the feature; do not keep backward-compatibility
  tests for unpromoted behavior.

## Realistic Performance Bottlenecks

The current profiler artifacts point to engineering bottlenecks rather than a
single obvious line-level bug:

- Cold start and JAX compile time are user-visible for quickstart and small
  examples. Fix with stable compiled boundaries, smaller default demo, warm
  cache reuse, and live progress output; do not hide compilation behind silent
  waits.
- Linear cache construction and linear RHS term evaluation dominate some
  CPU-side profiles. Fix by reusing geometry/gyroaverage/field-solve
  coefficients and avoiding shape-changing wrappers around hot kernels.
- Nonlinear RHS cost is split across field solve, linear RHS, bracket, and
  diagnostic materialization. Fix one stage at a time with before/after profiler
  artifacts and serial-vs-refactor identity gates.
- Nonlinear whole-state sharding is identity-correct but not a production
  speedup yet. Keep it as diagnostic until the communication model and
  transport-window gates predict and verify real speedup.
- VMEC/Boozer workflows can waste time in file I/O and repeated geometry
  conversion. Differentiable objectives must use in-memory PyTree/array
  contracts and cache static geometry sampling where possible.
- Long nonlinear examples can waste memory by saving dense histories. Default
  to streamed/reduced diagnostics and make full histories explicit opt-ins.

Performance work must be coupled to architecture: the code should expose a few
stable JIT kernels and a few orchestration layers, not dozens of wrappers that
force recompilation or obscure array layout.

## Non-Negotiable Invariants

- Preserve validated numerical behavior for promoted linear, nonlinear,
  quasilinear, geometry, plotting, executable, and artifact workflows.
- Preserve the user entry points `spectraxgk`, `spectrax-gk`, `spectraxgk --plot`,
  default demo execution, TOML-driven runtime execution, and documented Python
  workflows.
- Keep comparison-code references only in explicit benchmark/comparison contexts:
  benchmark panels, parity tools, reference reruns, comparison tests, and docs
  sections that discuss external validation.
- Do not keep native source names such as `gx_*`, `GX-reference`, or
  `GX-style` when the object is really a physical convention, numerical scheme,
  file schema, or benchmark-compatible normalization.
- Keep differentiable Python workflows pure: no file I/O, plotting,
  subprocesses, terminal progress, global mutable state, or host callbacks
  inside promoted objective functions.
- The executable path may be faster and non-differentiable; the Python research
  API must remain JAX-compatible where promoted.
- Do not make new performance claims without profiler artifacts, identity gates,
  and equivalent workload definitions.
- Do not track transient raw outputs, NetCDF dumps, scratch logs, profiler
  traces, or generated office campaign directories in git.
- Keep `benchmarks/` small: benchmark drivers and manifests only, not raw
  results or large campaign code.
- Keep `examples/` user-facing: scripts should be runnable and pedagogical, not
  hidden release machinery.

## Target Repository Layout

The final layout should make ownership obvious from the path.

```text
src/spectraxgk/
  __init__.py
  _version.py
  cli.py
  config.py
  runtime.py
  linear.py
  nonlinear.py
  quasilinear.py
  core/
  geometry/
  operators/
  solvers/
  diagnostics/
  objectives/
  parallel/
  io/
  workflows/

tests/
  unit/
  integration/
  validation/
  tools/
  release/
  conftest.py

benchmarks/
  README.md
  linear/
  nonlinear/
  performance/
  manifests/

tools/
  release/
  artifacts/
  profiling/
  comparison/
  campaigns/
  README.md

examples/
  README.md
  quickstart/
  linear/
  nonlinear/
  geometry/
  quasilinear/
  optimization/
  plotting/
  parallelization/

docs/
  _static/
  user guides, theory, numerics, validation, performance, API, developer docs
```

The target `src/spectraxgk` package should be around 60-90 Python files. A
possible allocation is:

| Package | Target files | Notes |
| --- | ---: | --- |
| root facades | 7-8 | Stable user imports and executable only. |
| `core` | 5-7 | Grids, velocity basis, species, PyTree/state contracts. |
| `geometry` | 8-10 | Analytic, Miller, VMEC/Boozer bridge, flux-tube sampling, sensitivities. |
| `operators` | 8-12 | Linear RHS, nonlinear bracket/RHS, fields, collisions, gyroaverages, caches. |
| `solvers` | 8-10 | Time stepping, eigen/Krylov, IMEX, explicit, Diffrax wrappers. |
| `diagnostics` | 6-8 | Growth/frequency, modes, transport, quasilinear diagnostics, normalization. |
| `objectives` | 7-10 | Linear, quasilinear, nonlinear-window, stellarator, AD/FD/UQ helpers. |
| `parallel` | 4-6 | Independent work, velocity sharding, experimental domain decomposition gates. |
| `io` / `artifacts` | 5-7 | NetCDF, restart, plotting, TOML, artifact schemas. |
| `workflows` | 6-8 | Runtime orchestration, named cases, examples support, optimization workflows. |
| `validation` | 0-5 | Only reusable public metrics if they cannot live under `diagnostics` or `benchmarks`. |

## What Moves Out Of `src/spectraxgk`

The installable package should contain solver functionality, reusable public
APIs, and lightweight runtime workflows. It should not contain manuscript
campaign orchestration or one-off validation report builders.

Move out of `src/spectraxgk`:

- Long benchmark campaigns and scan-specific branch logic.
- External-reference comparison harnesses that are not needed by normal users.
- Manuscript status report builders.
- Nonlinear-gradient campaign design and follow-up generators.
- Quasilinear calibration holdout admission ledgers when they are artifact
  builders rather than reusable diagnostics.
- Stellarator transport campaign launchers and selection policies that only
  postprocess a specific research campaign.
- Any file whose only caller is a `tools/build_*`, `tools/write_*`,
  `tools/postprocess_*`, or one-off test.

Move to one of these destinations:

- `benchmarks/` for small reproducible benchmark drivers and manifests.
- `tools/artifacts/` for docs/readme figure builders.
- `tools/comparison/` for explicit external-code comparison utilities.
- `tools/campaigns/` for long-run launch and postprocess helpers.
- `tests/validation/` for test-only reference data assembly.
- A draft PR or experiment branch for non-promoted research code that should not
  ship on `main`.

Keep in `src/spectraxgk` only if the code is a reusable solver, diagnostic,
objective, geometry bridge, artifact schema, or documented public workflow.

## Obsolete, Experimental, And Non-Promoted Code Policy

Every file should be classified before it is retained.

| Classification | Action |
| --- | --- |
| Promoted runtime/library functionality | Keep in `src`, consolidate into target domains. |
| Public example functionality | Keep in `examples`, with README navigation and tested commands. |
| Reproducible benchmark driver | Keep in `benchmarks`, with small manifest and no raw outputs. |
| Release gate | Keep in `tools/release`, with tests under `tests/release` or `tests/tools`. |
| Figure/artifact builder | Keep in `tools/artifacts`, if the output is referenced by docs/readme. |
| Profiler/performance reproducer | Keep in `tools/profiling`, if referenced by performance docs or manifests. |
| External comparison utility | Keep in `tools/comparison`, only if used by benchmark docs or parity reruns. |
| Long campaign launcher | Keep in `tools/campaigns`, only if documented and still useful. |
| Debug/probe script | Delete from `main` or move to a draft PR/experiment branch. |
| Legacy behavior | Delete unless it is a documented promoted feature. |
| Non-validated physics path | Remove from README/docs claims and move out of `main` unless it has gates. |

Specific first candidate:

- The cETG reduced-model path is already retired from `main`. Keep only the
  fail-closed runtime input validation and progress-log history. Do not preserve
  old examples, docs, tools, compatibility wrappers, or tests for that
  non-promoted solver family.

## Test Consolidation Plan

Current problem: `tests/` has 246 Python files after adding a shared path
helper and consolidating the first parallel identity artifact-gate family. The
root now has only `conftest.py`. `tests/tools` still has many
one-file-per-script tests and must keep consolidating by tool family instead of
preserving one test file per script.

Target: fewer than 100 Python test files while preserving >=95% package-wide
coverage and physics confidence.

Final test folders:

```text
tests/unit/core/
tests/unit/geometry/
tests/unit/operators/
tests/unit/solvers/
tests/unit/diagnostics/
tests/unit/objectives/
tests/unit/parallel/
tests/integration/runtime/
tests/integration/examples/
tests/validation/benchmarks/
tests/validation/physics_gates/
tests/tools/
tests/release/
```

Consolidation rules:

- Replace one-test-file-per-tool with parametrized test modules by tool family:
  `test_release_gates.py`, `test_artifact_builders.py`,
  `test_campaign_generators.py`, `test_comparison_tools.py`, and
  `test_profilers.py`.
- Replace large monolithic runtime files with domain files:
  runtime config, startup, initial conditions, progress/output, artifact writing,
  linear execution, nonlinear execution, and plotting.
- Move shared builders to fixtures instead of copy-pasting state setup across
  many tests.
- Keep physics tests as tests, not smoke-only scaffolds: basis orthogonality,
  conservation/symmetry, manufactured solutions, observed-order checks,
  late-time growth/frequency metrics, nonlinear window metrics, restart parity,
  geometry parity, AD/FD consistency, and artifact schema stability.
- Long office/GPU/external comparison campaigns remain explicit benchmark or
  validation commands, not default local tests.
- Keep local tests under five minutes. Keep wide coverage in CI shards.

Suggested target file budget:

| Test area | Target files |
| --- | ---: |
| unit tests | 35-45 |
| runtime/integration tests | 10-15 |
| validation/physics gate tests | 15-20 |
| tool/release tests | 15-20 |
| examples/CLI tests | 5-8 |
| total | 80-95 |

## Tool Consolidation Plan

Current problem: `tools/` has 259 Python scripts. The flat root has been closed
down to `tools/__init__.py` after release, comparison, artifact, campaign,
profiling, benchmark, generator, compression-helper, reference-helper,
diagnostic, and VMEC-helper moves. The remaining problem is duplication inside
purpose folders, especially artifact/status builders and one-tool-one-test
coverage.

Target: fewer than 100 Python tool scripts, organized by purpose.

Final tool folders:

```text
tools/release/
tools/artifacts/
tools/profiling/
tools/comparison/
tools/campaigns/
tools/README.md
```

Retain only:

- Release gates used by CI or release workflows.
- Artifact builders for figures/tables referenced by README/docs.
- Profilers tied to `docs/performance.rst` and performance manifests.
- External comparison utilities tied to benchmark validation docs.
- Campaign launch/postprocess helpers that are still needed and documented.

Delete or move out of `main`:

- One-off debug/probe scripts.
- Duplicated plot/build scripts that differ only by one case name.
- Historical manuscript status builders not referenced by current docs.
- Launch writers for blocked/deferred campaigns.
- Any script whose generated artifact is no longer tracked or referenced.

Flat-tool audit outcome:

| Family | Outcome |
| --- | --- |
| `make_*`, `derive_*`, `digitize_*` builders | Moved to `tools/artifacts`; docs, manifests, and tests now use artifact paths. |
| RHS/ky diagnostic helpers | Moved to `tools/comparison` because comparison tests still exercise them. |
| VMEC metadata patch helper | Moved to `tools/campaigns` because it supports the documented VMEC-JAX workflow. |
| no-reference probes | `cyclone_resolution_sweep.py` and `etg_eigenspectrum.py` deleted from `main`. |

The next tool tranche should merge duplicate builders and status tools inside
`tools/artifacts` and collapse one-tool-one-test files. The target is to drive
total tool scripts below 180 before validation code moves, then below 100 before
release.

Suggested target file budget:

| Tool area | Target files |
| --- | ---: |
| release gates | 15-20 |
| artifact builders | 25-35 |
| profiling/performance | 10-15 |
| comparison utilities | 10-15 |
| campaigns | 15-20 |
| total | 75-95 |

## Source Consolidation Plan

Current problem: `src/spectraxgk` has 338 Python files, and 69 are validation
modules. The package contains too much campaign and validation machinery.

Target: at most 100 Python files in `src/spectraxgk`.

Source consolidation rules:

- Merge over-split modules when they share one conceptual contract and one test
  ownership area.
- Do not merge unrelated physics just to reduce file count.
- Prefer cohesive domain modules over chains of tiny `_core`, `_helpers`,
  `_reports`, `_policies`, and `_contracts` files.
- Replace broad public API re-export lists with small documented facades.
- Remove migration-era wrappers once examples and tests use canonical imports.
- Keep low-level kernels close to their public operator/solver owner.
- Keep runtime orchestration outside pure differentiable objective code.
- Keep optional VMEC/Boozer integration isolated from the base solver import path.

Primary source moves:

1. Move `src/spectraxgk/validation/benchmarks` into `benchmarks/` and
   `tests/validation/benchmarks`, keeping only reusable metrics in package code.
2. Move `src/spectraxgk/validation/nonlinear_transport` and most stellarator
   campaign code into `diagnostics`, `objectives`, `tools/campaigns`,
   `tools/release`, or `tests/validation` depending on whether each function is
   reusable math, objective policy, release policy, campaign orchestration, or
   test-only validation.
3. Merge `geometry` and `geometry_backends` into one `geometry` package with
   clear files: `analytic.py`, `miller.py`, `vmec.py`, `booz_xform.py`,
   `flux_tube.py`, `sensitivity.py`, and `io.py`.
4. Merge `terms` into `operators` unless a term is a public mathematical
   interface. Users should see operators and solvers, not two overlapping
   implementation namespaces.
5. Merge `artifacts` and runtime IO into `io` where possible. Keep plotting
   interfaces clear, but remove one-file-per-NetCDF-section sprawl.
6. Merge `api/*` into a smaller root API registry or package-level `__init__`
   files. The broad public export list should shrink.
7. Consolidate `objectives` into a few research workflows: linear growth,
   quasilinear flux, nonlinear window, stellarator, autodiff checks, and UQ.
8. Keep `parallel` focused on production independent-work parallelism and
   identity-gated experimental nonlinear decomposition. Move historical
   sharding status/report builders out of the package.

## Examples Plan

Current problem: `examples/` contains useful workflows but also legacy configs,
research scaffolds, and scripts whose purpose is not obvious.

Target: a small, navigable examples tree with a README and tested commands.

Actions:

- Add or rewrite `examples/README.md` as the navigation entry point.
- Split examples by user task: quickstart, linear, nonlinear, geometry,
  quasilinear, optimization, plotting, and parallelization.
- Every example folder should state:
  - what the example demonstrates,
  - expected runtime on a laptop,
  - required optional dependencies,
  - expected output files,
  - whether it is pedagogical, benchmark-level, or long-run research.
- Remove examples for retired or non-promoted features.
- Remove or move examples that depend on untracked private artifacts unless the
  README gives a reproducible path to generate inputs.
- Keep optimization examples close to the VMEC-JAX style requested by users, but
  label long nonlinear optimization as research/long-run unless fully gated.
- Add a plotting example showing `spectraxgk --plot output_file` and Python
  plotting from saved outputs.

## Benchmarks Plan

`benchmarks/` should stay at the repository root. It is currently small and
mostly well scoped.

Target role:

- Small benchmark drivers.
- Small TOML configs.
- Small manifests pointing to promoted docs artifacts.
- No raw outputs, NetCDF dumps, restart bundles, or long campaign directories.

Actions:

- Move reusable benchmark harness code out of `src/spectraxgk/validation` into
  `benchmarks/` if it is not a runtime library feature.
- Keep benchmark results as small CSV/JSON summaries and compressed figures in
  `docs/_static` only after review.
- Add docs pages that explain what each benchmark proves and what it does not
  prove.
- Keep external-code comparisons in benchmarks/docs context only.

## Performance Refactor Plan

Refactoring must improve simplicity and performance together. The goal is not
just fewer files.

Known performance-sensitive areas:

- JAX cold-start and first compile.
- Linear cache construction.
- Linear RHS kernels.
- Nonlinear bracket and field solve.
- Diagnostic materialization in nonlinear runs.
- Restart/artifact writing for long runs.
- VMEC/Boozer geometry sampling and repeated equilibrium conversion.
- Python orchestration overhead in runtime and campaign tools.
- Nonlinear domain decomposition communication overhead.

Performance actions:

1. Preserve stable JIT boundaries while consolidating modules. Do not introduce
   shape-changing wrappers or dynamic Python branches inside hot kernels.
2. Move Python orchestration out of inner loops and into workflow layers.
3. Cache geometry, linear terms, gyroaverages, and field-solve coefficients
   where shapes/configuration are static.
4. Stream diagnostics by default for long nonlinear runs; make full histories
   opt-in.
5. Keep differentiable objectives pure and small, but use fast executable paths
   for CLI runs where AD is not required.
6. Use profiler-backed tranches: collect before/after CPU and GPU artifacts for
   any runtime claim.
7. Keep nonlinear domain decomposition diagnostic until a real transport-window
   identity gate and speedup gate pass.
8. Do not let refactor increase memory footprint. Track peak RSS and device
   memory for headline cases.

Performance gates:

- Existing benchmark diagnostics remain within validation envelopes.
- Runtime/memory panels use equivalent workloads and hardware metadata.
- Any new speedup claim has serial-vs-parallel identity, timing, memory, and
  profiler artifacts.
- CLI quickstart remains responsive and prints progress/ETA.

## Documentation Plan

Documentation should explain the smaller architecture and claim boundaries.

Actions:

- Update `docs/code_structure.rst` after each major phase.
- Update `docs/architecture_refactor_plan.rst` or replace it with this plan as
  the current authority once implementation starts.
- Add a developer guide explaining the target file layout, where to add new
  physics, where to add tests, and where not to add campaign code.
- Keep README concise: install, quickstart, executable plotting, headline
  benchmark/runtime panel, main validation figures, examples map, and claim
  scope.
- Move long derivations, equations, algorithms, and validation details to docs.
- Add clear docs for benchmarks vs examples vs tools vs tests.

## Branch And PR Policy

The user-facing `main` branch should contain only promoted, validated,
maintainable functionality.

- Keep one main refactor branch or draft PR for this consolidation if a PR is
  needed. Do not create many overlapping refactor PRs.
- Obsolete closed PR branches stay closed and should not be resurrected unless a
  specific piece is intentionally ported.
- Experimental or non-validated code should leave `main`. If it may be useful
  later, move it to a draft PR or experiment branch with explicit non-release
  status.
- Do not keep backward-compatible legacy behavior solely because tests exist.
  If the behavior is not part of the next promoted product, remove it and remove
  or rewrite its tests/docs/examples.

## Phase Plan

### Phase 0: Inventory And Classification

Goal: every tracked Python file has an owner and a keep/move/delete decision.

Steps:

1. Generate a machine-readable inventory for `src`, `tests`, `tools`,
   `examples`, and `benchmarks` with file owner, importers, line count, test
   coverage relevance, docs references, and artifact outputs.
2. Classify each file as promoted library, public example, benchmark driver,
   release gate, artifact builder, profiler, comparison utility, campaign
   helper, test-only helper, legacy, or delete.
3. Identify files with no importers, no tests, no docs references, or only
   blocked/deferred campaign references.
4. Produce deletion and move lists before code edits.
5. Add a repository-topology manifest that records target counts and fails when
   new flat files are added in disallowed locations.

Exit gates:

- Inventory exists and is reproducible.
- Every file has a classification.
- Initial deletion/move list is reviewed in git diff.
- No solver behavior changes yet.

### Phase 1: Test Tree Collapse

Goal: reduce `tests/` below 100 Python files without losing coverage or physics
confidence.

Steps:

1. Create the new test folder hierarchy.
2. Move tests mechanically first; keep names and assertions unchanged where
   possible.
3. Merge one-file-per-tool tests into parametrized modules by tool family.
4. Extract common fixtures for runtime configs, geometry stubs, small arrays,
   artifact tempdirs, and mock CLI dependencies.
5. Split huge runtime and benchmark runner tests by behavior, not by historical
   file name.
6. Remove tests that only preserve deleted legacy behavior.
7. Update CI shards to point at folders rather than long flat file lists.

Exit gates:

- `find tests -name '*.py' | wc -l` is below 100.
- Wide package coverage is at least 95%.
- Fast local tests stay below five minutes.
- Physics/numerics gate coverage is preserved or improved.

### Phase 2: Tools Collapse

Goal: reduce `tools/` below 100 Python scripts and make each remaining tool's
purpose obvious.

Steps:

1. Create `tools/release`, `tools/artifacts`, `tools/profiling`,
   `tools/comparison`, and `tools/campaigns`.
2. Move scripts mechanically into folders and update tests/CI/docs references.
3. Merge repetitive figure builders into shared artifact-builder modules with
   case manifests.
4. Merge repetitive checkers into manifest-driven release gates where possible.
5. Delete debug/probe scripts that are not referenced by current docs, tests, or
   release manifests.
6. Move blocked/deferred campaign launchers out of `main` unless they are still
   explicitly part of the roadmap.
7. Add `tools/README.md` explaining what belongs in each folder.

Exit gates:

- `find tools -name '*.py' | wc -l` is below 100.
- Release workflow still runs all release gates.
- Docs/readme artifact builders still regenerate promoted figures.
- No raw generated outputs are tracked.

### Phase 3: Validation Leaves The Installable Package

Goal: remove campaign validation machinery from `src/spectraxgk`.

Steps:

1. Move benchmark harnesses to `benchmarks/` unless they are public runtime APIs.
2. Move manuscript/campaign validation code to `tools/campaigns` or
   `tests/validation`.
3. Keep only reusable metrics under `src/spectraxgk/diagnostics` or a tiny
   `src/spectraxgk/validation` facade if unavoidable.
4. Update public API exports and docs.
5. Remove imports from runtime/library code into campaign validation code.
6. Replace package validation references in examples with benchmark or tools
   entry points.

Exit gates:

- `src/spectraxgk/validation` has 0-5 Python files.
- Installable package file count drops substantially.
- Runtime package imports do not depend on benchmark/campaign modules.
- Benchmark and validation docs still explain external comparisons.

### Phase 4: Source Package Consolidation

Goal: reduce `src/spectraxgk` to at most 100 Python files.

Steps:

1. Consolidate `geometry` and `geometry_backends` into one geometry package.
2. Consolidate `terms` into `operators` and remove overlapping abstractions.
3. Consolidate `artifacts` and runtime IO into a smaller `io`/artifact surface.
4. Reduce broad `api/*` re-export modules and shrink public API exports.
5. Consolidate objectives into linear, quasilinear, nonlinear-window,
   stellarator, AD/FD/UQ modules.
6. Keep public root facades stable but thin.
7. Remove migration wrappers and compatibility aliases after tests/examples use
   canonical names.
8. Remove remaining compatibility wrappers for retired or non-promoted paths.

Exit gates:

- `find src/spectraxgk -name '*.py' | wc -l` is at most 100.
- Public CLI and documented Python examples still work.
- No unintentional import cycles or optional-dependency import failures.
- Ruff, mypy, fast tests, wide coverage, and docs build pass.

### Phase 5: Examples And Benchmarks Cleanup

Goal: examples are pedagogical and benchmarks are reproducible.

Steps:

1. Rewrite `examples/README.md` with a task-oriented map.
2. Remove or move legacy examples whose underlying feature is retired or not
   promoted.
3. Ensure each example has bounded runtime guidance and expected outputs.
4. Move any benchmark-like example into `benchmarks/`.
5. Move any test-only/example-smoke helper into `tests/integration/examples`.
6. Keep root `benchmarks/README.md` and manifests current.

Exit gates:

- Example scripts run or are explicitly marked long-run/manual.
- README links only current examples and current figures.
- Benchmarks contain no raw transient outputs.

### Phase 6: Performance-Aware Kernel Refactor

Goal: use the cleaner architecture to reduce runtime and memory.

Steps:

1. Run baseline CPU/GPU profiling for quickstart, Cyclone linear/nonlinear, KBM,
   W7-X, HSX, and a VMEC/Boozer optimization micro-workflow.
2. Identify compile, cache-build, RHS, bracket, field-solve, diagnostic, and IO
   shares separately.
3. Refactor one hot path at a time, with before/after profiler artifacts.
4. Prefer vectorized/JIT-fused kernels over Python orchestration.
5. Stream diagnostics and reduce full-history materialization.
6. Validate numerical identity or physics envelopes after each change.
7. Refresh runtime/memory panel only after measured wins are real.

Exit gates:

- No headline benchmark regresses without a documented accuracy tradeoff.
- Runtime/memory panel is regenerated from fresh measured artifacts.
- Any speedup claim has profiler and identity evidence.

### Phase 7: Documentation, Claim Scope, And Release

Goal: ship the refactored version from a clean, documented, green state.

Steps:

1. Update README to describe the smaller package, examples map, benchmark map,
   runtime/memory panel, and claim boundaries.
2. Update docs for architecture, code structure, testing, validation strategy,
   performance, examples, and API.
3. Remove stale references to deleted files, retired reduced-model paths, old
   tool paths, and obsolete validation lanes.
4. Run release gates:
   - repository size manifest,
   - architecture/topology manifest,
   - release artifact manifest,
   - release readiness,
   - ruff,
   - mypy,
   - fast tests,
   - wide coverage,
   - docs build,
   - package build.
5. Bump version only after gates pass.
6. Tag and release only from green `main`.

Exit gates:

- All hard file-count targets pass.
- Documentation matches actual file layout and promoted claims.
- CI is green.
- Package builds and publishes cleanly.

## Implementation Order

1. Keep the topology manifest current and fail any regression in source, test,
   tool, flat-test, flat-tool, or installable-validation file counts.
2. Generate and review a file-classification inventory for `src`, `tests`,
   `tools`, `examples`, `benchmarks`, and docs artifacts.
3. Remove remaining stale references and scaffolding for retired/non-promoted
   paths.
4. Reorganize and merge the remaining flat tests below 100 total files.
5. Reorganize, merge, and delete tools below 100 total files.
6. Move validation/campaign code out of `src`.
7. Consolidate source domains below 100 files.
8. Clean examples and benchmarks.
9. Run performance profiling and targeted hot-path refactors.
10. Update docs/readme/API references.
11. Run full local and CI release gates.
12. Bump, tag, release.

## Progress Log

- 2026-07-08: consolidated the VMEC/Boozer aggregate alpha-heldout and
  surface-heldout artifact builders into the single
  `tools/artifacts/build_vmec_boozer_aggregate_holdout_gate.py` family command
  with `alpha` and `surface` subcommands. The existing evidence artifact names
  remain unchanged, the JSON metadata now points to the family command, and the
  shared writer keeps the same split objective, reduction, CSV, PNG, and PDF
  contracts. Focused aggregate-artifact tests, the VMEC/Boozer aggregate release
  gate tests, package-architecture tests, release artifact manifest check,
  architecture/differentiable refactor manifest checks, ruff, py_compile, and
  mypy on the changed builder passed. Tool Python files dropped to 238.

- 2026-07-08: consolidated the VMEC/Boozer solver-frequency, quasilinear, and
  reduced nonlinear-window gradient artifact wrappers into
  `tools/artifacts/build_vmec_boozer_gradient_gate.py` with `frequency`,
  `quasilinear`, and `nonlinear-window` subcommands. The documented artifact
  basenames and backend report functions are unchanged; the command now owns
  shared VMEC/Boozer parser defaults, artifact writing, JSON-only output, and
  nonlinear-window stencil validation. Added focused tests for all three
  subcommands, and the VMEC/Boozer artifact shard, package-architecture tests,
  stale-reference scan, ruff, py_compile, and mypy on the changed builder
  passed. Tool Python files dropped to 236.

- 2026-07-08: consolidated the parallel identity gate artifact scripts into
  `tools/artifacts/generate_parallel_identity_gate.py` with `ky-scan`,
  `logical-cpu`, and `quasilinear-runtime` subcommands. The release artifact
  basenames stay unchanged: `parallel_ky_scan_gate`,
  `logical_cpu_parallel_scan_gate`, and `quasilinear_runtime_parallel_gate`.
  The performance manifest now records the family script as the profiling tool
  and keeps subcommand details in the optimization action text. Focused
  parallel artifact tests, CLI help smoke tests for all subcommands, stale
  script-reference scan, performance-manifest check, ruff, py_compile, and
  mypy on the changed builder passed. Tool Python files dropped to 234.

- 2026-07-08: converted the root `spectraxgk` package,
  `spectraxgk.api` registry, and `spectraxgk.parallel` facade to lazy
  public-export resolution. This preserves `__all__`,
  `from spectraxgk import ...`, `from spectraxgk import benchmarks`, grouped
  API-registry imports, and parallel facade imports, but avoids importing
  NumPy/JAX-heavy solver stacks when callers only need version metadata or pure
  contracts such as `spectraxgk.parallel.decomposition`. No-site-packages
  regression tests now assert that root import, API-registry import, and pure
  decomposition import do not load NumPy or JAX. Cold root import time in the
  local environment dropped from about 1.17 s to about 0.003 s, and
  `spectraxgk.api` import dropped from about 1.22 s to about 0.004 s. Focused
  public API, parallel decomposition, parallel facade, velocity/nonlinear
  parallel, ruff, mypy on changed source, py_compile, architecture,
  release-readiness, and dependency-light artifact smoke checks passed.

- 2026-07-08: folded decomposition-contract status into the release-facing
  `tools/artifacts/build_parallelization_completion_status.py` owner as a
  `decomposition` subcommand and deleted the standalone decomposition-status
  wrapper. The default command remains the release closure artifact, while the
  subcommand preserves deterministic shard assignment, serial reconstruction
  identity, and claim-separation checks for lower-level parallelization work.
  Focused parallel unit/status tests, release-readiness tests, Python syntax,
  ruff, CLI help, shell subcommand smoke, and stale-reference checks passed.
  Tool Python files dropped to 239.

- 2026-07-08: consolidated the nonlinear-gradient variance-reduction and
  independent control-mean artifact builders into
  `tools/artifacts/build_nonlinear_gradient_evidence.py` with
  `variance-plan` and `control-mean` subcommands. The reusable evidence
  algorithms remain in `tools/campaigns/nonlinear_gradient_followup.py`, and
  the postprocess campaign now routes through the shared artifact owner. The
  focused nonlinear-gradient validation shard, Python syntax, ruff, CLI help,
  and stale-reference checks passed. Tool Python files dropped to 240.

- 2026-07-08: consolidated the nonlinear landscape-admission and
  campaign-admission wrappers into
  `tools/artifacts/build_nonlinear_transport_admission.py` with explicit
  `landscape` and `campaign` subcommands. The underlying physics/policy
  functions stay in `spectraxgk.diagnostics.stellarator_transport_reports`
  and `spectraxgk.objectives.vmec_transport_admission`; this removes two
  one-off artifact entry points without changing admission semantics. Focused
  nonlinear artifact tests, Python syntax, ruff, architecture, validation
  coverage, release-artifact manifest, and stale-reference checks passed for
  this tranche. Tool Python files dropped to 241.

- 2026-07-08: consolidated standalone quasilinear spectrum, spectrum-shape,
  and UQ ensemble scaling plotters into `tools/artifacts/plot_quasilinear_diagnostics.py`
  with `spectrum`, `shape-gate`, and `uq-ensemble-scaling` subcommands.
  Documentation command snippets and the quasilinear artifact test family now
  use the shared owner, and the tool-count no-regression baseline tightened
  to 242 Python files.
- 2026-07-08: folded the VMEC/Boozer aggregate line-search artifact
  builder into `tools/artifacts/build_vmec_boozer_aggregate_objective_gate.py`
  as a `line-search` subcommand. The finite-difference and line-search
  gate artifacts now share one objective-family owner, and the tool-count
  no-regression baseline tightened to 244 Python files.
- 2026-07-08: consolidated the three tracked reference-comparison panel
  builders into `tools/comparison/make_reference_panels.py` with `tokamak`,
  `publication`, and `summary` subcommands. The three one-script tests were
  collapsed into one family test, default docs paths now resolve from the
  repository root, and topology no-regression baselines were tightened to
  277 source, 241 test, and 245 tool Python files.
- 2026-07-08: fixed CI release-runner discovery after the test tree
  consolidation. `run_tests_fast.py` and `run_wide_coverage_gate.py` now
  discover nested `test_*.py` files recursively, their unit tests lock that
  behavior, and the fast-coverage workflow no longer requires the stale
  deleted `secondary.py` module entry while retaining the package-wide 95%
  coverage gate.
- 2026-07-07: added topology counts to
  `tools/package_architecture_manifest.toml` and extended
  `tools/release/check_package_architecture_manifest.py` so source, test, tool, flat
  test/tool, and installable-validation file counts are executable gates. The
  default mode fails on count regressions while reporting the remaining gap to
  the final targets; `--require-topology-targets` fails closed until all final
  file-count targets are met.
- 2026-07-07: removed four unreferenced probe/audit scripts from `main`:
  `tools/debug_vmec_nonzero_check.py`, `tools/diffrax_mismatch_sweep.py`,
  `tools/etg_physics_audit.py`, and `tools/hl_evolution.py`. Tool Python count
  dropped from 269 to 265, and the topology baseline was tightened to prevent
  reintroducing those files.
- 2026-07-07: added root navigation documents for `tools/` and `examples/` so
  future moves have explicit ownership rules. `tools/README.md` separates
  release gates, artifact builders, profilers, comparison utilities, and active
  campaigns from probes and local debugging. `examples/README.md` defines the
  user-facing example map and the required metadata for maintained examples.
- 2026-07-07: moved 181 release/tool-related test files out of the flat
  `tests/` root into `tests/release/` and `tests/tools/{artifacts,campaigns,
  comparison,generators,profiling}/`. Flat test files dropped from 322 to 141.
  CI path references, validation coverage manifest paths, and nested test
  repository-root lookups were updated. The moved release shard and full
  `tests/tools` shard passed locally under the five-minute cap.
- 2026-07-07: refreshed this plan and `docs/architecture_refactor_plan.rst`
  after the topology audit. The refined strategy is deletion-first: retire
  cETG/non-promoted paths, collapse flat tests/tools, move validation campaigns
  out of installable source, then refactor hot kernels behind profiler and
  identity gates.
- 2026-07-07: updated the architecture manifest so validation subpackages are
  no longer required permanent package domains, and updated the validation
  coverage manifest checker to accept nested `tests/**/test_*.py` fast tests
  after the test-tree folderization.
- 2026-07-07: retired the non-promoted reduced cETG runtime path from `main`.
  Deleted `src/spectraxgk/terms/reduced/`,
  `src/spectraxgk/workflows/reduced_models.py`, the cETG runtime example,
  the reduced-model parser tool, and cETG/reduced-model tests. Runtime
  `physics.reduced_model` values outside full-GK aliases now fail closed with a
  clear unsupported-model error. Source Python files dropped from 357 to 351,
  test Python files from 322 to 320, tool Python files from 265 to 264, flat
  test files from 141 to 139, and flat tool scripts from 265 to 264.

- 2026-07-07: added `tools/release/inventory_repository.py`, a reproducible
  tracked-file inventory/classification tool for the consolidation plan. Removed
  three unreferenced historical root tools (`calibrate_cyclone.py`,
  `extract_cyclone_reference.py`, and `freeze_gx_big_eigenfunction_bundle.py`).
  Tool Python files dropped from 264 to 262 and flat root tool scripts from 264
  to 261; the architecture manifest baselines were tightened accordingly.

- 2026-07-07: moved repository-size release gates into `tools/release/`:
  `audit_repository_size.py` and `check_repository_size_manifest.py`. CI,
  release workflow, docs, tests, and release-readiness snippets now reference
  the new paths. Total tool count stayed at 262, and flat root tool scripts
  dropped from 261 to 259.

- 2026-07-07: moved the package architecture manifest checker into
  `tools/release/check_package_architecture_manifest.py` and updated CI,
  release, docs, tests, technical-release status metadata, and release-readiness
  snippets. Total tool count stayed at 262, and flat root tool scripts dropped
  from 259 to 258.

- 2026-07-07: moved the release-readiness checker into
  `tools/release/check_release_readiness.py` and updated CI, release workflow,
  tests, release status metadata, and internal required-snippet checks. Total
  tool count stayed at 262, and flat root tool scripts dropped from 258 to 257.

- 2026-07-07: moved three small release-only gates into `tools/release/`:
  `check_release_version.py`, `check_release_artifact_manifest.py`, and
  `check_performance_optimization_manifest.py`. CI, release workflows, docs,
  tests, and release-readiness snippets now use the new paths. Total tool count
  stayed at 262, and flat root tool scripts dropped from 257 to 254.

- 2026-07-07: moved six release/claim-validation gates into `tools/release/`:
  `check_differentiable_refactor_manifest.py`,
  `check_validation_coverage_manifest.py`, `check_parallel_scaling_artifacts.py`,
  `check_quasilinear_calibration_inputs.py`,
  `check_quasilinear_promotion_guardrails.py`, and
  `check_vmec_boozer_differentiability_claim.py`. CI, release workflow, docs,
  tests, manifest references, and release-readiness snippets now use the new
  paths. Total tool count stayed at 262, and flat root tool scripts dropped
  from 254 to 248.

- 2026-07-07: moved the remaining thirteen root-level `check_*` gate
  scripts into `tools/release/`, including nonlinear-window, nonlinear-output,
  nonlinear-transport, VMEC/Boozer holdout, external-VMEC, and overdetermined
  nonlinear-gradient checkers. Command strings, docs, tests, workflow snippets,
  and tracked JSON metadata now reference the new paths. Total tool count stayed
  at 262, and flat root tool scripts dropped from 248 to 235.

- 2026-07-07: moved thirty-one explicit external-comparison/reference utilities
  into `tools/comparison/`, including `compare_*`, `generate_gx_*`,
  `make_gx_*`, and `probe_gx_*` scripts. Tests, docs, command strings, and
  sibling imports now use `tools/comparison/...` or `tools.comparison.*`. Total
  tool count increased by one package initializer to 263, and flat root tool
  scripts dropped from 235 to 204.

- 2026-07-07: moved ninety publication/readme/manuscript artifact builders
  into `tools/artifacts/`, including root-level `build_*` and `plot_*`
  scripts. Docs, tracked replay metadata, tests, CI/release commands, and
  internal imports now use `tools/artifacts/...` or `tools.artifacts.*`. Total
  tool count increased by one package initializer to 264, and flat root tool
  scripts dropped from 204 to 114.

- 2026-07-07: moved forty-six active campaign helpers into `tools/campaigns/`,
  including `run_*`, `write_*`, `design_*`, postprocessors, nonlinear-gradient
  summarizers, VMEC transport-metric helpers, and the renamed
  `audit_vmec_jax_boundary_chain.py`. Moved the local fast-test and wide
  coverage runners into `tools/release/`. Docs, examples, tracked command
  metadata, tests, and imports now use `tools/campaigns/...`,
  `tools.campaigns.*`, `tools/release/...`, or `tools.release.*`. Total tool
  count increased by one campaign package initializer to 265, and flat root
  tool scripts dropped from 114 to 66.

- 2026-07-07: moved twenty-one profiling reproducers plus profiler options into
  `tools/profiling/`. Performance docs, manifests, runtime profile commands,
  and profiling tests now use `tools/profiling/...` or `tools.profiling.*`.
  Total tool count increased by one profiling package initializer to 266, and
  flat root tool scripts dropped from 66 to 45.

- 2026-07-07: moved the remaining gate/generator artifact refreshers into
  `tools/artifacts/`, including parallelization gates, zonal/eigenfunction
  artifact generators, observed-order reports, Miller geometry generation, and
  optional nonlinear fast-path gates. The former generator-test shard
  now lives under `tests/tools/artifacts`. Total tool count stayed at 266, and
  flat root tool scripts dropped from 45 to 20.

- 2026-07-07: moved the four performance benchmark drivers into
  `benchmarks/performance/`: integrator, linear-solve, nonlinear-suite, and
  runtime/memory benchmark entry points. Docs, manifests, and tests now use
  `benchmarks/performance/...` or `benchmarks.performance.*`. Tool Python files
  dropped from 266 to 262, and flat root tool scripts dropped from 20 to 16.

- 2026-07-07: moved three documentation/release image compression helpers into
  `tools/artifacts/`. CI and artifact tests originally called separate docs
  and release preview compressors.

- 2026-07-07: closed the remaining flat-tool tranche. Artifact/reference
  helpers now live under `tools/artifacts`, RHS/ky diagnostics under
  `tools/comparison`, and the VMEC-JAX WOUT metadata patch helper under
  `tools/campaigns`. Deleted the unowned `cyclone_resolution_sweep.py` and
  `etg_eigenspectrum.py` probes from `main`. Tool Python files dropped from
  262 to 260, and flat root tool scripts dropped from 13 to 1
  (`tools/__init__.py`).

- 2026-07-07: moved the first safe root-test tranche into domain folders:
  unit core, diagnostics, geometry, linear, nonlinear, objectives, operators,
  parallel, quasilinear, solvers; integration runtime/examples; validation
  benchmarks/physics gates; and the profiling-options tool test. Test Python
  file count stayed at 320, but flat root Python files dropped from 139 to 47.
  The remaining flat tests are path-sensitive and should be moved only after
  replacing parent-depth assumptions with shared fixtures/helpers.

- 2026-07-07: moved the second path-sensitive root-test tranche after adding
  `tests/support/paths.py` and putting `tests/` on `sys.path` in
  `tests/conftest.py`. Runtime/example demos, benchmark manifests, release
  manifests, nonlinear-gradient gates, stellarator/VMEC gates, quasilinear
  gates, objective tests, VMEC backend helpers, and parallel artifact tests now
  live under their domain folders. Test Python files increased from 320 to 321
  because of the shared helper, while flat root Python files dropped from 47 to
  4 (`conftest.py` plus `test_cli.py`, `test_runtime_config.py`, and
  `test_runtime_runner.py`).

- 2026-07-07: moved the final runtime/executable root tests
  (`test_cli.py`, `test_runtime_config.py`, and `test_runtime_runner.py`) into
  `tests/integration/runtime` and updated CI, docs, and manifests. The flat
  root test topology target is now met: only `tests/conftest.py` remains at the
  root.

- 2026-07-07: consolidated thirteen one-file-per-gate parallel artifact tests
  into `tests/tools/artifacts/test_parallel_identity_gate_artifacts.py`. The
  new suite keeps the same physics/numerics contracts for velocity reduction,
  Hermite exchange/streaming, electrostatic field/drive/drift routes, linear
  RHS routes, independent `k_y` batching, logical CPU batching, and
  quasilinear runtime batching while sharing artifact writer assertions. Test
  Python files dropped from 321 to 309, and `tests/tools/artifacts` dropped
  from 101 to 89 files.

- 2026-07-07: consolidated seven VMEC/Boozer aggregate artifact tests
  into `tests/tools/artifacts/test_vmec_boozer_aggregate_artifacts.py`. The
  suite preserves aggregate objective, line-search, alpha/surface holdout,
  multi-point, second-equilibrium, CLI, fail-closed, physical-`k_y`, and
  artifact writer assertions while removing repeated dynamic-import scaffolding.
  Test Python files dropped from 309 to 303, and `tests/tools/artifacts`
  dropped from 89 to 83 files.


- 2026-07-07: consolidated twelve one-file-per-tool quasilinear plotting
  artifact tests into `tests/tools/artifacts/test_quasilinear_artifact_plots.py`.
  The suite preserves calibration, spectrum, spectrum-shape, saturation-rule,
  shape-aware saturation, candidate uncertainty, dataset sufficiency, model
  selection, screening-skill, stellarator-usefulness, and UQ ensemble scaling
  assertions while sharing the artifact-tool loader. Test Python files dropped
  from 303 to 292, and `tests/tools/artifacts` dropped from 83 to 72 files.


- 2026-07-07: consolidated ten W7-X and zonal-response artifact tests into
  `tests/tools/artifacts/test_w7x_artifact_panels.py`. The suite preserves
  W7-X reference overlay, zonal response, exact-state audit, fluctuation
  spectrum, closure ladder, contract audit, moment-tail, recurrence sweep,
  state-convention, and TEM-extension status assertions. Test Python files
  dropped from 292 to 283, and `tests/tools/artifacts` dropped from 72 to 63
  files.


- 2026-07-07: consolidated fifteen nonlinear artifact/report tests into
  `tests/tools/artifacts/test_nonlinear_artifact_reports.py`. The suite
  preserves matched transport, baseline-vs-optimized audits, nonlinear
  landscape/campaign/prelaunch gates, finite-difference window audits,
  transport-horizon classification, Laguerre nonlinear-mode output checks,
  nonlinear sharding production gates, external-VMEC convergence gates,
  feasibility pilots, RHS profiling, strong-scaling summaries, and nonlinear
  window-statistics assertions. Test Python files dropped from 283 to 269, and
  `tests/tools/artifacts` dropped from 63 to 49 files.


- 2026-07-07: consolidated five status/readiness artifact tests into
  `tests/tools/artifacts/test_status_readiness_artifacts.py`. The suite
  preserves manuscript-readiness, open-lane, parallelization completion,
  pre-manuscript closure status, and pre-manuscript runbook assertions while
  centralizing script loading. Test Python files dropped from 269 to 265, and
  `tests/tools/artifacts` dropped from 49 to 45 files.


- 2026-07-07: consolidated seven VMEC miscellaneous artifact tests into
  `tests/tools/artifacts/test_vmec_misc_artifact_reports.py`. The suite
  preserves external-VMEC replicate ensembles, boundary-chain collection,
  candidate-screen gates, state-control bracket status, state-to-input mapping,
  external-VMEC time-horizon gates, and VMEC-JAX equilibrium inventory
  assertions. Test Python files dropped from 265 to 259, and
  `tests/tools/artifacts` dropped from 45 to 39 files.

- 2026-07-07: consolidated five VMEC/Boozer gradient, parity, nonlinear-window,
  and production-holdout artifact tests into
  `tests/tools/artifacts/test_vmec_boozer_artifact_reports.py`. The suite
  preserves mode-21 parity matrices, reduced gradient holdouts, nonlinear-window
  finite-difference audits, reduced nonlinear-gradient artifacts, and
  production holdout promotion/fail-closed assertions. Test Python files dropped
  from 259 to 255, and `tests/tools/artifacts` dropped from 39 to 35 files.

- 2026-07-07: consolidated nine linear-validation artifact tests into
  `tests/tools/artifacts/test_linear_validation_artifact_reports.py`. The suite
  preserves QI branch-refinement gates, TEM branch-audit reports, imported-linear
  last-value tables, W7-X zonal reference digitization, linear-RHS zero-norm
  window gates, KBM branch/eigenfunction artifacts, observed-order gates, and
  validation-gate index assertions. Test Python files dropped from 255 to 247,
  and `tests/tools/artifacts` dropped from 35 to 27 files.

- 2026-07-07: renamed the KBM extractor comparison utility from a probe-oriented
  name to `tools/comparison/audit_gx_kbm_extractors.py`, with the corresponding
  test moved to `tests/tools/comparison/test_audit_gx_kbm_extractors.py`. This
  keeps the active comparison audit in the explicit comparison-tool namespace
  while removing a misleading experimental/probe filename from `main`.

- 2026-07-07: renamed the active zonal-response and nonlinear-feasibility
  artifact builders from pilot-oriented tool filenames to
  `tools/artifacts/generate_miller_zonal_response_panel.py` and
  `tools/artifacts/plot_nonlinear_feasibility_panel.py`. Generated artifact
  filenames and payload kinds remain scoped where they intentionally describe
  open feasibility evidence, but the executable tool names now reflect the
  maintained panel-builder role.

- 2026-07-07: merged duplicated documentation and release PNG preview
  compressors into `tools/artifacts/compress_previews.py` and consolidated
  their tests into `tests/tools/artifacts/test_compress_previews.py`. The
  unified tool supports `--mode docs` and `--mode release`, keeps manifest
  skip/target behavior explicit, and removes one tool script plus one test
  wrapper. Tool Python files dropped from 260 to 259, tests from 247 to 246,
  `tools/artifacts` from 126 to 125, and `tests/tools/artifacts` from 27 to
  26 files.





- 2026-07-07: moved the external-VMEC nonlinear holdout runbook implementation
  out of `src/spectraxgk/validation` and merged it into the existing
  `tools/artifacts/build_external_vmec_holdout_runbook.py` artifact builder.
  The campaign helper is no longer exported from the package API, source Python
  files dropped from 348 to 347, installable validation files dropped from 82
  to 81, and tool count stayed flat at 259 because no new tool entry point was
  added.

- 2026-07-07: moved W7-X/Miller zonal trace loading, normalization,
  reference-table, and tail-metric helpers from `src/spectraxgk/validation` to
  `src/spectraxgk/diagnostics/zonal_validation.py`. Importers/docs/manifests
  now use the diagnostics owner, and installable validation files dropped from
  83 to 82.

- 2026-07-07: moved autodiff finite-difference, covariance, dense-operator,
  and isolated eigenbranch sensitivity helpers out of
  `src/spectraxgk/validation` into
  `src/spectraxgk/objectives/autodiff_validation.py`. The old
  `validation/autodiff.py` and `validation/autodiff_finite_difference.py`
  files were deleted, examples/objectives/docs/manifests now use the objectives
  owner, source Python files dropped from 349 to 348, and installable validation
  files dropped from 85 to 83.

- 2026-07-07: moved validation gate dataclasses, scalar tolerance evaluation,
  JSON serialization, and report builders out of `src/spectraxgk/validation`
  into `src/spectraxgk/diagnostics/validation_gates.py`. The old
  `validation/gates.py`, `validation/gate_types.py`, and
  `validation/gate_reports.py` files were deleted, importers/docs/manifests now
  use the diagnostics owner, source Python files dropped from 351 to 349, and
  installable validation files dropped from 88 to 85.

- 2026-07-07: tightened the refactor plan after a fresh repository audit. The
  new planning reset records the current `b503c0d4` topology, clarifies that
  obsolete branches are not present in this clone, identifies validation-in-src,
  tool-family sprawl, test-family sprawl, and non-promoted examples/artifacts as
  the actual complexity drivers, and defines the keep/move/delete test for every
  file. The architecture and testing docs now point to this ownership model, and
  `tests/README.md` defines how to add or merge tests without increasing
  file-count and maintenance debt.


- 2026-07-07: extracted `validation.nonlinear_transport` from the installable
  package. Production nonlinear optimization promotion diagnostics now live in
  `spectraxgk.diagnostics.nonlinear_transport_optimization`; replicated
  transport-window spread diagnostics live in
  `spectraxgk.diagnostics.nonlinear_replicates`; targeted seed/timestep
  follow-up planning lives in `tools/campaigns/nonlinear_replicate_followup.py`.
  Deleted twelve zero-reference obsolete tool/probe scripts from artifact,
  comparison, and profiling folders. Current counts: `src/spectraxgk` 318
  Python files, `src/spectraxgk/validation` 46, `tools` 248.


- 2026-07-07: extracted `validation.stellarator` from the installable package.
  VMEC candidate gates now live in `spectraxgk.objectives.vmec_candidate_admission`;
  transport-admission policy, sample coverage, and candidate selection live in
  `spectraxgk.objectives.vmec_transport_admission`; nonlinear landscape,
  prelaunch, campaign-admission, and matched-audit redesign reports live in
  `spectraxgk.diagnostics.stellarator_transport_reports`. Current counts:
  `src/spectraxgk` 312 Python files and `src/spectraxgk/validation` 37, leaving
  benchmark validation as the only remaining installable validation family.

- 2026-07-07: completed a benchmark/tool/test consolidation audit after the
  stellarator extraction. The remaining installed benchmark-validation family is
  36 files and about 17.4k LOC; root `benchmarks/` is small at 12 Python files
  and about 1.6k LOC. The next tranche must first freeze the
  `spectraxgk.benchmarks` public API, remove private helper re-exports as a
  compatibility target, move benchmark config/diagnostic helpers to physical
  owners, and only then delete `src/spectraxgk/validation/benchmarks`.

- 2026-07-07: started benchmark-family extraction by moving benchmark case
  presets out of `src/spectraxgk/validation/benchmarks/case_configs.py`.
  `spectraxgk.config` now directly owns Cyclone, ETG, kinetic-electron, KBM,
  and TEM preset dataclasses; the validation case-config module was deleted.
  Current counts: `src/spectraxgk` 311 Python files,
  `src/spectraxgk/validation` 36, and `validation/benchmarks` 35.

- 2026-07-07: moved benchmark eigenfunction normalization, phase alignment,
  comparison metrics, and reference-bundle IO from
  `validation/benchmarks/harness_eigenfunctions.py` into
  `diagnostics/modes.py`. The validation eigenfunction helper module was
  deleted without adding a replacement file. Current counts:
  `src/spectraxgk` 310 Python files, `src/spectraxgk/validation` 35, and
  `validation/benchmarks` 34.

- 2026-07-07: moved benchmark diagnostic time-series loading, late/leading/
  explicit-window helpers, analytic-signal construction, and real-FFT ky-grid
  inference from `validation/benchmarks/harness_timeseries.py` into
  `diagnostics/validation_gates.py`. NetCDF loading stays lazy inside the
  loader so normal diagnostics imports remain lightweight. Current counts:
  `src/spectraxgk` 309 Python files, `src/spectraxgk/validation` 34, and
  `validation/benchmarks` 33.

- 2026-07-07: moved benchmark zonal-flow residual/GAM metric extraction from
  `validation/benchmarks/harness_zonal_metrics.py` into
  `diagnostics/zonal_validation.py`, reusing the existing zonal-validation
  owner. Current counts: `src/spectraxgk` 308 Python files,
  `src/spectraxgk/validation` 33, and `validation/benchmarks` 32.

- 2026-07-07: moved late-time linear, nonlinear-window, heat-flux
  convergence, observed-order, and branch-continuity metric extraction into
  `diagnostics/validation_gates.py`, deleting the old benchmark trace-metrics
  helper. The public benchmark facade still preserves documented imports and
  monkeypatch seams. Current counts: `src/spectraxgk` 307 Python files,
  `src/spectraxgk/validation` 32, and `validation/benchmarks` 31.

- 2026-07-07: moved fit-signal selection, mode-only trace extraction,
  auto-fit scoring, and diagnostic growth-rate normalization from the benchmark
  validation package into `diagnostics/growth_rates.py`. Current counts:
  `src/spectraxgk` 306 Python files, `src/spectraxgk/validation` 31, and
  `validation/benchmarks` 30.

- 2026-07-07: moved scan batching and streaming-window helpers into
  `validation/benchmarks/defaults.py`, deleting the separate batching helper module.
  Current counts: `src/spectraxgk` 305 Python files,
  `src/spectraxgk/validation` 30, and `validation/benchmarks` 29.

- 2026-07-07: moved benchmark solver-selection policy, KBM explicit-solver
  locks, multi-target Krylov selection, and benchmark midplane indexing into
  `validation/benchmarks/defaults.py`, deleting the separate solver-policy
  helper module. Current counts: `src/spectraxgk` 304 Python files,
  `src/spectraxgk/validation` 29, and `validation/benchmarks` 28.

- 2026-07-07: moved benchmark reference containers, CSV loaders, and
  reference comparison helpers into `validation/benchmarks/defaults.py`,
  deleting the separate reference helper module. Current counts:
  `src/spectraxgk` 303 Python files, `src/spectraxgk/validation` 28, and
  `validation/benchmarks` 27.

- 2026-07-07: moved benchmark species-to-`LinearParams` builders,
  reference hypercollision constants, and linked-boundary damping policy into
  `validation/benchmarks/defaults.py`, deleting the separate species helper
  module. Current counts: `src/spectraxgk` 302 Python files,
  `src/spectraxgk/validation` 27, and `validation/benchmarks` 26.

- 2026-07-07: moved benchmark Gaussian/moment initial-condition builders and
  kinetic reference seed policy into `validation/benchmarks/defaults.py`,
  deleting the separate initialization helper module. Current counts:
  `src/spectraxgk` 301 Python files, `src/spectraxgk/validation` 26, and
  `validation/benchmarks` 25.

- 2026-07-07: moved benchmark scan/mode orchestration and
  representative eigenfunction extraction into `spectraxgk.benchmarks`,
  deleting the separate harness-scan helper module. Current counts:
  `src/spectraxgk` 300 Python files, `src/spectraxgk/validation` 25, and
  `validation/benchmarks` 24.

## Immediate Next Steps

1. Shrink `spectraxgk.benchmarks` without adding another source forest:
   - define the stable facade API/result contracts;
   - move benchmark-only case workflows to root `benchmarks/` drivers or delete
     unsupported historical branch paths;
   - update benchmark tests to patch stable contracts instead of facade globals.
2. Collapse the biggest tests without weakening assertions:
   - replace repeated runtime, benchmark, artifact, and comparison monkeypatch
     tests with shared fake-runner fixtures and parametrized contract tables;
   - remove tests that only preserve deleted legacy behavior.
3. Collapse artifact and campaign tools by family:
   - replace one-file-per-panel and one-file-per-status scripts with
     manifest-driven family commands;
   - classify or move the current comparison-code naming blockers:
     `tools/comparison/run_reference_linear_stress_matrix.py`, `tools/comparison/fixtures/etg_ky25_reference.in`,
     and `tools/comparison/fixtures/etg_runtime_ky15_reference.in`.
4. Run a docs/static and examples deletion audit:
   - keep only README/docs/release-manifest referenced evidence;
   - remove stale pilot/probe/reduced-window companions from `main`;
   - document root `benchmarks/` results without storing raw long-run output.
5. Collapse source domains after the benchmark/test/tool tranches:
   - merge `geometry_backends` into `geometry`;
   - resolve `terms` versus `operators` into one obvious mathematical-kernel
     ownership model;
   - merge tiny objective-family wrappers while preserving AD/FD and physics
     gates.
6. Profile and optimize only after the owning modules are stable:
   - quickstart compile/progress path;
   - linear cache/RHS setup;
   - nonlinear RHS/bracket/field solve;
   - diagnostics IO and VMEC/Boozer transforms;
   - every performance change needs before/after profiler artifacts and
     numerical identity or physics gates.

## Completion Definition

This plan is complete only when current repository evidence proves all of the
following:

- `src/spectraxgk` has at most 100 Python files.
- `tests` has fewer than 100 Python files.
- `tools` has fewer than 100 Python files.
- Validation/campaign code no longer lives in the installable package except for
  tiny reusable metric helpers.
- cETG and other obsolete/non-promoted paths are removed from `main`, except
  fail-closed unsupported-input validation where needed.
- Examples have a clear README and only current runnable/promoted workflows.
- Benchmarks are root-level, small, documented, and free of raw outputs.
- README and docs match the new structure and claim scope.
- Package-wide coverage remains at least 95%.
- Fast local tests, release gates, docs build, package build, and CI pass.
- Runtime/memory and speedup claims are backed by fresh profiler artifacts and
  numerical identity or physics gates.

- 2026-07-07: consolidated nonlinear transport-window diagnostics from five
  quasilinear validation modules into
  `src/spectraxgk/diagnostics/transport_windows.py`: configuration contracts,
  late-window statistics, CSV/summary readers, promotion-readiness checks, and
  replicated ensemble gates now share one diagnostics owner. Deleted
  `validation/quasilinear/window_config.py`, `window_statistics.py`,
  `window_io.py`, `window_promotion.py`, and `window_ensemble.py`. Source
  Python files dropped from 347 to 343, installable validation files dropped
  from 81 to 76, and tool/test file counts stayed flat.

- 2026-07-07: consolidated quasilinear calibration point/report math,
  spectrum integration, nonlinear-window CSV/NetCDF ingestion, and report JSON
  writing from `validation/quasilinear/calibration_core.py`,
  `calibration_spectrum.py`, and `calibration_io.py` into
  `src/spectraxgk/diagnostics/quasilinear_calibration.py`. Source Python files
  dropped from 343 to 341, installable validation files dropped from 76 to 73,
  and public API exports continue through `spectraxgk.api.validation`.

- 2026-07-07: moved external-VMEC holdout admission policy out of the
  installable validation package and into the existing
  `tools/release/check_quasilinear_calibration_inputs.py` release
  checker, with artifact builders importing from that tool owner. Source Python
  files dropped from 341 to 340, installable validation files dropped from 73
  to 72, and tool count stayed flat.

- 2026-07-07: removed the remaining quasilinear validation subpackage by
  consolidating model-selection input normalization and claim-boundary status
  builders into `src/spectraxgk/diagnostics/quasilinear_model_selection.py`.
  Source Python files dropped from 340 to 338, installable validation files
  dropped from 72 to 69, and old quasilinear-validation import paths are no longer present.

- 2026-07-07: extracted the nonlinear turbulent-gradient validation subpackage
  out of installable validation code. Reusable fail-closed evidence, finite-
  difference, replicated-window, candidate-ranking, bracket-sweep, and evidence-
  gap diagnostics now live in
  `src/spectraxgk/diagnostics/nonlinear_gradient_evidence.py`. Follow-up,
  QL-seed, state-runbook, composite-control, and control-variate campaign
  planning now live in `tools/campaigns/nonlinear_gradient_followup.py` because
  they are repository campaign tooling, not runtime library code. Deleted the
  old `src/spectraxgk/validation/nonlinear_gradient/` package and the
  unreferenced one-off `tools/campaigns/run_nonlinear_cyclone.py` script so the
  tool-count gate remains non-regressing. Source Python files dropped from 338
  to 322 and installable validation files dropped from 69 to 52. Focused
  nonlinear-gradient tests and manifest checks pass for this tranche.

- 2026-07-07: folded Cyclone scan explicit-time reselection, Krylov branch-following, and trace-seed helpers into `src/spectraxgk/validation/benchmarks/cyclone_scan_branches.py`, deleting `cyclone_scan_explicit.py`, `cyclone_scan_krylov.py`, and `cyclone_scan_seed.py`. Cyclone scan validation now has one patchable branch-policy owner while preserving the public `spectraxgk.benchmarks` API. Source Python files dropped to 297, installable validation files to 22, and validation benchmark files to 21.

- 2026-07-07: consolidated small benchmark contract tests into `tests/validation/benchmarks/test_benchmark_contracts.py`, deleting the separate scan-policy, results-manifest, runtime-memory, and reference-consistency test files. Benchmark validation coverage is unchanged, but the benchmark test folder is smaller and CI now points at the consolidated contract shard. Test Python files dropped to 243.

- 2026-07-07: folded `src/spectraxgk/validation/benchmarks/cyclone_linear_paths.py` into `cyclone_linear.py` and removed the hook-sync compatibility layer. Cyclone single-mode Krylov and time-integration policies now live with the runner that owns their patchable numerical hooks. Source Python files dropped to 296, installable validation files to 21, and validation benchmark files to 20.

- 2026-07-07: folded the old ETG scan path helper into the ETG scan owner at that time, removing another hook-sync path module. ETG scan Krylov continuation, streaming, configured-history, direct-fit, auto-fit, and Krylov-fallback helpers later moved into the public benchmark facade with the rest of the ETG runner. Source Python files dropped to 295, installable validation files to 20, and validation benchmark files to 19.

- 2026-07-07: folded `src/spectraxgk/validation/benchmarks/kbm_linear_paths.py` into `kbm_linear.py`, removing the KBM single-ky hook-sync path module. KBM explicit-time and Krylov target-selection policies now live with the runner and remain covered by the same benchmark branch tests. Source Python files dropped to 294, installable validation files to 19, and validation benchmark files to 18.

- 2026-07-07: folded `src/spectraxgk/validation/benchmarks/kbm_beta_solver_paths.py` into `kbm_beta.py`. Fixed-ky KBM beta explicit-time, Krylov, and saved-time sample policies now live with the beta-scan owner instead of a separate installable path module. Source Python files dropped to 293, installable validation files to 18, and validation benchmark files to 17.

- 2026-07-07: folded the old TEM path helper into the TEM benchmark owner at that time. TEM remains scoped as a validation lane, but its scan/single-mode path helpers now live with the public benchmark owner instead of a separate installable path module. Source Python files dropped to 292, installable validation files to 17, and validation benchmark files to 16.

- 2026-07-07: consolidated the standalone device-z fused-RHS profiling script into
  `tools/profiling/profile_device_z_pencil_transport_window.py --mode rhs`,
  deleting the old RHS-only profiler while keeping the existing RHS profile
  artifact schema and manifest metric. Tool
  Python files dropped to 247 and `tools/profiling` dropped to 19 scripts.
  Focused profiling tests plus performance, architecture, and differentiable-
  refactor manifest checks passed for this tranche.

- 2026-07-07: folded the TEM benchmark runners into the public
  `spectraxgk.benchmarks` facade and deleted the installable validation
  module. TEM setup, parameter construction, species validation, Krylov
  path, saved-time fit path, streaming scan branch, scan batch loop, and
  explicit hook bundle now live beside `run_tem_linear` and `run_tem_scan`.
  Source Python files dropped to 283, installable validation files to 8, and
  validation benchmark files to 7. Focused py_compile, ruff, validation
  coverage, differentiable-refactor, public facade smoke, and bounded
  benchmark-validation shard checks passed for this tranche.


- 2026-07-07: folded the KBM single-ky and fixed-beta benchmark runners into
  the public `spectraxgk.benchmarks` facade and deleted their installable
  validation modules. KBM explicit-time diagnostics, saved-time fits,
  multi-target Krylov branch selection, continuation policy, per-beta sample
  construction, and `run_kbm_linear`/`run_kbm_beta_scan` now live beside the
  public `run_kbm_scan` wrapper. Source Python files dropped to 281,
  installable validation files to 6, and validation benchmark files to 5.


- 2026-07-07: folded the remaining Cyclone benchmark defaults, single-mode
  runner, scan branch policies, and scan runner into the public
  `spectraxgk.benchmarks` facade and deleted `src/spectraxgk/validation`.
  Source Python files dropped to 275 and installable validation files dropped
  to 0. The next benchmark refactor step is no longer validation-package
  removal; it is splitting the oversized public facade into clearer benchmark
  data, fit-policy, and case-runner owners while preserving the documented API.

- 2026-07-07: folded the ETG single-ky and ky-scan benchmark runners into
  the public `spectraxgk.benchmarks` facade and deleted their installable
  validation modules. ETG setup, electrostatic term defaults, Krylov forwarded
  keys, continuation shifts, streaming Diffrax fitting, configured-history
  fitting, direct/auto/fallback time fits, and scan batching now live beside
  `run_etg_linear` and `run_etg_scan`. Source Python files dropped to 284,
  installable validation files to 9, and validation benchmark files to 8.
  Focused py_compile, ruff, validation coverage, differentiable-refactor,
  public facade smoke, and bounded benchmark-validation shard checks passed
  for this tranche.

- 2026-07-07: folded the kinetic-electron ky-scan benchmark runner into
  the public `spectraxgk.benchmarks` facade and deleted its installable
  validation module. Scan batching, kinetic setup normalization, streaming
  Diffrax fitting, sampled-history fitting, time-config routing, and Krylov
  scan rows now live beside the public `run_kinetic_scan` API and share the
  kinetic species-index contract already in the facade. Source Python files
  dropped to 286, installable validation files to 11, and validation benchmark
  files to 10. Focused py_compile, ruff, validation coverage, differentiable-
  refactor, public facade smoke, and bounded benchmark-validation shard checks
  passed for this tranche.

- 2026-07-07: folded the kinetic-electron single-ky benchmark runner into
  the public `spectraxgk.benchmarks` facade and deleted its installable
  validation module. Kinetic-electron setup normalization, species-index
  validation, selected-state construction, Krylov solving, configured and
  unconfigured time-history integration, sampled-signal fitting, and result
  packing now live beside the public `run_kinetic_linear` API. Source Python
  files dropped to 287, installable validation files to 12, and validation
  benchmark files to 11. Focused py_compile, ruff, validation coverage,
  differentiable-refactor, stale-reference, and public facade smoke checks
  passed for this tranche.

- 2026-07-07: folded benchmark scan-window, fit-signal normalization, and
  ky-batching policy into `validation/benchmarks/defaults.py`, deleting
  the previous standalone scan-policy module without changing the public
  `spectraxgk.benchmarks` facade. Source Python files dropped to 291,
  installable validation files to 16, and validation benchmark files to 15.
  Bounded benchmark-validation shard, ruff, validation coverage, architecture,
  differentiable-refactor, technical-release-status, stale-reference, and diff
  checks passed for this tranche.

- 2026-07-07: folded the KBM fixed-beta ky-scan wrapper into the
  public `spectraxgk.benchmarks` facade and deleted its installable validation
  module. `run_kbm_scan` still forwards per-mode arrays to the beta-scan runner,
  but branch tests now patch the public facade rather than a validation package
  path. Source Python files dropped to 288, installable validation files to 13,
  and validation benchmark files to 12. Bounded benchmark shard, full-source
  mypy, ruff, py_compile, validation coverage, architecture, differentiable-
  refactor, technical-release-status, stale-reference, and diff checks passed
  for this tranche.

- 2026-07-07: folded benchmark scan/eigenfunction orchestration and
  validation-gate re-exports into the public `spectraxgk.benchmarks` facade,
  then deleted the old installable benchmark harness module. Gate metrics still
  live in `diagnostics.validation_gates` and eigenfunction helpers still live in
  `diagnostics.modes`; the facade now exposes the documented benchmark workflow
  imports directly. Source Python files dropped to 289, installable validation
  files to 14, and validation benchmark files to 13. Focused benchmark, physics
  gate, linear-helper, terms-integrator, ruff, py_compile, validation coverage,
  architecture, differentiable-refactor, technical-release-status, stale-reference,
  and diff checks passed for this tranche.

- 2026-07-07: folded the secondary-slab staged benchmark workflow into the
  public `spectraxgk.benchmarks` facade and deleted its installable
  `validation.benchmarks` module. The root secondary benchmark and comparison
  tool now import the public facade directly. Source Python files dropped to
  290, installable validation files to 15, and validation benchmark files to 14.
  Focused secondary tests, ruff, py_compile, validation coverage, architecture,
  differentiable-refactor, technical-release-status, stale-reference, and diff
  checks passed for this tranche.

- 2026-07-07: contracted the largest benchmark branch test by introducing
  local reusable grid/geometry/cache/growth-normalization scaffolds inside
  `tests/validation/benchmarks/test_benchmarks_runner_branches.py` and replacing
  repeated Cyclone, ETG, kinetic-electron, and TEM setup blocks. The file
  dropped from 2975 to 2690 lines without adding new test files. The focused
  branch-test shard, the full bounded `tests/validation/benchmarks` shard, ruff,
  architecture manifest, differentiable-refactor manifest, and `git diff
  --check` passed for this tranche.

- 2026-07-07: split shared benchmark mechanics out of the large public
  `spectraxgk.benchmarks` facade into `spectraxgk.benchmarking.shared`.
  Reference containers/loaders, scan-window policy, normalization constants,
  KBM solver defaults, reference hypercollision/end-damping policy, species
  parameter helpers, and initial-condition builders now have one internal
  owner while the public facade keeps legacy/documented imports stable.
  `benchmarks.py` dropped from about 13.96k to about 13.21k lines; source file
  count rises from 275 to 277 for this tranche, which is acceptable because the
  added package replaces hidden complexity inside a monolithic facade. Next
  benchmark-refactor tranche should move one complete case family behind the
  facade rather than adding ad hoc helper modules.

- 2026-07-07: checked the nonlinear transport matrix lane while office SSH was
  unavailable. Local docs only contain strict negative broad-promotion evidence:
  accepted QA/ESS is recorded from office as 9/18 passed samples with 9.18%
  mean reduction but fails the current broad all-sample pass-fraction policy;
  projected 1e-3 and 5e-4 reports are early failed. Generated one ignored
  target-aware fallback matrix launch package at
  `tools_out/nonlinear_transport_matrix_targeted/projected_0p001_targeted/`
  using the local strict QA baseline and projected-weight-1e-3 VMEC files. The
  package covers 18 surface/field-line/ky samples, 108 final-horizon outputs,
  `t=[1100,1500]`, final-time target checks, lock files, and two GPU split
  scripts. The target-time progress checker correctly reports 0/108 confirmed
  outputs until office runs the scripts. Do not promote this lane or import a
  portfolio artifact until the actual matrix report is fetched/regenerated and
  passes the configured policy.

- 2026-07-08: centralized test-side script loading through `tests/support/paths.py`
  across release, profiling, comparison, runtime/objective/parallel,
  nonlinear-validation, quasilinear-validation, stellarator-validation, and
  nonlinear-gradient validation shards. This removed repeated direct
  `importlib.util.spec_from_file_location` blocks from tests, leaving only
  optional-dependency discovery and a stdlib-only production loader. Focused
  shards passed locally: `tests/release`, `tests/tools/profiling`, targeted
  runtime/objective/parallel tests, nonlinear validation, quasilinear +
  stellarator validation, nonlinear-gradient validation, ruff, py_compile, and
  the package architecture manifest.

- 2026-07-08: consolidated five one-file release-hygiene tests into
  `tests/release/test_release_hygiene_gates.py`: release-version checks,
  release-artifact manifest checks, repository-size manifest checks,
  repository-size audit checks, and technical-release-status checks. The
  `tests/release` shard passed locally, test Python files dropped from 241 to
  237, and the architecture manifest still passes the no-regression policy.
  Remaining topology targets are still open: source 277/100, tests 237/99,
  tools 234/99. Next structural target is another domain-level test/tool
  consolidation, followed by shrinking the oversized public
  `spectraxgk.benchmarks` facade without adding another source-file forest.

- 2026-07-08: consolidated six small campaign command tests into
  `tests/tools/campaigns/test_campaign_gate_commands.py`, covering device
  parity, VMEC roundtrip, restart parity, benchmark refresh, imported-linear
  targeted audit, and KBM low-ky audit command contracts. Updated the
  validation-coverage manifest to reference the consolidated test. The
  `tests/tools/campaigns` shard, validation-coverage manifest, ruff,
  py_compile, and package architecture manifest passed locally. Test Python
  files dropped further to 232 while source and tool counts remain unchanged.

- 2026-07-08: consolidated four unreferenced small artifact plot/report tests
  into `tests/tools/artifacts/test_artifact_plot_smoke.py`, covering
  independent-ky scaling summaries, QA ITG README panel pending-artifact
  behavior, quasilinear residual-anatomy fail-closed sidecars, and stellarator
  optimization UQ plots. The full `tests/tools/artifacts` shard, ruff,
  py_compile, and package architecture manifest passed locally. Test Python
  files dropped to 229; source and tool file counts are unchanged.

- 2026-07-08: tightened `tools/package_architecture_manifest.toml` topology
  baselines after the consolidation pass so future commits cannot regress above
  the new current counts: 277 source files, 229 test files, and 234 tool files.
  The final topology targets remain unchanged at 100 source, 99 tests, and
  99 tools.

- 2026-07-08: consolidated the referenced zonal/transport artifact tests into
  `tests/tools/artifacts/test_zonal_transport_artifacts.py`, covering matched
  nonlinear audit redesign, zonal output plotting, reduced zonal objective
  gate artifacts, and the Miller zonal response panel. Updated
  `tools/validation_coverage_manifest.toml`,
  `tools/differentiable_refactor_manifest.toml`, and
  `docs/stellarator_optimization.rst` to point at the consolidated test. The
  focused artifact test, validation-coverage manifest, differentiable-refactor
  manifest, ruff, py_compile, and package architecture manifest passed locally.
  Test Python files dropped to 226, and the topology baseline was tightened to
  preserve that reduction.

- 2026-07-08: consolidated four small campaign manifest/report tests into
  `tests/tools/campaigns/test_campaign_manifest_reports.py`, covering QA
  t=1500 postprocessing manifests, W7-X zonal closure sweep manifests,
  external-VMEC holdout selection, and nonlinear replicate-spread summaries.
  Updated the validation-coverage manifest to use the consolidated path. The
  full `tests/tools/campaigns` shard, validation-coverage manifest, ruff,
  py_compile, and package architecture manifest passed locally. Test Python
  files dropped to 223, and the topology baseline was tightened accordingly.

- 2026-07-08: consolidated the two small eigenfunction plotting tools into
  `tools/artifacts/plot_eigenfunction_diagnostics.py` with `overlap-summary`
  and `reference-overlay` subcommands. Updated
  `docs/manuscript_figures.rst` and added focused tests for both subcommands
  to `tests/tools/artifacts/test_artifact_plot_smoke.py`. Ruff, focused
  artifact tests, py_compile, and the package architecture manifest passed
  locally. Tool Python files dropped to 233, and the topology baseline was
  tightened to preserve the reduction.
