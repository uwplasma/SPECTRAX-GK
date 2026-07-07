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

## Current Audited State

Last audited: 2026-07-07 on `main`.

- Latest local head: `7817628e Bump version to 1.6.10`.
- Latest tag at local head: `v1.6.10`.
- Git state at audit: clean `main`, tracking `origin/main`.
- Latest GitHub `main` CI, dependency graph, release workflow, and PyPI publish
  for `v1.6.10` passed.
- Active local/remote branches: only `main` and `origin/main`.
- Closed obsolete experimental PRs remain closed: #4, #5, and #6.
- Tracked repository size is acceptable for now; no tracked file is above 1 MB.
  The largest tracked file is `docs/_static/qa_low_turbulence_comparison.json`.
- Current tracked file counts:
  - `src/spectraxgk`: 362 tracked files, 357 Python files, about 109k lines.
  - `tests`: 322 Python files, about 94k lines.
  - `tools`: 282 tracked files, 265 Python files, about 101k lines.
  - `examples`: 83 tracked files, 43 Python files, about 10k lines.
  - `benchmarks`: 13 tracked files, 7 Python files, about 1k lines.
- Source-package Python file counts by domain:
  - `validation`: 88 files.
  - `objectives`: 38 files.
  - `operators`: 34 files.
  - `solvers`: 34 files.
  - `geometry`: 25 files.
  - `workflows`: 25 files.
  - `terms`: 21 files.
  - `artifacts`: 18 files.
  - `geometry_backends`: 18 files.
  - `diagnostics`: 16 files.
  - `parallel`: 12 files.
  - `api`: 11 files.
  - root facades: 9 files.
  - `core`: 6 files.
  - `utils`: 2 files.
- Source-function size is no longer the main problem: there are currently no
  source functions at or above 90 lines and none in the 80-89 line band.
- The main maintainability problem is repository topology: validation, campaign,
  plotting, profiling, release, comparison, and debug code has grown to roughly
  the same size as the solver and is spread across flat `tests/` and `tools/`
  namespaces.

## Hard Targets For The Refactor

These targets are release gates for this refactor. They are intentionally strict
because the goal is not another incremental split, but a much smaller and more
usable codebase.

| Area | Current | Target | Requirement |
| --- | ---: | ---: | --- |
| Installable source Python files | 357 | <= 100 | Move validation/campaign code out of `src`; consolidate domain modules. |
| Test Python files | 322 | < 100 | Reorganize and parametrize tests by domain; merge one-file-per-script tests. |
| Tool Python files | 265 | < 100 | Keep release gates, artifact builders, profilers, and comparison entry points only. |
| Root public facades | 9 | <= 8 | Keep only user-facing facades; no new root prefix modules. |
| `src/spectraxgk/validation` package | 88 | 0-5 | Remove installable validation campaigns; keep only tiny public metric helpers if necessary. |
| Legacy/non-promoted paths | many | 0 promoted by accident | Delete from `main` or move to a draft PR/experiment branch. |
| Default local test runtime | variable | < 5 min | Keep local gates bounded; long physics campaigns stay explicit. |
| Wide package coverage | >= 95% gate | >= 95% | Preserve or improve coverage after consolidation. |
| README/docs claims | scoped | current and scoped | No unsupported speedup, optimization, or validation claims. |

A file-count reduction that merely hides code in larger files is not acceptable.
A successful consolidation must also reduce duplicated policy, redundant tests,
patch-heavy test design, broad public exports, and unclear ownership.

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

- The cETG reduced-model path is legacy/non-promoted for the simplified next
  version. It is currently wired through source, examples, docs, and tests. The
  default plan is to remove it from `main`. If it must be preserved for research
  archaeology, move it to a draft PR or external branch with clear non-release
  status. Do not keep cETG as a hidden compatibility burden in the core runtime.

## Test Consolidation Plan

Current problem: `tests/` has 322 top-level files, including 188 files that
mostly test individual tool scripts. This is not maintainable.

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

Current problem: `tools/` has 269 Python scripts in one flat namespace.

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

Current problem: `src/spectraxgk` has 357 Python files, and 88 are validation
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
2. Move `src/spectraxgk/validation/nonlinear_gradient`,
   `src/spectraxgk/validation/nonlinear_transport`, and most stellarator
   campaign code into `tools/campaigns` or `tests/validation`.
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
- Remove cETG examples if cETG is retired.
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
8. Retire cETG or move it out of `main`.

Exit gates:

- `find src/spectraxgk -name '*.py' | wc -l` is at most 100.
- Public CLI and documented Python examples still work.
- No unintentional import cycles or optional-dependency import failures.
- Ruff, mypy, fast tests, wide coverage, and docs build pass.

### Phase 5: Examples And Benchmarks Cleanup

Goal: examples are pedagogical and benchmarks are reproducible.

Steps:

1. Rewrite `examples/README.md` with a task-oriented map.
2. Remove or move legacy examples such as cETG if the feature is retired.
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
3. Remove stale references to deleted files, cETG if retired, old tool paths,
   and obsolete validation lanes.
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

1. Add inventory/topology tooling and manifests.
2. Reorganize and merge tests below 100 files.
3. Reorganize and merge tools below 100 files.
4. Move validation/campaign code out of `src`.
5. Remove cETG and other legacy/non-promoted paths from `main` or move them to
   an experiment PR/branch.
6. Consolidate source domains below 100 files.
7. Clean examples and benchmarks.
8. Run performance profiling and targeted hot-path refactors.
9. Update docs/readme/API references.
10. Run full local and CI release gates.
11. Bump, tag, release.

## Progress Log

- 2026-07-07: added topology counts to
  `tools/package_architecture_manifest.toml` and extended
  `tools/check_package_architecture_manifest.py` so source, test, tool, flat
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

## Immediate Next Steps

1. Build the inventory script and topology manifest so the reduction targets are
   enforced by tooling rather than memory.
2. Start with test consolidation because it reduces friction without changing
   solver behavior.
3. Collapse tool-script tests into parametrized families, then move actual tools
   into purpose-specific folders.
4. Remove cETG from examples/docs/tests/source unless a draft PR is created to
   preserve it outside `main`.
5. Move `src/spectraxgk/validation` campaign code into benchmarks/tools/tests
   and shrink the installable package.
6. Only then perform source package consolidation and performance hot-path work.

## Completion Definition

This plan is complete only when current repository evidence proves all of the
following:

- `src/spectraxgk` has at most 100 Python files.
- `tests` has fewer than 100 Python files.
- `tools` has fewer than 100 Python files.
- Validation/campaign code no longer lives in the installable package except for
  tiny reusable metric helpers.
- cETG and other obsolete/non-promoted paths are removed from `main` or moved to
  a non-release PR/branch.
- Examples have a clear README and only current runnable/promoted workflows.
- Benchmarks are root-level, small, documented, and free of raw outputs.
- README and docs match the new structure and claim scope.
- Package-wide coverage remains at least 95%.
- Fast local tests, release gates, docs build, package build, and CI pass.
- Runtime/memory and speedup claims are backed by fresh profiler artifacts and
  numerical identity or physics gates.
