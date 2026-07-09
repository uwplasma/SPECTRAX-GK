# SPECTRAX-GK Consolidation And Release Plan

This is the single active plan for the current refactor/release tranche. It
replaces older campaign-log plans and keeps only current goals, current state,
open lanes, and the recent implementation log.

## One-Sentence Plan

Make SPECTRAX-GK a compact, JAX-native gyrokinetic package with a small and
navigable source tree, a smaller set of grouped maintainer tools, fewer and more
physics-driven tests, documented benchmark/comparison workflows, preserved
validated solver behavior, and profiler-backed CPU/GPU performance claims.

## Definition Of Done

- `src/spectraxgk` has fewer than 100 Python files or every remaining excess file
  is justified by a domain boundary and tracked in the architecture manifest.
- `tools` has fewer than 99 Python files, with one grouped command per artifact,
  campaign, release-gate, or profiling family.
- `tests` has fewer than 99 Python files while preserving the fast release gates,
  physics validation gates, numerics gates, and package-wide coverage above 95%.
- The executable user path remains simple: `spectraxgk`, `spectrax-gk`, and
  `spectraxgk --plot` keep working with documented examples.
- Python workflows remain differentiable where advertised; executable workflows
  may use faster non-differentiable code paths when explicitly documented.
- Benchmark/comparison references to external gyrokinetic codes appear only in
  benchmark/comparison contexts, docs explaining validation, and explicitly
  labeled comparison figures/tables.
- Runtime and memory claims in README/docs are backed by current artifacts and
  profiler or benchmark records.
- The repository remains light: no generated caches, raw long-run outputs, build
  directories, or large transient artifacts are tracked.

## Current State

Date: 2026-07-09.

| Area | Current state | Target | Status |
| --- | ---: | ---: | --- |
| Installable source Python files | 243 | 100 | active |
| Tool Python files | 139 | 99 | active |
| Test Python files | 98 | 98 | closed |
| Tracked files above 2 MB | 0 | 0 | closed |
| Fast release-surface coverage | local pass | pass | closed for current tranche |
| Package-wide coverage | above 95% in CI gate | >=95% | release gate retained |
| Public API facade | compact lazy registry | compact registry | closed |
| Runtime/plot executable path | implemented and tested | stable | closed |

## File Retention Rule

Every retained file must have all four fields below. Files that fail the rule are
merged into a family owner, deleted from `main`, or moved to an external draft
branch/PR.

| Field | Allowed answer | Not allowed |
| --- | --- | --- |
| Owner | runtime package, example, root benchmark, maintainer tool, docs artifact, test, active campaign | historical branch, local probe, unused manuscript scratch |
| Reason | promoted workflow, reusable physics/numerics API, reproducible benchmark, CI/release gate, documented figure/table, active long-run campaign | convenience wrapper, old branch behavior, unreferenced output builder |
| Test owner | unit/integration/validation/tool/release test family or explicit manual office benchmark | no test and no documented manual gate |
| Destination | `src`, `examples`, `benchmarks`, `tools`, `tests`, `docs/_static`, external draft PR | installable validation package, raw output directory, duplicate tool script |

## Repository Role Model

| Location | Role | Remove or move |
| --- | --- | --- |
| `src/spectraxgk` | Installed user/developer API: kernels, solvers, geometry, diagnostics, objectives, workflows | manuscript campaigns, raw comparison scripts, one-off artifact builders |
| `examples` | Small runnable tutorials and optimization examples | long campaigns, raw sweeps, unreleased probes |
| `benchmarks` | Researcher-facing benchmark drivers and compact inputs | generated outputs, hidden policy code imported by package |
| `tools` | Maintainer commands for artifacts, campaigns, profiling, release checks, comparisons | one-file-per-panel scripts, stale fallback launchers, local probes |
| `tests` | Automated correctness, physics, numerics, and release gates | one-file-per-helper mirrors and redundant monkeypatch branch tests |
| `docs/_static` | Curated evidence referenced by README/docs/release manifests | stale pilot traces and unreferenced historical panels |

## Open Lanes And Progress

| Lane | Completion | Next concrete action |
| --- | ---: | --- |
| Tool consolidation | 50% | Fold artifact builders into grouped domain commands; delete stale comparison/probe scripts; update docs command lines. |
| Test consolidation | 100% | Collapse large `tests/tools` families into parametrized contracts with shared fixtures while preserving gate semantics. |
| Source consolidation | 38% | Shrink `spectraxgk.benchmarks`, merge `geometry_backends` into `geometry`, and resolve `terms`/`operators` ownership. |
| Differentiable API clarity | 72% | Keep compact API registry; document differentiable versus executable-fast paths; finish objective-family cleanup. |
| Performance/release claims | 78% | Keep only profiler-backed speed claims; refresh runtime/memory panel after topology cleanup. |
| Docs/readme release pass | 74% | Update code-structure, benchmark, performance, and optimization docs after each grouped consolidation. |
| CI/release hygiene | 88% | Maintain fast checks under 5 minutes locally; inspect CI only after failures complete. |

## Prioritized Implementation Steps

1. **Tool pruning and grouping.** Delete unreferenced tools, replace stale command
   references, then consolidate artifact builders by domain: nonlinear transport,
   quasilinear model development, stellarator/VMEC-Boozer, W7-X/zonal, and
   performance panels.
2. **Test family collapse.** Merge one-file-per-tool tests into table-driven
   families with shared fake artifacts and shared command-load fixtures.
3. **Benchmark facade shrink.** Keep stable benchmark result contracts in
   `spectraxgk.benchmarks`; move case-policy and manuscript-like benchmark
   drivers to root `benchmarks` or maintainer tools.
4. **Source ownership cleanup.** Merge `geometry_backends` into `geometry`, choose
   a single public mathematical-kernel namespace for `terms`/`operators`, and
   consolidate objective helper shards into fewer family modules.
5. **Performance pass.** Profile quickstart, linear RHS/cache, nonlinear RHS and
   bracket, diagnostics IO, and VMEC/Boozer transforms. Add speed claims only when
   before/after artifacts and numerical identity gates exist.
6. **Docs and release pass.** Regenerate referenced figures/tables, run fast
   release tests, package build, docs build, package-wide coverage gate, then bump
   version and tag only when CI is green.

## Recent Implementation Log

- 2026-07-09: Consolidated runtime startup and linear-cache profiling into
  `tools/profiling/profile_startup_and_cache.py`.
- 2026-07-09: Consolidated VMEC boundary campaign writers into
  `tools/campaigns/write_vmec_boundary_campaigns.py`.
- 2026-07-09: Consolidated nonlinear transport release gates into
  `tools/release/check_nonlinear_transport_gates.py`.
- 2026-07-09: Consolidated nonlinear optimization release gates into
  `tools/release/check_nonlinear_optimization_gates.py`.
- 2026-07-09: Replaced ten thin `spectraxgk.api.*` re-export modules with one
  compact lazy registry in `spectraxgk.api`; fixed the fast coverage plotting
  gate and pushed commit `880ea3ed`.
- 2026-07-09: Audited comparison tools and kept `compare_gx_nonlinear.py`
  because the comparison contract tests cover it; tool count remains 139.
- 2026-07-09: Collapsed eight small release test files into
  `tests/release/test_release_gates.py`, reducing the test file count from 136
  to 129 while preserving all release, manifest, coverage, and hygiene gates.
- 2026-07-09: Collapsed ten comparison-tool test files into
  `tests/tools/comparison/test_reference_comparison_tools.py`, reducing the
  test file count from 129 to 120 while keeping the comparison contracts.

- 2026-07-09: Collapsed eleven artifact-tool test files into four domain
  suites: general, transport, quasilinear, and stellarator/status artifacts;
  total test file count dropped from 120 to 113 while preserving all artifact
  contracts.
- 2026-07-09: Consolidated objectives and parallel unit tests into seven
  domain suites, reducing total test file count from 113 to 98 and closing the
  active test-file-count target without changing advertised solver behavior.

## Validation Commands For This Tranche

```bash
ruff check src tests tools
python tools/release/check_package_architecture_manifest.py
python tools/release/check_validation_coverage_manifest.py
python tools/release/check_differentiable_refactor_manifest.py
python tools/release/check_performance_optimization_manifest.py
python tools/release/check_repository_size_manifest.py
pytest -q -o addopts= tests/tools tests/release --maxfail=1
```

Use narrower focused subsets during editing, and keep each local test invocation
under 5 minutes.
