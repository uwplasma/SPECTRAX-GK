# SPECTRAX-GK Consolidation And Release Plan

This is the single active plan for the current refactor/release tranche. It
replaces older campaign-log plans and keeps only current goals, current state,
open lanes, and the recent implementation log.

## One-Sentence Plan

Make SPECTRAX-GK a compact, JAX-native gyrokinetic package with a navigable
physics-first architecture, documented comparison contracts, research-grade
validation, efficient CPU/GPU execution, production-safe parallelization, and
explicitly verified differentiable workflows for analysis and optimization.

## Definition Of Done

- The source tree follows explicit domain ownership: model/state, geometry,
  physical operators, solvers/parallelism, diagnostics, design/optimization,
  and I/O/workflows. File count is a review signal, not an optimization target.
- Hand-written production modules stay below 1000 lines and public facades below
  500 lines unless an architecture manifest records a reviewed exception. No
  compatibility facade may own physical or numerical implementation policy.
- Maintainer tools use one grouped command per artifact, campaign, release-gate,
  comparison, or profiling family; generated outputs and campaign policy do not
  live in the installed package.
- Tests mirror scientific domains and preserve fast release gates, conservation
  and observed-order checks, literature/reference comparisons, and package-wide
  coverage above 95%. Coverage is necessary but does not replace physics gates.
- The executable user path remains simple: `spectraxgk`, `spectrax-gk`, and
  `spectraxgk --plot` keep working with documented examples.
- Python workflows remain differentiable where advertised; executable workflows
  may use faster non-differentiable code paths when explicitly documented.
- Benchmark/comparison references to external gyrokinetic codes appear only in
  benchmark/comparison contexts, docs explaining validation, and explicitly
  labeled comparison figures/tables.
- Runtime and memory claims in README/docs are backed by current artifacts and
  profiler or benchmark records.
- Core electrostatic/electromagnetic equations, supported geometries and
  boundaries, species response, restarts, and resolved diagnostics have an
  explicit capability/parity matrix. Specialized reduced models are either
  validated features or clearly out of scope, never implied by broad wording.
- GPU profiles separate cold compilation, fixed per-run overhead, warm per-step
  throughput, memory, and device utilization. Multi-device speedups require
  active sharding plus state, invariant, diagnostic, and transport-window identity.
- Collision operators use a common extension contract. The shipped baseline is
  retained while conserving Dougherty and linearized gyrokinetic Sugama/Coulomb
  implementations progress through invariant and literature-benchmark gates.
- The repository remains light: no generated caches, raw long-run outputs, build
  directories, or large transient artifacts are tracked.

## Current State

Date: 2026-07-09.

| Area | Current state | Target | Status |
| --- | ---: | ---: | --- |
| Installable source Python files | 228 | reviewed domain ownership | active |
| Source modules above 1000 lines | 8 including a 13209-line facade | 0 unreviewed | active |
| Public/compatibility facade maximum | 13209 lines | <=500 lines | active |
| Tool Python files | 134 | grouped commands; no duplicate owners | active |
| Test Python files | 98 | domain-organized; no duplicate behavior | closed for count, active for structure |
| Tracked files above 2 MB | 0 | 0 | closed |
| Fast release-surface coverage | local pass | pass | closed for current tranche |
| Package-wide coverage | above 95% in CI gate | >=95% | release gate retained |
| Public API facade | compact lazy registry | compact registry | closed |
| Runtime/plot executable path | implemented and tested | stable | closed |
| Nonlinear A4000 warm step, 192x64x24 | 109 ms fixed / 160 ms adaptive | <=1.25x matched comparison baseline | active |
| Nonlinear two-GPU whole-state path | 0.706x and identity failure | no production claim; replace decomposition | blocked from claims |

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

## Capability And Comparison Policy

The implementation does not need every specialized feature in another
gyrokinetic code. Each capability belongs to one tier:

| Tier | Meaning | Current examples |
| --- | --- | --- |
| Required core | Must have equation, normalization, state-level, and observable gates | electrostatic ITG, electromagnetic fields, kinetic/Boltzmann species, linked/periodic flux tubes, Miller/VMEC geometry, restart and resolved diagnostics |
| Differentiable extension | SPECTRAX-GK-specific research capability with AD/FD and conditioning gates | implicit eigenvalue gradients, UQ, quasilinear objectives, vmec_jax/booz_xform_jax geometry and optimization |
| Optional extension | Implement only with a scientific owner and validation campaign | advanced closures, broad forcing, transport-framework coupling |
| Explicitly out of scope | No release implication; may be reconsidered later | unrelated reduced plasma models and retired collisional-ETG compatibility paths |

External-code names remain confined to comparison inputs, benchmark tooling,
validation documentation, and figures/tables. Source-level physics and numerics
use mathematical names independent of comparison provenance.

## Open Lanes And Progress

| Lane | Completion | Next concrete action |
| --- | ---: | --- |
| Capability/parity specification | 75% | Freeze the required-core matrix and exact matched geometry/grid/diagnostic contracts. |
| Tool consolidation | 60% | Fold artifact builders into grouped domain commands; delete stale comparison/probe scripts; update docs command lines. |
| Test consolidation | 100% | Collapse large `tests/tools` families into parametrized contracts with shared fixtures while preserving gate semantics. |
| Source consolidation | 51% | Shrink `spectraxgk.benchmarks` below facade budget, resolve `terms`/`operators` ownership, and reduce oversized domain modules without creating tiny shards. |
| Differentiable API clarity | 72% | Define forward, reverse/checkpointed, and implicit differentiation policies; document differentiable versus executable-fast paths. |
| Advanced collision operators | 10% | Introduce operator protocol, conserving baseline, then Sugama and linearized Coulomb with invariant and literature gates. |
| Nonlinear GPU performance | 60% | Move CFL/sampling device-resident, then match fixed-step workloads before optimizing kernels. |
| Production parallelization | 35% | Replace failed whole-state spatial sharding with species/Hermite decomposition and explicit collectives. |
| Performance/release claims | 82% | Keep only profiler-backed claims; refresh matched runtime/memory panel after integrator and topology corrections. |
| Docs/readme release pass | 80% | Update code-structure, benchmark, performance, and optimization docs after each grouped consolidation. |
| CI/release hygiene | 89% | Maintain fast checks under 5 minutes locally; inspect CI only after failures complete. |

## Prioritized Implementation Steps

1. **Freeze the required-core comparison contract.** Record exact equations,
   normalization, geometry arrays, grid layout, initialization, timestepping,
   precision, and diagnostics for each promoted linear/nonlinear comparison.
2. **Correct nonlinear execution and profiling.** Remove unused RHS evaluations,
   keep CFL and sampling device-resident, report only active-sharding speedups,
   and separate cold, fixed-overhead, warm-throughput, utilization, and memory.
3. **Benchmark facade shrink.** Keep stable benchmark result contracts in
   `spectraxgk.benchmarks`; move case-policy and manuscript-like benchmark
   drivers to root `benchmarks` or maintainer tools.
4. **Source ownership cleanup.** Keep imported Miller/VMEC geometry in `geometry`, choose
   a single public mathematical-kernel namespace for `terms`/`operators`, and
   consolidate objective helper shards into fewer family modules.
5. **Close required-core physics gates.** Maintain state-level short gates and
   converged long-window gates for axisymmetric/stellarator, electrostatic/
   electromagnetic, adiabatic/kinetic-electron, and restart/spectral diagnostics.
6. **Add collision-operator extensibility.** Land a tested operator protocol,
   conserving Dougherty parity model, and linearized Sugama/Coulomb operators.
   Require null-space, particle/momentum/energy, adjointness, entropy-production,
   collisional ITG, zonal-flow damping, conductivity, and convergence evidence.
7. **Formalize differentiation.** Use forward JVPs for few design parameters,
   reverse checkpointing for many-parameter scalar objectives, and implicit JVP/
   VJP rules for converged eigen/root solves. Adaptive and turbulent objectives
   require tolerance/window/seed refinement plus AD/finite-difference checks.
8. **Implement production parallelism.** Decompose species first and Hermite
   moments second, exchange Hermite halos explicitly, reduce field moments with
   collectives, and keep perpendicular FFTs local until memory requires pencils.
9. **Tool pruning and test normalization.** Delete unreferenced tools, group
   artifact commands, and use table-driven domain tests with shared fixtures.
10. **Docs and release pass.** Regenerate referenced figures/tables, run fast
   release tests, package build, docs build, package-wide coverage gate, then bump
   version and tag only when CI is green.

## Recent Implementation Log

- 2026-07-09: Audited a clean comparison-code build and SPECTRAX-GK on office
  RTX A4000 GPUs. The isolated SPECTRAX nonlinear RHS is GPU-efficient, but the
  full integrator is dominated by fixed overhead and low device occupancy; the
  two-GPU whole-state `kx` route is slower and fails trajectory identity.
- 2026-07-09: Replaced raw source-file-count completion pressure with domain
  ownership and complexity budgets, and added explicit core-parity, collision,
  differentiation, and production-parallelization exit gates.
- 2026-07-09: Added a final-state-only nonlinear integration mode that skips
  the post-step RHS used solely for field-history recovery. A controlled office
  A4000 A/B at `(4,8,64,192,24)` reduced median warm time by about 5--6% while
  preserving the existing field-returning default and its numerical tests.
- 2026-07-09: Re-ran two-GPU whole-state `kx` sharding on that benchmark grid.
  It was slower than serial (`0.706x`) and failed trajectory identity
  (`max_abs_state_error=33.32`), so the result is now a tracked fail-closed
  artifact and the old tiny-grid identity profile is explicitly smoke-only.

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
- 2026-07-09: Replaced the separate fast-test and wide-coverage release runners
  with `tools/release/run_test_gates.py` subcommands, reducing tool count from
  139 to 138 while keeping bounded local and CI coverage semantics.
- 2026-07-09: Grouped reference-validation, benchmark-refresh, and runtime-gate
  campaign commands into `tools/campaigns/run_validation_campaigns.py`, reducing
  tool count from 138 to 137 and updating docs/manifests/tests to the smaller
  command surface.
- 2026-07-09: Folded the Miller imported-geometry backend into
  `spectraxgk.geometry.imported_miller` and moved shared JAX geometry kernels
  into `spectraxgk.geometry.kernels`, reducing source count from 243 to 239
  while preserving Miller/VMEC geometry tests.
- 2026-07-09: Folded the VMEC imported-geometry backend into
  `spectraxgk.geometry.imported_vmec`, removing the deleted legacy geometry
  namespace and reducing source count from 239 to 231 while preserving VMEC/EIK
  geometry tests.
- 2026-07-09: Folded cached RHS assembly helpers into
  `spectraxgk.terms.assembly`, reducing source count from 231 to 229 while
  preserving term-assembly and linear helper tests.
- 2026-07-09: Moved the fixed-step nonlinear scan policy from
  `spectraxgk.terms` into `spectraxgk.solvers.nonlinear.explicit`, reducing
  source count from 229 to 228 while preserving explicit-scan tests.
- 2026-07-09: Grouped three VMEC/Boozer release-gate scripts into
  `tools/release/check_vmec_boozer_gates.py` subcommands
  (`differentiability-claim`, `aggregate-holdout`, and `reduced-portfolio`),
  reducing tool count from 137 to 135 while preserving release and stellarator
  validation tests.
- 2026-07-09: Folded the reduced nonlinear-audit prelaunch artifact builder
  into `tools/artifacts/build_nonlinear_transport_admission.py prelaunch`,
  reducing tool count from 135 to 134 and keeping landscape, prelaunch,
  campaign, and redesign policy in one nonlinear transport owner.

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
