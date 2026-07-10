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
- Collision operators use a common extension contract. The shipped conserving
  Lenard--Bernstein/Dougherty-like baseline is retained while species-coupled
  Dougherty and linearized gyrokinetic Sugama/Coulomb implementations progress
  through invariant and literature-benchmark gates.
- The repository remains light: no generated caches, raw long-run outputs, build
  directories, or large transient artifacts are tracked.

## Current State

Date: 2026-07-10.

| Area | Current state | Target | Status |
| --- | ---: | ---: | --- |
| Installable source Python files | 228 | reviewed domain ownership | active |
| Source modules above 1000 lines | 8 including a 12512-line facade | 0 unreviewed | active |
| Public/compatibility facade maximum | 12512 lines | <=500 lines | active |
| Tool Python files | 134 | grouped commands; no duplicate owners | active |
| Test Python files | 98 | domain-organized; no duplicate behavior | closed for count, active for structure |
| README lines | 261 | <=350 user-facing lines | closed |
| Tracked files above 2 MB | 0 | 0 | closed |
| Fast release-surface coverage | local pass | pass | closed for current tranche |
| Package-wide coverage | above 95% in CI gate | >=95% | release gate retained |
| Public API facade | compact lazy registry | compact registry | closed |
| Runtime/plot executable path | implemented and tested | stable | closed |
| Nonlinear A4000 warm step, 192x64x24 | 109 ms fixed / 160 ms adaptive | <=1.25x matched comparison baseline | active |
| Nonlinear two-GPU whole-state path | 0.211x and identity failure | no production claim; replace decomposition | blocked from claims |

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
| Capability/parity specification | 98% | Keep source fingerprints and the machine-readable matrix synchronized with promoted benchmark gates. |
| Tool consolidation | 60% | Fold artifact builders into grouped domain commands; delete stale comparison/probe scripts; update docs command lines. |
| Test consolidation | 100% | Collapse large `tests/tools` families into parametrized contracts with shared fixtures while preserving gate semantics. |
| Source consolidation | 66% | Keep named cases comparison-only, migrate maintained benchmark drivers to the unified runtime, then delete each duplicated case solver after its parity gate passes. |
| Differentiable API clarity | 76% | Define dynamic cache/geometry rebuild boundaries, then complete forward, reverse/checkpointed, and implicit differentiation policies. |
| Advanced collision operators | 15% | Route the operator through full integration, then add species-coupled Dougherty, Sugama, and linearized Coulomb models with invariant and literature gates. |
| Nonlinear GPU performance | 84% | Make geometry and parameter pytrees dynamic in the prepared runner; then profile long-window memory and diagnostic streaming. |
| Production parallelization | 42% | Retain the corrected identity-gated combined-ky path, then replace failed whole-state spatial sharding with species/Hermite decomposition and explicit collectives. |
| Performance/release claims | 87% | Add prepared CPU/GPU rows to the next matched runtime/memory panel while keeping cold executable and warm Python claims separate. |
| Docs/readme release pass | 94% | Keep README concise and complete the developer/API updates as source owners move. |
| CI/release hygiene | 89% | Maintain fast checks under 5 minutes locally; inspect CI only after failures complete. |

## Prioritized Implementation Steps

1. **Freeze the required-core comparison contract.** Record exact equations,
   normalization, geometry arrays, grid layout, initialization, timestepping,
   precision, and diagnostics for each promoted linear/nonlinear comparison.
2. **Correct nonlinear execution and profiling.** Add a prepared nonlinear
   simulation object whose dynamic state/cache/parameter pytrees enter stable
   JIT boundaries, while methods, layouts, and output schemas are explicit
   static policies. Require an identical repeated call to compile the scan once.
   Keep CFL and sampling device-resident, report only active-sharding speedups,
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
   Treat equilibrium ExB flow shear as the next complete physics extension:
   zero-shear recovery, analytic shearing-wave evolution, remap/phase identity,
   linear mode suppression, nonlinear transport, and matched comparison gates.
6. **Add collision-operator extensibility.** Land a protocol with a complete
   RHS contribution plus an optional mathematically valid split step; do not
   model field-particle terms as diagonal damping. Preserve the current
   conserving Lenard--Bernstein/Dougherty-like result, then add species-coupled
   Dougherty and linearized Sugama/Coulomb operators. Require Maxwellian
   null-space, particle/total-momentum/total-energy conservation, adjointness,
   entropy production, collisional ITG, zonal damping, conductivity, and
   velocity-resolution evidence.
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

## GX-Informed Gap Assessment

The source audit at revision `bc2fe552` is a design input, not a mandate to
copy every feature. GX prioritizes species decomposition, then Hermite
decomposition, exchanges one or two Hermite ghost layers, and reduces field
moments through MPI/NCCL collectives while keeping perpendicular FFTs local.
That topology is the reference design for the production parallel lane.

| Capability | SPECTRAX-GK assessment | Decision |
| --- | --- | --- |
| Standard electrostatic/electromagnetic full gyrokinetics | implemented with scoped linear/nonlinear parity gates | required core |
| Boltzmann and kinetic species, Miller/VMEC, linked/periodic boundaries | implemented with scoped validation | required core |
| Equilibrium ExB flow shear | missing | add as a fully gated research extension |
| Species/Hermite multi-device execution | kernels/plans exist; production routing absent | implement after prepared-runner stabilization |
| Linearized Landau/Sugama collisions | missing; current model is a limited conserving Dougherty-like operator | add through a collision protocol and literature gates |
| Long-wavelength reduced field solve and Beer/Smith closures | missing | optional, only with a scientific owner |
| KREHM, Vlasov--Poisson, collisional-ETG, forcing, Trinity coupling | not complete equations in SPECTRAX-GK | keep out of scope; remove orphan compatibility fragments |
| JAX autodiff, implicit gradients, UQ, in-memory VMEC/Boozer optimization | SPECTRAX-GK extensions | retain and strengthen conditioning/FD/performance gates |

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
  It was slower than serial (`0.211x`) and failed trajectory identity
  (`max_abs_state_error=20.0`), so the result is now a tracked fail-closed
  artifact and the old tiny-grid identity profile is explicitly smoke-only.
- 2026-07-09: Added `benchmarks/capability_matrix.toml` and a release contract
  that separates required core, differentiable extensions, optional research
  extensions, and unsupported features. It freezes the ten-field matched-run
  contract and records the comparison audit at GX revision `bc2fe552`.
- 2026-07-09: Removed generic diagnostic compatibility wrappers from
  `spectraxgk.benchmarks`; eigenfunction, validation-gate, observed-order,
  late-window, and zonal metrics now come from their diagnostic owners.
- 2026-07-09: Moved the complete secondary-instability seed/stage workflow into
  the existing `spectraxgk.workflows.nonlinear` owner. The benchmark facade fell
  from 13209 to 12854 lines without adding source files, while the destination
  remains within the 1000-line module budget and all secondary tests pass.
- 2026-07-09: Moved generic pointwise linear-scan and representative-mode
  extraction into `spectraxgk.workflows.linear`, removed the unused `scan_fn`
  argument, and retained an explicit empty-scan contract. The benchmark facade
  is now 12608 lines, the destination remains below its 1000-line budget, and
  focused benchmark, executable, and core-contract tests pass.
- 2026-07-09: Audited the collision implementation against GX source and the
  Hermite--Laguerre collision literature. The current operator already contains
  low-order momentum/temperature corrections, so it is now described as a
  conserving Lenard--Bernstein/Dougherty-like limited model. The next protocol
  must expose a complete RHS contribution and only optional valid split steps;
  species-coupled Dougherty and Sugama/Coulomb remain separately gated lanes.
- 2026-07-09: Re-audited GX at upstream revision `bc2fe552` and fingerprinted
  the clean source as `sha256:bfaaadfa...20b`; the office instrumented snapshot
  is separately fingerprinted as `sha256:436e403e...a004`. GX prioritizes species then Hermite decomposition,
  with explicit Hermite ghost exchange and MPI/NCCL field reductions. The
  capability matrix now distinguishes standard gyrokinetic parity from
  separately scoped boundary policies, time schemes, reduced equation sets,
  moment closures, and SPECTRAX-GK differentiable extensions. The stale office
  GX binary needs a clean rebuild before new timing comparisons.
- 2026-07-09: Removed repeated JAX compilation from serial and diagnostic
  sharded nonlinear integration. Cache and parameter pytrees are now dynamic,
  mathematical switches remain static, the small Hermitian projector is
  memoized by grid signature, and sharded compiled runners are reused. A local
  post-warmup smoke profile fell from about 0.41 seconds to about 1 millisecond
  per diagnostic sharded call with exact state identity; benchmark-grid GPU
  profiling remains required before changing performance claims.
- 2026-07-09: Refreshed the benchmark-grid office GPU profile from clean commit
  `91c0c2a7`. Compile-stable serial execution is `0.893 s` for 20 RK2 steps,
  down from `15.37 s`; diagnostic two-GPU `kx` sharding is `4.22 s` median,
  remains slower (`0.211x`), and still fails state identity. Adaptive CFL is
  already device-resident inside the JAX scan, so the next profiling target is
  sampled diagnostics and remaining synchronization/materialization overhead.
- 2026-07-09: Replaced fourteen identical benchmark request/context packers
  with one typed dataclass-field contract in `spectraxgk.benchmarking.shared`.
  The benchmark facade is now 12512 lines, with hook-bearing TEM policy kept
  explicit because it performs real assembly rather than mechanical packing.
- 2026-07-09: Corrected the end-to-end runtime profiler to use an existing
  nonlinear input and block every returned JAX leaf before stopping timers.
  Added bounded repeat reporting. The shipped `64x64x24` Cyclone input takes
  `6.05 s` warm for 20 CPU steps at diagnostic stride 10. The matched office
  A4000 run takes `9.78 s` with resolved spectra and `8.69 s` with compact
  scalar diagnostics; final-state-only integration takes `0.263 s`. This
  localizes the next optimization target to synchronized diagnostics and
  runtime materialization rather than the compiled fixed-step update.
- 2026-07-09: Exposed compact nonlinear artifact output through
  `[output] resolved_diagnostics = false`. Publication/restart artifacts keep
  the resolved default, while scalar-only production runs can opt out of
  expensive spectral histories without changing scalar transport channels.
- 2026-07-09: Compile logging on three identical compact Cyclone calls showed
  that the nonlinear diagnostic scan compiled three times; each scan compile
  took about one second and the two nominal warm calls still took about
  `2.28 s` for only two steps. The next performance/API milestone is therefore
  a prepared nonlinear simulation with a stable compiled diagnostic entry
  point. The same GX source audit promoted equilibrium ExB flow shear to a
  separately gated research extension and kept unrelated reduced systems out
  of scope.
- 2026-07-09: Added a first `PreparedExplicitNonlinearDiagnostics` contract.
  It keeps one stable compiled scan across repeated same-signature Python calls
  and accepts replacement initial states. A direct-versus-prepared trajectory
  and scalar-transport identity test protects the new path. Geometry and model
  parameters remain fixed in this tranche; making those pytrees dynamic is the
  next differentiable-optimization milestone.
- 2026-07-09: The benchmark-size `64x64x24` Cyclone compile-log gate records
  one `jit(run_raw)` compile across three prepared calls. The first two-step
  call takes `3.25 s`; repeats take `0.297 s` and `0.290 s`, versus
  `2.27-2.29 s` when the runtime scan closure is rebuilt. This closes the
  compile-stability defect for fixed-policy repeated Python calls.
- 2026-07-09: Added clean-revision CPU/A4000 prepared-runtime artifacts at
  `8d2baf4e`. Matched adaptive RK3 20-step compact runs take `4.078 s` on the
  local CPU and `0.4648 s` on one A4000, an `8.77x` GPU throughput advantage.
  The earlier `8.69 s` nominal warm GPU result rebuilt and recompiled its scan;
  it did not measure steady prepared execution.
- 2026-07-09: Removed two public KREHM-only field-energy helpers and their
  compatibility tests because SPECTRAX-GK does not ship the corresponding
  reduced equation set. Standard full-gyrokinetic field-energy diagnostics and
  their geometry-weighted/resolved-sum gates remain unchanged. This applies the
  rule that an orphan diagnostic is not a supported physics capability.
- 2026-07-10: Replaced the 1294-line README with a 261-line user-facing entry
  point containing installation, executable/Python quickstarts, six headline
  evidence panels, conservative claim scope, and links to detailed docs.
  Release, quasilinear, QA-optimization, relative-link, and public-import gates
  pass. The rewrite also corrected a stale `spectraxgk.grids` import to the
  actual `spectraxgk.core.grid` owner and added a regression test.
- 2026-07-10: Removed the orphan named-case executable stack after confirming
  that every tracked TOML input uses the unified runtime schema. The no-input
  demo now exercises the same runtime configuration and solver path as ordinary
  Python and executable runs. This tranche removed one installed module and
  1,219 net lines across source/tests, retained local output and progress/ETA,
  and completed the real CPU demo in 10.1 s with 100 saved samples and a finite
  fitted mode (gamma=0.085141, omega=0.291379). Focused runtime tests, docs,
  package build, architecture, coverage-manifest, and release-readiness gates
  pass.
- 2026-07-10: Removed the remaining named-case TOML parser, private solver
  section loaders, and three public compatibility exports. All maintained
  inputs now enter through `load_runtime_from_toml`; focused core, runtime
  configuration, executable, and public-API tests pass. This removes another
  118 net lines and leaves one documented input schema.
- 2026-07-10: Added `PreparedExplicitNonlinearDiagnostics.run_arrays()` as the
  host-conversion-free differentiable boundary for repeated nonlinear scans.
  A physical tiny-grid reverse-mode test differentiates through the explicit
  time loop and agrees with centered finite differences to 2% while the docs
  state exactly which cache-dependent geometry and velocity parameters remain
  fixed.
- 2026-07-10: Wired the structural `CollisionOperator` into the cached
  nonlinear RHS. A supplied JAX callback replaces only the built-in collision
  contribution, preserves independent hypercollisions, is multiplied by the
  configured collision weight, and must preserve the state shape. Unit tests
  cover replacement/no-double-counting and invalid shapes; docs keep Sugama and
  full linearized Coulomb models explicitly unpromoted pending invariant gates.
- 2026-07-10: Restored the office comparison-code runtime environment using the
  locally built parallel NetCDF/HDF5, OpenMPI, CUDA tensor, NCCL, and GSL
  libraries, then completed a fresh bounded Cyclone linear comparison on one
  A4000. The unified runtime initially exposed a diagnostic mismatch: projected
  fitting understated growth by 23%. Using the benchmark-aligned midplane
  observable gives gamma=0.09076 and omega=0.27828 versus late-window reference
  means 0.09582 and 0.28106 (5.3% and 1.0% relative errors). The canonical
  Cyclone TOML now records that observable and `kz`-proportional dissipation;
  the duplicate `runtime_cyclone.toml` was deleted. Explicit-time executable
  runs now reject non-divisible `steps/sample_stride` values before geometry
  and cache setup instead of failing after compilation work begins.
- 2026-07-10: Corrected combined linear ``ky`` scans to slice a compact
  linear-only grid instead of inheriting the nonlinear two-thirds dealias
  mask. A physical regression now includes a high mode outside that mask and
  matches independent serial integration; the fresh 11-mode A4000 run retains
  all requested modes. Cold-process timing is recorded as diagnostic evidence,
  not a speedup claim. Named benchmark solvers and reference tables were also
  removed from the general top-level API and remain explicitly available from
  ``spectraxgk.benchmarks``; the quickstart now teaches the unified runtime API.

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
