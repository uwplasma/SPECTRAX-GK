# GKX 2.0 Simplification, Optimization, Documentation, and Rename Plan

Date: 2026-07-17
Status: implementation active on `refactor/gkx-2.0`
Baseline: SPECTRAX-GK `v1.7.0` (`644bad30`)

The complete pre-2.0 engineering log is preserved permanently in the `v1.7.0`
tag. This file is the single execution authority for the next release. Older
refactor-plan pages are historical input and will be removed once their still
valid technical requirements are represented here.

## One-Sentence Plan

Delete unpromoted and duplicated machinery first, consolidate the retained
solver into a shallow physics-first JAX package with substantially fewer files
and lines, reproduce and then extend the latest VMEC-JAX precise-QA optimization
with validated linear, quasilinear, and true nonlinear ITG objectives, rebuild
the README around three consistent evidence-backed figures and a sourced
capability table, and only then perform the breaking package/repository rename to
GKX and release version 2.0.

## Non-Negotiable Outcomes

1. Preserve every currently promoted physics, parity, differentiability,
   restart, plotting, and scoped performance gate from `v1.7.0`.
2. Reduce code by deletion and unified abstractions, not by moving complexity
   into giant files, generated code, compatibility layers, or hidden scripts.
3. Keep the differentiable Python path pure JAX. The executable may use faster
   non-differentiable orchestration, host I/O, progress reporting, and compiled
   caches.
4. Keep comparison-code names out of solver source and user APIs. They may occur
   only in benchmark provenance, comparison scripts, validation documentation,
   and citations.
5. Do not preserve obsolete private imports or legacy output formats. This is a
   deliberate 2.0 API break; examples and documentation migrate with the code.
6. Do not publish a transport-optimization result unless the equilibrium,
   objective derivative/statistics, and independent nonlinear audit all pass.
7. Do not make runtime or scaling claims without a fresh matched CPU/GPU
   artifact and numerical-identity gate.

## Measured Baseline and Targets

The previous refactor reduced source modules from 357 at `v1.6.10` to 220 at
`v1.7.0`, but 200 commits changed about 335,000 lines for only a 5,550-line net
repository reduction. The next tranche therefore measures deletion and
navigation cost directly.

| Surface | `v1.7.0` baseline | 2.0 target | Acceptance rule |
| --- | ---: | ---: | --- |
| Installable Python source | 220 files / 87,847 lines | <=70 files / <=50,000 lines | Both limits pass; no denominator exclusions |
| Package nesting | up to three domain levels | one domain level below `gkx` | `gkx/domain/module.py`; no `operators/linear/...` trees |
| Public root API | large lazy registry plus many facades | <=30 documented names | Every name appears in API docs and one example |
| Source-file size | many files near 900-1,000 lines | median <=650; hard max 1,200 | Exceptions require equation-level cohesion and review |
| Tests | 95 files / 96,202 lines | <=45 files / <=60,000 lines | >=95% package coverage and all physics gates retained |
| Developer tools | 95 files / 97,346 lines | <=15 scripts / <=25,000 lines | One command owner per release, figure, campaign, and profile family |
| README | about 295 lines / eight unrelated figures | <=180 lines / <=3 figures | Installation and first run visible before technical detail |
| Documentation pages | 29 top-level RST pages | <=16 user/research pages | No duplicate plan, status, or claim prose |
| Tracked documentation assets | about 39 MB | <=15 MB | Compact source data + compressed publication previews |
| Whole tracked repository | 49.2 MB | <=25 MB | Build caches and raw campaigns remain untracked |

The targets are budgets, not incentives to create monoliths. A retained module
must own one coherent mathematical, physical, numerical, I/O, or user-workflow
responsibility and have independent consumers or tests.

## Target Repository

```text
GKX/
├── src/gkx/
│   ├── __init__.py          # <=30 stable public names
│   ├── cli.py               # executable parsing and progress only
│   ├── config.py            # typed run configuration
│   ├── model/               # gyrokinetic equations and physical closures
│   ├── geometry/            # analytic, Miller, VMEC, and flux-tube contracts
│   ├── numerics/            # grids, bases, fields, time/eigen solves, parallel maps
│   ├── solve/               # linear, nonlinear, scan, and prepared-run APIs
│   ├── diagnostics/         # growth, transport, spectra, and convergence statistics
│   ├── optimize/            # differentiable observables and stellarator objectives
│   └── io/                  # NetCDF, restart, and plotting
├── examples/                # short user-owned scripts and TOML inputs
├── benchmarks/              # scientific cases, frozen references, comparison/plots
├── tests/                   # unit, integration, physics, and release suites
├── scripts/                 # <=15 developer/release/profile commands
├── docs/                    # user guide, equations, validation, and API
├── README.md
└── pyproject.toml
```

No domain contains another domain package. Modules use concrete names such as
`model/collisions.py`, `numerics/time.py`, `solve/nonlinear.py`, and
`diagnostics/transport.py`, not historical names such as `runtime_*`,
`artifacts_*`, `*_reports`, `*_contracts_strategy`, or comparison-code labels.

## Retention and Deletion Test

Every file and public function must answer all four questions:

1. What current user, equation, numerical method, or release gate owns it?
2. Who imports it outside its own test?
3. What would fail scientifically or operationally if it were removed?
4. Why can it not live in the nearest coherent domain owner?

If any answer is missing, delete it. One-off campaign logic is not package code.
Small frozen reference data belongs in `benchmarks`; raw runs belong in release
assets or external storage. Generated status dashboards and historical planning
artifacts are not retained merely because they existed in 1.x.

## Phase 0: Freeze the 1.7 Reference Contract

Deliverables:

- Record hashes and compact outputs for the promoted Cyclone, Cyclone Miller,
  KBM, W7-X, HSX, collision, restart, geometry, autodiff, and optimization gates.
- Record cold/warm CPU and office-GPU runtime plus peak-memory baselines for the
  representative linear and nonlinear cases.
- Generate one machine-readable inventory mapping each public API name, source
  owner, test owner, documentation page, benchmark, and claim level.
- Classify every source/tool/test file as `retain`, `merge`, `move`, or `delete`.
- Create one implementation branch, `refactor/gkx-2.0`, and one draft PR only.

Exit gate: the frozen 1.7 wheel and the development branch produce equivalent
promoted observables before any deletion tranche begins.

## Phase 1: Delete Before Refactoring

Delete or move first:

- reduced/synthetic QA models and panels that are not solved VMEC equilibria;
- superseded campaign admission, status-dashboard, and one-off artifact builders;
- dormant nonlinear sharding experiments that failed identity or speed gates;
- duplicate plotting, NetCDF, metadata, policy, and report wrappers;
- retired compatibility facades and import aliases;
- generated PDFs, redundant CSV/JSON encodings, and old plan/status images;
- stale build, egg-info, output, and cache directories from local workflows.

Consolidation priorities from the measured tree:

| Current family | Current files | Target owner count | Decision |
| --- | ---: | ---: | --- |
| `objectives` | 34 | 5-7 | Keep physical objectives, eigen derivatives, optimization, and gates; delete campaign-policy plumbing |
| `operators` | 33 | 8-10 | Organize by equations: fields, streaming/drifts, collisions, nonlinear bracket |
| `solvers` | 29 | 7-9 | One linear, nonlinear, time, eigen/Krylov, and prepared-run owner each |
| `geometry` | 25 | 6-8 | Analytic, Miller, VMEC, flux-tube, Boozer, and derivative owners |
| `diagnostics` | 24 | 5-7 | Growth, transport, spectra, conservation, and validation statistics |
| `workflows` | 21 | 3-4 | Run, scan, examples, executable orchestration |
| `artifacts` | 16 | 3-4 in `io` | NetCDF/restart, plotting, benchmark payloads |
| `parallel` plus parallel solver/operator files | >20 | 3-5 | Independent maps, decomposition, identity/performance gates |

Each commit must reduce both file count and net lines. Temporary duplicate old
and new implementations may exist only within a single commit.

Exit gate: source <=120 files and <=68,000 lines, tools <=45 files, all frozen
contracts and >=95% coverage pass.

## Phase 2: Consolidate the Physics and Numerical Core

### Model ownership

- Put the normalized gyrokinetic equation, species coupling, field equations,
  drive/drift terms, nonlinear bracket, dissipation, and collisions under
  `gkx.model`.
- Keep term switches as data, not branches duplicated across linear, nonlinear,
  diagnostic, and parallel implementations.
- Make collision operators implement one documented protocol. Retain the
  validated diagonal finite-wavelength Coulomb/original/improved-Sugama research
  path without inflating the default executable model claim.

### Numerical ownership

- Keep one spectral-grid type, one Hermite-Laguerre basis owner, one field solve,
  one linear operator assembly, and one nonlinear bracket implementation.
- Keep SOLVAX responsible for generic structured linear algebra; GKX owns plasma
  equations, branch selection, physical preconditioners, and acceptance gates.
- Express fixed-step differentiable integration with `jax.lax.scan`; keep
  adaptive Diffrax and IMEX paths only where distinct validated behavior exists.
- Replace parallel reference/report/strategy modules with one decomposition data
  model plus pure local kernels and explicit collectives.
- Profile every consolidation on CPU and GPU; reject abstractions that add
  dispatches, transposes, global reconstruction, or compilation memory.

### API ownership

The intended public workflow is:

```python
import gkx

case = gkx.load("cyclone.toml")
result = gkx.solve(case)
gkx.plot(result)
```

Advanced imports come from one domain, for example `gkx.geometry.vmec` and
`gkx.optimize`. Internal helpers remain private and are not re-exported through
registries.

Exit gate: source <=70 files and <=50,000 lines; public API <=30 names; all
promoted values, derivatives, restart files, and warm-runtime budgets pass.

## Phase 3: Simplify Tests, Benchmarks, and Developer Commands

### Tests

Use four suites with one file per coherent domain:

- `tests/unit`: equations, grids/bases, geometry, solvers, diagnostics, I/O;
- `tests/integration`: executable, restart, prepared runs, parallel identity;
- `tests/physics`: literature limits, convergence, conservation, and benchmark
  observables;
- `tests/release`: package, docs, repository, API, and claim gates.

Parametrize cases instead of creating files per case. Shared physical fixtures
must be immutable and explicit. Coverage remains >=95%, but the acceptance
criterion is detection power: manufactured solutions, observed order,
conservation/free-energy checks, branch continuity, grid/time convergence,
AD-vs-FD, and independent comparison results.

### Benchmarks

`benchmarks/` owns all scientific comparisons and publication regeneration:

- compact TOML/JSON case definitions;
- small reference arrays with revision/source hashes;
- one linear, one nonlinear, one collision, one geometry, and one performance
  runner;
- one figure command that applies a shared publication style.

Comparison-code names are allowed here because provenance is the purpose.

### Scripts

Replace `tools/` with at most 15 conventional developer commands under
`scripts/`: `ci.py`, `release.py`, `repo_audit.py`, `figures.py`,
`campaign.py`, and focused profiling commands. Reusable science belongs in
`gkx` or `benchmarks`; shell orchestration does not become importable package
code.

Exit gate: <=45 test files/60,000 lines, <=15 scripts/25,000 lines, bounded
local shards finish within five minutes each, and CI reproduces the 95% gate.

## Phase 4: Rebuild the QA ITG Optimization Evidence

### 4.1 Exact QA baseline

Use the latest `vmec_jax` main branch and reproduce
`examples/optimization/QA_optimization_ess.py` without modification:

- `input.minimal_seed_nfp2` and its `RBC/ZBS(1,1)` perturbation;
- `nfp=2`, `max_mode=5`, one least-squares call;
- Exponential Spectral Scaling with `ess_alpha=0.7`;
- quasisymmetry on ten surfaces, aspect target 6.0, mean-iota target 0.42;
- implicit equilibrium adjoint and the upstream solve tolerances/budget.

Freeze the resulting input, WOUT hash, QS residual, aspect, iota profile,
boundary coefficients, convergence history, and VMEC-JAX revision. This is the
only baseline. No reduced surface or synthetic proxy may replace it.

### 4.2 Shared transport portfolio

All three optimizations use the same normalized training portfolio and separate
holdouts. Initial training points are three radii (`s=0.25, 0.50, 0.75`), two
field lines per radius, and a resolved positive `ky` spectrum. Resolution and
spectral ranges are promoted only after `ky`, parallel-grid, Hermite, Laguerre,
time-window, and timestep convergence. Holdouts use unseen radii, alpha values,
and gradient values.

The equilibrium terms retain the exact baseline targets. Transport weights are
selected by a Pareto scan, not manually chosen after seeing the final result.
Accepted geometries must satisfy:

- aspect within 1% of 6.0;
- mean iota within 0.01 of 0.42 and no evaluated profile point below 0.39;
- QS residual no more than 2x the precise-QA baseline and below a fixed absolute
  publication threshold;
- solved equilibrium, finite gradients, and unchanged field-line conventions.

### 4.3 Linear-growth objective

Define a differentiable portfolio residual from the positive part of the
isolated ITG growth rate over the converged `ky` spectrum. Use a smooth maximum
or weighted RMS so one branch cannot hide behind stable modes. Differentiate
with the implicit left/right eigenpair rule and the VMEC fixed-point adjoint.
Require spectral-gap/branch continuity plus directional JVP/VJP versus centered
finite differences before optimization.

Run one `max_mode=5`, ESS-scaled VMEC-JAX least-squares solve with the added
linear residual tuple. Promotion requires >=20% training-objective reduction,
>=10% holdout reduction, and no regression in the equilibrium gates.

### 4.4 Quasilinear objective

Do not optimize the current single-`ky` uncalibrated scalar. Compare smooth,
differentiable spectrum-integrated candidates on the existing nonlinear ledger:
positive-growth mixing length, heat-flux-weighted mixing length, and the fixed
`spectral_envelope_ridge` coefficients. Select the objective by held-out rank
skill and local landscape conditioning, not absolute-flux appearance. Keep the
claim as screening unless absolute-flux gates independently pass.

Implement a custom implicit eigenpair JVP/VJP for eigenvector-dependent heat
flux and `k_perp` observables, using biorthogonal left/right eigenvectors and a
branch-gap fail-closed rule. Only then use the VMEC implicit-Jacobian
least-squares path. Promotion requires AD/FD agreement, >=20% training reduction,
>=10% held-out screening reduction, and matched nonlinear audits of the final
geometry.

### 4.5 True nonlinear heat-flux objective

A reduced startup/window formula is not a nonlinear optimization objective.
Use the actual post-transient ion heat flux from a converged nonlinear run:

- automatic transient exclusion based on stationarity diagnostics;
- weighted Birkhoff late-time average as in Kim et al. (2024);
- common initial phases/random numbers for paired comparisons;
- at least three replicas for accepted candidates;
- grid, timestep, and time-horizon convergence with uncertainty bars.

Chaotic long-time flux derivatives are not promoted through naive reverse-mode
unrolling. Use SPSA with common random numbers as the primary optimizer, compare
against a derivative-free trust-region/CMA-ES fallback, and use the
linear/quasilinear gradient only as a warm-start proposal. The objective combines
normalized QS, aspect, iota, and nonlinear-flux residuals in the same physical
portfolio. Promotion requires a statistically resolved >=15% nonlinear-flux
reduction on training points and >=10% on holdouts, with confidence intervals
and all equilibrium gates passing.

### 4.6 Examples and final comparison

Retain exactly four optimization examples:

- `QA_baseline.py`;
- `QA_linear_ITG.py`;
- `QA_quasilinear_ITG.py`;
- `QA_nonlinear_ITG.py`.

They follow VMEC-JAX's top-to-bottom example style: parameters at the top,
visible objective tuples, one optimization call or explicit SPSA loop, output,
and plotting. Shared solver code stays in packages; examples contain no hidden
campaign framework.

The publication panel compares baseline, linear-, QL-, and nonlinear-optimized
configurations using consistent columns: 3-D boundary colored by `|B|`, Boozer
LCFS contour lines, iota profile excluding the magnetic-axis plotting artifact,
objective history, matched nonlinear `Q_i(t)`, and flux-gradient curves with
uncertainty. Every plotted point links to its input, WOUT, run manifest, and
convergence gate.

## Phase 5: Rebuild README and Documentation

### README order

1. Name, badges, and a two-sentence purpose statement.
2. Installation and a working `gkx` first run.
3. A concise feature comparison table.
4. One consistent linear/nonlinear validation panel.
5. One matched runtime/memory panel.
6. One differentiable QA optimization panel.
7. Links to examples, documentation, citation, scope, and license.

The README is a product overview, not a manuscript or status ledger. Collision
derivations, quasilinear caveats, convergence tables, optimizer details,
parallel decomposition, and open research questions move to documentation.

### Capability table policy

Use `✅` validated, `◐` scoped/partial, `❌` not available, and `—` outside the
code's purpose. Compare GKX, GX, and GENE only on sourced categories:

- local linear and nonlinear flux-tube simulation;
- tokamak and stellarator geometry;
- electrostatic and electromagnetic models;
- adiabatic and kinetic electrons/multispecies;
- Hermite-Laguerre velocity representation;
- advanced collision operators;
- CPU and GPU execution;
- distributed production execution;
- differentiable solver/objectives;
- in-memory differentiable VMEC/Boozer geometry;
- global and neoclassical models;
- executable plotting/restart workflow.

The table is generated from a dated capability manifest with links to official
code documentation and exact GKX evidence. It must not imply that a scoped GKX
feature equals a broader production capability in GENE or GX.

### Visual system

All README figures use one typography/style module, matching width, consistent
aspect ratio (prefer 16:9), panel labels, accessible colors, and raster previews
<=350 KB. Overlapping curves use line style, markers, z-order, and inset/error
bands. Publication PDFs and raw arrays are release assets; documentation keeps
only compact PNG/WebP previews and source data needed to reproduce them.

### Documentation consolidation

Keep at most these user-facing pages: quickstart, inputs, model/equations,
geometry, numerics, collisions, linear/nonlinear workflows, diagnostics/QL,
optimization/autodiff, parallel/performance, benchmarks/validation, API,
development/testing, release scope, and references. Merge or delete duplicate
architecture, roadmap, manuscript-status, and refactor-plan prose.

## Phase 6: Final Rename to GKX

The rename is last because it is disruptive and should touch the final compact
surface only.

Planned names:

- project and repository: `uwplasma/GKX`;
- distribution: `gkx` if still available on PyPI;
- import package: `gkx`;
- executable: `gkx`;
- documentation title and artifact labels: `GKX`;
- release: `v2.0.0`.

No `spectraxgk` import or executable compatibility package will remain in the
2.0 source tree. The final 1.x package stays available on PyPI and its README/
release notes point users to the 2.0 migration guide. GitHub's repository rename
redirect handles old repository URLs; citations and archived artifacts retain
the historical SPECTRAX-GK name where provenance requires it.

Naming due diligence is mandatory before the irreversible step. As of
2026-07-17, `uwplasma/GKX` and the PyPI distribution `gkx` are available, but an
unrelated GitHub project already uses `gkx`, and the plasma code name
`TRIMEG-GKX` appears in a 2025 preprint. Reserve the namespace, check scientific
name confusion and trademarks, and use the subtitle “GKX: differentiable
Hermite-Laguerre gyrokinetics in JAX” consistently. If the collision is judged
material, stop before renaming rather than creating a second ambiguous plasma
code.

Rename gates:

- zero unintended `SPECTRAX`, `spectraxgk`, or `spectrax-gk` tokens outside the
  migration guide, historical citations, and benchmark provenance;
- clean install from PyPI candidate, `import gkx`, `gkx`, and `gkx --plot`;
- all package, docs, examples, benchmarks, coverage, repository-size, and wheel
  tests on the renamed tree;
- GitHub Actions trusted publishing configured for the new project;
- no duplicate package data or compatibility modules in the wheel.

## Phase 7: Release Gate

Release `v2.0.0` only when:

- source, test, script, README, docs, and repository budgets all pass;
- package-wide coverage is >=95%; every retained physics gate passes;
- frozen 1.7 promoted observables remain within their stated tolerances;
- the three QA transport optimizations satisfy their separate claim gates;
- current CPU/GPU runtime and memory artifacts show no unexplained regression;
- strict documentation and all four examples reproduce from a clean install;
- one release-candidate CI run is entirely green;
- GitHub release and PyPI installation are verified in isolated environments.

## Execution Order and Progress

| Phase | Completion | Next proof |
| --- | ---: | --- |
| 0. Freeze 1.7 contract and migration inventory | 35% | complete API-to-test/docs/benchmark ownership mapping |
| 1. Delete unpromoted/duplicate code | 18% | <=120 source and <=45 tool files |
| 2. Consolidate package core and API | 10% | <=70 source files / <=50k lines with parity |
| 3. Simplify tests, benchmarks, scripts | 0% | <=45 tests, <=15 scripts, >=95% coverage |
| 4. QA linear/QL/nonlinear optimization | 5% | exact QA-ESS baseline, new differentiable/noisy objectives |
| 5. README/docs/figure redesign | 2% | <=180-line README and three standardized figures |
| 6. Rename to GKX | 0% | namespace gate and complete renamed CI candidate |
| 7. Version 2.0 release | 0% | tag, GitHub release, PyPI verification |

Overall completion: 11%.

## Evidence Sources

- Latest VMEC-JAX precise-QA and ESS workflows:
  <https://github.com/uwplasma/vmec_jax/tree/main/examples/optimization>
- GX equations, Hermite-Laguerre representation, GPU/multi-GPU implementation,
  and benchmark scope: <https://gx.readthedocs.io/en/latest/> and
  <https://doi.org/10.1017/S0022377824000631>
- GENE physics and geometry capability scope:
  <https://genecode.org/details.html>
- Direct quasilinear microstability optimization:
  <https://doi.org/10.1103/PhysRevE.110.035201>
- Direct nonlinear stellarator turbulence optimization, weighted Birkhoff
  objective, and SPSA:
  <https://doi.org/10.1017/S0022377824000369>
- JAX custom/implicit derivative guidance:
  <https://docs.jax.dev/en/latest/hijax_custom_derivatives.html> and
  <https://docs.jax.dev/en/latest/advanced_autodiff.html>

## Implementation Log

- 2026-07-17: Audited `v1.7.0`, measured 220 source files/87,847 lines,
  95 test files/96,202 lines, 95 tool files/97,346 lines, eight README figures,
  and 39 MB of tracked documentation assets. Fast-forwarded the clean local
  VMEC-JAX clone to `a16d399c`, reviewed its precise-QA staged and one-call ESS
  optimizations plus turbulence bridge, and identified why the current
  SPECTRAX-GK QA examples do not move transport reliably: old staged workflow,
  single-point low-resolution objectives, finite-difference eigenvector paths,
  and a reduced nonlinear proxy rather than a true post-transient flux. Replaced
  the completed 1.x log with this finite 2.0 execution plan; no solver behavior
  or repository name changed in this planning tranche.
- 2026-07-17: Created `refactor/gkx-2.0`; made `plan.md` the sole migration
  authority; added no-regression topology and aggregate line-budget gates for
  source, tests, tools, and developer scripts; and generated
  `scripts/gkx_2_code_inventory.csv` with an explicit 2.0 disposition for every
  tracked source, test, and tool file. Removed the superseded 2,719-line
  differentiable-refactor ledger, its 658-line duplicate documentation page,
  checker command, release tests, and CI/release hooks. The focused release
  suite, Ruff, architecture policy, release-readiness policy, and validation
  coverage manifest pass; no numerical implementation changed.
- 2026-07-17: Deleted the unpromoted synthetic QA low-turbulence envelope from
  the installable API: three source modules, their dedicated validation file,
  one 785-line artifact builder, and seven generated figure/data sidecars.
  Retained the solved VMEC-JAX campaign, VMEC/Boozer objective contracts,
  RBC(1,1) long-window landscape, and matched nonlinear transport gates. Source
  is now 217 files/86,516 lines; tests are 94 files/95,997 lines; tools are 94
  files/96,264 lines. The solved-QA and objective suites, 95 release tests,
  public API import audit, Ruff, coverage-manifest regeneration, architecture
  budgets, and warning-free strict documentation build pass.
- 2026-07-17: Consolidated nonlinear NetCDF geometry, field, diagnostics, and
  bundle orchestration into one 1,069-line schema owner, reducing the source
  tree by three files and 106 lines without changing dimensions, axis order,
  spectral condensation, restart layout, or output variables. Added a reviewed
  1,000-line target rather than weakening the global module limit. The merge
  uncovered and fixed a fallback import that looked for the nonlinear
  diagnostic loader in the writer module instead of its actual owner. All 39
  runtime/restart artifact tests, 95 release tests, Ruff, architecture policy,
  coverage ownership, and strict docs pass. Source is now 214 files/86,410
  lines.
- 2026-07-17: Replaced the lazy plotting registry plus three implementation
  modules with one concrete 812-line publication-plotting owner. Benchmark,
  growth/eigenfunction, saved-runtime, and style functions retain their public
  names; zonal fitting remains in its physics-specific module and is imported
  only when requested to avoid a circular dependency. The change removes three
  source files and 75 net lines. All 26 plotting/example tests, focused release
  gates, documented API identity checks, Ruff, architecture policy, and strict
  docs pass. Source is now 211 files/86,335 lines.
- 2026-07-17: Consolidated linear, quasilinear, nonlinear-table, restart, and
  nonlinear-diagnostic artifact serialization into one runtime I/O owner.
  Deleted four format-fragment modules, removed 236 net source lines, retained
  the separate coherent nonlinear NetCDF schema and spectral-layout owners,
  and migrated every internal, test, campaign, and documentation import. All
  43 focused artifact/restart tests, focused release gates, public artifact API
  identity checks, Ruff, architecture/coverage manifests, and warning-free
  strict documentation pass. Source is now 207 files/86,265 lines.
