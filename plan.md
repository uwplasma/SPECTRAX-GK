# GKX 2.0 Simplification, Optimization, and Rename Plan

This file is the single implementation plan and execution log for the 2.0
release. Older plans, generated status dashboards, and one-off campaign plans
do not define release scope.

## Mission

Turn SPECTRAX-GK into **GKX**: a small, comprehensible, JAX-native local
gyrokinetic solver that preserves the validated 1.7 physics and comparison
results, runs efficiently on CPU and GPU, supports end-to-end derivatives for
Python optimization workflows, and exposes one obvious public workflow:

```python
import gkx

case = gkx.load("cyclone.toml")
result = gkx.solve(case)
gkx.plot(result)
```

The executable workflow is equally small:

```console
gkx cyclone.toml
gkx --plot cyclone.out.nc
```

## Measured Starting Point

Measured on `refactor/gkx-2.0` at `0fea14e9` on 2026-07-17:

| Area | Current | 2.0 target |
| --- | ---: | ---: |
| Solver package | 196 Python files / 86,073 lines | <=45 files / <=45,000 lines |
| Tests | 94 Python files / 95,986 lines | <=36 files / <=55,000 lines |
| Developer tools | 94 Python files / 96,264 lines | <=12 commands / <=18,000 lines |
| Examples | 5,270 Python lines | <=3,500 lines |
| README | 295 lines / 6 figures | <=180 lines / <=3 figures |
| Documentation | 30 pages / 1,592 static files / 34.15 MB | <=16 pages / <=250 static files / <=12 MB |
| Git database | 155 MB | <=35 MB after the 2.0 history rewrite |
| Public API | registry-driven, broad | <=30 documented names |

The final budgets are intentionally stricter than the earlier refactor plan.
Reducing file count by creating unreviewable giant modules is not acceptable:
the target median module is <=650 lines, no production module may exceed 1,200
lines, and each module must have one scientific or operational owner.

## Non-Negotiable Release Contracts

Every deletion, move, and rewrite must preserve these contracts unless a
documented bug fix deliberately changes one:

1. Frozen linear growth rates, frequencies, eigenfunction overlaps, nonlinear
   window statistics, collision limits, restart trajectories, and geometry
   arrays remain within their existing promoted tolerances.
2. The comparison suite remains the only place where comparison-code names are
   used. Physics and numerical implementation names must describe equations or
   algorithms rather than provenance.
3. CPU and GPU warm runtime and peak memory may not regress by more than 5% on
   the representative linear and nonlinear cases. A claimed speedup requires a
   fresh profiler artifact and numerical-identity gate.
4. Package-wide statement coverage remains >=95%. More importantly, literature
   limits, observed order, conservation, AD-vs-FD, branch continuity, and
   independent comparison gates retain detection power.
5. Differentiable Python paths use JAX arrays and pure functions. The executable
   may choose faster non-differentiable I/O, progress, and adaptive orchestration.
6. Generated runs, profiler traces, WOUT files, NetCDF outputs, and intermediate
   figures do not enter Git. Only compact inputs, reference summaries, and final
   compressed publication figures are tracked.
7. No reduced, synthetic, startup-window, or saturation-rule proxy is presented
   as evidence for a production nonlinear heat-flux reduction.

## Target Repository

The package will be shallow: at most two directory levels below `src/gkx`, no
parallel hierarchy for “reports”, “contracts”, “strategies”, “runtime”, or
“artifacts”, and no folder that owns fewer than two coherent modules.

```text
src/gkx/
  __init__.py          # lazy public API only
  api.py               # load, solve, scan, plot
  cli.py               # executable parsing and progress
  case.py              # immutable user configuration pytrees
  physics/
    equations.py       # normalized gyrokinetic terms and switches
    fields.py          # quasineutrality and electromagnetic field solves
    collisions.py      # collision protocol and promoted operators
    transport.py       # physical moments and flux definitions
  geometry/
    analytic.py        # slab, s-alpha, and simple analytic geometry
    miller.py          # Miller construction and derivatives
    vmec.py             # VMEC/VMEC-JAX field-line geometry
    boozer.py           # Boozer transform bridge and QS observables
  numerics/
    grids.py            # spatial and Hermite-Laguerre grids/bases
    spectral.py         # transforms, dealiasing, nonlinear bracket
    linear.py           # matrix-free linear operator and eigen solve
    nonlinear.py        # nonlinear RHS and state operations
    time.py             # fixed, adaptive, IMEX, and checkpointed stepping
    parallel.py         # independent maps and accepted decomposition
  solve/
    linear.py           # linear run and ky scan
    nonlinear.py        # nonlinear run and restart
    diagnostics.py      # growth, spectra, windows, convergence
  optimize/
    objectives.py       # growth, QL, nonlinear, and geometry residuals
    derivatives.py      # eigen, trajectory, implicit, and FD gates
    stellarator.py      # VMEC-JAX objective composition and campaigns
  io/
    config.py           # TOML loading and schema validation
    results.py          # one NetCDF schema and restart contract
    plotting.py         # user plots and publication style
```

The exact final count may be below this tree. A new file is permitted only if
it removes more code than it adds, creates a real ownership boundary, and does
not duplicate a nearby domain.

Outside the package:

```text
tests/{unit,integration,physics,release}/
benchmarks/{cases,references,run.py,figures.py}
scripts/{ci.py,release.py,audit.py,figures.py,campaign.py,profile.py}
examples/{linear,nonlinear,geometry,optimization}/
docs/
```

## Deletion Test

Before moving code, answer for every file and public function:

1. Which current equation, method, user workflow, benchmark, or release gate
   owns it?
2. Who imports it outside its own test?
3. What scientific or operational behavior would fail if it disappeared?
4. Why can it not live in the nearest domain owner?

Delete it if any answer is missing. One-off campaign logic and status rendering
are not reusable library code. Tests that merely assert that a historical file
or wrapper exists are deleted with that wrapper.

## Execution Order

### Phase 0 - Freeze the 1.7 Contract

1. Record package, dependency, VMEC-JAX, Boozer, SOLVAX, and comparison-code
   revisions in one compact manifest.
2. Freeze promoted scalar/array hashes for Cyclone, Cyclone Miller, KBM, W7-X,
   HSX, collision, restart, geometry, derivative, and optimization gates.
3. Refresh representative cold/warm CPU and office-GPU runtime and memory.
4. Map public names to implementation, tests, docs, and scientific claims.

Exit: the wheel and branch reproduce the frozen values before further deletion.

Progress: **70%**. ``gkx_1_7_release_contract.json`` now freezes dependency
revisions, the 362-name legacy public surface, 30 compact CPU/GPU performance
records, and the seven promoted release lanes. Fresh representative timings
and compact scalar/array reference hashes remain before destructive core moves.

### Phase 1 - Delete Historical Evidence Machinery

1. Classify every `tools` command as release gate, reproducible benchmark,
   active campaign, profiler, or delete.
2. Delete generated research-status dashboards, superseded admission planners,
   historical campaign launchers, failed sharding experiments, duplicate figure
   builders, and their ownership-only tests.
3. Retain compact accepted evidence in `benchmarks/references`; move raw runs to
   release assets and document the checksum/fetch procedure.
4. Delete duplicate PDF/PNG/CSV/JSON encodings. Keep one compressed PNG or SVG
   plus one small machine-readable summary per promoted result.
5. Remove ignored outputs, caches, build products, and `__pycache__` trees.

Exit: source <=120 files/68,000 lines, tools <=45 files, docs static <=20 MB,
all frozen gates pass.

Progress: **30%**. Synthetic QA evidence, the generated research-status builder,
the nonlinear release finalizer, one duplicate roadmap, 16 dashboard assets,
and their ownership-only tests are gone. The remaining 92-file tool layer and
1,576-file static evidence tree are the main blockers.

### Phase 2 - Build the Shallow Scientific Core

Apply these tranches in order; each commit must reduce both files and lines.

1. **Case/data model:** replace registries, policy wrappers, and duplicated
   configuration dataclasses with immutable pytrees in `case.py`.
2. **Physics:** merge term assembly, fields, moments, dissipation, and collision
   switches into the four `physics` owners. Keep collision implementations
   selectable through one small protocol.
3. **Geometry:** merge the 25 current files into analytic, Miller, VMEC, and
   Boozer owners. Fold report/table wrappers into tests or benchmark scripts.
4. **Numerics:** retain one grid, basis, field solve, linear operator, nonlinear
   bracket, time policy, and decomposition model. SOLVAX owns generic Krylov,
   tridiagonal, preconditioner, and matrix-free linear algebra.
5. **Solves:** eliminate `runtime.py`, `linear.py`, `nonlinear.py`, nested
   workflow runtime orchestration, and facade layers in favor of three direct
   solve modules.
6. **Diagnostics:** keep physical observables and convergence statistics near
   `solve/diagnostics.py`; move manuscript model fitting to benchmarks.
7. **Optimization:** retain only physical residuals, derivative rules, and the
   VMEC-JAX composition layer. Delete campaign status/report policy code.
8. **I/O and plotting:** one TOML schema, one result/restart NetCDF schema, one
   plotting API, and one publication style.

For every tranche run unit/physics parity, AD-vs-FD, CPU/GPU compile and warm
timing, peak memory, and package coverage. Reject abstractions that add XLA
dispatches, transposes, global reconstruction, or recompilation.

Exit: target package topology, <=45 files/45,000 lines, <=30 public names, no
module >1,200 lines, frozen physics/derivative/performance contracts pass.

Progress: **25%**. NetCDF, plotting, portfolio, zonal, VMEC transport,
diagnostic moments/growth, and Krylov ownership have been consolidated, but the
old directory topology remains.

### Phase 3 - Simplify Tests, Benchmarks, and Commands

1. Rebuild tests as four suites with one parametrized file per domain:
   `unit`, `integration`, `physics`, and `release`.
2. Preserve tests that can catch defects: manufactured solutions, exact limits,
   free-energy/conservation identities, observed-order sweeps, branch
   continuity, velocity/grid/time convergence, restart identity, AD-vs-FD,
   nonlinear window stability, and independent reference comparisons.
3. Delete tests of private wrapper names, parser plumbing, generated status
   layout, and historical file ownership.
4. Make `benchmarks/run.py` own linear, nonlinear, collision, geometry, and
   performance comparisons from compact case manifests.
5. Make `benchmarks/figures.py` regenerate every shipped scientific figure with
   one style and deterministic metadata.
6. Replace `tools/` with <=12 `scripts` commands. Reusable physics stays in
   `gkx`; shell orchestration is not importable package code.
7. Keep each local shard below five minutes and run wide coverage in bounded CI
   matrix shards.

Exit: <=36 test files/55,000 lines, >=95% package coverage, <=12 scripts/18,000
lines, and all documentation commands are reproducible.

Progress: **8%**. Suite labels exist and 1,637 ownership-only test lines were
removed with the status machinery, but 94 test files and oversized tool-test
owners still dominate.

### Phase 4 - Rebuild the QA ITG Optimizations

#### 4.1 Exact VMEC-JAX baseline

Pin the latest reviewed VMEC-JAX revision; the planning reference is
`adf2d334` (2026-07-17). Reproduce
[`QA_optimization.py`](https://github.com/uwplasma/vmec_jax/blob/main/examples/optimization/QA_optimization.py)
without changing its seed, perturbation, staged `max_mode=1..5`, ESS policy,
least-squares tolerances, QS surfaces, aspect target 6.0, or mean-iota target
0.42. Freeze the final input, WOUT checksum, QS residual, aspect, iota profile,
boundary coefficients, convergence history, and VMEC-JAX revision. This solved
equilibrium is the sole baseline for all three transport campaigns.

#### 4.2 Shared resolved portfolio

Use the same training portfolio for all objectives: at least three radii, two
field lines per radius, and a resolved positive-`ky` spectrum. Use unseen radii,
field lines, and gradients as holdouts. Promote resolution only after `ky`,
parallel-grid, Hermite, Laguerre, timestep, horizon, and statistical-window
convergence pass.

Each transport residual is dimensionless and normalized by the fixed baseline
portfolio scale. Optimization weights come from a logged Pareto sweep, not from
manual tuning after observing the final geometry. Geometry acceptance is:

- QS residual no worse than 5x the precise-QA baseline;
- aspect within 1% of 6.0;
- mean iota in `[0.40, 0.44]` with no resolved surface below 0.39;
- valid nested equilibrium and Boozer transform;
- training transport reduction >=20% and held-out reduction >=10%;
- every claimed derivative passes directional AD-vs-centered-FD agreement.

#### 4.3 Linear-growth objective

Add one normalized residual-vector callable to the exact VMEC-JAX
`objective_terms`. Use a smooth positive-growth aggregate over the full
portfolio, not one selected `ky` point. Differentiate VMEC equilibrium
implicitly and the dominant eigenvalue with the accepted eigenpair rule. Gate
spectral separation, branch continuity, directional derivatives, and holdouts.

#### 4.4 Quasilinear objective

Choose the saturation rule from the frozen stellarator holdout ranking and
state the claim as screening/correlation unless absolute-flux gates pass. The
current value-only eigenvector path is insufficient for a differentiable
optimization. Implement and test a left/right-eigenvector JVP/VJP with spectral
gap conditioning, or use a differentiable resolvent observable if the selected
model supports it. Compare AD, tangent, and centered FD before optimization.

#### 4.5 Nonlinear heat-flux objective

Optimize the actual post-transient ion heat-flux mean, not
`nonlinear_heat_flux_proxy`. Use a deterministic fixed-step differentiable
trajectory, common-random-number initial states, a smooth late window, and
checkpointed discrete adjoints. Horizon continuation is allowed, but the final
objective window must pass running-mean, subwindow, grid, timestep, and replicate
gates. If chaotic tangent growth makes the gradient unusable, switch the outer
optimization to an explicitly documented noise-robust trust-region/SPSA/CMA
method; do not label a proxy or finite startup trace as nonlinear optimization.

#### 4.6 Optimization acceptance and figures

Run matched baseline-versus-final long nonlinear audits for the linear, QL, and
nonlinear optimized equilibria. Require confidence intervals and report negative
results honestly. Produce one consistent publication panel containing:

1. 3D boundaries colored by `|B|` with the same camera and color limits;
2. unfilled Boozer `|B|` contours with the same levels;
3. iota profiles excluding the unresolved axis point;
4. normalized objective and constraint histories;
5. matched nonlinear `Q_i(t)` traces and late-window means with uncertainty;
6. training/holdout transport summaries.

Only this final panel appears in the README. Landscape, conditioning,
convergence, Pareto, and derivative figures belong in documentation.

Exit: all three scripts differ from upstream QA only by imports, portfolio
configuration, one objective term, reporting, and accepted validation; all
geometry and transport gates pass.

Progress: **15%**. Three examples exist, but they use a fixed one-point weight;
the QL and nonlinear scripts are value/proxy workflows, and the tracked panel
does not demonstrate an accepted transport change.

### Phase 5 - Rebuild README and Documentation

#### README structure

Keep the README below 180 lines:

1. title, badges, and a two-sentence description;
2. `pip install gkx`, one executable command, and the three-line Python API;
3. six concise highlights;
4. one capabilities/comparison table;
5. one combined validation/performance figure;
6. one differentiability/QA-optimization figure;
7. one collision/physics-verification figure;
8. links to examples, full documentation, contributing, citation, and license.

No release policy, detailed equations, convergence discussion, claim matrix,
or campaign narrative belongs in the README.

#### Comparison table

Use `yes`, `limited`, and `no` with a visible legend. Compare GKX, GX, and GENE
only on source-verifiable features: local flux-tube linear/nonlinear physics,
electrostatic/electromagnetic models, adiabatic/kinetic electrons, tokamak and
stellarator geometry, collisions, CPU/GPU execution, distributed execution,
Python API, built-in plotting, automatic differentiation, and equilibrium-to-
turbulence optimization. Pin the audited revisions and link every column to its
source/documentation. “Limited” is mandatory for experimental decomposition or
scoped physics; absence of evidence is not converted to `no`.

Primary comparison sources:

- [GX repository](https://bitbucket.org/gyrokinetics/gx/) and
  [documentation](https://gx.readthedocs.io/en/latest/)
- [GENE project and release documentation](https://genecode.org/)
- [GX method paper](https://arxiv.org/abs/2209.06731)

#### Documentation structure

Consolidate 30 pages into <=16 user-facing pages: quickstart, cases/inputs,
physics/model, geometry, collisions, numerics/solvers, outputs/plotting,
parallel execution, differentiability, stellarator optimization, benchmarks,
performance, validation/testing, API, examples, and references. Generated
status/roadmap pages are removed. Equations, derivations, normalization,
algorithms, conditioning, convergence, negative evidence, and exact reproduction
commands move here from the README.

Exit: README/docs links and image dimensions pass automated gates, all examples
run, and every displayed value is regenerated from a tracked compact manifest.

Progress: **8%**. The duplicate roadmap and generated status sections are gone,
and the warning-as-error Sphinx build passes; page and static-asset counts still
exceed the final budgets.

### Phase 6 - Atomic Rename to GKX

This is the final code-change phase, after all scientific and structural gates
pass. Do not maintain dual internal package trees.

1. Reserve the currently unclaimed PyPI project `gkx` and rename the GitHub
   repository from `uwplasma/SPECTRAX-GK` to `uwplasma/GKX`.
2. Rename `src/spectraxgk` to `src/gkx`, imports to `gkx`, executable to `gkx`,
   project metadata to `gkx`, and user-facing names to GKX.
3. Rename TOML/output metadata keys only where they are package branding rather
   than physical schema. Provide a one-shot 1.7-to-2.0 input migration command,
   not permanent runtime aliases.
4. Coordinate VMEC-JAX's optional turbulence import from `spectraxgk` to `gkx`
   and pin mutually compatible release versions.
5. Update CI, release, Codecov, docs URLs, badges, issue templates, citation,
   examples, output names, environment names, and benchmark provenance.
6. Permit “SPECTRAX-GK” only in migration/release-history text and “GX”/“GENE”
   only in comparison/benchmark provenance.
7. Build and install the wheel in a clean environment; verify `import gkx`,
   `gkx`, `gkx --plot`, CPU execution, and GPU execution.

Exit: repository, package, import, executable, documentation, and generated
metadata consistently say GKX; no internal legacy alias remains.

Progress: **0%**. The names are available as of 2026-07-17 but are not reserved.

### Phase 7 - Release and History Rewrite

1. Run Ruff, type checks, bounded test shards, >=95% merged coverage, docs build,
   package build, wheel smoke, frozen physics gates, derivative gates, comparison
   gates, and CPU/GPU runtime-memory gates.
2. Confirm no tracked file exceeds 2 MB and no unapproved generated output is
   present.
3. Rewrite pre-2.0 history to remove historical figures, plans, raw outputs, and
   transient artifacts; force-push only after preserving the signed 1.7 tag and
   publishing a migration notice.
4. Require fresh-clone `.git` <=35 MB and checkout <=25 MB excluding the local
   environment.
5. Merge the single draft refactor PR, tag `v2.0.0`, publish GitHub release notes
   and PyPI `gkx`, and verify clean installation on CPU and office GPU.

Exit: all gates green, release artifacts reproducible, and published claims match
the evidence.

Progress: **0%**.

## Test and Evidence Matrix

| Change | Required gates |
| --- | --- |
| Equation/operator consolidation | manufactured solution, exact limit, free-energy identity, reference parity |
| Geometry consolidation | VMEC array parity, Boozer parity, coordinate identities, AD-vs-FD |
| Linear solver | observed order, eigen residual, branch continuity, growth/frequency/eigenfunction parity |
| Nonlinear solver | bracket identities, conservation, timestep/grid convergence, restart identity, window statistics |
| Collision operator | null space, conservation, entropy production, collisional limits, finite-wavelength convergence |
| Differentiable objective | primal parity, tangent/adjoint consistency, centered FD, conditioning and step-size sweep |
| Parallel path | serial identity, deterministic reduction, CPU/GPU memory, strong scaling on a large case |
| QA optimization | exact baseline, constraint gates, train/holdout reductions, long replicated nonlinear audit |
| CLI/I/O | clean install, default run, progress/ETA, TOML echo, NetCDF round trip, `--plot` |

## Commit Sequence

Use large coherent commits, each independently green:

1. freeze compact 1.7 manifest and performance baseline;
2. delete status/campaign/artifact machinery and redundant static evidence;
3. consolidate case model and physics;
4. consolidate geometry;
5. consolidate numerical operators and time integration;
6. consolidate solve, diagnostics, I/O, and plotting;
7. consolidate objectives and derivative rules;
8. rebuild tests, benchmarks, scripts, and coverage shards;
9. reproduce exact VMEC-JAX QA baseline;
10. close linear, QL, and nonlinear QA campaigns and final panel;
11. rebuild README/docs and comparison table;
12. perform atomic GKX rename, history rewrite, and release.

## Execution Log

- `0ec50924`: added architecture budgets and repository inventory.
- `b958b6e2`: removed synthetic QA turbulence evidence.
- `4ad40db9`: unified nonlinear NetCDF schema writing.
- `949c66ab`: consolidated publication plotting.
- `b25dc427`: consolidated result and restart I/O.
- `3dde7313`: consolidated objective portfolio and zonal owners.
- `9ed4e619`: simplified VMEC transport objective ownership.
- `5b973d95`: consolidated diagnostic moments and growth owners.
- `0fea14e9`: unified matrix-free Krylov kernels.
- 2026-07-17 compact-contract tranche: froze the 1.7 dependency/API/performance
  and release-lane contract, deleted 7,505 lines of status/finalizer machinery,
  removed 16 duplicate dashboard assets and the duplicate roadmap, retained the
  numerical QL evidence, and passed 238 focused tests plus release, validation,
  Ruff, and warning-as-error documentation gates.
- 2026-07-17 planning audit: pulled VMEC-JAX `adf2d334`; identified the exact
  turbulence objective seam, fixed-weight one-point limitation in current QA
  examples, 94-file/96k-line tool blocker, 1,592-file static evidence blocker,
  and atomic rename requirements.

## Current Completion

| Lane | Completion |
| --- | ---: |
| Frozen 1.7 contract | 70% |
| Deletion/repository trim | 30% |
| Scientific-core simplification | 25% |
| Test/benchmark/command simplification | 8% |
| QA linear/QL/nonlinear optimization | 15% |
| README/docs rebuild | 8% |
| GKX rename | 0% |
| Release/history rewrite | 0% |

Weighted overall completion: **20%**. The next implementation tranche remains
Phase 0 plus Phase 1: add compact promoted scalar/array hashes and refresh a
bounded representative performance subset, then classify and delete the next
superseded campaign/figure family before moving scientific-core files.
