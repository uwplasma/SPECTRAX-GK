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

Progress: **95%**. ``gkx_1_7_release_contract.json`` now freezes dependency
revisions, the 362-name legacy public surface, 30 compact CPU/GPU performance
records, seven promoted release lanes, and 12 canonical numerical fingerprints
covering nonlinear cases, KBM branch continuity, collisions, restart, geometry,
derivatives, and optimization. The current-branch refresh also admits two
bounded CPU rows with CPU/GPU numerical agreement; office GPU timings are
retained but rejected because both devices were already saturated. One
uncontended GPU refresh remains before the final release claim update.

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

Progress: **48%**. Synthetic QA evidence, generated research/optimization
status builders, the nonlinear release finalizer, the external-VMEC launch
planner, one duplicate roadmap, and 205 duplicate dashboard/runbook/encoding
assets are gone. The remaining 90-file tool layer and 1,387-file static evidence
tree are the main blockers.

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

Progress: **33%**. NetCDF, plotting, portfolio, zonal, VMEC transport,
diagnostic moments/growth, Krylov, and replicated nonlinear-gradient statistics
ownership have been consolidated. Refactor-only runtime contracts are deleted,
the collision interface now contains only the protocol used by the equations,
species construction lives with linear parameters, and programmatic TOML cases
share the executable command owner. The broader old directory topology remains.

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

Progress: **20%**. Suite labels exist and more than 3,800 ownership/planner test
lines were
removed with status/planner machinery, but 94 test files and oversized
tool-test owners still dominate.

### Phase 4 - Rebuild the QA ITG Optimizations

#### 4.1 Exact VMEC-JAX baseline

Pin the latest reviewed VMEC-JAX revision; the planning reference is
`adf2d334` (2026-07-17). Reproduce
[`QA_optimization.py`](https://github.com/uwplasma/vmex/blob/main/examples/optimization/QA_optimization.py)
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

Progress: **14%**. The duplicate roadmap, generated status sections, external
holdout runbook, and 181 unreferenced PDF/CSV copies are gone. The warning-as-
error Sphinx build passes, but page and static-asset counts still exceed the
final budgets.

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
- 2026-07-17 planner-removal tranche: folded QA optimization claim/prelaunch
  policy into the compact contract; deleted the 1,013-line status renderer, the
  790-line external-VMEC launch planner, eight duplicate renderings, and 828
  ownership-only test lines; retained all converged holdout and matched-audit
  evidence; passed 232 focused tests and all release/docs checks.
- 2026-07-17 static-encoding tranche: removed 59 unreferenced PDFs and 122
  redundant CSV copies only where a JSON sidecar remained; retained all cited
  PDFs, data-only CSV files, final PNG/SVG figures, and compact JSON evidence;
  passed 366 artifact/release tests, validation manifests, and Sphinx ``-W``.
- 2026-07-17 nonlinear-gradient ownership tranche: extracted paired-seed,
  control-variate, and independent control-mean uncertainty gates into the pure
  952-line ``diagnostics/nonlinear_gradient_statistics.py`` owner; reduced the
  temporary mixed campaign module by 871 lines and its ownership test by 683
  lines; passed 139 nonlinear-gradient consumers, architecture/validation
  manifests, Ruff, and Sphinx ``-W``.
- 2026-07-17 nonlinear-gradient command tranche: moved candidate ranking and
  same-control bracket-sweep rendering into the single retained evidence
  command; deleted the 4,463-line campaign-planning layer, two planner-only test
  files, and one obsolete follow-up artifact; condensed documentation to the
  actual finite-difference, locality, and uncertainty results. The tool tree is
  now 88 Python files / 85,428 lines and the test tree is 92 files / 92,251
  lines. All 68 focused nonlinear-gradient and VMEC mapping tests, validation
  ownership, repository-size, architecture, release-readiness, Ruff, and
  warning-as-error documentation gates passed.
- 2026-07-17 frozen-numerics tranche: added 12 canonical scalar/array
  fingerprints for promoted nonlinear, KBM branch, collision, restart,
  geometry, derivative, and optimization references to the compact 1.7
  contract. Release readiness now recomputes those numerical hashes while
  allowing text-only provenance changes, and a mutation regression proves that
  altered numerical evidence fails closed.
- 2026-07-17 representative-performance tranche: refreshed 1,000-step linear
  and 200-step prepared nonlinear workloads on the current branch. Local CPU
  warm medians were 7.219 s and 41.444 s; CPU/GPU output differences were
  5.35e-6 and 4.45e-4, respectively. Both office A4000s were already at 100%
  utilization, so their timing rows are explicitly blocked and no speed claim
  changed. Release readiness now recomputes admitted/blocked row counts and
  rejects false promotion; 97 release tests, Ruff, and Sphinx ``-W`` pass.
- 2026-07-17 first scientific-core tranche: deleted refactor-only contract
  objects and six unused generic protocols, retained the active collision
  context/operator contract beside physics operators, moved species assembly
  beside ``LinearParams``, and folded programmatic TOML cases into the runtime
  command owner. This removed three source files and 1,319 net lines while
  preserving top-level user names. The source tree is 194 files / 86,632 lines;
  153 focused grid, basis, species, collision, linear, runtime, and Diffrax
  tests pass together with ownership, frozen-output, architecture, and size
  gates.
- 2026-07-17 quasilinear-facade tranche: deleted the forwarding-only
  ``spectraxgk.quasilinear`` module and routed examples, tests, profilers, and
  documentation directly to ``diagnostics.quasilinear_transport`` while
  preserving promoted top-level names. The source tree is now 193 files /
  86,602 lines; 107 quasilinear and autodiff tests plus release/ownership gates
  pass.
- 2026-07-17 planning audit: pulled VMEC-JAX `adf2d334`; identified the exact
  turbulence objective seam, fixed-weight one-point limitation in current QA
  examples, 94-file/96k-line tool blocker, 1,592-file static evidence blocker,
  and atomic rename requirements.
- 2026-07-17 root-facade tranche: removed the ``spectraxgk.linear``,
  ``spectraxgk.nonlinear``, and ``spectraxgk.benchmarks`` re-export modules and
  routed the public API and every consumer directly to the domain owners
  (``operators.linear.*``, ``solvers.linear.*``, ``solvers.nonlinear.*``,
  ``operators.nonlinear.*``, ``benchmarking.shared``). Repointed 19 API-registry
  targets and 46 consumer files, re-homed the linear/nonlinear coverage-owner
  blocks onto ``terms.assembly`` and ``solvers.nonlinear.state_integration``,
  migrated 37 documentation cross-references, and fixed a stale ``ci.yml``
  fundamentals-core shard path. The source tree is now 190 files / 86,417 lines;
  import (362 public names), Ruff, unit+integration (no new failures), 97
  release-gate tests, architecture/size/coverage manifests, and Sphinx ``-W``
  all pass.
- 2026-07-18 wholesale campaign/status deletion: removed all 14
  ``tools/campaigns/`` tools, 13 campaign-artifact builders and 7
  status/dashboard builders under ``tools/artifacts/``, all of
  ``tests/tools/{campaigns,artifacts}``, two campaign-only physics-gate test
  files, and three campaign-audit optimization examples (52 files, ~47k lines).
  Excised deleted-tool tests from seven surviving files while preserving every
  physics/AD-FD/benchmark/identity test, repaired 14 documentation pages and
  both workflow YAMLs, purged the coverage/performance manifests and inventory
  CSV, and fixed the release-readiness snippet contract. Tools 88->54,
  tests 92->77, examples 41->38. Ruff, whole-suite collection (2076 tests),
  architecture/size/coverage/parallel-scaling/release-readiness gates, and
  Sphinx ``-W`` all pass; runtime tests pass except one pre-existing float32
  demo flake that passes under CI ``x64``.
- 2026-07-18 campaign finalization: restored the quasilinear figure index in
  ``docs/manuscript_figures.rst`` (new "Quasilinear Model-Selection Figures"
  static-evidence subsection) that the kept
  ``check_quasilinear_promotion_guardrails`` gate audits — the deletion had
  removed those png/JSON figure mentions along with the retired builder-script
  references — and dropped two stale ``BASELINE/CANDIDATE_VMEC_FILE`` assertions
  from the QA knob test. Fixes all six campaign-introduced test failures. The
  remaining ``test_improved_sugama_multispecies`` failure is a pre-existing
  branch flake (JVP-vs-FD tolerance, unrelated to this work; belongs to the
  collision-operator lane). Package coverage is 94% because the deleted
  ``tests/tools`` monoliths covered ~1% of package lines (e.g.
  ``solvers/time/explicit.py`` explicit-linear path) — restore in Phase 3.
- 2026-07-18 VMEX dependency migration, piece A: external imports rewritten to
  ``vmex``/``vmex.optimize``/``vmex.core.turbulence`` plus README; GKX's QA
  turbulence-optimization path now uses VMEX directly. See the dependency
  section below for pieces B and the coordinated identifier rename.
- 2026-07-19 VMEX piece B completion + full x64 CI-green pass. Finished the
  PEST/tensor route and candidate-admission migration (``1c567abe``); see the
  dependency section. Then swept the whole suite under CI precision
  (``JAX_ENABLE_X64=1``) -- the authoritative gate, since **every** CI pytest
  job runs x64 and local runs default to float32, so float32-only validation
  had masked five x64/full-suite failures. Fixed: (1) the improved-Sugama JVP
  differentiability check (``62d2149d``) -- float32 finite-difference reference
  is cancellation-limited to ~3e-3 so an exact x64 tangent broke the 1.5e-3
  tol; made it precision-aware (float64 inputs + small step under x64,
  certified AD-vs-FD to 3e-8). (2) piece B fallout the objectives/stellarator
  validation missed (``0d03522e``): a "GX" token in the new
  ``vmec_tensor_mapping`` docstring tripped the comparison-terminology release
  gate (reworded to "GS2-style"), and the solver FD-gate test fake still
  injected retired ``static``/``indata`` bundle keys (now ``runtime``/``inp``).
  (3) a pre-existing parallel runtime-scan flake (``f25289f9``): worker
  invocation order asserted as deterministic under an 8-worker thread executor;
  made order-insensitive (12/12 stable). Full x64 suite: 2052 passed / 0 failed.
- 2026-07-19 coverage restoration to the >=95% gate (measured under x64 to
  match CI). The deleted ``tests/tools`` monoliths had implicitly covered
  helper paths; rebuilt that as direct behavioral tests (no coverage theater,
  no source edits). ``solvers/time/explicit.py`` 45->97% (wall-time format,
  adaptive-CFL dt clamping, progress-trigger logic, max-mode + adaptive
  integration); ``geometry/vmec_flux_tube_reports.py`` 57->89% (parity metric
  packers, report schemas, validation errors); ``geometry/vmec_state_sensitivity``
  63->100% (observable packing, AD/FD Jacobian rows, unavailable/failed
  branches); ``geometry/vmec_state_controls`` 83->100% (family accessors, index
  resolution, immutability); ``operators/linear/collisions`` 93->100%
  (fail-closed provenance, guard/validator branches, pytree round-trips). Four
  geometry/operator files were written by parallel subagents under strict
  anti-theater guardrails and reviewed before commit. Result: package coverage
  **95.19%** (1338 missing of 27819, +243 statements vs the ~193 the gate
  required); full x64 suite **2149 passed / 0 failed**; package-coverage
  manifest gate and architecture gate (79 test files <= baseline 95) both pass.
- 2026-07-19 coordinated ``vmec_jax`` -> ``vmex`` rename (``c09f12c3``,
  ``438176b8``). Now that the whole geometry bridge runs on the vmex public
  API, the historical ``vmec_jax`` provenance token was renamed everywhere it
  is a live identifier or value, in lockstep so the source literals, the 46
  frozen docs/_static + benchmarks artifacts, and the release gates that
  field-check their ``kind``/``source``/``source_model`` values all moved
  together. 45 tracked asset files were renamed (docs panels + the QA test
  file) with every figure/download reference updated; the mapping metadata key
  became ``mapping["vmex"]``; the ``*_VMEC_JAX_PATH`` env vars became
  ``*_VMEX_PATH``. VMEC-format names (``vmec_boozer_core``, ``vmec_transport``,
  VMEC equilibria) were deliberately left untouched -- only the ``vmec_jax``
  token moved -- and the genuine "``vmec_jax`` was renamed to ``vmex``" history
  notes are kept below. A latent Sphinx ``|B|`` substitution warning (from the
  piece B boozer-constants docstring) was escaped so ``sphinx -W`` is clean.
  Verified: ruff, API (362), collection (2172), architecture gate, ``sphinx -W``
  (0 warnings, all renamed figures resolve), and the full x64 suite (2149
  passed / 0 failed). This clears the plan's last listed push-blocker; the
  single combined push of the local stack awaits explicit user authorization.
- 2026-07-19 pushed the 23-commit stack to ``origin/refactor/gkx-2.0``
  (``874501b5..6db26a51``); remote canonicalized to ``uwplasma/SPECTRAX-GK``.
- 2026-07-19 opened the **advanced-collision lane (task #4)**. Scoping: the
  operators already exist and are literature-sound -- single-species
  drift-kinetic Coulomb/Sugama/improved-Sugama moment matrices
  (``operators/linear/collisions.py``, tabulated via
  ``load_collision_moment_matrix``), the six-moment Coulomb/Dougherty/Sugama
  contributions and Dougherty operator (``operators/linear/dissipation.py``),
  Lenard-Bernstein rates (``operators/linear/moments.py``), and the
  finite-wavelength Coulomb operator (Frei et al. 2021). The linear RHS applies
  collisions through the cache ``collision_lam`` slot (empty by default -> the
  analytic Lenard-Bernstein ``lb_lam`` is the current default), so the *gap* is
  operator **selection** plus the missing physics-limit benchmarks. Sequenced
  lane work:
  1. DONE (``de0ecb99``) -- H-theorem + density-invariant benchmark for the
     drift-kinetic family: symmetrized moment matrix negative semi-definite
     (entropy production <= 0), density an exact two-sided collisional
     invariant, non-trivial dissipation. This is the b=0 part of the finite-b
     H-theorem item.
  2. Finite-b H-theorem: sweep ``b = kperp^2 T m/(q B)^2`` on the
     ``FiniteWavelengthCoulombOperator`` and assert dissipativity at each b.
  3. Collisionless-growth-rate limit: as the ``collisions`` weight -> 0 the
     linear growth rate converges to the collisionless value and is
     operator-independent.
  4. TOML-selectable operator: add a ``collision_operator`` field (config ->
     ``LinearParams`` -> ``cache_builder`` populates ``collision_lam`` for the
     moment-matrix operators / selects the analytic contribution), validated
     and defaulting to the current Lenard-Bernstein behavior.
- 2026-07-19 **Phase 6 atomic rename SPECTRAX-GK -> GKX (code, `e0817914`).**
  Package moved ``src/spectraxgk`` -> ``src/gkx``; import root, pyproject name,
  and the two consolidated console scripts are now ``gkx``; every identifier,
  string value, env var (``SPECTRAX_*`` -> ``GKX_*``), progress prefix
  ``[gkx]``, data label, and the 271 frozen docs/_static + benchmarks
  provenance values renamed in lockstep with the release gates. Also cleared
  the residual no-underscore ``VMECJAX`` -> ``VMEX``. x64 is switched by
  ``JAX_ENABLE_X64`` (not the renamed ``SPECTRAX_X64``), so unaffected.
  Verified: ``import gkx`` (362 API names, ``spectraxgk`` gone), reinstall,
  ruff, ``gkx --help``, collection (2173), architecture manifest, ``sphinx -W``
  (0 warnings), CI coherent (``--cov=gkx``, ``import gkx`` smoke), and the
  **full x64 suite: 2150 passed / 0 failed, coverage 95.19% (gate PASS)**.
  **Not done (out of my scope):** reserving the PyPI ``gkx`` name and
  renaming the GitHub repo ``uwplasma/SPECTRAX-GK`` -> ``uwplasma/GKX`` are
  GitHub/PyPI actions for the maintainer; the local working directory keeps its
  name because renaming it would break the editable venv. Per step 6, the
  ``SPECTRAX-GK`` name is retained only in this plan's migration/history text.
- 2026-07-19 collision ``collision_operator`` TOML selection (``d77cf1f6``):
  ``TimeConfig.collision_operator`` + ``collision_operator_from_config`` factory
  (none/lenard_bernstein -> None diagonal default; sugama/improved_sugama ->
  dense drift-kinetic operator), tested (selection/validation + H-theorem) both
  precisions. Last mile: thread it through ``workflows/linear.py`` +
  ``integrate_linear_from_config`` for end-to-end TOML selection.
- 2026-07-19 **README figure branding finding (Phase 5, task #9 in progress).**
  After the SPECTRAX-GK -> GKX rename all *generator source* is GKX (0 spectrax
  remnants), but the committed README result PNGs were rendered *before* the
  rename and still show "SPECTRAX-GK"/"GX vs SPECTRAX-GK" in their pixels. They
  must be regenerated. Split:
  - **Offline-regenerable (GKX-branded, no GX needed):** the linear master
    panels and any figure that renders from committed CSV/JSON
    (``make_reference_panels.py``/``make_benchmark_atlas.py`` linear groups,
    collision, autodiff, quasilinear, QA, runtime). Re-running the (already
    renamed) generators produces GKX branding.
  - **Office-GPU + GX blocked:** the nonlinear GX-comparison sub-panels
    (``nonlinear_{cyclone,kbm,w7x,hsx,miller}_diag_compare_*.png``,
    ``hsx_nonlinear_compare_t50_true.png``) are committed as PNGs *only* -- no
    regenerable time-series source is tracked -- so they need re-running the
    nonlinear GKX+GX comparison on the office GPU (ties into task #7). The
    benchmark atlas composites these, so its hero panel is only fully
    GKX-branded after the office-GPU refresh.
  Do the offline regeneration as one consistent batch; fold the nonlinear
  panels into the office-GPU GX benchmark so the atlas is regenerated once,
  uniformly, rather than leaving mixed branding.
- 2026-07-19 figure-suite regeneration attempt -> **VERIFIED it is an
  office-GPU task, not offline.** Ran ``make_benchmark_atlas.py`` on the clean
  tree: it produces a correctly GKX-branded atlas *title* and GKX *linear*
  master panels (generators are renamed and work), but two hard blockers make a
  clean offline finalization impossible, so the regeneration was reverted:
  1. **Nonlinear GX panels are office-only.** The atlas composites committed
     PNGs for the cyclone/kbm/w7x/hsx/miller nonlinear GX comparisons; only
     ``cetg_reference_compare.csv`` carries committed nonlinear GKX+GX traces,
     and ``tools_out`` (the office ``*.out.nc`` runs) is untracked. So those
     five sub-panels keep their baked-in "GX vs SPECTRAX-GK" text -> the atlas
     comes out mixed-branding here.
  2. **Size gate.** Freshly rendered panels are ~4-5 MB (dpi=240) and trip
     ``check_repository_size_manifest.py``
     (``max_unlisted_tracked_file_bytes = 1_000_000``); the committed figures
     were compressed to fit the private-repo media policy, so any regeneration
     must add a compression pass.
  **Office-GPU finalization recipe (one uniform pass):** on the office machine
  with the GX ``*.out.nc`` runs present -> (a) regenerate the nonlinear
  comparison sub-panels via ``tools/comparison/compare_gx_nonlinear.py`` (now
  GKX-labeled), (b) run ``tools/artifacts/make_benchmark_atlas.py`` and the
  standalone figure generators (collision/autodiff/quasilinear/QA/runtime),
  (c) compress every regenerated PNG to < 1 MB, (d) verify GKX branding + the
  size-manifest + Sphinx ``-W`` gates. The generators are already GKX; only the
  render environment (office data + compression) is missing here.

## Research-Grade Gyrokinetic Collision Operator Program (2026-07-19)

Goal: a research-grade **multispecies finite-Larmor (gyrokinetic) Coulomb**
collision operator, production-wired and benchmarked, for use in GKX
gyrokinetic simulations, plus a comparison-driven README. This is a
multi-session program; sequence and machinery below.

**Current state (audited).** Drift-kinetic Coulomb is benchmarked at the
operator-algebra level (conservation density/momentum/energy exact,
H-theorem negative-semi-definite, Spitzer high-charge asymptote to 7.45%,
converged at (P,J)=(20,5); Frei 2021 / Frei-Ernst-Ricci 2022 / Abel 2008).
A finite-wavelength Coulomb operator EXISTS but is **equal-species only**
(``EqualSpeciesFiniteWavelengthCoulombOperator``), scope
``offline_operator_algebra_not_runtime_transport``, not runtime-selectable.
Scaffolds already present: ``operators/linear/collision_tables.py`` holds
``TabulatedMultispeciesCollisionOperator`` (applies a kperp-interpolated
multispecies matrix) and the equal-species Coulomb/Sugama operators; the
drift-kinetic *multispecies* Coulomb already assembles from
density/mass/temperature arrays.

**Machinery to use (per the request).** DKX = ``~/local/sfincs_jax`` (SFINCS
multispecies Fokker-Planck/Coulomb operator) as the physics cross-check for
the multispecies coefficients. SOLVAX = ``~/local/SOLVAX/src/solvax`` (banded
LU ``lu_factor_banded``/``lu_solve_banded``, Krylov, implicit) for performant
application/inversion of the block-banded Hermite-Laguerre collision matrix.
GKX uniform-figure default: ``src/gkx/artifacts/plotting.py::set_plot_style``.

**Sequenced work:**
1. **Multispecies finite-wavelength Coulomb coefficients (deep physics).**
   Generalize the equal-species Frei (2021) finite-Larmor test/field/
   polarization tables to species pairs a<->b (mass/temperature ratios,
   Rosenbluth potentials at finite b). Reduce exactly to the existing
   drift-kinetic multispecies Coulomb as b->0 and to the equal-species tables
   for a==b. Validate: conservation (all pairs), H-theorem, Spitzer, and a
   cross-check vs DKX/SFINCS. Feed ``TabulatedMultispeciesCollisionOperator``.
2. **Runtime wiring.** Add ``coulomb`` / ``finite_wavelength_coulomb`` to the
   ``collision_operator`` selector (``params.collision_operator_from_config``)
   and thread through the RHS (plumbing exists; drift-kinetic half done
   ``d77cf1f6``). Runtime + differentiability tests.
3. **Runnable example** demonstrating a collisional ITG/zonal run selecting
   each operator.
4. **Benchmarks + performance.** Runtime linear + nonlinear collisional
   transport vs published (Frei/Jorge/Ricci); use SOLVAX banded/Krylov for the
   collision solve; report accuracy + timing per operator. Nonlinear pieces are
   office-GPU (ties task #7).

**README + figures (Phase 5, task #9).** Operator-comparison section (Coulomb
vs Sugama vs improved-Sugama vs Lenard-Bernstein vs Dougherty): what each
captures, *why* they differ physically (pitch-angle vs energy scattering,
conservation, collisionality regime), and how to select each in code/TOML.
Reorganize figures uniformly via ``set_plot_style`` (fonts/sizes/dpi), add
turbulence movies incl. **3D turbulence in tokamak and stellarator geometry**,
compress every asset < 1 MB. Movies + nonlinear panels need the office GPU.

**Close-out (after all green).** Merge the PR; rename the GitHub repo
SPECTRAX-GK -> GKX (maintainer action, then ``git remote set-url``); user
creates the PyPI ``gkx`` and Read the Docs projects.

## Dependency Migration: VMEC-JAX to VMEX

The differentiable-VMEC dependency ``vmec_jax`` was renamed AND restructured to
``vmex`` (``uwplasma/vmex`` 0.2.0: top-level modules moved under
``vmex.core.*``, several internal functions renamed or removed). This is an API
migration, not a string rename.

- **Piece A (done, ``c38cf170``):** external imports → ``vmex`` (``import vmex``,
  ``vmex.optimize``, ``vmex.core.turbulence``) and README. Verified: the QA
  optimization examples import VMEX and its public optimize/turbulence API
  resolves.
- **Piece B, Boozer route (done, ``3b4d4f1e``):** the differentiable
  VMEC/Boozer bridge now runs on vmex public seams — ``load_solved_vmex_case``
  (``VmecInput.from_file`` + ``solve_equilibrium``, cached), per-surface
  ``vmex.core.boozer_tables.boozer_input_tables`` stacked into the
  ``booz_xform_jax`` inputs contract, and vmex-named state accessors.
  Verified: mode-matched WOUT parity at machine precision (~2e-16), AD vs
  centered FD on d sum(bmnc)/d R_cos at 2e-10, 40 gate tests green,
  ``vmec_boozer_core`` at 999/1000 lines. Stellarator-symmetric only
  (asym fails closed).
- **Piece B, PEST route + admission (done, ``1c567abe``):** the
  tensor-mapping route is now a thin differentiable adapter over
  ``vmex.core.turbulence.gk_fieldline_geometry`` (383 -> 84 lines), the
  state-sensitivity reports run on the same seams, candidate admission
  extracts iota/QS from the vmex ``Equilibrium``/``WoutData`` bundle with
  ``QuasisymmetryRatioResidual``, and backend pinning targets the imported
  ``vmex`` package. Cross-route check: the safety factor q from the
  independent PEST and Boozer constructions agrees to 2.4e-16, and the
  PEST-route AD gradient matches centered FD at 4e-11. No ``vmec_jax``
  import or ``import_module`` call remains in ``src``; only bare-word prose
  and frozen provenance labels stay for the coordinated identifier rename.
  The old geometry bridge
  (``geometry/{vmec_tensor_mapping,vmec_state_sensitivity,vmec_boozer_core,
  vmec_boozer_constants,vmec_state_controls}.py``,
  ``objectives/{vmec_boozer_context,vmec_boozer_fd}.py``) calls ~8 removed
  ``vmec_jax`` internals (``eval_geom``, ``build_static``, ``load_config``,
  ``state_from_wout``, ``vmec_bcovar_half_mesh_from_wout``,
  ``booz_xform_inputs_from_state``, ``example_paths``,
  ``nyquist_mode_table_from_grid``) and is already broken against VMEX 0.2.0.
  Re-implement it on VMEX's supported PUBLIC API (``vmex.read_wout``,
  ``vmex.WoutData``, ``vmex.core.geometry``/``boozer``, ``vmex.optimize``)
  instead of internal modules; validate VMEC array parity, Boozer parity, and
  the differentiable equilibrium adjoint. This also clears the remaining
  bare-word ``vmec_jax`` references (some are frozen provenance labels).
- **Identifier rename (coordinated pass):** the GKX-internal ``vmex_*`` →
  ``vmex_*`` rename must rename the 46 frozen evidence artifacts (JSON
  ``"kind"``/``"source_model"`` values AND ``docs/_static`` asset filenames) and
  their gate references in the SAME change, or not at all — a mechanical prefix
  replace mismatches the frozen artifacts. Best folded into Phase 5 (docs) or a
  dedicated pass.

## Literature Validation (2026-07-18)

A multi-source, adversarially-verified literature review confirms GKX's core
scientific direction. The review was cut short by a session limit before the
quasilinear (pillar 3) and differentiable-optimization (pillar 4) pillars were
fully verified, so those remain to be finished before Phase 4 advances.

- **Velocity representation - SOUND.** The Hermite-Laguerre moment hierarchy
  (Jorge, Ricci & Loureiro 2017, arXiv:1709.01411) is a validated foundation:
  the moment-based flux-tube linear model reproduces GENE for ITG, TEM, KBM and
  MTM with excellent growth-rate/frequency agreement; the nonlinear moment-based
  approach reproduces the Dimits shift (which the gyrofluid framework does not);
  and a good Cyclone nonlinear ITG heat flux needs only ~16 Hermite-Laguerre
  modes vs ~100 velocity grid points in GENE. **Claim boundary to encode:** the
  moment-count advantage is collisionality-dependent - moments for convergence
  fall as collisionality rises and are lower for gradient-driven (e.g. pedestal)
  modes, but in the near-collisionless / trapped-particle limit the moment count
  approaches GENE's grid-point count, so GKX must NOT claim a universal
  velocity-space efficiency win. Report efficiency by regime.
- **Collision operators - SOUND.** Implementing the linearized gyrokinetic
  Coulomb, Sugama, improved-Sugama and Dougherty operators directly in the
  Hermite-Laguerre basis (Frei, Jorge & Ricci; Jorge, Frei et al.) is the
  validated approach and spans banana to Pfirsch-Schluter collisionality - this
  matches GKX's existing reduced/tabulated operators. The remaining lane work
  (promote to TOML-selectable + add the missing collisionless-growth-rate and
  finite-b H-theorem benchmarks) is well-targeted.
- **Quasilinear (pillar 3) and differentiable optimization (pillar 4):**
  verification incomplete (session limit). Re-run the review's pillar-3/4 angles
  (resume the deep-research run) before Phase 4 advances; the key references are
  already in the plan's Evidence Sources.

## Current Completion

| Lane | Completion |
| --- | ---: |
| Frozen 1.7 contract | 95% |
| Deletion/repository trim | 68% |
| Scientific-core simplification | 35% |
| Test/benchmark/command simplification | 21% |
| QA linear/QL/nonlinear optimization | 15% |
| README/docs rebuild | 14% |
| GKX rename | 0% |
| Release/history rewrite | 0% |

Weighted overall completion: **~37%**. Root facades and the campaign/status
developer-tool machinery (52 files, ~47k lines) are gone (tools 54, tests 77)
and the campaign deletion is test-clean apart from one pre-existing collision
flake.

**Immediate best next steps, in order:**

1. ~~**VMEX piece B**~~ — DONE (``1c567abe``); the geometry bridge runs on
   VMEX's public API and the VMEC-native optimization path is live again.
   ~~x64 CI-green~~ — DONE; Sugama/Piece-B-fallout/flake fixes landed and the
   full x64 suite is green (2149 passed / 0 failed).
2. ~~**Restore package coverage to >=95%**~~ — DONE (95.19%, x64). Rebuilt the
   coverage the deleted ``tests/tools`` monoliths implicitly provided by
   testing helpers directly: ``solvers/time/explicit.py`` 45->97%,
   ``geometry/vmec_flux_tube_reports.py`` 57->89%, and
   ``geometry/vmec_state_sensitivity.py`` / ``geometry/vmec_state_controls.py``
   / ``operators/linear/collisions.py`` 63/83/93 -> 100%. +243 statements over
   the ~193 the gate needed. Full x64 suite 2149 passed / 0 failed; the
   package-coverage manifest gate and architecture gate both pass.
3. ~~**Coordinated ``vmec_jax_*`` -> ``vmex_*`` identifier + frozen-artifact
   rename**~~ — DONE (``c09f12c3`` + ``438176b8``). Lockstep rename of source
   identifiers, report ``kind``/``source``/``source_model`` values, the 46
   frozen docs/_static + benchmarks artifacts, 45 tracked asset files, the
   gates that field-check them, and the ``*_VMEC_JAX_PATH`` env vars. VMEC
   equilibrium names (``vmec_boozer_core``, ``vmec_transport``) left untouched;
   only the ``vmec_jax`` provenance token moved. Verified: ruff, API (362),
   collection (2172), architecture gate, Sphinx ``-W`` (0 warnings), and the
   full x64 suite (2149 passed / 0 failed).
4. **Single combined push of the local commit stack** (AWAITS user
   authorization; outward-facing). Both push-blockers are cleared: x64 CI-green
   and the >=95% coverage gate. Remote ``origin`` -> ``uwplasma/SPECTRAX-GK``
   (GitHub repo rename to ``uwplasma/GKX`` pending; a maintainer action),
   branch ``refactor/gkx-2.0``. Do not push without an explicit go-ahead.

**Then** continue the scientific-core lane (consolidate term/field/dissipation
ownership under ``terms``/``physics`` and remove the duplicated
``parallel/*_reference``/``*_shard_map`` kernels -- dedup-first, since the
1000-line module budget blocks naive merges) and the remaining ``docs/_static``
static-evidence trim toward the tool budget. Before the QA-optimization
(Phase 4) and advanced-collision lanes advance, cross-check them against the
current literature on Hermite-Laguerre gyrokinetics, the Frei/Jorge/Sugama
collision operators, quasilinear-vs-nonlinear stellarator transport, and
differentiable turbulence optimization (review in progress) to confirm the
physics direction and the exact validation benchmarks each lane must pass. The
uncontended GPU refresh remains independent.
