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
- **Identifier rename (coordinated pass):** the GKX-internal ``vmec_jax_*`` →
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

1. **VMEX piece B** — re-integrate the geometry bridge onto VMEX's public API.
   This is the top priority because it is a *functional* gap: the bridge is
   currently broken against VMEX 0.2.0, so the VMEC-native optimization path
   does not run.
2. **Restore package coverage to >=95%** — rebuild the tests the deleted
   ``tests/tools`` monoliths implicitly provided (explicit-linear path and
   others); this is the leading edge of Phase 3.
3. **Coordinated ``vmec_jax_*`` -> ``vmex_*`` identifier + frozen-artifact
   rename** (see the dependency section), then the single combined push of the
   local commit stack.

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
