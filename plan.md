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

Date: 2026-07-12.

| Area | Current state | Target | Status |
| --- | ---: | ---: | --- |
| Installable source Python files | 226 | reviewed domain ownership | closed |
| Source modules above 1000 lines | 0 | 0 unreviewed | closed |
| Public/compatibility facade maximum | 472 lines | <=500 lines | closed |
| Tool Python files | 134 | grouped commands; no duplicate owners | active |
| Test Python files | 98 | domain-organized; no duplicate behavior | closed for count, active for structure |
| README lines | 261 | <=350 user-facing lines | closed |
| Tracked files above 2 MB | 0 | 0 | closed |
| Fast release-surface coverage | owner test restored; CI rerun in progress | pass | active pending CI |
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

## SOLVAX Ownership And Integration

SOLVAX owns reusable structured numerical algebra; SPECTRAX-GK owns the
gyrokinetic equations, normalization, state layout, branch selection,
convergence policy, and physical diagnostics. This boundary is intended to
delete generic solver code from SPECTRAX-GK, not add a second abstraction layer.

The integration is pinned to reviewed releases, never an unversioned Git
dependency. SOLVAX 0.6.0 was reviewed in PR 2, passed its minimum/current JAX
matrix plus office GPU compatibility gates, and was published to PyPI on
2026-07-11. The PEP 561 packaging correction was released as 0.6.1, so
SPECTRAX-GK requires ``solvax>=0.6.1,<0.7``. The first
downstream tranche owns only the Hermite-line tridiagonal layout conversion and
the geometry-Jacobian chunk policy; local physics coefficients, diagnostics,
and Krylov policy remain unchanged until their separate gates pass.

| Owner | Retained responsibility | Candidate SOLVAX primitive | Admission gate |
| --- | --- | --- | --- |
| SPECTRAX-GK geometry/objectives | observable definitions, parameter pytrees, conditioning, FD policy | chunked ``jacfwd``/``jacrev`` | dense-JAX identity, JIT/JVP/VJP, peak-memory and wall-time evidence |
| SPECTRAX-GK implicit preconditioner | Hermite-line coefficients, linked-boundary assembly, coarse projection | backend-aware batched tridiagonal solve | CPU/GPU residual, x32/x64, complex64/128, linked/periodic identity |
| SPECTRAX-GK linear/IMEX policy | gyrokinetic operator, tolerances, branch and fallback policy | GMRES, GCROT, implicit linear solve | complex adjoint support, residual history, breakdown, AD/FD, physics parity |
| SPECTRAX-GK nonlinear/optimization policy | physical residual and late-window acceptance | Aitken/Anderson and implicit root solve | real/complex dtype, convergence basin, deterministic replay, tangent/FD gate |
| SPECTRAX-GK continuation | ``ky``/beta/geometry branch identity | recycled GCROT | branch continuity, cold/warm iteration count, no false convergence |

Implementation proceeds in five bounded tranches:

1. **Compatibility contract upstream.** Add supported-minimum and current-JAX
   CI rows, Python 3.10--3.12, import-origin/version checks, complex64/128
   Krylov and adjoint tests, CPU/GPU structured-solve tests, and fix the two
   current JAX 0.9.2 failures. Implement mathematically correct complex Givens,
   conjugate inner products, transpose/Hermitian-adjoint contracts, and real
   relaxation scalars for complex fixed-point states. Keep this work in one
   SOLVAX draft PR until all gates and docs pass.
2. **Low-risk primitives.** After a reviewed SOLVAX release, add a bounded
   dependency and replace the two Hermite-line tridiagonal implementations.
   Use memory-chunked Jacobians in VMEC/Boozer sensitivity and UQ workflows
   where measured peak memory decreases without changing values. One focused
   SPECTRAX-GK algebra adapter may normalize shapes and policies; no fallback
   copy of SOLVAX algorithms is retained.
3. **Complex implicit/Krylov migration.** Replace the three generic GMRES
   routes only after complex and adjoint gates pass. Compare operator action,
   preconditioned residuals, iteration counts, branch identity, gradients,
   compile time, warm throughput, and memory before deleting local generic
   implementations. Retain physics-specific eigenbranch and fallback logic.
4. **Continuation and fixed-point acceleration.** Gate GCROT recycling across
   ordered ``ky``, beta, and geometry scans. Gate Aitken/Anderson on IMEX and
   converged optimization residuals; do not use them for noisy turbulent
   transport windows. Promote only methods that reduce total operator calls
   without changing accepted physics observables.
5. **Delete and document.** Remove superseded local Krylov/direct-solve tests
   and implementation, targeting 500--900 source lines in the first migration
   and a further 200--400 only if Arnoldi/continuation ownership also moves.
   Update API docs, numerics equations, dependency rationale, examples, and
   profiler artifacts in the same commit that changes ownership.

Required examples and evidence are deliberately small in number:

- SOLVAX: complex matrix-free GMRES with implicit gradient, recycled parameter
  continuation, batched complex tridiagonal line solve on CPU/GPU, and a
  memory-chunked Jacobian benchmark;
- SPECTRAX-GK: implicit linear ITG solve, KBM beta continuation, VMEC/Boozer
  sensitivity with bounded Jacobian memory, and an accelerated deterministic
  fixed-point example;
- one publication-quality solver panel comparing residual/iteration history,
  warm runtime, and peak memory. README receives only a concise result after
  all gates pass; equations, algorithms, limitations, and reproduction commands
  live in the numerical and differentiability documentation.

CI acceptance is non-negotiable: ruff, package build, docs with warnings as
errors, >=95% package coverage, supported-minimum/current JAX, CPU x32/x64,
complex64/128, JIT/vmap/JVP/VJP/VJP-transpose, and import-origin/version checks.
GPU gates run on office before any performance claim and record device,
software stack, compilation, warm throughput, memory, residual, and physical
identity. Every local command remains bounded below five minutes; long physics
and accelerator campaigns are explicit external artifacts.

Code contributed to either repository follows the useful parts of SOLVAX's
current style: one numerical responsibility per module, pure array functions,
explicit public exports, typed signatures, mathematical docstrings with shapes
and references, import-safe optional backends, and one focused test family per
module. SPECTRAX-GK does not mirror SOLVAX's file tree or expose solver knobs at
the physics API indiscriminately. User-facing options remain a small set of
physical/numerical policies with stable defaults; expert solver configuration
is grouped under one typed configuration object. Examples are executable,
deterministic, bounded, and build their figures from saved machine-readable
data. CI follows SOLVAX's separate test/docs/publish workflows, extended with
the compatibility matrix and SPECTRAX-GK physics gates above.

## Open Lanes And Progress

| Lane | Completion | Next concrete action |
| --- | ---: | --- |
| Capability/parity specification | 99% | Keep source fingerprints and the machine-readable matrix synchronized; retain ETG as a time-integrated gate until its Krylov branch selector is independently repaired. |
| Tool consolidation | 70% | Fold remaining artifact builders into grouped domain commands; delete stale comparison/probe scripts; update docs command lines. |
| Test consolidation | 100% | Collapse large `tests/tools` families into parametrized contracts with shared fixtures while preserving gate semantics. |
| Source consolidation | 100% | Preserve zero complexity exceptions and the 226-file no-regression baseline while feature lanes evolve. |
| Structured solver ownership | 94% | Develop a residual-convergent restart/preconditioner for shift-invert; the corrected complex Ritz and fail-closed outer-residual contracts now prevent invalid branch promotion. |
| Differentiable API clarity | 93% | The explicit electrostatic species-pmap trajectory has a reverse-mode/finite-difference parameter gate; next add adaptive-controller derivative policy gates and held-out implicit-VJP transport objectives. |
| Advanced collision operators | 37% | Built-in conserving and high-mode hypercollision terms run in the explicit species pmap; next add decomposed invariant artifacts, then species-coupled Dougherty, Sugama, and linearized Coulomb models. |
| Nonlinear GPU performance | 96% | Use the admitted memory/streaming profiles to target bracket kernels; require fresh identity and memory evidence for every optimization. |
| Production parallelization | 71% | The two-species explicit pmap includes conserving/high-mode collision terms, electromagnetic fields, and a reverse-mode parameter gate; obtain an uncontended large GPU integration profile, then design mixed Hermite routing. |
| Performance/release claims | 95% | The refreshed CPU species artifact is identity-exact at 3.41x RHS/0.96x integration; refresh the broader multi-case panel and keep cold executable, warm Python, and parallel scaling claims separate. |
| Docs/readme release pass | 97% | Keep README concise and refresh API ownership text when differentiability/parallel interfaces change. |
| CI/release hygiene | 98% | Verify the corrected fast-coverage owner test on the current CI run; retain the green 95% wide gate. |

## Prioritized Implementation Steps

1. **Freeze the required-core comparison contract.** Record exact equations,
   normalization, geometry arrays, grid layout, initialization, timestepping,
   precision, and diagnostics for each promoted linear/nonlinear comparison.
2. **Close the SOLVAX compatibility contract.** Fix current-JAX transpose and
   mixed-precision failures upstream, add complex Krylov/adjoint/fixed-point
   gates, publish a reviewed release, and migrate tridiagonal plus chunked
   Jacobian paths first. Do not add local compatibility copies.
3. **Correct nonlinear execution and profiling.** Add a prepared nonlinear
   simulation object whose dynamic state/cache/parameter pytrees enter stable
   JIT boundaries, while methods, layouts, and output schemas are explicit
   static policies. Require an identical repeated call to compile the scan once.
   Keep CFL and sampling device-resident, report only active-sharding speedups,
   and separate cold, fixed-overhead, warm-throughput, utilization, and memory.
4. **Benchmark facade shrink.** Keep stable benchmark result contracts in
   `spectraxgk.benchmarks`; move case-policy and manuscript-like benchmark
   drivers to root `benchmarks` or maintainer tools.
5. **Source ownership cleanup.** Keep imported Miller/VMEC geometry in `geometry`, choose
   a single public mathematical-kernel namespace for `terms`/`operators`, and
   consolidate objective helper shards into fewer family modules.
6. **Close required-core physics gates.** Maintain state-level short gates and
   converged long-window gates for axisymmetric/stellarator, electrostatic/
   electromagnetic, adiabatic/kinetic-electron, and restart/spectral diagnostics.
   Treat equilibrium ExB flow shear as the next complete physics extension:
   zero-shear recovery, analytic shearing-wave evolution, remap/phase identity,
   linear mode suppression, nonlinear transport, and matched comparison gates.
7. **Add collision-operator extensibility.** Land a protocol with a complete
   RHS contribution plus an optional mathematically valid split step; do not
   model field-particle terms as diagonal damping. Preserve the current
   conserving Lenard--Bernstein/Dougherty-like result, then add species-coupled
   Dougherty and linearized Sugama/Coulomb operators. Require Maxwellian
   null-space, particle/total-momentum/total-energy conservation, adjointness,
   entropy production, collisional ITG, zonal damping, conductivity, and
   velocity-resolution evidence.
8. **Formalize differentiation.** Use forward JVPs for few design parameters,
   reverse checkpointing for many-parameter scalar objectives, and implicit JVP/
   VJP rules for converged eigen/root solves. Adaptive and turbulent objectives
   require tolerance/window/seed refinement plus AD/finite-difference checks.
9. **Implement production parallelism.** Decompose species first and Hermite
   moments second, exchange Hermite halos explicitly, reduce field moments with
   collectives, and keep perpendicular FFTs local until memory requires pencils.
10. **Tool pruning and test normalization.** Delete unreferenced tools, group
   artifact commands, and use table-driven domain tests with shared fixtures.
11. **Docs and release pass.** Regenerate referenced figures/tables, run fast
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

- 2026-07-12: Migrated the reference-aligned kinetic-electron scan to a
  canonical runtime TOML and deleted its named linear/scan engine, hidden seed
  override, public case dataclass, and implementation-only tests. The effective
  ``1e-3`` electron-density seed and linked-boundary damping are now explicit.
  All parameter leaves are identical; old/new states differ only by a fitted
  global complex phase and the phase-aligned RHS relative error is below
  ``2e-6`` (``6.7e-8`` in x64). This removes 1,977 net lines and reduces the
  benchmark facade from 5,029 to 3,549 lines. Public-export, non-slow benchmark,
  tool-entry, typing, and operator gates pass.
- 2026-07-12: Completed the TEM execution migration and deleted the named TEM
  runner family, request/setup dataclasses, hook tables, public case dataclasses,
  and implementation-only tests. The root benchmark and three maintainer tools
  now load the canonical runtime TOML. This removes 1,654 net lines in the
  deletion tranche and reduces ``spectraxgk.benchmarks`` from 6,306 to 5,029
  lines while preserving the reference loader, normalization contract,
  operator-identity gate, mismatch tables, and publication audit. The full
  non-slow benchmark validation shards and focused tool contracts pass.
- 2026-07-12: Added the canonical runtime-configured TEM stress case and moved
  the root TEM benchmark script onto the unified runtime scan. The audit found
  and fixed a runtime initializer discrepancy: single-mode Gaussian envelopes
  were previously honored only for potential seeding, while density and other
  moments silently used constant parallel profiles. The non-potential path now
  uses the configured Gaussian with the established complex phase, while the
  real target-potential convention is unchanged. The shipped TOML agrees with
  the transitional TEM path in every generated parameter leaf, initial state
  to ``6.8e-21`` absolute L2, and the full assembled complex RHS.
- 2026-07-11: Removed the final local GMRES keyword adapter and now import
  ``solvax.gmres`` directly in the linear and nonlinear implicit owners. The
  upstream generic algebra test was removed while SPECTRAX-GK trajectory and
  forwarding tests remain. The expanded nonlinear shard caught and repaired a
  mixed-line deletion that had dropped fixed-mode and preconditioner diagnostic
  forwarding; strict lint/type checks and all nonlinear unit tests now pass.
- 2026-07-11: Removed the obsolete implicit time-step backend selector after
  both linear and nonlinear paths converged on the admitted SOLVAX FGMRES.
  Tolerance, restart, iteration limit, and physical preconditioning remain
  explicit; the independent shift-invert policy is unchanged. This deletes 84
  net lines across configuration, runtime forwarding, solver APIs, tests, and
  docs. Strict lint/type checks and the complete linear, solver, and runtime
  test shards pass.
- 2026-07-11: Released SOLVAX 0.6.1 with its PEP 561 marker after SPECTRAX-GK's
  package-wide mypy job identified the missing distribution metadata. The
  corrected floor passes strict analysis of all 227 installed source files,
  architecture and differentiability manifests, warning-as-error docs, and
  wheel metadata checks. A separate shift-invert experiment reduced
  preconditioned inner-solve residuals below ``1e-10`` but left outer
  eigenpair residuals above ``0.6`` through Krylov dimension 24; that path was
  rejected rather than changing the selected physical branch.
- 2026-07-11: Merged SOLVAX PR 2 at ``89f95ba``, tagged/published version
  0.6.0 through trusted PyPI publishing, and admitted the bounded dependency
  ``solvax>=0.6.1,<0.7``. SPECTRAX-GK now delegates its two Hermite-line
  tridiagonal paths through one last-axis layout helper and exposes SOLVAX
  forward-Jacobian chunking in the existing geometry derivative-report owner;
  no fallback algorithm or new package was added. Direct fused-reference and
  JVP identity tests pass, chunked/unchunked Jacobians agree, 173 focused
  implicit/geometry tests pass against the installed PyPI wheel, docs build
  with warnings as errors, and the wheel metadata contains the bounded
  dependency. Krylov migration and local-code deletion remain separately gated.
- 2026-07-11: Measured the first downstream performance gates rather than
  inferring them from unit tests. On a CPU SPECTRAX-shaped complex Hermite-line
  batch ``(1,4,8,8,32,24)``, the SOLVAX automatic backend agreed with the
  prior fused solve to ``1.15e-16`` relative norm and reduced warm time from
  9.43 ms to 2.16 ms, while cold compile increased from 137 ms to 172 ms. A
  large hidden-state forward Jacobian produced identical values with peak RSS
  reduced from 1.71 GB to 1.35 GB (21%) at a 4.7% runtime cost when chunked;
  an output-dominated Jacobian showed no memory reduction. Chunking therefore
  remains an explicit memory policy with recorded metadata, not a universal
  speed claim.
- 2026-07-11: Migrated implicit linear and nonlinear IMEX time-step solves to
  one shared SOLVAX FGMRES policy and changed new configuration defaults from
  the implementation-specific ``"batched"`` name to ``"gmres"``. Against the
  pre-migration commit, a five-step nonzero Cyclone implicit trajectory agrees
  to ``2.35e-10`` relative state norm and a three-step nonlinear IMEX trajectory
  agrees to ``8.96e-12``; field histories agree to ``1.07e-10`` and
  ``4.27e-12`` respectively. Shift-invert was deliberately not migrated: its
  inner Hermite-preconditioned solves stagnate near ``5e-4`` relative residual,
  and the replacement changed the selected eigenbranch despite 0.921 vector
  overlap. Restoring the prior shift solver gives exactly identical eigenvalue
  and eigenvector. This remains a preconditioner/branch-continuity lane rather
  than a tolerance relaxation or hidden behavioral change.
- 2026-07-11: Repeated the accepted time-step migration on one office RTX
  A4000 using exact pre/post Git worktrees. The five-step linear implicit case
  reduced cold runtime from 12.44 s to 12.08 s and warm runtime from 6.32 s to
  5.25 s (``0.831x``), while the nonlinear IMEX case reduced cold runtime from
  8.37 s to 7.62 s and warm runtime from 6.02 s to 4.99 s (``0.829x``).
  Trajectory differences match the CPU gates. These small solver-migration
  audits admit the backend change but are not promoted to README/end-to-end
  performance claims.
- 2026-07-11: Completed the first office GPU gate for SOLVAX PR 2 on one
  RTX A4000. Complex matrix-free GMRES at ``n=1024`` converged in eight
  iterations with ``7.24e-11`` relative residual and 17.7 ms warm runtime;
  the fused complex tridiagonal solve at shape ``(256, 2048)`` reached
  ``2.41e-16`` relative residual in 3.63 ms warm; and 64-by-8 block Thomas
  reached ``2.09e-16`` in 2.69 ms warm. A CPU-only backend-selection test was
  correctly guarded on CUDA, after which the GPU tridiagonal family passed
  25 tests with one expected skip. These are compatibility measurements, not
  SPECTRAX-GK speedup claims. Manual test/docs dispatch was added because the
  draft PR initially produced no automatic check runs.
- 2026-07-11: Implemented the first SOLVAX compatibility tranche in draft PR
  2 (commit ``632f14b``). Complex GMRES/GCROT now use scaled unitary Givens
  rotations and Hermitian projections; complex Aitken/Anderson use real
  safeguards and the correct residual Gram matrix; Jacobi factors are explicit
  PyTrees for real mixed-precision execution; and block Thomas again supports
  ``linear_transpose`` on JAX 0.9 through a transform-safe recurrence. Current
  JAX passes 201 tests at 98.54% coverage; the validated minimum stack (JAX
  0.4.38, Equinox 0.11.12, Lineax 0.0.8) passes 200 tests. The complex implicit
  gradient example agrees with central differences to ``5.8e-12``. SPECTRAX-GK
  still does not depend on the source branch: review, office GPU gates, merge,
  and a published SOLVAX release precede the low-risk downstream migration.
- 2026-07-11: Pulled SOLVAX through ``38bb094`` and audited its source, tests,
  documentation, examples, CI, and published-package status against the three
  SPECTRAX-GK GMRES routes, two Hermite-line tridiagonal solves, geometry/UQ
  Jacobians, and deterministic fixed-point policies. The source suite passes
  192 tests on local JAX 0.9.2; block-Thomas linear transpose and mixed-precision
  block Jacobi still fail, PyPI trails source version 0.4.0 at 0.2.0, and
  complex GMRES/Aitken remain unsupported. The plan therefore establishes one
  solver-ownership boundary and stages tridiagonal/chunked-autodiff adoption
  before complex Krylov migration, with explicit deletion, CI, examples,
  documentation, physical-identity, and GPU-profile gates.
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
- 2026-07-10: Migrated the root Cyclone benchmark driver from the specialized
  named-case solver to canonical TOML plus ``run_runtime_scan`` and
  ``run_runtime_linear``. This also corrected the example's transposed
  Laguerre/Hermite resolution. A real single-mode CPU execution completed in
  23.1 seconds and generated both validation figures.
- 2026-07-10: Attempted the same unified-runtime migration for ETG and rejected
  it after real execution. At ``ky=15``, both the direct runtime Krylov policy
  and the retained specialized driver select branches far from the tracked
  reference; fixed ``dt=0.01`` time integration is unstable. ETG therefore
  remains an explicit branch-selection/parity blocker rather than being hidden
  by an API refactor. The retained driver now imports its comparison-only API
  from ``spectraxgk.benchmarks`` and all root benchmark ``--help`` paths pass.
- 2026-07-10: Extended the structural custom-collision contract through the
  linear cached RHS and serial explicit/IMEX integration. The hook replaces
  built-in collisions without disabling hypercollisions, validates output
  shape, and passes a real two-step damping gate. Unsupported implicit and
  decomposed routes now fail explicitly. Numerics documentation records the
  JAX boundary, and non-comparison operator tests use physical terminology.
- 2026-07-10: Routed the same custom-collision contract through serial explicit
  nonlinear state integration. A physical two-step gate matches the analytical
  damping factor exactly. The active extension-point and numerics docs now
  distinguish supported state paths from pending diagnostic, implicit, and
  decomposed routes.
- 2026-07-10: Refreshed prepared nonlinear profiling on local CPU and one
  office A4000. The unmatched JAX stacks produced 4.55-second and 1.46-second
  20-step medians, so no README speedup was changed. Kernel splitting shows the
  nonlinear bracket owns about 70% of GPU RHS time. A 400-step Cyclone spectral
  Laguerre gate passes at 0.087% maximum scalar difference and 1.73x runtime
  ratio. An exact FFT batching experiment was neutral and was reverted rather
  than retaining extra code without measured benefit.
- 2026-07-10: Re-ran the historical prepared-runtime revision and current
  revision back-to-back with the original office JAX 0.6.2 environment; both
  measure about 1.47 seconds rather than the tracked 0.465-second GPU value.
  This rules out a current code regression and identifies uncontrolled node or
  GPU operating state in the old 8.77x claim. The README no longer promotes
  that ratio, and the detailed docs retain it only as historical provenance.
- 2026-07-10: Closed the canonical linear ETG runtime mismatch against a fresh
  office comparison run. Runtime startup had omitted the Boltzmann-ion
  quasineutrality coefficient, both ETG TOMLs selected the wrong constant
  hypercollision policy, and the normalization retained an empirical 0.95
  curvature factor. With the physical coefficient, the ``|k_z|``
  hypercollision route, unit drift scaling, and converged RK4 integration,
  SPECTRAX-GK gives ``(gamma, omega)=(4.0028,-8.7067)`` at ``ky=10`` versus
  ``(4.0036,-8.7262)`` in the fresh comparison, and
  ``(5.7306,-13.7039)`` at ``ky=15`` versus ``(5.7261,-13.7050)``. The ETG
  ``ky=30`` result is ``(3.8807,-26.5449)`` versus
  ``(3.8782,-26.5447)``. The refreshed reference replaces a stale high-ky
  growth row. The ETG root benchmark now uses the unified runtime API and the
  obsolete tuned propagator policy is no longer part of the publication driver.
- 2026-07-10: Deleted the obsolete ``etg_linear_auto.py`` example, which still
  selected the rejected Krylov/IMEX path. User documentation now points to one
  config-backed ETG runtime example and one publication benchmark. The
  architecture docs mark the remaining specialized ETG facade as temporary
  maintainer migration debt instead of presenting duplicate policy as design.
- 2026-07-10: Migrated ETG figure and table generation to the canonical TOML
  runtime, removed the stale two-species replay helper, and deleted the complete
  specialized ETG single-mode/scan implementation plus branch-only tests.
  ``benchmarks.py`` fell from 12512 to 10464 lines, the affected repository
  tranche removed more than 3200 lines net, and ETG now has one validated
  solver/configuration path.
- 2026-07-10: Migrated the remaining ETG comparison diagnostics to runtime
  startup, then removed the orphaned ``ETGBaseCase`` and ``ETGModelConfig``
  presets from source and the public API. New comparison-tool tests pin the
  one-electron/Boltzmann-ion response, unit drift/drive normalization, and
  parallel hypercollision policy.

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
- 2026-07-10: Removed the 4,150-line specialized Cyclone execution stack from
  `spectraxgk.benchmarks`. Artifact generation, operator validation, profiling,
  and parallel identity gates now load the canonical Cyclone TOML and call the
  shared runtime. A real four-mode serial-versus-combined scan agrees to
  `7.8e-16` maximum relative growth error; runtime timing remains diagnostic
  until a large, warm workload demonstrates speedup.
- 2026-07-10: Rebuilt the publication Cyclone spectrum on one office A4000
  after rejecting a startup-contaminated `t=10` artifact. The accepted RK4
  policy uses `t=80` only below `ky=0.15` and `t=40` elsewhere; all eight
  points pass with at most 8.5% growth and 5.2% frequency error. Figure
  generation now consumes the reviewed table instead of owning a duplicate
  simulation path, and visual QA uses open markers and sparse linear ticks so
  overlapping curves remain legible.
- 2026-07-10: Audited the remaining 6,306-line benchmark facade rather than
  deleting another family blindly. KBM requires a generic parameter-scan
  abstraction because its tracked scan varies beta rather than `ky`; TEM and
  kinetic-electron cases still lack canonical runtime TOMLs. These are now
  explicit migration prerequisites. The code-structure guide was reduced from
  an implementation diary to a concise ownership contract, and stale retired
  ETG presets were removed from the differentiable manifest.
- 2026-07-10: Fixed the documented nonlinear-sharding profiler entry point and
  rejected its zero-bracket initial condition. A fresh `(4,8,32,32,64)`
  interacting-multimode run on two A4000 GPUs shows that both whole-state `ky`
  and `kx` placement are slower than serial and fail physical trajectory/RHS
  identity. The profiler now refuses zero nonlinear activity. This closes the
  false-positive diagnostic route; production work must implement explicit
  communication-complete FFT/bracket decomposition rather than relaxing gates.

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
- 2026-07-12: Completed the benchmark-engine deletion tranche. Fresh KBM
  comparison and eigenfunction-overlay execution now use the canonical runtime
  trajectory, alternate mode fits reuse that trajectory, and historical cache
  reading is confined to the comparison tool. Replaced the 3,549-line
  ``spectraxgk.benchmarks`` solver stack with a 22-line reviewed-reference
  facade and deleted 2,033 lines of branch-mocking tests for the retired
  duplicate engine. Imported VMEC/DESC geometry coverage now exercises the
  promoted runtime path. Coverage/refactor manifests and user documentation
  name ``benchmarking.shared``, runtime workflows, and diagnostics as the
  actual owners.
- 2026-07-12: Closed the runtime-facade line target without adding another
  module. Removed duplicate TOML case wrappers from ``runtime.py`` and routed
  the stable package API directly to ``workflows.cases``; promoted examples now
  import those workflows from ``spectraxgk``. The facade fell from 535 to 473
  lines, its architecture exception was removed, and five executable case
  workflow tests plus public lazy-import identity passed.
- 2026-07-12: Closed the public-API architecture target without removing any
  promoted capability. ``spectraxgk.api`` now derives its ordered ``__all__``
  from the single lazy target mapping, reducing the facade from 789 to 410
  lines while preserving all 379 export names, order, and target modules.
  Added an invariant against duplicate or divergent exports, removed the API
  architecture exception, and passed lazy-import, docs warnings-as-errors, and
  wheel/sdist builds.
- 2026-07-12: Closed the VMEC/Boozer line-search complexity exception at the
  1,000-line domain budget. Holdout-improvement validation now belongs to the
  immutable holdout configuration itself, so direct and factory construction
  enforce the same non-negative threshold. Focused scalar/aggregate holdout
  tests and the architecture checker pass.
- 2026-07-12: Closed the nonlinear transport-optimization report complexity
  exception without adding modules. Removed two stale intermediate export
  declarations left by earlier assembly, retained one authoritative public
  contract, and passed all 13 production guard, replicated-window, reduction,
  uncertainty, and scope tests. The module now meets the 1,000-line budget.
- 2026-07-12: Closed the transport-window complexity exception by moving the
  complete seed/timestep ensemble-readiness manifest contract into the existing
  ``diagnostics.nonlinear_replicates`` owner. ``transport_windows.py`` fell
  from 1,237 to 938 lines and the replicate owner remains bounded at 742 lines;
  no source file was added. Public API identities, 36 window/ensemble/tool/
  release tests, all architecture/coverage/refactor manifests, Sphinx ``-W``,
  and wheel/sdist builds pass.
- 2026-07-12: Closed the validation-gates complexity exception with no new
  modules. Moved physical metric contracts/extraction into
  ``diagnostics.analysis``, eigenfunction contracts into ``diagnostics.modes``,
  generic time-window helpers into ``diagnostics.growth_windows``, diagnostic
  NetCDF series loading into ``artifacts.nonlinear_diagnostics``, and real-FFT
  grid inference into ``artifacts.spectral_layout``. ``validation_gates.py``
  fell from 1,418 to 540 lines; every destination remains below 700 lines.
  The complete 175-test benchmark/physics/analysis/artifact suite, architecture
  and refactor manifests, Sphinx ``-W``, and wheel/sdist builds pass.
- 2026-07-12: Closed the imported-Miller complexity exception and removed the
  obsolete 284-line ``geometry.miller`` stub, which intentionally wrote an
  incomplete NetCDF file and duplicated production kernels. Generic JAX/NumPy
  differences, quadrature, period extension, reflection, and safe division now
  live in ``geometry.kernels``; analytic Miller parameters/collocation live in
  ``geometry.analytic``; the production EIK backend remains in
  ``geometry.imported_miller`` at 956 lines. Source count fell to 226 and 68
  Miller/analytic/runtime geometry tests plus all architecture manifests pass.
- 2026-07-12: Closed the imported-VMEC complexity exception without adding a
  source file. Optional-backend discovery, Boozer field-line sampling,
  Hegna--Nakajima/metric derivatives, and VMEC state construction now live in
  their existing focused owners; ``imported_vmec.py`` fell from 2,540 to 699
  lines. The broader geometry/objective suite passed 263 tests, package-wide
  mypy passed, and an intentionally dynamic optional-backend record now has an
  explicit typed boundary.
- 2026-07-12: Closed the final package complexity exception. Nonlinear-gradient
  artifact conditioning now belongs to diagnostic metadata, replicated window
  evidence to replicate diagnostics, and central finite-difference transport
  response to transport diagnostics; the report facade fell from 2,339 to 956
  lines. The architecture manifest now has zero exceptions, all 226 source
  files satisfy the domain budget, 118 nonlinear validation tests and 91
  release tests pass, warning-strict docs build, and wheel/sdist checks pass.
  CI's prior sole failure was a stale fast-coverage shard that omitted the
  benchmark metric-owner test; the owner test is restored without weakening
  the package-wide 95% gate.
- 2026-07-12: Added an explicit geometry Jacobian direction policy without
  growing the module past its architecture budget. ``jacobian_mode="auto"``
  selects forward mode for few controls and reverse mode for few observables;
  SOLVAX chunking remains an explicit forward-mode memory policy. Reports
  record the resolved mode, explicit forward/reverse Jacobians agree, invalid
  mode/chunk combinations fail early, and 60 differentiable-geometry tests,
  package-wide mypy, and zero-exception architecture gates pass.
- 2026-07-12: Extended the prepared explicit nonlinear runner from a
  state-only boundary to fixed-step matched dynamic ``(geometry, cache,
  params)`` PyTrees. Physical changes must rebuild and pass the corresponding
  cache, preventing stale gyroaverage/drift/collision arrays; partial and
  adaptive model overrides fail closed. A resolved nonzonal-mode curvature
  objective differentiates through geometry data, cache construction, and
  three RK2 steps and agrees with centered differences. Default prepared
  adaptive execution remains on its static CFL path. Five explicitly collected
  prepared-run tests pass in 20 seconds; the repository integration marker is
  overridden for this evidence rather than silently deselecting the file.
- 2026-07-12: Routed nonlinear IMEX time-step solves through SOLVAX's
  ``linear_solve`` implicit-differentiation primitive. Reverse mode now
  differentiates the converged matrix-free system with one transposed solve
  instead of tracing tolerance-driven GMRES loops, which JAX rejects. Physical
  two-step state objectives agree with centered finite differences both with
  and without scan checkpointing. Explicit geometry VJPs likewise pass in both
  checkpoint policies, while adaptive model-parameter derivatives remain
  deliberately unpromoted.
- 2026-07-12: Extended the nonlinear IMEX gate from state-only sensitivity to
  a parameter-dependent matrix-free operator. The test rebuilds the complete
  cache and implicit operator from traced ``R/L_Ti``; the resulting VJP
  includes operator dependence and agrees with a centered physical-parameter
  finite difference to the 2% gate. This closes the tiny-case implicit model
  derivative contract without claiming adaptive or long-window transport
  derivatives.
- 2026-07-12: Replaced the stale 20-step prepared-runtime profiles with a
  controlled 200-step adaptive Cyclone run on one office node and one software
  stack. Warm CPU/A4000 times are 108.864/9.557 seconds (11.39x), while final
  state, timestep, potential, and heat-flux fingerprints agree within 3.8e-6.
  The profiler now emits those fingerprints and the artifact test enforces
  revision, software, configuration, finite outputs, numerical identity, and a
  conservative 5x throughput floor. This is a one-device warm-throughput
  result, not an executable or parallel-scaling claim.
- 2026-07-12: Added machine-readable host peak RSS and JAX allocator metrics,
  then profiled compact and resolved diagnostics on the same 200-step CPU/GPU
  trajectory. Resolved histories preserve exact recorded output identity while
  adding 0.62% CPU time/1.18% RSS and 2.36% A4000 time/2.78% device peak. The
  four small JSON artifacts are now regression-gated at 25% runtime and 10%
  memory overhead; opaque profiler traces remain untracked.
- 2026-07-12: Corrected complex Arnoldi Ritz reconstruction from
  ``V @ conj(y)`` to ``V @ y`` in both shared and shift-invert restart paths,
  and excluded zero Ritz values caused by Arnoldi breakdown from inverse
  spectral mapping. Added a matrix-free outer residual gate for primary and
  fallback pairs. The reviewed reduced KBM shift policy now fails honestly at
  residual 0.99978 versus tolerance 0.1; a tighter 24-vector/three-restart
  probe exceeded the five-minute cap and was terminated. Shift-invert remains
  unpromoted, while validated time integration is unchanged.
- 2026-07-12: Implemented the first production species-decomposed linear
  route. Two-species electrostatic quasineutrality now reduces density and
  polarization moments with ``shard_map``/``psum``; the complete local
  streaming, mirror, curvature, grad-B, and diamagnetic RHS then executes on
  one species per device without reconstructing the full distribution. The
  two-logical-CPU route agrees with the serial production field solve and RHS,
  while mixed species--Hermite, collisions, electromagnetic terms, and speedup
  remain fail-closed pending their own identity and profiler evidence.
- 2026-07-12: Closed the species-route fixed-step identity gate on two office
  A4000s. A one-time host staging boundary avoids a JAX 0.6.2 device-to-device
  resharding defect, and a compiled host-controlled step loop avoids corruption
  of the electron carry under nested ``lax.scan``/``shard_map``. Three Euler
  steps agree with serial state/field histories to ``4.61e-8``/``1.59e-9``
  relative. The medium profile remains overhead-limited, but the clean
  ``2x8x32x128x1x128`` run passes at ``5.26e-8`` relative and improves warm RHS
  time from 8.21 to 7.11 ms (``1.16x``). This is a scoped workload crossover,
  not a broad strong-scaling claim.
- 2026-07-12: Replaced the correct but launch-limited species host loop with a
  single enclosing ``pmap``. Combining ``shard_map`` collectives with either
  ``scan`` or ``fori_loop`` corrupted the electron shard on office JAX 0.6.2;
  the mature named-collective ``pmap`` path passes one-, three-, and nine-step
  state/field identity, RK2, and sampled-history gates. Field sampling now
  performs quasineutrality only instead of assembling a redundant RHS. The
  clean logical-CPU 100-step artifact is identity-exact and records ``0.91x``;
  the available GPU integration timings were externally contended and are not
  used for a speedup claim.
- 2026-07-12: Removed a redundant full-RHS assembly from serial linear
  diagnostics. Fixed-step integration now calls the dedicated field solve after
  advancing state; the matched 100-step CPU profile preserves exact state and
  field histories while reducing serial wall time by about four percent. The
  species ``pmap`` already used the same field-only sampling policy.
- 2026-07-12: Extended the enclosing species ``pmap`` through the complete
  built-in conserving collision contribution. A collision-only gate uses
  unequal nonzero ion/electron rates, verifies a nonzero operator, and matches
  serial three-step evolution on logical CPUs and two office GPUs. The direct
  ``shard_map`` helper remains collision-free because JAX 0.6.2 rejects the
  varying-axis conditional branch; electromagnetic and IMEX paths still fail
  closed.
- 2026-07-12: Added a distinct species-parallel hypercollision gate with
  populated ``l=1,m=5`` moments and nonzero Hermite/Laguerre rates. The
  operator is demonstrably nonzero and three-step evolution matches serial on
  logical CPUs and both office GPUs, extending the decomposed dissipation
  contract beyond the low-order conserving correction.
- 2026-07-12: Preserved reverse-mode differentiation through the enclosing
  species ``pmap`` by keeping traced parameters out of the one-time host
  staging boundary. A two-species explicit electrostatic trajectory now
  differentiates a fixed ion-mode projection with respect to
  ``R_over_LTi``; the tangent agrees with centered finite differences to one
  percent in float32. Adaptive, IMEX, and electromagnetic derivatives remain
  explicitly outside this gate.
- 2026-07-12: Extended the same species ``pmap`` through the production
  electromagnetic field equations without duplicating their physics. Serial
  species sums are now optionally followed by named ``psum`` reductions for
  density, polarization, parallel current, and perpendicular pressure. A
  two-species trajectory with nonzero ``apar`` and ``bpar`` matches serial
  integration locally; the exact pushed commit is validated separately on the
  two-A4000 office stack. The broader electromagnetic derivative and mixed
  species--Hermite lanes remain fail-closed.
- 2026-07-12: Removed integration-only parallel overhead by returning the
  replicated field history once and applying ``sample_stride`` inside the
  mapped scan instead of computing and storing diagnostics on every step. The
  exact ``2x4x16x64x1x64`` logical-CPU artifact remains state/field exact,
  improves isolated RHS throughput to ``3.41x``, and moves 100-step
  end-to-end throughput from ``0.91x`` to ``0.96x``. Since the latter remains
  below one, no integration speedup is claimed. A larger office rerun was
  externally contended and is retained only as transient debugging evidence.
