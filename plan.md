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

Date: 2026-07-14.

| Area | Current state | Target | Status |
| --- | ---: | ---: | --- |
| Installable source Python files | 223 | reviewed domain ownership | closed |
| Source modules above 1000 lines | 0 | 0 unreviewed | closed |
| Public/compatibility facade maximum | 472 lines | <=500 lines | closed |
| Tool Python files | 95 | <=95 grouped commands; no duplicate owners | closed |
| Test Python files | 95 | domain-organized; no duplicate behavior | closed for count, active for structure |
| README lines | 261 | <=350 user-facing lines | closed |
| Tracked files above 2 MB | 0 | 0 | closed |
| Fast release-surface coverage | compile-heavy nonlinear owner retained in bounded node batches; exact x64 coverage passes locally | pass | active pending CI |
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
2026-07-11. The PEP 561 packaging correction was released as 0.6.1. The
published 0.7.3 release subsequently passed 227 downstream structured-solver,
IMEX, geometry-gradient, and implicit-objective checks on the current JAX
stack, so SPECTRAX-GK now requires ``solvax>=0.7.3,<0.8``. The first
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
| Capability/parity specification | 100% | Keep source fingerprints and the machine-readable matrix synchronized; retain ETG as a time-integrated gate until its Krylov branch selector is independently repaired. Flow shear is explicitly unpromoted after its fixed-step response gate failed. |
| Tool consolidation | 100% | Runtime comparisons, imported-linear fields/growth/windows, term-resolved RHS and nonlinear comparison workflows, VMEC state mapping and admission, holdout selection, nonlinear-gradient evidence, transport admission and window statistics, geometry generation, validation artifacts, repository hygiene, validation traceability, architecture/refactor policy, quasilinear calibration/promotion policy, and performance/scaling release checks have one current owner per domain. The existing linear-validation owner now includes the factorized multiprecision Coulomb/Sugama generators, finite-wavelength gyro-diffusion and coupled assembled-block convergence gates, and paper-facing verification artifacts; no tool was added. Four tools tied to VMEC-JAX's removed private optimizer generation were deleted, and the enforced 95-tool target is met at 93,130 lines. |
| Test consolidation | 100% | The enforced 95-file topology target and 95% package-wide coverage gate are met; the suite has 94,464 lines after adding ordered-pair interpolation, full-RHS tabulated-collision, and independent finite-Larmor kernel/projection/reconstruction, Coulomb speed-integral/moment, generalized-Laguerre, velocity-projection, finite-m parity-block/analytic-inverse, direct overlap-factorization and polynomial-projection/oracle identity, Laguerre-product, gyroaveraged-spherical-moment, inverse-shell, high-condition, runtime pair-loop, full-RHS, JIT/JVP, Bessel/spherical/radial matrix-truncation, component-resolved gyro-diffusion, driven-response saturation/AD, arbitrary-order original/improved-Sugama, exact cache-reuse, fast tracked-artifact, and nightly regeneration gates. Current-API campaign contracts replaced compatibility tests, status-dashboard metrics moved to mapping-subset assertions, installed-wheel/nonlinear-gate/Miller-cache regressions were added, and repeated runtime, geometry, objective, diagnostic, and artifact setup moved to shared fixtures. Large objective, parallel, artifact, and comparison owners retain domain-local scientific cases and one import boundary. An AST/fan-in audit found that the remaining repeated two-line tool loaders intentionally reload monkeypatch-heavy scripts per test; centralizing them would leak state or add indirection without reducing setup. The planned test-normalization lane is closed: future reductions must remove a demonstrated duplicate contract, never a physics, convergence, failure-mode, or coverage gate. |
| Source consolidation | 100% | The zero-exception architecture gate passes at 219 installable modules after deleting redundant facades, merging nonlinear Laguerre transforms into their velocity-basis owner, reducing duplicated solver/cache/workflow/gradient/nonlinear-report policy, and removing private compatibility exports and obsolete optimizer adapters. The package has 87,559 lines after adding the differentiable finite-wavelength Coulomb runtime and driven collision-response solve. One deliberate domain owner, ``operators/linear/collisions.py``, keeps collision equations, interpolation, application, and response within the 1,000-line ceiling and separate from the 218-line generic cache-array module; no compatibility facade or duplicate implementation was retained, and no module reaches the 1,000-line ceiling. Single-consumer QA, stellarator residual, growth-series, and VMEC boundary-chain policy splits remain consolidated. Reusable contracts, fit/window algorithms, tables, tensor mappings, collision equations, and solved-state sensitivity retain separate owners because they have independent consumers or numerical responsibilities. Future merges require fan-in evidence and must not trade domain clarity for file-count reduction. |
| Structured solver ownership | 97% | Dtype-aware Arnoldi breakdown and true shifted-system residual retries close false convergence. Retained-Ritz, one- and two-sided Jacobi--Davidson, ordered-Schur thick restart, alternative seeds, nested field solves, and explicit field blocks with diagonal/Hermite complements were physically rejected. KBM remains on validated time integration; future acceleration requires a different branch-preserving interior spectral transformation. |
| Differentiable API clarity | 100% | Fixed-step pmap reverse mode, adaptive forward/checkpointed-reverse derivatives, and a physical IMEX endpoint heat-flux implicit VJP pass finite-difference gates; converged noisy transport optimization remains a separate science claim. |
| Advanced collision operators | 98% | The shipped model has independent drift-kinetic and finite-b equation, dissipation, asymptotic, and AD gates. Published drift-kinetic original/improved-Sugama and Coulomb low-order matrices pass exact coefficient, null-space, symmetry, dissipation, invariant, and derivative gates. Ordered pairs support unequal mass/temperature species, conserve physical multispecies invariants in the drift-kinetic limit, and approach the original-Sugama collision null space in a time-domain relaxation gate. The improved correction passes its independent equal-species endpoint, matrix-wide equal-temperature dissipation, and heat-flow proximity-to-Coulomb gates. An 80-digit generator, checksummed package table, and device-side finite-b interpolation support shared, per-species, and ordered target/source pair tables with target-species spatial Bessel arguments. A JAX-pytree tabulated operator derives :math:`B=\sqrt{2b_\mathrm{cache}}` from the real cache and is full-RHS identical to the reduced Sugama protocol for a constant pair table; JIT/JVP/application and zero-wavelength multispecies invariant gates pass. The exact Bessel--Laguerre kernel, arbitrary-order associated expansion coefficients, Appendix-A Coulomb speed integrals and integrated test/field moments, generalized-Laguerre monomial and product coefficients, isotropic/finite-m Hermite--Legendre transforms, and the finite-b gyro-moment-to-spherical-moment map from the primary papers pass independent projection, reconstruction, improper-quadrature, truncation, high-condition inverse, JIT, and derivative gates. The complete offline Coulomb blocks have a machine-readable publication artifact covering coupled finite-b Bessel/spherical/radial convergence, direct velocity-space projection, published coefficients, symmetry, drift-kinetic invariant nulls, the H-theorem, and separate test/field/combined classical gyro-diffusion with leading :math:`O(b^2)` scaling. The runtime research API bilinearly resolves independent target/source Bessel-argument axes, contracts test/field blocks with gyrocenter moments, and couples all four polarization vectors to the solved field with JIT, JVP, pair-loop, cancellation, generated-coefficient, and full-RHS gates. The generator caches combinatorial transforms and factorizes output/input contractions exactly. Coulomb, original-Sugama, and improved-Sugama conductivity hierarchies are converged through P=20, J=5 and K=5; dimensional normalization, a 100x electric-field linearity scan, stationary saturation, and the high-charge Spitzer asymptote pass fail-closed artifact gates. The exact paper-wavelength :math:`B=1/\sqrt{2}` hierarchy reaches :math:`(P,J)=(12,5)` and passes its prospective 5% common-block gate. Finite-b collisional ITG still requires the published :math:`(18,6)` endpoint or equivalent demonstrated growth convergence, and collisional-zonal response remains required before input-file promotion. Local particle-coordinate conservation is not reconstructible from the already gyroaveraged matrix and is required only for a future pre-gyroaverage particle operator. |
| Nonlinear GPU performance | 97% | The bracket has one numerical owner and a clean A4000 profile; an identity-breaking FFT-layout rewrite was rejected. Require fresh identity and memory evidence for every future optimization. |
| Production parallelization | 98% | Periodic and linked 2x2 species-Hermite routes cover the complete electrostatic operator; four-device GPU evidence and mixed electromagnetic integration remain hardware/future scope. |
| Performance/release claims | 100% | Release checks and scoped CPU/GPU artifacts pass; the mixed operator records 3.11x RHS but 0.97x integration, and two-GPU nonlinear sharding records 0.586x, so no unsupported end-to-end or nonlinear multi-GPU speedup is claimed. |
| Docs/readme release pass | 100% | Keep README concise and refresh API ownership text when differentiability/parallel interfaces change. |
| CI/release hygiene | 99% | The exact MyPy command passes and installed-wheel smoke coverage now guards both executable names, plotting help, and lazy parallel imports. Confirm the latest queued CI run while preserving the bounded 95% package gate. |

## Prioritized Implementation Steps

1. **Freeze the required-core comparison contract (closed 2026-07-13).** Record exact equations,
   normalization, geometry arrays, grid layout, initialization, timestepping,
   precision, and diagnostics for each promoted linear/nonlinear comparison.
2. **Close the SOLVAX compatibility contract (closed for 0.7.3).** Fix current-JAX transpose and
   mixed-precision failures upstream, add complex Krylov/adjoint/fixed-point
   gates, publish a reviewed release, and migrate tridiagonal plus chunked
   Jacobian paths first. Do not add local compatibility copies.
3. **Correct nonlinear execution and profiling (closed for serial prepared execution).** Add a prepared nonlinear
   simulation object whose dynamic state/cache/parameter pytrees enter stable
   JIT boundaries, while methods, layouts, and output schemas are explicit
   static policies. Require an identical repeated call to compile the scan once.
   Keep CFL and sampling device-resident, report only active-sharding speedups,
   and separate cold, fixed-overhead, warm-throughput, utilization, and memory.
4. **Benchmark facade shrink (closed).** Keep stable benchmark result contracts in
   `spectraxgk.benchmarks`; move case-policy and manuscript-like benchmark
   drivers to root `benchmarks` or maintainer tools.
5. **Source ownership cleanup (closed 2026-07-14).** Imported Miller/VMEC
   geometry lives in `geometry`; `terms` owns physical contribution contracts,
   while `operators` owns their numerical application. Single-consumer
   objective and diagnostic policy shards were consolidated without flattening
   independently reusable contracts, tables, or numerical algorithms.
6. **Close required-core physics gates.** Maintain state-level short gates and
   converged long-window gates for axisymmetric/stellarator, electrostatic/
   electromagnetic, adiabatic/kinetic-electron, and restart/spectral diagnostics.
   Equilibrium ExB flow shear completed zero-shear recovery, analytic shearing-
   wave evolution, remap/phase identity, linear suppression, nonlinear
   transport, and matched comparison audits. Its final fixed-step physical-
   response gate failed, so retain only the validated Python research API and
   keep input-file/executable exposure disabled.
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

## Collision Verification And Promotion Plan

Collision validation follows the physical hierarchy in [Abel et al.
(2008)](https://arxiv.org/abs/0808.1300), the numerical tests in [Barnes et al.
(2009)](https://arxiv.org/abs/0809.3945), and the Hermite--Laguerre Coulomb
implementation and figures in [Frei et al.
(2021)](https://arxiv.org/abs/2104.11480) and [Frei, Ernst & Ricci
(2022)](https://arxiv.org/abs/2202.06293). A lower-level pass never substitutes
for a higher-level physical result.

1. **Operator algebra and MMS/projection (closed).** Generate equations
   (3.41) and (3.48)--(3.50) at arbitrary precision. Require independent
   Gauss--Hermite/Laguerre velocity projection, reconstruction of manufactured
   moment states, published low-order coefficients, Maxwellian/density nulls,
   total momentum and energy invariants, adjointness, and non-positive entropy
   production. The tracked JSON/PNG artifact passes all six numerical gates.
2. **Finite-truncation convergence (drift-kinetic response closed; finite-b hierarchy open).** Track
   Bessel--Laguerre convergence at finite :math:`k_\perp\rho`, then repeat the
   assembled operator over increasing Hermite, Laguerre, spherical, and Bessel
   cutoffs. Require stable low-order matrix entries, drift-kinetic invariant
   residuals, equation-(3.5) finite-wavelength gyro-diffusion, and entropy
   spectrum.
   Runtime tables are admitted only where interpolation and truncation errors
   are separately below the stated tolerance. The direct drift-kinetic Coulomb
   hierarchy now reaches :math:`(P,J)=(20,5)` and changes its current by at
   most :math:`1.66\times10^{-4}` from :math:`(15,5)` over the tracked ion-
   charge scan. This closes the normalized Coulomb response, not the finite-
   :math:`b` hierarchy or absolute transport normalization. At
   :math:`k_\perp\rho=0.5`, the exact :math:`(7,3)\rightarrow(9,4)` and
   :math:`(9,4)\rightarrow(12,5)` common blocks now pass a prospective 5%
   intermediate matrix gate, with much smaller polarization changes. This
   establishes a controlled path to :math:`(18,6)` but does not replace it.
3. **Multispecies runtime assembly.** Couple the four polarization vectors
   through quasineutrality without collapsing target/source
   :math:`q_s\phi/T_s` factors. Do not impose local density, momentum, or energy
   nulls on finite-:math:`b` gyrocenter moments, which include classical
   gyro-diffusion. Require equal-species polarization cancellation, ordered-pair
   drift-kinetic invariants, JIT/JVP/VJP, CPU/GPU identity, and direct agreement
   between generated and runtime blocks. Test equations (3.2)--(3.4) at fixed
   particle position only if a pre-gyroaverage particle operator is implemented;
   equation (3.35) is not an inverse of the collision gyroaverage.
4. **Strongly collisional transport (Spitzer conductivity closed;
   remaining coefficients open).** Retain the converged Spitzer--Härm
   normalization, field-linearity, and saturation gates. Reproduce the
   remaining Braginskii/Chapman--Enskog friction, heat-flow, and viscosity
   coefficients as resolution scans before making a broad fluid-closure claim.
5. **Weakly collisional kinetic limit.** Scan collision frequency toward zero
   and require continuous recovery of collisionless growth/frequency while
   resolving the Hermite--Laguerre tail. Track the spectral cutoff and entropy
   dissipation rather than adding artificial damping. Reproduce the exponential
   velocity-resolution trend and its deterioration at low collisionality shown
   by Frei et al., Figure 9.
6. **Integrated ITG gate.** Reproduce the numerical slab/toroidal
   :math:`k_\perp` scans, collision-model comparison, and short-wavelength
   branch in sections 5--6 of Frei, Hoffmann & Ricci (2022), followed by the
   independent :math:`(P,J)` scans in its convergence section. Figures 4--5
   are analytical peak estimates and are not acceptance references. Require
   the paper's :math:`k_\perp=0.5`, :math:`k_\parallel=0.1`,
   :math:`\eta=3`, :math:`R_B=0.1` collisionality scan and at least its
   converged :math:`(P,J)=(18,6)` endpoint. Then reproduce the fixed
   :math:`P=18` perpendicular-wavenumber/J scan, the weak-collision
   :math:`J=10` P-scan and :math:`P=32` J-scan, and the magnetic-gradient scan.
   Compare drift-kinetic and finite-:math:`b` Coulomb under identical geometry,
   normalization, and branch tracking. Add original/improved Sugama only when
   their finite-:math:`b` arbitrary-order tables have independently converged.
7. **Zonal-response gates.** Keep the collisionless Rosenbluth--Hinton residual
   separate from Hinton--Rosenbluth collisional damping. Reproduce the
   ion--ion Pfirsch--Schluter campaign of Frei, Ernst & Ricci (2022), Figures
   12--14: :math:`q=1.4`, :math:`\epsilon=0.1`, :math:`\nu_i^*=3.13`, and the
   converged :math:`(P,J)=(24,10)` hierarchy. Gate the drift-kinetic
   :math:`k_x=0.05` trace, gyrokinetic :math:`k_x=0.1,0.2` traces, Coulomb / OS
   / IS ordering, the Xiao long-time estimate, and the :math:`t\nu=5`
   parallel/perpendicular velocity-space sections. Admit a claim only after
   damping-window and velocity-resolution convergence.
8. **Publication and release.** Put the algebra/convergence panel in README and
   all operator, transport, ITG, and zonal panels in the collision docs. Promote
   the full operator to input files only after steps 3--7 pass; until then it
   remains an explicitly scoped Python research boundary.

9. **Nonlinear full-distribution extension (post-release research).** Do not
   materialize the dense Landau collision tensor. Follow Jorge et al. (2026),
   arXiv:2606.31035: approximate the Boys kernel by a convergence-controlled
   sum of exponentials, apply one-centre Coulomb/Talmi factors as separable
   matrix-free contractions, and rotate to the Hermite--Laguerre basis. Admit
   only after dense low-order overlap, quadrature refinement, Maxwellian,
   conservation, entropy, isotropic/two-species relaxation, JVP/VJP, and memory-
   scaling gates. Contribute only generic separable Kronecker operators and
   constrained-solve primitives to SOLVAX; retain collision physics here.

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
| Equilibrium ExB flow shear | coordinate/cache/split-phase and canonical compressed brackets, periodic/linked RK2/RK3 trajectory, fixed-step sheared IMEX, canonical heat-flux trace, linear suppression, and transport-objective AD validated; the final fixed-step response audit failed promotion | retain as a Python research API; do not expose an input-file option unless a new, prospectively gated physical campaign overturns the negative fixed-step evidence |
| Species/Hermite multi-device execution | periodic and linked 2x2 species/Hermite routes cover the complete electrostatic operator and pass identity gates; mixed electromagnetic and four-device evidence are not available | retain the scoped electrostatic production route; require new hardware and identity evidence before broadening the claim |
| Linearized Landau/Sugama collisions | reduced published original/improved-Sugama and Coulomb matrices, unequal-species pair assembly, relaxation, invariants, dissipation, and AD gates pass through the collision protocol; the complete finite-``b`` multispecies hierarchy is not implemented | retain the Python research boundary and keep TOML promotion fail-closed until arbitrary generated couplings plus conductivity, ITG, zonal, and resolution gates pass |
| Long-wavelength reduced field solve and Beer/Smith closures | missing | optional, only with a scientific owner |
| KREHM, Vlasov--Poisson, collisional-ETG, forcing, Trinity coupling | not complete equations in SPECTRAX-GK | keep out of scope; remove orphan compatibility fragments |
| JAX autodiff, implicit gradients, UQ, in-memory VMEC/Boozer optimization | SPECTRAX-GK extensions | retain and strengthen conditioning/FD/performance gates |

## Recent Implementation Log

- 2026-07-16: Added the first runtime-level finite-wavelength Coulomb slab-ITG
  convergence artifact at the exact paper wavelength and protocol. The
  dominant mode is collisionality-stabilized across :math:`(P,J)=(7,3)`,
  :math:`(9,4)`, and :math:`(12,5)`. The final nested growth change is below
  0.12% over the unstable :math:`\nu\geq0.03` interval, but reaches 7.44% in
  the collisionless limit. A separate runtime hierarchy through the published
  :math:`(18,6)` endpoint changes by 0.59% from :math:`(15,6)`, showing that
  the endpoint itself is converged despite the non-monotone P9/P12 sequence.
  The publication panel shades the still-unresolved finite-collision interval
  and reports ``OPEN``; tests require the intermediate and collisionless
  passes alongside the finite-collision failure, preventing selective
  promotion. The next collision action is a parallelized exact higher-moment
  table build, followed by the independent collisional-zonal gate.

- 2026-07-16: Generated the exact finite-wavelength Coulomb hierarchy at the
  paper's :math:`k_\perp=0.5`, :math:`\tau=1` point
  (:math:`B=1/\sqrt{2}`). The :math:`(7,3)`, :math:`(9,4)`, and
  :math:`(12,5)` builds complete in 43.48, 145.71, and 557.87 seconds. The
  common test/field matrix change falls from 5.87%/2.35% to 4.38%/1.92%; the
  latter passes the prospective 5% intermediate gate, while polarization
  changes are below :math:`2.2\times10^{-9}`. The tracked artifact now records
  the actual paper wavelength but keeps literature acceptance false because
  the published :math:`(18,6)` endpoint and its growth-rate scan remain open.
  Generator and public-kernel arguments now use the physical name
  ``bessel_argument`` to prevent another normalization ambiguity.

- 2026-07-16: Added an independent equation-level slab ITG gate for equations
  (2.14)--(2.18) of Frei, Hoffmann & Ricci (2022). It found that the
  diamagnetic drive incorrectly zero-padded the known Bessel coefficient
  :math:`K_{J+1}` at the retained Laguerre boundary. The serial and
  Hermite-sharded paths now evaluate that coefficient analytically while still
  truncating the distribution moment :math:`N_{p,J+1}`. The complete
  collisionless matrix at :math:`k_\perp=0.5`, :math:`k_\parallel=0.1`,
  :math:`\eta=3`, and :math:`\tau=1` agrees with the independently assembled
  published hierarchy to roundoff in x64 and :math:`2\times10^{-6}` in default
  precision. The collisional scan remains fail-closed pending the exact table.

- 2026-07-16: Corrected the finite-wavelength runtime and evidence convention
  against equations (2.4)--(2.5) of Frei, Hoffmann & Ricci (2022). The table
  coordinate is the Bessel argument :math:`B=k_\perp v_T/\Omega`, so the
  runtime must interpolate at :math:`B=\sqrt{2b_\mathrm{cache}}`, not
  :math:`\sqrt{b_\mathrm{cache}}`. A nonconstant-table regression now fixes
  that factor. The existing hierarchy is relabeled :math:`B=0.5`, equivalent
  to paper :math:`k_\perp=0.5/\sqrt{2}` at :math:`\tau=1`; the actual paper
  point :math:`B=1/\sqrt{2}` remains fail-closed and is the next bounded build.

- 2026-07-16: Advanced the finite-wavelength Coulomb generator to an
  intermediate hierarchy point without weakening its
  equations. Analytic Laguerre moments, radial-index factorization of the
  matrix and polarization contractions, shared pair caches, and a direct
  weighted inverse projection reduce the exact :math:`(P,J)=(9,4)`,
  Bessel-argument :math:`B=0.5` build from 464.05 to 183.59 seconds on the same
  office CPU. All six arrays retain checksum -152.93360627939981 and finite
  values. Correct full-block indexing gives 3.74% and 1.28% test/field changes
  from :math:`(7,3)` and polarization changes below :math:`1.2\times10^{-8}`.
  The subsequent :math:`(12,5)` point completes in 556.21 seconds with the GMP
  backend; its common :math:`(9,4)` block changes by 2.87% and 1.07%, with
  polarization below :math:`10^{-11}`. The tracked hierarchy therefore passes
  its prospective 5% intermediate-resolution gate while keeping the required
  :math:`(18,6)` resolution and transport acceptance false. The next bounded
  action is the literature geometry/branch gate before the endpoint build;
  brute-force promotion is forbidden.

- 2026-07-16: Replaced the production finite-:math:`m` associated-basis
  transform with a direct polynomial projection after analytically factoring
  :math:`s_\perp^m`. Across all 24,750 coefficients requested by the
  representative :math:`(5,2)` table, it agrees with the cached equation-(B5)
  overlap route to :math:`4.1\times10^{-26}` absolute and is 2.63x faster for
  that transform set. Equation (B5) remains an independent oracle rather than
  production plumbing; direct velocity quadrature, parity-block inversion,
  high-condition cases, and the complete 140-test artifact owner pass. The
  two-wavelength :math:`(5,2)` build now takes 12.44 s versus the original
  184.01 s (14.8x faster), and :math:`(7,3)` takes 247.04 s versus 411.22 s
  after the preceding factorization, with unchanged checksums. Full parity
  blocks were also rejected as a runtime strategy: 19 degree-28/29 blocks of
  size 120 took 79.9 s to invert and lost high-condition accuracy at 32 digits.

- 2026-07-16: Replaced equation (A4)'s six-deep combinatorial basis-transform
  sum with its exact polynomial projection and analytic Gaussian/Laguerre
  moments. Shared caches reduce the 624 transforms used by the representative
  :math:`(5,2)` table by 25.4x, with a worst multiprecision discrepancy of
  :math:`1.7\times10^{-24}` through the exercised set. Independent velocity
  quadrature, shell inversion through the high-condition degree-20 cases, the
  complete 139-test artifact owner, and all table checksums pass. Caching the
  angular associated-basis overlap independently from its radial-order pairs
  then lowers the two-wavelength :math:`(5,2)` build from 26.41 s to 14.58 s.
  The exact
  :math:`(7,3)` office build falls from 411.22 s to 291.61 s with checksum
  -85.857400169932319. This removes another low-order algebra bottleneck but
  does not make the paper's :math:`(18,6)` endpoint affordable yet.

- 2026-07-16: Factored the finite-:math:`m` equation-(B5) overlap into its
  independent radial Laguerre/Gamma and angular differentiated-Legendre sums,
  and cached the result across every Hermite--Laguerre member of a parity
  shell. The direct four-index sum, velocity-space projection, parity-block
  inverse, and ordered-pair table gates remain unchanged. A two-wavelength
  :math:`(5,2)` build with spherical/radial cutoffs 9/4, Bessel--Laguerre
  cutoff 6, and 32 digits falls from 184.01 s to 26.41 s (7.0x) with identical
  checksum after also convolving the three-index Laguerre product and the
  radial/angular overlap coefficients by combined polynomial degree. The
  complete 139-test artifact owner falls from 85.88 s to 44.89 s. A serial
  :math:`(7,3)` two-wavelength table then completes on office in 411.22 s,
  where the former implementation and the rejected row decomposition both
  exceeded 600 s. Its zero-wavelength test/field blocks match the independent
  drift-kinetic generator to relative errors :math:`1.08\times10^{-31}` and
  :math:`1.92\times10^{-32}`. This admits the hierarchy point as development
  evidence, not the still-required :math:`(18,6)` ITG endpoint.

- 2026-07-16: Rejected process-level row decomposition for the offline
  finite-wavelength Coulomb table generator. Serial and decomposed low-order
  tables were bitwise identical, but worker-private multiprecision caches made
  the low-order case slower, and a 32-worker :math:`(P,J)=(7,3)` build on the
  36-core office host produced no artifact within the fixed 600 s campaign
  bound. The experimental API and its added maintenance surface were removed;
  no speedup claim is made. The next performance step must reduce the exact
  transform/contraction complexity before the published :math:`(18,6)` ITG
  endpoint is attempted.

- 2026-07-16: Corrected the finite-wavelength ITG development protocol before
  interpreting moment convergence. The earlier :math:`(3,1)` and
  :math:`(5,2)` probes drew different shape-dependent random states and fit
  only through :math:`t=10`; their reported 0.773-to-0.520 short-wave change
  was not a controlled resolution comparison. A bounded office campaign built
  the complete four-wavelength :math:`(7,3)` table in 530.99 s, then reran all
  three resolutions from one common high-resolution seed through :math:`t=20`
  with a :math:`10\leq t\leq20` fit. At collision weight three, the
  :math:`(5,2)` and :math:`(7,3)` short-wave rates are 0.76098 and 0.76084
  (0.018% relative change); the damped intermediate rates are -0.44503 and
  -0.45584 (2.43%). Collisionless branches remain resolution sensitive. The
  small tracked JSON records the corrected result and fails closed because
  this is still a Cyclone s-alpha development probe, not the paper's geometry
  or required :math:`(18,6)` endpoint.

- 2026-07-16: Re-audited the collisional zonal-flow acceptance target against
  the complete source of Frei, Ernst & Ricci (2022), arXiv:2202.06293. The
  relevant references are Figures 12--14, not Figures 6--8. The prospective
  gate now fixes ion--ion :math:`\nu_i^*=3.13`, :math:`q=1.4`,
  :math:`\epsilon=0.1`, :math:`(P,J)=(24,10)`, drift-kinetic
  :math:`k_x=0.05`, gyrokinetic :math:`k_x=0.1,0.2`, the Xiao long-time
  residual, Coulomb/OS/IS ordering, and the :math:`t\nu=5` velocity-space
  sections. No lower-resolution zonal trace is promoted in its place.

- 2026-07-16: Audited the finite-:math:`b` collisional-ITG plan against the
  complete source of Frei, Hoffmann & Ricci (2022), arXiv:2201.02860. Corrected
  the acceptance target from its analytical Figures 4--5 to the numerical
  operator/short-wave scans in sections 5--6 and the separate velocity-space
  convergence section, whose reported endpoint is :math:`(P,J)=(18,6)`. Shared
  exact multiprecision basis caches reduce one :math:`(3,1)` matrix build from
  12.31 s to 3.06 s and its polarization build from 15.64 s to 4.31 s. A new
  pair-table builder amortizes all wavelength-independent coefficients and
  applies the runtime Laguerre convention explicitly. A bounded six-point
  :math:`(3,1)` Cyclone development probe still excites the short-wave branch
  (:math:`\gamma=0.773` at collision weight three) while damping the
  intermediate branch. This is expected negative evidence below the paper's
  required velocity resolution, not an ITG promotion result; no figure or
  tracked artifact was added. The same cache sharing reduces the complete
  general-artifact test owner from 251 s to under 90 s locally, resolving the
  release-artifact CI timeout without relaxing its eight-minute limit or
  dropping any test. Sharing wavelength-dependent moments across the pair grid
  subsequently reduces the six-point table from 97.2 s to 21.4 s; sharing the
  remaining wavelength-independent speed, product, and inverse contractions,
  then routing polarization through the same cached exact transforms, reduces
  it again to 8.50 s after applying the exact finite-:math:`m` parity selector,
  without changing the equation-level checksum. The next
  :math:`(5,2)` hierarchy point falls from 75.4 s to 44.0 s on four wavelengths.
  Its original short-window growth comparison was subsequently superseded by
  the matched-seed, late-window protocol above.

- 2026-07-16: Closed dimensional Spitzer--Härm normalization for the
  unmagnetized equal-temperature drift-kinetic problem. The generated response
  is now identified explicitly as
  :math:`\sigma_\parallel/[n_e e^2/(m_e\nu_{ee})]`, and the analytic high-charge
  limit is derived as :math:`64/[3\,2^{3/2}\pi Z]`. The converged Coulomb point
  at :math:`Z=100` is 7.453% below this limit, inside the fixed 8% gate. Exact
  matrix-exponential traces use the paper's
  :math:`eE/(\sqrt{m_eT_e}\nu_{ee})=10^{-3}` drive and saturate by
  :math:`t\nu_{ee}=50`; independent steady solves at :math:`10^{-4}`,
  :math:`10^{-3}`,
  and :math:`10^{-2}` close field linearity to roundoff. The publication panel
  now shows the analytic conductivity and saturation trace instead of a
  generation-cost subplot. Finite-:math:`b` ITG and collisional zonal response
  remain open.

- 2026-07-16: Rebalanced the quick CI matrix after the release-artifact shard
  reached its fixed eight-minute job limit. The four-minute general artifact
  owner now runs with release gates, while quasilinear, profiling, and parallel
  artifact owners form a separate matrix shard. The duplicate example suite
  was removed from this shard because ``runtime-core`` already owns it. No
  unique test or acceptance gate was dropped, and the global timeout was not
  relaxed.

- 2026-07-16: Closed the arbitrary-order equal-temperature improved-Sugama
  response hierarchy. The generator evaluates the Coulomb Braginskii
  :math:`N` matrix in multiprecision, removes its momentum-restoring Schur
  complement, and applies the exact drift-kinetic forward/inverse transforms
  from Frei, Ernst & Ricci (2022), equations (79)--(81). Complete total-degree
  shells are enforced at every correction order. The :math:`K=1` boundary
  reproduces every published C103 coefficient to roundoff; the publication
  artifact advances to :math:`K=5`, separately gates the
  :math:`K=4\rightarrow5` response change (0.439%), collision invariants, and
  the fixed 1% Coulomb-response envelope. The measured maximum difference is
  0.307%, with a final velocity-hierarchy change of 0.0237%. This is a
  normalized driven response; the subsequent log entry closes its dimensional
  conductivity and saturation gates.

- 2026-07-16: Extended the converged response to the arbitrary-order,
  equal-temperature original-Sugama boundary without a second coefficient
  generator. The Coulomb test matrix is shared, while a self-adjoint rank-one
  field correction restores parallel momentum and thermal energy. This is the
  exact Sugama--Watanabe--Nunami construction in the orthonormal moment basis:
  it reproduces every tracked C6 coefficient to :math:`3\times10^{-15}` and
  remains non-positive. At :math:`(P,J)=(20,5)`, its normalized current is
  11.29% below Coulomb for :math:`Z=1`, 10.10% below for :math:`Z=2`, and only
  0.61% below for :math:`Z=100`. Literature-anchored 8% low-charge and 2%
  high-charge gates fail closed on this Figure-16 ordering. Subsequent log
  entries close the improved hierarchy and conductivity normalization.

- 2026-07-16: Closed the normalized drift-kinetic Coulomb response hierarchy.
  A direct multiprecision implementation of Frei et al. (2021), equations
  (3.53)--(3.56), removes the finite-wavelength Bessel/product loops and caches
  Appendix-A integrals and raw speed moments. Independent electron--electron
  and electron--ion :math:`(P,J)=(20,5)` blocks now generate in 55.7 and 61.7
  seconds on the tracked CPU, versus the prior generic :math:`(7,3)` path that
  exceeded 600 seconds. Direct and general formulas agree to
  :math:`2\times10^{-15}` at low order for equal and unequal species. The
  nested :math:`(3,1)` through :math:`(20,5)` ion-charge scan has a final
  maximum current change of :math:`1.657\times10^{-4}`; collision invariants,
  self-adjointness, non-positive spectrum, and solve residual pass the fixed
  :math:`2\times10^{-12}` gate. Compact JSON/CSV and publication PNG/PDF
  artifacts are tracked and regenerated from equations. Subsequent entries
  close dimensional conductivity and saturation. A separate post-release
  nonlinear Landau plan follows
  Jorge et al. (2026) with separable Boys-kernel contractions rather than a
  dense collision tensor.

- 2026-07-15: Bounded the next conductivity-resolution attempt on office CPUs.
  Independent electron--electron and electron--ion Coulomb blocks at
  :math:`(P,J)=(7,3)` and spherical/radial cutoff :math:`(13,6)` each used one
  CPU core and about 0.5 GB but reached the fixed 600-second limit without an
  output table. The lower :math:`(3,1)` to :math:`(5,2)` pilot changes the
  normalized :math:`Z=1` current by 1.0%, with the largest sampled change 3.6%
  at :math:`Z=100`; this is encouraging but not a convergence gate. Repeating
  the generic finite-:math:`b` contraction for longer is rejected. The next
  generator must implement the direct drift-kinetic Coulomb equations
  (3.53)--(3.56) of Frei et al. (2021), validate them bitwise on the low-order
  overlap, and only then target the paper's :math:`(20,5)` conductivity scan.

- 2026-07-15: Removed eager runtime imports from the shared linear-validation
  artifact owner. Importing the multiprecision collision generator now loads no
  SPECTRAX-GK or JAX modules and performs no CPU/GPU device discovery; linear
  benchmark and observed-order dependencies remain lazy in their respective
  subcommands. A subprocess regression enforces this boundary. This fixes an
  office failure where a nominal CPU coefficient job attempted CUDA
  initialization and exhausted GPU memory.

- 2026-07-15: Specialized the multiprecision Coulomb generator at the exact
  drift-kinetic endpoint. Because :math:`K_{n>0}(0)=0` and the outer
  gyroaverage removes :math:`m>0`, those branches are skipped before moment
  construction. Test and field blocks are bitwise identical to pre-change
  references; the converged eight-mode :math:`(p_{\max},j_{\max})=(6,3)` point
  falls from 11.1 to 4.24 seconds. A first resolved current pilot changes by
  1.0% at :math:`Z=1` between :math:`(P,J)=(3,1)` and :math:`(5,2)`, but the
  :math:`(7,3)` endpoint remains required before calling the velocity response
  converged.

- 2026-07-15: Implemented the strongly collisional driven-response algorithm
  from Frei, Ernst & Ricci (2022), equation (81). The JAX source places the
  linearized electric force only in :math:`N^{10}`, and a constrained dense
  solve removes stated invariant/truncated modes before computing the steady
  current. Analytic damping, matrix-exponential saturation, JIT, and AD/finite-
  difference gates pass. The published low-order original/improved Sugama
  blocks reproduce the expected greater-than-10% low-:math:`Z` current split
  and less-than-1% high-:math:`Z` convergence. This closes the reusable response
  algorithm, not Spitzer conductivity: the Coulomb/original/improved
  :math:`(P,J)=(20,5)` tables, time saturation, resolution scan, and Figure-16
  ion-charge comparison remain required.

- 2026-07-15: Added the first coupled finite-wavelength moment-resolution gate.
  A Bessel-only audit had hidden a 28.7% error in the tracked 4-by-4 block from
  freezing the spherical hierarchy at :math:`(p_{\max},j_{\max})=(3,1)`.
  Against a converged :math:`(9,4)` reference, :math:`(8,4)` passes a fixed
  :math:`2\times10^{-6}` gate at :math:`8.685\times10^{-7}`. At that admitted
  hierarchy, Bessel order four agrees with order six to
  :math:`4.499\times10^{-7}`. The JSON, publication panel, and fast tracked
  test now fail closed on Bessel, spherical, and radial truncation separately.
  Call-local matrix reuse keeps complete regeneration at 244 seconds under the
  five-minute policy.

- 2026-07-15: Replaced an ill-posed finite-wavelength particle-conservation
  proposal with the actual equation-(3.5) gyro-diffusion gate. Equation (3.35)
  maps a gyrocenter distribution to particle moments but cannot invert a
  gyrophase-averaged collision operator. The regenerated publication artifact
  now resolves test, field, and combined density rows separately; their
  small-:math:`b` observed orders are 2.000, 1.978, and 1.940, respectively,
  and all lie inside the prospectively fixed 1.7--2.3 quadratic window. Local
  equations-(3.2)--(3.4) conservation remains a valid requirement only for a
  future operator evaluated before gyroaveraging at fixed particle position.

- 2026-07-15: Profiled the finite-wavelength coefficient generator before the
  particle-space conservation sweep. Integer polynomial indices were taking
  mpmath's generalized gamma-based binomial path 562,228 times. Replacing that
  path with exact ``math.comb`` reduced a representative :math:`N_\ell=3`
  matrix from 34.1 to 5.43 seconds (6.3x) with bitwise-identical test and field
  blocks. Call-local memoization of the inverse associated-Legendre and
  Laguerre-product contractions reduces the same point to 3.46 seconds (9.9x
  total), while :math:`N_\ell=4` and 5 now finish in 5.25 and 8.19 seconds.
  All optimized blocks are bitwise identical to their saved references. The
  equation-(3.35) :math:`M^{000}` adjoint-gyroaverage contraction uses positive
  paper-convention weights and leaves a nonzero residual. The audit then
  identified the underlying category error: equation (3.35) maps a distribution
  from gyrocenter to particle moments, but does not invert equation (3.5)'s
  gyroaverage of the collision operator. The residual is therefore rejected as
  a conservation gate. The valid finite-:math:`b` checks are the separate
  test/field/combined :math:`O(b^2)` gyro-diffusion laws, while local
  particle-space conservation must be tested on a future pre-gyroaverage
  particle operator.

- 2026-07-15: Kept the five-minute CI policy after the expanded multiprecision
  collision artifact exceeded the wide-coverage shard timeout. Every push now
  validates the tracked JSON gates, matrix/null-space shapes, finite-:math:`b`
  convergence, gyro-diffusion order, and PNG integrity in a fast test; complete
  arbitrary-precision regeneration is marked as a nightly slow physics test.
  This changes scheduling, not the scientific acceptance criteria.

- 2026-07-15: Corrected the finite-wavelength conservation policy against Frei
  et al. (2021). Local invariant nulls apply at particle position and in the
  drift-kinetic limit, not to finite-:math:`b` gyrocenter density. The tracked
  gate now recovers the :math:`b=0` null and measures nonzero classical gyro-
  diffusion with observed leading :math:`O(b^2)` scaling. The multiprecision
  generator also caches repeated combinatorial transforms and factorizes the
  output/input contraction with bitwise-identical six-mode coefficients. A
  matrix-level finite-:math:`b` sweep reduces the assembled-block error from
  :math:`1.17\times10^{-3}` at Bessel order two to :math:`1.57\times10^{-7}`
  at order four against the order-six reference; broader velocity resolution
  remains open.

- 2026-07-15: Extracted all linearized collision equations, table interpolation,
  and runtime application from the generic cache-array module into one focused
  ``operators/linear/collisions.py`` owner. The public lazy facade is unchanged;
  cache arrays fall from 874 to 218 lines, the new owner is 923 lines, and the
  architecture, API, coverage, and differentiability manifests name the new
  boundary explicitly.

- 2026-07-15: Added the differentiable finite-wavelength Coulomb runtime
  assembly. Independent target/source table axes are bilinearly interpolated
  from the real cache, test and field matrices act on the correct gyrocenter
  species moments, and four polarization vectors couple to the solved field
  through distinct charge/temperature factors. Pair-loop, JIT, JVP/finite-
  difference, cancellation, generated-coefficient, and full linear-RHS tests
  pass. Efficient converged table generation and physical transport remain
  open; no input-file selector was added.

- 2026-07-15: Corrected unlike-species finite-wavelength Coulomb generation.
  The test-particle block and outer gyroaverage use target :math:`b_a`, while
  the field-particle moment map uses source :math:`b_b`, as required by
  equations (3.48)--(3.49). A held-target regression changes :math:`b_b` and
  verifies that the test block is unchanged while the field block responds.

- 2026-07-14: Added the tracked Coulomb operator closure artifact. One command
  writes a 228-KB publication panel and strict JSON covering finite-:math:`b`
  Bessel convergence, five independent manufactured velocity projections,
  published coefficients, symmetry, three invariant nulls, and the H-theorem.
  All six gates pass; README and the operator/numerics docs state explicitly
  that Braginskii/Spitzer transport, collisional ITG, and zonal damping remain
  runtime promotion gates.

- 2026-07-14: Completed the offline finite-:math:`b` Coulomb algebra from Frei
  et al. (2021), equations (3.41) and (3.48)--(3.50). Test and field matrices
  retain explicit basis/Bessel truncations, while four polarization vectors
  retain the distinct target/source :math:`q\phi/T` couplings required by
  quasineutrality. At :math:`b=0`, all independent six-moment entries match;
  the full block is symmetric, negative semidefinite, and conserves density,
  momentum, and energy. Five :math:`\Pi^{pjm}` values match direct
  :math:`J_0J_m` velocity projection, and like-species polarization cancels.
  Runtime multispecies assembly and physical transport promotion remain open.

- 2026-07-14: Resolved the finite-:math:`m` inverse used by the full collision
  sum. The literal Jorge et al. (2019) equation-(B6) normalization remains
  rejected, while Frei et al. (2021), equation (3.33), includes the weighted
  Laguerre-product contraction and matches every entry of independently
  inverted degree-six blocks for :math:`m=0,1,2,3`. Matrix inversion remains
  the independent gate; equation (3.33) now provides the scalar coefficient
  for equations (3.48)--(3.50).

- 2026-07-14: Completed Appendix A's velocity-integrated Coulomb speed
  functions. ``coulomb_speed_moments`` evaluates the test and field moments in
  equations (A5) and (A13) for arbitrary positive mass and temperature ratios.
  Six cases over mass ratios ``0.25..4``, temperature ratios ``0.5..3``, and
  spherical order through three agree with direct Maxwellian quadrature of
  equations (A2) and (A10). Equal-species density vanishes and momentum
  test/field pieces cancel. The complete equations-(3.48)--(3.50) matrix
  contraction remains the next promotion boundary.

- 2026-07-14: Implemented the two associated-Laguerre product contractions
  from Frei et al. (2021), equations (3.36)--(3.37) and (3.44)--(3.45), and
  composed them with the finite-:math:`m` transform and exact Bessel kernel to
  generate equation (3.35)'s gyro-moment-to-spherical-moment coefficients.
  Four product families reconstruct pointwise. Six finite-:math:`b`
  coefficients through :math:`m=3` agree with independent Bessel-weighted
  velocity projection, and 20- versus 32-term sums agree. This is a required
  contraction primitive, not promotion of the complete Coulomb operator.

- 2026-07-14: Closed finite-:math:`m` collision-basis coefficient generation
  without adding a module or test file. A literal Jorge et al. (2019)
  equation-(B5) transcription was shown to require :math:`2(-1)^m` under the
  stated associated-Legendre convention; the correction is fixed by the
  isotropic endpoint, eight independent Gauss--Hermite/Gauss--Laguerre
  projections, and pointwise reconstruction. The generator retains every
  lower reduced-degree shell of matching parity. Even and odd blocks through
  degree six and :math:`m=0,1,2,3` reconstruct and invert; the literal printed
  equation-(B6) inverse is rejected for :math:`m>0`, so accepted blocks are
  inverted before the final ``float64`` cast. Full collision contractions and
  transport validation remain open.

- 2026-07-14: Implemented the cancellation-sensitive isotropic
  Legendre/associated-Laguerre to Hermite/Laguerre basis transform and inverse
  from Jorge et al. (2017), equations (A4) and (A3), inside the existing
  multiprecision generator. Eight selected coefficients agree with independent
  80-point Gauss--Hermite/Gauss--Laguerre projections, and all degree shells
  through 12 close under forward/inverse multiplication. The degree-12
  condition number is ``1.926e8`` while inverse error is ``8.73e-15``;
  40- and 100-digit evaluations also agree on high-order extremal entries. A
  review caught and fixed an intermediate ``float64`` downcast in the inverse
  path before commit. A literal Jorge et al. (2019) equation-(B5) prototype is
  rejected: it gives half the required :math:`m=0` endpoint and opposite odd-
  :math:`m` sign under the paper's stated associated-Legendre convention. The
  finite-:math:`m` path remains unshipped pending a convention-corrected
  derivation and pointwise reconstruction gate.

- 2026-07-14: Added the generalized-Laguerre monomial coefficients from Frei
  et al. (2021), equation (3.10), to the same multiprecision collision
  generator. Tensor orders ``0``, ``1``, and ``4`` and polynomial orders ``0``,
  ``3``, and ``8`` reconstruct independent SciPy generalized-Laguerre values
  over ``x=0..12`` within ``2e-8`` absolute tolerance. The next required
  Hermite/Legendre transform is not fully defined in that paper and exhibits
  the 40-order cancellation growth reported in its Figure 1. A primary-source
  follow-up located the base transform in Jorge et al. (2017), Appendix A
  equation (A4), and the finite-:math:`m` extension/inverse in Jorge et al.
  (2019), Appendix B equations (B5)--(B6). The next tranche can therefore
  implement those sums with arbitrary precision and independent
  projection/inverse gates rather than relying on OCR or unavailable COSOlver
  files.

- 2026-07-14: Added the first exact collision-specific Appendix-A generator
  block without adding a tool or test file. ``coulomb_speed_integrals``
  evaluates :math:`e_{ab}^k` and :math:`E_{ab}^k` from Frei et al. (2021),
  equations (A8a)--(A8b), with configurable multiprecision arithmetic. Direct
  improper quadrature of the defining error-function integrals agrees through
  order five at thermal-speed ratios ``0.35``, ``1.0``, and ``2.2`` to tighter
  than ``2e-12`` relative tolerance. This independently resolves the
  PDF/OCR-ambiguous :math:`\chi/2` factor and advances the offline generator;
  no runtime collision claim changes until complete matrix contractions pass.

- 2026-07-14: Completed the associated-Laguerre Bessel expansion layer from
  Frei et al. (2021), equation (2.12). The recurrence-generated coefficients
  for :math:`m=0,1,2` reconstruct independent :math:`J_m(b\sqrt{x})` values at
  four wavenumbers and 41 velocity points; maximum absolute errors are
  ``4.99e-7``, ``1.45e-8``, and ``1.06e-7``, respectively. The :math:`m=0`
  coefficients also agree with the separately gated :math:`K_n` kernel. This
  completes the exact Bessel-expansion prerequisite without adding modules;
  the speed-function contractions and complete collision matrices remain the
  fail-closed scientific boundary.

- 2026-07-14: Implemented the first exact finite-Larmor generator primitive
  from Frei et al. (2021), equation (2.13), without adding a source or test
  file. ``bessel_laguerre_kernels`` uses the stable recurrence
  :math:`K_{n+1}=K_n b^2/[4(n+1)]`, preserves the exact zero-wavenumber limit,
  and remains JIT/JVP compatible. An independent 96-point Gauss--Laguerre
  projection of :math:`J_0(b\sqrt{x})` verifies orders zero through five at
  four wavenumbers; normalization, recurrence, the paper's sub-0.1% tail at
  :math:`b=1,N=3`, and centered finite-difference gates also pass. This closes
  a coefficient-kernel prerequisite only; the cancellation-sensitive
  collision-specific sums and their conductivity/ITG/zonal evidence remain
  open.

- 2026-07-14: Implemented the complete lowest-order improved-Sugama correction
  from Frei, Ernst & Ricci (2022), Appendix C, equations (101)--(103). The
  driven/response index orientation is explicitly converted to the runtime
  matrix convention; an initial untransposed implementation was rejected when
  its unequal-species momentum gate failed. The corrected JAX matrix conserves
  all multispecies invariants, differentiates through temperature ratios, is
  non-positive over the full equal-temperature reduced space, and reduces the
  equal-species heat-flow distance to Coulomb by about 61%. The 80-digit
  package table now includes the independent equation-(103) endpoint. This is
  a friction-flow equation gate, not a conductivity promotion.

- 2026-07-14: Added a time-integrated finite-wavelength admission boundary for
  the reduced improved-Sugama operator. In a deterministic ``Nl=2, Nm=4``
  Cyclone probe, strong collisions damp the intermediate ``ky`` mode but
  destabilize the higher-``ky`` short-wave branch. This reproduces the failure
  mechanism reported when collisional FLR terms are omitted and is retained as
  negative evidence: the reduced drift-kinetic operator remains Python-only
  and configuration-file promotion requires the full finite-``b`` hierarchy.

- 2026-07-14: Added a vectorized physical-species assembler for the reduced
  multispecies Sugama pair operator and an independent matrix-exponential
  relaxation gate. A two-species state preserves particle number, total
  parallel momentum, and total thermal energy at the shipped float32 precision
  while its collision residual falls by more than five orders of magnitude.
  This closes the reduced time-domain relaxation check; conductivity,
  collisional ITG, zonal damping, and the full finite-``b`` hierarchy remain.

- 2026-07-14: Routed the reduced multispecies Sugama matrix through the real
  collision extension seam as a JAX-pytree operator. A two-species,
  post-field-Hamiltonian linear RHS produces a nonzero collision response while
  retaining particle, total-momentum, and total-energy invariants. This is a
  Python research boundary only; TOML selection remains fail-closed until the
  conductivity and collisional-ITG promotion gates exist.

- 2026-07-14: Implemented the published original-Sugama low-order
  multispecies boundary from Frei, Ernst & Ricci (2022), Appendix C, equations
  (C4)--(C5). Separate differentiable test- and field-particle matrices depend
  on :math:`m_a/m_b` and :math:`T_a/T_b`; a target/source-species JAX
  contraction applies directed pair blocks without Python runtime loops. Their
  equal-species sum reproduces the independent C6 table, unequal-species
  coefficients match fixed high-precision values, and a physical two-species
  frequency matrix conserves particles, total momentum, and total thermal
  energy while dissipating the weighted quadratic norm. Ratio JVPs agree with
  centered finite differences. This promotes a real reduced multispecies
  Sugama equation slice, not improved-Sugama or full finite-``b`` runtime use.

- 2026-07-14: Replaced the controlled finite-``b`` table check as the sole
  interpolation evidence with a held-out physical-operator convergence gate.
  Dense matrices are constructed by direct application of the implemented
  Mandell--Dorland--Landreman finite-Larmor-radius equations, independently
  interpolated at four off-grid :math:`k_\perp` values, and compared on a
  nontrivial complex state. Refining table spacing from 0.4 to 0.1 reduces the
  relative error from about ``1.46e-2`` to ``8.87e-4`` with observed order near
  two. This validates the runtime table boundary on real finite-``b``
  coefficients while leaving Sugama/Coulomb finite-``b`` promotion blocked.

- 2026-07-14: Extended the checked collision-table boundary to the spatially
  varying finite-``b`` application required by a full operator. A JAX-native
  piecewise-linear interpolator accepts shared or species-specific
  :math:`k_\perp` coefficient tables, clamps outside the generated interval,
  and feeds static or pointwise dense matrices into the existing Hermite-major
  kernel. Node/endpoint identity, JIT execution, species-local application,
  direct-equation identity, malformed-grid rejection, and interior-target
  JVP/finite-difference gates pass. The tracked coefficients remain the
  drift-kinetic C6/C9 slice; no finite-``b`` physics claim is promoted until the
  complete published hierarchy is generated and scientifically gated.

- 2026-07-14: Completed the first advanced-collision generated-table vertical
  slice without adding a source, test, or tool file. The consolidated linear
  validation command now evaluates the published C6/C9 matrices with 80-digit
  ``mpmath`` arithmetic and writes a deterministic 1.2 KB ``.npy`` table plus
  SHA-256/source/convention metadata. Package-resource loading fails closed on
  checksum or shape drift, and a generic species-aware JAX moment-matrix kernel
  preserves Hermite-major paper ordering while restoring the code's state
  layout. Generated Sugama/Coulomb results agree with both direct equation
  kernels, and state/frequency JVPs agree with centered finite differences.
  The office audit found no retained COSOlver source or coefficient matrices,
  so the next tranche must implement or provenance-review the complete
  finite-``b`` coefficient generator rather than silently import unknown data.

- 2026-07-14: Added the exact linearized Coulomb companion to the reduced
  Sugama equation gate using Frei, Ernst & Ricci (2022), Appendix C, equations
  (C9a)--(C9f). Both published matrices now share one JAX kernel and retain
  their distinct thermal and heat-flux blocks. Exact coefficient, symmetry,
  invariant, dissipation, null-space, and derivative tests pass, while the
  linear dissipation owner stays below the 1000-line budget. A source audit of
  GYACOMO revision ``a5d2c5ca`` confirms its full advanced operators use
  offline COSOlver tables with separate self/test/field matrices and runtime
  :math:`k_\perp` interpolation. SPECTRAX-GK will follow that robust
  generation/application boundary rather than evaluate cancellation-sensitive
  nested sums inside traced runtime kernels.

- 2026-07-14: Implemented the published like-species drift-kinetic Sugama
  six-gyromoment operator from Frei, Ernst & Ricci (2022), Appendix C,
  equations (C6a)--(C6f). The implementation maps ``(p,j)`` to the code's
  ``(ell=j,m=p)`` storage and applies the required Laguerre-sign transform.
  Exact matrix, Maxwellian null-space, density/momentum/thermal invariant,
  symmetric negative-semidefinite block, quadratic dissipation, and
  collision-frequency JVP/finite-difference gates pass. The source remains
  within its 1000-line architecture budget without a new file. This is a
  validated high-collisionality reduced equation slice, not a TOML-selectable
  full-hierarchy finite-``b`` or multispecies Sugama/Coulomb claim.

- 2026-07-13: Fixed the repository-hygiene CI regression introduced by command
  consolidation. Pillow is now imported only by the opt-in
  ``compress-previews`` path, so the default size and release-artifact checks
  retain their minimal dependency contract. A subprocess gate blocks every
  ``PIL`` import while loading the checker and reproduces the GitHub Actions
  environment; all nine focused repository-hygiene gates pass.

- 2026-07-13: Folded PNG preview compression into the existing repository-size
  and release-artifact policy command as the explicit ``compress-previews``
  mode. Compression functions remain directly tested, but image mutation is
  now opt-in under the same manifest owner that selects previews and verifies
  their size/checksum provenance. The standalone artifact script is removed,
  the combined command remains below its complexity budget, and the tool
  inventory falls to 119.

- 2026-07-13: Unified linear comparison figures, observed-order reports, and
  KBM branch-continuity gates under ``build_linear_validation_artifacts.py``.
  The explicit ``figures``, ``observed-order``, and ``kbm-branch`` modes retain
  distinct promotion policies while sharing one parser and artifact owner.
  All 78 artifact tests and 10 focused campaign/release contracts pass; the
  retired duplicate figure script lowers the tool inventory to 120. The
  collision source audit also established that the published finite-Larmor-
  radius Sugama/Coulomb sums require an offline multiple-precision coefficient
  generator, checksummed tables, and JAX runtime application rather than
  cancellation-prone direct ``float64`` evaluation.

- 2026-07-13: Repaired the source-complexity regression exposed by the strict
  architecture gate. The nonlinear full-:math:`f` Dougherty cross-species
  reference now lives with nonlinear collision policies rather than in the
  linear dissipation hotspot; the public re-export remains stable. The
  nonlinear operator facade is lazy, removing an eager RHS assembly cycle and
  allowing low-level collision utilities to import independently. The hotspot
  falls from 1,021 to 901 lines with no new file, 15 focused collision gates
  and the fresh-interpreter import gate pass, and the architecture baselines
  now freeze the actual 226 source, 96 test, and 120 tool files.

- 2026-07-13: Added an independently auditable drift-kinetic Dougherty kernel
  from Frei, Hoffmann & Ricci (2022), Appendix C, equation (C6), mapped to the
  code's Laguerre sign and ``(species, ell, m, ky, kx, z)`` conventions. A new
  gate proves that the production finite-Larmor-radius operator reduces to
  this equation at ``b=0`` and separately checks density, momentum, and thermal
  invariants, non-positive quadratic rate, and a collision-frequency JVP
  against centered finite differences. The office source audit confirms the
  comparison implementation uses the same limited Dougherty family, so it is
  not evidence for Sugama/Coulomb promotion.

- 2026-07-13: Made the benchmark-refresh manifest's figure command executable
  as written. It had passed a removed ``--reuse-cyclone-mismatch`` option even
  though the current figure builder always consumes the reviewed mismatch CSV.
  The stale flag is removed and the manifest test now parses the exact
  configured figure arguments through the production parser, preventing future
  command/API drift.

- 2026-07-13: Consolidated zonal-response plotting and optimization-row
  generation under ``build_zonal_flow_artifacts.py``. The explicit
  ``response-csv``, ``response-output``, and ``objective-gate`` modes preserve
  separate data and promotion policies while sharing extraction, metric, and
  plotting ownership. Five focused behavior gates pass, the standalone
  response plotter is removed, and the tool inventory is now 121.

- 2026-07-13: Folded the standalone eigenfunction-diagnostics plotter into the
  existing linear-reference artifact command. ``overlap-summary``,
  ``reference-overlay``, ``kbm``, and ``w7x`` now share one owner and the same
  plotting kernels; focused artifact reproduction passes and the duplicate
  script is removed. Tool inventory and its no-regression baseline are 122.

- 2026-07-13: Corrected the advanced-collision roadmap after a primary-source
  audit. Francisquez et al. (2022) derive a nonlinear full-:math:`f`
  multispecies Dougherty operator, while SPECTRAX-GK evolves a linearized
  delta-:math:`f` gyrokinetic state. Renamed the reference kernel to
  ``conservative_full_f_dougherty_cross_moments`` and documented that its exact
  primitive targets cannot be inserted directly into the shipped linearized
  field-particle correction. The runtime lane now targets the published
  linearized gyro-moment Sugama/Coulomb coefficients; a multispecies Dougherty
  runtime variant requires its own explicit delta-:math:`f` projection.
  A fresh-interpreter import check also exposed and fixed a circular dependency
  introduced by post-field collision context construction; ``build_H`` is now
  resolved only when the optional custom operator runs, and a subprocess
  regression gate protects low-level term imports from facade order.

- 2026-07-13: Removed the backend-free synthetic stellarator portfolio panel,
  generator, and three tracked companions. The objective reducer remains unit
  tested, while paper-facing documentation and coverage ownership now point to
  the real QH VMEC/Boozer aggregate-objective artifact and its fail-closed
  provenance, sample-coverage, objective-column, and AD/FD guard. That guard,
  warning-strict docs, and the validation manifest pass. Tool inventory is 123,
  with the no-regression baseline lowered accordingly.

- 2026-07-13: Replaced the pre-field, distribution-only collision callback with
  a typed post-field ``CollisionContext`` carrying :math:`G`, :math:`H`, solved
  fields, cache, and parameters. Linear and nonlinear RHS routes disable the
  built-in term before assembly, then evaluate the custom operator with the
  same electrostatic/electromagnetic field policy; normal runs pay no new cost
  and extension runs do not repeat the field solve. Focused tests prove the
  operator sees :math:`H=G+J_0\phi/T` and preserve weighted replacement,
  hypercollision independence, shape rejection, JIT integration, and protocol
  behavior. All 62 linear integration tests and warning-strict documentation
  pass. The exact full-f Francisquez cross moments are not inserted directly into the
  existing self-collision restoration because that would mix primitive and
  normalized gyrokinetic moments without a derivation; the complete projected
  operator remains explicitly blocked on conservation and entropy gates.

- 2026-07-13: Implemented the exact conservative cross-species primitive
  moments from equations (2.11)--(2.12) of Francisquez et al. (2022). The JAX
  kernel supports arbitrary mass ratios, directed rates, and independent
  trailing spatial samples; rejects statically invalid density, mass,
  temperature, and rate inputs; and passes pairwise momentum/energy,
  equal-species, positivity, shape, and AD/finite-difference gates. This closes
  a nonlinear full-f reference contract. They do not promote or directly
  parameterize the linearized gyroaveraged collision RHS.

- 2026-07-13: Promoted VMEC and Miller EIK generation from a maintainer artifact
  script to the installed ``spectraxgk geometry`` executable. Both backends now
  share the explicit ``--config``/``--out``/``--force`` contract; Miller keeps
  its optional helper interpreter and repository controls, while the ambiguous
  legacy positional config was removed. Tests moved to the executable suite,
  documentation now teaches the public command, and tool inventory is 124. The
  new nested command exposed an obsolete top-level argparse positional that
  greedily consumed explicit subcommands; direct TOML shorthand already has a
  pre-parser, so removing the unused positional repairs explicit nested command
  dispatch without changing ``spectraxgk case.toml``.

- 2026-07-13: Removed the orphan nonlinear comparison executable. It was absent
  from the benchmark manifest and documentation and retained a second
  CSV/NetCDF parser plus print-only early/late tolerance policy. The maintained
  nonlinear diagnostics comparator owns trace and resolved-spectrum comparison,
  while the nonlinear-window gate owns native-window statistics and release
  acceptance. Tool inventory is 126.

- 2026-07-13: Consolidated the solved-equilibrium linear launch screen into
  ``tools/artifacts/build_nonlinear_transport_admission.py linear-screen``.
  The fail-closed growth, effective-perpendicular-wavenumber, heat-flux-weight,
  and sampled-``ky`` gates retain their tracked JSON/CSV schema, while one
  duplicate artifact executable and its generic ``build_report`` API were
  removed. Tool inventory is 125; the maintained command now owns the complete
  progression from linear triage through nonlinear landscape, prelaunch,
  redesign, and optimizer-campaign admission. The consolidation audit also
  fixed a fail-closed bug: non-finite effective-wavenumber and heat-flux-weight
  rows had previously been discarded before policy evaluation and can no
  longer allow a partially corrupt spectrum to pass.

- 2026-07-13: Consolidated the nonlinear-gradient evidence chain. Central
  finite-difference, variance-reduction planning, and independent control-mean
  gating now use the ``finite-difference``, ``variance-plan``, and
  ``control-mean`` subcommands of one artifact owner. Existing JSON, CSV, and
  publication-figure schemas are unchanged, tracked replay commands were
  migrated, and the tool inventory is 127.

- 2026-07-13: Removed the duplicate external-VMEC holdout selector. The
  maintained runbook already reads the candidate screen, incorporates the
  calibration-gap report, rejects stable and near-marginal rows, accounts for
  represented and failed geometry families, and emits the canonical config
  generator command. Deleting the weaker largest-growth selector leaves one
  scientific policy owner and reduces the tool inventory to 128.

- 2026-07-13: Added the physically weighted multispecies collision-invariant
  diagnostic required before implementing species-coupled field-particle
  terms. It checks particle conservation for every species and total parallel
  momentum and thermal energy with the code's species normalization; an AD gate
  verifies the diagnostic remains differentiable. This deliberately precedes,
  rather than substitutes for, the mass/temperature-ratio-dependent Sugama or
  Coulomb operator.

- 2026-07-13: Corrected the finite-Larmor-radius collision claim against
  Mandell, Dorland & Landreman (2018). Finite-``b`` guiding-centre moments are
  not locally conserved because collisions are local in real space; this is not
  a coefficient residual to tune away. Added a direct equations (3.38)--(3.42)
  gate covering diagonal damping, parallel/perpendicular flow restoration, and
  temperature restoration at ``b=0.7``, plus negative quadratic free-energy
  response. The next model remains genuinely species-coupled Dougherty, then
  Sugama/Coulomb.

- 2026-07-13: Removed the experimental VMEC-JAX WOUT metadata patcher. Current
  VMEC-JAX computes and tests ``Aminor_p``, ``Rmajor_p``, ``aspect``, and
  ``volume_p`` from the solved equilibrium, so retaining an approximate LCFS
  scalar rewrite could mask an upstream output failure. SPECTRAX-GK now relies
  on the authoritative WOUT contract; runtime EIK generation rejects invalid
  reference length, while in-memory workflows can supply an explicit scale.
  Tool inventory is 129.

- 2026-07-13: Folded the standalone linear stress-matrix executable into
  ``tools/comparison/compare_runtime.py stress-matrix``. The KAW, kinetic-
  electron Cyclone, and Miller KBM developer comparisons retain the same input
  and combined-CSV contract while one duplicate command owner is removed. Tool
  inventory is now 130; parser and subprocess/CSV routing tests pass.

- 2026-07-13: Closed a collision-splitting correctness gap. The nonlinear split
  policy previously removed the complete conserving collision contribution but
  advanced only diagonal damping, omitting its low-order field-particle terms.
  It now splits only diagonal hypercollisions and leaves conserving collisions
  in the explicit/IMEX RHS. Added a structural ``SplitCollisionOperator``
  contract plus reusable long-wavelength invariant and quadratic free-energy
  diagnostics. Literature-anchored tests verify density, parallel momentum,
  thermal energy, the local-Maxwellian null space, and dissipative response at
  ``b=0``; finite-``b`` Sugama/Coulomb promotion remains explicitly open.

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
python tools/release/check_package_architecture_manifest.py differentiable-refactor
python tools/release/check_parallel_scaling_artifacts.py --performance-manifest-only
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
- 2026-07-12: Added an operator-level long-wavelength collision invariant
  trajectory. Populated high moments produce a nonzero collision response,
  while density, parallel momentum, and the temperature-like low moment remain
  fixed over five serial and species-pmap steps. A separate random-state audit
  found finite-``b`` residuals at a few percent of the collision RHS norm, so
  the documentation now limits conservation claims to ``k_perp*rho=0`` and
  keeps finite-Larmor-radius repair open.
- 2026-07-12: Implemented the first mixed species--Hermite equation path on a
  four-device ``(species,m)=(2,2)`` mesh. Electrostatic density and polarization
  moments use correctly scoped named reductions, Hermite neighbors exchange
  within each species row, and global ``m`` indices select the field drive.
  Direct and explicit-backend calls match the serial periodic streaming RHS on
  four logical CPUs. Broader terms remain fail closed; the two-GPU office host
  cannot validate this four-device GPU topology.
- 2026-07-12: Extended the consolidated linear profiler to the exact mixed
  backend and froze a revision-pinned four-logical-CPU artifact. The
  ``2x4x16x64x1x64`` streaming RHS matches serial to ``5.7e-8`` relative and
  records ``5.13x`` warm speedup. The above-ideal value is scoped to cache and
  working-set effects on this RHS; integration, GPU, and general scaling claims
  remained blocked at this stage.
- 2026-07-12: Promoted the mixed periodic streaming route through bounded time
  integration after three-step Euler and sampled RK2 identity gates passed.
  The revision-pinned 100-step artifact is state/field exact and records
  ``1.68x`` end-to-end speedup on four logical CPUs while the isolated RHS is
  ``5.29x``. Adiabatic quasineutrality also matches serial at RHS level. Linked
  boundaries, drifts, collisions, other methods, and four-device GPU evidence
  remain explicitly outside the claim.
- 2026-07-12: Made the Diffrax derivative policy explicit. The default
  ``derivative_mode="reverse"`` preserves the custom-VJP field solve, while
  ``"forward"`` uses native JAX rules through fixed or adaptive trajectories.
  A nonzero adaptive Tsit5 thermodynamic-drive JVP agrees with centered finite
  differences to ``1.9e-5`` relative and is stable when ``rtol`` is tightened
  from ``1e-3`` to ``3e-4``. The focused Diffrax shard passes 14 tests in 54 s.
  Adaptive reverse mode remains unpromoted after an exploratory probe exceeded
  the bounded local memory envelope; the next route must use an explicit
  checkpointed/direct adjoint rather than relying on accidental controller AD.
- 2026-07-12: Replaced the streaming-only mixed backend with the clearer
  ``electrostatic_species_hermite`` route and extended the same 2x2 mesh through
  the complete collision-free electrostatic linear operator. Width-one and
  width-two Hermite exchanges now cover mirror and curvature coupling; grad-B
  remains local in Hermite space, and global mode indices place diamagnetic
  drives. Isolated term, combined RHS, automatic dispatch, Euler, and RK2 gates
  match the serial production equations on four logical CPUs. The earlier
  streaming timing is retained only as historical scoped evidence until a
  revision-pinned full-operator profile is generated.
- 2026-07-12: Froze the revision-pinned full mixed-operator CPU artifact. The
  ``2x4x16x64x1x64`` workload matches the serial RHS to ``5.6e-8`` relative and
  records ``2.93x`` warm-RHS speedup. A matched 100-step Euler trajectory has
  exact state and field histories but only ``0.89x`` throughput, so the release
  explicitly rejects an end-to-end integration-speedup claim. Generated PNG,
  PDF, and CSV byproducts were deleted; only the 2.7 KiB JSON evidence is kept.
- 2026-07-12: Removed two integration overhead hypotheses without changing the
  equations. A dedicated field-only diagnostic avoided requesting unused RHS
  outputs, and the initial state is now placed on the 2x2 mesh once before the
  scan. The revision-pinned rerun remains exact, records ``3.11x`` RHS speedup,
  and improves 100-step throughput from ``0.89x`` to ``0.97x``. Since it is
  still below crossover, the release retains no end-to-end speedup claim and
  defers deeper collective fusion to the next performance campaign.
- 2026-07-12: Extended the 2x2 species--Hermite route through every shard-local
  dissipative term. Hypercollision rates use the global, not local, Hermite
  resolution; the ``|k_z|`` branch uses each global mode's mask and power;
  hyperdiffusion preserves the dealiased perpendicular spectrum; and end
  damping preserves the production ``dt`` scaling. Each nonzero operator and a
  combined dissipative Euler/RK2 trajectory match serial on four logical CPUs
  in a 147 s bounded gate. Conserving collisions and linked boundaries remain
  fail-closed because they require additional cross-shard moment or topology
  collectives.
- 2026-07-12: Closed the standard conserving-collision equation path on the
  mixed mesh. Species-local :math:`m=0,1,2` density, momentum, and temperature
  moments are reduced only over the Hermite axis, preserving separate unequal
  ion/electron collision rates. The nonzero isolated operator and a combined
  all-electrostatic Euler/RK2 trajectory match serial on four logical CPUs in a
  265 s bounded gate. Pre-expanded collision matrices remain fail-closed; the
  promoted implementation is the standard factorized operator used by runtime
  cases. Production parallelization is now limited primarily by linked
  boundary topology and unavailable four-device GPU hardware.
- 2026-07-12: Closed linked-boundary topology without reconstructing the global
  distribution. Since every species--Hermite shard owns full ``(ky,kx,z)``
  chains, the production linked gradient, linked ``|k_z|`` hypercollision, and
  linked end-damping profile execute locally after Hermite exchange. A
  nontrivial ``Nx=Ny=Nz=8`` linked case with unequal collision rates matches the
  serial combined RHS and two-step state/field trajectory in a 25 s bounded
  four-logical-CPU gate. The remaining mixed-mesh gaps are electromagnetic
  integration and four-device GPU evidence; office has only two GPUs.
- 2026-07-12: Promoted bounded-memory adaptive reverse differentiation. The
  explicit ``derivative_mode="reverse", checkpoint=True`` policy selects
  Diffrax's recursive checkpoint adjoint; its nonzero Tsit5 thermodynamic-drive
  gradient matches the existing forward JVP and centered finite difference to
  ``1.9e-5`` relative. Direct non-checkpointed adaptive reverse remains
  unpromoted after exceeding the local memory envelope, so documentation now
  distinguishes these policies rather than making a generic adaptive-AD claim.
- 2026-07-13: Closed the held-out nonlinear IMEX transport derivative gate.
  A nonzero electrostatic ion heat flux after three nonlinear IMEX steps now
  differentiates through the physical field diagnostic, cache rebuilt from
  ``R/L_Ti``, and the parameter-dependent matrix-free SOLVAX solve. The
  implicit VJP agrees with centered finite differences to about ``9e-6``
  relative and is unchanged when the Krylov tolerance tightens from ``1e-6``
  to ``1e-7``. This closes the bounded equation-level API contract without
  promoting short endpoints as converged turbulent transport averages.
- 2026-07-13: Refined every selected shift-invert Ritz value with the physical
  operator Rayleigh quotient. This scalar minimizes the Euclidean eigenpair
  residual for the fixed Ritz vector and removes avoidable error from mapping
  an inexact inverse Ritz value through ``lambda = sigma + 1 / mu``. A resolved
  streaming case improves from ``0.0655`` to ``0.0472`` and passes a ``0.06``
  nearest-shift gate, while the deliberately under-resolved targeted case still
  fails closed. KBM remains unpromoted pending a stronger restart/preconditioner.
- 2026-07-13: Corrected the KBM Krylov target direction. The tracked KBM
  benchmark reports positive physical frequencies, so its matrix eigenvalue
  target is on the negative imaginary axis; KBM had accidentally shared the
  electron-branch sign policy and targeted the opposite half-plane. On the
  full ``Nl=16, Nm=48, Nz=96`` physical probe the corrected target lowers the
  outer residual from ``0.995`` to ``0.884``, but still fails the unchanged
  ``0.1`` gate. This fixes branch semantics without promoting the unresolved
  solver or weakening its acceptance policy.
- 2026-07-13: Consolidated validation traceability under
  ``check_validation_coverage_manifest.py``. Its backward-compatible default
  validates package ownership and measured coverage, while the explicit
  ``gate-index`` mode scans artifact gate reports and writes the tracked
  JSON/CSV/publication panel. Plotting dependencies remain lazy, so normal CI
  validation stays dependency-light. The duplicate artifact command is removed
  and tool inventory falls to ``118``.
- 2026-07-13: Consolidated the provisional axisymmetric TEM branch audit and
  W7-X kinetic-electron extension tracker under
  ``build_tem_validation_artifacts.py``. The ``axisymmetric-branch`` and
  ``w7x-extension`` modes retain distinct scientific claim boundaries and
  artifact schemas while sharing finite-value cleaning, plotting style, and
  TEM mismatch ownership. Two duplicate executables become one 670-line tool,
  reducing inventory to ``117`` without changing the lane's open status.
- 2026-07-13: Consolidated the reduced QA low-turbulence comparison and its
  time-horizon audit under ``build_qa_transport_validation_artifacts.py``.
  Explicit ``comparison`` and ``horizon-audit`` modes retain separate payloads,
  writers, and pass criteria while sharing the same reduced differentiable
  transport model. Two commands become one 783-line owner, tool inventory falls
  to ``116``, and documentation continues to scope these artifacts as reduced
  envelope evidence rather than converged full-GK turbulent transport.
- 2026-07-13: Consolidated nonlinear feasibility and frozen release-window
  statistics under ``build_nonlinear_validation_panels.py``. Explicit
  ``feasibility`` and ``window-statistics`` modes preserve separate artifact
  schemas and claim boundaries while sharing finite-value serialization and
  plotting ownership. The generic NetCDF loader now has a physical diagnostic
  name rather than comparison-code terminology, and tool inventory falls to
  ``115``.
- 2026-07-13: Consolidated the W7-X zonal literature-contract and
  initializer/observable convention audits under
  ``build_w7x_zonal_validation_artifacts.py``. The ``contract`` mode remains
  explicitly open against digitized long-window traces, while
  ``state-convention`` retains its closed state-level gate. Recurrence,
  closure, and moment-tail hypotheses remain separate owners; inventory falls
  to ``114``.
- 2026-07-13: Consolidated the W7-X zonal moment-tail diagnostic and bounded
  recurrence sweep under ``build_w7x_zonal_recurrence_artifacts.py``. The
  ``moment-tail`` and ``sweep`` modes share NetCDF/path/serialization ownership
  while preserving their respective artifact schemas and open physics status;
  the closure-intervention ladder remains separate. Inventory falls to ``113``.
- 2026-07-13: Repaired the exact-state comparison orchestrator after its move
  into ``tools/campaigns``. It had still searched for comparison commands in
  the campaign directory and built ``PYTHONPATH`` from ``tools/``; checked
  repository/comparison roots now make all three command paths executable, and
  nine focused tests pass. The architecture snapshot now records the actual
  134-tool inventory instead of the stale 213-script migration count.
- 2026-07-13: Audited the ETG Krylov blocker after Rayleigh refinement. At
  ``k_y rho_i=15`` the model-aware target shift is physically located near
  ``+14.94i``, but damping-preconditioned shift-invert returns outer residual
  ``0.584`` and Arnoldi fallback returns ``0.997``. Reusing the real-time IMEX
  Hermite-line factorization as a complex shift preconditioner worsened the
  residual to ``0.951`` and was removed. ETG therefore remains on its validated
  time-integrated path pending a dedicated complex block preconditioner.
- 2026-07-13: Tested a direct complex damping-plus-streaming Hermite-line
  shift-invert factorization at full tracked resolution. The corrected KBM
  probe worsened from damping-preconditioned residual ``0.884`` to ``0.993``
  and selected a damped branch; the ETG ``k_y rho_i=15`` probe worsened from
  ``0.634`` to ``0.968`` and selected the wrong high-frequency branch. The
  implementation was removed. These controlled negative results narrow the
  next design to a field-coupled low-moment Schur/block preconditioner rather
  than another scalar Hermite factorization; unchanged outer residual gates
  remain authoritative.
- 2026-07-13: Re-ran all fast release, performance, parallel-artifact, and
  repository-size checks. Technical release status is ``100%`` with all five
  active scoped manuscript lanes closed, and the tracked tree is ``47.94 MB``
  against its ``50 MB`` budget. The two-GPU whole-state nonlinear row is
  identity-complete but only ``0.586x``; it remains diagnostic negative evidence
  rather than a production speedup claim. Release-claim hygiene is closed while
  a different nonlinear communication algorithm remains future work.
- 2026-07-13: Closed the README/docs release pass after checking every local
  README link and figure, the installed API snippet, scoped claim guards,
  release/performance/size manifests, and all 31 documentation pages with a
  warning-strict Sphinx build. User-facing wording now consistently describes
  nonlinear parallelization without implying a multi-device speedup; the
  documented ``0.586x`` two-GPU result remains explicit negative evidence.
- 2026-07-13: Consolidated exact-state runtime comparison tooling. Startup,
  saved diagnostic-state, and evolved-window audits now use the ``startup``,
  ``diagnostic-state``, and ``window`` subcommands of one 690-line owner;
  imported growth/window diagnostics reuse its binary loaders, and the
  orchestrator invokes the unified command. Three scripts were replaced by
  one, lowering the tool inventory from 134 to 132. The architecture manifest
  now enforces the honest ``<=99`` final target instead of treating 134 as
  complete, while the normal no-regression check remains green.
- 2026-07-13: Consolidated the symmetric and ``LASYM=true`` VMEC
  state-to-input mapping launchers into the explicit ``symmetric`` and
  ``asymmetric`` subcommands of ``write_vmec_state_mapping_campaign.py``.
  Their coefficient restrictions, output schemas, plots, and fail-closed
  mapping claims remain separate, while admitted-control parsing and command
  ownership are shared. The merge exposed and repaired an upstream API and
  physics bug: current ``vmec_jax`` uses the public ``VmecInput`` API, and a
  requested asymmetric mode outside the seed ``NTOR/MPOL`` extent must expand
  all boundary/axis arrays before writing or the parser silently discards it.
  A round-trip test now proves ``LASYM=true``, ``NTOR=1``, and the requested
  ``ZBC(1,1)`` value. The weighted state-control short-bracket launcher now
  uses the same public-API writer, eliminating the same stale dependency and
  silent-mode risk there. The tool inventory is now 131.

- 2026-07-13: Consolidated the Merlo Case-III Miller zonal-response generator
  into ``build_zonal_flow_artifacts.py miller-panel``. The unified command now
  owns CSV, saved-output, optimization-gate, and literature-gated Miller
  zonal-response artifacts without changing the Merlo normalization, fit
  policy, or JSON schema. The physical runtime-to-NetCDF regression gate is
  retained, figures are explicitly closed after writing, and the tool inventory
  falls to ``112`` against the enforced ``99`` target.

- 2026-07-13: Unified W7-X zonal-reference digitization and comparison under
  ``build_w7x_zonal_reference_artifacts.py`` with explicit ``digitize`` and
  ``compare`` modes. Pixel calibration, residual extraction, normalization,
  time-coverage, envelope, and open-gate exit semantics remain independently
  tested; the tracked W7-X residual/envelope mismatch remains open scientific
  evidence rather than being obscured by the refactor. The tool inventory falls
  to ``111`` against the enforced ``99`` target.

- 2026-07-13: Consolidated exact-state comparison execution and publication
  reporting under ``tools/comparison/build_exact_state_audit.py`` with explicit
  ``run`` and ``report`` modes. Manifest path/environment handling, comparison
  command discovery, log parsing, scalar/array convention gates, and artifact
  writing remain directly tested. This keeps comparison-code terminology inside
  an explicit benchmark boundary and lowers the tool inventory to ``110``.

- 2026-07-13: Prototyped a JAX-linear field-coupled low-moment defect
  correction for the shift-invert damping preconditioner. A synthetic coupled
  shifted-system gate reproduced the expected residual reduction and linearity,
  but the full x64 ``Nl=16,Nm=48,Nz=96`` KBM nested-GMRES audit exceeded safe
  local memory before returning a physical outer residual. All probe processes
  were terminated and the prototype was removed. No solver claim or default is
  changed; any revisit must run damping and candidate modes under supervised
  office resources and retain the unchanged matrix-free outer residual gate.

- 2026-07-13: Completed the supervised office A/B audit for that low-moment
  correction at the full tracked resolutions. For KBM at ``Nl=16, Nm=48,
  Nz=96`` and the reference-aligned shift ``0.21944-1.14065i``, damping returned
  residual ``0.790`` in 66.6 s while the correction worsened it to ``0.976`` in
  106.2 s and selected a different damped branch. For ETG at ``Nl=24, Nm=8,
  Nz=96`` and shift ``+14.94i``, damping returned ``0.954`` in 30.6 s; the
  correction reached ``0.850`` in 45.2 s but changed the branch and remained
  far above the unchanged ``0.1`` gate. Replacing JAX's inner GMRES with SOLVAX
  GMRES was also insufficient: ETG returned residual ``0.806`` in 38.3 s and
  KBM ``0.744`` in 84.2 s, both on different branches. No experimental solver
  code is promoted. The next admissible design must improve full physical
  residual, branch identity, and runtime together rather than only the nested
  linear-solve diagnostic.

- 2026-07-13: Consolidated the final standalone W7-X zonal plotting commands
  into their scientific owners. ``build_w7x_zonal_validation_artifacts.py``
  now exposes ``response-panel``, ``contract``, and ``state-convention``;
  ``build_w7x_zonal_recurrence_artifacts.py`` exposes ``moment-tail``,
  ``closure-ladder``, and ``sweep``. All replay manifests and documentation use
  those subcommands, 13 focused artifact/campaign tests pass, and tool inventory
  falls from 110 to 108 against the unchanged 99-file target.

- 2026-07-13: Consolidated imported-geometry linear comparison execution under
  ``compare_gx_imported_linear.py`` with explicit ``fields``, ``growth-dump``,
  and ``window`` modes. Shared binary readers preserve the saved-state
  ``(species, Laguerre, Hermite, ky, kx, z)`` layout without a circular tool
  import; benchmark manifests and campaign launchers now name their mode. The
  complete comparison-tool suite passes 115 tests with one expected skip, and
  tool inventory falls from 108 to 106.

- 2026-07-13: Consolidated term-resolved RHS dump generation and comparison
  under ``compare_gx_rhs_terms.py`` with explicit ``write`` and ``compare``
  modes. The shared owner preserves Cyclone, ETG, KBM, kinetic-electron, and TEM
  initialization policies while keeping binary-state comparison helpers
  importable by exact-state tooling. The complete 115-test comparison suite
  remains green with one expected skip, and tool inventory falls to 105.

- 2026-07-13: Consolidated nonlinear transport-window diagnostics and
  term-resolved state comparison under ``compare_gx_nonlinear.py`` with
  explicit ``diagnostics`` and ``terms`` modes. The merged owner keeps
  publication-window gates separate from equation-level bracket diagnostics,
  while sharing runtime geometry and normalization imports. The full comparison
  suite remains at 115 passed with one expected skip; tool inventory is 104.

- 2026-07-13: Consolidated profiling-manifest validation into
  ``check_parallel_scaling_artifacts.py --performance-manifest-only``. The
  unified release owner now validates both the declared CPU/GPU optimization
  campaign and its tracked scaling evidence while retaining independently
  callable validation functions for focused tests. CI and release readiness use
  the same command, and tool inventory falls to 103.

- 2026-07-13: Moved the differentiable-refactor manifest gate under
  ``check_package_architecture_manifest.py differentiable-refactor``. Source
  topology, hotspot ownership, public-facade, test, documentation, parity, and
  autodiff contracts now share one architecture-policy owner and are all run by
  CI/release readiness. Focused APIs remain independently testable; tool
  inventory falls to 102.

- 2026-07-13: Consolidated nonlinear-window convergence, replicate readiness,
  and ensemble robustness under ``check_nonlinear_transport_gates.py``. The
  explicit ``convergence``, ``readiness``, and ``ensemble`` subcommands now sit
  beside target-time, runtime-output, matrix-progress, and portfolio admission
  checks, removing an implicit legacy command mode while preserving the same
  statistical and fail-closed physics gates. Tool inventory falls to 101.

- 2026-07-13: Moved nonlinear calibration-input provenance auditing into
  ``check_quasilinear_promotion_guardrails.py calibration-inputs``. External
  VMEC admission policy, nonlinear-gate matching, negative-evidence reporting,
  and publication plots remain reusable by artifact builders, while calibration
  and claim promotion now have one release owner. Tool inventory falls to 100.

- 2026-07-13: Consolidated external-VMEC high-grid holdout admission under
  ``check_vmec_boozer_gates.py high-grid-admission``. The same fail-closed
  coarse-grid exclusion, retained-grid convergence, time-horizon, replicate,
  and claim-scope gates remain directly tested beside differentiability and
  held-out geometry gates. Tool inventory reaches the enforced target of 99.

- 2026-07-13: Kept the consolidated quasilinear promotion guardrail usable in
  dependency-minimal repository-hygiene jobs by moving NumPy/Matplotlib imports
  into calibration serialization and plotting functions. Metadata-only release
  checks therefore remain lightweight, while docs jobs retain the same optional
  publication-plot path.

- 2026-07-13: Applied the same dependency boundary to the consolidated
  VMEC/Boozer owner: validation-gate imports now occur only when the scientific
  ``high-grid-admission`` payload is built. Metadata-only differentiability and
  portfolio checks therefore run in the pre-install repository-hygiene job
  without importing JAX.

- 2026-07-13: Re-probed both office comparison binaries under explicit local
  CUDA/cuBLAS, cuTENSOR, NCCL, HDF5, and GSL library roots. Linkage resolves and
  the Cyclone s-alpha input reaches geometry initialization, but the
  clean-revision binary fails a parallel-NetCDF operation on a serial-opened
  file and the instrumented binary aborts on an HDF5 1.10.7/1.14.5 mismatch.
  The capability matrix now records these exact runtime blockers rather than
  the older missing-library shorthand; no new comparison output is admitted.

- 2026-07-13: Admitted the published SOLVAX 0.7.3 release after its four
  consumed interfaces and the 228-test structured-solver, IMEX, linear and
  nonlinear helper, implicit-objective, and differentiable-geometry suite
  passed on the current JAX stack. The dependency contract is now
  ``solvax>=0.7.3,<0.8`` and has an executable interface/version gate. The
  audit also found and fixed one SPECTRAX-GK facade regression: the canonical
  ``CollisionInvariantRates`` record had not been re-exported from the linear
  term facade after consolidation.

- 2026-07-13: Rebuilt the exact comparison revision ``bc2fe552`` in an isolated
  office tree against one consistent OpenMPI 4.1.6, parallel netCDF 4.9.2, and
  HDF5 1.14.5 stack. The canonical Cyclone s-alpha run completed 2,145 steps to
  ``t=10`` in 23.1 seconds and produced valid netCDF/restart files. This audit
  also found a SPECTRAX-GK publication-driver bug: an early automatic fit chose
  ``t=5.068--5.384`` and reported ``gamma=0.0413`` while the trajectory was
  still selecting its asymptotic mode. The corrected explicit ``t=7--10`` fit
  gives ``(gamma, omega)=(0.095109, 0.297253)`` at ``ky=0.3``, within 2.24% and
  5.41% of the tracked reference. A regression now keeps this benchmark on the
  measured terminal window.

- 2026-07-13: Removed duplicate velocity and field-line algebra from
  ``terms/operators.py`` and ``operators/linear/moments.py``. Spatial/linked
  derivatives, zero-padded shifts, Hermite/Laguerre multiplication, and the
  precomputed-coefficient ``streaming_ladder_term`` now have one owner in
  ``operators.linear.streaming``; the moment module retains field coupling and
  a small user-facing streaming adapter. The tranche removes 58 net lines,
  keeps the source-file target unchanged, and passes 172 selected operator,
  linear, and parallel tests plus strict architecture and coverage-manifest
  ownership gates.

- 2026-07-13: Falsified recycled GCROT as a drop-in KBM shift-invert repair.
  On the reduced physical ``Nl=8, Nm=24, Nz=96`` problem at the tracked shift,
  JAX GMRES returned residual ``0.974`` in 66.6 s. SOLVAX GCROT reduced the
  neutral-selection residual to ``0.799`` in 22.4 s but selected a strongly
  damped wrong branch; the production KBM frequency masks returned residual
  ``0.992`` on another wrong branch. No mutable recycling or solver-specific
  complexity is admitted. The validated time-integrated KBM path remains the
  release path.

- 2026-07-13: Fixed a direct-import cycle exposed by the collision audit after
  streaming consolidation. The ``operators`` and ``operators.linear`` facades
  now resolve public names lazily, allowing the canonical collision module to
  import independently while removing 17 net source lines. The finite-Larmor-
  radius gate now checks the exact Mandell et al. equation (4.10) free-energy
  rate and measured first-order convergence to the independent drift-kinetic
  equation, in addition to the existing equation, invariant, full-f, and AD
  gates.

- 2026-07-13: Consolidated four duplicated single/multi-field, real/full FFT
  nonlinear bracket implementations behind two shape-aware numerical cores.
  The public/internal call surfaces and operation order remain unchanged, the
  complete focused bracket/RHS shard passes, and a same-process realistic
  ``1x4x8x64x96x24`` CPU A/B is bitwise identical with medians ``31.721 ms``
  before and ``31.671 ms`` after. This removes 85 source lines without making
  a speedup claim. A direct single-to-multi delegation was measured and
  rejected first because it added about 1.6% overhead.

- 2026-07-13: Re-profiled the fused benchmark-size Cyclone Miller RHS from a
  clean office checkout at exact commit ``694fbc42`` with JAX 0.6.2. The valid
  idle-A4000 grid-mode row measured ``6.60 ms`` bracket, ``6.78 ms`` linear
  RHS, and ``13.16 ms`` full RHS with unchanged finite norms. Its HLO summary
  has 2,403 lines, 1,016 reshapes, and 1,439 broadcasts, versus 3,336, 1,545,
  and 1,822 in the older tracked trace. A following spectral-mode run overlapped
  unrelated two-GPU work at 100% utilization and is rejected; no mixed-load
  performance artifact or panel update is admitted.

- 2026-07-13: Repaired the CI fallout from linear-streaming consolidation.
  Both quick-test and terms-coverage shards now reference the canonical
  ``test_linear_streaming.py`` owner, and the zero-weight benchmark guard
  patches ``streaming_ladder_term`` rather than the removed adapter. The
  release path contract plus focused streaming regressions pass 19/19.

- 2026-07-13: Extended the Francisquez full-``f`` primitive-target gate beyond
  the two-species scalar derivation check. A three-species, multi-sample matrix
  now verifies pairwise momentum and energy conservation for ``d_v=1,2,3`` and
  Galilean invariance, while the documentation continues to block any claim
  that these targets constitute a complete distribution-space collision
  operator. All 19 focused collision integration tests pass.

- 2026-07-13: Tested an explicit layout-normalized compressed-``ky`` FFT route
  on the benchmark-size nonlinear bracket before changing production code. It
  was CPU timing-neutral (``30.75`` versus ``30.90 ms``) and failed numerical
  identity with relative norm error ``0.474``, so it is rejected. Both office
  GPUs were concurrently occupied at 100% by unrelated work; no contaminated
  GPU row is admitted. The clean A4000 profile at ``694fbc42`` remains the
  current hot-path evidence, and no new speedup claim is made.

- 2026-07-13: Re-ran the bounded release-equivalent surface after the streaming,
  collision, and bracket changes. Both architecture policies, release readiness,
  Ruff, strict Sphinx, wheel/sdist construction, and Twine metadata pass. The
  source/test/tool inventories are ``226/96/99``; the plan's stale 129-tool row
  was corrected. GitHub CI for the final checkpoint is pending, not assumed.

- 2026-07-13: Completed a two-domain source-ownership migration without
  compatibility shims. Linear collision/hypercollision/hyperdiffusion/end-
  damping kernels moved from ``terms.linear_dissipation`` to
  ``operators.linear.dissipation`` and pseudo-spectral Poisson brackets moved
  from ``terms.brackets`` to ``operators.nonlinear.brackets``. All internal,
  test, tool, manifest, capability-matrix, and API references use the canonical
  paths; 146 affected operator/assembly/benchmark tests pass in bounded shards.

- 2026-07-13: Added the differentiable foundation for equilibrium
  :math:`E\times B` flow shear without exposing an incomplete runtime option.
  The shearing-coordinate kernel follows ``kx*(t)=kx(0)-ky*gamma_E*t``, applies
  nearest-cell non-circular remaps and two-thirds-band loss, and returns the
  residual real-space phase. Zero-shear, analytic trajectory, inverse-remap,
  de-alias-boundary, and ``gamma_E``/radial-scale AD-versus-FD gates pass. Full
  runtime admission remains blocked until all kx-dependent linear,
  gyroaverage, field-solve, linked-boundary, and nonlinear quantities are
  updated and linear-suppression/nonlinear-transport comparison gates pass.

- 2026-07-13: Added one fail-closed owner for periodic sheared-``kx`` cache
  updates. It rebuilds perpendicular metrics, drift frequencies, gyroaverages,
  Bessel tables, field-solve inputs, bracket multipliers, and hyperdiffusion
  from a two-dimensional effective ``kx`` grid. Zero-shear cache arrays and the
  assembled linear RHS reproduce the static path, while a nonzero-shear JVP
  agrees with the stable centered-FD plateau. Linked and non-twist boundaries
  intentionally raise until their boundary phase and integration policy land.

- 2026-07-13: Completed the first end-to-end periodic flow-shear trajectory
  foundation without exposing it in TOML. The full-complex bracket now applies
  and removes the residual radial phase between split ``kx``/``ky`` transforms;
  a canonical-coordinate gate shows that this preserves the Poisson bracket.
  The fixed-step midpoint RK2 path remaps stage states and derivatives into the
  correct time-dependent basis, passes zero-shear trajectory identity, and
  reproduces exact cumulative full-step remapping. Compressed-real,
  linked/non-twist, adaptive, and IMEX paths fail closed or remain absent, so no
  production flow-shear or transport claim is made.

- 2026-07-13: Verified the sheared midpoint route beyond identity tests. A
  physical drift/diamagnetic trajectory converges at observed orders ``2.02``
  and ``2.07``. In a compact Cyclone-like linear ITG pilot, both zero and strong
  shear final-potential norms change by less than 1% when ``dt`` is halved from
  ``0.02`` to ``0.01``; ``gamma_E=1`` lowers the refined final norm by more than
  20%. This literature-consistent suppression direction is admitted only as a
  linear pilot, not as saturated-transport evidence.

- 2026-07-13: Added canonical sampled transport to the periodic flow-shear
  trajectory without retaining the kinetic-state history. The trace uses the
  production flux-surface weights, instantaneous sheared cache, and per-species
  gyro-Bohm heat-flux kernel. A separate JAX-native field-solve policy now makes
  the complete RK2/remap/cache/field/transport objective differentiable in both
  forward and reverse mode; JVP and gradient agree with each other and with the
  converged centered finite-difference plateau. The faster custom-VJP policy is
  numerically identical. This closes transport instrumentation and derivative
  plumbing, but not the required saturated-transport or matched-comparison gate.

- 2026-07-13: Added the stage-basis-correct three-stage Heun RK3 method to the
  periodic sheared integrator. Every intermediate state is advanced into its
  instantaneous shearing basis and every derivative is returned to the step
  basis before the Runge--Kutta combination. Zero-shear trajectory identity
  with the production RK3 route and third-order convergence on the physical
  drift/diamagnetic RHS pass. This closes a higher-order fixed-step numerical
  route for long-window research campaigns; it does not admit an unconverged
  low-resolution transport trace or expose equilibrium flow shear in TOML.

- 2026-07-13: Added an explicit state-only policy to the sheared integrator,
  matching the established nonlinear API. ``return_fields=False`` skips the
  endpoint field/RHS evaluation and field-history output at every step while
  preserving the final trajectory exactly. The default field-returning route
  is unchanged, and transport traces continue to evaluate endpoint fields
  because the canonical heat flux requires them.

- 2026-07-13: Reused the production nonlinear CFL policy in the differentiable
  sheared transport scan. Adaptive runs carry physical time and accepted step
  size through JAX, combining the linear-frequency bound with instantaneous
  pseudo-spectral ExB frequencies. A nonlinear-CFL-dominated test reduces the
  step below its cap and its final-time JVP agrees with centered finite
  differences to ``1.9e-5`` relative error. Fixed-step behavior remains the
  default and is numerically unchanged.

- 2026-07-13: Removed a redundant RHS/field solve from every field-returning or
  transport sheared step by carrying each endpoint derivative and fields into
  the next accepted step at the same physical time. The state-only path remains
  separate and does not compute endpoint fields. All trajectory, transport,
  observed-order, and AD gates remain unchanged; a warm 100-step CPU pilot
  decreased from ``1.391`` to ``1.314 s`` (5.5%), which is recorded as local
  implementation evidence rather than a release-level speedup claim.

- 2026-07-13: Rejected the first long fixed-step transport campaign rather than
  treating pre-saturation values as evidence. On local CPU, ``24x24x24`` with
  ``Nl=4``, ``Nm=8``, Heun RK3, and ``dt=0.02`` first becomes non-finite at
  ``t=134.96``. On clean office A4000 GPUs, the full ``64x64x24`` baseline and
  ``gamma_E=0.01`` runs become non-finite at ``t=93.00`` and ``t=98.98``.
  Timestep refinement had already failed at a similar physical time, so these
  are recorded as nonlinear-CFL failures and are excluded from transport gates.

- 2026-07-13: The production nonlinear CFL policy reduced 17,000 full-grid GPU
  steps into ``t=92.98`` (baseline) and ``t=102.92`` (``gamma_E=0.01``), but
  both still became non-finite. This falsified the timestep-only hypothesis.
  A state-level audit then found that the full-complex sheared path accumulated
  a 3.15% Hermitian defect by ``t=5`` even from an exactly physical initial
  state, while the production real-FFT layout enforces this constraint by
  construction. Projecting every remap and RK stage now keeps the defect at
  machine zero and restores zero-shear identity with the compressed-real path.
  All pre-fix long traces remain rejected; the corrected long campaign must be
  rerun before any saturated-transport claim.

- 2026-07-13: Added restartable physical-time chunks to the compact sheared
  transport API. ``initial_time`` preserves the absolute shearing coordinate
  and ``initial_dt`` preserves adaptive-CFL history. A two-chunk run reproduces
  a single scan's state, accepted times, and canonical heat flux, enabling
  bounded long campaigns without discarding completed GPU work.

- 2026-07-13: Added one reusable matched nonlinear-transport acceptance gate
  rather than embedding flow-shear decisions in a campaign script. It requires
  both post-transient windows to pass their finite, stationarity, block, and
  uncertainty gates before computing a relative reduction and quadrature-SEM
  separation. A synthetic resolved reduction passes while a drifting treatment
  fails closed; the wider 58-test transport-window tranche and focused MyPy
  checks pass. This is diagnostic infrastructure, not a flow-shear promotion.

- 2026-07-13: The first clean x64 office relaunch failed before stepping because
  the initial endpoint derivative entered the JAX scan as ``complex64`` while
  the requested state carry was ``complex128``. The endpoint policy now casts
  the initial derivative exactly as it already cast subsequent derivatives; an
  adaptive-RK3 x64 regression preserves the final-state dtype. A stale linear
  suppression test was also corrected from a shear that removed its sole mode
  beyond the retained two-thirds band to a resolved ``gamma_E=0.5`` point; the
  latter is timestep-converged to 0.4% and suppresses amplitude by 22%.

- 2026-07-13: Routed the matched-window API through the existing grouped
  nonlinear transport release executable as ``matched-windows``. The command
  consumes two convergence-report JSON files, writes the complete paired gate,
  and exits nonzero when either source or the requested effect threshold fails.
  This adds no new tool file; 22 transport release/contract tests pass.

- 2026-07-13: Fixed the resulting CI architecture failure rather than raising
  the complexity allowance. The paired treatment decision moved from the
  1,055-line transport-statistics owner into the existing validation-gate
  domain, leaving those modules at 943 and 642 lines. The architecture gate
  again passes with zero exceptions and unchanged ``226/96/99`` source, test,
  and tool inventories; the 57-test core/release/transport tranche passes.

- 2026-07-13: Tightened the equilibrium-flow-shear comparison scope from the
  actual comparison-executable source. Its production path applies continuous
  ``kx*`` geometry updates, nearest-cell remaps, and the residual FFT phase. A
  separately labeled ``m=1`` flow-shear expression is accumulated only by the
  term-resolved diagnostic kernel; in the production linear kernel the same
  expression is not assigned to the RHS. The current matched campaign therefore
  validates perpendicular decorrelation only. Parallel-velocity-gradient and
  toroidal-rotation physics remain a separate unimplemented lane rather than an
  implicit part of ``gamma_E``.

- 2026-07-13: Anchored that scope and numerical convention to primary
  literature. Schekochihin--Highcock--Cowley and Ball--Brunner--McMillan
  distinguish perpendicular decorrelation from parallel-velocity-gradient
  drive, while McMillan--Ball--Brunner show why the residual sub-cell Fourier
  phase must be retained rather than rounding the nonlinear coupling to an
  integer wavevector. The documentation now cites all three results directly;
  no broader rotation-physics claim was added.

- 2026-07-13: Rejected a 6,000-step external checkpoint interval after both
  full-grid x64 A4000 traces reached the 600-second segment ceiling before
  writing output. The replacement keeps the identical ``64x64x24``, ``Nl=4``,
  ``Nm=8``, periodic ``x0=y0=28.2``, adaptive-RK3 contract but checkpoints every
  4,000 accepted steps and reuses the compiled executable for three chunks.
  This is campaign execution metadata only; no transport result is admitted
  until the combined post-transient windows and matched comparison pass.

- 2026-07-13: Halved the package-wide coverage matrix from 48 to 24
  cost-balanced shards. The five known compile-heavy test owners remain isolated;
  every other shard contains four or five files under the unchanged 300-second
  test timeout. This removes 24 duplicate checkout/environment/install sequences,
  cuts action-download pressure, and reduces the matrix from six to three waves
  at ``max-parallel=8`` without weakening the combined 95% package gate. A
  workflow contract now keeps the matrix cardinality, test command, label, and
  coverage-combine command synchronized.

- 2026-07-13: Isolated ``test_nonlinear.py`` as the fifth compile-heavy wide-
  coverage owner after its former mixed shard completed all tests at 294.9 s but
  crossed the 300-second process cap while writing coverage. The replacement
  mixed shard passes 141 tests in 23.1 s and the isolated nonlinear shard passes
  24 tests in 115.4 s. The current CI run has all quick, docs/package, MyPy,
  fast-coverage, and 24 wide shards green while the aggregate coverage job
  combines their data.

- 2026-07-13: Closed the periodic research path's internal saturated-transport
  gate at full ``64x64x24``, ``Nl=4``, ``Nm=8`` resolution with adaptive Heun
  RK3 and x64 precision. Independently accepted ``t=[240,300]`` windows give
  ``Q_i=10.5009 +/- 0.0949`` without shear and ``9.8603 +/- 0.0569`` at
  ``gamma_E=0.01``: a ``6.10%`` reduction separated by ``5.79`` combined SEMs.
  Starts from ``t=240`` through ``280`` retain a positive ``4.46--6.28%``
  reduction. The model remains unshipped pending the matched external response,
  linked-boundary, compressed-real, and IMEX gates.

- 2026-07-13: Completed one clean matched ``t=300`` comparison pair from
  identical initial states on the two office A4000 GPUs rather than attempting
  another unsafe restart from output that does not contain the distribution
  state. Both ``t=[240,300]`` windows pass the independent stationarity and
  uncertainty gates. Their heat fluxes are ``5.9963 +/- 0.0321`` without shear
  and ``6.0014 +/- 0.0416`` with shear, a ``-0.084%`` reduction separated by
  only ``-0.10`` combined SEM. This conflicts with the internal ``6.10%``
  reduction and is retained as negative evidence: flow shear remains unshipped,
  and the next research action is a state-level full-complex/compressed-real
  treatment-delta audit at remap boundaries, not another long transport run.

- 2026-07-13: The first 24-shard CI run passed every test, documentation,
  package, typing, and coverage shard but exposed a single-device coverage blind
  spot in the real species/Hermite collective kernels, leaving the aggregate at
  94%. The parallel owner is now isolated and supplements its ordinary shard
  with two bounded identity tests in fresh four-logical-CPU subprocesses. A
  reconstructed combination of the unchanged CI data and these supplements
  reports ``28,093`` statements with ``1,271`` missed (``95.48%``), while each
  supplemental process completes in under 20 seconds without the out-of-memory
  behavior of running the complete file with four devices.

- 2026-07-13: Tightened the existing compressed-real/full-complex bracket test
  from a finiteness check on a nonphysical random spectrum to numerical identity
  on a real, Hermitian, two-thirds-dealiased field. The kernels agree to
  ``1.4e-7`` relative error, and the same gate passes immediately after an exact
  integer remap where the residual phase is unity. This rules out a generic FFT
  representation error at the remap itself; fractional-phase ordering remains
  the next state-level localization target, and no production option was added.

- 2026-07-13: Closed two numerical-correctness defects in the experimental
  shift-invert path without promoting its unresolved KBM branch. Arnoldi now
  rejects dtype-scale residual directions instead of normalizing roundoff, and
  each preconditioned GMRES result is checked in the original shifted physical
  system before an unpreconditioned retry. On a reduced physical KBM operator,
  this lowers the falsely converged shifted-solve residual from ``4.76`` to
  ``1.14e-5``; a restart from the dense-reference mode recovers that mode with
  ``3.18e-6`` residual and unit overlap for all preconditioner settings. The
  full ``Nl=16``, ``Nm=48``, ``Nz=96`` canonical audit still fails closed at
  ``0.981`` outer residual versus the ``0.1`` gate. A fixed-step continuation
  attempt overflowed and was rejected, while the tracked KBM branch table was
  confirmed to use its separate controlled adaptive time-integration workflow.
  The next structured-solver action is therefore a stable physical continuation
  seed or field-coupled complex preconditioner, not more ungated parameter
  searches; time integration remains the release path.

- 2026-07-13: Audited the canonical KBM TOML as an actual user-facing office
  GPU run. Neither the generic propagator default nor a full 4,000-step adaptive
  RK4 attempt is admissible: the latter overflowed yet the fit layer reported
  finite but meaningless ``gamma=373.3`` and ``omega=1135.7``. Runtime fitting
  now rejects any non-finite time, field, or density history before analysis,
  and KBM-normalized Krylov requests receive the targeted KBM policy so the
  existing physical residual gate raises instead of returning the generic
  propagator result. Documentation no longer presents this research TOML as a
  standalone smoke run; the lightweight reviewed KBM comparison driver remains
  the reproducible promoted entry point.

- 2026-07-13: The complete GitHub Actions run ``29261912348`` passed repository
  hygiene, MyPy, documentation/package checks, all quick-test groups, all 24
  bounded wide-coverage shards, and the aggregate 95% package gate. The local
  release-equivalent rerun also passed 93 release tests, strict Sphinx, Ruff,
  focused MyPy, and the 47.999 MB repository-size manifest.

- 2026-07-13: Fixed the immediate architecture failure in CI run
  ``29263875566`` without adding an exception. Non-finite trajectory validation
  now lives in the runtime-diagnostics owner that performs growth fitting,
  rather than pushing ``workflows/linear.py`` above its 1,000-line budget. The
  architecture manifest again reports ``226/96/99`` source/test/tool files,
  zero complexity exceptions, and all topology targets met.

- 2026-07-14: CI run ``29264009360`` exposed a selection-policy defect made
  visible by the corrected shifted solve. A nearest-shift streaming mode passed
  its physical residual gate (``0.0243 < 0.06``) but was rejected because the
  growth floor was applied unconditionally. The floor now applies only to
  growth-selected searches; nearest, overlap, and explicit-shift selections may
  admit stable modes when finite and residual-converged. Distinct errors report
  residual, growth-floor, and non-finite failures. The exact CI environment
  passes both physical shift-invert tests, and the complete former shard 8
  passes ``95`` tests in ``144.69 s`` under the unchanged 300-second cap.

- 2026-07-14: Bounded full-resolution KBM continuation probes ruled out the
  current time trajectory as a Krylov seed. Fixed RK4 at ``dt=1e-3`` is already
  numerically unstable by ``t=0.2`` (state norm ``4.37e28`` and spurious fitted
  growth ``613.7``). IMEX2 at ``dt=1e-4`` keeps the state norm bounded through
  ``t=0.1`` but remains transient-dominated (fitted growth ``33.36``), and its
  seeded GPU Arnoldi solve terminates in cuSolver rather than producing a
  residual-qualified pair. No artifact or option was promoted. The next solver
  action is a field-coupled complex preconditioner or better-conditioned
  physical shifted solve, not additional time-seed sweeps.

- 2026-07-14: Closed the fractional flow-shear Poisson-bracket representation
  audit without promoting the model. For physical Hermitian, dealiased states,
  the common residual radial phase cancels from the bracket and the canonical
  compressed-real route agrees with the split-phase full-complex route within
  ``2e-5``. A JAX tangent agrees with centered finite differences, and a
  three-step RK3 physical integration reproduces both the final state and heat-
  flux trace. The matched ``-0.084%`` versus internal ``6.10%`` response cannot
  be attributed to the bracket representation; the next audit is now confined
  to sheared linear/cache, field-solve, and diagnostic conventions. Linked-
  boundary and IMEX support remain closed, and no input-file option was added.

- 2026-07-14: Audited the remaining flow-shear response discrepancy against the
  comparison source at revision ``bc2fe552``. Its production ``m=1`` shear
  expression is a discarded statement and therefore does not broaden the
  perpendicular-decorrelation model. The material difference is temporal: its
  adaptive RK3 path shifts geometry once with the previous ``dt`` before the
  next CFL selection and holds that basis fixed through every RK stage, whereas
  SPECTRAX-GK advances accepted physical time and evaluates exact stage bases.
  The prior stationary ``t=300`` pair is retained as negative cross-
  discretization evidence but is no longer called a model-identical parity
  failure. The next bounded comparison is a fixed-dt refinement of the shear
  response; no further long adaptive campaign is justified until that trend is
  established.

- 2026-07-14: Completed the bounded fixed-step response refinement before
  launching any further long campaign. A deterministic reduced periodic
  Cyclone-like pair at ``gamma_E=0.5`` gives terminal ``Phi2`` treatment ratios
  ``0.64032``, ``0.64051``, and ``0.64060`` at ``dt=0.02``, ``0.01``, and
  ``0.005``. Thus the external implementation is active and its short response
  is stable to roughly ``0.05%`` across this refinement. Startup heat-flux
  ratios ``0.5079--0.5142`` are explicitly not admitted as saturated transport.
  Raw office outputs were removed after harvesting these scalar diagnostics.
  The remaining promotion gate is one full-resolution stationary fixed-step
  weak-shear window, followed by linked-boundary and IMEX identity; it is not a
  release blocker because the option remains unavailable to input files.

- 2026-07-14: Reproduced the unresolved canonical full-resolution KBM
  shift-invert gate locally in ``133.04`` seconds at ``Nl=16``, ``Nm=48``, and
  ``Nz=96``: the finite baseline remains ``0.980704`` versus the unchanged
  ``0.1`` physical residual threshold. A physical Rayleigh--Ritz projection in
  the shift-focused subspace returned a non-finite pair and raised maximum RSS
  from ``0.87`` to ``1.18`` GB; a standards-style zero initial guess for the
  unpreconditioned retry also returned a non-finite pair in ``132.52`` seconds.
  Both experiments were removed. The promoted time-integrated KBM branch gate
  remains passed, while the experimental eigensolver stays fail-closed pending
  a genuinely branch-preserving thick restart or complex field-coupled
  preconditioner.

- 2026-07-14: Removed ambiguity among the three tracked KBM extraction
  snapshots. Publication-table generation and the standalone comparison plot
  now consume only ``selected`` rows from the continuity-first candidate table;
  the pointwise mismatch and low-``k_y`` checkpoint remain historical inputs,
  not alternate release-facing branches. A focused regression proves rejected
  candidates cannot replace the selected branch.

- 2026-07-14: Fixed CI run ``29325096544`` after the x64 linear-core shard
  exposed a test-only tangent dtype mismatch in the new collision JVP gate.
  The collision frequency primal is explicitly ``float32`` and its tangent now
  inherits that dtype instead of becoming ``float64`` when x64 is enabled. The
  exact seven-file CI shard passes all ``111`` tests locally with the CI x64
  environment.

- 2026-07-14: Closed the linked-standard-boundary and fixed-step IMEX
  implementation gates for equilibrium ExB flow shear. Shearing coordinates
  now use the cache-normalized radial grid, preserving the small fixed-aspect
  rescaling applied while constructing linked twist-and-shift chains. Linked
  RK2/RK3 trajectories are exactly identical to the established integrator at
  zero shear; nonzero-shear link spacing remains invariant and the cache JVP
  matches centered finite differences. The new first-order sheared IMEX route
  evaluates explicit nonlinear forcing in the current basis, remaps the right-
  hand side and warm start, rebuilds ``I-dt L`` at the endpoint, and solves it
  through the shared SOLVAX implicit derivative. It passes linked zero-shear
  multistep identity, physical first-order convergence, endpoint heat-flux, and
  JVP/VJP finite-difference gates. Adaptive sheared IMEX, non-twist geometry,
  and custom collision operators remain fail-closed. Flow shear remains absent
  from input files until the full-resolution fixed-step matched-response window
  is closed.

- 2026-07-14: The first clean full-grid A4000/x64 sheared-IMEX probe found a
  dtype-policy bug before taking a timestep: a runtime-built ``complex64``
  initial state entered an implicit operator whose x64 state was
  ``complex128``. The sheared scan now follows the established implicit-solver
  precision policy and casts the endpoint warm start and right-hand side to the
  operator dtype. A dedicated x64 regression starts from ``complex64``, returns
  ``complex128``, and preserves finite transport; the forward/reverse transport
  gradient gate remains passed.

- 2026-07-14: Started the final full-resolution fixed-step flow-shear response
  gate on two office A4000 GPUs from the same ``64x64x24``, ``Nl=4``, ``Nm=8``,
  periodic Cyclone state. The baseline and ``gamma_E=0.01`` runs use x64
  sheared IMEX with ``dt=0.02`` and independent restart checkpoints. Both first
  1,000-step chunks reached ``t=20`` in about 143 seconds, remained finite, and
  gave startup ion heat fluxes ``2.467e-4`` and ``2.449e-4``. These startup
  values are recorded only as stability evidence. Promotion still requires
  both independent ``t=[240,300]`` windows to pass the predeclared stationarity
  and uncertainty gates before applying the 5% reduction and two-combined-SEM
  treatment gate. The testing guide and verification matrix now enumerate the
  linked, IMEX, derivative, long-window, and executable-exposure boundaries.

- 2026-07-14: Addressed the only failure in CI run ``29328450243`` without
  raising the five-minute limit or dropping coverage. Initial file isolation
  passed locally in ``137.42 s`` but still timed out after 34 of 57 tests on the
  slower x64 hosted runner in run ``29330282613``. The shard runner now collects
  deterministic node IDs and executes designated compile-heavy owners as five
  disjoint, contiguous coverage batches. Their union is exactly all 57 tests;
  the exact x64 route passes locally, while each subprocess retains the 300 s
  cap. A fresh 24-shard run remains the final CI confirmation.

- 2026-07-14: Completed the prospectively gated fixed-step equilibrium-flow-
  shear campaign and rejected physical-model promotion. The ``64x64x24``,
  ``Nl=4``, ``Nm=8``, x64 fixed-IMEX pair reached ``t=300`` without non-finite
  state, but both ``t=[240,300]`` windows failed stationarity; their means are
  ``15.4508 +/- 0.2628`` and ``16.1948 +/- 0.1602``, a 4.82% increase. A clean
  fixed-RK4 comparison at source revision ``bc2fe552`` completed the same
  grid/timestep/window contract. Both comparison windows pass independently and
  give ``11.7154 +/- 0.2157`` and ``14.6236 +/- 0.1407``, a 24.82% increase
  separated by 11.29 combined SEM. Thus neither fixed-step route supports the
  earlier adaptive 6.10% suppression. The numerical research API remains
  validated, but flow shear stays absent from TOML and executable claims. The
  compact JSON/CSV evidence is tracked; 50--63 MB raw states and outputs remain
  off-repository.

- 2026-07-14: Deleted the redundant ``objectives.solver_gradients`` aggregation
  layer. The stable top-level ``spectraxgk`` API now resolves each objective,
  eigenvalue, sampling, VMEC/Boozer, and gradient-report symbol directly from
  its canonical owner; tests import private sampling and state helpers only
  from those owners. The strict architecture target falls from 226 to 225
  source files, validation coverage ownership moves to
  ``objectives.solver_gradient_reports``, and no numerical implementation or
  documented top-level symbol changed.

- 2026-07-14: Consolidated nonlinear Laguerre quadrature and gyroaverage
  kernels into ``core.velocity``, which already owns the Laguerre basis,
  transform construction, and Bessel coefficients. The standalone
  ``terms.gyroaveraging`` module had one production consumer and no public
  symbol, so deleting it lowers the source target from 225 to 224 without
  changing equations, array contractions, precision, or JAX behavior.

- 2026-07-14: Removed the duplicate ``objectives.qa_low_turbulence`` facade
  and its eager package-level re-exports. The stable top-level API now resolves
  QA contracts, model observables, residual gates, optimizer, and artifact
  builders directly from their focused owners. This lowers the source target
  from 224 to 223 files and removes 102 lines of aggregation without changing
  the reduced model or its explicitly scoped validation claims.

- 2026-07-14: Began line-oriented test consolidation without deleting
  scientific gates. Four external-VMEC replicate filename/protocol cases now
  share one parameterized contract covering joint seed/timestep labels, device
  suffixes, and protection against protocol ``dt`` tokens in case slugs. The
  collected behavioral cases increase from three tests to four parameter rows
  while repeated setup and assertions are removed.

- 2026-07-14: Reduced ``objectives.vmec_boozer_line_search`` from 1,000 to 882
  lines without adding another source file. Aggregate training and held-out
  probes now use the same sample/probe option builders instead of three
  argument-forwarding layers, and held-out reduction uses the common reduction
  calculation. Public reports, dependency-injected finite-difference gates,
  fail-closed line-search behavior, and artifact schemas are unchanged; the
  focused x64 objective tests and strict zero-exception architecture gate pass.

- 2026-07-14: Moved
  ``diagnostics.nonlinear_transport_optimization`` below the source ceiling
  (1,000 to 997 lines) by centralizing guard-config validation, artifact-pass
  recognition, matched-audit blocker extraction, and failed-gate extraction.
  All nonlinear transport thresholds, promotion conditions, claim boundaries,
  and report keys remain unchanged; all 13 focused policy tests pass. Only
  ``workflows.linear`` now remains exactly at the 1,000-line ceiling.

- 2026-07-14: Removed the last exact-ceiling source owner by reducing
  ``workflows.linear`` from 1,000 to 977 lines. One-use fit-policy and
  eigenvector-result wrappers are now constructed at their call sites, and the
  finite-growth predicate is expressed directly. Solver routing, runtime
  dependency injection, output contracts, scan policy, and progress callbacks
  are unchanged. Four scan/mode tests and 11 x64 runtime branch tests pass; the
  installable package now has 88,336 lines and no 1,000-line module.

- 2026-07-14: Continued semantic test consolidation in the largest artifact
  owner, reducing it from 6,393 to 6,358 lines while retaining all 126 collected
  cases. Closed-vs-open quasilinear manuscript status now shares one fixture
  whose explicit switch is the presence of model-selection and promotion-
  guardrail evidence. Objective finite-difference/line-search executable
  contracts and independent time-horizon failure modes are named parameter
  rows. The complete owner passes under x64; no physics or release assertion was
  removed.

- 2026-07-14: Reduced ``diagnostics.quasilinear_calibration`` from 997 to 948
  lines by removing a one-use controls object and one-use report builder. The
  public calibration function now validates its four controls and assembles its
  own report, while train/holdout metrics, through-origin scale fitting,
  nonlinear-window convergence, claim levels, metadata, and JSON keys are
  unchanged. All 16 calibration tests and 30 model-selection/promotion tests
  pass under x64; package source is now 88,287 lines.

- 2026-07-14: Reduced the runtime-runner test owner from 4,466 to 4,424 lines
  without dropping a case. RK3's inferred CFL factor and an explicit factor
  override now share one two-row runtime contract instead of duplicating the
  complete nonlinear integrator stub. Both rows and the adjacent nonlinear
  dealias/species-target checks pass under x64; the 96-file suite now has
  94,256 lines.

- 2026-07-14: Reduced ``operators.linear.dissipation`` from 993 to 921 lines by
  deleting four one-use hypercollision forwarding layers and routing inactive,
  constant, and parallel damping directly from the public contribution owner.
  The existing policy objects still group coefficients, masks, and linked-tube
  metadata; all damping equations, the linked ``|k_z|`` FFT, tracer-safe static
  short circuits, and public arguments are unchanged. Fifteen collision
  equation/invariant/derivative tests, two helper tests, and four benchmark
  contracts pass under x64; package source is now 88,215 lines.

- 2026-07-14: Made ``solvers.time.explicit_steps`` the single owner of explicit
  linear Runge--Kutta stage equations used by fixed-step, diagnostic-sampling,
  and species-parallel integration. This removes independent SSPX3 and
  Euler/RK2/RK4 copies from both linear front ends and reduces package source by
  another 65 lines to 88,150. It also fixes an unused first-stage evaluation in
  SSPX3 and K10: SSPX3 now performs its prescribed three RHS evaluations rather
  than four, a 25% kernel-call reduction per step. The new exact call-count gate,
  all low-level explicit-method tests, five observed-order rows, seven
  integration-marked linear solves, and four forced two-device trajectory,
  electromagnetic, collision-invariant, and derivative identity gates pass
  under x64. The matched physical profile described below confirms exact
  compiled-trajectory identity but no measurable end-to-end speedup.

- 2026-07-14: Restored discoverability of the two shortest executable workflows
  by advertising the no-argument self-contained demo and top-level ``--plot``
  route in ``spectraxgk --help``. The full 35-test executable contract passes.
  A clean-directory run of the installed executable completed the 500-step
  Cyclone initial-value demo in 8.15 s on the local CPU, reported live progress,
  recovered ``gamma=0.089982`` and ``omega=0.289838``, and wrote the reproducible
  TOML, summary, time series, eigenfunction, and 2290x921 panel directly to the
  current directory with no auxiliary output folder. This deliberate five-line
  usability addition leaves package source at 88,155 lines.

- 2026-07-14: Reduced the two largest remaining test owners by another 84 lines
  without removing a test or assertion. The stellarator artifact owner now uses
  one QA candidate-comparison fixture and one nonlinear-optimization portfolio
  fixture for authoritative gate, failed reproducibility, rerun admission,
  plotting, and claim-level negative controls; all 126 artifact tests pass in
  7.2 s. The runtime owner now expresses single-mode, Gaussian, and random
  initialization as named rows of one Hermite moment-normalization invariant,
  removing non-benchmark comparison-code terminology; all three x64 rows pass.
  The 96-file suite is now 94,200 lines, with its largest owners at 6,303 and
  4,395 lines.

- 2026-07-14: Repaired and expanded the existing fixed-step integrator benchmark
  rather than adding another profiler. It now imports the canonical Diffrax
  owner, blocks warmups correctly, accepts method/timestep/resolution controls,
  reuses the physical cache, and writes machine-readable timing plus finite
  trajectory fingerprints. A matched ``Nl=7``, ``Nm=14``, 480-step Cyclone
  SSPX3 run at ``dt=0.005`` has exactly equal finite state and field-history
  norms before and after stage consolidation. Median CPU times are 3.202 s and
  3.223 s, so the 0.7% change fails the 3% reportable-speedup gate. The tracked
  artifact and contract test explicitly prohibit a compiled speedup claim:
  XLA had already eliminated the unused result. Performance evidence remains
  honest while the source still benefits from one stage-equation owner and a
  cheaper eager route. The 96-file suite is now 94,215 lines.

- 2026-07-14: Closed the private linear-facade compatibility tranche.
  ``spectraxgk.linear`` now declares 36 supported public operators, cache types,
  parallel kernels, and integrators and exports no underscore-prefixed
  implementation helpers. Kernel tests, profiler scripts, and comparison tools
  import private implementation seams from their canonical owners instead.
  Seven alias-preservation tests were deleted and replaced by one public-boundary
  regression. The complete focused linear/operator tranche, two-device parallel
  contracts, tool contracts, architecture/validation/release gates, and strict
  Sphinx build pass. The package remains at 223 files and 88,156 lines; the
  96-file test suite falls to 94,182 lines.

- 2026-07-14: Applied the same supported-boundary rule to the nonlinear API.
  ``spectraxgk.nonlinear`` falls from 112 to 76 lines and now exposes 28 public
  state-integration, diagnostic, collision/time-step policy, and IMEX symbols
  with no private aliases. The sharded integrator and nonlinear kernel tests
  import projection, collision, CFL, and diagnostic internals from their
  canonical owners. Two facade-preservation tests were replaced by the shared
  public solver-boundary contract. The complete nonlinear and two-device
  sharded tranche, strict documentation build, Ruff, and architecture gates
  pass; package source falls to 88,121 lines.

- 2026-07-14: Removed another 249 lines of repeated test infrastructure while
  preserving every collected case. Manuscript and open-research dashboards now
  share one status writer and one nonlinear performance-evidence fixture; all
  126 artifact rows pass. Runtime geometry tests now share one root-level EIK
  writer across Krylov, explicit-time, generated VMEC/Miller, TOML parity, and
  nonlinear VMEC-EIK/DESC-EIK aliases. Drift normalization, Jacobian,
  ``drhodpsi``, and ``nfp`` remain explicit parameters, and all eight affected
  integration cases pass. The two largest owners fall from 6,303/4,395 to
  6,227/4,222 lines, and the 96-file suite falls to 93,933 lines.

- 2026-07-14: Removed the remaining private aggregation aliases from the
  linear-operator and nonlinear-policy ``__all__`` surfaces. Nonlinear state
  and diagnostic drivers now import projection and collision kernels directly
  from their physical owners; owned CFL and mode-mask helpers remain private
  implementation details. The public-boundary regression now covers all four
  solver/operator surfaces, and the full nonlinear kernel/integration tranche
  passes. Package source falls to 88,099 lines. A call-graph audit retained the
  near-ceiling cache-builder stages because each owns a coherent array-build or
  linked-boundary setup phase rather than compatibility forwarding.

- 2026-07-14: Removed runtime dependency synchronization from the public
  ``objectives.vmec_transport`` facade. The facade falls from 253 to 137 lines
  and now exports only seven supported optimizer-facing names; backend lookup,
  differentiable sample-table kernels, and branch-locality gates are injected
  and tested at their canonical config/table/branch owners. This deletes 116
  package lines and eliminates mutation of implementation-module globals from
  the differentiable objective path. All 145 focused objective, campaign-tool,
  and public-boundary cases pass. Updating the clean local ``vmec_jax`` checkout
  also exposed an upstream API-generation change: current ``main`` uses
  ``VmecInput``/``solve_equilibrium``/``opt.least_squares`` and no longer ships
  the old ``FixedBoundaryVMEC``/``LeastSquaresProblem`` protocol. Migrating the
  QA campaign and examples to that current, substantially simpler protocol is
  the next compatibility and source-reduction tranche; old API emulation will
  not be added to package source.

- 2026-07-14: Migrated the three public QA ITG optimization examples from the
  removed VMEC-JAX ``FixedBoundaryVMEC``/``LeastSquaresProblem`` object model to
  current ``VmecInput``/``solve_equilibrium``/``opt.least_squares`` callbacks.
  They now mirror current upstream ``QA_optimization.py``: the same perturbed
  circular seed, mode-1-through-5 continuation, ``A=6`` and mean-``iota=0.42``
  targets, and one explicit turbulence tuple. The examples fall from 1,173 to
  395 lines. Growth uses the equilibrium implicit Jacobian and differentiable
  eigenvalue path; eigenvector-weighted QL and reduced nonlinear-window
  objectives state and use finite-difference outer Jacobians. The upstream
  12-case turbulence integration suite passes in 54.5 s. A real bounded
  max-mode-1 implicit optimization reduced ``gamma`` from 0.299785 to 0.258962
  and the QA residual from 0.2043 to 0.001381; its four-evaluation CI budget did
  not converge iota and is therefore recorded only as API/derivative evidence.
  Public docs now separate these optimizer objectives from the replicated long-
  window nonlinear promotion workflow.

- 2026-07-14: Migrated the reproducibility-critical VMEC transport metric
  evaluator to current VMEC-JAX. It now parses ``VmecInput``, performs one
  ``solve_equilibrium``, maps normalized surfaces to explicit interior radial
  indices, evaluates every surface/alpha/``k_y`` row through
  ``vmec_jax.core.turbulence.turbulence_objective_vector``, and retains the
  established weighted-reduction/JSON schema. It no longer constructs a hidden
  optimizer stage or calls a private ``_solve_forward`` method. All eight
  focused contracts pass. A real shaped-equilibrium CPU evaluation at
  ``s=0.64``, ``alpha=0``, ``k_y=0.3``, ``Nl=2``, ``Nm=3`` completed in 21 s
  and returned finite raw/log1p growth metrics 0.328296/0.283897, inside the
  literature-anchored unstable-ITG gate used by upstream integration tests.

- 2026-07-14: Replaced the 942-line legacy QA campaign driver with a 301-line
  current-VMEC-JAX owner. It now exposes one explicit protocol for the simple
  QA seed, aspect/mean-iota/quasisymmetry terms, selectable growth/QL/reduced
  nonlinear transport objectives, mode continuation, derivative policy, and
  machine-readable input/WOUT/history outputs. Removed optimizer methods and
  flags from the former ``FixedBoundaryVMEC`` generation are not emulated.
  The public optimization chapter now documents the real
  ``VmecInput``/``solve_equilibrium``/``opt.least_squares`` path and separates
  historical campaign artifacts from current reproduction commands. All 21
  focused driver and admission tests pass, including fail-fast gates for
  invalid radial grids, scales, tolerances, solve budgets, and seed controls.

- 2026-07-14: Removed the public ``ObjectiveTerm`` constructor and
  ``LeastSquaresProblem`` tuple adapter left in ``objectives.vmec_transport``
  after VMEC-JAX removed those classes. The module now only evaluates configured
  transport metrics from solved states and falls from 137 to 56 lines; its
  public export map no longer advertises a callback with the wrong current-API
  signature. The complete 96-case VMEC transport objective owner, package
  architecture, differentiable-refactor, validation-coverage, and 94-case
  release suite pass. Installable source is 87,901 lines and tests are 93,740
  lines.

- 2026-07-14: Replaced the 709-line optimizer-comparison command generator with
  a 420-line current-API campaign contract. Deterministic entries now describe
  the algorithm that actually exists: VMEC-JAX least squares with implicit
  equilibrium derivatives for growth and finite-difference outer Jacobians for
  eigenvector-weighted QL/reduced nonlinear objectives. SPSA, CMA-ES, and
  Bayesian optimization remain separate common-random-number outer-loop
  protocols and cannot be mistaken for VMEC-JAX methods. All 13 campaign-writer
  tests pass, generated baseline/growth/QL/nonlinear commands parse with the
  current driver, and strict Sphinx succeeds. Tool source falls to 91,243 lines
  and tests to 93,669 lines.

- 2026-07-14: Deleted four experimental campaign tools totaling 2,346 lines
  plus their 1,457-line compatibility-test/documentation surface. The guarded
  weight ladder, projected-input writer, private-tape boundary-chain probe, and
  transport-gradient builder all depended on VMEC-JAX symbols removed from
  current ``main`` and could not execute. Their tracked JSON/figures remain as
  historical conditioning and negative-transfer evidence. Current campaigns
  use VMEC-JAX's public implicit equilibrium derivative for growth,
  finite-difference outer Jacobians for eigenvector-weighted objectives,
  independent WOUT replay, and matched nonlinear audits. Architecture now
  enforces 95 tool and 95 test files; tool/test source falls to 88,892/92,526
  lines without removing a current physics gate.

- 2026-07-14: Reduced the largest remaining package policy owner,
  ``diagnostics.nonlinear_transport_optimization``, from 997 to 966 lines by
  folding one-use empty-scope, artifact-map, claim-level, and gate-payload
  wrappers into the report owner. Every safety/promotion threshold, blocker,
  claim boundary, and JSON key is unchanged; all 28 nonlinear optimization and
  release-gate tests pass. Installable source is now 87,870 lines.

- 2026-07-14: Reduced ``operators.linear.cache_builder`` from 988 to 947 lines
  by making the twist-shift resolver return its typed policy directly instead
  of returning a positional ten-element tuple that a one-use wrapper repacked.
  Grid, geometry, drift, gyroaverage, linked-boundary, and moment arrays are
  unchanged. All nine twist/linked/NTFT/sheared-cache branches and the complete
  linear, operator-kernel, and benchmark-contract tranche pass under x64.
  Installable source is now 87,829 lines; no runtime claim is made because the
  numerical work is identical.

- 2026-07-14: Reduced ``workflows.linear`` from 977 to 934 lines by replacing
  its raw four-value trajectory tuples with one typed trajectory and folding
  two one-use runtime/fallback wrappers into the visible solver branch. Fixed,
  Diffrax, explicit-CFL, density-fit, state-return, parallel-forwarding, scan,
  and Krylov-fallback behavior is unchanged. Four scan tests, eleven focused
  integration tests, all architecture/coverage manifests, and the 94-case
  release suite pass under x64. Installable source is now 87,786 lines; the
  numerical kernels and therefore runtime and memory behavior are unchanged.

- 2026-07-14: Reduced ``geometry.autodiff_checks`` from 976 to 888 lines while
  retaining its complete AD/FD validation contract. Input validation now
  creates the typed policy directly, conditioning gates are evaluated by their
  sole report owner, and the JSON builder consumes the existing input,
  derivative, and gate records instead of 22 unpacked arguments. All 34 direct
  geometry and solver-objective derivative tests pass, including forward,
  reverse, chunked, tangent, nonfinite, rank, and conditioning branches. The
  architecture/coverage manifests, 94 release tests, and warning-as-error docs
  build also pass. Installable source is now 87,698 lines.

- 2026-07-14: Reduced ``solvers.linear.integrators`` from 956 to 843 lines by
  removing one-use implicit, explicit, and serial forwarding layers. The public
  integration entry point now exposes the actual custom-collision, implicit,
  parallel, and serial policy directly while retaining the same jitted kernels,
  stage formulas, cache, sampling, donation, and progress behavior. The full
  direct unit-consumer tranche passes, including observed-order, IMEX/implicit,
  sampled diagnostics, custom-collision, serial/parallel identity, and
  parameter-gradient checks. Architecture/coverage manifests and all 94 release
  tests pass. Installable source is now 87,585 lines; no performance claim is
  changed because compiled numerical work is identical.

- 2026-07-14: Removed the remaining linear-diagnostics compatibility bridge.
  ``solvers.linear.integrators`` previously reassigned four globals in
  ``integrator_diagnostics`` on every call so tests could monkeypatch the wrong
  module. It now exports the diagnostics owner's function directly, and the
  tests patch cache, damping, and RHS dependencies at that actual owner. All
  nine diagnostic branches, both runtime density-fit consumers, architecture
  and coverage manifests, and 94 release tests pass. The linear integrator is
  now 803 lines and installable source is 87,545 lines.

- 2026-07-14: Reduced ``diagnostics.nonlinear_replicates`` from 970 to 892
  lines. Finite replicate means, fallback ensemble statistics, extrema labels,
  and relative spread are now computed in one state-diagnostics pass; one-use
  summary/manifest packers and duplicate export/module-string blocks were
  removed. Classification remains explicitly a spread diagnostic while the
  upstream ensemble readiness flag remains separately reported. All 222 direct
  diagnostics, gradient-evidence, optimization, quasilinear-window, and
  artifact consumers pass, as do 109 transport/release gates, all architecture
  and coverage manifests, and strict Sphinx. Installable source is 87,467
  lines. The two 299-line replicate unit owners remain separate because one
  tests package diagnostics and the other tests campaign planning.

- 2026-07-14: Preserved useful Krylov progress when a preconditioned
  shift-invert solve fails its true shifted-system residual check. The
  unpreconditioned retry now starts from the rejected finite iterate instead of
  discarding it and restarting from the right-hand side; a focused regression
  fixes that contract. On the physical KBM audit this reduced the residual from
  the tracked ``0.980704`` baseline to ``0.935747`` at full
  ``Nl=16, Nm=48, Nz=96`` resolution (``134.2 s``, ``0.92 GB`` peak RSS), and
  to ``0.583557`` at ``Nl=8, Nm=24``. Both remain above the unchanged ``0.1``
  acceptance gate, so the experimental eigensolver remains fail-closed and no
  branch claim is promoted. The next bounded solver action is the planned
  field-coupled preconditioner, followed by a branch-preserving thick restart
  only if that physical residual remains inadequate.

- 2026-07-14: Completed the bounded field-coupled/restart discriminator without
  promoting experimental code. One full-operator defect correction reduced the
  physical KBM residual to ``0.379115`` at ``Nl=8,Nm=24`` and ``0.329959`` at
  canonical ``Nl=16,Nm=48,Nz=96``; the latter cost ``149.75 s`` and ``0.96 GB``
  on CPU. A second branch-preserving cycle improved the reduced residual to
  ``0.217062`` but returned ``0.325558`` on the canonical operator in ``69.26 s``
  on an RTX A4000, so reduced-grid convergence did not transfer and the
  candidate was removed. SLEPc's non-Hermitian guidance identifies
  Krylov--Schur with retained basis vectors and harmonic extraction as the next
  technically admissible interior-eigenvalue design. That larger change is
  explicitly separated from the released, residual-qualified time-integrated
  KBM path; no threshold, branch claim, or default was changed.

- 2026-07-14: Removed 39 private implementation re-exports from the nonlinear
  parallelization facade. FFT, bracket, host-staging, device-sharding, and
  transport-observable tests/profilers now import ``spectral_core`` or
  ``device_z``, which actually own those kernels; the facade retains only
  documented plans, reports, strategies, routed operations, and identity
  gates. This lowers the facade from 196 to 157 lines and package source to
  87,434 lines without changing compiled work. All 181 focused parallel,
  profiler, and release tests pass (two hardware-dependent skips).

- 2026-07-14: Fixed the systemic MyPy failure visible in CI runs
  ``29344589199`` through ``29346726628``. Consolidation had left replicate
  extrema labels inferred as non-null strings before a nullable empty-state
  assignment, and the typed linear trajectory allowed pre-materialization
  ``t=None`` at the fitting boundary. Labels are now explicitly nullable, and
  fitting enforces/materializes its time-array invariant before use. The exact
  CI ``mypy`` command passes all 223 source modules, and 150 focused replicate,
  runtime, architecture, and release tests pass locally.

- 2026-07-14: Reduced the largest test owner from 5,867 to 5,782 lines without
  removing a collected case or scientific assertion. Manuscript-readiness,
  open-lane, holdout, zonal, gradient, and profiler expectations now compare
  explicit metric subsets as dictionaries, making each claim boundary visible
  in one place while still failing on missing keys. The duplicate
  pre-manuscript JSON writer was removed in favor of the file's common writer.
  All 119 owner tests and the combined 213-case owner/release tranche pass; the
  95-file suite now contains 92,436 lines.

- 2026-07-14: Re-audited the last generic JAX GMRES owner against published
  SOLVAX 0.7.3 after the warm-retry correction. At matched reduced KBM
  ``Nl=8,Nm=24`` resolution, SOLVAX FGMRES returned the damped
  ``-2.1301+0.8333i`` branch with physical residual ``0.912099`` in ``36.8 s``;
  the current JAX route returns residual ``0.583557`` on the target branch in
  ``38.4 s``. The adapter was never added to source. SOLVAX continues to own
  admitted implicit linear/nonlinear solves, while shift-invert remains on JAX
  GMRES until retained-subspace extraction passes branch, residual, memory,
  and runtime gates together.

- 2026-07-14: Found and fixed an installed-wheel release defect that checkout
  imports could not expose. The ``spectraxgk.parallel`` facade discovered its
  public API by reading sibling ``.py`` files, which fails for zip/non-filesystem
  import loaders. It now uses a static lazy-export registry, with a unit gate
  that compares the registry against every owning module's ``__all__``. CI and
  the PyPI release workflow now install the built wheel into an isolated
  environment and verify the package, parallel facade, plotting help, and both
  ``spectraxgk`` executable spellings before publication. The focused parallel,
  executable, and release tranche passes 177 tests with one expected skip.

- 2026-07-14: Fixed a fail-open edge case in matched nonlinear transport
  promotion. Uncertainty extraction used truth-value fallback, so an explicit
  finite ``0.0`` separation could be replaced by a stale positive statistic.
  Metric precedence now selects the first finite value without treating zero
  as absent, and a regression proves that zero separation blocks promotion.
  Repeated artifact-row decoding and guard-summary plumbing were consolidated
  at the same time, reducing the owner from 966 to 954 lines. All 108 direct
  nonlinear-optimization and release tests pass; thresholds and report schemas
  are unchanged.

- 2026-07-14: Simplified the internal Miller geometry path and fixed its cache
  contract. Three unused profile-assembly parameters, a redundant JAX
  availability probe, the obsolete external-helper executable overrides, and
  a stale "writeout pending" branch were removed. Existing explicit or hashed
  EIK targets are now reused unless ``--force`` is supplied; a regression
  proves reuse skips geometry generation and forced generation replaces the
  target. User documentation now distinguishes the in-package Miller backend
  from optional VMEC helper discovery. The Miller owner fell from 957 to 941
  lines, package source fell to 87,417 lines, and all 258 geometry, executable,
  comparison-tool, and release tests pass with one expected skip. Architecture,
  differentiable-refactor, validation-coverage, typing, and strict-docs checks
  pass without changing geometry equations or comparison tolerances.

- 2026-07-14: Implemented and then removed a backend-portable retained-Ritz
  restart after its physical KBM discriminator failed. The prototype retained
  two or four target-neighbor Ritz vectors, orthonormalized them with JAX QR,
  rebuilt an augmented matrix-free inverse subspace, and preserved the same
  physical Rayleigh quotient and outer-residual gate. The matched
  ``Nl=8,Nm=24`` baseline returned residual ``0.881`` in ``69.4 s``; retained
  dimensions two and four selected damped/opposite-frequency branches with
  residuals ``0.922`` and ``0.991`` in ``68.1`` and ``68.3 s``. Synthetic
  streaming tests passed, demonstrating why equation-level physical gates are
  necessary. No implementation or option was retained. SLEPc's algorithmic
  contract clarifies that ordered Schur compression, not an arbitrary retained
  Ritz block, supplies the branch-preserving restart; JAX Schur is currently
  CPU-only. Future work is narrowed to true ordered Krylov--Schur,
  Jacobi--Davidson, or a complex field-coupled correction equation. The
  time-integrated KBM release path and all thresholds remain unchanged.

- 2026-07-14: Consolidated the nonlinear-gradient evidence owner without
  changing any acceptance threshold or report schema. Four dead merged-module
  strings and three overwritten intermediate export lists were removed; its
  facade test now checks imported contracts against their actual metadata,
  replicate, and transport owners instead of importing the same facade under
  multiple aliases. The owner fell from 956 to 915 lines, package source is
  87,376 lines, and the 95-file test suite is 92,439 lines. Ruff, MyPy, and all
  164 focused nonlinear-evidence and release-gate tests pass.

- 2026-07-14: Hardened and consolidated stellarator nonlinear-transport report
  decoding. Persisted ``passed`` fields now accept only explicit Boolean
  values, while malformed, fractional, negative, and nonfinite replicate
  counts fail closed to insufficient-evidence blockers instead of becoming
  truthy or raising. Regressions cover landscape selection, campaign admission,
  and matched nonlinear-audit redesign with corrupt artifacts. One-use policy,
  action, and report-packing wrappers were folded into their owners, reducing
  ``diagnostics.stellarator_transport_reports`` from 954 to 926 lines and
  package source to 87,348 lines. Ruff, MyPy, and all 200 focused objective and
  release tests pass; thresholds and JSON schemas are unchanged.

- 2026-07-14: Simplified quasilinear nonlinear-window ingestion and removed
  duplicate artifact I/O. CSV and NetCDF heat-flux traces are now each loaded
  once, then shared by the selected-window statistics and convergence report;
  the previous path parsed every artifact twice. Replicated ensemble summaries
  also require an explicit Boolean pass flag, so a persisted ``"true"`` string
  fails closed. The calibration owner fell from 948 to 885 lines and package
  source to 87,285 lines. All 86 calibration, model-selection, promotion, and
  artifact-tool tests pass with Ruff and MyPy; window bounds, statistics,
  calibration equations, schemas, and thresholds are unchanged.

- 2026-07-14: Closed a central persisted-artifact truthiness gap in nonlinear
  transport-window promotion. Single-window reports, ensemble reports, nested
  gate reports, and ensemble rows now require explicit Boolean pass flags;
  strings such as ``"false"`` and ``"true"`` cannot support quasilinear or
  nonlinear promotion. Regressions cover both single and replicated reports.
  The two duplicate report-statistic decoders were merged, their direct
  replicate consumer now uses the real shared owner, and the one-use ensemble
  packer was folded into the public builder. ``diagnostics.transport_windows``
  fell from 943 to 919 lines and package source to 87,260 lines. All 178 focused
  transport, calibration, nonlinear-evidence, and release tests pass with Ruff
  and MyPy. Documentation was also corrected to remove references to deleted
  ``validation/quasilinear`` facades.

- 2026-07-14: Extended explicit persisted-Boolean decoding to production
  nonlinear-optimization and turbulence-gradient promotion. Ensemble pass
  fields, nested gate reports, matched baseline/candidate qualification rows,
  selected-audit gates, and gradient-artifact pass fields now reject string or
  numeric lookalikes. Replicate counts also require finite nonnegative integer
  values. Production-claim markers remain deliberately conservative: malformed
  claim flags still block release rather than weakening overclaim detection.
  Regressions cover corrupt ensemble counts, matched audits, and production
  gradient artifacts. One-use guard summary/evidence packers were folded into
  their owner, leaving ``diagnostics.nonlinear_transport_optimization`` at 951
  lines versus 954 before this tranche; package source is 87,264 lines. Ruff,
  MyPy, and all 184 focused nonlinear promotion and release tests pass.

- 2026-07-14: Repaired the export contracts of the two consolidated diagnostic
  owners. ``diagnostics.metadata`` had three section-local ``__all__``
  assignments, so Python retained only the last helper and omitted its core
  diagnostics containers and shared evidence contracts; ``diagnostics.transport``
  likewise overwrote all heat-, particle-, and heating-diagnostic exports with
  one finite-difference report. Each owner now has one complete contract, inert
  merged-module strings are removed, and a regression covers both owners. The
  223/95/95 source/test/tool topology remains at target with zero exceptions;
  package source is 87,262 lines and the 95-file test suite is 92,637 lines.
  Ruff, focused MyPy, 185 diagnostics/release tests, and the strict architecture
  manifest pass without changing numerical equations, report schemas, or gates.

- 2026-07-14: Removed the remaining circular ownership assertions left by
  earlier source merges. The imported-VMEC test no longer imports one
  orchestrator under five fictitious section-owner aliases; it now verifies
  only the contracts genuinely shared with backend discovery and field-line
  sampling. The autodiff owner likewise has one module alias rather than two
  self-comparisons. The final inert merged-module string was removed from
  installable source. Package source is 87,261 lines and the 95-file test suite
  is 92,619 lines. All 39 imported-VMEC, 93 autodiff-objective, and 94 release
  tests pass with focused MyPy; numerical kernels and public APIs are unchanged.

- 2026-07-14: Consolidated repeated imports and merge-boundary residue in four
  large test owners without removing any collected test. Autodiff objectives,
  parallel core, general artifact tooling, and stellarator artifact tooling now
  define shared standard-library, NumPy, Pytest, and repository-path imports
  once, while retained section markers still identify each physics family. All
  343 tests pass with one expected hardware-dependent skip. The 95-file suite
  falls from 92,619 to 92,576 lines, and its largest owner falls from 5,782 to
  5,770 lines; source, public APIs, numerical kernels, and gates are unchanged.

- 2026-07-14: Consolidated the 4,385-line reference-comparison test owner
  without changing benchmark scope or comparison terminology. Nine former-file
  sections now share one ``Path``/``sys``/NumPy/Pytest import boundary and one
  comparison-tool search-path insertion; duplicate section docstrings and two
  redundant global path insertions were removed while per-test isolated import
  setup remains intact. All 116 comparison tests pass with one expected
  optional-dependency skip. The owner is 4,325 lines and the 95-file suite is
  92,516 lines; comparison equations, tolerances, and artifacts are unchanged.

- 2026-07-14: Completed three additional physical KBM interior-eigenvalue
  discriminators without adding failed experimental code. At reduced
  ``Nl=4,Nm=8`` resolution, the current shift-invert pair had residual
  ``0.742``. An exact projected Jacobi--Davidson correction solved its inner
  equation to relative residual ``0.035`` but worsened the physical eigenpair
  residual to ``0.876`` on the first step and continued onto poorer vectors.
  Ordered complex-Schur compression retaining four vectors changed residual
  ``0.755`` to ``0.956``, selected a damped branch, and cost about ``60.8 s``
  per restart even at this small grid. Propagator and power seeds did not repair
  selection; the best residual was ``0.521`` on the wrong damped branch. These
  results reject one-vector projection, generic thick restart, and seed tuning
  as release paths. The time-integrated KBM gate remains the validated default;
  structured promotion now requires a two-sided or field-coupled low-moment
  correction that passes physical branch, residual, runtime, and memory gates.
  A bounded A4000 discriminator at ``Nl=8,Nm=24`` confirmed the boundary: the
  baseline residual was ``0.975`` and three projected corrections reduced it
  only to ``0.429`` while moving to the wrong high-frequency branch; projected
  correction residuals remained ``0.59--0.96``. The run completed in about
  ``30 s`` and changed no persistent office workspace.

- 2026-07-14: Rejected the remaining generic two-sided and field-coupled KBM
  correction prototypes. Adjoint inverse iteration did not provide a usable
  left mode (relative residuals ``0.67--2.12``); despite a first oblique
  correction-equation residual of ``0.015``, the physical eigenpair worsened
  from ``0.742`` to ``0.870`` and changed branch. A matrix-free block
  preconditioner retaining ``l<2,m<4`` and the self-consistent field response
  was mathematically closer to the required coupling but took ``239.9 s``
  versus ``29.6 s`` for the damping baseline, used about ``1.2 GB`` resident
  memory, selected the wrong branch, and returned residual ``0.972``. No code
  was retained. Further structured KBM work must derive an explicitly reduced
  field/moment Schur block with a bounded factorization; another nested
  full-operator Krylov solve is ruled out by accuracy, runtime, and memory.

- 2026-07-14: Consolidated VMEC/Boozer aggregate artifact-test call capture in
  the largest test owner. Alpha and surface holdouts, scalar and multi-point
  objectives, line-search comparisons, and second-equilibrium gates now use one
  keyword-call recorder while retaining separate physical payloads, failure
  paths, mode-21 assertions, and executable arguments. All 119 collected tests
  pass; the owner falls from 5,771 to 5,723 lines and the 95-file suite from
  92,516 to 92,468 lines. Ruff, Python compilation, and the 223/95/95
  architecture and complexity gates pass with no production-code change.

- 2026-07-14: Removed duplicate static-zero policy from gyrokinetic term
  assembly. Collision short-circuiting, electromagnetic field omission, linear
  term kernels, and nonlinear routing now share the dtype-aware tracer-safe
  helper owned by ``operators.linear.dissipation``; the assembly owner no
  longer carries a second NumPy implementation or a nested copy. Package source
  falls from 87,261 to 87,246 lines and the owner from 893 to 878 lines. Ruff,
  MyPy, 57 focused assembly/operator tests, 14 collision/electrostatic helper
  tests, 94 release tests, and the architecture gate pass with unchanged JAX
  trace behavior and collision equations.

- 2026-07-14: Completed the explicit field/moment Schur discriminator for the
  unresolved KBM interior mode. At ``Nl=4,Nm=8,Nz=96``, the exact ``m=0/1`` or
  ``l<2,m<4`` coarse space has 1,536 complex unknowns; batched matrix-free
  assembly took ``0.49 s``, LU factorization ``0.15 s``, and storage ``36 MiB``.
  The cheap construction did not translate into a valid solve. A diagonal
  high-moment complement took ``55.4 s``, peaked at ``1.37 GB``, selected a
  far-off branch, and returned residual ``0.936``. A multiplicative
  SOLVAX-backed Hermite-line smoother plus exact field coarse solve took
  ``54.5 s``, peaked at ``1.41 GB``, selected another wrong branch, and returned
  residual ``0.999`` versus the unchanged damping baseline ``0.742`` in about
  ``10 s``. No code was retained. This closes generic preconditioner layering
  for the current cycle; KBM stays fail-closed on validated time integration,
  and future acceleration must start from a different interior spectral
  transformation with branch identity as its first gate.

- 2026-07-14: Consolidated four repeated full-signature nonlinear-integrator
  stubs in the runtime owner into keyword-oriented call contracts. Adaptive
  chunk continuation, CFL-factor resolution, dealias-aware mode selection, and
  explicit radial-wavenumber targeting still assert the same forwarded values
  and diagnostics; only 35-argument mock signatures that duplicated the
  production API were removed. The runtime owner falls from 4,222 to 4,088
  lines and the 95-file suite from 92,468 to 92,334 lines. Ruff, Python
  compilation, the complete runtime owner suite, all 94 release tests, and the
  strict architecture and validation manifests pass with unchanged runtime
  equations and policies.

- 2026-07-14: Made ``diagnostics.metadata`` the single owner of strict
  persisted-evidence Boolean, finite-number, nonnegative-count, and gate-row
  decoding. Nonlinear optimization, replicated windows, stellarator transport,
  transport-window, and quasilinear model-selection reports no longer carry
  five divergent copies; Boolean values now consistently fail closed when a
  numeric count is required. Package source falls from 87,246 to 87,196 lines.
  Ruff, Python compilation, strict MyPy, and all 483 affected diagnostics,
  artifact-tool, VMEC-admission, gradient-evidence, and release tests pass with
  unchanged report schemas and scientific thresholds.

- 2026-07-14: Consolidated the real Cyclone ITG linear-RHS eigen-objective
  fixture shared by the isolated-eigenpair and implicit left/right derivative
  gates, and the canonical tiny linear result shared by four executable output
  routing tests. The separate tests still distinguish unsupported symmetric-
  eigenvector differentiation from the passing implicit Jacobian/finite-
  difference gate, and still cover explicit, TOML, override, and shorthand
  output paths. All 127 tests in both owners pass. The 95-file suite falls from
  92,334 to 92,256 lines; the autodiff owner is 3,519 lines and the executable
  owner is 1,936 lines.

- 2026-07-14: Consolidated the reduced QA low-turbulence contracts and
  residual/sensitivity modules into the existing model owner. Configuration,
  stable observable names, result serialization, reduced QA/ITG equations,
  fixed-step envelope, weighted residuals, scalar objective, covariance, and
  AD/finite-difference gates now form one 653-line mathematical model; optimizer
  orchestration and publication-artifact construction remain separate owners.
  Obsolete internal module paths were removed without shims, while every root
  API export was redirected to the current owner. The package falls from 223 to
  221 source modules and from 87,196 to 87,153 lines. The validation and
  differentiable-refactor manifests, generated traceability summary, API docs,
  and code-structure map were migrated with the equations. Ruff, MyPy, four QA
  validation tests, 93 autodiff-objective tests, 94 release tests, public-import
  checks, the architecture manifest, and strict 31-page Sphinx build pass.

- 2026-07-14: Hardened distribution builds after the installed-wheel audit
  detected that a local pre-existing ``build/lib`` directory could reintroduce
  deleted modules into a nominally new wheel. Both CI and release workflows now
  remove ``build`` and ``dist`` immediately before ``python -m build``;
  release-readiness policy requires that clean-build guard, and installed-wheel
  smoke tests require the retired QA contract/residual module paths to be
  absent. A deliberately dirty first build reproduced the defect; a clean
  rebuild produced a 682 KiB wheel with no stale members, imported every QA
  root API from the consolidated model, and evaluated the same objective.
  Workflow YAML, all 95 release tests, both architecture manifests, and the
  generated validation summary pass.

- 2026-07-14: Merged the single-consumer stellarator ITG residual and
  conditioning policy into the reduced stellarator model, while retaining the
  shared contracts and independently consumed table builder as separate
  owners. The public facade and root API are unchanged, the obsolete internal
  residual module is removed without a compatibility shim, and CI/release
  installed-wheel checks require that retired path to remain absent. The
  package falls from 221 to 220 source modules and from 87,153 to 87,119 lines;
  the combined model owner is 637 lines. Ruff, strict MyPy, both architecture
  manifests, validation-trace regeneration, all 93 differentiable-objective
  tests, all 95 release tests, the warnings-as-errors 31-page documentation
  build, and a clean installed 681 KiB wheel pass.

- 2026-07-14: Merged the single-consumer resolved mode-series growth and
  frequency diagnostics into their public growth-rate owner. Least-squares fit
  kernels and automatic fit-window selection retain focused modules because
  they are independently consumed and tested numerical algorithms. The retired
  internal module has no compatibility shim and is explicitly rejected by
  clean installed-wheel checks. The package falls from 220 to 219 source
  modules and from 87,119 to 87,088 lines; the combined owner is 580 lines.
  Ruff, strict MyPy, both architecture manifests, validation-trace generation,
  all 44 owner tests, all 356 direct runtime/artifact/benchmark consumers, all
  95 release tests, strict documentation, and the clean installed wheel pass.

- 2026-07-14: Consolidated private collection-level VMEC-JAX boundary-chain
  row/count/decision policy into the only production consumer, the scalar
  boundary-chain diagnostics owner. The 730-line owner now makes exact-solve
  finite-difference, frozen-axis replay, branch-locality, and multi-probe
  promotion policy testable in one place; unrelated tensor mapping and solved-
  state sensitivity remain separate. The package falls from 219 to 218 modules
  and from 87,088 to 87,068 lines, and the test suite falls to 92,263 lines.
  Ruff, strict MyPy, architecture and validation manifests, and all 225 direct
  objective/artifact tests pass. This closes the planned source-consolidation
  lane without merging independently reusable numerical domains.

- 2026-07-14: Reconciled the prioritized roadmap and comparison-scope docs
  against the implemented evidence. Source ownership is now explicitly closed;
  periodic/linked ``2 species x 2 Hermite`` electrostatic routing is described
  as an identity-gated operator path rather than absent planning metadata; and
  the published reduced original/improved-Sugama and Coulomb matrices are
  distinguished from the still-unimplemented complete finite-``b``
  multispecies hierarchy. No nonlinear speedup or full Landau claim was added.

- 2026-07-14: Closed test consolidation after auditing all 95 owners by size,
  helper name, and import fan-in. Shared physical ITG, runtime integrator,
  executable result, EIK protocol, status, and artifact contracts already own
  repeated setup. The remaining repeated tool-loader wrappers deliberately
  provide fresh module state to monkeypatch-heavy tests and are not candidates
  for a global fixture. The 95-file topology and 95% package-wide coverage
  gates remain the acceptance criteria; scientific cases are not deleted to
  chase an ungrounded line-count target.

- 2026-07-14: Extended the finite-wavelength collision-table boundary from
  shared/per-species matrices to ordered target/source pair tables with shape
  ``(target, source, kperp, moment, moment)``. Interpolation uses each target
  species' spatial :math:`k_\perp` field, stays inside JAX, and feeds the
  existing multispecies contraction without global reconstruction. A two-
  species improved-Sugama zero-:math:`b` endpoint preserves particles, total
  momentum, and total thermal energy; nonzero controlled pair slopes pass
  spatial application identity and JVP/finite-difference gates. All 30 operator
  tests pass. This closes required target/source runtime plumbing but does not
  promote finite-:math:`b` Sugama/Coulomb coefficients, which still require the
  arbitrary-moment high-precision generator and transport-level gates.

- 2026-07-14: Added ``TabulatedMultispeciesCollisionOperator`` as the direct
  collision-protocol owner for generated target/source coefficient tables. The
  JAX-pytree operator derives each target species' normalized perpendicular
  wavenumber from ``sqrt(cache.b)``, interpolates in traced execution, and acts
  on the post-field Hamiltonian. A real two-species linear-RHS gate with a
  constant improved-Sugama pair table is identical to the existing reduced
  operator, including collision-term weighting; malformed table rank fails
  closed. This completes end-to-end table routing but deliberately leaves the
  unavailable arbitrary-moment finite-:math:`b` coefficients unpromoted.
