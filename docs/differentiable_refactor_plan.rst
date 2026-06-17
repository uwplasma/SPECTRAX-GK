Differentiable Refactor Plan
============================

Purpose
-------

This page defines the planned research-grade differentiable architecture for
SPECTRAX-GK. The goal is not to rewrite the solver for aesthetics. The goal is
to make the code easier to use, easier to test, easier to extend, and safer to
differentiate through while preserving the validated physics behavior.

Authority note: :doc:`architecture_refactor_plan` is the authoritative plan for
future package layout, file naming, migration order, and conflict resolution.
This page is a technical appendix for differentiability contracts, active
manifest rows, historical split inventory, and AD/physics/performance gate
requirements. If a layout or naming detail here conflicts with
:doc:`architecture_refactor_plan`, the architecture plan wins and this appendix
or ``tools/differentiable_refactor_manifest.toml`` should be updated.

The active branch for this work starts with planning infrastructure only:
``tools/differentiable_refactor_manifest.toml`` and
``tools/check_differentiable_refactor_manifest.py``. Large code moves should
land in later PRs only after the relevant manifest row, compatibility facade,
fast tests, parity gates, and documentation have been updated.

Design Principles
-----------------

1. Preserve public behavior first.
   Existing user-facing imports such as ``spectraxgk.linear``,
   ``spectraxgk.nonlinear``, ``spectraxgk.runtime``, and
   ``spectraxgk.geometry.differentiable`` remain compatibility facades while
   internals are split.
2. Keep differentiable paths pure.
   Objective functions must take explicit PyTree parameters and arrays and
   return arrays or typed reports. File I/O, plotting, subprocesses, and
   progress printing stay outside differentiated functions.
3. Make static and dynamic data explicit.
   JIT-compiled kernels should separate static model choices from dynamic array
   values so scans and optimizations do not recompile unnecessarily.
4. Prefer small kernel modules over broad scripts.
   Physics terms, basis recurrences, field solves, brackets, integrators,
   diagnostics, and artifact writing each get their own testable module.
5. Use custom derivatives only when justified.
   Default to native JAX autodiff. Add ``custom_jvp`` or ``custom_vjp`` only for
   solver, eigenvalue, fixed-point, or adjoint paths that have finite-difference
   and tangent tests.
6. Treat validation adapters as boundary code.
   Reference-code comparison helpers remain in validation or adapter modules.
   Solver kernels should not import them.
7. Make extension points deliberate.
   New collision operators, closure models, field solvers, polynomial bases,
   geometry providers, and transport models should be registered through narrow
   protocols instead of edited into monolithic files.
8. Document physics and numerics at the public boundary.
   Public functions need docstrings describing equations, normalization, array
   shapes, differentiability status, static arguments, and relevant tests.

JAX and Scientific-Python Guidance
----------------------------------

The architecture should follow current JAX ecosystem practice:

- `JAX pytrees <https://docs.jax.dev/en/latest/pytrees.html>`_ are the natural
  container model for solver state, geometry, parameters, and objective reports.
- `JAX custom derivative rules <https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_
  should be reserved for transformable functions whose default derivative is
  unstable, too expensive, or mathematically wrong for the promoted objective.
- `Diffrax <https://docs.kidger.site/diffrax/>`_ already uses PyTree states,
  vmappable solves, and multiple adjoint methods; its adjoint guidance is a
  useful model for separating forward-mode, reverse-mode, direct, and
  checkpointed differentiation policies.
- `Equinox <https://docs.kidger.site/equinox/all-of-equinox/>`_ provides a
  minimal PyTree module pattern and filtered transforms without imposing a
  framework; use it selectively where callable parameter containers improve
  clarity.
- `Optax <https://optax.readthedocs.io/>`_ is appropriate for composable
  gradient transformations in noisy or staged objectives.
- `JAXopt <https://jaxopt.github.io/>`_ is appropriate for differentiable
  deterministic solvers, implicit differentiation, and tree-structured
  parameters.
- `Lineax <https://docs.kidger.site/lineax/>`_ is a good model for linear
  operator abstractions and differentiable linear solves.
- `Orthax <https://orthax.readthedocs.io/>`_ is relevant for future
  interchangeable orthogonal-polynomial basis families in JAX.
- `PEP 257 <https://peps.python.org/pep-0257/>`_ and the
  `Google Python style guide <https://google.github.io/styleguide/pyguide.html>`_
  are the baseline for docstring and API documentation conventions.

Technical Layer Map
-------------------

The detailed target source layout is maintained in
:doc:`architecture_refactor_plan`. The following layer map is retained here
only to connect differentiability and validation contracts to broad
responsibility groups:

.. code-block:: text

   src/spectraxgk/
     core/              # pytrees, dtypes, normalization, static/dynamic helpers
     basis/             # basis protocols, Hermite-Laguerre, Orthax adapters
     grids/             # spectral and field-line grids
     geometry/          # analytic, VMEC, vmec_jax, booz_xform_jax contracts
     physics/           # species, closures, fields, collisions, drives
     operators/         # pure linear/nonlinear kernels and field solves
     solvers/           # linear, nonlinear, adjoints, time steppers
     diagnostics/       # growth, frequency, transport, spectra, UQ, windows
     validation/        # literature gates, reference adapters, benchmark families
     workflows/         # runtime, scans, optimization, plotting, provenance
     io/                # TOML, NetCDF, restart, artifact schemas
     cli/               # executable commands and default demo

Existing public modules remain as facades until the planned API cleanup. New
implementation code should be placed under the domain packages named in
:doc:`architecture_refactor_plan`, not added as new root-level prefix modules.

High-Risk Module Split Plan
---------------------------

``benchmarks.py``
  Split into benchmark-family modules, reference-data loaders, and fit policies.
  Keep ``spectraxgk.benchmarks`` as the public facade. Required gates: Cyclone,
  ETG, KBM, TEM, W7-X/HSX where applicable, and branch-continuity policies.

``geometry/differentiable.py``
  Split backend discovery, geometry contracts, VMEC-JAX bridge, Boozer bridge,
  equal-arc mapping, sensitivity reports, and parity reports. Required gates:
  optional-backend import behavior, same-WOUT provenance, geometry parity,
  JVP/VJP/finite-difference agreement, and conditioning diagnostics.

``nonlinear_parallel.py``
  Split domain plans, spectral communication, device-z pencil route, observable
  reductions, and profiling. Required gates: serial-vs-decomposed RHS identity,
  physical transport-window identity, profiler artifact schema, and no speedup
  claim without matched CPU/GPU artifacts. Local spectral-state construction,
  chunk/layout utilities, communication/work models, pencil FFT/bracket
  primitives, RHS micro-routes, and tolerance helpers now live in
  ``nonlinear_parallel_spectral_core.py`` behind the unchanged
  ``spectraxgk.nonlinear_parallel`` facade.

``solver_objective_gradients.py``
  Split eigenvalue objectives, linear-growth objectives, quasilinear flux
  objectives, nonlinear-window objectives, VMEC/Boozer objective plumbing, and
  gradient gates. Required gates: branch-locality, spectral-gap guards,
  finite-difference/JVP/VJP checks, UQ covariance, and objective conditioning.
  The dominant-growth implicit eigenpair VJP and branch-locality report now
  live in ``solver_eigen_objectives.py`` and remain re-exported by the legacy
  solver-objective facade. Solver-objective sampling axes, physical-``ky`` grid
  construction, and aggregate weights now live in
  ``solver_objective_sampling.py`` behind the same facade. Core
  linear/quasilinear objective constants, scalar selectors, operator
  materialization, growth-rate, and objective-vector evaluators now live in
  ``solver_objective_core.py``. Solver-ready geometry objective gates, solver
  geometry-gradient reports, mode-21 VMEC/Boozer state-gradient reports,
  reduced nonlinear-window estimator metrics, VMEC/Boozer state coefficient
  helpers, VMEC/Boozer objective-table plumbing, and VMEC/Boozer
  finite-difference/line-search gates now live in
  ``solver_geometry_objectives.py``, ``solver_nonlinear_window_objective.py``,
  ``solver_ready_gradient_gates.py``,
  ``solver_vmec_boozer_gradient_gates.py``, ``solver_vmec_state.py``,
  ``solver_vmec_boozer_objectives.py``,
  ``solver_vmec_boozer_fd_gates.py``, and
  ``solver_vmec_boozer_line_search_gates.py`` while
  ``solver_objective_gradients.py`` remains the higher-level public objective
  surface.

``nonlinear.py``
  Split RHS kernels, integrator policies, nonlinear diagnostics, and IMEX paths.
  Required gates: RHS identity, transport windows, spectral diagnostics, and
  parity-preserving output schemas. The nonlinear RHS linear-path routing and
  electromagnetic bracket composition now live in
  ``operators/nonlinear/rhs.py`` while
  ``spectraxgk.nonlinear`` remains the compatibility facade for public imports,
  monkeypatch-based diagnostics, and runtime workflows. Duplicated explicit and
  IMEX state-to-diagnostic tuple assembly now lives in
  ``operators/nonlinear/diagnostic_state.py`` with facade-injected diagnostic
  kernels so existing debug seams remain intact. Shared scan-output sampling,
  resolved-diagnostic packing, and ``SimulationDiagnostics`` construction now
  live in ``spectraxgk.nonlinear_diagnostics``. Shared diagnostic cache,
  quadrature-weight, omega-mask, z-index, and state-projection setup now lives
  in ``spectraxgk.nonlinear_helpers`` with injected compatibility seams.
  Explicit RK/SSP/K10 one-step policy now lives in
  ``solvers/nonlinear/explicit.py`` and the shared IMEX fixed-point/GMRES solve
  plus SSPX3 stage-composition policy lives in ``solvers/nonlinear/imex.py``.

``runtime.py`` and ``cli.py``
  Split executable commands, runtime workflows, scan dispatch, progress/ETA,
  plotting, and artifact handoff. Runtime TOML case wrappers now delegate to
  ``spectraxgk.workflows.cases`` through dependency-injected facades so
  ``spectraxgk.runtime`` remains the public import and monkeypatch surface.
  Required gates: default-run behavior, ``--plot`` behavior, TOML provenance,
  restart/output schema, and import compatibility.

``runtime_artifacts.py``
  Split artifact schema, NetCDF persistence, restart append, and provenance.
  Generic artifact I/O, linear/quasilinear writers, nonlinear table writers,
  nonlinear diagnostic reload helpers, and finite-value artifact validation now
  live under ``spectraxgk.artifacts``. The root ``runtime_artifacts.py`` module
  remains the public dispatcher and monkeypatch-compatible executable seam.
  Required gates: round-trip persistence, restart append normalization, and
  plot reload contracts.

``linear.py``
  Split linear RHS, field solves, integrators, and diagnostics. Required gates:
  field solve identity, late-time growth/frequency metrics, eigenfunction
  overlap, branch continuity, and JAX transform consistency.

Execution Phases
----------------

Phase 0: freeze contracts
  Land this plan, the manifest checker, and a CI-safe test. Record current
  public imports, coverage targets, benchmark gates, and docs entry points.

Phase 1: introduce protocols and containers
  Add small protocol/dataclass modules for basis families, geometry providers,
  collision operators, field solvers, RHS assembly, diagnostics, objective
  reports, and artifact schemas. Avoid behavior changes. The first Phase-1
  tranche now lives in ``spectraxgk.core.contracts`` and
  ``spectraxgk.core.extension_points``. The first compatibility-preserving
  benchmark split also lives in this phase:
  ``spectraxgk.benchmark_initialization`` owns benchmark initial-condition
  construction and ``spectraxgk.benchmark_reference`` owns reference containers
  and CSV loaders. ``spectraxgk.benchmark_species`` owns benchmark
  species-to-``LinearParams`` construction and reference hypercollision policy,
  ``spectraxgk.benchmark_fit_signals`` owns fit-signal and diagnostic
  normalization policies, ``spectraxgk.benchmark_batching`` owns scan batching
  and streaming windows, and ``spectraxgk.benchmark_solver_policy`` owns
  branch-selection policies. ``spectraxgk.benchmark_cyclone``,
  ``spectraxgk.benchmark_kbm``, ``spectraxgk.benchmark_tem``,
  ``spectraxgk.benchmark_kinetic``, and ``spectraxgk.benchmark_etg`` own the
  extracted family runners while ``spectraxgk.benchmarks`` remains the public
  benchmark entry point. The obsolete benchmark helper bridge has been removed;
  runners and tests import focused benchmark modules directly. The
  first differentiable-geometry support split also lives in this phase:
  ``spectraxgk.geometry.backend_discovery`` owns
  optional ``vmec_jax`` / ``booz_xform_jax`` path discovery and
  ``spectraxgk.geometry.autodiff_checks`` owns finite-difference Jacobians,
  AD/FD reports, conditioning metadata, and strict JSON sanitation.
  ``spectraxgk.geometry.flux_tube_contract`` owns solver-ready in-memory
  flux-tube mapping validation and geometry-observable reductions.
  ``spectraxgk.geometry.sensitivity`` owns geometry sensitivity,
  inverse-design, conditioning, and local UQ reports for solver-ready mappings.
  ``spectraxgk.geometry.booz_xform_bridge`` owns bounded VMEC boundary and
  Boozer-spectrum bridge checks, Boozer field-line ``|B|`` evaluation, and
  Boozer-to-flux-tube sensitivity diagnostics.
  ``spectraxgk.geometry.vmec_state_sensitivity`` owns optional-backend
  ``VMECState`` sensitivity reports for VMEC-to-Boozer, VMEC metric tensor, and
  VMEC field-line tensor AD/FD gates.
  ``spectraxgk.geometry.vmec_boozer_core`` owns the ``vmec_jax`` state to
  ``booz_xform_jax`` equal-arc core-profile bridge, including Boozer constants
  cache/prewarm and solver-facing core arrays.
  ``spectraxgk.geometry.vmec_tensor_mapping`` owns direct ``vmec_jax`` tensor
  sampling and conversion into the solver-ready flux-tube mapping contract.
  ``spectraxgk.geometry.vmec_flux_tube_reports`` owns VMEC flux-tube
  sensitivity and array-parity report orchestration that combines the direct
  tensor, Boozer equal-arc, and imported-geometry comparison paths.
  ``spectraxgk.geometry.numerics`` owns pure parity metrics, interpolation,
  radial derivative, Boozer half-mesh, Fourier field-line, and periodic
  sampling helpers.
  ``spectraxgk.geometry.differentiable`` retains object-identical re-exports
  for pure helpers and thin wrappers for optional-backend bridge functions whose
  legacy tests patch facade-level backend discovery.

Phase 2: split pure kernels
  Move basis, gyroaverage, field-solve, linear-term, nonlinear-bracket, and
  diagnostic kernels first. These have the clearest unit and numerical tests.

Phase 3: split differentiable geometry
  Move the remaining VMEC/Boozer bridge and parity routines behind in-memory
  geometry contracts. Optional backend discovery and geometry AD/FD gate
  utilities, the in-memory flux-tube contract, sensitivity reports, and pure
  numerical helpers are already split into Phase-1 support modules. Keep
  same-WOUT and finite-difference gates mandatory.

Phase 4: split objectives and AD policies
  Separate linear, quasilinear, nonlinear-window, and VMEC/Boozer objectives.
  Add explicit adjoint policy selection and branch-locality guards.

Phase 5: split runtime, CLI, and workflows
  Move side-effectful command/runtime/plotting code out of solver kernels. Add
  progress and provenance tests for the executable.

Phase 6: extension registries
  Add registries for collision operators, basis families, closure models,
  field solvers, geometry providers, transport diagnostics, and benchmark
  families. Each registry entry must declare tests and docs.

Phase 7: deprecation cleanup
  Keep compatibility facades through the release series. Remove legacy imports
  only in a major-version plan after downstream examples have been migrated.

Differentiability Contract
--------------------------

Every promoted differentiable workflow should declare:

- the optimized parameters and their PyTree structure;
- which inputs are static and which are traced arrays;
- whether gradients use forward mode, reverse mode, direct adjoint,
  checkpointed adjoint, implicit differentiation, or custom JVP/VJP;
- the finite-difference step policy and accepted error window;
- spectral-gap or branch-locality criteria for eigenvalue objectives;
- conditioning, rank, and covariance diagnostics for inverse/UQ examples;
- whether the observable is a production physics observable or a reduced
  differentiable proxy.

Recommended default choices:

- low-dimensional geometry or scalar scans: ``jax.jvp`` and finite differences;
- many-parameter smooth deterministic objectives: reverse mode or JAXopt
  implicit differentiation after branch-locality gates pass;
- time-integration objectives: Diffrax-style explicit adjoint policy selection
  with checkpointing documented;
- noisy nonlinear transport windows: common-random-number replicas, robust
  window statistics, derivative-free outer loops when gradients are not
  conditioned, and no production claim until long post-transient audits pass.

Test Matrix
-----------

Unit and numerical tests
  Basis orthonormality and recurrence, gyroaverage limits, field-solve
  consistency, nonlinear bracket symmetries, conservation/free-energy reduced
  limits, observed-order checks, and artifact round trips.

Autodiff tests
  Finite-difference versus JVP/VJP for geometry observables, eigenvalue
  objectives, quasilinear flux weights, reduced nonlinear-window estimators,
  UQ covariance, and sensitivity maps.

Parity tests
  Reference-code comparisons remain external validation inputs. Required
  metrics include growth rate, real frequency, eigenfunction overlap,
  late-window nonlinear transport, spectra, and diagnostic schema equality.

Literature-anchored physics gates
  Keep the existing Cyclone, shaped tokamak, KBM, ETG, TEM, W7-X/HSX,
  Rosenbluth-Hinton/Merlo, and quasilinear/nonlinear holdout gates. Add new
  literature gates only when the observable, window, normalization, and
  tolerance are documented.

Performance and parallel tests
  Require numerical identity before speedup. Require profiler artifacts before
  claims. Track compile time, warm throughput, memory, communication model,
  and CPU/GPU behavior separately.

Acceptance Criteria
-------------------

A refactor tranche is ready only when all of the following are true:

- public imports either remain object-compatible or have documented deprecation
  shims;
- package-wide coverage remains at or above 95%;
- each moved module has unit tests and at least one physics, numerical,
  autodiff, parity, or artifact gate appropriate to its responsibility;
- no public module exceeds 800 lines unless the manifest records an exception;
- no internal module exceeds 1200 lines unless the manifest records an
  exception;
- docs list equations, normalization, shapes, differentiability status, and
  extension points for public APIs;
- benchmark and reference-code adapters are isolated from solver kernels;
- all published figures or claims have replay commands and artifact provenance.

Developer Checklist For Future PRs
----------------------------------

- Update ``tools/differentiable_refactor_manifest.toml`` before moving code.
- Add or update fast tests before changing numerical behavior.
- Keep compatibility facades until examples and docs are migrated.
- Run the manifest checker and the affected fast tests locally.
- If a gate tolerance changes, update the validation docs and artifact ledger.
- If a new extension point is added, include one minimal real implementation
  and one test that demonstrates how a contributor would add another one.

References and External Models
------------------------------

The plan is anchored in current JAX and gyrokinetic validation practice:

- JAX PyTrees and transforms: https://docs.jax.dev/en/latest/pytrees.html
- JAX custom derivative rules: https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
- Diffrax ODE solvers and adjoints: https://docs.kidger.site/diffrax/
- Equinox PyTree modules and filtered transforms: https://docs.kidger.site/equinox/all-of-equinox/
- Optax gradient transformations: https://optax.readthedocs.io/
- JAXopt differentiable optimizers: https://jaxopt.github.io/
- Lineax differentiable linear operators: https://docs.kidger.site/lineax/
- Orthax orthogonal polynomials in JAX: https://orthax.readthedocs.io/
- Python docstrings: https://peps.python.org/pep-0257/
- W7-X stella/GENE benchmark: https://doi.org/10.1017/S002237782100126X
- Linear multispecies shaped-geometry benchmarks: https://doi.org/10.1063/1.4943004
- Rosenbluth-Hinton and stellarator zonal-flow response literature cited in
  :doc:`references`.
