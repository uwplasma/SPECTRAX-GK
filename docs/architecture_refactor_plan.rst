Architecture Refactor Plan
==========================

Purpose
-------

This page is the authoritative refactor plan companion for SPECTRAX-GK. The
recent behavior-preserving extractions reduced several large files, but they
also created too many root-level modules with prefix-based names. That makes the
code harder to navigate, harder to teach, harder to extend, and harder to test.

The new target is a small set of domain packages with stable public facades,
private implementation modules, explicit contracts, and tests that mirror the
package structure. The goal is not minimal file count at all costs. The goal is
clear ownership, research-grade validation, strong JAX transformability, and a
code layout that a new developer can understand without knowing the previous
refactor sequence.

Plan Authority And Conflict Resolution
--------------------------------------

``plan.md`` in the repository root is the active refactor authority and work
log. This page is the architectural companion: it explains target package
layout, naming rules, and acceptance gates, but it should be updated whenever
``plan.md`` changes direction.

- :doc:`code_structure` documents the current source tree and public facades.
  It should not be read as the target architecture.
- :doc:`differentiable_refactor_plan` is now a technical appendix for
  differentiability contracts, completed split inventory, and validation-gate
  traceability. It does not override this page's target package layout or
  naming rules.
- ``tools/differentiable_refactor_manifest.toml`` remains the executable
  migration ledger for active tranches. When a manifest row conflicts with this
  page, update the manifest row rather than adding a new root-level module.
- ``plan.md`` records priority, current status, recent checkpoints, and the
  current execution order. If this page conflicts with ``plan.md``, update this
  page or the relevant manifest rather than creating another plan.
- README and user docs should describe stable user workflows, not internal
  migration scaffolding.

Any future refactor PR should cite this page, list the package ownership it is
changing, and state which public facade preserves documented user behavior.

Planning Snapshot
-----------------

The planning audit on 2026-06-21 found:

- 357 Python source files under ``src/spectraxgk`` after the runtime command-artifact consolidation.
- about 106,600 source lines under ``src/spectraxgk``.
- no blocked root-prefix modules under the architecture manifest.
- 315 top-level test files under ``tests``.
- about 89,000 test lines.
- 9 root facade modules: ``benchmarks.py``, ``cli.py``, ``config.py``,
  ``linear.py``, ``nonlinear.py``, ``quasilinear.py``, ``runtime.py``,
  ``_version.py``, and ``__init__.py``.
- the former root-level prefix families such as ``runtime_*``,
  ``nonlinear_*``, ``vmec_jax_*``, ``quasilinear_*``, and ``benchmark_*`` have
  been moved behind domain packages or public facades.

That structure is close to the desired public shape. Future work should reduce
navigation cost inside domain packages and should not add more root-level prefix
modules unless they are deliberate public facades tracked in the migration
manifest.

The refreshed topology audit on 2026-07-07 found that root-prefix modules are no
longer the main problem. The current blockers are installable validation
campaign code and flat maintenance namespaces:

- 351 Python source files under ``src/spectraxgk`` after retiring the
  non-promoted reduced cETG runtime path.
- 88 Python files under ``src/spectraxgk/validation``.
- 255 Python test files, including the shared ``tests/support/paths.py`` helper;
  only ``conftest.py`` still lives directly under ``tests`` after the flat
  runtime/executable tests and the first artifact-gate families were
  consolidated.
- 260 Python tool scripts, with only ``tools/__init__.py`` left at the flat
  top level after release, comparison, artifact, campaign, profiling,
  benchmark, generator, compression-helper, reference-helper, diagnostic, and
  VMEC-helper moves.
- no tracked files above 1 MB and no tracked ``__pycache__`` / ``.pyc`` /
  ``.DS_Store`` files.

The next refactor should therefore delete, merge, or move non-promoted code
before adding new modules. In particular, validation campaigns should leave the
installable package, tool scripts should consolidate inside their
purpose-specific folders, one-file-per-tool tests should become parametrized
family tests, and retired/non-promoted or synthetic workflows should not remain
on ``main`` unless they are promoted and documented.

External Design Guidance
------------------------

The package should follow patterns that work well in modern scientific Python
and JAX codes:

- `JAX pytrees <https://docs.jax.dev/en/latest/pytrees.html>`_ should be the
  native container model for solver state, geometry, parameters, objective
  reports, optimizer state, uncertainty reports, and restart metadata.
- `JAX custom derivative rules <https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_
  should be used only when the promoted observable has finite-difference,
  tangent, and conditioning tests.
- `JAX structured control flow <https://docs.jax.dev/en/latest/control-flow.html>`_
  makes adaptive branches a deliberate solver-design choice: ``scan`` is
  reverse-mode differentiable, ``fori_loop`` is reverse-mode differentiable
  when endpoints are static, and ``while_loop`` is forward-mode differentiable
  but not a general reverse-mode path.
- `Diffrax <https://docs.kidger.site/diffrax/>`_ is the model for separating
  terms, solvers, controllers, adjoints, and PyTree-valued state.
- `Equinox <https://docs.kidger.site/equinox/all-of-equinox/>`_ is the model
  for regular JAX code with lightweight PyTree modules and filtered
  transformations rather than a heavy framework.
- `Lineax <https://docs.kidger.site/lineax/>`_ is the model for explicit linear
  operator and linear-solve abstractions.
- `Optax <https://optax.readthedocs.io/>`_ and
  `JAXopt <https://jaxopt.github.io/>`_ are the model for separating objective
  construction from optimizer policy.
- `DESC <https://github.com/PlasmaControl/DESC>`_ is the closest stellarator
  design reference for separating equilibria, compute kernels, objective
  functions, optimizer policies, and validation.
- The spectral-PDE adjoint strategy in
  `Fast automated adjoints for spectral PDE solvers <https://arxiv.org/abs/2506.14792>`_
  is a useful long-term model: build adjoints from the operator graph so
  gradients preserve sparse/spectral solver structure instead of recording a
  memory-heavy primal trace.
- `Dedalus <https://dedalus-project.readthedocs.io/>`_ is a useful reference
  for spectral-method software separating equations, bases/domains, solvers,
  and analysis tasks.
- `PlasmaPy <https://docs.plasmapy.org/>`_ is a useful reference for a
  community plasma-code documentation surface that distinguishes user guides,
  API references, examples, and developer guidance.
- `SimPEG <https://docs.simpeg.xyz/>`_ is a useful reference for separating
  simulations, maps, objective functions, inversions, regularization, and
  directives in an optimization-oriented scientific package.
- `JAX just-in-time compilation <https://docs.jax.dev/en/latest/jit-compilation.html>`_
  motivates coarse, stable compiled boundaries rather than many shape-changing
  tiny wrappers.
- `JAX distributed arrays <https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>`_
  and `manual ``shard_map`` parallelism <https://docs.jax.dev/en/latest/notebooks/shard_map.html>`_
  motivate explicit global/local layout contracts for CPU/GPU parallel paths.

These references point to the same practical rule: user workflows should be
simple, but internals should be separated by responsibility, not by previous
file prefix.

Non-Negotiable Constraints
--------------------------

- Preserve documented user behavior during the refactor.
- Keep stable public imports as facades, but remove old helper shims once
  examples, docs, and tests use the canonical package owners.
- Keep package-wide coverage at or above the release gate.
- Keep physics parity, validation gates, and benchmark claims unchanged unless
  a dedicated validation artifact promotes a change.
- Keep differentiated paths pure: no file I/O, plotting, subprocess calls,
  terminal progress, mutable global state, or host callbacks inside promoted
  objective functions.
- Keep executable/runtime paths allowed to be user-friendly and non-pure:
  progress output, TOML emission, plotting, and saved-output management belong
  there, not inside solver kernels.
- Use comparison-code names only in explicit benchmark/comparison contexts.
  Source modules, variables, tests, and docs should otherwise use physics,
  numerics, and data-schema names.
- Do not make new performance claims without profiler artifacts and
  identity-gated numerical evidence.

What "Simpler" Means
--------------------

The refactor should make SPECTRAX-GK simpler for users and developers, but not
by hiding physics or making every file artificially tiny.

Simpler for users
  A small public API, an executable that works out of the box, clear examples,
  stable TOML schema, clear saved-output paths, and plotting commands that do
  not require internal package knowledge.

Simpler for developers
  Domain packages with one responsibility, public facades with documented
  exports, private helper modules with narrow contracts, tests that live next
  to the behavior they protect, and no need to know the migration
  sequence.

Simpler for reviewers
  Each physics claim points to equations, normalization, validation gates,
  convergence criteria, and reproducible outputs. Validation code is isolated
  from solver kernels.

Simpler for performance work
  Hot kernels have stable JIT boundaries, static/dynamic arguments are explicit,
  memory layout is documented, and profiler output maps to named solver stages.

Simplicity anti-patterns to avoid:

- many root-level modules distinguished only by prefixes;
- facades that re-export undocumented internals;
- duplicate data containers for the same physical object;
- side-effectful runtime code inside differentiable objectives;
- one giant "misc helpers" module;
- tests that patch private names across many unrelated modules;
- performance wrappers that change array layout without identity gates.

Architecture Principles
-----------------------

Domain packages instead of prefix files
  Organize by physics and numerical responsibility. Do not continue building
  long sequences of ``nonlinear_*`` or ``vmec_jax_*`` files at package root.

Small public surface, private implementation modules
  Each domain package exposes a small ``__init__.py`` or public facade. Detailed
  implementation can live in private modules such as ``_rhs.py`` or
  ``_assembly.py`` when that helps testability.

Stable facades during migration
  ``spectraxgk.linear``, ``spectraxgk.nonlinear``, ``spectraxgk.runtime``,
  ``spectraxgk.geometry``, ``spectraxgk.quasilinear``, and
  ``spectraxgk.benchmarks`` remain public facades while internals move.

Explicit contracts at package boundaries
  Shared dataclasses, protocols, and PyTree containers live in ``core`` and
  package-specific ``types.py`` modules. Function signatures should make static
  configuration, dynamic arrays, differentiability status, shapes, and units
  explicit.

Pedagogical docs without kernel clutter
  Public functions get docstrings explaining equations, normalization, array
  shape, differentiability, and relevant validation tests. Low-level kernels
  should have short comments only where the implementation is not obvious.

Tests mirror packages
  Top-level mega test files should be split into package directories. Shared
  fixtures should live in ``tests/fixtures`` or package-specific
  ``conftest.py`` files.

Python API, Executable, And Differentiability Boundary
-------------------------------------------------------

SPECTRAX-GK should support two intentionally different execution surfaces:

Python research API
  Pure functions and PyTree state for linear solves, nonlinear windows,
  quasilinear models, geometry sensitivity, UQ, and stellarator optimization.
  This path should remain compatible with ``jit``, ``vmap``, ``grad``,
  ``jvp``, ``vjp``, checkpointing, and distributed arrays where the relevant
  gates pass.

Executable workflow
  Friendly runtime behavior: progress output, ETA, TOML echoing, output-file
  writing, plotting, restart handling, and human-readable errors. This path can
  be non-differentiable and can use host-side I/O because it is not the
  promoted optimization API.

Shared numerical kernels
  Both surfaces call the same operator and solver kernels. The executable
  should wrap pure kernels; pure kernels should not import executable code.

Differentiation Method Policy
-----------------------------

SPECTRAX-GK should not use one differentiation strategy everywhere. The method
must match the observable, solver structure, parameter dimension, and memory
budget. The following ladder is the default policy for promoted Python
research APIs.

Native JAX differentiation
  Use ``jacfwd``, ``jvp``, ``grad``, ``vjp``, and ``scan``-based reverse-mode
  autodiff for small dense validation problems, smooth reduced objectives,
  fixed-step windows, and pure algebraic geometry/diagnostic maps. This is the
  first choice when memory is bounded and FD/tangent gates pass.

Implicit eigenpair differentiation
  Use the left/right eigenpair sensitivity for isolated linear branches. This
  is already the right method for growth-rate and frequency objectives because
  it avoids differentiating through non-Hermitian eigenvector internals.
  Every promoted row must keep branch-gap, nearest-branch finite-difference,
  and phase-invariant observable gates.

Implicit root/fixed-point differentiation
  Use JAXopt-style implicit differentiation, or a local custom VJP around a
  stated optimality/root equation, for converged fixed points, equilibria,
  optimizer inner solves, and nonlinear steady/window conditions. Do not
  differentiate through arbitrary optimizer iteration history unless the
  iteration count is intentionally part of the objective.

Linear-solve adjoints
  Use explicit matrix-free operator abstractions, following the Lineax pattern,
  for Krylov, preconditioned, and least-squares sensitivities. This keeps
  transposes, solve tolerances, and preconditioner assumptions visible and
  testable.

Checkpointed unrolled solves
  Use ``jax.checkpoint``/``remat`` or Diffrax recursive checkpoint adjoints for
  fixed-step transient objectives when the time history matters and implicit
  differentiation is not yet justified. Gate memory and runtime separately.

Adaptive-step differentiation
  Treat adaptive branches as a promoted feature only after explicit gates pass.
  Preferred routes are: (1) differentiable fixed-grid replay using the accepted
  adaptive step sequence, (2) forward-mode differentiation through bounded
  ``while_loop``/controller logic for low-dimensional design directions, or
  (3) custom/implicit adjoints for controller-independent accepted trajectories.
  The executable may use a faster non-differentiable adaptive controller; the
  Python research API must expose which adaptive derivative policy is active.

Noisy nonlinear transport objectives
  Use common-random-number finite differences, SPSA/CMA/Bayesian outer loops,
  and replicated late-window statistics until a smoother differentiable
  surrogate passes conditioning and transfer gates. Do not promote an AD
  nonlinear turbulent-flux optimization claim from startup windows or reduced
  proxies.

Method admission gates
  A differentiated observable is accepted only when it records the method,
  static/dynamic arguments, branch or controller assumptions, FD/JVP/VJP or
  tangent checks, conditioning/UQ metadata, and a performance/memory profile
  when runtime claims are made.

Performance And Memory Rules
----------------------------

The package layout should make high performance easier, not harder:

- keep compiled solver stages coarse enough to avoid excessive dispatch and
  recompilation;
- make shape-changing options static and documented;
- use struct-of-arrays or PyTree containers that support ``vmap`` and sharding;
- keep file I/O, progress printing, and plotting outside JIT-compiled kernels;
- expose explicit caches for geometry, gyroaverages, closures, and field solves;
- keep global-vs-local sharded array layouts in typed parallel contracts;
- require serial-vs-parallel identity gates before parallel speedup claims;
- keep memory-heavy diagnostics opt-in and streamed through ``io``/``diagnostics``
  rather than stored inside solver state by default.

Target Source Layout
--------------------

The target root should be mostly packages plus a few user-facing facades:

.. code-block:: text

   src/spectraxgk/
     __init__.py
     _version.py
     api.py                    # optional high-level Python API facade
     linear.py                 # public facade
     nonlinear.py              # public facade
     runtime.py                # public facade
     quasilinear.py            # public facade
     artifacts/plotting.py     # user plotting and figure utilities
     benchmarks.py             # validation facade
     cli/
       main.py
       plot.py
       defaults.py
     core/
       arrays.py
       pytrees.py
       contracts.py
       precision.py
       normalization.py
       extension_points.py
     config/
       model.py
       defaults.py
       toml.py
       validation.py
     basis/
       hermite_laguerre.py
       gyroaverages.py
       transforms.py
       protocols.py
     grids/
       spectral.py
       fieldline.py
       velocity.py
       dealiasing.py
     geometry/
       analytic/
       imported/
       vmec/
       boozer/
       differentiable/
       flux_tube.py
       contracts.py
     physics/
       species.py
       fields.py
       collisions.py
       closures.py
       drives.py
     operators/
       linear/
         rhs.py
         fields.py
         cache.py
       nonlinear/
         bracket.py
         rhs.py
         fields.py
         diagnostics_state.py
       electromagnetic.py
     solvers/
       linear/
         time.py
         eigen.py
         krylov.py
         branch.py
       nonlinear/
         explicit.py
         imex.py
         restart.py
       adjoints.py
       controllers.py
     diagnostics/
       growth.py
       frequency.py
       transport.py
       spectra.py
       windows.py
       uncertainty.py
       schema.py
     objectives/
       linear.py
       quasilinear.py
       nonlinear.py
       stellarator.py
       gradients.py
       gates.py
     optimization/
       stellarator.py
       portfolios.py
       optimizer_ladder.py
       line_search.py
     parallel/
       devices.py
       scans.py
       ensembles.py
       domain.py
       communication.py
       profiling.py
     workflows/
       run.py
       scans.py
       progress.py
       examples.py
       provenance.py
     artifacts/
       io.py
       netcdf.py
       restart.py
       schema.py
     workflows/
       runtime/
         toml.py
     validation/
       literature/
       benchmarks/
       comparisons/
       gates/
       calibration/

This is the target ownership map, not a mandate to create every directory at
once. Empty packages should be avoided. Create a package when a concrete group
of modules is being moved into it.

Naming Rules
------------

Good names should tell a new developer what physics or numerical responsibility
the module owns. They should not encode the migration history.

.. list-table::
   :header-rows: 1

   * - Current pattern
     - Target home
     - Better naming convention
   * - ``nonlinear_*``
     - ``operators/nonlinear`` or ``solvers/nonlinear`` or ``diagnostics``
     - ``rhs``, ``bracket``, ``explicit``, ``imex``, ``transport``
   * - ``runtime_*``
     - ``workflows`` or ``io``
     - ``run``, ``progress``, ``results``, ``artifacts``, ``restart``
   * - ``vmec_jax_*``
     - ``geometry/vmec`` or ``geometry/differentiable``
     - ``bridge``, ``boundary``, ``flux_tube``, ``sensitivity``
   * - ``solver_vmec_boozer_*``
     - ``objectives`` or ``geometry/boozer`` or ``validation/gates``
     - ``geometry_objectives``, ``boozer_bridge``, ``gradient_gates``
   * - ``quasilinear_*``
     - ``objectives/quasilinear`` or ``validation/calibration``
     - ``model``, ``calibration``, ``holdouts``, ``window_metrics``
   * - ``benchmark_*``
     - ``validation/benchmarks``
     - ``cyclone``, ``kbm``, ``stellarator``, ``fit_policy``
   * - ``stellarator_*``
     - ``optimization`` or ``objectives``
     - ``stellarator``, ``transport_objectives``, ``portfolio``
   * - ``*_artifact_*``
     - ``io``
     - ``saved_output``, ``netcdf``, ``schema``, ``restart``

Specific root-level names to stop adding:

- ``runtime_<thing>.py``
- ``nonlinear_<thing>.py``
- ``linear_<thing>.py`` unless it is a deliberate public facade
- ``solver_<thing>.py``
- ``vmec_jax_<thing>.py``
- ``quasilinear_<thing>.py``
- ``benchmark_<thing>.py`` outside ``validation/benchmarks``

Migration Phases
----------------

Phase A: architecture lock
  Add this plan, an architecture manifest, and a checker that fails if new
  root-level prefix modules are added without a migration entry.

Phase B: package skeletons and public facades
  Create only the packages needed for the first concrete moves. Re-export
  public names from existing facades. Add import-identity tests before moving
  implementation.

Phase C: nonlinear package consolidation
  Move nonlinear implementation helpers into ``operators/nonlinear`` and
  ``solvers/nonlinear``. ``spectraxgk.nonlinear`` remains the public nonlinear
  API, while developer imports use ``spectraxgk.operators.nonlinear`` and
  ``spectraxgk.solvers.nonlinear`` directly. The old root nonlinear helper
  shims were removed after package-level tests covered the canonical imports.

Phase D: runtime and output consolidation
  Move runtime orchestration to ``workflows/run.py`` and artifact persistence to
  ``spectraxgk.artifacts``. Keep ``spectraxgk.runtime`` and
  ``spectraxgk.workflows.runtime.artifacts`` as facades. This should make the executable
  path easier to read and keep file I/O out of solver kernels.

Phase E: linear, basis, grids, and operator consolidation
  Move linear cache, linked-boundary helpers, Krylov helpers, moments, and field
  solves under ``operators/linear`` and ``solvers/linear``. Move basis and grid
  primitives into packages with short names and shared contracts.

Phase F: geometry and differentiable-geometry consolidation
  Move imported geometry, VMEC, Boozer, and differentiable bridges under
  ``geometry``. Keep optimization objectives in ``objectives`` rather than
  geometry modules. Keep optional backend discovery isolated from solver code.

Phase G: objectives and optimization consolidation
  Move linear, quasilinear, nonlinear-window, and stellarator objectives into
  ``objectives``. Move optimizer ladders, portfolios, line-search policies, and
  candidate admission logic into ``optimization``. Promoted objectives must
  carry FD/tangent/conditioning tests.

Phase H: validation and benchmark consolidation
  Move benchmark family runners, calibration reports, holdout ledgers,
  literature gates, and comparison adapters into ``validation``. The solver
  packages should not import validation code.

Phase I: tests mirror packages
  Convert top-level test files into package directories:
  ``tests/operators``, ``tests/solvers``, ``tests/geometry``,
  ``tests/objectives``, ``tests/optimization``, ``tests/workflows``,
  ``tests/io``, and ``tests/validation``. Keep a small set of public import
  contract tests at top level.

Phase J: remove migration scaffolding
  After all public imports are covered by facades and docs, remove temporary
  aliases in the next planned API cleanup. This is the only phase that should
  intentionally break undocumented internal imports.

Finite Completion Sequence For The Current Branch
-------------------------------------------------

The active draft PR should finish in a finite sequence rather than continuing
open-ended extraction work.

1. Finish nonlinear consolidation.
   Keep ``spectraxgk.nonlinear`` as the public facade, but move remaining
   scan orchestration, collision-split policy, and diagnostic-output helpers
   into ``operators/nonlinear``, ``solvers/nonlinear``, and diagnostics/io
   packages. Stop once the facade is small enough to document and tests no
   longer need broad cross-module monkeypatches.
2. Consolidate runtime/executable code.
   Move progress, default-demo, plotting dispatch, saved-output handoff, and
   restart orchestration toward ``workflows`` and ``io``. Keep the executable
   user-friendly and non-pure, but keep solver kernels free of I/O.
3. Consolidate objectives and optimization.
   Move solver, quasilinear, nonlinear-window, VMEC/Boozer, optimizer ladder,
   and portfolio logic into ``objectives`` and ``optimization`` packages.
   Add the adaptive-branch differentiation policy gates before claiming
   end-to-end differentiability through adaptive controllers.
4. Consolidate validation and benchmark code.
   Move benchmark-family runners, comparison adapters, calibration ledgers,
   and literature gates under ``validation``. Keep external-code names only in
   benchmark/comparison paths.
5. Mirror tests by package.
   Split the largest top-level tests into package-aligned directories only
   after the source package move is stable, preserving the wide 95% coverage
   gate.
6. Remove temporary migration allowances.
   Shrink ``tools/package_architecture_manifest.toml`` allowed root-prefix
   modules as each family moves. This is the measurable endpoint for the
   simplification lane.

Acceptance Gates
----------------

Every migration tranche must pass:

- import-identity tests for old facade names and new package names;
- focused unit tests for the moved implementation;
- package-wide coverage gate when the tranche touches public behavior;
- docs build with warnings treated as errors;
- release-readiness checks if docs, artifacts, or workflow contracts move;
- benchmark or physics gate checks when a solver, objective, or diagnostic path
  changes;
- FD/JVP/VJP checks when a differentiated objective path changes;
- serial-vs-parallel identity gates when a parallel route changes;
- no new public performance claim unless profiler artifacts are refreshed.

Architecture-specific gates:

- no new root-level prefix module without an explicit migration-manifest row;
- no new public module without a docstring and documented ownership;
- no public package facade with undocumented re-exports;
- no solver or operator module importing validation/comparison adapters;
- no differentiated objective importing executable, plotting, or file-I/O code.

Documentation Requirements
--------------------------

Each public package should have:

- a short module docstring stating responsibility and differentiability status;
- examples of public imports;
- shape and normalization notes for public numerical functions;
- links to the relevant theory/numerics docs;
- links to the tests and validation gates that protect it.

The docs should be organized by audience:

- quickstart and examples for new users;
- equations, normalization, and numerical methods for physics users;
- package architecture and extension points for developers;
- validation matrices and literature gates for reviewers;
- API reference for public facades, not every private helper.

Test Structure Plan
-------------------

The test tree should mirror the source tree:

.. code-block:: text

   tests/
     fixtures/
       geometries.py
       spectra.py
       runtime_outputs.py
     core/
     basis/
     grids/
     geometry/
     physics/
     operators/
     solvers/
     diagnostics/
     objectives/
     optimization/
     parallel/
     workflows/
     io/
     validation/
     import_contracts/

Large top-level tests should be split by behavior, not by original file name.
For example, ``test_runtime_runner.py`` should become focused workflow, config,
progress, artifact-handoff, and executable tests. ``test_benchmarks.py`` should
become validation-family tests under ``tests/validation/benchmarks``.

First Concrete Tranche
----------------------

The next implementation tranche should be deliberately small in API scope but
large enough to establish the new structure:

- create ``operators/nonlinear`` and ``solvers/nonlinear``;
- move the recently extracted nonlinear RHS, diagnostic-state, explicit-step,
  and IMEX modules into those packages;
- keep ``spectraxgk.nonlinear`` as the facade;
- add import-identity tests proving old and new paths return the same objects;
- update API docs and code-structure docs;
- add the architecture checker that prevents new root-level prefix files;
- run focused nonlinear tests, docs, package build, and release-readiness
  checks.

This is the right first tranche because the code was already extracted, the
behavioral seams are known, and it will convert the refactor from "more files at
root" into the target domain-package pattern.

Longer-Term Success Criteria
----------------------------

The refactor is successful when:

- the root package contains only public facades, version metadata, and small
  import-contract modules;
- most implementation code lives in domain packages;
- tests mirror the domain packages;
- public APIs are smaller and easier to document;
- solver kernels remain pure and JAX-transformable;
- runtime and plotting paths are user-friendly but outside differentiable
  kernels;
- new collision operators, basis families, geometry providers, objectives, and
  diagnostics can be added through narrow contracts;
- package-wide coverage, physics gates, benchmark parity, docs, and release
  checks pass without special-case knowledge of the migration history.
