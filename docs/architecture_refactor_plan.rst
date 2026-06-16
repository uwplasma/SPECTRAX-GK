Architecture Refactor Plan
==========================

Purpose
-------

This plan resets the refactor direction for SPECTRAX-GK. The recent
compatibility-preserving extractions reduced several large files, but they also
created too many root-level modules with prefix-based names. That makes the code
harder to navigate, harder to teach, harder to extend, and harder to test.

The new target is a small set of domain packages with stable public facades,
private implementation modules, explicit contracts, and tests that mirror the
package structure. The goal is not minimal file count at all costs. The goal is
clear ownership, research-grade validation, strong JAX transformability, and a
code layout that a new developer can understand without knowing the historical
refactor sequence.

Planning Snapshot
-----------------

The planning audit on 2026-06-16 found:

- 167 Python source files under ``src/spectraxgk``.
- about 70,000 source lines under ``src/spectraxgk``.
- 134 root-level modules in ``src/spectraxgk``.
- 315 top-level test files under ``tests``.
- about 89,000 test lines.
- many prefix families at package root:
  ``runtime_*``, ``nonlinear_*``, ``solver_*``, ``vmec_jax_*``,
  ``quasilinear_*``, ``benchmark_*``, and ``stellarator_*``.

That structure is a transition state, not the desired architecture. Future
splits should not add more root-level prefix modules unless they are temporary
compatibility facades tracked in the migration manifest.

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
- `Dedalus <https://dedalus-project.readthedocs.io/>`_ is a useful reference
  for spectral-method software separating equations, bases/domains, solvers,
  and analysis tasks.
- `PlasmaPy <https://docs.plasmapy.org/>`_ is a useful reference for a
  community plasma-code documentation surface that distinguishes user guides,
  API references, examples, and developer guidance.
- `SimPEG <https://docs.simpeg.xyz/>`_ is a useful reference for separating
  simulations, maps, objective functions, inversions, regularization, and
  directives in an optimization-oriented scientific package.

These references point to the same practical rule: user workflows should be
simple, but internals should be separated by responsibility, not by historical
file prefix.

Non-Negotiable Constraints
--------------------------

- Preserve documented user behavior during the refactor.
- Keep public imports as facades until an intentional major-version cleanup.
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
  ``spectraxgk.benchmarks`` remain compatibility facades while internals move.

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

Target Source Layout
--------------------

The target root should be mostly packages plus a few user-facing facades:

.. code-block:: text

   src/spectraxgk/
     __init__.py
     _version.py
     api.py                    # optional high-level Python API facade
     linear.py                 # compatibility facade
     nonlinear.py              # compatibility facade
     runtime.py                # compatibility facade
     quasilinear.py            # compatibility facade
     plotting.py               # user plotting facade
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
     io/
       artifacts.py
       netcdf.py
       restart.py
       toml.py
       schema.py
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
- ``linear_<thing>.py`` unless it is a compatibility facade
- ``solver_<thing>.py``
- ``vmec_jax_<thing>.py``
- ``quasilinear_<thing>.py``
- ``benchmark_<thing>.py`` outside ``validation/benchmarks``

Migration Phases
----------------

Phase A: architecture lock
  Add this plan, an architecture manifest, and a checker that fails if new
  root-level prefix modules are added without a migration entry.

Phase B: package skeletons and compatibility facades
  Create only the packages needed for the first concrete moves. Re-export
  public names from existing facades. Add import-identity tests before moving
  implementation.

Phase C: nonlinear package consolidation
  Move the already extracted nonlinear implementation modules into
  ``operators/nonlinear`` and ``solvers/nonlinear``:
  ``nonlinear_rhs.py`` to ``operators/nonlinear/rhs.py``,
  ``nonlinear_diagnostic_state.py`` to
  ``operators/nonlinear/diagnostics_state.py``,
  ``nonlinear_explicit_step.py`` to ``solvers/nonlinear/explicit.py``, and
  ``nonlinear_imex.py`` to ``solvers/nonlinear/imex.py``. Keep
  ``spectraxgk.nonlinear`` as the public facade.

Phase D: runtime and output consolidation
  Move runtime orchestration to ``workflows/run.py`` and output persistence to
  ``io``. Keep ``spectraxgk.runtime`` and ``spectraxgk.runtime_artifacts`` as
  facades. This should make the executable path easier to read and keep file
  I/O out of solver kernels.

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
  ``tests/io``, and ``tests/validation``. Keep a small set of compatibility
  import tests at top level.

Phase J: remove migration scaffolding
  After all public imports are covered by facades and docs, remove temporary
  aliases in the next planned API cleanup. This is the only phase that should
  intentionally break undocumented internal imports.

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
     compatibility/

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
  compatibility modules;
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
