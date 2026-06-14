Code Structure
==============

Purpose
-------

This page documents where the main physics, numerics, runtime, and artifact
surfaces live in the source tree. It is meant to make refactoring safer by
keeping the boundary between public APIs and internal implementation modules
explicit.

Public API vs Internal Modules
------------------------------

Public surfaces that examples, scripts, and external users are expected to rely
on:

- documented module pages in :doc:`api`
- ``spectraxgk.geometry``
- ``spectraxgk.cli``
- ``spectraxgk.runtime``
- ``spectraxgk.runtime_config``
- ``spectraxgk.runtime_artifacts``
- ``spectraxgk.plotting``
- ``spectraxgk.parallel``
- ``spectraxgk.nonlinear_parallel``
- documented benchmark/example scripts under ``examples/`` and ``tools/``

Internal modules that are free to move as long as the public behavior and tests
remain unchanged:

- ``spectraxgk.terms.*``
- ``spectraxgk.runtime_startup``
- ``spectraxgk.runtime_diagnostics``
- ``spectraxgk.runtime_chunks``
- ``spectraxgk.runtime_results``
- ``spectraxgk.from_gx.*``
- low-level geometry adapters and import bridges

Large refactor status for this push: the split runtime, diagnostics,
validation-gate, zonal-validation, and parallelization-policy modules are
documented as behavior-preserving infrastructure. They should not be cited as
new physics validation, production nonlinear optimization, or broad performance
claims unless the corresponding artifact gate in :doc:`release_scope` promotes
that claim separately.

Runtime Flow
------------

The executable-facing runtime path is split conceptually into four layers:

1. **configuration and startup**
   - ``runtime_config.py``
   - ``runtime_startup.py``
2. **solver execution**
   - ``runtime.py``
   - ``linear.py``
   - ``nonlinear.py``
   - ``diffrax_integrators.py``
3. **diagnostics and artifacts**
   - ``diagnostics.py``
   - ``runtime_diagnostics.py``
   - ``runtime_results.py``
   - ``runtime_artifacts.py``
   - ``plotting.py``
4. **benchmark and validation tooling**
   - ``benchmarking.py``
   - ``benchmarks.py``
   - ``tools/*.py``

Physics / Numerics / IO Map
---------------------------

.. list-table::
   :header-rows: 1

   * - Responsibility
     - Primary files
     - Typical tests
   * - Basis and spectral grids
     - ``basis.py``, ``grids.py``
     - orthonormality, indexing, symmetry
   * - Geometry and imported equilibria
     - ``geometry/core.py``, ``miller_eik.py``, ``vmec_eik.py``, ``from_gx/vmec.py``
     - parser, remap, normalization, geometry-response tests
   * - Linear operators and fields
     - ``linear.py``, ``terms/linear_terms.py``, ``terms/fields.py``, ``terms/assembly.py``
     - manufactured solutions, observed-order, eigenfunction and branch tests
   * - Nonlinear operators
     - ``nonlinear.py``, ``terms/nonlinear.py``
     - fixed-mode, diagnostics, collision-split, transport-window tests
   * - Parallelization policy and helpers
     - ``parallel.py``, ``sharding.py``, ``nonlinear_parallel.py``
     - identity gates, one-device fallback, diagnostic-only nonlinear sharding policy
   * - Runtime/executable behavior
     - ``runtime.py``, ``runtime_startup.py``, ``runtime_chunks.py``, ``runtime_results.py``, ``cli.py``
     - runtime contract, startup/restart, output-path, chunking, result assembly, executable smoke tests
   * - Artifacts and plots
     - ``runtime_artifacts.py``, ``plotting.py``
     - serialization, reload, restart append schema, plotting contract tests
   * - Benchmark harness
     - ``benchmarking.py``, ``benchmarks.py``, ``validation_gates.py``, ``zonal_validation.py``
     - late-time/windowed gate tests, reference loading, fallback policy tests

Refactor Mapping
----------------

The current modularization branch is preserving the public runtime surface while
extracting internal responsibilities out of ``runtime.py`` and other large
modules.

Completed extractions:

- startup/loading/initial-condition helpers:
  ``runtime_startup.py``
- runtime diagnostic chunk helpers used by GX-comparison artifacts:
  ``runtime_diagnostics.py``
- adaptive chunk execution used by GX-comparison artifacts:
  ``runtime_chunks.py``
- runtime result containers and nonlinear result assembly:
  ``runtime_results.py``
- validation gate dataclasses and JSON-ready gate helpers:
  ``validation_gates.py``
- zonal-response reference/trace normalization helpers:
  ``zonal_validation.py``
- nonlinear parallelization policy metadata:
  ``nonlinear_parallel.py``
- runtime artifact read/write and restart-append schema coverage:
  ``runtime_artifacts.py``

Next planned extractions:

- benchmark-family runners and fit-policy helpers
- remaining linear result-assembly helpers
- runtime output/artifact handoff helpers

Traceability For Refactors
--------------------------

Refactor work is tracked in ``tools/validation_coverage_manifest.toml``. The
manifest is checked by ``tools/check_validation_coverage_manifest.py`` and
requires every high-priority module to name its source path, owning lane,
reference anchors, physics and numerics contracts, fast tests, artifacts, and
next coverage tests. Update it whenever a source extraction changes module
ownership or validation responsibility.

The manifest also owns the package inventory. Direct rows cover the public or
high-risk refactor modules. Smaller implementation modules are listed in an
owner row's ``owned_modules`` field when their behavior is validated by the
same fast tests and artifacts. Package plumbing such as ``__init__.py`` and
version metadata is the only normal exclusion. Adding a new
``src/spectraxgk/*.py`` file without one of those declarations should fail the
manifest checker.

Use this rule of thumb when changing ownership:

- add a direct row when a module has public imports, independent physics or
  numerics contracts, separate artifact traceability, or high refactor risk;
- add it to ``owned_modules`` when it is a narrow helper split whose contract is
  still fully exercised by the owning row's tests;
- update the owning row's ``next_tests`` when the helper split creates coverage
  debt that is not closed in the same change.

The refactor manifest tests also cross-check public and size-sensitive
surfaces. Every ``.. automodule:: spectraxgk.*`` entry in :doc:`api` must
resolve to a source file or package ``__init__`` that is accounted for by the
manifest, and public package ``__init__`` entries must be explicit exceptions
rather than generic plumbing. The same test requires any non-``__init__``
source module with at least 2,000 non-comment source lines to have a direct
manifest row, so large modules cannot be hidden under another owner's
``owned_modules`` list.

The next differentiable architecture refactor has a separate executable plan in
``tools/differentiable_refactor_manifest.toml`` and
``tools/check_differentiable_refactor_manifest.py``. See
:doc:`differentiable_refactor_plan` for the target package layout, extension
points, public compatibility facades, and physics/autodiff/parity gates that
must be declared before large modules are split.

The first behavior-preserving contract modules are ``spectraxgk.core.contracts``
and ``spectraxgk.core.extension_points``. They introduce typed refactor,
validation-gate, differentiability, and extension-point protocols without
moving solver kernels or changing public numerical behavior.

The first benchmark-helper split keeps ``spectraxgk.benchmark_helpers`` and
``spectraxgk.benchmarks`` as compatibility facades while moving narrow helper
responsibilities into ``spectraxgk.benchmark_initialization`` and
``spectraxgk.benchmark_reference``. Benchmark species-to-``LinearParams``
construction and reference hypercollision/end-damping policy live in
``spectraxgk.benchmark_species``. Fit-signal selection, scan batching, and
solver-selection policies live in ``spectraxgk.benchmark_fit_signals``,
``spectraxgk.benchmark_batching``, and
``spectraxgk.benchmark_solver_policy``. Import-identity tests pin the old
helper symbols to the new modules before larger benchmark-family runners are
moved.

The first differentiable-geometry split keeps
``spectraxgk.geometry.differentiable`` as the public compatibility facade while
moving optional backend lookup and strict AD/finite-difference gate utilities
into ``spectraxgk.geometry.backend_discovery`` and
``spectraxgk.geometry.autodiff_checks``. The solver-ready in-memory flux-tube
mapping and geometry-observable contract lives in
``spectraxgk.geometry.flux_tube_contract``. Pure parity metrics,
interpolation, radial derivative, Boozer half-mesh, Fourier field-line, and
periodic sampling helpers live in ``spectraxgk.geometry.numerics``. This
separates import-side effects, validation-report plumbing, public contract
validation, and small numerical kernels from the VMEC/Boozer field-line bridge.
Analytic, slab, sampled, imported-NetCDF, twist-shift, and grid-default
geometry contracts live in ``spectraxgk.geometry.core``. The
``spectraxgk.geometry`` package remains a thin compatibility facade that
re-exports the same classes and functions for existing user code.
Geometry sensitivity, inverse-design, and local UQ reports live in
``spectraxgk.geometry.sensitivity`` so backend bridge modules can depend on
the report contract without importing the compatibility facade. Bounded VMEC
boundary and Boozer-spectrum bridge checks, Boozer ``|B|`` field-line
evaluation, and Boozer-to-flux-tube sensitivity diagnostics live in
``spectraxgk.geometry.booz_xform_bridge``. Pure helper imports retain object
identity; backend-discovery-dependent bridge functions use thin facade wrappers
so existing monkeypatch-based optional-backend tests still target
``spectraxgk.geometry.differentiable``. VMEC-state-to-Boozer, VMEC metric
tensor, and VMEC field-line tensor AD/FD sensitivity reports live in
``spectraxgk.geometry.vmec_state_sensitivity``; the public facade uses the
same hook-preserving wrapper pattern. Boozer equal-arc constants, cache
prewarm, and core profile construction live in
``spectraxgk.geometry.vmec_boozer_core``. Direct ``vmec_jax`` tensor sampling
and conversion into the solver-ready flux-tube mapping contract lives in
``spectraxgk.geometry.vmec_tensor_mapping``. VMEC flux-tube sensitivity and
array-parity report orchestration lives in
``spectraxgk.geometry.vmec_flux_tube_reports``.

Release-scope synchronization for refactors is tracked separately in
:doc:`release_scope`. In particular, the current restartable NetCDF append
contract normalizes diagnostics loaded from ``*.out.nc`` to the persisted
schema before concatenation; transient in-memory traces that are not stored in
the NetCDF artifact are not treated as release data on continuation.

Testing Taxonomy
----------------

The source tree should be validated through five distinct test classes:

1. **unit tests**
   - cheap, deterministic, local behavior
2. **numerical verification tests**
   - observed-order, manufactured solutions, invariants
3. **benchmark/validation tests**
   - growth rates, frequencies, eigenfunctions, transport windows
4. **autodiff tests**
   - finite-difference and complex-step gradient checks, tangent/adjoint consistency
5. **regression tests**
   - runtime contracts, saved artifacts, reference-lane preservation

Every future source extraction should update this page if module ownership
changes materially.

Repository Artifact Hygiene
---------------------------

The repository should keep reproducible source inputs, tests, small gate
reports, and lightweight documentation previews in Git. Heavy runtime outputs,
profiler traces, raw NetCDF comparisons, and high-resolution publication
exports should live in release artifacts with checksums and replay commands.

Use the non-destructive audit helper before release cleanups:

.. code-block:: bash

   python tools/audit_repository_size.py --top 30
   python tools/check_repository_size_manifest.py
   python tools/check_release_artifact_manifest.py

The report separates tracked file size from ignored local artifact roots such
as ``tools_out/``, ``docs/_build/``, ``dist/``, virtual environments, and caches.
The checked manifest is ``tools/repository_size_manifest.toml``. It defines the
tracked-size budget, the maximum size of any unlisted tracked file, and the
temporary whitelist for any intentionally retained large files.

The release migration manifest is ``tools/release_artifact_manifest.toml``. It
records checksums, replay commands, and planned destinations for high-resolution
panels and other large assets. The checker validates provenance only; it does
not upload or delete artifacts. ``move_to_release`` entries may be absent from
Git after migration, but only when the manifest records the immutable
``release_tag`` and ``release_url`` together with the original size and SHA-256
checksum.

Documentation figures should use lightweight checked-in previews, with
high-resolution publication exports regenerated from the replay commands or
hosted as release assets. The reproducible preview-compression command is:

.. code-block:: bash

   python tools/compress_release_previews.py --max-width 2200 --colors 192
   python tools/compress_docs_previews.py --min-bytes 300000 --max-width 1800 --colors 192

The first command only touches release-manifest previews, so update
``tools/release_artifact_manifest.toml`` with the new sizes and checksums after
running it. The second command skips release-manifest paths by default and is
intended for ordinary checked-in documentation previews. Rerun both manifest
checkers after either cleanup.

History rewrites are not part of routine development; they require a coordinated
maintenance window because every collaborator must reclone or reset local
branches after a force push.
