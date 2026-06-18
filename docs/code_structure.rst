Code Structure
==============

Purpose
-------

This page documents where the main physics, numerics, runtime, and artifact
surfaces live in the source tree. It is meant to make refactoring safer by
keeping the boundary between public APIs and internal implementation modules
explicit.

The long-term consolidation target is documented in
:doc:`architecture_refactor_plan`. New refactor work should move implementation
into domain packages such as ``operators``, ``solvers``, ``objectives``,
``parallel``, ``diagnostics``, ``workflows``, ``artifacts``, and ``validation``
instead of adding more root-level ``runtime_*``, ``nonlinear_*``,
``vmec_jax_*``, ``quasilinear_*``, or ``benchmark_*`` modules.

Public API vs Internal Modules
------------------------------

Public surfaces that examples, scripts, and external users are expected to rely
on:

- documented module pages in :doc:`api`
- ``spectraxgk.artifacts``
- ``spectraxgk.geometry``
- ``spectraxgk.cli``
- ``spectraxgk.runtime``
- ``spectraxgk.workflows.runtime.config``
- ``spectraxgk.workflows.runtime.artifacts``
- ``spectraxgk.artifacts.plotting``
- ``spectraxgk.parallel``
- ``spectraxgk.operators.nonlinear.parallel``
- documented benchmark/example scripts under ``examples/`` and ``tools/``

Internal modules that are free to move as long as the public behavior and tests
remain unchanged:

- ``spectraxgk.terms.*``
- ``spectraxgk.workflows.runtime.startup``
- ``spectraxgk.workflows.runtime.diagnostics``
- ``spectraxgk.workflows.runtime.diagnostic_arrays``
- ``spectraxgk.workflows.runtime.chunks``
- ``spectraxgk.workflows.runtime.results``
- ``spectraxgk.geometry_backends.*``
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
   - ``workflows/runtime/config.py``
   - ``workflows/runtime/startup.py``
   - ``workflows/runtime/policies.py``
2. **solver execution**
   - ``runtime.py``
   - ``linear.py``
   - ``nonlinear.py``
   - ``solvers/time/explicit.py``
   - ``solvers/time/explicit_steps.py``
   - ``solvers/time/explicit_cfl.py``
   - ``solvers/time/explicit_progress.py``
   - ``solvers/time/diffrax.py``
   - ``solvers/time/runners.py``
3. **diagnostics and artifacts**
   - ``diagnostics/core.py``
   - ``diagnostics/energy.py``
   - ``diagnostics/transport.py``
   - ``diagnostics/resolved.py``
   - ``diagnostics/modes.py``
   - ``diagnostics/growth_rates.py``
   - ``workflows/runtime/diagnostics.py``
   - ``workflows/runtime/diagnostic_arrays.py``
   - ``workflows/runtime/results.py``
   - ``workflows/runtime/orchestration.py``
   - ``workflows/runtime/artifacts.py``
   - ``artifacts/``
   - ``artifacts/plotting.py``
4. **executable workflows**
   - ``workflows/linear.py``
   - ``workflows/nonlinear.py``
   - ``workflows/cases.py``
   - ``workflows/demo.py``
   - ``workflows/reduced_models.py``
   - ``cli.py``
5. **benchmark and validation tooling**
   - ``validation/benchmarks/harness.py``
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
     - ``core/velocity.py``, ``core/grid.py``
     - orthonormality, indexing, symmetry
   * - Geometry and imported equilibria
     - ``geometry/boundaries.py``, ``geometry/analytic.py``, ``geometry/flux_tube.py``, ``geometry/core.py``, ``geometry/miller_eik.py``, ``geometry/vmec_eik.py``, ``geometry_backends/vmec.py``
     - parser, remap, normalization, geometry-response tests
   * - Linear operators and fields
     - ``linear.py``, ``operators/linear/rhs.py``, ``operators/linear/``, ``solvers/linear/``, ``terms/linear_terms.py``, ``terms/fields.py``, ``terms/assembly.py``
     - manufactured solutions, observed-order, eigenfunction and branch tests
   * - Solver objectives and eigen-AD gates
     - ``solver_objective_gradients.py``, ``objectives/gradient_gates.py``, ``objectives/vmec_boozer_gradients.py``, ``objectives/core.py``, ``objectives/eigen.py``, ``objectives/sampling.py``, ``objectives/portfolio_contracts.py``, ``objectives/portfolio_sensitivity.py``, ``objectives/portfolio_artifacts.py``, ``objectives/geometry.py``, ``objectives/nonlinear_window.py``, ``objectives/stellarator.py``, ``objectives/stellarator_contracts.py``, ``objectives/stellarator_reduced.py``, ``objectives/vmec_state.py``, ``objectives/vmec_boozer.py``, ``objectives/vmec_boozer_fd.py``, ``objectives/vmec_boozer_line_search.py``
     - core linear/quasilinear observables, implicit eigenpair VJP, branch-locality, sampling-axis, solver-ready and VMEC/Boozer gradient gates, reduced nonlinear-window metrics, VMEC-state coefficient helpers, and finite-difference line-search tests
   * - Nonlinear operators
     - ``nonlinear.py``, ``operators/nonlinear/rhs.py``, ``operators/nonlinear/diagnostic_state.py``, ``operators/nonlinear/diagnostics.py``, ``solvers/nonlinear/explicit.py``, ``solvers/nonlinear/diagnostics.py``, ``solvers/nonlinear/imex.py``, ``terms/brackets.py``, ``terms/gyroaveraging.py``, ``terms/nonlinear.py``
     - RHS routing, bracket payload, explicit stepping, explicit diagnostic orchestration, cached IMEX scan policy, diagnostic tuple assembly, fixed-mode, collision-split, transport-window tests
   * - Parallelization policy and helpers
     - ``parallel.py``, ``sharding.py``, ``operators/nonlinear/parallel.py``, ``operators/nonlinear/parallel_contracts.py``, ``operators/nonlinear/domain_decomposition.py``, ``operators/nonlinear/spectral_core.py``, ``operators/nonlinear/spectral_identity.py``, ``operators/nonlinear/device_z.py``
     - identity gates, one-device fallback, spectral-core work models, logical spectral identity gates, device-z routing gates, diagnostic-only nonlinear sharding policy
   * - Runtime/executable behavior
     - ``runtime.py``, ``workflows/runtime/startup.py``, ``workflows/runtime/policies.py``, ``workflows/runtime/execution.py``, ``workflows/runtime/diagnostics.py``, ``workflows/runtime/diagnostic_arrays.py``, ``workflows/runtime/initial_conditions.py``, ``workflows/runtime/chunks.py``, ``workflows/runtime/results.py``, ``workflows/runtime/orchestration.py``, ``workflows/linear.py``, ``workflows/nonlinear.py``, ``workflows/cases.py``, ``workflows/demo.py``, ``workflows/named_cases.py``, ``workflows/reduced_models.py``, ``cli.py``
     - runtime contract, startup/restart, output-path, full-GK linear/nonlinear workflows, linear-fit diagnostics, quasilinear finalization, diagnostic-array validation/composition, reduced-model workflows, named-case executable workflows, chunking, result assembly, runtime command workflows, executable smoke tests
   * - Diagnostic extraction and growth-rate fitting
     - ``diagnostics/analysis.py``, ``diagnostics/modes.py``, ``diagnostics/growth_rates.py``
     - mode selection, eigenfunction extraction, automatic fit-window selection, late-time growth/frequency tests
   * - Artifacts and plots
     - ``workflows/runtime/artifacts.py``, ``artifacts/``, ``artifacts/spectral_layout.py``, ``artifacts/plot_style.py``, ``artifacts/runtime_plots.py``, ``artifacts/benchmark_plots.py``, ``artifacts/diagnostic_plots.py``, ``artifacts/zonal_plots.py``, ``artifacts/plotting.py``
     - serialization, reload, restart append schema, dealiased-axis contracts, runtime-output plots, benchmark/scan panels, diagnostic/eigenfunction figures, zonal-response figures, plotting contract tests
   * - Benchmark harness
     - ``validation/benchmarks/harness.py``, ``benchmarks.py``, ``validation/benchmarks/cyclone.py``, ``validation/benchmarks/etg.py``, ``validation/benchmarks/kbm.py``, ``validation/benchmarks/kinetic.py``, ``validation/benchmarks/tem.py``, ``validation/gates.py``, ``validation/zonal.py``
     - late-time/windowed gate tests, reference loading, fallback policy tests

Refactor Mapping
----------------

The current modularization branch is preserving the public runtime surface while
extracting internal responsibilities out of ``runtime.py`` and other large
modules.

Completed extractions:

- zero-shear boundary promotion, analytic s-alpha/slab geometry models, and
  sampled/imported flux-tube geometry data/loading:
  ``geometry/boundaries.py``, ``geometry/analytic.py``, and
  ``geometry/flux_tube.py``
- mode selection/eigenfunction extraction and late-time growth/frequency
  fitting:
  ``diagnostics/modes.py`` and ``diagnostics/growth_rates.py``. The public
  ``diagnostics.analysis`` module remains a small compatibility facade.
- scalar energy, species transport/heating, and resolved spectral diagnostics:
  ``diagnostics/energy.py``, ``diagnostics/transport.py``, and
  ``diagnostics/resolved.py``. The public ``diagnostics.core`` module remains
  a compatibility facade re-exported by ``spectraxgk.diagnostics``.
- explicit linear step kernels, explicit CFL/frequency-bound policy, and
  progress formatting:
  ``solvers/time/explicit_steps.py``, ``solvers/time/explicit_cfl.py``, and
  ``solvers/time/explicit_progress.py``. The public
  ``solvers.time.explicit`` module remains the import facade for existing
  debug tools and tests.
- startup/loading/initial-condition helpers:
  ``workflows/runtime/startup.py``
- runtime mode-index, nonlinear step-count, external-source, parallel-scan,
  and nonlinear diagnostics keyword policies:
  ``workflows/runtime/policies.py``
- runtime linear fit/eigenfunction extraction and quasilinear finalization:
  ``workflows/runtime/diagnostics.py``
- finite-value checks plus runtime diagnostic slicing, truncation, striding,
  and concatenation:
  ``workflows/runtime/diagnostic_arrays.py``
- adaptive chunk execution used by runtime and comparison artifacts:
  ``workflows/runtime/chunks.py``
- runtime result containers and nonlinear result assembly:
  ``workflows/runtime/results.py``
- runtime progress formatting, combined-``ky`` scan batching, serial/worker
  scan orchestration, and nonlinear artifact handoff policy:
  ``workflows/runtime/orchestration.py``
- full-GK executable linear runtime workflow:
  ``workflows/linear.py``
- full-GK executable nonlinear runtime workflow:
  ``workflows/nonlinear.py``
- executable reduced-model runtime workflows:
  ``workflows/reduced_models.py``
- shared plot style plus runtime-output, benchmark/scan, diagnostic, and
  zonal-response figure families:
  ``artifacts/plot_style.py``, ``artifacts/runtime_plots.py``,
  ``artifacts/benchmark_plots.py``, ``artifacts/diagnostic_plots.py``, and
  ``artifacts/zonal_plots.py``. The public ``artifacts.plotting`` module
  remains a stable import facade for examples and user scripts.
- validation gate dataclasses and JSON-ready gate helpers:
  ``validation/gates.py``
- zonal-response reference/trace normalization helpers:
  ``validation/zonal.py``
- dominant-eigenvalue custom VJP and branch-locality diagnostics:
  ``objectives/eigen.py``
- core solver-objective constants plus value-level linear/quasilinear
  observables:
  ``objectives/core.py``
- solver-objective sampling axes, physical-``ky`` grid mapping, and aggregate
  weights:
  ``objectives/sampling.py``
- solver-ready geometry objective gates, reduced nonlinear-window metrics,
  solver-ready gradient gates, mode-21 VMEC/Boozer gradient gates,
  backend-free portfolio row/weight contracts, portfolio AD/FD sensitivity
  gates, artifact promotion guards, VMEC/Boozer state coefficient helpers,
  VMEC/Boozer objective-table plumbing, and VMEC/Boozer finite-difference/
  line-search gates:
  ``objectives/geometry.py``, ``objectives/nonlinear_window.py``,
  ``objectives/gradient_gates.py``,
  ``objectives/vmec_boozer_gradients.py``,
  ``objectives/portfolio_contracts.py``,
  ``objectives/portfolio_sensitivity.py``,
  ``objectives/portfolio_artifacts.py``, ``objectives/vmec_state.py``,
  ``objectives/vmec_boozer.py``,
  ``objectives/vmec_boozer_fd.py``,
  ``objectives/vmec_boozer_line_search.py``. The public
  ``objectives.stellarator_portfolio`` module remains a small facade for
  existing imports.
- nonlinear parallelization policy metadata, local domain prototypes, and
  spectral-core work models/RHS primitives plus device-z shard-map routes:
  ``operators/nonlinear/parallel.py``, ``operators/nonlinear/parallel_contracts.py``,
  ``operators/nonlinear/domain_decomposition.py``,
  ``operators/nonlinear/spectral_core.py``,
  ``operators/nonlinear/device_z.py``
- nonlinear RHS composition and state-to-diagnostic tuple assembly:
  ``operators/nonlinear/rhs.py`` and
  ``operators/nonlinear/diagnostic_state.py``. The obsolete root nonlinear
  helper shims were removed; normal users should use ``spectraxgk.nonlinear``
  and developer helpers should import from ``spectraxgk.operators.nonlinear``.
- explicit RK/SSP/K10 one-step policy, cached explicit scan policy, explicit
  diagnostic step and scan-selection policy, explicit/IMEX diagnostic integration
  orchestration, cached IMEX scan policy, IMEX diagnostic step and
  scan-execution policy, and IMEX fixed-point/GMRES solve policy:
  ``solvers/nonlinear/explicit.py``, ``solvers/nonlinear/diagnostics.py``, and
  ``solvers/nonlinear/imex.py``. Developer helpers should import from
  ``spectraxgk.solvers.nonlinear``.
- linear cache, linked-boundary maps, Hermite-Laguerre moments, parameter
  pytrees, cache-backed RHS assembly, implicit linear GMRES/preconditioner
  policy, fixed-step/diagnostic integration policy, Krylov eigensolver policy,
  and velocity-parallel RHS dispatch live under ``operators/linear/`` and
  ``solvers/linear/``. The obsolete root ``linear_*`` helper shims were
  removed; normal users should use ``spectraxgk.linear`` for the public linear
  API or import focused developer helpers from the domain packages.
- nonlinear turbulence-gradient follow-up shared configs, JSON parsing, and
  candidate design, composite-control, matched-replicate, QL-seed,
  state-runbook, and variance-reduction/control-variate report helpers:
  ``validation/nonlinear_gradient/followup_core.py``,
  ``validation/nonlinear_gradient/followup_candidate.py``,
  ``validation/nonlinear_gradient/followup_composite.py``,
  ``validation/nonlinear_gradient/followup_plan.py``,
  ``validation/nonlinear_gradient/followup_ql_seed.py``,
  ``validation/nonlinear_gradient/followup_state_runbook.py``,
  ``validation/nonlinear_gradient/followup_variance.py``
- nonlinear turbulence-gradient evidence scope markers, acceptance config
  dataclasses, JSON-safe parsing, finite-difference conditioning gates,
  artifact classification, replicated window summaries, central
  finite-difference report assembly, candidate/bracket screening reports, and
  production evidence-gap report orchestration:
  ``validation/nonlinear_gradient/evidence_core.py``,
  ``validation/nonlinear_gradient/evidence_classification.py``,
  ``validation/nonlinear_gradient/evidence_windows.py``,
  ``validation/nonlinear_gradient/evidence_fd.py``,
  ``validation/nonlinear_gradient/evidence_screening.py``,
  ``validation/nonlinear_gradient/evidence_gap.py``
- runtime artifact read/write, generic I/O helpers, linear/quasilinear
  artifact writers, generic nonlinear table writers, dealiased-axis
  layout, NetCDF schema writing, nonlinear diagnostic reload helpers,
  and restart-append schema coverage:
  ``workflows/runtime/artifacts.py``, ``artifacts/io.py``,
  ``artifacts/linear.py``,
  ``artifacts/nonlinear.py``,
  ``artifacts/spectral_layout.py``,
  ``artifacts/nonlinear_netcdf.py``,
  ``artifacts/nonlinear_diagnostics.py``. The obsolete root
  ``runtime_artifact_*`` helper modules were removed; import implementation
  helpers from ``spectraxgk.artifacts`` instead.

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

The authoritative target package layout, naming policy, and conflict-resolution
rules live in :doc:`architecture_refactor_plan`. The executable migration ledger
is ``tools/differentiable_refactor_manifest.toml``, checked by
``tools/check_differentiable_refactor_manifest.py``. See
:doc:`differentiable_refactor_plan` for differentiability contracts, extension
points, active manifest rows, and physics/autodiff/parity gates. If this page or
the manifest conflicts with :doc:`architecture_refactor_plan`, update the
current-tree documentation or manifest rather than adding another root-level
prefix module.

The first behavior-preserving contract modules are ``spectraxgk.core.contracts``
and ``spectraxgk.core.extension_points``. They introduce typed refactor,
validation-gate, differentiability, and extension-point protocols without
moving solver kernels or changing public numerical behavior.

Runtime command dispatch now keeps parser construction in ``spectraxgk.cli`` and moves runtime linear, runtime scan, and runtime nonlinear command execution into ``spectraxgk.workflows.cases``. The CLI facade still exposes compatibility helper names for tests and downstream scripts, but path override, progress, and quasilinear override policies have one workflow owner.

The benchmark helper split now uses focused domain modules directly.
Benchmark initial conditions and reference data live in
``spectraxgk.validation.benchmarks.initialization`` and
``spectraxgk.validation.benchmarks.reference``. Benchmark species-to-``LinearParams``
construction and reference hypercollision/end-damping policy live in
``spectraxgk.validation.benchmarks.species``. Fit-signal selection, scan batching, and
solver-selection policies live in ``spectraxgk.validation.benchmarks.fit_signals``,
``spectraxgk.validation.benchmarks.batching``, and
``spectraxgk.validation.benchmarks.solver_policy``. Import-identity tests pin the old
helper symbols to the new modules before larger benchmark-family runners are
moved. The KBM benchmark family runner now lives in
``spectraxgk.validation.benchmarks.kbm`` while ``spectraxgk.benchmarks`` remains the
public compatibility facade for ``run_kbm_linear``, ``run_kbm_scan``, and
``run_kbm_beta_scan``. The TEM benchmark family follows the same pattern in
``spectraxgk.validation.benchmarks.tem`` for ``run_tem_linear`` and ``run_tem_scan``.
Kinetic-electron ITG/TEM runners are in ``spectraxgk.validation.benchmarks.kinetic`` with
the same public facade guarantees. ETG runners are in
``spectraxgk.validation.benchmarks.etg`` for ``run_etg_linear`` and ``run_etg_scan``;
Cyclone runners are in ``spectraxgk.validation.benchmarks.cyclone`` for
``run_cyclone_linear`` and ``run_cyclone_scan``. Family-specific branch tests
patch those implementation modules directly while examples and downstream
scripts keep importing through ``spectraxgk.benchmarks``.

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
Zero-shear boundary policy lives in ``spectraxgk.geometry.boundaries``.
Analytic s-alpha and slab geometry models live in
``spectraxgk.geometry.analytic``. Sampled solver-ready geometry data, analytic
sampling, imported-NetCDF loading, and periodic mirror-term reconstruction live
in ``spectraxgk.geometry.flux_tube``. Twist-shift and grid-default geometry
policy live in ``spectraxgk.geometry.core``. The
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
