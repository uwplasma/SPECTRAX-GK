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
``parallel``, ``diagnostics``, ``workflows``, and ``artifacts`` instead of
adding more root-level ``runtime_*``, ``nonlinear_*``, ``vmec_jax_*``,
``quasilinear_*``, or ``benchmark_*`` modules. The ``validation`` package is a
temporary compatibility and metrics surface during the consolidation; campaign
and benchmark-branch implementation should move to root ``benchmarks``,
``tools``, or ``tests/validation`` rather than deeper installable validation
subpackages.

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
- documented benchmark/example scripts under ``benchmarks/`` and ``examples/``
- documented repository-maintenance entry points under purpose-specific
  ``tools/`` subfolders

Internal modules that are free to move as long as the public behavior and tests
remain unchanged:

- ``spectraxgk.terms.*``
- ``spectraxgk.workflows.runtime.startup``
- ``spectraxgk.workflows.runtime.diagnostics``
- ``spectraxgk.workflows.runtime.diagnostic_arrays``
- ``spectraxgk.workflows.runtime.initial_phi``
- ``spectraxgk.workflows.runtime.chunks``
- ``spectraxgk.workflows.runtime.results``
- ``spectraxgk.workflows.runtime.commands``
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
   - ``solvers/time/diffrax.py`` facade plus ``solvers/time/diffrax_*`` owner modules
   - ``solvers/time/runners.py``
3. **diagnostics and artifacts**
   - ``diagnostics/core.py``
   - ``diagnostics/energy.py``
   - ``diagnostics/transport.py``
   - ``diagnostics/resolved.py``
   - ``diagnostics/modes.py``
   - ``diagnostics/growth_rates.py``
   - ``diagnostics/growth_fit.py``
   - ``diagnostics/growth_windows.py``
   - ``diagnostics/growth_series.py``
   - ``workflows/runtime/diagnostics.py``
   - ``workflows/runtime/diagnostic_arrays.py``
   - ``workflows/runtime/results.py``
   - ``workflows/runtime/orchestration.py`` facade plus ``workflows/runtime/orchestration_scan.py``, ``workflows/runtime/orchestration_progress.py``, and ``workflows/runtime/orchestration_artifacts.py``
   - ``workflows/runtime/artifacts.py``
   - ``artifacts/``
   - ``artifacts/plotting.py``
4. **executable workflows**
   - ``workflows/runtime/commands.py``
   - ``workflows/linear.py``
   - ``workflows/nonlinear.py``
   - ``workflows/cases.py``
   - ``workflows/demo.py``
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
     - ``geometry/boundaries.py``, ``geometry/analytic.py``, ``geometry/flux_tube.py``, ``geometry/core.py``, ``geometry/miller_eik.py``, ``geometry/vmec_eik.py``, ``geometry_backends/miller.py`` and ``geometry_backends/vmec.py`` facades plus focused Miller and VMEC backend modules
     - parser, remap, normalization, geometry-response tests, Miller/VMEC finite-difference geometry and NetCDF writeout gates
   * - Linear operators and fields
     - ``linear.py``, ``operators/linear/rhs.py``, ``operators/linear/cache_builder.py``, ``operators/linear/``, ``solvers/linear/``, ``terms/linear_terms.py``, ``terms/fields.py``, ``terms/assembly.py`` facade plus ``terms/assembly_*`` owner modules
     - manufactured solutions, observed-order, eigenfunction and branch tests; cache-builder tests cover staged grid, geometry, twist-shift, gyro/moment, drift, and linked-boundary packing
   * - Solver objectives and eigen-AD gates
     - ``solver_objective_gradients.py`` facade, ``objectives/solver_vmec.py``, ``objectives/solver_gradient_reports.py``, ``objectives/gradient_gates.py``, ``objectives/vmec_boozer_gradients.py``, ``objectives/vmec_boozer_context.py``, ``objectives/core.py``, ``objectives/eigen.py``, ``objectives/sampling.py``, ``objectives/portfolio_contracts.py``, ``objectives/portfolio_sensitivity.py``, ``objectives/portfolio_artifacts.py``, ``objectives/geometry.py``, ``objectives/nonlinear_window.py``, ``objectives/stellarator.py``, ``objectives/stellarator_tables.py``, ``objectives/stellarator_residuals.py``, ``objectives/stellarator_contracts.py``, ``objectives/stellarator_reduced.py``, ``objectives/qa_low_turbulence.py``, ``objectives/qa_low_turbulence_contracts.py``, ``objectives/qa_low_turbulence_model.py``, ``objectives/qa_low_turbulence_residuals.py``, ``objectives/qa_low_turbulence_optimizer.py``, ``objectives/qa_low_turbulence_artifacts.py``, ``objectives/vmec_state.py``, ``objectives/vmec_transport.py``, ``objectives/vmec_transport_config.py``, ``objectives/vmec_transport_tables.py``, ``objectives/vmec_transport_branch.py``, ``objectives/vmec_boozer.py``, ``objectives/vmec_boozer_fd.py``, ``objectives/vmec_boozer_line_search.py``
     - core linear/quasilinear observables, implicit eigenpair VJP, branch-locality, sampling-axis, shared solver-ready linear context for gradient gates, solver-ready and VMEC/Boozer gradient gates, VMEC/Boozer context/feature extraction, reduced nonlinear-window metrics, stellarator ITG table/residual gates, reduced QA low-turbulence model/residual/artifact gates, VMEC transport objective adapters, staged VMEC growth-branch locality setup/sample evaluation/report packing, VMEC-state coefficient helpers, and finite-difference line-search tests
   * - Nonlinear operators
     - ``nonlinear.py`` facade, ``solvers/nonlinear/state_integration.py``, ``solvers/nonlinear/diagnostic_integration.py``, ``operators/nonlinear/rhs.py``, ``operators/nonlinear/diagnostic_state.py``, ``operators/nonlinear/diagnostics.py``, ``operators/nonlinear/projection.py``, ``operators/nonlinear/collisions.py``, ``solvers/nonlinear/explicit.py``, ``solvers/nonlinear/diagnostics.py``, ``solvers/nonlinear/imex.py``, ``solvers/nonlinear/imex_diagnostics.py``, ``terms/brackets.py``, ``terms/gyroaveraging.py``, ``terms/nonlinear.py``
     - RHS routing, bracket payload, explicit stepping, explicit diagnostic orchestration, IMEX diagnostic orchestration, cached IMEX operator/state policy, diagnostic tuple assembly, fixed-mode and Hermitian projection, collision-split, staged electrostatic/electromagnetic nonlinear contribution helpers, transport-window tests
   * - Parallelization policy and helpers
     - ``parallel.py``, ``sharding.py``, ``parallel/velocity.py``, ``parallel/velocity_plan.py``, ``parallel/velocity_hermite.py``, ``parallel/velocity_streaming.py``, ``parallel/velocity_drive.py``, ``operators/nonlinear/parallel.py``, ``operators/nonlinear/parallel_contracts_domain.py``, ``operators/nonlinear/parallel_contracts_spectral.py``, ``operators/nonlinear/parallel_contracts_strategy.py``, ``operators/nonlinear/domain_decomposition.py``, ``operators/nonlinear/spectral_core.py``, ``operators/nonlinear/spectral_state.py``, ``operators/nonlinear/spectral_layout.py``, ``operators/nonlinear/spectral_work_models.py``, ``operators/nonlinear/spectral_brackets.py``, ``operators/nonlinear/spectral_tolerances.py``, ``operators/nonlinear/spectral_identity_reports.py``, ``operators/nonlinear/spectral_identity_rhs.py``, ``operators/nonlinear/spectral_identity_integrator.py``, ``operators/nonlinear/device_z.py``
     - identity gates, one-device fallback, velocity-space plan/exchange/streaming/field-reduction microkernels, domain/spectral/strategy contracts, spectral state/layout/work-model/bracket/tolerance helpers, logical spectral reports/RHS/integrator gates, device-z routing gates with explicit RHS and transport-trace/report policies, diagnostic-only nonlinear sharding policy
   * - Runtime/executable behavior
     - ``runtime.py``, ``workflows/runtime/startup.py``, ``workflows/runtime/policies.py``, ``workflows/runtime/execution.py``, ``workflows/runtime/diagnostics.py``, ``workflows/runtime/diagnostic_arrays.py``, ``workflows/runtime/initial_conditions.py``, ``workflows/runtime/initial_phi.py``, ``workflows/runtime/chunks.py``, ``workflows/runtime/results.py``, ``workflows/runtime/orchestration.py``, ``workflows/runtime/orchestration_artifacts.py``, ``workflows/runtime/commands.py``, ``workflows/linear.py``, ``workflows/nonlinear.py``, ``workflows/cases.py``, ``workflows/demo.py``, ``workflows/named_cases.py``, ``cli.py``
     - runtime contract, startup/restart, output-path, full-GK linear/nonlinear workflows, runtime TOML case dependency defaults, saved-output plot command routing, executable artifact path display and progress/summary printing, linear-fit diagnostics, electrostatic-potential initializers, quasilinear finalization, diagnostic-array validation/composition, named-case executable workflows, chunking, result assembly, runtime command workflows, executable smoke tests
   * - Public import registry
     - ``api/configuration.py``, ``api/geometry.py``, ``api/diagnostics.py``, ``api/runtime.py``, ``api/solvers.py``, ``api/benchmarks.py``, ``api/validation.py``, ``api/parallel.py``, ``api/objectives.py``, ``api/artifacts.py``
     - top-level ``spectraxgk`` export membership/order checks, public-object identity tests, API documentation build
   * - Diagnostic extraction and growth-rate fitting
     - ``diagnostics/analysis.py``, ``diagnostics/modes.py``, ``diagnostics/growth_rates.py``, ``diagnostics/growth_fit.py``, ``diagnostics/growth_windows.py``, ``diagnostics/growth_series.py``, ``diagnostics/quasilinear_transport.py``
     - mode selection, eigenfunction extraction, least-squares growth/frequency fitting, automatic fit-window selection, quasilinear transport weights and saturation helpers, late-time growth/frequency tests
   * - Artifacts and plots
     - ``workflows/runtime/artifacts.py``, ``artifacts/``, ``artifacts/spectral_layout.py``, ``artifacts/plot_style.py``, ``artifacts/runtime_plots.py``, ``artifacts/benchmark_plots.py``, ``artifacts/diagnostic_plots.py``, ``artifacts/zonal_plots.py``, ``artifacts/plotting.py``
     - serialization, reload, restart append schema, dealiased-axis contracts, runtime-output plots, benchmark/scan panels, diagnostic/eigenfunction figures, zonal-response figures, plotting contract tests
   * - Benchmark harness
     - ``config.py``, ``validation/benchmarks/harness.py``, ``validation/benchmarks/harness_metrics.py``, ``validation/benchmarks/harness_scan.py``, ``benchmarks.py``, ``validation/benchmarks/cyclone_linear.py``, ``validation/benchmarks/cyclone_scan.py``, ``validation/benchmarks/etg_linear.py``, ``validation/benchmarks/etg_scan.py``, ``validation/benchmarks/kbm_beta.py``, ``validation/benchmarks/kbm_linear.py``, ``validation/benchmarks/kbm_scan.py``, ``validation/benchmarks/kinetic_linear.py``, ``validation/benchmarks/kinetic_scan.py``, ``validation/benchmarks/tem.py`` plus ``validation/benchmarks/tem_paths.py``, ``diagnostics/modes.py``, ``diagnostics/validation_gates.py``, ``diagnostics/zonal_validation.py``
     - late-time/windowed gate tests, eigenfunction reference/phase utilities, diagnostics time-series loading, benchmark case presets, physics metric extraction, scan/eigenmode orchestration, reference loading, fallback policy tests

Refactor Mapping
----------------

The current modularization branch is preserving the public runtime surface while
extracting internal responsibilities out of ``runtime.py`` and other large
modules.

Completed extractions:

- domain-organized public API registry in ``spectraxgk.api.*``. The root
  ``spectraxgk`` package is now a small facade that keeps a stable documented
  ``__all__`` order while the grouped API modules make the exported
  configuration, geometry, solver, validation, parallelization, objective, and
  plotting surfaces easier to audit.
- zero-shear boundary promotion, analytic s-alpha/slab geometry models, and
  sampled/imported flux-tube geometry data/loading:
  ``geometry/boundaries.py``, ``geometry/analytic.py``, and
  ``geometry/flux_tube.py``. Imported NetCDF/eik loading keeps schema
  selection, scalar/profile reads, root-level terminal-theta inference,
  mirror-term reconstruction, drift/Jacobian normalization, and
  ``FluxTubeGeometryData`` packing as separate private stages so geometry-file
  variants can be tested without one large loader body.
- focused imported-geometry backends. ``geometry_backends.miller`` and
  ``geometry_backends.vmec`` are now stable facades, while numerics,
  field-line/core assembly, remap, IO, optional-backend discovery, and pipeline
  ownership live in smaller ``geometry_backends.miller_*`` and
  ``geometry_backends.vmec_*`` modules. Imported Miller profile assembly keeps
  central-surface normalization, period extension, Bishop coefficients, metric
  coefficients, magnetic drifts, target-grid interpolation, ballooning
  conversion, and final EIK profile packing as explicit stages inside
  ``geometry_backends.miller_profiles``.
- mode selection/eigenfunction extraction and late-time growth/frequency
  fitting:
  ``diagnostics/modes.py``, ``diagnostics/growth_rates.py``,
  ``diagnostics/growth_fit.py``, ``diagnostics/growth_windows.py``, and
  ``diagnostics/growth_series.py``. The public ``diagnostics.analysis`` and
  ``diagnostics.growth_rates`` modules remain small facades over focused
  diagnostic owners. Fit-window selection keeps argument validation,
  least-squares scoring, amplitude/slope thresholds, candidate-window search,
  and fallback policy as named stages inside ``diagnostics/growth_windows.py``
  so benchmark auto-windowing can be tested without duplicating fit logic.
- scalar energy, species transport/heating, and resolved spectral diagnostics:
  ``diagnostics/energy.py``, ``diagnostics/transport.py``, and
  ``diagnostics/resolved.py``. The public ``diagnostics.core`` module remains
  a small facade re-exported by ``spectraxgk.diagnostics``.
- explicit linear step kernels, diagnostics-rich linear IVP integration,
  explicit CFL/frequency-bound policy, and progress formatting:
  ``solvers/time/explicit_steps.py``, ``solvers/time/explicit_diagnostics.py``,
  ``solvers/time/explicit_cfl.py``, and ``solvers/time/explicit_progress.py``. The public
  ``solvers.time.explicit`` module remains the import facade for existing
  debug tools and tests. Its linear IVP facade now keeps method validation,
  adaptive CFL timing, JIT stepper construction, sample-history collection,
  progress emission, and array packaging as named private stages. The
  diagnostics owner separately stages method/time-policy validation, JIT stepper
  construction, energy/transport sampling, progress rendering, and
  ``SimulationDiagnostics`` construction so saved explicit-time benchmark paths
  exercise named numerical pieces instead of one monolithic loop.
- Diffrax time-integration internals. ``solvers/time/diffrax.py`` remains the
  public facade while optional dependency/policy helpers, linear save-path
  integration, streaming growth/frequency fits, and nonlinear integration live
  in ``solvers/time/diffrax_core.py``, ``diffrax_linear.py``,
  ``diffrax_streaming.py``, and ``diffrax_nonlinear.py``. The streaming owner
  stages state/cache preparation, monitored-mode extraction, optimized
  density-mode extraction, weighted log-derivative accumulation, Diffrax RHS
  construction, IMEX zero-term routing, and saved-result finalization so
  differentiable streaming fits can be audited without changing the public
  ``integrate_linear_diffrax_streaming`` contract.
  The saved-field linear owner stages state/cache preparation, packed-state
  sharding, RHS construction, save-field/mode extraction, save-time policy,
  Diffrax solve execution, and final unpacking while keeping the public
  ``integrate_linear_diffrax`` contract stable.
  The fixed-step cached linear integrator stages method validation,
  state/damping preparation, serial/parallel RHS routing, RK/IMEX/SSPX3
  stepping, progress callbacks, and sample-stride scans separately while
  keeping the public ``integrate_linear`` and donated-buffer wrappers stable.
  The nonlinear owner stages state/cache preparation, packed-state sharding,
  linear and nonlinear RHS construction, IMEX term routing, saved-``phi``
  extraction, solve execution, and final ``FieldState`` packing while keeping
  the public ``integrate_nonlinear_diffrax`` contract stable.
- term-wise RHS assembly internals. ``terms/assembly.py`` remains the public
  facade while cached RHS composition, per-term diagnostic decomposition,
  field-only solves, and shared helper policies live in
  ``terms/assembly_core.py``, ``terms/assembly_diagnostics.py``,
  ``terms/assembly_fields.py``, and ``terms/assembly_helpers.py``. The
  production RHS and diagnostic decomposition share one helper-owned staging
  layer for state/species normalization, scalar parameter expansion, field and
  Hamiltonian construction, drift/drive/dissipation contribution assembly,
  fixed-order term summation, and species-axis restoration. This keeps
  ``assemble_rhs_cached`` and ``assemble_rhs_terms_cached`` numerically aligned
  without duplicating RHS policy. Linear
  electromagnetic field solves in ``terms/fields.py`` are staged as
  coefficient casting, gyrocenter moments, electrostatic potential,
  compressional ``bpar`` coupling, parallel ``apar`` solve, and final
  ``FieldState`` packing, while the public custom-VJP wrapper keeps the same
  differentiability boundary. Linear
  contribution kernels keep Hermite-mode drive insertion centralized in
  ``terms/linear_terms.py`` so streaming, diamagnetic, and collision
  corrections share one reviewed convention.
- nonlinear public-driver internals. ``nonlinear.py`` remains the public
  facade while cached RHS/state integration lives in ``solvers/nonlinear/state_integration.py`` and
  explicit/IMEX diagnostic entry points live in ``solvers/nonlinear/diagnostic_integration.py``.
  Lower-level nonlinear RHS, diagnostic-state, policy, explicit-step, explicit
  and IMEX diagnostic scan preparation/finalization, and IMEX mechanics remain
  owned by ``operators/nonlinear/*`` and
  ``solvers/nonlinear/*`` modules. The explicit nonlinear diagnostic
  implementation keeps the broad public signature as a facade, then packs
  method, timestep, stride, Fourier-layout, selected-mode, collision, and
  output-resolution knobs into a private options object before constructing
  state/policy/closure scan components. This keeps runtime diagnostics
  patchable while making the numerical stages testable without a monolithic
  integration body.
- startup/loading/initial-condition helpers:
  ``workflows/runtime/startup.py``
- runtime mode-index, nonlinear step-count, external-source, parallel-scan,
  and nonlinear diagnostics keyword policies:
  ``workflows/runtime/policies.py``
- runtime linear fit/eigenfunction extraction and quasilinear finalization:
  ``workflows/runtime/diagnostics.py``
- finite-value checks plus runtime diagnostic slicing, truncation, striding,
  and concatenation:
  ``workflows/runtime/diagnostic_arrays.py``. These helpers are imported
  directly from this owner rather than through the runtime fit module.
- adaptive chunk execution used by runtime and comparison artifacts:
  ``workflows/runtime/chunks.py``. Progress/ETA formatting is imported from the
  canonical runtime orchestration formatter rather than wrapped in the chunk
  owner.
- runtime result containers and nonlinear result assembly:
  ``workflows/runtime/results.py``
- runtime progress formatting, combined-``ky`` scan batching, serial/worker
  scan orchestration, progress formatting, and nonlinear artifact handoff policy:
  ``workflows/runtime/orchestration.py`` facade plus
  ``workflows/runtime/orchestration_scan.py``,
  ``workflows/runtime/orchestration_progress.py``, and
  ``workflows/runtime/orchestration_artifacts.py``. Scan dependency-bundle
  builders live with the scan owner and read the public ``runtime.py`` facade
  only as a patchable symbol source. Scan routing keeps combined-``ky``
  eligibility, independent-worker task generation, ordered result packing,
  batch initial-state assembly, diagnostic extraction, and fit selection as
  named stages behind one private scan-options contract. Nonlinear artifact handoff keeps restart
  input resolution, append-history loading, checkpoint chunk sizing, and
  diagnostic-history merging as explicit policies in the artifact owner.
- saved runtime-output plotting command routing:
  ``workflows/runtime/commands.py`` plus ``workflows/runtime/orchestration_artifacts.py``.
  The public ``cli.py`` facade still owns executable parser dispatch and
  renderer/runtime-command patch seams, while the command workflow owns
  ``spectraxgk --plot`` argument validation and runtime command dependency
  construction. Command artifact display, executable headers, and nonlinear
  summaries live with the runtime artifact orchestration policy so saved-output
  behavior and restart/checkpoint handoff share one owner.
- runtime restart-state dispatch: the public ``runtime.py`` facade keeps the
  patchable NetCDF/raw state loaders but shares one shape-keyword payload
  between both paths, so restart dimensions cannot drift between loader calls.
- runtime linear time/fit option forwarding: ``run_runtime_linear``,
  ``run_runtime_scan``, and the combined-``ky`` batch wrapper share one private
  option bundle in the public ``runtime.py`` facade, keeping method, timestep,
  sample-stride, fit-window, mode-method, and fit-signal forwarding aligned.
- runtime TOML case dependency defaults:
  ``workflows/cases.py``. The public ``runtime.py`` facade owns the stable
  ``run_linear_case`` and ``run_nonlinear_case`` signatures, while
  ``workflows/cases.py`` owns default case-workflow wiring.
- root benchmark drivers and result pointers:
  ``benchmarks/``. This directory is the canonical lightweight benchmark
  entry point at repository root. It stores drivers, TOML inputs, and
  ``benchmarks/results/manifest.toml`` only; raw solver products stay in
  scratch directories and promoted benchmark results are displayed from
  :doc:`benchmarks`.
- full-GK executable linear runtime workflow:
  ``workflows/linear.py``. Context preparation, time integration, Krylov
  fallback, linear fitting, and quasilinear finalization are separate private
  stages so runtime dispatch can stay compact while preserving the public
  ``spectraxgk.runtime`` facade and monkeypatch seams.
- full-GK executable nonlinear runtime workflow:
  ``workflows/nonlinear.py``
- shared plot style plus runtime-output, benchmark/scan, diagnostic, and
  zonal-response figure families:
  ``artifacts/plot_style.py``, ``artifacts/runtime_plots.py``,
  ``artifacts/benchmark_plots.py``, ``artifacts/diagnostic_plots.py``, and
  ``artifacts/zonal_plots.py``. The public ``artifacts.plotting`` module
  remains a stable import facade for examples and user scripts.
- validation gate dataclasses and JSON-ready gate helpers:
  ``diagnostics/validation_gates.py`` owns metric containers, scalar
  tolerance evaluation, JSON serialization, and report builders. One private
  upper-limit scalar-gate policy is shared for convergence, mismatch, deficit,
  and branch-jump thresholds so tolerance semantics remain auditable.
- autodiff validation helpers:
  ``objectives/autodiff_validation.py`` owns finite-difference Jacobians,
  Gauss-Newton covariance diagnostics, dense operator materialization, and
  isolated eigenbranch sensitivity reports. Eigenbranch AD gates share
  selector/gap classification, complex-observable realification, unsupported-AD
  fallback packing, implicit left/right eigenpair tangent solves, split
  observable chain-rule assembly, and
  finite-difference comparison stages.
- benchmark-harness physics metric extraction and scan/mode orchestration:
  ``validation/benchmarks/harness_metrics.py`` and
  ``validation/benchmarks/harness_scan.py``. Eigenfunction normalization,
  phase alignment, comparison metrics, and reference-bundle IO live in
  ``diagnostics/modes.py``; diagnostic time-series loading, late/leading
  windows, analytic-signal construction, and real-FFT ky-grid inference live in
  ``diagnostics/validation_gates.py``. Zonal-flow residual/GAM metric extraction lives in
  ``diagnostics/zonal_validation.py``. The public
  ``validation/benchmarks/harness.py`` facade keeps existing imports and test
  monkeypatch seams stable. Zonal-response metrics are staged as trace
  coercion, initial-level normalization, tail residual extraction,
  extrema/envelope detection, damping/frequency fits, and final
  ``ZonalFlowResponseMetrics`` packing, keeping Rosenbluth-Hinton/GAM
  conventions explicit.
- zonal-response reference/trace normalization helpers:
  ``diagnostics/zonal_validation.py``
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
  ``objectives/vmec_boozer_line_search.py``. Scalar and aggregate
  VMEC/Boozer finite-difference reports share one settings validator, one
  VMEC-state coefficient context, one perturbation helper, and one three-point
  response/curvature diagnostic so scalar and aggregate sensitivity gates use
  the same finite-value, response-resolution, and curvature policy. Scalar and
  aggregate reports now keep aggregate surface/field-line/``ky`` planning,
  three-point VMEC-state evaluation, weighted-sample metadata, and payload
  assembly in separate helpers so optimizer-facing sensitivity reports remain
  auditable without changing their public JSON schema. Mode-21 VMEC/Boozer
  gradient reports now share context construction, observable-vector assembly,
  implicit sensitivity-gate execution, and common payload metadata, while the
  public report functions remain the physics-facing entry points for linear,
  quasilinear, and reduced nonlinear-window differentiability claims. Scalar and
  aggregate VMEC/Boozer line-search reports share one private curvature-gated
  one-parameter search loop plus focused scalar/aggregate probe builders,
  common payload assembly, and explicit held-out training/probe helpers,
  keeping finite-difference, training, and held-out aggregate gates on the same
  accept/reject policy. Reduced QA low-turbulence comparison artifacts keep
  optimized-design generation, per-design diagnostic payloads, gate booleans,
  comparison metrics, and differentiability-plumbing metadata in separate
  helpers so publication JSON structure and optimization diagnostics can evolve
  independently. Solver-ready
  gradient gates share one normalized heat/particle transport-weight helper
  for eigenmode observables so linear-growth, quasilinear, and particle-flux
  AD checks use the same quadrature and normalization path. Solver-ready
  branch-continuity and linear-RHS gradient reports now stage parameter
  validation, objective construction, eigensystem branch checks, implicit
  AD/FD gate rows, value-evaluator checks, and report packing separately.
  The top-level
  ``spectraxgk.objectives`` API re-exports the portfolio helpers directly from
  these owner modules.
- production nonlinear turbulent-flux optimization guardrails now live in
  ``diagnostics/nonlinear_transport_optimization.py``. The diagnostics owner
  stages optimization-scope normalization, reduced-artifact scope checks,
  replicated long-window transport extraction, matched baseline-to-optimized
  audit gates, safety gates, promotion gates, evidence-gap accounting, and
  summary assembly so release safety and production-claim promotion cannot be
  conflated. Replicate-spread diagnostics now live in
  ``diagnostics/nonlinear_replicates.py`` and stage ensemble row normalization,
  high/low variant selection, state classification, replicate-row packing, and
  summary assembly so seed/timestep spread decisions are testable without
  rerunning nonlinear simulations. Follow-up launch planning is not runtime
  package functionality; it lives in
  ``tools/campaigns/nonlinear_replicate_followup.py``, where report
  normalization, classification-specific cross-run selection, dedupe/limits,
  state-plan packing, and config serialization keep GPU follow-up campaigns
  deterministic and reviewable.
- quasilinear nonlinear-window convergence metadata is split into focused
  config, statistics, CSV/summary IO, promotion-readiness, and ensemble-gate
  modules under ``validation/quasilinear/window_*.py``. The public
  ``validation/quasilinear/window.py`` module remains the stable facade used by
  calibration and tool scripts. The statistics owner stages validated
  late-window selection, finite-sample counts, drift/terminal-window metrics,
  block/bootstrap uncertainty, and gate-report assembly in one file so
  nonlinear transport admission rules remain auditable. The ensemble owner
  stages replicate-row normalization, uncertainty statistics, gate packing,
  artifact grouping, missing-replicate hints, and readiness-manifest packing
  so seed/timestep promotion evidence can be tested without rerunning
  simulations.
- quasilinear model-selection claim boundaries live in
  ``validation/quasilinear/model_selection.py``. The public owner now separates
  candidate-skill gate rows, absolute-flux overclaim guardrails, optional
  optimized-equilibrium audit gates, and final ledger assembly, while
  ``validation/quasilinear/model_selection_inputs.py`` owns artifact loading and
  required-candidate metric normalization.
- nonlinear parallelization policy metadata, local domain prototypes, and
  spectral-core work models/RHS primitives plus device-z shard-map routes:
  ``operators/nonlinear/parallel.py``,
  ``operators/nonlinear/parallel_contracts_domain.py``,
  ``operators/nonlinear/parallel_contracts_spectral.py``,
  ``operators/nonlinear/parallel_contracts_strategy.py``,
  ``operators/nonlinear/domain_decomposition.py``,
  ``operators/nonlinear/spectral_core.py``,
  ``operators/nonlinear/spectral_state.py``,
  ``operators/nonlinear/spectral_layout.py``,
  ``operators/nonlinear/spectral_work_models.py``,
  ``operators/nonlinear/spectral_brackets.py``,
  ``operators/nonlinear/spectral_tolerances.py``,
  ``operators/nonlinear/spectral_identity_reports.py``,
  ``operators/nonlinear/spectral_identity_rhs.py``,
  ``operators/nonlinear/spectral_identity_integrator.py``,
  ``operators/nonlinear/device_z.py``
  The local-domain transport gate stages trace collection, trace-error scoring,
  fail-closed blockers, and report packing in
  ``operators/nonlinear/domain_decomposition.py``. The spectral fixed-window
  and pencil-transport gates use the same structure in
  ``operators/nonlinear/spectral_identity_integrator.py`` so future
  profiler-backed nonlinear decomposition work can change routing without
  weakening the numerical-identity policy. The device-z route keeps sharding
  setup, fail-closed topology checks, shard-map RHS execution, transport-window
  sampling, compute-only final-state identity, and final report packing as
  separate stages inside
  ``operators/nonlinear/device_z.py`` while report-schema construction remains
  in ``operators/nonlinear/device_z_reports.py``.
- velocity-space parallelization is split into decomposition metadata
  (``parallel/velocity_plan.py``), Hermite exchange and velocity-field
  reductions (``parallel/velocity_hermite.py``), streaming/magnetic-drift
  microkernels (``parallel/velocity_streaming.py``), and electrostatic/
  diamagnetic field-drive microkernels (``parallel/velocity_drive.py``). The
  public ``parallel/velocity.py`` module remains the stable facade used by
  tools and performance gates.
- independent-work parallelization is split into numerical-identity reports
  (``parallel/identity.py``), JAX-array batch mapping
  (``parallel/batch.py``), and ordered Python ensemble execution
  (``parallel/independent.py``). The public ``parallel/core.py`` and
  ``parallel/__init__.py`` modules remain stable facades for user imports.
  ``parallel/independent.py`` keeps indexed payload collection, reconstruction
  contracts, identity report construction, exception provenance, ordered
  executor routing, and metadata packing as separate helpers so UQ and
  optimization ensembles stay auditable without changing solver layout.
- nonlinear RHS composition and state-to-diagnostic tuple assembly:
  ``operators/nonlinear/rhs.py`` and
  ``operators/nonlinear/diagnostic_state.py``. The diagnostic-state owner
  separates field defaulting, growth/frequency mode extraction, scalar
  diagnostics, resolved field/transport group evaluation, and resolved
  spectra/channel schema packing while preserving the
  explicit/IMEX scan tuple schema. The old root nonlinear helper shims were
  removed; normal users should use ``spectraxgk.nonlinear`` and developer
  helpers should import from ``spectraxgk.operators.nonlinear``.
- full-GK nonlinear executable orchestration lives in
  ``workflows/nonlinear.py`` behind the public ``spectraxgk.runtime`` facade.
  The owner separates runtime context construction, fixed-mode/source policy,
  diagnostic keyword forwarding, adaptive/fixed diagnostic execution,
  final-state integration, and result assembly so runtime branch tests can
  exercise each policy without duplicating executable wiring.
- fixed-step linear integration keeps public dispatch in
  ``solvers/linear/integrators.py`` while diagnostic sampling lives in
  ``solvers/linear/integrator_diagnostics.py``. The diagnostic owner now keeps
  sampling validation, cache/state setup, damping assembly, explicit/IMEX/RK
  step policy, density/Hermite-Laguerre observables, progress callbacks, and
  every-step versus strided scans in named stages. The facade preserves the
  public ``integrate_linear_diagnostics`` import and test monkeypatch seams.
- velocity-parallel linear RHS routing is split into common eligibility/device
  policy (``solvers/linear/parallel_common.py``), Hermite streaming routes
  (``solvers/linear/parallel_streaming.py``), and electrostatic slice/fused
  shard-map routes (``solvers/linear/parallel_electrostatic.py``). The public
  ``solvers/linear/parallel.py`` module remains the stable dispatcher. The
  electrostatic fused route is staged as route validation, sharding specs,
  closure constants, Hermite exchange, phi solve, streaming/mirror/drift/drive
  term builders, and JIT cache lookup; the serial single-device path has a
  separate helper so identity tests can exercise both routes.
- explicit RK/SSP/K10 one-step policy, cached explicit scan policy, explicit
  diagnostic step and scan-selection policy, explicit diagnostic integration
  orchestration, IMEX diagnostic integration orchestration, cached IMEX scan
  policy, IMEX diagnostic step and scan-execution policy, and IMEX
  fixed-point/GMRES solve policy: ``solvers/nonlinear/explicit.py``,
  ``solvers/nonlinear/diagnostics.py``, ``solvers/nonlinear/imex_diagnostics.py``,
  and ``solvers/nonlinear/imex.py``. Developer helpers should import from
  ``spectraxgk.solvers.nonlinear``.
- linear cache, linked-boundary maps, Hermite-Laguerre moments, parameter
  pytrees, cache-backed RHS assembly, implicit linear GMRES/preconditioner
  policy, fixed-step/diagnostic integration policy, eigenmode policy/operator/
  branch-selection/preconditioner/Krylov algorithms, and velocity-parallel RHS
  dispatch live under ``operators/linear/`` and ``solvers/linear/``. The cache
  path is split into ``cache_model.py`` for the JAX pytree, ``cache_arrays.py``
  for moment/damping/gyroaverage array factories, and ``cache_builder.py`` for
  geometry-dependent construction. The builder itself now has explicit private
  stages for twist-shift policy, perpendicular wavenumber/drift arrays,
  Laguerre gyroaverage construction, and linked-boundary metadata so extension
  work can test one numerical policy at a time; ``cache.py`` remains the stable
  public facade. The public Krylov import path remains ``solvers/linear/krylov.py``;
  that facade now keeps option normalization, user-facing progress messages,
  shift-invert seed selection, shift-selection flags, and fallback policy as
  explicit private stages while delegating compiled kernels to the focused
  owner modules;
  ``solvers/linear/implicit.py`` keeps implicit state normalization, damping/
  drift diagonal assembly, linked Hermite-line solves, coarse kx projection,
  preconditioner selection, and matrix-free matvec construction as separate
  private stages;
  focused developer helpers live in ``eigen_policy.py``, ``eigen_operator.py``,
  ``eigen_selection.py``, ``eigen_preconditioners.py``, and
  ``krylov_algorithms.py``. The old root ``linear_*`` helper shims were
  removed; normal users should use ``spectraxgk.linear`` for the public linear
  API or import focused developer helpers from the domain packages.
- nonlinear turbulence-gradient follow-up shared configs, JSON parsing,
  candidate design, composite-control, matched-replicate, QL-seed,
  state-runbook, and variance-reduction/control-variate report helpers now live
  in ``tools/campaigns/nonlinear_gradient_followup.py``. They are campaign
  planning tools, not runtime package functionality. Variance-reduction,
  control-mean campaign, and control-mean gate reports share one
  control-variate candidate parsing/ranking policy and stage paired-label
  extraction, control-candidate construction, uncertainty propagation,
  campaign sizing, independent-control pairing, and report packing so noisy
  follow-up campaign decisions stay deterministic. QL-seed screening,
  state-control runbooks, matched-replicate follow-up planning, and candidate
  campaign design are all staged inside that single tools owner so they do not
  re-enter ``src/spectraxgk/validation``.
- nonlinear turbulence-gradient evidence scope markers, acceptance config
  dataclasses, JSON-safe parsing, finite-difference conditioning gates,
  artifact classification, replicated window summaries, central
  finite-difference report assembly, candidate/bracket screening reports, and
  production evidence-gap report orchestration now live in
  ``diagnostics/nonlinear_gradient_evidence.py``. The central finite-difference
  gate is staged as matched-window normalization, transport-response
  extraction, uncertainty propagation, source/window quality gates,
  gradient-resolution gates, and JSON-ready report packing. Candidate ranking
  and evidence-gap reporting remain fail-closed so production
  nonlinear-gradient promotion cannot be inferred from startup, pilot, reduced,
  or single-window artifacts.
- runtime artifact read/write, generic I/O helpers, linear/quasilinear
  artifact writers, generic nonlinear table writers, dealiased-axis
  layout, NetCDF output-bundle orchestration, NetCDF diagnostic-history schema
  writing with staged Phi2, base species-history, split electromagnetic,
  zonal-field, resolved species-spectra, and turbulent-heating helpers, NetCDF
  output geometry,
  restart-file writing, final-field big-file writing, nonlinear diagnostic
  reload helpers, and restart-append schema coverage:
  ``workflows/runtime/artifacts.py``, ``artifacts/io.py``,
  ``artifacts/linear.py``,
  ``artifacts/nonlinear.py``,
  ``artifacts/spectral_layout.py``,
  ``artifacts/nonlinear_netcdf.py``,
  ``artifacts/nonlinear_netcdf_diagnostics.py``,
  ``artifacts/nonlinear_netcdf_geometry.py``,
  ``artifacts/nonlinear_netcdf_fields.py``,
  ``artifacts/nonlinear_diagnostics.py``. The old root
  ``runtime_artifact_*`` helper modules were removed; import implementation
  helpers from ``spectraxgk.artifacts`` instead.

Next planned extractions:

- benchmark-family runners and fit-policy helpers
- remaining linear result-assembly helpers
- runtime output/artifact handoff helpers

Traceability For Refactors
--------------------------

Refactor work is tracked in ``tools/validation_coverage_manifest.toml``. The
manifest is checked by ``tools/release/check_validation_coverage_manifest.py`` and
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
``tools/release/check_differentiable_refactor_manifest.py``. See
:doc:`differentiable_refactor_plan` for differentiability contracts, extension
points, active manifest rows, and physics/autodiff/parity gates. If this page or
the manifest conflicts with :doc:`architecture_refactor_plan`, update the
current-tree documentation or manifest rather than adding another root-level
prefix module.

The first behavior-preserving contract modules are ``spectraxgk.core.contracts``
and ``spectraxgk.core.extension_points``. They introduce typed refactor,
validation-gate, differentiability, and extension-point protocols without
moving solver kernels or changing public numerical behavior.

Runtime command dispatch now keeps parser construction in ``spectraxgk.cli``,
with parser registration split by command family, and moves runtime linear,
runtime scan, and runtime nonlinear command execution into
``spectraxgk.workflows.runtime.commands``. The generic ``run`` executable path
attaches the already-loaded runtime config/data to the parser namespace before
dispatch, so command execution does not parse the same TOML twice.
Linear, scan, and nonlinear command flags are resolved once into typed
command-option records before solver calls are made, which keeps precedence
between CLI flags, TOML sections, and runtime defaults inspectable without
spreading executable policy across command bodies. Saved-artifact display order
and optional artifact writing for linear, scan, quasilinear, and nonlinear
commands also live in focused command-output helpers so user-facing executable
messages can be tested without launching solver runs.
``spectraxgk.workflows.cases`` remains the TOML case-workflow owner and
re-exports the command helpers only for public executable dispatch; path override, progress,
quasilinear override, and preload-reuse policies have one canonical workflow
owner.
Runtime initial-condition construction is staged inside
``spectraxgk.workflows.runtime.initial_conditions``. Validation of initializer
options, restart-state scaling, kinetic-species targeting, single-mode profile
assembly, random/Gaussian multimode seeding, electrostatic-potential seeding,
Hermitian full-``ky`` completion, and restart merge policy now have named helper
stages. The startup facade still exposes the same ``_build_initial_condition``
entry point for runtime workflows and tests, but the implementation no longer
mixes restart I/O, moment normalization policy, and phi-inversion cache setup in
one long branch.

The benchmark helper split now uses focused domain modules directly.
Benchmark case presets live directly in ``spectraxgk.config`` so user-facing
configuration objects do not depend on the temporary validation package.
Benchmark initial conditions and reference data live in
``spectraxgk.validation.benchmarks.initialization`` and
``spectraxgk.validation.benchmarks.reference``. Benchmark species-to-``LinearParams``
construction and reference hypercollision/end-damping policy live in
``spectraxgk.validation.benchmarks.species``. Fit-signal selection, scan batching, and
solver-selection policies live in ``spectraxgk.validation.benchmarks.fit_signals``,
``spectraxgk.validation.benchmarks.batching``, and
``spectraxgk.validation.benchmarks.solver_policy``. Import-identity tests pin the old
helper symbols to the new modules before larger benchmark-family runners are
moved. KBM beta-scan, single-point, and ky-scan implementations live in
``spectraxgk.validation.benchmarks.kbm_beta``,
``spectraxgk.validation.benchmarks.kbm_linear``, and
``spectraxgk.validation.benchmarks.kbm_scan`` and are re-exported through
``spectraxgk.benchmarks``, while
``spectraxgk.validation.benchmarks.kbm_beta_solver_paths`` owns the explicit-time
diagnostic fallback ladder and multi-target Krylov policy used by the beta-scan
runner. Its beta-scan Krylov path shares one forwarded-key policy for
multi-target branch selection and continuation/shifted solves, so target and
shift variants cannot silently diverge. Multi-target transition-threshold and
fastest-growth fallback candidate selection lives in a focused local helper so
the solver path keeps branch-choice policy separate from solve orchestration.
The explicit-time fallback ladder, saved-time auto-fit branch, and
Diffrax-streaming window resolution share the scan fit-window policy, so
beta-scan fit knobs are not duplicated across time-integration paths.
Saved-time KBM beta samples also use one dispatcher for
non-Diffrax time-config and no-config integration, with stride resolution kept
explicit before fitting; Diffrax-streaming samples read the same
``ScanFitWindowPolicy`` for their resolved fit window. The single-point runner delegates
explicit-time diagnostics and single/multi-target Krylov branch selection to
``spectraxgk.validation.benchmarks.kbm_linear_paths`` while retaining geometry
setup, state/cache construction, saved/configured trajectory integration,
saved-signal fitting, and result packaging through focused helper seams in the
public owner. The public beta runner still owns per-beta setup and time/diffrax
fallback.
Single-point KBM dispatch now passes one explicit run-options object through
the explicit-time, Krylov, and saved-time helper paths, keeping solver selection
separate from the numerical fitting and trajectory code that remains local to
``kbm_linear``.
``spectraxgk.benchmarks`` remains the public facade for
``run_kbm_linear``, ``run_kbm_scan``, and ``run_kbm_beta_scan``. The TEM benchmark family follows the same pattern in
``spectraxgk.validation.benchmarks.tem`` for ``run_tem_linear`` and ``run_tem_scan``.
The KBM beta-scan owner keeps patchable numerical hooks local while staging
shared setup, fit-window policy construction, kinetic-species index validation,
per-beta state/cache construction, solver-path dispatch, Krylov continuation
updates, and result packing through focused private helpers. The existing
``kbm_beta_solver_paths`` module remains the owner of explicit-time, Krylov,
streaming, and saved-time fitting details.
The KBM single-point saved-time direct-fit path shares one automatic-fit keyword
policy between primary auto-window fitting and invalid-window fallback fitting,
and configured-time versus fixed-time integration is split before signal
selection so stride and density-output policy cannot drift.
The TEM public owner now keeps setup, parameter construction, and species
validation local while ``spectraxgk.validation.benchmarks.tem_paths`` owns the
single-ky Krylov path, saved-time fit path, streaming scan branch, and scan
batch loop through an explicit hook bundle. The TEM Krylov path shares one
forwarded-key policy for dominant-eigenpair configuration, matching the KBM
benchmark-path guard against target/shift policy drift. The TEM single-ky
saved-time path shares one automatic-fit keyword policy between primary
auto-window fitting and invalid-window fallback fitting. The same single-ky
time path resolves time-config, ``dt``/``steps``, and stride once before
dispatching to density diagnostics, configured ``phi``, or explicit ``phi``
integration, so those saved-time branches cannot drift. TEM scan streaming
also resolves its fit window through the same ``ScanFitWindowPolicy`` used by
saved-time scan fitting.
Kinetic-electron ITG/TEM runners now live directly in ``spectraxgk.validation.benchmarks.kinetic_linear`` and ``spectraxgk.validation.benchmarks.kinetic_scan``; the supported public import remains ``spectraxgk.benchmarks``. The kinetic single-ky owner keeps the public API and result schema local while staging setup normalization, species-index validation, selected-state construction, Krylov solving, configured and unconfigured time-history integration, sampled-signal fitting, and result packing in focused private helpers in the same module. The kinetic scan owner keeps public setup and result packaging local while delegating setup normalization, batch-state construction, Krylov fitting, Diffrax streaming fitting, saved/configured trajectory integration, and sampled-signal fitting to focused private helpers in the same module.
The kinetic scan path carries separate run-options, fit-options, and output
containers through a single batch router, keeping Krylov, Diffrax streaming, and
sampled-history branches testable without changing the public scan signature.
ETG single-point and scan implementations live in
``spectraxgk.validation.benchmarks.etg_linear`` and
``spectraxgk.validation.benchmarks.etg_scan`` and are re-exported through
``spectraxgk.benchmarks``. The scan runner keeps geometry/species setup,
electrostatic term defaults, fit-window policy construction, ky-batch state
construction, and per-batch result packaging in focused local helpers while delegating
Krylov continuation, streaming fit, saved-signal integration, and fallback
fit/appending policy to ``spectraxgk.validation.benchmarks.etg_scan_paths``
for solver-path details.
ETG single-point and scan Krylov paths share one forwarded-key policy, with
scan continuation overrides applied explicitly for carried shifts. The ETG
single-point saved-time direct-fit path also shares one automatic-fit keyword
policy between primary auto-window fitting and invalid-window fallback fitting.
The single-point runner keeps patchable solver hooks in the public ETG module
but now separates setup, Krylov result packing, time-configuration resolution,
streaming-density fitting, configured/unconfigured saved-history integration,
and saved-trace fitting into focused helpers. The saved-time path carries one
private fit-policy object through those stages so ``phi``, density, automatic,
and reference-window fits cannot silently diverge in their normalization or
window-selection rules. ETG scan path internals carry separate batch and fit
context objects through staged streaming, configured-history,
unconfigured-history, direct-fit, auto-fit, and Krylov-fallback helpers, keeping
the scan owner patchable while making each numerical branch locally auditable.
Cyclone single-mode and scan implementations now live in
``spectraxgk.validation.benchmarks.cyclone_linear`` and
``spectraxgk.validation.benchmarks.cyclone_scan`` and are re-exported through
``spectraxgk.benchmarks``. The single-mode runner
keeps public setup and solver fallback orchestration local while staging
default parameter/term construction, reference-aligned geometry policy,
fit-signal validation, resolved run setup, time/Krylov dispatch, and
``CycloneRunResult`` packing into private helpers. It delegates Krylov
seeding/branch selection and time-integration fit policy to
``spectraxgk.validation.benchmarks.cyclone_linear_paths``. The Krylov path now
separates explicit frequency-seed fitting, primary/reduced seed fallback,
shift-target construction, dominant-eigenpair option forwarding, branch-guard
selection, field packing, and normalization into named stages. The time path now
separates runtime-config resolution, reference-aligned explicit integration,
configured/unconfigured fixed-step integration, shared automatic-window
keyword packing, and saved-trace fitting into named stages. The scan runner delegates
Krylov branch-following, reference-aligned explicit-time reselection, and
standard saved-time/streaming scan execution to
``spectraxgk.validation.benchmarks.cyclone_scan_branches`` through an explicit
hook bundle while keeping scan setup, default species/term policy,
reference-aligned normalization, fit-window policy packing, ky-batch selection,
and branch dispatch in focused local helpers. Trace-seed branch initialization lives in
``spectraxgk.validation.benchmarks.cyclone_scan_seed`` and reference-aligned
explicit-time reselection lives in
``spectraxgk.validation.benchmarks.cyclone_scan_explicit``. The branch module
owns the patchable hook bundle used by Cyclone scan tests while the solver
policies stay isolated for review. The Cyclone single-mode time path shares
one local automatic-fit keyword policy between automatic signal selection and
direct signal fitting, avoiding drift in late-window fit semantics. The Cyclone
scan time path now keeps batch construction, per-batch time-configuration
resolution, Diffrax streaming fits, saved/configured trajectory integration,
and per-ky fit/appending policy as explicit helper seams inside the same branch
owner. The saved-time scan path carries separate run-options, fit-options, and
output containers through a single batch router, so future fit-policy changes
can be tested without modifying scan setup or result packing.
Family-specific branch tests now patch the
family owner modules directly, and examples/downstream scripts keep importing
through ``spectraxgk.benchmarks``.

Quasilinear calibration now lives in
``spectraxgk.diagnostics.quasilinear_calibration``. It owns calibration-point
schemas, spectrum integration, train/holdout scale fitting, nonlinear-window
CSV/NetCDF ingestion, and report writing behind one diagnostics owner.
Late-window transport gates live in ``spectraxgk.diagnostics.transport_windows``.
The public validation API re-exports user-facing helpers while
campaign launch and artifact-building policy stays in ``tools``. Model-selection status construction keeps scoped candidate-skill gates,
input normalization, optimized-equilibrium audit summaries, and absolute-flux
claim guardrails in ``diagnostics/quasilinear_model_selection.py``.

VMEC-JAX candidate and transport admission gates now have explicit owners.
``spectraxgk.objectives.vmec_candidate_admission`` owns solved-equilibrium,
authoritative-WOUT, and WOUT-reproducibility candidate gates. It keeps aspect,
iota, iota-profile, quasisymmetry, and pass/fail helpers together so optimizer
state and WOUT gates share one JSON schema and threshold semantics.
``spectraxgk.objectives.vmec_transport_admission`` owns transport-admission
policy dataclasses, reduced transport metric selection, multi-surface/
field-line/``k_y`` sample coverage, and promoted transport-candidate selection.
``spectraxgk.diagnostics.stellarator_transport_reports`` owns report-style
nonlinear transport diagnostics: landscape admission, reduced prelaunch gates,
next-campaign admission, and matched nonlinear audit redesign. The public
``spectraxgk.validation`` API re-exports user-facing admission helpers directly
from these owners, while the old installable stellarator-validation subpackage has
been removed.

The first differentiable-geometry split keeps
``spectraxgk.geometry.differentiable`` as the public facade while
moving optional backend lookup and strict AD/finite-difference gate utilities
into ``spectraxgk.geometry.backend_discovery`` and
``spectraxgk.geometry.autodiff_checks``. The solver-ready in-memory flux-tube
mapping and geometry-observable contract lives in
``spectraxgk.geometry.flux_tube_contract``. Pure parity metrics,
interpolation, radial derivative, Boozer half-mesh, Fourier field-line, and
periodic sampling helpers live in ``spectraxgk.geometry.numerics``. This
separates import-side effects, validation-report plumbing, public contract
validation, and small numerical kernels from the VMEC/Boozer field-line bridge.
The AD/FD validation owner stages parameter validation, observable flattening,
Jacobian construction, tangent checks, conditioning gates, failure reasons, and
strict JSON report assembly so differentiability tests can target each
research-grade gate directly.
Imported VMEC/Boozer radial spline construction lives in
``spectraxgk.geometry_backends.vmec_splines`` and is re-exported through
``spectraxgk.geometry_backends.vmec_fieldlines`` for the existing VMEC backend
facade. The VMEC field-line backend keeps Boozer-object fallback,
Boozer-mode table sampling, alpha-line coordinate construction, axisymmetric
flip detection, angle construction, resonant-denominator guarding,
field-line tensor algebra, alpha/coordinate-gradient construction, local
shear, metric/drift coefficient assembly, flux-surface averaging, and centered
field-line integral policies as focused helpers inside
``spectraxgk.geometry_backends.vmec_fieldline_numerics`` so the
imported-geometry equations remain in one owner while the numerical kernels are
unit-testable. The field-line metric/drift path is staged as curvature
components, normalized metric profiles, magnetic-drift profiles,
gradient-vector packing, and final coefficient assembly. The flux-surface
Hegna-Nakajima average path is staged as grid construction, Boozer-geometry
sampling, ``|grad psi|`` normalization, and Jacobian-weighted averages, which
keeps each physics convention independently testable without splitting the
VMEC-specific formulas across unrelated packages.
``spectraxgk.geometry_backends.vmec_fieldlines`` now keeps only
the imported-geometry orchestration stages in that file: backend fallback,
scalar VMEC profile sampling, Boozer field-line state assembly,
Hegna-Nakajima mode corrections, metric/drift coefficient assembly, and
normalized flux-tube packaging.
Zero-shear boundary policy lives in ``spectraxgk.geometry.boundaries``.
Analytic s-alpha and slab geometry models live in
``spectraxgk.geometry.analytic``. Sampled solver-ready geometry data, analytic
sampling, imported-NetCDF loading, and periodic mirror-term reconstruction live
in ``spectraxgk.geometry.flux_tube``. Twist-shift and grid-default geometry
policy live in ``spectraxgk.geometry.core``. The
``spectraxgk.geometry`` package remains a thin public facade that
re-exports the same classes and functions for existing user code.
Geometry sensitivity, inverse-design, and local UQ reports live in
``spectraxgk.geometry.sensitivity`` so backend bridge modules can depend on
the report contract without importing the public facade. Bounded VMEC
boundary and Boozer-spectrum bridge checks, Boozer ``|B|`` field-line
evaluation, and Boozer-to-flux-tube sensitivity diagnostics live in
``spectraxgk.geometry.booz_xform_bridge``. Pure helper imports retain object
identity; backend-discovery-dependent bridge functions use thin facade wrappers
so existing monkeypatch-based optional-backend tests still target
``spectraxgk.geometry.differentiable``. VMEC-state-to-Boozer, VMEC metric
tensor, and VMEC field-line tensor AD/FD sensitivity reports live in
``spectraxgk.geometry.vmec_state_sensitivity``; the public facade uses the
same hook-preserving wrapper pattern, while the implementation owns shared
VMEC example loading, coefficient-index validation, perturbation policy, and
the common tensor-observable AD/finite-difference payload builder used by
metric and field-line gates. Metric and field-line tensor reports also share
one VMEC geometry context/index helper before diverging into their
observable-specific sampling paths.
Boozer constant preparation and equal-arc cache prewarm helpers live in
``spectraxgk.geometry.vmec_boozer_constants``. Core Boozer equal-arc profile
construction, radial Boozer-profile interpolation, and equal-arc remapping
live in ``spectraxgk.geometry.vmec_boozer_core``. That owner now stages the
bridge as radial Boozer-profile interpolation, equal-arc field-line
construction, zero-beta metric/drift profile assembly, and final solver-ready
mapping assembly; optional backend execution and Boozer radial-grid validation
remain private helper seams so the public state-to-profile bridge stays
focused on physics profile assembly. Metric/drift profile assembly is further
split into differential-geometry evaluation, raw metric coefficients, raw
curvature-drift coefficients, open equal-arc remapping, and final
``_MetricDriftProfiles`` packing so the tensor algebra and solver-normalized
coefficients can be reviewed independently. The core bridge delegates Boozer
field-line spectral sums, cylindrical derivatives, and coordinate-gradient
algebra to
``spectraxgk.geometry.vmec_boozer_derivatives`` so the differentiable geometry
path has a small, unit-testable tensor-algebra owner rather than one long
VMEC/Boozer orchestration function. The core
profile assembly shares one dtype-aware numerical floor across ``|B|``,
``gradpar``, Jacobian, metric, curvature, and ``q`` denominators so
regularization policy is visible and consistent. Boozer metric-gradient terms
use a separate float32-safe toroidal-flux denominator floor before
``grad(theta)``, ``grad(phi)``, and ``grad(alpha)`` divisions. Direct
``vmec_jax`` tensor sampling and conversion into the solver-ready flux-tube
mapping contract lives in ``spectraxgk.geometry.vmec_tensor_mapping``. That
bridge is staged as surface/reference-scale validation, shared VMEC field-line
coordinate construction, raw tensor loading, periodic line sampling,
perpendicular metric assembly, local grad-``B`` drift closure, and final mapping
packaging. VMEC flux-tube sensitivity and array-parity report orchestration
lives in
``spectraxgk.geometry.vmec_flux_tube_reports``; it reuses the shared
VMEC-state example loading, coefficient-index validation, and perturbation
policy from ``spectraxgk.geometry.vmec_state_sensitivity`` so the flux-tube,
Boozer, metric-tensor, and field-line AD/FD gates stay on one setup contract.
Direct-array parity, imported-EIK loading, optional Boozer equal-arc parity,
production parity metrics, and final JSON packing now have separate private
stages in that owner. The public report functions remain schema-preserving
facades over those stages, which keeps the VMEC/Boozer differentiability and
parity gates easier to review without adding another module.
VMEC boundary-gradient
probe classification, collection row assembly, and projected-transport
line-search admission summaries live in
``spectraxgk.geometry.vmec_boundary_chain``. Boundary-chain scalar error
construction and pass/fail policy helpers are kept explicit in that owner so
VMEC/Boozer finite-difference, JVP, and VJP convention gates can be tested
without launching expensive VMEC solves.

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

   python tools/release/audit_repository_size.py --top 30
   python tools/release/check_repository_size_manifest.py
   python tools/release/check_release_artifact_manifest.py

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

   python tools/artifacts/compress_previews.py --mode release --max-width 2200 --colors 192
   python tools/artifacts/compress_previews.py --mode docs --min-bytes 300000 --max-width 1800 --colors 192

The first command only touches release-manifest previews, so update
``tools/release_artifact_manifest.toml`` with the new sizes and checksums after
running it. The second command skips release-manifest paths by default and is
intended for ordinary checked-in documentation previews. Rerun both manifest
checkers after either cleanup.

History rewrites are not part of routine development; they require a coordinated
maintenance window because every collaborator must reclone or reset local
branches after a force push.
