Release Scope and Claim Boundaries
==================================

This page is the canonical claim-scope checklist for the current development
state. It keeps README, documentation, release notes, and manuscript drafts
aligned with the tracked artifacts in ``docs/_static``. If a claim is not
listed here or in the referenced gate JSON, treat it as unpromoted.
Here, "current" means artifact-backed and release/manuscript scoped. "Deferred"
means visible as an audit or planning lane but unavailable for release notes,
README highlights, abstracts, or paper conclusions until a later gate promotes
it.

Current scoped claims
---------------------

Claim scope for this release is intentionally artifact-limited: each
release-ready claim below must be backed by the cited tracked figure, JSON
report, test, or workflow gate. Open manuscript physics lanes stay visible in
the guardrail artifacts, but they are not promoted by the release-readiness
score.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Lane
     - Status
     - Supported claim
   * - Linear and nonlinear benchmark atlas
     - release-ready for named cases
     - Linear growth/frequency/eigenfunction and nonlinear window statistics
       are validated for the tracked release cases. The nonlinear window
       statistics gate includes only Cyclone, Cyclone Miller, KBM, W7-X, and
       HSX. ETG nonlinear pilots and KAW/TEM stress lanes are not part of the
       release nonlinear parity claim unless a later gate index admits them.
   * - Runtime/refactor artifact contract
     - release-ready as infrastructure
     - The large runtime and diagnostics refactor is covered as a behavior
       preservation claim: extracted startup, chunk, result, validation-gate,
       and artifact helpers keep the public runtime and NetCDF restart/append
       contracts stable. This does not promote new physics validation,
       nonlinear optimization, or performance claims.
   * - Quasilinear diagnostics
     - release-ready as diagnostics
     - Electrostatic linear heat/particle weights, spectra, and model-selection
       artifacts are reproducible. Simple one-scalar saturation rules are
       rejected on the seven-case train/holdout portfolio. The
       ``spectral_envelope_ridge`` candidate is accepted only as a scoped
       manuscript model-selection result, not as a runtime/TOML absolute-flux
       predictor. Any future absolute-flux promotion additionally requires
       finite passed nonlinear late-window convergence metadata for every
       holdout: transient cutoff, running-mean drift, block/bootstrap SEM,
       finite sample count, and source provenance. Electromagnetic
       quasilinear field-channel normalization and KBM calibration remain
       future gates.
   * - Differentiable geometry
     - release-ready for equal-arc parity and reduced QH/Li383 gates
     - The ``vmec_jax -> booz_xform_jax -> SPECTRAX-GK`` bridge is validated
       for zero-beta equal-arc field-line parity where the current
       ``mboz=nboz=21`` parity artifact passes. The fixed-resolution QI row
       uses the selected ``ntheta=16`` field-line floor, with drift mismatch
       about ``7.02e-2`` against the unchanged ``8e-2`` tolerance. A later
       ``ntheta=8`` rerun reached ``8.1879e-2`` and is tracked as fragile,
       not as the floor. This is still not a broad
       QI transport or optimization claim. Reduced frequency, quasilinear, and
       nonlinear-window-estimator gradients pass AD/finite-difference gates on
       QH and Li383. The actual nonlinear
       finite-difference audits are startup plumbing checks with false
       transport-average gates; they do not validate production turbulence
       gradients.
   * - Stellarator optimization examples
     - release-ready as reduced examples
     - The examples demonstrate differentiable reduced ITG objectives, UQ, and
       AD/finite-difference checks. The nonlinear objective is a reduced
       window-estimator path, not a converged post-transient nonlinear
       heat-flux optimization claim.
   * - Parallelization
     - production-ready for independent work
     - Independent ``k_y`` scans, quasilinear spectra, sensitivity batches, and
       UQ ensembles preserve serial ordering and have solver-backed scaling
       artifacts. Runtime scan TOMLs may use ``[parallel] strategy = "batch"``
       with ``axis = "ky"`` for this independent scan path. Whole-state
       nonlinear sharding is a correctness/profiler gate only.
   * - Performance
     - release-ready for scoped profiler evidence
     - Runtime/memory panels, RHS profiler artifacts, and state-sharding
       identity checks are tracked. No broad nonlinear multi-GPU speedup or
       production domain-decomposition claim is made.

Explicitly unpromoted claims
----------------------------

Do not make these claims from the current artifacts:

- universal or user-facing absolute quasilinear flux prediction;
- treating refactor/test coverage as new physics validation or as a nonlinear
  performance claim;
- using ``spectral_envelope_ridge`` as a shipped runtime or TOML saturation
  option;
- electromagnetic quasilinear transport calibration for KBM;
- production nonlinear heat-flux stellarator optimization;
- converged nonlinear transport gradients through ``vmec_jax`` and
  ``booz_xform_jax``;
- treating compact nonlinear finite-difference startup audits as saturated
  transport averages;
- broad W7-X validation beyond the tracked single-flux-tube ITG windows;
- broad QI validation beyond the fixed-resolution mode-21 equal-arc parity row;
- citing even the fixed-resolution QI mode-21 row when the latest regenerated
  parity artifact fails, errors, or is missing;
- W7-X TEM / kinetic-electron validation;
- W7-X long-window zonal recurrence/damping closure;
- nonlinear multi-GPU speedup from whole-state sharding;
- FFT-axis nonlinear domain decomposition.

Release figure and artifact inventory
-------------------------------------

Use this inventory when deciding which figures can support release notes,
README claims, or manuscript claims.

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Claim family
     - Current release/manuscript artifacts
     - Boundary
   * - Benchmark validation
     - ``benchmark_core_linear_atlas.png``,
       ``benchmark_core_nonlinear_atlas.png``,
       ``nonlinear_window_statistics.{png,json}``, and
       ``validation_gate_index.{png,json}``
     - Nonlinear release parity is the five-case window-statistics set only.
       Stress, pilot, and non-indexed example figures are not promoted.
   * - Quasilinear diagnostics and model selection
     - ``quasilinear_*_spectrum.*``,
       ``quasilinear_validated_calibration_inputs.*``,
       ``quasilinear_stellarator_train_holdout.*``,
       ``quasilinear_saturation_rule_sweep.*``,
       ``quasilinear_candidate_uncertainty.*``, and
       ``quasilinear_dataset_sufficiency.*``
     - Electrostatic diagnostics and manuscript model selection are in scope.
       Runtime absolute-flux prediction and electromagnetic calibration are not.
   * - Autodiff and differentiable geometry
     - ``autodiff_inverse_growth.*``, ``autodiff_inverse_twomode.*``,
       ``differentiable_geometry_bridge.*``, ``vmec_boozer_parity_matrix.*``,
       ``vmec_boozer_qi_robustness.json``,
       ``vmec_boozer_gradient_holdout_matrix.*``,
       ``nonlinear_window_fd_audit.*``, and
       ``vmec_boozer_nonlinear_window_fd_audit.*``
     - Reduced AD/finite-difference gates are in scope. Production nonlinear
       turbulence-gradient and optimized-equilibrium heat-flux claims are not.
   * - Scope guardrails
     - ``technical_release_status.json``,
       ``release_readiness.json``, ``manuscript_readiness_status.*``,
       ``open_research_lane_status.*``, and
       ``w7x_tem_extension_status.*``
     - These panels record what is closed, deferred, partial, or open; they do
       not promote the underlying deferred physics lanes by themselves.
   * - Performance and parallelization
     - ``runtime_memory_benchmark.*``,
       ``independent_ky_scan_scaling_large.*``,
       ``quasilinear_uq_ensemble_scaling_large.*``, and
       ``nonlinear_sharding_*``
     - Independent-work parallelization and profiler localization are in scope.
       Whole-state nonlinear sharding is not a production speedup claim.

Artifact-backed details
-----------------------

Runtime/refactor state:

- The current large refactor has extracted runtime startup, diagnostics,
  adaptive chunks, result assembly, validation-gate helpers, zonal-validation
  helpers, parallelization policy helpers, and runtime artifact boundaries into
  smaller tested modules. This is a maintainability and public-behavior
  preservation lane.
- Restartable nonlinear NetCDF append now normalizes loaded diagnostics to the
  persisted schema before concatenation. Monitored complex mode traces that are
  transient in memory and not written to ``*.out.nc`` remain absent on reload,
  so continuation artifacts do not mix persisted and non-persisted diagnostic
  fields.
- These refactor checks support release engineering only. They do not change
  the benchmark, quasilinear, QI, nonlinear optimization, or performance claim
  surface without the artifact gates listed below.

Quasilinear model-selection state:

- ``docs/_static/quasilinear_stellarator_train_holdout_report.json``:
  nonlinear inputs are valid, but the one-constant absolute-flux model remains
  ``passed = false`` with held-out mean relative error about ``2.57``.
- ``tools/check_nonlinear_window_convergence.py`` and
  ``spectraxgk.quasilinear_window`` provide the reusable late-window
  convergence metadata required before any future holdout report can be
  promoted to ``calibrated_absolute_flux``. This is a metadata/finite-window
  guardrail over existing traces, not a substitute for new long nonlinear
  simulations.
- ``docs/_static/quasilinear_saturation_rule_sweep.json``:
  no simple saturation rule is accepted. Positive-growth mixing length is the
  least-bad simple rule with mean held-out relative error about ``2.51``;
  the training-mean null is about ``1.39``.
- ``docs/_static/quasilinear_candidate_uncertainty.json``:
  ``spectral_envelope_ridge`` is the accepted scoped candidate with mean
  relative error about ``0.244`` and interval coverage about ``0.857`` on the
  current seven-case electrostatic-compatible portfolio. Its claim level is
  ``candidate_model_development_not_runtime_option``.

Nonlinear benchmark state:

- ``docs/_static/nonlinear_window_statistics.json`` records five passed
  release-window cases. KBM and HSX use tightened gates, Cyclone Miller is
  tighter than the broad release envelope, while Cyclone and W7-X remain at the
  ``0.10`` release envelope pending paper-level retuning.
- ``docs/_static/validation_gate_index.json`` currently records ``10`` passed
  gate-indexed reports and ``0`` open reports. It is a gate index, not a
  blanket promotion of every figure under ``docs/_static``.
- ``docs/_static/nonlinear_transport_time_horizon_audit.json`` separates
  long post-transient transport windows from startup finite-difference and
  reduced-envelope checks. Startup windows must never be described as saturated
  heat-flux averages.

Differentiable-geometry state:

- ``docs/_static/vmec_boozer_parity_matrix.json`` is the source of truth for
  the multi-equilibrium zero-beta equal-arc field-line convention gate at
  ``mboz=nboz=21``. The current regenerated artifact passes QH, fixed-
  resolution QI, and shaped-tokamak rows when QI uses the selected
  ``ntheta=16`` floor.
- ``docs/_static/vmec_boozer_qi_robustness.json`` records the QI robustness
  selection. The older ``ntheta=8`` point has a later drift-only rerun at
  ``8.1879e-2`` against the ``8e-2`` release tolerance, so that exact
  configuration is blocked from floor selection. The selected complete QI row
  has drift mismatch about ``7.02e-2`` at ``ntheta=16`` and
  ``mboz=nboz=21``. The full declared QI seed campaign is still
  artifact-limited because three QI input variants have no bundled ``wout``
  reference. The builders reject ``mboz,nboz < 21`` so QI is not silently
  evaluated on the under-resolved low-mode setting.
- ``docs/_static/vmec_boozer_gradient_holdout_matrix.json`` passes reduced
  linear, quasilinear, and nonlinear-window-estimator gradient gates on QH and
  Li383 with maximum relative mismatch about ``2.7e-2``.
- ``docs/_static/nonlinear_window_fd_audit.json`` and
  ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.json`` pass only startup
  finite-difference plumbing checks. Both record ``transport_average_gate =
  false``.
- Finite-beta drift reconstruction, converged nonlinear turbulence gradients,
  multi-surface/multi-alpha optimization, and optimized-equilibrium nonlinear
  audits remain future promotion gates.

Parallelization and performance state:

- ``docs/_static/independent_ky_scan_scaling_large.json`` and
  ``docs/_static/quasilinear_uq_ensemble_scaling_large.json`` support
  production independent-work parallelization for scans and ensembles.
- ``docs/_static/nonlinear_sharding_strong_scaling_large.json`` is an identity
  and profiler-direction artifact. It shows whole-state nonlinear sharding is
  identity-correct but not a production speedup path for the current
  decomposition.
- ``docs/_static/nonlinear_domain_parallel_identity_gate.json`` and
  ``docs/_static/nonlinear_spectral_communication_identity_gate.json`` are
  diagnostic identity gates for local halo chunks and spectral
  split/reassemble communication layout, respectively. They are correctness
  prerequisites for future nonlinear domain decomposition, not runtime
  distributed-FFT or nonlinear speedup claims.
- ``docs/_static/nonlinear_sharding_profile_office_gpu.json`` and related RHS
  profiler artifacts support scoped hot-path localization only.

Deferred manuscript lanes
-------------------------

The current manuscript/readme scope intentionally defers:

- W7-X zonal long-window recurrence/damping closure under the paper-facing
  initializer and observable;
- W7-X multi-flux-tube, multi-surface, and TEM / kinetic-electron validation;
- experimental W7-X fluctuation-spectrum claims through diagnostic transfer
  functions.

These are tracked in ``docs/_static/manuscript_readiness_status.json``,
``docs/_static/open_research_lane_status.json``, and
``docs/_static/w7x_tem_extension_status.json``. In the narrower manuscript
readiness report, W7-X zonal recurrence and TEM/kinetic-electron extension are
``deferred``. In the broader research tracker, W7-X zonal recurrence remains
``open`` and W7-X fluctuation/TEM remains ``partial``. The W7-X
fluctuation-spectrum panel is a validated simulation diagnostic only; it is not
an experimental density-spectrum validation.

Pre-release checklist
---------------------

Before tagging a new public release:

1. Run the fast shard set, docs build, package build, repo hygiene, mypy, and
   wide coverage matrix.
2. Confirm the coverage workflow reports the package-wide ``95%`` gate and
   that ``coverage-wide-shard-manifest.json`` has labeled data for every wide
   coverage shard.
3. Confirm README and this page agree with
   ``docs/_static/manuscript_readiness_status.json`` and
   ``docs/_static/open_research_lane_status.json``.
4. Confirm runtime/performance claims point to fresh profiler artifacts for
   the exact backend, device count, problem size, and identity tolerance being
   claimed.
5. Bump the package version before tagging; PyPI rejects duplicate versions.
