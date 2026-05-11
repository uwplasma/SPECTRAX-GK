Release Scope and Claim Boundaries
==================================

This page is the canonical claim-scope checklist for the current development
state. It keeps README, documentation, release notes, and manuscript drafts
aligned with the tracked artifacts in ``docs/_static``. If a claim is not
listed here or in the referenced gate JSON, treat it as unpromoted.

Current scoped claims
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Lane
     - Status
     - Supported claim
   * - Linear and nonlinear benchmark atlas
     - release-ready
     - Linear growth/frequency/eigenfunction and nonlinear window statistics
       are validated for the tracked release cases. The nonlinear window
       statistics gate includes Cyclone, Cyclone Miller, KBM, W7-X, and HSX.
   * - Quasilinear diagnostics
     - release-ready as diagnostics
     - Electrostatic linear heat/particle weights, spectra, and model-selection
       artifacts are reproducible. Simple one-scalar saturation rules are
       rejected on the seven-case train/holdout portfolio. The
       ``spectral_envelope_ridge`` candidate is accepted only as a scoped
       model-development result, not as a runtime absolute-flux predictor.
   * - Differentiable geometry
     - release-ready for reduced gates
     - The ``vmec_jax -> booz_xform_jax -> SPECTRAX-GK`` bridge is validated
       for zero-beta equal-arc field-line parity at ``mboz=nboz=21`` on the
       tracked QH, QI, and shaped-tokamak fixtures. Reduced frequency,
       quasilinear, and nonlinear-window-estimator gradients pass AD/finite-
       difference gates on QH and Li383.
   * - Stellarator optimization examples
     - release-ready as reduced examples
     - The examples demonstrate differentiable reduced ITG objectives, UQ, and
       AD/finite-difference checks. They do not claim a production nonlinear
       heat-flux-optimized stellarator.
   * - Parallelization
     - production-ready for independent work
     - Independent ``k_y`` scans, quasilinear spectra, sensitivity batches, and
       UQ ensembles preserve serial ordering and have solver-backed scaling
       artifacts. Whole-state nonlinear sharding is a correctness/profiler gate
       only.
   * - Performance
     - release-ready for scoped profiler evidence
     - Runtime/memory panels, RHS profiler artifacts, and state-sharding
       identity checks are tracked. No broad nonlinear multi-GPU speedup or
       production domain-decomposition claim is made.

Explicitly unpromoted claims
----------------------------

Do not make these claims from the current artifacts:

- universal or user-facing absolute quasilinear flux prediction;
- electromagnetic quasilinear transport calibration for KBM;
- production nonlinear heat-flux stellarator optimization;
- converged nonlinear transport gradients through ``vmec_jax`` and
  ``booz_xform_jax``;
- broad W7-X validation beyond the tracked single-flux-tube ITG windows;
- W7-X TEM / kinetic-electron validation;
- W7-X long-window zonal recurrence/damping closure;
- nonlinear multi-GPU speedup from whole-state sharding;
- FFT-axis nonlinear domain decomposition.

Artifact-backed details
-----------------------

Quasilinear model-selection state:

- ``docs/_static/quasilinear_stellarator_train_holdout_report.json``:
  nonlinear inputs are valid, but the one-constant absolute-flux model remains
  ``passed = false`` with held-out mean relative error about ``2.57``.
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
- ``docs/_static/nonlinear_transport_time_horizon_audit.json`` separates
  long post-transient transport windows from startup finite-difference and
  reduced-envelope checks. Startup windows must never be described as saturated
  heat-flux averages.

Differentiable-geometry state:

- ``docs/_static/vmec_boozer_parity_matrix.json`` passes the current
  multi-equilibrium zero-beta equal-arc field-line convention gate at
  ``mboz=nboz=21``.
- ``docs/_static/vmec_boozer_gradient_holdout_matrix.json`` passes reduced
  linear, quasilinear, and nonlinear-window-estimator gradient gates on QH and
  Li383 with maximum relative mismatch about ``2.7e-2``.
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

These are tracked in ``docs/_static/open_research_lane_status.json`` and
``docs/_static/w7x_tem_extension_status.json``. The W7-X fluctuation-spectrum
panel is a validated simulation diagnostic only; it is not an experimental
density-spectrum validation.

Pre-release checklist
---------------------

Before tagging a new public release:

1. Run the fast shard set, docs build, package build, repo hygiene, mypy, and
   wide coverage matrix.
2. Confirm the coverage workflow reports the package-wide ``95%`` gate.
3. Confirm README and this page agree with
   ``docs/_static/manuscript_readiness_status.json`` and
   ``docs/_static/open_research_lane_status.json``.
4. Confirm runtime/performance claims point to fresh profiler artifacts for
   the exact backend, device count, problem size, and identity tolerance being
   claimed.
5. Bump the package version before tagging; PyPI rejects duplicate versions.
