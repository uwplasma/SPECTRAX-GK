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
       rejected on the eight-case train/holdout portfolio. The
       ``spectral_envelope_ridge`` candidate is accepted only as a scoped
       manuscript model-selection result. The passed
       ``quasilinear_model_selection_status.json`` gate does not promote a
       runtime/TOML absolute-flux predictor, universal nonlinear transport
       model, or user-facing saturation law. Any future absolute-flux
       promotion additionally requires finite passed nonlinear late-window
       convergence metadata for every holdout: transient cutoff, running-mean
       drift, block/bootstrap SEM, finite sample count, and source provenance.
       Electromagnetic quasilinear field-channel normalization and KBM
       calibration remain future gates.
   * - Differentiable geometry
     - release-ready for equal-arc parity and reduced QH/Li383 gates
     - The ``vmec_jax -> booz_xform_jax -> SPECTRAX-GK`` bridge is validated
       for zero-beta equal-arc field-line parity where the current
       ``mboz=nboz=21`` parity artifact passes. The fixed-resolution QI row
       now passes after the Boozer half-mesh convention fix, with drift
       mismatch about ``7.13e-2`` against the ``8e-2`` tolerance, and the
       evaluated QI ``ntheta=8,16`` variants pass. This is still not a broad
       QI transport or optimization claim. Reduced frequency, quasilinear, and
       nonlinear-window-estimator gradients pass AD/finite-difference gates on
       QH and Li383. The actual nonlinear
       finite-difference audits are startup plumbing checks with false
       transport-average gates; they do not validate production turbulence
       gradients.
   * - VMEC/Boozer reduced objectives
     - release-ready for reduced gradient and UQ plumbing
     - The public in-memory objective path supports reduced linear frequency,
       electrostatic quasilinear proxy, and smooth nonlinear-window-estimator
       objectives through the mode-21 VMEC/Boozer bridge. The QH and Li383
       holdout matrix is the citeable gate for these reduced objectives. This
       row does not promote multi-surface/multi-alpha optimization, calibrated
       absolute quasilinear flux prediction, or converged nonlinear heat-flux
       gradients.
   * - Stellarator optimization examples
     - release-ready as reduced examples plus selected optimized-equilibrium audit
     - The examples demonstrate differentiable reduced ITG objectives, UQ, and
       AD/finite-difference checks. The nonlinear objective is a reduced
       window-estimator path, not a nonlinear turbulence-gradient path. The
       selected optimized QA equilibrium now has a converged post-transient
       seed/timestep transport-window audit, so the production guard is closed
       for that scoped audit. Broad multi-surface nonlinear optimization and
       nonlinear turbulence gradients remain unpromoted.
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
- broad multi-surface production nonlinear heat-flux stellarator optimization;
- production nonlinear optimization without converged post-transient audits of
  optimized equilibria; the selected QA optimized-equilibrium audit is the
  current scoped exception;
- converged nonlinear transport gradients through ``vmec_jax`` and
  ``booz_xform_jax``;
- treating compact nonlinear finite-difference startup audits as saturated
  transport averages;
- treating reduced nonlinear-window estimators or startup finite-difference
  audits as optimized-equilibrium nonlinear heat-flux audit bars;
- multi-surface, multi-alpha, or multi-``k_y`` stellarator optimization from
  the current reduced single-fixture objective evidence;
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
       ``external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.*``,
       ``external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.*``,
       ``quasilinear_saturation_rule_sweep.*``,
       ``quasilinear_candidate_uncertainty.*``, and
       ``quasilinear_dataset_sufficiency.*``
     - Electrostatic diagnostics and manuscript model selection are in scope.
       Runtime absolute-flux prediction and electromagnetic calibration are not.
   * - Autodiff and differentiable geometry
     - ``autodiff_inverse_growth.*``, ``autodiff_inverse_twomode.*``,
       ``differentiable_geometry_bridge.*``, ``vmec_boozer_parity_matrix.*``,
       ``vmec_boozer_gradient_holdout_matrix.*``,
       ``nonlinear_window_fd_audit.*``, and
       ``vmec_boozer_nonlinear_window_fd_audit.*``
     - Reduced AD/finite-difference gates are in scope. Production nonlinear
       turbulence-gradient and optimized-equilibrium heat-flux claims are not.
   * - VMEC/Boozer objective and optimization checklist
     - ``vmec_boozer_solver_frequency_gradient_gate.*``,
       ``vmec_boozer_quasilinear_gradient_gate.*``,
       ``vmec_boozer_nonlinear_window_gradient_gate.*``,
       ``vmec_boozer_li383_solver_frequency_gradient_gate.*``,
       ``vmec_boozer_li383_quasilinear_gradient_gate.*``,
       ``vmec_boozer_li383_nonlinear_window_gradient_gate.*``,
       ``vmec_boozer_gradient_holdout_matrix.*``,
       ``vmec_boozer_multi_point_objective_gate.*``,
       ``vmec_boozer_reduced_portfolio_guard.json``,
       ``vmec_boozer_aggregate_line_search_comparison.*``,
       ``vmec_boozer_aggregate_alpha_holdout_gate.*``,
       ``vmec_boozer_aggregate_surface_holdout_gate.*``,
       ``vmec_boozer_second_equilibrium_aggregate_gate.*``,
       ``vmec_boozer_aggregate_holdout_promotion_gate.json``,
       ``nonlinear_window_ensemble_readiness_manifest.json``,
       ``nonlinear_window_convergence_reports/*.json``,
       ``stellarator_itg_optimization_comparison.*``, and
       ``stellarator_itg_optimization_uq.*``
     - These artifacts support reduced objective differentiability, optimizer
       plumbing, local UQ, and explicit nonlinear ensemble-readiness blockers.
       They do not support calibrated saturated-flux prediction, production
       nonlinear turbulence gradients, or nonlinear audits of optimized
       equilibria.
   * - Scope guardrails
     - ``technical_release_status.json``,
       ``parallelization_completion_status.*``,
       ``release_readiness.json``, ``manuscript_readiness_status.*``,
       ``open_research_lane_status.*``, and
       ``w7x_tem_extension_status.*``
     - These panels record what is closed, deferred, partial, or open; they do
       not promote the underlying deferred physics lanes by themselves.
   * - Performance and parallelization
     - ``runtime_memory_benchmark.*``,
       ``independent_ky_scan_scaling_large.*``,
       ``quasilinear_uq_ensemble_scaling_large.*``, and
       ``parallelization_completion_status.*``, plus
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
  ``passed = false`` with held-out mean relative error about ``2.11``.
- ``tools/check_nonlinear_window_convergence.py`` and
  ``spectraxgk.quasilinear_window`` provide the reusable late-window
  convergence metadata required before any future holdout report can be
  promoted to ``calibrated_absolute_flux``. This is a metadata/finite-window
  guardrail over existing traces, not a substitute for new long nonlinear
  simulations.
- ``spectraxgk.quasilinear_window.nonlinear_window_ensemble_report`` provides
  the next guardrail for replicated windows: seed, initial-condition, timestep,
  or restart variants must have individually passed late-window reports and
  mutually consistent late means before a nonlinear turbulent-flux optimization
  artifact can claim robustness. ``tools/check_nonlinear_window_ensemble.py``
  is the tracked artifact wrapper for this gate.
- ``tools/check_nonlinear_window_ensemble_readiness.py`` converts tracked
  transport-window summaries into explicit convergence-report JSON files and a
  readiness manifest. The older global
  ``docs/_static/nonlinear_window_ensemble_readiness_manifest.json`` remains a
  base-window manifest, but the current D-shaped and circular case-local
  replicate campaigns now pass their own ensemble gates. Those case-local
  artifacts supersede the stale global missing-replicate message for those two
  cases. The selected optimized-equilibrium audit now also passes its local
  seed/timestep ensemble gate.
- ``tools/check_vmec_boozer_aggregate_holdout_gate.py`` now requires a passed
  replicated nonlinear-window ensemble artifact in addition to aggregate
  finite-difference, line-search, and held-out surface/field-line evidence
  before any optimized-equilibrium production nonlinear heat-flux claim can be
  promoted. Single-window convergence reports remain necessary but insufficient
  for that claim level.
- ``tools/check_production_nonlinear_optimization_guard.py`` is the explicit
  production nonlinear turbulent-flux optimization guard. Its tracked artifact,
  ``docs/_static/production_nonlinear_optimization_guard.json``, passes release
  safety because reduced/startup estimators are blocked and two long
  post-transient replicated holdout ensembles pass. The selected optimized QA
  equilibrium also satisfies this guard because the ``t=[350,700]``
  seed/timestep replicated transport-window audit is attached; that is a scoped
  candidate audit, not a broad nonlinear transport-optimization claim.
- ``tools/build_baseline_optimized_nonlinear_audit.py`` now records the matched
  QA no-ESS reference to optimized QA/ESS comparison. The tracked
  ``docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json`` artifact passes
  with a relative ion-heat-flux reduction of ``0.184`` and a ``7.82`` combined
  SEM separation. This is a scoped finite-transform VMEC campaign comparison,
  not a broad multi-surface stellarator optimization claim.
- ``tools/check_nonlinear_turbulence_gradient_evidence.py`` is the stricter
  nonlinear turbulence-gradient claim gate. The tracked
  ``docs/_static/nonlinear_turbulence_gradient_evidence_status.json`` artifact
  passes the replicated long-window uncertainty side but fails closed on the
  gradient side. The current tracked production-candidate artifact is the
  re-equilibrated optimized QA/ESS ``ZBS(1,0)`` 5% campaign at ``t=[450,900]``:
  all baseline/plus/minus replicated nonlinear windows pass, and the finite
  difference has bounded response fraction, subtraction condition number, and
  forward/backward locality. The remaining blocker is propagated uncertainty:
  ``gradient_uncertainty_rel = 0.768`` exceeds the ``0.5`` gate. The companion
  ``ZBS(1,1)`` 5% campaign passes uncertainty at ``0.225`` but remains mildly
  nonlocal with ``fd_asymmetry_rel = 0.663``. The
  ``docs/_static/nonlinear_turbulence_gradient_evidence_gap_report.json`` now
  records this as a failed production-candidate gate, not as a missing campaign.
  Until a paired post-transient artifact passes all response, asymmetry,
  conditioning, and propagated uncertainty gates, nonlinear turbulence-gradient
  evidence remains explicitly unpromoted.
- ``tools/build_nonlinear_turbulence_gradient_fd_gate.py`` is the paired
  long-window promotion builder for that missing evidence. It takes the
  finished ``baseline``, ``plus_delta``, and ``minus_delta`` replicated
  nonlinear-window ensemble JSON files, computes the central finite-difference
  heat-flux gradient, propagates ensemble SEM into a gradient-uncertainty gate,
  writes reviewer-facing JSON/CSV/PNG/PDF sidecars, and fails closed unless the
  response, forward/backward asymmetry, condition number, and all three window
  uncertainty gates pass.
- Future perturbation refreshes must use distinct artifact slugs rather than
  overwriting the tracked failed candidate. For example, a new coefficient or
  amplitude campaign should write a slug such as
  ``docs/_static/qa_ess_zbs11_rel5_nonlinear_gradient_zbs_1_1_central_fd_gradient_gate.*``
  and a matching refreshed
  ``nonlinear_turbulence_gradient_evidence_status.json``. Release prose can
  promote the result only if the central finite-difference artifact passes
  and the evidence-status JSON reports the production gradient gate as true;
  otherwise it remains a documented production-candidate audit.
- ``tools/rank_nonlinear_turbulence_gradient_candidates.py`` ranks failed
  central finite-difference candidates without promoting them. The current
  ``docs/_static/nonlinear_turbulence_gradient_candidate_ranking.json`` summary
  compares the completed ``RBC(1,1)``, ``ZBS(1,1)``, and ``ZBS(1,0)`` campaigns
  and recommends an overdetermined least-squares/profile-gradient campaign next
  because the best single-control candidates have complementary locality and
  uncertainty failures.
- ``tools/summarize_nonlinear_gradient_bracket_sweep.py`` is the bounded
  follow-up for a same-control perturbation-amplitude sweep. It writes
  JSON/CSV/PNG/PDF sidecars from completed central finite-difference artifacts
  and recommends whether to add replicas, shrink/enlarge the bracket, or switch
  controls. It is deliberately not a promotion checker; it only promotes when
  one of the supplied long-window central-FD artifacts already passes all
  production gates.
- ``tools/write_overdetermined_nonlinear_gradient_campaign.py`` is the concrete
  launch-contract writer for that next campaign shape. The current tracked
  ``docs/_static/qa_ess_overdetermined_nonlinear_gradient_campaign_plan.json``
  uses the optimized-QA/ESS baseline input and prepares ``ZBS(1,1)``,
  ``ZBS(1,0)``, and ``RBC(1,1)`` controls at 3% relative amplitude with the
  same ``t=[450,900]`` analysis window. This artifact is planning/provenance
  only; it does not promote a nonlinear turbulence-gradient claim.
- ``tools/write_nonlinear_turbulence_gradient_campaign.py`` is the paired
  launch-contract writer for the same lane. Given explicit baseline,
  plus-perturbation, and minus-perturbation VMEC files, it writes the matched
  fixed-step nonlinear TOML ladders, per-state ensemble commands, the central
  finite-difference gate command, and the final evidence-check command. It
  fails closed before writing production launch contracts if any VMEC file is
  missing, if the same path is reused for more than one state, or if the three
  files have byte-identical SHA256 contents. The only override is
  ``--allow-identical-vmec-content``, which is recorded as a smoke-test-only
  manifest flag and must not be used for production turbulence-gradient claims.
- ``tools/write_vmec_boundary_perturbation_inputs.py`` is the upstream
  boundary-gradient launch helper. It writes matched ``baseline``,
  ``plus_delta``, and ``minus_delta`` VMEC input files for one explicit
  ``RBC/RBS/ZBC/ZBS(m,n)`` coefficient and records the ``vmec_jax`` commands
  that must be run before the resulting ``wout`` files can enter the
  nonlinear-gradient campaign writer.
- ``docs/_static/quasilinear_saturation_rule_sweep.json``:
  no simple saturation rule is accepted. Positive-growth mixing length is the
  least-bad simple rule with mean held-out relative error about ``2.11``;
  the training-mean null is about ``1.20``.
- ``docs/_static/quasilinear_candidate_uncertainty.json``:
  ``spectral_envelope_ridge`` is the accepted scoped candidate with mean
  relative error about ``0.295`` and interval coverage ``7/8`` on the
  current eight-case electrostatic-compatible portfolio. Its claim level is
  ``candidate_model_development_not_runtime_option``.
- ``docs/_static/quasilinear_holdout_gap_report.json``:
  absolute-flux promotion remains explicitly blocked. The
  ``absolute_flux_promotion_requirements`` block quantifies the current gap:
  the absolute train/holdout mean relative error is about ``6.04`` times the
  ``0.35`` gate, three additional independent passed holdouts are still
  required, one additional external-VMEC holdout family is required, and one
  non-axisymmetric external-VMEC holdout family is required before promotion
  can be reconsidered. These are evidence prerequisites, not a promoted
  runtime absolute-flux option.
- ``docs/_static/external_vmec_shaped_tokamak_pressure_t450_high_grid_convergence_gate.json``:
  finite shaped-tokamak pressure traces at ``t = 450`` are explicitly
  excluded from calibration because the ``n48``/``n64`` heat-flux windows
  differ by about ``0.306``, above the ``0.15`` grid-convergence gate. This is
  negative validation evidence, not an admitted holdout.

Nonlinear benchmark state:

- ``docs/_static/nonlinear_window_statistics.json`` records five passed
  release-window cases. KBM and HSX use tightened gates, Cyclone Miller is
  tighter than the broad release envelope, while Cyclone and W7-X remain at the
  ``0.10`` release envelope pending paper-level retuning.
- ``docs/_static/validation_gate_index.json`` currently records ``16`` passed
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
  resolution QI, and shaped-tokamak rows. The QI row
  ``nfp3_QI_fixed_resolution_final`` has drift mismatch about ``7.13e-2``
  against the ``8e-2`` release tolerance, and evaluated QI ``ntheta=8,16``
  robustness variants pass. The full declared QI seed campaign is still
  artifact-limited because three QI input variants have no bundled ``wout``
  reference. The builder rejects ``mboz,nboz < 21`` so QI is not silently
  evaluated on the under-resolved low-mode setting.
- ``docs/_static/vmec_boozer_gradient_holdout_matrix.json`` passes reduced
  linear, quasilinear, and nonlinear-window-estimator gradient gates on QH and
  Li383 with maximum relative mismatch about ``2.7e-2``.
- The VMEC/Boozer objective artifact checklist for README and manuscript use is
  the parity matrix, the six single-equilibrium frequency/quasilinear/reduced
  nonlinear-window gradient-gate figures, the combined holdout matrix, the
  multi-alpha aggregate objective gate, the reduced-portfolio provenance guard,
  the growth-vs-quasilinear line-search comparison, the positive reduced
  alpha-heldout and surface-heldout splits, the Li383 second-equilibrium
  aggregate gate, the blocked aggregate promotion JSON, and the reduced
  stellarator ITG optimization/UQ panels. This checklist is the current
  boundary between objective plumbing and transport prediction.
- ``docs/_static/vmec_boozer_reduced_portfolio_guard.json`` is the
  artifact-level guard that ties the backend-free portfolio reducer to real
  VMEC/Boozer rows. It requires VMEC/Boozer path/mode provenance, two
  field-line ``alpha`` values, two ``k_y`` samples, finite aggregate FD fields,
  finite growth/QL AD/FD objective gates, and an explicit non-production
  nonlinear claim boundary.
- ``docs/_static/nonlinear_window_fd_audit.json`` and
  ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.json`` pass only startup
  finite-difference plumbing checks. Both record ``transport_average_gate =
  false``.
- Finite-beta drift reconstruction, converged nonlinear turbulence gradients,
  held-out surface/field-line aggregate promotion, and optimized-equilibrium
  nonlinear audits remain future promotion gates.

Parallelization and performance state:

- ``docs/_static/independent_ky_scan_scaling_large.json`` and
  ``docs/_static/quasilinear_uq_ensemble_scaling_large.json`` support
  production independent-work parallelization for scans and ensembles.
- ``docs/_static/parallelization_completion_status.json`` is the release
  closure ledger for parallelization: production independent-work CPU/GPU
  scaling is closed, while nonlinear domain and FFT-axis decomposition remain
  diagnostic.
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
