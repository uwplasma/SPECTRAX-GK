Manuscript Figures
==================

Purpose
-------

This page tracks the target figure set for the future SPECTRAX-GK paper. A
figure is only ready for manuscript use when it has:

- one owning script,
- one reproducible artifact path,
- a declared reference,
- a declared acceptance status.

Current Readiness Snapshot
--------------------------

The current scoped manuscript stack is ready for claims about validated
quasilinear diagnostics/model selection, nonlinear-window comparison gates,
compact nonlinear startup-window finite-difference plumbing audits, mode-21
VMEC/Boozer geometry parity, reduced differentiable stellarator ITG
optimization examples, and linear/quasilinear VMEC/Boozer AD-vs-finite-
difference gradient gates on QH and Li383. The required release CI stack is the
quick-shard, docs/packaging, mypy, repo-hygiene, fast-coverage, and
wide-coverage matrix; treat the latest ``main`` run as the source of truth before
tagging. The companion ``docs/_static/manuscript_readiness_status.json`` report
currently has five active manuscript lanes closed and two lanes explicitly
deferred: W7-X zonal recurrence/damping and TEM / kinetic-electron stellarator
extension.

The broader plan is not fully closed. The current quasilinear figures are
publication-ready as diagnostics, model-selection evidence, and explicit
negative promotion gates, but they do not support a calibrated absolute-flux
predictor. The stellarator optimization figures are publication-ready for
reduced differentiable optimization/UQ plumbing and gradient validation, but
not yet for broad production nonlinear heat-flux optimization; the selected QA
optimized-equilibrium audit is a bounded positive audit, not a broad turbulent
optimization result. Those stronger claims require
converged post-transient nonlinear heat-flux windows, VMEC/Boozer nonlinear
turbulence-gradient or robust finite-difference gates, local-gradient
conditioning, and nonlinear audits of additional optimized equilibria. W7-X
zonal recurrence and TEM/kinetic-electron stellarator validation remain
deferred from the current manuscript scope.

The latest manuscript-stack additions are deliberately contract-level figures:
``docs/_static/quasilinear_holdout_gap_report.png`` with CSV/JSON/PDF
companions states exactly why absolute-flux promotion remains blocked;
``docs/_static/stellarator_objective_portfolio_gate.png``
validates the aggregate reduced-objective reducer used before expensive
VMEC/Boozer row production; and
``docs/_static/parallel_decomposition_status.png`` keeps production
independent-work parallelization claims separated from diagnostic nonlinear
whole-state/domain sharding metadata. The newer
``docs/_static/nonlinear_gradient_state_control_runbook.png`` is a claim
guardrail rather than a physics result: it shows that the QL-seeded
``Rsin_mid_surface_m1`` and ``Zcos_mid_surface_m1`` controls must be mapped to
perturbable VMEC input directions before nonlinear-gradient launches. The
newest ``LASYM=true`` ``RBS/ZBC`` response artifact provides that mapping with
rank ``2`` and condition number about ``1.02``, so the runbook now passes for
checked short-bracket launches. The older
``docs/_static/nonlinear_gradient_state_to_input_mapping_response.png`` is a
negative measured-response figure: the current stellarator-symmetric
``RBC/ZBS`` input perturbations do not move those asymmetric ``Rsin/Zcos``
state controls. The companion
``docs/_static/nonlinear_gradient_asymmetric_state_to_input_mapping_response.png``
is the positive symmetry-compatible mapping figure. The new
``docs/_static/nonlinear_gradient_state_control_short_bracket_launch_status.png``
then records that the mapped-control VMEC launch decks solved normally and that
bounded nonlinear campaign manifests are prepared, without promoting nonlinear
transport-gradient evidence. The follow-up
``docs/_static/nonlinear_gradient_state_control_short_bracket_nonlinear_audit_status.png``
records the first actual bounded nonlinear audit: all runtime and ensemble
window gates pass, but both finite-difference gradients fail closed because the
``1e-3`` bracket response is unresolved and asymmetric.

The new ``docs/_static/qa_low_turbulence_comparison.png`` panel adds the
aspect-6 QA low-turbulence optimization comparison requested for the
stellarator-design narrative. It is publication-ready for reduced
differentiable optimization plumbing, AD/finite-difference gates, and
side-by-side visualization of the control-only and transport-aware reduced
designs. It now labels the smooth heat-flux trace as the reduced
``Q_env = W_i E`` envelope and shows reduced ``|B|`` coloring on both the LCFS
surface renderings and the Boozer-LCFS maps. It is intentionally scoped away
from full VMEC/nonlinear-GK production claims. The companion
``docs/_static/qa_low_turbulence_time_horizon_audit.png`` panel checks the
reduced nonlinear-envelope horizon directly: the ``t=400`` late-window means
are unchanged at the ``1e-7`` level or better relative to a ``t=1000``
reference for the tracked designs, so the compact comparison panel does not
need to extend beyond ``t v_ti/a = 400``. The refreshed reduced objectives also
include a high-weight QA-compatible helical-shaping residual, so the displayed
LCFS surfaces are visibly non-axisymmetric rather than collapsing to an
axisymmetric boundary.
The companion solved-boundary guardrail
``docs/_static/vmec_jax_qa_transport_candidate_comparison.png`` is not a
promoted optimization result. It documents the VMEC-JAX/SPECTRAX-GK objective
assembly and WOUT-writing path, while deliberately failing closed when the
transport-weight refinement degrades the solved WOUT profile-iota and
quasisymmetry margins. The refreshed builder also treats gates reconstructed
from ``history.json`` and ``wout_final.nc`` as advisory only; paper-facing
admission requires an authoritative final ``solved_wout_gate.json``. A future
solved-boundary optimization figure must pass this full solved-candidate gate
before launching long-window nonlinear transport audits.
For VMEC-JAX replay-sensitive runs, the figure artifact must also state which
WOUT is authoritative. If ``wout_final_rerun.nc`` is chosen, require the
separate rerun-WOUT aspect/iota/QS gate and use that WOUT in the transport
audit commands; do not mix optimizer-state geometry with rerun-state
transport metrics.

Current Vs Deferred Figure Inventory
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 42 34

   * - Figure group
     - Current manuscript use
     - Deferred or blocked interpretation
   * - Benchmark atlas and nonlinear windows
     - Release atlas, nonlinear window statistics, eigenfunction overlays, and
       gate index support the scoped linear/nonlinear validation narrative.
     - ETG nonlinear pilots, TEM/KAW stress lanes, extra W7-X flux tubes, and
       non-indexed figures are not promoted release parity claims.
   * - Quasilinear diagnostics
     - Electrostatic spectra, shape gates, calibration provenance, negative
       simple-rule gates, and the ``spectral_envelope_ridge`` uncertainty panel
       support a model-selection result.
     - No runtime/TOML absolute-flux predictor, universal saturation law, or
       electromagnetic/KBM quasilinear calibration is promoted.
   * - Autodiff and VMEC/Boozer gradients
     - Inverse/UQ demos, zero-beta equal-arc parity, solver-ready gradients, and
       mode-21 QH/Li383 linear, quasilinear, and reduced nonlinear-window
       estimator gates are in scope.
     - Compact nonlinear FD audits are startup plumbing checks only; production
       nonlinear transport gradients and broader optimized-equilibrium audits
       beyond the selected QA candidate remain future gates.
   * - Performance and parallelization
     - Runtime/memory figures, independent ``k_y`` scan scaling,
       quasilinear/UQ ensemble scaling, and nonlinear RHS profiler artifacts are
       release-facing engineering evidence.
     - Production parallelization is the independent-work path. Whole-state
       nonlinear sharding and nonlinear domain sharding are identity/profiler
       evidence only unless the exact workload passes promotion gates.
   * - W7-X zonal and TEM guardrails
     - Open-research and TEM-status panels are useful guardrails for the paper
       plan and release notes.
     - W7-X long-window zonal recurrence, W7-X experimental fluctuation-spectrum
       validation, W7-X TEM/kinetic-electron nonlinear windows, and broad
       multi-flux-tube stellarator validation are deferred.

Core Validation Figures
-----------------------

.. list-table::
   :header-rows: 1

   * - Figure
     - Owning script
     - Status
     - Notes
   * - Linear benchmark master panel
     - ``tools/make_benchmark_atlas.py``
     - Closed for the release atlas; paper-level extensions remain scoped
     - Cyclone ITG, ETG, KBM, W7-X, HSX, and shaped tokamak coverage are represented in ``docs/_static/benchmark_core_linear_atlas.png``. TEM/kinetic-electron branch parity and additional W7-X multi-flux-tube scans remain outside the current release claim.
   * - Eigenfunction validation panel
     - ``tools/plot_eigenfunction_overlap_summary.py``, ``tools/plot_eigenfunction_reference_overlay.py``, ``tools/generate_kbm_reference_overlay.py``, and ``tools/generate_w7x_reference_overlay.py``
     - Closed for KBM and W7-X raw overlays
     - first shipped overlap artifact is ``docs/_static/kbm_eigenfunction_overlap_summary.png``; frozen raw GX bundles now exist for KBM and W7-X under ``docs/_static/reference_modes/``. The closed KBM raw overlay is ``docs/_static/kbm_eigenfunction_reference_overlay_ky0p3000.png`` with overlap ``0.999985`` and relative ``L^2`` mismatch ``0.00721``. The closed W7-X raw overlay is ``docs/_static/w7x_eigenfunction_reference_overlay_ky0p3000.png`` with overlap ``0.9999999994`` and relative ``L^2`` mismatch ``3.33e-5`` against the finite GX ``t≈2`` raw-mode bundle. Both overlay generators write JSON gate reports with ``overlap >= 0.95`` and ``relative L^2 <= 0.25`` requirements. ``tools/compare_gx_kbm.py --branch-summary-json`` writes branch-continuity gate metadata for selected KBM scans, and ``tools/generate_kbm_branch_gate_summary.py`` refreshes the no-rerun tracked artifact ``docs/_static/kbm_branch_gate_summary.json`` from ``docs/_static/kbm_gx_candidates.csv``. The current continuity-first branch summary passes the strict adjacent growth/frequency jump and successive-overlap gates.
   * - Nonlinear transport panel
     - ``tools/make_gx_summary_panel.py`` / ``tools/make_gx_publication_panel.py``
     - Closed for release-window gates; stricter manuscript tightening remains open
     - Cyclone, Miller, KBM, W7-X, HSX with matched windows. Current component artifacts: ``docs/_static/nonlinear_cyclone_diag_compare_t400.png``, ``docs/_static/nonlinear_cyclone_miller_diag_compare_t122.png``, ``docs/_static/nonlinear_kbm_diag_compare_t400_stats.png``, ``docs/_static/nonlinear_w7x_diag_compare_t200.png``, ``docs/_static/hsx_nonlinear_compare_t50_true.png``. ``tools/compare_gx_nonlinear_diagnostics.py --summary-json`` writes machine-readable mean-relative-mismatch gates for each plotted diagnostic with explicit transport-window bounds. The materialized release-window metadata are ``docs/_static/nonlinear_cyclone_gate_summary.json``, ``docs/_static/nonlinear_cyclone_miller_gate_summary.json``, ``docs/_static/nonlinear_kbm_gate_summary.json``, ``docs/_static/nonlinear_hsx_gate_summary.json``, and ``docs/_static/nonlinear_w7x_gate_summary.json``; all pass the current release gate. This is sufficient for the release validation atlas. Paper-level tightening remains open where case-specific references support narrower tolerances, and the older short Cyclone ``t=5`` diagnostic remains documented as an exploratory startup/resolved-spectrum audit, not a release gate.
   * - W7-X exact-state convention audit
     - ``tools/run_exact_state_audit.py`` and ``tools/plot_w7x_exact_state_audit.py``
     - Closed
     - current artifact base: ``docs/_static/w7x_exact_state_audit.png`` with CSV/JSON/PDF companions. It compares W7-X nonlinear VMEC startup state, late geometry/field arrays, and re-evaluated scalar diagnostics directly against GX exact-state dumps. The maximum finite pointwise relative error is ``4.62e-5`` under the explicit ``1e-4`` convention gate, while scalar diagnostics are below ``1.8e-7``. This closes the geometry/diagnostic convention layer but does not close the separate W7-X zonal-response recurrence lane.
   * - Windowed-statistics summary
     - ``tools/plot_nonlinear_window_statistics.py``
     - Closed for current release-window gates
     - current artifact base: ``docs/_static/nonlinear_window_statistics.png`` with CSV/JSON/PDF companions. It summarizes the per-diagnostic ``mean_rel_abs`` and ``max_rel_abs`` statistics from the frozen nonlinear GX comparison gate JSONs for Cyclone, Cyclone Miller, KBM, W7-X, and HSX. Exploratory/short-run diagnostics are explicitly excluded with ``gate_index_include=false``.
   * - Nonlinear startup-window finite-difference audit
     - ``tools/build_nonlinear_window_fd_audit.py``
     - Closed only as compact startup plumbing; transport-average and gradient promotion open
     - current artifact base: ``docs/_static/nonlinear_window_fd_audit.png`` with CSV/JSON/PDF companions. It runs actual compact SPECTRAX-GK nonlinear Cyclone startup windows at ``R/LTi = base +/- step`` plus a repeated base point, then checks finite outputs, repeatability, monotonic drive response, startup-window coefficient of variation, startup-window trend, and resolved central finite-difference response. The tracked response/base fraction is about ``0.111``. Its ``transport_average_gate`` is false because the run is too short for a post-transient running average; it is not a production nonlinear heat-flux, VMEC/Boozer nonlinear state-gradient, or optimized-equilibrium transport claim.
   * - VMEC/Boozer nonlinear startup finite-difference audit
     - ``tools/build_vmec_boozer_nonlinear_window_fd_audit.py``
     - Closed only as VMEC/Boozer geometry-perturbed startup plumbing; transport-average and local-gradient promotion open
     - current artifact base: ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.png`` with CSV/JSON/PDF companions. It starts from the real mode-21 ``vmec_jax -> booz_xform_jax`` QH state bridge, writes perturbed sampled geometries to temporary NetCDF files, and runs compact nonlinear startup windows at ``Rcos_mid_surface_m1 = base +/- 1e-5`` plus a repeated base point. The gate checks finite outputs, deterministic repeatability, bounded startup-window coefficient of variation and trend, resolved geometry perturbation, and resolved central finite-difference response; response/base is about ``0.040``. Its ``transport_average_gate`` is false and the forward/backward response is asymmetric, so this is a startup observable-path audit rather than a promoted transport average, local nonlinear gradient, or optimized-equilibrium transport claim.
   * - VMEC-state nonlinear-gradient launch runbook
     - ``tools/design_nonlinear_gradient_ql_seed_screen.py``, ``tools/design_nonlinear_gradient_state_control_runbook.py``, ``tools/write_vmec_state_to_input_mapping_campaign.py``, ``tools/write_vmec_asymmetric_state_to_input_mapping_campaign.py``, ``tools/write_vmec_state_control_short_bracket_launch.py``, and ``tools/build_vmec_state_to_input_mapping_response.py``
     - Closed for checked short-bracket launch mapping; long-window nonlinear-gradient evidence still required
     - current artifact bases: ``docs/_static/nonlinear_gradient_state_control_runbook.png``, ``docs/_static/nonlinear_gradient_state_to_input_mapping_campaign.png``, ``docs/_static/nonlinear_gradient_state_to_input_mapping_response.png``, ``docs/_static/nonlinear_gradient_asymmetric_state_to_input_mapping_campaign.png``, ``docs/_static/nonlinear_gradient_asymmetric_state_to_input_mapping_response.png``, ``docs/_static/nonlinear_gradient_state_control_short_bracket_launch_status.png``, ``docs/_static/nonlinear_gradient_state_control_short_bracket_nonlinear_audit_status.png``, and ``docs/_static/nonlinear_gradient_state_control_bracket_sweep_status.png`` with CSV/JSON/PDF companions. The QH/Li383 QL seed screen admits ``Rsin_mid_surface_m1`` and ``Zcos_mid_surface_m1`` as sign-consistent internal VMEC-state controls. The measured ``RBC/ZBS`` response matrix is rank zero, as expected for the symmetry-forbidden branch, but the follow-up ``LASYM=true`` ``RBS/ZBC`` response has rank ``2`` and condition number about ``1.02``. The runbook now carries explicit least-squares input-control directions for both admitted controls, and the short-bracket launch status records six normally terminated VMEC solves plus two prepared bounded nonlinear campaign manifests. The first nonlinear audit completes all ``18`` runs and passes output/ensemble gates, but both central-FD gates fail closed because the ``1e-3`` bracket response is too small and asymmetric. The follow-up ``3e-3``/``1e-2`` bracket-amplitude sweep completes all ``36`` office-GPU runs, but all four central-FD gates still fail with response fractions below ``0.005``. This is launch-mapping and negative single-control bracket evidence, not yet a converged long-window nonlinear-gradient result.
   * - Nonlinear-gradient control-variate campaign
     - ``tools/build_nonlinear_gradient_variance_reduction_plan.py`` and ``tools/write_nonlinear_gradient_control_variate_campaign.py``
     - Closed for the rel7.5 independent control-mean uncertainty gate; broader nonlinear-gradient claims remain scoped
     - current artifact bases: ``docs/_static/qa_ess_zbs10_rel7p5_variance_reduction_plan.png``, ``docs/_static/qa_ess_zbs10_rel7p5_control_variate_campaign_plan.png``, and ``docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.png`` with JSON/CSV/PDF companions. The rel7.5 ``ZBS(1,0)`` follow-up is local and response-resolved but variance limited. The midpoint common-mode control variate reduces apparent residual uncertainty to ``0.238``; the independent follow-up completes ``21`` matched plus/minus pairs and the strict late-window gate over ``t=[600,1100]`` passes with combined response uncertainty ``0.311 < 0.5``. This closes the evidence record for this specific variance-reduced nonlinear-gradient lane, not a universal nonlinear turbulent-flux optimization result.
   * - Nonlinear transport time-horizon audit
     - ``tools/build_nonlinear_transport_horizon_audit.py``
     - Closed as claim-scope guardrail; QH/CTH convergence promotion open
     - current artifact base: ``docs/_static/nonlinear_transport_time_horizon_audit.png`` with CSV/JSON/PDF companions. It audits the actual simulated time and claim scope for release nonlinear gates, startup finite-difference audits, reduced nonlinear-window estimators, and external-VMEC feasibility pilots. The new QH reduced-grid nonlinear pilot is extended from the earlier startup-scale ``t=20`` trace to ``t=150`` and reaches a meaningful late heat-flux window with mean about ``19.6``; it remains a feasibility result until a grid/window convergence gate passes. This panel prevents startup ``1e-11``-scale heat fluxes or reduced-envelope outputs from being described as post-transient nonlinear transport averages.
   * - Validation gate index
     - ``tools/make_validation_gate_index.py``
     - Closed for currently tracked gates
     - current artifact base: ``docs/_static/validation_gate_index.png`` and ``docs/_static/validation_gate_index.json``. This is not a physics result by itself; it is the audit panel for release-window gates, currently ``16/16`` passed.
   * - Open research lane status
     - ``tools/build_open_research_lane_status.py``
     - Closed as a claim-scope audit; underlying physics lanes remain scoped
     - current artifact base: ``docs/_static/open_research_lane_status.png`` with CSV/JSON/PDF companions. It reads the W7-X zonal recurrence, W7-X hypercollision probe, W7-X fluctuation/TEM extension status, quasilinear holdout, differentiable-geometry, and nonlinear-profiler artifacts and records which lanes are closed, partial, open, or blocked. The current status is intentionally conservative: nonlinear holdouts for the scoped quasilinear model-development claim and profiler-backed nonlinear hot-path localization are closed; W7-X fluctuation/TEM and differentiable geometry are partial bounded diagnostics; and W7-X long-window zonal recurrence/damping remains open. This panel is useful for the paper plan and release notes because it prevents partial diagnostics from being described as completed physics claims.
   * - Manuscript-readiness status panel
     - ``tools/build_manuscript_readiness_status.py``
     - Current manuscript scope with W7-X zonal and TEM deferred
     - current artifact base: ``docs/_static/manuscript_readiness_status.png`` with CSV/JSON/PDF companions. It records the narrower manuscript scope where W7-X zonal recurrence and TEM/kinetic-electron extensions are deferred. In that scope, quasilinear diagnostics and saturation-model selection are closed as a validated negative/model-selection result rather than as an absolute-flux predictor; VMEC/Boozer zero-beta equal-arc geometry parity is closed at ``mboz=nboz=21``; reduced differentiable stellarator ITG optimization is closed with AD/FD gates; and production solver-objective geometry gradients are closed for solver-ready arrays plus mode-21 VMEC/Boozer eigenfrequency, quasilinear heat-flux-weight, and reduced nonlinear-window estimator gates on QH and Li383. The compact nonlinear FD audits are retained only as startup plumbing checks with false transport-average gates. The production nonlinear optimization guard adds the D-shaped and circular long post-transient replicated holdout ensembles plus the selected optimized-equilibrium ``t=[350,700]`` seed/timestep replicated audit. Broader nonlinear turbulence-gradient, absolute-flux prediction, and multi-surface stellarator optimization claims remain separate gates.
   * - Aspect-6 QA low-turbulence optimization comparison
     - ``tools/build_qa_low_turbulence_comparison.py`` and ``tools/build_qa_low_turbulence_time_horizon_audit.py``
     - Closed for reduced differentiable optimization-plumbing claims
     - current artifact bases: ``docs/_static/qa_low_turbulence_comparison.png`` and ``docs/_static/qa_low_turbulence_time_horizon_audit.png`` with JSON/CSV/PDF companions. The panel compares a reduced QA constraints-only optimum against a reduced QA plus nonlinear-envelope optimum at aspect ``A = 6`` and minimum mean ``iota = 0.41``. It includes the fixed-``a/L_T`` ``Q_env`` versus ``a/L_n`` scan, fixed-gradient reduced-envelope traces, objective histories, reduced non-axisymmetric LCFS surfaces colored by ``|B|``, reduced Boozer-LCFS ``|B|`` maps, and gradient/constraint gates. The tracked artifact passes scalar, residual, and observable AD/finite-difference gates, runs the fixed-gradient reduced envelope to ``t v_ti/a = 400``, enforces the formal ``iota >= 0.41`` floor plus an operating ``iota >= 0.70`` floor, keeps a helical boundary amplitude near ``0.16``, and shows about ``10.7%`` reduced late-window heat flux at the fixed gradient. The horizon audit compares ``t=400`` against a ``t=1000`` reduced-envelope reference and finds relative late-window mean changes of ``1.2e-7`` or smaller, with CV/trend/half-window drift below ``1e-3``. The figure supports a reduced differentiable optimization and visualization claim only; it is not the upstream VMEC-JAX ``QA_optimization.py`` final WOUT, and long-window full nonlinear transport optimization remains governed by the production nonlinear audit gates.
   * - Quasilinear spectrum panel
     - ``tools/plot_quasilinear_spectrum.py``
     - Electrostatic diagnostic closed; absolute-flux prediction not promoted
     - current artifact bases: ``docs/_static/quasilinear_cyclone_spectrum.png``, ``docs/_static/quasilinear_cyclone_miller_spectrum.png``, ``docs/_static/quasilinear_hsx_spectrum.png``, and ``docs/_static/quasilinear_w7x_spectrum.png`` with CSV/JSON/PDF companions. They show electrostatic linear weights and explicitly uncalibrated mixing-length outputs from ``examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml``, ``examples/linear/axisymmetric/runtime_cyclone_miller_quasilinear.toml``, ``examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml``, and ``examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml``. Scan spectra use requested ``ky`` for the x-axis and retain signed selected-mode coordinates as ``mode_ky`` when applicable. W7-X was generated from an external VMEC benchmark file via ``W7X_VMEC_FILE``; the equilibrium itself is not shipped. Absolute saturated-flux claims remain open until a held-out nonlinear calibration report passes.
   * - Quasilinear calibration audit
     - ``tools/build_quasilinear_calibration_report.py`` and ``tools/plot_quasilinear_calibration.py``
     - Initial train/holdout artifact closed as a failed model-transfer gate
     - current artifact bases: ``docs/_static/quasilinear_cyclone_calibration_audit.png``, ``docs/_static/quasilinear_cyclone_miller_calibration_audit.png``, ``docs/_static/quasilinear_cyclone_miller_train_holdout.png``, ``docs/_static/quasilinear_hsx_train_holdout.png``, ``docs/_static/quasilinear_w7x_train_holdout.png``, the manuscript-facing combined panel ``docs/_static/quasilinear_stellarator_train_holdout.png``, and the input-provenance audit ``docs/_static/quasilinear_validated_calibration_inputs.png`` with JSON/PDF companions. The current one-constant train/holdout report fits the heat-flux scale on Cyclone and the external-VMEC ITERModel case, then scores six held-out windows: Cyclone Miller, HSX, W7-X, D-shaped external VMEC, up-down asymmetric external VMEC, and circular external VMEC. It intentionally remains ``calibration_dataset`` with ``passed = false`` because held-out errors exceed the ``0.35`` gate; the current holdout mean relative error is about ``2.11``. HSX, W7-X, and the up-down asymmetric VMEC point are useful negative stellarator/tokamak transfer checks because the simple positive-growth mixing-length family predicts little or near-zero flux while the nonlinear windows are finite. The D-shaped external-VMEC point is a converged negative transfer constraint with finite late-window nonlinear heat flux but a grossly overpredicted Cyclone/ITERModel-scaled mixing-length estimate. The newly admitted circular external-VMEC point is a positive transfer check for the scaled one-constant diagnostic, but it does not rescue the aggregate absolute-flux gate. The input audit confirms that every current train/holdout nonlinear artifact maps to a passed nonlinear gate, while failed QH and CTH-like external-VMEC feasibility pilots remain excluded. This closes the NetCDF/CSV calibration machinery and provenance gate but not a calibrated absolute-flux claim.
   * - Quasilinear saturation-rule sweep
     - ``tools/plot_quasilinear_saturation_rule_sweep.py``
     - Initial model-development diagnostic closed as a negative result
     - current artifact base: ``docs/_static/quasilinear_saturation_rule_sweep.png`` with JSON/PDF companions. It fits one scalar on the two training cases, Cyclone and external-VMEC ITERModel, then scores the same six held-out windows for three simple rules: positive-growth mixing length, raw linear heat-flux weight, and an absolute-growth diagnostic. All three fail the held-out absolute-flux gate. The least-bad simple rule is now positive-growth mixing length with holdout mean relative error about ``2.11``; raw linear weight is about ``2.68`` and the absolute-growth diagnostic is about ``3.32``. The panel also includes a training-mean null baseline with holdout mean relative error about ``1.20``. Its JSON ``promotion_gate`` has no accepted rules, so future calibrated rules must beat both the quasilinear baselines and this null baseline before being promoted. This supports the next saturation-model lane while preventing premature absolute quasilinear transport claims.
   * - Shape-aware quasilinear saturation diagnostic
     - ``tools/plot_quasilinear_shape_aware_saturation.py``
     - Initial leave-one-geometry-out diagnostic closed as a negative result
     - current artifact base: ``docs/_static/quasilinear_shape_aware_saturation.png`` with JSON/PDF companions. It fits a shared nonlinear/quasilinear spectrum-shape exponent with per-case intercepts, uses only passed shape gates for the exponent fit, then fits the absolute heat-flux scale on training cases and scores each held-out geometry. The shape-aware model gives mean absolute relative error about ``0.664`` versus ``0.624`` for the linear-weight baseline and ``0.170`` for a deliberately simple training-mean null baseline. The JSON ``promotion_gate`` is false because the model fails the ``0.35`` transport gate and does not beat the null baseline on the current four-case dataset. This is retained as a manuscript-facing negative result because it rules out a too-simple one-exponent envelope before stellarator optimization claims.
   * - Quasilinear candidate uncertainty gate
     - ``tools/plot_quasilinear_candidate_uncertainty.py``
     - Eight-case uncertainty-aware candidate gate closed as a scoped model-selection result
     - current artifact base: ``docs/_static/quasilinear_candidate_uncertainty.png`` with JSON/PDF companions. It adds training-residual ``95%`` prediction intervals to leave-one-geometry-out candidate scoring on the current eight-case electrostatic-compatible dataset. The legacy calibrated linear-weight and one-exponent shape-power-law candidates remain rejected relative to the null/skill gates. The accepted research candidate is ``spectral_envelope_ridge``: it uses the positive-growth ``k_y`` centroid and heat-flux-weighted ``k_y`` width in a three-parameter log-linear ridge model, reaches leave-one-geometry-out mean relative error about ``0.295``, and has interval coverage ``7/8``. This is a bounded model-development result, not a runtime/TOML absolute-flux predictor or a universal saturation law.
   * - Stellarator quasilinear usefulness summary
     - ``tools/plot_quasilinear_stellarator_usefulness.py``
     - Stellarator-facing model-skill and scope summary closed as a scoped model-development figure
     - current artifact base: ``docs/_static/quasilinear_stellarator_usefulness.png`` with CSV/JSON/PDF companions. It compares admitted HSX and W7-X nonlinear heat-flux windows with the simple positive-growth mixing-length rule, the calibrated linear-weight rule, and the accepted ``spectral_envelope_ridge`` candidate. The simple rule predicts zero for HSX and W7-X while the nonlinear windows are finite, so it is explicitly rejected as a stellarator absolute-flux proxy. The spectral-envelope candidate is much closer on those two admitted stellarator holdouts and remains the best scoped model-selection result, but the figure also records that QA is matched-nonlinear-audit-only and QH is excluded until grid/window convergence passes. This figure is a manuscript-facing usefulness/limitation summary, not a runtime/TOML absolute-flux predictor.
   * - Quasilinear screening and rank-correlation skill
     - ``tools/plot_quasilinear_screening_skill.py``
     - Screening/correlation gate closed for the scoped spectral-envelope candidate; absolute-flux promotion remains blocked
     - current artifact base: ``docs/_static/quasilinear_screening_skill.png`` with CSV/JSON/PDF companions. It scores the current simple rules and candidate models by Spearman rank correlation, pairwise order accuracy, and mean relative error on the eight-case electrostatic portfolio. Only ``spectral_envelope_ridge`` passes the full-portfolio screening gate and the mean-error gate, with Spearman correlation about ``0.81`` and pairwise order accuracy about ``0.79``. The held-out-only gate is plotted separately and currently remains below threshold (about ``0.71`` Spearman and ``0.73`` pairwise order accuracy), so the figure deliberately reports ``absolute promotion: none`` and points to additional independent nonlinear holdouts as the next requirement.
   * - Quasilinear dataset-sufficiency gate
     - ``tools/plot_quasilinear_dataset_sufficiency.py``
     - Promotion guard closed for the scoped spectral-envelope candidate; higher-parameter and electromagnetic claims remain blocked
     - current artifact base: ``docs/_static/quasilinear_dataset_sufficiency.png`` with JSON/PDF companions. It audits the validated nonlinear windows before any richer saturation model is promoted. The current electrostatic-compatible set has eight cases, two explicit training geometries, and six held-out geometries. That is sufficient for the one-parameter linear-weight candidate, the two-parameter shape-power-law candidate, and the three-parameter ``spectral_envelope_ridge`` candidate, but not for the five-parameter ``linear_state_ridge`` model. KBM is explicitly excluded from this electrostatic quasilinear promotion lane because electromagnetic field-channel normalization and calibration remain future work. The gate therefore supports the scoped spectral-envelope model-selection claim while preventing broader absolute-flux or electromagnetic quasilinear claims.
   * - Quasilinear model-selection status
     - ``tools/plot_quasilinear_model_selection_status.py`` and :mod:`spectraxgk.quasilinear_model_selection`
     - Scoped model-selection lane closed; absolute-flux runtime promotion remains blocked
     - current artifact base: ``docs/_static/quasilinear_model_selection_status.png`` with CSV/JSON/PDF companions. It consolidates the dataset-sufficiency gate, uncertainty/skill gate, and tracked train/holdout calibration reports into one claim-boundary panel. The accepted ``spectral_envelope_ridge`` candidate reaches leave-one-geometry-out mean relative error about ``0.295`` with prediction-interval coverage ``7/8`` and beats both the training-mean null and linear-weight baselines. The same artifact also verifies that no tracked train/holdout report is promoted to ``calibrated_absolute_flux``. This is the manuscript-facing positive result for reduced candidate selection, not a runtime/TOML absolute-flux predictor.
   * - Quasilinear holdout-gap report
     - ``tools/build_quasilinear_holdout_gap_report.py``
     - Absolute-flux promotion boundary quantified; next nonlinear holdout requirements explicit
     - current artifact base: ``docs/_static/quasilinear_holdout_gap_report.png`` with CSV/JSON/PDF companions. It keeps ``absolute_flux_promoted=false`` and records both ``absolute_flux_promotion_requirements`` and ``screening_promotion_requirements`` blocks. The current absolute train/holdout error is about ``2.11`` against the ``0.35`` gate, i.e. about ``6.04`` times too large, with Cyclone Miller as the worst admitted holdout. The full-portfolio screening gate passes for ``spectral_envelope_ridge``, but the held-out-only screening gate remains below threshold (about ``0.71`` Spearman and ``0.73`` pairwise order accuracy). The report requires three additional independent passed holdouts, one additional external-VMEC holdout family, and one non-axisymmetric external-VMEC holdout family before absolute-flux or held-out screening promotion can even be reconsidered.
   * - External-VMEC next-holdout runbook
     - ``tools/build_external_vmec_holdout_runbook.py`` and :mod:`spectraxgk.external_holdout_plan`
     - Launch-plan artifact closed; unchanged same-family and failed-family replays blocked
     - current artifact base: ``docs/_static/external_vmec_next_holdout_runbook.png`` with CSV/JSON/PDF companions. It converts the holdout-gap report and external-VMEC linear screen into a fail-closed nonlinear launch plan. After the circular external-VMEC case closed at ``t=450`` and entered the holdout set, the shaped-tokamak pressure candidate was run to ``t=450`` at ``n48``/``n64`` but failed the high-grid convergence gate with common and least-window relative heat-flux differences about ``0.306``. The follow-on ``ITERModel_reference_nc`` same-family audit passes at ``t=450`` with common/least grid differences about ``0.056``/``0.055``, but it is scoped as reproducibility evidence because ITERModel is already consumed by the training reference. The refreshed linear screen includes the tracked CTH-like spectrum point with ``gamma≈0.0488`` at ``ky≈0.2857``. The current runbook selects that row only through an explicit modified-protocol contract: ``n48/n64/n80`` grids and ``t=150,250,350`` horizons after the earlier CTH-like ``32->48`` grid/window failure. The same runbook still blocks unchanged failed-family replays, ``QI_stel_seed_3127`` remains near-marginal with ``gamma≈3.8e-3``, and ``basic_non_stellsym`` remains a VMEC flux-tube contract failure. This records the fail-closed admission requirements: ``split=holdout``, sufficiently unstable linear branch, passed grid/window convergence, post-transient transport window, and independence from the training reference. This is a planning artifact only; it does not promote an absolute-flux predictor.
   * - Quasilinear promotion guardrail audit
     - ``tools/check_quasilinear_promotion_guardrails.py``
     - Fast metadata gate closed; nonlinear simulation validation remains delegated to the source gates
     - current artifact: ``docs/_static/quasilinear_promotion_guardrails.json``. It scans the train/holdout calibration reports, saturation-model reports, nonlinear input-validation blocks, promotion gates, claim-scope README/docs wording, the quasilinear row in ``docs/_static/manuscript_readiness_status.json``, and the manuscript quasilinear model-development figure index. It requires finite nonlinear window means and standard deviations for train/holdout calibration points, explicit nonlinear and quasilinear artifact provenance, JSON sidecars for the tracked model-development figures, scoped non-absolute claim levels, explicit failed-baseline or blocker metadata, passed held-out gates before any ``calibrated_absolute_flux`` claim, and a manuscript-readiness quasilinear lane that remains scoped as diagnostic/model-selection evidence rather than a runtime absolute-flux predictor. This is deliberately a wording and metadata guard, not a calibrated absolute-flux claim or a replacement for nonlinear convergence simulations.
   * - Release claim-scope ledger
     - ``docs/release_scope.rst``
     - Closed as documentation guardrail
     - This documentation page centralizes the current claim boundaries across validation, quasilinear model selection, differentiable geometry, parallelization, performance, and deferred W7-X/TEM lanes. It should be updated whenever a new artifact promotes or demotes a claim.
   * - VMEC equilibrium portfolio for future quasilinear holdouts
     - ``tools/plot_vmec_jax_equilibrium_inventory.py``
     - Planning artifact closed; bounded linear smoke checks started; transport validation open
     - current artifact bases: ``docs/_static/vmec_jax_equilibrium_inventory.png``, ``docs/_static/external_vmec_candidate_linear_screen.csv``, ``docs/_static/quasilinear_vmec_qi_seed_linear_spectrum.png``, ``docs/_static/quasilinear_vmec_qi_seed_branch_refinement_gate.png``, ``docs/_static/quasilinear_vmec_dshape_linear_spectrum.png``, ``docs/_static/external_vmec_dshape_grid_convergence_gate.png``, ``docs/_static/external_vmec_dshape_t250_high_grid_convergence_gate.png``, ``docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.png``, ``docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.png``, ``docs/_static/quasilinear_vmec_jax_qh_linear_spectrum.png``, ``docs/_static/quasilinear_vmec_jax_cth_like_linear_spectrum.png``, ``docs/_static/external_vmec_qh_nonlinear_t150_pilot.png``, ``docs/_static/external_vmec_qh_nonlinear_t150_n48_pilot.png``, ``docs/_static/external_vmec_qh_nonlinear_t150_n64_pilot.png``, ``docs/_static/external_vmec_qh_grid_convergence_gate.png``, ``docs/_static/external_vmec_qh_high_grid_convergence_gate.png``, ``docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.png``, ``docs/_static/external_vmec_cth_like_nonlinear_t150_n48_pilot.png``, and ``docs/_static/external_vmec_cth_like_grid_convergence_gate.png`` with JSON/PDF companions. The inventory scans external VMEC files from ``vmec_jax/examples/data`` without checking them into SPECTRAX-GK and now records ``11`` equilibria, including the newly detected ``wout_QI_stel_seed_3127.nc``. A broader five-point candidate screen selected DSHAPE as the strongest finite unstable branch with ``gamma≈0.096`` at ``ky≈0.476``. The refreshed local portfolio records li383 as stable, QI seed as near-marginal rather than launchable, and basic non-stellarator-symmetric geometry as a flux-tube contract failure; the QI branch-refinement gate passes finite/positive-run/Krylov-consistency subgates but fails nonlinear-launch growth because ``max(gamma)≈3.8e-3 < 0.02``. DSHAPE passes low-to-mid-grid convergence at ``t=150`` and passes the ``48x48x32`` to ``64x64x40`` high-grid gate after extension to ``t=250``: common-window and least-window symmetric relative differences are about ``0.139`` and ``0.108``, below the ``0.15`` threshold. The follow-up ``64x64x40`` DSHAPE seed/timestep replicate campaign passes the late-window ensemble gate on ``t=[170,250]`` with mean-relative spread ``0.141`` and combined SEM/mean ``0.054``. The independent circular external-VMEC holdout also has replicated evidence after extending from a non-promotable ``t=450`` terminal-window drift to a closed ``t=700`` window: ``docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.png`` passes with mean-relative spread ``0.035`` and combined SEM/mean ``0.043`` on ``t=[350,700]``. QH and CTH-like remain useful feasibility and negative convergence results rather than transport validation claims: QH fails both ``32->48`` and ``48->64`` gates, and CTH-like fails its first grid check. DSHAPE and circular tokamak are ready for calibration-report admission with replicated nonlinear-window uncertainty evidence; QH and CTH-like should stay excluded until production-resolution convergence gates pass.
   * - Quasilinear spectrum-shape gate
     - ``tools/plot_quasilinear_spectrum_shape_gate.py``
     - HSX, W7-X, and Cyclone Miller gates closed; Cyclone retained as a failed model gate
     - current artifact bases: ``docs/_static/quasilinear_hsx_spectrum_shape_gate.png``, ``docs/_static/quasilinear_w7x_spectrum_shape_gate.png``, ``docs/_static/quasilinear_cyclone_miller_spectrum_shape_gate.png``, and ``docs/_static/quasilinear_cyclone_spectrum_shape_gate.png`` with JSON/PDF companions. They compare normalized linear heat-flux-weight spectra against normalized nonlinear resolved ``HeatFlux_kyst`` spectra. HSX passes with ``TV≈0.112`` and cosine ``≈0.971``; W7-X passes with ``TV≈0.056`` and cosine ``≈0.992``; Cyclone Miller passes with ``TV≈0.094`` and cosine ``≈0.983``; Cyclone is kept as a failed gate with ``TV≈0.215`` and cosine ``≈0.896``. This supports spectrum-shape diagnostics while identifying a real saturation/window/model limitation before absolute saturated-flux claims. KBM is deferred from this gate because the current quasilinear diagnostic validates electrostatic channels only and the KBM lane is electromagnetic.
   * - Zonal-flow / GAM response panel
     - ``tools/plot_zonal_flow_response.py``, ``tools/plot_zonal_flow_response_from_output.py``, ``tools/generate_miller_zonal_response_pilot.py``, ``tools/generate_w7x_zonal_response_panel.py``, ``tools/digitize_w7x_zonal_reference.py``, ``tools/compare_w7x_zonal_reference.py``, ``tools/plot_w7x_zonal_contract_audit.py``, ``tools/plot_w7x_zonal_moment_tail_audit.py``, ``tools/plot_w7x_zonal_closure_ladder.py``, ``tools/plot_w7x_zonal_state_convention_audit.py``, and ``tools/plot_w7x_zonal_recurrence_sweep.py``
     - Open
     - should combine shaped-tokamak Rosenbluth-Hinton-style residuals with W7-X residual/damping envelopes using one figure convention; use signed ``Phi_zonal_mode_kxt`` or case-specific signed line averages for publication claims and keep ``Phi2_zonal_t`` only as an intermediate cross-check. The current Merlo Case-III artifact is ``docs/_static/miller_zonal_response_pilot.png`` from the initial-density setup at ``Nz=32``, ``Nl=4``, ``Nm=24``, ``dt=0.005``, and ``t≈60``. With Rosenbluth-Hinton first-sample normalization it gives ``residual≈0.192`` against the Merlo et al. Figs. 12/16 read-off of about ``0.19``; a literature-faithful common fit window ``t≈30`` with separate positive/negative-extrema damping fits gives ``γ_GAM R0 / v_i≈-0.176`` against the paper-scale read-off near ``-0.17``; and Hilbert-phase frequency extraction on that same window gives ``ω_GAM R0 / v_i≈2.20`` against the paper-scale read-off near ``2.24``. A higher-moment audit lowers the recurrence ratio but over-damps the GAM, while weak hypercollision scans are effectively inert, so the frozen Merlo artifact remains on the current ``Nm=24`` baseline. The W7-X side now uses the potential initializer, signed line-average observable, paper-facing line-first normalization, and no hidden time-axis scaling. The tracked long-window W7-X artifact is ``docs/_static/w7x_zonal_response_panel.png`` with replayable traces in ``docs/_static/w7x_zonal_response_panel.traces.csv``; it reaches the digitized Fig. 11 windows, but ``docs/_static/w7x_zonal_reference_compare.json`` remains open because residuals fail at ``k_x rho_i=0.07``, ``0.10``, and ``0.30`` and the late envelopes are much larger than the digitized stella/GENE traces. ``docs/_static/w7x_zonal_contract_audit.png`` is now the paper-facing diagnostic panel for that open mismatch and is intentionally excluded from the release gate index. ``docs/_static/w7x_zonal_state_convention_audit.png`` closes the paper-facing state convention layer: the recovered Gaussian potential has relative ``L2`` error ``1.85e-6``, off-target spectral content is zero to reported precision, and the diagnostic helpers agree with manual line/volume reductions near ``2e-16``. ``docs/_static/w7x_zonal_moment_tail_audit.png``, ``docs/_static/w7x_zonal_closure_ladder_kx070.png``, ``docs/_static/w7x_zonal_recurrence_sweep_kx070.png``, and ``docs/_static/w7x_zonal_hypercollision_probe_kx070.png`` are companion open diagnostics; together they support a recurrence / moment-closure hypothesis and show that weak or constant closure can reduce velocity-space tails without closing the paper trace. The refreshed closure ladder now covers constant Hermite, ``k_z``-weighted Hermite, mixed Laguerre-Hermite, Laguerre-only, and isotropic hypercollision families at ``0.01`` and ``0.03``. The best mean trace error is the isotropic ``nu_hyper=0.01`` row at about ``0.2755`` versus baseline ``0.2861``, but its late-window standard-deviation ratio is about ``4.25`` versus baseline ``4.10``. Thus no bounded closure family improves trace error, late-envelope recurrence, and moment-tail metrics simultaneously. The W7-X generator exposes explicit hypercollision and Gaussian-width audit overrides so future closure probes are reproducible from the tracked tool. A newer high-moment four-wavelength audit under ``tools_out/zonal_response/w7x_publication_nl16_nm64_dt005_t100`` verifies finite signed traces to ``t≈100`` after restart-continuation fixes. The tracked W7-X TOML keeps ``gaussian_width=1`` because the benchmark source writes the initializer as ``exp[-(z-z0)^2]``; wider profiles and non-unit time scales are retained only as audits. The lane remains open pending a more physical W7-X damping/closure and velocity-space recurrence fix under the paper-facing convention.
   * - W7-X fluctuation-spectrum panel
     - ``tools/plot_w7x_fluctuation_spectrum_panel.py`` and ``tools/build_w7x_tem_extension_status.py``
     - Initial simulation-spectrum diagnostic closed; TEM/multi-flux validation open
     - current artifact bases: ``docs/_static/w7x_fluctuation_spectrum_panel.png``, ``docs/_static/tem_branch_parity_audit.png``, and ``docs/_static/w7x_tem_extension_status.png`` with CSV/JSON/PDF companions. The fluctuation panel is regenerated from the gated W7-X nonlinear ``t≈200`` NetCDF artifact, requires the corresponding nonlinear gate summary to pass before plotting, and records ``gate_index_include=false`` because it is a diagnostic figure rather than an additional release gate. It shows normalized ``k_y`` spectra for ``|\phi|^2``, ``W_\phi``, and ``|Q_i|``, the time-averaged ``k_x``-``k_y`` fluctuation-power map, the signed heat-flux spectrum, and a windowed temporal spectrum for the dominant nonzonal and zonal traces. The TEM audit explicitly keeps TEM linear parity open: maximum absolute relative growth-rate mismatch is about ``4.25``, maximum absolute relative frequency mismatch is about ``3.3`` away from the near-zero reference denominator, and the frequency branch has Spearman coefficient about ``-0.986``. Because the TEM reference is a provisional literature digitization rather than a direct case dump, this artifact blocks broad W7-X/TEM validation claims without being a standalone tuning target. The extension-status panel also keeps W7-X multi-alpha/multi-surface scans and kinetic-electron nonlinear windows open. This closes the reproducible simulation-spectrum panel needed for the current manuscript stack but not broad W7-X/TEM validation.
   * - Velocity-space convergence panel
     - ``tools/generate_observed_order_gate.py`` plus dedicated full convergence refresh script to add
     - Open
     - should follow GX-style convergence evidence and write an observed-order gate report through ``spectraxgk.benchmarking.observed_order_gate_report`` so convergence rate and final-grid error are tracked explicitly. The current atlas summary already records a machine-readable high-vs-low Cyclone grid convergence gate for the tracked convergence tile. The CSV-backed Cyclone velocity-space artifact is ``docs/_static/cyclone_resolution_observed_order.png`` with metadata in ``docs/_static/cyclone_resolution_observed_order.json``; the current office/GPU ``ky=0.30`` sweep passes the strict pairwise-order and final-error gate.
   * - Stellarator validation panel
     - dedicated script to add
     - Open
     - W7-X multi-flux-tube + zonal-flow response + HSX summary as needed; add heavy-electron EM verification before realistic-electron EM claims
   * - Performance panel
     - existing performance tooling
     - Closed for release-level scoped claims
     - current artifact bases: ``docs/_static/runtime_memory_benchmark.png``, ``docs/_static/nonlinear_rhs_profile_miller.png``, ``docs/_static/nonlinear_rhs_profile_stellarator_runtime.png``, ``docs/_static/full_nonlinear_rhs_trace_summary.json``, ``docs/_static/full_nonlinear_rhs_trace_gpu_summary.json``, and ``docs/_static/nonlinear_sharding_profile_office_gpu.json``. The release claim is limited to current cold/warm runtime accounting, CPU/GPU nonlinear RHS hot-path localization, W7-X/HSX runtime-mode stellarator smoke profiles, and numerical-identity gates. It is not a production nonlinear domain-decomposition speedup claim.
   * - Parallelization identity gate
     - ``tools/generate_parallel_ky_scan_gate.py``
     - Closed for independent Cyclone ``k_y`` batching
     - current artifact base: ``docs/_static/parallel_ky_scan_gate.png`` with CSV/JSON/PDF companions. This is a real linear-solver gate: it compares serial and fixed-shape ``k_y``-batched Cyclone scans, requires numerical identity for ``gamma`` and ``omega``, and reports batch speedup separately from the acceptance criterion.

Differentiable-Physics Figures
------------------------------

.. list-table::
   :header-rows: 1

   * - Figure
     - Owning script
     - Status
     - Notes
   * - Sensitivity-analysis figure
     - ``examples/theory_and_demos/quasilinear_implicit_sensitivity.py``
     - Initial quasilinear eigenbranch gate closed
     - current artifact base: ``docs/_static/quasilinear_implicit_sensitivity.png`` with JSON/PDF companions. It differentiates a tiny Cyclone linear-RHS quasilinear objective ``[gamma, omega, kperp_eff^2, Qhat_i, Q_i^ML]`` with respect to ``[R/Ln, R/LTi]`` using the implicit left/right eigenpair system and checks the result against nearest-branch central finite differences. This is a differentiability/conditioning gate, not an absolute nonlinear-flux validation.
   * - Solver-objective geometry-gradient gate
     - ``tools/build_solver_objective_gradient_gate.py``, ``tools/build_vmec_boozer_solver_frequency_gradient_gate.py``, ``tools/build_vmec_boozer_quasilinear_gradient_gate.py``, ``tools/build_vmec_boozer_nonlinear_window_gradient_gate.py``, ``tools/build_vmec_boozer_gradient_holdout_matrix.py``, ``tools/build_vmec_boozer_multi_point_objective_gate.py``, ``tools/build_vmec_boozer_aggregate_line_search_comparison.py``, ``tools/build_vmec_boozer_aggregate_alpha_holdout_gate.py``, ``tools/build_vmec_boozer_aggregate_surface_holdout_gate.py``, ``tools/build_vmec_boozer_second_equilibrium_aggregate_gate.py``, ``tools/check_vmec_boozer_aggregate_holdout_gate.py``, ``tools/check_production_nonlinear_optimization_guard.py``, ``tools/build_nonlinear_window_fd_audit.py``, and ``tools/build_vmec_boozer_nonlinear_window_fd_audit.py``
     - Solver-ready linear-RHS gradient gate closed; mode-21 VMEC/Boozer state-to-solver eigenfrequency, quasilinear heat-flux-weight, and reduced nonlinear-window estimator gates closed for QH plus Li383; compact and VMEC/Boozer-perturbed nonlinear startup FD audits closed only as plumbing checks; multi-alpha reduced aggregate objective plumbing passes; selected optimized-equilibrium post-transient transport-window audit closed; earlier aspect-6 projected transport-gradient candidate and strict rerun-WOUT top-12 QA candidate audited as negative long-window transfer results; nonlinear turbulence-gradient promotion remains open
     - current artifact bases: ``docs/_static/solver_objective_gradient_gate.png``, ``docs/_static/vmec_boozer_solver_frequency_gradient_gate.png``, ``docs/_static/vmec_boozer_quasilinear_gradient_gate.png``, ``docs/_static/vmec_boozer_nonlinear_window_gradient_gate.png``, ``docs/_static/vmec_boozer_li383_nonlinear_window_gradient_gate.png``, ``docs/_static/vmec_boozer_gradient_holdout_matrix.png``, ``docs/_static/vmec_boozer_multi_point_objective_gate.png``, ``docs/_static/vmec_boozer_aggregate_line_search_comparison.png``, ``docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.png``, ``docs/_static/vmec_boozer_aggregate_surface_holdout_gate.png``, ``docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.png``, ``docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json``, ``docs/_static/production_nonlinear_optimization_guard.png``, ``docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.png``, ``docs/_static/qa_projected_transport_step1e3_matched_comparison.png``, ``docs/_static/qa_projected_transport_step1e3_redesign_report.json``, ``docs/_static/strict_qa_top12_edge_prelaunch_gate.json``, ``docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.png``, ``docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.json``, ``docs/_static/strict_qa_top12_edge_redesign_report.json``, ``docs/_static/strict_qa_rerun_baseline_ensemble_gate.json``, ``docs/_static/strict_qa_top12_step1p25em3_candidate_ensemble_gate.json``, ``docs/_static/nonlinear_window_fd_audit.png``, and ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.png`` with CSV/JSON/PDF companions where available. The first differentiates actual electrostatic linear-RHS eigenpair observables with respect to solver-ready geometry arrays using the implicit left/right eigenpair system and checks ``gamma``, ``omega``, ``<k_perp^2>``, linear heat/particle-flux weights, and a mixing-length heat-flux proxy against nearest-branch central finite differences. The VMEC/Boozer frequency gate starts from a real ``vmec_jax`` state coefficient, maps through ``booz_xform_jax`` with ``mboz=nboz=21``, builds the SPECTRAX-GK linear RHS, and verifies the eigenfrequency gradient. The VMEC/Boozer quasilinear gate uses a richer ``Nl=2, Nm=3`` moment basis and checks ``gamma``, ``omega``, ``<k_perp^2>``, ``Q_i`` weight, and ``gamma Q_i/kperp^2`` against finite differences. The nonlinear-window estimator gates feed those observables into a smooth RK2 late-window envelope and check heat-flux mean, coefficient of variation, and normalized trend gradients. The multi-equilibrium matrix repeats the frequency, quasilinear, and estimator gates on the tracked QH and Li383 fixtures. The multi-alpha aggregate artifact shows reduced quasilinear objective sensitivity across two field lines and two ``k_y`` samples, while the growth-vs-quasilinear comparison shows that the two reduced objectives can select different VMEC coefficient directions. The alpha-heldout and surface-heldout splits pass reduced generalization checks, Li383 passes the second-equilibrium aggregate finite-difference plus line-search gate, and the production nonlinear optimization guard now includes D-shaped/circular replicated long-window holdouts plus the optimized-equilibrium ``t=[350,700]`` seed/timestep audit. The earlier aspect-6 projected transport-gradient candidate also has passed baseline/candidate seed/timestep ensembles, but the matched comparison gives a relative reduction of ``-0.00585`` and is not promoted. The stricter rerun-WOUT top-12 QA edge candidate improves the 18-point reduced metric by ``2.29%`` and passes both long-window ensemble gates, but its matched ``t=[350,700]`` nonlinear comparison gives only ``0.58%`` reduction with uncertainty z-score ``0.20`` and is also not promoted; the prelaunch gate now records that this reduced margin would be blocked against the calibrated ``4%`` threshold, so the next blocker is predictive transfer margin. These negative transfers require a better-conditioned multi-surface, multi-alpha transport objective before another expensive nonlinear audit. The compact nonlinear FD audits run actual nonlinear windows and verify startup conditioning/response only; their transport-average gates are false because they do not discard a long transient or demonstrate running-mean convergence. A surface-stencil path is available for memory-bounded diagnostics; nonlinear turbulence-gradient and broader multi-surface optimization claims remain separate gates.
   * - True ``t=1500`` strict QA matched nonlinear audits
     - ``tools/build_external_vmec_replicate_ensemble.py`` and ``tools/check_nonlinear_runtime_outputs.py``
     - Baseline, growth-objective, quasilinear-objective, and nonlinear-window-objective triplets admitted as robust long-window signals; all three transport-candidate optimization claims rejected by matched comparison
     - current artifact bases: ``docs/_static/vmec_qa_t1500_replicates/qa_baseline_scipy_t1500_ensemble_gate.png``, ``docs/_static/vmec_qa_t1500_replicates/growth_from_strict_baseline_t1500_ensemble_gate.png``, ``docs/_static/vmec_qa_t1500_replicates/quasilinear_from_strict_baseline_t1500_ensemble_gate.png``, ``docs/_static/vmec_qa_t1500_replicates/nonlinear_window_from_strict_baseline_t1500_ensemble_gate.png``, ``docs/_static/vmec_qa_t1500_baseline_to_growth_comparison.png``, ``docs/_static/vmec_qa_t1500_baseline_to_quasilinear_comparison.png``, and ``docs/_static/vmec_qa_t1500_baseline_to_nonlinear_window_comparison.png`` with JSON companions. All use true full-horizon ``t=1500`` office runs, two seeds plus one timestep variant, and the strict ``t=[1100,1500]`` averaging window. The strict QA baseline passes with ``<Q_i> = 11.580``, mean relative spread ``0.0381``, and combined SEM/mean ``0.0195``. The growth candidate passes as a trace with ``<Q_i> = 11.510`` but fails the matched reduction gate: ``0.60%`` relative reduction, ``z=0.26``, below the ``4%`` threshold. The quasilinear candidate passes as a trace with ``<Q_i> = 11.636`` but is slightly worse than baseline: ``-0.49%``, ``z=-0.19``. The nonlinear-window candidate also passes as a trace with ``<Q_i> = 11.609`` but is slightly worse than baseline: ``-0.25%``, ``z=-0.09``. These panels demonstrate saturated trace robustness and a negative optimization-transfer result for all strict-QA candidate rows, not a successful nonlinear turbulent-flux optimization.
   * - Inverse/UQ figure
     - ``examples/theory_and_demos/autodiff_inverse_growth.py``, ``examples/theory_and_demos/autodiff_inverse_twomode.py``, and ``tools/plot_stellarator_optimization_uq.py``
     - Scoped inverse and UQ validation closed; global identifiability and production optimization claims remain scoped
     - current artifact bases: ``docs/_static/autodiff_inverse_growth.png``, ``docs/_static/autodiff_inverse_twomode.png``, and ``docs/_static/stellarator_itg_optimization_uq.png`` with JSON/PDF/CSV companions. The inverse examples check AD/finite-difference consistency and conditioning for one- and two-parameter reduced objectives, while the UQ panel reports local Gauss-Newton covariance, rank, and sensitivity-map diagnostics. These figures validate the differentiable workflow and uncertainty reporting, not global uniqueness or production nonlinear stellarator optimization.
   * - Solved VMEC-JAX QA geometry figure
     - upstream ``vmec_jax`` ``QA_optimization.py`` workflow plus local panel
       stitch from solved-boundary and Boozer diagnostics
     - Solved-equilibrium geometry visual ready for README/docs baseline context
     - current artifact base:
       ``docs/_static/vmec_jax_qa_solved_boundary_boozer_panel.png``. The
       figure compares the initial and optimized solved VMEC LCFS surfaces
       colored by ``|B|`` and the corresponding Boozer-LCFS ``|B|`` contours.
       This is the manuscript-facing geometry visual for the QA baseline. It
       is not a nonlinear heat-flux optimization claim and should not be
       conflated with the reduced synthetic max-mode-1 optimization panels.
   * - Full max-mode-5 QA transport optimizer sweep
     - ``tools/build_vmec_jax_qa_full_sweep_panel.py`` from office
       ``vmec_jax_qa_full_sweep_20260605`` VMEC-JAX outputs
     - Strict QA baseline provenance closed; transport rows are scoped
       optimizer-output diagnostics with nonlinear ``Q(t)`` promotion failed
       under the current strict accepted-window audit
     - current artifact base:
       ``docs/_static/vmec_jax_qa_full_sweep_panel.png`` with JSON/CSV
       companions; a PDF companion is regenerated locally with ``--pdf`` when
       needed. The current campaign uses the upstream-style VMEC-JAX
       max-mode-5 simple seed and records an admitted constraints-only QA
       baseline with aspect ``4.999999``, mean iota ``0.410200``, and QS
       residual ``8.91e-6``. The growth, quasilinear, and nonlinear-window
       rows restart from that solved QA baseline and use one representative
       sample, ``s=0.64``, ``alpha=0``, ``k_y rho_i=0.30``, with
       ``mboz=nboz=21``. They produce real optimizer outputs, not synthetic
       surfaces, and their mean iota remains within the exploratory diagnostic
       acceptance band ``|iota| >= 0.39`` even though the stricter QA-admission
       gate records the small shortfall from ``0.41``. The follow-on matched
       office nonlinear audit completed raw baseline, growth, quasilinear, and
       nonlinear-window runs, but every trace ends near ``t=400`` while the
       strict admission window is ``t=[1100,1500]``. The harvested ensemble
       artifacts under
       ``docs/_static/optimized_equilibrium_replicates/vmec_qa_full_sweep_*``
       therefore have ``n_finite_means = 0``, and the matched comparison
       artifacts under ``docs/_static/qa_strict_baseline_to_*_strict_baseline.*``
       are failed negative admission evidence. No turbulent-flux reduction,
       quasilinear holdout, or nonlinear optimization success is claimed from
       this sweep. The rerun manifest generator now emits both staged
       restart-ladder commands and direct full-horizon ``t=1500`` commands,
       plus a runtime-output gate over ``t=[1100,1500]``; the figure remains
       blocked until true long-window outputs are available. The Boozer-LCFS
       ``|B|`` row is drawn with unfilled contours rather than a filled density
       map to expose quasisymmetry deterioration. The old projected-weight
       nonlinear audits remain historical single-point evidence tied to an
       earlier baseline and should not be conflated with this strict-baseline
       panel.
   * - Boundary-coefficient transport landscape
     - ``tools/build_vmec_boundary_transport_landscape.py`` and ``tools/patch_vmec_jax_wout_metadata.py``
     - Partial overlay complete; full 31-point nonlinear side remains open
     - current artifact base: ``docs/_static/vmec_boundary_transport_landscape_rbc11_full.png`` with JSON/CSV companions. The scan perturbs the strict QA ``RBC(1,1)`` coefficient over ``[-75%, +75%]`` with 31 points and evaluates growth plus all explicit electrostatic quasilinear rules on the optimizer sample set, ``s=(0.45,0.64,0.78)``, ``alpha=(0,pi/4)``, and ``k_y rho_i=(0.10,0.30,0.50)``, at ``ntheta=16``, ``mboz=nboz=21``, ``Nl=1``, and ``Nm=2``. The bottom panel overlays the 23 strict coefficients that have passed true long-window post-transient nonlinear heat-flux ensemble gates over ``t=[1100,1500]``: the negative side, the zero-offset baseline, and seven positive coefficients, ``+5%``, ``+10%``, ``+15%``, ``+25%``, ``+30%``, ``+35%``, and ``+40%``. Reduced/startup nonlinear-window diagnostics are deliberately excluded. The first ``-75%`` nonlinear point failed running-mean convergence over ``t=[350,700]`` and passed after continuation. The neighboring ``-70%`` point still failed the ``t=[700,1100]`` timestep-spread gate and passed after continuation to ``t=[1100,1500]``. The positive ``+5%`` point passes with mean ``Q_i≈10.8433``, mean-relative spread about ``1.70%``, and combined SEM/mean about ``1.91%``; ``+10%`` passes with mean ``Q_i≈9.6448``, spread about ``2.27%``, and SEM/mean about ``1.84%``; ``+15%`` passes with mean ``Q_i≈10.9084``, spread about ``4.20%``, and SEM/mean about ``1.57%``; ``+25%`` passes with mean ``Q_i≈10.0771``, spread about ``6.27%``, and SEM/mean about ``2.42%``; ``+30%`` passes with mean ``Q_i≈9.6482``, spread about ``0.67%``, and SEM/mean about ``2.09%``; ``+35%`` passes with mean ``Q_i≈8.5866``, spread about ``2.96%``, and SEM/mean about ``2.08%``; and ``+40%`` passes with mean ``Q_i≈7.1067``, spread about ``3.50%``, and SEM/mean about ``2.01%``. The positive ``+20%`` point failed the strict ``t=[1100,1500]`` ensemble only on mean-relative spread, ``15.48%`` versus the ``15%`` gate, so it is being continued to a later window instead of relaxing the threshold. Positive-side accepted points use seed/timestep replicated variants after the pathological ``seed31`` base runs were excluded from the positive-side campaign. The remaining higher positive-side coefficients remain pending, so this figure is a launch/noise diagnostic and optimizer-design input, not a promoted nonlinear turbulent-flux optimization result.
   * - Development-only optimization-plumbing figure
     - ``examples/theory_and_demos/reduced_stellarator_itg/compare_stellarator_itg_optimizations.py`` and ``tools/plot_stellarator_optimization_uq.py``
     - Initial differentiable objective-reduction and weighted-residual UQ gates closed for development diagnostics; full VMEC/Boozer/GK optimization open
     - primary artifact base: ``docs/_static/stellarator_itg_optimization_uq.png`` with JSON/PDF companions and individual diagnostic panels ``docs/_static/stellarator_itg_growth_optimization.png``, ``docs/_static/stellarator_itg_quasilinear_optimization.png``, and ``docs/_static/stellarator_itg_nonlinear_optimization.png``. The supporting reduced comparison sidecar ``docs/_static/stellarator_itg_optimization_comparison.json`` records objective histories and reduction ratios, but its companion PNG is a synthetic reduced max-mode-1 surface diagnostic and is not a solved-geometry optimization figure. These files live under ``examples/theory_and_demos/reduced_stellarator_itg`` rather than ``examples/optimization``. The UQ panel computes covariance from the final weighted objective residual Jacobian and shows derivative parity, control uncertainty, covariance projection, and rank/conditioning diagnostics. The production QA optimization examples are the VMEC-JAX-style scripts in ``examples/optimization``; they remain separate from this diagnostic artifact stack and still require solved-WOUT gates plus converged nonlinear audits before transport-optimization claims.
   * - VMEC/Boozer parity matrix
     - ``tools/build_vmec_boozer_parity_matrix.py``
     - Multi-equilibrium zero-beta equal-arc geometry gate closed at ``mboz=nboz=21``
     - current artifact base: ``docs/_static/vmec_boozer_parity_matrix.png`` with JSON/CSV/PDF companions. It checks QH, QI, and shaped-tokamak fixtures against the imported VMEC/EIK runtime convention and rejects ``mboz,nboz < 21``. The current limiting matrix row is QI drift at ``7.13e-2`` against the ``8e-2`` release tolerance; evaluated QI ``ntheta=8,16`` variants pass, while input-only QI seeds without bundled ``wout`` references are explicitly artifact-limited. This should be presented as a field-line geometry convention gate, not as a finite-beta transport-gradient validation.

Caption Policy
--------------

Every manuscript-facing figure should answer these questions directly in the
caption:

1. what case and model are shown,
2. what horizon or fit window is used,
3. what the reference is,
4. what agreement was expected,
5. what agreement was found.

Open Issues Before Drafting
---------------------------

- close the W7-X zonal-flow residual and late-envelope mismatch under the
  paper-facing line-first normalization; current time coverage is sufficient,
  but residuals fail at three wavelengths and late envelopes remain too large
- either close the long-time recurrence follow-up for the shaped-tokamak
  Rosenbluth-Hinton / GAM response benchmark or explicitly scope publication
  claims to the benchmark-scale pre-recurrence metrics now frozen in the
  Merlo Case-III artifact
- extend the W7-X fluctuation-spectrum diagnostic to a Doppler-reflectometry
  transfer-function comparison if experimental-facing claims enter the paper
- extend eigenfunction-overlap metrics beyond the closed KBM and W7-X raw
  overlays where additional literature-backed mode-shape references are useful
- tighten the current windowed nonlinear statistics panel with stricter case-specific gates where reference windows justify them
- tie ETG nonlinear claims to the benchmark literature or keep them framed as a pilot
- add or refine publication-ready zonal-flow closure figures before making W7-X
  recurrence claims
- add an experimental transfer-function fluctuation-spectrum panel only if
  experimental-facing W7-X claims enter the paper
