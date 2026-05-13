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
not yet for production nonlinear heat-flux optimization. Those stronger claims
require converged post-transient nonlinear heat-flux windows, VMEC/Boozer
nonlinear turbulence-gradient or robust finite-difference gates, local-gradient
conditioning, and nonlinear audits of optimized equilibria. W7-X zonal
recurrence and TEM/kinetic-electron stellarator validation remain deferred from
the current manuscript scope.

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
       nonlinear transport gradients and optimized-equilibrium audits remain
       future gates.
   * - Performance and parallelization
     - Runtime/memory figures, independent ``k_y`` scan scaling,
       quasilinear/UQ ensemble scaling, and nonlinear RHS profiler artifacts are
       release-facing engineering evidence.
     - Whole-state nonlinear sharding is identity/profiler evidence only, not a
       production nonlinear multi-GPU speedup claim.
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
   * - Nonlinear transport time-horizon audit
     - ``tools/build_nonlinear_transport_horizon_audit.py``
     - Closed as claim-scope guardrail; QH/CTH convergence promotion open
     - current artifact base: ``docs/_static/nonlinear_transport_time_horizon_audit.png`` with CSV/JSON/PDF companions. It audits the actual simulated time and claim scope for release nonlinear gates, startup finite-difference audits, reduced nonlinear-window estimators, and external-VMEC feasibility pilots. The new QH reduced-grid nonlinear pilot is extended from the earlier startup-scale ``t=20`` trace to ``t=150`` and reaches a meaningful late heat-flux window with mean about ``19.6``; it remains a feasibility result until a grid/window convergence gate passes. This panel prevents startup ``1e-11``-scale heat fluxes or reduced-envelope outputs from being described as post-transient nonlinear transport averages.
   * - Validation gate index
     - ``tools/make_validation_gate_index.py``
     - Closed for currently tracked gates
     - current artifact base: ``docs/_static/validation_gate_index.png`` and ``docs/_static/validation_gate_index.json``. This is not a physics result by itself; it is the audit panel for release-window gates, currently ``10/10`` passed.
   * - Open research lane status
     - ``tools/build_open_research_lane_status.py``
     - Closed as a claim-scope audit; underlying physics lanes remain scoped
     - current artifact base: ``docs/_static/open_research_lane_status.png`` with CSV/JSON/PDF companions. It reads the W7-X zonal recurrence, W7-X hypercollision probe, W7-X fluctuation/TEM extension status, quasilinear holdout, differentiable-geometry, and nonlinear-profiler artifacts and records which lanes are closed, partial, open, or blocked. The current status is intentionally conservative: nonlinear holdouts for the scoped quasilinear model-development claim and profiler-backed nonlinear hot-path localization are closed; W7-X fluctuation/TEM and differentiable geometry are partial bounded diagnostics; and W7-X long-window zonal recurrence/damping remains open. This panel is useful for the paper plan and release notes because it prevents partial diagnostics from being described as completed physics claims.
   * - Manuscript-readiness status panel
     - ``tools/build_manuscript_readiness_status.py``
     - Current manuscript scope with W7-X zonal and TEM deferred
     - current artifact base: ``docs/_static/manuscript_readiness_status.png`` with CSV/JSON/PDF companions. It records the narrower manuscript scope where W7-X zonal recurrence and TEM/kinetic-electron extensions are deferred. In that scope, quasilinear diagnostics and saturation-model selection are closed as a validated negative/model-selection result rather than as an absolute-flux predictor; VMEC/Boozer zero-beta equal-arc geometry parity is closed at ``mboz=nboz=21``; reduced differentiable stellarator ITG optimization is closed with AD/FD gates; and production solver-objective geometry gradients are closed for solver-ready arrays plus mode-21 VMEC/Boozer eigenfrequency, quasilinear heat-flux-weight, and reduced nonlinear-window estimator gates on QH and Li383. The compact nonlinear FD audits are retained only as startup plumbing checks with false transport-average gates. Converged post-transient nonlinear heat-flux averages, VMEC/Boozer nonlinear-window state-gradient, and optimized-equilibrium nonlinear audits remain required before production nonlinear heat-flux optimization claims.
   * - Quasilinear spectrum panel
     - ``tools/plot_quasilinear_spectrum.py``
     - Electrostatic diagnostic closed; absolute-flux prediction not promoted
     - current artifact bases: ``docs/_static/quasilinear_cyclone_spectrum.png``, ``docs/_static/quasilinear_cyclone_miller_spectrum.png``, ``docs/_static/quasilinear_hsx_spectrum.png``, and ``docs/_static/quasilinear_w7x_spectrum.png`` with CSV/JSON/PDF companions. They show electrostatic linear weights and explicitly uncalibrated mixing-length outputs from ``examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml``, ``examples/linear/axisymmetric/runtime_cyclone_miller_quasilinear.toml``, ``examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml``, and ``examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml``. Scan spectra use requested ``ky`` for the x-axis and retain signed selected-mode coordinates as ``mode_ky`` when applicable. W7-X was generated from an external VMEC benchmark file via ``W7X_VMEC_FILE``; the equilibrium itself is not shipped. Absolute saturated-flux claims remain open until a held-out nonlinear calibration report passes.
   * - Quasilinear calibration audit
     - ``tools/build_quasilinear_calibration_report.py`` and ``tools/plot_quasilinear_calibration.py``
     - Initial train/holdout artifact closed as a failed model-transfer gate
     - current artifact bases: ``docs/_static/quasilinear_cyclone_calibration_audit.png``, ``docs/_static/quasilinear_cyclone_miller_calibration_audit.png``, ``docs/_static/quasilinear_cyclone_miller_train_holdout.png``, ``docs/_static/quasilinear_hsx_train_holdout.png``, ``docs/_static/quasilinear_w7x_train_holdout.png``, the manuscript-facing combined panel ``docs/_static/quasilinear_stellarator_train_holdout.png``, and the input-provenance audit ``docs/_static/quasilinear_validated_calibration_inputs.png`` with JSON/PDF companions. The current one-constant train/holdout report fits the heat-flux scale on Cyclone and the external-VMEC ITERModel case, then scores five held-out windows: Cyclone Miller, HSX, W7-X, D-shaped external VMEC, and an up-down asymmetric external-VMEC tokamak. It intentionally remains ``calibration_dataset`` with ``passed = false`` because held-out errors exceed the ``0.35`` gate; the current holdout mean relative error is about ``2.57``. HSX, W7-X, and the up-down asymmetric VMEC point are useful negative stellarator/tokamak transfer checks because the simple positive-growth mixing-length family predicts little or zero flux while the nonlinear windows are finite. The D-shaped external-VMEC point is a converged negative transfer constraint with finite late-window nonlinear heat flux but a grossly overpredicted Cyclone/ITERModel-scaled mixing-length estimate. The input audit confirms that every current train/holdout nonlinear artifact maps to a passed nonlinear gate, while failed QH and CTH-like external-VMEC feasibility pilots remain excluded. This closes the NetCDF/CSV calibration machinery and provenance gate but not a calibrated absolute-flux claim.
   * - Quasilinear saturation-rule sweep
     - ``tools/plot_quasilinear_saturation_rule_sweep.py``
     - Initial model-development diagnostic closed as a negative result
     - current artifact base: ``docs/_static/quasilinear_saturation_rule_sweep.png`` with JSON/PDF companions. It fits one scalar on the two training cases, Cyclone and external-VMEC ITERModel, then scores the same five held-out windows for three simple rules: positive-growth mixing length, raw linear heat-flux weight, and an absolute-growth diagnostic. All three fail the held-out absolute-flux gate. The least-bad simple rule is now positive-growth mixing length with holdout mean relative error about ``2.51``; raw linear weight is about ``3.19`` and the absolute-growth diagnostic is about ``3.96``. The panel also includes a training-mean null baseline with holdout mean relative error about ``1.39``. Its JSON ``promotion_gate`` has no accepted rules, so future calibrated rules must beat both the quasilinear baselines and this null baseline before being promoted. This supports the next saturation-model lane while preventing premature absolute quasilinear transport claims.
   * - Shape-aware quasilinear saturation diagnostic
     - ``tools/plot_quasilinear_shape_aware_saturation.py``
     - Initial leave-one-geometry-out diagnostic closed as a negative result
     - current artifact base: ``docs/_static/quasilinear_shape_aware_saturation.png`` with JSON/PDF companions. It fits a shared nonlinear/quasilinear spectrum-shape exponent with per-case intercepts, uses only passed shape gates for the exponent fit, then fits the absolute heat-flux scale on training cases and scores each held-out geometry. The shape-aware model gives mean absolute relative error about ``0.664`` versus ``0.624`` for the linear-weight baseline and ``0.170`` for a deliberately simple training-mean null baseline. The JSON ``promotion_gate`` is false because the model fails the ``0.35`` transport gate and does not beat the null baseline on the current four-case dataset. This is retained as a manuscript-facing negative result because it rules out a too-simple one-exponent envelope before stellarator optimization claims.
   * - Quasilinear candidate uncertainty gate
     - ``tools/plot_quasilinear_candidate_uncertainty.py``
     - Seven-case uncertainty-aware candidate gate closed as a scoped model-selection result
     - current artifact base: ``docs/_static/quasilinear_candidate_uncertainty.png`` with JSON/PDF companions. It adds training-residual ``95%`` prediction intervals to leave-one-geometry-out candidate scoring on the current seven-case electrostatic-compatible dataset. The legacy calibrated linear-weight and one-exponent shape-power-law candidates remain rejected relative to the null/skill gates. The accepted research candidate is ``spectral_envelope_ridge``: it uses the positive-growth ``k_y`` centroid and heat-flux-weighted ``k_y`` width in a three-parameter log-linear ridge model, reaches leave-one-geometry-out mean relative error about ``0.244``, and has interval coverage about ``0.857``. This is a bounded model-development result, not a runtime/TOML absolute-flux predictor or a universal saturation law.
   * - Quasilinear dataset-sufficiency gate
     - ``tools/plot_quasilinear_dataset_sufficiency.py``
     - Promotion guard closed for the scoped spectral-envelope candidate; higher-parameter and electromagnetic claims remain blocked
     - current artifact base: ``docs/_static/quasilinear_dataset_sufficiency.png`` with JSON/PDF companions. It audits the validated nonlinear windows before any richer saturation model is promoted. The current electrostatic-compatible set has seven cases, two explicit training geometries, and five held-out geometries. That is sufficient for the one-parameter linear-weight candidate, the two-parameter shape-power-law candidate, and the three-parameter ``spectral_envelope_ridge`` candidate, but not for the five-parameter ``linear_state_ridge`` model. KBM is explicitly excluded from this electrostatic quasilinear promotion lane because electromagnetic field-channel normalization and calibration remain future work. The gate therefore supports the scoped spectral-envelope model-selection claim while preventing broader absolute-flux or electromagnetic quasilinear claims.
   * - Quasilinear promotion guardrail audit
     - ``tools/check_quasilinear_promotion_guardrails.py``
     - Fast metadata gate closed; nonlinear simulation validation remains delegated to the source gates
     - current artifact: ``docs/_static/quasilinear_promotion_guardrails.json``. It scans the train/holdout calibration reports, saturation-model reports, nonlinear input-validation blocks, promotion gates, claim-scope docs, and the quasilinear row in ``docs/_static/manuscript_readiness_status.json``. It requires finite nonlinear window means and standard deviations for train/holdout calibration points, explicit nonlinear and quasilinear artifact provenance, passed held-out gates before any ``calibrated_absolute_flux`` claim, non-promotion wording in the docs, and a manuscript-readiness quasilinear lane that remains scoped as diagnostic/model-selection evidence rather than a runtime absolute-flux predictor. This is deliberately a wording and metadata guard, not a calibrated absolute-flux claim or a replacement for nonlinear convergence simulations.
   * - Release claim-scope ledger
     - ``docs/release_scope.rst``
     - Closed as documentation guardrail
     - This documentation page centralizes the current claim boundaries across validation, quasilinear model selection, differentiable geometry, parallelization, performance, and deferred W7-X/TEM lanes. It should be updated whenever a new artifact promotes or demotes a claim.
   * - VMEC equilibrium portfolio for future quasilinear holdouts
     - ``tools/plot_vmec_jax_equilibrium_inventory.py``
     - Planning artifact closed; bounded linear smoke checks started; transport validation open
     - current artifact bases: ``docs/_static/vmec_jax_equilibrium_inventory.png``, ``docs/_static/external_vmec_candidate_linear_screen.csv``, ``docs/_static/quasilinear_vmec_dshape_linear_spectrum.png``, ``docs/_static/external_vmec_dshape_grid_convergence_gate.png``, ``docs/_static/external_vmec_dshape_t250_high_grid_convergence_gate.png``, ``docs/_static/quasilinear_vmec_jax_qh_linear_spectrum.png``, ``docs/_static/quasilinear_vmec_jax_cth_like_linear_spectrum.png``, ``docs/_static/external_vmec_qh_nonlinear_t150_pilot.png``, ``docs/_static/external_vmec_qh_nonlinear_t150_n48_pilot.png``, ``docs/_static/external_vmec_qh_nonlinear_t150_n64_pilot.png``, ``docs/_static/external_vmec_qh_grid_convergence_gate.png``, ``docs/_static/external_vmec_qh_high_grid_convergence_gate.png``, ``docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.png``, ``docs/_static/external_vmec_cth_like_nonlinear_t150_n48_pilot.png``, and ``docs/_static/external_vmec_cth_like_grid_convergence_gate.png`` with JSON/PDF companions. The inventory scans external VMEC files from ``vmec_jax/examples/data`` without checking them into SPECTRAX-GK. A broader five-point candidate screen selected DSHAPE as the strongest finite unstable branch with ``gamma≈0.096`` at ``ky≈0.476``. DSHAPE passes low-to-mid-grid convergence at ``t=150`` and passes the ``48x48x32`` to ``64x64x40`` high-grid gate after extension to ``t=250``: common-window and least-window symmetric relative differences are about ``0.139`` and ``0.108``, below the ``0.15`` threshold. QH and CTH-like remain useful feasibility and negative convergence results rather than transport validation claims: QH fails both ``32->48`` and ``48->64`` gates, and CTH-like fails its first grid check. DSHAPE is ready for calibration-report admission; QH and CTH-like should stay excluded until production-resolution convergence gates pass.
   * - Quasilinear spectrum-shape gate
     - ``tools/plot_quasilinear_spectrum_shape_gate.py``
     - HSX, W7-X, and Cyclone Miller gates closed; Cyclone retained as a failed model gate
     - current artifact bases: ``docs/_static/quasilinear_hsx_spectrum_shape_gate.png``, ``docs/_static/quasilinear_w7x_spectrum_shape_gate.png``, ``docs/_static/quasilinear_cyclone_miller_spectrum_shape_gate.png``, and ``docs/_static/quasilinear_cyclone_spectrum_shape_gate.png`` with JSON/PDF companions. They compare normalized linear heat-flux-weight spectra against normalized nonlinear resolved ``HeatFlux_kyst`` spectra. HSX passes with ``TV≈0.112`` and cosine ``≈0.971``; W7-X passes with ``TV≈0.056`` and cosine ``≈0.992``; Cyclone Miller passes with ``TV≈0.094`` and cosine ``≈0.983``; Cyclone is kept as a failed gate with ``TV≈0.215`` and cosine ``≈0.896``. This supports spectrum-shape diagnostics while identifying a real saturation/window/model limitation before absolute saturated-flux claims. KBM is deferred from this gate because the current quasilinear diagnostic validates electrostatic channels only and the KBM lane is electromagnetic.
   * - Zonal-flow / GAM response panel
     - ``tools/plot_zonal_flow_response.py``, ``tools/plot_zonal_flow_response_from_output.py``, ``tools/generate_miller_zonal_response_pilot.py``, ``tools/generate_w7x_zonal_response_panel.py``, ``tools/digitize_w7x_zonal_reference.py``, ``tools/compare_w7x_zonal_reference.py``, ``tools/plot_w7x_zonal_contract_audit.py``, ``tools/plot_w7x_zonal_moment_tail_audit.py``, ``tools/plot_w7x_zonal_closure_ladder.py``, ``tools/plot_w7x_zonal_state_convention_audit.py``, and ``tools/plot_w7x_zonal_recurrence_sweep.py``
     - Open
     - should combine shaped-tokamak Rosenbluth-Hinton-style residuals with W7-X residual/damping envelopes using one figure convention; use signed ``Phi_zonal_mode_kxt`` or case-specific signed line averages for publication claims and keep ``Phi2_zonal_t`` only as an intermediate cross-check. The current Merlo Case-III artifact is ``docs/_static/miller_zonal_response_pilot.png`` from the initial-density setup at ``Nz=32``, ``Nl=4``, ``Nm=24``, ``dt=0.005``, and ``t≈60``. With Rosenbluth-Hinton first-sample normalization it gives ``residual≈0.192`` against the Merlo et al. Figs. 12/16 read-off of about ``0.19``; a literature-faithful common fit window ``t≈30`` with separate positive/negative-extrema damping fits gives ``γ_GAM R0 / v_i≈-0.176`` against the paper-scale read-off near ``-0.17``; and Hilbert-phase frequency extraction on that same window gives ``ω_GAM R0 / v_i≈2.20`` against the paper-scale read-off near ``2.24``. A higher-moment audit lowers the recurrence ratio but over-damps the GAM, while weak hypercollision scans are effectively inert, so the frozen Merlo artifact remains on the current ``Nm=24`` baseline. The W7-X side now uses the potential initializer, signed line-average observable, paper-facing line-first normalization, and no hidden time-axis scaling. The tracked long-window W7-X artifact is ``docs/_static/w7x_zonal_response_panel.png`` with replayable traces in ``docs/_static/w7x_zonal_response_panel.traces.csv``; it reaches the digitized Fig. 11 windows, but ``docs/_static/w7x_zonal_reference_compare.json`` remains open because residuals fail at ``k_x rho_i=0.07``, ``0.10``, and ``0.30`` and the late envelopes are much larger than the digitized stella/GENE traces. ``docs/_static/w7x_zonal_contract_audit.png`` is now the paper-facing diagnostic panel for that open mismatch and is intentionally excluded from the release gate index. ``docs/_static/w7x_zonal_state_convention_audit.png`` closes the paper-facing state convention layer: the recovered Gaussian potential has relative ``L2`` error ``1.85e-6``, off-target spectral content is zero to reported precision, and the diagnostic helpers agree with manual line/volume reductions near ``2e-16``. ``docs/_static/w7x_zonal_moment_tail_audit.png``, ``docs/_static/w7x_zonal_closure_ladder_kx070.png``, ``docs/_static/w7x_zonal_recurrence_sweep_kx070.png``, and ``docs/_static/w7x_zonal_hypercollision_probe_kx070.png`` are companion open diagnostics; together they support a recurrence / moment-closure hypothesis and show that weak or constant closure can reduce Hermite-tail metrics without closing the paper trace. The bounded recurrence sweep separates moment-resolution and closure-source effects over ``t v_t/a <= 100``: ``Nl=12,Nm=48`` gives the lowest no-closure mean absolute trace error, while constant-source closure lowers final Hermite-tail energy from ``0.388`` to ``0.062`` but worsens the trace error. The newer constant-Hermite-hypercollision probe lowers the Hermite-tail fraction to about ``0.099`` but still leaves mean trace error near ``0.289`` and late-envelope variance about ``4.3`` times the digitized reference. The W7-X generator now exposes explicit hypercollision and Gaussian-width audit overrides so future closure probes are reproducible from the tracked tool. A newer high-moment four-wavelength audit under ``tools_out/zonal_response/w7x_publication_nl16_nm64_dt005_t100`` verifies finite signed traces to ``t≈100`` after restart-continuation fixes. The tracked W7-X TOML keeps ``gaussian_width=1`` because the benchmark source writes the initializer as ``exp[-(z-z0)^2]``; wider profiles and non-unit time scales are retained only as audits. The lane remains open pending the W7-X damping/closure and velocity-space recurrence fix under the paper-facing convention.
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
     - ``tools/build_solver_objective_gradient_gate.py``, ``tools/build_vmec_boozer_solver_frequency_gradient_gate.py``, ``tools/build_vmec_boozer_quasilinear_gradient_gate.py``, ``tools/build_vmec_boozer_nonlinear_window_gradient_gate.py``, ``tools/build_vmec_boozer_gradient_holdout_matrix.py``, ``tools/build_vmec_boozer_multi_point_objective_gate.py``, ``tools/build_vmec_boozer_aggregate_line_search_comparison.py``, ``tools/check_vmec_boozer_aggregate_holdout_gate.py``, ``tools/build_nonlinear_window_fd_audit.py``, and ``tools/build_vmec_boozer_nonlinear_window_fd_audit.py``
     - Solver-ready linear-RHS gradient gate closed; mode-21 VMEC/Boozer state-to-solver eigenfrequency, quasilinear heat-flux-weight, and reduced nonlinear-window estimator gates closed for QH plus Li383; compact and VMEC/Boozer-perturbed nonlinear startup FD audits closed only as plumbing checks; multi-alpha reduced aggregate objective plumbing passes; production nonlinear-window transport-average and turbulence-gradient promotion open
     - current artifact bases: ``docs/_static/solver_objective_gradient_gate.png``, ``docs/_static/vmec_boozer_solver_frequency_gradient_gate.png``, ``docs/_static/vmec_boozer_quasilinear_gradient_gate.png``, ``docs/_static/vmec_boozer_nonlinear_window_gradient_gate.png``, ``docs/_static/vmec_boozer_li383_nonlinear_window_gradient_gate.png``, ``docs/_static/vmec_boozer_gradient_holdout_matrix.png``, ``docs/_static/vmec_boozer_multi_point_objective_gate.png``, ``docs/_static/vmec_boozer_aggregate_line_search_comparison.png``, ``docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json``, ``docs/_static/nonlinear_window_fd_audit.png``, and ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.png`` with CSV/JSON/PDF companions. The first differentiates actual electrostatic linear-RHS eigenpair observables with respect to solver-ready geometry arrays using the implicit left/right eigenpair system and checks ``gamma``, ``omega``, ``<k_perp^2>``, linear heat/particle-flux weights, and a mixing-length heat-flux proxy against nearest-branch central finite differences. The VMEC/Boozer frequency gate starts from a real ``vmec_jax`` state coefficient, maps through ``booz_xform_jax`` with ``mboz=nboz=21``, builds the SPECTRAX-GK linear RHS, and verifies the eigenfrequency gradient. The VMEC/Boozer quasilinear gate uses a richer ``Nl=2, Nm=3`` moment basis and checks ``gamma``, ``omega``, ``<k_perp^2>``, ``Q_i`` weight, and ``gamma Q_i/kperp^2`` against finite differences. The nonlinear-window estimator gates feed those observables into a smooth RK2 late-window envelope and check heat-flux mean, coefficient of variation, and normalized trend gradients. The multi-equilibrium matrix repeats the frequency, quasilinear, and estimator gates on the tracked QH and Li383 fixtures. The new multi-alpha aggregate artifact shows reduced quasilinear objective sensitivity across two field lines and two ``k_y`` samples, while the growth-vs-quasilinear comparison shows that the two reduced objectives can select different VMEC coefficient directions. The holdout promotion JSON is currently blocked as intended until a separate passed held-out surface or field-line validation artifact exists. The compact nonlinear FD audits run actual nonlinear windows and verify startup conditioning/response only; their transport-average gates are false because they do not discard a long transient or demonstrate running-mean convergence. A surface-stencil path is available for memory-bounded diagnostics; converged nonlinear-window turbulence gradients remain a separate promotion gate.
   * - Inverse/UQ figure
     - ``examples/theory_and_demos/autodiff_inverse_growth.py``, ``examples/theory_and_demos/autodiff_inverse_twomode.py``, and ``tools/plot_stellarator_optimization_uq.py``
     - Scoped inverse and UQ validation closed; global identifiability and production optimization claims remain scoped
     - current artifact bases: ``docs/_static/autodiff_inverse_growth.png``, ``docs/_static/autodiff_inverse_twomode.png``, and ``docs/_static/stellarator_itg_optimization_uq.png`` with JSON/PDF/CSV companions. The inverse examples check AD/finite-difference consistency and conditioning for one- and two-parameter reduced objectives, while the UQ panel reports local Gauss-Newton covariance, rank, and sensitivity-map diagnostics. These figures validate the differentiable workflow and uncertainty reporting, not global uniqueness or production nonlinear stellarator optimization.
   * - Optimization figure
     - ``examples/optimization/compare_stellarator_itg_optimizations.py`` and ``tools/plot_stellarator_optimization_uq.py``
     - Initial differentiable objective-reduction and weighted-residual UQ gates closed; full VMEC/Boozer/GK optimization open
     - current artifact bases: ``docs/_static/stellarator_itg_optimization_comparison.png`` and ``docs/_static/stellarator_itg_optimization_uq.png`` with JSON/PDF companions and individual objective panels ``docs/_static/stellarator_itg_growth_optimization.png``, ``docs/_static/stellarator_itg_quasilinear_optimization.png``, and ``docs/_static/stellarator_itg_nonlinear_optimization.png``. The examples optimize a QA max-mode-1 control vector with ``A = 7`` and ``iota = 0.41`` constraints for growth-rate, quasilinear-flux, and nonlinear-window objectives. All three pass AD-vs-finite-difference gates and reduce the tracked ITG observables from the shared initial point. The UQ panel now computes covariance from the final weighted objective residual Jacobian, not from initial-to-final displacement, and shows derivative parity, control uncertainty, covariance projection, and rank/conditioning diagnostics. The geometry bridge now includes real ``vmec_jax`` metric-tensor derivatives, real non-axisymmetric VMEC field-line tensor derivatives through ``vmec_jax.geom`` plus ``vmec_jax.vmec_bcovar``, a direct VMEC tensor-derived flux-tube derivative gate, a direct-VMEC-tensor vs imported-VMEC/EIK array-parity audit, a Boozer equal-arc core/metric parity audit, and a real ``vmec_jax`` ``VMECState`` to ``booz_xform_jax`` to SPECTRAX-GK derivative gate. This is a differentiable optimization and UQ gate, not a full production stellarator-transport optimization claim; the latter remains open pending finite-beta and broader production-runtime drift parity, production nonlinear turbulence-gradient or robust finite-difference audits, and converged nonlinear audits of optimized equilibria.
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
