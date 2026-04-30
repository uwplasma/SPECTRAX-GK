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
     - Open
     - Cyclone ITG, ETG, KBM, W7-X, HSX, plus shaped tokamak lane if closed. Current artifact base: ``docs/_static/benchmark_core_linear_atlas.png``.
   * - Eigenfunction validation panel
     - ``tools/plot_eigenfunction_overlap_summary.py``, ``tools/plot_eigenfunction_reference_overlay.py``, ``tools/generate_kbm_reference_overlay.py``, and ``tools/generate_w7x_reference_overlay.py``
     - Closed for KBM and W7-X raw overlays
     - first shipped overlap artifact is ``docs/_static/kbm_eigenfunction_overlap_summary.png``; frozen raw GX bundles now exist for KBM and W7-X under ``docs/_static/reference_modes/``. The closed KBM raw overlay is ``docs/_static/kbm_eigenfunction_reference_overlay_ky0p3000.png`` with overlap ``0.999985`` and relative ``L^2`` mismatch ``0.00721``. The closed W7-X raw overlay is ``docs/_static/w7x_eigenfunction_reference_overlay_ky0p3000.png`` with overlap ``0.9999999994`` and relative ``L^2`` mismatch ``3.33e-5`` against the finite GX ``t≈2`` raw-mode bundle. Both overlay generators write JSON gate reports with ``overlap >= 0.95`` and ``relative L^2 <= 0.25`` requirements. ``tools/compare_gx_kbm.py --branch-summary-json`` writes branch-continuity gate metadata for selected KBM scans, and ``tools/generate_kbm_branch_gate_summary.py`` refreshes the no-rerun tracked artifact ``docs/_static/kbm_branch_gate_summary.json`` from ``docs/_static/kbm_gx_candidates.csv``. The current continuity-first branch summary passes the strict adjacent growth/frequency jump and successive-overlap gates.
   * - Nonlinear transport panel
     - ``tools/make_gx_summary_panel.py`` / ``tools/make_gx_publication_panel.py``
     - Open
     - Cyclone, Miller, KBM, W7-X, HSX with matched windows. Current component artifacts: ``docs/_static/nonlinear_cyclone_diag_compare_t400.png``, ``docs/_static/nonlinear_cyclone_miller_diag_compare_t122.png``, ``docs/_static/nonlinear_kbm_diag_compare_t400_stats.png``, ``docs/_static/nonlinear_w7x_diag_compare_t200.png``, ``docs/_static/hsx_nonlinear_compare_t50_true.png``. ``tools/compare_gx_nonlinear_diagnostics.py --summary-json`` now writes machine-readable mean-relative-mismatch gates for each plotted diagnostic with explicit transport-window bounds. The materialized release-window metadata are ``docs/_static/nonlinear_cyclone_gate_summary.json``, ``docs/_static/nonlinear_cyclone_miller_gate_summary.json``, ``docs/_static/nonlinear_kbm_gate_summary.json``, ``docs/_static/nonlinear_hsx_gate_summary.json``, and ``docs/_static/nonlinear_w7x_gate_summary.json``; all pass the current ``0.10`` mean-relative release gate. The older short Cyclone ``t=5`` diagnostic remains documented as an exploratory startup/resolved-spectrum audit, not a release gate.
   * - W7-X exact-state convention audit
     - ``tools/run_exact_state_audit.py`` and ``tools/plot_w7x_exact_state_audit.py``
     - Closed
     - current artifact base: ``docs/_static/w7x_exact_state_audit.png`` with CSV/JSON/PDF companions. It compares W7-X nonlinear VMEC startup state, late geometry/field arrays, and re-evaluated scalar diagnostics directly against GX exact-state dumps. The maximum finite pointwise relative error is ``4.62e-5`` under the explicit ``1e-4`` convention gate, while scalar diagnostics are below ``1.8e-7``. This closes the geometry/diagnostic convention layer but does not close the separate W7-X zonal-response recurrence lane.
   * - Windowed-statistics summary
     - ``tools/plot_nonlinear_window_statistics.py``
     - Closed for current release-window gates
     - current artifact base: ``docs/_static/nonlinear_window_statistics.png`` with CSV/JSON/PDF companions. It summarizes the per-diagnostic ``mean_rel_abs`` and ``max_rel_abs`` statistics from the frozen nonlinear GX comparison gate JSONs for Cyclone, Cyclone Miller, KBM, W7-X, and HSX. Exploratory/short-run diagnostics are explicitly excluded with ``gate_index_include=false``.
   * - Validation gate index
     - ``tools/make_validation_gate_index.py``
     - Closed for currently tracked gates
     - current artifact base: ``docs/_static/validation_gate_index.png`` and ``docs/_static/validation_gate_index.json``. This is not a physics result by itself; it is the audit panel for release-window gates, currently ``10/10`` passed.
   * - Quasilinear spectrum panel
     - ``tools/plot_quasilinear_spectrum.py``
     - Initial diagnostic closed; nonlinear calibration open
     - current artifact bases: ``docs/_static/quasilinear_cyclone_spectrum.png``, ``docs/_static/quasilinear_cyclone_miller_spectrum.png``, ``docs/_static/quasilinear_hsx_spectrum.png``, and ``docs/_static/quasilinear_w7x_spectrum.png`` with CSV/JSON/PDF companions. They show electrostatic linear weights and explicitly uncalibrated mixing-length outputs from ``examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml``, ``examples/linear/axisymmetric/runtime_cyclone_miller_quasilinear.toml``, ``examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml``, and ``examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml``. Scan spectra use requested ``ky`` for the x-axis and retain signed selected-mode coordinates as ``mode_ky`` when applicable. W7-X was generated from an external VMEC benchmark file via ``W7X_VMEC_FILE``; the equilibrium itself is not shipped. Absolute saturated-flux claims remain open until a held-out nonlinear calibration report passes.
   * - Quasilinear calibration audit
     - ``tools/build_quasilinear_calibration_report.py`` and ``tools/plot_quasilinear_calibration.py``
     - Initial train/holdout artifact closed as a failed model-transfer gate
     - current artifact bases: ``docs/_static/quasilinear_cyclone_calibration_audit.png``, ``docs/_static/quasilinear_cyclone_miller_calibration_audit.png``, ``docs/_static/quasilinear_cyclone_miller_train_holdout.png``, ``docs/_static/quasilinear_hsx_train_holdout.png``, ``docs/_static/quasilinear_w7x_train_holdout.png``, the manuscript-facing combined panel ``docs/_static/quasilinear_stellarator_train_holdout.png``, and the input-provenance audit ``docs/_static/quasilinear_validated_calibration_inputs.png`` with JSON/PDF companions. The one-constant train/holdout report fits the heat-flux scale on Cyclone and scores Cyclone Miller plus HSX/W7-X as held-out geometries; it intentionally remains ``calibration_dataset`` with ``passed = false`` because held-out errors exceed the ``0.35`` gate. HSX and W7-X are useful negative stellarator holdouts: the current short linear spectra are stable under ``gamma_floor = 0`` and therefore predict zero mixing-length flux while the nonlinear windows are finite. The input audit confirms that every current train/holdout nonlinear artifact maps to a passed nonlinear gate, while failed external-VMEC feasibility pilots remain excluded. This closes the NetCDF calibration machinery and provenance gate but not a calibrated absolute-flux claim.
   * - Quasilinear saturation-rule sweep
     - ``tools/plot_quasilinear_saturation_rule_sweep.py``
     - Initial model-development diagnostic closed as a negative result
     - current artifact base: ``docs/_static/quasilinear_saturation_rule_sweep.png`` with JSON/PDF companions. It fits one scalar on Cyclone and scores Cyclone Miller, HSX, and W7-X for three simple rules: positive-growth mixing length, raw linear heat-flux weight, and an absolute-growth diagnostic. All three fail the held-out absolute-flux gate; the raw linear-weight rule is least bad with holdout mean relative error about ``25``. The panel also includes a Cyclone-training-mean null baseline with holdout mean relative error about ``0.372``. Its JSON ``promotion_gate`` has no accepted rules, so future calibrated rules must beat both the quasilinear baselines and this null baseline before being promoted. This supports the next saturation-model lane while preventing premature absolute quasilinear transport claims.
   * - Shape-aware quasilinear saturation diagnostic
     - ``tools/plot_quasilinear_shape_aware_saturation.py``
     - Initial leave-one-geometry-out diagnostic closed as a negative result
     - current artifact base: ``docs/_static/quasilinear_shape_aware_saturation.png`` with JSON/PDF companions. It fits a shared nonlinear/quasilinear spectrum-shape exponent with per-case intercepts, uses only passed shape gates for the exponent fit, then fits the absolute heat-flux scale on training cases and scores each held-out geometry. The shape-aware model gives mean absolute relative error about ``0.664`` versus ``0.624`` for the linear-weight baseline and ``0.170`` for a deliberately simple training-mean null baseline. The JSON ``promotion_gate`` is false because the model fails the ``0.35`` transport gate and does not beat the null baseline on the current four-case dataset. This is retained as a manuscript-facing negative result because it rules out a too-simple one-exponent envelope before stellarator optimization claims.
   * - Quasilinear candidate uncertainty gate
     - ``tools/plot_quasilinear_candidate_uncertainty.py``
     - Initial uncertainty-aware candidate gate closed as a negative result
     - current artifact base: ``docs/_static/quasilinear_candidate_uncertainty.png`` with JSON/PDF companions. It adds training-residual ``95%`` prediction intervals to leave-one-geometry-out candidate scoring. The calibrated linear-weight candidate has mean relative error about ``0.624`` and interval coverage ``0.75``; the shape-power-law candidate has mean relative error about ``0.664`` and interval coverage ``0.75``. The linear-state ridge candidate uses only linear-spectrum branch/state features and reaches mean relative error about ``0.173`` with full interval coverage, but is explicitly marked ineligible because the current four-case dataset leaves too few training cases per fitted parameter. The training-mean null remains about ``0.170``. The JSON ``promotion_gate`` is false with no accepted candidates, so no candidate is promoted into runtime/TOML saturation options.
   * - VMEC equilibrium portfolio for future quasilinear holdouts
     - ``tools/plot_vmec_jax_equilibrium_inventory.py``
     - Planning artifact closed; bounded linear smoke checks started; transport validation open
     - current artifact bases: ``docs/_static/vmec_jax_equilibrium_inventory.png``, ``docs/_static/quasilinear_vmec_jax_qh_linear_spectrum.png``, ``docs/_static/quasilinear_vmec_jax_cth_like_linear_spectrum.png``, ``docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.png``, ``docs/_static/external_vmec_cth_like_nonlinear_t150_n48_pilot.png``, and ``docs/_static/external_vmec_cth_like_grid_convergence_gate.png`` with JSON/PDF companions. The inventory scans external VMEC files from ``vmec_jax/examples/data`` without checking them into SPECTRAX-GK. The recommended next candidates include Li383, nfp4 QH, CTH-like, shaped tokamak, circular tokamak, DSHAPE, and purely toroidal equilibria. The low-resolution QA fixture is deferred because its VMEC reference-scale metadata are degenerate. Short ``ky=0.10,0.20`` smoke scans are finite for Li383, nfp4 QH, CTH-like, and shaped tokamak. Six-point QH and CTH-like linear scans show stable low-``ky`` branches and positive growth at high ``ky``; Li383 and shaped tokamak remain stable on this grid. Reduced-grid QH and CTH-like nonlinear pilots are finite; CTH-like reaches a long finite state to ``t=150``. The first ``48x48x32`` CTH-like grid check remains finite but fails the explicit convergence gate: the common-window heat-flux symmetric relative difference is about ``0.571`` and the independently selected least-trending windows still differ by about ``0.453``, both above the ``0.15`` threshold. This is a useful negative convergence result rather than a transport validation claim. Each candidate still needs a passed production-resolution convergence check and acceptance gate before entering calibration reports.
   * - Quasilinear spectrum-shape gate
     - ``tools/plot_quasilinear_spectrum_shape_gate.py``
     - HSX, W7-X, and Cyclone Miller gates closed; Cyclone retained as a failed model gate
     - current artifact bases: ``docs/_static/quasilinear_hsx_spectrum_shape_gate.png``, ``docs/_static/quasilinear_w7x_spectrum_shape_gate.png``, ``docs/_static/quasilinear_cyclone_miller_spectrum_shape_gate.png``, and ``docs/_static/quasilinear_cyclone_spectrum_shape_gate.png`` with JSON/PDF companions. They compare normalized linear heat-flux-weight spectra against normalized nonlinear resolved ``HeatFlux_kyst`` spectra. HSX passes with ``TV≈0.112`` and cosine ``≈0.971``; W7-X passes with ``TV≈0.056`` and cosine ``≈0.992``; Cyclone Miller passes with ``TV≈0.094`` and cosine ``≈0.983``; Cyclone is kept as a failed gate with ``TV≈0.215`` and cosine ``≈0.896``. This supports spectrum-shape diagnostics while identifying a real saturation/window/model limitation before absolute saturated-flux claims. KBM is deferred from this gate because the current quasilinear diagnostic validates electrostatic channels only and the KBM lane is electromagnetic.
   * - Zonal-flow / GAM response panel
     - ``tools/plot_zonal_flow_response.py``, ``tools/plot_zonal_flow_response_from_output.py``, ``tools/generate_miller_zonal_response_pilot.py``, ``tools/generate_w7x_zonal_response_panel.py``, ``tools/digitize_w7x_zonal_reference.py``, ``tools/compare_w7x_zonal_reference.py``, ``tools/plot_w7x_zonal_contract_audit.py``, ``tools/plot_w7x_zonal_moment_tail_audit.py``, ``tools/plot_w7x_zonal_closure_ladder.py``, ``tools/plot_w7x_zonal_state_convention_audit.py``, and ``tools/plot_w7x_zonal_recurrence_sweep.py``
     - Open
     - should combine shaped-tokamak Rosenbluth-Hinton-style residuals with W7-X residual/damping envelopes using one figure convention; use signed ``Phi_zonal_mode_kxt`` or case-specific signed line averages for publication claims and keep ``Phi2_zonal_t`` only as an intermediate cross-check. The current Merlo Case-III artifact is ``docs/_static/miller_zonal_response_pilot.png`` from the initial-density setup at ``Nz=32``, ``Nl=4``, ``Nm=24``, ``dt=0.005``, and ``t≈60``. With Rosenbluth-Hinton first-sample normalization it gives ``residual≈0.192`` against the Merlo et al. Figs. 12/16 read-off of about ``0.19``; a literature-faithful common fit window ``t≈30`` with separate positive/negative-extrema damping fits gives ``γ_GAM R0 / v_i≈-0.176`` against the paper-scale read-off near ``-0.17``; and Hilbert-phase frequency extraction on that same window gives ``ω_GAM R0 / v_i≈2.20`` against the paper-scale read-off near ``2.24``. A higher-moment audit lowers the recurrence ratio but over-damps the GAM, while weak hypercollision scans are effectively inert, so the frozen Merlo artifact remains on the current ``Nm=24`` baseline. The W7-X side now uses the potential initializer, signed line-average observable, paper-facing line-first normalization, and no hidden time-axis scaling. The tracked long-window W7-X artifact is ``docs/_static/w7x_zonal_response_panel.png`` with replayable traces in ``docs/_static/w7x_zonal_response_panel.traces.csv``; it reaches the digitized Fig. 11 windows, but ``docs/_static/w7x_zonal_reference_compare.json`` remains open because residuals fail at ``k_x rho_i=0.07``, ``0.10``, and ``0.30`` and the late envelopes are much larger than the digitized stella/GENE traces. ``docs/_static/w7x_zonal_contract_audit.png`` is now the paper-facing diagnostic panel for that open mismatch and is intentionally excluded from the release gate index. ``docs/_static/w7x_zonal_state_convention_audit.png`` closes the paper-facing state convention layer: the recovered Gaussian potential has relative ``L2`` error ``1.85e-6``, off-target spectral content is zero to reported precision, and the diagnostic helpers agree with manual line/volume reductions near ``2e-16``. ``docs/_static/w7x_zonal_moment_tail_audit.png``, ``docs/_static/w7x_zonal_closure_ladder_kx070.png``, and ``docs/_static/w7x_zonal_recurrence_sweep_kx070.png`` are companion open diagnostics; together they support a recurrence / moment-closure hypothesis and show that weak closure can reduce Hermite-tail metrics without closing the paper trace. The bounded recurrence sweep separates moment-resolution and closure-source effects over ``t v_t/a <= 100``: ``Nl=12,Nm=48`` gives the lowest no-closure mean absolute trace error, while constant-source closure lowers final Hermite-tail energy from ``0.388`` to ``0.062`` but worsens the trace error. The W7-X generator now exposes explicit hypercollision and Gaussian-width audit overrides so future closure probes are reproducible from the tracked tool. A newer high-moment four-wavelength audit under ``tools_out/zonal_response/w7x_publication_nl16_nm64_dt005_t100`` verifies finite signed traces to ``t≈100`` after restart-continuation fixes. The tracked W7-X TOML keeps ``gaussian_width=1`` because the benchmark source writes the initializer as ``exp[-(z-z0)^2]``; wider profiles and non-unit time scales are retained only as audits. The lane remains open pending the W7-X damping/closure and velocity-space recurrence fix under the paper-facing convention.
   * - W7-X fluctuation-spectrum panel
     - dedicated script to add
     - Open
     - should follow the W7-X Doppler-reflectometry comparison figure family for density and zonal-flow spectra
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
     - Open
     - keep separate from validation figures
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
   * - Inverse/UQ figure
     - ``examples/theory_and_demos/autodiff_inverse_twomode.py`` plus UQ follow-on
     - Open
     - must separate identifiability from recovery quality
   * - Optimization figure
     - dedicated script to add
     - Open
     - low-dimensional stellarator objective reduction with validated gradients

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
- add W7-X fluctuation-spectrum figures tied to the DR-comparison conventions
- extend eigenfunction-overlap metrics beyond the closed KBM and W7-X raw
  overlays where additional literature-backed mode-shape references are useful
- tighten the current windowed nonlinear statistics panel with stricter case-specific gates where reference windows justify them
- tie ETG nonlinear claims to the benchmark literature or keep them framed as a pilot
- add publication-ready figure scripts for zonal-flow and fluctuation-spectrum panels
