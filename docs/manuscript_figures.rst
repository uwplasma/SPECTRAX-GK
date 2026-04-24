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
   * - Windowed-statistics summary
     - ``tools/plot_nonlinear_window_statistics.py``
     - Closed for current release-window gates
     - current artifact base: ``docs/_static/nonlinear_window_statistics.png`` with CSV/JSON/PDF companions. It summarizes the per-diagnostic ``mean_rel_abs`` and ``max_rel_abs`` statistics from the frozen nonlinear GX comparison gate JSONs for Cyclone, Cyclone Miller, KBM, W7-X, and HSX. Exploratory/short-run diagnostics are explicitly excluded with ``gate_index_include=false``.
   * - Validation gate index
     - ``tools/make_validation_gate_index.py``
     - Closed for currently tracked gates
     - current artifact base: ``docs/_static/validation_gate_index.png`` and ``docs/_static/validation_gate_index.json``. This is not a physics result by itself; it is the audit panel for release-window gates, currently ``10/10`` passed.
   * - Zonal-flow / GAM response panel
     - ``tools/plot_zonal_flow_response.py``, ``tools/plot_zonal_flow_response_from_output.py``, ``tools/generate_miller_zonal_response_pilot.py``, ``tools/generate_w7x_zonal_response_panel.py``, ``tools/digitize_w7x_zonal_reference.py``, and ``tools/compare_w7x_zonal_reference.py``
     - Open
     - should combine shaped-tokamak Rosenbluth-Hinton-style residuals with W7-X residual/damping envelopes using one figure convention; use signed ``Phi_zonal_mode_kxt`` or case-specific signed line averages for publication claims and keep ``Phi2_zonal_t`` only as an intermediate cross-check. The current Merlo Case-III artifact is ``docs/_static/miller_zonal_response_pilot.png`` from the initial-density setup at ``Nz=32``, ``Nl=4``, ``Nm=24``, ``dt=0.005``, and ``t≈60``. With Rosenbluth-Hinton first-sample normalization it gives ``residual≈0.192`` against the Merlo et al. Figs. 12/16 read-off of about ``0.19``; a literature-faithful common fit window ``t≈30`` with separate positive/negative-extrema damping fits gives ``γ_GAM R0 / v_i≈-0.176`` against the paper-scale read-off near ``-0.17``; and Hilbert-phase frequency extraction on that same window gives ``ω_GAM R0 / v_i≈2.20`` against the paper-scale read-off near ``2.24``. A higher-moment audit lowers the recurrence ratio but over-damps the GAM, while weak hypercollision scans are effectively inert, so the frozen Merlo artifact remains on the current ``Nm=24`` baseline. The W7-X side now uses the potential initializer, line-average observable, and initial-Gaussian-maximum normalization. The long-window W7-X artifact is ``docs/_static/w7x_zonal_response_panel.png``; it reaches the digitized Fig. 11 windows, and ``docs/_static/w7x_zonal_reference_compare.json`` now passes the residual and time-coverage gates for all four wavelengths. A newer high-moment four-wavelength audit under ``tools_out/zonal_response/w7x_publication_nl16_nm64_dt005_t100`` verifies finite signed traces to ``t≈100`` after restart-continuation fixes. The tracked W7-X TOML now uses ``gaussian_width=4`` because it matches the digitized initial line-average level better than the earlier width-1 setup, and the panel generator records ``time_scale=2`` to place runtime traces on the digitized ``t v_ti/a`` reference axis. The lane remains open pending the longer-time velocity-space recurrence / closure fix after these explicit comparison conventions are applied.
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

Differentiable-Physics Figures
------------------------------

.. list-table::
   :header-rows: 1

   * - Figure
     - Owning script
     - Status
     - Notes
   * - Sensitivity-analysis figure
     - dedicated script to add
     - Open
     - local derivatives of linear/nonlinear observables
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

- close the W7-X zonal-flow late-envelope mismatch now that the long-window
  Gaussian-potential artifact passes the residual and time-coverage gates
  against digitized stella/GENE traces
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
