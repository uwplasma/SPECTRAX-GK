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
     - ``tools/plot_eigenfunction_overlap_summary.py`` and ``tools/plot_eigenfunction_reference_overlay.py``
     - Open
     - first shipped overlap artifact is ``docs/_static/kbm_eigenfunction_overlap_summary.png``; raw GX bundles now exist for KBM and W7-X under ``docs/_static/reference_modes/`` and the first bounded-cost raw overlay is ``docs/_static/kbm_eigenfunction_reference_overlay_ky0p3000.png``. The reproducible generator for that artifact is now ``tools/generate_kbm_reference_overlay.py`` and writes a JSON gate report with ``overlap >= 0.95`` and ``relative L^2 <= 0.25`` requirements. ``tools/compare_gx_kbm.py --branch-summary-json`` writes branch-continuity gate metadata for selected KBM scans, and ``tools/generate_kbm_branch_gate_summary.py`` refreshes the no-rerun tracked artifact ``docs/_static/kbm_branch_gate_summary.json`` from ``docs/_static/kbm_gx_candidates.csv``. The current branch summary is open because the largest adjacent growth-rate jump is about ``0.60`` against the strict ``0.50`` gate, while the adjacent frequency-jump and successive-overlap gates pass. The raw KBM overlay remains an open validation figure because the bounded exact-contract extraction still shows only about ``0.63`` overlap and ``0.79`` relative ``L^2`` mismatch.
   * - Nonlinear transport panel
     - ``tools/make_gx_summary_panel.py`` / ``tools/make_gx_publication_panel.py``
     - Open
     - Cyclone, Miller, KBM, W7-X, HSX with matched windows. Current component artifacts: ``docs/_static/nonlinear_cyclone_diag_compare_t400.png``, ``docs/_static/nonlinear_cyclone_miller_diag_compare_t122.png``, ``docs/_static/nonlinear_kbm_diag_compare_t400_stats.png``, ``docs/_static/nonlinear_w7x_diag_compare_t200.png``, ``docs/_static/hsx_nonlinear_compare_t50_true.png``. ``tools/compare_gx_nonlinear_diagnostics.py --summary-json`` now writes machine-readable mean-relative-mismatch gates for each plotted diagnostic, so refreshed panels can carry a JSON acceptance record instead of only printed console output. The first materialized window-gate metadata are ``docs/_static/nonlinear_cyclone_miller_gate_summary.json``, ``docs/_static/nonlinear_kbm_gate_summary.json``, ``docs/_static/nonlinear_hsx_gate_summary.json``, ``docs/_static/nonlinear_w7x_gate_summary.json``, and ``docs/_static/nonlinear_cyclone_short_gate_summary.json``; with the current ``0.10`` mean-relative release gate, Cyclone Miller, KBM, HSX, and W7-X pass, while the short Cyclone diagnostic remains open.
   * - Windowed-statistics summary
     - ``tools/compare_gx_nonlinear_diagnostics.py`` plus ``tools/make_validation_gate_index.py``
     - Open
     - per-case nonlinear gate JSONs are now indexed by ``docs/_static/validation_gate_index.png``. The next step is a dedicated nonlinear-only summary panel with mature-lane thresholds separated from exploratory/short-run diagnostics.
   * - Validation gate index
     - ``tools/make_validation_gate_index.py``
     - Open
     - current artifact base: ``docs/_static/validation_gate_index.png`` and ``docs/_static/validation_gate_index.json``. This is not a physics result by itself; it is the audit panel that reports which tracked gate artifacts are closed or open before manuscript drafting.
   * - Zonal-flow / GAM response panel
     - ``tools/plot_zonal_flow_response.py``, ``tools/plot_zonal_flow_response_from_output.py``, ``tools/generate_miller_zonal_response_pilot.py``, and ``tools/generate_w7x_zonal_response_panel.py``
     - Open
     - should combine shaped-tokamak Rosenbluth-Hinton-style residuals with W7-X residual/damping envelopes using one figure convention; use signed ``Phi_zonal_mode_kxt`` for publication claims and keep ``Phi2_zonal_t`` only as an intermediate cross-check. The current Merlo Case-III artifact is ``docs/_static/miller_zonal_response_pilot.png`` from the initial-density setup at ``Nz=32``, ``Nl=4``, ``Nm=24``, ``dt=0.005``, and ``t≈60``. With Rosenbluth-Hinton first-sample normalization it gives ``residual≈0.192`` against the Merlo et al. Figs. 12/16 read-off of about ``0.19``; a literature-faithful common fit window ``t≈30`` with separate positive/negative-extrema damping fits gives ``γ_GAM R0 / v_i≈-0.176`` against the paper-scale read-off near ``-0.17``; and Hilbert-phase frequency extraction on that same window gives ``ω_GAM R0 / v_i≈2.20`` against the paper-scale read-off near ``2.24``. A higher-moment audit lowers the recurrence ratio but over-damps the GAM, while weak hypercollision scans are effectively inert, so the frozen Merlo artifact remains on the current ``Nm=24`` baseline. The W7-X side now has a dedicated panel generator and runtime contract, but the frozen VMEC-backed artifact is still open.
   * - W7-X fluctuation-spectrum panel
     - dedicated script to add
     - Open
     - should follow the W7-X Doppler-reflectometry comparison figure family for density and zonal-flow spectra
   * - Velocity-space convergence panel
     - ``tools/generate_observed_order_gate.py`` plus dedicated full convergence refresh script to add
     - Open
     - should follow GX-style convergence evidence and write an observed-order gate report through ``spectraxgk.benchmarking.observed_order_gate_report`` so convergence rate and final-grid error are tracked explicitly. The current atlas summary already records a machine-readable high-vs-low Cyclone grid convergence gate for the tracked convergence tile. The first generic CSV-backed pilot is ``docs/_static/cyclone_resolution_observed_order.png`` with metadata in ``docs/_static/cyclone_resolution_observed_order.json``; it is correctly open because the coarse-to-mid refinement is nonmonotone, so it is a gate-path validation artifact rather than final manuscript evidence.
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

- close or explicitly defer W7-X zonal-flow response
- either close the long-time recurrence follow-up for the shaped-tokamak
  Rosenbluth-Hinton / GAM response benchmark or explicitly scope publication
  claims to the benchmark-scale pre-recurrence metrics now frozen in the
  Merlo Case-III artifact
- add W7-X fluctuation-spectrum figures tied to the DR-comparison conventions
- add eigenfunction-overlap metrics to the linear figure stack
- freeze representative reference mode bundles under ``docs/_static/reference_modes/`` before drafting raw overlay figures
- add windowed nonlinear statistics as first-class manuscript artifacts
- tie ETG nonlinear claims to the benchmark literature or keep them framed as a pilot
- add publication-ready figure scripts for eigenfunction-overlap, zonal-flow, and fluctuation-spectrum panels
