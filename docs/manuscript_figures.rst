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
     - first shipped overlap artifact is ``docs/_static/kbm_eigenfunction_overlap_summary.png``; raw GX bundles now exist for KBM and W7-X under ``docs/_static/reference_modes/`` and the first bounded-cost raw overlay is ``docs/_static/kbm_eigenfunction_reference_overlay_ky0p3000.png``. The reproducible generator for that artifact is now ``tools/generate_kbm_reference_overlay.py``. The raw KBM overlay remains an open validation figure because the bounded exact-contract extraction still shows only about ``0.63`` overlap and ``0.79`` relative ``L^2`` mismatch.
   * - Nonlinear transport panel
     - ``tools/make_gx_summary_panel.py`` / ``tools/make_gx_publication_panel.py``
     - Open
     - Cyclone, Miller, KBM, W7-X, HSX with matched windows. Current component artifacts: ``docs/_static/nonlinear_cyclone_diag_compare_t400.png``, ``docs/_static/nonlinear_cyclone_miller_diag_compare_t122.png``, ``docs/_static/nonlinear_kbm_diag_compare_t400_stats.png``, ``docs/_static/nonlinear_w7x_diag_compare_t200.png``, ``docs/_static/hsx_nonlinear_compare_t50_true.png``.
   * - Windowed-statistics summary
     - dedicated script to add
     - Open
     - this should be the acceptance-summary figure for nonlinear parity
   * - Zonal-flow / GAM response panel
     - dedicated script to add
     - Open
     - should combine shaped-tokamak Rosenbluth-Hinton-style residuals with W7-X residual/damping envelopes using one figure convention
   * - W7-X fluctuation-spectrum panel
     - dedicated script to add
     - Open
     - should follow the W7-X Doppler-reflectometry comparison figure family for density and zonal-flow spectra
   * - Velocity-space convergence panel
     - dedicated script to add
     - Open
     - should follow GX-style convergence evidence
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
- add shaped-tokamak Rosenbluth-Hinton / GAM response benchmarks
- add W7-X fluctuation-spectrum figures tied to the DR-comparison conventions
- add eigenfunction-overlap metrics to the linear figure stack
- freeze representative reference mode bundles under ``docs/_static/reference_modes/`` before drafting raw overlay figures
- add windowed nonlinear statistics as first-class manuscript artifacts
- tie ETG nonlinear claims to the benchmark literature or keep them framed as a pilot
- add publication-ready figure scripts for eigenfunction-overlap, zonal-flow, and fluctuation-spectrum panels
