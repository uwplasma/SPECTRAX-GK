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
     - first shipped overlap artifact is ``docs/_static/kbm_eigenfunction_overlap_summary.png``; frozen GX raw-mode bundles now exist for KBM and W7-X under ``docs/_static/reference_modes/``, but the first full raw overlay still needs a bounded matched SPECTRAX extraction pass
   * - Nonlinear transport panel
     - ``tools/make_gx_summary_panel.py`` / ``tools/make_gx_publication_panel.py``
     - Open
     - Cyclone, Miller, KBM, W7-X, HSX with matched windows. Current component artifacts: ``docs/_static/nonlinear_cyclone_diag_compare_t400.png``, ``docs/_static/nonlinear_cyclone_miller_diag_compare_t122.png``, ``docs/_static/nonlinear_kbm_diag_compare_t400_stats.png``, ``docs/_static/nonlinear_w7x_diag_compare_t200.png``, ``docs/_static/hsx_nonlinear_compare_t50_true.png``.
   * - Windowed-statistics summary
     - dedicated script to add
     - Open
     - this should be the acceptance-summary figure for nonlinear parity
   * - Velocity-space convergence panel
     - dedicated script to add
     - Open
     - should follow GX-style convergence evidence
   * - Stellarator validation panel
     - dedicated script to add
     - Open
     - W7-X multi-flux-tube + zonal-flow response + HSX summary as needed
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
- add eigenfunction-overlap metrics to the linear figure stack
- freeze representative reference mode bundles under ``docs/_static/reference_modes/`` before drafting raw overlay figures
- add windowed nonlinear statistics as first-class manuscript artifacts
- tie ETG nonlinear claims to the benchmark literature or keep them framed as a pilot
- add publication-ready figure scripts for eigenfunction-overlap and stellarator zonal-flow panels
