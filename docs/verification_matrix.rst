Verification Matrix
===================

Purpose
-------

This page is the research-facing index of what SPECTRAX-GK treats as verified,
validated, exploratory, or deferred. It is meant to answer four questions for
each lane:

1. what physical model is being exercised,
2. what observable is compared,
3. what the reference is,
4. what acceptance gate applies.

Literature Baselines Reviewed
-----------------------------

The current matrix is anchored on these published baselines:

- Tronko et al., *Verification of Gyrokinetic codes: theoretical background and applications*:
  verification methodology, observed-order checks, and benchmark-observable
  framing.
- Mandell et al., *GX: a GPU-native gyrokinetic turbulence code for tokamak and stellarator design*:
  CBC, W7-X, KBM, nonlinear transport, velocity-space convergence, and
  performance figure conventions.
- González-Jerez et al., *Electrostatic gyrokinetic simulations in W7-X geometry*:
  W7-X ITG/TEM scans, zonal-flow response, and nonlinear ITG heat flux.
- Nevins et al., *Characterizing electron temperature gradient turbulence*:
  ETG operating-point conventions.
- Monreal et al., *Residual zonal flows in tokamaks and stellarators at arbitrary wavelengths*:
  residual-zonal-flow metrics and damping interpretation.

Status Legend
-------------

- ``Closed``: benchmark lane is accepted for research claims.
- ``Open``: lane is active and expected to close.
- ``Exploratory``: useful for development, not yet a paper claim.
- ``Deferred``: intentionally out of scope for the current paper/release.

Tokamak Linear
--------------

.. list-table::
   :header-rows: 1

   * - Lane
     - Observable
     - Reference
     - Status
     - Baseline gate
   * - Cyclone ITG
     - ``gamma(k_y)``, ``omega(k_y)``, eigenfunction overlap
     - GX + CBC literature
     - Closed
     - ``rtol <= 1e-2`` except documented low-``k_y`` / near-marginal cases
   * - ETG
     - ``gamma(k_y)``, ``omega(k_y)``
     - GX + ETG benchmark literature
     - Closed
     - ``rtol <= 1e-2`` on the tracked branch
   * - KBM
     - ``gamma(k_y)``, ``omega(k_y)``, branch continuity vs ``beta``
     - GX
     - Closed
     - ``rtol <= 1e-2`` on the accepted branch
   * - KAW
     - branch-followed linear response
     - GX
     - Deferred
     - close branch identity before publication use
   * - TEM
     - ``gamma(k_y)``, ``omega(k_y)``
     - GX / literature
     - Open
     - close branch-following and reference selection first
   * - Shaped multispecies tokamak
     - ``gamma``, ``omega``, eigenfunction shape
     - Sauter benchmark set
     - Open
     - literature-backed operating point and overlap gate required

Frozen artifact paths for the currently closed tokamak linear lanes:

- ``docs/_static/cyclone_comparison.png``
- ``docs/_static/etg_comparison.png``
- ``docs/_static/kbm_comparison.png``
- ``docs/_static/kbm_eigenfunction_overlap_summary.png``
- ``docs/_static/benchmark_core_linear_atlas.png``

Stellarator Linear
------------------

.. list-table::
   :header-rows: 1

   * - Lane
     - Observable
     - Reference
     - Status
     - Baseline gate
   * - W7-X ITG/TEM flux tube
     - ``gamma(k_y)``, ``omega(k_y)``
     - stella/GENE benchmark paper + GX
     - Closed
     - ``rtol <= 1e-2`` on closed branches
   * - W7-X zonal flow
     - residual level, damping envelope
     - stella/GENE benchmark paper + zonal-flow literature
     - Open
     - residual and damping metrics must be defined before acceptance
   * - HSX
     - ``gamma(k_y)``, ``omega(k_y)``
     - GX / internal frozen references
     - Closed
     - near-marginal deviations documented explicitly

Frozen artifact paths for the currently closed stellarator linear lanes:

- ``docs/_static/w7x_linear_t2_scan.csv``
- ``docs/_static/hsx_linear_t2_scan.csv``
- ``docs/_static/benchmark_core_linear_atlas.png``

Nonlinear Validation
--------------------

.. list-table::
   :header-rows: 1

   * - Lane
     - Observable
     - Reference
     - Status
     - Baseline gate
   * - Cyclone ITG
     - heat-flux window mean/std/RMS, ``Wphi``, ``Wg``
     - GX
     - Closed
     - mature lanes target ``<= 5e-2`` windowed mismatch
   * - Cyclone Miller
     - same as above
     - GX
     - Closed
     - allow documented low-amplitude / overlap-only adjustments
   * - KBM
     - ``Wg``, ``Wphi``, ``Wapar``, heat flux
     - GX
     - Closed
     - mature lane
   * - W7-X
     - heat-flux windows, saturation trend
     - GX + W7-X benchmark conventions
     - Closed
     - release gate ``<= 1e-1``; manuscript target tighter where feasible
   * - HSX
     - heat-flux windows, saturation trend
     - GX / internal frozen references
     - Closed
     - near-threshold behavior documented
   * - ETG full-GK pilot
     - short-window nonlinear transport
     - GX + ETG operating-point convention
     - Exploratory
     - manuscript use only if the pilot is explicitly framed as such
   * - kinetic-electron Cyclone
     - electromagnetic nonlinear transport
     - GX
     - Deferred
     - keep out of the paper until branch identity and runtime cost are closed

Frozen artifact paths for the currently closed nonlinear lanes:

- ``docs/_static/nonlinear_cyclone_diag_compare_t400.png``
- ``docs/_static/nonlinear_cyclone_miller_diag_compare_t122.png``
- ``docs/_static/nonlinear_kbm_diag_compare_t400_stats.png``
- ``docs/_static/nonlinear_w7x_diag_compare_t200.png``
- ``docs/_static/hsx_nonlinear_compare_t50_true.png``
- ``docs/_static/benchmark_core_nonlinear_atlas.png``

Autodiff Validation
-------------------

.. list-table::
   :header-rows: 1

   * - Workflow
     - Observable
     - Validation type
     - Status
   * - Sensitivity analysis
     - ``gamma``, ``omega``, transport windows
     - finite-difference / complex-step / tangent consistency
     - Open
   * - Two-mode inverse problem
     - planted parameter recovery
     - gradient check + covariance estimate
     - Closed
   * - UQ / Laplace example
     - posterior covariance and propagated uncertainty
     - Hessian/Jacobian validation
     - Open
   * - Stellarator optimization prototype
     - low-dimensional objective reduction
     - gradient consistency + constrained solve behavior
     - Open

Notes
-----

- A lane should not move from ``Open`` to ``Closed`` without an owning script,
  frozen artifact path, and literature/reference statement.
- README figures should use only ``Closed`` lanes unless a panel is explicitly
  marked exploratory.
