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
- Merlo et al., *Linear multispecies gyrokinetic flux tube benchmarks in shaped
  tokamak plasmas*:
  shaping scans, ballooning-angle handling, Rosenbluth-Hinton residuals, and
  GAM damping.
- González-Jerez et al., *Electrostatic microturbulence in W7-X: comparison of
  local gyrokinetic simulations with Doppler reflectometry measurements*:
  fluctuation amplitudes, frequency spectra, and zonal-flow spectral content.
- Maurer et al., *Global electromagnetic turbulence simulations of W7-X-like
  plasmas with GENE-3D*:
  heavy-electron electromagnetic verification before realistic-mass stellarator
  production runs.

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
   * - Shaped tokamak zonal-flow / GAM
     - residual level, damping rate, GAM envelope
     - Merlo et al. + analytical Rosenbluth-Hinton estimates where applicable
     - Open
     - residual and damping must match literature/code-backed references before publication use; signed ``Phi_zonal_mode_kxt`` is now available. The current artifact is ``docs/_static/miller_zonal_response_pilot.png`` from ``tools/generate_miller_zonal_response_pilot.py`` using Merlo Case-III Table-III parameters, an initial density perturbation, a common pre-recurrence fit window ``t≈30``, separate positive/negative-extrema damping fits, and a Hilbert-phase frequency extraction on that same window. It gives ``residual≈0.192`` against a paper-scale target of about ``0.19``, ``ω_GAM R0 / v_i≈2.20`` against a figure read-off near ``2.24``, and ``γ_GAM R0 / v_i≈-0.176`` against a figure read-off near ``-0.17``. The remaining explicit follow-up item is the later finite-moment recurrence rather than the benchmark-scale Merlo gate

Frozen artifact paths for the currently closed tokamak linear lanes:

- ``docs/_static/cyclone_comparison.png``
- ``docs/_static/etg_comparison.png``
- ``docs/_static/kbm_comparison.png``
- ``docs/_static/kbm_eigenfunction_overlap_summary.png``
- ``docs/_static/reference_modes/kbm_linear_gx_ky0p3000.npz``
- ``docs/_static/benchmark_core_linear_atlas.png``

Open raw-overlay diagnostic artifacts for the KBM lane:

- ``docs/_static/reference_modes/kbm_linear_spectrax_ky0p3000.csv``
- ``docs/_static/kbm_eigenfunction_reference_overlay_ky0p3000.png``
- ``tools/generate_kbm_reference_overlay.py``

These are useful for manuscript drafting and debugging, but they are not yet
accepted as closed validation artifacts. The current bounded-cost extraction
produces approximately ``0.63`` normalized overlap and ``0.79`` relative
``L^2`` mismatch against the frozen GX raw mode at ``k_y \approx 0.3`` when
run with the exact KBM grid contract, the selected growth-fit window, and a
late-time eigenfunction tail window. This indicates the bounded short-horizon
raw overlay is still not converged enough for paper use. The generator now
writes a machine-readable gate report with ``overlap >= 0.95`` and
``relative L^2 <= 0.25`` as the acceptance policy for closing this raw-overlay
artifact.

Branch-followed scan tables should use the same gate-report convention:
observed-order gates for resolution or velocity-space convergence, and
branch-continuity gates for adjacent ``gamma``/``omega`` jumps and successive
eigenfunction overlap when overlap data are available. The tracked KBM
candidate table now has a no-rerun summary path through
``tools/generate_kbm_branch_gate_summary.py`` and
``docs/_static/kbm_branch_gate_summary.json``. That summary currently marks
the branch-continuity lane open: the largest adjacent growth-rate jump is about
``0.60`` against the strict ``0.50`` threshold, while the corresponding
frequency-jump and successive-overlap checks pass.

Observed-order convergence tables should also gate both the asymptotic finest
refinement and the full set of pairwise refinement orders. The generic
``tools/generate_observed_order_gate.py`` path now records this policy in JSON.
The tracked Cyclone resolution pilot
``docs/_static/cyclone_resolution_observed_order.json`` is open because the
coarse-to-mid pair has a negative observed order, even though the final-grid
relative growth-rate error is about ``1.3e-2``.

The current materialized gate reports are indexed by
``tools/make_validation_gate_index.py`` in
``docs/_static/validation_gate_index.json`` and
``docs/_static/validation_gate_index.png``. At this stage the index is an audit
artifact: it should expand as each validation lane gets its own frozen gate
metadata.

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
     - a case-specific runtime/tool path now exists through ``examples/benchmarks/runtime_w7x_zonal_response_vmec.toml`` and ``tools/generate_w7x_zonal_response_panel.py`` using the same first-sample / branchwise-extrema / Hilbert-phase extraction policy as the Merlo lane. The remaining open item is generation of a frozen VMEC-backed artifact and acceptance metrics on an actual W7-X geometry file
   * - W7-X fluctuation spectra
     - density and zonal-flow frequency spectra
     - W7-X Doppler-reflectometry comparison paper
     - Open
     - requires a reproducible spectral estimator and windowing policy
   * - HSX
     - ``gamma(k_y)``, ``omega(k_y)``
     - GX / internal frozen references
     - Closed
     - near-marginal deviations documented explicitly
   * - Electromagnetic stellarator verification
     - heavy-electron linear/nonlinear EM response
     - GENE-3D verification conventions
     - Open
     - close heavy-electron EM lane before realistic-mass claims

Frozen artifact paths for the currently closed stellarator linear lanes:

- ``docs/_static/w7x_linear_t2_scan.csv``
- ``docs/_static/hsx_linear_t2_scan.csv``
- ``docs/_static/w7x_linear_t2_lastvalue.csv``
- ``docs/_static/hsx_linear_t2_lastvalue.csv``
- ``docs/_static/reference_modes/w7x_linear_gx_ky0p3000.npz``
- ``docs/_static/benchmark_core_linear_atlas.png``

For W7-X, the whole-window scan and the late-time last-value reduction tell the
same story. For HSX, the whole-window ``mean_rel_gamma`` metric is kept as an
honest near-marginal stress signal, but the late-time closure should be read
from ``docs/_static/hsx_linear_t2_lastvalue.csv`` because the final
``(gamma, omega)`` values are much tighter than the whole-window average.

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

Machine-readable nonlinear window gates are now tracked for the first refreshed
subset:

- ``docs/_static/nonlinear_cyclone_miller_gate_summary.json``: passed at the
  current ``0.10`` mean-relative release gate.
- ``docs/_static/nonlinear_kbm_gate_summary.json``: passed at the current
  ``0.10`` mean-relative release gate.
- ``docs/_static/nonlinear_hsx_gate_summary.json``: passed at the current
  ``0.10`` mean-relative release gate.
- ``docs/_static/nonlinear_w7x_gate_summary.json``: open because
  ``Wphi`` mean-relative mismatch is about ``0.116``.
- ``docs/_static/nonlinear_cyclone_short_gate_summary.json``: open because this
  short diagnostic window is not a mature transport-window acceptance run.

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
- Raw eigenfunction overlays for manuscript use should be rendered only from
  frozen reference bundles checked into ``docs/_static/reference_modes/``.
  Do not build publication figures from transient external files or ad hoc
  office-machine outputs.
- Experimental-facing figures such as W7-X fluctuation spectra should be marked
  as literature-aligned unless the diagnostic transfer function and access
  model are encoded directly in the repo.
- Electromagnetic stellarator claims should be split explicitly into
  heavy-electron verification and realistic-electron research runs.
