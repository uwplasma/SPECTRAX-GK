Differentiable Stellarator Optimization
=======================================

Purpose
-------

SPECTRAX-GK now includes a compact, fully JAX-differentiable stellarator ITG
optimization layer. It is designed as the validation gate before promoting a
full ``vmec_jax -> booz_xform_jax -> SPECTRAX-GK`` nonlinear optimization loop.
The current examples optimize a quasi-axisymmetric, max-mode-1 control vector
around a target aspect ratio ``A = 7`` and mean rotational transform
``iota = 0.41``. They follow the same objective-block logic used by the local
``vmec_jax`` fixed-boundary QA examples: preserve aspect, preserve iota,
reduce quasisymmetry error, and add a turbulence objective.

This page is deliberately conservative about claims:

- the examples below are end-to-end differentiable and finite-difference
  checked;
- they validate optimization, sensitivity, covariance, and plotting machinery;
- they do **not** yet claim a production VMEC/Boozer/nonlinear gyrokinetic
  stellarator optimization, because the high-fidelity geometry path still
  needs parity and gradient gates before it can replace the reduced feature
  map used here.

Source Map
----------

- Core API: :mod:`spectraxgk.stellarator_optimization`
- Production in-memory geometry boundary:
  :func:`spectraxgk.flux_tube_geometry_from_vmec_boozer_state`
- Production-adjacent linear/quasilinear objective evaluator:
  :func:`spectraxgk.vmec_boozer_solver_objective_vector_from_state`
- Scalar optimizer hook:
  :func:`spectraxgk.vmec_boozer_scalar_objective_from_state`
- Multi-point objective table and aggregate hooks:
  :func:`spectraxgk.vmec_boozer_solver_objective_table_from_state`,
  :func:`spectraxgk.vmec_boozer_aggregate_scalar_objective_from_state`
- VMEC-state finite-difference sensitivity audit:
  :func:`spectraxgk.vmec_boozer_scalar_objective_finite_difference_report`
- Multi-point finite-difference sensitivity audit:
  :func:`spectraxgk.vmec_boozer_aggregate_scalar_objective_finite_difference_report`
- Curvature-gated one-parameter line search:
  :func:`spectraxgk.vmec_boozer_scalar_objective_line_search_report`
- Multi-point curvature-gated one-parameter line search:
  :func:`spectraxgk.vmec_boozer_aggregate_scalar_objective_line_search_report`
- Held-out aggregate line-search validation:
  :func:`spectraxgk.vmec_boozer_aggregate_line_search_holdout_report`
- Held-out aggregate promotion artifact check:
  ``tools/check_vmec_boozer_aggregate_holdout_gate.py``
- Fast branch-continuity and sensitivity gate:
  :func:`spectraxgk.solver_objective_branch_gradient_report`
- Tests: ``tests/test_stellarator_optimization.py``
- Growth-rate example:
  :download:`stellarator_itg_growth_optimization.py <../examples/optimization/stellarator_itg_growth_optimization.py>`
- Quasilinear-flux example:
  :download:`stellarator_itg_quasilinear_flux_optimization.py <../examples/optimization/stellarator_itg_quasilinear_flux_optimization.py>`
- Nonlinear-window example:
  :download:`stellarator_itg_nonlinear_heat_flux_optimization.py <../examples/optimization/stellarator_itg_nonlinear_heat_flux_optimization.py>`
- Three-objective comparison:
  :download:`compare_stellarator_itg_optimizations.py <../examples/optimization/compare_stellarator_itg_optimizations.py>`
- Plotting helper:
  :download:`_stellarator_itg_plotting.py <../examples/optimization/_stellarator_itg_plotting.py>`

The corresponding ``vmec_jax`` workflow that motivated this structure is the
local fixed-boundary QA script
``/Users/rogeriojorge/local/vmec_jax/examples/optimization/qa_fixed_resolution_jax_ess.py``.
That script builds residual blocks for aspect ratio, mean iota, and
quasisymmetry, then minimizes them over boundary Fourier coefficients. The
SPECTRAX-GK examples use the same optimization pattern but keep the transport
objective inside a trace-safe reduced map until the production geometry bridge
is fully gated.

The next implementation stage replaces that reduced map with
``flux_tube_geometry_from_vmec_boozer_state`` followed by
``vmec_boozer_solver_objective_vector_from_state``. The latter evaluates the
dominant SPECTRAX-GK linear/quasilinear objective vector from the in-memory
geometry path. It is a forward evaluator, not by itself a gradient claim:
end-to-end differentiability is claimed only after VMEC/Boozer geometry parity,
branch-continuity, and AD/finite-difference gates pass for the optimized
equilibrium and held-out field lines.

For CI-scale development, ``solver_objective_branch_gradient_report`` applies
the same branch-continuity and implicit AD/finite-difference logic to the
solver-ready differentiable geometry contract. It verifies that the selected
max-growth eigenbranch stays dominant under central perturbations and that
the objective-vector sensitivities pass the implicit left/right eigenpair
gate. The VMEC/Boozer offline gates remain the authority for production
stellarator optimization claims.

Optimizer drivers should use ``vmec_boozer_scalar_objective_from_state`` once
they have a solved ``vmec_jax`` state. The supported aliases are
``growth``/``gamma``, ``frequency``/``omega``, and
``quasilinear_flux``/``mixing_length_heat_flux_proxy``. This selector prevents
each optimization example from silently using a different objective index.

The production-facing geometry objective should not stay tied to one field
line or one ``k_y`` point. ``vmec_boozer_solver_objective_table_from_state``
evaluates the same solver-objective vector over explicit ``surface_indices``,
field-line ``alphas``, and ``selected_ky_indices`` and returns the full table
before any reduction. ``vmec_boozer_aggregate_scalar_objective_from_state``
then reduces that table with a mean, weighted mean, or worst-case max. Mean and
weighted mean are the preferred gradient-development targets because the
sample set is fixed. A max reduction is useful as a conservative diagnostic,
but it must not be treated as a smooth optimizer objective unless active-set
and branch-continuity diagnostics are also passed.

Before any optimizer loop is promoted, run
``vmec_boozer_scalar_objective_finite_difference_report`` on the selected
VMEC coefficient, field line, and objective. It evaluates the scalar objective
at ``x-h``, ``x``, and ``x+h`` through the same in-memory VMEC/Boozer path and
records the central finite-difference sensitivity. The report also checks a
curvature/branch-switch indicator so a non-smooth max-growth branch is not
mistaken for a usable optimization gradient. This is intentionally a
finite-difference/SPSA-compatible audit, not an automatic-differentiation claim
for eigenvector-dependent quasilinear observables.

For multi-surface or multi-``k_y`` objectives, run
``vmec_boozer_aggregate_scalar_objective_finite_difference_report`` with the
same sample set and weights used by the optimizer. The report records the
sample metadata, scalar values, objective tables, and the same
curvature/branch-switch indicator. This is the minimum gate before a
stellarator optimization study can claim that a reduced growth-rate or
quasilinear objective decreased across more than one field line, surface, or
``k_y`` point.

The first optimizer scaffold is
``vmec_boozer_scalar_objective_line_search_report``. It repeatedly applies the
finite-difference audit at the current VMEC coefficient offset and accepts only
candidate updates that both pass the same curvature gate and reduce the scalar
objective. This is useful for growth-rate and quasilinear-flux optimizer
plumbing, but it remains a one-parameter audit rather than a multi-parameter
stellarator optimization claim.

For multi-point reduced objectives, use
``vmec_boozer_aggregate_scalar_objective_line_search_report`` instead. It
applies the aggregate finite-difference gate at every attempted VMEC
coefficient update and records the same sample metadata as the aggregate gate.
This is now the preferred scaffold for growth-rate and quasilinear-flux
optimization studies that need more than one field line, surface, or ``k_y``
point before entering a full optimizer loop.

Training improvement is not enough for a geometry-wide claim.
``vmec_boozer_aggregate_line_search_holdout_report`` runs the same aggregate
line-search on a training sample set, then evaluates the final coefficient
offset on a disjoint held-out sample set. The split gate passes only if both
the training line-search gate and held-out aggregate reduction pass. This is
the minimum reduced-objective validation step before using an optimized VMEC
coefficient in manuscript figures.

Production promotion adds a stricter surface/field-line rule. The aggregate
finite-difference, line-search, and reduced holdout reports must be paired with
at least one separate passed validation artifact whose sample metadata covers a
held-out ``surface_index`` or field-line ``alpha``. A held-out ``k_y`` point
alone is useful spectrum coverage, but it is not sufficient for the
surface/field-line generalization gate. The repository-level check
``tools/check_vmec_boozer_aggregate_holdout_gate.py`` encodes that boundary for
frozen artifacts: it accepts the aggregate FD and line-search artifacts as
necessary optimizer-plumbing evidence, then blocks promotion until independent
surface/field-line holdout evidence is supplied. It also requires a passed
replicated nonlinear-window ensemble artifact from
``tools/check_nonlinear_window_ensemble.py`` before any optimized-equilibrium
production nonlinear heat-flux claim can be made. The ensemble requirement is
deliberately separate from the single-window convergence rule: a single
post-transient mean can establish a candidate window, but seed/timestep/restart
replicates are needed before that mean becomes a robust optimization target.

Objective
---------

Let

.. math::

   p =
   \left[
   \Delta \log a,\ \Delta \kappa,\ \epsilon_h,\ \Delta \hat{s}
   \right]

denote the four active max-mode-1 controls used by the examples: a minor-radius
shift, vertical-elongation shift, helical-ripple amplitude, and magnetic-shear
shift. The constrained objective is

.. math::

   J_k(p) =
      w_A \left(\frac{A(p) - A_*}{A_*}\right)^2
      + w_\iota \left(\iota(p) - \iota_*\right)^2
      + w_{QS} R_{QS}(p)^2
      + w_r \|p\|_2^2
      + w_T T_k(p),

with ``A_* = 7`` and ``iota_* = 0.41``. The turbulence term ``T_k`` selects
one of three differentiable objectives.

Growth-rate objective:

.. math::

   T_\gamma(p) = \gamma(p).

Quasilinear heat-flux objective:

.. math::

   T_{QL}(p) =
      C_{sat}\,
      \frac{\gamma_+(p)\, W_i(p)}
           {k_{\perp,\mathrm{eff}}^2(p) + \epsilon},

where ``W_i`` is the linear heat-flux weight and ``gamma_+`` is a smooth
positive growth-rate part. This mirrors the mixing-length-style objective
used in the quasilinear module and is a differentiable optimization proxy, not
a promoted absolute-flux predictor. The current train/holdout quasilinear
calibration pages show why absolute saturated-flux claims remain gated.

Nonlinear-window objective:

.. math::

   \frac{dE}{dt} = 2\gamma(p) E - \alpha(p) E^2,
   \qquad
   Q_i(t; p) = W_i(p) E(t; p),

integrated with a fixed-step RK2 update. The objective is the late-window
average

.. math::

   T_{NL}(p) =
      \frac{1}{t_2 - t_1}
      \int_{t_1}^{t_2} Q_i(t; p)\,dt,

with companion quality metrics

.. math::

   C_V = \frac{\mathrm{std}(Q_i)}{|\langle Q_i \rangle|},
   \qquad
   \mathrm{trend} =
      \frac{|dQ_i/dt| (t_2-t_1)}
           {|\langle Q_i \rangle|}.

The nonlinear objective is intentionally an envelope gate. It is useful for
testing differentiable late-window averaging and optimizer behavior before the
full nonlinear GK RHS is promoted into an end-to-end differentiated objective.

Numerics and Differentiation
----------------------------

The optimizer uses JAX reverse-mode gradients through the scalar objective and
a bounded Adam update with clipped controls. Every shipped artifact records:

- the full objective history;
- the parameter and observable histories;
- an autodiff-vs-central-finite-difference Jacobian report;
- a Gauss-Newton covariance diagnostic from the final weighted objective
  residual Jacobian;
- for the nonlinear-window objective, the initial and optimized heat-flux
  traces, averaging window, coefficient of variation, and trend.

The finite-difference gate is

.. math::

   \max_{ij} |J^{AD}_{ij} - J^{FD}_{ij}| < \epsilon_{abs}
   \quad\mathrm{or}\quad
   \frac{|J^{AD}_{ij} - J^{FD}_{ij}|}{|J^{FD}_{ij}| + \epsilon}
   < \epsilon_{rel},

with tighter tolerances when JAX x64 is enabled.

Small geometry and objective-observable checks should use the shared
``observable_gradient_validation_report`` helper. The helper reports finite
flags, absolute and relative AD/finite-difference errors, tangent-direction
agreement, rank, singular values, condition number, and a pass/fail gate in a
strict JSON-compatible payload. The tiny solver-ready objective gate in
``spectraxgk.solver_objective_gradients`` exercises this path without running
VMEC, Boozer, or a linear eigenproblem; it is a CI and documentation check for
the reporting contract, not a transport-gradient claim.

For the VMEC/Boozer bridge reports, passing this AD/FD tolerance is necessary
but not sufficient for optimization readiness. The geometry sensitivity reports
also carry a ``conditioning`` block with singular values, numerical rank,
condition number, AD row/column norms, per-parameter finite-difference step
scaling, and the worst error location. This keeps three cases separate in the
artifacts: a failed derivative implementation, a correct but ill-conditioned
control direction, and a well-conditioned reduced optimization gate. The
current full-chain ``vmec_jax`` state-coefficient reports should therefore be
read as reduced linear/quasilinear/nonlinear-window estimator differentiability
evidence until converged nonlinear heat-flux gradients or optimized-equilibrium
finite-difference audits also pass.

The UQ diagnostic uses the weighted residual vector whose squared norm is the
reported objective:

.. math::

   r =
   \left[
   \sqrt{w_A}\frac{A-A_*}{A_*},\
   \sqrt{w_\iota}(\iota-\iota_*),\
   \sqrt{w_{QS}}R_{QS},\
   \sqrt{w_r}p,\
   \sqrt{w_T T_k}
   \right].

The local covariance is then

.. math::

   C_p =
   \sigma^2
   \left(J_r^T J_r + \lambda I\right)^{-1},

where ``J_r = dr/dp`` and ``sigma^2`` is estimated from the final residual.
This is intentionally tied to the optimization objective. It is not computed
from the initial-to-final parameter displacement, which would measure optimizer
travel rather than local uncertainty at the optimized point.

Objective-portfolio reducer gate
--------------------------------

Multi-surface, multi-field-line, and multi-``k_y`` stellarator studies should
separate two contracts:

- row production, where VMEC/Boozer/SPECTRAX-GK evaluates one objective vector
  per sample;
- row reduction, where those fixed samples are combined into one scalar for an
  optimizer or UQ ensemble.

The lightweight reducer in
:mod:`spectraxgk.stellarator_objective_portfolio` validates the second contract
without importing optional VMEC or Boozer backends. It requires a real numeric
``(surface, alpha, ky, objective)`` table, finite non-negative normalized
weights, and an explicit reduction policy. The gate below checks the weighted
mean reducer, directional JVP, reverse-mode gradient projection, and central
finite difference on a deterministic nonlinear row fixture.

.. code-block:: bash

   python tools/build_stellarator_objective_portfolio_gate.py \
     --out docs/_static/stellarator_objective_portfolio_gate.png

.. figure:: _static/stellarator_objective_portfolio_gate.png
   :width: 95%
   :align: center
   :alt: Stellarator objective portfolio reducer gate

   Backend-free aggregate-objective reducer gate. It passes AD/JVP/central-FD
   parity for fixed surface/alpha/``k_y`` rows and validates the normalized
   sample/objective weights. This is a required optimization-plumbing contract,
   not a standalone VMEC/Boozer geometry-gradient or nonlinear heat-flux
   optimization claim.

Results
-------

Generate the three individual optimization panels with:

.. code-block:: bash

   JAX_ENABLE_X64=1 python examples/optimization/stellarator_itg_growth_optimization.py --finite-difference-workers 2
   JAX_ENABLE_X64=1 python examples/optimization/stellarator_itg_quasilinear_flux_optimization.py --finite-difference-workers 2
   JAX_ENABLE_X64=1 python examples/optimization/stellarator_itg_nonlinear_heat_flux_optimization.py --finite-difference-workers 2

Generate the comparison panel with:

.. code-block:: bash

   JAX_ENABLE_X64=1 python examples/optimization/compare_stellarator_itg_optimizations.py --workers 3 --finite-difference-workers 2
   JAX_ENABLE_X64=1 python tools/plot_stellarator_optimization_uq.py

The ``--workers`` option parallelizes the independent growth-rate,
quasilinear-flux, and nonlinear-window objective reports while preserving the
serial ordering of the JSON payload. The ``--finite-difference-workers``
option parallelizes central finite-difference columns inside each AD/FD gate
using threads, which avoids pickling JAX objective closures. Both paths record
their worker metadata and identity contract in the JSON artifacts.

.. figure:: _static/stellarator_itg_optimization_comparison.png
   :width: 95%
   :align: center
   :alt: Differentiable QA stellarator ITG optimization comparison

   Three differentiable QA stellarator ITG objectives from the same initial
   control vector. All three keep the final geometry near ``A = 7`` and
   ``iota = 0.41`` while reducing the tracked transport observables. In the
   current artifact, the optimized growth rate is about ``57%`` of the initial
   value and both quasilinear and nonlinear-window heat-flux observables are
   about ``41%`` of their initial values. The comparison is a gradient and
   objective-reduction validation, not a claim that these reduced objectives
   replace converged nonlinear transport simulations.

.. figure:: _static/stellarator_itg_optimization_uq.png
   :width: 95%
   :align: center
   :alt: Stellarator ITG optimization UQ and sensitivity diagnostics

   UQ and sensitivity diagnostics for the same three reduced objectives. The
   first panel verifies AD/FD derivative parity for every active control. The
   covariance panels use the weighted objective residual map above, so the
   reported uncertainty is a local identifiability diagnostic at the optimized
   point. All three reduced objectives remain full-rank and finite-difference
   checked in this artifact; the claim is still limited to optimization
   plumbing, not full VMEC/Boozer/GK nonlinear optimization.

.. figure:: _static/stellarator_itg_growth_optimization.png
   :width: 90%
   :align: center
   :alt: QA stellarator growth-rate optimization

   Growth-rate objective history and coupled transport observables.

.. figure:: _static/stellarator_itg_quasilinear_optimization.png
   :width: 90%
   :align: center
   :alt: QA stellarator quasilinear-flux optimization

   Quasilinear heat-flux objective history. The quasilinear objective uses the
   same differentiable mixing-length feature map tested in
   :doc:`quasilinear`.

.. figure:: _static/stellarator_itg_nonlinear_optimization.png
   :width: 90%
   :align: center
   :alt: QA stellarator nonlinear-window heat-flux optimization

   Nonlinear-window objective history and heat-flux envelope. The shaded region
   is the averaging window used in the objective. The shipped artifact records
   a low coefficient of variation and trend for the optimized late-time window,
   so the plotted average is meaningful for this reduced model.

Connection to Literature
------------------------

The current implementation is closest in spirit to direct microstability
optimization [Jorge24]_: it makes a differentiable linear or quasilinear
transport proxy available inside a stellarator objective. It also follows the
lesson from nonlinear turbulence optimization [Kim24]_: final heat-flux claims
must be audited with nonlinear windows because linear and quasilinear proxies
can fail when nonlinear saturation physics changes.

The next manuscript-level step is therefore not to promote this reduced model
as an absolute flux predictor. The correct next step is to replace the reduced
feature map with a parity-checked in-memory geometry pipeline and then audit
the optimized shapes with converged nonlinear SPECTRAX-GK runs.

Solver-objective Geometry Gradients
-----------------------------------

The first production-adjacent solver-gradient gate now differentiates actual
electrostatic linear-RHS eigenpair observables with respect to solver-ready
geometry arrays. The gate uses the implicit left/right non-Hermitian eigenpair
sensitivity system and compares the result against nearest-branch central
finite differences for ``gamma``, ``omega``, ``<k_perp^2>``, linear
heat/particle-flux weights, and a mixing-length heat-flux proxy. This closes
the ``FluxTubeGeometryData`` contract-level solver-gradient check and the first
full ``vmec_jax`` state-coefficient to ``booz_xform_jax`` to solver
eigenfrequency-gradient gate. The companion QH all-surface artifact closes the
reduced full-chain quasilinear heat-flux-weight gradient gate for the tracked
manuscript fixture. A second Li383 low-resolution holdout now verifies the
same frequency and quasilinear gradient contracts at ``mboz=nboz=21``; the
combined holdout matrix has maximum relative AD/finite-difference mismatch
``4.9e-3`` across the reduced linear/quasilinear objectives. Companion QH and
Li383 reduced nonlinear-window estimator gates now differentiate a smooth
late-window heat-flux envelope through the same VMEC/Boozer state path; the
expanded matrix including those estimator rows has maximum relative mismatch
``2.7e-2``. This closes a multi-equilibrium bounded estimator-gradient check for
nonlinear-window-style reduced objectives, but it does not close converged
nonlinear-window turbulence gradients or broad optimized-equilibrium nonlinear
transport claims.

.. figure:: _static/solver_objective_gradient_gate.png
   :width: 90%
   :align: center
   :alt: Solver-objective geometry-gradient validation gate

   Solver-ready geometry-gradient gate. The left panel compares implicit
   eigenpair sensitivities with central finite differences; the right panel
   shows per-observable relative errors for the two geometry controls.

.. figure:: _static/vmec_boozer_solver_frequency_gradient_gate.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer state-to-solver frequency-gradient validation gate

   Full-chain VMEC/Boozer eigenfrequency-gradient gate. A real ``vmec_jax``
   state coefficient is perturbed, converted through ``booz_xform_jax`` with
   ``mboz=nboz=21``, mapped into the SPECTRAX-GK linear solver, and checked
   against central finite differences.
   The artifact tools also accept explicit VMEC ``radial_index``,
   ``mode_index``, and ``surface_index`` controls so conditioning scans can
   choose physically meaningful state perturbations without changing source
   code.

.. figure:: _static/vmec_boozer_quasilinear_gradient_gate.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer state-to-solver quasilinear-gradient validation gate

   Full-chain VMEC/Boozer quasilinear-gradient gate. The same state
   coefficient is mapped through ``vmec_jax`` and ``booz_xform_jax`` with
   ``mboz=nboz=21`` and a richer ``Nl=2, Nm=3`` SPECTRAX-GK moment basis.
   The implicit left/right eigenpair sensitivity of ``gamma``, ``omega``,
   ``<k_perp^2>``, the electrostatic heat-flux weight, and
   ``gamma Q_i/k_perp^2`` agrees with central finite differences to
   ``4.3e-3`` relative error in the tracked artifact. This closes the
   linear/quasilinear full-chain gradient gate for reduced stellarator
   objectives on the all-surface QH fixture. The optional Boozer surface-stencil
   path is a memory-bounded diagnostic for larger equilibria, not the published
   accuracy gate; QI/QA multi-equilibrium transport-gradient promotion remains
   open until all-surface or otherwise accuracy-equivalent gates pass. This is
   still not a nonlinear-window heat-flux gradient claim.

.. figure:: _static/vmec_boozer_aggregate_objective_gate.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer multi-point aggregate-objective finite-difference gate

   Multi-point VMEC/Boozer aggregate-objective gate. The tracked QH fixture
   evaluates the quasilinear proxy at two resolved ``k_y`` samples using
   ``mboz=nboz=21`` and records the aggregate finite-difference response
   through the same in-memory VMEC/Boozer/SPECTRAX-GK value path. This closes
   the software and artifact path for multi-``k_y`` reduced objectives; it is
   not a nonlinear turbulent heat-flux optimization claim. The tracked
   two-``k_y`` artifact intentionally does not satisfy the held-out
   surface/field-line promotion gate by itself.

.. figure:: _static/vmec_boozer_multi_point_objective_gate.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer multi-alpha aggregate-objective finite-difference gate

   Multi-alpha VMEC/Boozer aggregate-objective gate. The QH fixture repeats the
   same quasilinear finite-difference audit over two field lines
   (``alpha = 0`` and ``0.5``) and two ``k_y`` samples using ``mboz=nboz=21``.
   The tracked artifact has four samples, passes the curvature gate with
   curvature ratio about ``6.9e-3``, and is the current reduced-objective
   evidence for field-line coverage. It still remains a reduced
   linear/quasilinear objective gate, not an optimized-equilibrium nonlinear
   transport claim.

.. figure:: _static/vmec_boozer_aggregate_line_search_gate.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer multi-point aggregate-objective line-search gate

   Multi-point VMEC/Boozer aggregate-objective line-search gate. The tracked
   QH fixture applies one curvature-gated VMEC coefficient update to the
   two-``k_y`` quasilinear proxy aggregate and accepts it only because the
   candidate decreases the objective while the finite-difference gate remains
   conditioned. This is optimizer control-flow evidence for reduced objectives,
   not a nonlinear turbulent transport optimization claim. It must be paired
   with held-out ``surface_index`` or field-line ``alpha`` validation before it
   can support an optimized-equilibrium transport claim.

.. figure:: _static/vmec_boozer_aggregate_line_search_comparison.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer aggregate growth and quasilinear line-search comparison

   Growth-vs-quasilinear aggregate line-search comparison. The growth and
   quasilinear proxy objectives both pass a one-step curvature-gated line
   search on the same QH sample set, but their initial descent directions
   differ. This is important for manuscript claims: optimizing growth rate,
   quasilinear proxy, and nonlinear transport are related but not identical
   objective choices, so each must carry its own validation and holdout gate.

.. figure:: _static/vmec_boozer_aggregate_alpha_holdout_gate.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer aggregate alpha-heldout line-search gate

   Alpha-heldout aggregate line-search gate. The same accepted quasilinear
   update is trained on the ``alpha=0`` QH field line and evaluated on the
   held-out ``alpha=0.5`` field line with the same two ``k_y`` samples. The
   tracked artifact passes, with training relative reduction about ``2.2e-3``
   and held-out relative reduction about ``6.8e-5``. This is useful reduced
   field-line generalization evidence, but it is intentionally blocked from the
   production promotion gate because it is still a reduced linear/quasilinear
   objective split, not a nonlinear transport validation.

.. figure:: _static/vmec_boozer_aggregate_surface_holdout_gate.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer aggregate surface-heldout line-search gate

   Surface-heldout aggregate line-search gate. The QH quasilinear update is
   trained on explicit ``surface_index = 18`` and evaluated on held-out
   ``surface_index = 19`` with the same ``alpha=0`` and two ``k_y`` samples.
   The tracked artifact passes with training relative reduction about
   ``1.31e-3`` and held-out-surface relative reduction about ``4.59e-4``. This
   closes a true reduced surface-generalization check; it still remains a
   reduced linear/quasilinear objective gate rather than a nonlinear transport
   validation.

.. figure:: _static/vmec_boozer_second_equilibrium_aggregate_gate.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer second-equilibrium aggregate-objective gate

   Second-equilibrium aggregate-objective gate. The Li383 fixture passes the
   same mode-21 VMEC/Boozer aggregate finite-difference and one-step
   line-search path with two ``k_y`` samples. The finite-difference curvature
   ratio is about ``3.4e-3`` and the line search reduces the reduced
   quasilinear objective by about ``1.34e-4``. This is second-equilibrium
   optimizer-plumbing evidence, not a calibrated saturated-flux or nonlinear
   transport claim.

.. figure:: _static/vmec_boozer_nonlinear_window_gradient_gate.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer state-to-solver reduced nonlinear-window-gradient validation gate

   Reduced nonlinear-window estimator-gradient gate for the QH fixture. The
   full VMEC/Boozer state-to-solver path produces linear-RHS observables, then
   a smooth RK2 late-window envelope maps those observables to mean heat flux,
   window coefficient of variation, and normalized trend. The plot compares
   implicit eigenpair AD sensitivities against central finite differences. The
   companion Li383 artifact is included in the holdout matrix. These are
   differentiability and conditioning gates for reduced objectives, not
   converged nonlinear-turbulence heat-flux-gradient claims.

.. figure:: _static/vmec_boozer_gradient_holdout_matrix.png
   :width: 100%
   :align: center
   :alt: VMEC/Boozer multi-equilibrium gradient holdout matrix

   Multi-equilibrium VMEC/Boozer gradient holdout matrix. QH and Li383
   frequency, quasilinear, and reduced nonlinear-window estimator gates all
   pass with ``mboz=nboz=21``. The matrix is a reduced differentiability gate;
   converged nonlinear-window turbulence gradients remain a separate promotion
   requirement.

Promotion Gates for Full VMEC/Boozer/GK Optimization
----------------------------------------------------

The full production stellarator optimization claim remains open until all of
the following pass:

1. ``vmec_jax`` state to ``booz_xform_jax`` to ``FluxTubeGeometryData`` works
   in memory without writing intermediate VMEC or EIK files.
   The current bridge already validates the optional ``vmec_jax`` boundary
   derivative, real ``vmec_jax`` metric-tensor derivatives, a real
   non-axisymmetric VMEC field-line tensor derivative through
   ``vmec_jax.geom`` plus ``vmec_jax.vmec_bcovar``, a real
   VMEC tensor-derived flux-tube mapping derivative, a real
   ``booz_xform_jax`` spectral derivative, and a bounded
   Boozer-``|B|``-to-flux-tube mapping derivative. It now also starts from a
   real ``vmec_jax`` ``VMECState``, perturbs VMEC Fourier coefficients,
   converts through ``booz_xform_jax``, and checks SPECTRAX-GK field-line
   geometry-observable derivatives against central finite differences. The
   same artifact now records a direct-VMEC-tensor vs imported-VMEC/EIK
   array-parity audit plus a Boozer equal-arc core audit. The core audit now
   matches the imported ``bmag``, ``bgrad``, ``gradpar``, ``q``, ``s_hat``,
   Jacobian, zero-beta ``gds*``/``grho`` metric convention, and zero-beta
   loaded ``cvdrift``/``gbdrift`` drift convention at release tolerance. The
   remaining gap is finite-beta and broader production-runtime drift parity
   beyond the tracked zero-beta equal-arc fixtures before broad transport-gradient
   claims are promoted.
2. The sampled field-line arrays match the existing imported-VMEC/EIK runtime
   path for at least one small equilibrium.
3. Geometry-observable gradients match central finite differences for the
   in-memory bridge.
4. Linear growth-rate, frequency, and quasilinear-weight gradients through the
   solver-ready geometry contract pass finite-difference or implicit-eigenpair
   checks. This is closed by
   ``docs/_static/solver_objective_gradient_gate.json`` for a small actual
   linear-RHS fixture. The full mode-21 VMEC/Boozer state-to-solver
   eigenfrequency gate is also closed by
   ``docs/_static/vmec_boozer_solver_frequency_gradient_gate.json``. The
   full mode-21 VMEC/Boozer state-to-solver quasilinear heat-flux-weight gate
   is closed by
   ``docs/_static/vmec_boozer_quasilinear_gradient_gate.json`` on the tracked
   all-surface QH fixture. The multi-equilibrium reduced linear/quasilinear
   and nonlinear-window estimator holdout gate is closed by
   ``docs/_static/vmec_boozer_gradient_holdout_matrix.json`` for QH and
   Li383 at ``mboz=nboz=21``. Larger QI/QA nonlinear-window transport holdouts
   are still promotion work: QI is currently conditioning-limited when forced
   through the narrow diagnostic stencil, while the QA low-resolution
   all-surface Boozer transform exceeds the available office GPU memory at
   ``mboz=nboz=21``.
5. Host scalar materialization in production runtime caches is removed or
   isolated so geometry parameters remain traceable.
6. A nonlinear heat-flux objective has a validated adjoint, VJP, or robust
   stochastic/finite-difference estimator with a documented window rule. The
   reduced estimator-gradient gate at
   ``docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json`` and the
   Li383 holdout at
   ``docs/_static/vmec_boozer_li383_nonlinear_window_gradient_gate.json``
   cover the first multi-equilibrium bounded estimator path; production claims
   still need converged nonlinear-window turbulence gradients or robust
   optimized-equilibrium finite-difference audits.
7. Optimized geometries pass multi-field-line, multi-surface, grid/window
   convergence, and nonlinear holdout gates before being used for transport
   claims. The current multi-point VMEC/Boozer aggregate API closes the
   software plumbing for this gate, but the manuscript claim remains bounded
   until the corresponding aggregate artifacts pass on the selected
   equilibria. ``tools/check_vmec_boozer_aggregate_holdout_gate.py`` is the
   artifact-level promotion check for this boundary: aggregate finite-difference
   and line-search artifacts must pass on the same training sample set, and at
   least one independent passed validation artifact must cover a held-out
   ``surface_index`` or field-line ``alpha``. Additional ``k_y`` coverage is
   useful, but ``k_y``-only holdout evidence does not satisfy the
   surface/field-line gate. A passed replicated nonlinear-window ensemble is
   also required before the optimized-equilibrium claim can be promoted from
   reduced-objective evidence to production nonlinear transport evidence. The
   current frozen promotion artifact,
   ``docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json``, is
   blocked as intended: reduced held-out-alpha and held-out-surface artifacts
   now pass, but they are not production nonlinear transport validation
   artifacts, and no replicated nonlinear-window ensemble artifact has been
   supplied.

Until those gates pass, the release claim is: SPECTRAX-GK has a tested
differentiable stellarator ITG objective-reduction workflow and the validation
infrastructure needed to promote that workflow to full VMEC/Boozer/nonlinear
optimization.
