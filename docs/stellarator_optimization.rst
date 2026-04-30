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
- a Gauss-Newton covariance diagnostic from the final observable Jacobian;
- for the nonlinear-window objective, the initial and optimized heat-flux
  traces, averaging window, coefficient of variation, and trend.

The finite-difference gate is

.. math::

   \max_{ij} |J^{AD}_{ij} - J^{FD}_{ij}| < \epsilon_{abs}
   \quad\mathrm{or}\quad
   \frac{|J^{AD}_{ij} - J^{FD}_{ij}|}{|J^{FD}_{ij}| + \epsilon}
   < \epsilon_{rel},

with tighter tolerances when JAX x64 is enabled.

Results
-------

Generate the three individual optimization panels with:

.. code-block:: bash

   JAX_ENABLE_X64=1 python examples/optimization/stellarator_itg_growth_optimization.py
   JAX_ENABLE_X64=1 python examples/optimization/stellarator_itg_quasilinear_flux_optimization.py
   JAX_ENABLE_X64=1 python examples/optimization/stellarator_itg_nonlinear_heat_flux_optimization.py

Generate the comparison panel with:

.. code-block:: bash

   JAX_ENABLE_X64=1 python examples/optimization/compare_stellarator_itg_optimizations.py

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
eigenfrequency-gradient gate. It does not yet close the full quasilinear
flux-weight or nonlinear-window state-gradient promotion gate.

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
   objectives; it is still not a nonlinear-window heat-flux gradient claim.

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
   remaining gap is broad finite-beta and multi-equilibrium drift parity before
   transport gradients are promoted.
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
   ``docs/_static/vmec_boozer_quasilinear_gradient_gate.json``.
5. Host scalar materialization in production runtime caches is removed or
   isolated so geometry parameters remain traceable.
6. A nonlinear heat-flux objective has a validated adjoint, VJP, or robust
   stochastic/finite-difference estimator with a documented window rule.
7. Optimized geometries pass multi-field-line, multi-surface, grid/window
   convergence, and nonlinear holdout gates before being used for transport
   claims.

Until those gates pass, the release claim is: SPECTRAX-GK has a tested
differentiable stellarator ITG objective-reduction workflow and the validation
infrastructure needed to promote that workflow to full VMEC/Boozer/nonlinear
optimization.
