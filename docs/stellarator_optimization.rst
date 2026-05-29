Differentiable Stellarator Optimization
=======================================

Purpose
-------

SPECTRAX-GK now includes a compact, fully JAX-differentiable reduced
stellarator ITG optimization layer. It is designed as the validation gate
before promoting a full ``vmec_jax -> booz_xform_jax -> SPECTRAX-GK``
optimization loop. The current examples optimize a quasi-axisymmetric,
max-mode-1 control vector around a target aspect ratio ``A = 7`` and mean
rotational transform ``iota = 0.41``. They follow the same objective-block
logic used by the local ``vmec_jax`` fixed-boundary QA examples: preserve
aspect, preserve iota, reduce quasisymmetry error, and add a turbulence
objective.

This page is deliberately conservative about claims:

- the reduced examples below are end-to-end differentiable and
  finite-difference checked;
- they validate optimization, sensitivity, covariance, sample-set reduction,
  and plotting machinery;
- the new multi-surface/alpha/``k_y`` portfolio gate is reduced
  model-development evidence for objective plumbing;
- they do **not** yet claim a production VMEC/Boozer/nonlinear gyrokinetic
  stellarator optimization, and production nonlinear heat-flux optimization
  requires long post-transient replicated windows.

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
- Multi-surface/field-line portfolio gate:
  :download:`stellarator_itg_portfolio_gate.py <../examples/optimization/stellarator_itg_portfolio_gate.py>`
- Plotting helper:
  :download:`_stellarator_itg_plotting.py <../examples/optimization/_stellarator_itg_plotting.py>`

The corresponding ``vmec_jax`` workflow that motivated this structure is the
local fixed-boundary QA script
``/Users/rogeriojorge/local/vmec_jax/examples/optimization/QA_optimization.py``.
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

VMEC-JAX Geometry Examples
--------------------------

The user-facing VMEC geometry examples are WOUT-backed runtime workflows. They
use small ``vmec_jax`` input decks shipped under ``examples/vmec`` and avoid
separate EIK generation for the common demo path:

.. code-block:: bash

   pip install vmec-jax
   cd examples/vmec
   ./generate_wouts.sh
   cd ../..

   spectraxgk run --config examples/linear/axisymmetric/runtime_circular_vmec_linear.toml
   spectraxgk run --config examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml
   spectraxgk run --config examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml

Run ``vmec_jax input.NAME`` inside ``examples/vmec`` when only one WOUT is
needed. These bundled QHS/QI/QA decks are self-contained demonstration
equilibria. Machine-specific HSX or W7-X validation should use the same TOMLs
with ``--vmec-file`` pointing to the benchmark WOUT.

This disk-WOUT path is the runtime example path, not the production optimizer
gradient contract. The production optimizer starts from an in-memory solved
``vmec_jax`` state, transforms through ``booz_xform_jax``, and then builds the
SPECTRAX-GK flux tube without relying on intermediate NetCDF files.

Production VMEC-JAX Optimization Plan
-------------------------------------

The production lane starts from the ``vmec_jax`` fixed-boundary QA optimizer:
aspect ratio and mean iota are constrained, the quasisymmetry residual is
penalized, and a SPECTRAX-GK transport objective is added as another residual
block. The default paper-facing seed targets ``A = 7`` and ``iota = 0.41`` at
a fixed ITG flux tube, initially ``torflux = 0.64`` and ``alpha = 0.0``. The
optimized result must also pass held-out field-line and surface gates before
any stellarator-wide claim.

The three reduced optimization examples are:

- ``stellarator_itg_growth_optimization.py``: minimize a smooth reduction of
  the ITG linear growth rate over selected ``k_y``, surface, and ``alpha``
  samples. The gate is branch-continuity plus AD/JVP/finite-difference
  agreement.
- ``stellarator_itg_quasilinear_flux_optimization.py``: minimize the
  electrostatic quasilinear heat-flux diagnostic over the same sample set.
  The output carries saturation-rule metadata and uncertainty intervals; it is
  not an absolute turbulent-flux claim until nonlinear holdouts calibrate it.
- ``stellarator_itg_nonlinear_heat_flux_optimization.py``: generate nonlinear
  candidates using a cheap differentiable surrogate, then promote only if
  matched baseline and optimized equilibria pass replicated long-window
  post-transient heat-flux audits.

For the geometry layer, the user-facing runtime examples use WOUT files
generated from the small ``examples/vmec/input.*`` decks with ``vmec_jax``.
The optimizer path should avoid disk I/O: it should pass a solved
``vmec_jax`` state through ``booz_xform_jax`` with ``mboz >= 21`` and
``nboz >= 21``, then into the SPECTRAX-GK flux-tube contract. Disk WOUTs
remain useful for reproducibility, release artifacts, and external benchmark
comparison.

Every promoted optimization figure needs a sidecar JSON/CSV artifact and a
gate:

- objective history and coefficient trajectory;
- before/after Boozer ``|B|`` contours and quasisymmetry residuals;
- before/after growth-rate spectra;
- before/after quasilinear spectra with uncertainty intervals;
- nonlinear heat-flux time traces with transient cut, running mean, block/SEM
  uncertainty, and replicate spread;
- AD-vs-finite-difference gradient parity and sensitivity/covariance maps;
- Pareto plot of quasisymmetry residual, aspect/iota constraint error, and
  transport reduction.

The nonlinear heat-flux optimizer must not use startup or reduced-window
values as final evidence. Production evidence requires long post-transient
averages whose running means are converged and whose seed/timestep/grid
replicates agree within the documented gate.

Reduced Portfolio Gate
----------------------

Before the VMEC/Boozer optimizer is promoted, the same reducer used by the
future production objective is exercised on a cheap differentiable sample
table. The table is rectangular in normalized toroidal flux, field-line
``alpha``, and ``k_y rho_i``. The default gate covers three surfaces, two
field-line ``alpha`` values, and three ``k_y`` values with growth-rate and
quasilinear-flux columns. It checks both the scalar reduced objective and
every unreduced row against central finite differences.

.. code-block:: bash

   python examples/optimization/stellarator_itg_portfolio_gate.py \
     --finite-difference-workers 2

By default this writes
``docs/_static/stellarator_itg_portfolio_gate.{json,png,pdf}``. Use
``--surfaces``, ``--alphas``, ``--ky-values``, and ``--objectives`` only when
the changed sample set is also recorded in the artifact sidecar. The JSON
sidecar is the audit source: it records ``claim_level``, ``sample_set``,
``backend_boundary``, scalar/row-wise pass status, and the reduced objective
table. The PNG/PDF are human-readable renderings of that sidecar, not separate
validation evidence.

.. figure:: _static/stellarator_itg_portfolio_gate.png
   :alt: Reduced multi-surface field-line ITG objective portfolio gate
   :width: 100%

   Reduced multi-surface/field-line ITG objective portfolio gate. The
   heatmaps show the alpha-averaged growth and quasilinear-flux objective
   rows across three surfaces and three ``k_y`` values. The side panel records
   scalar and row-wise AD/finite-difference agreement, full-rank sensitivity,
   and the explicit claim boundary: this is a reduced portfolio gate, not a
   nonlinear turbulent-transport optimization claim.

The production bridge now uses the same portfolio layout for real
``vmec_jax -> booz_xform_jax -> SPECTRAX-GK`` row production:
``stellarator_itg_vmec_boozer_sample_objective_table_from_state`` evaluates
physical toroidal-flux, field-line ``alpha``, and ``k_y rho_i`` samples, while
``stellarator_itg_vmec_boozer_portfolio_objective_from_state`` applies the
same weighted reducer. The aggregate VMEC/Boozer artifact tool accepts
physical ``--ky-values`` and records the resolved solver indices, ``Ly``,
``Ny``, and per-sample ``ky_abs_error`` in the sidecar. The remaining
promotion step is scientific, not plumbing: rerun held-out surface/field-line
gates and promote nonlinear candidates only after long post-transient
replicated windows pass for matched baseline and optimized equilibria.

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

The corresponding real-artifact guard is
``tools/check_vmec_boozer_reduced_portfolio_guard.py``. It consumes the tracked
multi-alpha VMEC/Boozer aggregate-objective JSON plus a VMEC/Boozer AD/FD
gradient JSON, rebuilds a backend-free reducer table from the real rows, and
fails closed unless the artifact has VMEC/Boozer provenance, at least two
field-line ``alpha`` values, at least two ``k_y`` samples, finite FD and AD/FD
diagnostics, growth and quasilinear objective columns, and an explicit
non-production nonlinear claim boundary.

.. code-block:: bash

   python tools/check_vmec_boozer_reduced_portfolio_guard.py

The tracked guard lives at
``docs/_static/vmec_boozer_reduced_portfolio_guard.json`` and passes on the QH
mode-21 multi-alpha/two-``k_y`` artifact. It admits reduced growth/QL
portfolio plumbing only; production nonlinear turbulent-transport optimization
now additionally requires the separate optimized-equilibrium long-window
transport audit tracked below. That audit is closed for the selected QA
candidate, while nonlinear turbulence gradients and broad multi-surface
optimization remain separate gates.

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

Zonal-flow Objective Contract
-----------------------------

The next stellarator-optimization lane targets geometries with stronger zonal
response before claiming nonlinear turbulence suppression. The backend-free
contract lives in :mod:`spectraxgk.zonal_objective`. It reduces tensors of
``residual_level``, ``damping_rate``, optional ``linear_growth_rate``, and
optional ``recurrence_amplitude`` over a ``(surface, alpha, kx)`` portfolio.
The minimization objective rewards large residual zonal flow through an
``inverse_residual`` column and penalizes damping, growth not screened by the
residual, and late-time recurrence amplitude.

This is deliberately a reduced objective gate. It is appropriate for
``vmec_jax -> booz_xform_jax -> SPECTRAX-GK`` sensitivity analysis once each
row is produced by a validated zonal-response run and the
AD/finite-difference gate passes. It is not, by itself, a turbulence-reduction
claim. A promoted result must still show matched baseline and optimized
long-window nonlinear heat-flux audits, with post-transient running averages
and seed/timestep/grid uncertainty.

The CI-scale gate is:

.. code-block:: bash

   pytest -q tests/test_zonal_objective.py tests/test_build_zonal_flow_objective_gate.py
   python tools/build_zonal_flow_objective_gate.py

The test exercises the optimization contract that the literature motivates:
larger residuals and lower damping lower the scalar objective, the
surface/field-line/wavenumber portfolio shape is explicit, and the resulting
row map passes AD/finite-difference and conditioning checks before optimizer
use.

``tools/build_zonal_flow_objective_gate.py`` is the artifact bridge from
validated zonal-response outputs to optimizer rows.  It currently emits a
W7-X diagnostic artifact from ``w7x_zonal_response_panel.csv`` and
``w7x_zonal_reference_compare.csv``.  Because the frozen W7-X trace still has
open long-window recurrence/damping gates, the artifact is intentionally
marked ``promotion_ready=false`` and ``gate_index_include=false``.  A promoted
QA/QH/Miller-style optimization gate should instead run the same builder with
``--missing-damping-policy=fail`` so absent GAM damping or recurrence metrics
stop the workflow.

.. figure:: _static/zonal_flow_objective_gate.png
   :width: 90%
   :align: center
   :alt: Zonal-flow objective row-production gate

   Zonal-flow objective row-production gate.  The panel shows the row metrics
   consumed by the reduced objective for each W7-X ``k_x``.  Large residuals
   lower the inverse-residual penalty, while large late-window tail ratios
   remain explicit penalties.  The current W7-X artifact is a diagnostic
   bridge, not a promoted optimization claim, because the damping fits are not
   closed under the paper-facing normalization.

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

.. figure:: _static/vmec_boozer_torflux_aggregate_objective_gate.png
   :width: 90%
   :align: center
   :alt: VMEC/Boozer toroidal-flux aggregate-objective finite-difference gate

   Physical-surface VMEC/Boozer aggregate-objective gate. The same QH fixture
   evaluates the quasilinear proxy at two normalized toroidal-flux values
   (``torflux = 0.5`` and ``0.7``) and two physical ``k_y rho_i`` values
   (``0.1`` and ``0.2``). The JSON/CSV sidecars record the requested physical
   ``torflux`` and ``k_y`` values, the resolved solver ``selected_ky_index``,
   and ``ky_abs_error``. The companion
   ``vmec_boozer_torflux_reduced_portfolio_guard.json`` passes with two
   surfaces, one field line, and two ``k_y`` samples; this is surface-axis
   reduced-objective evidence, not nonlinear turbulent-transport optimization
   evidence.

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
   transport claim. The reduced-portfolio guard in
   ``docs/_static/vmec_boozer_reduced_portfolio_guard.json`` now verifies that
   these real rows satisfy the backend-free reducer contract and the
   growth/QL AD/FD provenance boundary.

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
   now pass, but they remain reduced-objective evidence, while the D-shaped and
   circular replicated nonlinear-window ensembles are holdout/calibration
   evidence. The selected optimized QA equilibrium now has its own replicated
   long-window nonlinear audit, which closes the optimized-equilibrium
   post-transient transport-window evidence requirement for this scoped
   candidate.
   ``tools/write_optimized_equilibrium_transport_configs.py`` is the launch
   contract for that final audit. Given a concrete post-optimization
   ``wout*.nc``, it writes the release ``n64`` nonlinear transport replicate
   ladder, including ``t=250,350,450,700`` continuations, two random-seed
   replicates, one timestep replicate, and the exact ensemble/guard commands.
   The current selected candidate has completed that ladder. The generated
   ``t=[350,700]`` ensemble passes finite-flux, running-window, block/SEM,
   replicate-spread, and optimized-equilibrium marker gates, with ensemble mean
   ion heat flux ``10.19``, mean-relative spread ``0.038``, and combined
   SEM/mean ``0.021``.

   Example launch-contract generation:

   .. code-block:: bash

      python tools/write_optimized_equilibrium_transport_configs.py \
        --vmec-file /path/to/wout_optimized_equilibrium.nc \
        --case optimized_equilibrium_post_optimization \
        --out-dir tools_out/optimized_equilibrium_replicates

   The current QA ``vmec_jax`` optimized-equilibrium candidate has also been
   screened through the SPECTRAX-GK linear/quasilinear runtime before launching
   the large nonlinear campaign. On the sampled ITG branch
   ``k_y rho_i = 0.095, 0.190, 0.300, 0.476, 0.667``, all fitted growth rates
   are negative, with the least damped point at ``gamma≈-0.015``. The
   quasilinear mixing-length diagnostic therefore reports zero saturated heat
   flux because stable modes are excluded by the current growth-floor rule. The
   nonlinear audit for this candidate is therefore interpreted as a
   post-transient optimized-equilibrium transport-window check, not as evidence
   that the uncalibrated quasilinear zero-flux estimate predicts an absolute
   saturated flux.

.. figure:: _static/optimized_equilibrium_linear_screen.png
   :width: 90%
   :align: center
   :alt: Optimized QA equilibrium linear and quasilinear screen

   Linear/quasilinear screen for the QA optimized-equilibrium candidate from
   ``vmec_jax``. The sampled ITG branch is linearly damped across the scan, so
   the uncalibrated quasilinear heat-flux estimate is zero under the stable-mode
   exclusion rule. The subsequent nonlinear audit shows finite post-transient
   heat flux, so this panel should be read as a stability/branch screen rather
   than as an absolute-flux prediction.

.. figure:: _static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.png
   :width: 90%
   :align: center
   :alt: Optimized QA equilibrium nonlinear replicate gate

   Optimized-equilibrium nonlinear replicate gate. Two seed replicates and one
   timestep replicate are advanced to ``t≈700`` at ``n64`` and accepted over the
   post-transient window ``t=[350,700]``. The ensemble passes with mean ion heat
   flux ``10.19``, mean-relative spread ``0.038``, and combined SEM/mean
   ``0.021``. This closes the scoped optimized-equilibrium transport-window
   evidence gate; broader nonlinear turbulence-gradient and absolute-flux model
   claims remain separate gates.

.. figure:: _static/qa_no_ess_reference_replicates/qa_no_ess_reference_t700_ensemble_gate.png
   :width: 90%
   :align: center
   :alt: QA no-ESS reference nonlinear replicate gate

   Matched no-ESS reference replicate gate. The valid finite-transform QA
   ``no_ess`` equilibrium from the same ``vmec_jax`` campaign is advanced with
   the same grid, seeds, timestep variant, and post-transient window as the
   selected optimized QA/ESS equilibrium. The reference ensemble passes with
   mean ion heat flux ``12.50``, mean-relative spread ``0.046``, and combined
   SEM/mean ``0.016``.

.. figure:: _static/qa_no_ess_to_optimized_nonlinear_audit.png
   :width: 70%
   :align: center
   :alt: Matched no-ESS to optimized QA/ESS nonlinear audit

   Matched baseline-to-optimized nonlinear audit. Against the validated no-ESS
   reference ensemble, the optimized QA/ESS equilibrium reduces the
   post-transient ion heat flux from ``12.50`` to ``10.19``. The relative
   reduction is ``0.184`` and the difference is separated by ``7.82`` combined
   SEMs. The zero-transform raw ``wout_initial.nc`` from the VMEC optimization
   is intentionally excluded because it cannot define the same finite-twist
   flux-tube baseline.

.. figure:: _static/production_nonlinear_optimization_guard.png
   :width: 90%
   :align: center
   :alt: Production nonlinear turbulent-flux optimization promotion guard

   Production nonlinear turbulent-flux optimization guard. The release-safety
   side passes because startup and reduced nonlinear artifacts are explicitly
   blocked from production promotion and two long post-transient replicated
   holdout ensembles pass. The selected optimized-equilibrium audit also
   satisfies this guard because seed and timestep post-transient windows are
   attached and converged; broader nonlinear transport-optimization claims
   still require separate gates.

The release claim is now: SPECTRAX-GK has a tested differentiable stellarator
ITG objective-reduction workflow, long-window nonlinear holdout evidence, and a
scoped optimized-equilibrium replicated nonlinear transport audit with a matched
finite-transform no-ESS reference comparison. It is still not a universal
absolute-flux quasilinear model, a nonlinear turbulence-gradient optimizer, or a
broad multi-surface stellarator transport-optimization claim.

The next nonlinear turbulence-gradient promotion is now encoded as a
fail-closed evidence gate in
``docs/_static/nonlinear_turbulence_gradient_evidence_gap_report.json``. That
gate requires paired ``baseline``, ``plus_delta``, and ``minus_delta`` nonlinear
campaigns around the same VMEC/profile parameter, the same seed/timestep
replicate set for every parameter state, post-transient heat-flux averages,
passed ensemble uncertainty gates for all three states, and a central
finite-difference audit with bounded response, asymmetry, condition number, and
gradient uncertainty. Existing standalone replicated transport windows remain
necessary evidence but are not sufficient to claim a production nonlinear
turbulence gradient.
The current real boundary-gradient sweep starts from the optimized QA/ESS
equilibrium, re-equilibrates each perturbed VMEC input with ``vmec_jax``, runs
three seed/timestep nonlinear replicates to ``t=900`` for each parameter state,
and analyzes the common ``t=[450,900]`` transport window. The tracked
``ZBS(1,0)`` 5% campaign closes the earlier finite-difference locality blocker:
``fd_asymmetry_rel = 0.274`` and the response fraction is ``0.0685``. It still
fails closed because the propagated gradient uncertainty is ``0.768 > 0.5``.
The companion ``ZBS(1,1)`` 5% campaign gives the complementary negative result:
``gradient_uncertainty_rel = 0.225`` passes, but ``fd_asymmetry_rel = 0.663`` is
still above the locality gate. This is now a robust production-candidate audit
set, not a promoted nonlinear turbulence-gradient validation.
A later seed-5 follow-up of the best ``ZBS(1,0)`` bracket kept the
baseline/plus/minus long-window ensembles valid but did not close the claim:
``gradient_uncertainty_rel`` increased to about ``1.18``, ``fd_asymmetry_rel``
was about ``0.520``, and one matched-seed central finite difference changed
sign. The scientific conclusion is therefore fail-closed: the current
single-control bracket is a useful diagnostic, but it is not efficient to keep
adding replicas at the same amplitude without a new locality/amplitude sweep or
a smoother composite profile-gradient direction.
``docs/_static/nonlinear_turbulence_gradient_candidate_ranking.json`` ranks the
completed ``RBC(1,1)``, ``ZBS(1,1)``, and ``ZBS(1,0)`` attempts. Its current
recommendation is to move to an overdetermined least-squares/profile-gradient
campaign: the best single-control candidates fail in complementary ways, with
``ZBS(1,1)`` statistically clean but nonlocal and ``ZBS(1,0)`` local but too
noisy.
``tools/summarize_nonlinear_gradient_bracket_sweep.py`` is the companion
amplitude-sweep utility for this decision. It consumes completed central-FD
artifacts for one control, plots gradient, response, asymmetry, and uncertainty
against perturbation amplitude, and preserves the same claim boundary: the
sweep can recommend the next campaign, but it does not promote a nonlinear
turbulence-gradient claim unless one input artifact already passes the
production long-window gate. If resolved central finite differences change sign
across nearby amplitudes, the utility recommends a new locality/amplitude sweep
or smoother composite profile-gradient control instead of more replicas at one
amplitude. The tool now also enforces the same-control contract explicitly: a
mixed-control input set is rejected as a candidate-ranking problem, not a
bracket sweep. The tracked ``RBC(1,1)`` 5%/8% sweep is a negative but useful
result: both amplitudes have resolved responses and acceptable-to-marginal
uncertainty, but the finite-difference asymmetry worsens from about ``0.897``
to ``1.895`` as the bracket grows. The recommendation is therefore to shrink
the perturbation or move to a more local/composite profile-gradient control
before spending more nonlinear GPU time.
The concrete overdetermined campaign is tracked in
``docs/_static/qa_ess_overdetermined_nonlinear_gradient_campaign_plan.json``.
It starts from the same optimized-QA/ESS VMEC input, writes matched
``vmec_jax`` perturbation inputs for ``ZBS(1,1)``, ``ZBS(1,0)``, and
``RBC(1,1)`` at 3% relative amplitude, and launches identical
``t=900``, ``n64:64:64:40:40`` nonlinear ladders. That full campaign and the
targeted ``RBC(1,1)`` seed follow-up have now completed: all 33 relevant
runtime outputs pass the output gates, all three ``RBC(1,1)``
baseline/plus/minus five-member replicated ensembles pass, and the central-FD
artifact is local and response-resolved. It remains fail-closed because the
propagated gradient uncertainty is still above the promotion gate:
``gradient_uncertainty_rel = 0.683 > 0.5``. The companion controls fail for
complementary reasons: ``ZBS(1,1)`` passes uncertainty but is nonlocal
(``fd_asymmetry_rel = 0.605``), while ``ZBS(1,0)`` is not response-resolved.
The final status artifact,
``docs/_static/qa_ess_overdetermined_nonlinear_gradient_campaign_status.json``,
therefore reports complete runtime coverage but zero promoted controls. The
post-runtime command
``tools/postprocess_overdetermined_nonlinear_gradient_campaign.py`` is the
reproducible fail-closed path that produced these output, ensemble,
central-FD, ranking, and status artifacts.
The bounded follow-up decision is tracked separately in
``docs/_static/qa_ess_overdetermined_nonlinear_gradient_followup_plan.json``
and can be regenerated with
``tools/plan_nonlinear_gradient_followup.py``. That follow-up recommended only
two new matched nominal-timestep ``RBC(1,1)`` seed replicas per state
(``seed33`` and ``seed34`` for baseline, plus, and minus), because that was the
only completed overdetermined candidate whose response and locality already
passed. Those six office-GPU runs are now folded into the tracked five-member
state ensembles. The result is scientifically useful but negative: extra
replicas lowered the individual state SEMs, but the finite-difference response
remains too uncertain relative to the slope. More blind same-bracket replicas
are not the best next action; the next candidate should use a larger
response-resolved but locality-checked perturbation, variance reduction, or a
better-conditioned composite direction.
``tools/design_nonlinear_gradient_next_campaign.py`` now materializes that
decision into ``docs/_static/nonlinear_gradient_next_campaign_design.json``.
The design gate estimates the bracket scale needed to satisfy propagated
uncertainty, the locality-safe bracket scale implied by the asymmetry gate,
and the number of extra matched replicas needed after applying that locality
cap. For the refreshed ``RBC(1,1)`` evidence, the uncertainty gate would need a
``1.37`` times larger response, but the locality margin allows only about
``1.00`` times the current bracket; even at that cap, the estimated requirement
is six additional replicas per state. The planner therefore recommends a
better-conditioned control or variance-reduced observable instead of more
same-bracket replicas.
Because both single-control amplitude sweeps point away from more blind
replicas, SPECTRAX-GK now also includes
``tools/write_vmec_boundary_profile_perturbation_inputs.py`` for a smoother
multi-coefficient direction. The tracked
``docs/_static/qa_ess_descent_profile_direction_rel2_manifest.json`` uses the
current long-window evidence signs to define a 2% descent-oriented direction:
decrease ``ZBS(1,1)`` while increasing ``ZBS(1,0)`` and ``RBC(1,1)`` with
smaller weights. The finite-difference scalar is the Euclidean norm of the
actual coefficient-change vector, so a successful downstream audit would
measure a directional sensitivity per boundary-coefficient norm. That full
campaign has now been run with real re-equilibrated VMEC files and matched
``t=900``, ``n64:64:64:40:40`` seed/timestep replicates. The output gates pass
for all nine runtime files, and the baseline and minus replicated ensembles
pass, but the plus ensemble fails its spread gate. The resulting central
finite-difference artifact remains fail-closed:
``response_fraction≈0.052`` is resolved, but
``fd_asymmetry_rel≈1.37`` and ``gradient_uncertainty_rel≈1.13`` exceed the
promotion gates. This is a useful negative production-candidate audit, not a
promoted nonlinear turbulence-gradient claim.

.. figure:: _static/qa_ess_zbs10_rel5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.png
   :width: 90%
   :align: center
   :alt: QA/ESS ZBS(1,0) long-window nonlinear turbulence-gradient gate

   QA/ESS ``ZBS(1,0)`` 5% long-window nonlinear turbulence-gradient gate. The left
   panel shows the replicated ``t=[450,900]`` heat-flux means for minus,
   baseline, and plus states; the right panel compares backward, central, and
   forward finite-difference gradients. The artifact is a production-candidate
   long-window campaign, but it remains blocked because the propagated
   gradient uncertainty is still above the release gate.

.. figure:: _static/qa_ess_profile_gradient_rbc_1_1_nonlinear_gradient_rbc_1_1_central_fd_gradient_gate.png
   :width: 90%
   :align: center
   :alt: QA/ESS overdetermined RBC(1,1) nonlinear turbulence-gradient gate

   QA/ESS overdetermined ``RBC(1,1)`` 3% long-window nonlinear-gradient gate.
   This remains the best completed overdetermined candidate after the targeted
   ``seed33``/``seed34`` follow-up: all five-member state ensembles pass and
   the backward/forward finite-difference asymmetry is below the locality gate.
   The artifact still fails closed because propagated uncertainty remains above
   the production threshold, so it supports model-development and
   next-campaign design rather than a promoted nonlinear turbulence-gradient
   claim.

.. figure:: _static/nonlinear_gradient_next_campaign_design.png
   :width: 90%
   :align: center
   :alt: Next nonlinear-gradient campaign design gate

   Next nonlinear-gradient campaign design gate.  The left panel shows the
   current uncertainty and asymmetry margins, the middle panel compares the
   bracket scale required by uncertainty to the locality-safe bracket scale,
   and the right panel estimates extra replicas per state after applying the
   locality cap.  The current conclusion is fail-closed and actionable: do not
   launch more blind ``RBC(1,1)`` replicas; design a better-conditioned
   control, variance-reduced observable, or new checked bracket first.

.. figure:: _static/nonlinear_gradient_composite_control_design.png
   :width: 90%
   :align: center
   :alt: Nonlinear-gradient composite-control admission gate

   Nonlinear-gradient composite-control admission gate.  The left panel shows
   the measured long-window central gradients and the corresponding descent
   signs, the middle panel shows locality and uncertainty admission gates, and
   the right panel shows the VMEC input weights that would be passed to the
   profile-direction writer.  The current artifact intentionally fails closed:
   only ``RBC(1,1)`` is local, resolved, and sign-robust enough to enter the
   composite direction. ``ZBS(1,1)`` is still nonlocal, and ``ZBS(1,0)`` is
   unresolved and nonlocal, so the next production campaign needs another
   local/resolved control or a checked single-control bracket before long
   nonlinear GPU runs.

.. figure:: _static/nonlinear_gradient_ql_seed_screen.png
   :width: 90%
   :align: center
   :alt: Quasilinear-seeded nonlinear-gradient control screen

   Quasilinear-seeded nonlinear-gradient control screen.  This upstream gate
   uses full-chain ``vmec_jax`` state sensitivities to decide which controls
   should even be considered for nonlinear long-window finite differences. The
   current QH/Li383 screen is now launch-ready only for checked short
   nonlinear bracket screens.  The tracked ``Rcos`` and ``Zsin`` controls
   remain fail-closed because their primary quasilinear-proxy signs are not
   robust across the two equilibria, while ``Rsin_mid_surface_m1`` and
   ``Zcos_mid_surface_m1`` are admitted with two-case sign consistency. This is
   still an upstream control-admission result, not a converged nonlinear
   transport-gradient or optimized-equilibrium claim.

.. figure:: _static/qa_ess_descent_profile_rel2_nonlinear_gradient_profile_direction_zbs_1_1_zbs_1_0_rbc_1_1_central_fd_gradient_gate.png
   :width: 90%
   :align: center
   :alt: QA/ESS composite profile-direction nonlinear turbulence-gradient gate

   QA/ESS composite profile-direction nonlinear-gradient audit. The nine
   matched ``t=900`` nonlinear runs all pass runtime-output gates; baseline and
   minus ensembles pass their replicated-window gates, but the plus ensemble
   has too much replicate spread. The finite-difference response is resolved
   but nonlocal and too uncertain, so the result is retained as fail-closed
   evidence for the next campaign design rather than a promoted gradient.

.. figure:: _static/qa_ess_descent_profile_rel2_replicate_spread_diagnostic.png
   :width: 90%
   :align: center
   :alt: QA/ESS composite profile-direction replicate-spread diagnostic

   QA/ESS composite profile-direction replicate-spread diagnostic. The
   baseline and minus states remain within the seed/timestep spread gate, while
   the plus state fails because the high window is the ``seed32`` run and the
   low window is the ``dt0p04`` timestep run. This mixed seed/timestep failure
   is exactly the case where the correct action is not to add blind replicas at
   the same bracket: the next campaign must first separate timestep sensitivity
   from seed sensitivity or shrink the finite-difference bracket.

The targeted follow-up launch artifact for that decision is
``docs/_static/qa_ess_descent_profile_rel2_plus_delta_replicate_followup_plan.json``.
It kept the nonlinear-gradient claim fail-closed and wrote only three
``plus_delta`` cross variants: ``seed22_dt0p05`` tested whether the low window
followed the seed at the nominal timestep, ``seed32_dt0p04`` tested whether the
high window persisted at the refined timestep, and ``seed33_dt0p05`` added one
fresh nominal-timestep seed. Those runs have now completed on the office GPUs.
All six ``plus_delta`` outputs pass the runtime-output gate, but the extended
plus ensemble still fails the spread gate:
``mean_rel_spread = 0.166 > 0.15``. The recomputed central finite-difference
artifact therefore remains blocked by the plus ensemble, by
``fd_asymmetry_rel = 2.84 > 0.5``, and by
``gradient_uncertainty_rel = 1.22 > 0.5``. This closes the blind-replica
question: more of the same plus-state runs are not the right next step. The
next scientifically useful campaign should shrink the finite-difference
bracket or use an overdetermined/profile-gradient design with matched
seed/timestep labels across all parameter states.

.. figure:: _static/qa_ess_descent_profile_rel2_plus_delta_followup_replicate_spread_diagnostic.png
   :width: 90%
   :align: center
   :alt: QA/ESS composite profile-direction targeted plus-state follow-up spread diagnostic

   Targeted ``plus_delta`` follow-up for the QA/ESS composite profile-direction
   nonlinear-gradient audit. The baseline and minus states remain within the
   replicate-spread gate, but the expanded plus-state ensemble still exceeds
   the spread threshold after crossing seed and timestep variants. The shaded
   plus-state panel is the reason the nonlinear turbulence-gradient claim stays
   fail-closed.

.. figure:: _static/qa_ess_descent_profile_rel2_nonlinear_gradient_plus_delta_followup_central_fd_gradient_gate.png
   :width: 90%
   :align: center
   :alt: QA/ESS composite profile-direction targeted follow-up central finite-difference gate

   Central finite-difference audit after the targeted plus-state follow-up. The
   mean response is resolved, but the forward/backward finite differences are
   inconsistent and the propagated uncertainty is too large. This is recorded
   as a bounded negative result, not as nonlinear turbulence-gradient evidence.

.. figure:: _static/qa_ess_rbc11_bracket_sweep.png
   :width: 90%
   :align: center
   :alt: QA/ESS RBC(1,1) nonlinear-gradient perturbation-amplitude sweep

   QA/ESS ``RBC(1,1)`` same-control perturbation-amplitude sweep. The two
   completed long-window central-FD artifacts show that increasing the
   perturbation amplitude does not improve locality: the response remains
   resolved, but the forward/backward asymmetry grows. This keeps the
   single-control ``RBC(1,1)`` path fail-closed and supports the move to a
   smaller locality sweep or an overdetermined profile-gradient campaign.

For boundary-coefficient gradients, first use
``tools/write_vmec_boundary_perturbation_inputs.py``. It starts from a concrete
VMEC input file such as the optimized-equilibrium ``input.final``, writes
matched ``baseline``, ``plus_delta``, and ``minus_delta`` input files for an
explicit ``RBC/RBS/ZBC/ZBS(m,n)`` coefficient, and records the exact
``vmec_jax`` commands plus the downstream nonlinear-gradient campaign command.
The generated files are still launch artifacts, not evidence: production
promotion only begins after ``vmec_jax`` has re-equilibrated all three inputs
and produced distinct ``wout`` files.
Once the three matched ensembles exist,
``tools/build_nonlinear_turbulence_gradient_fd_gate.py`` is the promotion
artifact builder. It consumes the ``baseline``, ``plus_delta``, and
``minus_delta`` replicated ensemble JSON files, computes
``dQ/dp = (Q_+ - Q_-)/(2 delta_p)``, propagates the ensemble SEM into
``gradient_uncertainty_rel``, checks the response fraction, forward/backward
asymmetry, subtraction condition number, and per-state window uncertainty, and
writes JSON/CSV/PNG/PDF sidecars. When the three ensembles contain matching
``seedNN`` or ``dtNN`` replicate labels, the JSON also records
diagnostic-only paired-replicate finite-difference rows. Those rows diagnose
weak or sign-changing stochastic responses; they are not production gates and
do not relax the uncertainty, asymmetry, response, or conditioning thresholds.
The resulting JSON is then supplied to
``tools/check_nonlinear_turbulence_gradient_evidence.py`` together with the
three ensemble artifacts; only that paired long-window workflow can promote a
nonlinear turbulence-gradient claim.
``tools/write_nonlinear_turbulence_gradient_campaign.py`` writes the matching
launch ladder from three explicit VMEC files first: baseline, positive
perturbation, and negative perturbation. Its manifest records the per-state run
manifests, the ensemble-builder commands, the central-FD command, and the final
evidence-check command, so office GPU campaigns and later manuscript artifacts
use one reproducible contract. The writer also performs a fail-closed VMEC
preflight: all three files must exist, must be distinct resolved paths, and must
have distinct SHA256 contents by default. Byte-identical files can only be
accepted with ``--allow-identical-vmec-content`` for plumbing smoke tests, and
that flag is recorded in the manifest as non-production evidence.
