Operators And Terms
===================

This page documents the implemented operator set in SPECTRAX-GK and ties each
term to its runtime parameters and source files.

State And Coupled Variable
--------------------------

For each species :math:`s`, SPECTRAX-GK evolves Laguerre-Hermite moments
:math:`G^{(s)}_{\ell m}(k_x,k_y,z,t)`. The field-coupled variable used by the
linear operator is

.. math::

   H_{\ell m}^{(s)}
   =
   G_{\ell m}^{(s)}
   + \frac{Z_s}{T_s} J_\ell \phi \,\delta_{m0}
   - \frac{Z_s v_{th,s}}{T_s} J_\ell A_\parallel \,\delta_{m1}
   + J_\ell^B B_\parallel \,\delta_{m0},

with :math:`J_\ell^B = J_\ell + J_{\ell-1}`.

In the explicit-time reference-compatible path, streaming is applied to the
field-coupled streamed variable built from the same field terms before the
Hermite ladder is taken.

Source mapping:

- ``src/spectraxgk/linear.py``
- ``src/spectraxgk/terms/fields.py``
- ``src/spectraxgk/terms/assembly.py``

Implemented Linear Operator
---------------------------

The assembled RHS is

.. math::

   \partial_t G
   =
   \mathcal{R}_{stream}
   + \mathcal{R}_{mirror}
   + \mathcal{R}_{curv}
   + \mathcal{R}_{gradB}
   + \mathcal{R}_{dia}
   + \mathcal{R}_{coll}
   + \mathcal{R}_{hyper}
   + \mathcal{R}_{k_\perp\text{-hyper}}
   + \mathcal{R}_{end}.

Every term has a matching multiplicative weight in ``TermConfig`` and
``RuntimeTermsConfig``.

Gyroaverage And Bessel Factors
------------------------------

The Laguerre gyroaverage coefficients are

.. math::

   J_\ell(b) = \frac{1}{\ell!}\left(-\frac{b}{2}\right)^\ell e^{-b/2},
   \qquad
   b = k_\perp^2 \rho_s^2.

Nonlinear electromagnetic terms additionally use :math:`J_0(\alpha)` and
:math:`J_1(\alpha)` on the quadrature grid.

Source mapping:

- ``src/spectraxgk/core/velocity.py``
- ``src/spectraxgk/terms/nonlinear.py``

Streaming
---------

The Hermite ladder streaming term is

.. math::

   \mathcal{R}_{stream}
   =
   -w_{stream}\,k_\parallel v_{th,s}
   \left(\sqrt{m+1}\,X_{\ell,m+1} + \sqrt{m}\,X_{\ell,m-1}\right),

where :math:`X` denotes either :math:`H` or the benchmark-compatible streamed
variable, depending on the solver path.

Controls:

- ``LinearParams.kpar_scale``
- ``RuntimeTermsConfig.streaming``
- boundary/link metadata from the geometry/grid

Mirror
------

The mirror term uses :math:`b'(z)` and couples both Laguerre and Hermite
indices:

.. math::

   \mathcal{R}_{mirror}
   =
   -w_{mirror}\,v_{th,s}\,b'(z)\,
   \Big[
   -\sqrt{m+1}(\ell+1)H_{\ell,m+1}
   -\sqrt{m+1}\ell H_{\ell-1,m+1}
   +\sqrt{m}\ell H_{\ell,m-1}
   +\sqrt{m}(\ell+1)H_{\ell+1,m-1}
   \Big].

Curvature And Grad-B
--------------------

The drift terms are

.. math::

   \mathcal{R}_{curv}
   =
   - i\,w_{curv}\,\tau_z\,\omega_d\,c_v(z)
   \Big[
   \sqrt{(m+1)(m+2)}H_{\ell,m+2}
   + (2m+1)H_{\ell m}
   + \sqrt{m(m-1)}H_{\ell,m-2}
   \Big],

.. math::

   \mathcal{R}_{gradB}
   =
   - i\,w_{gradB}\,\tau_z\,\omega_d\,g_b(z)
   \Big[
   (\ell+1)H_{\ell+1,m}
   + (2\ell+1)H_{\ell m}
   + \ell H_{\ell-1,m}
   \Big].

Controls:

- ``LinearParams.omega_d_scale``
- ``RuntimeTermsConfig.curvature``
- ``RuntimeTermsConfig.gradb``

Diamagnetic Drive
-----------------

The diamagnetic drive acts through density and temperature-gradient couplings
in the low Hermite moments. In code it drives:

- ``m=0`` through density and perpendicular-energy combinations,
- ``m=2`` through temperature-gradient coupling,
- ``m=1`` and ``m=3`` for electromagnetic ``A_parallel`` terms when enabled.

Controls:

- ``LinearParams.omega_star_scale``
- ``LinearParams.R_over_Ln``
- ``LinearParams.R_over_LTi``
- ``RuntimeTermsConfig.diamagnetic``

Collisions
----------

The implemented collisional model is a Lenard-Bernstein-style diagonal damping
plus conservation-restoring low-order moment corrections.

Base damping:

.. math::

   \mathcal{R}_{coll}^{base}
   =
   - w_{coll}\,\nu_s\,\Lambda_{\ell m}\,H_{\ell m},

where ``lb_lam`` is the cached Hermite/Laguerre collision eigenvalue.

The code then reconstructs low moments:

.. math::

   \bar{u}_\perp = \sqrt{b}\sum_\ell J_\ell^B H_{\ell,0},
   \qquad
   \bar{u}_\parallel = \sum_\ell J_\ell H_{\ell,1},

and a temperature-like correction :math:`\bar{T}` from ``m=0`` and ``m=2``.
These are added back only into the ``m=0,1,2`` channels.

Claim boundary and extension plan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a conserving Lenard--Bernstein/Dougherty-like model, not a complete
linearized gyrokinetic Landau operator. The low-order field-particle correction
is important: the operator cannot be represented only by a diagonal damping
array. The implementation contract for collision extensions therefore has two
paths:

- ``apply(state, cache, parameters)`` for the complete unit-weight RHS,
  including low-rank or dense field-particle terms;
- ``SplitCollisionOperator.split_step(state, dt, cache, parameters)`` as an
  optional contract only when the model supplies a mathematically valid exact
  or implicit finite-time update. The runtime does not automatically route this
  method yet. Diagonal hypercollision splitting must not be reused for a
  non-diagonal conserving operator.

The first path is available from Python through
``nonlinear_rhs_cached(..., collision_operator=operator)``. The callback must
return a JAX array with the state shape. SPECTRAX-GK removes the built-in
collision contribution before adding ``terms.collisions * operator.apply(...)``;
hypercollisions remain independent:

.. code-block:: python

   class CollisionModel:
       def apply(self, state, cache, parameters):
           return collision_rhs(state, cache, parameters)

   rhs, fields = nonlinear_rhs_cached(
       state, cache, parameters, terms,
       collision_operator=CollisionModel(),
   )

The callback is traced by JAX, so its array operations remain differentiable.
This is an extension contract, not a claim that a Sugama or full linearized
Coulomb model is already shipped. TOML selection and split integration remain
disabled until an operator passes the conservation and entropy gates below.

The built-in ``collision_split`` policy consequently splits only diagonal
hypercollisions. The conserving collision term remains in the Runge--Kutta or
IMEX RHS, including its field-particle corrections. Earlier implementations
removed the complete collision RHS and advanced only its diagonal part; that
violated the stated conservation model and is no longer supported.

``collision_invariant_rates`` returns the discrete long-wavelength density,
parallel-momentum, and thermal-energy rates of a state-shaped contribution.
``collision_quadratic_rate`` evaluates
:math:`\operatorname{Re}\langle H,C[H]\rangle` with optional species/spatial
weights. Release tests use these functions to verify a local-Maxwellian null
space, all three fluid invariants, and dissipative non-fluid response at
:math:`b=0`.

The next model tier is a species-coupled conserving Dougherty operator. The
research tier after that is the linearized gyrokinetic Sugama/Coulomb operator
in the Hermite--Laguerre moment basis. Promotion requires discrete Maxwellian
null-space, particle conservation per species, total momentum and energy
conservation, adjointness, non-positive entropy production, velocity-resolution
convergence, collisional ITG, conductivity, and zonal-flow damping gates.

Relevant derivations and verification targets include the
`Laguerre--Hermite pseudo-spectral formulation <https://doi.org/10.1017/S0022377818000041>`_,
the `advanced linearized gyrokinetic moment operators <https://arxiv.org/abs/2104.11480>`_,
and the `local collisional ITG study <https://arxiv.org/abs/2201.02860>`_.

Controls:

- ``RuntimePhysicsConfig.collisions``
- ``RuntimeTermsConfig.collisions``
- ``RuntimeSpeciesConfig.nu``
- ``RuntimeCollisionConfig.nu_hermite``
- ``RuntimeCollisionConfig.nu_laguerre``

For two kinetic species, the explicit species-parallel integrator evaluates
this complete collision contribution locally on each device after the shared
field reduction. Nonzero, unequal ion/electron rates are identity-gated against
serial RHS evolution on logical CPUs and two office GPUs. The direct
species-sharded RHS helper remains collision-free; use
``integrate_linear(..., parallel=RuntimeParallelConfig(strategy="velocity",
axis="species", num_devices=2))`` for the validated collisional route.

The conservation claim is explicitly long-wavelength. At
:math:`k_\perp\rho=0`, a five-step gate starts from populated high moments,
requires a nonzero collision response, and preserves each species' density,
parallel-momentum, and temperature-like moments in both serial and decomposed
integration. At finite :math:`k_\perp\rho`, the current finite-Larmor-radius
field-particle correction is not exactly conservative; those residuals remain
a blocker for promoting this baseline to a complete gyrokinetic Landau/Sugama
operator.

The same enclosing route is gated for Hermite/Laguerre hypercollisions with
explicitly populated high moments and nonzero ``nu_hyper_l``/``nu_hyper_m``.
This verifies that the decomposed operator damps its intended high-order
subspace while retaining serial evolution identity; it is separate from the
low-order conserving collision correction above.

Electromagnetic species decomposition reuses the serial field equations rather
than maintaining a second approximation. Local density, parallel-current,
polarization, and perpendicular-pressure moments are summed over the local
species axis and then reduced across the named device axis. The resulting
``phi``, ``apar``, and ``bpar`` are shared by each local RHS assembly. A
two-species gate requires nonzero magnetic fields and serial/decomposed
trajectory identity; it does not yet cover mixed species--Hermite meshes.

Hypercollisions
---------------

SPECTRAX-GK implements three Hermite/Laguerre hypercollision branches and an
optional :math:`|k_z|`-scaled branch:

.. math::

   \mathcal{R}_{hyper}^{const}
   =
   -w_{hyper}
   \Big[
      v_{th,s}\big(\tilde{\nu}_\ell r_\ell + \tilde{\nu}_m r_m\big)
      + \nu_{\ell m} r_{\ell m}
   \Big]G,

.. math::

   \mathcal{R}_{hyper}^{iso}
   =
   -w_{hyper}\,\nu_{hyper}\,r_{hyper}\,G,

.. math::

   \mathcal{R}_{hyper}^{|k_z|}
   \propto
   -w_{hyper}\,\nu_{k_z}\,|k_z|\,m^{p_m}\,G.

Controls:

- ``RuntimePhysicsConfig.hypercollisions``
- ``RuntimeTermsConfig.hypercollisions``
- ``RuntimeCollisionConfig.nu_hyper``
- ``RuntimeCollisionConfig.nu_hyper_l``
- ``RuntimeCollisionConfig.nu_hyper_m``
- ``RuntimeCollisionConfig.nu_hyper_lm``
- ``RuntimeCollisionConfig.p_hyper``
- ``RuntimeCollisionConfig.p_hyper_l``
- ``RuntimeCollisionConfig.p_hyper_m``
- ``RuntimeCollisionConfig.p_hyper_lm``
- ``RuntimeCollisionConfig.hypercollisions_const``
- ``RuntimeCollisionConfig.hypercollisions_kz``

Hyperdiffusion And End Damping
------------------------------

The perpendicular hyperdiffusion term is

.. math::

   \mathcal{R}_{k_\perp\text{-hyper}}
   =
   -w_{hyperdiff}\,D_{hyper}
   \left(\frac{k_\perp^2}{k_{\perp,\max}^2}\right)^{p_{hyper,k_\perp}} G,

masked by the dealias region.

The field-line end damping is

.. math::

   \mathcal{R}_{end} = -w_{end}\,A_{end}\,d(z)\,H.

Controls:

- ``RuntimeTermsConfig.hyperdiffusion``
- ``RuntimeCollisionConfig.D_hyper``
- ``RuntimeCollisionConfig.p_hyper_kperp``
- ``RuntimeCollisionConfig.damp_ends_amp``
- ``RuntimeCollisionConfig.damp_ends_widthfrac``
- ``RuntimeCollisionConfig.damp_ends_scale_by_dt``

Nonlinear :math:`E \\times B` And Flutter
-----------------------------------------

The nonlinear bracket is evaluated pseudospectrally:

.. math::

   \{f,g\} = \partial_x f\,\partial_y g - \partial_y f\,\partial_x g.

The electrostatic nonlinear term is

.. math::

   \mathcal{R}_{NL,E\times B} = -w_{nl}\,\{g,\langle \chi \rangle\},

and the electromagnetic flutter contribution couples adjacent Hermite moments:

.. math::

   \mathcal{R}_{NL,flutter}
   =
   -v_{th,s}
   \left(
   \sqrt{m}\,\{\langle A_\parallel \rangle,g\}_{m-1}
   +
   \sqrt{m+1}\,\{\langle A_\parallel \rangle,g\}_{m+1}
   \right).

Controls:

- ``TimeConfig.compressed_real_fft``
- ``TimeConfig.laguerre_nonlinear_mode``
- ``TimeConfig.nonlinear_dealias``
- ``RuntimeTermsConfig.nonlinear``

Source Mapping
--------------

- linear term kernels:
  ``src/spectraxgk/terms/linear_terms.py``
- nonlinear term kernels:
  ``src/spectraxgk/terms/nonlinear.py``
- assembly:
  ``src/spectraxgk/terms/assembly.py``
- low-level parameter container:
  ``src/spectraxgk/linear.py``
- runtime parameter surface:
  ``src/spectraxgk/workflows/runtime/config.py``

Parameter Surface
-----------------

The primary parameter groups are:

- ``RuntimePhysicsConfig``
- ``RuntimeCollisionConfig``
- ``RuntimeNormalizationConfig``
- ``RuntimeTermsConfig``
- ``LinearParams``

For TOML syntax and all supported keys, see :doc:`inputs`.
