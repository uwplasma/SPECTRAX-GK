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
paths. Both receive a post-field ``CollisionContext`` so finite-Larmor-radius
models can distinguish the evolved distribution :math:`G` from the
Hamiltonian response :math:`H`:

- ``apply(context)`` for the complete unit-weight RHS,
  including low-rank or dense field-particle terms;
- ``SplitCollisionOperator.split_step(context, dt)`` as an
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
       def apply(self, context):
           return collision_rhs(
               context.distribution,
               context.hamiltonian,
               context.cache,
               context.parameters,
           )

   rhs, fields = nonlinear_rhs_cached(
       state, cache, parameters, terms,
       collision_operator=CollisionModel(),
   )

The callback is evaluated after the field solve and traced by JAX, so its array
operations remain differentiable. ``context.fields`` carries ``phi``, ``apar``,
and ``bpar``; ``context.hamiltonian`` uses the same enabled-field policy as the
gyrokinetic RHS. This avoids silently replacing a finite-:math:`b`
field-particle model by a :math:`G`-only approximation.
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

``multispecies_collision_invariant_rates`` supplies the stricter acceptance
contract for a species-coupled model. For species-normalized coefficients it
returns each particle-density rate and the physically weighted sums

.. math::

   \dot P_\parallel = \sum_s n_s\sqrt{m_s T_s}\,\dot N_s^{10},
   \qquad
   \dot E = \sum_s n_s T_s
   \left(\sqrt{2}\,\dot N_s^{20}+2\dot N_s^{01}\right).

The model is promotable only when every particle rate and both summed rates
vanish to discretization tolerance. This diagnostic is implemented and
autodiff-tested; it does not by itself promote a multispecies collision model.

Reduced drift-kinetic Sugama equation gate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``drift_kinetic_sugama_six_moment_contribution`` and
``drift_kinetic_coulomb_six_moment_contribution`` implement the complete
like-species six-gyromoment matrices reported in Appendix C, equations
(C6a)--(C6f) and (C9a)--(C9f), of the `improved Sugama moment formulation
<https://arxiv.org/abs/2202.06293>`_.  In SPECTRAX-GK ordering, the nontrivial
moment vector is

.. math::

   \boldsymbol{N}=(N^{20},-N^{01},N^{30},-N^{11})^T,

where the signs on Laguerre moments account for the code's polynomial
convention.  The operator is :math:`\nu_s M\boldsymbol{N}` with two symmetric
blocks.  The thermal block is

.. math::

   M_T=\frac{1}{45\sqrt{\pi}}
   \begin{pmatrix}
   -64\sqrt{2} & 64\\
   64 & -32\sqrt{2}
   \end{pmatrix},

and the heat-flux block is

.. math::

   M_q=\frac{1}{525\sqrt{\pi}}
   \begin{pmatrix}
   -1083\sqrt{2} & 624/\sqrt{3}\\
   624/\sqrt{3} & -1187\sqrt{2}
   \end{pmatrix}.

For the exact linearized Coulomb operator, the corresponding blocks are

.. math::

   M_T^L=\frac{1}{15\sqrt{\pi}}
   \begin{pmatrix}
   -16\sqrt{2} & 16\\
   16 & -8\sqrt{2}
   \end{pmatrix},
   \qquad
   M_q^L=\frac{1}{15\sqrt{\pi}}
   \begin{pmatrix}
   -24\sqrt{2} & 24/\sqrt{3}\\
   24/\sqrt{3} & -28\sqrt{2}
   \end{pmatrix}.

Tests evaluate every matrix entry, symmetry, non-positive eigenvalues, the
Maxwellian thermal null direction, density/momentum/energy invariants, and a
collision-frequency JVP against centered finite differences.  This is a real
high-collisionality reduced operator and an equation-level acceptance gate for
the future coefficient generator.  It intentionally returns zero outside the
six-moment projection and is therefore not selected by TOML or presented as a
full Hermite--Laguerre hierarchy, finite-Larmor-radius, multispecies, or
production Sugama/Coulomb implementation.

The same C6/C9 coefficients also exercise the production table boundary. The
maintainer command

.. code-block:: console

   python tools/artifacts/build_linear_validation_artifacts.py collision-table

evaluates the analytic coefficients with 80-decimal-digit ``mpmath``
arithmetic, writes a deterministic ``float64`` array, and records its SHA-256,
mode ordering, polynomial convention, source equations, and claim scope in a
JSON sidecar. ``load_collision_moment_matrix(model)`` verifies the package-data
checksum before returning a host array. ``apply_collision_moment_matrix`` then
packs ``(ell,m)`` into the paper's Hermite-major ordering, applies either one
shared or one matrix per species in JAX, and restores the state layout.
``interpolate_collision_moment_matrix`` provides the finite-:math:`b` runtime
boundary: it linearly interpolates a strictly increasing :math:`k_\perp` table
on device, clamps only outside the generated range, and accepts either one
shared table or an explicit table per species. The resulting matrix may vary at
every perpendicular/parallel grid point without leaving traced JAX execution.
Tests require node and endpoint identity, generated-table/direct-equation
identity for both models, species-local spatial application, and JVP/finite-
difference agreement through state amplitude, collision frequency, and an
interior :math:`k_\perp` target. A separate held-out gate constructs matrices
from the implemented Mandell--Dorland--Landreman finite-Larmor-radius collision
equations, never from the interpolator, and recovers the expected second-order
table-spacing convergence against direct operator evaluations.

This vertical slice establishes the table format, finite-:math:`b` interpolation,
and traced spatial application needed by the full operator. Its current table
contains only the drift-kinetic six-moment matrices, so the advanced-operator
interpolation is tested with controlled coefficient families and the independent
finite-:math:`b` Dougherty-like operator rather than presented as finite-
:math:`b` Sugama/Coulomb physics. Repeating the six-moment matrix at higher
resolution is not valid.
Full tables must populate every retained Hermite--Laguerre coupling from the
published finite-:math:`b`, mass-ratio, and temperature-ratio sums and pass the
stronger gates below.

The lowest-order multispecies drift-kinetic boundary is also implemented
without assuming equal species. For an ordered pair :math:`(a,b)`,
``drift_kinetic_sugama_pair_matrices`` evaluates Appendix C, equations
(C4)--(C5), of Frei, Ernst & Ricci (2022) as

.. math::

   \mathcal C_a = \sum_b \nu_{ab}
   \left(T_{ab}\,N_a + F_{ab}\,N_b\right),
   \qquad
   \sigma=\frac{m_a}{m_b},\quad \tau=\frac{T_a}{T_b}.

Here :math:`T_{ab}` and :math:`F_{ab}` are the published test- and
field-particle matrices on the eight-mode ``Nl=2, Nm=4`` space; coefficients
outside the six active moments are exactly zero. The helper returns matrices
normalized by the directed frequency

.. math::

   \nu_{ab}\propto \frac{n_b}{\sqrt{m_a}\,T_a^{3/2}},

so callers retain explicit ownership of normalization. The separate
``apply_multispecies_collision_moment_matrix`` contract stores target species
first and source species second, applies all source blocks in one JAX
contraction, and supports pointwise spatial matrices. At equal mass and
temperature, :math:`T_{aa}+F_{aa}` reproduces the independent 80-digit C6
table. ``assemble_drift_kinetic_sugama_matrix`` vectorizes all ordered pairs
and adds each test-particle block to its target-species diagonal. An unequal
ion-pair gate checks published coefficients directly; a physical
directed-frequency gate conserves each species' particles and total momentum
and thermal energy, produces a negative weighted quadratic rate, and matches
finite differences through :math:`\sigma` and :math:`\tau`. An independent
matrix-exponential trajectory preserves those invariants through unequal-
species relaxation and reduces the collision residual by more than five
orders of magnitude.

For Python solver experiments, ``DriftKineticSugamaOperator.from_species``
wraps that matrix in the standard collision protocol. Its ``apply`` method
uses ``CollisionContext.hamiltonian`` rather than the evolved distribution,
so the real linear and nonlinear RHS paths supply the post-field
nonadiabatic response. A collision-only two-species linear-RHS gate verifies a
nonzero response and the same physical invariants. The operator is a JAX
pytree, preserving differentiation when species parameters are constructed
inside an objective.

This is the original Sugama model's real low-order drift-kinetic projection.
It is useful for reduced-model verification but is not the improved Sugama
correction, an arbitrary-moment hierarchy, or a finite-:math:`b` multispecies
runtime model.

As a separate full-distribution reference utility,
``conservative_full_f_dougherty_cross_moments``. For directed collision rates
:math:`\nu_{sr}`, it evaluates the pairwise primitive moments

.. math::

   u_{sr} =
   \frac{m_s n_s \nu_{sr} u_s + m_r n_r \nu_{rs} u_r}
        {m_s n_s \nu_{sr} + m_r n_r \nu_{rs}},

.. math::

   (n_s\nu_{sr}+n_r\nu_{rs})m_s v_{t,sr}^2
   =m_s n_s\nu_{sr}v_{t,s}^2+m_r n_r\nu_{rs}v_{t,r}^2
   +\frac{m_s n_s\nu_{sr}m_r n_r\nu_{rs}}
          {m_s n_s\nu_{sr}+m_r n_r\nu_{rs}}
    \frac{(u_s-u_r)^2}{d_v}.

These are equations (2.11)--(2.12) of the
`improved multispecies Dougherty derivation <https://doi.org/10.1017/S0022377822000289>`_.
Francisquez et al. derive a nonlinear full-:math:`f` Fokker--Planck model,
whereas SPECTRAX-GK evolves a linearized delta-:math:`f` gyrokinetic state. The
JAX implementation accepts arbitrary mass ratios and directed rates, keeps
zero-rate and self pairs unchanged, and is equation-gated for pairwise momentum
and energy conservation, the equal-species limit, positive target temperature,
and AD/finite-difference agreement. A three-species, multi-sample gate also
checks every interacting pair for :math:`d_v=1,2,3` and verifies Galilean
invariance of the target flow and thermal speed. It supports derivation checks
and future full-:math:`f` work; it must not be inserted directly into the
shipped linearized field-particle restoration.

The next runtime model tier extends this verified low-order normalization to
the published linearized gyrokinetic Sugama/Coulomb operators in the full
Hermite--Laguerre moment basis. A linearized
multispecies Dougherty variant is admissible only after its delta-:math:`f`
projection is derived explicitly; the full-:math:`f` primitive targets are not
used as a shortcut. Promotion requires discrete Maxwellian
null-space, particle conservation per species, total momentum and energy
conservation, adjointness, non-positive entropy production, velocity-resolution
convergence, collisional ITG, conductivity, and zonal-flow damping gates.

Relevant derivations and verification targets include the
`Laguerre--Hermite pseudo-spectral formulation <https://doi.org/10.1017/S0022377818000041>`_,
the `advanced linearized gyrokinetic moment operators <https://arxiv.org/abs/2104.11480>`_,
the `improved Sugama moment implementation <https://arxiv.org/abs/2202.06293>`_,
and the `local collisional ITG study <https://arxiv.org/abs/2201.02860>`_.
The independent GYACOMO source implementation loads full Sugama/Landau test and
field matrices generated offline by COSOlver, interpolates them in
:math:`k_\perp`, and applies the dense moment coupling at runtime. That audit
supports the same separation here: generate cancellation-sensitive
coefficients in high precision, store provenance and checksums, then keep the
JAX runtime to validated table interpolation and matrix application.

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

The guiding-centre invariant gate is explicitly long-wavelength. At
:math:`k_\perp\rho=0`, a five-step gate starts from populated high moments,
requires a nonzero collision response, and preserves each species' density,
parallel-momentum, and temperature-like moments in both serial and decomposed
integration. At finite :math:`k_\perp\rho`, guiding-centre density, momentum,
and energy are not locally conserved; collisions are local in real space, and
the gyrocentre change is nonlocal. This is physical behavior of the published
model, not a residual to tune away. A direct finite-:math:`b` gate checks every
term of equations (3.38)--(3.42), including parallel/perpendicular flow and
temperature restoration. Promotion to Landau/Sugama remains blocked because
those operators contain different velocity-dependent test-particle and
species-coupled field-particle physics, not because this model should be forced
to conserve finite-:math:`b` guiding-centre moments.

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
