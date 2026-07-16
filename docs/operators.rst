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
shared table, an explicit table per species, or an ordered target/source pair
table with shape ``(target, source, kperp, moment, moment)``. Pair tables use
the target species' :math:`k_\perp` field, matching the species-dependent
gyroradius convention, and return the spatial matrix layout consumed directly
by ``apply_multispecies_collision_moment_matrix``. The resulting matrix may
vary at every perpendicular/parallel grid point without leaving traced JAX
execution.
``TabulatedMultispeciesCollisionOperator`` exposes this boundary through the
standard collision protocol. It is a JAX pytree containing the coefficient
grid and fully assembled, collision-frequency-weighted pair table; its
``apply`` method obtains :math:`k_\perp\rho_s=\sqrt{b_s}` from the solver cache,
interpolates each target/source block, and acts on the post-field Hamiltonian.
A constant-table full-RHS gate is identical to
``DriftKineticSugamaOperator``. Generated tables, rather than the runtime
operator, own directed-frequency normalization and coefficient provenance.
Tests require node and endpoint identity, generated-table/direct-equation
identity for both models, species-local spatial application, and JVP/finite-
difference agreement through state amplitude, collision frequency, and an
interior :math:`k_\perp` target. The ordered-pair gate additionally checks JIT
application, pair-block identity, zero-:math:`b` multispecies invariants, and
target-species-leading spatial interpolation. A separate held-out gate constructs matrices
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

The first exact primitive for generating the missing hierarchy is
``bessel_laguerre_kernels(kperp_rho, n_max)`` in ``core.velocity``. It
implements Frei et al. (2021), equation (2.13),

.. math::

   K_n(b) = e^{-b^2/4}\frac{(b^2/4)^n}{n!},

using the stable recurrence :math:`K_{n+1}=K_n b^2/[4(n+1)]`. These
coefficients expand :math:`J_0(b\sqrt{x})` in Laguerre polynomials and decay
factorially, motivating the paper's finite-sum rule :math:`N>b^2/4`. The
validation suite compares them with an independent 96-point Gauss--Laguerre
projection, verifies the reported sub-0.1% tail at :math:`b=1,N=3`, and checks
JIT and JVP/finite-difference agreement. This validates one generator building
block; it does not supply the collision-specific coupling coefficients.

``associated_bessel_laguerre_coefficients`` implements the complete
equation-(2.12) prefactor for arbitrary non-negative Bessel order :math:`m`,

.. math::

   A_n^m(b)=\frac{n!}{(n+m)!}\left(\frac{b}{2}\right)^m K_n(b).

Direct reconstructions of :math:`J_m(b\sqrt{x})` for :math:`m=0,1,2` agree
with an independent special-function implementation over the tested
wavenumber and velocity domain. This closes the Bessel-expansion layer used by
the collision sums, but not the speed-function or test-/field-particle
contractions themselves.

The offline generator also evaluates the Coulomb speed integrals
:math:`e_{ab}^k` and :math:`E_{ab}^k` from Appendix A, equations (A8a)--(A8b),
with 80-digit arithmetic. Orders zero through five at three unequal thermal-
speed ratios agree with direct improper quadrature of their defining
integrals. ``coulomb_speed_moments`` now composes those integrals into the
velocity-integrated test and field functions in equations (A5) and (A13).
Six cases spanning :math:`0.25\leq m_a/m_b\leq4`,
:math:`0.5\leq T_a/T_b\leq3`, and spherical order zero through three agree
with direct three-dimensional Maxwellian quadrature of equations (A2) and
(A10). The equal-species density moment vanishes and the momentum test/field
pieces cancel. These remain generator internals because the complete matrix
contractions that consume them are not yet implemented.

The same generator evaluates equation (3.10)'s monomial coefficients for
:math:`L_j^{p+1/2}(x)`. Independent generalized-Laguerre evaluations verify
tensor orders :math:`p=0,1,4` through polynomial order :math:`j=8`. The next
basis-transform coefficients are cancellation-sensitive. Their provenance is
now closed: the base transform is Appendix A, equation (A4), of
`Jorge, Ricci & Loureiro (2017) <https://arxiv.org/abs/1709.01411>`_, and the
finite-:math:`m` transform and inverse are Appendix B, equations (B5)--(B6), of
`Jorge, Frei & Ricci (2019) <https://arxiv.org/abs/1906.03252>`_. Both formulas
were audited against their defining basis identity rather than accepted as
printed.

The isotropic base transform and inverse are now implemented in the offline
generator from equations (A4) and (A3), respectively. Selected coefficients
agree with independent 80-point Gauss--Hermite/Gauss--Laguerre velocity
projections, including the hand identities :math:`cP_1=H_1/2` and
:math:`c^2P_2=H_2/4+L_1/2`. Forward/inverse products close through total degree
12 with maximum error ``8.73e-15`` even though that shell's condition number is
``1.93e8``. All nested sums remain multiprecision until the final table cast.

The finite-:math:`m` forward transform is now generated as a complete lower-
triangular parity block. Under SciPy's associated-Legendre convention, a
literal equation-(B5) transcription gives half the independently projected
:math:`m=0` coefficients and the opposite sign for odd :math:`m`; the required
factor :math:`2(-1)^m` is fixed independently by the :math:`m=0` endpoint,
eight velocity-space projections, and pointwise reconstruction. Unlike the
isotropic map, every lower reduced-degree shell of the same parity is retained.
Even and odd blocks through reduced degree six reconstruct the physical basis
for :math:`m=0,1,2,3`. Literal equation (B6) fails the finite-:math:`m` inverse
identity. Equation (3.33) of Frei et al. (2021), which includes the weighted
Laguerre-product contraction omitted from that direct normalization, matches
every entry of independently inverted degree-six blocks for
:math:`m=0,1,2,3`. Complete 80-digit block inversion remains the independent
oracle; equation (3.33) supplies the scalar inverse used by collision-matrix
assembly. This closes coefficient generation, not the test-/field-particle
contractions or their transport validation.

The next algebraic layer is also generated and independently checked.
``laguerre_product_expansion_coefficient`` implements both the unweighted
product in equations (3.44)--(3.45) and the :math:`x^m`-weighted product in
equations (3.36)--(3.37) of Frei et al. (2021). Pointwise polynomial
reconstruction covers :math:`m=0,1,2`. Combining that product with the
finite-:math:`m` transform and :math:`K_n(b)` yields
``gyroaveraged_spherical_moment_coefficient``, one coefficient of equation
(3.35). Six coefficients through :math:`m=3` and :math:`b=1.3` agree with
independent Bessel-weighted velocity projection; 20- and 32-term Bessel sums
also agree. This validates the gyro-moment-to-spherical-moment map consumed by
the Coulomb contractions.

The offline generator now contracts equations (3.48)--(3.49) into complete
finite-:math:`b` test- and field-particle matrices with explicit Hermite,
Laguerre, spherical-harmonic, and Bessel truncations. Unlike-species generation
keeps :math:`b_a=k_\perp\rho_a` in the test and outer gyroaverage factors and
:math:`b_b=k_\perp\rho_b` in the field-particle source moments; a regression
holds :math:`b_a` fixed and verifies that only the field block changes with
:math:`b_b`. At :math:`b=0`, every
published nonzero six-moment Coulomb entry is recovered. The larger generated
block is symmetric to ``8.33e-17``, negative semidefinite, and preserves
density, parallel momentum, and thermal energy within ``3.3e-16``. Equation
(3.41)'s :math:`\Pi^{pjm}` agrees with five independent
:math:`J_0J_m`-weighted velocity projections to about :math:`10^{-13}`.
Equation (3.50) remains four separate vectors multiplying
:math:`q_a\phi/T_a` and :math:`q_b\phi/T_b`, because quasineutrality couples
species; like-species test/field polarization cancels. This closes the offline
algebra, not runtime multispecies assembly or transport promotion.

The reproducible algebra/convergence artifact is generated with

.. code-block:: console

   python tools/artifacts/build_linear_validation_artifacts.py collision-verification

.. figure:: _static/collision_operator_verification.png
   :alt: Coulomb collision operator convergence, projection, entropy, and matrix gates
   :width: 100%

   **Offline Coulomb-operator closure.** Panel (a) shows Bessel--Laguerre
   convergence of a finite-:math:`b` polarization coefficient to a 24-term
   reference, Bessel convergence of an assembled 4-by-4 collision block, and
   the independent spherical/radial hierarchy convergence of that block;
   panel (b) compares five generated coefficients with independent
   80-by-80 Gauss--Hermite/Laguerre velocity projection; panel (c) shows five
   dissipative modes and the three density, parallel-momentum, and thermal-
   energy null modes, while its inset verifies leading :math:`O(b^2)` classical
   gyro-diffusion away from the drift-kinetic limit; panel (d) exposes the
   complete retained drift-kinetic moment block. The machine-readable gate is
   stored beside the figure in ``collision_operator_verification.json``.

The largest direct-projection relative error is :math:`3.2\times10^{-13}`;
the published-coefficient, symmetry, and invariant residuals are at or below
:math:`1.2\times10^{-16}`. At :math:`b=0.8`, the assembled 4-by-4 block changes
by :math:`1.6\times10^{-7}` between Bessel orders four and six. The former
default spherical cutoff :math:`(p_{\max},j_{\max})=(3,1)` is rejected because
it differs by 29% from the converged :math:`(9,4)` reference. The :math:`(6,3)`
and :math:`(8,4)` blocks reduce that error to :math:`4.70\times10^{-4}` and
:math:`8.68\times10^{-7}`, respectively. These checks
implement the conservation,
Maxwellian-null, adjointness, and H-theorem requirements emphasized by
`Abel et al. (2008) <https://arxiv.org/abs/0808.1300>`_ and the
finite-:math:`b` moment algebra of
`Frei et al. (2021) <https://arxiv.org/abs/2104.11480>`_. The quadrature check
is a deterministic manufactured-projection test: it verifies the generated
operator against its continuous velocity-space definition, independently of
the symbolic contraction path.

Physical promotion is deliberately stricter. After multispecies
quasineutrality is assembled, the operator must reproduce Spitzer--Härm/
Braginskii transport, the weakly collisional Hermite--Laguerre convergence and
finite-:math:`b` ITG scans, and the separately defined collisionless
Rosenbluth--Hinton residual and Hinton--Rosenbluth collisional damping traces.
The target resolutions, observables, and figure protocols follow Figures
4--9 of Frei et al. (2021) and the conductivity study of
`Frei, Ernst & Ricci (2022) <https://arxiv.org/abs/2202.06293>`_. Until those
runtime gates pass, the panel supports operator algebra and numerical closure,
not a production Landau-transport claim.

The runtime research boundary now mirrors the same decomposition.
``FiniteWavelengthCoulombOperator`` stores test, field, and four polarization
tables with independent target/source :math:`k_\perp\rho` axes. A bilinear JAX
interpolator evaluates those axes at :math:`b_a` and :math:`b_b`; the resolved
kernel applies equations (3.48)--(3.49) to gyrocenter moments :math:`G_a` and
:math:`G_b`, then adds equation (3.50) using the solved potential and distinct
:math:`q_a/T_a` and :math:`q_b/T_b` factors. It intentionally does not apply
the matrices to ``build_H`` because that would double-count the pullback
polarization.

Independent Python pair loops, JIT execution, JVP/finite-difference checks,
like-species polarization cancellation, generated-coefficient application,
and the complete cached linear-RHS seam pass. This closes runtime algebra and
differentiable interpolation. It does not yet close Hermite/Laguerre
truncation or any transport benchmark; therefore this class remains a Python
research API and has no input-file selector.

Finite-:math:`b` conservation must be stated carefully. Collisions are local at
the particle position, whereas the evolved moments are defined at the
gyrocenter. Consequently, the gyrocenter density row is not a null row at
finite :math:`k_\perp\rho`; it represents classical gyro-diffusion. The tracked
verification artifact recovers the density null at :math:`b=0`, obtains a
nonzero finite-:math:`b` row, and measures the expected leading
:math:`O(b^2)` scaling separately for the test, field, and combined rows.
Equation (3.35) maps a gyrocenter distribution to particle moments; it is not
an inverse of the gyrophase average in equation (3.5). Local particle-space
conservation therefore cannot be inferred by applying that moment map to the
already gyroaveraged collision matrix. A future direct particle-coordinate
implementation must test equations (3.2)--(3.4) before gyroaveraging instead.

The multiprecision generator uses exact integer combinatorics for polynomial
binomial factors, memoizes repeated inverse-basis contractions within one
assembly, and retains arbitrary precision where gamma functions and
non-integer coefficients require it. These changes preserve the generated
blocks bit for bit while reducing a representative assembly from 34.1 to 3.46
seconds. Applying equation (3.35)'s :math:`M^{000}` map to the gyroaveraged
collision matrix leaves a nonzero residual, as equation (3.5) predicts; that
quantity is retained only as a rejected diagnostic, not a conservation or
resolution gate.

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
It is useful for reduced-model verification but is not an arbitrary-moment
hierarchy or a finite-:math:`b` multispecies runtime model.

Lowest-order improved-Sugama correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``drift_kinetic_improved_sugama_pair_matrices`` adds the complete low-order
test- and field-particle corrections in Appendix C, equations (101)--(102), to
the original-Sugama ordered pair. The paper labels the driven moment with a
superscript and the response with a subscript, so the published coefficient
array is transposed once into the runtime row/column application convention.
For equal species the test correction vanishes and the field correction
reproduces equations (103a)--(103c) from an independently generated 80-digit
table. Unequal-mass and unequal-temperature pair coefficients are checked
directly, and their assembled matrix conserves particle number, total parallel
momentum, and total thermal energy. At equal species the correction reduces
the heat-flow-block Frobenius distance to the Coulomb matrix from about
``0.521`` to ``0.205``; the equal-temperature multispecies weighted symmetric
operator is non-positive over the complete reduced moment space.

``assemble_drift_kinetic_improved_sugama_matrix`` and
``DriftKineticSugamaOperator.from_improved_species`` expose this equation slice
through the same vectorized JAX and collision-protocol paths. This is a
friction-flow matrix validation, not a parallel-conductivity claim. The
published conductivity comparison retains more moments and reports that the
original operator can underpredict current by at least 10%, while the improved
operator approaches Coulomb within 1%; SPECTRAX-GK therefore keeps
conductivity promotion blocked until the arbitrary-moment correction hierarchy
and its driven steady-state gate are implemented.

A deterministic Cyclone ITG probe also records the finite-wavelength failure
boundary rather than hiding it. At :math:`k_y\rho\simeq0.63`, increasing the
normalized collision weight from zero to three damps the fitted growth rate;
at :math:`k_y\rho\simeq0.94`, the same drift-kinetic model instead excites the
short-wave branch. This is the behavior identified when collisional FLR terms
are omitted in the local collisional-ITG study. The regression test therefore
requires both observations and keeps configuration-file selection fail-closed.
Only a finite-:math:`b` operator may promote the short-wavelength lane.

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
