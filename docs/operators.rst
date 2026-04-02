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
GX-compatible streamed variable built from the same field terms before the
Hermite ladder is taken.

Source mapping:

- [/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/linear.py](/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/linear.py)
- [/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/fields.py](/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/fields.py)
- [/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/assembly.py](/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/assembly.py)

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

- [/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/gyroaverage.py](/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/gyroaverage.py)
- [/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/nonlinear.py](/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/nonlinear.py)

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

Controls:

- ``RuntimePhysicsConfig.collisions``
- ``RuntimeTermsConfig.collisions``
- ``RuntimeSpeciesConfig.nu``
- ``RuntimeCollisionConfig.nu_hermite``
- ``RuntimeCollisionConfig.nu_laguerre``

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

- ``TimeConfig.gx_real_fft``
- ``TimeConfig.laguerre_nonlinear_mode``
- ``TimeConfig.nonlinear_dealias``
- ``RuntimeTermsConfig.nonlinear``

Source Mapping
--------------

- linear term kernels:
  [/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/linear_terms.py](/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/linear_terms.py)
- nonlinear term kernels:
  [/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/nonlinear.py](/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/nonlinear.py)
- assembly:
  [/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/assembly.py](/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/terms/assembly.py)
- low-level parameter container:
  [/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/linear.py](/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/linear.py)
- runtime parameter surface:
  [/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/runtime_config.py](/Users/rogeriojorge/local/SPECTRAX-GK/src/spectraxgk/runtime_config.py)

Parameter Surface
-----------------

The primary parameter groups are:

- ``RuntimePhysicsConfig``
- ``RuntimeCollisionConfig``
- ``RuntimeNormalizationConfig``
- ``RuntimeTermsConfig``
- ``LinearParams``

For TOML syntax and all supported keys, see :doc:`inputs`.
