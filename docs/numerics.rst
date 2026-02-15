Numerics
========

Spectral discretization
-----------------------

Perpendicular spatial coordinates are discretized with Fourier modes on a
uniform grid in :math:`x` and :math:`y`, while the parallel coordinate is
resolved in real space along the field line. The velocity space uses a
Hermite-Laguerre basis. The resulting data layout for a single species is

``(N_l, N_m, N_y, N_x, N_z)``.

Gyroaverage and polarization
----------------------------

The Laguerre gyroaverage coefficients are

.. math::

   J_\ell(b) = e^{-b/2} L_\ell(b),

with :math:`b = k_\perp^2 \rho^2`. The truncated sum
:math:`\sum_{\ell=0}^{N_\ell-1} J_\ell^2` approaches
:math:`\Gamma_0(b) = e^{-b} I_0(b)` as :math:`N_\ell \to \infty`, which provides a
useful convergence diagnostic.

Parallel streaming
------------------

The streaming operator is applied in real space using a centered periodic
finite-difference in :math:`z` and the Hermite ladder coupling

.. math::

   \mathcal{L}_m[H] = \sqrt{m+1} H_{m+1} + \sqrt{m} H_{m-1}.

Curvature, grad-B, and mirror couplings
---------------------------------------

The magnetic drift terms follow a Laguerre-Hermite stencil: curvature
(``cv``) couples Hermite indices :math:`m\pm 2`, grad-:math:`B` (``gb``) couples
Laguerre indices :math:`\ell\pm 1`, and the mirror term couples :math:`m\pm 1`
and :math:`\ell\pm 1` with a :math:`b^\prime(\theta)` prefactor. These couplings
are applied directly to the gyrokinetic variable :math:`H_{\ell m}` built from
the non-adiabatic moments and the gyroaveraged potential.

For regression tests and reference matching, an ``operator="energy"`` mode is
available that reverts to the energy-weighted drift closure used in earlier
benchmarks. This option preserves historical Cyclone fits while the full
drift/mirror operator is validated across ky and resolution scans.

Normalization control
---------------------

To align with published Cyclone base case data, ``LinearParams`` exposes a
``rho_star`` factor that scales the perpendicular wave numbers used in the
drift and drive terms. This allows fine adjustments of the effective
:math:`k_\perp \rho` without changing the FFT grid spacing.

Diamagnetic drive
-----------------

The diamagnetic drive follows a Laguerre form with explicit gradient
dependence,

.. math::

   \mathcal{D}_{\ell m} \propto
   \begin{cases}
     J_{\ell-1} \ell\, t^\prime
     + J_\ell \left(f^\prime + 2\ell\, t^\prime\right)
     + J_{\ell+1} (\ell+1)\, t^\prime, & m = 0, \\\\
     J_\ell t^\prime / \sqrt{2}, & m = 2, \\\\
     0, & \text{otherwise},
   \end{cases}

with :math:`f^\prime = R/L_n` and :math:`t^\prime = R/L_T`.

Time integration
----------------

The linear system is integrated using explicit fixed-step schemes (Euler, RK2,
RK4) implemented inside a ``jax.lax.scan`` loop. For higher-order Hermite-Laguerre
scans, the ``imex`` and ``implicit`` options provide additional stability by
treating damping terms implicitly. RK4 remains the default for the Cyclone
harness.

Dealiasing
----------

For nonlinear terms (to be added), we will use the 2/3 de-aliasing rule in
perpendicular Fourier space, consistent with standard pseudo-spectral practice.
