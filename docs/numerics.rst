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

Time integration
----------------

We currently integrate the linear system using a forward Euler step wrapped in a
``jax.lax.scan`` loop, which is fully JIT-compilable and differentiable. Higher
order schemes (RK2/RK4) will be introduced after the curvature and gradient
terms are added.

Dealiasing
----------

For nonlinear terms (to be added), we will use the 2/3 de-aliasing rule in
perpendicular Fourier space, consistent with GX and related pseudo-spectral
codes. [GX]_
