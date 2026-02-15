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

Curvature and diamagnetic drive
-------------------------------

The magnetic drift term is evaluated using the s-alpha curvature/grad-:math:`B`
frequency :math:`\omega_d(\theta)` and an energy operator
:math:`\mathcal{E}[H]`. In the initial Cyclone harness we use a constant-energy
weighting,

.. math::

   \mathcal{E}[H] \approx H,

while retaining optional Hermite-Laguerre ladder operators for higher-order
velocity dependence. The diamagnetic drive is represented as

.. math::

   \omega_*^T \, \mathcal{W}[\phi] = \omega_*^T (1 + \eta_i(E - 3/2)) J_\ell \phi,

with :math:`\eta_i = L_n / L_{Ti}`. The current defaults set the temperature
weighting to zero for stability on coarse moment grids, and the scaling factors
are tuned to match published Cyclone growth rates before full velocity-space
physics is enabled.

Time integration
----------------

The linear system is integrated using explicit fixed-step schemes (Euler, RK2,
RK4) implemented inside a ``jax.lax.scan`` loop. RK4 is used by default in the
Cyclone harness to reduce phase and amplitude errors in the growth-rate fits.

Dealiasing
----------

For nonlinear terms (to be added), we will use the 2/3 de-aliasing rule in
perpendicular Fourier space, consistent with GX and related pseudo-spectral
codes. [GX]_
