Numerics
========

Spectral discretization
-----------------------

Perpendicular spatial coordinates are discretized with Fourier modes on a
uniform grid in :math:`x` and :math:`y`, while the parallel coordinate is
resolved in real space along the field line. The velocity space uses a
Hermite-Laguerre basis. The resulting data layout for a single species is

``(N_l, N_m, N_y, N_x, N_z)``.

Algorithm mapping (numerics → code)
-----------------------------------

The core numerical algorithms and their implementation entry points are:

- **Hermite–Laguerre pseudo-spectral expansion**:
  :mod:`spectraxgk.basis`, :mod:`spectraxgk.gyroaverage`.
- **Gyroaverage / polarization**:
  :func:`spectraxgk.gyroaverage.J_l_all`,
  :func:`spectraxgk.linear.quasineutrality_phi`.
- **Centered periodic derivative in z**:
  :func:`spectraxgk.linear.grad_z_periodic`.
- **Hermite ladder streaming**:
  :func:`spectraxgk.linear.streaming_term`.
- **Curvature / grad-B / mirror couplings**:
  :func:`spectraxgk.linear.linear_rhs_cached`,
  :func:`spectraxgk.geometry.SAlphaGeometry.drift_components`,
  :func:`spectraxgk.geometry.SAlphaGeometry.bgrad`.
- **Diamagnetic drive**:
  :func:`spectraxgk.linear.diamagnetic_drive_coeffs`.
- **Time integration (explicit RK, IMEX)**:
  :func:`spectraxgk.linear.integrate_linear`.
- **Implicit solve (Backward Euler + GMRES)**:
  :func:`spectraxgk.linear.integrate_linear`.

JAX execution model
-------------------

The implementation leverages the following JAX primitives:

- **JIT compilation**: ``jax.jit`` is used in
  :func:`spectraxgk.linear._integrate_linear_cached` to stage time-stepping
  kernels.
- **Loop fusion**: ``jax.lax.scan`` drives the time integration loop.
- **FFT grids**: ``jax.numpy.fft.fftfreq`` is used in
  :func:`spectraxgk.grids.build_spectral_grid`.
- **Sparse Krylov solver**: ``jax.scipy.sparse.linalg.gmres`` is used for the
  implicit linear solve in :func:`spectraxgk.linear.integrate_linear`.
- **Stencil operations**: ``jax.numpy.roll`` and ``jax.numpy.pad`` implement
  the centered ``z`` derivative and Hermite/Laguerre ladder couplings in
  :func:`spectraxgk.linear.grad_z_periodic`,
  :func:`spectraxgk.linear.streaming_term`,
  :func:`spectraxgk.linear.apply_hermite_v`,
  :func:`spectraxgk.linear.apply_laguerre_x`.

These links are clickable in the HTML docs via the ``viewcode`` extension.

Time integration algorithms
---------------------------

The linear solver supports:

- **Forward Euler** (``method="euler"``) and **RK2/RK4** explicit schemes for
  non-stiff runs.
- **IMEX (semi-implicit)** where the collisional/hyper-diffusion terms are
  treated implicitly and the remaining terms explicitly.
- **Backward Euler + GMRES** in ``method="implicit"`` for stiff scans, with a
  diagonal preconditioner that includes damping and drift/mirror diagonals.

These are all implemented in :func:`spectraxgk.linear.integrate_linear` and
share the cached operator data assembled by
:func:`spectraxgk.linear.build_linear_cache`.

Gyroaverage and polarization
----------------------------

The Laguerre gyroaverage coefficients follow the Laguerre–Hermite convention
used in Hermite–Laguerre gyrokinetic moment closures,

.. math::

   J_\ell(b) = \frac{1}{\ell!}\left(-\frac{b}{2}\right)^\ell e^{-b/2},

with :math:`b = k_\perp^2 \rho^2`. This definition is consistent with the
Laguerre projection of the gyroaveraged potential in the Hermite–Laguerre
closure used by the linear operator.

Parallel streaming
------------------

The streaming operator is applied in real space using a spectral periodic
derivative in :math:`z` (FFT-based, via ``jax.numpy.fft``) and the Hermite
ladder coupling

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

Putting the pieces together, the linear operator is assembled from:

- **Streaming**: :math:`v_{th}\,\partial_z` with Hermite ladder couplings.
- **Mirror**: :math:`b'(\theta)` coupling across :math:`(\ell\pm1, m\pm1)`.
- **Curvature drift**: ``cv_d`` coupling across :math:`m\pm2`.
- **Grad-B drift**: ``gb_d`` coupling across :math:`\ell\pm1`.
- **Diamagnetic drive**: :math:`\omega_*` energy-weighted source in ``m=0,2``.

Operator toggles live in :class:`spectraxgk.linear.LinearTerms`, so the same
equation is always solved while individual contributions (streaming, mirror,
curvature, grad-:math:`B`, diamagnetic drive, collisions, hyper-collisions,
end damping, :math:`A_\parallel`, :math:`B_\parallel`) can be switched on or
off for controlled studies.

Field solve and electromagnetic coupling
----------------------------------------

Electrostatic runs solve quasineutrality for :math:`\phi` with optional
Boltzmann response (``tau_e``). Electromagnetic runs solve the coupled
quasineutrality/perpendicular-Ampere system for :math:`(\phi, B_\parallel)` and
then compute :math:`A_\parallel` from parallel Ampere’s law. The implementation
is in :func:`spectraxgk.linear.linear_rhs_cached`.

Normalization control
---------------------

``LinearParams`` exposes a ``rho_star`` factor that scales the perpendicular
wave numbers used in the drift and drive terms. This allows fine adjustments
of the effective :math:`k_\perp \rho` without changing the FFT grid spacing.

Diamagnetic drive
-----------------

The diamagnetic drive is written in the standard energy form,

.. math::

   \mathcal{D}_{\ell m} = i \omega_*\, J_\ell(b)\, \phi
   \left[1 + \eta_i \left(\mathcal{E}_{\ell m} - \frac{3}{2}\right)\right],

where :math:`\omega_* = k_y R/L_n`, :math:`\eta_i = (R/L_T)/(R/L_n)`, and
:math:`\mathcal{E}_{\ell m}` is the Hermite–Laguerre energy operator applied to
the basis. The coefficients are generated by
:func:`spectraxgk.linear.diamagnetic_drive_coeffs`.

Time integration
----------------

The linear system is integrated using explicit fixed-step schemes (Euler, RK2,
RK4) implemented inside a ``jax.lax.scan`` loop. For higher-order Hermite-Laguerre
scans, the ``imex`` and ``implicit`` options provide additional stability by
treating damping terms implicitly. RK4 remains the default for the Cyclone
harness.

Boundary damping
----------------

For field-aligned domains with extended :math:`z` coverage, the linear operator
optionally applies a smooth end-cap damping profile (matching the analytic
linked-boundary taper used in flux-tube calculations). The damping profile is
controlled by:

- ``damp_ends_widthfrac``: fraction of the domain used for the taper.
- ``damp_ends_amp``: damping amplitude applied to :math:`H_{\ell m}`.

The damping is only applied to nonzonal modes (:math:`k_y>0`) and can be
disabled by setting ``damp_ends_amp = 0`` in ``LinearParams``.

Dealiasing
----------

For nonlinear terms (to be added), we will use the 2/3 de-aliasing rule in
perpendicular Fourier space, consistent with standard pseudo-spectral practice.
