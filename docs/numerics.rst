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
- **GX-style RK4 (CFL adaptive, GX growth-rate diagnostics)**:
  :func:`spectraxgk.gx_integrators.integrate_linear_gx`.
- **Diffrax integration (explicit/implicit/IMEX)**:
  :func:`spectraxgk.diffrax_integrators.integrate_linear_diffrax`,
  :func:`spectraxgk.diffrax_integrators.integrate_nonlinear_diffrax`.
- **Config-driven runner**:
  :func:`spectraxgk.runners.integrate_linear_from_config`.
- **Implicit solve (Backward Euler + GMRES)**:
  :func:`spectraxgk.linear.integrate_linear`.
- **Nonlinear IMEX (implicit linear + explicit nonlinear)**:
  :func:`spectraxgk.nonlinear.integrate_nonlinear`.

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
- **GX-style RK4 with CFL step control**, matching the GX timestep estimator
  (``integrate_linear_gx``). The timestep is recomputed from the linear
  max-frequency estimate using the GX CFL rule, and growth rates are extracted
  from the midplane ``phi`` ratio exactly as in the GX diagnostics kernel.
- **IMEX (semi-implicit)** where the collisional/hyper-diffusion terms are
  treated implicitly and the remaining terms explicitly.
- **Backward Euler + GMRES** in ``method="implicit"`` for stiff scans, with a
  diagonal preconditioner that includes damping and drift/mirror diagonals.
- **IMEX (implicit linear operator + explicit nonlinear term)** in
  ``method="imex"`` for nonlinear runs, using the same GMRES-based linear
  solve and preconditioner.

These are all implemented in :func:`spectraxgk.linear.integrate_linear` and
share the cached operator data assembled by
:func:`spectraxgk.linear.build_linear_cache`.

Diffrax integration
-------------------

Diffrax-backed solvers are available via
:func:`spectraxgk.diffrax_integrators.integrate_linear_diffrax` and
:func:`spectraxgk.diffrax_integrators.integrate_nonlinear_diffrax`. Explicit
solvers (e.g., ``Tsit5``) and implicit/IMEX solvers (e.g., ``KenCarp``) are
supported. Progress reporting is disabled by default; enable it by setting
``TimeConfig.progress_bar=True`` (or ``progress_bar=True`` in the integrator
call). Diffrax currently emits a warning when evolving complex-valued states;
the solvers still run, but treat this as experimental behavior.

Use :class:`spectraxgk.config.TimeConfig` and
:func:`spectraxgk.runners.integrate_linear_from_config` to select diffrax
integration from input configuration without changing call sites. By default,
``TimeConfig`` enables diffrax with a fixed-step Dopri8 solver; set
``use_diffrax=False`` to force the built-in fixed-step integrators.

For the ETG/KBM baseline cases, the default configurations switch to
adaptive Tsit5 with ``diffrax_rtol=1e-4``, ``diffrax_atol=1e-7``, and
``diffrax_max_steps=20000`` to avoid fixed-step instabilities. These defaults
can be overridden on a per-run basis via the ``TimeConfig`` fields.

Performance tuning
------------------

SPECTRAX-GK includes several performance-oriented options that preserve
end-to-end JAX differentiability:

- **Streaming growth-rate fits**: use
  :func:`spectraxgk.diffrax_integrators.integrate_linear_diffrax_streaming`
  to compute ``(gamma, omega)`` online without storing time series. This reduces
  memory pressure during long scans. The streaming fit supports ``phi`` or
  density moments via ``fit_signal`` and uses a fixed ``tmin/tmax`` window.
- **Batched ky scans**: pass ``ky_batch>1`` to the benchmark scan helpers to
  integrate multiple ky values at once using a sliced ky grid. Set
  ``fixed_batch_shape=True`` (default) to edge-pad the final batch and avoid
  recompilation on short tail batches.
- **Donation and sharded buffers**: time integrators donate state buffers in
  JIT-compiled paths to reduce allocations. The diffrax integrators accept a
  ``state_sharding`` argument if you want to preserve explicit JAX sharding on
  the state array.
- **Implicit preconditioning hooks**: ``implicit_preconditioner`` accepts
  ``"auto"/"diag"/"physics"/"block"`` (full diagonal preconditioner),
  ``"damping"`` (collisional/hyper-only), ``"pas"`` (PAS line preconditioner),
  ``"pas-coarse"`` (line + coarse correction in kx/linked-kx chains),
  ``"hermite-line"``
  (Hermite streaming line solve in ``m`` at fixed :math:`k_z`), or
  ``"hermite-line-coarse"`` (Hermite line solve + kx-coarse correction), or
  ``"identity"`` to disable preconditioning.
- **Shift-invert preconditioning hooks**: the shift-invert Krylov solver uses
  GMRES solves for ``(A - \sigma I)^{-1}``. Configure
  ``KrylovConfig.shift_preconditioner`` to accelerate these solves with
  ``"damping"`` (element-wise inverse of the collisional/hyper damping) or
  ``"hermite-line"`` (Hermite streaming line solve via FFT in ``z`` and a
  tridiagonal solve in ``m``). The ``"-coarse"`` variants add a lightweight
  coarse correction in the kx direction (for linked boundaries this averages
  within linked chains; for periodic boundaries this reduces to a kx-mean).
- **Targeted shift-invert mode selection**: set ``KrylovConfig.mode_family``
  (for example ``"cyclone"``, ``"etg"``, ``"kbm"``) and
  ``KrylovConfig.shift_selection`` to stabilize branch selection in stiff
  spectra. ``KrylovConfig.fallback_method`` controls the automatic fallback
  policy when shift-invert returns a non-finite or strongly damped mode.
- **Reusable IMEX operators**: nonlinear IMEX runs can prebuild and reuse the
  matrix-free linear operator with
  :func:`spectraxgk.nonlinear.build_nonlinear_imex_operator` and pass it to
  :func:`spectraxgk.nonlinear.integrate_nonlinear_imex_cached` via
  ``implicit_operator``.
- **Cached hypercollision factors**: the linear cache now stores the Hermite–
  Laguerre hypercollision ratios and masks to avoid repeated power operations
  inside the RHS assembly.

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

In the GX-aligned formulation we apply the parallel derivative to the
non-adiabatic moments plus explicit field terms before the Hermite ladder is
applied. In other words, the streamed quantity is

.. math::

   \tilde{G}_{\ell m} = G_{\ell m}
   + \frac{Z_s}{T_s} J_\ell \phi\,\delta_{m0}
   - \frac{Z_s v_{th}}{T_s} J_\ell A_\parallel\,\delta_{m1}
   + J_\ell^B B_\parallel\,\delta_{m0},

so that the GX-style streaming term uses :math:`\partial_z \tilde{G}` instead of
the full :math:`H_{\ell m}` derivative. This matches the ordering and ghost
exchange used by GX’s ``grad_parallel_linked`` operator.

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

For the nonlinear generalization, SPECTRAX-GK also exposes a term-wise
assembly interface in :mod:`spectraxgk.terms`. The
:class:`spectraxgk.terms.TermConfig` toggles the same operator components,
and :func:`spectraxgk.terms.assemble_rhs` builds the RHS from per-term
functions. This keeps term implementations isolated for easier extension,
while preserving JAX differentiability and performance.

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
