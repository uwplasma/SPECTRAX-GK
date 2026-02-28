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

For scan workloads, the default path is custom fixed-step ``imex2`` with
``TimeConfig.use_diffrax=False``. This keeps stepping shape-stable and improves
throughput for multi-ky scans. Diffrax adaptive stepping remains available as
an optional mode through ``TimeConfig.use_diffrax=True``.

Nonlinear FFT bracket
---------------------

The nonlinear :math:`E\times B` term is evaluated pseudospectrally using
FFT-based derivatives in the perpendicular plane. By default SPECTRAX-GK uses
the GX-style real FFT path (``TimeConfig.gx_real_fft = true``), which computes
gradients from the Nyquist-compressed (``N_y/2+1``) spectrum and expands the
result back to full :math:`k_y`. This exactly matches the GX nonlinear bracket
normalization and minimizes memory traffic. Set ``gx_real_fft = false`` to use
the full complex FFT bracket instead.

For electromagnetic nonlinear runs, SPECTRAX-GK stacks the gyro-averaged
potentials ``J0*phi``, ``J0*apar``, and the ``bpar`` correction into a single
FFT batch. This collapses multiple rFFT/iFFT passes into one pipeline per
step and reuses the same real-space gradients for all channels.
Laguerre/Bessel factors on the GX quadrature grid (``J0`` and ``J1/alpha``) are
precomputed once per grid and cached in the linear operator, so the nonlinear
kernel only applies them via inexpensive elementwise multiplies.
For nonlinear runs that do not require the GX quadrature grid, set
``TimeConfig.laguerre_nonlinear_mode="spectral"`` to skip the Laguerre
quadrature transform and instead use the spectral gyroaverage factors ``Jl``
directly. The default ``"grid"`` mode matches GX and applies the quadrature
transform.

De-aliasing and hyperdiffusion
------------------------------

Nonlinear brackets are filtered using the standard ``2/3`` de-alias mask. The
mask lives on the spectral grid and is applied after each bracket evaluation.
Additional numerical stabilization is provided by hyperdiffusion in
:math:`k_\perp` (``TermConfig.hyperdiffusion`` / ``D_hyper`` settings), which
acts as a scale-selective damping term and is treated implicitly in IMEX
schemes.

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
- **Stacked FFT channels**: nonlinear brackets batch ``phi/apar/bpar`` into a
  single FFT pipeline so the spatial derivatives are computed once and reused
  across fields. This removes redundant transforms and reduces FFT calls.
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

Automatic solver + fit-signal selection
---------------------------------------

For newcomer-friendly runs, the benchmark and runtime drivers accept
``solver="auto"`` and ``fit_signal="auto"``. The auto solver tries the
preferred path for the case (time integration for ion-scale Cyclone/KBM
benchmarks, Krylov for ETG) and falls back to the alternative if the returned
``(gamma, omega)`` is non-finite or violates ``require_positive``. The auto
fit-signal choice computes both ``phi`` and density moment time traces (when
available), scores each using the same windowing rules (``R^2`` of log-amplitude
and phase fits plus an optional growth-rate weight), and selects the higher
score. To make this decision robust, auto mode disables streaming fits and
stores the minimal time traces needed for the comparison.

Advanced users can override these defaults in TOML or Python drivers by setting
``solver="time"``/``"krylov"`` and ``fit_signal="phi"``/``"density"`` together
with custom fit-window parameters.
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

Operator toggles start from :class:`spectraxgk.linear.LinearTerms` and are
converted into one canonical :class:`spectraxgk.terms.TermConfig` through
:func:`spectraxgk.linear.linear_terms_to_term_config`. The same modular RHS
path is then used by fixed-step linear integrators, diffrax integrators,
Krylov operator applications, and nonlinear IMEX linear solves.

The RHS is assembled in :mod:`spectraxgk.terms` via
:func:`spectraxgk.terms.assemble_rhs_cached`, which sums per-term kernels
(streaming, mirror, drifts, diamagnetic drive, collisions, hyper-collisions,
and end damping). This keeps the physics core branch-free and easier to extend,
while preserving JAX differentiability and performance.

Field solve and electromagnetic coupling
----------------------------------------

Electrostatic runs solve quasineutrality for :math:`\phi` with optional
Boltzmann response (``tau_e``). Electromagnetic runs solve the coupled
quasineutrality/perpendicular-Ampere system for :math:`(\phi, B_\parallel)` and
then compute :math:`A_\parallel` from parallel Ampere’s law. The implementation
is in :mod:`spectraxgk.terms.fields` and is called from
:func:`spectraxgk.terms.assemble_rhs_cached`.

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

Nonlinear E×B terms use the 2/3 de-aliasing rule in perpendicular Fourier space,
consistent with standard pseudo-spectral practice. The current implementation
applies the mask before and after the real-space bracket evaluation.

Nonlinear Electromagnetic Terms
-------------------------------

The nonlinear kernel evaluates gyro-averaged Poisson brackets in spectral space
and converts to real space only for the perpendicular derivatives. The E×B term
advects each Hermite–Laguerre moment with a gyro-averaged potential
:math:`\chi = J_0 \phi + J_1 b_\parallel` (implemented via :math:`J_l` and
:math:`J_l^B` in the Laguerre basis). The electromagnetic flutter contribution
uses :math:`\{g_m, J_0 A_\parallel\}` and couples adjacent Hermite moments with the
standard ladder factors, matching the GX nonlinear formulation. See
:cite:`FC82,AL80,GX` for the governing electromagnetic gyrokinetic equations and
GX's implementation details.
