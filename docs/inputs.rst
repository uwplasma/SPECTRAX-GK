Input Files and CLI
===================

SPECTRAX-GK supports lightweight TOML inputs that map directly onto the
``GridConfig``, ``TimeConfig``, ``GeometryConfig``, and ``ModelConfig`` dataclasses.
You can use these inputs from the CLI or from a Python driver.

Unified Runtime Schema
----------------------

In addition to benchmark-case TOMLs, SPECTRAX-GK supports a **case-agnostic**
runtime schema (``RuntimeConfig``) with explicit species and physics toggles.
This allows Cyclone/ETG/KBM to run through the same solver path without
changing solver internals.

Minimal runtime TOML example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: toml

   [[species]]
   name = "ion"
   charge = 1.0
   mass = 1.0
   density = 1.0
   temperature = 1.0
   tprim = 2.49
   fprim = 0.8
   kinetic = true

   [physics]
   linear = true
   nonlinear = false
   electrostatic = true
   electromagnetic = false
   adiabatic_electrons = true
   tau_e = 1.0

   [normalization]
   contract = "cyclone"
   diagnostic_norm = "gx"

   [terms]
   streaming = 1.0
   mirror = 1.0
   curvature = 1.0
   gradb = 1.0
   diamagnetic = 1.0
   collisions = 1.0
   hypercollisions = 1.0
   end_damping = 1.0
   apar = 0.0
   bpar = 0.0
   nonlinear = 0.0

   [run]
   ky = 0.3
   Nl = 24
   Nm = 12
   solver = "auto"
   fit_signal = "auto"

Minimal TOML example
--------------------

.. code-block:: toml

   case = "cyclone"
   gx_reference = true

   [grid]
   Nx = 1
   Ny = 24
   Nz = 96
   Lx = 62.8
   Ly = 62.8
   boundary = "linked"
   y0 = 20.0
   ntheta = 32
   nperiod = 2

   [time]
   t_max = 10.0
   dt = 0.002
   use_diffrax = true
   diffrax_solver = "Dopri8"
   diffrax_adaptive = true
   diffrax_rtol = 1.0e-6
   diffrax_atol = 1.0e-8
   gx_real_fft = true

   [run]
   ky = 0.3
   Nl = 24
   Nm = 12
   solver = "auto"
   method = "imex2"

   [fit]
   auto_window = true
   window_method = "loglinear"
   fit_signal = "auto"

The ``[time]`` section also accepts ``gx_real_fft`` (default ``true``) to
select the compressed real-FFT nonlinear bracket. Set ``gx_real_fft = false`` to
use a full complex FFT for the nonlinear term. Diagnostics output can be
decimated with ``sample_stride`` (record every ``N`` steps) and
``diagnostics_stride`` (compute streaming diagnostics every ``N`` steps). Set
``diagnostics = false`` in ``[time]`` (or ``--no-diagnostics`` on the CLI) to
disable diagnostics entirely for speed. For CFL-controlled timestep control, use
``fixed_dt = false`` along with ``cfl`` and optional ``cfl_fac`` /
``dt_min`` / ``dt_max`` limits. When ``cfl_fac`` is omitted, SPECTRAX uses
the GX method default instead of a universal constant:
``rk3``/``sspx3`` use ``1.73``, ``rk4`` uses ``2.82``, and other methods keep
``1.0``. When adaptive timestepping is enabled, diagnostics include
``dt_t`` (per-sample timestep history) and ``dt_mean`` (average effective dt)
to quantify CFL-driven savings. In reference-compatible nonlinear runs the adaptive
``dt`` estimate combines the GX linear frequency cap with the instantaneous
nonlinear cap, matching GX's CFL update instead of using the nonlinear
bracket alone. To control the Laguerre handling in nonlinear
brackets, set ``laguerre_nonlinear_mode = "grid"`` (reference quadrature,
default) or ``laguerre_nonlinear_mode = "spectral"`` (use spectral ``Jl``
without the quadrature transform).
Use ``nonlinear_dealias = false`` to disable nonlinear dealias masking for
reference/debug runs where you want to preserve all configured base modes.
When ``nonlinear_dealias = true``, nonlinear runtime mode selection is
dealias-aware: if the requested ``ky`` is filtered out by the 2/3 mask, the
runner automatically picks the nearest retained ``ky``. The CLI prints the
effective ``ky_sel``/``kx_sel`` used by diagnostics.
For GX-reference runs, leaving ``dt_max`` unset uses GX's default behavior
(``dt_max = dt``). Increase ``dt_max`` explicitly only when you intentionally
trade strict GX matching for throughput.

Nonlinear collision/hypercollision splitting is enabled with
``collision_split = true``. The ``collision_scheme`` key selects the update:
``implicit`` (backward-Euler), ``exp`` (exact diagonal exponential), and
``sts``/``rkc`` aliases (treated as stabilized explicit/exponential updates for
diagonal operators).

The ``[geometry]`` section supports ``drift_scale`` to switch between reference-compatible
(``drift_scale = 1.0``) and GS2-style (``drift_scale = 2.0``) drift
normalizations. The default configuration in SPECTRAX-GK uses the GX-reference
value.
For GX slab benchmarks, set ``model = "slab"``. Optional slab-specific keys are
``z0`` (sets ``gradpar = 1/z0`` when positive, matching GX's slab domain
normalization) and ``zero_shat = true`` (forces the GX zero-shear slab metric
``gds2 = 1, gds21 = 0, gds22 = 1``).
It also accepts ``model = "gx-netcdf"`` with
``geometry_file = "/path/to/gx_geometry.nc"`` to run from imported sampled
field-line geometry instead of the analytic ``s-alpha`` model. The imported
file can be a GX output ``*.out.nc`` or a root-level GX ``*.eik.nc`` geometry
file produced by the VMEC workflow. When that imported geometry is used with a
linked boundary, SPECTRAX-GK now follows the file's own ``theta`` range,
``jtwist/x0`` geometry factor, and ``kxfac`` metadata instead of forcing the
analytic s-alpha grid defaults.
For direct VMEC workflows, the runtime also accepts ``model = "vmec"``.
In that mode SPECTRAX-GK calls GX's ``gx_geo_vmec.py`` helper to generate a
matching ``*.eik.nc`` file on demand, then immediately reuses the same imported
geometry path as the W7-X examples. Set ``vmec_file`` plus the flux-tube keys
``torflux``, ``npol`` and optionally ``alpha``. ``geometry_file`` can be used
as an explicit output path for the generated ``*.eik.nc`` file, and
``gx_repo`` can point to a non-default GX checkout if needed. If GX's VMEC
helper must run under a different Python interpreter than SPECTRAX itself
(for example when ``booz_xform`` is installed in a separate environment), set
``gx_python`` or the ``GX_VMEC_PYTHON`` environment variable. This is now the
recommended parity-first route for new stellarator cases such as HSX.
``vmec_file`` supports ``$ENV_VAR`` expansion, and relative paths are resolved
against ``gx_repo`` before falling back to the current working directory. The
W7-X runtime TOML uses that contract so the same config works on both local
and office-style GX checkouts.
When ``geometry_file`` is set for ``model = "vmec"``, SPECTRAX regenerates
that target instead of reusing a stale file from an older VMEC conversion.
For VMEC ``fix aspect`` runs, SPECTRAX now follows GX's default helper
contract and does not inject ``x0`` from the runtime ``Lx``. That keeps the
generated ``*.eik.nc`` file aligned with GX's own W7-X/HSX geometry output.

For Miller tokamak workflows, the runtime also accepts ``model = "miller"``.
In that mode SPECTRAX-GK calls GX's ``geometry_modules/miller/gx_geo.py``
helper to generate a matching root-level ``*.eiknc.nc`` file, then immediately
re-enters the same imported geometry path used for VMEC ``eik.nc`` files.
Set the Miller inputs directly in ``[geometry]``:
``rhoc``, ``q``, ``s_hat``, ``R0``, optional ``R_geo``, ``shift``,
``akappa``, ``akappri``, ``tri``, ``tripri``, and ``betaprim``.
``geometry_file`` can be used as an explicit output path for the generated
Miller ``*.eiknc.nc`` file, and ``gx_python`` applies here as well when the GX
helper must run in a different Python environment.

Solver and fit-signal keys
--------------------------

The ``[run]`` and ``[scan]`` sections accept ``solver`` and ``fit_signal`` keys:

* ``solver = "auto"`` (default): choose time vs Krylov and fall back if needed
* ``solver = "time"``: always use time integration
* ``solver = "krylov"``: always use the matrix-free eigen solver

* ``fit_signal = "auto"`` (default): pick ``phi`` vs density based on fit quality
* ``fit_signal = "phi"``: use the electrostatic potential time trace
* ``fit_signal = "density"``: use the density moment time trace

For large ky scans, ``scan-runtime-linear --batch-ky`` integrates all ky values
in a single time integration pass (time integrator only) and then extracts the
growth rates from the per-ky traces.

CLI usage
---------

.. code-block:: bash

   spectrax-gk run-linear --config examples/configs/cyclone.toml --plot --outdir docs/_static
   spectrax-gk scan-linear --config examples/configs/etg.toml --plot --outdir docs/_static
   spectrax-gk run-runtime-linear --config examples/configs/runtime_cyclone.toml
   spectrax-gk scan-runtime-linear --config examples/configs/runtime_etg.toml --batch-ky
   spectrax-gk run-runtime-nonlinear --config examples/configs/runtime_cyclone.toml --sample-stride 5 --out docs/_static/nonlinear_cyclone_diag.csv

For ``run-runtime-nonlinear``, omit ``--steps`` when ``fixed_dt = false`` unless
you explicitly want a capped step count. The CLI now preserves ``steps = None``
for adaptive nonlinear runs so the runtime can keep integrating in chunks until
it reaches the requested ``t_max`` instead of silently reverting to the old
``round(t_max / dt)`` ceiling.

When ``run-runtime-nonlinear`` writes ``--out`` CSV diagnostics, the base columns are
``t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux`` and species-resolved
columns are appended when available:
``heat_flux_s{i}``, ``particle_flux_s{i}`` for species index ``i``.

Python driver
-------------

.. code-block:: bash

   python examples/run_from_toml.py --config examples/configs/etg.toml --plot --outdir docs/_static
   python examples/runtime_from_toml.py --config examples/configs/runtime_kbm.toml

TOML sections
-------------

Supported sections include:

* ``[grid]`` (``GridConfig``)
* ``[time]`` (``TimeConfig``)
* ``[geometry]`` (``GeometryConfig``)
* ``[model]`` (case-specific model config)
* ``[init]`` (``InitializationConfig``)
* ``[run]`` (single-ky run settings)
* ``[scan]`` (ky scan settings)
* ``[fit]`` (growth-rate windowing options)
* ``gx_reference`` (top-level flag or ``[gx_reference] enabled = true`` to enforce GX-reference defaults)
* ``[terms]`` (toggle linear terms)
* ``[krylov]`` (Krylov solver settings)

Runtime sections
^^^^^^^^^^^^^^^^

For runtime-configured inputs (``load_runtime_from_toml``), supported sections
are:

* ``[[species]]`` (kinetic species definitions)
* ``[physics]`` (electrostatic/electromagnetic, adiabatic/kinetic, linear/nonlinear)
* ``[collisions]`` (collision and hypercollision controls)
* ``[normalization]`` (contract key + optional overrides)
* ``[terms]`` (term toggles used by modular RHS assembly)
* ``[expert]`` (advanced fixed-mode controls for specialized workflows)
* ``[run]`` / ``[scan]`` / ``[fit]`` (driver controls)

Notable runtime-only keys:

* ``[collisions] damp_ends_amp`` / ``damp_ends_widthfrac``: reference-compatible end
  damping defaults are ``0.1`` and ``0.125``.
* ``[physics] reduced_model``: explicit physics-family selector for benchmark
  inputs that are not full gyrokinetics. The default is ``"gyrokinetic"``.
  ``"cetg"`` and ``"krehm"`` are accepted as explicit boundary markers, but
  the runtime currently raises ``NotImplementedError`` for them instead of
  silently routing those inputs through the wrong full-GK equations.
* ``[collisions] damp_ends_scale_by_dt``: compatibility escape hatch for older
  per-step inputs. The reference-compatible default is ``false`` because
  ``damp_ends_amp`` is already a per-unit-time damping rate.
* ``[collisions] hypercollisions_const`` / ``hypercollisions_kz``: defaults are
  the reference-compatible ``0.0`` / ``1.0`` (kz-proportional hypercollisions enabled by
  default, constant hypercollisions off).
* ``[collisions] p_hyper_m``: when omitted, the runtime path follows the GX
  default ``min(20, Nm/2)`` instead of using a fixed exponent across Hermite
  resolutions.
* ``[normalization] flux_scale``: multiplicative factor applied to heat/particle
  flux diagnostics (GX-reference default ``1.0``).
* ``[normalization] wphi_scale``: multiplicative factor applied to ``Wphi``
  diagnostics (Cyclone GX-reference uses ``1.155``).
* ``[init] init_single`` with ``gaussian_init = false`` and ``init_single = false``:
  initialize a random perturbation across the exact GX startup loop
  bounds in ``(ky,kx)``.
* ``[init] init_single`` with ``gaussian_init = true`` and ``init_single = false``:
  initialize a Gaussian envelope across the same GX startup loop
  bounds.
* ``[init] init_single = true``:
  initialize only the selected ``(ky,kx)`` mode.
* ``[init] init_field = "all"``: the runtime/TOML path follows GX moment
  scaling for this initializer, using reduced amplitudes for ``tpar``
  (``1/sqrt(2)``) and ``qpar`` (``1/sqrt(6)``).
* ``[init] init_electrons_only``: if ``true`` in multispecies runs, initialize
  only electron species (GX ``init_electrons_only`` behavior). If ``false``
  (default), initialize all kinetic species.
* ``[init] random_seed``: RNG seed used for reference-compatible random initial conditions
  with the same glibc ``rand()`` sequence and startup mode ordering that GX
  uses on Linux
  (default ``22``, matching GX). The runtime now follows the Linux ``glibc``
  ``rand()`` sequence used by GX together with GX's positive-``kx``-major loop
  order and exact startup loop bounds, so random multi-mode perturbations are
  host-platform independent and reproduce the same seeded pattern on macOS and
  Linux.
* ``[init] init_file``: load a saved complex state from either the full-``ky``
  SPECTRAX layout or GX's packed positive-``ky`` layout.
* ``[init] init_file_scale`` / ``init_file_mode``: scale a loaded restart state
  and either ``replace`` the analytic seed (default) or ``add`` it to the
  fresh perturbation. This is the general runtime equivalent of GX's
  restart-scaling and ``restart_with_perturb`` workflows.
* ``[expert] fixed_mode`` with ``iky_fixed`` / ``ikx_fixed``: keep one Fourier
  mode exactly frozen during nonlinear evolution, matching GX's ``eqfix``
  behavior used by the ``secondary`` benchmark.
* ``[time] method = "sspx3"``: use the GX SSPx3 scheme directly. This is the
  relevant explicit method for GX's ``secondary`` and ``cETG`` benchmark
  families. Plain ``rk3`` now follows GX's three-stage Heun-style timestepper;
  ``rk3_classic`` keeps the older classical RK3 update if you need it for
  controlled comparisons, and ``rk3_gx`` remains as a compatibility alias.
