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
   gx_parity = true

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
select the GX-style real FFT nonlinear bracket. Set ``gx_real_fft = false`` to
use a full complex FFT for the nonlinear term. Nonlinear diagnostics can be
decimated with ``diagnostics_stride`` (compute/output every ``N`` steps). To
control the Laguerre handling in nonlinear brackets, set
``laguerre_nonlinear_mode = "grid"`` (GX-style quadrature, default) or
``laguerre_nonlinear_mode = "spectral"`` (use spectral ``Jl`` without the
quadrature transform).

The ``[geometry]`` section supports ``drift_scale`` to switch between GX-style
(``drift_scale = 1.0``) and GS2-style (``drift_scale = 2.0``) drift
normalizations. The default configuration in SPECTRAX-GK uses the GX parity
value.

Solver and fit-signal keys
--------------------------

The ``[run]`` and ``[scan]`` sections accept ``solver`` and ``fit_signal`` keys:

* ``solver = "auto"`` (default): choose time vs Krylov and fall back if needed
* ``solver = "time"``: always use time integration
* ``solver = "krylov"``: always use the matrix-free eigen solver

* ``fit_signal = "auto"`` (default): pick ``phi`` vs density based on fit quality
* ``fit_signal = "phi"``: use the electrostatic potential time trace
* ``fit_signal = "density"``: use the density moment time trace

CLI usage
---------

.. code-block:: bash

   spectrax-gk run-linear --config examples/configs/cyclone.toml --plot --outdir docs/_static
   spectrax-gk scan-linear --config examples/configs/etg.toml --plot --outdir docs/_static
   spectrax-gk run-runtime-linear --config examples/configs/runtime_cyclone.toml
   spectrax-gk scan-runtime-linear --config examples/configs/runtime_etg.toml
   spectrax-gk run-runtime-nonlinear --config examples/configs/runtime_cyclone.toml --out docs/_static/nonlinear_cyclone_diag.csv

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
* ``gx_parity`` (top-level flag or ``[gx_parity] enabled = true`` to enforce GX parity defaults)
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
* ``[run]`` / ``[scan]`` / ``[fit]`` (driver controls)

Notable runtime-only keys:

* ``[collisions] damp_ends_scale_by_dt``: if true, interpret ``damp_ends_amp`` as
  a per-unit-time strength and scale it internally by ``1/dt`` (GX parity
  default).
* ``[normalization] flux_scale``: multiplicative factor applied to heat/particle
  flux diagnostics (GX parity default ``2.0``).
* ``[normalization] wphi_scale``: multiplicative factor applied to ``Wphi``
  diagnostics (Cyclone GX parity uses ``1.155``).
