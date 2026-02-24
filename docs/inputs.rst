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
   diagnostic_norm = "none"

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
   solver = "krylov"

Minimal TOML example
--------------------

.. code-block:: toml

   case = "cyclone"

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

   [run]
   ky = 0.3
   Nl = 24
   Nm = 12
   solver = "time"
   method = "imex2"

   [fit]
   auto_window = true
   window_method = "loglinear"

CLI usage
---------

.. code-block:: bash

   spectrax-gk run-linear --config examples/configs/cyclone.toml --plot --outdir docs/_static
   spectrax-gk scan-linear --config examples/configs/etg.toml --plot --outdir docs/_static
   spectrax-gk run-runtime-linear --config examples/configs/runtime_cyclone.toml
   spectrax-gk scan-runtime-linear --config examples/configs/runtime_etg.toml

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
