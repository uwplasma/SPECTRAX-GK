Input Files and CLI
===================

SPECTRAX-GK supports lightweight TOML inputs that map directly onto the
``GridConfig``, ``TimeConfig``, ``GeometryConfig``, and ``ModelConfig`` dataclasses.
You can use these inputs from the CLI or from a Python driver.

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

Python driver
-------------

.. code-block:: bash

   python examples/run_from_toml.py --config examples/configs/etg.toml --plot --outdir docs/_static

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
