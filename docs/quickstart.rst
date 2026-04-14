Quickstart
==========

Install
-------

.. code-block:: bash

   git clone https://github.com/uwplasma/SPECTRAX-GK
   cd SPECTRAX-GK
   pip install -e .

Executable demo
---------------

.. code-block:: bash

   spectrax-gk cyclone-info
   spectrax-gk cyclone-kperp --kx0 0.0 --ky 0.3
   cd examples/linear/axisymmetric && spectrax-gk cyclone.toml
   spectrax-gk scan-runtime-linear --config examples/linear/axisymmetric/runtime_etg.toml --plot --outdir docs/_static
   spectrax-gk run-runtime-linear --config examples/linear/axisymmetric/cyclone.toml --out tools_out/cyclone_runtime
   spectrax-gk run-runtime-nonlinear --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml --steps 50 --out tools_out/cyclone_nonlinear
   spectrax-gk run-runtime-nonlinear --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml --steps 50 --out tools_out/cyclone_nonlinear.out.nc
   spectrax-gk examples/nonlinear/axisymmetric/runtime_cetg_reference.toml --steps 100

When ``--out`` is provided for runtime-configured single-point runs, the executable writes
a JSON summary plus sidecar time-series/state artifacts using the supplied path
as a prefix.

If the nonlinear target ends in ``.out.nc`` or another ``.nc`` suffix, the
runtime writes a restartable NetCDF bundle instead:

- ``*.out.nc``: diagnostic history, geometry, and input metadata
- ``*.big.nc``: final fields and moments in spectral and real-space layouts
- ``*.restart.nc``: restart state for continuation runs

The same artifact prefix can be stored in the runtime TOML itself:

.. code-block:: toml

   [output]
   path = "tools_out/cyclone_runtime"

To make the run restart-aware, add the restart controls directly to the TOML:

.. code-block:: toml

   [time]
   nstep_restart = 100

   [output]
   path = "tools_out/cyclone_runtime.out.nc"
   restart_if_exists = true
   save_for_restart = true
   append_on_restart = true

Rerunning the same nonlinear command then resumes from the saved
``*.restart.nc`` checkpoint and appends the continued history to ``*.out.nc``.

Plot diagnostics directly from the output:

.. code-block:: bash

   python examples/utilities/plot_runtime_outputs.py tools_out/cyclone_nonlinear.out.nc \
     --out tools_out/cyclone_nonlinear_diagnostics.png

Python demo
-----------

.. code-block:: python

   from spectraxgk import load_cyclone_reference, run_cyclone_linear

   ref = load_cyclone_reference()
   result = run_cyclone_linear(ky_target=0.3, method="rk4")

   print(result.gamma, result.omega)

Run from TOML
-------------

.. code-block:: bash

   python examples/utilities/runtime_from_toml.py --config examples/linear/axisymmetric/cyclone.toml

Figure generation
-----------------

.. code-block:: bash

   PYTHONPATH=src python tools/make_benchmark_atlas.py
