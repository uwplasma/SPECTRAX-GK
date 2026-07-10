Quickstart
==========

Install
-------

.. code-block:: bash

   pip install spectraxgk

or install the development checkout:

.. code-block:: bash

   git clone https://github.com/uwplasma/SPECTRAX-GK
   cd SPECTRAX-GK
   pip install -e .

Executable demo
---------------

.. code-block:: bash

   spectraxgk
   spectrax-gk
   spectraxgk examples/linear/axisymmetric/cyclone.toml
   spectraxgk run-runtime-linear --config examples/linear/axisymmetric/cyclone.toml --out cyclone_runtime
   spectraxgk run-runtime-nonlinear --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml --steps 50 --out tools_out/cyclone_nonlinear.out.nc
   spectraxgk --plot tools_out/cyclone_nonlinear.out.nc
   spectraxgk --plot spectraxgk_default_linear.summary.json

Running ``spectraxgk`` with no TOML launches a short Cyclone initial-value
linear demo, prints live progress with elapsed time and ETA, and writes the
artifacts in the current directory:

- ``spectraxgk_default_linear.toml``: the input file that reproduces the run
- ``spectraxgk_default_linear.summary.json``
- ``spectraxgk_default_linear.timeseries.csv``
- ``spectraxgk_default_linear.eigenfunction.csv``
- ``spectraxgk_default_linear.png``

The plot is the standard two-panel linear quickstart view: the log-scale
``|\phi|^2`` time history with fitted ``(\gamma, \omega)`` on the left, and
the normalized real/imaginary eigenfunction on the right. To rerun the same
numerical case explicitly:

.. code-block:: bash

   spectraxgk spectraxgk_default_linear.toml --progress

When progress output is enabled (for example on a TTY or with the explicit
progress flags), the executable prints live status lines with step/time
progress, wall elapsed time, and an estimated wall-clock time remaining. Long
adaptive nonlinear runs also report chunk-level elapsed/ETA updates at the
runtime layer.

When ``--out`` is provided for runtime-configured single-point runs, the
executable writes a JSON summary plus sidecar time-series/state artifacts using
the supplied path as a prefix.

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

   spectraxgk --plot tools_out/cyclone_nonlinear.out.nc
   spectraxgk --plot spectraxgk_default_linear.summary.json

Self-contained VMEC geometry
----------------------------

The VMEC-backed examples are prefilled with relative ``wout_*.nc`` paths. The
repository ships small ``vmec_jax`` input decks, not large generated WOUT
files. Generate the needed equilibria locally, then run the TOMLs directly:

.. code-block:: bash

   pip install vmec-jax
   cd examples/vmec
   vmec_jax input.circular_tokamak
   vmec_jax input.NuhrenbergZille_1988_QHS
   vmec_jax input.nfp3_QI_fixed_resolution_final
   cd ../..

   spectraxgk run --config examples/linear/axisymmetric/runtime_circular_vmec_linear.toml
   spectraxgk run --config examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml
   spectraxgk run --config examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml

The bundled QHS/QI/QA decks are self-contained demonstrators. Exact
machine-specific HSX or W7-X validation should use the same TOMLs with
``--vmec-file`` pointing to the corresponding benchmark ``wout_*.nc``.

Geometry path overrides
-----------------------

The executable can still override geometry paths without editing the TOML.
These command-line paths are resolved from the shell's current working
directory, while paths written in the TOML remain resolved from the TOML
location. Use ``--vmec-file`` when the runtime config already uses a
VMEC-backed geometry model:

.. code-block:: bash

   spectrax-gk run \
     --config examples/nonlinear/non-axisymmetric/runtime_hsx_nonlinear_vmec_geometry.toml \
     --vmec-file /absolute/or/relative/wout_machine_specific.nc \
     --out tools_out/hsx_vmec_run

Use ``--geometry-file`` only for advanced imported-geometry configs that
already use ``model = "vmec-eik"``, ``model = "imported-eik"``, or
``model = "imported-netcdf"``. This is not needed for the shipped VMEC examples:

.. code-block:: bash

   spectrax-gk run \
     --config external_imported_geometry_case.toml \
     --geometry-file /absolute/or/relative/external_geometry.eik.nc \
     --out tools_out/imported_run

``--geometry-file`` only replaces ``[geometry].geometry_file``; it does not
switch ``model = "vmec"`` into imported-geometry mode. For ``model = "vmec"``,
``geometry_file`` remains the generated ``*.eik.nc`` target/cache path.

Python demo
-----------

.. code-block:: python

   from spectraxgk import load_runtime_from_toml, run_runtime_linear

   config, _ = load_runtime_from_toml(
       "examples/linear/axisymmetric/cyclone.toml"
   )
   result = run_runtime_linear(config, ky_target=0.3)

   print(result.gamma, result.omega)

Named comparison cases and their tracked reference tables are intentionally
kept in :mod:`spectraxgk.benchmarks`; ordinary simulations use the unified
runtime API above.

Run from TOML
-------------

.. code-block:: bash

   python examples/utilities/runtime_from_toml.py --config examples/linear/axisymmetric/cyclone.toml

Figure generation
-----------------

.. code-block:: bash

   PYTHONPATH=src python tools/artifacts/make_benchmark_atlas.py
