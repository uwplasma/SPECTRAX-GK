Examples
========

The ``examples`` directory is organized around two layers:

- case-backed runtime drivers that map directly onto the tracked runtime TOMLs,
- focused demos and benchmark helpers for theory, operators, and scan workflows.

Config-backed runtime cases
---------------------------

These scripts are the closest match to the production benchmark workflows.
They load the checked-in runtime TOMLs and expose only the most useful runtime
overrides at the command line.

Tokamak cases
^^^^^^^^^^^^^

.. code-block:: bash

   python examples/cyclone_runtime_linear.py
   python examples/cyclone_runtime_nonlinear.py --steps 200
   python examples/cetg_runtime_nonlinear.py --steps 1000
   python examples/etg_runtime_linear.py
   python examples/kaw_runtime_linear.py
   python examples/kbm_runtime_linear.py
   python examples/kbm_runtime_nonlinear.py --steps 200
   python examples/miller_nonlinear_runtime.py --steps 200

Stellarator and imported-geometry cases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python examples/w7x_linear_imported_geometry.py \
     --geometry-file /path/to/itg_w7x_adiabatic_electrons.eik.nc

   python examples/hsx_linear_imported_geometry.py \
     --geometry-file /path/to/hsx_linear.eik.nc

   python examples/w7x_nonlinear_imported_geometry.py \
     --geometry-file /path/to/w7x_adiabatic_electrons.eik.nc

   python examples/w7x_nonlinear_vmec_geometry.py --steps 200
   python examples/hsx_nonlinear_vmec_geometry.py --steps 200

For the VMEC-backed stellarator examples, omit ``--steps`` when you want the
default adaptive horizon. Set ``--steps`` only when you intentionally want a
short profiling or diagnostic window.

Runtime TOML entry points
-------------------------

When you want the full config surface instead of the thin case wrappers, use
the CLI or the generic example drivers directly:

.. code-block:: bash

   python examples/runtime_from_toml.py --config examples/configs/runtime_cyclone.toml
   python examples/runtime_from_toml.py --config examples/configs/runtime_etg.toml
   python examples/runtime_from_toml.py --config examples/configs/runtime_kbm.toml

   python -m spectraxgk.cli run-runtime-linear \
     --config examples/configs/runtime_w7x_linear_imported_geometry.toml

   python -m spectraxgk.cli run-runtime-nonlinear \
     --config examples/configs/runtime_w7x_nonlinear_vmec_geometry.toml

Geometry helper workflows
-------------------------

The runtime geometry path can generate imported geometry files from VMEC and
Miller inputs when the external helper scripts are available:

.. code-block:: bash

   export HSX_VMEC_FILE=/absolute/path/to/wout_HSX_QHS_vac.nc
   export GX_VMEC_PYTHON=python3
   python tools/generate_gx_vmec_eik.py \
     --config examples/configs/runtime_hsx_nonlinear_vmec_geometry.toml

   python tools/generate_gx_miller_eik.py \
     --config examples/configs/runtime_cyclone_nonlinear_gx_miller.toml

Benchmark and scan helpers
--------------------------

These scripts produce the scan-level plots and tables used in the benchmark
discussion:

.. code-block:: bash

   python examples/cyclone_linear_benchmark.py
   python examples/etg_linear_auto.py
   python examples/etg_linear_benchmark.py
   python examples/kbm_beta_scan.py
   python examples/kinetic_linear_benchmark.py
   python examples/tem_linear_benchmark.py

Foundational demos
------------------

These smaller examples are useful for understanding the numerical building
blocks without running a full benchmark case:

.. code-block:: bash

   python examples/basis_orthonormality.py
   python examples/cyclone_geometry.py
   python examples/diffrax_linear_demo.py
   python examples/example.py
   python examples/gradB_coupling_hl_1d.py
   python examples/linear_rhs_demo.py
   python examples/two_stream_hermite_1d.py

Secondary slab workflow
-----------------------

.. code-block:: bash

   python -m spectraxgk.cli run-runtime-linear \
     --config examples/configs/runtime_secondary_slab.toml

   python examples/secondary_slab_workflow.py

The staged helper runs the linear seed, writes a restart state in the runtime
binary layout, and then launches the nonlinear follow-up with the matching
restart and fixed-mode controls used in the tracked secondary benchmark.

Reduced-model runtime
---------------------

.. code-block:: bash

   python examples/cetg_runtime_nonlinear.py --steps 1000
   python -m spectraxgk.cli run-runtime-nonlinear \
     --config examples/configs/runtime_cetg_reference.toml

The reduced collisional slab ETG workflow uses the dedicated cETG runtime
solver rather than the full-GK field solve path.
