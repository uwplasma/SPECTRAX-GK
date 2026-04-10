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

   python examples/linear/axisymmetric/cyclone_runtime_linear.py
   python examples/nonlinear/axisymmetric/cyclone_runtime_nonlinear.py --steps 200
   python examples/nonlinear/axisymmetric/cetg_runtime_nonlinear.py --steps 1000
   python examples/linear/axisymmetric/etg_runtime_linear.py
   python examples/linear/axisymmetric/kaw_runtime_linear.py
   python examples/linear/axisymmetric/kbm_runtime_linear.py
   python examples/nonlinear/axisymmetric/kbm_runtime_nonlinear.py --steps 200
   python examples/nonlinear/axisymmetric/miller_nonlinear_runtime.py --steps 200

Stellarator and imported-geometry cases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python examples/linear/non-axisymmetric/w7x_linear_imported_geometry.py \
     --geometry-file /path/to/itg_w7x_adiabatic_electrons.eik.nc

   python examples/linear/non-axisymmetric/hsx_linear_imported_geometry.py \
     --geometry-file /path/to/hsx_linear.eik.nc

   python examples/nonlinear/non-axisymmetric/w7x_nonlinear_imported_geometry.py \
     --geometry-file /path/to/w7x_adiabatic_electrons.eik.nc

   export W7X_VMEC_FILE=/absolute/path/to/wout_w7x.nc
   export HSX_VMEC_FILE=/absolute/path/to/wout_HSX_QHS_vac.nc
   python examples/nonlinear/non-axisymmetric/w7x_nonlinear_vmec_geometry.py --steps 200
   python examples/nonlinear/non-axisymmetric/hsx_nonlinear_vmec_geometry.py --steps 200

For the VMEC-backed stellarator examples, omit ``--steps`` when you want the
default adaptive horizon. Set ``--steps`` only when you intentionally want a
short profiling or diagnostic window.

The shipped nonlinear stellarator runtime TOMLs now also emit artifact bundles
under ``tools_out/`` by default:

- ``tools_out/w7x_nonlinear_vmec_runtime.diagnostics.csv``
- ``tools_out/hsx_nonlinear_vmec_runtime.diagnostics.csv``
- ``tools_out/w7x_nonlinear_imported_runtime.diagnostics.csv``

Those diagnostics and their matching ``*.summary.json`` files are the intended
inputs for the parity helpers under ``tools/``.
The direct Python runtime wrappers now route through the same artifact-aware
nonlinear path as the CLI, so long adaptive runs update that bundle as each
chunk completes.

Runtime TOML entry points
-------------------------

When you want the full config surface instead of the thin case wrappers, use
the CLI or the generic example drivers directly. These runtime utilities are
best treated as solver-smoke and exploration entry points; the benchmark
examples remain the audited parity surface for ETG and the other validation
lanes:

.. code-block:: bash

   python examples/utilities/runtime_from_toml.py --config examples/linear/axisymmetric/runtime_cyclone.toml
   python examples/utilities/runtime_from_toml.py --config examples/linear/axisymmetric/runtime_etg.toml
   python examples/utilities/runtime_from_toml.py --config examples/linear/axisymmetric/runtime_kbm.toml
   python examples/linear/axisymmetric/etg_linear_auto.py --outdir tools_out/etg_auto

   python -m spectraxgk.cli run-runtime-linear \
     --config examples/linear/non-axisymmetric/runtime_w7x_linear_imported_geometry.toml

   spectrax-gk examples/nonlinear/axisymmetric/runtime_cetg_reference.toml --steps 200

Nonlinear restart and continuation
----------------------------------

The tracked nonlinear runtime path supports a GX-style ``out/big/restart``
bundle together with continuation from the saved restart state.

One-shot nonlinear bundle write:

.. code-block:: bash

   spectrax-gk run-runtime-nonlinear \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_gx.toml \
     --steps 200 \
     --out tools_out/cyclone_release.out.nc

Restart-aware TOML snippet:

.. code-block:: toml

   [time]
   nstep_restart = 100

   [output]
   path = "tools_out/cyclone_release.out.nc"
   restart_if_exists = true
   save_for_restart = true
   append_on_restart = true
   restart_with_perturb = false

With that configuration, rerunning the same nonlinear command resumes from
``tools_out/cyclone_release.restart.nc`` when it already exists and appends the
continued history to ``tools_out/cyclone_release.out.nc``. This is the
recommended user-facing workflow for long nonlinear turbulence jobs.

Geometry helper workflows
-------------------------

The runtime geometry path can generate imported geometry files from VMEC and
Miller inputs when the external helper scripts are available:

.. code-block:: bash

   export W7X_VMEC_FILE=/absolute/path/to/wout_w7x.nc
   export HSX_VMEC_FILE=/absolute/path/to/wout_HSX_QHS_vac.nc
   export SPECTRAX_BOOZ_XFORM_JAX_PATH=/absolute/path/to/booz_xform_jax
   python tools/generate_gx_vmec_eik.py \
     --config examples/nonlinear/non-axisymmetric/runtime_hsx_nonlinear_vmec_geometry.toml

   python tools/generate_gx_miller_eik.py \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_gx_miller.toml

Benchmark and scan helpers
--------------------------

These scripts produce the scan-level plots and tables used in the benchmark
discussion:

.. code-block:: bash

   python examples/benchmarks/cyclone_linear_benchmark.py
   python examples/linear/axisymmetric/etg_linear_auto.py
   python examples/benchmarks/etg_linear_benchmark.py
   python examples/benchmarks/kbm_beta_scan.py
   python examples/benchmarks/kinetic_linear_benchmark.py
   python examples/benchmarks/tem_linear_benchmark.py

Foundational demos
------------------

These smaller examples are useful for understanding the numerical building
blocks without running a full benchmark case:

.. code-block:: bash

   python examples/benchmarks/basis_orthonormality.py
   python examples/theory_and_demos/cyclone_geometry.py
   python examples/theory_and_demos/diffrax_linear_demo.py
   python examples/theory_and_demos/example.py
   python examples/theory_and_demos/gradB_coupling_hl_1d.py
   python examples/theory_and_demos/linear_rhs_demo.py
   python examples/theory_and_demos/two_stream_hermite_1d.py

Secondary slab workflow
-----------------------

.. code-block:: bash

   python -m spectraxgk.cli run-runtime-linear \
     --config examples/benchmarks/runtime_secondary_slab.toml

   python examples/benchmarks/secondary_slab_workflow.py

The staged helper runs the linear seed, writes a restart state in the runtime
binary layout, and then launches the nonlinear follow-up with the matching
restart and fixed-mode controls used in the tracked secondary benchmark.

Reduced-model runtime
---------------------

.. code-block:: bash

   python examples/nonlinear/axisymmetric/cetg_runtime_nonlinear.py --steps 1000
   spectrax-gk examples/nonlinear/axisymmetric/runtime_cetg_reference.toml --steps 1000

The reduced collisional slab ETG workflow uses the dedicated cETG runtime
solver rather than the full-GK field solve path.

Full-GK ETG nonlinear pilot
---------------------------

.. code-block:: bash

   python examples/nonlinear/axisymmetric/etg_runtime_nonlinear.py --steps 200
   JAX_ENABLE_X64=1 spectrax-gk examples/nonlinear/axisymmetric/runtime_etg_nonlinear.toml --steps 200

This is the full-GK two-species ETG nonlinear pilot lane. It is intentionally
separate from the reduced cETG workflow and is the correct starting point for
future GX-backed nonlinear ETG parity work.
