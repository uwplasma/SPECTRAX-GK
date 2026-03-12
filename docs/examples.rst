Examples
========

Cyclone geometry
----------------

.. code-block:: bash

   python examples/cyclone_geometry.py

Basis orthonormality
--------------------

.. code-block:: bash

   python examples/basis_orthonormality.py

Linear RHS demo
----------------

.. code-block:: bash

   python examples/linear_rhs_demo.py

Cyclone linear benchmark
------------------------

.. code-block:: bash

   python examples/cyclone_linear_benchmark.py

Cyclone linear driver (JAX/JIT)
-------------------------------

.. code-block:: bash

   python examples/cyclone_linear_driver.py

Cyclone nonlinear driver (JAX/JIT)
----------------------------------

.. code-block:: bash

   python examples/cyclone_nonlinear_driver.py

ETG auto benchmark (newcomer-friendly)
--------------------------------------

.. code-block:: bash

   python examples/etg_linear_auto.py

ETG linear trend
----------------

.. code-block:: bash

   python examples/etg_linear_benchmark.py

TOML-driven run
---------------

.. code-block:: bash

   python examples/run_from_toml.py --config examples/configs/cyclone.toml --plot

Unified runtime TOML run
------------------------

.. code-block:: bash

   python examples/runtime_from_toml.py --config examples/configs/runtime_cyclone.toml
   python examples/runtime_from_toml.py --config examples/configs/runtime_etg.toml
   python examples/runtime_from_toml.py --config examples/configs/runtime_kbm.toml

W7-X imported geometry runtime example
-------------------------------------

.. code-block:: bash

   python examples/w7x_linear_imported_geometry.py \
     --geometry-file /path/to/itg_w7x_adiabatic_electrons.eik.nc \
     --ky 0.3

W7-X imported geometry TOML run
-------------------------------

.. code-block:: bash

   python -m spectraxgk.cli run-runtime-linear \
     --config examples/configs/runtime_w7x_linear_imported_geometry.toml

W7-X nonlinear imported geometry runtime example
------------------------------------------------

.. code-block:: bash

   python examples/w7x_nonlinear_imported_geometry.py \
     --geometry-file /path/to/w7x_adiabatic_electrons.eik.nc

Leave ``--steps`` unset for the default GX-style adaptive horizon. Set it only
when you intentionally want a capped step count for profiling or reduced
benchmark windows.

W7-X nonlinear imported geometry TOML run
-----------------------------------------

.. code-block:: bash

   python -m spectraxgk.cli run-runtime-nonlinear \
     --config examples/configs/runtime_w7x_nonlinear_imported_geometry.toml

W7-X nonlinear VMEC TOML run
----------------------------

.. code-block:: bash

   export GX_VMEC_PYTHON=python3
   python -m spectraxgk.cli run-runtime-nonlinear \
     --config examples/configs/runtime_w7x_nonlinear_vmec_geometry.toml

The W7-X VMEC TOML now resolves ``geometry.vmec_file`` relative to the active
GX repo, so it no longer depends on a workstation-specific absolute path.

Generate GX-compatible geometry from a runtime TOML
---------------------------------------------------

.. code-block:: bash

   export HSX_VMEC_FILE=/absolute/path/to/wout_HSX_QHS_vac.nc
   python tools/generate_gx_vmec_eik.py \
     --config examples/configs/runtime_hsx_nonlinear_vmec_geometry.toml

Generate GX-compatible Miller geometry from a runtime TOML
----------------------------------------------------------

.. code-block:: bash

   python tools/generate_gx_miller_eik.py \
     --config examples/configs/runtime_cyclone_nonlinear_gx_miller.toml

HSX nonlinear VMEC runtime example
----------------------------------

.. code-block:: bash

   python examples/hsx_nonlinear_vmec_geometry.py \
     --vmec-file /Users/rogeriojorge/local/vmec_equilibria/HSX/QHS_vac/wout_HSX_QHS_vac.nc \
     --gx-python python3

As with the W7-X example above, omit ``--steps`` for the default adaptive
nonlinear runtime path.

HSX nonlinear VMEC TOML run
---------------------------

.. code-block:: bash

   export HSX_VMEC_FILE=/absolute/path/to/wout_HSX_QHS_vac.nc
   export GX_VMEC_PYTHON=python3
   python -m spectraxgk.cli run-runtime-nonlinear \
     --config examples/configs/runtime_hsx_nonlinear_vmec_geometry.toml

HSX matched GX input template
-----------------------------

.. code-block:: bash

   /Users/rogeriojorge/local/gx/gx \
     examples/configs/gx_hsx_nonlinear_adiabatic_electrons.in

Secondary slab runtime TOML run
-------------------------------

.. code-block:: bash

   python -m spectraxgk.cli run-runtime-linear \
     --config examples/configs/runtime_secondary_slab.toml

The runtime path now also supports the two GX controls needed for the
nonlinear ``kh01 -> kh01a`` secondary workflow:

* ``[init] init_file_scale`` / ``init_file_mode = "add"`` for scaled restart
  states with an added fresh perturbation.
* ``[expert] fixed_mode = true`` with ``iky_fixed`` / ``ikx_fixed`` for the
  frozen-primary-mode evolution used by GX's ``eqfix`` path.

Secondary slab staged workflow
------------------------------

.. code-block:: bash

   python examples/secondary_slab_workflow.py

The staged helper runs the linear ``kh01`` seed, writes a restart state in the
runtime binary layout, then launches the nonlinear ``kh01a`` follow-up with the
matching scaled additive restart and frozen pump mode controls.

To compare that staged run against a real GX ``kh01a.out.nc`` reference:

.. code-block:: bash

   python tools/compare_gx_secondary.py \
     --gx-source out-nc \
     --gx-out .cache/gx/secondary_legacy/kh01a.out.nc \
     --out docs/_static/secondary_gx_out_compare.csv

The README-target fallback remains available when no real GX output file is present:

.. code-block:: bash

   python tools/compare_gx_secondary.py \
     --gx-source readme \
     --out docs/_static/secondary_gx_readme_compare.csv

When ``--gx-source out-nc`` is used, the helper averages the last half of the
GX ``omega_kxkyt`` time series instead of taking the final instantaneous
sample. That keeps the tiny secondary-frequency comparison tied to the
steady-state plateau rather than to the last noisy point.

cETG reduced-model reference boundary
-------------------------------------

.. code-block:: bash

   python tools/inspect_gx_reduced_model.py \
     /Users/rogeriojorge/local/gx/benchmarks/nonlinear/cETG/cetg.in --json

.. code-block:: bash

   python -m spectraxgk.cli run-runtime-nonlinear \
     --config examples/configs/runtime_cetg_reference.toml

The second command currently raises an explicit ``NotImplementedError``. That
is intentional: SPECTRAX-GK does not yet route GX ``cETG``/``KREHM`` benchmark
inputs through the full-GK slab path, because that would silently compare the
wrong equations. The runtime config and inspection tool define the benchmark
contract that the dedicated reduced-model implementation will need to satisfy.

For the older runnable GX cETG reference lane, inspect the grouped legacy
NetCDF output with:

.. code-block:: bash

   python tools/inspect_gx_legacy_cetg.py /path/to/cetg_smoke.nc --json

KBM beta scan
-------------

.. code-block:: bash

   python examples/kbm_beta_scan.py

W7-X imported geometry parity
-----------------------------

.. code-block:: bash

   python tools/compare_gx_imported_linear.py \
     --gx /path/to/itg_w7x_adiabatic_electrons.out.nc \
     --geometry-file /path/to/itg_w7x_adiabatic_electrons.eik.nc \
     --ky 0.1 0.2 0.3 0.4 \
     --out docs/_static/w7x_linear_t2_scan.csv
