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

W7-X nonlinear imported geometry TOML run
-----------------------------------------

.. code-block:: bash

   python -m spectraxgk.cli run-runtime-nonlinear \
     --config examples/configs/runtime_w7x_nonlinear_imported_geometry.toml

Generate GX-compatible geometry from a runtime TOML
---------------------------------------------------

.. code-block:: bash

   python tools/generate_gx_vmec_eik.py \
     --config examples/configs/runtime_hsx_nonlinear_vmec_geometry.toml

HSX nonlinear VMEC runtime example
----------------------------------

.. code-block:: bash

   python examples/hsx_nonlinear_vmec_geometry.py \
     --vmec-file /home/user/local/vmec_equilibria/HSX/QHS_vac/wout_HSX_QHS_vac.nc

HSX nonlinear VMEC TOML run
---------------------------

.. code-block:: bash

   python -m spectraxgk.cli run-runtime-nonlinear \
     --config examples/configs/runtime_hsx_nonlinear_vmec_geometry.toml

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
