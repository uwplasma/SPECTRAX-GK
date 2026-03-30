Quickstart
==========

Install
-------

.. code-block:: bash

   pip install -e .

CLI demo
--------

.. code-block:: bash

   spectrax-gk cyclone-info
   spectrax-gk cyclone-kperp --kx0 0.0 --ky 0.3
   spectrax-gk run-linear --config examples/linear/axisymmetric/cyclone.toml --plot --outdir docs/_static
   spectrax-gk scan-linear --config examples/linear/axisymmetric/etg.toml --plot --outdir docs/_static

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

   python examples/run_from_toml.py --config examples/linear/axisymmetric/cyclone.toml --plot --outdir docs/_static

Figure generation
-----------------

.. code-block:: bash

   PYTHONPATH=src python tools/make_benchmark_atlas.py
