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

Python demo
-----------

.. code-block:: python

   from spectraxgk import load_cyclone_reference, run_cyclone_linear

   ref = load_cyclone_reference()
   result = run_cyclone_linear(ky_target=0.3, steps=200, dt=0.05, tmin=5.0)

   print(result.gamma, result.omega)

Figure generation
-----------------

.. code-block:: bash

   python tools/make_figures.py
