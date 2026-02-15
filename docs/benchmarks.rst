Benchmarks
==========

Cyclone Base Case (Linear, Adiabatic Electrons)
----------------------------------------------

We include a reference dataset for the linear Cyclone base case derived from the
GX benchmark outputs for s-alpha geometry with Boltzmann electrons. [GX]_ The values
are stored in:

- ``spectraxgk/data/cyclone_gx_adiabatic_ref.csv``

These correspond to the normalized growth rates and real frequencies shown in
Fig. 1 of the GX paper (Cyclone base case, Boltzmann electrons). The harness
loads these values and provides utilities to extract growth rates from time
series signals. The current linear model in this rebuild is streaming-only, so
numerical agreement with Cyclone growth rates is not expected until curvature
and gradient drive terms are implemented.

How to run the harness
----------------------

.. code-block:: bash

   python examples/cyclone_linear_benchmark.py

Regenerating the reference CSV
-------------------------------

.. code-block:: bash

   python tools/extract_gx_cyclone_reference.py \
     /path/to/itg_salpha_adiabatic_electrons_correct.out.nc \
     src/spectraxgk/data/cyclone_gx_adiabatic_ref.csv

Benchmark harness
-----------------

The Python helper ``run_cyclone_linear`` runs a short linear simulation and
extracts growth rates using a log-amplitude and phase fit. The current
implementation uses the streaming-only operator, so agreement with the
reference values is a future milestone tied to curvature and gradient-drive
terms.
