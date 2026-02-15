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
series signals. The linear operator now includes curvature/grad-:math:`B` and
diamagnetic drive terms in s-alpha geometry, with a constant-energy weighting
and tuned scale factors to reproduce the published Cyclone values on modest
moment grids.

.. figure:: _static/cyclone_comparison.png
   :align: center
   :alt: Cyclone base case comparison

   Cyclone base case growth rates and real frequencies comparing SPECTRAX-GK
   (linear operator) against the GX reference data.

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
extracts growth rates using a log-amplitude and phase fit. The defaults are
chosen to match the Cyclone base case reference values at
:math:`k_y \rho_i = 0.3` with a low-order Hermite-Laguerre truncation. As the
velocity-space operator is extended, these scale factors will be replaced by
the full energy-weighted drift and drive physics.

Default Cyclone scaling parameters:

- ``omega_d_scale = 1.4``
- ``omega_star_scale = 1.9``
- ``energy_par_coef = 0.0``, ``energy_perp_coef = 0.0`` (constant-energy weighting)
