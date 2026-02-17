Benchmarks
==========

Benchmark runners follow the ``TimeConfig`` defaults (diffrax enabled with a
fixed-step Heun solver). The fixed-step integrators remain available by setting
``use_diffrax=False`` in the time configuration. A small runtime/memory
comparison script is available in ``tools/benchmark_integrators.py``.

Performance defaults
--------------------

The diffrax defaults aim for stable, consistent step control rather than
absolute speed. A quick sweep with the benchmark script shows the relative
runtime and host-memory costs on the Cyclone defaults (t_max=8, dt=0.01, Nl=6,
Nm=12):

- Custom fixed-step RK2: ~0.23 s, ~0.06 MB host peak.
- Diffrax Heun fixed-step: ~2.68 s, ~8.65 MB host peak.
- Diffrax Tsit5 adaptive: ~2.65 s, ~9.12 MB host peak.

The diffrax Heun default matches the RK2 stability region while keeping the
step size explicit and predictable. For the fastest linear scans, disable
diffrax via ``TimeConfig(use_diffrax=False)``.

For speed-critical diffrax scans, set ``TimeConfig(progress_bar=False)`` to
enable JIT compilation of the solver loop, and reuse consistent step counts to
avoid recompilation. Adaptive runs may also require higher
``diffrax_max_steps`` to prevent early termination.

Cyclone Base Case (Linear, Adiabatic Electrons)
-----------------------------------------------

The Cyclone base case is the canonical ion-temperature-gradient validation
target. SPECTRAX-GK ships a reference dataset stored in:

- ``spectraxgk/data/cyclone_reference_adiabatic.csv``

The benchmark harness loads these values and compares growth rates and
frequencies across a reduced :math:`k_y` scan on the field-aligned grid.

.. list-table:: Cyclone base case parameters (GX Fig. 1)
   :header-rows: 1

   * - Parameter
     - Value
   * - Geometry
     - ``q=1.4``, ``s_hat=0.8``, ``epsilon=0.18``, ``R0=2.77778``
   * - Gradients
     - ``R/LTi=2.49``, ``R/LTe=0.0``, ``R/Ln=0.8``
   * - Species
     - ions only; adiabatic electrons with ``tau_e=1``
   * - Electromagnetic
     - ``beta=0``, ``A_parallel=off``, ``B_parallel=off``
   * - Collisions
     - ``nu_i=1e-2``, hypercollisions off
   * - Operator toggles
     - streaming/mirror/curvature/grad-B/diamagnetic on; nonlinear off
   * - Grid
     - ``Nx=1, Ny=24, Nz=96, y0=20, ntheta=32, nperiod=2``
   * - Velocity resolution
     - ``Nl=6, Nm=16`` (figure generation)
   * - Reference
     - :cite:`GX`

.. figure:: _static/linear_summary.png
   :align: center
   :alt: Linear validation summary

   Multi-panel summary of eigenfunctions, growth rates, and frequencies
   across the linear benchmark suite.

.. figure:: _static/cyclone_comparison.png
   :align: center
   :alt: Cyclone base case comparison

   Cyclone base case growth rates and real frequencies comparing SPECTRAX-GK
   against the published reference dataset.

Kinetic-Electron ITG (Ion-Scale)
--------------------------------

Kinetic electrons introduce the trapped-electron and ion-scale branches used
in published validation studies. The ion-scale kinetic-electron reference data are
stored in:

- ``spectraxgk/data/cyclone_reference_kinetic.csv``

These values are used in the multi-panel validation figure and in the
kinetic-electron regression checks.

.. list-table:: Kinetic-electron ITG parameters (GX Fig. 2a)
   :header-rows: 1

   * - Parameter
     - Value
   * - Geometry
     - ``q=1.4``, ``s_hat=0.8``, ``epsilon=0.18``, ``R0=2.77778``
   * - Gradients
     - ``R/LTi=2.49``, ``R/LTe=2.49``, ``R/Ln=0.8``
   * - Species
     - ions + electrons, ``Te/Ti=1``, ``mi/me=3670``
   * - Electromagnetic
     - ``beta=1e-5``, ``A_parallel=on``, ``B_parallel=on``
   * - Collisions
     - ``nu_i=1e-2``, ``nu_e=1.65e-4``, hypercollisions off
   * - Operator toggles
     - streaming/mirror/curvature/grad-B/diamagnetic on; nonlinear off
   * - Grid
     - ``Nx=1, Ny=24, Nz=96, y0=20, ntheta=32, nperiod=2``
   * - Velocity resolution
     - ``Nl=6, Nm=16`` (figure generation)
   * - Reference
     - :cite:`GX`

ETG (Electron-Scale)
--------------------

Electron-temperature-gradient validation uses a reduced electron-scale box
and a digitized reference dataset from the published electron-scale scan:

- ``spectraxgk/data/etg_reference.csv``

The scan is plotted alongside the SPECTRAX-GK output in the validation
summary figure.

.. list-table:: ETG parameters (GX Fig. 2b)
   :header-rows: 1

   * - Parameter
     - Value
   * - Geometry
     - ``q=1.4``, ``s_hat=0.8``, ``epsilon=0.18``, ``R0=2.77778``
   * - Gradients
     - ``R/LTi=2.49``, ``R/LTe=2.49``, ``R/Ln=0.8``
   * - Species
     - ions + electrons, ``Te/Ti=1``, ``mi/me=3670``
   * - Electromagnetic
     - ``beta=1e-5``, ``A_parallel=on``, ``B_parallel=on``
   * - Collisions
     - ``nu_i=1e-2``, ``nu_e=1.65e-4``, hypercollisions off
   * - Operator toggles
     - streaming/mirror/curvature/grad-B/diamagnetic on; nonlinear off
   * - Grid
     - ``Nx=1, Ny=24, Nz=96, y0=0.2, ntheta=32, nperiod=2``
   * - Velocity resolution
     - ``Nl=6, Nm=16`` (figure generation)
   * - Reference
     - :cite:`GX`

KBM (Electromagnetic Beta Scan)
-------------------------------

Electromagnetic ballooning validation uses a fixed :math:`k_y` and a scan over
:math:`\beta_{ref}`. The reference data are stored in:

- ``spectraxgk/data/kbm_reference.csv``

.. list-table:: KBM parameters (GX Fig. 3)
   :header-rows: 1

   * - Parameter
     - Value
   * - Geometry
     - ``q=1.4``, ``s_hat=0.8``, ``epsilon=0.18``, ``R0=2.77778``
   * - Gradients
     - ``R/LTi=2.49``, ``R/LTe=2.49``, ``R/Ln=0.8``
   * - Species
     - ions + electrons, ``Te/Ti=1``, ``mi/me=3670``
   * - Electromagnetic
     - ``beta_ref`` scan, ``A_parallel=on``, ``B_parallel=off``
   * - Collisions
     - ``nu_i=0``, ``nu_e=0``, hypercollisions off
   * - Operator toggles
     - streaming/mirror/curvature/grad-B/diamagnetic on; nonlinear off
   * - Grid
     - ``Nx=1, Ny=12, Nz=96, y0=10, ntheta=32, nperiod=2``
   * - Velocity resolution
     - ``Nl=6, Nm=16`` (figure generation)
   * - Reference
     - :cite:`GX`

TEM (Trapped-Electron Mode)
---------------------------

The TEM validation case follows the s-alpha parameters reported in Frei et al.
with steep gradients (:math:`R/L_{Ti} = R/L_{Te} = R/L_n = 20`),
:math:`q=2.7`, :math:`\hat{s}=0.5`, :math:`\epsilon=0.18`, and
:math:`m_e/m_i = 0.0027`. The digitized low-:math:`k_y` reference branch is
stored in:

- ``spectraxgk/data/tem_reference.csv``

.. list-table:: TEM parameters (Frei et al. 2022 Fig. 4)
   :header-rows: 1

   * - Parameter
     - Value
   * - Geometry
     - ``q=2.7``, ``s_hat=0.5``, ``epsilon=0.18``, ``R0=2.77778``, ``alpha=0``
   * - Gradients
     - ``R/LTi=20``, ``R/LTe=20``, ``R/Ln=20``
   * - Species
     - ions + electrons, ``Te/Ti=1``, ``mi/me=370``
   * - Electromagnetic
     - ``beta=1e-4``, ``A_parallel=on``, ``B_parallel=off``
   * - Collisions
     - ``nu_i=0``, ``nu_e=0``, hypercollisions off
   * - Operator toggles
     - streaming/mirror/curvature/grad-B/diamagnetic on; nonlinear off
   * - Grid
     - ``Nx=1, Ny=24, Nz=160, y0=20, ntheta=32, nperiod=3``
   * - Velocity resolution
     - ``Nl=6, Nm=16`` (figure generation)
   * - Reference
     - :cite:`Frei22`

Reduced ky scan tables
----------------------

The reduced scan tables below are generated by ``tools/make_tables.py``. The
low-order table provides a quick regression target, while the higher-order
one demonstrates convergence of the Hermite–Laguerre expansion.

Low-order scan (``Nl=2, Nm=4``):

.. csv-table:: Cyclone base case reduced scan (low order)
   :file: _static/cyclone_scan_table_lowres.csv
   :header-rows: 1

Higher-order scan (``Nl=3, Nm=6``):

.. csv-table:: Cyclone base case reduced scan (higher order)
   :file: _static/cyclone_scan_table_highres.csv
   :header-rows: 1

Convergence summary:

.. csv-table:: Cyclone base case convergence check
   :file: _static/cyclone_scan_convergence.csv
   :header-rows: 1

Field-aligned regression
------------------------

We track a reduced :math:`k_y` scan on the field-aligned grid
(``Nx=1, Ny=24, Nz=96, y0=20, ntheta=32, nperiod=2``) with
``Nl=6, Nm=12`` to guard against regressions in geometry, normalization, and
operator assembly:

.. csv-table:: Field-aligned reduced scan
   :file: _static/cyclone_full_operator_scan_table.csv
   :header-rows: 1

Normalization sensitivity
-------------------------

A short scan over ``rho_star`` reports the mean ratios
``|gamma|/gamma_ref`` and ``|omega|/omega_ref`` for the reduced ky subset:

.. csv-table:: rho_star convergence scan
   :file: _static/cyclone_rhostar_convergence.csv
   :header-rows: 1

Reference mismatch tables
-------------------------

The tables below compare the current solver outputs to the digitized reference
datasets, reporting absolute values and relative errors (``rel_*``). These are
regenerated by ``tools/make_tables.py``.

.. csv-table:: Cyclone mismatch table
   :file: _static/cyclone_mismatch_table.csv
   :header-rows: 1

.. csv-table:: Kinetic ITG mismatch table
   :file: _static/kinetic_mismatch_table.csv
   :header-rows: 1

.. csv-table:: ETG mismatch table
   :file: _static/etg_mismatch_table.csv
   :header-rows: 1

.. csv-table:: KBM mismatch table
   :file: _static/kbm_mismatch_table.csv
   :header-rows: 1

.. csv-table:: TEM mismatch table
   :file: _static/tem_mismatch_table.csv
   :header-rows: 1

Reproducibility
---------------

To regenerate the benchmark tables and figures:

.. code-block:: bash

   python tools/make_tables.py
   python tools/make_figures.py

Reference data extraction
-------------------------

The Cyclone and KBM reference CSVs are extracted from external solver outputs
via the helper script:

.. code-block:: bash

   python tools/extract_cyclone_reference.py \
     /path/to/itg_salpha_adiabatic_electrons_correct.out.nc \
     src/spectraxgk/data/cyclone_reference_adiabatic.csv

Update this step only when the reference dataset changes.
