Testing
=======

Testing philosophy
------------------

SPECTRAX-GK enforces **100% test coverage** and requires physics-based checks
for each numerical component. The test suite is designed to be:

- **pedagogic**: each test explains the concept being validated
- **deterministic**: no stochastic outcomes or tolerance drift
- **future-proof**: targeted at invariants and well-posed regressions

Test categories
---------------

- **Basis tests**: orthonormality and recurrence checks.
- **Operator tests**: Hermite ladder streaming and mode extraction.
- **Benchmark tests**: loading reference data and growth-rate fitting.
- **Physics sanity checks**: conservation properties under simplified limits.

Unit tests (numerical invariants)
---------------------------------

Representative unit checks include:

- **Hermite/Laguerre ladder identities**:
  :func:`spectraxgk.linear.apply_hermite_v`,
  :func:`spectraxgk.linear.apply_laguerre_x`.
- **Quasineutrality consistency**:
  :func:`spectraxgk.linear.quasineutrality_phi`.
- **Streaming term validation**:
  :func:`spectraxgk.linear.grad_z_periodic`,
  :func:`spectraxgk.linear.streaming_term`.
- **Growth-rate fitting windows**:
  :func:`spectraxgk.analysis.select_fit_window`,
  :func:`spectraxgk.analysis.fit_growth_rate_auto`.
- **Grid construction and normalization**:
  :func:`spectraxgk.grids.build_spectral_grid`.

These tests live in ``tests/test_linear.py`` and ``tests/test_grids.py`` and
are designed to fail deterministically if a discretization or normalization
changes.

Physics regression tests
------------------------

The physics-focused tests exercise reduced or symmetry limits that should
remain invariant across refactors:

- **Drive-off equivalence**: the full drift/mirror operator matches the
  energy-closure when drift/drive terms are disabled.
- **Full-operator alias**: ``operator="full"`` is exercised in both the cached
  RHS and cached integrator paths to ensure the alias remains valid.
- **Mirror/curvature activation**: nonzero drift terms create nonzero response
  when streaming and drive are turned off.
- **Diamagnetic drive structure**: the energy-weighted drive produces a
  nonzero response when gradients are enabled and vanishes at :math:`k_y=0`.
- **Normalization scaling**: ``rho_star`` rescales the cached :math:`k_y`
  values exactly.

These checks are in ``tests/test_linear.py`` and are meant to be future-proof
physics invariants.

Benchmark regression tests
--------------------------

Benchmark regression tests validate the Cyclone base case reference dataset and
growth-rate extraction pipeline:

- Loading the reference CSV via :func:`spectraxgk.benchmarks.load_cyclone_reference`.
- Running short linear scans via :func:`spectraxgk.benchmarks.run_cyclone_linear`
  and :func:`spectraxgk.benchmarks.run_cyclone_scan`.
- Full-operator regression with relaxed tolerances for the reduced ky scan.

These tests live in ``tests/test_benchmarks.py`` and ``tests/test_full_operator.py``.

Planned linear physics checks
-----------------------------

Before nonlinear validation, we will add linear physics checks grounded in
published benchmarks:

- **ITG/Cyclone base case**: reproduce the standard Cyclone base case growth
  rates and frequencies across a reduced ky scan. [Dimits00]_ [Lin99]_
- **ETG linear instability**: verify growth-rate trends with
  :math:`R/L_{Te}` and compare against published ETG turbulence studies. [Dorland00]_ [Jenko00]_
- **Microtearing (MTM)**: verify that MTMs are driven by electron temperature
  gradients and magnetic drifts by comparing against published dispersion
  relations. [Chandran24]_

Running tests
-------------

.. code-block:: bash

   pytest
