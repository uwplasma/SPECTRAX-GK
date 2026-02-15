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
