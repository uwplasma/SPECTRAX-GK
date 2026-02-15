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

Running tests
-------------

.. code-block:: bash

   pytest
