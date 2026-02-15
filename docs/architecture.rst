Architecture
============

Core modules
------------

- ``spectraxgk.basis``: Hermite and Laguerre basis functions.
- ``spectraxgk.gyroaverage``: gyroaverage coefficients and polarization helpers.
- ``spectraxgk.geometry``: analytic s-alpha flux-tube geometry.
- ``spectraxgk.linear``: linear electrostatic operator and JAX time integrator.
- ``spectraxgk.benchmarks``: reference datasets and benchmark harness.
- ``spectraxgk.plotting``: reusable, publication-ready plotting utilities.

Data flow
---------

The linear solve is structured as:

1. build the spectral grid and geometry
2. compute gyroaverage coefficients
3. solve quasineutrality
4. build the gyrokinetic variable ``H``
5. apply the Hermite streaming operator
6. advance in time using ``integrate_linear``

This structure is intentionally modular so that nonlinear terms, collisions,
geometry adapters, and electromagnetic extensions can be inserted with minimal
refactoring.
