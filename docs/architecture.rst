Architecture
============

Core modules
------------

- ``spectraxgk.basis``: Hermite and Laguerre basis functions.
- ``spectraxgk.gyroaverage``: gyroaverage coefficients and polarization helpers.
- ``spectraxgk.geometry``: analytic s-alpha flux-tube geometry.
- ``spectraxgk.terms``: term-wise RHS kernels (streaming, mirror, drifts, drive, collisions, fields).
- ``spectraxgk.linear``: cache construction, public linear API, and integrators that call modular RHS assembly.
- ``spectraxgk.nonlinear``: nonlinear runtime integrators and cached IMEX paths.
- ``spectraxgk.runtime`` / ``spectraxgk.runtime_config``: user-facing runtime entrypoints and configuration schema.
- ``spectraxgk.benchmarks``: reference datasets and benchmark harness.
- ``spectraxgk.plotting``: reusable, publication-ready plotting utilities.

Term-level source mapping
-------------------------

- streaming, mirror, curvature, grad-B, diamagnetic, collisions,
  hypercollisions, hyperdiffusion, end damping:
  ``src/spectraxgk/terms/linear_terms.py``
- field solves:
  ``src/spectraxgk/terms/fields.py``
- nonlinear E×B, flutter, and Bessel-grid transforms:
  ``src/spectraxgk/terms/nonlinear.py``
- assembled RHS:
  ``src/spectraxgk/terms/assembly.py``

For the full operator equations, see :doc:`operators`.

Data flow
---------

The linear solve is structured as:

1. build the spectral grid and geometry
2. compute gyroaverage coefficients
3. convert ``LinearTerms`` into one canonical ``TermConfig``
4. solve the field equations for :math:`(\\phi, B_\\parallel, A_\\parallel)`
5. build the gyrokinetic variable ``H``
6. assemble RHS by summing per-term kernels from ``spectraxgk.terms``
7. advance in time using ``integrate_linear``/diffrax/Krylov with the same RHS

This structure is intentionally modular so that nonlinear terms, collisions,
geometry adapters, and electromagnetic extensions can be inserted with minimal
refactoring.
