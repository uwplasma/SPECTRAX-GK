SPECTRAX-GK
===========

SPECTRAX-GK is a JAX-native gyrokinetic solver using Hermite-Laguerre velocity
space, Fourier perpendicular coordinates, and field-aligned flux-tube geometry.

Documentation map
-----------------

The documentation is organized so that the core physics, equations, numerical
methods, and model-specific paths are easy to locate:

- :doc:`theory` summarizes the gyrokinetic ordering, field equations, and
  flux-tube assumptions.
- :doc:`linear_model` gives the operator-level derivation tied directly to the
  implemented linear equations and diagnostics.
- :doc:`operators` lists every implemented term, collision model,
  hyperdiffusion/hypercollision control, and the runtime parameters that select
  them.
- :doc:`numerics` documents discretization, time integration, FFT brackets,
  solver contracts, and JAX parallelization.
- :doc:`geometry` and :doc:`inputs` cover the supported model paths
  (analytic s-alpha, Miller, VMEC/imported geometry, slab) together with the
  TOML schema used by the executable and Python drivers.

For a first technical read, start with :doc:`theory`, then move to
:doc:`operators` and :doc:`numerics`. That path gives the governing equations,
the term-by-term implementation contract, and the numerical approximations in
the same order they appear in the code.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   inputs
   outputs
   theory
   linear_model
   operators
   numerics
   normalization
   geometry
   algorithms
   solvers
   architecture
   benchmarks
   examples
   codes
   performance
   testing
   references
   roadmap
   api
