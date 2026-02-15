Performance
===========

JAX performance model
---------------------

SPECTRAX-GK uses JAX to compile array kernels ahead of time, enabling
vectorized, accelerator-ready performance while retaining automatic
differentiation. The linear operator and time integrator are designed to be
``jit``-friendly and to avoid Python-side loops in performance-critical paths.

The linear solver precomputes geometry-dependent arrays (gyroaverage
coefficients, drift components, mirror term, and zero-mode masks) in a ``LinearCache`` to
avoid recomputing them at each time step. This cache is reused inside the JIT
compiled integrator.

Cache profiling
---------------

We include a small timing harness that compares cached and uncached RHS
evaluation on a modest grid:

.. code-block:: bash

   python tools/profile_linear_cache.py

On a reference CPU run (Nx=Ny=16, Nz=32, Nl=2, Nm=4), this reported:

.. code-block:: text

   uncached_s=0.000461
   cached_s=0.000473
   speedup=0.97x

The exact speedup depends on hardware and problem size. As more geometry and
operator terms are cached (cv/gb/bgrad, hyper ratios), the overhead balance may
shift; in this run the cached path was roughly cost-neutral.

Planned optimizations
---------------------

- ``vmap`` over species and parameter scans
- ``pjit``/sharding for multi-device acceleration
- FFT acceleration and layout tuning
- operator fusion for nonlinear terms
