Performance
===========

JAX performance model
---------------------

SPECTRAX-GK uses JAX to compile array kernels ahead of time, enabling
vectorized, accelerator-ready performance while retaining automatic
differentiation. The linear operator and time integrator are designed to be
``jit``-friendly and to avoid Python-side loops in performance-critical paths.

Planned optimizations
---------------------

- ``vmap`` over species and parameter scans
- ``pjit``/sharding for multi-device acceleration
- FFT acceleration and layout tuning
- operator fusion for nonlinear terms
