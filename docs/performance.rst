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

   uncached_s=0.000426
   cached_s=0.000455
   speedup=0.94x

The exact speedup depends on hardware and problem size. As more geometry and
operator terms are cached (cv/gb/bgrad, hyper ratios), the overhead balance may
shift; in this run the cached path was roughly cost-neutral.

Cached basis indices
--------------------

To reduce per-step overhead, the linear cache now stores Laguerre/Hermite index
arrays (:math:`l`, :math:`m`) and derived coefficients (``l+1``, ``m+1``,
``sqrt(m)``, ``sqrt(m+1)``). These are reused inside the mirror/curvature
terms and the implicit preconditioner instead of re-allocating on every RHS
call. The change is small in absolute cost for low-order runs, but becomes
noticeable in higher-order scans and tight profiling loops.

GMRES preconditioner iterations
--------------------------------

For the implicit linear solver, we include a small iteration-count harness that
solves a reduced system and compares the GMRES iteration count with multiple
preconditioners:

- ``diag``: full diagonal (damping + drift + mirror)
- ``pas``: PAS line preconditioner (streaming + diagonal damping/drifts)
- ``pas-coarse``: line + kx-coarse additive correction (Schur-style)

.. code-block:: bash

   python tools/profile_gmres_precond.py

On the reference run (Nl=2, Nm=3, Ny=4, Nz=8), this reported:

.. code-block:: text

   iters_plain=10
   iters_diag=10
   iters_pas=11
   iters_pas_coarse=11

On a larger run (Ny=8, Nz=64, Nl=12, Nm=12, dt=0.1), this reported:

.. code-block:: text

   iters_plain=38
   iters_diag=38
   iters_pas=39
   iters_pas_coarse=39

JIT considerations
------------------

The linear integrator is ``jit``-compiled with the number of steps and method
as static arguments. The operator term switches (:class:`spectraxgk.linear.LinearTerms`)
should also remain static inside a compiled loop to avoid recompilation. The
cached operator arrays can be constructed once and reused across multiple runs
to avoid repeated geometry setup costs.

Planned optimizations
---------------------

- ``vmap`` over species and parameter scans
- ``pjit``/sharding for multi-device acceleration
- FFT acceleration and layout tuning
- operator fusion for nonlinear terms
