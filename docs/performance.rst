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

Nonlinear profiling
-------------------

For end-to-end nonlinear performance, use the dedicated Cyclone profiling
driver. It supports Perfetto traces, XLA HLO dumps, and memory snapshots.

.. code-block:: bash

   python tools/profile_nonlinear_cyclone.py \
     --trace-dir /tmp/spectrax_nl_trace \
     --xla-dump-dir /tmp/spectrax_nl_xla \
     --steps 400 --dt 0.0377 --Nl 4 --Nm 8

The trace directory can be opened with Perfetto. For GPU profiling, set
``JAX_PLATFORM_NAME=gpu`` before invoking the script.

Recent nonlinear profiling (Cyclone, GX-balanced config)
--------------------------------------------------------

Reference run configuration:

- ``ky=0.3``, ``Nl=4``, ``Nm=8``
- ``dt=0.03769945``, ``steps=200``
- ``examples/configs/runtime_cyclone_nonlinear_gx.toml``

CPU profiling (Apple CPU, JAX CPU backend):

.. code-block:: text

   warmup_time_s=146.594
   run_time_s=137.604

GPU profiling (A100-class GPU, JAX CUDA backend):

.. code-block:: text

   warmup_time_s=50.212
   run_time_s=30.467

HLO summary (``jit_scan.*_after_optimizations``):

- CPU: ``fft=623``, ``scatter=72``, ``gather=375``, ``dot=88``, ``fusion=1053``
- GPU: ``fft=440``, ``scatter=30``, ``gather=322``, ``dot=44``, ``fusion=831``

The nonlinear RHS remains FFT-heavy with nontrivial gather/scatter density.
Primary optimization targets are the FFT pipeline (channel stacking, reuse of
real-space gradients) and scatter removal in linked-FFT paths.

GPU memory report (jit_scan module):

- Total bytes used: ``228.21 MiB`` (XLA memory usage report).

Nonlinear benchmark harness
---------------------------

To capture per-step runtime and end-of-run diagnostics, use the nonlinear
benchmark harness:

.. code-block:: bash

   python tools/benchmark_nonlinear_suite.py --steps 200 --dt 0.0377 \
     --out /tmp/spectrax_nl_bench.csv

To test the faster spectral nonlinear mode (no Laguerre quadrature grid):

.. code-block:: bash

   python tools/benchmark_nonlinear_suite.py --laguerre-mode spectral

You can optionally pass a GX log file to compare runtime per step:

.. code-block:: bash

   python tools/benchmark_nonlinear_suite.py --gx-log /path/to/gx_run.out

Runtime comparison (nonlinear Cyclone)
--------------------------------------

Using the benchmark harness (same config as above), the per-step runtime is:

.. code-block:: text

   SPECTRAX CPU: 0.67704 s / step
   SPECTRAX GPU: 0.11375 s / step
   GX (A100):    0.016576 s / step

This puts SPECTRAX at ~6.9x slower than GX on GPU and ~40.8x slower on CPU for
the same nonlinear Cyclone base case configuration. The gap is dominated by the
nonlinear FFT pipeline and residual gather/scatter operations.

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
- ``hermite-line``: Hermite streaming line solve (tridiagonal in ``m`` at fixed :math:`k_z`)
- ``hermite-line-coarse``: Hermite line solve + kx-coarse correction

.. code-block:: bash

   python tools/profile_gmres_precond.py

On the reference run (Nl=2, Nm=3, Ny=4, Nz=8), this reported:

.. code-block:: text

   iters_plain=6
   iters_diag=6
   iters_pas=6
   iters_pas_coarse=6
   iters_hermite_line=4
   iters_hermite_line_coarse=4

On a larger run (Ny=8, Nz=64, Nl=12, Nm=12, dt=0.1), this reported:

.. code-block:: text

   iters_plain=38
   iters_diag=38
   iters_pas=39
   iters_pas_coarse=39
   iters_hermite_line=22
   iters_hermite_line_coarse=22

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

Linear-to-nonlinear optimization roadmap
----------------------------------------

The current GS2 vs SPECTRAX runtime gap on CPU is dominated by JAX compile
latency and repeated small-shape scan launches. The next implementation phase
targets both linear and nonlinear performance with a single operator strategy:

1. **Compile-once scan kernels**

   - enforce fixed batch shapes across ``ky`` and ``beta`` scans,
   - pre-JIT a small set of canonical ``(Nl, Nm, Ny, Nz)`` signatures,
   - cache compiled executables on disk for repeated benchmark sweeps.

2. **Operator fusion in RHS assembly**

   - merge streaming/mirror/curvature/grad-B stencils into one fused kernel,
   - remove scatter-heavy intermediate writes,
   - keep field coupling and species sums contiguous in memory.

3. **Matrix-free eigen path as default for linear scans**

   - use Krylov/shift-invert for scan tables and figures,
   - reserve long time integration for spot-check diagnostics only.

4. **Preconditioner reuse**

   - persist Hermite-line and shift-invert preconditioner structures across
     neighboring scan points (same geometry/grid),
   - reuse Jacobian-like linearization objects in IMEX stages.

5. **Streaming diagnostics by default**

   - avoid storing full time traces unless explicitly requested,
   - compute growth/frequency online from selected mode signals.

These steps are chosen to carry directly into nonlinear runs, where the same
fused RHS, scan batching, and preconditioner reuse will dominate throughput and
memory behavior.
