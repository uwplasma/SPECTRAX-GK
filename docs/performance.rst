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
JAX writes the trace under
``<trace-dir>/plugins/profile/<timestamp>/*.trace.json.gz`` together with the
corresponding ``*.xplane.pb`` metadata; the same directory can be opened in
XProf, while the optional ``memory.prof`` snapshot can be inspected with
``pprof`` or XProf's memory tooling.

JAX/XProf operational notes
---------------------------

Two JAX runtime details matter when reading short-run performance numbers:

- JAX's persistent compilation cache can remove repeated recompilation cost for
  fixed signatures. For repeated local profiling runs, set
  ``JAX_COMPILATION_CACHE_DIR`` before the first compilation. This is useful
  for engineering sweeps, but the shipped runtime panel should remain a cold
  end-to-end measurement unless stated otherwise.
- JAX GPU runs preallocate most device memory by default. When diagnosing an
  out-of-memory failure on a shared machine, use
  ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` or a reduced
  ``XLA_PYTHON_CLIENT_MEM_FRACTION`` during the profiling run. Those knobs are
  useful for debugging and tracing, but they should not silently change the
  published benchmark contract.

Recent nonlinear profiling (Cyclone, benchmark-locked config)
-------------------------------------------------------------

Reference run configuration (March 4, 2026):

- ``ky=0.3``, ``Nl=4``, ``Nm=8``
- ``dt=0.01``, ``steps=400``
- ``sample_stride=10``, ``diagnostics_stride=10``
- ``tools/profile_nonlinear_cyclone.py`` with the tracked Cyclone runtime config

CPU profiling (Apple CPU, JAX CPU backend):

.. code-block:: text

   warmup_time_s=117.803
   run_time_s=109.147

GPU profiling (A100-class GPU, JAX CUDA backend):

.. code-block:: text

   warmup_time_s=38.950
   run_time_s=21.350

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

The harness records scalar diagnostics through the compact diagnostics path, so
it measures runtime without materializing mode-resolved history arrays unless a
separate publication artifact explicitly requests them.

To test the optional spectral nonlinear mode (no Laguerre quadrature grid):

.. code-block:: bash

   python tools/benchmark_nonlinear_suite.py --laguerre-mode spectral

You can optionally pass a reference-code log file to compare runtime per step:

.. code-block:: bash

   python tools/benchmark_nonlinear_suite.py --gx-log /path/to/gx_run.out

RHS kernel profile (nonlinear Cyclone)
--------------------------------------

The RHS split profiler measures field solve, nonlinear bracket, linear RHS, and
full RHS kernels after compilation:

.. code-block:: bash

   python tools/profile_nonlinear_step_split.py \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_short.toml \
     --repeats 10 \
     --out docs/_static/nonlinear_rhs_profile_gpu.csv

.. image:: _static/nonlinear_rhs_profile.png
   :alt: SPECTRAX-GK nonlinear RHS kernel profile
   :align: center

The current bounded Cyclone profile separates CPU and ``office`` GPU timings
for default grid-mode and optional spectral-mode nonlinear brackets. The
machine-readable companion ``docs/_static/nonlinear_rhs_profile.json`` records
the dominant measured kernel, kernel fractions relative to the full RHS, and
grid-to-spectral speedups for each backend. The May 9, 2026 refresh used the
same short-case 10-repeat harness on local CPU and one ``office`` RTX A4000
with ``CUDA_VISIBLE_DEVICES=0`` after the linear-RHS fast-path and linked-FFT
refactor tranche. The GPU environment reported
``jax==0.6.2``/``jaxlib==0.6.2``; these are profiler-local hot-path
measurements, not a broad production runtime claim. The refreshed GPU
grid-mode split is:

.. code-block:: text

   field_solve=4.65e-4 s
   nonlinear_bracket=3.36e-3 s
   linear_rhs=6.13e-3 s
   full_rhs=9.66e-3 s

The same GPU profile with ``laguerre_mode="spectral"`` measured
``nonlinear_bracket=1.50e-3 s`` and ``full_rhs=6.38e-3 s``. CPU full-RHS
timings from the same harness were ``1.01e-1 s`` for grid mode and
``7.73e-2 s`` for spectral mode. The short-harness spectral full-RHS ratios
are now ``1.30`` on CPU and ``1.51`` on GPU for this Cyclone case, while the
nonlinear-bracket-only ratios are ``1.54`` on CPU and ``2.24`` on GPU. The
spectral mode therefore remains an opt-in mode guarded by the case-level
parity gate below rather than a global default.

The dominant remaining warm-throughput cost is the compiled linear RHS, with
the nonlinear FFT pipeline still relevant for larger production grids. The next
performance step is to repeat this split on larger benchmark-size cases and
then use profiler traces to decide whether fusion, layout changes, or
production decomposition give the largest verified win.

Benchmark-size Cyclone Miller RHS profile
-----------------------------------------

The larger Cyclone Miller profile uses the shipped nonlinear Miller input with
``Nx=192``, ``Ny=64``, ``Nz=24``, ``Nl=4``, and ``Nm=8``. This is still a
single compiled-RHS split profile rather than a full transport-average runtime
claim, but it is large enough to expose a different bottleneck balance than the
short Cyclone case.

.. code-block:: bash

   python tools/profile_nonlinear_step_split.py \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml \
     --repeats 3 \
     --out docs/_static/nonlinear_rhs_profile_miller_cpu.csv

.. image:: _static/nonlinear_rhs_profile_miller.png
   :alt: SPECTRAX-GK nonlinear RHS kernel profile on the Cyclone Miller benchmark-size case
   :align: center

The matched May 9, 2026 profile measured CPU full-RHS timings of
``2.84e-1 s`` in grid mode and ``2.07e-1 s`` in spectral Laguerre mode. On one
``office`` RTX A4000, the corresponding timings were ``1.48e-2 s`` and
``1.46e-2 s``. Spectral mode reduced the nonlinear bracket by ``1.39x`` on CPU
and ``2.09x`` on GPU, but the GPU full-RHS speedup was only ``1.01x`` because
the compiled linear RHS became comparable to or larger than the bracket. This
points the next optimization pass at linear-RHS fusion/cache layout and then
larger-grid bracket decomposition, not at claiming a broad nonlinear speedup
from spectral mode alone.

The full fused nonlinear-RHS trace companion is generated with:

.. code-block:: bash

   python tools/profile_full_nonlinear_rhs_trace.py \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml \
     --ky 0.3 \
     --Nl 4 \
     --Nm 8 \
     --repeats 3 \
     --summary-json docs/_static/full_nonlinear_rhs_trace_summary.json

The tracked local CPU artifact
``docs/_static/full_nonlinear_rhs_trace_summary.json`` reports
``warm_seconds=2.96e-1`` and ``3345`` HLO lines. The matched one-RTX-A4000
artifact ``docs/_static/full_nonlinear_rhs_trace_gpu_summary.json`` reports
``warm_seconds=1.49e-2`` and ``3338`` HLO lines. The GPU token triage is
dominated by reshapes (``1539``), broadcasts (``1822``), multiplies (``871``),
FFTs (``229``), slices (``215``), and reductions (``132``). This confirms that
the next nonlinear performance tranche should target fused layout and bracket
data movement rather than claiming a new runtime speedup from the linear-RHS
specialization alone. The same tranche removed a duplicated non-Laguerre field
mask from the nonlinear bracket path and added a regression test; the updated
trace confirms this is a code cleanup, not a material HLO-size reduction.

Linear RHS term profile
-----------------------

The linear RHS split profiler drills into the compiled linear contribution used
inside nonlinear runs:

.. code-block:: bash

   python tools/profile_linear_rhs_terms.py \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml \
     --ky 0.3 \
     --Nl 4 \
     --Nm 8 \
     --repeats 8 \
     --out docs/_static/linear_rhs_terms_profile_cpu.csv \
     --summary-json docs/_static/linear_rhs_terms_profile.json

After the zero-collision fast path and linked-FFT refactor, the May 9, 2026
CPU Cyclone artifact reports ``full_linear_rhs=1.08e-1 s`` for the compiled
full linear RHS call in this profiling harness. The independently timed term
kernels sum to ``1.68e-2 s``; this remaining gap is a localization signal, not
a speedup claim, because the full path recomputes the field solve, ``H``
assembly, and all weighted contributions as one compiled graph. The largest
standalone terms are hypercollisions (``2.39e-3 s``), linked ``|k_z|`` setup
(``2.38e-3 s``), and streaming (``2.34e-3 s``). The accepted
zero-collision branch now costs ``1.11e-3 s`` in the standalone CPU timing and
is guarded by the state-window identity gate below.

The active-state CPU companion
``docs/_static/linear_rhs_terms_profile_z_wave_cpu.json`` profiles the same
state after injecting a resolved parallel perturbation. There the hypercollision
and linked ``|k_z|`` norms are both ``2.35e-4`` and the linked ``|k_z|`` path
costs ``2.33e-3 s`` on CPU. This is the artifact that should be used for
linked-``|k_z|`` optimization decisions; the initial-state profile is only a
zero-source baseline.

The matching ``office`` GPU profile is tracked in
``docs/_static/linear_rhs_terms_profile_gpu.json`` and
``docs/_static/linear_rhs_terms_profile_gpu.csv``. On one RTX A4000 with the
same ``jax==0.6.2``/``jaxlib==0.6.2`` environment used for the nonlinear RHS
refresh, it reports ``full_linear_rhs=5.50e-3 s`` and independently timed terms
summing to ``3.41e-3 s``. The accepted zero-collision branch costs
``1.24e-4 s`` in the standalone GPU timing; hypercollisions and linked
``|k_z|`` remain present as separately profiled rows. The active-state GPU
companion
``docs/_static/linear_rhs_terms_profile_z_wave_gpu.json`` activates the same
operator pair with matched norms ``2.35e-4`` and records linked ``|k_z|`` at
``3.63e-4 s`` with ``full_linear_rhs=5.48e-3 s``.

The companion state-window gate is generated with:

.. code-block:: bash

   python tools/gate_linear_rhs_zero_norm_state_window.py \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml \
     --ky 0.3 \
     --Nl 4 \
     --Nm 8 \
     --out-json docs/_static/linear_rhs_zero_norm_state_window_gate.json

The current gate passes by accepting the zero-collision skip for this
``nu=0`` Cyclone window while rejecting a hypercollision skip: the initial
state has zero relative hypercollision skip error, but the resolved
``z``-varying state reaches ``3.59e-3``. This protects the optimization path
from incorrectly disabling linked ``|k_z|`` hypercollisions based only on the
initial-state profile.

Full fused linear RHS trace
---------------------------

The term profiler above times independently isolated kernels. The companion
full-graph profiler lowers and times the fused production linear-RHS assembly
for a real runtime TOML so optimization work can target the compiled graph
rather than only the standalone term calls:

.. code-block:: bash

   python tools/profile_full_linear_rhs_trace.py \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml \
     --ky 0.3 \
     --Nl 4 \
     --Nm 8 \
     --repeats 3 \
     --summary-json docs/_static/full_linear_rhs_trace_summary.json

The initial Cyclone Miller CPU artifact
``docs/_static/full_linear_rhs_trace_summary.json`` reports
``warm_seconds=8.09e-2`` and ``compile_execute_seconds=1.40`` for the bounded
local profile after electrostatic field specialization. The previous
pre-specialization local artifact reported ``warm_seconds=1.19e-1`` and
``compile_execute_seconds=1.94``, so the initial-state CPU profiler shows a
graph-localized improvement of about ``1.47x``. The active-state companion
``docs/_static/full_linear_rhs_trace_z_wave_summary.json`` injects resolved
parallel variation and reports ``warm_seconds=1.29e-1`` with the same
specialized HLO shape. Both current summaries contain ``2225`` HLO lines and
highlight the remaining graph-level pressure points: broadcasts (``748`` coarse
token hits), reshapes (``377``), FFT mentions (``312``), reductions (``304``),
multiplies (``127``), and gathers (``51``). These are localization metrics, not
standalone runtime claims. They point the next source optimization tranche at
fused layout, broadcast/reshape reduction, and linked derivative staging before
changing physics gates or documentation speedup claims.

The matched one-RTX-A4000 artifacts
``docs/_static/full_linear_rhs_trace_gpu_summary.json`` and
``docs/_static/full_linear_rhs_trace_gpu_z_wave_summary.json`` report
``warm_seconds=5.28e-3`` and ``5.25e-3`` for the initial and active ``z_wave``
states, respectively, with ``force_electrostatic_fields=true``. A same-commit
benchmark-size nonlinear split on ``office`` measured GPU full-RHS timings of
``1.71e-2 s`` in grid mode and ``1.48e-2 s`` in spectral Laguerre mode, so the
fresh GPU evidence supports the linear-RHS specialization but does not yet
justify a broader nonlinear speedup claim.

Parallelization scaling (diffrax + distributed linear loop)
-----------------------------------------------------------

The shipped scaling figure is intentionally limited to the release-grade
2-device diffrax speedup sweep. It captures CPU (macOS) and GPU (``office``)
performance for a fixed linear ITG configuration (Ny=64, Nz=128, Nl=6, Nm=6),
comparing one vs two devices across several time horizons. GPU runs use
``sample_stride=5`` to limit memory pressure.

.. image:: _static/scaling_speedup.png
   :alt: SPECTRAX-GK scaling speedup
   :align: center

The raw sweep data lives in ``docs/_static/scaling_speedup_data.csv`` and can
be replotted with:

.. code-block:: bash

   python tools/plot_scaling_speedup.py

The exploratory distributed-RK2 strong-scaling data is still tracked in the CSV
for engineering work, but it is intentionally not presented as a headline
publication figure because the current curve is dominated by communication
overhead rather than near-ideal scaling.

Production parallelization should start with independent work rather than
nonlinear domain decomposition. The public helpers
``spectraxgk.ky_scan_batches`` and ``spectraxgk.batch_map`` split ``k_y``
scans, sensitivity sweeps, and UQ ensembles while preserving serial ordering.
On one device they reduce to batched ``vmap`` execution; on multiple devices
they use JAX device batching and trim padded edge samples deterministically.
Every performance claim from this path should include a numerical-identity
gate against the serial result before a speedup plot is promoted.

The first release-grade gate for this policy is a real Cyclone linear
``k_y``-scan comparison:

.. image:: _static/parallel_ky_scan_gate.png
   :alt: SPECTRAX-GK ky-batch parallelization identity gate
   :align: center

It is regenerated with:

.. code-block:: bash

   python tools/generate_parallel_ky_scan_gate.py

This gate runs the same linear solver serially and with fixed-shape
``k_y`` batching, checks ``gamma`` and ``omega`` numerical identity, and
reports observed speedup as an engineering metric. The gate intentionally does
not claim nonlinear domain scaling; that remains a separate communication and
FFT-decomposition problem.

Fixed-step nonlinear state sharding
-----------------------------------

The fixed-step nonlinear runner now has the same full-state sharding contract
as the linear path for release-gated state axes. Set
``TimeConfig.state_sharding = "auto"`` (or a concrete axis such as ``"ky"`` or
``"kx"``) with ``use_diffrax = false`` to route through
``spectraxgk.integrate_nonlinear_sharded``. The implementation uses a ``pjit``
scan and preserves the serial Runge-Kutta update; it is therefore an
identity-gated state-sharding primitive, not a halo-exchange FFT domain
decomposition claim. Sharding the ``z`` FFT axis is deliberately not exposed as
a release-gated nonlinear runtime path because the current JAX/XLA FFT layout
does not pass the multi-device identity gate.

The profiler/identity artifact is generated with:

.. code-block:: bash

   python tools/profile_nonlinear_sharding.py \
     --sharding auto --sharding-options auto,kx \
     --out-json docs/_static/nonlinear_sharding_profile.json

The JSON records device count, requested sharding axis, warm serial/sharded
timings, profiler-trace status, final-state errors, and the fastest
identity-preserving candidate among the requested state-axis options. The
local checked-in artifact is deliberately small and only establishes the
control-flow and single-device identity gate. The two-GPU office artifact at
``docs/_static/nonlinear_sharding_profile_office_gpu.json`` records active
``auto``/``kx`` state sharding with zero final-state error on both candidate
axes. In the current bounded run the requested ``auto`` path is slower
(``0.86x``), while the best identity-preserving candidate is explicit ``kx``
sharding at about ``1.03x``. That is not enough for a publication speedup
claim. Do not promote new nonlinear runtime speedup claims until this tool is
rerun on matched
benchmark-size CPU and GPU cases and the runtime/memory panel is refreshed.

Spectral nonlinear mode (gated fast toggle)
-------------------------------------------

The spectral nonlinear mode skips Laguerre quadrature for the nonlinear bracket
(``laguerre_nonlinear_mode = "spectral"`` or ``"fast"``). It is not the default
mode because the speedup is case and backend dependent. The release gate runs
the same bounded nonlinear case twice, once with default grid-mode brackets and
once with spectral brackets, then compares end-of-run scalar diagnostics.

.. code-block:: bash

   python tools/gate_laguerre_nonlinear_modes.py \
     --case cyclone --case kbm --case w7x --case hsx \
     --out-json docs/_static/laguerre_mode_gate.json \
     --out-csv docs/_static/laguerre_mode_gate.csv \
     --plot-out docs/_static/laguerre_mode_gate.png

For a GPU reference artifact, run the same command on the target GPU node with
GPU-specific output paths, for example:

.. code-block:: bash

   python tools/gate_laguerre_nonlinear_modes.py \
     --case cyclone --case kbm --case w7x --case hsx \
     --out-json docs/_static/laguerre_mode_gate_gpu.json \
     --out-csv docs/_static/laguerre_mode_gate_gpu.csv \
     --plot-out docs/_static/laguerre_mode_gate_gpu.png

For W7-X/HSX runs, pass ``--w7x-geometry-file`` and
``--hsx-geometry-file`` if the local pre-generated ``*.eik.nc`` files live
outside the default cache paths.

.. image:: _static/laguerre_mode_gate.png
   :alt: SPECTRAX-GK spectral Laguerre nonlinear mode gate on CPU
   :align: center

.. image:: _static/laguerre_mode_gate_gpu.png
   :alt: SPECTRAX-GK spectral Laguerre nonlinear mode gate on GPU
   :align: center

On the bounded local CPU gate, Cyclone, KBM, W7-X, and HSX all passed the
scalar-diagnostic parity threshold with maximum relative differences below
``8.9e-4``. The measured grid/spectral runtime ratios were:

- Cyclone: ``2.90``
- KBM: ``3.31``
- W7-X: ``1.67``
- HSX: ``0.66``

On the bounded ``office`` GPU gate, all four cases also passed with maximum
relative differences below ``2.2e-5``. The measured grid/spectral runtime
ratios were:

- Cyclone: ``1.66``
- KBM: ``2.69``
- W7-X: ``1.63``
- HSX: ``0.74``

Because HSX is slower in both bounded gates, the spectral mode should be
treated as a validated optional engineering mode, not a global fast default.
Production use should rerun the gate on the target case and backend before
claiming speedup.

Runtime and memory comparison workflow
--------------------------------------

For the publication runtime comparison pass, use the manifest-driven runner:

.. code-block:: bash

   python tools/benchmark_runtime_memory.py --list
   python tools/benchmark_runtime_memory.py --dry-run --case cyclone-linear --backend spectrax_cpu
   python tools/benchmark_runtime_memory.py --continue-on-error --log-dir tools_out/runtime_memory_logs

The runner reads ``tools/runtime_memory_manifest.toml`` and writes:

- ``tools_out/runtime_memory_results.csv``
- ``tools_out/runtime_memory_summary.json``
- ``tools_out/runtime_memory_logs/*.stdout.log``
- ``tools_out/runtime_memory_logs/*.stderr.log``
- ``docs/_static/runtime_memory_benchmark.png``

The manifest is designed to hold three rows per case:

- ``spectrax_cpu``
- ``spectrax_gpu``
- ``gx``

Each row may also carry a ``host`` so the same runner can execute local and
remote measurements through one manifest while still collecting wall time and
peak RSS from the target machine.
Rows may also carry a ``profile_command``. When that secondary command succeeds
and prints ``warmup_time_s=...`` / ``run_time_s=...``, the runner merges those
warm measurements back into the same CSV/JSON summary row as the cold pass.
If a profiling command prints ``warmup_time_s=...`` or ``run_time_s=...``, the
runner also records those fields in the CSV/JSON summary so cold and warm JAX
timings can be tracked without a separate sidecar note.

The checked-in case inventory for the current release panel covers the shipped
runtime families:

- Cyclone ITG linear and nonlinear
- ETG linear
- KBM linear and nonlinear
- W7-X linear and nonlinear
- HSX linear and nonlinear
- Cyclone Miller nonlinear

These rows are the ones shown in the README/runtime panel. ETG nonlinear,
KAW, and TEM remain separate tracked work items and are intentionally excluded
from the shipped runtime figure until their release-grade benchmark contracts
are closed.

For the stellarator rows on the `office` benchmark host, the shipped panel
uses pre-generated `*.eik.nc` geometry files instead of live VMEC
regeneration. The GX reference rows on that host also need a consistent local
`netcdf-c` / `hdf5` runtime stack; the default `office` stellarator runtime
environment mixed incompatible HDF5 / NetCDF libraries and lacked the Python
geometry helper dependencies needed for VMEC-driven geometry generation.

Final runtime/memory figure
---------------------------

.. image:: _static/runtime_memory_benchmark.png
   :alt: Runtime and memory comparison across published benchmark cases
   :width: 100%

The runtime subplot uses a log scale because the measured wall times span
roughly three orders of magnitude across the linear, nonlinear, and imported
geometry cases. The memory subplot stays linear because the peak RSS spread is
much narrower.

The assembled figure is generated from the collected per-case summaries with
``tools/benchmark_runtime_memory.py --summary-glob ...`` and written to:

- ``docs/_static/runtime_memory_benchmark.png``
- ``docs/_static/runtime_memory_benchmark.pdf``

For the shipped refresh shown here, use the successful release summary rather
than the older interrupted summary that contains failed W7-X/HSX nonlinear
rows:

.. code-block:: bash

   python tools/benchmark_runtime_memory.py \
     --summary-glob tools_out/runtime_memory_summary_ship_refresh.json \
     --csv-out tools_out/runtime_memory_results_ship_refresh_regenerated.csv \
     --summary-out tools_out/runtime_memory_summary_ship_refresh_regenerated.json \
     --plot-out docs/_static/runtime_memory_benchmark.png

The published runtime figure complements the atlas instead of duplicating it:
the atlas carries growth/frequency and nonlinear transport/energy comparisons,
while the runtime figure carries CPU/GPU/reference wall time and peak RSS for
the shipped runtime cases.

Interpretation of short nonlinear GPU rows
------------------------------------------

The shipped runtime panel reports cold wall time. For the JAX backends, this
includes startup and compilation, so short nonlinear cases can look worse than
their steady-state throughput would suggest.

Targeted ``office`` GPU profiles on the same shipped short nonlinear configs
measured:

.. code-block:: text

   Cyclone nonlinear: warmup_time_s=33.957  run_time_s=15.054
   KBM nonlinear:     warmup_time_s=27.485  run_time_s= 9.725

Compared with the cold runtime panel rows:

- Cyclone nonlinear GPU: ``38.27 s`` in the shipped panel, versus ``15.05 s``
  for the second run on the same compiled executable.
- KBM nonlinear GPU: ``44.33 s`` in the shipped panel, versus ``9.73 s`` for
  the second run on the same compiled executable.

This changes the optimization reading:

- the current short-run Cyclone GPU deficit in the shipped panel is primarily a
  cold-start effect, since the warm run is already faster than the tracked GX
  row,
- the current short-run KBM GPU gap is mostly compile amortization, with warm
  performance already close to GX.

The runtime figure now overlays those warm second-run measurements as hollow
diamond markers on the runtime bars wherever ``run_time_s`` is present in the
summary input.

The highest-value performance work for these short nonlinear lanes is therefore
compile/startup reduction and executable reuse, not just per-step kernel work.

Startup phase profiler
----------------------

For cold-start deep dives, use the dedicated startup profiler:

.. code-block:: bash

   python tools/profile_runtime_startup.py \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml \
     --ky 0.3 --Nl 4 --Nm 8 --compile-steps 1 \
     --json-out tools_out/startup_cyclone_gpu.json \
     --csv-out tools_out/startup_cyclone_gpu.csv

The profiler breaks the cold path into the main setup and first-compile phases:

- runtime config load
- geometry resolution
- grid/default construction
- parameter and term setup
- initial-condition construction
- linear-cache construction
- first field solve compile+execute
- first linear/full RHS compile+execute
- first nonlinear integrator compile+execute

It supports ``--trace-dir`` and ``--memory-profile`` for XProf/Perfetto
inspection with phase-level annotations, and ``--debug-log-cache`` /
``--explain-cache-misses`` for JAX cache diagnostics when a repeated compile
path looks suspicious.
By default the trace tools now start JAX profiling with
``python_tracer_level=0`` and ``host_tracer_level=0``. On the lightweight
``office`` environment this avoids the optional TensorFlow Python-hook import
path, so traces are emitted cleanly without installing TensorFlow just to
silence profiler startup noise.

The current ``office`` GPU startup profiles for the shipped short nonlinear
cases show the same dominant structure:

- Cyclone nonlinear startup total: ``35-36 s`` after the low-rank
  collision-cache and host-cache cleanup passes (previously ``41.47 s`` on the
  earlier office snapshot)
- KBM nonlinear startup total: ``32.23 s``
- dominant phases in both cases:

  - ``compile_first_integrator_run``: about ``22 s`` (Cyclone), ``19.28 s`` (KBM)
  - ``build_linear_cache``: about ``5.6 s`` (Cyclone), ``7.73 s`` (KBM)
  - ``compile_first_linear_rhs`` / ``compile_first_full_rhs``: another
    ``3.0 + 3.0 s`` (Cyclone) or ``1.7 + 1.7 s`` (KBM)

So the next high-value performance work is no longer the analytic geometry
startup path or the collision prefactor path; it is the first compiled
nonlinear integrator path, followed by the remaining Laguerre and drift/cache
construction subphases.

To break the cache-construction lump down further, use:

.. code-block:: bash

   python tools/profile_linear_cache_build.py \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml \
     --Nl 4 --Nm 8 \
     --json-out tools_out/linear_cache_cyclone_gpu.json \
     --csv-out tools_out/linear_cache_cyclone_gpu.csv

The current ``office`` GPU decomposition for the shipped Cyclone short
nonlinear lane is:

- total measured decomposition: ``6.86 s`` after the low-rank collision-cache,
  host-cache, and broadcasted-gyroaverage passes
- dominant subphases:

  - ``gyro_bessel_cache``: ``1.33 s``
  - ``laguerre_cache``: ``1.21 s``
  - ``kperp_and_drifts``: ``0.99 s``
  - ``geometry_coefficients``: ``0.68 s``
  - ``collision_and_damping_cache``: ``0.17 s``

The low-rank collision cache, host-built moment/damping factors, and
broadcasted gyroaverage construction remove the old collision/damping
bottleneck from the cache profile. The overall cold-start wall clock is still
dominated by the first nonlinear integrator compile. The next cache-build
optimization work should therefore focus on Laguerre and drift/cache
construction, while the broader startup campaign should prioritize the first
integrator compile surface.

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
- JAX mesh-based parallelization across multiple devices
- FFT acceleration and layout tuning
- operator fusion for nonlinear terms

Linear-to-nonlinear optimization roadmap
----------------------------------------

The current benchmark runtime gap on CPU is dominated by JAX compile latency and repeated small-shape scan launches. The next implementation phase
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
