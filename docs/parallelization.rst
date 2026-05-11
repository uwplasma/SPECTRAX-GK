Parallelization policy
======================

SPECTRAX-GK parallelization claims are separated by workload class and by the
identity gates that currently exist. Treat this page as the short policy; the
long artifact history remains in :doc:`performance` and runnable examples remain
in :doc:`examples`.

Production path: independent work
---------------------------------

Production-ready parallelism is currently scoped to independent solver calls:

- independent ``k_y`` scans;
- quasilinear calibration grids;
- finite-difference and sensitivity batches;
- UQ and ensemble workloads.

Use ``spectraxgk.ky_scan_batches`` and ``spectraxgk.batch_map`` for JAX-array
workloads, and ``spectraxgk.independent_map`` for file-backed Python tasks.
These helpers preserve serial ordering and restrict communication to result
aggregation. Any timing claim from this path must be paired with a serial
numerical-identity gate for the reported observables, such as ``gamma``,
``omega``, quasilinear weights, or covariance summaries.

Diagnostic path: whole-state nonlinear sharding
-----------------------------------------------

Fixed-step whole-state nonlinear sharding is diagnostic-only. The
``integrate_nonlinear_sharded`` / ``TimeConfig.state_sharding`` path is useful
for control-flow validation, state-axis identity gates, profiler localization,
and testing candidate layouts. It is not a production nonlinear domain
decomposition or multi-GPU speedup claim.

In particular, current whole-state sharding does not close the communication
problem for nonlinear FFTs, halo exchange, conservation checks, or benchmark-size
transport runs. ``z``-axis FFT sharding is not release-gated until it has a
separate communication/layout design and a passing identity gate.

Velocity-space communication gates
----------------------------------

Velocity-space decomposition is gated from the bottom up. The accepted planning
contract is species-first and Hermite-second, with explicit communication flags
for field reductions/broadcasts and Hermite ghost exchange. Each added runtime
path must preserve those contracts before being used for performance claims.

The currently gated communication and call-graph layers are:

- species/Hermite velocity-sharding planner metadata;
- nearest-neighbor Hermite ghost exchange;
- Hermite-sharded field reduction and electrostatic field reduction;
- Hermite streaming-ladder coefficients;
- periodic streaming microkernel and streaming-only linear-RHS call graph;
- electrostatic streaming, drift-slice, diamagnetic-drive, and composed
  single-species periodic electrostatic linear-slices gates.

These gates validate communication and numerical identity for bounded linear or
microkernel paths. They do not validate collisions, linked boundaries,
electromagnetic terms, multi-species nonlinear field solves, nonlinear brackets,
or nonlinear transport speedup unless those paths have their own identity gates
and profiler artifacts.

Claim rules
-----------

Use the following rules when writing docs, release notes, or papers:

- Call independent ``k_y``/UQ/ensemble batching the production-ready
  parallelization path when the serial identity gate is current.
- Call whole-state nonlinear sharding a diagnostic correctness/profiler gate,
  not production nonlinear parallelism.
- Call velocity-space ``shard_map`` work communication-gated and opt-in until
  the relevant full-RHS and workload gates are closed.
- Do not claim nonlinear speedup from sharding, velocity decomposition, spectral
  toggles, or linear-slice profiles without fresh profiler artifacts for the
  exact workload, backend, device count, software stack, and identity tolerance
  being claimed.
- Keep speedup plots separate from identity gates: identity establishes
  correctness; profiler artifacts establish only the scoped timing claim they
  measure.
