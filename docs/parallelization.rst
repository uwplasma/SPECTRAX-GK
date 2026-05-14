Parallelization policy
======================

SPECTRAX-GK parallelization claims are separated by workload class and by the
identity gates that currently exist. Treat this page as the short policy; the
long artifact history remains in :doc:`performance` and runnable examples remain
in :doc:`examples`.

For release notes and manuscripts, read this page together with
:doc:`release_scope`. Independent scans and ensembles are the current
production path. Whole-state nonlinear sharding and velocity-space
decomposition are correctness/profiler development paths until they pass
workload-specific identity, conservation, and profiler gates.

Strategy registry
-----------------

The metadata API exposes a JSON-friendly strategy table. Release-ready
independent-work rows are intentionally ordered first: ``independent_ky_scan``,
then ``uq_ensemble``.

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 24

   * - ``name``
     - ``readiness``
     - ``independent_work``
     - ``changes_solver_layout``
   * - ``independent_ky_scan``
     - ``release_ready``
     - ``true``
     - ``false``
   * - ``uq_ensemble``
     - ``release_ready``
     - ``true``
     - ``false``
   * - ``whole_state_kx_ky``
     - ``diagnostic``
     - ``false``
     - ``true``
   * - ``velocity_species_hermite``
     - ``diagnostic``
     - ``false``
     - ``true``
   * - ``fft_axis_domain``
     - ``diagnostic``
     - ``false``
     - ``true``

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

For UQ and optimization portfolios, ``spectraxgk.independent_ensemble_provenance_gate``
is the compact production-readiness check. It runs the same member function
serially and through ``independent_map``, verifies numerical identity and result
ordering, checks that oversubscribed worker requests clip to the ensemble size,
reconstructs the deterministic independent-work decomposition, and probes
``IndependentMapExecutionError`` metadata for worker failures. This is a
provenance and identity gate only; it does not make a nonlinear
domain-decomposition speedup claim.

Runtime ``k_y`` scans can request the same independent-worker policy directly
from TOML. This is a scan orchestration path, not a solver-layout sharding path:

.. code-block:: toml

   [parallel]
   strategy = "batch"
   axis = "ky"
   num_devices = 4      # or batch_size = 4
   backend = "auto"     # "thread" or "process" are explicit alternatives

When command-line scan workers are not set explicitly, ``strategy = "batch"``
with ``axis = "ky"`` resolves to independent per-``k_y`` solver calls and
records the resolved worker policy in runtime scan artifacts.

The large tracked artifacts use real solver work rather than synthetic sleeps:
``docs/_static/independent_ky_scan_scaling_large.json`` covers Cyclone linear
``k_y`` scans, and ``docs/_static/quasilinear_uq_ensemble_scaling_large.json``
covers a late-time linear/quasilinear UQ ensemble. These are the figures to cite
for current parallelization speedup claims.

Production closure status
-------------------------

The release status artifact combines the production scaling evidence and the
diagnostic decomposition gates into one machine-readable claim boundary:

.. image:: _static/parallelization_completion_status.png
   :alt: SPECTRAX-GK parallelization closure status
   :align: center

``docs/_static/parallelization_completion_status.json`` reports the release
production-completion percentage and the status of each lane. For the current
tracked artifacts, production independent-work parallelization is closed:
independent ``k_y`` scans reach ``7.18x`` on eight CPU workers and ``1.88x`` on
two RTX A4000 GPUs, while the quasilinear/UQ ensemble reaches ``5.41x`` on CPU
and ``1.71x`` on GPU. The same status now embeds the independent
UQ/optimization provenance gate for serial-vs-parallel ordering, worker
clipping, exception metadata, and deterministic reconstruction. Whole-state
nonlinear sharding and FFT-axis decomposition remain diagnostic, not production
nonlinear speedup claims.

Regenerate the closure status after refreshing any scaling artifact:

.. code-block:: bash

   python tools/build_parallelization_completion_status.py

The lower-level decomposition-contract status is generated separately. It is
useful when editing orchestration code because it checks deterministic shard
assignment, serial reconstruction identity, and claim-level separation without
rerunning large profiles.

.. code-block:: bash

   python tools/build_parallel_decomposition_status.py

.. image:: _static/parallel_decomposition_status.png
   :alt: Parallel decomposition contract status
   :align: center

This status passes for production independent ``k_y`` and UQ portfolios and
for a diagnostic nonlinear state-domain partition. Passing the diagnostic row
does not imply runtime nonlinear domain decomposition: it only proves that the
metadata split/reassemble contract is internally consistent and correctly
scoped as non-production.

Diagnostic path: whole-state nonlinear sharding
-----------------------------------------------

Fixed-step whole-state nonlinear sharding is diagnostic-only. The
``integrate_nonlinear_sharded`` / ``TimeConfig.state_sharding`` path is useful
for control-flow validation, state-axis identity gates, profiler localization,
and testing candidate layouts. It is not a production nonlinear domain
decomposition or multi-GPU speedup claim. Do not use it as evidence for a
whole-state nonlinear sharding speedup; it has no scoped speedup claim until
separate identity gates and matched profiler artifacts exist for that exact
workload.

In particular, current whole-state sharding does not close the communication
problem for nonlinear FFTs, halo exchange, conservation checks, or benchmark-size
transport runs. ``z``-axis FFT sharding is not release-gated until it has a
separate communication/layout design and a passing identity gate.

The large CPU/GPU sweep in
``docs/_static/nonlinear_sharding_strong_scaling_large.json`` confirms the
policy: the final state is identity-correct, but logical-CPU speedup saturates
near ``1.39x`` and the current two-GPU path is slower than one GPU for the
tracked larger fixed-step case. That artifact is therefore valuable engineering
evidence, not a production nonlinear speedup result.

The next decomposition step is also gated, but still diagnostic. The artifact
``docs/_static/nonlinear_domain_parallel_identity_gate.json`` exercises a
deterministic local nonlinear state update with one-cell halo chunks and checks
the decomposed result against the serial update before enabling that prototype
path. This validates the fail-closed identity-gate contract for a bounded local
stencil. The report records the gate name, plan-validity status, and any
explicit blocker reasons such as noncanonical axes, incomplete chunk coverage,
or serial/decomposed shape mismatches; any blocker disables the decomposed
prototype path even if the arrays being compared are numerically equal. The
same JSON now embeds a stricter transport-window sub-gate,
``nonlinear_domain_transport_window_identity``, that advances the serial and
halo-decomposed prototypes over a short fixed-step window and compares final
state identity, boundary identity, mass-trace identity, free-energy-proxy trace
identity, and boundary-flux-proxy trace identity. The drift values in that
sub-gate are serial-vs-decomposed agreement checks for the damped diagnostic
stencil; they are not production conservation claims. The artifact still does
not validate distributed FFTs, field solves, runtime routing, benchmark
transport acceptance, or speedup.

The spectral communication layer now has the same fail-closed treatment. The
artifact ``docs/_static/nonlinear_spectral_communication_identity_gate.json``
uses deterministic complex spectral coefficients in ``(N_l,N_m,N_y,N_x,N_z)``
layout, applies the split/reassemble and axis-transpose operations that a
distributed FFT route would need, and compares three serial observables against
the communicated layout: FFT forward/inverse round trip, pseudo-spectral
nonlinear bracket, and spectral field-solve layout. Passing this gate promotes
``fft_axis_domain`` from blocked to diagnostic. It still does not add runtime
distributed FFT routing, conservation checks, nonlinear transport-window
acceptance, profiler evidence, or any speedup claim.

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
- For runtime scan TOMLs, use ``[parallel] strategy = "batch"`` with
  ``axis = "ky"`` only for independent ``k_y`` scan orchestration.
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

Large-run scaling acceptance checklist
--------------------------------------

A CPU/GPU strong-scaling result is release-ready only when the tracked
artifacts satisfy all of the following:

- the combined ``*_large`` JSON/CSV/PNG/PDF files point back to split CPU and
  GPU source artifacts for the same workload family;
- every split artifact records the actual problem size, backend, requested
  device counts, warmup/repeat policy, and positive per-worker or per-profile
  timing samples;
- every row has ``identity_gate_pass = true`` and compares against the
  one-worker or one-device serial reference for the observable being claimed;
- nonlinear whole-state sharding rows embed the per-device profiler/profile
  payload, including trace-request status, serial timing stats, sharded timing
  stats, selected axis, and final-state error metrics;
- any speedup wording names the exact backend, device count, workload, grid,
  software stack, identity tolerance, and artifact files that produced it.

If any item is missing, the result can be kept as local engineering evidence
only. In particular, whole-state nonlinear sharding remains not a production
nonlinear speedup claim, even when the embedded profile reports a positive
engineering timing ratio. Promoting that lane requires fresh profiler artifacts
for the exact workload plus full nonlinear identity, conservation, field-solve,
FFT/bracket communication, and transport-window gates.

Fast artifact contract check
----------------------------

Before editing scaling docs or manifests, run the checked-in artifact contract:

.. code-block:: bash

   python tools/check_parallel_scaling_artifacts.py

This command does not rerun large profiles and does not enforce any minimum
speedup. It validates that the tracked JSON/CSV/PNG/PDF sidecars exist, the
``parallel_scaling`` manifest lists them, split CPU/GPU source artifacts are
attached where required, numerical identity gates pass, error fields are finite,
and timing/profiler payloads are positive and scoped to their documented claim.

Release artifact policy
-----------------------

The release-gated parallelization artifacts are grouped by what they are
allowed to support:

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Artifact family
     - Primary files
     - Claim allowed
     - Claim not allowed
   * - Independent ``k_y`` scans
     - ``independent_ky_scan_scaling_large.{json,csv,png,pdf}``
     - Production parallelization for independent linear scans when
       ``gamma``/``omega`` identity is current.
     - Nonlinear domain decomposition or nonlinear transport speedup.
   * - Quasilinear/UQ ensembles
     - ``quasilinear_uq_ensemble_scaling_large.{json,csv,png,pdf}``
     - Production batching for independent reduced-feature and UQ workloads.
     - Promoted absolute nonlinear heat-flux prediction.
   * - Whole-state nonlinear sharding
     - ``nonlinear_sharding_strong_scaling_large.{json,csv,png,pdf}``
     - Correctness and profiler-direction evidence for the current ``pjit``
       state-axis layout.
     - Production nonlinear multi-GPU speedup.
   * - Prototype nonlinear state-domain gate
     - ``nonlinear_domain_parallel_identity_gate.{json,png}``
     - Fail-closed serial-vs-halo-decomposed identity evidence for one bounded
       local stencil, including the embedded transport-window proxy traces.
     - Distributed FFT, field-solve, production conservation, transport-runtime,
       or speedup claims.
   * - Prototype nonlinear spectral communication gate
     - ``nonlinear_spectral_communication_identity_gate.{json,png}``
     - Fail-closed split/reassemble identity evidence for FFT round trip,
       pseudo-spectral bracket, and spectral field-solve layout.
     - Runtime distributed FFT routing, nonlinear conservation,
       transport-window, or speedup claims.
   * - Velocity-space linear slices
     - ``linear_rhs_parallel_slices_sweep.{json,png,pdf}``
     - Bounded engineering evidence for opt-in electrostatic linear RHS slices.
     - Electromagnetic, linked-boundary, collision, or nonlinear speedup.

Both ``tools/performance_optimization_manifest.toml`` and
``tools/validation_coverage_manifest.toml`` list these artifacts explicitly.
The tests require the manifests, files, and claim scopes to stay synchronized,
so deleting or silently reinterpreting a scaling artifact fails the fast
parallelization gate.
