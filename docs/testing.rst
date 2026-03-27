Testing
=======

Testing philosophy
------------------

SPECTRAX-GK enforces high coverage on critical solver modules and requires
physics-based checks for each numerical component. The test suite is designed
to be:

- **pedagogic**: each test explains the concept being validated
- **deterministic**: no stochastic outcomes or tolerance drift
- **future-proof**: targeted at invariants and well-posed regressions

Test categories
---------------

- **Basis tests**: orthonormality and recurrence checks.
- **Operator tests**: Hermite ladder streaming and mode extraction.
- **Benchmark tests**: loading reference data and growth-rate fitting.
- **Physics sanity checks**: conservation properties under simplified limits.

Unit tests (numerical invariants)
---------------------------------

Representative unit checks include:

- **Hermite/Laguerre ladder identities**:
  :func:`spectraxgk.linear.apply_hermite_v`,
  :func:`spectraxgk.linear.apply_laguerre_x`.
- **Quasineutrality consistency**:
  :func:`spectraxgk.linear.quasineutrality_phi`.
- **Streaming term validation**:
  :func:`spectraxgk.linear.grad_z_periodic`,
  :func:`spectraxgk.linear.streaming_term`.
- **Growth-rate fitting windows**:
  :func:`spectraxgk.analysis.select_fit_window`,
  :func:`spectraxgk.analysis.fit_growth_rate_auto`.
- **Grid construction and normalization**:
  :func:`spectraxgk.grids.build_spectral_grid`.
- **Normalization contract consistency**:
  :func:`spectraxgk.normalization.get_normalization_contract`,
  :func:`spectraxgk.normalization.apply_diagnostic_normalization`.
- **Modular RHS equivalence**:
  :func:`spectraxgk.linear.linear_terms_to_term_config`,
  :func:`spectraxgk.terms.assemble_rhs_cached`,
  :func:`spectraxgk.linear.linear_rhs_cached`.

These tests live in ``tests/test_linear.py`` and ``tests/test_grids.py`` and
``tests/test_normalization.py`` and ``tests/test_terms_assembly.py`` and are
designed to fail deterministically if a discretization, assembly path, or
normalization changes.

Physics regression tests
------------------------

The physics-focused tests exercise reduced or symmetry limits that should
remain invariant across refactors:

- **Term toggles**: :class:`spectraxgk.linear.LinearTerms` switches individual
  operator components without changing the equation structure.
- **Mirror/curvature activation**: nonzero drift terms create nonzero response
  when streaming and drive are turned off.
- **Diamagnetic drive structure**: the energy-weighted drive produces a
  nonzero response when gradients are enabled and vanishes at :math:`k_y=0`.
- **Normalization scaling**: ``rho_star`` rescales the cached :math:`k_y`
  values exactly.
- **End-cap damping**: the linked-boundary taper only affects :math:`k_y>0`
  modes and vanishes when ``damp_ends_amp = 0``.

These checks are in ``tests/test_linear.py`` and are meant to be future-proof
physics invariants.

Benchmark regression tests
--------------------------

Benchmark regression tests validate the Cyclone base case reference dataset and
growth-rate extraction pipeline:

- Loading the reference CSV via :func:`spectraxgk.benchmarks.load_cyclone_reference`.
- Running short linear scans via :func:`spectraxgk.benchmarks.run_cyclone_linear`
  and :func:`spectraxgk.benchmarks.run_cyclone_scan`.
- Reduced ky regression with tightened tolerances on the field-aligned grid.

These tests live in ``tests/test_benchmarks.py`` and ``tests/test_full_operator.py``.

Diffrax and nonlinear smoke tests
---------------------------------

Diffrax integration and the nonlinear driver are exercised with fast smoke
tests:

- ``tests/test_diffrax_integrators.py`` runs explicit and IMEX diffrax solvers
  on tiny grids.
- ``tests/test_diffrax_integrators_core.py`` hardens branch coverage for
  diffrax helper paths (solver selection, save modes, streaming fits, IMEX
  branches, sharding, and validation errors).
- ``tests/test_linear_krylov_core.py`` hardens matrix-free Krylov internals
  (mode-family targeting, shift-invert preconditioner selection, fallback
  policy, and dominant eigenpair wrappers).
- ``tests/test_example_smoke.py`` verifies the config-driven runner (diffrax
  enabled) and a short nonlinear scan with placeholder nonlinear terms.
- ``tests/test_nonlinear_exb.py`` exercises the nonlinear bracket sign,
  real-FFT path, flutter coupling, and EM toggle behavior.
- ``tests/test_runtime_config.py`` and ``tests/test_runtime_runner.py`` verify
  unified runtime TOML loading and case-agnostic linear runs (Cyclone/ETG/KBM)
  through the same solver path.

Linear physics checks
---------------------

Before nonlinear validation, we exercise linear physics checks grounded in
published benchmarks and trend tests:

- **ITG/Cyclone base case**: reproduce the standard Cyclone base case growth
  rates and frequencies across a reduced ky scan. [Dimits00]_ [Lin99]_
- **GX term-by-term audit**: use the term-dump tooling to compare SPECTRAX-GK
  streaming and linear-kernel RHS components against GX for a single Cyclone
  state (see ``tools/dump_rhs_terms.py`` and ``tools/compare_gx_rhs_terms.py``).
- **GX nonlinear term audit (KBM/Cyclone)**: compare nonlinear
  derivative, bracket, electromagnetic split, and total RHS dumps using
  ``tools/compare_gx_nonlinear_terms.py``. The tool supports GX dump folders
  with ``nl_apar.bin``/``nl_bpar.bin`` and can infer shape metadata when
  ``rhs_terms_shape.txt`` is absent.
- **ETG linear instability**: verify that growth rates remain positive across
  reduced electron-scale gradients and that the real frequency follows the
  electron diamagnetic direction. [Dorland00]_ [Jenko00]_
- **KBM beta scan**: verify the transition between ITG-like and KBM branches
  in a fixed-:math:`k_y` beta sweep, with GS2 as the primary electromagnetic
  cross-code reference.

Running tests
-------------

.. code-block:: bash

   pytest

Stress-matrix parity gates
--------------------------

In addition to unit/regression tests, SPECTRAX-GK includes a small set of
"stress-matrix" gates meant to catch parity regressions early (before tracked
benchmark figures move):

- **Restart parity**: ``tests/test_restart_gate.py`` verifies that a nonlinear
  run resumed from a GX-compatible binary restart file reproduces the same
  final state as a continuous run.
- **CPU/GPU short-window parity** (optional): ``tests/test_device_parity_gate.py``
  compares a short nonlinear trajectory norm on CPU vs GPU. Enable explicitly:

  .. code-block:: bash

     SPECTRAXGK_DEVICE_PARITY=1 pytest -q tests/test_device_parity_gate.py

- **VMEC roundtrip determinism** (optional): ``tests/test_vmec_roundtrip_gate.py``
  regenerates an ``*.eik.nc`` from a provided VMEC file twice and asserts the
  imported geometry arrays are bitwise identical. Enable explicitly:

  .. code-block:: bash

     SPECTRAXGK_VMEC_FILE=/path/to/wout.nc pytest -q tests/test_vmec_roundtrip_gate.py

For developer workflows that require local GX benchmark NetCDFs or GX dump
artifacts, use:

- ``tools/run_gx_linear_stress_matrix.py`` (KAW, Cyclone kinetic electrons, KBM Miller)
- ``tools/run_imported_linear_targeted_audit.py`` (generic per-``ky`` targeted imported-linear wrapper)
- ``tools/compare_gx_imported_window.py`` (exact imported-linear one-window replay against GX ``diag_state`` dumps)
- ``tools/run_kbm_lowky_extractor_audit.py`` (direct cached-trajectory KBM low-``ky`` extractor audit)
- ``tools/run_exact_state_audit.py`` (manifest-driven wrapper around the exact-state audit tools)
- ``tools/run_restart_parity_gate.py`` (manifest-driven nonlinear restart/continuation parity gate)
- ``tools/run_device_parity_gate.py`` (manifest-driven CPU/GPU short-window parity gate)
- ``tools/run_vmec_roundtrip_gate.py`` (manifest-driven VMEC ``vmec -> eik.nc`` determinism gate)

The targeted imported-linear wrapper and the underlying
``compare_gx_imported_linear.py`` comparator now support two important controls
for honest stress-lane scoring without changing the default full-window
behavior:

- ``--sample-step-stride``: subsample the saved GX diagnostic sample indices
  before scoring.
- ``--max-samples``: truncate scoring to the first N selected samples.

The lower-level comparator also supports ``--cache-dir`` plus ``--reuse-cache``
to persist per-``ky`` trajectory/result arrays (``gamma``, ``omega``,
``Wg``, ``Wphi``, ``Wapar``) as compressed ``.npz`` files keyed by the actual
GX file, geometry file, GX input, selected ``ky``, Hermite/Laguerre
resolution, mode selector, and sample-window contract. This makes the
stress-lane tooling incremental instead of rerunning a full lane every time.

For VMEC-backed exact-state audits, the runtime bridge now prefers a local
``booz_xform_jax`` checkout and injects a temporary ``booz_xform`` compatibility
shim only into the GX geometry-helper subprocess. This preserves GX as ground
truth while avoiding a host-level dependency on the original ``booz_xform``
Python package.

The bridge auto-discovers ``booz_xform_jax`` in standard locations
(``/Users/rogeriojorge/local/booz_xform_jax`` and ``/home/rjorge/booz_xform_jax``)
or from ``GX_BOOZ_XFORM_JAX_PATH`` / ``BOOZ_XFORM_JAX_PATH``. When a specific
Python is still needed for GX's helper, it can be provided through
``geometry.gx_python`` or ``GX_VMEC_PYTHON``. On ``office``, the normal audited
path is:

.. code-block:: bash

   rsync -a /path/to/booz_xform_jax/ office:/home/rjorge/booz_xform_jax/
   HSX_VMEC_FILE=/path/to/wout_HSX_QHS_vac.nc \
   /home/rjorge/venvs/spectrax/bin/python tools/run_exact_state_audit.py \
     --manifest tools/exact_state_lanes.office.toml \
     --outdir tools_out/exact_state_audit_office

The restart/continuation gate uses the same environment model and should be
run against the tracked nonlinear lanes with ``PYTHONPATH`` set to the source
tree so the office venv does not pick up a stale installed package:

.. code-block:: bash

   PYTHONPATH=/home/rjorge/SPECTRAX-GK/src \
   /home/rjorge/venvs/spectrax/bin/python tools/run_restart_parity_gate.py \
     --manifest tools/restart_gate_lanes.office.toml \
     --outdir tools_out/restart_parity_office

The device-parity gate now has audited ``office`` manifests for one tokamak and
one stellarator lane, both requiring stable nonzero outputs rather than the
older zero-norm smoke probe:

.. code-block:: bash

   PYTHONPATH=/home/rjorge/SPECTRAX-GK/src \
   /home/rjorge/venvs/spectrax/bin/python tools/run_device_parity_gate.py \
     --manifest tools/device_parity_lanes.office.toml \
     --outdir tools_out/device_parity_office

The VMEC roundtrip gate uses the same manifest pattern and currently covers the
tracked W7-X and HSX VMEC lanes:

.. code-block:: bash

   PYTHONPATH=/home/rjorge/SPECTRAX-GK/src \
   /home/rjorge/venvs/spectrax/bin/python tools/run_vmec_roundtrip_gate.py \
     --manifest tools/vmec_roundtrip_lanes.office.toml \
     --outdir tools_out/vmec_roundtrip_office

If the helper must be forced to another interpreter, the fallback remains:

.. code-block:: bash

   GX_VMEC_PYTHON=/usr/bin/python3 \
   HSX_VMEC_FILE=/path/to/wout_HSX_QHS_vac.nc \
   /home/rjorge/venvs/spectrax/bin/python tools/run_exact_state_audit.py \
     --manifest tools/exact_state_lanes.office.toml \
     --outdir tools_out/exact_state_audit_office

CI split: fast PR vs manual full
--------------------------------

CI is split into two tiers to keep pull requests fast while preserving full
physics rigor:

- **Fast PR/push tier**: three parallel shards run mypy and targeted test
  subsets (fundamentals, linear core, runtime/nonlinear). This catches solver
  and dtype regressions quickly.
- **Manual full tier**: full ``pytest`` suite plus strict coverage gates:
  ``spectraxgk.terms >= 90%`` and per-module core gates for
  ``linear_krylov.py`` and ``diffrax_integrators.py``.

This keeps iteration latency low for development and still enforces complete
coverage and regression checks on demand without relying on scheduled runners.

Core modular coverage gate
--------------------------

To keep the modular RHS path future-proof, CI also enforces a dedicated
coverage gate for ``spectraxgk.terms``:

.. code-block:: bash

   pytest -q tests/test_terms_assembly.py \
          tests/test_terms_operators.py \
          tests/test_terms_fields.py \
          tests/test_terms_integrators.py \
          tests/test_terms_validation.py \
          --maxfail=1 --disable-warnings \
          --cov=src/spectraxgk/terms \
          --cov-fail-under=90

This guard ensures term-wise kernels, field solves, custom-VJP behavior, and
assembly plumbing stay highly covered while the rest of the benchmark and
cross-code harness keeps evolving.

Core solver coverage gates
--------------------------

CI also enforces dedicated per-module thresholds for the two linear solver
engines that are most likely to regress during algorithm work:

- ``spectraxgk.linear_krylov`` (matrix-free Arnoldi/shift-invert path)
- ``spectraxgk.diffrax_integrators`` (explicit/IMEX/implicit diffrax path)

The gate runs focused tests and checks each module from ``coverage-core.xml``:

.. code-block:: bash

   pytest -q tests/test_linear_krylov_core.py \
          tests/test_diffrax_integrators.py \
          tests/test_diffrax_integrators_core.py \
          --maxfail=1 --disable-warnings \
          --cov=src/spectraxgk \
          --cov-report=xml:coverage-core.xml

Both modules are required to stay at or above 90% line coverage in CI.
