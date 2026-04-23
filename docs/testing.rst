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
- **Response-function tests**: zonal-flow residuals, GAM damping, and late-time
  envelopes.
- **Spectral tests**: fluctuation spectra and windowed nonlinear statistics.
- **Autodiff tests**: tangent, finite-difference, and inverse/UQ consistency.

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

Literature-anchored response and spectrum tests
-----------------------------------------------

The next research-facing additions should follow the published benchmark
observables rather than inventing repo-local metrics:

- **Rosenbluth-Hinton / GAM response in shaped tokamaks**: use the shaped
  benchmark conventions summarized by Merlo et al. to track residual levels and
  GAM damping alongside the linear shaping scan.
- **W7-X zonal-flow response**: use the stella/GENE W7-X benchmark conventions
  for residual level and damping envelope.
- **W7-X fluctuation spectra**: follow the W7-X Doppler-reflectometry
  comparison work for density and zonal-flow frequency spectra.
- **Electromagnetic stellarator verification**: adopt a heavy-electron
  electromagnetic lane before realistic-mass claims, following the GENE-3D
  verification pattern.

These should be implemented as reproducible, script-owned figure/artifact
lanes, not as ad hoc notebooks.

The first reusable tooling for this lane now exists:

- :func:`spectraxgk.benchmarking.zonal_flow_response_metrics`
- :func:`spectraxgk.benchmarking.load_diagnostic_time_series`
- :func:`spectraxgk.plotting.zonal_flow_response_figure`
- ``tools/plot_zonal_flow_response.py``
- ``tools/plot_zonal_flow_response_from_output.py``

The diagnostics stream now also carries ``Diagnostics/Phi_zonal_mode_kxt``, a
signed complex zonal-potential history reduced over ``z`` with the same volume
weights used elsewhere. That is the primitive to use for manuscript-grade
Rosenbluth-Hinton / GAM work. ``Diagnostics/Phi2_zonal_t`` remains useful as a
zonal-energy proxy for intermediate checks, but it is no longer the target
observable for the final paper lane.

The first case-specific shaped-Miller pilot for this lane is now reproducible
through ``examples/benchmarks/runtime_miller_zonal_response.toml`` and
``tools/generate_miller_zonal_response_pilot.py``. Its frozen artifact lives in
``docs/_static/miller_zonal_response_pilot.png``. The current frozen artifact
is a Merlo-style source-driven zonal-relaxation run with zero gradients,
adiabatic electrons, ``k_xρ_i≈0.05``, and a GX-style ``phiext_full`` forcing
contract. It remains open because the present trace is only weakly damped: the
long run reaches ``t≈100`` with
``ω_GAM≈0.66`` and about ``20`` envelope peaks, but it does **not** settle to
a clean stationary residual window yet. That makes it a useful research-grade
diagnostic artifact and a real stepping-stone, but not a closed
Rosenbluth-Hinton residual benchmark.

Diffrax and nonlinear smoke tests
---------------------------------

Diffrax integration and the nonlinear driver are exercised with fast smoke
tests:

- ``tests/test_diffrax_integrators.py`` runs explicit and IMEX diffrax solvers
  on tiny grids.
- ``tests/test_diffrax_integrators_core.py`` hardens branch coverage for
  diffrax helper paths (solver selection, save modes, streaming fits, IMEX
  branches, parallelization, and validation errors).
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
- ``tests/test_runtime_config.py`` also locks the public nonlinear stellarator
  runtime contract, including the absence of adaptive-step truncation caps and
  the presence of default ``tools_out/...`` artifact paths for W7-X and HSX.

Nonlinear parity snapshots
--------------------------

Recent GX parity spot checks are tracked outside the automated test suite:

- **Cyclone nonlinear short replay**: the GX `cyclone_salpha_short.in` replay
  (`dt=0.05`, `t_max=5`, collisions off, diagnostics stride 1) now uses the
  explicit short-reference runtime contract in
  ``examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_short.toml``.
  The main short-run drift turned out to be configuration-level: the replay
  needed ``p_hyper = 2`` and no end damping to match the public GX short input.
  With that contract restored, the tracked comparison improves to
  ``mean_rel_abs(Wphi) ~= 2.11e-1`` and
  ``mean_rel_abs(HeatFlux) ~= 2.51e-1``. The resolved audit remains in
  ``docs/_static/nonlinear_cyclone_short_resolved_audit_t5.{png,csv}``, where
  ``Wphi_kyst`` is still the dominant residual mismatch.
- **Secondary (`kh01a`)**: the tracked secondary comparison now uses a dense
  real GX run (`kh01a_shortdense.out.nc`, 10 samples in ``omega_kxkyt``) and
  the rebuilt ``secondary_gx_out_compare.csv``. The comparison helper now uses
  the GX file horizon automatically in ``out-nc`` mode, so it no longer mixes a
  short GX replay with a ``t_max = 100`` SPECTRAX stage-2 run. On the matched
  short window, growth rates match tightly (``max rel_gamma ~= 1.87e-4``) and
  the non-zonal ``omega`` modes also close tightly
  (``rel_omega ~= 3.23e-4`` and ``9.92e-4`` on the ``k_y = 0.1`` sidebands).
  The only large relative ``omega`` values left are the effectively zero-
  frequency ``k_y = 0`` sidebands, where the absolute mismatch stays
  ``O(1e-6)``.
- **W7-X nonlinear (`t \\approx 200`)**: the refreshed long-window CSV-backed
  comparison now closes at
  ``mean_rel_abs(Wg) ~= 9.24e-2``,
  ``mean_rel_abs(Wphi) ~= 1.16e-1``,
  ``mean_rel_abs(HeatFlux) ~= 7.55e-2``.
- **HSX nonlinear (`t = 50`)**: the refreshed comparison closes at
  ``mean_rel_abs(Wg) ~= 2.75e-2``,
  ``mean_rel_abs(Wphi) ~= 3.61e-2``,
  ``mean_rel_abs(HeatFlux) ~= 2.91e-2``.
- **KBM nonlinear (`t = 100`)**: the refreshed long-window comparison closes at
  roughly ``9.3e-3`` mean-relative error across
  ``Wg/Wphi/Wapar/HeatFlux/ParticleFlux``.

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
  in a fixed-:math:`k_y` beta sweep against the tracked benchmark reference and
  exact-diagnostic audits.

Running tests
-------------

.. code-block:: bash

   pytest

Benchmark reproducibility stack
-------------------------------

The public CI and the tracked benchmark atlas are currently validated against a
tested numerical stack:

- ``jax>=0.8,<0.9``
- ``jaxlib>=0.8,<0.9``
- ``numpy>=2.3,<2.4``
- ``diffrax>=0.7,<0.8``
- ``equinox>=0.13,<0.14``

This is not a claim that newer releases are unsupported. It is a statement
about benchmark reproducibility. Near-marginal or branch-sensitive lanes such
as TEM, ETG runtime scans, and some imported-linear stellarator cases can move
materially under newer JAX/NumPy combinations even when the code still runs.
When investigating parity regressions, reproduce the issue on the tested stack
first before changing solver logic.
For runtime-example parity reproduction across recent precision-policy changes,
also set ``JAX_ENABLE_X64=1``. Default precision can be faster while still
moving parity-sensitive linear example outputs.

Stress-matrix parity gates
--------------------------

In addition to unit/regression tests, SPECTRAX-GK includes a small set of
"stress-matrix" gates meant to catch parity regressions early (before tracked
benchmark figures move):

- **Restart parity**: ``tests/test_restart_gate.py`` verifies that a nonlinear
  run resumed from a compatible restart reproduces the same final state as a
  continuous run. This now covers both the raw binary state path and the
  nonlinear ``*.restart.nc`` bundle path, together with append-on-restart
  history preservation in ``*.out.nc``.
- **CPU/GPU short-window parity** (optional): ``tests/test_device_parity_gate.py``
  compares a short nonlinear trajectory norm on CPU vs GPU. Enable explicitly:

  .. code-block:: bash

     SPECTRAXGK_DEVICE_PARITY=1 pytest -q tests/test_device_parity_gate.py

- **VMEC roundtrip determinism** (optional): ``tests/test_vmec_roundtrip_gate.py``
  regenerates an ``*.eik.nc`` from a provided VMEC file twice and asserts the
  imported geometry arrays are bitwise identical. Enable explicitly:

  .. code-block:: bash

     SPECTRAXGK_VMEC_FILE=/path/to/wout.nc pytest -q tests/test_vmec_roundtrip_gate.py

For developer workflows that require local reference benchmark NetCDFs or dump
artifacts, use:

- ``tools/run_gx_linear_stress_matrix.py`` (KAW, Cyclone kinetic electrons, KBM Miller)
- ``tools/run_imported_linear_targeted_audit.py`` (generic per-``ky`` targeted imported-linear wrapper)
- ``tools/compare_gx_imported_window.py`` (exact imported-linear one-window replay against reference ``diag_state`` dumps)
- ``tools/run_kbm_lowky_extractor_audit.py`` (direct cached-trajectory KBM low-``ky`` extractor audit)
- ``tools/run_exact_state_audit.py`` (manifest-driven wrapper around the exact-state audit tools)
- ``tools/run_restart_parity_gate.py`` (manifest-driven nonlinear restart/continuation parity gate)
- ``tools/run_device_parity_gate.py`` (manifest-driven CPU/GPU short-window parity gate)
- ``tools/run_vmec_roundtrip_gate.py`` (manifest-driven VMEC ``vmec -> eik.nc`` determinism gate)

The current full-GK nonlinear ETG lane is now explicitly tracked as a pilot
runtime contract via
``examples/nonlinear/axisymmetric/runtime_etg_nonlinear.toml``. That lane is
separate from the reduced ``cETG`` solver and should be used for future
GX-backed nonlinear ETG parity work.

For ETG nonlinear audit runs, use dense short-window overrides first:

.. code-block:: bash

   JAX_ENABLE_X64=1 spectrax-gk examples/nonlinear/axisymmetric/runtime_etg_nonlinear.toml \
     --steps 10 \
     --sample-stride 1 \
     --diagnostics-stride 1

This lane is currently expensive enough that short persisted windows are the
right first diagnostic step before attempting long production horizons.

The ETG short-window startup mismatch was traced to the GX input contract, not
the nonlinear ETG operator. GX reads ``init_single`` from ``[Expert]`` rather
than ``[Initialization]``, so the audited GX pilot was actually using the
Gaussian startup branch. The shipped runtime ETG pilot now matches that
contract with ``gaussian_init = true``, ``init_single = false``,
``Lx = 1.25``, and GX-style ``kz`` hypercollisions. On the matched
``Nx=10``, ``Ny=22``, ``ntheta=16``, ``Nl=4``, ``Nm=4``, ``dt=1e-4``,
``t_max=0.001`` pilot, the refreshed short-window comparison lands at
``mean_rel_abs(Wg) ~= 1.31e-2`` and ``mean_rel_abs(Wphi) ~= 5.18e-3``, with
the final heat-flux point within a few percent of GX.

The targeted imported-linear wrapper and the underlying
``compare_gx_imported_linear.py`` comparator now support two important controls
for honest stress-lane scoring without changing the default full-window
behavior:

- ``--sample-step-stride``: subsample the saved diagnostic sample indices
  before scoring.
- ``--max-samples``: truncate scoring to the first N selected samples.

The lower-level comparator also supports ``--cache-dir`` plus ``--reuse-cache``
to persist per-``ky`` trajectory/result arrays (``gamma``, ``omega``,
``Wg``, ``Wphi``, ``Wapar``) as compressed ``.npz`` files keyed by the actual
reference file, geometry file, reference input, selected ``ky``, Hermite/Laguerre
resolution, mode selector, and sample-window contract. This makes the
stress-lane tooling incremental instead of rerunning a full lane every time.
It now also writes absolute diagnostic-error columns and the reference
``|gamma|`` / ``|omega|`` scales alongside the relative metrics. That matters
for near-marginal imported-linear stellarator lanes such as HSX, where
``mean_rel_gamma`` can look large simply because the reference growth rate is
close to zero even while the absolute growth-rate mismatch and the field-energy
diagnostics remain small.

For VMEC-backed exact-state audits, the runtime bridge now prefers a local
``booz_xform_jax`` checkout and injects a temporary ``booz_xform`` compatibility
shim only into the external geometry-helper subprocess. This preserves the
audited reference workflow while avoiding a host-level dependency on the original ``booz_xform``
Python package.

The bridge auto-discovers ``booz_xform_jax`` from
``BOOZ_XFORM_JAX_PATH`` / ``SPECTRAX_BOOZ_XFORM_JAX_PATH`` or from a checkout placed
next to the SPECTRAX-GK workspace. When a specific
Python environment is needed for the helper subprocesses, set
``geometry.gx_python`` in the runtime TOML. On ``office``, the normal audited
path is:

.. code-block:: bash

   export BOOZ_XFORM_JAX_PATH=/path/to/booz_xform_jax
   export SPECTRAX_VENV_PYTHON=/path/to/venv/bin/python
   export SPECTRAX_OFFICE_ROOT=/path/to/SPECTRAX-GK
   W7X_VMEC_FILE=/path/to/wout_w7x.nc \
   HSX_VMEC_FILE=/path/to/wout_HSX_QHS_vac.nc \
   "$SPECTRAX_VENV_PYTHON" tools/run_exact_state_audit.py \
     --manifest tools/exact_state_lanes.office.toml \
     --outdir tools_out/exact_state_audit_office

The restart/continuation gate uses the same environment model and should be
run against the tracked nonlinear lanes with ``PYTHONPATH`` set to the source
tree so the office venv does not pick up a stale installed package:

.. code-block:: bash

   PYTHONPATH="$SPECTRAX_OFFICE_ROOT/src" \
   "$SPECTRAX_VENV_PYTHON" tools/run_restart_parity_gate.py \
     --manifest tools/restart_gate_lanes.office.toml \
     --outdir tools_out/restart_parity_office

The current ``office`` exact-state manifest now includes:

- startup audits for Cyclone, KBM, W7-X, and HSX
- late dumped-state audits for Cyclone Miller, Cyclone runtime, W7-X, and KBM

For KBM specifically, the startup audit, late dumped-state audit, nonlinear
term replay, and first RK4 partial-step replay now all close on the shipped
nonlinear config for the current release pass. The remaining KBM work is
therefore future long-window cleanup rather than a blocking startup-state,
diagnostic-reconstruction, or first-step assembly mismatch.

The device-parity gate now has audited ``office`` manifests for one tokamak and
one stellarator lane, both requiring stable nonzero outputs rather than the
older zero-norm smoke probe:

.. code-block:: bash

   PYTHONPATH="$SPECTRAX_OFFICE_ROOT/src" \
   "$SPECTRAX_VENV_PYTHON" tools/run_device_parity_gate.py \
     --manifest tools/device_parity_lanes.office.toml \
     --outdir tools_out/device_parity_office

The VMEC roundtrip gate uses the same manifest pattern and currently covers the
tracked W7-X and HSX VMEC lanes:

.. code-block:: bash

   PYTHONPATH="$SPECTRAX_OFFICE_ROOT/src" \
   "$SPECTRAX_VENV_PYTHON" tools/run_vmec_roundtrip_gate.py \
     --manifest tools/vmec_roundtrip_lanes.office.toml \
     --outdir tools_out/vmec_roundtrip_office

If the helper must be forced to another interpreter, set ``geometry.gx_python``
in the runtime TOML used by the audit and rerun the same command. The old
environment-variable override is no longer documented because the preferred
path is the internal ``booz_xform_jax`` backend.

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
