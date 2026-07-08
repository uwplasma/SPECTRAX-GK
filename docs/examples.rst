Examples
========

The ``examples`` directory is organized around two layers:

- case-backed runtime drivers that map directly onto the tracked runtime TOMLs,
- focused demos and benchmark helpers for theory, operators, and scan workflows.

Config-backed runtime cases
---------------------------

These scripts are the closest match to the production benchmark workflows.
They load the checked-in runtime TOMLs and expose only the most useful runtime
overrides at the command line.

Tokamak cases
^^^^^^^^^^^^^

.. code-block:: bash

   python examples/linear/axisymmetric/cyclone_runtime_linear.py
   python examples/nonlinear/axisymmetric/cyclone_runtime_nonlinear.py --steps 200
   python examples/linear/axisymmetric/etg_runtime_linear.py
   python examples/linear/axisymmetric/kaw_runtime_linear.py
   python examples/linear/axisymmetric/kbm_runtime_linear.py
   python examples/nonlinear/axisymmetric/kbm_runtime_nonlinear.py --steps 200
   python examples/nonlinear/axisymmetric/miller_nonlinear_runtime.py --steps 200

VMEC-backed tokamak and stellarator cases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install vmec-jax
   cd examples/vmec
   ./generate_wouts.sh
   cd ../..

   spectraxgk run --config examples/linear/axisymmetric/runtime_circular_vmec_linear.toml
   spectraxgk run --config examples/nonlinear/axisymmetric/runtime_circular_vmec_nonlinear.toml

   spectraxgk run --config examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml
   spectraxgk run --config examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml
   python examples/nonlinear/non-axisymmetric/w7x_nonlinear_vmec_geometry.py --steps 200
   python examples/nonlinear/non-axisymmetric/hsx_nonlinear_vmec_geometry.py --steps 200

For the VMEC-backed stellarator examples, omit ``--steps`` when you want the
default adaptive horizon. Set ``--steps`` only when you intentionally want a
short profiling or diagnostic window. For longer W7-X nonlinear runs, keep
adaptive timesteps enabled (the default for the examples) or reduce ``dt`` if
you need a fixed-step stability study.

The bundled VMEC decks are self-contained examples. Exact HSX or W7-X
validation should use the same TOMLs with ``--vmec-file`` pointing to the
machine-specific benchmark WOUT. If you only need one local WOUT, run
``vmec_jax input.NAME`` in ``examples/vmec`` instead of the full
``generate_wouts.sh`` helper.

The shipped nonlinear stellarator runtime TOMLs now also emit artifact bundles
under ``tools_out/`` by default:

- ``tools_out/w7x_nonlinear_vmec_runtime.diagnostics.csv``
- ``tools_out/hsx_nonlinear_vmec_runtime.diagnostics.csv``
- ``tools_out/w7x_nonlinear_imported_runtime.diagnostics.csv``

Those diagnostics and their matching ``*.summary.json`` files are the intended
inputs for the parity helpers under ``tools/``.
The direct Python runtime wrappers now route through the same artifact-aware
nonlinear path as the executable, so long adaptive runs update that bundle as each
chunk completes.

Runtime TOML entry points
-------------------------

When you want the full config surface instead of the thin case wrappers, use
the executable or the generic example drivers directly. These runtime utilities are
best treated as solver-smoke and exploration entry points; the benchmark
examples remain the audited parity surface for ETG and the other validation
lanes:

.. code-block:: bash

   python examples/utilities/runtime_from_toml.py --config examples/linear/axisymmetric/runtime_cyclone.toml
   python examples/utilities/runtime_from_toml.py --config examples/linear/axisymmetric/runtime_etg.toml
   python examples/utilities/runtime_from_toml.py --config examples/linear/axisymmetric/runtime_kbm.toml
   python examples/linear/axisymmetric/etg_linear_auto.py --outdir tools_out/etg_auto

   spectrax-gk run-runtime-linear \
     --config examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml \
     --out tools_out/cyclone_quasilinear

   spectrax-gk run-runtime-linear \
     --config examples/linear/non-axisymmetric/runtime_w7x_linear_imported_geometry.toml

   spectrax-gk examples/linear/axisymmetric/runtime_cyclone.toml

For a bounded runtime-configured independent ``k_y`` scan that uses
``[parallel] strategy = "batch"`` without changing the single-``k_y`` solver
layout, run:

.. code-block:: bash

   python examples/parallelization/independent_ky_runtime_batch_scan.py

The companion
``examples/parallelization/runtime_batch_ky_scan.toml`` selects two thread
workers through ``[parallel].num_devices``. The runtime still dispatches normal
single-``k_y`` solver calls and gathers results in input order; it does not opt
into the combined-``k_y`` solver path.

Scaling utilities
-----------------

For production parallelization of independent scans and UQ ensembles, prefer
the package helpers:

.. code-block:: python

   import jax.numpy as jnp
   import spectraxgk as sgk

   ky = jnp.asarray([0.1, 0.2, 0.3, 0.4])
   chunks = sgk.ky_scan_batches(ky, n_batches=2)
   values = sgk.batch_map(
       lambda x: {"gamma": x, "ql_weight": x**2},
       ky,
       batch_size=2,
   )

For file-backed calibration and uncertainty workflows that are independent but
not JAX-array ``vmap`` workloads, use ``sgk.independent_map``:

.. code-block:: python

   rows = sgk.independent_map(
       lambda case: {"case": case, "score": len(case)},
       ["cyclone", "hsx", "w7x"],
       workers=2,
   )

These helpers preserve serial ordering and fall back to a one-device ``vmap``
path on laptops. Multi-device runs should still be checked against the serial
result before publication speedups are claimed.

Autodiff validation reports also accept ``workers`` for thread-parallel
central finite-difference columns. The development-only reduced diagnostic
comparison exposes the same pattern:

.. code-block:: bash

   JAX_ENABLE_X64=1 python examples/theory_and_demos/reduced_stellarator_itg/compare_stellarator_itg_optimizations.py \
     --workers 3 \
     --finite-difference-workers 2

The generated JSON records both worker counts and keeps the acceptance
criterion as numerical identity with the serial report.

For a solver-backed identity gate, run the Cyclone ``k_y``-batch scan artifact:

.. code-block:: bash

   python tools/artifacts/generate_parallel_identity_gate.py ky-scan

.. figure:: _static/parallel_ky_scan_gate.png
   :alt: SPECTRAX-GK ky-batch parallelization identity gate
   :width: 100%

   Real Cyclone linear solver comparison between serial and fixed-shape
   ``k_y``-batched scans. The figure verifies that ``gamma`` and ``omega``
   are identical while reporting the observed batch speedup separately.

For a logical-CPU API gate that exercises ``RuntimeParallelConfig`` and pytree
outputs, run:

.. code-block:: bash

   python tools/artifacts/generate_parallel_identity_gate.py logical-cpu --logical-devices 2

.. figure:: _static/logical_cpu_parallel_scan_gate.png
   :alt: SPECTRAX-GK logical CPU parallel scan identity gate
   :width: 100%

   Independent-scan interface gate for structured outputs. This validates the
   parallel API used by UQ and sensitivity ensembles; it is not a nonlinear
   performance claim.

The first lower-level communication gate for velocity-space decomposition is
the Hermite ghost exchange:

.. code-block:: bash

   python tools/artifacts/generate_velocity_parallel_gates.py hermite-exchange --logical-devices 2

.. figure:: _static/hermite_exchange_gate.png
   :alt: SPECTRAX-GK Hermite ghost-exchange identity gate
   :width: 100%

   ``shard_map`` nearest-neighbor exchange for Hermite moments. This validates
   the communication primitive that a future nonlinear velocity-space sharding
   path needs before field reductions and full-RHS identity gates are added.

The paired field-reduction gate is:

.. code-block:: bash

   python tools/artifacts/generate_velocity_parallel_gates.py field-reduce --logical-devices 2

.. figure:: _static/velocity_field_reduce_gate.png
   :alt: SPECTRAX-GK velocity field-reduction identity gate
   :width: 100%

   ``shard_map`` reduction/broadcast over a Hermite mesh. This establishes the
   field-solve communication primitive before streaming-ladder and nonlinear
   RHS identity gates are attempted.

The first production-field-solve reduction gate is:

.. code-block:: bash

   python tools/artifacts/generate_electrostatic_parallel_gates.py field-reduce --logical-devices 2

.. figure:: _static/electrostatic_field_reduce_gate.png
   :alt: SPECTRAX-GK electrostatic field-reduction identity gate
   :width: 100%

   Hermite-sharded ``m=0`` density reduction for the electrostatic
   quasineutrality solve, compared against the production field solve.

The Hermite streaming-ladder coefficient gate is:

.. code-block:: bash

   python tools/artifacts/generate_hermite_streaming_ladder_gate.py --logical-devices 2

.. figure:: _static/hermite_streaming_ladder_gate.png
   :alt: SPECTRAX-GK Hermite streaming-ladder identity gate
   :width: 100%

   ``shard_map`` Hermite exchange plus the ``sqrt(m+1)`` / ``sqrt(m)``
   streaming-ladder coefficients. This is still a communication/coefficient
   gate; full linear streaming also needs the parallel derivative identity
   gate before production runtime wiring.

The first electrostatic drift-slice gate is:

.. code-block:: bash

   python tools/artifacts/generate_electrostatic_parallel_gates.py drift --logical-devices 2

.. figure:: _static/electrostatic_drift_gate.png
   :alt: SPECTRAX-GK electrostatic drift-slice identity gate
   :width: 100%

   Hermite-sharded mirror and curvature/grad-B drift slices, including
   offset-1 and offset-2 Hermite exchanges, compared against the production
   linear RHS with only those terms enabled.

The matching electrostatic diamagnetic-drive gate is:

.. code-block:: bash

   python tools/artifacts/generate_electrostatic_parallel_gates.py diamagnetic --logical-devices 2

.. figure:: _static/electrostatic_diamagnetic_gate.png
   :alt: SPECTRAX-GK electrostatic diamagnetic-drive identity gate
   :width: 100%

   Hermite-sharded electrostatic diamagnetic drive. The sharded route first
   uses the electrostatic field-reduction gate, then applies the local
   ``m=0`` and ``m=2`` drive masks on each Hermite shard. This closes the
   diamagnetic slice for the opt-in electrostatic linear-slices backend.

The periodic streaming microkernel gate adds that field-line derivative:

.. code-block:: bash

   python tools/artifacts/generate_periodic_streaming_microkernel_gate.py --logical-devices 2

.. figure:: _static/periodic_streaming_microkernel_gate.png
   :alt: SPECTRAX-GK periodic streaming microkernel identity gate
   :width: 100%

   Periodic spectral parallel derivative plus Hermite streaming ladder through
   the ``shard_map`` path, compared directly against the production streaming
   operator.

The next gate places that same sharded streaming kernel under the production
linear-RHS call graph with every non-streaming contribution disabled:

.. code-block:: bash

   python tools/artifacts/generate_linear_rhs_streaming_gate.py --logical-devices 2

.. figure:: _static/linear_rhs_streaming_gate.png
   :alt: SPECTRAX-GK streaming-only linear RHS identity gate
   :width: 100%

   Streaming-only ``linear_rhs_cached`` comparison against the velocity-sharded
   periodic streaming path. This closes the first full-RHS call-graph identity
   gate for the streaming term only; it is not yet a full linear scan or
   nonlinear speedup claim.

With a nonzero electrostatic response, use:

.. code-block:: bash

   python tools/artifacts/generate_linear_rhs_streaming_electrostatic_gate.py --logical-devices 2

.. figure:: _static/linear_rhs_streaming_electrostatic_gate.png
   :alt: SPECTRAX-GK electrostatic streaming linear RHS identity gate
   :width: 100%

   Streaming plus electrostatic ``phi`` call-graph comparison. The field solve
   uses the Hermite-sharded electrostatic reduction gate; this validates the
   next velocity-sharded streaming slice before drift, diamagnetic-drive, and
   nonlinear terms are introduced.

For the composed electrostatic linear-slices backend, use:

.. code-block:: bash

   python tools/artifacts/generate_linear_rhs_electrostatic_slices_gate.py --logical-devices 2

.. figure:: _static/linear_rhs_electrostatic_slices_gate.png
   :alt: SPECTRAX-GK composed electrostatic linear-slices identity gate
   :width: 100%

   Full opt-in electrostatic linear-slices call-graph comparison for
   streaming, mirror, curvature, grad-B, and diamagnetic drive. This is an
   opt-in electrostatic linear-RHS identity artifact for the single-species
   periodic electrostatic RHS path; collisions, linked boundaries,
   electromagnetic terms, and nonlinear brackets remain separate gates.

Use the strong-scaling sweep helper to collect parallelization timings for the
distributed linear RK2 loop:

.. code-block:: bash

   python examples/utilities/strong_scaling_sweep.py \
     --ny 128 --nz 256 --nl 8 --nm 8 --steps 120 \
     --devices 1,2,4,8 \
     --backend cpu_parallel_large \
     --out tools_out/strong_scaling_cpu.csv

On multi-GPU systems, point ``--devices`` at the available accelerators and
update ``--backend`` accordingly (for example ``cuda_parallel_large``). The
backend labels are just sweep names for the output table; they do not change
the runtime physics or solver path.

For the current opt-in Hermite-sharded electrostatic linear RHS path, use the
engineering sweep helper:

.. code-block:: bash

   python tools/profiling/profile_linear_rhs_parallel_slices_sweep.py \
     --platform cpu --devices 1,2,4,8 --nms 64,128 \
     --nl 4 --ny 32 --nz 128 --rtol 1e-5

.. figure:: _static/linear_rhs_parallel_slices_sweep.png
   :alt: SPECTRAX-GK electrostatic linear-slices parallelization sweep
   :width: 100%

   Device-count and Hermite-resolution sweep for the opt-in electrostatic
   linear-slices backend. The right panel is the identity gate; the left panel
   is engineering timing only and should not be promoted as a nonlinear or
   publication speedup claim.

Plotting outputs
----------------

To visualize nonlinear diagnostic histories from ``*.out.nc`` files:

.. code-block:: bash

   python examples/utilities/plot_runtime_outputs.py tools_out/cyclone_release.out.nc \
     --out tools_out/cyclone_release_diagnostics.png

Geometry examples
-----------------

VMEC and Miller geometry usage examples are documented in :doc:`geometry`.

Nonlinear restart and continuation
----------------------------------

The tracked nonlinear runtime path supports a NetCDF ``out/big/restart``
bundle together with continuation from the saved restart state.

One-shot nonlinear bundle write:

.. code-block:: bash

   spectrax-gk run-runtime-nonlinear \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml \
     --steps 200 \
     --out tools_out/cyclone_release.out.nc

For the short Cyclone comparison replay (``t_max = 5``, no collisions), use
``examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_short.toml``.
That file pins the short-run dissipation contract explicitly
(``p_hyper = 2``, ``damp_ends_amp = 0``) instead of relying on the longer
production defaults.

Restart-aware TOML snippet:

.. code-block:: toml

   [time]
   nstep_restart = 100

   [output]
   path = "tools_out/cyclone_release.out.nc"
   restart_if_exists = true
   save_for_restart = true
   append_on_restart = true
   restart_with_perturb = false

With that configuration, rerunning the same nonlinear command resumes from
``tools_out/cyclone_release.restart.nc`` when it already exists and appends the
continued history to ``tools_out/cyclone_release.out.nc``. This is the
recommended user-facing workflow for long nonlinear turbulence jobs.

Geometry helper workflows
-------------------------

The runtime geometry path can generate imported geometry files from VMEC and
Miller inputs when the external helper scripts are available:

.. code-block:: bash

   cd examples/vmec
   vmec_jax input.NuhrenbergZille_1988_QHS
   cd ../..
   export SPECTRAX_BOOZ_XFORM_JAX_PATH=/absolute/or/relative/booz_xform_jax
   python tools/artifacts/generate_geometry_eik.py vmec \
     --config examples/nonlinear/non-axisymmetric/runtime_hsx_nonlinear_vmec_geometry.toml

   python tools/artifacts/generate_geometry_eik.py miller \
     --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml

Benchmark and scan helpers
--------------------------

These scripts produce the scan-level plots and tables used in the benchmark
discussion:

.. code-block:: bash

   python benchmarks/cyclone_linear_benchmark.py
   python examples/linear/axisymmetric/etg_linear_auto.py
   python benchmarks/etg_linear_benchmark.py
   python benchmarks/kbm_beta_scan.py
   python benchmarks/kinetic_linear_benchmark.py
   python benchmarks/tem_linear_benchmark.py

Foundational demos
------------------

These smaller examples are useful for understanding the numerical building
blocks without running a full benchmark case:

.. code-block:: bash

   python benchmarks/basis_orthonormality.py
   python examples/theory_and_demos/cyclone_geometry.py
   python examples/theory_and_demos/autodiff_inverse_growth.py
   python examples/theory_and_demos/autodiff_inverse_twomode.py
   python examples/theory_and_demos/diffrax_linear_demo.py
   python examples/theory_and_demos/example.py
   python examples/theory_and_demos/gradB_coupling_hl_1d.py
   python examples/theory_and_demos/linear_rhs_demo.py
   python examples/theory_and_demos/two_stream_hermite_1d.py

Differentiable optimization examples
------------------------------------

The public optimization examples are actual VMEC-JAX QA stellarator workflows
with one SPECTRAX-GK transport tuple appended to the VMEC-JAX objective list:

.. code-block:: bash

   python examples/optimization/QA_optimization_linear_ITG.py
   python examples/optimization/QA_optimization_quasilinear_ITG.py
   python examples/optimization/QA_optimization_nonlinear_ITG.py
   python examples/optimization/QA_nonlinear_ITG_matched_audit.py
   python examples/optimization/QA_parameter_scan.py

The three ``QA_optimization_*_ITG.py`` scripts intentionally mirror upstream
``vmec_jax/examples/optimization/QA_optimization.py`` and preserve the
high-weight ``iota = 0.41`` target. Keep the SPECTRAX-GK transport weight small
until the solved-equilibrium aspect, iota, and quasisymmetry gates pass.
They are deliberately edited through top-level constants, not command-line
arguments. Reproducible campaign drivers, dry-runs, and plotting/gate
generation scripts live under ``tools/``; for example:

``QA_nonlinear_ITG_matched_audit.py`` is the production-evidence companion:
after long SPECTRAX-GK nonlinear baseline/candidate campaigns finish, edit its
ensemble paths and run it to build the matched reduction and uncertainty gate.
It does not launch simulations and does not consume reduced/startup nonlinear
optimizer residuals.

.. code-block:: bash

   python tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py --dry-run

For a paper-facing constraints-only baseline that uses the same simple seed,
ESS scaling, and max-mode-5 objective recipe but tighter admission tolerances,
run:

.. code-block:: bash

   python tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py \
     --strict-upstream-qa-baseline --solver-device gpu \
     --outdir tools_out/vmec_jax_qa_strict_baseline

Use that strict baseline before comparing transport-weight candidates; a
baseline that terminates just below ``iota >= 0.41`` should be refined rather
than promoted by relaxing the solved-WOUT gate. The strict preset keeps the
admission gate at ``iota >= 0.41`` and uses a small default optimizer target
buffer, ``target iota = 0.4102``, so roundoff-level target undershoot does not
invalidate an otherwise precise QA solve.
The tracked strict-baseline sidecar
``docs/_static/vmec_jax_qa_strict_baseline/summary.json`` records the current
office-GPU exact SciPy/ESS result: ``nfev = 39``, aspect ``5.000154``,
mean iota ``0.4101997``, QS residual ``2.60e-4``, and a passed solved-WOUT
gate. This is a constraints-only QA reference, not a transport-optimized
stellarator.

Reduced synthetic scripts are kept outside ``examples/optimization`` as
development diagnostics only:

.. code-block:: bash

   python examples/theory_and_demos/reduced_stellarator_itg/stellarator_itg_growth_optimization.py
   python examples/theory_and_demos/reduced_stellarator_itg/stellarator_itg_quasilinear_flux_optimization.py
   python examples/theory_and_demos/reduced_stellarator_itg/stellarator_itg_nonlinear_heat_flux_optimization.py
   python examples/theory_and_demos/reduced_stellarator_itg/compare_stellarator_itg_optimizations.py
   python examples/theory_and_demos/reduced_stellarator_itg/stellarator_itg_portfolio_gate.py --finite-difference-workers 2
   python tools/artifacts/build_qa_low_turbulence_comparison.py --pdf
   python tools/artifacts/build_qa_low_turbulence_time_horizon_audit.py --pdf

The portfolio gate writes JSON/PNG/PDF artifacts and checks scalar plus
row-wise AD/finite-difference agreement for the same surface/alpha/``k_y``
reduction that will be used by the production VMEC/Boozer objective rows. Its
default table covers three surfaces, two field-line ``alpha`` values, and
three ``k_y`` values with growth and quasilinear-flux columns. This is a
reduced/model-development gate; it does not claim optimized nonlinear heat
flux or calibrated saturated transport. Treat the JSON sidecar as the audit
source; the PNG/PDF summarize the same sidecar for docs and review.

The aspect-6 QA low-turbulence comparison tool writes
``docs/_static/qa_low_turbulence_comparison.{json,png,pdf}`` plus CSV sidecars.
It compares a quasisymmetry/aspect/iota-floor design with a design that adds a
reduced nonlinear heat-flux envelope residual, then plots the fixed-``a/L_T``
``Q_env`` versus ``a/L_n`` scan, fixed-gradient reduced-envelope traces,
reduced LCFS surfaces colored by ``|B|``, and reduced Boozer-LCFS ``|B|`` maps.
The trace is smooth by construction because it integrates
``dE/dt = 2 gamma E - alpha E^2``; it should not be read as a turbulent
nonlinear SPECTRAX-GK heat-flux time series. This is a reduced
differentiability and visualization example; production nonlinear optimization
still requires long post-transient transport-window audits.
The companion time-horizon audit writes
``docs/_static/qa_low_turbulence_time_horizon_audit.{json,csv,png,pdf}`` and
shows that ``t v_ti/a = 400`` is already converged relative to the
``t=1000`` reduced-envelope reference for the tracked designs.

The production bridge now exposes the same portfolio layout for real
``vmec_jax -> booz_xform_jax -> SPECTRAX-GK`` rows:
``stellarator_itg_vmec_boozer_sample_objective_table_from_state`` returns a
``(surface, alpha, ky, objective)`` table and
``stellarator_itg_vmec_boozer_portfolio_objective_from_state`` reduces it with
the same weights as the cheap gate. Promotion still requires held-out
surface/field-line artifacts and matched baseline/optimized long
post-transient nonlinear windows, not startup traces or reduced-window
estimators.

The autodiff demos write summary JSON plus `R/L_Ti` and `R/L_n` sweep CSVs in
the chosen output directory alongside the publication-ready plots. The
single-mode figure is a local inverse/sensitivity example; the two-mode figure
is the release-grade parameter-recovery validation.

.. figure:: _static/autodiff_inverse_growth.png
   :width: 90%
   :align: center

   Single-mode inverse/sensitivity demo. The goal is to verify the autodiff
   Jacobian and show what one measured mode constrains locally; the expected
   outcome is small observable and derivative error, not unique recovery of
   both gradients. The shipped result matches that expectation: `(gamma, omega)`
   are reproduced closely while the recovered `(R/L_Ti, R/L_n)` remains offset
   because the one-mode inverse is not globally identifiable.

.. figure:: _static/autodiff_inverse_twomode.png
   :width: 90%
   :align: center

   Two-mode inverse validation. The goal is to recover the planted gradients
   from two independent mode observables and verify that the autodiff Jacobian
   stays consistent with finite differences. The shipped result reaches the
   target to numerical precision and is the reviewer-facing parameter-recovery
   validation.

Secondary slab workflow
-----------------------

.. code-block:: bash

   python -m spectraxgk.cli run-runtime-linear \
     --config benchmarks/runtime_secondary_slab.toml

   python benchmarks/secondary_slab_workflow.py

The staged helper runs the linear seed, writes a restart state in the runtime
binary layout, and then launches the nonlinear follow-up with the matching
restart and fixed-mode controls used in the tracked secondary benchmark.

Full-GK ETG nonlinear pilot
---------------------------

.. code-block:: bash

   python examples/nonlinear/axisymmetric/etg_runtime_nonlinear.py --steps 200
   JAX_ENABLE_X64=1 spectrax-gk examples/nonlinear/axisymmetric/runtime_etg_nonlinear.toml --steps 200

This is the full-GK two-species ETG nonlinear pilot lane. The shipped
contract now matches the audited short-window startup path: ``Lx = 1.25`` for the linked ETG box and
``gaussian_init = true`` with ``init_single = false`` because GX reads
``init_single`` from its ``[Expert]`` section, not from ``[Initialization]``.
