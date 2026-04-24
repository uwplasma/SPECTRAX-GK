Validation And Coverage Strategy
================================

Purpose
-------

The refactor branch uses a traceability manifest to keep software coverage,
physics validation, and publication artifacts tied together. The package-wide
target is still 95% coverage, but coverage is treated as an engineering
guardrail. A test only counts as useful when it protects an equation, a
numerical method, a diagnostic convention, an artifact contract, or an
autodiff/performance guarantee.

The machine-readable manifest lives at
``tools/validation_coverage_manifest.toml`` and is checked by
``tools/check_validation_coverage_manifest.py``. Each critical module entry
records:

- the source file and owning refactor lane;
- reference anchors from the literature, independent-code comparisons, or
  documented numerical methods;
- physics contracts that should remain true across refactors;
- numerical contracts such as observed order, conservation, window policy, or
  finite-value handling;
- fast tests that run locally;
- shipped artifacts or gate reports that document the validation state;
- the next tests needed to reach the package-wide 95% target.

How to check the manifest
-------------------------

Run:

.. code-block:: bash

   python tools/check_validation_coverage_manifest.py

For CI or release bookkeeping, write a JSON summary:

.. code-block:: bash

   python tools/check_validation_coverage_manifest.py \
     --out-json docs/_static/validation_coverage_manifest_summary.json

The manifest complements ``tools/make_validation_gate_index.py``. The gate
index reports which validation artifacts currently pass. The coverage manifest
reports whether the remaining refactor and testing work is traceable to
physics, numerics, artifacts, and tests.

Finalization sequence
---------------------

The remaining work should be closed in this order.

1. **Freeze module contracts before moving code.**
   For each large file, write or update tests for current public behavior, then
   extract only cohesive helpers. Keep compatibility exports until examples,
   docs, and benchmark scripts use the new module boundaries.

2. **Finish the high-priority refactor modules.**
   The active blockers are ``runtime.py``, ``linear.py``, ``nonlinear.py``,
   ``benchmarks.py``, ``diagnostics.py``, ``runtime_artifacts.py``,
   ``validation_gates.py``, ``zonal_validation.py``, and
   ``from_gx/vmec.py``. Each slice should land with targeted tests and no
   physics-model change.

3. **Turn open physics lanes into explicit gates.**
   Literature-facing lanes should produce JSON/CSV/PNG/PDF artifacts with the
   same observable, window, tolerance, and source recorded in metadata. The
   current priority list is W7-X zonal response, W7-X fluctuation spectra,
   monotone Cyclone velocity-space convergence, KBM branch continuity,
   nonlinear window statistics, and Merlo/Rosenbluth-Hinton response panels.

4. **Replace coverage gaps with physics or numerics tests.**
   Do not add shallow import-only tests to chase the number. Prefer tests for
   ladder identities, field-solve limits, bracket antisymmetry, diagnostic
   normalization, strict JSON output, observed-order gates, restart invariants,
   and artifact reload behavior.

5. **Validate differentiability explicitly.**
   Autodiff examples should carry finite-difference or tangent checks, inverse
   recovery diagnostics, and covariance/uncertainty estimates. The later
   ``vmec_jax`` and ``booz_xform_jax`` bridge should add geometry-gradient
   checks before claiming optimization readiness.

6. **Keep performance measurements separated from validation.**
   Performance panels should report cold compile, warm runtime, memory, output
   time, and parallelization speedup separately. Parallelization gates should
   first target independent scans, UQ ensembles, and sensitivity batches where
   strong scaling is scientifically useful and robust.

7. **Raise CI thresholds in phases.**
   Keep default tests under the five-minute local budget. Enforce fast critical
   modules first, then broad package coverage on the wide CI lane, then manual
   office/GPU parity and performance sweeps. The merge target is package-wide
   coverage at or above 95% with the validation manifest and gate index both
   passing.

Release readiness criteria
--------------------------

The refactor branch is ready to merge when:

- package-wide coverage is at least 95% on the wide lane;
- all high-priority manifest modules have their planned extraction tests;
- the tracked validation gate index is passing or open lanes are explicitly
  labeled as non-release exploratory artifacts;
- shipped examples still run and plot from output files;
- W7-X, HSX, Cyclone, Cyclone-Miller, KBM, ETG, Miller, and VMEC examples have
  current documentation that matches the artifacts;
- autodiff examples validate gradients and inverse/UQ outputs;
- docs build with warnings treated as errors;
- package build, release workflow, and PyPI metadata checks pass.
