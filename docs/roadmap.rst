Roadmap
=======

SPECTRAX-GK is being developed as a research-grade, JAX-native gyrokinetic
solver: accurate against independent benchmarks, differentiable end to end,
fast enough for production studies, and simple enough for researchers to run,
test, and extend.

Current target
--------------

The active milestone is not just "more features". The release target is a
validated codebase with:

- documented install, run, plot, and artifact workflows;
- literature-anchored physics gates for linear, nonlinear, response-function,
  geometry, and autodiff examples;
- measured cold-start, warm-runtime, memory, and parallelization behavior;
- CI gates that cover unit tests, regression tests, docs, packaging, and
  selected validation artifacts;
- clear module boundaries so equations, numerics, runtime I/O, plotting, and
  benchmark policy can be tested independently.

Active refactor lane
--------------------

The current branch is splitting large modules into smaller, tested units while
preserving public behavior and benchmark parity. Refactors should only land
when they add or preserve tests for the extracted behavior.

Highest-value remaining slices:

- ``runtime.py`` and runtime orchestration;
- ``linear.py`` and linear operator assembly;
- ``nonlinear.py`` and nonlinear bracket/diagnostic paths;
- ``benchmarks.py`` and benchmark artifact policy;
- plotting and publication-figure helpers;
- VMEC/Miller geometry adapter boundaries.

Validation gates
----------------

Research-facing validation is organized around artifact gates. Each gate should
have an owning script, frozen output path, reference source, fit/window policy,
and explicit numeric tolerance.

Linear gates:

- growth rate and real frequency from late-time fits;
- branch continuation and near-marginal behavior;
- eigenfunction overlap and phase/sign conventions;
- velocity-space and field-line resolution convergence;
- geometry-contract parity for Miller and VMEC imports.

Nonlinear gates:

- windowed heat-flux statistics rather than single-time comparisons;
- mode-resolved ``Wphi`` and heat-flux spectra;
- conservation/free-energy behavior in reduced limits;
- restart and diagnostic-order regressions;
- stable long-window behavior for Cyclone, Miller, KBM, W7-X, and HSX lanes.

Response-function gates:

- Rosenbluth-Hinton/GAM response in shaped tokamak cases;
- W7-X and HSX zonal-flow response using the same extraction protocol;
- damping, frequency, residual, and recurrence diagnostics reported together.

Autodiff gates:

- finite-difference, tangent, and adjoint consistency where applicable;
- sensitivity maps for growth rates, frequencies, and selected transport
  metrics;
- two-parameter inverse problems with covariance/uncertainty estimates;
- differentiable geometry gradients after the ``vmec_jax`` bridge is added.

Near-term physics priorities
----------------------------

The next physics lanes should be closed in this order:

1. W7-X zonal-response late-envelope closure. The VMEC-backed SPECTRAX-GK
   artifact now uses the paper-faithful potential initializer, signed
   line-average observable, and initial-Gaussian-maximum normalization, and the
   long-window comparison passes the digitized stella/GENE residual and
   time-coverage gates for all four wavelengths. The remaining open item is the
   excessive late-window envelope and too-fast early decay, which should be
   treated as a velocity-space recurrence / moment-closure audit plus a final
   Gaussian-width contract decision. The runtime now preserves final samples
   under strided diagnostics, aborts checkpointed artifact runs on the first
   non-finite diagnostic chunk, and preserves signed zonal line/mode diagnostics
   across external restart continuation. A four-wavelength ``Nl=16``,
   ``Nm=64``, ``dt=0.05`` refresh reached ``t≈100`` with finite signed traces,
   so longer restart-continued W7-X traces can now be used to study the
   remaining physics/numerics issue directly.
2. Tighten the now-materialized windowed nonlinear-statistics panel beyond the current ``0.10`` release gate where the literature/reference windows justify stricter tolerances.
3. W7-X multi-flux-tube ITG/TEM extension and fluctuation-spectrum lane.
4. Shaped multispecies tokamak linear lane.
5. ETG nonlinear only after its benchmark operating point and observable
   contract are explicit.

Performance and memory
----------------------

Performance work should stay measurement-driven. The current priorities are:

- separate cold compile, cache construction, first-step compile, warm runtime,
  and output/plot time in every reported panel;
- reduce integrator compile cost before optimizing small runtime kernels;
- keep large constants out of closed-over JIT state when possible;
- stream diagnostics rather than materializing full histories by default;
- expose JAX memory-allocation and persistent-cache guidance for production
  sweeps;
- use multi-device parallelization first for independent scans, UQ ensembles,
  and linear batches before attempting nonlinear domain decomposition;
- only introduce custom kernels after profiling shows a persistent XLA
  bottleneck.

Differentiable geometry and optimization
----------------------------------------

After the refactor/testing lane is stable, the differentiable geometry plan is:

1. Add an in-memory ``vmec_jax`` bridge into the existing SPECTRAX-GK geometry
   contract without writing or reading ``wout`` files.
2. Validate geometry fields against the existing VMEC-file path.
3. Add ``booz_xform_jax`` where Boozer-coordinate quantities are needed.
4. Add gradient checks for geometry-to-observable paths.
5. Promote examples from sensitivity analysis to inverse design, uncertainty
   quantification, and stellarator optimization only after the derivative
   checks and benchmark artifacts are frozen.

Testing and CI
--------------

The package-wide coverage target remains 95%, but coverage alone is not the
goal. Tests should map to equations, numerical schemes, diagnostics, artifacts,
benchmark observables, or differentiability contracts.

CI tiers:

- pull requests: type checks, fast test shards, docs build, package build, and
  release-surface coverage;
- main/manual: wider package coverage and selected artifact checks;
- workflow dispatch: full local validation suite;
- office/manual: GX parity, VMEC/W7-X validation, runtime/memory sweeps, and
  multi-GPU checks.

Documentation and examples
--------------------------

Documentation should remain user-first at the top level: install, run the
executable, plot an output file, inspect diagnostics, and reproduce shipped
figures. Longer benchmark caveats belong in the verification and benchmark
pages.

Examples that should stay maintained:

- default Cyclone linear run and plotting;
- Miller geometry;
- VMEC imported geometry;
- W7-X and HSX nonlinear runs;
- plotting from output files;
- autodiff inverse/UQ examples;
- profiling and memory diagnostics;
- parallelization examples.

Release policy
--------------

PyPI publishing is handled through the tag-driven GitHub release workflow.
Release notes should distinguish closed lanes from open paper lanes. Open
research artifacts can be included in the roadmap, but should not be described
as validated until their scripts, outputs, references, and gates are frozen.
