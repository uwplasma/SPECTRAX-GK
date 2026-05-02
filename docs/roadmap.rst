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

Pre-release scope
-----------------

The pre-release lane is limited to work that can be bounded, tested, and
documented without changing the physics claim surface:

- keep the refreshed runtime/memory panel complete for Cyclone, Cyclone
  Miller, KBM, W7-X, and HSX release rows;
- tighten case-specific nonlinear window-statistics gates where the frozen
  reference windows support it;
- strengthen autodiff validation with finite-difference checks, sensitivity
  map conditioning, and UQ covariance metadata;
- add the Phase-A ``vmec_jax`` / ``booz_xform_jax`` bridge contract into the
  existing sampled flux-tube geometry interface;
- land production parallelization first for independent ``k_y`` scans and UQ
  ensembles, with serial numerical-identity gates. The first closed artifact
  is ``docs/_static/parallel_ky_scan_gate.png`` from
  ``tools/generate_parallel_ky_scan_gate.py``;
- keep nonlinear hot-path optimization profiling-driven and tied to existing
  window-statistics and exact-state gates.

Current pre-release status snapshot:

- runtime/memory and nonlinear atlas figures include W7-X and HSX release rows;
- targeted nonlinear coverage now exercises explicit diagnostic branches,
  Hermitian projection, fixed-mode frequency extraction, IMEX nonlinear terms,
  and scalar/gyroaveraged electromagnetic bracket components;
- autodiff UQ validation includes finite-difference demo checks, closed-form
  Gauss-Newton covariance checks, rank-deficient sensitivity-map checks, and an
  explicit rejection path for empty parameter maps;
- solver-objective geometry-gradient validation now includes an actual
  electrostatic linear-RHS implicit-eigenpair gate for growth rate, real
  frequency, ``<k_perp^2>``, linear heat/particle-flux weights, and a
  mixing-length heat-flux proxy with respect to solver-ready geometry arrays,
  plus a mode-21 full-chain ``vmec_jax`` state-coefficient to
  ``booz_xform_jax`` to SPECTRAX-GK eigenfrequency-gradient artifact at
  ``docs/_static/vmec_boozer_solver_frequency_gradient_gate.png`` and a
  matching quasilinear heat-flux-weight gradient artifact at
  ``docs/_static/vmec_boozer_quasilinear_gradient_gate.png``;
- the Phase-A differentiable-geometry bridge is an in-memory sampled
  flux-tube contract with 100% targeted coverage, optional
  ``vmec_jax`` / ``booz_xform_jax`` discovery, tracer-safe mapping into
  ``FluxTubeGeometryData``, real ``vmec_jax`` metric-tensor derivatives, a
  real non-axisymmetric VMEC field-line tensor derivative through
  ``vmec_jax.geom`` plus ``vmec_jax.vmec_bcovar``, a direct VMEC
  tensor-derived flux-tube mapping derivative, a direct-VMEC-tensor vs
  imported-VMEC/EIK array-parity audit, a Boozer equal-arc core parity audit
  that closes the ``bmag``/``bgrad``/``gradpar``/Jacobian, zero-beta
  ``gds*``/``grho`` metric convention, and zero-beta loaded drift convention at
  release tolerance, a separate ``mboz=nboz=21`` QH/QI/tokamak parity matrix
  artifact at ``docs/_static/vmec_boozer_parity_matrix.png``, a real
  ``vmec_jax`` ``VMECState`` to ``booz_xform_jax`` to SPECTRAX-GK derivative
  gate, and a
  tracked AD-vs-finite-difference inverse/UQ artifact at
  ``docs/_static/differentiable_geometry_bridge.png``;
- production parallelization is currently claimed only for independent
  ``k_y``/batch/UQ-style workloads and the sharded linear RK2 identity path,
  not nonlinear domain decomposition;
- profiling remains manifest-driven through
  ``tools/performance_optimization_manifest.toml`` and the runtime/cold-start
  profilers before any hot-path optimization claim is made.

Executable open-lane status
---------------------------

The active post-``v1.5`` research lanes are now summarized by
``tools/build_open_research_lane_status.py``. The generated artifact is a
claim-scope gate, not a substitute for the underlying physics figures. It reads
the W7-X zonal, W7-X fluctuation-spectrum, quasilinear holdout, differentiable
geometry, and nonlinear-profiler artifacts and reports whether each lane is
closed, partial, open, or blocked.

.. image:: _static/open_research_lane_status.png
   :alt: Open research lane status summary

The current snapshot has no closed broad manuscript lanes in this group:
W7-X fluctuation spectra, differentiable geometry, and profiler identity checks
are partial bounded diagnostics, while W7-X long-window zonal recurrence and
absolute quasilinear flux promotion remain open. This keeps the README/docs
claim surface honest while still preserving publication-ready diagnostic panels
for the pieces that are already reproducible.

Current manuscript-scope readiness is tracked separately by
``tools/build_manuscript_readiness_status.py`` because W7-X zonal recurrence
and TEM/kinetic-electron extensions are intentionally deferred from this
manuscript. In that narrower scope, the quasilinear lane is closed as a
validated diagnostic/model-selection result rather than as an absolute-flux
predictor, VMEC/Boozer equal-arc geometry parity is closed at
``mboz=nboz=21``, and the reduced differentiable stellarator ITG optimization
examples are closed with AD/FD gates. The production solver-objective gradient
lane now has a passed actual linear-RHS gate at the solver-ready geometry
contract plus passed mode-21 VMEC/Boozer state-to-solver eigenfrequency and
quasilinear heat-flux-weight gates on QH and Li383 holdouts. The remaining
promotion step is now the nonlinear-window state-gradient gate and converged
nonlinear audits of optimized equilibria; those are required before claiming a
production nonlinear heat-flux stellarator optimizer.

The latest public CI run for commit ``5790e0e`` passed repo hygiene, mypy,
quick shards, docs/packaging, fast coverage, and the full wide-coverage matrix.
The combined wide-coverage job reported ``TOTAL 16134 787 95%`` package-wide
coverage. Some individual modules still sit below ``95%`` because the gate is
package-wide; notably ``nonlinear.py`` and ``zonal_validation.py`` remain useful
targets for future targeted physics tests.

.. image:: _static/manuscript_readiness_status.png
   :alt: Manuscript-scope readiness status summary

The latest W7-X zonal follow-up is
``docs/_static/w7x_zonal_hypercollision_probe_kx070.png``. It varies constant
Hermite hypercollision at fixed paper-facing initializer and normalization.
The stronger ``nu_hyper_m=0.03`` row reduces the final Hermite-tail fraction to
about ``0.099`` but leaves the mean trace error near ``0.289`` and the
late-window envelope about ``4.3`` times the digitized reference. The next
step is therefore a physically motivated velocity-space closure/operator
study, not a normalization change or another single-parameter constant-damping
scan.

The W7-X fluctuation/TEM extension lane is tracked by
``docs/_static/w7x_tem_extension_status.png`` and the TEM branch audit
``docs/_static/tem_branch_parity_audit.png``. The nonlinear fluctuation
spectrum estimator is closed as a simulation diagnostic with ``76`` samples,
but the TEM branch audit remains far outside any publication parity envelope:
``max |rel gamma|≈4.25``, ``max |rel omega|≈3.3`` away from the near-zero
reference denominator, one growth-rate sign mismatch, three frequency sign
mismatches, and an inverted frequency-branch ordering. No multi-alpha,
multi-surface, or kinetic-electron W7-X nonlinear windows are admitted yet.
These are now explicit blockers before broad W7-X/TEM validation or
optimization claims.

Post-release scope
------------------

The following remain post-release manuscript lanes until their literature
contracts and gates close:

- W7-X zonal long-window damping, recurrence, and closure under paper-facing
  normalization;
- W7-X fluctuation-spectrum experimental extension: the simulation-spectrum
  panel now has a reproducible estimator and gated artifact, but density and
  zonal-frequency comparison through a Doppler-reflectometry transfer function
  remains post-release;
- W7-X multi-flux-tube and TEM extension before broad stellarator-validation
  claims.

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

The next physics lanes should be closed in this order after the pre-release
scope above is stable:

1. W7-X zonal-response residual and late-envelope closure. The VMEC-backed
   SPECTRAX-GK artifact now uses the paper-facing potential initializer,
   signed line-average observable, line-first normalization, and no hidden
   time-axis scaling. The long-window comparison reaches the digitized
   stella/GENE time windows for all four wavelengths, but residuals fail at
   ``k_x rho_i=0.07``, ``0.10``, and ``0.30`` and the late envelopes are much
   larger than the digitized traces. The tracked TOML keeps
   ``gaussian_width=1`` because the benchmark source writes the initializer as
   ``exp[-(z-z0)^2]``; wider profiles and non-unit time scales are explicit
   audits only. The runtime now preserves final samples under strided
   diagnostics, aborts checkpointed artifact runs on the first non-finite
   diagnostic chunk, and preserves signed zonal line/mode diagnostics across
   external restart continuation. A four-wavelength ``Nl=16``, ``Nm=64``,
   ``dt=0.05`` refresh reached raw runtime ``t≈100`` with finite signed traces,
   so longer restart-continued W7-X traces can now be used to study the
   remaining physics/numerics issue directly. A constant-Hermite
   hypercollision follow-up reduced moment-tail energy but did not close the
   trace residual or late envelope, so the next sweep should vary the
   closure/operator physics rather than simply increasing constant damping.
2. Tighten the now-materialized windowed nonlinear-statistics panel beyond the current ``0.10`` release gate where the literature/reference windows justify stricter tolerances.
3. W7-X multi-flux-tube ITG/TEM extension and fluctuation-spectrum lane. The
   simulation-spectrum estimator is closed, while TEM linear parity,
   alpha/surface-resolved scans, and kinetic-electron nonlinear windows remain
   open in ``docs/_static/w7x_tem_extension_status.json`` and
   ``docs/_static/tem_branch_parity_audit.json``.
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
- use distributed parallelization first for independent scans, UQ ensembles,
  and linear batches before attempting nonlinear domain decomposition;
- only introduce custom kernels after profiling shows a persistent XLA
  bottleneck.

Differentiable geometry and optimization
----------------------------------------

After the refactor/testing lane is stable, the differentiable geometry plan is:

1. Replace the remaining local grad-:math:`B` drift closure with the
   production VMEC/EIK drift convention.
2. Add gradient checks for geometry-to-observable paths.
3. Promote examples from sensitivity analysis to inverse design, uncertainty
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
