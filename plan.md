# SPECTRAX-GK Ship Readiness Plan

Last updated: 2026-04-23
Current public baseline under review: `fb6fabc add large-grid scaling sweep and fix tools imports`

## Strategic Audit and Next-Step Roadmap (2026-04-23)

This section is the current decision layer for the project. It is based on:

- local git history through `678c3dd Reduce collision cache startup cost`,
- the current source/docs/tests layout on `refactor/modularize-core-for-validation`,
- live CI status on GitHub, where the latest `main` CI run observed in this
  audit completed successfully,
- the local `office` GX checkout under `/home/rjorge/GX`,
- local `vmec_jax` and `booz_xform_jax` checkouts under
  `/Users/rogeriojorge/local/`,
- a renewed literature/source pass over GX, stella/GENE W7-X benchmarks,
  Merlo/Rosenbluth-Hinton/GAM response, Tronko-style verification, ETG
  benchmark literature, DESC/TORAX differentiable-design patterns, and JAX
  performance/autodiff documentation.

### Main project goals

SPECTRAX-GK should become a research-grade, JAX-native gyrokinetic code that is:

1. accurate against literature and independent-code benchmarks,
2. end-to-end differentiable for sensitivity analysis, inverse design,
   uncertainty quantification, and stellarator optimization,
3. fast enough that cold-start, warm throughput, memory use, and multi-device
   scaling are all measured and actively optimized,
4. easy to run from `pip install spectraxgk` with a documented executable,
   plotting workflow, examples, and artifact format,
5. maintainable by researchers who need readable equations, tests, diagnostics,
   and failure modes, not just black-box benchmark figures.

### External anchors from the renewed pass

- GX remains the closest algorithmic and parity reference: it uses
  Fourier-Hermite-Laguerre phase-space methods, GPU-native CUDA kernels, and
  benchmark panels for CBC, KBM, W7-X, nonlinear transport, velocity-space
  convergence, and performance/scaling. The local GX source confirms several
  implementation lessons worth carrying into SPECTRAX-GK: preallocated work
  buffers, fused CUDA kernels for linear/nonlinear RHS paths, explicit linked
  parallel-gradient operators, and old but still useful unit-test topics
  around grids, geometry, gradients, Laguerre transforms, linear RHS,
  nonlinear RHS, moments, and solvers.
- The stella/GENE W7-X benchmark is still the canonical stellarator validation
  ladder: multiple flux tubes, linear ITG/TEM scans, zonal-flow response, and
  nonlinear ITG heat flux. SPECTRAX-GK should not present W7-X as closed from
  one flux tube alone when making paper-level claims.
- Merlo/Rosenbluth-Hinton/GAM remains the strongest shaped-tokamak response
  benchmark. The current Merlo Case-III artifact is close on benchmark-scale
  residual/frequency/damping; the remaining long-time recurrence work should
  be framed as a numerical-resolution/closure study rather than a replacement
  for the accepted extraction protocol.
- Tronko-style verification argues that equation verification and numerical
  verification have to be tied together. This means every solver refactor
  should be covered by tests on invariants, manufactured/closed-form limits,
  observed order, conservation/free-energy behavior, and benchmark observables.
- The ETG benchmark literature gives a real operating-point and transport
  context. Current ETG nonlinear work should stay framed as a pilot until it is
  tied to a recognized ETG benchmark window and transport observable.
- DESC and TORAX show the bar for differentiable plasma codes: exact or
  validated derivatives, clearly separated static/dynamic state, reusable
  objective APIs, progress/logging, persistent compilation cache guidance, and
  examples that solve real inverse/optimization tasks.
- JAX's official guidance reinforces the current performance direction:
  profile first, separate cold compile from warm throughput, use persistent
  compilation cache as an engineering option, manage GPU preallocation
  explicitly for memory/OOM work, use `shard_map`/SPMD patterns for real
  multi-device decomposition, and reserve Pallas for measured hotspots where
  XLA cannot generate the needed kernel.
- Equinox is a good fit for future typed PyTree model/config objects because
  `filter_jit` and related filtered transforms handle mixed static/dynamic
  PyTrees cleanly. Lineax is a plausible future path for differentiable
  matrix-free linear solves and adjoint-friendly linear algebra, but it should
  be introduced only behind a narrow solver adapter after a benchmark against
  the existing `jax.scipy.sparse.linalg.gmres` path.

### Current state from git and source layout

Recent work has closed real issues, not just documentation:

- runtime ETA/live output and adaptive chunk reporting were added in
  `c837a72`;
- imported W7-X linear geometry propagation was fixed in `3b6506f`;
- office/GX audits were stabilized and W7-X/HSX late-time linear reductions
  were tracked in `a6aed79` and `67dd533`;
- cold/warm runtime accounting, profiling tools, and trace defaults were added
  across `92952f5` through `a6e3dd1`;
- the most recent performance patch, `678c3dd`, reduced Cyclone
  `build_linear_cache` from about `7.74 s` to `6.92 s` on `office` GPU by
  storing the collision cache in low-rank form.
- the current validation-gate tranche adds JSON-ready scalar gate reports for
  late-time linear metrics, windowed nonlinear metrics, and zonal-response
  residual/frequency/damping metrics, with tests and API documentation; the
  Merlo/Miller zonal-response generator is now the first artifact script to
  write this gate report into its JSON metadata.
- the first refactor slice in the post-audit ship plan moved validation gate
  dataclasses and scalar acceptance-policy helpers into
  `src/spectraxgk/validation_gates.py`, while keeping the old
  `spectraxgk.benchmarking` and top-level compatibility exports intact.

The source tree is now organized around a credible target architecture:

- public user surfaces: `cli.py`, `runtime.py`, `runtime_config.py`,
  `runtime_artifacts.py`, `plotting.py`, examples, and documented tools;
- modular numerical kernels: `terms/*`, `linear.py`, `linear_krylov.py`,
  `nonlinear.py`, `diffrax_integrators.py`;
- runtime decomposition: `runtime_startup.py`, `runtime_diagnostics.py`,
  `runtime_chunks.py`, `runtime_results.py`;
- validation and artifact tools: `benchmarking.py`, `benchmarks.py`,
  `tools/compare_*`, `tools/make_*`, `tools/profile_*`;
- geometry bridge layer: `geometry`, `miller_eik.py`, `vmec_eik.py`,
  `from_gx/vmec.py`, `from_gx/miller.py`.

The remaining structure issue is that several large modules still mix physics,
numerics, user I/O, and benchmark policy. The refactor should continue, but
only with parity and coverage gates attached to each extraction.

### Current open-lane ordering before merge/tag/ship

The active pre-merge sequence is:

1. code refactoring with behavior-preserving compatibility exports,
2. better physics gates and validation artifacts,
3. 95% package-wide coverage with literature-anchored tests,
4. stronger multi-CPU and multi-GPU algorithms with numerical-identity gates,
5. `vmec_jax` / `booz_xform_jax` integration for differentiable geometry,
6. broader GX comparison examples and tracked reference artifacts,
7. documentation expansion for physics, equations, numerics, examples, and
   validation status,
8. close manuscript validation lanes: KBM branch continuity, real Cyclone
   velocity-space convergence, W7-X zonal response, W7-X fluctuation spectra,
   nonlinear window-statistics acceptance, autodiff FD/tangent/UQ gates,
   differentiable stellarator optimization, and nonlinear profiling/performance
   improvements,
9. merge to `main`, tag, publish, and ship only after the release branch has a
   clean CI/CD and artifact-gate story.

### Best next steps by priority

1. **Finish the refactor/testing lane before new feature expansion.**
   - Keep public behavior unchanged.
   - Continue splitting large modules only when each extraction gains tests.
   - Highest-value remaining slices: `runtime.py`, `linear.py`,
     `nonlinear.py`, `benchmarks.py`, `plotting.py`, and geometry adapters.

2. **Turn validation into a gated artifact matrix.**
   - Every paper-facing lane needs one owning script, one frozen artifact path,
     one reference source, one fit/window policy, and one numeric gate.
   - First-class scalar gates now exist for late-time linear metrics, windowed
     nonlinear statistics, and zonal response; next connect them to artifact
     refresh scripts.
   - Eigenfunction-overlap gates now exist and the KBM raw-overlay generator
     writes them into its JSON metadata, with the current bounded artifact kept
     explicitly open until it reaches the overlap/relative-L2 thresholds.
   - Nonlinear diagnostic comparison runs can now write JSON gate reports via
     `tools/compare_gx_nonlinear_diagnostics.py --summary-json`, giving the
     publication plots a machine-readable mean-relative-mismatch acceptance
     record.
   - The nonlinear summary writer now supports explicit case/source labels and
     strict JSON serialization. The first tracked nonlinear window gate JSONs
     cover Cyclone Miller, KBM, HSX, W7-X, and a short Cyclone diagnostic
     window. At the current `0.10` mean-relative release gate, Cyclone Miller,
     KBM, and HSX pass; W7-X remains open on `Wphi` at about `0.116`, and the
     short Cyclone diagnostic remains open because it is not yet the mature
     long-window transport acceptance artifact.
   - Observed-order and branch-continuity gate reports now exist for
     velocity-space convergence panels and branch-followed scan tables; the
     benchmark atlas summary already writes a high-vs-low Cyclone grid
     convergence gate, and the remaining step is wiring the observed-order and
     branch-continuity helpers into the relevant velocity-space and
     branch-following artifact refresh scripts.
   - `tools/generate_observed_order_gate.py` now provides the generic
     CSV-backed observed-order artifact path and writes the first tracked
     Cyclone resolution pilot to
     `docs/_static/cyclone_resolution_observed_order.json/png`. The tightened
     gate checks both final-pair order and all pairwise orders, so the current
     pilot is honestly marked open due a nonmonotone coarse-to-mid refinement
     even though the final-grid relative growth-rate error is small.
   - A bounded local Krylov probe for replacing that pilot was attempted at
     `ky=0.45` for `(Nl,Nm)=(4,8),(6,12),(8,16),...` and hit the 300-second
     cap before the higher-resolution points finished. The completed points
     were still nonmonotone in growth rate, so the replacement artifact should
     be generated from an office/GPU or cached manifest with explicit branch
     locking rather than another local compile-heavy sweep.
   - KBM branch-following now has a `--branch-summary-json` path in
     `tools/compare_gx_kbm.py`, so selected branch tables can record adjacent
     `gamma`/`omega` jump gates and successive eigenfunction-overlap gates.
   - The tracked KBM selected-branch table now also has a no-rerun refresh
     script, `tools/generate_kbm_branch_gate_summary.py`, which writes
     `docs/_static/kbm_branch_gate_summary.json`. The current summary keeps
     this lane open because the largest adjacent growth-rate jump is about
     `0.60` against the strict `0.50` gate, while the adjacent frequency and
     successive-overlap gates pass.
   - `tools/make_validation_gate_index.py` now scans tracked JSON metadata and
     writes `docs/_static/validation_gate_index.json/csv/png`, giving the
     manuscript/docs one compact audit view of currently materialized
     pass/open gate reports. The current index contains Merlo zonal response
     as passed, and the Cyclone resolution pilot plus KBM branch continuity as
     open.

3. **Close the next physics gates in this order.**
   - W7-X zonal-response artifact using VMEC-backed geometry and the same
     branchwise-extrema/Hilbert extraction used for Merlo.
   - KBM raw eigenfunction overlay, because the current bounded artifact has
     only about `0.63` overlap and is not manuscript-ready.
   - Windowed nonlinear-statistics panel for Cyclone, Miller, KBM, W7-X, and
     HSX.
   - W7-X multi-flux-tube linear/TEM extension and fluctuation-spectrum lane.
   - Shaped multispecies tokamak linear lane.
   - ETG nonlinear only after its benchmark operating point and observable
     contract are explicit.

4. **Make the test suite research-grade, not only coverage-heavy.**
   - Keep the 95% wide-package target, but require tests to map to:
     equations, numerical schemes, diagnostics, artifact contracts, benchmark
     observables, or gradients.
   - Add manufactured/closed-form tests for streaming, field solve,
     collision/hypercollision damping, linked-boundary gradients, ExB bracket
     antisymmetry, and reduced IMEX solves.
   - Add regression tests for every bug recently found: VMEC contract
     propagation, `s_hat_input` return, nonlinear diagnostics horizon mixing,
     restart zonal scatter order, default geometry scalar allocation, and
     low-rank collision-cache shape handling.
   - Keep local default tests under the 5-minute expectation; move
     office/GX/reference-data runs to explicit manifests and CI/manual tiers.

5. **Attack performance from measured bottlenecks.**
   - Current cold-start priority: `compile_first_integrator_run`, then
     `gyro_bessel_cache` and `laguerre_cache`.
   - Current memory priority: avoid large closed-over constants, avoid
     materialized full-history traces by default, stream diagnostics, and
     expose memory allocator guidance.
   - Current warm-throughput priority: fuse nonlinear FFT/gradient/bracket
     paths, reduce gather/scatter density, donate buffers where possible, and
     keep scan shapes stable.
   - Use persistent compilation cache for engineering/repeated sweeps, but keep
     published cold and warm timings separate.
   - Evaluate Pallas only after XProf/HLO shows a stable kernel hotspot that
     XLA cannot fuse well.

6. **Define a real multi-device parallelization target.**
   - Stop treating sharding as a figure-only feature.
   - Define one production decomposition first: likely `ky` or batch/scan
     sharding for independent linear scans and ensemble/UQ; nonlinear domain
     sharding should come later because it requires communication in FFT and
     nonlinear bracket paths.
   - Use `shard_map`/SPMD-style tests on CPU locally and two-GPU `office`
     validation for GPU.
   - Gate on speedup, memory per device, and numerical identity.

7. **Move differentiability from demos to validated workflows.**
   - For each differentiated observable, add finite-difference checks,
     tangent/adjoint consistency where available, and conditioning diagnostics.
   - Promote the two-mode inverse example as the current identifiable baseline.
   - Add a UQ/Laplace example with covariance and propagated uncertainty.
   - Add a sensitivity-map example for `gamma`, `omega`, and one windowed
     nonlinear metric.
   - After the refactor/testing lane, start the `vmec_jax` Phase A bridge:
     in-memory `vmec_jax` output into the existing SPECTRAX-GK geometry
     contract, no `wout` write/read step.
   - Then add direct `vmec_jax -> booz_xform_jax.jax_api -> SPECTRAX-GK`
     geometry, with geometry parity and gradient checks before optimization
     claims.

8. **Tighten docs and examples around user workflows.**
   - Keep top-level docs focused on install, run, plot, inspect outputs, and
     reproduce shipped figures.
   - Move long benchmark caveats into `verification_matrix.rst` and
     `benchmarks.rst`.
   - Add example pages for:
     - plotting from output files,
     - Miller geometry,
     - VMEC imported geometry,
     - W7-X/HSX nonlinear runs,
     - autodiff inverse/UQ,
     - performance profiling,
     - parallelization.

9. **Keep CI/CD and PyPI boring and automatic.**
   - Current PyPI version observed in this audit is `1.2.0`.
   - Release workflow uses trusted publishing through `release.yml`; keep it
     tag-driven and verify metadata before publish.
   - CI should stay layered:
     - PR: type checks, fast shards, docs/package build, release-surface
       coverage;
     - main/manual: wide package coverage;
     - workflow-dispatch/manual: full suite and core coverage;
     - office/manual: GX parity, VMEC/W7-X, runtime/memory, multi-GPU scaling.

10. **Do not overclaim.**
    - Closed release lanes can be advertised.
    - Open paper lanes should stay labeled as open until their numeric gates
      and artifact scripts are frozen.
    - Digitized literature figures are acceptable for planning and sanity
      checks, but direct code-backed or published-table references should be
      preferred for gates.

## Research Validation / 95% Coverage / Differentiable Optimization Roadmap (2026-04-22)

This section supersedes the older ship-only focus when deciding what to build
next. The target is no longer just "release-ready"; it is:

1. package-wide line coverage at or above 95% on the wide-coverage lane,
2. a benchmark and verification suite that is publishable and reviewer-proof,
3. differentiable workflows that support sensitivity analysis, inverse
   problems, uncertainty quantification, and stellarator optimization.

This roadmap is anchored on four external baselines:

- **Verification methodology**:
  Tronko et al., *Verification of Gyrokinetic codes: theoretical background and applications*  
  <https://arxiv.org/abs/1703.07582>
- **Modern tokamak/stellarator benchmark set and solver behavior**:
  Mandell et al., *GX: a GPU-native gyrokinetic turbulence code for tokamak and stellarator design*  
  <https://arxiv.org/abs/2209.06731>
- **Published stellarator benchmark set**:
  González-Jerez et al., *Electrostatic gyrokinetic simulations in W7-X geometry: benchmark between the codes stella and GENE*  
  <https://arxiv.org/abs/2107.06060>
- **Differentiable-JAX gyrokinetic precedent**:
  *gyaradax: Local Gyrokinetics JAX Code*  
  <https://arxiv.org/abs/2604.06085>

Additional optimization/stellarator anchors:

- stella documentation and code structure: <https://stellagk.github.io/stella/>
- GX documentation: <https://gx.readthedocs.io/>
- DESC stellarator optimization / autodiff:
  <https://arxiv.org/abs/2203.17173>,
  <https://arxiv.org/abs/2204.00078>
- turbulence optimization in stellarators with GX + DESC:
  <https://arxiv.org/abs/2310.18842>
- SIMSOPT repository for optimization workflow patterns:
  <https://github.com/hiddenSymmetries/simsopt>
- PORTALS-style optimization/surrogate loop motivation:
  <https://arxiv.org/abs/2312.12610>
- linear multispecies shaped-tokamak benchmark set:
  <https://crppwww.epfl.ch/~sauter/benchmark/>
- ETG benchmark operating-point literature:
  Nevins et al., *Characterizing electron temperature gradient turbulence*,
  Phys. Plasmas 13, 122306 (2006)  
  <https://w3.pppl.gov/~hammett/gyrofluid/papers/2006/Nevins-ETG-Benchmark.pdf>
- stellarator residual-zonal-flow theory:
  Monreal et al., *Residual zonal flows in tokamaks and stellarators at
  arbitrary wavelengths*  
  <https://arxiv.org/abs/1505.03000>
- W7-X fluctuation-spectrum / Doppler-reflectometry comparison:
  González-Jerez et al., *Electrostatic microturbulence in W7-X: comparison of
  local gyrokinetic simulations with Doppler reflectometry measurements*  
  <https://arxiv.org/abs/2312.10221>
- global electromagnetic stellarator verification:
  Maurer et al., *Global electromagnetic turbulence simulations of W7-X-like
  plasmas with GENE-3D*  
  <https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/global-electromagnetic-turbulence-simulations-of-w7xlike-plasmas-with-gene3d/AFF0F24A1A52D397D7983BAB2E872E9F>
- low-dimensional geometry learning for stellarator turbulence optimization:
  Wei et al., *Low-dimensional geometry learning for turbulence prediction in
  optimized stellarators*  
  <https://arxiv.org/abs/2603.17366>

#### Literature baselines reviewed directly for figure planning

The plan below is based on direct inspection of the published figure sets, not
only on abstracts:

- GX JPP 2024:
  - nonlinear CBC heat-flux traces (figure 5),
  - W7-X linear `gamma(k_y)` / `omega(k_y)` panel (figure 6),
  - velocity-space convergence spectra (figure 9),
  - performance/scaling panels (figures 10-12).
- W7-X stella/GENE benchmark JPP 2022:
  - linear ITG/TEM scan panels,
  - zonal-flow response section,
  - nonlinear ITG heat-flux trace (figure 12 as cited by GX).
- W7-X Doppler-reflectometry comparison preprint 2023:
  - density-fluctuation amplitude trends,
  - fluctuation-frequency spectra,
  - zonal-flow spectral characterization.
- Merlo et al. shaped-tokamak benchmark:
  - residual-potential panel (figure 12),
  - GAM time-trace / envelope extraction (figure 13),
  - GAM frequency and damping summary (figure 14),
  - shaping dependence of residual and GAM metrics (figures 15-16).
- shaped multispecies tokamak benchmark / Rosenbluth-Hinton tests:
  - linear shaping scan,
  - non-zero ballooning-angle handling,
  - zonal-flow residual and GAM damping.
- GENE-3D electromagnetic stellarator verification:
  - heavy-electron linear/nonlinear verification before realistic-mass runs.
- gyaradax 2026:
  - inverse-problem and sensitivity-analysis figures are the immediate
    precedent for the autodiff validation narrative.

These reviewed figure families define what SPECTRAX-GK should reproduce or
adapt for a credible future manuscript.

#### Frozen raw-reference status (2026-04-22)

The repository now includes compact frozen GX raw-mode bundles for two closed
linear lanes:

- `docs/_static/reference_modes/kbm_linear_gx_ky0p3000.npz`
- `docs/_static/reference_modes/w7x_linear_gx_ky0p3000.npz`

These are extracted from real GX `.big.nc` field histories and are the first
checked-in raw eigenfunction references suitable for manuscript-grade overlay
figures. The remaining blocker for the first raw overlay panel is now only the
SPECTRAX side of the matched extraction, not reference availability.

The first full-resolution matched KBM SPECTRAX extraction on `office` using the
tracked GX contract and selected fit window did not finish inside a `420 s`
wall-clock budget. The next pass should therefore target a bounded raw-overlay
extraction path with the same physics contract but lower runtime cost, staying
below the hard `600 s` ceiling.

That bounded-cost KBM raw-overlay pass has now been run once and produced the
first matched SPECTRAX-side raw-mode artifacts:

- `docs/_static/reference_modes/kbm_linear_spectrax_ky0p3000.csv`
- `docs/_static/kbm_eigenfunction_reference_overlay_ky0p3000.png`

The run stayed within the explicit `600 s` budget only narrowly and the result
is **not** yet manuscript-ready: the current bounded extraction gives roughly
`0.63` normalized overlap and `1.74` relative `L^2` mismatch against the frozen
GX raw mode.

That first bounded attempt was then tightened in two ways:

1. the exact KBM transverse-grid contract was restored by inferring `Ny` from
   the GX `k_y` grid (`Ny = 3 * (nky - 1) + 1`),
2. the eigenfunction extraction window was corrected to use a late-time tail
   instead of the growth-fit window from `kbm_gx_candidates.csv`.

Even after those fixes, the bounded exact-contract run under the `600 s`
ceiling still gives only about `0.63` normalized overlap and `0.79` relative
`L^2` mismatch. That makes the remaining task concrete:

1. improve the SPECTRAX-side raw eigenfunction extraction quality without
   changing the KBM physics contract,
2. keep the bounded runtime below `600 s`,
3. only then promote the raw overlay figure from an open diagnostic to a paper
   figure.

### Planning Principles

- Tests must target **physics contracts**, **numerical contracts**, or
  **differentiation contracts**. Coverage-only tests are not sufficient.
- Every benchmark lane must define:
  - the governing model,
  - the observable being compared,
  - the reference code or literature source,
  - the time/window used for the comparison,
  - the acceptance tolerance.
- Every autodiff-facing workflow must define:
  - what quantity is differentiated,
  - what variable is differentiated with respect to,
  - how the gradient is validated,
  - what optimization or inference task that derivative enables.
- Every public example must be either:
  - **validated** and included in the research claim, or
  - **demoted** to demonstration status with that stated explicitly.

### Current Gap Map

#### A. Coverage gaps

The present wide-package bottlenecks are not in the already-hardened helper
layers; they are in the large solver and infrastructure modules:

- `src/spectraxgk/runtime.py`
- `src/spectraxgk/linear.py`
- `src/spectraxgk/nonlinear.py`
- `src/spectraxgk/benchmarks.py`
- `src/spectraxgk/diffrax_integrators.py`
- `src/spectraxgk/runtime_artifacts.py`
- `src/spectraxgk/diagnostics.py`
- `src/spectraxgk/from_gx/vmec.py`
- `src/spectraxgk/terms/assembly.py`
- `src/spectraxgk/terms/nonlinear.py`
- `src/spectraxgk/terms/operators.py`

Low-value ways to chase 95%:

- more integration-only solves,
- asserting incidental internal arrays,
- branch pokes without physics or numerical meaning.

High-value ways to reach 95%:

- manufactured solutions,
- observed-order tests,
- invariant and symmetry tests,
- regression tests on benchmark observables,
- runtime/result contract tests with monkeypatched kernels,
- gradient-consistency tests for differentiable paths.

#### B. Benchmark/validation gaps

The repo has many benchmark-facing assets, but the **research claim** is still
narrower than the inventory.

Current validated/publicly defensible lanes:

- Cyclone ITG linear
- ETG linear
- KBM linear
- W7-X linear
- HSX linear
- Cyclone nonlinear
- Cyclone Miller nonlinear
- KBM nonlinear
- W7-X nonlinear
- HSX nonlinear
- short-window full-GK ETG nonlinear pilot
- secondary instability short-window lane

Open or demoted lanes:

- TEM
- KAW runtime lane
- kinetic-electron Cyclone
- reduced `cETG` as a physics-complete benchmark

Missing validation dimensions relative to the literature:

- linear zonal-flow response,
- explicit eigenfunction-shape comparisons in addition to gamma/omega,
- multi-window nonlinear statistics instead of single traces only,
- cross-geometry trend tests,
- clearer electromagnetic validation progression for KBM / KAW / kinetic-electron cases.

#### C. Autodiff and optimization gaps

What exists now:

- `examples/theory_and_demos/autodiff_inverse_growth.py`
- `examples/theory_and_demos/autodiff_inverse_twomode.py`

What is still missing for a research-grade differentiable story:

- gradient tests against finite differences for all public differentiated observables,
- local sensitivity maps over physical parameters,
- uncertainty quantification with a documented covariance or posterior approximation,
- geometry-to-transport derivatives for non-axisymmetric configurations,
- optimization loops that actually drive a design variable,
- a staged path from local inverse demo to stellarator optimization.

### Workstream 1: Wide-Package 95% Coverage

#### Objective

Raise package-wide coverage to at least 95% without turning the suite into a
slow or fragile branch-chasing exercise.

#### Acceptance criteria

- `wide-coverage` CI job passes with **package-wide** coverage `>= 95%`.
- Each newly covered module has at least one test of one of these forms:
  - exact/manufactured solution,
  - symmetry/invariant test,
  - benchmark-observable regression,
  - contract/serialization test,
  - gradient-consistency test.
- no test file exceeds the local 5-minute cap.

#### Coverage plan by module

1. `runtime.py`
   - add startup/import/output path tests for:
     - default runtime loading,
     - imported geometry,
     - restart/restart-append behavior,
     - diagnostics off/on,
     - adaptive chunking truncation,
     - fixed-mode routing,
     - artifact writing and output path handling.
   - use monkeypatched solver kernels so these stay cheap.

2. `linear.py`
   - add manufactured linear systems with known growth rate and exact mode
     evolution.
   - extend observed-order tests for all explicit and IMEX time paths used
     publicly.
   - add symmetry-limit tests:
     - zero drive,
     - zero curvature,
     - purely streaming,
     - `k_y = 0`,
     - electrostatic/electromagnetic toggles.

3. `nonlinear.py`
   - deepen actual-step tests for:
     - diagnostics stride and windowing,
     - fixed-mode projection,
     - collision split,
     - adaptive vs fixed-step agreement on manufactured problems,
     - explicit vs IMEX contract agreement.
   - add conservation/sanity tests in reduced nonlinear settings where the
     expected qualitative behavior is known.

4. `benchmarks.py`
   - convert more runner tests from ad hoc assertions to the new
     `late_time_linear_metrics()` and `windowed_nonlinear_metrics()` gate
     utilities.
   - cover all scan families with:
     - invalid input branches,
     - solver fallback logic,
     - exact reference loading,
     - benchmark-observable extraction.

5. `diffrax_integrators.py`
   - extend observed-order tests from explicit ODEs to save-mode/state-return
     branches and streaming-fit branches.
   - add parity tests between small diffrax and native explicit paths on tiny
     manufactured systems.

6. `runtime_artifacts.py`, `diagnostics.py`, `plotting.py`, `io.py`
   - cover all artifact serialization, reload, and plotting code with
     lightweight synthetic diagnostics bundles.
   - these modules should be near-complete because they define the research
     artifact surface.

7. geometry/import bridge layers
   - `from_gx/vmec.py`
   - `from_gx/miller.py`
   - `geometry/*`
   - add parser, remap, normalization, and cut/remesh tests using tiny
     fixtures and synthetic equilibrium metadata.

#### Sequence

1. artifact + diagnostics + plotting
2. runtime + linear
3. nonlinear + diffrax
4. benchmarks
5. geometry/import bridges

### Workstream 2: Benchmark and Validation Matrix

#### Objective

Turn the current collection of examples and comparison scripts into an explicit
validation matrix with publishable acceptance gates.

#### Benchmark families to support

1. **Tokamak electrostatic linear**
   - Cyclone ITG
   - ETG
   - TEM
2. **Tokamak electromagnetic linear**
   - KBM
   - KAW
   - kinetic-electron Cyclone
3. **Tokamak nonlinear**
   - Cyclone ITG
   - Cyclone Miller
   - KBM
   - ETG pilot
   - secondary instability
4. **Stellarator linear**
   - W7-X ITG/TEM family
   - HSX
5. **Stellarator nonlinear**
   - W7-X
   - HSX
6. **Additional response tests from the literature**
   - linear zonal-flow response
   - branch-following scans in near-marginal cases

#### Existing and near-ready figure inventory

Existing figure families that are already plausible paper inputs and should be
preserved as manuscript candidates:

- `docs/_static/benchmark_core_linear_atlas.png`
- `docs/_static/benchmark_core_nonlinear_atlas.png`
- `docs/_static/gx_summary_panel.png`
- `docs/_static/gx_publication_panel.png`
- `docs/_static/kbm_eigenfunction_overlap_summary.png`
- `docs/_static/nonlinear_w7x_diag_compare_t200.png`

New raw-reference assets now available:

- `docs/_static/reference_modes/kbm_linear_gx_ky0p3000.npz`
- `docs/_static/reference_modes/w7x_linear_gx_ky0p3000.npz`

Immediate next manuscript-facing deliverables:

1. Improve the bounded-cost KBM raw overlay until the overlap/mismatch metrics
   are publication quality, using:
   - `docs/_static/reference_modes/kbm_linear_gx_ky0p3000.npz`
   - `docs/_static/reference_modes/kbm_linear_spectrax_ky0p3000.csv`
   - `docs/_static/kbm_eigenfunction_reference_overlay_ky0p3000.png`
   - `tools/generate_kbm_reference_overlay.py`
2. First raw phase-aligned W7-X overlay figure from:
   - `docs/_static/reference_modes/w7x_linear_gx_ky0p3000.npz`
   - a bounded-cost matched SPECTRAX eigenfunction extraction.
3. Windowed nonlinear-statistics figure summarizing the already-closed
   nonlinear lanes.
4. Zonal-flow / GAM response figure family with:
   - shaped tokamak Rosenbluth-Hinton residuals,
   - W7-X residual and damping envelopes,
   - one figure convention shared across geometries.
   - current stepping-stone artifact: `docs/_static/miller_zonal_response_pilot.png`
     from `tools/generate_miller_zonal_response_pilot.py`; keep it in the paper
     inventory only as a pending/failing diagnostic until the residual and GAM
     envelope match the Merlo Case-III acceptance window.
5. W7-X fluctuation-spectrum figure family aligned with the Doppler-
   reflectometry comparison literature.

#### Validation observables

For each linear lane:

- `gamma(k_y)`
- `omega(k_y)`
- selected eigenfunction shape in `z`
- branch continuity under parameter continuation where relevant
- residual-zonal-flow level and damping envelope for zonal-flow response lanes
- branch identity and window used for the fit
- phase-aligned eigenfunction overlap and relative `L^2`
- where relevant, Rosenbluth-Hinton residual and GAM damping

For each nonlinear lane:

- windowed mean/std/RMS of heat flux
- `Wphi`, `Wg`, optionally `Wapar`
- mode-envelope statistics
- where relevant, resolved spectra by `k_x` / `k_y`

#### Acceptance gates

Linear:

- baseline target: `rtol <= 1e-2`
- accepted near-marginal / low-`k_y` exceptions must be documented explicitly
- eigenfunction normalized overlap should be reported, not just plotted

Nonlinear:

- compare windowed statistics, not only pointwise traces
- baseline target:
  - `<= 1e-1` for release-level parity
  - `<= 5e-2` for mature lanes
- also require visual agreement of saturation trends and mode envelopes

#### Reference hierarchy

Use references in this order:

1. **published benchmark datasets**,
2. **GX / stella / GENE / SIMSOPT-adjacent reproducible runs**,
3. **repo-checked reference tables generated from those runs**.

Do not treat digitized literature curves as equivalent to direct code-backed
reference data when a reference code is available.

When a published paper introduces an observable that is not already encoded in
the repo, prefer implementing that observable explicitly (for example zonal-flow
residuals, damping envelopes, or fluctuation spectra) rather than approximating
it from a visually similar existing figure.

#### Example-to-validation ownership

Validate and retain:

- `examples/linear/axisymmetric/cyclone.toml`
- `examples/linear/axisymmetric/etg.toml`
- `examples/linear/axisymmetric/runtime_kbm.toml`
- `examples/linear/axisymmetric/runtime_kaw.toml`
- `examples/linear/non-axisymmetric/runtime_w7x_linear_imported_geometry.toml`
- `examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml`
- `examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml`
- `examples/nonlinear/axisymmetric/runtime_kbm_nonlinear_t100.toml`
- `examples/nonlinear/axisymmetric/runtime_etg_nonlinear.toml`
- `examples/nonlinear/non-axisymmetric/runtime_w7x_nonlinear_imported_geometry.toml`
- `examples/nonlinear/non-axisymmetric/runtime_hsx_nonlinear_vmec_geometry.toml`
- `examples/benchmarks/secondary_slab_workflow.py`

Needs explicit closure or demotion:

- `examples/benchmarks/tem_linear_benchmark.py`
- `examples/linear/axisymmetric/runtime_kaw.toml`
- kinetic-electron Cyclone reference/helper paths in `src/spectraxgk/benchmarks.py`
- reduced `cETG` examples as a benchmark claim

#### Missing benchmark assets to add

- linear zonal-flow response example
- explicit eigenfunction-comparison example
- one published W7-X linear TEM example matching the stella/GENE benchmark paper
- one kinetic-electron Cyclone audit deck with a frozen accepted horizon

#### Additional literature-anchored tests to add

These are not optional "nice to have" items if the goal is a stronger paper.
They come directly from what the benchmark and verification literature actually
uses as evidence.

1. **W7-X zonal-flow response**
   - The W7-X benchmark paper explicitly includes linear zonal-flow response
     calculations, not only ITG/TEM growth rates and nonlinear heat flux.
   - Add:
     - a reproducible zonal-flow response example,
     - a comparison metric on residual level / damping envelope,
     - tests on the extracted residual and damping timescale.
   - Current status:
      - reusable ``Phi2_zonal_t`` extraction/plotting tooling exists,
      - signed ``Phi_zonal_mode_kxt`` now exists in the diagnostics/output path,
      - the first case-specific stepping-stone artifact exists via
        ``examples/benchmarks/runtime_miller_zonal_response.toml`` and
        ``tools/generate_miller_zonal_response_pilot.py``,
      - the current frozen Miller artifact is now pinned to the actual Merlo
        Case-III Table-III parameters with an initial ion-density perturbation,
        zero gradients, adiabatic electrons, and ``kxρ_i≈0.05`` with
        ``ky=0``,
      - the artifact has moved off the old ``Nm=16``, ``dt=0.01``,
        ``t≈150`` pilot and onto the current better-resolved
        ``Nm=24``, ``dt=0.005``, ``t≈60`` setup because the old trace's late
        peaks were recurrence-contaminated,
      - using the Rosenbluth-Hinton first-sample convention now gives
        ``residual≈0.192`` against the Merlo Case-III Figs. 12/16 read-off of
        about ``0.19``,
      - the extractor now follows the paper more closely with a common
        pre-recurrence window ``t≈30``, separate positive/negative-extrema
        damping fits, and Hilbert-phase frequency extraction on that same
        window,
      - this gives ``ω_GAM R0 / v_i≈2.20`` against the paper-scale read-off
        near ``2.24`` and ``γ_GAM R0 / v_i≈-0.176`` against the Merlo
        read-off near ``-0.17``,
      - a follow-up recurrence audit now shows that pushing the same case to
        ``Nm=28``, ``Nl=4`` improves the late-time recurrence ratio from about
        ``0.60`` to about ``0.54`` and moves the frequency closer to the paper
        read-off, but it also over-damps the GAM to roughly
        ``γ_GAM R0 / v_i≈-0.192``,
      - a minimal ``hypercollisions_const`` ladder through ``10^{-4}`` is
        effectively inert for this case, and even ``10^{-3}`` only nudges the
        recurrence ratio to about ``0.589``, so weak hypercollisions do not
        outperform the clean higher-moment run,
      - the remaining numerical follow-up item is long-time recurrence in
        finite moment runs rather than the benchmark-scale Merlo
        residual/frequency/damping gate,
      - the external ``phiext_full`` source path remains covered as a runtime
        contract, but it is not the Merlo RH/GAM validation protocol,
      - the restart-checkpoint issue exposed by this zonal initial-density run
        is fixed and covered: NetCDF restart loading now scatters active
        radial modes using the same order used by the writer, so chunked
        long-window artifacts no longer zero the post-checkpoint zonal signal.

2. **Multiple W7-X flux tubes**
   - The published W7-X benchmark is not a single-point story.
   - Add:
     - multiple flux-tube cases,
     - at least one near-threshold flux tube,
     - a figure/table showing branch ordering across tubes.

3. **Cyclone/Dimits-threshold evidence**
   - The CBC literature and later GX benchmarking both rely on nonlinear heat
     flux and threshold behavior, not only one nonlinear trace.
   - Add:
     - a small `R/LTi` threshold or Dimits-shift style scan,
     - a reduced but explicit zonal-flow suppression benchmark,
     - tests on threshold ordering and qualitative regime separation.

4. **Velocity-space convergence**
   - The GX paper explicitly shows Laguerre/Hermite free-energy spectra and
     convergence tables for nonlinear CBC.
   - Add:
     - spectra-based convergence tests,
     - manuscript figures showing convergence of heat flux and free-energy
       spectra with `(Nl, Nm)`,
     - tests that the convergence trend is monotone enough in the resolved
       ranges.

5. **Eigenfunction-overlap metrics**
   - Reviewers will not be satisfied with gamma/omega only if mode-branch
     ambiguity exists.
   - Add:
     - normalized complex overlap,
     - phase-aligned `Re/Im` eigenfunction panels,
     - tests on overlap thresholds for accepted linear lanes.

6. **Electromagnetic branch-following**
   - For KBM and KAW-like cases, add tests that the tracked branch is actually
     the intended branch under parameter continuation.
   - Use:
     - continuation in `beta`,
     - overlap continuity,
     - frequency-sign and parity diagnostics.

7. **Secondary-instability growth extraction**
   - Keep the existing secondary lane, but add:
     - mode-by-mode uncertainty/fit-window sensitivity,
     - explicit sideband envelope checks,
     - documentation of the zero-frequency sideband handling.

8. **Stellarator geometry-response tests**
   - Add at least one test class around quasi-symmetry / zonal-flow behavior
     motivated by Sugama-Watanabe and later stellarator zonal-flow papers.
   - These need not be full expensive runs; reduced response calculations are
     enough if they are literature anchored.

9. **Shaped multispecies tokamak linear benchmark**
   - Add at least one literature-backed shaped-tokamak linear lane beyond
     circular CBC, using the published benchmark collection referenced by
     Sauter et al.
   - The point is not just more scans; it is to verify that geometry import,
     multispecies response, and branch tracking remain correct away from the
     simplified CBC limit.

10. **Published ETG operating-point benchmark**
    - The short-window ETG lane should be tied to a recognized ETG benchmark
      operating point and its expected transport/growth observables, not only
      to internal reference files.
    - Add at least one explicit validation note and figure against the
      established ETG benchmark literature.

11. **Stellarator nonlinear fluctuation diagnostics**
    - The W7-X literature increasingly reports not only heat flux but also
      fluctuation spectra and zonal components.
    - Add a future lane for frequency-spectrum or zonal-component comparison on
      nonlinear W7-X once the core heat-flux lane is frozen.

#### Other codes to mine for benchmark structure

Use these codebases/papers as structural references for what to compare and how
to present it:

- **GX**: linear CBC, KBM, KAW, W7-X; nonlinear Cyclone, KBM, W7-X, secondary
- **stella/GENE W7-X benchmark**: ITG, TEM, zonal-flow response, nonlinear heat flux
- **GYRO/GS2 historical CBC literature**: Dimits / threshold framing and
  electromagnetic CBC reference conventions
- **XGC-S / EUTERPE / GENE-3D stellarator papers**:
  useful for future geometry-response and electromagnetic-stellarator tests

The plan should explicitly prefer benchmarks that are reproducible locally from
GX or from published open-access datasets over tests that rely on digitized
figures alone.

### Workstream 3: Differentiable Physics / Autodiff Validation

#### Objective

Move from "autodiff is possible" to "autodiff is validated and useful for
research."

#### Acceptance criteria

Every differentiated observable must have:

- a finite-difference gradient check,
- a complex-step check where applicable,
- a statement of conditioning/identifiability,
- a reproducible example with saved figure and numeric summary.

#### Differentiable task ladder

1. **Sensitivity analysis**
   - gradients of `gamma`, `omega`, and selected nonlinear windowed metrics
     with respect to:
     - `a/LTi`, `a/LTe`, `a/Ln`,
     - `beta`,
     - collisionality,
     - geometry scalars (`q`, `s_hat`, Miller shaping),
     - selected VMEC/geometry descriptors.
   - deliverables:
     - local sensitivity curves,
     - gradient validation tables,
     - doc page with interpretation.

2. **Inverse problems**
   - current two-mode inverse demo becomes the baseline.
   - add:
     - three-parameter inverse with regularization,
     - noisy-observation case,
     - branch-aware fitting window.
   - deliverables:
     - recovery plot,
     - Hessian / covariance estimate,
     - identifiability discussion.

3. **Uncertainty quantification**
   - start with local Gaussian/Laplace UQ:
     - Jacobian-based covariance,
     - Hessian-vector products,
     - propagated uncertainty on `gamma`, `omega`, and transport windows.
   - then add ensemble or unscented transform tests on reduced problems.
   - deliverables:
     - confidence intervals,
     - parameter posterior approximation,
     - propagated output uncertainty plots.

4. **Stellarator optimization**
   - stage 1: optimize cheap proxy objectives on imported geometry or small
     local descriptors.
   - stage 2: couple to a differentiable geometry backend (`vmec_jax` where
     available, or DESC/SIMSOPT-compatible descriptors).
   - stage 3: optimize turbulence proxies or nonlinear windowed metrics.
   - deliverables:
     - one end-to-end optimization example,
     - gradient verification,
     - documented failure modes and regularization.

#### Additional differentiable research tasks to add

1. **Derivative validation hierarchy**
   - for each public differentiated quantity:
     - finite-difference check,
     - complex-step check when applicable,
     - tangent/adjoint consistency check when both are available.

2. **Uncertainty quantification examples**
   - one local Laplace-approximation example around a two-mode inverse problem,
   - one propagated uncertainty example on `gamma(k_y)` or on nonlinear windowed
     heat flux.

3. **Stellarator-shape sensitivity prototype**
   - start with a low-dimensional geometry parameterization,
   - compute sensitivities of linear growth rate and at least one nonlinear
     proxy,
   - show conditioning and regularization explicitly.

4. **Optimization workflow comparison**
   - align interfaces with DESC/SIMSOPT-style objective + constraint APIs so
     SPECTRAX-GK can be embedded cleanly in a larger optimization stack.

### Workstream 4A: Manuscript Figure Plan

The manuscript should be planned now, not after the tests are done.

The literature pass implies a concrete figure philosophy:

- tokamak claims should be anchored in CBC/ETG/KBM-style benchmark panels and
  threshold/convergence evidence, not just isolated traces;
- stellarator claims should be anchored in the W7-X benchmark paper's actual
  observable mix: linear scans, zonal-flow response, and nonlinear heat flux;
- autodiff claims should separate sensitivity/gradient correctness from inverse
  recovery and from optimization.

#### Core validation figures

1. **Linear benchmark master panel**
   - `gamma(k_y)` and `omega(k_y)` for:
     - Cyclone ITG
     - ETG
     - KBM
     - W7-X
     - HSX
     - one shaped multispecies tokamak lane if closed
   - accepted/demoted lanes clearly marked
   - manuscript note:
     this is the benchmark-summary figure that should visually match the
     conventions used in GX and the W7-X stella/GENE paper.

2. **Eigenfunction validation panel**
   - representative `Re(phi)` / `Im(phi)` and `|phi|` or overlap for:
     - Cyclone ITG
     - W7-X
     - KBM or Miller
   - include normalized overlap numbers and phase alignment in the caption

3. **Nonlinear transport panel**
   - heat flux traces for:
     - Cyclone
     - Cyclone Miller
     - KBM
     - W7-X
     - HSX
   - use matched windows and make both curves visible even when overlapping

4. **Windowed-statistics summary**
   - bar/table/point plot of windowed mean/std/RMS mismatch for the nonlinear
     closed lanes
   - this should carry the actual manuscript acceptance story, because it is
     more robust than eyeballing traces

5. **Velocity-space convergence panel**
   - Laguerre/Hermite free-energy spectra plus scalar convergence of transport
     for the nonlinear CBC/kinetic-electron case or best-available surrogate
   - this is directly motivated by the GX paper's convergence evidence and is
     stronger than just a resolution table

6. **Stellarator-specific validation panel**
   - W7-X multi-flux-tube linear comparisons
   - W7-X zonal-flow response:
     - runtime contract now exists via
       ``examples/benchmarks/runtime_w7x_zonal_response_vmec.toml``
     - panel generator now exists via
       ``tools/generate_w7x_zonal_response_panel.py``
     - the tool uses the same first-sample / branchwise-extrema /
       Hilbert-phase extraction policy as the Merlo lane
     - still needs a frozen VMEC-backed artifact and acceptance window on a
       machine with W7-X geometry access
   - HSX linear/nonlinear summary if that lane remains in the paper
   - if zonal-flow is not closed, the paper should say so explicitly instead of
     silently omitting it

7. **Performance panel**
   - runtime/memory on the closed benchmark set only
   - no weak figures that do not show meaningful speedup
   - keep CPU/GPU/parallelization panels separate from validation panels

#### Differentiable-physics figures

8. **Sensitivity-analysis figure**
   - local derivatives of `gamma` / `omega` or transport metrics with respect to
     key physical parameters

9. **Inverse/UQ figure**
   - two-mode inverse recovery,
   - covariance ellipse or uncertainty bands,
   - gradient validation inset

10. **Optimization figure**
   - low-dimensional stellarator objective reduction,
   - objective vs iteration,
   - gradient-consistency evidence

#### Figure-to-script ownership

Before manuscript drafting starts, each target figure must have one owning
script path and one artifact path. At minimum:

- linear master panel: `tools/make_benchmark_atlas.py`
- nonlinear transport panel: `tools/make_gx_summary_panel.py` and
  `tools/make_gx_publication_panel.py`
- windowed-statistics summary: add a dedicated script under `tools/`
- stellarator-specific validation panel: add a dedicated script under `tools/`
- sensitivity/inverse/UQ figures:
  `examples/theory_and_demos/autodiff_inverse_growth.py`,
  `examples/theory_and_demos/autodiff_inverse_twomode.py`, plus follow-on
  scripts for UQ and optimization

#### Figure policy

- every figure must have a script in `examples/` or `tools/`,
- every figure must state:
  - case,
  - model,
  - horizon/window,
  - reference code or paper,
  - acceptance status,
- do not include empty or redundant subplots,
- if curves overlap, make both visible through line style / ordering / insets.
- captions should explicitly say what is expected, what was measured, and what
  level of agreement was found.

### Workstream 4: Stellarator-Optimization Architecture

#### Objective

Make SPECTRAX-GK usable inside a modern optimization loop for stellarators.

#### External pattern to follow

- **DESC** shows how to expose geometry and equilibrium quantities through an
  autodiff-friendly stack.
- **SIMSOPT** shows practical optimization orchestration, constraints, and
  objective composition.
- **GX + DESC optimization** shows that turbulence metrics can be used inside
  stellarator design loops.

#### Architecture plan

1. **Geometry differentiation layer**
   - standardize geometry inputs into differentiable parameter vectors
   - prefer `vmec_jax` or DESC-derived differentiable geometry paths where
     available
   - keep imported `*.eik.nc` paths as non-differentiable frozen references

2. **Objective layer**
   - expose scalar objectives:
     - linear growth rate,
     - real frequency,
     - heat-flux window mean,
     - weighted multi-objective turbulence score
   - expose constraints:
     - geometry validity,
     - profile bounds,
     - resolution adequacy,
     - optimization trust region

3. **Derivative layer**
   - support:
     - `grad`,
     - `jacfwd/jacrev`,
     - Hessian-vector products,
     - checkpointed differentiation for memory control

4. **Optimization layer**
   - start with local deterministic optimizers:
     - L-BFGS-B
     - trust-constr
     - projected gradient
   - then add robust/noisy alternatives for nonlinear objectives.

5. **Research examples**
   - sensitivity map on W7-X or HSX local geometry
   - inverse fit to a target `gamma(k_y)` spectrum
   - uncertainty propagation through a linear scan
   - small stellarator shape optimization loop using a low-dimensional geometry
     parameterization

#### Post-refactor differentiable equilibrium/geometry program

This starts only after the current SPECTRAX-GK refactor/testing-creation lane
has landed. The point is not to add another geometry adapter. The point is to
replace the current host/file-oriented VMEC helper path with a JAX-native,
end-to-end differentiable geometry chain while preserving the frozen
`vmec-eik`/`gx-netcdf` paths as reference and fallback modes.

Current state to plan against:

- SPECTRAX-GK still builds VMEC geometry through
  `src/spectraxgk/from_gx/vmec.py`, i.e. a GX-style helper path around
  `wout_*.nc`-compatible data and `booz_xform(_jax)` object APIs.
- `vmec_jax` already exposes the exact seams we need:
  - `run_fixed_boundary`
  - `wout_from_fixed_boundary_run`
  - `state_from_wout`
  - `booz_xform_inputs_from_state`
- `booz_xform_jax` already exposes both:
  - an in-memory object path via `Booz_xform.read_wout_data(...).run_jax()`
  - a lower-level JAX path in `booz_xform_jax.jax_api`
- Therefore the correct long-term target is:
  `vmec_jax state -> booz_xform_inputs_from_state -> booz_xform_jax.jax_api -> SPECTRAX-GK geometry bundle`
  with no required NetCDF round-trip on the hot path.

#### Concrete integration phases

1. **Phase A: compatibility bridge**
   - Add an optional in-memory geometry path that accepts `vmec_jax` output
     (`FixedBoundaryRun`, `wout`, or `VMECState`) and feeds the current
     SPECTRAX-GK imported-geometry contract without writing a `wout_*.nc` file.
   - Keep the public runtime behavior conservative:
     - current `geometry.model = "vmec-eik"` path remains valid,
     - new differentiable path is opt-in,
     - existing frozen `*.eik.nc` and `gx-netcdf` artifacts remain the gold
       references for regression tests.
   - Goal:
     remove file I/O from local optimization/sensitivity loops before changing
     the core geometry numerics.

2. **Phase B: direct JAX Boozer geometry path**
   - Add a new module such as `src/spectraxgk/geometry/vmec_jax.py`.
   - Input:
     `vmec_jax` state or a `BoozXformInputs` bundle.
   - Internal path:
     - `vmec_jax.booz_xform_inputs_from_state(...)`
     - `booz_xform_jax.prepare_booz_xform_constants_from_inputs(...)`
     - `booz_xform_jax.booz_xform_jax_impl(...)`
   - Output:
     a JAX-native flux-tube geometry bundle that populates the same physical
     quantities currently produced by the VMEC/GX adapter:
     - `bmag`
     - `gradpar`
     - `gds2`, `gds21`, `gds22`
     - `gbdrift`, `gbdrift0`
     - `cvdrift`, `cvdrift0`
     - `jacob`
     - geometry metadata (`alpha`, `nfp`, field-line labels, etc.)
   - Keep the geometry output contract aligned with
     `FluxTubeGeometryData` so the solver stack does not need to care whether
     the geometry came from a file, GX-style adapter, or a differentiable JAX
     pipeline.

3. **Phase C: optimization-ready geometry API**
   - Expose a stable geometry-objective interface that maps:
     low-dimensional equilibrium parameters -> flux-tube geometry ->
     linear/nonlinear observables.
   - Add memory-control hooks needed for serious optimization:
     - checkpointing / rematerialization for long traces,
     - selective `jit`,
     - batching/vmap over surfaces, `k_y`, and parameter ensembles,
     - explicit cold/warm compile accounting.
   - Keep `vmec_jax` and `booz_xform_jax` as optional extras so the package
     remains usable in file-based mode on systems that only want frozen
     benchmarks.

#### Validation gates

1. **Equilibrium parity inheritance**
   - Do not duplicate the entire VMEC validation program inside SPECTRAX-GK.
   - Instead, explicitly inherit `vmec_jax` equilibrium parity from its
     VMEC2000-backed validation cases and document which upstream references are
     being relied on:
     - axisymmetric tokamak
     - QH
     - QA
     - QI
     - `lasym=True` coverage where available
   - SPECTRAX-GK should only re-test the geometry quantities and solver
     observables that sit downstream of equilibrium generation.

2. **Geometry parity**
   - For a fixed set of axisymmetric and stellarator cases, compare the new
     differentiable path against the current `from_gx/vmec.py` adapter and
     frozen `*.eik.nc` references.
   - Minimum case set:
     - circular/shaped tokamak
     - one QA/QH stellarator
     - HSX-like VMEC case
     - W7-X-like VMEC case
   - Minimum quantities:
     - `bmag`
     - `gradpar`
     - `gds2`, `gds21`, `gds22`
     - `gbdrift`, `gbdrift0`
     - `cvdrift`, `cvdrift0`
     - `jacob`
   - Acceptance policy:
     combine pointwise tolerances, weighted relative norms, and convention-aware
     checks for sign/phase/field-line shifts so we do not confuse coordinate
     conventions with actual physics errors.

3. **Derivative validation**
   - For geometry-derived scalars and solver outputs, require:
     - finite-difference checks,
     - complex-step checks where the path is holomorphic enough,
     - tangent/reverse consistency,
     - first-order Taylor remainder tests over a ladder of perturbation sizes.
   - Start with low-dimensional parameters:
     - boundary Fourier amplitudes,
     - `s`, `alpha`, or field-line labels where physically meaningful,
     - selected Miller-shaping surrogates for cross-validation against analytic
       geometry lanes.

4. **End-to-end physics validation**
   - Demonstrate that geometry derivatives propagate correctly into physics:
     - linear `gamma` and `omega` sensitivities on W7-X/HSX-like cases,
     - one reduced nonlinear windowed transport metric on a small stellarator
       case once the linear path is stable.
   - Acceptance must be based on observable agreement, not just internal
     geometry coefficient agreement.

5. **Performance and scaling**
   - Benchmark:
     - cold vs warm geometry generation,
     - cold vs warm end-to-end linear runs,
     - CPU and GPU paths separately,
     - memory footprint,
     - compile/runtime split.
   - Compare:
     - current file-backed VMEC path,
     - in-memory compatibility bridge,
     - fully JAX-native geometry path.

#### Research-grade example and figure program

1. **Geometry-to-growth sensitivity map**
   - Example:
     perturb a small set of equilibrium/boundary coefficients and report the
     Jacobian of `gamma(k_y)` and `omega(k_y)` for a stellarator case.
   - Deliverables:
     - sensitivity heatmap,
     - gradient-validation table,
     - figure-ready caption stating conditioning and trusted perturbation range.

2. **Inverse or target-matching geometry demo**
   - Example:
     recover a low-dimensional equilibrium perturbation that best matches a
     target `gamma(k_y)` or `omega(k_y)` signature.
   - This should be a real optimization over equilibrium parameters, not only a
     local transport coefficient fit.

3. **Uncertainty propagation**
   - Example:
     propagate uncertainty in boundary coefficients or equilibrium descriptors
     into `gamma(k_y)` and a small number of transport proxies using local
     covariance propagation first, then stochastic sampling if needed.
   - This should connect directly to the robust/stochastic stellarator
     optimization literature, not just to generic autodiff demos.

4. **Pilot turbulence-informed equilibrium optimization**
   - First objective:
     reduced linear growth proxy on a stellarator case.
   - Second objective after linear closure:
     windowed nonlinear transport proxy on a reduced case.
   - This is the clean SPECTRAX-GK analogue of the published GX+DESC
     turbulence-in-the-loop optimization work, but with a fully differentiable
     local geometry chain instead of a black-box equilibrium subprocess.

5. **Boozer-coordinate diagnostics example**
   - Add one example that uses `booz_xform_jax` outputs directly to visualize
     the geometry features that correlate with the gyrokinetic objective.
   - This is where Boozer harmonics, symmetry-breaking content, and turbulence
     response should be shown on the same page instead of as disconnected
     diagnostics.

#### Documentation, testing, and artifact requirements

- Add a dedicated doc chapter for differentiable geometry backends:
  - architecture,
  - dependency model,
  - validated cases,
  - unsupported modes / failure cases.
- Add a geometry validation page that maps:
  equilibrium source -> Boozer transform -> flux-tube coefficients -> solver
  observables -> tests / figures.
- Every research-grade example above must ship with:
  - a runnable script under `examples/` or `tools/`,
  - a saved JSON/NetCDF numeric summary,
  - a publication-ready figure path under `docs/_static/`,
  - at least one fast regression test and one slower marked integration test.

#### Literature-anchored use cases to keep explicit

- `vmec_jax` validation/optimization docs and examples define the equilibrium
  parity and exact-derivative baseline we should inherit, not re-invent.
- `booz_xform_jax` defines the in-memory Boozer transform route and the
  differentiable Boozer-spectrum API.
- DESC quasi-symmetry optimization is the model for exact-derivative,
  optimization-ready equilibrium workflows.
- GX + DESC nonlinear turbulence optimization is the model for where the
  turbulence objective layer should ultimately go.
- Adjoint and stochastic stellarator-optimization literature should guide the
  sensitivity, robustness, and uncertainty examples so they read as actual
  plasma-physics validation rather than generic autodiff demonstrations.

#### Sequencing rule

- Do not start this program before the current SPECTRAX-GK refactor/testing
  lane is stable.
- Once that lane is closed, start with Phase A and geometry parity before
  attempting a fully end-to-end optimization demo.
- Do not claim a fully differentiable stellarator-optimization workflow until:
  - geometry parity is frozen,
  - derivative checks pass on real equilibrium parameters,
  - at least one observable-level stellarator sensitivity figure is closed.

### Workstream 5: Documentation and Research Artifact Discipline

#### Objective

Make the code publishable as a research tool, not just runnable.

#### Deliverables

1. `docs/testing.rst`
   - split into:
     - verification methodology,
     - benchmark matrix,
     - acceptance tolerances,
     - open vs closed lanes.
   - add a subsection called `Literature Baselines Reviewed` listing the
     published benchmark papers and what observable each contributes.

2. `docs/theory.rst` / `docs/numerics.rst`
   - explicitly connect each operator and discretization choice to the tests
     that validate it.
   - add a table mapping equations/operators to source files and validation
     tests.

3. `docs/examples.rst`
   - mark each example as:
     - validated benchmark,
     - validated differentiable demo,
     - exploratory demo,
     - deprecated/demoted.
   - add a short note for each benchmark example stating whether its reference
     comes from literature, GX, stella/GENE, or internal frozen artifacts.

4. new `docs/autodiff.rst`
   - sensitivity analysis,
   - inverse design,
   - UQ,
   - optimization workflows,
   - gradient-validation methodology.

4a. new `docs/verification_matrix.rst`
   - one table per benchmark family,
   - closed/open/demoted status,
   - observable,
   - reference,
   - acceptance threshold,
   - artifact path.

4b. new `docs/code_structure.rst`
   - module boundaries,
   - runtime flow,
   - where each operator, diagnostic, and artifact writer lives,
   - how refactored modules map to tests.
   - include a `public API vs internal modules` section so future refactors do
     not leak unstable internals into examples/tests.

4c. new `docs/manuscript_figures.rst`
   - target paper figures,
   - owning scripts,
   - artifact paths,
   - data provenance,
   - acceptance status,
   - open issues before submission.

5. artifact discipline
   - every figure in README/docs must be reproducible from checked-in scripts
   - every published benchmark figure must declare:
     - case,
     - horizon/window,
     - reference,
     - acceptance status.

6. testing/code-structure documentation expansion
   - `docs/testing.rst` should become research-facing:
     - verification vs validation,
     - literature anchors,
     - benchmark-observable definitions,
     - numerical verification methodology,
     - gradient verification methodology.
   - `docs/architecture.rst` should be expanded with a real source-tree map and
     ownership of physics/numerics/IO layers.
   - add a `testing taxonomy` section:
     - unit tests,
     - numerical verification tests,
     - benchmark/validation tests,
     - autodiff tests,
     - regression tests.

### Workstream 6: Source-Tree Modularization and Testability Refactor

#### Objective

Create a dedicated refactor track that splits the current large source files
into smaller, testable, reviewable modules that match standard software
engineering practice, while keeping solver functionality, numerical behavior,
and parity contracts unchanged.

This work must happen on the dedicated branch:

- `refactor/modularize-core-for-validation`

#### Non-negotiable constraints

- no intentional physics-model change,
- no intentional numerical-contract change,
- no intentional benchmark-reference change,
- no intentional public parity drift against the currently accepted GX-backed
  lanes,
- no silent API break on the public executable/runtime surface.

Every refactor PR or checkpoint on this branch must satisfy:

- targeted unit tests for the extracted module,
- regression tests against the pre-refactor behavior,
- parity tests unchanged or tighter,
- docs/comments/docstrings updated together with the code move.

#### Modules that should be decomposed first

1. `src/spectraxgk/runtime.py`
   split into likely submodules such as:
   - runtime loading / startup
   - runtime execution wrappers
   - adaptive chunking helpers
   - diagnostics/result assembly
   - restart/output handling

2. `src/spectraxgk/benchmarks.py`
   split into:
   - reference data loading
   - scan-family runners (Cyclone / ETG / KBM / TEM / kinetic / stellarator)
   - fit-signal and fallback policies
   - benchmark result dataclasses

3. `src/spectraxgk/linear.py`
   split into:
   - linear parameter/cache setup
   - linear RHS/assembly-facing helpers
   - explicit integrators
   - diagnostics extractors
   - runtime-facing wrappers

4. `src/spectraxgk/nonlinear.py`
   split into:
   - nonlinear parameter/config setup
   - explicit step helpers
   - IMEX step helpers
   - diagnostics accumulation
   - GX-style resolved outputs

5. `src/spectraxgk/runtime_artifacts.py`
   split into:
   - serialization schema/dataclasses
   - NetCDF/HDF5/JSON writers
   - summary builders
   - plotting-facing artifact readers

6. `src/spectraxgk/diffrax_integrators.py`
   split into:
   - diffrax linear wrappers
   - diffrax nonlinear wrappers
   - save-mode adapters
   - helper utilities

7. `src/spectraxgk/diagnostics.py`
   split into:
   - scalar diagnostics
   - resolved diagnostics
   - flux diagnostics
   - window/statistics helpers

8. `src/spectraxgk/from_gx/vmec.py`
   split into:
   - file loading / dependency detection
   - geometry remap/cut helpers
   - VMEC/Boozer transforms
   - normalization/output adapters

#### Refactor method

For each file family:

1. freeze current behavior with regression tests,
2. identify cohesive internal APIs,
3. extract pure helpers first,
4. extract dataclasses/config structures next,
5. extract runtime/IO wrappers last,
6. preserve compatibility shims until the whole tree is migrated,
7. only remove old compatibility paths after all tests and docs are updated.

#### Test strategy for the refactor branch

Each extraction must be covered by four test layers:

1. **unit tests**
   - pure functions
   - validators
   - dataclass conversions
   - shape/contract checks

2. **numerical tests**
   - manufactured solutions
   - observed-order tests
   - symmetry/invariant limits

3. **physics/literature tests**
   - gamma/omega benchmark observables
   - nonlinear windowed transport observables
   - eigenfunction or mode-envelope behavior

4. **regression tests**
   - artifact compatibility
   - CLI/runtime contracts
   - unchanged outputs on tracked benchmark cases

#### Documentation/comments/docstrings policy

This refactor branch should also raise the internal documentation quality of
the codebase.

Requirements:

- every public function/class gets a concise docstring with:
  - purpose,
  - expected shapes/contracts,
  - units or normalization assumptions where relevant
- every internal helper that is nontrivial gets either:
  - a docstring, or
  - a short local comment explaining the algorithmic role
- comments should explain **why** the step exists, not restate the code
- physics-facing routines should identify the operator or equation term they
  implement
- benchmark-facing routines should identify the benchmark family and reference
  contract they correspond to

#### Acceptance criteria for the refactor branch

The branch is ready to merge only when:

- package-wide coverage is at or above 95%,
- the benchmark matrix still passes within the accepted tolerances,
- the current shipped examples still run,
- public artifact formats remain readable,
- autodiff demos still validate gradients and inverse recovery,
- docstrings/comments are present on all refactored public APIs,
- the source tree is materially easier to navigate:
  - smaller files,
  - fewer mixed responsibilities,
  - clearer module boundaries.

#### Concrete first refactor tranche

1. `runtime.py`
   - extract startup/loading helpers
   - extract adaptive chunk helpers
   - extract result assembly helpers

2. `benchmarks.py`
   - extract fit-window/signal policy helpers
   - extract scan-runner families into separate modules

3. `linear.py`
   - extract explicit integrator helpers and diagnostics helpers

4. carry over:
   - targeted unit tests,
   - parity regressions,
   - docstrings/comments for every newly extracted API.

Status on this branch:

- completed first extraction step:
  - startup/loading/build helpers moved from `src/spectraxgk/runtime.py`
    into `src/spectraxgk/runtime_startup.py`
  - `runtime.py` remains the public compatibility surface
  - compatibility aliases/wrappers were kept for geometry generation and
    restart loading so the existing regression tests and patch points remain
    stable
- validated locally after extraction:
  - `tests/test_runtime_helpers.py`
  - `tests/test_runtime_runner.py`
  - `tests/test_runtime_artifacts.py`

### Workstream 7: Concrete Execution Order

#### Phase 1: Close the measurement layer

- raise `runtime_artifacts.py`, `diagnostics.py`, `io.py`, and plotting to
  near-complete coverage,
- ensure every benchmark and autodiff example writes a stable machine-readable
  artifact bundle,
- make benchmark gate utilities the standard way to evaluate runs.

#### Phase 2: Close the solver-core coverage

- `runtime.py`
- `linear.py`
- `nonlinear.py`
- `diffrax_integrators.py`
- `terms/assembly.py`
- `terms/nonlinear.py`

Target after Phase 2:

- all core solver modules above 85%,
- package-wide coverage above 75%,
- no known untested runtime-result contracts.

#### Phase 3: Close the benchmark matrix

- freeze accepted reference datasets,
- add missing zonal-flow/eigenfunction/near-marginal cases,
- decide TEM/KAW/kinetic-electron Cyclone status honestly:
  - close,
  - or demote from headline validation.

Target after Phase 3:

- full benchmark matrix with acceptance tables,
- README/docs only claim closed lanes.

#### Phase 4: Validate differentiated observables

- sensitivity gradients,
- inverse problem recovery,
- covariance/UQ,
- gradient checks on geometry parameters.

Target after Phase 4:

- `docs/autodiff.rst` complete,
- research-grade autodiff examples shipped,
- every public derivative backed by a validation test.

#### Phase 5: Stellarator optimization prototype

- one small but end-to-end differentiable optimization example
  using imported or differentiable geometry,
- compare with DESC/SIMSOPT workflow expectations,
- document performance and conditioning limits.

Target after Phase 5:

- SPECTRAX-GK is usable as a validated local differentiable turbulence engine
  inside a stellarator optimization workflow.

#### Phase 6: Merge the modular refactor branch

- land the source-tree decomposition once the coverage and parity gates hold,
- remove compatibility shims only after the new module boundaries are stable,
- treat this as the software-engineering consolidation phase that makes future
  validation and optimization work sustainable.

### Immediate Next Actions

1. **Coverage**
   - keep pushing on:
     - `runtime.py`
     - `linear.py`
     - `nonlinear.py`
     - `benchmarks.py`
     - `diffrax_integrators.py`
     - `runtime_artifacts.py`
     - `diagnostics.py`
     - `from_gx/vmec.py`

2. **Benchmark closure**
   - formalize the current closed matrix in a machine-readable manifest,
   - add missing linear zonal-flow and eigenfunction overlap checks,
   - decide whether TEM and KAW remain public examples or become experimental.

3. **Autodiff**
   - add a sensitivity-analysis example before adding more inverse demos,
   - add gradient-vs-finite-difference tests for the current two-mode inverse,
   - add local covariance/UQ around the existing inverse examples.

4. **Optimization**
   - define the first low-dimensional stellarator objective and geometry
     parameterization,
   - prototype the geometry derivative path using a differentiable backend,
     preferring `vmec_jax`/DESC-compatible routes over legacy VMEC-only paths.

5. **Documentation**
   - split validated vs exploratory examples,
   - add benchmark acceptance tables,
   - add an autodiff/UQ/optimization documentation chapter.

## Current Ship Status

`main` is ready for shipment from a software-quality standpoint after the sampled-progress linear diagnostics fix and the 2026-04-08 upstream-regression recovery.

Validated gates on the current recovery pass:

- `python3 -m mypy src`: passed
- `python3 -m py_compile` on touched files: passed
- Targeted sampled-progress regressions: passed
- Runtime / CLI / nonlinear CI shard: `154 passed`
- Focused linear/runtime/CLI regression slice: passed
- Full pytest at pulled head before the narrow local fix: `613 passed, 3 skipped`
- Full pytest after the latest upstream merge and local recovery: `616 passed, 3 skipped`
- Sphinx HTML build with warnings as errors: passed
- Package build and `twine check dist/*`: passed

The current repo is clean and synced to `origin/main` after each pushed checkpoint.

## Test Strategy Update (2026-04-11)

Goal: keep default test runs fast while preserving full integration coverage.

Changes:

- Default `pytest` run now excludes integration tests (`-m "not integration"`).
- Heavy suites (`tests/test_benchmarks.py`, `tests/test_runtime_runner.py`, `tests/test_linear.py`, `tests/test_nonlinear.py`) are tagged as integration.
- Added `tools/run_tests_fast.py` to enforce a 5-minute per-file cap for local runs.
- Added plotting helper for nonlinear `*.out.nc` diagnostics and documented
  geometry (VMEC/Miller) and output plotting examples in the docs.

To run the full integration suite:

- `pytest -m integration`

Latest local integration run (2026-04-11):

- `pytest -m integration` => `175 passed, 1 skipped`

## Fixed In Latest Checkpoint

The latest runtime visibility changes introduced a sampled-progress regression in linear time integration: progress callbacks read `phi_out` before it was computed when `sample_stride > 1`.

Fixed paths:

- `src/spectraxgk/linear.py`: base sampled linear integrator
- `src/spectraxgk/linear.py`: sampled diagnostics integrator
- `tests/test_linear.py`: lower-level regression coverage
- `tests/test_runtime_runner.py`: public runtime regression coverage

The latest upstream merge (`d04d014`) then introduced two separate issues:

1. `run_cyclone_linear(..., solver="auto")` stopped falling back to Krylov
   when the time-path GX growth extractor returned no finite samples.
2. Several benchmark-facing linear runtime/example TOMLs drifted away from their
   parity-oriented collision contracts.

Recovered paths:

- `src/spectraxgk/benchmarks.py`: auto time-path failure now falls back to Krylov
- `examples/linear/axisymmetric/*.toml`: restored parity-facing collision contracts
- `tests/test_runtime_config.py`: locks the linear benchmark runtime examples to the expected collision contract

## Performance / Accuracy Check

Representative CPU examples compared against checkpoint `61aac05` show broadly unchanged runtime and identical printed outputs after the fix:

| Case | Runtime status | Accuracy status |
| --- | --- | --- |
| Cyclone linear | same order, no clear speedup | same printed `gamma`, `omega` |
| ETG linear runtime | same order | same known branch outlier |
| KAW linear runtime | same order | same known runtime-contract mismatch |
| Cyclone nonlinear short | same order | same printed `Wg`, `Wphi`, heat |

Conclusion: the sampled-progress fix itself did not materially change runtime.
A later upstream precision-policy/config update made some runtime examples
faster, but not in a parity-neutral way.

Precision-policy finding:

- With default precision, several runtime example outputs moved materially.
- With `JAX_ENABLE_X64=1`, current Cyclone, ETG, and KAW runtime examples reproduce the previous-head outputs exactly.

So the remaining runtime-example drift is primarily a precision-policy issue,
not a benchmark-builder or reference-data issue.

## Current TEM Status

The TEM benchmark still remains open after the x64 recovery work.

Using the exact published TEM table contract in x64 mode
(`dt=0.001`, `steps=2000`, fixed late window, `mode_method="z_index"`),
the current TEM branch at `ky=0.3` is still:

- current: `gamma ~= 4.67`, `omega ~= 1.17`
- reference: `gamma ~= 2.18`, `omega ~= 1.23`

Additional x64 discriminator:

- current EM sub-scales `(0.5, 0.5, 0.5)`: `gamma ~= 4.67`, `omega ~= 1.17`
- unit EM sub-scales `(1.0, 1.0, 1.0)`: `gamma ~= 4.07`, `omega ~= 0.63`
- EM response off `(0.0, 0.0, 0.0)`: `gamma ~= 4.62`, `omega ~= 1.51`

Conclusion from the x64/EM sweeps alone: TEM is not primarily a
precision-policy problem, and the added EM sub-scales do not by themselves
explain the remaining mismatch.

Additional TEM root-cause finding (2026-04-08):

- The shipped TEM lane is not GX-backed. ``src/spectraxgk/data/tem_reference.csv``
  is digitized from the literature via ``tools/digitize_reference.py``.
- The repo's present TEM case definition is not the same contract as the cited
  literature comparison. The current ``TEMBaseCase`` uses ``q=2.7``,
  ``s_hat=0.5``, and ``R/L* = 20``; the cited trapped-electron literature uses
  Cyclone-like parameters instead.
- Direct x64 reruns under the exact published TEM table contract show that the
  older parity-era commit ``9a2bd47`` is much worse than current under that
  same contract (``gamma ~= 64.68``, ``omega ~= -202.22``), so the remaining
  TEM issue is not a simple regression from a previously closed GX lane.

Revised TEM next step:

- Rebuild the literature case definition and its digitized reference
  consistently, or demote/remove TEM from the public stress panel.

## Kinetic-Electron Cyclone Recovery Notes (2026-04-08)

- The public kinetic mismatch lane had real contract drift in source:
  - ``run_kinetic_linear()`` was using full ``LinearTerms()`` instead of the
    electrostatic ``bpar=0`` contract already used by the kinetic scan helper.
  - kinetic benchmark helpers were not applying the GX-linked end damping or GX
    hypercollision reference contract.
  - ``KINETIC_KRYLOV_DEFAULT`` had drifted onto a TEM-like negative-frequency
    branch policy even though this lane is the GX kinetic-electron Cyclone ITG case.
- The public kinetic table builder also drifted away from the older tracked
  contract:
  - ``solver="time"`` instead of ``"krylov"``
  - ``Ny=16`` instead of ``Ny=12``
  - auto-windowing instead of the fixed late window
  - time/phi override path instead of the older fixed-window Krylov path
- Current source now restores the benchmark-side contract:
  - kinetic helpers use the GX electrostatic reference defaults
  - kinetic builder is back on the fixed-window Krylov path
  - tests now lock those defaults in place
- Early imported GX replay on the real GX kinetic-electron Cyclone output is
  materially healthier than the public mismatch table suggested:
  - ``mean_abs_gamma ~= 0.581``
  - ``mean_rel_gamma ~= 0.392``
  - ``gamma_last ~= 0.347`` vs ``gamma_ref_last ~= 0.466``
  - energies are effectively closed in the early window
- Interpretation:
  - the kinetic core is not obviously broken at the same level as TEM
  - the remaining public kinetic issue is now narrowed to late-window /
    branch-selection closure on top of the benchmark contract

Additional kinetic finding from the current recovery pass:

- The kinetic helper had also drifted away from the historical benchmark seed.
  Older parity runs seeded a constant electron-density moment at amplitude
  ``1e-3``. The current public helper was instead using the generic default
  kinetic init, i.e. a tiny Gaussian density seed.
- That seed drift materially changes the selected Krylov branch:
  - with the restored table contract but current default seed, the refreshed
    public kinetic table still lands on a catastrophic high-frequency branch
    for ``ky=0.3``--``0.5``
  - at ``ky=0.3`` and table-like resolution, forcing the historical constant
    density seed collapses that catastrophic branch and brings ``gamma`` close
    to the GX reference, although ``omega`` still needs follow-up closure
- Source now restores that historical seed only on the GX-reference kinetic
  helper path, and only when the caller is still using the exact default
  kinetic init. Explicit user init overrides are preserved.
- The current public kinetic scan contract also had a structural high-``k_y``
  representation bug: with ``Ny=12``, the actual spectral grid only represents
  positive ``k_y`` up to ``0.5`` before aliasing onto negative modes. That made
  the published ``k_y=0.6`` and ``0.7`` rows invalid on the public kinetic
  table builder. The table builder now uses a non-aliasing ``Ny`` derived from
  the reference scan length.
- Focused local branch sweeps on ``ky=0.4`` and ``0.5`` identified the next
  live discriminator: for the GX-reference kinetic helper path, a
  history-based shift performs materially better than the current target-based
  shift. The public kinetic helper and table builder now use that
  history-based shift on the GX-reference path.

Revised kinetic next step:

- Re-run the exact public kinetic table on the restored contract plus restored
  legacy reference seed and non-aliasing ``k_y`` grid.
- If the remaining rows are still open after the history-based shift, then the
  next fix is deeper kinetic branch targeting / species-seeding alignment
  against direct GX forensic replay, not more table-contract cleanup.

## Known Benchmark Mismatches To Track Post-Ship

Current checked-in public mismatch tables remain unchanged across the latest public update range:

| Lane | Current status |
| --- | --- |
| Cyclone ITG linear | close, low-`k_y` residual: max `|rel_gamma| ~= 0.099`, max `|rel_omega| ~= 0.129` |
| ETG benchmark linear | acceptable for current atlas: max `|rel_gamma| ~= 0.038`, max `|rel_omega| ~= 0.048` |
| KBM linear | moderate residual: max `|rel_gamma| ~= 0.113`, max `|rel_omega| ~= 0.135` |
| TEM linear | not ship-grade parity: max `|rel_gamma| ~= 4.25`, max `|rel_omega| ~= 352` |
| Kinetic-electron Cyclone linear | not ship-grade parity: max `|rel_gamma| ~= 0.982`, max `|rel_omega| ~= 0.231` |
| ETG runtime scan | known branch outlier remains in runtime example path |
| KAW runtime example | known runtime-contract mismatch remains |

These are numerical benchmark follow-up items, not blockers for the current sampled-progress software fix.

## New Work Items (2026-04-10)

- Autodiff validation example (Gyaradax-style inverse/sensitivity):
  - Goal: add a published-grade, autodiff-driven inverse or sensitivity example with polished figures.
  - Reference: https://arxiv.org/pdf/2604.06085
  - Plan:
    - Choose one robust, cheap observable to differentiate (e.g., linear growth rate at fixed `ky` or a short-window nonlinear diagnostic).
    - Implement an inverse problem or sensitivity sweep using JAX autodiff.
    - Generate a single consolidated figure with:
      - inferred parameter vs. ground truth
      - gradient check / finite-diff validation
      - uncertainty or local sensitivity summary
    - Deliver in `examples/` with reproducible script and `docs/_static` output.
  - Status (2026-04-10):
    - Implemented `examples/theory_and_demos/autodiff_inverse_growth.py`.
    - Generated `docs/_static/autodiff_inverse_growth.png` (+ PDF/CSV).
    - Documented in README + docs examples list.

- Multi-device parallelization for long runs:
  - Goal: add multi-device execution so long kinetic-electron and stellarator runs can be accelerated.
  - Requirements:
    - JAX `pmap` / `pjit` pathway for multi-CPU and multi-GPU.
    - Support for running on 2 GPUs in `office` and multi-core CPU on macOS.
  - Plan:
    - Implement a minimal multi-device path on one nonlinear case (Cyclone or KBM).
    - Add a CLI/runner flag to select the distributed mode.
    - Validate on macOS with multiple CPU devices (`XLA_FLAGS=--xla_force_host_platform_device_count`).
    - Validate on `office` with 2 GPUs.
  - Status (2026-04-10):
    - Added `TimeConfig.state_sharding` and `spectraxgk.sharding.resolve_state_sharding`.
    - Runners now pass `state_sharding` into diffrax integrators.
    - Documented `state_sharding` in inputs/numerics docs + README.
    - Added unit tests for sharding helper.
    - Validated on macOS CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=2`.
      - Diffrax linear run completed and produced expected `phi_t` shape.
      - Added `with_sharding_constraint` hooks and disabled diffrax `throw`
        when sharded to avoid XLA rematerialization warnings.
      - Large CPU strong-scaling pass (Ny=64, Nz=128, Nl=6, Nm=6, steps=220):
        - steps=120: 1 device 40.17s, 2 devices 35.81s (1.12x speedup)
        - steps=220: 1 device 72.49s, 2 devices 63.02s (1.15x speedup)
    - Validated on `office` with 2 GPUs using `PYTHONPATH=.../src`:
      - `jax.devices()` detected 2 CUDA devices.
      - Sharded diffrax linear run completed with expected `phi_t` shape.
      - Large GPU sweep (Ny=64, Nz=128, Nl=6, Nm=6, sample_stride=5):
        - steps=400: 1 GPU 24.45s, 2 GPUs 18.94s (1.29x speedup)
        - steps=800: 1 GPU 56.43s, 2 GPUs 31.55s (1.79x speedup)
        - steps=1200: 1 GPU 121.10s, 2 GPUs 64.91s (1.87x speedup)
      - Attempting steps=500 with sample_stride=1 on 1 GPU was killed (memory), so
        long GPU runs should use a coarser `sample_stride` to avoid OOM.
      - Speedup plot saved to ``docs/_static/scaling_speedup.png``.

## Current Lane Order (2026-04-09)

Working order agreed for the next parity pass:

1. Cyclone Miller linear
2. HSX linear
3. KBM linear
4. Cyclone nonlinear
5. W7-X nonlinear
6. HSX nonlinear
7. ETG nonlinear
8. KBM nonlinear
9. Kinetic-electron Cyclone

TEM is intentionally out of scope for this pass.

Current linear-lane status under that ordering:

- `Cyclone Miller linear`
  - Now acceptable for the current pass.
  - The remaining public low-`k_y` issue was narrowed to the imported-linear
    contract, not the nonlinear Miller solver core:
    - exact-state nonlinear Cyclone Miller audits remain extremely tight
    - the imported-linear comparator had been generating internal Miller
      geometry from the nonlinear example TOML's field-line grid
      (`ntheta=24`, `nperiod=1`, `y0=28.2`) instead of the GX linear input
      contract (`ntheta=32`, `nperiod=2`, `y0=20.0`)
  - Fixed in `83d112b`.
  - Public Miller refresh pipeline also restored in `b91d599`:
    - refresh now writes the raw scan to `cyclone_miller_linear_scan.csv`
    - then derives the legacy panel table from the scan
  - Corrected targeted `ky=0.05` row under the fixed contract:
    - `gamma = 0.012058`, `gamma_ref = 0.012291`, `rel_gamma = -1.89e-2`
    - `omega = 0.028325`, `omega_ref = 0.028820`, `rel_omega = -1.72e-2`
  - Corrected targeted `ky=0.10` row under the fixed contract:
    - `gamma = 0.032866`, `gamma_ref = 0.032862`, `rel_gamma = 1.36e-4`
    - `omega = 0.058837`, `omega_ref = 0.058881`, `rel_omega = -7.44e-4`
  - Current published maxima after the refresh:
    - `max |rel_gamma| ~= 0.058`
    - `max |rel_omega| ~= 0.017`
  - This satisfies the agreed `rtol ~= 1.5e-1` low-`k_y` acceptance target.

- `HSX linear`
  - Acceptable under the current absolute-error framing.
  - The lane is still near-marginal, so relative growth error alone is not a
    good acceptance metric.
  - Current published maxima:
    - `mean_abs_gamma ~= 4.79e-03`
    - `mean_abs_omega ~= 3.75e-03`
    - `mean_rel_omega ~= 5.09e-02`
  - Unless a later refresh regresses these absolute metrics, no solver change
    is currently justified here.

- `KBM linear`
  - Already inside the requested tolerance envelope for this pass.
  - Current published maxima:
    - `max |rel_gamma| ~= 0.113`
    - `max |rel_omega| ~= 0.135`
  - No linear KBM code change is currently required for the agreed
    `rtol ~= 1.5e-1` target.

Current nonlinear-lane status at the handoff point:

- `Cyclone nonlinear`
  - Current tracked GX/SPECTRAX comparison remains acceptable for this pass.
  - Direct diagnostic comparison at `t <= 122` gave:
    - `mean_rel_abs(Wg) ~= 6.63e-2`
    - `mean_rel_abs(Wphi) ~= 6.84e-2`
    - `mean_rel_abs(HeatFlux) ~= 9.19e-2`
    - `final_rel(HeatFlux) ~= 7.79e-2`
  - Short GX-cyclone replay (GX `cyclone_salpha_short.in`, `dt=0.05`,
    `t_max=5`, collisions off, diagnostic stride 1) was partly a replay-config
    drift rather than a kernel-level defect.
  - The main contract differences were:
    - the short GX input uses `p_hyper = 2`, while the ad hoc SPECTRAX replay
      had fallen back to the runtime default `p_hyper = 4`
    - the short GX input does not use linked-end damping, while the ad hoc
      SPECTRAX replay still carried the longer production `damp_ends_amp = 0.1`
      contract
  - Re-running the short replay with the explicit short contract
    (`examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_short.toml`)
    improves the comparison against the GX `.out.nc` to:
    - `mean_rel_abs(Wg) ~= 2.61e-1`
    - `mean_rel_abs(Wphi) ~= 2.11e-1`
    - `mean_rel_abs(HeatFlux) ~= 2.51e-1`
    - `final_rel(HeatFlux) ~= 3.09e-1`
  - This lane remains open, but it is no longer a broad "0.5 / 0.68" mismatch;
    the remaining gap is the residual short no-collision `k_y`-resolved drift
    after the short-reference contract has been restored.
  - The short replay required explicit `--diagnostics` to emit GX-style
    diagnostics (the default `.out.nc` only contains metadata without the
    Diagnostics group).
  - The localization pass is now closed with a resolved audit artifact
    (`docs/_static/nonlinear_cyclone_short_resolved_audit_t5.{png,csv}`):
    - `Wphi_kyst` remains the dominant mismatch, but the corrected short
      contract lowers the active-`k_y` mean-relative range to roughly
      `3.12e-1 .. 5.33e-1`
    - `Wphi_kxst` sits around `1.70e-1 .. 3.20e-1`
    - `HeatFlux_kxst` sits around `2.59e-1 .. 4.02e-1`
  - That closes the diagnostic-localization lane: the short replay drift is
    not a single zonal-`k_x` defect, but a broader `k_y`-resolved field-energy
    imbalance in the short no-collision replay.

- `Secondary linear (GX kh01)`
  - The multi-sample stage-2 comparison lane is now refreshed from a dense GX
    replay (`kh01a_shortdense.out.nc`, `nstep=200`, `nwrite=20`), which
    produces a populated `omega_kxkyt` history with 10 output samples.
  - The tracked `secondary_gx_out_compare.csv` has been rebuilt from that real
    GX output.
  - The original remaining `omega` mismatch turned out to be a contract issue:
    the comparison helper was mixing that short GX replay with a stage-2
    SPECTRAX run at `t_max = 100`.
  - The helper now defaults the SPECTRAX stage-2 horizon to the real GX
    `out.nc` final time in `out-nc` mode, and the secondary post-processing now
    uses the mode-trace fit for `gamma` but the diagnostic tail for `omega`.
  - Current refreshed outcome on the matched short window:
    - `max rel_gamma ~= 1.87e-4`
    - `rel_omega ~= 3.23e-4` and `9.92e-4` on the `k_y = 0.1` sidebands
    - the `k_y = 0` sidebands still show large *relative* `omega` because both
      codes are effectively zero there, but the absolute mismatch stays
      `O(1e-6)`
  - This lane is now closed for the current release pass.

- `W7-X nonlinear`
  - Refreshed publication-grade long-window comparison now uses the current
    cached `t ~= 200` SPECTRAX diagnostic CSV against the tracked GX
    `w7x_adiabatic_electrons.out.nc`.
  - Current refreshed long-window metrics:
    - `mean_rel_abs(Wg) ~= 9.24e-2`
    - `mean_rel_abs(Wphi) ~= 1.16e-1`
    - `mean_rel_abs(HeatFlux) ~= 7.55e-2`
    - `final_rel(Wg) ~= -1.69e-1`
    - `final_rel(Wphi) ~= -1.91e-1`
    - `final_rel(HeatFlux) ~= -1.26e-1`
  - Exact-state audit still closes tightly through the tracked startup / late
    dump window, so the remaining drift is in later-time nonlinear evolution,
    not the startup state or VMEC import itself.
  - The office parity manifests had also drifted off the real public example:
    several still pointed to the nonexistent
    `examples/linear/axisymmetric/runtime_w7x_nonlinear_vmec_geometry.toml`.
    Those manifests are now corrected and covered by tests so future restart /
    exact-state / device-parity audits are anchored to the shipped nonlinear
    W7-X configuration.
  - The programmatic imported-geometry wrapper had also drifted from the
    published TOML contract by silently disabling collisions. That wrapper is
    now aligned with the shipped example and covered by tests.
  - The shipped VMEC TOML was also not actually portable on a clean clone:
    it pointed at a machine-local relative `wout` path that does not exist in
    the repo. The public contract is now explicit through `W7X_VMEC_FILE`,
    matching the existing HSX pattern.
  - The exact-state audit runner also had a clean-clone subprocess bug:
    helper tools were launched from `tools/` without an absolute repo
    `PYTHONPATH`, so `spectraxgk` imports failed on a fresh checkout. That
    runner is now hardened to prepend absolute repo paths.
  - Clean `office` restart/continuation parity is now explicitly closed on the
    shipped W7-X VMEC lane:
    - `state max_abs = 0`
    - `state max_rel = 0`
    - `Wg/Wphi/heat/pflux` restart diagnostics all `abs = rel = 0`
  - Direct startup replay on the same corrected head also matches GX tightly:
    - `g_state max_rel ~= 1.33e-7`
    - `phi max_rel ~= 1.54e-6`
  - So the remaining open W7-X nonlinear issue is now narrowed further:
    long-window evolution drift remains, but it is not a startup-state bug and
    it is not a restart/continuation bug.
  - A separate public runtime-contract bug was also identified and fixed:
    the checked-in adaptive stellarator nonlinear TOMLs were still carrying
    `run.steps = 200`, which truncates the public W7-X/HSX VMEC runs (and the
    W7-X imported runtime TOML) well before `t_max` under adaptive stepping.
    Those caps are now removed and covered by tests.
  - The shipped W7-X/HSX nonlinear runtime TOMLs now also emit default
    `tools_out/...` diagnostics/summary artifacts, so long `office` parity
    runs no longer depend on ad hoc CLI output overrides.
  - A second runtime-entry bug was then closed on `main`: the direct Python
    runtime wrappers (`run_linear_case()` / `run_nonlinear_case()`) were still
    ignoring `cfg.output.path`, so example-driven W7-X/HSX runs printed live
    progress but silently discarded the artifact bundle. That wrapper path now
    honors the TOML output contract and is covered by runtime tests.
  - The nonlinear wrapper path now routes through the chunk-aware artifact
    helper, so long adaptive stellarator runs persist diagnostics after each
    completed chunk instead of only at the full-horizon exit.
  - Fresh `office` audit on the corrected bundle-backed runtime path confirms
    the first persisted chunk is already close through `t <= 55.3072`:
    - `mean_rel_abs(Wg) ~= 1.36e-2`
    - `mean_rel_abs(Wphi) ~= 1.30e-2`
    - `mean_rel_abs(HeatFlux) ~= 1.54e-2`
    - `final_rel(HeatFlux) ~= -1.10e-1`
  - Interpretation:
    - the corrected public W7-X nonlinear lane is healthy through the first
      long chunk
    - any remaining mismatch is now a later-window evolution question only
  - Deeper live `office` replay on the same corrected path remains materially
    better than the older public long-window trace through `t <= 108.2958`:
    - `mean_rel_abs(Wg) ~= 1.27e-1`
    - `mean_rel_abs(Wphi) ~= 1.14e-1`
    - `mean_rel_abs(HeatFlux) ~= 1.50e-1`
    - `final_rel(Wg) ~= 3.88e-2`
    - `final_rel(Wphi) ~= 4.78e-2`
  - Under the current acceptance target, W7-X nonlinear is now closed for the
    current release pass and stays in the benchmark/publication set.

- `HSX nonlinear`
  - Refreshed `t = 50` comparison metrics:
    - `mean_rel_abs(Wg) ~= 2.75e-2`
    - `mean_rel_abs(Wphi) ~= 3.61e-2`
    - `mean_rel_abs(HeatFlux) ~= 2.91e-2`
    - `final_rel(HeatFlux) ~= 8.27e-3`
  - Interpretation:
    - keep this lane in the public benchmark set
    - do not spend more solver/debug time here before the remaining nonlinear
      lanes are checked
  - The programmatic VMEC wrapper had likewise drifted from the published TOML
    collision contract; it is now aligned and covered by tests.
  - The user-facing wrapper surface had also drifted from W7-X:
    `hsx_nonlinear_vmec_geometry.py` still required a manual `--vmec-file`
    even though the checked-in runtime TOML already carried the intended
    config-backed contract. That surface is now being normalized so HSX and
    W7-X follow the same runtime-wrapper pattern by default.
  - Fresh bundle-backed `office` replay on the normalized wrapper path now
    closes at a similar level through `t <= 50.8582`:
    - `mean_rel_abs(Wg) ~= 3.88e-2`
    - `mean_rel_abs(Wphi) ~= 3.69e-2`
    - `mean_rel_abs(HeatFlux) ~= 8.66e-2`
    - `final_rel(Wg) ~= -3.36e-1`
    - `final_rel(Wphi) ~= -3.74e-1`
  - HSX nonlinear is now closed for the current release pass.

- `KBM nonlinear`
  - The first refreshed audit hit a concrete tooling/config bug before any
    physics comparison:
    - both `tools/exact_state_lanes.office.toml` and
      `tools/restart_gate_lanes.office.toml` still pointed at the nonexistent
      `examples/linear/axisymmetric/runtime_kbm_nonlinear_t100.toml`
      instead of the shipped nonlinear config under
      `examples/nonlinear/axisymmetric/`
  - Those office manifests are now corrected and covered by tests.
  - Corrected `office` startup audit on the shipped nonlinear config is now
    closed:
    - `g_state max_rel ~= 1.33e-5`
    - `phi max_rel ~= 1.30e-5`
    - `apar max_rel ~= 1.34e-5`
  - The `office` late dumped-state audit at `t ~= 8.00043` is also mostly
    closed on the same shipped config:
    - `Wg rel ~= 1.16e-4`
    - `Wphi rel ~= 1.09e-5`
    - `heat rel ~= 2.25e-4`
    - `pflux rel ~= 3.61e-4`
    - `Wapar rel ~= 5.00e-1`
  - The nonlinear term-dump comparator is now repaired for the KBM dump-grid
    path: it synthesizes missing `ky/kx` axes from the dump shape instead of
    shrinking to the reduced `out.nc` grid.
  - Direct KBM nonlinear term replay against `office` dump
    `gx_runs/kbm_call325` is now close on the startup state:
    - `dJ0phi_dx rms_rel ~= 1.15e-7`
    - `dJ0phi_dy rms_rel ~= 1.36e-7`
    - `j0phi rms_rel ~= 2.05e-7`
    - `j0apar rms_rel ~= 2.04e-7`
    - `exb_total`, `bracket_apar`, and `total` all match to tiny absolute
      error on the dumped state
  - Interpretation:
    - the immediate KBM nonlinear mismatch is no longer pointing at basic
      nonlinear term assembly
  - Direct KBM nonlinear RK4 partial-stage replay against `office` dump
    `gx_runs/kbm_init_call1` is now also close enough for the current pass:
    - startup `phi rms_rel ~= 7.73e-6`
    - startup `apar rms_rel ~= 7.65e-8`
    - partial-step `rhs_total rms_rel ~= 5.02e-3`
    - partial-step `rhs_linear rms_rel ~= 5.02e-3`
    - partial-step `phi rms_rel ~= 1.26e-4`
    - partial-step state deltas remain tiny in absolute value even where
      pointwise relative error is inflated by near-zero reference entries
  - Refreshed long-window `t = 100` CSV-backed comparison now also closes
    tightly:
    - `mean_rel_abs(Wg) ~= 9.33e-3`
    - `mean_rel_abs(Wphi) ~= 9.35e-3`
    - `mean_rel_abs(Wapar) ~= 9.34e-3`
    - `mean_rel_abs(HeatFlux) ~= 9.36e-3`
    - `mean_rel_abs(ParticleFlux) ~= 9.36e-3`
  - KBM nonlinear is now closed for the current release pass.

## Next Work Order

1. Treat Cyclone Miller linear, HSX linear, KBM linear, Cyclone nonlinear, HSX nonlinear, and W7-X nonlinear as acceptable for the current pass unless refreshed data regresses.
2. Full-GK ETG nonlinear is now closed as a shipped short-window pilot:
   - `/Users/rogeriojorge/local/SPECTRAX-GK/examples/nonlinear/axisymmetric/runtime_etg_nonlinear.toml`
   - `/Users/rogeriojorge/local/SPECTRAX-GK/examples/nonlinear/axisymmetric/etg_runtime_nonlinear.py`
   - it is two-species, electrostatic, nonlinear, and intentionally separate from reduced `cETG`
   - the matched ETG box uses `y0 = 0.2`, `ky = 5.0`, and `Lx = 1.25`
   - the startup mismatch was traced to a GX input-contract detail:
     - GX reads `init_single` from `[Expert]`, not `[Initialization]`
     - the audited GX pilot was therefore using the Gaussian startup branch
     - the shipped SPECTRAX pilot now matches that with `gaussian_init = true` and `init_single = false`
   - matched short-window audit (`Nx=10`, `Ny=22`, `ntheta=16`, `Nl=4`, `Nm=4`, `dt=1e-4`, `t_max=0.001`) now lands at:
     - `mean_rel_abs(Wg) ~= 1.31e-2`
     - `mean_rel_abs(Wphi) ~= 5.18e-3`
     - final `HeatFlux` within a few percent of GX
   - continuation from the GX restart also matches over the next 10 steps, so the ETG nonlinear operator is not the open issue anymore
3. Kinetic-electron Cyclone is explicitly deferred to a future pass and is not
   part of the present shipped validation/performance stack refresh.
4. Leave KAW and TEM out of the active parity-recovery path until the above GX-backed lanes are honestly closed.
5. Consider making `ruff` a future CI gate only after a dedicated lint cleanup; current repo-wide `ruff check .` still reports pre-existing style debt.

## Release Readiness Checkpoint (2026-04-10)

- Local QA on the current tree is green:
  - `python3 -m mypy src` -> clean
  - `PYTHONPATH=src:. pytest -q -o addopts=''` -> `645 passed, 3 skipped`
  - `python3 -m sphinx -W -b dummy docs docs/_build/dummy` -> passed
- The runtime/memory manifest has been narrowed to the shipped 1.0 lanes:
  - Cyclone ITG linear/nonlinear
  - ETG linear
  - KBM linear/nonlinear
  - W7-X linear/nonlinear
  - HSX linear/nonlinear
  - Cyclone Miller nonlinear
- Reduced `cETG` and out-of-scope `KAW` rows have been removed from the 1.0
  performance panel workflow.
- The `office` reference-runtime contract needed an explicit compatibility
  environment in the manifest:
  - add `HDF5_DISABLE_VERSION_CHECK=1` to the reference-code rows
  - retain the explicit shared-library path bundle for `cutensor`, `nccl`,
    `hdf5`, `netcdf`, and `gsl`
- The runtime/memory blocker is now closed:
  - the shipped 1.0 runtime/memory panel has been regenerated from a completed
    merged summary with all shipped rows present
  - the final shipped outputs are:
    - `tools_out/runtime_memory_results_ship.csv`
    - `tools_out/runtime_memory_summary_ship.json`
    - `docs/_static/runtime_memory_benchmark.png`
    - `docs/_static/runtime_memory_benchmark.pdf`
- The first low-risk adapter renames toward more physical/mathematical naming
  are now in place with compatibility shims preserved:
  - `GXMillerGeometryRequest` -> `MillerGeometryRequest`
  - `build_gx_miller_geometry_request()` -> `build_miller_geometry_request()`
  - `GXReducedModelContract` -> `ReducedModelContract`
  - `load_gx_reduced_model_contract()` -> `load_reduced_model_contract()`
  - `gx_default_cfl_fac()` -> `explicit_method_default_cfl_fac()`
  - `GXVmecGeometryRequest` -> `VmecGeometryRequest`
  - `build_gx_vmec_geometry_request()` -> `build_vmec_geometry_request()`
  - `GXDiagnostics` -> `SimulationDiagnostics`
  - `GXResolvedDiagnostics` -> `ResolvedDiagnostics`
  - `GXTimeConfig` -> `ExplicitTimeConfig`
  - the older GX-prefixed names remain as aliases so the broader rename can
    proceed incrementally without breaking the current release branch
- The next low-risk runtime-facing helper rename slice is also now in place with
  compatibility aliases preserved:
  - `gx_zero_shat_enabled()` -> `zero_shear_enabled()`
  - `gx_effective_boundary()` -> `effective_boundary()`
  - `gx_twist_shift_params()` -> `twist_shift_params()`
  - `apply_gx_geometry_grid_defaults()` -> `apply_geometry_grid_defaults()`
  - `gx_real_fft_kx()` -> `real_fft_ordered_kx()`
  - `gx_real_fft_ky()` -> `real_fft_unique_ky()`
  - `gx_real_fft_mesh()` -> `real_fft_mesh()`
  - `select_gx_real_fft_ky_grid()` -> `select_real_fft_ky_grid()`
  - the old names remain as aliases so comparison tooling and older tests keep
    working while the broader rename continues
- The earlier RC runtime/memory outputs were discarded after three overlapping
  local benchmark writers were found to be appending to the same files.
- The final runtime/memory sweep completed with 30 manifest rows. Most rows are
  now authoritative and preserved in:
  - `tools_out/runtime_memory_results_final.csv`
  - `tools_out/runtime_memory_summary_final.json`
- The completed clean sweep surfaced two separate stellarator runtime issues,
  and both are now fixed for the shipped panel:
  - local SPECTRAX stellarator nonlinear rows had still been using VMEC
    generation on `office`; the shipped performance rows now use imported
    `*.eik.nc` geometry instead
  - GX stellarator runtime rows (`w7x-linear`, `w7x-nonlinear`, `hsx-linear`,
    `hsx-nonlinear`) were crashing because the default `office` runtime mixed
    incompatible HDF5 / NetCDF libraries and also expected a `python`
    / `booz_xform` VMEC helper path that was not available
  - those GX rows now run under a consistent local `netcdf-c` / `hdf5`
    library stack and are patched onto the already-shipped `*.eik.nc` geometry
    files instead of live VMEC regeneration
- The final shipped stellarator runtime measurements now include:
  - `w7x-linear [gx]`: `34.43 s`, `1969.37 MiB`
  - `w7x-nonlinear [gx]`: `110.97 s`, `2186.76 MiB`
  - `hsx-linear [gx]`: `124.48 s`, `2117.12 MiB`
  - `hsx-nonlinear [gx]`: `135.75 s`, `2186.70 MiB`
  - `hsx-nonlinear [spectrax_cpu]`: `646.54 s`, `6368.16 MiB`
  - `hsx-nonlinear [spectrax_gpu]`: `49.25 s`, `2128.44 MiB`
- The runtime/memory runner now supports better long-batch recovery:
  - `--continue-on-error` keeps the batch moving across later rows
  - `--log-dir` writes per-row stdout/stderr logs for postmortem debugging
  - per-row CSV/summary persistence now happens after every completed row, so
    long office sweeps no longer lose all partial progress when a later row
    fails or the session is interrupted
- The runtime/memory manifest now rewrites mismatched reference-code `t_max`
  values on the fly inside the temp copy so the performance panel compares
  runtime-equivalent workloads instead of mixing short SPECTRAX runtime examples
  with full reference benchmark horizons.
- Follow-up GPU profiling on the same shipped short nonlinear cases now makes
  the cold-vs-warm picture explicit:
  - Cyclone nonlinear GPU: `warmup_time_s = 33.957`, `run_time_s = 15.054`
    versus the shipped cold panel row `38.27 s`
  - KBM nonlinear GPU: `warmup_time_s = 27.485`, `run_time_s = 9.725`
    versus the shipped cold panel row `44.33 s`
  - interpretation: these two short nonlinear runtime gaps are dominated by
    JAX startup/compile latency rather than by steady-state timestep throughput
  - the shipped runtime panel now overlays warm second-run timings as explicit
    markers on top of the cold wall-time bars wherever `run_time_s` is present
  - the runtime harness now carries manifest-level `profile_command` entries,
    so those warm timings come from the tracked benchmark contract itself
  - compile-side collision prefactors have been hoisted out of the jitted RHS
    assembly path so the previous slow XLA constant-fold warning at
    `terms/linear_terms.py:272` no longer appears in the tracked Cyclone GPU
  - the new startup profiler now shows the remaining cold-path bottlenecks
    explicitly on `office` GPU:
    - Cyclone startup total `36.78 s` after the low-rank collision-cache pass,
      dominated by `compile_first_integrator_run` (`22.39 s`) and
      `build_linear_cache` (`6.92 s`)
    - KBM startup total `32.23 s`, dominated by `compile_first_integrator_run`
      (`19.28 s`) and `build_linear_cache` (`7.73 s`)
    - next runtime-performance tranche should therefore decompose
      `build_linear_cache` and the first nonlinear integrator compile path,
      not geometry/default setup
  - the `office` profiler environment is now fixed without adding TensorFlow:
    trace tools default to `python_tracer_level=0` and `host_tracer_level=0`,
    which removes the optional TensorFlow profiler-hook import noise while still
    emitting `.trace.json.gz` and `.xplane.pb` artifacts
  - the first `build_linear_cache` decomposition on `office` GPU for the shipped
    Cyclone short nonlinear case is now available:
    - low-rank collision caching reduced `collision_and_damping_cache` from
      `2.71 s` to `2.20 s`
    - the same pass reduced `build_linear_cache` from `7.74 s` to `6.92 s`
    - updated dominant subphases are now:
      - `gyro_bessel_cache`: `1.38 s`
      - `laguerre_cache`: `0.96 s`
      - `kperp_and_drifts`: `0.91 s`
    - next cache-build optimization should therefore move to the gyro/Laguerre
      cache path, while the broader startup path still needs first-integrator
      compile-surface reduction
  - concrete next optimization target: reduce the remaining compile/startup
    cost beyond the collision prefactor path, while keeping the current cold
    wall-time panel for honest end-to-end reproducibility

## CI/CD Status (2026-04-09)

- GitHub Actions `runtime-nonlinear` had two concrete regressions on current
  `main`:
  - `tests/test_cli.py` still expected the old relative `saved ...` message
    even though TOML output paths are now resolved and printed absolutely.
  - `tests/test_nonlinear.py` still used `jax.enable_x64()`, which is no
    longer available on the current JAX stack used in CI.
- Local CI-equivalent validation after the fixes:
  - `python3 -m mypy src` -> clean
  - runtime-nonlinear shard command from `.github/workflows/ci.yml` -> passed
- The full-GK ETG nonlinear smoke now shrinks the loaded pilot config to a
  CI-sized grid/time problem before calling `run_runtime_nonlinear()`. This
  preserves the shipped user-facing pilot while keeping GitHub runners below
  the nonlinear diagnostics memory ceiling.
- The remaining GitHub warning about Node.js 20 actions is not currently a
  failing CI condition; it is a future workflow-maintenance item.
