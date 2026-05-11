# SPECTRAX-GK Quasilinear Transport and Optimization Plan

Last updated: 2026-04-30
Active repository: `uwplasma/SPECTRAX-GK`
Historical planning archive: private repo `rogeriojorge/spectraxgk_plan`
Current public baseline: `main` at v1.4.0, with the historical ship-readiness log archived before this file was reset.

This file is both the active plan and the running log. Keep entries concise, dated, and tied to artifacts, tests, and figures.

## Current Goal

Bake a research-grade quasilinear transport capability into SPECTRAX-GK and use it as the reduced-model layer for differentiable stellarator optimization. The work must remain honest about what quasilinear theory can and cannot claim: linear weights and sensitivities can be exact within the implemented model; absolute saturated flux prediction requires saturation-rule calibration and nonlinear validation.

The target paper should show:

1. A JAX-native implementation of eigenfunction-resolved quasilinear heat and particle flux diagnostics.
2. A controlled comparison of saturation rules against nonlinear gyrokinetic runs across axisymmetric and non-axisymmetric cases.
3. Differentiable quasilinear objectives with finite-difference, tangent, and covariance validation.
4. A full `vmec_jax -> booz_xform_jax -> SPECTRAX-GK` pipeline for stellarator sensitivity analysis, uncertainty quantification, inverse design, and optimization.
5. Nonlinear audit runs that confirm where the reduced objective does and does not predict saturated transport trends.

## Literature Anchors From Final Pass

Use these references to define the paper claims and required figures:

- Parker et al. 2023, "Comparison of Saturation Rules Used for Gyrokinetic Quasilinear Transport Modeling": compares multiple saturation rules using linear gyrokinetic fluxes and discusses nonlinear/global comparisons. This is the primary saturation-rule benchmark template.
  https://www.mdpi.com/2571-6182/6/4/42
- Stephens et al. 2021, "Quasilinear gyrokinetic theory: A derivation of QuaLiKiz": self-contained derivation of a quasilinear gyrokinetic transport model; use for notation, flux weights, and transport channels.
  https://arxiv.org/abs/2103.10569
- Citrin et al. 2017, QuaLiKiz profile evolution: establishes the integrated-modelling use case and the importance of multi-channel heat, particle, and momentum transport.
  https://arxiv.org/abs/1708.01224
- Staebler, Bourdelle, Citrin, Waltz 2024 review: use for validation philosophy of reduced quasilinear models, cross-phases, dominant instabilities, and transport-property tests.
  https://www.ornl.gov/publication/quasilinear-theory-and-modelling-gyrokinetic-turbulent-transport-tokamaks
- Waltz, Casati, Staebler 2009: use to add quasilinear transport approximation tests with nonlinear spectral intensity, flux ratios, and tracer-style validation ideas.
  https://impact.ornl.gov/en/publications/gyrokinetic-simulation-tests-of-quasilinear-and-tracer-transport/
- Dudding et al. SAT3 paper and Sar et al. 2026 SAT3-NN preprint: use as evidence that modern saturation models are database-calibrated, spectrum-aware, and must be validated against held-out nonlinear data, especially kinetic-electron/TEM regimes.
  https://eprints.whiterose.ac.uk/188258/1/10_06_hgdudding_SAT3_paper_accepted.pdf
  https://arxiv.org/abs/2604.00462
- Jorge et al. 2023, "Direct Microstability Optimization of Stellarator Devices": primary stellarator quasilinear optimization anchor, especially `sum gamma / <k_perp^2>` and the need to compare against W7-X/HSX-like devices.
  https://arxiv.org/abs/2301.09356
- Kim et al. 2024, "Optimization of nonlinear turbulence in stellarators": primary nonlinear-optimization comparison; use for Birkhoff/window averaging, noisy objective handling, multi-surface and multi-field-line audits.
  https://arxiv.org/abs/2310.18842
- Gonzalez-Jerez et al. 2022 W7-X stella/GENE benchmark: canonical stellarator validation ladder with multiple flux tubes, ITG/TEM scans, zonal response, and nonlinear ITG heat flux.
  https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/electrostatic-gyrokinetic-simulations-in-wendelstein-7x-geometry-benchmark-between-the-codes-stella-and-gene/434DAA225FC5340D6E2C929C773644E3
- Maximum-J / density-gradient W7-X/HSX paper: use as a required kinetic-electron and nonlinear-stabilization stress test, because simple quasilinear or mixing-length rules can fail when stable-mode energy transfer or maximum-J physics matters.
  https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/turbulence-mitigation-in-maximumj-stellarators-with-electrondensity-gradient/CE3F398EB40134ADFF8D6D9E2758FCEF
- Tiwari et al. 2025 W7-X/QSTK zonal-flow paper: use to justify zonal-response and recurrence closure as paper-level stellarator physics, not only code validation.
  https://arxiv.org/abs/2501.12722
- Mandell et al. 2024 code paper and related reference-code examples: use only for independent-code validation and performance comparison where needed.
  https://arxiv.org/abs/2209.06731

## Honest Claim Ladder

Only make claims at the level supported by gates:

1. **Implemented diagnostic**: linear heat/particle flux weights are computed and phase/amplitude invariant.
2. **Validated linear physics**: growth rates, frequencies, eigenfunctions, and `kperp_eff` match reference cases within gates.
3. **Validated quasilinear weights**: species heat/particle weights, signs, and flux ratios agree with direct linear diagnostics and independent references.
4. **Trend-level saturation model**: uncalibrated saturation rules reproduce qualitative spectra, rankings, or gradient trends.
5. **Calibrated reduced model**: a fitted saturation rule predicts held-out nonlinear total fluxes and spectra with uncertainty intervals.
6. **Optimization proxy**: reducing quasilinear objective reduces held-out nonlinear flux in audited cases, with documented failures.
7. **Stellarator optimization claim**: optimization is multi-field-line and multi-surface, derivative-checked end-to-end, and audited by nonlinear runs.

Do not claim universal nonlinear flux prediction. Do not claim production stellarator optimization until multi-surface, multi-alpha, kinetic-electron and nonlinear audit gates pass.

## Ordered Execution Plan From 2026-05-10 Deep Audit

This section fixes the execution order for the next development cycle. Work
should be tackled in this order unless a blocking defect in an earlier lane
forces a short prerequisite patch:

1. parallelization;
2. refactor completion;
3. differentiable geometry extension;
4. docs and examples;
5. quasilinear absolute-flux promotion;
6. performance;
7. W7-X zonal long-window recurrence/damping;
8. W7-X fluctuation/TEM/multi-flux-tube extension.

The ordering is intentional. Parallelization touches solver state layout and
therefore should happen before further module extraction. Refactor completion
then makes geometry, quasilinear, documentation, and validation lanes easier to
maintain. Physics claim expansion comes only after the code paths are modular,
tested, and traceable.

### Source, Code, And Literature Audit Snapshot

Current SPECTRAX-GK state:

- Existing independent-work helpers live in `src/spectraxgk/parallel.py`.
  They cover `batch_map`, `ky_scan_batches`, padding, and deterministic
  ordering for independent scans and UQ-style workloads.
- Existing state-sharding helpers live in `src/spectraxgk/sharding.py` and
  `src/spectraxgk/sharded_integrators.py`. They provide `NamedSharding` /
  `PartitionSpec` state placement and fixed-step linear/nonlinear pjit scans.
- Current nonlinear config exposure is intentionally limited by
  `src/spectraxgk/runners.py` to `state_sharding in {auto, ky, kx, none}`;
  `z` is rejected because FFT-axis sharding has no identity gate.
- Current office two-GPU artifact
  `docs/_static/nonlinear_sharding_profile_office_gpu.json` passes final-state,
  final-field, and final-RHS diagnostic identity for active `auto`/`kx`
  sharding, but the best bounded `kx` engineering timing is about `0.96x`.
  It is therefore a correctness/profiler gate, not a production nonlinear
  speedup claim.
- Focused local validation on 2026-05-10 passed:
  `python -m pytest -q tests/test_parallel.py tests/test_sharding.py tests/test_sharded_integrators.py tests/test_generate_parallel_ky_scan_gate.py tests/test_profile_nonlinear_sharding.py tests/test_nonlinear_sharding_artifacts.py`
  with `30` tests passing.

GX source-code audit on `ssh office`:

- GX multi-GPU documentation states that parallelization is over species and
  Hermite indices, with species decomposition prioritized and exact
  divisibility constraints for `Nspecies`, `Nm`, and GPU count:
  https://gx.readthedocs.io/en/latest/MultiGPU.html
- `/home/rjorge/GX/src/grids.cu` implements this policy with `nprocs_s`,
  `nprocs_m`, `iproc_s`, `iproc_m`, species ranges, Hermite ranges, and
  `m_ghost` cells.
- `/home/rjorge/GX/src/moments.cu` exchanges Hermite ghost cells through MPI
  or NCCL with nearest-neighbor and second-neighbor transfers when required.
- `/home/rjorge/GX/src/solver.cu` performs reductions over the sharded
  species/moment layout, then broadcasts fields back to the devices.
- `/home/rjorge/GX/src/nonlinear.cu` keeps the spatial/spectral grid local
  while looping over the local Hermite block. This avoids distributed FFTs in
  the first production multi-GPU strategy.

JAX documentation audit:

- `shard_map` is the right API for explicit SPMD collectives and debugging
  because it exposes per-shard inputs, explicit `psum`, `ppermute`, and output
  sharding checks:
  https://docs.jax.dev/en/latest/notebooks/shard_map.html
- `pmap` works but current JAX docs recommend `shard_map` or `smap` for new
  code because `pmap` is implemented on top of `jit` and `shard_map`:
  https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html
- Multi-process/multi-host JAX requires `jax.distributed.initialize()` and the
  same collective order on every process:
  https://docs.jax.dev/en/latest/multi_process.html
- `Mesh`, `NamedSharding`, and `PartitionSpec` remain the common sharding
  vocabulary for automatic and explicit partitioning:
  https://docs.jax.dev/en/latest/jax.sharding.html

Other gyrokinetic/HPC anchors:

- The GX paper motivates the spectral Laguerre-Hermite velocity formulation and
  GPU-native implementation for time-to-solution and design studies:
  https://arxiv.org/abs/2209.06731
- GKW documents parallelization over the distribution-function grid and
  species, reinforcing that species/velocity decomposition is a standard
  continuum gyrokinetic path:
  https://www.sciencedirect.com/science/article/pii/S0010465509002112
- ORB5 and TRIMEG-GKX emphasize that production gyrokinetic codes need explicit
  data-structure refactors before portable CPU/GPU scaling claims:
  https://arxiv.org/abs/1908.02219
  https://arxiv.org/abs/2504.21837
- W7-X validation scope is anchored by the stella/GENE W7-X benchmark:
  https://arxiv.org/abs/2107.06060
- Quasilinear absolute-flux claims are anchored by Parker et al. saturation-rule
  comparisons and must remain calibrated/held-out, not universal:
  https://arxiv.org/abs/2308.09181
- Stellarator optimization claim scope is anchored by direct microstability
  optimization and nonlinear turbulence optimization:
  https://arxiv.org/abs/2301.09356
  https://arxiv.org/abs/2310.18842

Local differentiable-geometry audit:

- `/Users/rogeriojorge/local/vmec_jax` has fixed-boundary and free-boundary
  JAX VMEC workflows, quasisymmetry optimization examples, exact/discrete
  adjoint planning, and boundary-parameter APIs.
- `/Users/rogeriojorge/local/booz_xform_jax` has JAX-native Boozer transform
  APIs, VMEC-object input support, Jacobian sensitivity examples, and
  autodiff optimization examples.
- SPECTRAX-GK already has a large differentiable bridge in
  `src/spectraxgk/geometry/differentiable.py` and gradient gates in
  `src/spectraxgk/solver_objective_gradients.py`, but production nonlinear
  heat-flux transport-average gradients are still not promoted.

### Lane 1: Parallelization

Goal: convert the current identity-gated engineering sharding into a production
parallelization strategy that works on CPU and GPU while preserving physics
outputs exactly within gate tolerances.

1. Freeze the claim policy.
   - Keep current `kx`/`ky` whole-state pjit sharding as an engineering
     identity/profiling artifact only.
   - Do not use it for broad nonlinear speedup claims until larger matched
     CPU/GPU sweeps and profiler traces pass.
   - Keep `z`/FFT-axis sharding disabled in runtime config.

2. Formalize a `ParallelConfig`.
   - Add a config dataclass with fields for `strategy`, `axis`, `num_devices`,
     `batch_size`, `strict_identity`, `profile`, and `backend`.
   - Wire it through runtime scan, quasilinear scan, UQ ensemble, and
     optimization examples without changing default serial behavior.
   - Preserve `spectraxgk.batch_map` and `spectraxgk.ky_scan_batches` as
     public stable helpers.

3. Finish production independent-work parallelism.
   - Extend independent batching to linear `ky` scans, quasilinear spectra,
     saturation-rule sweeps, UQ covariance ensembles, and optimization
     population/member evaluations.
   - Add deterministic padding/trimming tests for scalar, vector, and pytree
     outputs.
   - Add serial identity gates for `gamma`, `omega`, eigenfunction phase-invariant
     norms, quasilinear heat/particle weights, and UQ covariance summaries.
   - Add local CPU logical-device artifact with
     `XLA_FLAGS=--xla_force_host_platform_device_count=2,4,8`.
   - Add office GPU artifact with `CUDA_VISIBLE_DEVICES=0,1`.

4. Prototype GX-inspired velocity/species sharding.
   - Add a new module, likely `src/spectraxgk/velocity_sharding.py`.
   - Implement a decomposition planner with GX-like priorities:
     species first, Hermite `m` second, optional Laguerre `l` only after
     species/moment gates pass.
   - Require exact divisibility or pad only when padding is mathematically
     invisible to the solver and diagnostics.
   - Keep `kx`, `ky`, and `z` replicated initially to avoid distributed FFTs.
   - Use `shard_map` for explicit control and `check_vma=True` where possible.
   - Use `lax.psum` for field-density/current reductions.
   - Use `lax.ppermute` or equivalent neighbor collectives for Hermite ghost
     exchange.

5. Add a linear RHS sharding gate before time integration.
   - Build minimal states with one and two species and several Hermite blocks.
   - Compare serial RHS vs sharded RHS for electrostatic adiabatic, full
     kinetic-electron electrostatic, and electromagnetic-coupling disabled and
     enabled paths where supported.
   - Gate density/current reductions and field replication separately.
   - Gate Hermite streaming/collision/closure ghost exchange separately.
   - Run locally on logical CPU devices and on office two-GPU.

6. Add fixed-step linear and nonlinear time gates.
   - Start with fixed-step RK2/RK3 only, not adaptive diffrax.
   - Gate one-step identity, multi-step identity, and diagnostic identity.
   - For nonlinear runs, compare heat-flux window means, window CV, free-energy
     change, `Wphi_kx/ky` spectra, and final state.
   - Use Cyclone first, then Cyclone Miller, KBM, W7-X, and HSX.

7. Add parallel autodiff gates.
   - Check `jax.grad`, `jax.jvp`, and `jax.vjp` through independent parallel
     scans and through the velocity/species sharded linear objective.
   - Compare AD vs central finite differences for `gamma`, `omega`,
     `kperp_eff`, quasilinear weights, and selected geometry-derived
     observables.

8. Add multi-host as post-single-host extension.
   - Use `jax.distributed.initialize()` only after single-host CPU/GPU gates
     pass.
   - Add SLURM/OpenMPI launcher examples for office or cluster environments.
   - Require all processes to execute identical collectives in identical order.

9. Parallelization exit gates.
   - `tests/test_parallel.py`, `tests/test_sharding.py`,
     `tests/test_sharded_integrators.py`, and new velocity-sharding tests pass.
   - `parallel_ky_scan_gate` passes on serial, local logical CPU devices, and
     office GPU.
   - New velocity/species linear RHS identity gate passes.
   - Nonlinear fixed-step identity gate passes at least Cyclone and one
     stellarator case.
   - CPU/GPU speedup curves are published only if identity gates pass.

### Lane 2: Refactor Completion

Goal: make the codebase easier to test and extend while preserving all physics
and benchmark behavior.

1. Freeze public behavior with compatibility tests.
   - Before each extraction, add a test against the current public API and
     artifact schema.
   - Keep compatibility exports in `src/spectraxgk/__init__.py`.

2. Split runtime orchestration.
   - Extract input/config parsing, run dispatch, restart handling, progress/ETA,
     artifact writing, and plotting hooks out of `runtime.py`.
   - Add tests for default executable behavior, default TOML selection, restart
     continuation, ETA output, and plot artifact dispatch.

3. Split linear assembly.
   - Move geometry cache construction, linked parallel derivative maps,
     field solves, velocity operators, and branch/frequency extraction into
     tested submodules.
   - Keep operator-level tests tied to equations and branch-continuity gates.

4. Split nonlinear assembly.
   - Separate bracket transforms, field solve calls, diagnostic extraction,
     Hermitian projection, spectral/grid Laguerre modes, and fixed-step
     integrators.
   - Preserve current profiler labels so performance artifacts remain
     comparable.

5. Split benchmark policy.
   - Break `benchmarks.py` into data loading, fit metrics, window metrics,
     gate reports, and figure builders.
   - Keep all benchmark tolerance policy machine-readable.

6. Refactor exit gates.
   - No public API breakage.
   - Full fast CI shards pass.
   - Package-wide coverage remains at or above `95%`.
   - Benchmark/readme figures regenerate with unchanged accepted metrics unless
     a deliberate physics bug fix is logged.

### Lane 3: Differentiable Geometry Extension

Goal: move from a closed zero-beta equal-arc bridge to a broader, derivative
checked `vmec_jax -> booz_xform_jax -> SPECTRAX-GK` optimization pipeline.

1. Stabilize backend discovery.
   - Keep explicit `SPECTRAX_VMEC_JAX_PATH` and
     `SPECTRAX_BOOZ_XFORM_JAX_PATH` overrides.
   - Add tests that prefer local editable checkouts over stale site packages.

2. Promote in-memory VMEC/Boozer contracts.
   - Avoid writing `wout`/`boozmn` files in differentiable examples unless the
     example is explicitly demonstrating compatibility output.
   - Require `mboz,nboz >= 21` for manuscript-facing Boozer gates.
   - Add finite-beta pressure/drift convention checks before finite-beta
     optimization claims.

3. Extend solver-objective gradients.
   - Keep current linear frequency and quasilinear gradient gates.
   - Add multi-surface and multi-alpha reduced-objective gradients.
   - Add local conditioning diagnostics for each differentiated observable.
   - Keep compact nonlinear startup-window FD artifacts labeled as plumbing
     only.

4. Add production nonlinear transport-gradient prerequisites.
   - Require post-transient heat-flux running averages.
   - Require window convergence in time and grid resolution.
   - Require AD/FD or adjoint/finite-difference agreement on the final
     promoted nonlinear observable.
   - Require nonlinear audits of optimized equilibria before claiming
     stellarator heat-flux optimization.

5. Geometry extension exit gates.
   - Geometry array parity passes for VMEC/EIK imports and in-memory
     VMEC/Boozer outputs.
   - AD/FD gates pass for `gamma`, `omega`, `kperp_eff`, quasilinear weights,
     and selected reduced nonlinear-window observables.
   - Docs clearly distinguish reduced differentiable objectives from full
     nonlinear transport-average objectives.

### Lane 4: Docs And Examples

Goal: make every accepted capability easy to run, inspect, reproduce, and cite.

1. Add a dedicated `docs/parallelization.rst`.
   - Cover independent batches, logical CPU devices, office/multi-GPU use,
     identity gates, current limitations, and profiler workflow.
   - Move long parallelization details out of `docs/performance.rst` while
     keeping performance plots there.

2. Add examples.
   - `examples/parallelization/linear_ky_scan_parallel.py`.
   - `examples/parallelization/quasilinear_uq_ensemble_parallel.py`.
   - `examples/parallelization/nonlinear_velocity_sharding_identity.py` after
     Lane 1 gates pass.
   - `examples/geometry/vmec_jax_boozer_flux_tube_bridge.py`.
   - `examples/quasilinear/calibrated_holdout_workflow.py`.

3. Add publication artifact docs.
   - Every figure gets a generator command, JSON/CSV companion, acceptance
     criteria, and claim scope.
   - README keeps only high-information figures; docs hold the full figure
     stack and caveats.

4. Docs/examples exit gates.
   - `sphinx-build -b html docs docs/_build/html -q` passes.
   - Example smoke tests run under bounded CI or are explicitly marked
     `slow`/manual with replay artifacts.
   - README claim surface matches `docs/_static/manuscript_readiness_status.json`
     and `docs/_static/open_research_lane_status.json`.

### Lane 5: Quasilinear Absolute-Flux Promotion

Goal: promote only saturation models that survive converged nonlinear holdouts
with uncertainty, and keep all weaker models scoped as diagnostics.

1. Freeze the accepted data contract.
   - Require converged nonlinear heat-flux windows after transient removal.
   - Require time-window and grid-resolution convergence before a case enters
     train/holdout calibration.
   - Keep circular, QH, CTH-like, or any other failed convergence cases
     excluded until they pass common-window and grid-refinement gates.

2. Expand saturation-rule comparison.
   - Keep simple mixing-length/Lapillonne/Bourdelle/Kumar-style rules as
     transparent baselines.
   - Keep richer spectrum-aware candidates separate from runtime TOML exposure
     until holdout uncertainty gates pass.
   - Compare spectra, total fluxes, flux ratios, cross-phases, and geometry
     transferability.

3. Add differentiated quasilinear workflows.
   - AD/FD for heat and particle flux weights.
   - UQ covariance for fitted saturation parameters.
   - Sensitivity maps over geometry and gradient parameters.
   - Optimization examples that demonstrate trend-level improvement and then
     nonlinear audits.

4. Quasilinear exit gates.
   - At least one train/holdout split with multiple geometries and nonlinear
     convergence evidence.
   - Prediction intervals calibrated and reported.
   - Runtime-exposed saturation models limited to those with passing gates.
   - README wording states whether outputs are diagnostics, trend-level models,
     or calibrated absolute-flux predictors.

### Lane 6: Performance

Goal: reduce runtime and memory with profiler-backed changes only.

1. Preserve measurement separation.
   - Cold compile, cache construction, first-step compile, warm runtime,
     output I/O, plotting, and memory must be reported separately.

2. Repeat profiler traces after parallelization/refactor.
   - CPU and GPU split profiles for Cyclone, Cyclone Miller, W7-X, HSX, and KBM.
   - Full nonlinear RHS traces with HLO token summaries.
   - Memory snapshots for representative nonlinear runs.

3. Optimize only confirmed bottlenecks.
   - Linear RHS fusion/cache layout.
   - Nonlinear bracket FFT/data movement.
   - Diagnostic streaming and output materialization.
   - Donation/persistent-cache guidance.
   - Custom kernels only after XLA traces show a stable bottleneck that JAX
     transformations cannot address.

4. Performance exit gates.
   - Fresh profiler artifacts in `docs/_static`.
   - Runtime/memory panel refreshed from final artifacts.
   - No speedup claim without numerical identity and physics gates.

### Lane 7: W7-X Zonal Long-Window Recurrence/Damping

Goal: close the physical recurrence/damping mismatch without normalization
changes.

1. Keep current closed convention layer fixed.
   - Paper-facing initializer, signed line-average observable, and line-first
     normalization stay unchanged.

2. Study closure/operator physics.
   - Move beyond constant Hermite or mixed Laguerre-Hermite damping.
   - Test velocity-space closure operators that improve trace error,
     late-window envelope, residual, and moment-tail gates together.
   - Preserve high-moment stability and recurrence behavior.

3. Run convergence ladder.
   - Moment resolution: `Nl,Nm`.
   - Time step and total time.
   - Radial wavelength set.
   - Closure source and strength.
   - Compare against digitized stella/GENE W7-X traces.

4. Zonal exit gates.
   - Residuals pass for all tracked `kx rho_i` values.
   - Late-envelope gates pass.
   - Moment tails remain bounded.
   - Publication panel includes trace, residual, moment-tail, and closure-ladder
     diagnostics.

### Lane 8: W7-X Fluctuation/TEM/Multi-Flux-Tube Extension

Goal: turn the current partial diagnostic lane into broad W7-X validation.

1. Keep current fluctuation-spectrum estimator.
   - Preserve the existing simulation-spectrum panel and JSON companion.
   - Add density/zonal-frequency transfer-function work only if experimental
     comparison remains in manuscript scope.

2. Add W7-X multi-alpha and multi-surface ITG scans.
   - Use the stella/GENE benchmark as the reference contract.
   - Track growth, frequency, mode structure, heat flux, spectra, and
     windowed nonlinear statistics.

3. Add kinetic-electron/TEM validation.
   - Fix current TEM branch parity before nonlinear TEM claims.
   - Include density-gradient scans and branch-ordering diagnostics.
   - Add nonlinear kinetic-electron W7-X windows only after linear branch
     gates pass.

4. Extension exit gates.
   - Multi-alpha/multi-surface ITG gates pass.
   - TEM linear parity passes with branch-continuity diagnostics.
   - Kinetic-electron nonlinear windows pass case-specific statistics gates.
   - Broad W7-X stellarator-validation language is allowed only after these
     gates close.

### Immediate Next Implementation Tranche

Start Lane 1 with the smallest production-useful changes:

1. Add `ParallelConfig` and wire it into runtime scan/quasilinear scan without
   changing defaults.
2. Extend `batch_map` tests for pytrees and device-count edge cases.
3. Add a local logical-CPU parallel scan artifact generator.
4. Add `velocity_sharding.py` with only the decomposition planner and tests.
5. Then implement the first `shard_map` Hermite ghost-exchange unit test.

## Repository Trim and Artifact Hygiene Plan

Current audit command:

```bash
python tools/audit_repository_size.py --top 30
```

Current local snapshot from 2026-04-29:

- tracked HEAD payload: about `38.8 MB` across `742` files;
- `.git` history payload: about `154 MB`;
- ignored local artifact roots dominate the checkout size: `tools_out` about `657 MB`, `.venv` about `511 MB`, `.mypy_cache` about `167 MB`, `docs/_build` about `71 MB`, and `dist` about `24 MB`;
- largest tracked payloads are documentation/release assets, especially `docs/_static` about `30.5 MB` and `examples/wout_HSX_QHS_vacuum_ns201.nc` about `3.9 MB`.

Non-destructive trim steps:

1. Keep source code, tests, small input TOMLs, small JSON/CSV gate reports, and figure-generation scripts in Git.
2. Move high-resolution PNG/PDF panels, raw NetCDF outputs, profiler traces, and nonessential VMEC/raw reference data to GitHub Releases or another artifact store with checksums and replay commands.
3. Keep only lightweight README/docs figures in Git, preferably compressed publication previews; link high-resolution PDF/PNG artifacts from releases.
4. Add a repository-size CI gate after the first trim pass: fail if a newly tracked file exceeds an agreed threshold unless it is whitelisted in a manifest.
5. Add a release-artifact manifest mapping each moved artifact to a release URL, checksum, generating command, and validation gate.

History rewrite policy:

- Do not rewrite history during ordinary development.
- If clone size remains too large after non-destructive trimming, coordinate a dedicated maintenance window.
- Before rewriting, tag a backup ref, freeze merges, publish migration instructions, and verify PyPI/release artifacts remain reproducible.
- Use path-specific `git filter-repo` rules rather than broad deletion where possible; then force-push only after collaborators agree to reclone or reset local branches.

Exit gate:

- fresh clone excluding optional release artifacts should stay below the agreed size budget;
- docs and examples must still render without downloading heavy raw outputs;
- all publication figures must be regenerable from scripts plus release-manifest artifacts;
- CI must enforce future artifact-size hygiene automatically.

## Quasilinear Model Design

### Tier 1: Linear Transport Weights

Add `src/spectraxgk/quasilinear.py` with functions for:

- normalized eigenstate extraction from time-path or Krylov linear runs;
- `phi_rms`, `phi_midplane`, and `field_energy` normalization;
- species-resolved amplitude-normalized heat flux weights;
- species-resolved amplitude-normalized particle flux weights;
- electrostatic channel first, with electromagnetic channel placeholders guarded by explicit unsupported errors until validated;
- `kperp_eff^2 = <kperp^2 |phi|^2 J dz> / <|phi|^2 J dz>`;
- flux-weight phase invariance and amplitude-scaling diagnostics;
- JSON-friendly result dataclasses.

Initial formulas should reuse the existing SPECTRAX-GK diagnostic conventions rather than duplicating flux kernels. The diagnostic must reject or clearly mark particle flux for single-kinetic-species adiabatic-electron cases where the model cannot support a meaningful ambipolar particle-flux claim.

### Tier 2: Saturation Rules

Implement rules as explicit named models with serialized metadata:

- `none`: linear weights only;
- `mixing_length`: `A2(ky) = Csat * max(gamma, 0) / kperp_eff^2` or the squared-amplitude convention selected in the code, documented precisely;
- `lapillonne_2011`: eigenfunction-weighted `gamma / kperp_eff^2` amplitude scaling;
- `bourdelle_2007`: diffusivity-style scaling with gradient-length dependence;
- `kumar_2021`: non-divergent broader-spectrum rule from Parker et al.;
- `tglf_like_zonal_mixing`: planned, only after zonal-response ingredients and calibration data exist;
- `calibrated_spectral`: planned, trained on nonlinear spectra and validated on holdout cases;
- `ml_saturation`: post-core feature, inspired by SAT3-NN, only after a transparent non-ML baseline is validated.

Every output must include:

- rule name and version;
- amplitude normalization;
- `Csat` and any fitted parameters;
- training cases if calibrated;
- held-out cases if validated;
- whether output is trend-level or calibrated absolute flux.

### Tier 3: Runtime and Artifact Integration

Add config section:

```toml
[quasilinear]
enabled = false
mode = "weights"                  # weights | saturated | calibrated
saturation_rule = "none"          # none | mixing_length | lapillonne_2011 | bourdelle_2007 | kumar_2021
amplitude_normalization = "phi_rms" # phi_rms | phi_midplane | field_energy
kperp_average = "phi_weighted"
csat = 1.0
gamma_floor = 0.0
include_stable_modes = false
delta_ky = "auto"
species = "all"
channels = ["es"]
write_spectrum = true
```

Add executable flags:

- `--quasilinear`
- `--ql-mode weights|saturated|calibrated`
- `--ql-saturation-rule ...`
- `--ql-csat VALUE`
- `--ql-output PATH`
- `--ql-normalization phi_rms|phi_midplane|field_energy`

Add artifacts:

- `*.quasilinear.summary.json`
- `*.quasilinear_spectrum.csv`
- `*.quasilinear_species.csv`
- `*.quasilinear.png`
- calibration tables under `docs/_static/quasilinear_*`

## Benchmark Matrix

### Axisymmetric Adiabatic-Electron ITG

Cases:

- Cyclone circular linear `ky` scan;
- Cyclone circular nonlinear saturation;
- Cyclone Miller linear and nonlinear long-window runs;
- selected Miller-shaping scan for geometry sensitivity.

Figures:

- growth/frequency/eigenfunction panel;
- `kperp_eff`, `gamma/kperp_eff^2`, and flux-weight spectra;
- nonlinear heat-flux time trace with windowed/Birkhoff mean;
- quasilinear rule comparison against nonlinear spectrum;
- model-error table and heatmap.

Gates:

- linear `gamma`, `omega`, eigenfunction overlap;
- phase invariance and amplitude scaling of weights;
- nonlinear window mean/RMS/std;
- saturation-rule trend ranking across `a/LTi` and `ky`.

### Axisymmetric Kinetic-Electron ITG/TEM/ETG

Cases:

- kinetic-electron Cyclone ITG/TEM branch scan;
- density-gradient TEM scan;
- ETG pilot only if benchmark operating point is literature anchored;
- optional isotope-scaling stress test if runtime is feasible.

Figures:

- growth/frequency branch map with mode classification;
- heat and particle flux spectra by species;
- flux-ratio and cross-phase panel;
- failure-mode panel showing where mixing length misses kinetic-electron trends.

Gates:

- species heat/particle sign and ratio checks;
- finite-difference vs AD derivatives for `gamma`, `omega`, and quasilinear flux;
- branch-continuity and near-marginality handling;
- clear demotion if a case remains outside validated scope.

### Electromagnetic / KBM

Cases:

- KBM linear beta scan;
- KBM branch-continuity scan;
- KBM nonlinear long-window audit.

Figures:

- beta-threshold and branch-continuity panel;
- electromagnetic channel contribution panel after validation;
- nonlinear heat-flux comparison with window statistics.

Gates:

- branch overlap and continuity;
- beta threshold monotonicity where expected;
- nonlinear window-statistics gate;
- electromagnetic quasilinear output remains disabled until field-channel normalization is validated.

### Stellarator Adiabatic-Electron ITG

Cases:

- W7-X multi-alpha and multi-surface ITG;
- HSX multi-alpha and multi-surface ITG;
- W7-X/HSX nonlinear long-window validation;
- W7-X zonal-response closure remains a dedicated paper-level lane.

Figures:

- W7-X and HSX linear spectrum atlas;
- field-line dependence heatmaps over `(rho, alpha)`;
- nonlinear heat-flux windows with uncertainty;
- quasilinear-predicted vs nonlinear-observed reduction scatter;
- zonal-response panel only after long-window recurrence is closed.

Gates:

- geometry parity and field-line reproducibility;
- multi-alpha spread and uncertainty;
- nonlinear window acceptance;
- no broad W7-X claim from a single flux tube.

### Stellarator Kinetic-Electron ITG/TEM

Cases:

- W7-X kinetic-electron ITG with and without density gradient;
- HSX kinetic-electron ITG with and without density gradient;
- W7-X/HSX TEM density-gradient scans.

Figures:

- heat and particle flux panels for adiabatic vs kinetic electrons;
- growth-rate and real-frequency panels at low and broad `ky`;
- maximum-J / stable-mode failure-mode panel;
- quasilinear-vs-nonlinear flux-ratio plot.

Gates:

- particle and heat flux signs;
- low-`ky` growth-rate sensitivity;
- model failure is acceptable if it is explained and bounded;
- saturation-rule calibration must use holdout geometry, not just holdout `ky`.

## Full Quasilinear Model Study

### Training and Holdout Policy

Use three tiers:

1. **No calibration**: compare spectrum shapes and trends only.
2. **One-constant calibration**: fit `Csat` on one axisymmetric adiabatic ITG case; validate on Miller, HSX, W7-X.
3. **Multi-parameter spectral calibration**: train on a small mixed set and hold out complete geometries and physics regimes.

Never calibrate and validate on slices of the same nonlinear run without marking it as interpolation. Hold out at least one geometry and one kinetic-electron regime.

### Error Metrics

Report:

- total heat flux relative error;
- total particle flux relative error where meaningful;
- spectrum-shape cosine similarity;
- peak `ky` location error;
- species flux-ratio error;
- electron/ion heat-flux ratio;
- particle-to-heat flux ratio;
- cross-phase agreement where available;
- ranking accuracy across gradient/geometric scans;
- uncertainty intervals from nonlinear windows and calibration covariance.

### Required Publication Figures

Main-paper figures:

1. **Theory and implementation schematic**: linear eigenstate -> flux weights -> saturation rule -> flux prediction -> nonlinear audit.
2. **Linear benchmark atlas**: `gamma`, `omega`, eigenfunction overlap, `kperp_eff` across representative axisymmetric and stellarator cases.
3. **Flux-weight validation**: heat and particle flux weights by species and `ky`, including phase/amplitude invariance tests.
4. **Saturation-rule comparison**: Parker-style rule comparison with total and spectral fluxes.
5. **Nonlinear validation atlas**: time traces, Birkhoff/window averages, spectra, and uncertainty bands.
6. **Model-error map**: cases vs saturation rules, split into uncalibrated and calibrated performance.
7. **Kinetic-electron failure-mode panel**: TEM/ITG-TEM cases showing where simple mixing length fails.
8. **Stellarator field-line/surface panel**: W7-X/HSX multi-alpha and multi-rho maps.
9. **AD validation panel**: AD vs central finite difference for `gamma`, `f_Q`, quasilinear heat/particle flux, and geometry observables.
10. **UQ panel**: covariance ellipses and propagated uncertainty in quasilinear flux.
11. **Optimization panel**: boundary before/after, Boozer `|B|`, QS residual, quasilinear objective history, nonlinear audit bars.
12. **Performance panel**: cold compile, warm throughput, memory, batched `ky` scan speedup, and nonlinear runtime context.

README figures:

- one concise quasilinear diagnostic example;
- one saturation-rule benchmark summary;
- one differentiable optimization summary;
- one runtime/performance summary.

Docs figures:

- all main-paper figures plus diagnostic details, artifact schema, and failure cases.

## Differentiable Geometry and Stellarator Optimization Plan

### Existing Starting Point

SPECTRAX-GK already has a tracer-safe geometry contract in `spectraxgk.geometry.differentiable`:

- backend discovery for `vmec_jax` and `booz_xform_jax`;
- `flux_tube_geometry_from_mapping(...)`;
- geometry observables;
- finite-difference Jacobian helper;
- inverse-design and covariance report utilities.

`vmec_jax` already has boundary-param optimization APIs, exact/discrete-adjoint machinery, fixed-boundary optimization examples, quasisymmetry objectives, aspect-ratio objectives, and Boozer input generation from VMEC state.

### Integration Stages

1. **In-memory geometry parity**
   - Build `vmec_jax VMECState -> booz_xform_jax -> FluxTubeGeometryData` without writing `wout` or `*.eik.nc`.
   - Compare arrays against the existing imported VMEC/eik path.
   - Gate: max relative errors per geometry coefficient and exact shape/normalization metadata.

2. **Geometry derivative validation**
   - Differentiate geometry observables with respect to boundary modes.
   - Compare AD/JVP/VJP against central finite differences.
   - Gate: absolute/relative derivative tolerances with step-size scan.

3. **Linear observable derivative validation**
   - Differentiate `gamma`, `omega`, `kperp_eff`, flux weights, and `f_Q` with respect to boundary modes.
   - Use smooth branch tracking and smooth positive-growth functions.
   - Gate: AD vs finite differences and branch-continuity diagnostics.

4. **Quasilinear inverse design**
   - Recover target `gamma`, `kperp_eff`, or flux-weight observables using a bounded two- to four-parameter problem.
   - Add covariance estimates and conditioning diagnostics.
   - Gate: inverse recovery, gradient check, covariance finite and interpretable.

5. **Stellarator optimization**
   - Optimize boundary parameters for reduced quasilinear heat flux plus quasisymmetry/aspect/iota regularization.
   - Run multi-alpha and multi-surface objective variants.
   - Audit initial and optimized geometries with nonlinear runs.
   - Gate: nonlinear audit confirms reduction within uncertainty or clearly documents failure.

### Optimization Objectives

Base objective:

```text
J = w_Q * f_Q
  + w_QS * f_QS
  + w_A * (A - A_target)^2
  + w_iota * iota_penalty
  + w_reg * ||boundary - boundary0||^2
```

Use smooth optimization forms:

- `gamma_plus = eps * softplus(gamma / eps)` instead of hard `max(gamma, 0)`;
- softmax-weighted dominant-mode objectives when branches cross;
- multi-`ky` objective as a differentiable weighted sum;
- robust loss or Birkhoff/window objective only for nonlinear audit, not first-pass AD optimization.

### What Goes Beyond Existing Literature

The target contribution should go beyond previous quasilinear or nonlinear stellarator optimization by combining:

- a JAX-native differentiable gyrokinetic quasilinear objective;
- direct AD through geometry-to-linear-transport observables;
- explicit uncertainty and conditioning diagnostics;
- calibrated and uncalibrated saturation-rule comparison;
- multi-surface and multi-field-line stellarator optimization;
- nonlinear audit of optimized geometries;
- documented failure modes for kinetic-electron and nonlinear-stabilization regimes.

## Functionality Checklist

### New Source Functionality

- `spectraxgk.quasilinear` module.
- `RuntimeQuasilinearConfig` in `runtime_config.py`.
- Runtime linear artifact integration.
- Batched `ky` quasilinear scan support.
- Plotting helper for quasilinear spectra and model comparison.
- Calibration utility with explicit train/holdout split.
- Differentiable objective wrappers for `f_Q` and saturated quasilinear flux.
- Geometry-to-linear observable bridge for `vmec_jax` output.
- Optimization example using `vmec_jax` boundary parameters.

### New Examples

- `examples/quasilinear/cyclone_ql_weights.py`
- `examples/quasilinear/cyclone_saturation_rules.py`
- `examples/quasilinear/kinetic_electron_flux_channels.py`
- `examples/quasilinear/w7x_hsx_ql_scan.py`
- `examples/quasilinear/calibrate_saturation_rule.py`
- `examples/optimization/vmec_jax_quasilinear_stellarator.py`
- `examples/optimization/nonlinear_audit_optimized_geometry.py`

### New Documentation

- `docs/quasilinear.rst`: equations, assumptions, rules, outputs, limitations.
- `docs/quasilinear_validation.rst`: benchmark matrix, gates, figures, failure cases.
- `docs/differentiable_stellarator_optimization.rst`: vmec_jax/booz_xform_jax pipeline and examples.
- `docs/artifacts.rst`: JSON/CSV schemas for quasilinear outputs.
- `docs/testing.rst`: quasilinear and AD gate policy.
- README section with install/run/plot plus one quasilinear example and one optimization summary.

## Test and CI Plan

### Fast Unit/Property Tests

- phase invariance of linear weights;
- amplitude quadratic scaling of fluxes;
- stable-mode zeroing for saturated rules when `include_stable_modes=false`;
- `delta_ky` integration and units metadata;
- species-channel sign and shape checks;
- adiabatic-electron particle-flux behavior is explicitly marked;
- serialization round trips for quasilinear result dataclasses;
- CLI/TOML parsing and artifact names.

### Numerical and Physics Tests

- growth/frequency parity for selected linear cases;
- eigenfunction overlap and branch continuity;
- `kperp_eff` analytic limits in simple geometry;
- flux weight consistency with existing nonlinear diagnostic kernels in a linearized state;
- observed convergence with `Nl`, `Nm`, `ntheta`, and `ky` grid;
- nonlinear window statistics for selected audit cases;
- W7-X/HSX multi-alpha sensitivity gates.

### Autodiff Tests

- AD vs finite difference for `kperp_eff`;
- AD vs finite difference for linear `gamma` and `omega` where branch is isolated;
- AD vs finite difference for `f_Q`;
- JVP/VJP consistency on small linear objectives;
- covariance positive-semidefinite checks;
- inverse-design convergence and conditioning diagnostics;
- vmec_jax boundary derivative check through geometry bridge.

### Coverage and CI

- Keep package-wide coverage target at 95%.
- Shard wide coverage so local and CI shards stay bounded.
- Add quasilinear tests to fast PR CI.
- Keep heavy nonlinear validation in manual/office workflows with tracked artifacts.
- Every paper-facing figure must have a script, input manifest, output JSON, and test that validates schema and key metrics.

## Remaining Open Lanes From Previous Plan

These remain active and must be integrated with the quasilinear/optimization work:

1. **Code refactoring**
   - Continue behavior-preserving decomposition of `runtime.py`, `linear.py`, `nonlinear.py`, `benchmarks.py`, plotting, and geometry adapters.
   - Every extraction needs tests and compatibility exports.

2. **Physics gates and validation artifacts**
   - Keep `tools/validation_coverage_manifest.toml` current.
   - Keep gate reports machine-readable and connected to docs figures.
   - Add quasilinear gates to the validation index.

3. **95% package-wide coverage**
   - Continue focused tranches on runtime, linear, nonlinear, benchmarks, geometry, diagnostics, plotting, and new quasilinear code.
   - Coverage must come from physics/numerics/AD tests, not shallow import tests.

4. **Performance and profiling**
   - Keep cold compile, warm runtime, memory, output, and plotting time separated.
   - Profile nonlinear RHS hot paths before making optimization claims.
   - Keep batched `ky` scans and UQ ensembles as the first production parallelization path.

5. **Parallelization**
   - Production path starts with independent `ky` scans, sensitivity sweeps, calibration grids, and UQ ensembles.
   - Nonlinear domain decomposition remains post-release unless a numerical-identity gate is implemented.

6. **Differentiable geometry**
   - Complete `vmec_jax -> booz_xform_jax -> FluxTubeGeometryData` in-memory parity.
   - Add derivative checks before optimizing growth rates or fluxes.

7. **W7-X zonal response**
   - State initializer and observable conventions are closed.
   - Long-window damping, recurrence, moment-resolution, and closure remain paper-level work.
   - Use this lane to support stellarator nonlinear-stabilization claims, not as a release blocker.

8. **W7-X fluctuation spectrum**
   - Add reproducible spectral estimator and publication panel.
   - Tie to nonlinear heat-flux and zonal-response interpretation.

9. **W7-X multi-flux-tube/TEM extension**
   - Required before broad W7-X stellarator-validation claims.
   - Must include kinetic-electron density-gradient/TEM cases.

10. **Nonlinear window statistics tightening**
    - Current release-level gates are useful but broad.
    - Paper-level tolerances must become case-specific where references support them.

11. **Autodiff validation**
    - Keep finite-difference and tangent checks.
    - Strengthen UQ covariance and sensitivity-map workflows for every differentiated observable.

12. **KBM branch continuity and nonlinear audit**
    - Keep selected-branch continuity gates.
    - Treat electromagnetic quasilinear channels as disabled until validated.

13. **Cyclone velocity-space convergence**
    - Maintain the real monotone velocity-space convergence sweep.
    - Extend to quasilinear weights and flux spectra.

14. **Documentation and README**
    - Keep docs user-first: install, run, plot, inspect artifacts.
    - Move long benchmark caveats into validation pages.
    - Main README should show only clean, stable, high-information figures.

15. **Release hygiene**
    - Keep untracked local artifacts such as `.cache_w7x_jpp_2022.pdf` out of release.
    - Keep CI/CD green before tags.
    - PyPI release workflow remains `release.yml`.

## Initial Milestones

### Milestone A: Quasilinear Core Diagnostic

Deliverables:

- source module and dataclasses;
- TOML and executable flags;
- linear runtime artifact writer;
- Cyclone linear weight example;
- unit/property/AD tests;
- docs page with equations and limitations.

Exit gate:

- fast tests pass;
- phase/amplitude invariance passes;
- `gamma`, `omega`, `kperp_eff`, and weights are serialized and plotted.

### Milestone B: Saturation Rule Study

Deliverables:

- named saturation rules;
- calibration utility;
- Cyclone and Miller nonlinear comparisons;
- W7-X/HSX exploratory scans;
- model-error panel.

Exit gate:

- uncalibrated rules clearly marked as trend-only;
- one-constant calibration has held-out validation;
- figure scripts write JSON/CSV/PNG/PDF.

### Milestone C: Kinetic-Electron and Failure-Mode Study

Deliverables:

- kinetic-electron heat/particle flux channels;
- TEM/ITG-TEM branch scans;
- W7-X/HSX density-gradient comparison;
- cross-phase/flux-ratio diagnostics.

Exit gate:

- failures of simple saturation models are documented;
- no unsupported absolute particle-flux claim remains.

### Milestone D: Differentiable Geometry Integration

Deliverables:

- in-memory `vmec_jax` bridge;
- optional `booz_xform_jax` bridge;
- geometry parity artifact;
- AD/FD geometry derivative artifact.

Exit gate:

- geometry arrays match existing path;
- derivative checks pass with step-size scan.

### Milestone E: Quasilinear Stellarator Optimization

Deliverables:

- differentiable quasilinear objective;
- optimization example with QS/aspect regularization;
- UQ/covariance panel;
- nonlinear audit of initial/final geometry.

Exit gate:

- optimization reduces quasilinear objective;
- derivative checks pass;
- nonlinear audit confirms or bounds the reduced-model prediction.

### Milestone F: Paper and Release Packaging

Deliverables:

- docs figure stack;
- README summary figures;
- validation matrix;
- performance panel;
- CI and coverage gates;
- release notes.

Exit gate:

- 95% package-wide coverage confirmed;
- docs build;
- package build;
- fast and selected artifact tests pass;
- all claims in README/docs map to artifacts.

## Running Log

### 2026-04-29

- Archived the historical 2,755-line root `plan.md` into private repo `rogeriojorge/spectraxgk_plan`.
- Replaced root `plan.md` with this active quasilinear transport and differentiable stellarator optimization plan.
- Final literature pass added missing requirements:
  - flux-ratio and cross-phase diagnostics for reviewer-proof quasilinear validation;
  - kinetic-electron W7-X/HSX cases where maximum-J physics and nonlinear stable-mode transfer can invalidate simple mixing-length expectations;
  - calibration/holdout split for any absolute flux claim;
  - nonlinear audit of optimized stellarators before broad optimization claims;
  - SAT3/SAT3-NN style spectrum-aware saturation as future calibrated work, not the first implementation step.
- Removed the stale untracked `.cache_w7x_jpp_2022.pdf` from the active repo checkout.
- Started Milestone A implementation:
  - added `src/spectraxgk/quasilinear.py` with electrostatic linear heat/particle flux weights, `kperp_eff2`, amplitude normalizations, and explicit saturation-rule metadata;
  - added `[quasilinear]` to the runtime TOML schema and executable flags (`--quasilinear`, `--ql-mode`, `--ql-saturation-rule`, `--ql-csat`, `--ql-normalization`, `--ql-output`);
  - wired Krylov/time linear states into quasilinear artifacts without forcing state artifacts unless requested;
  - added `*.quasilinear.summary.json` and `*.quasilinear_species.csv` writers;
  - added `examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml`, README usage, and docs page `docs/quasilinear.rst`.
- Fast validation run:
  - `pytest -q tests/test_quasilinear.py tests/test_runtime_config.py::test_runtime_config_to_dict_contains_sections tests/test_runtime_config.py::test_load_runtime_from_toml_roundtrip tests/test_runtime_artifacts.py::test_write_runtime_linear_artifacts_writes_bundle tests/test_cli.py::test_cmd_run_runtime_linear_applies_quasilinear_flags` passed.
  - Broader adjacent shard with runtime/CLI/artifact tests passed (`31` tests).
  - Example smoke passed: `spectraxgk run-runtime-linear --config examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml --out tools_out/ql_smoke --no-progress`.
- Next best steps:
  - add quasilinear scan aggregation so `scan-runtime-linear` can produce `*.quasilinear_spectrum.csv`;
  - add nonlinear calibration table scaffolding and holdout metadata before any absolute flux claims;
  - add finite-difference vs autodiff checks for `kperp_eff2`, linear weights, and mixing-length objective;
  - then start nonlinear Cyclone/Cyclone Miller calibration comparisons for the first publication panel.
- Completed the first scan-level quasilinear spectrum slice:
  - serial `run_runtime_scan` now collects per-ky quasilinear payloads;
  - `scan-runtime-linear --quasilinear --out ...` writes `*.scan.csv` and `*.quasilinear_spectrum.csv`;
  - batched quasilinear scans are rejected until per-ky state extraction has a numerical-identity gate.
- Validation for scan slice:
  - `pytest -q tests/test_quasilinear.py tests/test_runtime_artifacts.py::test_write_runtime_linear_artifacts_writes_bundle tests/test_runtime_artifacts.py::test_write_runtime_linear_scan_artifacts_with_quasilinear_spectrum tests/test_cli.py::test_cmd_run_runtime_linear_applies_quasilinear_flags tests/test_cli.py::test_cmd_scan_runtime_linear_writes_quasilinear_spectrum tests/test_cli.py::test_cmd_scan_runtime_linear_branches tests/test_runtime_config.py::test_load_runtime_from_toml_roundtrip` passed.
  - Example scan smoke passed: `spectraxgk scan-runtime-linear --config examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml --ky-values 0.2,0.3 --quasilinear --out tools_out/ql_scan_smoke --no-progress`.
  - `sphinx-build -b html docs docs/_build/html -q` passed.
- Updated next best steps:
  - add differentiable quasilinear objective helpers and AD-vs-finite-difference validation;
  - add calibration/holdout artifact schema and nonlinear comparison script skeleton;
  - produce the first publication-ready quasilinear spectrum figure from the Cyclone scan.
- Completed first differentiability gate slice:
  - added JAX-native `mixing_length_amplitude2_jax`, `saturated_flux_from_linear_weight`, and `quasilinear_feature_objective`;
  - added generic `central_finite_difference_jacobian` and `autodiff_finite_difference_report` validation helpers;
  - added derivative tests for a closed-form vector function and the quasilinear reduced objective, including directional tangent diagnostics.
- Validation for differentiability slice:
  - `pytest -q tests/test_autodiff_validation.py tests/test_quasilinear.py` passed.
  - `ruff check src/spectraxgk/quasilinear.py src/spectraxgk/autodiff_validation.py tests/test_quasilinear.py tests/test_autodiff_validation.py` passed.
- Next best steps:
  - add nonlinear calibration/holdout artifact schema and scripts;
  - add the first quasilinear spectrum figure generator for Cyclone;
  - then wire finite-difference derivative gates to full linear-run outputs, not only reduced features.
- Completed calibration/holdout artifact schema slice:
  - added `QuasilinearCalibrationPoint`, `quasilinear_calibration_report`, and `write_quasilinear_calibration_report`;
  - added `tools/build_quasilinear_calibration_report.py` for JSON point lists;
  - reports are promoted to `calibrated_absolute_flux` only when train and holdout points exist and the holdout mean-relative gate passes.
- Validation for calibration slice:
  - `pytest -q tests/test_quasilinear_calibration.py tests/test_autodiff_validation.py tests/test_quasilinear.py` passed.
  - `ruff check src/spectraxgk/quasilinear.py src/spectraxgk/quasilinear_calibration.py src/spectraxgk/autodiff_validation.py tests/test_quasilinear.py tests/test_quasilinear_calibration.py tests/test_autodiff_validation.py tools/build_quasilinear_calibration_report.py` passed.

- Tool smoke with temporary train/holdout points produced a valid JSON report.
- Next best steps:
  - implement the first publication-ready quasilinear spectrum plotting script;
  - generate a Cyclone spectrum figure from `runtime_cyclone_quasilinear.toml`;
  - then add nonlinear comparison ingestion that maps existing nonlinear window summaries into calibration points.
- Completed first quasilinear spectrum figure slice:
  - added `tools/plot_quasilinear_spectrum.py`;
  - generated tracked artifacts under `docs/_static/quasilinear_cyclone_spectrum*`;
  - added the figure to README, docs, and the manuscript figure index with the nonlinear-calibration caveat.
- Validation for figure slice:
  - `pytest -q tests/test_plot_quasilinear_spectrum.py tests/test_quasilinear_calibration.py tests/test_autodiff_validation.py tests/test_quasilinear.py` passed.
  - `ruff check src/spectraxgk/quasilinear.py src/spectraxgk/quasilinear_calibration.py src/spectraxgk/autodiff_validation.py tests/test_quasilinear.py tests/test_quasilinear_calibration.py tests/test_autodiff_validation.py tests/test_plot_quasilinear_spectrum.py tools/build_quasilinear_calibration_report.py tools/plot_quasilinear_spectrum.py` passed.
  - `sphinx-build -b html docs docs/_build/html -q` passed.
- Next best steps:
  - add nonlinear-window-summary ingestion into calibration points;
  - add a light calibration figure once actual nonlinear holdout data are mapped;
  - add full-output derivative gates that differentiate through a small linear solve, then connect to `vmec_jax` geometry parameters.
- Completed nonlinear-window ingestion slice:
  - added `calibration_point_from_nonlinear_window_summary`;
  - the helper reads tracked nonlinear window JSON, applies the summary `tmin`/`tmax`, and records mean/std heat flux from diagnostics CSVs;
  - NetCDF-only ingestion remains explicitly unsupported until it has the same observable/window contract.
- Validation for ingestion slice:
  - `pytest -q tests/test_quasilinear_calibration.py tests/test_plot_quasilinear_spectrum.py tests/test_autodiff_validation.py tests/test_quasilinear.py` passed.
  - targeted `ruff check` passed for the new quasilinear/autodiff/tool files.
- Next best steps:
  - add full-output derivative gates through a tiny differentiable linear solve;
  - start `vmec_jax` bridge planning/implementation from in-memory geometry arrays;
  - add a light nonlinear-calibration figure once calibration points are produced from real nonlinear windows.
- Completed first nonlinear-calibration audit slice:
  - added quasilinear spectrum integration into `spectraxgk.quasilinear_calibration`;
  - extended `tools/build_quasilinear_calibration_report.py` so it can generate an audit point directly from a quasilinear spectrum CSV and nonlinear window-summary JSON;
  - added `tools/plot_quasilinear_calibration.py`;
  - generated the tracked Cyclone audit artifacts `docs/_static/quasilinear_cyclone_calibration_audit*` from the current quasilinear spectrum and long-window nonlinear heat-flux summary;
  - kept the claim level at `training_or_audit_only` because the current artifact has no train/holdout split and uses uncalibrated `C_sat = 1`.
- Validation for calibration-audit slice:
  - `pytest -q tests/test_quasilinear_calibration.py tests/test_plot_quasilinear_calibration.py tests/test_plot_quasilinear_spectrum.py tests/test_quasilinear.py` passed.
  - `ruff check src/spectraxgk/quasilinear_calibration.py tests/test_quasilinear_calibration.py tests/test_plot_quasilinear_calibration.py tools/build_quasilinear_calibration_report.py tools/plot_quasilinear_calibration.py` passed.
  - `sphinx-build -b html docs docs/_build/html -q` passed.
- Next best steps:
  - add full-output derivative gates through a tiny differentiable linear/eigen solve;
  - build the first train/holdout quasilinear calibration dataset across Cyclone/Cyclone Miller/KBM/HSX/W7-X once matching linear spectra exist;
  - start the `vmec_jax` in-memory geometry bridge so quasilinear objectives can be differentiated with respect to boundary parameters.
- Completed branch-isolated eigenvalue derivative gate slice:
  - added `isolated_eigenvalue_sensitivity_report` to `spectraxgk.autodiff_validation`;
  - the gate selects an eigenvalue branch at the base point, checks the eigenvalue gap, and compares AD Jacobians against central finite differences for `[real(lambda), imag(lambda)]`;
  - this is the lightweight precursor to differentiating full linear growth/frequency and quasilinear objectives through geometry.
- Validation for eigenvalue derivative slice:
  - `pytest -q tests/test_autodiff_validation.py tests/test_quasilinear.py tests/test_quasilinear_calibration.py` passed.
  - `ruff check src/spectraxgk/autodiff_validation.py tests/test_autodiff_validation.py src/spectraxgk/__init__.py` passed.
  - `sphinx-build -b html docs docs/_build/html -q` passed.
- Next best steps:
  - connect the eigenvalue derivative gate to a tiny actual SPECTRAX-GK linear operator fixture;
  - create matching quasilinear spectra for Cyclone Miller/KBM/HSX/W7-X so calibration can move from audit-only to train/holdout;
  - start `vmec_jax`/`booz_xform_jax` bridge interfaces for differentiable geometry inputs.
- Completed the first shaped-tokamak quasilinear transfer slice:
  - added `examples/linear/axisymmetric/runtime_cyclone_miller_quasilinear.toml`;
  - fixed scan quasilinear spectra so `ky` is the requested scan coordinate and `mode_ky` retains the selected signed grid-mode coordinate, which prevents linked-boundary aliases from corrupting publication x-axes;
  - regenerated `docs/_static/quasilinear_cyclone_miller_spectrum*` with the nonlinear Miller `Ny=64` positive-ky grid;
  - added train-fitted multiplicative heat-flux scaling utilities to `spectraxgk.quasilinear_calibration`;
  - generated `docs/_static/quasilinear_cyclone_miller_calibration_audit*` and the first Cyclone-to-Cyclone-Miller train/holdout artifact `docs/_static/quasilinear_cyclone_miller_train_holdout*`.
- Current calibration outcome:
  - fitting one constant on Cyclone gives `C_sat = 3839.966` in the current normalization;
  - the held-out Cyclone Miller mean-relative heat-flux error is `612.9`, so the report correctly stays at `calibration_dataset` with `passed = false`;
  - this is a useful research result, not a release failure: it shows that absolute quasilinear flux prediction needs a richer saturation/normalization model before optimization claims.
- Next best steps:
  - add a second held-out axisymmetric/non-axisymmetric electrostatic case with a validated nonlinear CSV window before attempting any multi-parameter saturation model;
  - connect quasilinear objective derivatives to a tiny SPECTRAX-GK linear-operator fixture, then to the existing differentiable geometry bridge;
  - add spectrum-shape gates that compare normalized quasilinear spectra against nonlinear spectral heat-flux distributions where those diagnostics are available.
- Completed the first actual-linear-operator derivative gate:
  - added `explicit_complex_operator_matrix` for tiny validation fixtures that materialize matrix-free operators without changing production solvers;
  - added a SPECTRAX-GK linear-RHS eigenvalue sensitivity test that differentiates through a small dense operator with `use_custom_vjp=false` and compares AD against finite differences;
  - this closes the gap between toy eigenvalue derivative gates and the real linear RHS path, while keeping production field-solve custom VJPs untouched.
- Validation:
  - `pytest -q tests/test_autodiff_validation.py` passed.
  - `ruff check src/spectraxgk/autodiff_validation.py tests/test_autodiff_validation.py src/spectraxgk/__init__.py` passed.
- Next best steps:
  - wire the dense linear-RHS derivative gate to a reduced quasilinear objective from the resulting eigenvector/state, not just the eigenvalue;
  - connect the derivative gate to the existing differentiable geometry bridge so geometry parameters perturb linear weights and quasilinear objectives;
  - create a second train/holdout calibration artifact with a non-axisymmetric electrostatic case once a CSV-backed nonlinear heat-flux window is available.
- Added the first branch-objective differentiability guard:
  - added `isolated_eigenpair_observable_sensitivity_report` for isolated-branch observables;
  - the actual SPECTRAX-GK linear-RHS phase-invariant objective currently reports `ad_supported=false`, because JAX does not support forward-mode derivatives of non-Hermitian eigenvectors;
  - this prevents false end-to-end differentiability claims for eigenfunction-dependent quasilinear objectives.
- Updated next best steps:
  - implement an adjoint/implicit eigenvector-sensitivity path for isolated non-Hermitian branches, then re-open the quasilinear objective AD gate;
  - in parallel, continue calibration/validation on observable values that do not rely on differentiating eigenvectors;
  - keep eigenvalue-only growth/frequency sensitivities as the currently validated differentiable branch.
- Completed implicit non-Hermitian eigenpair sensitivity slice:
  - added `implicit_eigenpair_observable_sensitivity_report`;
  - the helper differentiates matrix entries with JAX, solves the left/right eigenvector perturbation system with `w^H v = 1` and `w^H dv = 0`, and compares phase-invariant observables against nearest-branch central finite differences;
  - added a closed-form non-Hermitian branch test and a tiny SPECTRAX-GK linear-RHS quasilinear-style objective test.
- Expanded quasilinear documentation:
  - added equations for the linear eigenproblem, electrostatic heat/particle weights, amplitude normalization, effective `k_perp`, mixing-length saturation, calibration scoring, and implicit eigenpair sensitivities;
  - added source-code implementation map linking the quasilinear, diagnostic, runtime, calibration, plotting, and autodiff modules;
  - converted quasilinear references to clickable links in `docs/references.rst`, including QuaLiKiz, Parker saturation-rule comparisons, SAT3/SAT3-NN, Waltz/Casati/Staebler, and microstability optimization references.
- Completed user-facing implicit quasilinear sensitivity example:
  - added `examples/theory_and_demos/quasilinear_implicit_sensitivity.py`;
  - the example differentiates `[gamma, omega, kperp_eff^2, Qhat_i, Q_i^ML]` with respect to `[R/Ln, R/LTi]` on a tiny Cyclone linear-RHS fixture;
  - fixed the implicit eigenpair finite-difference validation protocol to follow the nearest isolated eigenvalue branch instead of raw eigensolver index order;
  - generated `docs/_static/quasilinear_implicit_sensitivity.{png,pdf,json}` and documented the figure in `docs/quasilinear.rst` and `docs/manuscript_figures.rst`.
- Completed second electrostatic calibration/holdout machinery gate:
  - fixed `gx_volume_factors` for closed-interval sampled/imported geometries by reusing the existing terminal-theta trim contract;
  - added `examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml`;
  - generated `docs/_static/quasilinear_hsx_spectrum.*` and `docs/_static/quasilinear_hsx_spectrum_scan.*` from a six-point HSX adiabatic-electron linear spectrum;
  - generated `docs/_static/quasilinear_hsx_train_holdout_report.json` and `docs/_static/quasilinear_hsx_train_holdout.*` by adding HSX as a non-axisymmetric held-out nonlinear heat-flux window;
  - the HSX point is a negative result: the current short HSX linear spectrum is stable under the uncalibrated `gamma_floor=0` mixing-length rule, giving zero predicted saturated heat flux against a finite nonlinear window.
- Completed first spectrum-shape gate:
  - added `tools/plot_quasilinear_spectrum_shape_gate.py`;
  - added tests with a synthetic resolved nonlinear NetCDF fixture;
  - generated `docs/_static/quasilinear_hsx_spectrum_shape_gate.{png,pdf,json}`;
  - the HSX gate compares normalized `heat_flux_weight_total` to nonlinear `Diagnostics/HeatFlux_kyst` and passes with total-variation distance about `0.11` and cosine similarity about `0.97`.
- Current next best steps:
  - add a NetCDF nonlinear-window ingestion path so W7-X can enter the same quasilinear calibration machinery without manual CSV conversion.
  - extend the spectrum-shape gate to Cyclone/Cyclone Miller/KBM where resolved `HeatFlux_kyst` artifacts are available and document case-specific gates.
- Completed NetCDF nonlinear-window ingestion:
  - `calibration_point_from_nonlinear_window_summary` now supports diagnostics CSV and runtime NetCDF summaries;
  - NetCDF summaries read `Grids/time` and map `heat_flux` to `Diagnostics/HeatFlux_st`, with species summed by default and optional `species_index` for single-species nonlinear targets;
  - `tools/build_quasilinear_calibration_report.py` now exposes `--species-index`;
  - added a synthetic NetCDF calibration test so W7-X-style `.out.nc` summaries are covered by the fast suite.
- Validation:
  - `ruff check src/spectraxgk/quasilinear_calibration.py tests/test_quasilinear_calibration.py tools/build_quasilinear_calibration_report.py` passed.
  - `pytest -q tests/test_quasilinear_calibration.py` passed.
  - `sphinx-build -b html docs docs/_build/html -W -q` passed.
- Completed additional electrostatic spectrum-shape gates:
  - generated `docs/_static/quasilinear_cyclone_miller_spectrum_shape_gate.{png,pdf,json}`;
  - generated `docs/_static/quasilinear_cyclone_spectrum_shape_gate.{png,pdf,json}`;
  - Cyclone Miller passes the initial shape gate with `TV=0.09395` and cosine `0.98254`;
  - Cyclone is intentionally retained as a failed model/window gate with `TV=0.21473` and cosine `0.89643`, just outside the initial thresholds;
  - KBM remains deferred from quasilinear shape-gate claims because the current quasilinear diagnostic validates electrostatic channels only and the KBM lane is electromagnetic.
- Current next best steps:
  - generate a reproducible W7-X electrostatic quasilinear spectrum only after the W7-X imported geometry source is either tracked or regenerated from a tracked recipe;
  - use the new NetCDF nonlinear-window path to add W7-X to the calibration report once that spectrum exists;
  - start the next saturation-model sweep with shape-aware calibration features, because one-constant mixing length fails absolute-flux transfer and Cyclone shape matching.
- Added the reproducible W7-X quasilinear example entry point:
  - added `examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml`;
  - the example uses `geometry.model = "vmec"` and `vmec_file = "$W7X_VMEC_FILE"` so the VMEC-derived geometry cache is regenerated from a declared source instead of relying on ignored local `tools_out` geometry;
  - added a fast config contract test covering HSX and W7-X non-axisymmetric electrostatic quasilinear examples.
- Current next best steps:
  - run the W7-X VMEC-backed quasilinear scan on a machine with `W7X_VMEC_FILE` available and commit the resulting spectrum/report only if the input provenance is replayable;
  - add the W7-X holdout calibration artifact through the new NetCDF nonlinear window path;
  - then implement the first shape-aware saturation-model sweep: compare linear weights, nonlinear `HeatFlux_kyst`, and one- or two-parameter intensity rules across Cyclone, Cyclone Miller, HSX, and W7-X.
- Completed the first W7-X quasilinear NetCDF calibration artifact:
  - ran the six-point W7-X VMEC-backed quasilinear scan on `ssh office` from a fresh `main` clone with `W7X_VMEC_FILE=/home/rjorge/gx_refs/main_clean_20260312/nonlinear/w7x/wout_w7x.nc`;
  - tracked `docs/_static/quasilinear_w7x_spectrum*` and `docs/_static/quasilinear_w7x_spectrum_scan*`;
  - generated `docs/_static/quasilinear_w7x_train_holdout_report.json` and `docs/_static/quasilinear_w7x_train_holdout.{png,pdf,json}` using the NetCDF nonlinear-window ingestion path;
  - generated `docs/_static/quasilinear_w7x_spectrum_shape_gate.{png,pdf,json}`;
  - W7-X absolute flux remains a negative holdout: all six short-window linear branches are stable under `gamma_floor=0`, so the uncalibrated saturated heat-flux prediction is zero against a finite nonlinear window mean of about `5.38`;
  - W7-X normalized spectrum shape passes with `TV=0.05564` and cosine `0.99167`.
- Added combined stellarator holdout artifact:
  - generated `docs/_static/quasilinear_stellarator_train_holdout_points.json`;
  - generated `docs/_static/quasilinear_stellarator_train_holdout_report.json`;
  - generated `docs/_static/quasilinear_stellarator_train_holdout.{png,pdf,json}`;
  - the combined panel uses Cyclone as the single training point and holds out Cyclone Miller, HSX, and W7-X, preserving the failed `calibration_dataset` result in one manuscript-facing figure.
- Improved calibration plotting:
  - the relative-error panel now switches to a log axis when point errors span orders of magnitude, so small failed holdouts remain visible beside very large failed holdouts.
- Current next best steps:
  - implement a shape-aware saturation-model sweep that uses linear weights plus nonlinear `HeatFlux_kyst` intensity information instead of only one constant `C_sat`;
  - add finite-difference/implicit sensitivity checks for the shape-aware quasilinear objective before using it in optimization;
  - keep KBM out of quasilinear claims until electromagnetic quasilinear channels have separate validation gates.
- Completed first saturation-rule sweep diagnostic:
  - added `tools/plot_quasilinear_saturation_rule_sweep.py`;
  - added `tests/test_plot_quasilinear_saturation_rule_sweep.py`;
  - generated `docs/_static/quasilinear_saturation_rule_sweep.{png,pdf,json}`;
  - the sweep fits one scalar on Cyclone and scores Cyclone Miller, HSX, and W7-X for positive-growth mixing length, raw linear heat-flux weight, and an absolute-growth diagnostic;
  - all three rules fail the held-out absolute-flux gate: holdout mean relative errors are about `205`, `25`, and `510`, respectively;
  - the result is a negative model-development diagnostic, not a validated transport model.
- Current next best steps:
  - formulate the next saturation model using actual shape-aware information instead of one global scalar, for example a low-dimensional intensity envelope constrained by nonlinear `HeatFlux_kyst` and tested by leave-one-geometry-out validation;
  - add differentiability checks for any new quasilinear objective before connecting it to `vmec_jax`/`booz_xform_jax`;
  - keep expanding tests around these tools so each publication artifact has a fast synthetic replay gate.
- Promoted sweep diagnostic rules into the differentiable quasilinear objective path:
  - `saturation_amplitude2` now supports `linear_weight` and `absolute_growth_mixing_length` / `abs_growth_mixing_length`;
  - `quasilinear_feature_objective` now supports those rules for feature vectors `[gamma, kperp_eff2, flux_weight]`;
  - added finite-difference derivative gates for the new differentiable rules.
- Current next best steps:
  - connect the saturation-rule sweep tool to the core rule names instead of duplicating formulas;
  - then move to a true low-dimensional shape-aware model with leave-one-geometry-out validation and AD checks.
- Completed the first low-dimensional shape-aware saturation diagnostic:
  - added `shape_aware_power_law_objective` to `spectraxgk.quasilinear` for a differentiable shape-envelope objective using feature vectors `[gamma, kperp_eff2, flux_weight]` and a geometric `ky` reference;
  - added finite-difference derivative coverage for the objective so future geometry/optimization work does not rely on unchecked autodiff plumbing;
  - added `tools/plot_quasilinear_shape_aware_saturation.py` and fast synthetic/replay tests;
  - generated `docs/_static/quasilinear_shape_aware_saturation.{png,pdf,json}` with the `--passed-shape-only` exponent fit so the failed Cyclone shape gate is not used to define the training shape correction.
- Result:
  - the one-exponent nonlinear/quasilinear shape envelope is not sufficient for absolute heat-flux transfer;
  - mean leave-one-geometry-out relative error is about `0.664`, compared with about `0.624` for the linear-weight baseline and about `0.170` for a deliberately simple training-mean null baseline, so the quasilinear variants fail the `0.35` transport gate and do not beat the null baseline;
  - the result is a research-grade negative diagnostic, not a validated quasilinear transport claim.
- Current next best steps:
  - replace the one-exponent shape envelope with a richer but still low-dimensional model that can include branch/state features, stellarator-vs-axisymmetric family features, and uncertainty diagnostics;
  - require any next saturation model to beat both the linear-weight baseline and the training-mean null baseline in leave-one-geometry-out scoring before exposing it in user-facing TOML;
  - connect the accepted objective to finite-difference/implicit AD checks and then to the `vmec_jax` / `booz_xform_jax` differentiable-geometry bridge.
- Tightened the saturation-rule sweep and started repository-trim hygiene:
  - added a training-mean null baseline to `tools/plot_quasilinear_saturation_rule_sweep.py`;
  - regenerated `docs/_static/quasilinear_saturation_rule_sweep.{png,pdf,json}`;
  - the Cyclone-trained null has holdout mean relative error about `0.372`, far better than the tested one-scalar quasilinear rules, so future saturation candidates must beat it before promotion;
  - added `tools/audit_repository_size.py` and a fast test so tracked-file and local-artifact size audits are reproducible;
  - added the repository trim/history-rewrite policy above.
- Current next best steps:
  - design the next saturation candidate around physically meaningful branch/state features and uncertainty intervals, then require improvement over both the linear-weight baseline and the null baseline;
  - perform the first non-destructive repo trim by moving nonessential high-resolution docs/static artifacts to a release manifest while preserving lightweight docs previews;
  - add a CI size guard after the manifest exists.
- Added executable promotion and repository-size gates:
  - `tools/plot_quasilinear_saturation_rule_sweep.py` now writes a `promotion_gate` with `accepted_rules=[]` for the current one-scalar sweep;
  - `tools/plot_quasilinear_shape_aware_saturation.py` now writes a `promotion_gate` requiring the shape-aware model to beat the linear-weight and training-mean null baselines;
  - regenerated both quasilinear saturation JSON/PNG/PDF artifacts with those machine-readable rejection gates;
  - added `tools/repository_size_manifest.toml` and `tools/check_repository_size_manifest.py`;
  - wired the repository-size manifest checker into CI as `repo-hygiene` and added the checker tests to the fast matrix.
- Current next best steps:
  - implement the release-artifact manifest for moving high-resolution documentation panels out of Git while keeping lightweight previews;
  - prototype the next saturation candidate only as a rejected-or-promoted report, with uncertainty intervals and leave-one-geometry-out scoring against both baselines;
  - keep unvalidated quasilinear candidates out of TOML/runtime user-facing options until `promotion_gate.passed` is true.
- Added the release-artifact and uncertainty-candidate gates:
  - added `tools/release_artifact_manifest.toml` and `tools/check_release_artifact_manifest.py`;
  - the release manifest currently tracks 10 large assets, with about `12.6 MB` planned for GitHub-release migration and the `3.9 MB` HSX VMEC fixture explicitly kept in-repo until a smaller fixture exists;
  - wired the release-artifact checker into CI `repo-hygiene` and the fast test matrix;
  - added `tools/plot_quasilinear_candidate_uncertainty.py`;
  - generated `docs/_static/quasilinear_candidate_uncertainty.{png,pdf,json}`;
  - candidate uncertainty gate result: calibrated linear-weight mean error is about `0.624` with `0.75` interval coverage; shape-power-law mean error is about `0.664` with `0.75` interval coverage; training-mean null remains about `0.170`; `promotion_gate.passed=false` and no candidates are accepted.
- Current next best steps:
  - create compressed lightweight previews for the release-manifest assets, then move high-resolution companions to a GitHub release in a dedicated artifact migration commit;
  - expand the quasilinear candidate family only after adding more held-out nonlinear cases or additional physics features, because the current four-case set is too small for a trustworthy calibrated saturation claim;
  - keep the next candidate report focused on branch/state features and uncertainty intervals rather than adding runtime/TOML exposure.
- Completed first non-destructive documentation-preview trim:
  - added `tools/compress_release_previews.py` plus a fast compression test;
  - extended `tools/check_release_artifact_manifest.py` with `keep_preview_in_repo` so lightweight docs previews can be tracked separately from release-only high-resolution artifacts;
  - compressed six checked-in PNG documentation previews from `10,486,731` bytes to `1,784,130` bytes while keeping the rendered README/docs figures available in Git;
  - removed those compressed PNGs from the repository-size whitelist because they are now below the `1 MB` per-file guard;
  - documented the reproducible preview-compression command in `docs/code_structure.rst`.
- Current next best steps:
  - move release-only PDF/high-resolution companions to GitHub release assets once stable asset URLs exist, then update `tools/release_artifact_manifest.toml`;
  - avoid any git-history rewrite until after non-destructive trimming is complete and collaborators explicitly agree to a coordinated reclone/reset window;
  - continue the quasilinear model-development lane only with rejected-or-promoted reports that include uncertainty intervals, leave-one-geometry-out scoring, and a null-baseline comparison.
- Added a branch/state quasilinear candidate without promoting it:
  - extended `tools/plot_quasilinear_candidate_uncertainty.py` with a `linear_state_ridge` candidate based only on linear-spectrum features: `log_linear_weight`, `log_abs_growth_mixing_length`, `unstable_weight_fraction`, and `log_weighted_ky_centroid`;
  - added candidate-specific eligibility diagnostics for training cases per fitted parameter and ridge normal-matrix conditioning;
  - regenerated `docs/_static/quasilinear_candidate_uncertainty.{png,pdf,json}`;
  - the linear-state ridge candidate has mean leave-one-geometry-out relative error about `0.173` and interval coverage `1.0`, but `promotion_eligible=false` because the current four-case dataset is under-sampled for the five-parameter ridge fit;
  - `promotion_gate.passed=false` remains unchanged, preventing user-facing runtime/TOML exposure.
- Current next best steps:
  - add at least two additional nonlinear holdout cases before attempting to promote any branch/state quasilinear saturation candidate;
  - keep the next quasilinear model-development figure focused on whether new cases reduce uncertainty and beat the training-mean null without overfitting;
  - perform release-only PDF migration only after GitHub Release asset URLs are available.
- Added external VMEC equilibrium portfolio inventory for the next holdout campaign:
  - added `tools/plot_vmec_jax_equilibrium_inventory.py` and a synthetic NetCDF replay test;
  - generated `docs/_static/vmec_jax_equilibrium_inventory.{png,pdf,json}` from `/Users/rogeriojorge/local/vmec_jax/examples/data`;
  - the inventory finds `10` local VMEC equilibria and recommends Li383, nfp4 QH, CTH-like, shaped tokamak, circular tokamak, DSHAPE, and purely toroidal cases as the next linear/nonlinear holdout candidates;
  - the low-resolution QA fixture is now explicitly deferred because its VMEC reference-scale metadata are degenerate and trigger the current VMEC-EIK path's reference-length division guard;
  - the JSON explicitly marks this as `equilibrium_selection_not_transport_validation`, so none of these cases enter quasilinear calibration until matched nonlinear heat-flux windows and physics gates exist.
- Completed bounded external-VMEC linear smoke checks:
  - used `examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml` with `W7X_VMEC_FILE` pointed at `vmec_jax/examples/data`, `ky=0.10,0.20`, `Nl=2`, `Nm=4`, `dt=0.005`, and `80` steps;
  - finite stable branches were obtained for `wout_li383_low_res.nc` (`gamma=-0.0258,-0.0297`), `wout_nfp4_QH_warm_start.nc` (`gamma=-0.0243,-0.0186`), `wout_cth_like_fixed_bdy.nc` (`gamma=-0.0230,-0.0282`), and `wout_shaped_tokamak_pressure.nc` (`gamma=-0.0669,-0.0562`);
  - `wout_basic_non_stellsym_simsopt.nc` is not yet a holdout candidate because the current centered flux-tube cut finds no crossing for that fixture;
  - all successful smoke checks are stable under the current short setup, so their positive-growth mixing-length saturated flux is zero by construction and they are not quasilinear transport validation points.
- Current next best steps:
  - promote Li383 and nfp4 QH to production-resolution linear scans only if branch selection remains finite and interpretable over the full `ky` grid;
  - then generate matched nonlinear windows for the same cases and add them to the leave-one-out calibration reports only if nonlinear comparison/physics gates pass;
  - separately harden the VMEC-EIK geometry path with explicit user-facing errors for degenerate reference scales and no-crossing flux-tube cuts before exposing arbitrary external VMEC files as a polished workflow.
- Completed first full-`ky` external-VMEC linear feasibility scans:
  - Li383 full-`ky` scan stays stable over the six-point stellarator grid (`gamma=-0.0222,-0.0429,-0.0573,-0.0700,-0.0769,-0.0783`);
  - nfp4 QH has stable low-`ky` modes and positive growth at high `ky` (`gamma=-0.0199,-0.0180,-0.0108,+0.00850,+0.0206,+0.0328`), making it the stronger next nonlinear-window candidate;
  - tracked `docs/_static/quasilinear_vmec_jax_qh_linear_spectrum.{png,pdf,json}` plus source CSV companions as a linear-feasibility artifact, explicitly not a transport-validation or calibrated quasilinear claim.
  - added regression coverage for VMEC-EIK invalid-reference-scale and no-crossing errors so arbitrary external VMEC fixtures produce actionable user messages.
- Current next best steps:
  - run a bounded nfp4 QH nonlinear adiabatic-electron pilot with the same VMEC fixture and enough diagnostics to form a window-statistics gate;
  - if QH nonlinear is finite and interpretable, add a QH spectrum-shape gate and leave-one-geometry-out calibration point; otherwise record the failure mode and use Li383/CTH-like as the next production candidate;
  - keep arbitrary external VMEC files behind explicit feasibility and nonlinear-window gates until the geometry cut conventions are validated case by case.
- Completed reduced-grid nfp4 QH nonlinear pilots:
  - local `Nx=Ny=32`, `Nz=24`, `Nl=4`, `Nm=8`, `dt=0.05` run is finite to `t=5` and `t=20`;
  - the `t=20` late-half window has mean heat flux `1.78e-4`, final heat flux `3.58e-4`, and a positive late-half heat-flux slope of about `3.1e-5` per time unit;
  - follow-up QA extended the same reduced-grid QH lane to `t=150`; the late
    window is no longer startup-scale (mean heat flux about `19.6`), but QH is
    still a feasibility result until a grid/window convergence gate passes.
- Completed additional full-`ky` external-VMEC linear feasibility scans:
  - CTH-like has a useful unstable high-`ky` branch (`gamma=-0.0227,-0.0161,+0.00418,+0.0114,+0.0309,+0.0488`);
  - shaped tokamak remains stable over the sampled grid (`gamma=-0.0799,-0.0692,-0.0488,-0.0396,-0.0292,-0.0186`);
  - tracked `docs/_static/quasilinear_vmec_jax_cth_like_linear_spectrum.{png,pdf,json}` plus source CSV companions as another linear-feasibility artifact.
- Current next best steps:
  - run CTH-like as the next reduced-grid nonlinear pilot because its linear branch is stronger than QH on the current grid and it may saturate more clearly;
  - after saturated external-VMEC nonlinear windows exist, add QH/CTH/shape
    grid/window convergence and spectrum-shape gates before extending the
    leave-one-geometry-out quasilinear calibration set.
- Completed CTH-like reduced-grid nonlinear pilots:
  - local `Nx=Ny=32`, `Nz=24`, `Nl=4`, `Nm=8`, `dt=0.05` run is finite to `t=20` and `t=50`;
  - the `t=20` late-half heat-flux mean is `3.76e-4`, but the positive slope `8.33e-5` per time unit shows startup/growth rather than saturation;
  - the `t=50` run remains finite but is still strongly growing, with late-half mean heat flux `0.627`, final heat flux `3.15`, and late-half slope `0.102` per time unit;
  - CTH-like is therefore the strongest current external-VMEC nonlinear candidate, but it is not yet a saturated transport-calibration window.
- Completed office-backed long CTH-like nonlinear feasibility pilot:
  - ran fresh `Nx=Ny=32`, `Nz=24`, `Nl=4`, `Nm=8`, `dt=0.05`, `t=100` and `t=150` fixed-step RK3 pilots on office GPU from the clean `main` clone at `/home/rjorge/spectraxgk_main_runs`;
  - the `t=100` run is finite and enters a high-heat-flux state, but late-window slope diagnostics were mixed across windows;
  - the `t=150` run is finite with `61` samples, final heat flux `26.27`, final `Wphi=7.80`, and least-trending window `t=75.05..150.00` with mean heat flux `23.06`, standard deviation `1.79`, and relative heat-flux trend `1.20e-3` per time unit;
  - added `tools/plot_nonlinear_feasibility_pilot.py`, tests, and tracked `docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.{png,pdf,json,traces.csv}`;
  - the panel is deliberately labeled `finite_long_nonlinear_feasibility_not_transport_validation`; the promotion gate remains false until a production-resolution convergence/reference acceptance protocol is defined and passed.
- Current next best steps:
  - define the external-VMEC nonlinear acceptance protocol for CTH-like: production grid, window-start rule, trend tolerance, resolved-spectrum shape gate, and optional independent reference;
  - run a bounded CTH-like convergence check (`32^2x24` vs `48^2x32`, same `Nl/Nm`, same `dt`) before admitting it as a quasilinear calibration holdout;
  - only after that, add a CTH-like spectrum-shape gate against the linear quasilinear spectrum and a leave-one-geometry-out calibration point;
  - continue keeping these external-VMEC pilots out of quasilinear calibration until a saturated nonlinear window and spectrum-shape gate pass.
- Completed first CTH-like grid-convergence check:
  - ran office GPU `Nx=Ny=48`, `Nz=32`, `Nl=4`, `Nm=8`, `dt=0.05`, `t=150` with the same external VMEC fixture;
  - run completed in about `211 s`, stayed finite, and produced `61` diagnostic samples;
  - on the common `t=75.05..150.00` window, higher-grid mean heat flux is `12.82` versus `23.06` on the `32^2x24` run;
  - the higher-grid least-trending `t=120.05..150.00` window has mean heat flux `14.54`, standard deviation `0.271`, and relative trend `5.77e-4` per time unit;
  - tracked `docs/_static/external_vmec_cth_like_nonlinear_t150_n48_pilot.{png,pdf,json,traces.csv}`;
  - conclusion: CTH-like is finite and promising but not grid-converged; it remains excluded from quasilinear calibration.
- Current next best steps:
  - decide the production CTH-like nonlinear protocol before more long runs: either increase grid again, adjust dissipation/hypercollision with a documented physics rationale, or defer external-VMEC transport holdouts until a reference exists;
  - if continuing CTH-like, run one controlled parameter at a time and require late-window heat flux to be stable under both window choice and grid refinement;
  - otherwise return to the core quasilinear model lane with the current four validated nonlinear cases and keep external VMEC as a documented future holdout campaign.
- Closed the first explicit external-VMEC nonlinear convergence-gate slice:
  - added `tools/plot_external_vmec_nonlinear_convergence_gate.py` plus fast synthetic tests;
  - generated `docs/_static/external_vmec_cth_like_grid_convergence_gate.{png,pdf,json,csv}` from the existing `32^2x24` and `48^2x32` CTH-like `t=150` pilots;
  - the gate is intentionally excluded from the release validation index (`gate_index_include=false`) because it documents a negative research result rather than a release blocker;
  - gate outcome: common-window heat-flux symmetric relative difference `0.571`, least-trending-window difference `0.453`, threshold `0.15`; common-window relative trend also fails (`5.42e-3` per time unit versus `2e-3`);
  - conclusion unchanged but now machine-readable: CTH-like is finite and promising, but it is not admitted into quasilinear calibration until the grid/window protocol passes.
- Current next best steps:
  - return to the core quasilinear saturation-model lane using the currently validated nonlinear holdouts, because the external CTH-like holdout failed convergence;
  - if external-VMEC holdouts remain a priority, choose one controlled next run: either increase the CTH-like grid again, vary hypercollision/dissipation with a documented physics rationale, or choose a different VMEC candidate with a stronger reference basis;
  - add any future external-VMEC nonlinear candidate through the same pilot -> convergence-gate -> spectrum-shape-gate path before using it for calibration or optimization claims.
- Added a calibration-admission guard so the workflow always uses validated nonlinear inputs:
  - added `tools/check_quasilinear_calibration_inputs.py` and tests covering passed gates, missing gates, non-required audit points, and failed external-pilot promotion gates;
  - generated `docs/_static/quasilinear_validated_calibration_inputs.{png,pdf,json}` from the current Cyclone/Cyclone-Miller/HSX/W7-X train-holdout reports;
  - current audit passes: every train/holdout point maps to a passed nonlinear gate (`cyclone_nonlinear_long_window`, `cyclone_miller_nonlinear_window`, `hsx_nonlinear_window`, `w7x_nonlinear_window`);
  - this makes the validation/convergence policy executable: failed finite pilots can be documented, but they cannot silently enter quasilinear calibration.
- Current next best steps:
  - use the validated-input audit as a required precondition for future quasilinear saturation-model figures and optimization examples;
  - continue the saturation-model lane only on the validated four-case set unless a new nonlinear case passes pilot, convergence, and validation gates;
  - add the next candidate model only with a null baseline, leave-one-geometry-out scoring, prediction intervals, and finite-difference/autodiff checks before making any optimization-facing claim.
- Promoted the validated-input policy into CI:
  - added a tracked-regression test that recomputes the current quasilinear train/holdout input audit and requires all `12/12` train/holdout usages to map to passed nonlinear gates;
  - added a docs/packaging CI step that runs `tools/check_quasilinear_calibration_inputs.py --no-plot` on the current quasilinear train/holdout reports before building docs;
  - this makes the rule enforceable for future pull requests: calibration docs and figures cannot be rebuilt successfully if a non-converged or exploratory nonlinear artifact is inserted as train/holdout data.
- Current next best steps:
  - build the next saturation-model diagnostic on the validated four-case set, with the CI admission gate as a precondition;
  - keep external VMEC candidates out of calibration until a new candidate passes pilot, convergence, and validation gates;
  - when adding any new nonlinear reference, update both the nonlinear gate JSON and the quasilinear input audit before using it in a model figure.
- Propagated validated-input enforcement into quasilinear model-development scripts:
  - `tools/plot_quasilinear_saturation_rule_sweep.py`, `tools/plot_quasilinear_shape_aware_saturation.py`, and `tools/plot_quasilinear_candidate_uncertainty.py` now require passed nonlinear summary gates by default;
  - regenerated `docs/_static/quasilinear_saturation_rule_sweep.json`, `docs/_static/quasilinear_shape_aware_saturation.json`, and `docs/_static/quasilinear_candidate_uncertainty.json` with `input_validation.passed=true` for the validated four-case set;
  - added tests proving failed nonlinear summary gates are rejected and synthetic tests carry explicit passed gates.
- Current next best steps:
  - proceed to the next quasilinear saturation-candidate improvement only within the validated four-case dataset;
  - add at least one additional converged nonlinear holdout before promoting any higher-parameter candidate beyond a negative/model-development result;
  - keep the finalization plan split into release-ready validated tooling versus post-release external-VMEC/stellarator-optimization expansion.
- Added the W7-X fluctuation-spectrum simulation diagnostic panel:
  - added `tools/plot_w7x_fluctuation_spectrum_panel.py` plus a synthetic NetCDF regression test;
  - generated `docs/_static/w7x_fluctuation_spectrum_panel.{png,pdf,json,csv}` from the gated W7-X nonlinear `t≈200` NetCDF artifact;
  - the report requires the W7-X nonlinear gate summary to pass by default, records `claim_level="validated_nonlinear_simulation_spectrum_not_experimental_validation"`, and sets `gate_index_include=false`;
  - the current diagnostic has `76` samples over `t≈0.065..197.77`, dominant nonzonal `|\phi|^2` at `k_y rho_i≈0.190`, dominant heat-flux power near `k_y rho_i≈1.286`, and dominant zonal `k_x rho_i≈-0.100`;
  - documentation now treats the simulation-spectrum estimator as closed while leaving Doppler-reflectometry transfer-function validation as a post-release manuscript extension.
- Current next best steps:
  - explicitly keep W7-X zonal long-window recurrence/damping as a deferred manuscript blocker rather than a release blocker;
  - run bounded docs/build/package checks for the new spectrum artifact and push the release-scope updates;
  - wait for GitHub CI to go green before any version bump, tag, or PyPI release.
- Started the post-v1.5 open-lane execution pass:
  - added `tools/build_open_research_lane_status.py` to read the current W7-X zonal, W7-X fluctuation-spectrum, quasilinear holdout, differentiable-geometry, and nonlinear-profiler artifacts and summarize each lane as `closed`, `partial`, `open`, or `blocked`;
  - generated `docs/_static/open_research_lane_status.{png,pdf,json,csv}` as a claim-scope figure and machine-readable dashboard;
  - current executable status: W7-X long-window zonal recurrence/damping is open with seven failed reference/envelope gates; quasilinear absolute-flux promotion is open with three current holdouts but `passed=false`; W7-X fluctuation spectra, the differentiable-geometry bridge, and nonlinear profiler identity artifacts are partial bounded diagnostics;
  - updated the validation coverage manifest, roadmap, and manuscript figure index so these lanes cannot be accidentally described as closed physics claims.
- Current next best steps:
  - run the next W7-X zonal physical closure sweep on office using one controlled knob at a time (moment resolution, Hermite/Laguerre hypercollision source, then time horizon) and admit a candidate only if residual, late-envelope, and moment-tail gates improve together;
  - keep CTH-like external VMEC excluded and choose either a higher-grid CTH run or a different VMEC candidate for the next converged nonlinear holdout;
  - extend the W7-X fluctuation/TEM lane with multi-alpha/multi-surface kinetic-electron scans before broad W7-X claims;
  - connect real in-memory `vmec_jax`/`booz_xform_jax` output into `FluxTubeGeometryData` and add parity plus gradient gates before optimization claims;
  - collect matched CPU/GPU profiler traces before making any new speedup claim.
- Advanced the W7-X zonal physical-closure lane with a bounded office-GPU constant-Hermite-hypercollision probe:
  - ran paper-facing W7-X test-4 `k_x rho_i=0.07`, `Nl=16`, `Nm=64`, `dt=0.05`, `t v_t/a=100` for `nu_hyper_m=0.01` and `0.03` with constant hypercollision source;
  - generated `docs/_static/w7x_zonal_hypercollision_probe_kx070.{png,pdf,json,csv}` and wired the result into `tools/build_open_research_lane_status.py`;
  - result is negative but useful: `nu_hyper_m=0.03` lowers final Hermite-tail fraction to about `0.099` and free-energy ratio to about `0.600`, but mean trace error remains about `0.289` and the late-window standard-deviation ratio is about `4.28`, so the W7-X zonal lane remains open physically;
  - updated roadmap/testing/manuscript docs and the validation coverage manifest so the result is tracked as an open recurrence/closure diagnostic, not a validation closure.
- Current next best steps:
  - W7-X zonal: stop increasing constant damping alone; test one physically motivated closure/operator change at a time and require residual, late-envelope, and moment-tail gates to improve together before promotion;
  - W7-X fluctuation/TEM: extend the existing simulation-spectrum panel to multi-alpha/multi-surface and kinetic-electron/TEM scans before broad W7-X claims;
  - quasilinear holdouts: add a new nonlinear holdout only after pilot, window, grid-convergence, and spectrum-shape gates pass; keep failed CTH-like external VMEC out of calibration;
  - differentiable geometry: connect a real in-memory `vmec_jax`/`booz_xform_jax` state into `FluxTubeGeometryData` and add geometry parity plus gradient gates before optimization claims;
  - performance: collect matched CPU/GPU profiler traces for nonlinear bracket/field-solve hot paths before making any speedup claim.
- Added an executable W7-X fluctuation/TEM extension-status gate:
  - added `tools/build_w7x_tem_extension_status.py` and `tests/test_build_w7x_tem_extension_status.py`;
  - generated `docs/_static/w7x_tem_extension_status.{png,pdf,json,csv}` from the current W7-X nonlinear fluctuation panel and `docs/_static/tem_mismatch_table.csv`;
  - current result: W7-X simulation-spectrum estimator is closed (`76` samples, dominant nonzonal `ky rho_i≈0.190`), but TEM linear parity remains open with max absolute relative growth mismatch `4.254` and frequency mismatch `351.534`; multi-alpha/multi-surface W7-X scans and kinetic-electron nonlinear windows are explicitly not admitted yet;
  - wired this artifact into `tools/build_open_research_lane_status.py`, docs, and the validation coverage manifest so broad W7-X/TEM claims stay blocked until the missing physics gates exist.
- Current next best steps:
  - fix the TEM branch/frequency mismatch before starting kinetic-electron W7-X nonlinear validation;
  - then run W7-X alpha/surface-resolved linear scans and only promote nonlinear windows after branch, transport-window, and resolved-spectrum gates pass;
  - continue nonlinear holdout expansion through the existing pilot -> convergence -> validation -> calibration-admission path, not by adding unconverged external VMEC runs;
  - continue geometry bridge and profiler lanes with real parity/gradient/profiler artifacts rather than claim-only docs.
- Refreshed the bounded local CPU nonlinear RHS split profile:
  - reran `tools/profile_nonlinear_step_split.py` for Cyclone grid and spectral Laguerre modes with a hard `240 s` process timeout and `5` warm kernel repeats;
  - regenerated `docs/_static/nonlinear_rhs_profile_cpu.csv`, `docs/_static/nonlinear_rhs_profile_cpu_spectral.csv`, and `docs/_static/nonlinear_rhs_profile.{png,pdf}`;
  - current local CPU numbers: grid `full_rhs≈8.14e-2 s`, spectral `full_rhs≈7.97e-2 s`, nonlinear bracket `3.93e-2 s` vs `2.38e-2 s`; this is useful profiler evidence but not a production speedup claim;
  - documented the bounded CPU refresh in `docs/performance.rst` while keeping spectral Laguerre as opt-in until case-level parity and larger matched CPU/GPU profiler gates support broader claims.
- Current next best steps:
  - run the matching GPU profiler refresh on office for the same script/config before changing performance claims;
  - use those traces to decide whether to optimize bracket FFT/gather paths, field-solve assembly, or cold-start cache construction first.
- Refreshed the matched office GPU nonlinear RHS split profile sequentially after a first parallel attempt showed host/GPU contention:
  - reran grid and spectral Laguerre profiles on one office GPU at a time with `10` warm kernel repeats and `300 s` timeouts;
  - regenerated `docs/_static/nonlinear_rhs_profile_gpu.csv`, `docs/_static/nonlinear_rhs_profile_gpu_spectral.csv`, and the combined `docs/_static/nonlinear_rhs_profile.{png,pdf}`;
  - current office GPU numbers: grid `full_rhs≈1.14e-2 s`, spectral `full_rhs≈7.29e-3 s`, nonlinear bracket `3.15e-3 s` vs `1.52e-3 s`;
  - kept the claim scoped: this supports bracket/spectral-mode profiling, but production speedup claims still require case-level parity gates and larger matched CPU/GPU sweeps.
- Current next best steps:
  - start the TEM branch/frequency audit by comparing the existing reference table, runtime parameters, and fitted signals term by term;
  - if no bounded code fix is obvious, create a machine-readable TEM blocker artifact like the W7-X status artifacts so kinetic-electron W7-X remains gated.
- Added the TEM branch/frequency blocker artifact:
  - added `tools/build_tem_branch_parity_audit.py` and `tests/test_build_tem_branch_parity_audit.py`;
  - regenerated `docs/_static/tem_branch_parity_audit.{png,pdf,json,csv}`, `docs/_static/w7x_tem_extension_status.{png,pdf,json,csv}`, and `docs/_static/open_research_lane_status.{png,pdf,json,csv}`;
  - current audit result: TEM remains open with `max |rel gamma|≈4.254`, `max |rel omega|≈3.3` away from the near-zero reference denominator, `max |Delta gamma|≈1.815`, `max |Delta omega|≈3.976`, one growth-rate sign mismatch, three frequency sign mismatches, and an inverted frequency branch (`Spearman≈-0.986`);
  - quick reduced-moment local probe at `ky=0.3` showed signal choice matters (`phi` gives the wrong frequency sign, electron-density signal is closer but still misses the digitized target), so the next fix must reconstruct the exact TEM case/reference before changing solver physics.
- Current next best steps:
  - keep W7-X/TEM kinetic-electron validation blocked until the full TEM case definition or an independent reference dump is available;
  - resume W7-X zonal physical closure under the paper-facing initializer/observable convention;
  - advance the real `vmec_jax`/`booz_xform_jax` geometry bridge with parity and gradient gates.
- Tightened the W7-X zonal bounded-recurrence artifact:
  - added explicit `tail_std_ratio = tail_std/reference_tail_std` to `tools/plot_w7x_zonal_recurrence_sweep.py` rows and its focused tests;
  - repaired the tracked `docs/_static/w7x_zonal_recurrence_sweep_kx070.{json,csv}` metadata from existing stored row values because the source NetCDF sweep outputs are not present in the local checkout;
  - current best bounded candidate remains `Nl12 Nm48 none` with `mean_abs_error≈0.276`, `tail_std_ratio≈3.78`, and Hermite-tail fraction `≈0.310`; the best constant hypercollision probe still worsens trace error while suppressing Hermite tail, so the lane remains a physical closure issue, not a normalization/doc issue.
- Current next best steps:
  - if the W7-X source NetCDFs are needed for a refreshed plot, rerun the bounded recurrence sweep on office or restore the release artifacts from external storage;
  - test a physically motivated velocity-space closure/operator rather than increasing constant Hermite damping;
  - continue the differentiable geometry bridge and keep W7-X zonal closure open until residual, late-envelope, and moment-tail gates pass together.
- Advanced the differentiable geometry bridge:
  - promoted the real `vmec_jax` boundary-aspect AD-vs-finite-difference check into `spectraxgk.geometry.differentiable.vmec_boundary_aspect_sensitivity_report`;
  - expanded backend auto-discovery to check `~/vmec_jax`, `~/local/vmec_jax`, `~/booz_xform_jax`, and `~/local/booz_xform_jax` in addition to sibling checkouts and explicit environment variables;
  - updated the example to use the package API and refreshed `docs/_static/differentiable_geometry_bridge.json` with the real local `vmec_jax`/`booz_xform_jax` APIs discovered; current real boundary-aspect derivative AD/FD max absolute error is `≈1.97e-10`;
  - added CI-safe fake-backend tests for the VMEC boundary sensitivity bridge, so the API contract is tested even when the real optional backends are not installed.
- Current next best steps:
  - connect a real `vmec_jax` equilibrium output to the sampled `FluxTubeGeometryData` contract, then compare against the existing VMEC/eik import path on a small equilibrium;
  - add a gradient gate for a solver-facing geometric observable after geometry parity passes;
  - keep full stellarator optimization claims scoped until both geometry parity and geometry-gradient gates pass.
- Added the differentiable QA stellarator ITG objective-reduction gate:
  - added `src/spectraxgk/stellarator_optimization.py` with a JAX-native max-mode-1 QA control map targeting aspect `7`, mean iota `0.41`, and smooth quasisymmetry/ITG observables;
  - added three optimization objectives: linear ITG growth rate, differentiable quasilinear ITG heat-flux proxy, and a differentiable nonlinear heat-flux envelope with late-window mean/CV/trend diagnostics;
  - added example scripts under `examples/optimization/` for growth-rate, quasilinear-flux, nonlinear-window, and three-objective comparison workflows;
  - generated `docs/_static/stellarator_itg_growth_optimization.{png,pdf,json,history.csv}`, `docs/_static/stellarator_itg_quasilinear_optimization.{png,pdf,json,history.csv}`, `docs/_static/stellarator_itg_nonlinear_optimization.{png,pdf,json,history.csv}`, and `docs/_static/stellarator_itg_optimization_comparison.{png,pdf,json}`;
  - current result: all three objectives pass AD-vs-finite-difference gates, keep the optimized state near the target aspect/iota constraints, reduce growth to about `57%` of its initial value, and reduce quasilinear plus nonlinear-window heat-flux observables to about `41%` of their initial values;
  - added `tests/test_stellarator_optimization.py` so the objective contracts, gradient gates, nonlinear window metrics, optimizer reduction, UQ/covariance diagnostics, and comparison payload shape remain covered in the fast test suite;
  - documented the workflow in `docs/stellarator_optimization.rst`, README, and `docs/manuscript_figures.rst`.
- Scope note:
  - this closes a differentiable optimization and UQ gate, not the full production `vmec_jax -> booz_xform_jax -> SPECTRAX-GK` nonlinear optimization claim;
  - the full claim still requires in-memory VMEC/Boozer-to-`FluxTubeGeometryData` parity, geometry-gradient gates through production solver caches, removal or isolation of host scalar materialization in traced geometry paths, and converged nonlinear audits of optimized geometries.
- Current next best steps:
  - implement the real `vmec_jax VMECState -> booz_xform_jax -> FluxTubeGeometryData` bridge using a small equilibrium and compare sampled field-line arrays against the existing VMEC/eik import path;
  - add finite-difference and implicit-eigenpair gradient checks for growth rate, frequency, and quasilinear weights through that bridge;
  - replace the reduced nonlinear envelope with either a trace-safe production nonlinear objective or a documented stochastic/window estimator only after nonlinear identity, convergence, and profiler gates pass;
  - keep W7-X zonal long-window recurrence/damping and TEM branch/frequency blockers deferred while the differentiability lane advances.
- Strengthened the differentiable Boozer bridge beyond import discovery:
  - added `spectraxgk.geometry.differentiable.booz_xform_spectral_sensitivity_report`, a bounded real-API `booz_xform_jax` spectral transform gate on a one-surface axisymmetric bundle;
  - the gate checks the derivative of a Boozer magnetic-spectrum norm with respect to a ripple coefficient against central finite differences when `booz_xform_jax` is available, and reports an explicit unavailable status otherwise;
  - wired the report into the differentiable-geometry example JSON/panel, README, docs, package exports, and fast tests.
- Current next best steps:
  - implement the real `vmec_jax VMECState -> booz_xform_jax -> FluxTubeGeometryData` bridge with a small equilibrium and compare sampled field-line arrays against the existing VMEC/eik import path;
  - after parity passes, add growth-rate/frequency/quasilinear-weight gradient gates through the production linear solver cache;
  - only then promote full stellarator optimization claims beyond the reduced objective-reduction gate.
- Advanced the Boozer-to-flux-tube differentiability bridge:
  - added `evaluate_boozer_bmag_on_field_line`, `booz_xform_flux_tube_mapping_from_inputs`, and `booz_xform_flux_tube_sensitivity_report`;
  - the new bounded gate runs the real `booz_xform_jax` functional transform, samples the Boozer `|B|` spectrum on a field line, builds the solver-ready `FluxTubeGeometryData` mapping, and checks geometry-observable sensitivities against central finite differences;
  - refreshed `docs/_static/differentiable_geometry_bridge.{png,json}` so the publication artifact now reports VMEC boundary AD/FD `≈1.97e-10`, Boozer spectral AD/FD `≈2.88e-12`, and Boozer flux-tube AD/FD max absolute error `≈1.54e-08` / relative error `≈5.21e-08`;
  - added fast tests and public exports for the new bridge.
- Scope note:
  - this is a real `booz_xform_jax` derivative path into the SPECTRAX-GK geometry contract, but it still uses a smooth metric/drift closure; it is not yet full VMEC/Boozer metric parity or a production nonlinear optimization claim.
- Current next best steps:
  - replace the smooth metric/drift closure with sampled VMEC/Boozer metric tensors for a small equilibrium and compare against the existing imported VMEC/eik contract;
  - add geometry-gradient gates through one production linear solve after sampled-array parity passes;
  - then connect quasilinear objective gradients to the full geometry bridge before any nonlinear optimization promotion.
- Advanced the differentiable geometry bridge all the way back to a real
  ``vmec_jax`` state:
  - added ``spectraxgk.geometry.differentiable.vmec_jax_boozer_flux_tube_sensitivity_report``;
  - the new optional gate loads the local ``vmec_jax`` ``circular_tokamak``
    example, perturbs two ``VMECState`` Fourier coefficients
    ``[Rcos(radial_index, mode_index), Zsin(radial_index, mode_index)]``,
    converts the traced state through ``vmec_jax.booz_input`` and
    ``booz_xform_jax``, samples the resulting Boozer ``|B|`` along a field
    line, builds the SPECTRAX-GK ``FluxTubeGeometryData`` input mapping, and
    checks geometry-observable sensitivities against central finite
    differences;
  - refreshed ``docs/_static/differentiable_geometry_bridge.{png,json}``;
    current VMEC-state-to-Boozer-to-SPECTRAX AD/FD max absolute error is
    ``5.77e-7`` and max relative error is ``1.39e-8``;
  - updated the open-lane dashboard so the remaining differentiable-geometry
    action is sampled VMEC/Boozer metric/drift parity and solver-observable
    gradient gates, not the initial VMEC-state plumbing.
- Scope note:
  - this is now a real ``vmec_jax -> booz_xform_jax -> SPECTRAX-GK``
    differentiability gate for the tracked geometry observables;
  - it still uses the current smooth metric/drift closure, so it is not yet a
    production VMEC/Boozer metric-parity claim and not yet a nonlinear
    stellarator-optimization claim.
- Current next best steps:
  - replace the smooth closure in the bridge with sampled VMEC/Boozer metric
    tensors for a small equilibrium;
  - compare sampled arrays against the imported VMEC/eik runtime path;
  - add finite-difference or implicit-eigenpair checks for linear growth,
    frequency, and quasilinear weights through the production solver cache;
  - only then connect the QA optimization examples to the full geometry bridge
    and keep nonlinear heat-flux optimization gated on converged nonlinear
    windows.
- Added the upstream VMEC metric-tensor differentiability gate:
  - added ``spectraxgk.geometry.differentiable.vmec_jax_metric_tensor_sensitivity_report`` and
    ``vmec_metric_tensor_observable_names``;
  - the gate loads the same real ``vmec_jax`` ``circular_tokamak`` state,
    perturbs the VMEC Fourier coefficients, evaluates
    ``vmec_jax.geom.eval_geom``, and checks sampled covariant metric/Jacobian
    observables ``[sqrtg_rms, mean_g_ss, mean_g_tt, mean_g_pp, g_st_rms,
    g_sp_rms, g_tp_rms]`` against central finite differences;
  - refreshed ``docs/_static/differentiable_geometry_bridge.{png,json}``;
    current VMEC metric-tensor AD/FD max absolute error is ``5.88e-8`` and max
    relative error is ``1.30e-7``;
  - updated the open-lane dashboard and docs so the bridge status records both
    real VMEC metric derivatives and real VMEC-state-to-Boozer derivatives.
- Scope note:
  - the path now differentiates through real ``vmec_jax`` state geometry and
    real ``booz_xform_jax`` spectra, but the SPECTRAX-GK field-line mapping
    still uses a smooth placeholder for metric/drift closure;
  - production stellarator optimization remains gated on replacing that closure
    with sampled VMEC/Boozer field-line tensors and matching the imported
    VMEC/eik runtime path.
- Current next best steps:
  - build a small-equilibrium sampled field-line tensor fixture from the VMEC
    metric output plus Boozer straight-field-line coordinates;
  - compare ``bmag``, ``gradpar``, ``gds2``, ``gds21``, ``gds22``, drifts,
    Jacobian, and ``grho`` against the imported VMEC/eik runtime path;
  - then add production linear/quasilinear AD-vs-FD gates through this fixture.
- Added the first non-axisymmetric VMEC field-line tensor differentiability gate:
  - added ``spectraxgk.geometry.differentiable.vmec_jax_field_line_tensor_sensitivity_report`` and
    ``vmec_field_line_tensor_observable_names``;
  - the gate loads the real ``vmec_jax`` ``nfp4_QH_warm_start`` stellarator
    fixture, perturbs VMEC Fourier coefficients, evaluates
    ``vmec_jax.geom.eval_geom`` plus ``vmec_jax.vmec_bcovar``, samples a fixed
    VMEC field-line convention, and checks ``|B|`` ripple plus raw metric
    tensor observables ``[mean_bmag, relative_bmag_ripple, sqrtg_rms,
    mean_g_tt, mean_g_pp, g_tp_rms, mean_g_ss]`` against central finite
    differences;
  - this explicitly addresses the optimization requirement that
    differentiability reaches the VMEC state and field-line tensors, not only
    the Boozer spectral adapter;
  - refreshed ``docs/_static/differentiable_geometry_bridge.{png,json}`` and
    the open-lane dashboard to record this field-line tensor gate.
- Scope note:
  - this is a real upstream VMEC field-line tensor check, but it intentionally
    does not relabel the current smooth SPECTRAX-GK metric/drift closure as
    production-ready;
  - the next production step remains deriving the SPECTRAX-GK ``gds*``, drift,
    Jacobian, ``gradpar``, and ``grho`` arrays from sampled VMEC/Boozer tensors
    and matching those arrays against the imported VMEC/eik path.
- Added the first direct VMEC tensor-to-SPECTRAX flux-tube differentiability gate:
  - fixed optional-backend discovery so explicitly configured/local
    ``vmec_jax`` and ``booz_xform_jax`` checkouts are preferred over globally
    installed packages; this matters because the local ``vmec_jax`` checkout
    carries the example ``wout`` files used by real derivative gates;
  - added ``spectraxgk.geometry.differentiable.vmec_jax_flux_tube_mapping_from_state``
    and ``vmec_jax_flux_tube_sensitivity_report``;
  - the new gate starts from the real ``nfp4_QH_warm_start`` ``VMECState``,
    evaluates ``vmec_jax.geom`` and ``vmec_jax.vmec_bcovar``, samples a fixed
    VMEC field line, inverts the sampled VMEC metric tensor, derives
    SPECTRAX-GK ``bmag``, ``gradpar``, ``gds*``, Jacobian, ``grho``, and a
    local grad-``B`` drift closure, builds the ``FluxTubeGeometryData`` mapping,
    and checks geometry-observable sensitivities against central finite
    differences;
  - focused local gate result: max relative AD/FD error is ``≈1.35e-4`` with
    ``fd_step=2e-6`` on the tracked geometry observables;
  - refreshed the differentiable-geometry example/status tooling so the JSON,
    panel, docs, and open-lane dashboard report the direct VMEC tensor
    flux-tube derivative gate.
- Scope note:
  - this addresses the requirement that differentiability reaches back to
    ``vmec_jax`` state coefficients and not only ``booz_xform_jax`` spectra;
  - this still is not a full production stellarator transport optimization
    claim: the direct VMEC tensor-derived arrays must be compared against the
    imported VMEC/eik runtime path, and the local grad-``B`` drift closure must
    be replaced or matched to the production VMEC/EIK drift convention before
    solver-observable gradient gates are promoted.
- Current next best steps:
  - build a small array-parity harness comparing direct VMEC tensor-derived
    ``bmag``, ``gradpar``, ``gds2``, ``gds21``, ``gds22``, Jacobian, ``grho``,
    and drifts against the imported VMEC/eik path on the same equilibrium;
  - once array parity passes, add finite-difference/implicit-eigenpair gates
    for growth rate, frequency, and quasilinear weights through the production
    solver cache;
  - only after those gates pass connect the QA optimization examples to the
    production ``vmec_jax`` geometry bridge for stellarator optimization.
- Added the first direct VMEC tensor-vs-imported-VMEC/EIK array parity audit:
  - added ``spectraxgk.geometry.differentiable.vmec_jax_flux_tube_array_parity_report``;
  - the audit starts from the same real ``nfp4_QH_warm_start`` ``VMECState``, builds the direct
    ``vmec_jax`` tensor-derived SPECTRAX-GK flux-tube mapping, generates the existing imported
    VMEC/EIK geometry on the same surface, trims the closed endpoint, and compares solver-facing
    arrays ``bmag``, ``bgrad``, ``gds*``, drifts, Jacobian, and ``grho`` plus scalar ``gradpar``,
    ``q``, and ``s_hat``;
  - the current result is intentionally reported as ``diagnostic_open``: ``q`` and ``s_hat`` are
    close, while the metric/Jacobian/``grho``/drift arrays are not yet production-parity because the
    direct path still uses a VMEC-coordinate/equal-theta convention and a local grad-``B`` closure
    rather than the imported Boozer equal-arc/Hegna-Nakajima geometry convention;
  - refreshed the differentiable-geometry artifact and open-lane dashboard so this open production
    gap is visible instead of being hidden behind the successful AD/FD gate.
- Current next best steps:
  - derive the direct JAX path in the same Boozer equal-arc coordinate convention as the imported
    VMEC/EIK generator, or reuse the JAX-native Boozer coefficients with the full metric/drift
    reconstruction instead of only the ``|B|`` spectrum;
  - replace the local grad-``B`` drift closure with the production Hegna-Nakajima/imported-VMEC
    drift convention;
  - rerun the array-parity audit until ``bmag``, ``gradpar``, ``gds*``, Jacobian, ``grho``, and
    drift profiles pass field-level tolerances; only then promote growth-rate, quasilinear-flux,
    and nonlinear-window gradients through the production solver cache.
- Added a Boozer equal-arc core parity gate for the differentiable VMEC bridge:
  - added ``spectraxgk.geometry.differentiable.vmec_jax_boozer_equal_arc_core_profiles_from_state``;
  - the gate starts from a real ``vmec_jax`` ``VMECState``, converts to
    ``booz_xform_jax`` inputs, evaluates the JAX-native Boozer magnetic
    spectrum, applies the imported VMEC/EIK equal-arc remap, and compares
    ``theta``, ``bmag``, ``bgrad``, Jacobian, scalar ``gradpar``, ``q``, and
    ``s_hat`` against the existing imported VMEC/EIK runtime path on the same
    ``nfp4_QH_warm_start`` surface;
  - current artifact result: equal-arc core normalized worst error is
    ``4.46e-3``, scalar worst relative error is ``2.35e-3``, and the
    derivative-like ``bgrad`` worst normalized error is ``2.26e-2`` under its
    separate derivative tolerance;
  - refreshed ``docs/_static/differentiable_geometry_bridge.{png,json}`` and
    the open-lane dashboard so the bridge now distinguishes the closed
    Boozer equal-arc core convention from the still-open full metric/drift
    production parity gap.
- Follow-up from this entry: reconstructing ``gds*``/``grho`` from the
  matched equal-arc path is now closed by the metric subgate below.
- Extended the Boozer equal-arc parity gate to zero-beta metric profiles:
  - ``vmec_jax_boozer_equal_arc_core_profiles_from_state`` now reconstructs
    ``gds2``, ``gds21``, ``gds22``, and ``grho`` from the real
    ``booz_xform_jax`` Boozer ``R/Z/nu/B`` spectra, radial derivatives, and
    the same equal-arc remap used by the imported VMEC/EIK path;
  - the tracked ``nfp4_QH_warm_start`` artifact now passes a separate metric
    subgate with worst normalized mismatch ``3.45e-2``:
    ``gds2=2.49e-2``, ``gds21=3.45e-2``, ``gds22=3.10e-2``, and
    ``grho=1.58e-2``;
  - this closes the release-tolerance field-line normalization and zero-beta
    metric convention for the differentiable VMEC/Boozer bridge while keeping
    the production Hegna-Nakajima curvature/drift parity gap explicit.
- Current next best steps:
  - broaden the JAX-native Boozer equal-arc drift parity gate beyond the
    current zero-beta tracked fixture to finite-beta pressure corrections and
    additional stellarator equilibria;
  - after the broader finite-beta drift gates pass, add production solver
    gradients for linear growth/frequency, quasilinear weights, and then
    nonlinear-window objectives before making stellarator optimization claims.
- Added the first zero-beta Boozer equal-arc drift parity subgate:
  - ``vmec_jax_boozer_equal_arc_core_profiles_from_state`` now reconstructs
    loaded-convention ``cvdrift``, ``gbdrift``, ``cvdrift0``, and
    ``gbdrift0`` from the real Boozer ``B`` spectrum, radial ``dB/ds``,
    equal-arc remap, and the same root-level drift-loader factor used by
    ``load_gx_geometry_netcdf``;
  - the tracked ``nfp4_QH_warm_start`` artifact now passes a separate drift
    subgate with worst normalized mismatch ``3.50e-2`` after raising and
    enforcing the default Boozer parity mode count to ``mboz=nboz=21``:
    ``cvdrift=3.50e-2``, ``gbdrift=3.50e-2``, ``cvdrift0=3.03e-2``, and
    ``gbdrift0=3.03e-2``;
  - the release claim is now closed for the tracked zero-beta Boozer
    equal-arc field-line, metric, and drift convention, while finite-beta,
    multi-equilibrium drift parity and solver-objective gradients remain
    required before stellarator heat-flux optimization claims.
- Bounded follow-up probe after the drift subgate:
  - ``LandremanPaul2021_QA_lowres`` is not usable for this runtime EIK parity
    path as shipped in ``vmec_jax`` because its bundled ``wout`` reports
    ``Aminor_p=0`` and the runtime generator correctly rejects it;
  - ``shaped_tokamak_pressure`` passes the small ``ntheta=8`` equal-arc
    core/metric/drift smoke gate, including a drift worst normalized mismatch
    of ``7.10e-3``;
  - ``nfp3_QI_fixed_resolution_final`` passes core/metric smoke gates but fails
    the drift smoke gate with ``mboz=nboz=8`` at worst normalized mismatch
    ``1.82e-1``; increasing the Boozer parity mode count to ``21`` reduces the
    QI drift mismatch to ``7.13e-2`` and passes the release drift tolerance,
    which identifies the immediate issue as spectral truncation rather than a
    normalization change;
  - a trial shear-HNGC correction using the wrong input-convention factor was
    explicitly rejected because it worsened the tracked QH metric gate.
- Enforced ``mboz,nboz >= 21`` for the VMEC/Boozer equal-arc parity helpers so
  future runs do not silently fall back to the under-resolved QI drift setting.
- Added ``tools/build_vmec_boozer_parity_matrix.py`` and
  ``docs/_static/vmec_boozer_parity_matrix.{png,pdf,json,csv}`` to make the
  mode-21 result replayable across the tracked QH, QI, and shaped-tokamak
  examples. All three current rows pass the equal-arc core, scalar, ``bgrad``,
  metric, and drift subgates at ``mboz=nboz=21``; the QI drift row remains the
  limiting release-level value at ``7.13e-2`` against the ``8e-2`` tolerance.

### 2026-04-30

- Added the manuscript-scope readiness dashboard:
  - ``tools/build_manuscript_readiness_status.py`` now reads the quasilinear
    calibration/model-selection artifacts, the differentiable-geometry bridge,
    the mode-21 VMEC/Boozer parity matrix, the reduced stellarator ITG
    optimization comparison, and the nonlinear sharding profiler artifact;
  - ``docs/_static/manuscript_readiness_status.{png,pdf,json,csv}`` records
    the current manuscript claim surface with W7-X zonal recurrence/damping
    and TEM/kinetic-electron stellarator extension explicitly deferred;
  - in this narrower scope, the quasilinear lane is closed as a validated
    diagnostic/model-selection negative result rather than as an absolute-flux
    predictor, VMEC/Boozer equal-arc geometry parity is closed at
    ``mboz=nboz=21``, and reduced differentiable stellarator ITG optimization
    is closed with AD/FD gates;
  - the active manuscript blocker is now production solver-objective geometry
    gradients through the mode-21 VMEC/Boozer bridge, while profiler-backed
    nonlinear speedup claims remain partial and require fresh CPU/GPU profiler
    artifacts before any new performance claim.
- Current next best steps:
  - add a finite-difference/implicit-eigenpair gate for linear growth rate,
    real frequency, and electrostatic quasilinear weights through the matched
    mode-21 VMEC/Boozer geometry bridge;
  - connect that gate to a small, replayable stellarator fixture before
    promoting any full end-to-end stellarator heat-flux optimization claim;
  - keep W7-X zonal recurrence/damping and TEM/kinetic-electron stellarator
    validation as post-manuscript lanes unless the manuscript scope changes.
- Added the first production-adjacent solver-objective geometry-gradient gate:
  - ``src/spectraxgk/solver_objective_gradients.py`` differentiates actual
    electrostatic linear-RHS eigenpair observables with respect to
    solver-ready ``FluxTubeGeometryData`` arrays;
  - ``tools/build_solver_objective_gradient_gate.py`` writes
    ``docs/_static/solver_objective_gradient_gate.{png,pdf,json,csv}``;
  - the gate uses the implicit left/right eigenpair sensitivity system and
    checks ``gamma``, ``omega``, ``<k_perp^2>``, linear heat/particle-flux
    weights, and a mixing-length heat-flux proxy against nearest-branch
    central finite differences;
  - the current gate passes with maximum relative AD/FD error below ``5e-3``
    on the tracked small electrostatic linear-RHS fixture, and the manuscript
    readiness dashboard now marks production solver-objective gradients as
    ``partial`` rather than ``open``;
  - this is intentionally scoped as a solver-ready geometry-gradient gate, not
    yet a full ``vmec_jax -> booz_xform_jax -> SPECTRAX-GK``
    state-coefficient gradient claim.
- Current next best steps:
  - feed actual mode-21 VMEC/Boozer equal-arc state-coefficient perturbations
    into the same solver-objective gradient gate;
  - add a small stellarator linear/quasilinear objective fixture once that
    state-gradient path passes;
  - only after linear/quasilinear state-gradient gates pass, add nonlinear
    window objective gradients or a documented stochastic/finite-difference
    estimator with convergence diagnostics.
- Added the first full VMEC/Boozer state-to-solver gradient gate:
  - fixed a real traceability bug in ``build_linear_cache`` where sampled
    magnetic shear was host-materialized with ``float(...)`` even on
    non-twist-shift grids; periodic sampled geometry now remains
    differentiable in traced ``s_hat``, while traced shear under twist-shift is
    rejected explicitly because that path still changes host topology
    (``jtwist``/``x0``);
  - added ``mode21_vmec_boozer_linear_frequency_gradient_report`` and
    ``tools/build_vmec_boozer_solver_frequency_gradient_gate.py``;
  - generated
    ``docs/_static/vmec_boozer_solver_frequency_gradient_gate.{png,pdf,json,csv}``;
  - the gate starts from a real ``vmec_jax`` ``VMECState`` coefficient,
    converts through ``booz_xform_jax`` with ``mboz=nboz=21``, builds the
    SPECTRAX-GK linear RHS, and checks the eigenfrequency gradient with the
    implicit left/right eigenpair method against central finite differences;
  - the current artifact passes with maximum relative AD/FD error
    ``4.89e-3`` for the frequency derivative at conditioned step ``1e-6``.
- Scope note:
  - this closes the full-chain linear eigenfrequency-gradient trace from
    ``vmec_jax`` state coefficients through the solver;
  - it does **not** yet close full-chain quasilinear flux-weight gradients or
    nonlinear-window gradients. Two bounded quasilinear-observable attempts
    timed out at 180--240 seconds, so the next step is profiling/conditioning
    that heavier eigenvector-dependent diagnostic before using it in the
    manuscript.
- Current next best steps:
  - profile and reduce the full-chain quasilinear flux-weight state-gradient
    gate, likely by caching/reusing VMEC/Boozer geometry products and reducing
    repeated objective-time geometry reconstruction;
  - add a publication figure comparing solver-ready versus full-chain
    frequency-gradient conditioning;
  - then connect the differentiable stellarator optimization examples to the
    passed full-chain linear gate while keeping quasilinear and nonlinear
    optimization claims scoped until their gates pass.
- Closed the full-chain VMEC/Boozer quasilinear-gradient gate for reduced
  stellarator objectives:
  - replaced the implicit eigenpair observable Jacobian with a split chain
    rule in ``spectraxgk.autodiff_validation``. The math is unchanged, but the
    expensive VMEC/Boozer parameter derivative is now evaluated only along the
    actual parameter directions instead of being carried through every
    eigenvector-component tangent;
  - added ``mode21_vmec_boozer_quasilinear_gradient_report`` and
    ``tools/build_vmec_boozer_quasilinear_gradient_gate.py``;
  - generated
    ``docs/_static/vmec_boozer_quasilinear_gradient_gate.{png,pdf,json,csv}``;
  - the tracked gate uses ``mboz=nboz=21`` and a richer ``Nl=2, Nm=3`` moment
    basis so the electrostatic heat-flux weight is nonzero;
  - the gate checks ``gamma``, ``omega``, ``<k_perp^2>``, linear ion
    heat-flux weight, and ``gamma Q_i/k_perp^2`` against nearest-branch
    central finite differences and passes with maximum relative error
    ``4.28e-3`` in about 34 seconds on the local CPU.
- Scope note:
  - this closes the reduced linear/quasilinear full-chain stellarator
    objective-gradient path from a real ``vmec_jax`` state coefficient through
    ``booz_xform_jax`` and the SPECTRAX-GK linear solver;
  - it still does **not** close full nonlinear-window heat-flux gradients or
    broad multi-equilibrium stellarator transport-gradient optimization.
- Current next best steps:
  - add a second VMEC/Boozer equilibrium to the quasilinear-gradient gate
    before making broad stellarator-transport-gradient claims;
  - connect the stellarator optimization examples to the passed full-chain
    linear/quasilinear gate while preserving the reduced/nonlinear claim
    boundary;
  - keep W7-X zonal recurrence/damping and TEM/kinetic-electron validation
    deferred for the current manuscript as previously agreed.
- Added memory-bounded Boozer surface-stencil support for large-equilibrium
  diagnostics:
  - ``vmec_jax_boozer_equal_arc_core_profiles_from_state`` now accepts
    ``surface_stencil_width`` and forwards a local radial stencil to
    ``booz_xform_jax`` when requested;
  - the VMEC/Boozer frequency and quasilinear gradient-gate tools expose the
    same option, while preserving the all-surface default used by the
    published QH artifact;
  - a mocked backend regression test checks that the selected Boozer surfaces
    are passed through and recorded in the returned metadata.
- Exposed ``radial_index``, ``mode_index``, and ``surface_index`` controls in
  the VMEC/Boozer frequency and quasilinear gradient-gate APIs/tools. This is
  needed for reviewer-proof conditioning scans because a broad VMEC/Boozer
  gradient claim should not depend on one hard-coded Fourier coefficient.
- Office GPU holdout diagnostics:
  - ``nfp3_QI_fixed_resolution_final`` now runs with the stencil and
    ``mboz=nboz=21`` without the earlier all-surface OOM, but both the
    frequency and quasilinear AD-vs-FD gates fail by order-unity relative
    errors. Enabling ``JAX_ENABLE_X64=1`` does not change the result, so this
    is a stencil/conditioning blocker rather than float32 finite-difference
    noise.
  - ``LandremanPaul2021_QA_lowres`` is a better small bundled QA holdout by
    radial count, but the all-surface Boozer transform still exceeds the
    available office GPU memory at ``mboz=nboz=21``. It should be retried only
    after a memory-reduced Boozer transform or bounded CPU/offline artifact
    path is available.
- Current manuscript stance:
  - reduced full-chain linear/quasilinear differentiability is closed for the
    tracked all-surface QH fixture;
  - multi-equilibrium transport-gradient promotion remains open;
  - nonlinear-window VMEC/Boozer gradients remain future work;
  - W7-X zonal and TEM remain explicitly deferred for this manuscript.
- Validation for this slice:
  - ``python -m ruff check src/spectraxgk/geometry/differentiable.py src/spectraxgk/solver_objective_gradients.py tools/build_vmec_boozer_quasilinear_gradient_gate.py tools/build_vmec_boozer_solver_frequency_gradient_gate.py tests/test_differentiable_geometry_bridge.py tests/test_solver_objective_gradients.py``
  - ``PYTHONPATH=src python -m mypy src/spectraxgk/geometry/differentiable.py src/spectraxgk/solver_objective_gradients.py tools/build_vmec_boozer_quasilinear_gradient_gate.py tools/build_vmec_boozer_solver_frequency_gradient_gate.py``
  - ``python -m pytest tests/test_differentiable_geometry_bridge.py::test_vmec_jax_boozer_equal_arc_core_profiles_supports_surface_stencil tests/test_solver_objective_gradients.py tests/test_build_manuscript_readiness_status.py -q``
  - ``PYTHONPATH=src python -m sphinx -b html -W docs docs/_build/html``
- Current next best steps:
  - build a memory-aware multi-equilibrium VMEC/Boozer gradient artifact path
    before broadening transport-gradient claims beyond QH;
  - continue quasilinear nonlinear-holdout validation and saturation-model
    calibration with explicit train/holdout splits;
  - strengthen the stellarator optimization examples by adding nonlinear
    audit bars for optimized geometries and finite-difference conditioning
    reports for each differentiated observable.
- CI coverage follow-up:
  - the public ``wide-coverage`` combine job failed at ``92%`` after the
    optional VMEC/Boozer artifact builders were added to the package-wide
    denominator, while the public CI cannot install or execute the local
    ``vmec_jax``/``booz_xform_jax`` repositories required by those paths;
  - scoped the external-backend artifact builders out of the default coverage
    denominator and documented that policy in ``docs/testing.rst``;
  - added fast low-level geometry helper tests and mocked optional-backend
    contract tests so the public suite still exercises radial interpolation,
    radial derivatives, periodic Boozer sampling, parity metrics, unavailable
    backend reporting, and solver-objective parameter naming;
  - local focused coverage for the affected modules is now ``96%`` for
    ``geometry/differentiable.py`` and ``100%`` for
    ``solver_objective_gradients.py`` on the bounded targeted test pair.
- Differentiable stellarator optimization UQ tightening:
  - while adding a paper-facing UQ/sensitivity panel, found that the previous
    optimization covariance estimate used the initial-to-final observable
    displacement as the residual, which measures optimizer travel rather than
    local uncertainty at the optimized point;
  - added ``stellarator_itg_objective_residual_vector`` and residual-name
    metadata so the covariance is now computed from the final weighted
    objective residual Jacobian, whose squared norm is the scalar objective;
  - added ``tools/plot_stellarator_optimization_uq.py`` and
    ``docs/_static/stellarator_itg_optimization_uq.{png,pdf,json}`` showing
    AD/FD derivative parity, local control uncertainty, covariance projection,
    and rank/conditioning diagnostics for the growth, quasilinear, and
    nonlinear-window reduced objectives;
  - updated README, manuscript-figure docs, and stellarator-optimization docs
    to keep the claim scoped to reduced optimization/UQ plumbing until
    production VMEC/Boozer/GK nonlinear audits pass.
- Quasilinear dataset-sufficiency promotion gate:
  - added ``tools/plot_quasilinear_dataset_sufficiency.py`` and
    ``docs/_static/quasilinear_dataset_sufficiency.{png,pdf,json}``;
  - the gate validates that every current quasilinear model-development input
    maps to a passed nonlinear summary gate, records the four
    electrostatic-compatible cases, and explicitly lists KBM as validated but
    excluded because the present quasilinear diagnostics are electrostatic;
  - the tracked promotion gate is intentionally blocked: current total cases
    are ``4`` versus a ``6``-case minimum, explicit training geometries are
    ``1`` versus a ``2``-geometry minimum, and downstream candidate skill gates
    remain false;
  - the figure and JSON now prevent the docs/README from accidentally
    promoting richer quasilinear absolute-flux candidates before more
    converged nonlinear holdouts exist.
- Multi-equilibrium VMEC/Boozer solver-objective gradient holdout:
  - added Li383 low-resolution holdouts for both the mode-21 full-chain
    eigenfrequency gate and the full-chain quasilinear heat-flux-weight gate;
  - both Li383 gates use ``mboz=nboz=21`` and pass the same AD-vs-central-
    finite-difference objective checks as the tracked QH fixture;
  - added ``tools/build_vmec_boozer_gradient_holdout_matrix.py`` and
    ``docs/_static/vmec_boozer_gradient_holdout_matrix.{png,pdf,json,csv}``
    to summarize QH plus Li383 frequency/quasilinear gate status;
  - the matrix closes the reduced multi-equilibrium linear/quasilinear
    VMEC/Boozer gradient gate with maximum relative mismatch ``4.9e-3``;
  - this is still not a nonlinear-window heat-flux gradient or broad
    optimized-equilibrium nonlinear transport claim, so those remain future
    promotion gates.
- Profiler-backed nonlinear performance artifact tightening:
  - added a JSON companion for the nonlinear RHS split profile so kernel
    fractions, dominant kernels, and grid-to-spectral speedups are
    machine-readable rather than only plotted;
  - after the zero-collision fast path, the current tracked Cyclone profile
    reports GPU spectral as the fastest full-RHS row, with grid/spectral
    full-RHS ratio ``1.66`` on GPU and ``1.11`` on CPU;
  - the CPU and GPU spectral brackets are faster in this short bounded
    harness, but the claim remains scoped to the tracked Cyclone profiler case
    until larger matched runtime/memory sweeps and case-level parity gates
    support broader defaults;
  - this supports the existing scoped performance stance: spectral nonlinear
    mode is a validated opt-in engineering mode for selected cases, not a
    global default or broad runtime claim.
- Linear RHS term-profile follow-up:
  - fixed the standalone linear RHS profiler default config path and added a
    JSON companion artifact so term-level timings, dominant nonzero terms, and
    zero-norm initial-state rows are machine-readable;
  - regenerated ``docs/_static/linear_rhs_terms_profile_cpu.csv`` and
    ``docs/_static/linear_rhs_terms_profile.json`` for the Cyclone nonlinear
    runtime state at ``ky=0.3, Nl=4, Nm=8``;
  - fixed the standalone linked-``|k_z|`` profiling source to match the
    production hypercollision formula
    ``nu_hyper_m * m_norm_kz_factor * 2.3 * vth * |kpar_scale|`` rather than
    the unrelated ``nu_hyper`` path;
  - after the zero-collision fast path, the current bounded CPU profile records
    ``full_linear_rhs≈5.04e-2 s`` in the profiler harness and independently
    measured term kernels summing to ``1.71e-2 s``;
  - added the active-state CPU profile
    ``docs/_static/linear_rhs_terms_profile_z_wave_cpu.{csv,json}``, where
    resolved parallel variation activates hypercollisions and linked
    ``|k_z|`` with norm ``2.35e-4`` and linked-operator cost ``2.09e-3 s``;
  - refreshed the matching ``office`` GPU profiles after the profiler-source
    fix: the initial-state artifact records ``full_linear_rhs≈6.52e-3 s`` and
    measured terms summing to ``3.94e-3 s``, while
    ``docs/_static/linear_rhs_terms_profile_z_wave_gpu.{csv,json}`` activates
    the same linked ``|k_z|``/hypercollision pair with matched norm
    ``2.35e-4`` and linked-operator cost ``3.62e-4 s``;
  - keep this as localization evidence only: zero-norm rows must not be skipped
    in production until a state-window identity gate shows they remain inactive
    after nonlinear evolution.
  - added a linked-``|k_z|`` hypercollision guardrail test showing the profiled
    zero norm is an initial-state property: constant-in-``z`` states remain
    zero, while resolved ``z``-varying states activate the term when
    ``hypercollisions_kz`` is nonzero.
  - fixed the term-resolved diagnostic assembly total so active
    perpendicular hyperdiffusion is included in ``assemble_rhs_terms_cached``;
    added the regression by enabling ``D_hyper`` and ``hyperdiffusion`` in the
    total-vs-term-sum test.
  - refactored the field solve so electrostatic runs avoid the electromagnetic
    ``A_parallel``/``B_parallel`` branches when ``beta`` or the field toggles
    disable them; this is a correctness/performance guard rather than a new
    speedup claim, and it adds a regression showing electrostatic ``Nm=1``
    states no longer need the absent ``m=1`` moment.
  - added ``tools/gate_linear_rhs_zero_norm_state_window.py`` and
    ``docs/_static/linear_rhs_zero_norm_state_window_gate.json``;
  - the gate directly compares the full RHS with candidate skip configurations
    over initial, linear-kick, ``z``-wave, and ``z``-wave-kick states;
  - current result: zero-collision skip is identity-safe for this ``nu=0``
    Cyclone window, while skipping hypercollisions is correctly rejected with a
    maximum relative RHS error of ``3.59e-3`` on the resolved ``z``-varying
    state.
  - implemented the accepted zero-collision fast path by disabling runtime
    collision terms when every species has ``nu=0`` and returning early from
    the low-rank collision contribution for static zero-weight/zero-``nu``
    cases while preserving pre-expanded collision matrices.
- Repository/release artifact hygiene audit:
  - reran ``tools/audit_repository_size.py --top 30`` and the checked
    repository/release artifact manifests after the profiler refreshes;
  - current tracked size is ``39.55 MB`` across ``913`` tracked files, below
    the ``45 MB`` gate, with no unlisted large tracked files;
  - release-artifact provenance passes for ``10`` tracked assets, with
    ``2.07 MB`` still planned for GitHub Releases and the HSX VMEC fixture
    intentionally kept in the repo.
- Readiness dashboard profiler-source tightening:
  - updated the manuscript-readiness and open-research-lane dashboards to
    consume ``docs/_static/nonlinear_rhs_profile.json`` alongside the
    nonlinear sharding identity artifact;
  - the dashboard JSON/CSV/PNG/PDF artifacts now expose the current RHS split
    profile metrics, including GPU full-RHS grid/spectral ratio ``1.66``, GPU
    nonlinear-bracket ratio ``2.25``, CPU full-RHS ratio ``1.11``, and CPU
    nonlinear-bracket ratio ``1.66``;
  - the profiler lanes intentionally remain ``partial`` because these are
    bounded split-profile and identity artifacts, not broad nonlinear runtime
    claims.
- Spectral Laguerre nonlinear-mode gate refresh:
  - regenerated the local CPU ``docs/_static/laguerre_mode_gate.{png,pdf,json,csv}``
    artifact across Cyclone, KBM, W7-X, and HSX with a bounded two-step paired
    grid/spectral comparison;
  - all four cases pass the scalar-diagnostic parity threshold, with maximum
    relative differences ``8.9e-4`` (Cyclone), ``0`` (KBM), ``5.1e-6``
    (W7-X), and ``2.2e-5`` (HSX);
  - grid/spectral runtime ratios are ``2.90`` (Cyclone), ``3.31`` (KBM),
    ``1.67`` (W7-X), and ``0.66`` (HSX), so the documentation keeps spectral
    Laguerre mode as opt-in rather than a global default.
- CI-backed readiness and claim-scope update:
  - latest public CI for commit ``5790e0e`` passed repo hygiene, mypy, quick
    shards, docs/packaging, fast coverage, and all 24 wide-coverage shards;
  - the wide-coverage combine job reported ``TOTAL 16134 787 95%`` package-wide
    coverage, so the current package-wide 95% lane is green even though
    targeted follow-up remains valuable for modules below the threshold such as
    ``nonlinear.py`` and ``zonal_validation.py``;
  - updated ``docs/roadmap.rst``, ``docs/manuscript_figures.rst``, and
    ``README.md`` so the claim surface matches the artifacts: quasilinear
    diagnostics/model-selection, nonlinear-window release gates, VMEC/Boozer
    parity, reduced stellarator ITG optimization/UQ, and QH+Li383
    linear/quasilinear VMEC/Boozer gradient gates are ready for the scoped
    manuscript/release narrative;
  - explicitly kept calibrated absolute quasilinear flux prediction, production
    nonlinear heat-flux stellarator optimization, W7-X zonal recurrence, and
    TEM/kinetic-electron stellarator validation out of the finished claim
    surface;
  - next highest-value blockers before stronger claims are at least two more
    converged electrostatic nonlinear holdouts, a nonlinear-window
    VMEC/Boozer state-gradient gate, nonlinear audits of optimized equilibria,
    and the deferred W7-X zonal/TEM lanes.
- Wide-coverage CI rebalance:
  - commit ``5f16b11`` exposed that wide-coverage shard ``3/24`` was too close
    to the bounded test budget: the tests passed but the coverage command timed
    out after ``300`` seconds during shutdown/reporting;
  - kept the per-shard timeout at ``300`` seconds and split the matrix into
    ``48`` shards instead of increasing the timeout;
  - raised the wide-coverage matrix concurrency from ``2`` to ``8`` so the
    safer shard split does not make the package-wide gate slower overall;
  - locally checked the two new heavy descendants under coverage: shard
    ``3/48`` passed in about ``74`` seconds and shard ``27/48`` passed in about
    ``46`` seconds;
  - confirmed on GitHub CI at commit ``6dbfddb``: all quick shards,
    docs/packaging, fast coverage, all ``48`` wide-coverage shards, and the
    final wide-coverage combine passed with ``TOTAL 16134 787 95%``.
- Nonlinear startup-window finite-difference plumbing audit:
  - added ``tools/build_nonlinear_window_fd_audit.py`` and
    ``docs/_static/nonlinear_window_fd_audit.{png,pdf,json,csv}``;
  - the tool runs actual compact SPECTRAX-GK nonlinear Cyclone startup windows at
    ``R/LTi = base +/- step`` plus a repeated base point and checks finite
    outputs, exact repeatability for the deterministic repeated run, monotonic
    drive response, startup-window coefficient of variation, startup-window trend,
    and resolved central finite-difference response;
  - current artifact passes the startup plumbing gate with response/base
    ``0.1109``, repeatability relative error ``0``, maximum startup-window
    coefficient of variation ``0.095``, and maximum normalized window trend
    ``0.313``;
  - correction after heat-flux QA: the run reaches only ``t <= 0.64`` and is not
    a post-transient transport average; the artifact is now wired as
    ``startup_nonlinear_plumbing_fd_path_gate = true`` with
    ``transport_average_gate = false`` and
    ``production_nonlinear_observable_fd_path_gate = false``;
  - this intentionally does not claim a production nonlinear heat-flux average,
    VMEC/Boozer nonlinear state-gradient, converged nonlinear turbulence
    gradient, or optimized-equilibrium nonlinear heat-flux optimization result.
- Next promotion steps:
  - replace the startup FD audit with long post-transient transport windows that
    discard the initial transient, retain enough late samples, pass cumulative
    running-mean and block-window stability gates, and compare the same window
    against tracked nonlinear reference cases;
  - run converged nonlinear-window audits on optimized-equilibrium candidates
    before any production nonlinear stellarator-transport optimization claim;
  - keep W7-X zonal recurrence and TEM/kinetic-electron stellarator lanes
    deferred from the current manuscript scope unless they are reopened
    explicitly.
- VMEC/Boozer-perturbed nonlinear startup-window FD audit:
  - added ``tools/build_vmec_boozer_nonlinear_window_fd_audit.py`` and
    ``docs/_static/vmec_boozer_nonlinear_window_fd_audit.{png,pdf,json,csv}``;
  - the tool starts from the real mode-21
    ``vmec_jax -> booz_xform_jax`` QH state bridge, perturbs
    ``Rcos_mid_surface_m1`` by ``+/- 1e-5``, writes temporary sampled-geometry
    NetCDF files, and runs compact nonlinear startup windows with
    ``dt=0.002`` and ``16`` fixed RK2 steps;
  - current artifact passes the startup plumbing gate with finite outputs,
    deterministic repeated-base agreement, maximum window CV ``0.101``,
    maximum normalized trend ``0.286``, resolved central response/base
    ``0.0401``, and resolved geometry perturbation;
  - correction after heat-flux QA: the run reaches only ``t <= 0.032`` and is
    not a post-transient transport average; the artifact is now wired as
    ``vmec_boozer_startup_nonlinear_plumbing_fd_path_gate = true`` with
    ``transport_average_gate = false`` and
    ``vmec_boozer_production_nonlinear_observable_fd_path_gate = false``;
  - the forward/backward response is asymmetric and not monotone, so the
    artifact is explicitly scoped as a VMEC/Boozer geometry-perturbed
    nonlinear startup observable audit, not as a local nonlinear gradient,
    transport average, optimized-equilibrium audit, or production heat-flux
    optimization claim.
- Next promotion steps:
  - turn the VMEC/Boozer nonlinear FD audit into a local-gradient conditioning
    gate only after a better-conditioned perturbation basis and longer converged
    post-transient running-average nonlinear windows are available;
  - add optimized-equilibrium nonlinear-window audits before any production
    nonlinear stellarator optimization claim;
  - keep the current reduced optimization figures scoped to objective-plumbing
    and UQ validation.
- Nonlinear transport time-horizon QA and QH long-window extension:
  - audited the earlier nonlinear artifacts after the heat-flux scale concern:
    the compact Cyclone FD audit reaches only ``t <= 0.64`` and the VMEC/Boozer
    FD audit reaches only ``t <= 0.032``, so both remain startup plumbing checks;
  - identified the nfp4 QH external-VMEC nonlinear pilot as the earlier case that
    genuinely needed a longer simulation rather than only demotion: the original
    ``t = 20`` run was finite but still startup/growth scale with final heat flux
    ``3.58e-4``;
  - ran a targeted reduced-grid QH extension locally at ``Nx = Ny = 32``,
    ``Nz = 24``, ``Nl = 4``, ``Nm = 8``, ``dt = 0.05`` to ``t = 100`` and then
    continued from restart to ``t = 150``;
  - the ``t = 150`` trace is finite and reaches a meaningful late heat-flux
    level: the least-trending window is ``t = 77.55..150.00`` with mean heat
    flux ``19.64``, standard deviation ``1.14``, and relative trend
    ``-3.25e-4`` per time unit;
  - generated
    ``docs/_static/external_vmec_qh_nonlinear_t150_pilot.{png,pdf,json,traces.csv}``
    and ``docs/_static/nonlinear_transport_time_horizon_audit.{png,pdf,json,csv}``;
  - the audit separates release transport gates, long feasibility pilots,
    failed convergence results, startup plumbing audits, and reduced-envelope
    estimators so startup/noise-floor heat fluxes cannot be confused with
    post-transient nonlinear transport averages;
  - QH is now a useful long-window feasibility candidate but remains outside
    quasilinear calibration until a grid/window convergence gate passes.
- QH external-VMEC grid/window convergence check:
  - ran a higher-grid QH companion on office GPU at ``Nx = Ny = 48``,
    ``Nz = 32``, ``Nl = 4``, ``Nm = 8``, ``dt = 0.05`` to ``t = 150``;
  - the run is finite with a flat late trace, but the late heat-flux level is
    not grid converged: the common-window means are ``19.76`` and ``11.56``,
    while the least-trending-window means are ``19.64`` and ``12.03``;
  - generated
    ``docs/_static/external_vmec_qh_nonlinear_t150_n48_pilot.{png,pdf,json,traces.csv}``
    and
    ``docs/_static/external_vmec_qh_grid_convergence_gate.{png,pdf,json,csv}``;
  - the QH convergence gate fails with common-window symmetric relative
    heat-flux difference ``0.523`` and least-window difference ``0.480`` against
    the ``0.15`` gate, plus a marginal common-window trend excess
    ``2.29e-3`` against ``2.0e-3``;
  - conclusion: QH now joins CTH-like as a finite long external-VMEC feasibility
    and negative convergence result, not a quasilinear calibration holdout.
- QH high-grid follow-up:
  - ran a bounded office-GPU companion at ``Nx = Ny = 64``, ``Nz = 40``,
    ``Nl = 4``, ``Nm = 8``, ``dt = 0.05`` to ``t = 150``; wall time was
    about ``423 s`` and the run stayed finite with final ``Wg = 136.86`` and
    ``Wphi = 2.20``;
  - generated
    ``docs/_static/external_vmec_qh_nonlinear_t150_n64_pilot.{png,pdf,json,traces.csv}``
    and
    ``docs/_static/external_vmec_qh_high_grid_convergence_gate.{png,pdf,json,csv}``;
  - the ``48x48x32`` to ``64x64x40`` gate also fails: common-window means are
    ``11.58`` and ``6.03``, least-window means are ``12.03`` and ``5.76``, and
    the symmetric relative differences are ``0.630`` and ``0.704``;
  - conclusion: QH is not a near-term nonlinear calibration holdout under this
    grid/window/hypercollision protocol. The next quasilinear-calibration step
    should prioritize another holdout candidate or a redesigned QH convergence
    campaign rather than promoting this result.
- Broader external-VMEC convergence correction and DSHAPE holdout:
  - after QH/CTH-like grid failures, ran a five-point linear candidate screen
    over additional ``vmec_jax`` equilibria at ``Nx = Ny = 48``, ``Nz = 32``,
    ``Nl = 4``, ``Nm = 8`` and ``ky = 0.095..0.476``;
  - strongest finite unstable candidates were DSHAPE
    (``gamma = 0.096`` at ``ky = 0.476``), circular tokamak
    (``gamma = 0.089``), and ITER-model (``gamma = 0.089``); QI/QA/QH
    reference fixtures were stable or failed the current geometry screen;
  - ran DSHAPE nonlinear ``t = 150`` pilots at ``32x32x24``, ``48x48x32``,
    and ``64x64x40``; the low-to-mid gate passed but the ``t = 150`` mid-to-high
    gate was close and failed (common-window difference ``0.201`` and
    least-window difference ``0.262``);
  - extended ``48x48x32`` and ``64x64x40`` DSHAPE from restart to ``t = 250``;
    the high-grid gate then passed with common-window means ``16.07`` and
    ``18.48`` (relative difference ``0.139``) and least-window means ``15.86``
    and ``17.66`` (relative difference ``0.108``);
  - generated
    ``docs/_static/quasilinear_vmec_dshape_linear_spectrum.{png,pdf,json}``,
    ``docs/_static/external_vmec_dshape_*`` nonlinear pilot/gate artifacts, and
    ``docs/_static/external_vmec_candidate_linear_screen.csv``;
  - conclusion: DSHAPE is the first external-VMEC nonlinear transport holdout
    candidate from this correction campaign that passes time-window and
    high-grid convergence. Next step is calibration-report admission and a
    quasilinear model gate; QH and CTH-like remain excluded.
- Admitted the converged DSHAPE holdout into the quasilinear calibration gate:
  - added ``docs/_static/external_vmec_dshape_t250_n64_transport_window.json``
    as the high-grid common-window summary tied to the passed ``t = 250``
    convergence gate;
  - appended ``dshape_external_vmec_t250_window`` to
    ``docs/_static/quasilinear_stellarator_train_holdout_points.json`` and
    regenerated ``docs/_static/quasilinear_stellarator_train_holdout.*`` plus
    ``docs/_static/quasilinear_validated_calibration_inputs.*``;
  - input provenance still passes: all train/holdout nonlinear artifacts now map
    to passed gates, including the admitted DSHAPE external-VMEC CSV, while QH
    and CTH-like remain excluded;
  - result: the one-constant Cyclone-trained mixing-length absolute-flux model
    remains rejected. DSHAPE has observed late-window heat flux about ``18.5``
    but a scaled estimate about ``3.49e3``, giving a stronger negative transfer
    constraint for the next saturation-model development step.
  - regenerated the saturation-rule and dataset-sufficiency diagnostics with
    DSHAPE included as a fourth holdout. The best one-scalar rule is still raw
    linear weight with holdout mean relative error about ``21``; the
    training-mean null is about ``0.439``. Dataset sufficiency is improved to
    five validated electrostatic cases but still fails the six-case and
    two-training-geometry promotion requirements.
- Resumed the next external-VMEC nonlinear holdout lane after interruption:
  - current local branch is clean at ``742cc93`` and GitHub CI for that commit
    passed;
  - the office host timed out during artifact inspection, so no new remote runs
    were launched blindly;
  - the interrupted circular-tokamak attempt should be treated as a negative
    or pending convergence artifact until its remote JSON/CSV files are pulled:
    existing notes show finite ``t = 150`` and ``t = 250`` runs, but the
    ``t = 250`` high-grid heat-flux means differed by about ``18%`` on the
    common window and about ``31%`` on the least-trending window, failing the
    ``0.15`` high-grid promotion envelope;
  - added ``tools/write_external_vmec_holdout_configs.py`` and tests so the
    next candidates use a reproducible two-grid ``t = 150`` plus restart
    ``t = 250`` ladder instead of ad hoc TOMLs;
  - documented that generator in ``docs/testing.rst`` and
    ``docs/quasilinear.rst``.
- Next best steps:
  - when office is reachable, pull the compact circular JSON/CSV/PNG gate
    artifacts, record circular as a failed high-grid convergence candidate, and
    keep it out of calibration;
  - use the new selector to launch the next finite unstable candidate.
    Current dry-run output picks ``ITERModel_reference_nc`` at ``ky = 0.4762``
    and resolves
    ``/Users/rogeriojorge/local/vmec_jax/examples_single_grid/data/wout_ITERModel_reference.nc``
    into the standard ``48x48x32`` and ``64x64x40`` ``t = 150`` plus restart
    ``t = 250`` ladder;
  - admit a new candidate only if the high-grid gate passes, then regenerate
    quasilinear train/holdout, saturation-rule, dataset-sufficiency, and
    manuscript-readiness panels.
- Closed the next external-VMEC holdout tranche:
  - pulled the circular-tokamak ``t = 150`` and ``t = 250`` gate artifacts
    back into the repo and kept circular excluded. It is now a documented
    negative convergence result: ``t = 150`` had good grid agreement but failed
    trend gates, while ``t = 250`` removed the trend defect but failed common
    and least-window grid/CV gates;
  - launched the next screened unstable axisymmetric external-VMEC candidate,
    ``ITERModel_reference_nc`` at ``ky = 0.4762``, with the standard
    ``48x48x32`` and ``64x64x40`` ITG/adiabatic-electron ladder on office GPUs;
  - found and fixed a real reproducibility bug in
    ``tools/write_external_vmec_holdout_configs.py``: generated
    ``output.path`` entries were being interpreted relative to the TOML
    location, causing nested duplicated output directories. Added a regression
    test and propagated the fix to office before restart continuations;
  - found and corrected the continuation seeding bug operationally: the
    ``restart_if_exists`` path keys off the current output basename, so the
    restart files must be copied from ``t150`` to ``t250`` and from ``t250`` to
    ``t350`` stems before launching the bounded continuation;
  - ``ITERModel_reference`` was a genuine extend-not-reject case:
    ``t = 150`` passed the common-window grid-difference gate but missed the
    common trend and least-window grid-difference gates; ``t = 250`` passed all
    grid-difference checks but still missed the common trend/CV gates by a
    small margin; ``t = 350`` finally passed every gate;
  - final ``t = 350`` result:
    common-window heat-flux means ``22.41`` and ``22.05``,
    common-window relative difference ``0.0165``,
    least-window relative difference ``0.1415``,
    common trend ``0.00160`` per time unit,
    least-window max trend ``2.92e-4`` per time unit,
    common-window max CV ``0.176``;
  - generated and tracked compact artifacts:
    ``docs/_static/quasilinear_vmec_itermodel_linear_spectrum.*``,
    ``docs/_static/external_vmec_itermodel_t150_high_grid_convergence_gate.*``,
    ``docs/_static/external_vmec_itermodel_t250_high_grid_convergence_gate.*``,
    ``docs/_static/external_vmec_itermodel_t350_high_grid_convergence_gate.*``,
    ``docs/_static/external_vmec_itermodel_nonlinear_t350_n48_pilot.{json,png}``,
    ``docs/_static/external_vmec_itermodel_nonlinear_t350_n64_pilot.{json,png,traces.csv}``,
    and ``docs/_static/external_vmec_itermodel_t350_n64_transport_window.json``;
  - admitted ``itermodel_external_vmec_t350_window`` into
    ``docs/_static/quasilinear_stellarator_train_holdout_points.json`` as a
    second training geometry and regenerated the tracked quasilinear holdout,
    input-validation, saturation-rule, dataset-sufficiency, nonlinear-horizon,
    open-lane, and manuscript-readiness artifacts;
  - the important scientific change is that the dataset-sufficiency gate is now
    blocked only by downstream model-skill gates, not by insufficient validated
    nonlinear data volume. The one-constant mixing-length model remains
    rejected on held-out skill, which is the correct remaining blocker.
- Closed a third external-VMEC nonlinear transport holdout:
  - expanded the office linear candidate screen beyond the original shortlist
    and found a finite unstable branch for
    ``wout_up_down_asymmetric_tokamak_reference.nc`` at ``ky = 0.4762`` with
    ``gamma ≈ 0.0328``;
  - the initial ``t = 150`` high-grid pair was finite, but the gate remained
    open because only the common-window trend failed
    (``7.32e-3`` per time unit versus the ``2e-3`` threshold) even though the
    grid difference already passed (``0.138``);
  - extending the same ``48x48x32`` and ``64x64x40`` runs to ``t = 250`` and
    then ``t = 350`` steadily reduced the common-window trend to
    ``2.78e-3`` and ``2.18e-3`` while all CV, sample-count, and pairwise-grid
    gates stayed passed;
  - one final bounded extension to ``t = 450`` closed the gate cleanly with
    common-window relative slope ``7.48e-4``, least-window slope
    ``1.26e-4``, common-window relative difference ``0.0435``, and
    least-window relative difference ``0.0242``;
  - tracked compact artifacts:
    ``docs/_static/quasilinear_vmec_updown_asym_linear_spectrum.*``,
    ``docs/_static/external_vmec_updown_asym_t450_high_grid_convergence_gate.*``,
    ``docs/_static/external_vmec_updown_asym_nonlinear_t450_n48_pilot.{json,png}``,
    ``docs/_static/external_vmec_updown_asym_nonlinear_t450_n64_pilot.{json,png,traces.csv}``,
    and ``docs/_static/external_vmec_updown_asym_t450_n64_transport_window.json``;
  - admitted ``updown_asym_external_vmec_t450_window`` into
    ``docs/_static/quasilinear_stellarator_train_holdout_points.json`` as a new
    validated holdout and refreshed the open-lane, horizon-audit, validated-input,
    and saturation-rule artifacts accordingly;
  - the quasilinear skill gate remains the downstream blocker: validated
    nonlinear data volume is stronger now, but the current one-constant
    mixing-length model is still not good enough to promote as an absolute-flux
    predictor.
- Finalized the publication-facing quasilinear documentation refresh:
  - updated ``README.md`` to promote the combined stellarator/external-VMEC
    train/holdout figure plus the ITERModel and up-down asymmetric convergence
    gates instead of the older Cyclone-only calibration narrative;
  - refreshed ``docs/_static/quasilinear_dataset_sufficiency.*`` so the figure
    now reflects the admitted seven-case electrostatic portfolio
    (``2`` training geometries, ``5`` holdouts);
  - updated ``docs/quasilinear.rst`` to match the current claim surface:
    absolute-flux promotion is blocked by held-out model skill, not by missing
    validated nonlinear input volume.
- Closed the richer quasilinear held-out candidate lane on the current
  electrostatic portfolio:
  - audited the leave-one-geometry-out candidate space against the admitted
    seven-case dataset and identified a small spectrum-envelope model as the
    first candidate that simultaneously beats the training-mean null baseline,
    beats the calibrated linear-weight baseline, satisfies the ``0.35``
    transport gate, and clears the interval-coverage gate;
  - implemented ``spectral_envelope_ridge`` in
    ``tools/plot_quasilinear_candidate_uncertainty.py`` using only two
    physically legible linear-spectrum features: the positive-growth
    ``k_y`` centroid and the heat-flux-weighted ``k_y`` width;
  - regenerated
    ``docs/_static/quasilinear_candidate_uncertainty.*``,
    ``docs/_static/quasilinear_dataset_sufficiency.*``,
    ``docs/_static/open_research_lane_status.*``, and
    ``docs/_static/manuscript_readiness_status.*``;
  - current gated result:
    ``spectral_envelope_ridge`` reaches leave-one-geometry-out mean relative
    error ``0.244``, interval coverage ``0.857``, and becomes the accepted
    richer candidate on the present seven-case electrostatic portfolio;
  - the scientific scope remains explicit:
    one-scalar saturation rules stay rejected, the higher-parameter
    ``linear_state_ridge`` candidate remains blocked by the
    train-to-parameter-ratio gate, and electromagnetic/KBM quasilinear
    promotion remains separate future work.
- Tightened the reproducibility path for the remaining W7-X zonal blocker:
  - office was unreachable during this pass, so no new long zonal runs were
    launched blindly;
  - added ``tools/write_w7x_zonal_closure_sweep.py`` plus focused tests to
    write one manifest for the next paper-facing ``k_x rho_i = 0.07`` operator
    sweep;
  - the manifest separates closure families one knob at a time:
    baseline, constant-Hermite, ``|k_z|``-weighted Hermite, mixed
    Laguerre-Hermite, Laguerre-only, and isotropic hypercollision variants;
  - the JSON output carries both the exact
    ``tools/generate_w7x_zonal_response_panel.py`` launch commands and the
    matching ``tools/plot_w7x_zonal_closure_ladder.py`` refresh command, so the
    next office window can be used entirely for bounded physics runs instead of
    ad hoc setup.
- Resumed the bounded W7-X zonal closure sweep on ``office`` and closed the
  first operator tranche as a tracked negative result:
  - syncing the current zonal driver onto ``office`` surfaced a real runtime
    bug in the paper-facing zonal initializer path: ``build_linear_cache`` used
    ``shat`` inside the non-twist FFT branch even when no twist-shift boundary
    had defined it;
  - fixed ``src/spectraxgk/linear.py`` so the non-twist branch uses
    ``geom_data.s_hat`` directly and added a regression in
    ``tests/test_linear_helpers_extra.py`` covering periodic non-twist cache
    construction;
  - after syncing the fixed package to ``office``, completed a bounded
    ``k_x rho_i = 0.07``, ``Nl=16``, ``Nm=64``, ``dt=0.05``, ``t_max=100``
    closure tranche against the digitized W7-X reference and stored the ladder
    summary in
    ``tools_out/zonal_response/closure_sweep_manifest/w7x_zonal_closure_ladder_partial.{json,csv,png}``;
  - outcome of the first physically distinct closure families:
    baseline MAE ``0.2861`` with late-window ``tail_std`` ``0.1131`` and
    Hermite-tail fraction ``0.388``;
    ``|k_z|``-weighted Hermite hypercollision reduces the Hermite tail
    (down to ``0.282`` at ``nu_hyper_m=0.03``) but worsens trace error and
    late-window variability;
    mixed Laguerre-Hermite hypercollision improves MAE slightly
    (``0.2755`` at ``nu_hyper_lm=0.01``/``0.03``) and nearly eliminates the
    Hermite tail, but it also increases late-window variability to about
    ``0.117``;
  - current conclusion remains unchanged at manuscript scope:
    no tested physical closure in this bounded tranche improves the paper-facing
    W7-X zonal trace and the late-time recurrence metric together, so the lane
    stays open as a physics blocker rather than a normalization or runtime bug.
- Extended the W7-X mixed Laguerre-Hermite closure lane with a bounded
  moment-resolution audit:
  - the first ``Nl=24, Nm=96`` and ``Nl=32, Nm=128`` retries at ``dt=0.05``
    failed with non-finite ``Wg_t`` diagnostics, which exposed a second
    numerical issue rather than a clean physics conclusion;
  - fixed the high-order Hermite hypercollision cache representation in
    ``src/spectraxgk/linear.py`` by storing the ``k_z`` Hermite factor in a
    normalized finite form and moved the zero-weight early return ahead of the
    ``k_z`` contribution assembly in ``src/spectraxgk/terms/linear_terms.py``;
    mirrored the same safe algebra in
    ``tools/profile_linear_rhs_terms.py`` and added a regression in
    ``tests/test_linear_helpers_extra.py`` for ``Nm=128``, ``p_hyper_m=20``;
  - with that fix in place, the mixed ``LM`` closure at ``Nl=24, Nm=96``
    remains stable when the time step is reduced to ``dt=0.025`` out to
    ``t_max=100`` and yields
    ``MAE=0.2768``, ``tail_std=0.1127``, and
    ``tail_std_ratio=4.11`` versus the digitized reference;
  - compared with the baseline mixed ``LM`` run
    (``Nl=16, Nm=64``, ``dt=0.05``:
    ``MAE=0.2753``, ``tail_std=0.1162``, ``tail_std_ratio=4.24``),
    higher moment resolution reduces the late-window variability modestly and
    further suppresses Hermite/Laguerre tail fractions, but it does not improve
    the paper-facing trace error;
  - the more aggressive ``Nl=32, Nm=128`` point still goes non-finite even at
    ``dt=0.025`` (failure around ``t≈10``), so the current mixed ``LM``
    closure is not robust under further moment refinement;
  - current W7-X zonal interpretation is therefore sharper:
    part of the earlier recurrence growth was a time-step limitation at higher
    moments, but even after removing that numerical artifact the closure family
    still does not close the paper-facing trace mismatch, and its stability
    margin degrades as moments increase.
- Audited and refreshed the repository-facing status layer after the latest
  W7-X zonal runs:
  - promoted the compact mixed Laguerre-Hermite resolution artifact to
    ``docs/_static/w7x_zonal_mixedlm_resolution_kx070.{json,csv,png}`` and
    linked it from ``docs/testing.rst``;
  - updated ``tools/build_open_research_lane_status.py`` so the W7-X zonal lane
    records both the lowest-error mixed-``LM`` row and the highest stable
    moment-resolution row; the open-lane dashboard now preserves the
    ``Nl=24,Nm=96,dt=0.025`` evidence instead of only the older constant
    hypercollision probes;
  - regenerated ``docs/_static/open_research_lane_status.{json,csv,png,pdf}``;
  - the current plan state remains: quasilinear candidate-model selection is
    closed for the scoped electrostatic portfolio; manuscript readiness remains
    mostly closed with W7-X zonal and TEM deferred by scope; the broader
    research status still has one open physics lane (W7-X zonal) plus partial
    W7-X fluctuation/TEM, differentiable-geometry, and profiler-backed
    performance lanes.
- Completed a repository-hygiene follow-up after the audit showed only
  ``289 kB`` of tracked-size headroom under the ``50 MB`` CI gate:
  - compressed large checked-in documentation PNG previews to lightweight
    ``1800 px``/``192``-color previews while leaving JSON/CSV evidence, raw
    local ``tools_out`` outputs, and release-manifest-pinned previews
    untouched;
  - tracked size dropped from ``49.71 MB`` to about ``40.16 MB``, restoring
    enough headroom for near-term validation-dashboard updates without raising
    the CI size cap;
  - added ``tools/compress_docs_previews.py`` and a regression test so future
    figure additions have a repeatable cleanup path that skips
    ``tools/release_artifact_manifest.toml`` entries by default;
  - updated the artifact-hygiene documentation to distinguish release-manifest
    preview compression from ordinary docs-preview compression.
- Migrated the current high-resolution PDF companions out of Git and into the
  ``v1.5.0`` GitHub release:
  - uploaded ``benchmark_readme_panel.pdf``,
    ``benchmark_core_nonlinear_atlas.pdf``, and
    ``gx_publication_panel.pdf`` as immutable release assets;
  - recorded ``release_tag``/``release_url`` in
    ``tools/release_artifact_manifest.toml`` while preserving each original
    byte count and SHA-256 checksum;
  - updated ``tools/check_release_artifact_manifest.py`` so migrated
    ``move_to_release`` entries can be absent from Git only when the manifest
    carries release provenance;
  - removed the PDFs from the tracked tree and removed the temporary
    repository-size whitelist for ``benchmark_readme_panel.pdf``.
- Refreshed the local CPU nonlinear RHS split profile for the performance lane:
  - reran ``tools/profile_nonlinear_step_split.py`` sequentially for Cyclone
    grid and spectral Laguerre nonlinear modes with ``20`` repeats to avoid
    local profiler contention;
  - regenerated ``docs/_static/nonlinear_rhs_profile.{csv,json,png,pdf}``
    companions through ``tools/plot_nonlinear_rhs_profile.py``;
  - current bounded CPU result is a modest full-RHS spectral ratio
    (``grid/spectral = 1.03``) but a clearer nonlinear-bracket ratio
    (``1.49``), so the performance conclusion remains scoped: spectral
    Laguerre mode is a gated opt-in path, not a global default, and the linear
    RHS remains the dominant warm-throughput target.
- Tightened release-facing nonlinear terminology:
  - removed stale source and testing-doc wording that described the nonlinear
    path as a placeholder even though the pseudo-spectral E×B/electromagnetic
    bracket is implemented and tested;
  - kept the explicit zero-output helper named as a shape-only/disabled-term
    test utility so source comments match the implemented physics path.
- Refreshed the matched short-harness nonlinear RHS profile on CPU and the
  ``office`` GPU:
  - used a fresh detached ``de826d8`` clone on ``office`` to avoid dirty
    worktrees and forced ``PYTHONPATH`` to the current checkout;
  - reran the Cyclone short-case grid/spectral profiler with ``10`` repeats on
    local CPU and one RTX A4000 (``CUDA_VISIBLE_DEVICES=0``);
  - regenerated ``docs/_static/nonlinear_rhs_profile.{csv,json,png,pdf}``
    companions;
  - current scoped result: GPU spectral mode reduces the nonlinear bracket by
    about ``3.25x`` and full RHS by about ``1.69x`` relative to grid mode, but
    the refreshed ``office`` timings are slower than the older stale artifact;
  - next performance action is not to claim a new speedup, but to isolate
    environment/runtime effects from source effects and then profile the
    linear RHS hot path.
- Refreshed the linear RHS term profile on current CPU and ``office`` GPU
  stacks:
  - reran initial-state and active ``z_wave`` Cyclone profiles with ``8``
    repeats locally and on one ``office`` RTX A4000 from a fresh detached
    ``d357597`` clone;
  - regenerated ``docs/_static/linear_rhs_terms_profile*.{csv,json}``;
  - current CPU full linear RHS is about ``1.46e-1`` to ``1.53e-1 s`` in the
    profiler harness, with streaming, linked ``grad_z``, collisions,
    hypercollisions, and linked ``|k_z|`` as the largest standalone costs;
  - current GPU full linear RHS is about ``8.95e-3 s`` in both initial and
    active states, with collisions/hypercollisions/streaming/build-``H`` as the
    largest standalone costs;
  - optimization should first target state-window-safe zero-source branches and
    linked derivative assembly only where identity gates prove the terms remain
    inactive or equivalent.
- Implemented the first state-window-gated linear RHS fast path:
  - added a dynamic ``jax.lax.cond`` guard around the collision contribution so
    production RHS assembly skips the collision operator only when the current
    species collision frequencies are exactly zero and no pre-expanded
    collision matrix is present, or when the collision term weight is exactly
    zero;
  - preserved correctness for cache reuse by keying the guard on the current
    ``LinearParams.nu`` rather than on a cache-build-time flag, and preserved
    pre-expanded collision matrices with ``nu=0`` through new regression tests;
  - reran the state-window gate; it still accepts the zero-collision skip and
    rejects the linked ``|k_z|`` hypercollision skip with maximum relative RHS
    error ``3.59e-3`` on resolved ``z``-varying states;
  - refreshed local CPU and ``office`` GPU linear RHS profiles from the same
    Cyclone nonlinear runtime harness: CPU full linear RHS is now about
    ``1.17e-1`` to ``1.26e-1 s`` and one-RTX-A4000 GPU full linear RHS is now
    about ``6.18e-3`` to ``6.43e-3 s``;
  - updated README and performance docs with the refreshed numbers while
    keeping the claim scoped to this bounded profiler artifact.
- Added conservative exact-zero guards for additional linear RHS terms:
  - factored a shared static-zero helper for non-traced term weights and
    coefficients in ``spectraxgk.terms.linear_terms``;
  - skipped streaming, GX-style streaming, hypercollision, hyperdiffusion, and
    end-damping work only when the corresponding contribution is
    mathematically zero at trace time;
  - preserved linked ``|k_z|`` hypercollision safety by keeping the existing
    z-varying activation regression and adding a guard test that only an
    exactly zero operator may bypass the linked transform;
  - verified with focused normal-precision and ``JAX_ENABLE_X64=1`` shards.
- Refactored linked-FFT z-operators without changing numerics:
  - consolidated the duplicated linked-chain/gather/full-cover/scatter
    machinery behind ``_linked_fft_apply`` while keeping public
    ``grad_z_linked_fft`` and ``abs_z_linked_fft`` wrappers unchanged;
  - preserved the separate ``i k_z`` and ``|k_z|`` multipliers and the
    real-FFT conjugate restoration path;
  - verified with normal-precision and ``JAX_ENABLE_X64=1`` linked-operator and
    GX-consistency shards.
- Refreshed the post-refactor linear RHS profiler artifacts:
  - confirmed the full GitHub CI run for ``d661c06`` passed, including
    quick-test shards, docs/package, fast coverage, and wide coverage;
  - reran CPU initial and active ``z_wave`` Cyclone linear-RHS profiles with
    ``8`` repeats and refreshed ``docs/_static/linear_rhs_terms_profile*.{csv,json}``;
  - reran the same GPU profiles on ``office`` from a fresh ``d661c06`` clone on
    one RTX A4000 and copied back the tracked GPU artifacts;
  - reran the zero-norm state-window gate, which still accepts the
    zero-collision skip and rejects linked ``|k_z|`` hypercollision disabling
    with maximum relative skip error ``3.59e-3``;
  - current bounded CPU full linear RHS timings are ``1.08e-1 s`` initial and
    ``1.27e-1 s`` active ``z_wave``; current one-RTX-A4000 timings are
    ``5.50e-3 s`` initial and ``5.48e-3 s`` active ``z_wave``.
- Refreshed the nonlinear RHS hot-path profile after the linear-RHS tranche:
  - reran the short Cyclone split profiler locally for grid and spectral
    Laguerre modes with ``10`` repeats and on ``office`` from a fresh
    ``80e8594`` clone on one RTX A4000;
  - regenerated ``docs/_static/nonlinear_rhs_profile.{csv,json,png,pdf}`` and
    visually checked that the publication-facing PNG is readable on log scale;
  - current bounded full-RHS timings are ``1.01e-1 s`` CPU grid,
    ``7.73e-2 s`` CPU spectral, ``9.66e-3 s`` GPU grid, and ``6.38e-3 s`` GPU
    spectral;
  - current spectral grid-over-spectral ratios are ``1.30``/``1.51`` for
    full RHS on CPU/GPU and ``1.54``/``2.24`` for nonlinear bracket on CPU/GPU;
  - next performance gate is a larger benchmark-size profile with profiler
    traces before making any new broad runtime claim.
- Added the first larger benchmark-size nonlinear RHS split profile:
  - extended ``tools/plot_nonlinear_rhs_profile.py`` so the same publication
    plotter can consume labeled arbitrary CSV inputs and write a case-specific
    JSON summary;
  - profiled the shipped Cyclone Miller nonlinear case
    (``Nx=192``, ``Ny=64``, ``Nz=24``, ``Nl=4``, ``Nm=8``) with ``3`` repeats
    on local CPU and on one ``office`` RTX A4000;
  - added Miller CSV companions plus
    ``docs/_static/nonlinear_rhs_profile_miller.{json,png,pdf}`` and
    documented the result in the performance guide;
  - current Miller full-RHS timings are ``2.84e-1 s`` CPU grid,
    ``2.07e-1 s`` CPU spectral, ``1.48e-2 s`` GPU grid, and ``1.46e-2 s`` GPU
    spectral;
  - the GPU bracket improves by ``2.09x`` but full RHS improves only ``1.01x``
    because the linear RHS is now the limiting kernel, so the next optimization
    tranche should target linear-RHS fusion/cache layout before broader speedup
    claims.
- Audited disabled electromagnetic field handling in the RHS hot path:
  - added a shared trace-safe static-zero helper in
    ``spectraxgk.terms.assembly`` and route disabled ``A_parallel``/``B_parallel``
    fields as ``None`` to ``build_H`` and the nonlinear bracket where this is
    safe, while preserving zero-filled arrays for terms whose signatures expect
    arrays;
  - caught and fixed an attempted static-``TermConfig`` JIT optimization because
    it broke autodiff paths where term switches are tracers; differentiability
    takes priority over that compile-time specialization;
  - validated with targeted assembly/nonlinear/autodiff tests and a docs build;
  - bounded local Cyclone Miller profiling did not show a reliable speedup
    (latest local CPU split profile was about ``linear_rhs=1.17e-1 s`` and
    ``full_rhs=3.25e-1 s``), so no README/runtime claim is updated from this
    cleanup;
  - next performance step remains a fused full-linear-RHS profiler/trace pass
    and only then a source change with matched before/after artifacts.
- Tightened the local fast-test runner after a timeout audit:
  - replaced ``tools/run_tests_fast.py`` with a bounded per-file runner that
    uses ``python -m pytest -q --maxfail=1 --disable-warnings`` and enforces a
    300-second whole-run budget by default;
  - remaining files are reported as ``not_run(total_timeout)`` instead of
    relying on an external timeout wrapper that can leave child pytest
    processes running;
  - added runner unit tests and documented ``--total-timeout 0`` for an
    explicit full sequential local pass;
  - verified the runner tests, targeted autodiff/RHS regressions, and docs
    build locally.
- Added the fused full-linear-RHS trace profiler:
  - introduced ``tools/profile_full_linear_rhs_trace.py`` to lower and time the
    production linear-RHS assembly for real runtime TOML cases, including
    optional JAX trace, memory profile, HLO text output, and a compact JSON
    summary;
  - generated the first Cyclone Miller artifacts
    ``docs/_static/full_linear_rhs_trace_summary.json`` and
    ``docs/_static/full_linear_rhs_trace_z_wave_summary.json`` with local CPU
    ``warm_seconds=1.19e-1`` (initial), ``1.22e-1`` (active ``z_wave``), and
    ``compile_execute_seconds≈1.94``;
  - HLO triage shows the next optimization tranche should target graph/layout
    pressure rather than another scalar zero branch: broadcasts ``861``,
    reshapes ``422``, FFT mentions ``312``, reductions ``304``, and gathers
    ``51``;
  - documented the artifact in README/performance docs and added manifest plus
    unit-test coverage;
  - no new speedup claim is made until a source change is backed by matched
    before/after CPU and GPU profiler artifacts.
- Completed the first source optimization tranche from the fused linear-RHS
  trace:
  - electrostatic runs now pass disabled ``A_parallel`` and ``B_parallel``
    fields as static ``None`` through the linear streaming/diamagnetic kernels,
    while the generic dynamic path remains available for electromagnetic and
    autodiff-tracer switches;
  - added parity tests showing explicit zero electromagnetic arrays and the
    electrostatic-specialized JIT path produce matching RHS values;
  - refreshed ``docs/_static/full_linear_rhs_trace_summary.json`` and
    ``docs/_static/full_linear_rhs_trace_z_wave_summary.json`` with local CPU
    electrostatic-specialized artifacts: initial ``warm_seconds=8.09e-2`` and
    ``compile_execute_seconds=1.40``; active ``z_wave``
    ``warm_seconds=1.29e-1`` and ``compile_execute_seconds=1.74``;
  - the graph shrank from the pre-specialization ``2425`` HLO lines to
    ``2225`` lines, with broadcasts ``861 -> 748``, reshapes ``422 -> 377``,
    and multiplies ``161 -> 127``; FFT and reduction counts remain unchanged
    at ``312`` and ``304``;
  - this supports a bounded local CPU initial-state graph-speedup statement
    only. The active ``z_wave`` timing remains noisy, so the next performance
    gate is a matched GPU and nonlinear full-RHS profile before any broad
    runtime claim.
- Closed the matched one-GPU follow-up for the electrostatic linear-RHS
  specialization:
  - used a clean ``office`` clone at ``1469202`` because the existing mainline
    checkout was dirty and on an old refactor branch;
  - added ``docs/_static/full_linear_rhs_trace_gpu_summary.json`` and
    ``docs/_static/full_linear_rhs_trace_gpu_z_wave_summary.json`` to the
    performance manifest;
  - one RTX A4000 measured fused linear-RHS ``warm_seconds=5.28e-3`` initial
    and ``5.25e-3`` active ``z_wave`` with ``force_electrostatic_fields=true``;
  - same-commit benchmark-size nonlinear split measured GPU
    ``full_rhs=1.71e-2 s`` grid and ``1.48e-2 s`` spectral, with the nonlinear
    bracket improving from ``8.60e-3`` to ``4.14e-3 s`` but full RHS still
    limited by mixed linear/bracket costs;
  - no README headline runtime claim is updated from this run. The next
    performance tranche should use profiler traces on the full nonlinear RHS
    and target fused layout plus larger-grid bracket decomposition.
- Added the fused full-nonlinear-RHS trace profiler:
  - introduced ``tools/profile_full_nonlinear_rhs_trace.py`` to lower and time
    the complete ``nonlinear_rhs_cached`` graph, including optional trace,
    memory profile, HLO text output, Laguerre grid/spectral mode selection, and
    compact JSON summaries;
  - added unit coverage for the nonlinear trace summary schema and missing
    electromagnetic field norms;
  - generated ``docs/_static/full_nonlinear_rhs_trace_summary.json`` locally:
    CPU ``warm_seconds=2.96e-1``, ``3345`` HLO lines, electrostatic
    specialized;
  - generated ``docs/_static/full_nonlinear_rhs_trace_gpu_summary.json`` on one
    ``office`` RTX A4000: ``warm_seconds=1.49e-2``, ``3338`` HLO lines;
  - GPU HLO token triage is dominated by reshapes ``1539``, broadcasts
    ``1822``, multiplies ``871``, FFT mentions ``229``, slices ``215``, and
    reductions ``132``. The next source tranche should target fused layout and
    bracket data movement with parity gates, not a broad speedup claim.
- Removed a duplicate non-Laguerre field-mask pass from
  ``nonlinear_em_contribution`` and added a regression that ensures the
  electrostatic non-Laguerre path masks ``phi`` once. The refreshed CPU/GPU
  nonlinear trace artifacts show unchanged HLO counts, so this is treated as a
  cleanup/guardrail rather than a performance claim.
- Completed the next nonlinear transform source tranche:
  - replaced the production grid-Laguerre nonlinear transforms with
    precision-controlled ``einsum`` contractions that preserve the previous
    ``moveaxis``/``tensordot`` algebra without the extra layout transposes;
  - added a direct regression comparing both Laguerre directions against the
    old explicit algebra on complex test states;
  - transform-only probes showed exact agreement at the tested precision, with
    CPU grid/inverse timings improving from about ``1.08e-2``/``1.05e-2 s`` to
    ``6.33e-3``/``4.91e-3 s`` and one-RTX-A4000 timings improving from about
    ``1.50e-3``/``1.55e-3 s`` to ``8.96e-4``/``9.33e-4 s``;
  - refreshed the benchmark-size Cyclone Miller split profile: CPU
    ``full_rhs=3.19e-1 s`` grid and ``2.76e-1 s`` spectral; one ``office`` RTX
    A4000 ``full_rhs=1.28e-2 s`` grid and ``1.48e-2 s`` spectral;
  - refreshed the fused full-nonlinear-RHS trace: local CPU
    ``warm_seconds=3.16e-1`` and ``3343`` HLO lines; one ``office`` RTX A4000
    ``warm_seconds=1.28e-2`` and ``3336`` HLO lines, with transposes dropping
    from ``44`` to ``32`` relative to the previous GPU artifact;
  - this is now a bounded profiler-state GPU source improvement. It is not a
    transport-runtime claim, and the next production performance lane remains
    larger-state nonlinear profiling plus linear-RHS/bracket layout work.
- Closed the release-level technical/performance readiness lane:
  - confirmed the latest checked public CI run on ``main`` passed before this
    push, including hygiene, docs/package, quick shards, and wide coverage;
  - added matched W7-X and HSX runtime-mode nonlinear RHS profiler artifacts:
    ``docs/_static/nonlinear_rhs_profile_stellarator_runtime.{png,pdf,json}``
    and CPU/GPU CSV companions;
  - the W7-X runtime-mode profile records ``full_rhs≈3.09e-1 s`` on CPU and
    ``2.73e-2 s`` on one GPU; HSX records ``full_rhs≈3.09e-1 s`` on CPU and
    ``2.71e-2 s`` on one GPU;
  - updated the manuscript-readiness and open-research dashboards so the
    profiler/performance lane closes only when the nonlinear sharding identity
    gate, Miller CPU/GPU split profiles, W7-X/HSX CPU/GPU split profiles, and
    fused full-nonlinear CPU/GPU traces are all present and GPU rows are faster
    than their CPU counterparts;
  - regenerated ``docs/_static/manuscript_readiness_status.*`` and
    ``docs/_static/open_research_lane_status.*``. The manuscript/release scope
    now reports ``5/5`` active lanes closed with W7-X zonal recurrence and
    TEM/kinetic-electron stellarator validation deferred by scope. The broader
    research tracker reports two closed lanes, two partial lanes, and one open
    lane;
  - kept the claim surface conservative: release performance evidence is
    closed for runtime/memory accounting, CPU/GPU profiler coverage, and
    numerical-identity gates, while broad production nonlinear speedup and
    nonlinear domain-decomposition claims remain future science/performance
    work.

### 2026-05-10

- Completed a deep planning audit for the next development cycle:
  - reviewed current SPECTRAX-GK parallelization, sharding, runner, geometry,
    solver-gradient, and quasilinear calibration modules;
  - inspected GX source on ``ssh office`` and confirmed its production
    multi-GPU path decomposes species first and Hermite moments second, with
    Hermite ghost exchange and field-solve reductions/broadcasts rather than
    spatial FFT-axis decomposition;
  - reviewed current JAX ``shard_map``, ``pmap``, ``Mesh``/``NamedSharding``,
    and multi-process documentation;
  - reviewed local ``vmec_jax`` and ``booz_xform_jax`` examples/docs for
    boundary-parameter optimization, JAX-native Boozer transforms, and
    sensitivity examples;
  - refreshed the literature anchors for parallelization, quasilinear
    saturation-rule validation, stellarator microstability optimization,
    nonlinear turbulence optimization, and W7-X stella/GENE validation.
- Added the ordered execution plan section above. The fixed order is:
  parallelization, refactor completion, differentiable geometry extension,
  docs/examples, quasilinear absolute-flux promotion, performance, W7-X zonal
  recurrence/damping, and W7-X fluctuation/TEM/multi-flux-tube extension.
- Focused validation passed:
  - ``python -m pytest -q tests/test_parallel.py tests/test_sharding.py tests/test_sharded_integrators.py tests/test_generate_parallel_ky_scan_gate.py tests/test_profile_nonlinear_sharding.py tests/test_nonlinear_sharding_artifacts.py``
    passed with ``30`` tests.
- Next best implementation steps:
  - add ``ParallelConfig`` and wire it into scan/quasilinear/UQ paths without
    changing serial defaults;
  - extend ``batch_map`` tests for pytrees and more device-count edge cases;
  - add a logical-CPU parallel scan artifact generator;
  - add ``velocity_sharding.py`` with the GX-inspired species/Hermite
    decomposition planner and tests;
  - then implement the first ``shard_map`` Hermite ghost-exchange unit test.
- Completed the first parallelization implementation tranche:
  - added ``RuntimeParallelConfig`` with serial defaults, validated strategy
    aliases, device-count fields, and TOML round-trip coverage;
  - exported the config through the public package API and runtime TOML loader;
  - made ``batch_map`` pytree-safe so UQ, sensitivity, and optimization
    workloads can return structured diagnostics instead of only one array;
  - wired explicit ``strategy = "combined-ky"`` / ``"batch-ky"`` runtime
    config into the existing combined-``k_y`` scan path without changing
    default serial behavior;
  - added regression tests for pytree batching on single-device and simulated
    multi-device branches, invalid parallel config values, and config-driven
    combined-``k_y`` scan selection.
- Validation for this tranche:
  - ``python -m pytest -q tests/test_parallel.py tests/test_runtime_config.py tests/test_runtime_runner.py::test_run_runtime_scan_serial_forwards_per_ky tests/test_runtime_runner.py::test_run_runtime_scan_parallel_config_selects_combined_ky tests/test_runtime_runner.py::test_run_runtime_scan_batch_ky_rejects_krylov``
    passed with ``33`` tests;
  - ``python -m pytest -q tests/test_parallel.py tests/test_sharding.py tests/test_sharded_integrators.py tests/test_generate_parallel_ky_scan_gate.py tests/test_profile_nonlinear_sharding.py tests/test_nonlinear_sharding_artifacts.py tests/test_runtime_config.py tests/test_runtime_runner.py::test_run_runtime_scan_parallel_config_selects_combined_ky``
    passed with ``57`` tests;
  - ``ruff check --extend-ignore F401`` passed on the touched source/tests.
    The ``F401`` ignore is scoped to this tranche because ``runtime.py`` keeps
    several imported names as a historical module patch surface.
- Next best implementation steps:
  - add a logical-CPU parallel scan artifact generator using
    ``RuntimeParallelConfig`` metadata;
  - add ``velocity_sharding.py`` with a species/Hermite decomposition planner,
    including numerical-identity and load-balance tests;
  - implement the first ``shard_map`` Hermite ghost-exchange unit test before
    moving any nonlinear production path.
- Completed the second parallelization implementation tranche:
  - added ``tools/generate_logical_cpu_parallel_scan_gate.py`` and generated
    ``docs/_static/logical_cpu_parallel_scan_gate.{png,pdf,csv,json}``;
  - the tracked logical-CPU gate used two logical CPU devices and passed:
    ``max_gamma_rel_error=6.7e-8``, ``max_ql_rel_error=1.1e-7``, and
    ``max_omega_abs_error=0``;
  - added ``spectraxgk.velocity_sharding`` with a GX-inspired species-first,
    Hermite-second decomposition planner that records active axes, shard shape,
    Hermite ghost-exchange requirements, field-reduction axes, communication
    pattern, and load balance;
  - exposed ``VelocityShardingPlan`` and ``build_velocity_sharding_plan`` from
    the public package API;
  - updated ``docs/performance.rst``, ``docs/examples.rst``,
    ``docs/inputs.rst``, ``docs/testing.rst``, ``docs/api.rst``, and
    ``tools/performance_optimization_manifest.toml`` with the new identity
    gate and decomposition planner.
- Validation for this tranche:
  - ``python -m pytest -q tests/test_generate_logical_cpu_parallel_scan_gate.py tests/test_parallel.py tests/test_runtime_config.py tests/test_runtime_runner.py::test_run_runtime_scan_parallel_config_selects_combined_ky``
    passed with ``35`` tests;
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_sharding.py tests/test_sharded_integrators.py tests/test_parallel.py tests/test_generate_logical_cpu_parallel_scan_gate.py``
    passed with ``31`` tests;
  - targeted ``ruff check --extend-ignore F401`` passed on the new/touched
    source, tool, and tests.
- Next best implementation steps:
  - add a local ``shard_map`` Hermite ghost-exchange unit test that uses the
    velocity plan metadata but does not yet alter production nonlinear
    evolution;
  - add a two-device CPU artifact for the Hermite-exchange kernel;
  - only after that, start wiring the production nonlinear velocity
    decomposition behind an opt-in strategy with strict numerical-identity
    gates.
- Completed the third parallelization implementation tranche:
  - added ``hermite_neighbor_reference`` and ``hermite_neighbor_shard_map`` to
    ``spectraxgk.velocity_sharding``;
  - the shard-map primitive exchanges nearest Hermite neighbors across a
    one-dimensional Hermite mesh with zero physical boundaries at ``m=0`` and
    ``m=Nm-1``;
  - added tests for the full-array reference, one-device fallback,
    multi-device shard-map path when logical devices are available, and
    explicit rejection of unsupported multi-axis species/Hermite plans;
  - added ``tools/generate_hermite_exchange_gate.py`` and generated
    ``docs/_static/hermite_exchange_gate.{png,pdf,csv,json}``;
  - the tracked two-logical-CPU artifact passes with
    ``max_lower_abs_error=0`` and ``max_upper_abs_error=0``;
  - documented the gate in ``docs/performance.rst``, ``docs/examples.rst``,
    ``docs/testing.rst``, and ``tools/performance_optimization_manifest.toml``.
- Validation for this tranche:
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_generate_hermite_exchange_gate.py``
    passed with ``9`` tests and one multi-device test skipped in the
    single-device default process;
  - targeted ``ruff check --extend-ignore F401`` passed on the new/touched
    source, tool, and tests;
  - ``python tools/generate_hermite_exchange_gate.py --logical-devices 2
    --out-prefix docs/_static/hermite_exchange_gate`` generated the tracked
    passing artifact.
- Next best implementation steps:
  - add a field-reduction/broadcast identity gate for velocity-space sharding;
  - add a streaming-Hermite-ladder identity gate that combines the ghost
    exchange with the actual Hermite coupling coefficients;
  - after both pass, wire an opt-in velocity-decomposed linear streaming
    microkernel before touching nonlinear production RHS.
- Completed the fourth parallelization implementation tranche:
  - added ``velocity_field_reduce_reference`` and
    ``velocity_field_reduce_shard_map`` to ``spectraxgk.velocity_sharding``;
  - the shard-map primitive reduces local Hermite-sharded contributions with
    ``lax.psum`` and broadcasts the reduced field contribution across the
    Hermite mesh;
  - added unit coverage for the reference reduction, single-device fallback,
    multi-device shard-map path when logical devices are available, and the
    artifact writer;
  - added ``tools/generate_velocity_field_reduce_gate.py`` and generated
    ``docs/_static/velocity_field_reduce_gate.{png,pdf,csv,json}``;
  - the tracked two-logical-CPU artifact passes with
    ``max_abs_error=3.814697265625e-6`` under ``atol=1e-5``. This is a
    float32 reduction-tree tolerance, not a physics tolerance;
  - documented the gate in ``docs/performance.rst``, ``docs/examples.rst``,
    ``docs/testing.rst``, and ``tools/performance_optimization_manifest.toml``.
- Validation for this tranche:
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_generate_velocity_field_reduce_gate.py tests/test_generate_hermite_exchange_gate.py``
    passed with ``13`` tests and two multi-device tests skipped in the
    single-device default process;
  - targeted ``ruff check --extend-ignore F401`` passed on the new/touched
    source, tool, and tests;
  - ``python tools/generate_velocity_field_reduce_gate.py --logical-devices 2
    --out-prefix docs/_static/velocity_field_reduce_gate`` generated the
    tracked passing artifact.
- Next best implementation steps:
  - add a streaming-Hermite-ladder identity gate combining neighbor exchange,
    field reduction, and the actual Hermite streaming coefficients;
  - then implement an opt-in velocity-decomposed linear streaming microkernel
    with serial numerical-identity gates;
  - only after that, evaluate nonlinear RHS integration behind a disabled-by-
    default ``RuntimeParallelConfig(strategy="velocity")`` path.
- Completed the fifth parallelization implementation tranche:
  - added ``hermite_streaming_ladder_reference`` and
    ``hermite_streaming_ladder_shard_map`` to ``spectraxgk.velocity_sharding``;
  - the ladder applies the production Hermite coefficients ``sqrt(m+1)`` and
    ``sqrt(m)`` on top of the shard-map neighbor exchange, with scalar or
    per-species ``vth`` broadcasting;
  - added unit coverage for manual coefficient placement, one-device fallback,
    and multi-device shard-map behavior when logical devices are available;
  - added ``tools/generate_hermite_streaming_ladder_gate.py`` and generated
    ``docs/_static/hermite_streaming_ladder_gate.{png,pdf,csv,json}``;
  - the tracked two-logical-CPU artifact passes with zero ladder absolute and
    relative error, and records the paired Hermite field-reduction error
    ``1.9073486328125e-6``;
  - documented the gate in ``docs/performance.rst``, ``docs/examples.rst``,
    ``docs/testing.rst``, and ``tools/performance_optimization_manifest.toml``.
- Validation for this tranche:
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_generate_hermite_streaming_ladder_gate.py tests/test_generate_velocity_field_reduce_gate.py tests/test_generate_hermite_exchange_gate.py``
    passed with ``17`` tests and three multi-device tests skipped in the
    single-device default process;
  - targeted ``ruff check --extend-ignore F401`` passed on the new/touched
    source, tool, and tests;
  - ``python tools/generate_hermite_streaming_ladder_gate.py --logical-devices
    2 --out-prefix docs/_static/hermite_streaming_ladder_gate`` generated the
    tracked passing artifact.
- Next best implementation steps:
  - implement an opt-in linear streaming microkernel that combines the
    parallel derivative contract with the Hermite streaming ladder;
  - gate that microkernel against ``spectraxgk.terms.operators.streaming_term``
    on periodic field-line grids;
  - after that, start a disabled-by-default ``RuntimeParallelConfig`` route for
    velocity-decomposed streaming in linear scans before nonlinear RHS work.
- Completed the sixth parallelization implementation tranche:
  - added ``periodic_streaming_reference`` and
    ``periodic_streaming_shard_map`` to ``spectraxgk.velocity_sharding``;
  - the microkernel applies the periodic spectral field-line derivative and
    then the Hermite streaming ladder through the shard-map communication path;
  - added tests comparing the reference path directly against
    ``spectraxgk.terms.operators.streaming_term``, plus one-device fallback and
    multi-device shard-map behavior when logical devices are available;
  - added ``tools/generate_periodic_streaming_microkernel_gate.py`` and
    generated
    ``docs/_static/periodic_streaming_microkernel_gate.{png,pdf,csv,json}``;
  - the tracked two-logical-CPU artifact passes with zero reported absolute and
    relative error against the production streaming operator;
  - documented the gate in ``docs/performance.rst``, ``docs/examples.rst``,
    ``docs/testing.rst``, and ``tools/performance_optimization_manifest.toml``.
- Validation for this tranche:
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_generate_periodic_streaming_microkernel_gate.py tests/test_generate_hermite_streaming_ladder_gate.py``
    passed with ``17`` tests and four multi-device tests skipped in the
    single-device default process;
  - targeted ``ruff check --extend-ignore F401`` passed on the new/touched
    source, tool, and tests;
  - ``python tools/generate_periodic_streaming_microkernel_gate.py
    --logical-devices 2 --out-prefix
    docs/_static/periodic_streaming_microkernel_gate`` generated the tracked
    passing artifact.
- Next best implementation steps:
  - add a disabled-by-default runtime/config route for the periodic
    velocity-decomposed streaming microkernel in a reduced linear
    streaming-only path;
  - gate full ``linear_rhs`` identity with all non-streaming terms disabled
    before enabling any general linear scan path;
  - then profile the streaming-only microkernel locally and on office GPUs.
- Completed the seventh parallelization implementation tranche:
  - added ``tools/generate_linear_rhs_streaming_gate.py`` to compare the
    production ``linear_rhs_cached`` call graph, with only streaming enabled,
    against ``spectraxgk.velocity_sharding.periodic_streaming_shard_map``;
  - the gate disables drive, curvature, magnetic-drift, mirror, nonlinear,
    collision, source, and electromagnetic channels, and uses non-density
    Hermite moments so the electrostatic field solve is exactly zero;
  - added ``tests/test_generate_linear_rhs_streaming_gate.py`` and generated
    ``docs/_static/linear_rhs_streaming_gate.{png,pdf,csv,json}``;
  - the tracked two-logical-CPU artifact passes with
    ``max_abs_error=9.62942522164667e-7``,
    ``max_rel_error=5.559545002142841e-7``, and ``phi_norm=0``;
  - documented the gate in ``docs/performance.rst``, ``docs/examples.rst``,
    ``docs/testing.rst``, and ``tools/performance_optimization_manifest.toml``.
- Validation for this tranche:
  - ``python -m pytest -q tests/test_generate_linear_rhs_streaming_gate.py tests/test_velocity_sharding.py``
    passed with ``15`` tests and four multi-device tests skipped in the
    single-device default process;
  - ``python tools/generate_linear_rhs_streaming_gate.py --logical-devices 2
    --out-prefix docs/_static/linear_rhs_streaming_gate`` generated the tracked
    passing artifact.
- Next best implementation steps:
  - run the bounded parallelization test shard, targeted ruff, and docs build;
  - add a disabled-by-default runtime/config route for a streaming-only
    velocity-decomposed linear diagnostic path;
  - then gate a full linear RHS slice with streaming plus field-solve/drift
    terms before any broad runtime exposure.
- Bounded verification after the seventh tranche:
  - ``python -m pytest -q tests/test_parallel.py tests/test_velocity_sharding.py tests/test_generate_logical_cpu_parallel_scan_gate.py tests/test_generate_hermite_exchange_gate.py tests/test_generate_velocity_field_reduce_gate.py tests/test_generate_hermite_streaming_ladder_gate.py tests/test_generate_periodic_streaming_microkernel_gate.py tests/test_generate_linear_rhs_streaming_gate.py tests/test_generate_parallel_ky_scan_gate.py tests/test_sharding.py tests/test_sharded_integrators.py tests/test_profile_nonlinear_sharding.py tests/test_nonlinear_sharding_artifacts.py tests/test_runtime_config.py tests/test_runtime_runner.py::test_run_runtime_scan_parallel_config_selects_combined_ky tests/test_runtime_runner.py::test_run_runtime_scan_batch_ky_rejects_krylov``
    passed under the 300-second cap with the expected logical-device skips;
  - targeted ``ruff check --extend-ignore F401`` passed on all touched source,
    tools, and tests;
  - ``python -m sphinx -q -b html docs docs/_build/html`` passed under the
    300-second cap.
- Started the eighth parallelization implementation tranche:
  - added ``linear_rhs_streaming_velocity_sharded`` and
    ``linear_rhs_parallel_cached`` as a disabled-by-default code-level route
    for the Hermite velocity-sharded streaming operator;
  - ``linear_rhs_parallel_cached`` is serial by default and only dispatches to
    the sharded route for ``RuntimeParallelConfig(strategy="velocity",
    axis="hermite", backend="streaming_only")``;
  - the route rejects non-streaming ``LinearTerms`` so it cannot silently alter
    full linear or nonlinear physics;
  - added unit coverage showing that the explicit velocity route matches
    production ``linear_rhs_cached`` on a streaming-only periodic problem and
    rejects default full-physics terms;
  - documented the diagnostic route in ``docs/inputs.rst`` and
    ``docs/performance.rst``.
- Validation for this tranche so far:
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_generate_linear_rhs_streaming_gate.py``
    passed under the 300-second cap with expected logical-device skips;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched
    linear/API/test files.
  - ``python -m pytest -q tests/test_parallel.py tests/test_velocity_sharding.py tests/test_generate_linear_rhs_streaming_gate.py tests/test_runtime_config.py tests/test_runtime_runner.py::test_run_runtime_scan_parallel_config_selects_combined_ky tests/test_runtime_runner.py::test_run_runtime_scan_batch_ky_rejects_krylov``
    passed under the 300-second cap;
  - ``python -m sphinx -q -b html docs docs/_build/html`` passed under the
    300-second cap after documenting the hook.
- Continued the eighth parallelization implementation tranche:
  - added ``linear_rhs_streaming_electrostatic_velocity_sharded`` and the
    explicit ``RuntimeParallelConfig(strategy="velocity", axis="hermite",
    backend="streaming_electrostatic")`` dispatch route;
  - this route solves electrostatic ``phi`` with the production serial field
    solve, applies the Hermite velocity-sharded particle-streaming operator,
    and adds the GX-style electrostatic streaming field term;
  - the route rejects linked-boundary/twist-shift grids and non-streaming
    ``LinearTerms`` until those receive separate identity gates;
  - added unit coverage against production ``linear_rhs_cached`` with a
    nonzero ``m=0`` density perturbation so ``phi`` is nonzero;
  - added ``tools/generate_linear_rhs_streaming_electrostatic_gate.py`` and
    generated
    ``docs/_static/linear_rhs_streaming_electrostatic_gate.{png,pdf,csv,json}``;
  - the tracked two-logical-CPU artifact passes with ``phi_norm=0.1342447549``,
    ``max_phi_abs_error=1.862645149230957e-9``,
    ``max_abs_error=1.3943616750111687e-7``, and
    ``max_rel_error=4.0251720179185213e-7``.
- Validation for this tranche:
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_generate_linear_rhs_streaming_electrostatic_gate.py``
    passed under the 300-second cap with expected logical-device skips;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched
    source, tool, and tests;
  - ``python tools/generate_linear_rhs_streaming_electrostatic_gate.py
    --logical-devices 2 --out-prefix
    docs/_static/linear_rhs_streaming_electrostatic_gate`` generated the
    tracked passing artifact.
- Next best implementation steps:
  - run the bounded parallelization/docs verification shard for the updated
    route and artifact;
  - implement a true sharded electrostatic field-reduction gate, replacing the
    serial field solve in the diagnostic route only after identity passes;
  - then add mirror/curvature/grad-B drift slices one at a time.
- Bounded verification after the electrostatic streaming gate:
  - ``python -m pytest -q tests/test_parallel.py tests/test_velocity_sharding.py tests/test_generate_logical_cpu_parallel_scan_gate.py tests/test_generate_hermite_exchange_gate.py tests/test_generate_velocity_field_reduce_gate.py tests/test_generate_hermite_streaming_ladder_gate.py tests/test_generate_periodic_streaming_microkernel_gate.py tests/test_generate_linear_rhs_streaming_gate.py tests/test_generate_linear_rhs_streaming_electrostatic_gate.py tests/test_runtime_config.py tests/test_runtime_runner.py::test_run_runtime_scan_parallel_config_selects_combined_ky tests/test_runtime_runner.py::test_run_runtime_scan_batch_ky_rejects_krylov``
    passed under the 300-second cap;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched
    source, tool, tests, and docs;
  - ``python -m sphinx -q -b html docs docs/_build/html`` passed under the
    300-second cap.
- Started the ninth parallelization implementation tranche:
  - added ``electrostatic_phi_reference`` and ``electrostatic_phi_shard_map``
    to ``spectraxgk.velocity_sharding``;
  - the sharded path selects the global ``m=0`` density moment on a Hermite
    mesh, reduces it with ``lax.psum``, and applies the electrostatic
    quasineutrality denominator;
  - current scope is single-species, 5D, periodic electrostatic field solves;
    multi-species, linked-boundary, electromagnetic, and nonlinear fields are
    separate gates;
  - added unit coverage against the production ``linear_rhs_cached`` field
    solve and one-device/multi-device shard-map identity;
  - added ``tools/generate_electrostatic_field_reduce_gate.py`` and generated
    ``docs/_static/electrostatic_field_reduce_gate.{png,pdf,csv,json}``;
  - the tracked two-logical-CPU artifact passes with
    ``phi_norm=0.16790585219860077`` and zero reported absolute/relative error.
- Validation for this tranche so far:
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_generate_electrostatic_field_reduce_gate.py``
    passed under the 300-second cap with expected logical-device skips;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched
    source, tool, and tests;
  - ``python tools/generate_electrostatic_field_reduce_gate.py
    --logical-devices 2 --out-prefix docs/_static/electrostatic_field_reduce_gate``
    generated the tracked passing artifact.
- Completed the single-species periodic electrostatic route wiring:
  - replaced the serial field solve inside
    ``linear_rhs_streaming_electrostatic_velocity_sharded`` with
    ``electrostatic_phi_shard_map`` for supported 5D periodic electrostatic
    states;
  - unsupported multi-species/6D states and linked-boundary/twist-shift states
    now fail explicitly instead of silently falling back to a serial field
    solve;
  - regenerated
    ``docs/_static/linear_rhs_streaming_electrostatic_gate.{png,pdf,csv,json}``
    so the artifact records that ``phi`` comes from the Hermite-sharded field
    reduction gate;
  - the regenerated streaming-electrostatic artifact passes with
    ``phi_norm=0.13424475491046906``,
    ``max_phi_abs_error=1.862645149230957e-9``,
    ``max_abs_error=1.3943616750111687e-7``, and
    ``max_rel_error=4.0251720179185213e-7``.
- Next best implementation steps:
  - run the bounded parallelization/docs verification shard for the
    field-reduction and electrostatic streaming route;
  - add mirror/curvature/grad-B drift identity slices using the same
    single-species periodic gate discipline;
  - defer multi-species, linked-boundary, electromagnetic, and nonlinear field
    sharding until each has its own isolated reduction/communication gate.
- Bounded verification after field-reduction wiring:
  - ``python -m pytest -q tests/test_parallel.py tests/test_velocity_sharding.py tests/test_generate_logical_cpu_parallel_scan_gate.py tests/test_generate_hermite_exchange_gate.py tests/test_generate_velocity_field_reduce_gate.py tests/test_generate_electrostatic_field_reduce_gate.py tests/test_generate_hermite_streaming_ladder_gate.py tests/test_generate_periodic_streaming_microkernel_gate.py tests/test_generate_linear_rhs_streaming_gate.py tests/test_generate_linear_rhs_streaming_electrostatic_gate.py tests/test_runtime_config.py tests/test_runtime_runner.py::test_run_runtime_scan_parallel_config_selects_combined_ky tests/test_runtime_runner.py::test_run_runtime_scan_batch_ky_rejects_krylov``
    passed under the 300-second cap;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched
    source, tools, and tests;
  - ``python -m sphinx -q -b html docs docs/_build/html`` passed under the
    300-second cap.
- Continued the ninth parallelization implementation tranche with drift slices:
  - added generic ``hermite_shift_reference`` and ``hermite_shift_shard_map``
    for offset-1 and offset-2 Hermite exchanges;
  - added ``mirror_drift_reference`` / ``mirror_drift_shard_map`` and
    ``curvature_gradb_drift_reference`` /
    ``curvature_gradb_drift_shard_map``;
  - added unit coverage comparing those paths against production
    ``mirror_contribution`` and ``curvature_gradb_contribution`` and against
    multi-device shard-map references when logical devices are available;
  - added ``tools/generate_electrostatic_drift_gate.py`` and generated
    ``docs/_static/electrostatic_drift_gate.{png,pdf,csv,json}``;
  - the tracked two-logical-CPU artifact passes with
    ``phi_norm=0.12082028388977051`` and zero reported absolute/relative error
    for mirror, curvature/grad-B, and total drift slices.
- Validation for this tranche so far:
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_generate_electrostatic_drift_gate.py``
    passed under the 300-second cap with expected logical-device skips;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched
    source, tool, and tests;
  - ``python tools/generate_electrostatic_drift_gate.py --logical-devices 2
    --out-prefix docs/_static/electrostatic_drift_gate`` generated the tracked
    passing artifact.
- Next best implementation steps:
  - run the bounded parallelization/docs verification shard for the drift gate;
  - add a disabled-by-default ``backend="electrostatic_linear_slices"`` route
    combining sharded field reduction, streaming, mirror, curvature, and
    grad-B before adding the separately gated diamagnetic slice;
  - keep collisions, linked boundaries, electromagnetic terms, and
    nonlinear brackets behind separate isolated gates.
- Bounded verification after the drift gate:
  - ``python -m pytest -q tests/test_parallel.py tests/test_velocity_sharding.py tests/test_generate_logical_cpu_parallel_scan_gate.py tests/test_generate_hermite_exchange_gate.py tests/test_generate_velocity_field_reduce_gate.py tests/test_generate_electrostatic_field_reduce_gate.py tests/test_generate_hermite_streaming_ladder_gate.py tests/test_generate_electrostatic_drift_gate.py tests/test_generate_periodic_streaming_microkernel_gate.py tests/test_generate_linear_rhs_streaming_gate.py tests/test_generate_linear_rhs_streaming_electrostatic_gate.py tests/test_runtime_config.py tests/test_runtime_runner.py::test_run_runtime_scan_parallel_config_selects_combined_ky tests/test_runtime_runner.py::test_run_runtime_scan_batch_ky_rejects_krylov``
    passed under the 300-second cap;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched
    source, tool, and tests;
  - ``python -m sphinx -q -b html docs docs/_build/html`` passed under the
    300-second cap.
- Added the disabled-by-default composed electrostatic linear-slice backend:
  - added ``linear_rhs_electrostatic_slices_velocity_sharded`` and
    ``RuntimeParallelConfig(strategy="velocity", axis="hermite",
    backend="electrostatic_linear_slices")`` dispatch through
    ``linear_rhs_parallel_cached``;
  - the route combines the already gated Hermite-sharded electrostatic field
    reduction, electrostatic streaming, mirror, curvature, and grad-B slices;
  - a follow-on tranche promoted the gated diamagnetic-drive slice into this
    backend; it now rejects collisions, hypercollisions, end damping,
    electromagnetic terms, linked-boundary/twist-shift grids, multi-species
    states, and nonlinear terms until each has its own isolated gate;
  - added unit coverage comparing the composed backend with production
    ``linear_rhs_cached`` and verifying rejection of ungated terms.
- Validation for the composed backend so far:
  - ``python -m pytest -q tests/test_velocity_sharding.py`` passed under the
    300-second cap with expected logical-device skips;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched
    source/API/test files.
- Bounded verification after the composed backend:
  - ``python -m pytest -q tests/test_parallel.py tests/test_velocity_sharding.py tests/test_generate_electrostatic_drift_gate.py tests/test_generate_electrostatic_field_reduce_gate.py tests/test_generate_linear_rhs_streaming_electrostatic_gate.py tests/test_generate_linear_rhs_streaming_gate.py tests/test_runtime_config.py tests/test_runtime_runner.py::test_run_runtime_scan_parallel_config_selects_combined_ky tests/test_runtime_runner.py::test_run_runtime_scan_batch_ky_rejects_krylov``
    passed under the 300-second cap;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched
    source, tool, and tests;
  - ``python -m sphinx -q -b html docs docs/_build/html`` passed under the
    300-second cap.
- Continued the parallelization implementation tranche with the electrostatic
  diamagnetic-drive slice:
  - added ``diamagnetic_drive_reference`` and ``diamagnetic_drive_shard_map``
    to ``spectraxgk.velocity_sharding``;
  - the sharded path reuses the Hermite-sharded electrostatic field reduction,
    then applies local global-``m`` masks for the ``m=0`` density/temperature
    gradient drive and ``m=2`` temperature-gradient drive;
  - added unit coverage comparing the new primitive against the production
    diamagnetic-only ``linear_rhs_cached`` path and against the reference
    implementation when multiple logical devices are available;
  - added ``tools/generate_electrostatic_diamagnetic_gate.py`` and generated
    ``docs/_static/electrostatic_diamagnetic_gate.{png,pdf,csv,json}``;
  - the tracked two-logical-CPU artifact passes with
    ``phi_norm=0.16790585219860077`` and zero reported absolute/relative
    error.
- Promoted the gated diamagnetic slice into the disabled-by-default
  ``backend="electrostatic_linear_slices"`` route:
  - the composed backend now covers streaming, mirror, curvature, grad-B, and
    diamagnetic-drive slices for single-species periodic electrostatic 5D
    states;
  - it still rejects collisions, hypercollisions, hyperdiffusion, end damping,
    electromagnetic terms, linked-boundary/twist-shift grids, multi-species
    states, and nonlinear terms until each path has its own identity gate.
- Validation for this tranche so far:
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_generate_electrostatic_diamagnetic_gate.py``
    passed under the 300-second cap with expected logical-device skips;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched
    source, tool, and tests;
  - ``python tools/generate_electrostatic_diamagnetic_gate.py
    --logical-devices 2 --out-prefix
    docs/_static/electrostatic_diamagnetic_gate`` generated the tracked
    passing artifact.
- Next best implementation steps:
  - run the bounded parallelization/docs verification shard including the new
    diamagnetic gate;
  - profile the composed electrostatic linear-slices route on a larger
    CPU/GPU linear problem before making any speedup claim;
  - start the next isolated production-parallelization gate with either
    electrostatic collision/hypercollision slices or a ky/batch linear-scan
    composition gate, keeping nonlinear domain decomposition separate.
- Added the composed electrostatic linear-slices call-graph artifact:
  - added ``tools/generate_linear_rhs_electrostatic_slices_gate.py`` and
    ``tests/test_generate_linear_rhs_electrostatic_slices_gate.py``;
  - the tool compares serial production ``linear_rhs_cached`` against
    ``linear_rhs_parallel_cached`` with
    ``RuntimeParallelConfig(strategy="velocity", axis="hermite",
    backend="electrostatic_linear_slices")`` and streaming, mirror, curvature,
    grad-B, and diamagnetic drive enabled;
  - generated
    ``docs/_static/linear_rhs_electrostatic_slices_gate.{png,pdf,csv,json}``;
  - the tracked two-logical-CPU artifact passes with
    ``phi_norm=0.16790585219860077``,
    ``max_abs_error=1.467594188397925e-07``,
    ``max_rel_error=3.6947963621969393e-07``, and zero potential error.
- Validation for the composed artifact so far:
  - ``python -m pytest -q tests/test_generate_linear_rhs_electrostatic_slices_gate.py tests/test_velocity_sharding.py``
    passed under the 300-second cap with expected logical-device skips;
  - targeted ``ruff check --extend-ignore F401`` passed for the touched tool,
    tests, and linear backend source;
  - ``python tools/generate_linear_rhs_electrostatic_slices_gate.py
    --logical-devices 2 --out-prefix
    docs/_static/linear_rhs_electrostatic_slices_gate`` generated the tracked
    passing artifact.
- Next best implementation steps:
  - run the bounded docs/test shard including the composed artifact test;
  - run a larger CPU timing/profiling probe for the serial and sharded
    electrostatic linear-slices route and record it as an engineering artifact,
    not a claim unless the workload and identity gate are publication-sized;
  - keep the next physics-scope expansion focused on either collision slices
    or linked-boundary/twist-shift communication, both as isolated gates before
    exposing them through the runtime route.
- Added the electrostatic linear-slices engineering profile:
  - added ``tools/profile_linear_rhs_parallel_slices.py`` and
    ``tests/test_profile_linear_rhs_parallel_slices.py``;
  - the tool times serial production ``linear_rhs_cached`` against the opt-in
    Hermite-sharded ``backend="electrostatic_linear_slices"`` route on a
    larger bounded CPU workload, while also recording identity errors;
  - generated
    ``docs/_static/linear_rhs_parallel_slices_profile.{png,pdf,csv,json}``;
  - the first tracked local profile passed identity but was not performant:
    ``serial_median_s=0.14060408296063542``,
    ``sharded_median_s=3.856044082902372``, ``speedup=0.036463297601827566``.
- Optimization result from this profile:
  - removed one redundant electrostatic field solve from the composed backend
    by reusing the precomputed ``phi`` in the streaming slice;
  - a pure fused ``shard_map`` preserved identity but was slower without
    callable caching, confirming compile/setup overhead as the main issue;
  - added a per-cache/device cached fused Hermite ``shard_map`` callable for
    the multi-device route;
  - the updated tracked Hermite-heavy CPU profile passes the engineering
    identity gate with ``max_abs_error=2.366846729273675e-06``,
    ``max_rel_error=5.955886081210338e-06``, and
    ``max_phi_abs_error=7.508562660518692e-09``;
  - warm timings on eight logical CPU devices are
    ``serial_median_s=0.043669458013027906``,
    ``sharded_median_s=0.031135583063587546``,
    ``speedup=1.4025579005166755``;
  - this is an engineering profile, not a publication speedup claim; the
    stricter small-grid composed identity gate remains the release correctness
    gate.
- Validation for this profile tranche so far:
  - ``python -m pytest -q tests/test_profile_linear_rhs_parallel_slices.py``
    passed under the 300-second cap;
  - ``python -m pytest -q tests/test_velocity_sharding.py tests/test_generate_linear_rhs_electrostatic_slices_gate.py tests/test_profile_linear_rhs_parallel_slices.py``
    passed under the 300-second cap with expected logical-device skips;
  - targeted ``ruff check`` passed for the touched profiler, tests, and
    backend source;
  - ``python tools/generate_linear_rhs_electrostatic_slices_gate.py
    --logical-devices 2 --out-prefix
    docs/_static/linear_rhs_electrostatic_slices_gate`` regenerated the
    passing identity artifact after the reuse patch;
  - ``python tools/profile_linear_rhs_parallel_slices.py --logical-devices 8
    --nl 4 --nm 128 --ny 32 --nz 128 --warmups 1 --repeats 3 --rtol 1e-5
    --out-prefix docs/_static/linear_rhs_parallel_slices_profile`` generated
    the tracked engineering profile.
- Next best implementation steps:
  - run the bounded docs/test shard including the new profile test;
  - commit/push the profile and reuse patch;
  - run the same cached fused profile on ``ssh office`` GPUs;
  - add a profile sweep artifact over ``num_devices`` and Hermite size before
    promoting any broad parallel-speedup claim.
- Office GPU profile result:
  - cloned the pushed repository at ``4f43f1b`` into
    ``/tmp/spectrax-gk-profile`` on ``ssh office`` and verified JAX sees two
    ``CudaDevice`` GPUs;
  - added ``--platform cpu|gpu`` support to
    ``tools/profile_linear_rhs_parallel_slices.py`` so the same profiler can
    target either logical CPU devices or physical GPUs;
  - ran the two-GPU profile with
    ``PYTHONPATH=/tmp/spectrax-gk-profile/src python3
    tools/profile_linear_rhs_parallel_slices.py --platform gpu
    --logical-devices 2 --nl 4 --nm 64 --ny 32 --nz 128 --warmups 1
    --repeats 3 --rtol 1e-5 --out-prefix
    docs/_static/linear_rhs_parallel_slices_profile_gpu``;
  - copied
    ``docs/_static/linear_rhs_parallel_slices_profile_gpu.{png,pdf,csv,json}``
    back into the local repository;
  - the GPU artifact passes the engineering identity gate with
    ``max_abs_error=1.8670061763259582e-06``,
    ``max_rel_error=4.698092652688501e-06``, and
    ``max_phi_abs_error=7.451490091625601e-09``;
  - warm two-GPU timings are negative:
    ``serial_median_s=0.0042040119878947735``,
    ``sharded_median_s=0.12247901689261198``,
    ``speedup=0.034324344647383945``;
  - a larger two-GPU probe
    (``nl=4,nm=128,ny=64,nz=256``) remained slow
    (``speedup=0.05451575420187544``) and missed the ``1e-5`` relative gate,
    so the GPU Hermite-sharding lane stays open.
- Next best implementation steps:
  - commit/push the CPU/GPU profile updates;
  - add a compact profile-sweep artifact across CPU logical-device count and
    Hermite size to map the useful CPU regime;
  - for GPU, investigate a different decomposition (batch/ky or ensemble
    sharding first) rather than promoting Hermite domain sharding.
- CI coverage repair and CPU sweep follow-up:
  - investigated the failed CI run ``25629475335`` and found that all wide
    coverage shards finished, but the final combine job failed because package
    coverage rounded to ``94%``; the largest new blocker was
    ``src/spectraxgk/velocity_sharding.py`` at ``63%`` coverage;
  - added fast branch, validation, mocked ``shard_map``, and 6D species
    broadcast tests in ``tests/test_velocity_sharding.py`` so the module now
    reaches ``99%`` in the targeted coverage run
    (``508`` statements, ``6`` misses);
  - pushed ``18c2438 Add velocity sharding coverage gates`` and cancelled the
    stale superseded CI run so the fixed head can run;
  - added ``tools/profile_linear_rhs_parallel_slices_sweep.py`` plus
    ``tests/test_profile_linear_rhs_parallel_slices_sweep.py``;
  - generated
    ``docs/_static/linear_rhs_parallel_slices_sweep.{png,pdf,csv,json}`` with
    ``Nm=64,128`` over ``1,2,4,8`` logical CPU devices;
  - all sweep points pass the engineering identity gate at ``rtol=1e-5``; the
    best bounded point is ``1.57x`` at ``Nm=128`` on four logical CPU devices,
    while one/two-device points remain overhead-limited;
  - documented this as a development regime map in ``docs/performance.rst``
    and ``docs/examples.rst``, explicitly not as a nonlinear or publication
    speedup claim.
- Next best implementation steps:
  - wait for CI head ``18c2438`` to finish; if wide coverage still fails,
    inspect the combined report and patch the next real coverage blocker;
  - after CI is green, commit and push the sweep tool, artifact, and docs;
  - run an analogous GPU-side decomposition investigation on office focused on
    independent ``ky``/ensemble sharding rather than the current Hermite
    domain sharding, since the two-GPU Hermite profile is negative.
- Follow-up CI coverage repair:
  - CI head ``18c2438`` passed repo hygiene, mypy, quick tests,
    docs-and-packaging, fast coverage, and all ``48`` wide-coverage shards;
    only the final combined wide-coverage job failed, with
    ``16969`` statements, ``943`` misses, and total coverage still reported as
    ``94%``;
  - added fast benchmark-runner branch tests for:
    GX-seeded Cyclone Krylov branch selection;
    reduced Hermite-Laguerre seed fallback after primary seed failure;
    runtime-configured Cyclone auto signal selection with saved density;
    explicit-density diagnostic integration; and
    KBM ``gx_time`` diagnostic fallback ordering;
  - these tests are benchmark-logic gates, not synthetic coverage scaffolds:
    they lock branch-following, seed fallback, diagnostic fallback, and
    density/phi fit routing used by the benchmark validation lanes;
  - targeted benchmark coverage improved from ``530`` local missed benchmark
    lines before this tranche to ``426`` after it, which should recover more
    than the remaining package-wide ``95%`` gap.
- Next best implementation steps:
  - commit and push the CI coverage repair plus the CPU sweep artifact;
  - watch the new CI run through the final wide-coverage combine job;
  - if the new combine job still misses ``95%``, inspect the next largest
    remaining true blocker instead of lowering the threshold.
- CI and large strong-scaling follow-up:
  - pushed ``92924c3 Add benchmark coverage gates and parallel sweep``;
  - confirmed GitHub CI run ``25631175560`` passed hygiene, mypy, quick-test
    shards, docs/package, fast coverage, all ``48`` wide-coverage shards, and
    the final wide-coverage combine;
  - package-wide coverage now passes the ``95%`` gate with ``16969``
    statements and ``903`` misses;
  - added ``tools/profile_nonlinear_sharding_sweep.py`` so fixed-step
    nonlinear strong-scaling sweeps run each device count in a clean
    subprocess with explicit CPU logical-device or GPU visibility controls;
  - added ``tools/plot_nonlinear_sharding_strong_scaling.py`` to combine the
    CPU and GPU artifacts into one paper-facing engineering panel;
  - ran a large logical-CPU nonlinear sweep on ``ssh office`` with
    ``Nx=24, Ny=48, Nz=96, Nl=4, Nm=8, steps=8`` over ``1,2,4,8`` logical CPU
    devices; all points passed final-state identity, speedup peaked at about
    ``1.39x`` on four logical devices and flattened by eight devices;
  - ran a larger two-GPU nonlinear sweep on ``ssh office`` with
    ``Nx=48, Ny=96, Nz=128, Nl=4, Nm=8, steps=12``; both one- and two-GPU
    points passed final-state identity, but two-GPU whole-state sharding was
    slower (about ``0.63x`` speedup);
  - generated
    ``docs/_static/nonlinear_sharding_strong_scaling_cpu_large.{json,csv,png,pdf}``,
    ``docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.{json,csv,png,pdf}``,
    and combined
    ``docs/_static/nonlinear_sharding_strong_scaling_large.{json,csv,png,pdf}``;
  - updated ``docs/performance.rst`` and ``README.md`` to make the result
    explicit: current whole-state nonlinear sharding is an identity/profiler
    gate, not a production speedup path.
- Next best implementation steps:
  - commit/push the large-sweep artifacts and documentation;
  - wait for CI on the new head;
  - move production parallelization work to independent ``ky`` scans,
    quasilinear/UQ ensembles, and only then redesign nonlinear communication
    decomposition; do not keep chasing whole-state nonlinear sharding speedups
    without a new communication layout.
- Production independent-work parallelization follow-up:
  - added ``tools/profile_independent_ky_scan_scaling.py`` to run real
    Cyclone linear ``k_y`` scans in isolated CPU/GPU workers, with warmup
    scans before timed repeats and ``gamma``/``omega`` identity gates against
    the one-worker reference;
  - added ``tools/plot_independent_ky_scan_scaling.py`` to combine CPU and GPU
    artifacts into one publication-ready panel;
  - ran the large solver-backed sweep on ``ssh office`` with twelve
    independent ``k_y`` values, ``Ny=128``, ``Nz=96``, ``Nl=4``, ``Nm=8``,
    and ``240`` RK2 steps per mode;
  - the GPU run over one and two RTX A4000 workers passed exact identity and
    reached ``1.6288309508300018x`` speedup on two GPUs;
  - the CPU process run over ``1,2,4,8`` workers passed exact identity and
    reached ``1.9203793204316209x``, ``3.511221992252178x``, and
    ``5.335095480330332x`` speedup respectively;
  - generated
    ``docs/_static/independent_ky_scan_scaling_cpu_large.{json,csv,png,pdf}``,
    ``docs/_static/independent_ky_scan_scaling_gpu_large.{json,csv,png,pdf}``,
    and combined
    ``docs/_static/independent_ky_scan_scaling_large.{json,csv,png,pdf}``;
  - documented the result in ``README.md`` and ``docs/performance.rst`` as
    the preferred production parallelization path for linear scans,
    quasilinear studies, sensitivity sweeps, and UQ ensembles.
- Next best implementation steps:
  - commit/push the independent-work scaling artifacts and docs;
  - watch CI on the new head;
  - connect the same independent-worker scheduler to quasilinear calibration
    and UQ ensemble scripts so the scaling result directly supports the
    manuscript quasilinear and differentiable-optimization lanes.
- Quasilinear/UQ ensemble parallelization closure:
  - added ``tools/profile_quasilinear_uq_ensemble_scaling.py`` to run late-time
    Cyclone ITG gradient ensembles as independent CPU/GPU workers, compute real
    linear growth/frequency fits, and reduce them to a deterministic
    mixing-length feature observable for UQ/calibration plumbing;
  - initially found that the short default ``t=3.2`` window measured transient
    decay and produced zero quasilinear information, so the profiler defaults
    and tracked sweep were tightened to a late-time ``t=40`` window with
    ``fit_start_fraction=0.5`` and ``fit_end_fraction=0.95``;
  - ran the large ``ssh office`` sweep with six ``R/LTi`` samples, five
    ``k_y`` values, ``Ny=96``, ``Nz=64``, ``Nl=3``, ``Nm=6``, and ``2000`` RK2
    steps per mode;
  - CPU process scaling passed exact serial identity and reached ``1.70x`` on
    two workers, ``2.75x`` on four workers, and ``5.41x`` on eight requested
    workers using six actual ensemble chunks;
  - the two-RTX-A4000 GPU run passed exact serial identity and reached
    ``1.71x`` speedup with about ``86%`` parallel efficiency;
  - generated
    ``docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.{json,csv,png,pdf}``,
    ``docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.{json,csv,png,pdf}``,
    and combined
    ``docs/_static/quasilinear_uq_ensemble_scaling_large.{json,csv,png,pdf}``;
  - documented this as a production independent-work parallelization result
    for quasilinear calibration grids, finite-difference checks, sensitivity
    sweeps, and UQ ensembles, while keeping absolute nonlinear heat-flux
    promotion explicitly out of scope.
- Next best implementation steps:
  - run the focused artifact/doc/test shard and push the QL/UQ scaling tranche;
  - monitor CI on the new head;
  - continue toward production integration by wiring the independent-worker
    scheduler into the actual quasilinear calibration and UQ scripts, while
    preserving the existing prohibition on batched quasilinear scan artifacts
    until per-ky state-extraction identity is separately gated.
- Quasilinear calibration/UQ independent-worker integration:
  - added ``spectraxgk.independent_map`` as an ordered Python-task mapper for
    file-backed calibration, finite-difference, and UQ workloads that are not
    JAX-array ``vmap`` jobs;
  - wired ``tools/plot_quasilinear_saturation_rule_sweep.py`` through the new
    mapper so nonlinear-window case rows and linear-spectrum reductions can be
    evaluated with ``--workers`` while preserving serial report ordering;
  - wired ``tools/plot_quasilinear_candidate_uncertainty.py`` through the same
    mapper so leave-one-geometry-out candidate/UQ holdout rows can be evaluated
    with ``--workers`` while preserving serial report ordering;
  - added identity tests proving worker-parallel reports match serial reports
    for the saturation-rule and candidate-uncertainty gates;
  - documented the claim boundary: this parallelizes calibration/UQ report
    rows, not quasilinear per-ky state extraction, which remains serial until
    its own state-extraction identity gate exists.
- Next best implementation steps:
  - run the focused quasilinear/parallel/docs verification shard and push;
  - regenerate the quasilinear saturation-rule and candidate-uncertainty
    publication artifacts with ``--workers`` so their JSON companions record
    the parallel identity contract;
  - extend the same ``independent_map`` path to finite-difference sensitivity
    and stellarator-optimization UQ ensembles after adding report-identity
    tests for those workflows.
- Stellarator optimization autodiff/UQ worker integration:
  - extended ``central_finite_difference_jacobian`` and
    ``autodiff_finite_difference_report`` with thread-parallel finite-difference
    columns while preserving the serial AD/FD acceptance contract;
  - wired ``optimize_stellarator_itg`` and
    ``compare_stellarator_itg_objectives`` with worker controls for independent
    objective reports plus finite-difference gradient-gate columns;
  - updated the optimization example scripts and documentation to expose
    ``--workers`` and ``--finite-difference-workers`` and to record worker
    metadata in JSON artifacts;
  - regenerated the growth, quasilinear-flux, nonlinear-window, comparison,
    and UQ stellarator optimization artifacts with ``JAX_ENABLE_X64=1`` and
    worker metadata;
  - verified the reduced objectives still pass AD/FD gates, with nonlinear
    heat-flux objective ``1.592690e-01 -> 8.412824e-03`` and late-window
    ``CV=5.830e-03``, ``trend=1.978e-02``.
- Verification for this tranche:
  - ``python -m ruff check src/spectraxgk/autodiff_validation.py src/spectraxgk/stellarator_optimization.py examples/optimization/compare_stellarator_itg_optimizations.py examples/optimization/stellarator_itg_growth_optimization.py examples/optimization/stellarator_itg_quasilinear_flux_optimization.py examples/optimization/stellarator_itg_nonlinear_heat_flux_optimization.py tools/plot_stellarator_optimization_uq.py tests/test_autodiff_validation.py tests/test_stellarator_optimization.py tests/test_plot_stellarator_optimization_uq.py``;
  - ``mypy src/spectraxgk/autodiff_validation.py src/spectraxgk/stellarator_optimization.py``;
  - ``pytest -q tests/test_autodiff_validation.py tests/test_stellarator_optimization.py tests/test_plot_stellarator_optimization_uq.py tests/test_parallel.py``;
  - ``python -m sphinx -b html docs docs/_build/html``.
- Next best implementation steps:
  - commit/push this stellarator autodiff/UQ worker tranche and monitor CI;
  - add the missing per-ky quasilinear state-extraction identity gate before
    allowing worker parallelism inside richer quasilinear scan artifacts;
  - continue the production parallelization lane with large-run profiling of
    remaining worst-offender nonlinear RHS kernels before making new speedup
    claims.
- Quasilinear per-ky state-extraction parallelization gate:
  - added ordered independent-worker execution to ``run_runtime_scan`` for the
    non-batched per-``ky`` path, including quasilinear payload/state extraction;
  - preserved the existing restriction that combined ``--batch-ky`` quasilinear
    scan artifacts remain disabled until a separate batched state-extraction
    identity gate is closed;
  - exposed ``scan-runtime-linear --workers`` and ``--parallel-executor`` for
    independent quasilinear spectra and recorded worker metadata in scan
    summary artifacts;
  - documented the distinction between independent-worker quasilinear spectra
    and combined-``ky`` batching in ``docs/quasilinear.rst`` and
    ``docs/inputs.rst``.
- Verification for this tranche:
  - ``mypy src/spectraxgk/runtime.py src/spectraxgk/runtime_results.py src/spectraxgk/runtime_artifacts.py src/spectraxgk/cli.py``;
  - ``pytest -q tests/test_runtime_runner.py tests/test_quasilinear.py tests/test_runtime_artifacts.py tests/test_cli.py``;
  - ``python -m sphinx -b html docs docs/_build/html``;
  - ``python -m ruff check tests/test_runtime_runner.py tests/test_runtime_artifacts.py tests/test_cli.py``;
  - ``python -m py_compile src/spectraxgk/runtime.py src/spectraxgk/runtime_results.py src/spectraxgk/runtime_artifacts.py src/spectraxgk/cli.py``.
- Next best implementation steps:
  - commit/push the independent-``ky`` quasilinear scan path and monitor CI;
  - generate a small docs artifact that compares serial vs worker-parallel
    quasilinear spectra from the same runtime TOML;
  - only after that, consider a true combined-``ky`` quasilinear extraction
    implementation with a separate serial-identity gate.
- Quasilinear runtime-scan worker artifact:
  - added ``tools/generate_quasilinear_runtime_parallel_gate.py`` as a real
    runtime-scan identity artifact for quasilinear spectra;
  - generated ``docs/_static/quasilinear_runtime_parallel_gate.{json,csv,png,pdf}``
    with ``JAX_ENABLE_X64=1``, two Cyclone ``ky`` points, and two independent
    workers;
  - the artifact passed exact ordered state-extraction identity with
    ``max_abs_error=0`` and ``max_rel_error=0`` for the tracked quasilinear
    spectrum columns;
  - documented the figure and command in ``docs/quasilinear.rst`` with explicit
    wording that timing metadata is for tracking only, not a production speedup
    claim.
- Verification for this tranche:
  - ``python -m ruff check tools/generate_quasilinear_runtime_parallel_gate.py tests/test_generate_quasilinear_runtime_parallel_gate.py``;
  - ``MYPYPATH=src mypy tools/generate_quasilinear_runtime_parallel_gate.py``;
  - ``pytest -q tests/test_generate_quasilinear_runtime_parallel_gate.py tests/test_runtime_runner.py::test_run_runtime_scan_independent_workers_preserve_quasilinear_order tests/test_runtime_artifacts.py::test_write_runtime_linear_scan_artifacts_with_quasilinear_spectrum tests/test_cli.py::test_cmd_scan_runtime_linear_writes_quasilinear_spectrum``;
  - ``python -m sphinx -b html docs docs/_build/html``.
- Next best implementation steps:
  - commit/push the quasilinear runtime worker artifact and monitor CI;
  - move to profiler-backed nonlinear worst-offender cleanup, keeping claims
    separated from the independent-worker linear/quasilinear path;
  - defer combined-``ky`` quasilinear extraction until a dedicated batched-state
    identity implementation is ready.
- Nonlinear hot-path profiler refresh:
  - reran the local CPU full fused nonlinear-RHS trace for the benchmark-size
    Cyclone Miller runtime case with ``Nl=4``, ``Nm=8``, and five warm repeats;
  - refreshed ``docs/_static/full_nonlinear_rhs_trace_summary.json`` with
    ``warm_seconds=3.35e-1`` and unchanged HLO structure (``3343`` HLO lines),
    confirming no new graph-size regression after the independent-worker
    parallelization work;
  - reran local CPU split profiles for grid and spectral Laguerre nonlinear
    modes and regenerated ``docs/_static/nonlinear_rhs_profile_miller_cpu.csv``,
    ``docs/_static/nonlinear_rhs_profile_miller_cpu_spectral.csv``,
    ``docs/_static/nonlinear_rhs_profile_miller.{json,png,pdf}``;
  - current local CPU split: grid ``full_rhs=3.48e-1 s``,
    ``linear_rhs=1.24e-1 s``, ``nonlinear_bracket=9.89e-2 s``; spectral
    ``full_rhs=2.20e-1 s`` and ``nonlinear_bracket=7.65e-2 s``;
  - kept performance documentation scoped: the fastest tracked full RHS for
    this mixed CPU/GPU artifact is still GPU grid, and no new production
    nonlinear speedup claim is made from the bounded CPU refresh.
- Verification for this profiler refresh:
  - ``JAX_ENABLE_X64=0 python tools/profile_full_nonlinear_rhs_trace.py --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml --ky 0.3 --Nl 4 --Nm 8 --repeats 5 --summary-json docs/_static/full_nonlinear_rhs_trace_summary.json``;
  - ``JAX_ENABLE_X64=0 python tools/profile_nonlinear_step_split.py --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml --ky 0.3 --Nl 4 --Nm 8 --repeats 5 --out docs/_static/nonlinear_rhs_profile_miller_cpu.csv``;
  - ``JAX_ENABLE_X64=0 python tools/profile_nonlinear_step_split.py --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml --ky 0.3 --Nl 4 --Nm 8 --repeats 5 --laguerre-mode spectral --out docs/_static/nonlinear_rhs_profile_miller_cpu_spectral.csv``;
  - ``python tools/plot_nonlinear_rhs_profile.py --case cyclone_miller_benchmark_size --title \"Cyclone Miller benchmark-size case\" --out docs/_static/nonlinear_rhs_profile_miller.png --summary-json docs/_static/nonlinear_rhs_profile_miller.json --input \"CPU grid=docs/_static/nonlinear_rhs_profile_miller_cpu.csv\" --input \"CPU spectral=docs/_static/nonlinear_rhs_profile_miller_cpu_spectral.csv\" --input \"GPU grid=docs/_static/nonlinear_rhs_profile_miller_gpu.csv\" --input \"GPU spectral=docs/_static/nonlinear_rhs_profile_miller_gpu_spectral.csv\"``.
- Next best implementation steps:
  - run the bounded artifact/docs test shard and commit this profiler refresh;
  - inspect the linear-RHS cache/layout path next, since the refreshed split
    again identifies linear RHS as the dominant measured CPU sub-kernel;
  - keep nonlinear multi-GPU production decomposition separate from the current
    whole-state sharding correctness gate.
- Linear-RHS Miller term-profile follow-up:
  - generated a bounded active-state CPU split profile for the same Cyclone
    Miller benchmark-size runtime case used in the refreshed nonlinear RHS
    panel:
    ``JAX_ENABLE_X64=0 python tools/profile_linear_rhs_terms.py --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml --ky 0.3 --Nl 4 --Nm 8 --repeats 5 --state z_wave_linear_kick --out docs/_static/linear_rhs_terms_profile_miller_cpu.csv --summary-json docs/_static/linear_rhs_terms_profile_miller_cpu.json``;
  - result: ``full_linear_rhs=2.93e-1 s`` and independently timed terms sum to
    ``4.83e-2 s``; the largest nonzero standalone rows are streaming
    (``7.33e-3 s``), linked ``partial_z`` (``6.39e-3 s``), linked ``|k_z|``
    (``6.18e-3 s``), and hypercollisions (``6.20e-3 s``);
  - interpretation: optimize full-graph layout/fusion and reusable transformed
    state paths next; do not claim standalone-term speedup from this profile.
- IMEX electrostatic fast-path cleanup:
  - factored nonlinear linear-RHS JIT selection into a single helper so the
    explicit nonlinear RHS, nonlinear IMEX fixed-point predictor, and nonlinear
    IMEX post-step field path all choose the electrostatic compiled RHS when
    ``apar=bpar=0``;
  - added a regression test that monkeypatches the generic RHS to fail and
    verifies ``integrate_nonlinear_imex_cached`` uses the electrostatic path for
    adiabatic-electron electrostatic terms;
  - documentation updated in ``docs/numerics.rst`` and ``docs/performance.rst``;
    no new end-to-end speedup claim is made until a fresh profile is recorded.
- Static-quality cleanup / scan fit-signal forwarding:
  - cleared the known broad ``ruff`` blockers in ``src/spectraxgk/runtime.py``
    and ``src/spectraxgk/cli.py`` by removing stale imports;
  - fixed ``scan-linear`` so TOML/argument ``fit_signal`` is forwarded through
    ``run_linear_scan(..., run_kwargs={...})`` instead of being computed and
    dropped;
  - added a CLI regression test that verifies ``fit_signal`` reaches the scan
    run kwargs.
- Repo-wide source ruff cleanup:
  - cleaned stale variables/imports and ambiguous Laguerre-index names across
    ``basis.py``, ``gyroaverage.py``, ``gx_legacy_output.py``, Miller geometry
    helpers, VMEC geometry stub, GX integrators, Krylov selection, and
    secondary runtime helpers;
  - preserved behavior by only removing dead locals/imports or renaming local
    loop/index variables;
  - verification: ``python -m ruff check src/spectraxgk`` is clean, mypy passed
    on touched modules, and the touched-module pytest shard passed.
- CI/runtime compatibility fix:
  - restored the legacy private-helper re-export surface in ``src/spectraxgk/runtime.py`` after the source ruff cleanup removed imports still exercised by the runtime helper tests;
  - added an explicit ``__all__`` so the compatibility surface is intentional and ruff-clean;
  - verification: ``python -m ruff check src/spectraxgk/runtime.py``, ``mypy src/spectraxgk/runtime.py``, and the CI-equivalent runtime pytest shard passed locally.
- Tests/tools static-quality cleanup:
  - made ``python -m ruff check src/spectraxgk tests tools`` clean;
  - scoped ``E402`` per-file ignores to ``tests/*.py`` and ``tools/*.py`` because those files intentionally bootstrap local source paths for direct execution;
  - fixed real lint-exposed issues rather than suppressing them: completed the multispecies linear-RHS shape test, used the geometry-contract override values in the imported-geometry test, restored the missing ``SpectralGrid`` plotting import, and removed dead profiling/table locals;
  - verification: targeted tests for the changed physics/test paths passed, touched tools compiled with ``py_compile``, full source/test/tool ruff passed, and ``git diff --check`` passed.
- Local release checks after this tranche:
  - ``python -m build --sdist --wheel`` passed for ``spectraxgk-1.5.0``;
  - ``python tools/check_repository_size_manifest.py`` passed with tracked size ``44.08 MB``;
  - ``python tools/check_validation_coverage_manifest.py --skip-artifact-check`` passed schema checks and still lists 15 high-priority modules active/open for the package-wide 95% coverage/validation-depth lane.
- Next best implementation steps:
  - monitor the new CI run for ``3c4c435`` and fix any remote-only failure immediately;
  - run a bounded local fast-coverage shard if CI remains queued, then inspect any module below target before adding tests;
  - resume the active ordered plan at parallelization/coverage/refactor with no new nonlinear speedup claims unless backed by fresh profiler artifacts.
- Velocity-parallel ``auto`` backend gate:
  - made ``linear_rhs_parallel_cached(..., parallel.backend="auto")`` resolve
    only to the most complete currently gated Hermite-axis electrostatic route
    (``electrostatic_linear_slices``) when the active linear terms satisfy that
    backend's identity gate;
  - ungated term sets now raise a clear ``NotImplementedError`` instead of
    silently falling back to a serial or partially validated route;
  - added a serial-identity regression for the auto-selected route and a
    collision-term rejection regression for the same entry point;
  - documented the ``auto`` behavior in ``docs/inputs.rst`` so user-facing
    parallelization language matches the implemented validation boundary.
- Verification for this tranche:
  - ``python -m ruff check src/spectraxgk/linear.py tests/test_velocity_sharding.py``;
  - ``mypy src/spectraxgk/linear.py``;
  - ``pytest -q --maxfail=1 --disable-warnings tests/test_velocity_sharding.py tests/test_parallel.py`` with a 300 s local timeout;
  - ``python -m sphinx -W -b html docs docs/_build/html``;
  - ``git diff --check``.
- Next best implementation steps:
  - commit/push the velocity-parallel auto gate and monitor CI;
  - add a small runtime-level TOML/CLI regression proving ``[parallel] backend =
    "auto"`` reaches this gate for eligible linear cases;
  - then move to a profiler-backed linear-RHS layout/cache cleanup before making
    any new performance claim.
- Runtime-level velocity-parallel threading:
  - threaded the runtime ``parallel`` policy through the fixed-step cached
    linear executable path down to ``integrate_linear`` and the gated
    ``linear_rhs_parallel_cached`` route;
  - kept default behavior serial, while non-serial velocity requests now fail
    explicitly on unsupported diffrax, density-assisted fitting, implicit, and
    donated-buffer paths instead of being silently ignored;
  - added focused regressions covering runner forwarding, runtime forwarding,
    non-serial wrapper routing, parallel RHS dispatch inside the cached
    integrator, and explicit unsupported-route errors;
  - documented the fixed-step/``fit_signal="phi"`` release boundary in
    ``docs/inputs.rst``.
- Verification for this tranche:
  - ``python -m ruff check src/spectraxgk/linear.py src/spectraxgk/runners.py src/spectraxgk/runtime.py tests/test_linear_helpers_extra.py tests/test_runtime_runner.py tests/test_runners.py``;
  - ``mypy src/spectraxgk/linear.py src/spectraxgk/runners.py src/spectraxgk/runtime.py``;
  - focused runtime-parallel pytest nodes under a 300 s timeout;
  - ``pytest -q --maxfail=1 --disable-warnings tests/test_runners.py tests/test_linear_helpers_extra.py tests/test_velocity_sharding.py`` under a 300 s timeout;
  - ``python -m sphinx -W -b html docs docs/_build/html``;
  - ``git diff --check``.
- Next best implementation steps:
  - commit/push the runtime-level velocity parallel threading and monitor CI;
  - add a tiny TOML-backed executable smoke test for an eligible
    ``strategy="velocity", backend="auto"`` linear case;
  - only after this gate is green, start the profiler-backed linear-RHS
    layout/cache cleanup and keep all speedup claims tied to fresh artifacts.
- TOML-backed velocity-parallel executable smoke:
  - added an integration smoke that writes a minimal runtime TOML with
    ``[parallel] strategy="velocity"``, ``axis="hermite"``, and
    ``backend="auto"``, then runs ``run_linear_case`` through the executable
    helper path;
  - the test verifies the parsed TOML policy reaches
    ``linear_rhs_parallel_cached`` and uses a grid whose spacing contains the
    requested ``ky=0.1`` mode;
  - this closes the release-level wiring gate from TOML parsing to the
    low-level velocity RHS identity gate.
- Verification for this smoke:
  - ``python -m ruff check tests/test_runtime_runner.py``;
  - ``pytest -q -m integration --maxfail=1 --disable-warnings tests/test_runtime_runner.py::test_runtime_linear_forwards_velocity_parallel_config tests/test_runtime_runner.py::test_run_linear_case_toml_velocity_auto_reaches_parallel_rhs`` under a 300 s timeout;
  - ``git diff --check``.
- Next best implementation steps:
  - commit/push the TOML-backed smoke and monitor CI;
  - move to profiler-backed linear-RHS cache/layout analysis, starting from the
    existing Miller term-profile artifact, before attempting any speedup claim;
  - keep nonlinear production sharding separate until the whole-state identity
    gate is extended into a decomposition with physics diagnostics.
- Electrostatic cached-linear RHS specialization hook:
  - routed serial ``linear_rhs_cached`` and the fixed-step cached linear
    integrator through the existing electrostatic compiled RHS when the Python
    term policy has ``apar=bpar=0``;
  - kept the specialization as a static integrator flag so default
    electromagnetic-capable calls remain unchanged and non-serial velocity
    gates keep their own explicit backend dispatch;
  - added tests proving direct specialized-RHS dispatch and integrator-level
    static-flag forwarding;
  - no new speedup claim is made yet: this closes source wiring so the next
    profiler refresh can measure the production path, not just the standalone
    assembly helper.
- Verification for this source tranche:
  - ``python -m ruff check src/spectraxgk/linear.py tests/test_linear_helpers_extra.py``;
  - ``mypy src/spectraxgk/linear.py``;
  - focused linear-helper pytest nodes under a 300 s timeout;
  - ``pytest -q --maxfail=1 --disable-warnings tests/test_linear_helpers_extra.py tests/test_runners.py tests/test_velocity_sharding.py`` under a 300 s timeout;
  - ``git diff --check``.
- Next best implementation steps:
  - commit/push this production-path specialization hook and monitor CI;
  - refresh the full fused linear-RHS trace using the production
    ``linear_rhs_cached`` path, then update the profiler docs only if the new
    artifact supports a claim;
  - after that, move to nonlinear production sharding with numerical-identity
    and physics-diagnostic gates.
- Production-path fused linear-RHS profiler refresh:
  - changed ``tools/profile_full_linear_rhs_trace.py`` to time and lower the
    production ``spectraxgk.linear.linear_rhs_cached`` entry point rather than
    the lower-level assembly helper;
  - regenerated the local CPU Cyclone Miller initial and active ``z_wave``
    summaries in ``docs/_static/full_linear_rhs_trace*_summary.json``;
  - results: initial ``warm_seconds=1.54e-1`` and active ``z_wave``
    ``warm_seconds=8.38e-2`` with ``source="spectraxgk.linear.linear_rhs_cached"``,
    ``force_electrostatic_fields=true``, and ``2779`` HLO lines;
  - updated ``docs/performance.rst`` to scope these artifacts as production-path
    localization only, and explicitly removed the stale lower-level helper
    speedup wording; GPU production-path traces remain a required next refresh
    before any GPU speedup claim.
- Verification for this profiler refresh:
  - ``python -m ruff check tools/profile_full_linear_rhs_trace.py tests/test_profile_full_linear_rhs_trace.py src/spectraxgk/linear.py tests/test_linear_helpers_extra.py``;
  - ``MYPYPATH=src mypy tools/profile_full_linear_rhs_trace.py`` and
    ``mypy src/spectraxgk/linear.py``;
  - ``pytest -q --maxfail=1 --disable-warnings tests/test_profile_full_linear_rhs_trace.py tests/test_linear_helpers_extra.py::test_integrate_linear_wrapper_enables_electrostatic_field_specialization tests/test_linear_helpers_extra.py::test_linear_rhs_cached_can_use_electrostatic_specialized_jit``;
  - two bounded local CPU profiler runs through
    ``tools/profile_full_linear_rhs_trace.py`` with 300 s timeout;
  - ``python -m sphinx -W -b html docs docs/_build/html``;
  - ``git diff --check``.
- Next best implementation steps:
  - commit/push the profiler-source refresh and monitor CI;
  - run the same production-path fused linear-RHS profiler on ``office`` GPU
    before updating any GPU performance claim;
  - then resume nonlinear sharding production decomposition with identity and
    physics-diagnostic gates.
- Office GPU production-path fused linear-RHS profiler refresh:
  - created a fresh temporary ``office`` clone at commit ``6e5550f`` to avoid
    touching existing dirty worktrees, then ran the production-path fused
    linear-RHS profiler on one RTX A4000 with ``CUDA_VISIBLE_DEVICES=0`` and
    ``jax==0.6.2``/``jaxlib==0.6.2``;
  - refreshed ``docs/_static/full_linear_rhs_trace_gpu_summary.json`` and
    ``docs/_static/full_linear_rhs_trace_gpu_z_wave_summary.json``;
  - results: initial ``warm_seconds=5.13e-3`` and active ``z_wave``
    ``warm_seconds=5.15e-3`` with
    ``source="spectraxgk.linear.linear_rhs_cached"``,
    ``force_electrostatic_fields=true``, and ``2779`` HLO lines;
  - updated ``docs/performance.rst`` to state the refreshed GPU production-path
    evidence while keeping the claim scoped to kernel-localization rather than
    full nonlinear runtime speedup.
- Verification for this GPU refresh:
  - fresh ``office`` clone imported SPECTRAX-GK from ``PYTHONPATH=src`` and
    reported GPU backend with two CUDA devices;
  - ``CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=0 PYTHONPATH=src python3 tools/profile_full_linear_rhs_trace.py --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml --ky 0.3 --Nl 4 --Nm 8 --repeats 5 --summary-json docs/_static/full_linear_rhs_trace_gpu_summary.json``;
  - same command with ``--state z_wave`` and
    ``--summary-json docs/_static/full_linear_rhs_trace_gpu_z_wave_summary.json``;
  - artifacts copied back with ``scp``.
- Next best implementation steps:
  - validate docs/static checks locally, commit/push the GPU artifact refresh,
    and monitor CI;
  - start nonlinear production sharding decomposition only after current CI is
    green or any failure is fixed;
  - keep W7-X zonal/TEM deferred and keep scientific claims scoped to validated
    artifacts.
- Nonlinear sharding final-field/RHS diagnostic gate:
  - extended ``tools/profile_nonlinear_sharding.py`` so each candidate
    sharded final state is also re-evaluated through the nonlinear field solve
    and RHS, not only compared as a raw state array;
  - added a focused regression that perturbs one state entry and verifies the
    diagnostic metric catches both RHS and ``phi`` differences;
  - regenerated the bounded local CPU artifact
    ``docs/_static/nonlinear_sharding_profile.json`` with ``warmups=1`` and
    ``repeats=2``; the single-device control-flow gate passes with zero
    final-state, final-field, and final-RHS diagnostic error;
  - regenerated the two-GPU ``office`` artifact
    ``docs/_static/nonlinear_sharding_profile_office_gpu.json`` from a fresh
    temporary clone with ``nx=4, ny=8, nz=16, nl=4, nm=6, steps=16,
    warmups=1, repeats=3`` and a TensorBoard trace path under
    ``tools_out/profiles/office_gpu_nonlinear_sharding_trace_20260511``;
  - results: both active ``auto`` and ``kx`` sharded GPU candidates preserve
    final state, field solve, and RHS diagnostics exactly in this gate; the
    bounded timing still does not support a nonlinear speedup claim
    (``auto=0.81x``, best ``kx=0.96x``), so whole-state nonlinear sharding
    remains a correctness/profiler artifact while production parallelism should
    continue through scan/ensemble batching and a communication-aware domain
    decomposition.
- Verification for this nonlinear sharding gate:
  - fresh ``office`` clone imported SPECTRAX-GK from ``PYTHONPATH=src`` and
    reported GPU backend with two CUDA devices;
  - ``python tools/profile_nonlinear_sharding.py --out-json docs/_static/nonlinear_sharding_profile.json --sharding auto --sharding-options auto,kx --warmups 1 --repeats 2 --steps 2``;
  - ``PYTHONPATH=src JAX_ENABLE_X64=0 python3 tools/profile_nonlinear_sharding.py --out-json docs/_static/nonlinear_sharding_profile_office_gpu.json --sharding auto --sharding-options auto,kx --nx 4 --ny 8 --nz 16 --nl 4 --nm 6 --dt 0.01 --steps 16 --warmups 1 --repeats 3 --trace-dir tools_out/profiles/office_gpu_nonlinear_sharding_trace_20260511`` on ``office``.
- Next best implementation steps:
  - run bounded lint, type, unit, JSON-artifact, docs, and whitespace checks;
  - commit/push the nonlinear diagnostic-gate tranche and monitor CI;
  - resume the next parallelization step by designing a production
    communication-aware nonlinear decomposition, but do not document any
    nonlinear multi-GPU speedup until fresh profiler artifacts support it.
- CI fast-shard follow-up:
  - added ``tests/test_nonlinear_sharding_artifacts.py`` to the
    ``parallel-autodiff`` quick-test shard so the final-state, final-field, and
    final-RHS diagnostic artifact gates run before wide coverage.
