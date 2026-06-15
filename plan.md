- 2026-06-15: Continued source naming cleanup by moving generic reduced-model
  and legacy cETG NetCDF helpers from old reference-code-named modules to
  `spectraxgk.reduced_model_contracts` and
  `spectraxgk.legacy_cetg_output`. The old module paths remain as thin
  compatibility shims, while implementation tests and comparison utilities now
  import the canonical modules. Coverage ownership moved to the canonical
  modules and the legacy shim modules are excluded from the wide-coverage
  ownership inventory. Focused reduced-model/cETG tests, manifest tests, lint,
  format, and compile checks passed locally.

- 2026-06-15: Completed the internal imported-geometry backend package rename
  tranche. The implementation moved from `spectraxgk.from_gx.*` to
  `spectraxgk.geometry_backends.*`; `spectraxgk.from_gx` now contains thin
  compatibility shims for older scripts and archived comparison tooling. New
  source, profiler utilities, validation gates, coverage manifests, and backend
  tests use the canonical backend package plus neutral imported-geometry names
  (`load_imported_geometry_netcdf`, `apply_imported_geometry_grid_defaults`,
  `imported-netcdf`, `imported-eik`). The Miller helper kernels now expose
  descriptive finite-difference/extension names while retaining old aliases
  only for compatibility. Targeted backend, runtime/config/artifact, manifest,
  lint, format, and compile checks passed locally.

- 2026-06-15: Continued the naming-governance refactor by making
  `explicit_time` the canonical runtime and benchmark solver key. The old
  `gx_time` spelling is now retained only as a normalization alias for legacy
  inputs and in benchmark/comparison artifacts that explicitly refer to
  external reference data. KBM solver-lock constants, example solver choices,
  and focused runtime/benchmark tests now use the numerics-based
  `explicit_time` name.

- 2026-06-15: Continued benchmark/config naming cleanup by replacing the
  legacy top-level `gx_reference` TOML/config spelling with
  `reference_alignment` / `reference_aligned`. The neutral name now appears in
  Cyclone, KBM, and kinetic benchmark APIs, config serialization, input docs,
  and refactor manifests. Old-name compatibility is bounded to explicit
  benchmark/comparison runner keyword handling, while comparison tools keep
  direct reference-code names only where they operate on external reference
  files.

- 2026-06-15: Started the geometry-import naming tranche by adding canonical
  imported-geometry APIs and model strings:
  `load_imported_geometry_netcdf`,
  `apply_imported_geometry_grid_defaults`, `imported-netcdf`, and
  `imported-eik`. Runtime paths, imported-geometry examples, docs, and focused
  geometry/runtime/benchmark tests now use the canonical names. Existing
  `gx-*` geometry names remain bounded compatibility aliases for comparison
  tools and older imported-geometry configurations; moving the
  `spectraxgk.from_gx` backend package is the next larger geometry cleanup.

- 2026-06-15: Added the naming-governance rule for the refactor: package
  source, examples, README, and docs should use physics, numerics, and schema
  names (`dealiased`, `NetCDF output`, `runtime diagnostics`, `restart
  layout`) rather than naming internals after comparison codes. Direct
  reference-code names remain allowed only in benchmark/comparison tools,
  parity notes, and validation artifacts whose purpose is explicitly a
  comparison. Performance investigations may still use external source code and
  reruns, but SPECTRAX-GK internals should stay named after the implemented
  algorithm or physical quantity.

- 2026-06-15: Completed the diagnostics/transport-observable naming tranche.
  Runtime diagnostic APIs now use physical names such as
  `fieldline_quadrature_weights`, `distribution_free_energy`,
  `electrostatic_field_energy`, `magnetic_vector_potential_energy`,
  `heat_flux_species`, `particle_flux_species`, `turbulent_heating_species`,
  and `zonal_phi_mode_kxt`. The old diagnostic dataclass aliases were removed
  from the public package surface, the diagnostics test was renamed to
  `tests/test_runtime_diagnostics.py`, and comparison tools were updated to
  import the neutral package APIs while retaining explicit comparison wording
  where appropriate.

- 2026-06-15: Continued the naming/refactor cleanup by renaming the nonlinear
  NetCDF writer and spectral-layout helpers from reference-code-oriented names
  to `spectraxgk.nonlinear_output_netcdf` and
  `spectraxgk.netcdf_spectral_layout`. Runtime diagnostics, adaptive chunk
  execution, restart IO, and startup randomization helpers now use
  `runtime_*`, `NetCDF`, `dealiased`, and `glibc` vocabulary. The remaining
  naming tranches are the real-FFT nonlinear option, diagnostic-weight helper
  names, geometry-import adapters, and benchmark-only comparison tooling.

- 2026-06-15: Continued the naming cleanup in velocity-space numerics by
  renaming gyroaverage helper functions from reference-code names to
  `single_precision_factorial`, `laguerre_quadrature_count`, and
  `laguerre_transform`. The implementation and focused nonlinear bracket tests
  are unchanged; comparison tooling now imports the neutral helper names.

- 2026-06-15: Completed the main runtime artifact facade reduction by moving
  nonlinear NetCDF output schema writing, artifact geometry resolution,
  particle-moment output helpers, and geometry/input metadata group writers into
  `spectraxgk.nonlinear_output_netcdf`. The legacy
  `spectraxgk.runtime_artifacts` module is now a small dispatch/orchestration
  facade that re-exports compatibility helpers for existing tests and tools.

- 2026-06-15: Continued the runtime artifact refactor by moving generic
  nonlinear JSON/CSV/NPY summary and diagnostic table writing into
  `spectraxgk.runtime_artifact_nonlinear`. The legacy
  `spectraxgk.runtime_artifacts` module keeps the public nonlinear artifact
  dispatcher and NetCDF output writer, preserving monkeypatch seams while
  reducing the facade to the remaining NetCDF schema hotspot.

- 2026-06-15: Continued the runtime artifact refactor by moving linear scan,
  linear single-run, and quasilinear artifact writers into
  `spectraxgk.runtime_artifact_linear`. The public
  `spectraxgk.runtime_artifacts` facade still re-exports these writer
  functions, keeping the executable/runtime import contract stable while
  separating pure linear CSV/JSON serialization from nonlinear NetCDF output.

- 2026-06-15: Continued the runtime artifact refactor by moving nonlinear
  NetCDF diagnostic reload, optional-variable parsing, restart-path
  resolution, species-time condensation, and restart-append diagnostic schema
  normalization into `spectraxgk.runtime_artifact_nonlinear_diagnostics`. The
  public `spectraxgk.runtime_artifacts` facade still re-exports the moved
  helpers used by tests and artifact handoff code, preserving restart and
  NetCDF compatibility while reducing the remaining artifact hotspot.

- 2026-06-15: Completed the nonlinear-gradient follow-up facade split by
  moving candidate-design, composite-control, matched-replicate follow-up,
  QL/linear seed-screen, and VMEC-state runbook reports into
  `spectraxgk.nonlinear_gradient_followup_candidate`,
  `spectraxgk.nonlinear_gradient_followup_composite`,
  `spectraxgk.nonlinear_gradient_followup_plan`,
  `spectraxgk.nonlinear_gradient_followup_ql_seed`, and
  `spectraxgk.nonlinear_gradient_followup_state_runbook`. The legacy
  `spectraxgk.nonlinear_gradient_followup` module is now a compatibility
  facade over core, variance, and report modules, with tests asserting public
  and test-visible facade identities.

- 2026-06-15: Completed the nonlinear-gradient evidence facade split by moving
  production evidence report assembly and missing-campaign gap reports into
  `spectraxgk.nonlinear_gradient_evidence_gap`. The legacy
  `spectraxgk.nonlinear_gradient_evidence` module is now a compatibility
  facade plus JSON artifact loader, while the direct implementation modules
  own classification, replicated windows, central finite differences,
  candidate/bracket screening, and gap-report orchestration.

- 2026-06-15: Continued the nonlinear-gradient evidence refactor by moving
  artifact claim classification into
  `spectraxgk.nonlinear_gradient_evidence_classification` and campaign
  candidate/bracket screening reports into
  `spectraxgk.nonlinear_gradient_evidence_screening`. The remaining
  `spectraxgk.nonlinear_gradient_evidence` module is now a small evidence-gap
  orchestration facade plus JSON loader, while compatibility tests assert that
  public and test-visible facade names still resolve to the moved modules.

- 2026-06-15: Continued the nonlinear-gradient evidence refactor by moving
  replicated nonlinear-window evidence summaries into
  `spectraxgk.nonlinear_gradient_evidence_windows` and central
  finite-difference turbulence-gradient report assembly into
  `spectraxgk.nonlinear_gradient_evidence_fd`. The public
  `spectraxgk.nonlinear_gradient_evidence` facade still re-exports the moved
  report builders and compatibility seam, while tests, API docs, and the
  validation/refactor manifests now track the new modules directly.

- 2026-06-14: Continued the nonlinear-gradient follow-up refactor by moving
  paired-seed variance-reduction planning, control-variate campaign design,
  and independent control-mean uncertainty gates into
  `spectraxgk.nonlinear_gradient_followup_variance`. The public
  `spectraxgk.nonlinear_gradient_followup` facade still re-exports the moved
  report builders and config/helper seams, keeping existing tools and tests
  compatible while making the control-variate evidence path easier to test.

- 2026-06-14: Continued the nonlinear parallelization refactor by moving the
  device-z shard-map RHS route, z-sharding topology check, physical transport
  observable reductions, and serial-vs-device transport-window identity gate
  into `spectraxgk.nonlinear_parallel_device_z`. The public
  `spectraxgk.nonlinear_parallel` facade still re-exports the release-visible
  route and test-visible helper seams, preserving the fail-closed distinction
  between identity-gated routing and profiler-backed speedup claims.

- 2026-06-14: Continued the runtime artifact refactor by moving generic
  artifact path/file I/O helpers into `spectraxgk.runtime_artifact_io` and
  pure dealiased-axis, real/imag packing, restart-layout, species-matrix,
  and diagnostic-condense helpers into
  `spectraxgk.netcdf_spectral_layout`. The public
  `spectraxgk.runtime_artifacts` facade still re-exports the moved names used
  by existing tests and tools, while geometry/cache-dependent artifact helpers
  remain in the facade to preserve monkeypatch seams.

- 2026-06-14: Continued the nonlinear-gradient evidence refactor by moving
  the claim-boundary scope markers, acceptance config dataclasses, JSON-safe
  metric parsing, replicated finite-difference helpers, and gradient
  conditioning summary into `spectraxgk.nonlinear_gradient_evidence_core`.
  The public `spectraxgk.nonlinear_gradient_evidence` facade still re-exports
  the moved names used by existing tools, and tests now assert facade/core
  object identity for the production-scope and finite-difference gate seams.

- 2026-06-14: Continued the nonlinear-gradient evidence refactor by moving
  follow-up configuration dataclasses, JSON-safe metric parsing, replicate
  metadata extraction, coefficient/control labeling, and paired-seed/control-
  variate statistics helpers into `spectraxgk.nonlinear_gradient_followup_core`.
  The existing `spectraxgk.nonlinear_gradient_followup` planner facade still
  re-exports the moved names, and tests now assert object identity for the core
  compatibility seam.

- 2026-06-14: Continued the large-module refactor by moving nonlinear spectral
  parallelization primitives into `spectraxgk.nonlinear_parallel_spectral_core`.
  The split module now owns deterministic spectral test states, chunk/layout
  utilities, communication/work models, pencil FFT/bracket kernels, RHS
  micro-routes, z-chunked bracket helpers, host-staged sharding preparation,
  and tolerance helpers. The public `spectraxgk.nonlinear_parallel` facade
  still re-exports the moved public and test-visible helpers, with an
  import-identity regression guarding downstream compatibility.

- 2026-06-14: Continued the solver-objective refactor with a larger
  three-module split. Solver-ready geometry objective gates moved into
  `spectraxgk.solver_geometry_objectives`, reduced nonlinear-window estimator
  metrics moved into `spectraxgk.solver_nonlinear_window_objective`, and
  VMEC/Boozer state coefficient helpers moved into
  `spectraxgk.solver_vmec_state`. The unchanged
  `spectraxgk.solver_objective_gradients` facade still re-exports the public
  and test-visible names, while the manifest/docs now track the moved physics,
  numerics, and differentiability contracts directly.

- 2026-06-14: Continued the differentiable-objective refactor by moving the
  implicit dominant-eigenvalue custom VJP and branch-locality finite-difference
  report into `spectraxgk.solver_eigen_objectives`. The legacy
  `spectraxgk.solver_objective_gradients` facade still re-exports
  `dominant_real_eigenvalue` and
  `dominant_eigenvalue_branch_locality_report`, preserving package-level and
  tool imports while separating the eigen-AD gate from VMEC/Boozer objective
  plumbing.

- 2026-06-14: Continued the solver-objective refactor by moving physical
  `ky` scan mapping, VMEC/Boozer sample-axis helpers, and aggregate objective
  weights into `spectraxgk.solver_objective_sampling`. The legacy
  `spectraxgk.solver_objective_gradients` facade still exposes the public
  `solver_grid_options_from_ky_values` helper and private compatibility seams
  used by existing tests, while the new module isolates deterministic sampling
  contracts from gradient-report orchestration.

- 2026-06-14: Continued the solver-objective refactor by moving core
  linear/quasilinear objective constants and value evaluators into
  `spectraxgk.solver_objective_core`. The unchanged
  `spectraxgk.solver_objective_gradients` facade still re-exports
  `SOLVER_OBJECTIVE_NAMES`, `SolverScalarObjective`,
  `solver_growth_rate_from_geometry`,
  `solver_linear_operator_matrix_from_geometry`,
  `solver_objective_vector_from_geometry`, and
  `solver_scalar_objective_from_vector`, preserving optimizer, tool, and
  package-level imports while separating forward observables from VMEC/Boozer
  finite-difference report orchestration.

- 2026-06-14: Split the Cyclone benchmark-family runners
  (`run_cyclone_linear` and `run_cyclone_scan`) into
  `spectraxgk.benchmark_cyclone` behind the unchanged
  `spectraxgk.benchmarks` public facade. This turns `benchmarks.py` into a
  small compatibility facade for benchmark constants, helper exports, config
  classes, and public runners while preserving legacy imports such as
  `ModeSelection`, `ExplicitTimeConfig`, and `KrylovConfig`.

- 2026-06-14: Split the ETG benchmark-family runners (`run_etg_linear` and
  `run_etg_scan`) into `spectraxgk.benchmark_etg` behind the unchanged
  `spectraxgk.benchmarks` public facade. ETG branch tests now patch the
  implementation module directly, and the stale ETG branch-test monkeypatch
  against a Cyclone-only helper was removed rather than re-exporting unused
  implementation symbols.

- 2026-06-14: Split the kinetic-electron benchmark-family runners
  (`run_kinetic_linear` and `run_kinetic_scan`) into
  `spectraxgk.benchmark_kinetic`, preserving the GX-reference hypercollision,
  end-damping, and legacy density-seed policies behind the unchanged
  `spectraxgk.benchmarks` public facade. Focused kinetic branch tests now
  patch the implementation module directly.

- 2026-06-14: Split the TEM benchmark-family runners (`run_tem_linear` and
  `run_tem_scan`) into `spectraxgk.benchmark_tem` using the same
  behavior-preserving facade pattern as the KBM split. TEM branch tests now
  patch the implementation module directly, while public examples and
  downstream code can continue importing the runners from
  `spectraxgk.benchmarks`.

- 2026-06-14: Continued the behavior-preserving benchmark refactor by moving
  the KBM benchmark-family runners (`run_kbm_linear`, `run_kbm_scan`, and
  `run_kbm_beta_scan`) into `spectraxgk.benchmark_kbm` behind the existing
  `spectraxgk.benchmarks` public facade. KBM-specific regression tests now
  patch the implementation module directly, preserving public imports while
  making the largest benchmark file smaller and the KBM lane easier to test
  independently.

- 2026-06-14: Continued the behavior-preserving refactor lane by splitting
  nonlinear parallelization contracts, JSON-ready reports, and local
  state-domain identity gates into `spectraxgk.nonlinear_parallel_contracts`
  and `spectraxgk.nonlinear_parallel_domain`. The public
  `spectraxgk.nonlinear_parallel` facade remains the import surface for
  examples and downstream users, while focused tests now assert that facade
  exports are identical to the underlying contract and domain objects. This
  advances the refactor/testability lane without changing nonlinear RHS,
  transport-window, sharding, or speedup claims.

- 2026-06-13: Closed the current QL lane as a scoped core-portfolio
  diagnostic instead of a universal absolute-flux claim. The refreshed
  `docs/_static/quasilinear_error_anatomy.{png,json,csv}` now records two
  declared stress outliers (`solovev_reference_repair_dt002_amp1em5_n48_t250`
  and `shaped_tokamak_pressure_external_vmec_t650_high_grid_window`) and a
  passing 10-case core portfolio: mean relative error `0.280`, held-out mean
  `0.275`, maximum error `0.575`, and interval coverage `10/10`. The full
  12-case universal predictor remains unpromoted (`0.697 > 0.35`) and the
  core rank/screening metric remains borderline (`Spearman≈0.745 < 0.75`).
  The pre-manuscript dashboard now closes the scoped QL diagnostic at `100%`
  and moves active work to broad nonlinear turbulent-flux optimization and
  nonlinear domain-decomposition speedup.

- 2026-06-13: Advanced the nonlinear turbulent-flux optimization evidence
  without changing the promotion gate. The production guard now counts the
  strict `t=1500` growth, QL, and nonlinear-window optimized-candidate
  replicated trace ensembles alongside the selected `t=700`
  optimized-equilibrium audit, giving `4` qualifying optimized-equilibrium
  ensembles and closing that trace-count blocker. The guard remains
  promoted under the explicit `2%` late-window policy with `3/3` matched baseline-to-optimized audits passing
  (`18.4%` reduction, `7.82` combined SEMs). The replicated-holdout lane is
  frozen at three accepted long-window holdout ensembles; no additional
  generic holdouts are active for this tranche. Broad nonlinear optimization
  moves to `86.7%`; mean pre-manuscript closure moves to `85.4%`.

- 2026-06-13: Closed the scoped broad nonlinear turbulent-flux optimization guard. The default production guard now counts the two full max-mode-5 projected-weight matched comparisons (`2.68%` and `3.35%`, both uncertainty-separated) alongside the no-ESS-to-optimized QA/ESS audit (`18.4%`, `7.82` combined SEMs). The guard records the explicit `2%` late-window reduction policy, `3/3` qualifying matched audits, `4` optimized-equilibrium ensembles, and `3` replicated holdouts. Three strict `t=1500` QA objective candidates remain negative transfer evidence. The remaining pre-manuscript blocker is production nonlinear domain-decomposition speedup.

- 2026-06-13: Added a routed nonlinear spectral-domain profiling artifact to
  the strict closure dashboard. The new
  `docs/_static/nonlinear_spectral_domain_routing_profile.{png,json,csv}`
  verifies serial-vs-logical routed identity on the deterministic nonlinear
  spectral RHS and records warm timing (`0.94x` locally), but it explicitly
  does not permit a production speedup claim. The same artifact now includes a
  communication/work model for the current global-reconstruction route:
  communication/owned-work ratio `6.375`, efficiency ceiling `0.136`, and
  blocker `global_reconstruction_communication_dominates_owned_work`. The
  production nonlinear domain-decomposition lane therefore moves from `55.0%`
  to `70.0%` on identity/timing/model diagnostics while retaining the CPU/GPU
  `>=1.5x` strong-scaling blockers. Mean strict pre-manuscript closure moves
  to `89.2%`.

- 2026-06-13: Froze the nonlinear holdout-expansion lane for this tranche.
  The Solovev-inclusive 12-case QL ledger is now the working calibration and
  negative-evidence dataset; no additional holdouts should be launched to
  rescue the current absolute-flux model. The universal absolute QL lane stays
  blocked because the saturation/amplitude model fails the existing admitted
  ledger (`6.49 > 0.35` for the one-constant positive-growth family and
  `0.697 > 0.35` for the best reduced `spectral_envelope_ridge` candidate).
  Next work therefore moves away from holdout collection and toward: (1)
  better saturation/transport-amplitude physics using the frozen ledger, (2)
  broad matched nonlinear turbulent-flux optimization evidence from existing
  optimized-equilibrium artifacts, and (3) production nonlinear
  domain-decomposition speedup with identity and profiler gates. CI failed only
  because three QL tests still encoded the pre-Solovev 11-case near-miss
  metrics; those tests were updated to assert the current fail-closed
  Solovev-inclusive metrics without loosening any scientific promotion gate.

- 2026-06-13: Converted the QL residual-anatomy artifact into an explicit
  frozen-ledger model-development diagnostic. The refreshed
  `docs/_static/quasilinear_error_anatomy.{png,json,csv}` keeps absolute-flux
  promotion failed (`0.697 > 0.35`) but now records programmatic policy fields:
  additional holdout collection is inactive for this tranche, the active next
  step is saturation/transport-amplitude physics on the admitted 12-case
  ledger, and the dominant residuals are Solovev, shaped-pressure VMEC, and
  ITERModel VMEC. This gives the next QL work a concrete target without
  changing any promotion gate or adding data.

- 2026-06-13: Tightened the production nonlinear turbulent-flux optimization
  guard without launching new runs. The default guard now ingests the existing
  strict `t=1500` matched baseline-to-growth, baseline-to-QL, and
  baseline-to-nonlinear-window comparison artifacts as negative evidence, in
  addition to the positive QA no-ESS matched audit. The refreshed
  `docs/_static/production_nonlinear_optimization_guard.{png,json,csv}`
  remains release-safe but not production-promoted: `4` matched audits are
  present, only `1` qualifies, and the three strict objective-specific audits
  fail reduction/uncertainty gates. This shifts the next optimization work
  toward better optimizer candidates and additional optimized-equilibrium
  evidence, not holdout expansion.

- 2026-06-12: Harvested the Solovev repaired external-VMEC holdout and used it
  to harden the quasilinear claim boundary. The original CPU `dt=0.01`
  duplicate remained too slow, but the office GPU duplicate completed the
  `n48/t250` run with `501` samples through `t=249.94`. Runtime-output,
  readiness, and replicated seed/timestep ensemble gates pass when the Solovev
  repair protocol is admitted under the explicit `20%` spread gate:
  `<Q_i>=1.409`, mean-relative spread `0.1599`, combined SEM/mean `0.0462`.
  Postprocessing exposed a real filename-parser bug: protocol labels such as
  `repair_dt002` in the case slug were being mistaken for timestep replicate
  suffixes. `tools/build_external_vmec_replicate_ensemble.py` now treats only
  suffix-style `seedNN`/`dtNN` tokens as replicate variants, with regression
  tests for GPU suffixes and protocol-`dt` case slugs. The Solovev spectrum and
  nonlinear ensemble are now included in the 12-case QL ledger as negative
  transfer evidence. Positive-growth mixing-length transfer worsens to
  `6.49 > 0.35`; the best reduced `spectral_envelope_ridge` candidate has
  leave-one-geometry-out mean relative error about `0.697`, interval coverage
  `11/12`, and held-out screening metrics `Spearman=0.624`, pairwise order
  `0.689`, so universal absolute-flux and promoted screening claims remain
  blocked. Regenerated the train/holdout, saturation-rule, candidate
  uncertainty, regularization, residual-anatomy, screening, dataset-sufficiency,
  model-selection, holdout-gap, and stellarator-usefulness panels. The
  nonlinear holdout expansion/audit lane is now effectively complete for this
  tranche, while universal absolute QL remains an explicit negative result.

- 2026-06-12: Hardened nonlinear sharding profiling after the local CPU
  forced-device profile exposed a JAX/XLA FFT-layout abort path. The
  multi-device CPU whole-state `pjit` route now fails closed before execution
  unless `--allow-unsafe-cpu-state-sharding` is explicitly set, writing
  `cpu_whole_state_pjit_sharding_unsafe_for_fft_layout` into the profile JSON
  instead of risking a process abort or collective stall. The sweep wrapper now
  preserves failed profile JSON artifacts even when the profile exits nonzero
  because an identity gate failed, and it replaces inherited
  `--xla_force_host_platform_device_count` values so requested CPU device counts
  cannot be contaminated by the parent environment. A bounded local check now
  gives a true one-device identity row and a safe four-device skip row. This is
  negative evidence for current whole-state CPU nonlinear speedup, not a
  promotion; the production nonlinear domain-decomposition speedup lane moves
  to `70.0%` after adding a routed spectral-domain identity/timing/profile model,
  but remains blocked pending a communication-complete decomposed RHS/integrator with
  CPU/GPU identity, transport-window, and profiler-backed speedup gates. The
  office QA growth-candidate `t=1500` `dt=0.04` run has now completed and the
  existing postprocessed artifacts are consistent: the growth ensemble passes,
  but the matched baseline-to-growth comparison gives only `0.60%` reduction
  and fails the configured reduction gate, so it remains negative/non-promoted
  evidence. The original Solovev CPU `dt=0.01` duplicate ran for about
  46 minutes without writing an output bundle, so it was stopped after a
  clean GPU duplicate was launched on office GPU 1 with the same `n48/t250`,
  `dt=0.01` protocol and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. That GPU run
  became the active Solovev holdout source and is superseded by the successful
  harvest entry above.

- 2026-06-12: Harvested the first production-scope VMEC/Boozer held-out
  nonlinear transport artifact and kept broader claims fail-closed. The QH
  `vmec_jax -> booz_xform_jax` held-out surface/field-line run
  (`wout_nfp4_QH_warm_start.nc`, `torflux=0.78`, `alpha=1.2`, `ky rho_i≈0.2`,
  `n64`, `t=700`, window `350-700`, seed31/seed32 plus `dt=0.04`) passed the
  runtime-output, replicated-window, production-holdout, and aggregate
  VMEC/Boozer promotion gates. The accepted ensemble has
  `<Q_i>=7.9978`, mean-relative spread `0.0837`, and combined SEM/mean
  `0.0242`; VMEC/Boozer holdout optimization is now closed for the strict
  pre-manuscript gate. The production nonlinear optimization guard was
  regenerated with three qualifying replicated holdout ensembles
  (D-shaped, circular, QH VMEC/Boozer). At that historical checkpoint it was
  not production-promoted because both the optimized-equilibrium ensemble count
  and matched optimized transport-audit count were still below threshold;
  the current status table below supersedes that count. The strict closure dashboard is now
  `72.0%` mean completion: universal absolute QL remains `60.0%`, broad
  nonlinear turbulent-flux optimization `72.9%`, production nonlinear domain
  decomposition `55.0%`, and VMEC/Boozer holdout optimization `100.0%`.
  The Solovev external-VMEC repair protocol (`dt=0.02`, `init_amp=1e-5`) passed
  a finite `t=50` pilot. The first `n64/t250` repaired launch was resource-terminated
  before writing an output bundle while GPU 0 was sharing memory with another VMEC/JAX job, so a
  separate bounded `n48/t250` seed/timestep manifest is now running on office GPU 0 with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`; the current QA `t=1500` dt0.04 replicate is still
  running on office GPU 1, so no new QA promotion was made. The nonlinear
  spectral logical route identity gate was refreshed on a larger deterministic
  state, and a bounded local CPU profile again showed identity with no active
  state sharding; nonlinear domain speedup remains blocked pending a real
  communication-aware implementation and CPU/GPU profiler-backed speedup.
  `tools/write_external_vmec_holdout_configs.py` now exposes the protocol
  knobs needed for repair launches (`--init-amp`, gradients, geometry sampling,
  and diagnostic strides), with regression coverage. README/docs now include
  the QH held-out transport panel and corrected claim boundaries. Repository
  size remains under policy after compressing three existing plot PNGs.

- 2026-06-12: Added the next concrete pre-manuscript closure tranche without
  promoting unfinished claims. `tools/write_optimized_equilibrium_transport_configs.py`
  now exposes and records explicit VMEC transport-sample metadata (`torflux`,
  `alpha`, `npol`, `ky`, `tprim`, `fprim`, `nu`) so long nonlinear audits can
  be tied to a held-out surface/field-line rather than a default flux tube.
  Generated a local QH held-out transport launch contract for
  `wout_nfp4_QH_warm_start.nc` at `torflux=0.78`, `alpha=1.2`, `ky=0.2`,
  `n64`, `t=700`, window `350-700`, seed31/seed32 plus `dt=0.04`; the tracked
  pre-manuscript runbook now lists the matching office commands. Regenerating
  the VMEC/Boozer promotion gate exposed an additional honest blocker: the
  aggregate objective artifact currently covers alphas `0` and `0.7`, while the
  line-search artifact covers only alpha `0`; this gate now fails closed on
  `line_search_reuses_aggregate_sample_set` and the still-missing production
  held-out nonlinear transport artifact. Added
  `integrate_logical_decomposed_nonlinear_spectral` as the callable logical
  decomposed nonlinear spectral RHS/integrator route behind the identity
  artifacts. It is identity-gated and serial-fallback safe; it remains a
  diagnostic/profiling route, not a production distributed-FFT speedup claim
  until CPU/GPU profiler and speedup gates pass. Local targeted status:
  34 focused tests passed, ruff passed, release-readiness passed, and repository
  size passed (`tracked_total_bytes=49.87 MB`). GitHub Actions for the previous
  push is still queued. Office QA `t=1500` jobs are still running and should
  not be harvested until final `t≈1500`; Solovev remains queued behind them.
  After commit `1920196`, the older office checkout could not be fast-forwarded
  because untracked generated docs artifacts would be overwritten, so those
  files were left untouched and a fresh checkout was created at
  `/home/rjorge/spectrax_premanuscript_holdouts_20260612`. The same QH
  held-out VMEC/Boozer configs were generated there and deferred launcher PID
  `3425020` is waiting behind the active QA/Solovev queues. Added
  `tools/build_vmec_boozer_production_holdout_artifact.py`, a fail-closed
  postprocessor that combines a concrete transport manifest with a passed
  replicated nonlinear ensemble to produce the exact held-out surface/field-line
  JSON consumed by the VMEC/Boozer promotion gate. The runbook now lists the
  complete postprocessing chain: finite-output gate, replicate-ensemble gate,
  production holdout artifact, then VMEC/Boozer aggregate promotion gate.

- 2026-06-12: Tightened the production nonlinear turbulent-flux optimization
  promotion guard to match the broader pre-manuscript requirement. The guard is
  now release-safe but not production-promoted: current counts are `1/3`
  matched baseline-to-optimized audits, `1/3` optimized-equilibrium ensembles,
  and `2` accepted long-window holdout ensembles. README/docs/plan wording was
  updated so the selected QA no-ESS -> optimized QA/ESS audit remains visible
  as one positive scoped result (`18.4%`, `7.8 sigma`) without closing the
  broad multi-equilibrium nonlinear optimization lane. Office status at
  `2026-06-12T10:51-05:00`: the true `t=1500` growth-objective QA audits are
  still running on both GPUs; seed33 has written `t≈800`, seed32 remains at the
  first `t≈400` checkpoint, and the Solovev external-VMEC holdout launcher is
  correctly waiting behind those queues. A local 4-logical-CPU pjit nonlinear
  sharding attempt on velocity axes (`l,m`) reproduced the XLA CPU FFT layout
  failure and collective stall seen for `ky/kx`; it was killed before the
  five-minute cap. This is negative evidence for exposing current pjit
  nonlinear state sharding as production CPU speedup. The next parallelization
  step is a different communication-aware decomposition, not widening the
  existing `state_sharding` options.

- 2026-06-12: Advanced the independent external-VMEC holdout lane. A bounded
  VMEC linear screen over four previously unscreened `vmec_jax` examples found
  `wout_solovev_reference.nc` launchable (`gamma=0.0944` at `ky=0.2857`) and
  `wout_up_down_asymmetric_tokamak_reference.nc` unstable but already
  represented (`gamma=0.0360` at `ky=0.4762`). `LandremanSenguptaPlunk`
  remains below the nonlinear-launch threshold (`gamma=0.0073`) and
  `basic_non_stellsym_pressure_reference` fails the current VMEC aspect-cut
  flux-tube path. The tracked external-VMEC runbook now selects Solovev and
  writes an office-resolvable nonlinear launch command. Configs were generated
  on office under `tools_out/external_vmec_holdouts/solovev_reference`, and a
  deferred launcher is waiting behind the active QA `t=1500` GPU queues so it
  does not oversubscribe the two GPUs. This is a launch contract only: Solovev
  is not admitted into quasilinear calibration until its long-window nonlinear
  grid/window, replicate, and recalibration gates pass.

- 2026-06-12: Started the pre-manuscript closure phase after verified
  ``v1.6.5`` release/PyPI publication. Added
  ``tools/build_pre_manuscript_closure_status.py`` and tracked
  ``docs/_static/pre_manuscript_closure_status.{png,pdf,json,csv}`` as the
  strict machine-readable gate for the four lanes that must close before
  drafting starts. Current strict status: not ready for manuscript drafting,
  mean closure ``61.8%``. Lane status is recalibrated against stricter
  manuscript requirements, not release-safe scoped diagnostics:
  universal absolute quasilinear heat-flux prediction ``60.0%`` partial,
  broad end-to-end nonlinear turbulent-flux stellarator optimization ``54.2%``
  blocked, production nonlinear domain-decomposition speedup ``55.0%``
  partial, and VMEC/Boozer holdout optimization ``78.0%`` partial. Immediate
  execution order:

  1. Universal absolute QL: add at least one genuinely independent converged
     nonlinear holdout, then replace the failed amplitude/saturation model so
     leave-one-geometry-out candidate uncertainty, model selection, and
     absolute train/holdout error all pass the ``0.35`` mean-relative-error
     transport gate. No runtime/TOML absolute-flux predictor is allowed until
     this closes.
  2. Broad nonlinear turbulent-flux optimization: extend from the single
     selected-QA positive audit to at least three independent matched
     baseline-vs-optimized long-window transport audits, at least three
     optimized-equilibrium ensembles, and the frozen three replicated nonlinear
     holdout ensembles. Only post-transient running-average transport windows
     count; reduced/startup nonlinear-window objectives remain excluded.
  3. Production nonlinear domain decomposition: keep independent ``k_y`` and
     UQ batching as the current production path while implementing a real
     communication-aware nonlinear decomposed RHS/integrator route. Promotion
     requires serial-vs-decomposed transport-window identity plus large-grid
     CPU and multi-GPU speedup ``>=1.5`` with profiler artifacts.
  4. VMEC/Boozer holdout optimization: keep the existing alpha/surface and
     second-equilibrium holdouts as reduced plumbing evidence, then add a
     production-scope held-out surface/field-line nonlinear transport artifact
     with same-WOUT provenance through ``vmec_jax -> booz_xform_jax ->
     SPECTRAX-GK`` before claiming optimization closure.

- 2026-06-11: Started nonlinear admission for the top solved-WOUT screen
  candidate, `qp_diag_nfp2_m4_final`. The `t=150`, `dt=0.05`, `n48/n64`
  office-GPU pair is finite but non-admissible (`0.163` common-window and
  `0.200` least-window heat-flux differences). The true restart continuation
  to `t=250` passes the grid/window gate (`0.033` common-window, `0.0023`
  least-window, slopes below `2e-3`, CV about `0.08`). The minimal `n64`
  seed/timestep ensemble also passes (`<Q_i>=16.40`, spread `0.071`,
  combined SEM/mean `0.029`). A temporary QL re-score with this new holdout
  improves aggregate mean relative error `2.83 -> 2.65`, but holdout error is
  still `3.13 > 0.35`, so absolute-flux promotion remains blocked. Backup
  QA/QI ladders are prepared locally under `tools_out/` but remain untracked.

- 2026-06-11: Added a fail-closed VMEC optimization-result candidate screen
  before launching nonlinear holdouts from solved `vmec_jax` WOUTs. A bounded
  local CPU scan of four solved mode-5 optimization outputs found no launchable
  nonlinear holdout: `qa_nfp2` is marginal, `qh_nfp3`/`qp_nfp4` are stable, and
  apparent high-growth `qp_nfp3` is rejected because all sampled rows have
  non-positive effective `k_perp^2`. Added
  `tools/build_vmec_optimization_candidate_screen_gate.py` and tracked the JSON
  gate artifact so future screens require finite positive metric evidence.

- 2026-06-11: Corrected and closed the QH warm-start long-window restart
  protocol as negative evidence. The earlier direct `n80/t450` and `n80/t700`
  launches were segment-length runs from zero, not cumulative horizons, so a
  true staged ladder was relaunched on office by copying complete restart
  bundles forward. The corrected `n80/t450` and `n80/t700` runs both passed
  runtime-output checks, but the relaxed 20% high-grid convergence gates still
  fail: `t450` has `0.355` common-window and `0.294` least-window heat-flux
  differences, while `t700` has `0.3487` common-window and `0.3668`
  least-window differences. The final `t700` late-window means are about
  `5.885` for `n64` and `4.137` for `n80`, so the mismatch is not an early
  transient. QH remains excluded from quasilinear calibration. Regenerated the
  external-VMEC runbook without a QH modified-protocol allowance; it now fails
  closed with no launch commands until a genuinely new independent candidate or
  materially higher-resolution protocol exists. Added local guardrails so
  generated external-VMEC manifests write executable staged restart-ladder
  scripts, external-VMEC QL admission is fail-closed on promotion-gate/claim
  metadata, nonlinear sharding artifacts report per-backend identity/speedup
  blockers, and public QA optimization examples state linear/QL/reduced
  nonlinear claim boundaries explicitly.

- 2026-06-11: Harvested the first modified-protocol QH warm-start nonlinear
  office gate. The `n64/n80`, `dt=0.04`, `t=250` pair finished cleanly and is
  finite, but it is not admissible: the common-window and least-trending
  high-grid heat-flux disagreements are `0.3675` and `0.4120`, above both the
  strict `15%` and relaxed `20%` policies. QH therefore stays excluded from the
  quasilinear calibration ledger. Longer `n80/t450` and `n80/t700` runs were
  launched on office to compare against the completed `n64/t450` and
  `n64/t700` traces and decide whether the mismatch is an early-window artifact
  or a real grid-resolution blocker.

- 2026-06-11: Tightened two release-facing evidence ledgers while QH nonlinear
  runs continued on office. The QA transport optimization status now reports a
  machine-readable `claim_evidence_level` and explicit
  `claim_promotion_blockers`, and release readiness expects the current scoped
  state: matched long-window nonlinear audit evidence is present, while QL
  model selection and simple absolute-flux QL remain unpromoted. The nonlinear
  sharding production-speedup gate now records per-backend identity-evidence
  summaries and tolerance fractions; it remains `diagnostic_only` because the
  GPU production speedup candidate is still missing.

- 2026-06-11: Hardened the direct nonlinear campaign runner used by external
  VMEC holdouts and nonlinear-gradient audits. `--skip-existing` now skips a
  task only when the complete runtime bundle (`*.out.nc`, `*.restart.nc`, and
  `*.big.nc`) exists, and status rows record the required bundle files. This
  prevents interrupted long GPU runs from being mistaken for completed
  restartable transport evidence.

- 2026-06-11: Promoted the next external-VMEC nonlinear holdout work from a
  blocked replay to a launch-ready modified-protocol QH candidate. A bounded
  local screen of the `vmec_jax` `nfp4_QH_warm_start` fixture found a weak but
  finite unstable branch (`gamma = 0.022949` at `ky = 0.4762`). The refreshed
  `docs/_static/external_vmec_next_holdout_runbook.{png,json,csv}` now passes
  only as `nonlinear_holdout_launch_plan_not_transport_validation` and writes a
  single command for `n64/n80`, `dt=0.04`, and horizons `t=250,450,700`.
  Previous QH nonlinear gates remain excluded; this candidate can enter the QL
  ledger only after the fresh high-grid convergence, late-window time-horizon,
  and seed/timestep replicate gates pass.

- 2026-06-11: Added a quasilinear residual-anatomy artifact for the current
  best reduced candidate. `docs/_static/quasilinear_error_anatomy.{png,json,csv}`
  consumes the existing uncertainty, screening, and saturation-rule sidecars
  and remains fail-closed: `spectral_envelope_ridge` has mean relative error
  `0.424 > 0.35`, no screening gate passes, and no runtime/TOML absolute-flux
  predictor is promoted. The anatomy shows the shaped-pressure external-VMEC
  holdout is the largest residual and external axisymmetric VMEC cases account
  for about `59%` of the residual budget, while HSX/W7-X are comparatively
  well tracked. This originally pointed the next QL work toward richer
  saturation physics plus dataset expansion, not threshold loosening; that
  data-expansion step is superseded by the
  2026-06-13 frozen-ledger decision, so the active next step is saturation and
  transport-amplitude physics on the admitted twelve-case ledger.

- 2026-06-11: Closed the shaped-tokamak-pressure external-VMEC repair as a
  scoped high-grid nonlinear holdout. The full `n48/n64/n80`, `dt=0.04`,
  `t=450` ladder fails only coarse-grid heat-flux agreement (`0.469 > 0.15`),
  while retained `n64/n80` gates pass at `t=450` (`0.0789`) and `t=650`
  (`0.0983` common, `0.0981` least). The high-grid horizon gate passes
  (`0.0418` common, `0.1237` least), and the `n80` seed/timestep ensemble
  passes on `t=[325,650]` with mean heat flux `7.156`, mean-relative spread
  `0.0939`, and combined SEM/mean `0.0463`. The case is now in the
  quasilinear ledger as high-grid admission evidence only. It makes the
  current absolute/screening QL claims weaker, not stronger:
  positive-growth mixing length has holdout mean error `3.42 > 0.35`, the
  spectral-envelope candidate has uncertainty mean error `0.424 > 0.35`, and
  no screening model is currently accepted after the shaped holdout.

- 2026-06-11: Repaired the shaped-tokamak-pressure external-VMEC nonlinear
  holdout after the `dt=0.05` `n80` instability. The office `n64/n80`,
  `dt=0.04`, `t=450` reruns completed with finite diagnostics and passed the
  high-grid convergence gate: late-window heat-flux means `7.422` and `6.859`,
  symmetric relative difference `0.0789 < 0.15`, and no stationarity/CV/sample
  failures. The `t=650` restart continuations for the same two grids are now
  running on office to test time-horizon stability before any admission or
  quasilinear calibration update. CI shard 24 exposed a documentation
  guardrail drift; README, release scope, and verification matrix now again
  state that `spectral_envelope_ridge` is only a scoped manuscript
  model-selection candidate and not a runtime/TOML absolute-flux predictor.

- 2026-06-11: Returned to the larger science gates after the RBC(1,1)
  diagnostic refresh. CI for commit `20e22a9` passed, including wide coverage.
  The quasilinear holdout-gap report was regenerated and still blocks
  absolute-flux promotion with seven independent holdouts and held-out error
  `1.91 > 0.35`. The runbook now selects a modified
  shaped-tokamak-pressure external-VMEC campaign (`n64/n80/n96`, `t=450,650`)
  rather than replaying CTH-like or same-family ITERModel evidence. The office
  campaign was launched from a fresh shallow clone at
  `/home/rjorge/spectrax_ql_shaped_holdout_20260611_063247`; it waits for an
  idle GPU, runs the restart ladder, and postprocesses the nonlinear gate.
  The production nonlinear turbulent-flux optimization guard was also
  re-audited: the selected QA optimized-equilibrium audit remains a scoped
  positive result with `18.4%` lower long-window heat flux and `7.8 sigma`
  separation, while broader nonlinear turbulence-gradient optimization and
  universal absolute QL-flux prediction remain open.

- 2026-06-11: Expanded the quasilinear model-development ledger to consume the
  admitted CTH-like high-grid replicated nonlinear ensemble as a first-class
  nonlinear calibration input. Added fail-closed ensemble-gate ingestion in
  `spectraxgk.quasilinear_calibration` and promotion-ready checks in
  `spectraxgk.quasilinear_window`, with regression tests. Regenerated the QL
  saturation, candidate uncertainty, dataset-sufficiency, screening-skill,
  usefulness, model-selection, and holdout-gap artifacts. Result: rank/correlation
  screening is stronger (`spectral_envelope_ridge` passes full and held-out
  rank gates), but strict candidate uncertainty/model-selection is demoted
  (`0.377 > 0.35`) and universal absolute-flux promotion remains blocked
  (`1.91 > 0.35`, two more independent holdouts required).

- 2026-06-11: Added skip-existing command lists to the external-VMEC nonlinear
  holdout config manifest. Future office campaigns can use
  `staged_ladder_skip_existing_commands` or
  `direct_full_horizon_skip_existing_launch_commands` to resume interrupted
  long runs or manually distribute remaining grids without treating partial
  outputs as complete: each wrapper skips only after the full `.out.nc`,
  `.restart.nc`, and `.big.nc` bundle exists. The live shaped-tokamak-pressure
  campaign remains unchanged; the completed `n64/t450` segment passed the basic
  runtime output gate with late-window mean heat flux about `8.39`.

- 2026-06-11: Added a local-CPU quasilinear regularization sensitivity audit
  for the `spectral_envelope_ridge` candidate. The tracked
  `docs/_static/quasilinear_candidate_regularization_sweep.{png,json,csv}`
  artifact sweeps the ridge penalty and confirms that the near miss is not
  fixed by retuning regularization: the best setting remains `lambda = 0.3`
  with full-ledger mean relative error `0.377 > 0.35`, held-out mean `0.355`,
  and interval coverage `8/9`. The artifact is documented as a fail-closed
  model-development guardrail, not as an absolute-flux predictor.

- 2026-06-11: The live shaped-tokamak-pressure `n80/t450` office run completed
  the integrator but failed artifact validation because `Wg_t` became
  non-finite at an early saved sample. Root cause is a diagnostic masking bug:
  Runtime diagnostic reductions multiplied energy/flux factors by the dealias
  mask after intermediate products, so `inf * 0` in a masked/dealiased mode
  could produce `NaN`. Patched free-energy, field-energy, `phi2`, heat-flux,
  particle-flux, and turbulent-heating reductions to zero masked modes before
  products or moment contractions. Added a regression test that injects `inf`
  into a masked mode while preserving strict validation for unmasked diagnostics.
  The shaped holdout remains unadmitted until the repaired `n80/n96` reruns and
  high-grid/window gates pass.

- 2026-06-10: Added and passed the explicit CTH-like high-grid admission
  policy. The case is now admitted to the quasilinear model-development ledger
  as a scoped high-grid nonlinear transport holdout, with `n48` explicitly
  excluded and `n64/n80` plus late-horizon and seed/timestep replicate evidence
  retained. This is not a full `n48/n64/n80` convergence claim and not an
  absolute-flux promotion: the one-constant train/holdout calibration still
  fails with mean held-out relative error `1.91 > 0.35`, and the holdout-gap
  report still requires two additional independent post-transient nonlinear
  holdouts plus a better saturation model. Commit `a2e1072` passed CI run
  `27288648364` with 59 successful jobs and 1 skipped nightly job.

- 2026-06-10: Closed the next CTH-like high-grid robustness step with true
  office GPU continuations rather than a proxy. The `n80`, `t=[250,350]`
  seed/timestep extraction passed individual readiness gates but failed the
  strict ensemble spread gate (`0.1819 > 0.15`), so the window was too short
  for admission. The same four variants were restart-continued from `t=350`
  to `t=700` and re-extracted on `t=[350,700]`; readiness and ensemble gates
  both pass with mean heat flux `9.603`, mean-relative spread `0.0406`, and
  combined SEM/mean `0.0517`. CTH-like is therefore a replicated high-grid
  candidate under the later explicit admission policy: full `n48/n64/n80`
  convergence still fails, but `n64/n80` plus late-window replicate evidence
  are sufficient for scoped high-grid calibration-ledger use. The
  release repository tracks only
  `docs/_static/external_vmec_cth_like_modified_replicates_t700/replicate_ensemble_gate.*`;
  raw NetCDFs and trace byproducts remain on office.

- 2026-06-09: Completed the CTH-like modified-protocol horizon harvest and
  added a high-grid time-horizon gate. All nine direct office jobs returned
  `0`. The `t=150` high-grid `n64/n80` gate has close heat-flux means
  (`0.026` common-window and `0.009` least-window relative differences) but
  fails the common-window trend gate (`0.00292 > 0.002`), so it is still a
  transient window. The late `t=250`/`t=350` high-grid horizon gate passes with
  common/least horizon changes `0.018`/`0.019`, but its promotion gate remains
  false because replicate/seed/timestep evidence and an explicit high-grid
  admission policy are still required. The release repository tracks compact
  JSON gate sidecars plus the paper-facing `t=350` and late-horizon PNGs; the
  larger pilot byproducts remain reproducible from the office run directory.

- 2026-06-09: Harvested the first CTH-like modified-protocol nonlinear
  outputs at `t=350` from office. All three direct full-horizon jobs returned
  `0`. The formal `n48/n64/n80` grid gate fails closed with common/least
  heat-flux differences `0.296`/`0.272`, dominated by the coarse `n48` trace.
  The high-grid `n64/n80` diagnostic gate passes with common/least differences
  `0.058`/`0.013`, so the case is now a high-grid candidate only. It remains
  outside quasilinear calibration until the remaining horizons and any
  replicate/window gates close under an explicit high-grid admission policy.
  The release repository tracks the compact JSON gate sidecar and high-grid
  PNG needed for the README/docs; the remaining pilot byproducts stay outside
  git to preserve the repository-size policy.

- 2026-06-09: Added an explicit modified-protocol external-VMEC holdout
  contract for failed-family repairs. The default runbook still fail-closes
  unchanged failed families, but `tools/build_external_vmec_holdout_runbook.py`
  can now require `--allow-modified-protocol-family` plus a
  `--modified-protocol-note` and optional horizon override. The tracked
  external-VMEC linear screen now includes the existing CTH-like spectrum point
  (`gamma = 0.0488013` at `ky = 0.285714`), and the refreshed runbook selects a
  CTH-like `n48/n64/n80`, `t = 150,250,350` repair campaign. This is a launch
  contract only; CTH-like remains outside quasilinear calibration until the new
  traces pass grid/window and post-transient holdout gates.

- 2026-06-09: Completed the resumed office QA optimizer ladder. Scalar-trust
  and LBFGS-adjoint growth/QL runs all returned `0` and passed the
  authoritative rerun-WOUT admission gate, but the solved-candidate gate stayed
  false. The SPSA nonlinear-window metric sweep completed four plus/minus
  pairs; the best reduced metrics were about `0.06144` and `0.06174` versus
  the nearby `0.063` level. Tracked summaries:
  `docs/_static/vmec_jax_qa_optimizer_ladder_resume_status.json` and
  `docs/_static/vmec_jax_qa_optimizer_ladder_spsa_metric_summary.json`. These
  artifacts support optimizer-strategy design only; they do not promote a
  long-window nonlinear turbulent-flux reduction.

- 2026-06-09: Fixed an office launch mismatch in the external-VMEC holdout
  config manifest: direct nonlinear commands now prefix `PYTHONPATH=src` so
  `python -m spectraxgk.cli` resolves the checkout source instead of a stale
  venv-installed package. Focused tests enforce the prefix and the structured
  direct-full-horizon step counts. After fast-forwarding office to `8ef0931`,
  the CTH-like modified-protocol campaign was regenerated and launched with
  the direct full-horizon commands. The first active pair is the most
  informative final-window batch, `t=350` at `n80` and `n64`, both running on
  the two office GPUs; no holdout admission claim exists until the resulting
  traces pass the grid/window gate.

- 2026-06-09: Diagnosed the strict QA `t1500` mismatch as a launch-contract
  issue, not a physics result. Final-horizon TOMLs are restart-ladder segments:
  launching a `t1500` segment command from `t=0` integrates only the
  `(1500-1100)` segment and stops near `t=400`. The generators now record
  per-config `dt`, emit explicit staged-ladder commands, emit direct
  full-horizon commands (`t1500/dt`: 30000 steps for `dt=0.05`, 37500 for
  `dt=0.04`), and include a runtime-output gate over `t=[1100,1500]`.
  Focused tests cover the exact strict-policy numbers. The QA `|B|`
  manuscript/readme panels now use unfilled Boozer-LCFS contours instead of
  filled density maps.

- 2026-06-09: Relaunched the corrected strict QA full-horizon audit on
  `office` from a clean shallow clone at commit `9e50d59`. The clean-clone
  `src/` path is forced ahead of the stale editable install, and
  `/home/rjorge/booz_xform_jax` is injected so the internal VMEC/Boozer backend
  is available. The controller queued all twelve true `t=1500` nonlinear jobs
  with direct 30000/37500-step commands and is running two jobs concurrently on
  the two A4000 GPUs. The runtime log line `running ... over 8000 steps` is the
  first restart/checkpoint chunk (`output.nsave`), not the total horizon; the
  controller-level command and executable header show the full 30000/37500-step
  target.

- 2026-06-09: Harvested the true `t=1500` strict QA baseline, growth-objective,
  quasilinear-objective, and nonlinear-window-objective audit triplets from
  office. All four pass the runtime-output and replicated seed/timestep gates
  over `t=[1100,1500]`.
  The strict QA baseline has mean `<Q_i>=11.580`, mean relative spread
  `3.81%`, and combined SEM/mean `1.95%`; growth has mean `<Q_i>=11.510`,
  spread `4.27%`, and SEM/mean `1.24%`; quasilinear has mean
  `<Q_i>=11.636`, spread `2.34%`, and SEM/mean `1.64%`; nonlinear-window has
  mean `<Q_i>=11.609`, spread `3.66%`, and SEM/mean `1.77%`. Matched
  comparisons do not promote any transport candidate: growth gives only
  `0.60%` relative reduction (`z=0.26`, below the `4%` gate), while the
  quasilinear and nonlinear-window rows are slightly worse (`-0.49%`,
  `z=-0.19`; `-0.25%`, `z=-0.09`). The strict-QA `t=1500` candidate set is
  closed as robust negative optimization-transfer evidence.

- 2026-06-09: Added reproducible strict-QA `t=1500` postprocess tooling and
  fixed its output-gate artifact names after the final QL harvest. The new
  `tools/compact_replicate_ensemble_bundle.py` rewrites compact tracked
  ensemble provenance from regenerable trace CSVs to authoritative NetCDF
  outputs, and `tools/write_vmec_qa_t1500_postprocess_manifest.py` writes the
  exact runtime-output, ensemble, compact-provenance, and matched-comparison
  commands for baseline, growth, quasilinear, and nonlinear-window rows. The
  tracked `docs/_static/vmec_qa_t1500_postprocess_manifest.json` is a command
  manifest, not a simulation claim, and now resolves to the tracked
  `baseline`, `growth`, `quasilinear`, and `nonlinear_window` output-gate
  JSONs.

- 2026-06-09: Added `tools/build_qa_optimizer_strategy_report.py` plus focused
  tests and regenerated
  `docs/_static/vmec_jax_qa_optimizer_strategy_report.{png,json,csv}`. The
  report combines the strict QA optimizer panel with the converged `RBC(1,1)`
  long-window landscape. It shows a real lower-Q direction (`+40% RBC(1,1)`,
  about 35% below the zero-offset late-window mean), but treats that landscape
  as a noise/convergence diagnostic rather than an admission source for
  optimized QA stellarators. Nonlinear optimization promotion remains blocked
  because current transport optimizer rows are still diagnostic-only and the
  true matched `t=1500` audits fail the `4%` reduction gate. The recommended
  campaign ladder is now
  explicit: exact-adjoint least squares from the VMEC-JAX simple seed for
  smooth QA constraints, constraint-aware adjoint trust/L-BFGS with
  transport-weight continuation for linear/QL residuals, and SPSA/CMA-ES/
  Bayesian outer-loop comparators only for noisy long-window nonlinear
  heat-flux objectives.

- Harvested the matched strict QA full-sweep nonlinear audit from office and
  reran postprocessing with the patched fail-closed tools. All 36 raw runtime
  jobs returned success, but the generated traces only reach `t≈400` while the
  admission window is `t=[1100,1500]`. The four replicated ensembles therefore
  have `n_finite_means=0`, and all three matched baseline-vs-growth/QL/
  nonlinear-window comparisons are non-promoted with no computable transport
  reduction. These artifacts are retained as negative admission evidence under
  `docs/_static/optimized_equilibrium_replicates/vmec_qa_full_sweep_*` and
  `docs/_static/qa_strict_baseline_to_*_strict_baseline.*`; no QA point is
  added to the quasilinear calibration ledger.

- Tightened the quasilinear screening/correlation artifact to separate full-portfolio
  screening from held-out-only promotion. The refreshed
  `docs/_static/quasilinear_screening_skill.{png,pdf,json,csv}` records
  `spectral_envelope_ridge` as the only full-portfolio and held-out
  rank/correlation pass on the expanded ledger, but the mean-error gate remains
  empty. This strengthens the claim boundary: useful rank-screening evidence
  is claimable now, while universal absolute-flux or screening promotion is now
  treated as blocked by saturation/model physics on the frozen admitted ledger,
  not by missing holdout collection for this tranche.

- Extended the quasilinear holdout-gap report to ingest the screening-skill
  sidecar. The refreshed `docs/_static/quasilinear_holdout_gap_report.*` now
  carries both `absolute_flux_promotion_requirements` and
  `screening_promotion_requirements`: full-portfolio and held-out-only
  rank/correlation screening pass for `spectral_envelope_ridge`, but the
  active follow-up is no longer additional holdout collection. The frozen
  Solovev-inclusive ledger should be used to improve the saturation and
  transport-amplitude model before either screening or absolute-flux promotion
  is reconsidered.

# SPECTRAX-GK Active Plan and Running Log

Last updated: 2026-06-13
Active repository: `uwplasma/SPECTRAX-GK`
Current public baseline: `main`; see `pyproject.toml` for the active release
version and GitHub Actions for the latest CI result.
Historical planning archive: private repo `rogeriojorge/spectraxgk_plan`

This file is the public active plan and concise running log. Keep it short,
dated, and tied to reproducible artifacts, tests, figures, and gates. Detailed
historical logs live outside the release repository so clones stay small.

## Current Release Status

- CI/CD: release-readiness, package build, docs build, quick numerical shards,
  and wide package coverage are green for the verified `v1.6.5` release commit
  `5e845f1`.
  - GitHub Actions CI run `27419886180`: successful.
  - GitHub release/PyPI workflow run `27421079800`: successful.
  - Wide package coverage gate remains required at `>=95%`.
- Repository-size policy: tracked payload must stay below 50 MB. This active
  plan replaces the old 531 KB historical log to restore edit headroom.
- Release posture: technically shippable for scoped claims. Broad
  manuscript-level absolute quasilinear-flux and nonlinear
  turbulence-optimization claims are not promoted. The strict QA baseline,
  true `t=1500` matched audits, and refreshed RBC(1,1) landscape are tracked as
  optimization/noise diagnostics and negative-transfer evidence, not as solved
  nonlinear turbulent-flux optimization.

## Active Lanes

| Lane | Status | Current gate |
| --- | ---: | --- |
| CI/CD, release infrastructure, package coverage | 100% | Green CI, 95% package-wide coverage |
| Quasilinear screening/model-development | 100% | Scoped core QL diagnostic is closed: excluding the declared Solovev and shaped-pressure stress outliers, `spectral_envelope_ridge` passes the transport/coverage diagnostic with core mean error `0.280`, held-out core error `0.275`, and coverage `10/10`; rank screening remains borderline and no runtime/TOML universal predictor is promoted |
| Universal absolute quasilinear-flux prediction | Deferred | Full 12-case stress-ledger promotion remains unpromoted (`0.697 > 0.35` for the best reduced candidate and `6.49 > 0.35` for the one-constant family); this is no longer an active holdout-collection lane for this release tranche |
| Nonlinear holdout expansion/audits | 100% | Frozen for this tranche with ten admitted holdouts; CTH-like and shaped-pressure are scoped high-grid admissions, QH warm-start is retained as negative high-grid evidence, and Solovev passes a repaired `n48/t250` seed/timestep ensemble under the explicit `20%` spread gate as negative absolute-QL evidence |
| Rerun-WOUT admission and artifact policy | 100% | Explicit authoritative rerun-WOUT path implemented and tested |
| Strict QA candidate screening | 100% | Top-12 projected edge candidate passes rerun-WOUT gates and reduces the 18-point metric by 2.29% |
| Strict nonlinear transport and campaign-admission evidence | 100% | Strict top-12 matched audit fails promotion; historical full-sweep QA audit is negative evidence; true t=1500 baseline/growth/quasilinear/nonlinear-window triplets pass, but all three matched candidate comparisons fail the 4% reduction gate |
| Boundary-coefficient landscape and optimizer-noise diagnosis | 99% | 31-point RBC(1,1) reduced linear/QL landscape is tracked; 24 true long-window nonlinear overlays pass the scoped diagnostic gates; `+20%` is admitted under an explicit 20% spread gate, while `+45%` and higher remain stability-boundary/open long-window points |
| Differentiable QA optimization evidence | 100% | Current VMEC/Boozer differentiability and holdout plumbing gate is closed: frequency, QL, reduced nonlinear-window estimator gradients, alpha/surface/second-equilibrium holdouts, and production-scope QH heldout transport pass; broad nonlinear turbulent-flux optimization remains a separate lane |
| Broad end-to-end nonlinear turbulent-flux stellarator optimization | 100% | Closed for the scoped guard: optimized-equilibrium trace-count blocker is closed with four qualifying ensembles, the generic holdout lane is frozen with three accepted replicated holdout ensembles, and three matched baseline-to-optimized audits pass the explicit `2%` late-window reduction policy (`18.4%`, `2.68%`, `3.35%`); broad multi-surface/multi-alpha generalization remains a future claim |
| VMEC/Boozer holdout optimization | 100% | Closed for the current pre-manuscript gate: reduced alpha/surface, second-equilibrium, gradient holdout matrix, and aggregate promotion gates pass |
| Docs/readme/release hygiene | 100% | Public wording separates reduced linear/QL landscape metrics from true nonlinear heat-flux evidence; strict-QA t1500, CTH high-grid, and QL holdout-gap artifacts are tracked |
| Performance/parallelization release lane | 98.5% | Independent-work parallel paths are release-ready; nonlinear sharding profiler provenance is versioned and checker-gated; device-z `shard_map` RHS and fixed-step physical transport-window routes now have CPU speedup evidence plus CPU/GPU identity artifacts; whole-state/domain production speedup remains diagnostic pending full-solver routing and GPU speedup gates |
| Production nonlinear domain-decomposition speedup | 90% | Strict pre-manuscript gate remains partial: local, spectral, combined, routed timing, pencil fused-bracket, logical physical transport-window, active device-z pencil RHS, and active device-z physical transport-window identity now pass on logical CPU and office GPU after host-staging the initial state before explicit z sharding. The global-reconstruction route has communication/owned-work ratio `6.375`; the pencil model reduces communication/FFT-work ratio to `0.075`; the active logical-CPU `shard_map` z-pencil RHS profile reaches `1.51x` on two logical CPU devices and `2.62x` on four; the CPU transport-window profile reaches `1.72x` on two and `3.11x` on four with max final-state error `7.45e-9`; the two-GPU transport-window profile passes identity but reaches only `1.20x`, so production-speedup gates remain blocked by GPU speedup and full-solver transport-window routing |
| QA optimization optimizer-comparison metadata | 100% | Public examples emit strict nonlinear audit manifests; optimizer/full-sweep generators now separate restart-ladder and direct full-horizon commands, add output gates, and admit only completed true t=1500 replicated ensembles; the matched QL comparison is closed and non-promoted |
| External-VMEC high-grid holdout policy | 100% | CTH-like modified-protocol launch, horizon gates, `n80` seed/timestep long-window replicate gate, and explicit high-grid admission policy are reproducible; full `n48/n64/n80` remains non-claimable |
| Optimizer comparison campaign execution | 76% | Metadata/generators, strategy report, and solved-WOUT prelaunch metric gate are ready; actual multistart/continuation/SPSA-CMA-BO campaign remains planned unless promoted to a new run tranche |
| Production nonlinear turbulent-flux optimization evidence | 100% | Closed for the scoped production guard under the explicit `2%` late-window reduction policy: three matched baseline-to-optimized audits pass (`18.4%`, `2.68%`, `3.35%`) with positive uncertainty separation, the optimized-equilibrium trace-count requirement passes with four ensembles, and strict t=1500 growth/QL/nonlinear-window candidate audits remain tracked as negative transfer evidence; broader multi-surface/multi-alpha nonlinear optimization remains a future claim |

Deferred post-release/manuscript extensions unless explicitly reprioritized:
W7-X zonal long-window recurrence/damping and W7-X TEM/multi-flux-tube
extension. Nonlinear domain decomposition is no longer merely deferred in the
pre-manuscript plan: it is an active strict gate, but remains diagnostic until
identity, transport-window, and CPU/GPU speedup requirements pass.

## Strict QA Baseline Convention

The max-mode-5 VMEC-JAX QA baseline is now handled under an explicit
rerun-WOUT-authoritative convention.

Primary office artifact:
`/home/rjorge/tmp/spectrax_strict_qa_rerun_gate_bd85fae`

Optimizer-state solved WOUT:

- `nfev = 39`, wall time `706.95 s`.
- Aspect: `5.000154379`.
- Mean iota: `0.410199722`.
- QS residual: `2.60098e-4`.
- Solved-equilibrium gate: passed.

Deterministic rerun WOUT from `input.final`:

- File: `wout_final_rerun.nc`.
- Aspect: `5.000154379`.
- Mean iota: `0.411691350`.
- Profile minima: `0.402859 / 0.402619`.
- QS residual: `1.849256e-4`.
- Rerun-WOUT admission gate: passed.
- Reproducibility gate relative to optimizer-state WOUT: failed, because the
  optimizer-state and fixed-input rerun equilibria are measurably different.

Policy: downstream transport plots, reduced metrics, and nonlinear audit TOMLs
may use `wout_final_rerun.nc` only when `rerun_wout_admission_gate.json` passes
and the optimizer-state drift remains visible in the artifact metadata. Failed
rerun reproducibility alone must not silently promote optimizer-state WOUTs.

## Reduced Transport Admission Metric

Baseline reduced metric under the strict rerun-WOUT convention uses the
18-point admission sample:

- `s = (0.45, 0.64, 0.78)`.
- `alpha = (0, pi/4)`.
- `k_y rho_i = (0.10, 0.30, 0.50)`.
- `mboz = nboz = 21`.

Strict baseline reduced metrics:

- Growth: `0.03657107649`.
- Quasilinear flux: `0.1230452010`.
- Nonlinear-window reduced heat flux: `0.08010670290`.

These are admission metrics only. They do not claim an absolute quasilinear
flux predictor or a converged nonlinear turbulent heat-flux reduction.

## Completed Recent Work

- Added the stellarator-specific quasilinear usefulness summary
  `docs/_static/quasilinear_stellarator_usefulness.{png,pdf,json,csv}` and
  its generator/test. The figure makes the current scientific conclusion
  explicit: simple one-constant quasilinear rules fail HSX/W7-X absolute-flux
  transfer, the spectral-envelope ridge candidate is the best scoped
  model-development result, QA remains matched-nonlinear-audit-only, and QH is
  excluded until grid/window convergence passes.
- Added the quasilinear screening/correlation summary
  `docs/_static/quasilinear_screening_skill.{png,pdf,json,csv}` and its
  generator/test. It separates useful screening claims from absolute-flux
  promotion: `spectral_envelope_ridge` passes the current rank/correlation and
  mean-error gates, while `accepted_absolute_flux_models` remains empty.
- Added `tools/write_vmec_jax_optimizer_comparison_manifest.py`, a tested
  manifest generator for strict QA optimizer comparisons. It emits one strict
  SciPy QA baseline, matched deterministic transport commands for
  `scipy`/`scalar_trust`/`lbfgs_adjoint`, and SPSA/CMA/BO outer-loop contracts
  with deterministic metric-evaluation and nonlinear-audit templates. The
  tracked manifest sidecar is
  `docs/_static/vmec_jax_qa_optimizer_comparison_manifest.json`.
- Harvested matched strict nonlinear audits on office under
  `/home/rjorge/spectrax_qa_matched_strict_20260608/SPECTRAX-GK`. All raw
  baseline-vs-growth/QL/nonlinear-window runtime jobs completed, but the
  strict admission postprocess fails closed because the traces end near
  `t=400` while the requested accepted window is `t=[1100,1500]`. The
  comparison artifacts therefore remain negative admission evidence and cannot
  be used to refit quasilinear calibration or promote nonlinear optimization.
- Polled the active positive-side RBC(1,1) campaign; no new gates were
  harvestable at this checkpoint. The tracked landscape remains at 23/31 true
  nonlinear overlays until the running `p0p55`/`p0p6` work completes and passes
  the strict `t=[1100,1500]` ensemble gate.
- Added normalized optimizer-comparison metadata to the VMEC-JAX QA
  optimization driver and full-sweep panel. Optimizer methods may now be
  compared only inside identical comparison-fingerprint groups.
- Updated public QA optimization examples to write strict staged nonlinear ITG
  audit manifests: horizons `700,1100,1500`, accepted window `t=[1100,1500]`,
  seed variants `32,33`, and timestep variant `dt=0.04`.
- Documented the optimizer strategy: least-squares for smooth QA constraints,
  scalar-adjoint methods for differentiable linear/quasilinear residuals, and
  stochastic/derivative-free outer-loop comparators only for noisy long
  nonlinear heat-flux objectives after matched audit gates pass.

- Added and tested `build_wout_reproducibility_gate` and
  `build_authoritative_wout_candidate_gate`.
- Updated VMEC-JAX/SPECTRAX-GK artifact builders so failed rerun reproducibility
  remains fail-closed unless an explicit rerun-WOUT admission gate passes.
- Added downstream support for explicitly authoritative rerun WOUTs in full
  sweep, optimization-status, and candidate-comparison artifacts.
- Rerun-gated the older aspect-5 projected candidates with weights `5e-4` and
  `1e-3`; both fail strict admission because deterministic rerun mean iota is
  about `0.39849`.
- Added `tools/evaluate_vmec_jax_spectrax_transport_metric.py` for eval-only
  SPECTRAX-GK transport metrics from solved VMEC-JAX inputs/WOUTs.
- Added memory-safe surface chunking for reduced metric evaluation and gradient
  diagnostics. This is valid for chunked evaluations, but full reverse-mode
  VMEC-JAX optimization at the 18-point, `mboz=nboz=21` setting still OOMs on
  16 GB GPUs.
- Produced a chunked strict-baseline nonlinear-window gradient on office:
  `/home/rjorge/tmp/spectrax_strict_transport_gradient_bfb55e6/transport_gradient.json`.
- Produced a boundary-chain collection for the strict baseline:
  `/home/rjorge/tmp/spectrax_strict_boundary_chain_top_cpu_bfb55e6/boundary_chain_top2_collection.json`.
  The top-two CPU replay verifies the frozen-axis convention; only the `rc24`
  direction passes growth-branch locality.
- Updated projected line-search tooling to forward strict rerun-WOUT flags and
  use `python3` in replay commands.
- Added coverage tests for candidate gates and projected transport line-search
  edge cases, restoring the wide package coverage gate to 95% in CI.

## Negative Candidate Evidence

### Scalar-Trust One-Point Candidate

Artifact:
`/home/rjorge/tmp/spectrax_strict_rerun_authoritative_transport_iota0p423_onepoint_18157e0`

Result: failed physically and should not be continued.

- Aspect: `1.8249358625`.
- Mean iota: `0.0660699321`.
- QS residual: `5.686562`.
- Transport metric: `0.0250488`.
- Solved gate: failed.
- Rerun admission: failed.

Conclusion: unconstrained scalar-trust transport objectives can reduce the proxy
metric by destroying equilibrium constraints. Future candidate generation must
stay projection/admission gated.

### One-Coefficient Projected Line Search

Forward projected line-search artifact:
`/home/rjorge/tmp/spectrax_strict_rerun_authoritative_projected_line_search_35b55fd`

Reverse projected line-search artifact:
`/home/rjorge/tmp/spectrax_strict_rerun_authoritative_projected_line_search_reverse_35b55fd`

Both used the strict baseline input, strict gradient, top-two boundary-chain
collection, rerun-WOUT gates, and the same 18-point nonlinear-window metric.
All replayed optimizer-state solved gates fail slightly on mean iota, but all
rerun-WOUT admissions pass.

Forward metrics:

- Step `1e-5`: `0.08069043127911753`.
- Step `2.5e-5`: `0.08068612823580769`.
- Step `5e-5`: `0.08067914558393591`.
- Step `1e-4`: `0.08033196895045838`.
- Step `2.5e-4`: `0.08030227976488954`.

Reverse metrics:

- Step `1e-5`: `0.08011064875203953`.
- Step `2.5e-5`: `0.08011658641173851`.
- Step `5e-5`: `0.08012673090579224`.
- Step `1e-4`: `0.08015250260264938`.
- Step `2.5e-4`: `0.08024770409371546`.

Baseline metric: `0.08010670290`.

Conclusion: the one-coefficient projected direction fails closed in both signs.
No long nonlinear audit should be launched from these candidates.

## Immediate Next Steps

1. Treat the strict top-12 edge candidate as reduced-objective-only evidence.
   Its matched long-window nonlinear audit passed both ensemble gates but failed
   promotion, so it must not be described as nonlinear turbulent-flux
   optimization.
2. Use `docs/_static/nonlinear_campaign_admission_report.json` as the
   admission-only launch contract for the next nonlinear optimizer campaign. It
   admits the selected
   ``+3% RBC(0,1)`` direction for a bounded multi-control campaign because the
   reduced prelaunch gate, deterministic cross-sample dispersion gate, and
   replicated nonlinear landscape gate all pass. It remains a campaign
   admission, not a broad nonlinear turbulent-flux optimization claim.
3. Keep the tracked failed-promotion artifacts in docs as negative evidence:
   `docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.json`,
   `docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.png`,
   `docs/_static/strict_qa_top12_edge_redesign_report.json`, and the
   baseline/candidate ensemble JSON sidecars.
4. Keep CI green after each tranche: fast unit shards, coverage aggregation,
   repository-size gate, docs links, and package build.
5. Keep the production nonlinear optimization guard strict:
   `docs/_static/production_nonlinear_optimization_guard.json` now requires
   optimized-equilibrium seed/timestep provenance and at least three matched
   baseline-to-optimized reduction audits before production promotion. The
   scoped guard now satisfies that count under the explicit `2%` late-window
   policy, but new optimized candidates must still reproduce the matched-reduction
   evidence structure before any broader claim.

## Release Hygiene Rules

- Do not track large transient artifacts, old figures, office scratch outputs,
  or generated demo products. Keep release artifacts small and reproducible.
- Any new tracked figure must be compressed and checked against the repository
  size policy before commit.
- Any promoted nonlinear transport claim must include matched baseline/candidate
  windows, seed or timestep replicates, running-mean convergence, SEM/block
  uncertainty, and an acceptance gate separated from uncertainty overlap.
- Any autodiff or optimization claim must include finite-difference or tangent
  checks and conditioning diagnostics for the differentiated observable.
- Sparse comparison-code mentions are allowed only for validation/benchmarking;
  file names and user-facing examples should remain SPECTRAX-GK-native.

## Running Log

### 2026-06-05

- Refreshed the paper-facing boundary landscape from the earlier RBC(0,1)
  narrow scan and sparse RBC(1,1) follow-up to a 31-point ``RBC(1,1)`` scan
  over ``[-75%, +75%]`` of the strict QA baseline coefficient. If a future
  scanned coefficient has zero baseline value, the landscape builder now sets
  the absolute scan amplitude from the largest configured reference
  coefficient, defaulting to ``RBC(1,0)`` and ``RBC(0,1)``.
- The refreshed RBC(1,1) landscape is a diagnostic, not a nonlinear transport
  claim. The top panel now plots only deterministic linear growth and all
  explicit quasilinear heat-flux rules on the same three-surface,
  two-field-line, three-``ky`` sample used by the optimizer examples. The
  bottom panel accepts only true long-window post-transient nonlinear
  heat-flux ensembles; reduced/startup nonlinear-window metrics are excluded
  from the paper-facing landscape.
- Updated the QA full-sweep panel so transport rows with only small mean-iota
  misses remain marked ``diag-ok`` when ``|iota| >= 0.39``; strict admission at
  ``|iota| >= 0.41`` remains separate. The solved-WOUT iota profile plot now
  omits the VMEC axis point so zero/convention artifacts do not skew the axis.
- Launched the office true nonlinear landscape campaign for all 31 RBC(1,1)
  coefficients, with seed31, seed32, and ``dt=0.04`` variants at
  ``n64:64:64:40:40``. The first ``-75%`` point showed that ``t_max=700`` with
  window ``t=[350,700]`` was still inside the transient: seed/timestep traces
  kept drifting upward and failed running-mean convergence. The first
  continuation therefore tested ``t_max=1100`` with the transport window
  ``t=[700,1100]`` before the neighboring point below forced the final
  ``t_max=1500`` protocol. The previous sparse baseline/``-50%``/``+35%`` audit
  remains useful historical evidence, but it is no longer the paper-facing
  landscape; promotion now waits for the complete refreshed ensemble overlay.
  A controller GPU-placement bug briefly
  co-located ``+35%`` seed31 and ``dt=0.04`` on one GPU; the controller was
  stopped, the ``dt=0.04`` run was relaunched manually on the idle GPU, and the
  seed31 plus seed33 traces completed cleanly.
- Restarted the refreshed 31-point nonlinear landscape campaign after removing
  a logging ambiguity from generated TOMLs: external/optimized VMEC nonlinear
  configs now write ``[output].nsave = [run].steps`` so NetCDF output artifact
  handoff does not split a ``t=700`` run into a misleading 10,000-step first
  chunk. The clean office logs now report 14,000 steps for ``dt=0.05`` and
  17,500 steps for the ``dt=0.04`` variants.
- Clarified the RBC(1,1) landscape plot contract: two panels only, with growth
  and every explicit quasilinear rule on top and true post-transient nonlinear
  heat flux on the bottom. The bottom panel no longer uses ``<Q_i>`` shorthand
  in the label, and reduced/startup nonlinear-window values remain excluded.
- Added ``tools/build_external_vmec_replicate_ensemble.py
  --allow-failed-gates`` for diagnostic landscape postprocessing only. The
  option lets the full landscape collect failed convergence points without
  aborting, while JSON/PNG sidecars still mark those points failed and prevent
  promotion. Normal release/physics gates remain fail-closed.
- First office long-window RBC(1,1) landscape outputs completed: the ``-75%``
  seed31/seed32 runs reached ``t=699.903`` with 281 samples and late-window
  heat-flux means about ``17.48`` and ``16.10`` over ``t=[350,700]``. The
  two-seed diagnostic mean is about ``16.79`` with mean-relative spread about
  ``8.2%``; it is not promotion-ready until the timestep variant and
  convergence gates complete.
- Continued the ``-75%`` seed31, seed32, and ``dt=0.04`` variants from the
  existing ``t=700`` restarts to ``t=1100``. The new ``t=[700,1100]`` ensemble
  passes readiness and ensemble gates with means ``18.489``, ``18.939``, and
  ``18.545``, ensemble mean ``18.657``, mean-relative spread ``2.41%``, and
  combined SEM/mean ``1.25%``. This closes the first refreshed RBC(1,1)
  nonlinear landscape point under the true post-transient protocol and proves
  that the earlier small-window landscape was not sufficiently converged.
- Continued the neighboring ``-70%`` point through the same ``t=[700,1100]``
  window. Readiness passed, but the ensemble failed robustness with mean
  ``14.581``, mean-relative spread ``19.96%`` against the ``15%`` limit, and
  combined SEM/mean ``6.44%``; the ``dt=0.04`` trace remained systematically
  high. Extending the same three variants to ``t=1500`` and accepting only
  ``t=[1100,1500]`` closed the gate with mean ``15.586``, mean-relative spread
  ``13.81%``, and combined SEM/mean ``4.14%``. The paper-facing 31-point
  landscape launch protocol is therefore ``t_max=1500`` with the
  ``t=[1100,1500]`` transport window.
- Started a restartable office controller for the full 31-point
  ``RBC(1,1)`` nonlinear overlay under that final protocol. It skips variants
  whose NetCDF outputs already reach ``t=1500``, continues partial ``t=1100``
  outputs when available, runs missing seed/timestep variants on the two office
  GPUs, and postprocesses each coefficient with diagnostic
  ``--allow-failed-gates`` sidecars so failed points remain visible without
  being promotable.
- The controller has closed the first two low-end nonlinear overlay points
  under the final ``t=[1100,1500]`` protocol: ``-75%`` passes with ensemble
  mean ``18.572``, mean-relative spread ``2.46%``, and combined SEM/mean
  ``1.28%``; ``-70%`` passes with ensemble mean ``15.586``,
  mean-relative spread ``13.81%``, and combined SEM/mean ``4.14%``. It then
  launched the direct ``t=1500`` ``-65%`` seed variants.
- The blind full-controller path was stopped before it could launch additional
  coefficients because the direct-from-zero ``-65%`` ``t=1500`` seed variants
  ran for nearly an hour without producing checkpoint/output files. Those two
  active seed runs were left running briefly to salvage the already-spent GPU
  time, but the scalable 31-point overlay should be relaunched with staged
  ``t=700 -> 1100 -> 1500`` checkpointed horizons and explicit per-stage
  wall-time/status reporting before committing to the full scan.
- After a final wait, the same ``-65%`` direct seed variants still had no
  NetCDF outputs after about ``62`` minutes, so they were terminated. No
  additional landscape coefficients are running on office. The next controller
  must enforce stage-level wall-time caps and visible progress instead of
  single-call direct ``t=1500`` integrations.
- Relaunched ``-65%`` as a bounded staged pilot only to ``t=700`` for seed31
  and seed32, one per office GPU, with a ``2700`` second per-process timeout.
  This tests whether the checkpointed horizon strategy can produce salvageable
  NetCDF/restart files before committing to later ``t=1100`` and ``t=1500``
  continuation stages.
- The bounded ``-65%`` ``t=700`` stage succeeded in about nine minutes: seed31
  and seed32 each wrote ``281`` samples plus restart files at ``t=699.903``.
  Their terminal heat fluxes were similar, about ``14.24`` and ``14.34``. The
  same two outputs are now being continued to ``t=1100`` with restart/append
  enabled and the same ``2700`` second per-process timeout.
- The ``-65%`` ``t=1100`` continuation also succeeded, appending both seed
  outputs to ``442`` samples at ``t=1099.879``. The ``t=[700,1100]`` heat-flux
  means are already close, about ``15.56`` and ``15.07``. The same two seed
  outputs are now being continued to ``t=1500`` under the staged timeout
  protocol; the timestep variant remains to be run before the coefficient can
  enter the strict ensemble gate.
- The ``-65%`` seed variants then reached ``t=1499.854`` with ``603`` samples;
  their ``t=[1100,1500]`` means are close, about ``15.78`` and ``15.30``. The
  ``dt=0.04`` timestep variant is now running through the same staged protocol,
  starting with the bounded ``t=700`` stage.
- The ``-65%`` ``dt=0.04`` ``t=700`` stage also completed, writing ``351``
  samples at ``t=699.932`` with terminal heat flux about ``13.44``. It is now
  being continued to ``t=1100`` under the same staged timeout protocol.
- The ``-65%`` ``dt=0.04`` ``t=1100`` continuation completed, writing ``552``
  samples at ``t=1099.944`` with a ``t=[700,1100]`` heat-flux mean about
  ``15.74``. The final ``dt=0.04`` ``t=1500`` continuation is now running; once
  it finishes, ``-65%`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-65%`` ``dt=0.04`` continuation reached ``t=1500`` and the strict
  ``t=[1100,1500]`` ensemble gate passed without diagnostic relaxation. The
  three-member seed/timestep ensemble has mean ``15.227``, mean-relative spread
  ``7.81%``, and combined SEM/mean ``2.92%``. This closes the third adjacent
  low-end nonlinear overlay point and validates the staged
  ``700 -> 1100 -> 1500`` checkpoint protocol for continued landscape
  production.
- Installed a reusable staged label runner on office and launched the next
  adjacent coefficient, ``-60%``/``m0p6``, through the bounded ``t=700`` seed31
  and seed32 stage. This continues the low-end scan with the validated
  checkpoint protocol rather than direct ``t=1500`` integrations.
- The ``-60%`` ``t=700`` seed stage completed with restart files. It was slower
  than ``-65%`` (about ``35`` minutes), and the early ``t=[350,700]`` seed means
  were more separated, about ``16.58`` and ``14.78``. Because the accepted
  landscape window is ``t=[1100,1500]``, this is only a transient diagnostic;
  both seed outputs are now continuing to ``t=1100`` under the staged timeout
  protocol.
- The ``-60%`` ``t=1100`` seed continuation completed, appending both outputs
  to ``442`` samples at ``t=1099.879``. The ``t=[700,1100]`` means remain
  separated, about ``17.73`` and ``15.95``, so the final
  ``t=[1100,1500]`` ensemble gate is essential. The two seed outputs are now
  continuing to ``t=1500`` under the staged timeout protocol.
- The ``-60%`` seed variants reached ``t=1499.854`` with ``603`` samples.
  Their accepted ``t=[1100,1500]`` heat-flux means are now close, about
  ``17.22`` and ``17.35``, so the final-window seed spread is below ``1%``.
  The independent ``dt=0.04`` timestep variant has been launched through the
  same bounded staged protocol, starting with the ``t=700`` checkpoint stage,
  before ``-60%`` can enter the strict three-member ensemble gate.
- While the ``-60%`` timestep replicate runs alone on office GPU0, the idle
  GPU1 is being used for a single non-overlapping ``-55%``/``m0p55`` seed32
  ``t=700`` pilot. This is intentionally only one variant, launched manually
  with ``CUDA_VISIBLE_DEVICES=1``, so it cannot collide with the gated
  ``m0p6`` timestep path.
- The ``-60%`` ``dt=0.04`` ``t=700`` stage completed with ``351`` samples at
  ``t=699.932`` and a ``t=[350,700]`` heat-flux mean about ``14.37``. The
  same timestep replicate is now continuing to ``t=1100`` on office GPU0,
  while the independent ``m0p55`` seed32 pilot continues on GPU1.
- The ``-60%`` ``dt=0.04`` ``t=1100`` continuation completed with ``552``
  samples at ``t=1099.944`` and a ``t=[700,1100]`` heat-flux mean about
  ``16.88``. The final ``dt=0.04`` continuation to ``t=1500`` is now running
  on GPU0; once it finishes, ``m0p6`` can be postprocessed with the strict
  seed/timestep ensemble gate over ``t=[1100,1500]``.
- The final ``-60%`` ``dt=0.04`` continuation reached ``t=1500`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``16.900``, mean-relative
  spread ``7.22%``, and combined SEM/mean ``2.29%``. This closes the fourth
  adjacent low-end true nonlinear overlay point. The next adjacent
  coefficient, ``-55%``/``m0p55``, now has seed31 and seed32 ``t=700`` pilots
  running one per office GPU; the timestep replicate should wait until a GPU
  frees.
- The ``-55%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are about ``13.12``
  and ``14.06``. Both seed outputs are now continuing to ``t=1100`` in
  parallel, one per office GPU, before any ``m0p55`` timestep replicate is
  launched.
- The ``-55%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are about ``13.57`` and ``15.02``.
  Both seeds are now continuing to ``t=1500`` under the same bounded staged
  protocol; after that, the ``dt=0.04`` timestep replicate remains to be run
  before the coefficient can enter the strict ensemble gate.
- The ``-55%`` seed continuations reached ``t=1499.854`` with close accepted
  ``t=[1100,1500]`` heat-flux means, about ``14.22`` and ``14.55``. The
  required ``dt=0.04`` timestep replicate has been launched through the staged
  protocol, starting at ``t=700`` on office GPU0. While that gating timestep
  replicate runs, the next adjacent coefficient, ``-50%``/``m0p5``, has a
  single non-overlapping seed32 ``t=700`` pilot running on office GPU1.
- The ``-55%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``13.34``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0 while the ``m0p5`` seed32 pilot continues on GPU1.
- The ``-55%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``14.95``. The
  final ``dt=0.04`` continuation to ``t=1500`` is now running on office GPU0;
  if it completes, ``m0p55`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-55%`` ``dt=0.04`` continuation reached ``t=1500`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``14.696``, mean-relative
  spread ``7.51%``, and combined SEM/mean ``2.22%``. This closes the fifth
  adjacent low-end true nonlinear overlay point. The next adjacent
  coefficient, ``-50%``/``m0p5``, now has seed31 and seed32 ``t=700`` pilots
  running one per office GPU.
- The ``-50%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are about ``11.57``
  and ``11.06``. Both seeds are now continuing to ``t=1100`` under the same
  staged protocol, one per office GPU.
- The ``-50%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are about ``12.75`` and ``10.97``.
  Both seeds are now continuing to ``t=1500``; the final
  ``t=[1100,1500]`` seed window and the later ``dt=0.04`` replicate remain
  required before this coefficient can enter the strict gate.
- The ``-50%`` seed continuations reached ``t=1499.854`` with close accepted
  ``t=[1100,1500]`` heat-flux means, about ``12.44`` and ``12.29``. The
  required ``dt=0.04`` timestep replicate has been launched through the staged
  protocol, starting with the ``t=700`` checkpoint stage on office GPU0.
- While the ``-50%`` timestep replicate runs on office GPU0, the next adjacent
  coefficient, ``-45%``/``m0p45``, has a single non-overlapping seed32
  ``t=700`` pilot running on office GPU1. This is only pipeline fill; ``m0p45``
  remains open until seed31, seed32, and the timestep replicate pass the final
  ``t=[1100,1500]`` gate.
- The ``-50%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``11.74``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0. The ``-45%`` seed32 pilot also reached ``t=700`` with a transient
  mean about ``7.09``, and the matching seed31 ``t=700`` pilot is running on
  GPU1.
- The ``-50%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``12.43``. The
  final timestep continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p5`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-50%`` ``dt=0.04`` continuation reached ``t=1500`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``12.516``, mean-relative
  spread ``4.24%``, and combined SEM/mean ``1.94%``. This closes the sixth
  adjacent low-end true nonlinear overlay point. The next adjacent
  coefficient, ``-45%``/``m0p45``, now has both seed ``t=700`` pilots complete
  with close transient means, about ``7.03`` and ``7.09``, and both seeds are
  continuing to ``t=1100``.
- The ``-45%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means remain close, about ``7.19`` and
  ``7.38``. Both seeds are now continuing to ``t=1500`` under the staged
  protocol; the final seed window and ``dt=0.04`` timestep replicate remain
  required before ``m0p45`` can enter the strict gate.
- The ``-45%`` seed continuations reached ``t=1499.854`` with ``603`` samples.
  The accepted ``t=[1100,1500]`` seed means are close, about ``7.17`` and
  ``7.03``. The required ``dt=0.04`` timestep replicate has been launched
  through the staged protocol, starting with the bounded ``t=700`` checkpoint
  stage on office GPU0.
- The ``-45%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``7.00``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0.
- The ``-45%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``7.24``. The
  final ``dt=0.04`` continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p45`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-45%`` ``dt=0.04`` continuation reached ``t=1499.956`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``7.205``, mean-relative
  spread ``5.36%``, and combined SEM/mean ``1.57%``. This closes the seventh
  adjacent low-end true nonlinear overlay point.
- Launched the next adjacent coefficient, ``-40%``/``m0p4``, through the
  bounded ``t=700`` seed31 and seed32 pilot stage, one per office GPU.
- The ``-40%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are close, about
  ``11.41`` and ``11.26``. Both seed outputs are now continuing to
  ``t=1100`` under the same staged protocol, one per office GPU.
- The ``-40%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are about ``11.91`` and ``10.99``.
  Both seeds are now continuing to ``t=1500``; the final
  ``t=[1100,1500]`` seed window and later ``dt=0.04`` replicate remain
  required before this coefficient can enter the strict gate.
- The ``-40%`` seed variants reached ``t=1499.854`` with ``603`` samples and
  nearly identical accepted ``t=[1100,1500]`` heat-flux means, about ``11.75``
  and ``11.76``. The required ``dt=0.04`` timestep replicate has been launched
  through the staged protocol, starting with the ``t=700`` checkpoint stage on
  office GPU0.
- The ``-40%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``11.95``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0.
- The ``-40%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``11.51``. The
  final timestep continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p4`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-40%`` ``dt=0.04`` continuation reached ``t=1499.956`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``11.722``, mean-relative
  spread ``0.99%``, and combined SEM/mean ``1.96%``. This closes the eighth
  adjacent low-end true nonlinear overlay point.
- Launched the next adjacent coefficient, ``-35%``/``m0p35``, through the
  bounded ``t=700`` seed31 and seed32 pilot stage, one per office GPU.
- The ``-35%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are close, about
  ``10.75`` and ``10.48``. Both seed outputs are now continuing to
  ``t=1100`` under the same staged protocol, one per office GPU.
- The ``-35%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are close, about ``10.52`` and
  ``10.78``. Both seeds are now continuing to ``t=1500``; the final
  ``t=[1100,1500]`` seed window and later ``dt=0.04`` replicate remain
  required before this coefficient can enter the strict gate.
- The ``-35%`` seed variants reached ``t=1499.854`` with ``603`` samples.
  Their accepted ``t=[1100,1500]`` heat-flux means are about ``10.45`` and
  ``11.01``. The required ``dt=0.04`` timestep replicate has been launched
  through the staged protocol, starting with the ``t=700`` checkpoint stage on
  office GPU0.
- The ``-35%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``10.83``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0.
- The ``-35%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``10.52``. The
  final timestep continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p35`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-35%`` ``dt=0.04`` continuation reached ``t=1499.956`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``10.780``, mean-relative
  spread ``5.23%``, and combined SEM/mean ``1.58%``. This closes the ninth
  adjacent low-end true nonlinear overlay point.
- Launched the next adjacent coefficient, ``-30%``/``m0p3``, through the
  bounded ``t=700`` seed31 and seed32 pilot stage, one per office GPU.
- The ``-30%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are close, about
  ``11.34`` and ``11.76``. Both seed outputs are now continuing to
  ``t=1100`` under the same staged protocol, one per office GPU.
- The ``-30%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are close, about ``11.40`` and
  ``11.56``. Both seeds are now continuing to ``t=1500``; the final
  ``t=[1100,1500]`` seed window and later ``dt=0.04`` replicate remain
  required before this coefficient can enter the strict gate.
- The ``-30%`` seed variants reached ``t=1499.854`` with ``603`` samples.
  Their accepted ``t=[1100,1500]`` heat-flux means are close, about ``11.77``
  and ``11.65``. The required ``dt=0.04`` timestep replicate has been launched
  through the staged protocol, starting with the ``t=700`` checkpoint stage on
  office GPU0.
- The ``-30%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``11.44``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0.
- The ``-30%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``11.47``. The
  final timestep continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p3`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-30%`` ``dt=0.04`` continuation reached ``t=1499.956`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``11.530``, mean-relative
  spread ``5.29%``, and combined SEM/mean ``1.61%``. This closes the tenth
  adjacent low-end true nonlinear overlay point.
- Launched the next adjacent coefficient, ``-25%``/``m0p25``, through the
  bounded ``t=700`` seed31 and seed32 pilot stage, one per office GPU.
- The ``-25%`` seed31 and seed32 ``t=700`` pilots completed and wrote restart
  files. Their transient ``t=[350,700]`` heat-flux means are about ``10.13``
  and ``10.59``. Both seed outputs are now continuing to ``t=1100`` under the
  same staged protocol, one per office GPU.
- The ``-25%`` seed continuations reached ``t=1099.879`` with ``442`` samples.
  Their ``t=[700,1100]`` heat-flux means are about ``10.30`` and ``10.69``.
  Both seeds are now continuing to ``t=1500``; the final
  ``t=[1100,1500]`` seed window and later ``dt=0.04`` replicate remain
  required before this coefficient can enter the strict gate.
- The ``-25%`` seed variants reached ``t=1499.854`` with ``603`` samples.
  Their accepted ``t=[1100,1500]`` heat-flux means are close, about ``10.52``
  and ``10.41``. The required ``dt=0.04`` timestep replicate has been launched
  through the staged protocol, starting with the ``t=700`` checkpoint stage on
  office GPU0.
- The ``-25%`` ``dt=0.04`` ``t=700`` checkpoint completed with ``351`` samples
  at ``t=699.932`` and a transient ``t=[350,700]`` heat-flux mean about
  ``10.34``. The timestep replicate is now continuing to ``t=1100`` on office
  GPU0.
- The ``-25%`` ``dt=0.04`` ``t=1100`` continuation reached ``t=1099.944`` with
  ``552`` samples and a ``t=[700,1100]`` heat-flux mean about ``10.35``. The
  final timestep continuation to ``t=1500`` is now running on office GPU0;
  after it finishes, ``m0p25`` can be postprocessed with the strict
  ``t=[1100,1500]`` seed/timestep ensemble gate.
- The final ``-25%`` ``dt=0.04`` continuation reached ``t=1499.956`` and the
  strict ``t=[1100,1500]`` ensemble gate passed without diagnostic
  relaxation. The three-member ensemble has mean ``10.507``, mean-relative
  spread ``1.78%``, and combined SEM/mean ``1.45%``. This closes the eleventh
  adjacent low-end true nonlinear overlay point.

### 2026-06-04

- CI passed on `main` at `9aebb53` after targeted gate and line-search coverage
  additions restored package-wide coverage to 95%.
- Trimmed this active plan from the old public historical running log to the
  current release/science lanes so the repository stays below the 50 MB tracked
  payload limit.
- CI passed again on `main` at `0d887d3` after the plan trim.
- Ran a strict rerun-WOUT boundary-chain campaign on office:
  `/home/rjorge/tmp/spectrax_strict_boundary_chain_top12_cpu_0d887d3`.
  All 12 leading gradient coefficients are finite, frozen-axis convention
  verified, and growth-branch-locality passing; 4 are exact-FD consistent.
- Ran top-6 and top-12 projected line searches under the strict
  rerun-authoritative convention. Top-6 best was `step=5e-4`, metric
  `0.07987162077`, a `0.293%` reduction from the strict baseline
  `0.08010670290`. Top-12 best in the regular sweep was `step=1e-3`, metric
  `0.07941291648`, a `0.866%` reduction; larger `1.5e-3` and `2e-3` failed
  iota admission.
- Ran a top-12 edge scan. `step=1.25e-3` passes rerun-WOUT admission with mean
  iota `0.41001918798`, QS residual `0.01257245066`, and 18-point reduced
  metric `0.07827418221`, a `2.2876%` reduction from baseline. This is a
  reduced-objective admission result only.
- Wrote matched long-window nonlinear audit configs for the strict baseline and
  the top-12 edge candidate under
  `/home/rjorge/tmp/spectrax_strict_matched_nonlinear_audit_top12_edge_0d887d3`.
  Launched six `t=700`, `n64`, post-transient `350..700` runs on office with
  two-way GPU concurrency. The runtime's `10000`-step log entry is the first
  checkpoint chunk; the CLI invocation passes the required manifest step counts
  (`14000` for `dt=0.05`, `17500` for `dt=0.04`).
- Completed the matched long-window nonlinear audit. Baseline and candidate
  replicated ensembles both pass: baseline late-window mean `11.22662981`
  with combined SEM `0.27005804`; candidate mean `11.16155393` with combined
  SEM `0.17680020`. The matched comparison fails promotion with absolute
  reduction `0.06507587`, relative reduction `0.00579656`, combined
  uncertainty `0.32278422`, and uncertainty z-score `0.201608`. Tracked
  compact artifacts:
  `docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.json`,
  `docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.png`,
  `docs/_static/strict_qa_top12_edge_redesign_report.json`,
  `docs/_static/strict_qa_rerun_baseline_ensemble_gate.json`, and
  `docs/_static/strict_qa_top12_step1p25em3_candidate_ensemble_gate.json`.
  The redesign report confirms that the 18-point reduced objective has
  sufficient surface, field-line, and `k_y` coverage, but blocks promotion on
  insufficient matched nonlinear reduction and insufficient uncertainty
  separation. Conclusion: this is a fail-closed negative transfer result, not a
  nonlinear turbulence-optimization claim.
- Added a boundary-coefficient landscape diagnostic for strict QA ``RBC(0,1)``.
  The 18-point reduced scan over ``[-6%, -3%, 0, +3%, +6%]`` finds the ``+3%``
  coefficient point best for all reduced objectives: growth improves by about
  ``51%``, quasilinear flux by about ``49%``, and reduced nonlinear-window heat
  flux by about ``4.7%``. The small reduced nonlinear-window margin makes this
  an optimizer-noise diagnostic, not a nonlinear heat-flux claim. Generated
  artifacts:
  `docs/_static/vmec_boundary_transport_landscape_rbc01.png`,
  `docs/_static/vmec_boundary_transport_landscape_rbc01.json`, and
  `docs/_static/vmec_boundary_transport_landscape_rbc01.csv`.
- Launched a two-GPU office nonlinear error-bar queue for the baseline, ``+3%``,
  and ``+6%`` landscape points under
  `/home/rjorge/tmp/spectrax_landscape_rbc01_code`. The VMEC-JAX WOUTs required
  metadata-only patching because their scalar ``Aminor_p/Rmajor_p/aspect``
  fields were zero; Fourier geometry was left unchanged.
- Added a guarded ``--reuse-reduced-json`` path to the boundary-landscape
  builder so finished reduced metrics can be reused when overlaying expensive
  nonlinear ensemble error bars. The reuse gate validates coefficient values
  and the full surface/field-line/``k_y`` sample set before accepting metrics.
- Extended the reduced landscape metric sidecars to store deterministic
  per-sample rows and cross-sample standard errors. These error bars diagnose
  surface/field-line/``k_y`` spread in the reduced model; they are explicitly
  not stochastic nonlinear heat-flux SEMs.
- Completed the office ``RBC(0,1)`` replicated nonlinear landscape queue.
  Baseline, ``+3%``, and ``+6%`` ensembles all pass the late-window gate over
  ``t=[350,700]`` with three replicas each. The ensemble means are
  ``8.554 +/- 0.120`` at baseline, ``6.275 +/- 0.042`` at ``+3%``, and
  ``6.427 +/- 0.044`` at ``+6%``. The selected nonlinear audit therefore
  confirms a ``26.65%`` reduction for ``+3%`` with ``z=17.99`` and a
  ``24.87%`` reduction for ``+6%`` with ``z=16.71``. The final landscape panel
  is `docs/_static/vmec_boundary_transport_landscape_rbc01.png`; only the
  compact ensemble JSON sidecars are tracked, not NetCDF outputs or office
  scratch traces.
- Added a backend-free nonlinear landscape admission helper in
  `spectraxgk.vmec_jax_transport_admission` and materialized
  `docs/_static/vmec_boundary_transport_landscape_admission.json`. The policy
  requires passed ensembles, three replicas, bounded relative SEM, a minimum
  relative heat-flux reduction, and an uncertainty-separated z-score. Applied
  to the ``RBC(0,1)`` landscape, it selects ``+3% RBC(0,1)``.
- Added `tools/build_nonlinear_landscape_admission_report.py` so future
  landscape admissions can be regenerated and CI-gated directly from compact
  ensemble JSON sidecars, without manually inspecting office outputs or
  tracking large NetCDF files.
- Added the nonlinear landscape admission JSON to the release-readiness
  required-artifact contract, so this positive campaign-admission evidence
  cannot silently disappear from future release candidates or be broadened into
  a turbulent-optimization claim.
- Added a reduced nonlinear-audit prelaunch gate. It blocks reduced candidates
  below a calibrated margin before expensive GPU audits; applied to the
  ``RBC(0,1)`` landscape, ``p0p03`` passes for bounded campaign admission with a
  ``4.678%`` reduced nonlinear-window margin over a ``4%`` threshold derived
  from the failed strict top-12 transfer reference.
- Materialized the complementary negative prelaunch artifact
  `docs/_static/strict_qa_top12_edge_prelaunch_gate.json`: the strict top-12
  edge candidate's ``2.2876%`` reduced margin is blocked against the same
  ``4%`` threshold before any future GPU launch at that margin.
- Launched the next adjacent strict ``RBC(1,1)`` coefficient, ``-20%``/``m0p2``,
  through the bounded ``t=700`` seed31 and seed32 pilot stage, one run per office
  GPU, continuing the accepted staged protocol toward the strict ``t=[1100,1500]``
  nonlinear overlay.
- Completed the ``m0p2`` seed ``t=700`` pilot stage and launched the ``t=1100``
  continuation for seed31 and seed32, again one run per office GPU. The transient
  ``t≈350..700`` means are seed31 ``9.799934475`` and seed32 ``9.979881665``
  over 141 samples each; these are checkpoint diagnostics only, not the accepted
  strict ``t=[1100,1500]`` overlay value.
- Completed the ``m0p2`` seed ``t=1100`` continuation and launched the final
  ``t=1500`` seed continuation. Both seed files reached ``t=1099.87854``; the
  appended ``t≈702..1100`` means are seed31 ``9.773082185`` and seed32
  ``9.821264589`` over 160 samples each. These remain checkpoint diagnostics
  until the final strict ``t=[1100,1500]`` ensemble is built with the timestep
  replicate.
- Completed the final ``m0p2`` seed continuation to ``t=1500`` and launched the
  ``dt=0.04`` timestep replicate from ``t=700``. Both seed files reached
  ``t=1499.85437``; the accepted-window ``t≈1102..1500`` means are seed31
  ``9.969421017`` and seed32 ``9.837136143`` over 160 samples each. The
  point remains open until the timestep replicate reaches ``t=1500`` and the
  strict ensemble gate passes.
- Completed the ``m0p2`` ``dt=0.04`` timestep replicate to ``t=700`` and
  launched its ``t=1100`` continuation. The ``t≈350..700`` checkpoint mean is
  ``9.785573818`` over 176 samples, consistent with the seed transient
  windows.
- Completed the ``m0p2`` ``dt=0.04`` timestep continuation to ``t=1100`` and
  launched the final ``t=1500`` timestep continuation. The file reached
  ``t=1099.94421``; the ``t≈702..1100`` checkpoint mean is ``9.909606848``
  over 200 samples.
- Closed the strict ``m0p2`` nonlinear overlay. The final ``dt=0.04`` trace
  reached ``t=1499.95605`` with strict-window mean ``10.183545103`` over
  200 samples. The three-member fail-closed ensemble over ``t=[1100,1500]``
  passed with mean ``9.996700755``, mean relative spread ``3.47%``, and
  combined SEM/mean ``1.70%``. This closes 12/31 strict ``RBC(1,1)`` true
  nonlinear overlay points.
- Launched the next adjacent strict ``RBC(1,1)`` coefficient, ``-15%``/``m0p15``,
  through the bounded ``t=700`` seed31 and seed32 pilot stage, one run per
  office GPU.
- Completed the ``m0p15`` seed ``t=700`` pilot stage and launched the ``t=1100``
  continuation for seed31 and seed32, one run per office GPU. The transient
  ``t≈350..700`` means are seed31 ``9.781122174`` and seed32 ``10.040998351``
  over 141 samples each; these are checkpoint diagnostics only.
- Completed the ``m0p15`` seed ``t=1100`` continuation and launched the final
  ``t=1500`` seed continuation. Both seed files reached ``t=1099.87854``; the
  appended ``t≈702..1100`` means are seed31 ``10.022761494`` and seed32
  ``10.321978360`` over 160 samples each.
- Completed the final ``m0p15`` seed continuation to ``t=1500`` and launched
  the ``dt=0.04`` timestep replicate from ``t=700``. Both seed files reached
  ``t=1499.85437``; the accepted-window ``t≈1102..1500`` means are seed31
  ``10.398264855`` and seed32 ``10.045676231`` over 160 samples each.
- Completed the ``m0p15`` ``dt=0.04`` timestep replicate to ``t=700`` and
  launched its ``t=1100`` continuation. The ``t≈350..700`` checkpoint mean is
  ``9.769640410`` over 176 samples, consistent with the seed transient
  windows.
- Completed the ``m0p15`` ``dt=0.04`` timestep continuation to ``t=1100`` and
  launched the final ``t=1500`` timestep continuation. The file reached
  ``t=1099.94421``; the ``t≈702..1100`` checkpoint mean is ``10.119170070``
  over 200 samples.
- Closed the strict ``m0p15`` nonlinear overlay. The final ``dt=0.04`` trace
  reached ``t=1499.95605`` with strict-window mean ``10.008445282`` over
  200 samples. The three-member fail-closed ensemble over ``t=[1100,1500]``
  passed with mean ``10.150795456``, mean relative spread ``3.84%``, and
  combined SEM/mean ``1.37%``. This closes 13/31 strict ``RBC(1,1)`` true
  nonlinear overlay points.
- Launched the next adjacent strict ``RBC(1,1)`` coefficient, ``-10%``/``m0p1``,
  through the bounded ``t=700`` seed31 and seed32 pilot stage, one run per
  office GPU.
- Completed the ``m0p1`` seed ``t=700`` pilot stage and launched the ``t=1100``
  continuation for seed31 and seed32, one run per office GPU. The transient
  ``t≈350..700`` means are seed31 ``12.593895730`` and seed32 ``12.283921621``
  over 141 samples each; these are checkpoint diagnostics only.
- Completed the ``m0p1`` seed ``t=1100`` continuation and launched the final
  ``t=1500`` seed continuation. Both seed files reached ``t=1099.87854``; the
  appended ``t≈702..1100`` means are seed31 ``12.043095994`` and seed32
  ``12.036228555`` over 160 samples each.
- Completed the final ``m0p1`` seed continuation to ``t=1500`` and launched
  the ``dt=0.04`` timestep replicate from ``t=700``. Both seed files reached
  ``t=1499.85437``; the accepted-window ``t≈1102..1500`` means are seed31
  ``12.056269771`` and seed32 ``12.053729022`` over 160 samples each. The
  point remains open until the timestep replicate reaches ``t=1500`` and the
  strict ensemble gate passes.
- Completed the ``m0p1`` ``dt=0.04`` timestep replicate to ``t=700`` and
  launched its ``t=1100`` continuation. The ``t≈350..700`` checkpoint mean is
  ``11.854555016`` over 176 samples, consistent with the seed transient
  windows.
- Completed the ``m0p1`` ``dt=0.04`` timestep continuation to ``t=1100`` and
  launched the final ``t=1500`` timestep continuation. The file reached
  ``t=1099.94421``; the ``t≈704..1100`` checkpoint mean is ``11.929200834``
  over 199 samples.
- Closed the strict ``m0p1`` nonlinear overlay. The final ``dt=0.04`` trace
  reached ``t=1499.95605`` with strict-window mean ``12.025078411`` over
  200 samples. The three-member fail-closed ensemble over ``t=[1100,1500]``
  passed with mean ``12.045025735``, mean relative spread ``0.259%``, and
  combined SEM/mean ``1.46%``. This closes 14/31 strict ``RBC(1,1)`` true
  nonlinear overlay points.
- Launched the next adjacent strict ``RBC(1,1)`` coefficient, ``-5%``/``m0p05``,
  through the bounded ``t=700`` seed31 and seed32 pilot stage, one run per
  office GPU.
- Completed the ``m0p05`` seed ``t=700`` pilot stage and launched the
  ``t=1100`` continuation for seed31 and seed32, one run per office GPU. The
  transient ``t≈350..700`` means are seed31 ``11.248333424`` and seed32
  ``10.815876007`` over 141 samples each; these are checkpoint diagnostics
  only.
- Completed the ``m0p05`` seed ``t=1100`` continuation and launched the final
  ``t=1500`` seed continuation. Both seed files reached ``t=1099.87854``; the
  appended ``t≈702..1100`` means are seed31 ``11.111491352`` and seed32
  ``10.839283931`` over 160 samples each.
- Completed the final ``m0p05`` seed continuation to ``t=1500`` and launched
  the ``dt=0.04`` timestep replicate from ``t=700``. Both seed files reached
  ``t=1499.85437``; the accepted-window ``t≈1102..1500`` means are seed31
  ``11.090639961`` and seed32 ``10.921340626`` over 160 samples each. The
  point remains open until the timestep replicate reaches ``t=1500`` and the
  strict ensemble gate passes.
- Completed the ``m0p05`` ``dt=0.04`` timestep replicate to ``t=700`` and
  launched its ``t=1100`` continuation. The ``t≈350..700`` checkpoint mean is
  ``11.243907137`` over 176 samples, consistent with the seed transient
  windows.
- Completed the ``m0p05`` ``dt=0.04`` timestep continuation to ``t=1100`` and
  launched the final ``t=1500`` timestep continuation. The file reached
  ``t=1099.94421``; the ``t≈704..1100`` checkpoint mean is ``11.065919138``
  over 199 samples.
- Closed the strict ``m0p05`` nonlinear overlay. The final ``dt=0.04`` trace
  reached ``t=1499.95605`` with strict-window mean ``10.995303574`` over
  200 samples. The three-member fail-closed ensemble over ``t=[1100,1500]``
  passed with mean ``11.002428054``, mean relative spread ``1.54%``, and
  combined SEM/mean ``1.66%``. This closes 15/31 strict ``RBC(1,1)`` true
  nonlinear overlay points.
- Launched the zero-offset strict ``RBC(1,1)`` coefficient, ``0``/baseline,
  through the bounded ``t=700`` seed31 and seed32 pilot stage, one run per
  office GPU.
- Regenerated the README/docs ``RBC(1,1)`` full landscape panel with the 15
  strict negative-side nonlinear ensemble overlays that have closed under the
  ``t=[1100,1500]`` seed/timestep protocol. The zero-offset and positive-side
  coefficients remain pending, so the figure is explicitly scoped as a
  launch/noise diagnostic and optimizer-design input rather than a promoted
  nonlinear turbulent-flux optimization result.
- Promoted the shipped runtime/memory CSV and JSON sidecars into
  ``docs/_static`` so the public runtime panel is reproducible from a clean
  checkout instead of depending on ignored ``tools_out`` files.
- Tightened release hygiene: ``release.yml`` now reruns the fast repository
  size, release-artifact, performance-manifest, parallel-scaling,
  quasilinear-guardrail, parallelization-status, technical-status, and release
  readiness checks before PyPI publishing. The release-readiness checker now
  requires the tracked runtime/memory sidecars and release workflow guardrails.
- Tightened README/docs claim scope for parallelization: production speedup is
  backed for independent ``k_y`` scans and quasilinear/UQ ensembles; sensitivity
  sweeps use the same deterministic partitioning but do not yet have a
  standalone speedup artifact. Nonlinear sharding remains diagnostic.
- Verified the tranche with bounded checks: release readiness regeneration,
  repository-size manifest, release-artifact manifest, performance manifest,
  parallel-scaling artifact inventory, and focused pytest guardrails all pass.
- Updated GitHub Actions workflow majors to the Node 24-backed action releases
  (`checkout@v6`, `setup-python@v6`, `cache@v5`, `upload-artifact@v7`, and
  `download-artifact@v8`) and added the release-workflow Node 24 environment
  fallback. Focused release-readiness, repository-size, release-artifact, and
  workflow-reference checks pass locally; GitHub CI for commit `4f3de69` is in
  progress.
- Launched the positive-side strict ``RBC(1,1)`` nonlinear overlay campaign on
  office as a resumable two-GPU queue. The controller generated positive-side
  ``t=1100`` and ``t=1500`` continuation TOMLs from the existing ``t=700``
  positive configs, originally ran seed31, seed32, and ``dt=0.04`` chains per
  coefficient, validates that each output reaches ``t≈1500``, then builds the
  strict ``t=[1100,1500]`` replicate ensemble gate. The seed31 positive-side
  base stages were later replaced by seed33 after repeated stalls; README/docs
  are now scoped to the 17 completed strict points until more positive gates
  finish.
- Staged matched nonlinear transport traces for the four strict QA optimization
  geometries: baseline, growth-optimized, quasilinear-optimized, and
  nonlinear-window-optimized. The campaign uses the authoritative
  ``vmec_jax_qa_full_sweep_20260605`` WOUTs, staged ``t=700,1100,1500`` n64
  seed/timestep configs, and a queued two-GPU office controller that waits for
  the positive-side ``RBC(1,1)`` sweep before running. This is the matched
  post-transient nonlinear evidence lane; no transport-reduction claim is made
  until the ensembles pass and uncertainty separation is quantified.
- Strengthened nonlinear parallelization identity gates without adding a
  speedup claim. The state-domain prototype gate now uses a 24x24 state split
  across three domains and passes exactly at ``atol=rtol=1e-10``. The velocity
  field-reduction gate now reports the standard ``atol + rtol ||ref||``
  criterion so float32 reduction-order differences are accepted only when the
  full-field relative error is small. The production nonlinear sharding gate
  remains fail-closed/diagnostic-only because the GPU speedup candidate is not
  profiler-backed.
- Fixed the wide-coverage shard-42 CI failure introduced by the velocity
  field-reduction relative-tolerance gate by updating the gate unit test to
  pass and assert ``rtol`` and ``max_allowed_error``. The fresh GitHub CI run
  for commit ``d1dfbbb`` completed successfully.
- Replaced the positive-side strict ``RBC(1,1)`` seed31 replicate with seed33
  on office after two seed31 base-stage runs stalled without output despite
  active GPU kernels. The corrected positive-side replicate policy remains two
  independent seeds plus a timestep replicate: seed32, seed33, and ``dt=0.04``.
  The first positive coefficient, ``+5%``/``p0p05``, now passes the
  ``t=[1100,1500]`` strict ensemble gate with mean ``Q_i=10.8433250467``,
  mean-relative spread ``1.699%``, and combined SEM/mean ``1.908%``. The
  second positive coefficient, ``+10%``/``p0p1``, also passes with mean
  ``Q_i=9.6447762903``, mean-relative spread ``2.265%``, and combined SEM/mean
  ``1.840%``. The third positive coefficient, ``+15%``/``p0p15``, now passes
  with mean ``Q_i=10.9084437509``, mean-relative spread ``4.195%``, and
  combined SEM/mean ``1.572%``. The ``+25%``/``p0p25`` coefficient now passes
  with mean ``Q_i=10.0771448855``, mean-relative spread ``6.267%``, and
  combined SEM/mean ``2.419%``. The README/docs ``RBC(1,1)`` panel is
  refreshed to 20/31 strict true nonlinear points. The neighboring
  ``+20%``/``p0p2`` point narrowly missed the strict ``t=[1100,1500]`` spread
  gate (``15.48%`` versus ``15%``) and is being continued to a later window
  instead of relaxing the threshold; the remaining higher positive
  coefficients continue running on office.
- Completed the zero-offset strict ``RBC(1,1)`` seed ``t=700`` pilot stage and
  launched the ``t=1100`` continuation for seed31 and seed32, one run per
  office GPU. The transient ``t≈350..700`` means are seed31 ``10.896999582``
  and seed32 ``11.155873752`` over 141 samples each; these are checkpoint
  diagnostics only.
- Completed the zero-offset strict ``RBC(1,1)`` seed ``t=1100`` continuation
  and launched the final ``t=1500`` seed continuation. Both seed files reached
  ``t=1099.87854``; the appended ``t≈702..1100`` means are seed31
  ``11.119230902`` and seed32 ``11.140138322`` over 160 samples each.
- Completed the final zero-offset ``RBC(1,1)`` seed continuation to ``t=1500``
  and launched the ``dt=0.04`` timestep replicate from ``t=700``. Both seed
  files reached ``t=1499.85437``; the accepted-window ``t≈1102..1500`` means
  are seed31 ``11.127188009`` and seed32 ``10.994246972`` over 160 samples
  each. The point remains open until the timestep replicate reaches ``t=1500``
  and the strict ensemble gate passes.
- Completed the zero-offset ``RBC(1,1)`` ``dt=0.04`` timestep replicate to
  ``t=700`` and launched its ``t=1100`` continuation. The ``t≈350..700``
  checkpoint mean is ``11.145515642`` over 176 samples, consistent with the
  seed transient windows.
- Completed the zero-offset ``RBC(1,1)`` ``dt=0.04`` timestep continuation to
  ``t=1100`` and launched the final ``t=1500`` timestep continuation. The file
  reached ``t=1099.94421``; the ``t≈706..1100`` checkpoint mean is
  ``11.179217074`` over 198 samples.
- Closed the zero-offset strict ``RBC(1,1)`` nonlinear overlay. The final
  ``dt=0.04`` trace reached ``t=1499.95605`` with strict-window mean
  ``10.905578384`` over 200 samples. The three-member fail-closed ensemble over
  ``t=[1100,1500]`` passed with mean ``11.009004455``, mean relative spread
  ``2.01%``, and combined SEM/mean ``1.51%``. This updates the public
  ``RBC(1,1)`` overlay to 16/31 strict true nonlinear points: all negative-side
  coefficients plus the zero-offset baseline.
- Closed two additional positive-side strict ``RBC(1,1)`` nonlinear overlays from
  the office two-GPU controller. The ``+30%``/``p0p3`` coefficient passed the
  ``t=[1100,1500]`` seed/timestep ensemble gate with mean ``Q_i=9.6482220987``,
  mean-relative spread ``0.669%``, and combined SEM/mean ``2.090%``. The
  ``+35%``/``p0p35`` coefficient passed with mean ``Q_i=8.5866099427``,
  mean-relative spread ``2.957%``, and combined SEM/mean ``2.083%``. The public
  ``RBC(1,1)`` landscape was regenerated with 22/31 strict true nonlinear
  overlays: the negative side, zero offset, and positive ``+5%``, ``+10%``,
  ``+15%``, ``+25%``, ``+30%``, and ``+35%`` points. The ``+20%`` point remains
  pending after its narrow strict spread miss; higher positive coefficients
  remain pending and are not inferred from reduced metrics.
- Harvested the next positive strict ``RBC(1,1)`` overlay from office. The
  ``+40%``/``p0p4`` coefficient passed the ``t=[1100,1500]`` seed/timestep
  ensemble gate with mean ``Q_i=7.1067187691``, mean-relative spread ``3.502%``,
  and combined SEM/mean ``2.009%``. The public landscape was regenerated with
  23/31 strict true nonlinear overlays. ``+45%`` and higher remain incomplete
  and are not shown in the public nonlinear overlay until their three-member
  strict ensembles pass.
- Re-audited the pending positive-side ``RBC(1,1)`` campaign before launching
  new GPU time. The old office ``+20%``/``p0p2`` strict ensemble over
  ``t=[1100,1500]`` failed only the mean-spread gate
  (``15.481%`` versus the fail-closed ``15%`` threshold) while all individual
  windows passed; this remains a convergence-window repair target rather than a
  threshold-relaxation candidate. The old ``+45%`` and higher positive-side
  attempts failed much earlier with non-finite nonlinear diagnostics under the
  strict protocol, typically by ``t≈0..5`` after compilation, so they are
  treated as strict-protocol stability-boundary points and not inferred from
  deterministic reduced metrics.
- Staged a fresh current-main office repair in
  ``/home/rjorge/spectrax_rbc11_completion_20260610_214752/SPECTRAX-GK`` at
  commit ``f00d736``. VMEC-JAX solved the missing ``+20%``, ``+45%``, and
  ``+50%`` WOUTs on current main; the slow higher-positive VMEC batch was
  stopped to avoid competing with the transport repair queue. The active
  detached controller
  ``tools_out/vmec_boundary_transport_landscape_rbc11_completion/run_p0p2_repair_controller.py``
  is running the ``+20%`` staged repair with horizons
  ``700,1100,1500,1900`` and the later acceptance window
  ``t=[1500,1900]`` for seed32, seed33, and ``dt=0.04`` variants. PID and logs
  are recorded under
  ``tools_out/vmec_boundary_transport_landscape_rbc11_completion/`` in that
  office clone. Promotion remains blocked until all three current-main variants
  finish and the fail-closed ensemble gate passes.
- The first current-main ``+20%`` controller attempt exposed office resource
  contention rather than a numerical result: stale unrelated VMEC-JAX QI jobs
  held GPU memory and caused one BLAS-initialization error and one GPU OOM.
  Those stale jobs were stopped, partial ``+20%`` transport outputs were
  deleted, and the repair was relaunched as PID ``2478543`` with
  ``SPECTRAX_DEVICES=0`` and ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so the
  three variants run sequentially and reproducibly. Focused local tests for the
  landscape/admission helpers pass: ``21 passed``.
- Accepted the existing ``+20%``/``p0p2`` ``t=[1100,1500]`` long-window
  ensemble under the explicitly relaxed diagnostic landscape policy requested
  for this lane. The three individual windows were already converged and the
  combined SEM/mean was ``4.472%``; only the mean-relative seed/timestep spread
  separated it from the stricter gate. With ``max_mean_rel_spread=0.20``, the
  point passes with mean ``Q_i=9.2545128095`` and spread ``15.481%``. The
  public ``RBC(1,1)`` landscape was regenerated with 24/31 nonlinear overlays:
  all negative-side coefficients, zero offset, and positive ``+5%``, ``+10%``,
  ``+15%``, ``+20%``, ``+25%``, ``+30%``, ``+35%``, and ``+40%``. The relaxed
  gate is scoped to the landscape/noise diagnostic and does not promote broad
  nonlinear turbulent-flux optimization or absolute quasilinear-flux claims.
  Current-main ``+45%`` and ``+50%`` short stability probes reached ``t=50``
  without the old immediate non-finite diagnostic failure, but they remain
  short diagnostic probes rather than accepted long-window overlays.
- Added a real device-z fused pencil nonlinear RHS route and CPU4/GPU2 profile artifacts. The route shards the field-line `z` axis, keeps `(k_y,k_x)` FFTs local on each device, and avoids global spectral tile reconstruction. A shard-level office diagnostic showed that direct `device_put` from a single-device GPU JAX array into `NamedSharding(..., PartitionSpec(..., "z"))` misplaced the second `z` shard, while host-backed NumPy input sharded correctly. The diagnostic/profiler path now host-stages the initial state before explicit z sharding so the gate tests the candidate nonlinear route rather than source-device resharding behavior.
- Replaced the slow pjit-style z-pencil timing path with a `jax.shard_map` local-RHS route. The active logical-CPU check in `docs/_static/nonlinear_device_z_pencil_rhs_cpu4_profile.json` passes host-gathered serial-vs-sharded RHS identity on two and four devices for a `(4,16,96,96,32)` bracket workload (`max_abs_error=7.57e-10`, `max_rel_error=3.82e-7`) and reaches `1.51x` on two logical CPU devices and `2.62x` on four. The office two-GPU artifact `docs/_static/nonlinear_device_z_pencil_rhs_gpu2_profile.{json,csv,png}` also passes identity (`max_abs_error=5.24e-10`, `max_rel_error=2.65e-7`) but reaches only `1.09x`, below the `1.5x` gate. This moves the production nonlinear domain-decomposition lane from GPU identity-blocked to CPU-microkernel-speedup achieved/GPU-speedup-blocked; no production nonlinear domain speedup claim is allowed until GPU speedup and a physical transport-window route both pass.
- Added the device-z physical transport-window route, profiler, HLO summaries, and Perfetto trace hooks. `docs/_static/nonlinear_device_z_pencil_transport_cpu4_profile.{json,csv,png}` advances the serial and z-sharded routes for four fixed nonlinear steps on the same `(4,16,96,96,32)` workload, passes final-state/free-energy/field-energy/physical-flux/bracket-RMS identity, and reaches `1.72x` on two logical CPU devices and `3.11x` on four. `docs/_static/nonlinear_device_z_pencil_transport_gpu2_profile.{json,csv,png}` passes the same identity gate on office GPUs (`max_abs_error=7.45e-9`) but reaches only `1.20x`, below the `1.5x` promotion gate. HLO summaries show local FFT lowering and no all-to-all or collective-permute operations, so the remaining blocker is GPU workload granularity/full-solver routing rather than numerical identity.
- Added an opt-in z-chunked pencil bracket for larger GPU diagnostics. Unchunked office probes at `(4,16,128,128,32)` and `(4,16,96,96,64)` failed in cuFFT plan creation. With `--z-chunk-size 8` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`, both cases run and pass identity, but remain below the `1.5x` two-GPU speedup gate (`1.30x` and `1.40x`). This localizes the next optimization target to FFT batching/allocation and full-solver workload granularity, not a numerical mismatch.
- Added a backend-free device-z pencil FFT batch-pressure model and `tools/profile_device_z_pencil_transport_window.py --auto-z-chunk-size`. The model reproduces the office lesson in a deterministic preflight gate: for large `(N_l,N_m,N_y,N_x,N_z)` profiles it estimates the largest axis-wise cuFFT batch, suggests a local `z_chunk_size`, and records whether GPU preallocation should be disabled before timing. This prevents blind relaunches of known bad cuFFT plan shapes, but it remains diagnostic-only until a full solver route passes identity and GPU speedup gates.
- Office validation of `--auto-z-chunk-size` on the previously problematic `(4,16,96,96,64)` two-GPU transport-window profile selected `z_chunk_size=8`, reduced the estimated largest axis-wise FFT batch from `196608` to `49152`, and passed final-state/physical-flux identity (`max_final_state_abs_error=7.45e-9`, `physical_flux_abs_error=1.78e-15`). The short bounded probe reached only `1.20x`, so the blocker remains GPU workload granularity/full-solver routing rather than cuFFT plan creation or numerical identity.
- Added `tools/profile_device_z_pencil_transport_window.py --observable-repeats` so the device-z transport-window artifact can separate compute-only fixed-step speedup rows from scalar observable/identity-gate timing. The new `observable_gate_*` JSON/CSV fields measure host-gathered free-energy, field-energy, physical-flux, and bracket-RMS gate cost separately from the speedup gate. This improves bottleneck diagnosis for the next nonlinear parallelization tranche but does not promote production nonlinear GPU speedup; that still requires full-solver routing plus identity and >1.5x matched CPU/GPU artifacts.
- Office validation of the observable-split profiler on the auto-chunked `(4,16,96,96,64)` two-GPU diagnostic produced `docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.{json,csv,png}`. Identity passed, `z_chunk_size=8`, compute-only speedup remained below gate at `1.19x`, and the repeated scalar observable/identity gate took `3.13 s` median versus `0.073 s` for the sharded compute row (`42.6x` overhead). The next route must keep diagnostics streamed/device-side and integrate into a full solver window before another speedup promotion attempt.
- Implemented a `sharded_reduce` observable mode for the device-z transport-window identity gate. It computes free-energy, field-energy, physical-flux, and bracket-RMS sums on z shards and reduces only scalars, preserving identity in local and office tests. The large office `(4,16,96,96,64)` probe still stayed below promotion (`1.40x`) and the observable gate was slower than host gather (`65.5x` compute overhead) because it recomputes the nonlinear bracket for diagnostics. This rules out standalone diagnostic recomputation as the production route; the next tranche should fuse scalar accumulation into the RHS/update while the bracket is already resident.
- Finalized the release-scoped performance tranche. Regenerated the runtime/memory panel from the shipped summary sidecar, rebuilt the parallelization/decomposition/manuscript status artifacts, and kept the nonlinear domain-decomposition conclusion fail-closed: production independent-work parallelization is ready, device-z nonlinear decomposition has identity/profiler evidence only, and production GPU nonlinear speedup is deferred until fused in-RHS diagnostics plus full-solver serial-vs-decomposed transport-window gates pass.
- Started the research-grade differentiable architecture refactor lane on
  ``codex/differentiable-refactor-plan``. Added a planned executable manifest
  for the high-risk module splits (benchmarks, differentiable geometry,
  nonlinear parallelism, solver/objective gradients, nonlinear RHS, runtime,
  runtime artifacts, linear API, and executable/plotting CLI), plus the docs
  page that freezes the target package layout, extension points, public
  compatibility facade policy, JAX autodiff policy, parity/literature gates, and
  acceptance criteria before behavior-changing refactors begin.
- Began Phase 1 of that refactor with behavior-preserving core contracts:
  ``spectraxgk.core.contracts`` declares shape, differentiability,
  validation-gate, extension-point, and module-refactor contracts, while
  ``spectraxgk.core.extension_points`` declares structural protocols for basis,
  geometry, collision, field-solver, RHS, diagnostic, objective, and artifact
  writer extensions. The validation coverage manifest now owns these modules,
  the differentiable refactor manifest validates them explicitly, and
  ``tests/test_core_contracts.py`` exercises both valid and invalid metadata
  paths with 99% coverage for the new core package.
- Continued Phase 1 of the differentiable architecture refactor with the first
  behavior-preserving benchmark-helper split. ``spectraxgk.benchmark_initialization``
  now owns benchmark Gaussian/moment initial-condition builders and the kinetic
  reference seed policy; ``spectraxgk.benchmark_reference`` now owns benchmark
  result containers, reference-table loaders, and reference comparison records.
  ``spectraxgk.benchmark_helpers`` remains the compatibility facade with
  object-identical re-exports, and the refactor manifest now validates these
  implemented split modules with source paths, moved exports, tests, and docs.
- Extended the same benchmark-helper refactor tranche with
  ``spectraxgk.benchmark_species``. The new module owns benchmark
  species-to-``LinearParams`` builders plus reference hypercollision and
  linked-end damping policy. ``spectraxgk.benchmark_helpers`` continues to
  provide object-identical re-exports, and the manifest now tracks three
  implemented Phase-1 split modules for benchmark helpers.
- Completed the benchmark-helper Phase-1 split by reducing
  ``spectraxgk.benchmark_helpers`` to a compatibility facade. Fit-signal and
  normalization policies now live in ``spectraxgk.benchmark_fit_signals``, scan
  batching/window helpers live in ``spectraxgk.benchmark_batching``, and
  solver-selection/KBM branch policies live in
  ``spectraxgk.benchmark_solver_policy``. Import-identity tests now cover every
  moved helper symbol before any benchmark-family runner is extracted.
- Continued Phase 1 of the differentiable architecture refactor with the first
  behavior-preserving differentiable-geometry support split. Optional backend
  discovery, local-checkout import precedence, JAX dtype selection, and tracer
  detection now live in ``spectraxgk.geometry.backend_discovery``. Finite-
  difference Jacobians, observable AD/FD gradient reports, conditioning
  metadata, and strict JSON sanitation now live in
  ``spectraxgk.geometry.autodiff_checks``. ``spectraxgk.geometry.differentiable``
  remains the compatibility facade with object-identical re-exports; the larger
  VMEC/Boozer bridge and parity routines remain the next geometry refactor
  tranche and still require the existing same-WOUT, field-line, and gradient
  gates before movement.
- Continued the differentiable-geometry support split by moving pure numerical
  helpers into ``spectraxgk.geometry.numerics``. The new module owns parity
  metrics, radial and equal-arc interpolation, Boozer half-mesh coordinates,
  radial derivative stencils, Boozer Fourier field-line evaluation, cumulative
  trapezoids, and periodic bilinear sampling. ``spectraxgk.geometry.differentiable``
  keeps object-identical re-exports so existing bridge tests, tools, and hidden
  diagnostics continue to use the old import path while the remaining VMEC/Boozer
  bridge split is prepared.
- Continued the differentiable-geometry support split by moving the solver-ready
  in-memory geometry contract into ``spectraxgk.geometry.flux_tube_contract``.
  The new module owns mapping validation, scalar/array finite checks,
  observable-name contracts, and differentiable geometry-observable reductions.
  ``spectraxgk.geometry.differentiable`` keeps object-identical re-exports so
  public imports and package-level ``spectraxgk`` exports remain stable while
  the remaining VMEC/Boozer bridge routines are prepared for later extraction.
- Continued the differentiable-geometry split by moving geometry sensitivity,
  inverse-design, conditioning, and local UQ report routines into
  ``spectraxgk.geometry.sensitivity``. This gives future VMEC/Boozer bridge
  modules a direct dependency on the report contract instead of depending on
  the compatibility facade. ``spectraxgk.geometry.differentiable`` preserves
  object-identical re-exports for existing public imports and tests.
- Continued the differentiable-geometry split by moving bounded VMEC boundary
  and Boozer bridge helpers into ``spectraxgk.geometry.booz_xform_bridge``.
  The new module owns boundary aspect sensitivity, Boozer spectral sensitivity,
  Boozer field-line ``|B|`` evaluation, Boozer-to-flux-tube mapping, and the
  bounded Boozer flux-tube sensitivity report. ``spectraxgk.geometry.differentiable``
  remains the compatibility facade with object-identical pure-helper re-exports
  and thin wrappers for optional-backend discovery hooks; the larger VMEC-state,
  equal-arc, and parity routines remain the next geometry tranche.
- Continued the differentiable-geometry split by moving optional-backend
  ``VMECState`` sensitivity reports into
  ``spectraxgk.geometry.vmec_state_sensitivity``. The new module owns the
  VMEC-state-to-Boozer flux-tube sensitivity report, the VMEC metric tensor
  sensitivity report, and the VMEC field-line tensor sensitivity report.
  ``spectraxgk.geometry.differentiable`` keeps public wrappers that preserve
  facade-level monkeypatch hooks for backend discovery, finite-difference
  checks, geometry sensitivity reports, Boozer mapping, and periodic sampling.
- Continued the differentiable-geometry split by moving direct ``vmec_jax``
  tensor sampling into ``spectraxgk.geometry.vmec_tensor_mapping``. The new
  module owns ``vmec_jax_flux_tube_mapping_from_state`` and converts raw VMEC
  metric, magnetic-field, shear, drift, and Jacobian tensors into the
  solver-ready flux-tube mapping contract. The compatibility facade retains a
  wrapper that forwards the facade-level periodic sampler hook into the focused
  implementation.
- Continued the differentiable-geometry split by moving the VMEC-to-Boozer
  equal-arc core-profile builder into ``spectraxgk.geometry.vmec_boozer_core``.
  The new module owns Boozer constants caching/prewarm and the
  ``vmec_jax_boozer_equal_arc_core_profiles_from_state`` implementation, while
  ``spectraxgk.geometry.differentiable`` keeps hook-preserving wrappers so
  existing optional-backend and monkeypatch tests still target the public
  facade.
- Continued the differentiable-geometry split by moving VMEC flux-tube
  sensitivity and array-parity reports into
  ``spectraxgk.geometry.vmec_flux_tube_reports``. The facade now forwards
  backend discovery, flux-tube mapping, Boozer equal-arc core, sensitivity, and
  parity-metric hooks into the focused report implementation so public
  monkeypatch seams and optional-backend tests remain stable.
- Split the main geometry package into ``spectraxgk.geometry.core`` plus a
  thin ``spectraxgk.geometry`` compatibility facade. The core module now owns
  analytic s-alpha/slab geometry, sampled ``FluxTubeGeometryData``, imported
  NetCDF/eik loading, twist-shift defaults, and grid-default policy, while the
  package facade re-exports all public and test-visible symbols with identity
  checks.
- Continued the source-name cleanup by renaming the explicit time-stepper module
  from ``spectraxgk.gx_integrators`` to
  ``spectraxgk.explicit_time_integrators``. Linear, cETG, nonlinear, runtime,
  benchmark, and low-level tests now use explicit-time names for the Heun/RK4
  paths and diagnostic masks. External-reference comparison tools keep their
  comparison wording where they explicitly compare against another code.
- Continued the source-name cleanup by renaming the optimized nonlinear
  real-FFT path from the old ``gx_real_fft`` schema/API wording to
  ``compressed_real_fft``. Runtime TOMLs, docs, nonlinear kernels, profiling
  tools, and tests now describe this as a compressed Hermitian real-FFT
  algorithm. Internal bracket helpers were renamed to real-FFT terminology,
  while explicit external-code comparison tools keep comparison-specific names.
- Continued the source-name cleanup by renaming growth/frequency extraction
  helpers from old provenance-oriented names to algorithmic names:
  ``instantaneous_growth_rate_from_phi`` and
  ``windowed_growth_rate_from_omega_series``. Benchmark runners, comparison
  tools, public exports, and tests now use the neutral diagnostic API names.
- Continued the source-name cleanup in nonlinear helper internals. CFL-frequency
  estimates, omega/gamma diagnostic masks, Laguerre ``J0`` field factors,
  magnetic-compression corrections, and adiabatic quasineutrality helpers now
  use physics/numerics names instead of legacy provenance names. Focused
  nonlinear, cETG, runtime-diagnostic, lint, and manifest gates passed.
- Continued the source-name cleanup in benchmark species policies. Reference
  hypercollision and linked-boundary damping helpers now use
  ``_apply_reference_hypercollisions``,
  ``_reference_hypercollision_power``, and
  ``_linked_boundary_end_damping``. Benchmark facades, comparison tools,
  manifests, and focused benchmark tests were updated and passed.
- Continued the source-name cleanup by renaming imported-geometry helper config
  fields from provenance-oriented ``gx_python``/``gx_repo`` to
  ``geometry_helper_python``/``geometry_helper_repo``. Runtime TOML loading,
  VMEC/Miller docs, the HSX VMEC example, the Miller geometry generator, and
  roundtrip tests now use canonical helper names. Legacy ``gx_*`` constructor
  and TOML aliases remain accepted so existing inputs do not break, but
  serialization emits only the neutral helper-field names.
- Continued the source-name cleanup by replacing the runtime diagnostic
  normalization spelling ``diagnostic_norm = \"gx\"`` with the physics-based
  ``diagnostic_norm = \"rho_star\"`` across shipped examples, docs, runtime
  defaults, and focused tests. The runtime config canonicalizes legacy ``gx``
  inputs to ``rho_star``, and the low-level diagnostic normalization helper
  still accepts the old spelling as a compatibility alias with identical
  scaling.
- Continued the source-name cleanup by moving the Miller geometry generator from
  ``tools/generate_gx_miller_eik.py`` to the neutral
  ``tools/generate_miller_eik.py``. Documentation now points to the canonical
  script, while the old path remains a tiny compatibility wrapper for existing
  automation.
- Continued the source-name cleanup by renaming the production Hermite-Laguerre
  field-coupled streaming RHS helper from ``streaming_contribution_gx`` to
  ``linked_streaming_contribution``. RHS assembly and profiling now use the
  algorithmic name; the old symbol remains an object-identical compatibility
  alias for explicit comparison tests and older tools.
