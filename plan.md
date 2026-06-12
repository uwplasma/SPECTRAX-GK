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
     optimized-equilibrium ensembles, and at least four replicated nonlinear
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
  well tracked. This points the next QL work toward richer saturation physics
  and one additional independent converged holdout, not threshold loosening.

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
  GX-style diagnostic reductions multiplied energy/flux factors by the dealias
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
  is claimable now, while universal absolute-flux or screening promotion still
  requires additional independent, replicated, post-transient nonlinear
  holdouts.

- Extended the quasilinear holdout-gap report to ingest the screening-skill
  sidecar. The refreshed `docs/_static/quasilinear_holdout_gap_report.*` now
  carries both `absolute_flux_promotion_requirements` and
  `screening_promotion_requirements`: full-portfolio and held-out-only
  rank/correlation screening pass for `spectral_envelope_ridge`, but the same
  next evidence item is required before either screening or absolute-flux
  promotion can be reconsidered: additional independent, replicated,
  post-transient nonlinear holdouts.

# SPECTRAX-GK Active Plan and Running Log

Last updated: 2026-06-12
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
| Quasilinear screening/model-development | 99.5% | Scoped model-development artifacts are current; the shaped-pressure holdout demotes rank/correlation screening, so no screening model is currently accepted |
| Universal absolute quasilinear-flux prediction | 60% | Strict pre-manuscript gate remains partial: train/holdout absolute report fails, holdout mean relative error is `3.13 > 0.35`, candidate uncertainty/model-selection fail, and no accepted runtime absolute-flux candidate exists |
| Nonlinear holdout expansion/audits | 96% | Eight admitted holdouts; CTH-like and shaped-pressure are admitted only under scoped high-grid policies, QH warm-start is closed as negative high-grid evidence through corrected `t700`, and the next independent Solovev nonlinear holdout is staged on office but not admitted |
| Rerun-WOUT admission and artifact policy | 100% | Explicit authoritative rerun-WOUT path implemented and tested |
| Strict QA candidate screening | 100% | Top-12 projected edge candidate passes rerun-WOUT gates and reduces the 18-point metric by 2.29% |
| Strict nonlinear transport and campaign-admission evidence | 100% | Strict top-12 matched audit fails promotion; historical full-sweep QA audit is negative evidence; true t=1500 baseline/growth/quasilinear/nonlinear-window triplets pass, but all three matched candidate comparisons fail the 4% reduction gate |
| Boundary-coefficient landscape and optimizer-noise diagnosis | 99% | 31-point RBC(1,1) reduced linear/QL landscape is tracked; 24 true long-window nonlinear overlays pass the scoped diagnostic gates; `+20%` is admitted under an explicit 20% spread gate, while `+45%` and higher remain stability-boundary/open long-window points |
| Differentiable QA optimization evidence | 93% | Full VMEC/Boozer reduced-gradient and true `t=1500` matched-audit plumbing are tracked; a new solved-WOUT candidate screen prevents invalid metric/high-growth artifacts from entering nonlinear launches; successful broad nonlinear turbulent-flux optimization is still not promoted |
| Broad end-to-end nonlinear turbulent-flux stellarator optimization | 54.2% | Strict pre-manuscript gate is blocked until at least three matched optimized transport audits, three optimized-equilibrium ensembles, four replicated holdout ensembles, and one production-scope VMEC/Boozer held-out nonlinear transport artifact pass |
| VMEC/Boozer holdout optimization | 78% | Reduced alpha/surface and second-equilibrium gates pass, but aggregate promotion fails because no production-scope held-out surface/field-line nonlinear transport artifact qualifies |
| Docs/readme/release hygiene | 100% | Public wording separates reduced linear/QL landscape metrics from true nonlinear heat-flux evidence; strict-QA t1500, CTH high-grid, and QL holdout-gap artifacts are tracked |
| Performance/parallelization release lane | 96% | Independent-work parallel paths are release-ready; nonlinear sharding profiler provenance is versioned and checker-gated, while whole-state/domain speedup remains diagnostic |
| Production nonlinear domain-decomposition speedup | 55% | Strict pre-manuscript gate remains partial: local and spectral identity pass, but combined strong-scaling speedup and production-speedup gates fail; CPU and GPU speedup are below `1.5x` |
| QA optimization optimizer-comparison metadata | 100% | Public examples emit strict nonlinear audit manifests; optimizer/full-sweep generators now separate restart-ladder and direct full-horizon commands, add output gates, and admit only completed true t=1500 replicated ensembles; the matched QL comparison is closed and non-promoted |
| External-VMEC high-grid holdout policy | 100% | CTH-like modified-protocol launch, horizon gates, `n80` seed/timestep long-window replicate gate, and explicit high-grid admission policy are reproducible; full `n48/n64/n80` remains non-claimable |
| Optimizer comparison campaign execution | 76% | Metadata/generators, strategy report, and solved-WOUT prelaunch metric gate are ready; actual multistart/continuation/SPSA-CMA-BO campaign remains planned unless promoted to a new run tranche |
| Production nonlinear turbulent-flux optimization evidence | 90% | Scoped selected-QA optimized-equilibrium audit is one positive long-window matched audit (`18.4%` reduction, `7.8 sigma`), but production promotion now requires three optimized-equilibrium ensembles and three matched audits; broad nonlinear turbulence-gradient and multi-equilibrium optimization claims remain open |

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
   optimized-equilibrium seed/timestep provenance plus at least three
   independent optimized-equilibrium ensembles and three matched
   baseline-to-optimized reduction audits before production promotion. New
   optimized candidates must reproduce that evidence structure before any
   broader claim.

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
  configs now write ``[output].nsave = [run].steps`` so GX-style artifact
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
