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

- 2026-06-09: Added `tools/build_qa_optimizer_strategy_report.py` plus focused
  tests and regenerated
  `docs/_static/vmec_jax_qa_optimizer_strategy_report.{png,json,csv}`. The
  report combines the strict QA optimizer panel with the admitted `RBC(1,1)`
  long-window landscape. It shows a real lower-Q direction (`+40% RBC(1,1)`,
  about 35% below the zero-offset late-window mean), but keeps nonlinear
  optimization promotion blocked because current transport optimizer rows are
  still diagnostic-only and true matched `t=1500` audits are pending. The
  recommended campaign ladder is now explicit: exact-adjoint least squares for
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
  `spectral_envelope_ridge` as the only full-portfolio screening and mean-error
  pass, but `accepted_holdout_screening_models` remains empty (`holdout`
  Spearman about `0.71`, pairwise order about `0.73`). This strengthens the
  claim boundary: useful screening/model-development evidence is claimable now,
  while universal absolute-flux or held-out screening promotion still requires
  additional independent, replicated, post-transient nonlinear holdouts.

- Extended the quasilinear holdout-gap report to ingest the screening-skill
  sidecar. The refreshed `docs/_static/quasilinear_holdout_gap_report.*` now
  carries both `absolute_flux_promotion_requirements` and
  `screening_promotion_requirements`: full-portfolio screening passes for
  `spectral_envelope_ridge`, held-out-only screening remains below gate, and
  the same next evidence item is required for both claim classes: additional
  independent, replicated, post-transient nonlinear holdouts.

# SPECTRAX-GK Active Plan and Running Log

Last updated: 2026-06-09
Active repository: `uwplasma/SPECTRAX-GK`
Current public baseline: `main`; see `pyproject.toml` for the active release
version and GitHub Actions for the latest CI result.
Historical planning archive: private repo `rogeriojorge/spectraxgk_plan`

This file is the public active plan and concise running log. Keep it short,
dated, and tied to reproducible artifacts, tests, figures, and gates. Detailed
historical logs live outside the release repository so clones stay small.

## Current Release Status

- CI/CD: release-readiness, package build, docs build, and local focused
  release checks are green for the current working baseline. The GitHub Actions
  head run is the source of truth for the latest full shard matrix.
  - Wide package coverage gate remains required at `>=95%`.
- Repository-size policy: tracked payload must stay below 50 MB. This active
  plan replaces the old 531 KB historical log to restore edit headroom.
- Release posture: technically shippable after the current patch release lands;
  broad manuscript-level nonlinear turbulence-optimization claims are not
  promoted. The strict QA baseline and refreshed RBC(1,1) landscape are tracked
  as optimization/noise diagnostics; the old sparse branch audit is historical
  evidence only while the full 31-point true nonlinear landscape campaign runs.

## Active Lanes

| Lane | Status | Current gate |
| --- | ---: | --- |
| CI/CD, release infrastructure, package coverage | 100% | Green CI, 95% package-wide coverage |
| Rerun-WOUT admission and artifact policy | 100% | Explicit authoritative rerun-WOUT path implemented and tested |
| Strict QA candidate screening | 100% | Top-12 projected edge candidate passes rerun-WOUT gates and reduces the 18-point metric by 2.29% |
| Strict nonlinear transport and campaign-admission evidence | 98% | Strict top-12 matched audit fails promotion; full-sweep QA matched audit is harvested as historical negative evidence; generator mismatch is fixed, but true t=1500 matched outputs are pending |
| Boundary-coefficient landscape and optimizer-noise diagnosis | 99% | 31-point RBC(1,1) reduced linear/QL landscape is tracked; 23 true long-window nonlinear overlays are admitted; strategy report quantifies a 35% lower-Q direction while keeping absolute claims blocked |
| Docs/readme/release hygiene | 98% | Public wording now separates reduced linear/QL landscape metrics from true nonlinear heat-flux evidence |
| Performance/parallelization release lane | 96% | Independent-work parallel paths are release-ready; nonlinear sharding profiler provenance is versioned and checker-gated, while whole-state/domain speedup remains diagnostic |
| QA optimization optimizer-comparison metadata | 98% | Public examples emit strict nonlinear audit manifests; optimizer/full-sweep generators now separate restart-ladder and direct full-horizon commands, add output gates, and keep matched full-sweep QA audits non-admitted until the running true t=1500 office campaign finishes |

Deferred post-release/manuscript extensions unless explicitly reprioritized:
W7-X zonal long-window recurrence/damping, W7-X TEM/multi-flux-tube extension,
and promotion of nonlinear domain decomposition beyond diagnostic evidence.

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
   optimized-equilibrium seed/timestep provenance and a matched
   baseline-to-optimized reduction audit before promotion. New optimized
   candidates must reproduce that evidence structure before any broader claim.

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
