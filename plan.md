# SPECTRAX-GK Ship Readiness Plan

Last updated: 2026-04-09
Current public baseline under review: `092aeb9 benchmarks: close Miller low-ky linear lane`

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

- `W7-X nonlinear`
  - Still the next active open GX-backed lane.
  - Best current validated long-window trace on `office` is the VMEC-fixed
    branch (`w7x_spectrax_t200_vmecfix.csv`, tied with `chunked2`):
    - `mean_rel_abs(Wg) ~= 4.70e-1`
    - `mean_rel_abs(Wphi) ~= 5.77e-1`
    - `mean_rel_abs(HeatFlux) ~= 3.39e-1`
    - `final_rel(HeatFlux) ~= 8.15e-3`
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

- `HSX nonlinear`
  - Acceptable for this pass on the best validated `t <= 50` trace.
  - Current best validated metrics:
    - `mean_rel_abs(Wg) ~= 7.77e-2`
    - `mean_rel_abs(Wphi) ~= 9.56e-2`
    - `mean_rel_abs(HeatFlux) ~= 7.97e-2`
    - `final_rel(HeatFlux) ~= 3.05e-2`
  - Interpretation:
    - keep this lane in the public benchmark set
    - do not spend more solver/debug time here before the remaining nonlinear
      lanes are checked
  - The programmatic VMEC wrapper had likewise drifted from the published TOML
    collision contract; it is now aligned and covered by tests.

## Next Work Order

1. Treat Cyclone Miller linear, HSX linear, and KBM linear as effectively closed for this pass unless refreshed data regresses.
2. Treat Cyclone nonlinear and HSX nonlinear as acceptable on the current tracked comparisons.
3. Keep W7-X nonlinear as the next active open nonlinear lane.
4. Then continue in order: ETG nonlinear, KBM nonlinear, kinetic-electron Cyclone.
5. Leave KAW and TEM out of the active parity-recovery path until the above GX-backed lanes are honestly closed.
6. Consider making `ruff` a future CI gate only after a dedicated lint cleanup; current repo-wide `ruff check .` still reports pre-existing style debt.
