# Benchmark Recovery And Diagnostics Plan

## Scope

This plan is for the next parity-recovery pass between SPECTRAX-GK and GX.

Primary goals:

1. Re-run the benchmark cases that matter for the public atlas and runtime/performance story.
2. Confirm that no new mismatches were introduced in the recent standalone/public transition.
3. Localize and fix remaining mismatches with enough diagnostics that we stop re-running blind.
4. Decide which lanes belong in the public atlas and which should be demoted or removed.
5. Remove the reduced cETG lane from the public benchmark story, and likely from the source tree, unless a later explicit product decision reverses that.

Working acceptance target for this pass:

- linear: `rtol ~ 1e-1` is acceptable as a temporary screening threshold
- nonlinear: slightly larger tolerance is acceptable when traces oscillate around the same attractor or average level

This is a recovery-and-hardening pass, not a publication-quality final freeze.


## Fresh Sweep Status (2026-04-02)

The current live sweep on `main` confirms the following:

- Cyclone ITG linear remains acceptable but not closed.
  - tracked worst public row:
    - `ky=0.1`
    - `rel_gamma=+0.099`
    - `rel_omega=+0.039`
  - representative live point at `ky=0.3`:
    - `gamma≈0.0539`
    - `omega≈0.2996`
    - reference `gamma≈0.0930`, `omega≈0.2820`

- TEM benchmark contract was partially repaired and pushed, but TEM is still not acceptable.
  - restored tracked late-window contract now gives at `ky=0.3`:
    - `gamma≈3.700`
    - `omega≈1.503`
    - reference `gamma≈2.184`, `omega≈1.230`
  - current tracked worst row:
    - `ky=0.45`
    - `rel_gamma≈-4.25`
    - `rel_omega≈-52.8`
  - this is still a real physics/operator mismatch, not just a plotting or fitting issue.

- KAW generic runtime contract is still broken.
  - live runtime at requested `ky=0.01` collapses to:
    - `ky=0.0`
    - `gamma≈0`
    - `omega≈0`
  - this is still the zonal-collapse issue.

- ETG generic runtime contract is still broken.
  - live runtime at `ky=15` gives:
    - `gamma≈34.46`
    - `omega≈8.22`
  - this is still a branch-following failure.

- KBM public tracked benchmark lane is moderate, but the generic direct helper path is badly drifted.
  - tracked worst public row:
    - `ky=0.1`
    - `rel_gamma≈-0.113`
    - `rel_omega≈+0.011`
  - naive direct `run_kbm_linear()` discriminator at `ky=0.3` is catastrophically wrong:
    - `gamma≈191.8`
    - `omega≈-602.7`
    - reference `gamma≈0.314`, `omega≈1.076`
  - this means the benchmark-grade KBM path and the generic helper path have drifted apart.

- Miller linear remains an open low-`ky` lane.
  - tracked worst row:
    - `ky=0.05`
    - `rel_gamma≈+0.294`
    - `rel_omega≈-0.043`

- W7-X imported linear remains good.
  - tracked worst absolute row:
    - `ky=0.3`
    - `mean_abs_gamma≈1.87e-05`
    - `mean_abs_omega≈7.02e-06`
  - representative live runtime point at `ky=0.3` runs and returns finite values:
    - `gamma≈-0.0256`
    - `omega≈-0.1967`

- HSX imported linear remains the weaker stellarator lane.
  - tracked worst row:
    - `ky≈0.1905`
    - `mean_abs_gamma≈4.79e-03`
    - `mean_abs_omega≈3.75e-03`
    - `mean_rel_gamma≈1.14`

- Short nonlinear execution smokes currently succeed for:
  - Cyclone nonlinear
  - KBM nonlinear short
  - Miller nonlinear
  - W7-X nonlinear and HSX nonlinear are still being checked on local geometry inputs.

- TEM root-cause narrowing from live discriminators:
  - restoring the old late-time fit window materially improves TEM
  - changing `imex2` to `rk4` does not materially change the restored TEM point
  - the remaining mismatch is therefore not an integrator-family issue
  - parity-era `build_linear_params()` did not have the current electromagnetic sub-scaling hooks
  - current EM scaling variants move TEM, but no simple global setting recovers the reference branch

Current repair order after the fresh sweep:

1. TEM operator/parameter contract
2. KAW runtime contract
3. ETG runtime branch-following
4. KBM generic-helper / benchmark-contract drift
5. Miller low-`ky`
6. HSX imported-linear interpretation and, if needed, late-time audit


## Current Public Atlas State

Tracked linear comparison sources:

- `docs/_static/cyclone_mismatch_table.csv`
- `docs/_static/etg_mismatch_table.csv`
- `docs/_static/kbm_mismatch_table.csv`
- `docs/_static/kinetic_mismatch_table.csv`
- `docs/_static/tem_mismatch_table.csv`
- `docs/_static/cyclone_miller_linear_mismatch.csv`
- `docs/_static/w7x_linear_t2_scan.csv`
- `docs/_static/hsx_linear_t2_scan.csv`
- `docs/_static/kaw_exact_growth_dump.csv`
- `docs/_static/kbm_miller_exact_growth_dump.csv`

Tracked nonlinear atlas inputs:

- `docs/_static/nonlinear_cyclone_diag_compare.png`
- `docs/_static/nonlinear_kbm_diag_compare_t400_stats.png`
- `docs/_static/nonlinear_w7x_diag_compare_t200.png`
- `docs/_static/hsx_nonlinear_compare_t50_true.png`
- `docs/_static/nonlinear_cyclone_miller_diag_compare_t122.png`
- `docs/_static/cetg_gx_compare.csv`

Current known linear status from tracked assets:

- Cyclone ITG linear: acceptable for now, low-`k_y` still open
- ETG linear: acceptable for now
- KBM linear: still open
- W7-X imported linear: good in absolute error, relative growth inflated by near-marginality
- HSX imported linear: still open, especially in relative growth/field-energy interpretation
- Miller linear: still open
- kinetic-electron Cyclone linear: clearly wrong
- TEM linear: clearly wrong
- KAW exact linear diagnostic: good
- KBM Miller exact linear replay: good


## Benchmark Matrix For This Pass

### Keep And Re-run

These are the lanes to execute again in both SPECTRAX-GK and GX, because they belong in the product story and/or the current public atlas.

#### ITG

- Cyclone linear
- Cyclone nonlinear
- Miller linear
- Miller nonlinear

#### ETG

- ETG linear
- ETG nonlinear

Notes:

- ETG nonlinear is new as a practical benchmark target here. It is not currently a clean public atlas lane, so before the long run we must do a short-horizon runtime estimate on both codes.

#### KAW

- KAW linear
- KAW nonlinear

Notes:

- KAW nonlinear is not currently in the atlas.
- First run a short nonlinear pilot on both codes to estimate runtime and diagnostic usefulness.

#### KBM

- KBM linear
- KBM nonlinear

Notes:

- For nonlinear KBM, first do a short finite-electron-mass pilot in both codes.
- Only commit to the long run after checking wall time, output stability, and whether the chosen diagnostic set is actually informative.

#### HSX

- HSX linear
- HSX nonlinear

#### W7-X

- W7-X linear
- W7-X nonlinear


### Drop From Public Atlas

#### TEM

Drop TEM from the public atlas for now.

Reason:

- Current tracked TEM linear mismatch is too large:
  - `max |rel_gamma| ~ 1.06`
  - `max |rel_omega| ~ 2.4e2`
- This lane is still useful internally as a stress/debug lane.
- It should remain as an internal diagnostic target until parity is restored.


### Remove From Product Story

#### cETG

Plan to remove cETG from the public atlas and likely from the source tree.

Reason:

- In this repository, cETG is a reduced collisional-slab ETG model, not the main full gyrokinetic ETG path.
- It currently occupies the nonlinear ETG slot in the atlas, which overstates what it validates.
- It brings legacy-reduced-model compatibility baggage that is not central to the standalone product story.

Why ETG still matters:

- ETG turbulence is physically important for electron heat transport, especially in pedestal and electron-scale turbulence studies.
- References:
  - `Complex structure of turbulence across the ASDEX Upgrade pedestal`:
    https://arxiv.org/abs/2303.10596
  - `Three-Dimensional Inhomogeneity of Electron-Temperature-Gradient Turbulence in the Edge of Tokamak Plasmas`:
    https://arxiv.org/abs/2203.00831

Conclusion:

- Keep full ETG benchmarking.
- Remove reduced cETG from the public benchmark narrative.
- Replace it with real ETG linear and, if feasible, nonlinear ETG.


## Case-By-Case Execution Plan

### 1. Cyclone ITG

#### Linear

Run:

- SPECTRAX-GK current public benchmark path
- GX reference scan

Expectations:

- low-`k_y` may remain imperfect for now
- keep the lane if the mismatch remains localized and documented

Artifacts:

- refresh `docs/_static/cyclone_mismatch_table.csv`
- refresh `docs/_static/cyclone_gx_mismatch.csv` if needed

#### Nonlinear

Run:

- SPECTRAX-GK nonlinear runtime
- GX nonlinear benchmark

Diagnostics to compare:

- `Wg`
- `Wphi`
- `Phi2`
- heat flux
- particle flux


### 2. ETG

#### Linear

Run:

- SPECTRAX-GK benchmark path
- GX benchmark path

Known issue:

- generic runtime ETG config still has branch-following outliers at some `k_y`
- the tracked benchmark lane is better than the generic runtime path

Action:

- benchmark using the tracked benchmark contract, not the generic runtime utility

#### Nonlinear

Run short pilot first:

- SPECTRAX-GK ETG nonlinear
- GX ETG nonlinear

Pilot outputs:

- wall time
- peak memory
- first stable diagnostics
- whether growth/transport settles or remains unusably transient over the pilot window

Only then decide on the long run.


### 3. KAW

#### Linear

Run:

- SPECTRAX-GK linear KAW exact-window path
- GX exact-growth dump / exact-window reference

Known issue:

- runtime KAW example is not benchmark-aligned
- tracked exact diagnostic is good

Action:

- use the exact KAW lane as the truth path
- fix the runtime example only after the benchmark contract is revalidated

#### Nonlinear

Run short pilot first:

- finite-electron-mass KAW on both codes

Collect:

- runtime estimate
- whether diagnostics are meaningful enough for extrapolation
- whether the case is stable enough to merit atlas inclusion


### 4. KBM

#### Linear

Run:

- SPECTRAX-GK linear KBM public scan
- GX linear KBM scan

Known issue:

- startup parity is already good
- remaining mismatch is later-evolution / branch-following

Action:

- use late-state / late-window diagnostics
- if needed, produce matched late dump on GX and compare against SPECTRAX at the same window

#### Nonlinear

Run short pilot first:

- finite-electron-mass KBM nonlinear on both codes

Collect:

- wall time
- whether diagnostics settle
- whether selected `k_y` and mode set are informative

Then decide on the long run.


### 5. HSX

#### Linear

Run:

- SPECTRAX-GK imported-linear HSX scan
- GX imported-linear HSX scan

Known issue:

- absolute errors are much more meaningful than relative growth because some points are near marginal

Action:

- report both absolute and relative errors
- avoid declaring failure from relative-only growth in near-zero cases

#### Nonlinear

Run:

- SPECTRAX-GK HSX nonlinear
- GX HSX nonlinear

Compare:

- `Wg`
- `Wphi`
- `Phi2`
- heat flux
- particle flux


### 6. W7-X

#### Linear

Run:

- SPECTRAX-GK imported-linear W7-X scan
- GX imported-linear W7-X scan

Status:

- current absolute agreement is already good

Action:

- rerun mainly as a regression guard against newly introduced mismatches

#### Nonlinear

Run:

- SPECTRAX-GK W7-X nonlinear
- GX W7-X nonlinear

Compare:

- `Wg`
- `Wphi`
- `Phi2`
- heat flux
- particle flux


### 7. Miller

#### Linear

Run:

- SPECTRAX-GK linear Miller
- GX linear Miller

Known issue:

- current linear Miller mismatch is still large enough to keep off the “closed” list

#### Nonlinear

Run:

- SPECTRAX-GK nonlinear Miller
- GX nonlinear Miller

Compare:

- `Wg`
- `Wphi`
- `Phi2`
- heat flux
- particle flux


## What To Keep In The Public Atlas During Recovery

### Keep

- Cyclone ITG linear/nonlinear
- ETG linear
- KBM linear/nonlinear
- W7-X linear/nonlinear
- HSX linear/nonlinear
- KAW exact linear
- Miller nonlinear

### Keep With Explicit Caveat

- Cyclone ITG linear low-`k_y`
- W7-X and HSX imported-linear relative growth metrics near marginality

### Demote Or Remove For Now

- TEM linear
- kinetic-electron Cyclone linear
- cETG nonlinear
- linear Miller if it still blocks a clean public story after rerun


## Atlas / Asset Regeneration Policy

Do not regenerate all atlas images after every code change.

Regenerate only when the corresponding source data has been refreshed honestly.

### Assets That Will Need Regeneration Once Data Changes

- `docs/_static/benchmark_core_linear_atlas.png`
- `docs/_static/benchmark_core_linear_atlas.pdf`
- `docs/_static/benchmark_extended_linear_panel.png`
- `docs/_static/benchmark_extended_linear_panel.pdf`
- `docs/_static/benchmark_imported_linear_panel.png`
- `docs/_static/benchmark_imported_linear_panel.pdf`
- `docs/_static/benchmark_core_nonlinear_atlas.png`
- `docs/_static/benchmark_core_nonlinear_atlas.pdf`
- `docs/_static/benchmark_readme_panel.png`
- `docs/_static/benchmark_readme_panel.pdf`

### Immediate Atlas Decision

If cETG is removed from the public story, then:

- regenerate `benchmark_core_nonlinear_atlas.*`
- regenerate `benchmark_readme_panel.*`
- update `tools/benchmark_atlas_manifest.toml`
- update `docs/benchmarks.rst`
- update `README.md`


## Diagnostics And Dumping Plan

The current main productivity problem is lack of enough matched observables per run.

We should add richer, low-overhead dumping in both codes so each benchmark run can answer:

1. Did startup already differ?
2. Did the first field solve differ?
3. Did the first few timesteps differ?
4. Did the late-time branch differ?
5. Did diagnostics differ because of dynamics or because of extraction/post-processing?


### SPECTRAX-GK: Additions

Source locations to extend:

- `src/spectraxgk/runtime.py`
- `src/spectraxgk/benchmarks.py`
- `src/spectraxgk/analysis.py`
- `src/spectraxgk/linear.py`
- `src/spectraxgk/nonlinear.py`
- `src/spectraxgk/diagnostics.py`
- `src/spectraxgk/gx_integrators.py`

Existing useful comparison tooling:

- `tools/compare_gx_startup.py`
- `tools/compare_gx_runtime_diag_state.py`
- `tools/compare_gx_runtime_window.py`
- `tools/compare_gx_rhs_terms.py`
- `tools/compare_gx_imported_linear.py`

New dumps to add:

- startup state dump:
  - `G`
  - `phi`
  - `apar`
  - `bpar`
  - parameter summary
  - geometry summary
- first-step dump:
  - same fields after one RHS evaluation / one full step
- late-window diagnostic dump:
  - time array
  - selected mode signal
  - extracted `gamma`
  - extracted `omega`
  - fit window bounds
- nonlinear diagnostic dump:
  - `Wg_t`
  - `Wphi_t`
  - `Phi2_t`
  - heat flux
  - particle flux
  - optional per-species flux components
- operator term dump:
  - streaming
  - mirror
  - curvature
  - grad-B
  - diamagnetic
  - collisions
  - hypercollisions / hyper terms
  - nonlinear contribution

Output format:

- small structured JSON metadata
- NumPy `.npy` / `.npz` for arrays
- no verbose text parsing as the primary diagnostic path


### GX: Additions

Relevant source locations already identified:

- `src/diagnostic_classes.cu`
- `include/diagnostics.h`
- `include/ncdf.h`
- `include/parameters.h`
- `src/geometry.cu`
- `include/moments.h`

Useful existing capabilities:

- `GX_DUMP_DIAG_INDEX`-driven late-state dump path in `src/diagnostic_classes.cu`
- restart write/read
- rich NetCDF diagnostics:
  - `Phi2`
  - `Wg`
  - `Wphi`
  - heat flux
  - particle flux
  - fields
  - moments

Requested new GX diagnostics work:

- add explicit startup dump trigger
- add first-step / first-write dump trigger
- add richer metadata dump:
  - selected `ky`
  - selected `kx`
  - geometry factors
  - normalization factors
  - collision settings
  - hyper settings
- add per-term RHS dump if feasible for a single selected mode
- add lightweight text summary matching the structured dump

Key GX control knobs already present:

- `nwrite`
- `omega`
- restart write/read controls
- NetCDF diagnostics toggles
- `GX_DUMP_DIAG_INDEX`


## Execution Order

### Phase 1: Short-Horizon Runtime Estimation

Do first for expensive nonlinear lanes:

- nonlinear KAW
- nonlinear KBM
- nonlinear ETG

On both codes:

- run a short pilot
- record wall time, memory, and diagnostic quality
- extrapolate to target horizon


### Phase 2: Linear Regression Guard

Re-run linear lanes in this order:

1. Cyclone ITG
2. ETG
3. KAW
4. KBM
5. W7-X
6. HSX
7. Miller

Internal-only linear debug lanes:

8. kinetic-electron Cyclone
9. TEM


### Phase 3: Nonlinear Public Lanes

Re-run nonlinear public lanes:

1. Cyclone ITG
2. KBM
3. HSX
4. W7-X
5. Miller


### Phase 4: New/Expanded Nonlinear Lanes

Only after pilot timing estimates:

1. nonlinear KAW
2. nonlinear ETG


### Phase 5: Atlas/Docs Update

Only after the rerun data is real:

- refresh CSVs / NetCDF-derived summaries
- regenerate only the affected atlas images
- update README/docs claims
- remove cETG from atlas/docs if the product decision is confirmed


## Immediate Next Actions

1. Freeze cETG removal as a product decision.
2. Add structured startup / first-step / late-window dumps in SPECTRAX-GK and GX.
3. Run short nonlinear pilot jobs for:
   - KAW
   - KBM
   - ETG
4. Re-run the linear public lanes:
   - Cyclone ITG
   - ETG
   - KAW
   - KBM
   - HSX
   - W7-X
   - Miller
5. Keep TEM and kinetic-electron Cyclone as internal debug lanes until parity is restored.
6. After the fresh data exists, decide which atlas tiles stay public and regenerate only those panels.
