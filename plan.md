# SPECTRAX-GK Ship Readiness Plan

Last updated: 2026-04-07
Current public head: `430e301 linear: fix sampled progress diagnostics`

## Current Ship Status

`main` is ready for shipment from a software-quality standpoint after the sampled-progress linear diagnostics fix.

Validated gates on the current recovery pass:

- `python3 -m mypy src`: passed
- `python3 -m py_compile` on touched files: passed
- Targeted sampled-progress regressions: passed
- Runtime / CLI / nonlinear CI shard: `154 passed`
- Focused linear/runtime/CLI regression slice: passed
- Full pytest at pulled head before the narrow local fix: `613 passed, 3 skipped`
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

## Performance / Accuracy Check

Representative CPU examples compared against checkpoint `61aac05` show broadly unchanged runtime and identical printed outputs after the fix:

| Case | Runtime status | Accuracy status |
| --- | --- | --- |
| Cyclone linear | same order, no clear speedup | same printed `gamma`, `omega` |
| ETG linear runtime | same order | same known branch outlier |
| KAW linear runtime | same order | same known runtime-contract mismatch |
| Cyclone nonlinear short | same order | same printed `Wg`, `Wphi`, heat |

Conclusion: this update did not make the code materially faster, but it also did not introduce new benchmark-asset accuracy drift after the sampled-progress fix.

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

## Next Work Order

1. Keep `main` shipped at the current fixed head unless CI reports a new failure.
2. Treat TEM and kinetic-electron Cyclone as the next parity-recovery lanes.
3. Then repair KAW runtime contract and ETG runtime branch-following.
4. Rebuild benchmark assets only after the corresponding numerical lane is honestly improved.
5. Consider making `ruff` a future CI gate only after a dedicated lint cleanup; current repo-wide `ruff check .` still reports pre-existing style debt.
