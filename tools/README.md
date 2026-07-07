# Tools

This directory is for repository-maintenance entry points, not solver library
code and not end-user examples. The refactor target is fewer than 100 Python
scripts in `tools/`, organized by purpose.

## Ownership Rules

Keep a script in `tools/` only when it has a current owner and one of these
roles:

- `release`: bounded CI/release gates and repository-policy checks.
- `artifacts`: builders for reviewed README/docs figures, tables, and JSON/CSV
  summaries.
- `profiling`: CPU/GPU runtime, memory, and hot-path profiling reproducers.
- `comparison`: explicit external-code comparison and parity utilities.
- `campaigns`: documented long-run launch or postprocess helpers that are still
  part of an active validation or optimization lane.

Move or delete scripts that are only local probes, historical audits, blocked
campaign launchers, or one-off debugging helpers. If a removed script may be
useful later, keep it outside `main` in a draft PR or experiment branch rather
than shipping it as part of the maintained repository.

## What Does Not Belong Here

- Solver, geometry, diagnostic, objective, or artifact-library functionality.
  Put that in `src/spectraxgk` only if it is promoted and reusable.
- User tutorials. Put those in `examples/`.
- Lightweight benchmark entry points and manifests. Put those in `benchmarks/`.
- Test-only helpers. Put those under `tests/` fixtures or test packages.
- Raw generated outputs, NetCDF files, logs, scratch directories, or profiler
  traces. Keep those ignored or attach them to releases when needed.

## Inventory

Use the maintained inventory tool before large moves or deletions:

```bash
python tools/release/inventory_repository.py \
  --json-out tools_out/repository_inventory.json \
  --summary-json-out tools_out/repository_inventory_summary.json
```

The inventory classifies tracked files by role and recommended action. It is a
planning aid, not a substitute for checking imports, docs references, tests, and
release manifests before deleting or moving files.

## Refactor Gate

`tools/package_architecture_manifest.toml` tracks the current Python-file
baseline and the final target. The default architecture check fails if the tool
count regresses upward. The final consolidation release should additionally run:

```bash
python tools/release/check_package_architecture_manifest.py --require-topology-targets
```

That strict mode fails until `tools/` contains fewer than 100 Python scripts and
flat top-level tool scripts have been moved into purpose-specific folders.
