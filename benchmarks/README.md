# Benchmarks

This directory contains the lightweight benchmark drivers and runtime TOML files
used to reproduce the tracked validation panels in the documentation.

The drivers intentionally keep generated outputs out of git:

- small scripts and TOML inputs live here,
- generated plots, NetCDF files, restart files, and logs should be written to
  `tools_out/` or an explicit scratch directory,
- publication-facing figures promoted to the docs are curated under
  `docs/_static/` after review and compression.

Run from the repository root, for example:

```bash
python benchmarks/cyclone_linear_benchmark.py --outdir tools_out/cyclone_benchmark
python benchmarks/kbm_beta_scan.py
python -m spectraxgk.cli run-runtime-linear --config benchmarks/runtime_secondary_slab.toml
python benchmarks/secondary_slab_workflow.py
```

The full atlas is built from tracked CSV/JSON assets rather than large transient
simulation directories:

```bash
python tools/make_benchmark_atlas.py
python tools/run_benchmark_refresh.py --list
```
