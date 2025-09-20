
# Quickstart

Run a two-stream instability example:

```bash
spectraxgk --input examples/two_stream.toml
````

You’ll see console summaries (normalizations, grid, species) and a figure with:

* Energy traces (per-species kinetic + field + total)
* $E(x,t)$ heatmap
* Per-species phase-mixing and animated $f(x,v,t)$

> Want to tweak? Open `examples/two_stream.toml`, then re-run the command.

## Minimal Python API

```python
from spectraxgk.io_config import read_toml
from spectraxgk.backends import run_fourier, run_dg
from spectraxgk.plots import render_suite_onefigure

cfg = read_toml("examples/two_stream.toml")
ts, out = run_dg(cfg)  # or run_fourier(cfg)
render_suite_onefigure(cfg, ts, out)
```
