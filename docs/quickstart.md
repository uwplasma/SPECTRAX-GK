# Quickstart

Let’s run your **first simulation** in just a few lines.

---

## 🔹 CLI usage (recommended)

Run the built-in two-stream instability example:

```bash
spectraxgk --input examples/two_stream.toml
```

You’ll see:

* Console output with:

  * Plasma frequency, Debye length normalizations
  * Box size, time step, species info

* A figure with:

  * **Energy traces** (per-species kinetic + field + total)
  * **Electric field heatmap** $E(x,t)$
  * **Per-species diagnostics** (Hermite spectra, animated distribution functions)

---

## 🔹 Minimal Python API

You can also drive SPECTRAX-GK as a library:

```python
from spectraxgk.io_config import read_toml
from spectraxgk.backends import run_fourier, run_dg
from spectraxgk.plots import render_suite_onefigure

# Load config
cfg = read_toml("examples/two_stream.toml")

# Run simulation
ts, out = run_dg(cfg)  # or run_fourier(cfg)

# Plot results
render_suite_onefigure(cfg, ts, out)
```

---

## 🔹 Tweak and explore

Open the config file:

```toml
[sim]
mode = "dg"      # switch to "fourier" for Hermite–Fourier
tmax = 20.0      # run longer (20 / ω_p)
nt = 400         # more time steps
```

Re-run:

```bash
spectraxgk --input examples/two_stream.toml
```

---

## 🔹 Next steps

* Try other configs in `examples/` (Landau damping, bump-on-tail).
* Explore inputs: [Inputs Guide](inputs.md).
* Learn physics: [Physics](physics.md).
