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
from spectraxgk.solver import run_simulation
from spectraxgk.post import load_result, plot_energy

cfg = read_toml("examples/linear_slab.toml")
info = run_simulation(cfg)
res = load_result(info["outfile"])
plot_energy(res)
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
