# Outputs & Diagnostics

Running a simulation produces structured outputs (NumPy/JAX arrays) plus **plots & animations**.

---

## 🔹 Raw backend output

- **Fourier mode solver**:
  ```python
  {
    "C_kSnt": (Nk, S, N, nt),  # Hermite-Fourier coefficients
    "Ek_kt": (Nk, nt),         # Electric field Fourier modes
    "k": (Nk,)                 # Wavenumber array
  }
```

* **DG solver**:

  ```python
  {
    "C_St": (S, N, Nx, nt),  # Hermite coefficients per cell
    "E_xt": (Nx, nt),        # Electric field in real space
    "x": (Nx,)               # Spatial grid
  }
  ```

---

## 🔹 Plots & diagnostics

The plotting suite automatically produces **publication-quality figures**:

1. **Energy panel**

   * Kinetic energy per species
   * Field energy
   * Total energy (check conservation)
   * Log scale for dynamic range

2. **Electric field**

   * Heatmap of $E(x,t)$
   * Shows wave growth, damping, nonlinear saturation

3. **Per-species diagnostics**

   * **Hermite spectrum**: phase-mixing, $\log |c_n(t)|$ vs time
   * **Distribution function**: animated $f(x,v,t)$ reconstruction

---

## 🔹 Example: Plot configuration

In the TOML file:

```toml
[plot]
nv = 257           # velocity grid for reconstruction
vmin_c = -0.3      # min velocity in units of c
vmax_c = 0.3       # max velocity in units of c
fig_width = 10.0   # inches
fig_row_height = 2.2
fps = 30           # animation frames per second
dpi = 150
# save_anim = "out.mp4"  # enable to save video
```

---

## 🔹 How to interpret results

* **Energy panel**

  * Landau damping: field energy decays, kinetic increases slightly.
  * Two-stream instability: field energy grows exponentially, then saturates.

* **Hermite spectrum**

  * Diagonal ridges = phase mixing.
  * Broadening = nonlinear cascade in velocity space.

* **Distribution function**

  * Flattening of slopes = quasilinear diffusion.
  * Phase-space holes/clumps = BGK modes.

---

## 📝 Tips

* Always compare total energy: drift should be <1% for well-resolved runs.
* Use **`vmin_c` / `vmax_c`** to zoom in on relevant phase-space region.
* For movies, keep `fps=30` and `dpi=150` for smooth, high-quality output.
* Outputs are NumPy arrays → you can load them directly for further analysis.
