
# Outputs & Diagnostics

The backend returns:
- **Fourier**: `{"C_kSnt": (Nk,S,N,nt), "Ek_kt": (Nk,nt), "k": (Nk,) }`
- **DG**: `{"C_St": (S,N,Nx,nt), "E_xt": (Nx,nt), "x": (Nx,) }`

Plots include:
- **Energy panel**: per-species kinetic, field, total (log scale)
- **Electric field**: \(E(x,t)\) heatmap
- **Per-species**:
  - Phase-mixing plot of \(\log |c_n(t)|\)
  - Animated \(f(x,v,t)\) (Hermite reconstruction)

Tuning (TOML):
```toml
[plot]
nv = 257
vmin_c = -0.3
vmax_c = 0.3
fig_width = 10.0
fig_row_height = 2.2
fps = 30
dpi = 150
# save_anim = "out.mp4"
````
