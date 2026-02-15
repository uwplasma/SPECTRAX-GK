import numpy as np
import jax.numpy as jnp

from spectraxgk.benchmarks import fit_growth_rate, load_cyclone_reference
from spectraxgk.config import CycloneBaseCase
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, linear_rhs


def integrate_linear(G0, grid, geom, params, dt, steps):
    G = G0
    phis = []
    for _ in range(steps):
        dG, phi = linear_rhs(G, grid, geom, params)
        G = G + dt * dG
        phis.append(phi)
    return jnp.stack(phis, axis=0)


def main():
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    ref = load_cyclone_reference()
    ky_target = 0.3
    ky_idx = int(np.argmin(np.abs(ref.ky - ky_target)))
    print(f"Reference ky={ref.ky[ky_idx]:.3f} gamma={ref.gamma[ky_idx]:.6f} omega={ref.omega[ky_idx]:.6f}")

    Nl, Nm = 2, 4
    G = jnp.zeros((Nl, Nm, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    # excite a single ky, kx, z mode
    ky_index = int(np.argmin(np.abs(grid.ky - ky_target)))
    kx_index = 0
    G = G.at[0, 0, ky_index, kx_index, :].set(1e-3 + 0.0j)

    dt = 0.05
    steps = 200
    phis = integrate_linear(G, grid, geom, params, dt, steps)
    t = np.arange(steps) * dt

    phi_signal = np.array(phis[:, ky_index, kx_index, 0])
    gamma, omega = fit_growth_rate(t, phi_signal, tmin=0.5 * t[-1])
    print(f"Extracted gamma={gamma:.6f} omega={omega:.6f}")
    print("Note: streaming-only model is not expected to match Cyclone growth rates yet.")


if __name__ == "__main__":
    main()
