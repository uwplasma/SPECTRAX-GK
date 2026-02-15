import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, integrate_linear


def main():
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    Nl, Nm = 2, 4
    G = jnp.zeros((Nl, Nm, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)
    G = G.at[0, 0, 0, 1, :].set(1e-3 + 0.0j)

    _, phi_t = integrate_linear(G, grid, geom, params, dt=0.05, steps=10)
    print("linear_rhs demo")
    print("phi_t shape:", phi_t.shape)
    print("phi_t min/max:", phi_t.min(), phi_t.max())


if __name__ == "__main__":
    main()
