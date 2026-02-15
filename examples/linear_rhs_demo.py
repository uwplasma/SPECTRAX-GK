import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, linear_rhs


def main():
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    Nl, Nm = 2, 4
    G = jnp.zeros((Nl, Nm, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    G = G.at[0, 0, 0, 1, :].set(1e-3)

    dG, phi = linear_rhs(G, grid, geom, params)
    print("linear_rhs demo")
    print("dG norm:", jnp.linalg.norm(dG))
    print("phi min/max:", phi.min(), phi.max())


if __name__ == "__main__":
    main()
