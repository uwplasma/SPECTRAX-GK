import numpy as np
import jax.numpy as jnp

from spectraxgk.config import GridConfig
from spectraxgk.grids import build_spectral_grid
from spectraxgk.terms.nonlinear import exb_nonlinear_contribution


def _ifft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(x, axes=(-3, -2))


def test_exb_bracket_zero_mean_mode():
    grid = build_spectral_grid(GridConfig(Nx=8, Ny=8, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(0)
    G = rng.normal(size=(1, 2, 3, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(1, 2, 3, grid.ky.size, grid.kx.size, grid.z.size)
    )
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    dG = exb_nonlinear_contribution(
        jnp.asarray(G),
        phi=jnp.asarray(phi),
        dealias_mask=grid.dealias_mask,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
    )
    mean_mode = np.asarray(dG[..., 0, 0, :])
    assert np.max(np.abs(mean_mode)) < 1.0e-6


def test_exb_bracket_energy_conserves():
    grid = build_spectral_grid(GridConfig(Nx=8, Ny=8, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(1)
    G = rng.normal(size=(2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    )
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    G_j = jnp.asarray(G)
    dG = exb_nonlinear_contribution(
        G_j,
        phi=jnp.asarray(phi),
        dealias_mask=grid.dealias_mask,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
    )
    G_real = _ifft2_xy(G_j)
    dG_real = _ifft2_xy(dG)
    energy_rate = jnp.sum(jnp.real(jnp.conj(G_real) * dG_real))
    energy_norm = jnp.sum(jnp.abs(G_real) ** 2)
    tol = 1.0e-3 * jnp.where(energy_norm == 0.0, 1.0, energy_norm)
    assert float(jnp.abs(energy_rate)) < float(tol)
