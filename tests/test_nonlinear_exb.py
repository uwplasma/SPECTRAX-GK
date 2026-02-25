import numpy as np
import jax.numpy as jnp

from spectraxgk.config import GridConfig
from spectraxgk.grids import build_spectral_grid
from spectraxgk.terms.nonlinear import exb_nonlinear_contribution


def _ifft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(x, axes=(-3, -2))


def _finite_diff_periodic(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * dx)


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


def test_exb_bracket_respects_dealias_mask():
    grid = build_spectral_grid(GridConfig(Nx=8, Ny=8, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(2)
    G = rng.normal(size=(2, 2, 2, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(2, 2, 2, grid.ky.size, grid.kx.size, grid.z.size)
    )
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    dG_zero = exb_nonlinear_contribution(
        jnp.asarray(G),
        phi=jnp.asarray(phi),
        dealias_mask=jnp.zeros_like(grid.dealias_mask),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
    )
    assert np.allclose(np.asarray(dG_zero), 0.0)


def test_exb_bracket_matches_finite_difference():
    grid = build_spectral_grid(GridConfig(Nx=8, Ny=8, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    phi_hat = np.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex128)
    G_hat = np.zeros_like(phi_hat)
    phi_hat[1, 2, 0] = 1.0
    phi_hat[-1, -2, 0] = 1.0
    G_hat[2, 1, 0] = 0.5
    G_hat[-2, -1, 0] = 0.5
    phi = np.fft.ifft2(phi_hat[:, :, 0]).real
    G = np.fft.ifft2(G_hat[:, :, 0]).real
    dx = 2.0 * np.pi / grid.kx.size
    dy = 2.0 * np.pi / grid.ky.size
    dphi_dx = _finite_diff_periodic(phi, dx, axis=1)
    dphi_dy = _finite_diff_periodic(phi, dy, axis=0)
    dG_dx = _finite_diff_periodic(G, dx, axis=1)
    dG_dy = _finite_diff_periodic(G, dy, axis=0)
    bracket_fd = dphi_dx * dG_dy - dphi_dy * dG_dx
    bracket_spec = exb_nonlinear_contribution(
        jnp.asarray(G_hat[None, None, None, ...]),
        phi=jnp.asarray(phi_hat),
        dealias_mask=jnp.ones_like(grid.dealias_mask),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
    )
    bracket_spec_real = np.fft.ifft2(np.asarray(bracket_spec[0, 0, 0, :, :, 0])).real
    assert np.max(np.abs(bracket_spec_real - bracket_fd)) < 5.0e-3


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
