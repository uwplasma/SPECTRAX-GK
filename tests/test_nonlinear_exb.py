import numpy as np
import jax.numpy as jnp

from spectraxgk.config import GridConfig
from spectraxgk.gyroaverage import bessel_j0, bessel_j1, gx_laguerre_transform
from spectraxgk.grids import build_spectral_grid, real_fft_mesh
from spectraxgk.terms.nonlinear import (
    _apply_flutter,
    exb_nonlinear_contribution,
    nonlinear_em_contribution,
    nonlinear_em_components,
    _apply_mask_xy,
    _broadcast_grid,
    _broadcast_mask,
    _broadcast_to_G,
    _gx_bpar_term,
    _gx_bpar_term_precomputed,
    _gx_j0_field,
    _gx_j0_field_precomputed,
    _laguerre_to_grid,
    _laguerre_to_spectral,
    _spectral_bracket,
    _spectral_bracket_multi,
    _stack_fields,
    placeholder_nonlinear_contribution,
)


def _ifft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(x, axes=(-3, -2))


def _finite_diff_periodic(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * dx)


def _expand_hermitian(pos: np.ndarray, ny_full: int) -> np.ndarray:
    """Expand rFFT-unique ky rows into full Hermitian spectrum."""
    nyc = ny_full // 2 + 1
    if nyc <= 2:
        return pos
    neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
    neg = np.conj(pos[..., 1:neg_hi, :, :])
    neg = neg[..., ::-1, :, :]
    nx = pos.shape[-2]
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
        neg = neg[..., kx_neg, :]
    return np.concatenate([pos, neg], axis=-3)


def test_exb_bracket_zero_mean_mode():
    grid = build_spectral_grid(GridConfig(Nx=8, Ny=8, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(0)
    G_real = rng.normal(size=(1, 2, 3, grid.ky.size, grid.kx.size, grid.z.size))
    phi_real = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size))
    G = np.fft.fft2(G_real, axes=(-3, -2))
    phi = np.fft.fft2(phi_real, axes=(0, 1))
    dG = exb_nonlinear_contribution(
        jnp.asarray(G),
        phi=jnp.asarray(phi),
        dealias_mask=grid.dealias_mask,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
    )
    mean_mode = np.asarray(dG[..., 0, 0, :])
    assert np.max(np.abs(mean_mode)) < 1.0e-3


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
        gx_real_fft=False,
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
    kx = np.asarray(grid.kx_grid)
    ky = np.asarray(grid.ky_grid)
    Nxy = grid.kx.size * grid.ky.size
    dphi_dx = np.fft.ifft2(1j * kx * phi_hat[:, :, 0]) * Nxy
    dphi_dy = np.fft.ifft2(1j * ky * phi_hat[:, :, 0]) * Nxy
    dG_dx = np.fft.ifft2(1j * kx * G_hat[:, :, 0]) * Nxy
    dG_dy = np.fft.ifft2(1j * ky * G_hat[:, :, 0]) * Nxy
    bracket = dG_dx * dphi_dy - dG_dy * dphi_dx
    bracket_ref = np.fft.fft2(bracket) / Nxy
    bracket_spec = exb_nonlinear_contribution(
        jnp.asarray(G_hat[None, None, None, ...]),
        phi=jnp.asarray(phi_hat),
        dealias_mask=jnp.ones_like(grid.dealias_mask),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
        gx_real_fft=False,
    )
    bracket_spec_hat = np.asarray(bracket_spec[0, 0, 0, :, :, 0])
    assert np.max(np.abs(bracket_spec_hat - bracket_ref)) < 5.0e-6


def test_apar_flutter_hermite_ladder():
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(3)
    G = rng.normal(size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size)
    )
    apar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    Jl = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    JlB = jnp.ones_like(Jl)
    vth = jnp.asarray([1.2])
    sqrt_m = jnp.sqrt(jnp.arange(3, dtype=jnp.float32))[None, :, None, None, None]
    sqrt_m_p1 = jnp.sqrt(jnp.arange(1, 4, dtype=jnp.float32))[None, :, None, None, None]

    apar_masked = _apply_mask_xy(jnp.asarray(apar), grid.dealias_mask)
    bracket_apar = _spectral_bracket(
        jnp.asarray(G),
        Jl * apar_masked[None, None, ...],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
    )
    flutter_expected = _apply_flutter(bracket_apar, vth, sqrt_m, sqrt_m_p1)

    dG = nonlinear_em_contribution(
        jnp.asarray(G),
        phi=jnp.zeros_like(jnp.asarray(apar)),
        apar=apar_masked,
        bpar=None,
        Jl=Jl,
        JlB=JlB,
        tz=jnp.asarray([1.0]),
        vth=vth,
        sqrt_m=sqrt_m,
        sqrt_m_p1=sqrt_m_p1,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=1.0,
    )
    assert np.max(np.abs(np.asarray(dG - flutter_expected))) < 1.0e-6


def test_apar_flutter_no_wrap_boundary():
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    Nm = 4
    bracket = jnp.zeros(
        (1, 1, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    bracket = bracket.at[:, :, -1, 0, 0, 0].set(1.0 + 0.0j)
    sqrt_m = jnp.sqrt(jnp.arange(Nm, dtype=jnp.float32))[None, :, None, None, None]
    sqrt_m_p1 = jnp.sqrt(jnp.arange(1, Nm + 1, dtype=jnp.float32))[None, :, None, None, None]
    flutter = _apply_flutter(bracket, jnp.asarray([1.0]), sqrt_m, sqrt_m_p1)
    flutter_np = np.asarray(flutter)
    assert np.allclose(flutter_np[:, :, 0, 0, 0, 0], 0.0)
    assert np.allclose(flutter_np[:, :, -1, 0, 0, 0], 0.0)
    expected = -np.sqrt(Nm - 1)
    assert np.allclose(flutter_np[:, :, -2, 0, 0, 0], expected)


def test_em_weights_are_toggles():
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(5)
    G = rng.normal(size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size)
    )
    apar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    bpar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    Jl = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    JlB = jnp.ones_like(Jl)
    common = dict(
        phi=jnp.zeros_like(jnp.asarray(apar)),
        apar=jnp.asarray(apar),
        bpar=jnp.asarray(bpar),
        Jl=Jl,
        JlB=JlB,
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(3, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, 4, dtype=jnp.float32))[None, :, None, None, None],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
    )
    dG_on = nonlinear_em_contribution(
        jnp.asarray(G), apar_weight=1.0, bpar_weight=1.0, weight=jnp.asarray(1.0), **common
    )
    dG_off = nonlinear_em_contribution(
        jnp.asarray(G), apar_weight=1.0, bpar_weight=1.0, weight=jnp.asarray(0.0), **common
    )
    assert np.max(np.abs(np.asarray(dG_off))) < 1.0e-8
    assert np.max(np.abs(np.asarray(dG_on))) > 0.0


def test_gx_real_fft_bracket_match():
    grid = build_spectral_grid(GridConfig(Nx=6, Ny=8, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    ny = grid.ky.size
    nx = grid.kx.size
    nyc = ny // 2 + 1
    rng = np.random.default_rng(42)
    G_nyc = rng.normal(size=(1, 1, 1, nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 1, nyc, nx, grid.z.size)
    )
    phi_nyc = rng.normal(size=(nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(nyc, nx, grid.z.size)
    )

    G_full = _expand_hermitian(G_nyc, ny)
    phi_full = _expand_hermitian(phi_nyc[None, ...], ny)[0]

    bracket_hat = _spectral_bracket(
        jnp.asarray(G_full),
        jnp.asarray(phi_full),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        gx_real_fft=True,
    )

    Nxy = nx * ny
    _, _, kx_nyc, ky_nyc = real_fft_mesh(grid.kx_grid, grid.ky_grid)
    kx_b = kx_nyc[None, None, None, :, :, None]
    ky_b = ky_nyc[None, None, None, :, :, None]
    kx_c = kx_nyc[None, None, :, :, None]
    ky_c = ky_nyc[None, None, :, :, None]

    dG_dx = np.fft.irfft2(1j * kx_b * G_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dG_dy = np.fft.irfft2(1j * ky_b * G_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dphi_dx = np.fft.irfft2(1j * kx_c * phi_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dphi_dy = np.fft.irfft2(1j * ky_c * phi_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    bracket = dG_dx * dphi_dy[:, :, None, ...] - dG_dy * dphi_dx[:, :, None, ...]
    bracket_hat_nyc = np.fft.rfft2(bracket, axes=(-2, -3)) / Nxy
    mask_nyc = np.asarray(grid.dealias_mask)[:nyc, :]
    bracket_hat_nyc = bracket_hat_nyc * mask_nyc[None, None, None, :, :, None]
    bracket_hat_ref = _expand_hermitian(bracket_hat_nyc, ny)

    assert np.allclose(
        np.asarray(bracket_hat),
        bracket_hat_ref,
        rtol=1.0e-4,
        atol=2.0e-5,
    )


def test_gx_real_fft_toggle_matches_full_fft_for_hermitian():
    grid = build_spectral_grid(GridConfig(Nx=6, Ny=8, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    ny = grid.ky.size
    nx = grid.kx.size
    nyc = ny // 2 + 1
    rng = np.random.default_rng(123)
    G_nyc = rng.normal(size=(1, 1, 1, nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 1, nyc, nx, grid.z.size)
    )
    chi_nyc = rng.normal(size=(nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(nyc, nx, grid.z.size)
    )

    G_full = _expand_hermitian(G_nyc, ny)
    chi_full = _expand_hermitian(chi_nyc[None, ...], ny)[0]

    bracket_gx = _spectral_bracket(
        jnp.asarray(G_full),
        jnp.asarray(chi_full),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        gx_real_fft=True,
    )
    bracket_full = _spectral_bracket(
        jnp.asarray(G_full),
        jnp.asarray(chi_full),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        gx_real_fft=False,
    )
    assert bracket_gx.shape == bracket_full.shape
    assert np.all(np.isfinite(np.asarray(bracket_gx)))
    assert np.all(np.isfinite(np.asarray(bracket_full)))


def test_gx_real_fft_bracket_preserves_full_hermitian_symmetry():
    grid = build_spectral_grid(GridConfig(Nx=6, Ny=8, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    ny = grid.ky.size
    nx = grid.kx.size
    nyc = ny // 2 + 1
    rng = np.random.default_rng(7)
    G_nyc = rng.normal(size=(1, 1, 2, nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 2, nyc, nx, grid.z.size)
    )
    phi_nyc = rng.normal(size=(nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(nyc, nx, grid.z.size)
    )
    G_full = _expand_hermitian(G_nyc, ny)
    phi_full = _expand_hermitian(phi_nyc[None, ...], ny)[0]

    bracket_hat = np.asarray(
        _spectral_bracket(
            jnp.asarray(G_full),
            jnp.asarray(phi_full),
            kx_grid=grid.kx_grid,
            ky_grid=grid.ky_grid,
            dealias_mask=grid.dealias_mask,
            kxfac=jnp.asarray(1.0),
            gx_real_fft=True,
        )
    )

    neg_hi = nyc - 1 if (ny % 2 == 0) else nyc
    kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
    neg_ref = np.conj(bracket_hat[..., 1:neg_hi, :, :])[..., ::-1, :, :]
    neg_ref = neg_ref[..., kx_neg, :]
    assert np.allclose(bracket_hat[..., nyc:, :, :], neg_ref, rtol=1.0e-6, atol=1.0e-6)


def test_gx_real_fft_bracket_match_odd_ny():
    grid = build_spectral_grid(GridConfig(Nx=6, Ny=7, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    ny = grid.ky.size
    nx = grid.kx.size
    nyc = ny // 2 + 1
    rng = np.random.default_rng(321)
    G_nyc = rng.normal(size=(1, 1, 1, nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 1, nyc, nx, grid.z.size)
    )
    phi_nyc = rng.normal(size=(nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(nyc, nx, grid.z.size)
    )
    G_full = _expand_hermitian(G_nyc, ny)
    phi_full = _expand_hermitian(phi_nyc[None, ...], ny)[0]

    bracket_hat = _spectral_bracket(
        jnp.asarray(G_full),
        jnp.asarray(phi_full),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        gx_real_fft=True,
    )

    Nxy = nx * ny
    _, _, kx_nyc, ky_nyc = real_fft_mesh(grid.kx_grid, grid.ky_grid)
    kx_b = kx_nyc[None, None, None, :, :, None]
    ky_b = ky_nyc[None, None, None, :, :, None]
    kx_c = kx_nyc[None, None, :, :, None]
    ky_c = ky_nyc[None, None, :, :, None]

    dG_dx = np.fft.irfft2(1j * kx_b * G_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dG_dy = np.fft.irfft2(1j * ky_b * G_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dphi_dx = np.fft.irfft2(1j * kx_c * phi_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dphi_dy = np.fft.irfft2(1j * ky_c * phi_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    bracket = dG_dx * dphi_dy[:, :, None, ...] - dG_dy * dphi_dx[:, :, None, ...]
    bracket_hat_nyc = np.fft.rfft2(bracket, axes=(-2, -3)) / Nxy
    mask_nyc = np.asarray(grid.dealias_mask)[:nyc, :]
    bracket_hat_nyc = bracket_hat_nyc * mask_nyc[None, None, None, :, :, None]
    bracket_hat_ref = _expand_hermitian(bracket_hat_nyc, ny)

    assert np.allclose(
        np.asarray(bracket_hat),
        bracket_hat_ref,
        rtol=1.0e-4,
        atol=2.0e-5,
    )


def test_bpar_contributes_to_chi():
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(4)
    G = rng.normal(size=(1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size)
    )
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    bpar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    Jl = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    JlB = 2.0 * jnp.ones_like(Jl)
    phi_masked = _apply_mask_xy(jnp.asarray(phi), grid.dealias_mask)
    bpar_masked = _apply_mask_xy(jnp.asarray(bpar), grid.dealias_mask)
    chi = Jl * phi_masked[None, None, ...] + JlB * bpar_masked[None, None, ...]
    bracket_expected = _spectral_bracket(
        jnp.asarray(G),
        chi,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
    )
    dG = nonlinear_em_contribution(
        jnp.asarray(G),
        phi=phi_masked,
        apar=None,
        bpar=bpar_masked,
        Jl=Jl,
        JlB=JlB,
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(2, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, 3, dtype=jnp.float32))[None, :, None, None, None],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=1.0,
    )
    assert np.max(np.abs(np.asarray(dG - bracket_expected))) < 1.0e-5


def test_bracket_multi_matches_separate():
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(11)
    G = rng.normal(size=(1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size)
    )
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    bpar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    Jl = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    JlB = 2.0 * jnp.ones_like(Jl)
    chi_phi = Jl * phi[None, None, ...]
    chi_bpar = JlB * bpar[None, None, ...]
    chi_stack = _stack_fields(jnp.asarray(G), [jnp.asarray(chi_phi), jnp.asarray(chi_bpar)])
    brackets = _spectral_bracket_multi(
        jnp.asarray(G),
        chi_stack,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        gx_real_fft=False,
    )
    bracket_phi = _spectral_bracket(
        jnp.asarray(G),
        jnp.asarray(chi_phi),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        gx_real_fft=False,
    )
    bracket_bpar = _spectral_bracket(
        jnp.asarray(G),
        jnp.asarray(chi_bpar),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        gx_real_fft=False,
    )
    assert np.allclose(np.asarray(brackets[0]), np.asarray(bracket_phi), rtol=1.0e-6, atol=1.0e-6)
    assert np.allclose(np.asarray(brackets[1]), np.asarray(bracket_bpar), rtol=1.0e-6, atol=1.0e-6)


def test_laguerre_precompute_matches_direct():
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(21)
    Ns, Nl, Nm = 1, 2, 2
    G = rng.normal(size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    )
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    apar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    bpar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    b = np.full((Ns, grid.ky.size, grid.kx.size, grid.z.size), 0.05, dtype=np.float64)
    lag_to_grid, lag_to_spec, lag_roots = gx_laguerre_transform(Nl)
    laguerre_to_grid = jnp.asarray(lag_to_grid, dtype=jnp.float64)
    laguerre_to_spectral = jnp.asarray(lag_to_spec, dtype=jnp.float64)
    laguerre_roots = jnp.asarray(lag_roots, dtype=jnp.float64)
    alpha = jnp.sqrt(
        jnp.maximum(
            0.0,
            2.0 * laguerre_roots[None, :, None, None, None] * jnp.asarray(b)[:, None, ...],
        )
    )
    j0 = bessel_j0(alpha)
    j1 = bessel_j1(alpha)
    j1_over_alpha = jnp.where(alpha < 1.0e-8, 0.5, j1 / alpha)

    common = dict(
        phi=jnp.asarray(phi),
        apar=jnp.asarray(apar),
        bpar=jnp.asarray(bpar),
        Jl=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        JlB=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(Nm, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, Nm + 1, dtype=jnp.float32))[None, :, None, None, None],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=1.0,
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        b=jnp.asarray(b),
        gx_real_fft=False,
    )
    dG_direct = nonlinear_em_contribution(jnp.asarray(G), **common)
    dG_cached = nonlinear_em_contribution(
        jnp.asarray(G),
        laguerre_j0=j0,
        laguerre_j1_over_alpha=j1_over_alpha,
        **common,
    )
    assert np.allclose(np.asarray(dG_direct), np.asarray(dG_cached), rtol=1.0e-6, atol=1.0e-6)


def test_laguerre_mode_spectral_matches_identity_grid():
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(12)
    Ns, Nl, Nm = 1, 1, 1
    G = rng.normal(size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    )
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    laguerre_to_grid = jnp.asarray([[1.0]], dtype=jnp.float64)
    laguerre_to_spectral = jnp.asarray([[1.0]], dtype=jnp.float64)
    laguerre_roots = jnp.asarray([0.0], dtype=jnp.float64)
    b = jnp.zeros((Ns, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float64)
    j0 = jnp.ones((Ns, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float64)
    j1_over_alpha = jnp.full_like(j0, 0.5)

    common = dict(
        phi=jnp.asarray(phi),
        apar=None,
        bpar=None,
        Jl=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        JlB=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.zeros((Ns, Nm, 1, 1, 1), dtype=jnp.float32),
        sqrt_m_p1=jnp.ones((Ns, Nm, 1, 1, 1), dtype=jnp.float32),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=0.0,
        bpar_weight=0.0,
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        laguerre_j0=j0,
        laguerre_j1_over_alpha=j1_over_alpha,
        b=b,
        gx_real_fft=False,
    )

    dG_grid = nonlinear_em_contribution(jnp.asarray(G), laguerre_mode="grid", **common)
    dG_spec = nonlinear_em_contribution(jnp.asarray(G), laguerre_mode="spectral", **common)
    assert np.allclose(np.asarray(dG_grid), np.asarray(dG_spec), rtol=1.0e-6, atol=1.0e-6)


def test_nonlinear_components_total_matches_contribution():
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(21)
    G = rng.normal(size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size)
    )
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    apar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    Jl = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    JlB = jnp.ones_like(Jl)
    common = dict(
        phi=jnp.asarray(phi),
        apar=jnp.asarray(apar),
        bpar=None,
        Jl=Jl,
        JlB=JlB,
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(3, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, 4, dtype=jnp.float32))[None, :, None, None, None],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=0.0,
        gx_real_fft=False,
    )
    dG = nonlinear_em_contribution(jnp.asarray(G), **common)
    comps = nonlinear_em_components(jnp.asarray(G), laguerre_mode="spectral", **common)
    assert np.allclose(np.asarray(dG), np.asarray(comps["total"]), rtol=1.0e-6, atol=1.0e-6)


def test_laguerre_components_total_matches_contribution():
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    rng = np.random.default_rng(22)
    Ns, Nl, Nm = 1, 1, 2
    G = rng.normal(size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    )
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    apar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    laguerre_to_grid = jnp.asarray([[1.0], [1.0]], dtype=jnp.float64).T
    laguerre_to_spectral = jnp.asarray([[1.0], [1.0]], dtype=jnp.float64)
    laguerre_roots = jnp.asarray([0.0], dtype=jnp.float64)
    b = jnp.zeros((Ns, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float64)
    j0 = jnp.ones((Ns, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float64)
    j1_over_alpha = jnp.full_like(j0, 0.5)
    common = dict(
        phi=jnp.asarray(phi),
        apar=jnp.asarray(apar),
        bpar=None,
        Jl=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        JlB=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(Nm, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, Nm + 1, dtype=jnp.float32))[None, :, None, None, None],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=0.0,
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        laguerre_j0=j0,
        laguerre_j1_over_alpha=j1_over_alpha,
        b=b,
        gx_real_fft=False,
    )
    dG = nonlinear_em_contribution(jnp.asarray(G), laguerre_mode="grid", **common)
    comps = nonlinear_em_components(jnp.asarray(G), laguerre_mode="grid", **common)
    assert np.allclose(np.asarray(dG), np.asarray(comps["total"]), rtol=1.0e-6, atol=1.0e-6)


def test_exb_bracket_energy_conserves():
    grid = build_spectral_grid(GridConfig(Nx=8, Ny=8, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi))
    G_hat = np.zeros((1, 1, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex128)
    phi_hat = np.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex128)
    G_hat[0, 0, 0, 1, 2, 0] = 1.0
    phi_hat[1, 2, 0] = 2.0
    dG = exb_nonlinear_contribution(
        jnp.asarray(G_hat),
        phi=jnp.asarray(phi_hat),
        dealias_mask=grid.dealias_mask,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
        gx_real_fft=False,
    )
    assert np.max(np.abs(np.asarray(dG))) < 1.0e-6


def test_nonlinear_broadcast_helpers_cover_shape_contracts():
    mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    grid = jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32)
    field = jnp.ones((2, 2, 3), dtype=jnp.complex64)
    G = jnp.zeros((1, 2, 3, 2, 2, 3), dtype=jnp.complex64)

    mask_b = _broadcast_mask(mask, 5)
    grid_b = _broadcast_grid(grid, 5)
    assert mask_b.shape == (1, 1, 2, 2, 1)
    assert grid_b.shape == (1, 1, 2, 2, 1)

    masked = _apply_mask_xy(field, mask)
    assert masked.shape == field.shape
    assert np.allclose(np.asarray(masked[0, 1, :]), 0.0)
    assert np.allclose(np.asarray(_apply_mask_xy(field, None)), np.asarray(field))

    same = _broadcast_to_G(G, G)
    assert same.shape == G.shape
    assert _broadcast_to_G(field, G).shape == (1, 1, 1, 2, 2, 3)
    assert _broadcast_to_G(jnp.ones((2, 3), dtype=jnp.float32), G).shape == (1, 1, 1, 1, 2, 3)


def test_laguerre_transform_helpers_roundtrip():
    rng = np.random.default_rng(1234)
    G = rng.normal(size=(1, 2, 3, 2, 2, 1)) + 1j * rng.normal(size=(1, 2, 3, 2, 2, 1))
    laguerre_to_grid = np.asarray([[1.0, 0.25], [0.0, 1.0]], dtype=np.float64)
    laguerre_to_spectral = np.linalg.inv(laguerre_to_grid)

    g_mu = _laguerre_to_grid(jnp.asarray(G), jnp.asarray(laguerre_to_grid))
    restored = _laguerre_to_spectral(g_mu, jnp.asarray(laguerre_to_spectral))
    assert g_mu.shape == (1, 2, 3, 2, 2, 1)
    assert np.allclose(np.asarray(restored), G, rtol=1.0e-6, atol=1.0e-6)


def test_gx_precomputed_bessel_helpers_match_direct():
    field = jnp.asarray([[[1.0 + 0.5j]]], dtype=jnp.complex64)
    b = jnp.asarray([[[0.05]]], dtype=jnp.float32)
    roots = jnp.asarray([0.0, 1.5], dtype=jnp.float32)
    tz = jnp.asarray([1.0], dtype=jnp.float32)

    direct_j0 = _gx_j0_field(field, b, roots, 1.0)
    b_species = b[None, ...]
    alpha = jnp.sqrt(
        jnp.maximum(0.0, 2.0 * roots[None, :, None, None, None] * b_species[:, None, ...])
    )
    from spectraxgk.gyroaverage import bessel_j0 as _b0, bessel_j1 as _b1

    j0 = _b0(alpha)
    j1 = _b1(alpha)
    j1_over_alpha = jnp.where(alpha < 1.0e-8, 0.5, j1 / alpha)
    cached_j0 = _gx_j0_field_precomputed(field, j0, 1.0)
    assert np.allclose(np.asarray(direct_j0), np.asarray(cached_j0), rtol=1.0e-6, atol=1.0e-6)

    bpar = jnp.asarray([[[0.3 + 0.1j]]], dtype=jnp.complex64)
    direct_bpar = _gx_bpar_term(bpar, b, roots, tz, 1.0)
    cached_bpar = _gx_bpar_term_precomputed(bpar, j1_over_alpha, roots, tz, 1.0)
    assert np.allclose(np.asarray(direct_bpar), np.asarray(cached_bpar), rtol=1.0e-6, atol=1.0e-6)


def test_stack_fields_and_placeholder_output_contract():
    G = jnp.zeros((1, 2, 3, 4, 5, 1), dtype=jnp.complex64)
    phi = jnp.ones((4, 5, 1), dtype=jnp.complex64)
    apar = 2.0 * jnp.ones((1, 1, 4, 5, 1), dtype=jnp.complex64)

    stacked = _stack_fields(G, [phi, apar])
    assert stacked.shape == (2, 1, 1, 1, 4, 5, 1)
    assert np.allclose(np.asarray(stacked[0, 0, 0, 0]), np.asarray(phi))
    assert np.allclose(np.asarray(stacked[1, 0, 0, 0]), 2.0)

    out = placeholder_nonlinear_contribution(G, weight=jnp.asarray(3.0))
    assert out.shape == G.shape
    assert np.allclose(np.asarray(out), 0.0)
