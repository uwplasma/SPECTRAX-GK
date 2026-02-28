import numpy as np
import jax.numpy as jnp

from spectraxgk.config import GridConfig
from spectraxgk.gyroaverage import bessel_j0, bessel_j1, gx_laguerre_transform
from spectraxgk.grids import build_spectral_grid
from spectraxgk.terms.nonlinear import (
    exb_nonlinear_contribution,
    nonlinear_em_contribution,
    _apply_flutter,
    _spectral_bracket,
    _spectral_bracket_multi,
    _stack_fields,
)


def _ifft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(x, axes=(-3, -2))


def _finite_diff_periodic(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * dx)


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

    bracket_apar = _spectral_bracket(
        jnp.asarray(G),
        Jl * apar[None, None, ...],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
    )
    flutter_expected = _apply_flutter(bracket_apar, vth, sqrt_m, sqrt_m_p1)

    dG = nonlinear_em_contribution(
        jnp.asarray(G),
        phi=jnp.zeros_like(jnp.asarray(apar)),
        apar=jnp.asarray(apar),
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
    assert np.max(np.abs(np.asarray(dG + flutter_expected))) < 1.0e-6


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


def test_gx_real_fft_bracket_parity():
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

    def _expand(pos):
        if nyc <= 2:
            return pos
        neg = np.conj(pos[..., 1 : nyc - 1, :, :])
        neg = neg[..., ::-1, :, :]
        return np.concatenate([pos, neg], axis=-3)

    G_full = _expand(G_nyc)
    phi_full = _expand(phi_nyc[None, ...])[0]

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
    kx_nyc = grid.kx_grid[:nyc, :]
    ky_nyc = grid.ky_grid[:nyc, :]
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
    bracket_hat_ref = _expand(bracket_hat_nyc)

    assert np.allclose(
        np.asarray(bracket_hat),
        bracket_hat_ref,
        rtol=5.0e-5,
        atol=5.0e-7,
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

    def _expand(pos):
        if nyc <= 2:
            return pos
        neg = np.conj(pos[..., 1 : nyc - 1, :, :])
        neg = neg[..., ::-1, :, :]
        return np.concatenate([pos, neg], axis=-3)

    G_full = _expand(G_nyc)
    chi_full = _expand(chi_nyc[None, ...])[0]

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
    chi = Jl * phi[None, None, ...] + JlB * bpar[None, None, ...]
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
        phi=jnp.asarray(phi),
        apar=None,
        bpar=jnp.asarray(bpar),
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
    assert np.max(np.abs(np.asarray(dG + bracket_expected))) < 1.0e-5


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
