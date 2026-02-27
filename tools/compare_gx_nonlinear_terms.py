#!/usr/bin/env python3
"""Compare GX nonlinear RHS term dumps against SPECTRAX nonlinear components."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from netCDF4 import Dataset

from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    CycloneBaseCase,
    _apply_gx_hypercollisions,
)
from spectraxgk.config import GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.gyroaverage import gx_laguerre_nj
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.terms.config import TermConfig
from spectraxgk.terms.nonlinear import (
    _gx_j0_field,
    _laguerre_to_grid,
    nonlinear_em_components,
)


def _load_shape(path: Path) -> dict[str, int]:
    data: dict[str, int] = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            data[parts[0]] = int(parts[1])
    return data


def _load_bin(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.complex64)
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"{path} size {raw.size} does not match expected {shape}")
    return raw.reshape(shape)


def _load_float_bin(path: Path, n: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != n:
        raise ValueError(f"{path} size {raw.size} does not match expected {n}")
    return raw


def _reshape_gx(
    raw: np.ndarray,
    *,
    nspec: int,
    nl: int,
    nm: int,
    nyc: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    nR = nyc * nx * nz
    # GX flattens moments with moment_idx = l + Nl * m (m-major).
    arr = raw.reshape((nspec, nm, nl, nR)).transpose(0, 2, 1, 3)
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    arr = arr[..., idxyz.ravel()]
    return arr.reshape((nspec, nl, nm, nyc, nx, nz))


def _expand_ky(arr: np.ndarray, *, nyc: int) -> np.ndarray:
    """Expand Nyc (real FFT) axis to full Ny using conjugate symmetry."""
    if arr.shape[-3] != nyc:
        raise ValueError("Expected ky axis at position -3 with length nyc")
    ny_full = 2 * (nyc - 1)
    if ny_full <= 0:
        return arr
    pos = arr
    if nyc <= 2:
        return pos
    neg = np.conj(pos[..., 1 : nyc - 1, :, :])
    neg = neg[..., ::-1, :, :]
    return np.concatenate([pos, neg], axis=-3)


def _summary(label: str, ref: np.ndarray, test: np.ndarray) -> None:
    diff = test - ref
    max_ref = float(np.max(np.abs(ref)))
    max_test = float(np.max(np.abs(test)))
    max_diff = float(np.max(np.abs(diff)))
    if max_ref == 0.0:
        max_rel = float("nan")
        rms_rel = float("nan")
    else:
        thresh = max_ref * 1.0e-12
        mask = np.abs(ref) > thresh
        if not np.any(mask):
            max_rel = float("nan")
            rms_rel = float("nan")
        else:
            rel = diff[mask] / ref[mask]
            max_rel = float(np.max(np.abs(rel)))
            rms_rel = float(
                np.sqrt(np.mean(np.abs(diff[mask]) ** 2))
                / (np.sqrt(np.mean(np.abs(ref[mask]) ** 2)) + 1.0e-30)
            )
    idx_diff = np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape)
    ref_at = ref[idx_diff]
    test_at = test[idx_diff]
    print(
        f"{label:12s} max|ref|={max_ref:.3e} max|test|={max_test:.3e} "
        f"max|diff|={max_diff:.3e} max|rel|={max_rel:.3e} rms_rel={rms_rel:.3e} "
        f"idx={idx_diff} ref={ref_at:.3e} test={test_at:.3e}"
    )


def _apply_kx_order(arr: np.ndarray, *, order: str, kx_axis: int) -> np.ndarray:
    if order == "native":
        return arr
    if order == "fftshift":
        return np.fft.fftshift(arr, axes=(kx_axis,))
    if order == "ifftshift":
        return np.fft.ifftshift(arr, axes=(kx_axis,))
    raise ValueError(f"Unknown kx order '{order}'")


def _broadcast_grid(grid: jnp.ndarray, ndim: int) -> jnp.ndarray:
    shape = (1,) * (ndim - 3) + grid.shape + (1,)
    return jnp.reshape(grid, shape)


def _ifft2_xy(arr: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(arr, axes=(-3, -2))


def _bracket_real(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    ny_full: int | None = None,
) -> jnp.ndarray:
    """Return real-space bracket {G, chi} without spectral masking."""
    dchi_dx, dchi_dy = _grad_xy_real(
        chi_hat, kx_grid=kx_grid, ky_grid=ky_grid, ny_full=ny_full
    )
    dchi_dx = dchi_dx[:, :, None, ...]
    dchi_dy = dchi_dy[:, :, None, ...]
    dG_dx, dG_dy = _grad_xy_real(G_hat, kx_grid=kx_grid, ky_grid=ky_grid, ny_full=ny_full)
    bracket = dG_dx * dchi_dy - dG_dy * dchi_dx
    return bracket


def _grad_xy_real(
    F_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    ny_full: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    imag = jnp.asarray(1j, dtype=F_hat.dtype)
    kx = jnp.asarray(kx_grid)
    ky = jnp.asarray(ky_grid)
    if ny_full is None:
        ny_full = int(ky.shape[0])
    use_rfft = int(ky.shape[0]) != int(ny_full)
    nxy = int(kx.shape[1]) * int(ny_full)
    kx_b = _broadcast_grid(kx, F_hat.ndim)
    ky_b = _broadcast_grid(ky, F_hat.ndim)
    if use_rfft:
        # ky axis is Nyc; use irfft2 with ky axis last in axes tuple.
        dF_dx = jnp.fft.irfft2(
            imag * kx_b * F_hat, s=(int(kx.shape[1]), int(ny_full)), axes=(-2, -3)
        )
        dF_dy = jnp.fft.irfft2(
            imag * ky_b * F_hat, s=(int(kx.shape[1]), int(ny_full)), axes=(-2, -3)
        )
        dF_dx = dF_dx * nxy
        dF_dy = dF_dy * nxy
    else:
        dF_dx = _ifft2_xy(imag * kx_b * F_hat) * nxy
        dF_dy = _ifft2_xy(imag * ky_b * F_hat) * nxy
    dF_dx = jnp.real(dF_dx)
    dF_dy = jnp.real(dF_dy)
    return dF_dx, dF_dy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory with GX nonlinear term dumps")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file to map ky indices")
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nl", type=int, default=48)
    parser.add_argument("--Nm", type=int, default=16)
    parser.add_argument("--Ny", type=int, default=24)
    parser.add_argument("--Nz", type=int, default=96)
    parser.add_argument("--Lx", type=float, default=62.8)
    parser.add_argument("--Ly", type=float, default=62.8)
    parser.add_argument("--y0", type=float, default=20.0)
    parser.add_argument("--boundary", type=str, default="linked")
    parser.add_argument("--ntheta", type=int, default=None)
    parser.add_argument("--nperiod", type=int, default=None)
    parser.add_argument(
        "--kx-order",
        type=str,
        default="auto",
        help="kx ordering for spectral arrays: native, fftshift, ifftshift, or auto",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional npz output for SPECTRAX terms")
    args = parser.parse_args()

    shape_path = args.gx_dir / "rhs_terms_shape.txt"
    if not shape_path.exists():
        raise FileNotFoundError(f"Missing shape file {shape_path}")
    shape = _load_shape(shape_path)
    nspec = shape.get("nspec", 1)
    nl = shape.get("nl", args.Nl)
    nm = shape.get("nm", args.Nm)
    nyc = shape.get("nyc", args.Ny // 2 + 1)
    nx = shape.get("nx", 1)
    nz = shape.get("nz", args.Nz)
    nj = shape.get("nj")
    if nj is None:
        nj = gx_laguerre_nj(nl)
    gx_shape = (nspec, nl, nm, nyc, nx, nz)

    g_state_path = args.gx_dir / "nl_g_state.bin"
    if not g_state_path.exists():
        g_state_path = args.gx_dir / "g_state.bin"
    g_state_nyc = _reshape_gx(
        _load_bin(g_state_path, gx_shape),
        nspec=nspec,
        nl=nl,
        nm=nm,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )
    g_state = _expand_ky(g_state_nyc, nyc=nyc)

    kx_vals = None
    ky_vals = None
    kx_dump = args.gx_dir / "nl_kx.bin"
    ky_dump = args.gx_dir / "nl_ky.bin"
    if kx_dump.exists() and ky_dump.exists():
        kx_vals = _load_float_bin(kx_dump, nx).astype(float)
        ky_vals = _load_float_bin(ky_dump, nyc).astype(float)
    if kx_vals is None or ky_vals is None:
        with Dataset(args.gx_out, "r") as root:
            ky_vals = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)
            kx_vals = np.asarray(root.groups["Grids"].variables["kx"][:], dtype=float)
    if kx_vals is None or ky_vals is None:
        raise ValueError("Failed to load GX kx/ky values")
    if ky_vals.size == nyc:
        ky_vals_nyc = ky_vals
        ny_full = 2 * (nyc - 1)
        if ny_full > nyc:
            ky_neg = -ky_vals[1 : nyc - 1][::-1]
            ky_vals_full = np.concatenate([ky_vals, ky_neg])
        else:
            ky_vals_full = ky_vals
    else:
        ky_vals_full = ky_vals
        ny_full = int(ky_vals_full.size)
        ky_vals_nyc = ky_vals_full[:nyc]
    ky_idx = int(np.argmin(np.abs(ky_vals_nyc - float(args.ky))))
    ky_grid_full, kx_grid_full = np.meshgrid(ky_vals_full, kx_vals, indexing="ij")
    ky_grid_nyc, kx_grid_nyc = np.meshgrid(ky_vals_nyc, kx_vals, indexing="ij")

    phi_path = args.gx_dir / "nl_phi.bin"
    if not phi_path.exists():
        phi_path = args.gx_dir / "phi.bin"
    phi_nyc = _reshape_gx(
        _load_bin(phi_path, (1, 1, 1, nyc, nx, nz)),
        nspec=1,
        nl=1,
        nm=1,
        nyc=nyc,
        nx=nx,
        nz=nz,
    )[0, 0, 0, ...]
    phi = _expand_ky(phi_nyc[None, ...], nyc=nyc)[0]
    apar_path = args.gx_dir / "apar.bin"
    bpar_path = args.gx_dir / "bpar.bin"
    apar = None
    bpar = None
    if apar_path.exists():
        apar = _reshape_gx(
            _load_bin(apar_path, (1, 1, 1, nyc, nx, nz)),
            nspec=1,
            nl=1,
            nm=1,
            nyc=nyc,
            nx=nx,
            nz=nz,
        )[0, 0, 0, ...]
        apar = _expand_ky(apar[None, ...], nyc=nyc)[0]
    if bpar_path.exists():
        bpar = _reshape_gx(
            _load_bin(bpar_path, (1, 1, 1, nyc, nx, nz)),
            nspec=1,
            nl=1,
            nm=1,
            nyc=nyc,
            nx=nx,
            nz=nz,
        )[0, 0, 0, ...]
        bpar = _expand_ky(bpar[None, ...], nyc=nyc)[0]

    use_gx_spacing = ky_vals_nyc.size > 1 and kx_vals.size > 1
    if use_gx_spacing and ky_vals_nyc.size > 1:
        delta_ky = float(ky_vals_nyc[1] - ky_vals_nyc[0])
        y0 = 1.0 / delta_ky if delta_ky != 0.0 else args.y0
    else:
        y0 = args.y0
    if use_gx_spacing and kx_vals.size > 1:
        delta_kx = float(kx_vals[1] - kx_vals[0])
        Lx = float(2.0 * np.pi / delta_kx) if delta_kx != 0.0 else args.Lx
    else:
        Lx = args.Lx
    if args.boundary == "linked" and y0 is not None:
        geom_tmp = SAlphaGeometry.from_config(CycloneBaseCase().geometry)
        theta_tmp = np.linspace(-np.pi, np.pi, nz, endpoint=False)
        gds2_tmp, gds21_tmp, gds22_tmp = geom_tmp.metric_coeffs(theta_tmp)
        gds21_min = float(gds21_tmp[0]) if np.ndim(gds21_tmp) else float(gds21_tmp)
        gds22_min = float(gds22_tmp[0]) if np.ndim(gds22_tmp) else float(gds22_tmp)
        shat = float(geom_tmp.s_hat)
        twist_shift_geo_fac = 0.0
        if gds22_min != 0.0:
            twist_shift_geo_fac = float(2.0 * shat * gds21_min / gds22_min)
        if twist_shift_geo_fac != 0.0:
            jtwist = int(np.round(twist_shift_geo_fac))
            if jtwist == 0:
                jtwist = 1
            x0 = float(y0) * abs(jtwist) / abs(twist_shift_geo_fac)
            Lx = float(2.0 * np.pi * x0)
    cfg = CycloneBaseCase(
        grid=GridConfig(
            Nx=nx,
            Ny=ny_full,
            Nz=nz,
            Lx=Lx,
            Ly=args.Ly,
            boundary=args.boundary,
            y0=y0,
            ntheta=args.ntheta,
            nperiod=args.nperiod,
        )
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - float(args.ky))))
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        omega_d_scale=CYCLONE_OMEGA_D_SCALE,
        omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
        rho_star=CYCLONE_RHO_STAR,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    params = _apply_gx_hypercollisions(params, nhermite=nm)
    cache = build_linear_cache(grid, geom, params, nl, nm)

    gx_dg_dx = args.gx_dir / "nl_dg_dx.bin"
    gx_dg_dy = args.gx_dir / "nl_dg_dy.bin"
    gx_dj_dx = args.gx_dir / "nl_dJ0phi_dx.bin"
    gx_dj_dy = args.gx_dir / "nl_dJ0phi_dy.bin"
    have_derivs = gx_dg_dx.exists() and gx_dg_dy.exists() and gx_dj_dx.exists() and gx_dj_dy.exists()
    ref_dg_dx = None
    ref_dg_dy = None
    ref_dj_dx = None
    ref_dj_dy = None
    if have_derivs:
        raw = np.fromfile(gx_dj_dx, dtype=np.float32)
        expected = nj * nz * nx * ny_full
        if raw.size == expected:
            ref_dj_dx = raw.reshape((nj, nz, nx, ny_full)).transpose(0, 3, 2, 1)
        raw = np.fromfile(gx_dj_dy, dtype=np.float32)
        if raw.size == expected:
            ref_dj_dy = raw.reshape((nj, nz, nx, ny_full)).transpose(0, 3, 2, 1)
        raw = np.fromfile(gx_dg_dx, dtype=np.float32)
        expected_g = nm * nj * nz * nx * ny_full
        if raw.size == expected_g:
            ref_dg_dx = raw.reshape((nm, nj, nz, nx, ny_full)).transpose(0, 1, 4, 3, 2)
        raw = np.fromfile(gx_dg_dy, dtype=np.float32)
        if raw.size == expected_g:
            ref_dg_dy = raw.reshape((nm, nj, nz, nx, ny_full)).transpose(0, 1, 4, 3, 2)
        have_derivs = ref_dg_dx is not None and ref_dg_dy is not None and ref_dj_dx is not None and ref_dj_dy is not None

    def _rms_rel(ref: np.ndarray, test: np.ndarray) -> float:
        diff = test - ref
        rms = np.sqrt(np.mean(diff * diff))
        rms_ref = np.sqrt(np.mean(ref * ref)) + 1.0e-30
        return float(rms / rms_ref)

    def _prepare_order(order: str):
        g_np = _apply_kx_order(g_state, order=order, kx_axis=-2)
        phi_np = _apply_kx_order(phi, order=order, kx_axis=-2)
        apar_np = _apply_kx_order(apar, order=order, kx_axis=-2) if apar is not None else None
        bpar_np = _apply_kx_order(bpar, order=order, kx_axis=-2) if bpar is not None else None
        kx_grid = _apply_kx_order(kx_grid_full, order=order, kx_axis=1)
        return g_np, phi_np, apar_np, bpar_np, kx_grid, ky_grid_full

    def _prepare_order_nyc(order: str):
        g_np = _apply_kx_order(g_state_nyc, order=order, kx_axis=-2)
        phi_np = _apply_kx_order(phi_nyc, order=order, kx_axis=-2)
        kx_grid = _apply_kx_order(kx_grid_nyc, order=order, kx_axis=1)
        return g_np, phi_np, kx_grid, ky_grid_nyc

    def _compare_derivs(order: str) -> float:
        g_np, phi_np, kx_grid, ky_grid = _prepare_order_nyc(order)
        g_mu = _laguerre_to_grid(jnp.asarray(g_np), cache.laguerre_to_grid)
        b_nyc = cache.b[:, :nyc, :, :]
        chi_phi = _gx_j0_field(
            jnp.asarray(phi_np.astype(np.complex64)),
            b_nyc,
            cache.laguerre_roots,
            1.0,
        )
        dchi_dx, dchi_dy = _grad_xy_real(
            chi_phi,
            kx_grid=jnp.asarray(kx_grid),
            ky_grid=jnp.asarray(ky_grid),
            ny_full=ny_full,
        )
        dG_dx, dG_dy = _grad_xy_real(
            g_mu,
            kx_grid=jnp.asarray(kx_grid),
            ky_grid=jnp.asarray(ky_grid),
            ny_full=ny_full,
        )
        test_dj_dx = np.asarray(dchi_dx[0])
        test_dj_dy = np.asarray(dchi_dy[0])
        test_dg_dx = np.asarray(dG_dx[0]).transpose(1, 0, 2, 3, 4)
        test_dg_dy = np.asarray(dG_dy[0]).transpose(1, 0, 2, 3, 4)
        rms = (
            _rms_rel(ref_dj_dx, test_dj_dx)
            + _rms_rel(ref_dj_dy, test_dj_dy)
            + _rms_rel(ref_dg_dx, test_dg_dx)
            + _rms_rel(ref_dg_dy, test_dg_dy)
        )
        print(f"deriv rms sum ({order}) = {rms:.3e}")
        return rms

    order = args.kx_order.lower()
    if order == "auto" and have_derivs:
        candidates = ["native", "fftshift", "ifftshift"]
        scores = {cand: _compare_derivs(cand) for cand in candidates}
        order = min(scores, key=scores.get)
        print(f"Selected kx order: {order}")
    elif order == "auto":
        order = "native"
    g_state, phi, apar, bpar, kx_grid, ky_grid = _prepare_order(order)

    term_cfg = TermConfig(nonlinear=1.0, apar=1.0, bpar=1.0)
    G = jnp.asarray(g_state.astype(np.complex64))
    comps = nonlinear_em_components(
        G,
        phi=jnp.asarray(phi.astype(np.complex64)),
        apar=jnp.asarray(apar.astype(np.complex64)) if apar is not None else None,
        bpar=jnp.asarray(bpar.astype(np.complex64)) if bpar is not None else None,
        Jl=cache.Jl,
        JlB=cache.JlB,
        tz=params.tz,
        vth=params.vth,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        kx_grid=jnp.asarray(kx_grid),
        ky_grid=jnp.asarray(ky_grid),
        dealias_mask=cache.dealias_mask,
        kxfac=cache.kxfac,
        weight=term_cfg.nonlinear,
        apar_weight=term_cfg.apar,
        bpar_weight=term_cfg.bpar,
        laguerre_to_grid=cache.laguerre_to_grid,
        laguerre_to_spectral=cache.laguerre_to_spectral,
        laguerre_roots=cache.laguerre_roots,
        b=cache.b,
    )

    # Real-space bracket comparison if GX dump is available (use Nyc path).
    g_nyc, phi_nyc_ord, kx_grid_nyc_ord, ky_grid_nyc_ord = _prepare_order_nyc(order)
    g_mu_nyc = _laguerre_to_grid(jnp.asarray(g_nyc), cache.laguerre_to_grid)
    b_nyc = cache.b[:, :nyc, :, :]
    chi_phi_nyc = _gx_j0_field(
        jnp.asarray(phi_nyc_ord.astype(np.complex64)),
        b_nyc,
        cache.laguerre_roots,
        1.0,
    )
    bracket_real = _bracket_real(
        g_mu_nyc,
        chi_phi_nyc,
        kx_grid=jnp.asarray(kx_grid_nyc_ord),
        ky_grid=jnp.asarray(ky_grid_nyc_ord),
        ny_full=ny_full,
    )
    if have_derivs:
        g_nyc, phi_nyc_ord, kx_grid_nyc_ord, ky_grid_nyc_ord = _prepare_order_nyc(order)
        g_mu_nyc = _laguerre_to_grid(jnp.asarray(g_nyc), cache.laguerre_to_grid)
        b_nyc = cache.b[:, :nyc, :, :]
        chi_phi_nyc = _gx_j0_field(
            jnp.asarray(phi_nyc_ord.astype(np.complex64)),
            b_nyc,
            cache.laguerre_roots,
            1.0,
        )
        dchi_dx, dchi_dy = _grad_xy_real(
            chi_phi_nyc,
            kx_grid=jnp.asarray(kx_grid_nyc_ord),
            ky_grid=jnp.asarray(ky_grid_nyc_ord),
            ny_full=ny_full,
        )
        dG_dx, dG_dy = _grad_xy_real(
            g_mu_nyc,
            kx_grid=jnp.asarray(kx_grid_nyc_ord),
            ky_grid=jnp.asarray(ky_grid_nyc_ord),
            ny_full=ny_full,
        )
        test_dj_dx = np.asarray(dchi_dx[0])
        test_dj_dy = np.asarray(dchi_dy[0])
        test_dg_dx = np.asarray(dG_dx[0]).transpose(1, 0, 2, 3, 4)
        test_dg_dy = np.asarray(dG_dy[0]).transpose(1, 0, 2, 3, 4)
        _summary("dJ0phi_dx", ref_dj_dx, test_dj_dx)
        _summary("dJ0phi_dy", ref_dj_dy, test_dj_dy)
        _summary("dg_dx", ref_dg_dx, test_dg_dx)
        _summary("dg_dy", ref_dg_dy, test_dg_dy)
    gx_bracket_real = args.gx_dir / "nl_bracket_phi_real.bin"
    if gx_bracket_real.exists():
        raw_real = np.fromfile(gx_bracket_real, dtype=np.float32)
        denom = nj * nz * ny_full * nx
        m_tot = int(raw_real.size // denom) if denom > 0 else 0
        if denom == 0 or m_tot <= 0 or raw_real.size % denom != 0:
            print(f"Skipping bracket_phi_real: size {raw_real.size} not divisible by {denom}")
        else:
            m_ghost = max(0, (m_tot - nm) // 2)
            if m_tot != nm + 2 * m_ghost:
                print(f"Skipping bracket_phi_real: unexpected m_tot {m_tot} for nm {nm}")
            else:
                ref_full = raw_real.reshape((m_tot, nj, nz, nx, ny_full)).transpose(
                    0, 1, 2, 4, 3
                )
                if m_ghost > 0:
                    ref_real = ref_full[m_ghost:-m_ghost]
                else:
                    ref_real = ref_full
                # bracket_real[0]: (Nj, Nm, Ny, Nx, Nz) -> (Nm, Nj, Nz, Ny, Nx)
                test_real = np.asarray(bracket_real[0]).transpose(1, 0, 4, 2, 3)
                _summary("bracket_real", ref_real, test_real)

    gx_j0phi = args.gx_dir / "nl_j0phi.bin"
    if gx_j0phi.exists():
        raw_j0 = np.fromfile(gx_j0phi, dtype=np.complex64)
        expected = nj * nz * nyc * nx
        if raw_j0.size != expected:
            print(f"Skipping j0phi: size {raw_j0.size} != {expected}")
        else:
            # J0phi layout: idxyz = idy + nyc*(idx + nx*idz)
            j0phi = raw_j0.reshape((nj, nz, nx, nyc)).transpose(0, 3, 2, 1)
            ky_idx = int(np.argmin(np.abs(ky_vals_nyc - float(args.ky))))
            b_nyc = cache.b[:, :nyc, :, :]
            chi_phi_nyc = _gx_j0_field(
                jnp.asarray(phi_nyc.astype(np.complex64)),
                b_nyc,
                cache.laguerre_roots,
                1.0,
            )
            test_j0 = np.asarray(chi_phi_nyc[0, :, ky_idx, :, :])
            ref_j0 = j0phi[:, ky_idx, :, :]
            _summary("j0phi", ref_j0, test_j0)

    exb_total = np.asarray(comps["exb_phi"]) + np.asarray(comps["exb_bpar"])
    gx_map = {
        "exb_total": ("nl_exb_phi.bin", exb_total),
        "exb_bpar": ("nl_exb_bpar.bin", np.asarray(comps["exb_bpar"])),
        "bracket_apar": ("nl_bracket_apar.bin", np.asarray(comps["bracket_apar"])),
        "flutter": ("nl_flutter.bin", np.asarray(comps["flutter"])),
        "total": ("nl_total.bin", np.asarray(comps["total"])),
    }
    for key, (fname, test_arr) in gx_map.items():
        path = args.gx_dir / fname
        if not path.exists():
            print(f"Skipping {key}: {path} not found")
            continue
        ref = _reshape_gx(
            _load_bin(path, gx_shape),
            nspec=nspec,
            nl=nl,
            nm=nm,
            nyc=nyc,
            nx=nx,
            nz=nz,
        )
        ref = ref[:, :, :, ky_idx : ky_idx + 1, :, :]
        test_slice = test_arr[:, :, :, ky_index : ky_index + 1, :, :]
        _summary(key, ref, test_slice)

    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            bracket_phi=np.asarray(comps["exb_phi"]),
            bracket_bpar=np.asarray(comps["exb_bpar"]),
            bracket_apar=np.asarray(comps["bracket_apar"]),
            flutter=np.asarray(comps["flutter"]),
            nl_total=np.asarray(comps["total"]),
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
