#!/usr/bin/env python3
"""Focused ETG high-ky physics audit against GX analytic formulas."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pprint import pformat

import numpy as np
import jax.numpy as jnp

from spectraxgk.benchmarks import ETGBaseCase
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.gyroaverage import J_l_all
from spectraxgk.terms.linear_terms import curvature_gradb_contribution


def _gx_geometry(theta: np.ndarray, q: float, s_hat: float, eps: float, R0: float, shift: float = 0.0):
    """GX s-alpha geometry formulas (see gx/src/geometry.cu)."""

    shear = s_hat * theta - shift * np.sin(theta)
    bmag = 1.0 / (1.0 + eps * np.cos(theta))
    gradpar = abs(1.0 / (q * R0))
    bgrad = gradpar * eps * np.sin(theta) * bmag
    gds2 = 1.0 + shear * shear
    gds21 = -s_hat * shear
    gds22 = (s_hat * s_hat) * np.ones_like(theta)
    gb = (np.cos(theta) + shear * np.sin(theta)) / R0
    cv = gb
    gb0 = (-s_hat * np.sin(theta)) / R0
    cv0 = gb0
    return bmag, bgrad, gds2, gds21, gds22, cv, gb, cv0, gb0


def _gx_kperp2(kx: np.ndarray, ky: np.ndarray, gds2, gds21, gds22, bmag_inv, s_hat: float):
    kx0 = kx[None, :, None]
    ky0 = ky[:, None, None]
    if s_hat == 0.0:
        kperp2 = ky0 * (ky0 * gds2[None, None, :] + 2.0 * kx0 * gds21[None, None, :])
        kperp2 = kperp2 + (kx0 * kx0) * gds22[None, None, :]
    else:
        shat_inv = 1.0 / s_hat
        kperp2 = ky0 * (ky0 * gds2[None, None, :] + 2.0 * kx0 * shat_inv * gds21[None, None, :])
        kperp2 = kperp2 + (kx0 * shat_inv) * (kx0 * shat_inv) * gds22[None, None, :]
    return kperp2 * (bmag_inv[None, None, :] ** 2)


def _gx_drift(kx: np.ndarray, ky: np.ndarray, cv, gb, cv0, gb0, s_hat: float):
    kx0 = kx[None, :, None]
    ky0 = ky[:, None, None]
    if s_hat == 0.0:
        cv_d = ky0 * cv[None, None, :] + kx0 * cv0[None, None, :]
        gb_d = ky0 * gb[None, None, :] + kx0 * gb0[None, None, :]
    else:
        shat_inv = 1.0 / s_hat
        cv_d = ky0 * cv[None, None, :] + kx0 * shat_inv * cv0[None, None, :]
        gb_d = ky0 * gb[None, None, :] + kx0 * shat_inv * gb0[None, None, :]
    return cv_d, gb_d


def _max_diff(name: str, a: np.ndarray, b: np.ndarray) -> None:
    diff = np.max(np.abs(a - b))
    denom = np.max(np.abs(b)) if np.max(np.abs(b)) > 0 else 1.0
    rel = diff / denom
    print(f"{name:>18s} | max abs={diff:.3e} rel={rel:.3e}")


def _rho_e_over_rho_i(mass_ratio: float, te_over_ti: float) -> float:
    return math.sqrt(te_over_ti / mass_ratio)


def main() -> int:
    parser = argparse.ArgumentParser(description="ETG high-ky physics audit vs GX formulas")
    parser.add_argument("--ky", type=float, default=25.0)
    parser.add_argument("--Nl", type=int, default=48)
    parser.add_argument("--Nm", type=int, default=16)
    args = parser.parse_args()

    cfg = ETGBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    theta = np.asarray(grid.z)

    print("=== ETG physics audit ===")
    print(pformat(asdict(cfg), width=120, sort_dicts=False))
    print(f"ky target = {args.ky}")

    bmag_s = np.asarray(geom.bmag(grid.z))
    bgrad_s = np.asarray(geom.bgrad(grid.z))
    gds2_s, gds21_s, gds22_s = (np.asarray(x) for x in geom.metric_coeffs(grid.z))
    cv_s, gb_s, cv0_s, gb0_s = (np.asarray(x) for x in geom.drift_coeffs(grid.z))
    bmag_gx, bgrad_gx, gds2_gx, gds21_gx, gds22_gx, cv_gx, gb_gx, cv0_gx, gb0_gx = _gx_geometry(
        theta, cfg.geometry.q, cfg.geometry.s_hat, cfg.geometry.epsilon, cfg.geometry.R0, cfg.geometry.alpha
    )

    _max_diff("bmag", bmag_s, bmag_gx)
    _max_diff("bgrad", bgrad_s, bgrad_gx)
    _max_diff("gds2", gds2_s, gds2_gx)
    _max_diff("gds21", gds21_s, gds21_gx)
    _max_diff("gds22", gds22_s, gds22_gx)
    _max_diff("cv", cv_s, cv_gx)
    _max_diff("gb", gb_s, gb_gx)
    _max_diff("cv0", cv0_s, cv0_gx)
    _max_diff("gb0", gb0_s, gb0_gx)

    ky_idx = int(np.argmin(np.abs(np.asarray(grid.ky) - args.ky)))
    ky = np.asarray(grid.ky)
    kx = np.asarray(grid.kx)

    kperp2_s = np.asarray(geom.k_perp2(kx[None, :, None], ky[:, None, None], grid.z[None, None, :]))
    kperp2_gx = _gx_kperp2(kx, ky, gds2_gx, gds21_gx, gds22_gx, 1.0 / bmag_gx, cfg.geometry.s_hat)
    _max_diff("kperp2", kperp2_s, kperp2_gx)

    cv_d_s, gb_d_s = (np.asarray(x) for x in geom.drift_components(grid.kx, grid.ky, grid.z))
    cv_d_gx, gb_d_gx = _gx_drift(kx, ky, cv_gx, gb_gx, cv0_gx, gb0_gx, cfg.geometry.s_hat)
    _max_diff("cv_d", cv_d_s, cv_d_gx)
    _max_diff("gb_d", gb_d_s, gb_d_gx)

    b = kperp2_s * (_rho_e_over_rho_i(float(cfg.model.mass_ratio), float(cfg.model.Te_over_Ti)) ** 2)
    Jl = np.asarray(J_l_all(jnp.asarray(b), l_max=args.Nl - 1))
    Jl_gx = np.exp(-0.5 * b)[None, ...] * ((-0.5 * b) ** np.arange(args.Nl)[:, None, None, None]) / np.array(
        [math.factorial(l) for l in range(args.Nl)]
    )[:, None, None, None]
    _max_diff("J_l", Jl, Jl_gx)

    # curvature / grad-B operator sign check (compare to explicit GX formula)
    Nm = args.Nm
    Nl = args.Nl
    rng = np.random.default_rng(0)
    H = rng.normal(size=(1, Nl, Nm, ky.shape[0], kx.shape[0], theta.shape[0])) + 1j * rng.normal(
        size=(1, Nl, Nm, ky.shape[0], kx.shape[0], theta.shape[0])
    )
    l = np.arange(Nl)[:, None, None, None, None]
    m = np.arange(Nm)[None, :, None, None, None]
    imag = 1j
    tz = np.array([1.0])
    omega_d_scale = np.array([1.0])
    dG = curvature_gradb_contribution(
        jnp.asarray(H),
        tz=jnp.asarray(tz),
        omega_d_scale=jnp.asarray(omega_d_scale),
        cv_d=jnp.asarray(cv_d_s),
        gb_d=jnp.asarray(gb_d_s),
        l=jnp.asarray(l),
        m=jnp.asarray(m),
        imag=jnp.asarray(imag),
        weight_curv=jnp.asarray(1.0),
        weight_gradb=jnp.asarray(1.0),
    )
    dG = np.asarray(dG)
    # Manual GX form
    def shift(arr, off, axis):
        pad = [(0, 0)] * arr.ndim
        if off > 0:
            pad[axis] = (0, off)
            arr_p = np.pad(arr, pad)
            slc = [slice(None)] * arr.ndim
            slc[axis] = slice(off, off + arr.shape[axis])
            return arr_p[tuple(slc)]
        if off < 0:
            pad[axis] = (-off, 0)
            arr_p = np.pad(arr, pad)
            slc = [slice(None)] * arr.ndim
            slc[axis] = slice(0, arr.shape[axis])
            return arr_p[tuple(slc)]
        return arr

    H_m_p2 = shift(H, 2, axis=2)
    H_m_m2 = shift(H, -2, axis=2)
    curv = np.sqrt((m + 1.0) * (m + 2.0)) * H_m_p2 + (2.0 * m + 1.0) * H + np.sqrt(
        m * (m - 1.0)
    ) * H_m_m2
    H_l_p1 = shift(H, 1, axis=1)
    H_l_m1 = shift(H, -1, axis=1)
    gradb = (l + 1.0) * H_l_p1 + (2.0 * l + 1.0) * H + l * H_l_m1
    dG_gx = -1j * cv_d_s[None, None, None, ...] * curv - 1j * gb_d_s[None, None, None, ...] * gradb
    _max_diff("curv/gradb rhs", dG, dG_gx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
