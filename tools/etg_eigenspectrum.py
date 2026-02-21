#!/usr/bin/env python3
"""Dump leading eigenvalues for an ETG ky using Arnoldi."""

from __future__ import annotations

import argparse
import numpy as np
import jax.numpy as jnp

from spectraxgk.analysis import select_ky_index
from spectraxgk.benchmarks import ETGBaseCase, _electron_only_params, ETG_OMEGA_D_SCALE, ETG_OMEGA_STAR_SCALE, ETG_RHO_STAR
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearTerms, build_linear_cache
from spectraxgk.linear_krylov import _arnoldi, _apply_operator
from spectraxgk.terms.config import TermConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ky", type=float, default=25.0)
    parser.add_argument("--Nl", type=int, default=48)
    parser.add_argument("--Nm", type=int, default=16)
    parser.add_argument("--krylov-dim", type=int, default=24)
    parser.add_argument("--mode", type=str, default="propagator", choices=("operator", "propagator"))
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    cfg = ETGBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), args.ky)
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    params = _electron_only_params(
        cfg.model,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=ETG_OMEGA_D_SCALE,
        omega_star_scale=ETG_OMEGA_STAR_SCALE,
        rho_star=ETG_RHO_STAR,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        nhermite=args.Nm,
    )
    terms = LinearTerms(bpar=0.0, hypercollisions=0.0)
    term_cfg = TermConfig(
        streaming=terms.streaming,
        mirror=terms.mirror,
        curvature=terms.curvature,
        gradb=terms.gradb,
        diamagnetic=terms.diamagnetic,
        collisions=terms.collisions,
        hypercollisions=terms.hypercollisions,
        end_damping=terms.end_damping,
        apar=terms.apar,
        bpar=terms.bpar,
        nonlinear=0.0,
    )

    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    rng = np.random.default_rng(0)
    v0 = rng.normal(size=(1, args.Nl, args.Nm, grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(1, args.Nl, args.Nm, grid.ky.size, grid.kx.size, grid.z.size)
    )
    v0 = jnp.asarray(v0)

    if args.mode == "operator":
        apply_op = _apply_operator
    else:
        from spectraxgk.linear_krylov import _advance_imex2

        dt_val = jnp.asarray(args.dt, dtype=jnp.real(v0).dtype)

        def apply_op(x, cache, params, term_cfg):
            return _advance_imex2(x, cache, params, term_cfg, dt_val)

    V, H = _arnoldi(v0, apply_op, cache, params, term_cfg, args.krylov_dim)
    Hk = np.asarray(H[: args.krylov_dim, : args.krylov_dim])
    eigvals = np.linalg.eigvals(Hk)
    if args.mode == "propagator":
        eigvals = np.log(eigvals) / args.dt
    order = np.argsort(np.real(eigvals))[::-1]
    print(f"ETG ky={args.ky:.3f} (ky_index={ky_index}) top eigenvalues:")
    for i in range(min(args.top, eigvals.size)):
        lam = eigvals[order[i]]
        gamma = np.real(lam)
        omega = -np.imag(lam)
        print(f"{i:2d}: gamma={gamma:+.6f} omega={omega:+.6f} (eig={lam.real:+.6f}{lam.imag:+.6f}j)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
