#!/usr/bin/env python3
"""Profile the main nonlinear Cyclone kernels on a real runtime state."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.io import load_runtime_from_toml
from spectraxgk.linear import build_linear_cache
from spectraxgk.runtime import (
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from spectraxgk.geometry import apply_gx_geometry_grid_defaults
from spectraxgk.grids import build_spectral_grid
from spectraxgk.nonlinear import nonlinear_rhs_cached
from spectraxgk.terms.assembly import assemble_rhs_cached_jit, compute_fields_cached
from spectraxgk.terms.nonlinear import nonlinear_em_contribution


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile nonlinear field solve vs bracket vs full RHS.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("examples/configs/runtime_cyclone_nonlinear_gx.toml"),
    )
    p.add_argument("--ky", type=float, default=0.3)
    p.add_argument("--kx", type=float, default=None)
    p.add_argument("--Nl", type=int, default=4)
    p.add_argument("--Nm", type=int, default=8)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--laguerre-mode", type=str, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _block_tree(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        jax.block_until_ready(leaf)


def _time_callable(fn, *args, repeats: int) -> tuple[float, object]:
    out = fn(*args)
    _block_tree(out)
    t0 = time.perf_counter()
    last = out
    for _ in range(repeats):
        last = fn(*args)
        _block_tree(last)
    t1 = time.perf_counter()
    return (t1 - t0) / float(repeats), last


def main() -> None:
    args = _parse_args()
    cfg, _ = load_runtime_from_toml(args.config)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
    term_cfg = build_runtime_term_config(cfg)
    laguerre_mode = cfg.time.laguerre_nonlinear_mode if args.laguerre_mode is None else str(args.laguerre_mode)

    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=args.ky,
        kx_target=args.kx,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    G0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=args.Nl,
        Nm=args.Nm,
        nspecies=len(cfg.species),
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    G0 = jnp.asarray(G0)

    field_fn = jax.jit(lambda G: compute_fields_cached(G, cache, params, terms=term_cfg))
    fields = field_fn(G0)
    _block_tree(fields)

    nonlinear_fn = jax.jit(
        lambda G, phi, apar, bpar: nonlinear_em_contribution(
            G,
            phi=phi,
            apar=apar,
            bpar=bpar,
            Jl=cache.Jl,
            JlB=cache.JlB,
            tz=jnp.asarray(params.tz),
            vth=jnp.asarray(params.vth),
            sqrt_m=cache.sqrt_m,
            sqrt_m_p1=cache.sqrt_m_p1,
            kx_grid=cache.kx_grid,
            ky_grid=cache.ky_grid,
            dealias_mask=cache.dealias_mask,
            kxfac=cache.kxfac,
            weight=jnp.asarray(term_cfg.nonlinear, dtype=jnp.real(jnp.empty((), dtype=G.dtype)).dtype),
            apar_weight=float(term_cfg.apar),
            bpar_weight=float(term_cfg.bpar),
            laguerre_to_grid=cache.laguerre_to_grid,
            laguerre_to_spectral=cache.laguerre_to_spectral,
            laguerre_roots=cache.laguerre_roots,
            laguerre_j0=cache.laguerre_j0,
            laguerre_j1_over_alpha=cache.laguerre_j1_over_alpha,
            b=cache.b,
            gx_real_fft=bool(cfg.time.gx_real_fft),
            laguerre_mode=laguerre_mode,
        )
    )
    linear_terms = replace(term_cfg, nonlinear=0.0)
    linear_rhs_fn = jax.jit(lambda G: assemble_rhs_cached_jit(G, cache, params, linear_terms))
    rhs_fn = jax.jit(
        lambda G: nonlinear_rhs_cached(
            G,
            cache,
            params,
            term_cfg,
            gx_real_fft=bool(cfg.time.gx_real_fft),
            laguerre_mode=laguerre_mode,
        )
    )

    field_time, fields = _time_callable(field_fn, G0, repeats=args.repeats)
    nl_time, nl_out = _time_callable(
        nonlinear_fn,
        G0,
        fields.phi,
        fields.apar,
        fields.bpar,
        repeats=args.repeats,
    )
    linear_rhs_time, linear_rhs_out = _time_callable(linear_rhs_fn, G0, repeats=args.repeats)
    rhs_time, rhs_out = _time_callable(rhs_fn, G0, repeats=args.repeats)
    linear_rhs_state, _linear_rhs_fields = linear_rhs_out
    rhs_state, rhs_fields = rhs_out

    rows = [
        {
            "kernel": "field_solve",
            "seconds": field_time,
            "repeats": args.repeats,
            "norm": float(np.asarray(jnp.linalg.norm(fields.phi))),
        },
        {
            "kernel": "nonlinear_bracket",
            "seconds": nl_time,
            "repeats": args.repeats,
            "norm": float(np.asarray(jnp.linalg.norm(nl_out))),
        },
        {
            "kernel": "linear_rhs",
            "seconds": linear_rhs_time,
            "repeats": args.repeats,
            "norm": float(np.asarray(jnp.linalg.norm(linear_rhs_state))),
        },
        {
            "kernel": "full_rhs",
            "seconds": rhs_time,
            "repeats": args.repeats,
            "norm": float(np.asarray(jnp.linalg.norm(rhs_state))),
        },
    ]

    for row in rows:
        print(f"{row['kernel']}: seconds={row['seconds']:.6f} norm={row['norm']:.6e}")

    print(
        "rhs_fields:",
        f"phi_norm={float(np.asarray(jnp.linalg.norm(rhs_fields.phi))):.6e}",
        f"apar_norm={float(np.asarray(jnp.linalg.norm(rhs_fields.apar))) if rhs_fields.apar is not None else 0.0:.6e}",
        f"bpar_norm={float(np.asarray(jnp.linalg.norm(rhs_fields.bpar))) if rhs_fields.bpar is not None else 0.0:.6e}",
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["kernel", "seconds", "repeats", "norm"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
