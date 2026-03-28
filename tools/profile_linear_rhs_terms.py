#!/usr/bin/env python3
"""Profile the dominant cached linear RHS term kernels on a real runtime state."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import apply_gx_geometry_grid_defaults
from spectraxgk.grids import build_spectral_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.linear import _as_species_array, build_H, build_linear_cache
from spectraxgk.runtime import (
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from spectraxgk.terms.assembly import assemble_rhs_cached_jit, compute_fields_cached
from spectraxgk.terms.linear_terms import (
    collisions_contribution,
    curvature_gradb_contribution,
    diamagnetic_contribution,
    end_damping_contribution,
    hypercollisions_contribution,
    hyperdiffusion_contribution,
    mirror_contribution,
    streaming_contribution_gx,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile per-term linear RHS kernels.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("examples/configs/runtime_cyclone_nonlinear_gx.toml"),
    )
    p.add_argument("--ky", type=float, default=0.3)
    p.add_argument("--kx", type=float, default=None)
    p.add_argument("--Nl", type=int, default=4)
    p.add_argument("--Nm", type=int, default=8)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _block_tree(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        jax.block_until_ready(leaf)


def _time_callable(fn, *args, repeats: int):
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
    out_dtype = jnp.result_type(G0, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype
    imag = jnp.asarray(1j, dtype=out_dtype)
    ns = int(G0.shape[0])

    charge = _as_species_array(params.charge_sign, ns, "charge_sign").astype(real_dtype)
    density = _as_species_array(params.density, ns, "density").astype(real_dtype)
    mass = _as_species_array(params.mass, ns, "mass").astype(real_dtype)
    temp = _as_species_array(params.temp, ns, "temp").astype(real_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tprim = _as_species_array(params.R_over_LTi, ns, "R_over_LTi").astype(real_dtype)
    fprim = _as_species_array(params.R_over_Ln, ns, "R_over_Ln").astype(real_dtype)
    nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)

    field_fn = jax.jit(lambda G: compute_fields_cached(G, cache, params, terms=term_cfg))
    fields_time, fields = _time_callable(field_fn, G0, repeats=args.repeats)
    apar = fields.apar if fields.apar is not None else jnp.zeros_like(fields.phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(fields.phi)
    H = build_H(G0, cache.Jl, fields.phi, tz, apar=apar, vth=vth, bpar=bpar, JlB=cache.JlB)

    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    omega_star_scale = jnp.asarray(params.omega_star_scale, dtype=real_dtype)
    kpar_scale = jnp.asarray(params.kpar_scale, dtype=real_dtype)
    nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
    nu_hyper_l = jnp.asarray(params.nu_hyper_l, dtype=real_dtype)
    nu_hyper_m = jnp.asarray(params.nu_hyper_m, dtype=real_dtype)
    nu_hyper_lm = jnp.asarray(params.nu_hyper_lm, dtype=real_dtype)
    hypercollisions_const = jnp.asarray(params.hypercollisions_const, dtype=real_dtype)
    hypercollisions_kz = jnp.asarray(params.hypercollisions_kz, dtype=real_dtype)
    D_hyper = jnp.asarray(params.D_hyper, dtype=real_dtype)
    p_hyper_kperp = jnp.asarray(params.p_hyper_kperp, dtype=real_dtype)
    damp_amp = jnp.asarray(params.damp_ends_amp, dtype=real_dtype)

    kernel_fns = {
        "build_H": jax.jit(lambda: build_H(G0, cache.Jl, fields.phi, tz, apar=apar, vth=vth, bpar=bpar, JlB=cache.JlB)),
        "streaming": jax.jit(
            lambda: streaming_contribution_gx(
                G0,
                phi=fields.phi,
                apar=apar,
                bpar=bpar,
                Jl=cache.Jl,
                JlB=cache.JlB,
                tz=tz,
                kz=cache.kz,
                dz=cache.dz,
                vth=vth,
                sqrt_p=cache.sqrt_p,
                sqrt_m=cache.sqrt_m_ladder,
                kpar_scale=kpar_scale,
                weight=jnp.asarray(term_cfg.streaming, dtype=real_dtype),
                linked_indices=cache.linked_indices,
                linked_kz=cache.linked_kz,
                linked_inverse_permutation=cache.linked_inverse_permutation,
                linked_full_cover=cache.linked_full_cover,
                linked_gather_map=cache.linked_gather_map,
                linked_gather_mask=cache.linked_gather_mask,
                linked_use_gather=cache.linked_use_gather,
                use_twist_shift=cache.use_twist_shift,
            )
        ),
        "mirror": jax.jit(
            lambda: mirror_contribution(
                H,
                vth=vth,
                bgrad=cache.bgrad,
                l=cache.l,
                sqrt_m=cache.sqrt_m,
                sqrt_m_p1=cache.sqrt_m_p1,
                weight=jnp.asarray(term_cfg.mirror, dtype=real_dtype),
            )
        ),
        "curvature_gradb": jax.jit(
            lambda: curvature_gradb_contribution(
                H,
                tz=tz,
                omega_d_scale=omega_d_scale,
                cv_d=cache.cv_d,
                gb_d=cache.gb_d,
                l=cache.l,
                m=cache.m,
                imag=imag,
                weight_curv=jnp.asarray(term_cfg.curvature, dtype=real_dtype),
                weight_gradb=jnp.asarray(term_cfg.gradb, dtype=real_dtype),
            )
        ),
        "diamagnetic": jax.jit(
            lambda: diamagnetic_contribution(
                jnp.zeros_like(G0),
                phi=fields.phi,
                apar=apar,
                bpar=bpar,
                Jl=cache.Jl,
                JlB=cache.JlB,
                l4=cache.l4,
                tprim=tprim,
                fprim=fprim,
                tz=tz,
                vth=vth,
                omega_star_scale=omega_star_scale,
                ky=cache.ky,
                imag=imag,
                weight=jnp.asarray(term_cfg.diamagnetic, dtype=real_dtype),
            )
        ),
        "collisions": jax.jit(
            lambda: collisions_contribution(
                H,
                G=G0,
                Jl=cache.Jl,
                JlB=cache.JlB,
                b=cache.b,
                nu=nu,
                lb_lam=cache.lb_lam,
                weight=jnp.asarray(term_cfg.collisions, dtype=real_dtype),
            )
        ),
        "hypercollisions": jax.jit(
            lambda: hypercollisions_contribution(
                G0,
                vth=vth,
                nu_hyper=nu_hyper,
                nu_hyper_l=nu_hyper_l,
                nu_hyper_m=nu_hyper_m,
                nu_hyper_lm=nu_hyper_lm,
                hyper_ratio=cache.hyper_ratio,
                ratio_l=cache.ratio_l,
                ratio_m=cache.ratio_m,
                ratio_lm=cache.ratio_lm,
                mask_const=cache.mask_const,
                mask_kz=cache.mask_kz,
                m_pow=cache.m_pow,
                m_norm_kz_factor=cache.m_norm_kz_factor,
                kz=cache.kz,
                kpar_scale=kpar_scale,
                hypercollisions_const=hypercollisions_const,
                hypercollisions_kz=hypercollisions_kz,
                weight=jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype),
                linked_indices=cache.linked_indices,
                linked_kz=cache.linked_kz,
                linked_inverse_permutation=cache.linked_inverse_permutation,
                linked_full_cover=cache.linked_full_cover,
                linked_gather_map=cache.linked_gather_map,
                linked_gather_mask=cache.linked_gather_mask,
                linked_use_gather=cache.linked_use_gather,
            )
        ),
        "hyperdiffusion": jax.jit(
            lambda: hyperdiffusion_contribution(
                G0,
                kx=cache.kx,
                ky=cache.ky,
                dealias_mask=cache.dealias_mask,
                D_hyper=D_hyper,
                p_hyper_kperp=p_hyper_kperp,
                weight=jnp.asarray(term_cfg.hyperdiffusion, dtype=real_dtype),
            )
        ),
        "end_damping": jax.jit(
            lambda: end_damping_contribution(
                H,
                ky=cache.ky,
                damp_profile=cache.damp_profile,
                linked_damp_profile=cache.linked_damp_profile,
                damp_amp=damp_amp,
                weight=jnp.asarray(term_cfg.end_damping, dtype=real_dtype),
            )
        ),
        "full_linear_rhs": jax.jit(lambda: assemble_rhs_cached_jit(G0, cache, params, term_cfg)),
    }

    rows = [
        {
            "term": "field_solve",
            "seconds": fields_time,
            "norm": float(np.asarray(jnp.linalg.norm(fields.phi))),
        }
    ]
    for name, fn in kernel_fns.items():
        seconds, out = _time_callable(fn, repeats=args.repeats)
        arr = out[0] if name == "full_linear_rhs" else out
        rows.append(
            {
                "term": name,
                "seconds": seconds,
                "norm": float(np.asarray(jnp.linalg.norm(arr))),
            }
        )

    for row in rows:
        print(f"{row['term']}: seconds={row['seconds']:.6f} norm={row['norm']:.6e}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["term", "seconds", "norm"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
