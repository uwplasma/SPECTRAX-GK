#!/usr/bin/env python3
"""Compare RHS term implementations against GX formula replicas."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
import jax.numpy as jnp

from spectraxgk.benchmarks import CycloneBaseCase
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, build_linear_cache
from spectraxgk.terms import assembly as rhs_assembly
from spectraxgk.terms.config import TermConfig
from spectraxgk.terms.fields import _solve_fields_impl
from spectraxgk.terms.linear_terms import (
    curvature_gradb_contribution,
    diamagnetic_contribution,
    mirror_contribution,
    streaming_contribution,
)
from spectraxgk.terms.operators import grad_z_periodic


def _shift_axis(arr: jnp.ndarray, offset: int, axis: int) -> jnp.ndarray:
    if offset == 0:
        return arr
    pad = [(0, 0)] * arr.ndim
    if offset > 0:
        pad[axis] = (0, offset)
        arr_pad = jnp.pad(arr, pad)
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(offset, offset + arr.shape[axis])
        return arr_pad[tuple(slc)]
    pad[axis] = (-offset, 0)
    arr_pad = jnp.pad(arr, pad)
    slc = [slice(None)] * arr.ndim
    slc[axis] = slice(0, arr.shape[axis])
    return arr_pad[tuple(slc)]


def _gx_build_H(
    G: jnp.ndarray,
    Jl: jnp.ndarray,
    phi: jnp.ndarray,
    tz: jnp.ndarray,
    *,
    apar: jnp.ndarray,
    vth: jnp.ndarray,
    bpar: jnp.ndarray,
    JlB: jnp.ndarray,
) -> jnp.ndarray:
    H = G
    H = H.at[:, :, 0, ...].add(tz[:, None, None, None, None] * Jl * phi)
    H = H.at[:, :, 0, ...].add(JlB * bpar)
    H = H.at[:, :, 1, ...].add(-tz[:, None, None, None, None] * vth[:, None, None, None, None] * Jl * apar)
    return H


def _gx_streaming(
    H: jnp.ndarray,
    *,
    dz: jnp.ndarray,
    vth: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    use_twist_shift: bool = False,
    kx_link_plus: jnp.ndarray | None = None,
    kx_link_minus: jnp.ndarray | None = None,
    kx_mask_plus: jnp.ndarray | None = None,
    kx_mask_minus: jnp.ndarray | None = None,
) -> jnp.ndarray:
    axis_m = -4
    Nm = H.shape[axis_m]
    sqrt_p = jnp.sqrt(jnp.arange(Nm) + 1.0)
    sqrt_m = jnp.sqrt(jnp.arange(Nm))
    H_p1 = _shift_axis(H, 1, axis=axis_m)
    H_m1 = _shift_axis(H, -1, axis=axis_m)
    shape = [1] * H.ndim
    shape[axis_m] = Nm
    ladder = sqrt_p.reshape(shape) * H_p1 + sqrt_m.reshape(shape) * H_m1
    vth_s = vth[:, None, None, None, None, None]
    rhs = -vth_s * ladder
    if use_twist_shift:
        if kx_link_plus is None or kx_link_minus is None or kx_mask_plus is None or kx_mask_minus is None:
            raise ValueError("kx_link arrays must be provided for twist-shift comparison")
        from spectraxgk.terms.operators import grad_z_linked

        rhs = grad_z_linked(
            rhs,
            dz=dz,
            kx_link_plus=kx_link_plus,
            kx_link_minus=kx_link_minus,
            kx_mask_plus=kx_mask_plus,
            kx_mask_minus=kx_mask_minus,
        )
    else:
        rhs = grad_z_periodic(rhs, dz)
    return kpar_scale * rhs


def _gx_mirror(
    H: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    bgrad: jnp.ndarray,
    l: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    sqrt_m_p1: jnp.ndarray,
) -> jnp.ndarray:
    axis_l = -5
    axis_m = -4
    l_p1 = l + 1.0
    H_m_p1 = _shift_axis(H, 1, axis=axis_m)
    H_m_m1 = _shift_axis(H, -1, axis=axis_m)
    mirror_term = (
        -sqrt_m_p1 * l_p1 * H_m_p1
        - sqrt_m_p1 * l * _shift_axis(H_m_p1, -1, axis=axis_l)
        + sqrt_m * l * H_m_m1
        + sqrt_m * l_p1 * _shift_axis(H_m_m1, 1, axis=axis_l)
    )
    bgrad_s = bgrad[None, None, None, None, None, :]
    vth_s = vth[:, None, None, None, None, None]
    return -vth_s * bgrad_s * mirror_term


def _gx_curv_gradb(
    H: jnp.ndarray,
    *,
    tz: jnp.ndarray,
    omega_d_scale: jnp.ndarray,
    cv_d: jnp.ndarray,
    gb_d: jnp.ndarray,
    l: jnp.ndarray,
    m: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    axis_m = -4
    H_m_p2 = _shift_axis(H, 2, axis=axis_m)
    H_m_m2 = _shift_axis(H, -2, axis=axis_m)
    curv_term = (
        jnp.sqrt((m + 1.0) * (m + 2.0)) * H_m_p2
        + (2.0 * m + 1.0) * H
        + jnp.sqrt(m * (m - 1.0)) * H_m_m2
    )
    axis_l = -5
    gradb_term = (
        (l + 1.0) * _shift_axis(H, 1, axis=axis_l)
        + (2.0 * l + 1.0) * H
        + l * _shift_axis(H, -1, axis=axis_l)
    )
    icv = 1j * tz[:, None, None, None, None, None] * omega_d_scale * cv_d[None, None, None, ...]
    igb = 1j * tz[:, None, None, None, None, None] * omega_d_scale * gb_d[None, None, None, ...]
    return -icv * curv_term, -igb * gradb_term


def _gx_diamagnetic(
    dG: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    l4: jnp.ndarray,
    tprim: jnp.ndarray,
    fprim: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    omega_star_scale: jnp.ndarray,
    ky: jnp.ndarray,
) -> jnp.ndarray:
    Nm = dG.shape[2]
    Jl_m1 = _shift_axis(Jl, -1, axis=1)
    Jl_p1 = _shift_axis(Jl, 1, axis=1)
    JlB_m1 = _shift_axis(JlB, -1, axis=1)
    JlB_p1 = _shift_axis(JlB, 1, axis=1)
    omega_star = 1j * omega_star_scale * ky
    tprim_s = tprim[:, None, None, None, None]
    fprim_s = fprim[:, None, None, None, None]
    tz_s = tz[:, None, None, None, None]
    omega_star_s = omega_star[None, None, :, None, None]
    omega_star_bpar = omega_star_s / tz_s
    drive_m0 = omega_star_s * phi * (
        Jl_m1 * (l4 * tprim_s)
        + Jl * (fprim_s + 2.0 * l4 * tprim_s)
        + Jl_p1 * ((l4 + 1.0) * tprim_s)
    )
    drive_m0 = drive_m0 + omega_star_bpar * bpar * (
        JlB_m1 * (l4 * tprim_s)
        + JlB * (fprim_s + 2.0 * l4 * tprim_s)
        + JlB_p1 * ((l4 + 1.0) * tprim_s)
    )
    dG = dG.at[:, :, 0, ...].add(drive_m0)
    if Nm > 2:
        drive_m2 = omega_star_s * phi * Jl * (tprim_s / jnp.sqrt(2.0))
        drive_m2 = drive_m2 + omega_star_bpar * bpar * JlB * (tprim_s / jnp.sqrt(2.0))
        dG = dG.at[:, :, 2, ...].add(drive_m2)
    if Nm > 1:
        vth_s = vth[:, None, None, None, None]
        apar_drive = -vth_s * omega_star_s * apar * (
            Jl_m1 * (l4 * tprim_s)
            + Jl * (fprim_s + (2.0 * l4 + 1.0) * tprim_s)
            + Jl_p1 * ((l4 + 1.0) * tprim_s)
        )
        dG = dG.at[:, :, 1, ...].add(apar_drive)
    if Nm > 3:
        vth_s = vth[:, None, None, None, None]
        drive_m3 = -vth_s * omega_star_s * apar * Jl * (tprim_s * jnp.sqrt(3.0 / 2.0))
        dG = dG.at[:, :, 3, ...].add(drive_m3)
    return dG


def _norm(x: jnp.ndarray) -> float:
    return float(jnp.linalg.norm(jnp.ravel(x)))


def _rel_err(a: jnp.ndarray, b: jnp.ndarray) -> float:
    denom = max(_norm(b), 1.0e-14)
    return float(_norm(a - b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare RHS terms with GX formula replicas.")
    parser.add_argument("--ky", type=float, default=0.05)
    parser.add_argument("--Nl", type=int, default=6)
    parser.add_argument("--Nm", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        R_over_Ln=cfg.model.R_over_Ln,
        R_over_LTi=cfg.model.R_over_LTi,
        R_over_LTe=cfg.model.R_over_LTe,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    params = replace(
        params,
        nu_hyper=0.0,
        nu_hyper_l=0.0,
        nu_hyper_m=1.0,
        p_hyper_l=6.0,
        p_hyper_m=20.0,
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)

    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - args.ky)))
    G = jnp.zeros((args.Nl, args.Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    rng = np.random.default_rng(args.seed)
    g_slice = rng.normal(size=(args.Nl, args.Nm, grid.z.size)) + 1j * rng.normal(
        size=(args.Nl, args.Nm, grid.z.size)
    )
    G = G.at[:, :, ky_index, 0, :].set(jnp.asarray(g_slice, dtype=jnp.complex64))
    G = G[None, ...]

    ns = 1
    real_dtype = jnp.float32
    charge = jnp.ones((ns,), dtype=real_dtype)
    density = jnp.ones((ns,), dtype=real_dtype)
    mass = jnp.ones((ns,), dtype=real_dtype)
    temp = jnp.ones((ns,), dtype=real_dtype)
    tz = jnp.ones((ns,), dtype=real_dtype)
    vth = jnp.ones((ns,), dtype=real_dtype)
    tprim = jnp.asarray([cfg.model.R_over_LTi], dtype=real_dtype)
    fprim = jnp.asarray([cfg.model.R_over_Ln], dtype=real_dtype)

    fields = _solve_fields_impl(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=jnp.asarray(0.0, dtype=real_dtype),
        w_bpar=jnp.asarray(0.0, dtype=real_dtype),
    )
    Jl = cache.Jl.astype(real_dtype)
    JlB = Jl + _shift_axis(Jl, -1, axis=1)
    apar = fields.apar if fields.apar is not None else jnp.zeros_like(fields.phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(fields.phi)
    H = _gx_build_H(G, Jl, fields.phi, tz, apar=apar, vth=vth, bpar=bpar, JlB=JlB)

    gx_stream = _gx_streaming(
        H,
        dz=cache.dz.astype(real_dtype),
        vth=vth,
        kpar_scale=jnp.asarray(params.kpar_scale, dtype=real_dtype),
        use_twist_shift=cache.use_twist_shift,
        kx_link_plus=cache.kx_link_plus,
        kx_link_minus=cache.kx_link_minus,
        kx_mask_plus=cache.kx_link_mask_plus,
        kx_mask_minus=cache.kx_link_mask_minus,
    )
    gx_mirror = _gx_mirror(
        H,
        vth=vth,
        bgrad=cache.bgrad.astype(real_dtype),
        l=cache.l.astype(real_dtype),
        sqrt_m=cache.sqrt_m.astype(real_dtype),
        sqrt_m_p1=cache.sqrt_m_p1.astype(real_dtype),
    )
    gx_curv, gx_gradb = _gx_curv_gradb(
        H,
        tz=tz,
        omega_d_scale=jnp.asarray(params.omega_d_scale, dtype=real_dtype),
        cv_d=cache.cv_d.astype(real_dtype),
        gb_d=cache.gb_d.astype(real_dtype),
        l=cache.l.astype(real_dtype),
        m=cache.m.astype(real_dtype),
    )
    gx_dia = _gx_diamagnetic(
        jnp.zeros_like(G),
        phi=fields.phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        l4=cache.l4.astype(real_dtype),
        tprim=tprim,
        fprim=fprim,
        tz=tz,
        vth=vth,
        omega_star_scale=jnp.asarray(params.omega_star_scale, dtype=real_dtype),
        ky=cache.ky.astype(real_dtype),
    )

    our_terms = rhs_assembly.assemble_rhs_cached(
        G,
        cache,
        params,
        terms=TermConfig(
            streaming=1.0,
            mirror=1.0,
            curvature=1.0,
            gradb=1.0,
            diamagnetic=1.0,
            collisions=0.0,
            hypercollisions=0.0,
            end_damping=0.0,
            apar=0.0,
            bpar=0.0,
            nonlinear=0.0,
        ),
        use_custom_vjp=False,
    )[0]
    our_stream = streaming_contribution(
        H,
        kz=cache.kz.astype(real_dtype),
        dz=cache.dz.astype(real_dtype),
        vth=vth,
        sqrt_p=cache.sqrt_p.astype(real_dtype),
        sqrt_m=cache.sqrt_m_ladder.astype(real_dtype),
        kpar_scale=jnp.asarray(params.kpar_scale, dtype=real_dtype),
        weight=jnp.asarray(1.0, dtype=real_dtype),
        kx_link_plus=cache.kx_link_plus,
        kx_link_minus=cache.kx_link_minus,
        kx_mask_plus=cache.kx_link_mask_plus,
        kx_mask_minus=cache.kx_link_mask_minus,
        use_twist_shift=cache.use_twist_shift,
    )
    our_mirror = mirror_contribution(
        H,
        vth=vth,
        bgrad=cache.bgrad.astype(real_dtype),
        l=cache.l.astype(real_dtype),
        sqrt_m=cache.sqrt_m.astype(real_dtype),
        sqrt_m_p1=cache.sqrt_m_p1.astype(real_dtype),
        weight=jnp.asarray(1.0, dtype=real_dtype),
    )
    our_curv_gradb = curvature_gradb_contribution(
        H,
        tz=tz,
        omega_d_scale=jnp.asarray(params.omega_d_scale, dtype=real_dtype),
        cv_d=cache.cv_d.astype(real_dtype),
        gb_d=cache.gb_d.astype(real_dtype),
        l=cache.l.astype(real_dtype),
        m=cache.m.astype(real_dtype),
        imag=jnp.asarray(1j, dtype=jnp.complex64),
        weight_curv=jnp.asarray(1.0, dtype=real_dtype),
        weight_gradb=jnp.asarray(1.0, dtype=real_dtype),
    )
    our_dia = diamagnetic_contribution(
        jnp.zeros_like(G),
        phi=fields.phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        l4=cache.l4.astype(real_dtype),
        tprim=tprim,
        fprim=fprim,
        tz=tz,
        vth=vth,
        omega_star_scale=jnp.asarray(params.omega_star_scale, dtype=real_dtype),
        ky=cache.ky.astype(real_dtype),
        imag=jnp.asarray(1j, dtype=jnp.complex64),
        weight=jnp.asarray(1.0, dtype=real_dtype),
    )

    comparisons: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]] = {
        "streaming": (our_stream, gx_stream),
        "mirror": (our_mirror, gx_mirror),
        "curv+gradB": (our_curv_gradb, gx_curv + gx_gradb),
        "diamagnetic": (our_dia, gx_dia),
        "assembled": (our_terms, gx_stream + gx_mirror + gx_curv + gx_gradb + gx_dia),
    }

    print("RHS term comparison (relative L2 error vs GX replicas):")
    for name, (ours, gx) in comparisons.items():
        print(f"{name:>12}: {_rel_err(ours, gx):.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
