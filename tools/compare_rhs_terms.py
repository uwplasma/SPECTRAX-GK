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
from spectraxgk.linear import LinearParams, build_H, build_linear_cache
from spectraxgk.terms import assembly as rhs_assembly
from spectraxgk.terms.config import TermConfig
from spectraxgk.terms.fields import _solve_fields_impl
from spectraxgk.terms.linear_terms import (
    curvature_gradb_contribution,
    diamagnetic_contribution,
    mirror_contribution,
    streaming_contribution_gx,
)
from spectraxgk.terms.operators import grad_z_linked_fft, grad_z_periodic


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
    zt = jnp.where(tz == 0.0, 0.0, 1.0 / tz)
    H = G
    H = H.at[:, :, 0, ...].add(zt[:, None, None, None, None] * Jl * phi)
    H = H.at[:, :, 0, ...].add(JlB * bpar)
    H = H.at[:, :, 1, ...].add(
        -zt[:, None, None, None, None] * vth[:, None, None, None, None] * Jl * apar
    )
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
    linked_indices: tuple[jnp.ndarray, ...] | None = None,
    linked_kz: tuple[jnp.ndarray, ...] | None = None,
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
        if linked_indices is not None and linked_kz is not None:
            rhs = grad_z_linked_fft(
                rhs, dz=dz, linked_indices=linked_indices, linked_kz=linked_kz
            )
        else:
            if kx_link_plus is None or kx_link_minus is None or kx_mask_plus is None or kx_mask_minus is None:
                raise ValueError("kx_link arrays must be provided for twist-shift comparison")
            from spectraxgk.terms.operators import _grad_z_linked_fd

            rhs = _grad_z_linked_fd(
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
    omega_star_bpar = omega_star_s * tz_s
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


def _gx_geometry_arrays(geom: SAlphaGeometry, theta: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Compute GX-style s-alpha geometry arrays for comparison."""

    shear = geom.s_hat * theta - geom.alpha * jnp.sin(theta)
    gds2 = 1.0 + shear * shear
    gds21 = -geom.s_hat * shear
    gds22 = jnp.asarray(geom.s_hat) * jnp.asarray(geom.s_hat)
    bmag = 1.0 / (1.0 + geom.epsilon * jnp.cos(theta))
    bgrad = geom.gradpar() * geom.epsilon * jnp.sin(theta) * bmag
    base = jnp.cos(theta) + shear * jnp.sin(theta)
    cv = base / geom.R0
    gb = cv
    cv0 = (-geom.s_hat * jnp.sin(theta)) / geom.R0
    gb0 = cv0
    return {
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "bmag": bmag,
        "bgrad": bgrad,
        "cv": cv,
        "gb": gb,
        "cv0": cv0,
        "gb0": gb0,
    }


def _report_geom_diffs(cache: "LinearCache", geom: SAlphaGeometry, grid) -> None:
    theta = jnp.asarray(grid.z)
    gx = _gx_geometry_arrays(geom, theta)
    spec = {
        "bmag": cache.bmag,
        "bgrad": cache.bgrad,
    }
    cv, gb, cv0, gb0 = geom.drift_coeffs(theta)
    spec.update({"cv": cv, "gb": gb, "cv0": cv0, "gb0": gb0})
    for key in ["bmag", "bgrad", "cv", "gb", "cv0", "gb0"]:
        arr_spec = jnp.asarray(spec[key])
        arr_gx = jnp.asarray(gx[key])
        diff = jnp.abs(arr_spec - arr_gx)
        flat_idx = int(jnp.argmax(diff))
        idx = np.unravel_index(flat_idx, diff.shape)
        print(
            f"geom diff {key}: max|spec-gx|={float(diff[idx]):.3e} at {idx} "
            f"(spec={float(arr_spec[idx]):.6e}, gx={float(arr_gx[idx]):.6e})"
        )


def _report_max_diff(
    label: str,
    ours: jnp.ndarray,
    gx: jnp.ndarray,
    cache: "LinearCache",
    H: jnp.ndarray,
    H_gx: jnp.ndarray,
    field_label: str,
) -> None:
    diff = ours - gx
    absdiff = jnp.abs(diff)
    flat_idx = int(jnp.argmax(absdiff))
    idx = np.unravel_index(flat_idx, absdiff.shape)
    ours_val = np.asarray(ours)[idx]
    gx_val = np.asarray(gx)[idx]
    diff_val = np.asarray(diff)[idx]
    print(f"{label} max |Δ| at {field_label} index {idx}: ours={ours_val} gx={gx_val} diff={diff_val}")
    if len(idx) == 6:
        s, l, m, ky_i, kx_i, z_i = idx
        bgrad = float(np.asarray(cache.bgrad)[z_i])
        cv_d = float(np.asarray(cache.cv_d)[ky_i, kx_i, z_i])
        gb_d = float(np.asarray(cache.gb_d)[ky_i, kx_i, z_i])
        print(
            f"  coeffs: bgrad={bgrad:.6e} cv_d={cv_d:.6e} gb_d={gb_d:.6e} ky={float(cache.ky[ky_i]):.6e}"
        )
        H_val = np.asarray(H)[idx]
        H_gx_val = np.asarray(H_gx)[idx]
        print(f"  H_spec[{idx}]={H_val} H_gx[{idx}]={H_gx_val}")


def _report_shift_diffs(H: jnp.ndarray) -> None:
    from spectraxgk.terms.operators import shift_axis as shift_ops

    for axis, name in [(-4, "m"), (-5, "l")]:
        for offset in (-1, 1):
            ours = shift_ops(H, offset, axis=axis)
            ref = _shift_axis(H, offset, axis=axis)
            diff = float(jnp.max(jnp.abs(ours - ref)))
            print(f"shift_axis diff (axis {name}, offset {offset}): {diff:.3e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare RHS terms with GX formula replicas.")
    parser.add_argument("--ky", type=float, default=0.05)
    parser.add_argument("--Nl", type=int, default=6)
    parser.add_argument("--Nm", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action="store_true", help="Print max-diff indices and coefficients.")
    parser.add_argument("--geom-out", type=str, default="", help="Optional npz path for geometry arrays.")
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
    if args.debug:
        _report_geom_diffs(cache, geom, grid)
        if args.geom_out:
            theta = np.asarray(grid.z)
            gx = _gx_geometry_arrays(geom, jnp.asarray(grid.z))
            np.savez(
                args.geom_out,
                theta=theta,
                bmag=np.asarray(cache.bmag),
                bgrad=np.asarray(cache.bgrad),
                cv=np.asarray(geom.drift_coeffs(jnp.asarray(grid.z))[0]),
                gb=np.asarray(geom.drift_coeffs(jnp.asarray(grid.z))[1]),
                cv0=np.asarray(geom.drift_coeffs(jnp.asarray(grid.z))[2]),
                gb0=np.asarray(geom.drift_coeffs(jnp.asarray(grid.z))[3]),
                gx_bmag=np.asarray(gx["bmag"]),
                gx_bgrad=np.asarray(gx["bgrad"]),
                gx_cv=np.asarray(gx["cv"]),
                gx_gb=np.asarray(gx["gb"]),
                gx_cv0=np.asarray(gx["cv0"]),
                gx_gb0=np.asarray(gx["gb0"]),
            )

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
    H_gx = H
    H_ref = build_H(G, Jl, fields.phi, tz, apar=apar, vth=vth, bpar=bpar, JlB=JlB)
    if args.debug:
        _report_shift_diffs(H)
        hdiff = jnp.abs(H_ref - H_gx)
        hmax = float(jnp.max(hdiff))
        if hmax > 0.0:
            idx = np.unravel_index(int(jnp.argmax(hdiff)), hdiff.shape)
            print(f"H diff max|Δ|={hmax:.3e} at {idx} (spec={np.asarray(H_ref)[idx]}, gx={np.asarray(H_gx)[idx]})")

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
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
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
    our_stream = streaming_contribution_gx(
        G,
        phi=fields.phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        tz=tz,
        kz=cache.kz.astype(real_dtype),
        dz=cache.dz.astype(real_dtype),
        vth=vth,
        sqrt_p=cache.sqrt_p.astype(real_dtype),
        sqrt_m=cache.sqrt_m_ladder.astype(real_dtype),
        kpar_scale=jnp.asarray(params.kpar_scale, dtype=real_dtype),
        weight=jnp.asarray(1.0, dtype=real_dtype),
        use_twist_shift=cache.use_twist_shift,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
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
    if args.debug:
        _report_max_diff("mirror", our_mirror, gx_mirror, cache, H_ref, H_gx, "G")
        _report_max_diff("curv+gradB", our_curv_gradb, gx_curv + gx_gradb, cache, H_ref, H_gx, "G")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
