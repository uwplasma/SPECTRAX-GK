#!/usr/bin/env python3
"""Decompose build_linear_cache() into measured subphases."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

try:
    from tools._profiler_options import make_profile_options
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from _profiler_options import make_profile_options


@dataclass(frozen=True)
class CachePhaseTiming:
    phase: str
    seconds: float
    note: str = ""


def build_low_rank_moment_cache(
    *,
    nl: int,
    nm: int,
    params: Any,
    real_dtype: Any,
) -> dict[str, Any]:
    """Build the small moment-space factors used by ``build_linear_cache``.

    The production cache stores Lenard-Bernstein factors in low-rank form:
    ``lb_lam`` has shape ``(Nl, Nm)`` and collision expansion happens inside
    the RHS. The profiler must preserve that convention; otherwise it times an
    obsolete full ``(species, l, m, ky, kx, z)`` allocation and points
    optimization work at the wrong bottleneck.
    """

    import jax.numpy as jnp
    from spectraxgk.linear import _build_low_rank_moment_cache_arrays

    return {
        **_build_low_rank_moment_cache_arrays(nl, nm, params, real_dtype),
        "collision_lam": jnp.asarray([], dtype=real_dtype),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"),
    )
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--python-tracer-level", type=int, default=0)
    parser.add_argument("--host-tracer-level", type=int, default=0)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _block_tree(tree: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        try:
            jax.block_until_ready(leaf)
        except TypeError:
            continue


def _time_phase(
    timings: list[CachePhaseTiming],
    phase: str,
    fn: Callable[[], Any],
    *,
    note: str = "",
) -> Any:
    from jax import profiler

    t0 = time.perf_counter()
    with profiler.TraceAnnotation(phase):
        out = fn()
        _block_tree(out)
    timings.append(CachePhaseTiming(phase=phase, seconds=time.perf_counter() - t0, note=note))
    return out


def _write_csv(path: Path, timings: list[CachePhaseTiming]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["phase", "seconds", "note"])
        writer.writeheader()
        for row in timings:
            writer.writerow(asdict(row))


def _write_json(path: Path, timings: list[CachePhaseTiming], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "phases": [asdict(row) for row in timings],
                "total_s": sum(row.seconds for row in timings),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import profiler
    from spectraxgk.geometry import apply_gx_geometry_grid_defaults, ensure_flux_tube_geometry_data
    from spectraxgk.grids import build_spectral_grid
    from spectraxgk.gyroaverage import J_l_all, bessel_j0, bessel_j1, gx_laguerre_transform
    from spectraxgk.io import load_runtime_from_toml
    from spectraxgk.linear import (
        _build_end_damping_profile_array,
        _build_linked_end_damping_profile,
        _build_linked_fft_maps,
        _x64_enabled,
        build_linear_cache,
        shift_axis,
    )
    from spectraxgk.runtime import build_runtime_geometry, build_runtime_linear_params

    timings: list[CachePhaseTiming] = []

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        profiler.start_trace(
            str(args.trace_dir),
            profiler_options=make_profile_options(
                python_tracer_level=args.python_tracer_level,
                host_tracer_level=args.host_tracer_level,
            ),
        )

    try:
        cfg, _ = _time_phase(
            timings,
            "load_runtime_config",
            lambda: load_runtime_from_toml(args.config),
            note=str(args.config),
        )
        geom = _time_phase(timings, "build_runtime_geometry", lambda: build_runtime_geometry(cfg))
        grid_cfg = _time_phase(timings, "apply_geometry_grid_defaults", lambda: apply_gx_geometry_grid_defaults(geom, cfg.grid))
        grid = _time_phase(timings, "build_spectral_grid", lambda: build_spectral_grid(grid_cfg))
        params = _time_phase(timings, "build_runtime_linear_params", lambda: build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom))

        ctx: dict[str, Any] = {}

        def _grid_scalars():
            real_dtype = jnp.float64 if _x64_enabled() else jnp.float32
            dz = jnp.asarray(grid.z[1] - grid.z[0], dtype=real_dtype)
            kz = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(grid.z.size, d=dz), dtype=real_dtype)
            rho_star = jnp.asarray(params.rho_star, dtype=real_dtype)
            kx_raw = jnp.asarray(grid.kx, dtype=real_dtype)
            ky_raw = jnp.asarray(grid.ky, dtype=real_dtype)
            kx_eff = rho_star * kx_raw
            ky_eff = rho_star * ky_raw
            kx_grid = jnp.asarray(grid.kx_grid, dtype=real_dtype) * rho_star
            ky_grid = jnp.asarray(grid.ky_grid, dtype=real_dtype) * rho_star
            dealias_mask = jnp.asarray(grid.dealias_mask, dtype=bool)
            theta = jnp.asarray(grid.z, dtype=real_dtype)
            ctx.update(
                dict(
                    real_dtype=real_dtype,
                    dz=dz,
                    kz=kz,
                    rho_star=rho_star,
                    kx_raw=kx_raw,
                    ky_raw=ky_raw,
                    kx_eff=kx_eff,
                    ky_eff=ky_eff,
                    kx_grid=kx_grid,
                    ky_grid=ky_grid,
                    dealias_mask=dealias_mask,
                    theta=theta,
                    kxfac_val=float(getattr(grid, "kxfac", 1.0)),
                )
            )
            return dz, kz, kx_eff, ky_eff, kx_grid, ky_grid

        _time_phase(timings, "grid_scalars", _grid_scalars)

        def _geometry_coeffs():
            geom_data = ensure_flux_tube_geometry_data(geom, ctx["theta"])
            gds2, gds21, gds22 = geom_data.metric_coeffs(ctx["theta"])
            gds22_arr = gds22 if gds22.ndim else jnp.full_like(ctx["theta"], gds22)
            bmag = geom_data.bmag(ctx["theta"]).astype(ctx["real_dtype"])
            bgrad = geom_data.bgrad(ctx["theta"]).astype(ctx["real_dtype"])
            jacobian = geom_data.jacobian(ctx["theta"]).astype(ctx["real_dtype"])
            cv, gb, cv0, gb0 = geom_data.drift_coeffs(ctx["theta"])
            ctx.update(
                dict(
                    geom_data=geom_data,
                    gds2=gds2,
                    gds21=gds21,
                    gds22=gds22,
                    gds22_arr=gds22_arr,
                    bmag=bmag,
                    bgrad=bgrad,
                    jacobian=jacobian,
                    cv=cv,
                    gb=gb,
                    cv0=cv0,
                    gb0=gb0,
                )
            )
            return gds2, gds21, gds22_arr, bmag, jacobian

        _time_phase(timings, "geometry_coefficients", _geometry_coeffs)

        def _kperp_and_drifts():
            geom_data = ctx["geom_data"]
            boundary = str(getattr(grid, "boundary", "periodic")).lower()
            use_twist_shift = boundary in {"linked", "fix aspect", "continuous drifts"}
            use_ntft = bool(getattr(grid, "non_twist", False))
            y0 = getattr(grid, "y0", None)
            if y0 is None:
                y0 = float(1.0 / float(grid.ky[1] - grid.ky[0])) if grid.ky.size > 1 else 1.0
            shat = float(geom_data.s_hat)
            x0_eff = float(getattr(grid, "x0", 1.0))
            jtwist = 0
            x0_target = x0_eff
            kx_eff = ctx["kx_eff"]
            kx_grid = ctx["kx_grid"]
            if use_twist_shift:
                gds21_min = float(ctx["gds21"][0]) if ctx["gds21"].ndim else float(ctx["gds21"])
                gds22_min = float(ctx["gds22"][0]) if ctx["gds22"].ndim else float(ctx["gds22"])
                twist_shift_geo_fac = 0.0 if gds22_min == 0.0 else float(2.0 * shat * gds21_min / gds22_min)
                if twist_shift_geo_fac != 0.0:
                    jtwist = int(np.round(twist_shift_geo_fac))
                    if jtwist == 0:
                        jtwist = 1
                    x0_target = float(y0) * abs(jtwist) / abs(twist_shift_geo_fac)
                    if use_ntft:
                        x0_eff = x0_target
                else:
                    jtwist = 1
                if use_ntft and float(getattr(grid, "x0", x0_eff)) != 0.0:
                    kx_eff = kx_eff * (float(getattr(grid, "x0", x0_eff)) / float(x0_eff))
                if not use_ntft and x0_target != 0.0 and x0_target != x0_eff:
                    scale = float(x0_eff) / float(x0_target)
                    kx_eff = kx_eff * scale
                    kx_grid = kx_grid * scale
                    x0_eff = x0_target
            kperp2_bmag = bool(getattr(geom_data, "kperp2_bmag", True))
            if use_ntft:
                ftwist = (geom_data.s_hat * ctx["gds21"] / ctx["gds22_arr"]).astype(ctx["real_dtype"])
                delta = jnp.asarray(0.01313, dtype=ctx["real_dtype"])
                ftwist_next = jnp.roll(ftwist, -1)
                mid_idx = int(grid.z.size // 2)
                mid_next = (mid_idx + 1) % grid.z.size
                ftwist_mid = ftwist[mid_idx]
                ftwist_mid_next = ftwist[mid_next]
                m0 = -jnp.rint(
                    float(x0_eff)
                    * ctx["ky_raw"][:, None]
                    * ((1.0 - delta) * ftwist[None, :] + delta * ftwist_next[None, :])
                ) + jnp.rint(
                    float(x0_eff)
                    * ctx["ky_raw"][:, None]
                    * ((1.0 - delta) * ftwist_mid + delta * ftwist_mid_next)
                )
                m0 = m0.astype(ctx["real_dtype"])
                shat_inv = 1.0 / shat
                delta_kx = ctx["ky_eff"][:, None] * ftwist[None, :] + (ctx["rho_star"] * m0 / float(x0_eff))
                term_ky = ctx["ky_eff"][:, None, None] ** 2 * (
                    ctx["gds2"][None, None, :]
                    - 2.0 * ftwist[None, None, :] * ctx["gds21"][None, None, :] * shat_inv
                    + (ftwist[None, None, :] ** 2) * ctx["gds22_arr"][None, None, :] * shat_inv * shat_inv
                )
                term_kx = (kx_eff[None, :, None] + delta_kx[:, None, :]) ** 2 * ctx["gds22_arr"][None, None, :] * shat_inv * shat_inv
                bmag_inv = 1.0 / ctx["bmag"]
                kperp2 = term_ky + term_kx
                if kperp2_bmag:
                    kperp2 = kperp2 * (bmag_inv[None, None, :] ** 2)
                kx_shift = kx_eff[None, :, None] + (ctx["rho_star"] * m0 / float(x0_eff))[:, None, :]
                cv_d = ctx["ky_eff"][:, None, None] * ctx["cv"][None, None, :] + shat_inv * kx_shift * ctx["cv0"][None, None, :]
                gb_d = ctx["ky_eff"][:, None, None] * ctx["gb"][None, None, :] + shat_inv * kx_shift * ctx["gb0"][None, None, :]
                omega_d = cv_d + gb_d
            else:
                kx0 = kx_eff[None, :, None]
                ky0 = ctx["ky_eff"][:, None, None]
                theta_b = ctx["theta"][None, None, :]
                kperp2 = geom_data.k_perp2(kx0, ky0, theta_b).astype(ctx["real_dtype"])
                cv_d, gb_d = geom_data.drift_components(kx_eff, ctx["ky_eff"], ctx["theta"])
                cv_d = cv_d.astype(ctx["real_dtype"])
                gb_d = gb_d.astype(ctx["real_dtype"])
                omega_d = (cv_d + gb_d).astype(ctx["real_dtype"])
            apply_dealias_mask = ctx["dealias_mask"] is not None and int(grid.ky.size) > 1
            if apply_dealias_mask:
                mask = ctx["dealias_mask"][:, :, None]
                kperp2 = kperp2 * mask
                cv_d = cv_d * mask
                gb_d = gb_d * mask
                omega_d = omega_d * mask
            ctx.update(
                dict(
                    boundary=boundary,
                    use_twist_shift=use_twist_shift,
                    y0=float(y0),
                    jtwist=int(jtwist),
                    x0_eff=float(x0_eff),
                    kx_eff=kx_eff,
                    kx_grid=kx_grid,
                    kperp2_bmag=kperp2_bmag,
                    kperp2=kperp2,
                    cv_d=cv_d,
                    gb_d=gb_d,
                    omega_d=omega_d,
                )
            )
            return kperp2, omega_d

        _time_phase(timings, "kperp_and_drifts", _kperp_and_drifts)

        def _gyro_cache():
            rho = jnp.asarray(params.rho, dtype=ctx["real_dtype"])
            if rho.ndim == 0:
                rho = rho[None]
            b = (rho[:, None, None, None] * rho[:, None, None, None]) * ctx["kperp2"][None, ...]
            bessel_bmag_power = float(getattr(ctx["geom_data"], "bessel_bmag_power", 0.0))
            if bessel_bmag_power != 0.0:
                bmag_factor = ctx["bmag"][None, None, None, :] ** (-bessel_bmag_power)
                b = b * bmag_factor
            Jl = jax.vmap(lambda bs: J_l_all(bs, l_max=args.Nl - 1))(b).astype(ctx["real_dtype"])
            JlB = Jl + shift_axis(Jl, -1, axis=1)
            ctx.update(dict(rho=rho, b=b, Jl=Jl, JlB=JlB))
            return b, Jl, JlB

        _time_phase(timings, "gyro_bessel_cache", _gyro_cache)

        def _laguerre_cache():
            lag_to_grid_np, lag_to_spec_np, lag_roots_np = gx_laguerre_transform(args.Nl)
            laguerre_to_grid = jnp.asarray(lag_to_grid_np, dtype=ctx["real_dtype"])
            laguerre_to_spectral = jnp.asarray(lag_to_spec_np, dtype=ctx["real_dtype"])
            laguerre_roots = jnp.asarray(lag_roots_np, dtype=ctx["real_dtype"])
            alpha = jnp.sqrt(
                jnp.maximum(
                    0.0,
                    2.0 * laguerre_roots[None, :, None, None, None] * ctx["b"][:, None, ...],
                )
            )
            laguerre_j0 = bessel_j0(alpha).astype(ctx["real_dtype"])
            laguerre_j1 = bessel_j1(alpha)
            laguerre_j1_over_alpha = jnp.where(alpha < 1.0e-8, 0.5, laguerre_j1 / alpha).astype(ctx["real_dtype"])
            ctx.update(
                dict(
                    laguerre_to_grid=laguerre_to_grid,
                    laguerre_to_spectral=laguerre_to_spectral,
                    laguerre_roots=laguerre_roots,
                    laguerre_j0=laguerre_j0,
                    laguerre_j1_over_alpha=laguerre_j1_over_alpha,
                )
            )
            return laguerre_to_grid, laguerre_j0, laguerre_j1_over_alpha

        _time_phase(timings, "laguerre_cache", _laguerre_cache)

        def _collision_cache():
            mask0 = (grid.ky == 0.0)[:, None, None] & (grid.kx == 0.0)[None, :, None]
            moment = build_low_rank_moment_cache(
                nl=args.Nl,
                nm=args.Nm,
                params=params,
                real_dtype=ctx["real_dtype"],
            )
            damp_profile = _build_end_damping_profile_array(
                int(grid.z.size),
                float(params.damp_ends_widthfrac),
                str(ctx["boundary"]),
                ctx["real_dtype"],
            )
            ctx.update(
                dict(
                    mask0=mask0,
                    damp_profile=damp_profile,
                    **moment,
                )
            )
            return moment["lb_lam"], moment["collision_lam"], damp_profile

        _time_phase(timings, "collision_and_damping_cache", _collision_cache)

        def _linked_boundary_cache():
            use_twist_shift = bool(ctx["use_twist_shift"])
            linked_damp_profile = jnp.asarray([], dtype=ctx["real_dtype"])
            linked_indices = ()
            linked_kz = ()
            if use_twist_shift:
                ky_mode = getattr(grid, "ky_mode", None)
                linked_indices, linked_kz = _build_linked_fft_maps(
                    np.asarray(grid.kx),
                    np.asarray(grid.ky),
                    float(ctx["y0"]),
                    int(ctx["jtwist"]),
                    float(ctx["dz"]),
                    int(grid.z.size),
                    ctx["real_dtype"],
                    None if ky_mode is None else np.asarray(ky_mode),
                )
                if ctx["boundary"] != "periodic":
                    linked_damp_profile = jnp.asarray(
                        _build_linked_end_damping_profile(
                            linked_indices=linked_indices,
                            ny=int(grid.ky.size),
                            nx=int(grid.kx.size),
                            nz=int(grid.z.size),
                            widthfrac=float(params.damp_ends_widthfrac),
                            ky_mode=None if ky_mode is None else np.asarray(ky_mode, dtype=np.int32),
                        ),
                        dtype=ctx["real_dtype"],
                    )
            ctx.update(
                dict(
                    linked_indices=linked_indices,
                    linked_kz=linked_kz,
                    linked_damp_profile=linked_damp_profile,
                )
            )
            return linked_damp_profile

        _time_phase(timings, "linked_boundary_cache", _linked_boundary_cache)

        cache = _time_phase(
            timings,
            "full_build_linear_cache_reference",
            lambda: build_linear_cache(grid, geom, params, args.Nl, args.Nm),
            note="reference cold call after subphase decomposition",
        )

        metadata = {
            "config": str(args.config),
            "Nl": args.Nl,
            "Nm": args.Nm,
            "device_count": int(jax.device_count()),
            "kperp2_shape": tuple(int(x) for x in ctx["kperp2"].shape),
            "Jl_shape": tuple(int(x) for x in cache.Jl.shape),
        }
        for row in timings:
            note = f" {row.note}" if row.note else ""
            print(f"{row.phase}: seconds={row.seconds:.6f}{note}")
        print(f"total_s={sum(row.seconds for row in timings):.6f}")
        if args.csv_out is not None:
            _write_csv(args.csv_out, timings)
        if args.json_out is not None:
            _write_json(args.json_out, timings, metadata)
    finally:
        if args.trace_dir is not None:
            profiler.stop_trace()


if __name__ == "__main__":
    main()
