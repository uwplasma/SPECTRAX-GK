#!/usr/bin/env python3
"""Compare GX RHS term dumps against SPECTRAX term contributions."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import jax.numpy as jnp
import jax
from netCDF4 import Dataset

from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    CycloneBaseCase,
    KBMBaseCase,
    _apply_gx_hypercollisions,
    _build_initial_condition,
    _two_species_params,
)
from spectraxgk.config import GridConfig
from spectraxgk.geometry import SAlphaGeometry, apply_gx_geometry_grid_defaults
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    _as_species_array,
    build_H,
    build_linear_cache,
    linear_terms_to_term_config,
)
from spectraxgk.runtime import build_runtime_geometry, build_runtime_linear_params, build_runtime_term_config
from spectraxgk.terms.linear_terms import (
    collisions_contribution,
    curvature_gradb_contribution,
    diamagnetic_contribution,
    mirror_contribution,
    streaming_contribution_gx,
)
from spectraxgk.terms.assembly import assemble_rhs_terms_cached, compute_fields_cached
from spectraxgk.terms.config import TermConfig


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


def _load_field(path: Path, nyc: int, nx: int, nz: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.complex64)
    expected = nyc * nx * nz
    if raw.size != expected:
        raise ValueError(f"{path} size {raw.size} does not match expected {expected}")
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    return raw[idxyz.ravel()].reshape(nyc, nx, nz)


def _cast_cache(cache: Any, *, real_dtype: jnp.dtype, complex_dtype: jnp.dtype) -> Any:
    def _cast(x):
        if isinstance(x, jnp.ndarray):
            if jnp.issubdtype(x.dtype, jnp.complexfloating):
                return x.astype(complex_dtype)
            if jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(real_dtype)
        return x

    return jax.tree_util.tree_map(_cast, cache)


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


def _infer_y0(ky: np.ndarray) -> float:
    ky_pos = np.asarray(ky, dtype=float)
    ky_pos = ky_pos[ky_pos > 0.0]
    if ky_pos.size == 0:
        raise ValueError("Need at least one positive ky value to infer y0")
    return float(1.0 / np.min(ky_pos))


def _manual_linear_contributions_from_fields(
    G: jnp.ndarray,
    cache: Any,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    phi: np.ndarray | jnp.ndarray,
    apar: np.ndarray | jnp.ndarray,
    bpar: np.ndarray | jnp.ndarray,
):
    """Assemble linear term contributions using externally supplied fields.

    GX term dumps are most useful when SPECTRAX evaluates the operator on the
    exact same ``G, phi, apar, bpar`` state rather than on a recomputed field
    solve. Keep this helper close to the comparison tool so tests can lock its
    argument/shape contract to the current linear-term APIs.
    """

    G_arr = jnp.asarray(G)
    out_dtype = jnp.result_type(G_arr, jnp.complex64)
    G_arr = jnp.asarray(G_arr, dtype=out_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype
    imag = jnp.asarray(1j, dtype=out_dtype)

    ns = int(G_arr.shape[0]) if G_arr.ndim == 6 else 1
    charge = _as_species_array(params.charge_sign, ns, "charge_sign").astype(real_dtype)
    density = _as_species_array(params.density, ns, "density").astype(real_dtype)
    mass = _as_species_array(params.mass, ns, "mass").astype(real_dtype)
    temp = _as_species_array(params.temp, ns, "temp").astype(real_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tprim = _as_species_array(params.R_over_LTi, ns, "R_over_LTi").astype(real_dtype)
    fprim = _as_species_array(params.R_over_Ln, ns, "R_over_Ln").astype(real_dtype)
    nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)

    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    omega_star_scale = jnp.asarray(params.omega_star_scale, dtype=real_dtype)
    kpar_scale = jnp.asarray(params.kpar_scale, dtype=real_dtype)

    w_stream = jnp.asarray(term_cfg.streaming, dtype=real_dtype)
    w_mirror = jnp.asarray(term_cfg.mirror, dtype=real_dtype)
    w_curv = jnp.asarray(term_cfg.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(term_cfg.gradb, dtype=real_dtype)
    w_dia = jnp.asarray(term_cfg.diamagnetic, dtype=real_dtype)
    w_coll = jnp.asarray(term_cfg.collisions, dtype=real_dtype)

    Jl = jnp.asarray(cache.Jl, dtype=real_dtype)
    JlB = jnp.asarray(cache.JlB, dtype=real_dtype)
    phi_j = jnp.asarray(phi, dtype=out_dtype)
    apar_j = jnp.asarray(apar, dtype=out_dtype)
    bpar_j = jnp.asarray(bpar, dtype=out_dtype)

    H = build_H(G_arr, Jl, phi_j, tz, apar=apar_j, vth=vth, bpar=bpar_j, JlB=JlB)
    zero = jnp.zeros_like(G_arr)

    contrib: dict[str, jnp.ndarray] = {}
    contrib["streaming"] = streaming_contribution_gx(
        G_arr,
        phi=phi_j,
        apar=apar_j,
        bpar=bpar_j,
        Jl=Jl,
        JlB=JlB,
        tz=tz,
        kz=cache.kz,
        dz=cache.dz,
        vth=vth,
        sqrt_p=cache.sqrt_p,
        sqrt_m=cache.sqrt_m_ladder,
        kpar_scale=kpar_scale,
        weight=w_stream,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
        linked_inverse_permutation=cache.linked_inverse_permutation,
        linked_full_cover=cache.linked_full_cover,
        linked_gather_map=cache.linked_gather_map,
        linked_gather_mask=cache.linked_gather_mask,
        linked_use_gather=cache.linked_use_gather,
        use_twist_shift=cache.use_twist_shift,
    )
    contrib["mirror"] = mirror_contribution(
        H,
        vth=vth,
        bgrad=cache.bgrad,
        l=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        weight=w_mirror,
    )
    contrib["curvature"] = curvature_gradb_contribution(
        H,
        tz=tz,
        omega_d_scale=omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        l=cache.l,
        m=cache.m,
        imag=imag,
        weight_curv=w_curv,
        weight_gradb=jnp.asarray(0.0, dtype=real_dtype),
    )
    contrib["gradb"] = curvature_gradb_contribution(
        H,
        tz=tz,
        omega_d_scale=omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        l=cache.l,
        m=cache.m,
        imag=imag,
        weight_curv=jnp.asarray(0.0, dtype=real_dtype),
        weight_gradb=w_gradb,
    )
    contrib["diamagnetic"] = diamagnetic_contribution(
        zero,
        phi=phi_j,
        apar=apar_j,
        bpar=bpar_j,
        Jl=Jl,
        JlB=JlB,
        l4=cache.l4,
        tprim=tprim,
        fprim=fprim,
        tz=tz,
        vth=vth,
        omega_star_scale=omega_star_scale,
        ky=cache.ky,
        imag=imag,
        weight=w_dia,
    )
    contrib["collisions"] = collisions_contribution(
        H,
        G=G_arr,
        Jl=Jl,
        JlB=JlB,
        b=cache.b,
        nu=nu,
        lb_lam=cache.lb_lam,
        weight=w_coll,
    )
    fields = compute_fields_cached(G_arr, cache, params, terms=term_cfg, use_custom_vjp=False)
    return fields, contrib


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx-dir", type=Path, required=True, help="Directory with rhs_stream.bin, rhs_linear.bin")
    parser.add_argument("--gx-out", type=Path, required=True, help="GX .out.nc file to map ky indices")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Runtime TOML config. When set, geometry/physics come from the runtime path instead of --case.",
    )
    parser.add_argument("--case", type=str, default="cyclone", choices=("cyclone", "kbm"))
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--Nl", type=int, default=None, help="Laguerre resolution (defaults to dump metadata)")
    parser.add_argument("--Nm", type=int, default=None, help="Hermite resolution (defaults to dump metadata)")
    parser.add_argument("--Ny", type=int, default=24)
    parser.add_argument("--Nz", type=int, default=96)
    parser.add_argument("--y0", type=float, default=None, help="Perpendicular box parameter (defaults to GX ky grid)")
    parser.add_argument("--ntheta", type=int, default=None)
    parser.add_argument("--nperiod", type=int, default=None)
    return parser


def _build_runtime_compare_context(
    config_path: Path,
    *,
    nx: int,
    ny_full: int,
    nz: int,
    nm: int,
    ky_vals: np.ndarray,
    y0_override: float | None,
):
    cfg, _data = load_runtime_from_toml(config_path)
    y0_use = float(y0_override) if y0_override is not None else _infer_y0(ky_vals)
    cfg_use = replace(
        cfg,
        grid=replace(
            cfg.grid,
            Nx=int(nx),
            Ny=int(ny_full),
            Nz=int(nz),
            y0=float(y0_use),
        ),
    )
    geom = build_runtime_geometry(cfg_use)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg_use.grid)
    grid_full = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg_use, Nm=nm, geom=geom)
    term_cfg = replace(build_runtime_term_config(cfg_use), hypercollisions=0.0, end_damping=0.0)
    return cfg_use, geom, grid_full, params, term_cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    shape_path = args.gx_dir / "rhs_terms_shape.txt"
    stream_path = args.gx_dir / "rhs_stream.bin"
    linear_path = args.gx_dir / "rhs_linear.bin"
    if not shape_path.exists() or not stream_path.exists() or not linear_path.exists():
        raise FileNotFoundError("Missing GX rhs dump files")

    shape = _load_shape(shape_path)
    nspec = shape.get("nspec", 1)
    nl = shape.get("nl", int(args.Nl) if args.Nl is not None else 1)
    nm = shape.get("nm", int(args.Nm) if args.Nm is not None else 1)
    Nl_use = int(args.Nl) if args.Nl is not None else int(nl)
    Nm_use = int(args.Nm) if args.Nm is not None else int(nm)
    nyc = shape.get("nyc", args.Ny // 2 + 1)
    nx = shape.get("nx", 1)
    nz = shape.get("nz", args.Nz)
    gx_shape = (nspec, nl, nm, nyc, nx, nz)

    gx_stream = _reshape_gx(_load_bin(stream_path, gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_linear = _reshape_gx(_load_bin(linear_path, gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_mirror = _reshape_gx(_load_bin(args.gx_dir / "rhs_mirror.bin", gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_curv = _reshape_gx(_load_bin(args.gx_dir / "rhs_curv.bin", gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_gradb = _reshape_gx(_load_bin(args.gx_dir / "rhs_gradb.bin", gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_dia = _reshape_gx(_load_bin(args.gx_dir / "rhs_diamagnetic.bin", gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_coll = _reshape_gx(_load_bin(args.gx_dir / "rhs_collisions.bin", gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
    gx_g_path = args.gx_dir / "g_state.bin"

    with Dataset(args.gx_out, "r") as root:
        ky_vals = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)
    y0_use = float(args.y0) if args.y0 is not None else _infer_y0(ky_vals)
    ky_idx = int(np.argmin(np.abs(ky_vals - float(args.ky))))
    gx_stream = gx_stream[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_linear = gx_linear[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_mirror = gx_mirror[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_curv = gx_curv[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_gradb = gx_gradb[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_dia = gx_dia[:, :, :, ky_idx : ky_idx + 1, :, :]
    gx_coll = gx_coll[:, :, :, ky_idx : ky_idx + 1, :, :]

    cfg: Any
    if args.config is not None:
        cfg, geom, grid_full, params, term_cfg = _build_runtime_compare_context(
            args.config,
            nx=nx,
            ny_full=int(2 * (nyc - 1)),
            nz=nz,
            nm=Nm_use,
            ky_vals=ky_vals,
            y0_override=args.y0,
        )
        params = params
        init_species_index = 0
    elif args.case == "cyclone":
        cfg = CycloneBaseCase(
            grid=GridConfig(
                Nx=nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=62.8,
                Ly=62.8,
                boundary="linked",
                y0=y0_use,
                ntheta=None,
                nperiod=None,
            )
        )
        geom = SAlphaGeometry.from_config(cfg.geometry)
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
        params = _apply_gx_hypercollisions(params, nhermite=Nm_use)
        init_species_index = 0
        grid_full = build_spectral_grid(cfg.grid)
        term_cfg = TermConfig(hypercollisions=0.0, end_damping=0.0)
    else:
        ntheta = args.ntheta if args.ntheta is not None else 32
        nperiod = args.nperiod if args.nperiod is not None else 2
        cfg = KBMBaseCase(
            grid=GridConfig(
                Nx=nx,
                Ny=args.Ny,
                Nz=args.Nz,
                Lx=62.8,
                Ly=62.8,
                boundary="linked",
                y0=y0_use,
                ntheta=ntheta,
                nperiod=nperiod,
            )
        )
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            nhermite=Nm_use,
        )
        init_species_index = 0
        grid_full = build_spectral_grid(cfg.grid)
        term_cfg = TermConfig(hypercollisions=0.0, end_damping=0.0, bpar=0.0)

    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(args.ky))))
    grid = select_ky_grid(grid_full, ky_index)
    cache = build_linear_cache(grid, geom, params, Nl_use, Nm_use)
    cache = _cast_cache(cache, real_dtype=jnp.float32, complex_dtype=jnp.complex64)
    if gx_g_path.exists():
        gx_g = _reshape_gx(_load_bin(gx_g_path, gx_shape), nspec=nspec, nl=nl, nm=nm, nyc=nyc, nx=nx, nz=nz)
        gx_g = gx_g[:, :, :, ky_idx : ky_idx + 1, :, :]
        G0 = jnp.asarray(gx_g.astype(np.complex64))
    else:
        G0 = _build_initial_condition(
            grid,
            geom,
            ky_index=0,
            kx_index=0,
            Nl=Nl_use,
            Nm=Nm_use,
            init_cfg=cfg.init,
        )
    phi_path = args.gx_dir / "phi.bin"
    if phi_path.exists():
        apar_path = args.gx_dir / "apar.bin"
        bpar_path = args.gx_dir / "bpar.bin"
        phi = _load_field(phi_path, nyc, nx, nz)[ky_idx : ky_idx + 1, :, :]
        apar = _load_field(apar_path, nyc, nx, nz)[ky_idx : ky_idx + 1, :, :]
        bpar = _load_field(bpar_path, nyc, nx, nz)[ky_idx : ky_idx + 1, :, :]

        G = G0 if G0.ndim == 6 else G0[None, ...]
        fields, contrib = _manual_linear_contributions_from_fields(
            jnp.asarray(G),
            cache,
            params,
            term_cfg,
            phi=phi,
            apar=apar,
            bpar=bpar,
        )
    else:
        _rhs_total, fields, contrib = assemble_rhs_terms_cached(G0, cache, params, terms=term_cfg)

    def _with_species(arr: jnp.ndarray | np.ndarray) -> np.ndarray:
        arr_np = np.asarray(arr)
        if arr_np.ndim == 5:
            return arr_np[None, ...]
        return arr_np

    spectrax_stream = _with_species(contrib["streaming"])
    spectrax_mirror = _with_species(contrib["mirror"])
    spectrax_curv = _with_species(contrib["curvature"])
    spectrax_gradb = _with_species(contrib["gradb"])
    spectrax_dia = _with_species(contrib["diamagnetic"])
    spectrax_coll = _with_species(contrib["collisions"])

    spectrax_linear = (
        spectrax_mirror + spectrax_curv + spectrax_gradb + spectrax_dia + spectrax_coll
    )

    _summary("streaming", gx_stream, spectrax_stream)
    _summary("mirror", gx_mirror, spectrax_mirror)
    _summary("curvature", gx_curv, spectrax_curv)
    _summary("gradb", gx_gradb, spectrax_gradb)
    _summary("diamag", gx_dia, spectrax_dia)
    _summary("collisions", gx_coll, spectrax_coll)
    _summary("linear_sum", gx_linear, spectrax_linear)
    if phi_path.exists():
        spectrax_phi = np.asarray(fields.phi)
        spectrax_apar = np.asarray(fields.apar) if fields.apar is not None else None
        spectrax_bpar = np.asarray(fields.bpar) if fields.bpar is not None else None
        _summary("phi", phi, spectrax_phi)
        if spectrax_apar is not None:
            _summary("apar", apar, spectrax_apar)
        if spectrax_bpar is not None:
            _summary("bpar", bpar, spectrax_bpar)


if __name__ == "__main__":
    main()
