#!/usr/bin/env python3
"""Compare a GX linear run against SPECTRAX-GK using imported GX/VMEC geometry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from spectraxgk.benchmarks import _apply_gx_hypercollisions
from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, resolve_cfl_fac
from spectraxgk.geometry import SlabGeometry, apply_gx_geometry_grid_defaults, load_gx_geometry_netcdf
from spectraxgk.gyroaverage import gamma0
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.analysis import select_ky_index
from spectraxgk.gx_integrators import (
    GXTimeConfig,
    _gx_growth_rate_step,
    _gx_linear_omega_max,
    _gx_midplane_index,
    _gx_term_config,
    _linear_explicit_step,
    integrate_linear_gx_diagnostics,
)
from spectraxgk.io import load_toml
from spectraxgk.linear import LinearTerms, build_linear_cache
from spectraxgk.runtime import _build_initial_condition as _build_runtime_initial_condition
from spectraxgk.runtime_config import RuntimeConfig, RuntimeSpeciesConfig
from spectraxgk.species import Species, build_linear_params
from spectraxgk.terms.assembly import assemble_rhs_cached


@dataclass(frozen=True)
class GXInputContract:
    Nx: int
    Ny: int
    nperiod: int
    ntheta: int
    nlaguerre: int
    nhermite: int
    boundary: str
    geo_option: str
    s_hat: float
    zero_shat: bool
    y0: float
    fapar: float
    fbpar: float
    species: tuple[Species, ...]
    tau_e: float
    beta: float
    dt: float | None
    scheme: str
    nwrite: int
    init_field: str
    init_amp: float
    init_single: bool
    ikx_single: int
    iky_single: int
    gaussian_init: bool
    gaussian_width: float
    gaussian_envelope_constant: float
    gaussian_envelope_sine: float
    kpar_init: float
    random_seed: int
    init_electrons_only: bool
    random_init: bool
    hypercollisions: bool
    hyper: bool
    D_hyper: float
    damp_ends_amp: float
    damp_ends_widthfrac: float


def _file_cache_token(path: Path | None) -> dict[str, str | int | None]:
    if path is None:
        return {"path": None, "size": None, "mtime_ns": None}
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_gx_reference(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _read_ky_series(diag_group, name: str) -> np.ndarray:
        data = np.asarray(diag_group.variables[name][:], dtype=float)
        if data.ndim == 3:
            if data.shape[1] == 1 or name.startswith("Wapar_"):
                return np.asarray(data[:, 0, :], dtype=float)
            return np.asarray(np.sum(data, axis=1), dtype=float)
        raise ValueError(f"Unexpected shape for {name}: {data.shape}")

    root = Dataset(path, "r")
    try:
        grids = root.groups["Grids"]
        diag = root.groups["Diagnostics"]
        time = np.asarray(grids.variables["time"][:], dtype=float)
        ky = np.asarray(grids.variables["ky"][:], dtype=float)
        kx = np.asarray(grids.variables["kx"][:], dtype=float)
        omega = np.asarray(diag.variables["omega_kxkyt"][:], dtype=float)
        Wg = _read_ky_series(diag, "Wg_kyst")
        Wphi = _read_ky_series(diag, "Wphi_kyst")
        Wapar = _read_ky_series(diag, "Wapar_kyst")
    finally:
        root.close()
    return time, ky, kx, omega, Wg, Wphi, Wapar


def _read_gx_output_bool(path: Path, name: str, *, default: bool = False) -> bool:
    root = Dataset(path, "r")
    try:
        inputs = root.groups.get("Inputs")
        if inputs is None or name not in inputs.variables:
            return bool(default)
        return bool(np.asarray(inputs.variables[name][:]).item())
    finally:
        root.close()


def _infer_real_fft_ny(ky: np.ndarray) -> int:
    positive = ky[ky > 0.0]
    if positive.size == 0:
        raise ValueError("GX reference does not contain positive ky modes")
    # Invert GX's "dealiased positive ky count" convention.
    #
    # Most GX benchmark outputs store only the dealiased non-negative ky block.
    # To build a full FFT grid that:
    # - contains the requested smallest positive ky, and
    # - does not immediately mask it out under GX's strict 2/3 rule
    # we need Ny >= 4 even when only one positive ky is present.
    return max(4, int(3 * (positive.size - 1) + 1))


def _infer_y0(ky: np.ndarray) -> float:
    positive = ky[ky > 0.0]
    if positive.size == 0:
        raise ValueError("GX reference does not contain positive ky modes")
    return float(1.0 / np.min(positive))


def _resolve_imported_real_fft_ny(gx_ky: np.ndarray, gx_contract: GXInputContract | None) -> int:
    """Return the full real-FFT ``Ny`` implied by the stored GX ``ky`` grid."""

    inferred = _infer_real_fft_ny(gx_ky)
    if gx_contract is None:
        return inferred
    ny_input = int(gx_contract.Ny)
    if ny_input <= 0:
        return inferred
    # GX input files store the dealiased positive-ky count (`nky`), while the
    # imported SPECTRAX grid needs the full real-FFT layout represented in the
    # NetCDF `ky` coordinate.
    if ny_input == inferred:
        return inferred
    positive_ky = int(np.asarray(gx_ky)[np.asarray(gx_ky) > 0.0].size)
    if ny_input == positive_ky or ny_input == positive_ky + 1:
        return inferred
    return inferred


def _load_gx_input_contract(path: Path) -> GXInputContract:
    data = load_toml(path)
    dims = data.get("Dimensions", {})
    domain = data.get("Domain", {})
    physics = data.get("Physics", {})
    time = data.get("Time", {})
    init = data.get("Initialization", {})
    expert = data.get("Expert", {})
    geometry = data.get("Geometry", {})
    species_raw = data.get("species", {})
    boltz = data.get("Boltzmann", {})
    diagnostics = data.get("Diagnostics", {})
    dissipation = data.get("Dissipation", {})

    nspecies = int(dims.get("nspecies", 1))

    def _species_array(name: str, *, default: float) -> np.ndarray:
        raw = species_raw.get(name)
        if raw is None:
            return np.full(nspecies, default, dtype=float)
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            arr = np.full(nspecies, float(arr), dtype=float)
        if arr.size < nspecies:
            raise ValueError(f"{path} species.{name} has {arr.size} entries, expected at least {nspecies}")
        return np.asarray(arr[:nspecies], dtype=float)

    charge = _species_array("z", default=1.0)
    mass = _species_array("mass", default=1.0)
    dens = _species_array("dens", default=1.0)
    temp = _species_array("temp", default=1.0)
    tprim = _species_array("tprim", default=0.0)
    fprim = _species_array("fprim", default=0.0)
    nu = _species_array("vnewk", default=0.0)
    species = tuple(
        Species(
            charge=float(charge[i]),
            mass=float(mass[i]),
            density=float(dens[i]),
            temperature=float(temp[i]),
            tprim=float(tprim[i]),
            fprim=float(fprim[i]),
            nu=float(nu[i]),
        )
        for i in range(nspecies)
    )

    add_boltz = bool(boltz.get("add_Boltzmann_species", False))
    boltz_type = str(boltz.get("Boltzmann_type", "")).strip().lower()
    tau_e = float(boltz.get("tau_fac", 0.0)) if add_boltz and boltz_type == "electrons" else 0.0
    kpar_init = float(init["ikpar_init"]) if "ikpar_init" in init else float(init.get("kpar_init", 0.0))

    return GXInputContract(
        Nx=int(dims.get("nkx", dims.get("nx", 1))),
        Ny=int(dims.get("nky", dims.get("ny", 0))),
        nperiod=int(dims.get("nperiod", 1)),
        ntheta=int(dims.get("ntheta", 0)),
        nlaguerre=int(dims.get("nlaguerre", 8)),
        nhermite=int(dims.get("nhermite", 16)),
        boundary=str(domain.get("boundary", "linked")),
        geo_option=str(geometry.get("geo_option", "s-alpha")).strip().lower(),
        s_hat=float(geometry.get("shat", 0.0)),
        zero_shat=bool(geometry.get("zero_shat", False)),
        y0=float(domain["y0"]) if "y0" in domain else float("nan"),
        fapar=float(physics.get("fapar", 1.0 if float(physics.get("beta", 0.0)) > 0.0 else 0.0)),
        fbpar=float(physics.get("fbpar", 1.0 if float(physics.get("beta", 0.0)) > 0.0 else 0.0)),
        species=species,
        tau_e=tau_e,
        beta=float(physics.get("beta", 0.0)),
        dt=None if "dt" not in time else float(time["dt"]),
        scheme=str(time.get("scheme", "rk4")).strip().lower(),
        nwrite=max(1, int(diagnostics.get("nwrite", 1))),
        init_field=str(init.get("init_field", "density")).strip().lower(),
        init_amp=float(init.get("init_amp", 1.0e-5)),
        init_single=bool(expert.get("init_single", False)),
        ikx_single=int(expert.get("ikx_single", 0)),
        iky_single=int(expert.get("iky_single", 1)),
        gaussian_init=bool(init.get("gaussian_init", False)),
        gaussian_width=float(init.get("gaussian_width", 0.5)),
        gaussian_envelope_constant=float(init.get("gaussian_envelope_constant_coefficient", 1.0)),
        gaussian_envelope_sine=float(init.get("gaussian_envelope_sine_coefficient", 0.0)),
        kpar_init=kpar_init,
        random_seed=int(init.get("random_seed", 22)),
        init_electrons_only=bool(init.get("init_electrons_only", False)),
        random_init=bool(init.get("random_init", False)),
        hypercollisions=bool(dissipation.get("hypercollisions", False)),
        hyper=bool(dissipation.get("hyper", False)),
        D_hyper=float(dissipation.get("D_hyper", 0.0)),
        damp_ends_amp=float(dissipation.get("damp_ends_amp", 0.1)),
        damp_ends_widthfrac=float(dissipation.get("damp_ends_widthfrac", 1.0 / 8.0)),
    )


def _select_geometry_source(
    gx_out: Path,
    geometry_file: Path,
    gx_contract: GXInputContract | None,
) -> Path:
    """Choose the authoritative geometry source for imported GX comparisons."""

    if gx_contract is None:
        return geometry_file
    if gx_contract.geo_option in {"vmec", "desc"}:
        return gx_out
    return geometry_file


def _runtime_species_tuple(species: tuple[Species, ...]) -> tuple[RuntimeSpeciesConfig, ...]:
    return tuple(
        RuntimeSpeciesConfig(
            name=f"species_{idx}",
            charge=float(sp.charge),
            mass=float(sp.mass),
            density=float(sp.density),
            temperature=float(sp.temperature),
            tprim=float(sp.tprim),
            fprim=float(sp.fprim),
            nu=float(sp.nu),
            kinetic=True,
        )
        for idx, sp in enumerate(species)
    )


def _build_imported_initial_condition(
    *,
    grid,
    geom,
    gx_contract: GXInputContract | None,
    species: tuple[Species, ...],
    ky_index: int,
    kx_index: int,
    Nl: int,
    Nm: int,
):
    if gx_contract is None:
        init_cfg = InitializationConfig(
            gaussian_init=True,
            init_field="density",
            init_amp=1.0e-10,
        )
        seed_ky = ky_index
        seed_kx = kx_index
    else:
        if gx_contract.random_init:
            raise NotImplementedError(
                "Imported linear comparison does not yet support GX random_init=true startup"
            )
        init_cfg = InitializationConfig(
            init_field=gx_contract.init_field,
            init_amp=gx_contract.init_amp,
            init_single=gx_contract.init_single,
            random_seed=gx_contract.random_seed,
            gaussian_init=gx_contract.gaussian_init,
            gaussian_width=gx_contract.gaussian_width,
            gaussian_envelope_constant=gx_contract.gaussian_envelope_constant,
            gaussian_envelope_sine=gx_contract.gaussian_envelope_sine,
            kpar_init=gx_contract.kpar_init,
            init_electrons_only=gx_contract.init_electrons_only,
        )
        seed_ky = gx_contract.iky_single if gx_contract.init_single else ky_index
        seed_kx = gx_contract.ikx_single if gx_contract.init_single else kx_index
    cfg = RuntimeConfig(
        init=init_cfg,
        species=_runtime_species_tuple(species),
    )
    return _build_runtime_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=int(seed_ky),
        kx_index=int(seed_kx),
        Nl=Nl,
        Nm=Nm,
        nspecies=len(species),
    )


def _mean_rel_error(lhs: np.ndarray, rhs: np.ndarray, *, floor_fraction: float) -> float:
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    scale = max(float(np.nanmax(np.abs(rhs))), 1.0e-12)
    floor = floor_fraction * scale
    denom = np.maximum(np.abs(rhs), floor)
    return float(np.mean(np.abs(lhs - rhs) / denom))


def _match_local_kx_index(grid_kx: np.ndarray, gx_kx_value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(grid_kx, dtype=float) - float(gx_kx_value))))


def _select_gx_kx_index(gx_kx: np.ndarray, gx_contract: GXInputContract | None) -> int:
    gx_kx_arr = np.asarray(gx_kx, dtype=float)
    if gx_contract is not None and gx_contract.init_single:
        if 0 <= int(gx_contract.ikx_single) < int(gx_kx_arr.size):
            return int(gx_contract.ikx_single)
    return int(np.argmin(np.abs(gx_kx_arr)))


def _gx_fac_mask_cached(cache, *, use_dealias: bool) -> jnp.ndarray:
    ky = jnp.asarray(cache.ky)
    has_negative = jnp.any(ky < 0.0)
    fac = jnp.where(has_negative, 1.0, jnp.where(ky == 0.0, 1.0, 2.0))
    fac = fac[:, None] * jnp.ones((1, cache.kx.size), dtype=fac.dtype)
    if use_dealias:
        fac = fac * cache.dealias_mask.astype(fac.dtype)
    return fac


def _gx_kyst_fac_mask_cached(cache, *, use_dealias: bool) -> jnp.ndarray:
    """Return GX kyst fac*mask on a full SPECTRAX ky layout.

    GX stores ``*_kyst`` diagnostics on the positive-rFFT ky half only, while the
    evolved SPECTRAX state may carry the full ``±ky`` layout. To compare a full
    state against GX ``*_kyst`` data, keep the positive-ky rows with the Hermitian
    factor of 2, keep ky=0 with unit weight, and exclude ky<0 entirely.
    """

    ky = jnp.asarray(cache.ky)
    fac = jnp.where(ky > 0.0, 2.0, jnp.where(ky == 0.0, 1.0, 0.0))
    fac = fac[:, None] * jnp.ones((1, cache.kx.size), dtype=fac.dtype)
    if use_dealias:
        fac = fac * cache.dealias_mask.astype(fac.dtype)
    return fac


def _species_array(val: float | jnp.ndarray, ns: int) -> jnp.ndarray:
    arr = jnp.asarray(val)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (ns,))
    return arr


def _gx_Wg_by_ky(G: jnp.ndarray, cache, params, vol_fac: jnp.ndarray, *, use_dealias: bool = True) -> jnp.ndarray:
    Gs = G if G.ndim == 6 else G[None, ...]
    ns = Gs.shape[0]
    nt = _species_array(params.density, ns) * _species_array(params.temp, ns)
    fac = _gx_kyst_fac_mask_cached(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    contrib = 0.5 * (jnp.abs(Gs) ** 2) * nt[:, None, None, None, None, None]
    contrib = contrib * weight[None, None, None, :, :, :]
    return jnp.sum(contrib, axis=(0, 1, 2, 4, 5))


def _gx_Wphi_by_ky(phi: jnp.ndarray, cache, params, vol_fac: jnp.ndarray, *, use_dealias: bool = True) -> jnp.ndarray:
    fac = _gx_kyst_fac_mask_cached(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    rho = jnp.asarray(params.rho)
    if rho.ndim == 0:
        rho = rho[None]
    phi2 = jnp.abs(phi) ** 2
    wphi = jnp.zeros((phi.shape[0],), dtype=jnp.real(phi).dtype)
    for rho_s in rho:
        b = cache.kperp2 * (rho_s * rho_s)
        contrib = 0.5 * phi2 * (1.0 - gamma0(b)) * weight
        wphi = wphi + jnp.sum(contrib, axis=(1, 2))
    return wphi


def _gx_Wapar_by_ky(apar: jnp.ndarray, cache, vol_fac: jnp.ndarray, *, use_dealias: bool = True) -> jnp.ndarray:
    fac = _gx_kyst_fac_mask_cached(cache, use_dealias=use_dealias)
    weight = fac[:, :, None] * vol_fac[None, None, :]
    bmag2 = cache.bmag[None, None, :] ** 2 if cache.kperp2_bmag else 1.0
    contrib = 0.5 * (jnp.abs(apar) ** 2) * cache.kperp2 * bmag2 * weight
    return jnp.sum(contrib, axis=(1, 2))


def _integrate_target_mode_series(
    *,
    G0: jnp.ndarray,
    grid,
    geom,
    cache,
    params,
    time_cfg: GXTimeConfig,
    terms: LinearTerms,
    mode_method: str,
    ky_index: int,
    kx_index: int,
    sample_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mode_method not in {"z_index", "max"}:
        raise ValueError("mode_method must be 'z_index' or 'max'")

    term_cfg = _gx_term_config(terms)
    mask = jnp.asarray(grid.dealias_mask, dtype=bool)
    z_index = _gx_midplane_index(grid.z.size)
    dt = float(time_cfg.dt)
    dt_min = float(time_cfg.dt_min)
    dt_max = float(time_cfg.dt_max) if time_cfg.dt_max is not None else dt
    target_times = np.asarray(sample_times, dtype=float)
    target_samples = int(target_times.size)
    if target_samples <= 0:
        raise ValueError("sample_times must request at least one sample")
    if np.any(np.diff(target_times) < -1.0e-14):
        raise ValueError("sample_times must be monotonically nondecreasing")
    max_target_time = float(target_times[-1])
    dt_ceiling = float(time_cfg.dt_max) if time_cfg.dt_max is not None else dt
    dt_floor = max(min(dt, dt_ceiling), float(time_cfg.dt_min))
    max_steps = max(
        int(np.ceil(max_target_time / max(dt_floor, 1.0e-12))) + 8 * target_samples + 8,
        8 * target_samples + 8,
    )
    vol_fac = cache.jacobian / jnp.sum(cache.jacobian)

    G = jnp.asarray(G0)
    t = 0.0
    step = 0

    omega_max = _gx_linear_omega_max(grid, geom, params, G.shape[-5], G.shape[-4])
    wmax = float(np.sum(omega_max))
    if not time_cfg.fixed_dt and wmax > 0.0:
        dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
        dt = min(max(dt_guess, dt_min), dt_max)

    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg)
    phi_prev = fields0.phi

    def _step(G_state, cache_state, params_state, term_cfg_state, dt_state):
        return _linear_explicit_step(
            G_state,
            cache_state,
            params_state,
            term_cfg_state,
            dt_state,
            method=time_cfg.method,
        )

    stepper = jax.jit(_step, donate_argnums=(0,))

    gamma_list: list[float] = []
    omega_list: list[float] = []
    Wg_list: list[float] = []
    Wphi_list: list[float] = []
    Wapar_list: list[float] = []

    target_idx = 0
    while target_idx < target_samples and step < max_steps:
        dt_step = dt
        if not time_cfg.fixed_dt and wmax > 0.0:
            dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
            dt_step = min(max(dt_guess, dt_min), dt_max)
        next_target = float(target_times[target_idx])
        remaining = next_target - t
        if remaining <= 1.0e-14:
            remaining = 0.0
        elif dt_step > remaining:
            dt_step = max(remaining, float(time_cfg.dt_min))

        G, fields = stepper(G, cache, params, term_cfg, dt_step)
        step += 1
        t += dt_step

        if t >= next_target - 1.0e-12:
            phi = fields.phi
            apar = fields.apar if fields.apar is not None else jnp.zeros_like(phi)
            gamma, omega = _gx_growth_rate_step(
                phi,
                phi_prev,
                dt_step,
                z_index=z_index,
                mask=mask,
                mode_method=mode_method,
            )
            Wg = _gx_Wg_by_ky(G, cache, params, vol_fac)
            Wphi = _gx_Wphi_by_ky(phi, cache, params, vol_fac)
            Wapar = _gx_Wapar_by_ky(apar, cache, vol_fac)
            gamma_list.append(float(np.asarray(gamma)[ky_index, kx_index]))
            omega_list.append(float(np.asarray(omega)[ky_index, kx_index]))
            Wg_list.append(float(np.asarray(Wg)[ky_index]))
            Wphi_list.append(float(np.asarray(Wphi)[ky_index]))
            Wapar_list.append(float(np.asarray(Wapar)[ky_index]))
            target_idx += 1
            phi_prev = phi
        else:
            phi_prev = fields.phi

    if len(gamma_list) != target_samples:
        raise RuntimeError(
            "Imported-linear integration produced "
            f"{len(gamma_list)} samples, expected {target_samples} "
            f"(dt={dt}, max_steps={max_steps})"
        )
    return (
        np.asarray(gamma_list, dtype=float),
        np.asarray(omega_list, dtype=float),
        np.asarray(Wg_list, dtype=float),
        np.asarray(Wphi_list, dtype=float),
        np.asarray(Wapar_list, dtype=float),
    )


def _run_single_ky(
    *,
    ky_target: float,
    geom,
    grid_full,
    params,
    time_cfg: GXTimeConfig,
    gx_contract: GXInputContract | None,
    species: tuple[Species, ...],
    Nl: int,
    Nm: int,
    sample_times: np.ndarray,
    mode_method: str,
    kx_index: int,
    terms: LinearTerms,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    G0_full = _build_imported_initial_condition(
        grid=grid_full,
        geom=geom,
        gx_contract=gx_contract,
        species=species,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=Nl,
        Nm=Nm,
    )
    use_full_grid = gx_contract is not None
    if use_full_grid:
        G0 = G0_full
        grid = grid_full
        ky_diag_index = ky_index
    else:
        G0 = G0_full[:, :, :, ky_index : ky_index + 1, :, :]
        grid = select_ky_grid(grid_full, ky_index)
        ky_diag_index = 0
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    return _integrate_target_mode_series(
        G0=G0,
        grid=grid,
        geom=geom,
        cache=cache,
        params=params,
        time_cfg=time_cfg,
        terms=terms,
        mode_method=mode_method,
        ky_index=ky_diag_index,
        kx_index=kx_index,
        sample_times=sample_times,
    )


def _write_scan_rows(rows: list[dict[str, float]], out: Path | None) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values("ky").reset_index(drop=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def _infer_gx_linear_dt(gx_time: np.ndarray, gx_contract: GXInputContract | None) -> float:
    """Infer the underlying GX timestep from saved diagnostic times."""

    if gx_contract is not None and gx_contract.dt is not None:
        return float(gx_contract.dt)

    time_arr = np.asarray(gx_time, dtype=float)
    if time_arr.size == 0:
        raise ValueError("gx_time cannot be empty")
    nwrite = 1 if gx_contract is None else max(1, int(gx_contract.nwrite))
    positive = time_arr[time_arr > 0.0]
    if time_arr.size >= 2:
        diffs = np.diff(time_arr)
        positive_diffs = diffs[diffs > 0.0]
        if positive_diffs.size > 0:
            return float(np.median(positive_diffs) / float(nwrite))
    if positive.size > 0:
        return float(positive[0] / float(nwrite))
    raise ValueError("Could not infer a positive GX timestep from diagnostic times")


def _build_sample_steps(
    gx_time: np.ndarray,
    *,
    sample_step_stride: int,
    max_samples: int | None,
) -> np.ndarray:
    steps = np.arange(np.asarray(gx_time).size, dtype=int)
    stride = max(1, int(sample_step_stride))
    if stride > 1:
        steps = steps[::stride]
    if max_samples is not None:
        steps = steps[: max(0, int(max_samples))]
    if steps.size == 0:
        raise ValueError("Selected sample window is empty; increase --max-samples or lower --sample-step-stride")
    return np.asarray(steps, dtype=int)


def _series_cache_path(
    *,
    cache_dir: Path,
    gx_path: Path,
    geometry_file: Path,
    gx_input: Path | None,
    ky_target: float,
    Nl: int,
    Nm: int,
    mode_method: str,
    rel_floor_fraction: float,
    sample_steps: np.ndarray,
) -> Path:
    payload = {
        "version": 1,
        "gx": _file_cache_token(gx_path),
        "geometry_file": _file_cache_token(geometry_file),
        "gx_input": _file_cache_token(gx_input),
        "ky_target": float(ky_target),
        "Nl": int(Nl),
        "Nm": int(Nm),
        "mode_method": str(mode_method),
        "rel_floor_fraction": float(rel_floor_fraction),
        "sample_steps": [int(v) for v in np.asarray(sample_steps, dtype=int)],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    tag = f"{float(ky_target):0.4f}".replace(".", "p")
    return cache_dir / f"ky_{tag}_{digest}.npz"


def _load_cached_ky_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            np.asarray(data["gamma"], dtype=float),
            np.asarray(data["omega"], dtype=float),
            np.asarray(data["Wg"], dtype=float),
            np.asarray(data["Wphi"], dtype=float),
            np.asarray(data["Wapar"], dtype=float),
        )


def _save_cached_ky_series(
    path: Path,
    *,
    gamma: np.ndarray,
    omega: np.ndarray,
    Wg: np.ndarray,
    Wphi: np.ndarray,
    Wapar: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gamma=np.asarray(gamma, dtype=float),
        omega=np.asarray(omega, dtype=float),
        Wg=np.asarray(Wg, dtype=float),
        Wphi=np.asarray(Wphi, dtype=float),
        Wapar=np.asarray(Wapar, dtype=float),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare GX linear diagnostics against SPECTRAX-GK using imported GX/VMEC geometry."
    )
    parser.add_argument("--gx", type=Path, required=True, help="Path to the GX .out.nc file")
    parser.add_argument(
        "--geometry-file",
        type=Path,
        required=True,
        help="Path to the GX/VMEC geometry file (for example *.eik.nc)",
    )
    parser.add_argument(
        "--gx-input",
        type=Path,
        default=None,
        help="Optional GX input file used to infer boundary/grid defaults for imported geometry cases.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory for per-ky cached trajectory/result arrays.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse matching per-ky cached trajectory/result arrays when available.",
    )
    parser.add_argument("--ky", type=float, nargs="*", default=None, help="Specific ky values to compare")
    parser.add_argument(
        "--sample-step-stride",
        type=int,
        default=1,
        help="Subsample the saved GX diagnostic sample indices by this stride before scoring.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only score the first N selected GX diagnostic samples.",
    )
    parser.add_argument("--Nl", type=int, default=None)
    parser.add_argument("--Nm", type=int, default=None)
    parser.add_argument("--tprim", type=float, default=3.0)
    parser.add_argument("--fprim", type=float, default=1.0)
    parser.add_argument("--tau-e", type=float, default=1.0, dest="tau_e")
    parser.add_argument("--damp-ends-amp", type=float, default=0.1)
    parser.add_argument("--damp-ends-widthfrac", type=float, default=1.0 / 8.0)
    parser.add_argument("--mode-method", choices=("z_index", "max"), default="z_index")
    parser.add_argument(
        "--rel-floor-fraction",
        type=float,
        default=1.0e-2,
        help="Relative-error floor as a fraction of the peak reference magnitude for each series",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    gx_time, gx_ky, gx_kx, gx_omega, gx_Wg, gx_Wphi, gx_Wapar = _load_gx_reference(args.gx)
    positive_ky = gx_ky[gx_ky > 0.0]
    ky_values = positive_ky if args.ky is None or len(args.ky) == 0 else np.asarray(args.ky, dtype=float)
    sample_steps = _build_sample_steps(
        gx_time,
        sample_step_stride=int(args.sample_step_stride),
        max_samples=args.max_samples,
    )
    sample_times = np.asarray(gx_time[sample_steps], dtype=float)

    boundary = "linked"
    y0 = _infer_y0(gx_ky)
    nx = 1
    ny = _resolve_imported_real_fft_ny(gx_ky, None)
    nperiod = 1
    ntheta = 0
    gx_contract: GXInputContract | None = None
    species = [
        Species(
            charge=1.0,
            mass=1.0,
            density=1.0,
            temperature=1.0,
            tprim=float(args.tprim),
            fprim=float(args.fprim),
        )
    ]
    tau_e = float(args.tau_e)
    beta = 0.0
    if args.gx_input is not None:
        gx_contract = _load_gx_input_contract(args.gx_input)
        boundary = str(gx_contract.boundary)
        y0 = float(gx_contract.y0) if np.isfinite(float(gx_contract.y0)) else y0
        nx = max(1, int(gx_contract.Nx))
        ny = _resolve_imported_real_fft_ny(gx_ky, gx_contract)
        nperiod = max(1, int(gx_contract.nperiod))
        ntheta_in = int(gx_contract.ntheta)
        if ntheta_in > 0:
            ntheta = ntheta_in
        species = list(gx_contract.species)
        tau_e = float(gx_contract.tau_e)
        beta = float(gx_contract.beta)
        sample_steps = _build_sample_steps(
            gx_time,
            sample_step_stride=int(args.sample_step_stride),
            max_samples=args.max_samples,
        )
        sample_times = np.asarray(gx_time[sample_steps], dtype=float)
    dt = _infer_gx_linear_dt(gx_time, gx_contract)
    if gx_contract is not None and gx_contract.geo_option == "slab":
        gx_contract = replace(
            gx_contract,
            zero_shat=_read_gx_output_bool(args.gx, "zero_shat", default=gx_contract.zero_shat),
        )
        geom = SlabGeometry.from_config(
            GeometryConfig(model="slab", s_hat=float(gx_contract.s_hat), zero_shat=bool(gx_contract.zero_shat))
        )
        nz = int(gx_contract.ntheta) if int(gx_contract.ntheta) > 0 else 16
    else:
        geom = load_gx_geometry_netcdf(_select_geometry_source(args.gx, args.geometry_file, gx_contract))
        nz = int(np.asarray(geom.theta).size)
    if ntheta <= 0:
        ntheta = nz

    lx = 62.8
    boundary_eff = boundary
    if gx_contract is not None and gx_contract.zero_shat:
        boundary_eff = "periodic"
        lx = 2.0 * np.pi * y0

    grid_cfg = GridConfig(
        Nx=nx,
        Ny=ny,
        Nz=nz,
        Lx=lx,
        Ly=2.0 * np.pi * y0,
        boundary=boundary_eff,
        y0=y0,
        nperiod=nperiod,
        ntheta=ntheta,
    )
    grid_full = build_spectral_grid(apply_gx_geometry_grid_defaults(geom, grid_cfg))

    nl_use = int(args.Nl) if args.Nl is not None else int(gx_contract.nlaguerre if gx_contract is not None else 8)
    nm_use = int(args.Nm) if args.Nm is not None else int(gx_contract.nhermite if gx_contract is not None else 16)

    params = build_linear_params(
        species,
        tau_e=tau_e,
        kpar_scale=float(geom.gradpar()),
        beta=beta,
        fapar=(float(gx_contract.fapar) if gx_contract is not None else (1.0 if beta > 0.0 else 0.0)),
    )
    terms = LinearTerms()
    if gx_contract is not None:
        if gx_contract.hypercollisions:
            params = _apply_gx_hypercollisions(params, nhermite=nm_use)
        params = replace(
            params,
            D_hyper=float(gx_contract.D_hyper),
            damp_ends_amp=float(gx_contract.damp_ends_amp),
            damp_ends_widthfrac=float(gx_contract.damp_ends_widthfrac),
        )
        terms = replace(
            terms,
            hypercollisions=1.0 if gx_contract.hypercollisions else 0.0,
            hyperdiffusion=1.0 if gx_contract.hyper else 0.0,
        )
    else:
        params = _apply_gx_hypercollisions(params, nhermite=nm_use)
        params = replace(
            params,
            damp_ends_amp=float(args.damp_ends_amp),
            damp_ends_widthfrac=float(args.damp_ends_widthfrac),
        )
    time_cfg = GXTimeConfig(
        dt=dt,
        t_max=float(gx_time[-1]),
        method=(gx_contract.scheme if gx_contract is not None else "rk4"),
        sample_stride=(gx_contract.nwrite if gx_contract is not None else 1),
        fixed_dt=bool(gx_contract is not None and gx_contract.dt is not None),
        cfl_fac=resolve_cfl_fac((gx_contract.scheme if gx_contract is not None else "rk4"), None),
    )

    rows: list[dict[str, float]] = []
    cache_dir = None if args.cache_dir is None else args.cache_dir.expanduser().resolve()
    for ky_target in ky_values:
        ky_idx = int(np.argmin(np.abs(gx_ky - float(ky_target))))
        gx_kx_idx = _select_gx_kx_index(gx_kx, gx_contract)
        kx_idx = _match_local_kx_index(np.asarray(grid_full.kx), float(gx_kx[gx_kx_idx]))
        cache_path = None
        if cache_dir is not None:
            cache_path = _series_cache_path(
                cache_dir=cache_dir,
                gx_path=args.gx,
                geometry_file=args.geometry_file,
                gx_input=args.gx_input,
                ky_target=float(ky_target),
                Nl=nl_use,
                Nm=nm_use,
                mode_method=str(args.mode_method),
                rel_floor_fraction=float(args.rel_floor_fraction),
                sample_steps=sample_steps,
            )
        if cache_path is not None and args.reuse_cache and cache_path.exists():
            gamma, omega, Wg, Wphi, Wapar = _load_cached_ky_series(cache_path)
        else:
            gamma, omega, Wg, Wphi, Wapar = _run_single_ky(
                ky_target=float(ky_target),
                geom=geom,
                grid_full=grid_full,
                params=params,
                time_cfg=time_cfg,
                gx_contract=gx_contract,
                species=tuple(species),
                Nl=nl_use,
                Nm=nm_use,
                sample_times=sample_times,
                mode_method=args.mode_method,
                kx_index=kx_idx,
                terms=terms,
            )
            if cache_path is not None:
                _save_cached_ky_series(
                    cache_path,
                    gamma=gamma,
                    omega=omega,
                    Wg=Wg,
                    Wphi=Wphi,
                    Wapar=Wapar,
                )
        omega_ref = gx_omega[sample_steps, ky_idx, gx_kx_idx, 0]
        gamma_ref = gx_omega[sample_steps, ky_idx, gx_kx_idx, 1]
        rows.append(
            {
                "ky": float(ky_target),
                "kx_ref": float(gx_kx[gx_kx_idx]),
                "kx_local": float(np.asarray(grid_full.kx)[kx_idx]),
                "mean_abs_omega": float(np.mean(np.abs(omega - omega_ref))),
                "mean_rel_omega": _mean_rel_error(
                    omega, omega_ref, floor_fraction=float(args.rel_floor_fraction)
                ),
                "mean_abs_gamma": float(np.mean(np.abs(gamma - gamma_ref))),
                "mean_rel_gamma": _mean_rel_error(
                    gamma, gamma_ref, floor_fraction=float(args.rel_floor_fraction)
                ),
                "mean_rel_Wg": _mean_rel_error(Wg, gx_Wg[sample_steps, ky_idx], floor_fraction=1.0e-6),
                "mean_rel_Wphi": _mean_rel_error(Wphi, gx_Wphi[sample_steps, ky_idx], floor_fraction=1.0e-6),
                "mean_rel_Wapar": _mean_rel_error(Wapar, gx_Wapar[sample_steps, ky_idx], floor_fraction=1.0e-6),
                "omega_last": float(omega[-1]),
                "omega_ref_last": float(omega_ref[-1]),
                "gamma_last": float(gamma[-1]),
                "gamma_ref_last": float(gamma_ref[-1]),
            }
        )
        _write_scan_rows(rows, args.out)

    df = _write_scan_rows(rows, args.out)
    print(df.to_string(index=False))
    if args.out is not None:
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
