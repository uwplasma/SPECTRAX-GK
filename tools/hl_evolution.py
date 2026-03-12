"""Hermite-Laguerre evolution diagnostics for linear runs."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate_auto,
    fit_growth_rate_with_stats,
    select_ky_index,
)
from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
    Kinetic_OMEGA_D_SCALE,
    Kinetic_OMEGA_STAR_SCALE,
    Kinetic_RHO_STAR,
    TEM_OMEGA_D_SCALE,
    TEM_OMEGA_STAR_SCALE,
    TEM_RHO_STAR,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    _apply_gx_hypercollisions,
    _build_initial_condition,
    GX_DAMP_ENDS_AMP,
    GX_DAMP_ENDS_WIDTHFRAC,
    _electron_only_params,
    _two_species_params,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
)
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    KineticElectronBaseCase,
    KBMBaseCase,
    TEMBaseCase,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache, integrate_linear_diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hermite/Laguerre evolution diagnostics.")
    parser.add_argument(
        "--case",
        required=True,
        choices=["cyclone", "kinetic", "etg", "kbm", "tem"],
        help="Benchmark case to run.",
    )
    parser.add_argument("--ky", type=float, default=None, help="Target ky (default per case).")
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Beta value for KBM case (default per case).",
    )
    parser.add_argument("--Nl", type=int, default=None, help="Hermite resolution.")
    parser.add_argument("--Nm", type=int, default=None, help="Laguerre resolution.")
    parser.add_argument("--dt", type=float, default=None, help="Time step.")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps.")
    parser.add_argument("--method", type=str, default="imex2", help="Time integration method.")
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Sample stride for diagnostics (>=1).",
    )
    parser.add_argument(
        "--mode-method",
        type=str,
        choices=["z_index", "max", "project"],
        default="project",
        help="Mode extraction method for fitting.",
    )
    parser.add_argument(
        "--z-index",
        type=int,
        default=None,
        help="Z index for mode extraction (default: mid-plane).",
    )
    parser.add_argument(
        "--fit-signal",
        type=str,
        choices=["phi", "density", "auto"],
        default="auto",
        help="Signal used for growth-rate fitting.",
    )
    parser.add_argument(
        "--diagnostic-norm",
        type=str,
        choices=["none", "initial", "max", "time", "gx"],
        default="gx",
        help="Normalize diagnostics (GX-style uses time-normalized spectra + initial energy).",
    )
    parser.add_argument(
        "--omega-star-scale",
        type=float,
        default=None,
        help="Override omega_star_scale for diagnostics.",
    )
    parser.add_argument(
        "--omega-d-scale",
        type=float,
        default=None,
        help="Override omega_d_scale for diagnostics.",
    )
    parser.add_argument(
        "--window-max-fraction",
        type=float,
        default=0.6,
        help="Upper fraction of time to consider for fit window.",
    )
    parser.add_argument(
        "--window-start-fraction",
        type=float,
        default=0.1,
        help="Earliest fraction of time to consider for fit window.",
    )
    parser.add_argument(
        "--window-end-fraction",
        type=float,
        default=0.8,
        help="Latest fraction of time allowed for fit window end.",
    )
    parser.add_argument(
        "--window-min-amp-fraction",
        type=float,
        default=0.05,
        help="Minimum amplitude fraction of max for fit window start.",
    )
    parser.add_argument(
        "--window-max-amp-fraction",
        type=float,
        default=0.8,
        help="Upper amplitude fraction of max to consider for fit window.",
    )
    parser.add_argument(
        "--window-growth-weight",
        type=float,
        default=0.0,
        help="Weight for preferring higher growth rates in window selection.",
    )
    parser.add_argument(
        "--window-late-penalty",
        type=float,
        default=0.1,
        help="Penalty for windows late in time (higher favors earlier windows).",
    )
    parser.add_argument(
        "--window-min-slope-frac",
        type=float,
        default=0.2,
        help="Minimum slope as a fraction of the 90th percentile slope.",
    )
    parser.add_argument(
        "--window-slope-var-weight",
        type=float,
        default=0.3,
        help="Penalty weight for slope variability inside the window.",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=0.0,
        help="Minimum log-linear R2 for auto window selection.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "docs" / "_static"),
        help="Output directory for diagnostic files.",
    )
    return parser.parse_args()


def _defaults(case: str) -> dict[str, float | int]:
    if case == "cyclone":
        return dict(ky=0.3, dt=0.01, steps=15000, Nl=48, Nm=16)
    if case == "kinetic":
        return dict(ky=0.3, dt=0.0005, steps=80000, Nl=48, Nm=16)
    if case == "etg":
        return dict(ky=20.0, dt=0.0002, steps=1200, Nl=48, Nm=16)
    if case == "kbm":
        return dict(ky=0.3, dt=0.0005, steps=80000, Nl=48, Nm=16, beta=0.015)
    if case == "tem":
        return dict(ky=0.3, dt=0.001, steps=1200, Nl=48, Nm=16)
    raise ValueError(f"Unknown case '{case}'")


def _build_problem(
    case: str,
    ky: float,
    beta: float | None,
    Nl: int,
    Nm: int,
    *,
    omega_star_scale: float | None,
    omega_d_scale: float | None,
):
    if case == "cyclone":
        cfg = CycloneBaseCase()
        grid_full = build_spectral_grid(cfg.grid)
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
            damp_ends_amp=GX_DAMP_ENDS_AMP,
            damp_ends_widthfrac=GX_DAMP_ENDS_WIDTHFRAC,
        )
        params = _apply_gx_hypercollisions(params)
        terms = LinearTerms()
        electron_index = 0
        ns = 1
    elif case == "kinetic":
        cfg = KineticElectronBaseCase()
        grid_full = build_spectral_grid(cfg.grid)
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=Kinetic_OMEGA_D_SCALE,
            omega_star_scale=Kinetic_OMEGA_STAR_SCALE,
            rho_star=Kinetic_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
        terms = LinearTerms()
        electron_index = 1
        ns = 2
    elif case == "etg":
        cfg = ETGBaseCase()
        grid_full = build_spectral_grid(cfg.grid)
        geom = SAlphaGeometry.from_config(cfg.geometry)
        if getattr(cfg.model, "adiabatic_ions", False):
            params = _electron_only_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
            )
        else:
            params = _two_species_params(
                cfg.model,
                kpar_scale=float(geom.gradpar()),
                omega_d_scale=ETG_OMEGA_D_SCALE,
                omega_star_scale=ETG_OMEGA_STAR_SCALE,
                rho_star=ETG_RHO_STAR,
                damp_ends_amp=0.0,
                damp_ends_widthfrac=0.0,
            )
        terms = LinearTerms()
        charge = np.atleast_1d(np.asarray(params.charge_sign))
        electron_index = int(np.argmin(charge))
        ns = int(charge.size)
    elif case == "kbm":
        cfg = KBMBaseCase()
        grid_full = build_spectral_grid(cfg.grid)
        geom = SAlphaGeometry.from_config(cfg.geometry)
        beta_val = float(beta) if beta is not None else 0.015
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=KBM_OMEGA_D_SCALE,
            omega_star_scale=KBM_OMEGA_STAR_SCALE,
            rho_star=KBM_RHO_STAR,
            beta_override=beta_val,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
        terms = LinearTerms(bpar=0.0)
        electron_index = 1
        ns = 2
    elif case == "tem":
        cfg = TEMBaseCase()
        grid_full = build_spectral_grid(cfg.grid)
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = _two_species_params(
            cfg.model,
            kpar_scale=float(geom.gradpar()),
            omega_d_scale=TEM_OMEGA_D_SCALE,
            omega_star_scale=TEM_OMEGA_STAR_SCALE,
            rho_star=TEM_RHO_STAR,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
        terms = LinearTerms(bpar=0.0)
        electron_index = 1
        ns = 2
    else:
        raise ValueError(f"Unknown case '{case}'")

    ky_index = select_ky_index(np.asarray(grid_full.ky), float(ky))
    grid = select_ky_grid(grid_full, ky_index)

    init_cfg = getattr(cfg, "init", None)
    if ns == 1:
        if init_cfg is not None:
            G0 = np.asarray(
                _build_initial_condition(
                    grid,
                    geom,
                    ky_index=0,
                    kx_index=0,
                    Nl=Nl,
                    Nm=Nm,
                    init_cfg=init_cfg,
                )
            )
        else:
            G0 = np.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
            G0[0, 0, 0, 0, :] = 1e-3 + 0.0j
    else:
        if init_cfg is not None:
            G0_single = np.asarray(
                _build_initial_condition(
                    grid,
                    geom,
                    ky_index=0,
                    kx_index=0,
                    Nl=Nl,
                    Nm=Nm,
                    init_cfg=init_cfg,
                )
            )
            G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
            G0[electron_index] = G0_single
        else:
            G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
            G0[electron_index, 0, 0, 0, 0, :] = 1e-3 + 0.0j

    if omega_star_scale is not None:
        params = replace(params, omega_star_scale=float(omega_star_scale))
    if omega_d_scale is not None:
        params = replace(params, omega_d_scale=float(omega_d_scale))

    return cfg, grid, geom, params, terms, G0, electron_index


def _plot_time_series(
    outdir: Path,
    prefix: str,
    t: np.ndarray,
    phi_log_energy: np.ndarray,
    dens_log_energy: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    title: str | None = None,
    signal_log_energy: np.ndarray | None = None,
    fit_log_energy: np.ndarray | None = None,
):
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    base = float(np.nanmax(phi_log_energy)) if phi_log_energy.size else 0.0
    if not np.isfinite(base):
        base = 0.0
    lin_phi = np.exp(np.clip(phi_log_energy - base, -200.0, 0.0))
    ax[0].plot(t, lin_phi, label=r"$|\phi|^2$ (normalized)")
    ax[0].set_ylabel("energy")
    ax[0].legend(loc="best")
    ax[1].plot(t, phi_log_energy, label=r"$\log |\phi|^2$")
    ax[1].plot(t, dens_log_energy, label=r"$\log |n|^2$")
    if signal_log_energy is not None:
        ax[1].plot(t, signal_log_energy, "--", linewidth=1.2, label="fit signal")
    if fit_log_energy is not None:
        ax[1].plot(t, fit_log_energy, ":", linewidth=1.5, label="fit line")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("log energy")
    ax[1].legend(loc="best")
    if tmin is not None and tmax is not None:
        for axis in ax:
            axis.axvspan(tmin, tmax, color="#f0c674", alpha=0.25, label="fit window")
    if title:
        ax[0].set_title(title)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_energy_timeseries.png", dpi=200)
    plt.close(fig)


def _plot_hl_imshow(
    outdir: Path,
    prefix: str,
    t: np.ndarray,
    hl_energy: np.ndarray,
    *,
    diagnostic_norm: str,
):
    hl_vals = np.nan_to_num(hl_energy, nan=0.0, posinf=0.0, neginf=0.0)
    hl_vals = np.maximum(hl_vals, 1e-300)
    if diagnostic_norm in {"time", "gx"}:
        total = np.sum(hl_vals, axis=(1, 2), keepdims=True)
        total = np.where(np.isfinite(total) & (total > 0), total, 1.0)
        hl_vals = hl_vals / total
    elif diagnostic_norm == "initial":
        denom = np.sum(hl_vals[0]) if hl_vals.shape[0] > 0 else 1.0
        denom = denom if np.isfinite(denom) and denom > 0 else 1.0
        hl_vals = hl_vals / denom
    elif diagnostic_norm == "max":
        denom = np.max(hl_vals)
        denom = denom if np.isfinite(denom) and denom > 0 else 1.0
        hl_vals = hl_vals / denom
    hl_vals = np.maximum(hl_vals, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl_log = np.log10(hl_vals)
    hl_flat = hl_log.reshape(hl_log.shape[0], -1).T
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    im = ax.imshow(
        hl_flat,
        aspect="auto",
        origin="lower",
        extent=[float(t[0]), float(t[-1]), 0, hl_flat.shape[0]],
        interpolation="nearest",
    )
    ax.set_xlabel("t")
    ax.set_ylabel("flattened (l,m) index")
    fig.colorbar(im, ax=ax, label="log10 energy")
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_hl_energy.png", dpi=200)
    plt.close(fig)


def _log_energy(series: np.ndarray) -> np.ndarray:
    """Compute log(mean(|series|^2)) safely."""

    series = np.asarray(series)
    if series.size == 0:
        return np.array([])
    abs_val = np.abs(series)
    abs_val = np.nan_to_num(abs_val, nan=0.0, posinf=0.0, neginf=0.0)
    eps = np.finfo(abs_val.dtype).tiny
    axes = tuple(range(1, abs_val.ndim))
    scale = np.max(abs_val, axis=axes, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    scaled = abs_val / scale
    mean_scaled = np.mean(scaled**2, axis=axes)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_energy = np.log(mean_scaled + eps) + 2.0 * np.log(np.squeeze(scale, axis=axes))
    return log_energy


def _norm_base(log_energy: np.ndarray, mode: str) -> float:
    if mode == "none":
        return 0.0
    if mode in {"initial", "gx"}:
        base = float(log_energy[0]) if log_energy.size else 0.0
        if not np.isfinite(base):
            finite = log_energy[np.isfinite(log_energy)]
            base = float(finite[0]) if finite.size else 0.0
        return base
    if mode == "max":
        base = float(np.nanmax(log_energy)) if log_energy.size else 0.0
        return base if np.isfinite(base) else 0.0
    if mode == "time":
        base = float(np.nanmean(log_energy)) if log_energy.size else 0.0
        return base if np.isfinite(base) else 0.0
    raise ValueError(f"Unknown diagnostic_norm '{mode}'")


def _normalize_log_energy(log_energy: np.ndarray, mode: str) -> np.ndarray:
    base = _norm_base(log_energy, mode)
    return log_energy - base


def _log_amp(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.size == 0:
        return np.array([])
    finite = np.isfinite(signal)
    if np.any(finite):
        scale = float(np.max(np.abs(signal[finite])))
    else:
        scale = 1.0
    if not np.all(finite):
        signal = np.where(finite, signal, 0.0)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scaled = signal / scale
    amp = np.abs(scaled)
    eps = np.finfo(amp.dtype).tiny
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.maximum(amp, eps)) + np.log(scale)


def _default_fit_signal(case: str) -> str:
    if case in {"etg", "kinetic", "kbm"}:
        return "density"
    return "phi"


def _reference_value(case: str, ky: float, beta: float | None) -> tuple[float | None, float | None]:
    if case == "cyclone":
        ref = load_cyclone_reference()
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        return float(ref.gamma[idx]), float(ref.omega[idx])
    if case == "kinetic":
        ref = load_cyclone_reference_kinetic()
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        return float(ref.gamma[idx]), float(ref.omega[idx])
    if case == "etg":
        ref = load_etg_reference()
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        return float(ref.gamma[idx]), float(ref.omega[idx])
    if case == "tem":
        ref = load_tem_reference()
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        return float(ref.gamma[idx]), float(ref.omega[idx])
    if case == "kbm":
        if beta is None:
            return None, None
        ref = load_kbm_reference()
        idx = int(np.argmin(np.abs(ref.ky - beta)))
        return float(ref.gamma[idx]), float(ref.omega[idx])
    return None, None


def main() -> int:
    args = _parse_args()
    defaults = _defaults(args.case)

    ky = float(args.ky) if args.ky is not None else float(defaults["ky"])
    beta = args.beta if args.beta is not None else defaults.get("beta", None)
    Nl = int(args.Nl) if args.Nl is not None else int(defaults["Nl"])
    Nm = int(args.Nm) if args.Nm is not None else int(defaults["Nm"])
    dt = float(args.dt) if args.dt is not None else float(defaults["dt"])
    steps = int(args.steps) if args.steps is not None else int(defaults["steps"])
    method = str(args.method)
    sample_stride = int(args.sample_stride)
    mode_method = args.mode_method
    fit_signal = args.fit_signal
    if fit_signal == "auto":
        fit_signal = _default_fit_signal(args.case)
    diagnostic_norm = args.diagnostic_norm

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg, grid, geom, params, terms, G0, electron_index = _build_problem(
        args.case,
        ky,
        beta,
        Nl,
        Nm,
        omega_star_scale=args.omega_star_scale,
        omega_d_scale=args.omega_d_scale,
    )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)

    print(f"Case: {args.case}")
    print(f"Config: {cfg}")
    print(f"ky={ky} beta={beta} Nl={Nl} Nm={Nm} dt={dt} steps={steps} method={method}")
    print(f"sample_stride={sample_stride} outdir={outdir}")
    z_index = args.z_index
    if z_index is None:
        z_index = int(min(grid.z.size - 1, grid.z.size // 2 + 1))
    print(
        "fit_signal="
        f"{fit_signal} mode_method={mode_method} z_index={z_index} min_r2={args.min_r2} "
        f"window_start_fraction={args.window_start_fraction} window_max_fraction={args.window_max_fraction} "
        f"window_end_fraction={args.window_end_fraction} window_min_amp_fraction={args.window_min_amp_fraction} "
        f"window_max_amp_fraction={args.window_max_amp_fraction} window_growth_weight={args.window_growth_weight} "
        f"late_penalty={args.window_late_penalty} "
        f"min_slope_frac={args.window_min_slope_frac} slope_var_weight={args.window_slope_var_weight} "
        f"diagnostic_norm={diagnostic_norm}"
    )

    params_use = params
    _, phi_t, density_t, hl_t = integrate_linear_diagnostics(
        np.asarray(G0),
        grid,
        geom,
        params_use,
        dt=dt,
        steps=steps,
        method=method,
        cache=cache,
        terms=terms,
        sample_stride=sample_stride,
        species_index=electron_index,
        record_hl_energy=True,
    )

    phi_t_np = np.asarray(phi_t)
    density_t_np = np.asarray(density_t)
    hl_t_np = np.asarray(hl_t)
    t = np.arange(phi_t_np.shape[0]) * dt * sample_stride

    phi_log = _log_energy(phi_t_np)
    dens_log = _log_energy(density_t_np)
    phi_log_norm = _normalize_log_energy(phi_log, diagnostic_norm)
    dens_log_norm = _normalize_log_energy(dens_log, diagnostic_norm)

    sel = ModeSelection(ky_index=0, kx_index=0, z_index=int(z_index))
    signal_src = phi_t_np if fit_signal == "phi" else density_t_np
    signal = extract_mode_time_series(signal_src, sel, method=mode_method)
    signal_log_amp = _log_amp(signal)
    signal_log_energy = 2.0 * signal_log_amp
    signal_log_norm = _normalize_log_energy(signal_log_energy, diagnostic_norm)
    gamma_ref, omega_ref = _reference_value(args.case, ky, beta)
    require_positive = False
    if gamma_ref is not None and gamma_ref > 0.0:
        require_positive = True
    gamma, omega, tmin, tmax = fit_growth_rate_auto(
        t,
        signal,
        min_points=40,
        start_fraction=args.window_start_fraction,
        min_amp_fraction=args.window_min_amp_fraction,
        window_method="loglinear",
        min_r2=args.min_r2,
        max_fraction=args.window_max_fraction,
        end_fraction=args.window_end_fraction,
        max_amp_fraction=args.window_max_amp_fraction,
        growth_weight=args.window_growth_weight,
        late_penalty=args.window_late_penalty,
        min_slope_frac=args.window_min_slope_frac,
        slope_var_weight=args.window_slope_var_weight,
        require_positive=require_positive,
    )
    gamma_fit, omega_fit, r2_log, r2_phase = fit_growth_rate_with_stats(
        t, signal, tmin=tmin, tmax=tmax
    )
    fit_log_norm = None
    if np.isfinite(tmin) and np.isfinite(tmax):
        mask = (t >= tmin) & (t <= tmax)
        tt = t[mask]
        yy = signal_log_amp[mask]
        if tt.size >= 2:
            A = np.vstack([tt, np.ones_like(tt)]).T
            slope, offset = np.linalg.lstsq(A, yy, rcond=None)[0]
            fit_log_energy = 2.0 * (slope * t + offset)
            base = _norm_base(signal_log_energy, diagnostic_norm)
            fit_log_norm = fit_log_energy - base
    rel_gamma = None
    rel_omega = None
    if gamma_ref is not None and gamma_ref != 0.0:
        rel_gamma = (gamma_fit - gamma_ref) / gamma_ref
    if omega_ref is not None and omega_ref != 0.0:
        rel_omega = (omega_fit - omega_ref) / omega_ref

    tag = f"{args.case}_ky{ky:.3g}"
    if args.case == "kbm":
        tag = f"{tag}_beta{float(beta):.3g}"
    prefix = tag.replace(".", "p")

    np.save(outdir / f"{prefix}_t.npy", t)
    np.save(outdir / f"{prefix}_hl_energy.npy", hl_t_np)
    phi_base = float(np.nanmax(phi_log_norm)) if phi_log_norm.size else 0.0
    dens_base = float(np.nanmax(dens_log_norm)) if dens_log_norm.size else 0.0
    if not np.isfinite(phi_base):
        phi_base = 0.0
    if not np.isfinite(dens_base):
        dens_base = 0.0
    np.save(outdir / f"{prefix}_phi_energy.npy", np.exp(np.clip(phi_log_norm - phi_base, -200.0, 0.0)))
    np.save(outdir / f"{prefix}_density_energy.npy", np.exp(np.clip(dens_log_norm - dens_base, -200.0, 0.0)))
    np.save(outdir / f"{prefix}_phi_log_energy.npy", phi_log_norm)
    np.save(outdir / f"{prefix}_density_log_energy.npy", dens_log_norm)
    title = (
        f"fit: gamma={gamma_fit:.4g} omega={omega_fit:.4g} "
        f"r2_log={r2_log:.3f} r2_phase={r2_phase:.3f}"
    )
    if gamma_ref is not None and omega_ref is not None:
        title += f" | ref: gamma={gamma_ref:.4g} omega={omega_ref:.4g}"
        if rel_gamma is not None and rel_omega is not None:
            title += f" rel=({rel_gamma:.2%}, {rel_omega:.2%})"

    _plot_time_series(
        outdir,
        prefix,
        t,
        phi_log_norm,
        dens_log_norm,
        tmin=tmin,
        tmax=tmax,
        title=title,
        signal_log_energy=signal_log_norm,
        fit_log_energy=fit_log_norm,
    )
    _plot_hl_imshow(outdir, prefix, t, hl_t_np, diagnostic_norm=diagnostic_norm)

    print(f"Fit window: tmin={tmin:.4g} tmax={tmax:.4g}")
    print(
        f"Fit results: gamma={gamma_fit:.6g} omega={omega_fit:.6g} "
        f"r2_log={r2_log:.4f} r2_phase={r2_phase:.4f}"
    )
    if gamma_ref is not None and omega_ref is not None:
        print(f"Reference: gamma={gamma_ref:.6g} omega={omega_ref:.6g}")
        if rel_gamma is not None and rel_omega is not None:
            print(f"Relative error: gamma={rel_gamma:.3%} omega={rel_omega:.3%}")
    print(f"Saved diagnostics with prefix '{prefix}' in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
