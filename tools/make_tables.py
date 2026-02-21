"""Generate CSV tables for documentation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from pprint import pformat
import argparse
import sys

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.benchmarks import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    GX_DAMP_ENDS_AMP,
    GX_DAMP_ENDS_WIDTHFRAC,
    _apply_gx_hypercollisions,
    _build_initial_condition,
    _midplane_index,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
    CycloneScanResult,
    LinearScanResult,
    run_cyclone_linear,
    run_etg_linear,
    run_kinetic_linear,
    run_kbm_beta_scan,
    run_tem_linear,
)
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    ETGModelConfig,
    GridConfig,
    KineticElectronBaseCase,
    KBMBaseCase,
    TEMBaseCase,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.gx_integrators import GXTimeConfig, integrate_linear_gx
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache
from spectraxgk.linear_krylov import KrylovConfig
from spectraxgk.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate_auto,
    select_ky_index,
)

CYCLONE_SOLVER = "time"
KINETIC_SOLVER = "time"
ETG_SOLVER = "time"
KBM_SOLVER = "time"
TEM_SOLVER = "time"

CYCLONE_KRYLOV = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    power_iters=60,
    power_dt=0.01,
    shift_maxiter=30,
    shift_restart=10,
    shift_tol=1.0e-3,
)
KINETIC_KRYLOV = KrylovConfig(
    method="propagator",
    krylov_dim=16,
    restarts=1,
    power_iters=60,
    power_dt=0.0005,
    shift_maxiter=30,
    shift_restart=10,
    shift_tol=1.0e-3,
)
ETG_KRYLOV = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    power_iters=80,
    power_dt=0.002,
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=2.0e-3,
)
KBM_KRYLOV = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    power_iters=60,
    power_dt=0.005,
    shift_maxiter=30,
    shift_restart=10,
    shift_tol=1.0e-3,
)
TEM_KRYLOV = KrylovConfig(
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    power_iters=60,
    power_dt=0.005,
    shift_maxiter=30,
    shift_restart=10,
    shift_tol=1.0e-3,
)



def _build_rows(scan, ref):
    rows = ["ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega"]
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        rel_gamma = (gamma - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
        rel_omega = (omega - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
        rows.append(
            f"{ky:.3f},{gamma_ref:.6f},{omega_ref:.6f},{gamma:.6f},{omega:.6f},{rel_gamma:.3f},{rel_omega:.3f}"
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validation tables.")
    parser.add_argument(
        "--case",
        choices=["all", "cyclone"],
        default="all",
        help="Limit table generation to a specific case.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging.")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm progress bars."
    )
    return parser.parse_args()


def _log(msg: str, *, verbose: bool, use_tqdm: bool) -> None:
    if not verbose:
        return
    if use_tqdm:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _format_cfg(cfg) -> str:
    if is_dataclass(cfg):
        return pformat(asdict(cfg), width=120, sort_dicts=False)
    return pformat(cfg, width=120, sort_dicts=False)


def _window_value(val, idx: int) -> float | None:
    if val is None:
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        return float(val[idx])
    return float(val)


def _scan_linear_verbose(
    *,
    ky_values: np.ndarray,
    run_linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    label: str,
    run_kwargs: dict | None = None,
    ref=None,
    verbose: bool,
    progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _log(f"\n=== {label} scan ===", verbose=verbose, use_tqdm=progress)
    _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
    _log(
        f"Numerics: Nl={Nl} Nm={Nm} method={method} solver={solver} dt={dt} steps={steps}",
        verbose=verbose,
        use_tqdm=progress,
    )
    _log(f"Window params: {window_kw}", verbose=verbose, use_tqdm=progress)
    if run_kwargs:
        _log(f"Extra kwargs: {run_kwargs}", verbose=verbose, use_tqdm=progress)
    if tmin is not None or tmax is not None:
        _log(f"Manual window tmin={tmin} tmax={tmax}", verbose=verbose, use_tqdm=progress)

    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    iterator = tqdm(ky_values, desc=f"{label} ky scan") if progress else ky_values
    for i, ky in enumerate(iterator):
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        tmin_i = _window_value(tmin, i)
        tmax_i = _window_value(tmax, i)
        _log(
            f"[{label}] start ky={float(ky):.4g} dt={dt_i:.4g} steps={steps_i} tmax={dt_i*steps_i:.4g}",
            verbose=verbose,
            use_tqdm=progress,
        )
        extra = run_kwargs or {}
        result = run_linear_fn(
            ky_target=float(ky),
            cfg=cfg,
            Nl=Nl,
            Nm=Nm,
            dt=dt_i,
            steps=steps_i,
            method=method,
            solver=solver,
            krylov_cfg=krylov_cfg,
            auto_window=auto_window,
            tmin=tmin_i,
            tmax=tmax_i,
            **window_kw,
            **extra,
        )
        gammas.append(float(result.gamma))
        omegas.append(float(result.omega))
        ky_out.append(float(result.ky))
        msg = f"[{label}] done ky={float(result.ky):.4g} gamma={result.gamma:.6g} omega={result.omega:.6g}"
        if ref is not None:
            idx = int(np.argmin(np.abs(ref.ky - result.ky)))
            gamma_ref = float(ref.gamma[idx])
            omega_ref = float(ref.omega[idx])
            rel_gamma = (result.gamma - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
            rel_omega = (result.omega - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
            msg += (
                f" | ref gamma={gamma_ref:.6g} omega={omega_ref:.6g}"
                f" rel_gamma={rel_gamma:.3g} rel_omega={rel_omega:.3g}"
            )
        _log(msg, verbose=verbose, use_tqdm=progress)

    return np.array(ky_out), np.array(gammas), np.array(omegas)


def _scan_kbm_verbose(
    *,
    betas: np.ndarray,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    label: str,
    ref=None,
    run_kwargs: dict | None = None,
    verbose: bool,
    progress: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _log(f"\n=== {label} beta scan ===", verbose=verbose, use_tqdm=progress)
    _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
    _log(
        f"Numerics: Nl={Nl} Nm={Nm} method={method} solver={solver} dt={dt} steps={steps}",
        verbose=verbose,
        use_tqdm=progress,
    )
    _log(f"Window params: {window_kw}", verbose=verbose, use_tqdm=progress)
    if run_kwargs:
        _log(f"Extra kwargs: {run_kwargs}", verbose=verbose, use_tqdm=progress)
    if tmin is not None or tmax is not None:
        _log(f"Manual window tmin={tmin} tmax={tmax}", verbose=verbose, use_tqdm=progress)

    gammas: list[float] = []
    omegas: list[float] = []
    beta_out: list[float] = []
    iterator = tqdm(betas, desc=f"{label} beta scan") if progress else betas
    for i, beta in enumerate(iterator):
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        tmin_i = _window_value(tmin, i)
        tmax_i = _window_value(tmax, i)
        _log(
            f"[{label}] start beta={float(beta):.4g} dt={dt_i:.4g} steps={steps_i} tmax={dt_i*steps_i:.4g}",
            verbose=verbose,
            use_tqdm=progress,
        )
        extra = run_kwargs or {}
        result = run_kbm_beta_scan(
            np.asarray([float(beta)]),
            cfg=cfg,
            ky_target=0.3,
            Nl=Nl,
            Nm=Nm,
            dt=dt_i,
            steps=steps_i,
            method=method,
            solver=solver,
            krylov_cfg=krylov_cfg,
            auto_window=auto_window,
            tmin=tmin_i,
            tmax=tmax_i,
            **window_kw,
            **extra,
        )
        gamma = float(result.gamma[0])
        omega = float(result.omega[0])
        gammas.append(gamma)
        omegas.append(omega)
        beta_out.append(float(beta))
        msg = f"[{label}] done beta={float(beta):.4g} gamma={gamma:.6g} omega={omega:.6g}"
        if ref is not None:
            idx = int(np.argmin(np.abs(ref.ky - beta)))
            gamma_ref = float(ref.gamma[idx])
            omega_ref = float(ref.omega[idx])
            rel_gamma = (gamma - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
            rel_omega = (omega - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
            msg += (
                f" | ref gamma={gamma_ref:.6g} omega={omega_ref:.6g}"
                f" rel_gamma={rel_gamma:.3g} rel_omega={rel_omega:.3g}"
            )
        _log(msg, verbose=verbose, use_tqdm=progress)

    return np.array(beta_out), np.array(gammas), np.array(omegas)


WINDOWS = {
    "cyclone": dict(
        window_fraction=0.3,
        min_points=80,
        start_fraction=0.58,
        growth_weight=0.0,
        require_positive=True,
        min_amp_fraction=0.05,
        max_fraction=0.6,
        end_fraction=0.8,
        max_amp_fraction=0.8,
        late_penalty=0.3,
        window_method="loglinear",
        mode_method="project",
    ),
    "kinetic": dict(
        window_fraction=0.3,
        min_points=120,
        start_fraction=0.5,
        growth_weight=0.1,
        require_positive=True,
        min_amp_fraction=0.1,
    ),
    "etg": dict(
        window_fraction=0.25,
        min_points=100,
        start_fraction=0.45,
        growth_weight=0.2,
        require_positive=True,
        min_amp_fraction=0.2,
    ),
    "kbm": dict(
        window_fraction=0.3,
        min_points=120,
        start_fraction=0.35,
        growth_weight=0.0,
        require_positive=False,
        min_amp_fraction=0.05,
    ),
    "tem": dict(
        window_fraction=0.35,
        min_points=120,
        start_fraction=0.5,
        growth_weight=0.2,
        require_positive=True,
        min_amp_fraction=0.1,
    ),
}

GX_CYCLONE_WINDOW = dict(
    window_method="loglinear",
    min_points=40,
    start_fraction=0.1,
    max_fraction=0.8,
    end_fraction=0.8,
    require_positive=True,
    min_amp_fraction=0.05,
    max_amp_fraction=0.8,
    growth_weight=0.1,
    late_penalty=0.1,
)


def _gx_balanced_policy(ky: float) -> tuple[int, int, float]:
    if ky < 0.08:
        return 16, 8, 80.0
    if ky < 0.15:
        return 16, 8, 20.0
    if ky <= 0.25:
        return 24, 12, 20.0
    return 24, 12, 10.0


def _gx_mode_policy(ky: float) -> str:
    return "max" if ky < 0.3 else "project"


def _gx_window_policy(ky: float, base_window: dict) -> dict:
    window = dict(base_window)
    if ky < 0.08:
        window["start_fraction"] = 0.415
        window["min_points"] = max(int(window.get("min_points", 0)), 80)
        window["min_slope_frac"] = 0.25
    if ky >= 0.3:
        window["start_fraction"] = 0.3
        window["end_fraction"] = 0.9
        window["min_amp_fraction"] = 0.0
        window["max_amp_fraction"] = 1.0
        window["late_penalty"] = 0.0
    return window


def _cyclone_gx_scan(
    ky_values: np.ndarray,
    cfg: CycloneBaseCase,
    window_kw: dict,
    *,
    verbose: bool,
    progress: bool,
) -> LinearScanResult:
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(cfg.grid)
    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    iterator = tqdm(ky_values, desc="Cyclone GX ky scan") if progress else ky_values
    for ky in iterator:
        ky_val = float(ky)
        Nl, Nm, tmax = _gx_balanced_policy(ky_val)
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
        params = _apply_gx_hypercollisions(params, nhermite=Nm)
        terms = LinearTerms()

        ky_idx = select_ky_index(np.asarray(grid_full.ky), ky_val)
        grid = select_ky_grid(grid_full, ky_idx)
        G0 = _build_initial_condition(
            grid,
            geom,
            ky_index=0,
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            init_cfg=cfg.init,
        )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        time_cfg = GXTimeConfig(
            t_max=tmax,
            dt=0.01,
            fixed_dt=False,
            dt_min=1.0e-7,
            dt_max=0.1,
            cfl_fac=0.3,
        )
        _log(
            f"[Cyclone GX] ky={ky_val:.3f} Nl={Nl} Nm={Nm} tmax={tmax}",
            verbose=verbose,
            use_tqdm=progress,
        )
        t, phi_t, _gamma_t, _omega_t = integrate_linear_gx(
            G0,
            grid,
            cache,
            params,
            geom,
            time_cfg,
            terms,
            mode_method="z_index",
            z_index=_midplane_index(grid),
        )
        sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
        mode_method = _gx_mode_policy(ky_val)
        signal = extract_mode_time_series(np.asarray(phi_t), sel, method=mode_method)
        fit_kw = _gx_window_policy(ky_val, window_kw)
        fit_kw = {k: v for k, v in fit_kw.items() if k != "mode_method"}
        gamma, omega, tmin, tmax_fit = fit_growth_rate_auto(np.asarray(t), signal, **fit_kw)
        _log(
            f"[Cyclone GX] ky={ky_val:.3f} method={mode_method} fit=[{tmin:.3g}, {tmax_fit:.3g}] "
            f"gamma={gamma:.6g} omega={omega:.6g}",
            verbose=verbose,
            use_tqdm=progress,
        )
        ky_out.append(ky_val)
        gammas.append(float(gamma))
        omegas.append(float(omega))
    return LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


def _scale_steps(ky: np.ndarray, base_steps: int, ky_ref: float, max_steps: int) -> np.ndarray:
    scale = ky_ref / np.maximum(ky, 1.0e-6)
    steps = base_steps * np.maximum(1.0, scale)
    return np.clip(steps.astype(int), base_steps, max_steps)


def _scale_dt(ky: np.ndarray, base_dt: float, ky_ref: float) -> np.ndarray:
    scale = np.minimum(1.0, ky_ref / np.maximum(ky, 1.0e-6))
    return base_dt * scale


def main() -> int:
    args = _parse_args()
    verbose = not args.quiet
    progress = not args.no_progress

    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)

    ref = load_cyclone_reference()
    ky_subset = np.array([0.3, 0.4])
    cfg = CycloneBaseCase(
        grid=GridConfig(Nx=1, Ny=18, Nz=96, Lx=62.8, Ly=62.8, y0=20.0, ntheta=32, nperiod=2)
    )

    ky_low, g_low, w_low = _scan_linear_verbose(
        ky_values=ky_subset,
        run_linear_fn=run_cyclone_linear,
        cfg=cfg,
        Nl=48,
        Nm=16,
        dt=0.002,
        steps=5000,
        method="imex2",
        solver=CYCLONE_SOLVER,
        krylov_cfg=CYCLONE_KRYLOV,
        window_kw=WINDOWS["cyclone"],
        label="Cyclone low-res",
        ref=ref,
        verbose=verbose,
        progress=progress,
    )
    ky_high, g_high, w_high = _scan_linear_verbose(
        ky_values=ky_subset,
        run_linear_fn=run_cyclone_linear,
        cfg=cfg,
        Nl=48,
        Nm=16,
        dt=0.002,
        steps=5000,
        method="imex2",
        solver=CYCLONE_SOLVER,
        krylov_cfg=CYCLONE_KRYLOV,
        window_kw=WINDOWS["cyclone"],
        label="Cyclone high-res",
        ref=ref,
        verbose=verbose,
        progress=progress,
    )
    low_scan = CycloneScanResult(ky=ky_low, gamma=g_low, omega=w_low)
    high_scan = CycloneScanResult(ky=ky_high, gamma=g_high, omega=w_high)

    (outdir / "cyclone_scan_table_lowres.csv").write_text(
        "\n".join(_build_rows(low_scan, ref)) + "\n", encoding="utf-8"
    )
    (outdir / "cyclone_scan_table_highres.csv").write_text(
        "\n".join(_build_rows(high_scan, ref)) + "\n", encoding="utf-8"
    )

    conv_rows = [
        "ky,gamma_low,gamma_high,omega_low,omega_high,rel_gamma_change,rel_omega_change"
    ]
    for ky, g_lo, g_hi, w_lo, w_hi in zip(ky_low, g_low, g_high, w_low, w_high):
        rel_g = (g_hi - g_lo) / g_hi if g_hi != 0.0 else np.nan
        rel_w = (w_hi - w_lo) / w_hi if w_hi != 0.0 else np.nan
        row = f"{ky:.3f},{g_lo:.6f},{g_hi:.6f},{w_lo:.6f},{w_hi:.6f},{rel_g:.3f},{rel_w:.3f}"
        conv_rows.append(row)
        _log(f"[Cyclone conv] {row}", verbose=verbose, use_tqdm=progress)
    (outdir / "cyclone_scan_convergence.csv").write_text(
        "\n".join(conv_rows) + "\n", encoding="utf-8"
    )

    full_cfg = CycloneBaseCase()
    full_geom = SAlphaGeometry.from_config(full_cfg.geometry)
    full_params = LinearParams(
        R_over_Ln=full_cfg.model.R_over_Ln,
        R_over_LTi=full_cfg.model.R_over_LTi,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(full_geom.gradpar()),
    )
    full_ky, full_g, full_w = _scan_linear_verbose(
        ky_values=np.array([0.2, 0.3, 0.4]),
        run_linear_fn=run_cyclone_linear,
        cfg=full_cfg,
        Nl=48,
        Nm=16,
        dt=0.002,
        steps=5000,
        method="imex2",
        solver=CYCLONE_SOLVER,
        krylov_cfg=CYCLONE_KRYLOV,
        window_kw=WINDOWS["cyclone"],
        label="Cyclone full-operator",
        run_kwargs={"params": full_params, "terms": LinearTerms()},
        ref=ref,
        verbose=verbose,
        progress=progress,
    )
    full_scan = CycloneScanResult(ky=full_ky, gamma=full_g, omega=full_w)

    full_rows = [
        "ky,gamma_ref,omega_ref,gamma_full,omega_full,abs_gamma,abs_omega,rel_gamma,rel_omega"
    ]
    for ky, gamma, omega in zip(full_ky, full_g, full_w):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        gamma_abs = abs(float(gamma))
        omega_abs = abs(float(omega))
        rel_gamma = (gamma_abs - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
        rel_omega = (omega_abs - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
        row = (
            f"{ky:.3f},{gamma_ref:.6f},{omega_ref:.6f},{gamma:.6f},{omega:.6f},"
            f"{gamma_abs:.6f},{omega_abs:.6f},{rel_gamma:.3f},{rel_omega:.3f}"
        )
        full_rows.append(row)
        _log(f"[Cyclone full] {row}", verbose=verbose, use_tqdm=progress)
    (outdir / "cyclone_full_operator_scan_table.csv").write_text(
        "\n".join(full_rows) + "\n", encoding="utf-8"
    )

    rho_values = np.array([0.8, 1.0, 1.2])
    rho_rows = ["rho_star,mean_gamma_ratio,mean_omega_ratio"]
    for rho in rho_values:
        params = LinearParams(
            R_over_Ln=full_cfg.model.R_over_Ln,
            R_over_LTi=full_cfg.model.R_over_LTi,
            omega_d_scale=1.0,
            omega_star_scale=1.0,
            rho_star=float(rho),
            kpar_scale=float(full_geom.gradpar()),
        )
        scan_ky, scan_g, scan_w = _scan_linear_verbose(
            ky_values=np.array([0.2, 0.3, 0.4]),
            run_linear_fn=run_cyclone_linear,
            cfg=full_cfg,
            Nl=48,
            Nm=16,
            dt=0.002,
            steps=5000,
            method="imex2",
            solver=CYCLONE_SOLVER,
            krylov_cfg=CYCLONE_KRYLOV,
            window_kw=WINDOWS["cyclone"],
            label=f"Cyclone rho_star={rho:.2f}",
            run_kwargs={"params": params, "terms": LinearTerms()},
            ref=ref,
            verbose=verbose,
            progress=progress,
        )
        rel_g = []
        rel_w = []
        for ky, gamma, omega in zip(scan_ky, scan_g, scan_w):
            idx = int(np.argmin(np.abs(ref.ky - ky)))
            gamma_ref = float(ref.gamma[idx])
            omega_ref = float(ref.omega[idx])
            rel_g.append(abs(float(gamma)) / gamma_ref if gamma_ref != 0.0 else np.nan)
            rel_w.append(abs(float(omega)) / omega_ref if omega_ref != 0.0 else np.nan)
        row = f"{rho:.2f},{np.nanmean(rel_g):.3f},{np.nanmean(rel_w):.3f}"
        rho_rows.append(row)
        _log(f"[Cyclone rho_star] {row}", verbose=verbose, use_tqdm=progress)
    (outdir / "cyclone_rhostar_convergence.csv").write_text(
        "\n".join(rho_rows) + "\n", encoding="utf-8"
    )

    # Mismatch tables against reference data (full ky list) using GX-balanced runs
    gx_cfg = CycloneBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8, y0=20.0, ntheta=32, nperiod=2)
    )
    cyclone_mismatch = _cyclone_gx_scan(
        ref.ky,
        gx_cfg,
        GX_CYCLONE_WINDOW,
        verbose=verbose,
        progress=progress,
    )
    (outdir / "cyclone_mismatch_table.csv").write_text(
        "\n".join(_build_rows(cyclone_mismatch, ref)) + "\n", encoding="utf-8"
    )
    if args.case == "cyclone":
        return 0

    etg_grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28, y0=0.2)
    etg_R = np.array([4.0, 6.0, 8.0, 10.0])
    etg_rows = ["R_over_LTe,gamma,omega"]
    for R in etg_R:
        cfg = ETGBaseCase(grid=etg_grid, model=ETGModelConfig(R_over_LTe=float(R)))
        _log(
            f"\n=== ETG trend R/LTe={float(R):.2f} ===",
            verbose=verbose,
            use_tqdm=progress,
        )
        _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
        res = run_etg_linear(
            cfg=cfg,
            ky_target=5.0,
            Nl=4,
            Nm=8,
            steps=1000,
            dt=0.001,
            solver=ETG_SOLVER,
            krylov_cfg=ETG_KRYLOV,
            mode_method="z_index",
            auto_window=True,
            **WINDOWS["etg"],
        )
        etg_rows.append(f"{R:.2f},{res.gamma:.6f},{res.omega:.6f}")
        _log(
            f"[ETG trend] gamma={res.gamma:.6g} omega={res.omega:.6g}",
            verbose=verbose,
            use_tqdm=progress,
        )
    (outdir / "etg_trend_table.csv").write_text(
        "\n".join(etg_rows) + "\n", encoding="utf-8"
    )

    kinetic_ref = load_cyclone_reference_kinetic()
    kinetic_steps = _scale_steps(kinetic_ref.ky, base_steps=20000, ky_ref=0.3, max_steps=30000)
    kinetic_dt = _scale_dt(kinetic_ref.ky, base_dt=0.0005, ky_ref=0.3)
    kinetic_tmax = kinetic_dt * kinetic_steps
    kinetic_tmin = 0.6 * kinetic_tmax
    kinetic_tmax = 0.95 * kinetic_tmax
    kinetic_cfg = KineticElectronBaseCase(
        grid=GridConfig(Nx=1, Ny=16, Nz=96, Lx=62.8, Ly=62.8, y0=10.0, ntheta=32, nperiod=2)
    )
    kin_ky, kin_g, kin_w = _scan_linear_verbose(
        ky_values=kinetic_ref.ky,
        run_linear_fn=run_kinetic_linear,
        cfg=kinetic_cfg,
        Nl=48,
        Nm=16,
        dt=kinetic_dt,
        steps=kinetic_steps,
        method="imex2",
        solver=KINETIC_SOLVER,
        krylov_cfg=KINETIC_KRYLOV,
        window_kw=WINDOWS["kinetic"],
        tmin=kinetic_tmin,
        tmax=kinetic_tmax,
        auto_window=False,
        run_kwargs={"fit_signal": "phi", "mode_method": "z_index"},
        label="Kinetic ITG mismatch",
        ref=kinetic_ref,
        verbose=verbose,
        progress=progress,
    )
    kinetic_mismatch = LinearScanResult(ky=kin_ky, gamma=kin_g, omega=kin_w)
    (outdir / "kinetic_mismatch_table.csv").write_text(
        "\n".join(_build_rows(kinetic_mismatch, kinetic_ref)) + "\n", encoding="utf-8"
    )

    etg_ref = load_etg_reference()
    etg_dt = _scale_dt(etg_ref.ky, base_dt=0.0002, ky_ref=20.0)
    etg_steps = _scale_steps(etg_ref.ky, base_steps=1200, ky_ref=20.0, max_steps=4000)
    etg_tmax = etg_dt * etg_steps
    etg_tmin = 0.4 * etg_tmax
    etg_tmax = 0.85 * etg_tmax
    etg_cfg = ETGBaseCase()
    etg_ky, etg_g, etg_w = _scan_linear_verbose(
        ky_values=etg_ref.ky,
        run_linear_fn=run_etg_linear,
        cfg=etg_cfg,
        Nl=48,
        Nm=16,
        dt=etg_dt,
        steps=etg_steps,
        method="imex2",
        solver=ETG_SOLVER,
        krylov_cfg=ETG_KRYLOV,
        window_kw=WINDOWS["etg"],
        tmin=etg_tmin,
        tmax=etg_tmax,
        auto_window=False,
        run_kwargs={"mode_method": "z_index"},
        label="ETG mismatch",
        ref=etg_ref,
        verbose=verbose,
        progress=progress,
    )
    etg_mismatch = LinearScanResult(ky=etg_ky, gamma=etg_g, omega=etg_w)
    (outdir / "etg_mismatch_table.csv").write_text(
        "\n".join(_build_rows(etg_mismatch, etg_ref)) + "\n", encoding="utf-8"
    )

    kbm_ref = load_kbm_reference()
    kbm_dt = _scale_dt(kbm_ref.ky, base_dt=0.0005, ky_ref=0.3)
    kbm_steps = _scale_steps(kbm_ref.ky, base_steps=4000, ky_ref=0.3, max_steps=8000)
    kbm_tmax = kbm_dt * kbm_steps
    kbm_tmin = 0.4 * kbm_tmax
    kbm_tmax = 0.8 * kbm_tmax
    kbm_cfg = KBMBaseCase(
        grid=GridConfig(Nx=1, Ny=12, Nz=96, Lx=62.8, Ly=62.8, y0=10.0, ntheta=32, nperiod=2)
    )
    kbm_beta, kbm_g, kbm_w = _scan_kbm_verbose(
        betas=kbm_ref.ky,
        cfg=kbm_cfg,
        Nl=48,
        Nm=16,
        dt=kbm_dt,
        steps=kbm_steps,
        method="imex2",
        solver=KBM_SOLVER,
        krylov_cfg=KBM_KRYLOV,
        window_kw=WINDOWS["kbm"],
        tmin=kbm_tmin,
        tmax=kbm_tmax,
        auto_window=False,
        run_kwargs={"fit_signal": "phi", "mode_method": "z_index"},
        label="KBM mismatch",
        ref=kbm_ref,
        verbose=verbose,
        progress=progress,
    )
    kbm_mismatch = LinearScanResult(ky=kbm_beta, gamma=kbm_g, omega=kbm_w)
    (outdir / "kbm_mismatch_table.csv").write_text(
        "\n".join(_build_rows(kbm_mismatch, kbm_ref)) + "\n", encoding="utf-8"
    )

    tem_ref = load_tem_reference()
    tem_dt = _scale_dt(tem_ref.ky, base_dt=0.001, ky_ref=0.3)
    tem_steps = _scale_steps(tem_ref.ky, base_steps=2000, ky_ref=0.3, max_steps=6000)
    tem_tmax = tem_dt * tem_steps
    tem_tmin = 0.4 * tem_tmax
    tem_tmax = 0.85 * tem_tmax
    tem_cfg = TEMBaseCase()
    tem_ky, tem_g, tem_w = _scan_linear_verbose(
        ky_values=tem_ref.ky,
        run_linear_fn=run_tem_linear,
        cfg=tem_cfg,
        Nl=48,
        Nm=16,
        dt=tem_dt,
        steps=tem_steps,
        method="imex2",
        solver=TEM_SOLVER,
        krylov_cfg=TEM_KRYLOV,
        window_kw=WINDOWS["tem"],
        tmin=tem_tmin,
        tmax=tem_tmax,
        auto_window=False,
        run_kwargs={"fit_signal": "phi", "mode_method": "z_index"},
        label="TEM mismatch",
        ref=tem_ref,
        verbose=verbose,
        progress=progress,
    )
    tem_mismatch = LinearScanResult(ky=tem_ky, gamma=tem_g, omega=tem_w)
    (outdir / "tem_mismatch_table.csv").write_text(
        "\n".join(_build_rows(tem_mismatch, tem_ref)) + "\n", encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
