"""Generate publication-ready figures for docs and README."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from pprint import pformat
import argparse
import sys
from types import SimpleNamespace
from typing import Callable

import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.analysis import (
    ModeSelection,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate_auto,
    select_ky_index,
)
from spectraxgk.benchmarks import (
    CYCLONE_KRYLOV_DEFAULT,
    ETG_KRYLOV_DEFAULT,
    KBM_KRYLOV_DEFAULT,
    KINETIC_KRYLOV_DEFAULT,
    TEM_KRYLOV_DEFAULT,
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
    GX_DAMP_ENDS_AMP,
    GX_DAMP_ENDS_WIDTHFRAC,
    _apply_gx_hypercollisions,
    _build_initial_condition,
    _two_species_params,
    _midplane_index,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_cyclone_reference_gs2,
    load_cyclone_reference_stella,
    load_etg_reference_gs2,
    load_etg_reference_stella,
    load_kbm_reference_gs2,
    load_tem_reference,
    LinearScanResult,
    run_cyclone_linear,
    run_cyclone_scan,
    run_etg_linear,
    run_etg_scan,
    run_kinetic_linear,
    run_kinetic_scan,
    run_kbm_beta_scan,
    run_tem_linear,
    run_tem_scan,
)
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    ETGModelConfig,
    GridConfig,
    KineticElectronBaseCase,
    KBMBaseCase,
    TimeConfig,
    TEMBaseCase,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.gx_integrators import GXTimeConfig, integrate_linear_gx
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache
from spectraxgk.runners import integrate_linear_from_config
from spectraxgk.plotting import (
    cyclone_comparison_figure,
    cyclone_reference_figure,
    linear_validation_figure,
    LinearValidationPanel,
    linear_validation_multi_reference_figure,
    MultiReferenceValidationPanel,
    ReferenceSeries,
    scan_multi_reference_figure,
    scan_comparison_figure,
)
from spectraxgk.linear_krylov import KrylovConfig


def _scale_steps(ky: np.ndarray, base_steps: int, ky_ref: float, max_steps: int) -> np.ndarray:
    scale = ky_ref / np.maximum(ky, 1.0e-6)
    steps = base_steps * np.maximum(1.0, scale)
    return np.clip(steps.astype(int), base_steps, max_steps)


def _scale_dt(ky: np.ndarray, base_dt: float, ky_ref: float) -> np.ndarray:
    scale = np.minimum(1.0, ky_ref / np.maximum(ky, 1.0e-6))
    return base_dt * scale


def _etg_time_controls(
    ky: np.ndarray,
    *,
    base_dt: float = 0.01,
    ky_ref: float = 20.0,
    base_steps: int = 500,
    max_steps: int = 2000,
    tmin_frac: float = 0.2,
    tmax_frac: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = _scale_dt(ky, base_dt=base_dt, ky_ref=ky_ref)
    steps = _scale_steps(ky, base_steps=base_steps, ky_ref=ky_ref, max_steps=max_steps)
    t_total = dt * steps
    tmin = tmin_frac * t_total
    tmax = tmax_frac * t_total
    return dt, steps, tmin, tmax


def _etg_resolution_policy(ky: float) -> tuple[int, int]:
    """Per-ky Hermite/Laguerre resolution for ETG scans."""

    if ky < 10.0:
        return 48, 16
    return 48, 16


def _etg_krylov_policy(ky: float) -> KrylovConfig:
    if ky < 10.0:
        return ETG_KRYLOV_LOW
    return ETG_KRYLOV


CYCLONE_SCAN_SOLVER = "time"
KINETIC_SCAN_SOLVER = "time"
ETG_SCAN_SOLVER = "time"
KBM_SCAN_SOLVER = "time"
TEM_SCAN_SOLVER = "time"
MODE_SOLVER = "time"
MODE_METHOD = "imex2"
DIAGNOSTIC_NORM = "gx"
DEFAULT_RUN_KW = {"diagnostic_norm": DIAGNOSTIC_NORM}
CYCLONE_KRYLOV = CYCLONE_KRYLOV_DEFAULT
KINETIC_KRYLOV = KINETIC_KRYLOV_DEFAULT
ETG_KRYLOV = ETG_KRYLOV_DEFAULT
ETG_KRYLOV_LOW = KrylovConfig(
    method="propagator",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.0,
    omega_target_factor=0.0,
    omega_cap_factor=2.0,
    omega_sign=-1,
    power_iters=80,
    power_dt=0.002,
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=2.0e-3,
)
KBM_KRYLOV = KBM_KRYLOV_DEFAULT
TEM_KRYLOV = TEM_KRYLOV_DEFAULT



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validation figures.")
    parser.add_argument(
        "--case",
        choices=["all", "cyclone", "etg"],
        default="all",
        help="Limit figure generation to a specific case.",
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
    run_kwargs: dict | None = None,
    label: str,
    verbose: bool,
    progress: bool,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], KrylovConfig] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _log(f"\n=== {label} scan ===", verbose=verbose, use_tqdm=progress)
    _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
    _log(
        f"Numerics: Nl={Nl} Nm={Nm} method={method} solver={solver} dt={dt} steps={steps}",
        verbose=verbose,
        use_tqdm=progress,
    )
    _log(f"Window params: {window_kw}", verbose=verbose, use_tqdm=progress)
    if tmin is not None or tmax is not None:
        _log(f"Manual window tmin={tmin} tmax={tmax}", verbose=verbose, use_tqdm=progress)

    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    iterator = tqdm(ky_values, desc=f"{label} ky scan") if progress else ky_values
    for i, ky in enumerate(iterator):
        if resolution_policy is not None:
            Nl_i, Nm_i = resolution_policy(float(ky))
        else:
            Nl_i, Nm_i = int(Nl), int(Nm)
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        tmin_i = _window_value(tmin, i)
        tmax_i = _window_value(tmax, i)
        _log(
            f"[{label}] start ky={float(ky):.4g} dt={dt_i:.4g} steps={steps_i} tmax={dt_i*steps_i:.4g}",
            verbose=verbose,
            use_tqdm=progress,
        )
        krylov_cfg_use = krylov_policy(float(ky)) if krylov_policy is not None else krylov_cfg
        extra = dict(DEFAULT_RUN_KW)
        if run_kwargs:
            extra.update(run_kwargs)
        result = run_linear_fn(
            ky_target=float(ky),
            cfg=cfg,
            Nl=int(Nl_i),
            Nm=int(Nm_i),
            dt=dt_i,
            steps=steps_i,
            method=method,
            solver=solver,
            krylov_cfg=krylov_cfg_use,
            auto_window=auto_window,
            tmin=tmin_i,
            tmax=tmax_i,
            **window_kw,
            **extra,
        )
        gammas.append(float(result.gamma))
        omegas.append(float(result.omega))
        ky_out.append(float(result.ky))
        _log(
            f"[{label}] done ky={float(result.ky):.4g} gamma={result.gamma:.6g} omega={result.omega:.6g}",
            verbose=verbose,
            use_tqdm=progress,
        )

    return np.array(ky_out), np.array(gammas), np.array(omegas)


def _scan_kbm_verbose(
    *,
    betas: np.ndarray,
    cfg,
    Nl: int,
    Nm: int,
    dt: float,
    steps: int,
    method: str,
    solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    label: str,
    run_kwargs: dict | None = None,
    verbose: bool,
    progress: bool,
) -> LinearScanResult:
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
        extra = dict(DEFAULT_RUN_KW)
        if run_kwargs:
            extra.update(run_kwargs)
        result = run_kbm_beta_scan(
            np.asarray([float(beta)]),
            cfg=cfg,
            ky_target=0.3,
            Nl=Nl,
            Nm=Nm,
            steps=steps_i,
            dt=dt_i,
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
        _log(
            f"[{label}] done beta={float(beta):.4g} gamma={gamma:.6g} omega={omega:.6g}",
            verbose=verbose,
            use_tqdm=progress,
        )

    return LinearScanResult(ky=np.array(beta_out), gamma=np.array(gammas), omega=np.array(omegas))

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
        min_points=160,
        start_fraction=0.45,
        growth_weight=0.1,
        require_positive=True,
        min_amp_fraction=0.05,
    ),
    "etg": dict(
        window_fraction=0.25,
        min_points=120,
        start_fraction=0.4,
        growth_weight=0.2,
        require_positive=True,
        min_amp_fraction=0.1,
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


def _run_cyclone_gx_case(
    *,
    ky: float,
    cfg: CycloneBaseCase,
    geom: SAlphaGeometry,
    window_kw: dict,
    verbose: bool,
    progress: bool,
) -> tuple[float, float, np.ndarray, np.ndarray, ModeSelection, float, float, SpectralGrid]:
    Nl, Nm, tmax = _gx_balanced_policy(float(ky))
    grid_full = build_spectral_grid(cfg.grid)
    ky_idx = select_ky_index(np.asarray(grid_full.ky), float(ky))
    grid = select_ky_grid(grid_full, ky_idx)

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
        f"[Cyclone GX] ky={float(ky):.3f} Nl={Nl} Nm={Nm} tmax={tmax}",
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
    mode_method = _gx_mode_policy(float(ky))
    signal = extract_mode_time_series(np.asarray(phi_t), sel, method=mode_method)
    fit_kw = _gx_window_policy(float(ky), window_kw)
    fit_kw = {k: v for k, v in fit_kw.items() if k != "mode_method"}
    gamma, omega, tmin, tmax_fit = fit_growth_rate_auto(np.asarray(t), signal, **fit_kw)
    _log(
        f"[Cyclone GX] ky={float(ky):.3f} method={mode_method} fit=[{tmin:.3g}, {tmax_fit:.3g}] "
        f"gamma={gamma:.6g} omega={omega:.6g}",
        verbose=verbose,
        use_tqdm=progress,
    )
    return gamma, omega, np.asarray(t), np.asarray(phi_t), sel, tmin, tmax_fit, grid


def _cyclone_gx_scan(
    ky_values: np.ndarray,
    cfg: CycloneBaseCase,
    window_kw: dict,
    *,
    verbose: bool,
    progress: bool,
) -> tuple[LinearScanResult, float]:
    geom = SAlphaGeometry.from_config(cfg.geometry)
    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    iterator = tqdm(ky_values, desc="Cyclone GX ky scan") if progress else ky_values
    for ky in iterator:
        gamma, omega, _t, _phi_t, _sel, _tmin, _tmax, _grid = _run_cyclone_gx_case(
            ky=float(ky),
            cfg=cfg,
            geom=geom,
            window_kw=window_kw,
            verbose=verbose,
            progress=progress,
        )
        ky_out.append(float(ky))
        gammas.append(float(gamma))
        omegas.append(float(omega))
    scan = LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))
    ky_sel = float(scan.ky[int(np.nanargmax(scan.gamma))])
    return scan, ky_sel


def _eigenfunction_panel(run, grid, window_kw):
    signal = extract_mode_time_series(run.phi_t, run.selection, method="project")
    if run.t.size < 2:
        tmin = None
        tmax = None
    else:
        _g, _w, tmin, tmax = fit_growth_rate_auto(run.t, signal, **window_kw)
    eig = extract_eigenfunction(
        run.phi_t, run.t, run.selection, z=grid.z, method="snapshot", tmin=tmin, tmax=tmax
    )
    return eig


def _scan_and_mode(
    scan_fn,
    linear_fn,
    ky_values,
    cfg,
    Nl,
    Nm,
    steps,
    dt,
    window_kw,
    *,
    scan_solver: str,
    mode_solver: str,
    mode_method: str,
    verbose: bool,
    progress: bool,
    label: str,
    auto_window: bool = True,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    scan_kwargs: dict | None = None,
    mode_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], KrylovConfig] | None = None,
):
    krylov_cfg = None
    if scan_solver.lower() == "krylov":
        if scan_fn is run_cyclone_scan:
            krylov_cfg = CYCLONE_KRYLOV
        elif scan_fn is run_kinetic_scan:
            krylov_cfg = KINETIC_KRYLOV
        elif scan_fn is run_etg_scan:
            krylov_cfg = ETG_KRYLOV
        elif scan_fn is run_tem_scan:
            krylov_cfg = TEM_KRYLOV
    if verbose or progress:
        scan_ky, scan_g, scan_w = _scan_linear_verbose(
            ky_values=np.asarray(ky_values),
            run_linear_fn=linear_fn,
            cfg=cfg,
            Nl=Nl,
            Nm=Nm,
            dt=dt,
            steps=steps,
            method=mode_method,
            solver=scan_solver,
            krylov_cfg=krylov_cfg,
            window_kw=window_kw,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            label=label,
            run_kwargs=scan_kwargs,
            verbose=verbose,
            progress=progress,
            resolution_policy=resolution_policy,
            krylov_policy=krylov_policy,
        )
        scan = LinearScanResult(ky=scan_ky, gamma=scan_g, omega=scan_w)
    else:
        scan = scan_fn(
            ky_values,
            cfg=cfg,
            Nl=Nl,
            Nm=Nm,
            steps=steps,
            dt=dt,
            method=mode_method,
            solver=scan_solver,
            krylov_cfg=krylov_cfg,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            **window_kw,
            **(scan_kwargs or {}),
        )
    sel_idx = int(np.nanargmax(scan.gamma))
    ky_sel = float(scan.ky[sel_idx])
    if resolution_policy is not None:
        Nl_mode, Nm_mode = resolution_policy(ky_sel)
    else:
        Nl_mode, Nm_mode = int(Nl), int(Nm)
    steps_run = int(steps[sel_idx]) if isinstance(steps, np.ndarray) else int(steps)
    dt_run = float(dt[sel_idx]) if isinstance(dt, np.ndarray) else float(dt)
    _log(
        f"[{label}] eigenfunction ky={ky_sel:.4g} dt={dt_run:.4g} steps={steps_run}",
        verbose=verbose,
        use_tqdm=progress,
    )
    run = linear_fn(
        cfg=cfg,
        ky_target=ky_sel,
        Nl=Nl_mode,
        Nm=Nm_mode,
        steps=steps_run,
        dt=dt_run,
        method=mode_method,
        solver=mode_solver,
        **window_kw,
        **(mode_kwargs or {}),
    )
    grid = build_spectral_grid(cfg.grid)
    mode = _eigenfunction_panel(run, grid, window_kw)
    return scan, mode, grid, ky_sel


def _run_etg_figures(*, outdir: Path, verbose: bool, progress: bool) -> None:
    etg_ref = load_etg_reference_gs2()
    cfg_etg = ETGBaseCase(
        grid=GridConfig(
            Nx=1,
            Ny=24,
            Nz=96,
            Lx=6.28,
            Ly=6.28,
            y0=0.2,
            ntheta=32,
            nperiod=2,
            boundary="linked",
        )
    )
    etg_ky = etg_ref.ky[::2]
    etg_time = TimeConfig(
        t_max=2.4,
        dt=2.0e-4,
        method="imex2",
        use_diffrax=False,
        progress_bar=False,
        sample_stride=10,
    )
    etg_steps = int(round(etg_time.t_max / etg_time.dt))
    etg_scan, _etg_mode, _etg_grid, _ = _scan_and_mode(
        run_etg_scan,
        run_etg_linear,
        etg_ky,
        cfg_etg,
        Nl=24,
        Nm=8,
        steps=etg_steps,
        dt=etg_time.dt,
        window_kw=WINDOWS["etg"],
        scan_solver=ETG_SCAN_SOLVER,
        mode_solver=MODE_SOLVER,
        mode_method=MODE_METHOD,
        auto_window=True,
        tmin=None,
        tmax=None,
        scan_kwargs={
            "fit_signal": "phi",
            "mode_method": "z_index",
            "time_cfg": etg_time,
        },
        mode_kwargs={"fit_signal": "phi", "mode_method": "z_index", "time_cfg": etg_time},
        verbose=verbose,
        progress=progress,
        label="ETG panel",
        resolution_policy=_etg_resolution_policy,
        krylov_policy=_etg_krylov_policy,
    )
    fig, _axes = scan_comparison_figure(
        etg_scan.ky,
        etg_scan.gamma,
        etg_scan.omega,
        r"$k_y \rho_i$",
        "ETG comparison (GS2/stella matched set)",
        x_ref=etg_ref.ky,
        gamma_ref=etg_ref.gamma,
        omega_ref=etg_ref.omega,
        label="SPECTRAX-GK",
        ref_label="Reference",
        log_x=True,
    )
    fig.savefig(outdir / "etg_comparison.png", dpi=200)
    fig.savefig(outdir / "etg_comparison.pdf")


def _load_spectrax_scan_from_mismatch(csv_path: Path, *, x_col: str = "ky") -> LinearScanResult:
    df = pd.read_csv(csv_path).sort_values(x_col)
    return LinearScanResult(
        ky=df[x_col].to_numpy(dtype=float),
        gamma=df["gamma_spectrax"].to_numpy(dtype=float),
        omega=df["omega_spectrax"].to_numpy(dtype=float),
    )


def _load_reference_from_mismatch(csv_path: Path, *, x_col: str) -> LinearScanResult:
    df = pd.read_csv(csv_path).sort_values(x_col)
    return LinearScanResult(
        ky=df[x_col].to_numpy(dtype=float),
        gamma=df["gamma_ref"].to_numpy(dtype=float),
        omega=df["omega_ref"].to_numpy(dtype=float),
    )


def _run_crosscode_figures(*, outdir: Path, verbose: bool, progress: bool) -> None:
    for required in (
        outdir / "cyclone_gs2_mismatch.csv",
        outdir / "etg_gs2_mismatch.csv",
        outdir / "kbm_gs2_mismatch.csv",
    ):
        if not required.exists():
            raise FileNotFoundError(
                f"missing {required}; generate mismatch tables first with compare_gs2_linear.py / "
                "compare_stella_linear.py"
            )

    # Cyclone cross-code references
    cyclone_ref_gx = load_cyclone_reference()
    cyclone_ref_gs2 = load_cyclone_reference_gs2()
    cyclone_ref_stella = load_cyclone_reference_stella()
    cyclone_scan = _load_spectrax_scan_from_mismatch(outdir / "cyclone_gs2_mismatch.csv")
    fig, _axes = scan_multi_reference_figure(
        cyclone_scan.ky,
        cyclone_scan.gamma,
        cyclone_scan.omega,
        x_label=r"$k_y \rho_i$",
        title="Cyclone cross-code comparison",
        references=[
            ReferenceSeries(
                label="GX",
                x=cyclone_ref_gx.ky,
                gamma=cyclone_ref_gx.gamma,
                omega=cyclone_ref_gx.omega,
                color="#1f77b4",
                marker="o",
                linestyle="--",
            ),
            ReferenceSeries(
                label="GS2",
                x=cyclone_ref_gs2.ky,
                gamma=cyclone_ref_gs2.gamma,
                omega=cyclone_ref_gs2.omega,
                color="#ff7f0e",
                marker="s",
                linestyle="--",
            ),
            ReferenceSeries(
                label="stella",
                x=cyclone_ref_stella.ky,
                gamma=cyclone_ref_stella.gamma,
                omega=cyclone_ref_stella.omega,
                color="#9467bd",
                marker="d",
                linestyle="--",
            ),
        ],
        log_x=True,
    )
    fig.savefig(outdir / "cyclone_comparison.png", dpi=200)
    fig.savefig(outdir / "cyclone_comparison.pdf")

    # ETG cross-code references
    etg_ref_gs2 = load_etg_reference_gs2()
    etg_ref_stella = load_etg_reference_stella()
    etg_scan = _load_spectrax_scan_from_mismatch(outdir / "etg_gs2_mismatch.csv")
    kbm_ref_gs2 = load_kbm_reference_gs2()
    kbm_scan = _load_spectrax_scan_from_mismatch(outdir / "kbm_gs2_mismatch.csv", x_col="beta")

    # Representative eigenfunctions for summary panel.
    cfg_cyc = CycloneBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8, y0=20.0, ntheta=32, nperiod=2)
    )
    cyclone_window = {k: v for k, v in WINDOWS["cyclone"].items() if k != "mode_method"}
    run_cyc = run_cyclone_linear(
        cfg=cfg_cyc,
        ky_target=0.3,
        Nl=48,
        Nm=16,
        dt=0.01,
        steps=3000,
        method="imex2",
        solver="time",
        mode_method="z_index",
        diagnostic_norm="gx",
        **cyclone_window,
    )
    grid_cyc = build_spectral_grid(cfg_cyc.grid)
    mode_cyc = _eigenfunction_panel(run_cyc, grid_cyc, cyclone_window)

    cfg_etg = ETGBaseCase(
        grid=GridConfig(
            Nx=1,
            Ny=96,
            Nz=96,
            Lx=6.28,
            Ly=6.28,
            ntheta=32,
            nperiod=2,
            boundary="linked",
        ),
        model=ETGModelConfig(
            R_over_LTi=0.0,
            R_over_LTe=2.49,
            R_over_Ln=0.8,
            R_over_Lni=0.0,
            R_over_Lne=0.8,
            Te_over_Ti=1.0,
            mass_ratio=3670.0,
            nu_i=0.0,
            nu_e=0.0,
            beta=1.0e-5,
            adiabatic_ions=False,
        ),
    )
    geom_etg = SAlphaGeometry.from_config(cfg_etg.geometry)
    params_etg = _two_species_params(
        cfg_etg.model,
        kpar_scale=float(geom_etg.gradpar()),
        omega_d_scale=ETG_OMEGA_D_SCALE,
        omega_star_scale=ETG_OMEGA_STAR_SCALE,
        rho_star=ETG_RHO_STAR,
        nhermite=12,
    )
    terms_etg = LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0)
    run_etg = run_etg_linear(
        cfg=cfg_etg,
        ky_target=20.0,
        Nl=10,
        Nm=12,
        dt=2.0e-4,
        steps=12000,
        method="imex2",
        solver="time",
        params=params_etg,
        terms=terms_etg,
        fit_signal="density",
        mode_method="project",
        gx_growth=True,
        gx_navg_fraction=0.5,
        sample_stride=10,
    )
    grid_etg = build_spectral_grid(cfg_etg.grid)
    mode_etg = _eigenfunction_panel(run_etg, grid_etg, WINDOWS["etg"])

    cfg_kbm = KBMBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8, y0=20.0, ntheta=32, nperiod=2)
    )
    geom_kbm = SAlphaGeometry.from_config(cfg_kbm.geometry)
    params_kbm = _two_species_params(
        cfg_kbm.model,
        kpar_scale=float(geom_kbm.gradpar()),
        omega_d_scale=1.0,
        omega_star_scale=0.8,
        rho_star=1.0,
        beta_override=0.3,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        nhermite=16,
    )
    grid_kbm_full = build_spectral_grid(cfg_kbm.grid)
    ky_idx_kbm = select_ky_index(np.asarray(grid_kbm_full.ky), 0.3)
    grid_kbm = select_ky_grid(grid_kbm_full, ky_idx_kbm)
    cache_kbm = build_linear_cache(grid_kbm, geom_kbm, params_kbm, Nl=6, Nm=16)
    sel_kbm = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid_kbm))
    G0_kbm = np.zeros((2, 6, 16, grid_kbm.ky.size, grid_kbm.kx.size, grid_kbm.z.size), dtype=np.complex64)
    G0_kbm[1] = np.asarray(
        _build_initial_condition(
            grid_kbm,
            geom_kbm,
            ky_index=0,
            kx_index=0,
            Nl=6,
            Nm=16,
            init_cfg=cfg_kbm.init,
        ),
        dtype=np.complex64,
    )
    time_kbm = TimeConfig(
        t_max=6.0,
        dt=5.0e-4,
        method="rk2",
        sample_stride=1,
        checkpoint=False,
        implicit_restart=20,
        implicit_preconditioner=None,
        implicit_solve_method="batched",
        use_diffrax=True,
        diffrax_solver="Dopri8",
        diffrax_adaptive=False,
        diffrax_rtol=1.0e-5,
        diffrax_atol=1.0e-7,
        diffrax_max_steps=32768,
        progress_bar=False,
    )
    _, phi_t_kbm = integrate_linear_from_config(
        jnp.asarray(G0_kbm),
        grid_kbm,
        geom_kbm,
        params_kbm,
        time_kbm,
        cache=cache_kbm,
        terms=LinearTerms(bpar=0.0),
        save_mode=None,
        mode_method="z_index",
    )
    run_kbm = SimpleNamespace(
        t=np.arange(phi_t_kbm.shape[0]) * time_kbm.dt,
        phi_t=np.asarray(phi_t_kbm),
        selection=sel_kbm,
    )
    mode_kbm = _eigenfunction_panel(run_kbm, grid_kbm, WINDOWS["kbm"])

    panels = [
        MultiReferenceValidationPanel(
            name="Cyclone",
            z=grid_cyc.z,
            eigenfunction=mode_cyc,
            x=cyclone_scan.ky,
            gamma=cyclone_scan.gamma,
            omega=cyclone_scan.omega,
            x_label=r"$k_y \rho_i$",
            references=[
                ReferenceSeries(
                    label="GS2",
                    x=cyclone_ref_gs2.ky,
                    gamma=cyclone_ref_gs2.gamma,
                    omega=cyclone_ref_gs2.omega,
                    color="#ff7f0e",
                    marker="s",
                    linestyle="--",
                ),
                ReferenceSeries(
                    label="stella",
                    x=cyclone_ref_stella.ky,
                    gamma=cyclone_ref_stella.gamma,
                    omega=cyclone_ref_stella.omega,
                    color="#9467bd",
                    marker="d",
                    linestyle="--",
                ),
            ],
            log_x=True,
        ),
        MultiReferenceValidationPanel(
            name="ETG",
            z=grid_etg.z,
            eigenfunction=mode_etg,
            x=etg_scan.ky,
            gamma=etg_scan.gamma,
            omega=etg_scan.omega,
            x_label=r"$k_y \rho_i$",
            references=[
                ReferenceSeries(
                    label="GS2",
                    x=etg_ref_gs2.ky,
                    gamma=etg_ref_gs2.gamma,
                    omega=etg_ref_gs2.omega,
                    color="#ff7f0e",
                    marker="s",
                    linestyle="--",
                ),
                ReferenceSeries(
                    label="stella",
                    x=etg_ref_stella.ky,
                    gamma=etg_ref_stella.gamma,
                    omega=etg_ref_stella.omega,
                    color="#9467bd",
                    marker="d",
                    linestyle="--",
                ),
            ],
            log_x=True,
        ),
        MultiReferenceValidationPanel(
            name="KBM",
            z=grid_kbm.z,
            eigenfunction=mode_kbm,
            x=kbm_scan.ky,
            gamma=kbm_scan.gamma,
            omega=kbm_scan.omega,
            x_label=r"$\beta_{ref}$",
            references=[
                ReferenceSeries(
                    label="GS2",
                    x=kbm_ref_gs2.ky,
                    gamma=kbm_ref_gs2.gamma,
                    omega=kbm_ref_gs2.omega,
                    color="#ff7f0e",
                    marker="s",
                    linestyle="--",
                ),
            ],
            log_x=False,
        ),
    ]
    fig, _axes = linear_validation_multi_reference_figure(panels)
    fig.suptitle(
        "Cross-code summary (Cyclone, ETG, KBM): SPECTRAX-GK vs GS2/stella",
        fontsize=12,
        y=1.02,
    )
    fig.text(
        0.01,
        0.01,
        "Cyclone params: q=1.4, s_hat=0.8, eps=0.18, R0=2.77778, R/LTi=2.49, R/Ln=0.8, adiabatic e.\n"
        "ETG params: q=1.5, s_hat=0.8, eps=0.18, R0=3.0, ion R/LTi=0,R/Ln=0; electron R/LTe=2.49,R/Ln=0.8, "
        "kinetic ions/electrons, omega_d*=0.4, omega_*=0.8.\n"
        "KBM params: ky*rho_i=0.3, beta scan, kinetic ions/electrons, A_parallel on, B_parallel off, omega_d*=1.0, omega_*=0.8.",
        fontsize=8,
    )
    fig.savefig(outdir / "linear_summary.png", dpi=200, bbox_inches="tight")
    fig.savefig(outdir / "linear_summary.pdf", bbox_inches="tight")


def main() -> int:
    args = _parse_args()
    verbose = not args.quiet
    progress = not args.no_progress

    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.case == "etg":
        _run_etg_figures(outdir=outdir, verbose=verbose, progress=progress)
        return 0

    # Cyclone reference (adiabatic electrons)
    ref = load_cyclone_reference()
    fig, _axes = cyclone_reference_figure(ref)
    fig.savefig(outdir / "cyclone_reference.png", dpi=200)
    fig.savefig(outdir / "cyclone_reference.pdf")

    cfg_cyc = CycloneBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8, y0=20.0, ntheta=32, nperiod=2)
    )
    scan_ky = np.asarray(ref.ky)
    scan, ky_sel = _cyclone_gx_scan(
        scan_ky,
        cfg_cyc,
        GX_CYCLONE_WINDOW,
        verbose=verbose,
        progress=progress,
    )
    fig, _axes = cyclone_comparison_figure(ref, scan)
    fig.savefig(outdir / "cyclone_comparison.png", dpi=200)
    fig.savefig(outdir / "cyclone_comparison.pdf")
    if args.case == "cyclone":
        return 0

    _run_etg_figures(outdir=outdir, verbose=verbose, progress=progress)
    _run_crosscode_figures(outdir=outdir, verbose=verbose, progress=progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
