"""Generate publication-ready figures for docs and README."""

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

from spectraxgk.analysis import (
    ModeSelection,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate_auto,
    select_ky_index,
)
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
    TEMBaseCase,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.gx_integrators import GXTimeConfig, integrate_linear_gx
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache
from spectraxgk.plotting import (
    cyclone_comparison_figure,
    cyclone_reference_figure,
    linear_validation_figure,
    LinearValidationPanel,
)
from spectraxgk.linear_krylov import KrylovConfig


def _scale_steps(ky: np.ndarray, base_steps: int, ky_ref: float, max_steps: int) -> np.ndarray:
    scale = ky_ref / np.maximum(ky, 1.0e-6)
    steps = base_steps * np.maximum(1.0, scale)
    return np.clip(steps.astype(int), base_steps, max_steps)


def _scale_dt(ky: np.ndarray, base_dt: float, ky_ref: float) -> np.ndarray:
    scale = np.minimum(1.0, ky_ref / np.maximum(ky, 1.0e-6))
    return base_dt * scale


CYCLONE_SCAN_SOLVER = "time"
KINETIC_SCAN_SOLVER = "time"
ETG_SCAN_SOLVER = "time"
KBM_SCAN_SOLVER = "time"
TEM_SCAN_SOLVER = "time"
MODE_SOLVER = "time"
MODE_METHOD = "imex2"
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
    method="shift_invert",
    krylov_dim=16,
    restarts=1,
    power_iters=60,
    power_dt=0.005,
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



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validation figures.")
    parser.add_argument(
        "--case",
        choices=["all", "cyclone"],
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
    label: str,
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
    label: str,
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

    gammas: list[float] = []
    omegas: list[float] = []
    beta_out: list[float] = []
    iterator = tqdm(betas, desc=f"{label} beta scan") if progress else betas
    for beta in iterator:
        _log(
            f"[{label}] start beta={float(beta):.4g} dt={dt:.4g} steps={steps} tmax={dt*steps:.4g}",
            verbose=verbose,
            use_tqdm=progress,
        )
        result = run_kbm_beta_scan(
            np.asarray([float(beta)]),
            cfg=cfg,
            ky_target=0.3,
            Nl=Nl,
            Nm=Nm,
            steps=steps,
            dt=dt,
            method=method,
            solver=solver,
            krylov_cfg=krylov_cfg,
            **window_kw,
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
    scan_kwargs: dict | None = None,
    mode_kwargs: dict | None = None,
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
            label=label,
            run_kwargs=scan_kwargs,
            verbose=verbose,
            progress=progress,
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
            **window_kw,
            **(scan_kwargs or {}),
        )
    sel_idx = int(np.nanargmax(scan.gamma))
    ky_sel = float(scan.ky[sel_idx])
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
        Nl=Nl,
        Nm=Nm,
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


def main() -> int:
    args = _parse_args()
    verbose = not args.quiet
    progress = not args.no_progress

    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)

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

    # Multi-panel summary: cyclone, kinetic ITG, ETG, KBM, TEM
    geom_cyc = SAlphaGeometry.from_config(cfg_cyc.geometry)
    gamma_sel, omega_sel, t_sel, phi_sel, sel, tmin_sel, tmax_sel, grid_sel = _run_cyclone_gx_case(
        ky=ky_sel,
        cfg=cfg_cyc,
        geom=geom_cyc,
        window_kw=GX_CYCLONE_WINDOW,
        verbose=verbose,
        progress=progress,
    )
    cyclone_scan = scan
    cyclone_grid = grid_sel
    cyclone_mode = extract_eigenfunction(
        phi_sel, t_sel, sel, z=grid_sel.z, method="snapshot", tmin=tmin_sel, tmax=tmax_sel
    )

    kinetic_ref = load_cyclone_reference_kinetic()
    cfg_kin = KineticElectronBaseCase(
        grid=GridConfig(Nx=1, Ny=12, Nz=96, Lx=62.8, Ly=62.8, y0=10.0, ntheta=32, nperiod=2)
    )
    kinetic_ky = kinetic_ref.ky[::2]
    kinetic_steps = _scale_steps(kinetic_ky, base_steps=80000, ky_ref=0.3, max_steps=120000)
    kinetic_dt = _scale_dt(kinetic_ky, base_dt=0.0005, ky_ref=0.3)
    kinetic_tmax = kinetic_dt * kinetic_steps
    kinetic_tmin = 0.6 * kinetic_tmax
    kinetic_tmax_fit = 0.95 * kinetic_tmax
    kinetic_scan, kinetic_mode, kinetic_grid, _ = _scan_and_mode(
        run_kinetic_scan,
        run_kinetic_linear,
        kinetic_ky,
        cfg_kin,
        Nl=48,
        Nm=16,
        steps=kinetic_steps,
        dt=kinetic_dt,
        window_kw=WINDOWS["kinetic"],
        scan_solver=KINETIC_SCAN_SOLVER,
        mode_solver=MODE_SOLVER,
        mode_method=MODE_METHOD,
        scan_kwargs={"tmin": kinetic_tmin, "tmax": kinetic_tmax_fit, "auto_window": False},
        verbose=verbose,
        progress=progress,
        label="Kinetic ITG panel",
    )

    etg_ref = load_etg_reference()
    cfg_etg = ETGBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=6.28, Ly=6.28, y0=0.2, ntheta=32, nperiod=2)
    )
    etg_ky = etg_ref.ky[::2]
    etg_dt = _scale_dt(etg_ky, base_dt=0.0002, ky_ref=20.0)
    etg_steps = _scale_steps(etg_ky, base_steps=1200, ky_ref=20.0, max_steps=4000)
    etg_tmax = etg_dt * etg_steps
    etg_tmin = 0.4 * etg_tmax
    etg_tmax_fit = 0.85 * etg_tmax
    etg_scan, etg_mode, etg_grid, _ = _scan_and_mode(
        run_etg_scan,
        run_etg_linear,
        etg_ky,
        cfg_etg,
        Nl=48,
        Nm=16,
        steps=etg_steps,
        dt=etg_dt,
        window_kw=WINDOWS["etg"],
        scan_solver=ETG_SCAN_SOLVER,
        mode_solver=MODE_SOLVER,
        mode_method=MODE_METHOD,
        scan_kwargs={"tmin": etg_tmin, "tmax": etg_tmax_fit, "auto_window": False},
        verbose=verbose,
        progress=progress,
        label="ETG panel",
    )

    kbm_ref = load_kbm_reference()
    cfg_kbm = KBMBaseCase(
        grid=GridConfig(Nx=1, Ny=9, Nz=96, Lx=62.8, Ly=62.8, y0=10.0, ntheta=32, nperiod=2)
    )
    kbm_beta = kbm_ref.ky[::2]
    kbm_dt = _scale_dt(kbm_beta, base_dt=0.0005, ky_ref=0.3)
    kbm_steps = _scale_steps(kbm_beta, base_steps=4000, ky_ref=0.3, max_steps=8000)
    kbm_tmax = kbm_dt * kbm_steps
    kbm_tmin = 0.4 * kbm_tmax
    kbm_tmax = 0.8 * kbm_tmax
    kbm_scan = _scan_kbm_verbose(
        betas=kbm_beta,
        cfg=cfg_kbm,
        Nl=48,
        Nm=16,
        steps=kbm_steps,
        dt=kbm_dt,
        method=MODE_METHOD,
        solver=KBM_SCAN_SOLVER,
        krylov_cfg=KBM_KRYLOV,
        window_kw=WINDOWS["kbm"],
        tmin=kbm_tmin,
        tmax=kbm_tmax,
        auto_window=False,
        label="KBM panel",
        verbose=verbose,
        progress=progress,
    )
    kbm_idx = int(np.argmin(np.abs(kbm_beta - 0.3)))
    kbm_steps_run = int(kbm_steps[kbm_idx])
    kbm_dt_run = float(kbm_dt[kbm_idx])
    _log(
        f"[KBM panel] eigenfunction ky=0.3 dt={kbm_dt_run:.4g} steps={kbm_steps_run}",
        verbose=verbose,
        use_tqdm=progress,
    )
    kbm_run = run_kinetic_linear(
        cfg=cfg_kbm,
        ky_target=0.3,
        Nl=48,
        Nm=16,
        steps=kbm_steps_run,
        dt=kbm_dt_run,
        method=MODE_METHOD,
        solver=MODE_SOLVER,
        **WINDOWS["kbm"],
    )
    kbm_grid = build_spectral_grid(cfg_kbm.grid)
    kbm_mode = _eigenfunction_panel(kbm_run, kbm_grid, WINDOWS["kbm"])

    tem_ref = load_tem_reference()
    cfg_tem = TEMBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=160, Lx=62.8, Ly=62.8, y0=20.0, ntheta=32, nperiod=3)
    )
    tem_ky = tem_ref.ky[::2]
    tem_dt = _scale_dt(tem_ky, base_dt=0.001, ky_ref=0.3)
    tem_steps = _scale_steps(tem_ky, base_steps=2000, ky_ref=0.3, max_steps=6000)
    tem_tmax = tem_dt * tem_steps
    tem_tmin = 0.4 * tem_tmax
    tem_tmax_fit = 0.85 * tem_tmax
    tem_scan, tem_mode, tem_grid, _ = _scan_and_mode(
        run_tem_scan,
        run_tem_linear,
        tem_ky,
        cfg_tem,
        Nl=48,
        Nm=16,
        steps=tem_steps,
        dt=tem_dt,
        window_kw=WINDOWS["tem"],
        scan_solver=TEM_SCAN_SOLVER,
        mode_solver=MODE_SOLVER,
        mode_method=MODE_METHOD,
        scan_kwargs={"tmin": tem_tmin, "tmax": tem_tmax_fit, "auto_window": False},
        verbose=verbose,
        progress=progress,
        label="TEM panel",
    )

    panels = [
        LinearValidationPanel(
            name="Cyclone",
            z=cyclone_grid.z,
            eigenfunction=cyclone_mode,
            x=cyclone_scan.ky,
            gamma=cyclone_scan.gamma,
            omega=cyclone_scan.omega,
            x_label=r"$k_y \rho_i$",
            x_ref=ref.ky,
            gamma_ref=ref.gamma,
            omega_ref=ref.omega,
            ref_label="Reference",
            log_x=True,
        ),
        LinearValidationPanel(
            name="Kinetic ITG",
            z=kinetic_grid.z,
            eigenfunction=kinetic_mode,
            x=kinetic_scan.ky,
            gamma=kinetic_scan.gamma,
            omega=kinetic_scan.omega,
            x_label=r"$k_y \rho_i$",
            x_ref=kinetic_ref.ky,
            gamma_ref=kinetic_ref.gamma,
            omega_ref=kinetic_ref.omega,
            ref_label="Reference",
            log_x=True,
        ),
        LinearValidationPanel(
            name="ETG",
            z=etg_grid.z,
            eigenfunction=etg_mode,
            x=etg_scan.ky,
            gamma=etg_scan.gamma,
            omega=etg_scan.omega,
            x_label=r"$k_y \rho_i$",
            x_ref=etg_ref.ky,
            gamma_ref=etg_ref.gamma,
            omega_ref=etg_ref.omega,
            ref_label="Reference",
            log_x=True,
        ),
        LinearValidationPanel(
            name="KBM",
            z=kbm_grid.z,
            eigenfunction=kbm_mode,
            x=kbm_scan.ky,
            gamma=kbm_scan.gamma,
            omega=kbm_scan.omega,
            x_label=r"$\beta_{ref}$",
            x_ref=kbm_ref.ky,
            gamma_ref=kbm_ref.gamma,
            omega_ref=kbm_ref.omega,
            ref_label="Reference",
        ),
        LinearValidationPanel(
            name="TEM",
            z=tem_grid.z,
            eigenfunction=tem_mode,
            x=tem_scan.ky,
            gamma=tem_scan.gamma,
            omega=tem_scan.omega,
            x_label=r"$k_y \rho_s$",
            x_ref=tem_ref.ky,
            gamma_ref=tem_ref.gamma,
            omega_ref=tem_ref.omega,
            ref_label="Reference",
            log_x=True,
        ),
    ]
    fig, _axes = linear_validation_figure(panels)
    fig.savefig(outdir / "linear_summary.png", dpi=200)
    fig.savefig(outdir / "linear_summary.pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
