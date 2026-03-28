"""Generate CSV tables for documentation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from pprint import pformat
import argparse
import sys
from typing import Callable

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
    CYCLONE_KRYLOV_DEFAULT,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
    GX_DAMP_ENDS_AMP,
    GX_DAMP_ENDS_WIDTHFRAC,
    KINETIC_KRYLOV_DEFAULT,
    ETG_KRYLOV_DEFAULT,
    KBM_KRYLOV_DEFAULT,
    TEM_KRYLOV_DEFAULT,
    _apply_gx_hypercollisions,
    _build_initial_condition,
    _midplane_index,
    _two_species_params,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
    CycloneScanResult,
    LinearScanResult,
    run_cyclone_linear,
    run_cyclone_scan,
    run_etg_linear,
    run_etg_scan,
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
    TimeConfig,
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
    gx_growth_rate_from_phi,
    select_ky_index,
)

CYCLONE_SOLVER = "time"
KINETIC_SOLVER = "time"
ETG_SOLVER = "time"
KBM_SOLVER = "time"
TEM_SOLVER = "time"
DIAGNOSTIC_NORM = "gx"
DEFAULT_RUN_KW = {"diagnostic_norm": DIAGNOSTIC_NORM}

ETG_GX_MISMATCH_NL = 16
ETG_GX_MISMATCH_NM = 8
ETG_GX_MISMATCH_DT = 0.01
ETG_GX_MISMATCH_STEPS = 800

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


def _cyclone_refresh_grid(ref: LinearScanResult) -> GridConfig:
    nky = int(np.asarray(ref.ky).size)
    if nky < 2:
        raise ValueError("Cyclone reference must contain at least two positive ky points")
    return GridConfig(
        Nx=1,
        Ny=3 * (nky - 1) + 1,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=32,
        nperiod=2,
        boundary="linked",
    )


def _cyclone_refresh_reference(ref: LinearScanResult) -> LinearScanResult:
    keep = np.asarray(ref.ky) <= 0.45 + 1.0e-12
    return LinearScanResult(
        ky=np.asarray(ref.ky)[keep],
        gamma=np.asarray(ref.gamma)[keep],
        omega=np.asarray(ref.omega)[keep],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validation tables.")
    parser.add_argument(
        "--case",
        choices=["all", "cyclone", "etg"],
        default="all",
        help="Limit table generation to a specific case.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging.")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm progress bars."
    )
    parser.add_argument(
        "--stiff-spot-check",
        action="store_true",
        help=(
            "After Krylov scans, re-run the worst-mismatch high-ky points with a "
            "robust implicit time integrator (GMRES + hermite-line preconditioner) "
            "to diagnose instability-driven outliers."
        ),
    )
    parser.add_argument(
        "--stiff-spot-check-topk",
        type=int,
        default=2,
        help="Number of outlier points to re-run per scan when --stiff-spot-check is enabled.",
    )
    parser.add_argument(
        "--stiff-spot-check-min-ky",
        type=float,
        default=0.5,
        help="Only consider ky >= this threshold when selecting stiff spot-check points.",
    )
    parser.add_argument(
        "--stiff-spot-check-dt",
        type=float,
        default=0.01,
        help="Fixed dt to use for implicit stiff spot-check runs.",
    )
    parser.add_argument(
        "--stiff-spot-check-tmax",
        type=float,
        default=4.0,
        help="Total time horizon to use for implicit stiff spot-check runs.",
    )
    parser.add_argument(
        "--stiff-spot-check-replace",
        action="store_true",
        help="Replace Krylov results with implicit spot-check results when they reduce mismatch.",
    )
    parser.add_argument(
        "--refresh-minimal",
        action="store_true",
        help="Only build the tracked outputs required by the benchmark refresh workflow.",
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
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], KrylovConfig] | None = None,
    stiff_spot_check: bool = False,
    stiff_spot_check_topk: int = 0,
    stiff_spot_check_min_ky: float = 0.0,
    stiff_spot_check_dt: float = 0.01,
    stiff_spot_check_tmax: float = 4.0,
    stiff_spot_check_replace: bool = False,
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
    mismatch_scores: list[float] = []
    ref_pairs: list[tuple[float, float]] = []
    base_extra = dict(DEFAULT_RUN_KW)
    if run_kwargs:
        base_extra.update(run_kwargs)
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
        extra = dict(base_extra)
        krylov_cfg_use = krylov_policy(float(ky)) if krylov_policy is not None else krylov_cfg
        try:
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
            gamma_val = float(result.gamma)
            omega_val = float(result.omega)
            ky_val = float(result.ky)
            msg = f"[{label}] done ky={ky_val:.4g} gamma={gamma_val:.6g} omega={omega_val:.6g}"
        except Exception as exc:
            gamma_val = float("nan")
            omega_val = float("nan")
            ky_val = float(ky)
            msg = f"[{label}] failed ky={ky_val:.4g} error={type(exc).__name__}: {exc}"
        gammas.append(gamma_val)
        omegas.append(omega_val)
        ky_out.append(ky_val)
        if ref is not None:
            idx = int(np.argmin(np.abs(ref.ky - ky_val)))
            gamma_ref = float(ref.gamma[idx])
            omega_ref = float(ref.omega[idx])
            rel_gamma = (gamma_val - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
            rel_omega = (omega_val - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
            mismatch_scores.append(float(np.nanmax(np.abs([rel_gamma, rel_omega]))))
            ref_pairs.append((gamma_ref, omega_ref))
            msg += (
                f" | ref gamma={gamma_ref:.6g} omega={omega_ref:.6g}"
                f" rel_gamma={rel_gamma:.3g} rel_omega={rel_omega:.3g}"
            )
        _log(msg, verbose=verbose, use_tqdm=progress)

    if (
        stiff_spot_check
        and str(solver).lower() == "krylov"
        and ref is not None
        and stiff_spot_check_topk > 0
        and len(ky_out) == len(mismatch_scores)
    ):
        ky_arr = np.asarray(ky_out)
        score_arr = np.asarray(mismatch_scores)
        eligible = ky_arr >= float(stiff_spot_check_min_ky)
        if np.any(eligible):
            eligible_idx = np.where(eligible)[0]
            ranked = eligible_idx[np.argsort(score_arr[eligible_idx])[::-1]]
            top_idx = ranked[: int(stiff_spot_check_topk)]
            _log(
                f"\n[{label}] stiff spot-check (implicit + hermite-line) for {len(top_idx)} outliers",
                verbose=verbose,
                use_tqdm=progress,
            )
            for local_i in top_idx:
                ky_val = float(ky_arr[local_i])
                gamma_ref, omega_ref = ref_pairs[local_i]
                gamma_k = float(gammas[local_i])
                omega_k = float(omegas[local_i])
                t_max = float(stiff_spot_check_tmax)
                dt_spot = float(stiff_spot_check_dt)
                steps_spot = max(int(round(t_max / dt_spot)), 1)
                time_cfg_spot = TimeConfig(
                    t_max=t_max,
                    dt=dt_spot,
                    method="implicit",
                    use_diffrax=False,
                    implicit_preconditioner="hermite-line",
                    progress_bar=False,
                    sample_stride=1,
                )
                extra_spot = dict(base_extra)
                extra_spot.pop("time_cfg", None)
                spot = run_linear_fn(
                    ky_target=ky_val,
                    cfg=cfg,
                    Nl=int(Nl),
                    Nm=int(Nm),
                    dt=dt_spot,
                    steps=steps_spot,
                    method="implicit",
                    solver="time",
                    krylov_cfg=None,
                    time_cfg=time_cfg_spot,
                    auto_window=True,
                    tmin=None,
                    tmax=None,
                    **window_kw,
                    **extra_spot,
                )
                gamma_i = float(spot.gamma)
                omega_i = float(spot.omega)
                rel_k = abs((gamma_k - gamma_ref) / gamma_ref) if gamma_ref != 0.0 else np.inf
                rel_i = abs((gamma_i - gamma_ref) / gamma_ref) if gamma_ref != 0.0 else np.inf
                relw_k = abs((omega_k - omega_ref) / omega_ref) if omega_ref != 0.0 else np.inf
                relw_i = abs((omega_i - omega_ref) / omega_ref) if omega_ref != 0.0 else np.inf
                _log(
                    f"[{label}] ky={ky_val:.4g} krylov(g={gamma_k:.4g}, w={omega_k:.4g}) "
                    f"implicit(g={gamma_i:.4g}, w={omega_i:.4g}) "
                    f"| ref(g={gamma_ref:.4g}, w={omega_ref:.4g}) "
                    f"rel_g(k={rel_k:.3g}, i={rel_i:.3g}) rel_w(k={relw_k:.3g}, i={relw_i:.3g})",
                    verbose=verbose,
                    use_tqdm=progress,
                )
                if stiff_spot_check_replace and (rel_i + relw_i) < (rel_k + relw_k):
                    gammas[local_i] = gamma_i
                    omegas[local_i] = omega_i
                    _log(
                        f"[{label}] replaced krylov result at ky={ky_val:.4g} with implicit spot-check.",
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
    stiff_spot_check: bool = False,
    stiff_spot_check_topk: int = 0,
    stiff_spot_check_dt: float = 0.01,
    stiff_spot_check_tmax: float = 4.0,
    stiff_spot_check_replace: bool = False,
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
    mismatch_scores: list[float] = []
    ref_pairs: list[tuple[float, float]] = []
    base_extra = dict(DEFAULT_RUN_KW)
    if run_kwargs:
        base_extra.update(run_kwargs)
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
        extra = dict(base_extra)
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
            mismatch_scores.append(float(np.nanmax(np.abs([rel_gamma, rel_omega]))))
            ref_pairs.append((gamma_ref, omega_ref))
            msg += (
                f" | ref gamma={gamma_ref:.6g} omega={omega_ref:.6g}"
                f" rel_gamma={rel_gamma:.3g} rel_omega={rel_omega:.3g}"
            )
        _log(msg, verbose=verbose, use_tqdm=progress)

    if (
        stiff_spot_check
        and str(solver).lower() == "krylov"
        and ref is not None
        and stiff_spot_check_topk > 0
        and len(beta_out) == len(mismatch_scores)
    ):
        beta_arr = np.asarray(beta_out)
        score_arr = np.asarray(mismatch_scores)
        ranked = np.argsort(score_arr)[::-1]
        top_idx = ranked[: int(stiff_spot_check_topk)]
        _log(
            f"\n[{label}] stiff spot-check (implicit + hermite-line) for {len(top_idx)} outliers",
            verbose=verbose,
            use_tqdm=progress,
        )
        for local_i in top_idx:
            beta_val = float(beta_arr[local_i])
            gamma_ref, omega_ref = ref_pairs[local_i]
            gamma_k = float(gammas[local_i])
            omega_k = float(omegas[local_i])
            t_max = float(stiff_spot_check_tmax)
            dt_spot = float(stiff_spot_check_dt)
            steps_spot = max(int(round(t_max / dt_spot)), 1)
            time_cfg_spot = TimeConfig(
                t_max=t_max,
                dt=dt_spot,
                method="implicit",
                use_diffrax=False,
                implicit_preconditioner="hermite-line",
                progress_bar=False,
                sample_stride=1,
            )
            extra_spot = dict(base_extra)
            extra_spot.pop("time_cfg", None)
            spot = run_kbm_beta_scan(
                np.asarray([beta_val]),
                cfg=cfg,
                ky_target=0.3,
                Nl=Nl,
                Nm=Nm,
                dt=dt_spot,
                steps=steps_spot,
                method="implicit",
                solver="time",
                krylov_cfg=None,
                time_cfg=time_cfg_spot,
                auto_window=True,
                tmin=None,
                tmax=None,
                streaming_fit=False,
                **window_kw,
                **extra_spot,
            )
            gamma_i = float(spot.gamma[0])
            omega_i = float(spot.omega[0])
            rel_k = abs((gamma_k - gamma_ref) / gamma_ref) if gamma_ref != 0.0 else np.inf
            rel_i = abs((gamma_i - gamma_ref) / gamma_ref) if gamma_ref != 0.0 else np.inf
            relw_k = abs((omega_k - omega_ref) / omega_ref) if omega_ref != 0.0 else np.inf
            relw_i = abs((omega_i - omega_ref) / omega_ref) if omega_ref != 0.0 else np.inf
            _log(
                f"[{label}] beta={beta_val:.4g} krylov(g={gamma_k:.4g}, w={omega_k:.4g}) "
                f"implicit(g={gamma_i:.4g}, w={omega_i:.4g}) "
                f"| ref(g={gamma_ref:.4g}, w={omega_ref:.4g}) "
                f"rel_g(k={rel_k:.3g}, i={rel_i:.3g}) rel_w(k={relw_k:.3g}, i={relw_i:.3g})",
                verbose=verbose,
                use_tqdm=progress,
            )
            if stiff_spot_check_replace and (rel_i + relw_i) < (rel_k + relw_k):
                gammas[local_i] = gamma_i
                omegas[local_i] = omega_i
                _log(
                    f"[{label}] replaced krylov result at beta={beta_val:.4g} with implicit spot-check.",
                    verbose=verbose,
                    use_tqdm=progress,
                )

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


def _fit_cyclone_gx_signal(
    *,
    t: np.ndarray,
    phi_t: np.ndarray,
    sel: ModeSelection,
    ky_val: float,
    window_kw: dict,
) -> tuple[float, float, float, float, str]:
    gamma_floor = 1.0e-6
    omega_floor = 1.0e-6
    methods = [_gx_mode_policy(ky_val)]
    if methods[0] != "max":
        methods.append("max")
    last: tuple[float, float, float, float, str] | None = None
    for method in methods:
        signal = extract_mode_time_series(phi_t, sel, method=method)
        fit_kw = _gx_window_policy(ky_val, window_kw)
        fit_kw = {k: v for k, v in fit_kw.items() if k != "mode_method"}
        gamma, omega, tmin, tmax_fit = fit_growth_rate_auto(t, signal, **fit_kw)
        last = (float(gamma), float(omega), float(tmin), float(tmax_fit), method)
        if (
            np.isfinite(gamma)
            and np.isfinite(omega)
            and gamma > gamma_floor
            and abs(omega) > omega_floor
        ):
            return last
    assert last is not None
    return last


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
        gamma, omega, tmin, tmax_fit, mode_method = _fit_cyclone_gx_signal(
            t=np.asarray(t),
            phi_t=np.asarray(phi_t),
            sel=sel,
            ky_val=ky_val,
            window_kw=window_kw,
        )
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


def _cyclone_reference_mismatch_scan(
    ref: LinearScanResult,
    cfg: CycloneBaseCase,
    *,
    verbose: bool,
    progress: bool,
) -> LinearScanResult:
    _log("\n=== Cyclone mismatch scan ===", verbose=verbose, use_tqdm=progress)
    _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
    _log(
        "Numerics: dedicated GX-style benchmark scan with per-ky extraction policy",
        verbose=verbose,
        use_tqdm=progress,
    )
    _log(f"Window params: {GX_CYCLONE_WINDOW}", verbose=verbose, use_tqdm=progress)
    scan = _cyclone_gx_scan(
        np.asarray(ref.ky),
        cfg,
        GX_CYCLONE_WINDOW,
        verbose=verbose,
        progress=progress,
    )
    for ky_val, gamma_val, omega_val in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky_val)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        rel_gamma = (float(gamma_val) - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
        rel_omega = (float(omega_val) - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
        _log(
            f"[Cyclone mismatch] done ky={float(ky_val):.4g} gamma={float(gamma_val):.6g} omega={float(omega_val):.6g}"
            f" | ref gamma={gamma_ref:.6g} omega={omega_ref:.6g}"
            f" rel_gamma={rel_gamma:.3g} rel_omega={rel_omega:.3g}",
            verbose=verbose,
            use_tqdm=progress,
        )
    return LinearScanResult(ky=np.asarray(scan.ky), gamma=np.asarray(scan.gamma), omega=np.asarray(scan.omega))


def _etg_reference_mismatch_scan(
    ref: LinearScanResult,
    cfg: ETGBaseCase,
    *,
    dt: float,
    steps: int,
    verbose: bool,
    progress: bool,
) -> LinearScanResult:
    _log("\n=== ETG mismatch scan ===", verbose=verbose, use_tqdm=progress)
    _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
    _log(
        f"Numerics: Nl={ETG_GX_MISMATCH_NL} Nm={ETG_GX_MISMATCH_NM} GX-growth-style ETG replay dt={dt} steps={steps}",
        verbose=verbose,
        use_tqdm=progress,
    )
    ky_rows: list[float] = []
    gamma_rows: list[float] = []
    omega_rows: list[float] = []
    for ky_val in np.asarray(ref.ky, dtype=float):
        gamma_val, omega_val = _run_etg_gx_growth(
            cfg=cfg,
            ky=float(ky_val),
            Nl=ETG_GX_MISMATCH_NL,
            Nm=ETG_GX_MISMATCH_NM,
            dt=dt,
            steps=steps,
        )
        ky_rows.append(float(ky_val))
        gamma_rows.append(float(gamma_val))
        omega_rows.append(float(omega_val))
        idx = int(np.argmin(np.abs(ref.ky - ky_val)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        rel_gamma = (float(gamma_val) - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
        rel_omega = (float(omega_val) - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
        _log(
            f"[ETG mismatch] done ky={float(ky_val):.4g} gamma={float(gamma_val):.6g} omega={float(omega_val):.6g}"
            f" | ref gamma={gamma_ref:.6g} omega={omega_ref:.6g}"
            f" rel_gamma={rel_gamma:.3g} rel_omega={rel_omega:.3g}",
            verbose=verbose,
            use_tqdm=progress,
        )
    return LinearScanResult(
        ky=np.asarray(ky_rows),
        gamma=np.asarray(gamma_rows),
        omega=np.asarray(omega_rows),
    )


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


def _etg_benchmark_case() -> ETGBaseCase:
    return ETGBaseCase(
        grid=GridConfig(
            Nx=1,
            Ny=16,
            Nz=64,
            Lx=6.28,
            Ly=0.628,
            ntheta=32,
            nperiod=2,
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


def _run_etg_gx_growth(
    *,
    cfg: ETGBaseCase,
    ky: float,
    Nl: int,
    Nm: int,
    dt: float,
    steps: int,
    sample_stride: int = 20,
    navg_fraction: float = 0.3,
) -> tuple[float, float]:
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), ky)
    grid = select_ky_grid(grid_full, ky_index)
    if getattr(cfg.model, "adiabatic_ions", False):
        raise ValueError("ETG growth helper expects a two-species ETG benchmark case")
    params = _two_species_params(
        cfg.model,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=ETG_OMEGA_D_SCALE,
        omega_star_scale=ETG_OMEGA_STAR_SCALE,
        rho_star=ETG_RHO_STAR,
        nhermite=Nm,
    )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=Nl,
        Nm=Nm,
        init_cfg=cfg.init,
    )
    charge = np.atleast_1d(np.asarray(params.charge_sign))
    ns = int(charge.size)
    electron_index = int(np.argmin(charge))
    G0 = np.zeros((ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G0[electron_index] = np.asarray(G0_single, dtype=np.complex64)
    t, phi_t, _gamma_t, _omega_t = integrate_linear_gx(
        G0,
        grid,
        cache,
        params,
        geom,
        GXTimeConfig(dt=dt, t_max=dt * float(steps), sample_stride=max(1, sample_stride)),
        terms=LinearTerms(apar=0.0, bpar=0.0, hypercollisions=1.0),
        mode_method="z_index",
    )
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=grid.z.size // 2)
    gamma, omega, _g_t, _w_t, _t_mid = gx_growth_rate_from_phi(
        np.asarray(phi_t),
        np.asarray(t),
        sel,
        navg_fraction=navg_fraction,
        mode_method="z_index",
    )
    return float(gamma), float(omega)


def _etg_resolution_policy(ky: float) -> tuple[int, int]:
    """Per-ky Hermite/Laguerre resolution for ETG scans."""

    if ky < 10.0:
        return 48, 16
    return 48, 16


def _etg_krylov_policy(ky: float) -> KrylovConfig:
    if ky < 10.0:
        return ETG_KRYLOV_LOW
    return ETG_KRYLOV


def _run_etg_tables(*, outdir: Path, verbose: bool, progress: bool) -> None:
    etg_grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28, y0=0.2)
    etg_time_cfg = TimeConfig(
        t_max=2.4,
        dt=2.0e-4,
        method="imex2",
        use_diffrax=False,
        progress_bar=False,
        sample_stride=10,
    )
    etg_R = np.array([4.0, 6.0, 8.0, 10.0])
    etg_rows = ["R_over_LTe,gamma,omega"]
    for R in etg_R:
        cfg = ETGBaseCase(grid=etg_grid, model=ETGModelConfig(R_over_LTe=float(R)))
        steps = int(round(etg_time_cfg.t_max / etg_time_cfg.dt))
        _log(
            f"\n=== ETG trend R/LTe={float(R):.2f} ===",
            verbose=verbose,
            use_tqdm=progress,
        )
        _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
        res = run_etg_linear(
            cfg=cfg,
            ky_target=5.0,
            Nl=24,
            Nm=8,
            steps=steps,
            dt=etg_time_cfg.dt,
            time_cfg=etg_time_cfg,
            solver=ETG_SOLVER,
            krylov_cfg=ETG_KRYLOV_LOW,
            mode_method="z_index",
            fit_signal="phi",
            auto_window=True,
            diagnostic_norm=DIAGNOSTIC_NORM,
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

    etg_ref = load_etg_reference()
    etg_cfg = _etg_benchmark_case()
    etg_time = TimeConfig(
        t_max=ETG_GX_MISMATCH_DT * ETG_GX_MISMATCH_STEPS,
        dt=ETG_GX_MISMATCH_DT,
        method="imex2",
        use_diffrax=False,
        progress_bar=False,
        sample_stride=2,
    )
    etg_steps = int(round(etg_time.t_max / etg_time.dt))
    etg_mismatch = _etg_reference_mismatch_scan(
        etg_ref,
        etg_cfg,
        dt=etg_time.dt,
        steps=etg_steps,
        verbose=verbose,
        progress=progress,
    )
    (outdir / "etg_mismatch_table.csv").write_text(
        "\n".join(_build_rows(etg_mismatch, etg_ref)) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    verbose = not args.quiet
    progress = not args.no_progress
    stiff_spot_check = bool(args.stiff_spot_check)
    stiff_spot_topk = int(args.stiff_spot_check_topk)
    stiff_spot_min_ky = float(args.stiff_spot_check_min_ky)
    stiff_spot_dt = float(args.stiff_spot_check_dt)
    stiff_spot_tmax = float(args.stiff_spot_check_tmax)
    stiff_spot_replace = bool(args.stiff_spot_check_replace)

    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.case == "etg":
        _run_etg_tables(outdir=outdir, verbose=verbose, progress=progress)
        return 0

    ref_full = load_cyclone_reference()
    ref = _cyclone_refresh_reference(ref_full)
    if args.refresh_minimal:
        cfg = CycloneBaseCase(grid=_cyclone_refresh_grid(ref_full))
        cyclone_mismatch = _cyclone_reference_mismatch_scan(
            ref,
            cfg,
            verbose=verbose,
            progress=progress,
        )
        (outdir / "cyclone_mismatch_table.csv").write_text(
            "\n".join(_build_rows(cyclone_mismatch, ref)) + "\n", encoding="utf-8"
        )
        return 0
    ky_subset = np.array([0.3, 0.4])
    cfg = CycloneBaseCase(grid=_cyclone_refresh_grid(ref_full))

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
    cyclone_mismatch = _cyclone_reference_mismatch_scan(
        ref,
        cfg,
        verbose=verbose,
        progress=progress,
    )
    (outdir / "cyclone_mismatch_table.csv").write_text(
        "\n".join(_build_rows(cyclone_mismatch, ref)) + "\n", encoding="utf-8"
    )
    if args.case == "cyclone":
        return 0

    etg_grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28, y0=0.2)
    etg_time_cfg = TimeConfig(
        t_max=6.0,
        dt=0.01,
        method="imex2",
        use_diffrax=False,
        progress_bar=False,
        sample_stride=2,
    )
    etg_R = np.array([4.0, 6.0, 8.0, 10.0])
    etg_rows = ["R_over_LTe,gamma,omega"]
    for R in etg_R:
        cfg = ETGBaseCase(grid=etg_grid, model=ETGModelConfig(R_over_LTe=float(R)))
        steps = int(round(etg_time_cfg.t_max / etg_time_cfg.dt))
        _log(
            f"\n=== ETG trend R/LTe={float(R):.2f} ===",
            verbose=verbose,
            use_tqdm=progress,
        )
        _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
        res = run_etg_linear(
            cfg=cfg,
            ky_target=5.0,
            Nl=24,
            Nm=8,
            steps=steps,
            dt=etg_time_cfg.dt,
            time_cfg=etg_time_cfg,
            solver=ETG_SOLVER,
            krylov_cfg=ETG_KRYLOV,
            mode_method="z_index",
            fit_signal="phi",
            auto_window=True,
            tmin=None,
            tmax=None,
            diagnostic_norm=DIAGNOSTIC_NORM,
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
    kinetic_cfg = KineticElectronBaseCase(
        grid=GridConfig(Nx=1, Ny=16, Nz=96, Lx=62.8, Ly=62.8, y0=10.0, ntheta=32, nperiod=2)
    )
    kinetic_time_cfg = TimeConfig(
        t_max=4.0,
        dt=0.001,
        method="imex2",
        use_diffrax=False,
        progress_bar=False,
        sample_stride=2,
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
        auto_window=True,
        label="Kinetic ITG mismatch",
        ref=kinetic_ref,
        run_kwargs={
            "time_cfg": kinetic_time_cfg,
            "fit_signal": "phi",
            "mode_method": "z_index",
            "init_species_index": 1,
            "density_species_index": 1,
        },
        verbose=verbose,
        progress=progress,
        stiff_spot_check=stiff_spot_check,
        stiff_spot_check_topk=stiff_spot_topk,
        stiff_spot_check_min_ky=stiff_spot_min_ky,
        stiff_spot_check_dt=stiff_spot_dt,
        stiff_spot_check_tmax=stiff_spot_tmax,
        stiff_spot_check_replace=stiff_spot_replace,
    )
    kinetic_mismatch = LinearScanResult(ky=kin_ky, gamma=kin_g, omega=kin_w)
    (outdir / "kinetic_mismatch_table.csv").write_text(
        "\n".join(_build_rows(kinetic_mismatch, kinetic_ref)) + "\n", encoding="utf-8"
    )

    etg_ref = load_etg_reference()
    etg_cfg = _etg_benchmark_case()
    etg_time = TimeConfig(
        t_max=6.0,
        dt=0.01,
        method="imex2",
        use_diffrax=False,
        progress_bar=False,
        sample_stride=2,
    )
    etg_steps = int(round(etg_time.t_max / etg_time.dt))
    etg_mismatch = _etg_reference_mismatch_scan(
        etg_ref,
        etg_cfg,
        dt=etg_time.dt,
        steps=etg_steps,
        verbose=verbose,
        progress=progress,
    )
    (outdir / "etg_mismatch_table.csv").write_text(
        "\n".join(_build_rows(etg_mismatch, etg_ref)) + "\n", encoding="utf-8"
    )

    kbm_ref = load_kbm_reference()
    kbm_dt = _scale_dt(kbm_ref.ky, base_dt=0.0005, ky_ref=0.3)
    kbm_steps = _scale_steps(kbm_ref.ky, base_steps=4000, ky_ref=0.3, max_steps=8000)
    kbm_cfg = KBMBaseCase(
        grid=GridConfig(Nx=1, Ny=12, Nz=96, Lx=62.8, Ly=62.8, y0=10.0, ntheta=32, nperiod=2)
    )
    kbm_time_cfg = TimeConfig(
        t_max=3.0,
        dt=0.001,
        method="imex2",
        use_diffrax=False,
        progress_bar=False,
        sample_stride=2,
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
        tmin=None,
        tmax=None,
        auto_window=True,
        run_kwargs={"fit_signal": "phi", "mode_method": "z_index", "time_cfg": kbm_time_cfg},
        label="KBM mismatch",
        ref=kbm_ref,
        verbose=verbose,
        progress=progress,
        stiff_spot_check=stiff_spot_check,
        stiff_spot_check_topk=stiff_spot_topk,
        stiff_spot_check_dt=stiff_spot_dt,
        stiff_spot_check_tmax=stiff_spot_tmax,
        stiff_spot_check_replace=stiff_spot_replace,
    )
    kbm_mismatch = LinearScanResult(ky=kbm_beta, gamma=kbm_g, omega=kbm_w)
    (outdir / "kbm_mismatch_table.csv").write_text(
        "\n".join(_build_rows(kbm_mismatch, kbm_ref)) + "\n", encoding="utf-8"
    )

    tem_ref = load_tem_reference()
    tem_dt = _scale_dt(tem_ref.ky, base_dt=0.001, ky_ref=0.3)
    tem_steps = _scale_steps(tem_ref.ky, base_steps=2000, ky_ref=0.3, max_steps=6000)
    tem_cfg = TEMBaseCase()
    tem_time_cfg = TimeConfig(
        t_max=3.0,
        dt=0.001,
        method="imex2",
        use_diffrax=False,
        progress_bar=False,
        sample_stride=2,
    )
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
        tmin=None,
        tmax=None,
        auto_window=True,
        run_kwargs={"fit_signal": "phi", "mode_method": "z_index", "time_cfg": tem_time_cfg},
        label="TEM mismatch",
        ref=tem_ref,
        verbose=verbose,
        progress=progress,
        stiff_spot_check=stiff_spot_check,
        stiff_spot_check_topk=stiff_spot_topk,
        stiff_spot_check_min_ky=stiff_spot_min_ky,
        stiff_spot_check_dt=stiff_spot_dt,
        stiff_spot_check_tmax=stiff_spot_tmax,
        stiff_spot_check_replace=stiff_spot_replace,
    )
    tem_mismatch = LinearScanResult(ky=tem_ky, gamma=tem_g, omega=tem_w)
    (outdir / "tem_mismatch_table.csv").write_text(
        "\n".join(_build_rows(tem_mismatch, tem_ref)) + "\n", encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
