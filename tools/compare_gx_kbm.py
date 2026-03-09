#!/usr/bin/env python3
"""Compare GX KBM linear outputs against SPECTRAX-GK."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from spectraxgk.analysis import extract_eigenfunction, select_ky_index
from spectraxgk.benchmarks import KBM_KRYLOV_DEFAULT, run_kbm_linear
from spectraxgk.config import KBMBaseCase, GeometryConfig, GridConfig, KineticElectronModelConfig
from spectraxgk.grids import build_spectral_grid, select_ky_grid


def _load_gx_omega_gamma(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float, float | None, float | None, float | None, float | None]:
    root = Dataset(path, "r")
    try:
        grids = root.groups["Grids"]
        diagnostics = root.groups["Diagnostics"]
        inputs = root.groups["Inputs"]
    except KeyError as exc:
        raise ValueError(f"{path} missing expected GX groups") from exc

    ky = np.asarray(grids.variables["ky"][:], dtype=float)
    omega_kxkyt = np.asarray(diagnostics.variables["omega_kxkyt"][:], dtype=float)
    if omega_kxkyt.ndim != 4 or omega_kxkyt.shape[-1] != 2:
        raise ValueError(f"unexpected omega_kxkyt shape: {omega_kxkyt.shape}")

    omega = omega_kxkyt[:, :, 0, :]

    beta = float(np.asarray(inputs.variables["beta"][:]))

    def _maybe_scalar(name: str) -> float | None:
        if name not in inputs.variables:
            return None
        data = np.asarray(inputs.variables[name][:])
        if np.ma.is_masked(data):
            return None
        val = float(data)
        if abs(val) < 1.0e-12:
            return None
        return val

    q = _maybe_scalar("q")
    shat = _maybe_scalar("shat")
    eps = _maybe_scalar("eps")
    rmaj = _maybe_scalar("Rmaj")

    root.close()
    mask = ky > 0.0
    return ky[mask], omega[:, mask], beta, q, shat, eps, rmaj


def _infer_y0(ky: np.ndarray) -> float:
    if ky.size < 2:
        raise ValueError("Need at least two ky values to infer y0.")
    ky_min = float(np.min(ky[ky > 0.0]))
    if ky_min <= 0.0:
        raise ValueError("ky array does not contain positive values.")
    return 1.0 / ky_min


def _normalize_mode(theta: np.ndarray, mode: np.ndarray) -> np.ndarray:
    finite = np.isfinite(mode)
    if not np.any(finite):
        return np.zeros_like(mode)
    idx0 = int(np.argmin(np.abs(theta)))
    ref = mode[idx0]
    if not np.isfinite(ref) or abs(ref) < 1.0e-14:
        idx = int(np.nanargmax(np.abs(np.where(finite, mode, 0.0))))
        ref = mode[idx]
    if not np.isfinite(ref) or abs(ref) < 1.0e-14:
        scale = float(np.nanmax(np.abs(np.where(finite, mode, 0.0))))
        return mode if scale <= 0.0 else mode / scale
    return mode / ref


def _load_gx_eigenfunction(path: Path, ky_target: float) -> tuple[np.ndarray, np.ndarray]:
    root = Dataset(path, "r")
    grids = root.groups["Grids"]
    diag = root.groups["Diagnostics"]
    theta = np.asarray(grids.variables["theta"][:], dtype=float)
    ky = np.asarray(grids.variables["ky"][:], dtype=float)
    ky_idx = int(np.argmin(np.abs(ky - float(ky_target))))
    phi = np.asarray(diag.variables["Phi"][-1, ky_idx, 0, :, :], dtype=float)
    root.close()
    mode = phi[:, 0] + 1j * phi[:, 1]
    return theta, _normalize_mode(theta, mode)


def _mode_overlap(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom <= 0.0:
        return float("nan")
    return float(np.abs(np.vdot(lhs, rhs)) / denom)


def _mode_rel_l2(lhs: np.ndarray, rhs: np.ndarray) -> float:
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm <= 0.0:
        return float("nan")
    phase = np.vdot(lhs, rhs)
    lhs_align = lhs if abs(phase) <= 1.0e-30 else lhs * np.exp(-1j * np.angle(phase))
    return float(np.linalg.norm(lhs_align - rhs) / rhs_norm)


def _interp_complex(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y.real) + 1j * np.interp(x_new, x, y.imag)


def _candidate_objective(
    *,
    rel_gamma: float,
    rel_omega: float,
    eig_overlap_gx: float,
    eig_overlap_prev: float,
    gamma_weight: float,
    omega_weight: float,
    gx_overlap_weight: float,
    prev_overlap_weight: float,
) -> float:
    obj = gamma_weight * rel_gamma + omega_weight * rel_omega
    if np.isfinite(eig_overlap_gx):
        obj += gx_overlap_weight * (1.0 - eig_overlap_gx)
    if np.isfinite(eig_overlap_prev):
        obj += prev_overlap_weight * (1.0 - eig_overlap_prev)
    return float(obj)


def _extract_mode(
    result,
    theta: np.ndarray,
    *,
    method: str,
    tmin: float | None,
    tmax: float | None,
) -> np.ndarray:
    phi_t = np.asarray(result.phi_t)
    t = np.asarray(result.t, dtype=float)
    if t.size <= 1:
        return _normalize_mode(theta, np.asarray(phi_t[-1, 0, 0, :], dtype=np.complex128))
    mode = extract_eigenfunction(
        phi_t,
        t,
        result.selection,
        z=theta,
        method=method,
        tmin=tmin,
        tmax=tmax,
    )
    return _normalize_mode(theta, np.asarray(mode, dtype=np.complex128))


def _mode_metrics(
    result,
    *,
    grid_full,
    ky_value: float,
    gx_big: Path,
    eigen_method: str,
    eigen_tmin: float | None,
    eigen_tmax: float | None,
    prev_mode: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    ky_idx = select_ky_index(np.asarray(grid_full.ky), float(ky_value))
    grid = select_ky_grid(grid_full, ky_idx)
    theta = np.asarray(grid.z, dtype=float)
    if eigen_tmin is not None or eigen_tmax is not None:
        eig_tmin = eigen_tmin
        eig_tmax = eigen_tmax
    elif np.asarray(result.t).size > 1:
        eig_tmin = float(np.asarray(result.t)[-1]) * 0.6
        eig_tmax = float(np.asarray(result.t)[-1])
    else:
        eig_tmin = None
        eig_tmax = None

    mode_sp = _extract_mode(
        result,
        theta,
        method=eigen_method,
        tmin=eig_tmin,
        tmax=eig_tmax,
    )
    if gx_big.exists():
        theta_gx, mode_gx = _load_gx_eigenfunction(gx_big, float(ky_value))
        if theta_gx.shape != theta.shape or not np.allclose(theta_gx, theta, atol=1.0e-6, rtol=1.0e-6):
            mode_gx = _normalize_mode(theta, _interp_complex(theta, theta_gx, mode_gx))
        eig_overlap = _mode_overlap(mode_sp, mode_gx)
        eig_rel_l2 = _mode_rel_l2(mode_sp, mode_gx)
    else:
        eig_overlap = float("nan")
        eig_rel_l2 = float("nan")
    prev_overlap = float("nan") if prev_mode is None else _mode_overlap(mode_sp, prev_mode)
    return theta, mode_sp, eig_overlap, eig_rel_l2, prev_overlap


def _build_cfg(
    *,
    beta: float,
    q: float,
    shat: float,
    eps: float,
    rmaj: float,
    ny: int,
    ntheta: int,
    nperiod: int,
    y0: float,
) -> KBMBaseCase:
    grid = GridConfig(
        Nx=1,
        Ny=ny,
        Nz=ntheta * (2 * nperiod - 1),
        Lx=62.8,
        Ly=62.8,
        y0=y0,
        ntheta=ntheta,
        nperiod=nperiod,
        boundary="linked",
    )
    geom = GeometryConfig(
        q=q,
        s_hat=shat,
        epsilon=eps,
        R0=rmaj,
    )
    model = KineticElectronModelConfig(
        R_over_LTi=2.49,
        R_over_LTe=2.49,
        R_over_Ln=0.8,
        Te_over_Ti=1.0,
        mass_ratio=3670.0,
        nu_i=0.0,
        nu_e=0.0,
        beta=beta,
    )
    return KBMBaseCase(grid=grid, geometry=geom, model=model)


def _run_candidate(
    args,
    cfg: KBMBaseCase,
    ky_value: float,
    beta_value: float,
    solver_name: str,
    *,
    gx_gamma: float | None = None,
    gx_omega: float | None = None,
):
    fit_signal = args.time_fit_signal if solver_name in {"time", "gx_time"} else "phi"
    krylov_cfg = None
    if (
        solver_name == "krylov"
        and bool(getattr(args, "krylov_gx_shift", False))
        and gx_gamma is not None
        and gx_omega is not None
        and np.isfinite(gx_gamma)
        and np.isfinite(gx_omega)
    ):
        krylov_cfg = replace(
            KBM_KRYLOV_DEFAULT,
            shift=complex(float(gx_gamma), -float(gx_omega)),
            shift_source=str(getattr(args, "krylov_gx_shift_source", "target")),
            shift_selection="shift",
            omega_sign=0,
            omega_target_factor=0.0,
        )
    return run_kbm_linear(
        ky_target=float(ky_value),
        beta_value=float(beta_value),
        Nl=args.Nl,
        Nm=args.Nm,
        dt=args.dt,
        steps=args.steps,
        method=args.method,
        cfg=cfg,
        solver=solver_name,
        krylov_cfg=krylov_cfg,
        fit_signal=fit_signal,
        mode_method=args.mode_method,
        diagnostic_norm="gx",
        gx_reference=True,
        auto_window=not args.no_auto_window,
        tmin=args.tmin,
        tmax=args.tmax,
        sample_stride=args.sample_stride,
    )


def _write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _candidate_row(
    *,
    ky: float,
    solver: str,
    result,
    gx_gamma: float,
    gx_omega: float,
    eig_overlap_gx: float,
    eig_rel_l2: float,
    eig_overlap_prev: float,
    branch_score: float,
    selected: bool,
) -> dict[str, float | str | bool]:
    gamma = float(result.gamma)
    omega = float(result.omega)
    rel_gamma = abs(gamma - float(gx_gamma)) / max(abs(float(gx_gamma)), 1.0e-12)
    rel_omega = abs(omega - float(gx_omega)) / max(abs(float(gx_omega)), 1.0e-12)
    return {
        "ky": float(ky),
        "solver": solver,
        "gamma_gx": float(gx_gamma),
        "gamma": gamma,
        "rel_gamma": rel_gamma,
        "omega_gx": float(gx_omega),
        "omega": omega,
        "rel_omega": rel_omega,
        "eig_overlap_gx": eig_overlap_gx,
        "eig_rel_l2": eig_rel_l2,
        "eig_overlap_prev": eig_overlap_prev,
        "branch_score": branch_score,
        "selected": bool(selected),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare GX KBM output against SPECTRAX-GK.")
    parser.add_argument("--gx", type=Path, required=True, help="Path to GX .out.nc file")
    parser.add_argument(
        "--gx-big",
        type=Path,
        default=Path(".cache/gx/kbm_salpha.big.nc"),
        help="Path to GX .big.nc file for eigenfunction comparisons",
    )
    parser.add_argument("--Nl", type=int, default=16)
    parser.add_argument("--Nm", type=int, default=48)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--method", type=str, default="rk4")
    parser.add_argument("--solver", type=str, default="gx_time")
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--y0", type=float, default=10.0, help="Fallback y0 when ky list is truncated.")
    parser.add_argument(
        "--nky",
        type=int,
        default=None,
        help="Override GX nky when ky list is truncated (sets Ny = 3*(nky-1)+1).",
    )
    parser.add_argument(
        "--gx-avg-fraction",
        type=float,
        default=0.5,
        help="Average GX omega/gamma over the last fraction of time samples.",
    )
    parser.add_argument("--q", type=float, default=1.4)
    parser.add_argument("--shat", type=float, default=0.8)
    parser.add_argument("--eps", type=float, default=0.18)
    parser.add_argument("--Rmaj", type=float, default=2.77778)
    parser.add_argument(
        "--ky",
        type=str,
        default="",
        help="Comma-separated ky values to compare (default: all from GX output)",
    )
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--no-auto-window", action="store_true")
    parser.add_argument(
        "--branch-policy",
        type=str,
        choices=["fixed", "single", "gx-ref-auto", "auto", "continuation"],
        default="continuation",
        help="fixed/single: use one solver for all ky; gx-ref-auto/auto: legacy GX-scored solver selection; continuation: choose the most continuous candidate branch across ky.",
    )
    parser.add_argument(
        "--branch-solvers",
        type=str,
        default="gx_time,krylov,time",
        help="Comma-separated candidate solvers used when --branch-policy=gx-ref-auto.",
    )
    parser.add_argument("--branch-gamma-weight", type=float, default=1.0)
    parser.add_argument("--branch-omega-weight", type=float, default=1.0)
    parser.add_argument("--branch-overlap-gx-weight", type=float, default=1.0)
    parser.add_argument("--branch-overlap-prev-weight", type=float, default=2.0)
    parser.add_argument(
        "--krylov-gx-shift",
        action="store_true",
        help="When evaluating a Krylov candidate, use the GX reference eigenvalue as the shift target.",
    )
    parser.add_argument(
        "--krylov-gx-shift-source",
        type=str,
        default="target",
        choices=["target", "power", "propagator"],
        help="Seed source to pair with --krylov-gx-shift when probing an explicit shift-invert target.",
    )
    parser.add_argument(
        "--time-fit-signal",
        type=str,
        default="auto",
        choices=["auto", "phi", "density"],
        help="Fit signal used for time/gx_time candidate runs.",
    )
    parser.add_argument(
        "--mode-method",
        type=str,
        default="project",
        choices=["z_index", "max", "project"],
        help="Mode-extraction method for GX-time/fallback fits. The default matches run_kbm_linear and projects onto the late-time KBM structure.",
    )
    parser.add_argument(
        "--eigen-method",
        type=str,
        default="svd",
        choices=["svd", "snapshot"],
        help="Method used to extract SPECTRAX eigenfunctions from time histories.",
    )
    parser.add_argument("--eigen-tmin", type=float, default=None)
    parser.add_argument("--eigen-tmax", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV path for mismatch table")
    parser.add_argument(
        "--candidate-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-candidate branch metrics (one row per ky/solver).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    gx_ky, gx_omega_series, beta, q_gx, shat_gx, eps_gx, rmaj_gx = _load_gx_omega_gamma(args.gx)
    nky_full = int(len(gx_ky))
    if nky_full < 2:
        raise ValueError("GX output must contain at least two positive ky points.")
    if args.ky:
        ky_req = np.asarray([float(k.strip()) for k in args.ky.split(",") if k.strip()])
        if ky_req.size == 0:
            raise ValueError("No ky values parsed from --ky")
        idx = [int(np.argmin(np.abs(gx_ky - k))) for k in ky_req]
        gx_ky = gx_ky[idx]
        gx_omega_series = gx_omega_series[:, idx]

    nky = int(args.nky) if args.nky is not None else nky_full
    ny = 3 * (nky - 1) + 1
    y0 = _infer_y0(gx_ky) if len(gx_ky) > 1 else float(args.y0)

    if gx_omega_series.ndim != 3:
        raise ValueError(f"Unexpected GX omega series shape: {gx_omega_series.shape}")
    start = int((1.0 - float(args.gx_avg_fraction)) * gx_omega_series.shape[0])
    start = max(0, min(start, gx_omega_series.shape[0] - 1))
    gx_avg = np.mean(gx_omega_series[start:, :, :], axis=0)
    gx_omega = gx_avg[:, 0]
    gx_gamma = gx_avg[:, 1]

    cfg = _build_cfg(
        beta=beta,
        q=float(q_gx if q_gx is not None else args.q),
        shat=float(shat_gx if shat_gx is not None else args.shat),
        eps=float(eps_gx if eps_gx is not None else args.eps),
        rmaj=float(rmaj_gx if rmaj_gx is not None else args.Rmaj),
        ny=ny,
        ntheta=args.ntheta,
        nperiod=args.nperiod,
        y0=y0,
    )

    grid_full = build_spectral_grid(cfg.grid)
    use_legacy_auto = args.branch_policy in {"gx-ref-auto", "auto"}
    use_continuation = args.branch_policy == "continuation"
    branch_candidates = [s.strip() for s in args.branch_solvers.split(",") if s.strip()]
    if (use_legacy_auto or use_continuation) and not branch_candidates:
        raise ValueError("No candidate solvers parsed from --branch-solvers")

    rows: list[dict[str, float | str]] = []
    candidate_rows: list[dict[str, float | str | bool]] = []
    prev_mode: np.ndarray | None = None
    for i, ky_val in enumerate(gx_ky):
        branch_score = float("nan")
        if use_legacy_auto:
            best_row = None
            best_obj = np.inf
            candidate_start = len(candidate_rows)
            for solver_name in branch_candidates:
                result = _run_candidate(
                    args,
                    cfg,
                    float(ky_val),
                    beta,
                    solver_name,
                    gx_gamma=float(gx_gamma[i]),
                    gx_omega=float(gx_omega[i]),
                )
                rel_g = abs(float(result.gamma) - float(gx_gamma[i])) / max(abs(float(gx_gamma[i])), 1.0e-12)
                rel_o = abs(float(result.omega) - float(gx_omega[i])) / max(abs(float(gx_omega[i])), 1.0e-12)
                obj = rel_g + rel_o
                candidate_rows.append(
                    _candidate_row(
                        ky=float(ky_val),
                        solver=solver_name,
                        result=result,
                        gx_gamma=float(gx_gamma[i]),
                        gx_omega=float(gx_omega[i]),
                        eig_overlap_gx=float("nan"),
                        eig_rel_l2=float("nan"),
                        eig_overlap_prev=float("nan"),
                        branch_score=float(obj),
                        selected=False,
                    )
                )
                if args.candidate_out is not None:
                    _write_rows(args.candidate_out, candidate_rows)
                if obj < best_obj:
                    best_obj = obj
                    best_row = (solver_name, result)
            if best_row is None:
                raise RuntimeError(f"No valid solver candidate for ky={float(ky_val):.4f}")
            solver_used, result = best_row
            for row in candidate_rows[candidate_start:]:
                row["selected"] = row["solver"] == solver_used
        elif use_continuation:
            best_candidate = None
            best_obj = np.inf
            candidate_start = len(candidate_rows)
            for solver_name in branch_candidates:
                result_c = _run_candidate(
                    args,
                    cfg,
                    float(ky_val),
                    beta,
                    solver_name,
                    gx_gamma=float(gx_gamma[i]),
                    gx_omega=float(gx_omega[i]),
                )
                _theta, mode_c, eig_overlap_c, eig_rel_l2_c, prev_overlap_c = _mode_metrics(
                    result_c,
                    grid_full=grid_full,
                    ky_value=float(ky_val),
                    gx_big=args.gx_big,
                    eigen_method=args.eigen_method,
                    eigen_tmin=args.eigen_tmin,
                    eigen_tmax=args.eigen_tmax,
                    prev_mode=prev_mode,
                )
                rel_g = abs(float(result_c.gamma) - float(gx_gamma[i])) / max(abs(float(gx_gamma[i])), 1.0e-12)
                rel_o = abs(float(result_c.omega) - float(gx_omega[i])) / max(abs(float(gx_omega[i])), 1.0e-12)
                obj = _candidate_objective(
                    rel_gamma=rel_g,
                    rel_omega=rel_o,
                    eig_overlap_gx=eig_overlap_c,
                    eig_overlap_prev=prev_overlap_c,
                    gamma_weight=float(args.branch_gamma_weight),
                    omega_weight=float(args.branch_omega_weight),
                    gx_overlap_weight=float(args.branch_overlap_gx_weight),
                    prev_overlap_weight=float(args.branch_overlap_prev_weight),
                )
                candidate_rows.append(
                    _candidate_row(
                        ky=float(ky_val),
                        solver=solver_name,
                        result=result_c,
                        gx_gamma=float(gx_gamma[i]),
                        gx_omega=float(gx_omega[i]),
                        eig_overlap_gx=eig_overlap_c,
                        eig_rel_l2=eig_rel_l2_c,
                        eig_overlap_prev=prev_overlap_c,
                        branch_score=float(obj),
                        selected=False,
                    )
                )
                if args.candidate_out is not None:
                    _write_rows(args.candidate_out, candidate_rows)
                if obj < best_obj:
                    best_obj = obj
                    best_candidate = (solver_name, result_c, mode_c, eig_overlap_c, eig_rel_l2_c, prev_overlap_c)
            if best_candidate is None:
                raise RuntimeError(f"No valid solver candidate for ky={float(ky_val):.4f}")
            solver_used, result, mode_sp, eig_overlap, eig_rel_l2, prev_overlap = best_candidate
            branch_score = float(best_obj)
            for row in candidate_rows[candidate_start:]:
                row["selected"] = row["solver"] == solver_used
        else:
            solver_used = args.solver
            result = _run_candidate(
                args,
                cfg,
                float(ky_val),
                beta,
                solver_used,
                gx_gamma=float(gx_gamma[i]),
                gx_omega=float(gx_omega[i]),
            )
            candidate_rows.append(
                _candidate_row(
                    ky=float(ky_val),
                    solver=solver_used,
                    result=result,
                    gx_gamma=float(gx_gamma[i]),
                    gx_omega=float(gx_omega[i]),
                    eig_overlap_gx=float("nan"),
                    eig_rel_l2=float("nan"),
                    eig_overlap_prev=float("nan"),
                    branch_score=float("nan"),
                    selected=True,
                )
            )
            if args.candidate_out is not None:
                _write_rows(args.candidate_out, candidate_rows)

        if not use_continuation:
            _theta, mode_sp, eig_overlap, eig_rel_l2, prev_overlap = _mode_metrics(
                result,
                grid_full=grid_full,
                ky_value=float(ky_val),
                gx_big=args.gx_big,
                eigen_method=args.eigen_method,
                eigen_tmin=args.eigen_tmin,
                eigen_tmax=args.eigen_tmax,
                prev_mode=prev_mode,
            )
        prev_mode = mode_sp

        fit_window_tmin = float("nan")
        fit_window_tmax = float("nan")
        t_series = np.asarray(result.t, dtype=float)
        if t_series.size > 1 and solver_used == "gx_time":
            fit_window_tmin = float(t_series[-1]) * (1.0 - float(args.gx_avg_fraction))
            fit_window_tmax = float(t_series[-1])
        elif args.tmin is not None and args.tmax is not None:
            fit_window_tmin = float(args.tmin)
            fit_window_tmax = float(args.tmax)

        row = {
            "ky": float(ky_val),
            "solver": solver_used,
            "gamma_gx": float(gx_gamma[i]),
            "gamma": float(result.gamma),
            "rel_gamma": abs(float(result.gamma) - float(gx_gamma[i])) / max(abs(float(gx_gamma[i])), 1.0e-12),
            "omega_gx": float(gx_omega[i]),
            "omega": float(result.omega),
            "rel_omega": abs(float(result.omega) - float(gx_omega[i])) / max(abs(float(gx_omega[i])), 1.0e-12),
            "eig_overlap_gx": eig_overlap,
            "eig_rel_l2": eig_rel_l2,
            "eig_overlap_prev": prev_overlap,
            "branch_score": branch_score,
            "fit_window_tmin": fit_window_tmin,
            "fit_window_tmax": fit_window_tmax,
        }
        rows.append(row)
        if args.out is not None:
            _write_rows(args.out, rows)
        if args.candidate_out is not None:
            _write_rows(args.candidate_out, candidate_rows)

    table = pd.DataFrame(rows)
    print(
        "ky    solver      gx_gamma   sp_gamma   rel_gamma    gx_omega   sp_omega   rel_omega   eig_ovlp   eig_rel"
    )
    for entry in table.itertuples(index=False):
        print(
            f"{entry.ky:5.3f} {entry.solver:10s} {entry.gamma_gx:9.5f} {entry.gamma:9.5f} {entry.rel_gamma:9.3f} "
            f"{entry.omega_gx:9.5f} {entry.omega:9.5f} {entry.rel_omega:9.3f} {entry.eig_overlap_gx:9.3f} {entry.eig_rel_l2:9.3f}"
        )

    if args.out is not None:
        _write_rows(args.out, rows)
    if args.candidate_out is not None:
        _write_rows(args.candidate_out, candidate_rows)


if __name__ == "__main__":
    main()
