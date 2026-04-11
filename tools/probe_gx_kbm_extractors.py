#!/usr/bin/env python3
"""Run one GX-time KBM trajectory per ky and score multiple extractors."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from spectraxgk.analysis import ModeSelection
from spectraxgk.benchmarks import run_kbm_linear
from spectraxgk.grids import build_spectral_grid

from tools.compare_gx_kbm import (
    _build_cfg,
    _mode_metrics,
    _prepare_gx_reference,
    _recompute_time_history_growth_on_grid,
)


def _trajectory_path(base_dir: Path, ky_value: float, *, steps: int | None = None) -> Path:
    tag = f"{float(ky_value):0.4f}".replace(".", "p")
    suffix = "" if steps is None else f"_steps_{int(steps)}"
    return base_dir / f"kbm_ky_{tag}{suffix}_trajectory.npz"


def _parse_checkpoint_steps(raw: str, default_steps: int) -> list[int]:
    if not str(raw).strip():
        return [int(default_steps)]
    out: list[int] = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        step = int(item)
        if step < 1:
            raise ValueError("checkpoint step counts must be >= 1")
        out.append(step)
    if not out:
        raise ValueError("No checkpoint steps parsed from --checkpoint-steps")
    return sorted(dict.fromkeys(out))


def _save_trajectory(path: Path, result) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gamma_t = getattr(result, "gamma_t", None)
    omega_t = getattr(result, "omega_t", None)
    np.savez_compressed(
        path,
        t=np.asarray(result.t, dtype=float),
        phi_t=np.asarray(result.phi_t),
        gamma_t=np.asarray(gamma_t, dtype=float) if gamma_t is not None else np.array([], dtype=float),
        omega_t=np.asarray(omega_t, dtype=float) if omega_t is not None else np.array([], dtype=float),
        ky=float(result.ky),
        sel_ky=int(result.selection.ky_index),
        sel_kx=int(result.selection.kx_index),
        sel_z=int(result.selection.z_index),
    )
    return path


def _load_trajectory(path: Path):
    data = np.load(path, allow_pickle=False)
    gamma_t = np.asarray(data["gamma_t"], dtype=float)
    omega_t = np.asarray(data["omega_t"], dtype=float)
    return SimpleNamespace(
        t=np.asarray(data["t"], dtype=float),
        phi_t=np.asarray(data["phi_t"]),
        gamma=float("nan"),
        omega=float("nan"),
        ky=float(np.asarray(data["ky"])),
        gamma_t=None if gamma_t.size == 0 else gamma_t,
        omega_t=None if omega_t.size == 0 else omega_t,
        selection=ModeSelection(
            ky_index=int(np.asarray(data["sel_ky"])),
            kx_index=int(np.asarray(data["sel_kx"])),
            z_index=int(np.asarray(data["sel_z"])),
        ),
    )


def _row(
    *,
    ky: float,
    steps: int,
    horizon_t: float,
    mode_method: str,
    result,
    gx_gamma: float,
    gx_omega: float,
    eig_overlap_gx: float,
    eig_rel_l2: float,
) -> dict[str, float | str]:
    gamma = float(result.gamma)
    omega = float(result.omega)
    return {
        "ky": float(ky),
        "steps": int(steps),
        "horizon_t": float(horizon_t),
        "solver": f"gx_time@{mode_method}",
        "gamma_gx": float(gx_gamma),
        "gamma": gamma,
        "rel_gamma": abs(gamma - float(gx_gamma)) / max(abs(float(gx_gamma)), 1.0e-12),
        "omega_gx": float(gx_omega),
        "omega": omega,
        "rel_omega": abs(omega - float(gx_omega)) / max(abs(float(gx_omega)), 1.0e-12),
        "eig_overlap_gx": float(eig_overlap_gx),
        "eig_rel_l2": float(eig_rel_l2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx", type=Path, required=True, help="Path to GX KBM .out.nc or .npz reference.")
    parser.add_argument(
        "--gx-big",
        type=Path,
        default=Path(".cache/gx/kbm_salpha.big.nc"),
        help="Optional GX .big.nc for eigenfunction comparisons.",
    )
    parser.add_argument("--out", type=Path, required=True, help="CSV output path.")
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Optional directory for cached trajectory npz files. Defaults next to --out.",
    )
    parser.add_argument(
        "--reuse-trajectory",
        action="store_true",
        help="Reuse an existing trajectory cache instead of rerunning GX-time dynamics.",
    )
    parser.add_argument("--Nl", type=int, default=16)
    parser.add_argument("--Nm", type=int, default=48)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of SPECTRAX steps. Defaults to the GX horizon rounded up by --dt.",
    )
    parser.add_argument("--method", type=str, default="rk4")
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--y0", type=float, default=10.0)
    parser.add_argument("--nky", type=int, default=None)
    parser.add_argument("--q", type=float, default=1.4)
    parser.add_argument("--shat", type=float, default=0.8)
    parser.add_argument("--eps", type=float, default=0.18)
    parser.add_argument("--Rmaj", type=float, default=2.77778)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--gx-avg-fraction", type=float, default=0.5)
    parser.add_argument("--ky", type=str, default="", help="Comma-separated ky values to probe.")
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument(
        "--checkpoint-steps",
        type=str,
        default="",
        help="Optional comma-separated step horizons. The probe writes rows after each horizon.",
    )
    parser.add_argument(
        "--mode-methods",
        type=str,
        default="project,svd,max,z_index",
        help="Comma-separated extractor methods applied to the same cached trajectory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gx_time, gx_ky, gx_omega_series, beta, q_gx, shat_gx, eps_gx, rmaj_gx, nky_full, y0 = _prepare_gx_reference(
        args.gx,
        ky_arg=str(args.ky),
        y0_fallback=float(args.y0),
    )
    if gx_omega_series.ndim != 3:
        raise ValueError(f"Unexpected GX omega series shape: {gx_omega_series.shape}")
    nky = int(args.nky) if args.nky is not None else max(int(np.max(np.arange(nky_full) + 1)), nky_full)
    ny = 3 * (nky - 1) + 1

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
        ntheta=int(args.ntheta),
        nperiod=int(args.nperiod),
        y0=y0,
    )
    steps = int(args.steps) if args.steps is not None else max(int(np.ceil(float(gx_time[-1]) / float(args.dt))), 1)
    checkpoint_steps = _parse_checkpoint_steps(args.checkpoint_steps, steps)
    mode_methods = [item.strip() for item in str(args.mode_methods).split(",") if item.strip()]
    if not mode_methods:
        raise ValueError("No mode methods parsed from --mode-methods")
    traj_dir = args.trajectory_dir or args.out.parent
    helper_args = SimpleNamespace(tmin=args.tmin, tmax=args.tmax, gx_avg_fraction=float(args.gx_avg_fraction))
    grid_full = build_spectral_grid(cfg.grid)

    rows: list[dict[str, float | str]] = []
    for i, ky_val in enumerate(gx_ky):
        for step_count in checkpoint_steps:
            traj_path = _trajectory_path(Path(traj_dir), float(ky_val), steps=step_count)
            if bool(args.reuse_trajectory) and traj_path.exists():
                result = _load_trajectory(traj_path)
            else:
                result = run_kbm_linear(
                    ky_target=float(ky_val),
                    beta_value=float(beta),
                    Nl=int(args.Nl),
                    Nm=int(args.Nm),
                    dt=float(args.dt),
                    steps=int(step_count),
                    method=str(args.method),
                    cfg=cfg,
                    solver="gx_time",
                    fit_signal="phi",
                    mode_method="z_index",
                    diagnostic_norm="gx",
                    gx_reference=True,
                    auto_window=False,
                    tmin=args.tmin,
                    tmax=args.tmax,
                    sample_stride=int(args.sample_stride),
                )
                _save_trajectory(traj_path, result)

            horizon_t = float(np.asarray(result.t, dtype=float)[-1]) if np.asarray(result.t).size else 0.0
            for mode_method in mode_methods:
                scored = _recompute_time_history_growth_on_grid(
                    helper_args,
                    result,
                    mode_method=mode_method,
                    t_ref=gx_time,
                )
                _theta, mode_sp, eig_overlap_gx, eig_rel_l2, _prev = _mode_metrics(
                    scored,
                    grid_full=grid_full,
                    ky_value=float(ky_val),
                    gx_big=args.gx_big,
                    eigen_method="svd",
                    eigen_tmin=args.tmin,
                    eigen_tmax=args.tmax,
                    prev_mode=None,
                )
                rows.append(
                    _row(
                        ky=float(ky_val),
                        steps=int(step_count),
                        horizon_t=horizon_t,
                        mode_method=mode_method,
                        result=scored,
                        gx_gamma=float(gx_gamma[i]),
                        gx_omega=float(gx_omega[i]),
                        eig_overlap_gx=eig_overlap_gx,
                        eig_rel_l2=eig_rel_l2,
                    )
                )
                print(
                    f"ky={float(ky_val):0.4f} steps={int(step_count):4d} solver=gx_time@{mode_method:<12s} "
                    f"gamma={float(scored.gamma): .6e} omega={float(scored.omega): .6e} "
                    f"rel_gamma={rows[-1]['rel_gamma']:.3e} rel_omega={rows[-1]['rel_omega']:.3e} "
                    f"eig_overlap={eig_overlap_gx:.3e}"
                )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
