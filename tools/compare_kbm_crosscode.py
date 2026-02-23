#!/usr/bin/env python3
"""Compare KBM beta-scan points from GS2/stella outputs against SPECTRAX-GK."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from spectraxgk.benchmarks import KBM_KRYLOV_DEFAULT, run_kbm_beta_scan


def _load_gs2_kbm_points(files: Iterable[Path], ky_target: float) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for path in files:
        ds = xr.open_dataset(path)
        beta = float(np.asarray(ds["beta"]).reshape(-1)[0])
        ky = np.asarray(ds["ky"]).reshape(-1)
        ky_idx = int(np.argmin(np.abs(ky - ky_target)))
        omavg = np.asarray(ds["omega_average"])
        if omavg.ndim != 4 or omavg.shape[-1] != 2:
            raise ValueError(f"unexpected omega_average shape in {path}: {omavg.shape}")
        final = omavg[-1, ky_idx, 0, :]
        rows.append(
            {
                "source": str(path),
                "beta": beta,
                "gamma_ref": float(final[1]),
                "omega_ref": float(final[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)


def _load_stella_kbm_points(files: Iterable[Path], ky_target: float, navg_frac: float) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for path in files:
        ds = xr.open_dataset(path)
        beta = float(np.asarray(ds["beta"]).reshape(-1)[0])
        omega_da = ds["omega"]
        omega_arr = np.asarray(omega_da)
        dims = list(omega_da.dims)

        if "ri" in dims:
            comp_axis = dims.index("ri")
        else:
            size2_axes = [i for i, s in enumerate(omega_arr.shape) if s == 2]
            if not size2_axes:
                raise ValueError(f"could not infer omega component axis in {path}: {omega_arr.shape}")
            comp_axis = size2_axes[-1]
        omega_arr = np.moveaxis(omega_arr, comp_axis, -1)
        dims = [d for i, d in enumerate(dims) if i != comp_axis] + ["ri"]

        if "ky" not in dims or "t" not in dims:
            raise ValueError(f"omega does not include required ky/t dims in {path}: {dims}")
        ky_vals = np.asarray(ds["ky"], dtype=float)
        ky_idx = int(np.argmin(np.abs(ky_vals - ky_target)))

        if "kx" in dims:
            kx_vals = np.asarray(ds["kx"], dtype=float)
            kx_idx = int(np.argmin(np.abs(kx_vals)))
            kx_axis = dims.index("kx")
            omega_arr = np.take(omega_arr, kx_idx, axis=kx_axis)
            dims.pop(kx_axis)

        ky_axis = dims.index("ky")
        t_axis = dims.index("t")
        omega_ky = np.take(omega_arr, ky_idx, axis=ky_axis)
        if ky_axis < t_axis:
            t_axis -= 1
        if t_axis != 0:
            omega_ky = np.moveaxis(omega_ky, t_axis, 0)

        omega_t = np.asarray(omega_ky[:, 0], dtype=float)
        gamma_t = np.asarray(omega_ky[:, 1], dtype=float)
        finite = np.isfinite(omega_t) & np.isfinite(gamma_t)
        if not np.any(finite):
            rows.append(
                {
                    "source": str(path),
                    "beta": beta,
                    "gamma_ref": np.nan,
                    "omega_ref": np.nan,
                }
            )
            continue
        omega_f = omega_t[finite]
        gamma_f = gamma_t[finite]
        n = max(1, int(np.floor(navg_frac * omega_f.size)))
        rows.append(
            {
                "source": str(path),
                "beta": beta,
                "gamma_ref": float(np.mean(gamma_f[-n:])),
                "omega_ref": float(np.mean(omega_f[-n:])),
            }
        )
    return pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)


def _merge_with_spectrax(df_ref: pd.DataFrame, df_sp: pd.DataFrame) -> pd.DataFrame:
    out = df_ref.merge(df_sp, on="beta", how="left", validate="many_to_one")
    out["rel_gamma"] = np.where(
        out["gamma_ref"] != 0.0, (out["gamma_spectrax"] - out["gamma_ref"]) / out["gamma_ref"], np.nan
    )
    out["rel_omega"] = np.where(
        out["omega_ref"] != 0.0, (out["omega_spectrax"] - out["omega_ref"]) / out["omega_ref"], np.nan
    )
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gs2-out", action="append", required=True, help="Path to GS2 KBM .out.nc (repeatable)")
    p.add_argument("--stella-out", action="append", required=True, help="Path to stella KBM .out.nc (repeatable)")
    p.add_argument("--ky-target", type=float, default=0.3)
    p.add_argument("--out-gs2-csv", type=Path, default=Path("docs/_static/kbm_gs2_mismatch.csv"))
    p.add_argument("--out-stella-csv", type=Path, default=Path("docs/_static/kbm_stella_mismatch.csv"))
    p.add_argument("--Nl", type=int, default=6)
    p.add_argument("--Nm", type=int, default=16)
    p.add_argument("--solver", choices=("krylov", "time"), default="krylov")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--method", type=str, default="rk4")
    p.add_argument("--fit-signal", choices=("phi", "density", "auto"), default="phi")
    p.add_argument("--streaming-fit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--window-fraction", type=float, default=0.4)
    p.add_argument("--start-fraction", type=float, default=0.2)
    p.add_argument("--min-points", type=int, default=40)
    p.add_argument("--growth-weight", type=float, default=1.0)
    p.add_argument("--require-positive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stella-navg-frac", type=float, default=0.3)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    gs2_ref = _load_gs2_kbm_points([Path(p) for p in args.gs2_out], args.ky_target)
    stella_ref = _load_stella_kbm_points([Path(p) for p in args.stella_out], args.ky_target, args.stella_navg_frac)

    beta_all = np.unique(np.concatenate([gs2_ref["beta"].to_numpy(), stella_ref["beta"].to_numpy()]))
    scan = run_kbm_beta_scan(
        beta_all,
        ky_target=args.ky_target,
        Nl=args.Nl,
        Nm=args.Nm,
        dt=args.dt,
        steps=args.steps,
        method=args.method,
        solver=args.solver,
        krylov_cfg=KBM_KRYLOV_DEFAULT,
        fit_signal=args.fit_signal,
        streaming_fit=args.streaming_fit,
        window_fraction=args.window_fraction,
        min_points=args.min_points,
        start_fraction=args.start_fraction,
        growth_weight=args.growth_weight,
        require_positive=args.require_positive,
    )
    sp = pd.DataFrame(
        {
            "beta": np.asarray(scan.ky, dtype=float),
            "gamma_spectrax": np.asarray(scan.gamma, dtype=float),
            "omega_spectrax": np.asarray(scan.omega, dtype=float),
        }
    )

    gs2_cmp = _merge_with_spectrax(gs2_ref, sp).sort_values("beta").reset_index(drop=True)
    stella_cmp = _merge_with_spectrax(stella_ref, sp).sort_values("beta").reset_index(drop=True)

    args.out_gs2_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_stella_csv.parent.mkdir(parents=True, exist_ok=True)
    gs2_cmp.to_csv(args.out_gs2_csv, index=False)
    stella_cmp.to_csv(args.out_stella_csv, index=False)

    def _summary(df: pd.DataFrame, label: str) -> None:
        print(
            f"{label}: mean(|rel_gamma|)={np.nanmean(np.abs(df['rel_gamma'])):.3%} "
            f"max(|rel_gamma|)={np.nanmax(np.abs(df['rel_gamma'])):.3%} "
            f"mean(|rel_omega|)={np.nanmean(np.abs(df['rel_omega'])):.3%} "
            f"max(|rel_omega|)={np.nanmax(np.abs(df['rel_omega'])):.3%}"
        )

    print(f"saved GS2 mismatch table: {args.out_gs2_csv}")
    print(f"saved stella mismatch table: {args.out_stella_csv}")
    _summary(gs2_cmp, "GS2")
    _summary(stella_cmp, "stella")


if __name__ == "__main__":
    main()
