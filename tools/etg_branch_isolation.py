#!/usr/bin/env python3
"""Isolate ETG branch selection for ky=5,25 using targeted shift-invert."""

from __future__ import annotations

import argparse
import numpy as np

from spectraxgk.benchmarks import ETGBaseCase, load_etg_reference_gs2, run_etg_linear
from spectraxgk.linear_krylov import KrylovConfig


def _run_case(
    name: str,
    cfg: ETGBaseCase,
    krylov_cfg: KrylovConfig,
    *,
    Nl: int,
    Nm: int,
) -> None:
    ref = load_etg_reference_gs2()
    print(f"\n=== {name} ===")
    for ky in (5.0, 25.0):
        print(f"running ky={ky:.1f} Nl={Nl} Nm={Nm}", flush=True)
        out = run_etg_linear(
            ky_target=ky,
            cfg=cfg,
            Nl=Nl,
            Nm=Nm,
            solver="krylov",
            krylov_cfg=krylov_cfg,
            mode_method="z_index",
            fit_signal="phi",
        )
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        rel_g = (out.gamma - ref.gamma[idx]) / ref.gamma[idx]
        rel_w = (out.omega - ref.omega[idx]) / ref.omega[idx]
        print(
            f"ky={ky:>5.1f} "
            f"gamma={out.gamma:>10.6f} omega={out.omega:>10.6f} "
            f"| ref gamma={ref.gamma[idx]:>10.6f} omega={ref.omega[idx]:>10.6f} "
            f"| rel_gamma={rel_g:+8.2%} rel_omega={rel_w:+8.2%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Nl", type=int, default=24)
    parser.add_argument("--Nm", type=int, default=12)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--shift-maxiter", type=int, default=20)
    parser.add_argument("--shift-restart", type=int, default=8)
    args = parser.parse_args()

    cfg = ETGBaseCase()
    runs = {
        "etg_propagator_default": KrylovConfig(
            method="propagator",
            krylov_dim=16,
            restarts=args.restarts,
            omega_min_factor=0.0,
            omega_target_factor=0.3,
            omega_cap_factor=0.6,
            omega_sign=-1,
            power_iters=80,
            power_dt=0.002,
            mode_family="etg",
            fallback_method="arnoldi",
        ),
        "etg_shift_invert_targeted": KrylovConfig(
            method="shift_invert",
            krylov_dim=16,
            restarts=args.restarts,
            omega_min_factor=0.0,
            omega_target_factor=0.2,
            omega_cap_factor=0.5,
            omega_sign=-1,
            power_iters=80,
            power_dt=0.002,
            shift_source="target",
            shift_tol=1.0e-3,
            shift_maxiter=args.shift_maxiter,
            shift_restart=args.shift_restart,
            shift_preconditioner="hermite-line",
            shift_selection="targeted",
            mode_family="etg",
            fallback_method="propagator",
        ),
        "etg_shift_invert_nearest": KrylovConfig(
            method="shift_invert",
            krylov_dim=16,
            restarts=args.restarts,
            omega_min_factor=0.0,
            omega_target_factor=0.2,
            omega_cap_factor=0.5,
            omega_sign=-1,
            power_iters=80,
            power_dt=0.002,
            shift_source="target",
            shift_tol=1.0e-3,
            shift_maxiter=args.shift_maxiter,
            shift_restart=args.shift_restart,
            shift_preconditioner="hermite-line",
            shift_selection="nearest",
            mode_family="etg",
            fallback_method="propagator",
        ),
    }
    for name, kcfg in runs.items():
        _run_case(name, cfg, kcfg, Nl=args.Nl, Nm=args.Nm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
