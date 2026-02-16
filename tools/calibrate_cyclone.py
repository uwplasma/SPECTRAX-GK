"""Parameter sweep to calibrate Cyclone base case normalization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from spectraxgk.benchmarks import load_cyclone_reference, run_cyclone_scan
from spectraxgk.config import CycloneBaseCase
from spectraxgk.linear import LinearParams


@dataclass(frozen=True)
class SweepResult:
    rho_star: float
    omega_d_scale: float
    omega_star_scale: float
    max_rel: float
    mean_rel: float


def _error_metrics(ref_ky: np.ndarray, ref_gamma: np.ndarray, ref_omega: np.ndarray, scan) -> tuple[float, float]:
    rel_errors = []
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref_ky - ky)))
        gamma_ref = float(ref_gamma[idx])
        omega_ref = float(ref_omega[idx])
        if gamma_ref != 0.0:
            rel_errors.append(abs(abs(float(gamma)) - gamma_ref) / gamma_ref)
        if omega_ref != 0.0:
            rel_errors.append(abs(abs(float(omega)) - omega_ref) / omega_ref)
    rel = np.asarray(rel_errors)
    return float(np.nanmax(rel)), float(np.nanmean(rel))


def _parse_range(values: Iterable[float]) -> np.ndarray:
    return np.array(list(values), dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho-star", nargs="+", type=float, default=[0.2, 0.3, 0.4])
    parser.add_argument("--omega-d-scale", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    parser.add_argument("--omega-star-scale", nargs="+", type=float, default=[7.5, 8.5, 9.5])
    parser.add_argument("--Nl", type=int, default=3)
    parser.add_argument("--Nm", type=int, default=6)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--tmin", type=float, default=4.0)
    parser.add_argument("--ky-subset", nargs="*", type=float, default=[0.3, 0.4])
    parser.add_argument("--output-csv", type=str, default="")
    args = parser.parse_args()

    ref = load_cyclone_reference()
    ky_values = ref.ky[1:] if args.ky_subset is None else np.array(args.ky_subset, dtype=float)

    cfg = CycloneBaseCase()

    results: list[SweepResult] = []
    for rho_star in _parse_range(args.rho_star):
        for omega_d_scale in _parse_range(args.omega_d_scale):
            for omega_star_scale in _parse_range(args.omega_star_scale):
                params = LinearParams(
                    R_over_Ln=cfg.model.R_over_Ln,
                    R_over_LTi=cfg.model.R_over_LTi,
                    omega_d_scale=float(omega_d_scale),
                    omega_star_scale=float(omega_star_scale),
                    rho_star=float(rho_star),
                )
                scan = run_cyclone_scan(
                    ky_values,
                    cfg=cfg,
                    Nl=args.Nl,
                    Nm=args.Nm,
                    steps=args.steps,
                    dt=args.dt,
                    tmin=args.tmin,
                    method="imex",
                    operator="gx",
                    params=params,
                )
                max_rel, mean_rel = _error_metrics(ref.ky, ref.gamma, ref.omega, scan)
                results.append(
                    SweepResult(
                        rho_star=float(rho_star),
                        omega_d_scale=float(omega_d_scale),
                        omega_star_scale=float(omega_star_scale),
                        max_rel=max_rel,
                        mean_rel=mean_rel,
                    )
                )
                print(
                    f"rho_star={rho_star:.3f} omega_d_scale={omega_d_scale:.3f} "
                    f"omega_star_scale={omega_star_scale:.3f} max_rel={max_rel:.3f} mean_rel={mean_rel:.3f}"
                )

    results.sort(key=lambda r: (r.max_rel, r.mean_rel))
    best = results[0]
    print(
        "best:",
        f"rho_star={best.rho_star:.3f}",
        f"omega_d_scale={best.omega_d_scale:.3f}",
        f"omega_star_scale={best.omega_star_scale:.3f}",
        f"max_rel={best.max_rel:.3f}",
        f"mean_rel={best.mean_rel:.3f}",
    )

    if args.output_csv:
        lines = ["rho_star,omega_d_scale,omega_star_scale,max_rel,mean_rel"]
        for r in results:
            lines.append(
                f"{r.rho_star:.6f},{r.omega_d_scale:.6f},{r.omega_star_scale:.6f},{r.max_rel:.6f},{r.mean_rel:.6f}"
            )
        with open(args.output_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
