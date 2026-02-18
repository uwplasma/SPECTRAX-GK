#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from spectraxgk.benchmarks import CycloneBaseCase, load_cyclone_reference, run_cyclone_scan
from spectraxgk.config import TimeConfig


def _parse_pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "x" not in entry:
            raise ValueError(f"Invalid pair {entry!r}, expected NxM.")
        n_l, n_m = entry.split("x", 1)
        pairs.append((int(n_l), int(n_m)))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Cyclone reduced-ky resolution sweep.")
    parser.add_argument(
        "--ky",
        default="0.10,0.30,0.45,0.55",
        help="Comma-separated ky values.",
    )
    parser.add_argument(
        "--pairs",
        default="6x12,8x16,12x24,16x32",
        help="Comma-separated Nl x Nm pairs (e.g. 6x12,8x16).",
    )
    parser.add_argument("--tmax", type=float, default=150.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--method", default="rk4")
    parser.add_argument("--use-diffrax", action="store_true")
    parser.add_argument("--out", default="docs/_static/cyclone_resolution_subset.csv")
    args = parser.parse_args()

    ky_vals = np.array([float(x) for x in args.ky.split(",") if x.strip()])
    pairs = _parse_pairs(args.pairs)
    steps = int(round(args.tmax / args.dt))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref = load_cyclone_reference()
    time_cfg = None
    if args.use_diffrax:
        time_cfg = TimeConfig(
            t_max=args.tmax,
            dt=args.dt,
            use_diffrax=True,
            diffrax_solver="Tsit5",
            diffrax_adaptive=True,
            diffrax_rtol=1.0e-5,
            diffrax_atol=1.0e-7,
            diffrax_max_steps=20000,
            progress_bar=False,
        )

    rows = []
    for Nl, Nm in pairs:
        scan = run_cyclone_scan(
            ky_vals,
            Nl=Nl,
            Nm=Nm,
            dt=args.dt,
            steps=steps,
            method=args.method,
            time_cfg=time_cfg,
        )
        for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
            idx = int(np.argmin(np.abs(ref.ky - ky)))
            gamma_ref = float(ref.gamma[idx])
            omega_ref = float(ref.omega[idx])
            rel_gamma = (gamma - gamma_ref) / gamma_ref
            rel_omega = (omega - omega_ref) / omega_ref
            rows.append(
                (
                    Nl,
                    Nm,
                    float(ky),
                    gamma_ref,
                    omega_ref,
                    float(gamma),
                    float(omega),
                    float(rel_gamma),
                    float(rel_omega),
                )
            )

    header = "Nl,Nm,ky,gamma_ref,omega_ref,gamma,omega,rel_gamma,rel_omega"
    lines = [header] + [
        ",".join(f"{val:.8g}" if isinstance(val, float) else str(val) for val in row)
        for row in rows
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
