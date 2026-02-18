"""Compute mismatch tables using diffrax time integration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.benchmarks import (
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
    run_cyclone_scan,
    run_etg_scan,
    run_kinetic_scan,
    run_kbm_beta_scan,
    run_tem_scan,
)
from spectraxgk.config import TimeConfig
from tools.make_tables import WINDOWS, _scale_dt, _scale_steps


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


def _summarize(name: str, scan, ref, eps: float = 1.0e-6) -> None:
    rel_gamma = []
    rel_omega = []
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        if abs(gamma_ref) > eps:
            rel_gamma.append(abs((gamma - gamma_ref) / gamma_ref))
        if abs(omega_ref) > eps:
            rel_omega.append(abs((omega - omega_ref) / omega_ref))
    rel_gamma = np.array(rel_gamma)
    rel_omega = np.array(rel_omega)
    print(
        f"{name}: mean|rel gamma|={np.nanmean(rel_gamma):.3f} "
        f"max|rel gamma|={np.nanmax(rel_gamma):.3f} "
        f"mean|rel omega|={np.nanmean(rel_omega):.3f} "
        f"max|rel omega|={np.nanmax(rel_omega):.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute diffrax mismatch tables.")
    parser.add_argument("--outdir", default=str(ROOT / "docs" / "_static"))
    parser.add_argument("--suffix", default="diffrax")
    parser.add_argument(
        "--case",
        choices=("cyclone", "kinetic", "etg", "kbm", "tem", "all"),
        default="all",
        help="Select a single benchmark case to run.",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-7)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--fixed-step", action="store_true")
    parser.add_argument("--Nl", type=int, default=6)
    parser.add_argument("--Nm", type=int, default=12)
    parser.add_argument("--tmax-cyclone", type=float, default=150.0)
    parser.add_argument("--tmax-kinetic", type=float, default=40.0)
    parser.add_argument("--tmax-etg", type=float, default=40.0)
    parser.add_argument("--tmax-kbm", type=float, default=40.0)
    parser.add_argument("--tmax-tem", type=float, default=40.0)
    parser.add_argument(
        "--no-scale-steps",
        action="store_true",
        help="Use fixed steps based on tmax/dt instead of ky-dependent scaling.",
    )
    args = parser.parse_args()

    adaptive = not args.fixed_step
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def time_cfg(dt: float, t_max: float) -> TimeConfig:
        return TimeConfig(
            t_max=t_max,
            dt=dt,
            diffrax_solver="Tsit5",
            diffrax_adaptive=adaptive,
            diffrax_rtol=args.rtol,
            diffrax_atol=args.atol,
            diffrax_max_steps=args.max_steps,
            progress_bar=False,
        )

    if args.case in ("cyclone", "all"):
        ref = load_cyclone_reference()
        if args.no_scale_steps:
            cyclone_steps = int(round(args.tmax_cyclone / 0.01))
        else:
            cyclone_steps = _scale_steps(ref.ky, base_steps=1200, ky_ref=0.2, max_steps=6000)
        cyclone_cfg = time_cfg(dt=0.01, t_max=args.tmax_cyclone)
        cyclone_scan = run_cyclone_scan(
            ref.ky,
            Nl=args.Nl,
            Nm=args.Nm,
            steps=cyclone_steps,
            dt=0.01,
            time_cfg=cyclone_cfg,
            **WINDOWS["cyclone"],
        )
        (outdir / f"cyclone_mismatch_table_{args.suffix}.csv").write_text(
            "\n".join(_build_rows(cyclone_scan, ref)) + "\n", encoding="utf-8"
        )
        _summarize("Cyclone", cyclone_scan, ref)

    if args.case in ("kinetic", "all"):
        kinetic_ref = load_cyclone_reference_kinetic()
        if args.no_scale_steps:
            kinetic_steps = int(round(args.tmax_kinetic / 0.001))
        else:
            kinetic_steps = _scale_steps(kinetic_ref.ky, base_steps=1200, ky_ref=0.3, max_steps=6000)
        kinetic_cfg = time_cfg(dt=0.001, t_max=args.tmax_kinetic)
        kinetic_scan = run_kinetic_scan(
            kinetic_ref.ky,
            Nl=args.Nl,
            Nm=args.Nm,
            steps=kinetic_steps,
            dt=0.001,
            time_cfg=kinetic_cfg,
            **WINDOWS["kinetic"],
        )
        (outdir / f"kinetic_mismatch_table_{args.suffix}.csv").write_text(
            "\n".join(_build_rows(kinetic_scan, kinetic_ref)) + "\n", encoding="utf-8"
        )
        _summarize("Kinetic ITG", kinetic_scan, kinetic_ref)

    if args.case in ("etg", "all"):
        etg_ref = load_etg_reference()
        if args.no_scale_steps:
            etg_dt = 0.0005
        else:
            etg_dt = _scale_dt(etg_ref.ky, base_dt=0.0005, ky_ref=20.0)
        etg_cfg = time_cfg(dt=0.0005, t_max=args.tmax_etg)
        etg_scan = run_etg_scan(
            etg_ref.ky,
            Nl=args.Nl,
            Nm=args.Nm,
            steps=1200,
            dt=etg_dt,
            time_cfg=etg_cfg,
            **WINDOWS["etg"],
        )
        (outdir / f"etg_mismatch_table_{args.suffix}.csv").write_text(
            "\n".join(_build_rows(etg_scan, etg_ref)) + "\n", encoding="utf-8"
        )
        _summarize("ETG", etg_scan, etg_ref)

    if args.case in ("kbm", "all"):
        kbm_ref = load_kbm_reference()
        kbm_cfg = time_cfg(dt=0.001, t_max=args.tmax_kbm)
        kbm_scan = run_kbm_beta_scan(
            kbm_ref.ky,
            ky_target=0.3,
            Nl=args.Nl,
            Nm=args.Nm,
            steps=1200,
            dt=0.001,
            time_cfg=kbm_cfg,
            **WINDOWS["kbm"],
        )
        (outdir / f"kbm_mismatch_table_{args.suffix}.csv").write_text(
            "\n".join(_build_rows(kbm_scan, kbm_ref)) + "\n", encoding="utf-8"
        )
        _summarize("KBM", kbm_scan, kbm_ref)

    if args.case in ("tem", "all"):
        tem_ref = load_tem_reference()
        tem_cfg = time_cfg(dt=0.001, t_max=args.tmax_tem)
        tem_scan = run_tem_scan(
            tem_ref.ky,
            Nl=args.Nl,
            Nm=args.Nm,
            steps=1200,
            dt=0.001,
            time_cfg=tem_cfg,
            **WINDOWS["tem"],
        )
        (outdir / f"tem_mismatch_table_{args.suffix}.csv").write_text(
            "\n".join(_build_rows(tem_scan, tem_ref)) + "\n", encoding="utf-8"
        )
        _summarize("TEM", tem_scan, tem_ref)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
