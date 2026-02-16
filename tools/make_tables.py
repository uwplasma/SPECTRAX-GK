"""Generate CSV tables for documentation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.benchmarks import load_cyclone_reference, run_cyclone_scan
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.linear import LinearParams


def main() -> int:
    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)

    ref = load_cyclone_reference()
    ky_subset = np.array([0.3, 0.4])
    gx_ky_subset = np.array([0.2, 0.3, 0.4])
    cfg = CycloneBaseCase(grid=GridConfig(Nx=8, Ny=12, Nz=24, Lx=62.8, Ly=62.8))
    low_scan = run_cyclone_scan(
        ky_subset, cfg=cfg, Nl=2, Nm=4, steps=400, dt=0.02, method="rk4"
    )
    high_scan = run_cyclone_scan(
        ky_subset, cfg=cfg, Nl=3, Nm=6, steps=300, dt=0.02, method="imex"
    )

    def build_rows(scan):
        rows = [
            "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega"
        ]
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

    low_path = outdir / "cyclone_scan_table_lowres.csv"
    low_path.write_text("\n".join(build_rows(low_scan)) + "\n", encoding="utf-8")
    high_path = outdir / "cyclone_scan_table_highres.csv"
    high_path.write_text("\n".join(build_rows(high_scan)) + "\n", encoding="utf-8")

    conv_rows = [
        "ky,gamma_low,gamma_high,omega_low,omega_high,rel_gamma_change,rel_omega_change"
    ]
    for ky, g_lo, g_hi, w_lo, w_hi in zip(
        low_scan.ky, low_scan.gamma, high_scan.gamma, low_scan.omega, high_scan.omega
    ):
        rel_g = (g_hi - g_lo) / g_hi if g_hi != 0.0 else np.nan
        rel_w = (w_hi - w_lo) / w_hi if w_hi != 0.0 else np.nan
        conv_rows.append(
            f"{ky:.3f},{g_lo:.6f},{g_hi:.6f},{w_lo:.6f},{w_hi:.6f},{rel_g:.3f},{rel_w:.3f}"
        )
    conv_path = outdir / "cyclone_scan_convergence.csv"
    conv_path.write_text("\n".join(conv_rows) + "\n", encoding="utf-8")

    full_cfg = CycloneBaseCase(
        grid=GridConfig(
            Nx=1,
            Ny=24,
            Nz=16,
            Lx=62.8,
            Ly=62.8,
            y0=20.0,
            ntheta=32,
            nperiod=2,
        )
    )
    full_params = LinearParams(
        R_over_Ln=full_cfg.model.R_over_Ln,
        R_over_LTi=full_cfg.model.R_over_LTi,
        omega_d_scale=0.2,
        omega_star_scale=0.18,
        rho_star=1.0,
    )
    full_scan = run_cyclone_scan(
        gx_ky_subset,
        cfg=full_cfg,
        Nl=2,
        Nm=4,
        steps=200,
        dt=0.02,
        tmin=2.0,
        method="imex",
        operator="full",
        params=full_params,
    )

    def build_rows_abs(scan):
        rows = [
            "ky,gamma_ref,omega_ref,gamma_full,omega_full,abs_gamma,abs_omega,rel_gamma,rel_omega"
        ]
        for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
            idx = int(np.argmin(np.abs(ref.ky - ky)))
            gamma_ref = float(ref.gamma[idx])
            omega_ref = float(ref.omega[idx])
            gamma_abs = abs(float(gamma))
            omega_abs = abs(float(omega))
            rel_gamma = (gamma_abs - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
            rel_omega = (omega_abs - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
            rows.append(
                f"{ky:.3f},{gamma_ref:.6f},{omega_ref:.6f},{gamma:.6f},{omega:.6f},{gamma_abs:.6f},{omega_abs:.6f},{rel_gamma:.3f},{rel_omega:.3f}"
            )
        return rows

    full_path = outdir / "cyclone_full_operator_scan_table.csv"
    full_path.write_text("\n".join(build_rows_abs(full_scan)) + "\n", encoding="utf-8")

    rho_values = np.array([0.9, 1.0, 1.1])
    rho_rows = ["rho_star,mean_gamma_ratio,mean_omega_ratio"]
    for rho in rho_values:
        params = LinearParams(
            R_over_Ln=full_cfg.model.R_over_Ln,
            R_over_LTi=full_cfg.model.R_over_LTi,
            omega_d_scale=0.2,
            omega_star_scale=0.18,
            rho_star=float(rho),
        )
        scan = run_cyclone_scan(
            gx_ky_subset,
            cfg=full_cfg,
            Nl=2,
            Nm=4,
            steps=200,
            dt=0.02,
            tmin=2.0,
            method="imex",
            operator="full",
            params=params,
        )
        rel_g = []
        rel_w = []
        for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
            idx = int(np.argmin(np.abs(ref.ky - ky)))
            gamma_ref = float(ref.gamma[idx])
            omega_ref = float(ref.omega[idx])
            rel_g.append(abs(float(gamma)) / gamma_ref if gamma_ref != 0.0 else np.nan)
            rel_w.append(abs(float(omega)) / omega_ref if omega_ref != 0.0 else np.nan)
        rho_rows.append(
            f"{rho:.2f},{np.nanmean(rel_g):.3f},{np.nanmean(rel_w):.3f}"
        )
    rho_path = outdir / "cyclone_rhostar_convergence.csv"
    rho_path.write_text("\n".join(rho_rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
