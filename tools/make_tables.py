"""Generate CSV tables for documentation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.benchmarks import (
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
    run_cyclone_scan,
    run_etg_linear,
    run_etg_scan,
    run_kinetic_scan,
    run_kbm_beta_scan,
    run_tem_scan,
)
from spectraxgk.config import CycloneBaseCase, ETGBaseCase, ETGModelConfig, GridConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.linear import LinearParams, LinearTerms

LINEAR_METHOD = "implicit"
TIME_SOLVER = "time"


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


WINDOWS = {
    "cyclone": dict(
        window_fraction=0.3,
        min_points=80,
        start_fraction=0.3,
        growth_weight=0.2,
        require_positive=True,
        min_amp_fraction=0.0,
    ),
    "kinetic": dict(
        window_fraction=0.35,
        min_points=120,
        start_fraction=0.4,
        growth_weight=0.2,
        require_positive=True,
        min_amp_fraction=0.05,
    ),
    "etg": dict(
        window_fraction=0.3,
        min_points=80,
        start_fraction=0.3,
        growth_weight=0.2,
        require_positive=True,
        min_amp_fraction=0.05,
    ),
    "kbm": dict(
        window_fraction=0.35,
        min_points=120,
        start_fraction=0.4,
        growth_weight=0.2,
        require_positive=True,
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


def _scale_steps(ky: np.ndarray, base_steps: int, ky_ref: float, max_steps: int) -> np.ndarray:
    scale = ky_ref / np.maximum(ky, 1.0e-6)
    steps = base_steps * np.maximum(1.0, scale)
    return np.clip(steps.astype(int), base_steps, max_steps)


def _scale_dt(ky: np.ndarray, base_dt: float, ky_ref: float) -> np.ndarray:
    scale = np.minimum(1.0, ky_ref / np.maximum(ky, 1.0e-6))
    return base_dt * scale


def main() -> int:
    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)

    ref = load_cyclone_reference()
    ky_subset = np.array([0.3, 0.4])
    cfg = CycloneBaseCase()

    low_scan = run_cyclone_scan(
        ky_subset,
        cfg=cfg,
        Nl=2,
        Nm=4,
        steps=1200,
        dt=0.01,
        method="rk4",
        **WINDOWS["cyclone"],
    )
    high_scan = run_cyclone_scan(
        ky_subset,
        cfg=cfg,
        Nl=3,
        Nm=6,
        steps=1200,
        dt=0.01,
        method="rk4",
        **WINDOWS["cyclone"],
    )

    (outdir / "cyclone_scan_table_lowres.csv").write_text(
        "\n".join(_build_rows(low_scan, ref)) + "\n", encoding="utf-8"
    )
    (outdir / "cyclone_scan_table_highres.csv").write_text(
        "\n".join(_build_rows(high_scan, ref)) + "\n", encoding="utf-8"
    )

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
    full_scan = run_cyclone_scan(
        np.array([0.2, 0.3, 0.4]),
        cfg=full_cfg,
        Nl=6,
        Nm=12,
        steps=1200,
        dt=0.01,
        method="rk4",
        terms=LinearTerms(),
        params=full_params,
        **WINDOWS["cyclone"],
    )

    full_rows = [
        "ky,gamma_ref,omega_ref,gamma_full,omega_full,abs_gamma,abs_omega,rel_gamma,rel_omega"
    ]
    for ky, gamma, omega in zip(full_scan.ky, full_scan.gamma, full_scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        gamma_abs = abs(float(gamma))
        omega_abs = abs(float(omega))
        rel_gamma = (gamma_abs - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
        rel_omega = (omega_abs - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
        full_rows.append(
            f"{ky:.3f},{gamma_ref:.6f},{omega_ref:.6f},{gamma:.6f},{omega:.6f},{gamma_abs:.6f},{omega_abs:.6f},{rel_gamma:.3f},{rel_omega:.3f}"
        )
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
        scan = run_cyclone_scan(
            np.array([0.2, 0.3, 0.4]),
            cfg=full_cfg,
            Nl=6,
            Nm=12,
            steps=1200,
            dt=0.01,
            method="rk4",
            terms=LinearTerms(),
            params=params,
            **WINDOWS["cyclone"],
        )
        rel_g = []
        rel_w = []
        for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
            idx = int(np.argmin(np.abs(ref.ky - ky)))
            gamma_ref = float(ref.gamma[idx])
            omega_ref = float(ref.omega[idx])
            rel_g.append(abs(float(gamma)) / gamma_ref if gamma_ref != 0.0 else np.nan)
            rel_w.append(abs(float(omega)) / omega_ref if omega_ref != 0.0 else np.nan)
        rho_rows.append(f"{rho:.2f},{np.nanmean(rel_g):.3f},{np.nanmean(rel_w):.3f}")
    (outdir / "cyclone_rhostar_convergence.csv").write_text(
        "\n".join(rho_rows) + "\n", encoding="utf-8"
    )

    etg_grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28, y0=0.2)
    etg_R = np.array([4.0, 6.0, 8.0, 10.0])
    etg_rows = ["R_over_LTe,gamma,omega"]
    for R in etg_R:
        cfg = ETGBaseCase(grid=etg_grid, model=ETGModelConfig(R_over_LTe=float(R)))
        res = run_etg_linear(
            cfg=cfg,
            ky_target=5.0,
            Nl=4,
            Nm=8,
            steps=1000,
            dt=0.001,
            method=LINEAR_METHOD,
            solver=TIME_SOLVER,
            mode_method="z_index",
            auto_window=True,
            **WINDOWS["etg"],
        )
        etg_rows.append(f"{R:.2f},{res.gamma:.6f},{res.omega:.6f}")
    (outdir / "etg_trend_table.csv").write_text(
        "\n".join(etg_rows) + "\n", encoding="utf-8"
    )

    # Mismatch tables against reference data (full ky/beta lists)
    cyclone_steps = _scale_steps(ref.ky, base_steps=1200, ky_ref=0.2, max_steps=6000)
    cyclone_mismatch = run_cyclone_scan(
        ref.ky,
        Nl=6,
        Nm=12,
        steps=cyclone_steps,
        dt=0.01,
        method=LINEAR_METHOD,
        solver=TIME_SOLVER,
        **WINDOWS["cyclone"],
    )
    (outdir / "cyclone_mismatch_table.csv").write_text(
        "\n".join(_build_rows(cyclone_mismatch, ref)) + "\n", encoding="utf-8"
    )

    kinetic_ref = load_cyclone_reference_kinetic()
    kinetic_steps = _scale_steps(kinetic_ref.ky, base_steps=1200, ky_ref=0.3, max_steps=6000)
    kinetic_dt = _scale_dt(kinetic_ref.ky, base_dt=0.001, ky_ref=0.3)
    kinetic_mismatch = run_kinetic_scan(
        kinetic_ref.ky,
        Nl=6,
        Nm=12,
        steps=kinetic_steps,
        dt=kinetic_dt,
        method=LINEAR_METHOD,
        solver=TIME_SOLVER,
        **WINDOWS["kinetic"],
    )
    (outdir / "kinetic_mismatch_table.csv").write_text(
        "\n".join(_build_rows(kinetic_mismatch, kinetic_ref)) + "\n", encoding="utf-8"
    )

    etg_ref = load_etg_reference()
    etg_dt = _scale_dt(etg_ref.ky, base_dt=0.0005, ky_ref=20.0)
    etg_mismatch = run_etg_scan(
        etg_ref.ky,
        Nl=6,
        Nm=12,
        steps=1200,
        dt=etg_dt,
        method=LINEAR_METHOD,
        solver=TIME_SOLVER,
        **WINDOWS["etg"],
    )
    (outdir / "etg_mismatch_table.csv").write_text(
        "\n".join(_build_rows(etg_mismatch, etg_ref)) + "\n", encoding="utf-8"
    )

    kbm_ref = load_kbm_reference()
    kbm_mismatch = run_kbm_beta_scan(
        kbm_ref.ky,
        ky_target=0.3,
        Nl=6,
        Nm=12,
        steps=1200,
        dt=0.001,
        method=LINEAR_METHOD,
        solver=TIME_SOLVER,
        **WINDOWS["kbm"],
    )
    (outdir / "kbm_mismatch_table.csv").write_text(
        "\n".join(_build_rows(kbm_mismatch, kbm_ref)) + "\n", encoding="utf-8"
    )

    tem_ref = load_tem_reference()
    tem_mismatch = run_tem_scan(
        tem_ref.ky,
        Nl=6,
        Nm=12,
        steps=1200,
        dt=0.001,
        method=LINEAR_METHOD,
        solver=TIME_SOLVER,
        **WINDOWS["tem"],
    )
    (outdir / "tem_mismatch_table.csv").write_text(
        "\n".join(_build_rows(tem_mismatch, tem_ref)) + "\n", encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
