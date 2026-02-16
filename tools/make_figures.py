"""Generate publication-ready figures for docs and README."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.analysis import extract_eigenfunction, fit_growth_rate_auto, extract_mode_time_series
from spectraxgk.benchmarks import (
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
    run_cyclone_linear,
    run_cyclone_scan,
    run_etg_linear,
    run_etg_scan,
    run_kinetic_linear,
    run_kinetic_scan,
    run_kbm_beta_scan,
    run_tem_linear,
    run_tem_scan,
)
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    ETGModelConfig,
    GridConfig,
    KineticElectronBaseCase,
    KBMBaseCase,
    TEMBaseCase,
)
from spectraxgk.grids import build_spectral_grid
from spectraxgk.plotting import (
    cyclone_comparison_figure,
    cyclone_reference_figure,
    growth_rate_heatmap,
    linear_validation_figure,
    LinearValidationPanel,
)


def _eigenfunction_panel(run, grid):
    signal = extract_mode_time_series(run.phi_t, run.selection, method="svd")
    _g, _w, tmin, tmax = fit_growth_rate_auto(run.t, signal)
    eig = extract_eigenfunction(
        run.phi_t, run.t, run.selection, z=grid.z, method="svd", tmin=tmin, tmax=tmax
    )
    return eig


def _scan_and_mode(scan_fn, linear_fn, ky_values, cfg, Nl, Nm, steps, dt):
    scan = scan_fn(ky_values, cfg=cfg, Nl=Nl, Nm=Nm, steps=steps, dt=dt, method="rk4")
    ky_sel = float(scan.ky[int(np.nanargmax(scan.gamma))])
    run = linear_fn(cfg=cfg, ky_target=ky_sel, Nl=Nl, Nm=Nm, steps=steps, dt=dt, method="rk4")
    grid = build_spectral_grid(cfg.grid)
    mode = _eigenfunction_panel(run, grid)
    return scan, mode, grid, ky_sel


def _gradient_heatmap(case, ky_target, cfg_factory, Rt_vals, Rn_vals, out_path, Nl, Nm):
    gamma = np.zeros((len(Rt_vals), len(Rn_vals)))
    for i, Rt in enumerate(Rt_vals):
        for j, Rn in enumerate(Rn_vals):
            cfg = cfg_factory(Rt, Rn)
            result = case(cfg=cfg, ky_target=ky_target, Nl=Nl, Nm=Nm, steps=120, dt=0.02, method="rk4")
            gamma[i, j] = result.gamma
    fig, _ax = growth_rate_heatmap(
        Rn_vals,
        Rt_vals,
        gamma,
        title="Growth rate vs gradients",
        x_label=r"$R/L_n$",
        y_label=r"$R/L_T$",
    )
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))


def main() -> int:
    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)

    # Cyclone reference (adiabatic electrons)
    ref = load_cyclone_reference()
    fig, _axes = cyclone_reference_figure(ref)
    fig.savefig(outdir / "cyclone_reference.png", dpi=200)
    fig.savefig(outdir / "cyclone_reference.pdf")

    cfg_cyc = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8))
    scan_ky = ref.ky[::2]
    scan = run_cyclone_scan(scan_ky, cfg=cfg_cyc, Nl=6, Nm=16, steps=500, dt=0.01, method="rk4")
    fig, _axes = cyclone_comparison_figure(ref, scan)
    fig.savefig(outdir / "cyclone_comparison.png", dpi=200)
    fig.savefig(outdir / "cyclone_comparison.pdf")

    # Multi-panel summary: cyclone, kinetic ITG, ETG, KBM, TEM
    cyclone_scan, cyclone_mode, cyclone_grid, _ = _scan_and_mode(
        run_cyclone_scan, run_cyclone_linear, scan_ky, cfg_cyc, Nl=6, Nm=16, steps=500, dt=0.01
    )

    kinetic_ref = load_cyclone_reference_kinetic()
    cfg_kin = KineticElectronBaseCase(grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8))
    kinetic_ky = kinetic_ref.ky[::2]
    kinetic_scan, kinetic_mode, kinetic_grid, _ = _scan_and_mode(
        run_kinetic_scan, run_kinetic_linear, kinetic_ky, cfg_kin, Nl=6, Nm=16, steps=500, dt=0.01
    )

    etg_ref = load_etg_reference()
    cfg_etg = ETGBaseCase(grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=6.28, Ly=6.28))
    etg_ky = etg_ref.ky[::2]
    etg_scan, etg_mode, etg_grid, _ = _scan_and_mode(
        run_etg_scan, run_etg_linear, etg_ky, cfg_etg, Nl=6, Nm=16, steps=400, dt=0.01
    )

    kbm_ref = load_kbm_reference()
    cfg_kbm = KBMBaseCase(grid=GridConfig(Nx=1, Ny=12, Nz=96, Lx=62.8, Ly=62.8, y0=10.0, ntheta=32, nperiod=2))
    kbm_beta = kbm_ref.ky[::2]
    kbm_scan = run_kbm_beta_scan(kbm_beta, cfg=cfg_kbm, ky_target=0.3, Nl=6, Nm=16, steps=400, dt=0.01)
    kbm_run = run_kinetic_linear(cfg=cfg_kbm, ky_target=0.3, Nl=6, Nm=16, steps=400, dt=0.01)
    kbm_grid = build_spectral_grid(cfg_kbm.grid)
    kbm_mode = _eigenfunction_panel(kbm_run, kbm_grid)

    tem_ref = load_tem_reference()
    cfg_tem = TEMBaseCase(grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8, y0=20.0, ntheta=32, nperiod=2))
    tem_ky = tem_ref.ky[::2]
    tem_scan, tem_mode, tem_grid, _ = _scan_and_mode(
        run_tem_scan, run_tem_linear, tem_ky, cfg_tem, Nl=6, Nm=16, steps=500, dt=0.01
    )

    panels = [
        LinearValidationPanel(
            name="Cyclone",
            z=cyclone_grid.z,
            eigenfunction=cyclone_mode,
            x=cyclone_scan.ky,
            gamma=cyclone_scan.gamma,
            omega=cyclone_scan.omega,
            x_label=r"$k_y \rho_i$",
            x_ref=ref.ky,
            gamma_ref=ref.gamma,
            omega_ref=ref.omega,
            ref_label="Reference",
        ),
        LinearValidationPanel(
            name="Kinetic ITG",
            z=kinetic_grid.z,
            eigenfunction=kinetic_mode,
            x=kinetic_scan.ky,
            gamma=kinetic_scan.gamma,
            omega=kinetic_scan.omega,
            x_label=r"$k_y \rho_i$",
            x_ref=kinetic_ref.ky,
            gamma_ref=kinetic_ref.gamma,
            omega_ref=kinetic_ref.omega,
            ref_label="Reference",
        ),
        LinearValidationPanel(
            name="ETG",
            z=etg_grid.z,
            eigenfunction=etg_mode,
            x=etg_scan.ky,
            gamma=etg_scan.gamma,
            omega=etg_scan.omega,
            x_label=r"$k_y \rho_i$",
            x_ref=etg_ref.ky,
            gamma_ref=etg_ref.gamma,
            omega_ref=etg_ref.omega,
            ref_label="Reference",
        ),
        LinearValidationPanel(
            name="KBM",
            z=kbm_grid.z,
            eigenfunction=kbm_mode,
            x=kbm_scan.ky,
            gamma=kbm_scan.gamma,
            omega=kbm_scan.omega,
            x_label=r"$\beta_{ref}$",
            x_ref=kbm_ref.ky,
            gamma_ref=kbm_ref.gamma,
            omega_ref=kbm_ref.omega,
            ref_label="Reference",
        ),
        LinearValidationPanel(
            name="TEM",
            z=tem_grid.z,
            eigenfunction=tem_mode,
            x=tem_scan.ky,
            gamma=tem_scan.gamma,
            omega=tem_scan.omega,
            x_label=r"$k_y \rho_s$",
            x_ref=tem_ref.ky,
            gamma_ref=tem_ref.gamma,
            omega_ref=tem_ref.omega,
            ref_label="Reference",
        ),
    ]
    fig, _axes = linear_validation_figure(panels)
    fig.savefig(outdir / "linear_summary.png", dpi=200)
    fig.savefig(outdir / "linear_summary.pdf")

    # Gradient heatmaps
    Rt_vals = np.linspace(2.0, 8.0, 3)
    Rn_vals = np.linspace(0.5, 3.0, 3)

    _gradient_heatmap(
        run_cyclone_linear,
        0.3,
        lambda Rt, Rn: CycloneBaseCase(grid=cfg_cyc.grid, model=cfg_cyc.model.__class__(R_over_LTi=float(Rt), R_over_Ln=float(Rn), R_over_LTe=0.0)),
        Rt_vals,
        Rn_vals,
        outdir / "cyclone_heatmap.png",
        Nl=6,
        Nm=16,
    )

    _gradient_heatmap(
        run_etg_linear,
        20.0,
        lambda Rt, Rn: ETGBaseCase(grid=cfg_etg.grid, model=ETGModelConfig(R_over_LTe=float(Rt), R_over_Ln=float(Rn))),
        Rt_vals,
        Rn_vals,
        outdir / "etg_heatmap.png",
        Nl=6,
        Nm=16,
    )

    _gradient_heatmap(
        run_tem_linear,
        0.25,
        lambda Rt, Rn: TEMBaseCase(grid=cfg_tem.grid, model=cfg_tem.model.__class__(R_over_LTi=float(Rt), R_over_LTe=float(Rt), R_over_Ln=float(Rn))),
        Rt_vals,
        Rn_vals,
        outdir / "tem_heatmap.png",
        Nl=6,
        Nm=16,
    )

    _gradient_heatmap(
        run_kinetic_linear,
        0.3,
        lambda Rt, Rn: KineticElectronBaseCase(grid=cfg_kin.grid, model=cfg_kin.model.__class__(R_over_LTi=float(Rt), R_over_LTe=float(Rt), R_over_Ln=float(Rn))),
        Rt_vals,
        Rn_vals,
        outdir / "kinetic_heatmap.png",
        Nl=6,
        Nm=16,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
