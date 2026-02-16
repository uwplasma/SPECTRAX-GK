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
    run_cyclone_linear,
    run_cyclone_scan,
    run_etg_linear,
    run_mtm_linear,
)
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    ETGModelConfig,
    GridConfig,
    ModelConfig,
    MTMBaseCase,
    MTMModelConfig,
)
from spectraxgk.grids import build_spectral_grid
from spectraxgk.plotting import (
    cyclone_comparison_figure,
    cyclone_reference_figure,
    etg_trend_figure,
    linear_validation_figure,
    LinearValidationPanel,
    mtm_trend_figure,
)


def main() -> int:
    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)

    ref = load_cyclone_reference()
    fig, _axes = cyclone_reference_figure(ref)
    fig.savefig(outdir / "cyclone_reference.png", dpi=200)
    fig.savefig(outdir / "cyclone_reference.pdf")

    ky_sample = ref.ky[::2]
    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8))
    scan = run_cyclone_scan(ky_sample, cfg=cfg, Nl=6, Nm=12, steps=800, dt=0.01, method="rk4")
    fig, _axes = cyclone_comparison_figure(ref, scan)
    fig.savefig(outdir / "cyclone_comparison.png", dpi=200)
    fig.savefig(outdir / "cyclone_comparison.pdf")

    etg_grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    R_vals = [4.0, 6.0, 8.0, 10.0]
    etg_gamma = []
    etg_omega = []
    for R in R_vals:
        cfg = ETGBaseCase(grid=etg_grid, model=ETGModelConfig(R_over_LTe=R))
        res = run_etg_linear(
            cfg=cfg,
            ky_target=3.0,
            Nl=4,
            Nm=8,
            steps=200,
            dt=0.01,
            method="rk4",
            mode_method="z_index",
            auto_window=False,
            tmin=0.2,
            tmax=0.6,
        )
        etg_gamma.append(res.gamma)
        etg_omega.append(res.omega)
    fig, _axes = etg_trend_figure(
        np.array(R_vals), np.array(etg_gamma), np.array(etg_omega), ky_target=3.0
    )
    fig.savefig(outdir / "etg_trend.png", dpi=200)
    fig.savefig(outdir / "etg_trend.pdf")

    mtm_grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    nu_vals = [0.0, 0.1, 0.2, 0.3]
    mtm_gamma = []
    mtm_omega = []
    for nu in nu_vals:
        cfg = MTMBaseCase(grid=mtm_grid, model=MTMModelConfig(R_over_LTe=6.0, nu=nu))
        res = run_mtm_linear(
            cfg=cfg,
            ky_target=3.0,
            Nl=4,
            Nm=8,
            steps=200,
            dt=0.01,
            method="rk4",
            mode_method="z_index",
            auto_window=False,
            tmin=0.2,
            tmax=0.6,
        )
        mtm_gamma.append(res.gamma)
        mtm_omega.append(res.omega)
    fig, _axes = mtm_trend_figure(
        np.array(nu_vals), np.array(mtm_gamma), np.array(mtm_omega), ky_target=3.0
    )
    fig.savefig(outdir / "mtm_trend.png", dpi=200)
    fig.savefig(outdir / "mtm_trend.pdf")

    # README multi-panel summary
    cyclone_cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8))
    cyclone_run = run_cyclone_linear(cfg=cyclone_cfg, ky_target=0.3, Nl=6, Nm=12, steps=800, dt=0.01)
    cyclone_grid = build_spectral_grid(cyclone_cfg.grid)
    signal = extract_mode_time_series(cyclone_run.phi_t, cyclone_run.selection, method="svd")
    _g, _w, tmin, tmax = fit_growth_rate_auto(cyclone_run.t, signal)
    cyclone_mode = extract_eigenfunction(
        cyclone_run.phi_t, cyclone_run.t, cyclone_run.selection, method="svd", tmin=tmin, tmax=tmax
    )
    cyclone_scan = run_cyclone_scan(ref.ky, cfg=cyclone_cfg, Nl=6, Nm=12, steps=800, dt=0.01)

    itg_cfg = CycloneBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8),
        model=ModelConfig(R_over_LTi=3.2, R_over_LTe=0.0, R_over_Ln=1.0),
    )
    itg_run = run_cyclone_linear(cfg=itg_cfg, ky_target=0.3, Nl=6, Nm=12, steps=800, dt=0.01)
    itg_grid = build_spectral_grid(itg_cfg.grid)
    itg_signal = extract_mode_time_series(itg_run.phi_t, itg_run.selection, method="svd")
    _g, _w, itg_tmin, itg_tmax = fit_growth_rate_auto(itg_run.t, itg_signal)
    itg_mode = extract_eigenfunction(
        itg_run.phi_t, itg_run.t, itg_run.selection, method="svd", tmin=itg_tmin, tmax=itg_tmax
    )
    itg_scan = run_cyclone_scan(ref.ky, cfg=itg_cfg, Nl=6, Nm=12, steps=800, dt=0.01)

    etg_cfg = ETGBaseCase(grid=GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28))
    etg_run = run_etg_linear(cfg=etg_cfg, ky_target=3.0, Nl=4, Nm=8, steps=200, dt=0.01, auto_window=False, tmin=0.2, tmax=0.6, mode_method="z_index")
    etg_grid = build_spectral_grid(etg_cfg.grid)
    etg_mode = extract_eigenfunction(
        etg_run.phi_t, etg_run.t, etg_run.selection, method="snapshot", tmin=0.2, tmax=0.6
    )
    etg_panel_x = np.array(R_vals)

    mtm_cfg = MTMBaseCase(grid=GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28))
    mtm_run = run_mtm_linear(cfg=mtm_cfg, ky_target=3.0, Nl=4, Nm=8, steps=200, dt=0.01, auto_window=False, tmin=0.2, tmax=0.6, mode_method="z_index")
    mtm_grid = build_spectral_grid(mtm_cfg.grid)
    mtm_mode = extract_eigenfunction(
        mtm_run.phi_t, mtm_run.t, mtm_run.selection, method="snapshot", tmin=0.2, tmax=0.6
    )
    mtm_panel_x = np.array(nu_vals)

    panels = [
        LinearValidationPanel(
            name="Cyclone",
            z=cyclone_grid.z,
            eigenfunction=cyclone_mode,
            x=cyclone_scan.ky,
            gamma=cyclone_scan.gamma,
            omega=cyclone_scan.omega,
            x_label=r"$k_y \rho_i$",
        ),
        LinearValidationPanel(
            name="ITG",
            z=itg_grid.z,
            eigenfunction=itg_mode,
            x=itg_scan.ky,
            gamma=itg_scan.gamma,
            omega=itg_scan.omega,
            x_label=r"$k_y \rho_i$",
        ),
        LinearValidationPanel(
            name="ETG",
            z=etg_grid.z,
            eigenfunction=etg_mode,
            x=etg_panel_x,
            gamma=np.array(etg_gamma),
            omega=np.array(etg_omega),
            x_label=r"$R/L_{Te}$",
        ),
        LinearValidationPanel(
            name="MTM",
            z=mtm_grid.z,
            eigenfunction=mtm_mode,
            x=mtm_panel_x,
            gamma=np.array(mtm_gamma),
            omega=np.array(mtm_omega),
            x_label=r"$\nu$",
        ),
    ]
    fig, _axes = linear_validation_figure(panels)
    fig.savefig(outdir / "linear_summary.png", dpi=200)
    fig.savefig(outdir / "linear_summary.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
