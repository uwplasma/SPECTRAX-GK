"""Generate publication-ready figures for docs and README."""

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
    run_cyclone_scan,
    run_etg_linear,
    run_mtm_linear,
)
from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    ETGModelConfig,
    GridConfig,
    MTMBaseCase,
    MTMModelConfig,
)
from spectraxgk.plotting import (
    cyclone_comparison_figure,
    cyclone_reference_figure,
    etg_trend_figure,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
