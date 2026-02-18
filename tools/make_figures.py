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
    linear_validation_figure,
    LinearValidationPanel,
)
from spectraxgk.linear_krylov import KrylovConfig


def _scale_steps(ky: np.ndarray, base_steps: int, ky_ref: float, max_steps: int) -> np.ndarray:
    scale = ky_ref / np.maximum(ky, 1.0e-6)
    steps = base_steps * np.maximum(1.0, scale)
    return np.clip(steps.astype(int), base_steps, max_steps)


def _scale_dt(ky: np.ndarray, base_dt: float, ky_ref: float) -> np.ndarray:
    scale = np.minimum(1.0, ky_ref / np.maximum(ky, 1.0e-6))
    return base_dt * scale


SCAN_SOLVER = "krylov"
MODE_SOLVER = "time"
MODE_METHOD = "imex2"
CYCLONE_KRYLOV = KrylovConfig(method="propagator", power_iters=200, power_dt=0.01)
KINETIC_KRYLOV = KrylovConfig(method="propagator", power_iters=240, power_dt=0.001)
ETG_KRYLOV = KrylovConfig(method="propagator", power_iters=240, power_dt=0.0005)
KBM_KRYLOV = KrylovConfig(method="propagator", power_iters=240, power_dt=0.001)
TEM_KRYLOV = KrylovConfig(method="propagator", power_iters=240, power_dt=0.001)

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


def _eigenfunction_panel(run, grid, window_kw):
    signal = extract_mode_time_series(run.phi_t, run.selection, method="project")
    _g, _w, tmin, tmax = fit_growth_rate_auto(run.t, signal, **window_kw)
    eig = extract_eigenfunction(
        run.phi_t, run.t, run.selection, z=grid.z, method="snapshot", tmin=tmin, tmax=tmax
    )
    return eig


def _scan_and_mode(
    scan_fn,
    linear_fn,
    ky_values,
    cfg,
    Nl,
    Nm,
    steps,
    dt,
    window_kw,
    *,
    scan_solver: str,
    mode_solver: str,
    mode_method: str,
):
    krylov_cfg = None
    if scan_solver.lower() == "krylov":
        if scan_fn is run_cyclone_scan:
            krylov_cfg = CYCLONE_KRYLOV
        elif scan_fn is run_kinetic_scan:
            krylov_cfg = KINETIC_KRYLOV
        elif scan_fn is run_etg_scan:
            krylov_cfg = ETG_KRYLOV
        elif scan_fn is run_tem_scan:
            krylov_cfg = TEM_KRYLOV
    scan = scan_fn(
        ky_values,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        steps=steps,
        dt=dt,
        method=mode_method,
        solver=scan_solver,
        krylov_cfg=krylov_cfg,
        **window_kw,
    )
    sel_idx = int(np.nanargmax(scan.gamma))
    ky_sel = float(scan.ky[sel_idx])
    steps_run = int(steps[sel_idx]) if isinstance(steps, np.ndarray) else int(steps)
    dt_run = float(dt[sel_idx]) if isinstance(dt, np.ndarray) else float(dt)
    run = linear_fn(
        cfg=cfg,
        ky_target=ky_sel,
        Nl=Nl,
        Nm=Nm,
        steps=steps_run,
        dt=dt_run,
        method=mode_method,
        solver=mode_solver,
        **window_kw,
    )
    grid = build_spectral_grid(cfg.grid)
    mode = _eigenfunction_panel(run, grid, window_kw)
    return scan, mode, grid, ky_sel


def main() -> int:
    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)

    # Cyclone reference (adiabatic electrons)
    ref = load_cyclone_reference()
    fig, _axes = cyclone_reference_figure(ref)
    fig.savefig(outdir / "cyclone_reference.png", dpi=200)
    fig.savefig(outdir / "cyclone_reference.pdf")

    cfg_cyc = CycloneBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8, y0=20.0, ntheta=32, nperiod=2)
    )
    scan_ky = ref.ky[::2]
    cyclone_steps = _scale_steps(scan_ky, base_steps=800, ky_ref=0.2, max_steps=4000)
    scan = run_cyclone_scan(
        scan_ky,
        cfg=cfg_cyc,
        Nl=6,
        Nm=16,
        steps=cyclone_steps,
        dt=0.01,
        method=MODE_METHOD,
        solver=SCAN_SOLVER,
        krylov_cfg=CYCLONE_KRYLOV,
        **WINDOWS["cyclone"],
    )
    fig, _axes = cyclone_comparison_figure(ref, scan)
    fig.savefig(outdir / "cyclone_comparison.png", dpi=200)
    fig.savefig(outdir / "cyclone_comparison.pdf")

    # Multi-panel summary: cyclone, kinetic ITG, ETG, KBM, TEM
    cyclone_scan, cyclone_mode, cyclone_grid, _ = _scan_and_mode(
        run_cyclone_scan,
        run_cyclone_linear,
        scan_ky,
        cfg_cyc,
        Nl=6,
        Nm=16,
        steps=cyclone_steps,
        dt=0.01,
        window_kw=WINDOWS["cyclone"],
        scan_solver=SCAN_SOLVER,
        mode_solver=MODE_SOLVER,
        mode_method=MODE_METHOD,
    )

    kinetic_ref = load_cyclone_reference_kinetic()
    cfg_kin = KineticElectronBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=62.8, Ly=62.8, y0=20.0, ntheta=32, nperiod=2)
    )
    kinetic_ky = kinetic_ref.ky[::2]
    kinetic_steps = _scale_steps(kinetic_ky, base_steps=1200, ky_ref=0.3, max_steps=6000)
    kinetic_dt = _scale_dt(kinetic_ky, base_dt=0.001, ky_ref=0.3)
    kinetic_scan, kinetic_mode, kinetic_grid, _ = _scan_and_mode(
        run_kinetic_scan,
        run_kinetic_linear,
        kinetic_ky,
        cfg_kin,
        Nl=6,
        Nm=16,
        steps=kinetic_steps,
        dt=kinetic_dt,
        window_kw=WINDOWS["kinetic"],
        scan_solver=SCAN_SOLVER,
        mode_solver=MODE_SOLVER,
        mode_method=MODE_METHOD,
    )

    etg_ref = load_etg_reference()
    cfg_etg = ETGBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=96, Lx=6.28, Ly=6.28, y0=0.2, ntheta=32, nperiod=2)
    )
    etg_ky = etg_ref.ky[::2]
    etg_dt = _scale_dt(etg_ky, base_dt=0.0005, ky_ref=20.0)
    etg_scan, etg_mode, etg_grid, _ = _scan_and_mode(
        run_etg_scan,
        run_etg_linear,
        etg_ky,
        cfg_etg,
        Nl=6,
        Nm=16,
        steps=1200,
        dt=etg_dt,
        window_kw=WINDOWS["etg"],
        scan_solver=SCAN_SOLVER,
        mode_solver=MODE_SOLVER,
        mode_method=MODE_METHOD,
    )

    kbm_ref = load_kbm_reference()
    cfg_kbm = KBMBaseCase(
        grid=GridConfig(Nx=1, Ny=12, Nz=96, Lx=62.8, Ly=62.8, y0=10.0, ntheta=32, nperiod=2)
    )
    kbm_beta = kbm_ref.ky[::2]
    kbm_scan = run_kbm_beta_scan(
        kbm_beta,
        cfg=cfg_kbm,
        ky_target=0.3,
        Nl=6,
        Nm=16,
        steps=1200,
        dt=0.001,
        method=MODE_METHOD,
        solver=SCAN_SOLVER,
        krylov_cfg=KBM_KRYLOV,
        **WINDOWS["kbm"],
    )
    kbm_run = run_kinetic_linear(
        cfg=cfg_kbm,
        ky_target=0.3,
        Nl=6,
        Nm=16,
        steps=1200,
        dt=0.001,
        method=MODE_METHOD,
        solver=MODE_SOLVER,
        **WINDOWS["kbm"],
    )
    kbm_grid = build_spectral_grid(cfg_kbm.grid)
    kbm_mode = _eigenfunction_panel(kbm_run, kbm_grid, WINDOWS["kbm"])

    tem_ref = load_tem_reference()
    cfg_tem = TEMBaseCase(
        grid=GridConfig(Nx=1, Ny=24, Nz=160, Lx=62.8, Ly=62.8, y0=20.0, ntheta=32, nperiod=3)
    )
    tem_ky = tem_ref.ky[::2]
    tem_scan, tem_mode, tem_grid, _ = _scan_and_mode(
        run_tem_scan,
        run_tem_linear,
        tem_ky,
        cfg_tem,
        Nl=6,
        Nm=16,
        steps=1200,
        dt=0.001,
        window_kw=WINDOWS["tem"],
        scan_solver=SCAN_SOLVER,
        mode_solver=MODE_SOLVER,
        mode_method=MODE_METHOD,
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
            log_x=True,
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
            log_x=True,
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
            log_x=True,
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
            log_x=True,
        ),
    ]
    fig, _axes = linear_validation_figure(panels)
    fig.savefig(outdir / "linear_summary.png", dpi=200)
    fig.savefig(outdir / "linear_summary.pdf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
