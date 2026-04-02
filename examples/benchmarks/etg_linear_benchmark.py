import argparse
from pathlib import Path

import numpy as np

from spectraxgk import (
    ETGBaseCase,
    LinearValidationPanel,
    extract_mode_time_series,
    fit_growth_rate_auto,
    growth_fit_figure,
    linear_validation_figure,
    load_etg_reference,
    normalize_eigenfunction,
    run_etg_linear,
    run_etg_scan,
    run_scan_and_mode,
    scan_comparison_figure,
)
from spectraxgk.linear_krylov import KrylovConfig


def _etg_resolution_policy(ky: float) -> tuple[int, int]:
    if ky < 10.0:
        return 48, 16
    return 48, 16


ETG_KRYLOV = KrylovConfig(
    method="propagator",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.0,
    omega_target_factor=0.5,
    omega_cap_factor=0.5,
    omega_sign=-1,
    power_iters=80,
    power_dt=0.002,
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=2.0e-3,
)

ETG_KRYLOV_LOW = KrylovConfig(
    method="propagator",
    krylov_dim=16,
    restarts=1,
    omega_min_factor=0.0,
    omega_target_factor=0.0,
    omega_cap_factor=2.0,
    omega_sign=-1,
    power_iters=80,
    power_dt=0.002,
    shift_maxiter=40,
    shift_restart=12,
    shift_tol=2.0e-3,
)


def _etg_krylov_policy(ky: float) -> KrylovConfig:
    if ky < 10.0:
        return ETG_KRYLOV_LOW
    return ETG_KRYLOV


def main() -> None:
    parser = argparse.ArgumentParser(description="ETG benchmark (scan or single-ky).")
    parser.add_argument("--ky", type=float, default=None, help="Run a single ky value.")
    parser.add_argument("--outdir", default=".", help="Output directory.")
    parser.add_argument("--no-fit", action="store_true", help="Skip fit-window plot.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = ETGBaseCase()
    ref = load_etg_reference()
    ky_values = np.array([float(args.ky)]) if args.ky is not None else np.asarray(ref.ky)

    window_kw = dict(
        window_fraction=0.25,
        min_points=100,
        start_fraction=0.45,
        growth_weight=0.2,
        require_positive=True,
        min_amp_fraction=0.2,
    )

    scan_result = run_scan_and_mode(
        ky_values=ky_values,
        scan_fn=run_etg_scan,
        linear_fn=run_etg_linear,
        cfg=cfg,
        Nl=48,
        Nm=16,
        dt=cfg.time.dt,
        steps=int(round(cfg.time.t_max / cfg.time.dt)),
        method="imex2",
        solver="krylov",
        mode_solver="krylov",
        krylov_cfg=ETG_KRYLOV,
        window_kw=window_kw,
        auto_window=False,
        tmin=2.0,
        tmax=cfg.time.t_max,
        run_kwargs={"mode_method": "z_index", "fit_signal": "phi", "time_cfg": cfg.time},
        mode_kwargs={"fit_signal": "phi", "mode_method": "z_index"},
        resolution_policy=_etg_resolution_policy,
        krylov_policy=_etg_krylov_policy,
    )

    eig = normalize_eigenfunction(scan_result.eigenfunction, scan_result.grid.z)
    panel = LinearValidationPanel(
        name="ETG",
        z=scan_result.grid.z,
        eigenfunction=eig,
        x=scan_result.scan.ky,
        gamma=scan_result.scan.gamma,
        omega=scan_result.scan.omega,
        x_label=r"$k_y \rho_i$",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        log_x=True,
    )
    fig, _axes = linear_validation_figure([panel])
    fig.savefig(outdir / "etg_validation.png", dpi=200)

    fig, _axes = scan_comparison_figure(
        scan_result.scan.ky,
        scan_result.scan.gamma,
        scan_result.scan.omega,
        r"$k_y \rho_i$",
        "ETG comparison (GX Fig. 2b)",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        log_x=True,
    )
    fig.savefig(outdir / "etg_comparison.png", dpi=200)

    if not args.no_fit:
        ky_sel = scan_result.ky_selected
        run = run_etg_linear(
            ky_target=ky_sel,
            cfg=cfg,
            Nl=48,
            Nm=16,
            dt=0.0005,
            steps=4000,
            method="imex2",
            solver="time",
            auto_window=True,
            **window_kw,
        )
        signal = extract_mode_time_series(run.phi_t, run.selection, method="project")
        _g, _w, tmin, tmax = fit_growth_rate_auto(run.t, signal, **window_kw)
        fig, _axes = growth_fit_figure(run.t, signal, tmin=tmin, tmax=tmax, title="ETG fit window")
        fig.savefig(outdir / "etg_fit_window.png", dpi=200)


if __name__ == "__main__":
    main()
