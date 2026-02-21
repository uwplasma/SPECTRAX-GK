import argparse
from pathlib import Path

import numpy as np

from spectraxgk import (
    CycloneBaseCase,
    LinearValidationPanel,
    extract_mode_time_series,
    fit_growth_rate_auto,
    growth_fit_figure,
    linear_validation_figure,
    load_cyclone_reference,
    normalize_eigenfunction,
    run_cyclone_linear,
    run_cyclone_scan,
    run_scan_and_mode,
    scan_comparison_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cyclone base case benchmark.")
    parser.add_argument("--ky", type=float, default=None, help="Run a single ky value.")
    parser.add_argument("--outdir", default=".", help="Output directory.")
    parser.add_argument("--no-fit", action="store_true", help="Skip fit-window plot.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = CycloneBaseCase()
    ref = load_cyclone_reference()
    ky_values = np.array([float(args.ky)]) if args.ky is not None else np.asarray(ref.ky)

    window_kw = dict(
        window_fraction=0.3,
        min_points=80,
        start_fraction=0.3,
        growth_weight=0.2,
        require_positive=True,
        min_amp_fraction=0.0,
    )

    scan_result = run_scan_and_mode(
        ky_values=ky_values,
        scan_fn=run_cyclone_scan,
        linear_fn=run_cyclone_linear,
        cfg=cfg,
        Nl=48,
        Nm=16,
        dt=0.002,
        steps=5000,
        method="imex2",
        solver="time",
        mode_solver="time",
        krylov_cfg=None,
        window_kw=window_kw,
        auto_window=True,
    )

    eig = normalize_eigenfunction(scan_result.eigenfunction, scan_result.grid.z)
    panel = LinearValidationPanel(
        name="Cyclone",
        z=scan_result.grid.z,
        eigenfunction=eig,
        x=scan_result.scan.ky,
        gamma=scan_result.scan.gamma,
        omega=scan_result.scan.omega,
        x_label=r"$k_y \\rho_i$",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        log_x=True,
    )
    fig, _axes = linear_validation_figure([panel])
    fig.savefig(outdir / "cyclone_validation.png", dpi=200)

    fig, _axes = scan_comparison_figure(
        scan_result.scan.ky,
        scan_result.scan.gamma,
        scan_result.scan.omega,
        r"$k_y \\rho_i$",
        "Cyclone comparison",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        log_x=True,
    )
    fig.savefig(outdir / "cyclone_comparison.png", dpi=200)

    if not args.no_fit:
        ky_sel = scan_result.ky_selected
        run = run_cyclone_linear(
            ky_target=ky_sel,
            cfg=cfg,
            Nl=48,
            Nm=16,
            dt=0.002,
            steps=5000,
            method="imex2",
            solver="time",
            auto_window=True,
            **window_kw,
        )
        signal = extract_mode_time_series(run.phi_t, run.selection, method="project")
        _g, _w, tmin, tmax = fit_growth_rate_auto(run.t, signal, **window_kw)
        fig, _axes = growth_fit_figure(run.t, signal, tmin=tmin, tmax=tmax, title="Cyclone fit window")
        fig.savefig(outdir / "cyclone_fit_window.png", dpi=200)


if __name__ == "__main__":
    main()
