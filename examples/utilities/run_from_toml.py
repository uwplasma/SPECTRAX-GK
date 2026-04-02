#!/usr/bin/env python3
"""Run a linear case from a TOML input file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spectraxgk.analysis import extract_eigenfunction, extract_mode_time_series
from spectraxgk.benchmarking import normalize_eigenfunction, run_linear_scan
from spectraxgk.benchmarks import load_cyclone_reference, load_etg_reference, run_cyclone_linear, run_etg_linear
from spectraxgk.grids import build_spectral_grid
from spectraxgk.io import load_case_from_toml, load_krylov_from_toml, load_linear_terms_from_toml
from spectraxgk.plotting import growth_fit_figure, scan_comparison_figure, set_plot_style


def _resolve_case(case_name: str):
    name = case_name.lower()
    if name == "cyclone":
        return run_cyclone_linear
    if name == "etg":
        return run_etg_linear
    raise ValueError(f"Unsupported case '{case_name}' (supported: cyclone, etg)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--plot", action="store_true", help="Save plots for fits/eigenfunctions/scan")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    case_name, cfg, data = load_case_from_toml(args.config, None)
    run_fn = _resolve_case(case_name)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    terms = load_linear_terms_from_toml(data)
    krylov_cfg = load_krylov_from_toml(data)
    fit_cfg = data.get("fit", {})

    scan_cfg = data.get("scan", {})
    if "ky" in scan_cfg:
        ky_values = np.asarray(scan_cfg["ky"], dtype=float)
        scan = run_linear_scan(
            ky_values=ky_values,
            run_linear_fn=run_fn,
            cfg=cfg,
            Nl=int(scan_cfg.get("Nl", 24)),
            Nm=int(scan_cfg.get("Nm", 12)),
            dt=float(scan_cfg.get("dt", cfg.time.dt)),
            steps=int(scan_cfg.get("steps", int(round(cfg.time.t_max / cfg.time.dt)))),
            method=str(scan_cfg.get("method", cfg.time.method)),
            solver=str(scan_cfg.get("solver", "krylov")),
            krylov_cfg=krylov_cfg,
            window_kw=fit_cfg,
            run_kwargs={"terms": terms} if terms is not None else None,
        )
        for ky, g, w in zip(scan.ky, scan.gamma, scan.omega):
            print(f"ky={ky:.4f} gamma={g:.6f} omega={w:.6f}")
        if args.plot:
            ref = None
            if case_name == "cyclone":
                ref = load_cyclone_reference()
            elif case_name == "etg":
                ref = load_etg_reference()
            if ref is not None:
                fig, _ax = scan_comparison_figure(
                    scan.ky,
                    scan.gamma,
                    scan.omega,
                    r"$k_y \rho_i$",
                    f"{case_name.upper()} scan",
                    x_ref=ref.ky,
                    gamma_ref=ref.gamma,
                    omega_ref=ref.omega,
                    log_x=True,
                )
                fig.savefig(outdir / f"{case_name}_scan_comparison.png")
        return 0

    run_cfg = data.get("run", {})
    ky = float(run_cfg.get("ky", 0.3))
    result = run_fn(
        ky_target=ky,
        cfg=cfg,
        Nl=int(run_cfg.get("Nl", 24)),
        Nm=int(run_cfg.get("Nm", 12)),
        solver=str(run_cfg.get("solver", "krylov")),
        method=str(run_cfg.get("method", cfg.time.method)),
        dt=float(run_cfg.get("dt", cfg.time.dt)),
        steps=int(run_cfg.get("steps", int(round(cfg.time.t_max / cfg.time.dt)))),
        krylov_cfg=krylov_cfg,
        terms=terms,
        **fit_cfg,
    )
    print(f"ky={result.ky:.4f} gamma={result.gamma:.6f} omega={result.omega:.6f}")

    if args.plot:
        grid = build_spectral_grid(cfg.grid)
        if result.t.size > 1:
            signal = extract_mode_time_series(result.phi_t, result.selection, method="project")
            fig, _ax = growth_fit_figure(result.t, signal, result.gamma, result.omega)
            fig.savefig(outdir / f"{case_name}_ky{result.ky:.3f}_fit.png")
        eigen = extract_eigenfunction(result.phi_t, result.t, result.selection, z=grid.z)
        eigen = normalize_eigenfunction(eigen, grid.z)
        set_plot_style()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(grid.z, eigen.real, label="Re")
        ax.plot(grid.z, eigen.imag, linestyle="--", label="Im")
        ax.set_xlabel(r"$\\theta$")
        ax.set_ylabel(r"$\\phi$")
        ax.set_title(f"{case_name} ky={result.ky:.3f}")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"{case_name}_ky{result.ky:.3f}_eig.png")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
