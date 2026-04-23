#!/usr/bin/env python3
"""Generate the W7-X test-4 zonal-response panel using one common extraction policy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from spectraxgk.benchmarking import load_diagnostic_time_series, zonal_flow_response_metrics
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.plotting import set_plot_style
from spectraxgk.runtime_artifacts import run_runtime_nonlinear_with_artifacts

ROOT = Path(__file__).resolve().parents[1]

W7X_TEST4_REFERENCE = {
    "paper": "Gonzalez-Jerez et al., J. Plasma Phys. 88, 905880310 (2022)",
    "configuration": "W7-X high-mirror",
    "test": 4,
    "flux_tube": "bean",
    "torflux": 0.64,
    "alpha": 0.0,
    "adiabatic_electrons": True,
    "a_over_LTi": 0.0,
    "a_over_Ln": 0.0,
    "ky": 0.0,
    "kx_rhoi_values": [0.05, 0.07, 0.10, 0.30],
    "nperiod": 4,
    "nz_reference": 512,
    "nvpar_reference": 256,
    "nmu_reference": 32,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "examples" / "benchmarks" / "runtime_w7x_zonal_response_vmec.toml",
        help="Runtime TOML for the W7-X test-4 zonal-response benchmark.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "tools_out" / "zonal_response" / "w7x_test4_vmec",
        help="Directory for per-kx runtime outputs and extracted traces.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_zonal_response_panel.png",
        help="Publication-facing output figure path.",
    )
    parser.add_argument(
        "--kx-values",
        type=float,
        nargs="+",
        default=[0.05, 0.07, 0.10, 0.30],
        help="kx rho_i values for the W7-X test-4 sweep.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.3,
        help="Late-time fraction used for the residual window.",
    )
    parser.add_argument(
        "--initial-fraction",
        type=float,
        default=0.1,
        help="Leading fraction used for the initial normalization window.",
    )
    parser.add_argument(
        "--initial-policy",
        choices=("first_abs", "window_abs_mean"),
        default="first_abs",
        help="Initial normalization convention for the residual/GAM metrics.",
    )
    parser.add_argument(
        "--peak-fit-max-peaks",
        type=int,
        default=4,
        help="Maximum number of extrema per branch used in the damping fit.",
    )
    parser.add_argument(
        "--fit-window-tmax",
        type=float,
        default=12.0,
        help=(
            "Upper bound of the shared early-time GAM fit window. This is chosen "
            "to isolate the initial GAM before the slower stellarator-specific oscillation."
        ),
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Reuse any existing per-kx out.nc bundles instead of rerunning them.",
    )
    return parser.parse_args()


def _nearest_kx_index(path: Path, target_kx: float) -> tuple[int, float]:
    with nc.Dataset(path, "r") as ds:
        grids = ds.groups.get("Grids")
        if grids is None or "kx" not in grids.variables:
            raise ValueError(f"missing Grids/kx in {path}")
        kx = np.asarray(grids.variables["kx"][:], dtype=float)
    idx = int(np.argmin(np.abs(kx - float(target_kx))))
    return idx, float(kx[idx])


def _kx_token(kx: float) -> str:
    return f"{int(round(1000.0 * float(kx))):03d}"


def _plot_panel(
    cases: list[dict[str, object]],
    *,
    title: str,
    y_label: str,
) -> plt.Figure:
    set_plot_style()
    ncases = len(cases)
    ncols = 2
    nrows = ceil(ncases / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.7 * nrows), squeeze=False)

    for axis, case in zip(axes.flat, cases, strict=False):
        t = np.asarray(case["t"], dtype=float)
        response = np.asarray(case["response"], dtype=float)
        metrics = case["metrics"]
        response_norm = response / float(metrics.initial_level)
        axis.plot(t, response_norm, color="#0f4c81", linewidth=2.0)
        axis.axhline(float(metrics.residual_level), color="#c44e52", linestyle="--", linewidth=1.8)
        axis.axvspan(float(metrics.fit_tmin), float(metrics.fit_tmax), color="#d9ead3", alpha=0.22, linewidth=0.0)

        max_peak_t = np.asarray(metrics.max_peak_times, dtype=float)
        max_peak_y = np.asarray(metrics.max_peak_values, dtype=float)
        min_peak_t = np.asarray(metrics.min_peak_times, dtype=float)
        min_peak_y = np.asarray(metrics.min_peak_values, dtype=float)
        fit_mask_max = (max_peak_t >= float(metrics.fit_tmin)) & (max_peak_t <= float(metrics.fit_tmax))
        fit_mask_min = (min_peak_t >= float(metrics.fit_tmin)) & (min_peak_t <= float(metrics.fit_tmax))
        if np.any(fit_mask_max):
            axis.plot(max_peak_t[fit_mask_max], max_peak_y[fit_mask_max], linestyle="none", marker="o", markersize=4.5, color="#2a9d8f")
        if np.any(fit_mask_min):
            axis.plot(min_peak_t[fit_mask_min], min_peak_y[fit_mask_min], linestyle="none", marker="o", markersize=4.5, color="#7b2cbf")

        axis.set_title(fr"$k_x \rho_i = {float(case['kx_target']):.2f}$")
        axis.set_xlabel("t")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
        axis.text(
            0.03,
            0.97,
            (
                f"residual = {float(metrics.residual_level):.4f}\n"
                f"ω_GAM = {float(case['omega_R0_over_vi']):.3f} R0/vti\n"
                f"γ_GAM = {float(case['gamma_R0_over_vi']):.3f} R0/vti"
            ),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
        )

    for axis in axes.flat[ncases:]:
        axis.set_visible(False)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def main() -> int:
    args = _parse_args()
    cfg, raw = load_runtime_from_toml(args.config)
    run_cfg = dict(raw.get("run", {}))
    ky_target = float(run_cfg.get("ky", 0.0))
    nl = int(run_cfg.get("Nl", 8))
    nm = int(run_cfg.get("Nm", 32))
    dt = float(run_cfg.get("dt", cfg.time.dt))
    steps = int(run_cfg.get("steps", max(int(round(float(cfg.time.t_max) / dt)), 1)))
    sample_stride = int(run_cfg.get("sample_stride", cfg.time.sample_stride))
    diagnostics = bool(run_cfg.get("diagnostics", cfg.time.diagnostics))
    r0 = float(getattr(cfg.geometry, "R0", 1.0))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for kx_target in [float(val) for val in args.kx_values]:
        token = _kx_token(kx_target)
        out_bundle = args.out_dir / f"w7x_test4_kx{token}.out.nc"
        if not args.reuse_output or not out_bundle.exists():
            cfg_case = replace(
                cfg,
                grid=replace(cfg.grid, Lx=float(2.0 * np.pi / kx_target)),
            )
            run_runtime_nonlinear_with_artifacts(
                cfg_case,
                out=out_bundle,
                ky_target=ky_target,
                kx_target=kx_target,
                Nl=nl,
                Nm=nm,
                dt=dt,
                steps=steps,
                sample_stride=sample_stride,
                diagnostics=diagnostics,
                show_progress=False,
            )

        kx_index, kx_selected = _nearest_kx_index(out_bundle, kx_target)
        series = load_diagnostic_time_series(
            out_bundle,
            variable="Phi_zonal_mode_kxt",
            kx_index=kx_index,
            component="real",
            align_phase=True,
        )
        values = np.asarray(series.values, dtype=float)
        metrics = zonal_flow_response_metrics(
            series.t,
            values,
            tail_fraction=float(args.tail_fraction),
            initial_fraction=float(args.initial_fraction),
            initial_policy=str(args.initial_policy),
            peak_fit_max_peaks=int(args.peak_fit_max_peaks),
            damping_fit_mode="branchwise_extrema",
            frequency_fit_mode="hilbert_phase",
            fit_window_tmax=float(args.fit_window_tmax),
            hilbert_trim_fraction=0.2,
        )
        omega_r0_over_vi = float(metrics.gam_frequency) * r0
        gamma_r0_over_vi = -float(metrics.gam_damping_rate) * r0
        row = {
            "kx_target": float(kx_target),
            "kx_selected": float(kx_selected),
            "kx_index": int(kx_index),
            "source_path": str(out_bundle),
            "initial_level": float(metrics.initial_level),
            "residual_level": float(metrics.residual_level),
            "residual_std": float(metrics.residual_std),
            "response_rms": float(metrics.response_rms),
            "gam_frequency": float(metrics.gam_frequency),
            "gam_damping_rate": float(metrics.gam_damping_rate),
            "omega_R0_over_vi": float(omega_r0_over_vi),
            "gamma_R0_over_vi": float(gamma_r0_over_vi),
            "peak_count": int(metrics.peak_count),
            "peak_fit_count": int(metrics.peak_fit_count),
            "tmin": float(metrics.tmin),
            "tmax": float(metrics.tmax),
            "fit_tmin": float(metrics.fit_tmin),
            "fit_tmax": float(metrics.fit_tmax),
        }
        summary_rows.append(row)
        cases.append(
            {
                **row,
                "t": np.asarray(series.t, dtype=float),
                "response": values,
                "metrics": metrics,
            }
        )
        trace_csv = args.out_dir / f"w7x_test4_kx{token}.csv"
        np.savetxt(
            trace_csv,
            np.column_stack([np.asarray(series.t, dtype=float), values]),
            delimiter=",",
            header="t,phi_zonal_real",
            comments="",
        )

    fig = _plot_panel(
        cases,
        title="W7-X bean-tube zonal-flow relaxation (test 4)",
        y_label=r"$\phi_\mathrm{zonal}/|\phi_\mathrm{zonal}(0)|$",
    )
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")

    summary_csv = args.out_png.with_suffix(".csv")
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    meta_out = args.out_png.with_suffix(".json")
    meta_out.write_text(
        json.dumps(
            {
                "config": str(args.config),
                "initial_policy": str(args.initial_policy),
                "damping_method": "branchwise_extrema",
                "frequency_method": "hilbert_phase",
                "fit_window_tmax": float(args.fit_window_tmax),
                "literature_reference": dict(W7X_TEST4_REFERENCE),
                "cases": summary_rows,
                "validation_status": "open",
                "notes": (
                    "This panel follows the W7-X stella/GENE benchmark test-4 contract: "
                    "bean flux tube, torflux=0.64, alpha=0, adiabatic electrons, zero gradients, ky=0, "
                    "and Gaussian initial perturbations at four kx rho_i values. "
                    "For consistency with the shaped-tokamak Merlo lane, each trace is normalized with the "
                    "first-sample convention and the initial GAM is extracted with separate positive/negative-extrema "
                    "damping fits plus a Hilbert-phase frequency estimate over a common early-time window. "
                    "The default fit window cap isolates the initial GAM before the slower stellarator-specific "
                    "oscillation described in section 4.4 of the benchmark paper; this cutoff is a manuscript-policy "
                    "inference, not a quoted number from the paper itself. A frozen VMEC-backed artifact still needs "
                    "to be generated on a machine with W7-X geometry access."
                ),
                "references": [
                    "Gonzalez-Jerez et al. 2022 W7-X test-4 zonal-flow relaxation benchmark",
                    "Merlo et al. 2016 shaped-tokamak GAM benchmark for extraction-policy consistency",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
