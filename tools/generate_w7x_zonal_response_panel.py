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
    "observable": "unweighted line-averaged electrostatic potential",
    "nperiod": 4,
    "nz_reference": 512,
    "nvpar_reference": 256,
    "nmu_reference": 32,
    "normalization": "line-averaged potential normalized to its t=0 line-average value",
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
        "--initial-normalization",
        choices=("init_amp", "line_first"),
        default="line_first",
        help=(
            "Reference normalization for the plotted response. The W7-X test-4 "
            "text normalizes the line-averaged response to its t=0 value, so "
            "the default uses the first nonzero line-average sample. init_amp "
            "is retained for explicit normalization audits."
        ),
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
        "--time-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to runtime times before plotting/extraction. "
            "The default is 1 because runtime samples are already interpreted "
            "on the paper's t v_ti/a axis; non-unit values are for explicit "
            "axis-calibration audits only."
        ),
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Reuse any existing per-kx out.nc bundles instead of rerunning them.",
    )
    parser.add_argument(
        "--resume-output",
        action="store_true",
        help=(
            "Continue each per-kx bundle from its matching restart file when it exists, "
            "appending diagnostics to the existing out.nc history."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override the runtime time step without editing the tracked benchmark TOML.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the number of fixed RK steps without editing the tracked benchmark TOML.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=None,
        help="Override the diagnostic sample stride without editing the tracked benchmark TOML.",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=None,
        help=(
            "Split fixed-step runtime generation into restartable chunks. "
            "This enables fail-fast nonfinite checks during long stability sweeps."
        ),
    )
    parser.add_argument(
        "--Nl",
        type=int,
        default=None,
        help="Override the Laguerre moment count without editing the tracked benchmark TOML.",
    )
    parser.add_argument(
        "--Nm",
        type=int,
        default=None,
        help="Override the Hermite moment count without editing the tracked benchmark TOML.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print runtime progress while generating missing per-kx bundles.",
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


def _artifact_path(path: Path | str) -> str:
    """Return a stable repo-relative path for tracked metadata when possible."""

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _finite_or_none(value: float) -> float | None:
    val = float(value)
    return val if np.isfinite(val) else None


def _format_metric(value: object, *, fmt: str = ".3f", missing: str = "not fitted") -> str:
    if value is None:
        return missing
    val = float(value)
    if not np.isfinite(val):
        return missing
    return f"{val:{fmt}}"


def _initial_level_override(args: argparse.Namespace, cfg: object) -> float | None:
    if str(args.initial_normalization) == "line_first":
        return None
    init = getattr(cfg, "init", None)
    init_amp = float(getattr(init, "init_amp", 1.0))
    if not np.isfinite(init_amp) or init_amp == 0.0:
        raise ValueError("init.init_amp must be finite and non-zero for --initial-normalization=init_amp")
    return abs(init_amp)


def _normalization_label(args: argparse.Namespace) -> str:
    if str(args.initial_normalization) == "init_amp":
        return r"$\langle\phi\rangle_z/|\phi_0|_{\max}$"
    return r"$\phi_\mathrm{zonal}/|\phi_\mathrm{zonal}(0)|$"


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
                rf"$\omega_{{GAM}}R_0/v_{{ti}}$ = {_format_metric(case['omega_R0_over_vi'])}" + "\n"
                rf"$\gamma_{{GAM}}R_0/v_{{ti}}$ = {_format_metric(case['gamma_R0_over_vi'])}"
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
    nl = int(args.Nl) if args.Nl is not None else int(run_cfg.get("Nl", 8))
    nm = int(args.Nm) if args.Nm is not None else int(run_cfg.get("Nm", 32))
    dt = float(args.dt) if args.dt is not None else float(run_cfg.get("dt", cfg.time.dt))
    steps = (
        int(args.steps)
        if args.steps is not None
        else int(run_cfg.get("steps", max(int(round(float(cfg.time.t_max) / dt)), 1)))
    )
    sample_stride = (
        int(args.sample_stride)
        if args.sample_stride is not None
        else int(run_cfg.get("sample_stride", cfg.time.sample_stride))
    )
    if dt <= 0.0:
        raise ValueError("--dt must be positive")
    if steps <= 0:
        raise ValueError("--steps must be positive")
    if sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if float(args.time_scale) <= 0.0:
        raise ValueError("--time-scale must be positive")
    if args.checkpoint_steps is not None and int(args.checkpoint_steps) <= 0:
        raise ValueError("--checkpoint-steps must be positive when provided")
    if bool(args.reuse_output) and bool(args.resume_output):
        raise ValueError("--reuse-output and --resume-output are mutually exclusive")
    if nl <= 0:
        raise ValueError("--Nl must be positive")
    if nm <= 0:
        raise ValueError("--Nm must be positive")
    diagnostics = bool(run_cfg.get("diagnostics", cfg.time.diagnostics))
    r0 = float(getattr(cfg.geometry, "R0", 1.0))
    initial_override = _initial_level_override(args, cfg)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for kx_target in [float(val) for val in args.kx_values]:
        token = _kx_token(kx_target)
        out_bundle = args.out_dir / f"w7x_test4_kx{token}.out.nc"
        if bool(args.resume_output) or not args.reuse_output or not out_bundle.exists():
            cfg_case = replace(
                cfg,
                grid=replace(
                    cfg.grid,
                    Lx=float(2.0 * np.pi / kx_target),
                    boundary="periodic",
                    jtwist=None,
                    non_twist=True,
                ),
                time=replace(
                    cfg.time,
                    nstep_restart=None if args.checkpoint_steps is None else int(args.checkpoint_steps),
                ),
                output=replace(
                    cfg.output,
                    path=str(out_bundle),
                    restart_if_exists=bool(args.resume_output),
                    append_on_restart=True,
                    save_for_restart=True,
                ),
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
                show_progress=bool(args.show_progress),
            )

        kx_index, kx_selected = _nearest_kx_index(out_bundle, kx_target)
        kx_tol = max(5.0e-4, 2.0e-2 * abs(float(kx_target)))
        if abs(float(kx_selected) - float(kx_target)) > kx_tol:
            raise ValueError(
                f"selected kx={kx_selected:.6g} differs from target {kx_target:.6g}; "
                "check the radial box and boundary settings for this zonal run"
            )
        series = load_diagnostic_time_series(
            out_bundle,
            variable="Phi_zonal_line_kxt",
            kx_index=kx_index,
            component="real",
            align_phase=True,
        )
        values = np.asarray(series.values, dtype=float)
        t_scaled = np.asarray(series.t, dtype=float) * float(args.time_scale)
        metrics = zonal_flow_response_metrics(
            t_scaled,
            values,
            tail_fraction=float(args.tail_fraction),
            initial_fraction=float(args.initial_fraction),
            initial_policy=str(args.initial_policy),
            initial_level_override=initial_override,
            peak_fit_max_peaks=int(args.peak_fit_max_peaks),
            damping_fit_mode="branchwise_extrema",
            frequency_fit_mode="hilbert_phase",
            fit_window_tmax=float(args.fit_window_tmax),
            hilbert_trim_fraction=0.2,
        )
        gam_frequency = _finite_or_none(metrics.gam_frequency)
        gam_damping_rate = _finite_or_none(metrics.gam_damping_rate)
        omega_r0_over_vi = None if gam_frequency is None else float(gam_frequency) * r0
        gamma_r0_over_vi = None if gam_damping_rate is None else -float(gam_damping_rate) * r0
        row = {
            "kx_target": float(kx_target),
            "kx_selected": float(kx_selected),
            "kx_index": int(kx_index),
            "source_path": _artifact_path(out_bundle),
            "initial_level": float(metrics.initial_level),
            "initial_normalization": str(args.initial_normalization),
            "initial_level_override": None if initial_override is None else float(initial_override),
            "residual_level": float(metrics.residual_level),
            "residual_std": float(metrics.residual_std),
            "response_rms": float(metrics.response_rms),
            "gam_frequency": gam_frequency,
            "gam_damping_rate": gam_damping_rate,
            "omega_R0_over_vi": omega_r0_over_vi,
            "gamma_R0_over_vi": gamma_r0_over_vi,
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
                "t": t_scaled,
                "response": values,
                "metrics": metrics,
            }
        )
        trace_csv = args.out_dir / f"w7x_test4_kx{token}.csv"
        np.savetxt(
            trace_csv,
            np.column_stack([t_scaled, values]),
            delimiter=",",
            header="t_reference,phi_zonal_real",
            comments="",
        )

    fig = _plot_panel(
        cases,
        title="W7-X bean-tube zonal-flow relaxation (test 4)",
        y_label=_normalization_label(args),
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
                "config": _artifact_path(args.config),
                "initial_policy": str(args.initial_policy),
                "initial_normalization": str(args.initial_normalization),
                "initial_level_override": None if initial_override is None else float(initial_override),
                "damping_method": "branchwise_extrema",
                "frequency_method": "hilbert_phase",
                "fit_window_tmax": float(args.fit_window_tmax),
                "runtime": {
                    "dt": float(dt),
                    "steps": int(steps),
                    "sample_stride": int(sample_stride),
                    "checkpoint_steps": None if args.checkpoint_steps is None else int(args.checkpoint_steps),
                    "resume_output": bool(args.resume_output),
                    "time_scale": float(args.time_scale),
                    "diagnostics": bool(diagnostics),
                    "show_progress": bool(args.show_progress),
                    "expected_tmax": float(dt) * float(steps),
                    "Nl": int(nl),
                    "Nm": int(nm),
                },
                "literature_reference": dict(W7X_TEST4_REFERENCE),
                "cases": summary_rows,
                "validation_status": "open",
                "notes": (
                    "This panel follows the W7-X stella/GENE benchmark test-4 contract: "
                    "bean flux tube, torflux=0.64, alpha=0, adiabatic electrons, zero gradients, ky=0, "
                    "and Gaussian electrostatic-potential initial perturbations at four kx rho_i values. "
                    "It uses the unweighted line-averaged signed potential observable requested by the paper; "
                    "the volume-weighted Phi_zonal_mode_kxt diagnostic remains available for shaped-tokamak "
                    "and energy-consistency checks. "
                    "The paper text normalizes the line-averaged response to its t=0 value; "
                    "therefore the default --initial-normalization=line_first follows that convention. "
                    "The init_amp option is retained for explicit audits of the caption wording and "
                    "the clipped initial portion of Fig. 11, but it is not the default validation contract. "
                    "The initial GAM is extracted with separate positive/negative-extrema "
                    "damping fits plus a Hilbert-phase frequency estimate over a common early-time window. "
                    "Runtime times are multiplied by the metadata time_scale before plotting and reference "
                    "comparison; the default time_scale=1 keeps the runtime axis on the paper's t v_ti/a "
                    "axis and non-unit values are treated as calibration probes. "
                    "The default fit window cap isolates the initial GAM before the slower stellarator-specific "
                    "oscillation described in section 4.4 of the benchmark paper; this cutoff is a manuscript-policy "
                    "inference, not a quoted number from the paper itself. The metadata remains open until the "
                    "separate digitized-reference gate closes both residual and late-envelope tolerances."
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
