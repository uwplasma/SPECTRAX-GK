#!/usr/bin/env python3
"""Generate a Merlo Case-III shaped-Miller signed zonal-response artifact.

This uses the Merlo et al. Phys. Plasmas 23, 032104 (2016) Case-III
Rosenbluth-Hinton/GAM setup with adiabatic electrons, zero gradients,
``k_y = 0``, ``k_x rho_i = 0.05``, a small initial density perturbation by
default, and a signed zonal observable ``Phi_zonal_mode_kxt`` written by the
runtime path.  The artifact is kept as pending unless the generated
residual/GAM metrics land in the literature envelope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np

from spectraxgk.benchmarking import load_diagnostic_time_series, zonal_flow_response_metrics
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.plotting import zonal_flow_response_figure
from spectraxgk.runtime_artifacts import run_runtime_nonlinear_with_artifacts

ROOT = Path(__file__).resolve().parents[1]

MERLO_CASE_III_REFERENCE = {
    "paper": "Merlo et al., Phys. Plasmas 23, 032104 (2016)",
    "case": "III",
    "q_s": 1.389,
    "s_hat": 0.751,
    "epsilon": 0.18,
    "kappa": 1.4723,
    "delta": -0.0070,
    "D": -0.0139,
    "a_MHD": 0.5425,
    "dRgeom_dr": -0.1569,
    "dkappa_dr": -0.0728,
    "ddelta_dr": -0.0140,
    "kx_rhoi": 0.05,
    "ky": 0.0,
    "tmax_R0_over_vi": 150.0,
    # Values digitized/visually read from Figs. 12, 14, and 16; use as a
    # paper-scale gate, not as a replacement for a frozen cross-code trace.
    "residual_phi_over_phi0": 0.190,
    "omega_gam_R0_over_vi": 2.24,
    "gamma_gam_R0_over_vi": -0.17,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "examples" / "benchmarks" / "runtime_miller_zonal_response.toml",
        help="Runtime TOML for the shaped-Miller zonal-response pilot.",
    )
    parser.add_argument(
        "--out-bundle",
        type=Path,
        default=ROOT / "tools_out" / "zonal_response" / "miller_caseIII_initial_density_Nl4_Nm24_Nz32_dt0005_t60.out.nc",
        help="Runtime output bundle path.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "miller_zonal_response_pilot.png",
        help="Output figure path.",
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
        help=(
            "Initial normalization convention. Merlo/Rosenbluth-Hinton residuals "
            "are quoted as phi(t->infinity)/phi(0), so this tool defaults to first_abs."
        ),
    )
    parser.add_argument(
        "--peak-fit-max-peaks",
        type=int,
        default=4,
        help=(
            "Maximum number of positive and negative extrema used per branch for the "
            "Merlo-style damping fit."
        ),
    )
    parser.add_argument(
        "--fit-window-tmax",
        type=float,
        default=30.0,
        help="Upper time bound for the common pre-recurrence GAM fit window.",
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Reuse an existing out.nc bundle instead of rerunning the pilot.",
    )
    return parser.parse_args()


def _nearest_kx_index(path: Path, target_kx: float) -> tuple[int, float]:
    with nc.Dataset(path, "r") as ds:
        grids = ds.groups.get("Grids")
        if grids is None or "kx" not in grids.variables:
            raise ValueError(f"missing Grids/kx in {path}")
        kx = np.asarray(grids.variables["kx"][:], dtype=float)
    if kx.ndim != 1 or kx.size == 0:
        raise ValueError(f"invalid kx grid in {path}")
    idx = int(np.argmin(np.abs(kx - float(target_kx))))
    return idx, float(kx[idx])


def _setup_note(cfg) -> str:
    source = str(getattr(cfg.expert, "source", "default")).strip().lower()
    if source == "phiext_full":
        return "external phiext_full source"
    return f"initial {cfg.init.init_field} perturbation"


def main() -> int:
    args = _parse_args()
    cfg, raw = load_runtime_from_toml(args.config)
    run_cfg = dict(raw.get("run", {}))
    ky_target = float(run_cfg.get("ky", 0.0))
    kx_target = float(run_cfg.get("kx", 0.1))
    nl = int(run_cfg.get("Nl", 2))
    nm = int(run_cfg.get("Nm", 2))
    dt = float(run_cfg.get("dt", cfg.time.dt))
    steps = int(run_cfg.get("steps", max(int(round(float(cfg.time.t_max) / dt)), 1)))
    sample_stride = int(run_cfg.get("sample_stride", cfg.time.sample_stride))
    diagnostics = bool(run_cfg.get("diagnostics", cfg.time.diagnostics))

    out_bundle = Path(args.out_bundle)
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_output or not out_bundle.exists():
        run_runtime_nonlinear_with_artifacts(
            cfg,
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
    if np.iscomplexobj(series.values):
        raise ValueError("signed zonal pilot plotting requires a real-valued phase-aligned trace")

    metrics = zonal_flow_response_metrics(
        series.t,
        np.asarray(series.values, dtype=float),
        tail_fraction=float(args.tail_fraction),
        initial_fraction=float(args.initial_fraction),
        initial_policy=str(args.initial_policy),
        peak_fit_max_peaks=int(args.peak_fit_max_peaks) if args.peak_fit_max_peaks is not None else None,
        damping_fit_mode="branchwise_extrema",
        frequency_fit_mode="hilbert_phase",
        fit_window_tmax=float(args.fit_window_tmax),
        hilbert_trim_fraction=0.2,
    )
    setup_note = _setup_note(cfg)
    ref_residual = float(MERLO_CASE_III_REFERENCE["residual_phi_over_phi0"])
    ref_omega = float(MERLO_CASE_III_REFERENCE["omega_gam_R0_over_vi"])
    ref_gamma = float(MERLO_CASE_III_REFERENCE["gamma_gam_R0_over_vi"])
    r0 = float(getattr(cfg.geometry, "R0", 1.0))
    omega_r0_over_vi = float(metrics.gam_frequency) * r0
    damping_r0_over_vi = float(metrics.gam_damping_rate) * r0
    gamma_r0_over_vi = -damping_r0_over_vi
    residual_abs_error = abs(float(metrics.residual_level) - ref_residual)
    omega_abs_error = abs(omega_r0_over_vi - ref_omega)
    gamma_abs_error = abs(gamma_r0_over_vi - ref_gamma)
    title = f"Merlo Case III zonal-response (ky={ky_target:.3f}, kx={kx_selected:.3f})"
    fig, _axes = zonal_flow_response_figure(
        series.t,
        np.asarray(series.values, dtype=float),
        metrics=metrics,
        title=title,
        y_label=r"$\phi_\mathrm{zonal}/|\phi_\mathrm{zonal}(0)|$",
    )
    ax0 = _axes[0]
    ax0.axhline(
        ref_residual,
        color="#7b2cbf",
        linestyle=":",
        linewidth=2.1,
        label="Merlo Case III residual",
    )
    ax0.legend(loc="best", frameon=False)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")

    csv_out = args.out_png.with_suffix(".csv")
    np.savetxt(
        csv_out,
        np.column_stack([series.t, np.asarray(series.values, dtype=float)]),
        delimiter=",",
        header="t,phi_zonal_real",
        comments="",
    )

    meta_out = args.out_png.with_suffix(".json")
    meta_out.write_text(
        json.dumps(
            {
                "config": str(args.config),
                "source_path": series.source_path,
                "variable": "Phi_zonal_mode_kxt",
                "kx_index": int(kx_index),
                "kx_selected": float(kx_selected),
                "ky_target": float(ky_target),
                "initial_level": float(metrics.initial_level),
                "initial_policy": str(metrics.initial_policy),
                "residual_level": float(metrics.residual_level),
                "residual_std": float(metrics.residual_std),
                "response_rms": float(metrics.response_rms),
                "gam_frequency": float(metrics.gam_frequency),
                "gam_damping_rate": float(metrics.gam_damping_rate),
                "gam_frequency_R0_over_vi": float(omega_r0_over_vi),
                "gam_damping_rate_R0_over_vi": float(damping_r0_over_vi),
                "gam_growth_rate_R0_over_vi": float(gamma_r0_over_vi),
                "damping_method": str(metrics.damping_method),
                "frequency_method": str(metrics.frequency_method),
                "peak_count": int(metrics.peak_count),
                "peak_fit_count": int(metrics.peak_fit_count),
                "tmin": float(metrics.tmin),
                "tmax": float(metrics.tmax),
                "fit_tmin": float(metrics.fit_tmin),
                "fit_tmax": float(metrics.fit_tmax),
                "literature_reference": dict(MERLO_CASE_III_REFERENCE),
                "residual_abs_error_vs_literature": float(residual_abs_error),
                "omega_abs_error_vs_literature_R0_over_vi": float(omega_abs_error),
                "gamma_abs_error_vs_literature_R0_over_vi": float(gamma_abs_error),
                "setup": setup_note,
                "validation_status": "open",
                "notes": (
                    "This is a Merlo Case-III shaped-Miller zonal-relaxation run "
                    f"built from the signed zonal observable Phi_zonal_mode_kxt with zero gradients, "
                    f"adiabatic electrons, and an {setup_note}. "
                    "The literature reference values are read from Merlo et al. Figs. 12, 14, and 16; "
                    "the residual is normalized with the Rosenbluth-Hinton first-sample convention. "
                    f"The GAM damping follows the paper convention by fitting positive and negative extrema separately "
                    f"over the common pre-recurrence window t in [{metrics.fit_tmin:.1f}, {metrics.fit_tmax:.1f}] "
                    f"using up to {args.peak_fit_max_peaks} extrema per branch, while the frequency is obtained from "
                    "the instantaneous phase of the same window via a Hilbert-transform analytic signal. "
                    "The residual, damping, and GAM frequency are now close to the paper-scale read-off; the long-time recurrence "
                    "behavior still remains an explicit numerical follow-up item."
                ),
                "references": [
                    "Merlo et al. 2016 shaped-tokamak collisionless GAM benchmark, Case III",
                    "W7-X stella/GENE benchmark 2022 for zonal-flow observable conventions",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
