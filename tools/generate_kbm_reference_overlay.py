#!/usr/bin/env python3
"""Generate a raw KBM eigenfunction overlay against a frozen GX reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from compare_gx_kbm import (
    _build_cfg,
    _load_gx_eigenfunction,
    _load_kbm_gx_input_contract,
    _normalize_mode,
    _prepare_gx_reference,
)
from spectraxgk.analysis import extract_eigenfunction
from spectraxgk.benchmarking import (
    compare_eigenfunctions,
    eigenfunction_gate_report,
    gate_report_to_dict,
    infer_triple_dealiased_ny,
    late_time_window,
    save_eigenfunction_reference_bundle,
)
from spectraxgk.benchmarks import run_kbm_linear
from spectraxgk.plotting import eigenfunction_reference_overlay_figure

ROOT = Path(__file__).resolve().parents[1]

KBM_EIGENFUNCTION_GATE_TOLERANCES = {
    "min_overlap": 0.95,
    "max_relative_l2": 0.25,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx", type=Path, required=True, help="Path to GX .out.nc file.")
    parser.add_argument("--gx-big", type=Path, required=True, help="Path to GX .big.nc file.")
    parser.add_argument("--gx-input", type=Path, required=True, help="Path to the GX input deck.")
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "kbm_gx_candidates.csv",
        help="Per-candidate KBM comparison table with the selected branch rows.",
    )
    parser.add_argument("--ky", type=float, default=0.3, help="Target ky value.")
    parser.add_argument("--dt", type=float, default=0.01, help="SPECTRAX time step.")
    parser.add_argument(
        "--fit-padding",
        type=float,
        default=0.5,
        help="Extra time beyond the selected growth-fit window before stopping the run.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=2,
        help="Sample stride for the bounded GX-time extraction path.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.4,
        help="Late-time fraction used for eigenfunction extraction.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "reference_modes" / "kbm_linear_spectrax_ky0p3000.csv",
        help="Output CSV path for the SPECTRAX eigenfunction.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "kbm_eigenfunction_reference_overlay_ky0p3000.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "docs" / "_static" / "reference_modes" / "kbm_eigenfunction_reference_overlay_ky0p3000.json",
        help="Output JSON path for overlay metrics.",
    )
    parser.add_argument(
        "--bundle-out",
        type=Path,
        default=ROOT / "docs" / "_static" / "reference_modes" / "kbm_linear_gx_ky0p3000.npz",
        help="Output path for the frozen GX reference bundle.",
    )
    return parser.parse_args()


def _selected_candidate_row(candidate_csv: Path, ky: float) -> pd.Series:
    table = pd.read_csv(candidate_csv)
    mask = (table["selected"] == True) & (table["ky"].round(4) == round(float(ky), 4))
    if not mask.any():
        raise ValueError(f"no selected KBM candidate found for ky={ky:.4f}")
    return table.loc[mask].iloc[0]


def _steps_for_fit_window(*, fit_tmax: float, dt: float, fit_padding: float, sample_stride: int) -> int:
    steps = max(int(math.ceil((float(fit_tmax) + float(fit_padding)) / float(dt))), 1)
    stride = max(int(sample_stride), 1)
    rem = steps % stride
    if rem != 0:
        steps += stride - rem
    return steps


def _kbm_eigenfunction_gate_report(metrics):
    return eigenfunction_gate_report(
        metrics,
        case="kbm_linear_eigenfunction_ky0p3000",
        source="GX raw eigenfunction bundle",
        min_overlap=float(KBM_EIGENFUNCTION_GATE_TOLERANCES["min_overlap"]),
        max_relative_l2=float(KBM_EIGENFUNCTION_GATE_TOLERANCES["max_relative_l2"]),
    )


def main() -> None:
    args = _parse_args()
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")

    candidate = _selected_candidate_row(args.candidate_csv, args.ky)
    fit_tmin = float(candidate["fit_window_tmin"])
    fit_tmax = float(candidate["fit_window_tmax"])
    steps = _steps_for_fit_window(
        fit_tmax=fit_tmax,
        dt=args.dt,
        fit_padding=args.fit_padding,
        sample_stride=args.sample_stride,
    )

    _gx_time, _gx_ky, _gx_omega, beta, _q, _shat, _eps, _rmaj, nky_full, _y0 = _prepare_gx_reference(
        args.gx,
        ky_arg=str(args.ky),
        y0_fallback=10.0,
    )
    contract = _load_kbm_gx_input_contract(args.gx_input)
    ny = infer_triple_dealiased_ny(int(nky_full))
    cfg = _build_cfg(
        beta=float(contract.beta),
        q=float(contract.q),
        shat=float(contract.shat),
        eps=float(contract.eps),
        rmaj=float(contract.rmaj),
        ny=ny,
        ntheta=int(contract.ntheta),
        nperiod=int(contract.nperiod),
        y0=float(contract.y0),
        mass_ratio=float(contract.mass_ratio),
        ion_tprim=float(contract.ion_tprim),
        ele_tprim=float(contract.ele_tprim),
        ion_fprim=float(contract.ion_fprim),
        ele_fprim=float(contract.ele_fprim),
        te_over_ti=float(contract.te_over_ti),
        init_field=str(contract.init_field),
        gaussian_init=bool(contract.gaussian_init),
        init_electrons_only=bool(contract.init_electrons_only),
    )

    theta_gx, mode_gx = _load_gx_eigenfunction(args.gx_big, float(args.ky))
    save_eigenfunction_reference_bundle(
        args.bundle_out,
        theta=theta_gx,
        mode=mode_gx,
        source="GX",
        case="kbm_linear",
        metadata={"ky": float(args.ky)},
    )

    result = run_kbm_linear(
        ky_target=float(args.ky),
        beta_value=float(beta),
        cfg=cfg,
        Nl=int(contract.nlaguerre),
        Nm=int(contract.nhermite),
        dt=float(args.dt),
        steps=int(steps),
        method="rk4",
        solver="gx_time",
        fit_signal="phi",
        mode_method="project",
        diagnostic_norm="gx",
        gx_reference=True,
        auto_window=False,
        tmin=fit_tmin,
        tmax=fit_tmax,
        sample_stride=int(args.sample_stride),
    )

    t_arr = np.asarray(result.t, dtype=float)
    eig_tmin, eig_tmax = late_time_window(t_arr, tail_fraction=float(args.tail_fraction))
    mode_sp = extract_eigenfunction(
        np.asarray(result.phi_t),
        t_arr,
        result.selection,
        z=theta_gx,
        method="svd",
        tmin=eig_tmin,
        tmax=eig_tmax,
    )
    mode_sp = _normalize_mode(theta_gx, np.asarray(mode_sp, dtype=np.complex128))
    metrics = compare_eigenfunctions(mode_sp, mode_gx)
    validation_gate_report = _kbm_eigenfunction_gate_report(metrics)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "z": theta_gx,
            "eigen_real": mode_sp.real,
            "eigen_imag": mode_sp.imag,
            "eigen_abs": np.abs(mode_sp),
        }
    ).to_csv(args.out_csv, index=False)

    fig, _axes = eigenfunction_reference_overlay_figure(
        theta_gx,
        mode_sp,
        theta_gx,
        mode_gx,
        title=f"KBM raw eigenfunction overlay (ky={float(args.ky):.3f})",
    )
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "ky": float(args.ky),
                "ny": int(ny),
                "nky_full": int(nky_full),
                "dt": float(args.dt),
                "steps": int(steps),
                "sample_stride": int(args.sample_stride),
                "fit_tmin": fit_tmin,
                "fit_tmax": fit_tmax,
                "eig_tmin": eig_tmin,
                "eig_tmax": eig_tmax,
                "gamma": float(result.gamma),
                "omega": float(result.omega),
                "overlap": float(metrics.overlap),
                "relative_l2": float(metrics.relative_l2),
                "phase_shift": float(metrics.phase_shift),
                "gate_tolerances": dict(KBM_EIGENFUNCTION_GATE_TOLERANCES),
                "gate_report": gate_report_to_dict(validation_gate_report),
                "eigenfunction_gate_passed": bool(validation_gate_report.passed),
                "validation_status": "closed" if validation_gate_report.passed else "open",
                "t_final": float(t_arr[-1]),
                "nsamples": int(t_arr.size),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
