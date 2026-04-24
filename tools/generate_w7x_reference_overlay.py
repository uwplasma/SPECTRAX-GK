#!/usr/bin/env python3
"""Generate a W7-X raw eigenfunction overlay against a frozen GX reference."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from netCDF4 import Dataset

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from compare_gx_imported_linear import (  # noqa: E402
    _build_imported_linear_terms,
    _build_sample_steps,
    _gx_has_uniform_linear_dt,
    _infer_gx_linear_dt,
    _load_gx_input_contract,
    _load_gx_reference,
    _match_local_kx_index,
    _resolve_imported_boundary,
    _resolve_imported_real_fft_ny,
    _run_single_ky,
    _select_gx_kx_index,
)
from spectraxgk.benchmarking import (  # noqa: E402
    compare_eigenfunctions,
    eigenfunction_gate_report,
    gate_report_to_dict,
    load_eigenfunction_reference_bundle,
    save_eigenfunction_reference_bundle,
)
from spectraxgk.benchmarks import _apply_gx_hypercollisions  # noqa: E402
from spectraxgk.config import GridConfig, resolve_cfl_fac  # noqa: E402
from spectraxgk.geometry import apply_gx_geometry_grid_defaults, load_gx_geometry_netcdf  # noqa: E402
from spectraxgk.grids import build_spectral_grid  # noqa: E402
from spectraxgk.gx_integrators import GXTimeConfig  # noqa: E402
from spectraxgk.plotting import eigenfunction_reference_overlay_figure  # noqa: E402
from spectraxgk.species import build_linear_params  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
W7X_EIGENFUNCTION_GATE_TOLERANCES = {
    "min_overlap": 0.95,
    "max_relative_l2": 0.25,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gx", type=Path, required=True, help="GX .out.nc file for the matched W7-X linear run.")
    parser.add_argument("--gx-input", type=Path, required=True, help="GX input deck for the matched W7-X linear run.")
    parser.add_argument("--gx-big", type=Path, default=None, help="Optional GX .big.nc file used to refresh the frozen reference bundle.")
    parser.add_argument("--geometry-file", type=Path, required=True, help="Imported GX/VMEC geometry file for the SPECTRAX-GK run.")
    parser.add_argument("--ky", type=float, default=0.3, help="Target ky value.")
    parser.add_argument("--Nl", type=int, default=None, help="Laguerre moments. Defaults to the GX input contract.")
    parser.add_argument("--Nm", type=int, default=None, help="Hermite moments. Defaults to the GX input contract.")
    parser.add_argument("--mode-method", choices=("z_index", "max", "project", "svd"), default="z_index")
    parser.add_argument("--sample-step-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-window", choices=("head", "tail"), default="head")
    parser.add_argument(
        "--bundle-out",
        type=Path,
        default=ROOT / "docs" / "_static" / "reference_modes" / "w7x_linear_gx_ky0p3000.npz",
        help="Frozen GX reference bundle path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "reference_modes" / "w7x_linear_spectrax_ky0p3000.csv",
        help="Output CSV path for the SPECTRAX-GK eigenfunction.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "w7x_eigenfunction_reference_overlay_ky0p3000.png",
        help="Output overlay figure path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "docs" / "_static" / "reference_modes" / "w7x_eigenfunction_reference_overlay_ky0p3000.json",
        help="Output JSON path for overlay metrics.",
    )
    return parser.parse_args(argv)


def _normalize_mode(theta: np.ndarray, mode: np.ndarray) -> np.ndarray:
    finite = np.isfinite(mode)
    if not np.any(finite):
        return np.zeros_like(mode)
    idx0 = int(np.argmin(np.abs(theta)))
    ref = mode[idx0]
    if not np.isfinite(ref) or abs(ref) < 1.0e-14:
        idx = int(np.nanargmax(np.abs(np.where(finite, mode, 0.0))))
        ref = mode[idx]
    if not np.isfinite(ref) or abs(ref) < 1.0e-14:
        scale = float(np.nanmax(np.abs(np.where(finite, mode, 0.0))))
        return mode if scale <= 0.0 else mode / scale
    return mode / ref


def _artifact_path(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _freeze_gx_big_reference(gx_big: Path, *, ky: float, out: Path) -> Path:
    with Dataset(gx_big, "r") as root:
        theta = np.asarray(root.groups["Grids"].variables["theta"][:], dtype=float)
        ky_grid = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)
        phi = root.groups["Diagnostics"].variables["Phi"]
        ky_idx = int(np.argmin(np.abs(ky_grid - float(ky))))
        raw = np.asarray(phi[-1, ky_idx, 0, :, :], dtype=float)
    mode = _normalize_mode(theta, raw[:, 0] + 1j * raw[:, 1])
    if not np.all(np.isfinite(mode)):
        raise ValueError(f"{gx_big} produced a non-finite W7-X reference eigenfunction")
    return save_eigenfunction_reference_bundle(
        out,
        theta=theta,
        mode=mode,
        source="GX",
        case="w7x_linear",
        metadata={"ky": float(ky_grid[ky_idx]), "gx_big": str(gx_big)},
    )


def _load_finite_reference(bundle_path: Path):
    bundle = load_eigenfunction_reference_bundle(bundle_path)
    if not np.all(np.isfinite(bundle.mode)):
        raise ValueError(f"{bundle_path} contains a non-finite reference mode; rerun with --gx-big")
    return bundle


def _run_w7x_spectrax_mode(args: argparse.Namespace, *, reference_times: np.ndarray, output_steps: np.ndarray):
    gx_contract = _load_gx_input_contract(args.gx_input)
    gx_time, gx_ky, gx_kx, _gx_omega, _gx_wg, _gx_wphi, _gx_wapar, _gx_phi2 = _load_gx_reference(args.gx)
    geom = load_gx_geometry_netcdf(args.geometry_file.expanduser().resolve())

    ny = _resolve_imported_real_fft_ny(gx_ky, gx_contract)
    boundary_eff = _resolve_imported_boundary(gx_contract.boundary, zero_shat=bool(gx_contract.zero_shat))
    y0 = float(gx_contract.y0)
    lx = 2.0 * np.pi * y0 if boundary_eff == "periodic" else 62.8
    grid_cfg = GridConfig(
        Nx=max(1, int(gx_contract.Nx)),
        Ny=ny,
        Nz=int(np.asarray(geom.theta).size),
        Lx=lx,
        Ly=2.0 * np.pi * y0,
        boundary=boundary_eff,
        y0=y0,
        nperiod=max(1, int(gx_contract.nperiod)),
        ntheta=int(gx_contract.ntheta),
    )
    grid_full = build_spectral_grid(apply_gx_geometry_grid_defaults(geom, grid_cfg))
    nl_use = int(args.Nl) if args.Nl is not None else int(gx_contract.nlaguerre)
    nm_use = int(args.Nm) if args.Nm is not None else int(gx_contract.nhermite)

    params = build_linear_params(
        list(gx_contract.species),
        tau_e=float(gx_contract.tau_e),
        kpar_scale=float(geom.gradpar()),
        beta=float(gx_contract.beta),
        fapar=float(gx_contract.fapar),
    )
    if gx_contract.hypercollisions:
        params = _apply_gx_hypercollisions(params, nhermite=nm_use)
    params = replace(
        params,
        D_hyper=float(gx_contract.D_hyper),
        damp_ends_amp=float(gx_contract.damp_ends_amp),
        damp_ends_widthfrac=float(gx_contract.damp_ends_widthfrac),
    )
    time_cfg = GXTimeConfig(
        dt=_infer_gx_linear_dt(gx_time, gx_contract),
        t_max=float(gx_time[-1]),
        method=gx_contract.scheme,
        sample_stride=gx_contract.nwrite,
        fixed_dt=bool(gx_contract.dt is not None or _gx_has_uniform_linear_dt(gx_time, gx_contract)),
        cfl_fac=resolve_cfl_fac(gx_contract.scheme, None),
    )
    gx_kx_idx = _select_gx_kx_index(gx_kx, gx_contract)
    kx_idx = _match_local_kx_index(np.asarray(grid_full.kx), float(gx_kx[gx_kx_idx]))
    result = _run_single_ky(
        ky_target=float(args.ky),
        geom=geom,
        grid_full=grid_full,
        params=params,
        time_cfg=time_cfg,
        gx_contract=gx_contract,
        species=tuple(gx_contract.species),
        Nl=nl_use,
        Nm=nm_use,
        reference_times=reference_times,
        output_steps=output_steps,
        mode_method=str(args.mode_method),
        kx_index=kx_idx,
        terms=_build_imported_linear_terms(gx_contract),
        return_phi_samples=True,
    )
    gamma, omega, wg, wphi, wapar, phi2, phi_sample_times, phi_samples = result
    theta = np.asarray(grid_full.z, dtype=float)
    mode = np.asarray(phi_samples[-1, 0, 0, :], dtype=np.complex128)
    return {
        "mode": _normalize_mode(theta, mode),
        "theta": theta,
        "gamma_last": float(np.asarray(gamma)[-1]),
        "omega_last": float(np.asarray(omega)[-1]),
        "Wg_last": float(np.asarray(wg)[-1]),
        "Wphi_last": float(np.asarray(wphi)[-1]),
        "Wapar_last": float(np.asarray(wapar)[-1]),
        "Phi2_last": float(np.asarray(phi2)[-1]),
        "t_final": float(np.asarray(phi_sample_times)[-1]),
        "nl": nl_use,
        "nm": nm_use,
        "ny": int(ny),
        "kx_local": float(np.asarray(grid_full.kx)[kx_idx]),
        "kx_ref": float(gx_kx[gx_kx_idx]),
    }


def _w7x_eigenfunction_gate_report(metrics):
    return eigenfunction_gate_report(
        metrics,
        case="w7x_linear_eigenfunction_ky0p3000",
        source="GX raw eigenfunction bundle",
        min_overlap=float(W7X_EIGENFUNCTION_GATE_TOLERANCES["min_overlap"]),
        max_relative_l2=float(W7X_EIGENFUNCTION_GATE_TOLERANCES["max_relative_l2"]),
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.gx_big is not None:
        _freeze_gx_big_reference(args.gx_big, ky=float(args.ky), out=args.bundle_out)
    bundle = _load_finite_reference(args.bundle_out)
    gx_time, _gx_ky, _gx_kx, _gx_omega, _gx_wg, _gx_wphi, _gx_wapar, _gx_phi2 = _load_gx_reference(args.gx)
    output_steps = _build_sample_steps(
        gx_time,
        sample_step_stride=int(args.sample_step_stride),
        max_samples=args.max_samples,
        sample_window=str(args.sample_window),
    )
    reference_times = np.asarray(gx_time[: int(output_steps[-1]) + 1], dtype=float)
    run = _run_w7x_spectrax_mode(args, reference_times=reference_times, output_steps=output_steps)

    if np.asarray(run["theta"]).shape != np.asarray(bundle.theta).shape or not np.allclose(run["theta"], bundle.theta):
        mode = np.interp(bundle.theta, run["theta"], run["mode"].real) + 1j * np.interp(
            bundle.theta, run["theta"], run["mode"].imag
        )
    else:
        mode = np.asarray(run["mode"], dtype=np.complex128)
    metrics = compare_eigenfunctions(mode, bundle.mode)
    validation_gate_report = _w7x_eigenfunction_gate_report(metrics)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "z": bundle.theta,
            "eigen_real": mode.real,
            "eigen_imag": mode.imag,
            "eigen_abs": np.abs(mode),
        }
    ).to_csv(args.out_csv, index=False)

    fig, _axes = eigenfunction_reference_overlay_figure(
        bundle.theta,
        mode,
        bundle.theta,
        bundle.mode,
        title=f"W7-X raw eigenfunction overlay (ky={float(args.ky):.3f})",
    )
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "ky": float(args.ky),
                "kx_ref": float(run["kx_ref"]),
                "kx_local": float(run["kx_local"]),
                "ny": int(run["ny"]),
                "Nl": int(run["nl"]),
                "Nm": int(run["nm"]),
                "mode_method": str(args.mode_method),
                "gamma_last": float(run["gamma_last"]),
                "omega_last": float(run["omega_last"]),
                "Wg_last": float(run["Wg_last"]),
                "Wphi_last": float(run["Wphi_last"]),
                "Wapar_last": float(run["Wapar_last"]),
                "Phi2_last": float(run["Phi2_last"]),
                "t_final": float(run["t_final"]),
                "overlap": float(metrics.overlap),
                "relative_l2": float(metrics.relative_l2),
                "phase_shift": float(metrics.phase_shift),
                "gate_tolerances": dict(W7X_EIGENFUNCTION_GATE_TOLERANCES),
                "gate_report": gate_report_to_dict(validation_gate_report),
                "eigenfunction_gate_passed": bool(validation_gate_report.passed),
                "validation_status": "closed" if validation_gate_report.passed else "open",
                "reference_bundle": _artifact_path(args.bundle_out),
                "spectrax_csv": _artifact_path(args.out_csv),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
