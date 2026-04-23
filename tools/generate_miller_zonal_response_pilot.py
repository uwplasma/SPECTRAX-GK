#!/usr/bin/env python3
"""Generate a shaped-Miller signed zonal-response artifact.

This uses a Merlo-style zonal relaxation setup with adiabatic electrons, zero
gradients, ``k_y = 0``, GX-style ``source="phiext_full"`` plus ``phi_ext``, and
a signed zonal observable ``Phi_zonal_mode_kxt`` written by the runtime path.
The artifact is still kept as pending until cross-code and/or analytic closure
is frozen, but the runtime contract itself is no longer ITG-like.
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
        default=ROOT / "tools_out" / "zonal_response" / "miller_phiext_merlo_5000.out.nc",
        help="GX-style runtime output bundle path.",
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
    )
    title = f"Shaped Miller zonal-response (ky={ky_target:.3f}, kx={kx_selected:.3f})"
    fig, _axes = zonal_flow_response_figure(
        series.t,
        np.asarray(series.values, dtype=float),
        metrics=metrics,
        title=title,
        y_label="phase-aligned zonal potential",
    )

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
                "residual_level": float(metrics.residual_level),
                "residual_std": float(metrics.residual_std),
                "response_rms": float(metrics.response_rms),
                "gam_frequency": float(metrics.gam_frequency),
                "gam_damping_rate": float(metrics.gam_damping_rate),
                "peak_count": int(metrics.peak_count),
                "tmin": float(metrics.tmin),
                "tmax": float(metrics.tmax),
                "notes": (
                    "This is a Merlo-style shaped-Miller zonal-relaxation run "
                    "built from the signed zonal observable Phi_zonal_mode_kxt with zero gradients, "
                    "adiabatic electrons, and a GX-style phiext_full source contract. "
                    "It remains pending until cross-code and/or analytic closure is frozen."
                ),
                "references": [
                    "Merlo et al. 2016 shaped-tokamak collisionless GAM benchmark",
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
